import numpy as np
import pandas as pd
import torch

from forex_signal.backtest.walk_forward import BacktestConfig, run_backtest
from forex_signal.data.features import FEATURE_COLUMNS
from forex_signal.model.predict import Predictor


def _synthetic_trending(n: int = 800, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = 0.00005
    rets = rng.normal(drift, 0.001, n)
    close = 1.10 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.0008, n))
    low = close * (1 - rng.uniform(0, 0.0008, n))
    open_ = close * (1 + rng.normal(0, 0.0003, n))
    vol = rng.uniform(100, 1000, n)
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": open_, "high": high, "low": low,
        "close": close, "tick_volume": vol,
    })


class _ConstPredictor(Predictor):
    """Predictor that always returns a fixed prediction — for deterministic backtest tests."""

    def __init__(self, pred: np.ndarray):
        from forex_signal.model.lnn import ForexLNN
        n_features = len(FEATURE_COLUMNS)
        super().__init__(
            model=ForexLNN(n_features, units=8, pred_horizon=len(pred), dropout=0.0).eval(),
            feature_columns=FEATURE_COLUMNS,
            feature_means=np.zeros(n_features, dtype=np.float32),
            feature_stds=np.ones(n_features, dtype=np.float32),
            pred_horizon=len(pred),
            seq_len=30,
            device=torch.device("cpu"),
        )
        self._pred = torch.tensor(pred, dtype=torch.float32)

        # Monkey-patch model forward
        const_pred = self._pred
        class _Const(torch.nn.Module):
            def __init__(self): super().__init__(); self.pred_horizon = const_pred.numel()
            def eval(self): return self
            def __call__(self, x): return const_pred.unsqueeze(0).repeat(x.shape[0], 1)
            def to(self, *_a, **_k): return self
        self.model = _Const()


def test_backtest_runs_and_produces_metrics():
    df = _synthetic_trending(800)
    predictor = _ConstPredictor(np.array([0.0008, 0.0006, 0.0004, 0.0002, 0.0001], dtype=np.float32))
    cfg = BacktestConfig(seq_len=30, initial_balance=10_000.0, lot_size=0.01)
    result = run_backtest(df, predictor, cfg)
    assert result.n_trades > 0
    assert len(result.equity_curve) > 0
    # With a constantly bullish forecast on a trending series we should at least try trades
    assert all(t.direction == 1 for t in result.trades)


def test_backtest_no_trades_when_signal_below_threshold():
    df = _synthetic_trending(500)
    # Predicted cum return ~ 5e-6 -> 0.05 bps, far below default 1.5 bps threshold
    predictor = _ConstPredictor(np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float32))
    cfg = BacktestConfig(seq_len=30, initial_balance=10_000.0)
    result = run_backtest(df, predictor, cfg)
    assert result.n_trades == 0
