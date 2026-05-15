"""Walk-forward backtest with SMC entry permission engine + LNN confirmation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forex_signal.data.features import compute_features
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import EntryConfig, compute_all_signals, evaluate_entry
from forex_signal.strategy.risk import compute_trade_plan

log = logging.getLogger(__name__)

PIP_EURUSD = 0.0001


@dataclass
class BacktestConfig:
    seq_len: int = 60
    initial_balance: float = 10_000.0
    lot_size: float = 0.01
    leverage: int = 400
    spread_pips: float = 0.8
    slippage_pips: float = 0.3
    commission_per_lot: float = 7.0
    sl_atr_multiplier: float = 1.5
    tp_atr_min_multiplier: float = 0.6
    tp_atr_max_multiplier: float = 1.5    # tighter cap for high-win-rate scalping
    max_concurrent_positions: int = 2
    cooldown_bars: int = 3
    pip_value: float = PIP_EURUSD
    contract_size: float = 100_000.0
    entry: EntryConfig = field(default_factory=EntryConfig)
    # Spread-aware TP/SL guards
    spread_min_tp_ratio: float = 2.5   # TP distance must be >= this × spread (else widen TP)
    spread_min_sl_ratio: float = 4.0   # SL distance must be >= this × spread (else widen SL)
    spread_skip_ratio: float = 0.45    # if spread > this × TP_distance, skip the trade entirely


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    direction: int
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    pnl: float
    reason: str  # "tp" | "sl" | "reverse" | "eod"
    lnn_prob: float = 0.0


@dataclass
class BacktestResult:
    n_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    sharpe: float
    max_drawdown_pct: float
    final_equity: float
    trades_per_day: float
    days_covered: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


def run_backtest(
    ohlcv: pd.DataFrame,
    predictor: Predictor,
    config: BacktestConfig,
) -> BacktestResult:
    features = compute_features(ohlcv)
    if len(features) <= config.seq_len + predictor.pred_horizon + 1:
        raise ValueError("not enough bars after warmup for backtest")

    # Compute SMC layers once on the post-feature DataFrame
    smc = compute_all_signals(features, config.entry)

    f_arr = features[predictor.feature_columns].to_numpy(dtype=np.float32)
    f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds
    close = features["close"].to_numpy(dtype=np.float64)
    high = features["high"].to_numpy(dtype=np.float64)
    low = features["low"].to_numpy(dtype=np.float64)
    atr = features["atr_14"].to_numpy(dtype=np.float64)
    times = pd.to_datetime(features["time"]).astype(str).to_numpy()

    n = len(features)
    equity = config.initial_balance
    open_positions: list[dict] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = []
    cooldown_until = -1

    pip = config.pip_value
    spread_price = config.spread_pips * pip
    slippage_price = config.slippage_pips * pip
    pnl_per_price = config.lot_size * config.contract_size
    commission = config.commission_per_lot * config.lot_size

    device = predictor.device
    model = predictor.model

    for i in range(config.seq_len, n - 1):
        bar_high = high[i]
        bar_low = low[i]

        # Manage open positions: SL/TP check on this bar
        still_open = []
        for pos in open_positions:
            hit_tp = bar_high >= pos["tp"] if pos["dir"] == 1 else bar_low <= pos["tp"]
            hit_sl = bar_low <= pos["sl"] if pos["dir"] == 1 else bar_high >= pos["sl"]
            exit_price = None
            reason = None
            if hit_tp and hit_sl:
                # SL hit first (conservative)
                exit_price = pos["sl"]
                reason = "sl"
            elif hit_tp:
                exit_price = pos["tp"]
                reason = "tp"
            elif hit_sl:
                exit_price = pos["sl"]
                reason = "sl"
            if exit_price is not None:
                # Entry price already includes spread+slippage adjustment, so gross
                # naturally accounts for spread on entry side. Just subtract commission.
                gross = (exit_price - pos["entry"]) * pos["dir"] * pnl_per_price
                pnl = gross - 2 * commission
                equity += pnl
                closed_trades.append(
                    Trade(
                        entry_time=str(pos["entry_time"]),
                        exit_time=str(times[i]),
                        direction=pos["dir"],
                        entry_price=pos["entry"],
                        exit_price=exit_price,
                        sl_price=pos["sl"],
                        tp_price=pos["tp"],
                        pnl=pnl,
                        reason=reason,
                        lnn_prob=pos.get("lnn_prob", 0.0),
                    )
                )
                cooldown_until = i + config.cooldown_bars
            else:
                still_open.append(pos)
        open_positions = still_open
        equity_curve.append(equity)

        if i <= cooldown_until:
            continue
        if len(open_positions) >= config.max_concurrent_positions:
            continue
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue

        # Run LNN to get direction probability
        window = f_norm[i - config.seq_len + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :]).to(device)
        with torch.no_grad():
            ret_norm, dir_logit = model(x)
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        pred_ret = ret_norm.cpu().numpy()[0] * predictor.target_std + predictor.target_mean
        cum_return = float(pred_ret.sum())

        # Evaluate entry permission
        decision = evaluate_entry(smc.iloc[i], prob, config.entry)
        if decision.direction == 0:
            continue

        # Build trade plan with spread-aware widening
        next_open = close[i]
        entry_price = next_open + (spread_price + slippage_price) * decision.direction
        plan = compute_trade_plan(
            direction=decision.direction,
            price=entry_price,
            atr=atr[i],
            predicted_cum_return=cum_return,
            lot_size=config.lot_size,
            sl_atr_multiplier=config.sl_atr_multiplier,
            tp_atr_min_multiplier=config.tp_atr_min_multiplier,
            tp_atr_max_multiplier=config.tp_atr_max_multiplier,
        )
        # Spread guard: widen TP/SL if too close to spread
        tp_dist = abs(plan.tp_price - entry_price)
        sl_dist = abs(plan.sl_price - entry_price)
        if tp_dist < config.spread_min_tp_ratio * spread_price:
            tp_dist = config.spread_min_tp_ratio * spread_price
            plan.tp_price = entry_price + tp_dist * decision.direction
            plan.tp_distance = tp_dist
        if sl_dist < config.spread_min_sl_ratio * spread_price:
            sl_dist = config.spread_min_sl_ratio * spread_price
            plan.sl_price = entry_price - sl_dist * decision.direction
            plan.sl_distance = sl_dist
        # Skip if spread is still too big relative to TP (very wide spread environment)
        if spread_price > tp_dist * config.spread_skip_ratio:
            continue
        open_positions.append(
            {
                "entry": entry_price,
                "entry_time": times[i],
                "dir": decision.direction,
                "sl": plan.sl_price,
                "tp": plan.tp_price,
                "lnn_prob": prob,
            }
        )

    # Close any open positions at last bar
    last_i = n - 1
    last_close = close[last_i]
    for pos in open_positions:
        gross = (last_close - pos["entry"]) * pos["dir"] * pnl_per_price
        pnl = gross - 2 * commission
        equity += pnl
        closed_trades.append(
            Trade(
                entry_time=str(pos["entry_time"]),
                exit_time=str(times[last_i]),
                direction=pos["dir"],
                entry_price=pos["entry"],
                exit_price=last_close,
                sl_price=pos["sl"],
                tp_price=pos["tp"],
                pnl=pnl,
                reason="eod",
                lnn_prob=pos.get("lnn_prob", 0.0),
            )
        )
    equity_curve.append(equity)

    # Compute days covered for trades-per-day
    t0 = pd.to_datetime(features["time"].iloc[config.seq_len])
    t1 = pd.to_datetime(features["time"].iloc[-1])
    days = max((t1 - t0).total_seconds() / 86400.0, 1.0)

    return _build_result(closed_trades, equity_curve, config.initial_balance, days)


def _build_result(trades: list[Trade], equity_curve: list[float], initial: float, days: float) -> BacktestResult:
    n = len(trades)
    if n == 0:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, initial, 0.0, days, [], equity_curve)
    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins)) / n
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")
    total = float(pnls.sum())

    eq = np.array(equity_curve)
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252 * 24 * 12)) if rets.size > 1 else 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    max_dd_pct = float(dd.min() * 100.0)

    return BacktestResult(
        n_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total,
        sharpe=sharpe,
        max_drawdown_pct=max_dd_pct,
        final_equity=float(eq[-1]),
        trades_per_day=n / days,
        days_covered=days,
        trades=trades,
        equity_curve=equity_curve,
    )


def save_result(result: BacktestResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
