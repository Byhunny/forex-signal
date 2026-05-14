"""Command-line entry points."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from forex_signal.backtest.walk_forward import BacktestConfig, run_backtest, save_result
from forex_signal.config import PROJECT_ROOT, load_config
from forex_signal.data.features import build_windows, compute_features
from forex_signal.data.mt5_client import is_mt5_available, make_client
from forex_signal.data.yfinance_loader import load_eurusd
from forex_signal.model.predict import Predictor
from forex_signal.model.train import TrainConfig, train
from forex_signal.strategy.entry_engine import EntryConfig

log = logging.getLogger("forex_signal")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lnn_eurusd_m5.pt"
DEFAULT_CACHE = PROJECT_ROOT / "data_cache"


def cmd_download(args: argparse.Namespace) -> int:
    cfg = load_config()
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    if args.source == "yfinance":
        df = load_eurusd(start=args.start, end=args.end, interval=args.interval, cache_dir=DEFAULT_CACHE)
        out = DEFAULT_CACHE / f"{cfg.symbol}_yf_{args.interval}.parquet"
        df.to_parquet(out, index=False)
        log.info("saved %d rows to %s", len(df), out)
        return 0
    if args.source == "mt5":
        if not is_mt5_available():
            log.error("MetaTrader5 not available — run on Windows")
            return 2
        client = make_client(prefer_real=True)
        ok = client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path) if cfg.mt5 else False
        if not ok:
            log.error("mt5 connect failed")
            return 2
        df = client.fetch_history(cfg.symbol, cfg.timeframe, args.bars)
        out = DEFAULT_CACHE / f"{cfg.symbol}_mt5_{cfg.timeframe}.parquet"
        df.to_parquet(out, index=False)
        log.info("saved %d rows to %s", len(df), out)
        client.shutdown()
        return 0
    log.error("unknown source: %s", args.source)
    return 2


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config()
    seq_len = int(cfg.get("data", "seq_len", default=60))
    pred_horizon = int(cfg.get("data", "pred_horizon", default=5))

    in_path = Path(args.input) if args.input else _autodetect_cache(cfg.symbol)
    if in_path is None or not in_path.exists():
        log.error("no input parquet — run `cli download` first or pass --input")
        return 2
    df = pd.read_parquet(in_path)
    features = compute_features(df)
    windows = build_windows(features, seq_len=seq_len, pred_horizon=pred_horizon)

    tc = TrainConfig(
        units=int(cfg.get("model", "units", default=48)),
        dropout=float(cfg.get("model", "dropout", default=0.1)),
        batch_size=int(cfg.get("train", "batch_size", default=128)),
        epochs=args.epochs if args.epochs else int(cfg.get("train", "epochs", default=40)),
        lr=float(cfg.get("train", "lr", default=1e-3)),
        weight_decay=float(cfg.get("train", "weight_decay", default=1e-5)),
        early_stopping_patience=int(cfg.get("train", "early_stopping_patience", default=6)),
        val_fraction=float(cfg.get("train", "val_fraction", default=0.15)),
        test_fraction=float(cfg.get("train", "test_fraction", default=0.15)),
        device="cpu",
    )
    save_path = Path(args.output) if args.output else DEFAULT_MODEL_PATH
    result = train(windows, tc, save_path=save_path)
    log.info(
        "trained: best_val=%.6f test_loss=%.6f dir_acc=%.3f epochs=%d -> %s",
        result.best_val_loss, result.test_loss, result.test_directional_accuracy, result.epochs_run, result.model_path,
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config()
    seq_len = int(cfg.get("data", "seq_len", default=60))
    in_path = Path(args.input) if args.input else _autodetect_cache(cfg.symbol)
    if in_path is None or not in_path.exists():
        log.error("no input parquet")
        return 2
    df = pd.read_parquet(in_path)
    predictor = Predictor.load(args.model or DEFAULT_MODEL_PATH, seq_len=seq_len)
    entry_cfg = EntryConfig(
        require_session_filter=bool(cfg.get("entry", "require_session_filter", default=True)),
        require_htf_bias=bool(cfg.get("entry", "require_htf_bias", default=True)),
        require_sweep=bool(cfg.get("entry", "require_sweep", default=True)),
        require_pullback=bool(cfg.get("entry", "require_pullback", default=True)),
        require_bos_or_choch=bool(cfg.get("entry", "require_bos_or_choch", default=False)),
        min_lnn_probability=float(cfg.get("entry", "min_lnn_probability", default=0.55)),
        sweep_recent_window=int(cfg.get("entry", "sweep_recent_window", default=3)),
        pullback_window=int(cfg.get("entry", "pullback_window", default=4)),
        invert_direction=bool(cfg.get("entry", "invert_direction", default=False)),
    )
    bc = BacktestConfig(
        seq_len=seq_len,
        lot_size=float(cfg.get("risk", "lot_size", default=0.01)),
        leverage=int(cfg.get("risk", "leverage", default=400)),
        spread_pips=float(cfg.get("backtest", "spread_pips", default=cfg.get("risk", "spread_pips_estimate", default=0.8))),
        slippage_pips=float(cfg.get("backtest", "slippage_pips", default=0.3)),
        commission_per_lot=float(cfg.get("backtest", "commission_per_lot", default=7.0)),
        sl_atr_multiplier=float(cfg.get("risk", "sl_atr_multiplier", default=1.5)),
        tp_atr_min_multiplier=float(cfg.get("risk", "tp_atr_min_multiplier", default=0.6)),
        tp_atr_max_multiplier=float(cfg.get("risk", "tp_atr_max_multiplier", default=1.5)),
        max_concurrent_positions=int(cfg.get("risk", "max_concurrent_positions", default=2)),
        cooldown_bars=int(cfg.get("signal", "cooldown_bars", default=3)),
        entry=entry_cfg,
    )
    result = run_backtest(df, predictor, bc)
    out = PROJECT_ROOT / "logs" / f"backtest_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    save_result(result, out)
    log.info(
        "backtest: n=%d win_rate=%.3f PF=%.3f total_pnl=%.2f sharpe=%.2f max_dd=%.2f%% final_equity=%.2f -> %s",
        result.n_trades, result.win_rate, result.profit_factor, result.total_pnl, result.sharpe, result.max_drawdown_pct, result.final_equity, out,
    )
    return 0


def cmd_predict_once(args: argparse.Namespace) -> int:
    cfg = load_config()
    seq_len = int(cfg.get("data", "seq_len", default=60))
    predictor = Predictor.load(args.model or DEFAULT_MODEL_PATH, seq_len=seq_len)

    if cfg.mt5 and is_mt5_available():
        client = make_client(prefer_real=True)
        client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)
        df = client.fetch_history(cfg.symbol, cfg.timeframe, 200)
        client.shutdown()
    else:
        in_path = _autodetect_cache(cfg.symbol)
        if in_path is None:
            log.error("no MT5 and no cache — run download first")
            return 2
        df = pd.read_parquet(in_path).tail(200)

    pred, atr = predictor.predict_returns(df)
    cum = float(pred.sum())
    log.info("predicted_returns=%s cum_log_ret=%.6f cum_bps=%.2f atr=%.6f", pred.tolist(), cum, cum * 1e4, atr)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    if not is_mt5_available():
        log.error("MetaTrader5 not available — run on Windows")
        return 2
    from forex_signal.execution.live_loop import run_live  # lazy import
    return run_live(mode=args.mode)


def cmd_battle(args: argparse.Namespace) -> int:
    if not is_mt5_available():
        log.error("MetaTrader5 not available — run on Windows")
        return 2
    from forex_signal.execution.battle_royale import run_battle_royale
    return run_battle_royale(mode=args.mode)


def _autodetect_cache(symbol: str) -> Path | None:
    candidates = sorted(DEFAULT_CACHE.glob(f"{symbol}_*.parquet"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    parser = argparse.ArgumentParser("forex-signal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--source", choices=["yfinance", "mt5"], default="yfinance")
    p_dl.add_argument("--start", default="2023-01-01")
    p_dl.add_argument("--end", default=None)
    p_dl.add_argument("--interval", default="1h", help="yfinance interval (1h, 1d)")
    p_dl.add_argument("--bars", type=int, default=10000, help="mt5 bar count")
    p_dl.set_defaults(func=cmd_download)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--input", default=None)
    p_tr.add_argument("--output", default=None)
    p_tr.add_argument("--epochs", type=int, default=0)
    p_tr.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--input", default=None)
    p_bt.add_argument("--model", default=None)
    p_bt.set_defaults(func=cmd_backtest)

    p_pr = sub.add_parser("predict-once")
    p_pr.add_argument("--model", default=None)
    p_pr.set_defaults(func=cmd_predict_once)

    p_li = sub.add_parser("live")
    p_li.add_argument("--mode", choices=["paper", "live"], default="paper")
    p_li.set_defaults(func=cmd_live)

    p_br = sub.add_parser("battle", help="Run all TOP 10 strategies in parallel — battle royale")
    p_br.add_argument("--mode", choices=["paper", "live"], default="paper")
    p_br.set_defaults(func=cmd_battle)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
