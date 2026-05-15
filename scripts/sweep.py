"""Systematic sweep: try multiple symbol × timeframe × config combinations,
report which produces the best (trade volume × win rate × PF) combination."""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("sweep")
log.setLevel(logging.INFO)

import pandas as pd
import torch

from forex_signal.data.features import build_windows, compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.config import load_config
from forex_signal.model.train import TrainConfig, train as train_model
from forex_signal.model.predict import Predictor
from forex_signal.backtest.walk_forward import BacktestConfig, run_backtest
from forex_signal.strategy.entry_engine import EntryConfig

# === CONFIGURATION ===
SYMBOLS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    # Crosses
    "EURJPY", "EURGBP", "EURAUD", "EURCAD",
    # Exotics
    "USDCNH", "USDSEK",
    # Metals
    "GOLD", "SILVER",
    # Indices
    "US100Cash", "US500Cash", "GER40Cash",
    # Crypto
    "BTCUSD",
]
TIMEFRAMES = ["M5", "M15", "M30"]
BARS = 15000
SEQ_LEN = 50
PRED_HORIZON = 5
TRAIN_EPOCHS = 12
UNITS = 48

# Strategy configurations to test for each (symbol, tf) combination
STRATEGY_CONFIGS = [
    # (name, EntryConfig overrides)
    ("smc_strict",     dict(require_session_filter=False, require_htf_bias=True,
                             require_sweep=True, require_pullback=False,
                             min_lnn_probability=0.55, sweep_recent_window=10)),
    ("lnn_strong",     dict(require_session_filter=False, require_htf_bias=False,
                             require_sweep=False, require_pullback=False,
                             min_lnn_probability=0.60)),
    ("lnn_very_strong", dict(require_session_filter=False, require_htf_bias=False,
                              require_sweep=False, require_pullback=False,
                              min_lnn_probability=0.65)),
    ("trend_lnn",      dict(require_session_filter=False, require_htf_bias=True,
                             require_sweep=False, require_pullback=False,
                             min_lnn_probability=0.55)),
]

# Per-symbol pip estimates (price unit per 1 pip in standard 5-digit terms)
PIP_VALUE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "USDSEK": 0.0001, "USDCNH": 0.0001,
    "USDJPY": 0.01, "EURJPY": 0.01,
    "GOLD": 0.10, "SILVER": 0.01,
    "BTCUSD": 1.0,
    "US500Cash": 0.1, "US100Cash": 0.1,
    "GER40Cash": 1.0, "US30Cash": 1.0,
}

CONTRACT_SIZE = {
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "USDCAD": 100_000,
    "NZDUSD": 100_000, "USDCHF": 100_000, "EURGBP": 100_000, "EURAUD": 100_000,
    "EURCAD": 100_000, "USDSEK": 100_000, "USDCNH": 100_000,
    "USDJPY": 100_000, "EURJPY": 100_000,
    "GOLD": 100, "SILVER": 5000,
    "BTCUSD": 1.0,
    "US500Cash": 1.0, "US100Cash": 1.0,
    "GER40Cash": 1.0, "US30Cash": 1.0,
}

# Load LIVE spreads captured from MT5 (run scripts/fetch_spreads.py to refresh).
# Add a safety buffer for off-peak times where spreads widen.
def _load_live_spreads():
    try:
        live = json.loads((ROOT / "logs" / "live_spreads.json").read_text())
    except Exception:
        return {}
    result = {}
    for sym, data in live.items():
        if data and "spread_pips" in data:
            # Add 20% safety buffer for off-peak conditions
            result[sym] = round(data["spread_pips"] * 1.2, 1)
    return result

_LIVE_SPREADS = _load_live_spreads()
# Fallback if symbol missing from live data
SPREAD_PIPS = {
    "EURUSD": 1.5, "GBPUSD": 2.5, "USDJPY": 2.0, "USDCHF": 2.5,
    "USDCAD": 3.0, "AUDUSD": 2.5, "NZDUSD": 3.0,
    "EURJPY": 3.0, "EURGBP": 2.5, "EURAUD": 4.0, "EURCAD": 4.0,
    "USDCNH": 40.0, "USDSEK": 150.0,   # exotics — confirmed wide via live data
    "GOLD": 6.0, "SILVER": 8.0,
    "BTCUSD": 60.0,
    "US500Cash": 2.0, "US100Cash": 30.0,
    "GER40Cash": 3.0, "US30Cash": 4.0,
}
SPREAD_PIPS.update(_LIVE_SPREADS)  # live data wins where available


def download_data(client, symbol, timeframe, bars):
    out = ROOT / "data_cache" / f"{symbol}_mt5_{timeframe}.parquet"
    if out.exists():
        cached = pd.read_parquet(out)
        if len(cached) >= bars * 0.9:
            log.info(f"using cached {out.name} ({len(cached)} bars)")
            return cached
    log.info(f"downloading {symbol} {timeframe} {bars} bars")
    df = client.fetch_history(symbol, timeframe, bars)
    df.to_parquet(out, index=False)
    return df


def train_one(df, symbol, timeframe):
    save_path = ROOT / "models" / f"sweep_{symbol}_{timeframe}.pt"
    json_path = save_path.with_suffix(".json")
    features = compute_features(df)
    if len(features) < SEQ_LEN + PRED_HORIZON + 100:
        return None
    # Resume: skip training if model + metrics already exist
    if save_path.exists() and json_path.exists():
        try:
            with open(json_path) as f:
                raw = json.load(f)
            class _R:  # minimal shim with the fields backtest reporting uses
                best_val_loss = raw.get("best_val_loss", 0.0)
                test_classifier_accuracy = raw.get("test_classifier_accuracy", 0.0)
                epochs_run = raw.get("epochs_run", 0)
            print(f"  cached model — skip train (val_loss={_R.best_val_loss:.4f}, cls_acc={_R.test_classifier_accuracy:.3f})", flush=True)
            return save_path, _R
        except Exception:
            pass
    windows = build_windows(features, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON)
    cfg = TrainConfig(
        units=UNITS, dropout=0.15, batch_size=128, epochs=TRAIN_EPOCHS, lr=8e-4,
        weight_decay=5e-5, early_stopping_patience=4, device="cpu",
    )
    result = train_model(windows, cfg, save_path=save_path)
    return save_path, result


def backtest_one(df, predictor_path, symbol, timeframe, strategy_name, strategy_overrides):
    predictor = Predictor.load(predictor_path, seq_len=SEQ_LEN)
    entry = EntryConfig(**strategy_overrides)

    pip = PIP_VALUE.get(symbol, 0.0001)
    contract = CONTRACT_SIZE.get(symbol, 100_000)
    spread = SPREAD_PIPS.get(symbol, 1.0)

    bc = BacktestConfig(
        seq_len=SEQ_LEN,
        initial_balance=10_000,
        lot_size=0.01,
        spread_pips=spread,
        slippage_pips=spread * 0.2,
        commission_per_lot=7.0,
        sl_atr_multiplier=1.5,
        tp_atr_min_multiplier=0.6,
        tp_atr_max_multiplier=1.5,
        max_concurrent_positions=2,
        cooldown_bars=3,
        pip_value=pip,
        contract_size=contract,
        entry=entry,
        spread_min_tp_ratio=2.5,
        spread_min_sl_ratio=4.0,
        spread_skip_ratio=0.45,
    )
    return run_backtest(df, predictor, bc)


def main():
    # Force unbuffered output so Monitor can stream progress
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config()
    client = make_client(prefer_real=True)
    if not client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path):
        print("MT5 connect failed", flush=True)
        return

    # SWEEP v2: realistic-spread + spread-aware backtest
    # JSON-only output (markdown rendered separately by render_sweep_md.py)
    results_path = ROOT / "logs" / "sweep_results_v2.json"
    results_path.parent.mkdir(exist_ok=True)
    results = []
    done_keys = set()
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            done_keys = {(r["symbol"], r["timeframe"], r["strategy"]) for r in results}
            print(f"resuming with {len(results)} previously-saved results", flush=True)
        except Exception:
            results = []

    def _save():
        results_path.write_text(json.dumps(results, indent=2))

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\n=== {symbol} {timeframe} ===")
            try:
                df = download_data(client, symbol, timeframe, BARS)
                if len(df) < SEQ_LEN + 100:
                    print(f"  insufficient data ({len(df)})")
                    continue
            except Exception as e:
                print(f"  download failed: {e}")
                continue

            try:
                trained = train_one(df, symbol, timeframe)
                if trained is None:
                    print("  training failed")
                    continue
                predictor_path, train_result = trained
                print(f"  trained: val_loss={train_result.best_val_loss:.4f}, "
                      f"cls_acc={train_result.test_classifier_accuracy:.3f}, "
                      f"epochs={train_result.epochs_run}")
            except Exception as e:
                print(f"  training error: {e}")
                continue

            for strat_name, strat_overrides in STRATEGY_CONFIGS:
                if (symbol, timeframe, strat_name) in done_keys:
                    print(f"  {strat_name:14s}  [cached]", flush=True)
                    continue
                try:
                    bt = backtest_one(df, predictor_path, symbol, timeframe, strat_name, strat_overrides)
                    row = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strat_name,
                        "n_trades": bt.n_trades,
                        "trades_per_day": round(bt.trades_per_day, 2),
                        "win_rate": round(bt.win_rate, 3),
                        "profit_factor": round(bt.profit_factor, 3),
                        "total_pnl": round(bt.total_pnl, 2),
                        "max_dd_pct": round(bt.max_drawdown_pct, 2),
                        "final_equity": round(bt.final_equity, 2),
                        "days_covered": round(bt.days_covered, 1),
                        "cls_acc": round(train_result.test_classifier_accuracy, 3),
                    }
                    results.append(row)
                    done_keys.add((symbol, timeframe, strat_name))
                    _save()
                    print(f"  {strat_name:14s}  n={bt.n_trades:4d}  td={bt.trades_per_day:5.2f}/day  "
                          f"WR={bt.win_rate:.3f}  PF={bt.profit_factor:.3f}  pnl=${bt.total_pnl:+7.2f}", flush=True)
                except Exception as e:
                    print(f"  backtest error {strat_name}: {e}")

    client.shutdown()
    _save()

    # Generate markdown report
    md_path = ROOT / "docs" / "sweep_report.md"
    md_path.parent.mkdir(exist_ok=True)
    df_r = pd.DataFrame(results)
    if not len(df_r):
        md_path.write_text("# Sweep Report\n\nNo results.\n")
        return

    df_r["score"] = df_r["win_rate"] * df_r["profit_factor"]

    profitable = df_r[(df_r["profit_factor"] > 1.0) & (df_r["n_trades"] >= 20)].copy().sort_values("score", ascending=False)
    high_wr = df_r[(df_r["win_rate"] >= 0.70) & (df_r["n_trades"] >= 20)].copy().sort_values("win_rate", ascending=False)

    lines = []
    lines.append(f"# Sweep Report")
    lines.append(f"\nGenerated: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"\nSymbols: {len(SYMBOLS)}  |  Timeframes: {', '.join(TIMEFRAMES)}  |  Strategies: {len(STRATEGY_CONFIGS)}")
    lines.append(f"\nBars per training: {BARS}  |  Train epochs cap: {TRAIN_EPOCHS}  |  Seq len: {SEQ_LEN}  |  Pred horizon: {PRED_HORIZON}")
    lines.append(f"\nTotal configurations tested: **{len(df_r)}**")

    lines.append("\n## Top profitable configurations (PF > 1.0, n_trades >= 20)")
    if len(profitable):
        cols = ["symbol", "timeframe", "strategy", "n_trades", "trades_per_day", "win_rate", "profit_factor", "total_pnl", "max_dd_pct", "cls_acc"]
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in profitable.head(20).iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    else:
        lines.append("\n_No profitable configurations found._")

    lines.append("\n## High win-rate configurations (WR >= 0.70, n_trades >= 20)")
    if len(high_wr):
        cols = ["symbol", "timeframe", "strategy", "n_trades", "trades_per_day", "win_rate", "profit_factor", "total_pnl", "cls_acc"]
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in high_wr.head(20).iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    lines.append("\n## All results (sorted by win_rate × profit_factor)")
    df_sorted = df_r.sort_values("score", ascending=False)
    cols = ["symbol", "timeframe", "strategy", "n_trades", "trades_per_day", "win_rate", "profit_factor", "total_pnl", "max_dd_pct", "final_equity", "cls_acc"]
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df_sorted.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    md_path.write_text("\n".join(lines))
    print(f"\nReport saved to {md_path}", flush=True)

    print("\n=== TOP 10 BY SCORE (WR × PF) ===", flush=True)
    print(df_sorted.head(10)[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
