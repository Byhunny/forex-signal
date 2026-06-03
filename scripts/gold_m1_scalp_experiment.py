"""GOLD M1 Scalping Experiment

Tests multiple lookback windows × multiple exit strategies on GOLD M1
to find the most profitable scalping configuration.

ML models:
- LNN classifier with seq_len in [10, 30, 50, 100]
- pred_horizon = 2 (predict next 2 bars)

Exit strategies tested per model:
- TIME1: close after 1 bar (next bar's close)
- TIME2: close after 2 bars
- TIGHT_TPSL: TP 0.5×ATR, SL 0.8×ATR, time exit 3 bars max

Plus baseline (no ML):
- BB_REV: BB lower band + RSI<30 BUY, BB upper + RSI>70 SELL
- RSI_EXT: RSI<20 BUY, RSI>80 SELL (extreme mean reversion)

Realistic spread (GOLD live: 5.2 pips = $0.52)
Spread + slippage on entry. Single spread cost (bug-fixed).
"""
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

# Speed up CPU
torch.set_num_threads(24)

from forex_signal.data.features import FEATURE_COLUMNS, build_windows, compute_features
from forex_signal.model.lnn import ForexLNN
from forex_signal.model.train import TrainConfig, train as train_model
from forex_signal.model.predict import Predictor

# === CONFIG ===
SYMBOL = "GOLD"
PIP = 0.10
CONTRACT = 100  # XM GOLD: 1 lot = 100 oz
SPREAD_PIPS = 5.2 * 1.2  # live + 20% buffer = 6.24 pips = $0.624
LOT = 0.01
COMMISSION = 7.0 * LOT
PRED_HORIZON = 2  # predict next 2 bars

SEQ_LENS = [10, 30, 50, 100]
TRAIN_EPOCHS = 25
UNITS = 32  # smaller model for M1 (faster + less overfit)


def train_lnn(features_df, seq_len, model_path):
    if model_path.exists():
        print(f"  [cached] {model_path.name}")
        return
    windows = build_windows(features_df, seq_len=seq_len, pred_horizon=PRED_HORIZON)
    cfg = TrainConfig(
        units=UNITS, dropout=0.15, batch_size=512, epochs=TRAIN_EPOCHS, lr=1e-3,
        weight_decay=1e-4, early_stopping_patience=4, device="cpu",
    )
    t0 = time.time()
    result = train_model(windows, cfg, save_path=model_path)
    print(f"  trained seq_len={seq_len}: val_loss={result.best_val_loss:.4f} cls_acc={result.test_classifier_accuracy:.3f} "
          f"epochs={result.epochs_run} time={time.time()-t0:.1f}s")


def backtest_ml(features_df, model_path, seq_len, exit_mode, prob_threshold=0.55):
    """Backtest an LNN model on the test slice (last 15%)."""
    predictor = Predictor.load(model_path, seq_len=seq_len)
    f_cols = predictor.feature_columns
    f_arr = features_df[f_cols].to_numpy(dtype=np.float32)
    f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds
    close = features_df["close"].to_numpy(dtype=np.float64)
    high = features_df["high"].to_numpy(dtype=np.float64)
    low = features_df["low"].to_numpy(dtype=np.float64)
    atr = features_df["atr_14"].to_numpy(dtype=np.float64)

    n = len(features_df)
    test_start = int(n * 0.85)
    if test_start < seq_len + 10:
        return None

    spread_price = SPREAD_PIPS * PIP
    pnl_per_price = LOT * CONTRACT

    open_pos = None
    trades = []

    for i in range(max(seq_len, test_start), n - PRED_HORIZON):
        # Manage open position
        if open_pos is not None:
            bars_held = i - open_pos["entry_bar"]
            exit_now = False
            exit_price = None
            reason = None

            # SL/TP check (intra-bar)
            if open_pos["sl"] != 0 and ((open_pos["dir"] == 1 and low[i] <= open_pos["sl"]) or
                                          (open_pos["dir"] == -1 and high[i] >= open_pos["sl"])):
                exit_price = open_pos["sl"]
                reason = "sl"
            elif open_pos["tp"] != 0 and ((open_pos["dir"] == 1 and high[i] >= open_pos["tp"]) or
                                            (open_pos["dir"] == -1 and low[i] <= open_pos["tp"])):
                exit_price = open_pos["tp"]
                reason = "tp"
            # Time exit
            elif (exit_mode == "TIME1" and bars_held >= 1) or \
                 (exit_mode == "TIME2" and bars_held >= 2) or \
                 (exit_mode == "TIGHT_TPSL" and bars_held >= 3):
                exit_price = close[i]
                reason = "time"

            if exit_price is not None:
                gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                pnl = gross - 2 * COMMISSION
                trades.append({"pnl": pnl, "bars": bars_held, "reason": reason})
                open_pos = None

        if open_pos is not None:
            continue

        # Inference
        window = f_norm[i - seq_len + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :])
        with torch.no_grad():
            ret_norm, dir_logit = predictor.model(x)
        prob = float(torch.sigmoid(dir_logit).item())

        # Entry decision
        if prob >= prob_threshold:
            direction = 1
        elif prob <= 1 - prob_threshold:
            direction = -1
        else:
            continue

        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue

        entry_price = close[i] + spread_price * direction
        tp = 0
        sl = 0
        if exit_mode == "TIGHT_TPSL":
            tp_dist = 0.5 * atr[i]
            sl_dist = 0.8 * atr[i]
            tp = entry_price + tp_dist * direction
            sl = entry_price - sl_dist * direction
        else:
            # Time exit modes: only SL for safety
            sl_dist = 2.0 * atr[i]
            sl = entry_price - sl_dist * direction

        open_pos = {
            "entry": entry_price, "entry_bar": i, "dir": direction,
            "sl": sl, "tp": tp, "prob": prob,
        }

    return _compute_metrics(trades)


def backtest_indicator(features_df, strategy_name):
    """Backtest a pure indicator strategy."""
    f_arr = features_df
    close = f_arr["close"].to_numpy(dtype=np.float64)
    high = f_arr["high"].to_numpy(dtype=np.float64)
    low = f_arr["low"].to_numpy(dtype=np.float64)
    atr = f_arr["atr_14"].to_numpy(dtype=np.float64)
    rsi = (f_arr["rsi_14"] * 100).to_numpy(dtype=np.float64)  # un-normalize
    bb_pos = f_arr["bb_position"].to_numpy(dtype=np.float64)

    n = len(features_df)
    test_start = int(n * 0.85)
    spread_price = SPREAD_PIPS * PIP
    pnl_per_price = LOT * CONTRACT

    open_pos = None
    trades = []

    for i in range(max(20, test_start), n - 1):
        if open_pos is not None:
            bars_held = i - open_pos["entry_bar"]
            exit_price = None
            reason = None
            if (open_pos["dir"] == 1 and low[i] <= open_pos["sl"]) or \
               (open_pos["dir"] == -1 and high[i] >= open_pos["sl"]):
                exit_price = open_pos["sl"]
                reason = "sl"
            elif open_pos["tp"] != 0 and ((open_pos["dir"] == 1 and high[i] >= open_pos["tp"]) or
                                          (open_pos["dir"] == -1 and low[i] <= open_pos["tp"])):
                exit_price = open_pos["tp"]
                reason = "tp"
            elif bars_held >= 3:
                exit_price = close[i]
                reason = "time"
            if exit_price is not None:
                gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                pnl = gross - 2 * COMMISSION
                trades.append({"pnl": pnl, "bars": bars_held, "reason": reason})
                open_pos = None
        if open_pos is not None:
            continue

        direction = 0
        if strategy_name == "BB_REV":
            # BB lower extreme + RSI oversold = BUY mean reversion
            if bb_pos[i] < -0.4 and rsi[i] < 30:
                direction = 1
            elif bb_pos[i] > 0.4 and rsi[i] > 70:
                direction = -1
        elif strategy_name == "RSI_EXT":
            if rsi[i] < 20:
                direction = 1
            elif rsi[i] > 80:
                direction = -1

        if direction == 0 or atr[i] <= 0:
            continue
        entry_price = close[i] + spread_price * direction
        sl_dist = 0.8 * atr[i]
        tp_dist = 0.6 * atr[i]
        open_pos = {
            "entry": entry_price, "entry_bar": i, "dir": direction,
            "sl": entry_price - sl_dist * direction,
            "tp": entry_price + tp_dist * direction,
        }
    return _compute_metrics(trades)


def _compute_metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_bars": 0, "tp_n": 0, "sl_n": 0, "time_n": 0}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        "n": len(trades),
        "wr": round(len(wins) / len(trades), 3),
        "pf": round(float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 99.99, 3),
        "pnl": round(float(pnls.sum()), 2),
        "avg_bars": round(float(np.mean([t["bars"] for t in trades])), 2),
        "tp_n": sum(1 for t in trades if t["reason"] == "tp"),
        "sl_n": sum(1 for t in trades if t["reason"] == "sl"),
        "time_n": sum(1 for t in trades if t["reason"] == "time"),
    }


# === MAIN ===
print("=" * 90)
print("GOLD M1 SCALPING EXPERIMENT")
print("=" * 90)

print("Loading data...")
df = pd.read_parquet(ROOT / "data_cache" / "GOLD_mt5_M1.parquet")
print(f"  {len(df)} bars, {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

print("Computing features...")
features_df = compute_features(df)
print(f"  {len(features_df)} bars after warmup")

# Train models
print(f"\nTraining {len(SEQ_LENS)} models (units={UNITS}, max epochs={TRAIN_EPOCHS})...")
for sl in SEQ_LENS:
    model_path = ROOT / "models" / f"gold_m1_lnn_sl{sl}.pt"
    print(f"seq_len={sl}:")
    train_lnn(features_df, sl, model_path)

# Backtest all combinations
print("\n" + "=" * 90)
print("BACKTESTING")
print("=" * 90)
print(f"{'Strategy':<32} {'n':>5} {'WR':>6} {'PF':>6} {'PnL':>9} {'AvgBars':>7} {'TP':>4} {'SL':>4} {'Time':>5}")
print("-" * 100)

results = []
for sl in SEQ_LENS:
    model_path = ROOT / "models" / f"gold_m1_lnn_sl{sl}.pt"
    if not model_path.exists():
        continue
    for mode in ["TIME1", "TIME2", "TIGHT_TPSL"]:
        r = backtest_ml(features_df, model_path, sl, mode)
        if r is None:
            continue
        name = f"LNN_sl{sl}_{mode}"
        r["strategy"] = name
        results.append(r)
        print(f"{name:<32} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f} {r['avg_bars']:>7.1f} {r['tp_n']:>4} {r['sl_n']:>4} {r['time_n']:>5}", flush=True)

# Indicator baselines
for strat_name in ["BB_REV", "RSI_EXT"]:
    r = backtest_indicator(features_df, strat_name)
    r["strategy"] = strat_name
    results.append(r)
    print(f"{strat_name:<32} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f} {r['avg_bars']:>7.1f} {r['tp_n']:>4} {r['sl_n']:>4} {r['time_n']:>5}", flush=True)

# Save
out = ROOT / "logs" / "gold_m1_experiment.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out}")

# Top
print("\n" + "=" * 90)
print("TOP 5 BY P&L")
print("=" * 90)
top = sorted(results, key=lambda x: x["pnl"], reverse=True)[:5]
for r in top:
    print(f"  {r['strategy']:<32} n={r['n']:>5} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} PnL=${r['pnl']:+8.2f}")
