"""GOLD M5 Scalping — focused hunt for profitable 5-15 min scalp system.

GOLD M5 specs:
- Spread: ~5.2 pip × $0.10 = $0.52
- ATR typical: $1.50-3.00 (3-6× bigger than M1)
- Spread:ATR ratio: ~15-35% (vs M1's 250%)

This is the sweet spot for true scalping.
"""
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force UTF-8 stdout on Windows (cp1254 default cannot encode emojis/checkmarks)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(24)

from forex_signal.data.features import build_windows, compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.config import load_config
from forex_signal.model.train import TrainConfig, train as train_model
from forex_signal.model.predict import Predictor

SYMBOL = "GOLD"
PIP = 0.10
CONTRACT = 100
SPREAD_PIPS = 5.2 * 1.3  # live + 30% buffer = 6.76 pips = $0.676
LOT = 0.01
COMMISSION = 7.0 * LOT


def backtest(features_df, strategy, exit_mode, predictor=None, vol_filter_mult=1.0, prob_threshold=0.55):
    spread_price = SPREAD_PIPS * PIP
    pnl_per_price = LOT * CONTRACT

    close = features_df["close"].to_numpy(dtype=np.float64)
    high = features_df["high"].to_numpy(dtype=np.float64)
    low = features_df["low"].to_numpy(dtype=np.float64)
    atr = features_df["atr_14"].to_numpy(dtype=np.float64)
    ret_1 = features_df["ret_1"].to_numpy(dtype=np.float64)
    rsi = (features_df["rsi_14"] * 100).to_numpy(dtype=np.float64)
    bb_pos = features_df["bb_position"].to_numpy(dtype=np.float64)
    co_range = features_df["co_range"].to_numpy(dtype=np.float64)
    ema_ratio = features_df["ema_ratio"].to_numpy(dtype=np.float64)
    abs_move = np.abs(co_range) * close

    if predictor is not None:
        f_arr = features_df[predictor.feature_columns].to_numpy(dtype=np.float32)
        f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds
        seq_len = predictor.seq_len
    else:
        seq_len = 30

    n = len(features_df)
    test_start = int(n * 0.85)
    open_pos = None
    trades = []

    for i in range(max(seq_len + 1, test_start), n - 1):
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
            elif (exit_mode == "T1" and bars_held >= 1) or \
                 (exit_mode == "T2" and bars_held >= 2) or \
                 (exit_mode == "T3" and bars_held >= 3) or \
                 (exit_mode == "T5" and bars_held >= 5) or \
                 (exit_mode == "TPSL" and bars_held >= 8):
                exit_price = close[i]
                reason = "time"
            if exit_price is not None:
                gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                pnl = gross - 2 * COMMISSION
                trades.append({"pnl": pnl, "bars": bars_held, "reason": reason})
                open_pos = None
        if open_pos is not None:
            continue

        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue
        if atr[i] < 1.0 * spread_price:
            continue
        current_bar_move = abs_move[i]
        if vol_filter_mult > 0 and current_bar_move < vol_filter_mult * spread_price:
            continue

        direction = 0
        if strategy == "BREAKOUT":
            direction = 1 if co_range[i] > 0 else -1
        elif strategy == "FADE":
            direction = -1 if co_range[i] > 0 else 1
        elif strategy == "BB_BURST_REV":
            if bb_pos[i] < -0.3 and co_range[i] < 0:
                direction = 1
            elif bb_pos[i] > 0.3 and co_range[i] > 0:
                direction = -1
        elif strategy == "BB_TREND":
            if ema_ratio[i] > 0.0005 and bb_pos[i] < -0.1 and co_range[i] > 0:
                direction = 1
            elif ema_ratio[i] < -0.0005 and bb_pos[i] > 0.1 and co_range[i] < 0:
                direction = -1
        elif strategy == "RSI_REV":
            if rsi[i] < 30 and co_range[i] > 0:  # RSI low + bullish bar = bounce
                direction = 1
            elif rsi[i] > 70 and co_range[i] < 0:
                direction = -1
        elif strategy == "LNN":
            window = f_norm[i - seq_len + 1 : i + 1]
            x = torch.from_numpy(window[None, :, :])
            with torch.no_grad():
                ret_norm, dir_logit = predictor.model(x)
            prob = float(torch.sigmoid(dir_logit).item())
            if prob >= prob_threshold:
                direction = 1
            elif prob <= 1 - prob_threshold:
                direction = -1
        elif strategy == "LNN_FADE":
            window = f_norm[i - seq_len + 1 : i + 1]
            x = torch.from_numpy(window[None, :, :])
            with torch.no_grad():
                ret_norm, dir_logit = predictor.model(x)
            prob = float(torch.sigmoid(dir_logit).item())
            if prob >= prob_threshold:
                direction = -1
            elif prob <= 1 - prob_threshold:
                direction = 1
        elif strategy == "LNN_TREND":
            # LNN + trend agreement
            window = f_norm[i - seq_len + 1 : i + 1]
            x = torch.from_numpy(window[None, :, :])
            with torch.no_grad():
                ret_norm, dir_logit = predictor.model(x)
            prob = float(torch.sigmoid(dir_logit).item())
            if prob >= prob_threshold and ema_ratio[i] > 0:
                direction = 1
            elif prob <= 1 - prob_threshold and ema_ratio[i] < 0:
                direction = -1

        if direction == 0:
            continue

        entry_price = close[i] + spread_price * direction
        if exit_mode == "TPSL":
            tp_dist = max(1.0 * atr[i], 2.0 * spread_price)
            sl_dist = max(1.5 * atr[i], 2.5 * spread_price)
        elif exit_mode == "TPSL_TIGHT":
            tp_dist = max(0.7 * atr[i], 1.5 * spread_price)
            sl_dist = max(1.5 * atr[i], 2.5 * spread_price)
        else:
            tp_dist = 0
            sl_dist = max(3.0 * atr[i], 3.5 * spread_price)
        tp = entry_price + tp_dist * direction if tp_dist > 0 else 0
        sl = entry_price - sl_dist * direction
        open_pos = {"entry": entry_price, "entry_bar": i, "dir": direction, "tp": tp, "sl": sl}

    return _metrics(trades)


def _metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_bars": 0, "tp": 0, "sl": 0, "time": 0}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        "n": len(trades),
        "wr": round(len(wins) / len(trades), 3),
        "pf": round(float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 99.99, 3),
        "pnl": round(float(pnls.sum()), 2),
        "avg_bars": round(float(np.mean([t["bars"] for t in trades])), 1),
        "tp": sum(1 for t in trades if t["reason"] == "tp"),
        "sl": sum(1 for t in trades if t["reason"] == "sl"),
        "time": sum(1 for t in trades if t["reason"] == "time"),
    }


# === MAIN ===
print("=" * 100)
print("GOLD M5 SCALPING HUNT")
print("=" * 100)

cfg = load_config()
client = make_client(prefer_real=True)
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)

parquet = ROOT / "data_cache" / f"{SYMBOL}_mt5_M5.parquet"
if not parquet.exists():
    print(f"Downloading {SYMBOL} M5...")
    df = client.fetch_history(SYMBOL, "M5", 50000)
    df.to_parquet(parquet, index=False)
df = pd.read_parquet(parquet)
features_df = compute_features(df)
print(f"GOLD M5: {len(features_df)} bars  spread=${SPREAD_PIPS * PIP:.2f}")
print(f"  ATR: mean=${features_df['atr_14'].mean():.2f}  median=${features_df['atr_14'].median():.2f}")
print(f"  Spread:ATR ratio: {(SPREAD_PIPS * PIP) / features_df['atr_14'].mean() * 100:.1f}%")

# Train LNN models with different seq_lens
predictors = {}
for sl in [20, 50, 100]:
    model_path = ROOT / "models" / f"gold_m5_lnn_sl{sl}.pt"
    if not model_path.exists():
        print(f"\nTraining GOLD M5 LNN seq_len={sl}...")
        windows = build_windows(features_df, seq_len=sl, pred_horizon=3)
        tc = TrainConfig(units=48, dropout=0.15, batch_size=256, epochs=25, lr=8e-4,
                        weight_decay=1e-4, early_stopping_patience=5, device="cpu")
        t0 = time.time()
        train_model(windows, tc, save_path=model_path)
        print(f"  done in {time.time()-t0:.1f}s")
    predictors[sl] = Predictor.load(model_path, seq_len=sl)

# Test all combinations
print(f"\n{'Strategy':<14} {'Exit':<10} {'VolMult':<8} {'SeqLen':<7} {'n':>5} {'WR':>6} {'PF':>6} {'PnL':>9}")
results = []
indicator_strategies = ["BREAKOUT", "FADE", "BB_BURST_REV", "BB_TREND", "RSI_REV"]
ml_strategies = ["LNN", "LNN_FADE", "LNN_TREND"]
exits = ["T2", "T3", "T5", "TPSL", "TPSL_TIGHT"]
vol_mults = [0.0, 1.0, 1.5]  # 0 = no filter

for strat in indicator_strategies:
    for ex in exits:
        for vm in vol_mults:
            r = backtest(features_df, strat, ex, vol_filter_mult=vm)
            if r["n"] < 10:
                continue
            r["strategy"] = strat; r["exit"] = ex; r["vol_mult"] = vm; r["seq_len"] = None
            results.append(r)
            m = "OK" if r["pf"] > 1.0 else " "
            print(f"{strat:<14} {ex:<10} {vm:<8} {'-':<7} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f}  {m}", flush=True)

for strat in ml_strategies:
    for sl, pred in predictors.items():
        for ex in exits:
            for vm in vol_mults:
                r = backtest(features_df, strat, ex, predictor=pred, vol_filter_mult=vm)
                if r["n"] < 10:
                    continue
                r["strategy"] = strat; r["exit"] = ex; r["vol_mult"] = vm; r["seq_len"] = sl
                results.append(r)
                m = "OK" if r["pf"] > 1.0 else " "
                print(f"{strat:<14} {ex:<10} {vm:<8} {sl:<7} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f}  {m}", flush=True)

client.shutdown()

out = ROOT / "logs" / "gold_m5_scalp.json"
out.write_text(json.dumps(results, indent=2))
profitable = sorted([r for r in results if r["pf"] > 1.0 and r["n"] >= 20], key=lambda x: x["pnl"], reverse=True)
print("\n" + "=" * 100)
print(f"PROFITABLE GOLD M5 (PF>1, n>=20): {len(profitable)} found")
print("=" * 100)
for r in profitable[:20]:
    sl_str = f"sl{r['seq_len']}" if r['seq_len'] else "no_ml"
    print(f"  {r['strategy']:<14} {r['exit']:<10} vm={r['vol_mult']:<4} {sl_str:<6} n={r['n']:>4} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} PnL=${r['pnl']:+8.2f}")
print(f"\nSaved {out}")
