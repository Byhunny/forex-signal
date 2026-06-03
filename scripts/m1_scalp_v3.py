"""M1 scalp hunt v3 — fixed SL/TP placement + volatility filter.

Key fixes from v2:
- SL distance must be >= 1.5 × spread (prevents instant SL trigger)
- TP distance must be >= 1.5 × spread (prevents impossible TP)
- VOLATILITY FILTER: only trade when |current bar return| > 1.5 × spread
  (i.e., only enter on strong moves — riding momentum bursts)
- Entry only when ATR > 0.8 × spread (avoid quiet periods)

Strategies:
- BREAKOUT: enter direction of current strong bar (momentum continuation)
- FADE: enter OPPOSITE of strong bar (mean reversion on overshoot)
- LNN: model signal but only on volatile bars
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
torch.set_num_threads(24)

from forex_signal.data.features import build_windows, compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.config import load_config
from forex_signal.model.train import TrainConfig, train as train_model
from forex_signal.model.predict import Predictor

# 4 paritede dene — daha geniş scan
SYMBOLS = {
    "EURUSD": {"pip": 0.0001, "contract": 100_000, "spread_pips": 1.9 * 1.3},
    "USDJPY": {"pip": 0.01,   "contract": 100_000, "spread_pips": 2.4 * 1.3},
    "EURJPY": {"pip": 0.01,   "contract": 100_000, "spread_pips": 3.2 * 1.3},
    "GBPJPY": {"pip": 0.01,   "contract": 100_000, "spread_pips": 5.0 * 1.3},  # volatile crosses
}
LOT = 0.01
COMMISSION = 7.0 * LOT


def backtest(features_df, symbol, strategy, exit_mode, predictor=None,
             vol_filter_mult=1.5, prob_threshold=0.55):
    """Backtest with proper SL/TP placement + volatility filter."""
    cfg = SYMBOLS[symbol]
    pip = cfg["pip"]
    contract = cfg["contract"]
    spread_price = cfg["spread_pips"] * pip
    pnl_per_price = LOT * contract

    close = features_df["close"].to_numpy(dtype=np.float64)
    high = features_df["high"].to_numpy(dtype=np.float64)
    low = features_df["low"].to_numpy(dtype=np.float64)
    atr = features_df["atr_14"].to_numpy(dtype=np.float64)
    ret_1 = features_df["ret_1"].to_numpy(dtype=np.float64)
    rsi = (features_df["rsi_14"] * 100).to_numpy(dtype=np.float64)
    bb_pos = features_df["bb_position"].to_numpy(dtype=np.float64)
    co_range = features_df["co_range"].to_numpy(dtype=np.float64)  # (close-open)/open

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

    # Pre-compute typical bar magnitude (used for vol filter)
    bar_size = np.abs(close - close * np.roll(np.ones(n), 1) / close)  # placeholder
    # Use abs(co_range) as proxy for bar move
    abs_move = np.abs(co_range) * close  # absolute move in price units

    for i in range(max(seq_len + 1, test_start), n - 1):
        # Manage open position
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
            elif (exit_mode == "T2" and bars_held >= 2) or \
                 (exit_mode == "T3" and bars_held >= 3) or \
                 (exit_mode == "T5" and bars_held >= 5) or \
                 (exit_mode == "TPSL" and bars_held >= 10):
                exit_price = close[i]
                reason = "time"
            if exit_price is not None:
                gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                pnl = gross - 2 * COMMISSION
                trades.append({"pnl": pnl, "bars": bars_held, "reason": reason})
                open_pos = None
        if open_pos is not None:
            continue

        # === FILTERS ===
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue
        # Volatility filter: only trade when ATR > spread (otherwise impossible to win)
        if atr[i] < 0.8 * spread_price:
            continue
        # Bar volatility filter: current bar must be significant move
        current_bar_move = abs_move[i]  # |close - open|
        if current_bar_move < vol_filter_mult * spread_price:
            continue

        # === STRATEGY ===
        direction = 0
        if strategy == "BREAKOUT":
            # Enter in direction of current strong bar
            if co_range[i] > 0:
                direction = 1
            else:
                direction = -1
        elif strategy == "FADE":
            # Enter OPPOSITE of strong bar (mean reversion)
            if co_range[i] > 0:
                direction = -1
            else:
                direction = 1
        elif strategy == "BB_BURST":
            # BB extreme + strong move = mean reversion expected
            if bb_pos[i] < -0.3 and co_range[i] < 0:  # already moved down, expect bounce
                direction = 1
            elif bb_pos[i] > 0.3 and co_range[i] > 0:
                direction = -1
        elif strategy == "BB_TREND":
            # BB middle pullback in trend direction
            ema_ratio = features_df["ema_ratio"].iloc[i]
            if ema_ratio > 0.0003 and bb_pos[i] < -0.1 and co_range[i] > 0:
                direction = 1
            elif ema_ratio < -0.0003 and bb_pos[i] > 0.1 and co_range[i] < 0:
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
            # Take the OPPOSITE of LNN prediction (some models systematically wrong)
            window = f_norm[i - seq_len + 1 : i + 1]
            x = torch.from_numpy(window[None, :, :])
            with torch.no_grad():
                ret_norm, dir_logit = predictor.model(x)
            prob = float(torch.sigmoid(dir_logit).item())
            if prob >= prob_threshold:
                direction = -1
            elif prob <= 1 - prob_threshold:
                direction = 1

        if direction == 0:
            continue

        # === ENTRY ===
        entry_price = close[i] + spread_price * direction

        # SL / TP — minimum distances based on spread to prevent instant trigger
        if exit_mode == "TPSL":
            tp_dist = max(1.5 * atr[i], 2.0 * spread_price)  # TP >= 2× spread
            sl_dist = max(2.0 * atr[i], 2.5 * spread_price)  # SL >= 2.5× spread
            tp = entry_price + tp_dist * direction
            sl = entry_price - sl_dist * direction
        else:
            # Time exit modes: SL only, but it must be > spread
            sl_dist = max(3.0 * atr[i], 3.0 * spread_price)
            tp = 0
            sl = entry_price - sl_dist * direction

        open_pos = {"entry": entry_price, "entry_bar": i, "dir": direction,
                    "tp": tp, "sl": sl}

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
print("M1 SCALP HUNT v3 — Spread-aware SL/TP + Volatility filter")
print("=" * 100)

cfg = load_config()
client = make_client(prefer_real=True)
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)

results = []
for symbol in SYMBOLS:
    parquet = ROOT / "data_cache" / f"{symbol}_mt5_M1.parquet"
    if not parquet.exists():
        print(f"\nDownloading {symbol} M1...")
        df = client.fetch_history(symbol, "M1", 50000)
        df.to_parquet(parquet, index=False)
    df = pd.read_parquet(parquet)
    features_df = compute_features(df)
    print(f"\n{symbol}: {len(features_df)} bars  spread={SYMBOLS[symbol]['spread_pips']:.1f}p")

    # LNN model
    model_path = ROOT / "models" / f"{symbol.lower()}_m1_sl30.pt"
    if not model_path.exists():
        print(f"  training LNN seq_len=30...")
        windows = build_windows(features_df, seq_len=30, pred_horizon=2)
        tc = TrainConfig(units=32, dropout=0.15, batch_size=512, epochs=20, lr=1e-3,
                        weight_decay=1e-4, early_stopping_patience=4, device="cpu")
        t0 = time.time()
        train_model(windows, tc, save_path=model_path)
        print(f"  trained in {time.time()-t0:.1f}s")
    predictor = Predictor.load(model_path, seq_len=30)

    strategies = ["BREAKOUT", "FADE", "BB_BURST", "BB_TREND", "LNN", "LNN_FADE"]
    exits = ["T2", "T3", "T5", "TPSL"]
    vol_mults = [1.0, 1.5, 2.0]  # vol filter strictness

    print(f"  {'Strategy':<12} {'Exit':<5} {'VolMult':<8} {'n':>5} {'WR':>6} {'PF':>6} {'PnL':>9}")
    for strat in strategies:
        for ex in exits:
            for vm in vol_mults:
                pred = predictor if strat.startswith("LNN") else None
                r = backtest(features_df, symbol, strat, ex, pred, vol_filter_mult=vm)
                if r["n"] < 10:
                    continue
                r["symbol"] = symbol
                r["strategy"] = strat
                r["exit"] = ex
                r["vol_mult"] = vm
                results.append(r)
                mark = "✓" if r["pf"] > 1.0 else " "
                print(f"  {strat:<12} {ex:<5} {vm:<8} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f}  {mark}", flush=True)

client.shutdown()

out = ROOT / "logs" / "m1_scalp_v3.json"
out.write_text(json.dumps(results, indent=2))

profitable = sorted([r for r in results if r["pf"] > 1.0 and r["n"] >= 20], key=lambda x: x["pnl"], reverse=True)
print("\n" + "=" * 100)
print(f"PROFITABLE (PF>1, n>=20): {len(profitable)} found")
print("=" * 100)
for r in profitable[:20]:
    print(f"  {r['symbol']:<8} {r['strategy']:<12} {r['exit']:<5} vm={r['vol_mult']:<4} n={r['n']:>4} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} PnL=${r['pnl']:+8.2f}")
print(f"\nSaved {out}")
