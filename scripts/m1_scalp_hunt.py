"""Hunt for a profitable M1 scalping strategy.

Tests multiple symbols + multiple classical/ML strategies + multiple exits.
Goal: find ANY combination that yields PF > 1.1 with realistic spread.
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

# === SYMBOL CONFIG ===
SYMBOLS = {
    "EURUSD": {"pip": 0.0001, "contract": 100_000, "spread_pips": 1.9 * 1.3},   # +30% buffer
    "EURJPY": {"pip": 0.01,   "contract": 100_000, "spread_pips": 3.2 * 1.3},
    "USDJPY": {"pip": 0.01,   "contract": 100_000, "spread_pips": 2.4 * 1.3},
}
LOT = 0.01
COMMISSION = 7.0 * LOT


def session_active(time_arr):
    """London (07:00-08:30 UTC) or NY (13:30-15:00 UTC) opening hour."""
    t = pd.to_datetime(time_arr, utc=True)
    if isinstance(t, pd.Series):
        hours = t.dt.hour
        mins = t.dt.minute
    else:
        hours = pd.Series(t).dt.hour
        mins = pd.Series(t).dt.minute
    london = (hours == 7) | ((hours == 8) & (mins < 30))
    ny = ((hours == 13) & (mins >= 30)) | (hours == 14)
    return (london | ny).to_numpy()


def backtest_strategy(features_df, symbol, strategy, exit_mode, predictor=None, prob_threshold=0.55):
    """Return metrics dict."""
    cfg = SYMBOLS[symbol]
    pip = cfg["pip"]
    contract = cfg["contract"]
    spread_price = cfg["spread_pips"] * pip
    pnl_per_price = LOT * contract
    use_session = strategy.endswith("_SESS")

    f_cols = predictor.feature_columns if predictor is not None else None
    if predictor is not None:
        f_arr = features_df[f_cols].to_numpy(dtype=np.float32)
        f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds

    close = features_df["close"].to_numpy(dtype=np.float64)
    high = features_df["high"].to_numpy(dtype=np.float64)
    low = features_df["low"].to_numpy(dtype=np.float64)
    atr = features_df["atr_14"].to_numpy(dtype=np.float64)
    rsi = (features_df["rsi_14"] * 100).to_numpy(dtype=np.float64)
    bb_pos = features_df["bb_position"].to_numpy(dtype=np.float64)
    ret_1 = features_df["ret_1"].to_numpy(dtype=np.float64)
    ema_ratio = features_df["ema_ratio"].to_numpy(dtype=np.float64)
    sess_active = session_active(features_df["time"]) if use_session else None

    n = len(features_df)
    test_start = int(n * 0.85)
    seq_len = predictor.seq_len if predictor else 30

    open_pos = None
    trades = []

    for i in range(max(seq_len, test_start), n - 1):
        if open_pos is not None:
            bars_held = i - open_pos["entry_bar"]
            exit_price = None
            reason = None
            if open_pos["sl"] != 0 and ((open_pos["dir"] == 1 and low[i] <= open_pos["sl"]) or
                                          (open_pos["dir"] == -1 and high[i] >= open_pos["sl"])):
                exit_price = open_pos["sl"]
                reason = "sl"
            elif open_pos["tp"] != 0 and ((open_pos["dir"] == 1 and high[i] >= open_pos["tp"]) or
                                            (open_pos["dir"] == -1 and low[i] <= open_pos["tp"])):
                exit_price = open_pos["tp"]
                reason = "tp"
            elif (exit_mode == "T1" and bars_held >= 1) or \
                 (exit_mode == "T2" and bars_held >= 2) or \
                 (exit_mode == "T3" and bars_held >= 3) or \
                 (exit_mode == "TPSL" and bars_held >= 5):
                exit_price = close[i]
                reason = "time"
            if exit_price is not None:
                gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                pnl = gross - 2 * COMMISSION
                trades.append({"pnl": pnl, "bars": bars_held, "reason": reason})
                open_pos = None
        if open_pos is not None:
            continue

        if use_session and not sess_active[i]:
            continue
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue

        direction = 0
        # === STRATEGY LOGIC ===
        if strategy.startswith("BB_REV"):
            # M1 BB extremes are gentler — relaxed thresholds
            if bb_pos[i] < -0.30 and rsi[i] < 35:
                direction = 1
            elif bb_pos[i] > 0.30 and rsi[i] > 65:
                direction = -1
        elif strategy.startswith("RSI_EXT"):
            if rsi[i] < 25:
                direction = 1
            elif rsi[i] > 75:
                direction = -1
        elif strategy.startswith("PULLBACK"):
            # Trend + opposite recent bar = pullback entry continuation
            trend_up = ema_ratio[i] > 0.0005
            trend_dn = ema_ratio[i] < -0.0005
            recent_pullback_up = ret_1[i-1] < 0 and ret_1[i] >= 0  # red then green
            recent_pullback_dn = ret_1[i-1] > 0 and ret_1[i] <= 0
            if trend_up and recent_pullback_up:
                direction = 1
            elif trend_dn and recent_pullback_dn:
                direction = -1
        elif strategy.startswith("MOM_3BAR"):
            # 3 consecutive same-direction bars = momentum continuation
            if ret_1[i-2] > 0 and ret_1[i-1] > 0 and ret_1[i] > 0:
                direction = 1
            elif ret_1[i-2] < 0 and ret_1[i-1] < 0 and ret_1[i] < 0:
                direction = -1
        elif strategy.startswith("LNN"):
            window = f_norm[i - seq_len + 1 : i + 1]
            x = torch.from_numpy(window[None, :, :])
            with torch.no_grad():
                ret_norm, dir_logit = predictor.model(x)
            prob = float(torch.sigmoid(dir_logit).item())
            if prob >= prob_threshold:
                direction = 1
            elif prob <= 1 - prob_threshold:
                direction = -1
        elif strategy.startswith("BB_TREND"):
            # Trade pullback to BB middle in established trend
            trend_up = ema_ratio[i] > 0.0008
            trend_dn = ema_ratio[i] < -0.0008
            if trend_up and bb_pos[i] < -0.2:
                direction = 1
            elif trend_dn and bb_pos[i] > 0.2:
                direction = -1

        if direction == 0:
            continue

        entry_price = close[i] + spread_price * direction
        if exit_mode == "TPSL":
            tp_dist = 0.6 * atr[i]
            sl_dist = 0.9 * atr[i]
            open_pos = {
                "entry": entry_price, "entry_bar": i, "dir": direction,
                "tp": entry_price + tp_dist * direction,
                "sl": entry_price - sl_dist * direction,
            }
        else:
            # Time exit mode: only protective SL
            sl_dist = 2.0 * atr[i]
            open_pos = {
                "entry": entry_price, "entry_bar": i, "dir": direction,
                "tp": 0, "sl": entry_price - sl_dist * direction,
            }
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
print("M1 SCALP HUNT — Looking for profitable 1-2 min scalping system")
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
        print(f"  saved {len(df)} bars")
    df = pd.read_parquet(parquet)
    features_df = compute_features(df)
    print(f"\n{symbol}: {len(features_df)} bars after features")
    print(f"  spread={SYMBOLS[symbol]['spread_pips']:.1f}p  pip={SYMBOLS[symbol]['pip']}  contract={SYMBOLS[symbol]['contract']}")

    # Train LNN model for this symbol (seq_len=30, the sweet spot from GOLD test)
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

    # Test all strategy x exit combinations
    strategies = [
        "BB_REV", "BB_REV_SESS",
        "RSI_EXT", "RSI_EXT_SESS",
        "PULLBACK", "PULLBACK_SESS",
        "MOM_3BAR", "MOM_3BAR_SESS",
        "BB_TREND", "BB_TREND_SESS",
        "LNN", "LNN_SESS",
    ]
    exits = ["T1", "T2", "T3", "TPSL"]

    print(f"  {'Strategy':<22} {'Exit':<5} {'n':>5} {'WR':>6} {'PF':>6} {'PnL':>9}")
    for strat in strategies:
        for ex in exits:
            r = backtest_strategy(features_df, symbol, strat, ex,
                                   predictor=predictor if strat.startswith("LNN") else None)
            if r["n"] < 5:
                continue
            r["symbol"] = symbol
            r["strategy"] = strat
            r["exit"] = ex
            results.append(r)
            mark = "✓" if r["pf"] > 1.0 else " "
            print(f"  {strat:<22} {ex:<5} {r['n']:>5} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['pnl']:>+9.2f}  {mark}")

client.shutdown()

# Save + top
out = ROOT / "logs" / "m1_scalp_hunt.json"
out.write_text(json.dumps(results, indent=2))

profitable = sorted([r for r in results if r["pf"] > 1.0 and r["n"] >= 20], key=lambda x: x["pnl"], reverse=True)
print("\n" + "=" * 100)
print(f"PROFITABLE (PF>1, n>=20): {len(profitable)} found")
print("=" * 100)
for r in profitable[:15]:
    print(f"  {r['symbol']:<8} {r['strategy']:<22} {r['exit']:<5} n={r['n']:>4} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} PnL=${r['pnl']:+8.2f}")
print(f"\nSaved {out}")
