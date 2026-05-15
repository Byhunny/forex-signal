"""For each TOP 20 strategy, measure the peak distance distribution
(in pips and in ATR units) — tells us where to place TP optimally.

Tracks only WINNING trades (peak exits, not SL).
Realistic spreads applied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from forex_signal.data.features import compute_features
from forex_signal.execution.battle_royale import TOP_10_STRATEGIES
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import EntryConfig, compute_all_signals, evaluate_entry

PIP_VALUE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "USDSEK": 0.0001, "USDCNH": 0.0001,
    "USDJPY": 0.01, "EURJPY": 0.01,
    "GOLD": 0.10, "SILVER": 0.01,
    "BTCUSD": 1.0, "US500Cash": 0.1, "US100Cash": 0.1, "GER40Cash": 1.0,
}

SPREAD_PIPS = {}
try:
    live = json.loads((ROOT / "logs" / "live_spreads.json").read_text())
    for sym, d in live.items():
        if d and "spread_pips" in d:
            SPREAD_PIPS[sym] = round(d["spread_pips"] * 1.2, 1)
except Exception:
    pass
SPREAD_PIPS.setdefault("USDSEK", 165.0)
SPREAD_PIPS.setdefault("USDCNH", 46.0)

SEQ_LEN = 50
SL_ATR_MULTIPLIER = 1.5


def measure_peak_distances(df, predictor, entry_cfg, symbol, spread_pips):
    features = compute_features(df)
    if len(features) <= SEQ_LEN + 10:
        return None
    smc = compute_all_signals(features, entry_cfg)
    f_arr = features[predictor.feature_columns].to_numpy(dtype=np.float32)
    f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds
    high = features["high"].to_numpy(dtype=np.float64)
    low = features["low"].to_numpy(dtype=np.float64)
    close = features["close"].to_numpy(dtype=np.float64)
    atr_arr = features["atr_14"].to_numpy(dtype=np.float64)

    pip = PIP_VALUE[symbol]
    spread_price = spread_pips * pip

    open_pos = None
    peak_distances_pips = []  # for trades that exit at peak
    peak_distances_atr = []   # in ATR units
    sl_distances_pips = []    # for trades that hit SL (in pips)
    avg_atr_pips = []

    device = predictor.device
    model = predictor.model

    for i in range(SEQ_LEN, len(features) - 1):
        window = f_norm[i - SEQ_LEN + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :]).to(device)
        with torch.no_grad():
            ret_norm, dir_logit = model(x)
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        decision = evaluate_entry(smc.iloc[i], prob, entry_cfg)

        if open_pos is not None:
            bar_high = high[i]
            bar_low = low[i]
            if open_pos["dir"] == 1:
                open_pos["peak"] = max(open_pos["peak"], bar_high)
                if bar_low <= open_pos["sl"]:
                    # SL hit
                    sl_dist = (open_pos["entry"] - open_pos["sl"]) / pip
                    sl_distances_pips.append(sl_dist)
                    open_pos = None
                elif decision.direction == -1:
                    # Peak exit
                    peak_dist_price = open_pos["peak"] - open_pos["entry"]
                    peak_distances_pips.append(peak_dist_price / pip)
                    peak_distances_atr.append(peak_dist_price / open_pos["atr"])
                    open_pos = None
            else:  # SELL
                open_pos["peak"] = min(open_pos["peak"], bar_low)
                if bar_high >= open_pos["sl"]:
                    sl_dist = (open_pos["sl"] - open_pos["entry"]) / pip
                    sl_distances_pips.append(sl_dist)
                    open_pos = None
                elif decision.direction == 1:
                    peak_dist_price = open_pos["entry"] - open_pos["peak"]
                    peak_distances_pips.append(peak_dist_price / pip)
                    peak_distances_atr.append(peak_dist_price / open_pos["atr"])
                    open_pos = None

        if open_pos is None and decision.direction != 0 and atr_arr[i] > 0:
            entry = close[i] + spread_price * decision.direction
            sl_distance = max(SL_ATR_MULTIPLIER * atr_arr[i], 4 * spread_price)
            sl_price = entry - sl_distance * decision.direction
            open_pos = {
                "dir": decision.direction,
                "entry": entry,
                "sl": sl_price,
                "peak": close[i],
                "atr": atr_arr[i],
            }
            avg_atr_pips.append(atr_arr[i] / pip)

    if not peak_distances_pips:
        return None

    p = np.array(peak_distances_pips)
    a = np.array(peak_distances_atr)
    return {
        "n_peaks": len(peak_distances_pips),
        "n_sls": len(sl_distances_pips),
        "peak_pip_mean": float(p.mean()),
        "peak_pip_median": float(np.median(p)),
        "peak_pip_p25": float(np.percentile(p, 25)),
        "peak_pip_p75": float(np.percentile(p, 75)),
        "peak_pip_p90": float(np.percentile(p, 90)),
        "peak_atr_mean": float(a.mean()),
        "peak_atr_median": float(np.median(a)),
        "atr_pip_mean": float(np.mean(avg_atr_pips)) if avg_atr_pips else 0,
        "spread_pips": spread_pips,
    }


results = []
print(f"{'Slot':<4} {'Sembol':<10} {'TF':<5} {'Strat':<18} | {'Peak':>4} {'SL':>4} | {'ATR':>5} {'Spread':>6} | {'Peak Pip MEAN':>13} {'MEDIAN':>7} {'P25':>5} {'P75':>5} {'P90':>5} | {'Peak/ATR':>8} {'Önerilen TP':>11}")
print("-"*150)

for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
    parquet = ROOT / "data_cache" / f"{sym}_mt5_{tf}.parquet"
    model_path = ROOT / "models" / model
    if not parquet.exists() or not model_path.exists() or sym not in SPREAD_PIPS:
        continue
    df = pd.read_parquet(parquet)
    pred = Predictor.load(model_path, seq_len=SEQ_LEN)
    entry_cfg = EntryConfig(**ekw)
    r = measure_peak_distances(df, pred, entry_cfg, sym, SPREAD_PIPS[sym])
    if r is None:
        print(f"{name[:2]:<4} {sym:<10} {tf:<5} INSUFFICIENT (0 peaks)")
        continue
    r["slot"] = name[:2]
    r["symbol"] = sym
    r["tf"] = tf
    r["name"] = name
    results.append(r)
    strat = name[3:].replace(sym+"_"+tf+"_", "")[:18]
    # Suggested TP: 60th percentile of peaks (captures majority while leaving room)
    suggested_tp_pips = np.percentile([r["peak_pip_p25"], r["peak_pip_median"], r["peak_pip_p75"]], 50)
    suggested_tp_atr = suggested_tp_pips / r["atr_pip_mean"] if r["atr_pip_mean"] else 0
    print(f"{r['slot']:<4} {sym:<10} {tf:<5} {strat:<18} | {r['n_peaks']:>4} {r['n_sls']:>4} | {r['atr_pip_mean']:>5.1f} {r['spread_pips']:>6.1f} | {r['peak_pip_mean']:>13.0f} {r['peak_pip_median']:>7.0f} {r['peak_pip_p25']:>5.0f} {r['peak_pip_p75']:>5.0f} {r['peak_pip_p90']:>5.0f} | {r['peak_atr_mean']:>8.2f} {r['peak_pip_median']:>7.0f}p={suggested_tp_atr:>3.1f}A", flush=True)
    (ROOT / "logs" / "peak_distance.json").write_text(json.dumps(results, indent=2))

print("\nSaved to logs/peak_distance.json")
