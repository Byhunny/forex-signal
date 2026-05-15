"""Two more exit modes:
- HALF_PEAK_TP: TP at HALF of median peak (per strategy, from peak_distance.json)
- TRAIL_1.2: trailing stop at 1.2×ATR (tighter than 1.5)

Plus baseline modes for comparison: PEAK (hindsight) and CLOSE_AT_REVERSE.
Realistic spreads.
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
CONTRACT = {
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "USDCAD": 100_000,
    "NZDUSD": 100_000, "USDCHF": 100_000, "EURGBP": 100_000, "EURAUD": 100_000,
    "EURCAD": 100_000, "USDSEK": 100_000, "USDCNH": 100_000,
    "USDJPY": 100_000, "EURJPY": 100_000,
    "GOLD": 100, "SILVER": 5000,
    "BTCUSD": 1.0, "US500Cash": 1.0, "US100Cash": 1.0, "GER40Cash": 1.0,
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

# Load median peak distances per slot from peak_distance.json
peak_dist = {}
try:
    pd_data = json.loads((ROOT / "logs" / "peak_distance.json").read_text())
    for r in pd_data:
        peak_dist[r["slot"]] = r["peak_pip_median"]
except Exception:
    pass

SEQ_LEN = 50
LOT = 0.01
SL_ATR_MULT = 1.5
COMMISSION = 7.0 * LOT


def run_modes(df, predictor, entry_cfg, symbol, spread_pips, half_peak_tp_pips):
    features = compute_features(df)
    if len(features) <= SEQ_LEN + 10:
        return None

    smc = compute_all_signals(features, entry_cfg)
    f_arr = features[predictor.feature_columns].to_numpy(dtype=np.float32)
    f_norm = (f_arr - predictor.feature_means) / predictor.feature_stds
    close = features["close"].to_numpy(dtype=np.float64)
    high = features["high"].to_numpy(dtype=np.float64)
    low = features["low"].to_numpy(dtype=np.float64)
    atr = features["atr_14"].to_numpy(dtype=np.float64)

    pip = PIP_VALUE[symbol]
    contract = CONTRACT[symbol]
    pnl_per_price = LOT * contract
    spread_price = spread_pips * pip
    slippage_price = 0.2 * spread_price
    half_peak_tp_price = half_peak_tp_pips * pip if half_peak_tp_pips else None

    device = predictor.device
    model = predictor.model

    # Pre-compute all decisions
    decisions = []
    for i in range(SEQ_LEN, len(features) - 1):
        window = f_norm[i - SEQ_LEN + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :]).to(device)
        with torch.no_grad():
            ret_norm, dir_logit = model(x)
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        decision = evaluate_entry(smc.iloc[i], prob, entry_cfg)
        decisions.append(decision)

    modes = ["peak", "close_at_reverse", "half_peak_tp", "trail_1.2"]
    results_by_mode = {}
    for mode in modes:
        if mode == "half_peak_tp" and half_peak_tp_price is None:
            results_by_mode[mode] = {"n": 0, "wr": 0, "pf": 0, "pnl": 0,
                                     "sl_count": 0, "tp_count": 0, "reverse_count": 0}
            continue

        open_pos = None
        trades = []
        for offset, decision in enumerate(decisions):
            i = SEQ_LEN + offset
            if open_pos is not None:
                bar_high = high[i]
                bar_low = low[i]
                exit_price = None
                reason = None

                if open_pos["dir"] == 1:
                    open_pos["peak"] = max(open_pos["peak"], bar_high)
                    # 1) SL check (always)
                    if bar_low <= open_pos["sl"]:
                        exit_price = open_pos["sl"]
                        reason = "sl"
                    # 2) Mode-specific TP / trail
                    elif mode == "half_peak_tp":
                        tp_level = open_pos["entry"] + half_peak_tp_price
                        if bar_high >= tp_level:
                            exit_price = tp_level
                            reason = "tp"
                    elif mode == "trail_1.2":
                        trail_stop = open_pos["peak"] - 1.2 * open_pos["atr"]
                        if trail_stop > open_pos["sl"] and bar_low <= trail_stop:
                            exit_price = trail_stop
                            reason = "trail"
                    # 3) Opposite signal
                    if exit_price is None and decision.direction == -1:
                        if mode == "peak":
                            exit_price = open_pos["peak"]; reason = "peak"
                        elif mode == "close_at_reverse":
                            exit_price = close[i]; reason = "reverse"
                        elif mode == "half_peak_tp":
                            exit_price = close[i]; reason = "reverse"
                        elif mode == "trail_1.2":
                            exit_price = close[i]; reason = "reverse"
                else:  # SELL
                    open_pos["peak"] = min(open_pos["peak"], bar_low)
                    if bar_high >= open_pos["sl"]:
                        exit_price = open_pos["sl"]; reason = "sl"
                    elif mode == "half_peak_tp":
                        tp_level = open_pos["entry"] - half_peak_tp_price
                        if bar_low <= tp_level:
                            exit_price = tp_level; reason = "tp"
                    elif mode == "trail_1.2":
                        trail_stop = open_pos["peak"] + 1.2 * open_pos["atr"]
                        if trail_stop < open_pos["sl"] and bar_high >= trail_stop:
                            exit_price = trail_stop; reason = "trail"
                    if exit_price is None and decision.direction == 1:
                        if mode == "peak":
                            exit_price = open_pos["peak"]; reason = "peak"
                        elif mode == "close_at_reverse":
                            exit_price = close[i]; reason = "reverse"
                        elif mode == "half_peak_tp":
                            exit_price = close[i]; reason = "reverse"
                        elif mode == "trail_1.2":
                            exit_price = close[i]; reason = "reverse"

                if exit_price is not None:
                    gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                    pnl = gross - 2 * COMMISSION
                    trades.append({"reason": reason, "pnl": pnl})
                    open_pos = None

            if open_pos is None and decision.direction != 0 and atr[i] > 0:
                entry = close[i] + (spread_price + slippage_price) * decision.direction
                sl_distance = max(SL_ATR_MULT * atr[i], 4 * spread_price)
                sl_price = entry - sl_distance * decision.direction
                open_pos = {"dir": decision.direction, "entry": entry, "sl": sl_price,
                            "peak": close[i], "atr": atr[i]}

        if not trades:
            results_by_mode[mode] = {"n": 0, "wr": 0, "pf": 0, "pnl": 0,
                                     "sl_count": 0, "tp_count": 0, "reverse_count": 0}
            continue
        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 99.99
        results_by_mode[mode] = {
            "n": len(trades),
            "wr": round(len(wins) / len(trades), 3),
            "pf": round(pf, 3),
            "pnl": round(float(pnls.sum()), 2),
            "sl_count": sum(1 for t in trades if t["reason"] == "sl"),
            "tp_count": sum(1 for t in trades if t["reason"] == "tp"),
            "trail_count": sum(1 for t in trades if t["reason"] == "trail"),
            "peak_count": sum(1 for t in trades if t["reason"] == "peak"),
            "reverse_count": sum(1 for t in trades if t["reason"] == "reverse"),
        }
    return results_by_mode


print(f"{'#':<3} {'Symbol':<10} {'TF':<5} | {'HalfPeak TP (pip)':<18} | {'PEAK':<18} {'CLOSE':<18} {'HALF-TP':<18} {'TRAIL 1.2':<18}")
print(f"{'':3} {'':10} {'':5} | {'':18} | {'PF':>5} {'PnL':>10} {'PF':>5} {'PnL':>10} {'PF':>5} {'PnL':>10} {'PF':>5} {'PnL':>10}")
print("="*150)

all_results = []
for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
    parquet = ROOT / "data_cache" / f"{sym}_mt5_{tf}.parquet"
    model_path = ROOT / "models" / model
    if not parquet.exists() or not model_path.exists() or sym not in SPREAD_PIPS:
        continue
    df = pd.read_parquet(parquet)
    pred = Predictor.load(model_path, seq_len=SEQ_LEN)
    entry_cfg = EntryConfig(**ekw)
    slot = name[:2]
    half_tp = peak_dist.get(slot, 0) / 2 if peak_dist.get(slot) else 0
    r = run_modes(df, pred, entry_cfg, sym, SPREAD_PIPS[sym], half_tp)
    if r is None:
        continue
    all_results.append({"slot": slot, "symbol": sym, "tf": tf, "name": name,
                       "half_peak_tp_pips": half_tp, **r})
    p = r["peak"]; c = r["close_at_reverse"]; h = r["half_peak_tp"]; t = r["trail_1.2"]
    print(f"{slot:<3} {sym:<10} {tf:<5} | TP={half_tp:>6.0f}p (median/2) | "
          f"{p['pf']:>5.2f} {p['pnl']:>+10.0f} {c['pf']:>5.2f} {c['pnl']:>+10.0f} "
          f"{h['pf']:>5.2f} {h['pnl']:>+10.0f} {t['pf']:>5.2f} {t['pnl']:>+10.0f}", flush=True)
    (ROOT / "logs" / "exit_modes_v2.json").write_text(json.dumps(all_results, indent=2, default=str))

print("\nSaved logs/exit_modes_v2.json")
