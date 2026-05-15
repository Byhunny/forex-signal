"""Compare 3 exit modes for TOP 20 strategies:
- PEAK: hindsight — exit at peak achieved between entry and opposite signal
- CLOSE_AT_REVERSE: realistic — exit at the close price of the bar that fires opposite signal
- TRAILING (1.5×ATR): realistic — trail stop at peak - 1.5×ATR; exit when retraced

All modes: SL still enforced at 1.5×ATR (with spread-aware floor).
All use realistic live spreads.
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

SEQ_LEN = 50
LOT = 0.01
SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.5
COMMISSION = 7.0 * LOT


def run_modes(df, predictor, entry_cfg, symbol, spread_pips):
    """Returns dict with results for each of 3 modes."""
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
    times = pd.to_datetime(features["time"]).astype(str).to_numpy()

    pip = PIP_VALUE[symbol]
    contract = CONTRACT[symbol]
    pnl_per_price = LOT * contract
    spread_price = spread_pips * pip
    slippage_price = 0.2 * spread_price

    device = predictor.device
    model = predictor.model

    # Pre-compute all decisions once
    decisions = []
    for i in range(SEQ_LEN, len(features) - 1):
        window = f_norm[i - SEQ_LEN + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :]).to(device)
        with torch.no_grad():
            ret_norm, dir_logit = model(x)
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        decision = evaluate_entry(smc.iloc[i], prob, entry_cfg)
        decisions.append(decision)

    # For each mode, run the position walk
    results_by_mode = {}
    for mode in ("peak", "close_at_reverse", "trailing"):
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
                    # SL check
                    if bar_low <= open_pos["sl"]:
                        exit_price = open_pos["sl"]
                        reason = "sl"
                    # Mode-specific exits
                    elif mode == "trailing":
                        trail_stop = open_pos["peak"] - TRAIL_ATR_MULT * open_pos["atr"]
                        if trail_stop > open_pos["sl"] and bar_low <= trail_stop:
                            exit_price = trail_stop
                            reason = "trail"
                    if exit_price is None and decision.direction == -1:
                        if mode == "peak":
                            exit_price = open_pos["peak"]
                            reason = "peak"
                        elif mode == "close_at_reverse":
                            exit_price = close[i]
                            reason = "reverse"
                        elif mode == "trailing":
                            # Already covered above if trailing hit; otherwise wait
                            pass
                else:  # SELL
                    open_pos["peak"] = min(open_pos["peak"], bar_low)
                    if bar_high >= open_pos["sl"]:
                        exit_price = open_pos["sl"]
                        reason = "sl"
                    elif mode == "trailing":
                        trail_stop = open_pos["peak"] + TRAIL_ATR_MULT * open_pos["atr"]
                        if trail_stop < open_pos["sl"] and bar_high >= trail_stop:
                            exit_price = trail_stop
                            reason = "trail"
                    if exit_price is None and decision.direction == 1:
                        if mode == "peak":
                            exit_price = open_pos["peak"]
                            reason = "peak"
                        elif mode == "close_at_reverse":
                            exit_price = close[i]
                            reason = "reverse"

                if exit_price is not None:
                    gross = (exit_price - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
                    pnl = gross - 2 * COMMISSION
                    trades.append({"reason": reason, "pnl": pnl})
                    open_pos = None

            if open_pos is None and decision.direction != 0 and atr[i] > 0:
                entry = close[i] + (spread_price + slippage_price) * decision.direction
                sl_distance = max(SL_ATR_MULT * atr[i], 4 * spread_price)
                sl_price = entry - sl_distance * decision.direction
                open_pos = {
                    "dir": decision.direction,
                    "entry": entry,
                    "sl": sl_price,
                    "peak": close[i],
                    "atr": atr[i],
                }

        if not trades:
            results_by_mode[mode] = {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "sl_count": 0, "exit_count": 0}
            continue
        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 99.99
        sl_count = sum(1 for t in trades if t["reason"] == "sl")
        exit_count = len(trades) - sl_count
        results_by_mode[mode] = {
            "n": len(trades),
            "wr": round(len(wins) / len(trades), 3),
            "pf": round(pf, 3),
            "pnl": round(float(pnls.sum()), 2),
            "sl_count": sl_count,
            "exit_count": exit_count,
        }

    return results_by_mode


# Main
print(f"{'Slot':<4} {'Sembol':<10} {'TF':<5} | {'PEAK':<22} | {'CLOSE_AT_REV':<22} | {'TRAILING 1.5xATR':<22}")
print(f"{'':4} {'':10} {'':5} | {'PF':>5} {'WR':>5} {'PnL':>9} | {'PF':>5} {'WR':>5} {'PnL':>9} | {'PF':>5} {'WR':>5} {'PnL':>9}")
print("-"*130)

all_results = []
for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
    parquet = ROOT / "data_cache" / f"{sym}_mt5_{tf}.parquet"
    model_path = ROOT / "models" / model
    if not parquet.exists() or not model_path.exists() or sym not in SPREAD_PIPS:
        continue
    df = pd.read_parquet(parquet)
    pred = Predictor.load(model_path, seq_len=SEQ_LEN)
    entry_cfg = EntryConfig(**ekw)
    r = run_modes(df, pred, entry_cfg, sym, SPREAD_PIPS[sym])
    if r is None:
        continue
    slot = name[:2]
    all_results.append({"slot": slot, "symbol": sym, "tf": tf, "name": name, **r})
    p, c, t = r["peak"], r["close_at_reverse"], r["trailing"]
    print(f"{slot:<4} {sym:<10} {tf:<5} | {p['pf']:>5.2f} {p['wr']:>5.3f} {p['pnl']:>+9.0f} | {c['pf']:>5.2f} {c['wr']:>5.3f} {c['pnl']:>+9.0f} | {t['pf']:>5.2f} {t['wr']:>5.3f} {t['pnl']:>+9.0f}", flush=True)
    (ROOT / "logs" / "exit_modes.json").write_text(json.dumps(all_results, indent=2, default=str))

print("\nSaved to logs/exit_modes.json")
