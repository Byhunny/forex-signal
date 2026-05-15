"""Peak-exit backtest for the TOP 20.

Hypothesis: our fixed TP (0.6-1.5 × ATR) might be leaving money on the table.
This backtest answers: "If we could sell at the PEAK reached during a position
(before the opposite signal fires), how profitable would each strategy be?"

Rules:
- Entry: same as normal (spread + slippage adjustment)
- SL: still enforced. If SL hits before opposite signal, exit at SL price.
- TP: REPLACED with peak-tracking. We record MFE (max favorable excursion):
    For BUY: max(high) seen since entry
    For SELL: min(low) seen since entry
- Exit trigger: opposite-direction signal fires on a new bar
    -> close at peak price (best fill achievable in theory)
- This is a PERFECT-HINDSIGHT exit — overstates real-world performance.
  It's a measure of MODEL DIRECTIONAL EDGE, not of executable strategy.

Realistic spreads from logs/live_spreads.json with 20% buffer.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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

# Load live spreads (with 20% buffer)
SPREAD_PIPS = {}
try:
    live = json.loads((ROOT / "logs" / "live_spreads.json").read_text())
    for sym, d in live.items():
        if d and "spread_pips" in d:
            SPREAD_PIPS[sym] = round(d["spread_pips"] * 1.2, 1)
except Exception:
    pass
# Fallback for symbols not in live data
SPREAD_PIPS.setdefault("USDSEK", 165.0)
SPREAD_PIPS.setdefault("USDCNH", 46.0)
SPREAD_PIPS.setdefault("US500Cash", 2.0)
SPREAD_PIPS.setdefault("GER40Cash", 3.0)

# Backtest params
SEQ_LEN = 50
LOT_SIZE = 0.01
SL_ATR_MULTIPLIER = 1.5
SPREAD_MIN_SL_RATIO = 4.0
COMMISSION_PER_LOT = 7.0


def run_peak_backtest(df: pd.DataFrame, predictor: Predictor, entry_cfg: EntryConfig,
                     symbol: str, spread_pips: float) -> dict:
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
    pnl_per_price = LOT_SIZE * contract
    spread_price = spread_pips * pip
    slippage_price = 0.2 * spread_price
    commission = COMMISSION_PER_LOT * LOT_SIZE

    open_pos = None
    trades = []
    device = predictor.device
    model = predictor.model

    for i in range(SEQ_LEN, len(features) - 1):
        # Run model + SMC for this bar
        window = f_norm[i - SEQ_LEN + 1 : i + 1]
        x = torch.from_numpy(window[None, :, :]).to(device)
        with torch.no_grad():
            ret_norm, dir_logit = model(x)
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        decision = evaluate_entry(smc.iloc[i], prob, entry_cfg)

        if open_pos is not None:
            # Update peak / check SL
            bar_high = high[i]
            bar_low = low[i]
            if open_pos["dir"] == 1:
                open_pos["peak"] = max(open_pos["peak"], bar_high)
                if bar_low <= open_pos["sl"]:
                    # SL hit — exit at SL
                    gross = (open_pos["sl"] - open_pos["entry"]) * 1 * pnl_per_price
                    pnl = gross - 2 * commission
                    trades.append({"reason": "sl", "pnl": pnl, "bars": i - open_pos["bar"]})
                    open_pos = None
                elif decision.direction == -1:
                    # Opposite signal — exit at peak
                    gross = (open_pos["peak"] - open_pos["entry"]) * 1 * pnl_per_price
                    pnl = gross - 2 * commission
                    trades.append({"reason": "peak", "pnl": pnl, "bars": i - open_pos["bar"]})
                    open_pos = None
            else:  # SELL
                open_pos["peak"] = min(open_pos["peak"], bar_low)
                if bar_high >= open_pos["sl"]:
                    gross = (open_pos["sl"] - open_pos["entry"]) * -1 * pnl_per_price
                    pnl = gross - 2 * commission
                    trades.append({"reason": "sl", "pnl": pnl, "bars": i - open_pos["bar"]})
                    open_pos = None
                elif decision.direction == 1:
                    gross = (open_pos["peak"] - open_pos["entry"]) * -1 * pnl_per_price
                    pnl = gross - 2 * commission
                    trades.append({"reason": "peak", "pnl": pnl, "bars": i - open_pos["bar"]})
                    open_pos = None

        # Open new if eligible
        if open_pos is None and decision.direction != 0 and atr[i] > 0:
            entry = close[i] + (spread_price + slippage_price) * decision.direction
            sl_distance = max(SL_ATR_MULTIPLIER * atr[i], SPREAD_MIN_SL_RATIO * spread_price)
            sl_price = entry - sl_distance * decision.direction
            open_pos = {
                "dir": decision.direction,
                "entry": entry,
                "sl": sl_price,
                "peak": close[i],  # init peak = entry-ish
                "bar": i,
            }

    # Close any leftover
    if open_pos is not None:
        last_close = close[-1]
        if open_pos["dir"] == 1:
            exit_p = max(open_pos["peak"], last_close)
        else:
            exit_p = min(open_pos["peak"], last_close)
        gross = (exit_p - open_pos["entry"]) * open_pos["dir"] * pnl_per_price
        pnl = gross - 2 * commission
        trades.append({"reason": "eod", "pnl": pnl, "bars": len(features) - 1 - open_pos["bar"]})

    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_bars": 0,
                "tp_count": 0, "sl_count": 0, "avg_winner": 0, "avg_loser": 0}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 99.99
    sl_count = sum(1 for t in trades if t["reason"] == "sl")
    peak_count = sum(1 for t in trades if t["reason"] == "peak")

    days = max((pd.to_datetime(times[-1]) - pd.to_datetime(times[SEQ_LEN])).total_seconds() / 86400.0, 1.0)

    return {
        "n": len(trades),
        "td": round(len(trades) / days, 2),
        "wr": round(len(wins) / len(trades), 3),
        "pf": round(pf, 3),
        "pnl": round(float(pnls.sum()), 2),
        "avg_bars": round(float(np.mean([t["bars"] for t in trades])), 1),
        "peak_count": peak_count,
        "sl_count": sl_count,
        "avg_winner": round(float(wins.mean()) if len(wins) else 0, 2),
        "avg_loser": round(float(losses.mean()) if len(losses) else 0, 2),
    }


# Main
results = []
print(f"Loaded spreads for {len(SPREAD_PIPS)} symbols")
print(f"\n{'Slot':<3} {'Sembol':<10} {'TF':<5} {'Strat':<20} | {'n':>4} {'/day':>5} {'WR':>5} {'PF':>6} {'PnL':>10} {'AvgBars':>7} {'Peak':>5} {'SL':>4}")
print("-"*110)

for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
    parquet = ROOT / "data_cache" / f"{sym}_mt5_{tf}.parquet"
    model_path = ROOT / "models" / model
    if not parquet.exists() or not model_path.exists() or sym not in SPREAD_PIPS:
        print(f"{name} MISSING")
        continue
    df = pd.read_parquet(parquet)
    pred = Predictor.load(model_path, seq_len=SEQ_LEN)
    entry_cfg = EntryConfig(**ekw)
    r = run_peak_backtest(df, pred, entry_cfg, sym, SPREAD_PIPS[sym])
    if r is None:
        print(f"{name} INSUFFICIENT_DATA")
        continue
    r["slot"] = name[:2]
    r["symbol"] = sym
    r["tf"] = tf
    r["name"] = name
    r["spread_pips"] = SPREAD_PIPS[sym]
    results.append(r)
    strat = name[3:].split("_", 2)[-1]
    print(f"{r['slot']:<3} {sym:<10} {tf:<5} {strat:<20} | {r['n']:>4d} {r['td']:>5.2f} {r['wr']:>5.3f} {r['pf']:>6.2f} {r['pnl']:>+10.2f} {r['avg_bars']:>7.1f} {r['peak_count']:>5d} {r['sl_count']:>4d}", flush=True)
    # Incremental save
    (ROOT / "logs" / "peak_backtest_top20.json").write_text(json.dumps(results, indent=2))

# Markdown
md = ROOT / "docs" / "peak_backtest_top20.md"
lines = ["# Peak-Exit Backtest — TOP 20 (theoretical max)\n",
         f"Generated: {pd.Timestamp.utcnow().isoformat()}\n",
         "**Exit logic:** position closes at the PEAK price achieved between entry and the opposing signal. SL still enforced. Realistic spreads applied.\n",
         "**Caveat:** Perfect-hindsight exit — this is the model's directional edge ceiling, NOT a tradable strategy.\n",
         "\n## Results\n",
         "| Slot | Symbol | TF | Strategy | Trades | /day | WR | PF | PnL | Avg bars | Peak exits | SL exits | Spread |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(results, key=lambda x: x["pf"], reverse=True):
    strat = r["name"][3:].replace("_", " ")
    lines.append(f"| {r['slot']} | {r['symbol']} | {r['tf']} | {strat} | {r['n']} | {r['td']} | {r['wr']:.3f} | {r['pf']:.2f} | ${r['pnl']:+.2f} | {r['avg_bars']} | {r['peak_count']} | {r['sl_count']} | {r['spread_pips']} |")
md.write_text("\n".join(lines))
print(f"\nMD saved to {md}")
