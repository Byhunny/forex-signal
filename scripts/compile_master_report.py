"""Combine all backtest results into a single master markdown report."""
import json
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]

# Load all data sources
sweep_v1 = json.load(open(ROOT / "logs" / "sweep_results.json"))
rebt = json.load(open(ROOT / "logs" / "rebacktest_top20.json"))
peak = json.load(open(ROOT / "logs" / "peak_backtest_top20.json"))
peak_dist = json.load(open(ROOT / "logs" / "peak_distance.json"))
exit_v1 = json.load(open(ROOT / "logs" / "exit_modes.json"))
exit_v2 = json.load(open(ROOT / "logs" / "exit_modes_v2.json"))
live_spreads = json.load(open(ROOT / "logs" / "live_spreads.json"))

# TOP 20 mapping
TOP_20 = [
    ("01","USDSEK","M30","lnn_very_strong"), ("02","USDSEK","M30","lnn_strong"),
    ("03","SILVER","M15","trend_lnn"), ("04","USDCNH","M30","lnn_very_strong"),
    ("05","SILVER","M15","lnn_strong"), ("06","EURJPY","M30","smc_strict"),
    ("07","EURUSD","M30","smc_strict"), ("08","AUDUSD","M30","smc_strict"),
    ("09","EURCAD","M30","lnn_strong"), ("10","US100Cash","M30","trend_lnn"),
    ("11","USDSEK","M5","lnn_very_strong"), ("12","USDSEK","M30","trend_lnn"),
    ("13","USDSEK","M5","lnn_strong"), ("14","SILVER","M5","trend_lnn"),
    ("15","SILVER","M30","trend_lnn"), ("16","EURJPY","M15","trend_lnn"),
    ("17","BTCUSD","M30","trend_lnn"), ("18","BTCUSD","M15","trend_lnn"),
    ("19","GOLD","M15","trend_lnn"), ("20","USDSEK","M15","trend_lnn"),
]

# Helper lookups
def by_slot(data_list, slot):
    for r in data_list:
        if r.get("slot") == slot:
            return r
    return None

def find_sweep(sym, tf, strat):
    for r in sweep_v1:
        if r["symbol"]==sym and r["timeframe"]==tf and r["strategy"]==strat:
            return r
    return None

out = []
out.append("# 📊 Master Backtest Report — All Results\n")
out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

out.append("\n## Test Setup\n")
out.append("- 20 strategies from original sweep (TOP 20 battle royale lineup)")
out.append("- Realistic live spreads from MT5 with 20% safety buffer")
out.append("- Spread-aware SL widening (max(1.5×ATR, 4×spread))")
out.append("- Single spread charge (bug fixed)")
out.append("- 0.01 lot, $7/lot commission")
out.append("- Bar data: 50k for USDSEK M30, 15k for others\n")

# Live spreads table
out.append("\n## Live Spreads (MT5 snapshot, +20% buffer)\n")
out.append("| Symbol | Live Spread (pips) |")
out.append("|---|---|")
syms_seen = set()
for s, t, _, _ in [(t[1],t[2],None,None) for t in TOP_20]:
    if s not in syms_seen:
        syms_seen.add(s)
        sp = live_spreads.get(s)
        if sp and "spread_pips" in sp:
            out.append(f"| {s} | {sp['spread_pips']:.1f} (×1.2 = {sp['spread_pips']*1.2:.1f}) |")

# Big master comparison table
out.append("\n## Master Comparison Table (all 20 strategies, all 7 metrics)\n")
out.append("Legend:")
out.append("- **OLD_SWEEP**: original sweep with optimistic 8-pip USDSEK spread + double-spread bug")
out.append("- **NORM_TP**: standard TP (0.6-1.5×ATR), realistic spread, bug fixed")
out.append("- **PEAK**: hindsight peak exit (theoretical ceiling)")
out.append("- **CLOSE**: hold until opposite signal, exit at close price (realistic)")
out.append("- **HALF-TP**: TP at half of median peak distance")
out.append("- **TRAIL 1.5**: trailing stop at 1.5×ATR from peak")
out.append("- **TRAIL 1.2**: trailing stop at 1.2×ATR from peak (tighter)\n")

out.append("| # | Symbol | TF | Strategy | OLD_SWEEP P&L | NORM_TP P&L | PEAK P&L | **CLOSE P&L** | HALF-TP P&L | TRAIL_1.5 P&L | TRAIL_1.2 P&L |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for slot, sym, tf, strat in TOP_20:
    old = find_sweep(sym, tf, strat)
    n = by_slot(rebt, slot)
    p = by_slot(peak, slot)
    e1 = by_slot(exit_v1, slot)
    e2 = by_slot(exit_v2, slot)
    old_pnl = f"${old['total_pnl']:+.0f}" if old else "-"
    norm_pnl = f"${n['pnl']:+.0f}" if n else "-"
    peak_pnl = f"${p['pnl']:+.0f}" if p else "-"
    close_pnl = f"${e1['close_at_reverse']['pnl']:+.0f}" if e1 else "-"
    half_pnl = f"${e2['half_peak_tp']['pnl']:+.0f}" if e2 and 'half_peak_tp' in e2 else "-"
    trail15 = f"${e1['trailing']['pnl']:+.0f}" if e1 else "-"
    trail12 = f"${e2['trail_1.2']['pnl']:+.0f}" if e2 and 'trail_1.2' in e2 else "-"
    out.append(f"| {slot} | {sym} | {tf} | {strat} | {old_pnl} | {norm_pnl} | {peak_pnl} | **{close_pnl}** | {half_pnl} | {trail15} | {trail12} |")

# PF table
out.append("\n## PF Comparison (Profit Factor — higher is better, >1 means profitable)\n")
out.append("| # | Symbol | TF | OLD_SWEEP | NORM_TP | PEAK | **CLOSE** | HALF-TP | TRAIL_1.5 | TRAIL_1.2 |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for slot, sym, tf, strat in TOP_20:
    old = find_sweep(sym, tf, strat)
    n = by_slot(rebt, slot)
    p = by_slot(peak, slot)
    e1 = by_slot(exit_v1, slot)
    e2 = by_slot(exit_v2, slot)
    old_pf = f"{min(old['profit_factor'],99):.2f}" if old else "-"
    out.append(f"| {slot} | {sym} | {tf} | {old_pf} | "
               f"{n['pf']:.2f} | {p['pf']:.2f} | **{e1['close_at_reverse']['pf']:.2f}** | "
               f"{e2['half_peak_tp']['pf']:.2f} | {e1['trailing']['pf']:.2f} | {e2['trail_1.2']['pf']:.2f} |")

# WR table
out.append("\n## WR Comparison (Win Rate %)\n")
out.append("| # | Symbol | TF | OLD_SWEEP | NORM_TP | PEAK | **CLOSE** | HALF-TP | TRAIL_1.5 | TRAIL_1.2 |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for slot, sym, tf, strat in TOP_20:
    old = find_sweep(sym, tf, strat)
    n = by_slot(rebt, slot)
    p = by_slot(peak, slot)
    e1 = by_slot(exit_v1, slot)
    e2 = by_slot(exit_v2, slot)
    out.append(f"| {slot} | {sym} | {tf} | "
               f"{old['win_rate']*100:.1f}% | "
               f"{n['wr']*100:.1f}% | {p['wr']*100:.1f}% | **{e1['close_at_reverse']['wr']*100:.1f}%** | "
               f"{e2['half_peak_tp']['wr']*100:.1f}% | {e1['trailing']['wr']*100:.1f}% | {e2['trail_1.2']['wr']*100:.1f}% |")

# Peak distance distribution table
out.append("\n## Peak Distance Distribution (per strategy, in pips)\n")
out.append("Tells us where TP could be placed. Median peak / 2 used for HALF-TP test.\n")
out.append("| # | Symbol | TF | Peaks | SLs | ATR (pip) | Spread | Peak Mean | Median | P25 | P75 | P90 | Peak/ATR |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in peak_dist:
    out.append(f"| {r['slot']} | {r['symbol']} | {r['tf']} | {r['n_peaks']} | {r['n_sls']} | "
               f"{r['atr_pip_mean']:.1f} | {r['spread_pips']:.1f} | "
               f"{r['peak_pip_mean']:.0f} | {r['peak_pip_median']:.0f} | "
               f"{r['peak_pip_p25']:.0f} | {r['peak_pip_p75']:.0f} | {r['peak_pip_p90']:.0f} | "
               f"{r['peak_atr_mean']:.2f}× |")

# Totals & verdict
out.append("\n## Totals (all 20 strategies combined)\n")
out.append("| Mode | Total P&L | Profitable count | Description |")
out.append("|---|---|---|---|")
totals = {
    "OLD_SWEEP (original)": (sum(r["total_pnl"] for r in sweep_v1 if any(r["symbol"]==sym and r["timeframe"]==tf and r["strategy"]==strat for _,sym,tf,strat in TOP_20)),
                              sum(1 for r in sweep_v1 if any(r["symbol"]==sym and r["timeframe"]==tf and r["strategy"]==strat for _,sym,tf,strat in TOP_20) and r["profit_factor"] > 1)),
    "NORM_TP (realistic spread)": (sum(r["pnl"] for r in rebt), sum(1 for r in rebt if r["pf"] > 1)),
    "PEAK (hindsight)": (sum(r["pnl"] for r in peak), sum(1 for r in peak if r["pf"] > 1)),
    "**CLOSE_AT_REVERSE**": (sum(r["close_at_reverse"]["pnl"] for r in exit_v1), sum(1 for r in exit_v1 if r["close_at_reverse"]["pf"] > 1)),
    "HALF-TP (median/2)": (sum(r["half_peak_tp"]["pnl"] for r in exit_v2), sum(1 for r in exit_v2 if r["half_peak_tp"]["pf"] > 1)),
    "TRAIL 1.5×ATR": (sum(r["trailing"]["pnl"] for r in exit_v1), sum(1 for r in exit_v1 if r["trailing"]["pf"] > 1)),
    "TRAIL 1.2×ATR": (sum(r["trail_1.2"]["pnl"] for r in exit_v2), sum(1 for r in exit_v2 if r["trail_1.2"]["pf"] > 1)),
}
for mode, (pnl, count) in totals.items():
    desc = ""
    if "OLD" in mode: desc = "with bug + optimistic spreads"
    elif "NORM_TP" in mode: desc = "realistic but tight TP"
    elif "PEAK" in mode: desc = "hindsight ceiling"
    elif "CLOSE" in mode: desc = "**recommended — most realistic kazanan**"
    elif "HALF-TP" in mode: desc = "TP-based, between close and norm"
    elif "TRAIL 1.5" in mode: desc = "trailing too tight"
    elif "TRAIL 1.2" in mode: desc = "trailing even tighter — worse"
    out.append(f"| {mode} | ${pnl:+,.0f} | {count}/20 | {desc} |")

out.append("\n## Conclusions\n")
out.append("1. **CLOSE_AT_REVERSE is the practical winner** — 11/20 profitable, total +$7,490")
out.append("2. **Original sweep was misleading** — double-spread bug + optimistic spreads showed false hopes")
out.append("3. **USDSEK is untrade-able** — 137 pip live spread destroys edge regardless of exit mode")
out.append("4. **Trailing stops too tight** at both 1.2 and 1.5 × ATR — volatility hits trail before move develops")
out.append("5. **Peak (hindsight)** shows model has REAL directional edge — TP placement is the harvest mechanism issue")
out.append("6. **Half-peak TP** captures portion of peak but underperforms close-at-reverse on most strategies")

out.append("\n## ✅ Recommended Battle Royale Lineup (11 profitable in CLOSE mode)\n")
out.append("| Rank | Slot | Symbol | TF | PF | Test P&L |")
out.append("|---|---|---|---|---|---|")
winners = [(r["slot"], r["symbol"], r["tf"], r["close_at_reverse"]["pf"], r["close_at_reverse"]["pnl"]) for r in exit_v1 if r["close_at_reverse"]["pf"] > 1]
winners.sort(key=lambda x: x[4], reverse=True)
for i, (slot, sym, tf, pf, pnl) in enumerate(winners, 1):
    out.append(f"| {i} | {slot} | {sym} | {tf} | {pf:.2f} | ${pnl:+,.0f} |")

# Write
out_path = ROOT / "docs" / "ALL_BACKTEST_RESULTS.md"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("\n".join(out))
print(f"Master report written to {out_path}")
print(f"Size: {out_path.stat().st_size} bytes")
