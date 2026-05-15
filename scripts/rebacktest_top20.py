"""Re-backtest the top 20 battle royale strategies with REALISTIC spreads
and spread-aware TP/SL logic. Compare new vs original sweep numbers.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forex_signal.backtest.walk_forward import BacktestConfig, run_backtest
from forex_signal.execution.battle_royale import TOP_10_STRATEGIES
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import EntryConfig

# REALISTIC spreads (updated)
SPREAD_PIPS = {
    "EURUSD": 1.2, "GBPUSD": 2.0, "USDJPY": 1.5, "USDCHF": 2.5,
    "USDCAD": 3.0, "AUDUSD": 2.0, "NZDUSD": 3.0,
    "EURJPY": 2.5, "EURGBP": 1.8, "EURAUD": 3.0, "EURCAD": 4.0,
    "USDCNH": 10.0, "USDSEK": 15.0,
    "GOLD": 35.0, "SILVER": 4.0,
    "BTCUSD": 60.0, "US500Cash": 1.5, "US100Cash": 3.0,
    "GER40Cash": 3.0, "US30Cash": 3.5,
}
PIP_VALUE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "USDSEK": 0.0001, "USDCNH": 0.0001,
    "USDJPY": 0.01, "EURJPY": 0.01,
    "GOLD": 0.10, "SILVER": 0.01,
    "BTCUSD": 1.0,
    "US500Cash": 0.1, "US100Cash": 0.1, "GER40Cash": 1.0, "US30Cash": 1.0,
}
CONTRACT = {
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "USDCAD": 100_000,
    "NZDUSD": 100_000, "USDCHF": 100_000, "EURGBP": 100_000, "EURAUD": 100_000,
    "EURCAD": 100_000, "USDSEK": 100_000, "USDCNH": 100_000,
    "USDJPY": 100_000, "EURJPY": 100_000,
    "GOLD": 100, "SILVER": 5000,
    "BTCUSD": 1.0, "US500Cash": 1.0, "US100Cash": 1.0, "GER40Cash": 1.0, "US30Cash": 1.0,
}

SYMBOLS_FILTER = None  # test all 20 strategies

# Load LIVE spreads from MT5 snapshot (run scripts/fetch_spreads.py to refresh)
try:
    live = json.loads((ROOT / "logs" / "live_spreads.json").read_text())
    for sym, d in live.items():
        if d and "spread_pips" in d:
            SPREAD_PIPS[sym] = round(d["spread_pips"] * 1.2, 1)  # 20% buffer
    print(f"loaded live spreads for {len(live)} symbols")
except Exception as e:
    print(f"(no live spreads loaded: {e})")

results = []
for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
    if SYMBOLS_FILTER is not None and sym not in SYMBOLS_FILTER:
        continue
    parquet = ROOT / "data_cache" / f"{sym}_mt5_{tf}.parquet"
    model_path = ROOT / "models" / model
    if not parquet.exists() or not model_path.exists():
        print(f"{name:40s} MISSING ({parquet.exists()=} {model_path.exists()=})")
        continue
    df = pd.read_parquet(parquet)
    pred = Predictor.load(model_path, seq_len=50)
    entry_cfg = EntryConfig(**ekw)
    bc = BacktestConfig(
        seq_len=50,
        initial_balance=10_000.0,
        lot_size=0.01,
        spread_pips=SPREAD_PIPS[sym],
        slippage_pips=SPREAD_PIPS[sym] * 0.2,
        commission_per_lot=7.0,
        sl_atr_multiplier=1.5,
        tp_atr_min_multiplier=0.6,
        tp_atr_max_multiplier=1.5,
        pip_value=PIP_VALUE[sym],
        contract_size=CONTRACT[sym],
        entry=entry_cfg,
        spread_min_tp_ratio=2.5,
        spread_min_sl_ratio=4.0,
        spread_skip_ratio=0.45,
    )
    bt = run_backtest(df, pred, bc)
    row = {
        "slot": name[:2], "symbol": sym, "tf": tf, "name": name, "n": bt.n_trades,
        "td": round(bt.trades_per_day, 2), "wr": round(bt.win_rate, 3),
        "pf": round(bt.profit_factor, 3) if bt.profit_factor != float("inf") else 99.99,
        "pnl": round(bt.total_pnl, 2), "dd": round(bt.max_drawdown_pct, 2),
        "spread_pips": SPREAD_PIPS[sym],
    }
    results.append(row)
    print(f"{name:40s} n={bt.n_trades:4d}/day={bt.trades_per_day:5.2f}  WR={bt.win_rate:.3f}  PF={row['pf']:6.2f}  PnL=${bt.total_pnl:+8.2f}  DD={bt.max_drawdown_pct:.2f}%", flush=True)
    # Save incrementally after each result
    (ROOT / "logs" / "rebacktest_top20.json").write_text(json.dumps(results, indent=2))

# Save + compare
df_new = pd.DataFrame(results)
out = ROOT / "logs" / "rebacktest_top20.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nsaved to {out}")

# Verdict
df_new["status"] = df_new.apply(lambda r: "KEEP" if r["pf"] >= 1.0 and r["n"] >= 20 else
                                          ("MARGINAL" if r["pf"] >= 0.9 else "DROP"), axis=1)
# Write to MD too
md_out = ROOT / "docs" / "rebacktest_top20.md"
lines = ["# TOP 20 Rebacktest (Realistic Spreads + Spread-Aware TP/SL)\n",
         f"Run: {pd.Timestamp.utcnow().isoformat()}\n",
         "\n## Verdict\n",
         "| Slot | Symbol | TF | Strategy | Trades | /day | WR | PF | P&L | DD% | Status |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df_new.iterrows():
    strat_short = next((n[3:] for n,s,t,_,_,_ in TOP_10_STRATEGIES if n.startswith(r['slot'])), "?")
    lines.append(f"| {r['slot']} | {r['symbol']} | {r['tf']} | {strat_short} | {r['n']} | {r['td']} | {r['wr']:.3f} | {r['pf']:.2f} | ${r['pnl']:+.2f} | {r['dd']:.1f} | {r['status']} |")
md_out.write_text("\n".join(lines))
print(f"\nMD report saved to {md_out}")
