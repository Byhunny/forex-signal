"""Render sweep results JSON -> markdown. Use SWEEP_FILE env var to point to a
specific results file (defaults to logs/sweep_results_v2.json — the realistic-
spread sweep). Falls back to sweep_results.json for the original sweep."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

env_file = os.getenv("SWEEP_FILE")
if env_file:
    results_path = Path(env_file)
else:
    v2 = ROOT / "logs" / "sweep_results_v2.json"
    v1 = ROOT / "logs" / "sweep_results.json"
    results_path = v2 if v2.exists() else v1
out_path = ROOT / "docs" / ("sweep_report_v2.md" if "v2" in results_path.name else "sweep_report.md")
print(f"reading {results_path}")
print(f"writing {out_path}")

if not results_path.exists():
    sys.exit("no sweep_results.json yet")

with open(results_path) as f:
    data = json.load(f)

if not data:
    out_path.write_text("# Sweep Report\n\nNo results yet.\n")
    sys.exit(0)

df = pd.DataFrame(data)
df["score"] = df["win_rate"] * df["profit_factor"].clip(upper=10)  # cap inf-PF outliers

profitable = df[(df["profit_factor"] > 1.0) & (df["n_trades"] >= 20)].copy().sort_values("score", ascending=False)
high_wr = df[(df["win_rate"] >= 0.70) & (df["n_trades"] >= 20)].copy().sort_values(["win_rate", "n_trades"], ascending=[False, False])

# Group by (symbol, tf) → best strategy per pair
by_pair = (
    df[df["n_trades"] >= 20]
    .sort_values("score", ascending=False)
    .drop_duplicates(subset=["symbol", "timeframe"])
    .sort_values("score", ascending=False)
)

lines = []
lines.append("# Sweep Report (LIVE — updates as sweep progresses)\n")
lines.append(f"Last update: {pd.Timestamp.utcnow().isoformat()}  ·  Results: **{len(df)}**  ·  Unique (symbol, TF) combos: **{df.groupby(['symbol','timeframe']).ngroups}**\n")

symbols_done = sorted(df["symbol"].unique())
lines.append(f"Symbols processed so far: {', '.join(symbols_done)}\n")

def render_table(name, frame, cols, limit=20):
    out = [f"\n## {name}\n"]
    if not len(frame):
        out.append("_None yet._\n")
        return out
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in frame.head(limit).iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}")
            else:
                row.append(str(v))
        out.append("| " + " | ".join(row) + " |")
    return out

key_cols = ["symbol", "timeframe", "strategy", "n_trades", "trades_per_day", "win_rate", "profit_factor", "total_pnl", "max_dd_pct", "cls_acc"]

lines += render_table(f"Profitable configs (PF > 1.0, n_trades ≥ 20) — {len(profitable)} found", profitable, key_cols)
lines += render_table(f"High win-rate configs (WR ≥ 0.70, n_trades ≥ 20) — {len(high_wr)} found", high_wr, key_cols)
lines += render_table("Best strategy per (symbol, TF) combo", by_pair, key_cols, limit=40)
lines += render_table(f"ALL results (sorted by WR × PF) — {len(df)} total", df.sort_values("score", ascending=False), key_cols, limit=300)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines))
print(f"wrote {out_path} ({len(df)} rows)")
