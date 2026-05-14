"""Battle royale leaderboard — query MT5 closed-trade history per magic number,
compute per-strategy stats, render a sortable markdown table."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import MetaTrader5 as mt5

from forex_signal.config import load_config
from forex_signal.execution.battle_royale import TOP_10_STRATEGIES

cfg = load_config()
if not cfg.mt5:
    sys.exit("no MT5 creds")

mt5.initialize()
ok = mt5.login(cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server)
if not ok:
    sys.exit(f"login failed: {mt5.last_error()}")

# Pull deals from last 30 days
since = datetime.now(timezone.utc) - timedelta(days=30)
deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
if deals is None:
    deals = ()

rows = []
for d in deals:
    rows.append({
        "ticket": d.ticket, "position_id": d.position_id, "symbol": d.symbol,
        "type": d.type, "entry": d.entry, "magic": d.magic, "volume": d.volume,
        "price": d.price, "profit": d.profit, "swap": d.swap, "commission": d.commission,
        "time": datetime.fromtimestamp(d.time, tz=timezone.utc),
    })
df = pd.DataFrame(rows)

if df.empty:
    print("No deals found yet. Battle hasn't produced any closed trades.")
    mt5.shutdown()
    sys.exit(0)

# Each closed position has 2 deals (entry + exit). Aggregate by position_id.
# Final P&L = sum(profit + swap + commission) across both deals of the position.
df["net"] = df["profit"] + df["swap"] + df["commission"]
per_pos = df.groupby(["position_id", "magic", "symbol"], as_index=False).agg(
    net=("net", "sum"), opens=("entry", "count"),
    first_time=("time", "min"), last_time=("time", "max"),
)
# Only closed positions (entry has both 0 IN and 1 OUT — opens count == 2)
closed = per_pos[per_pos["opens"] >= 2].copy()

magic_to_name = {m: n for (n, _, _, _, m, _) in TOP_10_STRATEGIES}

stats = []
for (n, sym, tf, model, magic, _) in TOP_10_STRATEGIES:
    sub = closed[closed["magic"] == magic]
    n_trades = len(sub)
    if n_trades == 0:
        stats.append({"strategy": n, "symbol": sym, "tf": tf, "n_trades": 0,
                      "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0,
                      "best": 0.0, "worst": 0.0})
        continue
    wins = sub[sub["net"] > 0]
    losses = sub[sub["net"] <= 0]
    stats.append({
        "strategy": n, "symbol": sym, "tf": tf,
        "n_trades": n_trades,
        "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / n_trades,
        "total_pnl": float(sub["net"].sum()),
        "best": float(sub["net"].max()),
        "worst": float(sub["net"].min()),
    })

# Also count currently OPEN positions per magic
open_pos = mt5.positions_get()
open_by_magic: dict[int, int] = {}
if open_pos:
    for p in open_pos:
        open_by_magic[p.magic] = open_by_magic.get(p.magic, 0) + 1

# Inject open counts
for s in stats:
    magic = next(m for (n, _, _, _, m, _) in TOP_10_STRATEGIES if n == s["strategy"])
    s["open_now"] = open_by_magic.get(magic, 0)

sdf = pd.DataFrame(stats)
sdf = sdf.sort_values(["total_pnl", "win_rate"], ascending=[False, False])

# Render markdown
out = ROOT / "docs" / "battle_leaderboard.md"
out.parent.mkdir(exist_ok=True)
lines = [
    f"# 🏆 Battle Royale Leaderboard",
    f"\nUpdated: {datetime.now(timezone.utc).isoformat()}  ·  Account equity: ${mt5.account_info().equity:.2f}",
    f"\nClosed trades since {since.date()}: **{len(closed)}**  ·  Currently open: **{sum(open_by_magic.values())}**\n",
]

cols = ["strategy", "n_trades", "wins", "losses", "win_rate", "total_pnl", "best", "worst", "open_now"]
lines.append("| Rank | " + " | ".join(cols) + " |")
lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
for i, row in enumerate(sdf.itertuples(index=False), 1):
    vals = []
    for c in cols:
        v = getattr(row, c)
        if isinstance(v, float):
            vals.append(f"{v:+.2f}" if c in ("total_pnl", "best", "worst") else f"{v:.3f}")
        else:
            vals.append(str(v))
    lines.append(f"| {i} | " + " | ".join(vals) + " |")

out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWritten to {out}")
mt5.shutdown()
