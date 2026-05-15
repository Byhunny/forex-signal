import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
import MetaTrader5 as mt5

cfg = load_config()
mt5.initialize()
mt5.login(cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server)

print("Server time:", datetime.fromtimestamp(mt5.symbol_info_tick("EURUSD").time, tz=timezone.utc) if mt5.symbol_info_tick("EURUSD") else "?")
print("Local UTC:", datetime.now(timezone.utc))
print()

# Method 1: known ticket (SILVER trade was ticket 748884450)
print("=== Method 1: history_deals_get(position=748884450) ===")
deals = mt5.history_deals_get(position=748884450) or ()
print(f"deals: {len(deals)}")
for d in deals:
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    print(f"  {t} ticket={d.ticket} magic={d.magic} entry={d.entry} type={d.type} profit={d.profit} reason={d.reason}")

# Method 2: Last 6 hours wide
print("\n=== Method 2: history_deals_get(date_from=6h ago, date_to=now) ===")
since = datetime.now(timezone.utc) - timedelta(hours=6)
deals = mt5.history_deals_get(since, datetime.now(timezone.utc)) or ()
print(f"deals in 6h window: {len(deals)}")
for d in deals[:10]:
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    print(f"  {t} ticket={d.ticket} magic={d.magic} entry={d.entry} profit={d.profit}")

# Method 3: from epoch 0
print("\n=== Method 3: history_deals_get(epoch_0, now) ===")
deals = mt5.history_deals_get(datetime(2020, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)) or ()
print(f"all deals: {len(deals)}")
if deals:
    print("First 3:")
    for d in deals[:3]:
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        print(f"  {t} ticket={d.ticket} magic={d.magic}")
    print("Last 3:")
    for d in deals[-3:]:
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        print(f"  {t} ticket={d.ticket} magic={d.magic} profit={d.profit}")

mt5.shutdown()
