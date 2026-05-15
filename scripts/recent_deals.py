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
since = datetime.now(timezone.utc) - timedelta(days=2)
deals = mt5.history_deals_get(since, datetime.now(timezone.utc)) or ()
print(f"Last {len(deals)} deals (4h window):")
for d in deals:
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    entry = 'IN' if d.entry == 0 else 'OUT'
    side = 'BUY' if d.type == 0 else 'SELL'
    reason_str = {0: 'client', 3: 'TP', 4: 'SL', 5: 'SO', 6: 'rollover', 7: 'VMargin', 8: 'split'}.get(d.reason, str(d.reason))
    print(f"  {t.strftime('%H:%M:%S')} ticket={d.ticket} magic={d.magic} {entry:3s} {side} {d.symbol} vol={d.volume} price={d.price} profit={d.profit:+.2f} reason={reason_str}")
mt5.shutdown()
