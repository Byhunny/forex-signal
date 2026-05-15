"""MT5 smoke test — opens a tiny trade and immediately closes it.

Uses magic=99999999 so it never conflicts with battle royale (26051001..26051010).
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.mt5_client import make_client
from forex_signal.notifier.telegram import notify

SYMBOL = "EURUSD"
LOT = 0.01
MAGIC = 99999999

cfg = load_config()
client = make_client(prefer_real=True)
if not client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path):
    sys.exit("MT5 connect failed")

info = client.get_account_info()
print(f"Account login={info['login']} equity=${info['equity']:.2f}")

# Fetch latest tick to know current price & set safe SL/TP
import MetaTrader5 as mt5
tick = mt5.symbol_info_tick(SYMBOL)
if tick is None:
    sys.exit(f"no tick for {SYMBOL}")

print(f"{SYMBOL} bid={tick.bid:.5f} ask={tick.ask:.5f} spread={(tick.ask - tick.bid) * 10000:.1f} pips")

# Open BUY with wide SL/TP (we'll close manually before they hit)
entry = tick.ask
sl = entry - 0.0100   # 100 pips below — far away
tp = entry + 0.0100   # 100 pips above — far away

print(f"\n-> Opening BUY {LOT} lot @ ~{entry:.5f}, SL={sl:.5f}, TP={tp:.5f}, magic={MAGIC}")
notify(f"🧪 *Smoke test* opening BUY `{SYMBOL}` {LOT} lot @ {entry:.5f}")

result = client.place_order(
    symbol=SYMBOL,
    order_type="buy",
    lot=LOT,
    sl_price=sl,
    tp_price=tp,
    magic=MAGIC,
    comment="smoke_test",
)

if not result.success:
    notify(f"WARN Smoke test OPEN failed: {result.error}")
    sys.exit(f"order failed: {result.error}\nraw: {result.raw}")

print(f"OK Opened ticket={result.ticket}")
notify(f"OK Smoke test OPEN ok — ticket `{result.ticket}`, holding 3 sec…")

time.sleep(3)

print(f"\n-> Closing ticket {result.ticket}")
close_result = client.close_position(result.ticket)

if not close_result.success:
    notify(f"WARN Smoke test CLOSE failed: {close_result.error}")
    sys.exit(f"close failed: {close_result.error}\nraw: {close_result.raw}")

# Read the closing deal to get final P&L
deals = mt5.history_deals_get(position=result.ticket)
if deals:
    total_pnl = sum(d.profit + d.swap + d.commission for d in deals)
    print(f"OK Closed — P&L: ${total_pnl:+.4f} ({len(deals)} deals)")
    notify(f"OK Smoke test CLOSED ticket `{result.ticket}` — P&L: ${total_pnl:+.4f}")
else:
    print("OK Closed (couldn't read deal history)")
    notify(f"OK Smoke test CLOSED ticket `{result.ticket}`")

info = client.get_account_info()
print(f"\nFinal account equity: ${info['equity']:.2f}")
client.shutdown()
print("\n=== smoke test PASSED ===")
