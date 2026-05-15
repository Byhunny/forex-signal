"""Query MT5 for live spread on every symbol used by battle royale.
Reports current bid/ask, spread in points and in pips (for the bot's pip unit).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.execution.battle_royale import PIP_VALUE, TOP_10_STRATEGIES
import MetaTrader5 as mt5

cfg = load_config()
mt5.initialize()
mt5.login(cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server)

# All symbols from the original sweep (19 symbols)
ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "USDCNH", "USDSEK",
    "GOLD", "SILVER", "US100Cash", "US500Cash", "GER40Cash", "BTCUSD",
]
symbols = ALL_SYMBOLS

import json
from pathlib import Path

out = {}
print(f"{'Symbol':12s} {'Bid':>12s} {'Ask':>12s} {'Spread(price)':>14s} {'Spread(pips)':>13s}")
print("-" * 80)
for sym in symbols:
    tick = mt5.symbol_info_tick(sym)
    if not tick:
        print(f"{sym:12s} NOT AVAILABLE")
        out[sym] = None
        continue
    spread_price = tick.ask - tick.bid
    pip = PIP_VALUE.get(sym, 0.0001)
    spread_pips = spread_price / pip
    print(f"{sym:12s} {tick.bid:12.5f} {tick.ask:12.5f} {spread_price:14.6f} {spread_pips:13.2f}")
    out[sym] = {"bid": tick.bid, "ask": tick.ask, "spread_price": spread_price, "spread_pips": spread_pips}

out_file = Path(__file__).resolve().parents[1] / "logs" / "live_spreads.json"
out_file.parent.mkdir(exist_ok=True)
out_file.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_file}")
mt5.shutdown()
