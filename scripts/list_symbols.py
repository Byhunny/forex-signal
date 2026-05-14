"""List symbols available on the MT5 terminal that are relevant to our strategy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import MetaTrader5 as mt5

from forex_signal.config import load_config

cfg = load_config()
mt5.initialize()
mt5.login(cfg.mt5.login, password=cfg.mt5.password, server=cfg.mt5.server)

symbols = mt5.symbols_get()
print(f"total symbols: {len(symbols)}")
print()

# Search for gold-like and crypto symbols
print("Gold / metals (containing GOLD/XAU/SILVER/XAG):")
for s in symbols:
    n = s.name.upper()
    if "GOLD" in n or "XAU" in n or "SILVER" in n or "XAG" in n:
        print(f"  {s.name:18s} spread={s.spread} digits={s.digits} contract={s.trade_contract_size} tick_value={s.trade_tick_value:.4f}")

print()
print("Crypto (BTC / ETH):")
for s in symbols:
    n = s.name.upper()
    if "BTC" in n or "ETH" in n or "SOL" in n:
        print(f"  {s.name:18s} spread={s.spread} digits={s.digits} contract={s.trade_contract_size} tick_value={s.trade_tick_value:.4f}")

print()
print("Indices (GER40, NAS100, US30, SPX, DAX):")
keywords = ["GER40", "NAS100", "US30", "US500", "SPX", "DAX", "DJI", "NDX"]
for s in symbols:
    n = s.name.upper()
    if any(k in n for k in keywords):
        print(f"  {s.name:18s} spread={s.spread} digits={s.digits} contract={s.trade_contract_size} tick_value={s.trade_tick_value:.4f}")

mt5.shutdown()
