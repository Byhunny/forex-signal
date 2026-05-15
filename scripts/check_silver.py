"""Check SILVER M30 latest bars + current price to see what happened."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.mt5_client import make_client
import MetaTrader5 as mt5

cfg = load_config()
client = make_client(prefer_real=True)
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)

# Last 12 M30 bars + current tick
df = client.fetch_history("SILVER", "M30", 12)
tick = mt5.symbol_info_tick("SILVER")

print("Son 12 SILVER M30 bar:")
print(df.to_string(index=False))
print(f"\nCurrent: bid={tick.bid:.3f}  ask={tick.ask:.3f}  spread={(tick.ask-tick.bid)*100:.1f} cent")

# Compute direction from last 3 bars
import pandas as pd
last_close = df["close"].iloc[-1]
prev_close = df["close"].iloc[-2]
older_close = df["close"].iloc[-4]
print(f"\n3-bar change:  {older_close:.3f} → {last_close:.3f}  ({(last_close-older_close)/older_close*100:+.2f}%)")
print(f"1-bar change:  {prev_close:.3f} → {last_close:.3f}  ({(last_close-prev_close)/prev_close*100:+.2f}%)")

client.shutdown()
