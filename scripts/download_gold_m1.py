"""Download GOLD M1 history for scalping experiment."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.mt5_client import make_client

cfg = load_config()
client = make_client(prefer_real=True)
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)

df = client.fetch_history("GOLD", "M1", 50000)
out = ROOT / "data_cache" / "GOLD_mt5_M1.parquet"
df.to_parquet(out, index=False)
print(f"GOLD M1: {len(df)} bars saved to {out}")
print(f"Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
client.shutdown()
