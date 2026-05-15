import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.mt5_client import MT5Client

cfg = load_config()
c = MT5Client()
c.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)
i = c.get_account_info()
pos = c.get_positions()
print(f"Account login: {i['login']}")
print(f"Equity:        ${i['equity']:.2f}")
print(f"Balance:       ${i['balance']:.2f}")
print(f"Free margin:   ${i['margin_free']:.2f}")
print(f"Open positions: {len(pos)}")
for p in pos:
    print(f"  - magic={p.get('magic')} symbol={p.get('symbol')} volume={p.get('volume')} profit={p.get('profit')}")
c.shutdown()
