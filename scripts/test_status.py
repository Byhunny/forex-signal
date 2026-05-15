"""Call _status_text and _leaderboard_text directly and print the output —
the same text Telegram would send."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.mt5_client import MT5Client
from forex_signal.execution.battle_royale import _leaderboard_text, _status_text, build_strategies

cfg = load_config()
client = MT5Client()
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)
strategies = build_strategies()

state_file = ROOT / "logs" / "battle_state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {}

out = ROOT / "logs" / "telegram_preview.txt"
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n/status\n" + "=" * 60 + "\n")
    f.write(_status_text(client, strategies, state))
    f.write("\n\n" + "=" * 60 + "\n/lb\n" + "=" * 60 + "\n")
    f.write(_leaderboard_text(client, strategies))

print(f"wrote {out}")
client.shutdown()
