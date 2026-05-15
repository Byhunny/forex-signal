"""For each of the 10 battle royale strategies, show:
- Current LNN probability
- Predicted cumulative return (bps)
- Each filter's pass/fail
- What's blocking the signal
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_signal.config import load_config
from forex_signal.data.features import compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.execution.battle_royale import build_strategies
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import compute_all_signals, evaluate_entry

cfg = load_config()
client = make_client(prefer_real=True)
ok = client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)
if not ok:
    sys.exit("connect failed")

strategies = build_strategies()

# Load predictors
cache = {}
for s in strategies:
    k = str(s.model_path)
    if k not in cache:
        cache[k] = Predictor.load(s.model_path, seq_len=s.seq_len)
    s.predictor = cache[k]

print(f"{'Strategy':40s} {'Prob':>6s} {'CumBps':>8s} {'ATR':>10s} {'Decision':>10s} {'Reason'}")
print("-" * 130)

for s in strategies:
    try:
        df = client.fetch_history(s.symbol, s.timeframe, 200)
        pred_ret, prob, atr = s.predictor.predict(df)
        cum_bps = float(pred_ret.sum()) * 1e4

        features = compute_features(df)
        smc = compute_all_signals(features, s.entry_cfg)
        decision = evaluate_entry(smc.iloc[-1], prob, s.entry_cfg)

        # Show per-filter detail
        perm = decision.permission
        failed = [k for k, v in perm.items() if not v]
        reason = ", ".join(failed) if failed else (decision.reasons[0] if decision.reasons else "ok")
        dir_str = {1: "BUY", -1: "SELL", 0: "no"}[decision.direction]
        print(f"{s.name:40s} {prob:6.3f} {cum_bps:+8.2f} {atr:10.5f} {dir_str:>10s}  {reason[:80]}")
    except Exception as e:
        print(f"{s.name:40s} ERROR: {e}")

client.shutdown()
