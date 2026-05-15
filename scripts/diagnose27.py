"""Diagnostic for the NEW 27-contender list. Shows each strategy's current
decision + the failing filter (if any)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import pandas as pd
from forex_signal.config import load_config
from forex_signal.data.features import compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.execution.battle_royale import build_contenders
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import compute_all_signals, evaluate_entry

cfg = load_config()
client = make_client(prefer_real=True)
client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path)
strats = build_contenders()
cache = {}
for s in strats:
    k = str(s.model_path)
    if k not in cache:
        cache[k] = Predictor.load(s.model_path, seq_len=s.seq_len)
    s.predictor = cache[k]

print(f"{'Magic':>9} {'Yarismaci':<42} {'Mode':<8} {'Prob':>5} {'CumBps':>8} {'Dec':>5}  Reason")
print("-" * 130)
for s in strats:
    try:
        df = client.fetch_history(s.symbol, s.timeframe, 200)
        pred, prob, atr = s.predictor.predict(df)
        cum_bps = float(pred.sum()) * 1e4
        features = compute_features(df)
        smc = compute_all_signals(features, s.entry_cfg)
        d = evaluate_entry(smc.iloc[-1], prob, s.entry_cfg)
        ds = "BUY" if d.direction == 1 else ("SELL" if d.direction == -1 else "no")
        reason = (d.reasons[0] if d.reasons else "")[:60]
        print(f"{s.magic:>9} {s.name:<42} {s.exit_mode:<8} {prob:>5.3f} {cum_bps:>+8.1f}  {ds:>5}  {reason}")
    except Exception as e:
        print(f"{s.magic} {s.name} ERROR: {e}")
client.shutdown()
