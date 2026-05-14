"""Diagnose which SMC filters fire on the test slice."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forex_signal.data.features import compute_features
from forex_signal.strategy.entry_engine import EntryConfig, compute_all_signals

df = pd.read_parquet(ROOT / "data_cache/EURUSD_mt5_M5.parquet")
features = compute_features(df)
smc = compute_all_signals(features, EntryConfig())

# Test slice = last 15%
n = len(smc)
test = smc.iloc[int(n * 0.85) :]
print(f"test bars: {len(test)}")
print()
print(f"session_active true:         {test['session_active'].sum()} / {len(test)}  ({test['session_active'].mean()*100:.1f}%)")
print(f"htf_bullish:                 {(test['htf_bias'] == 'bullish').sum()}")
print(f"htf_bearish:                 {(test['htf_bias'] == 'bearish').sum()}")
print(f"trend=uptrend:               {(test['trend'] == 'uptrend').sum()}")
print(f"trend=downtrend:             {(test['trend'] == 'downtrend').sum()}")
print(f"trend=range:                 {(test['trend'] == 'range').sum()}")
print(f"strong_bull:                 {test['strong_bull'].sum()}")
print(f"strong_bear:                 {test['strong_bear'].sum()}")
print(f"is_range (ADX<20):           {test['is_range'].sum()}")
print(f"bullish_sweep:               {test['bullish_sweep'].sum()}")
print(f"bearish_sweep:               {test['bearish_sweep'].sum()}")
print(f"bullish_sweep_recent:        {test['bullish_sweep_recent'].sum()}")
print(f"bearish_sweep_recent:        {test['bearish_sweep_recent'].sum()}")
print(f"bullish_pullback:            {test['bullish_pullback'].sum()}")
print(f"bearish_pullback:            {test['bearish_pullback'].sum()}")
print()
print(f"adx mean / median / max:     {test['adx'].mean():.1f} / {test['adx'].median():.1f} / {test['adx'].max():.1f}")
print()
# Compound: session + htf + trend + sweep_recent + pullback (no LNN)
cond_bull = (
    test["session_active"]
    & (test["htf_bias"] == "bullish")
    & ((test["trend"] == "uptrend") | test["strong_bull"])
    & ~test["strong_bear"]
    & test["bullish_sweep_recent"]
    & test["bullish_pullback"]
)
cond_bear = (
    test["session_active"]
    & (test["htf_bias"] == "bearish")
    & ((test["trend"] == "downtrend") | test["strong_bear"])
    & ~test["strong_bull"]
    & test["bearish_sweep_recent"]
    & test["bearish_pullback"]
)
print(f"ALL SMC bull-conditions met: {cond_bull.sum()}")
print(f"ALL SMC bear-conditions met: {cond_bear.sum()}")
print()
# Without pullback
cond_bull_np = (
    test["session_active"]
    & (test["htf_bias"] == "bullish")
    & ((test["trend"] == "uptrend") | test["strong_bull"])
    & ~test["strong_bear"]
    & test["bullish_sweep_recent"]
)
cond_bear_np = (
    test["session_active"]
    & (test["htf_bias"] == "bearish")
    & ((test["trend"] == "downtrend") | test["strong_bear"])
    & ~test["strong_bull"]
    & test["bearish_sweep_recent"]
)
print(f"Without pullback (bull/bear): {cond_bull_np.sum()} / {cond_bear_np.sum()}")
print()
# Without sweep too
cond_bull_ns = (
    test["session_active"]
    & (test["htf_bias"] == "bullish")
    & ((test["trend"] == "uptrend") | test["strong_bull"])
    & ~test["strong_bear"]
)
cond_bear_ns = (
    test["session_active"]
    & (test["htf_bias"] == "bearish")
    & ((test["trend"] == "downtrend") | test["strong_bear"])
    & ~test["strong_bull"]
)
print(f"Without sweep & pullback (bull/bear): {cond_bull_ns.sum()} / {cond_bear_ns.sum()}")
