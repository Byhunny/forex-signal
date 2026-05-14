"""Diagnostic — examine prediction magnitudes."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from forex_signal.data.features import compute_features
from forex_signal.model.predict import Predictor

df = pd.read_parquet(ROOT / "data_cache/EURUSD_mt5_M1.parquet")
p = Predictor.load(ROOT / "models/lnn_eurusd_m1.pt", seq_len=90)
print(f"target_mean={p.target_mean:.3e}, target_std={p.target_std:.3e}")

feats = compute_features(df)
arr = feats[p.feature_columns].to_numpy().astype("float32")
arr_norm = (arr - p.feature_means) / p.feature_stds

preds_raw = []
preds_norm = []
for i in range(p.seq_len, len(arr_norm), 50):
    x = torch.from_numpy(arr_norm[i - p.seq_len : i][None, :, :])
    with torch.no_grad():
        pn = p.model(x).numpy()[0]
    pr = pn * p.target_std + p.target_mean
    preds_norm.append(pn)
    preds_raw.append(pr)

pn_arr = np.array(preds_norm)
pr_arr = np.array(preds_raw)
cum_bps = pr_arr.sum(axis=1) * 1e4

print(f"n_samples: {len(cum_bps)}")
print(f"normalized pred range: {pn_arr.min():.3f} to {pn_arr.max():.3f}")
print(f"raw pred range (bps):  {(pr_arr.min()*1e4):.4f} to {(pr_arr.max()*1e4):.4f}")
print(f"CUM raw pred (bps):    min={cum_bps.min():.4f}, max={cum_bps.max():.4f}, mean_abs={np.abs(cum_bps).mean():.4f}")
print(f"How many cross 1.5 bps threshold: {(np.abs(cum_bps) >= 1.5).sum()} / {len(cum_bps)}")
print(f"How many cross 2.0 bps threshold: {(np.abs(cum_bps) >= 2.0).sum()} / {len(cum_bps)}")
print(f"How many cross 0.5 bps threshold: {(np.abs(cum_bps) >= 0.5).sum()} / {len(cum_bps)}")
print()
print(f"Sample (first 3):\n{pr_arr[:3]}")
