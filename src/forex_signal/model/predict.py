"""Inference wrapper — returns (predicted_returns, direction_probability, atr)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forex_signal.data.features import compute_features
from forex_signal.model.lnn import ForexLNN


@dataclass
class Predictor:
    model: ForexLNN
    feature_columns: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    pred_horizon: int
    seq_len: int
    device: torch.device
    target_mean: float = 0.0
    target_std: float = 1.0

    @classmethod
    def load(cls, path: Path | str, seq_len: int = 60, device: str = "cpu") -> "Predictor":
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = ForexLNN(
            n_features=ckpt["n_features"],
            units=ckpt["units"],
            pred_horizon=ckpt["pred_horizon"],
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(device)
        return cls(
            model=model,
            feature_columns=ckpt["feature_columns"],
            feature_means=np.array(ckpt["feature_means"], dtype=np.float32),
            feature_stds=np.array(ckpt["feature_stds"], dtype=np.float32),
            pred_horizon=ckpt["pred_horizon"],
            seq_len=seq_len,
            device=torch.device(device),
            target_mean=float(ckpt.get("target_mean", 0.0)),
            target_std=float(ckpt.get("target_std", 1.0)),
        )

    def predict(self, ohlcv_df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
        """Returns (pred_returns_raw_log, direction_probability, last_atr).

        - pred_returns_raw_log: (pred_horizon,) per-step predicted log returns (denormalized)
        - direction_probability: scalar in [0,1], P(cumulative return > 0)
        - last_atr: float, ATR(14) of the last bar — used for SL/TP sizing
        """
        features_df = compute_features(ohlcv_df)
        if len(features_df) < self.seq_len:
            raise ValueError(
                f"need at least {self.seq_len} bars after feature warmup, got {len(features_df)}"
            )
        last_window = features_df.tail(self.seq_len)
        arr = last_window[self.feature_columns].to_numpy(dtype=np.float32)
        arr_norm = (arr - self.feature_means) / self.feature_stds
        x = torch.from_numpy(arr_norm[None, :, :]).to(self.device)
        with torch.no_grad():
            ret_norm, dir_logit = self.model(x)
        pred_raw = ret_norm.cpu().numpy()[0] * self.target_std + self.target_mean
        prob = float(torch.sigmoid(dir_logit).cpu().item())
        atr = float(features_df["atr_14"].iloc[-1])
        return pred_raw.astype(np.float32), prob, atr

    # Back-compat alias for existing call sites that expect (returns, atr)
    def predict_returns(self, ohlcv_df: pd.DataFrame) -> tuple[np.ndarray, float]:
        ret, _prob, atr = self.predict(ohlcv_df)
        return ret, atr
