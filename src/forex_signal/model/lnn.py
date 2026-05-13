"""Liquid Neural Network (Closed-form Continuous-time) model for multi-step return prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC


class ForexLNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        units: int = 48,
        pred_horizon: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.units = units
        self.pred_horizon = pred_horizon
        self.cfc = CfC(n_features, units)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(units, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_features) -> (B, pred_horizon)"""
        out, _ = self.cfc(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))
