"""Liquid Neural Network (Closed-form Continuous-time) with multi-task heads.

Two heads:
- returns: per-step predicted log-returns for the next `pred_horizon` bars (regression, MSE)
- direction_logit: single logit for "cumulative return over horizon > 0" (binary, BCE)

The direction head gives a calibrated probability (after sigmoid) used as the LNN
confirmation in the SMC entry permission engine. The regression head still provides
magnitude for sizing TP.
"""
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
        self.return_head = nn.Linear(units, pred_horizon)
        self.direction_head = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, seq_len, n_features) -> (returns (B, pred_horizon), direction_logit (B,))"""
        out, _ = self.cfc(x)
        last = self.dropout(out[:, -1, :])
        returns = self.return_head(last)
        direction_logit = self.direction_head(last).squeeze(-1)
        return returns, direction_logit
