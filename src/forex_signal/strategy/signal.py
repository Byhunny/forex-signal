"""Signal generation from multi-step predicted returns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Signal:
    direction: int  # +1 buy, -1 sell, 0 no trade
    confidence: float  # 0..1 — directional consistency
    cum_return: float  # predicted cumulative log-return
    reason: str = ""


def generate_signal(
    pred_returns: np.ndarray,
    min_predicted_return_bps: float = 1.5,
    min_directional_consistency: float = 0.6,
) -> Signal:
    """Translate per-step predicted returns into a trade direction.

    - direction non-zero only when both magnitude AND consistency thresholds pass.
    - magnitude check uses absolute cumulative log-return in basis points (1 bp = 1e-4).
    """
    if pred_returns.ndim != 1 or pred_returns.size == 0:
        return Signal(0, 0.0, 0.0, "invalid prediction shape")

    cum = float(pred_returns.sum())
    cum_bps = abs(cum) * 1e4

    if cum_bps < min_predicted_return_bps:
        return Signal(0, 0.0, cum, f"magnitude too small ({cum_bps:.2f} bps)")

    direction = int(np.sign(cum))
    step_signs = np.sign(pred_returns)
    consistency = float((step_signs == direction).mean())

    if consistency < min_directional_consistency:
        return Signal(0, consistency, cum, f"low consistency ({consistency:.2f})")

    return Signal(direction, consistency, cum, "ok")
