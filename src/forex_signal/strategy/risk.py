"""Risk + position sizing logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradePlan:
    direction: int
    lot: float
    sl_price: float
    tp_price: float
    sl_distance: float
    tp_distance: float


def compute_trade_plan(
    direction: int,
    price: float,
    atr: float,
    predicted_cum_return: float,
    lot_size: float = 0.01,
    sl_atr_multiplier: float = 1.5,
    tp_atr_min_multiplier: float = 1.0,
    tp_atr_max_multiplier: float = 4.0,
) -> TradePlan:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if atr <= 0:
        raise ValueError("atr must be positive")

    sl_distance = sl_atr_multiplier * atr
    predicted_move = abs(predicted_cum_return) * price
    tp_distance = max(
        tp_atr_min_multiplier * atr,
        min(predicted_move, tp_atr_max_multiplier * atr),
    )

    if direction == 1:
        sl_price = price - sl_distance
        tp_price = price + tp_distance
    else:
        sl_price = price + sl_distance
        tp_price = price - tp_distance

    return TradePlan(
        direction=direction,
        lot=lot_size,
        sl_price=sl_price,
        tp_price=tp_price,
        sl_distance=sl_distance,
        tp_distance=tp_distance,
    )


@dataclass
class KillSwitchState:
    day_start_equity: float
    current_equity: float
    max_daily_loss_pct: float = 2.0

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.current_equity - self.day_start_equity) / self.day_start_equity * 100.0

    @property
    def tripped(self) -> bool:
        return self.daily_pnl_pct <= -self.max_daily_loss_pct
