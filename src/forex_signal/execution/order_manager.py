"""Order management — wraps MT5 client with strategy-aware open/close/manage logic."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from forex_signal.data.mt5_client import MT5Client, MockMT5, OrderResult
from forex_signal.strategy.risk import TradePlan, KillSwitchState
from forex_signal.strategy.signal import Signal

log = logging.getLogger(__name__)

MAGIC = 20260514


@dataclass
class OrderManagerConfig:
    symbol: str
    max_concurrent_positions: int = 2
    early_close_consistency_threshold: float = 0.7


class OrderManager:
    def __init__(
        self,
        client: MT5Client | MockMT5,
        config: OrderManagerConfig,
    ) -> None:
        self.client = client
        self.config = config

    def own_positions(self) -> list[dict]:
        return [p for p in self.client.get_positions(self.config.symbol) if p.get("magic") in (MAGIC, 0)]

    def can_open_new(self, kill_switch: KillSwitchState) -> tuple[bool, str]:
        if kill_switch.tripped:
            return False, f"kill switch tripped ({kill_switch.daily_pnl_pct:.2f}%)"
        n_open = len(self.own_positions())
        if n_open >= self.config.max_concurrent_positions:
            return False, f"max concurrent positions reached ({n_open})"
        return True, "ok"

    def open(self, plan: TradePlan, price: float, comment: str = "lnn") -> OrderResult:
        order_type = "buy" if plan.direction == 1 else "sell"
        return self.client.place_order(
            symbol=self.config.symbol,
            order_type=order_type,
            lot=plan.lot,
            sl_price=plan.sl_price,
            tp_price=plan.tp_price,
            magic=MAGIC,
            comment=comment,
        )

    def manage_reversal(self, current_signal: Signal) -> list[OrderResult]:
        """Close positions where the model now strongly predicts the opposite direction."""
        results: list[OrderResult] = []
        if current_signal.direction == 0 or current_signal.confidence < self.config.early_close_consistency_threshold:
            return results
        for pos in self.own_positions():
            pos_dir = 1 if str(pos.get("type")).lower().endswith("buy") or pos.get("type") in (0, "buy") else -1
            if pos_dir != current_signal.direction:
                ticket = int(pos["ticket"])
                log.info("reversal: closing ticket=%s pos_dir=%d signal_dir=%d", ticket, pos_dir, current_signal.direction)
                results.append(self.client.close_position(ticket))
        return results
