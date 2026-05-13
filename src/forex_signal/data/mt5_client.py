"""MT5 client + a mock fallback for non-Windows environments.

The real `MetaTrader5` package is Windows-only. On WSL / macOS / Linux dev, the import
fails — we expose a `MockMT5` that reads cached CSV / parquet so the rest of the
pipeline (features, training, backtesting) is fully testable without MT5.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

try:
    import MetaTrader5 as _mt5  # type: ignore
    _MT5_AVAILABLE = True
except ImportError:
    _mt5 = None
    _MT5_AVAILABLE = False


TIMEFRAME_MAP_NAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    error: str | None
    raw: Any | None = None


def _timeframe_const(tf: str):
    if _mt5 is None:
        return TIMEFRAME_MAP_NAMES[tf]
    mapping = {
        "M1": _mt5.TIMEFRAME_M1,
        "M5": _mt5.TIMEFRAME_M5,
        "M15": _mt5.TIMEFRAME_M15,
        "M30": _mt5.TIMEFRAME_M30,
        "H1": _mt5.TIMEFRAME_H1,
        "H4": _mt5.TIMEFRAME_H4,
        "D1": _mt5.TIMEFRAME_D1,
    }
    return mapping[tf]


class MT5Client:
    """Thin wrapper over the MetaTrader5 package. Only constructible on Windows."""

    def __init__(self) -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not available. Run on Windows or use MockMT5."
            )
        self._connected = False

    def connect(self, login: int, password: str, server: str, path: str = "") -> bool:
        kwargs: dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        if not _mt5.initialize(**kwargs):
            log.error("mt5 initialize failed: %s", _mt5.last_error())
            return False
        ok = _mt5.login(login, password=password, server=server)
        if not ok:
            log.error("mt5 login failed: %s", _mt5.last_error())
            _mt5.shutdown()
            return False
        self._connected = True
        log.info("mt5 connected: login=%s server=%s", login, server)
        return True

    def shutdown(self) -> None:
        if self._connected:
            _mt5.shutdown()
            self._connected = False

    def fetch_history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        tf = _timeframe_const(timeframe)
        rates = _mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no rates for {symbol} {timeframe}: {_mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        cols = ["time", "open", "high", "low", "close", "tick_volume"]
        if "spread" in df.columns:
            cols.append("spread")
        return df[cols].copy()

    def get_account_info(self) -> dict[str, Any]:
        info = _mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info failed: {_mt5.last_error()}")
        return info._asdict()

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        pos = _mt5.positions_get(symbol=symbol) if symbol else _mt5.positions_get()
        if pos is None:
            return []
        return [p._asdict() for p in pos]

    def place_order(
        self,
        symbol: str,
        order_type: str,  # "buy" | "sell"
        lot: float,
        sl_price: float,
        tp_price: float,
        magic: int = 20260514,
        comment: str = "lnn",
        deviation: int = 20,
    ) -> OrderResult:
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, f"no tick for {symbol}", None)
        price = tick.ask if order_type == "buy" else tick.bid
        ot = _mt5.ORDER_TYPE_BUY if order_type == "buy" else _mt5.ORDER_TYPE_SELL
        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": ot,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC,
        }
        result = _mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, f"order_send None: {_mt5.last_error()}", None)
        ok = result.retcode == _mt5.TRADE_RETCODE_DONE
        return OrderResult(
            success=ok,
            ticket=result.order if ok else None,
            error=None if ok else f"retcode={result.retcode}",
            raw=result._asdict(),
        )

    def close_position(self, ticket: int) -> OrderResult:
        positions = _mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(False, None, f"no position {ticket}", None)
        pos = positions[0]
        symbol = pos.symbol
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, "no tick", None)
        is_buy = pos.type == _mt5.POSITION_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        close_type = _mt5.ORDER_TYPE_SELL if is_buy else _mt5.ORDER_TYPE_BUY
        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "close",
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_IOC,
        }
        result = _mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, "order_send None", None)
        ok = result.retcode == _mt5.TRADE_RETCODE_DONE
        return OrderResult(ok, ticket if ok else None, None if ok else f"retcode={result.retcode}", result._asdict())


class MockMT5:
    """In-memory mock — reads from cached parquet/csv. For dev/test only."""

    def __init__(self, cache_dir: Path | str = "data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._equity = 10_000.0
        self._positions: list[dict[str, Any]] = []
        self._next_ticket = 1

    def connect(self, *_args, **_kwargs) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def fetch_history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        candidates = list(self.cache_dir.glob(f"{symbol}_*.parquet"))
        if not candidates:
            raise RuntimeError(
                f"MockMT5: no cached data for {symbol} in {self.cache_dir}. "
                "Run `python -m forex_signal.cli download` first."
            )
        df = pd.read_parquet(candidates[0])
        return df.tail(bars).reset_index(drop=True)

    def get_account_info(self) -> dict[str, Any]:
        return {"login": 0, "balance": self._equity, "equity": self._equity, "currency": "USD"}

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return [p for p in self._positions if p["symbol"] == symbol]
        return list(self._positions)

    def place_order(
        self, symbol: str, order_type: str, lot: float, sl_price: float, tp_price: float,
        magic: int = 0, comment: str = "", deviation: int = 0,
    ) -> OrderResult:
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(
            {
                "ticket": ticket, "symbol": symbol, "type": order_type, "volume": lot,
                "sl": sl_price, "tp": tp_price, "price_open": 0.0, "magic": magic,
                "comment": comment,
            }
        )
        return OrderResult(True, ticket, None, None)

    def close_position(self, ticket: int) -> OrderResult:
        before = len(self._positions)
        self._positions = [p for p in self._positions if p["ticket"] != ticket]
        ok = len(self._positions) < before
        return OrderResult(ok, ticket if ok else None, None if ok else "not found", None)


def make_client(prefer_real: bool = True) -> MT5Client | MockMT5:
    if prefer_real and _MT5_AVAILABLE:
        return MT5Client()
    if prefer_real:
        log.warning("MetaTrader5 not available — falling back to MockMT5")
    return MockMT5()


def is_mt5_available() -> bool:
    return _MT5_AVAILABLE
