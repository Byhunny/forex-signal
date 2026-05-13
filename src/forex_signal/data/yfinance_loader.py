"""yfinance loader — used for offline dev / smoke tests.

yfinance does not provide forex M5; we use 1h as a stand-in. The pipeline is identical;
the bot's actual training in production runs on MT5 history dumped by `mt5_client.fetch_history`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Yahoo intraday limits
_MAX_DAYS_BACK = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730, "90m": 60}


def _clip_to_limit(start: str, end: str | None, interval: str) -> tuple[str, str]:
    limit_days = _MAX_DAYS_BACK.get(interval)
    end_dt = datetime.now(timezone.utc) if end is None else datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    if limit_days is not None:
        earliest_allowed = datetime.now(timezone.utc) - timedelta(days=limit_days - 1)
        if start_dt < earliest_allowed:
            log.warning("yfinance %s only supports last %d days — clipping start", interval, limit_days)
            start_dt = earliest_allowed
        if end_dt < start_dt:
            end_dt = datetime.now(timezone.utc)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def load_eurusd(
    start: str = "2024-08-01",
    end: str | None = None,
    interval: str = "1h",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return DataFrame[time, open, high, low, close, tick_volume]."""
    start, end = _clip_to_limit(start, end, interval)
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"EURUSD_yf_{interval}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    df = yf.download(
        "EURUSD=X",
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty dataframe for {start}..{end} {interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(
        columns={
            "Datetime": "time",
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "tick_volume",
        }
    )
    df["tick_volume"] = df["tick_volume"].fillna(0).astype(float)
    if df["tick_volume"].sum() == 0:
        # yfinance forex has 0 volume — synthesize from range to keep z-scoring useful
        df["tick_volume"] = ((df["high"] - df["low"]) * 1e6).clip(lower=1.0)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    out = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()

    if cache_path is not None:
        out.to_parquet(cache_path, index=False)

    return out
