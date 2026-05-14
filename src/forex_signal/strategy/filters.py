"""Trading filters: trend strength (ADX + EMA slope), range, session, multi-timeframe bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

UTC_LONDON_NY_START = 12  # 12:00 UTC — London open + before NY
UTC_LONDON_NY_END = 16    # 16:00 UTC — end of overlap


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — used in ADX/ATR."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = _wilder_smooth(dx, period)
    return adx


def compute_trend_strength(
    df: pd.DataFrame,
    ema_fast_period: int = 50,
    slope_window: int = 5,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Return DataFrame with: ema_fast, ema_slope, adx, strong_bull, strong_bear, is_range."""
    ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False).mean()
    ema_slope = ema_fast.diff(slope_window) / df["close"]  # normalized slope
    adx = compute_adx(df, adx_period)

    # Threshold: 0.0001 ≈ 1 pip per 5 M5 bars (small but enough to indicate move)
    slope_threshold = 0.0001
    strong_bull = (ema_slope > slope_threshold) & (adx > 25)
    strong_bear = (ema_slope < -slope_threshold) & (adx > 25)
    is_range = adx < 20

    return pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slope": ema_slope,
            "adx": adx,
            "strong_bull": strong_bull,
            "strong_bear": strong_bear,
            "is_range": is_range,
        },
        index=df.index,
    )


def in_active_session(timestamps: pd.Series) -> pd.Series:
    """True for bars in the London-NY overlap (12-16 UTC)."""
    hours = pd.to_datetime(timestamps, utc=True).dt.hour
    return (hours >= UTC_LONDON_NY_START) & (hours < UTC_LONDON_NY_END)


def htf_bias_from_m5(
    df_m5: pd.DataFrame, htf_minutes: int = 60, ema_period: int = 50
) -> pd.Series:
    """Approximate H1 bias by resampling close to H1 and comparing to its EMA50.

    Returns Series aligned to df_m5: "bullish" if close > ema, else "bearish".
    """
    times = pd.to_datetime(df_m5["time"], utc=True)
    close = df_m5["close"]
    s = pd.Series(close.to_numpy(), index=times)
    h1 = s.resample(f"{htf_minutes}min").last().ffill()
    ema = h1.ewm(span=ema_period, adjust=False).mean()
    bias = np.where(h1 > ema, "bullish", "bearish")
    bias_series = pd.Series(bias, index=h1.index)
    # Forward-fill onto m5 timeline
    aligned = bias_series.reindex(times, method="ffill").to_numpy()
    return pd.Series(aligned, index=df_m5.index)
