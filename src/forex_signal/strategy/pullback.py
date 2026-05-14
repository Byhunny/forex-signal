"""Pullback zone detection — entries in trend direction after retrace to EMA50.

For an uptrend: price has pulled back AND touched the EMA50 area within the last
`recent_window` bars AND currently shows a bullish candle (close > open).
Symmetric for downtrend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_pullbacks(
    df: pd.DataFrame,
    structure: pd.DataFrame,
    trend_strength: pd.DataFrame,
    recent_window: int = 4,
    ema_band_atr_mult: float = 0.5,
) -> pd.DataFrame:
    """Returns DataFrame with bullish_pullback, bearish_pullback booleans.

    A bullish_pullback at bar i requires:
      - trend (from structure) is uptrend OR trend_strength.strong_bull
      - within the last `recent_window` bars, low touched (within `ema_band_atr_mult * ATR`)
        of EMA50
      - this bar is bullish (close > open)
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    ema = trend_strength["ema_fast"].to_numpy(dtype=np.float64)

    # Rolling ATR approximation for band width
    prev_close = pd.Series(close).shift(1).to_numpy()
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr[np.isnan(tr)] = 0.0
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()

    trend = structure["trend"].to_numpy()
    strong_bull = trend_strength["strong_bull"].to_numpy(dtype=bool)
    strong_bear = trend_strength["strong_bear"].to_numpy(dtype=bool)

    bullish_pb = np.zeros(n, dtype=bool)
    bearish_pb = np.zeros(n, dtype=bool)

    for i in range(recent_window, n):
        band = ema_band_atr_mult * atr[i] if not np.isnan(atr[i]) else 0.0
        if band <= 0:
            continue

        # Did price touch EMA50 from above recently? (low <= ema + band) within window
        touched_bull = any(
            low[j] <= ema[j] + band and low[j] >= ema[j] - band
            for j in range(i - recent_window, i + 1)
        )
        touched_bear = any(
            high[j] >= ema[j] - band and high[j] <= ema[j] + band
            for j in range(i - recent_window, i + 1)
        )

        is_bull_candle = close[i] > open_[i]
        is_bear_candle = close[i] < open_[i]

        if (trend[i] == "uptrend" or strong_bull[i]) and touched_bull and is_bull_candle:
            bullish_pb[i] = True
        if (trend[i] == "downtrend" or strong_bear[i]) and touched_bear and is_bear_candle:
            bearish_pb[i] = True

    return pd.DataFrame(
        {"bullish_pullback": bullish_pb, "bearish_pullback": bearish_pb},
        index=df.index,
    )
