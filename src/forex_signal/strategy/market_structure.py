"""Market structure: swing highs/lows, HH/HL/LH/LL, BOS, CHoCH.

All functions operate on a DataFrame with columns: time, open, high, low, close.
No intrabar logic — only candle-close decisions, so the structure values are stable
once the bar at index i+lookback completes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StructureState:
    trend: str  # "uptrend" | "downtrend" | "range"
    last_swing_high: float | None
    last_swing_high_idx: int | None
    last_swing_low: float | None
    last_swing_low_idx: int | None
    bos_bull: bool        # break above last swing high in this bar
    bos_bear: bool        # break below last swing low in this bar
    choch_bull: bool      # change of character to bullish
    choch_bear: bool      # change of character to bearish


def find_swings(df: pd.DataFrame, lookback: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing_high_mask, swing_low_mask) — booleans aligned to df rows.

    A swing high at index i requires high[i] strictly greater than all highs in
    [i-lookback, i+lookback] (excluding i). Symmetric for swing low. The first/last
    `lookback` rows can never be swings (mask is False there).
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window_h = high[i - lookback : i + lookback + 1]
        window_l = low[i - lookback : i + lookback + 1]
        if high[i] == window_h.max() and (window_h == high[i]).sum() == 1:
            sh[i] = True
        if low[i] == window_l.min() and (window_l == low[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def compute_structure(
    df: pd.DataFrame,
    lookback: int = 3,
) -> pd.DataFrame:
    """Return a DataFrame aligned to df with columns:
    swing_high, swing_low, last_swing_high, last_swing_low,
    trend ('uptrend' | 'downtrend' | 'range'),
    bos_bull, bos_bear, choch_bull, choch_bear.
    Swing values become known with a `lookback`-bar delay (only confirmed on close).
    """
    sh_mask, sl_mask = find_swings(df, lookback)
    n = len(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    # Track the last 2 swing highs and lows confirmed up to (and including) bar i-lookback.
    # We only "observe" a swing at index s when bar s+lookback closes.
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    prev_swing_high = np.full(n, np.nan)
    prev_swing_low = np.full(n, np.nan)

    cur_high = np.nan
    prev_high = np.nan
    cur_low = np.nan
    prev_low = np.nan
    for i in range(n):
        confirm_idx = i - lookback
        if confirm_idx >= 0:
            if sh_mask[confirm_idx]:
                prev_high = cur_high
                cur_high = high[confirm_idx]
            if sl_mask[confirm_idx]:
                prev_low = cur_low
                cur_low = low[confirm_idx]
        last_swing_high[i] = cur_high
        last_swing_low[i] = cur_low
        prev_swing_high[i] = prev_high
        prev_swing_low[i] = prev_low

    trend = np.array(["range"] * n, dtype=object)
    for i in range(n):
        ch = last_swing_high[i]
        ph = prev_swing_high[i]
        cl = last_swing_low[i]
        pl = prev_swing_low[i]
        if np.isnan(ch) or np.isnan(ph) or np.isnan(cl) or np.isnan(pl):
            trend[i] = "range"
            continue
        if ch > ph and cl > pl:
            trend[i] = "uptrend"
        elif ch < ph and cl < pl:
            trend[i] = "downtrend"
        else:
            trend[i] = "range"

    # BOS: close breaks above last_swing_high (bull) or below last_swing_low (bear)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev_ch = last_swing_high[i - 1]
        prev_cl = last_swing_low[i - 1]
        if not np.isnan(prev_ch) and close[i] > prev_ch and close[i - 1] <= prev_ch:
            bos_bull[i] = True
        if not np.isnan(prev_cl) and close[i] < prev_cl and close[i - 1] >= prev_cl:
            bos_bear[i] = True

    # CHoCH: in uptrend, close breaks below last_swing_low → bearish CHoCH
    #        in downtrend, close breaks above last_swing_high → bullish CHoCH
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if trend[i - 1] == "uptrend" and bos_bear[i]:
            choch_bear[i] = True
        if trend[i - 1] == "downtrend" and bos_bull[i]:
            choch_bull[i] = True

    return pd.DataFrame(
        {
            "swing_high": sh_mask,
            "swing_low": sl_mask,
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,
            "prev_swing_high": prev_swing_high,
            "prev_swing_low": prev_swing_low,
            "trend": trend,
            "bos_bull": bos_bull,
            "bos_bear": bos_bear,
            "choch_bull": choch_bull,
            "choch_bear": choch_bear,
        },
        index=df.index,
    )
