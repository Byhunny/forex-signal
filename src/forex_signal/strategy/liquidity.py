"""Liquidity sweep detection — fake breakouts / stop hunts.

A bearish sweep: this bar's high pokes above the previous swing high, BUT closes below it.
Suggests buy-side liquidity was taken and rejected → bearish setup.
A bullish sweep is symmetric: low pokes below previous swing low, closes above it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_sweeps(df: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with bearish_sweep, bullish_sweep booleans.

    `structure` must come from `market_structure.compute_structure(df)` — uses
    `last_swing_high` and `last_swing_low` from the bar BEFORE the current one
    (so we don't compare against a swing the current bar itself defined).
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)

    # Previous bar's last_swing_* — that's the swing reference we sweep against
    last_sh = structure["last_swing_high"].shift(1).to_numpy(dtype=np.float64)
    last_sl = structure["last_swing_low"].shift(1).to_numpy(dtype=np.float64)

    bearish_sweep = np.zeros(n, dtype=bool)
    bullish_sweep = np.zeros(n, dtype=bool)

    for i in range(n):
        sh = last_sh[i]
        sl = last_sl[i]
        if not np.isnan(sh) and high[i] > sh and close[i] < sh:
            bearish_sweep[i] = True
        if not np.isnan(sl) and low[i] < sl and close[i] > sl:
            bullish_sweep[i] = True

    return pd.DataFrame(
        {"bearish_sweep": bearish_sweep, "bullish_sweep": bullish_sweep},
        index=df.index,
    )
