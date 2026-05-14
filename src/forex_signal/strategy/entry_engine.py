"""Entry permission engine — combines all SMC modules + LNN confirmation.

This is the SINGLE place where a BUY/SELL decision is made. Every filter must
agree before a trade is permitted. RSI/LNN are timing confirmation only, not
primary signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_signal.strategy.filters import (
    compute_trend_strength,
    htf_bias_from_m5,
    in_active_session,
)
from forex_signal.strategy.liquidity import detect_sweeps
from forex_signal.strategy.market_structure import compute_structure
from forex_signal.strategy.pullback import detect_pullbacks


@dataclass
class EntryConfig:
    swing_lookback: int = 3
    ema_fast_period: int = 50
    ema_slope_window: int = 5
    adx_period: int = 14
    htf_minutes: int = 60
    pullback_window: int = 4
    pullback_band_atr_mult: float = 0.5
    require_session_filter: bool = True
    require_htf_bias: bool = True
    require_sweep: bool = True
    require_pullback: bool = True
    require_bos_or_choch: bool = False  # off by default — too restrictive on M5
    min_lnn_probability: float = 0.55   # LNN sigmoid output threshold
    sweep_recent_window: int = 3        # how many bars back a sweep counts
    invert_direction: bool = False      # flip BUY<->SELL — for mean-reversion regime


@dataclass
class EntryDecision:
    direction: int           # +1 buy, -1 sell, 0 no trade
    reasons: list[str]
    permission: dict[str, bool]  # per-filter pass/fail for debugging


def compute_all_signals(df: pd.DataFrame, config: EntryConfig) -> pd.DataFrame:
    """Compute every SMC layer once for the whole DataFrame. Returns one big DF
    aligned to `df` with all decision-relevant booleans + values."""
    structure = compute_structure(df, lookback=config.swing_lookback)
    trend_strength = compute_trend_strength(
        df,
        ema_fast_period=config.ema_fast_period,
        slope_window=config.ema_slope_window,
        adx_period=config.adx_period,
    )
    sweeps = detect_sweeps(df, structure)
    pullbacks = detect_pullbacks(
        df,
        structure,
        trend_strength,
        recent_window=config.pullback_window,
        ema_band_atr_mult=config.pullback_band_atr_mult,
    )
    session = in_active_session(df["time"])
    htf_bias = htf_bias_from_m5(df, htf_minutes=config.htf_minutes, ema_period=50)

    out = pd.concat(
        [
            df[["time", "open", "high", "low", "close"]].reset_index(drop=True),
            structure.reset_index(drop=True),
            trend_strength.reset_index(drop=True),
            sweeps.reset_index(drop=True),
            pullbacks.reset_index(drop=True),
        ],
        axis=1,
    )
    out["session_active"] = session.to_numpy()
    out["htf_bias"] = htf_bias.to_numpy()

    # Rolling "recent sweep" — sweep in last N bars (not just current bar)
    out["bearish_sweep_recent"] = (
        out["bearish_sweep"].rolling(config.sweep_recent_window, min_periods=1).max().astype(bool)
    )
    out["bullish_sweep_recent"] = (
        out["bullish_sweep"].rolling(config.sweep_recent_window, min_periods=1).max().astype(bool)
    )

    return out


def evaluate_entry(
    row: pd.Series,
    lnn_probability: float | None,
    config: EntryConfig,
) -> EntryDecision:
    """Evaluate filters for a single bar. Returns EntryDecision."""
    perm: dict[str, bool] = {}
    reasons: list[str] = []

    # Session
    if config.require_session_filter:
        perm["session"] = bool(row.get("session_active", False))
        if not perm["session"]:
            return EntryDecision(0, ["outside London-NY session"], perm)
    else:
        perm["session"] = True

    # HTF bias
    htf = row.get("htf_bias", None)
    if config.require_htf_bias and htf not in ("bullish", "bearish"):
        return EntryDecision(0, ["no HTF bias yet"], perm)

    # Structure trend
    trend = row.get("trend", "range")
    strong_bull = bool(row.get("strong_bull", False))
    strong_bear = bool(row.get("strong_bear", False))
    is_range = bool(row.get("is_range", False))

    if is_range:
        return EntryDecision(0, ["range market — no trade"], perm)

    # Now evaluate bullish and bearish setups
    bull_perm = {
        "structure_bull": trend == "uptrend" or strong_bull,
        "htf_bull": htf == "bullish" if config.require_htf_bias else True,
        "no_strong_bear": not strong_bear,
        "sweep_bull": bool(row.get("bullish_sweep_recent", False)) if config.require_sweep else True,
        "pullback_bull": bool(row.get("bullish_pullback", False)) if config.require_pullback else True,
        "bos_or_choch_bull": bool(row.get("bos_bull", False) or row.get("choch_bull", False)) if config.require_bos_or_choch else True,
    }
    bear_perm = {
        "structure_bear": trend == "downtrend" or strong_bear,
        "htf_bear": htf == "bearish" if config.require_htf_bias else True,
        "no_strong_bull": not strong_bull,
        "sweep_bear": bool(row.get("bearish_sweep_recent", False)) if config.require_sweep else True,
        "pullback_bear": bool(row.get("bearish_pullback", False)) if config.require_pullback else True,
        "bos_or_choch_bear": bool(row.get("bos_bear", False) or row.get("choch_bear", False)) if config.require_bos_or_choch else True,
    }

    bull_ok = all(bull_perm.values())
    bear_ok = all(bear_perm.values())

    # LNN confirmation — agree on direction with at least min probability
    if lnn_probability is not None:
        bull_perm["lnn_confirm"] = lnn_probability >= config.min_lnn_probability
        bear_perm["lnn_confirm"] = (1.0 - lnn_probability) >= config.min_lnn_probability
        bull_ok = bull_ok and bull_perm["lnn_confirm"]
        bear_ok = bear_ok and bear_perm["lnn_confirm"]

    if bull_ok and not bear_ok:
        d = -1 if config.invert_direction else 1
        return EntryDecision(d, ["all bull filters passed (inverted)" if config.invert_direction else "all bull filters passed"], bull_perm)
    if bear_ok and not bull_ok:
        d = 1 if config.invert_direction else -1
        return EntryDecision(d, ["all bear filters passed (inverted)" if config.invert_direction else "all bear filters passed"], bear_perm)
    if bull_ok and bear_ok:
        return EntryDecision(0, ["both directions valid — ambiguous"], {**bull_perm, **bear_perm})

    # Build reasons from failed bull and bear perms
    failed_bull = [k for k, v in bull_perm.items() if not v]
    failed_bear = [k for k, v in bear_perm.items() if not v]
    return EntryDecision(0, [f"failed bull: {failed_bull}; failed bear: {failed_bear}"], {**bull_perm, **bear_perm})
