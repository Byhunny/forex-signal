"""Feature engineering: OHLCV -> 12 normalized features + multi-step return targets.

All functions are pure: same input -> same output. No global state. No NaN past the warmup window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",
    "log_ret_1",
    "hl_range",
    "co_range",
    "rsi_14",
    "atr_14",
    "bb_position",
    "macd_diff",
    "ema_ratio",
    "volume_z",
    "hour_sin",
    "hour_cos",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features. Input must have columns: time, open, high, low, close, tick_volume.

    Returns a new DataFrame with the original columns plus FEATURE_COLUMNS.
    Drops rows that are NaN due to warmup.
    """
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    vol = out["tick_volume"].astype(float)

    out["ret_1"] = close.pct_change()
    out["log_ret_1"] = np.log(close / close.shift(1))
    out["hl_range"] = (high - low) / close
    out["co_range"] = (close - out["open"]) / out["open"]

    out["rsi_14"] = _rsi(close, 14) / 100.0
    out["atr_14"] = _atr(high, low, close, 14)

    bb_mid, bb_up, bb_lo = _bollinger(close, 20, 2.0)
    band = (bb_up - bb_lo).replace(0, np.nan)
    out["bb_position"] = (close - bb_mid) / band

    out["macd_diff"] = _macd(close) / close

    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    out["ema_ratio"] = ema_fast / ema_slow - 1.0

    vol_mean = vol.rolling(50, min_periods=10).mean()
    vol_std = vol.rolling(50, min_periods=10).std().replace(0, np.nan)
    out["volume_z"] = (vol - vol_mean) / vol_std

    times = pd.to_datetime(out["time"], utc=True, errors="coerce")
    hours = times.dt.hour.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    out = out.dropna(subset=FEATURE_COLUMNS + ["atr_14"]).reset_index(drop=True)
    return out


@dataclass
class WindowedDataset:
    X: np.ndarray  # (samples, seq_len, n_features)
    y: np.ndarray  # (samples, pred_horizon) — z-score normalized targets
    times: np.ndarray  # (samples,) the timestamp of the last bar in each window
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float = 0.0  # so predictions can be denormalized back to raw log returns
    target_std: float = 1.0


def build_windows(
    features_df: pd.DataFrame,
    seq_len: int = 60,
    pred_horizon: int = 5,
    feature_columns: list[str] | None = None,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> WindowedDataset:
    """Slice features into sliding windows + compute multi-step return targets.

    The target for window ending at index t is the per-step log returns at t+1..t+pred_horizon.
    """
    cols = feature_columns or FEATURE_COLUMNS
    if "close" not in features_df.columns:
        raise ValueError("features_df must include 'close' for target computation")

    arr = features_df[cols].to_numpy(dtype=np.float32)
    close = features_df["close"].to_numpy(dtype=np.float64)
    times = features_df["time"].to_numpy()

    n = len(features_df)
    last_idx = n - pred_horizon - 1
    if last_idx < seq_len:
        raise ValueError(
            f"not enough data: need at least {seq_len + pred_horizon + 1} rows, got {n}"
        )

    if means is None or stds is None:
        train_slice = arr[: int(n * 0.7)]
        means = train_slice.mean(axis=0)
        stds = train_slice.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)

    arr_norm = (arr - means) / stds

    n_samples = last_idx - seq_len + 1
    X = np.empty((n_samples, seq_len, len(cols)), dtype=np.float32)
    y = np.empty((n_samples, pred_horizon), dtype=np.float32)
    t_out = np.empty(n_samples, dtype=times.dtype)

    for i in range(n_samples):
        end = i + seq_len  # exclusive
        X[i] = arr_norm[i:end]
        future = close[end : end + pred_horizon]
        prev = close[end - 1 : end - 1 + pred_horizon]
        y[i] = np.log(future / prev).astype(np.float32)
        t_out[i] = times[end - 1]

    # Normalize targets using training slice stats — gives MSE meaningful gradient
    # (raw log returns are ~1e-4, MSE on those is ~1e-8 and rewards "predict zero")
    train_y = y[: int(n_samples * 0.7)]
    target_mean = float(train_y.mean())
    target_std = float(train_y.std()) or 1.0
    y_norm = ((y - target_mean) / target_std).astype(np.float32)

    return WindowedDataset(
        X=X, y=y_norm, times=t_out,
        feature_means=means, feature_stds=stds,
        target_mean=target_mean, target_std=target_std,
    )
