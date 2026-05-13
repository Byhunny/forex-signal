import numpy as np
import pandas as pd
import pytest

from forex_signal.data.features import FEATURE_COLUMNS, build_windows, compute_features


def _synthetic(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.001, n)
    close = 1.10 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.0008, n))
    low = close * (1 - rng.uniform(0, 0.0008, n))
    open_ = close * (1 + rng.normal(0, 0.0003, n))
    vol = rng.uniform(100, 1000, n)
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": open_, "high": high, "low": low,
        "close": close, "tick_volume": vol,
    })


def test_compute_features_has_all_columns_and_no_nans():
    df = _synthetic(500)
    out = compute_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, col
    assert not out[FEATURE_COLUMNS].isna().any().any(), "NaNs remain in features"


def test_compute_features_rejects_missing_columns():
    df = _synthetic(100).drop(columns=["tick_volume"])
    with pytest.raises(ValueError):
        compute_features(df)


def test_build_windows_shapes_and_target():
    df = _synthetic(500)
    feats = compute_features(df)
    ds = build_windows(feats, seq_len=30, pred_horizon=4)
    n_samples, seq_len, n_features = ds.X.shape
    assert seq_len == 30
    assert n_features == len(FEATURE_COLUMNS)
    assert ds.y.shape == (n_samples, 4)
    assert ds.feature_means.shape == (n_features,)
    assert ds.feature_stds.shape == (n_features,)
    # Standardization roughly works: training slice features ~mean 0
    train_slice = ds.X[: int(n_samples * 0.7)]
    mean_abs = np.abs(train_slice.mean(axis=(0, 1))).mean()
    assert mean_abs < 1.0  # loose: not zero because we standardize on a different slice, but bounded


def test_build_windows_insufficient_data():
    df = _synthetic(50)
    feats = compute_features(df)
    with pytest.raises(ValueError):
        build_windows(feats, seq_len=60, pred_horizon=5)
