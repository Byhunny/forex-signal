import numpy as np

from forex_signal.strategy.signal import generate_signal


def test_signal_buy_when_positive_consistent_above_threshold():
    pred = np.array([0.0005, 0.0004, 0.0003, 0.0002, 0.0001], dtype=np.float32)
    sig = generate_signal(pred, min_predicted_return_bps=1.5, min_directional_consistency=0.6)
    assert sig.direction == 1
    assert sig.confidence == 1.0
    assert sig.cum_return > 0


def test_signal_sell_when_negative_consistent():
    pred = np.array([-0.0006, -0.0001, -0.0001, -0.0001, -0.0001], dtype=np.float32)
    sig = generate_signal(pred, min_predicted_return_bps=1.5, min_directional_consistency=0.6)
    assert sig.direction == -1


def test_signal_zero_when_magnitude_too_small():
    pred = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float32)
    sig = generate_signal(pred, min_predicted_return_bps=1.5)
    assert sig.direction == 0
    assert "magnitude" in sig.reason


def test_signal_zero_when_consistency_too_low():
    # Cumulative big, but split halves
    pred = np.array([0.001, -0.0009, 0.001, -0.0009, 0.001], dtype=np.float32)
    sig = generate_signal(pred, min_predicted_return_bps=1.5, min_directional_consistency=0.8)
    assert sig.direction == 0


def test_signal_invalid_shape():
    sig = generate_signal(np.array([], dtype=np.float32))
    assert sig.direction == 0
