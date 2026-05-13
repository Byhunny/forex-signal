from pathlib import Path

from forex_signal.config import load_config


def test_load_config_smoke():
    cfg = load_config()
    assert cfg.symbol
    assert cfg.timeframe
    assert isinstance(cfg.raw, dict)
    assert cfg.get("risk", "lot_size") == 0.01
    assert cfg.get("non", "existent", default="X") == "X"
