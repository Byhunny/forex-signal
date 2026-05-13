import pytest

from forex_signal.strategy.risk import KillSwitchState, compute_trade_plan


def test_buy_plan_basic():
    plan = compute_trade_plan(
        direction=1, price=1.1000, atr=0.0010, predicted_cum_return=0.0015,
        lot_size=0.01, sl_atr_multiplier=1.5,
        tp_atr_min_multiplier=1.0, tp_atr_max_multiplier=4.0,
    )
    assert plan.direction == 1
    assert plan.lot == 0.01
    assert plan.sl_distance == pytest.approx(0.0015)
    # TP capped by predicted_move = 0.0015 * 1.10 = 0.00165, > atr_min(1.0*0.001) and < atr_max(4.0*0.001=0.004)
    assert plan.tp_distance == pytest.approx(0.00165, rel=1e-3)
    assert plan.sl_price < 1.1000 < plan.tp_price


def test_sell_plan():
    plan = compute_trade_plan(
        direction=-1, price=1.2000, atr=0.0008, predicted_cum_return=-0.002,
    )
    assert plan.direction == -1
    assert plan.sl_price > 1.2000 > plan.tp_price


def test_tp_floored_when_predicted_move_too_small():
    plan = compute_trade_plan(
        direction=1, price=1.1000, atr=0.001, predicted_cum_return=0.0001,
        tp_atr_min_multiplier=1.0, tp_atr_max_multiplier=4.0,
    )
    # predicted_move = 0.0001 * 1.1 = 0.00011 — below atr floor (1.0 * 0.001 = 0.001)
    assert plan.tp_distance == pytest.approx(0.001)


def test_tp_capped_when_predicted_move_too_large():
    plan = compute_trade_plan(
        direction=1, price=1.1000, atr=0.001, predicted_cum_return=0.05,
        tp_atr_min_multiplier=1.0, tp_atr_max_multiplier=4.0,
    )
    # predicted_move = 0.055 — above atr cap (4.0 * 0.001 = 0.004)
    assert plan.tp_distance == pytest.approx(0.004)


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        compute_trade_plan(direction=0, price=1.1, atr=0.001, predicted_cum_return=0.001)


def test_invalid_atr_raises():
    with pytest.raises(ValueError):
        compute_trade_plan(direction=1, price=1.1, atr=0.0, predicted_cum_return=0.001)


def test_kill_switch_trips_at_threshold():
    ks = KillSwitchState(day_start_equity=10000, current_equity=9799, max_daily_loss_pct=2.0)
    assert ks.tripped is True
    ks2 = KillSwitchState(day_start_equity=10000, current_equity=9801, max_daily_loss_pct=2.0)
    assert ks2.tripped is False
