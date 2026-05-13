"""Live trading loop — Windows + MT5 only.

Strategy: every M5 close + 2s buffer, fetch latest 200 bars, run prediction,
generate signal, manage existing positions, open new trades if eligible.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import json

from forex_signal.config import PROJECT_ROOT, load_config
from forex_signal.data.mt5_client import make_client
from forex_signal.execution.order_manager import OrderManager, OrderManagerConfig
from forex_signal.model.predict import Predictor
from forex_signal.strategy.risk import KillSwitchState, compute_trade_plan
from forex_signal.strategy.signal import generate_signal

log = logging.getLogger(__name__)

STATE_FILE = PROJECT_ROOT / "logs" / "state.json"
TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _seconds_until_next_close(tf: str) -> float:
    interval = TIMEFRAME_SECONDS[tf]
    now = time.time()
    return interval - (now % interval) + 2.0


def run_live(mode: str = "paper") -> int:
    cfg = load_config()
    if cfg.mt5 is None:
        log.error("MT5 credentials missing — fill .env")
        return 2

    seq_len = int(cfg.get("data", "seq_len", default=60))
    pred_horizon = int(cfg.get("data", "pred_horizon", default=5))
    symbol = cfg.symbol
    tf = cfg.timeframe

    predictor = Predictor.load(PROJECT_ROOT / "models" / "lnn_eurusd_m5.pt", seq_len=seq_len)
    client = make_client(prefer_real=True)
    if not client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path):
        log.error("MT5 connect failed")
        return 2

    om = OrderManager(
        client,
        OrderManagerConfig(
            symbol=symbol,
            max_concurrent_positions=int(cfg.get("risk", "max_concurrent_positions", default=2)),
        ),
    )

    state = _load_state()
    today = _today_utc()
    if state.get("day") != today:
        info = client.get_account_info()
        state = {"day": today, "day_start_equity": float(info["equity"])}
        _save_state(state)
    day_start_equity = float(state["day_start_equity"])

    log.info(
        "live loop started: mode=%s symbol=%s tf=%s day_start_equity=%.2f",
        mode, symbol, tf, day_start_equity,
    )

    try:
        while True:
            sleep_for = _seconds_until_next_close(tf)
            log.debug("sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)

            try:
                df = client.fetch_history(symbol, tf, 200)
                pred, atr = predictor.predict_returns(df)
                sig = generate_signal(
                    pred,
                    min_predicted_return_bps=float(cfg.get("signal", "min_predicted_return_bps", default=1.5)),
                    min_directional_consistency=float(cfg.get("signal", "min_directional_consistency", default=0.6)),
                )

                # Day rollover
                today_now = _today_utc()
                if state.get("day") != today_now:
                    info = client.get_account_info()
                    state = {"day": today_now, "day_start_equity": float(info["equity"])}
                    _save_state(state)
                    day_start_equity = float(state["day_start_equity"])

                info = client.get_account_info()
                ks = KillSwitchState(
                    day_start_equity=day_start_equity,
                    current_equity=float(info["equity"]),
                    max_daily_loss_pct=float(cfg.get("risk", "max_daily_loss_pct", default=2.0)),
                )

                # Manage open positions first
                close_results = om.manage_reversal(sig)
                for r in close_results:
                    log.info("close result: ok=%s ticket=%s err=%s", r.success, r.ticket, r.error)

                if sig.direction != 0:
                    log.info(
                        "signal: dir=%s conf=%.2f cum=%.6f atr=%.6f",
                        sig.direction, sig.confidence, sig.cum_return, atr,
                    )

                ok, why = om.can_open_new(ks)
                if sig.direction != 0 and ok:
                    last_close = float(df["close"].iloc[-1])
                    plan = compute_trade_plan(
                        direction=sig.direction,
                        price=last_close,
                        atr=atr,
                        predicted_cum_return=sig.cum_return,
                        lot_size=float(cfg.get("risk", "lot_size", default=0.01)),
                        sl_atr_multiplier=float(cfg.get("risk", "sl_atr_multiplier", default=1.5)),
                        tp_atr_min_multiplier=float(cfg.get("risk", "tp_atr_min_multiplier", default=1.0)),
                        tp_atr_max_multiplier=float(cfg.get("risk", "tp_atr_max_multiplier", default=4.0)),
                    )
                    if mode == "live":
                        r = om.open(plan, last_close)
                        log.info("open: ok=%s ticket=%s err=%s plan=%s", r.success, r.ticket, r.error, plan)
                    else:
                        log.info("[paper] would open: %s", plan)
                elif sig.direction != 0:
                    log.info("not opening: %s", why)
            except Exception:  # noqa: BLE001
                log.exception("loop iteration failed — continuing")
    except KeyboardInterrupt:
        log.info("interrupt — shutting down")
    finally:
        client.shutdown()
    return 0
