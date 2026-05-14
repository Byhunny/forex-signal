"""Battle royale — run all 10 top strategies in parallel on a single MT5 account.

Each strategy gets its own magic number so MT5 history can be sliced per-strategy
to compute a leaderboard. Positions are tracked independently (1 max per strategy
by default). Global daily kill switch protects the account.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from forex_signal.config import PROJECT_ROOT, load_config
from forex_signal.data.features import compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.model.predict import Predictor
from forex_signal.strategy.entry_engine import EntryConfig, compute_all_signals, evaluate_entry
from forex_signal.strategy.risk import KillSwitchState, compute_trade_plan

log = logging.getLogger("battle_royale")

STATE_FILE = PROJECT_ROOT / "logs" / "battle_state.json"
DECISIONS_LOG = PROJECT_ROOT / "logs" / "battle_decisions.log"

TIMEFRAME_SECONDS = {"M5": 300, "M15": 900, "M30": 1800, "H1": 3600}

# Per-symbol pip + contract size for SL/TP price math
PIP_VALUE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "USDSEK": 0.0001, "USDCNH": 0.0001,
    "USDJPY": 0.01, "EURJPY": 0.01,
    "GOLD": 0.10, "SILVER": 0.01,
    "BTCUSD": 1.0,
    "US500Cash": 0.1, "US100Cash": 0.1, "GER40Cash": 1.0, "US30Cash": 1.0,
}


@dataclass
class Strategy:
    """One contender in the battle royale."""

    name: str
    symbol: str
    timeframe: str
    model_path: Path
    magic: int
    entry_cfg: EntryConfig
    lot_size: float = 0.01
    max_positions: int = 1
    sl_atr_mult: float = 1.5
    tp_atr_min_mult: float = 0.6
    tp_atr_max_mult: float = 1.5
    seq_len: int = 50

    # Runtime
    predictor: "Predictor | None" = field(default=None, repr=False)
    last_bar_time: "datetime | None" = field(default=None, repr=False)


# === TOP 10 BATTLE ROYALE CONTENDERS ===
# Derived from the sweep of 19 symbols × 3 TFs × 4 strategies (228 backtests).
# Magic numbers: 26051001..26051010 (date-based, unique per strategy slot).
TOP_10_STRATEGIES = [
    # (name, symbol, tf, model, magic, entry_kwargs)
    ("01_USDSEK_M30_lnn_very_strong", "USDSEK", "M30", "sweep_USDSEK_M30.pt", 26051001,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.65)),
    ("02_USDSEK_M30_lnn_strong", "USDSEK", "M30", "sweep_USDSEK_M30.pt", 26051002,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60)),
    ("03_SILVER_M15_trend_lnn", "SILVER", "M15", "sweep_SILVER_M15.pt", 26051003,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("04_USDCNH_M30_lnn_very_strong", "USDCNH", "M30", "sweep_USDCNH_M30.pt", 26051004,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.65)),
    ("05_SILVER_M15_lnn_strong", "SILVER", "M15", "sweep_SILVER_M15.pt", 26051005,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60)),
    ("06_EURJPY_M30_smc_strict", "EURJPY", "M30", "sweep_EURJPY_M30.pt", 26051006,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10)),
    ("07_EURUSD_M30_smc_strict", "EURUSD", "M30", "sweep_EURUSD_M30.pt", 26051007,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10)),
    ("08_AUDUSD_M30_smc_strict", "AUDUSD", "M30", "sweep_AUDUSD_M30.pt", 26051008,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10)),
    ("09_EURCAD_M30_lnn_strong", "EURCAD", "M30", "sweep_EURCAD_M30.pt", 26051009,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60)),
    ("10_US100Cash_M30_trend_lnn", "US100Cash", "M30", "sweep_US100Cash_M30.pt", 26051010,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
]


def build_strategies() -> list[Strategy]:
    out = []
    for name, sym, tf, model, magic, ekw in TOP_10_STRATEGIES:
        out.append(Strategy(
            name=name,
            symbol=sym,
            timeframe=tf,
            model_path=PROJECT_ROOT / "models" / model,
            magic=magic,
            entry_cfg=EntryConfig(**ekw),
        ))
    return out


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _seconds_until_next_m5_close() -> float:
    now = time.time()
    return 300 - (now % 300) + 2.0


def _tf_aligned(tf: str, now: datetime) -> bool:
    m = now.minute
    if tf == "M5": return m % 5 == 0
    if tf == "M15": return m % 15 == 0
    if tf == "M30": return m % 30 == 0
    if tf == "H1": return m == 0
    return False


def _log_decision(strat: Strategy, decision_str: str) -> None:
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_LOG, "a") as f:
        f.write(f"{ts}  {strat.name}  {decision_str}\n")


def _check_and_act(strat: Strategy, client, mode: str, kill_switch: KillSwitchState) -> None:
    """Run one decision cycle for a single strategy."""
    try:
        df = client.fetch_history(strat.symbol, strat.timeframe, 200)
    except Exception as e:
        log.warning("fetch_history failed for %s: %s", strat.name, e)
        return

    if len(df) < strat.seq_len + 30:
        return

    # Inference
    try:
        pred_ret, prob, atr = strat.predictor.predict(df)
    except Exception as e:
        log.warning("predict failed for %s: %s", strat.name, e)
        return

    # SMC + entry decision
    try:
        features = compute_features(df)
        smc = compute_all_signals(features, strat.entry_cfg)
        decision = evaluate_entry(smc.iloc[-1], prob, strat.entry_cfg)
    except Exception as e:
        log.warning("decision failed for %s: %s", strat.name, e)
        return

    # Manage existing positions: close on reversal
    my_positions = [p for p in client.get_positions(strat.symbol) if p.get("magic") == strat.magic]
    if my_positions and decision.direction != 0 and abs(prob - 0.5) >= 0.2:
        for pos in my_positions:
            pos_dir = 1 if str(pos.get("type")).lower().endswith("buy") or pos.get("type") in (0, "buy") else -1
            if pos_dir != decision.direction:
                if mode == "live":
                    r = client.close_position(int(pos["ticket"]))
                    _log_decision(strat, f"REVERSAL_CLOSE ticket={pos['ticket']} ok={r.success} err={r.error}")
                else:
                    _log_decision(strat, f"[paper] would close ticket={pos['ticket']} on reversal")

    # Eligibility for new entry
    if kill_switch.tripped:
        return
    if decision.direction == 0:
        return
    if len(my_positions) >= strat.max_positions:
        return
    if atr <= 0:
        return

    last_close = float(df["close"].iloc[-1])
    cum_return = float(pred_ret.sum())
    plan = compute_trade_plan(
        direction=decision.direction,
        price=last_close,
        atr=atr,
        predicted_cum_return=cum_return,
        lot_size=strat.lot_size,
        sl_atr_multiplier=strat.sl_atr_mult,
        tp_atr_min_multiplier=strat.tp_atr_min_mult,
        tp_atr_max_multiplier=strat.tp_atr_max_mult,
    )

    direction_str = "BUY" if decision.direction == 1 else "SELL"
    if mode == "live":
        r = client.place_order(
            symbol=strat.symbol,
            order_type="buy" if decision.direction == 1 else "sell",
            lot=strat.lot_size,
            sl_price=plan.sl_price,
            tp_price=plan.tp_price,
            magic=strat.magic,
            comment=strat.name[:31],  # MT5 comment max ~32 chars
        )
        _log_decision(strat, f"OPEN {direction_str} prob={prob:.3f} sl={plan.sl_price:.5f} tp={plan.tp_price:.5f} ok={r.success} ticket={r.ticket} err={r.error}")
    else:
        _log_decision(strat, f"[paper] OPEN {direction_str} prob={prob:.3f} price={last_close:.5f} sl={plan.sl_price:.5f} tp={plan.tp_price:.5f}")


def run_battle_royale(mode: str = "paper") -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    if cfg.mt5 is None:
        log.error("MT5 credentials missing — fill .env")
        return 2

    strategies = build_strategies()

    # Load all predictors at startup (one per unique model path)
    cache: dict[str, Predictor] = {}
    for s in strategies:
        if not s.model_path.exists():
            log.warning("model not found for %s: %s — skipping", s.name, s.model_path)
            continue
        key = str(s.model_path)
        if key not in cache:
            cache[key] = Predictor.load(s.model_path, seq_len=s.seq_len)
        s.predictor = cache[key]
    strategies = [s for s in strategies if s.predictor is not None]
    log.info("loaded %d strategies", len(strategies))

    client = make_client(prefer_real=True)
    if not client.connect(cfg.mt5.login, cfg.mt5.password, cfg.mt5.server, cfg.mt5.path):
        log.error("MT5 connect failed")
        return 2

    # Day-rollover state for kill switch
    state = _load_state()
    today = _today_utc()
    if state.get("day") != today:
        info = client.get_account_info()
        state = {"day": today, "day_start_equity": float(info["equity"])}
        _save_state(state)

    log.info("BATTLE ROYALE started — mode=%s — %d contenders", mode, len(strategies))
    for s in strategies:
        log.info("  contender: %s (magic=%d)", s.name, s.magic)

    try:
        while True:
            sleep_for = _seconds_until_next_m5_close()
            log.debug("sleeping %.0fs to next M5 boundary", sleep_for)
            time.sleep(sleep_for)

            now = datetime.now(timezone.utc)

            # Day rollover
            today_now = _today_utc()
            if state.get("day") != today_now:
                info = client.get_account_info()
                state = {"day": today_now, "day_start_equity": float(info["equity"])}
                _save_state(state)

            info = client.get_account_info()
            ks = KillSwitchState(
                day_start_equity=float(state["day_start_equity"]),
                current_equity=float(info["equity"]),
                max_daily_loss_pct=2.0,
            )
            if ks.tripped:
                log.warning("KILL SWITCH TRIPPED — daily PnL %.2f%%", ks.daily_pnl_pct)

            n_open = sum(1 for s in strategies for _ in client.get_positions(s.symbol) if _.get("magic") == s.magic)
            log.info("tick %s UTC | equity=$%.2f | daily=%+.2f%% | open_positions=%d",
                     now.strftime("%H:%M"), info["equity"], ks.daily_pnl_pct, n_open)

            # Fire each strategy whose TF aligned on this tick
            for s in strategies:
                if not _tf_aligned(s.timeframe, now):
                    continue
                _check_and_act(s, client, mode, ks)

    except KeyboardInterrupt:
        log.info("interrupt — shutting down")
    finally:
        client.shutdown()
    return 0
