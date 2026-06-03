"""Battle royale — run all 10 top strategies in parallel on a single MT5 account.

Each strategy gets its own magic number so MT5 history can be sliced per-strategy
to compute a leaderboard. Positions are tracked independently (1 max per strategy
by default). Global daily kill switch protects the account.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from forex_signal.config import PROJECT_ROOT, load_config
from forex_signal.data.features import compute_features
from forex_signal.data.mt5_client import make_client
from forex_signal.model.predict import Predictor
from forex_signal.notifier.telegram import is_configured as telegram_configured, notify as tg_notify, start_command_listener
from forex_signal.strategy.entry_engine import EntryConfig, compute_all_signals, evaluate_entry
from forex_signal.strategy.risk import KillSwitchState, compute_trade_plan

log = logging.getLogger("battle_royale")

# MT5 Python package is NOT thread-safe — serialize all MT5 calls behind this lock
_MT5_LOCK = threading.RLock()

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
    exit_mode: str = "FixedTP"  # FixedTP | CLOSE | HALFTP | TRAIL15 | TRAIL12 | T2_TIMEEXIT
    half_peak_tp_pips: float = 0.0  # used by HALFTP mode
    pip_value: float = 0.0001       # for HALFTP/trail price conversion
    lot_size: float = 0.01
    max_positions: int = 1
    sl_atr_mult: float = 1.5
    tp_atr_min_mult: float = 0.6
    tp_atr_max_mult: float = 1.5
    seq_len: int = 50
    # T2_TIMEEXIT mode: force-close after N bars of position TF
    max_hold_bars: int = 0
    # Entry vol filter — only enter if current bar |close-open| >= mult × spread_price
    vol_filter_mult: float = 0.0
    spread_pips_estimate: float = 5.0  # used by vol filter to derive spread_price

    # Runtime
    predictor: "Predictor | None" = field(default=None, repr=False)
    last_bar_time: "datetime | None" = field(default=None, repr=False)

    @property
    def trail_atr_mult(self) -> float:
        if self.exit_mode == "TRAIL12": return 1.2
        if self.exit_mode == "TRAIL15": return 1.5
        return 0.0  # not used


# === TOP 27 PROFITABLE CONTENDERS (from full backtest sweep, realistic spreads) ===
# Each contender: (Symbol, TF, BaseStrategy, ExitMode, half_peak_tp_pips)
# Same Symbol+Strategy can appear in multiple modes — each gets unique magic.
# half_peak_tp_pips = median peak distance / 2 (from peak_distance.json), only used by HALFTP mode.

# Backwards-compat alias (kept for tests / other code)
# Derived from the sweep of 19 symbols × 3 TFs × 4 strategies (228 backtests).
# Slots 1-10: profitable in backtest (PF > 1). Slots 11-20: high WR (>=66%) candidates
# selected for volume + diversity, mostly break-even (PF 0.58-1.41).
# Magic numbers: 26051001..26051020 (date-based, unique per strategy slot).
TOP_10_STRATEGIES = [
    # (name, symbol, tf, model, magic, entry_kwargs)
    # --- TIER 1: BACKTEST-PROFITABLE (PF > 1) ---
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
    # --- TIER 2: HIGH WR + VOLUME PICKS (break-even to slightly losing in backtest) ---
    ("11_USDSEK_M5_lnn_very_strong", "USDSEK", "M5", "sweep_USDSEK_M5.pt", 26051011,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.65)),
    ("12_USDSEK_M30_trend_lnn", "USDSEK", "M30", "sweep_USDSEK_M30.pt", 26051012,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("13_USDSEK_M5_lnn_strong", "USDSEK", "M5", "sweep_USDSEK_M5.pt", 26051013,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60)),
    ("14_SILVER_M5_trend_lnn", "SILVER", "M5", "sweep_SILVER_M5.pt", 26051014,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("15_SILVER_M30_trend_lnn", "SILVER", "M30", "sweep_SILVER_M30.pt", 26051015,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("16_EURJPY_M15_trend_lnn", "EURJPY", "M15", "sweep_EURJPY_M15.pt", 26051016,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("17_BTCUSD_M30_trend_lnn", "BTCUSD", "M30", "sweep_BTCUSD_M30.pt", 26051017,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("18_BTCUSD_M15_trend_lnn", "BTCUSD", "M15", "sweep_BTCUSD_M15.pt", 26051018,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("19_GOLD_M15_trend_lnn", "GOLD", "M15", "sweep_GOLD_M15.pt", 26051019,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55)),
    ("20_USDSEK_M15_trend_lnn", "USDSEK", "M15", "sweep_USDSEK_M15.pt", 26051020,
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


# === M5-ONLY CONTENDERS (user request: only M5 + new GOLD M5 scalper) ===
TOP_M5_CONTENDERS: list[tuple] = [
    # (name, symbol, tf, model, magic, entry_kwargs, exit_mode, half_tp_pips, max_hold_bars, vol_filter_mult, spread_pips)
    # 1) New GOLD M5 LNN sl=50 T2 winner (PF 1.26, +$44.10, 10-min hold, vol filter)
    ("28_GOLD_M5_LNN_sl50_T2_vol1", "GOLD", "M5", "gold_m5_lnn_sl50.pt", 26052028,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55),
        "T2_TIMEEXIT", 0.0, 2, 1.0, 6.76),
    # 2) Existing SILVER M5 HALFTP (slot 09 from original 27)
    ("09_SILVER_M5_trend_lnn_HALFTP", "SILVER", "M5", "sweep_SILVER_M5.pt", 26052009,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55),
        "HALFTP", 237.0, 0, 0.0, 8.0),
    # 3) Existing SILVER M5 CLOSE (slot 10 from original 27)
    ("10_SILVER_M5_trend_lnn_CLOSE", "SILVER", "M5", "sweep_SILVER_M5.pt", 26052010,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55),
        "CLOSE", 0.0, 0, 0.0, 8.0),
]


# === ORIGINAL TOP 27 (kept for reference) ===
TOP_27_CONTENDERS: list[tuple] = [
    # (name, symbol, tf, model, magic, entry_kwargs, exit_mode, half_tp_pips)
    # --- EURJPY M15 trend_lnn ---
    ("01_EURJPY_M15_trend_lnn_HALFTP", "EURJPY", "M15", "sweep_EURJPY_M15.pt", 26052001,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 260.0),
    ("02_EURJPY_M15_trend_lnn_CLOSE", "EURJPY", "M15", "sweep_EURJPY_M15.pt", 26052002,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    # --- SILVER M15 trend_lnn ---
    ("03_SILVER_M15_trend_lnn_CLOSE", "SILVER", "M15", "sweep_SILVER_M15.pt", 26052003,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    ("05_SILVER_M15_trend_lnn_HALFTP", "SILVER", "M15", "sweep_SILVER_M15.pt", 26052005,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 1052.0),
    # --- SILVER M30 trend_lnn ---
    ("04_SILVER_M30_trend_lnn_CLOSE", "SILVER", "M30", "sweep_SILVER_M30.pt", 26052004,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    ("22_SILVER_M30_trend_lnn_HALFTP", "SILVER", "M30", "sweep_SILVER_M30.pt", 26052022,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 124.0),
    # --- GOLD M15 trend_lnn ---
    ("06_GOLD_M15_trend_lnn_HALFTP", "GOLD", "M15", "sweep_GOLD_M15.pt", 26052006,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 4106.0),
    ("07_GOLD_M15_trend_lnn_CLOSE", "GOLD", "M15", "sweep_GOLD_M15.pt", 26052007,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    ("08_GOLD_M15_trend_lnn_FixedTP", "GOLD", "M15", "sweep_GOLD_M15.pt", 26052008,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "FixedTP", 0.0),
    # --- SILVER M5 trend_lnn ---
    ("09_SILVER_M5_trend_lnn_HALFTP", "SILVER", "M5", "sweep_SILVER_M5.pt", 26052009,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 237.0),
    ("10_SILVER_M5_trend_lnn_CLOSE", "SILVER", "M5", "sweep_SILVER_M5.pt", 26052010,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    # --- EURJPY M30 smc_strict ---
    ("11_EURJPY_M30_smc_strict_FixedTP", "EURJPY", "M30", "sweep_EURJPY_M30.pt", 26052011,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "FixedTP", 0.0),
    # --- SILVER M15 lnn_strong ---
    ("12_SILVER_M15_lnn_strong_FixedTP", "SILVER", "M15", "sweep_SILVER_M15.pt", 26052012,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60), "FixedTP", 0.0),
    # --- BTCUSD ---
    ("13_BTCUSD_M30_trend_lnn_CLOSE", "BTCUSD", "M30", "sweep_BTCUSD_M30.pt", 26052013,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    ("14_BTCUSD_M15_trend_lnn_HALFTP", "BTCUSD", "M15", "sweep_BTCUSD_M15.pt", 26052014,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 1534.0),
    # --- US100Cash M30 trend_lnn ---
    ("15_US100Cash_M30_trend_lnn_HALFTP", "US100Cash", "M30", "sweep_US100Cash_M30.pt", 26052015,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "HALFTP", 12666.0),
    ("16_US100Cash_M30_trend_lnn_CLOSE", "US100Cash", "M30", "sweep_US100Cash_M30.pt", 26052016,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "CLOSE", 0.0),
    ("27_US100Cash_M30_trend_lnn_FixedTP", "US100Cash", "M30", "sweep_US100Cash_M30.pt", 26052027,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.55), "FixedTP", 0.0),
    # --- AUDUSD M30 smc_strict ---
    ("17_AUDUSD_M30_smc_strict_HALFTP", "AUDUSD", "M30", "sweep_AUDUSD_M30.pt", 26052017,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "HALFTP", 112.0),
    ("18_AUDUSD_M30_smc_strict_CLOSE", "AUDUSD", "M30", "sweep_AUDUSD_M30.pt", 26052018,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "CLOSE", 0.0),
    # --- EURCAD M30 lnn_strong ---
    ("19_EURCAD_M30_lnn_strong_CLOSE", "EURCAD", "M30", "sweep_EURCAD_M30.pt", 26052019,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60), "CLOSE", 0.0),
    ("20_EURCAD_M30_lnn_strong_FixedTP", "EURCAD", "M30", "sweep_EURCAD_M30.pt", 26052020,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60), "FixedTP", 0.0),
    ("24_EURCAD_M30_lnn_strong_TRAIL15", "EURCAD", "M30", "sweep_EURCAD_M30.pt", 26052024,
        dict(require_session_filter=False, require_htf_bias=False, require_sweep=False,
             require_pullback=False, min_lnn_probability=0.60), "TRAIL15", 0.0),
    # --- EURUSD M30 smc_strict ---
    ("21_EURUSD_M30_smc_strict_TRAIL15", "EURUSD", "M30", "sweep_EURUSD_M30.pt", 26052021,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "TRAIL15", 0.0),
    ("23_EURUSD_M30_smc_strict_FixedTP", "EURUSD", "M30", "sweep_EURUSD_M30.pt", 26052023,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "FixedTP", 0.0),
    ("25_EURUSD_M30_smc_strict_TRAIL12", "EURUSD", "M30", "sweep_EURUSD_M30.pt", 26052025,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "TRAIL12", 0.0),
    ("26_EURUSD_M30_smc_strict_HALFTP", "EURUSD", "M30", "sweep_EURUSD_M30.pt", 26052026,
        dict(require_session_filter=False, require_htf_bias=True, require_sweep=True,
             require_pullback=False, min_lnn_probability=0.55, sweep_recent_window=10), "HALFTP", 506.0),
]


def build_contenders() -> list[Strategy]:
    """Build M5-only contenders (current production setup)."""
    out = []
    for tup in TOP_M5_CONTENDERS:
        name, sym, tf, model, magic, ekw, mode, half_tp, max_hold, vol_mult, spread_p = tup
        out.append(Strategy(
            name=name,
            symbol=sym,
            timeframe=tf,
            model_path=PROJECT_ROOT / "models" / model,
            magic=magic,
            entry_cfg=EntryConfig(**ekw),
            exit_mode=mode,
            half_peak_tp_pips=half_tp,
            pip_value=PIP_VALUE.get(sym, 0.0001),
            max_hold_bars=max_hold,
            vol_filter_mult=vol_mult,
            spread_pips_estimate=spread_p,
        ))
    return out


def build_contenders_full() -> list[Strategy]:
    """Legacy: build all 27 (kept for reference)."""
    out = []
    for tup in TOP_27_CONTENDERS:
        name, sym, tf, model, magic, ekw, mode, half_tp = tup
        out.append(Strategy(
            name=name,
            symbol=sym,
            timeframe=tf,
            model_path=PROJECT_ROOT / "models" / model,
            magic=magic,
            entry_cfg=EntryConfig(**ekw),
            exit_mode=mode,
            half_peak_tp_pips=half_tp,
            pip_value=PIP_VALUE.get(sym, 0.0001),
        ))
    return out


# === Telegram command handlers ===

def _server_now(client) -> "datetime":
    """Get broker's wall-clock time via latest tick — broker can be in different TZ than UTC."""
    import MetaTrader5 as _mt5
    from datetime import datetime, timezone
    with _MT5_LOCK:
        tick = _mt5.symbol_info_tick("EURUSD")
    if tick and tick.time:
        return datetime.fromtimestamp(tick.time, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _status_text(client, strategies: list[Strategy], state: dict) -> str:
    import MetaTrader5 as _mt5
    from datetime import datetime, timedelta, timezone
    with _MT5_LOCK:
        info = client.get_account_info()
    eq = float(info["equity"])
    day_start = float(state.get("day_start_equity", eq))
    daily_pct = (eq - day_start) / day_start * 100.0 if day_start else 0.0

    our_magics = {s.magic for s in strategies}
    with _MT5_LOCK:
        open_pos = [p for p in (client.get_positions() or []) if p.get("magic") in our_magics]

    # Use SERVER time for the window — broker can be in UTC+N
    server_now = _server_now(client)
    day_start_server = server_now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Pad on both sides for safety
    with _MT5_LOCK:
        deals = _mt5.history_deals_get(day_start_server - timedelta(hours=1), server_now + timedelta(hours=1)) or ()
    closed: dict[int, dict] = {}
    for d in deals:
        if d.magic not in our_magics:
            continue
        if d.entry != 1:  # only count "OUT" deals
            continue
        # Filter to today server-time
        if datetime.fromtimestamp(d.time, tz=timezone.utc) < day_start_server:
            continue
        prev = closed.get(d.position_id, {"pnl": 0.0})
        closed[d.position_id] = {
            "pnl": prev["pnl"] + d.profit + d.swap + d.commission,
            "reason": d.reason,
            "magic": d.magic,
        }

    # MT5 DEAL_REASON: 0=client, 1=mobile, 2=web, 3=expert, 4=SL, 5=TP, 6=SO, 7=rollover
    wins = sum(1 for c in closed.values() if c["pnl"] > 0)
    losses = sum(1 for c in closed.values() if c["pnl"] <= 0)
    tp_n = sum(1 for c in closed.values() if c["reason"] == 5)
    sl_n = sum(1 for c in closed.values() if c["reason"] == 4)
    other_n = len(closed) - tp_n - sl_n
    pnl_today = sum(c["pnl"] for c in closed.values())
    win_rate = (wins / len(closed) * 100) if closed else 0.0

    txt = "📊 *Status*\n"
    txt += f"Equity: *${eq:.2f}* ({daily_pct:+.2f}%)\n"
    txt += f"Day baseline: ${day_start:.2f}\n"
    txt += f"Open positions: *{len(open_pos)}* / {len(strategies)}\n"
    txt += f"\n*Today closed: {len(closed)}*\n"
    txt += f"  ✅ {wins}  ❌ {losses}  WR {win_rate:.0f}%\n"
    txt += f"  🎯 TP: {tp_n}  🛑 SL: {sl_n}  📤 Other: {other_n}\n"
    txt += f"  P&L: *${pnl_today:+.2f}*"

    if open_pos:
        txt += "\n\n*Open now:*"
        for p in open_pos:
            name = next((s.name for s in strategies if s.magic == p["magic"]), f"magic={p['magic']}")
            side = "BUY" if str(p.get("type")).lower().endswith("buy") or p.get("type") in (0, "buy") else "SELL"
            txt += f"\n• `{name[:32]}` {side} {p['symbol']} ${p.get('profit', 0):+.2f}"
    return txt


def _leaderboard_text(client, strategies: list[Strategy]) -> str:
    import MetaTrader5 as _mt5
    from datetime import datetime, timedelta, timezone
    server_now = _server_now(client)
    since = server_now - timedelta(days=30)
    with _MT5_LOCK:
        deals = _mt5.history_deals_get(since, server_now + timedelta(hours=1)) or ()

    our_magics = {s.magic: s for s in strategies}
    pos_pnl: dict[int, dict] = {}
    for d in deals:
        if d.magic not in our_magics:
            continue
        if d.position_id not in pos_pnl:
            pos_pnl[d.position_id] = {"magic": d.magic, "pnl": 0.0, "deals": 0}
        pos_pnl[d.position_id]["pnl"] += d.profit + d.swap + d.commission
        pos_pnl[d.position_id]["deals"] += 1

    per_magic: dict[int, dict] = {}
    for pid, p in pos_pnl.items():
        if p["deals"] < 2:
            continue  # not yet closed
        m = p["magic"]
        per_magic.setdefault(m, {"trades": 0, "wins": 0, "pnl": 0.0})
        per_magic[m]["trades"] += 1
        if p["pnl"] > 0:
            per_magic[m]["wins"] += 1
        per_magic[m]["pnl"] += p["pnl"]

    rows = []
    for s in strategies:
        d = per_magic.get(s.magic, {"trades": 0, "wins": 0, "pnl": 0.0})
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0.0
        rows.append((s.name, d["trades"], d["wins"], wr, d["pnl"]))
    rows.sort(key=lambda r: r[4], reverse=True)

    total_pnl = sum(r[4] for r in rows)
    total_trades = sum(r[1] for r in rows)
    total_wins = sum(r[2] for r in rows)
    overall_wr = (total_wins / total_trades * 100) if total_trades else 0.0

    txt = f"🏆 *Leaderboard* (30d)\n"
    txt += f"Total: {total_trades} trades  WR {overall_wr:.0f}%  P&L *${total_pnl:+.2f}*\n\n"
    for i, (name, trades, wins, wr, pnl) in enumerate(rows, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"`{i:2d}`"
        short = name.replace("_", " ")[:28]
        if trades == 0:
            txt += f"{emoji} `{short:28s}` —\n"
        else:
            txt += f"{emoji} `{short:28s}` n={trades:3d} WR={wr:3.0f}% ${pnl:+7.2f}\n"
    return txt


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
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
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

    # Manage existing positions: close on reversal — ONLY for CLOSE-mode contenders.
    # FixedTP / HALFTP let broker handle exit. TRAIL has its own manager (in main loop).
    my_positions = [p for p in client.get_positions(strat.symbol) if p.get("magic") == strat.magic]
    if strat.exit_mode == "CLOSE" and my_positions and decision.direction != 0 and abs(prob - 0.5) >= 0.2:
        for pos in my_positions:
            pos_dir = 1 if str(pos.get("type")).lower().endswith("buy") or pos.get("type") in (0, "buy") else -1
            if pos_dir != decision.direction:
                if mode == "live":
                    r = client.close_position(int(pos["ticket"]))
                    _log_decision(strat, f"REVERSAL_CLOSE ticket={pos['ticket']} ok={r.success} err={r.error}")
                    if r.success:
                        tg_notify(f"🔄 *Reversal close* `{strat.name}` ticket `{pos['ticket']}`")
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

    # Vol filter: only enter when current bar move >= mult × spread
    if strat.vol_filter_mult > 0:
        try:
            features_check = compute_features(df)
            last_co = float(features_check["co_range"].iloc[-1])
            last_close_p = float(features_check["close"].iloc[-1])
            bar_move = abs(last_co * last_close_p)
            spread_price_estimate = strat.spread_pips_estimate * strat.pip_value
            if bar_move < strat.vol_filter_mult * spread_price_estimate:
                return  # bar too quiet for this scalper
        except Exception:
            pass

    # Override TP based on exit_mode
    if strat.exit_mode == "HALFTP" and strat.half_peak_tp_pips > 0:
        tp_distance = strat.half_peak_tp_pips * strat.pip_value
        plan.tp_price = last_close + tp_distance * decision.direction
        plan.tp_distance = tp_distance
    elif strat.exit_mode in ("CLOSE", "TRAIL15", "TRAIL12", "T2_TIMEEXIT"):
        # No broker TP — bot manages exit via reversal-close, trailing, or time-out
        plan.tp_price = 0.0
        plan.tp_distance = 0.0

    direction_str = "BUY" if decision.direction == 1 else "SELL"
    arrow = "🟢" if decision.direction == 1 else "🔴"
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
        if r.success and r.ticket:
            # Track for closure detection
            tracked_positions = getattr(_check_and_act, "_tracked", None)
            # Note: actual tracking happens in main loop on next tick when we re-query positions
            pass
        if r.success:
            tg_notify(
                f"{arrow} *{direction_str}* `{strat.symbol}` @ {last_close:.5f}\n"
                f"Strategy: `{strat.name}`\n"
                f"SL: `{plan.sl_price:.5f}`  TP: `{plan.tp_price:.5f}`\n"
                f"Prob: {prob:.3f}  ATR: {atr:.5f}  Ticket: `{r.ticket}`"
            )
        else:
            tg_notify(f"⚠️ OPEN FAILED `{strat.name}` {direction_str} — {r.error}")
    else:
        _log_decision(strat, f"[paper] OPEN {direction_str} prob={prob:.3f} price={last_close:.5f} sl={plan.sl_price:.5f} tp={plan.tp_price:.5f}")
        tg_notify(
            f"{arrow} *[paper] {direction_str}* `{strat.symbol}` @ {last_close:.5f}\n"
            f"Strategy: `{strat.name}`  Prob: {prob:.3f}"
        )


def run_battle_royale(mode: str = "paper") -> int:
    # Force UTF-8 stdout on Windows (cp1254 can't encode em dash / arrows)
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    if cfg.mt5 is None:
        log.error("MT5 credentials missing — fill .env")
        return 2

    # Use the new 27-contender lineup (each profitable matrix from full backtest)
    strategies = build_contenders()

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

    # Track open positions across ticks so we can detect broker-side closures (TP/SL)
    # Maps ticket -> (magic, symbol, open_price, direction)
    tracked_positions: dict[int, dict] = {}
    # Seed from any currently-open positions
    for s in strategies:
        for p in client.get_positions(s.symbol):
            if p.get("magic") == s.magic:
                tracked_positions[int(p["ticket"])] = {
                    "magic": s.magic, "name": s.name, "symbol": s.symbol,
                    "open_price": float(p.get("price_open", 0.0)),
                    "type": p.get("type"),
                }

    if telegram_configured():
        current_info = client.get_account_info()
        current_eq = float(current_info["equity"])
        day_start = float(state.get("day_start_equity", current_eq))
        daily_pct = (current_eq - day_start) / day_start * 100.0 if day_start else 0.0
        tg_notify(
            f"🎮 *Battle Royale started*\n"
            f"Mode: `{mode}`\n"
            f"Contenders: *{len(strategies)}*\n"
            f"Current equity: ${current_eq:.2f}\n"
            f"Day baseline: ${day_start:.2f} ({daily_pct:+.2f}%)\n"
            f"\nCommands: /status /lb /help"
        )
        # Start command listener — captures client + strategies + state by closure
        start_command_listener({
            "/start": lambda: "🎮 Battle Royale running. Commands: /status /lb /help",
            "/help": lambda: (
                "*Commands:*\n"
                "/status — current equity, open positions, today's TP/SL/PnL\n"
                "/lb — leaderboard (per-strategy stats, 30d window)\n"
                "/help — this message"
            ),
            "/status": lambda: _status_text(client, strategies, state),
            "/lb": lambda: _leaderboard_text(client, strategies),
            "/leaderboard": lambda: _leaderboard_text(client, strategies),
        })
    else:
        log.info("Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to enable notifications)")

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
                if not state.get("kill_notified_for") == today_now:
                    tg_notify(f"🛑 *KILL SWITCH TRIPPED* daily PnL {ks.daily_pnl_pct:+.2f}% — no new trades today")
                    state["kill_notified_for"] = today_now
                    _save_state(state)

            # T2_TIMEEXIT manager — force-close after N bars of TF have elapsed since open
            for s in strategies:
                if s.max_hold_bars <= 0:
                    continue
                tf_sec = TIMEFRAME_SECONDS.get(s.timeframe, 300)
                max_age_sec = s.max_hold_bars * tf_sec
                for p in client.get_positions(s.symbol):
                    if p.get("magic") != s.magic:
                        continue
                    open_time = float(p.get("time", 0))
                    if open_time <= 0:
                        continue
                    age_sec = time.time() - open_time
                    if age_sec >= max_age_sec:
                        ticket = int(p["ticket"])
                        if mode == "live":
                            r = client.close_position(ticket)
                            _log_decision(s, f"TIMEEXIT_CLOSE ticket={ticket} age_sec={age_sec:.0f} max={max_age_sec:.0f} ok={r.success}")
                            if r.success:
                                tg_notify(f"⏱️ *Time exit* `{s.name}` ticket `{ticket}` ({age_sec/60:.1f} min)")

            # Trail manager — for TRAIL mode positions, close if price retraced beyond trail
            import MetaTrader5 as _mt5
            for s in strategies:
                if s.exit_mode not in ("TRAIL15", "TRAIL12"):
                    continue
                for p in client.get_positions(s.symbol):
                    if p.get("magic") != s.magic:
                        continue
                    ticket = int(p["ticket"])
                    tick = _mt5.symbol_info_tick(s.symbol)
                    if tick is None:
                        continue
                    pos_dir = 1 if str(p.get("type")).lower().endswith("buy") or p.get("type") in (0, "buy") else -1
                    current = tick.bid if pos_dir == 1 else tick.ask
                    meta = tracked_positions.setdefault(ticket, {})
                    # Initialize peak + trail_distance on first sighting
                    if "trail_distance" not in meta:
                        meta.update({"magic": s.magic, "name": s.name, "symbol": s.symbol,
                                     "direction": pos_dir, "peak": current})
                        try:
                            df = client.fetch_history(s.symbol, s.timeframe, 30)
                            from forex_signal.data.features import compute_features
                            feats = compute_features(df)
                            atr = float(feats["atr_14"].iloc[-1])
                            meta["trail_distance"] = s.trail_atr_mult * atr
                        except Exception as e:
                            log.warning("trail init failed for %s: %s", s.name, e)
                            continue
                    # Update peak + check trail trigger
                    if pos_dir == 1:
                        meta["peak"] = max(meta["peak"], current)
                        trail_level = meta["peak"] - meta["trail_distance"]
                        if current <= trail_level:
                            r = client.close_position(ticket) if mode == "live" else None
                            _log_decision(s, f"TRAIL_CLOSE ticket={ticket} current={current:.5f} trail_level={trail_level:.5f} peak={meta['peak']:.5f}")
                            if mode == "live" and r and r.success:
                                tg_notify(f"📉 *Trail close* `{s.name}` ticket `{ticket}` @ {current:.5f}")
                    else:
                        meta["peak"] = min(meta["peak"], current)
                        trail_level = meta["peak"] + meta["trail_distance"]
                        if current >= trail_level:
                            r = client.close_position(ticket) if mode == "live" else None
                            _log_decision(s, f"TRAIL_CLOSE ticket={ticket} current={current:.5f} trail_level={trail_level:.5f} peak={meta['peak']:.5f}")
                            if mode == "live" and r and r.success:
                                tg_notify(f"📈 *Trail close* `{s.name}` ticket `{ticket}` @ {current:.5f}")

            # Detect closures (TP/SL/external): a previously-tracked ticket no longer open
            magic_to_strat = {s.magic: s for s in strategies}
            current_tickets: set[int] = set()
            for s in strategies:
                for p in client.get_positions(s.symbol):
                    if p.get("magic") == s.magic:
                        current_tickets.add(int(p["ticket"]))
                        if int(p["ticket"]) not in tracked_positions:
                            tracked_positions[int(p["ticket"])] = {
                                "magic": s.magic, "name": s.name, "symbol": s.symbol,
                                "open_price": float(p.get("price_open", 0.0)),
                                "type": p.get("type"),
                            }
            closed_tickets = set(tracked_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                info_pos = tracked_positions.pop(ticket)
                # Use the most reliable lookup: by position_id (broker time-zone agnostic)
                import MetaTrader5 as _mt5
                deals = _mt5.history_deals_get(position=ticket) or ()
                total_pnl = sum(d.profit + d.swap + d.commission for d in deals)
                reason_map = {0: "manual", 1: "mobile", 2: "web", 3: "expert", 4: "SL", 5: "TP", 6: "SO", 7: "rollover"}
                close_deals = [d for d in deals if d.entry != 0]  # entry=1 is OUT
                reason = reason_map.get(close_deals[0].reason if close_deals else 0, "?")
                close_price = close_deals[0].price if close_deals else 0.0
                strat_name = info_pos.get("name", f"magic={info_pos['magic']}")
                emoji = "✅" if total_pnl > 0 else "❌"
                _log_decision(
                    magic_to_strat.get(info_pos["magic"], strategies[0]),
                    f"CLOSED ticket={ticket} reason={reason} price={close_price:.5f} pnl={total_pnl:+.2f}",
                )
                tg_notify(
                    f"{emoji} *Closed* `{strat_name}` ({reason})\n"
                    f"Symbol: `{info_pos['symbol']}`  Ticket: `{ticket}`\n"
                    f"P&L: *${total_pnl:+.2f}*"
                )

            n_open = len(current_tickets)
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
