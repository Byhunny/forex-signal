# forex-signal — LNN Scalping Bot (Design Spec)

**Date:** 2026-05-14
**Status:** Approved (user delegated all design decisions to Claude)
**Author:** Claude (autonomous design)

## 1. Purpose

Build a working MT5 scalping bot that uses a **Liquid Neural Network (CfC)** to predict the next 5 candles' returns on EURUSD M5, and executes 0.01-lot trades with leverage 400x on an XM Global demo account.

The goal is **a working end-to-end pipeline** — data → features → LNN training → backtest → live execution — not a profitable strategy on day one. Profitability comes from iteration on top of this pipeline.

## 2. Scope

**In scope (v1):**
- Pure Python implementation
- EURUSD on M5 timeframe
- CfC (Closed-form Continuous-time) model from `ncps`
- Multi-step return prediction (next 5 candles)
- ATR-based SL, predicted-peak TP
- Walk-forward backtesting with realistic spread/slippage
- Live trading loop via `MetaTrader5` Python package (Windows)
- yfinance fallback for offline development and training when MT5 is unavailable
- Unit tests for features, signal logic, risk sizing, backtest engine

**Out of scope (v1, deferred to v2+):**
- Multi-symbol training
- Hyperparameter optimization (Optuna)
- Telegram notifier (stub only)
- Web dashboard
- Grid / averaging / martingale variants
- Online (continual) learning during live trading
- Multi-timeframe features

## 3. Key Decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Language | Pure Python | `ncps` is Python-only; faster ML iteration; user chose this |
| Model | CfC (Closed-form Continuous-time) | ~50–100× faster than LTC, same accuracy class, ONNX-friendly later |
| Prediction target | Next 5 M5 candle returns (regression) | Provides direction AND magnitude → drives TP |
| Loss | MSE on normalized returns | Standard, stable, interpretable |
| Sequence length | 60 candles (5 hours) | Enough context, not so long it blows up training |
| Features (12) | OHLCV-derived + RSI/ATR/BB/MACD + cyclical hour | LNN benefits from rich, normalized features; cyclical time captures session effects |
| SL | 1.5 × ATR(14) | Standard scalping baseline; survives noise |
| TP | min(predicted peak return × price, 4 × ATR), floor 1 × ATR | Captures model's edge while bounded |
| Risk | 0.01 lot, max 2 concurrent positions, 2% daily loss kill-switch | User's spec + sane guardrails |
| Spread/slippage in backtest | 0.8 pips spread + 0.3 pips slippage + $7/lot commission | XM EURUSD typical |
| Train data | yfinance EURUSD=X daily fallback for dev; MT5 history for prod | yfinance has no M5; use it only as smoke-test data. Real training needs MT5 history dump. |

## 4. Architecture

```
                ┌──────────────────────────────┐
                │       MT5 Terminal (Win)     │
                └──────────────┬───────────────┘
                               │ MetaTrader5 Python API
                               ▼
        ┌──────────────────────────────────────────────┐
        │  data/mt5_client.py     data/yfinance_loader │
        │  (live + history)        (offline dev)        │
        └──────────────────┬───────────────────────────┘
                           │ DataFrame[OHLCV]
                           ▼
                ┌──────────────────────┐
                │  data/features.py    │  → DataFrame[12 features + target]
                └──────────┬───────────┘
                           │
              ┌────────────┴──────────┐
              │                       │
              ▼                       ▼
   ┌────────────────────┐   ┌────────────────────┐
   │ model/train.py     │   │ model/predict.py   │
   │ (offline, CfC)     │   │ (load weights,     │
   │  → models/*.pt     │   │  forward pass)     │
   └────────────────────┘   └─────────┬──────────┘
                                      │ predicted returns [B,5]
                                      ▼
                          ┌──────────────────────┐
                          │ strategy/signal.py   │  Direction + size + TP/SL
                          │ strategy/risk.py     │
                          └──────────┬───────────┘
                                     │ Order intent
                          ┌──────────┴───────────┐
                          ▼                      ▼
              ┌────────────────────┐  ┌──────────────────────┐
              │ execution/         │  │ backtest/            │
              │  order_manager.py  │  │  walk_forward.py     │
              │ (live MT5)         │  │ (paper, historical)  │
              └────────────────────┘  └──────────────────────┘
                          │
                          ▼
                  ┌─────────────────┐
                  │ Telegram (stub) │
                  └─────────────────┘
```

## 5. Components

### 5.1 `data/mt5_client.py`
- `connect(login, password, server, path)` — initialize MT5
- `fetch_history(symbol, timeframe, bars)` → `pd.DataFrame[time, open, high, low, close, tick_volume, spread]`
- `fetch_latest(symbol, timeframe, n=60)` — for live inference
- `get_account_info()`, `get_positions()`, `close_position(ticket)`
- `place_order(symbol, type, lot, sl, tp, magic, comment)` → `OrderResult`
- Graceful handling: if `MetaTrader5` package import fails (non-Windows), expose a `MockMT5` that reads cached CSV — so the rest of the pipeline still tests on WSL.

### 5.2 `data/yfinance_loader.py`
- `load_eurusd(start, end, interval="1h")` → DataFrame
- Used for **offline dev only** (yfinance has no true M5 forex; we use 1h as smoke-test stand-in). Real model is retrained from MT5 history on Windows.

### 5.3 `data/features.py`
Computes 12 features from OHLCV:
1. `ret_1` — close-to-close return
2. `log_ret_1` — log return
3. `hl_range` — (high-low)/close
4. `co_range` — (close-open)/close
5. `rsi_14`
6. `atr_14` (also exported separately, used for SL/TP)
7. `bb_position` — (close - bb_mid) / (bb_upper - bb_lower)
8. `macd_diff` — MACD line - signal line
9. `ema_ratio` — EMA(9)/EMA(21) - 1
10. `volume_z` — z-scored tick volume over last 50 bars
11. `hour_sin`, `hour_cos` — cyclical hour-of-day

Plus target: rolling next-N returns (N=5) for training. Returns `(X, y, scaler)` where X is `(samples, seq_len, n_features)` and y is `(samples, pred_horizon)`.

### 5.4 `model/lnn.py`
```python
class ForexLNN(nn.Module):
    def __init__(self, n_features, units=48, pred_horizon=5, dropout=0.1):
        self.cfc = CfC(n_features, units)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(units, pred_horizon)
    def forward(self, x):
        out, _ = self.cfc(x)
        return self.head(self.drop(out[:, -1, :]))   # (B, pred_horizon)
```

### 5.5 `model/train.py`
- Walk-forward split: 70% train / 15% val / 15% test (no leakage)
- Adam optimizer, MSE loss, early stopping (patience=6)
- Checkpoint best val loss → `models/lnn_eurusd_m5.pt`
- Logs: train_loss, val_loss, val_directional_accuracy (sign-of-cumret match)

### 5.6 `strategy/signal.py`
Given predicted returns `pred_ret = [r1, r2, r3, r4, r5]`:
- `cum_ret = sum(pred_ret)`
- `direction = sign(cum_ret)` if `abs(cum_ret) >= min_predicted_return_bps`
- `directional_consistency = mean(sign(pred_ret) == direction)`
- Emit BUY/SELL signal only if `consistency >= 0.6` AND not in cooldown.

### 5.7 `strategy/risk.py`
- `sl_distance = sl_atr_multiplier * atr`
- `tp_distance = clip(abs(cum_ret * price), tp_atr_min * atr, tp_atr_max * atr)`
- `lot_size`: fixed 0.01 (v1; v2 could size by ATR/equity)
- Kill switch: refuse new trades if `(current_equity - day_start_equity) / day_start_equity < -0.02`. `day_start_equity` is captured on the first cycle of each UTC trading day and persisted to `logs/state.json` to survive restarts.

### 5.8 `execution/order_manager.py`
- `open(signal, atr, price)` → calls `mt5_client.place_order` with SL/TP
- `manage_open_positions(predictions)` — early exit if model now predicts opposite direction with high consistency
- `close_at_eod()` — close all positions before Friday rollover

### 5.9 `backtest/walk_forward.py`
- Iterates through historical bars, simulates fills with spread + slippage
- Tracks equity, drawdown, win rate, Sharpe, profit factor
- Outputs: `logs/backtest_<timestamp>.json` + equity curve CSV

### 5.10 `cli.py`
Subcommands:
- `python -m forex_signal.cli download` — fetch and cache history
- `python -m forex_signal.cli train` — train model
- `python -m forex_signal.cli backtest` — run backtest
- `python -m forex_signal.cli live` — start live loop (Windows only)
- `python -m forex_signal.cli predict-once` — debug, run a single inference

## 6. Data Flow (Live)

```
Every M5 candle close (with a 2-second buffer):
  1. fetch latest 60 candles from MT5
  2. compute features (using fitted scaler from training)
  3. forward pass through LNN → (5,) predicted returns
  4. compute signal (direction, consistency, magnitude)
  5. fetch open positions
     - if no position AND signal valid AND < max_concurrent AND not in cooldown:
         open trade with computed SL/TP
     - if position open AND model now predicts opposite with consistency > 0.7:
         close early
  6. enforce kill-switch (daily loss)
  7. log + optional telegram notify
  8. sleep until next M5 close
```

## 7. Error Handling

- MT5 disconnect: retry 3× with exponential backoff, then halt and notify
- Order rejection: log with full context, do not retry blindly
- NaN in features (e.g., not enough history): skip cycle, log warning
- Model load failure: refuse to start live; print clear instructions
- yfinance/MT5 history mismatch: log and use whichever is available

## 8. Testing Strategy

**Unit tests (run in WSL, no MT5 needed):**
- `test_features.py` — feature computations are deterministic, no NaN beyond warmup, scaler round-trip
- `test_signal.py` — signal generator behaviors on synthetic predictions (positive/negative/mixed)
- `test_risk.py` — SL/TP math, kill-switch activation, lot sizing
- `test_backtest.py` — backtest engine with mocked predictions yields deterministic equity curve

**Smoke test (WSL):**
- Download yfinance EURUSD 1h data
- Compute features, train a tiny model (3 epochs, units=16) — verify training loop completes and saves checkpoint
- Run backtest end-to-end on test split — verify metrics computed without exception

**Live integration test (Windows, manual):**
- User fills `.env`, runs `python -m forex_signal.cli predict-once` to verify MT5 connection + inference works on live data
- Then `python -m forex_signal.cli live` for actual paper-style trading on demo

## 9. Success Criteria (v1)

1. **Pipeline works end-to-end:** download → train → backtest → live (last step verified by user on Windows)
2. **All unit tests pass** in WSL
3. **Smoke backtest completes** and produces non-trivial trade count + metrics file
4. **Model converges:** val loss decreases over epochs on smoke-test data
5. **Code is modular:** components can be swapped (e.g., CfC → LTC, EURUSD → XAUUSD) by config change

**Non-goal:** profitable backtest. Realistic win rates emerge only after proper M5 MT5 history training, which the user runs on Windows after v1 lands.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `MetaTrader5` package only on Windows | Mock client for WSL dev; clear "run on Windows" instruction in README |
| yfinance only has hourly forex, not M5 | Used only for smoke test; real training uses MT5 history dump on Windows |
| LNN overfit on small dataset | Dropout 0.1 + early stopping + walk-forward validation |
| Live data drift from training distribution | Periodic retrain script + monitoring of inference distribution |
| Slippage worse than modeled | Conservative 0.3 pips slippage + 0.8 pips spread in backtest |
| User runs live by accident | Default mode is `paper`; live requires explicit `--mode live` flag |

## 11. File Layout

```
forex-signal/
├── .env.example
├── .gitignore
├── config.yaml
├── requirements.txt
├── README.md
├── docs/superpowers/specs/2026-05-14-lnn-scalping-design.md   (this doc)
├── src/forex_signal/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mt5_client.py
│   │   ├── yfinance_loader.py
│   │   └── features.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── lnn.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── signal.py
│   │   └── risk.py
│   ├── execution/
│   │   ├── __init__.py
│   │   └── order_manager.py
│   └── backtest/
│       ├── __init__.py
│       └── walk_forward.py
├── scripts/
│   └── (thin wrappers around cli for convenience)
├── tests/
│   ├── test_features.py
│   ├── test_signal.py
│   ├── test_risk.py
│   └── test_backtest.py
├── models/         (gitignored)
├── data_cache/     (gitignored)
└── logs/           (gitignored)
```

## 12. Open Questions (none — all resolved by author)

User explicitly delegated all design decisions. Open questions for **v2** (not blockers for v1):
- Optimal `seq_len` and `pred_horizon` — to be tuned post-baseline
- Symbol expansion (XAUUSD, GBPUSD)
- Whether to add session filter (London/NY only)
- Online learning / model refresh cadence
