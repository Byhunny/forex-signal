# 📊 Master Backtest Report — All Results

Generated: 2026-05-15T17:18:49.964871+00:00


## Test Setup

- 20 strategies from original sweep (TOP 20 battle royale lineup)
- Realistic live spreads from MT5 with 20% safety buffer
- Spread-aware SL widening (max(1.5×ATR, 4×spread))
- Single spread charge (bug fixed)
- 0.01 lot, $7/lot commission
- Bar data: 50k for USDSEK M30, 15k for others


## Live Spreads (MT5 snapshot, +20% buffer)

| Symbol | Live Spread (pips) |
|---|---|
| USDSEK | 136.8 (×1.2 = 164.2) |
| SILVER | 6.8 (×1.2 = 8.2) |
| USDCNH | 38.0 (×1.2 = 45.6) |
| EURJPY | 3.2 (×1.2 = 3.8) |
| EURUSD | 1.9 (×1.2 = 2.3) |
| AUDUSD | 2.3 (×1.2 = 2.8) |
| EURCAD | 3.3 (×1.2 = 4.0) |
| US100Cash | 26.0 (×1.2 = 31.2) |
| BTCUSD | 50.0 (×1.2 = 60.0) |
| GOLD | 5.2 (×1.2 = 6.2) |

## Master Comparison Table (all 20 strategies, all 7 metrics)

Legend:
- **OLD_SWEEP**: original sweep with optimistic 8-pip USDSEK spread + double-spread bug
- **NORM_TP**: standard TP (0.6-1.5×ATR), realistic spread, bug fixed
- **PEAK**: hindsight peak exit (theoretical ceiling)
- **CLOSE**: hold until opposite signal, exit at close price (realistic)
- **HALF-TP**: TP at half of median peak distance
- **TRAIL 1.5**: trailing stop at 1.5×ATR from peak
- **TRAIL 1.2**: trailing stop at 1.2×ATR from peak (tighter)

| # | Symbol | TF | Strategy | OLD_SWEEP P&L | NORM_TP P&L | PEAK P&L | **CLOSE P&L** | HALF-TP P&L | TRAIL_1.5 P&L | TRAIL_1.2 P&L |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | USDSEK | M30 | lnn_very_strong | $+594 | $-13516 | $+6832 | **$-1978** | $-7892 | $-12623 | $-14060 |
| 02 | USDSEK | M30 | lnn_strong | $+557 | $-20172 | $+3884 | **$-10786** | $-14730 | $-26816 | $-31654 |
| 03 | SILVER | M15 | trend_lnn | $+533 | $-4 | $+7813 | **$+3513** | $+3261 | $-2612 | $-3370 |
| 04 | USDCNH | M30 | lnn_very_strong | $+78 | $-4 | $+348 | **$-57** | $-176 | $-386 | $-417 |
| 05 | SILVER | M15 | lnn_strong | $+240 | $+220 | $+3866 | **$+0** | $+0 | $-690 | $-939 |
| 06 | EURJPY | M30 | smc_strict | $+556 | $+328 | $+20201 | **$+16910** | $+8581 | $-2880 | $-2121 |
| 07 | EURUSD | M30 | smc_strict | $+7 | $+8 | $+73 | **$+62** | $+2 | $+11 | $+3 |
| 08 | AUDUSD | M30 | smc_strict | $+7 | $-2 | $+54 | **$+20** | $+27 | $-27 | $-27 |
| 09 | EURCAD | M30 | lnn_strong | $+0 | $+15 | $+51 | **$+20** | $-18 | $+3 | $-10 |
| 10 | US100Cash | M30 | trend_lnn | $+6 | $+1 | $+140 | **$+51** | $+55 | $-29 | $-29 |
| 11 | USDSEK | M5 | lnn_very_strong | $+34 | $-119 | $+124 | **$-154** | $+223 | $-472 | $-679 |
| 12 | USDSEK | M30 | trend_lnn | $-60 | $-27594 | $+12589 | **$-7641** | $-14694 | $-48647 | $-61791 |
| 13 | USDSEK | M5 | lnn_strong | $-99 | $-1071 | $+761 | **$-253** | $-474 | $-2748 | $-4106 |
| 14 | SILVER | M5 | trend_lnn | $-179 | $-422 | $+2289 | **$+364** | $+505 | $-801 | $-1042 |
| 15 | SILVER | M30 | trend_lnn | $-431 | $-531 | $+9746 | **$+3456** | $+8 | $-5507 | $-7324 |
| 16 | EURJPY | M15 | trend_lnn | $-658 | $-3197 | $+12649 | **$+4786** | $+9153 | $-9448 | $-11901 |
| 17 | BTCUSD | M30 | trend_lnn | $-149 | $-194 | $+1026 | **$+164** | $-11 | $-648 | $-794 |
| 18 | BTCUSD | M15 | trend_lnn | $-244 | $-205 | $+446 | **$-95** | $+85 | $-479 | $-609 |
| 19 | GOLD | M15 | trend_lnn | $-1131 | $+761 | $+2044 | **$+923** | $+1055 | $-722 | $-1373 |
| 20 | USDSEK | M15 | trend_lnn | $-180 | $-2100 | $+971 | **$-1817** | $-1955 | $-5202 | $-6619 |

## PF Comparison (Profit Factor — higher is better, >1 means profitable)

| # | Symbol | TF | OLD_SWEEP | NORM_TP | PEAK | **CLOSE** | HALF-TP | TRAIL_1.5 | TRAIL_1.2 |
|---|---|---|---|---|---|---|---|---|---|
| 01 | USDSEK | M30 | 1.42 | 0.52 | 1.55 | **0.85** | 0.58 | 0.12 | 0.07 |
| 02 | USDSEK | M30 | 1.19 | 0.53 | 1.18 | **0.53** | 0.51 | 0.09 | 0.05 |
| 03 | SILVER | M15 | 1.25 | 1.00 | 9.54 | **4.82** | 2.22 | 0.46 | 0.32 |
| 04 | USDCNH | M30 | 2.41 | 0.99 | 1.75 | **0.88** | 0.70 | 0.11 | 0.08 |
| 05 | SILVER | M15 | 1.43 | 1.32 | 99.99 | **0.00** | 0.00 | 0.51 | 0.30 |
| 06 | EURJPY | M30 | 1.35 | 1.16 | 8.31 | **7.12** | 2.84 | 0.35 | 0.41 |
| 07 | EURUSD | M30 | 1.35 | 1.35 | 3.57 | **3.18** | 1.04 | 1.46 | 1.11 |
| 08 | AUDUSD | M30 | 1.38 | 0.93 | 2.28 | **1.48** | 1.53 | 0.28 | 0.31 |
| 09 | EURCAD | M30 | 1.00 | 1.36 | 2.13 | **1.44** | 0.78 | 1.08 | 0.80 |
| 10 | US100Cash | M30 | 1.18 | 1.03 | 4.71 | **2.35** | 1.77 | 0.58 | 0.56 |
| 11 | USDSEK | M5 | 1.41 | 0.86 | 1.27 | **0.67** | 1.42 | 0.01 | 0.01 |
| 12 | USDSEK | M30 | 0.99 | 0.51 | 1.68 | **0.64** | 0.58 | 0.08 | 0.05 |
| 13 | USDSEK | M5 | 0.85 | 0.62 | 2.28 | **0.65** | 0.65 | 0.04 | 0.01 |
| 14 | SILVER | M5 | 0.81 | 0.70 | 4.26 | **1.52** | 1.32 | 0.47 | 0.38 |
| 15 | SILVER | M30 | 0.94 | 0.94 | 3.52 | **1.86** | 1.00 | 0.47 | 0.39 |
| 16 | EURJPY | M15 | 0.91 | 0.74 | 4.39 | **2.28** | 2.05 | 0.32 | 0.19 |
| 17 | BTCUSD | M30 | 0.77 | 0.73 | 2.62 | **1.26** | 0.99 | 0.37 | 0.24 |
| 18 | BTCUSD | M15 | 0.58 | 0.67 | 1.85 | **0.82** | 1.10 | 0.45 | 0.32 |
| 19 | GOLD | M15 | 0.57 | 1.44 | 2.98 | **1.90** | 1.73 | 0.75 | 0.58 |
| 20 | USDSEK | M15 | 0.82 | 0.58 | 1.43 | **0.38** | 0.41 | 0.02 | 0.01 |

## WR Comparison (Win Rate %)

| # | Symbol | TF | OLD_SWEEP | NORM_TP | PEAK | **CLOSE** | HALF-TP | TRAIL_1.5 | TRAIL_1.2 |
|---|---|---|---|---|---|---|---|---|---|
| 01 | USDSEK | M30 | 79.4% | 45.5% | 33.8% | **27.5%** | 34.9% | 13.0% | 9.1% |
| 02 | USDSEK | M30 | 76.5% | 46.2% | 34.9% | **23.6%** | 37.3% | 12.1% | 8.7% |
| 03 | SILVER | M15 | 72.4% | 61.6% | 26.1% | **18.2%** | 19.0% | 27.0% | 20.8% |
| 04 | USDCNH | M30 | 89.8% | 61.9% | 35.0% | **33.3%** | 41.1% | 7.9% | 6.1% |
| 05 | SILVER | M15 | 80.5% | 74.7% | 100.0% | **0.0%** | 0.0% | 22.7% | 21.9% |
| 06 | EURJPY | M30 | 79.2% | 70.8% | 12.5% | **12.5%** | 11.8% | 31.0% | 26.7% |
| 07 | EURUSD | M30 | 82.5% | 78.0% | 6.2% | **6.2%** | 4.5% | 41.9% | 34.3% |
| 08 | AUDUSD | M30 | 81.8% | 67.9% | 14.8% | **14.8%** | 20.0% | 24.4% | 21.6% |
| 09 | EURCAD | M30 | 80.3% | 77.5% | 9.5% | **5.0%** | 6.2% | 39.2% | 36.4% |
| 10 | US100Cash | M30 | 76.8% | 73.8% | 15.2% | **12.5%** | 14.9% | 21.5% | 25.3% |
| 11 | USDSEK | M5 | 80.0% | 58.1% | 30.0% | **22.2%** | 46.7% | 3.4% | 5.3% |
| 12 | USDSEK | M30 | 73.6% | 45.1% | 41.2% | **25.6%** | 38.0% | 10.2% | 6.3% |
| 13 | USDSEK | M5 | 70.9% | 50.0% | 62.5% | **26.1%** | 65.6% | 5.8% | 1.4% |
| 14 | SILVER | M5 | 73.7% | 59.2% | 23.1% | **21.1%** | 20.7% | 33.1% | 28.3% |
| 15 | SILVER | M30 | 67.2% | 60.6% | 27.8% | **22.7%** | 40.1% | 25.0% | 20.5% |
| 16 | EURJPY | M15 | 76.8% | 56.9% | 13.6% | **9.5%** | 13.8% | 20.0% | 14.2% |
| 17 | BTCUSD | M30 | 69.3% | 62.9% | 18.2% | **16.4%** | 16.8% | 27.4% | 25.8% |
| 18 | BTCUSD | M15 | 68.2% | 64.0% | 16.4% | **15.7%** | 26.6% | 29.2% | 27.1% |
| 19 | GOLD | M15 | 65.9% | 76.2% | 9.3% | **7.1%** | 11.1% | 34.5% | 33.1% |
| 20 | USDSEK | M15 | 70.6% | 48.3% | 43.8% | **20.8%** | 53.2% | 3.8% | 2.8% |

## Peak Distance Distribution (per strategy, in pips)

Tells us where TP could be placed. Median peak / 2 used for HALF-TP test.

| # | Symbol | TF | Peaks | SLs | ATR (pip) | Spread | Peak Mean | Median | P25 | P75 | P90 | Peak/ATR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | USDSEK | M30 | 109 | 184 | 184.3 | 164.2 | 1802 | 1398 | 541 | 2347 | 3624 | 11.14× |
| 02 | USDSEK | M30 | 206 | 295 | 189.8 | 164.2 | 1290 | 1103 | 396 | 2028 | 2713 | 7.70× |
| 03 | SILVER | M15 | 5 | 17 | 78.0 | 8.2 | 3062 | 2103 | 1629 | 3160 | 5905 | 106.09× |
| 04 | USDCNH | M30 | 16 | 23 | 32.7 | 45.6 | 511 | 367 | 224 | 593 | 1084 | 21.89× |
| 06 | EURJPY | M30 | 1 | 7 | 25.3 | 3.8 | 2297 | 2297 | 2297 | 2297 | 2297 | 127.27× |
| 07 | EURUSD | M30 | 1 | 15 | 12.3 | 2.3 | 1012 | 1012 | 1012 | 1012 | 1012 | 46.03× |
| 08 | AUDUSD | M30 | 4 | 24 | 11.6 | 2.8 | 244 | 224 | 136 | 332 | 373 | 18.15× |
| 09 | EURCAD | M30 | 1 | 18 | 14.8 | 4.0 | 932 | 932 | 932 | 932 | 932 | 42.65× |
| 10 | US100Cash | M30 | 4 | 28 | 848.8 | 31.2 | 29581 | 25333 | 22816 | 32098 | 42792 | 35.99× |
| 11 | USDSEK | M5 | 2 | 7 | 81.6 | 164.2 | 2295 | 2295 | 1729 | 2861 | 3201 | 38.83× |
| 12 | USDSEK | M30 | 225 | 261 | 181.0 | 164.2 | 1440 | 1186 | 581 | 2052 | 2999 | 8.74× |
| 13 | USDSEK | M5 | 14 | 9 | 82.9 | 164.2 | 927 | 421 | 334 | 1245 | 2200 | 13.30× |
| 14 | SILVER | M5 | 8 | 30 | 28.5 | 8.2 | 550 | 473 | 344 | 693 | 944 | 22.72× |
| 15 | SILVER | M30 | 44 | 101 | 43.8 | 8.2 | 600 | 247 | 98 | 579 | 1575 | 17.90× |
| 16 | EURJPY | M15 | 2 | 18 | 12.7 | 3.8 | 520 | 520 | 351 | 689 | 791 | 51.60× |
| 17 | BTCUSD | M30 | 20 | 87 | 462.6 | 60.0 | 8310 | 7198 | 4656 | 9029 | 14481 | 21.35× |
| 18 | BTCUSD | M15 | 22 | 110 | 299.6 | 60.0 | 4441 | 3067 | 2547 | 4957 | 7389 | 16.76× |
| 19 | GOLD | M15 | 3 | 39 | 180.7 | 6.2 | 9618 | 8212 | 5245 | 13288 | 16333 | 133.30× |
| 20 | USDSEK | M15 | 40 | 31 | 96.4 | 164.2 | 817 | 412 | 101 | 1513 | 2007 | 9.84× |

## Totals (all 20 strategies combined)

| Mode | Total P&L | Profitable count | Description |
|---|---|---|---|
| OLD_SWEEP (original) | $-520 | 11/20 | with bug + optimistic spreads |
| NORM_TP (realistic spread) | $-67,802 | 6/20 | realistic but tight TP |
| PEAK (hindsight) | $+85,908 | 20/20 | hindsight ceiling |
| **CLOSE_AT_REVERSE** | $+7,490 | 11/20 | **recommended — most realistic kazanan** |
| HALF-TP (median/2) | $-16,994 | 11/20 | TP-based, between close and norm |
| TRAIL 1.5×ATR | $-120,724 | 2/20 | trailing too tight |
| TRAIL 1.2×ATR | $-148,861 | 1/20 | trailing even tighter — worse |

## Conclusions

1. **CLOSE_AT_REVERSE is the practical winner** — 11/20 profitable, total +$7,490
2. **Original sweep was misleading** — double-spread bug + optimistic spreads showed false hopes
3. **USDSEK is untrade-able** — 137 pip live spread destroys edge regardless of exit mode
4. **Trailing stops too tight** at both 1.2 and 1.5 × ATR — volatility hits trail before move develops
5. **Peak (hindsight)** shows model has REAL directional edge — TP placement is the harvest mechanism issue
6. **Half-peak TP** captures portion of peak but underperforms close-at-reverse on most strategies

## ✅ Recommended Battle Royale Lineup (11 profitable in CLOSE mode)

| Rank | Slot | Symbol | TF | PF | Test P&L |
|---|---|---|---|---|---|
| 1 | 06 | EURJPY | M30 | 7.12 | $+16,910 |
| 2 | 16 | EURJPY | M15 | 2.28 | $+4,786 |
| 3 | 03 | SILVER | M15 | 4.82 | $+3,513 |
| 4 | 15 | SILVER | M30 | 1.86 | $+3,456 |
| 5 | 19 | GOLD | M15 | 1.90 | $+923 |
| 6 | 14 | SILVER | M5 | 1.52 | $+364 |
| 7 | 17 | BTCUSD | M30 | 1.26 | $+164 |
| 8 | 07 | EURUSD | M30 | 3.18 | $+62 |
| 9 | 10 | US100Cash | M30 | 2.35 | $+51 |
| 10 | 08 | AUDUSD | M30 | 1.48 | $+20 |
| 11 | 09 | EURCAD | M30 | 1.44 | $+20 |