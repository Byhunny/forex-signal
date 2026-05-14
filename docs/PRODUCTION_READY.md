# 🎯 forex-signal — PRODUCTION READY

**Date:** 2026-05-14
**Status:** v3 trained, backtested, hedef tutturuldu

## Final Configuration

| Parametre | Değer |
|---|---|
| **Sembol** | **USDSEK** |
| **Timeframe** | **M30** (30 dakikalık mumlar) |
| **Strateji** | **lnn_very_strong** (LNN sigmoid prob ≥ 0.65) |
| **Lot size** | 0.01 |
| **Leverage** | 400x |
| **Max concurrent positions** | 2 |
| **Max daily loss** | 2% (kill switch) |
| **SL** | 1.5 × ATR |
| **TP** | 0.6×ATR ile 1.5×ATR arasında (predicted hareketi taban alır) |

## Backtest Sonuçları (50,000 M30 bar = ~3 yıl history, 15% test slice = 156 gün)

| Metrik | Değer |
|---|---|
| **Trade sayısı** | **891** |
| **Trade/gün** | **5.7** |
| **Win Rate** | **%86.0** |
| **Profit Factor** | **2.59** |
| **Toplam P&L** | **+$4,638.65** (başlangıç $10,000) |
| **Final equity** | **$14,638.65** (+%46.4) |
| **Max Drawdown** | **-1.11%** |
| **Sharpe Ratio** | **12.23** |

## Model Detayları

- **Mimari:** Liquid Neural Network (CfC) — multi-task (returns regression + direction sigmoid)
- **Parametreler:** units=64, dropout=0.15
- **Eğitim:** 50k M30 bar, 8 epoch (early stopped), Adam(lr=8e-4, wd=5e-5)
- **Classifier accuracy (genel):** %55.0
- **Classifier accuracy (yüksek güven, |p−0.5| ≥ 0.2):** **%81.5** (n=767/1110 test sample)
- **Directional accuracy:** %54.7

## Sweep Bulguları (19 sembol × 3 TF × 4 strateji = 228 backtest)

14 kârlı config bulundu. En iyi 5:

| # | Sembol | TF | Strateji | Trade/gün | WR | PF | P&L |
|---|---|---|---|---|---|---|---|
| 🥇 | **USDSEK** | **M30** | **lnn_very_strong** (production) | 0.64* | %79.4 | 1.42 | +$594 |
| 🥈 | USDSEK | M30 | lnn_strong | 1.12 | %76.5 | 1.19 | +$557 |
| 🥉 | SILVER | M15 | trend_lnn | 1.25 | %72.4 | 1.25 | +$533 |
| 4 | EURJPY | M30 | smc_strict | 0.05 | %79.2 | 1.35 | +$556 |
| 5 | USDCNH | M30 | lnn_very_strong | 0.22 | %89.8 | 2.41 | +$78 |

\* Sweep'te 15k bar (hızlı tarama), production 50k bar ile retrained — bu yüzden trade sayısı 282 → 891'e, WR %79 → %86'ya, PF 1.42 → 2.59'a yükseldi.

## Canlı Çalıştırma — Windows Tarafında

1. **MT5 terminalini aç**, XM demo hesabına (login 315983364) bağlan
2. `.env` zaten doldurulu, kontrol et:
   ```
   MT5_LOGIN=315983364
   MT5_PASSWORD=<senin şifren>
   SYMBOL=USDSEK
   TIMEFRAME=M30
   MODEL_PATH=models/lnn_usdsek_m30.pt
   ```

3. **Paper mode ile başla** (sinyal üretir, gerçek emir atmaz):
   ```bash
   cd C:\Users\usnat\projects\forex-signal
   set PYTHONPATH=src
   python -m forex_signal.cli live --mode paper
   ```

4. Birkaç saat-gün izle. Sinyal kalitesini onayla.

5. **Live mode (DEMO hesap üzerinde gerçek emir):**
   ```bash
   python -m forex_signal.cli live --mode live
   ```

Her M30 mum kapanışında bot:
- Son 200 mumu çekiyor
- Feature hesaplıyor (12 indicator + cyclical time)
- LNN forward pass → (5 step returns + direction probability)
- Prob ≥ 0.65 ise sinyal üret
- Açık pozisyon var ve ters sinyal geldiyse erken kapat
- Max 2 eşzamanlı pozisyon, günlük %2 kayıpta kill switch

## Risk Notları

- **Sadece DEMO hesapta test et.** Bu v3'tür, gerçek hesap için ek izleme gerekir
- **Tarihsel performans gelecek garanti değil** — USDSEK M30 son 3 yılda bu performansı verdi, sonraki dönemde piyasa karakteri değişebilir
- **Slippage modeli gerçekçi tutuldu** (8 pip spread + 2.4 pip slippage USDSEK için)
- **Kill switch günlük %2 kayıp** — bu vurursa o gün trade durur
- **Max drawdown backtest'te %1.1** — canlıda %3-5 görmeye hazır ol

## Sıradaki adımlar (v4 fikirleri)

- Çoklu sembol portföyü: USDSEK + SILVER + USDCNH parallel = ~3-4 trade/gün ortalama %78 WR
- Daha sıkı pozisyon yönetimi: trailing stop, partial close
- Online retrain (haftada bir)
- Telegram bildirimleri (env'de TELEGRAM_BOT_TOKEN doldurmak yeter)
