# forex-signal

LNN (Liquid Neural Network) tabanlı MT5 scalping botu. EURUSD M5 üzerinde sonraki 5 mumun return'ünü tahmin eder, ATR-bazlı SL ile koruyup tahmin edilen tepe noktasına TP koyar.

**Sürüm:** v1 — pipeline çalışıyor, eğitim/backtest/canlı döngü hazır.
**Tasarım:** [docs/superpowers/specs/2026-05-14-lnn-scalping-design.md](docs/superpowers/specs/2026-05-14-lnn-scalping-design.md)

## Hızlı Bakış

```
OHLCV -> features (12) -> CfC LNN (units=48) -> 5-step return -> signal -> SL/TP -> MT5 order
```

- **Model:** ncps CfC (Closed-form Continuous-time), ~10-50k parametre
- **Tahmin hedefi:** Sonraki 5 mumun log-return'leri (regresyon, MSE)
- **Sinyal:** Kümülatif tahmin >= 1.5 bps + adım yönü tutarlılığı >= 0.6
- **Risk:** 0.01 lot, max 2 eşzamanlı pozisyon, %2 günlük loss kill-switch
- **TP:** `clip(tahmin_pik, 1×ATR, 4×ATR)` — **SL:** `1.5 × ATR`
- **Reversal exit:** Model şimdi karşı yönü %70 güvenle gösterirse açık pozisyonu erken kapat

## Klasör Yapısı

```
src/forex_signal/
  config.py             # config.yaml + .env yükleyici
  cli.py                # Tek giriş noktası — tüm komutlar
  data/
    mt5_client.py       # MT5 client (Windows) + MockMT5 (WSL dev)
    yfinance_loader.py  # Offline dev için
    features.py         # 12 özellik + windowed dataset
  model/
    lnn.py              # CfC model (ncps)
    train.py            # Walk-forward split + Adam + early stopping
    predict.py          # Checkpoint yükle, inference
  strategy/
    signal.py           # Multi-step prediction -> Signal{dir, conf, cum_ret}
    risk.py             # ATR-bazlı SL/TP + KillSwitch
  execution/
    order_manager.py    # MT5 üzerinden aç/kapat/yönet
    live_loop.py        # Canlı M5-close döngüsü
  backtest/
    walk_forward.py     # Spread + slippage + komisyon dahil bar-bar simülasyon
tests/                  # 19 unit test
docs/superpowers/specs/ # Tasarım dokümanı
```

## Kurulum (Tek Seferlik)

```bash
pip install -r requirements.txt
# Canlıya geçeceğin Windows makinesinde ayrıca:
pip install MetaTrader5
```

### .env Dosyasını Doldur

```bash
cp .env.example .env
# .env'yi düzenle:
#   MT5_LOGIN=315983364
#   MT5_PASSWORD=...      <- XM hesap şifreni gir
#   MT5_SERVER=XMGlobal-MT5 7
#   MT5_PATH=             <- (opsiyonel) MT5 terminal.exe tam yolu
```

## Komutlar

Hepsi `python -m forex_signal.cli <komut>` ile çağrılır (root'tan `PYTHONPATH=src` olması yeterli).

### 1) Veri İndir

```bash
# WSL / offline dev için (1h, son ~730 gün):
PYTHONPATH=src python -m forex_signal.cli download --source yfinance --interval 1h

# Windows / canlı için (MT5 history, M5):
PYTHONPATH=src python -m forex_signal.cli download --source mt5 --bars 50000
```

### 2) Modeli Eğit

```bash
PYTHONPATH=src python -m forex_signal.cli train --epochs 40
# -> models/lnn_eurusd_m5.pt
# Yan dosya: models/lnn_eurusd_m5.json (metrikler + history)
```

### 3) Backtest

```bash
PYTHONPATH=src python -m forex_signal.cli backtest
# -> logs/backtest_<timestamp>.json
# Çıktı: trade sayısı, kazanma oranı, PF, sharpe, max DD, equity curve
```

### 4) Tek Tahmin (debug)

```bash
PYTHONPATH=src python -m forex_signal.cli predict-once
# 5 adımlı tahmini + cum bps + ATR'ı yazar
```

### 5) Canlı Trading (sadece Windows)

```bash
# Paper mode (sinyal üretir ama emir göndermez):
PYTHONPATH=src python -m forex_signal.cli live --mode paper

# Live mode (gerçek emir gönderir — DEMO hesabında dene):
PYTHONPATH=src python -m forex_signal.cli live --mode live
```

Canlı döngü:
- Her M5 close + 2 saniye sonra uyanır
- Son 200 mumu çeker
- LNN ile 5 adımlı tahmin yapar
- Sinyali değerlendirir, açık pozisyonları yönetir, yeni trade açar
- Kill switch tetiklendiyse veya max pozisyon doluysa açmaz
- `logs/state.json` → günlük başlangıç equity'sini saklar (restart-safe)

## Testler

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

19 test geçer. Hiçbiri MT5'e ihtiyaç duymaz, hepsi WSL'de çalışır.

## Tipik Workflow (Windows tarafında)

```bash
# 1. .env'i doldur
# 2. MT5 terminalini aç, XM demo hesabına bağlan
# 3. Veri indir (gerçek M5)
python -m forex_signal.cli download --source mt5 --bars 50000

# 4. Tam eğitim (~10-30 dakika CPU)
python -m forex_signal.cli train --epochs 40

# 5. Backtest — metrikler tatmin ediciyse devam
python -m forex_signal.cli backtest

# 6. Önce paper mode, demo hesapta 1 gün izle
python -m forex_signal.cli live --mode paper

# 7. Tatmin edici sinyaller görüyorsan live mode (demo hesap)
python -m forex_signal.cli live --mode live
```

## Önemli Notlar

- **Bu v1; karlılık değil pipeline odaklı.** Win rate < %50, PF < 1 olabilir. Tuning v2'de yapılır.
- **Asla gerçek hesapta v1'i çalıştırma.** XM demo hesabı veya ayrı bir demo'da test et.
- **Slippage/spread modeli gerçekçi tutuldu** (0.8 + 0.3 pip + $7/lot). Yine de canlı sonuçlar daha kötü olabilir.
- **`MetaTrader5` paketi sadece Windows'ta çalışır.** WSL'de `MockMT5` otomatik devreye girer.
- **yfinance forex'in M5'i yok.** Eğitim için MT5 history dump'ı kullan.

## Yan Proje Notları

Bu proje `forex-guru` (Go + MT5 EA File Server, indikatör stratejileri) yan projesinin **ML-bazlı** alternatifi. forex-guru'daki risk yönetimi / paper mode / telegram bildirim mantığı v2'de buraya taşınabilir.

## Sıradaki Adımlar (v2 fikirleri)

- Hiperparametre arama (Optuna): seq_len, units, threshold'lar
- Çoklu sembol (XAUUSD, GBPUSD) — paylaşılan model + sembol embedding
- Session filter (sadece London/NY)
- Telegram notifier
- Web dashboard (FastAPI + equity curve)
- Online retrain (haftalık)
