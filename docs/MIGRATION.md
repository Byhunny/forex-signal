# Yeni PC'ye Taşıma Rehberi

Bot artık **3 şey** ile her makineye taşınır:
1. **Git repo** (kod + modeller)
2. **`.env` dosyası** (gizli, manuel kopyala)
3. **MT5 terminal** (Windows'a kur, demo hesabına bağla)

`data_cache/` ve `logs/` git'e dahil değil — yeni PC'de otomatik üretilir.

---

## A) Eski PC'de — Push Et

```powershell
cd C:\Users\<user>\projects\forex-signal
git add -A
git commit -m "snapshot before migration"
git push origin master
```

`.env` dosyasını **USB'ye veya şifre yöneticine** kopyala (git'te yok).

---

## B) Yeni PC'de — Sıfırdan Kurulum

### 1. Önkoşullar (ilk seferlik)
- **Python 3.10+** kur (Microsoft Store veya python.org)
- **MetaTrader 5** terminalini kur, XM Global demo hesabına gir (login 315983364)

### 2. Repo'yu klonla
```powershell
cd C:\Users\<yeni_user>\projects
git clone <repo_url> forex-signal
cd forex-signal
```

### 3. Python bağımlılıkları
```powershell
pip install --user -r requirements.txt
pip install --user MetaTrader5
```

### 4. .env'i kur
```powershell
copy .env.example .env
notepad .env
```

Doldur:
```
MT5_LOGIN=315983364
MT5_PASSWORD=<XM demo şifren>
MT5_SERVER=XMGlobal-MT5 7
TELEGRAM_BOT_TOKEN=<senin token>
TELEGRAM_CHAT_ID=<senin chat id>
```

### 5. Bağlantı testi
```powershell
python scripts\account_status.py
```

Beklenen: `Account login: 315983364, Equity: $...`

### 6. Smoke test (opsiyonel — sistem doğrula)
```powershell
python scripts\mt5_smoke_test.py
```

EURUSD'de 0.01 lot açar/kapatır, Telegram bildirimi gelir.

### 7. Battle royale'ı başlat
```powershell
set PYTHONPATH=src
python -m forex_signal.cli battle --mode live
```

Telegram'da `🎮 Battle Royale started — Contenders: 20` mesajını alırsan TAMAM.

---

## Eski PC'deki Pozisyonlar Ne Olur?

**Hiç sorun değil** — MT5 sunucu tarafında tutuluyor (XM server). Botu yeni PC'de açtığında:

- Mevcut açık pozisyonları (eski PC'deki magic'lerle) MT5'ten okur
- Onları yönetmeye devam eder (TP/SL'leri zaten broker enforce ediyor)
- Reversal logic'i magic ile filtrelediği için doğru pozisyonu yönetir

Aynı anda iki PC'de çalıştırma — duplicate emir verir. **Eskisini durdur, yenisini başlat.**

---

## Sorun Giderme

**`MetaTrader5` paketi import edemiyor:**
→ Yalnızca Windows. WSL veya Mac'te çalışmaz. Windows native Python kullan.

**`MT5 connect failed`:**
→ MT5 terminal açık mı? Demo hesaba bağlı mı? `.env`'deki şifre doğru mu?

**`no rates for <sembol> <TF>`:**
→ MT5'te o sembolü "Market Watch"a ekle (Right click → Show All) — broker filtrelemiş olabilir.

**Telegram mesajı gelmiyor:**
→ `.env`'de TOKEN/CHAT_ID dolu mu? `python -c "from forex_signal.config import load_config; from forex_signal.notifier.telegram import notify; load_config(); notify('test')"` ile test et.

---

## Özet — Migration için neye ihtiyacın var?

| Dosya/Kaynak | Nereden? | Zorunlu mu? |
|---|---|---|
| Kaynak kod | `git clone` | ✅ Evet |
| Eğitilmiş modeller | git'te (`models/*.pt`) | ✅ Evet |
| `.env` | manuel kopyala | ✅ Evet (şifre + token) |
| MT5 terminal | Yeni PC'ye kur | ✅ Evet (Windows) |
| Python paketleri | `pip install` | ✅ Evet |
| `data_cache/` | yeniden indirilir | ❌ Hayır |
| `logs/` | sıfırdan başlar | ❌ Hayır |

**Toplam taşıma süresi:** ~10 dakika (Python + MT5 zaten kurulmuşsa 3 dakika).
