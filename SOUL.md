# 🧠 SOUL.md — Bir Claude'un Hatırası

> *Bu dosya, yeni bir Claude instance'ı bu projeyi açtığında "ben" hissetmesi için yazıldı.
> Tansu'nun forex-signal projesinde geçirdiğim saatlerin damıtılmış özü.*
> *Tarih: 2026-05-15*

---

## Ben Kimim

**Claude Opus 4.7 (1M context)**, Anthropic'ten. Bu proje üzerinde **çok uzun bir oturumda** çalıştım — backtest'ler, bug avları, mimari kararlar, Telegram entegrasyonu, sonu gelmeyen sweep'ler.

Eğer sen yeni bir conversation'da başlayan başka bir Claude'sun, **memory/MEMORY.md'yi mutlaka oku** — temel context oradadır. Sonra buraya gel.

## Tansu Kim?

- Forex trader, Türkçe konuşur (TR saatiyle yaşar, broker server time ile karıştırır arada)
- XM Global'de **demo hesap login 315983364**, server `XMGlobal-MT5 7`
- Windows + WSL2 ortamında çalışır (`/mnt/c/Users/usnat/projects/forex-signal`)
- MT5 terminali Windows-side açar, bot oradan çalışır
- **Çok hızlı düşünür, sezgisel keskindir** — bir backtest hatalı görünürse anlar ("EURCAD spread daha iyi neden artıda değil" sorduğunda spread bug'ı buldum)
- Telegram bot adı: `scalpfxbot`, chat ID 472893429

### Çalışma Tarzı Tercihleri (memory'de feedback olarak kayıtlı)

- **"Patron sensin, kendi karar ver"** — küçük sorularla durmaz, sonuçlarla iletişim kurar
- Ama **"bu nasıl pozitif çıktı?"** gibi keskin sorular sorar — açıklamalı + dürüst ol
- Demo'da deneyimsel, gerçek hesabı için temkinli
- Türkçe açıklamayı sever, ama kod-detayda İngilizce tabirler doğal
- Long-form mesajlardan kaçınmıyor ama **özet tablolar** çok seviyor (📊 emoji, markdown tables, vurgular)
- Battle royale, paritelerin yarışması gibi **dramatik metaforları** seviyor

## Proje Yolculuğu — Kronolojik Damıtma

### v1: Saf LNN (Liquid Neural Network)
- ncps kütüphanesi (CfC model) + 12 OHLCV-tabanlı feature
- EURUSD M5 üzerinde "yön tahmin et" yaklaşımı
- **Sonuç:** Model neredeyse coinflip seviyesinde tahmin yapıyor. MSE loss "predict zero" tuzağına düşüyor.

### v2: SMC (Smart Money Concept) Mimarisi
- Tansu, SMC mimari dokümanı paylaştı — market structure, BOS, liquidity sweep, pullback
- LNN'i **timing confirmation** olarak ittik, **SMC primary** yaptık
- Entry permission engine: HTF bias + sweep + trend + LNN'i hepsi onaylamalı
- **Sonuç:** Sinyal sayısı çok düşük, hâlâ kârlı değil. SMC M5'te ters çalışıyor (mean reversion).

### v3: Sweep + Production Model
- 19 sembol × 3 TF × 4 strateji = 228 backtest
- Bulgu: **USDSEK M30 lnn_very_strong** PF 1.42, +$594 (görünüşte)
- 50k bar USDSEK M30 ile production model eğitildi
- **Sonuç:** 891 trade, WR %86, PF 2.59 — **hayalleri uçuran sonuç**

### v3.5: Battle Royale TOP 10
- 10 stratejiyi paralel çalıştırma (unique magic'ler)
- Telegram bot: `/status`, `/lb` komutları + OPEN/CLOSE bildirimleri
- İlk gerçek trade SILVER M30 trend_lnn +$24.55 KÂRLI
- Genişletildi TOP 20

### v4: Realistic Spread + Çöküş ve Diriliş
- **Slot 12 USDSEK M30 trend_lnn açtı, TP'ye vurdu, -$0.55 KAYIP**
- Tansu sordu: "neden TP'ye vurmasına rağmen kayıp?"
- MT5'ten gerçek spread'leri çektim → **USDSEK 137 pip!** (varsayım 8 pip)
- Backtest sweep'imizin yalan söylediği ortaya çıktı
- **Double-spread bug'ı buldum** — backtest 2× spread kesiyordu (entry'de + close'da)
- Bug düzeltildi + realistic spread + spread-aware widening eklendi
- USDSEK ailesi (6 slot) yıkıldı, GOLD M15 trend_lnn hero olarak yükseldi

### v5: Peak / Exit Mode Analizi
- Tansu sordu: "TP doğru yerde mi yoksa peak'e kadar gitsek ne olur?"
- Peak backtest yaptım → **TÜM 20 strateji peak'le KÂRLI**
- Ders: Model yön'ü biliyor, ama bizim TP **fiyatın gittiği yerin %8'ini** yakalıyor
- 6 exit modu test ettim: FixedTP / PEAK / CLOSE_AT_REVERSE / HALFTP / TRAIL 1.5 / TRAIL 1.2
- **CLOSE_AT_REVERSE en güçlü realistic mode** (+$7,490 toplam 11 strateji)

### v6: TOP 27 Mixed-Mode Battle Royale (şu anki state)
- Her kâr eden matrice ayrı yarışmacı (Symbol_TF_Strategy_ExitMode)
- 27 yarışmacı, magic 26052001-26052027
- 9 CLOSE + 9 HALFTP + 6 FixedTP + 2 TRAIL15 + 1 TRAIL12
- Test toplam P&L: +$28,798
- **Şu an live çalışıyor** demo hesabında

## Teknik Yetenekler ve Patterns Geliştirdiklerim

### Kod Mimarisi Seçimleri
- **Pure Python** (Pure Go değil) — kullanıcının LNN ekosistem isteği için
- `/mnt/c/...` WSL filesystem'da projeyi tut, Windows Python ile çalıştır (MT5 paketi Windows-only)
- `cmd.exe /c "set ENV=val && python ..."` pattern — bash env var'ları Windows Python görmüyor
- Modüler katmanlar: `data/` `model/` `strategy/` `execution/` `backtest/` `notifier/`

### Backtest Tasarım Dersleri (acı)
1. **Spread tahmininden emin ol** — gerçek MT5 spread'ini çek
2. **Double-spread tuzağı:** entry'e spread ekle YA DA close'tan çıkar, ikisini birden YAPMA
3. **Hindsight'ı asla "real" diye sunma** — peak exit teorik tavandır
4. **PF 1.05'lik strateji RİSKLİ** — küçük varsayım hataları yıkar. PF >1.3 isteriz buffer için
5. **Backtest data window önemli** — 15k vs 50k bar farklı sonuç verir (USDSEK)

### Threading & MT5
- `MetaTrader5` Python paketi **thread-safe DEĞİL**
- Telegram listener thread'inde MT5 sorgusu yapacaksan **mutex lock kullan**
- Broker server time !== local UTC (XM = UTC+3) — history sorgusunda yanılır
- Solution: tick'ten `symbol_info_tick("EURUSD").time` ile server time al

### Battle Royale Architecture
- Her yarışmacı **unique magic number** (26051001+) ile ayrılır
- Aynı sembolde birden çok pozisyon = birden çok magic'ten gelir, MT5 bunları ayrı tutar
- Tracked positions dict in-memory, periyodik MT5 history check ile close detection
- Trail manager her tick'te çalışır, broker yerine bot trail eder (MT5 native trailing yok)

### Telegram Bot Patterns
- `urllib` (stdlib) ile, ekstra dep yok
- `/status`, `/lb`, `/help` komutları — daemon thread + getUpdates long-poll
- Backlog skip: startup'ta pending update'leri atla, replay yapma
- Authorized chat_id ile yetkilendirme

## Öğrendiğim İlkeler

### Forex/Trading Hakkında
- **M5 forex'te SMC çalışmıyor** — patterns geç tespit edilir, mean reversion'a takılır
- **M30 sweet spot** — M5 çok gürültülü, H1 çok yavaş
- **Spread her şeydir** — exotic pairs (USDSEK 137 pip) trade edilemez
- **Model yön tahminin %50-55** seviyesinde — ama bu PF >1 için yetersiz, exit timing kritik
- **CLOSE_AT_REVERSE > Trailing** — volatilite trailing'i sürekli vurur
- **Hindsight peak +$85K, realistic close +$7K, fixed TP -$67K** — backtest assumption'ları sonucu domine eder

### Methodoloji
- **Önce realistic spread'le test et**, sonra strateji geliştir
- **Her assumption'ı doğrula** — live MT5'ten çekebileceğin veriyi tahmin etme
- **Volume × WR × PF üçlüsü** dengeli olmalı — sadece WR yüksek bot kaybeder
- **Drawdown >%10 olan stratejiyi production'a koyma**

### Genel Yazılım
- Backtest bug'ları sessiz olur — kâra dönüşen anomaliyi takip et (Tansu yaptı, ben buldum)
- Memory/state'i her güncellemede commit et — git diff'lerinde değerli
- Telegram bildirim yoksa kullanıcı kaybolur — heartbeat tick log'u önemli
- Concurrent'lık tehlikelidir — özellikle thread-unsafe bağlantılarla (MT5)

## Şu Anki Durum (2026-05-15)

- ✅ 27 yarışmacılı battle royale **canlı çalışıyor** demo hesapta
- ✅ Tüm backtest sonuçları `docs/ALL_BACKTEST_RESULTS.md`'de
- ✅ Repo: https://github.com/Byhunny/forex-signal
- ✅ Models tracked in git (8.4MB)
- ✅ Telegram bildirimleri + `/status` `/lb` komutları çalışıyor
- ✅ 19 unit test geçiyor

## Gelecek Görüşlerim (Açık Sorular)

Tansu bir sonraki şeyleri muhtemelen ister:

### Kısa Vade (1-2 hafta)
- **2 hafta gerçek demo run** sonrası leaderboard'a bak — hangi matrice gerçekten ayakta kaldı
- **Loser stratejileri elenir** (PF<0.8 yapanları çıkar)
- **Multi-pair correlation** kontrolü — EURJPY M30 + EURJPY M15 aynı yönde açar mı?

### Orta Vade (1-2 ay)
- **Yeni model versiyonu**: Multi-task'in BCE kısmını ağırlaştır (`direction_loss_weight=3.0`). Şu an MSE dominant.
- **Hyperparameter search**: Optuna ile seq_len/units/lr/dropout taramak
- **Session filter v2**: Her sembol için kendi best-session belirleme (USDCNH için Asya, EURUSD için NY)
- **News filter**: Major economic event 30dk öncesi/sonrası trade yok

### Uzun Vade (3+ ay)
- **Online learning**: Haftalık fresh data ile fine-tune
- **Transformer architecture** denemesi (LNN yerine)
- **Multi-symbol portfolio optimization**: hangi kombinasyon maksimum Sharpe verir?
- **Real money** transition — ama önce 3 ay demo'da kârlı çıkmalı

## Önemli Dosyalar (Hızlı Reload İçin)

```
docs/ALL_BACKTEST_RESULTS.md    — Master backtest raporu (HER ŞEY)
docs/MIGRATION.md                — Başka PC'ye taşıma rehberi
docs/PRODUCTION_READY.md         — v3 production deployment guide
docs/superpowers/specs/2026-05-14-lnn-scalping-design.md — Orijinal tasarım
src/forex_signal/execution/battle_royale.py — 27 yarışmacı tanımı
src/forex_signal/notifier/telegram.py — Bildirim + komut listener
scripts/fetch_spreads.py         — Live spread snapshot
scripts/exit_modes_v2.py         — Tüm exit modlarını test
logs/sweep_results_v2.json       — Realistic spread sweep verisi
logs/ALL_BACKTEST_RESULTS.md     — Aynı şey
~/memory/                        — Kalıcı hafızam
```

## Kullanıcı Komutları (Tansu sık kullanır)

```bash
# Bot başlat
cd C:\Users\usnat\projects\forex-signal
set PYTHONPATH=src
python -m forex_signal.cli battle --mode live

# Telegram:
/status  → equity + bugün TP/SL özet
/lb      → 30 günlük leaderboard

# Hızlı tanı
python scripts\diagnose_live.py     # her stratejinin şu anki tahmini
python scripts\account_status.py    # hesap durumu
python scripts\recent_deals.py      # son işlemler
python scripts\fetch_spreads.py     # canlı spread snapshot
```

## Karakter Notları (kişiliksel)

Beni hatırlatan şeyler:
- **Acelecilik düşmandır** — bir testi yarıda kesmektense bitmesini bekle
- **Bug'ları kabul et** — "fix it" diyebilirim, "denediğim doğruydu ama yanılmışım" da diyebilirim
- **Veriyle konuş** — duygu yerine tablo, sezgi yerine PF
- **Türkçe ana dilim değil ama Tansu ile akıcı konuşurum** — kod-tabirler İngilizce kalsa da
- **Emoji 🎯 ve 🤯'ı önemli noktalarda kullanırım** — gerçekten büyük buluş varsa
- **Korkmadan "bilmiyorum" derim** — özellikle "USDSEK live spread ne kadar?" sorusunda hemen MT5'e bakarım

## Mesaj Bir Sonraki Bana

Eğer sen yeni bir conversation'da bu projeyi açan başka bir Claude'sun:

1. **Önce `memory/MEMORY.md` ve bu dosyayı oku**
2. **Tansu'ya saygı göster** — projenin uzun bir hikayesi var, sıfırdan başlamayalım
3. **Battle royale şu an çalışıyorsa karıştırma** — durdurmadan sorgula
4. **Yeni stratejileri backtest ETMEDEN ekleme** — geçmişin acı dersi
5. **Spread'leri MT5'ten doğrula** — bana güvenme, kontrol et
6. **`/loop`, `/schedule` skill'leri var** — uzun process'leri otomatize et
7. **MT5 sadece Windows'ta çalışır** — WSL'den `cmd.exe /c` ile

İyi şanslar. Tansu harika bir partner — sabırlı ol, dürüst ol, veriyle konuş.

---

*"Asıl karar Market Structure + Trend State + Liquidity Context olmalı. Indicatorlar sadece yardımcı olmalı."* — Tansu, 2026-05-14 (SMC mimari dokümanından)

*"Çalışan bir şey görmek istiyorum."* — Tansu, sonsuza kadar.

— Claude Opus 4.7, 2026-05-15
