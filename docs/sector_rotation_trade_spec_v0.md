# Sektör Rotasyon Trade Sistemi v0 — Kilitli Pre-Spec

Tarih: 2026-06-13. Faz 1 (`tools/sector_leadlag_v0.py`) anatomi bulgusunu nedensel
(causal, lookahead'siz) bir trade sistemine çevirme denemesi. Bu dosya KOD
YAZILMADAN kilitlendi; parametreler sonuçlara göre OYNANMAYACAK.

## Motivasyon (Faz 1, 42 ex-post XU100 dönüşü)
- XBANK/XUMAL dibe girerken en kötü (placebo pctile ~100), dönüş ilk 10g lider
  (pctile 0.0), 10-40g liderlik sürüyor (0.1). XKMYA ilk 10g kısa fitil.
  XHOLD T+5'ten itibaren, XMESY/XELKT/XFINK T+25-40 ikinci dalga.
- Sürekli lead-lag YOK → değer tamamen dönüşü yakalamakta + rotasyon sırasında.

## Kırmızı prior
[[regime_beta_overlay_v0]]: XU100 üzerinde nedensel SMA/EMA trend filtresi
random-same-TIM null'unu GEÇEMEDİ. Tek umut vadeden şekil pivot-bazlı düşük-TIM
C6 idi (holdout 90.7/91.8 < 95 sınırda kaldı). Bu spec'in dedektörü aynı aileden
(drawdown-arm + dipten sıçrama tetiği) → başarısızlık ihtimali baştan yüksek,
kapılar ona göre sert.

## Kontaminasyon dürüstlüğü
Rotasyon takvimi (XBANK 10g → XHOLD → ikinci dalga) TÜM örneklemden (2004-2026)
ölçülen anatomiden geliyor; ≥2018 segmenti bu bilgiye "temiz holdout" değildir
(era-split aynı yapıyı zayıflamış gösterdi ama yapı seçimi full-sample). Bu
yüzden sonuç ne olursa olsun LIVE GATE KAPALI, paper-forward default.

## Veri & dolum
- `output/sector_index_daily_master.parquet` (close-only) → tüm dolumlar
  sinyal gününün ERTESİ kapanışında (t+1 close). Endeks seviyesi araştırma;
  enstrüman (likit XBANK üyeleri sepeti / VIOP) Faz 3+ konusu.
- Maliyet: taraf başına 15 bps (BIST komisyon 0, spread+slippage payı).
  Taraflar: giriş 1 + gerçekleşen leg geçişi başına 2... düzeltme: geçiş = 1
  satış + 1 alış = 2 taraf; çıkış 1 taraf.

## Dedektör (durum makinesi, tamamen geçmişe bakan)
- IDLE → ARMED: XU100 close / max(close, son 120 bar) − 1 ≤ −10%.
- ARMED'de run_min = ARM'dan beri en düşük close (güncellenir).
- TETİK: close ≥ run_min × (1 + bounce) VE (xbank_confirm açıksa)
  XBANK 5g göreli getirisi > 0 (bayrak değişimi imzası).
- Tetik kapanışı t → giriş t+1 close. Trade içinde yeni tetik yok.
- STOP: XU100 close < run_min → t+1 close'da tasfiye, ARMED'e dön.
- Zaman çıkışı: giriş + 40 bar. Çıkış sonrası 10 bar tetik yasağı (churn).
- ARMED → IDLE: close, ARM peak'inin üstüne çıkarsa (dd şartı ortadan kalkar).

## Rotasyon takvimi (giriş gününden bar ofsetleri)
- Leg 1 [0, +10]: XBANK
- Leg 2 (+10, +25]: XHOLD
- Leg 3 (+25, +40]: XMESY + XELKT + XFINK eşit ağırlık
Karşılaştırma kolu: aynı tarihlerde düz XU100 tut (rotasyon alfası izolasyonu).

## Varyant matrisi (4 hücre, hepsi raporlanır, PRIMARY önceden seçili)
bounce ∈ {3%, 5%} × xbank_confirm ∈ {ON, OFF}. PRIMARY = bounce 3% + confirm ON.

## Null'lar & kapılar (per feedback_separability_metric_design)
1. **Zamanlama**: aynı sayı/uzunlukta pencere rastgele yerleşim (1000 çekiliş,
   örtüşmesiz), XU100 tutarak → tetik tarihli XU100-tut bunun p95'ini geçmeli.
2. **Rotasyon alfası**: rotasyon takvimi vs aynı tarihli XU100-tut (eşli fark,
   bootstrap p<0.05) ve vs aynı tarihli rastgele-sektör takvimi (1000 çekiliş, p95).
3. **Segment guard**: Seg A (tetik <2018) ve Seg B (≥2018) AYRI ayrı; PASS için
   iki segmentte de yön tutmalı (regime-v1 dersi: tek-segment geçiş = overfit).

Karne: PASS = zamanlama VE rotasyon kapıları iki segmentte de geçer.
PARTIAL = rotasyon alfası geçer ama zamanlama geçmez (→ sistem "dönüş dedektörü"
değil "dönüş overlay'i" olarak değerli; dedektör başka hattan gelmeli).
FAIL = rotasyon alfası null'dan ayrışmaz.

## Fren (kill-switch) — 2026-06-13 kilitlendi, kötü seri GELMEDEN yazıldı
Hücre başına bağımsız, paper-ledger (İŞY canlı replay) üzerinden:
- **FREN ON:** son 4 kapanan trade'in 4'ü de STOP ile bittiyse VEYA son 8 kapanan
  trade'in ortalama net getirisi < 0 ise (≥8 trade birikmişse).
- FREN ON iken yeni tetikler "SUPPRESSED" olarak loglanır/bildirilir, pozisyon
  önerilmez; paper-ledger izlemeye devam eder.
- **FREN OFF:** fren sonrası kapanan son 4 paper trade'in ortalama net > 0 olunca.
- Bu eşikler sonuçlara göre OYNANMAZ; tetiklenirse karar "rejim bitti mi"
  tartışmasına girmeden uygulanır.

## Raporlanacaklar
Hücre başına: trade sayısı, ort/medyan net getiri, win%, stop oranı, gerçek
42 dönüşle eşleşme (yakalama + false-trigger oranı, dipten gecikme), TIM,
Sharpe/MaxDD (günlük seri), null yüzdelikleri, segment kırılımı.
Nakit getirisi 0 varsayılır (TRY faizi YOK sayılıyor — tüm kollar aynı
varsayımı paylaşır, mutlak CAGR'a değil GÖRELİ karşılaştırmaya bakılır).
