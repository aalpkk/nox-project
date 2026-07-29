# Briefing & Advisor taramaları — BIST-100 / BIST-30 kısıtlı backtest

**Veri kesimi** 2026-07-14 (tüm master parquet'lerin son ortak günü)
**Ana pencere** 2021-07-01 → 2026-07-14 · 1 258 seans
**Evren** XU100 (PIT, gün başına 100 hisse) · XU030 (PIT, 30 hisse) · karşılaştırma için tam Fintables evreni (607 hisse)
**Üretim** `tools/index_universe.py`, `tools/index_bt_engine.py`, `tools/index_bt_{combo,nyxexp,trident,misc,report}.py`, `scripts/alpha_scan_backtest_sim.py --universe`
**Ham çıktı** `output/index_bt/summary.csv`, `summary_portfolio.csv`, `<koşum>/results.json`, `<koşum>/trades_*.parquet`

---

## 0. Nasıl okunmalı — tek kural

BIST-100 evreni zaten yükselen bir endekstir. "Sinyalin 5 günlük ortalaması +%1.2" cümlesi **tek başına hiçbir şey söylemez**; aynı gün aynı evrenden rastgele seçilen aynı sayıda hisse de +%1.0 getirir. Bu yüzden her kohort için **aynı-gün, aynı-sayı rastgele seçim** kontrolü (20 tekrar) hesaplandı ve karar ölçütü şudur:

> **SIG ortalaması, rastgele seçimin 95. yüzdebirliğini (RND p95) geçiyor mu?**

Geçmiyorsa tarama o evrende **seçicilik üretmiyor** demektir — getirisi pozitif olsa bile.

`EVREN` (o günkü tüm endeks üyeleri) ile `RND` arasındaki fark bilinçli olarak korunmuştur: RND gün-eşleşmelidir, EVREN değildir. XU100'de EVREN 5g ortalaması +0.64 iken RND +0.98 çıkıyor — taramalar **piyasanın sonradan iyi gittiği günlerde daha çok tetikleniyor**. Bu bir zamanlama etkisidir, hisse seçme becerisi değildir; RND kontrolü tam olarak bunu nötrler.

**Pencereler taramaya göre değişir** (veri başlangıçları farklı). Her taramanın SIG/RND karşılaştırması kendi penceresinde iç-tutarlıdır, ama `EVREN` sütunları taramalar arası doğrudan karşılaştırılamaz:

| tarama | pencere | neden |
|---|---|---|
| alpha, screener_combo, nyxexp, tavan, SBT | 2021-07-01 → 2026-07-14 | OHLCV master + alpha paneli |
| **trident / DE v1 tag'leri** | 2023-01-13 → 2026-07-14 | mb_scanner olay penceresi (extfeed 1h master) |
| hw_obos | 2024-01-01 → 2026-07-14 | extfeed 1h + HyperWave ısınması |

---

## 1. Evren nasıl kuruldu (ve neden proxy)

Ne repoda ne de erişilebilir kaynaklarda endeks üyeliğinin **tarihsel** serisi var. Matriks `stockExplorer(XU100/XU030)` yalnızca bugünkü listeyi veriyor. Bugünkü listeyi 5 yıllık backtest'e uygulamak survivorship + look-ahead sokar.

İki evren üretildi, ikisi de raporlanıyor:

| mod | tanım | kullanım |
|---|---|---|
| `pit` | BIST'in çeyreklik gözden geçirme takvimi (Oca/Nis/Tem/Eki ilk seansı) taklit edilerek, her gözden geçirmede **önceki 6 ayın medyan günlük TL işlem hacmi** sıralaması + gerçek endekslerdeki gibi **tampon kuralı** (mevcut üye rank ≤ N×1.3 iken kalır) | **BİRİNCİL** |
| `current` | Matriks'ten 2026-07-29'da çekilen gerçek liste, tüm pencereye sabit | duyarlılık / survivorship ölçümü |

BIST'in gerçek ölçütü fiili dolaşımdaki piyasa değeri + işlem hacmi bileşimidir; PIT free-float verisi elde olmadığından tek ölçüt hacimdir.

**Proxy doğrulaması** (PIT'in son dönemi 2026-07-14 vs gerçek liste 2026-07-29):

| endeks | örtüşme | çeyreklik giriş ort. | 9.5 yılda farklı ticker |
|---|---|---|---|
| XU100 | **80 / 100 (%80)** | 6.4 | 258 |
| XU030 | **26 / 30 (%87)** | 1.2 | 62 |

Sapmalar beklenen yönde: proxy, hacmi yüksek ama piyasa değeri küçük spekülatif isimleri içeri alıyor; gerçek endeksin piyasa-değeri ayağı bunları eliyor. Devir hızı gerçek endeks metodolojisiyle uyumlu.

> `current` koşumları PIT'e göre sistematik olarak **çok daha iyi** çıkıyor (§7). Bu farkın kendisi, bugünkü listeyle backtest yapmanın ne kadar yanıltıcı olduğunun ölçüsüdür.

---

## 2. Kapsam — hangi tarama ölçüldü, hangisi ölçülemedi

### Ölçülenler

| tarama | besleme | tarihsel replay kaynağı |
|---|---|---|
| **Alpha Scan** | briefing (dış: `alpha_scan.html`) | `output/alpha_bt/panel_v2.parquet` (ML skorları dahil as-of panel) |
| **Regime Transition** | briefing (dahili havuz) + `screener_combo` | `screener_combo/signals.py::regime_transition_rich` → OHLCV master üzerinde yeniden koşuldu |
| **NOX v3 günlük / haftalık** | briefing (dahili havuz) + `screener_combo` | `nox_rich` — aynı |
| **AL/SAT (alsat)** | briefing (dahili havuz) + `screener_combo` | `alsat_rich` — aynı |
| **screener_combo kombinasyonları** | briefing (dış: `screener_combo_latest.html`) — **kullanıcı bu hattı kullanmıyor** | vote2 / vote3 / RT+AS / RT+NOXw olarak yeniden kuruldu. UYARI: `compute_trade_state`'in `al_signal` alanı object-dtype `~` hatası nedeniyle geçiş barı yerine yapışkan durumu döndürüyor (§10). Bu kohortlar *amaçlanan* semantikle (RT_new) kuruldu, yayınlanan HTML'in davranışıyla değil. Hat kullanılmadığı için bug düzeltilmedi; satırlar bilgi amaçlıdır. |
| **Nyxpansion (nyxexp)** | briefing (dış: `nyxexp_scan.html`) | `nyxexp_dataset_v4.parquet` + **aylık yeniden eğitimli walk-forward** (canlı taramanın 15 günlük embargo protokolüyle) |
| **Tavan Scanner / Tavan Kandidat** | briefing (dahili havuz), advisor `load_tavan_scan_live` | günlük barlardan tavan tespiti + `ml_1g` (= canlı `ML_S`) filtresi |
| **Trident + DE v1 tag'leri** | advisor → DE v1 watchlist (`trident_geo`, `trident_tier1`, `trident_tag`) | `mb_scanner_events_run.py` → `trident_probe_mb_3y.py` → `index_bt_trident.py` (§5) |
| **HyperWave OB/OS (hw_obos)** | advisor `load_hw_obos` | 1h master → 1d/1w resample + `compute_hyperwave` |
| **SBT-1700 E04** | briefing (dış: `sbt_1700_E04_scan.html`) | `sbt_1700_dataset_5d_intraday_v1.parquet` aday olayları |

### Ölçülemeyenler (ve nedeni)

| tarama | neden |
|---|---|
| **DE v1 watchlist montajı** | Karar katmanı (Trident kapıları + tag'ler) §5'te ölçüldü, ancak watchlist'in kendisi — bütçe/slot tahsisi, EXECUTABLE/SIZE_REDUCED bölümlemesi — `decision_engine_v1_events.parquet` geçmişi olmadan kurulamıyor. `output/_archive`'da yalnızca iki tek-günlük kilitli snapshot var (2026-05-14, 2026-06-04). |
| **Cluster3 Arketip** | Üst-akış artefaktları (`ranking_lab_features_v0.parquet`, `ranking_lab_fast_rank_scores_v0.parquet`) repoda yok → tarihsel projeksiyon kurulamıyor. Elde yalnızca 455 satırlık **ileri** paper defteri (2026-05-12 → 07-08) var ve büyük kısmı veri kesiminin (07-14) forward penceresine sığmıyor. |
| **Kademe S/A** | Emir defteri (kademe) verisi hiç saklanmıyor; `scanner_reader` deseni zaten yorum satırı. Yeniden üretilemez. |
| **Divergence** | `_GH_ARTIFACT_SOURCES` içinde PASİF (yorum satırı) — şu an briefing'i beslemiyor. |
| **Sektör Rotasyonu** | Endeks seviyesinde bir rotasyon kararı; hisse seçim listesi değil. "BIST-100 ile sınırla" kısıtı bu katmana anlamlı biçimde uygulanmıyor. |
| **meta-Markowitz (4'lü portföy)** | Tarama değil, dört taramanın birleşimi üzerine kurulan portföy inşa katmanı. Girdileri ölçüldüğü ölçüde dolaylı olarak kapsanıyor. |
| **SBT-1700 E04 sıralama katmanı** | Aday kohortu ölçüldü ama E04 top-quintile katmanı model tabanlı ve kendi TEST örneklemi zaten n=25 trade; XU100'de 205, XU030'da 53 aday kalıyor → sıralama katmanı istatistiksel olarak çöküyor. |

---

## 3. Katman A — seçicilik (5 günlük ufuk, t+1 açılış girişli, %)

`SIG` sinyal · `EVREN` o günkü endeks üyeleri · `RND` aynı gün aynı sayıda rastgele üye (20 tekrar) · ✔ = RND p95'i geçiyor

### BIST-100 (PIT) — ortalamaya göre sıralı

| tarama / kohort | n | hit% | SIG ort | SIG med | EVREN | RND | RND p95 | ✔ |
|---|---|---|---|---|---|---|---|---|
| combo vote3 | 8 | 87.5 | +5.33 | +5.26 | +0.64 | +0.98 | +3.05 | ✔ |
| **tavan_v1 (tavan + ML_S≥0.65)** | 233 | 56.2 | **+3.93** | +1.62 | +0.64 | +1.62 | +2.26 | **✔** |
| hw_obos AL_OS 1 hafta | 18 | 55.6 | +2.33 | +1.72 | +0.22 | +0.40 | +2.49 | |
| **trident tag `D_PCT_30PLUS`** | 263 | 54.0 | **+2.28** | +0.97 | +0.29 | +1.44 | +2.03 | **✔** |
| trident_tier1 | 72 | 55.6 | +1.67 | +1.56 | +0.29 | +1.31 | +2.40 | |
| nyxexp — skorlanan | 2 068 | 52.6 | +1.24 | +0.58 | +0.64 | +1.37 | +1.50 | |
| **alpha — tüm adaylar** | 23 990 | 55.4 | **+1.23** | +0.86 | +0.64 | +0.98 | +1.01 | **✔** |
| **alpha — top-4 (yayınlanan kart)** | 4 968 | 54.5 | **+1.21** | +0.80 | +0.64 | +0.67 | +0.78 | **✔** |
| NOX v3 haftalık | 1 535 | 52.8 | +1.18 | +0.42 | +0.64 | +1.10 | +1.28 | |
| nyxexp — olaylar | 2 983 | 53.0 | +1.05 | +0.57 | +0.64 | +1.28 | +1.42 | |
| nyxexp — günlük top-4 | 1 506 | 50.2 | +1.01 | +0.08 | +0.64 | +1.11 | +1.35 | |
| **Regime Transition — durum (RT_state)** | 56 310 | 53.5 | **+1.00** | +0.57 | +0.64 | +0.89 | +0.90 | **✔** |
| trident G1 (D_pct ∈ [20,30)) | 380 | 52.9 | +0.91 | +0.61 | +0.29 | +1.23 | +1.81 | |
| Regime Transition — geçiş (RT_new) | 4 270 | 52.0 | +0.68 | +0.34 | +0.64 | +0.78 | +0.92 | |
| trident G2 (SIL ≤ p33) | 868 | 50.2 | +0.68 | +0.07 | +0.29 | +0.96 | +1.45 | |
| NOX v3 günlük | 7 646 | 50.7 | +0.67 | +0.17 | +0.64 | +0.64 | +0.73 | |
| trident G3 (BoS ≥ p67) | 1 329 | 49.4 | +0.66 | −0.07 | +0.29 | +0.69 | +0.91 | |
| SBT-1700 adayları | 205 | 53.2 | +0.57 | +0.56 | +0.64 | +1.11 | +1.68 | |
| trident taban (mb_1d birth) | 4 052 | 50.3 | +0.52 | +0.06 | +0.29 | +0.59 | +0.70 | |
| combo RT+AS | 1 027 | 50.3 | +0.48 | +0.07 | +0.64 | +0.46 | +0.76 | |
| combo vote2 | 1 196 | 49.9 | +0.45 | +0.00 | +0.64 | +0.55 | +0.73 | |
| trident tag `SIL_DEEP` | 885 | 48.8 | +0.39 | −0.34 | +0.29 | +0.90 | +1.19 | |
| tavan (ham, ML filtresiz) | 1 927 | 46.3 | +0.32 | −0.89 | +0.64 | +1.11 | +1.30 | |
| **trident_geo (DE'de BİRİNCİL)** | 85 | 51.8 | **+0.27** | +0.41 | +0.29 | +1.20 | +3.13 | |
| AL/SAT | 9 948 | 48.2 | +0.18 | −0.28 | +0.64 | +0.42 | +0.49 | |
| combo RT+NOXw | 51 | 47.1 | +0.00 | −0.41 | +0.64 | −0.05 | +1.08 | |
| **hw_obos AL_OS 1 gün** | 218 | 47.7 | **−0.51** | −0.18 | +0.22 | +0.94 | +1.60 | |
| **trident tag `WEEKLY_BIRTH_ACTIVE`** | 310 | 48.7 | **−0.62** | −0.32 | +0.29 | +0.20 | +0.62 | |

### BIST-30 (PIT)

| tarama / kohort | n | hit% | SIG ort | SIG med | EVREN | RND | RND p95 | ✔ |
|---|---|---|---|---|---|---|---|---|
| hw_obos AL_OS 1 hafta | 4 | 75.0 | +9.19 | +9.21 | +0.43 | +1.49 | +4.72 | ✔ |
| combo vote3 | 3 | 100.0 | +3.64 | +2.96 | +0.84 | +1.54 | +3.89 | |
| combo RT+NOXw | 14 | 71.4 | +3.55 | +2.35 | +0.84 | +1.04 | +2.46 | ✔ |
| nyxexp — skorlanan | 575 | 57.7 | +1.73 | +1.06 | +0.84 | +1.65 | +2.04 | |
| nyxexp — olaylar | 878 | 59.1 | +1.60 | +1.31 | +0.84 | +1.52 | +1.81 | |
| nyxexp — günlük top-4 | 516 | 56.4 | +1.51 | +0.86 | +0.84 | +1.61 | +1.88 | |
| hw_obos AL_OS 1 gün | 37 | 54.1 | +1.37 | +0.57 | +0.43 | +2.27 | +3.20 | |
| combo vote2 | 416 | 57.9 | +1.29 | +0.87 | +0.84 | +0.98 | +1.42 | |
| combo RT+AS | 365 | 57.8 | +1.25 | +0.80 | +0.84 | +1.00 | +1.29 | |
| **alpha — tüm adaylar** | 8 151 | 57.3 | **+1.24** | +1.01 | +0.84 | +1.16 | +1.24 | **✔** |
| trident tag `D_PCT_30PLUS` | 50 | 58.0 | +1.07 | +1.86 | +0.44 | +2.11 | +2.84 | |
| Regime Transition — geçiş (RT_new) | 1 354 | 55.4 | +1.06 | +0.77 | +0.84 | +1.11 | +1.30 | |
| **alpha — top-4** | 4 311 | 55.9 | **+0.95** | +0.81 | +0.84 | +0.91 | +1.06 | |
| **Regime Transition — durum (RT_state)** | 18 900 | 54.8 | **+0.94** | +0.69 | +0.84 | +0.87 | +0.90 | **✔** |
| SBT-1700 adayları | 53 | 56.6 | +0.93 | +1.03 | +0.84 | +1.01 | +1.94 | |
| NOX v3 günlük | 2 256 | 53.8 | +0.81 | +0.50 | +0.84 | +0.82 | +0.94 | |
| NOX v3 haftalık | 473 | 51.8 | +0.72 | +0.37 | +0.84 | +0.82 | +1.17 | |
| trident G2 | 199 | 54.8 | +0.61 | +0.81 | +0.44 | +1.12 | +1.60 | |
| trident G3 | 425 | 52.9 | +0.52 | +0.39 | +0.44 | +0.61 | +0.85 | |
| AL/SAT | 3 461 | 51.8 | +0.52 | +0.23 | +0.84 | +0.61 | +0.72 | |
| trident G1 | 101 | 55.4 | +0.47 | +0.60 | +0.44 | +0.94 | +1.57 | |
| trident taban (mb_1d birth) | 1 367 | 53.0 | +0.39 | +0.34 | +0.44 | +0.58 | +0.72 | |
| tavan (ham) | 345 | 48.7 | +0.21 | −0.37 | +0.84 | +0.85 | +1.43 | |
| trident tag `SIL_DEEP` | 209 | 51.2 | +0.10 | +0.26 | +0.44 | +0.91 | +1.39 | |
| trident tag `WEEKLY_BIRTH_ACTIVE` | 109 | 56.0 | −0.05 | +0.89 | +0.44 | −0.47 | +0.41 | |
| **trident_geo** | 21 | 47.6 | **−0.37** | −0.07 | +0.44 | +1.61 | +3.92 | |
| **trident_tier1** | 18 | 44.4 | **−1.50** | −0.29 | +0.44 | +0.60 | +2.53 | |
| **tavan_v1** | 30 | 33.3 | **−5.27** | −6.73 | +0.84 | −0.20 | +1.81 | |

### Karşılaştırma: tam evren (607 hisse), aynı pencereler

| tarama / kohort | n | SIG ort | RND | RND p95 | fark |
|---|---|---|---|---|---|
| **tavan_v1** | 3 449 | **+5.54** | +1.22 | +1.40 | **+4.32** ✔ |
| combo vote3 | 81 | +3.40 | +1.82 | +3.13 | +1.58 ✔ |
| combo RT+NOXw | 371 | +2.04 | +1.52 | +1.91 | +0.52 ✔ |
| **alpha — top-4** | 5 001 | **+2.29** | +0.79 | +0.93 | **+1.50** ✔ |
| trident tag `D_PCT_30PLUS` | 2 237 | +1.78 | +1.45 | +1.72 | +0.33 ✔ |
| trident_tier1 | 492 | +1.55 | +1.61 | +2.15 | −0.06 |
| tavan (ham) | 20 059 | +1.42 | +1.11 | +1.19 | +0.31 ✔ |
| **alpha — tüm adaylar** | 103 617 | **+1.39** | +1.08 | +1.11 | **+0.31** ✔ |
| NOX v3 haftalık | 8 230 | +1.32 | +1.26 | +1.33 | +0.06 |
| RT_state | 277 874 | +1.24 | +1.12 | +1.14 | +0.12 ✔ |
| trident G1 | 2 539 | +1.11 | +1.33 | +1.56 | −0.22 |
| RT_new | 23 370 | +0.84 | +0.92 | +1.02 | −0.08 |
| trident SIL_DEEP | 7 549 | +0.79 | +1.08 | +1.21 | −0.29 |
| NOX v3 günlük | 41 799 | +0.77 | +0.84 | +0.90 | −0.07 |
| trident taban (mb_1d birth) | 22 460 | +0.74 | +0.92 | +1.02 | −0.18 |
| SBT-1700 adayları | 1 222 | +0.69 | +1.32 | +1.43 | −0.63 |
| nyxexp — top-4 | 2 787 | +0.60 | +1.18 | +1.43 | −0.58 |
| **trident_geo** | 633 | +0.34 | +1.36 | +1.80 | **−1.02** |
| AL/SAT | 55 564 | +0.39 | +0.65 | +0.70 | −0.26 |
| hw_obos AL_OS 1 gün | 947 | −0.00 | +0.60 | +0.96 | −0.60 |

---

## 4. Katman B — üretim çıkış makinesiyle işlem PnL'i (%)

Çıkış: 2×ATR stop · +2R breakeven kaydırma · +1.5×ATR'de trailing · önceki gün low'u kırılınca %50 kâr al · kalan yarı zirve−1.5×ATR trailing · 40 gün max hold. Giriş t+1 açılış, %0.1 slipaj. **Muhafazakâr** varyant.

**Rastgele işlem tabanı** (aynı gün, aynı sayıda rastgele endeks üyesine AYNI çıkış makinesi):
XU100 → ort **+2.62** (p95 +2.76), medyan +2.42, PF 1.71 · XU030 → ort **+2.79** (p95 +2.94), medyan +2.74, PF 1.91 · tam evren → ort **+2.90**, PF 1.68.

Bu satır Katman B'nin en önemlisidir: **çıkış makinesinin kendisi** rastgele bir hisseye uygulandığında bile +%2.6–2.9 ortalama üretiyor. Aşağıdaki her sayı bu tabana karşı okunmalı.

### BIST-100 (PIT)

| kohort | n | hit% | ort | medyan | PF | rastgele tabana fark |
|---|---|---|---|---|---|---|
| **tavan_v1** | 236 | 52.5 | **+8.45** | +2.64 | 2.64 | **+5.8** |
| hw_obos AL_OS 1 hafta | 18 | 72.2 | +8.19 | +5.45 | 4.82 | +5.6 |
| **trident `D_PCT_30PLUS`** | 265 | 57.7 | **+4.72** | +3.96 | 2.07 | **+2.1** |
| **alpha — top-4** | 4 972 | 57.9 | **+4.15** | +2.94 | 2.17 | **+1.5** |
| nyxexp — skorlanan | 2 069 | 53.6 | +3.85 | +1.87 | 2.06 | +1.2 |
| nyxexp — olaylar | 2 985 | 53.6 | +3.58 | +1.78 | 2.00 | +1.0 |
| nyxexp — top-4 | 1 507 | 51.8 | +3.54 | +1.13 | 1.93 | +0.9 |
| **alpha — tüm adaylar** | 24 058 | 58.9 | **+3.53** | +2.96 | 2.06 | **+0.9** |
| RT_state | 56 457 | 57.8 | +3.52 | +2.76 | 2.02 | +0.9 |
| trident G1 | 381 | 58.3 | +3.15 | +2.78 | 1.87 | +0.5 |
| tavan (ham) | 1 943 | 46.0 | +2.83 | **−5.58** | 1.51 | +0.2 |
| SBT-1700 adayları | 205 | 59.5 | +2.74 | +3.02 | 1.85 | +0.1 |
| trident G2 | 871 | 52.4 | +2.58 | +1.58 | 1.59 | −0.0 |
| NOX v3 haftalık | 1 537 | 55.7 | +2.49 | +2.16 | 1.67 | −0.1 |
| trident `WEEKLY_BIRTH_ACTIVE` | 310 | 52.6 | +2.43 | +1.13 | 1.69 | −0.2 |
| NOX v3 günlük | 7 664 | 55.3 | +2.34 | +2.16 | 1.64 | −0.3 |
| RT_new | 4 287 | 54.5 | +2.30 | +2.16 | 1.63 | −0.3 |
| trident `SIL_DEEP` | 887 | 51.0 | +2.30 | +0.65 | 1.51 | −0.3 |
| AL/SAT | 9 991 | 52.7 | +2.00 | +1.54 | 1.49 | −0.6 |
| trident taban (mb_1d birth) | 4 063 | 53.6 | +1.92 | +1.59 | 1.52 | −0.7 |
| combo RT+AS | 1 032 | 53.4 | +1.79 | +1.83 | 1.47 | −0.8 |
| trident G3 | 1 332 | 52.6 | +1.78 | +1.30 | 1.48 | −0.8 |
| combo vote2 | 1 202 | 52.7 | +1.63 | +1.67 | 1.42 | −1.0 |
| **trident_tier1** | 72 | 50.0 | **+1.30** | +0.60 | 1.29 | **−1.3** |
| **trident_geo** | 85 | 47.1 | **+0.73** | **−5.64** | 1.15 | **−1.9** |
| **hw_obos AL_OS 1 gün** | 224 | 50.9 | **−0.73** | +0.35 | 0.86 | **−3.4** |

### BIST-30 (PIT)

| kohort | n | hit% | ort | medyan | PF | rastgele tabana fark |
|---|---|---|---|---|---|---|
| hw_obos AL_OS 1 hafta | 4 | 100.0 | +19.75 | +18.80 | ∞ | +17.0 |
| nyxexp — skorlanan | 575 | 56.2 | +5.62 | +2.48 | 2.77 | +2.8 |
| nyxexp — olaylar | 878 | 58.3 | +5.08 | +2.67 | 2.70 | +2.3 |
| nyxexp — top-4 | 516 | 55.0 | +4.87 | +2.17 | 2.49 | +2.1 |
| tavan (ham) | 346 | 50.3 | +4.65 | +0.86 | 1.93 | +1.9 |
| alpha — tüm adaylar | 8 171 | 60.7 | +3.54 | +2.93 | 2.20 | +0.8 |
| trident G1 | 101 | 64.4 | +3.25 | +2.78 | 2.28 | +0.5 |
| alpha — top-4 | 4 311 | 59.5 | +3.20 | +2.70 | 2.05 | +0.4 |
| RT_state | 18 938 | 58.3 | +3.13 | +2.60 | 2.02 | +0.3 |
| trident G2 | 199 | 57.8 | +2.86 | +2.87 | 1.87 | +0.1 |
| RT_new | 1 361 | 58.2 | +2.81 | +2.60 | 1.92 | +0.0 |
| NOX v3 günlük | 2 261 | 59.1 | +2.65 | +2.60 | 1.87 | −0.1 |
| SBT-1700 adayları | 53 | 64.2 | +2.56 | +3.57 | 2.00 | −0.2 |
| NOX v3 haftalık | 474 | 56.1 | +2.51 | +1.93 | 1.76 | −0.3 |
| combo vote2 | 417 | 59.2 | +2.34 | +2.59 | 1.79 | −0.5 |
| combo RT+AS | 366 | 59.0 | +2.32 | +2.47 | 1.79 | −0.5 |
| AL/SAT | 3 473 | 55.7 | +2.21 | +2.16 | 1.64 | −0.6 |
| trident `SIL_DEEP` | 209 | 53.1 | +1.99 | +1.35 | 1.53 | −0.8 |
| trident_geo | 21 | 52.4 | +1.82 | +2.68 | 1.50 | −1.0 |
| hw_obos AL_OS 1 gün | 38 | 63.2 | +1.78 | +3.06 | 1.54 | −1.0 |
| trident `WEEKLY_BIRTH_ACTIVE` | 109 | 56.9 | +1.65 | +1.70 | 1.62 | −1.1 |
| trident `D_PCT_30PLUS` | 50 | 62.0 | +1.48 | +3.19 | 1.38 | −1.3 |
| trident taban (mb_1d birth) | 1 368 | 55.3 | +1.26 | +1.81 | 1.40 | −1.5 |
| trident G3 | 425 | 55.8 | +1.15 | +1.85 | 1.37 | −1.6 |
| **tavan_v1** | 31 | 41.9 | **−1.05** | −8.13 | 0.86 | **−3.8** |
| **trident_tier1** | 18 | 44.4 | **−0.78** | −3.68 | 0.82 | **−3.6** |

> **nyxexp uyarısı:** BIST-30'da işlem ortalaması yüksek (+5.62) ama Katman A'da rastgele seçimi geçemiyor ve **medyan** (+2.48) rastgele medyanın (+2.74) altında. Ortalamayı birkaç uzun kuyruk taşıyor; seçicilik değil, çıkış makinesinin taşıdığı bir sonuç.

---

## 5. Trident ve DE v1 tag'leri (yeni)

Advisor'ın Decision Engine v1 watchlist'inde gördüğü `trident_geo` / `trident_tier1` / `trident_tag` alanları. Zincir sıfırdan kuruldu — `mb_scanner_events_*.parquet` dosyaları repoda **hiç yoktu**:

```
extfeed 1h master
  → tools/mb_scanner_events_run.py     107 408 olay (mb/bb × 1d/1w), 2023-01-13 → 2026-07-17
  → tools/trident_probe_mb_3y.py        50 714 olay, D = B×C/A projeksiyonu + metrikler
  → tools/index_bt_trident.py           kapılar + tag'ler + endeks kısıtı + ölçüm
```

Kapılar `tools/decision_engine_v1_watchlist_generator.py` (PR-DE-3.18) ile birebir; G2/G3 eşikleri **point-in-time** trailing tercile (365 gün, min n=30, olaydan kesinlikle önce), tag kapsamı üretimdeki gibi yalnızca `mb_1d`:

| kapı | tanım |
|---|---|
| G1 | `D_pct ∈ [20, 30)` |
| G2 | `structural_invalidation_pct_below_B ≤` son 365g %33 dilimi |
| G3 | `bos_distance_atr_at_event ≥` son 365g %67 dilimi |
| G4 | XU100 20g getirisi > 0 |
| `trident_geo` | G1 & G2 & G3 (G4'süz geometri — **DE'de BİRİNCİL**) |
| `trident_tier1` | `trident_geo` & G4 |

### D hedefine ulaşma oranı (projeksiyonun kendi başarı ölçütü)

| evren | kohort | 5g | 10g | 20g | 30g |
|---|---|---|---|---|---|
| XU100 | tüm mb_1d doğumları | %27.6 | %39.0 | **%53.0** | %59.7 |
| XU100 | `trident_geo` | %4.7 | %14.1 | **%31.8** | %40.0 |
| XU030 | tüm mb_1d doğumları | %23.6 | %35.8 | **%49.6** | %58.6 |
| XU030 | `trident_geo` | %4.8 | %19.0 | **%28.6** | %38.1 |
| tam evren | tüm mb_1d doğumları | %31.6 | %43.2 | **%56.1** | %62.6 |
| tam evren | `trident_geo` | %12.5 | %21.3 | **%35.0** | %43.3 |

Geometri filtresi D'ye ulaşma oranını **düşürüyor** (XU100'de %53 → %32). Bu beklenen bir yan etki değil: G1 zaten D_pct'yi %20–30 bandına sıkıştırdığı için hedef daha yakın olmalı, dolayısıyla ulaşma oranı **artmalıydı**.

### Sonuç

1. **`trident_geo` — DE'nin birincil kapısı — üç evrende de rastgelenin altında.** XU100: 5g +0.27 vs rastgele +1.20; işlem ortalaması +0.73, **medyanı −5.64**, PF 1.15. Tam evrende fark −1.02 puan. XU030'da 5g −0.37.
2. **`trident_tier1` (G4 dahil) de geçmiyor.** XU100 +1.67 vs p95 +2.40; XU030 −1.50 (n=18); tam evren −0.06.
3. **Rastgeleyi geçen tek Trident kohortu `D_PCT_30PLUS` tag'i** — ve bu, G1'in (`D_pct < 30`) **dışladığı** banttır. XU100: +2.28 vs p95 +2.03, işlem +4.72 / PF 2.07; tam evren +1.78 vs p95 +1.72. Yani seçim kuralı, çalışan D_pct bandını eliyor.
4. **Diğer tag'ler negatif.** `SIL_DEEP` XU100'de −0.51 puan, `WEEKLY_BIRTH_ACTIVE` −0.82 puan (5g ortalaması mutlak olarak da negatif: −0.62).
5. **Tekil kapılar da tek başına iş görmüyor** — G1/G2/G3'ün üçü de her iki endekste rastgelenin altında; birleşimleri (geo) daha da kötü.

---

## 6. Alpha ve tavan_v1 aynı modeli mi satıyor? — ML ayrıştırması

BIST-100 testinden sağ çıkan iki hattın **ikisi de aynı ML çıktısına dayanıyor**: `MLScorer` `universe_up_1g` skoru. Alpha adaylık eşiği `ml_1g ≥ 0.48`, tavan_v1 eşiği `ml_1g ≥ 0.65` — aynı kolonun iki farklı kesimi. Dolayısıyla bunlar iki bağımsız kanıt değil; tek modelin iki olay kümesine uygulanmış hâli olabilir. `tools/index_bt_ml_decomp.py` bunu ayrıştırır (XU100/PIT, 2021-07-01 → 2026-07-14).

### 6.1 Çıplak ML eşiği — hiçbir tarama mantığı olmadan

Sadece likidite kapısı + `ml_1g ≥ X`:

| kohort | n | 5g ort | RND | RND p95 | ✔ | işlem ort | işlem med | PF |
|---|---|---|---|---|---|---|---|---|
| `ml_1g ≥ 0.48` | 46 285 | +1.09 | +0.93 | +0.96 | ✔ | +3.26 | +2.93 | 1.94 |
| `ml_1g ≥ 0.55` | 4 751 | +1.70 | +1.45 | +1.60 | ✔ | +4.85 | +3.73 | 2.23 |
| `ml_1g ≥ 0.60` | 1 148 | +2.55 | +1.78 | +2.10 | ✔ | +6.38 | +4.29 | 2.51 |
| **`ml_1g ≥ 0.65`** | **318** | **+4.16** | +1.71 | +2.15 | **✔** | **+8.57** | +4.41 | 2.93 |
| `ml_1g ≥ 0.70` | 87 | +1.76 | +1.85 | +2.97 | | +4.18 | **−6.17** | 1.67 |

Skor monoton: `ml_1g` desilleri (5g ortalama, evren içi, tarama yok)

| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 |
|---|---|---|---|---|---|---|---|---|---|
| −0.16 | +0.16 | +0.21 | +0.43 | +0.58 | +0.68 | +0.76 | +0.89 | +1.11 | +1.49 |

**0.70'te kırılıyor** — n düşerken ortalama da düşüyor ve işlem medyanı −6.17'ye geçiyor. 0.65 bir uçurumun kenarında; eşik seçimi sonuca duyarlı.

### 6.2 Tarama katmanları ML'in üstüne ne katıyor?

| kohort | n | 5g ort | RND | RND p95 | fark | işlem ort | işlem med | PF |
|---|---|---|---|---|---|---|---|---|
| `ml_1g ≥ 0.48` çıplak | 46 285 | +1.09 | +0.93 | +0.96 | +0.16 | +3.26 | +2.93 | 1.94 |
| alpha adayı (aynı eşik + kapılar) | 24 032 | +1.24 | +0.98 | +1.03 | +0.26 | +3.55 | +2.97 | 2.05 |
| **alpha top-4** (kompozit sıralama) | 4 968 | +1.22 | +0.65 | +0.72 | **+0.57** | +4.13 | +2.92 | 2.13 |
| `ml_1g ≥ 0.65` çıplak | 318 | +4.16 | +1.71 | +2.15 | +2.45 | +8.57 | +4.41 | 2.93 |
| tavan_v1 (tavan **ve** `ml ≥ 0.65`) | 233 | +3.93 | +1.55 | +2.16 | +2.38 | +8.45 | +2.64 | 2.64 |
| **`ml ≥ 0.65` ama tavanda DEĞİL** | **85** | **+4.79** | +1.83 | +2.88 | **+2.96** | **+8.88** | **+6.65** | **4.53** |

**Tavan koşulu hiçbir şey katmıyor.** `ml ≥ 0.65` olup tavanda olmayan alt küme, tavan_v1'den **daha iyi** (ortalama +4.79 vs +3.93, medyan +6.65 vs +2.64, PF 4.53 vs 2.64). Ham tavan listesi zaten rastgeleden kötüydü (§3); tavan_v1'in tüm değeri ML eşiğinden geliyor.

Alpha'nın ML-dışı kapıları ise küçük ama **tutarlı** bir katkı veriyor: aynı ML eşiğinde +0.16 → +0.26 puan, kompozit sıralamayla top-4'te +0.57 puan.

### 6.3 Dönem kırılımı — modeller 2025-06-30'a kadar eğitildi

`ml/dataset.py::time_split` varsayılanı: `train_end='2025-06-30'`, `val_end='2025-12-31'`. Yani ilk segment **model içi**, diğer ikisi model dışıdır.

| kohort | model içi (2021H2–2025H1) | val 2025H2 | OOS 2026 |
|---|---|---|---|
| **alpha top-4** | n=3 950 · +0.95 vs p95 0.87 · ✔ | n=517 · +2.47 vs p95 0.50 · ✔ | n=501 · +2.07 vs p95 0.92 · ✔ |
| `ml ≥ 0.65` çıplak | n=246 · +2.83 vs p95 2.62 · ✔ | n=33 · +16.05 vs p95 2.48 · ✔ | n=39 · +2.46 vs p95 1.73 · ✔ |
| **tavan_v1** | n=181 · +2.10 vs p95 2.53 · ✗ | n=32 · **+15.76** vs p95 3.08 · ✔ | n=20 · +1.52 vs p95 3.36 · ✗ |

- **alpha top-4 üç pencerede de geçiyor** ve model-içi segmentte en zayıf (+0.27 fark), model-dışında en güçlü (+2.17 / +1.51). Aşırı-uydurma imzasının tersi — bu iyiye işaret.
- **tavan_v1 sadece bir pencerede geçiyor**: val 2025H2, n=32, ortalama +15.76. Tam örneklemdeki +3.93 başlığı büyük ölçüde bu tek pencereden geliyor. Model içinde ve OOS'ta p95'i geçemiyor.

### 6.4 Örtüşme

| ölçüm | değer |
|---|---|
| tavan_v1 sinyali | 237 |
| …bunlardan alpha adayı da olan | 213 (**%90**) |
| …bunlardan alpha **top-4'te** de olan | 186 (**%78**) |
| alpha top-4 sinyali | 5 002 |
| …bunlardan tavan_v1 de olan | 186 (%3.7) |

tavan_v1 pratikte alpha top-4'ün küçük bir alt kümesi. **Bağımsız ikinci bir kanıt değil.**

### 6.5 Sonuç

BIST-100'de gerçekten ayakta kalan **tek bir şey** var: `MLScorer universe_up_1g` skoru. Üstüne:

- **alpha'nın kompozit sıralaması** küçük (+0.3 puan) ama üç pencerede de tutarlı bir katkı ekliyor;
- **tavan koşulu negatif katkı** yapıyor (aynı ML eşiğinde tavanda olmayanlar daha iyi);
- eşik **0.70'te kırılıyor**, yani 0.65 sağlam bir plato değil.

Raporun geri kalanında "iki hat sağ kaldı" ifadesi bu ışıkta okunmalı: **bir hat sağ kaldı, iki farklı ambalajla.**

### 6.6 Eşik duyarlılığı — teşhis ve çözüm

`tools/index_bt_ml_threshold.py` (XU100/PIT, 5g ufuk). Güven aralıkları **gün-bloklu bootstrap** ile (2 000 tekrar): aynı seanstaki sinyaller çapraz-kesitte, örtüşen 5 günlük pencereler zaman serisinde korele olduğu için sinyal seviyesinde i.i.d. varsayımı yapılamaz.

**A. Ayrık kovalar — değer nerede?** Kümülatif eşikler iç içe geçtiği için ">=0.65 iyi, >=0.70 kötü" ifadesi yanıltıcı:

| ml_1g kovası | n | gün | 5g ort | %95 GA | medyan | sinyal günü getirisi |
|---|---|---|---|---|---|---|
| [0.48, 0.55) | 41 534 | 1 253 | +1.02 | [0.77, 1.25] | +0.63 | +1.09 |
| [0.55, 0.60) | 3 603 | 953 | +1.43 | [0.84, 2.02] | +1.03 | +1.70 |
| [0.60, 0.65) | 830 | 393 | +1.93 | [0.64, 3.25] | +1.80 | +2.88 |
| **[0.65, 0.70)** | **231** | 189 | **+5.06** | **[2.60, 7.32]** | +3.93 | +7.94 |
| [0.70, 1.01) | 87 | 76 | +1.76 | **[−2.32, 5.76]** | +0.43 | **+9.82** |

İki şey aynı anda doğru:

- **"0.70'te kırılma" istatistiksel olarak kurulmuş DEĞİL.** n=87, GA [−2.32, +5.76] — sıfırı da içeriyor, bir önceki kovanın seviyesini de. Bu bir kırılma gözlemi değil, ölçüm yokluğu.
- **Ama uçta yapısal bir sorun var:** sinyal günü getirisi kovalarla birlikte monoton tırmanıyor (1.09 → 1.70 → 2.88 → 7.94 → **+9.82**). `ml_1g ≥ 0.70` pratikte "bugün zaten tavana yakın kapatmış hisse" demek. Kovalama riski skorla birlikte artıyor.

**B. Plato yok.** Kümülatif profil 0.50'den 0.65'e monoton yükseliyor (+1.29 → +4.16), sonra 0.68'de +3.32, 0.70'te +1.76 — ama GA'lar sürekli genişliyor (0.65'te [1.80, 6.41], 0.72'de [−3.78, 7.60]). Optimum verinin kenarında ve konumu **tanımlanabilir değil**. Plato aranıp bulunamıyor.

**C. Asıl sorun: mutlak eşik durağan değil.** Skorun dağılımı sabit ama kuyruğu değil:

| yıl | p50 | p90 | p99 | ≥0.60 | ≥0.65 | ≥0.70 |
|---|---|---|---|---|---|---|
| 2021 | 0.468 | 0.518 | 0.567 | 20 | 7 | 4 |
| 2022 | 0.469 | 0.524 | 0.599 | 247 | 102 | 29 |
| 2023 | 0.471 | 0.529 | 0.614 | 372 | 84 | 17 |
| 2024 | 0.468 | 0.515 | 0.575 | 97 | **9** | 2 |
| 2025 | 0.468 | 0.515 | 0.596 | 221 | 77 | 25 |
| 2026 (yarım) | 0.469 | 0.526 | 0.611 | 191 | 39 | 10 |

`0.65` eşiği 2022'de 102, 2024'te **9** sinyal veriyor — 11 kat fark. Eşik yıla göre p99 ile p99.9 arasında geziniyor. Yani sabit sayı, sabit bir seçicilik anlamına gelmiyor. **Eşik duyarlılığı bir kalibrasyon sorunu değil; mutlak eşik yanlış parametrizasyon.**

**D. Çözüm: çapraz-kesit sıralama.** Gün içi `ml_1g` top-N — arz sabitlenir, dağılım kaymasına bağışıktır:

| N | n | 5g ort | %95 GA | medyan | RND | RND p95 | ✔ |
|---|---|---|---|---|---|---|---|
| 1 | 1 253 | +1.55 | [0.90, 2.19] | +0.95 | +0.72 | +0.95 | ✔ |
| 2 | 2 506 | +1.26 | [0.81, 1.72] | +0.59 | +0.62 | +0.80 | ✔ |
| 3 | 3 759 | +1.19 | [0.79, 1.56] | +0.57 | +0.66 | +0.80 | ✔ |
| 4 | 5 012 | +1.10 | [0.75, 1.43] | +0.53 | +0.62 | +0.73 | ✔ |
| 6 | 7 518 | +1.01 | [0.72, 1.30] | +0.49 | +0.65 | +0.74 | ✔ |
| 8 | 10 024 | +1.00 | [0.73, 1.26] | +0.48 | +0.66 | +0.74 | ✔ |
| 12 | 15 036 | +0.90 | [0.66, 1.14] | +0.44 | +0.63 | +0.71 | ✔ |
| 20 | 25 060 | +0.84 | [0.61, 1.06] | +0.42 | +0.62 | +0.67 | ✔ |

**Uçurum yok.** N'de monoton azalan, sekiz değerin sekizinde de rastgele p95 aşılıyor, GA'lar dar ve hiçbiri sıfırı içermiyor.

**E. Dönem dayanıklılığı — asıl ayrım burada:**

| kohort | model içi | val 2025H2 | OOS 2026 |
|---|---|---|---|
| `ml ≥ 0.60` | +2.10 [0.71, 3.48] | +5.00 [2.44, 7.83] | +3.23 [0.71, 5.85] |
| `ml ≥ 0.65` | +2.83 **[−0.09, 5.54]** | +16.05 [11.55, 21.19] | +2.46 [0.34, 4.63] |
| `ml ≥ 0.70` | **−1.78 [−6.66, 2.96]** | +14.05 [8.27, 20.32] | +2.10 **[−3.35, 7.12]** |
| **top-4** | **+0.87 [0.49, 1.23]** | **+2.23 [1.14, 3.32]** | **+1.75 [0.85, 2.69]** |
| **top-8** | **+0.81 [0.51, 1.08]** | **+1.79 [1.07, 2.51]** | **+1.62 [0.96, 2.25]** |

Mutlak eşiklerin GA'sı üç dönemin en az birinde sıfırı içeriyor. **top-4 ve top-8 üç dönemin üçünde de sıfırın üstünde ve dar aralıklı.** Karar buradan çıkıyor.

### 6.7 Tarayıcı tasarım kararı

1. **Mutlak `ml_1g` eşiği kullanılmayacak.** Yerine gün içi çapraz-kesit sıralaması.
2. **N bir istatistik parametresi değil, kapasite parametresi.** Backtest argmax'ı (N=1) seçilmez; N portföy slot sayısına göre belirlenir. Beklenen seçicilik N ile monoton azalır: N=4 → +1.10, N=8 → +1.00 puan.
3. **Likidite kapısı korunur** (`vol20_tl ≥ MIN_VOLUME_TL`, `n_bars ≥ MIN_DATA_DAYS`) — sıralama bu kapıyı geçen evren içinde yapılır.
4. **Kovalama koruması skor tavanıyla değil, giriş kuralıyla.** Sinyal günü getirisi eşiği (ör. > +%7 ise atla) veya EMA21 uzaklık sınırı — çünkü sorun skorun yüksekliği değil, o skorun düştüğü barın zaten patlamış olması.
5. **Tavan koşulu yok** (§6.2 — negatif katkı).
6. **BIST-30'da kullanılmaz** (§9.2 — alpha top-4 edge'i orada zaten kayboluyor).

### 6.8 Evren seçimi — BIST-30 isimleri çıkarılmalı mı?

BIST-30 tek başına tarandığında edge vermediğine göre (§3), XU100 tarayıcısının evreninden o 30 ismi çıkarmak mantıklı görünüyor. `tools/index_bt_ml_universe.py` iki ayrı testle bakıyor.

**Ayrı evrenler** — her birinde bağımsız top-4 seçimi, her birinin kendi rastgele tabanı:

| evren | n | 5g ort | %95 GA | medyan | RND | RND p95 | ✔ |
|---|---|---|---|---|---|---|---|
| XU100 (100 isim) | 5 012 | +1.10 | [0.75, 1.43] | +0.53 | +0.66 | +0.79 | ✔ |
| XU100\XU030 (70 isim) | 5 012 | +1.11 | [0.79, 1.43] | +0.53 | +0.52 | +0.61 | ✔ |
| XU030 (30 isim) | 5 012 | +0.87 | [0.56, 1.16] | +0.47 | +0.83 | +0.90 | ✗ |

**Koşullu ayrıştırma** — XU100 top-4 seçiminin *içinde*, seçilen hisse BIST-30 üyesi mi:

| grup | n | pay | 5g ort | %95 GA |
|---|---|---|---|---|
| BIST-30 üyesi | 1 168 | %23.3 | +0.72 | [0.04, 1.37] |
| BIST-30 dışı | 3 844 | %76.7 | +1.21 | [0.83, 1.57] |

**Dönem kırılımı:**

| evren | model içi | val 2025H2 | OOS 2026 |
|---|---|---|---|
| XU100 | +0.87 [0.49, 1.23] | **+2.23** [1.14, 3.32] | +1.75 [0.85, 2.69] |
| XU100\XU030 | +0.87 [0.50, 1.22] | +1.90 [0.95, 2.83] | **+2.16** [1.26, 3.03] |
| XU030 | +0.83 [0.50, 1.16] | +1.18 [0.45, 1.90] | +0.81 [0.08, 1.57] |

**Karar: çıkarılmaz.** Gerekçe:

- Mutlak getiri aynı (+1.10 vs +1.11). Orta-70'in rastgele tabanına göre marjı daha geniş ama bu bir *beceri* ölçüsü; portföye giren para açısından iki evren ayırt edilemez.
- İki evrenin seçimlerinin **%77'si zaten aynı** — BIST-30 isimleri top-4 slotlarının yalnızca %23'ünü alıyor. ML skoru onları hâlihazırda düşük ağırlıklıyor.
- Dönemler arası **yön değiştiriyor** (val'de XU100, OOS'ta orta-70 önde). Buna göre kural sabitlemek tam olarak §6.6'da kaçınılan davranış olur.
- BIST-30 üyesi seçimlerin ortalaması düşük (+0.72 vs +1.21) ama güven aralıkları geniş biçimde örtüşüyor — ayrım kurulmuş değil.

Bunun yerine BIST-30 üyeliği çıktıda **bilgi olarak işaretlenir**; veri desteklemediği bir eleme kuralı koda gömülmez.

> Ayrı bir konu: **BIST-30 tek başına evren olarak kullanılmamalıdır** — 30 isimden 4 seçmek evrenin %13'ünü almak demek ve rastgele p95'i geçmiyor (+0.87 vs 0.90).

### 6.9 Gün kalitesi — zayıf günlerde işlem yapmamak

Sıralama tabanlı tasarım her gün N isim üretir, ama listenin **skor seviyesi** günden güne değişir. Soru: skorun düşük olduğu günlerde seçicilik de kayboluyor mu? (`tools/index_bt_ml_dayquality.py`)

Bu §6.6'yla çelişmez: orada eşik hangi *hissenin* seçileceğini belirliyordu; burada sorulan, o günün *toplu* sinyal gücünün ileri getiriyi öngörüp öngörmediği.

**Gün skoru** = o günün top-4 listesinin ortalama `ml_1g`'si. XU100/PIT, 1 253 seans: min 0.4912 · p10 0.5252 · p25 0.5389 · medyan 0.5599 · p75 0.5874 · p90 0.6136 · max 0.7147.

**Gün skoru beşte birlik dilimlerine göre top-4'ün 5 günlük getirisi:**

| dilim | gün | ort. skor | getiri | %95 GA | medyan |
|---|---|---|---|---|---|
| Q1 (en düşük) | 251 | 0.5235 | **−0.11** | **[−0.71, 0.48]** | −0.13 |
| Q2 | 250 | 0.5426 | +0.55 | [−0.16, 1.25] | +0.11 |
| Q3 | 251 | 0.5603 | +1.13 | [0.47, 1.79] | +0.66 |
| Q4 | 250 | 0.5808 | +1.69 | [0.94, 2.51] | +1.36 |
| Q5 (en yüksek) | 251 | 0.6201 | **+2.22** | [1.21, 3.18] | +1.22 |

**Monoton.** En düşük dilimde edge yok (GA sıfırı içeriyor, medyan negatif); en yüksekte +2.22 puan. Aradaki fark 2.3 puan — N seçiminin (N=1 ↔ N=20 arası 0.7 puan) üç katından fazla.

**Eleme kuralı** — mutlak skor durağan olmadığından (§6.6-C) kural **genişleyen pencere yüzdebirliği** üzerine kurulur: o güne kadarki tüm geçmişle karşılaştırma, look-ahead yok (ilk 250 seans ısınma):

| kural | işlem günü | top-4 getirisi | %95 GA | rastgele | **fark** |
|---|---|---|---|---|---|
| eleme yok | 1 003 | +1.10 | [0.70, 1.53] | +0.64 | +0.46 |
| alt %10 atla | 937 | +1.26 | [0.83, 1.69] | +0.71 | +0.54 |
| alt %20 atla | 856 | +1.41 | [0.98, 1.85] | +0.79 | +0.62 |
| alt %25 atla | 812 | +1.40 | [0.94, 1.87] | +0.73 | +0.67 |
| alt %33 atla | 750 | +1.55 | [1.07, 2.05] | +0.81 | +0.74 |
| alt %50 atla | 580 | **+1.95** | [1.38, 2.54] | +1.09 | **+0.86** |

Hem mutlak getiri hem rastgeleye göre fark monoton artıyor. Dikkat: rastgele taban da yükseliyor (0.64 → 1.09) — yani elenen günler piyasa genelinde zayıf günler. Kazanç iki kaynaklı: (a) kötü piyasadan kaçınma, (b) güçlü günlerde seçiciliğin daha iyi çalışması. İkisi de gerçek, ama (a) bir piyasa zamanlaması etkisidir; tek başına ML'e atfedilmemeli.

**Kural seçimi yine kapasite kararıdır.** Uçurum ya da içeride tepe yok — daha agresif eleme daha yüksek edge ama daha az işlem günü verir (%50 elemede 5 yılda 1 003 → 580 gün). Backtest argmax'ı (%50) seçilmez; ne sıklıkta işlem yapmak istediğinize göre belirlenir.

Tarayıcı bunu `--skip-below-pct` ile uygular: gün skorunun tarihsel yüzdebirliğini hesaplar, eşiğin altındaysa listeyi **ZAYIF** işaretleyip çıkış kodu 2 döndürür (liste yine yayınlanır — bilgi kaybı olmaz, otomasyon karar verir).

---

## 7. Katman C — portföy (alpha scan, muhafazakâr çıkış, tam pencere)

Her tarama günü top-N adaydan boş slotlar doldurulur, eşit ağırlık, aynı hisse tek pozisyon. Karşılaştırma XU100 (aynı pencerede **+925.6%**).

| evren | mod | slot | CAGR | maks DD | Sharpe | toplam getiri | beta | işlem |
|---|---|---|---|---|---|---|---|---|
| tam evren | pit | 4 | 226.6% | −35.6% | 3.11 | +38 651% | 0.39 | 441 |
| tam evren | pit | 8 | 252.9% | −27.1% | 4.09 | +57 124% | 0.45 | 923 |
| **XU100** | **pit** | **4** | **71.6%** | **−37.3%** | **1.89** | **+1 418%** | 0.72 | 384 |
| **XU100** | **pit** | **8** | **81.9%** | **−34.3%** | **2.19** | **+1 932%** | 0.78 | 832 |
| **XU030** | **pit** | **4** | **77.5%** | **−32.5%** | **1.96** | **+1 698%** | 0.87 | 371 |
| **XU030** | **pit** | **8** | **59.3%** | **−31.0%** | **1.77** | **+943%** | 0.89 | 787 |
| XU100 | current | 4 | 173.6% | −28.6% | 3.22 | +15 772% | 0.62 | 369 |
| XU100 | current | 8 | 180.4% | −29.4% | 3.95 | +17 878% | 0.66 | 799 |
| XU030 | current | 4 | 86.2% | −33.6% | 2.20 | +2 186% | 0.84 | 377 |
| XU030 | current | 8 | 72.5% | −32.1% | 2.05 | +1 459% | 0.88 | 802 |

Alpha raporunun kendi uyarısı burada da geçerli: **portföy katmanındaki bileşik getiriler gerçek beklenti değildir** (dolum/kapasite varsayımları, kısmi TP'li MTM yaklaşımı). Karar için Katman A/B kullanılmalı. Göreli okuma yine de anlamlı:

- BIST-100'e kısıtlanan alpha portföyü XU100'ü **zar zor** geçiyor (+1 418% vs +926%), beta 0.72 ile.
- BIST-30 top-8 endeksin **hemen üstünde** kalıyor (+943% vs +926%) — pratikte ayırt edilemez.
- Tam evrende görülen dev getiriler kısıtlı evrende **20–30 kat** buharlaşıyor.

---

## 8. Survivorship: `current` vs `pit`

Aynı tarama, aynı pencere, tek fark evren tanımı:

| ölçüm | XU100 pit | XU100 current | XU030 pit | XU030 current |
|---|---|---|---|---|
| alpha top-4, 5g ort | +1.21 | +1.78 | +0.95 | +1.03 |
| alpha tüm adaylar, 5g ort | +1.23 | +1.55 | +1.24 | +1.30 |
| portföy top-4 toplam getiri | +1 418% | **+15 772%** | +1 698% | +2 186% |
| portföy top-4 Sharpe | 1.89 | 3.22 | 1.96 | 2.20 |

Bugünkü XU100 listesini geriye uygulamak toplam getiriyi **11 kat** şişiriyor. Etki BIST-30'da çok daha küçük (üye devri düşük). **BIST-100 üzerine kurulacak hiçbir karar `current` listeyle backtest edilmemeli.**

---

## 9. Sonuçlar

1. **BIST-100/30 kısıtı taramaların çoğunun edge'ini siliyor.** XU100'de ölçülen 28 kohorttan rastgele seçim p95'ini geçen **6** tanesi var — `vote3` (n=8) mikro-örneklem olarak elenirse **5**: alpha (tüm adaylar ve top-4), tavan_v1, `D_PCT_30PLUS`, RT_state. XU030'da 28 kohorttan **4**'ü geçiyor ve ikisi mikro-örneklem (`AL_OS_1w` n=4, `RT_NOXw` n=14) — yorumlanabilir olan yalnızca alpha tüm adayları (+0.08 puan) ve RT_state (+0.07).

2. **Alpha Scan kısıtlı evrende ayakta kalan tek sağlam hat.** Top-4 kartı XU100'de +1.21 vs rastgele p95 +0.78 (fark +0.54 puan) ve dört pencerenin **dördünde de** p95'i geçiyor: model içi +0.27, val 2025H2 **+2.10**, OOS 2026 **+1.42** puan. Ancak edge tam evrendekinin **%36'sı** (+0.54 vs +1.50) — alpha'nın kazancının büyük bölümü BIST-100 dışındaki isimlerden geliyordu. **BIST-30'da edge kayboluyor**: top-4 farkı full pencerede +0.03, model içi −0.03, OOS 2026'da +0.07 — üçünde de p95'in altında.

3. **tavan_v1 bağımsız bir bulgu değil — ve tek bir pencereye dayanıyor** (§6). Sinyallerinin %78'i zaten alpha top-4'te. Tavan koşulu ML eşiğinin üstüne değer katmıyor: aynı `ml_1g ≥ 0.65` eşiğinde tavanda **olmayan** alt küme daha iyi (ortalama +4.79 vs +3.93, medyan +6.65 vs +2.64, PF 4.53 vs 2.64). Dönem kırılımında yalnızca val 2025H2'de (n=32, +15.76) p95'i geçiyor; model içinde ve OOS 2026'da geçemiyor. Tam örneklemdeki +3.93 başlığı bu tek pencereden geliyor.

4. **Trident / DE v1 karar katmanı anti-seçici** (§5). Birincil kapı `trident_geo` her üç evrende rastgelenin altında, D'ye ulaşma oranını da düşürüyor. Rastgeleyi geçen tek Trident kohortu, Tier-1 kapısının dışladığı `D_PCT_30PLUS` bandı. Bu, advisor'ın AL gerekçesi ürettiği hattın kısıtlı evrende gerekçe üretmediği anlamına geliyor.

5. **Nyxpansion BIST-100/30'da seçicilik göstermiyor.** Walk-forward (aylık yeniden eğitim, 15g embargo) skorlarla üretilen günlük top-4 XU100'de +1.01 vs rastgele +1.11 — **rastgelenin altında**. BIST-30'da işlem ortalaması yüksek görünse de medyan rastgele medyanın altında; kuyruk etkisi.

6. **AL/SAT ve combo oylama kohortları negatif.** AL/SAT XU100'de −0.24 puan, işlem ortalaması rastgeleden −0.6 puan aşağıda. vote2 tekil kapılardan daha iyi değil. `vote3` (n=8 / n=3) yorumlanamaz.

7. **hw_obos'un kendi SPEC'indeki "no trading-edge claim" etiketi doğrulandı.** XU100'de AL_OS günlük kohortu 5g **−0.51** (rastgele +0.94), işlem **−0.73**, PF 0.86. Advisor'da bağlam sinyali olarak kalması doğru; AL gerekçesi olamaz.

8. **SBT-1700 endeks evreninde ölçüm gücünü kaybediyor.** XU100'de 205, XU030'da 53 aday; her ikisinde de rastgelenin altında. E04 sıralama katmanı bu örneklem büyüklüğünde kurulamıyor.

9. **RT_state teknik olarak p95'i geçiyor ama pratik değeri yok.** Kohort XU100 evren-günlerinin **%45'ini** kapsıyor (56 310 / 125 720) — bu bir tarama değil, "yükseliş rejimindeki hisseler" filtresi. Edge'i (+0.11 puan) buna uygun biçimde küçük. Taramanın "yeni sinyal" satırı olan RT_new rastgelenin altında.

10. **Sağ kalan iki hat aslında tek hat.** alpha ve tavan_v1'in ikisi de `MLScorer universe_up_1g` skorunu satıyor (eşikler 0.48 ve 0.65). Çıplak `ml_1g ≥ 0.65` kohortu her iki taramadan da iyi (+4.16, işlem +8.57, PF 2.93) ve skor desilleri monoton (D0 −0.16 → D9 +1.49). Ancak eşik **0.70'te kırılıyor** (n=87, +1.76, işlem medyanı −6.17) — 0.65 sağlam bir plato değil, uçurumun kenarı. Ayrıntı §6.

### Pratik okuma

BIST-100/BIST-30 ile sınırlı çalışılacaksa, briefing/advisor hattında **kanıtı olan tek şey `universe_up_1g` ML skorudur** — ve pratikte onu en iyi paketleyen liste alpha scan top-4'tür (yalnızca BIST-100'de; BIST-30'da değil). tavan koşulu bu paketlemeye değer katmıyor, çıkarılabilir. Eşik duyarlılığı (0.70'te kırılma) ayrıca çalışılmalı. Trident/DE v1 karar katmanı bu evrende düzeltilmeden kullanılmamalı; en azından G1'in D_pct üst sınırı (30) sorgulanmalı. Diğer taramalar bu evrende bağlam üretir, alım gerekçesi üretmez.

---

## 10. Veri durumu ve kısıtlar

**Veri güncellemesi (bu koşum).** Çalışma dizinindeki master parquet'ler `merge main: parquet=main` birleşmesinde 2026-04-24 sürümüne geri dönmüştü. `origin/main` daha yeni sürümü taşıyordu (#155 — extfeed 04-25→06-16 evren-deliği backfill); üç dosya oradan alındı, XU100 benchmark cache'i Fintables'tan uzatıldı:

| dosya | önce | sonra | kaynak |
|---|---|---|---|
| `ohlcv_10y_fintables_master.parquet` | 2026-04-24 | 2026-07-14 | `origin/main` |
| `extfeed_intraday_1h_3y_master.parquet` | 2026-04-24 | 2026-07-14 | `origin/main` |
| `nyxexp_dataset_v4.parquet` | 2026-04-24 | 2026-07-14 | `origin/main` |
| `xu100_cache.parquet` | 2026-04-24 | 2026-07-14 | Fintables `endeks_mumlar_gunluk_gh` (52 bar; örtüşen günlerde OHLC birebir doğrulandı, hacim kolonu ölçek farkı nedeniyle NaN) |

2026-07-14 sonrası (11 seans) hiçbir kaynakta yok: `origin/main` 07-28'e kadar aktif ama parquet'ler o tarihten beri güncellenmemiş. Yerelde `FINTABLES_MCP_TOKEN` ve `INTRADAY_*` kimlik bilgileri, `gh` CLI de yok — kanonik `master-data-pull` yolu çalıştırılamıyor.

**Diğer kısıtlar:**

- **Evren proxy'si.** §1'deki %80 / %87 örtüşme sonuçlara bir gürültü katmanı ekler. Yön ve sıralama `current` koşumlarıyla teyit edildi, ancak `current` survivorship-yanlıdır; aradaki belirsizlik giderilemez. Gerçek PIT endeks üyelik serisi (BIST gözden geçirme duyuruları) elde edilirse tüm koşumlar tek komutla yenilenir.
- **Rastgele işlem tabanı** 5 tekrarla hesaplandı (Katman A'daki 20 yerine), maliyet nedeniyle. Katman B farkları ±0.3 puan bandında yorumlanmalı.
- **Kohort büyüklüğü.** `vote3` (8/3), `AL_OS_1w` (18/4), `trident_geo` (85/21), `trident_tier1` (72/18), `tavan_v1`/XU030 (30), `RT_NOXw` (51/14), SBT/XU030 (53) örneklemleri yorum için yetersizdir; tabloda gösterilmelerinin nedeni eksiksizliktir.
- **2026-04-27 dikişi.** 17 ticker'da veri kaynağı süreksizliği var; alpha sim bu tickerları 04-27 sonrası evrenden çıkarıyor (782 süreksizlik barının tamamı işaretli, `discontinuity_bars.csv`).
- **Katman C** dolum/kapasite modellemez; bileşik getiriler beklenti değildir.
- **Yan düzeltmeler:** (a) `screener_combo/signals.py::_supertrend_dir` — `.to_numpy()` pandas copy-on-write altında salt-okunur dizi döndürüp koşumu kırıyordu, `copy=True` eklendi (değer değişmez). (b) `tools/trident_probe_mb_3y.py` — özet .md'yi Windows cp1254 ile yazmaya çalışıp U+2212'de patlıyordu, `encoding="utf-8"` eklendi.
