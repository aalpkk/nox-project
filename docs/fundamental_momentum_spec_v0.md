# Fundamental Momentum (Kâr İvmesi → Liderlik) v0 — Kilitli Pre-Spec

Tarih: 2026-06-13. KOD YAZILMADAN kilitlendi; parametreler sonuçlara göre OYNANMAZ.
Sektör-rotasyon hattının ([[sector-rotation-leadlag-line]]) yön-değiştiren
bulgusundan doğdu: endeks-üstü liderler (ASTOR/ASELS tipi) FİYAT/TEKNİK ile
bulunamıyor (~15 feature null/kontrarian, evren-momentum 51 olayda null), ama
TEMEL kâr ivmesi gerçekti (USD net kâr ASELS +%64, ASTOR +%29 / 2025).

## Hipotez
H: Bir hissenin (USD bazlı, point-in-time) kâr/ciro ivmesi, sonraki 1-3 ayın
EVREN-GENELİ β-düşülmüş alfasını (endeks-üstü getirisini) POZİTİF öngörür.
Birim sektör-içi DEĞİL, likit evren geneli (ASTOR/ASELS farklı sektörlerdeydi).

## Kırmızı prior / teorik zemin
Fundamental momentum + PEAD (post-earnings drift) literatürün en sağlam
anomalilerinden; fiyat-momentumun aksine kalıcı/öngörülebilir. AMA: (a) TR
yüksek-enflasyon → TL muhasebe çarpıtır, USD bazı ZORUNLU; (b) küçük örneklem +
tek-rejim (2018+) riski (bu hattın tekrarlayan FAIL kalıbı); (c) Q4/yıllık
denetim gecikmesi büyük → point-in-time şart, yoksa lookahead sahte-alfa üretir.

## Veri (Fintables, point-in-time)
- Finansallar: `hisse_finansal_tablolari` (yil, ay∈{3,6,9,12},
  `yayinlanma_tarihi_utc` = KAP yayın) + `..._gelir_tablosu_kalemleri`
  (kalem, `usd_donemsel`). Kapsam 2017-2026, 617 hisse.
- 🔴 POINT-IN-TIME KURALI: rebalans tarihi t'de SADECE
  `yayinlanma_tarihi_utc <= t` olan dönemler kullanılır. Period-end (yil,ay)
  DEĞİL yayın tarihi bağlayıcı. (Q4 ~10 hafta gecikmeli yayınlanır.)
- Fiyat/forward: `output/ohlcv_10y_fintables_master.parquet` (Close) +
  `output/sector_index_daily_master.parquet` (XU100 benchmark, β için).
- Kalemler: 'Dönem Karı (Zararı)' (net kâr), 'Brüt Kar (Zarar)', ciro
  ('Satışlar' boş dönerse 'Hasılat'/'Esas Faaliyet Gelirleri' fallback —
  build'de kalem-adı doğrulanır, tahmin edilmez).
- Çıktı: kalıcı panel `output/fundamental_pit_panel.parquet`
  (hisse, donem_yil, donem_ay, yayin_tarih, net_kar_usd, brut_kar_usd, ciro_usd).

## Evren (kademeli — user 2026-06-13)
- FAZ A: likit top-100 (60g ADV TL, her rebalansta yeniden). Temiz lab.
- FAZ B (yalnızca A geçerse): tüm-BIST genişleme; survivorship + likidite +
  split arızası taraması zorunlu (hisse master Close split-doğrulaması).

## Rebalans & ufuk
- Aylık (ay-sonu işlem günü), ≥2018-01.
- Forward hedef: β-düşülmüş alfa vs XU100, trailing-120g β ile.
  PRIMARY fwd = 60 işlem günü; ikincil 120g (PEAD çok-aylık sürer).

## Feature'lar (USD; PRIMARY önceden seçili = f1)
- **f1 (PRIMARY) YoY net-kâr büyümesi:** son yayınlanmış çeyreğin USD net kârı
  / 4-çeyrek-önceki aynı dönem − 1 (mevsimsellik nötr). NaN/negatif-taban
  guard: taban ≤0 ise sembol o rebalansta atlanır.
- f2 YoY ciro büyümesi (aynı kurgu).
- f3 Kâr İVMESİ (2. türev): f1(t) − f1(önceki çeyrek) — büyüme hızlanıyor mu.
- f4 Marj genişlemesi: brüt_kâr/ciro son çeyrek − 4-çeyrek-önce.
- f5 Trend-sürprizi: son çeyrek USD net kâr, kendi son-4-çeyrek log-trend
  ekstrapolasyonunun ne kadar üstünde (analist tahmini YOK → trend-proxy).

## Validasyon (per [[feedback_separability_metric_design]] + [[mtf_bull_fvg_strict_v4_outcome]])
1. **Kesitsel kalıcılık:** her rebalansta within-universe Spearman(feature,
   fwd_alfa); ortalama rho + bootstrap p + olayların %+'i.
2. **Q5−Q1 spread:** üst-quintile − alt-quintile forward β-alfa (eşli boot p).
3. **Separability:** GOOD (üst-quintile) vs FAILED (alt-quintile) AYRIMI null'a
   karşı — aynı tarihlerde RASTGELE feature ataması (1000 çekiliş) p95.
4. **Segment guard:** 2018-2021 / 2022-2026 AYRI; PASS için İKİ segmentte de
   yön tutmalı (bu hattın FAIL'lerinin hepsi tek-segmentti).
5. **β-temizlik (v6 dersi):** forward hedef ham göreli getiri DEĞİL β-düşülmüş
   alfa; ayrıca feature'ın β ile korelasyonu raporlanır (sinyal beta-proxy mi).
6. **Lookahead kontrolü:** aynı test period-end hizalamayla (yanlış) bir kez
   koşulur → publish-hizalamaya göre alfa DÜŞMELİ; düşmüyorsa sızıntı şüphesi.

## Karne
- PASS: f1 (veya önceden-seçili tek primary) kesitsel rho pozitif-anlamlı +
  Q5−Q1 pozitif + separability null'u geçer + İKİ segmentte de yön tutar +
  publish-vs-period-end alfa farkı doğru yönde (lookahead yok).
- PARTIAL: sinyal var ama tek segment / sadece 120g ufuk / Q5-Q1 var rho yok
  → paper-forward, canlı kapı kapalı.
- FAIL: rho ~0 veya beta-proxy veya lookahead'e bağımlı → fiyat eksenleri gibi
  kapat, dürüstçe raporla (5+ feature'a fishing yapma; primary f1 bağlayıcı).

## Açık riskler (baştan kayıtlı)
- USD dönüşümü Fintables'ta hazır (`usd_donemsel`) — kendi kur dönüşümü YOK.
- Tek-çeyrek gürültü (özellikle küçükler, Faz B) → yıllık-kayan (TTM) varyant
  ikincil olarak denenebilir ama primary çeyreklik-YoY.
- Banka/sigorta/leasing farklı şablon (gelir tablosu kalemleri farklı) →
  Faz A likit-100'de şablon-bazlı kalem eşleme doğrulanır; eşleşmeyen sembol
  atlanır (banka için 'Net Faiz Geliri'+'Net Dönem Karı' ayrı ele alınır).
- Maliyet/uygulama: bu bir SİNYAL araştırması; trade-maliyeti Faz A geçerse.
