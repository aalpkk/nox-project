# Takas / AKD Okuma Sistemi — Playbook (oturumlar-arası devir)

Bu dosya, BIST hisselerinde **takas (settlement) ve AKD (Aracı Kurum Dağılımı)**
verisini "bir operatör topluyor mu, yoksa dağıtım/perakende mi" sorusuna cevap
verecek şekilde okuma sistemini özetler. Yeni bir oturum bunu okuyup aynı
çerçeveyle çalışabilir.

## 1. Ne yapıyoruz
Bir hissenin günlük/haftalık/aylık/3-aylık broker akışına bakıp:
- **Gerçek bir el mi topluyor** (konsantre, ev-dışı, kârdayken bile ekliyor) →
  markup/akümülasyon, fiyat şablonuyla birlikte alım tezi.
- Yoksa **dağıtım mı** (toplayıcı el satıcıya döndü, perakende emiyor) → çıkış/uzak dur.
Takas tek başına değil, **fiyat bağlamı + KAP** ile birlikte okunur.

## 2. Veri kaynağı: Matriks (tek broker-AKD kaynağı)
`agent/matriks_client.MatriksClient` (REST; `.env`'de `MATRIKS_API_KEY` +
`MATRIKS_CLIENT_ID`). Kullanılan metotlar:
- `get_institutional_flow_periods(sym)` → G/H/A/3A (günlük/haftalık/aylık/3-aylık
  net broker birikimi). **Sistemin ana girdisi.**
- `get_institutional_flow(sym, start_date, end_date)` → belirli günün akışı
  (topBuyers/topSellers, netFlow TL, dominantSide).
- `get_daily_flow_history(sym, days)` → ICE: günlük net akış (`net_tip.diger`) +
  `top3_alici_pct` (alıcı yoğunluğu).
- `get_settlement(sym, dates=[...])` → pozisyon AKD: kurum bazında **pozisyon %,
  MALİYET**, yoğunlaşma (top3 %, HHI). `flows_to_takas_data` / `flow_to_takas_history_day`
  adapter'ları (`agent/matriks_adapter`) ham yanıtı normalize eder.
- `get_market_price(sym)` → canlı fiyat (ama fiyatı biz EXTFEED'den alıyoruz, aşağıda).
- ⚠️ **Matriks sık 500/timeout verir** (`mcp.matriks.ai Read timed out` / "Management
  API service unavailable"). Bu auth değil servis sorunudur; yeni key çözmez. Retry
  döngüsü kurma — tek atış, başarısızsa atla. Tarihsel AKD ~2024'ten öncesine yok.
- 🔴 Matriks'in broker-AKD'sinin **birebir alternatifi YOK**: Fintables'ta takas/AKD
  becerisi yok (açığa satış var ama küçük/IPO isimler listede değil); İş Yatırım/MKK
  kamu verisi oturum-auth ister (401). Matriks down ise → KAP bildirimleri (insider)
  ile kısmi telafi (aşağı bak).

## 3. Yorum çerçevesi (load-bearing kurallar — hepsi sahada öğrenildi)
1. **key_buyer = ev-DIŞI ana toplayıcı** (3A en büyük pozitif), arz aracısı ve
   perakende hariç. "El-dönüşü" = bu elin net SATICIYA dönmesi.
2. **HOUSE etkisi (🔴 kritik):** bir IPO'da top-alıcı çoğu zaman **arzın aracı
   kurumudur** (TERA→ATATR/SVGYO, A1→EKDMR, Deniz→GENKM). Bu, bağımsız akıllı para
   DEĞİL — arz envanteri + müşteri ağı + içsel-bilgi sahiplerinin custody'si.
   `halka_arzlar.araci_kurum` (Fintables) ile join et; top-alıcı = arz aracısıysa
   **sinyal değerini düşür**. (SVGYO'da TERA +132M "dev toplayıcı" sanılmıştı, house çıktı.)
3. **Agrega, tek isim değil:** el-dönüşü alarmı key_buyer **GRUBUNUN toplam** günlük
   net'ine bakmalı. Tek bir toplayıcının offset'lenen düşük günü (Piramit −437K,
   3A +7.7M iken) yanlış alarm verir; aynı gün BofA+Investaz +5M alıyorsa grup net
   alımdadır. Şart: grup net satıcı VE >%10 kümülatifi bir günde geri-veriş.
4. **kurum ≠ sahip:** broker kodu gerçek sahip değil (omnibus). BofA çoğunlukla
   yabancı-akış proxy'si (ama IPO'da spekülatif de olabilir). **MIDAS top-alıcı =
   perakende dağınık = tez ALEYHİNE** (LXGYO/UCAYM). BULLS = tekrarlayan operatör
   imzası (GENKM'e yeni girdi, DITAS'ta markup yaptı, EDATA'da devraldı).
5. **Yoğunlaşma (HHI / top3%):** düşük = dağınık/kırılgan (operatör kilidi yok);
   yüksek = sıkı tutulan/operatör-kontrollü. Düşük yoğunlaşmada "operatör hikayesi"
   zayıftır, hareket perakende/momentum yakıtlı olabilir.
6. **Maliyet (settlement):** kütle nerede oturuyor? Herkes **başabaşta** = arz duvarı
   (her toparlamada "kurtulma" satışı; ESCOM 5.4-5.6). Kütle **derin kârda** (ATATR
   arz fiyatı ~11 maliyetle +%50-80) = motive satıcı overhang'i, kırılım zorlaşır.
   Kütle kârda AMA satmıyorsa = sağlıklı markup (GENKM erken).
7. **Fiyatla eşle:** "distribution-into-strength" = fiyat yükselirken ana toplayıcı
   satıyor = en tehlikeli (DMRGD/EKOS: BofA dökerken yerli emiyor). Tepe genelde
   hacim klimaksı + el-dönüşüyle çakışır; göstergeyle (RSI/MACD/mum) ÖNCEDEN
   yakalanamaz — operatör likiditeye satar, göstergeye değil.
8. **KAP çapraz teyit (Matriks'siz de çalışır):** ortak/yönetici **pay alım-satım
   bildirimi**, **%5 eşik** bildirimi (takvimli sinyal), **tedbir/VBTS** (momentum
   öldürür). Fintables `dokumanlarda_ara` (filter `iliskili_semboller IN ['X'] AND
   kap_bildirim_tipi='ODA'`). 🔴 **Sosyal medyadaki rakamı HER ZAMAN kaynaktan
   doğrula** — bir tweet %3.58'i %35.38 yapmıştı (ESCOM/Hasan Yalçın).

## 4. Araçlar (repo)
- `tools/takas_batch_run.py --tickers "A B C" --slug X --hist-days 6` — JENERİK
  broker akışı + ICE batch (CSV+ICE+MD çıktı `output/`).
- `tools/portfolio_takas_advisor.py` — günlük rapor: **fiyat extfeed 1h master son
  barından**, **takas Matriks'ten**, portföy `portfolio-advisor.yaml`'dan; el-dönüşü
  (agrega) + çıkış-serisi alarmı → Telegram.
- `tools/ipo_rbr_takas_watch.py` — aday izleme listesi (`data/ipo_rbr_watch.json`)
  el-dönüşü/çıkış alarmı → Telegram.
- `tools/ipo_rbr_eldonusu_backtest_v0.py` — el-dönüşü exit'in tarihsel testi
  (AKD cache `output/akd_cache/`).

## 5. Cron / otomasyon (GitHub Actions, main'den okur)
- `.github/workflows/portfolio-takas-advisor.yml` — hafta içi 18:45 TR; önce
  `restore-extfeed-master` action ile taze extfeed master indirir, sonra advisor'ı koşar.
- `.github/workflows/ipo-rbr-takas-watch.yml` — hafta içi 19:45 TR; izleme listesi.
- `.github/workflows/takas-adhoc.yml` — `workflow_dispatch` ile **serbest sembol**:
  `gh workflow run takas-adhoc.yml -f tickers="AKHAN KBORU DITAS" -f slug=x`.
- Config dosyaları: `portfolio-advisor.yaml` (gerçek pozisyonlar: ticker/cost/lot/
  key_buyers/house), `data/ipo_rbr_watch.json` (aday izleme).

## 6. Fiyat neden extfeed (Matriks değil)
Fiyat = `output/extfeed_intraday_1h_3y_master.parquet` son barı (kolonlar:
ticker, ts_istanbul, close). CI'da master-data-pull (17:00) tazeler;
`restore-extfeed-master` action en taze master'ı çeker. Çok yeni IPO'lar (ör.
EKDMR) extfeed evreninde olmayabilir → "extfeed'de yok" fallback, takas yine gösterilir.

## 7. Sözlük (verdict kelimeleri)
el-dönüşü · markup · distribution-into-strength · house illusion (arz-aracısı şişmesi) ·
break-even wall (başabaş arz duvarı) · overhang (kârda kütle baskısı) · ICE (net akış +
top3 yoğunluk).

## 8. Standart ihtiyatlar (her okumaya eklenir)
- Veri seans-sonu (gün içi anlık değil); kurum ≠ sahip; Matriks down olabilir;
  tarihsel AKD ~2024 öncesi yok; bu okuma exploratory/gözlem, separability testi
  yapılmadıysa kanıt değil; küçük/IPO isimlerde tek-gün hareketi gürültülü, sinyal
  agregada.

İlgili memory: `rbr_anchor_free_v0_outcome.md` (IPO RBR paterni + takas katmanı),
`ipo_lockup_expiry_line_groundwork.md` (lock-up takvimi).
