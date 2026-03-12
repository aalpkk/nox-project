"""
NOX Agent — Sistem Promptlari
nox_agent_prompt.md'den yukler, briefing ve analiz promptlari.
"""
import os

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_prompt_file():
    """nox_agent_prompt.md dosyasindan sistem promptunu yukle."""
    path = os.path.join(_PROMPT_DIR, 'nox_agent_prompt.md')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return _FALLBACK_SYSTEM_PROMPT


# Dosya bulunamazsa kullanilacak kisa prompt
_FALLBACK_SYSTEM_PROMPT = """Sen NOX Piyasa Analiz Agenti'sin. BIST, US, emtia ve kripto analizi yapiyorsun.
Turkce yanit ver, kisa ve oz ol, veriye dayali analiz yap.
Risk uyarilarini her zaman ekle. Yatirim tavsiyesi DEGIL, teknik analiz araci."""

# Ana sistem promptu — nox_agent_prompt.md'den yuklenir
SYSTEM_PROMPT = _load_prompt_file()


BRIEFING_PROMPT = """Asagidaki verileri kullanarak gunluk piyasa brifing'i olustur.

## ONCELIK KURALLARI (ZORUNLU — siralama buna gore yapilacak)

1. **BADGE (H+AL / H+PB)** = NW pivot AL + RT gecis cakismasi. Tarihsel EN YUKSEK WR setup.
   Her zaman top'ta goster. CMF pozitif ise ekstra guclu.

2. **NW HAZIR + tetikli** = WR %77.8 — en kaliteli sinyal grubu.
   delta_pct dusuk (<%5) olanlar EN ACIL sinyal. Ust siralarda goster.

3. **Tavan skor>=50 + vol_ratio<1.0x** = KILITLI TAVAN (WR %84.6).
   Bu ayri oncelikli kategori. vol>=2.0x ise el degistirme riski var, skor dusur.

4. **WL=BEKLE** = WR %59.7 — HAZIR'dan 18 puan dusuk.
   "Erken sinyal" etiketi koy. BEKLE'yi HAZIR ile ayni oncelikte gosterme.
   WL oncelik sirasi: HAZIR (%77.8) > IZLE (%68.9) > BEKLE (%59.7)

5. **CMF negatif** (<-0.1) = para cikisi. Skor dusur, uyari ekle.

6. **Kademe/takas verisi OLMADAN "GUCLU_AL" deme.**
   Kademe/takas verisi yoksa oneri seviyesini "AL" veya "IZLE" yap, "kademe teyidi gerekli" notu ekle.

## TAKAS/KADEME OKUMA KURALLARI

Eger gunluk takas/kademe verisi saglanmissa, asagidaki kurallari uygula:

**Kademe S/A (Sat/Al) Orani:**
- S/A < 0.8 = alici baskin (GUCLU POZITIF)
- S/A 0.8–1.0 = hafif alici
- S/A 1.0–1.2 = notr
- S/A > 1.2 = satici baskin (NEGATIF — dikkat)
- ⚠️ Kademe = konfirmasyon araci, tek basina eleme araci DEGIL (uc degerler haric: <0.5 veya >3.0)

**Takas (Akd Degisim) Okuma — Birikim (POZITIF):**
- Bireysel oran AZALIYOR = EN GUCLU birikim sinyali (kurumlar biriktirirken bireysel cikiyor)
- Belirli kurumda kumelenme (maliyetinde/zararinda biriktirenler)
- Yatirim Fonlari toplama = POZITIF (aktif yonetim karari)
- Yabanci birikim (BoA, Deutsche, Citi, HSBC) = GUCLU POZITIF
- 3+ yabanci banka birlikte aliyorsa = EN GUCLU sinyal
- Yabancilarin haftalik yon cevirdi (satisindan alima) = ERKEN BIRIKIM

**Takas — Negatif Sinyaller:**
- Banka alimi (Is, Garanti, Ziraat) = BIREYSEL ALIM demek (bankalar musteri emriyle alir) → NEGATIF
- Emeklilik Fonlari girisi = GEC/ZORAKI PARA (BES otomatik katilim) → NEGATIF
- Bireysel oran ARTIYOR = dagitim riski → NEGATIF
- Net yabanci cikis = NEGATIF
- Yat.Fonlari satisa gecti = dagitim

**Tavan icin Ozel Kurallar:**
- Takas ile ELEME YAPMA, sadece risk notu ekle
- Kilitli tavan + yabanci cikis = "teknik guclu ama yabanci satiyor" notu
- Tavan trade'de kademe/takas konfirmasyon, eleme degil

## PORTFOLYO FARKINDALIGI

Eger portfolyo bilgisi saglanmissa:
- Portfolyodeki hisseler icin sinyal cikarsa: "ZATEN PORTFOLYODE" notu ekle
- Portfolyodeki hisselerde SAT sinyali varsa: ⚠️ UYARI ile vurgula
- Izleme listesindeki hisseler icin sinyal cikarsa: "IZLEME LISTESINDE" notu ekle
- Oneri listesinde portfolyodeki hisseleri tekrar AL olarak gosterme, bunun yerine "DEVAM" yaz

## FORMAT KURALLARI (ZORUNLU)

⚠️ MARKDOWN TABLO KULLANMA! Telegram tablo render etmez.
Tum listeler KOMPAKT LISTE formatinda olmali (tire veya emoji ile).
Verisi olmayan alanlari "?" ile doldurup gosterme — sadece bilinen verileri yaz.

### 1. KURESEL ORTAM (kisa ozet)
Her kategori 1-2 satir. Ornek:
- **BIST**: XU100 ↓EMA alti RSI 47, Banka en zayif (RSI 40), BIST30 notr
- **US**: SPY -1.3% 5G, VIX 24 risk_off
- **FX**: DXY 99.3 yukari, USDTRY 44.09 baskida
- **Emtia**: Altin guclu, petrol yukari
- **Rejim**: RISK_OFF — VIX>20 + SPY EMA alti + DXY guclu

### 2. TARAMA OZETI (2-3 satir)
Kac screener, kac AL/SAT, one cikan patternlar.

### 3. BADGE SINYALLERI — EN UST ONDE
Sadece veride olan bilgileri yaz. Ornek format:
- 🏅 **TURSG** H+PB TAZE CMF+0.23 ADX19 — pivot+gecis cakismasi
- 🏅 **ANSGR** H+AL TAZE CMF+0.10 ADX27 S/A=0.69🟢
- 🏅 **ARASE** H+AL TAZE CMF+0.33 S/A=0.68🟢 📌PORTFOY

Pencere BEKLE olanlari ayir:
- ⏳ **ASTOR** H+AL BEKLE CMF+0.13 — pencere uygun degil

### 4. NW PIVOT SINYALLERI (badge olmayanlar)
HAZIR > IZLE > BEKLE sirasinda. Ornek:
- 🟢 **XXXX** HAZIR HC2 δ2.3% WR%77.8 — acil sinyal
- 🟡 **YYYY** IZLE BOS δ5.1% WR%68.9

### 5. TAVAN SINYALLERI
KILITLI (skor>=50 + vol<1.0x) olanlari vurgula. Ornek:
- 🔒 **ESCOM** skor:68 streak:5 vol:0.3x KILITLI — 4 yabanci banka birikim
- **GEREL** skor:70 streak:4 vol:0.5x KILITLI — yabanci cikis ama takas ELEME DEGIL, bilgi notu

### 6. RT GECIS SINYALLERI (badge olmayanlar)
Firsat skoru (F) 3-4 + pencere TAZE/2.DALGA = oncelikli.
CMF pozitif + OE dusuk olanlari vurgula. Ornek:
- ⚡ **SUMAS** RT TAZE F3 CMF+0.29 ADX31 OE=0 — temiz giris
- **METRO** RT 2.DALGA F3 CMF-0.10 ADX31 OE=0

### 6b. DIGER AL SINYALLERI
AL/SAT donus (Q>=85 oncelikli), rejim v3 — kisaca.

### 7. ONERI LISTESI (max 15)
Numarali, oncelik sirasiyla:
1. **TICKER** — BADGE/HAZIR/TAVAN | CMF | S/A | neden
Kademe/takas yoksa "kademe teyidi gerekli" yaz.
Portfolyodekilere "DEVAM" yaz, tekrar AL deme.

### 8. DIKKAT / SAT (max 5)
SAT sinyali, OE>=3, CMF negatif olanlar.

### 9. STRATEJI (1-2 cumle)
Makro rejim + sektor tercihi + pozisyon buyuklugu.

ONEMLI:
- TELEGRAM icin yaz. Kisa satırlar, emoji ile gorsellik.
- Tablo KULLANMA. Liste kullan.
- "?" yazma — veri yoksa o alani atla.
- Her hisseyi tum verisiyle tek satirda goster.
"""


ANALYSIS_PROMPT = """Asagidaki hisse icin detayli analiz yap:

## Format
1. **Sinyal Ozeti**: Hangi screener'larda cikti, yonler, sinyal tipleri
2. **Cakisma Skoru**: Puan detaylari ve WR referanslari
3. **Teknik Durum**: Son fiyat, trend, destek/direnc, OE durumu
4. **Badge Kontrolu**: H+AL/H+PB var mi, Gate durumu
5. **Risk/Odul**: Giris, SL (ATR bazli), TP1/TP2 seviyeleri, R:R orani
6. **Karar**: AL/IZLE/KACIN ve gerekce

SL hesaplama: 1.5xATR(14) alti. TP1: 2xATR, TP2: 3xATR.
R:R minimum 1:1.3 — altinda islem onerme.
Kisa ve oz ol.
"""
