"""
NOX Agent — Smart Money Score (SMS)
5-pillar akıllı para analizi: birikim + yoğunlaşma + karşı taraf + süreklilik + MKK.

Puanlama: max 85p, negatif olabilir.
  S1: Birikim Sürekliliği (max 25p, min -5p)
  S2: Yoğunlaşma (max 20p)
  S3: Karşı Taraf Zayıflığı (max 15p, min -5p)
  S4: Çok Zamanlı Süreklilik Proxy (max 15p, min -8p)
  S5: MKK Rejim Teyidi (max 10p, min -5p)

Sınıflar:
  ≥45: 🟢GÜÇLÜ — SM uyumlu birikim
  30-44: 🟡ORTA — Kısmi birikim
  15-29: ⚪ZAYIF — Belirsiz
  <15: 🔴DAĞITIM — SM dağıtıyor
"""
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Kurum Sınıflandırma (SMS 5 kategori)
# ══════════════════════════════════════════════════════════════

_YAB_BANKA = [
    "citibank", "bank of america", "deutsche bank", "royal bank",
    "merrill lynch", "merrill", "hsbc", "jp morgan", "jpmorgan",
    "goldman sachs", "goldman", "ubs", "morgan stanley", "barclays",
    "credit suisse", "bnp paribas", "societe generale", "nomura",
    "clsa", "macquarie", "rbc", "wood &", "virtu", "citadel",
]

_FON = [
    "yatırım fonları", "yatirim fonlari",
    "emeklilik fonları", "emeklilik fonlari",
    "yatırım fonu", "yatirim fonu",
    "fon yönetimi", "fon yonetimi",
    "portföy yönetimi", "portfoy yonetimi",
    "asset management",
]

_PROP_KANAL = [
    "tera", "a1 capital", "a1 cap", "info yat", "info ya",
    "oyak", "marbas", "alnus", "bulls", "destek",
    "ikon", "piramit", "ünlü", "unlu",
]

_YERLI_BANKA = [
    "garanti", "iş yat", "is yat", "yapı kredi", "yapi kredi",
    "yapıkredi", "yapikredi", "qnb", "vakıf", "vakif",
    "deniz", "ziraat", "halk", "ak yat", "akbank",
    "teb", "şekerbank", "sekerbank",
]


def _tr_lower(s):
    """Türkçe uyumlu lowercase (İ→i, I→ı)."""
    return s.replace('İ', 'i').replace('I', 'ı').lower()


def classify_kurum_sms(name):
    """5 kategori: yab_banka / fon / prop / yerli_banka / diger.

    Çift kontrol: hem _tr_lower (Türkçe) hem ASCII lowercase.
    """
    n = _tr_lower(name)
    n_ascii = name.lower().replace('ı', 'i')

    def _match(keywords):
        return any(k in n for k in keywords) or any(k in n_ascii for k in keywords)

    if _match(_YAB_BANKA):
        return "yab_banka"
    if _match(_FON):
        return "fon"
    if _match(_PROP_KANAL):
        return "prop"
    if _match(_YERLI_BANKA):
        return "yerli_banka"
    return "diger"


# ══════════════════════════════════════════════════════════════
# Veri Modeli
# ══════════════════════════════════════════════════════════════

@dataclass
class SMSResult:
    """SMS skorlama sonucu."""
    ticker: str
    score: int
    s1: int = 0
    s2: int = 0
    s3: int = 0
    s4: int = 0
    s5: int = 0
    s1_detail: str = ""
    s2_detail: str = ""
    s3_detail: str = ""
    s4_detail: str = ""
    s5_detail: str = ""
    label: str = ""
    icon: str = ""

    def __post_init__(self):
        if self.score >= 45:
            self.label = "GÜÇLÜ"
            self.icon = "🟢"
        elif self.score >= 30:
            self.label = "ORTA"
            self.icon = "🟡"
        elif self.score >= 15:
            self.label = "ZAYIF"
            self.icon = "⚪"
        else:
            self.label = "DAĞITIM"
            self.icon = "🔴"


# ══════════════════════════════════════════════════════════════
# Yardımcı: Takas verisi normalizasyonu
# ══════════════════════════════════════════════════════════════

def _normalize_kurum(raw):
    """Takas JSON'daki kurum satırını normalize et.

    VDS export formatı:
      {"Aracı Kurum": "X", "Günlük Fark": N, "Haftalık Fark": N,
       "Aylık Fark": N, "3 Aylık Fark": N, "%": pay, "Pozisyon": lot}

    Returns: dict {name, tip, gunluk, haftalik, aylik, uc_aylik, pay, pozisyon}
    """
    name = raw.get("Aracı Kurum") or raw.get("kurum") or ""
    tip = classify_kurum_sms(name)
    return {
        "name": name,
        "tip": tip,
        "gunluk": raw.get("Günlük Fark") or raw.get("gunluk_fark") or 0,
        "haftalik": raw.get("Haftalık Fark") or raw.get("haftalik_fark") or 0,
        "aylik": raw.get("Aylık Fark") or raw.get("aylik_fark") or 0,
        "uc_aylik": raw.get("3 Aylık Fark") or raw.get("uc_aylik_fark") or 0,
        "pay": raw.get("%") or raw.get("pay") or 0,
        "pozisyon": raw.get("Pozisyon") or raw.get("takas_son") or 0,
    }


def _prepare_ticker_data(ticker_data):
    """Ticker'ın kurum verisini normalize et ve tip bazında grupla.

    S2/S3 haftalık veriye bakar (daha kararlı sinyal), S1 tüm periyotları kullanır.

    Args:
        ticker_data: {"kurumlar": [...]} (raw VDS format)

    Returns:
        dict: {
            "all": [normalized_list],
            "by_tip": {"yab_banka": [...], "fon": [...], ...},
            "net_tip": {"yab_banka": {"g": N, "h": N, "a": N, "3a": N}, ...},
            "alicilar": [...], "saticilar": [],
            "h_alicilar": [...], "h_saticilar": []  # haftalık bazlı
        }
    """
    kurumlar = ticker_data.get("kurumlar", [])
    normalized = [_normalize_kurum(k) for k in kurumlar]

    by_tip = {}
    for k in normalized:
        by_tip.setdefault(k["tip"], []).append(k)

    # Net akış: tip bazında günlük/haftalık/aylık toplam
    net_tip = {}
    for tip, klist in by_tip.items():
        net_tip[tip] = {
            "g": sum(k["gunluk"] for k in klist),
            "h": sum(k["haftalik"] for k in klist),
            "a": sum(k["aylik"] for k in klist),
            "3a": sum(k["uc_aylik"] for k in klist),
        }

    # Günlük farka göre alıcı/satıcı
    alicilar = sorted([k for k in normalized if k["gunluk"] > 0],
                      key=lambda x: -x["gunluk"])
    saticilar = sorted([k for k in normalized if k["gunluk"] < 0],
                       key=lambda x: x["gunluk"])

    # Haftalık farka göre alıcı/satıcı (S2/S3 için — daha kararlı)
    h_alicilar = sorted([k for k in normalized if k["haftalik"] > 0],
                        key=lambda x: -x["haftalik"])
    h_saticilar = sorted([k for k in normalized if k["haftalik"] < 0],
                         key=lambda x: x["haftalik"])

    return {
        "all": normalized,
        "by_tip": by_tip,
        "net_tip": net_tip,
        "alicilar": alicilar,
        "saticilar": saticilar,
        "h_alicilar": h_alicilar,
        "h_saticilar": h_saticilar,
    }


# ══════════════════════════════════════════════════════════════
# S1: Birikim Sürekliliği (max 25p, min -5p)
# ══════════════════════════════════════════════════════════════

def calc_s1_birikim(prepared, mkk_data=None, ticker=None):
    """Yabancı banka (max 15p) + Fon (max 10p) + Yerli banka cross-val (max 5p).

    Prop kanalları S1'de puan almaz (sinyal güvenilirliği düşük).
    Yerli banka: normalde puan almaz, AMA MKK kurumsal oranı artıyorsa
    ve yerli_banka kanalından yoğun alım varsa → kısmi puan (max 5p).
    Bu cross-validation ile TEB/İş üzerinden yapılan kurumsal birikimi yakalar.
    """
    net = prepared["net_tip"]
    details = []

    # Yabancı Banka (max 15p, min -5p)
    yb = net.get("yab_banka", {"g": 0, "h": 0, "a": 0, "3a": 0})
    yb_puan = 0
    # 3 Aylık
    if yb["3a"] > 0:
        yb_puan += 5
    elif yb["3a"] < 0:
        yb_puan -= 3
    # Aylık
    if yb["a"] > 0:
        yb_puan += 4
    elif yb["a"] < 0:
        yb_puan -= 3
    # Haftalık
    if yb["h"] > 0:
        yb_puan += 3
    elif yb["h"] < 0:
        yb_puan -= 2
    # Günlük
    if yb["g"] > 0:
        yb_puan += 3
    elif yb["g"] < 0:
        yb_puan -= 2

    yb_puan = max(-5, min(15, yb_puan))
    details.append(f"YB:{yb_puan} (G:{yb['g']:+,} H:{yb['h']:+,} A:{yb['a']:+,} 3A:{yb['3a']:+,})")

    # Fon (max 10p, min -4p)
    fon = net.get("fon", {"g": 0, "h": 0, "a": 0, "3a": 0})
    fon_puan = 0
    if fon["3a"] > 0:
        fon_puan += 3
    elif fon["3a"] < 0:
        fon_puan -= 2
    if fon["a"] > 0:
        fon_puan += 3
    elif fon["a"] < 0:
        fon_puan -= 2
    if fon["h"] > 0:
        fon_puan += 2
    elif fon["h"] < 0:
        fon_puan -= 1
    if fon["g"] > 0:
        fon_puan += 2
    elif fon["g"] < 0:
        fon_puan -= 1

    fon_puan = max(-4, min(10, fon_puan))
    details.append(f"Fon:{fon_puan} (G:{fon['g']:+,} H:{fon['h']:+,} A:{fon['a']:+,} 3A:{fon['3a']:+,})")

    # Yerli Banka cross-validation (max 5p)
    # MKK kurumsal oranı artıyorsa + yerli_banka alıyorsa → kısmi puan
    # Mantık: TEB/İş/Garanti üzerinden kurumsal birikim yapılıyor olabilir
    yerli_puan = 0
    yerli = net.get("yerli_banka", {"g": 0, "h": 0, "a": 0, "3a": 0})
    mkk = None
    if mkk_data and ticker:
        mkk = mkk_data.get(ticker) or mkk_data.get(f"{ticker}.IS")

    if mkk:
        # bireysel_fark_5g < 0 = bireysel azalma = kurumsal artışı
        kurumsallasma = -(mkk.get("bireysel_fark_5g", 0) or 0)
        if kurumsallasma > 1.0:  # haftalık %1+ kurumsal artış
            # Yerli banka haftalık/aylık alıyorsa → cross-validated birikim
            if yerli["a"] > 0 and yerli["h"] > 0:
                yerli_puan = 5
            elif yerli["a"] > 0 or yerli["h"] > 0:
                yerli_puan = 3
            elif yerli["g"] > 0:
                yerli_puan = 1
        elif kurumsallasma > 0.5:  # haftalık %0.5-1 kurumsal artış
            if yerli["a"] > 0 and yerli["h"] > 0:
                yerli_puan = 3
            elif yerli["a"] > 0 or yerli["h"] > 0:
                yerli_puan = 2

    if yerli_puan > 0:
        details.append(f"Yerli✓:{yerli_puan} (MKK K+{kurumsallasma:.1f}pp "
                        f"H:{yerli['h']:+,} A:{yerli['a']:+,})")

    total = max(-5, min(25, yb_puan + fon_puan + yerli_puan))
    return total, " | ".join(details)


# ══════════════════════════════════════════════════════════════
# S2: Yoğunlaşma (max 20p)
# ══════════════════════════════════════════════════════════════

def calc_s2_yogunlasma(prepared):
    """Alış konsantrasyonu + SM in top 3 + satış dağınıklığı + prop convergence.

    Haftalık veriye bakar (daha kararlı sinyal).
    """
    details = []
    puan = 0

    # Haftalık bazlı alıcı/satıcı (daha kararlı sinyal)
    alicilar = prepared["h_alicilar"]
    saticilar = prepared["h_saticilar"]

    # Toplam haftalık alış
    toplam_alis = sum(k["haftalik"] for k in alicilar) if alicilar else 0

    # 1. Alış konsantrasyonu (8p): Top 3 alıcı / toplam alış
    if toplam_alis > 0 and len(alicilar) >= 3:
        top3_alis = sum(k["haftalik"] for k in alicilar[:3])
        top3_pct = (top3_alis / toplam_alis) * 100
    elif toplam_alis > 0:
        top3_alis = sum(k["haftalik"] for k in alicilar)
        top3_pct = (top3_alis / toplam_alis) * 100
    else:
        top3_pct = 0

    if top3_pct > 70:
        puan += 8
        details.append(f"Top3:%{top3_pct:.0f}→8p")
    elif top3_pct > 50:
        puan += 5
        details.append(f"Top3:%{top3_pct:.0f}→5p")
    elif top3_pct > 30:
        puan += 2
        details.append(f"Top3:%{top3_pct:.0f}→2p")

    # 2. SM in top 3 (6p): yab_banka → +4, fon → +2
    sm_bonus = 0
    top3 = alicilar[:3]
    for k in top3:
        if k["tip"] == "yab_banka":
            sm_bonus += 4
        elif k["tip"] == "fon":
            sm_bonus += 2
    sm_bonus = min(6, sm_bonus)
    if sm_bonus > 0:
        puan += sm_bonus
        details.append(f"SM_top3:{sm_bonus}p")

    # 3. Satış dağınıklığı (6p)
    n_alici = len(alicilar)
    n_satici = len(saticilar)
    if n_satici > 5 and n_alici <= 3:
        puan += 6
        details.append(f"Dağınık satış ({n_satici}S/{n_alici}A)→6p")
    elif n_satici > n_alici:
        puan += 3
        details.append(f"Satıcı>Alıcı ({n_satici}S/{n_alici}A)→3p")

    # 4. Prop convergence bonus (+3p): 2+ farklı prop kanalı haftalık alıcıda
    prop_alici = [k for k in alicilar if k["tip"] == "prop"]
    if len(prop_alici) >= 2:
        puan += 3
        prop_names = [k["name"][:8] for k in prop_alici[:3]]
        details.append(f"Prop conv:{'+'.join(prop_names)}→3p")

    puan = min(20, puan)
    return puan, " | ".join(details)


# ══════════════════════════════════════════════════════════════
# S3: Karşı Taraf Zayıflığı (max 15p, min -5p)
# ══════════════════════════════════════════════════════════════

def calc_s3_karsi_taraf(prepared):
    """Yerli/diğer satıcı oranı + YB satış penalty.

    Haftalık veriye bakar (tek günlük satış = gürültü).
    """
    details = []
    puan = 5  # Base: karşı kanıt yoksa

    # Haftalık bazlı satıcılar (daha kararlı sinyal)
    saticilar = prepared["h_saticilar"]
    net = prepared["net_tip"]

    # Toplam haftalık satış (mutlak lot)
    toplam_satis = sum(abs(k["haftalik"]) for k in saticilar) if saticilar else 0

    if toplam_satis > 0:
        # Yerli/diğer/prop satıcı oranı — "zayıf el" satışı
        # Prop kanalları trader flow → SM değil, zayıf el sayılır
        # Gerçek SM = sadece yab_banka + fon
        yerli_satis = sum(abs(k["haftalik"]) for k in saticilar
                         if k["tip"] in ("yerli_banka", "diger", "prop"))
        yerli_pct = (yerli_satis / toplam_satis) * 100

        if yerli_pct > 70:
            puan += 10
            details.append(f"Yerli satıcı %{yerli_pct:.0f}→10p")
        elif yerli_pct > 50:
            puan += 6
            details.append(f"Yerli satıcı %{yerli_pct:.0f}→6p")
        elif yerli_pct > 30:
            puan += 3
            details.append(f"Yerli satıcı %{yerli_pct:.0f}→3p")
    else:
        details.append("Satış yok→base 5p")

    # YB haftalık satıyor ve lot>10K → -5p penalty
    # Günlük değil haftalık bakıyoruz — tek günlük satış birikim bozmuyor
    yb_h = net.get("yab_banka", {}).get("h", 0)
    if yb_h < -10_000:
        puan -= 5
        details.append(f"YB H.satış {yb_h:+,}→-5p")

    puan = max(-5, min(15, puan))
    return puan, " | ".join(details)


# ══════════════════════════════════════════════════════════════
# S4: Çok Zamanlı Süreklilik Proxy (max 15p, min -8p)
# ══════════════════════════════════════════════════════════════

def calc_s4_sureklilik(prepared):
    """Periyotlar arası yön tutarlılığı proxy'si.

    Gerçek taşıma kalitesi günlük history olmadan ölçülemez.
    Mevcut yaklaşım: H→A, A→3A yön tutarlılığı.
    """
    details = []
    puan = 0
    net = prepared["net_tip"]

    # YB: H→A aynı yön (+) → +5, A→3A aynı yön (+) → +5
    yb = net.get("yab_banka", {"g": 0, "h": 0, "a": 0, "3a": 0})
    if yb["h"] > 0 and yb["a"] > 0:
        puan += 5
        details.append("YB H→A↑→5p")
    elif yb["h"] < 0 and yb["a"] < 0:
        puan -= 5
        details.append("YB H→A↓→-5p")

    if yb["a"] > 0 and yb["3a"] > 0:
        puan += 5
        details.append("YB A→3A↑→5p")
    elif yb["a"] < 0 and yb["3a"] < 0:
        puan -= 5
        details.append("YB A→3A↓→-5p")

    # Fon: H→A aynı yön (+) → +5
    fon = net.get("fon", {"g": 0, "h": 0, "a": 0, "3a": 0})
    if fon["h"] > 0 and fon["a"] > 0:
        puan += 5
        details.append("Fon H→A↑→5p")
    elif fon["h"] < 0 and fon["a"] < 0:
        puan -= 3
        details.append("Fon H→A↓→-3p")

    # Prop: H→A aynı yön (+) → +2 (düşük güvenilirlik ama destek)
    prop = net.get("prop", {"g": 0, "h": 0, "a": 0, "3a": 0})
    if prop["h"] > 0 and prop["a"] > 0:
        puan += 2
        details.append("Prop H→A↑→2p")

    puan = max(-8, min(15, puan))
    return puan, " | ".join(details)


# ══════════════════════════════════════════════════════════════
# S5: MKK Rejim Teyidi (max 10p, min -5p)
# ══════════════════════════════════════════════════════════════

def calc_s5_mkk(ticker, mkk_data):
    """MKK yatırımcı dağılımı teyidi.

    Args:
        ticker: Hisse kodu
        mkk_data: {TICKER: {bireysel_pct, kurumsal_pct, bireysel_fark_1g, bireysel_fark_5g}}
    """
    if not mkk_data:
        return 0, "MKK veri yok"

    mkk = mkk_data.get(ticker) or mkk_data.get(f"{ticker}.IS")
    if not mkk:
        return 0, f"{ticker} MKK yok"

    details = []
    puan = 0

    kurumsal_pct = mkk.get("kurumsal_pct", 0) or 0
    bireysel_pct = mkk.get("bireysel_pct", 100) or 100

    # 3 aylık kurumsal oran değişimi — haftalık fark'tan çıkarım (yaklaşık)
    # Gerçek 3 aylık fark MKK history'den gelecek, şimdilik proxy:
    # bireysel_fark_5g * ~12 ≈ 3 aylık (kaba tahmin), ama mevcut veriden doğrudan bakarız
    bireysel_fark_5g = mkk.get("bireysel_fark_5g", 0) or 0
    bireysel_fark_1g = mkk.get("bireysel_fark_1g", 0) or 0

    # 3 aylık proxy: haftalık fark * 12 yaklaşımı yerine,
    # mevcut kurumsal seviyeyi kullanarak puan ver
    # Gerçek 3 aylık değişim verisi yoksa sadece haftalık fark kullan
    haftalik_kurumsallasma = -bireysel_fark_5g  # bireysel azalma = kurumsallaşma

    if haftalik_kurumsallasma > 3:
        puan += 7
        details.append(f"K.artış +{haftalik_kurumsallasma:.1f}pp/hafta→7p")
    elif haftalik_kurumsallasma > 1:
        puan += 4
        details.append(f"K.artış +{haftalik_kurumsallasma:.1f}pp/hafta→4p")
    elif haftalik_kurumsallasma < -3:
        puan -= 5
        details.append(f"K.düşüş {haftalik_kurumsallasma:.1f}pp/hafta→-5p")
    elif haftalik_kurumsallasma < -1:
        puan -= 2
        details.append(f"K.düşüş {haftalik_kurumsallasma:.1f}pp/hafta→-2p")

    # Mutlak seviye
    if kurumsal_pct > 65:
        puan += 3
        details.append(f"K=%{kurumsal_pct:.0f}→3p")
    elif kurumsal_pct > 50:
        puan += 1
        details.append(f"K=%{kurumsal_pct:.0f}→1p")
    elif kurumsal_pct < 30:
        puan -= 2
        details.append(f"K=%{kurumsal_pct:.0f}→-2p")

    puan = max(-5, min(10, puan))
    return puan, " | ".join(details) if details else f"K=%{kurumsal_pct:.0f}"


# ══════════════════════════════════════════════════════════════
# Ana Fonksiyon: Smart Money Score
# ══════════════════════════════════════════════════════════════

def calc_smart_money_score(ticker, takas_data, mkk_data=None):
    """Tek hisse SMS skoru hesapla.

    Args:
        ticker: Hisse kodu (ör: "GARAN")
        takas_data: dict {TICKER: {kurumlar: [...]}} (VDS takas_data.json format)
        mkk_data: dict {TICKER: {bireysel_pct, kurumsal_pct, ...}} (opsiyonel)

    Returns:
        SMSResult veya None (veri yoksa)
    """
    td = takas_data.get(ticker) or takas_data.get(f"{ticker}.IS")
    if not td:
        return None

    prepared = _prepare_ticker_data(td)

    s1, s1_d = calc_s1_birikim(prepared, mkk_data=mkk_data, ticker=ticker)
    s2, s2_d = calc_s2_yogunlasma(prepared)
    s3, s3_d = calc_s3_karsi_taraf(prepared)
    s4, s4_d = calc_s4_sureklilik(prepared)
    s5, s5_d = calc_s5_mkk(ticker, mkk_data)

    total = s1 + s2 + s3 + s4 + s5

    return SMSResult(
        ticker=ticker,
        score=total,
        s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
        s1_detail=s1_d, s2_detail=s2_d, s3_detail=s3_d,
        s4_detail=s4_d, s5_detail=s5_d,
    )


def calc_batch_sms(tickers, takas_data, mkk_data=None):
    """Toplu SMS hesapla.

    Args:
        tickers: Hisse kodu listesi veya None (tüm takas verisi)
        takas_data: dict {TICKER: {kurumlar: [...]}}
        mkk_data: dict {TICKER: {...}} (opsiyonel)

    Returns:
        dict[str, SMSResult]: {ticker: SMSResult}
    """
    if tickers is None:
        tickers = list(takas_data.keys())

    results = {}
    for ticker in tickers:
        sms = calc_smart_money_score(ticker, takas_data, mkk_data)
        if sms:
            results[ticker] = sms
    return results


def format_sms_line(sms):
    """Tek satır formatla: "GARAN 53🟢 S1:11 S2:15 S3:15 S4:5 S5:7"

    Args:
        sms: SMSResult

    Returns:
        str
    """
    return (f"{sms.ticker} {sms.score}{sms.icon} "
            f"S1:{sms.s1} S2:{sms.s2} S3:{sms.s3} S4:{sms.s4} S5:{sms.s5}")


def format_sms_detail(sms):
    """Detaylı çok satırlı format.

    Args:
        sms: SMSResult

    Returns:
        str
    """
    lines = [
        f"{sms.icon} {sms.ticker}: SMS={sms.score} ({sms.label})",
        f"  S1 Birikim [{sms.s1}]: {sms.s1_detail}",
        f"  S2 Yoğunlaşma [{sms.s2}]: {sms.s2_detail}",
        f"  S3 Karşı Taraf [{sms.s3}]: {sms.s3_detail}",
        f"  S4 Süreklilik [{sms.s4}]: {sms.s4_detail}",
        f"  S5 MKK [{sms.s5}]: {sms.s5_detail}",
    ]
    return "\n".join(lines)


def sms_icon(score):
    """Skor'dan ikon döndür."""
    if score >= 45:
        return "🟢"
    elif score >= 30:
        return "🟡"
    elif score >= 15:
        return "⚪"
    else:
        return "🔴"
