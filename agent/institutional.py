"""
NOX Agent — Kurumsal Teyit Motoru (ICE)
3 soru + 4 etiket + confirmation multiplier.

SM = yab_banka + fon + prop (3 kanal birlikte)
SM_genis = SM + yerli_banka (false negative azaltmak için destekleyici seri)

Etiket ayrımı:
  1. kurumsal_teyit = kısa/orta vade YÖN teyidi (DT20, cont20, dt_genis destek)
  2. tasinan_birikim = uzun vade KALICILIK (DT60, cont60, top3_pct, flow_efficiency)
  3. kisa_vade = son 5 gün davranışı (DT5, recent_strength)
  4. maliyet_avantaji = SM avg cost vs fiyat (Matriks settlement verisi)

Çarpan: 0.65 - 1.20 (scanner skoruna multiplier)

Veri kaynakları:
  - Matriks institutionalFlow — günlük kurumsal akış (snapshot)
  - Matriks settlementAnalysis — broker pozisyonları + maliyet (maliyet_avantaji)
  - Takas history (70g sliding window) — opsiyonel, varsa ICE güçlenir
  - MKK — bireysel/kurumsal %, fark 1g/5g (GitHub Pages)
  - Kademe — S/A bid/ask (Faz 3, henüz yok)

SMS v1 backward compat: score/label/icon alanları mevcut erişim kalıplarıyla uyumlu.
"""
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Eşik Konfigürasyonu (kalibrasyon için tek nokta)
# ══════════════════════════════════════════════════════════════

ICE_THRESHOLDS = {
    # kurumsal_teyit
    "teyit_cont20_strong": 0.55,    # var eşiği
    # tasinan_birikim
    "birikim_cont60_strong": 0.55,  # guclu eşiği
    "birikim_cont60_mid": 0.40,     # orta eşiği
    "birikim_efficiency_strong": 0.40,  # flow_efficiency guclu
    "birikim_efficiency_mid": 0.20,    # flow_efficiency orta
    # kisa_vade
    "kv_strength_positive": 0.3,    # recent_strength destekliyor eşiği
    "kv_strength_negative": -0.3,   # recent_strength dagitim_riski eşiği
    # geniş seri destek
    "genis_destek_ratio": 0.5,      # dt_genis > dt_sm * ratio → destek
}


# ══════════════════════════════════════════════════════════════
# Veri Modeli
# ══════════════════════════════════════════════════════════════

@dataclass
class ICELabel:
    """Tek etiket."""
    key: str        # "kurumsal_teyit" | "tasinan_birikim" | "maliyet_avantaji" | "kisa_vade"
    value: str      # "var"/"zayif"/"yok" vb.
    detail: str     # açıklama


@dataclass
class ICEResult:
    """ICE sonucu — 4 etiket + multiplier + SMS uyumlu alanlar."""
    ticker: str
    labels: dict                     # {key: ICELabel}
    multiplier: float                # 0.65 - 1.20
    metrics: dict                    # ham metrikler
    status: str = "ok"              # ok / partial / no_history / no_data
    warnings: list = field(default_factory=list)

    # SMS backward compat — confluence/briefing bunlara erişiyor:
    score: int = 0                   # sms_compat_score (kaba dönüşüm)
    score_100: int = 0               # 0-100 normalize (detaylı)
    label: str = ""                  # "GÜÇLÜ"/"ORTA"/"ZAYIF"/"DAĞITIM"
    icon: str = ""                   # 🟢🟡⚪🔴
    detail_lines: list = field(default_factory=list)

    def __post_init__(self):
        self.score, self.label, self.icon = _multiplier_to_sms_compat(self.multiplier)
        self.score_100 = _multiplier_to_score_100(self.multiplier)
        self._build_detail_lines()

    def _build_detail_lines(self):
        self.detail_lines = []
        for key in ("kurumsal_teyit", "tasinan_birikim", "kisa_vade", "maliyet_avantaji"):
            lbl = self.labels.get(key)
            if lbl:
                self.detail_lines.append(f"{key}: {lbl.value} — {lbl.detail}")
        # Ana metrik özeti
        dt20 = self.metrics.get("takas_20_change")
        dt60 = self.metrics.get("takas_60_change")
        eff = self.metrics.get("flow_efficiency_20")
        if dt20 is not None:
            parts = [f"ΔT20={dt20:+,}"]
            if dt60 is not None:
                parts.append(f"ΔT60={dt60:+,}")
            if eff is not None:
                parts.append(f"eff={eff:.0%}")
            self.detail_lines.append("metrik: " + " ".join(parts))


# ══════════════════════════════════════════════════════════════
# Takas History Metrikleri
# ══════════════════════════════════════════════════════════════

def _get_daily_sm_flows(history, ticker, window):
    """Son N günün günlük SM akışlarını liste olarak döndür.

    SM = yab_banka + fon + prop

    Returns:
        list[int]: günlük SM net akışları (en yeniden en eskiye)
    """
    if not history:
        return []
    dates = sorted(history.keys(), reverse=True)[:window]
    flows = []
    for d in dates:
        day_data = history.get(d, {})
        ticker_data = day_data.get(ticker, {})
        net_tip = ticker_data.get("net_tip", {})
        sm = (net_tip.get("yab_banka", 0) +
              net_tip.get("fon", 0) +
              net_tip.get("prop", 0))
        flows.append(sm)
    return flows


def _calc_takas_window_change(history, ticker, window):
    """Son N günün SM toplamı → ΔT lot."""
    flows = _get_daily_sm_flows(history, ticker, window)
    return sum(flows)


def _calc_takas_window_change_genis(history, ticker, window):
    """Geniş kurumsal seri: SM + yerli_banka.

    False negative azaltmak için destekleyici kontrol.
    Bazı hisselerde yerli banka prop flow veya kurumsal müşteri emirleri taşır.
    """
    if not history:
        return 0
    dates = sorted(history.keys(), reverse=True)[:window]
    total = 0
    for d in dates:
        day_data = history.get(d, {})
        ticker_data = day_data.get(ticker, {})
        net_tip = ticker_data.get("net_tip", {})
        total += (net_tip.get("yab_banka", 0) +
                  net_tip.get("fon", 0) +
                  net_tip.get("prop", 0) +
                  net_tip.get("yerli_banka", 0))
    return total


def _calc_recent_strength(dt5, dt20):
    """Güvenli ivme: DT5'in DT20'ye göre oransal gücü.

    recent_strength = DT5 / (|DT20| + epsilon)
    > 0.3: son 5 gün güçlü pozitif (hızlanma)
    ~0: paralel
    < -0.3: son 5 gün ters yönde (yavaşlama/dönüş)

    Eski acceleration (DT5/5)/(DT20/20) formülüne göre avantajları:
    - DT20~0 olduğunda şişmez
    - İşaret değişimlerinde anlamlı sonuç verir
    """
    eps = max(abs(dt20) * 0.01, 100)  # minimum epsilon = 100 lot
    return dt5 / (abs(dt20) + eps)


def _calc_continuity(history, ticker, window=20):
    """SM net>0 olan gün sayısı / N. 0.0-1.0."""
    flows = _get_daily_sm_flows(history, ticker, window)
    if not flows:
        return 0.0
    positive = sum(1 for f in flows if f > 0)
    return positive / len(flows)


def _calc_flow_efficiency(history, ticker, window=20):
    """Akış verimliliği: net / toplam mutlak.

    efficiency = DT_window / sum(|daily_flow|)
    Zigzag'ı yakalar: yüksek net ama düşük tutarlılık → düşük efficiency.
    Range: -1.0 (tam negatif) ~ +1.0 (tam pozitif)
    """
    flows = _get_daily_sm_flows(history, ticker, window)
    if not flows:
        return 0.0
    total_abs = sum(abs(f) for f in flows)
    if total_abs < 100:
        return 0.0
    net = sum(flows)
    return net / total_abs


def _calc_top3_trend(history, ticker, window=20):
    """Top3 alıcı yoğunlaşma trendi (son window gün ortalaması).

    Takas history'de her günün top3_alici_pct değeri var.
    Yüksek yoğunlaşma = koordineli birikim.
    """
    if not history:
        return None
    dates = sorted(history.keys(), reverse=True)[:window]
    pcts = []
    for d in dates:
        day_data = history.get(d, {})
        ticker_data = day_data.get(ticker, {})
        pct = ticker_data.get("top3_alici_pct")
        if pct is not None:
            pcts.append(pct)
    if not pcts:
        return None
    return sum(pcts) / len(pcts)


def _calc_takas_metrics(history, ticker):
    """Takas history'den tüm metrikleri hesapla.

    Returns:
        dict veya None
    """
    if not history:
        return None

    available_days = sum(1 for d in history if ticker in history[d])
    if available_days < 3:
        return None

    dt5 = _calc_takas_window_change(history, ticker, 5)
    dt20 = _calc_takas_window_change(history, ticker, 20)
    dt60 = _calc_takas_window_change(history, ticker, 60)
    dt20_genis = _calc_takas_window_change_genis(history, ticker, 20)
    recent_str = _calc_recent_strength(dt5, dt20)
    cont20 = _calc_continuity(history, ticker, 20)
    cont60 = _calc_continuity(history, ticker, 60)
    eff20 = _calc_flow_efficiency(history, ticker, 20)
    eff60 = _calc_flow_efficiency(history, ticker, 60)
    top3_avg = _calc_top3_trend(history, ticker, 20)

    return {
        "takas_5_change": dt5,
        "takas_20_change": dt20,
        "takas_60_change": dt60,
        "takas_20_genis": dt20_genis,
        "recent_strength": round(recent_str, 3),
        "continuity_20": round(cont20, 2),
        "continuity_60": round(cont60, 2),
        "flow_efficiency_20": round(eff20, 3),
        "flow_efficiency_60": round(eff60, 3),
        "top3_alici_avg": round(top3_avg, 1) if top3_avg is not None else None,
        "available_days": available_days,
    }


# ══════════════════════════════════════════════════════════════
# Snapshot Metrikleri (mevcut takas_data'dan)
# ══════════════════════════════════════════════════════════════

def _calc_snapshot_metrics(takas_snapshot, ticker):
    """Günlük takas snapshot'tan metrikler."""
    if not takas_snapshot:
        return None

    from agent.smart_money import _prepare_ticker_data

    td = takas_snapshot.get(ticker) or takas_snapshot.get(f"{ticker}.IS")
    if not td:
        return None

    prepared = _prepare_ticker_data(td)
    net = prepared["net_tip"]

    yb = net.get("yab_banka", {"g": 0, "h": 0, "a": 0, "3a": 0})
    fon = net.get("fon", {"g": 0, "h": 0, "a": 0, "3a": 0})
    prop = net.get("prop", {"g": 0, "h": 0, "a": 0, "3a": 0})
    yerli = net.get("yerli_banka", {"g": 0, "h": 0, "a": 0, "3a": 0})

    return {
        "yb_gunluk": yb.get("g", 0),
        "yb_haftalik": yb.get("h", 0),
        "yb_aylik": yb.get("a", 0),
        "yb_3aylik": yb.get("3a", 0),
        "fon_gunluk": fon.get("g", 0),
        "fon_haftalik": fon.get("h", 0),
        "fon_aylik": fon.get("a", 0),
        "fon_3aylik": fon.get("3a", 0),
        "prop_gunluk": prop.get("g", 0),
        "prop_haftalik": prop.get("h", 0),
        "prop_aylik": prop.get("a", 0),
        "prop_3aylik": prop.get("3a", 0),
        "yerli_haftalik": yerli.get("h", 0),
        "yerli_aylik": yerli.get("a", 0),
    }


# ══════════════════════════════════════════════════════════════
# Aggregate → Takas Metrikleri (daily history gereksiz)
# ══════════════════════════════════════════════════════════════

def _calc_aggregate_takas_metrics(snapshot_metrics):
    """G/H/A/3A aggregate'lerden ICE metrikleri türet.

    Daily history olmadan çalışır — Matriks 4 periyot flow yeterli.
    DT5 ≈ SM haftalık, DT20 ≈ SM aylık, DT60 ≈ SM 3 aylık.

    Continuity proxy: kaç periyotta (G/H/A/3A) SM pozitif.
    Flow efficiency proxy: kısa/uzun vade uyumu.
    """
    if not snapshot_metrics:
        return None

    # SM = yab_banka + fon + prop (her periyot)
    sm_g = (snapshot_metrics.get("yb_gunluk", 0) +
            snapshot_metrics.get("fon_gunluk", 0) +
            snapshot_metrics.get("prop_gunluk", 0))
    sm_h = (snapshot_metrics.get("yb_haftalik", 0) +
            snapshot_metrics.get("fon_haftalik", 0) +
            snapshot_metrics.get("prop_haftalik", 0))
    sm_a = (snapshot_metrics.get("yb_aylik", 0) +
            snapshot_metrics.get("fon_aylik", 0) +
            snapshot_metrics.get("prop_aylik", 0))
    sm_3a = (snapshot_metrics.get("yb_3aylik", 0) +
             snapshot_metrics.get("fon_3aylik", 0) +
             snapshot_metrics.get("prop_3aylik", 0))

    # Geniş seri (yerli_banka dahil) — 20g proxy
    yerli_a = snapshot_metrics.get("yerli_aylik", 0)
    dt20_genis = sm_a + yerli_a

    # Continuity proxy: 4 periyottan kaçı pozitif (0.00 - 1.00)
    periods = [sm_g, sm_h, sm_a, sm_3a]
    positive_count = sum(1 for p in periods if p > 0)
    cont_proxy = positive_count / 4.0

    # Flow efficiency proxy: net / toplam mutlak
    total_abs = abs(sm_g) + abs(sm_h) + abs(sm_a) + abs(sm_3a)
    net = sm_g + sm_h + sm_a + sm_3a
    eff_proxy = (net / total_abs) if total_abs > 100 else 0.0

    # Recent strength: G'nin H'ye göre gücü
    eps = max(abs(sm_h) * 0.01, 100)
    recent_str = sm_g / (abs(sm_h) + eps)

    return {
        "takas_5_change": sm_h,       # H ≈ DT5
        "takas_20_change": sm_a,      # A ≈ DT20
        "takas_60_change": sm_3a,     # 3A ≈ DT60
        "takas_20_genis": dt20_genis,
        "recent_strength": round(recent_str, 3),
        "continuity_20": round(cont_proxy, 2),   # proxy
        "continuity_60": round(cont_proxy, 2),   # aynı proxy (4 nokta)
        "flow_efficiency_20": round(eff_proxy, 3),
        "flow_efficiency_60": round(eff_proxy, 3),
        "top3_alici_avg": None,       # aggregate'den hesaplanamaz
        "available_days": 60,         # aggregate = 60 gün kapsıyor
        "_is_aggregate": True,        # daily history değil, aggregate'ten türetildi
    }


# ══════════════════════════════════════════════════════════════
# MKK Metrikleri
# ══════════════════════════════════════════════════════════════

def _calc_mkk_metrics(mkk_data, ticker):
    """MKK verisinden metrikler."""
    if not mkk_data:
        return None

    mkk = mkk_data.get(ticker) or mkk_data.get(f"{ticker}.IS")
    if not mkk:
        return None

    return {
        "kurumsal_pct": mkk.get("kurumsal_pct", 0) or 0,
        "bireysel_pct": mkk.get("bireysel_pct", 100) or 100,
        "bireysel_fark_1g": mkk.get("bireysel_fark_1g", 0) or 0,
        "bireysel_fark_5g": mkk.get("bireysel_fark_5g", 0) or 0,
    }


# ══════════════════════════════════════════════════════════════
# Etiket Üreteçleri
# ══════════════════════════════════════════════════════════════

def _label_kurumsal_teyit(takas_metrics, snapshot_metrics=None):
    """Kurumsal teyit: kısa/orta vade YÖN teyidi.

    Birincil: DT20, continuity_20
    Destekleyici: dt_genis (yerli_banka dahil) — false negative azaltır

    var: DT20>0 + cont20 >= teyit_cont20_strong
    zayif: DT20>0 ama cont düşük, VEYA dt_sm<=0 ama dt_genis pozitif
    yok: her iki seri de negatif
    """
    TH = ICE_THRESHOLDS

    if not takas_metrics:
        if snapshot_metrics:
            sm_h = (snapshot_metrics.get("yb_haftalik", 0) +
                    snapshot_metrics.get("fon_haftalik", 0) +
                    snapshot_metrics.get("prop_haftalik", 0))
            sm_a = (snapshot_metrics.get("yb_aylik", 0) +
                    snapshot_metrics.get("fon_aylik", 0) +
                    snapshot_metrics.get("prop_aylik", 0))
            sm_g = (snapshot_metrics.get("yb_gunluk", 0) +
                    snapshot_metrics.get("fon_gunluk", 0) +
                    snapshot_metrics.get("prop_gunluk", 0))
            if sm_h > 0 and sm_a > 0:
                return ICELabel("kurumsal_teyit", "zayif",
                                f"snap H={sm_h:+,} A={sm_a:+,} (history yok)")
            elif sm_h > 0:
                return ICELabel("kurumsal_teyit", "zayif",
                                f"snap H={sm_h:+,} (history yok)")
            # Matriks fallback: sadece günlük veri varsa (H/A=0)
            elif sm_h == 0 and sm_a == 0 and sm_g > 0:
                return ICELabel("kurumsal_teyit", "zayif",
                                f"snap G={sm_g:+,} (sadece günlük)")
            elif sm_g > 0:
                return ICELabel("kurumsal_teyit", "zayif",
                                f"snap G={sm_g:+,} H={sm_h:+,} (history yok)")
            return ICELabel("kurumsal_teyit", "yok",
                            "snap SM negatif (history yok)")
        return ICELabel("kurumsal_teyit", "yok", "veri yok")

    dt20 = takas_metrics["takas_20_change"]
    dt20_g = takas_metrics["takas_20_genis"]
    cont20 = takas_metrics["continuity_20"]

    if dt20 > 0 and cont20 >= TH["teyit_cont20_strong"]:
        return ICELabel("kurumsal_teyit", "var",
                        f"ΔT20={dt20:+,} cont={cont20:.0%}")

    if dt20 > 0:
        return ICELabel("kurumsal_teyit", "zayif",
                        f"ΔT20={dt20:+,} cont={cont20:.0%} (düşük süreklilik)")

    # SM negatif ama geniş seri pozitif → yerli banka taşıyor olabilir
    if dt20 <= 0 and dt20_g > 0:
        return ICELabel("kurumsal_teyit", "zayif",
                        f"ΔT20={dt20:+,} ΔT20geniş={dt20_g:+,} (yerli destek)")

    return ICELabel("kurumsal_teyit", "yok",
                    f"ΔT20={dt20:+,} cont={cont20:.0%}")


def _label_tasinan_birikim(takas_metrics, snapshot_metrics=None):
    """Taşınan birikim: uzun vade KALICILIK.

    Birincil: DT60, continuity_60, flow_efficiency_60
    Destekleyici: top3_alici_avg (koordineli birikim tespiti)

    guclu: DT60>0 + cont60 yüksek + efficiency yüksek
    orta: DT60>0 + cont60 veya efficiency orta
    suphe: diğer
    """
    TH = ICE_THRESHOLDS

    if not takas_metrics:
        if snapshot_metrics:
            sm_3a = (snapshot_metrics.get("yb_3aylik", 0) +
                     snapshot_metrics.get("fon_3aylik", 0) +
                     snapshot_metrics.get("prop_3aylik", 0))
            sm_a = (snapshot_metrics.get("yb_aylik", 0) +
                    snapshot_metrics.get("fon_aylik", 0) +
                    snapshot_metrics.get("prop_aylik", 0))
            if sm_3a > 0 and sm_a > 0:
                return ICELabel("tasinan_birikim", "orta",
                                f"snap 3A={sm_3a:+,} A={sm_a:+,} (history yok)")
            return ICELabel("tasinan_birikim", "suphe",
                            "snap yetersiz (history yok)")
        return ICELabel("tasinan_birikim", "suphe", "veri yok")

    dt60 = takas_metrics["takas_60_change"]
    cont60 = takas_metrics["continuity_60"]
    eff60 = takas_metrics["flow_efficiency_60"]
    top3 = takas_metrics.get("top3_alici_avg")

    top3_tag = f" top3={top3:.0f}%" if top3 is not None else ""

    if dt60 > 0 and cont60 >= TH["birikim_cont60_strong"] and eff60 >= TH["birikim_efficiency_strong"]:
        return ICELabel("tasinan_birikim", "guclu",
                        f"ΔT60={dt60:+,} cont60={cont60:.0%} eff={eff60:.0%}{top3_tag}")

    if dt60 > 0 and (cont60 >= TH["birikim_cont60_mid"] or eff60 >= TH["birikim_efficiency_mid"]):
        return ICELabel("tasinan_birikim", "orta",
                        f"ΔT60={dt60:+,} cont60={cont60:.0%} eff={eff60:.0%}{top3_tag}")

    return ICELabel("tasinan_birikim", "suphe",
                    f"ΔT60={dt60:+,} cont60={cont60:.0%} eff={eff60:.0%}{top3_tag}")


def _label_maliyet_avantaji(cost_data=None):
    """Maliyet avantajı — Matriks settlement verisinden SM avg cost vs fiyat.

    cost_data: matriks_adapter.calc_cost_advantage() çıktısı
        {value, detail, streak_days, momentum, position_change_pct}
    """
    if not cost_data or cost_data.get("value") == "veri_yok":
        # Trend bilgisi varsa detail'e ekle
        streak = cost_data.get("streak_days", 0) if cost_data else 0
        if streak > 0:
            mom = cost_data.get("momentum", "")
            return ICELabel("maliyet_avantaji", "veri_yok",
                            f"maliyet yok — SM streak={streak}g {mom}")
        return ICELabel("maliyet_avantaji", "veri_yok", "maliyet verisi yok")
    return ICELabel("maliyet_avantaji", cost_data["value"], cost_data["detail"])


def _label_kisa_vade(takas_metrics, snapshot_metrics=None, mkk_metrics=None):
    """Kısa vadeli davranış: son 5 günde SM ne yapıyor?

    Birincil: DT5, recent_strength (güvenli ivme formülü)

    destekliyor: DT5>0 + recent_strength > eşik
    notr: karışık
    dagitim_riski: DT5<0 + recent_strength < eşik
    """
    TH = ICE_THRESHOLDS

    if not takas_metrics:
        if snapshot_metrics:
            sm_g = (snapshot_metrics.get("yb_gunluk", 0) +
                    snapshot_metrics.get("fon_gunluk", 0) +
                    snapshot_metrics.get("prop_gunluk", 0))
            if sm_g > 0:
                return ICELabel("kisa_vade", "notr",
                                f"snap G={sm_g:+,} (history yok)")
            return ICELabel("kisa_vade", "notr",
                            "snap yetersiz (history yok)")
        return ICELabel("kisa_vade", "notr", "veri yok")

    dt5 = takas_metrics["takas_5_change"]
    rs = takas_metrics["recent_strength"]

    # MKK kısa vadeli kurumsallaşma bonus
    mkk_tag = ""
    if mkk_metrics:
        fark_1g = mkk_metrics.get("bireysel_fark_1g", 0)
        if fark_1g < -0.5:
            mkk_tag = f" MKK({fark_1g:+.1f}%)"

    if dt5 > 0 and rs >= TH["kv_strength_positive"]:
        return ICELabel("kisa_vade", "destekliyor",
                        f"ΔT5={dt5:+,} rs={rs:+.2f}{mkk_tag}")
    elif dt5 < 0 and rs <= TH["kv_strength_negative"]:
        return ICELabel("kisa_vade", "dagitim_riski",
                        f"ΔT5={dt5:+,} rs={rs:+.2f}{mkk_tag}")
    else:
        return ICELabel("kisa_vade", "notr",
                        f"ΔT5={dt5:+,} rs={rs:+.2f}{mkk_tag}")


# ══════════════════════════════════════════════════════════════
# Çarpan + SMS Uyumluluk
# ══════════════════════════════════════════════════════════════

def _calc_multiplier(labels, cost_data=None):
    """4 etiketten multiplier hesapla.

    guclu (1.20): teyit=var + birikim=guclu + kv=destekliyor
    iyi (1.10): teyit=var + birikim>=orta + kv!=dagitim_riski
    orta (1.05): teyit=var/zayif + birikim>=orta
    notr (1.00): karışık / veri yetersiz
    zayif (0.85): teyit=yok + birikim=suphe
    red_flag (0.65): teyit=yok + kv=dagitim_riski

    + maliyet_avantaji ayarlaması (±0.05 max)
    + SM ardışık birikim streak bonus (+0.02 per 3 gün, max +0.04)
    """
    teyit = labels.get("kurumsal_teyit")
    birikim = labels.get("tasinan_birikim")
    kv = labels.get("kisa_vade")

    t = teyit.value if teyit else "yok"
    b = birikim.value if birikim else "suphe"
    k = kv.value if kv else "notr"

    # Base multiplier belirleme
    if t == "var" and b == "guclu" and k == "destekliyor":
        base_mult = 1.20
    elif t == "var" and b in ("guclu", "orta") and k != "dagitim_riski":
        base_mult = 1.10
    elif t in ("var", "zayif") and b in ("guclu", "orta"):
        base_mult = 1.05
    elif t == "var":
        base_mult = 1.00
    elif t == "zayif":
        base_mult = 0.85 if k == "dagitim_riski" else 0.95
    elif k == "dagitim_riski":
        base_mult = 0.65
    elif b == "suphe":
        base_mult = 0.85
    else:
        base_mult = 0.90

    # Maliyet avantajı ayarlaması (tüm seviyelere uygulanır)
    ma = labels.get("maliyet_avantaji")
    if ma and ma.value not in ("veri_yok", "notr"):
        _MA_ADJUST = {"guclu": +0.05, "avantaj": +0.03, "risk": -0.03, "yuksek_risk": -0.05}
        adj = _MA_ADJUST.get(ma.value, 0)
        base_mult = max(0.65, min(1.20, base_mult + adj))

    # SM ardışık birikim streak bonusu
    # ≥3 gün → +0.02, ≥6 gün → +0.04 (max)
    if cost_data:
        streak = cost_data.get("streak_days", 0)
        if streak >= 6:
            base_mult = min(1.20, base_mult + 0.04)
        elif streak >= 3:
            base_mult = min(1.20, base_mult + 0.02)

    return base_mult


def _multiplier_to_sms_compat(mult):
    """Multiplier → SMS uyumlu kaba skor (eski sistem uyumu).

    ≥1.15 → (55, 'GÜÇLÜ', '🟢')
    ≥1.02 → (35, 'ORTA', '🟡')
    ≥0.90 → (20, 'ZAYIF', '⚪')
    else  → (5, 'DAĞITIM', '🔴')
    """
    if mult >= 1.15:
        return 55, "GÜÇLÜ", "🟢"
    elif mult >= 1.02:
        return 35, "ORTA", "🟡"
    elif mult >= 0.90:
        return 20, "ZAYIF", "⚪"
    else:
        return 5, "DAĞITIM", "🔴"


def _multiplier_to_score_100(mult):
    """Multiplier → 0-100 normalize skor (detay kaybetmez).

    0.65 → 0, 1.00 → 64, 1.20 → 100
    Linear interpolation.
    """
    clamped = max(0.65, min(1.20, mult))
    return round((clamped - 0.65) / (1.20 - 0.65) * 100)


# ══════════════════════════════════════════════════════════════
# Ana Fonksiyonlar
# ══════════════════════════════════════════════════════════════

def calc_ice(ticker, takas_history, takas_snapshot=None,
             mkk_data=None, kademe_data=None, cost_data=None):
    """Tek hisse ICE hesapla.

    Args:
        ticker: Hisse kodu
        takas_history: {date: {ticker: {net_tip: {...}, ...}}} (70g window)
        takas_snapshot: {ticker: {kurumlar: [...]}} (günlük VDS/Matriks format)
        mkk_data: {ticker: {bireysel_pct, kurumsal_pct, ...}}
        kademe_data: Faz 3 — şimdilik None
        cost_data: matriks_adapter.calc_cost_advantage() çıktısı (maliyet avantajı)

    Returns:
        ICEResult veya None
    """
    warnings = []

    # Metrikleri hesapla
    takas_m = _calc_takas_metrics(takas_history, ticker)
    snap_m = _calc_snapshot_metrics(takas_snapshot, ticker)
    mkk_m = _calc_mkk_metrics(mkk_data, ticker)

    # Daily history yoksa → aggregate'lerden türet (G/H/A/3A → DT5/20/60)
    if not takas_m and snap_m:
        takas_m = _calc_aggregate_takas_metrics(snap_m)

    # En az bir veri kaynağı olmalı
    has_cost = cost_data and cost_data.get("value") != "veri_yok"
    if not takas_m and not snap_m and not has_cost:
        return None

    # Status belirleme
    if takas_m:
        is_agg = takas_m.get("_is_aggregate", False)
        days = takas_m.get("available_days", 0)
        if is_agg:
            status = "aggregate"
            warnings.append("AGGREGATE: G/H/A/3A'dan türetildi (daily history yok)")
        elif days < 10:
            status = "partial"
            warnings.append(f"INSUFFICIENT_WINDOW: {days} gün (min 10 önerilir)")
        else:
            status = "ok"
    elif snap_m:
        status = "no_history"
        warnings.append("NO_HISTORY: sadece snapshot verisi")
    else:
        status = "cost_only"
        warnings.append("COST_ONLY: sadece maliyet verisi")

    if not snap_m:
        warnings.append("NO_SNAPSHOT: günlük takas verisi yok")
    if not mkk_m:
        warnings.append("NO_MKK: MKK verisi yok")

    # Etiketleri üret — takas_m artık aggregate olabilir, etiket fonksiyonları aynı çalışır
    labels = {
        "kurumsal_teyit": _label_kurumsal_teyit(takas_m, snap_m),
        "tasinan_birikim": _label_tasinan_birikim(takas_m, snap_m),
        "maliyet_avantaji": _label_maliyet_avantaji(cost_data),
        "kisa_vade": _label_kisa_vade(takas_m, snap_m, mkk_m),
    }

    # Multiplier
    mult = _calc_multiplier(labels, cost_data=cost_data)

    # Metrikleri birleştir
    metrics = {}
    if takas_m:
        metrics.update(takas_m)
    if snap_m:
        metrics.update({f"snap_{k}": v for k, v in snap_m.items()})
    if mkk_m:
        metrics.update({f"mkk_{k}": v for k, v in mkk_m.items()})
    if cost_data:
        if cost_data.get("cost_ratio"):
            metrics["cost_ratio"] = cost_data["cost_ratio"]
        if cost_data.get("streak_days"):
            metrics["streak_days"] = cost_data["streak_days"]
            metrics["streak_momentum"] = cost_data.get("momentum", "")
        if cost_data.get("position_change_pct") is not None:
            metrics["position_change_pct"] = cost_data["position_change_pct"]

    return ICEResult(
        ticker=ticker,
        labels=labels,
        multiplier=mult,
        metrics=metrics,
        status=status,
        warnings=warnings,
    )


def calc_batch_ice(tickers, takas_history, takas_snapshot=None,
                   mkk_data=None, kademe_data=None, cost_data_map=None):
    """Toplu ICE hesapla (event-driven: sadece verilen ticker'lar).

    Args:
        cost_data_map: {ticker: {value, detail, ...}} — maliyet avantajı verisi

    Returns:
        dict[str, ICEResult]
    """
    results = {}
    for ticker in tickers:
        cd = cost_data_map.get(ticker) if cost_data_map else None
        ice = calc_ice(ticker, takas_history, takas_snapshot,
                       mkk_data, kademe_data, cost_data=cd)
        if ice:
            results[ticker] = ice
    return results


# ══════════════════════════════════════════════════════════════
# SMS v1 Uyumluluk Wrapper'ları
# ══════════════════════════════════════════════════════════════

def format_sms_line(ice):
    """Tek satır format: "GARAN ×1.15🟢 T=var B=guc KV=dest" """
    t = ice.labels.get("kurumsal_teyit")
    b = ice.labels.get("tasinan_birikim")
    kv = ice.labels.get("kisa_vade")
    t_short = t.value[:3] if t else "?"
    b_short = b.value[:3] if b else "?"
    kv_short = kv.value[:4] if kv else "?"
    return f"{ice.ticker} ×{ice.multiplier:.2f}{ice.icon} T={t_short} B={b_short} KV={kv_short}"


def format_sms_detail(ice):
    """Detaylı çok satırlı format."""
    lines = [f"{ice.icon} {ice.ticker}: ICE ×{ice.multiplier:.2f} ({ice.label}) s100={ice.score_100}"]
    for dl in ice.detail_lines:
        lines.append(f"  {dl}")
    if ice.warnings:
        lines.append(f"  ⚠️ {'; '.join(ice.warnings)}")
    return "\n".join(lines)


def sms_icon(score):
    """Skor'dan ikon döndür (SMS uyumlu)."""
    if score >= 45:
        return "🟢"
    elif score >= 30:
        return "🟡"
    elif score >= 15:
        return "⚪"
    else:
        return "🔴"
