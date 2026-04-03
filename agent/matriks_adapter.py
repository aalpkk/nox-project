"""
NOX Agent — Matriks Response → SMS/ICE Format Dönüştürücüler

Matriks MCP API yanıtlarını mevcut SMS ve ICE motorlarının
beklediği veri formatlarına çevirir.

4 periyot çağrısı (G/H/A/3A):
  - daily (1g) → Günlük Fark
  - weekly (5g) → Haftalık Fark
  - monthly (20g) → Aylık Fark
  - quarterly (60g) → 3 Aylık Fark

Maliyet avantajı: settlement text parse → SM avg cost vs price
"""
import re
from typing import Optional

from agent.smart_money import classify_kurum_sms


# ══════════════════════════════════════════════════════════════
# Yardımcı: flow response → broker net quantity map
# ══════════════════════════════════════════════════════════════

def _extract_broker_quantities(flow_response: dict) -> dict:
    """institutionalFlow → {broker_name: {quantity, position, netPercent}}.

    topBuyers → pozitif, topSellers → negatif quantity.
    """
    brokers = {}
    for agent in flow_response.get("topBuyers", []):
        name = agent.get("name", "")
        if not name:
            continue
        brokers[name] = {
            "quantity": agent.get("quantity", 0),
            "position": agent.get("position", 0),
            "netPercent": agent.get("netPercent", 0),
        }
    for agent in flow_response.get("topSellers", []):
        name = agent.get("name", "")
        if not name or name in brokers:
            continue
        brokers[name] = {
            "quantity": -abs(agent.get("quantity", 0)),
            "position": agent.get("position", 0),
            "netPercent": agent.get("netPercent", 0),
        }
    return brokers


# ══════════════════════════════════════════════════════════════
# A. 4 periyot flows → takas_data (SMS format)
# ══════════════════════════════════════════════════════════════

def flows_to_takas_data(flows: dict, symbol: str) -> dict:
    """4 periyot Matriks flow → SMS takas_data formatı.

    Args:
        flows: {daily: flow_resp, weekly: flow_resp, monthly: flow_resp, quarterly: flow_resp}
        symbol: Hisse kodu

    Çıktı: {TICKER: {kurumlar: [{Aracı Kurum, Günlük Fark, Haftalık Fark, ...}]}}
    """
    # Her periyot için broker→quantity map
    daily_map = _extract_broker_quantities(flows.get("daily", {}))
    weekly_map = _extract_broker_quantities(flows.get("weekly", {}))
    monthly_map = _extract_broker_quantities(flows.get("monthly", {}))
    quarterly_map = _extract_broker_quantities(flows.get("quarterly", {}))

    # Tüm broker isimlerini birleştir (union)
    all_names = set()
    all_names.update(daily_map.keys())
    all_names.update(weekly_map.keys())
    all_names.update(monthly_map.keys())
    all_names.update(quarterly_map.keys())

    kurumlar = []
    for name in all_names:
        d = daily_map.get(name, {})
        w = weekly_map.get(name, {})
        m = monthly_map.get(name, {})
        q = quarterly_map.get(name, {})

        kurumlar.append({
            "Aracı Kurum": name,
            "Günlük Fark": d.get("quantity", 0),
            "Haftalık Fark": w.get("quantity", 0),
            "Aylık Fark": m.get("quantity", 0),
            "3 Aylık Fark": q.get("quantity", 0),
            "%": d.get("netPercent", 0) or w.get("netPercent", 0),
            "Pozisyon": d.get("position", 0) or w.get("position", 0),
        })

    # Günlük farka göre sırala (büyük alıcılar önce)
    kurumlar.sort(key=lambda x: -x["Günlük Fark"])

    return {symbol: {"kurumlar": kurumlar}}


def flow_to_takas_data(flow_response: dict, symbol: str) -> dict:
    """Tek periyot flow → takas_data (backward compat, H/A/3A=0)."""
    return flows_to_takas_data({"daily": flow_response}, symbol)


# ══════════════════════════════════════════════════════════════
# B. institutionalFlow → takas_history_day (ICE format)
# ══════════════════════════════════════════════════════════════

def flow_to_takas_history_day(flow_response: dict, symbol: str, date_str: str) -> dict:
    """Matriks institutionalFlow → ICE takas_history tek gün formatı.

    Çıktı: {date_str: {TICKER: {net_tip: {yab_banka: N, ...}, top3_alici_pct: float}}}
    """
    # Tüm broker'ları sınıflandır ve net akışı tip bazında grupla
    tip_net = {}
    all_buyers = []

    for agent in flow_response.get("topBuyers", []):
        name = agent.get("name", "")
        tip = classify_kurum_sms(name)
        qty = agent.get("quantity", 0)
        tip_net[tip] = tip_net.get(tip, 0) + qty
        all_buyers.append({"name": name, "quantity": qty})

    for agent in flow_response.get("topSellers", []):
        name = agent.get("name", "")
        tip = classify_kurum_sms(name)
        qty = -abs(agent.get("quantity", 0))
        tip_net[tip] = tip_net.get(tip, 0) + qty

    # Top3 alıcı yüzdesi
    top3_pct = 0.0
    total_buy_qty = sum(a["quantity"] for a in all_buyers)
    if total_buy_qty > 0 and len(all_buyers) >= 3:
        top3_qty = sum(a["quantity"] for a in sorted(all_buyers, key=lambda x: -x["quantity"])[:3])
        top3_pct = (top3_qty / total_buy_qty) * 100

    return {
        date_str: {
            symbol: {
                "net_tip": tip_net,
                "top3_alici_pct": round(top3_pct, 1),
            }
        }
    }


# ══════════════════════════════════════════════════════════════
# C. Settlement Text Parse
# ══════════════════════════════════════════════════════════════

def parse_settlement_text(analysis_text: str) -> list:
    """Matriks settlement analiz metnini parse et.

    Metin formatı:
        N. CODE (ID: X)
           Pozisyon: Xmn lot (Y%)
           [Net Değişim: ...]
           Maliyet: Z TL
           Hacim: W TL

    Returns:
        list[dict]: [{code, name, position_lot, position_pct, net_change_lot, cost}]
    """
    if not analysis_text:
        return []

    brokers = []
    # Her broker bloğunu yakala
    pattern = re.compile(
        r'(\d+)\.\s+(\S+)\s+\(ID:\s*(\d+)\)\s*\n'
        r'\s+Pozisyon:\s+([\d.,]+)(mn|K)?\s+lot\s+\(([\d.,]+)%\)'
        r'(?:\s*\n\s+Net Değişim:\s*([^\n]+))?'
        r'(?:\s*\n\s*\([^)]+\))?'
        r'\s*\n\s+Maliyet:\s+([\d.,]+)\s+TL',
        re.MULTILINE
    )

    for m in pattern.finditer(analysis_text):
        rank = int(m.group(1))
        code = m.group(2)
        broker_id = m.group(3)

        pos_val = float(m.group(4).replace(",", "."))
        pos_unit = m.group(5) or ""
        if pos_unit == "mn":
            position_lot = int(pos_val * 1_000_000)
        elif pos_unit == "K":
            position_lot = int(pos_val * 1_000)
        else:
            position_lot = int(pos_val)

        position_pct = float(m.group(6).replace(",", "."))
        cost = float(m.group(8).replace(",", "."))

        brokers.append({
            "rank": rank,
            "code": code,
            "broker_id": broker_id,
            "position_lot": position_lot,
            "position_pct": position_pct,
            "cost": cost,
        })

    return brokers


# Settlement broker kodlarını isim eşleştirme tablosu
# institutionalFlow'dan gelen name alanları ile zenginleştirilir
_SETTLEMENT_CODE_TO_NAME = {
    "YATFON": "YATIRIM FONLARI",
    "EMKFON": "EMEKLILIK FONLARI",
    "CIY": "CITIBANK",
    "IYM": "IS YATIRIM",
    "GRM": "GARANTI BBVA",
    "MLB": "BANK-OF-AMERICA YATIRIM BANK",
    "AKM": "AK YATIRIM",
    "YKR": "YAPI KREDI YAT.",
    "DBY": "DEUTSCHE BANK",
    "TEB": "TEB YATIRIM",
    "VKY": "VAKIF YAT.",
    "HLY": "HALK YATIRIM",
    "ALM": "PUSULA YATIRIM",
    "IYF": "INFO YATIRIM MENKUL",
    "GSM": "GOLDMAN SACHS",
    "ICT": "ICBC TURKEY YATIRIM MENKUL",
    "HSB": "HSBC YATIRIM",
    "JPM": "JP MORGAN",
    "UBS": "UBS",
    "MRL": "MERRILL LYNCH",
    "NOM": "NOMURA",
    "MOR": "MORGAN STANLEY",
    "BAR": "BARCLAYS",
    "BNP": "BNP PARIBAS",
    "MAC": "MACQUARIE",
    "WOD": "WOOD & CO",
    "VIR": "VIRTU",
    "CTD": "CITADEL",
    "TRA": "TERA YATIRIM",
    "DES": "DESTEK MENKUL",
    "OYK": "OYAK YATIRIM",
    "QNB": "QNB FINANS YATIRIM",
    "ZRY": "ZIRAAT YATIRIM",
    "DNZ": "DENIZ YATIRIM",
    "UNL": "ÜNLÜ MENKUL",
}


def _classify_settlement_broker(code: str, flow_brokers: dict = None) -> str:
    """Settlement broker kodunu SM sınıfına çevir.

    Önce flow_brokers'tan isim bul, yoksa _SETTLEMENT_CODE_TO_NAME tablosu kullan.
    Tireli isimleri normalize eder (BANK-OF-AMERICA → BANK OF AMERICA).
    """
    # institutionalFlow'dan gelen code→name eşleşmesi
    if flow_brokers and code in flow_brokers:
        name = flow_brokers[code].replace("-", " ")
        return classify_kurum_sms(name)

    name = _SETTLEMENT_CODE_TO_NAME.get(code, "")
    if name:
        return classify_kurum_sms(name.replace("-", " "))

    return "diger"


# ══════════════════════════════════════════════════════════════
# D. Maliyet Avantajı Hesaplama
# ══════════════════════════════════════════════════════════════

def calc_cost_advantage(settlement_brokers: list, current_price: float,
                        flow_response: dict = None) -> dict:
    """SM broker'ların pozisyon-ağırlıklı ortalama maliyetini hesapla.

    Args:
        settlement_brokers: parse_settlement_text() çıktısı
        current_price: Güncel fiyat
        flow_response: institutionalFlow yanıtı (broker code→name eşleşmesi için)

    Returns:
        dict: {value, detail, sm_avg_cost, cost_ratio, sm_brokers}

    Eşikler:
        cost_ratio < 0.90 → guclu (SM %10+ kârda)
        0.90-0.98 → avantaj (SM makul kârda)
        0.98-1.05 → notr (başa baş)
        1.05-1.15 → risk (SM hafif zararda)
        > 1.15 → yuksek_risk (SM derin zararda)
    """
    if not settlement_brokers or not current_price or current_price <= 0:
        return {"value": "veri_yok", "detail": "maliyet verisi yok"}

    # Flow'dan code→name haritası oluştur
    flow_brokers = {}
    if flow_response:
        for lst in ("topBuyers", "topSellers", "byVolume"):
            for agent in flow_response.get(lst, []):
                code = agent.get("code", "")
                name = agent.get("name", "")
                if code and name:
                    flow_brokers[code] = name

    # SM broker'ları filtrele (yab_banka + fon), cost > 0
    sm_brokers = []
    for b in settlement_brokers:
        tip = _classify_settlement_broker(b["code"], flow_brokers)
        if tip in ("yab_banka", "fon") and b["cost"] > 0:
            sm_brokers.append({**b, "tip": tip})

    if not sm_brokers:
        return {"value": "veri_yok", "detail": "SM broker maliyet verisi yok"}

    # Pozisyon-ağırlıklı ortalama maliyet
    total_pos = sum(b["position_lot"] for b in sm_brokers)
    if total_pos <= 0:
        return {"value": "veri_yok", "detail": "SM pozisyon verisi yok"}

    weighted_cost = sum(b["cost"] * b["position_lot"] for b in sm_brokers)
    sm_avg_cost = weighted_cost / total_pos
    cost_ratio = sm_avg_cost / current_price

    # Eşik belirleme
    if cost_ratio < 0.90:
        value = "guclu"
        desc = f"SM %{(1-cost_ratio)*100:.0f} kârda, güçlü tutma"
    elif cost_ratio < 0.98:
        value = "avantaj"
        desc = f"SM makul kârda"
    elif cost_ratio < 1.05:
        value = "notr"
        desc = "başa baş civarı"
    elif cost_ratio < 1.15:
        value = "risk"
        desc = "SM hafif zararda"
    else:
        value = "yuksek_risk"
        desc = "SM derin zararda, satış baskısı riski"

    broker_names = ", ".join(b["code"] for b in sm_brokers[:3])
    detail = (f"{desc} — SM maliyet {sm_avg_cost:.2f} vs fiyat {current_price:.2f} "
              f"(r={cost_ratio:.2f}, {len(sm_brokers)} broker: {broker_names})")

    return {
        "value": value,
        "detail": detail,
        "sm_avg_cost": round(sm_avg_cost, 2),
        "cost_ratio": round(cost_ratio, 4),
        "sm_broker_count": len(sm_brokers),
    }


# ══════════════════════════════════════════════════════════════
# E. Batch İşleme — briefing.py entegrasyonu
# ══════════════════════════════════════════════════════════════

def process_matriks_batch(matriks_data: dict) -> tuple:
    """Matriks batch verisini SMS + ICE formatlarına dönüştür.

    Args:
        matriks_data: MatriksClient.fetch_batch() çıktısı
            {TICKER: {flows: {daily, weekly, monthly, quarterly},
                      settlement: {...}, price: {...}}}

    Returns:
        (takas_data_map, cost_data_map):
            takas_data_map: {TICKER: {kurumlar: [...]}} — SMS input (G/H/A/3A dolu)
            cost_data_map: {TICKER: {value, detail, ...}} — ICE maliyet_avantaji input
    """
    takas_data_map = {}
    cost_data_map = {}

    for ticker, data in matriks_data.items():
        flows = data.get("flows", {})
        settlement = data.get("settlement")
        price_data = data.get("price")

        # SMS takas_data formatı (4 periyot birleştirilmiş)
        if flows:
            td = flows_to_takas_data(flows, ticker)
            takas_data_map.update(td)

        # Maliyet avantajı
        current_price = None
        if price_data:
            pd = price_data.get("data", {})
            current_price = pd.get("price")

        daily_flow = flows.get("daily")
        if settlement and current_price:
            analysis_text = settlement.get("analysis", "")
            if isinstance(analysis_text, dict):
                analysis_text = analysis_text.get("_raw_text", "")
            brokers = parse_settlement_text(analysis_text)
            if brokers:
                cost_data = calc_cost_advantage(brokers, current_price, daily_flow)
                cost_data_map[ticker] = cost_data

    return takas_data_map, cost_data_map
