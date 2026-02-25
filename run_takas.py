#!/usr/bin/env python3
"""
NOX TAKAS TARAYICI v1.0
========================
İş Yatırım'dan aracı kurum dağılımı (AKD) ve takas verisi çekerek
kurumsal birikim tespit eden tarayıcı.

Amaç: Takası az sayıda büyük aracı kurumda yoğunlaşmış,
kurumsal alım gören hisseleri bul.

Kullanım:
    python run_takas.py
    python run_takas.py --symbols GARAN THYAO AKBNK
    python run_takas.py --top 20

Çıktı: Kurumsal birikim skoru yüksek olan hisseler listesi
"""

import requests
import pandas as pd
import json
import time
import sys
import os
from datetime import datetime, timedelta

# ============================================================
# BIST100 HİSSE LİSTESİ (taranacak semboller)
# ============================================================
BIST30 = [
    "AKBNK", "ARCLK", "ASELS", "BIMAS", "EKGYO",
    "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF",
    "HEKTS", "ISCTR", "KCHOL", "KOZAA", "KOZAL",
    "KRDMD", "MGROS", "ODAS", "OYAKC", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL",
    "TCELL", "THYAO", "TKFEN", "TOASO", "TUPRS",
    "YKBNK"
]

# Yabancı kurumlar (takas yoğunlaşması önemli)
YABANCI_KURUMLAR = [
    "BANK OF AMERICA", "BOFA", "MERRILL",
    "GOLDMAN SACHS", "GS",
    "J.P. MORGAN", "JPMORGAN", "JP MORGAN",
    "MORGAN STANLEY",
    "UBS",
    "CREDIT SUISSE",
    "CITIGROUP", "CITI",
    "DEUTSCHE BANK", "DEUTSCHE",
    "BARCLAYS",
    "HSBC",
    "BNP PARIBAS", "BNP",
    "SOCIETE GENERALE",
    "WOOD", "WOOD & CO",  # Wood & Co — Doğu Avrupa odaklı
]

# Büyük yerli kurumlar
BUYUK_YERLI_KURUMLAR = [
    "İŞ YATIRIM", "IS YATIRIM",
    "YAPI KREDİ", "YAPI KREDI",
    "GARANTİ", "GARANTI",
    "AKBANK",
    "QNB FİNANS", "QNB FINANS",
    "DENİZ YATIRIM", "DENIZ",
    "HALK YATIRIM",
    "VAKIF YATIRIM",
    "ZİRAAT", "ZIRAAT",
]


def get_hisse_listesi():
    """İş Yatırım'dan BIST hisse listesini çek."""
    url = "https://www.isyatirimhisse.com/api/StockExchange/GetAllStocks"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json()
            symbols = [item.get("symbol", "") for item in data if item.get("symbol")]
            return symbols
    except Exception:
        pass
    return BIST30


def get_takas_data(symbol, date_str=None):
    """
    İş Yatırım'dan bir hisse için takas verisini çek.
    Takas verisi: Hangi aracı kurumda ne kadar lot saklanıyor.
    """
    if date_str is None:
        # T-2 günü (takas 2 gün gecikmelidir)
        dt = datetime.now() - timedelta(days=2)
        # Hafta sonu kontrolü
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        date_str = dt.strftime("%d-%m-%Y")

    # İş Yatırım takas endpoint'i
    urls_to_try = [
        f"https://www.isyatirimhisse.com/api/StockExchange/GetStockCustody?symbol={symbol}&date={date_str}",
        f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/TakasGetir?hisse={symbol}&tarih={date_str}",
        f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTakas?hisse={symbol}&tarih={date_str}",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.isyatirimhisse.com/",
    }

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code == 200:
                data = r.json()
                if data and (isinstance(data, list) or isinstance(data, dict)):
                    return data
        except Exception:
            continue

    return None


def get_akd_data(symbol, date_str=None):
    """
    İş Yatırım'dan aracı kurum dağılımı (AKD) verisini çek.
    AKD: Gün içinde hangi kurum ne kadar alıp sattı.
    """
    if date_str is None:
        dt = datetime.now() - timedelta(days=1)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        date_str = dt.strftime("%d-%m-%Y")

    urls_to_try = [
        f"https://www.isyatirimhisse.com/api/StockExchange/GetStockBrokerageDistribution?symbol={symbol}&date={date_str}",
        f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/AKDGetir?hisse={symbol}&tarih={date_str}",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.isyatirimhisse.com/",
    }

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code == 200:
                data = r.json()
                if data and (isinstance(data, list) or isinstance(data, dict)):
                    return data
        except Exception:
            continue

    return None


def get_fiyat_data(symbol, days=30):
    """İş Yatırım'dan son N günlük fiyat verisini çek."""
    end_date = datetime.now().strftime("%d-%m-%Y")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")

    url = (
        f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
        f"?hisse={symbol}&startdate={start_date}&enddate={end_date}.json"
    )

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, timeout=15, headers=headers)
        if r.status_code == 200:
            data = json.loads(r.text)
            if "value" in data and data["value"]:
                df = pd.DataFrame(data["value"])
                return df
    except Exception:
        pass

    return None


def is_yabanci_kurum(kurum_adi):
    """Kurum adının yabancı kurumlar listesinde olup olmadığını kontrol et."""
    kurum_upper = kurum_adi.upper()
    for yabanci in YABANCI_KURUMLAR:
        if yabanci in kurum_upper:
            return True
    return False


def is_buyuk_yerli(kurum_adi):
    """Kurum adının büyük yerli kurumlar listesinde olup olmadığını kontrol et."""
    kurum_upper = kurum_adi.upper()
    for yerli in BUYUK_YERLI_KURUMLAR:
        if yerli in kurum_upper:
            return True
    return False


def analyze_takas(takas_data, symbol):
    """
    Takas verisini analiz et ve kurumsal birikim skoru hesapla.

    Skor kriteri:
    1. Top 3 kurumun toplam takas payı (yoğunlaşma)
    2. Yabancı kurumların takas payı
    3. "Diğer" (küçük yatırımcı) oranının düşüklüğü
    """
    if not takas_data:
        return None

    result = {
        "symbol": symbol,
        "top3_pay": 0,
        "top5_pay": 0,
        "yabanci_pay": 0,
        "buyuk_yerli_pay": 0,
        "diger_pay": 0,  # küçük yatırımcı
        "toplam_lot": 0,
        "kurum_sayisi": 0,
        "yogunlasma_skoru": 0,
        "detay": [],
    }

    # Veri formatını algıla ve parse et
    kurumlar = []

    if isinstance(takas_data, list):
        for item in takas_data:
            kurum = {
                "ad": item.get("brokerageName", item.get("KURUM_ADI", item.get("name", ""))),
                "lot": float(item.get("quantity", item.get("LOT", item.get("lot", 0)))),
                "oran": float(item.get("rate", item.get("ORAN", item.get("percentage", 0)))),
            }
            if kurum["ad"]:
                kurumlar.append(kurum)

    elif isinstance(takas_data, dict):
        items = takas_data.get("value", takas_data.get("data", takas_data.get("items", [])))
        if isinstance(items, list):
            for item in items:
                kurum = {
                    "ad": item.get("brokerageName", item.get("KURUM_ADI", item.get("name", ""))),
                    "lot": float(item.get("quantity", item.get("LOT", item.get("lot", 0)))),
                    "oran": float(item.get("rate", item.get("ORAN", item.get("percentage", 0)))),
                }
                if kurum["ad"]:
                    kurumlar.append(kurum)

    if not kurumlar:
        return None

    # Lot'a göre sırala (büyükten küçüğe)
    kurumlar.sort(key=lambda x: x["lot"], reverse=True)

    toplam_lot = sum(k["lot"] for k in kurumlar)
    result["toplam_lot"] = toplam_lot
    result["kurum_sayisi"] = len(kurumlar)

    if toplam_lot == 0:
        return None

    # Oranları hesapla
    for k in kurumlar:
        if k["oran"] == 0 and toplam_lot > 0:
            k["oran"] = (k["lot"] / toplam_lot) * 100

    # Top 3 ve Top 5 pay
    top3_lot = sum(k["lot"] for k in kurumlar[:3])
    top5_lot = sum(k["lot"] for k in kurumlar[:5])
    result["top3_pay"] = round((top3_lot / toplam_lot) * 100, 2) if toplam_lot > 0 else 0
    result["top5_pay"] = round((top5_lot / toplam_lot) * 100, 2) if toplam_lot > 0 else 0

    # Yabancı ve yerli pay
    yabanci_lot = sum(k["lot"] for k in kurumlar if is_yabanci_kurum(k["ad"]))
    yerli_lot = sum(k["lot"] for k in kurumlar if is_buyuk_yerli(k["ad"]))
    result["yabanci_pay"] = round((yabanci_lot / toplam_lot) * 100, 2) if toplam_lot > 0 else 0
    result["buyuk_yerli_pay"] = round((yerli_lot / toplam_lot) * 100, 2) if toplam_lot > 0 else 0

    # "Diğer" oranı (top 10 dışı = küçük yatırımcı proxy'si)
    top10_lot = sum(k["lot"] for k in kurumlar[:10])
    diger_lot = toplam_lot - top10_lot
    result["diger_pay"] = round((diger_lot / toplam_lot) * 100, 2) if toplam_lot > 0 else 0

    # Detay (top 10 kurum)
    result["detay"] = [
        {
            "ad": k["ad"],
            "lot": k["lot"],
            "oran": round(k["oran"], 2),
            "tip": "YABANCI" if is_yabanci_kurum(k["ad"]) else ("BUYUK YERLI" if is_buyuk_yerli(k["ad"]) else "DIGER"),
        }
        for k in kurumlar[:10]
    ]

    # ==========================================
    # KURUMSAL BİRİKİM SKORU (0-100)
    # ==========================================
    # Ağırlıklar:
    # - Top 3 yoğunlaşma: %30 (yüksek = birkaç kurum topluyor)
    # - Yabancı pay: %30 (yüksek = uluslararası ilgi)
    # - Düşük "diğer" oranı: %20 (düşük = küçük yatırımcı az)
    # - Az kurum sayısı: %20 (az kurum = konsantre pozisyon)

    # Top3 skoru (0-100): %60+ top3 payı ideal
    top3_skor = min(100, (result["top3_pay"] / 60) * 100)

    # Yabancı skoru (0-100): %30+ yabancı payı ideal
    yabanci_skor = min(100, (result["yabanci_pay"] / 30) * 100)

    # Diğer skoru (0-100): %20 altı "diğer" ideal (ters oran)
    diger_skor = max(0, 100 - (result["diger_pay"] / 50) * 100)

    # Kurum sayısı skoru (0-100): 5-15 arası ideal
    if result["kurum_sayisi"] <= 15:
        kurum_skor = 100
    elif result["kurum_sayisi"] <= 30:
        kurum_skor = 60
    else:
        kurum_skor = max(0, 100 - (result["kurum_sayisi"] - 30) * 2)

    result["yogunlasma_skoru"] = round(
        top3_skor * 0.30 +
        yabanci_skor * 0.30 +
        diger_skor * 0.20 +
        kurum_skor * 0.20,
        1
    )

    return result


def scan_all(symbols=None, verbose=True):
    """Tüm sembolleri tara ve kurumsal birikim skoruna göre sırala."""
    if symbols is None:
        symbols = BIST30

    print(f"\n{'='*70}")
    print(f"  NOX TAKAS TARAYICI v1.0")
    print(f"  Taranan sembol sayisi: {len(symbols)}")
    print(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    results = []

    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"  [{i+1}/{len(symbols)}] {symbol} taraniyor...", end=" ")

        takas = get_takas_data(symbol)
        analysis = analyze_takas(takas, symbol)

        if analysis:
            results.append(analysis)
            if verbose:
                print(f"Skor: {analysis['yogunlasma_skoru']:.1f} | "
                      f"Top3: %{analysis['top3_pay']:.1f} | "
                      f"Yabanci: %{analysis['yabanci_pay']:.1f}")
        else:
            if verbose:
                print("VERI YOK (endpoint kapali veya ucretli olabilir)")

        # Rate limiting
        time.sleep(0.5)

    # Sonuçları skorla sırala
    results.sort(key=lambda x: x["yogunlasma_skoru"], reverse=True)

    return results


def print_results(results, top_n=20):
    """Sonuçları güzel formatta yazdır."""
    if not results:
        print("\n  ! Hic veri alinamadi. Olasi nedenler:")
        print("    1. Is Yatirim takas endpoint'i degismis olabilir")
        print("    2. BIST takas verileri 2025'ten itibaren ucretli — lisans gerekebilir")
        print("    3. Internet baglantisi sorunu")
        print("\n  Alternatif veri kaynaklari:")
        print("    - MatriksIQ (MKK/Hisse Takas Verisi lisansi)")
        print("    - Fintables Pro")
        print("    - Finnet2000")
        return

    print(f"\n{'='*80}")
    print(f"  NOX TAKAS TARAYICI — KURUMSAL BIRIKIM RAPORU")
    print(f"  Top {min(top_n, len(results))} hisse (yogunlasma skoruna gore)")
    print(f"{'='*80}")
    print(f"\n  {'#':>3} {'Sembol':<8} {'Skor':>6} {'Top3%':>7} {'Top5%':>7} "
          f"{'Ybnc%':>7} {'Yerli%':>7} {'Diger%':>7} {'Kurum#':>7}")
    print(f"  {'-'*68}")

    for i, r in enumerate(results[:top_n]):
        marker = " *" if r["yogunlasma_skoru"] >= 70 else "  " if r["yogunlasma_skoru"] >= 50 else ""
        print(f"  {i+1:>3} {r['symbol']:<8} {r['yogunlasma_skoru']:>5.1f}{marker} "
              f"{r['top3_pay']:>6.1f} {r['top5_pay']:>6.1f} "
              f"{r['yabanci_pay']:>6.1f} {r['buyuk_yerli_pay']:>6.1f} "
              f"{r['diger_pay']:>6.1f} {r['kurum_sayisi']:>6}")

    # Detay: En yüksek skorlu ilk 3 hissenin kurum dağılımı
    print(f"\n{'='*80}")
    print(f"  DETAY — En Yuksek Skorlu Hisseler")
    print(f"{'='*80}")

    for r in results[:3]:
        if r["detay"]:
            print(f"\n  {r['symbol']} (Skor: {r['yogunlasma_skoru']:.1f})")
            print(f"  {'Kurum':<35} {'Lot':>12} {'Oran%':>8} {'Tip':<12}")
            print(f"  {'-'*67}")
            for d in r["detay"]:
                print(f"  {d['ad'][:34]:<35} {d['lot']:>12,.0f} {d['oran']:>7.2f} {d['tip']:<12}")

    print(f"\n  * = Skor >= 70 (Guclu kurumsal birikim)")
    print(f"  Bos = Skor < 50 (Daginik yapi)")
    print()


def export_to_csv(results, filename=None):
    """Sonuçları CSV'ye kaydet."""
    if not results:
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = os.path.join(
            output_dir,
            f"nox_takas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )

    rows = []
    for r in results:
        rows.append({
            "Sembol": r["symbol"],
            "Birikim_Skoru": r["yogunlasma_skoru"],
            "Top3_Pay": r["top3_pay"],
            "Top5_Pay": r["top5_pay"],
            "Yabanci_Pay": r["yabanci_pay"],
            "Buyuk_Yerli_Pay": r["buyuk_yerli_pay"],
            "Diger_Pay": r["diger_pay"],
            "Kurum_Sayisi": r["kurum_sayisi"],
            "Toplam_Lot": r["toplam_lot"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"  CSV: {filename}")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    symbols = BIST30

    # Komut satırı argümanları
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        symbols = sys.argv[idx+1:]
        symbols = [s for s in symbols if not s.startswith("--")]

    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        top_n = int(sys.argv[idx+1])
    else:
        top_n = 20

    # Taramayı başlat
    results = scan_all(symbols)

    # Sonuçları yazdır
    print_results(results, top_n)

    # CSV'ye kaydet
    if results:
        export_to_csv(results)
