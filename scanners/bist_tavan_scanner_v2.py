"""
BIST TAVAN TARAYICI v3 — BIST100 DIŞI TARAMA + YABANCI ORAN
=============================================================
XTüm'deki BIST100 dışı hisseleri tarar.
Küçük/orta ölçekli, tavan potansiyeli yüksek segment.

v3 Yenilikler:
  - Backtest-kalibreli skor algoritması
  - İsyatırım API'den yabancı oran entegrasyonu
  - Streak-dominant scoring (uzun seri = güçlü momentum)
  - Ters hacim mantığı (düşük hacim = kilitli = iyi)
  - Likidite filtresi (500K TL günlük minimum)

Veri kaynağı: yfinance (.IS suffix) + İsyatırım yabancı oran API
Hisse listesi: İsyatırım web scraping + geniş statik fallback

Gereksinimler:
  pip install yfinance pandas numpy requests beautifulsoup4

Kullanım:
  python bist_tavan_scanner_v2.py
  python bist_tavan_scanner_v2.py --all          # BIST100 dahil tümünü tara
  python bist_tavan_scanner_v2.py --bist100      # Sadece BIST100 tara
  python bist_tavan_scanner_v2.py --no-yabanci   # Yabancı oran olmadan çalıştır
"""

# NOX: yfinance yerine extfeed (taze nox-project verisi). SADECE fetch_data/fetch_batch
# gövdeleri shim'e delege edildi; tavan tespit/yabancı-oran/yayın mantığı DOKUNULMADI.
from _extfeed_daily import (
    fetch_data_single as _xf_fetch_data_single,
    fetch_batch_tavan as _xf_fetch_batch_tavan,
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
import time
import sys
import json
import os
warnings.filterwarnings('ignore')


# ============================================================
# 0) YABANCI ORAN MODÜLÜnü (İsyatırım API)
# ============================================================

ISYATIRIM_YABANCI_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/"
    "IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetYabanciOranlarXHR"
)

ISYATIRIM_HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/111.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.isyatirim.com.tr",
    "Referer": "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx",
}

YABANCI_CACHE_FILE = "yabanci_oran_cache.json"


def fetch_yabanci_oran_bulk(tarih: str = None, days_back: int = 5) -> dict:
    """
    İsyatırım'dan tüm hisselerin yabancı oranlarını çeker.
    
    Endpoint kuralları (keşfedilen):
    - Sadece tek gün sorgulanabilir (baslangic = bitis)
    - Sadece iş günleri veri döner (hafta sonu = 0 sonuç)
    - Aralık sorgusu çalışmıyor (her zaman 0 döner)
    - hisse=null → tüm hisseler, endeks="09" → tüm BIST
    
    Strateji: Son iş günlerini geriye doğru tek tek sorgula,
    en güncel + days_back önceki oranları karşılaştır.
    
    Returns:
        dict: {
            'THYAO': {
                'oran': 23.98,          # En güncel yabancı oranı (%)
                'oran_onceki': 23.69,   # days_back iş günü önceki oran
                'degisim': 0.29,        # Puan değişimi
                'degisim_pct': 1.21,    # Yüzde değişim
                'tarih': '09-02-2026',  # Veri tarihi
            },
            ...
        }
    """
    if tarih is None:
        ref_date = datetime.now()
    else:
        ref_date = datetime.strptime(tarih, "%d-%m-%Y")
    
    # Son N iş gününü bul (geriye doğru)
    def find_business_days(from_date, count):
        """Geriye doğru count adet iş günü tarihi döner.
        Bayram tatilleri 9+ gün olabilir — yeterli margin bırak."""
        dates = []
        d = from_date
        max_attempts = count * 4 + 20  # Bayram güvenliği
        attempts = 0
        while len(dates) < count and attempts < max_attempts:
            d_str = d.strftime("%d-%m-%Y")
            if d.weekday() < 5:  # Pazartesi-Cuma
                dates.append(d_str)
            d -= timedelta(days=1)
            attempts += 1
        return dates
    
    # Yeterli iş günü üret (en güncel + karşılaştırma için eski)
    biz_days = find_business_days(ref_date, days_back + 5)
    
    print(f"📡 Yabancı oranları çekiliyor (son iş günleri deneniyor)...")
    
    # En güncel veriyi bul
    latest_data = None
    latest_tarih = None
    for tarih_str in biz_days[:5]:  # İlk 5 iş gününü dene
        data = _fetch_single_day(tarih_str)
        if data:
            latest_data = data
            latest_tarih = tarih_str
            print(f"  ✅ Güncel veri: {tarih_str} ({len(data)} hisse)")
            break
    
    if not latest_data:
        print("⚠️ Güncel yabancı oran verisi bulunamadı")
        return _load_yabanci_cache()
    
    # Karşılaştırma için eski veriyi bul (days_back iş günü önce)
    older_data = None
    older_tarih = None
    for tarih_str in biz_days[days_back:]:  # days_back sonrasından dene
        data = _fetch_single_day(tarih_str)
        if data:
            older_data = data
            older_tarih = tarih_str
            print(f"  ✅ Eski veri: {tarih_str} ({len(data)} hisse)")
            break
    
    # Sonuçları birleştir
    result = {}
    for kod, item in latest_data.items():
        oran_end = item.get("YAB_ORAN_END", 0) or 0
        
        # Eski veriyle karşılaştır
        oran_start = oran_end  # default: değişim yok
        if older_data and kod in older_data:
            oran_start = older_data[kod].get("YAB_ORAN_END", 0) or 0
        
        degisim = oran_end - oran_start
        degisim_pct = ((oran_end - oran_start) / oran_start * 100) if oran_start > 0 else 0
        
        result[kod] = {
            "oran": round(oran_end, 4),
            "oran_onceki": round(oran_start, 4),
            "degisim": round(degisim, 4),
            "degisim_pct": round(degisim_pct, 2),
            "tarih": latest_tarih,
        }
    
    period_str = f"{older_tarih} → {latest_tarih}" if older_tarih else latest_tarih
    print(f"✅ {len(result)} hisse için yabancı oranı alındı ({period_str})")
    
    # Cache'e kaydet
    _save_yabanci_cache(result, latest_tarih)
    
    return result


def _fetch_single_day(tarih_str: str) -> dict:
    """
    Tek bir iş günü için tüm hisselerin yabancı oranlarını çeker.
    
    Returns:
        dict: {HISSE_KODU: {YAB_ORAN_END: ..., ...}} veya boş dict
    """
    payload = {
        "baslangicTarih": tarih_str,
        "bitisTarihi": tarih_str,
        "sektor": None,
        "endeks": "09",
        "hisse": None,
    }
    
    try:
        resp = requests.post(
            ISYATIRIM_YABANCI_URL,
            json=payload,
            headers=ISYATIRIM_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("d", [])
        if not items:
            return {}
        
        return {item["HISSE_KODU"]: item for item in items if item.get("HISSE_KODU")}
    except Exception as e:
        print(f"  [HATA] Yabancı oran çekilemedi ({tarih_str}): {e}")
        return {}


def fetch_yabanci_oran_tek(hisse: str, days_back: int = 5) -> dict:
    """Tek hisse için yabancı oranı çeker (toplu endpoint'ten filtreleyerek)."""
    result = fetch_yabanci_oran_bulk(days_back=days_back)
    return result.get(hisse, {})


def _save_yabanci_cache(data: dict, tarih: str):
    """Yabancı oran verisini JSON cache'e yazar."""
    cache = {"tarih": tarih, "data": data}
    try:
        with open(YABANCI_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [UYARI] Cache kaydedilemedi: {e}")


def _load_yabanci_cache() -> dict:
    """Önceki cache'den yabancı oran verisini okur."""
    try:
        with open(YABANCI_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            tarih = cache.get("tarih", "?")
            data = cache.get("data", {})
            if data:
                print(f"📦 Cache'den yüklendi ({tarih}, {len(data)} hisse)")
            return data
    except Exception as e:
        print(f"  [UYARI] Cache okunamadı: {e}")
        return {}


def yabanci_skor_bonusu(yabanci_data: dict, ticker: str) -> tuple:
    """
    Yabancı oranı değişimine göre skor bonusu/cezası.

    Forward test bulgusu (18 gün, 232 tavan sinyali):
    - Yabancı artan: 1G WR %60.4 vs artmayan %55.6 — fark marjinal (+5pp)
    - Gerçek tavan sürücüsü yerli kurumlar (Tera, Marbas, Info, Albatros)
    - Bonus düşürüldü: +8/+5 → +3/+2, düşüş cezası kaldırıldı

    Returns:
        (bonus_puan: int, aciklama: str veya None)
    """
    if not yabanci_data or ticker not in yabanci_data:
        return (0, None)

    info = yabanci_data[ticker]
    oran = info.get("oran", 0)
    degisim = info.get("degisim", 0)

    if oran < 0.1:
        return (0, None)

    if degisim >= 0.5:
        return (+3, f"🌍 Yabancı artıyor: %{oran:.2f} ({degisim:+.2f}pp)")
    elif degisim >= 0.1:
        return (+2, f"🌍 Yabancı yükseliyor: %{oran:.2f} ({degisim:+.2f}pp)")
    elif degisim <= -0.5:
        return (0, f"🌍 Yabancı azalıyor: %{oran:.2f} ({degisim:+.2f}pp)")
    elif degisim <= -0.1:
        return (0, f"🌍 Yabancı düşüyor: %{oran:.2f} ({degisim:+.2f}pp)")

    return (0, f"🌍 Yabancı: %{oran:.2f} (değişim yok)")


# ============================================================
# 1) HİSSE LİSTELERİ
# ============================================================

BIST100 = {
    "ACSEL","ADEL","AEFES","AFYON","AGESA","AGHOL","AKBNK","AKCNS","AKENR",
    "AKFGY","AKGRT","AKSA","AKSEN","AKSGY","ALARK","ALFAS","ALGYO","ALKIM",
    "ANHYT","ANSGR","ARCLK","ARDYZ","ARENA","ASELS","ASGYO","ASTOR","ASUZU",
    "ATAGY","ATAKP","ATLAS","AYEN","AYGAZ","BAGFS","BERA","BFREN","BIENY",
    "BIGCH","BIMAS","BIOEN","BJKAS","BOBET","BRISA","BRSAN","BRYAT","BTCIM",
    "BUCIM","CCOLA","CIMSA","CWENE","DOAS","DOHOL","ECILC","ECZYT","EGEEN",
    "EKGYO","ENJSA","ENKAI","ERBOS","EREGL","EUPWR","EUREN","FROTO","GARAN",
    "GENIL","GESAN","GLYHO","GOLTS","GSRAY","GUBRF","HEKTS","HLGYO","HUBVC",
    "IMASM","INDES","IPEKE","ISDMR","ISGYO","ISMEN","ISSEN","KARSN","KAYSE",
    "KCHOL","KLSER","KMPUR","KONTR","KONYA","KOZAA","KOZAL","KRDMD","KZBGY",
    "LMKDC","MAVI","MGROS","MIATK","MPARK","ODAS","OTKAR","OYAKC","PAPIL",
    "PEKGY","PETKM","PGSUS","REEDR","SAHOL","SASA","SISE","SKBNK","SOKM",
    "SRVGY","TABGD","TATGD","TAVHL","TCELL","THYAO","TKFEN","TKNSA","TOASO",
    "TRGYO","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VERUS","VESTL",
    "YEOTK","YKBNK","YUNSA","ZOREN",
}

# XTüm geniş statik liste — BIST'te işlem gören ~550 hisse
# Bu liste periyodik olarak güncellenmelidir
XTUM_ALL = [
    # A
    "ACSEL","ADEL","ADESE","ADGYO","AEFES","AFYON","AGESA","AGHOL","AGROT",
    "AGYO","AHGAZ","AKBNK","AKCNS","AKENR","AKFGY","AKFYE","AKGRT","AKHAN","AKMGY",
    "AKSA","AKSEN","AKSGY","AKSUE","AKYHO","ALARK","ALBRK","ALCAR","ALFAS",
    "ALGYO","ALKA","ALKIM","ALMAD","ANELE","ANGEN","ANHYT","ANSGR","ARASE",
    "ARCLK","ARDYZ","ARENA","ARSAN","ARTMS","ARZUM","ASELS","ASGYO","ASTOR",
    "ASUZU","ATAGY","ATAKP","ATATP","ATEKS","ATLAS","ATSYH","AVHOL","AVOD",
    "AVTUR","AYCES","AYDEM","AYEN","AYGAZ",
    # B
    "BAGFS","BAKAB","BALAT","BANVT","BARMA","BASCM","BASGZ","BAYRK","BEGYO",
    "BERA","BEYAZ","BFREN","BIENY","BIGCH","BIMAS","BINHO","BIOEN","BIZIM",
    "BJKAS","BLCYT","BMSCH","BMSTL","BNTAS","BOBET","BORLS","BORSK","BOSSA",
    "BRISA","BRKO","BRKSN","BRKVY","BRLSM","BRMEN","BRSAN","BRYAT","BSOKE",
    "BTCIM","BUCIM","BURCE","BURVA",
    # C-Ç
    "CASA","CANTE","CCOLA","CELHA","CEMAS","CEMTS","CEOEM","CIMSA","CLEBI",
    "CMBTN","CMENT","CONSE","COSMO","CRDFA","CRFSA","CUSAN","CVKMD","CWENE",
    # D
    "DAGI","DAPGM","DARDL","DENGE","DERHL","DERIM","DESA","DESPC","DEVA",
    "DGATE","DGGYO","DGNMO","DIRIT","DITAS","DMRGD","DNISI","DOAS","DOBUR",
    "DOCO","DOHOL","DOKTA","DURDO","DYOBY",
    # E
    "ECILC","ECZYT","EDATA","EDIP","EGEEN","EGEPO","EGPRO","EGSER","EKGYO",
    "EKIZ","EKOS","EKSUN","ELITE","EMKEL","EMNIS","ENJSA","ENKAI","ENSRI",
    "EPLAS","ERBOS","ERCB","EREGL","ERSU","ESCAR","ESCOM","ESEN","ETILR",
    "ETYAT","EUHOL","EUKYO","EUPWR","EUREN","EUYO","EVREN","EXALP",
    # F
    "FADE","FENER","FLAP","FMIZP","FONET","FORMT","FORTE","FROTO","FZLGY",
    # G
    "GARAN","GARFA","GEDIK","GEDZA","GENIL","GENTS","GEREL","GESAN","GIPTS",
    "GLBMD","GLCVY","GLYHO","GMTAS","GOKNR","GOLTS","GOODY","GOZDE","GRSEL",
    "GRTRK","GSDDE","GSDHO","GSRAY","GUBRF","GWIND","GZNMI",
    # H
    "HATEK","HDFGS","HEDEF","HEKTS","HKTM","HLGYO","HTTBT","HUBVC","HUNER",
    "HURGZ",
    # I-İ
    "ICBCT","IEYHO","IHAAS","IHEVA","IHGZT","IHLAS","IHLGM","IHYAY","IMASM",
    "INDES","INFO","INGRM","INTEM","INVEO","INVES","IPEKE","ISDMR","ISGSY",
    "ISGYO","ISKPL","ISKUR","ISMEN","ISSEN","ISYAT","IZENR","IZFAS","IZMDC",
    # K
    "KAPLM","KARDM","KARTN","KARYE","KARSN","KATMR","KAYSE","KCAER","KCHOL",
    "KENT","KERVN","KERVT","KFEIN","KGYO","KIMMR","KLGYO","KLMSN","KLNMA",
    "KLSER","KLSYN","KMPUR","KNFRT","KONKA","KONTR","KONYA","KOPOL","KORDS",
    "KOZAA","KOZAL","KRDMA","KRDMB","KRDMD","KRGYO","KRONT","KRPLS","KRSTL",
    "KRTEK","KRVGD","KTLEV","KTSKR","KUTPO","KUVVA","KZBGY",
    # L
    "LIDER","LILAK","LKMNH","LMKDC","LOGO","LRSHO","LUKSK",
    # M
    "MACKO","MAKIM","MAKTK","MANAS","MARKA","MARTI","MAVI","MEDTR","MEGAP",
    "MEKAG","MERCN","MERKO","METRO","METUR","MGROS","MHRGY","MIATK","MNDRS",
    "MNDTR","MOBTL","MOGAN","MPARK","MRGYO","MRSHL","MSGYO","MTRKS","MTRYO",
    "MZHLD",
    # N
    "NATEN","NETCD","NETAS","NIBAS","NUGYO","NUHCM","NUREN",
    # O
    "OBAMS","ODAS","OFSYM","ONCSM","ORCAY","ORGE","ORMA","OSCYM","OSTIM",
    "OTKAR","OTTO","OYAKC","OYLUM","OYYAT","OZGYO","OZKGY","OZRDN","OZSUB",
    # P
    "PAGYO","PAMEL","PAPIL","PARSN","PASEU","PCILT","PEGYO","PEKGY","PENGD",
    "PENTA","PETKM","PETUN","PGSUS","PINSU","PKART","PKENT","PLTUR","PNLSN",
    "PNSUT","POLHO","POLTK","PRDGS","PRKAB","PRKME","PSGYO","PSDTC",
    # Q-R
    "RALYH","RAYSG","REEDR","RGYAS","RNPOL","RODRG","ROYAL","RUBNS","RYGYO",
    "RYSAS",
    # S-Ş
    "SAFKR","SAHOL","SAMAT","SANEL","SANFM","SANKO","SARKY","SASA","SAYAS",
    "SDTTR","SEKFK","SEKUR","SELEC","SELGD","SELVA","SEYKM","SILVR","SISE",
    "SKBNK","SKY","SKYMD","SMART","SMRTG","SNGYO","SNICA","SNKRN","SNPAM",
    "SODSN","SOKM","SRVGY","SUMAS","SUNEV","SUWEN","SYRMO",
    # T
    "TABGD","TARKM","TATEN","TATGD","TAVHL","TCELL","TCZB","TDGYO","TEKTU",
    "TERA","TETMT","TGSAS","THYAO","TKFEN","TKNSA","TLMAN","TMPOL","TMSN",
    "TNZTP","TOASO","TRCAS","TRGYO","TRILC","TSGYO","TSKB","TTKOM","TTRAK",
    "TUCLK","TUKAS","TUNCA","TUPRS","TUREX","TURSG","UFUK",
    # U-Ü
    "ULKER","ULUFA","ULUSE","ULUUN","UMPAS","UNLU","USAK","UZERB",
    # V
    "VAKBN","VAKFN","VAKKO","VANGD","VBTYZ","VERTU","VERUS","VESTL","VKFYO",
    "VKGYO","VRGYO",
    # Y
    "YAPRK","YATAS","YAYLA","YEOTK","YGGYO","YGYO","YKBNK","YKSLN","YUNSA",
    "YYAPI","YYLGD",
    # Z
    "ZEDUR","ZGYO","ZOREN","ZRGYO","UCAYM","FRMPL","MEYSU","ARFYE","ZERGY","PAHOL","VAKFA","ECOGR","MARMR","DOFRB","BULGS","BALSU","KLYPV","ENDAE",
]


def get_tickers(mode="ex_bist100"):
    """
    Hisse listesini döndürür.
    mode: 'ex_bist100' (BIST100 dışı), 'all' (tümü), 'bist100' (sadece BIST100)
    """
    # Önce yerel dosya var mı bak
    file_list = load_tickers_from_file()
    if file_list:
        base_list = file_list
    else:
        # Güncel listeyi çekmeyi dene (birden fazla kaynak)
        live_list = fetch_live_ticker_list()
        base_list = live_list if live_list else XTUM_ALL

    if mode == "bist100":
        return [t for t in base_list if t in BIST100]
    elif mode == "ex_bist100":
        return [t for t in base_list if t not in BIST100]
    else:  # all
        return base_list


def fetch_live_ticker_list():
    """
    Birden fazla kaynaktan güncel BIST hisse listesi çekmeye çalışır.
    Tüm kaynakları birleştirir — en geniş listeyi oluşturur.
    """
    all_tickers = set()
    
    # ---- Yöntem 1: TradingView Screener API ----
    tickers = _fetch_from_tradingview()
    if tickers:
        all_tickers.update(tickers)
    
    # ---- Yöntem 2: İsyatırım API endpointleri ----
    tickers = _fetch_from_isyatirim()
    if tickers:
        all_tickers.update(tickers)
    
    # ---- Yöntem 3: İsyatırım Yabancı Oran API (580+ hisse) ----
    tickers = _fetch_from_yabanci_api()
    if tickers:
        all_tickers.update(tickers)
    
    # ---- Yöntem 4: Statik listeyi de ekle ----
    all_tickers.update(XTUM_ALL)
    
    if len(all_tickers) > 100:
        result = sorted(all_tickers)
        print(f"✅ Toplam {len(result)} benzersiz BIST hissesi (tüm kaynaklar birleşik).")
        return result
    
    print("  ⚠️ Güncel liste çekilemedi. Statik liste kullanılacak.")
    return None


def _fetch_from_yabanci_api():
    """İsyatırım yabancı oran API'sinden hisse listesi çeker (580+ hisse)."""
    try:
        # Son iş gününü bul
        from datetime import datetime, timedelta
        d = datetime.now()
        for _ in range(10):
            d -= timedelta(days=1)
            if d.weekday() < 5:
                break
        tarih = d.strftime("%d-%m-%Y")
        
        payload = {
            "baslangicTarih": tarih,
            "bitisTarihi": tarih,
            "sektor": None,
            "endeks": "09",
            "hisse": None,
        }
        resp = requests.post(ISYATIRIM_YABANCI_URL, json=payload, 
                           headers=ISYATIRIM_HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("d", [])
            if items:
                tickers = [item["HISSE_KODU"] for item in items if item.get("HISSE_KODU")]
                if len(tickers) > 100:
                    print(f"  📡 Yabancı oran API'den {len(tickers)} hisse kodu alındı.")
                    return tickers
    except Exception as e:
        print(f"  [UYARI] Yabancı API hisse listesi hatası: {e}")
    return None


def _fetch_from_tradingview():
    """TradingView screener API'den BIST hisse listesi çeker."""
    try:
        url = "https://scanner.tradingview.com/turkey/scan"
        payload = {
            "columns": ["name"],
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BIST"},
                {"left": "type", "operation": "equal", "right": "stock"},
                {"left": "is_primary", "operation": "equal", "right": True},
            ],
            "options": {"lang": "tr"},
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [0, 1000],
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Content-Type': 'application/json',
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            tickers = []
            for item in data.get('data', []):
                symbol = item.get('s', '')
                # Format: "BIST:THYAO" → "THYAO"
                if ':' in symbol:
                    symbol = symbol.split(':')[1]
                if symbol and len(symbol) >= 2:
                    tickers.append(symbol)
            if tickers:
                print(f"✅ TradingView'dan {len(tickers)} BIST hissesi çekildi (güncel).")
                return tickers
    except Exception as e:
        print(f"  [UYARI] TradingView hatası: {e}")
    return None


def _fetch_from_isyatirim():
    """İsyatırım API endpointlerinden hisse listesi çeker."""
    endpoints = [
        {
            'url': 'https://www.isyatirim.com.tr/_layouts/15/Jeeves/Jes498/Ch.ashx',
            'params': {'Ession': '', 'Type': 'ALL', 'Lang': 'tr-TR'},
        },
        {
            'url': 'https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common/Data.aspx/StockList',
            'params': {},
        },
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'https://www.isyatirim.com.tr/',
        'Accept': 'application/json, text/plain, */*',
    }
    for ep in endpoints:
        try:
            resp = requests.get(ep['url'], params=ep['params'], headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                tickers = []
                items = data if isinstance(data, list) else data.get('d', []) if isinstance(data, dict) else []
                for item in items:
                    if isinstance(item, dict):
                        sym = (item.get('Sembol') or item.get('c') or 
                               item.get('code') or item.get('Kod', ''))
                        if sym and len(sym) >= 2 and sym.replace('.','').isalpha():
                            tickers.append(sym)
                if len(tickers) > 100:
                    print(f"✅ İsyatırım'dan {len(tickers)} hisse çekildi (güncel).")
                    return tickers
        except Exception as e:
            print(f"  [UYARI] İsyatırım hatası: {e}")
            continue
    return None


def load_tickers_from_file():
    """
    Yerel tickers.txt dosyasından hisse listesi okur.
    Her satırda bir hisse kodu olmalı.
    """
    for path in ['tickers.txt', 'hisseler.txt', 'xtum.txt']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            if tickers:
                print(f"✅ {path} dosyasından {len(tickers)} hisse okundu.")
                return tickers
    return None


# ============================================================
# 2) VERİ ÇEKME (yfinance)
# ============================================================

def fetch_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """NOX: extfeed günlük OHLCV (yf yerine). Dönüş: kapitalize-OHLCV df (DatetimeIndex)."""
    return _xf_fetch_data_single(ticker, period=period)


def fetch_batch(tickers: list, period: str = "6mo", batch_size: int = 50) -> dict:
    """NOX: extfeed toplu OHLCV (yf yerine). İmza + dönüş şekli ({ticker: df}) değişmedi."""
    return _xf_fetch_batch_tavan(tickers, period=period, batch_size=batch_size)


# ============================================================
# 3) TEKNİK İNDİKATÖRLER
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm teknik indikatörleri hesaplar."""
    d = df.copy()

    d['returns'] = d['Close'].pct_change()

    # Hareketli Ortalamalar
    for w in [5, 10, 20, 21, 50]:
        d[f'SMA_{w}'] = d['Close'].rolling(w).mean()
        d[f'EMA_{w}'] = d['Close'].ewm(span=w).mean()

    # EMA21 uzaklığı (%)
    d['ema21_dist'] = (d['Close'] / d['EMA_21'] - 1) * 100

    # ATR
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift(1)).abs()
    low_close = (d['Low'] - d['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d['ATR_14'] = true_range.rolling(14).mean()

    # RSI
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9).mean()
    d['MACD_hist'] = d['MACD'] - d['MACD_signal']

    # Bollinger Bands
    d['BB_mid'] = d['Close'].rolling(20).mean()
    bb_std = d['Close'].rolling(20).std()
    d['BB_upper'] = d['BB_mid'] + 2 * bb_std
    d['BB_lower'] = d['BB_mid'] - 2 * bb_std
    d['BB_width'] = (d['BB_upper'] - d['BB_lower']) / d['BB_mid']

    # Keltner Channel
    d['KC_mid'] = d['Close'].ewm(span=20).mean()
    d['KC_upper'] = d['KC_mid'] + 1.5 * d['ATR_14']
    d['KC_lower'] = d['KC_mid'] - 1.5 * d['ATR_14']

    # Squeeze
    d['squeeze_on'] = (d['BB_lower'] > d['KC_lower']) & (d['BB_upper'] < d['KC_upper'])

    # Squeeze Momentum
    highest_high = d['High'].rolling(20).max()
    lowest_low = d['Low'].rolling(20).min()
    d['squeeze_mom'] = d['Close'] - ((highest_high + lowest_low) / 2 + d['Close'].rolling(20).mean()) / 2

    # ADX
    plus_dm = d['High'].diff()
    minus_dm = -d['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = true_range.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    d['ADX'] = dx.rolling(14).mean()
    d['PLUS_DI'] = plus_di
    d['MINUS_DI'] = minus_di

    # Hacim Analizi
    d['vol_sma_20'] = d['Volume'].rolling(20).mean()
    d['vol_ratio'] = d['Volume'] / d['vol_sma_20'].replace(0, np.nan)

    # Kapanış Pozisyonu (0=low, 1=high)
    daily_range = (d['High'] - d['Low']).replace(0, np.nan)
    d['close_position'] = (d['Close'] - d['Low']) / daily_range

    # ROC
    d['ROC_5'] = d['Close'].pct_change(5) * 100
    d['ROC_10'] = d['Close'].pct_change(10) * 100

    # ---- CMF (Chaikin Money Flow) — para giriş/çıkış ----
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    # MFV = MFM * Volume
    # CMF = sum(MFV, 20) / sum(Volume, 20)
    mfm = ((d['Close'] - d['Low']) - (d['High'] - d['Close'])) / daily_range
    mfm = mfm.fillna(0)
    mfv = mfm * d['Volume']
    d['CMF_20'] = mfv.rolling(20).sum() / d['Volume'].rolling(20).sum().replace(0, np.nan)

    # ---- OBV (On Balance Volume) ----
    obv = [0]
    for i in range(1, len(d)):
        if d['Close'].iloc[i] > d['Close'].iloc[i-1]:
            obv.append(obv[-1] + d['Volume'].iloc[i])
        elif d['Close'].iloc[i] < d['Close'].iloc[i-1]:
            obv.append(obv[-1] - d['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    d['OBV'] = obv
    d['OBV_SMA_20'] = d['OBV'].rolling(20).mean()
    # OBV trend: OBV ortalamanın üstünde mi?
    d['OBV_trend'] = (d['OBV'] > d['OBV_SMA_20']).astype(int)

    # ---- GAP (Boşluklu açılış) ----
    # Gap = (Bugünün Open - Dünün Close) / Dünün Close
    d['gap_pct'] = (d['Open'] - d['Close'].shift(1)) / d['Close'].shift(1).replace(0, np.nan)

    return d


# ============================================================
# 4) TAVAN TESPİTİ
# ============================================================

TAVAN_ESIK = 0.095  # %9.5+ ≈ tavan (BIST %10 limit, tick yuvarlama toleransı)

def detect_tavan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limit-price bazlı tavan tespiti.

    auto_adjust=False ile gerçek işlem fiyatları üzerinden hesaplanır.
    Eklenen kolonlar:
      - limit_up_price: Önceki kapanışa göre tavan fiyatı
      - hit_tavan_intraday: Gün içi tavana değdi mi (High >= limit)
      - closed_at_tavan: Tavanda kapandı mı
      - tavan_locked: Kilitli tavan (tavanda kapanış + close_pos >= 0.95)
      - is_tavan: Ana flag (closed_at_tavan, geriye uyumlu)
      - is_taban: Tabanda kapanış
      - tavan_streak: Ardışık tavan gün sayısı
    """
    d = df.copy()

    prev_close = d['Close'].shift(1)
    d['limit_up_price'] = prev_close * (1 + TAVAN_ESIK)
    limit_down_price = prev_close * (1 - TAVAN_ESIK)

    # Tolerans: tick yuvarlama ve spread için %0.5
    tol = 0.995

    # Gün içi tavana değdi mi (High limit fiyatına ulaştı)
    # +%9 TP emriyle intraday tavan bile kârlı
    d['hit_tavan_intraday'] = d['High'] >= d['limit_up_price'] * tol

    # Tavanda kapandı mı
    d['closed_at_tavan'] = d['Close'] >= d['limit_up_price'] * tol

    # Kilitli tavan: tavanda kapanış + range'in tepesinde
    close_pos = d['close_position'] if 'close_position' in d.columns else 0.5
    d['tavan_locked'] = d['closed_at_tavan'] & (close_pos >= 0.95)

    # Ana flag (geriye uyumlu)
    d['is_tavan'] = d['closed_at_tavan']
    d['is_taban'] = d['Close'] <= limit_down_price * (2 - tol)

    # Streak
    tavan_streak = []
    count = 0
    for val in d['is_tavan']:
        count = count + 1 if val else 0
        tavan_streak.append(count)
    d['tavan_streak'] = tavan_streak

    return d


# ============================================================
# 5) TAVAN DEVAM SKORU (bugün tavan → yarın da tavan?)
#    v3 — Backtest-kalibreli versiyon
#    Backtest bulguları (2y, 548 hisse, 6971 tavan):
#      - Streak en güçlü prediktör: 1.tavan %22 → 4.tavan %59 → 8.tavan %73
#      - Düşük hacim > yüksek hacim (kilitli = devam)
#      - RSI/ADX/Squeeze fark yaratmıyor
#      - Kilitli kapanış önemli
# ============================================================

# Minimum günlük ortalama hacim (TL) — likidite filtresi
MIN_AVG_VOLUME_TL = 500_000  # 500K TL altı = trade edilemez

def tavan_continuation_score(df: pd.DataFrame, yabanci_data: dict = None, ticker: str = None, rs: float = 0.0) -> dict:
    if len(df) < 5:
        return {'score': 0, 'reasons': ['Yetersiz veri']}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    if not last.get('is_tavan', False):
        return {'score': 0, 'reasons': ['Bugün tavan değil']}

    score = 0
    reasons = ["✅ Bugün tavan"]

    # ---- LİKİDİTE FİLTRESİ ----
    avg_vol = last.get('vol_sma_20', 0)
    avg_price = last.get('Close', 0)
    avg_turnover = avg_vol * avg_price if avg_vol and avg_price else 0
    is_liquid = avg_turnover >= MIN_AVG_VOLUME_TL
    
    # ---- STREAK (en güçlü faktör — backtest: %22 → %73) ----
    streak = int(last.get('tavan_streak', 0))
    if streak == 1:
        score += 15
        reasons.append("1️⃣ İlk tavan (devam: ~%22)")
    elif streak == 2:
        score += 25
        reasons.append("2️⃣ 2. ardışık tavan (devam: ~%33)")
    elif streak == 3:
        score += 25
        reasons.append("3️⃣ 3. ardışık tavan (devam: ~%41)")
    elif streak == 4:
        score += 30
        reasons.append(f"4️⃣ 4. ardışık tavan (devam: ~%59)")
    elif streak == 5:
        score += 30
        reasons.append(f"5️⃣ 5. ardışık tavan (devam: ~%60)")
    elif streak >= 6:
        score += 35
        reasons.append(f"🔥 {streak}. ardışık tavan (devam: ~%65-75)")

    # ---- KİLİTLİ KAPANIŞ (backtest: en önemli 2. faktör) ----
    if last.get('tavan_locked', False):
        score += 20
        reasons.append("🔒 Kilitli tavan — close_pos ≥0.95, alıcı kuyruk var")
    else:
        close_pos = last.get('close_position', 0.5)
        if close_pos >= 0.90:
            score += 10
            reasons.append(f"📌 Tavana yakın kapanış (pos: {close_pos:.2f})")
        elif close_pos < 0.80:
            score -= 10
            reasons.append(f"⚠️ Tavandan uzak kapanış (pos: {close_pos:.2f}) — satış baskısı")

    # ---- HACİM (forward test: düşük hacim 1G WR %69.7, ort +3.44%) ----
    vol_ratio = last.get('vol_ratio', 1)
    if vol_ratio < 1.0:
        score += 15
        reasons.append(f"🔒 Düşük hacim ({vol_ratio:.1f}x) — kilitli, satıcı yok")
    elif vol_ratio >= 1.0 and vol_ratio < 2.0:
        score += 5
        reasons.append(f"📊 Normal hacim ({vol_ratio:.1f}x)")
    elif vol_ratio >= 3.0:
        score -= 5
        reasons.append(f"📊 Yüksek hacim ({vol_ratio:.1f}x) — el değiştirme riski")

    # ---- HACİM TRENDİ (forward test: hacim düşüşü 1G WR %84.6, ort +5.73% — EN İYİ sinyal) ----
    # Düşen hacim = satıcı yok, kilitli devam = güçlü sinyal
    if streak >= 2 and prev['Volume'] > 0:
        vol_change = last['Volume'] / prev['Volume']
        if vol_change < 0.5:
            score += 25
            reasons.append("📉 Hacim yarıdan fazla düştü — kilitli, satıcı yok (WR %85)")
        elif vol_change < 0.7:
            score += 20
            reasons.append("📉 Hacim azalıyor — satış baskısı düşük (WR %85)")

    # ---- YENİ HALKA ARZ (IPO) ----
    days = len(df)
    if days <= 15:
        score += 15
        reasons.append(f"🆕 Yeni halka arz ({days} gün) — düşük float")
    elif days <= 30:
        score += 8
        reasons.append(f"🆕 Yakın tarihli halka arz ({days} gün)")

    # ---- KÜÇÜK BONUSLAR (backtest'te marjinal etki) ----
    # Squeeze çıkışı — küçük bonus
    if len(df) >= 3:
        was_sq = df.iloc[-3].get('squeeze_on', False) or df.iloc[-2].get('squeeze_on', False)
        now_sq = last.get('squeeze_on', False)
        if was_sq and not now_sq:
            score += 5
            reasons.append("💥 Squeeze'den çıkış")

    # ---- CMF — sadece ilk tavan (streak 1) için geçerli ----
    # Backtest 2y (N=6383): Streak 1 → CMF>0 %23.5 vs CMF<0 %20.5 (+3pp fark)
    # Streak 2 → fark yok (%31.8 vs %31.4), Streak 3+ → CMF anlamsız
    # CMF sadece ilk tavanda hafif filtre, seri tavanda momentum domine ediyor
    cmf = last.get('CMF_20', 0)
    if cmf is not None and not np.isnan(cmf):
        if streak <= 1:
            if cmf < -0.25:
                score -= 10
                reasons.append(f"💸 Para çıkışı (CMF: {cmf:.2f}) — ilk tavanda dikkat")
            elif cmf < -0.15:
                score -= 5
                reasons.append(f"💸 Hafif para çıkışı (CMF: {cmf:.2f})")
            elif cmf > 0.15:
                score += 5
                reasons.append(f"💰 Para girişi (CMF: {cmf:.2f})")
        else:
            # Streak 2+ → CMF bilgi amaçlı, skor etkisi yok
            if cmf < -0.15:
                reasons.append(f"ℹ️ CMF negatif ({cmf:.2f}) ama seri tavan — momentum baskın")
            elif cmf > 0.15:
                reasons.append(f"💰 CMF pozitif ({cmf:.2f})")

    # ---- GAP ANALİZİ (backtest: gap≥9% → %55.9 devam, gap<-2% → %19.0) ----
    if streak >= 2:
        gap = last.get('gap_pct', 0)
        if gap is not None and not np.isnan(gap):
            if gap >= 0.09:  # Tavandan tavana açılış
                score += 10
                reasons.append(f"🔼 Tavandan açılış (gap: +{gap*100:.1f}%) — çok güçlü")
            elif gap >= 0.02:
                score += 3
                reasons.append(f"🔼 Pozitif açılış (gap: +{gap*100:.1f}%)")
            elif gap < -0.02:
                score -= 10
                reasons.append(f"🔽 Zayıf açılış (gap: {gap*100:.1f}%) — tavan bozma riski")

    # ---- RELATİF GÜÇ (forward test: RS<0 → 1G WR %34.6, RS 50-100% → WR %68.8) ----
    if rs < -10:
        score -= 20
        reasons.append(f"📉 Çok zayıf RS ({rs:+.1f}%) — WR %34.6, piyasanın gerisinde")
    elif rs < 0:
        score -= 10
        reasons.append(f"📉 Negatif RS ({rs:+.1f}%) — WR %34.6")
    elif rs >= 50:
        score += 10
        reasons.append(f"💪 Çok güçlü RS ({rs:+.1f}%)")
    elif rs >= 20:
        score += 5
        reasons.append(f"📈 Güçlü RS ({rs:+.1f}%)")

    # ---- RSI MOMENTUM (forward test: RSI 80-85 → 1G WR %72.2, en iyi band) ----
    rsi = last.get('RSI_14', None)
    if rsi is not None and not np.isnan(rsi):
        if 80 <= rsi < 85:
            score += 5
            reasons.append(f"📈 RSI sweet spot ({rsi:.0f}) — WR %72")

    # ---- YABANCI ORAN ----
    yab_bonus, yab_reason = yabanci_skor_bonusu(yabanci_data, ticker)
    if yab_bonus != 0:
        score += yab_bonus
    if yab_reason:
        reasons.append(yab_reason)

    final_score = max(0, min(100, score))

    return {
        'score': final_score,
        'streak': streak,
        'is_liquid': is_liquid,
        'avg_turnover': avg_turnover,
        'reasons': reasons,
    }


# ============================================================
# 6) PRE-TAVAN SKORU (henüz tavan değil ama yarın olabilir)
#    v3 — Backtest-kalibreli
#    Backtest bulguları: Skor 55+ → %7-9 hit rate
#    En önemli: günlük getiri + kapanış pozisyonu + hacim anomali
#    Az önemli: MACD, BB width, ADX (hepsi gürültü)
# ============================================================

def pre_tavan_score(df: pd.DataFrame, yabanci_data: dict = None, ticker: str = None, rs: float = 0.0) -> dict:
    if len(df) < 20:
        return {'score': 0, 'reasons': ['Yetersiz veri']}

    last = df.iloc[-1]
    if last.get('is_tavan', False):
        return {'score': 0, 'reasons': ['Zaten tavan — continuation_score kullan']}

    score = 0
    reasons = []

    # ---- GÜNLÜK GETİRİ (en güçlü prediktör) ----
    ret = last.get('returns', 0)
    if ret >= 0.08:
        score += 25
        reasons.append(f"🚀 Tavana çok yakın gün (+{ret*100:.1f}%)")
    elif ret >= 0.06:
        score += 18
        reasons.append(f"💪 Çok güçlü gün (+{ret*100:.1f}%)")
    elif ret >= 0.04:
        score += 10
        reasons.append(f"📈 Güçlü gün (+{ret*100:.1f}%)")
    elif ret >= 0.02:
        score += 4
        reasons.append(f"📈 İyi gün (+{ret*100:.1f}%)")

    # ---- KAPANIŞ POZİSYONU ----
    close_pos = last.get('close_position', 0.5)
    if close_pos >= 0.97:
        score += 20
        reasons.append("🔝 Günün en yükseğinde kapanış — momentum devam ediyor")
    elif close_pos >= 0.90:
        score += 12
        reasons.append("📌 Yüksekte kapanış")
    elif close_pos >= 0.80:
        score += 5
        reasons.append("📌 Ortalamanın üstünde kapanış")

    # ---- HACİM ANOMALİSİ ----
    vol_ratio = last.get('vol_ratio', 1)
    if vol_ratio >= 5:
        score += 15
        reasons.append(f"📊 Aşırı hacim anomalisi ({vol_ratio:.1f}x)")
    elif vol_ratio >= 3:
        score += 10
        reasons.append(f"📊 Güçlü hacim ({vol_ratio:.1f}x)")
    elif vol_ratio >= 2:
        score += 5
        reasons.append(f"📊 Ortalamanın üstü hacim ({vol_ratio:.1f}x)")

    # ---- SON GÜNLERDE TAVAN GEÇMİŞİ (yakın zamanda tavan yapmış mı?) ----
    recent_tavans = df['is_tavan'].iloc[-10:].sum() if len(df) >= 10 else 0
    recent_intraday = df['hit_tavan_intraday'].iloc[-10:].sum() if 'hit_tavan_intraday' in df.columns and len(df) >= 10 else 0
    if recent_tavans >= 2:
        score += 12
        reasons.append(f"🔄 Son 10 günde {int(recent_tavans)} tavan kapanış — aktif momentum")
    elif recent_tavans == 1:
        score += 6
        reasons.append("🔄 Son 10 günde 1 tavan kapanış")
    elif recent_intraday >= 1:
        # Gün içi tavana değip çözülmüş — %9 TP ile kârlı olabilir
        score += 4
        reasons.append(f"🔄 Son 10 günde {int(recent_intraday)}x gün içi tavana değdi (kapanışta düştü)")

    # ---- SQUEEZE ÇIKIŞI (küçük bonus) ----
    if len(df) >= 3:
        was_sq = df.iloc[-3].get('squeeze_on', False) or df.iloc[-2].get('squeeze_on', False)
        now_sq = last.get('squeeze_on', False)
        sq_mom = last.get('squeeze_mom', 0)
        if was_sq and not now_sq and sq_mom > 0:
            score += 8
            reasons.append("💥 Squeeze'den yukarı çıkış")

    # ---- YENİ HALKA ARZ ----
    days = len(df)
    if days <= 15:
        score += 12
        reasons.append(f"🆕 Yeni halka arz ({days} gün)")
    elif days <= 30:
        score += 6
        reasons.append(f"🆕 Yakın tarihli halka arz ({days} gün)")

    # ---- 5 GÜNLÜK MOMENTUM ----
    roc5 = last.get('ROC_5', 0)
    if roc5 >= 20:
        score += 8
        reasons.append(f"🔥 5 günlük momentum çok güçlü ({roc5:.0f}%)")

    # ---- RELATİF GÜÇ (kandidat: forward test genel negatif, RS filtre olarak kullan) ----
    if rs < -10:
        score -= 10
        reasons.append(f"📉 Zayıf RS ({rs:+.1f}%)")
    elif rs < 0:
        score -= 5

    # ---- YABANCI ORAN ----
    yab_bonus, yab_reason = yabanci_skor_bonusu(yabanci_data, ticker)
    if yab_bonus != 0:
        score += yab_bonus
    if yab_reason:
        reasons.append(yab_reason)

    return {
        'score': max(0, min(100, score)),
        'reasons': reasons,
    }


# ============================================================
# 7) RELATİF GÜÇ
# ============================================================

def relative_strength(stock_df, index_df, period=20):
    """Hisse vs endeks relatif güç (%). Endeks verisi yoksa None döner."""
    if index_df is None or index_df.empty:
        return None
    if len(stock_df) < period or len(index_df) < period:
        return None
    try:
        # Tarih hizalaması
        common_dates = stock_df.index.intersection(index_df.index)
        if len(common_dates) < period:
            return None
        s_ret = stock_df.loc[common_dates, 'Close'].pct_change(period).iloc[-1]
        i_ret = index_df.loc[common_dates, 'Close'].pct_change(period).iloc[-1]
        if pd.isna(s_ret) or pd.isna(i_ret) or i_ret == 0:
            return None
        return (s_ret - i_ret) * 100
    except Exception as e:
        print(f"  [UYARI] RS hesaplama hatası: {e}")
        return None


# ============================================================
# 8) ANA TARAYICI
# ============================================================

def run_scanner(mode="ex_bist100"):
    """
    Ana tarayıcı.
    mode: 'ex_bist100' → BIST100 dışı (varsayılan, küçük/orta boy)
          'all'        → tüm XTüm
          'bist100'    → sadece BIST100
    """
    tickers = get_tickers(mode)
    print(f"\n📋 Mod: {mode.upper()} — {len(tickers)} hisse taranacak")
    
    # Toplu veri çek
    all_data = fetch_batch(tickers)
    
    # XU100 (relatif güç için)
    xu100 = fetch_data("XU100", "6mo")
    if xu100.empty:
        print("⚠️ XU100 endeks verisi çekilemedi — RS hesaplanamayacak")

    # Yabancı oranları (İsyatırım API)
    yabanci_data = {}
    if "--no-yabanci" not in sys.argv:
        try:
            yabanci_data = fetch_yabanci_oran_bulk(days_back=5)
        except Exception as e:
            print(f"⚠️ Yabancı oran çekilemedi: {e}")
            yabanci_data = _load_yabanci_cache()

    tavan_devam = []
    tavan_kandidat = []
    n_illikit_tavan = 0
    n_illikit_kand = 0

    print("=" * 70)
    print(f"🔍 BIST TAVAN TARAYICI — {mode.upper()} MODU")
    if yabanci_data:
        print(f"🌍 Yabancı oranı: {len(yabanci_data)} hisse yüklendi")
    print("=" * 70)

    for ticker, df in all_data.items():
        df = add_indicators(df)
        df = detect_tavan(df)
        all_data[ticker] = df

        last = df.iloc[-1]
        rs_raw = relative_strength(df, xu100)
        rs = rs_raw if rs_raw is not None else 0.0

        # Likidite kontrolü (veto)
        avg_vol = last.get('vol_sma_20', 0) or 0
        avg_price = last.get('Close', 0) or 0
        is_liquid = (avg_vol * avg_price) >= MIN_AVG_VOLUME_TL

        # Yabancı oran bilgisi
        yab_info = yabanci_data.get(ticker, {})
        yab_oran = yab_info.get('oran', None)
        yab_degisim = yab_info.get('degisim', None)

        # Bugün tavan → devam skoru
        if last.get('is_tavan', False):
            result = tavan_continuation_score(df, yabanci_data=yabanci_data, ticker=ticker, rs=rs)
            result.update({
                'ticker': ticker,
                'rs': rs,
                'rs_available': rs_raw is not None,
                'close': last['Close'],
                'volume': last['Volume'],
                'vol_ratio': last.get('vol_ratio', 0),
                'ema21_dist': round(last.get('ema21_dist', 0), 1),
                'rsi': round(last.get('RSI_14', 0), 1),
                'tavan_locked': bool(last.get('tavan_locked', False)),
                'hit_intraday': bool(last.get('hit_tavan_intraday', False)),
                'yab_oran': yab_oran,
                'yab_degisim': yab_degisim,
            })
            # Likidite vetosu: illikit hisseler ana listeden çıkar
            if is_liquid:
                tavan_devam.append(result)
            else:
                n_illikit_tavan += 1
        else:
            # Pre-tavan skoru
            result = pre_tavan_score(df, yabanci_data=yabanci_data, ticker=ticker, rs=rs)
            if result['score'] >= 35:  # v4: forward test — skor<35 negatif getiri
                result.update({
                    'ticker': ticker,
                    'rs': rs,
                    'rs_available': rs_raw is not None,
                    'close': last['Close'],
                    'returns': last.get('returns', 0),
                    'volume': last['Volume'],
                    'vol_ratio': last.get('vol_ratio', 0),
                    'ema21_dist': round(last.get('ema21_dist', 0), 1),
                    'rsi': round(last.get('RSI_14', 0), 1),
                    'yab_oran': yab_oran,
                    'yab_degisim': yab_degisim,
                })
                # Likidite vetosu
                if is_liquid:
                    tavan_kandidat.append(result)
                else:
                    n_illikit_kand += 1

    # ==================== SONUÇLAR ====================
    
    print("\n" + "=" * 70)
    print("🔴 TAVAN SERİSİ — YARIN DEVAM EDER Mİ?")
    print("=" * 70)

    if tavan_devam:
        tavan_devam.sort(key=lambda x: x['score'], reverse=True)
        for r in tavan_devam:
            emoji = "🟢" if r['score'] >= 60 else "🟡" if r['score'] >= 40 else "🔴"
            lock_str = " 🔒" if r.get('tavan_locked') else ""
            yab_str = ""
            if r.get('yab_oran') is not None:
                yab_str = f"  Yab: %{r['yab_oran']:.1f}"
                if r.get('yab_degisim') is not None:
                    yab_str += f"({r['yab_degisim']:+.2f})"
            ema_str = f"  EMA21: {r.get('ema21_dist', 0):+.1f}%" if r.get('ema21_dist') is not None else ""
            rsi_str = f"  RSI: {r.get('rsi', 0):.0f}" if r.get('rsi') else ""
            rs_str = f"RS: {r['rs']:+.1f}%" if r.get('rs_available') else "RS: N/A"
            print(f"\n{emoji} {r['ticker']:10s}{lock_str}  DEVAM SKORU: {r['score']:3d}/100  "
                  f"Seri: {r['streak']} gün  {rs_str}  "
                  f"Hacim: {r['vol_ratio']:.1f}x  Fiyat: {r['close']:.2f}{ema_str}{rsi_str}{yab_str}")
            for reason in r['reasons']:
                print(f"    {reason}")
    else:
        print("\n  Bugün tavan yapan hisse yok (bu segmentte).")
    if n_illikit_tavan:
        print(f"\n  🚫 {n_illikit_tavan} tavan hisse likidite yetersizliği nedeniyle elendi")

    print("\n" + "=" * 70)
    print("🟡 TAVAN KANDİDATLARI — YARIN TAVAN OLABİLİR (Top 20)")
    print("=" * 70)

    if tavan_kandidat:
        tavan_kandidat.sort(key=lambda x: x['score'], reverse=True)
        for r in tavan_kandidat[:20]:
            emoji = "🟢" if r['score'] >= 60 else "🟡" if r['score'] >= 40 else "⚪"
            ret = r.get('returns', 0)
            ret_str = f"+{ret*100:.1f}%" if ret >= 0 else f"{ret*100:.1f}%"
            rs_str = f"RS: {r['rs']:+.1f}%" if r.get('rs_available') else "RS: N/A"
            print(f"\n{emoji} {r['ticker']:10s}  SKOR: {r['score']:3d}/100  "
                  f"Getiri: {ret_str}  {rs_str}  "
                  f"Hacim: {r['vol_ratio']:.1f}x  Fiyat: {r['close']:.2f}")
            for reason in r['reasons']:
                print(f"    {reason}")
    else:
        print("\n  Eşik üstü kandidat bulunamadı.")
    if n_illikit_kand:
        print(f"\n  🚫 {n_illikit_kand} kandidat likidite yetersizliği nedeniyle elendi")

    # ==================== ÖZET ====================

    print("\n" + "=" * 70)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 Taranan: {len(all_data)} hisse ({mode})")
    print(f"🔴 Tavan serisinde: {len(tavan_devam)}")
    print(f"🟡 Tavan kandidatı (skor≥35): {len(tavan_kandidat)}")
    if n_illikit_tavan or n_illikit_kand:
        print(f"🚫 Likidite vetosu: {n_illikit_tavan} tavan + {n_illikit_kand} kandidat elendi")
    print("=" * 70)

    # CSV kaydet
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    if tavan_devam:
        df_dev = pd.DataFrame(tavan_devam)
        cols = ['ticker', 'score', 'streak', 'rs', 'vol_ratio', 'close', 'ema21_dist', 'rsi', 'yab_oran', 'yab_degisim']
        df_dev = df_dev[[c for c in cols if c in df_dev.columns]]
        fname = f"tavan_devam_{timestamp}.csv"
        df_dev.to_csv(fname, index=False)
        print(f"\n💾 Tavan devam listesi: {fname}")

    if tavan_kandidat:
        df_kand = pd.DataFrame(tavan_kandidat)
        cols = ['ticker', 'score', 'rs', 'vol_ratio', 'returns', 'close', 'ema21_dist', 'rsi', 'yab_oran', 'yab_degisim']
        df_kand = df_kand[[c for c in cols if c in df_kand.columns]]
        fname = f"tavan_kandidat_{timestamp}.csv"
        df_kand.to_csv(fname, index=False)
        print(f"💾 Tavan kandidat listesi: {fname}")

    return {
        'tavan_devam': tavan_devam,
        'tavan_kandidat': tavan_kandidat,
        'data': all_data,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    mode = "all"  # Varsayılan: tüm BIST
    
    if "--all" in sys.argv:
        mode = "all"
    elif "--bist100" in sys.argv:
        mode = "bist100"
    elif "--ex100" in sys.argv or "--ex_bist100" in sys.argv:
        mode = "ex_bist100"
    
    results = run_scanner(mode=mode)
