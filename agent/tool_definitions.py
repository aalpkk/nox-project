"""
NOX Agent — Claude Tool JSON Semalari
8 tool tanimi: scanner, macro, stock, watchlist, confluence, price, kademe, takas
"""

TOOLS = [
    {
        "name": "get_scanner_signals",
        "description": "NOX tarama sonuclarini getir. 6 screener'dan sinyal filtrele: AL/SAT, Rejim v3, Tavan, NOX v3 Weekly, Divergence, Regime Transition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu filtresi (opsiyonel). Orn: 'THYAO' veya 'THYAO.IS'"
                },
                "screener": {
                    "type": "string",
                    "description": "Screener filtresi (opsiyonel). Degerler: alsat, rejim_v3, tavan, nox_v3_weekly, divergence, regime_transition",
                    "enum": ["alsat", "rejim_v3", "tavan", "nox_v3_weekly", "nox_v3_daily", "divergence", "regime_transition"]
                },
                "direction": {
                    "type": "string",
                    "description": "Yon filtresi (opsiyonel)",
                    "enum": ["AL", "SAT"]
                },
                "date": {
                    "type": "string",
                    "description": "Tarih filtresi, YYYYMMDD formatinda (opsiyonel)"
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_overview",
        "description": "Makro piyasa verileri: DXY, USDTRY, EURTRY, XU100, SPY, QQQ, VIX, altin, petrol, bakir, BTC, ETH, US 10Y faiz. Trend, degisim yuzdeleri ve risk rejimi (RISK_ON/RISK_OFF/NOTR).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_analysis",
        "description": "Tek hisse detayli analiz: tum screener sinyalleri + cakisma skoru + teknik durum + badge kontrolu.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu. Orn: 'THYAO' veya 'GARAN'"
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_watchlist",
        "description": "Watchlist CRUD: pozisyonlari listele, ekle, cikar veya guncelle. Portfolyo yonetimi icin.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Islem tipi",
                    "enum": ["list", "add", "remove", "update"]
                },
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (add/remove/update icin zorunlu)"
                },
                "entry_price": {
                    "type": "number",
                    "description": "Giris fiyati (add icin opsiyonel)"
                },
                "stop_price": {
                    "type": "number",
                    "description": "Stop-loss fiyati (add/update icin opsiyonel)"
                },
                "target_price": {
                    "type": "number",
                    "description": "Hedef fiyati (add/update icin opsiyonel)"
                },
                "note": {
                    "type": "string",
                    "description": "Not (add/update icin opsiyonel)"
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_confluence_score",
        "description": "Coklu tarama cakisma puani. 6 screener sonucunu birlestirerek AL/SAT/IZLE tavsiyesi uretir. Kaynak cakisma sayisini raporlar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel -- bossa tum hisseler)"
                },
                "min_score": {
                    "type": "integer",
                    "description": "Minimum skor filtresi (varsayilan: 1)",
                    "default": 1
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_live_price",
        "description": "Guncel fiyat bilgisi (yfinance). Son kapanis, gun ici degisim, hacim. ATR hesaplamasi icin de kullanilir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse/enstruman kodu. BIST icin '.IS' eklenir. Orn: 'THYAO' -> 'THYAO.IS'"
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_kademe",
        "description": "Kademe (emir defteri) analizi. CSV veya Excel dosyasindan S/A oranini hesaplar, destek/direnc seviyelerini bulur, tavan fiyat kontrolu yapar. MatriksIQ scanner CSV ve kullanici Excel formati desteklenir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "CSV veya Excel dosya yolu"
                },
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu filtresi (opsiyonel, scanner CSV icin)"
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "analyze_takas",
        "description": "Takas (araci kurum pozisyon degisimi) analizi. Excel dosyasindan yabanci kurum tespiti, net akis, ivme (acceleration), yon degisimi ve eleme kurallari uygular.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Excel veya CSV dosya yolu"
                },
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel)"
                },
            },
            "required": ["file_path"],
        },
    },
]
