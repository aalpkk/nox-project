"""
NOX Agent — Claude Tool JSON Semalari
11 tool tanimi: scanner, macro, stock, watchlist, confluence, price, kademe, takas, mkk, smart_money, ice
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
                    "description": "Screener filtresi (opsiyonel). Degerler: alsat, tavan, nox_v3_weekly, divergence, regime_transition",
                    "enum": ["alsat", "tavan", "nox_v3_weekly", "nox_v3_daily", "divergence", "regime_transition"]
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
        "description": "Kademe (emir defteri) analizi. Dosya verilmezse GitHub Pages'ten VDS verisini otomatik ceker. CSV/Excel dosyasindan S/A oranini hesaplar, destek/direnc seviyelerini bulur, tavan fiyat kontrolu yapar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "CSV veya Excel dosya yolu (opsiyonel — yoksa VDS verisinden auto-fetch)"
                },
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu filtresi (opsiyonel)"
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_takas",
        "description": "Takas (araci kurum pozisyon degisimi) analizi. Dosya verilmezse GitHub Pages'ten VDS verisini otomatik ceker. Excel dosyasindan yabanci kurum tespiti, net akis, ivme ve eleme kurallari uygular.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Excel veya CSV dosya yolu (opsiyonel — yoksa VDS verisinden auto-fetch)"
                },
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel)"
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_mkk_data",
        "description": "MKK (yatirimci dagilimi) verisi — gercek bireysel/kurumsal/yabanci oranlari. VDS scraper ile Matriks IQ'dan cekilir, GitHub Pages'ten sunulur. Takas tahminlerinden cok daha guvenilirdir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel — bossa tum veriler)"
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_smart_money_score",
        "description": "Smart Money Score (SMS) — 5 pillar akilli para analizi. Birikim surekliligi + yogunlasma + karsi taraf zayifligi + sureklilik + MKK rejim teyidi. Skor: >=45 GUCLU, 30-44 ORTA, 15-29 ZAYIF, <15 DAGITIM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel — bossa shortlist icin toplu hesapla)"
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_institutional_confirmation",
        "description": "Kurumsal Teyit Motoru (ICE) — 4 etiketli akilli para analizi. 70 gunluk takas history'den: kurumsal teyit (SM birikimi), tasinan birikim (uzun vadeli sureklilik), maliyet avantaji (Faz 2), kisa vadeli davranis. Carpan: 0.65-1.20.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Hisse kodu (opsiyonel — bossa sinyal listesi icin toplu hesapla)"
                },
            },
            "required": [],
        },
    },
]
