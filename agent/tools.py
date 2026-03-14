"""
NOX Agent — Tool İmplementasyonları
Claude tool use handler: scanner, macro, stock analysis, watchlist, confluence, price,
kademe, takas.
"""
import os
import yfinance as yf
import pandas as pd

from agent.scanner_reader import (
    get_latest_signals, get_signals_for_ticker, get_latest_date_signals,
    summarize_signals, SCREENER_NAMES,
)
from agent.macro import fetch_macro_snapshot, assess_macro_regime, format_macro_summary
from agent.confluence import calc_confluence_score, calc_all_confluence
from agent.state import Watchlist

# Lazy cache — session boyunca bir kez yükle
_signal_cache = {"signals": None, "csv_map": None}
_macro_cache = {"result": None}
_watchlist = None


def _get_signals():
    """Sinyal cache — session'da bir kez yükle."""
    if _signal_cache["signals"] is None:
        signals, csv_map = get_latest_signals()
        _signal_cache["signals"] = signals
        _signal_cache["csv_map"] = csv_map
    return _signal_cache["signals"]


def _get_macro():
    """Makro cache — session'da bir kez yükle."""
    if _macro_cache["result"] is None:
        snapshot = fetch_macro_snapshot()
        _macro_cache["result"] = assess_macro_regime(snapshot)
    return _macro_cache["result"]


def _get_watchlist():
    global _watchlist
    if _watchlist is None:
        _watchlist = Watchlist()
    return _watchlist


def invalidate_cache():
    """Cache'i temizle (yeni veri yüklemek için)."""
    _signal_cache["signals"] = None
    _signal_cache["csv_map"] = None
    _macro_cache["result"] = None


def handle_tool(name, input_data):
    """
    Claude tool use handler.
    name: tool adı
    input_data: tool input dict
    Returns: JSON serializable result
    """
    if name == "get_scanner_signals":
        return _tool_scanner_signals(input_data)
    elif name == "get_macro_overview":
        return _tool_macro_overview(input_data)
    elif name == "get_stock_analysis":
        return _tool_stock_analysis(input_data)
    elif name == "get_watchlist":
        return _tool_watchlist(input_data)
    elif name == "get_confluence_score":
        return _tool_confluence(input_data)
    elif name == "fetch_live_price":
        return _tool_live_price(input_data)
    elif name == "analyze_kademe":
        return _tool_analyze_kademe(input_data)
    elif name == "analyze_takas":
        return _tool_analyze_takas(input_data)
    else:
        return {"error": f"Bilinmeyen tool: {name}"}


# ── Tool implementasyonları ──

def _tool_scanner_signals(input_data):
    """Tarama sinyallerini filtrele ve döndür."""
    signals = _get_signals()
    filtered = list(signals)

    ticker = input_data.get("ticker")
    if ticker:
        filtered = get_signals_for_ticker(filtered, ticker)

    screener = input_data.get("screener")
    if screener:
        filtered = [s for s in filtered if s['screener'] == screener]

    direction = input_data.get("direction")
    if direction:
        filtered = [s for s in filtered if s['direction'] == direction.upper()]

    date = input_data.get("date")
    if date:
        filtered = [s for s in filtered if s.get('csv_date') == date]

    # Sonuç çok büyükse özetle
    if len(filtered) > 50:
        summary = summarize_signals(filtered)
        summary["note"] = f"{len(filtered)} sinyal bulundu, özet gösteriliyor"
        # İlk 20'yi de ekle
        summary["sample"] = filtered[:20]
        return summary

    return {
        "count": len(filtered),
        "signals": filtered,
    }


def _tool_macro_overview(input_data):
    """Makro piyasa özeti."""
    result = _get_macro()
    # Snapshot'ı kompakt formata dönüştür
    compact_snapshot = []
    for s in result.get("snapshot", []):
        compact_snapshot.append({
            "name": s["name"],
            "category": s["category"],
            "price": s["price"],
            "chg_1d": s["chg_1d"],
            "chg_5d": s["chg_5d"],
            "chg_1m": s["chg_1m"],
            "trend": s["trend"],
        })

    return {
        "regime": result["regime"],
        "risk_score": result["risk_score"],
        "signals": result["signals"],
        "instruments": compact_snapshot,
    }


def _tool_stock_analysis(input_data):
    """Tek hisse detaylı analiz."""
    ticker = input_data.get("ticker", "").upper().strip()
    if not ticker:
        return {"error": "Ticker belirtilmedi"}

    signals = _get_signals()
    macro = _get_macro()

    # Çakışma skoru
    confluence = calc_confluence_score(ticker, signals, macro)

    # Güncel fiyat
    price_info = _fetch_price(ticker)

    return {
        "ticker": ticker,
        "confluence": {
            "score": confluence["score"],
            "recommendation": confluence["recommendation"],
            "details": confluence["details"],
        },
        "signals": confluence["signals"],
        "price": price_info,
        "macro_regime": macro["regime"],
    }


def _tool_watchlist(input_data):
    """Watchlist CRUD."""
    wl = _get_watchlist()
    action = input_data.get("action", "list")

    if action == "list":
        positions = wl.list_positions()
        return {"positions": positions, "count": len(positions)}

    elif action == "add":
        ticker = input_data.get("ticker", "")
        if not ticker:
            return {"error": "Ticker belirtilmedi"}
        wl.add_position(
            ticker,
            entry_price=input_data.get("entry_price"),
            stop_price=input_data.get("stop_price"),
            target_price=input_data.get("target_price"),
            note=input_data.get("note", ""),
        )
        return {"success": True, "message": f"{ticker} watchlist'e eklendi"}

    elif action == "remove":
        ticker = input_data.get("ticker", "")
        if not ticker:
            return {"error": "Ticker belirtilmedi"}
        wl.remove_position(ticker)
        return {"success": True, "message": f"{ticker} watchlist'ten çıkarıldı"}

    elif action == "update":
        ticker = input_data.get("ticker", "")
        if not ticker:
            return {"error": "Ticker belirtilmedi"}
        kwargs = {}
        for key in ('entry_price', 'stop_price', 'target_price', 'note'):
            if key in input_data:
                kwargs[key] = input_data[key]
        wl.update_position(ticker, **kwargs)
        return {"success": True, "message": f"{ticker} güncellendi"}

    return {"error": f"Bilinmeyen action: {action}"}


def _tool_confluence(input_data):
    """Çakışma skoru."""
    signals = _get_signals()
    macro = _get_macro()

    ticker = input_data.get("ticker")
    if ticker:
        result = calc_confluence_score(ticker, signals, macro)
        return result

    min_score = input_data.get("min_score", 1)
    results = calc_all_confluence(signals, macro, min_score=min_score)
    return {
        "count": len(results),
        "results": results[:30],  # max 30 sonuç
    }


def _tool_live_price(input_data):
    """Güncel fiyat bilgisi."""
    ticker = input_data.get("ticker", "").upper().strip()
    if not ticker:
        return {"error": "Ticker belirtilmedi"}
    return _fetch_price(ticker)


def _fetch_price(ticker):
    """yfinance ile güncel fiyat çek."""
    ticker_yf = ticker
    # BIST hisseleri için .IS suffix
    if not any(x in ticker for x in ['.', '=', '-', '^']):
        ticker_yf = f"{ticker}.IS"

    try:
        data = yf.download(ticker_yf, period="5d", auto_adjust=True,
                           progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
        if data.empty:
            return {"error": f"{ticker} için veri bulunamadı"}

        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) >= 2 else last
        close = float(last['Close'])
        prev_close = float(prev['Close'])
        change_pct = round(((close / prev_close) - 1) * 100, 2) if prev_close > 0 else 0

        return {
            "ticker": ticker,
            "price": round(close, 4),
            "change_pct": change_pct,
            "volume": int(last.get('Volume', 0)),
            "high": round(float(last.get('High', close)), 4),
            "low": round(float(last.get('Low', close)), 4),
            "date": str(data.index[-1].date()),
        }
    except Exception as e:
        return {"error": f"{ticker} fiyat hatası: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════
# KADEME ANALİZİ
# ══════════════════════════════════════════════════════════════════════

# S/A eşikleri (nox_agent_prompt.md Section 5)
_SA_THRESHOLDS = [
    (0.80, "GUCLU_AL", "Güçlü alıcı baskısı"),
    (1.00, "ALICI_BASKIN", "Alıcı baskın"),
    (1.20, "NOTR", "Nötr"),
    (1.50, "SATICI_BASKIN", "Satıcı baskın"),
    (999, "AGIR_SATIS", "Ağır satış baskısı"),
]


def _sa_karar(sa_ratio):
    """S/A oranına göre karar ver."""
    for threshold, karar, aciklama in _SA_THRESHOLDS:
        if sa_ratio < threshold:
            return karar, aciklama
    return "AGIR_SATIS", "Ağır satış baskısı"


def _tool_analyze_kademe(input_data):
    """Kademe (emir defteri) analizi.
    file_path: CSV/Excel dosya yolu
    ticker: opsiyonel ticker (dosyadan bulunamazsa)
    """
    file_path = input_data.get("file_path", "")
    ticker = input_data.get("ticker", "").upper().strip()

    if not file_path or not os.path.exists(file_path):
        return {"error": f"Dosya bulunamadı: {file_path}"}

    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Dosya okuma hatası: {e}"}

    if df.empty:
        return {"error": "Dosya boş"}

    # İki format destekle:
    # 1. MatriksIQ kademe CSV: ticker,sat_al,sat_al_last,bid_depth,ask_depth,fark,...
    # 2. Kullanıcı Excel: Fiyat,Günlük Lot,%,Alış,Satış,Fark
    cols_lower = [c.lower().strip() for c in df.columns]

    if 'sat_al' in cols_lower:
        return _analyze_kademe_scanner_csv(df, ticker)
    else:
        return _analyze_kademe_excel(df, ticker)


def _analyze_kademe_scanner_csv(df, ticker_filter=""):
    """MatriksIQ kademe scanner CSV analizi."""
    results = []
    for _, row in df.iterrows():
        t = str(row.get('ticker', '')).strip()
        if ticker_filter and t.upper() != ticker_filter:
            continue
        sa = float(row.get('sat_al', 1.0))
        sa_last = float(row.get('sat_al_last', sa))
        bid = float(row.get('bid_depth', 0))
        ask = float(row.get('ask_depth', 0))
        fark = float(row.get('fark', bid - ask))
        karar_orig = str(row.get('karar', '')).strip()
        kilitli = str(row.get('kilitli', 'False')).strip().lower() == 'true'

        karar, aciklama = _sa_karar(sa)

        results.append({
            "ticker": t,
            "sa_ort": round(sa, 3),
            "sa_son": round(sa_last, 3),
            "bid_depth": int(bid),
            "ask_depth": int(ask),
            "fark": int(fark),
            "karar": karar,
            "karar_aciklama": aciklama,
            "karar_orig": karar_orig,
            "kilitli": kilitli,
        })

    if not results:
        return {"error": f"{'Ticker bulunamadı: ' + ticker_filter if ticker_filter else 'Veri yok'}"}

    # Sırala: düşük S/A (en iyi alıcı baskısı) önce
    results.sort(key=lambda x: x['sa_ort'])

    # Özet
    guclu_al = [r for r in results if r['karar'] == 'GUCLU_AL']
    dikkat = [r for r in results if r['sa_ort'] > 1.20]
    kilitli = [r for r in results if r['kilitli']]

    return {
        "format": "scanner_csv",
        "total": len(results),
        "guclu_al_count": len(guclu_al),
        "dikkat_count": len(dikkat),
        "kilitli_count": len(kilitli),
        "results": results if len(results) <= 30 else results[:15] + results[-5:],
    }


def _analyze_kademe_excel(df, ticker=""):
    """Kullanıcı Excel kademe analizi.
    Beklenen kolonlar: Fiyat, Günlük Lot, %, Alış, Satış, Fark
    """
    # Kolon isimlerini normalize et
    col_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if 'fiyat' in cl or 'price' in cl:
            col_map['fiyat'] = c
        elif 'alış' in cl or 'alis' in cl or 'bid' in cl:
            col_map['alis'] = c
        elif 'satış' in cl or 'satis' in cl or 'ask' in cl:
            col_map['satis'] = c
        elif 'fark' in cl or 'diff' in cl or 'net' in cl:
            col_map['fark'] = c
        elif 'lot' in cl or 'hacim' in cl or 'volume' in cl:
            col_map['lot'] = c
        elif 'yüzde' in cl or 'yuzde' in cl or '%' in cl:
            col_map['yuzde'] = c

    if 'alis' not in col_map or 'satis' not in col_map:
        return {"error": f"Alış/Satış kolonları bulunamadı. Mevcut kolonlar: {list(df.columns)}"}

    # Temizle
    for key in ('alis', 'satis', 'fark', 'lot'):
        if key in col_map:
            df[col_map[key]] = pd.to_numeric(
                df[col_map[key]].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                errors='coerce'
            ).fillna(0)

    alis_col = col_map['alis']
    satis_col = col_map['satis']

    toplam_alis = df[alis_col].sum()
    toplam_satis = df[satis_col].sum()
    sa_ratio = round(toplam_satis / toplam_alis, 3) if toplam_alis > 0 else 999

    karar, aciklama = _sa_karar(sa_ratio)

    # Destek/direnç: en yüksek lot yoğunlaşması
    destek_direnc = []
    if 'fiyat' in col_map and 'lot' in col_map:
        fiyat_col = col_map['fiyat']
        lot_col = col_map['lot']
        df[fiyat_col] = pd.to_numeric(
            df[fiyat_col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        sorted_df = df.nlargest(5, lot_col)
        for _, row in sorted_df.iterrows():
            fiyat = float(row[fiyat_col]) if pd.notna(row[fiyat_col]) else 0
            lot = int(row[lot_col]) if pd.notna(row[lot_col]) else 0
            alis = int(row[alis_col]) if pd.notna(row[alis_col]) else 0
            satis = int(row[satis_col]) if pd.notna(row[satis_col]) else 0
            destek_direnc.append({
                "fiyat": fiyat,
                "lot": lot,
                "alis": alis,
                "satis": satis,
                "tip": "destek" if alis > satis else "direnc",
            })

    # Tavan fiyat analizi
    tavan_info = None
    if 'fiyat' in col_map:
        fiyat_col = col_map['fiyat']
        valid_prices = df[fiyat_col].dropna()
        if not valid_prices.empty:
            max_fiyat = float(valid_prices.max())
            tavan_row = df[df[fiyat_col] == max_fiyat].iloc[0] if len(df[df[fiyat_col] == max_fiyat]) > 0 else None
            if tavan_row is not None:
                tavan_alis = int(tavan_row[alis_col]) if pd.notna(tavan_row[alis_col]) else 0
                tavan_satis = int(tavan_row[satis_col]) if pd.notna(tavan_row[satis_col]) else 0
                tavan_info = {
                    "fiyat": max_fiyat,
                    "alis": tavan_alis,
                    "satis": tavan_satis,
                    "kilitli": tavan_satis == 0 and tavan_alis > 0,
                    "yorum": "Kilitli tavan (alıcı kuyruğu)" if tavan_satis == 0 and tavan_alis > 0
                             else "Açık tavan" if tavan_alis > tavan_satis
                             else "Sahte kilit (satıcı dağıtıyor)" if tavan_satis > tavan_alis * 2
                             else "Tavan baskısı altında",
                }

    return {
        "format": "excel",
        "ticker": ticker,
        "toplam_alis": int(toplam_alis),
        "toplam_satis": int(toplam_satis),
        "sa_ratio": sa_ratio,
        "karar": karar,
        "karar_aciklama": aciklama,
        "destek_direnc": destek_direnc,
        "tavan": tavan_info,
        "satir_sayisi": len(df),
    }


# ══════════════════════════════════════════════════════════════════════
# TAKAS ANALİZİ
# ══════════════════════════════════════════════════════════════════════

# Yabancı kurum listesi (nox_agent_prompt.md Section 6)
_YABANCI_KURUMLAR = {
    'deutsche', 'bank of america', 'merrill lynch', 'merrill', 'boa',
    'citibank', 'citi', 'hsbc', 'jp morgan', 'jpmorgan', 'j.p. morgan',
    'goldman sachs', 'goldman', 'ubs', 'morgan stanley',
    'barclays', 'credit suisse', 'bnp paribas', 'bnp',
    'societe generale', 'nomura', 'clsa', 'macquarie', 'rbc',
    'wood & company', 'wood &', 'virtu', 'citadel', 'two sigma',
}

# Emeklilik/yatırım fonu tespiti
_EMEKLILIK_KEYWORDS = {'emeklilik', 'bes', 'pension'}
_YATIRIM_FONU_KEYWORDS = {'yatırım fonu', 'yatirim fonu', 'fon yönetimi',
                           'portföy yönetimi', 'portfoy yonetimi', 'asset management'}
_YATIRIM_ORTAKLIGI_KEYWORDS = {'yatırım ortaklığı', 'yatirim ortakligi', 'holding'}

# Bankalar = perakende proxy (müşteri emri yürütüyorlar)
_BANKA_KEYWORDS = {
    'iş bankası', 'is bankasi', 'isbank',
    'garanti', 'yapı kredi', 'yapi kredi', 'yapıkredi',
    'akbank', 'denizbank', 'halkbank', 'halk bankası', 'vakıfbank', 'vakifbank',
    'ziraat', 'teb', 'qnb', 'şekerbank', 'sekerbank', 'odeabank',
    'icbc', 'ing bank', 'fibabanka', 'alternatifbank',
}


def _tr_lower(s):
    """Türkçe uyumlu lowercase (İ→i, I→ı)."""
    return s.replace('İ', 'i').replace('I', 'ı').lower()


def _classify_kurum(name):
    """Kurum tipini belirle: yabanci/emeklilik/banka/yatirim_fonu/yatirim_ortakligi/yerli"""
    nl = _tr_lower(name)
    # Yabancı kurumlar İngilizce isimli → hem Türkçe hem ASCII lowercase ile kontrol
    nl_ascii = name.lower().replace('ı', 'i')  # AMERICA, CITIBANK gibi isimler için
    if any(k in nl for k in _YABANCI_KURUMLAR) or any(k in nl_ascii for k in _YABANCI_KURUMLAR):
        return 'yabanci'
    if any(k in nl for k in _EMEKLILIK_KEYWORDS):
        return 'emeklilik'
    if any(k in nl for k in _BANKA_KEYWORDS):
        return 'banka'
    if any(k in nl for k in _YATIRIM_FONU_KEYWORDS):
        return 'yatirim_fonu'
    if any(k in nl for k in _YATIRIM_ORTAKLIGI_KEYWORDS):
        return 'yatirim_ortakligi'
    return 'yerli'


def _tool_analyze_takas(input_data):
    """Takas (aracı kurum pozisyon değişimi) analizi.
    file_path: Excel dosya yolu
    ticker: opsiyonel ticker
    """
    file_path = input_data.get("file_path", "")
    ticker = input_data.get("ticker", "").upper().strip()

    if not file_path or not os.path.exists(file_path):
        return {"error": f"Dosya bulunamadı: {file_path}"}

    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Dosya okuma hatası: {e}"}

    if df.empty:
        return {"error": "Dosya boş"}

    # Kolon mapping — Matriks takas Excel formatı (esnek):
    # Tipik: Aracı Kurum | Takas Son | % Son | Lot Fark | % Lot Fark | 1G Fark | 1H Fark | 1A/3A Fark
    # Veya: Aracı Kurum | Takas Son | % Son | Lot Fark | % Lot Fark | Günlük | Haftalık | Aylık/3 Aylık
    col_map = {}
    col_list = list(df.columns)

    for c in col_list:
        cl = str(c).lower().strip()

        # Kurum
        if 'kurum' in cl or 'broker' in cl or 'aracı' in cl or 'araci' in cl:
            col_map['kurum'] = c

        # Pozisyon: Takas Son / Takas İlk
        elif 'takas son' in cl or 'son pozisyon' in cl or cl == 'quantity':
            col_map['takas_son'] = c
        elif 'takas ilk' in cl or 'ilk pozisyon' in cl:
            col_map['takas_ilk'] = c

        # Pay: "% Son" (0-100 arası pay oranı)
        elif '% son' in cl and 'lot' not in cl:
            col_map['pay'] = c

        # % Lot Fark (yüzdesel değişim, pay ile karıştırılmamalı)
        elif '% lot fark' in cl or '% lot_fark' in cl or 'pct_lot' in cl:
            col_map['pct_lot_fark'] = c

        # Lot Fark (mutlak lot değişimi — ana metrik)
        elif ('lot fark' in cl or 'lot_fark' in cl) and '%' not in cl:
            col_map['lot_fark'] = c

        # Periyot farkları — günlük
        elif any(k in cl for k in ('günlük', 'gunluk', 'daily', '1g fark', '1g lot')):
            col_map['gunluk_fark'] = c

        # Periyot farkları — haftalık
        elif any(k in cl for k in ('haftalık', 'haftalik', 'weekly', '1h fark', '1h lot')):
            col_map['haftalik_fark'] = c

        # Periyot farkları — aylık / 3 aylık
        elif any(k in cl for k in ('3 aylık', '3 aylik', '3a fark', '3a lot')):
            col_map['aylik_fark'] = c
        elif any(k in cl for k in ('aylık', 'aylik', 'monthly', '1a fark', '1a lot')):
            if 'aylik_fark' not in col_map:  # 3 aylık öncelikli
                col_map['aylik_fark'] = c

    if 'kurum' not in col_map:
        return {"error": f"Aracı Kurum kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}"}

    kurum_col = col_map['kurum']

    # Sayısal kolonları temizle
    # Lot kolonları: Türkçe format (1.234.567 → 1234567) — nokta binlik ayracı
    # Yüzde kolonları: ondalık nokta (8.38 → 8.38) — nokta kaldırılMAMALI
    lot_keys = ('lot_fark', 'takas_son', 'takas_ilk', 'gunluk_fark', 'haftalik_fark', 'aylik_fark')
    pct_keys = ('pay', 'pct_lot_fark')

    for key in lot_keys:
        if key in col_map:
            df[col_map[key]] = pd.to_numeric(
                df[col_map[key]].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                errors='coerce'
            ).fillna(0)

    for key in pct_keys:
        if key in col_map:
            # Yüzde: virgül → nokta, ama noktayı kaldırma
            df[col_map[key]] = pd.to_numeric(
                df[col_map[key]].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            ).fillna(0)

    # Kurum sınıflandırma
    kurumlar = []
    for _, row in df.iterrows():
        name = str(row[kurum_col]).strip()
        if not name or name == 'nan':
            continue
        tip = _classify_kurum(name)
        entry = {
            "kurum": name,
            "tip": tip,
        }
        if 'lot_fark' in col_map:
            entry['lot_fark'] = int(row[col_map['lot_fark']])
        if 'takas_son' in col_map:
            entry['takas_son'] = int(row[col_map['takas_son']])
        if 'pay' in col_map:
            entry['pay'] = round(float(row[col_map['pay']]), 2)
        if 'gunluk_fark' in col_map:
            entry['gunluk_fark'] = int(row[col_map['gunluk_fark']])
        if 'haftalik_fark' in col_map:
            entry['haftalik_fark'] = int(row[col_map['haftalik_fark']])
        if 'aylik_fark' in col_map:
            entry['aylik_fark'] = int(row[col_map['aylik_fark']])

        # İvme tespiti
        if 'haftalik_fark' in entry and 'aylik_fark' in entry:
            aylik = abs(entry['aylik_fark'])
            haftalik = abs(entry['haftalik_fark'])
            entry['hizlaniyor'] = haftalik > (aylik / 4) if aylik > 0 else False

        # Yön değişimi tespiti
        if 'haftalik_fark' in entry and 'aylik_fark' in entry:
            if entry['aylik_fark'] < 0 and entry['haftalik_fark'] > 0:
                entry['yon_degisimi'] = 'SATISTAN_ALIMA'
            elif entry['aylik_fark'] > 0 and entry['haftalik_fark'] < 0:
                entry['yon_degisimi'] = 'ALIMDAN_SATISA'

        kurumlar.append(entry)

    if not kurumlar:
        return {"error": "Kurum verisi bulunamadı"}

    # Grupla
    yabanci = [k for k in kurumlar if k['tip'] == 'yabanci']
    emeklilik = [k for k in kurumlar if k['tip'] == 'emeklilik']
    banka = [k for k in kurumlar if k['tip'] == 'banka']
    yat_fonu = [k for k in kurumlar if k['tip'] == 'yatirim_fonu']
    yerli = [k for k in kurumlar if k['tip'] == 'yerli']

    # Net yabancı akışı
    net_yabanci = sum(k.get('lot_fark', 0) for k in yabanci)

    # Yabancı alıcı/satıcı sayısı
    yabanci_alici = [k for k in yabanci if k.get('lot_fark', 0) > 0]
    yabanci_satici = [k for k in yabanci if k.get('lot_fark', 0) < 0]

    # Bireysel oran: banka + emeklilik = perakende proxy
    bireysel = banka + emeklilik
    bireysel_toplam_pay = sum(k.get('pay', 0) for k in bireysel)
    bireysel_lot_fark = sum(k.get('lot_fark', 0) for k in bireysel)
    kurumsal_lot_fark = sum(k.get('lot_fark', 0) for k in yabanci + yat_fonu)

    # Uyarılar — ALIŞ tarafına odaklı
    uyarilar = []

    # Yabancı birikim/çıkış
    if len(yabanci_alici) >= 3:
        uyarilar.append("🟢 3+ yabancı birlikte alıyor → ÇOK GÜÇLÜ SİNYAL")
    if len(yabanci_satici) >= 3:
        uyarilar.append("🔴 3+ yabancı birlikte satıyor → ELEME (zaman dilimi eşleştiğinde)")

    # Bireysel oran analizi (pre-tavan takas okuma)
    if bireysel_lot_fark < 0 and kurumsal_lot_fark > 0:
        uyarilar.append("🟢 Bireysel azalıyor + kurumsal alıyor → EN GÜÇLÜ BİRİKİM")
    elif bireysel_lot_fark > 0 and kurumsal_lot_fark < 0:
        uyarilar.append("🔴 Bireysel alıyor + kurumsal satıyor → DAĞITIM")

    # Emeklilik = perakende proxy, geç para
    emeklilik_alici = [k for k in emeklilik if k.get('lot_fark', 0) > 0]
    if emeklilik_alici and not yabanci_alici:
        uyarilar.append("⚠️ Sadece emeklilik alıyor (yabancı yok) → Geç para, trend sonu yakın")
    elif emeklilik_alici and yabanci_alici:
        uyarilar.append("🟡 Emeklilik + yabancı alıyor (emeklilik = geç para, dikkatli ol)")

    # Banka alımı = perakende proxy
    banka_alici = [k for k in banka if k.get('lot_fark', 0) > 0]
    if banka_alici and not yabanci_alici:
        uyarilar.append("⚠️ Banka alımı (perakende proxy) + yabancı yok → DİKKAT")

    # Yat.Fonu boşaltma
    yf_satici = [k for k in yat_fonu if k.get('lot_fark', 0) < -5_000_000]
    if yf_satici:
        names = [k['kurum'] for k in yf_satici]
        uyarilar.append(f"🔴 Yatırım Fonları dev boşaltma: {', '.join(names)}")

    # Dominant pozisyon
    dominant = [k for k in kurumlar if k.get('pay', 0) > 20]
    if dominant:
        for k in dominant:
            uyarilar.append(f"⚠️ Dominant pozisyon: {k['kurum']} %{k['pay']}")

    # Yabancı sıfırdan birikim tespiti
    yabanci_sifirdan = [k for k in yabanci
                        if k.get('lot_fark', 0) > 0
                        and k.get('takas_ilk', 0) == 0]
    if yabanci_sifirdan:
        names = [k['kurum'] for k in yabanci_sifirdan]
        uyarilar.append(f"🟢 Yabancı sıfırdan birikim: {', '.join(names)}")

    # Top alıcı/satıcılar (lot_fark'a göre)
    if any('lot_fark' in k for k in kurumlar):
        sorted_by_fark = sorted(kurumlar, key=lambda x: x.get('lot_fark', 0))
        top_satici = sorted_by_fark[:5]
        top_alici = sorted_by_fark[-5:][::-1]
    else:
        top_alici = []
        top_satici = []

    return {
        "ticker": ticker,
        "kolonlar": col_list,
        "kolon_eslesmesi": {k: v for k, v in col_map.items()},
        "toplam_kurum": len(kurumlar),
        "yabanci_count": len(yabanci),
        "net_yabanci_lot": net_yabanci,
        "yabanci_alici": len(yabanci_alici),
        "yabanci_satici": len(yabanci_satici),
        "emeklilik_count": len(emeklilik),
        "banka_count": len(banka),
        "yatirim_fonu_count": len(yat_fonu),
        "bireysel_pay": round(bireysel_toplam_pay, 2),
        "bireysel_lot_fark": bireysel_lot_fark,
        "kurumsal_lot_fark": kurumsal_lot_fark,
        "uyarilar": uyarilar,
        "top_alici": top_alici[:5],
        "top_satici": top_satici[:5],
        "yabanci_detay": yabanci,
    }
