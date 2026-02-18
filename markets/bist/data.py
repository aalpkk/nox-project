"""
BIST Screener — Veri Çekme
Ticker listesi + yfinance veri + benchmark + USDTRY
"""
import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from core.config import MIN_DATA_DAYS

pd.set_option('future.no_silent_downcasting', True)


# ── HİSSE LİSTESİ ──

def get_all_bist_tickers():
    for path in ['tickers.txt', 'hisseler.txt', 'xtum.txt']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                tickers = [l.strip().upper() for l in f if l.strip()]
            if len(tickers) > 50:
                print(f"✅ {path} → {len(tickers)} hisse")
                return tickers
    tickers = _fetch_from_tradingview()
    if tickers and len(tickers) > 100:
        return tickers
    tickers = _fetch_from_isyatirim()
    if tickers and len(tickers) > 100:
        return tickers
    print("⚠️ Statik liste kullanılıyor")
    return _static_tickers()


def _fetch_from_tradingview():
    try:
        url = "https://scanner.tradingview.com/turkey/scan"
        payload = {
            "columns": ["name"],
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BIST"},
                {"left": "type", "operation": "equal", "right": "stock"},
                {"left": "is_primary", "operation": "equal", "right": True},
            ],
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [0, 1000],
        }
        r = requests.post(url, json=payload, headers={
            'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'
        }, timeout=15)
        if r.status_code == 200:
            tickers = []
            for item in r.json().get('data', []):
                sym = item.get('s', '').split(':')[-1]
                if sym and len(sym) >= 2:
                    tickers.append(sym)
            if tickers:
                print(f"✅ TradingView → {len(tickers)} hisse")
            return tickers
    except Exception as e:
        print(f"  [!] TradingView: {e}")
    return None


def _fetch_from_isyatirim():
    try:
        d = datetime.now()
        for _ in range(10):
            d -= timedelta(days=1)
            if d.weekday() < 5:
                break
        tarih = d.strftime("%d-%m-%Y")
        url = ("https://www.isyatirim.com.tr/_layouts/15/"
               "IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetYabanciOranlarXHR")
        payload = {"baslangicTarih": tarih, "bitisTarihi": tarih,
                   "sektor": None, "endeks": "09", "hisse": None}
        headers = {"Content-Type": "application/json; charset=UTF-8",
                   "X-Requested-With": "XMLHttpRequest", "User-Agent": "Mozilla/5.0"}
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            tickers = [i["HISSE_KODU"] for i in r.json().get("d", []) if i.get("HISSE_KODU")]
            if len(tickers) > 100:
                print(f"✅ İsyatırım → {len(tickers)} hisse")
            return tickers
    except:
        pass
    return None


def _static_tickers():
    return [
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
    ]


# ── VERİ ÇEKME ──

def _normalize_df(raw, t, yf_t, yf_syms):
    """yfinance raw output'tan tek hisse DataFrame çıkar."""
    try:
        if len(yf_syms) == 1:
            df = raw.copy()
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                level_vals_0 = raw.columns.get_level_values(0).unique().tolist()
                level_vals_1 = raw.columns.get_level_values(1).unique().tolist()
                price_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                if any(v in price_cols for v in level_vals_0):
                    if yf_t in level_vals_1:
                        df = raw.xs(yf_t, level=1, axis=1).copy()
                    elif t in level_vals_1:
                        df = raw.xs(t, level=1, axis=1).copy()
                    else:
                        return None
                else:
                    if yf_t in level_vals_0:
                        df = raw[yf_t].copy()
                    elif t in level_vals_0:
                        df = raw[t].copy()
                    else:
                        return None
            else:
                return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        col_map = {}
        for col in df.columns:
            col_str = str(col).strip()
            if col_str.lower() in ('close', 'adj close'):
                col_map[col] = 'Close'
            elif col_str.lower() == 'open':
                col_map[col] = 'Open'
            elif col_str.lower() == 'high':
                col_map[col] = 'High'
            elif col_str.lower() == 'low':
                col_map[col] = 'Low'
            elif col_str.lower() == 'volume':
                col_map[col] = 'Volume'
        if col_map:
            df = df.rename(columns=col_map)

        df = df.dropna(how='all')
        if not df.empty and len(df) >= MIN_DATA_DAYS and 'Close' in df.columns:
            return df
    except:
        pass
    return None


def fetch_data(tickers, period="1y", batch_size=50):
    print(f"📡 {len(tickers)} hisse verisi çekiliyor (period={period})...")
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        yf_syms = [f"{t}.IS" for t in batch]
        try:
            raw = yf.download(" ".join(yf_syms), period=period,
                              progress=False, auto_adjust=True,
                              group_by='ticker', threads=True)
            if raw.empty:
                continue
            if i == 0:
                print(f"  [DEBUG] Raw columns type: {type(raw.columns)}")
                if isinstance(raw.columns, pd.MultiIndex):
                    print(f"  [DEBUG] Levels: {raw.columns.names}")

            for t, yf_t in zip(batch, yf_syms):
                df = _normalize_df(raw, t, yf_t, yf_syms)
                if df is not None:
                    all_data[t] = df
        except Exception as e:
            print(f"  [!] Batch hata: {e}")
        if i + batch_size < len(tickers):
            time.sleep(1)
    print(f"✅ {len(all_data)}/{len(tickers)} hisse yüklendi\n")
    return all_data


def _normalize_index_df(xu):
    """Benchmark / USDTRY DataFrame normalize."""
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    col_map = {}
    for col in xu.columns:
        cs = str(col).strip().lower()
        if cs in ('close', 'adj close'):
            col_map[col] = 'Close'
        elif cs == 'open':
            col_map[col] = 'Open'
        elif cs == 'high':
            col_map[col] = 'High'
        elif cs == 'low':
            col_map[col] = 'Low'
        elif cs == 'volume':
            col_map[col] = 'Volume'
    if col_map:
        xu = xu.rename(columns=col_map)
    return xu


def fetch_benchmark(period="1y"):
    try:
        xu = yf.download("XU100.IS", period=period, progress=False, auto_adjust=True)
        xu = _normalize_index_df(xu)
        print(f"  [DEBUG] XU100: {len(xu)} gün, kolonlar: {list(xu.columns)}")
        return xu
    except Exception as e:
        print(f"  [!] XU100 hata: {e}")
        return None


def fetch_usdtry(period="5y"):
    try:
        usd = yf.download("USDTRY=X", period=period, progress=False, auto_adjust=True)
        usd = _normalize_index_df(usd)
        print(f"  [DEBUG] USDTRY: {len(usd)} gün, son: {usd['Close'].iloc[-1]:.2f}")
        return usd
    except Exception as e:
        print(f"  [!] USDTRY hata: {e}")
        return None
