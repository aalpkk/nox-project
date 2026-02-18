"""
NOX Project — Commodity Market Data
Emtia verileri — yfinance üzerinden ETF ve futures proxy'leri.
Altın, gümüş, petrol, doğalgaz, buğday, mısır, bakır, vb.
"""
import pandas as pd
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)

# ── COMMODITY TICKERS ──
# yfinance format: futures veya ETF proxy
_COMMODITY_TICKERS = {
    # Değerli metaller
    "GC=F":     "Altın (Gold Futures)",
    "SI=F":     "Gümüş (Silver Futures)",
    "PL=F":     "Platin (Platinum Futures)",
    "PA=F":     "Paladyum (Palladium Futures)",
    # Enerji
    "CL=F":     "Ham Petrol (WTI Crude)",
    "BZ=F":     "Brent Petrol",
    "NG=F":     "Doğalgaz (Natural Gas)",
    "RB=F":     "Benzin (RBOB Gasoline)",
    "HO=F":     "Isıtma Yağı (Heating Oil)",
    # Tarım — Tahıllar
    "ZW=F":     "Buğday (Wheat)",
    "ZC=F":     "Mısır (Corn)",
    "ZS=F":     "Soya (Soybean)",
    "ZM=F":     "Soya Küspesi (Soybean Meal)",
    "ZL=F":     "Soya Yağı (Soybean Oil)",
    "ZO=F":     "Yulaf (Oats)",
    "ZR=F":     "Pirinç (Rice)",
    # Tarım — Softs
    "KC=F":     "Kahve (Coffee)",
    "SB=F":     "Şeker (Sugar)",
    "CC=F":     "Kakao (Cocoa)",
    "CT=F":     "Pamuk (Cotton)",
    "OJ=F":     "Portakal Suyu (Orange Juice)",
    # Endüstriyel metaller
    "HG=F":     "Bakır (Copper)",
    "ALI=F":    "Alüminyum (Aluminum)",
    # Hayvansal
    "LE=F":     "Canlı Sığır (Live Cattle)",
    "HE=F":     "Yağsız Domuz (Lean Hogs)",
    "GF=F":     "Besili Sığır (Feeder Cattle)",
    # ETF Proxy'ler (futures'a erişim yoksa)
    "GLD":      "SPDR Gold ETF",
    "SLV":      "iShares Silver ETF",
    "USO":      "USO Oil ETF",
    "UNG":      "UNG Natural Gas ETF",
    "DBA":      "Invesco Agriculture ETF",
    "DBC":      "Invesco Commodity Index ETF",
    "PDBC":     "Invesco Optimum Yield Diversified",
    "CPER":     "United States Copper ETF",
    "WEAT":     "Teucrium Wheat ETF",
    "CORN":     "Teucrium Corn ETF",
    "SOYB":     "Teucrium Soybean ETF",
    "JO":       "iPath Bloomberg Coffee ETN",
    "NIB":      "iPath Bloomberg Cocoa ETN",
    "BAL":      "iPath Bloomberg Cotton ETN",
    "SGG":      "iPath Bloomberg Sugar ETN",
}

# Kategoriler (raporlama için)
COMMODITY_CATEGORIES = {
    "Değerli Metaller": ["GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV"],
    "Enerji": ["CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "UNG"],
    "Tahıllar": ["ZW=F", "ZC=F", "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F", "WEAT", "CORN", "SOYB"],
    "Softs": ["KC=F", "SB=F", "CC=F", "CT=F", "OJ=F", "JO", "NIB", "BAL", "SGG"],
    "Endüstriyel Metaller": ["HG=F", "ALI=F", "CPER"],
    "Hayvansal": ["LE=F", "HE=F", "GF=F"],
    "Geniş Endeks": ["DBA", "DBC", "PDBC"],
}


def get_all_commodity_tickers():
    """Emtia ticker listesi."""
    tickers = list(_COMMODITY_TICKERS.keys())
    print(f"✅ Commodity → {len(tickers)} enstrüman")
    return tickers


def get_commodity_name(ticker):
    """Ticker'ın insan okunabilir adı."""
    return _COMMODITY_TICKERS.get(ticker, ticker)


def get_commodity_category(ticker):
    """Ticker hangi kategoride."""
    for cat, tickers in COMMODITY_CATEGORIES.items():
        if ticker in tickers:
            return cat
    return "Diğer"


def _normalize_df(df, ticker=None):
    """yfinance MultiIndex → düz DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker:
            try:
                cols = df.columns.get_level_values(0).unique().tolist()
                if 'Price' in cols:
                    sub = df.xs(ticker, level='Ticker', axis=1)
                else:
                    sub = df.xs(ticker, level=1, axis=1)
                sub = sub[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                return sub
            except:
                pass
        try:
            df.columns = df.columns.droplevel(0)
        except:
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            return None
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def fetch_data(tickers, period="2y"):
    """Batch download emtia verileri."""
    print(f"📡 {len(tickers)} emtia verisi çekiliyor (period={period})...")
    result = {}
    # Futures ve ETF'ler ayrı batch'lerde daha stabil
    futures = [t for t in tickers if '=F' in t]
    etfs = [t for t in tickers if '=F' not in t]

    for batch_label, batch_tickers in [("Futures", futures), ("ETF", etfs)]:
        if not batch_tickers:
            continue
        try:
            raw = yf.download(batch_tickers, period=period, group_by='ticker',
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            for t in batch_tickers:
                try:
                    if len(batch_tickers) == 1:
                        sub = _normalize_df(raw)
                    else:
                        sub = _normalize_df(raw, t)
                    if sub is not None and len(sub) >= 60:
                        result[t] = sub
                except:
                    pass
        except Exception as e:
            print(f"  ⚠️ {batch_label} batch hata: {e}")

    print(f"✅ {len(result)}/{len(tickers)} emtia yüklendi")
    return result


def fetch_benchmark(period="2y"):
    """DBC (geniş emtia endeksi) benchmark verisi."""
    try:
        df = yf.download("DBC", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"  [DEBUG] DBC: {len(df)} gün")
        return df
    except Exception as e:
        print(f"⚠️ DBC benchmark hatası: {e}")
        return None
