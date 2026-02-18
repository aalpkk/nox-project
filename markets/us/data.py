"""
NOX Project — US Market Data
S&P 500 + NASDAQ-100 ticker çekme + yfinance veri indirme.
"""
import pandas as pd
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)


# ── STATIC FALLBACKS ──
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_URL = "https://en.wikipedia.org/wiki/NASDAQ-100#Components"


def get_sp500_tickers():
    """S&P 500 ticker listesi — Wikipedia'dan."""
    try:
        tables = pd.read_html(_SP500_URL)
        df = tables[0]
        tickers = sorted(df['Symbol'].str.replace('.', '-', regex=False).tolist())
        print(f"✅ S&P 500 → {len(tickers)} hisse")
        return tickers
    except Exception as e:
        print(f"⚠️ S&P 500 Wikipedia hatası: {e}")
        return _SP500_STATIC


def get_ndx100_tickers():
    """NASDAQ-100 ticker listesi — Wikipedia'dan."""
    try:
        tables = pd.read_html(_NDX100_URL)
        # NASDAQ-100 tablosu genelde 4. veya 5. tablo
        for t in tables:
            if 'Ticker' in t.columns:
                tickers = sorted(t['Ticker'].str.replace('.', '-', regex=False).tolist())
                print(f"✅ NASDAQ-100 → {len(tickers)} hisse")
                return tickers
            if 'Symbol' in t.columns:
                tickers = sorted(t['Symbol'].str.replace('.', '-', regex=False).tolist())
                print(f"✅ NASDAQ-100 → {len(tickers)} hisse")
                return tickers
        print("⚠️ NASDAQ-100 tablosu bulunamadı, fallback kullanılıyor")
        return _NDX100_STATIC
    except Exception as e:
        print(f"⚠️ NASDAQ-100 Wikipedia hatası: {e}, fallback kullanılıyor")
        return _NDX100_STATIC


def get_all_us_tickers():
    """S&P 500 + NASDAQ-100 birleşik liste (deduplicate)."""
    sp = get_sp500_tickers()
    ndx = get_ndx100_tickers()
    combined = sorted(set(sp + ndx))
    print(f"📊 Toplam US: {len(combined)} benzersiz ticker ({len(sp)} S&P + {len(ndx)} NDX)")
    return combined


def _normalize_df(df, ticker=None):
    """yfinance MultiIndex → düz DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        levels = [l for l in df.columns.get_level_values(0).unique() if l != 'Ticker']
        if ticker:
            try:
                if ('Price', ticker) in df.columns or (ticker, 'Close') in df.columns:
                    pass
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


def fetch_data(tickers, period="1y"):
    """Batch download US hisseleri."""
    print(f"📡 {len(tickers)} US hisse verisi çekiliyor (period={period})...")
    result = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, period=period, group_by='ticker',
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                print(f"  [DEBUG] Raw columns type: {type(raw.columns)}")
                print(f"  [DEBUG] Levels: {[l for l in raw.columns.names]}")
            for t in batch:
                try:
                    if len(batch) == 1:
                        sub = _normalize_df(raw)
                    else:
                        sub = _normalize_df(raw, t)
                    if sub is not None and len(sub) >= 60:
                        result[t] = sub
                except:
                    pass
        except Exception as e:
            print(f"  ⚠️ Batch hata: {e}")

    print(f"✅ {len(result)}/{len(tickers)} hisse yüklendi")
    return result


def fetch_benchmark(period="1y"):
    """SPY benchmark verisi."""
    try:
        df = yf.download("SPY", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"  [DEBUG] SPY: {len(df)} gün, kolonlar: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"⚠️ SPY benchmark hatası: {e}")
        return None


# ── STATIC FALLBACKS (top 50 S&P + NDX for offline) ──
_SP500_STATIC = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BLK","BMY","BRK-B","C","CAT","CHTR",
    "CL","CMCSA","COF","COP","COST","CRM","CSCO","CVX","DE","DHR",
    "DIS","DOW","DUK","EMR","F","FDX","GD","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG","JNJ",
    "JPM","KHC","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    "MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX",
    "NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM",
    "RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UNH","UNP","UPS","V","VZ","WBA","WFC","WMT","XOM",
]

_NDX100_STATIC = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","ARM","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP",
    "CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSGP",
    "CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG",
    "FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN",
    "INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR",
    "MCHP","MDB","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU",
    "NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD",
    "PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS",
    "TSLA","TTD","TTWO","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS",
]
