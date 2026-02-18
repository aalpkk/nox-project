"""
NOX Project — Crypto Market Data
Top kripto paralar — yfinance üzerinden USD paritesi.
BTC, ETH + top altcoin'ler.
"""
import pandas as pd
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)


# ── TOP CRYPTO TICKERS (yfinance format: XXX-USD) ──
# Major + large-cap altcoins
_CRYPTO_TICKERS = [
    # Tier 1 — Majors
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    # Tier 2 — Large caps
    "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "ATOM-USD", "UNI-USD", "LTC-USD", "BCH-USD", "NEAR-USD",
    "APT-USD", "FIL-USD", "ARB-USD", "OP-USD", "IMX-USD",
    # Tier 3 — Mid caps (aktif trade edilen)
    "INJ-USD", "SEI-USD", "SUI-USD", "TIA-USD", "JUP-USD",
    "RENDER-USD", "FET-USD", "ONDO-USD", "PENDLE-USD", "STX-USD",
    "RUNE-USD", "AAVE-USD", "MKR-USD", "SNX-USD", "DYDX-USD",
    "TRX-USD", "TON11419-USD", "HBAR-USD", "VET-USD", "ALGO-USD",
    # Tier 4 — Meme / momentum
    "DOGE-USD", "SHIB-USD", "PEPE24478-USD", "WIF-USD", "BONK-USD",
    # Tier 5 — DeFi / Infrastructure
    "CRV-USD", "LDO-USD", "ENS-USD", "GRT-USD", "FXS-USD",
    "EIGEN-USD", "ENA-USD", "W-USD", "ZRO-USD", "STRK-USD",
]


def get_all_crypto_tickers():
    """Kripto ticker listesi — statik (yfinance format)."""
    print(f"✅ Crypto → {len(_CRYPTO_TICKERS)} coin")
    return list(_CRYPTO_TICKERS)


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


def fetch_data(tickers, period="1y"):
    """Batch download kripto verileri."""
    print(f"📡 {len(tickers)} kripto verisi çekiliyor (period={period})...")
    result = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, period=period, group_by='ticker',
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
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

    print(f"✅ {len(result)}/{len(tickers)} coin yüklendi")
    return result


def fetch_benchmark(period="1y"):
    """BTC benchmark verisi (kripto piyasasının lider'i)."""
    try:
        df = yf.download("BTC-USD", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"  [DEBUG] BTC: {len(df)} gün, kolonlar: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"⚠️ BTC benchmark hatası: {e}")
        return None
