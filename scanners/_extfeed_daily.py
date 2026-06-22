"""extfeed günlük OHLCV shim — yfinance yerine nox-project'in TAZE extfeed verisi.

Bu modül, taşınan harici tarayıcıların (bist_rejim_v3_screener / bist_alsat_screener /
bist_tavan_scanner_v2) SADECE veri-çekme katmanını değiştirir. Tarayıcıların indikatör/
sinyal/yayın mantığı dokunulmaz — bu fonksiyonlar yfinance `yf.download` ile BİREBİR aynı
şekilde DataFrame döndürür: kapitalize kolonlar (Open/High/Low/Close/Adj Close/Volume),
DatetimeIndex. Tek fark: kaynak yfinance(.IS) değil, extfeed 1h master → günlük resample.

Neden: yfinance BIST için gecikmeli/bayat (tarayıcılar 06-22 koşmasına rağmen 06-08'de
takılı kaldı). extfeed master master-data-pull zinciriyle taze tutuluyor (06-22'ye uzanır).
"""

import os
import sys

import pandas as pd

# Tarayıcılar scanners/ içinden çağrılır; data.intraday_1h repo kökünde → kökü path'e ekle.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.intraday_1h import MASTER, daily_resample  # noqa: E402

_XU100_DAILY = os.path.join(_REPO_ROOT, "output", "xu100_extfeed_daily.parquet")

# Günlük panel cache (ticker, date, open, high, low, close, volume) — bir kez kurulur.
_DAILY_LONG = None
_PER_TICKER = {}


def _load_daily_long():
    """extfeed 1h master → günlük OHLCV long panel (tek seferlik, cache)."""
    global _DAILY_LONG
    if _DAILY_LONG is None:
        bars = pd.read_parquet(MASTER)
        d = daily_resample(bars)  # cols: ticker, date, open, high, low, close, volume, n_bars
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values(["ticker", "date"])
        _DAILY_LONG = d
    return _DAILY_LONG


def _to_yf_shape(sub: pd.DataFrame) -> pd.DataFrame:
    """Long-panel alt-kümesini yfinance şekline çevir: kapitalize kolon + DatetimeIndex."""
    out = sub.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out["Adj Close"] = out["Close"]
    out = out[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    out.index.name = None
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _period_cut(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """yfinance period semantiği (6mo/1y/2y/max) → son N kadarını döndür."""
    if df.empty or not period or period == "max":
        return df
    p = period.strip().lower()
    last = df.index.max()
    try:
        if p.endswith("mo"):
            cut = last - pd.DateOffset(months=int(p[:-2]))
        elif p.endswith("y"):
            cut = last - pd.DateOffset(years=int(p[:-1]))
        elif p.endswith("d"):
            cut = last - pd.Timedelta(days=int(p[:-1]))
        else:
            return df
    except ValueError:
        return df
    return df[df.index >= cut]


def _ticker_df(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Tek ticker için yfinance-şekilli günlük OHLCV (boşsa boş df)."""
    if ticker in _PER_TICKER:
        df = _PER_TICKER[ticker]
    else:
        d = _load_daily_long()
        sub = d[d["ticker"] == ticker]
        df = _to_yf_shape(sub) if not sub.empty else pd.DataFrame()
        _PER_TICKER[ticker] = df
    return _period_cut(df, period) if not df.empty else df


# ── Tarayıcıların import ettiği isimlerle BİREBİR imzalı drop-in'ler ──────────

def get_all_bist_tickers():
    """extfeed master'da TAZE verisi olan tüm BIST tickerları (yfinance evreni yerine)."""
    d = _load_daily_long()
    tickers = sorted(d["ticker"].astype(str).unique().tolist())
    print(f"✅ extfeed master → {len(tickers)} hisse")
    return tickers


def fetch_data_batch(tickers, period="1y", batch_size=50, min_days=20):
    """hub bist_rejim_v3_screener.fetch_data drop-in: {ticker: yf-şekilli df}."""
    print(f"📡 {len(tickers)} hisse verisi extfeed'den çekiliyor...")
    all_data = {}
    for t in tickers:
        df = _ticker_df(t, period)
        if not df.empty and len(df) >= min_days and "Close" in df.columns:
            all_data[t] = df
    print(f"✅ {len(all_data)}/{len(tickers)} hisse yüklendi (extfeed)\n")
    return all_data


def fetch_data_single(ticker, period="6mo"):
    """tavan bist_tavan_scanner_v2.fetch_data drop-in (tek ticker)."""
    return _ticker_df(ticker, period)


def fetch_batch_tavan(tickers, period="6mo", batch_size=50):
    """tavan bist_tavan_scanner_v2.fetch_batch drop-in: {ticker: df} (len>20)."""
    print(f"📡 {len(tickers)} hisse için veri çekiliyor (extfeed toplu)...")
    all_data = {}
    for t in tickers:
        df = _ticker_df(t, period)
        if not df.empty and len(df) > 20:
            all_data[t] = df
    return all_data


def fetch_benchmark(period="1y"):
    """XU100 endeks günlük OHLCV — xu100_extfeed_daily.parquet (yf XU100.IS yerine)."""
    try:
        xu = pd.read_parquet(_XU100_DAILY)
        # cols: open/high/low/close/volume, DatetimeIndex(Date)
        col_map = {c: c.capitalize() for c in xu.columns}
        xu = xu.rename(columns=col_map)
        if "Adj close" in xu.columns:
            xu = xu.rename(columns={"Adj close": "Adj Close"})
        if "Close" not in xu.columns and "close" in xu.columns:
            xu = xu.rename(columns={"close": "Close"})
        xu.index = pd.to_datetime(xu.index)
        xu.index.name = None
        xu = _period_cut(xu.sort_index(), period)
        print(f"  [DEBUG] XU100 (extfeed): {len(xu)} gün, kolonlar: {list(xu.columns)}")
        return xu
    except Exception as e:
        print(f"  [!] XU100 (extfeed) hata: {e}")
        return None
