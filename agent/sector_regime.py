"""
Sektör Endeksi Regime Tespiti — RT trend kuralları bazlı soft gate.

sector_map.json'daki sektör endekslerini yfinance ile çekip RT'nin
compute_trend_score fonksiyonuyla trend/choppy sınıflandırması yapar.

yfinance'ta çoğu BIST sektör endeksi yetersiz veri döndürür.
Fallback olarak her sektör için en büyük proxy hissesi kullanılır.

Kullanım:
    from agent.sector_regime import load_sector_map, fetch_sector_regimes, get_ticker_sector_regime

Not: Sadece trend_score kullanılır — endeks hacmi gerçek birikim/dağılım
göstermez, participation skoru güvenilir değil.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SECTOR_MAP_PATH = os.path.join(ROOT, 'tools', 'sector_map.json')

# Sektör endeksi → proxy hisse (en büyük/en likit temsilci)
# yfinance'ta çoğu BIST sektör endeksi yeterli veri dönmüyor,
# bu proxy'ler o sektörün trend yönünü temsil eder.
_SECTOR_PROXY = {
    'XBANK': 'GARAN',
    'XUTEK': 'ASELS',
    'XHOLD': 'SAHOL',
    'XELKT': 'EUPWR',
    'XTRZM': 'THYAO',
    'XUSIN': 'TOASO',
    'XGIDA': 'ULKER',
    'XKMYA': 'SASA',
    'XMANA': 'EREGL',
    'XMESY': 'FROTO',
    'XTEKS': 'BRISA',
    'XTAST': 'AEFES',
    'XINSA': 'ENKAI',
    'XULAS': 'PGSUS',
    'XSGRT': 'ANHYT',
    'XBLSM': 'MGROS',
    'XTCRT': 'BIMAS',
    'XKAGT': 'KARTN',
    'XMADN': 'CEMAS',
    'XGMYO': 'ISGYO',
    'XFINK': 'YKBNK',
    'XSPOR': 'BJKAS',
    'XUMAL': 'AKBNK',
    'XUHIZ': 'TTKOM',
}

# Module-level cache
_CACHE = {'ticker_to_sector': None, 'sector_indexes': None}


def load_sector_map():
    """sector_map.json yükle → (ticker_to_sector dict, sector_indexes dict).

    ticker_to_sector: {'AKBNK': 'XBANK', 'FORTE': 'XUTEK', ...}
    sector_indexes:   {'XBANK': 'XBANK.IS', 'XUTEK': 'XUTEK.IS', ...}

    Cache: module-level _CACHE ile tekrar yükleme engellenir.
    """
    if _CACHE['ticker_to_sector'] is not None:
        return _CACHE['ticker_to_sector'], _CACHE['sector_indexes']

    with open(SECTOR_MAP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sector_indexes = data.get('sector_indexes', {})
    tickers = data.get('tickers', {})

    ticker_to_sector = {}
    for ticker, info in tickers.items():
        si = info.get('sector_index')
        if si:
            ticker_to_sector[ticker] = si

    _CACHE['ticker_to_sector'] = ticker_to_sector
    _CACHE['sector_indexes'] = sector_indexes
    return ticker_to_sector, sector_indexes


def _score_single_df(df):
    """Tek bir DataFrame için trend_score hesapla → dict veya None."""
    from markets.bist.regime_transition import compute_trend_score

    # Kolonları lowercase yap (RT convention)
    col_map = {}
    for col in df.columns:
        cs = str(col).strip().lower()
        if cs in ('close', 'adj close'):
            col_map[col] = 'close'
        elif cs == 'open':
            col_map[col] = 'open'
        elif cs == 'high':
            col_map[col] = 'high'
        elif cs == 'low':
            col_map[col] = 'low'
        elif cs == 'volume':
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)

    if 'close' not in df.columns or len(df) < 30:
        return None

    trend_result = compute_trend_score(df, weekly_df=None)

    ts = int(trend_result['trend_score'].iloc[-1])
    eb = bool(trend_result['ema_bull'].iloc[-1])
    sb = bool(trend_result['st_bull'].iloc[-1])

    if ts >= 2:
        label = 'TREND'
    elif ts == 1:
        label = 'GRI'
    else:
        label = 'CHOPPY'

    return {
        'trend_score': ts,
        'ema_bull': eb,
        'st_bull': sb,
        'regime_label': label,
    }


def fetch_sector_regimes(sector_codes):
    """Sektör endekslerini yfinance ile çek + RT trend_score hesapla.

    İki aşamalı: önce sektör endeksini dene, veri yoksa proxy hisse kullan.

    Args:
        sector_codes: {'XBANK', 'XUTEK', ...} unique set

    Returns: {
        'XBANK': {'trend_score': 2, 'ema_bull': True, 'st_bull': True, 'regime_label': 'TREND'},
        'XUTEK': {'trend_score': 1, 'ema_bull': True, 'st_bull': False, 'regime_label': 'CHOPPY'},
    }
    """
    import yfinance as yf
    import pandas as pd

    _, sector_indexes = load_sector_map()

    results = {}

    # ── Aşama 1: Endeks verisiyle dene ──
    yf_to_code = {}
    yf_tickers = []
    for code in sector_codes:
        yf_sym = sector_indexes.get(code)
        if yf_sym:
            yf_to_code[yf_sym] = code
            yf_tickers.append(yf_sym)

    if yf_tickers:
        try:
            raw = yf.download(" ".join(yf_tickers), period="6mo",
                               progress=False, auto_adjust=True,
                               group_by='ticker', threads=True)

            if not raw.empty:
                for yf_sym, code in yf_to_code.items():
                    try:
                        if len(yf_tickers) == 1:
                            df = raw.copy()
                        elif isinstance(raw.columns, pd.MultiIndex):
                            level_0 = raw.columns.get_level_values(0).unique().tolist()
                            price_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
                            if any(v in price_cols for v in level_0):
                                df = raw.xs(yf_sym, level=1, axis=1).copy()
                            else:
                                df = raw[yf_sym].copy()
                        else:
                            continue

                        df = df.dropna(subset=[c for c in df.columns
                                               if str(c).lower() == 'close'])

                        result = _score_single_df(df)
                        if result:
                            results[code] = result
                    except Exception:
                        continue
        except Exception:
            pass

    # ── Aşama 2: Proxy hisselerle eksikleri tamamla ──
    missing = sector_codes - set(results.keys())
    if missing:
        proxy_tickers = []
        proxy_to_codes = {}  # yf_sym → [code1, code2, ...] (aynı proxy çakışması)
        for code in missing:
            proxy = _SECTOR_PROXY.get(code)
            if proxy:
                yf_sym = f"{proxy}.IS"
                proxy_to_codes.setdefault(yf_sym, []).append(code)
                if yf_sym not in [p for p in proxy_tickers]:
                    proxy_tickers.append(yf_sym)

        if proxy_tickers:
            try:
                raw = yf.download(" ".join(proxy_tickers), period="6mo",
                                   progress=False, auto_adjust=True,
                                   group_by='ticker', threads=True)

                if not raw.empty:
                    for yf_sym, codes in proxy_to_codes.items():
                        try:
                            if len(proxy_tickers) == 1:
                                df = raw.copy()
                            elif isinstance(raw.columns, pd.MultiIndex):
                                level_0 = raw.columns.get_level_values(0).unique().tolist()
                                price_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
                                if any(v in price_cols for v in level_0):
                                    df = raw.xs(yf_sym, level=1, axis=1).copy()
                                else:
                                    df = raw[yf_sym].copy()
                            else:
                                continue

                            df = df.dropna(subset=[c for c in df.columns
                                                   if str(c).lower() == 'close'])

                            result = _score_single_df(df)
                            if result:
                                result['proxy'] = yf_sym.replace('.IS', '')
                                for code in codes:
                                    results[code] = result.copy()
                        except Exception:
                            continue
            except Exception:
                pass

    return results


def get_ticker_sector_regime(ticker, sector_regimes, ticker_to_sector):
    """Tek ticker için sektör regime bilgisi döndür.

    Returns: {'sector_index': 'XBANK', 'trend_score': 2,
              'regime_label': 'TREND', 'badge': '✅XBANK'} veya None
    """
    sector_code = ticker_to_sector.get(ticker)
    if not sector_code:
        return None

    regime = sector_regimes.get(sector_code)
    if not regime:
        return None

    ts = regime['trend_score']
    label = regime['regime_label']

    if ts >= 2:
        badge = f'✅{sector_code}'
    else:
        badge = f'⚠️{sector_code}↓'

    return {
        'sector_index': sector_code,
        'trend_score': ts,
        'regime_label': label,
        'badge': badge,
    }
