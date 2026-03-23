"""
Sektör & Ana Endeks Regime Tespiti — RT AL sinyali bazlı soft gate.

tvDatafeed ile gerçek BIST sektör endeksi verisi çekip RT'nin tam
pipeline'ını (scan_regime_transition + compute_trade_state) çalıştırır.
in_trade=True → AL aktif, False → pasif.

Kullanım:
    from agent.sector_regime import load_sector_map, fetch_sector_regimes, \
        fetch_index_regimes, get_ticker_sector_regime

Not: Endekslerde volume=0 olduğu için participation_score her zaman 0 kalır.
RT'nin AL sinyali trend_score + expansion_score + regime geçişine bakar.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SECTOR_MAP_PATH = os.path.join(ROOT, 'tools', 'sector_map.json')

# Tüm BIST endeksleri (şehir endeksleri hariç)
# Gruplar: piyasa, sektör, tematik, katılım
_ALL_INDICES = {
    'piyasa': [
        'XU100', 'XU050', 'XU030', 'XU500', 'XUTUM',
        'XYUZO', 'XTUMY', 'XYLDZ', 'XBANA', 'XLBNK', 'X10XB',
    ],
    'sektor': [
        'XUSIN', 'XGIDA', 'XKMYA', 'XMADN', 'XMANA', 'XMESY', 'XKAGT',
        'XTAST', 'XTEKS', 'XUHIZ', 'XELKT', 'XILTM', 'XINSA', 'XSPOR',
        'XTCRT', 'XTRZM', 'XULAS', 'XUMAL', 'XBANK', 'XSGRT', 'XFINK',
        'XHOLD', 'XGMYO', 'XAKUR', 'XYORT', 'XUTEK', 'XBLSM',
    ],
    'tematik': [
        'XHARZ', 'XKOBI', 'XTMTU', 'XTM25', 'XKURY',
        'XUSRD', 'XSD25', 'XUGRA', 'XT05Y', 'XT10Y',
    ],
    'katilim': [
        'XKTUM', 'XK100', 'XK050', 'XK030', 'XKTMT', 'XSRDK',
    ],
}

# Flat list
_ALL_INDEX_CODES = [c for group in _ALL_INDICES.values() for c in group]

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
    """Tek bir DataFrame için RT tam pipeline — AL sinyali aktif mi?

    scan_regime_transition + compute_trade_state çalıştırır.
    in_trade=True → AL aktif (pozitif), False → AL yok (negatif).
    """
    from markets.bist.regime_transition import scan_regime_transition, compute_trade_state

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

    rt = scan_regime_transition(df, weekly_df=None)
    trade = compute_trade_state(rt['regime'], rt['close'], rt['ema21'])

    regime = int(rt['regime'].iloc[-1])
    in_trade = bool(trade['in_trade'].iloc[-1])
    ts = int(rt['trend_score'].iloc[-1])
    eb = bool(rt['ema_bull'].iloc[-1])
    sb = bool(rt['st_bull'].iloc[-1])

    if in_trade:
        label = 'AL'
    else:
        label = 'PASIF'

    return {
        'trend_score': ts,
        'ema_bull': eb,
        'st_bull': sb,
        'regime': regime,
        'in_trade': in_trade,
        'regime_label': label,
    }


def _fetch_tv_batch(symbols):
    """tvDatafeed ile batch endeks verisi çek.

    Args:
        symbols: ['XBANK', 'XUTEK', 'XU100', ...]

    Returns: {code: DataFrame, ...} — başarılı olanlar
    """
    import time
    from tvDatafeed import TvDatafeed, Interval

    tv = TvDatafeed()
    results = {}
    failed = []

    for code in symbols:
        try:
            df = tv.get_hist(code, 'BIST', interval=Interval.in_daily, n_bars=130)
            if df is not None and len(df) >= 30:
                df = df.drop(columns=['symbol'], errors='ignore')
                results[code] = df
            else:
                failed.append(code)
        except Exception:
            failed.append(code)

    # Retry — connection drop olabilir
    if failed:
        time.sleep(1)
        try:
            tv2 = TvDatafeed()
            for code in failed:
                try:
                    df = tv2.get_hist(code, 'BIST', interval=Interval.in_daily, n_bars=130)
                    if df is not None and len(df) >= 30:
                        df = df.drop(columns=['symbol'], errors='ignore')
                        results[code] = df
                except Exception:
                    continue
        except Exception:
            pass

    return results


def fetch_sector_regimes(sector_codes):
    """Sektör endekslerini tvDatafeed ile çek + RT AL sinyali hesapla.

    Args:
        sector_codes: {'XBANK', 'XUTEK', ...} unique set

    Returns: {
        'XBANK': {'trend_score': 0, 'in_trade': False, 'regime_label': 'PASIF', ...},
        'XUTEK': {'trend_score': 2, 'in_trade': True, 'regime_label': 'AL', ...},
    }
    """
    dfs = _fetch_tv_batch(list(sector_codes))

    results = {}
    for code, df in dfs.items():
        result = _score_single_df(df)
        if result:
            results[code] = result

    return results


def fetch_index_regimes(codes=None):
    """BIST endekslerini tvDatafeed ile çek + RT AL sinyali.

    Args:
        codes: Çekilecek endeks kodları (None → tüm endeksler)

    Returns: {
        'XU100': {'trend_score': 0, 'in_trade': False, 'regime_label': 'PASIF', 'group': 'piyasa'},
        ...
    }
    """
    if codes is None:
        codes = _ALL_INDEX_CODES

    # Grup bilgisi ekle
    code_to_group = {}
    for group, members in _ALL_INDICES.items():
        for c in members:
            code_to_group[c] = group

    dfs = _fetch_tv_batch(codes)

    results = {}
    for code in codes:
        df = dfs.get(code)
        if df is not None:
            result = _score_single_df(df)
            if result:
                result['group'] = code_to_group.get(code, 'diger')
                results[code] = result

    return results


def get_ticker_sector_regime(ticker, sector_regimes, ticker_to_sector):
    """Tek ticker için sektör regime bilgisi döndür.

    in_trade=True → ✅ (AL aktif), False → ⚠️ (pasif).

    Returns: {'sector_index': 'XBANK', 'in_trade': True,
              'regime_label': 'AL', 'badge': '✅XBANK'} veya None
    """
    sector_code = ticker_to_sector.get(ticker)
    if not sector_code:
        return None

    regime = sector_regimes.get(sector_code)
    if not regime:
        return None

    in_trade = regime.get('in_trade', False)
    label = regime['regime_label']

    if in_trade:
        badge = f'✅{sector_code}'
    else:
        badge = f'⚠️{sector_code}↓'

    return {
        'sector_index': sector_code,
        'trend_score': regime['trend_score'],
        'in_trade': in_trade,
        'regime_label': label,
        'badge': badge,
    }
