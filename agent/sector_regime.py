"""
Sektör & Ana Endeks Regime Tespiti — Close-only EMA bazlı AL/PASIF.

İŞ Yatırım IndexHistoricalAll endpoint'i ile günlük Close çekilir
(REST, WebSocket yok → hang riski yok). Endeksler için OHLCV döndürmüyor;
volume=0 olduğu için CMF/OBV zaten anlamsızdı. Close-only logic:
    - close > EMA21 ve EMA21 > EMA55 → AL (trend up)
    - EMA21 son 5 bar slope > 0 → ek puan
    - 20g momentum (close pct) > 0 → ek puan
    trend_score: 0-3, in_trade = trend_score >= 2

Kullanım:
    from agent.sector_regime import load_sector_map, fetch_sector_regimes, \
        fetch_index_regimes, get_ticker_sector_regime
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
    """Close-only AL/PASIF skoru.

    Beklenen kolon: 'close' (lowercase). EMA21/EMA55 bazlı:
        +1: close > EMA21
        +1: EMA21 > EMA55
        +1: EMA21 son 5 bar slope > 0
    in_trade = trend_score >= 2
    """
    if 'close' not in df.columns or len(df) < 60:
        return None

    close = df['close'].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()

    c = float(close.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e55 = float(ema55.iloc[-1])
    e21_5ago = float(ema21.iloc[-6]) if len(ema21) >= 6 else e21
    slope = (e21 - e21_5ago) / max(abs(e21_5ago), 1e-9)

    above_ema21 = c > e21
    ema_bull = e21 > e55
    slope_up = slope > 0

    score = int(above_ema21) + int(ema_bull) + int(slope_up)
    in_trade = score >= 2

    return {
        'trend_score': score,
        'ema_bull': ema_bull,
        'st_bull': above_ema21,  # SuperTrend yerine close>EMA21 (geriye uyum)
        'regime': 1 if in_trade else -1,
        'in_trade': in_trade,
        'regime_label': 'AL' if in_trade else 'PASIF',
    }


_ISY_INDEX_URL = (
    "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/"
    "Common/ChartData.aspx/IndexHistoricalAll"
)


def _fetch_isy_single(code, lookback_days=200, timeout=15):
    """İŞ Yatırım'dan tek endeks Close serisi çek.

    Returns: DataFrame(close=...) veya None.
    """
    import requests
    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    params = {
        'period': 1440,
        'from': start.strftime('%Y%m%d') + '000000',
        'to': end.strftime('%Y%m%d') + '235959',
        'endeks': code,
    }
    try:
        r = requests.get(_ISY_INDEX_URL, params=params, timeout=timeout)
        r.raise_for_status()
        raw = r.json().get('data', [])
    except Exception:
        return None
    if not raw:
        return None

    import pandas as pd
    df = pd.DataFrame(raw, columns=['ts_ms', 'value'])
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.normalize()
    df = df.set_index('date').sort_index()
    return df[['value']].rename(columns={'value': 'close'})


def _fetch_isy_batch(symbols, lookback_days=200, max_workers=8,
                     per_symbol_timeout=15, throttle_sec=0.05):
    """İŞ Yatırım REST ile batch endeks verisi çek.

    Paralel ThreadPoolExecutor — REST çağrısı asılmaz, requests timeout'u
    sert keser. Hata/timeout olan sembol atlanır, diğerleri devam eder.

    Args:
        symbols: ['XBANK', 'XUTEK', 'XU100', ...]
        lookback_days: Geriye kaç gün (default 200, EMA55 + slope için yeterli).
        max_workers: Paralel istek sayısı.
        per_symbol_timeout: requests timeout (saniye).
        throttle_sec: Submit'ler arası küçük gecikme (siteyi yormamak için).

    Returns: {code: DataFrame, ...}
    """
    import concurrent.futures
    import time

    results = {}
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for code in symbols:
            fut = ex.submit(_fetch_isy_single, code, lookback_days,
                            per_symbol_timeout)
            futures[fut] = code
            if throttle_sec:
                time.sleep(throttle_sec)
        for fut in concurrent.futures.as_completed(futures):
            code = futures[fut]
            try:
                df = fut.result(timeout=per_symbol_timeout + 5)
            except Exception:
                df = None
            if df is not None and len(df) >= 60:
                results[code] = df
            else:
                failed.append(code)

    if failed:
        print(f"  ⚠️ {len(failed)}/{len(symbols)} endeks alınamadı: "
              f"{','.join(failed[:8])}{'...' if len(failed)>8 else ''}")

    return results


def fetch_sector_regimes(sector_codes):
    """Sektör endekslerini İŞY'den çek + Close-only AL sinyali hesapla.

    Args:
        sector_codes: {'XBANK', 'XUTEK', ...} unique set

    Returns: {
        'XBANK': {'trend_score': 0, 'in_trade': False, 'regime_label': 'PASIF', ...},
        'XUTEK': {'trend_score': 2, 'in_trade': True, 'regime_label': 'AL', ...},
    }
    """
    dfs = _fetch_isy_batch(list(sector_codes))

    results = {}
    for code, df in dfs.items():
        result = _score_single_df(df)
        if result:
            results[code] = result

    return results


def fetch_index_regimes(codes=None):
    """BIST endekslerini İŞY'den çek + Close-only AL sinyali.

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

    dfs = _fetch_isy_batch(codes)

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
