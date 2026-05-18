"""
ML Feature Hesaplama — Screener Primitive Feature Engine
Her feature, üretim screener'larından (regime.py, nox_v3_signals.py,
regime_transition.py, tavan, reversal_v2.py) birebir extract edilmiştir.
Uppercase OHLCV konvansiyonu (Close, High, Low, Open, Volume).
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════
# TEMEL İNDİKATÖR YARDIMCILARI
# ═══════════════════════════════════════════

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _sma(series, period):
    return series.rolling(period).mean()

def _rma(series, period):
    """Pine Script ta.rma (Wilder's smoothing)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()

def _true_range(df):
    h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def _calc_atr(df, period=14):
    return _rma(_true_range(df), period)

def _calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _calc_adx_with_di(df, length=14):
    """ADX + DI hesaplama. Returns (adx, plus_di, minus_di)."""
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    atr = _rma(tr, length)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=df.index), length) / atr
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, length)
    return adx, plus_di, minus_di

def _calc_supertrend(df, period=10, mult=3.0):
    """SuperTrend direction: 1=up, -1=down."""
    atr = _calc_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    st_dir = pd.Series(1, index=df.index)
    final_up = up.copy()
    final_dn = dn.copy()
    close = df['Close']
    for i in range(1, len(df)):
        if up.iloc[i] > final_up.iloc[i-1]:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = final_up.iloc[i-1] if close.iloc[i-1] > final_up.iloc[i-1] else up.iloc[i]
        if dn.iloc[i] < final_dn.iloc[i-1]:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = final_dn.iloc[i-1] if close.iloc[i-1] < final_dn.iloc[i-1] else dn.iloc[i]
        prev_dir = st_dir.iloc[i-1]
        if prev_dir == -1 and close.iloc[i] > final_dn.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif prev_dir == 1 and close.iloc[i] < final_up.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = prev_dir
    return st_dir

def _calc_wavetrend(df):
    """WaveTrend WT1/WT2. Returns (wt1, wt2)."""
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    esa = _ema(hlc3, 10)
    d = _ema((hlc3 - esa).abs(), 10)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, 21)
    wt2 = _sma(wt1, 4)
    return wt1, wt2

def _calc_pmax(df):
    """PMAX direction: 1=long, -1=short."""
    src = (df['High'] + df['Low']) / 2
    ma_val = _ema(src, 10)
    atr = _calc_atr(df, 10)
    n = len(df)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    pmax_dir = np.ones(n)
    for i in range(1, n):
        ls = ma_val.iloc[i] - 3.0 * atr.iloc[i]
        ss = ma_val.iloc[i] + 3.0 * atr.iloc[i]
        long_stop[i] = max(ls, long_stop[i-1]) if not np.isnan(long_stop[i-1]) and ma_val.iloc[i] > long_stop[i-1] else ls
        short_stop[i] = min(ss, short_stop[i-1]) if not np.isnan(short_stop[i-1]) and ma_val.iloc[i] < short_stop[i-1] else ss
        prev = pmax_dir[i-1]
        if prev == -1 and ma_val.iloc[i] > short_stop[i-1]:
            pmax_dir[i] = 1
        elif prev == 1 and ma_val.iloc[i] < long_stop[i-1]:
            pmax_dir[i] = -1
        else:
            pmax_dir[i] = prev
    return pd.Series(pmax_dir, index=df.index)

def _calc_smc(df):
    """SMC BOS/CHoCH. Returns (swing_bias_arr, bos_bar_prop, choch_bar_prop)."""
    n = len(df)
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    k = 5  # SMC_INTERNAL_LEN
    last_sh = np.nan
    last_sl = np.nan
    swing_bias = 0
    swing_bias_arr = np.zeros(n, dtype=int)
    bos_bar = np.full(n, -999)
    choch_bar = np.full(n, -999)
    for i in range(k, n):
        pi = i - k
        if pi >= k:
            is_ph = h[pi] == np.max(h[pi-k:pi+k+1])
            is_pl = l[pi] == np.min(l[pi-k:pi+k+1])
            if is_ph:
                last_sh = h[pi]
            if is_pl:
                last_sl = l[pi]
        if not np.isnan(last_sh) and i > 0:
            if c[i] > last_sh and c[i-1] <= last_sh:
                if swing_bias <= 0:
                    choch_bar[i] = i
                bos_bar[i] = i
                swing_bias = 1
        if not np.isnan(last_sl) and i > 0:
            if c[i] < last_sl and c[i-1] >= last_sl:
                swing_bias = -1
        swing_bias_arr[i] = swing_bias
    last_bos = -999
    last_choch = -999
    bos_prop = np.full(n, -999)
    choch_prop = np.full(n, -999)
    for i in range(n):
        if bos_bar[i] != -999:
            last_bos = bos_bar[i]
        if choch_bar[i] != -999:
            last_choch = choch_bar[i]
        bos_prop[i] = last_bos
        choch_prop[i] = last_choch
    return swing_bias_arr, bos_prop, choch_prop

def _calc_cmf(df, period=20):
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
    hl_range = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / hl_range
    clv = clv.fillna(0)
    mfv = clv * v
    return mfv.rolling(period).sum() / v.rolling(period).sum()

def _calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def _calc_mfi(df, period=14):
    """Money Flow Index."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)

def _calc_obv(df):
    sign = np.sign(df['Close'].diff()).fillna(0)
    return (sign * df['Volume']).cumsum()

def _linreg_slope(series, length):
    """Rolling linear regression slope."""
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    def _slope(y):
        if len(y) < length:
            return np.nan
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / x_var
    return series.rolling(length).apply(_slope, raw=True)

def _consecutive_count(mask):
    """Boolean mask → consecutive True count."""
    result = mask.astype(int).copy()
    for i in range(1, len(result)):
        if result.iloc[i]:
            result.iloc[i] = result.iloc[i-1] + 1
    return result

def _find_pivot_lows(low, lb=5):
    """Pivot low detection (Pine ta.pivotlow replica)."""
    n = len(low)
    result = pd.Series(np.nan, index=low.index)
    vals = low.values.astype(float)
    for i in range(2 * lb, n):
        mid = i - lb
        window = vals[i - 2 * lb: i + 1]
        if vals[mid] == np.nanmin(window):
            result.iloc[i] = vals[mid]
    return result

def _find_pivot_highs(high, lb=5):
    """Pivot high detection (Pine ta.pivothigh replica)."""
    n = len(high)
    result = pd.Series(np.nan, index=high.index)
    vals = high.values.astype(float)
    for i in range(2 * lb, n):
        mid = i - lb
        window = vals[i - 2 * lb: i + 1]
        if vals[mid] == np.nanmax(window):
            result.iloc[i] = vals[mid]
    return result


# ═══════════════════════════════════════════
# ANA FEATURE HESAPLAMA
# ═══════════════════════════════════════════

_OHLCV_CANONICAL = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}


def _normalize_ohlcv_columns(d):
    """OHLCV kolon adlarını case-insensitive canonical (Title) forma çevir.

    Open/open/OPEN → Open. Diğer kolonları aynı bırakır. None geçirilirse None döner.
    Idempotent: zaten Title case ise mapping boştur, .rename no-op olur.
    """
    if d is None:
        return None
    mapping = {}
    for col in d.columns:
        cl = str(col).lower()
        if cl in _OHLCV_CANONICAL:
            target = _OHLCV_CANONICAL[cl]
            if col != target:
                mapping[col] = target
    return d.rename(columns=mapping) if mapping else d


def compute_all_features(df, xu_df=None, weekly_df=None):
    """
    Tek bir hisse için tüm screener primitive feature'ları hesapla.

    Args:
        df: DataFrame — OHLCV (Open/High/Low/Close/Volume, case-insensitive)
        xu_df: XU100 DataFrame — relatif güç için (kolonlar case-insensitive)
        weekly_df: Haftalık resample — HTF feature'lar için (opsiyonel, case-insensitive)

    Returns:
        DataFrame — aynı index, her kolon bir feature
    """
    df = _normalize_ohlcv_columns(df)
    xu_df = _normalize_ohlcv_columns(xu_df)
    weekly_df = _normalize_ohlcv_columns(weekly_df)

    n = len(df)
    if n < 80:
        return pd.DataFrame(index=df.index)

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume']

    # Dict'e topla, sonunda tek seferde DataFrame oluştur (fragmentation önlemi)
    _c = {}

    # ═══════════════════════════════════════
    # A. FİYAT YAPISI (8)
    # ═══════════════════════════════════════
    _c['close'] = c
    _c['returns_1d'] = c.pct_change() * 100
    _c['returns_5d'] = c.pct_change(5) * 100
    _c['returns_10d'] = c.pct_change(10) * 100
    hl_range = (h - l).replace(0, np.nan)
    _c['close_position'] = (c - l) / hl_range  # Candle location (0-1)
    _c['gap_pct'] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100
    ema21 = _ema(c, 21)
    ema55 = _ema(c, 55)
    _c['ema21_dist_pct'] = (c / ema21.replace(0, np.nan) - 1) * 100
    _c['ema55_dist_pct'] = (c / ema55.replace(0, np.nan) - 1) * 100

    # ═══════════════════════════════════════
    # B. TREND (12)
    # ═══════════════════════════════════════
    adx, plus_di, minus_di = _calc_adx_with_di(df, 14)
    _c['adx_14'] = adx
    adx_slope = (adx - adx.shift(5)) / 5
    _c['adx_slope_5'] = adx_slope
    _c['plus_di'] = plus_di
    _c['minus_di'] = minus_di
    _c['di_spread'] = plus_di - minus_di
    _c['ema_trend_up'] = (ema21 > ema55).astype(int)
    st_dir = _calc_supertrend(df, 10, 3.0)
    _c['supertrend_dir'] = st_dir
    pmax_dir = _calc_pmax(df)
    _c['pmax_dir'] = pmax_dir
    _c['phase_above_ema21'] = (c > ema21).astype(int)

    # HTF features
    if weekly_df is None and n >= 60:
        weekly_df = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()

    if weekly_df is not None and len(weekly_df) >= 20:
        w_adx, _, _ = _calc_adx_with_di(weekly_df, 14)
        w_adx_slope = (w_adx - w_adx.shift(5)) / 5
        w_ema21 = _ema(weekly_df['Close'], 21)
        w_ema55 = _ema(weekly_df['Close'], 55)
        w_trend_up = (w_ema21 > w_ema55).astype(int)
        _c['htf_adx'] = w_adx.reindex(df.index, method='ffill')
        _c['htf_adx_slope'] = w_adx_slope.reindex(df.index, method='ffill')
        _c['htf_trend_up'] = w_trend_up.reindex(df.index, method='ffill')
    else:
        _c['htf_adx'] = np.nan
        _c['htf_adx_slope'] = np.nan
        _c['htf_trend_up'] = np.nan

    # ═══════════════════════════════════════
    # C. MOMENTUM (10)
    # ═══════════════════════════════════════
    rsi14 = _calc_rsi(c, 14)
    _c['rsi_14'] = rsi14
    _c['rsi_2'] = _calc_rsi(c, 2)
    macd_line, macd_signal, macd_hist = _calc_macd(c)
    _c['macd_line'] = macd_line
    _c['macd_signal'] = macd_signal
    _c['macd_hist'] = macd_hist
    wt1, wt2 = _calc_wavetrend(df)
    _c['wt1'] = wt1
    _c['wt2'] = wt2
    _c['wt_bullish'] = (wt1 > wt2).astype(int)

    # Squeeze
    bb_mid = _sma(c, 20)
    bb_dev = c.rolling(20).std() * 2.0
    bb_upper = bb_mid + bb_dev
    bb_lower = bb_mid - bb_dev
    atr_val = _calc_atr(df, 14)
    kc_mid = _ema(c, 20)
    kc_range = _rma(_true_range(df), 20) * 1.5
    kc_upper = kc_mid + kc_range
    kc_lower = kc_mid - kc_range
    _c['squeeze_on'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
    highest = h.rolling(20).max()
    lowest = l.rolling(20).min()
    sq_mid = (highest + lowest) / 2
    _c['squeeze_mom'] = c - (sq_mid + bb_mid) / 2

    # ═══════════════════════════════════════
    # D. VOLATİLİTE (6) — cross-stock comparable only
    # ═══════════════════════════════════════
    _c['atr_14'] = atr_val
    _c['atr_pct'] = atr_val / c.replace(0, np.nan) * 100
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    _c['bb_pctb'] = (c - bb_lower) / bb_range
    _c['bb_width'] = bb_range / bb_mid.replace(0, np.nan) * 100
    high_20 = h.rolling(20).max()
    _c['drawdown_20'] = (c - high_20) / high_20.replace(0, np.nan) * 100
    _c['daily_move_atr'] = (c - o) / atr_val.replace(0, np.nan)

    # ═══════════════════════════════════════
    # E. HACİM (7)
    # ═══════════════════════════════════════
    vol_sma20 = _sma(v, 20)
    vol_sma30 = _sma(v, 30)
    _c['vol_ratio_20'] = v / vol_sma20.replace(0, np.nan)
    _c['vol_ratio_30'] = v / vol_sma30.replace(0, np.nan)
    cmf20 = _calc_cmf(df, 20)
    _c['cmf_20'] = cmf20
    obv = _calc_obv(df)
    obv_sma20 = _sma(obv, 20)
    _c['obv_trend'] = (obv > obv_sma20).astype(int)
    obv_ema10 = _ema(obv, 10)
    _c['obv_slope_5'] = _linreg_slope(obv_ema10, 5)
    _c['mfi_14'] = _calc_mfi(df, 14)
    green_mask = c > o
    _c['consecutive_green'] = _consecutive_count(green_mask)

    # ═══════════════════════════════════════
    # F. YAPI / SMC (6)
    # ═══════════════════════════════════════
    swing_bias_arr, bos_prop, choch_prop = _calc_smc(df)
    _c['swing_bias'] = swing_bias_arr
    idx_arr = np.arange(n)
    bos_age = np.where(bos_prop == -999, np.nan, idx_arr - bos_prop)
    choch_age = np.where(choch_prop == -999, np.nan, idx_arr - choch_prop)
    _c['bos_age'] = bos_age
    _c['choch_age'] = choch_age

    pivot_highs = _find_pivot_highs(h, 5)
    last_pivot_high = pivot_highs.ffill()
    _c['structure_break'] = (c > last_pivot_high * 1.002).astype(int)

    pivot_lows = _find_pivot_lows(l, 5)
    last_pl = pivot_lows.ffill()
    prev_pl = pivot_lows.ffill().shift(1)
    _c['higher_low'] = (last_pl > prev_pl).astype(int)

    high_40 = h.rolling(40).max()
    _c['near_40high'] = (c > high_40 * 0.97).astype(int)

    # ═══════════════════════════════════════
    # G. AL/SAT Q SCORE PRİMİTİVLERİ (5) — regime.py exact
    # ═══════════════════════════════════════
    # rvol
    rvol = v / vol_sma20.replace(0, np.nan)

    # clv: candle location value (how close the close is to the high)
    candle_range = (h - l).replace(0, np.nan)
    clv = (c - l) / candle_range
    clv = clv.fillna(0.5)

    # wick: upper wick rejection ratio
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    wick_ratio = upper_wick / candle_range
    wick_ratio = wick_ratio.fillna(0)

    # range_atr: candle range normalized by ATR
    range_atr = candle_range / atr_val.replace(0, np.nan)

    # Scoring (exact thresholds from markets/bist/regime.py lines 215-218)
    _c['q_rvol_s'] = np.where(rvol >= 2, 25,
                        np.where(rvol >= 0.5, 20,   # RVOL_THRESH = 0.5
                        np.where(rvol >= 1, 10, 0))).astype(float)
    # Fix: RVOL_THRESH=0.5 is lower than 1.0, so order matters
    # regime.py: 25 if >=2, 20 if >=RVOL_THRESH(0.5), 10 if >=1, else 0
    # Since 0.5<1.0, the check >=RVOL_THRESH catches 0.5-0.99 range at 20,
    # but >=1 should be 10?  Re-reading: the chain is if-elif, so:
    # rvol>=2→25, elif rvol>=0.5→20, elif rvol>=1→10 (never reached)
    # So effectively: >=2→25, >=0.5→20, <0.5→0
    _c['q_rvol_s'] = np.where(rvol >= 2, 25,
                        np.where(rvol >= 0.5, 20, 0)).astype(float)

    _c['q_clv_s'] = np.where(clv >= 0.75, 25,
                       np.where(clv >= 0.5, 15,
                       np.where(clv >= 0.25, 5, 0))).astype(float)

    _c['q_wick_s'] = np.where(wick_ratio <= 0.15, 25,
                        np.where(wick_ratio <= 0.3, 15,
                        np.where(wick_ratio <= 0.5, 5, 0))).astype(float)

    _c['q_range_s'] = np.where(range_atr >= 1.2, 25,
                         np.where(range_atr >= 0.8, 15,
                         np.where(range_atr >= 0.5, 5, 0))).astype(float)

    _c['q_total'] = _c['q_rvol_s'] + _c['q_clv_s'] + _c['q_wick_s'] + _c['q_range_s']

    # ═══════════════════════════════════════
    # H. NW BREADTH PRİMİTİVLERİ (5) — nox_v3_signals.py exact
    # ═══════════════════════════════════════
    rsi_min10 = rsi14.rolling(10).min()

    # Sub-component 1: RSI thrust (50pt)
    br_rsi_thrust = ((rsi_min10 < 30) & (rsi14 > 55)).astype(int)
    _c['br_rsi_thrust'] = br_rsi_thrust

    # Sub-component 2: RSI gradual (25pt)
    rsi_delta5 = rsi14 - rsi14.shift(5)
    br_rsi_gradual = ((rsi_delta5 > 15) & (rsi14 > 40) & (rsi_min10 < 35)).astype(int)
    _c['br_rsi_gradual'] = br_rsi_gradual

    # Sub-component 3: AD proxy (25pt)
    consec_green = _consecutive_count(green_mask)
    br_ad_proxy = ((consec_green >= 3) & (v > vol_sma20 * 1.3)).astype(int)
    _c['br_ad_proxy'] = br_ad_proxy

    # Sub-component 4: EMA reclaim (15pt)
    br_ema_reclaim = ((c > ema21) & (c.shift(1) < ema21.shift(1))).astype(int)
    _c['br_ema_reclaim'] = br_ema_reclaim

    # Composite breadth score
    br_score = (br_rsi_thrust * 50 + br_rsi_gradual * 25 +
                br_ad_proxy * 25 + br_ema_reclaim * 15).clip(0, 100).astype(float)
    _c['br_score'] = br_score

    # ═══════════════════════════════════════
    # I. NW REGIME SCORE PRİMİTİVLERİ (6) — nox_v3_signals.py exact
    # ═══════════════════════════════════════
    atr_pct = atr_val / c.replace(0, np.nan) * 100
    di_diff_norm = (plus_di - minus_di) / (1 + atr_pct * 0.1)
    was_trending = (adx.rolling(20).max() > 25).astype(float)

    rg_slope_score = adx_slope.clip(lower=0, upper=2) * 15
    rg_di_score = di_diff_norm.clip(lower=0, upper=10) * 3
    rg_ema_above = (c > ema21).astype(float) * 15
    adx_min20 = adx.rolling(20).min()
    rg_adx_rebound = ((adx - adx_min20) > 5).astype(float) * 15

    _c['rg_slope_score'] = rg_slope_score
    _c['rg_di_score'] = rg_di_score
    _c['rg_ema_above'] = rg_ema_above
    _c['rg_adx_rebound'] = rg_adx_rebound
    _c['rg_was_trending'] = was_trending

    # Composite regime score
    rg_raw = (rg_slope_score + rg_di_score + rg_ema_above + rg_adx_rebound).clip(0, 100)
    rg_score = rg_raw * was_trending
    _c['rg_score'] = rg_score

    # Gate
    _c['gate_open'] = ((br_score >= 25) | (rg_score >= 25)).astype(int)

    # ═══════════════════════════════════════
    # J. SELL SEVERITY PRİMİTİVLERİ (5) — nox_v3_signals.py exact
    # ═══════════════════════════════════════
    red_mask = c < o
    red_count = _consecutive_count(red_mask)
    _c['red_count'] = red_count
    _c['drawdown_20_pct'] = (c - high_20) / high_20.replace(0, np.nan) * 100
    high_close_5 = c.rolling(5).max()
    decline_5d_atr = (c - high_close_5) / atr_val.replace(0, np.nan)
    _c['decline_5d_atr'] = decline_5d_atr

    # Composite severity
    severity = pd.Series(0, index=df.index)
    sev1 = (_c['daily_move_atr'] < -1.5) | (red_count >= 3) | (_c['drawdown_20_pct'] < -8)
    severity = severity.where(~sev1, 1)
    sev2 = (_c['daily_move_atr'] < -2.5) | ((red_count >= 5) & (decline_5d_atr < -3))
    severity = severity.where(~sev2, 2)
    sev3 = (_c['daily_move_atr'] < -3.5) | (_c['drawdown_20_pct'] < -12)
    severity = severity.where(~sev3, 3)
    _c['sell_severity'] = severity.astype(int)

    # Pivot delta
    _c['pivot_delta_pct'] = (c - last_pl) / last_pl.replace(0, np.nan) * 100

    # ═══════════════════════════════════════
    # K. RT TPE PRİMİTİVLERİ (9+3) — regime_transition.py exact
    # ═══════════════════════════════════════

    # -- Trend sub-components --
    rt_ema_bull = (c > ema21).astype(int)
    rt_st_bull = (st_dir == 1).astype(int)
    _c['rt_ema_bull'] = rt_ema_bull
    _c['rt_st_bull'] = rt_st_bull

    # Weekly trend up (reindexed)
    if weekly_df is not None and len(weekly_df) >= 20:
        w_ema21_rt = _ema(weekly_df['Close'], 21)
        w_trend_rising = (w_ema21_rt > w_ema21_rt.shift(1)).astype(int)
        _c['rt_wk_trend_up'] = w_trend_rising.reindex(df.index, method='ffill')
    else:
        _c['rt_wk_trend_up'] = np.nan

    # trend_score = ema_bull + st_bull + wk_trend_up
    trend_s = rt_ema_bull + rt_st_bull + _c['rt_wk_trend_up'].fillna(0).astype(int)
    trend_s = trend_s.clip(0, 3)

    # -- Participation sub-components --
    rt_cmf_pos = (cmf20 > 0).astype(int)
    rt_rvol_high = (rvol >= 1.0).astype(int)
    obv_slope = _linreg_slope(obv_ema10, 5)
    rt_obv_slope_pos = (obv_slope > 0).astype(int)
    _c['rt_cmf_pos'] = rt_cmf_pos
    _c['rt_rvol_high'] = rt_rvol_high
    _c['rt_obv_slope_pos'] = rt_obv_slope_pos

    part_s = (rt_cmf_pos + rt_rvol_high + rt_obv_slope_pos).clip(0, 3)

    # -- Expansion sub-components --
    rt_adx_slope_pos = (adx_slope > 0).astype(int)
    atr_sma20 = _sma(atr_val, 20)
    rt_atr_expanding = (atr_val > atr_sma20 * 1.05).astype(int)
    rt_di_bull = ((plus_di - minus_di) > 5).astype(int)
    _c['rt_adx_slope_pos'] = rt_adx_slope_pos
    _c['rt_atr_expanding'] = rt_atr_expanding
    _c['rt_di_bull'] = rt_di_bull

    exp_s = (rt_adx_slope_pos + rt_atr_expanding + rt_di_bull).clip(0, 3)

    # Composite scores
    _c['trend_score'] = trend_s
    _c['participation_score'] = part_s
    _c['expansion_score'] = exp_s
    _c['tpe_total'] = trend_s + part_s + exp_s

    # ═══════════════════════════════════════
    # L. RT REGIME + ENTRY + OE + EXIT (6) — regime_transition.py exact
    # ═══════════════════════════════════════

    # Regime label
    regime = _compute_regime_from_tpe(trend_s, part_s, exp_s)
    _c['regime_score'] = regime

    # Entry score (CORRECTED: from run_regime_transition.py)
    # Low vol: ATR% < 3.0
    # Early entry: ADX slope < 0
    # Growth room: regime <= 2
    # No pump: RVOL < 2.0
    entry_s = pd.Series(0, index=df.index)
    entry_s = entry_s + (_c['atr_pct'] < 3.0).astype(int)
    entry_s = entry_s + (adx_slope < 0).astype(int)
    entry_s = entry_s + (regime <= 2).astype(int)
    entry_s = entry_s + (rvol < 2.0).astype(int)
    _c['entry_score'] = entry_s.clip(0, 4)

    # OE score (CORRECTED: from regime_transition.py)
    # RSI > 80, close > BB upper, 5-bar mom > 8%, EMA dist > 5%
    oe_s = pd.Series(0, index=df.index)
    oe_s = oe_s + (rsi14 > 80).astype(int)
    oe_s = oe_s + (c > bb_upper).astype(int)
    mom_5 = c.pct_change(5) * 100
    oe_s = oe_s + (mom_5 > 8).astype(int)
    ema_dist_abs = ((c / ema21.replace(0, np.nan) - 1) * 100).abs()
    oe_s = oe_s + (ema_dist_abs > 5).astype(int)
    _c['oe_score'] = oe_s.clip(0, 4)

    # Pullback conditions (for H+PB detection)
    _c['pb_ema_dist'] = ema_dist_abs
    _c['pb_rsi_low'] = (rsi14 <= 55).astype(int)

    # Exit stage
    _c['exit_stage'] = _compute_exit_stage(df, ema21, st_dir, adx_slope, cmf20, trend_s)

    # Days in trade
    _c['days_in_trade'] = _compute_days_in_trade(regime)

    # ═══════════════════════════════════════
    # M. TAVAN PRİMİTİVLERİ (6)
    # ═══════════════════════════════════════
    prev_close = c.shift(1)
    limit_up = prev_close * 1.10
    _c['is_tavan'] = (c >= limit_up * 0.995).astype(int)
    _c['tavan_streak'] = _consecutive_count(_c['is_tavan'].astype(bool))
    # close_to_high: tavan-specific (close/high, NOT candle location value)
    _c['close_to_high'] = c / h.replace(0, np.nan)
    _c['tavan_locked'] = ((_c['is_tavan'] == 1) & (_c['close_to_high'] >= 0.95)).astype(int)
    _c['hit_tavan_intraday'] = (h >= limit_up * 0.995).astype(int)
    _c['recent_tavan_10d'] = _c['is_tavan'].rolling(10, min_periods=1).sum()

    # ═══════════════════════════════════════
    # N. RELATİF GÜÇ (3) — regime.py exact (RS_LEN1=10, RS_LEN2=60)
    # ═══════════════════════════════════════
    if xu_df is not None and 'Close' in xu_df.columns and len(xu_df) >= 65:
        xu_close = xu_df['Close'].reindex(df.index, method='ffill')
        stock_ret_10 = c.pct_change(10) * 100
        bench_ret_10 = xu_close.pct_change(10) * 100
        stock_ret_60 = c.pct_change(60) * 100
        bench_ret_60 = xu_close.pct_change(60) * 100
        _c['rs_10'] = stock_ret_10 - bench_ret_10
        _c['rs_60'] = stock_ret_60 - bench_ret_60
        _c['rs_composite'] = _c['rs_10'] * 0.6 + _c['rs_60'] * 0.4
    else:
        _c['rs_10'] = np.nan
        _c['rs_60'] = np.nan
        _c['rs_composite'] = np.nan

    # ═══════════════════════════════════════
    # O. PRE-BREAKOUT / BİRİKİM→BREAKOUT (22)
    # 20 gün birikim penceresine dayalı feature'lar
    # ═══════════════════════════════════════

    # O1. Range Sıkışması (4)
    range_5 = h.rolling(5).max() - l.rolling(5).min()
    range_20 = h.rolling(20).max() - l.rolling(20).min()
    range_40 = h.rolling(40).max() - l.rolling(40).min()
    _c['range_contraction_5_20'] = range_5 / range_20.replace(0, np.nan)
    _c['range_contraction_5_40'] = range_5 / range_40.replace(0, np.nan)
    atr_5 = _rma(_true_range(df), 5)
    _c['atr_contraction_5_20'] = atr_5 / atr_val.replace(0, np.nan)
    bb_width_raw = bb_range / bb_mid.replace(0, np.nan)
    _c['bb_width_pctile_20'] = bb_width_raw.rolling(60, min_periods=20).rank(pct=True)

    # O2. Hacim Birikim Paterni (5)
    vol_sma5 = _sma(v, 5)
    _c['vol_dryup_ratio'] = vol_sma5 / vol_sma20.replace(0, np.nan)
    _c['vol_surge_today'] = v / vol_sma5.replace(0, np.nan)
    _c['vol_acceleration'] = (vol_sma5 - vol_sma5.shift(5)) / vol_sma5.shift(5).replace(0, np.nan)
    _c['tl_volume_20d_avg'] = _sma(c * v, 20)
    vol_dryup = _c['vol_dryup_ratio'].clip(upper=2)
    vol_spike = _c['vol_surge_today'].clip(upper=5)
    _c['vol_pattern_score'] = (2 - vol_dryup) * 2 + np.where(vol_spike > 2, vol_spike * 3, 0)

    # O3. Momentum Oluşumu (4)
    _c['rsi_momentum_5d'] = rsi14 - rsi14.shift(5)
    _c['macd_hist_accel'] = macd_hist - macd_hist.shift(3)
    higher_close_mask = c > c.shift(1)
    _c['consecutive_higher_close'] = _consecutive_count(higher_close_mask)
    high_20d = h.rolling(20).max()
    _c['close_vs_20d_high_pct'] = (c - high_20d) / high_20d.replace(0, np.nan) * 100

    # O4. Yapısal Yakınlık (4)
    high_52w = h.rolling(252, min_periods=60).max()
    _c['dist_to_52w_high_pct'] = (c - high_52w) / high_52w.replace(0, np.nan) * 100
    _c['dist_to_20d_high_pct'] = _c['close_vs_20d_high_pct']  # same calc
    _c['near_52w_high'] = (c >= high_52w * 0.95).astype(int)
    low_20d = l.rolling(20).min()
    _c['price_range_position_20'] = (c - low_20d) / (high_20d - low_20d).replace(0, np.nan)

    # O5. Yakın Iskalama / Öncü Sinyal (3)
    daily_ret_pct = (c / c.shift(1) - 1) * 100
    _c['near_tavan_miss'] = ((daily_ret_pct >= 7) & (daily_ret_pct < 9.5)).astype(int)
    _c['recent_near_tavan_5d'] = _c['near_tavan_miss'].rolling(5, min_periods=1).sum()
    _c['max_daily_ret_5d'] = daily_ret_pct.rolling(5, min_periods=1).max()

    # O6. Relatif Güç İvmesi (2)
    if xu_df is not None and 'Close' in xu_df.columns and len(xu_df) >= 65:
        _c['rs_acceleration'] = _c['rs_10'] - _c['rs_10'].shift(5)
    else:
        _c['rs_acceleration'] = np.nan
    # market_tavan_count_10d: piyasa geneli, dışarıdan verilmeli — NaN olarak bırak
    _c['market_tavan_count_10d'] = np.nan

    # O7. Lag-1 Güvenli Feature'lar (8) — leakage-free versiyonlar
    _c['returns_1d_lag1'] = _c['returns_1d'].shift(1)
    _c['close_position_lag1'] = _c['close_position'].shift(1)
    _c['close_to_high_lag1'] = _c['close_to_high'].shift(1)
    daily_ret_pct_lag = daily_ret_pct.shift(1)
    _c['max_daily_ret_5d_lag1'] = daily_ret_pct_lag.rolling(5, min_periods=1).max()
    near_miss_lag = _c['near_tavan_miss'].shift(1)
    _c['recent_near_tavan_5d_lag1'] = near_miss_lag.rolling(5, min_periods=1).sum()
    _c['vol_surge_yesterday'] = v.shift(1) / _sma(v, 5).shift(1).replace(0, np.nan)
    _c['consecutive_green_lag1'] = _c['consecutive_green'].shift(1)
    is_tavan_lag = _c['is_tavan'].shift(1)
    _c['recent_tavan_10d_lag1'] = is_tavan_lag.rolling(10, min_periods=1).sum()

    # ═══════════════════════════════════════════════════════════
    # P-AK GROUPS — MARKER BEGIN
    # T-close konvansiyon: tüm yeni feature'lar ham + otomatik _lag1
    # ═══════════════════════════════════════════════════════════
    _keys_before_p_ak = set(_c.keys())

    # ═══════════════════════════════════════
    # S. VOLATİLİTE REJİM GEÇİŞLERİ
    # ═══════════════════════════════════════
    _c['atr_pct_rank_60'] = atr_pct.rolling(60, min_periods=20).rank(pct=True)
    _c['atr_pct_rank_252'] = atr_pct.rolling(252, min_periods=60).rank(pct=True)
    log_ret = np.log(c / c.shift(1).replace(0, np.nan))
    _c['realized_vol_5d'] = log_ret.rolling(5).std() * np.sqrt(252) * 100
    _c['realized_vol_10d'] = log_ret.rolling(10).std() * np.sqrt(252) * 100
    _c['realized_vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    _c['vol_of_vol_20d'] = _c['realized_vol_5d'].rolling(20).std()
    atr_5_s = _rma(_true_range(df), 5)
    atr_10_s = _rma(_true_range(df), 10)
    atr_20_s = _rma(_true_range(df), 20)
    atr_40_s = _rma(_true_range(df), 40)
    _c['atr_expansion_ratio_5_20'] = atr_5_s / atr_20_s.replace(0, np.nan)
    _c['atr_expansion_ratio_10_40'] = atr_10_s / atr_40_s.replace(0, np.nan)
    _c['bb_width_expansion_5d'] = (bb_width_raw / bb_width_raw.shift(5).replace(0, np.nan)) - 1
    _c['range_expansion_today'] = (h - l) / atr_val.replace(0, np.nan)
    _c['range_expansion_5d'] = (h - l).rolling(5).mean() / atr_val.replace(0, np.nan)
    _atr_rank60_filled = _c['atr_pct_rank_60'].fillna(0.5)
    _bb_rank60 = bb_width_raw.rolling(60, min_periods=20).rank(pct=True).fillna(0.5)
    _rng_rank60 = (h - l).rolling(60, min_periods=20).rank(pct=True).fillna(0.5)
    _c['volatility_compression_score'] = ((1 - _atr_rank60_filled) + (1 - _bb_rank60) + (1 - _rng_rank60)) / 3
    _prior_compression = _c['volatility_compression_score'].shift(1)
    _c['volatility_release_score'] = (_prior_compression * _c['range_expansion_today']).fillna(0)
    _prior_atr_rank = _c['atr_pct_rank_60'].shift(5)
    _c['volatility_regime_change'] = ((_prior_atr_rank < 0.3) & (_c['atr_pct_rank_60'] > 0.7)).astype(int)

    # ═══════════════════════════════════════
    # T. GETIRI DAĞILIMI / KUYRUK ŞEKLİ
    # ═══════════════════════════════════════
    _ret = daily_ret_pct
    _c['ret_skew_20d'] = _ret.rolling(20).skew()
    _c['ret_skew_60d'] = _ret.rolling(60, min_periods=30).skew()
    _c['ret_kurt_20d'] = _ret.rolling(20).kurt()
    _c['ret_kurt_60d'] = _ret.rolling(60, min_periods=30).kurt()
    _c['max_up_day_20d'] = _ret.rolling(20).max()
    _c['max_down_day_20d'] = _ret.rolling(20).min()
    _up_vol = _ret.where(_ret > 0).rolling(20, min_periods=5).std()
    _down_vol = _ret.where(_ret < 0).abs().rolling(20, min_periods=5).std()
    _c['up_down_vol_ratio_20d'] = _up_vol / _down_vol.replace(0, np.nan)
    _c['positive_day_ratio_20d'] = (_ret > 0).astype(float).rolling(20).mean()
    _ret_atr = (c - c.shift(1)) / atr_val.replace(0, np.nan)
    _c['large_green_count_20d'] = (_ret_atr > 1.5).astype(float).rolling(20).sum()
    _c['large_red_count_20d'] = (_ret_atr < -1.5).astype(float).rolling(20).sum()
    _c['tail_risk_score'] = (
        (-_ret.rolling(20).min().clip(upper=0) / 5).clip(0, 3) +
        (_c['drawdown_20_pct'].abs() / 10).clip(0, 3) +
        (atr_pct / 3).clip(0, 3)
    ) / 9
    _cum_ret_20 = _ret.rolling(20).sum()
    _c['lottery_profile_score'] = (_ret.rolling(20).max() / _cum_ret_20.abs().replace(0, np.nan)).clip(-3, 3)

    # ═══════════════════════════════════════
    # U. LİKİDİTE / TRADABILITY (per-ticker subset)
    # ═══════════════════════════════════════
    _tl_vol = c * v
    _c['tl_volume_5d_avg'] = _tl_vol.rolling(5).mean()
    _c['tl_volume_60d_avg'] = _tl_vol.rolling(60, min_periods=20).mean()
    _tl_z_mean = _tl_vol.rolling(20, min_periods=10).mean()
    _tl_z_std = _tl_vol.rolling(20, min_periods=10).std()
    _c['tl_volume_z_20'] = (_tl_vol - _tl_z_mean) / _tl_z_std.replace(0, np.nan)
    _vol_mean20 = v.rolling(20).mean()
    _vol_std20 = v.rolling(20).std()
    _c['volume_consistency_20d'] = _vol_mean20 / _vol_std20.replace(0, np.nan)
    _c['zero_volume_days_60d'] = (v < vol_sma20 * 0.1).astype(float).rolling(60, min_periods=20).sum()
    _abs_ret = daily_ret_pct.abs()
    _c['amihud_illiq_20d'] = (_abs_ret / _tl_vol.replace(0, np.nan)).rolling(20).mean() * 1e9
    _c['amihud_illiq_60d'] = (_abs_ret / _tl_vol.replace(0, np.nan)).rolling(60, min_periods=20).mean() * 1e9
    _c['spread_proxy'] = (h - l) / c.replace(0, np.nan)
    _c['impact_cost_proxy'] = atr_pct / np.log1p(_tl_vol).replace(0, np.nan)
    _c['min_tl_volume_20d'] = _tl_vol.rolling(20).min()
    _c['liquidity_drop_flag'] = (_c['tl_volume_5d_avg'] < _c['tl_volume_20d_avg'] * 0.5).astype(int)
    _liq_score = np.log1p(_c['tl_volume_20d_avg']).clip(0, 25) / 25
    _vol_score = (1 - atr_pct.clip(0, 10) / 10)
    _c['tradability_score'] = (_liq_score * 0.6 + _vol_score * 0.4)

    # ═══════════════════════════════════════
    # R. XU100-TÜREVİ (per-ticker; market-breadth ayrı aggregator'da)
    # ═══════════════════════════════════════
    if xu_df is not None and 'Close' in xu_df.columns and len(xu_df) >= 65:
        xu_c = xu_df['Close'].reindex(df.index, method='ffill')
        xu_h = xu_df['High'].reindex(df.index, method='ffill') if 'High' in xu_df.columns else xu_c
        xu_l = xu_df['Low'].reindex(df.index, method='ffill') if 'Low' in xu_df.columns else xu_c
        xu_o = xu_df['Open'].reindex(df.index, method='ffill') if 'Open' in xu_df.columns else xu_c
        xu_ohlc = pd.DataFrame({'Open': xu_o, 'High': xu_h, 'Low': xu_l, 'Close': xu_c})
        _c['xu100_return_1d'] = xu_c.pct_change() * 100
        _c['xu100_return_5d'] = xu_c.pct_change(5) * 100
        xu_ema21 = _ema(xu_c, 21)
        xu_ema55 = _ema(xu_c, 55)
        _c['xu100_above_ema21'] = (xu_c > xu_ema21).astype(int)
        _c['xu100_above_ema55'] = (xu_c > xu_ema55).astype(int)
        xu_adx, _, _ = _calc_adx_with_di(xu_ohlc, 14)
        _c['xu100_adx_14'] = xu_adx
        xu_bb_mid = _sma(xu_c, 20)
        xu_bb_std = xu_c.rolling(20).std()
        _c['xu100_bb_width'] = ((xu_bb_mid + 2 * xu_bb_std) - (xu_bb_mid - 2 * xu_bb_std)) / xu_bb_mid.replace(0, np.nan) * 100
        xu_atr14 = _calc_atr(xu_ohlc, 14)
        _c['xu100_atr_pct'] = xu_atr14 / xu_c.replace(0, np.nan) * 100
        xu_high20 = xu_h.rolling(20).max()
        _c['xu100_drawdown_20'] = (xu_c - xu_high20) / xu_high20.replace(0, np.nan) * 100
        xu_adx_slope = (xu_adx - xu_adx.shift(5)) / 5
        _c['xu100_regime_score'] = (
            _c['xu100_above_ema21'].astype(float) * 30 +
            _c['xu100_above_ema55'].astype(float) * 25 +
            (xu_adx_slope > 0).astype(float) * 20 +
            (_c['xu100_drawdown_20'] > -5).astype(float) * 25
        )
    else:
        for _k in ['xu100_return_1d', 'xu100_return_5d', 'xu100_above_ema21', 'xu100_above_ema55',
                   'xu100_adx_14', 'xu100_bb_width', 'xu100_atr_pct', 'xu100_drawdown_20', 'xu100_regime_score']:
            _c[_k] = pd.Series(np.nan, index=df.index)

    # ═══════════════════════════════════════
    # V. GAP DAVRANIŞI (daily-derived; intraday alt küme SKIP)
    # ═══════════════════════════════════════
    prior_c = c.shift(1)
    gap_abs = o - prior_c
    _c['gap_atr'] = gap_abs / atr_val.replace(0, np.nan)
    _c['gap_direction'] = pd.Series(np.sign(gap_abs.values), index=df.index, dtype=float)
    _c['gap_after_squeeze'] = ((_c['gap_pct'].abs() > 1.5).astype(int) *
                               _c['squeeze_on'].shift(1, fill_value=0).astype(int))
    _vol_dryup_yest = (_c['vol_dryup_ratio'].shift(1) < 0.7).astype(int)
    _c['gap_after_volume_dryup'] = (_c['gap_pct'].abs() > 1.5).astype(int) * _vol_dryup_yest
    _c['gap_after_tavan'] = ((_c['is_tavan'].shift(1) == 1) & (_c['gap_pct'] > 0)).astype(int)
    _c['avg_gap_5d'] = _c['gap_pct'].rolling(5).mean()
    _c['gap_volatility_20d'] = _c['gap_pct'].rolling(20).std()
    _c['overnight_return_1d'] = (o / prior_c.replace(0, np.nan) - 1) * 100
    _c['intraday_return_1d'] = (c / o.replace(0, np.nan) - 1) * 100
    _on_var = _c['overnight_return_1d'].rolling(20).var()
    _id_var = _c['intraday_return_1d'].rolling(20).var()
    _c['overnight_vs_intraday_20d'] = _on_var / (_on_var + _id_var).replace(0, np.nan)
    _gap_pos_mask = (_c['gap_pct'] > 1)
    _green_today = (c > o)
    _n_gap_20 = _gap_pos_mask.astype(float).rolling(20).sum()
    _n_continue_20 = (_gap_pos_mask & _green_today).astype(float).rolling(20).sum()
    _c['gap_continuation_rate_20d'] = _n_continue_20 / _n_gap_20.replace(0, np.nan)

    # ═══════════════════════════════════════
    # W. MUM ANATOMİSİ
    # ═══════════════════════════════════════
    body = c - o
    body_abs = body.abs()
    co_max = pd.concat([c, o], axis=1).max(axis=1)
    co_min = pd.concat([c, o], axis=1).min(axis=1)
    upper_wick_w = h - co_max
    lower_wick_w = co_min - l
    rng_w = (h - l).replace(0, np.nan)
    _c['body_pct_range'] = body_abs / rng_w
    _c['upper_wick_pct_range'] = upper_wick_w / rng_w
    _c['lower_wick_pct_range'] = lower_wick_w / rng_w
    _c['body_atr'] = body_abs / atr_val.replace(0, np.nan)
    _c['range_atr'] = (h - l) / atr_val.replace(0, np.nan)
    _c['close_location_value'] = (c - l) / rng_w
    _c['open_location_value'] = (o - l) / rng_w
    _yest_red = (c.shift(1) < o.shift(1))
    _yest_green = (c.shift(1) > o.shift(1))
    _today_green = (c > o)
    _today_red = (c < o)
    _c['bullish_engulfing_flag'] = (_yest_red & _today_green &
                                    (c >= o.shift(1)) & (o <= c.shift(1))).fillna(False).astype(int)
    _c['bearish_engulfing_flag'] = (_yest_green & _today_red &
                                    (c <= o.shift(1)) & (o >= c.shift(1))).fillna(False).astype(int)
    _c['hammer_flag'] = (
        (lower_wick_w > 2 * body_abs) &
        (upper_wick_w < body_abs) &
        ((c - l) / rng_w > 0.6)
    ).fillna(False).astype(int)
    _c['shooting_star_flag'] = (
        (upper_wick_w > 2 * body_abs) &
        (lower_wick_w < body_abs) &
        ((h - c) / rng_w > 0.6)
    ).fillna(False).astype(int)
    _c['inside_bar_flag'] = ((h < h.shift(1)) & (l > l.shift(1))).fillna(False).astype(int)
    _c['outside_bar_flag'] = ((h > h.shift(1)) & (l < l.shift(1))).fillna(False).astype(int)
    _rng_raw = (h - l)
    _c['nr7_flag'] = (_rng_raw == _rng_raw.rolling(7).min()).fillna(False).astype(int)
    _c['nr4_flag'] = (_rng_raw == _rng_raw.rolling(4).min()).fillna(False).astype(int)
    _c['wide_range_bar_flag'] = (_c['range_atr'] > 2.0).fillna(False).astype(int)
    _c['close_above_prev_high'] = (c > h.shift(1)).fillna(False).astype(int)
    _c['close_below_prev_low'] = (c < l.shift(1)).fillna(False).astype(int)
    _c['reversal_from_low_atr'] = (c - l) / atr_val.replace(0, np.nan)
    _c['rejection_from_high_atr'] = (h - c) / atr_val.replace(0, np.nan)

    # ═══════════════════════════════════════
    # X. DESTEK / DİRENÇ / PİVOT GEOMETRİSİ
    # ═══════════════════════════════════════
    _c['dist_to_pivot_high_pct'] = (c - last_pivot_high) / last_pivot_high.replace(0, np.nan) * 100
    _c['dist_to_pivot_low_pct'] = (c - last_pl) / last_pl.replace(0, np.nan) * 100
    res_20 = h.rolling(20).max()
    sup_20 = l.rolling(20).min()
    _c['dist_to_resistance_20d'] = (c - res_20) / res_20.replace(0, np.nan) * 100
    _c['dist_to_support_20d'] = (c - sup_20) / sup_20.replace(0, np.nan) * 100
    _touch_res = (h >= res_20 * 0.995).astype(float)
    _c['resistance_touch_count_20d'] = _touch_res.rolling(20).sum()
    _touch_sup = (l <= sup_20 * 1.005).astype(float)
    _c['support_touch_count_20d'] = _touch_sup.rolling(20).sum()
    prior_res_20 = res_20.shift(1)
    broke_above = (c > prior_res_20 * 1.005)
    broke_5_ago = broke_above.shift(5, fill_value=False)
    prior_res_5_ago = prior_res_20.shift(5)
    failed_event = (broke_5_ago & (c < prior_res_5_ago))
    success_event = (broke_5_ago & (c > prior_res_5_ago))
    _c['failed_breakout_count_60d'] = failed_event.astype(float).rolling(60, min_periods=20).sum()
    _c['successful_breakout_count_60d'] = success_event.astype(float).rolling(60, min_periods=20).sum()
    _idx_range = pd.Series(np.arange(n), index=df.index, dtype=float)
    _ph_mask = pivot_highs.notna()
    _ph_idx = _idx_range.where(_ph_mask).ffill()
    _c['pivot_high_age'] = _idx_range - _ph_idx
    _pl_mask = pivot_lows.notna()
    _pl_idx = _idx_range.where(_pl_mask).ffill()
    _c['pivot_low_age'] = _idx_range - _pl_idx
    range_mid = (high_20d + low_20d) / 2
    _c['range_mid_dist_pct'] = (c - range_mid) / range_mid.replace(0, np.nan) * 100
    _c['range_upper_dist_pct'] = (c - high_20d) / high_20d.replace(0, np.nan) * 100
    _c['range_lower_dist_pct'] = (c - low_20d) / low_20d.replace(0, np.nan) * 100
    range_pct_20 = (high_20d - low_20d) / c.replace(0, np.nan) * 100
    base_active = (range_pct_20 < 8)
    _c['base_duration_bars'] = _consecutive_count(base_active)
    _c['base_width_pct'] = range_pct_20
    _c['base_width_atr'] = (high_20d - low_20d) / atr_val.replace(0, np.nan)
    _c['base_slope'] = _linreg_slope(c, 20)
    _range_score = (1 - range_pct_20.clip(0, 15) / 15)
    _slope_score = (1 - _c['base_slope'].abs().clip(0, 0.5) / 0.5)
    _vol_dry_score = (1 - _c['vol_dryup_ratio'].clip(0.3, 1.5) / 1.5).clip(0, 1)
    _c['base_quality_score'] = (_range_score * 0.4 + _slope_score * 0.3 + _vol_dry_score * 0.3)

    # ═══════════════════════════════════════
    # Y. BREAKOUT KALİTESİ
    # ═══════════════════════════════════════
    brk_level = h.rolling(20).max().shift(1)
    _c['breakout_distance_pct'] = (c - brk_level) / brk_level.replace(0, np.nan) * 100
    _c['breakout_distance_atr'] = (c - brk_level) / atr_val.replace(0, np.nan)
    _c['breakout_volume_ratio'] = v / vol_sma20.replace(0, np.nan)
    _c['breakout_close_strength'] = (c - l) / rng_w
    _c['breakout_body_atr'] = body_abs / atr_val.replace(0, np.nan)
    _c['breakout_upper_wick_pct'] = upper_wick_w / rng_w
    _attempts_60 = broke_above.astype(float).rolling(60, min_periods=20).sum()
    _c['prior_breakout_failure_rate'] = (_c['failed_breakout_count_60d'] / _attempts_60.replace(0, np.nan)).clip(0, 1)
    _c['breakout_from_squeeze_flag'] = ((c > brk_level) & (_c['squeeze_on'].shift(1) == 1)).fillna(False).astype(int)
    _c['breakout_from_accumulation_flag'] = ((c > brk_level) & (_c['vol_dryup_ratio'].shift(1) < 0.8)).fillna(False).astype(int)
    _c['breakout_near_52w_high'] = ((c >= high_52w * 0.95) & (c > brk_level)).fillna(False).astype(int)
    _c['breakout_above_20d_high'] = (c > brk_level).fillna(False).astype(int)
    _c['breakout_above_60d_high'] = (c > h.rolling(60).max().shift(1)).fillna(False).astype(int)
    _c['breakout_above_252d_high'] = (c > h.rolling(252, min_periods=60).max().shift(1)).fillna(False).astype(int)
    _c['false_breakout_risk_score'] = (
        (upper_wick_w / rng_w).fillna(0).clip(0, 1) * 0.4 +
        (vol_sma20.replace(0, np.nan) / v.replace(0, np.nan)).fillna(0).clip(0, 3) / 3 * 0.3 +
        _c['breakout_distance_pct'].fillna(0).clip(0, 10) / 10 * 0.3
    )

    # ═══════════════════════════════════════
    # Z. PULLBACK / RETEST
    # ═══════════════════════════════════════
    recent_high = h.rolling(20).max()
    _c['pullback_depth_atr'] = (recent_high - c) / atr_val.replace(0, np.nan)
    _c['pullback_depth_pct'] = (c - recent_high) / recent_high.replace(0, np.nan) * 100
    in_pullback = (c < recent_high * 0.98)
    _c['pullback_duration'] = _consecutive_count(in_pullback)
    _pb_vol_ratio = v / vol_sma20.replace(0, np.nan)
    _pb_vol_during = _pb_vol_ratio.where(in_pullback)
    _c['pullback_volume_dryup'] = _pb_vol_during.rolling(5, min_periods=1).mean()
    _pb_wick = lower_wick_w / rng_w
    _c['pullback_lower_wick_score'] = _pb_wick.where(in_pullback).rolling(5, min_periods=1).mean()
    _c['pullback_to_ema21_dist'] = (c - ema21) / ema21.replace(0, np.nan) * 100
    _c['pullback_to_ema55_dist'] = (c - ema55) / ema55.replace(0, np.nan) * 100
    prior_brk_level = h.rolling(20).max().shift(20)
    _c['pullback_to_breakout_level_dist'] = (c - prior_brk_level) / prior_brk_level.replace(0, np.nan) * 100
    _c['retest_hold_flag'] = (
        (l <= prior_brk_level * 1.02) & (c > prior_brk_level) & (c > o)
    ).fillna(False).astype(int)
    _c['retest_break_flag'] = (
        (c < prior_brk_level) & (c.shift(1) > prior_brk_level)
    ).fillna(False).astype(int)
    _brk_event = ((c > prior_brk_level) & (c.shift(1) <= prior_brk_level))
    _brk_idx = _idx_range.where(_brk_event).ffill()
    _c['retest_age'] = _idx_range - _brk_idx
    _c['retest_quality_score'] = (
        _c['retest_hold_flag'].astype(float).rolling(5).sum() * 0.4 / 5 +
        (_pb_vol_ratio < 0.7).astype(float).rolling(5).sum() * 0.3 / 5 +
        _pb_wick.fillna(0).rolling(5).mean() * 0.3
    )
    _recent_hl = (last_pl > prev_pl).astype(int)
    _recent_hl_active = _recent_hl.rolling(10, min_periods=1).max()
    _c['higher_low_after_breakout'] = ((_recent_hl_active == 1) &
                                        (_c['retest_age'].fillna(999) < 15)).astype(int)
    _c['shakeout_flag'] = ((l < sup_20.shift(1)) & (c > sup_20.shift(1))).fillna(False).astype(int)

    # ═══════════════════════════════════════
    # AA. ORTALAMAYA DÖNÜŞ / TÜKENMİŞLİK
    # ═══════════════════════════════════════
    _c['rsi_2_percentile_252'] = _c['rsi_2'].rolling(252, min_periods=60).rank(pct=True)
    _c['rsi_14_percentile_252'] = rsi14.rolling(252, min_periods=60).rank(pct=True)
    typ_price = (h + l + c) / 3
    _tpv_sum = (typ_price * v).rolling(20).sum()
    _v_sum = v.rolling(20).sum()
    vwap_proxy_20 = _tpv_sum / _v_sum.replace(0, np.nan)
    _c['distance_from_vwap_proxy'] = (c - vwap_proxy_20) / vwap_proxy_20.replace(0, np.nan) * 100
    _c['ema21_overextension_atr'] = (c - ema21) / atr_val.replace(0, np.nan)
    _c['ema55_overextension_atr'] = (c - ema55) / atr_val.replace(0, np.nan)
    _c['bb_upper_extension_pct'] = ((c - bb_upper) / bb_upper.replace(0, np.nan) * 100).clip(lower=0)
    _c['consecutive_up_days'] = _consecutive_count(c > c.shift(1))
    _c['consecutive_down_days'] = _consecutive_count(c < c.shift(1))
    _c['climax_volume_flag'] = ((v > vol_sma20 * 2.5) & (daily_ret_pct.abs() > 4)).fillna(False).astype(int)
    _c['climax_return_flag'] = (daily_ret_pct.abs() > 7).fillna(False).astype(int)
    _c['exhaustion_gap_flag'] = (
        (_c['gap_pct'] > 2) & (_c['returns_10d'] > 15) & (rsi14 > 70)
    ).fillna(False).astype(int)
    _c['blowoff_top_risk'] = (
        (rsi14 > 75).astype(float) * 0.3 +
        (upper_wick_w / rng_w).fillna(0).clip(0, 1) * 0.3 +
        (v > vol_sma20 * 2).astype(float) * 0.2 +
        (_c['ema21_overextension_atr'] > 3).astype(float) * 0.2
    )
    _c['capitulation_flag'] = (
        (daily_ret_pct < -5) & (v > vol_sma20 * 2) & ((lower_wick_w / rng_w) > 0.4)
    ).fillna(False).astype(int)
    _c['mean_reversion_long_score'] = (
        (1 - _c['rsi_2_percentile_252'].fillna(0.5)) * 0.4 +
        (-_c['ema21_overextension_atr'].clip(-3, 0) / 3) * 0.3 +
        (lower_wick_w / rng_w).fillna(0).clip(0, 1) * 0.3
    )
    _c['mean_reversion_short_risk'] = (
        _c['rsi_14_percentile_252'].fillna(0.5) * 0.3 +
        _c['ema21_overextension_atr'].clip(0, 5).fillna(0) / 5 * 0.4 +
        _c['blowoff_top_risk'] * 0.3
    )

    # ═══════════════════════════════════════
    # AB. ÇOKLU ZAMAN ÇERÇEVESİ (Weekly + Monthly proxy)
    # ═══════════════════════════════════════
    if weekly_df is not None and len(weekly_df) >= 20:
        w_c = weekly_df['Close']
        w_h = weekly_df['High']
        w_l = weekly_df['Low']
        _c['weekly_return_1w'] = (w_c.pct_change() * 100).reindex(df.index, method='ffill')
        _c['weekly_return_4w'] = (w_c.pct_change(4) * 100).reindex(df.index, method='ffill')
        w_rsi14_ab = _calc_rsi(w_c, 14)
        _c['weekly_rsi_14'] = w_rsi14_ab.reindex(df.index, method='ffill')
        _, _, w_macd_hist_ab = _calc_macd(w_c)
        _c['weekly_macd_hist'] = w_macd_hist_ab.reindex(df.index, method='ffill')
        w_rng_ab = (w_h - w_l).replace(0, np.nan)
        _c['weekly_close_position'] = ((w_c - w_l) / w_rng_ab).reindex(df.index, method='ffill')
        w_bb_mid_ab = _sma(w_c, 20)
        w_bb_std_ab = w_c.rolling(20).std()
        w_bb_width_ab = ((w_bb_mid_ab + 2 * w_bb_std_ab) - (w_bb_mid_ab - 2 * w_bb_std_ab)) / w_bb_mid_ab.replace(0, np.nan) * 100
        _c['weekly_bb_width'] = w_bb_width_ab.reindex(df.index, method='ffill')
        w_atr_ab = _calc_atr(weekly_df, 14)
        _c['weekly_atr_pct'] = (w_atr_ab / w_c.replace(0, np.nan) * 100).reindex(df.index, method='ffill')
        w_high52 = w_h.rolling(52, min_periods=20).max()
        _c['weekly_near_52w_high'] = (w_c >= w_high52 * 0.95).astype(int).reindex(df.index, method='ffill')
        w_high20_brk = w_h.rolling(20).max().shift(1)
        _c['weekly_breakout_flag'] = (w_c > w_high20_brk).astype(int).reindex(df.index, method='ffill')
        w_ema21_ab = _ema(w_c, 21)
        w_ema55_ab = _ema(w_c, 55)
        w_adx_ab, _, _ = _calc_adx_with_di(weekly_df, 14)
        w_reg_ab = ((w_c > w_ema21_ab).astype(float) * 30 +
                    (w_c > w_ema55_ab).astype(float) * 25 +
                    (w_adx_ab > 25).astype(float) * 25 +
                    (w_macd_hist_ab > 0).astype(float) * 20)
        _c['weekly_regime_score'] = w_reg_ab.reindex(df.index, method='ffill')
        _daily_trend_up = (c > ema21).astype(int)
        _weekly_trend_up = (w_c > w_ema21_ab).astype(int).reindex(df.index, method='ffill').fillna(0).astype(int)
        _c['daily_weekly_alignment'] = (
            ((_daily_trend_up == 1) & (_weekly_trend_up == 1)).astype(int) -
            ((_daily_trend_up == 0) & (_weekly_trend_up == 0)).astype(int)
        )
    else:
        for _k in ['weekly_return_1w', 'weekly_return_4w', 'weekly_rsi_14', 'weekly_macd_hist',
                   'weekly_close_position', 'weekly_bb_width', 'weekly_atr_pct',
                   'weekly_near_52w_high', 'weekly_breakout_flag', 'weekly_regime_score',
                   'daily_weekly_alignment']:
            _c[_k] = pd.Series(np.nan, index=df.index)
    _c['monthly_return_1m'] = c.pct_change(21) * 100
    _c['monthly_trend_up'] = (c > _ema(c, 63)).astype(int)
    _m_h21 = h.rolling(21).max()
    _m_l21 = l.rolling(21).min()
    _c['monthly_close_position'] = (c - _m_l21) / (_m_h21 - _m_l21).replace(0, np.nan)

    # ═══════════════════════════════════════
    # AF. TAKVİM / MEVSİMSELLİK
    # ═══════════════════════════════════════
    if isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
        _c['day_of_week'] = pd.Series(dt.dayofweek.astype(float), index=df.index)
        _c['month_of_year'] = pd.Series(dt.month.astype(float), index=df.index)
        _c['week_of_month'] = pd.Series(((dt.day - 1) // 7 + 1).astype(float), index=df.index)
        _month_per = pd.Series(dt.to_period('M').astype(str), index=df.index)
        _c['month_start_flag'] = (_month_per != _month_per.shift(1)).astype(int)
        _c['month_end_flag'] = (_month_per != _month_per.shift(-1)).astype(int)
        _quarter_per = pd.Series(dt.to_period('Q').astype(str), index=df.index)
        _c['quarter_end_flag'] = (_quarter_per != _quarter_per.shift(-1)).astype(int)
        _df_idx_s = pd.Series(np.arange(n), index=df.index, dtype=float)
        _grp_codes = _month_per.factorize()[0]
        _grp_first = _df_idx_s.groupby(_grp_codes).transform('min')
        _grp_last = _df_idx_s.groupby(_grp_codes).transform('max')
        _c['days_from_month_start'] = _df_idx_s - _grp_first
        _c['days_to_month_end'] = _grp_last - _df_idx_s
        _idx_ts = df.index.to_series()
        _days_diff = _idx_ts.diff().dt.days.fillna(1).astype(int)
        _c['post_holiday_flag'] = (_days_diff > 3).astype(int)
        _days_to_next = (_idx_ts.shift(-1) - _idx_ts).dt.days.fillna(1).astype(int)
        _c['pre_holiday_flag'] = (_days_to_next > 3).astype(int)
        _week_per = pd.Series(dt.to_period('W').astype(str), index=df.index)
        _week_counts = _week_per.groupby(_week_per).transform('count')
        _c['short_week_flag'] = (_week_counts < 5).astype(int)
        _c['earnings_season_proxy'] = pd.Series(
            dt.month.isin([2, 5, 8, 11]).astype(int), index=df.index
        )
        _rebal_arr = (dt.month.isin([3, 6, 9, 12]) & (dt.day >= 22)).astype(int)
        _c['index_rebalance_window'] = pd.Series(_rebal_arr, index=df.index)
        _taxloss_arr = ((dt.month == 12) & (dt.day >= 15)).astype(int)
        _c['tax_loss_window_proxy'] = pd.Series(_taxloss_arr, index=df.index)
    else:
        for _k in ['day_of_week', 'month_of_year', 'week_of_month', 'month_start_flag', 'month_end_flag',
                   'quarter_end_flag', 'days_from_month_start', 'days_to_month_end',
                   'pre_holiday_flag', 'post_holiday_flag', 'short_week_flag',
                   'earnings_season_proxy', 'index_rebalance_window', 'tax_loss_window_proxy']:
            _c[_k] = pd.Series(np.nan, index=df.index)

    # ═══════════════════════════════════════
    # AI. RİSK / STOP / HEDEF GEOMETRİSİ (varsayılan STOP_K=2 ATR, TGT_K=4 ATR)
    # ═══════════════════════════════════════
    _STOP_K = 2.0
    _TGT_K = 4.0
    _stop_lvl = c - atr_val * _STOP_K
    _tgt_lvl = c + atr_val * _TGT_K
    _c['stop_distance_pct'] = (c - _stop_lvl) / c.replace(0, np.nan) * 100
    _c['stop_distance_atr'] = pd.Series(_STOP_K, index=df.index, dtype=float)
    _c['target_distance_pct'] = (_tgt_lvl - c) / c.replace(0, np.nan) * 100
    _c['target_distance_atr'] = pd.Series(_TGT_K, index=df.index, dtype=float)
    _c['reward_risk_ratio'] = pd.Series(_TGT_K / _STOP_K, index=df.index, dtype=float)
    _nearest_sup = pd.concat([last_pl, sup_20], axis=1).max(axis=1)
    _c['nearest_support_stop_dist'] = (c - _nearest_sup) / atr_val.replace(0, np.nan)
    _nearest_res = pd.concat([last_pivot_high, res_20], axis=1).min(axis=1)
    _c['nearest_resistance_target_dist'] = (_nearest_res - c) / atr_val.replace(0, np.nan)
    _c['atr_stop_viability'] = (_c['nearest_support_stop_dist'] / _STOP_K).clip(0, 3)
    _abs_gap_avg = _c['gap_pct'].abs().rolling(20).mean()
    _gap_atr_avg = (_abs_gap_avg / 100) * c / atr_val.replace(0, np.nan)
    _c['gap_stop_risk'] = (_gap_atr_avg / _STOP_K).clip(0, 3)
    _size_vol = (1 / atr_pct.clip(0.5, 10))
    _size_liq = np.log1p(_c['tl_volume_20d_avg']).clip(0, 25) / 25
    _c['position_size_score'] = (_size_vol * 0.5 + _size_liq * 0.5)
    _c['trail_stop_k_suggested'] = (2.0 + (atr_pct / 5).clip(0, 2)).clip(1.5, 5)

    # ═══════════════════════════════════════
    # AK. ETKİLEŞİM (sector-bağımlı olanlar SKIP — sector OHLCV yok)
    # ═══════════════════════════════════════
    _c['rvol_x_close_position'] = rvol * _c['close_position']
    _c['rvol_x_breakout_distance'] = rvol * _c['breakout_distance_atr']
    _c['squeeze_x_volume_surge'] = _c['squeeze_on'].astype(float) * _c['vol_surge_today']
    _xu_reg = _c.get('xu100_regime_score')
    if isinstance(_xu_reg, pd.Series) and _xu_reg.notna().any():
        _c['market_regime_x_signal'] = _c['q_total'] * _xu_reg / 100
    else:
        _c['market_regime_x_signal'] = pd.Series(np.nan, index=df.index)
    _c['atr_x_gap'] = atr_pct * _c['gap_pct'].abs()
    _c['trend_x_pullback_depth'] = trend_s * _c['pullback_depth_atr'].abs()
    _c['overextension_x_volume_climax'] = _c['ema21_overextension_atr'].clip(lower=0) * _c['climax_volume_flag'].astype(float)
    _c['near_high_x_rvol'] = _c['near_52w_high'].astype(float) * rvol
    _c['tavan_x_recent_tavan'] = _c['is_tavan'].astype(float) * _c['recent_tavan_10d']
    _c['liquidity_x_volatility'] = _c['tradability_score'] * atr_pct

    # ═══════════════════════════════════════
    # AUTO-LAG1: P-AK gruplarının her ham feature'ı için _lag1 varyantı
    # T-close konvansiyon (mevcut O7 pattern uniform uygulanır)
    # ═══════════════════════════════════════
    _new_keys_p_ak = sorted(set(_c.keys()) - _keys_before_p_ak)
    for _k in _new_keys_p_ak:
        if _k.endswith('_lag1'):
            continue
        _v = _c[_k]
        if isinstance(_v, pd.Series):
            _c[f'{_k}_lag1'] = _v.shift(1)
        else:
            _c[f'{_k}_lag1'] = pd.Series(_v, index=df.index)


    return pd.DataFrame(_c, index=df.index)


# ═══════════════════════════════════════════
# MAKRO FEATURE (tüm hisseler için aynı)
# ═══════════════════════════════════════════

def compute_macro_features(xu_df, macro_dfs=None):
    """
    Makro kontekst feature'ları hesapla.

    Args:
        xu_df: XU100 DataFrame (OHLCV)
        macro_dfs: dict — {'VIX': df, 'DXY': df, 'USDTRY': df, 'SPY': df}

    Returns:
        DataFrame — günlük index, makro feature'lar
    """
    if xu_df is None or len(xu_df) < 30:
        return pd.DataFrame()

    idx = xu_df.index
    mf = pd.DataFrame(index=idx)

    xu_close = xu_df['Close']
    xu_ema21 = _ema(xu_close, 21)
    mf['xu100_above_ema21'] = (xu_close > xu_ema21).astype(int)
    mf['xu100_ret_5d'] = xu_close.pct_change(5) * 100

    if macro_dfs is None:
        macro_dfs = {}

    # VIX
    if 'VIX' in macro_dfs and macro_dfs['VIX'] is not None:
        vix = macro_dfs['VIX']['Close'].reindex(idx, method='ffill')
        mf['vix'] = vix
        mf['vix_chg_5d'] = vix.pct_change(5) * 100
    else:
        mf['vix'] = np.nan
        mf['vix_chg_5d'] = np.nan

    # DXY
    if 'DXY' in macro_dfs and macro_dfs['DXY'] is not None:
        dxy = macro_dfs['DXY']['Close'].reindex(idx, method='ffill')
        dxy_ema = _ema(dxy, 21)
        mf['dxy_trend'] = np.where(dxy > dxy_ema * 1.005, 1,
                           np.where(dxy < dxy_ema * 0.995, -1, 0))
    else:
        mf['dxy_trend'] = np.nan

    # USDTRY
    if 'USDTRY' in macro_dfs and macro_dfs['USDTRY'] is not None:
        usdtry = macro_dfs['USDTRY']['Close'].reindex(idx, method='ffill')
        mf['usdtry_chg_1d'] = usdtry.pct_change() * 100
    else:
        mf['usdtry_chg_1d'] = np.nan

    # SPY
    if 'SPY' in macro_dfs and macro_dfs['SPY'] is not None:
        spy = macro_dfs['SPY']['Close'].reindex(idx, method='ffill')
        spy_ema = _ema(spy, 21)
        mf['spy_trend'] = np.where(spy > spy_ema * 1.005, 1,
                           np.where(spy < spy_ema * 0.995, -1, 0))
    else:
        mf['spy_trend'] = np.nan

    # Macro risk score
    mf['macro_risk_score'] = 0
    if 'vix' in mf.columns:
        mf['macro_risk_score'] = mf['macro_risk_score'] + np.where(mf['vix'] > 25, -2,
                                                            np.where(mf['vix'] > 20, -1,
                                                            np.where(mf['vix'] < 15, 1, 0)))
    mf['macro_risk_score'] = mf['macro_risk_score'] + mf.get('xu100_above_ema21', 0)
    if 'spy_trend' in mf.columns:
        mf['macro_risk_score'] = mf['macro_risk_score'] + mf['spy_trend'].fillna(0).astype(int)

    return mf


# ═══════════════════════════════════════════
# REJİM YARDIMCILARI
# ═══════════════════════════════════════════

def _compute_regime_from_tpe(trend_s, part_s, exp_s):
    """Regime 0-3 from T/P/E scores (regime_transition.py exact)."""
    regime = pd.Series(0, index=trend_s.index)
    has_trend = trend_s >= 2
    regime = regime.where(~has_trend, 1)  # GRI if trend >= 2
    trend_and_part_exp = has_trend & (part_s >= 1) & (exp_s >= 1)
    regime = regime.where(~trend_and_part_exp, 2)  # TREND
    full = has_trend & (part_s >= 2) & (exp_s >= 2)
    regime = regime.where(~full, 3)  # FULL_TREND
    return regime


def _compute_exit_stage(df, ema21, st_dir, adx_slope, cmf, trend_score):
    """Exit stage 0-3 (regime_transition.py exact)."""
    c = df['Close']
    # Stage 1: close < ema21 for 2 bars OR supertrend flip
    below_ema = c < ema21
    below_ema_2 = below_ema & below_ema.shift(1, fill_value=False)
    st_flip = (st_dir == -1)
    stage1 = (below_ema_2 | st_flip).astype(int)

    # Stage 2: adx_slope < 0 for 3 bars AND cmf < 0
    slope_neg = adx_slope < 0
    slope_neg_3 = slope_neg & slope_neg.shift(1, fill_value=False) & slope_neg.shift(2, fill_value=False)
    stage2 = (slope_neg_3 & (cmf < 0)).astype(int)

    # Stage 3: trend_score < 2
    stage3 = (trend_score < 2).astype(int)

    return (stage1 + stage2 + stage3).clip(0, 3)


def _compute_days_in_trade(regime):
    """Bars since regime went to 2+."""
    in_trade = regime >= 2
    days = pd.Series(0, index=regime.index)
    count = 0
    for i in range(len(regime)):
        if in_trade.iloc[i]:
            count += 1
            days.iloc[i] = count
        else:
            count = 0
    return days


# ═══════════════════════════════════════════
# TARGET HESAPLAMA
# ═══════════════════════════════════════════

def compute_targets(df):
    """
    Forward return ve binary target hesapla.
    Leakage-free: T gününden T+1, T+3 kapanışına.

    Returns:
        DataFrame — ret_1g, ret_3g, up_1g, up_3g
    """
    c = df['Close']
    targets = pd.DataFrame(index=df.index)
    targets['ret_1g'] = (c.shift(-1) / c - 1) * 100
    targets['ret_3g'] = (c.shift(-3) / c - 1) * 100
    targets['up_1g'] = (targets['ret_1g'] > 0).astype(int)
    targets['up_3g'] = (targets['ret_3g'] > 0).astype(int)
    return targets


def compute_breakout_targets(df):
    """
    Tavan + Sert Yükseliş target'ları hesapla.
    Leakage-free: tüm target'lar shift(-N) ile future data kullanır.

    Returns:
        DataFrame — tavan_1d, tavan_3d, tavan_5d, tavan_series,
                     rally_3d, rally_5d, rally_any
    """
    c = df['Close']
    n = len(df)
    targets = pd.DataFrame(index=df.index)

    # Günlük getiriler (T+1, T+2, ... T+5)
    daily_ret = c.pct_change().shift(-1) * 100  # T+1 günlük getiri

    # Tavan target'ları (günlük getiri ≥ %9.5)
    # tavan_1d: yarın günlük getiri ≥ %9.5
    targets['tavan_1d'] = (daily_ret >= 9.5).astype(int)

    # tavan_3d: 3 gün içinde herhangi bir gün ≥ %9.5
    tavan_mask = pd.Series(0, index=df.index)
    for offset in range(1, 4):
        ret_d = (c.shift(-offset) / c.shift(-offset + 1).where(
            c.shift(-offset + 1) > 0, np.nan) - 1) * 100
        tavan_mask = tavan_mask | (ret_d >= 9.5)
    targets['tavan_3d'] = tavan_mask.astype(int)

    # tavan_5d: 5 gün içinde herhangi bir gün ≥ %9.5
    tavan_5_mask = pd.Series(0, index=df.index)
    for offset in range(1, 6):
        ret_d = (c.shift(-offset) / c.shift(-offset + 1).where(
            c.shift(-offset + 1) > 0, np.nan) - 1) * 100
        tavan_5_mask = tavan_5_mask | (ret_d >= 9.5)
    targets['tavan_5d'] = tavan_5_mask.astype(int)

    # tavan_series: yarın tavan VE ertesi gün de tavan
    ret_t1 = (c.shift(-1) / c - 1) * 100
    ret_t2 = (c.shift(-2) / c.shift(-1).where(c.shift(-1) > 0, np.nan) - 1) * 100
    targets['tavan_series'] = ((ret_t1 >= 9.5) & (ret_t2 >= 9.5)).astype(int)

    # Sert Yükseliş target'ları
    # rally_3d: 3 günlük kümülatif getiri ≥ %15
    cum_ret_3d = (c.shift(-3) / c - 1) * 100
    targets['rally_3d'] = (cum_ret_3d >= 15).astype(int)

    # rally_5d: 5 günlük kümülatif getiri ≥ %20
    cum_ret_5d = (c.shift(-5) / c - 1) * 100
    targets['rally_5d'] = (cum_ret_5d >= 20).astype(int)

    # rally_any: rally_3d OR tavan_3d (union)
    targets['rally_any'] = ((targets['rally_3d'] == 1) |
                            (targets['tavan_3d'] == 1)).astype(int)

    return targets
