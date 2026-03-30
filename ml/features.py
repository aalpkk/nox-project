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

def compute_all_features(df, xu_df=None, weekly_df=None):
    """
    Tek bir hisse için tüm screener primitive feature'ları hesapla.

    Args:
        df: DataFrame — OHLCV, Uppercase kolonlar (Close, High, Low, Open, Volume)
        xu_df: XU100 DataFrame — relatif güç için
        weekly_df: Haftalık resample — HTF feature'lar için (opsiyonel)

    Returns:
        DataFrame — aynı index, her kolon bir feature
    """
    n = len(df)
    if n < 80:
        return pd.DataFrame(index=df.index)

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume']

    feats = pd.DataFrame(index=df.index)

    # ═══════════════════════════════════════
    # A. FİYAT YAPISI (8)
    # ═══════════════════════════════════════
    feats['close'] = c
    feats['returns_1d'] = c.pct_change() * 100
    feats['returns_5d'] = c.pct_change(5) * 100
    feats['returns_10d'] = c.pct_change(10) * 100
    hl_range = (h - l).replace(0, np.nan)
    feats['close_position'] = (c - l) / hl_range  # Candle location (0-1)
    feats['gap_pct'] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100
    ema21 = _ema(c, 21)
    ema55 = _ema(c, 55)
    feats['ema21_dist_pct'] = (c / ema21.replace(0, np.nan) - 1) * 100
    feats['ema55_dist_pct'] = (c / ema55.replace(0, np.nan) - 1) * 100

    # ═══════════════════════════════════════
    # B. TREND (12)
    # ═══════════════════════════════════════
    adx, plus_di, minus_di = _calc_adx_with_di(df, 14)
    feats['adx_14'] = adx
    adx_slope = (adx - adx.shift(5)) / 5
    feats['adx_slope_5'] = adx_slope
    feats['plus_di'] = plus_di
    feats['minus_di'] = minus_di
    feats['di_spread'] = plus_di - minus_di
    feats['ema_trend_up'] = (ema21 > ema55).astype(int)
    st_dir = _calc_supertrend(df, 10, 3.0)
    feats['supertrend_dir'] = st_dir
    pmax_dir = _calc_pmax(df)
    feats['pmax_dir'] = pmax_dir
    feats['phase_above_ema21'] = (c > ema21).astype(int)

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
        feats['htf_adx'] = w_adx.reindex(df.index, method='ffill')
        feats['htf_adx_slope'] = w_adx_slope.reindex(df.index, method='ffill')
        feats['htf_trend_up'] = w_trend_up.reindex(df.index, method='ffill')
    else:
        feats['htf_adx'] = np.nan
        feats['htf_adx_slope'] = np.nan
        feats['htf_trend_up'] = np.nan

    # ═══════════════════════════════════════
    # C. MOMENTUM (10)
    # ═══════════════════════════════════════
    rsi14 = _calc_rsi(c, 14)
    feats['rsi_14'] = rsi14
    feats['rsi_2'] = _calc_rsi(c, 2)
    macd_line, macd_signal, macd_hist = _calc_macd(c)
    feats['macd_line'] = macd_line
    feats['macd_signal'] = macd_signal
    feats['macd_hist'] = macd_hist
    wt1, wt2 = _calc_wavetrend(df)
    feats['wt1'] = wt1
    feats['wt2'] = wt2
    feats['wt_bullish'] = (wt1 > wt2).astype(int)

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
    feats['squeeze_on'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
    highest = h.rolling(20).max()
    lowest = l.rolling(20).min()
    sq_mid = (highest + lowest) / 2
    feats['squeeze_mom'] = c - (sq_mid + bb_mid) / 2

    # ═══════════════════════════════════════
    # D. VOLATİLİTE (6) — cross-stock comparable only
    # ═══════════════════════════════════════
    feats['atr_14'] = atr_val
    feats['atr_pct'] = atr_val / c.replace(0, np.nan) * 100
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feats['bb_pctb'] = (c - bb_lower) / bb_range
    feats['bb_width'] = bb_range / bb_mid.replace(0, np.nan) * 100
    high_20 = h.rolling(20).max()
    feats['drawdown_20'] = (c - high_20) / high_20.replace(0, np.nan) * 100
    feats['daily_move_atr'] = (c - o) / atr_val.replace(0, np.nan)

    # ═══════════════════════════════════════
    # E. HACİM (7)
    # ═══════════════════════════════════════
    vol_sma20 = _sma(v, 20)
    vol_sma30 = _sma(v, 30)
    feats['vol_ratio_20'] = v / vol_sma20.replace(0, np.nan)
    feats['vol_ratio_30'] = v / vol_sma30.replace(0, np.nan)
    cmf20 = _calc_cmf(df, 20)
    feats['cmf_20'] = cmf20
    obv = _calc_obv(df)
    obv_sma20 = _sma(obv, 20)
    feats['obv_trend'] = (obv > obv_sma20).astype(int)
    obv_ema10 = _ema(obv, 10)
    feats['obv_slope_5'] = _linreg_slope(obv_ema10, 5)
    feats['mfi_14'] = _calc_mfi(df, 14)
    green_mask = c > o
    feats['consecutive_green'] = _consecutive_count(green_mask)

    # ═══════════════════════════════════════
    # F. YAPI / SMC (6)
    # ═══════════════════════════════════════
    swing_bias_arr, bos_prop, choch_prop = _calc_smc(df)
    feats['swing_bias'] = swing_bias_arr
    idx_arr = np.arange(n)
    bos_age = np.where(bos_prop == -999, np.nan, idx_arr - bos_prop)
    choch_age = np.where(choch_prop == -999, np.nan, idx_arr - choch_prop)
    feats['bos_age'] = bos_age
    feats['choch_age'] = choch_age

    pivot_highs = _find_pivot_highs(h, 5)
    last_pivot_high = pivot_highs.ffill()
    feats['structure_break'] = (c > last_pivot_high * 1.002).astype(int)

    pivot_lows = _find_pivot_lows(l, 5)
    last_pl = pivot_lows.ffill()
    prev_pl = pivot_lows.ffill().shift(1)
    feats['higher_low'] = (last_pl > prev_pl).astype(int)

    high_40 = h.rolling(40).max()
    feats['near_40high'] = (c > high_40 * 0.97).astype(int)

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
    feats['q_rvol_s'] = np.where(rvol >= 2, 25,
                        np.where(rvol >= 0.5, 20,   # RVOL_THRESH = 0.5
                        np.where(rvol >= 1, 10, 0))).astype(float)
    # Fix: RVOL_THRESH=0.5 is lower than 1.0, so order matters
    # regime.py: 25 if >=2, 20 if >=RVOL_THRESH(0.5), 10 if >=1, else 0
    # Since 0.5<1.0, the check >=RVOL_THRESH catches 0.5-0.99 range at 20,
    # but >=1 should be 10?  Re-reading: the chain is if-elif, so:
    # rvol>=2→25, elif rvol>=0.5→20, elif rvol>=1→10 (never reached)
    # So effectively: >=2→25, >=0.5→20, <0.5→0
    feats['q_rvol_s'] = np.where(rvol >= 2, 25,
                        np.where(rvol >= 0.5, 20, 0)).astype(float)

    feats['q_clv_s'] = np.where(clv >= 0.75, 25,
                       np.where(clv >= 0.5, 15,
                       np.where(clv >= 0.25, 5, 0))).astype(float)

    feats['q_wick_s'] = np.where(wick_ratio <= 0.15, 25,
                        np.where(wick_ratio <= 0.3, 15,
                        np.where(wick_ratio <= 0.5, 5, 0))).astype(float)

    feats['q_range_s'] = np.where(range_atr >= 1.2, 25,
                         np.where(range_atr >= 0.8, 15,
                         np.where(range_atr >= 0.5, 5, 0))).astype(float)

    feats['q_total'] = feats['q_rvol_s'] + feats['q_clv_s'] + feats['q_wick_s'] + feats['q_range_s']

    # ═══════════════════════════════════════
    # H. NW BREADTH PRİMİTİVLERİ (5) — nox_v3_signals.py exact
    # ═══════════════════════════════════════
    rsi_min10 = rsi14.rolling(10).min()

    # Sub-component 1: RSI thrust (50pt)
    br_rsi_thrust = ((rsi_min10 < 30) & (rsi14 > 55)).astype(int)
    feats['br_rsi_thrust'] = br_rsi_thrust

    # Sub-component 2: RSI gradual (25pt)
    rsi_delta5 = rsi14 - rsi14.shift(5)
    br_rsi_gradual = ((rsi_delta5 > 15) & (rsi14 > 40) & (rsi_min10 < 35)).astype(int)
    feats['br_rsi_gradual'] = br_rsi_gradual

    # Sub-component 3: AD proxy (25pt)
    consec_green = _consecutive_count(green_mask)
    br_ad_proxy = ((consec_green >= 3) & (v > vol_sma20 * 1.3)).astype(int)
    feats['br_ad_proxy'] = br_ad_proxy

    # Sub-component 4: EMA reclaim (15pt)
    br_ema_reclaim = ((c > ema21) & (c.shift(1) < ema21.shift(1))).astype(int)
    feats['br_ema_reclaim'] = br_ema_reclaim

    # Composite breadth score
    br_score = (br_rsi_thrust * 50 + br_rsi_gradual * 25 +
                br_ad_proxy * 25 + br_ema_reclaim * 15).clip(0, 100).astype(float)
    feats['br_score'] = br_score

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

    feats['rg_slope_score'] = rg_slope_score
    feats['rg_di_score'] = rg_di_score
    feats['rg_ema_above'] = rg_ema_above
    feats['rg_adx_rebound'] = rg_adx_rebound
    feats['rg_was_trending'] = was_trending

    # Composite regime score
    rg_raw = (rg_slope_score + rg_di_score + rg_ema_above + rg_adx_rebound).clip(0, 100)
    rg_score = rg_raw * was_trending
    feats['rg_score'] = rg_score

    # Gate
    feats['gate_open'] = ((br_score >= 25) | (rg_score >= 25)).astype(int)

    # ═══════════════════════════════════════
    # J. SELL SEVERITY PRİMİTİVLERİ (5) — nox_v3_signals.py exact
    # ═══════════════════════════════════════
    red_mask = c < o
    red_count = _consecutive_count(red_mask)
    feats['red_count'] = red_count
    feats['drawdown_20_pct'] = (c - high_20) / high_20.replace(0, np.nan) * 100
    high_close_5 = c.rolling(5).max()
    decline_5d_atr = (c - high_close_5) / atr_val.replace(0, np.nan)
    feats['decline_5d_atr'] = decline_5d_atr

    # Composite severity
    severity = pd.Series(0, index=df.index)
    sev1 = (feats['daily_move_atr'] < -1.5) | (red_count >= 3) | (feats['drawdown_20_pct'] < -8)
    severity = severity.where(~sev1, 1)
    sev2 = (feats['daily_move_atr'] < -2.5) | ((red_count >= 5) & (decline_5d_atr < -3))
    severity = severity.where(~sev2, 2)
    sev3 = (feats['daily_move_atr'] < -3.5) | (feats['drawdown_20_pct'] < -12)
    severity = severity.where(~sev3, 3)
    feats['sell_severity'] = severity.astype(int)

    # Pivot delta
    feats['pivot_delta_pct'] = (c - last_pl) / last_pl.replace(0, np.nan) * 100

    # ═══════════════════════════════════════
    # K. RT TPE PRİMİTİVLERİ (9+3) — regime_transition.py exact
    # ═══════════════════════════════════════

    # -- Trend sub-components --
    rt_ema_bull = (c > ema21).astype(int)
    rt_st_bull = (st_dir == 1).astype(int)
    feats['rt_ema_bull'] = rt_ema_bull
    feats['rt_st_bull'] = rt_st_bull

    # Weekly trend up (reindexed)
    if weekly_df is not None and len(weekly_df) >= 20:
        w_ema21_rt = _ema(weekly_df['Close'], 21)
        w_trend_rising = (w_ema21_rt > w_ema21_rt.shift(1)).astype(int)
        feats['rt_wk_trend_up'] = w_trend_rising.reindex(df.index, method='ffill')
    else:
        feats['rt_wk_trend_up'] = np.nan

    # trend_score = ema_bull + st_bull + wk_trend_up
    trend_s = rt_ema_bull + rt_st_bull + feats['rt_wk_trend_up'].fillna(0).astype(int)
    trend_s = trend_s.clip(0, 3)

    # -- Participation sub-components --
    rt_cmf_pos = (cmf20 > 0).astype(int)
    rt_rvol_high = (rvol >= 1.0).astype(int)
    obv_slope = _linreg_slope(obv_ema10, 5)
    rt_obv_slope_pos = (obv_slope > 0).astype(int)
    feats['rt_cmf_pos'] = rt_cmf_pos
    feats['rt_rvol_high'] = rt_rvol_high
    feats['rt_obv_slope_pos'] = rt_obv_slope_pos

    part_s = (rt_cmf_pos + rt_rvol_high + rt_obv_slope_pos).clip(0, 3)

    # -- Expansion sub-components --
    rt_adx_slope_pos = (adx_slope > 0).astype(int)
    atr_sma20 = _sma(atr_val, 20)
    rt_atr_expanding = (atr_val > atr_sma20 * 1.05).astype(int)
    rt_di_bull = ((plus_di - minus_di) > 5).astype(int)
    feats['rt_adx_slope_pos'] = rt_adx_slope_pos
    feats['rt_atr_expanding'] = rt_atr_expanding
    feats['rt_di_bull'] = rt_di_bull

    exp_s = (rt_adx_slope_pos + rt_atr_expanding + rt_di_bull).clip(0, 3)

    # Composite scores
    feats['trend_score'] = trend_s
    feats['participation_score'] = part_s
    feats['expansion_score'] = exp_s
    feats['tpe_total'] = trend_s + part_s + exp_s

    # ═══════════════════════════════════════
    # L. RT REGIME + ENTRY + OE + EXIT (6) — regime_transition.py exact
    # ═══════════════════════════════════════

    # Regime label
    regime = _compute_regime_from_tpe(trend_s, part_s, exp_s)
    feats['regime_score'] = regime

    # Entry score (CORRECTED: from run_regime_transition.py)
    # Low vol: ATR% < 3.0
    # Early entry: ADX slope < 0
    # Growth room: regime <= 2
    # No pump: RVOL < 2.0
    entry_s = pd.Series(0, index=df.index)
    entry_s = entry_s + (feats['atr_pct'] < 3.0).astype(int)
    entry_s = entry_s + (adx_slope < 0).astype(int)
    entry_s = entry_s + (regime <= 2).astype(int)
    entry_s = entry_s + (rvol < 2.0).astype(int)
    feats['entry_score'] = entry_s.clip(0, 4)

    # OE score (CORRECTED: from regime_transition.py)
    # RSI > 80, close > BB upper, 5-bar mom > 8%, EMA dist > 5%
    oe_s = pd.Series(0, index=df.index)
    oe_s = oe_s + (rsi14 > 80).astype(int)
    oe_s = oe_s + (c > bb_upper).astype(int)
    mom_5 = c.pct_change(5) * 100
    oe_s = oe_s + (mom_5 > 8).astype(int)
    ema_dist_abs = ((c / ema21.replace(0, np.nan) - 1) * 100).abs()
    oe_s = oe_s + (ema_dist_abs > 5).astype(int)
    feats['oe_score'] = oe_s.clip(0, 4)

    # Pullback conditions (for H+PB detection)
    feats['pb_ema_dist'] = ema_dist_abs
    feats['pb_rsi_low'] = (rsi14 <= 55).astype(int)

    # Exit stage
    feats['exit_stage'] = _compute_exit_stage(df, ema21, st_dir, adx_slope, cmf20, trend_s)

    # Days in trade
    feats['days_in_trade'] = _compute_days_in_trade(regime)

    # ═══════════════════════════════════════
    # M. TAVAN PRİMİTİVLERİ (6)
    # ═══════════════════════════════════════
    prev_close = c.shift(1)
    limit_up = prev_close * 1.10
    feats['is_tavan'] = (c >= limit_up * 0.995).astype(int)
    feats['tavan_streak'] = _consecutive_count(feats['is_tavan'].astype(bool))
    # close_to_high: tavan-specific (close/high, NOT candle location value)
    feats['close_to_high'] = c / h.replace(0, np.nan)
    feats['tavan_locked'] = ((feats['is_tavan'] == 1) & (feats['close_to_high'] >= 0.95)).astype(int)
    feats['hit_tavan_intraday'] = (h >= limit_up * 0.995).astype(int)
    feats['recent_tavan_10d'] = feats['is_tavan'].rolling(10, min_periods=1).sum()

    # ═══════════════════════════════════════
    # N. RELATİF GÜÇ (3) — regime.py exact (RS_LEN1=10, RS_LEN2=60)
    # ═══════════════════════════════════════
    if xu_df is not None and 'Close' in xu_df.columns and len(xu_df) >= 65:
        xu_close = xu_df['Close'].reindex(df.index, method='ffill')
        stock_ret_10 = c.pct_change(10) * 100
        bench_ret_10 = xu_close.pct_change(10) * 100
        stock_ret_60 = c.pct_change(60) * 100
        bench_ret_60 = xu_close.pct_change(60) * 100
        feats['rs_10'] = stock_ret_10 - bench_ret_10
        feats['rs_60'] = stock_ret_60 - bench_ret_60
        feats['rs_composite'] = feats['rs_10'] * 0.6 + feats['rs_60'] * 0.4
    else:
        feats['rs_10'] = np.nan
        feats['rs_60'] = np.nan
        feats['rs_composite'] = np.nan

    # ═══════════════════════════════════════
    # O. PRE-BREAKOUT / BİRİKİM→BREAKOUT (22)
    # 20 gün birikim penceresine dayalı feature'lar
    # ═══════════════════════════════════════

    # O1. Range Sıkışması (4)
    range_5 = h.rolling(5).max() - l.rolling(5).min()
    range_20 = h.rolling(20).max() - l.rolling(20).min()
    range_40 = h.rolling(40).max() - l.rolling(40).min()
    feats['range_contraction_5_20'] = range_5 / range_20.replace(0, np.nan)
    feats['range_contraction_5_40'] = range_5 / range_40.replace(0, np.nan)
    atr_5 = _rma(_true_range(df), 5)
    feats['atr_contraction_5_20'] = atr_5 / atr_val.replace(0, np.nan)
    bb_width_raw = bb_range / bb_mid.replace(0, np.nan)
    feats['bb_width_pctile_20'] = bb_width_raw.rolling(60, min_periods=20).rank(pct=True)

    # O2. Hacim Birikim Paterni (5)
    vol_sma5 = _sma(v, 5)
    feats['vol_dryup_ratio'] = vol_sma5 / vol_sma20.replace(0, np.nan)
    feats['vol_surge_today'] = v / vol_sma5.replace(0, np.nan)
    feats['vol_acceleration'] = (vol_sma5 - vol_sma5.shift(5)) / vol_sma5.shift(5).replace(0, np.nan)
    feats['tl_volume_20d_avg'] = _sma(c * v, 20)
    vol_dryup = feats['vol_dryup_ratio'].clip(upper=2)
    vol_spike = feats['vol_surge_today'].clip(upper=5)
    feats['vol_pattern_score'] = (2 - vol_dryup) * 2 + np.where(vol_spike > 2, vol_spike * 3, 0)

    # O3. Momentum Oluşumu (4)
    feats['rsi_momentum_5d'] = rsi14 - rsi14.shift(5)
    feats['macd_hist_accel'] = macd_hist - macd_hist.shift(3)
    higher_close_mask = c > c.shift(1)
    feats['consecutive_higher_close'] = _consecutive_count(higher_close_mask)
    high_20d = h.rolling(20).max()
    feats['close_vs_20d_high_pct'] = (c - high_20d) / high_20d.replace(0, np.nan) * 100

    # O4. Yapısal Yakınlık (4)
    high_52w = h.rolling(252, min_periods=60).max()
    feats['dist_to_52w_high_pct'] = (c - high_52w) / high_52w.replace(0, np.nan) * 100
    feats['dist_to_20d_high_pct'] = feats['close_vs_20d_high_pct']  # same calc
    feats['near_52w_high'] = (c >= high_52w * 0.95).astype(int)
    low_20d = l.rolling(20).min()
    feats['price_range_position_20'] = (c - low_20d) / (high_20d - low_20d).replace(0, np.nan)

    # O5. Yakın Iskalama / Öncü Sinyal (3)
    daily_ret_pct = (c / c.shift(1) - 1) * 100
    feats['near_tavan_miss'] = ((daily_ret_pct >= 7) & (daily_ret_pct < 9.5)).astype(int)
    feats['recent_near_tavan_5d'] = feats['near_tavan_miss'].rolling(5, min_periods=1).sum()
    feats['max_daily_ret_5d'] = daily_ret_pct.rolling(5, min_periods=1).max()

    # O6. Relatif Güç İvmesi (2)
    if xu_df is not None and 'Close' in xu_df.columns and len(xu_df) >= 65:
        feats['rs_acceleration'] = feats['rs_10'] - feats['rs_10'].shift(5)
    else:
        feats['rs_acceleration'] = np.nan
    # market_tavan_count_10d: piyasa geneli, dışarıdan verilmeli — NaN olarak bırak
    feats['market_tavan_count_10d'] = np.nan

    # O7. Lag-1 Güvenli Feature'lar (8) — leakage-free versiyonlar
    feats['returns_1d_lag1'] = feats['returns_1d'].shift(1)
    feats['close_position_lag1'] = feats['close_position'].shift(1)
    feats['close_to_high_lag1'] = feats['close_to_high'].shift(1)
    daily_ret_pct_lag = daily_ret_pct.shift(1)
    feats['max_daily_ret_5d_lag1'] = daily_ret_pct_lag.rolling(5, min_periods=1).max()
    near_miss_lag = feats['near_tavan_miss'].shift(1)
    feats['recent_near_tavan_5d_lag1'] = near_miss_lag.rolling(5, min_periods=1).sum()
    feats['vol_surge_yesterday'] = v.shift(1) / _sma(v, 5).shift(1).replace(0, np.nan)
    feats['consecutive_green_lag1'] = feats['consecutive_green'].shift(1)
    is_tavan_lag = feats['is_tavan'].shift(1)
    feats['recent_tavan_10d_lag1'] = is_tavan_lag.rolling(10, min_periods=1).sum()

    return feats


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
