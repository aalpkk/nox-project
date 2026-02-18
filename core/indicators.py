"""
BIST Screener — Teknik Göstergeler
Her iki modülün (Rejim V3 + DIP) ortak kullandığı fonksiyonlar.
"""
import numpy as np
import pandas as pd
from core.config import (
    ADX_LEN, ATR_LEN, WT_CH_LEN, WT_AVG_LEN, WT_MA_LEN, WT_LOOKBACK,
    PMAX_ATR_LEN, PMAX_ATR_MULT, PMAX_MA_LEN, PMAX_MA_TYPE,
    SMC_INTERNAL_LEN, BOS_LOOKBACK, BOS_TIGHT, CHOCH_TIGHT,
    SQ_LEN, SQ_MULT_BB, SQ_MULT_KC,
    BB_LEN, BB_MULT, DONCH_LEN,
    ST_LEN, ST_MULT,
    OVEREXT_WT1_THRESH, OVEREXT_RSI_THRESH, OVEREXT_MOMENTUM_PCT,
    OVEREXT_MOMENTUM_DAYS,
)


# ── Temel MA'lar ──

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def wma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def rma(series, period):
    """Pine Script'in ta.rma karşılığı (Wilder's smoothing)."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


# ── True Range / ATR ──

def true_range(df):
    h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def calc_atr(df, period=ATR_LEN):
    return rma(true_range(df), period)

def calc_atr_sma(df, period=ATR_LEN):
    """SMA-based ATR (Pink V2 uses this)."""
    return true_range(df).rolling(period).mean()


def calc_atr_percentile(df, atr_period=ATR_LEN, window=100):
    """100-bar rolling ATR percentile (0-1 arası). Panic filter için."""
    atr_s = calc_atr(df, atr_period)
    return atr_s.rolling(window, min_periods=max(window // 2, 1)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).fillna(0.5)


# ── RSI ──

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_rsi_sma(series, period):
    """SMA-based RSI (Pink V2 uses this)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ── ADX ──

def calc_adx(df, length=ADX_LEN):
    """ADX hesaplama — Pine Script calcADX ile birebir."""
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr = rma(tr, length)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), length) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = rma(dx, length)
    return adx

def calc_adx_ema(df, length=ADX_LEN):
    """EMA-based ADX (Pink V2's original implementation)."""
    h, l = df['High'], df['Low']
    up = h - h.shift(1)
    dn = l.shift(1) - l
    pos = np.where((up > dn) & (up > 0), up, 0.0)
    neg = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr = tr.ewm(alpha=1/length, min_periods=length).mean()
    dip = pd.Series(pos, index=df.index).ewm(alpha=1/length, min_periods=length).mean() / atr * 100
    dim = pd.Series(neg, index=df.index).ewm(alpha=1/length, min_periods=length).mean() / atr * 100
    dx = ((dip - dim).abs() / (dip + dim) * 100).replace([np.inf, -np.inf], 0)
    return dx.ewm(alpha=1/length, min_periods=length).mean()


# ── SuperTrend ──

def calc_supertrend(df, period=ST_LEN, mult=ST_MULT):
    atr = calc_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    st_dir = pd.Series(1, index=df.index)
    final_up = up.copy()
    final_dn = dn.copy()
    for i in range(1, len(df)):
        if up.iloc[i] > final_up.iloc[i-1]:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = final_up.iloc[i-1] if df['Close'].iloc[i-1] > final_up.iloc[i-1] else up.iloc[i]
        if dn.iloc[i] < final_dn.iloc[i-1]:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = final_dn.iloc[i-1] if df['Close'].iloc[i-1] < final_dn.iloc[i-1] else dn.iloc[i]
        prev_dir = st_dir.iloc[i-1]
        if prev_dir == -1 and df['Close'].iloc[i] > final_dn.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif prev_dir == 1 and df['Close'].iloc[i] < final_up.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = prev_dir
    return st_dir  # 1 = up, -1 = down


# ── WaveTrend ──

def calc_wavetrend(df):
    """WaveTrend + bearish invalidation (v3 mantığı)."""
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    esa_val = ema(hlc3, WT_CH_LEN)
    d = ema((hlc3 - esa_val).abs(), WT_CH_LEN)
    ci = (hlc3 - esa_val) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, WT_AVG_LEN)
    wt2 = sma(wt1, WT_MA_LEN)
    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_dn = (wt2 > wt1) & (wt2.shift(1) <= wt1.shift(1))
    n = len(df)
    wt_recent = pd.Series(False, index=df.index)
    wt_killed = pd.Series(False, index=df.index)
    for i in range(WT_LOOKBACK, n):
        killed = False
        for j in range(WT_LOOKBACK + 1):
            if i - j >= 0 and cross_dn.iloc[i - j]:
                killed = True
                break
        wt_killed.iloc[i] = killed
        if not killed:
            for j in range(WT_LOOKBACK + 1):
                if i - j >= 0 and cross_up.iloc[i - j]:
                    wt_recent.iloc[i] = True
                    break
    return {
        'wt1': wt1, 'wt2': wt2,
        'cross_up': cross_up, 'cross_dn': cross_dn,
        'wt_recent': wt_recent, 'wt_killed': wt_killed,
        'wt_bullish': wt1 > wt2,
    }


# ── PMAX ──

def calc_pmax(df):
    """PMAX — Kıvanç Özbilgiç."""
    src = (df['High'] + df['Low']) / 2
    if PMAX_MA_TYPE == "EMA":
        ma_val = ema(src, PMAX_MA_LEN)
    elif PMAX_MA_TYPE == "WMA":
        ma_val = wma(src, PMAX_MA_LEN)
    else:
        ma_val = sma(src, PMAX_MA_LEN)
    atr = calc_atr(df, PMAX_ATR_LEN)
    n = len(df)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    pmax_dir = np.ones(n)
    for i in range(1, n):
        ls = ma_val.iloc[i] - PMAX_ATR_MULT * atr.iloc[i]
        ss = ma_val.iloc[i] + PMAX_ATR_MULT * atr.iloc[i]
        long_stop[i] = max(ls, long_stop[i-1]) if not np.isnan(long_stop[i-1]) and ma_val.iloc[i] > long_stop[i-1] else ls
        short_stop[i] = min(ss, short_stop[i-1]) if not np.isnan(short_stop[i-1]) and ma_val.iloc[i] < short_stop[i-1] else ss
        prev = pmax_dir[i-1]
        if prev == -1 and ma_val.iloc[i] > short_stop[i-1]:
            pmax_dir[i] = 1
        elif prev == 1 and ma_val.iloc[i] < long_stop[i-1]:
            pmax_dir[i] = -1
        else:
            pmax_dir[i] = prev
    pmax_long = pmax_dir == 1
    return {'pmax_dir': pmax_dir, 'pmax_long': pmax_long, 'pmax_val': long_stop}


# ── SMC (BOS / CHoCH) ──

def calc_smc(df):
    """SMC BOS / CHoCH — bullish yapı kırılımı, bearish invalidation dahil."""
    n = len(df)
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    k = SMC_INTERNAL_LEN
    last_sh = np.nan
    last_sl = np.nan
    swing_bias = 0
    swing_bias_arr = np.zeros(n, dtype=int)
    bull_bos = np.zeros(n, dtype=bool)
    bull_choch = np.zeros(n, dtype=bool)
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
                    bull_choch[i] = True
                    choch_bar[i] = i
                bull_bos[i] = True
                bos_bar[i] = i
                swing_bias = 1
        if not np.isnan(last_sl) and i > 0:
            if c[i] < last_sl and c[i-1] >= last_sl:
                swing_bias = -1
        swing_bias_arr[i] = swing_bias
    last_bos_bar = -999
    last_choch_bar = -999
    bos_bar_prop = np.full(n, -999)
    choch_bar_prop = np.full(n, -999)
    for i in range(n):
        if bull_bos[i]:
            last_bos_bar = i
        if bull_choch[i]:
            last_choch_bar = i
        bos_bar_prop[i] = last_bos_bar
        choch_bar_prop[i] = last_choch_bar
    return {
        'bull_bos': bull_bos, 'bull_choch': bull_choch,
        'bos_bar': bos_bar_prop, 'choch_bar': choch_bar_prop,
        'swing_bias': swing_bias,
        'swing_bias_arr': swing_bias_arr,
    }


def calc_order_blocks(df):
    """Bearish OB (direnç) tespiti."""
    n = len(df)
    h, l, o, c = df['High'].values, df['Low'].values, df['Open'].values, df['Close'].values
    k = SMC_INTERNAL_LEN
    ob_top, ob_bot = np.nan, np.nan
    for i in range(k, n - k):
        is_ph = h[i] == np.max(h[max(0, i-k):i+k+1])
        if is_ph:
            for j in range(i-1, max(i-k-10, 0), -1):
                if c[j] < o[j]:
                    ob_top, ob_bot = h[j], l[j]
                    break
    if not np.isnan(ob_top) and c[-1] > ob_top:
        ob_top, ob_bot = np.nan, np.nan
    return {'ob_resist_top': ob_top, 'ob_resist_bot': ob_bot}


# ── MACD ──

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Resample ──

def resample_weekly(df):
    wdf = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return wdf


# ── USD Çevirme ──

def to_usd(df, usd_df):
    """TL fiyatları USD'ye çevir."""
    if usd_df is None or len(usd_df) == 0:
        return None
    usd_aligned = usd_df['Close'].reindex(df.index, method='ffill')
    aligned = pd.DataFrame({
        'Close': df['Close'], 'High': df['High'], 'Low': df['Low'],
        'Open': df['Open'], 'Volume': df['Volume'],
        'usd': usd_aligned
    }).dropna()
    if len(aligned) < 55:
        return None
    usd_df_out = pd.DataFrame({
        'Close': aligned['Close'] / aligned['usd'],
        'High': aligned['High'] / aligned['usd'],
        'Low': aligned['Low'] / aligned['usd'],
        'Open': aligned['Open'] / aligned['usd'],
        'Volume': aligned['Volume'],
    }, index=aligned.index)
    return usd_df_out


# ── CMF (Chaikin Money Flow) ──

def calc_cmf(df, period=20):
    """
    Chaikin Money Flow.
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    CMF = sum(CLV * Volume, period) / sum(Volume, period)
    Pozitif = birikim (alım baskısı), Negatif = dağıtım (satış baskısı).
    """
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
    hl_range = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / hl_range
    clv = clv.fillna(0)
    money_flow_vol = clv * v
    cmf = money_flow_vol.rolling(period).sum() / v.rolling(period).sum()
    return cmf.fillna(0)


# ═══════════════════════════════════════════
# OVEREXTENDED UYARI (metadata-only, sinyal kararını ETKİLEMEZ)
# ═══════════════════════════════════════════

def calc_overextended(df, wt_data=None):
    """
    Overextended skoru hesapla (0-5). Sinyal kararını etkilemez, metadata.
    +1: WT1 > 40
    +1: RSI(14) > 70
    +1: Fiyat > Üst BB
    +1: Son 5 günde >8% yükseliş
    +1 ekstra risk: WT1 düşüşte (wt1 < wt1[-1])
    """
    score = 0
    tags = []
    c = df['Close']
    n = len(df)

    # WT1 > 40
    if wt_data is not None:
        wt1_val = float(wt_data['wt1'].iloc[-1])
        if wt1_val > OVEREXT_WT1_THRESH:
            score += 1
            tags.append(f"WT1:{wt1_val:.0f}")
        # WT1 düşüşte
        if n >= 2:
            wt1_prev = float(wt_data['wt1'].iloc[-2])
            if wt1_val < wt1_prev:
                score += 1
                tags.append("WT1↓")

    # RSI(14) > 70
    rsi14 = calc_rsi(c, 14)
    rsi_val = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50
    if rsi_val > OVEREXT_RSI_THRESH:
        score += 1
        tags.append(f"RSI:{rsi_val:.0f}")

    # Fiyat > Üst BB
    bb_basis = sma(c, BB_LEN)
    bb_dev = c.rolling(BB_LEN).std() * BB_MULT
    bb_upper = bb_basis + bb_dev
    if not np.isnan(bb_upper.iloc[-1]) and c.iloc[-1] > bb_upper.iloc[-1]:
        score += 1
        tags.append("BB↑")

    # Son 5 günde >8% yükseliş
    if n > OVEREXT_MOMENTUM_DAYS:
        mom = (c.iloc[-1] - c.iloc[-1 - OVEREXT_MOMENTUM_DAYS]) / c.iloc[-1 - OVEREXT_MOMENTUM_DAYS] * 100
        if mom > OVEREXT_MOMENTUM_PCT:
            score += 1
            tags.append(f"MOM:{mom:.1f}%")

    return {
        'overext_score': score,
        'overext_tags': tags,
        'overext_warning': score >= 3,
    }


# ═══════════════════════════════════════════
# XU100 MARKET STATE (Risk-On / Sideways / Risk-Off)
# ═══════════════════════════════════════════

def calc_xu100_market_state(xu_df, ema_len=50):
    """
    XU100 bazında piyasa durumu tespit et.
    risk_on  = xu_above_ema50 AND slope>0 AND weekly_st_up
    risk_off = NOT xu_above_ema50 AND slope<0 AND NOT weekly_st_up
    sideways = ne risk_on ne risk_off
    """
    if xu_df is None or len(xu_df) < ema_len + 20:
        return {'risk_on': True, 'sideways': False, 'risk_off': False, 'weekly_st_up': True}

    c = xu_df['Close']
    xu_ema = ema(c, ema_len)
    xu_above_ema = float(c.iloc[-1]) > float(xu_ema.iloc[-1])

    slope_window = 10
    if len(xu_ema) > slope_window:
        slope = (float(xu_ema.iloc[-1]) - float(xu_ema.iloc[-1 - slope_window])) / float(xu_ema.iloc[-1 - slope_window]) * 100
    else:
        slope = 0.0
    slope_pos = slope > 0

    wdf = resample_weekly(xu_df)
    if len(wdf) >= 20:
        wdf_st = calc_supertrend(wdf, ST_LEN, ST_MULT)
        weekly_st_up = bool(wdf_st.iloc[-1] == 1)
    else:
        weekly_st_up = True

    risk_on = xu_above_ema and slope_pos and weekly_st_up
    risk_off = (not xu_above_ema) and (slope < 0) and (not weekly_st_up)
    sideways = not risk_on and not risk_off

    return {
        'risk_on': risk_on,
        'sideways': sideways,
        'risk_off': risk_off,
        'weekly_st_up': weekly_st_up,
    }
