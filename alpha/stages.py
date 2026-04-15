"""
Alpha Pipeline — Sinyal Aşamaları (1-3)
========================================
Aşama 1: Momentum Yakalama (WaveTrend, Squeeze, SuperTrend)
Aşama 2: Eğim Doğrulama (fiyat slope + sinyal slope)
Aşama 3: Teknik Onay (ADX, CMF, mum formasyonu, RSI)
"""

import numpy as np
import pandas as pd

from core.indicators import (
    calc_wavetrend, calc_cmf, calc_adx, calc_supertrend,
    calc_atr, calc_rsi, ema, sma,
)
from alpha.config import (
    WT_CROSS_LOOKBACK, WT1_ZONE_LO, WT1_ZONE_HI,
    SBT_SQUEEZE_COMPLEMENT, ST_FLIP_LOOKBACK,
    BB_LENGTH, BB_MULT, BB_WIDTH_THRESH,
    ATR_LENGTH, ATR_SMA_LENGTH, ATR_SQUEEZE_RATIO,
    MIN_SQUEEZE_BARS, MAX_SQUEEZE_BARS, IMPULSE_ATR_MULT,
    VOL_SMA_LENGTH, VOL_MULT,
    SLOPE_EMA_FAST, SLOPE_EMA_SLOW, SLOPE_WINDOW,
    SLOPE_MIN_THRESHOLD, SIGNAL_SLOPE_WINDOW, SLOPE_MIN_CHECKS,
    ADX_MIN, CMF_MIN, RSI_LO, RSI_HI,
    CANDLE_LOOKBACK, CONFIRMATION_MIN_SCORE,
    MIN_DATA_DAYS,
)


# ═══════════════════════════════════════════
# Yardımcı: Mum Formasyonları (Uppercase)
# ═══════════════════════════════════════════

def _detect_bullish_candles(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Bullish mum formasyonlarını tespit et (reversal_v2 mantığı, dip_context yok).

    Returns:
        (scores, names) — scores: 0-37.5, names: pattern adı veya ''
    """
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    v = df['Volume']

    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan).fillna(1e-10)
    body_pct = body / candle_range
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    is_green = c > o
    is_red = c < o

    vol_sma20 = sma(v, 20)
    vol_high = v > (vol_sma20 * 1.2)

    scores = pd.Series(0.0, index=df.index)
    names = pd.Series('', index=df.index)

    # CEKIC (Hammer)
    cekic = (
        (lower_shadow >= 2 * body) &
        (upper_shadow < 0.5 * body) &
        (body_pct < 0.35)
    )
    scores = scores.where(~cekic, 25.0)
    names = names.where(~cekic, 'CEKIC')

    # ENGULF (Bullish Engulfing)
    engulf = (
        is_green &
        is_red.shift(1) &
        (c > o.shift(1)) &
        (o < c.shift(1))
    )
    scores = scores.where(~engulf, 25.0)
    names = names.where(~engulf, 'ENGULF')

    # SABAH_YILDIZI (Morning Star)
    big_red_2ago = is_red.shift(2) & (body.shift(2) > candle_range.shift(2) * 0.5)
    small_body_1ago = body_pct.shift(1) < 0.30
    big_green_now = is_green & (body > candle_range * 0.5)
    sabah = big_red_2ago & small_body_1ago & big_green_now
    mask_sabah = sabah & (scores == 0)
    scores = scores.where(~mask_sabah, 20.0)
    names = names.where(~mask_sabah, 'SABAH_YILDIZI')

    # DOJI_YILDIZ
    doji_star = body_pct < 0.10
    mask_doji = doji_star & (scores == 0)
    scores = scores.where(~mask_doji, 15.0)
    names = names.where(~mask_doji, 'DOJI_YILDIZ')

    # TOPAC (Spinning Top)
    topac = (body_pct < 0.35) & (lower_shadow > body) & (upper_shadow > body)
    mask_topac = topac & (scores == 0)
    scores = scores.where(~mask_topac, 10.0)
    names = names.where(~mask_topac, 'TOPAC')

    # Hacim teyidi: 1.5× bonus
    scores = scores * np.where(vol_high & (scores > 0), 1.5, 1.0)

    return scores, names


# ═══════════════════════════════════════════
# Yardımcı: Squeeze Breakout (SBT mantığı)
# ═══════════════════════════════════════════

def _detect_squeeze_breakout(df: pd.DataFrame) -> dict:
    """SBT squeeze→box→breakout tespiti (inline, import yok).

    Returns:
        dict: {is_breakout, squeeze_bars, strength, bars_ago}
    """
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    n = len(df)

    # BB width
    bb_sma = sma(c, BB_LENGTH)
    bb_std = c.rolling(BB_LENGTH).std()
    bb_width = (2 * BB_MULT * bb_std) / bb_sma.replace(0, np.nan)
    bb_width_sma = sma(bb_width, BB_LENGTH)

    # ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_LENGTH).mean()
    atr_sma = atr.rolling(ATR_SMA_LENGTH).mean()

    # Volume SMA
    vol_sma = sma(v, VOL_SMA_LENGTH)

    # EMA50 (HTF proxy)
    ema50 = ema(c, 50)

    # Squeeze koşulu
    sq_bb = bb_width < bb_width_sma * BB_WIDTH_THRESH
    sq_atr = atr < atr_sma * ATR_SQUEEZE_RATIO
    squeeze = sq_bb & sq_atr

    result = {'is_breakout': False, 'squeeze_bars': 0, 'strength': 0, 'bars_ago': 999}

    # Squeeze dönemlerini bul
    squeezes = []
    in_sq = False
    sq_start = 0
    for i in range(n):
        if pd.isna(squeeze.iloc[i]):
            in_sq = False
            continue
        if squeeze.iloc[i]:
            if not in_sq:
                sq_start = i
                in_sq = True
            elif (i - sq_start + 1) > MAX_SQUEEZE_BARS:
                sq_len = MAX_SQUEEZE_BARS
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append({'start': sq_start, 'end': sq_start + MAX_SQUEEZE_BARS - 1})
                sq_start = i
        else:
            if in_sq:
                sq_len = i - sq_start
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append({'start': sq_start, 'end': i - 1})
                in_sq = False

    if not squeezes:
        return result

    last_sq = squeezes[-1]
    sq_s, sq_e = last_sq['start'], last_sq['end']
    box_top = h.iloc[sq_s:sq_e + 1].max()

    # Son 5 bar içinde kırılma ara
    for i in range(max(sq_e + 1, n - 5), n):
        c_i = c.iloc[i]
        o_i = o.iloc[i]
        body_i = abs(c_i - o_i)
        atr_i = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        vol_i = v.iloc[i]
        vol_sma_i = vol_sma.iloc[i] if not pd.isna(vol_sma.iloc[i]) else 0

        if atr_i <= 0:
            continue

        impulse_ok = body_i > atr_i * IMPULSE_ATR_MULT
        vol_ok = vol_sma_i > 0 and vol_i > vol_sma_i * VOL_MULT
        htf_ok = c_i > ema50.iloc[i] if not pd.isna(ema50.iloc[i]) else False

        if c_i > box_top and c_i > o_i and impulse_ok:
            strength = sum([impulse_ok, vol_ok, htf_ok, (sq_e - sq_s + 1) >= 6])
            result = {
                'is_breakout': True,
                'squeeze_bars': sq_e - sq_s + 1,
                'strength': strength,
                'bars_ago': n - 1 - i,
            }
            break

    return result


# ═══════════════════════════════════════════
# AŞAMA 1: Momentum Yakalama
# ═══════════════════════════════════════════

def stage1_momentum(df: pd.DataFrame) -> dict:
    """Momentum adayı mı? 3 sinyal yolu: WaveTrend, Squeeze, SuperTrend.

    Returns:
        {is_candidate, momentum_type, wt1, wt2, score}
    """
    result = {
        'is_candidate': False,
        'momentum_type': None,
        'wt1': np.nan,
        'wt2': np.nan,
        'score': 0.0,
    }

    if len(df) < MIN_DATA_DAYS:
        return result

    # A) WaveTrend cross-up
    wt = calc_wavetrend(df)
    wt1_now = float(wt['wt1'].iloc[-1]) if not pd.isna(wt['wt1'].iloc[-1]) else np.nan
    wt2_now = float(wt['wt2'].iloc[-1]) if not pd.isna(wt['wt2'].iloc[-1]) else np.nan
    result['wt1'] = wt1_now
    result['wt2'] = wt2_now

    wt_bullish = bool(wt['wt_bullish'].iloc[-1]) if not pd.isna(wt['wt_bullish'].iloc[-1]) else False
    cross_up_recent = wt['cross_up'].iloc[-WT_CROSS_LOOKBACK:].any() if len(df) >= WT_CROSS_LOOKBACK else False
    wt_in_zone = WT1_ZONE_LO <= wt1_now <= WT1_ZONE_HI if not np.isnan(wt1_now) else False

    if cross_up_recent and wt_in_zone and wt_bullish:
        # WT1'in zona göre skoru: 0'a yakın en iyi (taze cross)
        zone_bonus = max(0, 15 - abs(wt1_now)) if not np.isnan(wt1_now) else 0
        result['is_candidate'] = True
        result['momentum_type'] = 'WT_CROSS'
        result['score'] = 60.0 + zone_bonus
        return result

    # B) Squeeze breakout (SBT complementary)
    if SBT_SQUEEZE_COMPLEMENT:
        sqz = _detect_squeeze_breakout(df)
        if sqz['is_breakout'] and sqz['bars_ago'] <= 3:
            result['is_candidate'] = True
            result['momentum_type'] = 'SBT_BREAKOUT'
            result['score'] = 45.0 + sqz['strength'] * 5
            return result

    # C) SuperTrend flip
    st = calc_supertrend(df)
    if len(st) >= ST_FLIP_LOOKBACK + 1:
        recent = st.iloc[-(ST_FLIP_LOOKBACK + 1):]
        # -1 → 1 geçişi var mı?
        for i in range(1, len(recent)):
            if recent.iloc[i] == 1 and recent.iloc[i - 1] == -1:
                result['is_candidate'] = True
                result['momentum_type'] = 'ST_FLIP'
                result['score'] = 40.0
                return result

    return result


# ═══════════════════════════════════════════
# AŞAMA 2: Eğim Doğrulama
# ═══════════════════════════════════════════

def stage2_slope_validation(df: pd.DataFrame, wt_data: dict = None) -> dict:
    """Fiyat eğimi + sinyal eğimi + trend hizası kontrolü.

    Args:
        df: OHLCV DataFrame (Uppercase)
        wt_data: önceden hesaplanmış WaveTrend (performans için)

    Returns:
        {slope_valid, price_slope, signal_slope, trend_aligned}
    """
    result = {
        'slope_valid': False,
        'price_slope': 0.0,
        'signal_slope': 0.0,
        'trend_aligned': False,
    }

    c = df['Close']
    n = len(df)
    if n < max(SLOPE_EMA_SLOW, SLOPE_WINDOW) + 5:
        return result

    # 1) Fiyat eğimi: EMA(fast) slope > 0
    ema_fast = ema(c, SLOPE_EMA_FAST)
    if n > SLOPE_WINDOW:
        ema_now = float(ema_fast.iloc[-1])
        ema_ago = float(ema_fast.iloc[-1 - SLOPE_WINDOW])
        if ema_ago > 0:
            price_slope = (ema_now - ema_ago) / ema_ago * 100
        else:
            price_slope = 0.0
    else:
        price_slope = 0.0
    result['price_slope'] = round(price_slope, 4)

    # 2) Sinyal eğimi: WT1 slope > 0
    if wt_data is None:
        wt_data = calc_wavetrend(df)
    wt1 = wt_data['wt1']
    if n > SIGNAL_SLOPE_WINDOW and not pd.isna(wt1.iloc[-1]):
        wt1_now = float(wt1.iloc[-1])
        wt1_ago = float(wt1.iloc[-1 - SIGNAL_SLOPE_WINDOW])
        signal_slope = wt1_now - wt1_ago if not np.isnan(wt1_ago) else 0.0
    else:
        signal_slope = 0.0
    result['signal_slope'] = round(signal_slope, 4)

    # 3) Trend hizası: EMA(fast) > EMA(slow)
    ema_slow = ema(c, SLOPE_EMA_SLOW)
    trend_aligned = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    result['trend_aligned'] = trend_aligned

    # Geçiş: 3 kontrolden en az SLOPE_MIN_CHECKS tanesi geçmeli
    checks_passed = sum([
        price_slope > SLOPE_MIN_THRESHOLD,
        signal_slope > 0,
        trend_aligned,
    ])
    result['slope_valid'] = checks_passed >= SLOPE_MIN_CHECKS

    return result


# ═══════════════════════════════════════════
# AŞAMA 3: Teknik Onay
# ═══════════════════════════════════════════

def stage3_confirmation(df: pd.DataFrame) -> dict:
    """ADX + CMF + mum formasyonu + RSI kontrolü.

    Returns:
        {confirmed, score, adx, cmf, candle_pattern, rsi, details}
    """
    result = {
        'confirmed': False,
        'score': 0,
        'adx': 0.0,
        'cmf': 0.0,
        'candle_pattern': None,
        'rsi': 0.0,
        'details': {},
    }

    if len(df) < MIN_DATA_DAYS:
        return result

    checks = 0

    # 1) ADX >= 20
    adx = calc_adx(df)
    adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
    result['adx'] = round(adx_val, 1)
    adx_ok = adx_val >= ADX_MIN
    if adx_ok:
        checks += 1
    result['details']['adx_ok'] = adx_ok

    # 2) CMF > 0
    cmf = calc_cmf(df)
    cmf_val = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0
    result['cmf'] = round(cmf_val, 3)
    cmf_ok = cmf_val > CMF_MIN
    if cmf_ok:
        checks += 1
    result['details']['cmf_ok'] = cmf_ok

    # 3) Bullish mum formasyonu son N barda
    candle_scores, candle_names = _detect_bullish_candles(df)
    recent_candles = candle_scores.iloc[-CANDLE_LOOKBACK:]
    best_idx = recent_candles.idxmax()
    best_score = recent_candles.max()
    candle_ok = best_score > 0
    if candle_ok:
        checks += 1
        result['candle_pattern'] = candle_names.loc[best_idx] if best_score > 0 else None
    result['details']['candle_ok'] = candle_ok

    # 4) RSI sweet spot [40, 70]
    rsi = calc_rsi(df['Close'], 14)
    rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    result['rsi'] = round(rsi_val, 1)
    rsi_ok = RSI_LO <= rsi_val <= RSI_HI
    if rsi_ok:
        checks += 1
    result['details']['rsi_ok'] = rsi_ok

    result['score'] = checks
    result['confirmed'] = checks >= CONFIRMATION_MIN_SCORE

    return result


# ═══════════════════════════════════════════
# Evren Taraması (3 aşama birleşik)
# ═══════════════════════════════════════════

def scan_universe(all_data: dict, date_idx: int = -1) -> list[dict]:
    """Tüm evreni 3 aşamadan geçir.

    Args:
        all_data: {ticker: DataFrame(OHLCV, Uppercase)} — fetch_data çıktısı
        date_idx: hangi bara kadar değerlendir (-1 = son bar)

    Returns:
        list of dicts: geçenler + geçemeyenler, 'passed' flag ile
    """
    results = []

    for ticker, df in all_data.items():
        if df is None or len(df) < MIN_DATA_DAYS:
            continue

        # date_idx'e kadar kes (look-ahead bias önleme)
        if date_idx != -1:
            df = df.iloc[:date_idx + 1].copy()
            if len(df) < MIN_DATA_DAYS:
                continue

        # Aşama 1
        momentum = stage1_momentum(df)
        if not momentum['is_candidate']:
            continue

        # WaveTrend'i bir kere hesapla (performans)
        wt = calc_wavetrend(df)

        # Aşama 2
        slope = stage2_slope_validation(df, wt_data=wt)
        if not slope['slope_valid']:
            continue

        # Aşama 3
        confirmation = stage3_confirmation(df)

        # Composite skor (100 üzerinden)
        composite = (
            momentum['score'] * 0.40 +          # max ~75 → ~30
            confirmation['score'] * 10 * 0.30 +  # max 40 → ~12
            slope['price_slope'] * 5 * 0.30       # slope bonus
        )
        composite = min(100.0, max(0.0, composite))

        results.append({
            'ticker': ticker,
            'date': df.index[-1],
            'close': float(df['Close'].iloc[-1]),
            'momentum': momentum,
            'slope': slope,
            'confirmation': confirmation,
            'composite_score': round(composite, 1),
            'passed': confirmation['confirmed'],
        })

    # Composite skora göre sırala
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results
