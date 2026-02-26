"""
NOX v3 PIVOT AL/SAT Daily Screener — Sinyal Hesaplama Modulu
=============================================================
PineScript "NOX Reversal Screener v3 - Overlay" 1:1 replika.
PIVOT AL (◆) ve PIVOT SAT (◆) sinyalleri.

NOT: Bu modul lowercase kolon isimleri kullanir (close, high, low, open, volume).
     Runner script (run_nox_v3.py) uppercase→lowercase donusumunu yapar.

Pine fidelity: Tum gostergeler _pine_rma() kullanir (ta.rma 1:1 — SMA ile init).
"""

import numpy as np
import pandas as pd

from core.indicators import ema, sma


# =============================================================================
# CONSTANTS — PineScript input parametreleri
# =============================================================================

NOX_V3_TRIGGER = {
    'daily_pivot_lb': 3,    # BOS icin gunluk swing high lb
    'hc2_count': 2,         # Ardisik higher close sayisi
    'ema_len': 21,          # EMA Reclaim periyodu
    'vol_sma_len': 20,      # Hacim SMA periyodu
    'vol_mult': 1.3,        # Hacim carpani
    'max_delta_pct': 15.0,  # Pivot zonundan max uzaklik (%)
}


NOX_V3 = {
    'pivot_lb': 5,
    'adx_len': 14,
    'rsi_len': 14,
    'atr_len': 14,
    'ema_len': 21,
    # Sell severity esikleri
    'sev3_move_atr': -3.5,
    'sev3_dd_pct': -12.0,
    'sev2_move_atr': -2.5,
    'sev2_red_count': 5,
    'sev2_decline_5d_atr': -3.0,
    'sev1_move_atr': -1.5,
    'sev1_red_count': 3,
    'sev1_dd_pct': -8.0,
    # Gate esikleri
    'br_gate': 25,
    'rg_gate': 25,
    # Regime gate
    'was_trending_adx': 25,
    # Sell slope gate
    'sell_slope_gate': -0.3,
}


# =============================================================================
# PINE-UYUMLU GOSTERGE FONKSIYONLARI
# =============================================================================

def _pine_rma(series, period):
    """
    Pine ta.rma 1:1 replika — ilk period barda SMA, sonra Wilder's EMA.

    Pine kaynak:
        rma(source, length) =>
            alpha = 1/length
            sum := na(sum[1]) ? ta.sma(source, length) : alpha*source + (1-alpha)*nz(sum[1])
    """
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    # Ilk gecerli pencereyi bul, SMA ile baslat
    valid = 0
    sma_sum = 0.0
    start = -1
    for i in range(n):
        if not np.isnan(values[i]):
            sma_sum += values[i]
            valid += 1
            if valid == period:
                start = i
                result[i] = sma_sum / period
                break

    if start < 0:
        return pd.Series(result, index=series.index)

    alpha = 1.0 / period
    for i in range(start + 1, n):
        v = values[i] if not np.isnan(values[i]) else 0.0
        result[i] = alpha * v + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def _pine_rsi(series, period):
    """
    Pine ta.rsi 1:1 replika — _pine_rma kullanir.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _pine_rma(gain, period)
    avg_loss = _pine_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(df):
    """True Range — lowercase columns."""
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _pine_atr(df, period=14):
    """Pine ta.atr 1:1 replika — _pine_rma kullanir."""
    return _pine_rma(_true_range(df), period)


def _consecutive_count(mask):
    """Boolean mask → ardisik True sayisi serisi."""
    arr = mask.values.astype(int)
    result = np.zeros(len(arr), dtype=int)
    for i in range(len(arr)):
        if arr[i]:
            result[i] = (result[i - 1] + 1) if i > 0 else 1
    return pd.Series(result, index=mask.index)


# =============================================================================
# 1. ADX WITH DI — Pine ta.dmi(length, length) replika
# =============================================================================

def calc_adx_with_di(df, length=14):
    """
    Pine ta.dmi(di_length, adx_smoothing) 1:1 replika.
    _pine_rma kullanir (SMA init).

    Returns: (adx, plus_di, minus_di)
    """
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )
    tr = _true_range(df)
    atr_val = _pine_rma(tr, length)
    plus_di = 100 * _pine_rma(plus_dm, length) / atr_val.replace(0, np.nan)
    minus_di = 100 * _pine_rma(minus_dm, length) / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _pine_rma(dx, length)
    return adx, plus_di, minus_di


# =============================================================================
# 2. PIVOT LOW
# =============================================================================

def find_pivot_lows(low, lb):
    """
    Pine ta.pivotlow(low, lb, lb) 1:1 replika.

    Bar i'de, low[i-lb]'nin [i-2*lb .. i] penceresinin minimumu olup olmadigini kontrol et.
    Pivot ONAY BARINDA (i) tetiklenir, pivot barinda (i-lb) degil.

    Returns: Series (pivot low fiyati veya NaN)
    """
    result = pd.Series(np.nan, index=low.index)
    low_vals = low.values
    n = len(low_vals)
    for i in range(2 * lb, n):
        pivot_bar = i - lb
        window_start = i - 2 * lb
        window_min = low_vals[window_start:i + 1].min()
        if low_vals[pivot_bar] <= window_min:
            result.iloc[i] = low_vals[pivot_bar]
    return result


# =============================================================================
# 3. PIVOT HIGH
# =============================================================================

def find_pivot_highs(high, lb):
    """
    Pine ta.pivothigh(high, lb, lb) 1:1 replika.

    Returns: Series (pivot high fiyati veya NaN)
    """
    result = pd.Series(np.nan, index=high.index)
    high_vals = high.values
    n = len(high_vals)
    for i in range(2 * lb, n):
        pivot_bar = i - lb
        window_start = i - 2 * lb
        window_max = high_vals[window_start:i + 1].max()
        if high_vals[pivot_bar] >= window_max:
            result.iloc[i] = high_vals[pivot_bar]
    return result


# =============================================================================
# 4. BREADTH PROXY
# =============================================================================

def compute_breadth_proxy(df):
    """
    Breadth proxy skoru (0-100) — tek hisse bazli Pine bilesenleri.

    - RSI thrust (50pt): rsi.rolling(10).min() < 30 AND rsi > 55
    - RSI gradual (25pt): (rsi - rsi.shift(5)) > 15 AND rsi > 40 AND rsi.rolling(10).min() < 35
    - AD proxy (25pt): consecutive_green >= 3 AND vol > sma(vol,20)*1.3
      green = close > open (yesil mum, close[1] degil)
    - EMA reclaim (15pt): close > ema21 AND close[1] < ema21[1]  (strict <)
    """
    rsi = _pine_rsi(df['close'], NOX_V3['rsi_len'])
    ema21 = ema(df['close'], NOX_V3['ema_len'])
    vol_sma20 = sma(df['volume'], 20)

    br_score = pd.Series(0.0, index=df.index)

    # RSI thrust (50pt)
    rsi_min10 = rsi.rolling(10).min()
    rsi_thrust = (rsi_min10 < 30) & (rsi > 55)
    br_score = br_score + rsi_thrust.astype(float) * 50

    # RSI gradual (25pt)
    rsi_change5 = rsi - rsi.shift(5)
    rsi_gradual = (rsi_change5 > 15) & (rsi > 40) & (rsi_min10 < 35)
    br_score = br_score + rsi_gradual.astype(float) * 25

    # AD proxy (25pt) — green = close > open
    green = df['close'] > df['open']
    green_count = _consecutive_count(green)
    ad_proxy = (green_count >= 3) & (df['volume'] > vol_sma20 * 1.3)
    br_score = br_score + ad_proxy.astype(float) * 25

    # EMA reclaim (15pt) — Pine: close[1] < ema21[1] (strict <)
    ema_reclaim = (df['close'] > ema21) & (df['close'].shift(1) < ema21.shift(1))
    br_score = br_score + ema_reclaim.astype(float) * 15

    return br_score.clip(0, 100)


# =============================================================================
# 5. REGIME SCORE
# =============================================================================

def compute_regime_score(df):
    """
    Regime shift skoru — Pine bilesenleri.

    - di_diff_norm = (plus_di - minus_di) / (1 + atr_pct * 0.1)
    - adx_slope = (adx - adx.shift(5)) / 5
    - was_trending = adx.rolling(20).max() > 25 — GATE: yoksa skor=0
    - Slope score (0-30) + DI score (0-30) + EMA above (15) + ADX rebound (15)
    - rg_score = raw * was_trending

    Returns: (rg_score, adx, adx_slope)
    """
    adx, plus_di, minus_di = calc_adx_with_di(df, NOX_V3['adx_len'])
    atr = _pine_atr(df, NOX_V3['atr_len'])
    atr_pct = atr / df['close'].replace(0, np.nan) * 100
    ema21 = ema(df['close'], NOX_V3['ema_len'])

    di_diff_norm = (plus_di - minus_di) / (1 + atr_pct * 0.1)
    adx_slope = (adx - adx.shift(5)) / 5
    was_trending = (adx.rolling(20).max() > NOX_V3['was_trending_adx']).astype(float)

    # Slope score (0-30)
    slope_score = (adx_slope.clip(0, 2) * 15).fillna(0)

    # DI score (0-30)
    di_score = (di_diff_norm.clip(0, 10) * 3).fillna(0)

    # EMA above (15)
    ema_above = (df['close'] > ema21).astype(float) * 15

    # ADX rebound (15)
    adx_min20 = adx.rolling(20).min()
    adx_rebound = ((adx - adx_min20) > 5).astype(float) * 15

    raw = (slope_score + di_score + ema_above + adx_rebound).clip(0, 100)
    rg_score = raw * was_trending

    return rg_score, adx, adx_slope


# =============================================================================
# 6. SELL SEVERITY
# =============================================================================

def compute_sell_severity(df):
    """
    Satis severity skoru (0-3) — Pine if-elif sirasi, yuksek override.

    - daily_move_atr = (close - open) / atr — INTRADAY (close-open, close[1] degil)
    - red_count = ardisik close < open gun sayisi (KIRMIZI MUM, close[1] degil)
    - drawdown_pct = (close - highest(high, 20)) / highest(high, 20) * 100
    - decline_5d_atr = (close - highest(close, 5)) / atr (TEPEDEN DUSUS, shift degil)

    Sev 3: move < -3.5 ATR OR drawdown < -12%
    Sev 2: move < -2.5 ATR OR (red >= 5 AND decline_5d < -3 ATR)
    Sev 1: move < -1.5 ATR OR red >= 3 OR drawdown < -8%
    """
    atr = _pine_atr(df, NOX_V3['atr_len'])

    # daily_move = (close - open) / atr — intraday hareket
    daily_move_atr = (df['close'] - df['open']) / atr.replace(0, np.nan)

    # red_count = ardisik close < open (kirmizi mum)
    is_red = df['close'] < df['open']
    red_count = _consecutive_count(is_red)

    # drawdown_pct = (close - highest(high, 20)) / highest(high, 20) * 100
    highest_high_20 = df['high'].rolling(20).max()
    drawdown_pct = (df['close'] - highest_high_20) / highest_high_20.replace(0, np.nan) * 100

    # decline_5d_atr = (close - highest(close, 5)) / atr — tepeden dusus
    highest_close_5 = df['close'].rolling(5).max()
    decline_5d_atr = (df['close'] - highest_close_5) / atr.replace(0, np.nan)

    C = NOX_V3
    severity = pd.Series(0, index=df.index, dtype=int)

    # Sev 1 (en dusuk)
    sev1 = (
        (daily_move_atr < C['sev1_move_atr']) |
        (red_count >= C['sev1_red_count']) |
        (drawdown_pct < C['sev1_dd_pct'])
    )
    severity = severity.where(~sev1, 1)

    # Sev 2 (override sev 1)
    sev2 = (
        (daily_move_atr < C['sev2_move_atr']) |
        ((red_count >= C['sev2_red_count']) & (decline_5d_atr < C['sev2_decline_5d_atr']))
    )
    severity = severity.where(~sev2, 2)

    # Sev 3 (en yuksek — override)
    sev3 = (
        (daily_move_atr < C['sev3_move_atr']) |
        (drawdown_pct < C['sev3_dd_pct'])
    )
    severity = severity.where(~sev3, 3)

    return severity


# =============================================================================
# 7. ANA FONKSIYON — compute_nox_v3
# =============================================================================

def compute_nox_v3(df, require_gate=False, min_sell_severity=0):
    """
    NOX v3 ana sinyal hesaplama fonksiyonu.
    Pine "NOX RS v3 - Overlay" 1:1 replika.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        require_gate: Pine i_pv_require_gate (default False = her pivot low AL)
        min_sell_severity: Pine i_pv_sell_severity (default 0 = her pivot high SAT)

    Returns: dict
    """
    lb = NOX_V3['pivot_lb']

    # 1. Pivot detection
    pivot_low = find_pivot_lows(df['low'], lb)
    pivot_high = find_pivot_highs(df['high'], lb)

    # 2. Breadth proxy
    br_score = compute_breadth_proxy(df)

    # 3. Regime shift
    rg_score, adx, adx_slope = compute_regime_score(df)

    # 4. Gate
    gate_open = (br_score >= NOX_V3['br_gate']) | (rg_score >= NOX_V3['rg_gate'])

    # 5. Sell severity
    severity = compute_sell_severity(df)

    # 6. PIVOT BUY — Pine: raw_pivot_buy and (gate_open if require_gate else true)
    if require_gate:
        pivot_buy = pivot_low.notna() & gate_open
    else:
        pivot_buy = pivot_low.notna()

    # 7. PIVOT SELL — Pine: raw_pivot_sell and (severity >= min_sev or adx_slope < -0.3)
    if min_sell_severity == 0:
        # severity >= 0 her zaman true, her pivot high = SAT
        pivot_sell = pivot_high.notna()
    else:
        pivot_sell = pivot_high.notna() & (
            (severity >= min_sell_severity) | (adx_slope < NOX_V3['sell_slope_gate'])
        )

    # 8. Context
    rsi = _pine_rsi(df['close'], NOX_V3['rsi_len'])
    atr = _pine_atr(df, NOX_V3['atr_len'])
    ema21 = ema(df['close'], NOX_V3['ema_len'])

    # Phase
    phase = pd.Series('', index=df.index)
    above = df['close'] > ema21
    phase = phase.where(~above, 'EMA Ustu')
    phase = phase.where(above, 'EMA Alti')

    # Drawdown %
    highest_high_20 = df['high'].rolling(20).max()
    drawdown_pct = (df['close'] - highest_high_20) / highest_high_20.replace(0, np.nan) * 100

    return {
        'pivot_buy': pivot_buy,
        'pivot_sell': pivot_sell,
        'close': df['close'],
        'br_score': br_score,
        'rg_score': rg_score,
        'sell_severity': severity,
        'gate_open': gate_open,
        'adx': adx,
        'adx_slope': adx_slope,
        'rsi': rsi,
        'atr': atr,
        'pivot_low_price': pivot_low,
        'pivot_high_price': pivot_high,
        'phase': phase,
        'drawdown_pct': drawdown_pct,
    }


# =============================================================================
# 8. GUNLUK TETIK — Pivot zona yakinken ates eden tetikler
# =============================================================================

def _no_trigger():
    """Tetik bulunamadi sentinel."""
    return {
        'triggered': False,
        'trigger_type': None,
        'trigger_date': None,
        'trigger_close': None,
        'delta_pct_at_trigger': None,
    }


def detect_daily_triggers(daily_df, pivot_price, pivot_confirm_date,
                          max_delta_pct=None):
    """
    Haftalik pivot zonuna yakinken gunluk tetik ara.

    Pivot confirm tarihinden sonraki gunluk barlarda:
      1. BOS  — close, son daily swing high'i kiriyor
      2. HC2  — 2 ardisik higher close
      3. EMA_R — close EMA21'i hacimli geri aliyor

    Args:
        daily_df: Gunluk DataFrame (lowercase kolonlar)
        pivot_price: Haftalik pivot low fiyati
        pivot_confirm_date: Pivot onay tarihi (str 'YYYY-MM-DD' veya Timestamp)
        max_delta_pct: Pivot zonundan max uzaklik (%), None ise default

    Returns: dict (triggered, trigger_type, trigger_date, trigger_close, delta_pct_at_trigger)
    """
    C = NOX_V3_TRIGGER
    if max_delta_pct is None:
        max_delta_pct = C['max_delta_pct']

    if len(daily_df) < 30:
        return _no_trigger()

    # Pivot confirm tarihinden sonraki barlari filtrele
    confirm_ts = pd.Timestamp(pivot_confirm_date)
    mask = daily_df.index > confirm_ts
    if not mask.any():
        return _no_trigger()

    start_idx = int(np.argmax(mask))  # ilk True indeksi

    # Onceden hesapla: swing highs, EMA, volume SMA, consecutive higher close
    swing_highs = find_pivot_highs(daily_df['high'], C['daily_pivot_lb'])
    ema21 = ema(daily_df['close'], C['ema_len'])
    vol_sma = sma(daily_df['volume'], C['vol_sma_len'])

    closes = daily_df['close'].values
    highs = daily_df['high'].values
    volumes = daily_df['volume'].values
    ema21_vals = ema21.values
    vol_sma_vals = vol_sma.values

    # Son bilinen swing high takibi
    last_swing_high = np.nan
    for i in range(start_idx):
        if pd.notna(swing_highs.iloc[i]):
            last_swing_high = swing_highs.iloc[i]

    # Tarama
    for i in range(start_idx, len(daily_df)):
        # Swing high guncelle (mevcut bar dahil)
        if pd.notna(swing_highs.iloc[i]):
            last_swing_high = swing_highs.iloc[i]

        c = closes[i]

        # Zone proximity kontrolu
        delta_pct = (c - pivot_price) / pivot_price * 100
        if delta_pct < 0:
            # Fiyat pivot altinda, bu bari atla ama aramaya devam et
            continue
        if delta_pct > max_delta_pct:
            # Zondan cok uzak, atla
            continue

        # — Tetik 1: BOS (Break of Structure) —
        if pd.notna(last_swing_high) and c > last_swing_high:
            return {
                'triggered': True,
                'trigger_type': 'BOS',
                'trigger_date': daily_df.index[i].strftime('%Y-%m-%d'),
                'trigger_close': float(c),
                'delta_pct_at_trigger': round(delta_pct, 2),
            }

        # — Tetik 2: HC2 (2 ardisik higher close) —
        if i >= C['hc2_count']:
            hc_ok = True
            for k in range(1, C['hc2_count'] + 1):
                if closes[i - k + 1] <= closes[i - k]:
                    hc_ok = False
                    break
            if hc_ok:
                return {
                    'triggered': True,
                    'trigger_type': 'HC2',
                    'trigger_date': daily_df.index[i].strftime('%Y-%m-%d'),
                    'trigger_close': float(c),
                    'delta_pct_at_trigger': round(delta_pct, 2),
                }

        # — Tetik 3: EMA_R (EMA21 Reclaim + hacim) —
        e = ema21_vals[i]
        v = volumes[i]
        vs = vol_sma_vals[i]
        if (i >= 1 and pd.notna(e) and pd.notna(ema21_vals[i - 1])
                and pd.notna(vs)):
            prev_below = closes[i - 1] < ema21_vals[i - 1]
            now_above = c > e
            vol_ok = v > vs * C['vol_mult']
            if prev_below and now_above and vol_ok:
                return {
                    'triggered': True,
                    'trigger_type': 'EMA_R',
                    'trigger_date': daily_df.index[i].strftime('%Y-%m-%d'),
                    'trigger_close': float(c),
                    'delta_pct_at_trigger': round(delta_pct, 2),
                }

    return _no_trigger()
