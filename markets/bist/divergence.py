"""
NOX Divergence Screener v2 — Pivotsuz + Yeni Detectorler
==========================================================
BIST hisseleri icin kapsamli divergence taramasi:
  1. RSI Divergence (Classic + Hidden, Bullish + Bearish)
  2. MACD Histogram Divergence (Classic + Hidden)
  3. OBV Divergence (Classic + Hidden)
  4. MFI Divergence (Classic + Hidden)
  5. ADX Exhaustion (Trend bitisi tespiti)
  6. Uclu Uyumsuzluk (RSI + MACD + MFI confluence)
  7. Fiyat-Hacim Uyumsuzlugu (Price-Volume trend divergence)

Pivotsuz yaklasim: Lightweight swing detection (order=2).
Gecikme = order bar (haftalikta 2 hafta, gunlukte 2 gun).

Lowercase kolon konvansiyonu: close, high, low, open, volume.
Runner script (run_divergence.py) uppercase→lowercase donusumunu yapar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from core.indicators import ema, sma


# =============================================================================
# CONSTANTS
# =============================================================================

DIV_CFG = {
    # Swing detection (pivot yerine)
    'swing_order': 2,        # Local extreme lookback (her yonde)
    'max_swing_gap': 50,     # Max bar arasi
    # Indicators
    'rsi_len': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'mfi_len': 14,
    'adx_len': 14,
    'atr_len': 14,
    'vol_sma_len': 20,
    'obv_ema_len': 20,
    # Scan
    'scan_bars': 10,
    # Thresholds
    'min_rsi_diff': 3.0,
    'min_mfi_diff': 3.0,
    'min_macd_diff_atr': 0.05,
    'adx_lookback': 10,
    'adx_min_slope': -0.3,
    'pv_div_lookback': 20,
    'pv_div_min_bars': 5,
}


# =============================================================================
# VERI YAPISI
# =============================================================================

@dataclass
class DivergenceSignal:
    bar_idx: int          # Sinyal bari (swing'in signal_bar'i)
    direction: str        # 'BUY' veya 'SELL'
    div_type: str         # RSI_CLASSIC, RSI_HIDDEN, MACD_CLASSIC, MACD_HIDDEN,
                          # OBV_CLASSIC, OBV_HIDDEN, MFI_CLASSIC, MFI_HIDDEN,
                          # ADX_EXHAUST, TRIPLE, PRICE_VOLUME
    quality: int          # 0-100
    details: dict = field(default_factory=dict)


# =============================================================================
# SWING DETECTION (Pivotsuz)
# =============================================================================

def _find_swings(series, order=2):
    """
    Local minima/maxima bul. Gecikme = order bar.

    Returns: (lows, highs)
        lows:  list of (swing_bar, signal_bar, price)
        highs: list of (swing_bar, signal_bar, price)
    """
    lows, highs = [], []
    vals = series.values
    n = len(vals)
    for i in range(order, n - order):
        window = vals[i - order: i + order + 1]
        if np.any(np.isnan(window)):
            continue
        if vals[i] <= np.nanmin(window):
            lows.append((i, i + order, vals[i]))
        if vals[i] >= np.nanmax(window):
            highs.append((i, i + order, vals[i]))
    return lows, highs


# =============================================================================
# YARDIMCI — ATR / RSI / MACD / OBV / MFI / ADX (lowercase kolonlar)
# =============================================================================

def _true_range(df):
    """True Range — lowercase columns."""
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _rma(series, period):
    """Wilder's RMA (EMA with alpha=1/period)."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def _calc_atr(df, period=14):
    """ATR — lowercase columns."""
    return _rma(_true_range(df), period)


def _calc_rsi(series, period=14):
    """RSI — standard Wilder."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _calc_macd(series, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_obv(df):
    """On-Balance Volume — lowercase columns."""
    close = df['close'].values
    volume = df['volume'].values
    n = len(close)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return pd.Series(obv, index=df.index)


def _calc_mfi(df, period=14):
    """Money Flow Index — hacim agirlikli RSI."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_sum / neg_sum.replace(0, np.nan))
    return mfi


def _calc_adx(df, period=14):
    """ADX — lowercase columns, Wilder's method."""
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM / -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # True Range
    tr = _true_range(df)

    # Wilder smoothing
    atr = _rma(tr, period)
    plus_di = 100 * _rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm, period) / atr.replace(0, np.nan)

    # DX → ADX
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = _rma(dx, period)

    return adx


# =============================================================================
# HACIM TEYIDI
# =============================================================================

def _check_volume_confirmation(df, bar_idx, obv_ema):
    """
    Hacim teyidi kontrol et.

    Returns: (vol_spike, vol_ratio, obv_trend, vol_score)
    """
    vol = df['volume'].values
    vol_sma = sma(df['volume'], DIV_CFG['vol_sma_len']).values

    n = len(df)
    if bar_idx >= n or bar_idx < 0:
        return False, 1.0, 'flat', 0

    v = vol[bar_idx]
    vs = vol_sma[bar_idx] if bar_idx < len(vol_sma) and not np.isnan(vol_sma[bar_idx]) else 1.0
    if vs <= 0:
        vs = 1.0

    vol_ratio = v / vs
    vol_spike = vol_ratio >= 1.2

    # OBV trend: son 5 bar slope
    obv_trend = 'flat'
    if bar_idx >= 5 and bar_idx < len(obv_ema):
        obv_vals = obv_ema.values
        if not np.isnan(obv_vals[bar_idx]) and not np.isnan(obv_vals[bar_idx - 5]):
            diff = obv_vals[bar_idx] - obv_vals[bar_idx - 5]
            if diff > 0:
                obv_trend = 'up'
            elif diff < 0:
                obv_trend = 'down'

    # Skor hesapla (0-20)
    vol_score = 0
    if vol_ratio >= 2.0:
        vol_score = 20
    elif vol_ratio >= 1.5:
        vol_score = 15
    elif vol_ratio >= 1.2:
        vol_score = 10
    elif vol_ratio >= 1.0:
        vol_score = 5

    return vol_spike, vol_ratio, obv_trend, vol_score


# =============================================================================
# RSI DIVERGENCE (swing bazli)
# =============================================================================

def detect_rsi_divergence(df, rsi, swing_lows, swing_highs, atr, obv_ema):
    """
    RSI divergence tespiti (swing bazli).

    Classic Bullish: Fiyat LL yapar, RSI HL yapar → BUY
    Hidden Bullish:  Fiyat HL yapar, RSI LL yapar → BUY
    Classic Bearish: Fiyat HH yapar, RSI LH yapar → SELL
    Hidden Bearish:  Fiyat LH yapar, RSI HH yapar → SELL
    """
    signals = []
    rsi_vals = rsi.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        rsi_prev = rsi_vals[prev_idx]
        rsi_curr = rsi_vals[curr_idx]
        if np.isnan(rsi_prev) or np.isnan(rsi_curr):
            continue

        rsi_diff = abs(rsi_curr - rsi_prev)
        if rsi_diff < cfg['min_rsi_diff']:
            continue

        div_type = None
        direction = 'BUY'

        # Classic Bullish: price LL, RSI HL
        if curr_price < prev_price and rsi_curr > rsi_prev:
            div_type = 'RSI_CLASSIC'
        # Hidden Bullish: price HL, RSI LL
        elif curr_price > prev_price and rsi_curr < rsi_prev:
            div_type = 'RSI_HIDDEN'

        if div_type is None:
            continue

        quality = _rsi_div_quality(
            rsi_prev, rsi_curr, rsi_diff, prev_idx, curr_idx,
            curr_signal, df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_rsi': round(rsi_prev, 1),
                'curr_rsi': round(rsi_curr, 1),
                'rsi_diff': round(rsi_diff, 1),
            },
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        rsi_prev = rsi_vals[prev_idx]
        rsi_curr = rsi_vals[curr_idx]
        if np.isnan(rsi_prev) or np.isnan(rsi_curr):
            continue

        rsi_diff = abs(rsi_curr - rsi_prev)
        if rsi_diff < cfg['min_rsi_diff']:
            continue

        div_type = None
        direction = 'SELL'

        # Classic Bearish: price HH, RSI LH
        if curr_price > prev_price and rsi_curr < rsi_prev:
            div_type = 'RSI_CLASSIC'
        # Hidden Bearish: price LH, RSI HH
        elif curr_price < prev_price and rsi_curr > rsi_prev:
            div_type = 'RSI_HIDDEN'

        if div_type is None:
            continue

        quality = _rsi_div_quality(
            rsi_prev, rsi_curr, rsi_diff, prev_idx, curr_idx,
            curr_signal, df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_rsi': round(rsi_prev, 1),
                'curr_rsi': round(rsi_curr, 1),
                'rsi_diff': round(rsi_diff, 1),
            },
        ))

    return signals


def _rsi_div_quality(rsi_prev, rsi_curr, rsi_diff, prev_idx, curr_idx,
                     signal_bar, df, atr, obv_ema, direction, div_type):
    """RSI divergence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. RSI buyuklugu (0-25)
    if rsi_diff >= 20:
        quality += 25
    elif rsi_diff >= 15:
        quality += 20
    elif rsi_diff >= 10:
        quality += 15
    elif rsi_diff >= 5:
        quality += 10
    else:
        quality += 5

    # 2. RSI zone (0-15) — oversold/overbought bolge daha anlamli
    rsi_at_signal = rsi_curr
    if direction == 'BUY':
        if rsi_at_signal < 30:
            quality += 15
        elif rsi_at_signal < 40:
            quality += 10
        elif rsi_at_signal < 50:
            quality += 5
    else:  # SELL
        if rsi_at_signal > 70:
            quality += 15
        elif rsi_at_signal > 60:
            quality += 10
        elif rsi_at_signal > 50:
            quality += 5

    # 3. Fiyat-RSI kontrast (0-15)
    if curr_idx < n and prev_idx < n:
        close_vals = df['close'].values
        atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]
        if atr_val > 0:
            price_change_atr = abs(close_vals[curr_idx] - close_vals[prev_idx]) / atr_val
            if price_change_atr >= 3.0:
                quality += 15
            elif price_change_atr >= 2.0:
                quality += 10
            elif price_change_atr >= 1.0:
                quality += 5

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Hacim teyidi (0-15)
    _, vol_ratio, obv_trend, vol_score = _check_volume_confirmation(df, signal_bar, obv_ema)
    quality += min(vol_score, 15)

    # Classic > Hidden bonus
    if div_type == 'RSI_CLASSIC':
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# MACD DIVERGENCE (swing bazli)
# =============================================================================

def detect_macd_divergence(df, macd_hist, macd_line, swing_lows, swing_highs, atr, obv_ema):
    """
    MACD histogram divergence tespiti (swing bazli).

    Classic Bullish: Fiyat LL, MACD hist HL → BUY
    Hidden Bullish:  Fiyat HL, MACD hist LL → BUY
    Classic Bearish: Fiyat HH, MACD hist LH → SELL
    Hidden Bearish:  Fiyat LH, MACD hist HH → SELL
    """
    signals = []
    hist_vals = macd_hist.values
    macd_vals = macd_line.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        hist_prev = hist_vals[prev_idx]
        hist_curr = hist_vals[curr_idx]
        if np.isnan(hist_prev) or np.isnan(hist_curr):
            continue

        atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]
        if atr_val <= 0:
            continue

        hist_diff = abs(hist_curr - hist_prev)
        if hist_diff < cfg['min_macd_diff_atr'] * atr_val:
            continue

        div_type = None
        direction = 'BUY'

        # Classic Bullish: price LL, hist HL
        if curr_price < prev_price and hist_curr > hist_prev:
            div_type = 'MACD_CLASSIC'
        # Hidden Bullish: price HL, hist LL
        elif curr_price > prev_price and hist_curr < hist_prev:
            div_type = 'MACD_HIDDEN'

        if div_type is None:
            continue

        quality = _macd_div_quality(
            hist_prev, hist_curr, hist_diff, macd_vals,
            prev_idx, curr_idx, curr_signal,
            df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_hist': round(hist_prev, 4),
                'curr_hist': round(hist_curr, 4),
                'hist_diff': round(hist_diff, 4),
            },
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        hist_prev = hist_vals[prev_idx]
        hist_curr = hist_vals[curr_idx]
        if np.isnan(hist_prev) or np.isnan(hist_curr):
            continue

        atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]
        if atr_val <= 0:
            continue

        hist_diff = abs(hist_curr - hist_prev)
        if hist_diff < cfg['min_macd_diff_atr'] * atr_val:
            continue

        div_type = None
        direction = 'SELL'

        # Classic Bearish: price HH, hist LH
        if curr_price > prev_price and hist_curr < hist_prev:
            div_type = 'MACD_CLASSIC'
        # Hidden Bearish: price LH, hist HH
        elif curr_price < prev_price and hist_curr > hist_prev:
            div_type = 'MACD_HIDDEN'

        if div_type is None:
            continue

        quality = _macd_div_quality(
            hist_prev, hist_curr, hist_diff, macd_vals,
            prev_idx, curr_idx, curr_signal,
            df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_hist': round(hist_prev, 4),
                'curr_hist': round(hist_curr, 4),
                'hist_diff': round(hist_diff, 4),
            },
        ))

    return signals


def _macd_div_quality(hist_prev, hist_curr, hist_diff, macd_vals,
                      prev_idx, curr_idx, signal_bar,
                      df, atr, obv_ema, direction, div_type):
    """MACD divergence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]

    # 1. Histogram buyuklugu (0-25)
    if atr_val > 0:
        ratio = hist_diff / atr_val
        if ratio >= 0.5:
            quality += 25
        elif ratio >= 0.3:
            quality += 20
        elif ratio >= 0.15:
            quality += 15
        elif ratio >= 0.08:
            quality += 10
        else:
            quality += 5

    # 2. Zero-line kontekst (0-15)
    if direction == 'BUY':
        if hist_curr < 0 and hist_curr > hist_prev:
            quality += 15
        elif hist_curr < 0:
            quality += 10
        else:
            quality += 5
    else:  # SELL
        if hist_curr > 0 and hist_curr < hist_prev:
            quality += 15
        elif hist_curr > 0:
            quality += 10
        else:
            quality += 5

    # 3. MACD line teyidi (0-10)
    if curr_idx < len(macd_vals) and prev_idx < len(macd_vals):
        ml_prev = macd_vals[prev_idx]
        ml_curr = macd_vals[curr_idx]
        if not (np.isnan(ml_prev) or np.isnan(ml_curr)):
            if direction == 'BUY' and ml_curr > ml_prev:
                quality += 10
            elif direction == 'SELL' and ml_curr < ml_prev:
                quality += 10
            else:
                quality += 3

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Hacim (0-15)
    _, _, _, vol_score = _check_volume_confirmation(df, signal_bar, obv_ema)
    quality += min(vol_score, 15)

    # Classic > Hidden bonus
    if div_type == 'MACD_CLASSIC':
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# OBV DIVERGENCE (swing bazli)
# =============================================================================

def detect_obv_divergence(df, obv, swing_lows, swing_highs, atr, obv_ema):
    """
    OBV divergence tespiti (swing bazli).

    Classic Bullish: Fiyat LL, OBV HL → BUY
    Hidden Bullish:  Fiyat HL, OBV LL → BUY
    Classic Bearish: Fiyat HH, OBV LH → SELL
    Hidden Bearish:  Fiyat LH, OBV HH → SELL
    """
    signals = []
    obv_vals = obv.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        obv_prev = obv_vals[prev_idx]
        obv_curr = obv_vals[curr_idx]
        if np.isnan(obv_prev) or np.isnan(obv_curr):
            continue

        div_type = None
        direction = 'BUY'

        # Classic Bullish: price LL, OBV HL
        if curr_price < prev_price and obv_curr > obv_prev:
            div_type = 'OBV_CLASSIC'
        # Hidden Bullish: price HL, OBV LL
        elif curr_price > prev_price and obv_curr < obv_prev:
            div_type = 'OBV_HIDDEN'

        if div_type is None:
            continue

        quality = _obv_div_quality(
            obv_prev, obv_curr, prev_idx, curr_idx, curr_signal,
            df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_obv': round(obv_prev, 0),
                'curr_obv': round(obv_curr, 0),
            },
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        obv_prev = obv_vals[prev_idx]
        obv_curr = obv_vals[curr_idx]
        if np.isnan(obv_prev) or np.isnan(obv_curr):
            continue

        div_type = None
        direction = 'SELL'

        # Classic Bearish: price HH, OBV LH
        if curr_price > prev_price and obv_curr < obv_prev:
            div_type = 'OBV_CLASSIC'
        # Hidden Bearish: price LH, OBV HH
        elif curr_price < prev_price and obv_curr > obv_prev:
            div_type = 'OBV_HIDDEN'

        if div_type is None:
            continue

        quality = _obv_div_quality(
            obv_prev, obv_curr, prev_idx, curr_idx, curr_signal,
            df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_obv': round(obv_prev, 0),
                'curr_obv': round(obv_curr, 0),
            },
        ))

    return signals


def _obv_div_quality(obv_prev, obv_curr, prev_idx, curr_idx, signal_bar,
                     df, atr, obv_ema, direction, div_type):
    """OBV divergence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. OBV degisim buyuklugu (0-25) — normalize by avg volume
    avg_vol = df['volume'].iloc[max(0, curr_idx - 20):curr_idx + 1].mean()
    if avg_vol > 0:
        obv_change_norm = abs(obv_curr - obv_prev) / avg_vol
        if obv_change_norm >= 10.0:
            quality += 25
        elif obv_change_norm >= 5.0:
            quality += 20
        elif obv_change_norm >= 2.0:
            quality += 15
        elif obv_change_norm >= 1.0:
            quality += 10
        else:
            quality += 5

    # 2. OBV EMA trend teyidi (0-15)
    if signal_bar < len(obv_ema) and signal_bar >= 5:
        obv_ema_vals = obv_ema.values
        if not np.isnan(obv_ema_vals[signal_bar]) and not np.isnan(obv_ema_vals[signal_bar - 5]):
            ema_diff = obv_ema_vals[signal_bar] - obv_ema_vals[signal_bar - 5]
            if direction == 'BUY' and ema_diff > 0:
                quality += 15
            elif direction == 'SELL' and ema_diff < 0:
                quality += 15
            else:
                quality += 5

    # 3. Fiyat-OBV kontrast (0-15)
    if curr_idx < n and prev_idx < n:
        atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]
        if atr_val > 0:
            close_vals = df['close'].values
            price_change_atr = abs(close_vals[curr_idx] - close_vals[prev_idx]) / atr_val
            if price_change_atr >= 3.0:
                quality += 15
            elif price_change_atr >= 2.0:
                quality += 10
            elif price_change_atr >= 1.0:
                quality += 5

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Hacim spike (0-15)
    _, _, _, vol_score = _check_volume_confirmation(df, signal_bar, obv_ema)
    quality += min(vol_score, 15)

    # Classic > Hidden bonus
    if div_type == 'OBV_CLASSIC':
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# MFI DIVERGENCE (swing bazli — RSI ile ayni kalip)
# =============================================================================

def detect_mfi_divergence(df, mfi, swing_lows, swing_highs, atr, obv_ema):
    """
    MFI divergence tespiti (swing bazli).

    Classic Bullish: Fiyat LL, MFI HL → BUY
    Hidden Bullish:  Fiyat HL, MFI LL → BUY
    Classic Bearish: Fiyat HH, MFI LH → SELL
    Hidden Bearish:  Fiyat LH, MFI HH → SELL
    """
    signals = []
    mfi_vals = mfi.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        mfi_prev = mfi_vals[prev_idx]
        mfi_curr = mfi_vals[curr_idx]
        if np.isnan(mfi_prev) or np.isnan(mfi_curr):
            continue

        mfi_diff = abs(mfi_curr - mfi_prev)
        if mfi_diff < cfg['min_mfi_diff']:
            continue

        div_type = None
        direction = 'BUY'

        # Classic Bullish: price LL, MFI HL
        if curr_price < prev_price and mfi_curr > mfi_prev:
            div_type = 'MFI_CLASSIC'
        # Hidden Bullish: price HL, MFI LL
        elif curr_price > prev_price and mfi_curr < mfi_prev:
            div_type = 'MFI_HIDDEN'

        if div_type is None:
            continue

        quality = _mfi_div_quality(
            mfi_prev, mfi_curr, mfi_diff, prev_idx, curr_idx,
            curr_signal, df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_mfi': round(mfi_prev, 1),
                'curr_mfi': round(mfi_curr, 1),
                'mfi_diff': round(mfi_diff, 1),
            },
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw
        curr_idx, curr_signal, curr_price = curr_sw

        if curr_idx - prev_idx > cfg['max_swing_gap']:
            continue
        if prev_idx >= n or curr_idx >= n:
            continue

        mfi_prev = mfi_vals[prev_idx]
        mfi_curr = mfi_vals[curr_idx]
        if np.isnan(mfi_prev) or np.isnan(mfi_curr):
            continue

        mfi_diff = abs(mfi_curr - mfi_prev)
        if mfi_diff < cfg['min_mfi_diff']:
            continue

        div_type = None
        direction = 'SELL'

        # Classic Bearish: price HH, MFI LH
        if curr_price > prev_price and mfi_curr < mfi_prev:
            div_type = 'MFI_CLASSIC'
        # Hidden Bearish: price LH, MFI HH
        elif curr_price < prev_price and mfi_curr > mfi_prev:
            div_type = 'MFI_HIDDEN'

        if div_type is None:
            continue

        quality = _mfi_div_quality(
            mfi_prev, mfi_curr, mfi_diff, prev_idx, curr_idx,
            curr_signal, df, atr, obv_ema, direction, div_type
        )

        signals.append(DivergenceSignal(
            bar_idx=curr_signal,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'prev_swing_idx': prev_idx,
                'curr_swing_idx': curr_idx,
                'prev_price': round(prev_price, 4),
                'curr_price': round(curr_price, 4),
                'prev_mfi': round(mfi_prev, 1),
                'curr_mfi': round(mfi_curr, 1),
                'mfi_diff': round(mfi_diff, 1),
            },
        ))

    return signals


def _mfi_div_quality(mfi_prev, mfi_curr, mfi_diff, prev_idx, curr_idx,
                     signal_bar, df, atr, obv_ema, direction, div_type):
    """MFI divergence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. MFI buyuklugu (0-25)
    if mfi_diff >= 20:
        quality += 25
    elif mfi_diff >= 15:
        quality += 20
    elif mfi_diff >= 10:
        quality += 15
    elif mfi_diff >= 5:
        quality += 10
    else:
        quality += 5

    # 2. MFI zone (0-15) — oversold/overbought bolge daha anlamli
    mfi_at_signal = mfi_curr
    if direction == 'BUY':
        if mfi_at_signal < 20:
            quality += 15
        elif mfi_at_signal < 30:
            quality += 10
        elif mfi_at_signal < 40:
            quality += 5
    else:  # SELL
        if mfi_at_signal > 80:
            quality += 15
        elif mfi_at_signal > 70:
            quality += 10
        elif mfi_at_signal > 60:
            quality += 5

    # 3. Fiyat-MFI kontrast (0-15)
    if curr_idx < n and prev_idx < n:
        close_vals = df['close'].values
        atr_val = atr.iloc[curr_idx] if curr_idx < len(atr) else atr.iloc[-1]
        if atr_val > 0:
            price_change_atr = abs(close_vals[curr_idx] - close_vals[prev_idx]) / atr_val
            if price_change_atr >= 3.0:
                quality += 15
            elif price_change_atr >= 2.0:
                quality += 10
            elif price_change_atr >= 1.0:
                quality += 5

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Hacim teyidi (0-15)
    _, _, _, vol_score = _check_volume_confirmation(df, signal_bar, obv_ema)
    quality += min(vol_score, 15)

    # Classic > Hidden bonus
    if div_type == 'MFI_CLASSIC':
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# ADX EXHAUSTION (slope bazli)
# =============================================================================

def detect_adx_exhaustion(df, adx, atr, obv_ema):
    """
    ADX Exhaustion tespiti (slope bazli).

    Fiyat trendi devam + ADX dusuyor → trend bitiyor.
    Bearish: price slope > 0 + ADX slope < threshold → SELL (yukselis bitiyor)
    Bullish: price slope < 0 + ADX slope < threshold → BUY (dusus bitiyor)
    """
    signals = []
    n = len(df)
    cfg = DIV_CFG
    lookback = cfg['adx_lookback']

    if n < lookback + 5:
        return signals

    adx_vals = adx.values
    close_vals = df['close'].values

    # Son birden fazla pencere kontrol et (en guncel oncelikli)
    for end_offset in range(0, min(5, n - lookback)):
        end_bar = n - 1 - end_offset
        start_bar = end_bar - lookback + 1

        if start_bar < 0:
            continue

        adx_window = adx_vals[start_bar:end_bar + 1]
        close_window = close_vals[start_bar:end_bar + 1]

        if np.any(np.isnan(adx_window)) or np.any(np.isnan(close_window)):
            continue

        x = np.arange(lookback, dtype=float)

        try:
            adx_slope = np.polyfit(x, adx_window, 1)[0]
            close_slope = np.polyfit(x, close_window, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            continue

        atr_val = atr.iloc[end_bar] if end_bar < len(atr) else atr.iloc[-1]
        if atr_val <= 0:
            continue

        close_slope_norm = close_slope / atr_val

        # ADX dusuyor mu?
        if adx_slope >= cfg['adx_min_slope']:
            continue

        # Son ADX degeri en az 20 olmali (anlamli trend vardi)
        if adx_vals[end_bar] < 20:
            continue

        direction = None
        if close_slope_norm > 0.1:
            direction = 'SELL'  # Yukselis trendi bitiyor
        elif close_slope_norm < -0.1:
            direction = 'BUY'  # Dusus trendi bitiyor

        if direction is None:
            continue

        quality = _adx_exhaust_quality(
            adx_slope, adx_vals[end_bar], close_slope_norm,
            end_bar, df, atr, obv_ema, direction
        )

        signals.append(DivergenceSignal(
            bar_idx=end_bar,
            direction=direction,
            div_type='ADX_EXHAUST',
            quality=quality,
            details={
                'adx_slope': round(adx_slope, 3),
                'adx_value': round(adx_vals[end_bar], 1),
                'close_slope': round(close_slope_norm, 3),
                'lookback': lookback,
                'start_bar': start_bar,
                'end_bar': end_bar,
            },
        ))

        break  # Ilk gecerli pencere yeterli

    return signals


def _adx_exhaust_quality(adx_slope, adx_value, close_slope_norm,
                         signal_bar, df, atr, obv_ema, direction):
    """ADX Exhaustion kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. ADX slope gucu (0-25) — ne kadar hizli dusuyorsa o kadar guclu
    abs_slope = abs(adx_slope)
    if abs_slope >= 2.0:
        quality += 25
    elif abs_slope >= 1.0:
        quality += 20
    elif abs_slope >= 0.5:
        quality += 15
    else:
        quality += 10

    # 2. ADX seviyesi (0-20) — yuksek ADX'ten dusus daha anlamli
    if adx_value >= 40:
        quality += 20
    elif adx_value >= 30:
        quality += 15
    elif adx_value >= 25:
        quality += 10
    else:
        quality += 5

    # 3. Fiyat slope gucu (0-15)
    abs_cs = abs(close_slope_norm)
    if abs_cs >= 1.0:
        quality += 15
    elif abs_cs >= 0.5:
        quality += 10
    elif abs_cs >= 0.2:
        quality += 5

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Hacim teyidi (0-15)
    _, _, _, vol_score = _check_volume_confirmation(df, signal_bar, obv_ema)
    quality += min(vol_score, 15)

    return max(min(quality, 100), 0)


# =============================================================================
# FIYAT-HACIM UYUMSUZLUGU
# =============================================================================

def detect_price_volume_divergence(df, atr, obv_ema):
    """
    Fiyat-Hacim trend uyumsuzlugu.

    Bullish: close slope < 0 AND volume slope < 0 (satis baskisi azaliyor)
    Bearish: close slope > 0 AND volume slope < 0 (alim ilgisi azaliyor)
    """
    signals = []
    n = len(df)
    cfg = DIV_CFG
    lookback = cfg['pv_div_lookback']
    min_bars = cfg['pv_div_min_bars']

    if n < lookback + min_bars:
        return signals

    close_vals = df['close'].values
    vol_vals = df['volume'].values

    for end_offset in range(0, min(10, n - lookback)):
        end_bar = n - 1 - end_offset
        start_bar = end_bar - lookback + 1

        if start_bar < 0:
            continue

        close_window = close_vals[start_bar:end_bar + 1]
        vol_window = vol_vals[start_bar:end_bar + 1]

        if len(close_window) < min_bars:
            continue

        if np.any(np.isnan(close_window)) or np.any(close_window <= 0):
            continue
        if np.any(np.isnan(vol_window)) or np.all(vol_window <= 0):
            continue

        x = np.arange(len(close_window), dtype=float)

        try:
            close_slope = np.polyfit(x, close_window, 1)[0]
            vol_slope = np.polyfit(x, vol_window, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            continue

        atr_val = atr.iloc[end_bar] if end_bar < len(atr) else atr.iloc[-1]
        if atr_val <= 0:
            continue

        close_slope_norm = close_slope / atr_val
        avg_vol = vol_window.mean()
        vol_slope_norm = vol_slope / avg_vol if avg_vol > 0 else 0

        div_type = None
        direction = None

        # Bullish PV Div: fiyat dusuyor + hacim azaliyor
        if close_slope_norm < -0.1 and vol_slope_norm < -0.05:
            div_type = 'PRICE_VOLUME'
            direction = 'BUY'
        # Bearish PV Div: fiyat yukseliyor + hacim azaliyor
        elif close_slope_norm > 0.1 and vol_slope_norm < -0.05:
            div_type = 'PRICE_VOLUME'
            direction = 'SELL'

        if div_type is None:
            continue

        quality = _pv_div_quality(
            close_slope_norm, vol_slope_norm, end_bar,
            df, atr, obv_ema, direction, lookback
        )

        signals.append(DivergenceSignal(
            bar_idx=end_bar,
            direction=direction,
            div_type=div_type,
            quality=quality,
            details={
                'close_slope': round(close_slope_norm, 3),
                'vol_slope': round(vol_slope_norm, 3),
                'lookback': lookback,
                'start_bar': start_bar,
                'end_bar': end_bar,
            },
        ))

        break  # Ilk gecerli pencere yeterli

    return signals


def _pv_div_quality(close_slope, vol_slope, signal_bar,
                    df, atr, obv_ema, direction, lookback):
    """Fiyat-Hacim divergence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. Fiyat trend gucu (0-25)
    abs_cs = abs(close_slope)
    if abs_cs >= 1.0:
        quality += 25
    elif abs_cs >= 0.5:
        quality += 20
    elif abs_cs >= 0.3:
        quality += 15
    elif abs_cs >= 0.15:
        quality += 10
    else:
        quality += 5

    # 2. Hacim trend gucu (0-25)
    abs_vs = abs(vol_slope)
    if abs_vs >= 0.5:
        quality += 25
    elif abs_vs >= 0.3:
        quality += 20
    elif abs_vs >= 0.15:
        quality += 15
    elif abs_vs >= 0.08:
        quality += 10
    else:
        quality += 5

    # 3. OBV teyidi (0-20)
    if signal_bar >= 5 and signal_bar < len(obv_ema):
        obv_vals = obv_ema.values
        if not np.isnan(obv_vals[signal_bar]) and not np.isnan(obv_vals[signal_bar - 5]):
            obv_diff = obv_vals[signal_bar] - obv_vals[signal_bar - 5]
            if direction == 'BUY' and obv_diff > 0:
                quality += 20
            elif direction == 'BUY' and obv_diff <= 0:
                quality += 8
            elif direction == 'SELL' and obv_diff < 0:
                quality += 20
            elif direction == 'SELL' and obv_diff >= 0:
                quality += 8

    # 4. Sure (0-15) — daha uzun divergence daha anlamli
    if lookback >= 30:
        quality += 15
    elif lookback >= 20:
        quality += 10
    else:
        quality += 5

    # 5. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# UCLU UYUMSUZLUK (Triple Confluence)
# =============================================================================

def detect_triple_confluence(rsi_signals, macd_signals, mfi_signals, df, atr, obv_ema):
    """
    RSI + MACD + (opsiyonel MFI) sinyallerini esle ve hacim spike teyidi ekle.
    Sadece CLASSIC tipler (hidden karismasin).

    Esleme: ayni yon + bar_idx farki <= 2
    Super confluence: RSI + MACD + MFI ayni bolgede → ekstra bonus.
    """
    signals = []

    rsi_classic = [s for s in rsi_signals if s.div_type == 'RSI_CLASSIC']
    macd_classic = [s for s in macd_signals if s.div_type == 'MACD_CLASSIC']
    mfi_classic = [s for s in mfi_signals if s.div_type == 'MFI_CLASSIC']

    for rsi_sig in rsi_classic:
        for macd_sig in macd_classic:
            if rsi_sig.direction != macd_sig.direction:
                continue
            if abs(rsi_sig.bar_idx - macd_sig.bar_idx) > 2:
                continue

            signal_bar = max(rsi_sig.bar_idx, macd_sig.bar_idx)
            direction = rsi_sig.direction

            # MFI de bu bolgede mi?
            has_mfi = False
            for mfi_sig in mfi_classic:
                if mfi_sig.direction == direction and abs(mfi_sig.bar_idx - signal_bar) <= 2:
                    has_mfi = True
                    break

            vol_spike, vol_ratio, obv_trend, vol_score = _check_volume_confirmation(
                df, signal_bar, obv_ema
            )

            quality = _triple_quality(
                rsi_sig, macd_sig, vol_spike, vol_ratio, signal_bar, df, has_mfi
            )

            signals.append(DivergenceSignal(
                bar_idx=signal_bar,
                direction=direction,
                div_type='TRIPLE',
                quality=quality,
                details={
                    'rsi_div': rsi_sig.details,
                    'macd_div': macd_sig.details,
                    'vol_spike': vol_spike,
                    'vol_ratio': round(vol_ratio, 2),
                    'obv_trend': obv_trend,
                    'has_mfi': has_mfi,
                },
            ))

    return signals


def _triple_quality(rsi_sig, macd_sig, vol_spike, vol_ratio, signal_bar, df, has_mfi=False):
    """Triple confluence kalite skoru (0-100)."""
    quality = 0
    n = len(df)

    # 1. RSI bileseni (0-25)
    quality += min(int(rsi_sig.quality * 0.35), 25)

    # 2. MACD bileseni (0-25)
    quality += min(int(macd_sig.quality * 0.35), 25)

    # 3. Hacim spike (0-20)
    if vol_ratio >= 2.0:
        quality += 20
    elif vol_ratio >= 1.5:
        quality += 15
    elif vol_ratio >= 1.2:
        quality += 10
    elif vol_spike:
        quality += 5

    # 4. Recency (0-15)
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 2:
        quality += 15
    elif bars_from_end <= 5:
        quality += 10
    elif bars_from_end <= 10:
        quality += 5

    # 5. Confluence bonus (15) + MFI super confluence (+5)
    quality += 15
    if has_mfi:
        quality += 5

    return max(min(quality, 100), 0)


# =============================================================================
# ANA TARAMA FONKSIYONU
# =============================================================================

def scan_divergences(df, scan_bars=None):
    """
    Tum divergence tiplerini tara.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        scan_bars: Son kac bar taranacak (default: DIV_CFG['scan_bars'])

    Returns: dict with keys: rsi, macd, obv, mfi, adx, triple, pv
    """
    if scan_bars is None:
        scan_bars = DIV_CFG['scan_bars']

    cfg = DIV_CFG
    n = len(df)

    # Indicatorleri hesapla
    atr = _calc_atr(df, cfg['atr_len'])
    rsi = _calc_rsi(df['close'], cfg['rsi_len'])
    macd_line, signal_line, macd_hist = _calc_macd(
        df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal']
    )
    obv = _calc_obv(df)
    obv_ema_series = ema(obv, cfg['obv_ema_len'])
    mfi = _calc_mfi(df, cfg['mfi_len'])
    adx = _calc_adx(df, cfg['adx_len'])

    # Swing'leri bul (pivot yerine)
    swing_lows, swing_highs = _find_swings(df['close'], cfg['swing_order'])

    # 7 detector calistir
    rsi_signals = detect_rsi_divergence(df, rsi, swing_lows, swing_highs, atr, obv_ema_series)
    macd_signals = detect_macd_divergence(df, macd_hist, macd_line, swing_lows, swing_highs, atr, obv_ema_series)
    obv_signals = detect_obv_divergence(df, obv, swing_lows, swing_highs, atr, obv_ema_series)
    mfi_signals = detect_mfi_divergence(df, mfi, swing_lows, swing_highs, atr, obv_ema_series)
    adx_signals = detect_adx_exhaustion(df, adx, atr, obv_ema_series)
    triple_signals = detect_triple_confluence(rsi_signals, macd_signals, mfi_signals, df, atr, obv_ema_series)
    pv_signals = detect_price_volume_divergence(df, atr, obv_ema_series)

    # scan_bars filtresi
    cutoff = n - 1 - scan_bars

    def _filter_and_dedup(sigs):
        filtered = [s for s in sigs if s.bar_idx > cutoff]
        seen = set()
        unique = []
        for s in filtered:
            key = (s.bar_idx, s.div_type, s.direction)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        unique.sort(key=lambda s: s.quality, reverse=True)
        return unique

    return {
        'rsi': _filter_and_dedup(rsi_signals),
        'macd': _filter_and_dedup(macd_signals),
        'obv': _filter_and_dedup(obv_signals),
        'mfi': _filter_and_dedup(mfi_signals),
        'adx': _filter_and_dedup(adx_signals),
        'triple': _filter_and_dedup(triple_signals),
        'pv': _filter_and_dedup(pv_signals),
    }
