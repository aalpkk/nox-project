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
from enum import Enum

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
    'pv_min_vol_ratio': 1.5,   # Hacim en az SMA * 1.5 olmali
    'pv_max_range_ratio': 0.5, # Bar range / ATR < 0.5 → kucuk bar (emilim)
    # Structural swing (Faz 1)
    'structural_min_atr_dist': 1.0,
    'structural_min_bar_gap': 5,
    'trigger_swing_order': 3,
    # State machine (Faz 1)
    'setup_fresh_bars': 2,      # 0-2: fresh
    'setup_active_bars': 6,     # 3-6: active, >6: stale
    'trigger_ema_len': 21,
    'trigger_vol_mult': 1.5,
    # Confirmation layer (Faz 1)
    'confirmation_obv_slope_bars': 5,
    'confirmation_max_mod': 15,
    # Location quality (Faz 2)
    'loc_bb_len': 20,
    'loc_bb_mult': 2.0,
    'loc_sr_lookback': 100,
    'loc_sr_atr_proximity': 1.5,
    'loc_sr_min_touches': 2,
    # Regime (Faz 2)
    'regime_reversal_bonus_choppy': 10,
    'regime_reversal_penalty_trend': -8,
    'regime_continuation_bonus_trend': 10,
    'regime_continuation_penalty_choppy': -8,
    'regime_exhaust_bonus_trend': 12,
    'regime_exhaust_penalty_choppy': -5,
    # Risk-Reward (Faz 2)
    'rr_default_target_atr': 2.0,
}


# =============================================================================
# VERI YAPISI
# =============================================================================

class DivBucket(Enum):
    REVERSAL = 'REVERSAL'
    CONTINUATION = 'CONTINUATION'
    CONFIRMATION = 'CONFIRMATION'
    EXHAUSTION = 'EXHAUSTION'

class SetupState(Enum):
    SETUP = 'SETUP'
    TRIGGERED = 'TRIGGERED'
    STALE = 'STALE'
    INVALIDATED = 'INVALIDATED'

BUCKET_MAP = {
    'RSI_CLASSIC': 'REVERSAL',    'RSI_HIDDEN': 'CONTINUATION',
    'MACD_CLASSIC': 'REVERSAL',   'MACD_HIDDEN': 'CONTINUATION',
    'MFI_CLASSIC': 'REVERSAL',    'MFI_HIDDEN': 'CONTINUATION',
    'OBV_CLASSIC': 'CONFIRMATION','OBV_HIDDEN': 'CONFIRMATION',
    'PRICE_VOLUME': 'CONFIRMATION',
    'ADX_EXHAUST': 'EXHAUSTION',
    'TRIPLE': 'REVERSAL',
}


@dataclass
class DivergenceSignal:
    """Legacy sinyal veri yapisi (geriye uyumluluk)."""
    bar_idx: int          # Sinyal bari (swing'in signal_bar'i)
    direction: str        # 'BUY' veya 'SELL'
    div_type: str         # RSI_CLASSIC, RSI_HIDDEN, MACD_CLASSIC, MACD_HIDDEN,
                          # OBV_CLASSIC, OBV_HIDDEN, MFI_CLASSIC, MFI_HIDDEN,
                          # ADX_EXHAUST, TRIPLE, PRICE_VOLUME
    quality: int          # 0-100
    details: dict = field(default_factory=dict)


@dataclass
class DivergenceSetup:
    """Faz 2 sinyal veri yapisi — bucket, state, structural, location, regime, label, RR."""
    bar_idx: int
    direction: str            # BUY / SELL
    div_type: str
    quality: int              # 0-100
    bucket: str               # REVERSAL / CONTINUATION / EXHAUSTION / CONFIRMATION
    state: str                # SETUP / TRIGGERED / STALE / INVALIDATED
    age: int                  # bar sayisi (0 = guncel)
    trigger_type: str         # NONE / SWING_BREAK / EMA_RECLAIM / VOLUME_REVERSAL
    trigger_bar: int          # tetik bari (-1 = yok)
    invalidation_level: float # setup'i gecersiz kilacak fiyat
    structural: bool          # yapisal swing mi
    confirmation_mod: int     # -15..+15 OBV/PV teyit modifiyesi
    details: dict = field(default_factory=dict)
    # Faz 2 alanlari
    location_q: int = 0       # 0-25
    regime: int = -1           # -1=unknown, 0=CHOPPY, 1=GRI, 2=TREND, 3=FULL_TREND
    regime_mod: int = 0
    signal_label: str = 'WATCH'
    risk_score: int = 0
    rr_ratio: float = 0.0


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


def _find_structural_swings(series, atr, order=2, min_atr_dist=1.0, min_bar_gap=5):
    """
    Yapisal swing'leri bul. ATR mesafesi ve bar arasi filtresi uygular.

    Returns: (lows, highs)
        lows:  list of (swing_bar, signal_bar, price, structural)
        highs: list of (swing_bar, signal_bar, price, structural)
    """
    raw_lows, raw_highs = _find_swings(series, order)
    atr_vals = atr.values

    def _mark_structural(swings):
        result = []
        last_accepted_idx = -999
        last_accepted_price = None
        for sw in swings:
            bar, sig, price = sw
            if bar >= len(atr_vals) or np.isnan(atr_vals[bar]):
                result.append((bar, sig, price, False))
                continue
            atr_at = atr_vals[bar]
            if atr_at <= 0:
                result.append((bar, sig, price, False))
                continue
            bar_ok = (bar - last_accepted_idx) >= min_bar_gap
            dist_ok = last_accepted_price is None or \
                      abs(price - last_accepted_price) >= min_atr_dist * atr_at
            if bar_ok and dist_ok:
                result.append((bar, sig, price, True))
                last_accepted_idx = bar
                last_accepted_price = price
            else:
                result.append((bar, sig, price, False))
        return result

    return _mark_structural(raw_lows), _mark_structural(raw_highs)


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
# FAZ 1: YENi KALiTE SKORLAMA (3 Sutun + Confirmation Mod)
# =============================================================================

def _structure_quality(prev_idx, curr_idx, signal_bar, n, structural_flag):
    """Structure sutunu: 0-25. Yapisal swing, kontrast, recency, bar ayrimi."""
    q = 0
    # Yapisal swing bonusu
    if structural_flag:
        q += 8
    # Bar ayrimi (0-5) — swing'ler arasi yeterli mesafe
    gap = curr_idx - prev_idx
    if gap >= 15:
        q += 5
    elif gap >= 10:
        q += 4
    elif gap >= 7:
        q += 3
    elif gap >= 5:
        q += 2
    else:
        q += 1
    # Recency (0-5) — sinyalin ne kadar yeni oldugu
    bars_from_end = n - 1 - signal_bar
    if bars_from_end <= 1:
        q += 5
    elif bars_from_end <= 3:
        q += 4
    elif bars_from_end <= 5:
        q += 3
    elif bars_from_end <= 8:
        q += 2
    elif bars_from_end <= 12:
        q += 1
    return min(q, 25)


def _momentum_quality_rsi(rsi_prev, rsi_curr, rsi_diff, direction, div_type):
    """RSI momentum sutunu: 0-25."""
    q = 0
    # Osc farki buyuklugu (0-10)
    if rsi_diff >= 20:
        q += 10
    elif rsi_diff >= 15:
        q += 8
    elif rsi_diff >= 10:
        q += 6
    elif rsi_diff >= 5:
        q += 4
    else:
        q += 2
    # Zone baglami (0-8) — oversold/overbought
    if direction == 'BUY':
        if rsi_curr < 30:
            q += 8
        elif rsi_curr < 40:
            q += 5
        elif rsi_curr < 50:
            q += 3
    else:
        if rsi_curr > 70:
            q += 8
        elif rsi_curr > 60:
            q += 5
        elif rsi_curr > 50:
            q += 3
    # Classic bonusu (0-4)
    if div_type == 'RSI_CLASSIC':
        q += 4
    # Tip bonusu (0-3) — RSI genel guvenilirlik
    q += 3
    return min(q, 25)


def _momentum_quality_macd(hist_prev, hist_curr, hist_diff, atr_val, macd_vals,
                           prev_idx, curr_idx, direction, div_type):
    """MACD momentum sutunu: 0-25."""
    q = 0
    # Histogram buyuklugu (0-10)
    if atr_val > 0:
        ratio = hist_diff / atr_val
        if ratio >= 0.5:
            q += 10
        elif ratio >= 0.3:
            q += 8
        elif ratio >= 0.15:
            q += 6
        elif ratio >= 0.08:
            q += 4
        else:
            q += 2
    # Zero-line kontekst (0-8)
    if direction == 'BUY':
        if hist_curr < 0 and hist_curr > hist_prev:
            q += 8
        elif hist_curr < 0:
            q += 5
        else:
            q += 2
    else:
        if hist_curr > 0 and hist_curr < hist_prev:
            q += 8
        elif hist_curr > 0:
            q += 5
        else:
            q += 2
    # MACD line teyidi (0-4)
    if curr_idx < len(macd_vals) and prev_idx < len(macd_vals):
        ml_prev = macd_vals[prev_idx]
        ml_curr = macd_vals[curr_idx]
        if not (np.isnan(ml_prev) or np.isnan(ml_curr)):
            if direction == 'BUY' and ml_curr > ml_prev:
                q += 4
            elif direction == 'SELL' and ml_curr < ml_prev:
                q += 4
            else:
                q += 1
    # Classic bonusu (0-3)
    if div_type == 'MACD_CLASSIC':
        q += 3
    return min(q, 25)


def _momentum_quality_mfi(mfi_prev, mfi_curr, mfi_diff, direction, div_type):
    """MFI momentum sutunu: 0-25."""
    q = 0
    # MFI farki buyuklugu (0-10)
    if mfi_diff >= 20:
        q += 10
    elif mfi_diff >= 15:
        q += 8
    elif mfi_diff >= 10:
        q += 6
    elif mfi_diff >= 5:
        q += 4
    else:
        q += 2
    # Zone baglami (0-8)
    if direction == 'BUY':
        if mfi_curr < 20:
            q += 8
        elif mfi_curr < 30:
            q += 5
        elif mfi_curr < 40:
            q += 3
    else:
        if mfi_curr > 80:
            q += 8
        elif mfi_curr > 70:
            q += 5
        elif mfi_curr > 60:
            q += 3
    # Classic bonusu (0-4)
    if div_type == 'MFI_CLASSIC':
        q += 4
    # Tip bonusu (0-3)
    q += 3
    return min(q, 25)


def _momentum_quality_adx(adx_slope, adx_value, close_slope_norm, direction):
    """ADX momentum sutunu: 0-25."""
    q = 0
    # ADX slope gucu (0-10)
    abs_slope = abs(adx_slope)
    if abs_slope >= 2.0:
        q += 10
    elif abs_slope >= 1.0:
        q += 8
    elif abs_slope >= 0.5:
        q += 6
    else:
        q += 4
    # ADX seviyesi (0-8) — yuksek ADX'ten dusus daha anlamli
    if adx_value >= 40:
        q += 8
    elif adx_value >= 30:
        q += 6
    elif adx_value >= 25:
        q += 4
    else:
        q += 2
    # Fiyat slope gucu (0-7)
    abs_cs = abs(close_slope_norm)
    if abs_cs >= 1.0:
        q += 7
    elif abs_cs >= 0.5:
        q += 5
    elif abs_cs >= 0.2:
        q += 3
    return min(q, 25)


def _participation_quality(df, signal_bar, obv_ema, direction):
    """Participation sutunu: 0-25. Hacim spike + OBV EMA trend uyumu."""
    q = 0
    n = len(df)
    if signal_bar < 0 or signal_bar >= n:
        return 0

    # Hacim spike (0-12)
    vol = df['volume'].values
    vol_sma = sma(df['volume'], DIV_CFG['vol_sma_len']).values
    v = vol[signal_bar]
    vs = vol_sma[signal_bar] if signal_bar < len(vol_sma) and not np.isnan(vol_sma[signal_bar]) else 1.0
    if vs <= 0:
        vs = 1.0
    vol_ratio = v / vs
    if vol_ratio >= 2.5:
        q += 12
    elif vol_ratio >= 2.0:
        q += 10
    elif vol_ratio >= 1.5:
        q += 8
    elif vol_ratio >= 1.2:
        q += 5
    elif vol_ratio >= 1.0:
        q += 2

    # OBV EMA trend uyumu (0-13)
    if signal_bar >= 5 and signal_bar < len(obv_ema):
        obv_vals = obv_ema.values
        if not np.isnan(obv_vals[signal_bar]) and not np.isnan(obv_vals[signal_bar - 5]):
            obv_diff = obv_vals[signal_bar] - obv_vals[signal_bar - 5]
            if direction == 'BUY' and obv_diff > 0:
                q += 13
            elif direction == 'SELL' and obv_diff < 0:
                q += 13
            elif (direction == 'BUY' and obv_diff <= 0) or \
                 (direction == 'SELL' and obv_diff >= 0):
                q += 3

    return min(q, 25)


def _compute_quality_v2(structure_q, momentum_q, participation_q, location_q, confirmation_mod):
    """Toplam kalite: 4 sutun (0-100) + confirmation mod (-15..+15) → 0-100."""
    raw = structure_q + momentum_q + participation_q + location_q + confirmation_mod
    return max(min(raw, 100), 0)


def _compute_confirmation_modifier(df, bar_idx, direction, obv, obv_ema, atr):
    """
    OBV + PV teyit modifiyesi. Returns: int (-15..+15).
    Primary sinyallerin kalitesini arttirir/azaltir.
    """
    cfg = DIV_CFG
    mod = 0
    n = len(df)
    slope_bars = cfg['confirmation_obv_slope_bars']

    # OBV EMA slope kontrolu (±8)
    if bar_idx >= slope_bars and bar_idx < len(obv_ema):
        obv_vals = obv_ema.values
        if not np.isnan(obv_vals[bar_idx]) and not np.isnan(obv_vals[bar_idx - slope_bars]):
            obv_diff = obv_vals[bar_idx] - obv_vals[bar_idx - slope_bars]
            if direction == 'BUY' and obv_diff > 0:
                mod += 8
            elif direction == 'SELL' and obv_diff < 0:
                mod += 8
            elif direction == 'BUY' and obv_diff < 0:
                mod -= 5
            elif direction == 'SELL' and obv_diff > 0:
                mod -= 5

    # Hacim + fiyat yapisi kontrolu — PV tarzi absorption/rejection (±7)
    if bar_idx < n and bar_idx > 0:
        h = df['high'].values[bar_idx]
        l = df['low'].values[bar_idx]
        o = df['open'].values[bar_idx]
        c = df['close'].values[bar_idx]
        v = df['volume'].values[bar_idx]
        vol_sma_val = sma(df['volume'], cfg['vol_sma_len']).values
        vs = vol_sma_val[bar_idx] if bar_idx < len(vol_sma_val) and not np.isnan(vol_sma_val[bar_idx]) else 1.0
        if vs > 0:
            vol_r = v / vs
        else:
            vol_r = 1.0
        atr_val = atr.values[bar_idx] if bar_idx < len(atr) and not np.isnan(atr.values[bar_idx]) else 1.0

        if atr_val > 0 and (h - l) > 0:
            range_atr = (h - l) / atr_val
            body = abs(c - o)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            # BUY: hacim yuksek + dip absorption veya rejection
            if direction == 'BUY' and vol_r >= 1.3:
                if range_atr < 0.5 and c >= (h + l) / 2:
                    mod += 7  # Absorption
                elif body > 0 and lower_wick / body >= 1.5:
                    mod += 5  # Rejection
            elif direction == 'SELL' and vol_r >= 1.3:
                if range_atr < 0.5 and c <= (h + l) / 2:
                    mod += 7  # Distribution
                elif body > 0 and upper_wick / body >= 1.5:
                    mod += 5  # Rejection

    max_mod = cfg['confirmation_max_mod']
    return max(min(mod, max_mod), -max_mod)


def _compute_invalidation_level(details, direction):
    """Invalidation seviyesi hesapla."""
    prev_price = details.get('prev_price', 0)
    curr_price = details.get('curr_price', 0)
    if direction == 'BUY':
        return min(prev_price, curr_price) if prev_price and curr_price else 0
    else:
        return max(prev_price, curr_price) if prev_price and curr_price else 0


# =============================================================================
# FAZ 2: LOCATION QUALITY (4. Sutun, 0-25)
# =============================================================================

def _find_sr_levels(df, atr, lookback=100, min_touches=2):
    """
    Son lookback bar'daki S/R seviyelerini bul.
    Swing high/low'lari cluster'la, min_touches dokunusu olan = S/R.

    Returns: list of (price, strength, 'support'|'resistance')
    """
    n = len(df)
    start = max(0, n - lookback)
    close_slice = df['close'].iloc[start:n]
    if len(close_slice) < 10:
        return []

    lows, highs = _find_swings(close_slice, order=2)
    atr_vals = atr.values
    last_atr = atr_vals[n - 1] if n > 0 and not np.isnan(atr_vals[n - 1]) else 1.0
    if last_atr <= 0:
        last_atr = 1.0
    cluster_dist = 0.5 * last_atr

    def _cluster(swings, sr_type):
        if not swings:
            return []
        prices = [sw[2] for sw in swings]
        prices.sort()
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if p - current[-1] <= cluster_dist:
                current.append(p)
            else:
                if len(current) >= min_touches:
                    clusters.append((np.mean(current), len(current), sr_type))
                current = [p]
        if len(current) >= min_touches:
            clusters.append((np.mean(current), len(current), sr_type))
        return clusters

    supports = _cluster(lows, 'support')
    resistances = _cluster(highs, 'resistance')
    return supports + resistances


def _location_quality(df, signal_bar, direction, atr, sr_levels, ema21, bb_upper, bb_lower):
    """
    Location sutunu: 0-25. Uc alt bilesen:
    - S/R Yakinligi (0-10): BUY→yakin destek, SELL→yakin direnc
    - EMA Mesafesi (0-8): BUY: close<=EMA=8, SELL ayna
    - BB Pozisyonu (0-7): BUY: close<=BB_alt=7, SELL ayna
    """
    n = len(df)
    if signal_bar < 0 or signal_bar >= n:
        return 0

    close = df['close'].values[signal_bar]
    atr_val = atr.values[signal_bar] if signal_bar < len(atr) and not np.isnan(atr.values[signal_bar]) else 1.0
    if atr_val <= 0:
        atr_val = 1.0

    q = 0

    # 1. S/R Yakinligi (0-10)
    if sr_levels:
        if direction == 'BUY':
            relevant = [(p, s) for p, s, t in sr_levels if t == 'support' and p <= close]
        else:
            relevant = [(p, s) for p, s, t in sr_levels if t == 'resistance' and p >= close]

        if relevant:
            dists = [abs(close - p) / atr_val for p, s in relevant]
            min_dist = min(dists)
            if min_dist <= 0.5:
                q += 10
            elif min_dist <= 1.0:
                q += 7
            elif min_dist <= 1.5:
                q += 5
            elif min_dist <= 2.0:
                q += 3

    # 2. EMA Mesafesi (0-8)
    if signal_bar < len(ema21) and not np.isnan(ema21.values[signal_bar]):
        ema_val = ema21.values[signal_bar]
        ema_dist = (close - ema_val) / atr_val
        if direction == 'BUY':
            if ema_dist <= 0:
                q += 8
            elif ema_dist <= 0.5:
                q += 6
            elif ema_dist <= 1.0:
                q += 4
            elif ema_dist <= 2.0:
                q += 2
        else:
            if ema_dist >= 0:
                q += 8
            elif ema_dist >= -0.5:
                q += 6
            elif ema_dist >= -1.0:
                q += 4
            elif ema_dist >= -2.0:
                q += 2

    # 3. BB Pozisyonu (0-7)
    if signal_bar < len(bb_upper) and signal_bar < len(bb_lower):
        bbu = bb_upper.values[signal_bar]
        bbl = bb_lower.values[signal_bar]
        if not (np.isnan(bbu) or np.isnan(bbl)) and bbu > bbl:
            bb_range = bbu - bbl
            if direction == 'BUY':
                if close <= bbl:
                    q += 7
                elif close <= bbl + 0.3 * bb_range:
                    q += 5
                elif close <= bbl + 0.5 * bb_range:
                    q += 3
            else:
                if close >= bbu:
                    q += 7
                elif close >= bbu - 0.3 * bb_range:
                    q += 5
                elif close >= bbu - 0.5 * bb_range:
                    q += 3

    return min(q, 25)


# =============================================================================
# FAZ 2: REGIME MODIFIER
# =============================================================================

def _compute_regime_modifier(bucket, regime_val, cfg):
    """
    Rejim bazli kalite modifiyesi.
    regime_val: 0=CHOPPY, 1=GRI_BOLGE, 2=TREND, 3=FULL_TREND
    """
    if regime_val < 0:
        return 0

    if bucket == 'REVERSAL':
        if regime_val <= 1:
            return cfg['regime_reversal_bonus_choppy']
        else:
            return cfg['regime_reversal_penalty_trend']
    elif bucket == 'CONTINUATION':
        if regime_val >= 2:
            return cfg['regime_continuation_bonus_trend']
        else:
            return cfg['regime_continuation_penalty_choppy']
    elif bucket == 'EXHAUSTION':
        if regime_val >= 2:
            return cfg['regime_exhaust_bonus_trend']
        else:
            return cfg['regime_exhaust_penalty_choppy']
    # CONFIRMATION → 0
    return 0


# =============================================================================
# FAZ 2: SIGNAL LABELS
# =============================================================================

def _compute_signal_label(setup, regime_val):
    """
    Sinyal amac etiketi hesapla.
    Returns: ENTRY_LONG | ENTRY_SHORT | REDUCE | COVER | WATCH
    """
    bucket = setup.bucket
    direction = setup.direction
    state = setup.state
    quality = setup.quality

    # INVALIDATED/STALE + dusuk kalite → WATCH
    if state in ('INVALIDATED', 'STALE') and quality < 50:
        return 'WATCH'

    choppy = regime_val <= 1  # choppy veya gri bolge
    trend = regime_val >= 2   # trend veya full trend

    if bucket == 'CONFIRMATION':
        if state == 'TRIGGERED' or quality >= 50:
            return 'ENTRY_LONG' if direction == 'BUY' else 'ENTRY_SHORT'
        return 'WATCH'

    if bucket == 'REVERSAL':
        if direction == 'BUY':
            return 'ENTRY_LONG' if choppy else 'COVER'
        else:
            return 'ENTRY_SHORT' if choppy else 'REDUCE'
    elif bucket == 'CONTINUATION':
        if direction == 'BUY':
            return 'ENTRY_LONG' if trend else 'WATCH'
        else:
            return 'ENTRY_SHORT' if trend else 'WATCH'
    elif bucket == 'EXHAUSTION':
        if direction == 'BUY':
            return 'ENTRY_LONG' if choppy else 'COVER'
        else:
            return 'ENTRY_SHORT' if choppy else 'REDUCE'

    return 'WATCH'


# =============================================================================
# FAZ 2: RISK-REWARD
# =============================================================================

def _compute_risk_reward(setup, df, atr, sr_levels):
    """
    Risk/Odul orani ve skoru hesapla.
    Returns: (risk_score, rr_ratio)
    """
    bar_idx = setup.bar_idx
    n = len(df)
    if bar_idx < 0 or bar_idx >= n:
        return 0, 0.0

    close = df['close'].values[bar_idx]
    atr_val = atr.values[bar_idx] if bar_idx < len(atr) and not np.isnan(atr.values[bar_idx]) else 0
    if atr_val <= 0:
        return 0, 0.0

    stop = setup.invalidation_level
    if stop <= 0:
        # Fallback: 1.5 ATR
        if setup.direction == 'BUY':
            stop = close - 1.5 * atr_val
        else:
            stop = close + 1.5 * atr_val

    stop_dist = abs(close - stop)
    if stop_dist <= 0:
        return 0, 0.0

    # Hedef: yon dogrultusundaki en yakin S/R, yoksa entry ± 2×ATR
    cfg = DIV_CFG
    target = None
    if sr_levels:
        if setup.direction == 'BUY':
            candidates = [(p, s) for p, s, t in sr_levels if t == 'resistance' and p > close]
            if candidates:
                target = min(candidates, key=lambda x: x[0])[0]
        else:
            candidates = [(p, s) for p, s, t in sr_levels if t == 'support' and p < close]
            if candidates:
                target = max(candidates, key=lambda x: x[0])[0]

    if target is None:
        target_atr = cfg['rr_default_target_atr']
        if setup.direction == 'BUY':
            target = close + target_atr * atr_val
        else:
            target = close - target_atr * atr_val

    target_dist = abs(target - close)
    if target_dist <= 0:
        return 0, 0.0

    rr = target_dist / stop_dist

    # risk_score hesapla
    if rr >= 3.0:
        score = 100
    elif rr >= 2.5:
        score = 85
    elif rr >= 2.0:
        score = 70
    elif rr >= 1.5:
        score = 55
    elif rr >= 1.0:
        score = 40
    elif rr >= 0.5:
        score = 20
    else:
        score = 5

    # Buyuk stop cezasi
    stop_atr_ratio = stop_dist / atr_val
    if stop_atr_ratio > 3.0:
        score -= 20
    elif stop_atr_ratio > 2.0:
        score -= 10

    score = max(min(score, 100), 0)
    return score, round(rr, 2)


# =============================================================================
# RSI DIVERGENCE (swing bazli)
# =============================================================================

def detect_rsi_divergence(df, rsi, swing_lows, swing_highs, atr, obv_ema,
                          sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    RSI divergence tespiti (swing bazli).
    Returns: list of DivergenceSetup.
    """
    signals = []
    rsi_vals = rsi.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price < prev_price and rsi_curr > rsi_prev:
            div_type = 'RSI_CLASSIC'
        elif curr_price > prev_price and rsi_curr < rsi_prev:
            div_type = 'RSI_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_rsi': round(rsi_prev, 1), 'curr_rsi': round(rsi_curr, 1),
            'rsi_diff': round(rsi_diff, 1),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_rsi(rsi_prev, rsi_curr, rsi_diff, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price > prev_price and rsi_curr < rsi_prev:
            div_type = 'RSI_CLASSIC'
        elif curr_price < prev_price and rsi_curr > rsi_prev:
            div_type = 'RSI_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_rsi': round(rsi_prev, 1), 'curr_rsi': round(rsi_curr, 1),
            'rsi_diff': round(rsi_diff, 1),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_rsi(rsi_prev, rsi_curr, rsi_diff, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    return signals


# =============================================================================
# MACD DIVERGENCE (swing bazli)
# =============================================================================

def detect_macd_divergence(df, macd_hist, macd_line, swing_lows, swing_highs, atr, obv_ema,
                           sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    MACD histogram divergence tespiti (swing bazli).
    Returns: list of DivergenceSetup.
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

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price < prev_price and hist_curr > hist_prev:
            div_type = 'MACD_CLASSIC'
        elif curr_price > prev_price and hist_curr < hist_prev:
            div_type = 'MACD_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_hist': round(hist_prev, 4), 'curr_hist': round(hist_curr, 4),
            'hist_diff': round(hist_diff, 4),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_macd(hist_prev, hist_curr, hist_diff, atr_val,
                                       macd_vals, prev_idx, curr_idx, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price > prev_price and hist_curr < hist_prev:
            div_type = 'MACD_CLASSIC'
        elif curr_price < prev_price and hist_curr > hist_prev:
            div_type = 'MACD_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_hist': round(hist_prev, 4), 'curr_hist': round(hist_curr, 4),
            'hist_diff': round(hist_diff, 4),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_macd(hist_prev, hist_curr, hist_diff, atr_val,
                                       macd_vals, prev_idx, curr_idx, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    return signals


# =============================================================================
# OBV DIVERGENCE (swing bazli)
# =============================================================================

def detect_obv_divergence(df, obv, swing_lows, swing_highs, atr, obv_ema,
                          sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    OBV divergence tespiti (swing bazli).
    Returns: list of DivergenceSetup (bucket=CONFIRMATION).
    """
    signals = []
    obv_vals = obv.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price < prev_price and obv_curr > obv_prev:
            div_type = 'OBV_CLASSIC'
        elif curr_price > prev_price and obv_curr < obv_prev:
            div_type = 'OBV_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_obv': round(obv_prev, 0), 'curr_obv': round(obv_curr, 0),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        # OBV icin momentum = participation-like skor
        avg_vol = df['volume'].iloc[max(0, curr_idx - 20):curr_idx + 1].mean()
        mom_q = 0
        if avg_vol > 0:
            obv_change_norm = abs(obv_curr - obv_prev) / avg_vol
            if obv_change_norm >= 10.0:
                mom_q = 15
            elif obv_change_norm >= 5.0:
                mom_q = 12
            elif obv_change_norm >= 2.0:
                mom_q = 8
            elif obv_change_norm >= 1.0:
                mom_q = 5
            else:
                mom_q = 2
        if div_type == 'OBV_CLASSIC':
            mom_q = min(mom_q + 4, 25)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price > prev_price and obv_curr < obv_prev:
            div_type = 'OBV_CLASSIC'
        elif curr_price < prev_price and obv_curr > obv_prev:
            div_type = 'OBV_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_obv': round(obv_prev, 0), 'curr_obv': round(obv_curr, 0),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        avg_vol = df['volume'].iloc[max(0, curr_idx - 20):curr_idx + 1].mean()
        mom_q = 0
        if avg_vol > 0:
            obv_change_norm = abs(obv_curr - obv_prev) / avg_vol
            if obv_change_norm >= 10.0:
                mom_q = 15
            elif obv_change_norm >= 5.0:
                mom_q = 12
            elif obv_change_norm >= 2.0:
                mom_q = 8
            elif obv_change_norm >= 1.0:
                mom_q = 5
            else:
                mom_q = 2
        if div_type == 'OBV_CLASSIC':
            mom_q = min(mom_q + 4, 25)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    return signals


# =============================================================================
# MFI DIVERGENCE (swing bazli — RSI ile ayni kalip)
# =============================================================================

def detect_mfi_divergence(df, mfi, swing_lows, swing_highs, atr, obv_ema,
                          sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    MFI divergence tespiti (swing bazli).
    Returns: list of DivergenceSetup.
    """
    signals = []
    mfi_vals = mfi.values
    n = len(df)
    cfg = DIV_CFG

    # --- BULLISH (swing low cifleri) ---
    for i in range(1, len(swing_lows)):
        prev_sw = swing_lows[i - 1]
        curr_sw = swing_lows[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price < prev_price and mfi_curr > mfi_prev:
            div_type = 'MFI_CLASSIC'
        elif curr_price > prev_price and mfi_curr < mfi_prev:
            div_type = 'MFI_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_mfi': round(mfi_prev, 1), 'curr_mfi': round(mfi_curr, 1),
            'mfi_diff': round(mfi_diff, 1),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_mfi(mfi_prev, mfi_curr, mfi_diff, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    # --- BEARISH (swing high cifleri) ---
    for i in range(1, len(swing_highs)):
        prev_sw = swing_highs[i - 1]
        curr_sw = swing_highs[i]

        prev_idx, prev_signal, prev_price = prev_sw[:3]
        curr_idx, curr_signal, curr_price = curr_sw[:3]
        structural = curr_sw[3] if len(curr_sw) > 3 else False

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

        if curr_price > prev_price and mfi_curr < mfi_prev:
            div_type = 'MFI_CLASSIC'
        elif curr_price < prev_price and mfi_curr > mfi_prev:
            div_type = 'MFI_HIDDEN'

        if div_type is None:
            continue

        details = {
            'prev_swing_idx': prev_idx, 'curr_swing_idx': curr_idx,
            'prev_price': round(prev_price, 4), 'curr_price': round(curr_price, 4),
            'prev_mfi': round(mfi_prev, 1), 'curr_mfi': round(mfi_curr, 1),
            'mfi_diff': round(mfi_diff, 1),
        }

        struct_q = _structure_quality(prev_idx, curr_idx, curr_signal, n, structural)
        mom_q = _momentum_quality_mfi(mfi_prev, mfi_curr, mfi_diff, direction, div_type)
        part_q = _participation_quality(df, curr_signal, obv_ema, direction)
        loc_q = _location_quality(df, curr_signal, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=curr_signal, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP[div_type],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=_compute_invalidation_level(details, direction),
            structural=structural, confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    return signals


# =============================================================================
# ADX EXHAUSTION (slope bazli)
# =============================================================================

def detect_adx_exhaustion(df, adx, atr, obv_ema,
                          sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    ADX Exhaustion tespiti (slope bazli).
    Returns: list of DivergenceSetup (bucket=EXHAUSTION).
    """
    signals = []
    n = len(df)
    cfg = DIV_CFG
    lookback = cfg['adx_lookback']

    if n < lookback + 5:
        return signals

    adx_vals = adx.values
    close_vals = df['close'].values

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

        if adx_slope >= cfg['adx_min_slope']:
            continue
        if adx_vals[end_bar] < 20:
            continue

        direction = None
        if close_slope_norm > 0.1:
            direction = 'SELL'
        elif close_slope_norm < -0.1:
            direction = 'BUY'

        if direction is None:
            continue

        details = {
            'adx_slope': round(adx_slope, 3),
            'adx_value': round(adx_vals[end_bar], 1),
            'close_slope': round(close_slope_norm, 3),
            'lookback': lookback,
            'start_bar': start_bar,
            'end_bar': end_bar,
        }

        struct_q = _structure_quality(start_bar, end_bar, end_bar, n, False)
        mom_q = _momentum_quality_adx(adx_slope, adx_vals[end_bar], close_slope_norm, direction)
        part_q = _participation_quality(df, end_bar, obv_ema, direction)
        loc_q = _location_quality(df, end_bar, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        quality = _compute_quality_v2(struct_q, mom_q, part_q, loc_q, 0)

        signals.append(DivergenceSetup(
            bar_idx=end_bar, direction=direction, div_type='ADX_EXHAUST',
            quality=quality, bucket=BUCKET_MAP['ADX_EXHAUST'],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=0.0, structural=False,
            confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

        break

    return signals


# =============================================================================
# FIYAT-HACIM UYUMSUZLUGU (VSA — Volume Spread Analysis)
# =============================================================================

def detect_price_volume_divergence(df, atr, obv_ema,
                                   sr_levels=None, ema21=None, bb_upper=None, bb_lower=None):
    """
    Hacim-Fiyat Uyumsuzlugu — VSA mantigi.
    Returns: list of DivergenceSetup (bucket=CONFIRMATION).
    """
    signals = []
    n = len(df)
    cfg = DIV_CFG

    high_vals = df['high'].values
    low_vals = df['low'].values
    open_vals = df['open'].values
    close_vals = df['close'].values
    vol_vals = df['volume'].values

    vol_sma = sma(df['volume'], cfg['vol_sma_len']).values
    atr_vals = atr.values

    min_vol_ratio = cfg.get('pv_min_vol_ratio', 1.5)
    max_range_ratio = cfg.get('pv_max_range_ratio', 0.5)

    for i in range(max(cfg['vol_sma_len'], 1), n):
        h, l, o, c = high_vals[i], low_vals[i], open_vals[i], close_vals[i]
        v = vol_vals[i]
        vs = vol_sma[i]
        atr_val = atr_vals[i]

        if np.isnan(vs) or np.isnan(atr_val) or vs <= 0 or atr_val <= 0:
            continue

        limit_tag = None
        if i >= 1:
            prev_c = close_vals[i - 1]
            if prev_c > 0:
                chg = (c / prev_c - 1)
                if c == h and chg >= 0.095:
                    limit_tag = 'TAVAN'
                elif c == l and chg <= -0.095:
                    limit_tag = 'TABAN'

        vol_ratio = v / vs
        if vol_ratio < min_vol_ratio:
            continue

        bar_range = h - l
        if bar_range <= 0:
            continue
        range_atr = bar_range / atr_val

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        mid = (h + l) / 2

        div_type = None
        direction = None
        sub_type = None

        if range_atr < max_range_ratio:
            if c >= mid:
                div_type = 'PRICE_VOLUME'
                direction = 'BUY'
                sub_type = 'ABSORB'
            else:
                div_type = 'PRICE_VOLUME'
                direction = 'SELL'
                sub_type = 'ABSORB'
        elif body > 0:
            wick_body = max(upper_wick, lower_wick) / body
            if wick_body >= 1.5:
                if lower_wick > upper_wick and lower_wick > body:
                    div_type = 'PRICE_VOLUME'
                    direction = 'BUY'
                    sub_type = 'REJECT'
                elif upper_wick > lower_wick and upper_wick > body:
                    div_type = 'PRICE_VOLUME'
                    direction = 'SELL'
                    sub_type = 'REJECT'

        if div_type is None:
            continue

        details = {
            'vol_ratio': round(vol_ratio, 2),
            'range_atr': round(range_atr, 3),
            'sub_type': sub_type,
            'close': round(c, 4),
            'bar_range': round(bar_range, 4),
            'limit_tag': limit_tag,
        }

        # PV kalite: hacim + daralma + OBV + recency
        q = 0
        # Hacim spike (0-12)
        if vol_ratio >= 3.0:
            q += 12
        elif vol_ratio >= 2.5:
            q += 10
        elif vol_ratio >= 2.0:
            q += 8
        elif vol_ratio >= 1.5:
            q += 6
        else:
            q += 3
        # Daralma/rejection (0-8)
        if sub_type == 'ABSORB':
            if range_atr <= 0.2:
                q += 8
            elif range_atr <= 0.3:
                q += 6
            elif range_atr <= 0.4:
                q += 4
            else:
                q += 2
        else:
            q += 4
        # OBV teyidi (0-8)
        if i >= 5 and i < len(obv_ema):
            ov = obv_ema.values
            if not np.isnan(ov[i]) and not np.isnan(ov[i - 5]):
                od = ov[i] - ov[i - 5]
                if direction == 'BUY' and od > 0:
                    q += 8
                elif direction == 'SELL' and od < 0:
                    q += 8
                else:
                    q += 2
        # Recency (0-5)
        bf = n - 1 - i
        if bf <= 1:
            q += 5
        elif bf <= 3:
            q += 4
        elif bf <= 5:
            q += 3
        elif bf <= 8:
            q += 2
        # Sub-type bonus
        if sub_type == 'ABSORB':
            q += 4
        else:
            q += 2
        # Location quality
        loc_q = _location_quality(df, i, direction, atr, sr_levels, ema21, bb_upper, bb_lower) if sr_levels is not None else 0
        q += loc_q
        quality = max(min(q, 100), 0)

        signals.append(DivergenceSetup(
            bar_idx=i, direction=direction, div_type=div_type,
            quality=quality, bucket=BUCKET_MAP['PRICE_VOLUME'],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=0.0, structural=False,
            confirmation_mod=0, details=details,
            location_q=loc_q,
        ))

    return signals


# =============================================================================
# UCLU UYUMSUZLUK (Triple Confluence)
# =============================================================================

def detect_triple_confluence(rsi_signals, macd_signals, mfi_signals, obv_signals,
                             adx_signals, df, atr, obv_ema):
    """
    Yeni triple tanim: 3 farkli eksen.
    1. Momentum: RSI veya MACD klasik divergence (±2 bar)
    2. Participation: MFI divergence (±2 bar) VEYA OBV confirmation_mod >= 5
    3. Structural: Momentum sinyali structural swing uzerinde VEYA ADX_EXHAUST ±3 bar

    Returns: list of DivergenceSetup.
    """
    signals = []
    n = len(df)

    # Momentum sinyalleri (RSI veya MACD klasik)
    momentum_sigs = [s for s in rsi_signals if s.div_type == 'RSI_CLASSIC'] + \
                    [s for s in macd_signals if s.div_type == 'MACD_CLASSIC']

    # Participation: MFI klasik
    mfi_classic = [s for s in mfi_signals if s.div_type == 'MFI_CLASSIC']

    # OBV sinyalleri (confirmation_mod icin)
    obv_by_bar = {}
    for s in obv_signals:
        obv_by_bar[s.bar_idx] = s

    seen = set()

    for mom_sig in momentum_sigs:
        direction = mom_sig.direction
        signal_bar = mom_sig.bar_idx

        # Eksen 2: Participation — MFI veya OBV
        has_participation = False
        has_mfi = False
        for mfi_sig in mfi_classic:
            if mfi_sig.direction == direction and abs(mfi_sig.bar_idx - signal_bar) <= 2:
                has_participation = True
                has_mfi = True
                break
        if not has_participation:
            # OBV confirmation mod kontrolu
            if mom_sig.confirmation_mod >= 5:
                has_participation = True

        # Eksen 3: Structural — structural swing veya ADX exhaust
        has_structural = False
        if hasattr(mom_sig, 'structural') and mom_sig.structural:
            has_structural = True
        if not has_structural:
            for adx_sig in adx_signals:
                if adx_sig.direction == direction and abs(adx_sig.bar_idx - signal_bar) <= 3:
                    has_structural = True
                    break

        if not has_participation or not has_structural:
            continue

        key = (signal_bar, direction)
        if key in seen:
            continue
        seen.add(key)

        vol_spike, vol_ratio, obv_trend, vol_score = _check_volume_confirmation(
            df, signal_bar, obv_ema
        )

        # Triple kalite: momentum bilesenlerinin ortalamasini al + confluence bonus
        q = 0
        q += min(int(mom_sig.quality * 0.35), 25)

        # Hacim (0-20)
        if vol_ratio >= 2.0:
            q += 20
        elif vol_ratio >= 1.5:
            q += 15
        elif vol_ratio >= 1.2:
            q += 10
        elif vol_spike:
            q += 5

        # Recency (0-10)
        bars_from_end = n - 1 - signal_bar
        if bars_from_end <= 2:
            q += 10
        elif bars_from_end <= 5:
            q += 7
        elif bars_from_end <= 10:
            q += 3

        # Confluence bonus (20) + MFI (+5) + structural swing (+5)
        q += 20
        if has_mfi:
            q += 5
        if hasattr(mom_sig, 'structural') and mom_sig.structural:
            q += 5
        quality = max(min(q, 100), 0)

        details = {
            'momentum_type': mom_sig.div_type,
            'momentum_div': mom_sig.details,
            'vol_spike': vol_spike,
            'vol_ratio': round(vol_ratio, 2),
            'obv_trend': obv_trend,
            'has_mfi': has_mfi,
            'has_structural': has_structural,
        }

        signals.append(DivergenceSetup(
            bar_idx=signal_bar, direction=direction, div_type='TRIPLE',
            quality=quality, bucket=BUCKET_MAP['TRIPLE'],
            state='SETUP', age=0, trigger_type='NONE', trigger_bar=-1,
            invalidation_level=mom_sig.invalidation_level,
            structural=getattr(mom_sig, 'structural', False),
            confirmation_mod=getattr(mom_sig, 'confirmation_mod', 0),
            details=details,
        ))

    return signals


# =============================================================================
# STATE MACHINE
# =============================================================================

def _evaluate_state(df, setup, trigger_swings_low, trigger_swings_high, ema21, n):
    """
    Setup durumunu degerlendir. Oncelik: invalidation > staleness > trigger.

    Modifies setup in place (state, trigger_type, trigger_bar).
    """
    cfg = DIV_CFG
    bar_idx = setup.bar_idx
    direction = setup.direction
    age = setup.age

    # 1. Staleness kontrolu
    if age > cfg['setup_active_bars'] and setup.state == 'SETUP':
        setup.state = 'STALE'
        return

    # 2. Invalidation kontrolu
    close_vals = df['close'].values
    inv_level = setup.invalidation_level

    if inv_level > 0 and bar_idx < n:
        # bar_idx'ten sonraki barlar icin kontrol
        check_start = min(bar_idx + 1, n)
        for j in range(check_start, n):
            if direction == 'BUY' and close_vals[j] < inv_level:
                setup.state = 'INVALIDATED'
                return
            elif direction == 'SELL' and close_vals[j] > inv_level:
                setup.state = 'INVALIDATED'
                return

    # 3. Trigger kontrolu (sadece SETUP state'de)
    if setup.state != 'SETUP':
        return

    # Trigger arama: bar_idx'ten sonraki barlar
    for j in range(min(bar_idx + 1, n), n):
        # SWING_BREAK: sonraki swing'i kirma
        if direction == 'BUY':
            # En yakin swing high'i bul (bar_idx'ten onceki)
            nearest_high = None
            for sw in trigger_swings_high:
                sw_bar = sw[0] if len(sw) == 3 else sw[0]
                if sw_bar <= bar_idx:
                    nearest_high = sw[2] if len(sw) >= 3 else sw[2]
            if nearest_high is not None and close_vals[j] > nearest_high:
                setup.state = 'TRIGGERED'
                setup.trigger_type = 'SWING_BREAK'
                setup.trigger_bar = j
                return
        else:  # SELL
            nearest_low = None
            for sw in trigger_swings_low:
                sw_bar = sw[0] if len(sw) == 3 else sw[0]
                if sw_bar <= bar_idx:
                    nearest_low = sw[2] if len(sw) >= 3 else sw[2]
            if nearest_low is not None and close_vals[j] < nearest_low:
                setup.state = 'TRIGGERED'
                setup.trigger_type = 'SWING_BREAK'
                setup.trigger_bar = j
                return

        # EMA_RECLAIM
        if j < len(ema21) and not np.isnan(ema21.values[j]):
            ema_val = ema21.values[j]
            if direction == 'BUY' and close_vals[j] > ema_val:
                # Onceki barda ema altindaydi mi?
                if j > 0 and close_vals[j - 1] <= ema_val:
                    setup.state = 'TRIGGERED'
                    setup.trigger_type = 'EMA_RECLAIM'
                    setup.trigger_bar = j
                    return
            elif direction == 'SELL' and close_vals[j] < ema_val:
                if j > 0 and close_vals[j - 1] >= ema_val:
                    setup.state = 'TRIGGERED'
                    setup.trigger_type = 'EMA_RECLAIM'
                    setup.trigger_bar = j
                    return

        # VOLUME_REVERSAL: guclu hacimli donus mumu
        if j < n:
            h = df['high'].values[j]
            l = df['low'].values[j]
            o = df['open'].values[j]
            c = close_vals[j]
            v = df['volume'].values[j]
            # vol_sma precomputed at scan_divergences level, fallback to simple calc
            vol_mean = df['volume'].iloc[max(0, j - cfg['vol_sma_len']):j + 1].mean()
            vs = vol_mean if vol_mean > 0 else 1.0
            body = abs(c - o)
            if vs > 0 and body > 0 and (h - l) > 0:
                wick = max(h - max(o, c), min(o, c) - l)
                if wick > body and v > cfg['trigger_vol_mult'] * vs:
                    if direction == 'BUY' and c > o:
                        setup.state = 'TRIGGERED'
                        setup.trigger_type = 'VOLUME_REVERSAL'
                        setup.trigger_bar = j
                        return
                    elif direction == 'SELL' and c < o:
                        setup.state = 'TRIGGERED'
                        setup.trigger_type = 'VOLUME_REVERSAL'
                        setup.trigger_bar = j
                        return


# =============================================================================
# ANA TARAMA FONKSIYONU
# =============================================================================

def scan_divergences(df, scan_bars=None, weekly_df=None):
    """
    Tum divergence tiplerini tara.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        scan_bars: Son kac bar taranacak (default: DIV_CFG['scan_bars'])
        weekly_df: Opsiyonel haftalik DataFrame (rejim hesabi icin)

    Returns: dict with keys:
        - primary, confirmation, exhaustion (yeni)
        - rsi, macd, obv, mfi, adx, triple, pv (legacy)
    """
    if scan_bars is None:
        scan_bars = DIV_CFG['scan_bars']

    cfg = DIV_CFG
    n = len(df)

    # 1. Indicatorleri hesapla (mevcut + EMA21)
    atr = _calc_atr(df, cfg['atr_len'])
    rsi = _calc_rsi(df['close'], cfg['rsi_len'])
    macd_line, signal_line, macd_hist = _calc_macd(
        df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal']
    )
    obv = _calc_obv(df)
    obv_ema_series = ema(obv, cfg['obv_ema_len'])
    mfi = _calc_mfi(df, cfg['mfi_len'])
    adx = _calc_adx(df, cfg['adx_len'])
    ema21 = ema(df['close'], cfg['trigger_ema_len'])

    # 1b. Faz 2: Bollinger Bands + S/R seviyeleri
    bb_sma = sma(df['close'], cfg['loc_bb_len'])
    bb_std = df['close'].rolling(cfg['loc_bb_len']).std()
    bb_upper = bb_sma + cfg['loc_bb_mult'] * bb_std
    bb_lower = bb_sma - cfg['loc_bb_mult'] * bb_std
    sr_levels = _find_sr_levels(df, atr, cfg['loc_sr_lookback'], cfg['loc_sr_min_touches'])

    # 1c. Faz 2: Rejim hesabi
    regime_val = -1
    try:
        from markets.bist.regime_transition import scan_regime_transition
        regime_data = scan_regime_transition(df, weekly_df=weekly_df)
        regime_series = regime_data.get('regime')
        if regime_series is not None and len(regime_series) > 0:
            regime_val = int(regime_series.iloc[-1])
    except Exception:
        pass

    # 2. Structural swing'leri bul
    struct_lows, struct_highs = _find_structural_swings(
        df['close'], atr, cfg['swing_order'],
        cfg['structural_min_atr_dist'], cfg['structural_min_bar_gap']
    )

    # 3. Trigger swing'leri bul (daha yuksek order)
    trigger_lows, trigger_highs = _find_swings(df['close'], cfg['trigger_swing_order'])

    # 4. 7 detektoru calistir → DivergenceSetup listeleri (Faz 2: location params)
    loc_params = dict(sr_levels=sr_levels, ema21=ema21, bb_upper=bb_upper, bb_lower=bb_lower)
    rsi_signals = detect_rsi_divergence(df, rsi, struct_lows, struct_highs, atr, obv_ema_series, **loc_params)
    macd_signals = detect_macd_divergence(df, macd_hist, macd_line, struct_lows, struct_highs, atr, obv_ema_series, **loc_params)
    obv_signals = detect_obv_divergence(df, obv, struct_lows, struct_highs, atr, obv_ema_series, **loc_params)
    mfi_signals = detect_mfi_divergence(df, mfi, struct_lows, struct_highs, atr, obv_ema_series, **loc_params)
    adx_signals = detect_adx_exhaustion(df, adx, atr, obv_ema_series, **loc_params)
    pv_signals = detect_price_volume_divergence(df, atr, obv_ema_series, **loc_params)

    # 5. Primary vs Confirmation vs Exhaustion ayir
    primary_signals = rsi_signals + macd_signals + mfi_signals
    confirmation_signals = obv_signals + pv_signals
    exhaustion_signals = adx_signals

    # 6. Her primary sinyal icin: confirmation_mod hesapla, kaliteyi guncelle
    for sig in primary_signals:
        conf_mod = _compute_confirmation_modifier(df, sig.bar_idx, sig.direction,
                                                   obv, obv_ema_series, atr)
        sig.confirmation_mod = conf_mod
        sig.quality = max(min(sig.quality + conf_mod, 100), 0)

    # 7. Triple confluence olustur (yeni tanim)
    triple_signals = detect_triple_confluence(
        rsi_signals, macd_signals, mfi_signals, obv_signals,
        adx_signals, df, atr, obv_ema_series
    )

    # 8. ADX + PV promotion: ADX ±3 bar icinde PV rejection → REVERSAL'a yukselt
    for adx_sig in adx_signals:
        for pv_sig in pv_signals:
            if pv_sig.direction == adx_sig.direction and \
               abs(pv_sig.bar_idx - adx_sig.bar_idx) <= 3 and \
               pv_sig.details.get('sub_type') == 'REJECT':
                adx_sig.bucket = 'REVERSAL'
                adx_sig.quality = min(adx_sig.quality + 10, 100)
                break

    # 9. State machine: her non-CONFIRMATION sinyal icin
    all_actionable = primary_signals + exhaustion_signals + triple_signals
    for sig in all_actionable:
        sig.age = max((n - 1) - sig.bar_idx, 0)
        _evaluate_state(df, sig, trigger_lows, trigger_highs, ema21, n)

    # 10. Faz 2: Regime modifier + Signal label + Risk-Reward
    all_signals = primary_signals + confirmation_signals + exhaustion_signals + triple_signals
    for sig in all_signals:
        sig.regime = regime_val
        # Regime modifier
        r_mod = _compute_regime_modifier(sig.bucket, regime_val, cfg)
        sig.regime_mod = r_mod
        sig.quality = max(min(sig.quality + r_mod, 100), 0)
        # Signal label
        sig.signal_label = _compute_signal_label(sig, regime_val)
        # Risk-Reward (sadece actionable sinyaller)
        if sig.state not in ('INVALIDATED',):
            rs, rr = _compute_risk_reward(sig, df, atr, sr_levels)
            sig.risk_score = rs
            sig.rr_ratio = rr

    # 11. scan_bars filtresi + dedup
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

    # 12. Legacy dict key'leri olustur (geriye uyumluluk)
    return {
        # Yeni key'ler
        'primary': _filter_and_dedup(primary_signals),
        'confirmation': _filter_and_dedup(confirmation_signals),
        'exhaustion': _filter_and_dedup(exhaustion_signals),
        # Legacy key'ler
        'rsi': _filter_and_dedup(rsi_signals),
        'macd': _filter_and_dedup(macd_signals),
        'obv': _filter_and_dedup(obv_signals),
        'mfi': _filter_and_dedup(mfi_signals),
        'adx': _filter_and_dedup(adx_signals),
        'triple': _filter_and_dedup(triple_signals),
        'pv': _filter_and_dedup(pv_signals),
    }
