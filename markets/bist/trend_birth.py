"""
NOX Trend Birth Screener — Trend Dogum Tespit Modulu
=====================================================
Trendin dogum anini yakalamak icin katmanli erken sinyal sistemi.
MACD crossover yerine histogram ivmesi, yapisal degisim (CHoCH),
hacim dinamikleri ve Bollinger squeeze kullanarak 3-5 bar daha erken
sinyal uretmeyi hedefler.

4 Katman:
  1. Hazirlik (prep_score) — Hacim kurulugu, ADX dususu, BB squeeze
  2. Tetik (triggers)     — Histogram slope, OBV donusu, CHoCH, hacim spike
  3. Teyit (confirmation) — ADX yukselis, RSI 50+
  4. Cikis (exit levels)  — Stop, trailing stop, cikis sinyali

Lowercase kolon konvansiyonu: close, high, low, open, volume.
Runner script (run_trend_birth.py) uppercase→lowercase donusumunu yapar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# SABITLER
# =============================================================================

TB_CFG = {
    # Katman 1 — Hazirlik
    'vol_dryup_ratio': 0.6,
    'vol_dryup_window': 10,
    'vol_dryup_base': 50,
    'adx_len': 14,
    'adx_slope_len': 5,
    'adx_min': 15,
    'adx_trend': 20,
    'bb_len': 20,
    'bb_mult': 2.0,
    'bb_pctile_window': 120,
    'bb_squeeze_thresh': 20,
    # Katman 2 — Tetik
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'hist_slope_len': 3,
    'obv_ema_len': 10,
    'obv_slope_len': 3,
    'pivot_lb': 5,
    'vol_spike_mult': 1.5,
    'vol_spike_base': 20,
    # Katman 3 — Teyit
    'rsi_len': 14,
    'rsi_confirm': 50,
    # Katman 4 — Cikis
    'atr_len': 14,
    'stop_atr_mult': 0.5,
    'trailing_atr_mult': 2.5,
    # Gate
    'prep_gate': 40,
    'min_triggers': 2,
}


# =============================================================================
# VERI YAPISI
# =============================================================================

@dataclass
class TrendBirthSignal:
    ticker: str
    date: Any
    direction: str          # 'AL' veya 'SAT'
    close: float
    # Katman 1
    prep_score: int
    squeeze_active: bool
    volume_dryup: bool
    adx_declining: bool
    # Katman 2
    trigger_count: int
    triggers: list = field(default_factory=list)
    # Katman 3
    confirmed: bool = False
    adx_rising: bool = False
    rsi_above_50: bool = False
    # Katman 4
    stop: float = 0.0
    trailing_stop_atr: float = 0.0
    # Meta
    adx: float = 0.0
    rsi: float = 0.0
    atr: float = 0.0
    macd_hist: float = 0.0
    details: dict = field(default_factory=dict)


# =============================================================================
# PINE-UYUMLU GOSTERGE FONKSIYONLARI (lowercase)
# =============================================================================

def _pine_rma(series, period):
    """Pine ta.rma 1:1 replika — SMA init, sonra Wilder's EMA."""
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

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


def _pine_rsi(close, length):
    """Pine ta.rsi replika."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _pine_rma(gain, length)
    avg_loss = _pine_rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(df):
    """True Range — lowercase columns."""
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _pine_atr(df, period=14):
    """Pine ta.atr replika."""
    return _pine_rma(_true_range(df), period)


def _calc_adx(df, length=14):
    """
    ADX + DI hesaplama — Pine ta.dmi replika.
    Returns: (adx, plus_di, minus_di) Series
    """
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
    )
    tr = _true_range(df)
    atr_val = _pine_rma(tr, length)
    plus_di = 100 * _pine_rma(plus_dm, length) / atr_val.replace(0, np.nan)
    minus_di = 100 * _pine_rma(minus_dm, length) / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _pine_rma(dx, length)
    return adx, plus_di, minus_di


def _calc_macd(close, fast=12, slow=26, signal=9):
    """
    MACD hesaplama.
    Returns: (macd_line, signal_line, histogram) Series
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_obv(close, volume):
    """On Balance Volume."""
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()


def _calc_bb(close, length=20, mult=2.0):
    """
    Bollinger Bands.
    Returns: (upper, middle, lower, width) Series
    """
    middle = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = middle + mult * std
    lower = middle - mult * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


def _find_pivot_lows(low, lb):
    """
    Pivot low tespiti. Bar i'de, low[i-lb] son 2*lb+1 bar'in minimumuysa pivot.
    Pivot degeri i-lb'de olustu, i'de onaylandi.
    Returns: Series (NaN = pivot yok, float = pivot low fiyati)
    """
    n = len(low)
    result = pd.Series(np.nan, index=low.index)
    vals = low.values.astype(float)
    for i in range(2 * lb, n):
        mid = i - lb
        window = vals[i - 2 * lb: i + 1]
        if vals[mid] == np.nanmin(window):
            result.iloc[i] = vals[mid]
    return result


def _find_pivot_highs(high, lb):
    """
    Pivot high tespiti.
    Returns: Series (NaN = pivot yok, float = pivot high fiyati)
    """
    n = len(high)
    result = pd.Series(np.nan, index=high.index)
    vals = high.values.astype(float)
    for i in range(2 * lb, n):
        mid = i - lb
        window = vals[i - 2 * lb: i + 1]
        if vals[mid] == np.nanmax(window):
            result.iloc[i] = vals[mid]
    return result


def _linreg_slope(series, length):
    """
    Linear regression slope (son length bar uzerinden).
    Returns: Series
    """
    result = pd.Series(np.nan, index=series.index)
    vals = series.values.astype(float)
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return result
    for i in range(length - 1, len(vals)):
        y = vals[i - length + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        result.iloc[i] = ss_xy / ss_xx
    return result


# =============================================================================
# KATMAN 1 — HAZIRLIK (Preparation)
# =============================================================================

def compute_prep_score(df, cfg=None):
    """
    Hazirlik skoru: hacim kurulugu, ADX dususu, Bollinger squeeze.
    Her bilesen esit agirlikli, toplam 0-100.

    Returns: dict with Series keys:
        prep_score, squeeze_active, volume_dryup, adx_declining
    """
    cfg = cfg or TB_CFG
    n = len(df)

    # — Hacim Kurulugu —
    vol = df['volume'].astype(float)
    vol_short = vol.rolling(cfg['vol_dryup_window']).mean()
    vol_long = vol.rolling(cfg['vol_dryup_base']).mean()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)
    volume_dryup = vol_ratio < cfg['vol_dryup_ratio']

    # — ADX Dususte —
    adx, plus_di, minus_di = _calc_adx(df, cfg['adx_len'])
    adx_slope = _linreg_slope(adx, cfg['adx_slope_len'])
    adx_declining = (adx_slope < 0) & (adx > cfg['adx_min'])

    # — Bollinger Squeeze —
    _, _, _, bb_width = _calc_bb(df['close'], cfg['bb_len'], cfg['bb_mult'])
    bb_pctile = bb_width.rolling(cfg['bb_pctile_window']).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )
    squeeze_active = bb_pctile < cfg['bb_squeeze_thresh']

    # — Skor (her bilesen ~33 puan) —
    prep_score = (
        volume_dryup.astype(int) * 33
        + adx_declining.astype(int) * 33
        + squeeze_active.astype(int) * 34
    )

    return {
        'prep_score': prep_score,
        'squeeze_active': squeeze_active,
        'volume_dryup': volume_dryup,
        'adx_declining': adx_declining,
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
    }


# =============================================================================
# KATMAN 2 — TETIK (Triggers)
# =============================================================================

def compute_triggers(df, prep, cfg=None):
    """
    Tetik sinyalleri: MACD histogram slope, OBV donusu, CHoCH, hacim spike.
    En az min_triggers kosul + prep_score >= prep_gate gerekli.

    Returns: dict with Series keys:
        trigger_count, direction,
        trig_hist_slope, trig_obv_turn, trig_choch, trig_vol_spike
    """
    cfg = cfg or TB_CFG
    n = len(df)
    close = df['close']
    volume = df['volume'].astype(float)

    # — MACD Histogram Slope Donusu —
    _, _, histogram = _calc_macd(
        close, cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'],
    )
    hist_slope = _linreg_slope(histogram, cfg['hist_slope_len'])
    # Onceki bar negatif, simdiki pozitif → bullish
    hist_slope_bull = (hist_slope > 0) & (hist_slope.shift(1) < 0)
    # Onceki bar pozitif, simdiki negatif → bearish
    hist_slope_bear = (hist_slope < 0) & (hist_slope.shift(1) > 0)
    trig_hist_slope_bull = hist_slope_bull
    trig_hist_slope_bear = hist_slope_bear

    # — OBV Donusu —
    obv = _calc_obv(close, volume)
    obv_ema = obv.ewm(span=cfg['obv_ema_len'], adjust=False).mean()
    obv_slope = _linreg_slope(obv_ema, cfg['obv_slope_len'])
    # Son 3 barda slope pozitife dondu
    trig_obv_bull = (obv_slope > 0) & (obv_slope.shift(cfg['obv_slope_len']) <= 0)
    trig_obv_bear = (obv_slope < 0) & (obv_slope.shift(cfg['obv_slope_len']) >= 0)

    # — Ilk Higher Low (CHoCH) —
    pivot_lows = _find_pivot_lows(df['low'], cfg['pivot_lb'])
    pivot_highs = _find_pivot_highs(df['high'], cfg['pivot_lb'])

    # CHoCH: mevcut pivot low > onceki pivot low (bullish)
    trig_choch_bull = pd.Series(False, index=df.index)
    trig_choch_bear = pd.Series(False, index=df.index)
    last_pl = np.nan
    for i in range(n):
        if pd.notna(pivot_lows.iloc[i]):
            if pd.notna(last_pl) and pivot_lows.iloc[i] > last_pl:
                trig_choch_bull.iloc[i] = True
            last_pl = pivot_lows.iloc[i]
    last_ph = np.nan
    for i in range(n):
        if pd.notna(pivot_highs.iloc[i]):
            if pd.notna(last_ph) and pivot_highs.iloc[i] < last_ph:
                trig_choch_bear.iloc[i] = True
            last_ph = pivot_highs.iloc[i]

    # — Hacim Spike —
    vol_avg = volume.rolling(cfg['vol_spike_base']).mean()
    trig_vol_spike = volume > (vol_avg * cfg['vol_spike_mult'])

    # — Yonu belirle ve trigger say —
    bull_count = (
        trig_hist_slope_bull.astype(int)
        + trig_obv_bull.astype(int)
        + trig_choch_bull.astype(int)
        + trig_vol_spike.astype(int)
    )
    bear_count = (
        trig_hist_slope_bear.astype(int)
        + trig_obv_bear.astype(int)
        + trig_choch_bear.astype(int)
        + trig_vol_spike.astype(int)  # vol spike her iki yone sayilir
    )

    # Prep gate
    gated = prep['prep_score'] >= cfg['prep_gate']

    # Yon: hangisi daha guclu? Gate acik mi?
    direction = pd.Series('', index=df.index)
    trigger_count = pd.Series(0, index=df.index)

    is_bull = (bull_count >= cfg['min_triggers']) & gated
    is_bear = (bear_count >= cfg['min_triggers']) & gated

    # Bull oncelikli (esitlikte bull)
    direction = direction.where(~is_bull, 'AL')
    trigger_count = trigger_count.where(~is_bull, bull_count)

    # Bear sadece bull yoksa
    bear_only = is_bear & ~is_bull
    direction = direction.where(~bear_only, 'SAT')
    trigger_count = trigger_count.where(~bear_only, bear_count)

    return {
        'trigger_count': trigger_count,
        'direction': direction,
        'histogram': histogram,
        'hist_slope': hist_slope,
        'trig_hist_slope_bull': trig_hist_slope_bull,
        'trig_hist_slope_bear': trig_hist_slope_bear,
        'trig_obv_bull': trig_obv_bull,
        'trig_obv_bear': trig_obv_bear,
        'trig_choch_bull': trig_choch_bull,
        'trig_choch_bear': trig_choch_bear,
        'trig_vol_spike': trig_vol_spike,
    }


# =============================================================================
# KATMAN 3 — TEYIT (Confirmation)
# =============================================================================

def compute_confirmation(df, prep, cfg=None):
    """
    Teyit sinyalleri: ADX yukseliste + > 20, RSI > 50.

    Returns: dict with Series keys:
        confirmed, adx_rising, rsi_above_50, rsi
    """
    cfg = cfg or TB_CFG

    adx = prep['adx']
    adx_slope = _linreg_slope(adx, cfg['adx_slope_len'])
    adx_rising = (adx_slope > 0) & (adx > cfg['adx_trend'])

    rsi = _pine_rsi(df['close'], cfg['rsi_len'])
    rsi_above_50 = rsi > cfg['rsi_confirm']

    confirmed = adx_rising & rsi_above_50

    return {
        'confirmed': confirmed,
        'adx_rising': adx_rising,
        'rsi_above_50': rsi_above_50,
        'rsi': rsi,
    }


# =============================================================================
# KATMAN 4 — CIKIS SEVIYELERI (Exit Levels)
# =============================================================================

def compute_exit_levels(df, cfg=None):
    """
    Stop ve trailing stop seviyeleri.

    Returns: dict with Series keys:
        stop, trailing_stop_atr, atr
    """
    cfg = cfg or TB_CFG

    atr = _pine_atr(df, cfg['atr_len'])

    # Son swing low (pivot low) bul — her bar icin en son pivot low
    pivot_lows = _find_pivot_lows(df['low'], cfg['pivot_lb'])
    last_swing_low = pivot_lows.ffill()

    # Stop = son swing low - 0.5 * ATR
    stop = last_swing_low - cfg['stop_atr_mult'] * atr

    # Trailing stop ATR degeri
    trailing_stop_atr = cfg['trailing_atr_mult'] * atr

    return {
        'stop': stop,
        'trailing_stop_atr': trailing_stop_atr,
        'atr': atr,
    }


# =============================================================================
# ANA FONKSIYON — scan_trend_birth
# =============================================================================

def scan_trend_birth(df, cfg=None):
    """
    Tum katmanlari birlestirerek trend birth taramasi yapar.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        cfg: Konfigrasyon dict (opsiyonel, default TB_CFG)

    Returns: dict — tum katman sonuclari birlesmis
    """
    cfg = cfg or TB_CFG

    # Katman 1
    prep = compute_prep_score(df, cfg)

    # Katman 2
    triggers = compute_triggers(df, prep, cfg)

    # Katman 3
    confirm = compute_confirmation(df, prep, cfg)

    # Katman 4
    exits = compute_exit_levels(df, cfg)

    return {
        # Katman 1
        'prep_score': prep['prep_score'],
        'squeeze_active': prep['squeeze_active'],
        'volume_dryup': prep['volume_dryup'],
        'adx_declining': prep['adx_declining'],
        'adx': prep['adx'],
        # Katman 2
        'trigger_count': triggers['trigger_count'],
        'direction': triggers['direction'],
        'histogram': triggers['histogram'],
        'hist_slope': triggers['hist_slope'],
        'trig_hist_slope_bull': triggers['trig_hist_slope_bull'],
        'trig_hist_slope_bear': triggers['trig_hist_slope_bear'],
        'trig_obv_bull': triggers['trig_obv_bull'],
        'trig_obv_bear': triggers['trig_obv_bear'],
        'trig_choch_bull': triggers['trig_choch_bull'],
        'trig_choch_bear': triggers['trig_choch_bear'],
        'trig_vol_spike': triggers['trig_vol_spike'],
        # Katman 3
        'confirmed': confirm['confirmed'],
        'adx_rising': confirm['adx_rising'],
        'rsi_above_50': confirm['rsi_above_50'],
        'rsi': confirm['rsi'],
        # Katman 4
        'stop': exits['stop'],
        'trailing_stop_atr': exits['trailing_stop_atr'],
        'atr': exits['atr'],
    }
