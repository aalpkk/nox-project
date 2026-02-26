"""
NOX Regime Transition Screener — Rejim Gecis Tespit Modulu
==========================================================
ADX tabanli regime tespiti gec tetiklenir. Bu modul "Trend + Participation +
Expansion" modeliyle giris sinyalini hizlandirir, cikisi 3 asamali yapar.

3 Bilesen Skoru:
  1. Trend Score (0-3)        — EMA21>EMA55, SuperTrend bullish, Haftalik EMA up
  2. Participation Score (0-3) — CMF>0, RVOL>=1.0, OBV EMA slope>0
  3. Expansion Score (0-3)     — ADX slope>0, ATR expanding, DI+ - DI- > 5

Regime Belirleme:
  trend_score < 2              → CHOPPY (0)
  trend_score >= 2:
    part>=2 AND exp>=2         → FULL_TREND (3)
    part>=1 AND exp>=1         → TREND (2)
    else                       → GRI_BOLGE (1)

3 Asamali Cikis:
  1. Structure Break — Fiyat < EMA21 (2 gun) VEYA SuperTrend flip
  2. Momentum Decay  — ADX slope < 0 (3 bar) VE CMF < 0
  3. Regime Collapse  — trend_score < 2

Lowercase kolon konvansiyonu: close, high, low, open, volume.
Runner script (run_regime_transition.py) uppercase→lowercase donusumunu yapar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# SABITLER
# =============================================================================

RT_CFG = {
    # Trend
    'ema_fast': 21,
    'ema_slow': 55,
    'st_period': 10,
    'st_mult': 3.0,
    'weekly_ema_len': 21,
    # Participation
    'cmf_period': 20,
    'rvol_period': 20,
    'obv_ema_len': 10,
    'obv_slope_len': 5,
    # Expansion
    'adx_len': 14,
    'adx_slope_len': 5,
    'atr_len': 14,
    'atr_sma_len': 20,
    'atr_expand_mult': 1.05,
    'di_spread_thresh': 5,
    # Exit
    'exit_ema_len': 21,
    'exit_close_below_bars': 2,
    'exit_adx_slope_bars': 3,
    # Lookback for transition
    'regime_lookback': 5,
}

REGIME_NAMES = {
    0: 'CHOPPY',
    1: 'GRI_BOLGE',
    2: 'TREND',
    3: 'FULL_TREND',
}


# =============================================================================
# VERI YAPISI
# =============================================================================

@dataclass
class RegimeTransitionSignal:
    ticker: str
    date: object                # verinin son gunu
    regime: int                 # 0-3 (bugunki regime)
    regime_name: str
    trend_score: int
    participation_score: int
    expansion_score: int
    exit_stage: int             # 0-3
    transition: str             # "CHOPPY→TREND", "TUT", vs.
    direction: str              # 'AL', 'SAT', 'TUT'
    close: float
    # Gecis bilgisi
    transition_date: object = None      # gecis gunu
    transition_close: float = 0.0       # gecis gunundeki fiyat
    gain_since_pct: float = 0.0         # gecisten bugune getiri %
    days_since: int = 0                 # gecisten bu yana gun
    prev_regime: int = 0
    # Meta
    atr: float = 0.0
    adx: float = 0.0
    cmf: float = 0.0
    rvol: float = 0.0
    di_spread: float = 0.0
    adx_slope: float = 0.0
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


def _true_range(df):
    """True Range — lowercase columns."""
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _calc_atr(df, period=14):
    """ATR via Pine RMA."""
    return _pine_rma(_true_range(df), period)


def _calc_adx_with_di(df, length=14):
    """
    ADX + DI+/DI- hesaplama — Pine ta.dmi replika.
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


def _calc_supertrend(df, period=10, mult=3.0):
    """SuperTrend — lowercase columns. Returns: Series (1=bullish, -1=bearish)."""
    atr = _calc_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr

    n = len(df)
    st_dir = pd.Series(1, index=df.index)
    final_up = up.copy()
    final_dn = dn.copy()
    close = df['close']

    for i in range(1, n):
        if up.iloc[i] > final_up.iloc[i - 1]:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = (
                final_up.iloc[i - 1]
                if close.iloc[i - 1] > final_up.iloc[i - 1]
                else up.iloc[i]
            )

        if dn.iloc[i] < final_dn.iloc[i - 1]:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = (
                final_dn.iloc[i - 1]
                if close.iloc[i - 1] < final_dn.iloc[i - 1]
                else dn.iloc[i]
            )

        prev_dir = st_dir.iloc[i - 1]
        if prev_dir == -1 and close.iloc[i] > final_dn.iloc[i - 1]:
            st_dir.iloc[i] = 1
        elif prev_dir == 1 and close.iloc[i] < final_up.iloc[i - 1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = prev_dir

    return st_dir


def _calc_cmf(df, period=20):
    """Chaikin Money Flow — lowercase columns."""
    h, l, c, v = df['high'], df['low'], df['close'], df['volume'].astype(float)
    hl_range = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / hl_range
    clv = clv.fillna(0)
    money_flow_vol = clv * v
    cmf = money_flow_vol.rolling(period).sum() / v.rolling(period).sum()
    return cmf.fillna(0)


def _calc_obv(close, volume):
    """On Balance Volume."""
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume.astype(float)).cumsum()


def _linreg_slope(series, length):
    """Linear regression slope (son length bar uzerinden)."""
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
# BILESEN 1 — TREND SCORE (0-3)
# =============================================================================

def compute_trend_score(df, weekly_df=None, cfg=None):
    """
    Trend Score:
      +1 Close > EMA21 (fiyat kisa EMA ustunde — hizli tepki)
      +1 SuperTrend bullish
      +1 Haftalik EMA trend up (weekly_df gerekli)
    Returns: dict with Series keys
    """
    cfg = cfg or RT_CFG
    close = df['close']

    # Close > EMA21 (EMA crossover yerine — crash recovery'de cok daha hizli)
    ema_fast = close.ewm(span=cfg['ema_fast'], adjust=False).mean()
    ema_bull = (close > ema_fast).astype(int)

    # SuperTrend bullish
    st_dir = _calc_supertrend(df, cfg['st_period'], cfg['st_mult'])
    st_bull = (st_dir == 1).astype(int)

    # Haftalik EMA trend up
    wk_trend_up = pd.Series(0, index=df.index)
    if weekly_df is not None and len(weekly_df) >= cfg['weekly_ema_len'] + 2:
        wk_close = weekly_df['close']
        wk_ema = wk_close.ewm(span=cfg['weekly_ema_len'], adjust=False).mean()
        wk_ema_up = (wk_ema > wk_ema.shift(1)).astype(int)
        # Haftalik veriyi gunluge yay
        wk_ema_up_daily = wk_ema_up.reindex(df.index, method='ffill').fillna(0).astype(int)
        wk_trend_up = wk_ema_up_daily

    trend_score = ema_bull + st_bull + wk_trend_up

    return {
        'trend_score': trend_score,
        'ema_bull': ema_bull,
        'st_bull': st_bull,
        'wk_trend_up': wk_trend_up,
        'st_dir': st_dir,
        'ema_fast': ema_fast,
    }


# =============================================================================
# BILESEN 2 — PARTICIPATION SCORE (0-3)
# =============================================================================

def compute_participation_score(df, cfg=None):
    """
    Participation Score:
      +1 CMF(20) > 0 (birikim)
      +1 RVOL >= 1.0 (ortalama ustu hacim)
      +1 OBV EMA(10) slope > 0 (alim baskisi artiyor)
    Returns: dict with Series keys
    """
    cfg = cfg or RT_CFG
    close = df['close']
    volume = df['volume'].astype(float)

    # CMF
    cmf = _calc_cmf(df, cfg['cmf_period'])
    cmf_pos = (cmf > 0).astype(int)

    # RVOL
    vol_avg = volume.rolling(cfg['rvol_period']).mean()
    rvol = volume / vol_avg.replace(0, np.nan)
    rvol_high = (rvol >= 1.0).astype(int)

    # OBV EMA slope
    obv = _calc_obv(close, volume)
    obv_ema = obv.ewm(span=cfg['obv_ema_len'], adjust=False).mean()
    obv_slope = _linreg_slope(obv_ema, cfg['obv_slope_len'])
    obv_slope_pos = (obv_slope > 0).astype(int)

    participation_score = cmf_pos + rvol_high + obv_slope_pos

    return {
        'participation_score': participation_score,
        'cmf': cmf,
        'rvol': rvol.fillna(0),
        'obv_slope_pos': obv_slope_pos,
    }


# =============================================================================
# BILESEN 3 — EXPANSION SCORE (0-3)
# =============================================================================

def compute_expansion_score(df, cfg=None):
    """
    Expansion Score:
      +1 ADX slope > 0 (yonlu enerji artiyor)
      +1 ATR > SMA(ATR,20) * 1.05 (ATR expanding)
      +1 DI+ - DI- > 5 (boga yonlu baski)
    Returns: dict with Series keys
    """
    cfg = cfg or RT_CFG

    # ADX + DI
    adx, plus_di, minus_di = _calc_adx_with_di(df, cfg['adx_len'])
    adx_slope = _linreg_slope(adx, cfg['adx_slope_len'])
    adx_slope_pos = (adx_slope > 0).astype(int)

    # ATR expanding
    atr = _calc_atr(df, cfg['atr_len'])
    atr_sma = atr.rolling(cfg['atr_sma_len']).mean()
    atr_expanding = (atr > atr_sma * cfg['atr_expand_mult']).astype(int)

    # DI spread
    di_spread = plus_di - minus_di
    di_bull = (di_spread > cfg['di_spread_thresh']).astype(int)

    expansion_score = adx_slope_pos + atr_expanding + di_bull

    return {
        'expansion_score': expansion_score,
        'adx': adx,
        'adx_slope': adx_slope,
        'atr': atr,
        'di_spread': di_spread,
        'plus_di': plus_di,
        'minus_di': minus_di,
    }


# =============================================================================
# REGIME BELIRLEME
# =============================================================================

def determine_regime(trend_score, participation_score, expansion_score):
    """
    Regime belirleme (vektorize).

    trend_score < 2                             → CHOPPY (0)
    trend_score >= 2 (confirmed):
      participation >= 2 AND expansion >= 2     → FULL_TREND (3)
      participation >= 1 AND expansion >= 1     → TREND (2)
      else                                      → GRI_BOLGE (1)

    Returns: pd.Series (int 0-3)
    """
    n = len(trend_score)
    regime = pd.Series(0, index=trend_score.index)

    confirmed = trend_score >= 2

    full_trend = confirmed & (participation_score >= 2) & (expansion_score >= 2)
    trend = confirmed & (participation_score >= 1) & (expansion_score >= 1) & ~full_trend
    gri = confirmed & ~full_trend & ~trend

    regime = regime.where(~gri, 1)
    regime = regime.where(~trend, 2)
    regime = regime.where(~full_trend, 3)

    return regime


# =============================================================================
# 3 ASAMALI CIKIS
# =============================================================================

def compute_exit_stages(df, trend_data, expansion_data, participation_data, cfg=None):
    """
    3 asamali cikis:
      Stage 1 — Structure Break: Fiyat < EMA21 (2 gun ust uste) VEYA SuperTrend flip
      Stage 2 — Momentum Decay:  ADX slope < 0 (3 bar) VE CMF < 0
      Stage 3 — Regime Collapse:  trend_score < 2

    Her asama kumulatif: exit_stage = stage1 + stage2 + stage3

    Returns: pd.Series (int 0-3)
    """
    cfg = cfg or RT_CFG
    close = df['close']
    ema21 = trend_data['ema_fast']
    st_dir = trend_data['st_dir']
    adx_slope = expansion_data['adx_slope']
    cmf = participation_data['cmf']
    trend_score = trend_data['trend_score']

    n = len(df)

    # Stage 1: Structure Break
    close_below_ema = close < ema21
    # 2 gun ust uste close < EMA21
    close_below_2 = close_below_ema & close_below_ema.shift(1).fillna(False)
    # SuperTrend flip (bullish → bearish)
    st_flip = (st_dir == -1) & (st_dir.shift(1) == 1)
    # Structure break: herhangisi TRUE ise ve devam ederse (sticky)
    # Non-sticky: her bar bagimsiz degerlendirilir
    stage1 = (close_below_2 | (st_dir == -1)).astype(int)

    # Stage 2: Momentum Decay
    adx_slope_neg = adx_slope < 0
    # 3 bar ust uste ADX slope negatif
    adx_slope_neg_3 = (
        adx_slope_neg
        & adx_slope_neg.shift(1).fillna(False)
        & adx_slope_neg.shift(2).fillna(False)
    )
    cmf_neg = cmf < 0
    stage2 = (adx_slope_neg_3 & cmf_neg).astype(int)

    # Stage 3: Regime Collapse
    stage3 = (trend_score < 2).astype(int)

    exit_stages = stage1 + stage2 + stage3
    return exit_stages


def apply_exit_adjustment(base_regime, exit_stages):
    """
    Cikis asamalarina gore regime'i dusur.
      exit_stages == 0 → regime = base_regime
      exit_stages == 1 → regime = max(base_regime - 1, 0)
      exit_stages >= 2 → regime = 0
    """
    regime = base_regime.copy()

    mask_1 = exit_stages == 1
    mask_2 = exit_stages >= 2

    regime = regime.where(~mask_1, (base_regime - 1).clip(lower=0))
    regime = regime.where(~mask_2, 0)

    return regime


# =============================================================================
# GECIS TESPITI
# =============================================================================

def detect_transitions(regime, lookback=1):
    """
    Iki gunun regime'ini karsilastirarak gecis tespit eder (bar bazli).

    Returns: (direction, transition_label, prev_regime) Series tuple
      direction: 'AL' (yukari gecis), 'SAT' (asagi gecis), 'TUT' (degismedi)
      transition_label: "CHOPPY→TREND", "TUT", vs.
    """
    prev_regime = regime.shift(lookback).fillna(0).astype(int)
    curr_regime = regime.astype(int)

    direction = pd.Series('TUT', index=regime.index)
    transition = pd.Series('TUT', index=regime.index)

    up = curr_regime > prev_regime
    down = curr_regime < prev_regime

    direction = direction.where(~up, 'AL')
    direction = direction.where(~down, 'SAT')

    # Gecis label olustur
    for i in range(len(regime)):
        if up.iloc[i] or down.iloc[i]:
            prev_name = REGIME_NAMES.get(int(prev_regime.iloc[i]), '?')
            curr_name = REGIME_NAMES.get(int(curr_regime.iloc[i]), '?')
            transition.iloc[i] = f"{prev_name}→{curr_name}"

    return direction, transition, prev_regime


def find_last_transition(regime, close, index, scan_bars=60):
    """
    Son scan_bars icerisindeki en son anlamli gecisi bul.

    Mantik:
    - Mevcut regime >= 2 ise (TREND/FULL): geriye dogru git,
      regime'in ilk kez mevcut seviyeye veya ustune ciktigi bari bul.
      Arada kucuk dalgalanmalari tolere et (1 bar dusus + geri donus).
    - Mevcut regime < 2 ise (CHOPPY/GRI): son 10 barda anlamli
      SAT gecisi (2/3 → 0/1) varsa raporla.

    Returns: dict veya None
    """
    n = len(regime)
    if n < 2:
        return None

    last = n - 1
    current_regime = int(regime.iloc[last])
    start = max(1, last - scan_bars + 1)

    # ── Mevcut regime >= 2: ilk giris noktasini bul ──
    if current_regime >= 2:
        # Geriye dogru git: regime < 2 olan ilk "gercek" dususu bul.
        # Gecici dusus toleransi: 3 bar ust uste < 2 olmadikca gercek dusus sayilmaz.
        # (Exit adjustment kaynaklı kisa sureli dalgalanmalari tolere eder)
        CONSEC_THRESH = 3
        entry_bar = start  # fallback: pencere basi
        consec_below = 0
        found_break = False
        for i in range(last, start - 1, -1):
            r = int(regime.iloc[i])
            if r < 2:
                consec_below += 1
                if consec_below >= CONSEC_THRESH:
                    # Gercek dusus: giris noktasi = bu bloktan sonraki ilk >= 2 bar
                    entry_bar = i + consec_below
                    if entry_bar > last:
                        entry_bar = last
                    found_break = True
                    break
            else:
                consec_below = 0

        if not found_break:
            # Pencere boyunca gercek dusus yok → en basi al
            entry_bar = start

        # entry_bar'dan onceki bari "from_regime" olarak al
        # to_regime = bugunki regime (entry bar'daki degil, cunku kademeli yukselis olabilir)
        if entry_bar > 0 and entry_bar <= last:
            from_r = int(regime.iloc[entry_bar - 1])
            to_r = current_regime
            return {
                'direction': 'AL',
                'transition': f"{REGIME_NAMES.get(from_r, '?')}→{REGIME_NAMES.get(to_r, '?')}",
                'bar_idx': entry_bar,
                'date': index[entry_bar],
                'close_at_transition': float(close.iloc[entry_bar]),
                'from_regime': from_r,
                'to_regime': to_r,
            }

    # ── Mevcut regime < 2: son 10 barda SAT gecisi bul ──
    sat_window = min(10, last - start + 1)
    for i in range(last, max(start, last - sat_window) - 1, -1):
        curr = int(regime.iloc[i])
        prev = int(regime.iloc[i - 1])
        if curr < prev and prev >= 2:
            # Anlamli dusus: TREND/FULL → GRI/CHOPPY
            return {
                'direction': 'SAT',
                'transition': f"{REGIME_NAMES.get(prev, '?')}→{REGIME_NAMES.get(curr, '?')}",
                'bar_idx': i,
                'date': index[i],
                'close_at_transition': float(close.iloc[i]),
                'from_regime': prev,
                'to_regime': curr,
            }

    return None


# =============================================================================
# ANA FONKSIYON — scan_regime_transition
# =============================================================================

def scan_regime_transition(df, weekly_df=None, cfg=None):
    """
    Tum bilesenleri birlestirerek regime transition taramasi yapar.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        weekly_df: Haftalik DataFrame (opsiyonel, trend score icin)
        cfg: Konfigrasyon dict (opsiyonel, default RT_CFG)

    Returns: dict — tum sonuclar (Series bazli)
    """
    cfg = cfg or RT_CFG

    # Bilesenler
    trend_data = compute_trend_score(df, weekly_df, cfg)
    part_data = compute_participation_score(df, cfg)
    exp_data = compute_expansion_score(df, cfg)

    # Base regime
    base_regime = determine_regime(
        trend_data['trend_score'],
        part_data['participation_score'],
        exp_data['expansion_score'],
    )

    # Exit stages
    exit_stages = compute_exit_stages(df, trend_data, exp_data, part_data, cfg)

    # Final regime (exit adjustment)
    regime = apply_exit_adjustment(base_regime, exit_stages)

    # Gecis tespiti
    direction, transition, prev_regime = detect_transitions(regime)

    return {
        'close': df['close'],
        'regime': regime,
        'base_regime': base_regime,
        'trend_score': trend_data['trend_score'],
        'participation_score': part_data['participation_score'],
        'expansion_score': exp_data['expansion_score'],
        'exit_stage': exit_stages,
        'direction': direction,
        'transition': transition,
        'prev_regime': prev_regime,
        # Meta
        'atr': exp_data['atr'],
        'adx': exp_data['adx'],
        'adx_slope': exp_data['adx_slope'],
        'cmf': part_data['cmf'],
        'rvol': part_data['rvol'],
        'di_spread': exp_data['di_spread'],
        'ema_bull': trend_data['ema_bull'],
        'st_bull': trend_data['st_bull'],
        'wk_trend_up': trend_data['wk_trend_up'],
    }
