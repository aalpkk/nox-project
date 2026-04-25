"""
nyxexpansion features — v1 core set.

Toplam 26 feature, 8 blok (A-H). İlk eğitim bununla; ablation'da extended 48'e
çıkıyoruz. Yasaklı oscillator'lar `features_extra.py`'de.

Leakage-safe kuralları:
  - Her feature bar t'nin BUGÜNE KADAR gelen bilgisiyle hesaplanır (rolling, shift).
  - XU100 BIST ile aynı seans → same-date reindex güvenli.
  - US-close makro (VIX/DXY/SPY/USDTRY/crypto) BU MODULDE YOK. Eklenirse
    _align_us_macro_to_bist() helper zorunlu (shift(1)).
  - Cross-sectional / universe-level feature'lar (rs_rank_cs, breadth) ayrı
    compute_panel_cs_features() içinde; panel snapshot aynı trade günü için güvenli.

API:
  compute_per_ticker_features(df, xu100_close, params)
      → DataFrame, df.index, A-H blok per-ticker + XU100-ilişkili feature'lar (24).
  compute_panel_cs_features(feats_panel)
      → DataFrame, same panel + cross-sectional 2 feature (rs_rank_cs_today,
        breadth_ad_20d). Panel = long (ticker, date) yapı.
  CORE_FEATURES: list — ilk training'e giren 26 kolon adı.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Feature listesi
# ═════════════════════════════════════════════════════════════════════════════

CORE_FEATURES_V1: list[str] = [
    # A. Pre-Breakout Energy (4)
    'range_contraction_5_20', 'atr_contraction_5_14',
    'bb_width_pctile_60', 'vol_dryup_5_20',
    # B. Breakout Bar Quality (4)
    'bar_range_atr', 'close_loc_bar', 'upper_wick_pct', 'dist_above_trigger_atr',
    # C. Participation (3)
    'rvol_today', 'vol_accel_5d', 'cmf_20_slope',
    # D. Relative Strength (3) — 2 per-ticker (XU100-relatif) + 1 cross-sectional
    'rs_10', 'rs_accel_5d', 'rs_rank_cs_today',
    # E. Regime (3)
    'xu100_trend_score_today', 'vol_regime_pctile', 'breadth_ad_20d',
    # F. Structure (3)
    'swing_bias_up', 'bos_age_bars', 'breakout_significance',
    # G. Exhaustion / Asymmetry (3)
    'trend_extension_ema50_atr', 'entry_to_stop_atr', 'upside_room_52w_atr',
    # H. EMA & Momentum Compression (3)
    'ema_cluster_width_atr', 'ema_compression_5d', 'mom_squeeze_on',
]

# J block — UP modelinde kullanılır; NONUP'ta değil (downtrend top-D bozulduğu için)
CORE_FEATURES_J: list[str] = [
    'new_high_count_20d', 'gap_up_count_20d', 'climax_bar_count_20d',
    'breakout_attempt_count_40d', 'dist_from_sma200_atr',
    'vol_regime_delta_20d', 'base_tightness_last5',
]

# V2 = V1 + J. Default dataset build set.
CORE_FEATURES: list[str] = CORE_FEATURES_V1 + CORE_FEATURES_J

# Chase score soft — sadece UP modelinde kullanılacak composite feature.
CHASE_FEATURE: str = 'chase_score_soft'

# UP modelinde kullanılacak set (v3): V2 + chase_score_soft
CORE_FEATURES_UP: list[str] = CORE_FEATURES + [CHASE_FEATURE]
# NONUP modelinde kullanılacak set: V1 (26 feature, J yok)
CORE_FEATURES_NONUP: list[str] = CORE_FEATURES_V1

# Panel-level (cross-sectional + universe) — ayrı hesap
PANEL_CS_FEATURES: list[str] = ['rs_rank_cs_today', 'breadth_ad_20d', CHASE_FEATURE]

# Chase score bileşenleri: (feature_name, weight, invert)
CHASE_COMPONENTS: list[tuple[str, float, bool]] = [
    ('vol_regime_pctile',   1.00, False),  # yüksek → chase
    ('bb_width_pctile_60',  1.00, False),  # yüksek → chase (geniş bant)
    ('upper_wick_pct',      1.00, False),  # yüksek → chase (supply)
    ('upside_room_52w_atr', 1.00, True),   # INVERT: düşük room → chase
    ('new_high_count_20d',  0.75, False),  # yüksek → overheating
    ('gap_up_count_20d',    0.75, False),  # yüksek → overheating
    ('dist_from_sma200_atr',1.25, False),  # yüksek → extended (boosted)
]


@dataclass(frozen=True)
class FeatureParams:
    # A
    contraction_short: int = 5
    contraction_long: int = 20
    atr_window: int = 14
    atr_short: int = 5
    bb_window: int = 20
    bb_pctile_window: int = 60
    vol_dryup_short: int = 5
    vol_dryup_long: int = 20
    # C
    cmf_window: int = 20
    cmf_slope_window: int = 5
    vol_accel_window: int = 5
    # D / RS
    rs_window: int = 10
    rs_accel_window: int = 5
    # E / Regime
    xu100_ema_fast: int = 21
    xu100_ema_slow: int = 55
    xu100_slope_window: int = 10
    vol_regime_window: int = 60
    breadth_window: int = 20
    # F / Structure
    swing_lookback: int = 5
    bos_ema_window: int = 21
    breakout_sig_window: int = 60
    breakout_sig_tol: float = 0.015   # ±%1.5 aralığı = "test"
    # G / Exhaustion
    trend_ema_window: int = 50
    swing_low_window: int = 10
    ty_52w_window: int = 252
    # H / Compression
    ema_short: int = 9
    ema_mid: int = 21
    ema_long: int = 50
    compression_window: int = 5
    kc_atr_mult: float = 1.5
    # J / Late-stage / Exhaustion
    j_new_high_win: int = 20              # son N gün new-high count
    j_gap_up_win: int = 20                # son N gün yukarı-gap count
    j_gap_up_thresh: float = 0.01         # open/prev_close - 1 ≥ %1
    j_climax_win: int = 20                # son N gün bar_range_atr ≥ eşik count
    j_climax_bar_atr: float = 2.5         # climax bar eşiği
    j_attempt_win: int = 40               # son N gün level-test count
    j_attempt_tol: float = 0.01           # seviyeye ±%1 yakınlık
    j_sma_window: int = 200               # dist_from_sma200_atr
    j_vol_delta_lookback: int = 20        # vol_regime_pctile delta penceresi
    j_base_tight_win: int = 5             # son N gün tightness


# ═════════════════════════════════════════════════════════════════════════════
# Yardımcılar
# ═════════════════════════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _pct_rank_in_window(s: pd.Series, window: int) -> pd.Series:
    """Son `window` bar içindeki bugünün percentile ranki (0-1). Trailing, leakage yok."""
    return s.rolling(window, min_periods=window).apply(
        lambda x: (x <= x[-1]).sum() / len(x), raw=True,
    )


def _align_us_macro_to_bist(us_close: pd.Series, bist_idx: pd.DatetimeIndex) -> pd.Series:
    """US-close kaynağını BIST günlüğüne hizala + shift(1) (4-saat leak fix).

    Detaylar: memory/macro_timing_leakage.md
    Bu v1'de kullanılmıyor; features_extra.py ve future US makro için.
    """
    return us_close.reindex(bist_idx, method='ffill').shift(1)


# ═════════════════════════════════════════════════════════════════════════════
# Blok A — Pre-Breakout Energy
# ═════════════════════════════════════════════════════════════════════════════

def _block_a(df: pd.DataFrame, p: FeatureParams) -> pd.DataFrame:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    vol = df['Volume'].astype(float)

    # range_contraction_5_20: son 5 bar'ın avg range'i / son 20 bar'ın avg range'i
    rng = high - low
    rc_5_20 = (rng.rolling(p.contraction_short).mean() /
               rng.rolling(p.contraction_long).mean().replace(0.0, np.nan))

    # atr_contraction_5_14: ATR(5) / ATR(14)
    atr_14 = _atr(df, p.atr_window)
    atr_5 = _atr(df, p.atr_short)
    atr_contr = atr_5 / atr_14.replace(0.0, np.nan)

    # bb_width_pctile_60: bugünün BB width'inin 60-bar penceresindeki percentile'ı
    ma = close.rolling(p.bb_window).mean()
    sd = close.rolling(p.bb_window).std()
    bb_width = (2 * 2 * sd) / ma.replace(0.0, np.nan)  # %B tarzı normalize
    bb_pct = _pct_rank_in_window(bb_width, p.bb_pctile_window)

    # vol_dryup_5_20: vol_sma5 / vol_sma20 — düşük değerler "hacim kurumuş"
    v5 = vol.rolling(p.vol_dryup_short).mean()
    v20 = vol.rolling(p.vol_dryup_long).mean()
    dryup = v5 / v20.replace(0.0, np.nan)

    return pd.DataFrame({
        'range_contraction_5_20': rc_5_20,
        'atr_contraction_5_14': atr_contr,
        'bb_width_pctile_60': bb_pct,
        'vol_dryup_5_20': dryup,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok B — Breakout Bar Quality
# ═════════════════════════════════════════════════════════════════════════════

def _block_b(df: pd.DataFrame, p: FeatureParams,
             trigger_level: pd.Series, atr_14: pd.Series) -> pd.DataFrame:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)

    bar_range = (high - low)
    bar_range_atr = bar_range / atr_14.replace(0.0, np.nan)

    # close_loc_bar: (close - low) / (high - low)
    close_loc = (close - low) / bar_range.replace(0.0, np.nan)

    # upper_wick_pct: (high - max(open, close)) / bar_range
    upper_body_top = pd.concat([open_, close], axis=1).max(axis=1)
    upper_wick = (high - upper_body_top).clip(lower=0.0)
    upper_wick_pct = upper_wick / bar_range.replace(0.0, np.nan)

    # dist_above_trigger_atr: (close - trigger_level) / ATR14
    dist_above = (close - trigger_level) / atr_14.replace(0.0, np.nan)

    return pd.DataFrame({
        'bar_range_atr': bar_range_atr,
        'close_loc_bar': close_loc,
        'upper_wick_pct': upper_wick_pct,
        'dist_above_trigger_atr': dist_above,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok C — Participation
# ═════════════════════════════════════════════════════════════════════════════

def _block_c(df: pd.DataFrame, p: FeatureParams) -> pd.DataFrame:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    vol = df['Volume'].astype(float)

    # rvol_today: vol / sma(vol, 20)
    vol_sma = vol.rolling(p.vol_dryup_long).mean()
    rvol = vol / vol_sma.replace(0.0, np.nan)

    # vol_accel_5d: (avg(vol, son 5) / avg(vol, 5 bar öncesinden 5 bar)) − 1
    v5_now = vol.rolling(p.vol_accel_window).mean()
    v5_prev = v5_now.shift(p.vol_accel_window)
    vol_accel = (v5_now / v5_prev.replace(0.0, np.nan)) - 1.0

    # CMF (Chaikin Money Flow) 20 + slope (5-bar)
    bar_range = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / bar_range  # -1..+1
    mfv = mfm * vol
    cmf = mfv.rolling(p.cmf_window).sum() / vol.rolling(p.cmf_window).sum().replace(0.0, np.nan)
    cmf_slope = cmf - cmf.shift(p.cmf_slope_window)

    return pd.DataFrame({
        'rvol_today': rvol,
        'vol_accel_5d': vol_accel,
        'cmf_20_slope': cmf_slope,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok D — Relative Strength (per-ticker, XU100 benchmarked)
# ═════════════════════════════════════════════════════════════════════════════

def _block_d(df: pd.DataFrame, p: FeatureParams,
             xu100_close: pd.Series | None) -> pd.DataFrame:
    """rs_10, rs_accel_5d. rs_rank_cs_today cross-sectional, burada YOK."""
    close = df['Close'].astype(float)
    if xu100_close is None or xu100_close.empty:
        return pd.DataFrame({
            'rs_10': np.nan, 'rs_accel_5d': np.nan,
        }, index=df.index)
    # BIST aynı seans → same-date reindex OK
    xu = xu100_close.reindex(df.index, method='ffill')

    def _ret_n(s: pd.Series, n: int) -> pd.Series:
        return s / s.shift(n) - 1.0

    rs_10 = _ret_n(close, p.rs_window) - _ret_n(xu, p.rs_window)
    rs_10_prev = rs_10.shift(p.rs_accel_window)
    rs_accel = rs_10 - rs_10_prev

    return pd.DataFrame({
        'rs_10': rs_10,
        'rs_accel_5d': rs_accel,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok E — Regime (XU100 + per-ticker vol regime)
# ═════════════════════════════════════════════════════════════════════════════

def _xu100_trend_score_series(xu100_close: pd.Series, p: FeatureParams) -> pd.Series:
    """XU100 trend skoru: ema_fast>ema_slow + close>ema_fast + slope>0 → sum / 3.
    Output 0.0–1.0 (bugün için sürekli). BIST aynı seans → reindex safe.
    """
    if xu100_close is None or xu100_close.empty:
        return pd.Series(dtype=float)
    c = xu100_close.astype(float)
    ema_f = c.ewm(span=p.xu100_ema_fast, adjust=False).mean()
    ema_s = c.ewm(span=p.xu100_ema_slow, adjust=False).mean()
    slope = ema_f.pct_change(p.xu100_slope_window, fill_method=None)
    score = ((c > ema_f).astype(float)
             + (ema_f > ema_s).astype(float)
             + (slope > 0).astype(float)) / 3.0
    return score


def _block_e(df: pd.DataFrame, p: FeatureParams,
             xu100_close: pd.Series | None,
             atr_14: pd.Series) -> pd.DataFrame:
    close = df['Close'].astype(float)

    # xu100_trend_score_today: score series → reindex to ticker index
    if xu100_close is None or xu100_close.empty:
        xu_score = pd.Series(np.nan, index=df.index)
    else:
        xu_full = _xu100_trend_score_series(xu100_close, p)
        xu_score = xu_full.reindex(df.index, method='ffill')

    # vol_regime_pctile: ATR% (atr/close) bugünün `window` bar içindeki percentile'ı
    atr_pct = atr_14 / close.replace(0.0, np.nan)
    vol_reg = _pct_rank_in_window(atr_pct, p.vol_regime_window)

    # breadth_ad_20d: panel-level, compute_panel_cs_features() doldurur
    return pd.DataFrame({
        'xu100_trend_score_today': xu_score,
        'vol_regime_pctile': vol_reg,
        'breadth_ad_20d': np.nan,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok F — Structure
# ═════════════════════════════════════════════════════════════════════════════

def _block_f(df: pd.DataFrame, p: FeatureParams,
             trigger_level: pd.Series) -> pd.DataFrame:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)

    # swing_bias_up: son p.swing_lookback bar higher high + higher low mu
    hh = high > high.shift(p.swing_lookback)
    hl = low > low.shift(p.swing_lookback)
    swing_bias_up = (hh & hl).astype(float)

    # bos_age_bars: son kaç bar önce close > ema_21 oldu (0+ int)
    ema_bos = close.ewm(span=p.bos_ema_window, adjust=False).mean()
    above = (close > ema_bos).astype(int)
    # Flip noktaları: 0→1 geçişi ne kadar önce?
    # Her bar için "son True'nun kaç bar önceki olduğu"
    # True → 0, False → küçük bir cezalı sayı değil; basit tutalım:
    # bos_age = ardışık True serisi içinde bugünkü pozisyon (0 = yeni flip).
    # Gerekirse ileride refine ederiz.
    flip = (above.diff().fillna(0) != 0).astype(int)
    # bos_age: son flip'ten bu yana geçen bar sayısı (above=1 iken anlamlı)
    # cumsum trick: her flip yeni segment başlatır
    seg_id = flip.cumsum()
    age = above.groupby(seg_id).cumcount()
    # above=0 iken NaN (aşağıdayız)
    bos_age = pd.Series(np.where(above == 1, age, np.nan), index=df.index)

    # breakout_significance: trigger_level ± %1.5 aralığını son 60 barda
    # kaç kez "test ettik" (high o aralığa değdi)
    win = p.breakout_sig_window
    tol = p.breakout_sig_tol
    sig = pd.Series(np.nan, index=df.index)
    high_arr = high.values
    low_arr = low.values
    trig_arr = trigger_level.values
    for t in range(win, len(df)):
        level = trig_arr[t]
        if not np.isfinite(level) or level <= 0:
            continue
        lo_b = level * (1 - tol)
        hi_b = level * (1 + tol)
        seg_h = high_arr[t - win: t]
        seg_l = low_arr[t - win: t]
        # Touch: bar'ın high/low aralığı level ± tol ile kesişiyor mu
        touched = ((seg_h >= lo_b) & (seg_l <= hi_b)).sum()
        sig.iloc[t] = touched

    return pd.DataFrame({
        'swing_bias_up': swing_bias_up,
        'bos_age_bars': bos_age,
        'breakout_significance': sig,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok G — Exhaustion / Asymmetry
# ═════════════════════════════════════════════════════════════════════════════

def _block_g(df: pd.DataFrame, p: FeatureParams,
             atr_14: pd.Series) -> pd.DataFrame:
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)

    # trend_extension_ema50_atr: (close - ema50) / ATR14
    ema50 = close.ewm(span=p.trend_ema_window, adjust=False).mean()
    trend_ext = (close - ema50) / atr_14.replace(0.0, np.nan)

    # entry_to_stop_atr: (close - swing_low_10) / ATR14
    swing_low = low.rolling(p.swing_low_window, min_periods=p.swing_low_window).min().shift(1)
    entry_to_stop = (close - swing_low) / atr_14.replace(0.0, np.nan)

    # upside_room_52w_atr: (high_52w - close) / ATR14 — yüksekse bol yer var
    high_52w = high.rolling(p.ty_52w_window, min_periods=20).max()
    # Kırılım barında high_52w bugünü de içerebilir → shift(1) güvenli
    high_52w_prior = high_52w.shift(1)
    upside_room = (high_52w_prior - close) / atr_14.replace(0.0, np.nan)

    return pd.DataFrame({
        'trend_extension_ema50_atr': trend_ext,
        'entry_to_stop_atr': entry_to_stop,
        'upside_room_52w_atr': upside_room,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok H — EMA Cluster & Momentum Compression
# ═════════════════════════════════════════════════════════════════════════════

def _block_h(df: pd.DataFrame, p: FeatureParams,
             atr_14: pd.Series) -> pd.DataFrame:
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    ema_s = close.ewm(span=p.ema_short, adjust=False).mean()
    ema_m = close.ewm(span=p.ema_mid, adjust=False).mean()
    ema_l = close.ewm(span=p.ema_long, adjust=False).mean()

    # ema_cluster_width_atr: (max(emas) - min(emas)) / ATR14
    ema_stack = pd.concat([ema_s, ema_m, ema_l], axis=1)
    cluster_width = ema_stack.max(axis=1) - ema_stack.min(axis=1)
    cluster_width_atr = cluster_width / atr_14.replace(0.0, np.nan)

    # ema_compression_5d: (cluster_width - cluster_width[-5]) / max(cluster_width[-5], tiny)
    cw_prev = cluster_width.shift(p.compression_window)
    ema_comp = (cluster_width / cw_prev.replace(0.0, np.nan)) - 1.0  # -1..+1 range, negatif = daralmış

    # mom_squeeze_on: BB inside Keltner Channel (BB_u<KC_u AND BB_l>KC_l)
    ma = close.rolling(p.bb_window).mean()
    sd = close.rolling(p.bb_window).std()
    bb_u = ma + 2 * sd
    bb_l = ma - 2 * sd
    kc_u = ma + p.kc_atr_mult * atr_14
    kc_l = ma - p.kc_atr_mult * atr_14
    squeeze_on = ((bb_u < kc_u) & (bb_l > kc_l)).astype(float)

    return pd.DataFrame({
        'ema_cluster_width_atr': cluster_width_atr,
        'ema_compression_5d': ema_comp,
        'mom_squeeze_on': squeeze_on,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Blok J — Late-stage / Exhaustion (Aşama 2)
# ═════════════════════════════════════════════════════════════════════════════

def _block_j(df: pd.DataFrame, p: FeatureParams,
             trigger_level: pd.Series, atr_14: pd.Series,
             vol_regime_pctile: pd.Series) -> pd.DataFrame:
    """Uptrend late-stage chase pattern'ini ayıran 7 feature.

    Leakage-safe:
      - Count tipi feature'lar pencere [t-W..t-1] → shift(1) sonrası rolling sum.
      - dist_from_sma200: sma200 bugün dahil (anchor referansı; close hesaba dahil,
        forward bilgi yok).
      - vol_regime_delta: vol_regime_pctile serisi zaten t-based expanding rank;
        delta = now - shift(20).
    """
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)

    # 1. new_high_count_20d: son 20g'de close > prior 20g close max flag'leri
    #    prior_close_high_20 = close.rolling(20).max().shift(1)
    pc_hi = close.rolling(p.j_new_high_win, min_periods=p.j_new_high_win).max().shift(1)
    new_hi_flag = (close > pc_hi).astype(float)
    new_hi_count = new_hi_flag.shift(1).rolling(
        p.j_new_high_win, min_periods=p.j_new_high_win,
    ).sum()

    # 2. gap_up_count_20d: open[t]/close[t-1] - 1 ≥ thresh → flag, 20g rolling sum
    gap_flag = ((open_ / close.shift(1).replace(0.0, np.nan)) - 1.0 >= p.j_gap_up_thresh
                ).astype(float)
    gap_count = gap_flag.shift(1).rolling(
        p.j_gap_up_win, min_periods=p.j_gap_up_win,
    ).sum()

    # 3. climax_bar_count_20d: bar_range_atr ≥ eşik bar sayısı, 20g rolling sum
    bar_rng_atr = (high - low) / atr_14.replace(0.0, np.nan)
    climax_flag = (bar_rng_atr >= p.j_climax_bar_atr).astype(float)
    climax_count = climax_flag.shift(1).rolling(
        p.j_climax_win, min_periods=p.j_climax_win,
    ).sum()

    # 4. breakout_attempt_count_40d: son 40g high'ı mevcut trigger_level ± tol
    #    içinde olan bar sayısı. trigger_level[t] = prior_high_20 (zaten t-based).
    #    Vektörize: shift(k) high ≥ 0.99 * tr[t] sayısı, k=1..40
    tr = trigger_level
    attempt_count = pd.Series(0.0, index=df.index)
    high_shifts = {}
    low_lower = 1.0 - p.j_attempt_tol
    high_upper = 1.0 + p.j_attempt_tol
    for k in range(1, p.j_attempt_win + 1):
        hk = high.shift(k)
        lk = low.shift(k)
        # Bar'ın [low,high] aralığı [tr*(1-tol), tr*(1+tol)] ile kesişiyor mu
        touched = ((hk >= tr * low_lower) & (lk <= tr * high_upper)).astype(float)
        attempt_count = attempt_count + touched.fillna(0.0)
    # tr NaN ise NaN bırak
    attempt_count = attempt_count.where(tr.notna())

    # 5. dist_from_sma200_atr
    sma200 = close.rolling(p.j_sma_window, min_periods=p.j_sma_window).mean()
    dist_sma = (close - sma200) / atr_14.replace(0.0, np.nan)

    # 6. vol_regime_delta_20d
    vol_delta = vol_regime_pctile - vol_regime_pctile.shift(p.j_vol_delta_lookback)

    # 7. base_tightness_last5: (max(high[t-5..t-1]) - min(low[t-5..t-1])) / ATR14
    hi5 = high.rolling(p.j_base_tight_win, min_periods=p.j_base_tight_win).max().shift(1)
    lo5 = low.rolling(p.j_base_tight_win, min_periods=p.j_base_tight_win).min().shift(1)
    base_tight = (hi5 - lo5) / atr_14.replace(0.0, np.nan)

    return pd.DataFrame({
        'new_high_count_20d': new_hi_count,
        'gap_up_count_20d': gap_count,
        'climax_bar_count_20d': climax_count,
        'breakout_attempt_count_40d': attempt_count,
        'dist_from_sma200_atr': dist_sma,
        'vol_regime_delta_20d': vol_delta,
        'base_tightness_last5': base_tight,
    }, index=df.index)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════

def compute_per_ticker_features(
    df: pd.DataFrame,
    xu100_close: pd.Series | None = None,
    trigger_level: pd.Series | None = None,
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """Tek ticker için A-H blok (24 per-ticker feature). Panel-CS (rs_rank_cs,
    breadth) burada NaN kalır, compute_panel_cs_features() doldurur.

    Args:
        df: OHLCV DatetimeIndex.
        xu100_close: XU100 Close serisi (DatetimeIndex). Yoksa D/E kısmen NaN.
        trigger_level: O ticker'ın prior_high_20 serisi. Yoksa hesaplanır.
        params: FeatureParams or None.
    """
    if params is None:
        params = FeatureParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    atr_14 = _atr(df, params.atr_window)
    if trigger_level is None:
        trigger_level = df['High'].astype(float).rolling(
            params.contraction_long, min_periods=params.contraction_long,
        ).max().shift(1)

    block_e = _block_e(df, params, xu100_close, atr_14)
    parts = [
        _block_a(df, params),
        _block_b(df, params, trigger_level, atr_14),
        _block_c(df, params),
        _block_d(df, params, xu100_close),
        block_e,
        _block_f(df, params, trigger_level),
        _block_g(df, params, atr_14),
        _block_h(df, params, atr_14),
        _block_j(df, params, trigger_level, atr_14, block_e['vol_regime_pctile']),
    ]
    feat = pd.concat(parts, axis=1)

    # Cross-sectional yer tutucular
    feat['rs_rank_cs_today'] = np.nan
    if 'breadth_ad_20d' not in feat.columns:
        feat['breadth_ad_20d'] = np.nan

    return feat


def compute_panel_cs_features(
    long_panel: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """Cross-sectional + universe feature'lar.

    Args:
        long_panel: kolonlar en az ['ticker', 'date', 'close', 'rs_10'] içeren
                    long panel. Rs_rank, rs_10 üzerinden hesaplanır.
        data_by_ticker: {ticker: OHLCV} — breadth_ad_20d için tüm universe returns'i
                        gerek.
    Returns:
        long_panel ile aynı uzunlukta, iki kolon eklemli:
          rs_rank_cs_today (o tarih için cross-sectional percentile rank, 0-1),
          breadth_ad_20d (universe-level, tarihe göre ffill — % advancers son 20g).
    """
    if params is None:
        params = FeatureParams()
    out = long_panel.copy()

    # rs_rank_cs_today: aynı tarihteki ticker'lar arasında rs_10 percentile
    if 'rs_10' in out.columns:
        out['rs_rank_cs_today'] = out.groupby('date')['rs_10'].rank(
            method='average', pct=True, na_option='keep',
        )
    else:
        out['rs_rank_cs_today'] = np.nan

    # breadth_ad_20d: her trade günü için universe % hisselerin son 20g getirisi > 0
    if data_by_ticker:
        # Daily returns panel
        rets = {}
        for t, d in data_by_ticker.items():
            if d is None or d.empty or 'Close' not in d.columns:
                continue
            c = d['Close'].astype(float)
            r = c.pct_change(params.breadth_window, fill_method=None)
            rets[t] = r
        if rets:
            rets_df = pd.DataFrame(rets)
            # her tarih için >0 oranı
            breadth = (rets_df > 0).sum(axis=1) / rets_df.notna().sum(axis=1).replace(0, np.nan)
            breadth.name = 'breadth_ad_20d'
            # Panel'e join: date alanında
            dates = pd.to_datetime(out['date']) if 'date' in out.columns else None
            if dates is not None:
                out = out.drop(columns=['breadth_ad_20d'], errors='ignore')
                out['breadth_ad_20d'] = dates.map(breadth).values

    # chase_score_soft: per-date cross-sectional rank weighted sum
    out = _add_chase_score_soft(out)

    return out


def _add_chase_score_soft(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank (pct=True) ile 7 bileşenin ağırlıklı toplamı.

    Her bileşen için:
      - Aynı tarihte ticker'lar arası percentile rank (0-1)
      - invert=True ise 1-rank kullan (düşük değer = yüksek chase)
      - component_rank * weight topla; sum(weights of non-NaN) ile normalize

    Leakage yok: same-date içi ranking, horizon ötesi bilgi girmez.
    Çıktı range: ~[0, 1]. NaN kalır eğer tüm bileşenler NaN.
    """
    if panel.empty or 'date' not in panel.columns:
        panel[CHASE_FEATURE] = np.nan
        return panel

    missing = [c for c, _, _ in CHASE_COMPONENTS if c not in panel.columns]
    if missing:
        # Bileşenler yoksa chase_score_soft hesaplanamaz
        panel[CHASE_FEATURE] = np.nan
        return panel

    # Per-date rank
    grouped = panel.groupby('date')
    weighted_sum = pd.Series(0.0, index=panel.index)
    weight_sum = pd.Series(0.0, index=panel.index)

    for feat, w, invert in CHASE_COMPONENTS:
        # pct rank 0-1, NaN preserved
        r = grouped[feat].rank(method='average', pct=True, na_option='keep')
        if invert:
            r = 1.0 - r
        mask = r.notna()
        weighted_sum = weighted_sum + (r.fillna(0.0) * w)
        weight_sum = weight_sum + mask.astype(float) * w

    score = weighted_sum / weight_sum.replace(0.0, np.nan)
    panel[CHASE_FEATURE] = score.values
    return panel


# ═════════════════════════════════════════════════════════════════════════════
# Universe → panel (feature + label merge için)
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_panel(
    signal_panel: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    xu100_close: pd.Series | None = None,
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """Trigger panel (signal_panel) üzerine tüm feature'ları (26) joinle.

    Args:
        signal_panel: `nyxexpansion.trigger.compute_trigger_a_panel` çıktısı.
            Kolonlar: ticker, date, close, prior_high_20, rvol, close_loc, trigger_level.
        data_by_ticker: {ticker: OHLCV}.
        xu100_close: XU100 Close serisi.

    Returns:
        signal_panel + 26 feature kolonu, leakage-safe.
    """
    if params is None:
        params = FeatureParams()
    if signal_panel.empty:
        return signal_panel.copy()

    pieces = []
    for ticker, sub in signal_panel.groupby('ticker', sort=False):
        df = data_by_ticker.get(ticker)
        if df is None or df.empty:
            continue
        # trigger_level tam seri: df'nin tüm barları
        high = df['High'].astype(float)
        trig_full = high.rolling(params.contraction_long, min_periods=params.contraction_long).max().shift(1)

        feats = compute_per_ticker_features(
            df, xu100_close=xu100_close, trigger_level=trig_full, params=params,
        )
        feats = feats.reset_index().rename(columns={feats.index.name or 'Date': 'date'})
        if 'date' not in feats.columns:
            feats = feats.rename(columns={feats.columns[0]: 'date'})
        feats['ticker'] = ticker

        merged = sub.merge(feats, on=['ticker', 'date'], how='left', suffixes=('', '_f'))
        pieces.append(merged)

    if not pieces:
        return signal_panel.copy()

    long_panel = pd.concat(pieces, ignore_index=True)
    long_panel = compute_panel_cs_features(long_panel, data_by_ticker, params)
    long_panel = long_panel.sort_values(['date', 'ticker']).reset_index(drop=True)
    return long_panel
