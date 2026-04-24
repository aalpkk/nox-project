"""
CTE structure detectors — horizontal base & falling channel.

Leakage guard
-------------
Tüm geometry hesapları `[t-W, t-1]` penceresinde yapılır; t barı asla dahil değil.
Pandas `rolling(window)` at index t = [t-W+1, t], bu yüzden her rolling sonucu
`.shift(1)` ile sola kaydırılır → pencere [t-W, t-1] olur ve breakout barı
(t) feature'a sızmaz.

Her iki detektör de per-bar DataFrame döner; `hb_valid` ve `fc_valid`
bağımsız üretilir (overlap korunur, ikisi de trigger'a aday olabilir).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import (
    CompressionParams,
    FallingChannelParams,
    HorizontalBaseParams,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _atr_prior(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder-lite ATR (SMA of True Range), ending at t-1.

    `.shift(1)` ile t barı dışarıda tutulur → leakage-safe normalizer.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr.shift(1)


def _ols_slope(y: np.ndarray) -> float:
    """Closed-form OLS slope over equally-spaced x = 0..n-1.

    Returns NaN if any value is NaN or n < 2.
    """
    if np.isnan(y).any() or len(y) < 2:
        return np.nan
    n = len(y)
    x_mean = (n - 1) / 2.0
    x_var = (n * n - 1) / 12.0  # var(0..n-1) = (n^2 - 1)/12
    y_mean = y.mean()
    cov = ((np.arange(n) - x_mean) * (y - y_mean)).mean()
    return cov / x_var


def _rolling_slope_prior(s: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope on `[t-W, t-1]`. NaN until enough history."""
    return (
        s.rolling(window, min_periods=window)
        .apply(_ols_slope, raw=True)
        .shift(1)
    )


def _rolling_apply_prior(
    s: pd.Series,
    window: int,
    func,
) -> pd.Series:
    """Generic rolling apply over `[t-W, t-1]`."""
    return (
        s.rolling(window, min_periods=window)
        .apply(func, raw=True)
        .shift(1)
    )


def _count_touches(values: np.ndarray, ref: float, tol: float) -> int:
    """Count entries within `tol` of `ref`."""
    if np.isnan(values).any() or np.isnan(ref) or np.isnan(tol):
        return np.nan
    return int((np.abs(values - ref) <= tol).sum())


def _lower_high_count(highs: np.ndarray) -> float:
    """Number of successive 3-bar local peaks that are lower than the prior peak.

    Bir 3-bar local peak: h[i-1] < h[i] and h[i+1] < h[i]. Window içinde
    peaklist = [p0, p1, p2, ...] ise lower_high_count = #{i: p_i < p_{i-1}}.
    """
    if len(highs) < 3 or np.isnan(highs).any():
        return np.nan
    peaks = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            peaks.append(highs[i])
    if len(peaks) < 2:
        return 0.0
    count = 0
    for i in range(1, len(peaks)):
        if peaks[i] < peaks[i - 1]:
            count += 1
    return float(count)


# ═══════════════════════════════════════════════════════════════════════════════
# Horizontal base
# ═══════════════════════════════════════════════════════════════════════════════

def detect_horizontal_base(
    df: pd.DataFrame,
    comp: CompressionParams | None = None,
    hb: HorizontalBaseParams | None = None,
) -> pd.DataFrame:
    """Per-bar horizontal-base geometry + validity flag.

    Returns DataFrame indexed like `df` with columns:
        hb_upper, hb_lower, hb_width, hb_width_atr,
        hb_slope, hb_slope_atr, hb_touches_upper, hb_touches_lower,
        hb_close_density_atr, hb_valid
    """
    if comp is None:
        comp = CompressionParams()
    if hb is None:
        hb = HorizontalBaseParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    W = comp.structure_window
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    atr = _atr_prior(df, comp.atr_window)

    # Envelope on [t-W, t-1]
    hb_upper = high.rolling(W, min_periods=W).max().shift(1)
    hb_lower = low.rolling(W, min_periods=W).min().shift(1)
    hb_width = hb_upper - hb_lower
    hb_width_atr = hb_width / atr.replace(0.0, np.nan)

    # Drift on closes over [t-W, t-1]
    hb_slope = _rolling_slope_prior(close, W)
    hb_slope_atr = hb_slope / atr.replace(0.0, np.nan)

    # Touches and close-density on [t-W, t-1]
    tol = comp.touch_tolerance_atr * atr

    # For each t, compute touches by pairing hb_upper/lower, tol with rolling highs/lows.
    # We use a loop keyed on integer position; W is small so O(N*W) is fine.
    n = len(df)
    touches_upper = np.full(n, np.nan)
    touches_lower = np.full(n, np.nan)
    close_density = np.full(n, np.nan)

    high_vals = high.to_numpy()
    low_vals = low.to_numpy()
    close_vals = close.to_numpy()
    upper_vals = hb_upper.to_numpy()
    lower_vals = hb_lower.to_numpy()
    tol_vals = tol.to_numpy()
    atr_vals = atr.to_numpy()

    min_bars = max(W, comp.min_bars_active, comp.close_density_window)
    for t in range(n):
        if t < min_bars:
            continue
        start = t - W
        if start < 0:
            continue
        win_high = high_vals[start:t]
        win_low = low_vals[start:t]
        win_close = close_vals[start:t]
        u = upper_vals[t]
        lo = lower_vals[t]
        tt = tol_vals[t]
        a = atr_vals[t]

        if np.isnan(u) or np.isnan(lo) or np.isnan(tt) or np.isnan(a):
            continue

        touches_upper[t] = int((np.abs(win_high - u) <= tt).sum())
        touches_lower[t] = int((np.abs(win_low - lo) <= tt).sum())

        if a > 0:
            cdw = comp.close_density_window
            if t - cdw >= 0:
                close_density[t] = float(np.std(close_vals[t - cdw : t])) / a

    out = pd.DataFrame(
        {
            "hb_upper": hb_upper,
            "hb_lower": hb_lower,
            "hb_width": hb_width,
            "hb_width_atr": hb_width_atr,
            "hb_slope": hb_slope,
            "hb_slope_atr": hb_slope_atr,
            "hb_touches_upper": touches_upper,
            "hb_touches_lower": touches_lower,
            "hb_close_density_atr": close_density,
        },
        index=df.index,
    )

    # Validity — yapısal filtre (soft: trigger tarafında opsiyonel hard kullanılır)
    valid = (
        (out["hb_width_atr"] <= hb.max_width_atr)
        & (out["hb_slope_atr"].abs() <= hb.max_abs_slope_atr_per_bar)
        & (out["hb_touches_upper"] >= hb.min_touches_upper)
        & (out["hb_touches_lower"] >= hb.min_touches_lower)
        & (out["hb_close_density_atr"] <= hb.max_close_std_atr)
    ).fillna(False)
    out["hb_valid"] = valid.astype(bool)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Falling channel
# ═══════════════════════════════════════════════════════════════════════════════

def detect_falling_channel(
    df: pd.DataFrame,
    comp: CompressionParams | None = None,
    fc: FallingChannelParams | None = None,
) -> pd.DataFrame:
    """Per-bar falling-channel geometry + validity flag.

    Returns DataFrame indexed like `df` with columns:
        fc_upper_slope, fc_lower_slope, fc_upper_slope_atr, fc_lower_slope_atr,
        fc_upper_last, fc_lower_last, fc_width_atr, fc_width_cv,
        fc_convergence, fc_lower_high_count, fc_valid
    """
    if comp is None:
        comp = CompressionParams()
    if fc is None:
        fc = FallingChannelParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    W = comp.structure_window
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr = _atr_prior(df, comp.atr_window)

    # OLS on highs / lows over [t-W, t-1]
    upper_slope = _rolling_slope_prior(high, W)
    lower_slope = _rolling_slope_prior(low, W)
    upper_slope_atr = upper_slope / atr.replace(0.0, np.nan)
    lower_slope_atr = lower_slope / atr.replace(0.0, np.nan)

    # Need upper_last = regression value AT position t-1 (last point of the fit
    # window). Compute via per-window apply so we have intercept on the fly.
    n = len(df)
    upper_last = np.full(n, np.nan)
    lower_last = np.full(n, np.nan)
    width_mean = np.full(n, np.nan)
    width_cv = np.full(n, np.nan)
    lhc = np.full(n, np.nan)

    high_vals = high.to_numpy()
    low_vals = low.to_numpy()
    atr_vals = atr.to_numpy()

    x = np.arange(W, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for t in range(n):
        start = t - W
        if start < 0:
            continue
        win_high = high_vals[start:t]
        win_low = low_vals[start:t]
        if np.isnan(win_high).any() or np.isnan(win_low).any():
            continue
        if len(win_high) < W:
            continue

        # upper line fit
        hy_mean = win_high.mean()
        cov_u = ((x - x_mean) * (win_high - hy_mean)).sum()
        slope_u = cov_u / x_var
        intercept_u = hy_mean - slope_u * x_mean
        upper_last[t] = slope_u * (W - 1) + intercept_u

        # lower line fit
        ly_mean = win_low.mean()
        cov_l = ((x - x_mean) * (win_low - ly_mean)).sum()
        slope_l = cov_l / x_var
        intercept_l = ly_mean - slope_l * x_mean
        lower_last[t] = slope_l * (W - 1) + intercept_l

        widths = win_high - win_low
        wm = widths.mean()
        width_mean[t] = wm
        if wm > 0:
            width_cv[t] = widths.std() / wm

        lhc[t] = _lower_high_count(win_high)

    width_atr = pd.Series(width_mean, index=df.index) / atr.replace(0.0, np.nan)

    out = pd.DataFrame(
        {
            "fc_upper_slope": upper_slope,
            "fc_lower_slope": lower_slope,
            "fc_upper_slope_atr": upper_slope_atr,
            "fc_lower_slope_atr": lower_slope_atr,
            "fc_upper_last": upper_last,
            "fc_lower_last": lower_last,
            "fc_width_atr": width_atr,
            "fc_width_cv": pd.Series(width_cv, index=df.index),
            "fc_convergence": lower_slope - upper_slope,  # >0 = yakınsıyor
            "fc_lower_high_count": pd.Series(lhc, index=df.index),
        },
        index=df.index,
    )

    # Validity — upper aşağı eğimli + lower-high yapı + width stabil
    valid = (
        (out["fc_upper_slope_atr"] <= fc.max_upper_slope_atr_per_bar)
        & (out["fc_width_atr"] <= fc.max_width_atr)
        & (out["fc_width_cv"] <= fc.max_width_cv)
        & (out["fc_convergence"] >= fc.min_convergence_ratio)
        & (out["fc_lower_high_count"] >= fc.min_lower_high_count)
    ).fillna(False)
    out["fc_valid"] = valid.astype(bool)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Combined per-bar structure frame
# ═══════════════════════════════════════════════════════════════════════════════

def compute_structure(
    df: pd.DataFrame,
    comp: CompressionParams | None = None,
    hb: HorizontalBaseParams | None = None,
    fc: FallingChannelParams | None = None,
) -> pd.DataFrame:
    """Single-ticker structure frame — HB + FC columns concatenated.

    hb_valid ve fc_valid bağımsızdır; overlap olabilir (trigger/label tarafında
    primary setup skorla seçilir, kör kural yok).
    """
    hb_df = detect_horizontal_base(df, comp=comp, hb=hb)
    fc_df = detect_falling_channel(df, comp=comp, fc=fc)
    if hb_df.empty and fc_df.empty:
        return pd.DataFrame(index=df.index)
    return pd.concat([hb_df, fc_df], axis=1)
