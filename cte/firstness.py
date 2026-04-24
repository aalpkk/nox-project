"""
First-breakout diskriminatoru — kaç defa aynı boundary zaten kırıldı, ne kadar
süredir yapı içinde.

İki boundary ayrı işlenir:
  - Horizontal base: hb_upper (rolling max of prior W highs)
  - Falling channel: fc_upper_last (regression line value AT t-1)

Her bar t için:
  prior_break_count_{setup} : son `lookback_window` barda
                              (close > boundary_at_that_bar) sayımı, t hariç.
  failed_break_count_{setup}: prior_break'lerin kaçında `failed_break_window` bar
                              içinde close < boundary - failback_atr_frac*ATR düşüş.
  last_break_age_{setup}    : en son break'ten bu yana bar sayısı; hiç break
                              yoksa NaN.
  is_first_break_{setup}    : bool, prior_break_count ≤ max_prior_attempts
                              AND bar-t kendi close > boundary.

Leakage
-------
- Boundary ve ATR hem zaten `[t-W, t-1]` yani t içermiyor; sayımda t hariç.
- "Failed break" lookforward'u prior break'ten sonraki K bar ölçer — ama bu K
  bar hala t'den küçük indekslerde (pencere içeride). Yani prior_break at i,
  failback'i i+1..i+K ölçer, i+K < t olmalı aksi halde NaN. Biz i ≤ t-1-K
  şartı koyarak bunu garantiliyoruz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import CompressionParams, FirstBreakParams
from cte.structure import _atr_prior


def _compute_firstness_for_boundary(
    close: pd.Series,
    boundary: pd.Series,
    atr: pd.Series,
    fb: FirstBreakParams,
) -> pd.DataFrame:
    """Generic firstness accounting for a per-bar boundary series.

    `boundary` ve `atr` zaten leakage-safe (shift(1)) olmalı.
    """
    n = len(close)
    close_v = close.to_numpy()
    bound_v = boundary.to_numpy()
    atr_v = atr.to_numpy()

    # break_flag[i] = close[i] > boundary[i]  (i = past bar)
    # boundary[i] already reflects prior window at i, so comparing close[i] to
    # boundary[i] asks "at bar i, did it close above its then-current boundary?"
    break_flag = np.zeros(n, dtype=bool)
    valid = ~(np.isnan(close_v) | np.isnan(bound_v))
    break_flag[valid] = close_v[valid] > bound_v[valid]

    # failed_break at i: close within [i+1, i+K] drops below boundary[i] - frac*atr[i]
    K = fb.failed_break_window
    frac = fb.failed_break_atr_frac
    failed_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if not break_flag[i]:
            continue
        if np.isnan(atr_v[i]) or np.isnan(bound_v[i]):
            continue
        thresh = bound_v[i] - frac * atr_v[i]
        hi = min(n, i + 1 + K)
        if (close_v[i + 1 : hi] < thresh).any():
            failed_flag[i] = True

    L = fb.lookback_window
    prior_count = np.full(n, np.nan)
    prior_failed = np.full(n, np.nan)
    last_age = np.full(n, np.nan)
    is_first = np.zeros(n, dtype=bool)

    for t in range(n):
        start = t - L
        if start < 0:
            continue
        win = break_flag[start:t]  # [t-L, t-1]
        prior_count[t] = float(win.sum())
        prior_failed[t] = float(failed_flag[start:t].sum())

        # last break age
        if win.any():
            last_idx = np.where(win)[0].max()  # relative to [start, t-1]
            last_age[t] = float(t - (start + last_idx))
        else:
            last_age[t] = float(L + 1)  # sentinel: no break within window

        # is_first_break: today's close > today's boundary AND prior_count ≤ max
        if not np.isnan(bound_v[t]) and not np.isnan(close_v[t]):
            today_break = close_v[t] > bound_v[t]
            is_first[t] = bool(
                today_break and prior_count[t] <= fb.max_prior_attempts
            )

    return pd.DataFrame(
        {
            "prior_break_count": prior_count,
            "failed_break_count": prior_failed,
            "last_break_age": last_age,
            "is_first_break": is_first,
        },
        index=close.index,
    )


def compute_firstness(
    df: pd.DataFrame,
    structure_frame: pd.DataFrame,
    fb: FirstBreakParams | None = None,
    comp: CompressionParams | None = None,
) -> pd.DataFrame:
    """Per-setup firstness columns.

    Parameters
    ----------
    df : OHLCV DataFrame.
    structure_frame : output of cte.structure.compute_structure — must contain
        hb_upper and fc_upper_last columns.
    fb, comp : params.

    Returns DataFrame with columns prefixed by setup:
        hb_prior_break_count, hb_failed_break_count, hb_last_break_age,
        hb_is_first_break, fc_prior_break_count, fc_failed_break_count,
        fc_last_break_age, fc_is_first_break
    """
    if fb is None:
        fb = FirstBreakParams()
    if comp is None:
        comp = CompressionParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    atr = _atr_prior(df, comp.atr_window)

    out_cols: dict[str, pd.Series] = {}

    if "hb_upper" in structure_frame.columns:
        hb = _compute_firstness_for_boundary(
            close, structure_frame["hb_upper"], atr, fb,
        )
        for c in hb.columns:
            out_cols[f"hb_{c}"] = hb[c]

    if "fc_upper_last" in structure_frame.columns:
        fc = _compute_firstness_for_boundary(
            close, structure_frame["fc_upper_last"], atr, fb,
        )
        for c in fc.columns:
            out_cols[f"fc_{c}"] = fc[c]

    return pd.DataFrame(out_cols, index=df.index)
