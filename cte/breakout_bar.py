"""
Breakout bar quality — trigger'ın giriş barı.

Stateless: sadece bar t ve prior close. Structure-agnostic, dolayısıyla HB
ve FC ikisi için de aynı bar-quality uygulanır.

Leakage
-------
- bar_return_1d: close[t] / close[t-1] — bar t verisi.
- bar_rvol, bar_close_loc, bar_body_pct_range: bar t geometry ve bar t vol.
  Denominator structure_vol_ref `cte/volume.py` içinden (prior window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import BreakoutBarParams


def compute_breakout_bar(
    df: pd.DataFrame,
    structure_vol_ref: pd.Series | None = None,
    bar: BreakoutBarParams | None = None,
) -> pd.DataFrame:
    """Per-bar bar-quality metrics + pass flag.

    Parameters
    ----------
    df : OHLCV DataFrame.
    structure_vol_ref : prior-window mean volume (from cte.volume.compute_volume).
        Optional; if None, bar_rvol = vol / sma(vol, 20) computed inline.
    bar : BreakoutBarParams.

    Returns DataFrame with columns:
        bar_return_1d, bar_rvol, bar_close_loc, bar_body_pct_range,
        bar_quality_pass
    """
    if bar is None:
        bar = BreakoutBarParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    prev_close = close.shift(1).replace(0.0, np.nan)
    bar_return_1d = close / prev_close - 1.0

    if structure_vol_ref is None:
        structure_vol_ref = vol.rolling(20, min_periods=20).mean().shift(1)
    bar_rvol = vol / structure_vol_ref.replace(0.0, np.nan)

    bar_range = (high - low).replace(0.0, np.nan)
    bar_close_loc = (close - low) / bar_range
    bar_body_pct_range = (close - open_).abs() / bar_range

    pass_flag = (
        (bar_return_1d >= bar.min_return_1d)
        & (bar_rvol >= bar.min_rvol)
        & (bar_close_loc >= bar.min_close_loc_bar)
        & (bar_body_pct_range >= bar.min_body_pct_range)
    ).fillna(False)

    return pd.DataFrame(
        {
            "bar_return_1d": bar_return_1d,
            "bar_rvol": bar_rvol,
            "bar_close_loc": bar_close_loc,
            "bar_body_pct_range": bar_body_pct_range,
            "bar_quality_pass": pass_flag.astype(bool),
        },
        index=df.index,
    )
