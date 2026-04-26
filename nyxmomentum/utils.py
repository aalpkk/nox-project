"""
Small shared helpers: date alignment, cross-sectional transforms, IO.
Keep thin — do not put logic that belongs in labels/features here.
"""
from __future__ import annotations

import os
import json
from typing import Iterable

import numpy as np
import pandas as pd


# ── Date / calendar ───────────────────────────────────────────────────────────

def to_naive_ts(x) -> pd.Timestamp:
    """Coerce any date-like input to tz-naive midnight pd.Timestamp."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def union_trading_calendar(panel: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all per-ticker indices, sorted, tz-naive."""
    if not panel:
        return pd.DatetimeIndex([])
    idxs = [df.index for df in panel.values() if df is not None and len(df) > 0]
    if not idxs:
        return pd.DatetimeIndex([])
    idx = idxs[0]
    for other in idxs[1:]:
        idx = idx.union(other)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.sort_values()


def slice_past(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return df.loc[:cutoff] — inclusive. Assumes sorted DatetimeIndex."""
    if df is None or len(df) == 0:
        return df
    return df.loc[:cutoff]


# ── Cross-sectional transforms ────────────────────────────────────────────────

def cs_rank(series: pd.Series, pct: bool = True) -> pd.Series:
    """Rank within a single cross-section. Input indexed by ticker."""
    return series.rank(pct=pct, method="average")


def cs_zscore(series: pd.Series, winsor: float | None = None) -> pd.Series:
    """Z-score within a single cross-section. Optional symmetric winsorization."""
    s = series.astype(float)
    if winsor is not None and 0 < winsor < 0.5:
        lo, hi = s.quantile(winsor), s.quantile(1 - winsor)
        s = s.clip(lower=lo, upper=hi)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def cs_percentile_by_date(panel_long: pd.DataFrame, value_col: str,
                          date_col: str = "rebalance_date",
                          out_col: str | None = None) -> pd.DataFrame:
    """Add percentile-rank-by-date column to a long-format panel."""
    out = panel_long.copy()
    tag = out_col or f"{value_col}_cs_rank"
    out[tag] = out.groupby(date_col)[value_col].rank(pct=True, method="average")
    return out


# ── IO ────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ── Pandas conveniences ───────────────────────────────────────────────────────

def safe_pct_change(close: pd.Series, periods: int = 1) -> pd.Series:
    """pct_change with inf/−inf → NaN."""
    r = close.pct_change(periods=periods)
    return r.replace([np.inf, -np.inf], np.nan)


def rolling_apply_by_ticker(panel: dict[str, pd.DataFrame], fn, **kwargs) -> dict[str, pd.DataFrame]:
    """Apply fn(df, **kwargs) → df for each ticker in panel."""
    return {t: fn(df, **kwargs) for t, df in panel.items() if df is not None and len(df) > 0}


def iter_panel_on_date(panel: dict[str, pd.DataFrame], date: pd.Timestamp
                       ) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield (ticker, df_up_to_date) for tickers with any data at/before date."""
    for t, df in panel.items():
        if df is None or len(df) == 0:
            continue
        if df.index[0] > date:
            continue
        yield t, slice_past(df, date)
