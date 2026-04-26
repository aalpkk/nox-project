"""
Rebalance calendar construction.

Rule: the trading calendar is externally supplied (e.g., XU100 trading days or
the union of BIST ticker calendars). This module does NOT synthesize trading
days — it only anchors rebalance points onto an already-real calendar. That
way weekends, BIST holidays, and half-days are automatically respected.

Anchor options:
  - "last_trading_day"  — default. The final session in each month/week.
    Feature timing uses close of this date; execution is next session open.
  - "first_trading_day" — alternative anchor if you prefer to execute at the
    open of the new period (then feature cutoff is the prior session).

v1 supports monthly (frequency="M") and weekly (frequency="W").
"""
from __future__ import annotations

import pandas as pd

from .config import RebalanceConfig


def _filter_calendar(cal: pd.DatetimeIndex, start: str | None, end: str | None) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(cal)
    if cal.tz is not None:
        cal = cal.tz_localize(None)
    cal = cal.sort_values().unique()
    if start:
        cal = cal[cal >= pd.Timestamp(start)]
    if end:
        cal = cal[cal <= pd.Timestamp(end)]
    return cal


def _monthly_anchors(cal: pd.DatetimeIndex, anchor: str) -> pd.DatetimeIndex:
    s = pd.Series(cal, index=cal)
    # Group by (year, month)
    grp = s.groupby([cal.year, cal.month])
    if anchor == "last_trading_day":
        out = grp.max()
    elif anchor == "first_trading_day":
        out = grp.min()
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    return pd.DatetimeIndex(sorted(out.values))


def _weekly_anchors(cal: pd.DatetimeIndex, anchor: str) -> pd.DatetimeIndex:
    s = pd.Series(cal, index=cal)
    iso = cal.isocalendar()
    grp = s.groupby([iso.year.values, iso.week.values])
    if anchor == "last_trading_day":
        out = grp.max()
    elif anchor == "first_trading_day":
        out = grp.min()
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    return pd.DatetimeIndex(sorted(out.values))


def build_rebalance_calendar(trading_calendar: pd.DatetimeIndex,
                             config: RebalanceConfig | None = None) -> pd.DatetimeIndex:
    """
    Anchor rebalance dates onto a real trading calendar.

    Parameters
    ----------
    trading_calendar : pd.DatetimeIndex
        Trading dates in ascending order (e.g., XU100 session dates or the
        union of all BIST ticker dates).
    config : RebalanceConfig
        Frequency + anchor + window.

    Returns
    -------
    pd.DatetimeIndex of rebalance dates, all drawn from `trading_calendar`.

    Raises
    ------
    ValueError on unknown frequency/anchor or if fewer than
    `config.min_rebalances` anchors are produced.
    """
    cfg = config or RebalanceConfig()
    cal = _filter_calendar(trading_calendar, cfg.start_date, cfg.end_date)
    if len(cal) == 0:
        raise ValueError("Empty trading calendar after start/end filtering.")

    if cfg.frequency == "M":
        anchors = _monthly_anchors(cal, cfg.anchor)
    elif cfg.frequency == "W":
        anchors = _weekly_anchors(cal, cfg.anchor)
    else:
        raise ValueError(f"Unknown frequency: {cfg.frequency}")

    if len(anchors) < cfg.min_rebalances:
        raise ValueError(
            f"Only {len(anchors)} rebalance anchors produced "
            f"(required ≥ {cfg.min_rebalances}). Check start_date / data range."
        )
    return anchors


def next_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first calendar date strictly greater than `date`, or None."""
    cal = pd.DatetimeIndex(calendar)
    pos = cal.searchsorted(pd.Timestamp(date), side="right")
    if pos >= len(cal):
        return None
    return cal[pos]


def prev_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the last calendar date strictly less than `date`, or None."""
    cal = pd.DatetimeIndex(calendar)
    pos = cal.searchsorted(pd.Timestamp(date), side="left")
    if pos == 0:
        return None
    return cal[pos - 1]


def rebalance_summary(anchors: pd.DatetimeIndex) -> dict:
    """Small diagnostic dict for reporting."""
    if len(anchors) == 0:
        return {"count": 0}
    gaps = anchors.to_series().diff().dt.days.dropna()
    return {
        "count": int(len(anchors)),
        "first": str(anchors[0].date()),
        "last": str(anchors[-1].date()),
        "mean_gap_days": float(gaps.mean()) if len(gaps) else None,
        "median_gap_days": float(gaps.median()) if len(gaps) else None,
        "min_gap_days": int(gaps.min()) if len(gaps) else None,
        "max_gap_days": int(gaps.max()) if len(gaps) else None,
    }
