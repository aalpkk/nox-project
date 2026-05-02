"""Scanner orchestrator (V1.4 — multi-frequency snapshot mode).

Loads the locked nox_intraday_v1 1h master once, builds per-family panels:
  - daily-class families (horizontal_base): 1h → daily resample
  - 1h-class families (mitigation_block, breaker_block): raw 1h bars

Runs each enabled family trigger AS-OF a snapshot timestamp, scores rows,
validates schema, and writes a parquet. Triggers emit at most ONE row per
ticker per asof; output has at most one row per (ticker, family) describing
its current state in the family's state vocabulary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from data import intraday_1h

from .context import build_market_context, fill_cross_sectional
from .schema import (
    FAMILIES,
    output_columns_for,
    validate_output_columns,
)
from .scoring import score_row
from .triggers import breaker_block, horizontal_base, mitigation_block


TRIGGERS = {
    "horizontal_base": horizontal_base.detect,
    "mitigation_block": mitigation_block.detect,
    "breaker_block": breaker_block.detect,
}

FAMILY_FREQUENCY: dict[str, Literal["1d", "1h"]] = {
    "horizontal_base": "1d",
    "mitigation_block": "1h",
    "breaker_block": "1h",
}


def _per_ticker_daily(daily_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker, g in daily_panel.groupby("ticker", observed=True):
        g = g.sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        df.attrs["ticker"] = str(ticker)
        out[str(ticker)] = df
    return out


def _per_ticker_intraday(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-ticker 1h DataFrames indexed by ts_istanbul (tz-aware)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, g in bars.groupby("ticker", observed=True):
        g = g.sort_values("ts_istanbul")
        idx = pd.DatetimeIndex(g["ts_istanbul"])
        df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        df.attrs["ticker"] = str(ticker)
        out[str(ticker)] = df
    return out


def scan(
    *,
    tickers: Iterable[str] | None = None,
    families: Iterable[str] = ("horizontal_base",),
    asof: str | pd.Timestamp | None = None,
    min_coverage: float = 0.0,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Snapshot scan as-of `asof`.

    For daily families, asof is normalized to a date; for 1h families, it is
    treated as an Istanbul timestamp (defaults to last bar in the panel).
    """
    fams = list(families)
    bad = [f for f in fams if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown families: {bad}; allowed={FAMILIES}")
    missing = [f for f in fams if f not in TRIGGERS]
    if missing:
        raise ValueError(f"families lack trigger impl yet: {missing}")

    bars = intraday_1h.load_intraday(
        tickers=list(tickers) if tickers is not None else None,
        start=None,
        end=None,
        min_coverage=min_coverage,
    )
    if bars.empty:
        return pd.DataFrame()

    daily_panel = intraday_1h.daily_resample(bars)
    if asof is None:
        asof_daily = pd.Timestamp(daily_panel["date"].max()).normalize()
        asof_intraday = pd.Timestamp(bars["ts_istanbul"].max())
    else:
        asof_ts = pd.Timestamp(asof)
        asof_daily = asof_ts.normalize()
        asof_intraday = asof_ts

    market_df, rs_df = build_market_context(daily_panel)

    needs_daily = any(FAMILY_FREQUENCY[f] == "1d" for f in fams)
    needs_intraday = any(FAMILY_FREQUENCY[f] == "1h" for f in fams)
    daily_by_t = _per_ticker_daily(daily_panel) if needs_daily else {}
    intraday_by_t = _per_ticker_intraday(bars) if needs_intraday else {}

    all_rows: list[dict] = []
    for fam in fams:
        detect = TRIGGERS[fam]
        cols = [c.name for c in output_columns_for(fam)]
        freq = FAMILY_FREQUENCY[fam]
        if freq == "1d":
            panel = daily_by_t
            asof_for_fam = asof_daily
        else:
            panel = intraday_by_t
            asof_for_fam = asof_intraday
        for ticker, df in panel.items():
            for row in detect(df, asof=asof_for_fam):
                fill_cross_sectional(row, market_df, rs_df)
                score_row(row)
                for col in cols:
                    row.setdefault(col, None)
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    out = pd.DataFrame(all_rows)
    validate_output_columns(out.columns)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out
