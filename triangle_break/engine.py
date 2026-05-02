"""triangle_break orchestrator — multi-TF descriptive scan."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data import intraday_1h
from mb_scanner.resample import per_ticker_panel, to_5h, to_daily, to_monthly, to_weekly

from .detect import detect
from .schema import FAMILIES, OUTPUT_COLUMNS

OUT_DIR = Path("output")


def _resolve_asof(df: pd.DataFrame, asof: Optional[pd.Timestamp]) -> int:
    if asof is None:
        return len(df) - 1
    asof_ts = pd.Timestamp(asof)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None and asof_ts.tz is None:
        asof_ts = asof_ts.tz_localize(idx.tz)
    elif idx.tz is None and asof_ts.tz is not None:
        asof_ts = asof_ts.tz_convert("Europe/Istanbul").tz_localize(None)
    mask = idx <= asof_ts
    if not mask.any():
        return -1
    return int(np.flatnonzero(mask)[-1])


def scan(
    *,
    families: Iterable[str] = tuple(FAMILIES),
    tickers: Iterable[str] | None = None,
    asof: str | pd.Timestamp | None = None,
    min_coverage: float = 0.0,
    out_dir: str | Path = OUT_DIR,
    write_parquet: bool = True,
) -> dict[str, pd.DataFrame]:
    fam_keys = list(families)
    bad = [f for f in fam_keys if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown families: {bad}; valid={list(FAMILIES)}")

    bars = intraday_1h.load_intraday(
        tickers=list(tickers) if tickers is not None else None,
        start=None, end=None, min_coverage=min_coverage,
    )
    if bars.empty:
        return {f: pd.DataFrame(columns=list(OUTPUT_COLUMNS)) for f in fam_keys}

    needed_freqs = {FAMILIES[f].frequency for f in fam_keys}
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    if "5h" in needed_freqs:
        panels["5h"] = per_ticker_panel(to_5h(bars), "ts_istanbul")
    if "1d" in needed_freqs:
        panels["1d"] = per_ticker_panel(to_daily(bars), "date")
    if "1w" in needed_freqs:
        panels["1w"] = per_ticker_panel(to_weekly(bars), "week_end")
    if "1M" in needed_freqs:
        panels["1M"] = per_ticker_panel(to_monthly(bars), "month_end")

    asof_ts = pd.Timestamp(asof) if asof is not None else None

    rows_by_fam: dict[str, list[dict]] = {f: [] for f in fam_keys}
    for f in fam_keys:
        params = FAMILIES[f]
        panel_map = panels[params.frequency]
        for ticker, df in panel_map.items():
            asof_idx = _resolve_asof(df, asof_ts)
            if asof_idx < 0:
                continue
            row = detect(df, asof_idx, ticker, f, params)
            if row is not None:
                rows_by_fam[f].append(row)

    out: dict[str, pd.DataFrame] = {}
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for f in fam_keys:
        rows = rows_by_fam[f]
        df_out = (
            pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)) if rows
            else pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        )
        out[f] = df_out
        if write_parquet:
            target = out_dir_p / f"triangle_break_{f}.parquet"
            df_out.to_parquet(target, index=False)
    return out
