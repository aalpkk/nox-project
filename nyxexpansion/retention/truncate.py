"""17:00 TR bar truncation + feature rebuild for the retention stage.

Two entry points:

- ``aggregate_truncated_bars(intraday_df)`` — batch path used by research /
  offline audit: collapse a 15m cache into per-(ticker, signal_date)
  daily bars that only reflect the 10:00-17:00 TR window.

- ``rebuild_truncated_features(bars_df, ...)`` — batch feature recompute over
  those truncated bars using ``compute_per_ticker_features`` with T's
  OHLCV row point-in-time replaced. Prior bars are untouched.

The end-of-window convention in the 15m cache is: ``bar_ts`` is the CLOSE time
of the 15m window. The 17:00 cutoff therefore keeps 28 bars (10:15..17:00
inclusive) and drops the final four (17:15..18:00).

Caller paths:
- research/truncated_17_00_rebuild.py — research batch (historical audit)
- scan live retention stage (step 3 of impl plan) — single-pair variant TBD
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


BAR_CUTOFF_HH = 17
BAR_CUTOFF_MM = 0
EXPECTED_BARS_PER_PAIR = 28

_OHLC_COLS = ("Open", "High", "Low", "Close", "Volume")


def aggregate_truncated_bars(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 15m bars with ``bar_ts`` ≤ 17:00 TR into a single daily bar
    per ``(ticker, signal_date)``.

    Expected columns on ``intraday_df``: ``ticker``, ``signal_date``, ``bar_ts``
    (ms since epoch, UTC), ``open``, ``high``, ``low``, ``close``, ``volume``.

    Returns a frame with canonical OHLCV column names (``Open`` .. ``Volume``)
    ready to be patched into the daily master.
    """
    df = intraday_df.copy()
    df["ts_utc"] = pd.to_datetime(df["bar_ts"], unit="ms", utc=True)
    df["ts_tr"] = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hh"] = df["ts_tr"].dt.hour
    df["mm"] = df["ts_tr"].dt.minute

    mask = (df["hh"] < BAR_CUTOFF_HH) | (
        (df["hh"] == BAR_CUTOFF_HH) & (df["mm"] <= BAR_CUTOFF_MM)
    )
    trunc = df[mask].copy()

    g = trunc.sort_values("ts_tr").groupby(["ticker", "signal_date"])
    agg = g.agg(
        Open=("open", "first"),
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
        Volume=("volume", "sum"),
        n_bars=("open", "size"),
        last_bar_ts_tr=("ts_tr", "last"),
    ).reset_index()

    agg["signal_date"] = pd.to_datetime(agg["signal_date"])
    return agg


def rebuild_truncated_features(
    bars: pd.DataFrame,
    master_ohlcv_path: Path | str,
    xu100_close: pd.Series | None = None,
    n_limit: int | None = None,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Per ``(ticker, signal_date)`` in ``bars``, patch T's row in the master
    OHLCV with the truncated bar and run ``compute_per_ticker_features``.

    Returns a long panel keyed by ``(ticker, date)`` holding every feature
    the standard pipeline emits for T. Prior (T-1, T-2, …) rows in the master
    remain untouched — they are legitimately available at 17:00 T because
    they closed on earlier calendar days.

    Known sub-leak: ``xu100_close`` stays on its full-close series (no 15m
    index data in cache). Flagged in the leakage audit.
    """
    # Local import to keep module import graph light — features.py drags
    # in the full pandas/numpy feature stack.
    from nyxexpansion.features import compute_per_ticker_features

    master = pd.read_parquet(master_ohlcv_path)
    if master.index.name:
        master = master.reset_index()
    master_by_tk = {tk: g.reset_index(drop=True)
                    for tk, g in master.groupby("ticker")}

    rows: list[dict] = []
    pairs = bars.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
    if n_limit:
        pairs = pairs.head(n_limit)

    total = len(pairs)
    for i, r in enumerate(pairs.itertuples(index=False), 1):
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  [{i}/{total}] {r.ticker} @ {r.signal_date.date()}")
        tk = r.ticker
        dte = pd.Timestamp(r.signal_date).normalize()
        g = master_by_tk.get(tk)
        if g is None or g.empty:
            continue

        sub = g.copy()
        sub["Date"] = pd.to_datetime(sub["Date"]).dt.normalize()
        sub = sub.set_index("Date").sort_index()
        if dte not in sub.index:
            continue
        for col in _OHLC_COLS:
            sub.at[dte, col] = float(getattr(r, col))

        feats = compute_per_ticker_features(
            sub, xu100_close=xu100_close, trigger_level=None,
        )
        if dte not in feats.index:
            continue
        row = feats.loc[dte].to_dict()
        row["ticker"] = tk
        row["date"] = dte
        rows.append(row)

    return pd.DataFrame(rows)
