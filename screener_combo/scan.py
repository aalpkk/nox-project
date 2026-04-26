"""Run 3 signals across full panel → long-form trigger table.

Output columns:
  ticker, date,
  regime_trig, weekly_trig, alsat_trig,            # boolean gates
  rt_subtype, rt_tier,                             # RT category labels
  nox_d_trig, nox_dw_type,                         # nox D vs W vs D+W
  as_subtype, as_decision,                         # AS sub-category + decision
  days_since_<gate>                                # signal age per gate
  + feature panel columns
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from screener_combo.signals import (
    regime_transition_rich,
    nox_rich,
    alsat_rich,
)
from screener_combo.features import compute_feature_panel, FEATURE_COLS


def _days_since_true(s: pd.Series) -> pd.Series:
    """For each row, bars elapsed since the most recent True in `s`.
    NaN before the first True occurrence.
    """
    s = s.astype(bool)
    idx = pd.Series(np.arange(len(s)), index=s.index)
    last_true_idx = idx.where(s).ffill()
    return (idx - last_true_idx)


def _to_indexed(per_ticker_daily: pd.DataFrame) -> pd.DataFrame:
    d = per_ticker_daily.set_index("date").drop(columns=["ticker"]).sort_index()
    # signals expect lowercase OHLCV columns — already lowercase via daily_resample
    return d


def _weekly_from_daily(daily_idx: pd.DataFrame) -> pd.DataFrame:
    return (
        daily_idx
        .resample("W-FRI")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )


def scan_panel(
    panel_daily: pd.DataFrame,
    bench_close: pd.Series,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    min_bars: int = 80,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Run 3 signals across all tickers in panel; return long-form trigger table."""
    rows = []
    tickers = sorted(panel_daily["ticker"].unique())
    t0 = time.time()
    for i, tkr in enumerate(tickers):
        sub = panel_daily[panel_daily.ticker == tkr]
        if len(sub) < min_bars:
            continue
        d = _to_indexed(sub)
        if start is not None:
            # Need history before `start` for warmup, so don't truncate input here.
            # We'll filter the trigger output below.
            pass
        try:
            w = _weekly_from_daily(d)
            if len(w) < 12:
                continue
            rt_df = regime_transition_rich(d, w)
            nw_df = nox_rich(d, w)
            as_df = alsat_rich(d, w, bench_close)
            feats = compute_feature_panel(d, w, bench_close)
        except Exception as e:
            print(f"  [skip] {tkr}: {type(e).__name__}: {e}")
            continue

        rt_trig = rt_df["al_signal"].astype(bool)
        nw_trig = nw_df["nox_w_trig"].astype(bool)
        as_trig = as_df["al_signal"].astype(bool)
        nox_d_trig = nw_df["nox_d_trig"].astype(bool)

        df = pd.DataFrame({
            # primary gates (boolean — keep for backward compat with existing analysis)
            "regime_trig": rt_trig,
            "weekly_trig": nw_trig,
            "alsat_trig":  as_trig,
            # RT category labels
            "rt_subtype":  rt_df["rt_subtype"],
            "rt_tier":     rt_df["rt_tier"],
            # RT entry-quality scores (UI columns; T/P/E/regime come from features panel)
            "rt_entry_score": rt_df["rt_entry_score"],     # 0-4 (GIR)
            "rt_oe_score":    rt_df["rt_oe_score"],        # 0-4 (OK)
            # NOX granularity
            "nox_d_trig":  nox_d_trig,
            "nox_dw_type": nw_df["nox_dw_type"],
            # AS category + decision
            "as_subtype":  as_df["as_subtype"],
            "as_decision": as_df["as_decision"],
            # signal age — bars since this gate last fired (NaN before first fire)
            "days_since_RT": _days_since_true(rt_trig),
            "days_since_NW": _days_since_true(nw_trig),
            "days_since_AS": _days_since_true(as_trig),
            "days_since_ND": _days_since_true(nox_d_trig),
        }, index=d.index)
        df = df.join(feats[FEATURE_COLS])
        df.index.name = "date"
        df = df.reset_index()
        df.insert(0, "ticker", tkr)
        if start is not None:
            df = df[df.date >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.date <= pd.Timestamp(end)]
        # Keep only days where at least one signal fired (to keep table small) +
        # also keep ALL ticker-days for forward-return alignment? We need fwd returns,
        # so we'll keep only signal days and look up forward returns from panel.
        any_trig = df[["regime_trig", "weekly_trig", "alsat_trig", "nox_d_trig"]].any(axis=1)
        df = df[any_trig]
        rows.append(df)

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tickers)}] {tkr} | elapsed {elapsed:.1f}s")

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "date",
            "regime_trig", "weekly_trig", "alsat_trig",
            "rt_subtype", "rt_tier",
            "nox_d_trig", "nox_dw_type",
            "as_subtype", "as_decision",
            "days_since_RT", "days_since_NW", "days_since_AS", "days_since_ND",
        ])

    out = pd.concat(rows, ignore_index=True)
    return out
