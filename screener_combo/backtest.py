"""Forward-return analysis and combination-method comparison.

Trigger semantics:
  Signal day = T (close known after BIST close).
  Entry      = T+1 open (next trading day on the same ticker's date axis).
  Exit horizons evaluated: H ∈ {5, 10, 20} bars.
  Return    = (close_{entry+H-1} − open_{entry}) / open_{entry}      [no slippage at scan stage]

Per-signal metrics (and per-combination):
  N_signals
  hit_rate (%)         = share of trades with R > 0
  mean_R               = average return
  median_R
  PF                   = sum(positive R) / |sum(negative R)|
  max_drawdown         = peak-to-trough on equal-weight signal stream

Combination methods:
  vote2  = ≥2 of 3 signals fire same day
  vote3  = all 3 fire same day
  cascade_RT_AS = regime_trig & alsat_trig  (cascading 2 of 3)
  rank_top  = NOT applicable here (no per-signal score) — fall back to vote-based.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


HORIZONS = (5, 10, 20)


# ============================================================
# Forward return computation
# ============================================================

def add_forward_returns(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add fwd return columns (entry = T+1 open, exit = T+H close) per ticker.

    Returns a DataFrame keyed (ticker, date) with columns:
       entry_open, fwd_R_5, fwd_R_10, fwd_R_20
    """
    out = []
    for tkr, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        # Entry = next bar's open
        entry_open = g["open"].shift(-1)
        row = {
            "ticker": tkr,
            "date": g["date"].values,
            "entry_open": entry_open.values,
        }
        for h in horizons:
            # Exit close at index i+h (i.e. holding period h bars from entry)
            exit_close = g["close"].shift(-h)
            r = (exit_close - entry_open) / entry_open
            row[f"fwd_R_{h}"] = r.values
        out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True)


def attach_returns(triggers: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Left-join forward returns onto trigger table."""
    return triggers.merge(returns, on=["ticker", "date"], how="left")


# ============================================================
# Combination columns
# ============================================================

def add_combinations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rt = df["regime_trig"].astype(bool)
    nw = df["weekly_trig"].astype(bool)
    al = df["alsat_trig"].astype(bool)
    df["count_trig"] = rt.astype(int) + nw.astype(int) + al.astype(int)
    df["vote2"] = df["count_trig"] >= 2
    df["vote3"] = df["count_trig"] >= 3
    df["cascade_RT_AS"] = rt & al
    df["cascade_RT_NW"] = rt & nw
    df["cascade_NW_AS"] = nw & al
    return df


# ============================================================
# Metrics
# ============================================================

@dataclass
class CohortStats:
    name: str
    horizon: int
    n: int
    hit: float
    mean_R: float
    median_R: float
    pf: float
    p10: float
    p90: float
    cum_eq_R: float  # equal-weight cumulative log-style sum (proxy for stacked equity)


def _safe_pf(r: pd.Series) -> float:
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def cohort_stats(name: str, horizon: int, returns: pd.Series) -> CohortStats:
    r = returns.dropna()
    if len(r) == 0:
        return CohortStats(name, horizon, 0, 0, 0, 0, 0, 0, 0, 0)
    return CohortStats(
        name=name,
        horizon=horizon,
        n=len(r),
        hit=float((r > 0).mean()) * 100,
        mean_R=float(r.mean()) * 100,
        median_R=float(r.median()) * 100,
        pf=_safe_pf(r),
        p10=float(r.quantile(0.10)) * 100,
        p90=float(r.quantile(0.90)) * 100,
        cum_eq_R=float(r.sum()) * 100,
    )


SIGNAL_COLS = ["regime_trig", "weekly_trig", "alsat_trig"]
COMBO_COLS = ["vote2", "vote3", "cascade_RT_AS", "cascade_RT_NW", "cascade_NW_AS"]


def all_cohorts(table: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Build a per-cohort × per-horizon stats table."""
    rows = []
    for h in horizons:
        col = f"fwd_R_{h}"
        # Per-signal cohorts
        for sig in SIGNAL_COLS:
            r = table.loc[table[sig], col]
            stats = cohort_stats(sig, h, r)
            rows.append(stats.__dict__)
        # Combination cohorts
        for cmb in COMBO_COLS:
            r = table.loc[table[cmb], col]
            stats = cohort_stats(cmb, h, r)
            rows.append(stats.__dict__)
    return pd.DataFrame(rows)


# ============================================================
# Per-day "best stock" picker
# ============================================================

def daily_picks(table: pd.DataFrame, gate: str, top_n: int | None = None) -> pd.DataFrame:
    """Filter to bars where `gate` is True; if top_n given, keep first N per date
    (alphabetical — no inter-signal ranking yet)."""
    sub = table[table[gate]].copy()
    if top_n is not None:
        sub = sub.sort_values(["date", "ticker"]).groupby("date").head(top_n)
    return sub


def daily_topk_metrics(table: pd.DataFrame, gate: str, horizon: int, top_k: int) -> dict:
    """Take top_k tickers per date for a given gate; equal-weight average return per day,
    then aggregate to PF/mean/Sharpe-like stats over the day stream."""
    col = f"fwd_R_{horizon}"
    sub = table.loc[table[gate], ["date", "ticker", col]].dropna()
    if sub.empty:
        return {"gate": gate, "horizon": horizon, "top_k": top_k, "days": 0}
    daily = (
        sub.sort_values(["date", "ticker"])
        .groupby("date")
        .head(top_k)
        .groupby("date")[col]
        .mean()
    )
    return {
        "gate": gate,
        "horizon": horizon,
        "top_k": top_k,
        "days": int(daily.shape[0]),
        "mean_daily_R_%": float(daily.mean()) * 100,
        "median_daily_R_%": float(daily.median()) * 100,
        "hit_daily_%": float((daily > 0).mean()) * 100,
        "PF_daily": _safe_pf(daily),
        "cum_R_%": float(daily.sum()) * 100,
    }
