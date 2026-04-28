"""
Daily OHLCV data-quality audit.

Why this exists:
  Step 1 surfaced a max +0.94 close-vs-intraperiod drawdown gap. At monthly
  holding horizons, a single bad Low print (Low = 0, Low = 0.01, or a
  mis-aligned tick) can shove forward_max_dd_intraperiod to −1.0 and drag
  downstream diagnostics. Low-based features in Step 2 (vol_parkinson_20d)
  are just as vulnerable. Rather than silently mask, we SURFACE the flagged
  rows in a report and let the downstream code decide.

Flags produced (all are heuristics, not ground truth):

  low_to_close_ratio        Low / Close(same day)      < threshold → flag
  low_to_prev_close_ratio   Low / Close(prev day)      < threshold → flag
  low_to_open_ratio         Low / Open(same day)       < threshold → flag
  zero_or_negative_low      Low ≤ 0 on a non-NaN day              → flag

A row can match multiple flags; each flag column is a bool and the row is
emitted if ANY flag is True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_AUDIT_COLUMNS = [
    "ticker", "date",
    "open", "high", "low", "close", "prev_close",
    "low_to_close", "low_to_prev_close", "low_to_open",
    "flag_low_to_close", "flag_low_to_prev_close",
    "flag_low_to_open", "flag_zero_or_negative_low",
]


def scan_suspicious_lows(panel: dict[str, pd.DataFrame],
                         low_to_close_min: float = 0.5,
                         low_to_prev_close_min: float = 0.5,
                         low_to_open_min: float = 0.5) -> pd.DataFrame:
    """
    Walk every ticker's OHLCV, flag rows where Low looks implausibly far
    below reference prices. Returns one row per flagged (ticker, date),
    with the underlying prices + each flag so downstream can filter.
    """
    chunks: list[pd.DataFrame] = []
    for ticker, df in panel.items():
        if df is None or len(df) == 0:
            continue
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            continue

        open_ = df["Open"].astype(float)
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)

        ratio_lc  = (low / close).replace([np.inf, -np.inf], np.nan)
        ratio_lpc = (low / prev_close).replace([np.inf, -np.inf], np.nan)
        ratio_lo  = (low / open_).replace([np.inf, -np.inf], np.nan)

        flag_lc  = ratio_lc < low_to_close_min
        flag_lpc = ratio_lpc < low_to_prev_close_min
        flag_lo  = ratio_lo < low_to_open_min
        flag_zn  = (low <= 0) & low.notna()

        any_flag = (
            flag_lc.fillna(False)
            | flag_lpc.fillna(False)
            | flag_lo.fillna(False)
            | flag_zn.fillna(False)
        )
        if not any_flag.any():
            continue

        idx = df.index[any_flag]
        chunk = pd.DataFrame({
            "ticker": ticker,
            "date": idx,
            "open":  open_.loc[idx].to_numpy(),
            "high":  high.loc[idx].to_numpy(),
            "low":   low.loc[idx].to_numpy(),
            "close": close.loc[idx].to_numpy(),
            "prev_close": prev_close.loc[idx].to_numpy(),
            "low_to_close":      ratio_lc.loc[idx].to_numpy(),
            "low_to_prev_close": ratio_lpc.loc[idx].to_numpy(),
            "low_to_open":       ratio_lo.loc[idx].to_numpy(),
            "flag_low_to_close":        flag_lc.loc[idx].fillna(False).to_numpy(),
            "flag_low_to_prev_close":   flag_lpc.loc[idx].fillna(False).to_numpy(),
            "flag_low_to_open":         flag_lo.loc[idx].fillna(False).to_numpy(),
            "flag_zero_or_negative_low": flag_zn.loc[idx].fillna(False).to_numpy(),
        })
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=_AUDIT_COLUMNS)

    out = pd.concat(chunks, ignore_index=True)
    # Sort by worst offender first (lowest low_to_prev_close)
    return out.sort_values("low_to_prev_close", na_position="last").reset_index(drop=True)


def dq_summary(flagged: pd.DataFrame) -> dict:
    """One-line summary stats for run_meta."""
    if flagged.empty:
        return {"n_flagged_rows": 0, "n_flagged_tickers": 0}
    return {
        "n_flagged_rows": int(len(flagged)),
        "n_flagged_tickers": int(flagged["ticker"].nunique()),
        "top_tickers": (
            flagged["ticker"].value_counts().head(10).to_dict()
        ),
        "flag_breakdown": {
            "low_to_close":        int(flagged["flag_low_to_close"].sum()),
            "low_to_prev_close":   int(flagged["flag_low_to_prev_close"].sum()),
            "low_to_open":         int(flagged["flag_low_to_open"].sum()),
            "zero_or_negative":    int(flagged["flag_zero_or_negative_low"].sum()),
        },
    }
