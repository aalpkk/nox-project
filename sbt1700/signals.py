"""SBT signal detection on the daily master with the T row patched
to the 17:00-truncated bar.

A (ticker, date) is a "17:00 SBT signal" when, evaluating the daily
indicator stack with T's row replaced by the 17:00 bar:

  - Yesterday (T-1) was inside an active squeeze of length ≥ MIN_SQUEEZE_BARS
    and ≤ MAX_SQUEEZE_BARS. (The squeeze prefix is computed on EOD-complete
    prior bars, so the patched T does not change it.)
  - The box (high/low) over that prior squeeze is well-formed.
  - close_1700 > box_top + IMPULSE_ATR_MULT * atr_prior   (impulse breakout)
  - vol_1700 ≥ VOL_MULT * vol_sma_prior * (n_bars / EXPECTED_BARS_PER_PAIR)
    (volume scaled to elapsed session, not full-day)
  - close_1700 > htf_ema_prior     (above HTF trend)

Box geometry uses ONLY the squeeze prior to T. T's bar is the breakout
candle.

Output columns are flat; downstream features.py reads from this same
patched daily panel without recomputing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sbt1700.config import (
    EXPECTED_BARS_PER_PAIR,
    SBT_BB_LENGTH,
    SBT_BB_MULT,
    SBT_BB_WIDTH_THRESH,
    SBT_ATR_LENGTH,
    SBT_ATR_SMA_LENGTH,
    SBT_ATR_SQUEEZE_RATIO,
    SBT_MIN_SQUEEZE_BARS,
    SBT_MAX_SQUEEZE_BARS,
    SBT_IMPULSE_ATR_MULT,
    SBT_VOL_SMA_LENGTH,
    SBT_VOL_MULT,
    SBT_HTF_EMA_LENGTH,
)


PRIOR_REQ_BARS = max(
    SBT_BB_LENGTH, SBT_ATR_SMA_LENGTH, SBT_VOL_SMA_LENGTH,
    SBT_HTF_EMA_LENGTH,
) + SBT_MAX_SQUEEZE_BARS + 5


def _calc_prior_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute squeeze flags + indicators using EOD-complete prior bars only.

    `df` must be ordered ascending by date and indexed by Date. Computes
    everything end-of-bar; we will use these only on T-1 to define the
    squeeze prefix.
    """
    out = df.copy()
    c, h, l = out["Close"], out["High"], out["Low"]
    v = out["Volume"]

    sma = c.rolling(SBT_BB_LENGTH).mean()
    std = c.rolling(SBT_BB_LENGTH).std()
    upper = sma + SBT_BB_MULT * std
    lower = sma - SBT_BB_MULT * std
    out["bb_width"] = (upper - lower) / sma
    out["bb_width_sma"] = out["bb_width"].rolling(SBT_BB_LENGTH).mean()

    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(SBT_ATR_LENGTH).mean()
    out["atr_sma"] = out["atr"].rolling(SBT_ATR_SMA_LENGTH).mean()

    out["vol_sma"] = v.rolling(SBT_VOL_SMA_LENGTH).mean()
    out["htf_ema"] = c.ewm(span=SBT_HTF_EMA_LENGTH, adjust=False).mean()

    out["sq_bb"] = out["bb_width"] < out["bb_width_sma"] * SBT_BB_WIDTH_THRESH
    out["sq_atr"] = out["atr"] < out["atr_sma"] * SBT_ATR_SQUEEZE_RATIO
    out["squeeze"] = out["sq_bb"] & out["sq_atr"]
    return out


def _squeeze_run_lengths(squeeze: pd.Series) -> pd.Series:
    """Length of the active squeeze run ending at each bar (0 if not in squeeze)."""
    sq = squeeze.fillna(False).astype(int).values
    n = len(sq)
    runs = np.zeros(n, dtype=int)
    c = 0
    for i in range(n):
        c = c + 1 if sq[i] else 0
        runs[i] = c
    return pd.Series(runs, index=squeeze.index)


@dataclass
class CandidateRow:
    ticker: str
    date: pd.Timestamp
    box_top: float
    box_bottom: float
    squeeze_run_prior: int
    atr_prior: float
    vol_sma_prior: float
    htf_ema_prior: float
    bb_width_prior: float
    bb_width_sma_prior: float
    open_1700: float
    high_1700: float
    low_1700: float
    close_1700: float
    vol_1700: float
    n_bars_1700: int
    intraday_coverage: float


def detect_candidates_for_ticker(
    daily: pd.DataFrame,
    truncated_bars: pd.DataFrame,
    expected_bars: int = EXPECTED_BARS_PER_PAIR,
) -> pd.DataFrame:
    """Return SBT-1700 candidate rows for a single ticker.

    Args:
        daily: daily OHLCV indexed by Date for ONE ticker, ascending.
            Columns: Open, High, Low, Close, Volume.
        truncated_bars: 17:00-truncated bars for THIS ticker only, columns:
            signal_date (datetime, normalized), Open, High, Low, Close, Volume,
            n_bars, intraday_coverage.
        expected_bars: full-session bar count used to scale the volume gate
            from a daily SMA to the elapsed intraday session. Default = 27
            (15m grid). 1h variant passes 8.

    For each truncated bar, patches the corresponding T row of `daily` with
    the truncated values, recomputes indicators on bars [T-PRIOR_REQ_BARS .. T],
    and emits a candidate if the breakout/volume/HTF gates pass at T using
    the squeeze prefix that was active at T-1.
    """
    if daily.empty or truncated_bars.empty:
        return pd.DataFrame()

    daily = daily.sort_index()

    # Pre-compute prior-only indicators on the un-patched series for the
    # squeeze prefix lookup (T-1 squeeze run, prior atr/vol/htf at T-1).
    prior_ind = _calc_prior_indicators(daily)
    prior_run = _squeeze_run_lengths(prior_ind["squeeze"])

    rows: list[dict] = []

    for r in truncated_bars.itertuples(index=False):
        sig_date = pd.Timestamp(r.signal_date).normalize()
        if sig_date not in daily.index:
            continue

        # T-1 must exist and have populated indicators
        idx_pos = daily.index.get_loc(sig_date)
        if idx_pos == 0:
            continue
        prev_date = daily.index[idx_pos - 1]

        sq_run = int(prior_run.loc[prev_date]) if prev_date in prior_run.index else 0
        atr_prev = float(prior_ind.loc[prev_date, "atr"])
        vol_sma_prev = float(prior_ind.loc[prev_date, "vol_sma"])
        htf_ema_prev = float(prior_ind.loc[prev_date, "htf_ema"])
        bb_w_prev = float(prior_ind.loc[prev_date, "bb_width"])
        bb_w_sma_prev = float(prior_ind.loc[prev_date, "bb_width_sma"])

        if not (
            np.isfinite(atr_prev) and atr_prev > 0
            and np.isfinite(vol_sma_prev) and vol_sma_prev > 0
            and np.isfinite(htf_ema_prev)
        ):
            continue

        if not (SBT_MIN_SQUEEZE_BARS <= sq_run <= SBT_MAX_SQUEEZE_BARS):
            continue

        # Box geometry from the squeeze window ending at T-1.
        sq_start = idx_pos - sq_run
        box_slice = daily.iloc[sq_start:idx_pos]
        if box_slice.empty:
            continue
        box_top = float(box_slice["High"].max())
        box_bottom = float(box_slice["Low"].min())
        if not (box_top > box_bottom > 0):
            continue

        # 17:00 truncated values for T
        o = float(r.Open)
        h = float(r.High)
        l = float(r.Low)
        c = float(r.Close)
        vol_1700 = float(r.Volume)
        n_bars = int(r.n_bars)
        cov = float(r.intraday_coverage)

        if not np.isfinite(c) or c <= 0:
            continue

        # Breakout gate: close_1700 > box_top + impulse * atr_prev
        if c <= box_top + SBT_IMPULSE_ATR_MULT * atr_prev:
            continue

        # Volume gate: scale 20d full-day SMA to elapsed session.
        # Elapsed fraction = n_bars / expected_bars.
        elapsed_frac = max(n_bars / expected_bars, 1e-6)
        if vol_1700 < SBT_VOL_MULT * vol_sma_prev * elapsed_frac:
            continue

        # HTF gate
        if c <= htf_ema_prev:
            continue

        rows.append({
            "ticker": daily.attrs.get("ticker") or "",
            "date": sig_date,
            "box_top": box_top,
            "box_bottom": box_bottom,
            "squeeze_run_prior": sq_run,
            "atr_prior": atr_prev,
            "vol_sma_prior": vol_sma_prev,
            "htf_ema_prior": htf_ema_prev,
            "bb_width_prior": bb_w_prev,
            "bb_width_sma_prior": bb_w_sma_prev,
            "open_1700": o,
            "high_1700": h,
            "low_1700": l,
            "close_1700": c,
            "vol_1700": vol_1700,
            "n_bars_1700": n_bars,
            "intraday_coverage": cov,
        })

    return pd.DataFrame(rows)


def detect_candidates(
    daily_master: pd.DataFrame,
    truncated_bars: pd.DataFrame,
    expected_bars: int = EXPECTED_BARS_PER_PAIR,
) -> pd.DataFrame:
    """Whole-panel candidate detection.

    Args:
        daily_master: long table indexed by Date, with `ticker` column and
            OHLCV (the Fintables 10y master schema).
        truncated_bars: output of aggregator.aggregate_truncated_bars,
            with `ticker`, `signal_date`, OHLCV, n_bars, intraday_coverage.
        expected_bars: full-session bar count for elapsed-session volume scaling
            (15m=27 default, 1h=8).
    """
    if daily_master.empty or truncated_bars.empty:
        return pd.DataFrame()

    if daily_master.index.name != "Date":
        daily_master = daily_master.copy()
        daily_master.index.name = "Date"

    out_chunks: list[pd.DataFrame] = []
    for tk, t_bars in truncated_bars.groupby("ticker"):
        sub = daily_master[daily_master["ticker"] == tk]
        if sub.empty:
            continue
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        sub.attrs["ticker"] = tk
        cands = detect_candidates_for_ticker(sub, t_bars, expected_bars=expected_bars)
        if not cands.empty:
            out_chunks.append(cands)

    if not out_chunks:
        return pd.DataFrame()
    return pd.concat(out_chunks, ignore_index=True)
