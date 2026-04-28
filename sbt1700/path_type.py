"""SBT-1700 — forward-path classifier.

Exit-agnostic tagger that labels every (ticker, signal_date) row by the
shape of its forward daily-bar OHLC, *independent of any exit choice*.

This is the diagnostic input for capture decomposition: we want to ask
"where does our %23 capture loss come from?" and that question is only
meaningful if the path label is the same across all candidate exits.

V1 definitions (deliberately simple + deterministic; relax later)
-----------------------------------------------------------------
Canonical R   = 1.5 × atr_1700  (matches the dominant initial_sl in
                                  carried variants)
Forward window = next 20 daily bars after T (T's daily bar excluded)

Per-row diagnostics:
    mfe_R_path           = max((high_i - entry_px) / R) over the window
    bars_to_mfe          = i in 1..N at which the max above is reached
    mfe_R_first3         = max((high_i - entry_px) / R) over i in 1..3
    post_mfe_giveback    = (max_high - min(low after mfe_bar)) /
                           (max_high - entry_px)   in (0, 1]
                           (NaN when mfe ≤ 0 or no bars after mfe_bar)

Tags:
    parabolic   : bars_to_mfe ≤ 3 AND mfe_R_path ≥ 1.5
                  (sharp early peak; the canonical "give-it-all-back"
                  shape diagnosed in nyxexp winner-path analysis)

    spike_fade  : NOT parabolic AND mfe_R_first3 ≥ 1.0
                  AND post_mfe_giveback ≥ 0.50
                  (touched +1R early but the MFE level itself was modest
                  AND price gave back at least half of it within the
                  window — looked good, ended weak)

    clean       : remainder
                  (sustained move, slow extension, or no movement)

Notes
-----
* Window is fixed at 20 to keep classification stable across variants
  whose `max_hold_bars` differ (10/20/40 in our grid).
* The classifier is read-only over the daily master; it never consumes
  the dataset row's label columns.
* Rows without forward bars are tagged `unknown` and excluded from
  decomposition tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


CANONICAL_SL_ATR_K = 1.5
CLASSIFY_WINDOW_BARS = 20

PARABOLIC_BARS_TO_MFE_MAX = 3
PARABOLIC_MFE_R_MIN = 1.5
SPIKE_FADE_FIRST3_R_MIN = 1.0
SPIKE_FADE_GIVEBACK_MIN = 0.50

PATH_PARABOLIC = "parabolic"
PATH_SPIKE_FADE = "spike_fade"
PATH_CLEAN = "clean"
PATH_UNKNOWN = "unknown"

VALID_PATH_TAGS = {PATH_PARABOLIC, PATH_SPIKE_FADE, PATH_CLEAN, PATH_UNKNOWN}


@dataclass(frozen=True)
class PathDiagnostic:
    path_type: str
    mfe_R_path: float
    bars_to_mfe: int
    mfe_R_first3: float
    post_mfe_giveback_pct: float
    n_forward_bars: int


def classify_path(
    entry_date: pd.Timestamp,
    entry_px: float,
    atr_1700: float,
    forward_ohlc: pd.DataFrame,
    window_bars: int = CLASSIFY_WINDOW_BARS,
) -> PathDiagnostic:
    """Tag the forward path of a single signal.

    Parameters
    ----------
    entry_date : Timestamp
        Signal date T. Forward bars are dates strictly greater.
    entry_px : float
        17:00-truncated close used as the long entry price.
    atr_1700 : float
        ATR(14) prior — same field the simulator reads.
    forward_ohlc : DataFrame
        Daily OHLC for the same ticker, indexed by date.
    """
    if not (np.isfinite(entry_px) and np.isfinite(atr_1700) and atr_1700 > 0):
        return PathDiagnostic(
            path_type=PATH_UNKNOWN,
            mfe_R_path=float("nan"),
            bars_to_mfe=0,
            mfe_R_first3=float("nan"),
            post_mfe_giveback_pct=float("nan"),
            n_forward_bars=0,
        )

    R = CANONICAL_SL_ATR_K * atr_1700
    fwd = forward_ohlc[forward_ohlc.index > entry_date].head(window_bars)
    if fwd.empty:
        return PathDiagnostic(
            path_type=PATH_UNKNOWN,
            mfe_R_path=float("nan"),
            bars_to_mfe=0,
            mfe_R_first3=float("nan"),
            post_mfe_giveback_pct=float("nan"),
            n_forward_bars=0,
        )

    highs = fwd["High"].to_numpy(dtype=float)
    lows = fwd["Low"].to_numpy(dtype=float)
    n = len(highs)

    excursion = (highs - entry_px) / R           # MFE in R per bar
    bar_to_mfe_idx = int(np.argmax(excursion))   # 0-indexed
    mfe_R_path = float(excursion[bar_to_mfe_idx])
    bars_to_mfe = bar_to_mfe_idx + 1             # 1-indexed bar number

    first3 = excursion[: min(3, n)]
    mfe_R_first3 = float(np.max(first3))

    if mfe_R_path > 0 and bar_to_mfe_idx + 1 < n:
        max_high = float(highs[bar_to_mfe_idx])
        min_low_after = float(np.min(lows[bar_to_mfe_idx + 1 :]))
        denom = max_high - entry_px
        post_mfe_giveback_pct = (
            (max_high - min_low_after) / denom if denom > 1e-12 else float("nan")
        )
    else:
        post_mfe_giveback_pct = float("nan")

    if (bars_to_mfe <= PARABOLIC_BARS_TO_MFE_MAX
            and mfe_R_path >= PARABOLIC_MFE_R_MIN):
        path = PATH_PARABOLIC
    elif (mfe_R_first3 >= SPIKE_FADE_FIRST3_R_MIN
            and np.isfinite(post_mfe_giveback_pct)
            and post_mfe_giveback_pct >= SPIKE_FADE_GIVEBACK_MIN):
        path = PATH_SPIKE_FADE
    else:
        path = PATH_CLEAN

    return PathDiagnostic(
        path_type=path,
        mfe_R_path=mfe_R_path,
        bars_to_mfe=bars_to_mfe,
        mfe_R_first3=mfe_R_first3,
        post_mfe_giveback_pct=post_mfe_giveback_pct,
        n_forward_bars=n,
    )


def classify_panel(
    panel: pd.DataFrame,
    daily_master: pd.DataFrame,
    window_bars: int = CLASSIFY_WINDOW_BARS,
) -> pd.DataFrame:
    """Vectorise `classify_path` across a TRAIN/VAL panel.

    Required panel columns: ticker, date, close_1700, atr14_prior.
    Returns a DataFrame keyed by (ticker, date) with diagnostic columns
    plus `path_type`. Caller joins back into the trade frame.
    """
    needed = {"ticker", "date", "close_1700", "atr14_prior"}
    missing = needed - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")
    if "ticker" not in daily_master.columns:
        raise KeyError("daily_master must have a 'ticker' column")

    by_ticker = {
        tk: g[["High", "Low", "Close"]].sort_index()
        for tk, g in daily_master.groupby("ticker")
    }

    pcl = panel.copy()
    pcl["date"] = pd.to_datetime(pcl["date"])

    rows: list[dict] = []
    for r in pcl.itertuples(index=False):
        tk = getattr(r, "ticker")
        sub = by_ticker.get(tk)
        if sub is None or sub.empty:
            rows.append({
                "ticker": tk,
                "date": pd.Timestamp(getattr(r, "date")),
                "path_type": PATH_UNKNOWN,
                "mfe_R_path": float("nan"),
                "bars_to_mfe": 0,
                "mfe_R_first3": float("nan"),
                "post_mfe_giveback_pct": float("nan"),
                "n_forward_bars": 0,
            })
            continue
        diag = classify_path(
            entry_date=pd.Timestamp(getattr(r, "date")),
            entry_px=float(getattr(r, "close_1700")),
            atr_1700=float(getattr(r, "atr14_prior")),
            forward_ohlc=sub,
            window_bars=window_bars,
        )
        rows.append({
            "ticker": tk,
            "date": pd.Timestamp(getattr(r, "date")),
            "path_type": diag.path_type,
            "mfe_R_path": diag.mfe_R_path,
            "bars_to_mfe": diag.bars_to_mfe,
            "mfe_R_first3": diag.mfe_R_first3,
            "post_mfe_giveback_pct": diag.post_mfe_giveback_pct,
            "n_forward_bars": diag.n_forward_bars,
        })
    return pd.DataFrame(rows)
