"""Forward 5-day potential labels for the SBT-1700 setup-quality ranker.

Replaces the E3 realized-R label with execution-free forward potential.
The ranker target answers: "given an SBT-1700 trigger at 17:00 on T,
how strong is the upward potential over the next 5 trading days?"

For each (ticker, T):
    entry      = close_1700 (T's 17:00 truncated close)
    atr_prior  = ATR(14) on EOD-complete bars up to T-1 (= atr14_prior)
    fwd_window = T+1 .. T+5 daily bars (T's EOD close is NEVER consumed)
    fwd_high_5d  = max High over the forward window
    fwd_close_5d = Close on T+5 (or last available bar, see notes)
    fwd_n_bars   = number of forward bars actually present

Emitted labels:
    mfe_5d_R          = (fwd_high_5d  - entry) / atr_prior
    ret_5d_close_R    = (fwd_close_5d - entry) / atr_prior
    mfe_5d_pct        = (fwd_high_5d  - entry) / entry
    ret_5d_close_pct  = (fwd_close_5d - entry) / entry
    hit_1R_5d         = mfe_5d_R >= 1.0
    hit_2R_5d         = mfe_5d_R >= 2.0
    close_positive_5d = ret_5d_close_R > 0

Coverage rules:
    - If fewer than ``min_fwd_bars`` forward bars exist (typically the
      tail of the panel where T+5 hasn't materialized yet), the row is
      kept but every 5d label is NaN. The caller drops these from
      training; they're useful for diagnostics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_COLS_5D = [
    "fwd_high_5d", "fwd_close_5d", "fwd_n_bars",
    "mfe_5d_R", "ret_5d_close_R",
    "mfe_5d_pct", "ret_5d_close_pct",
    "hit_1R_5d", "hit_2R_5d", "close_positive_5d",
]

DEFAULT_HORIZON = 5
DEFAULT_MIN_FWD_BARS = 5  # need full window for primary labels


def _compute_one(
    forward_ohlc: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_px: float,
    atr_prior: float,
    horizon: int,
    min_fwd_bars: int,
) -> dict:
    """Compute 5d labels for a single (ticker, T) row.

    forward_ohlc must be the per-ticker daily panel indexed by Date,
    ascending, with Open/High/Low/Close. The function slices T+1..T+H.
    """
    out = {c: np.nan for c in LABEL_COLS_5D}
    out["fwd_n_bars"] = 0

    if forward_ohlc.empty:
        return out
    if not (np.isfinite(entry_px) and entry_px > 0):
        return out
    if not (np.isfinite(atr_prior) and atr_prior > 0):
        return out

    # Forward window starts strictly after T. T's daily close MUST NOT enter.
    fwd = forward_ohlc.loc[forward_ohlc.index > entry_date].head(horizon)
    n = len(fwd)
    out["fwd_n_bars"] = int(n)

    if n < min_fwd_bars:
        return out

    fwd_high = float(fwd["High"].max())
    fwd_close = float(fwd["Close"].iloc[-1])

    out["fwd_high_5d"] = fwd_high
    out["fwd_close_5d"] = fwd_close
    out["mfe_5d_R"] = (fwd_high - entry_px) / atr_prior
    out["ret_5d_close_R"] = (fwd_close - entry_px) / atr_prior
    out["mfe_5d_pct"] = (fwd_high - entry_px) / entry_px
    out["ret_5d_close_pct"] = (fwd_close - entry_px) / entry_px
    out["hit_1R_5d"] = bool(out["mfe_5d_R"] >= 1.0)
    out["hit_2R_5d"] = bool(out["mfe_5d_R"] >= 2.0)
    out["close_positive_5d"] = bool(out["ret_5d_close_R"] > 0)
    return out


def attach_labels_5d(
    daily_master: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    min_fwd_bars: int = DEFAULT_MIN_FWD_BARS,
) -> pd.DataFrame:
    """Append forward 5d potential labels to the feature panel.

    Args:
        daily_master: long table indexed by Date with ``ticker`` column
            and ``Open/High/Low/Close[/Volume]``. Used only for the forward
            window — never reads T's row directly.
        feature_panel: rows from features.build_features, must contain
            ``ticker``, ``date``, ``close_1700``, ``atr14_prior``.
        horizon: forward bar count (default 5).
        min_fwd_bars: minimum forward bars required to emit non-NaN
            labels (default 5; tail-of-panel rows fall through).

    Returns:
        feature_panel with LABEL_COLS_5D appended (same row order).
    """
    if feature_panel.empty:
        return feature_panel

    daily_master = daily_master.sort_index()
    by_ticker = {
        tk: g[["Open", "High", "Low", "Close"]].sort_index()
        for tk, g in daily_master.groupby("ticker")
    }

    rows: list[dict] = []
    for r in feature_panel.itertuples(index=False):
        sub = by_ticker.get(r.ticker)
        if sub is None or sub.empty:
            rows.append({c: np.nan for c in LABEL_COLS_5D} | {"fwd_n_bars": 0})
            continue
        T = pd.Timestamp(r.date).normalize()
        rows.append(_compute_one(
            forward_ohlc=sub,
            entry_date=T,
            entry_px=float(r.close_1700),
            atr_prior=float(r.atr14_prior),
            horizon=horizon,
            min_fwd_bars=min_fwd_bars,
        ))

    label_df = pd.DataFrame(rows, index=feature_panel.index)
    return pd.concat(
        [feature_panel.reset_index(drop=True),
         label_df.reset_index(drop=True)],
        axis=1,
    )


def label_columns_5d() -> list[str]:
    return list(LABEL_COLS_5D)
