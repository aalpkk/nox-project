"""17:00-aware feature engineering for SBT-1700.

Strict information-set discipline:
  - Prior daily features use rows up to and including T-1 (EOD complete).
  - T's row is the 17:00-truncated bar; only ``Open``, ``High``, ``Low``,
    ``Close``, ``Volume`` from the 16:45 cutoff aggregation may be read.
  - No feature touches T+1 or later.
  - No feature reads the EOD close of T.
  - Old SBT ML / bucket scores are NOT consumed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbt1700.config import (
    EXPECTED_BARS_PER_PAIR,
    PRIOR_EMA_SPANS,
    PRIOR_ATR_WINDOW,
    PRIOR_BB_WINDOW,
    PRIOR_RETURNS_WINDOWS,
    PRIOR_VOL_WINDOW,
    PRIOR_52W_WINDOW,
)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _prior_panel(
    daily: pd.DataFrame,
    candidate_dates: set,
) -> pd.DataFrame:
    """Compute prior-only daily indicators for one ticker.

    Returns a DataFrame indexed by Date with prior-only columns. We then
    look up T-1 values when emitting features for T.
    """
    df = daily.sort_index().copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    v = df["Volume"]

    out = pd.DataFrame(index=df.index)
    for span in PRIOR_EMA_SPANS:
        out[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()

    out["atr14_prior"] = _atr(h, l, c, PRIOR_ATR_WINDOW)
    sma = c.rolling(PRIOR_BB_WINDOW).mean()
    std = c.rolling(PRIOR_BB_WINDOW).std()
    bb_upper = sma + 2.0 * std
    bb_lower = sma - 2.0 * std
    out["bb_width_prior"] = (bb_upper - bb_lower) / sma
    out["bb_width_sma_prior"] = out["bb_width_prior"].rolling(PRIOR_BB_WINDOW).mean()

    for w in PRIOR_RETURNS_WINDOWS:
        out[f"ret_{w}d_prior"] = c.pct_change(w)

    out["vol_sma_prior"] = v.rolling(PRIOR_VOL_WINDOW).mean()
    out["dollar_vol_sma_prior"] = (c * v).rolling(PRIOR_VOL_WINDOW).mean()

    out["high_52w_prior"] = h.rolling(PRIOR_52W_WINDOW).max()
    out["low_52w_prior"] = l.rolling(PRIOR_52W_WINDOW).min()

    # Realized vol on prior closes
    out["realized_vol_20d_prior"] = c.pct_change().rolling(20).std()

    return out


def build_features_for_ticker(
    daily: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Emit one feature row per candidate (ticker, date).

    Args:
        daily: per-ticker daily OHLCV indexed by Date, ascending.
        candidates: per-ticker rows from signals.detect_candidates_for_ticker
            (one row per (date) for this ticker).

    All prior-day reads index by T-1 (`prev_date`). T's information is
    only the 17:00 fields already attached to ``candidates``.
    """
    if daily.empty or candidates.empty:
        return pd.DataFrame()

    daily = daily.sort_index()
    cand_dates = set(pd.to_datetime(candidates["date"]).tolist())
    prior = _prior_panel(daily, cand_dates)

    rows: list[dict] = []
    for r in candidates.itertuples(index=False):
        T = pd.Timestamp(r.date).normalize()
        if T not in daily.index:
            continue
        idx_pos = daily.index.get_loc(T)
        if idx_pos == 0:
            continue
        Tm1 = daily.index[idx_pos - 1]
        if Tm1 not in prior.index:
            continue

        p = prior.loc[Tm1]
        ema20 = float(p["ema20"])
        ema50 = float(p["ema50"])
        ema100 = float(p["ema100"])
        ema200 = float(p["ema200"])
        atr_prior = float(p["atr14_prior"])
        bbw_prior = float(p["bb_width_prior"])
        bbw_sma_prior = float(p["bb_width_sma_prior"])
        vol_sma_prior = float(p["vol_sma_prior"])
        dollar_vol_sma_prior = float(p["dollar_vol_sma_prior"])
        hi52 = float(p["high_52w_prior"])
        lo52 = float(p["low_52w_prior"])
        rv20 = float(p["realized_vol_20d_prior"])

        # T's 17:00 fields (already coverage-checked upstream)
        o1700 = float(r.open_1700)
        h1700 = float(r.high_1700)
        l1700 = float(r.low_1700)
        c1700 = float(r.close_1700)
        v1700 = float(r.vol_1700)
        n_bars = int(r.n_bars_1700)
        coverage = float(r.intraday_coverage)

        if not (np.isfinite(c1700) and atr_prior > 0):
            continue

        # Distance / position
        feat = {
            "ticker": r.ticker if hasattr(r, "ticker") else daily.attrs.get("ticker", ""),
            "date": T,
            # Prior-only references (ticker provenance; no leak)
            "ema20_prior": ema20,
            "ema50_prior": ema50,
            "ema100_prior": ema100,
            "ema200_prior": ema200,
            "atr14_prior": atr_prior,
            "bb_width_prior": bbw_prior,
            "bb_width_sma_prior": bbw_sma_prior,
            "vol_sma20_prior": vol_sma_prior,
            "dollar_vol_sma20_prior": dollar_vol_sma_prior,
            "high_52w_prior": hi52,
            "low_52w_prior": lo52,
            "realized_vol_20d_prior": rv20,
            # SBT signal state
            "box_top": float(r.box_top),
            "box_bottom": float(r.box_bottom),
            "box_height_atr": (float(r.box_top) - float(r.box_bottom)) / atr_prior,
            "squeeze_run_prior": int(r.squeeze_run_prior),
            # T's 17:00 raw
            "open_1700": o1700,
            "high_1700": h1700,
            "low_1700": l1700,
            "close_1700": c1700,
            "vol_1700": v1700,
            "n_bars_1700": n_bars,
            "intraday_coverage": coverage,
            # Distance features (prior-only denominators)
            "close_vs_box_top_pct": (c1700 - float(r.box_top)) / float(r.box_top),
            "close_above_box_atr": (c1700 - float(r.box_top)) / atr_prior,
            "close_vs_ema20_pct": _safe_pct(c1700, ema20),
            "close_vs_ema50_pct": _safe_pct(c1700, ema50),
            "close_vs_ema100_pct": _safe_pct(c1700, ema100),
            "close_vs_ema200_pct": _safe_pct(c1700, ema200),
            "ema20_vs_ema50_pct": _safe_pct(ema20, ema50),
            "ema50_vs_ema200_pct": _safe_pct(ema50, ema200),
            "dist_to_52w_high_pct": _safe_pct(c1700, hi52),
            "dist_to_52w_low_pct": _safe_pct(c1700, lo52),
            # Intraday geometry @ 17:00
            "intraday_return_pct": _safe_pct(c1700, o1700),
            "intraday_range_atr": (h1700 - l1700) / atr_prior,
            "intraday_body_atr": (c1700 - o1700) / atr_prior,
            "close_loc_intraday": (c1700 - l1700) / max(h1700 - l1700, 1e-9),
            "high_above_box_atr": (h1700 - float(r.box_top)) / atr_prior,
            # Volume
            "vol_pace_ratio": _vol_pace(v1700, vol_sma_prior, n_bars),
            "dollar_vol_1700": c1700 * v1700,
            "dollar_vol_ratio_pace": _dollar_vol_pace(
                c1700, v1700, dollar_vol_sma_prior, n_bars,
            ),
            # Prior returns context
            "ret_1d_prior": float(p["ret_1d_prior"]),
            "ret_5d_prior": float(p["ret_5d_prior"]),
            "ret_10d_prior": float(p["ret_10d_prior"]),
            "ret_20d_prior": float(p["ret_20d_prior"]),
            # Coverage diagnostics (kept as feature so the ranker can
            # learn to discount partial sessions if it wants)
            "missing_bar_count": EXPECTED_BARS_PER_PAIR - n_bars,
        }
        rows.append(feat)

    return pd.DataFrame(rows)


def _safe_pct(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or denom == 0:
        return np.nan
    return (num - denom) / denom


def _vol_pace(vol_1700: float, vol_sma_prior: float, n_bars: int) -> float:
    if vol_sma_prior <= 0 or n_bars <= 0:
        return np.nan
    elapsed = n_bars / EXPECTED_BARS_PER_PAIR
    if elapsed <= 0:
        return np.nan
    return vol_1700 / (vol_sma_prior * elapsed)


def _dollar_vol_pace(
    close_1700: float,
    vol_1700: float,
    dollar_vol_sma_prior: float,
    n_bars: int,
) -> float:
    if dollar_vol_sma_prior <= 0 or n_bars <= 0:
        return np.nan
    elapsed = n_bars / EXPECTED_BARS_PER_PAIR
    if elapsed <= 0:
        return np.nan
    return (close_1700 * vol_1700) / (dollar_vol_sma_prior * elapsed)


def build_features(
    daily_master: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Whole-panel feature build."""
    if daily_master.empty or candidates.empty:
        return pd.DataFrame()

    out_chunks: list[pd.DataFrame] = []
    for tk, c_chunk in candidates.groupby("ticker"):
        sub = daily_master[daily_master["ticker"] == tk]
        if sub.empty:
            continue
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        sub.attrs["ticker"] = tk
        feats = build_features_for_ticker(sub, c_chunk)
        if not feats.empty:
            out_chunks.append(feats)

    if not out_chunks:
        return pd.DataFrame()
    return pd.concat(out_chunks, ignore_index=True)
