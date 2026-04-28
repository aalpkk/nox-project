"""
Label family for cross-sectional momentum — V1.

TRAIN TARGET (single, locked):
  l2_excess_vs_universe_median

DIAGNOSTIC ONLY (computed but never fed to a learner):
  l1_forward_return, l3_outperform_binary, l4_quality_adjusted_return,
  l5_drawdown_aware_binary, forward_max_dd, forward_max_dd_intraperiod,
  xu100_excess_return

  ⚠ L5 (l5_drawdown_aware_binary) is explicitly NOT a selection rule.
  It consumes forward_max_dd which is future-observed — feeding L5 back
  into portfolio selection would leak the future. It exists purely as a
  reporting lens ("of our top-K picks, how many also had shallow DD?").
  See config.LABEL_COLUMN_ROLES for the machine-readable manifest.

CONTEXT:
  entry_date, exit_date, holding_days, universe_median_return

TIMING
  entry_date  = first trading day with df.index > rebalance_date[i]
  entry_price = Open of entry_date                                   (next_open)
  exit_date   = last trading day with df.index ≤ rebalance_date[i+1]
  exit_price  = Close of exit_date                                   (rebalance_close)
  If the ticker halts before rebalance_date[i+1], exit_date falls to the last
  available close on or before rebalance_date[i+1] and holding_days reflects
  reality. The last rebalance_date has NO forward label (no next period).

FORWARD DRAWDOWN — TWO VARIANTS
  Both emitted as non-positive floats (-0.12 = 12% peak-to-trough).

  forward_max_dd             CLOSE-ONLY. Tighter (less negative) bound.
    seq          = [entry_price, Close(entry_date), …, Close(exit_date)]
    running_peak = cummax(seq)[1:]          # peak carried FORWARD from entry
    dd[t]        = Close[t] / running_peak[t] − 1          for t ∈ window
    forward_max_dd = min(dd)

  forward_max_dd_intraperiod INTRAPERIOD. Honest lower bound using Low/High.
    running_peak[t] = max(entry_price, cummax(High[entry_date..t]))
    dd[t]           = Low[t] / running_peak[t] − 1          for t ∈ window
    forward_max_dd_intraperiod = min(dd)
    Notes:
      • Entry day is INCLUDED — an Open→Low flush on day 1 shows up.
      • Peak can refresh at intraday High, not just Close — catches
        cases where a name rips up then faceplants inside the same day.
      • Upper-bounded by forward_max_dd (strictly ≤, typically more negative).

UNIVERSE MEDIAN
  Computed over tickers flagged eligible at the rebalance_date whose L1 is
  finite. This keeps the label tied to a tradeable peer group, not a synthetic
  universe that includes names we would never actually buy.

LEAKAGE
  Nothing before entry_date leaks into the labels (they describe the future).
  Nothing after rebalance_date[i] leaks into features (separate module).
  The last rebalance row is dropped from the output, not emitted with NaNs —
  consumers joining on (ticker, rebalance_date) will see the absence cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig, LABEL_COLUMN_ROLES


# Columns produced by this module, in canonical order.
LABEL_COLUMNS = [
    "ticker",
    "rebalance_date",
    "entry_date",
    "exit_date",
    "holding_days",
    "l1_forward_return",
    "forward_max_dd",
    "forward_max_dd_intraperiod",
    "xu100_return_window",
    "xu100_excess_return",
    "universe_median_return",
    "l2_excess_vs_universe_median",
    "l3_outperform_binary",
    "l4_quality_adjusted_return",
    "l5_drawdown_aware_binary",
    "partial_holding",   # True if ticker halted/delisted before next_rd
]


def _forward_max_dd(closes: pd.Series, entry_price: float) -> float:
    """
    CLOSE-ONLY forward max drawdown.

    Running peak initialized at entry_price; thereafter updates only on Close
    values over [entry_date, exit_date]. Intraperiod Highs and Lows are
    IGNORED — a ticker that prints High=+20% then Close=+3% gives the same
    dd measurement as a ticker that simply closed +3% straight.

    Formula:
        seq[0]         = entry_price
        seq[1..N]      = Close(entry_date..exit_date)
        running_peak   = np.maximum.accumulate(seq)[1:]
        dd[t]          = Close[t] / running_peak[t] − 1
        return           min(dd)

    Returns a non-positive float (-0.12 = 12% close-to-close peak→trough).
    """
    if len(closes) == 0:
        return 0.0
    seq = np.concatenate([[float(entry_price)], closes.values.astype(float)])
    running_peak = np.maximum.accumulate(seq)[1:]
    dd = closes.values.astype(float) / running_peak - 1.0
    if not np.any(np.isfinite(dd)):
        return 0.0
    return float(np.nanmin(dd))


def _forward_max_dd_intraperiod(highs: pd.Series,
                                lows: pd.Series,
                                entry_price: float) -> float:
    """
    INTRAPERIOD forward max drawdown.

    Running peak initialized at entry_price and refreshed using daily Highs;
    drawdown measured against daily Lows. Entry day IS included so an
    Open→Low flush on day 1 is visible. Upper-bounded by _forward_max_dd
    (strictly ≤, typically more negative).

    Formula:
        peak_sequence  = concat([entry_price], High(entry_date..exit_date))
        running_peak   = cummax(peak_sequence)[1:]
        dd[t]          = Low[t] / running_peak[t] − 1
        return           min(dd)

    Returns a non-positive float.
    """
    if len(highs) == 0 or len(lows) == 0:
        return 0.0
    h = highs.values.astype(float)
    l = lows.values.astype(float)
    peak_seq = np.concatenate([[float(entry_price)], h])
    running_peak = np.maximum.accumulate(peak_seq)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = l / running_peak - 1.0
    if not np.any(np.isfinite(dd)):
        return 0.0
    return float(np.nanmin(dd))


def _locate_entry_exit(df: pd.DataFrame,
                       rebalance_date: pd.Timestamp,
                       next_rebalance_date: pd.Timestamp,
                       entry_mode: str) -> tuple | None:
    """
    Return (entry_date, entry_price, exit_date, exit_price, partial) or None
    if the trade window is not feasible.
    """
    if entry_mode == "next_open":
        future = df.loc[df.index > rebalance_date]
        if future.empty:
            return None
        entry_date = future.index[0]
        entry_price = float(future["Open"].iloc[0])
    elif entry_mode == "rebalance_close":
        past = df.loc[df.index <= rebalance_date]
        if past.empty:
            return None
        entry_date = past.index[-1]
        entry_price = float(past["Close"].iloc[-1])
    else:
        raise ValueError(f"Unknown entry_mode: {entry_mode}")

    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    window = df.loc[(df.index >= entry_date) & (df.index <= next_rebalance_date)]
    if window.empty:
        return None

    exit_date = window.index[-1]
    exit_price = float(window["Close"].iloc[-1])
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None

    partial = bool(exit_date < next_rebalance_date)
    return entry_date, entry_price, exit_date, exit_price, partial


def _xu100_window_return(xu100_close: pd.Series | None,
                         entry_date: pd.Timestamp,
                         exit_date: pd.Timestamp) -> float | None:
    """XU100 close-to-close return over the ticker's actual holding window."""
    if xu100_close is None or len(xu100_close) == 0:
        return None
    try:
        xu_entry = xu100_close.asof(entry_date)
        xu_exit = xu100_close.asof(exit_date)
    except Exception:
        return None
    if not (np.isfinite(xu_entry) and np.isfinite(xu_exit)) or xu_entry <= 0:
        return None
    return float(xu_exit / xu_entry - 1.0)


def compute_labels(panel: dict[str, pd.DataFrame],
                   rebalance_dates: pd.DatetimeIndex,
                   xu100: pd.DataFrame | None,
                   eligible_panel: pd.DataFrame,
                   config: LabelConfig | None = None) -> pd.DataFrame:
    """
    Build the long-format label frame. One row per (eligible ticker,
    non-terminal rebalance_date). The terminal rebalance (no forward period)
    is omitted — not NaN-filled — so downstream joins see its absence.
    """
    cfg = config or LabelConfig()
    reb = pd.DatetimeIndex(sorted(set(rebalance_dates)))
    if len(reb) < 2:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    xu100_close = xu100["Close"] if xu100 is not None and "Close" in xu100.columns else None

    # Restrict to eligible rows (small speed win)
    elig = eligible_panel[eligible_panel["eligible"]][["ticker", "rebalance_date"]]
    elig_by_date = dict(tuple(elig.groupby("rebalance_date")))

    rows: list[dict] = []
    for i in range(len(reb) - 1):
        rd = reb[i]
        next_rd = reb[i + 1]
        if rd not in elig_by_date:
            continue
        tickers_this_date = elig_by_date[rd]["ticker"].tolist()

        for ticker in tickers_this_date:
            df = panel.get(ticker)
            if df is None or df.empty:
                continue
            loc = _locate_entry_exit(df, rd, next_rd, cfg.entry_mode)
            if loc is None:
                continue
            entry_date, entry_price, exit_date, exit_price, partial = loc

            l1 = exit_price / entry_price - 1.0

            win_mask = (df.index >= entry_date) & (df.index <= exit_date)
            close_win = df.loc[win_mask, "Close"]
            fwd_dd = _forward_max_dd(close_win, entry_price)

            if "High" in df.columns and "Low" in df.columns:
                high_win = df.loc[win_mask, "High"]
                low_win = df.loc[win_mask, "Low"]
                fwd_dd_intra = _forward_max_dd_intraperiod(high_win, low_win, entry_price)
            else:
                fwd_dd_intra = None

            xu_ret = _xu100_window_return(xu100_close, entry_date, exit_date)
            xu_excess = (l1 - xu_ret) if xu_ret is not None else None

            holding_days = int((exit_date - entry_date).days)

            rows.append({
                "ticker": ticker,
                "rebalance_date": rd,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_days": holding_days,
                "l1_forward_return": float(l1),
                "forward_max_dd": float(fwd_dd),
                "forward_max_dd_intraperiod": (
                    float(fwd_dd_intra) if fwd_dd_intra is not None else None
                ),
                "xu100_return_window": xu_ret,
                "xu100_excess_return": xu_excess,
                "partial_holding": partial,
            })

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    # Cross-sectional step: universe_median per rebalance_date over rows with
    # finite L1. Eligibility is already enforced (we iterated eligible only).
    df_raw["universe_median_return"] = df_raw.groupby("rebalance_date")[
        "l1_forward_return"
    ].transform(lambda s: s[s.notna()].median())

    df_raw["l2_excess_vs_universe_median"] = (
        df_raw["l1_forward_return"] - df_raw["universe_median_return"]
    )

    # L3 outperform binary
    mode = cfg.outperform_mode
    if mode == "vs_universe_median":
        df_raw["l3_outperform_binary"] = (df_raw["l2_excess_vs_universe_median"] > 0).astype(float)
    elif mode == "vs_xu100":
        df_raw["l3_outperform_binary"] = (df_raw["xu100_excess_return"] > 0).astype(float)
    elif mode == "top_quintile":
        thr = cfg.top_quintile_threshold
        def _top(g):
            cut = g["l1_forward_return"].quantile(1 - thr)
            return (g["l1_forward_return"] > cut).astype(float)
        df_raw["l3_outperform_binary"] = df_raw.groupby(
            "rebalance_date", group_keys=False
        ).apply(_top, include_groups=False)
    else:
        raise ValueError(f"Unknown outperform_mode: {mode}")

    # L4 quality-adjusted (diagnostic). forward_max_dd is ≤ 0, so adding
    # λ*dd subtracts |dd| from the return.
    if cfg.quality_mode == "dd_penalty":
        df_raw["l4_quality_adjusted_return"] = (
            df_raw["l1_forward_return"] + cfg.quality_lambda * df_raw["forward_max_dd"]
        )
    elif cfg.quality_mode == "vol_normalized":
        # Proper vol-normalization requires per-window realized vol; deferred.
        df_raw["l4_quality_adjusted_return"] = df_raw["l1_forward_return"]
    else:
        raise ValueError(f"Unknown quality_mode: {cfg.quality_mode}")

    # L5 drawdown-aware binary: excess > threshold AND intra-period dd shallow.
    df_raw["l5_drawdown_aware_binary"] = (
        (df_raw["l2_excess_vs_universe_median"] > cfg.l5_excess_threshold)
        & (df_raw["forward_max_dd"] > -cfg.l5_max_dd_threshold)
    ).astype(float)

    return df_raw[LABEL_COLUMNS]


# ── Reporting ─────────────────────────────────────────────────────────────────

def label_distribution_report(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-rebalance + global distribution summary for every label column."""
    if labels.empty:
        return pd.DataFrame()

    numeric_cols = [
        "l1_forward_return", "l2_excess_vs_universe_median",
        "l3_outperform_binary", "l4_quality_adjusted_return",
        "l5_drawdown_aware_binary", "forward_max_dd",
        "xu100_excess_return", "universe_median_return",
    ]
    numeric_cols = [c for c in numeric_cols if c in labels.columns]

    def _summarize(g: pd.DataFrame) -> pd.Series:
        out = {}
        for c in numeric_cols:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(s) == 0:
                out[f"{c}_mean"] = None
                out[f"{c}_median"] = None
                continue
            out[f"{c}_mean"] = float(s.mean())
            out[f"{c}_median"] = float(s.median())
        out["n_rows"] = int(len(g))
        out["n_partial_holding"] = int(g["partial_holding"].sum()) if "partial_holding" in g else 0
        return pd.Series(out)

    per_date = labels.groupby("rebalance_date").apply(_summarize, include_groups=False).reset_index()
    return per_date


_ROLE_NOTES: dict[str, str] = {
    "l2_excess_vs_universe_median":
        "V1 LOCKED train target — cross-sectional excess vs. universe median. "
        "Single learner target.",
    "l1_forward_return":
        "Raw forward return. Diagnostic only — training on raw return leaks "
        "cross-sectional beta into selection.",
    "l3_outperform_binary":
        "Sign of L2. Diagnostic only — binarizing discards magnitude the learner "
        "should see.",
    "l4_quality_adjusted_return":
        "L1 + λ·forward_max_dd. Diagnostic only — λ dominates outcome, model "
        "would learn anti-vol confound instead of momentum edge.",
    "l5_drawdown_aware_binary":
        "⚠ FUTURE-DD FILTER. Diagnostic only. NEVER use in selection — "
        "doing so leaks the future (uses forward_max_dd).",
    "forward_max_dd":
        "Close-only forward drawdown. Diagnostic. Never feeds selection.",
    "forward_max_dd_intraperiod":
        "High/Low intraperiod forward drawdown (tighter than close-only). "
        "Diagnostic. Never feeds selection.",
    "xu100_excess_return":
        "Excess vs XU100 benchmark. Diagnostic — universe_median is V1's "
        "training benchmark, XU100 is for external reporting.",
    "universe_median_return":  "Per-rebalance reference used to build L2.",
    "entry_date":  "First trading day > rebalance_date; basis of Open-entry.",
    "exit_date":   "Last trading day ≤ next rebalance_date (earlier if halted).",
    "holding_days":"exit_date − entry_date, in calendar days.",
}


def label_role_manifest() -> pd.DataFrame:
    """Export column roles (train_target / diagnostic / context) for audit.

    Includes a `note` column so the rationale for each role is surfaced in
    reports, not buried in source comments. L5 is explicitly flagged as
    never-for-selection to prevent the future-DD leakage trap.
    """
    return pd.DataFrame([
        {
            "column": c,
            "role": r,
            "is_train_target": r == "train_target",
            "is_diagnostic": r == "diagnostic",
            "is_context": r == "context",
            "never_in_selection": c in {
                "l5_drawdown_aware_binary",
                "forward_max_dd",
                "forward_max_dd_intraperiod",
                "l4_quality_adjusted_return",
                "xu100_excess_return",
                "l1_forward_return",
                "l3_outperform_binary",
            },
            "note": _ROLE_NOTES.get(c, ""),
        }
        for c, r in LABEL_COLUMN_ROLES.items()
    ])


def l2_decomposition_report(labels: pd.DataFrame) -> dict:
    """
    Decompose the global L2 mean into several views to answer "is the +X%
    we see in global stats driven by big-universe dates, a few tickers, or
    a calendar period?"

      date_ew       mean of per-rebalance means of L2
      ticker_ew     mean of per-ticker means of L2
      row_ew        naive flat mean (what the global stats file shows)
      by_year       per-calendar-year mean/std/count (based on rebalance_date)
      size_corr     Spearman corr between per-date N_eligible and per-date L2 mean
                    — if strongly non-zero, universe size is dragging the mean
      date_mean_distribution   q05/q25/q50/q75/q95 of per-date L2 means

    Returns a JSON-ready dict.
    """
    if labels.empty or "l2_excess_vs_universe_median" not in labels.columns:
        return {}

    l2 = pd.to_numeric(labels["l2_excess_vs_universe_median"], errors="coerce")
    mask = l2.notna()
    lab = labels.loc[mask].copy()
    lab["l2"] = l2[mask]

    per_date = lab.groupby("rebalance_date")["l2"].agg(["mean", "size"]).rename(
        columns={"size": "n"}
    )
    per_ticker = lab.groupby("ticker")["l2"].mean()
    lab["year"] = pd.to_datetime(lab["rebalance_date"]).dt.year
    by_year = lab.groupby("year")["l2"].agg(["mean", "std", "count"])

    # Spearman corr between date N_eligible and date mean L2
    if len(per_date) >= 3:
        size_corr = float(per_date["n"].corr(per_date["mean"], method="spearman"))
    else:
        size_corr = None

    dist = per_date["mean"]
    date_mean_dist = {
        "q05": float(dist.quantile(0.05)),
        "q25": float(dist.quantile(0.25)),
        "q50": float(dist.quantile(0.50)),
        "q75": float(dist.quantile(0.75)),
        "q95": float(dist.quantile(0.95)),
    }

    out = {
        "row_ew":        float(lab["l2"].mean()),
        "date_ew":       float(per_date["mean"].mean()),
        "ticker_ew":     float(per_ticker.mean()),
        "n_rows":        int(len(lab)),
        "n_dates":       int(len(per_date)),
        "n_tickers":     int(per_ticker.shape[0]),
        "by_year": {
            int(y): {
                "mean":  float(r["mean"]),
                "std":   float(r["std"]),
                "count": int(r["count"]),
            }
            for y, r in by_year.iterrows()
        },
        "size_spearman_corr_mean_vs_n": size_corr,
        "date_mean_distribution": date_mean_dist,
    }
    return out
