"""
Turnover-dampening layer for prediction-to-portfolio conversion.

The Step 4 diagnostic: M1 (ridge) genuinely improves ordering over M0
(better L2 monotonicity, near-perfect DD monotonicity, D10-D9 collapse),
but pays for it with 76% monthly turnover vs M0's 46% — 60bps RT cost
drag burns the Sharpe advantage.

This module implements rule-based, ex-ante dampeners that convert a
prediction series into a selected basket while suppressing one-off
churn. No per-fold tuning; parameters are fixed up-front.

THREE DAMPENERS (independent, composable):

  1. score_smoothing
       score_t = α · raw_t + (1-α) · score_{t-1}    per ticker
     Reduces week-to-week rank jitter. α=1.0 means no smoothing.

  2. hysteresis_selection
       name is IN if its rank ≤ n_enter,
       name STAYS IN as long as its rank ≤ n_exit (n_exit ≥ n_enter).
       name is OUT otherwise.
     Rank = cross-sectional rank on the (possibly smoothed) score,
     descending (1 = highest score).

  3. weight_change_cap  — reserved for score-weighted portfolios. Skip in v1
     (we ship equal-weight top-N; weight change per name is 0 or 1/N).

Outputs are a per-rebalance picks DataFrame with realized turnover and
names-changed counters, consumable by evaluation.portfolio_cost_table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DampenerConfig:
    """
    All four knobs. Set alpha=1.0 to disable smoothing.
    Set n_exit=n_enter to disable hysteresis.
    """
    n_enter: int = 20
    n_exit:  int = 20
    smoothing_alpha: float = 1.0     # 1.0 = no smoothing
    # weight_cap reserved for score-weighting; unused in v1 equal-weight
    weight_cap: float | None = None


# ── Score smoothing ──────────────────────────────────────────────────────────

def apply_score_smoothing(preds: pd.DataFrame,
                          alpha: float,
                          score_col: str = "prediction",
                          out_col: str = "prediction_smoothed") -> pd.DataFrame:
    """
    Per-ticker exponential smoothing across rebalance_date.
    score_t = α · raw_t + (1-α) · score_{t-1} with score_0 = raw_0.
    Preserves row order of the input. Missing (ticker, date) rows are
    NOT gap-filled — smoothing only happens along observed rebalance
    dates for that ticker.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        out = preds.copy()
        out[out_col] = out[score_col].astype(float)
        return out

    out = preds.sort_values(["ticker", "rebalance_date"]).copy()
    # Exponentially weighted mean is the clean closed form:
    #   y_t = α · x_t + (1-α) · y_{t-1}   ↔   ewm(alpha=α, adjust=False)
    out[out_col] = out.groupby("ticker", sort=False)[score_col].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    )
    # Restore original row order
    out = out.sort_index()
    return out


# ── Hysteresis selection ─────────────────────────────────────────────────────

def _rank_desc_per_date(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Per rebalance_date, rank descending (1 = highest score). Ties: 'first'."""
    return df.groupby("rebalance_date", sort=False)[score_col].rank(
        method="first", ascending=False
    )


def hysteresis_selection(preds: pd.DataFrame,
                          n_enter: int,
                          n_exit: int,
                          score_col: str = "prediction",
                          eligible_only: bool = True) -> pd.DataFrame:
    """
    Walk forward through unique rebalance_date in chronological order. At each
    date, a ticker is selected iff:
      (a) it is eligible, AND
      (b) either it is in the current portfolio AND its rank ≤ n_exit,
          OR its rank ≤ n_enter.

    Returns a long-format selection table with columns:
      rebalance_date, ticker, selected (bool), rank, prev_selected (bool).

    n_exit ≥ n_enter is enforced. Equal values → strict top-N (no hysteresis).
    """
    if n_exit < n_enter:
        raise ValueError(f"n_exit ({n_exit}) must be ≥ n_enter ({n_enter})")

    m = preds.copy()
    if eligible_only and "eligible" in m.columns:
        m = m.loc[m["eligible"].astype(bool)].copy()
    m = m.dropna(subset=[score_col])
    m["rank"] = _rank_desc_per_date(m, score_col)
    m = m.sort_values(["rebalance_date", "rank"])

    rows: list[dict] = []
    portfolio: set[str] = set()
    for d, g in m.groupby("rebalance_date", sort=True):
        g = g.sort_values("rank")
        # Step 1: KEEPERS — currently-held names still within n_exit slack.
        # These get first priority — that is the whole point of hysteresis.
        keepers_df = g.loc[(g["ticker"].isin(portfolio)) & (g["rank"] <= n_exit)]
        if len(keepers_df) > n_enter:
            # Can happen if cross-section shrinks; keep the best-ranked ones.
            keepers_df = keepers_df.head(n_enter)
        keepers = set(keepers_df["ticker"])

        # Step 2: FILL — remaining slots go to top-ranked new names (rank ≤ n_enter)
        # that are not already keepers. If keepers already fill all N slots,
        # no new name enters even if it is ranked #1.
        slots_left = n_enter - len(keepers)
        if slots_left > 0:
            fresh = g.loc[
                (g["rank"] <= n_enter) & (~g["ticker"].isin(keepers))
            ].head(slots_left)
            new_portfolio = keepers | set(fresh["ticker"])
        else:
            new_portfolio = keepers

        for _, row in g.iterrows():
            rows.append({
                "rebalance_date": d,
                "ticker":         row["ticker"],
                "selected":       row["ticker"] in new_portfolio,
                "prev_selected":  row["ticker"] in portfolio,
                "rank":           int(row["rank"]) if pd.notna(row["rank"]) else -1,
                score_col:        row[score_col],
            })
        portfolio = new_portfolio

    return pd.DataFrame(rows)


# ── Portfolio construction from selection ────────────────────────────────────

def portfolio_from_selection(selection: pd.DataFrame,
                              labels: pd.DataFrame,
                              score_col: str = "prediction") -> pd.DataFrame:
    """
    Given a hysteresis selection table, compute the per-rebalance portfolio
    return (equal weight across the selected names), realized turnover
    (names changed vs previous period / basket size), and the raw
    names_changed count.

    Inputs:
      selection: output of hysteresis_selection — includes 'selected' boolean
                 and row per (ticker, rebalance_date) observation
      labels:    long-format label panel with l1_forward_return,
                 xu100_return_window, universe_median_return,
                 keyed on (ticker, rebalance_date)

    Output (one row per rebalance_date):
      rebalance_date, n_names, portfolio_return, xu100_return,
      universe_median_return, excess_vs_xu100, excess_vs_median,
      turnover_fraction, names_changed, prev_held.
    """
    picks = selection.loc[selection["selected"]].copy()
    lab = labels[["ticker", "rebalance_date",
                  "l1_forward_return", "xu100_return_window",
                  "universe_median_return"]]
    picks = picks.merge(lab, on=["ticker", "rebalance_date"], how="inner")

    rows: list[dict] = []
    prev: set[str] | None = None
    for d, g in picks.sort_values("rebalance_date").groupby("rebalance_date", sort=False):
        basket = set(g["ticker"])
        n = len(basket)
        if n == 0:
            # Flat period — no position, 0 return and no turnover recorded
            rows.append({
                "rebalance_date": d,
                "n_names": 0,
                "portfolio_return": 0.0,
                "xu100_return": float(g["xu100_return_window"].iloc[0]) if len(g) else np.nan,
                "universe_median_return": float(g["universe_median_return"].iloc[0]) if len(g) else np.nan,
                "excess_vs_xu100": np.nan,
                "excess_vs_median": np.nan,
                "turnover_fraction": np.nan if prev is None else (len(prev) / max(len(prev), 1)),
                "names_changed": 0 if prev is None else len(prev),
                "prev_held": 0 if prev is None else len(prev),
            })
            prev = set()
            continue

        pw = float(g["l1_forward_return"].mean())
        xu = float(g["xu100_return_window"].iloc[0])
        um = float(g["universe_median_return"].iloc[0])
        if prev is None:
            turn = np.nan
            changed = np.nan
        else:
            # Symmetric turnover: names leaving + names entering divided by basket size.
            # For equal-weight top-N where basket size is fixed, the two counts are
            # equal, so we report one-sided (names entering) / basket_size.
            changed = len(basket - prev)
            turn = changed / max(n, 1)
        rows.append({
            "rebalance_date": d,
            "n_names": n,
            "portfolio_return": pw,
            "xu100_return": xu,
            "universe_median_return": um,
            "excess_vs_xu100": pw - xu,
            "excess_vs_median": pw - um,
            "turnover_fraction": turn,
            "names_changed": changed if not isinstance(changed, float) else np.nan if np.isnan(changed) else int(changed),
            "prev_held": 0 if prev is None else len(prev & basket),
        })
        prev = basket

    return pd.DataFrame(rows)


# ── Apply the config end-to-end ──────────────────────────────────────────────

def apply_dampener(preds: pd.DataFrame,
                    labels: pd.DataFrame,
                    cfg: DampenerConfig,
                    score_col: str = "prediction") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: smoothing → ranking → hysteresis selection → portfolio.

    Returns:
      (selection_long, portfolio_per_date)
    """
    smoothed = apply_score_smoothing(
        preds, alpha=cfg.smoothing_alpha,
        score_col=score_col, out_col="prediction_smoothed",
    )
    use_col = "prediction_smoothed"
    selection = hysteresis_selection(
        smoothed, n_enter=cfg.n_enter, n_exit=cfg.n_exit,
        score_col=use_col,
    )
    portfolio = portfolio_from_selection(
        selection, labels, score_col=use_col,
    )
    return selection, portfolio
