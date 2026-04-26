"""
OOS evaluation for cross-sectional ranking models.

What we are testing (rule set for Step 4):
  1. Decile monotonicity of predictions vs THREE realized targets:
       • l2_excess_vs_universe_median   (what was trained)
       • l1_forward_return              (raw return, pre-benchmark)
       • forward_max_dd_intraperiod     (downside risk, never in selection)
     If the model learns L2 but deciles on raw return are flat, the lift is
     artefactual; if deciles on forward_max_dd are NOT decreasing with rank,
     the "winners" carry bigger downside → red flag.

  2. D10-D1 long-short (top minus bottom) Sharpe as the ranking-quality scalar.

  3. D9 vs D10 feature delta — the overextension audit. If D10 systematically
     has higher exhaustion / stretch / vol than D9, the model is repeating
     the handcrafted baseline's D10 reversal failure.

  4. Top-N portfolio CAGR / Sharpe at 0/20/40/60/100 bps round-trip, reusing
     baselines_audit.cost_adjusted_summary against OOS returns.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _qcut_per_date(df: pd.DataFrame,
                   rank_col: str,
                   n_buckets: int,
                   min_names: int | None = None) -> pd.Series:
    """Per rebalance_date, bucket rows into n_buckets by rank_col ascending.
    D1 = lowest, D{n} = highest. Ties broken by 'first'. NaN rank_col → NaN."""
    min_req = min_names or n_buckets

    def _q(g: pd.DataFrame) -> pd.Series:
        if g[rank_col].notna().sum() < min_req:
            return pd.Series(np.nan, index=g.index)
        r = g[rank_col].rank(method="first")
        return pd.Series(pd.qcut(r, n_buckets, labels=False) + 1, index=g.index)

    out = df.groupby("rebalance_date", group_keys=False, sort=False).apply(_q)
    return out.reindex(df.index)


def decile_table(preds: pd.DataFrame,
                 labels: pd.DataFrame,
                 target_col: str,
                 n_buckets: int = 10,
                 eligible_only: bool = True) -> pd.DataFrame:
    """One row per decile + a D{n}-D1 long-short row (sentinel _decile=-1)."""
    m = preds[["ticker", "rebalance_date", "eligible", "prediction"]].merge(
        labels[["ticker", "rebalance_date", target_col]],
        on=["ticker", "rebalance_date"], how="inner"
    )
    if eligible_only:
        m = m.loc[m["eligible"].astype(bool)]
    m = m.dropna(subset=["prediction", target_col])
    m["_decile"] = _qcut_per_date(m, "prediction", n_buckets)

    per_date = (
        m.dropna(subset=["_decile"])
         .groupby(["rebalance_date", "_decile"], sort=False)[target_col]
         .mean()
         .reset_index()
    )
    if per_date.empty:
        return pd.DataFrame()

    # Sharpe ann-12 on per-date decile returns — interpret as monthly.
    def _stats(g: pd.DataFrame) -> pd.Series:
        r = g[target_col]
        mu = float(r.mean())
        sd = float(r.std(ddof=0))
        return pd.Series({
            "mean": mu,
            "std": sd,
            "hit_rate": float((r > 0).mean()),
            "n_dates": int(len(r)),
            "sharpe_annualized": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
        })

    agg = per_date.groupby("_decile", sort=True).apply(
        _stats, include_groups=False
    ).reset_index()
    agg["_decile"] = agg["_decile"].astype(int)
    agg["target"] = target_col

    # D{n} - D1 long-short as a sentinel row
    piv = per_date.pivot(index="rebalance_date", columns="_decile", values=target_col)
    if n_buckets in piv.columns and 1 in piv.columns:
        tmb = (piv[n_buckets] - piv[1]).dropna()
        if len(tmb) > 0:
            mu = float(tmb.mean()); sd = float(tmb.std(ddof=0))
            agg = pd.concat([agg, pd.DataFrame([{
                "_decile": -1,
                "mean": mu,
                "std": sd,
                "hit_rate": float((tmb > 0).mean()),
                "n_dates": int(len(tmb)),
                "sharpe_annualized": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
                "target": target_col,
            }])], ignore_index=True)
    return agg


def monotonicity_score(decile_tbl: pd.DataFrame) -> dict:
    """Quantify how monotonically decile means rise. Spearman rank-corr
    between decile number and mean return; count of inversions."""
    d = decile_tbl.loc[decile_tbl["_decile"] > 0].sort_values("_decile")
    if d.empty:
        return {"spearman_monotonicity": None, "n_inversions": None,
                "d10_minus_d9": None}
    rho = float(d[["_decile", "mean"]].corr(method="spearman").iloc[0, 1])
    # Count adjacent inversions (mean_{i+1} < mean_i)
    means = d["mean"].values
    invs = int(((means[1:] - means[:-1]) < 0).sum())
    d9 = means[-2] if len(means) >= 2 else np.nan
    d10 = means[-1]
    return {
        "spearman_monotonicity": rho,
        "n_inversions": invs,
        "d10_minus_d9": float(d10 - d9) if pd.notna(d9) else None,
    }


def d9_vs_d10_feature_delta(preds: pd.DataFrame,
                             features: pd.DataFrame,
                             feat_cols: Iterable[str],
                             n_buckets: int = 10,
                             eligible_only: bool = True) -> pd.DataFrame:
    """
    For each (ticker, rebalance_date) in preds, bucket by prediction. Then
    for each feature, compute mean(D10) − mean(D9) across all test rebalances.

    Interpretation: if D10 has systematically higher exhaustion /
    dist_from_52w_high near 0 / vol_* / px_over_ma50_z than D9, the model
    is selecting stretched names into the very top bucket. This is the
    failure mode that produced the handcrafted baseline's D10 reversal.
    """
    feat_cols = list(feat_cols)
    m = preds[["ticker", "rebalance_date", "eligible", "prediction"]].merge(
        features[["ticker", "rebalance_date", *feat_cols]],
        on=["ticker", "rebalance_date"], how="inner"
    )
    if eligible_only:
        m = m.loc[m["eligible"].astype(bool)]
    m = m.dropna(subset=["prediction"])
    m["_decile"] = _qcut_per_date(m, "prediction", n_buckets)

    d9 = m.loc[m["_decile"] == n_buckets - 1]
    d10 = m.loc[m["_decile"] == n_buckets]
    rows: list[dict] = []
    for c in feat_cols:
        mu9 = float(d9[c].mean()); mu10 = float(d10[c].mean())
        std9 = float(d9[c].std(ddof=0)); std10 = float(d10[c].std(ddof=0))
        rows.append({
            "feature": c,
            "mean_d9": mu9,
            "mean_d10": mu10,
            "delta_d10_minus_d9": mu10 - mu9,
            "std_d9": std9,
            "std_d10": std10,
            "n_d9": int(len(d9.dropna(subset=[c]))),
            "n_d10": int(len(d10.dropna(subset=[c]))),
        })
    return pd.DataFrame(rows).sort_values(
        "delta_d10_minus_d9", key=lambda s: s.abs(), ascending=False,
    )


def top_n_portfolio_returns(preds: pd.DataFrame,
                            labels: pd.DataFrame,
                            top_n: int = 20,
                            eligible_only: bool = True) -> pd.DataFrame:
    """
    Per rebalance_date, pick top_n by prediction among eligible names, equal
    weight, realized return = mean of l1_forward_return. Also computes a
    per-period turnover_fraction for cost application downstream.
    """
    m = preds[["ticker", "rebalance_date", "eligible", "prediction", "fold_id"]].merge(
        labels[["ticker", "rebalance_date",
                "l1_forward_return", "xu100_return_window",
                "universe_median_return"]],
        on=["ticker", "rebalance_date"], how="inner"
    )
    if eligible_only:
        m = m.loc[m["eligible"].astype(bool)]
    m = m.dropna(subset=["prediction"])
    m = m.sort_values(["rebalance_date", "prediction", "ticker"],
                      ascending=[True, False, True])
    picks = m.groupby("rebalance_date", sort=False).head(top_n)

    rows: list[dict] = []
    prev: set[str] | None = None
    for d, g in picks.sort_values("rebalance_date").groupby("rebalance_date", sort=False):
        basket = set(g["ticker"])
        pw = float(g["l1_forward_return"].mean())
        xu = float(g["xu100_return_window"].iloc[0])
        um = float(g["universe_median_return"].iloc[0])
        turn = np.nan if prev is None else len(basket - prev) / max(len(basket), 1)
        rows.append({
            "rebalance_date": d,
            "fold_id": g["fold_id"].iloc[0],
            "n_names": len(g),
            "portfolio_return": pw,
            "xu100_return": xu,
            "universe_median_return": um,
            "excess_vs_xu100": pw - xu,
            "excess_vs_median": pw - um,
            "turnover_fraction": turn,
        })
        prev = basket
    return pd.DataFrame(rows)


def holding_persistence(selection_or_picks: pd.DataFrame,
                         *,
                         rebalances_per_year: int = 12) -> dict:
    """
    Compute holding-persistence metrics from either:
      (a) a selection table (dampener output) — has 'selected' bool column
      (b) a picks table (explicit per-date holdings) — has no 'selected' col,
          every row is a held slot.

    Returns a dict with:
      n_unique_tickers           — distinct tickers ever held
      n_holding_spans            — total contiguous runs (ticker enters then
                                   leaves)
      mean_span_rebalances       — average contiguous run length
      median_span_rebalances     — median
      median_span_months         — median × (12 / rebalances_per_year)
      p25_span / p75_span        — IQR markers
      re_entry_rate              — fraction of unique tickers that were held
                                   more than once
    A "span" is consecutive rebalance dates (per ticker) where the ticker is
    present. Gaps break the span.
    """
    if selection_or_picks is None or selection_or_picks.empty:
        return {"n_unique_tickers": 0, "n_holding_spans": 0}

    if "selected" in selection_or_picks.columns:
        held = selection_or_picks.loc[selection_or_picks["selected"].astype(bool)]
    else:
        held = selection_or_picks

    if held.empty:
        return {"n_unique_tickers": 0, "n_holding_spans": 0}

    # Map each distinct rebalance_date to an ordinal index
    dates = sorted(held["rebalance_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    held = held.assign(_idx=held["rebalance_date"].map(date_to_idx))

    # Compute contiguous runs per ticker
    spans: list[int] = []
    n_per_ticker: dict[str, int] = {}
    for tkr, g in held.sort_values(["ticker", "_idx"]).groupby("ticker", sort=False):
        idxs = g["_idx"].tolist()
        n_per_ticker[tkr] = n_per_ticker.get(tkr, 0) + len(idxs)
        run = 1
        for a, b in zip(idxs[:-1], idxs[1:]):
            if b == a + 1:
                run += 1
            else:
                spans.append(run)
                run = 1
        spans.append(run)

    import numpy as np
    s = np.asarray(spans, dtype=float)
    factor = 12.0 / float(rebalances_per_year) if rebalances_per_year else 1.0
    return {
        "n_unique_tickers":       int(len(n_per_ticker)),
        "n_holding_spans":        int(len(s)),
        "mean_span_rebalances":   float(s.mean()) if len(s) else 0.0,
        "median_span_rebalances": float(np.median(s)) if len(s) else 0.0,
        "median_span_months":     float(np.median(s) * factor) if len(s) else 0.0,
        "p25_span_rebalances":    float(np.percentile(s, 25)) if len(s) else 0.0,
        "p75_span_rebalances":    float(np.percentile(s, 75)) if len(s) else 0.0,
        "re_entry_rate":          float(sum(
            1 for t, n in n_per_ticker.items() if n > 1
        ) / max(len(n_per_ticker), 1)),
    }


def portfolio_cost_table(returns: pd.DataFrame,
                          bps_list: Iterable[float] = (0, 20, 40, 60, 100)) -> pd.DataFrame:
    """Mirror of baselines_audit.cost_adjusted_summary but for ONE variant."""
    rows: list[dict] = []
    r_gross = returns["portfolio_return"].astype(float)
    turn = returns["turnover_fraction"].fillna(0.0).astype(float)
    for bps in bps_list:
        cost = turn * (bps / 10000.0)
        r_net = (r_gross - cost).dropna()
        n = len(r_net)
        if n == 0:
            rows.append({"bps_round_trip": int(bps), "n_rebalances": 0})
            continue
        eq = (1.0 + r_net).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        mu = float(r_net.mean()); sd = float(r_net.std(ddof=0))
        rows.append({
            "bps_round_trip": int(bps),
            "n_rebalances": int(n),
            "cagr_net": float(eq.iloc[-1] ** (12 / n) - 1.0),
            "sharpe_annualized_net": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
            "max_drawdown_net": float(dd.min()),
            "mean_monthly_net": mu,
            "avg_turnover": float(returns["turnover_fraction"].dropna().mean()),
        })
    return pd.DataFrame(rows)


def model_summary(model_name: str,
                   preds: pd.DataFrame,
                   labels: pd.DataFrame,
                   features: pd.DataFrame,
                   target_cols: tuple[str, ...] = (
                       "l2_excess_vs_universe_median",
                       "l1_forward_return",
                       "forward_max_dd_intraperiod",
                   ),
                   top_n: int = 20,
                   n_buckets: int = 10) -> dict:
    """Bundle of reports for one model."""
    decile_tables: dict[str, pd.DataFrame] = {}
    mono: dict[str, dict] = {}
    for tgt in target_cols:
        d = decile_table(preds, labels, target_col=tgt, n_buckets=n_buckets)
        decile_tables[tgt] = d
        mono[tgt] = monotonicity_score(d)

    feat_cols = [c for c in features.columns
                 if c not in {"ticker", "rebalance_date", "eligible"}]
    d9d10 = d9_vs_d10_feature_delta(
        preds, features, feat_cols=feat_cols,
        n_buckets=n_buckets,
    ) if (not features.empty and feat_cols) else pd.DataFrame()
    returns = top_n_portfolio_returns(preds, labels, top_n=top_n)
    cost = portfolio_cost_table(returns)

    return {
        "model": model_name,
        "deciles": decile_tables,
        "monotonicity": mono,
        "d9_vs_d10": d9d10,
        "returns": returns,
        "cost_table": cost,
    }
