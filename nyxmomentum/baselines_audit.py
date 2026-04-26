"""
Step 3 audit — skeptical checks on the baseline numbers before Step 4 (ML).

The rule that drives this module: "the baseline looked suspiciously good
→ confirm it, don't trust it." Six audits, each isolating one failure mode:

  A. Cost-adjusted table    — how much of the gross CAGR survives 20/40/60/100bps RT?
  B. Subperiod split        — is the whole story one bull leg (2022-2023) vs flat (2024-2026)?
  C. Breadth / concentration — are we really holding 20 names, or cycling 30 total?
  D. Decile monotonicity    — does the full cross-section stack, or is it just top-tail?
  E. Block contribution     — which block of the handcrafted score is actually driving?
  F. DQ cross-ref           — do the best names overlap suspicious-Low rows / outliers?

Outputs are plain dataframes + a print_* helper. The runner calls these and
serializes the artifacts. We never silently mask bad names — flags flow
through to the reports.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import (
    _HANDCRAFTED_WEIGHTS,
    _cs_zscore_per_date,
    classic_momentum_score,
    handcrafted_composite_score,
)


# ── A. Cost-adjusted summary ──────────────────────────────────────────────────

def cost_adjusted_summary(returns_by_variant: pd.DataFrame,
                          bps_round_trip_list: Iterable[float] = (0, 20, 40, 60, 100)
                          ) -> pd.DataFrame:
    """
    Monthly cost = turnover_fraction × (bps_rt / 10000). turnover_fraction is
    the share of names new this period → same share sold last leg → one
    round-trip on that fraction. Drag applied multiplicatively per period.

    Returns long-format:
      variant × bps_rt → CAGR_net, Sharpe_net, MaxDD_net, mean_monthly_net, avg_turnover
    """
    rows: list[dict] = []
    for var_name, g in returns_by_variant.groupby("variant", sort=False):
        r_gross = g["portfolio_return"].astype(float)
        turn = g["turnover_fraction"].astype(float)
        # First period has no prior basket → no turnover cost charged.
        turn_filled = turn.fillna(0.0)

        for bps in bps_round_trip_list:
            cost = turn_filled * (bps / 10000.0)
            r_net = r_gross - cost
            r_valid = r_net.dropna()
            n = len(r_valid)
            if n == 0:
                rows.append({"variant": var_name, "bps_round_trip": bps, "n_rebalances": 0})
                continue
            eq = (1.0 + r_valid).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1.0
            mu = float(r_valid.mean())
            sd = float(r_valid.std(ddof=0))
            rows.append({
                "variant": var_name,
                "bps_round_trip": int(bps),
                "n_rebalances": int(n),
                "cagr_net": float(eq.iloc[-1] ** (12 / n) - 1.0),
                "sharpe_annualized_net": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
                "max_drawdown_net": float(dd.min()),
                "mean_monthly_net": mu,
                "avg_turnover": float(turn.dropna().mean()),
                "annual_cost_drag_approx": float(turn_filled.mean() * (bps / 10000.0) * 12),
            })
    return pd.DataFrame(rows)


# ── B. Subperiod split + rolling window ───────────────────────────────────────

def subperiod_summary(returns_by_variant: pd.DataFrame,
                      split_date: str = "2024-01-01") -> pd.DataFrame:
    """Two windows: pre-split and post-split. Equal portfolio construction across."""
    split = pd.Timestamp(split_date)
    rows: list[dict] = []
    for var_name, g in returns_by_variant.groupby("variant", sort=False):
        g = g.sort_values("rebalance_date")
        for label, mask in (
            ("pre_" + split.strftime("%Y%m%d"),  g["rebalance_date"] < split),
            ("post_" + split.strftime("%Y%m%d"), g["rebalance_date"] >= split),
        ):
            sub = g.loc[mask]
            r = sub["portfolio_return"].dropna()
            n = len(r)
            if n == 0:
                rows.append({"variant": var_name, "period": label, "n_rebalances": 0})
                continue
            eq = (1.0 + r).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1.0
            mu = float(r.mean()); sd = float(r.std(ddof=0))
            rows.append({
                "variant": var_name,
                "period": label,
                "n_rebalances": int(n),
                "cagr": float(eq.iloc[-1] ** (12 / n) - 1.0),
                "sharpe_annualized": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
                "max_drawdown": float(dd.min()),
                "mean_monthly_return": mu,
                "std_monthly_return": sd,
                "hit_rate": float((r > 0).mean()),
                "mean_excess_xu100": float(sub["excess_vs_xu100"].dropna().mean()),
                "mean_excess_median": float(sub["excess_vs_median"].dropna().mean()),
            })
    return pd.DataFrame(rows)


def rolling_12m_return(returns_by_variant: pd.DataFrame) -> pd.DataFrame:
    """Rolling 12M geometric return by variant, for plotting / drift inspection."""
    frames: list[pd.DataFrame] = []
    for var_name, g in returns_by_variant.groupby("variant", sort=False):
        g = g.sort_values("rebalance_date").reset_index(drop=True)
        r = g["portfolio_return"].astype(float)
        # Rolling 12-period geometric return
        log_r = np.log1p(r)
        rolling_sum = log_r.rolling(12, min_periods=12).sum()
        rolling_ret = np.expm1(rolling_sum)
        frames.append(pd.DataFrame({
            "variant": var_name,
            "rebalance_date": g["rebalance_date"],
            "rolling_12m_return": rolling_ret,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── C. Breadth / concentration ────────────────────────────────────────────────

def concentration_report(portfolios: pd.DataFrame) -> pd.DataFrame:
    """Per variant: unique names, avg duration, top-10 frequent, Herfindahl index."""
    rows: list[dict] = []
    for var_name, g in portfolios.groupby("variant", sort=False):
        # Occurrences per ticker (number of rebalance dates holding that name)
        counts = g["ticker"].value_counts()
        total_slot_instances = int(len(g))  # sum of per-date basket sizes
        n_dates = int(g["rebalance_date"].nunique())
        unique = int(counts.shape[0])
        avg_duration = float(counts.mean())
        top10 = counts.head(10)
        top10_share = float(top10.sum() / total_slot_instances) if total_slot_instances else 0.0
        # Herfindahl on total-name-slot-share
        shares = counts / total_slot_instances
        hhi = float((shares ** 2).sum())
        rows.append({
            "variant": var_name,
            "n_dates": n_dates,
            "unique_tickers": unique,
            "mean_occurrences_per_ticker": avg_duration,
            "median_occurrences": float(counts.median()),
            "top10_share_of_slots": top10_share,
            "herfindahl_index": hhi,
            "top10_names": ", ".join(f"{t}({c})" for t, c in top10.items()),
        })
    return pd.DataFrame(rows)


# ── D. Decile monotonicity ────────────────────────────────────────────────────

def decile_performance(features: pd.DataFrame,
                       labels: pd.DataFrame,
                       score_fn,
                       score_name: str,
                       n_buckets: int = 10) -> pd.DataFrame:
    """
    Per rebalance_date, bucket eligible names into n_buckets by score (ascending).
    D1 = lowest score, D{n} = highest. Return mean forward return per decile.

    This tests whether the ranker orders the full cross-section or just
    sorts the top tail. Monotonic = stacking decile returns go up with rank.
    """
    feat = features.loc[features["eligible"].astype(bool)].copy()
    feat["_score"] = score_fn(feat).astype(float).values
    feat = feat.dropna(subset=["_score"])

    # bucket per date
    def _qcut(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < n_buckets:
            g = g.copy()
            g["_decile"] = np.nan
            return g
        # duplicates='drop' so ties don't explode
        g = g.copy()
        ranks = g["_score"].rank(method="first")
        g["_decile"] = pd.qcut(ranks, n_buckets, labels=False) + 1
        return g

    feat = feat.groupby("rebalance_date", group_keys=False, sort=False).apply(_qcut)

    merged = feat.merge(
        labels[["ticker", "rebalance_date", "l1_forward_return"]],
        on=["ticker", "rebalance_date"], how="left"
    )

    # Per-date per-decile mean; then aggregate across dates.
    per_date = (
        merged.dropna(subset=["_decile", "l1_forward_return"])
              .groupby(["rebalance_date", "_decile"], sort=False)["l1_forward_return"]
              .mean()
              .reset_index()
    )
    agg = per_date.groupby("_decile", sort=True)["l1_forward_return"].agg(
        mean_return="mean", std_return="std", hit_rate=lambda s: float((s > 0).mean()),
        n_dates="count",
    ).reset_index()
    agg["_decile"] = agg["_decile"].astype(int)
    agg["score_name"] = score_name
    agg["sharpe_annualized"] = agg.apply(
        lambda r: (r["mean_return"] / r["std_return"] * np.sqrt(12))
                  if r["std_return"] and r["std_return"] > 0 else 0.0,
        axis=1,
    )
    # Top-minus-bottom spread = D{n} - D1 per date
    td = per_date.pivot(index="rebalance_date", columns="_decile",
                        values="l1_forward_return")
    if n_buckets in td.columns and 1 in td.columns:
        tmb = (td[n_buckets] - td[1]).dropna()
        if len(tmb):
            mu = float(tmb.mean()); sd = float(tmb.std(ddof=0))
            row = pd.DataFrame([{
                "_decile": -1,  # sentinel for long-short
                "score_name": score_name,
                "mean_return": mu,
                "std_return": sd,
                "hit_rate": float((tmb > 0).mean()),
                "n_dates": int(len(tmb)),
                "sharpe_annualized": float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0,
            }])
            agg = pd.concat([agg, row], ignore_index=True)
    return agg


# ── E. Block contribution decomposition ───────────────────────────────────────

_HANDCRAFTED_BLOCK_OF: dict[str, str] = {
    "mom_252d_skip_21d":        "momentum",
    "mom_63d":                  "momentum",
    "rs_xu100_252d_skip_21d":   "relative_strength",
    "trend_r2_126d":            "trend_quality",
    "trend_above_ma200_pct_126d": "trend_quality",
    "vol_std_60d":              "volatility",
    "recent_extreme_21d":       "exhaustion",
    "px_over_ma50_zscore_20d":  "exhaustion",
}


def block_contribution_report(features: pd.DataFrame,
                              labels: pd.DataFrame,
                              n_buckets: int = 10) -> pd.DataFrame:
    """
    For each handcrafted-composite component feature:
      • weight
      • mean absolute per-date contribution (|w| × mean|z|) — normalized share
      • corr(feature z, full composite) — "how aligned is this component with
        the final score?" A block whose features barely correlate with the
        composite is being averaged out.
      • D10 - D1 spread using ONLY that feature as the ranker — single-feature
        ranking power, for attribution.
    """
    feat = features.loc[features["eligible"].astype(bool)].copy()

    # z-score each component per-date
    z_cols: dict[str, pd.Series] = {}
    for col in _HANDCRAFTED_WEIGHTS.keys():
        z_cols[col] = _cs_zscore_per_date(feat, col)

    # Full composite over this slice
    full = pd.Series(0.0, index=feat.index)
    for col, w in _HANDCRAFTED_WEIGHTS.items():
        full = full + w * z_cols[col]

    # Normalize absolute contributions
    abs_w = sum(abs(v) for v in _HANDCRAFTED_WEIGHTS.values())

    rows: list[dict] = []
    for col, w in _HANDCRAFTED_WEIGHTS.items():
        z = z_cols[col]
        contribution = (w * z).abs()
        share = float(abs(w) / abs_w)

        # Per-feature decile spread as single-feature ranker (directional)
        single = decile_performance(
            features=features, labels=labels,
            score_fn=lambda df, c=col: df[c].astype(float),
            score_name=col, n_buckets=n_buckets,
        )
        tmb_row = single.loc[single["_decile"] == -1]
        tmb_mu = float(tmb_row["mean_return"].iloc[0]) if len(tmb_row) else np.nan
        tmb_sh = float(tmb_row["sharpe_annualized"].iloc[0]) if len(tmb_row) else np.nan

        # Expected direction: sign of weight × (D10-D1)
        # If sign matches, single-feature ranking is aligned with the block's intent.
        rows.append({
            "feature": col,
            "block": _HANDCRAFTED_BLOCK_OF[col],
            "weight": w,
            "weight_share": share,
            "mean_abs_contribution": float(contribution.mean()),
            "corr_with_full_composite": float(z.corr(full)),
            "single_feature_D10_minus_D1_return": tmb_mu,
            "single_feature_D10_minus_D1_sharpe": tmb_sh,
            # Does ranking match the weight sign?
            "aligned_with_weight_sign": bool(
                (not pd.isna(tmb_mu)) and np.sign(tmb_mu) == np.sign(w)
            ),
        })
    return pd.DataFrame(rows)


# ── F. Suspicious-contribution audit ──────────────────────────────────────────

def top_contribution_audit(portfolios_for_variant: pd.DataFrame,
                           dq_flagged: pd.DataFrame | None = None,
                           n_top: int = 30) -> pd.DataFrame:
    """
    Take the top-N single-name contributions to a variant's portfolio return
    (weighted by position share within that rebalance's basket) and
    cross-reference with the DQ audit. If any of the biggest winners had a
    flagged Low print in the month before the rebalance or during the hold
    window, that is a red flag — not proof — to audit manually.
    """
    g = portfolios_for_variant.copy()
    # Equal-weight inside each rebalance basket
    counts = g.groupby("rebalance_date")["ticker"].transform("count")
    g["position_weight"] = 1.0 / counts
    g["contribution"] = g["position_weight"] * g["l1_forward_return"]
    top = g.sort_values("contribution", ascending=False).head(n_top).copy()

    # Cross-reference against DQ flags: within [rebalance_date - 30d, +35d]
    # catches setup-contaminating and hold-window anomalies.
    if dq_flagged is not None and not dq_flagged.empty:
        dq = dq_flagged.copy()
        dq["date"] = pd.to_datetime(dq["date"])
        flag_map: dict[tuple, list[str]] = {}
        for _, r in dq.iterrows():
            flag_map.setdefault(r["ticker"], []).append(pd.Timestamp(r["date"]))
        def _flag(row: pd.Series) -> str:
            events = flag_map.get(row["ticker"], [])
            if not events:
                return ""
            rd = pd.Timestamp(row["rebalance_date"])
            window_lo = rd - pd.Timedelta(days=30)
            window_hi = rd + pd.Timedelta(days=35)
            hits = [d for d in events if window_lo <= d <= window_hi]
            if not hits:
                return ""
            return ", ".join(d.strftime("%Y-%m-%d") for d in sorted(hits)[:3])
        top["dq_hit_dates"] = top.apply(_flag, axis=1)
        top["dq_flagged"] = top["dq_hit_dates"].ne("")
    else:
        top["dq_hit_dates"] = ""
        top["dq_flagged"] = False

    return top[[
        "rebalance_date", "ticker", "score", "rank_score",
        "position_weight", "l1_forward_return", "contribution",
        "dq_flagged", "dq_hit_dates",
    ]]
