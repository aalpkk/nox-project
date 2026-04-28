"""
Baseline portfolios — classic momentum, handcrafted composite, each evaluated
with and without the ex-ante execution overlay. No ML. No learned weights.

WHY THIS EXISTS (rule #6 from the locked Step-2 spec):
  Before any model is trained, we need to separate "does the overlay help?"
  from "does the feature set help?". If we skip straight to ML, those two
  confound — a gain could come from the model, from the features, or from
  the overlay being quietly load-bearing. Baselines lock the attribution.

FOUR VARIANTS:
  A. classic_no_overlay      — rank by mom_252d_skip_21d,                overlay OFF
  B. classic_with_overlay    — rank by mom_252d_skip_21d,                overlay ON
  C. handcrafted_no_overlay  — composite z-score across 6 blocks,        overlay OFF
  D. handcrafted_with_overlay — composite z-score across 6 blocks,       overlay ON

The handcrafted composite is a transparent, pre-registered weighted sum
of cross-sectionally z-scored features — no tuning. If the overlay lifts
it, that is evidence for the execution path, not the feature construction.

PORTFOLIO CONSTRUCTION (identical across variants):
  • eligible=True only (universe panel gate)
  • long-only, equal-weight, top_n per rebalance date
  • realized return = mean of l1_forward_return over selected names
  • no short, no rebalance cost model (baseline is PRE-COST by design)

DELIBERATELY OUT OF SCOPE:
  • winsorization / outlier clipping — distributional honesty first, clipping
    deferred to the dataset stage when we know where to clip
  • sector neutralization — sector block is deferred (spec §3)
  • transaction costs — baseline is pre-cost; costs land in Step 5
  • risk parity / vol targeting — identical sizing keeps attribution clean

OVERLAY:
  Applied identically across all variants via apply_risk_overlay(). With
  strength=0, overlay is a pure cross-sectional z-score (preserves rank
  order → same selection as raw-score ranking). We exploit that: the
  no-overlay variant sorts by raw score directly; the with-overlay variant
  sorts by score_adj.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .overlay import apply_risk_overlay


# ── Score functions ───────────────────────────────────────────────────────────

def classic_momentum_score(features: pd.DataFrame) -> pd.Series:
    """Raw 12-1 momentum (mom_252d_skip_21d). Textbook baseline."""
    return features["mom_252d_skip_21d"].astype(float)


# Handcrafted composite weights. Pre-registered, not tuned on the target.
# Intuition: longer-horizon momentum + RS pay off; vol hurts (low-vol
# anomaly on BIST matches global evidence); stretched names mean-revert.
#
# CHANGE LOG (2026-04-21):
#   Dropped `recent_extreme_21d` from the composite. Step 3 audit showed
#   single-feature D10-D1 Sharpe +0.06 (aligned=NO) — it does not rank
#   cross-sectionally, so carrying a −0.5 weight added noise. It stays in
#   the ML candidate feature set (models.py may discover a non-linear use
#   for it); it no longer participates in the handcrafted baseline.
_HANDCRAFTED_WEIGHTS: dict[str, float] = {
    # Momentum — 12-1 dominant, 3m for freshness
    "mom_252d_skip_21d": 1.0,
    "mom_63d":           0.5,
    # Relative strength vs XU100 — rewards beating the index, not market drift
    "rs_xu100_252d_skip_21d": 0.75,
    # Trend quality — smoothness premium (Step 3 audit: strongest single-feature
    # D10-D1 Sharpe at +1.46 / +1.19 for these two)
    "trend_r2_126d":              0.5,
    "trend_above_ma200_pct_126d": 0.5,
    # Volatility penalty — low-vol anomaly
    "vol_std_60d": -0.75,
    # Exhaustion penalty — px_over_ma50 z-score (directionally aligned)
    "px_over_ma50_zscore_20d": -0.25,
}


def _cs_zscore_per_date(features: pd.DataFrame, col: str) -> pd.Series:
    """Per-rebalance-date cross-sectional z-score of a feature column."""
    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0.0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / sd
    return features.groupby("rebalance_date", sort=False)[col].transform(_z).astype(float)


def handcrafted_composite_score(features: pd.DataFrame,
                                 weights: dict[str, float] | None = None) -> pd.Series:
    """
    Weighted sum of per-date cross-sectional z-scores of selected features.
    Balanced total weight (positive vs |negative|) keeps the composite
    direction economically motivated rather than numerically tilted.
    """
    w = weights or _HANDCRAFTED_WEIGHTS
    missing = [c for c in w if c not in features.columns]
    if missing:
        raise ValueError(f"handcrafted composite missing feature columns: {missing}")

    total = pd.Series(0.0, index=features.index)
    for col, wt in w.items():
        total = total + wt * _cs_zscore_per_date(features, col)
    return total


# ── Variant descriptor + evaluation ───────────────────────────────────────────

@dataclass(frozen=True)
class BaselineVariant:
    name: str
    description: str
    scorer: Callable[[pd.DataFrame], pd.Series]
    overlay_on: bool


def default_variants() -> tuple[BaselineVariant, ...]:
    return (
        BaselineVariant(
            name="classic_no_overlay",
            description="12-1 momentum (mom_252d_skip_21d), no overlay.",
            scorer=classic_momentum_score,
            overlay_on=False,
        ),
        BaselineVariant(
            name="classic_with_overlay",
            description="12-1 momentum with ex-ante execution overlay.",
            scorer=classic_momentum_score,
            overlay_on=True,
        ),
        BaselineVariant(
            name="handcrafted_no_overlay",
            description="Pre-registered handcrafted composite, no overlay.",
            scorer=handcrafted_composite_score,
            overlay_on=False,
        ),
        BaselineVariant(
            name="handcrafted_with_overlay",
            description="Handcrafted composite with ex-ante execution overlay.",
            scorer=handcrafted_composite_score,
            overlay_on=True,
        ),
    )


def _compute_scores(features: pd.DataFrame,
                    variant: BaselineVariant,
                    proxies: pd.DataFrame,
                    portfolio_cfg: PortfolioConfig) -> pd.DataFrame:
    """Return features rows keyed by (ticker, rebalance_date) with rank_score."""
    raw = variant.scorer(features)
    scored = features[["ticker", "rebalance_date"]].copy()
    scored["score"] = raw.values

    if not variant.overlay_on:
        scored["rank_score"] = scored["score"]
        return scored

    # Overlay path — join ex-ante proxies, penalize score.
    scored_nona = scored.dropna(subset=["score"]).copy()
    adj = apply_risk_overlay(
        scored_nona, proxies,
        weights=dict(portfolio_cfg.risk_overlay_weights),
        strength=portfolio_cfg.risk_overlay_strength,
    )
    out = scored.merge(
        adj[["ticker", "rebalance_date", "score_adj"]],
        on=["ticker", "rebalance_date"],
        how="left",
    )
    return out.rename(columns={"score_adj": "rank_score"})


def _select_top_n(scored: pd.DataFrame,
                  eligible_mask: pd.Series,
                  top_n: int) -> pd.DataFrame:
    """Per rebalance_date, top_n rows by rank_score desc among eligible names.
    Ties broken by ticker for determinism."""
    df = scored.loc[eligible_mask].copy()
    df = df.dropna(subset=["rank_score"])
    df = df.sort_values(["rebalance_date", "rank_score", "ticker"],
                        ascending=[True, False, True])
    return df.groupby("rebalance_date", sort=False).head(top_n)


def evaluate_variant(variant: BaselineVariant,
                     features: pd.DataFrame,
                     labels: pd.DataFrame,
                     proxies: pd.DataFrame,
                     portfolio_cfg: PortfolioConfig,
                     top_n: int = 20) -> dict:
    """Run one variant end-to-end. Returns portfolio / returns / summary."""
    need_feat = {"ticker", "rebalance_date", "eligible"}
    miss = need_feat - set(features.columns)
    if miss:
        raise ValueError(f"features missing required columns: {miss}")

    scored = _compute_scores(features, variant, proxies, portfolio_cfg)

    elig = features["eligible"].astype(bool)
    picks = _select_top_n(scored, elig, top_n)
    picks = picks[["ticker", "rebalance_date", "score", "rank_score"]]

    lab_cols = [
        "ticker", "rebalance_date",
        "l1_forward_return", "xu100_return_window", "universe_median_return",
    ]
    port = picks.merge(labels[lab_cols], on=["ticker", "rebalance_date"], how="left")

    rows: list[dict] = []
    prev_basket: set[str] | None = None
    for d, g in port.sort_values("rebalance_date").groupby("rebalance_date", sort=False):
        basket = set(g["ticker"])
        pw_ret = float(g["l1_forward_return"].mean()) if len(g) else np.nan
        xu = float(g["xu100_return_window"].iloc[0]) if len(g) else np.nan
        umed = float(g["universe_median_return"].iloc[0]) if len(g) else np.nan
        turnover = np.nan if prev_basket is None else (
            len(basket - prev_basket) / max(len(basket), 1)
        )
        rows.append({
            "rebalance_date": d,
            "n_names": len(g),
            "portfolio_return": pw_ret,
            "xu100_return": xu,
            "universe_median_return": umed,
            "excess_vs_xu100": pw_ret - xu if pd.notna(pw_ret) and pd.notna(xu) else np.nan,
            "excess_vs_median": pw_ret - umed if pd.notna(pw_ret) and pd.notna(umed) else np.nan,
            "turnover_fraction": turnover,
        })
        prev_basket = basket
    returns = pd.DataFrame(rows)

    r = returns["portfolio_return"].dropna()
    n = len(r)
    if n == 0:
        summary_row = {"variant": variant.name, "n_rebalances": 0}
    else:
        mu = float(r.mean())
        sd = float(r.std(ddof=0))
        sharpe = float(mu / sd * np.sqrt(12)) if sd > 0 else 0.0
        eq = (1.0 + r).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        summary_row = {
            "variant": variant.name,
            "n_rebalances": n,
            "mean_monthly_return": mu,
            "std_monthly_return": sd,
            "sharpe_annualized": sharpe,
            "cagr": float(eq.iloc[-1] ** (12 / n) - 1.0),
            "max_drawdown": float(dd.min()),
            "hit_rate": float((r > 0).mean()),
            "avg_turnover": float(returns["turnover_fraction"].dropna().mean())
                             if n > 1 else np.nan,
            "mean_excess_xu100":  float(returns["excess_vs_xu100"].dropna().mean()),
            "mean_excess_median": float(returns["excess_vs_median"].dropna().mean()),
        }

    return {
        "variant": variant.name,
        "description": variant.description,
        "portfolio": port,
        "returns": returns,
        "summary": summary_row,
    }


def evaluate_all(features: pd.DataFrame,
                 labels: pd.DataFrame,
                 proxies: pd.DataFrame,
                 portfolio_cfg: PortfolioConfig,
                 top_n: int = 20,
                 variants: tuple[BaselineVariant, ...] | None = None) -> dict:
    """Run every variant. Returns dict with portfolios, returns, summary, equity."""
    variants = variants or default_variants()
    results = [evaluate_variant(v, features, labels, proxies, portfolio_cfg, top_n)
               for v in variants]

    portfolios = pd.concat(
        [r["portfolio"].assign(variant=r["variant"]) for r in results],
        ignore_index=True,
    )
    returns = pd.concat(
        [r["returns"].assign(variant=r["variant"]) for r in results],
        ignore_index=True,
    )
    summary = pd.DataFrame([r["summary"] for r in results])

    eq_rows: list[pd.DataFrame] = []
    for r in results:
        rr = r["returns"][["rebalance_date", "portfolio_return"]].dropna()
        if rr.empty:
            continue
        eq = (1.0 + rr["portfolio_return"]).cumprod().reset_index(drop=True)
        eq_rows.append(pd.DataFrame({
            "variant": r["variant"],
            "rebalance_date": rr["rebalance_date"].values,
            "equity": eq.values,
        }))
    equity = pd.concat(eq_rows, ignore_index=True) if eq_rows else pd.DataFrame(
        columns=["variant", "rebalance_date", "equity"]
    )

    return {
        "portfolios": portfolios,
        "returns": returns,
        "summary": summary,
        "equity": equity,
    }
