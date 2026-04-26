"""
Ex-ante execution risk overlay — V1.

Applied identically to baseline and ML scores. Contract is deliberately
narrow: one knob (strength), one cap, one normalization path. The goal is
to downweight names with systematically poor execution proxies, NOT to
re-rank on a secret second target.

LOCKED CONTRACT (do not widen without an explicit note in the config):

  1. Cross-sectional percentile rank of each proxy PER rebalance_date.
     High proxy value (= worse execution friction) ⇒ high rank ⇒ MORE
     penalty after the weighted sum. Rank is in [0, 1].

  2. Missing proxy (NaN) ⇒ rank = 0.5 (median). "No signal" fallback.
     Never zero, never one — a missing proxy must not flip the direction
     of the penalty.

  3. raw_penalty = Σ_c (w_c × rank_c) / Σ_c |w_c|
     Weights are normalized by their absolute sum so raw_penalty ∈ [0, 1]
     regardless of how the caller scales individual weights.

  4. score_z = cross-sectional z-score of `score` per rebalance_date,
     with ddof=0 and a fallback to zeros if σ=0 on a given date.

  5. penalty_units = clip(raw_penalty − 0.5, −cap_sigma/2, cap_sigma/2) × 2
     (in [-1, 1] when uncapped; cap_sigma defines the cap in z-units).

  6. score_adj = score_z − strength × penalty_units

  7. ONLY columns listed as role='ex_ante' in EXECUTION_COLUMN_ROLES are
     permitted as weights. Realized (t+1) diagnostics will be rejected
     with a hard error.

REQUIRED INPUT SHAPE
  scores  : DataFrame with columns [ticker, rebalance_date, score]
  proxies : DataFrame with columns [ticker, rebalance_date, <proxy_cols…>]

OUTPUT COLUMNS
  ticker, rebalance_date, score,
  score_z, raw_penalty, penalty_units, penalty, score_adj,
  missing_proxy_count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXECUTION_COLUMN_ROLES


_ALLOWED_PROXY_COLS = frozenset(
    c for c, r in EXECUTION_COLUMN_ROLES.items() if r == "ex_ante"
)


def _validate_weights(weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError("overlay weights empty — pass at least one proxy")
    bad = [c for c in weights if c not in _ALLOWED_PROXY_COLS]
    if bad:
        raise ValueError(
            f"overlay weights include non-ex_ante columns {bad}. "
            f"Only EXECUTION_COLUMN_ROLES[role=='ex_ante'] are permitted: "
            f"{sorted(_ALLOWED_PROXY_COLS)}"
        )
    if any(not np.isfinite(w) for w in weights.values()):
        raise ValueError(f"overlay weights must be finite: {weights}")
    if sum(abs(w) for w in weights.values()) == 0.0:
        raise ValueError("overlay weights sum to zero in abs value")


def _cs_pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank in [0,1]. Uses 'average' ties. NaN stays NaN."""
    return s.rank(method="average", pct=True)


def _cs_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def apply_risk_overlay(
    scores: pd.DataFrame,
    proxies: pd.DataFrame,
    weights: dict[str, float],
    strength: float = 0.25,
    cap_sigma: float = 2.0,
) -> pd.DataFrame:
    """
    Apply ex-ante execution-risk overlay to `scores`. See module docstring
    for the contract.

    `strength` is in z-units of the raw score (σ of score_z = 1), so a
    strength of 0.25 means the worst-proxy stock is downweighted by up to
    ~0.25σ vs. the median-proxy stock. `cap_sigma` caps the penalty at
    ±cap_sigma/2 rank units before the ×2 spread (default 2.0 ⇒ uncapped
    at ±1.0 rank units, i.e. no effective cap for rank-based penalty).
    """
    if strength < 0:
        raise ValueError(f"strength must be ≥ 0, got {strength}")
    _validate_weights(weights)

    required = {"ticker", "rebalance_date", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {missing}")

    proxy_cols = list(weights.keys())
    keep = ["ticker", "rebalance_date", *proxy_cols]
    missing_p = [c for c in keep if c not in proxies.columns]
    if missing_p:
        raise ValueError(f"proxies missing required columns: {missing_p}")

    merged = scores.merge(
        proxies[keep],
        on=["ticker", "rebalance_date"],
        how="left",
        validate="one_to_one",
    )

    # Count missing proxies per row (before imputation) for diagnostics
    merged["missing_proxy_count"] = merged[proxy_cols].isna().sum(axis=1)

    # Cross-sectional pct rank per rebalance_date, then median-impute NaNs
    total_w = sum(abs(w) for w in weights.values())
    raw_penalty = pd.Series(0.0, index=merged.index)
    for col, w in weights.items():
        ranks = merged.groupby("rebalance_date", sort=False)[col].transform(_cs_pct_rank)
        ranks = ranks.fillna(0.5)       # missing → median rank
        raw_penalty = raw_penalty + (w / total_w) * ranks
    merged["raw_penalty"] = raw_penalty

    # z-score of raw score per rebalance_date (ddof=0, σ=0 → zeros)
    merged["score_z"] = merged.groupby("rebalance_date", sort=False)["score"].transform(_cs_zscore)

    # Centered rank → [-0.5, 0.5] → clip → ×2 → [-1, 1] units
    centered = merged["raw_penalty"] - 0.5
    half_cap = cap_sigma / 2.0
    centered_capped = centered.clip(-half_cap, half_cap)
    merged["penalty_units"] = centered_capped * 2.0
    merged["penalty"] = strength * merged["penalty_units"]
    merged["score_adj"] = merged["score_z"] - merged["penalty"]

    out_cols = [
        "ticker", "rebalance_date", "score",
        "score_z", "raw_penalty", "penalty_units", "penalty", "score_adj",
        "missing_proxy_count",
    ]
    return merged[out_cols]


# ── Self-check (unit tests without a framework) ───────────────────────────────

def _self_check_overlay(verbose: bool = False) -> None:
    """
    Deterministic checks for the overlay contract. Raises on any failure.
    Run as: python -m nyxmomentum.overlay
    """
    rng = np.random.default_rng(42)
    n_dates = 3
    n_stocks = 50
    dates = pd.to_datetime([f"2024-0{i}-28" for i in range(1, n_dates + 1)])

    rows = []
    for d in dates:
        for i in range(n_stocks):
            rows.append({
                "ticker": f"T{i:02d}",
                "rebalance_date": d,
                "score": rng.normal(0, 1),
                "proxy_limit_open_freq_60d":  rng.uniform(0, 0.3),
                "proxy_stale_open_freq_60d":  rng.uniform(0, 0.4),
                "proxy_open_dislocation_20d": rng.uniform(0, 2.0),
                "proxy_gap_dispersion_20d":   rng.uniform(0, 0.05),
            })
    full = pd.DataFrame(rows)
    scores = full[["ticker", "rebalance_date", "score"]].copy()
    proxies = full.drop(columns="score")

    weights = {
        "proxy_limit_open_freq_60d":  0.40,
        "proxy_stale_open_freq_60d":  0.30,
        "proxy_open_dislocation_20d": 0.20,
        "proxy_gap_dispersion_20d":   0.10,
    }

    # ── Check A: strength=0 ⇒ score_adj == score_z (no-op identity)
    out0 = apply_risk_overlay(scores, proxies, weights, strength=0.0)
    diff0 = (out0["score_adj"] - out0["score_z"]).abs().max()
    assert diff0 < 1e-12, f"idempotence at strength=0 broken: max_abs_diff={diff0}"

    # ── Check B: determinism — same inputs ⇒ same outputs
    out1 = apply_risk_overlay(scores, proxies, weights, strength=0.25)
    out2 = apply_risk_overlay(scores, proxies, weights, strength=0.25)
    assert np.allclose(out1["score_adj"].values, out2["score_adj"].values)

    # ── Check C: sign — higher proxy on a single ticker ⇒ lower score_adj
    # Construct a 2-ticker panel where ticker B has worse proxies than A.
    single = pd.DataFrame({
        "ticker": ["A", "B"],
        "rebalance_date": [dates[0]] * 2,
        "score": [1.0, 1.0],   # identical raw score
        "proxy_limit_open_freq_60d":  [0.01, 0.30],
        "proxy_stale_open_freq_60d":  [0.05, 0.40],
        "proxy_open_dislocation_20d": [0.10, 2.00],
        "proxy_gap_dispersion_20d":   [0.01, 0.05],
    })
    s_only = single[["ticker", "rebalance_date", "score"]]
    p_only = single.drop(columns="score")
    res = apply_risk_overlay(s_only, p_only, weights, strength=0.25)
    adj_A = res.loc[res.ticker == "A", "score_adj"].iloc[0]
    adj_B = res.loc[res.ticker == "B", "score_adj"].iloc[0]
    assert adj_A > adj_B, (
        f"sign check failed: cleaner proxies should yield higher score_adj "
        f"(A={adj_A}, B={adj_B})"
    )

    # ── Check D: missing proxy → median rank (no directional nudge)
    full_nan = full.copy()
    # Blank out all four proxies for ticker T00 on date[0]
    mask = (full_nan["ticker"] == "T00") & (full_nan["rebalance_date"] == dates[0])
    for c in weights:
        full_nan.loc[mask, c] = np.nan
    s_nan = full_nan[["ticker", "rebalance_date", "score"]]
    p_nan = full_nan.drop(columns="score")
    res_nan = apply_risk_overlay(s_nan, p_nan, weights, strength=0.25)
    t00_row = res_nan.loc[mask.values].iloc[0]
    # raw_penalty should be exactly 0.5 (median rank across all weights)
    assert abs(t00_row["raw_penalty"] - 0.5) < 1e-12, (
        f"missing-proxy → 0.5 median rank broken: raw_penalty={t00_row['raw_penalty']}"
    )
    # penalty magnitude should be 0 (centered rank = 0)
    assert abs(t00_row["penalty"]) < 1e-12

    # ── Check E: realized-column rejection
    try:
        apply_risk_overlay(
            scores, proxies,
            {"proxy_limit_open_freq_60d": 0.5, "realized_t1_traded": 0.5},
            strength=0.25,
        )
    except ValueError as e:
        assert "non-ex_ante" in str(e)
    else:
        raise AssertionError("overlay accepted a realized column — contract violated")

    # ── Check F: per-date penalty spans roughly [-strength, +strength]
    # (exact bounds depend on tie structure; check max-min spread)
    out_strong = apply_risk_overlay(scores, proxies, weights, strength=0.25)
    per_date_span = out_strong.groupby("rebalance_date")["penalty"].apply(
        lambda s: s.max() - s.min()
    )
    # span should be ≤ 2*strength + epsilon (upper bound on penalty range)
    assert per_date_span.max() <= 2 * 0.25 + 1e-9, (
        f"penalty span exceeds 2·strength: {per_date_span.max()}"
    )

    if verbose:
        print("[overlay] self-check PASSED")
        print(f"  strength=0 identity   : max |adj − z| = {diff0:.2e}")
        print(f"  determinism           : OK")
        print(f"  sign (A clean, B dirty): adj_A={adj_A:+.4f} > adj_B={adj_B:+.4f}")
        print(f"  missing → median rank : T00 raw_penalty = {t00_row['raw_penalty']:.4f}")
        print(f"  realized rejection    : OK")
        print(f"  per-date penalty span : max = {per_date_span.max():.4f} (≤ {2*0.25})")


if __name__ == "__main__":
    _self_check_overlay(verbose=True)
