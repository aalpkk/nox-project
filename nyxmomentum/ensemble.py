"""
Rank-blend ensemble of two prediction series (per-date cross-sectional).

Motivation from Step 4: M0 (handcrafted) has stronger economics/turnover,
M1 (ridge) has stronger ordering + DD-awareness. The natural combination is
a rank blend — average of percentile ranks is invariant to the underlying
prediction scales and robust to outliers in either model.

FORMULA (locked, V4 ensemble = rank_blend(M0, M1, 0.5, 0.5)):

  For each rebalance_date d:
    1. pct_rank_A[t, d] = rank(prediction_A[t, d], ascending=True, pct=True,
                               method="average")   over tickers at date d
    2. pct_rank_B[t, d] = rank(prediction_B[t, d], ascending=True, pct=True,
                               method="average")   over tickers at date d
    3. score_ensemble[t, d] = w_A / (w_A + w_B) * pct_rank_A[t, d]
                            + w_B / (w_A + w_B) * pct_rank_B[t, d]

  Ranks are PERCENTILES in [0, 1]. Ties resolved by pandas "average" method
  (fractional average of adjacent ranks). Missing predictions (NaN) yield
  NaN rank and are dropped downstream at selection time.

  Ensemble score is a weighted average of percentile ranks — NOT z-score,
  NOT raw-value sum. Percentile-rank blend is invariant to monotone
  transforms of the underlying predictions, so M0's composite score scale
  and M1's ridge-output scale cannot swamp one another.

  OVERLAY POLICY (Step 4/5 convention): risk overlay is NOT applied at
  ensemble time and NOT applied at selection time. All V0-V5 variants
  operate on raw cross-sectional scores. Overlay remains a separate
  downstream decision — if added later it must be applied identically to
  every variant to keep the comparison clean. (Step 3 evaluated overlay as
  its own attribution leg — the contribution was +2.4pp CAGR on classic,
  which is small relative to the 6-11pp differences we see between
  models.)

Interface is deliberately tiny: one function, weights as positional args.
"""
from __future__ import annotations

import pandas as pd


def rank_blend(preds_a: pd.DataFrame,
               preds_b: pd.DataFrame,
               weight_a: float = 0.5,
               weight_b: float = 0.5,
               name_a: str = "A",
               name_b: str = "B") -> pd.DataFrame:
    """
    Per-rebalance-date cross-sectional percentile rank of each model's
    prediction, then weighted sum. Inner-joined on (ticker, rebalance_date).

    Inputs: DataFrames with columns
      ticker, rebalance_date, eligible, prediction, fold_id
    (the long-format that run_m0/run_m1/run_m2 emits).

    Output columns:
      ticker, rebalance_date, eligible, fold_id, prediction,
      rank_{name_a}, rank_{name_b}
    where `prediction` is the ensemble score and ranks are per-date
    percentile (method='average', pct=True).

    `eligible` is logical-OR of both (eligible in either → counted);
    `fold_id` comes from preds_a for provenance.
    """
    if weight_a < 0 or weight_b < 0 or (weight_a + weight_b) == 0:
        raise ValueError(f"weights must be non-negative and not both zero; "
                         f"got ({weight_a}, {weight_b})")
    wa = weight_a / (weight_a + weight_b)
    wb = weight_b / (weight_a + weight_b)

    a = preds_a[["ticker", "rebalance_date", "eligible", "prediction", "fold_id"]].rename(
        columns={"prediction": f"prediction_{name_a}"}
    )
    b = preds_b[["ticker", "rebalance_date", "eligible", "prediction"]].rename(
        columns={"prediction": f"prediction_{name_b}",
                 "eligible":   f"eligible_{name_b}"}
    )
    m = a.merge(b, on=["ticker", "rebalance_date"], how="inner")
    m["eligible"] = m["eligible"].astype(bool) | m[f"eligible_{name_b}"].astype(bool)

    # Per-date percentile ranks. NaN prediction → NaN rank → excluded downstream.
    m[f"rank_{name_a}"] = m.groupby("rebalance_date", sort=False)[
        f"prediction_{name_a}"
    ].rank(pct=True, method="average")
    m[f"rank_{name_b}"] = m.groupby("rebalance_date", sort=False)[
        f"prediction_{name_b}"
    ].rank(pct=True, method="average")

    m["prediction"] = wa * m[f"rank_{name_a}"] + wb * m[f"rank_{name_b}"]

    return m[[
        "ticker", "rebalance_date", "eligible", "fold_id", "prediction",
        f"prediction_{name_a}", f"prediction_{name_b}",
        f"rank_{name_a}", f"rank_{name_b}",
    ]]
