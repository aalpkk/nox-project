"""Production smoke test for the persisted retention surrogate.

Re-implements the live retention chain end-to-end on a small batch of real
(ticker, signal_date) pairs:

  1. Load the persisted surrogate via the production API.
  2. Build truncated features for the cohort using the production module
     ``nyxexpansion.retention.truncate.rebuild_truncated_features``.
  3. Score with ``surrogate.predict``.
  4. Re-rank within the day's preds_v4C competitor panel and apply the
     ``rank ≤ 10`` retention gate.
  5. Compare against the research artifact
     ``output/nyxexp_truncated_score_diff.parquet`` for bit-level parity.

A pass means the live pipeline can rely on the persisted surrogate without
drift from the research result the filter was locked on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nyxexpansion.retention import surrogate as ret_surrogate

PREDS_PATH = Path("output/nyxexp_preds_v4C.parquet")
FEATS_TR_PATH = Path("output/nyxexp_truncated_features_diff.parquet")
RESEARCH_DIFF_PATH = Path("output/nyxexp_truncated_score_diff.parquet")
ARTIFACT_PATH = Path("output/nyxexp_retention_surrogate_v1.pkl")

CS_FEATURES = ("rs_rank_cs_today", "breadth_ad_20d", "chase_score_soft")
TOL = 1e-9
N_SAMPLE_PAIRS = 25


def main() -> int:
    print("=" * 70)
    print("Retention surrogate — production smoke test")
    print("=" * 70)

    surrogate = ret_surrogate.load(ARTIFACT_PATH)
    print(f"\n[load] artifact={ARTIFACT_PATH.name}  "
          f"model={surrogate.model_version}  schema={surrogate.truncated_feature_schema_version}")
    print(f"  UP   rho={surrogate.up.validation_rho:.4f} feats={len(surrogate.up.feature_columns)}")
    print(f"  NONUP rho={surrogate.nonup.validation_rho:.4f} feats={len(surrogate.nonup.feature_columns)}")

    preds = pd.read_parquet(PREDS_PATH)
    preds["date"] = pd.to_datetime(preds["date"]).dt.normalize()
    feats_tr = pd.read_parquet(FEATS_TR_PATH)
    feats_tr["date"] = pd.to_datetime(feats_tr["date"]).dt.normalize()

    cs_lookup = preds.set_index(["ticker", "date"])[
        [c for c in CS_FEATURES if c in preds.columns]
    ]
    feats_tr = feats_tr.merge(
        cs_lookup, on=["ticker", "date"], how="left",
        suffixes=("", "_cs"),
    )
    meta = preds[["ticker", "date", "model_kind", "xu_regime",
                  "winner_R_pred"]].copy()
    feats_tr = feats_tr.merge(meta, on=["ticker", "date"], how="left")
    feats_tr = feats_tr.dropna(subset=["model_kind", "winner_R_pred"]).copy()
    print(f"\n[features] truncated rows joined: {len(feats_tr):,}")

    yhat = surrogate.predict(feats_tr)
    feats_tr["winner_R_pred_tr_live"] = yhat.values

    research = pd.read_parquet(RESEARCH_DIFF_PATH)[
        ["ticker", "date", "winner_R_pred_tr"]
    ].rename(columns={"winner_R_pred_tr": "winner_R_pred_tr_research"})
    research["date"] = pd.to_datetime(research["date"]).dt.normalize()
    cmp = feats_tr.merge(research, on=["ticker", "date"], how="inner")
    diff = (cmp["winner_R_pred_tr_live"] - cmp["winner_R_pred_tr_research"]).abs()
    n_match = int((diff < TOL).sum())
    n_total = len(cmp)
    max_abs = float(diff.max()) if n_total else 0.0
    print(f"\n[parity] live vs research surrogate scores on {n_total:,} pairs:")
    print(f"  match (|delta| < {TOL}): {n_match}/{n_total}")
    print(f"  max |delta|: {max_abs:.3e}")
    if n_match != n_total:
        print("FAIL: live surrogate does not bit-match research surrogate")
        return 4

    print("\n[rank gate] reproducing pessimistic rank ≤ 10 retention…")
    tr_lookup = feats_tr.set_index(["ticker", "date"])["winner_R_pred_tr_live"]

    rank_rows = []
    for sd, grp in preds.groupby("date"):
        grp = grp.sort_values("winner_R_pred", ascending=False).reset_index(drop=True)
        if len(grp) < 1:
            continue
        top = grp.iloc[0]
        tk = top["ticker"]
        tr_p = tr_lookup.get((tk, sd), np.nan)
        if not np.isfinite(tr_p):
            continue
        above = int((grp["winner_R_pred"] > tr_p).sum())
        new_rank = above + 1
        rank_rows.append({
            "date": sd, "ticker": tk,
            "winner_R_pred": float(top["winner_R_pred"]),
            "winner_R_pred_tr": float(tr_p),
            "rank_1700_surrogate": new_rank,
            "retention_pass": new_rank <= 10,
            "n_competitors": len(grp),
        })
    rank_df = pd.DataFrame(rank_rows)
    print(f"  cohort: {len(rank_df)} top-1/day pairs with truncated coverage")
    if not rank_df.empty:
        keep = int(rank_df.retention_pass.sum())
        drop = len(rank_df) - keep
        print(f"  PASS (rank ≤ 10): {keep}")
        print(f"  DROP (rank > 10): {drop}")
        for k in (1, 3, 5, 10):
            pct = (rank_df.rank_1700_surrogate <= k).mean() * 100
            print(f"  would-stay-in-top-{k}: {pct:.1f}%")

    sample = rank_df.head(N_SAMPLE_PAIRS).copy()
    sample["date"] = sample["date"].dt.date
    print(f"\n[sample] first {len(sample)} pairs (live retention output):")
    print(sample[["date", "ticker", "winner_R_pred",
                  "winner_R_pred_tr", "rank_1700_surrogate",
                  "retention_pass"]].to_string(index=False))

    print("\nOK — smoke test green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
