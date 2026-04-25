"""LightGBM ranker training for SBT-1700 — STUB (PR-2).

Contract (PR-2):
  - Input: output/sbt_1700_dataset.parquet (built by build_dataset.py)
  - Target (primary): realized_R_net (regression)
  - Target (secondary, comparison only): win_label (binary)
  - Walk-forward 3-fold split by date, no same-(ticker, date) leakage
  - Permutation test (100 perms) on avg_R and PF_top_decile
  - Save per-fold models + canonical full-data model
  - Output: output/sbt1700_ranker_{seed}.txt + output/sbt1700_ranker_metrics.json
  - DO NOT consume any column from the legacy SBT ML / bucket system
  - DO NOT ensemble with old SBT ML
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "sbt1700.train_ranker is a PR-2 stub. "
        "Run sbt1700.build_dataset and sbt1700.validate_dataset first; "
        "open the next PR to wire LightGBM training."
    )


if __name__ == "__main__":
    raise SystemExit(main())
