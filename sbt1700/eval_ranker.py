"""Ranker evaluation + acceptance gate for SBT-1700 — STUB (PR-2).

Acceptance gate (locked):
  - mean Spearman rho ≥ 0.10
  - fold_rho_min ≥ 0
  - top-decile PF_net ≥ 1.5
  - top-decile avg_R_net > 0
  - permutation p(avg_R) ≤ 0.05
  - top-5 trades' total R must not exceed 50% of total realized R
    on a small-N cohort (concentration check)

If the gate fails, the live cron (run_scan_1700) MUST NOT be enabled.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "sbt1700.eval_ranker is a PR-2 stub. Wired after train_ranker.py."
    )


if __name__ == "__main__":
    raise SystemExit(main())
