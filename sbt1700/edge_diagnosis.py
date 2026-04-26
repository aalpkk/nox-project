"""Synthesize SBT-1700 edge diagnosis report.

Combines raw-cohort exit-matrix metrics with per-exit ranker fold results
into a single markdown verdict per exit variant. The classification used
is the one stated in PR-2:

    1. raw cohort profitable           → unconditional edge under <variant>
    2. raw bad / ranker top-decile good → conditional/rankable edge
    3. all bad                          → not tradable as coded

This is diagnosis, not deployment. No live wiring is enabled by this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Thresholds — deliberately permissive for diagnosis (not a gate).
RAW_PROFITABLE_PF = 1.20
RAW_PROFITABLE_AVG_R = 0.05
RANKER_GOOD_PF = 1.50
RANKER_GOOD_AVG_R = 0.10
MIN_FOLDS_RANKER_OK = 2  # of 3 walk-forward folds


def _classify(raw: dict, ranker_folds: pd.DataFrame) -> tuple[str, str]:
    raw_n = int(raw.get("N", 0))
    if raw_n == 0:
        return "no_data", "No labeled rows for this variant."
    raw_pf = raw.get("PF_net", float("nan"))
    raw_avg = raw.get("avg_R_net", float("nan"))
    raw_ok = (raw_avg is not None and raw_avg > RAW_PROFITABLE_AVG_R
              and raw_pf is not None and raw_pf > RAW_PROFITABLE_PF)

    wf = ranker_folds[ranker_folds["fold"].str.startswith("fold_")]
    folds_with_top = wf.dropna(subset=["top_decile_avg_R"])
    n_pos = int(((folds_with_top["top_decile_avg_R"] > RANKER_GOOD_AVG_R)
                 & (folds_with_top["top_decile_PF"] > RANKER_GOOD_PF)).sum())
    n_total = int(len(folds_with_top))
    ranker_ok = n_total > 0 and n_pos >= MIN_FOLDS_RANKER_OK

    rho_mean = float(folds_with_top["spearman_rho"].mean()) if n_total > 0 else float("nan")
    rho_min = float(folds_with_top["spearman_rho"].min()) if n_total > 0 else float("nan")

    if raw_ok and ranker_ok:
        verdict = "unconditional + rankable"
        reason = (f"Raw cohort PF={raw_pf:.2f}, avg_R={raw_avg:+.3f} both above thresholds; "
                  f"ranker top-decile good in {n_pos}/{n_total} folds (rho mean {rho_mean:+.3f}, min {rho_min:+.3f}).")
    elif raw_ok and not ranker_ok:
        verdict = "unconditional"
        reason = (f"Raw cohort already profitable (PF={raw_pf:.2f}, avg_R={raw_avg:+.3f}). "
                  f"Ranker did not improve consistently — top-decile good in {n_pos}/{n_total} folds, "
                  f"rho mean {rho_mean:+.3f} (negative or flat means ranker is anti-discriminating).")
    elif not raw_ok and ranker_ok:
        verdict = "conditional / rankable"
        reason = (f"Raw cohort weak (PF={raw_pf:.2f}, avg_R={raw_avg:+.3f}) but ranker top-decile clears "
                  f"thresholds in {n_pos}/{n_total} folds (rho mean {rho_mean:+.3f}, min {rho_min:+.3f}).")
    else:
        verdict = "not tradable as coded"
        reason = (f"Raw cohort weak (PF={raw_pf:.2f}, avg_R={raw_avg:+.3f}) and ranker top-decile failed in "
                  f"{n_total - n_pos}/{n_total} folds (rho mean {rho_mean:+.3f}).")
    return verdict, reason


def write_diagnosis(
    summary_csv: Path,
    ranker_csv: Path,
    out_md: Path,
) -> None:
    summary = pd.read_csv(summary_csv)
    ranker = pd.read_csv(ranker_csv)

    lines = ["# SBT-1700 — Edge Diagnosis (PR-2)",
             "",
             "_Diagnostic, not deployment. No live cron is enabled by this report._",
             "",
             "## Methodology disclaimer",
             "",
             "- Each exit variant is applied to the **same** entry-signal cohort.",
             "  Differences between variants reflect execution-rule sensitivity, not",
             "  signal-set differences.",
             "- LightGBM regressors are trained per variant on `realized_R_net` with",
             "  conservative defaults (seed=17, num_leaves=31, lr=0.05, n_estimators=200).",
             "- 3-fold walk-forward by signal date; train ≤ train_end, val on the next",
             "  contiguous date chunk. 2026 holdout is reported separately.",
             "- Top decile = top 10% of val rows ranked by predicted score.",
             "- Verdict thresholds (diagnosis-grade, NOT a deployment gate):",
             f"    * raw profitable: PF > {RAW_PROFITABLE_PF}, avg_R > {RAW_PROFITABLE_AVG_R}",
             f"    * ranker top-decile good in a fold: PF > {RANKER_GOOD_PF}, avg_R > {RANKER_GOOD_AVG_R}",
             f"    * 'rankable' requires {MIN_FOLDS_RANKER_OK}/3 walk-forward folds.",
             "",
             "## Verdicts",
             "",
             "| variant | verdict | raw N | raw PF | raw avg_R | ranker rho_mean | top-dec good folds |",
             "|---|---|---:|---:|---:|---:|---:|"]

    verdicts: list[dict] = []
    for _, raw in summary.iterrows():
        v = raw["exit_variant"]
        rk = ranker[ranker["exit_variant"] == v]
        verdict, reason = _classify(raw.to_dict(), rk)
        wf = rk[rk["fold"].str.startswith("fold_")].dropna(subset=["top_decile_avg_R"])
        n_pos = int(((wf["top_decile_avg_R"] > RANKER_GOOD_AVG_R)
                     & (wf["top_decile_PF"] > RANKER_GOOD_PF)).sum())
        n_total = int(len(wf))
        rho_mean = float(wf["spearman_rho"].mean()) if n_total > 0 else float("nan")
        verdicts.append({"variant": v, "verdict": verdict, "reason": reason})
        raw_pf = raw.get("PF_net", float("nan"))
        raw_avg = raw.get("avg_R_net", float("nan"))
        lines.append(
            f"| {v} | **{verdict}** | {int(raw['N'])} | "
            f"{raw_pf:.2f} | {raw_avg:+.3f} | "
            f"{rho_mean:+.3f} | {n_pos}/{n_total} |"
        )
    lines.append("")

    lines.append("## Per-variant detail")
    lines.append("")
    for vrec in verdicts:
        v = vrec["variant"]
        raw = summary[summary["exit_variant"] == v].iloc[0]
        rk = ranker[ranker["exit_variant"] == v]
        lines.append(f"### {v} — {vrec['verdict']}")
        lines.append("")
        lines.append(vrec["reason"])
        lines.append("")
        lines.append("**Raw cohort:**")
        lines.append("")
        lines.append(f"- N = {int(raw['N'])}, WR_net = {raw.get('WR_net', float('nan')):.3f}, "
                     f"PF_net = {raw.get('PF_net', float('nan')):.2f}, "
                     f"avg_R_net = {raw.get('avg_R_net', float('nan')):+.3f}, "
                     f"median = {raw.get('median_R_net', float('nan')):+.3f}")
        lines.append(f"- TP/SL/timeout/partial = "
                     f"{raw.get('tp_pct', 0):.2f} / {raw.get('sl_pct', 0):.2f} / "
                     f"{raw.get('timeout_pct', 0):.2f} / {raw.get('partial_pct', 0):.2f}, "
                     f"avg bars held = {raw.get('avg_bars_held', float('nan')):.2f}")
        lines.append(f"- top-5 R share = {raw.get('top5_R_share', float('nan')):+.3f}, "
                     f"bottom-5 R share = {raw.get('bot5_R_share', float('nan')):+.3f}, "
                     f"total R = {raw.get('total_R_net', float('nan')):+.2f}")
        lines.append("")
        lines.append("**Ranker folds (rho, top-decile metrics):**")
        lines.append("")
        lines.append("| fold | n_val | rho | all_avg_R | top_n | top_avg_R | top_PF | top_WR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in rk.iterrows():
            if pd.isna(row.get("spearman_rho")):
                lines.append(f"| {row['fold']} | {int(row.get('n_val', 0))} | – | – | – | – | – | – |")
                continue
            lines.append(
                f"| {row['fold']} | {int(row['n_val'])} | "
                f"{row['spearman_rho']:+.3f} | "
                f"{row['all_avg_R']:+.3f} | "
                f"{int(row['top_decile_n'])} | "
                f"{row['top_decile_avg_R']:+.3f} | "
                f"{row['top_decile_PF']:.2f} | "
                f"{row['top_decile_WR']:.3f} |"
            )
        lines.append("")

    lines.append("## Final disposition")
    lines.append("")
    unconditional_plus = [v["variant"] for v in verdicts if v["verdict"] == "unconditional + rankable"]
    unconditional_only = [v["variant"] for v in verdicts if v["verdict"] == "unconditional"]
    conditional = [v["variant"] for v in verdicts if v["verdict"] == "conditional / rankable"]
    bad = [v["variant"] for v in verdicts if v["verdict"] == "not tradable as coded"]
    if unconditional_plus:
        lines.append(f"- **Unconditional edge + ranker lifts top-decile**: {', '.join(unconditional_plus)}")
    if unconditional_only:
        lines.append(f"- **Unconditional edge (ranker does not help)**: {', '.join(unconditional_only)}")
    if conditional:
        lines.append(f"- **Conditional / rankable only**: {', '.join(conditional)}")
    if bad:
        lines.append(f"- **Not tradable as coded**: {', '.join(bad)}")
    lines.append("")
    lines.append("**No live deployment is enabled by this PR.** Promotion to a live cron requires "
                 "a separate, explicit decision after reviewing this diagnosis.")
    lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Render SBT-1700 edge diagnosis markdown.")
    ap.add_argument("--summary", type=Path, default=Path("output/sbt_1700_exit_matrix.csv"))
    ap.add_argument("--ranker", type=Path, default=Path("output/sbt_1700_ranker_by_exit.csv"))
    ap.add_argument("--out", type=Path, default=Path("output/sbt_1700_edge_diagnosis.md"))
    args = ap.parse_args()
    write_diagnosis(args.summary, args.ranker, args.out)
    print(f"[edge_diagnosis] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
