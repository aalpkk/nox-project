"""Validation report for the SBT-1700 dataset.

The acceptance gate (PR-2) is on the trained ranker, not on the raw
dataset — but a clean dataset is a precondition. This module emits
a human-readable markdown report covering coverage, label distribution,
trade concentration, and lookahead sanity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_validation_report(panel: pd.DataFrame, meta: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# SBT-1700 — Dataset Validation Report\n")
    lines.append(f"_Schema: **{meta.get('schema_version')}** · "
                 f"built: {meta.get('built_at')}_\n")

    lines.append("## Provenance\n")
    lines.append(f"- Rows: **{meta.get('rows'):,}**")
    lines.append(f"- Tickers: **{meta.get('tickers')}**")
    lines.append(f"- Date range: **{meta.get('date_min')} → {meta.get('date_max')}**")
    lines.append(f"- Expected 15m bars per pair: **{meta.get('expected_bars_per_pair')}** "
                 "(close ≤ 16:45 TR)")
    lines.append(f"- Coverage drop threshold: **{meta.get('coverage_drop_threshold'):.0%}**")
    lines.append("- E3 params: ```\n  " +
                 json.dumps(meta.get("e3_params", {}), indent=2).replace("\n", "\n  ") +
                 "\n  ```")

    if panel.empty:
        lines.append("\n⚠ Panel is empty — nothing to validate.")
        out_path.write_text("\n".join(lines))
        return

    lines.append("\n## Coverage distribution\n")
    cov = panel["intraday_coverage"]
    lines.append(_pct_table(cov.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])))
    partial = ((cov >= 0.80) & (cov < 0.95)).sum()
    full = (cov >= 0.95).sum()
    lines.append(f"- Full (≥95%): **{full:,}**")
    lines.append(f"- Partial (80–95%): **{partial:,}**")
    lines.append(f"- Pairs missing bars (n_bars < expected): "
                 f"{(panel['n_bars_1700'] < meta.get('expected_bars_per_pair', 27)).sum():,}")

    lines.append("\n## Label distribution\n")
    lab = panel.copy()
    win_rate = lab["win_label"].mean() if "win_label" in lab.columns else np.nan
    n_tp = lab.get("tp_hit", pd.Series(dtype=bool)).sum()
    n_sl = lab.get("sl_hit", pd.Series(dtype=bool)).sum()
    n_to = lab.get("timeout_hit", pd.Series(dtype=bool)).sum()
    lines.append(f"- N: {len(lab):,}")
    lines.append(f"- WR (net): **{win_rate:.1%}**" if pd.notna(win_rate) else "- WR: n/a")
    lines.append(f"- TP hits: **{int(n_tp):,}** "
                 f"({int(n_tp) / max(len(lab), 1):.1%})")
    lines.append(f"- SL hits: **{int(n_sl):,}** "
                 f"({int(n_sl) / max(len(lab), 1):.1%})")
    lines.append(f"- Timeouts: **{int(n_to):,}** "
                 f"({int(n_to) / max(len(lab), 1):.1%})")
    if "realized_R_net" in lab.columns:
        rn = lab["realized_R_net"].dropna()
        if len(rn):
            mean_R = rn.mean()
            med_R = rn.median()
            pf_num = rn[rn > 0].sum()
            pf_den = -rn[rn < 0].sum()
            pf = pf_num / pf_den if pf_den > 0 else float("inf")
            lines.append(f"- Mean realized_R_net: **{mean_R:+.3f}**")
            lines.append(f"- Median realized_R_net: **{med_R:+.3f}**")
            lines.append(f"- PF (net): **{pf:.2f}** "
                         f"(gross_wins {pf_num:.1f}R / gross_losses {pf_den:.1f}R)")

    lines.append("\n## Trade concentration\n")
    if "realized_R_net" in lab.columns and len(lab):
        top5 = lab.nlargest(5, "realized_R_net")["realized_R_net"].sum()
        total_R = lab["realized_R_net"].sum()
        lines.append(f"- Top-5 trades total R: **{top5:+.2f}** "
                     f"of **{total_R:+.2f}** "
                     f"({top5 / total_R * 100:.1f}% of total) "
                     "— tail-pocket flag if > 50% on small N.")
        ticker_counts = lab.groupby("ticker").size().sort_values(ascending=False)
        lines.append(f"- Top ticker rows: {ticker_counts.head(5).to_dict()}")
        date_counts = lab.groupby("date").size().sort_values(ascending=False)
        lines.append(f"- Most candidate-heavy dates: "
                     f"{ {str(k.date()): int(v) for k,v in date_counts.head(5).items()} }")

    lines.append("\n## Lookahead sanity checks\n")
    lines.append("- T's row: `Open`, `High`, `Low`, `Close`, `Volume` come "
                 "from the 17:00-truncated aggregator (close ≤ 16:45 TR cutoff).")
    lines.append("- Prior daily features (`*_prior`) all index by T-1 — no T close used.")
    lines.append("- Forward window starts at T+1 — T's daily close not consumed.")
    if "exit_date" in lab.columns:
        bad = lab[(pd.to_datetime(lab["exit_date"]) <= pd.to_datetime(lab["date"]))
                  & lab["exit_date"].notna()]
        lines.append(f"- exit_date ≤ entry_date violations: **{len(bad)}** "
                     "(expected 0).")

    lines.append("\n## Schema\n")
    feats = meta.get("feature_columns", [])
    labels = meta.get("label_columns", [])
    lines.append(f"- Feature columns ({len(feats)}): `{', '.join(feats)}`")
    lines.append(f"- Label columns ({len(labels)}): `{', '.join(labels)}`")

    out_path.write_text("\n".join(lines))


def _pct_table(series: pd.Series) -> str:
    rows = ["| stat | value |", "|---|---|"]
    for k, v in series.items():
        try:
            rows.append(f"| {k} | {v:.4f} |")
        except (TypeError, ValueError):
            rows.append(f"| {k} | {v} |")
    return "\n".join(rows)
