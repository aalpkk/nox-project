"""
CTE portfolio merge — optional cross-line allocation.

Takes per-line shortlists and merges them at the portfolio layer (NOT the
model layer). Three merge modes:

  1) fixed_quota       — take up to per_line_cap from each line, capped at
                         max_total_positions. Default.
  2) alternating       — interleave ranks: HB#1, FC#1, HB#2, FC#2, ...
  3) normalized_rank   — per-line percentile-rank score_model, then sort
                         globally by the normalized score. Addresses the
                         cross-line calibration mismatch documented in
                         memory/cte_v2_specialist.md.

Dedup rule: if a ticker appears on both lines on the same date (setup_type
== "both"), the better-ranked appearance wins; the other is dropped.

Input : output/cte_hb_shortlist.csv, output/cte_fc_shortlist.csv
Output: output/cte_portfolio_merged.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.config import CONFIG


def _normalize_rank(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Per-line per-date percentile rank (descending → 1.0 = best)."""
    return (
        df.groupby(["line", "date"])[score_col]
        .rank(method="average", pct=True, ascending=True, na_option="keep")
    )


def _dedup_both(df: pd.DataFrame) -> pd.DataFrame:
    """If a (ticker, date) pair appears on both lines, keep the one with the
    better (smaller) rank_within_line."""
    if df.empty:
        return df
    df = df.sort_values(["date", "ticker", "rank_within_line"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first")
    return df


def _merge_fixed_quota(
    hb: pd.DataFrame, fc: pd.DataFrame,
    per_line_cap: int, max_total: int,
) -> pd.DataFrame:
    parts = []
    for line, src in (("hb", hb), ("fc", fc)):
        if src is None or src.empty:
            continue
        lim = src.groupby("date").head(per_line_cap).copy()
        parts.append(lim)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["date", "line", "rank_within_line"])
    # Cap total per date
    merged = merged.groupby("date").head(max_total).reset_index(drop=True)
    merged["merge_rank"] = merged.groupby("date").cumcount() + 1
    return merged


def _merge_alternating(
    hb: pd.DataFrame, fc: pd.DataFrame,
    max_total: int,
) -> pd.DataFrame:
    if (hb is None or hb.empty) and (fc is None or fc.empty):
        return pd.DataFrame()

    out_rows = []
    for date, hb_day in (hb.groupby("date") if hb is not None and not hb.empty
                         else iter([])):
        fc_day = fc[fc["date"] == date].sort_values("rank_within_line") if fc is not None else pd.DataFrame()
        hb_day = hb_day.sort_values("rank_within_line")
        i_hb, i_fc = 0, 0
        picks = []
        while len(picks) < max_total and (i_hb < len(hb_day) or i_fc < len(fc_day)):
            if i_hb < len(hb_day):
                picks.append(hb_day.iloc[i_hb])
                i_hb += 1
                if len(picks) >= max_total:
                    break
            if i_fc < len(fc_day):
                picks.append(fc_day.iloc[i_fc])
                i_fc += 1
        if picks:
            out_rows.append(pd.DataFrame(picks))

    # Dates where only FC has signal
    if fc is not None and not fc.empty:
        hb_dates = set(hb["date"].unique()) if hb is not None else set()
        for date, fc_day in fc.groupby("date"):
            if date in hb_dates:
                continue
            fc_day = fc_day.sort_values("rank_within_line").head(max_total)
            out_rows.append(fc_day)

    if not out_rows:
        return pd.DataFrame()
    merged = pd.concat(out_rows, ignore_index=True)
    merged["merge_rank"] = merged.groupby("date").cumcount() + 1
    return merged


def _merge_normalized_rank(
    hb: pd.DataFrame, fc: pd.DataFrame,
    max_total: int,
) -> pd.DataFrame:
    parts = []
    for line, src in (("hb", hb), ("fc", fc)):
        if src is None or src.empty:
            continue
        s = src.copy()
        s["score_norm"] = _normalize_rank(s, "score_model")
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["date", "score_norm"], ascending=[True, False])
    merged = merged.groupby("date").head(max_total).reset_index(drop=True)
    merged["merge_rank"] = merged.groupby("date").cumcount() + 1
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-shortlist", default="output/cte_hb_shortlist.csv")
    ap.add_argument("--fc-shortlist", default="output/cte_fc_shortlist.csv")
    ap.add_argument("--merge-mode",
                    choices=["fixed_quota", "alternating", "normalized_rank"],
                    default=CONFIG.portfolio.merge_mode)
    ap.add_argument("--per-line-cap", type=int, default=CONFIG.portfolio.per_line_cap)
    ap.add_argument("--max-total", type=int, default=CONFIG.portfolio.max_total_positions)
    ap.add_argument("--dedup-both", action="store_true",
                    default=CONFIG.portfolio.dedup_on_both)
    ap.add_argument("--out", default="output/cte_portfolio_merged.csv")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    def _load(path: str, name: str) -> pd.DataFrame | None:
        if not Path(path).exists():
            print(f"⚠ {name} not found: {path}")
            return None
        return pd.read_csv(path, parse_dates=["date"])

    hb = _load(args.hb_shortlist, "HB shortlist")
    fc = _load(args.fc_shortlist, "FC shortlist")

    print(f"═══ portfolio_merge mode={args.merge_mode} ═══")
    print(f"  per_line_cap={args.per_line_cap}  max_total={args.max_total}  "
          f"dedup_both={args.dedup_both}")
    print(f"  HB rows={len(hb) if hb is not None else 0}  "
          f"FC rows={len(fc) if fc is not None else 0}")

    if args.merge_mode == "fixed_quota":
        merged = _merge_fixed_quota(hb, fc, args.per_line_cap, args.max_total)
    elif args.merge_mode == "alternating":
        merged = _merge_alternating(hb, fc, args.max_total)
    else:
        merged = _merge_normalized_rank(hb, fc, args.max_total)

    if args.dedup_both and not merged.empty:
        before = len(merged)
        merged = _dedup_both(merged)
        print(f"  dedup dropped {before - len(merged)} duplicate (ticker,date)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out}  rows={len(merged)}")

    # Summary diagnostics
    if not merged.empty:
        per_line_count = merged.groupby("line").size().to_dict()
        print(f"  picks per line: {per_line_count}")
        avg_per_day = merged.groupby("date").size().mean()
        print(f"  avg picks/day: {avg_per_day:.2f}")
        if "runner_15" in merged.columns:
            base_by_line = {}
            for line, src in (("hb", hb), ("fc", fc)):
                if src is not None and "runner_15" in src.columns:
                    base_by_line[line] = src["runner_15"].dropna().mean()
            hit = merged["runner_15"].dropna().mean()
            print(f"  merged runner_15 rate: {hit:.2%}  "
                  f"(per-line pool base: {base_by_line})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
