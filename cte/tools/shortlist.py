"""
CTE shortlist — daily top-K per line.

Two independent lists per date:
  - HB shortlist: top_k_hb from HB preds
  - FC shortlist: top_k_fc from FC preds

Output CSVs (default):
  output/cte_hb_shortlist.csv
  output/cte_fc_shortlist.csv

Columns: date, ticker, setup_type, score_model, rank_within_line,
         score_compression, compression_score, bar_return_1d,
         breakout_vol_ratio, <target>, and selected label diagnostics.

No cross-line merging here. For allocation across both lines, see
cte.tools.portfolio_merge.
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


COLS_DIAG = [
    "compression_score", "bar_return_1d", "breakout_vol_ratio",
    "hold_3_close", "hold_5_close", "failed_break_5_close",
    "runner_10", "runner_15", "runner_20",
    "mfe_15_atr", "mae_15_atr",
]


def _build_line_shortlist(
    preds: pd.DataFrame,
    line: str,
    top_k: int,
    min_score: float,
    drop_failed_break_5: bool,
    target: str,
) -> pd.DataFrame:
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])

    if drop_failed_break_5 and "failed_break_5_close" in df.columns:
        before = len(df)
        df = df[df["failed_break_5_close"] != 1]
        print(f"  [{line}] filter failed_break_5_close==1 dropped {before - len(df)} rows")

    if min_score > 0:
        before = len(df)
        df = df[df["score_model"] >= min_score]
        print(f"  [{line}] filter score_model>={min_score} dropped {before - len(df)} rows")

    df = df.sort_values(["date", "score_model"], ascending=[True, False])
    df["rank_within_line"] = df.groupby("date").cumcount() + 1
    df = df[df["rank_within_line"] <= top_k].copy()
    df["line"] = line

    keep = [
        "date", "ticker", "line", "setup_type", "fold_assigned",
        "score_model", "rank_within_line",
        "score_compression", "score_random", target,
    ] + [c for c in COLS_DIAG if c in df.columns]
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    return df[keep].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-preds", default="output/cte_hb_preds_v1.parquet")
    ap.add_argument("--fc-preds", default="output/cte_fc_preds_v1.parquet")
    ap.add_argument("--target", default=CONFIG.label.primary_target)
    ap.add_argument("--top-k-hb", type=int, default=CONFIG.shortlist.top_k_hb)
    ap.add_argument("--top-k-fc", type=int, default=CONFIG.shortlist.top_k_fc)
    ap.add_argument("--min-score", type=float, default=CONFIG.shortlist.min_score_model)
    ap.add_argument("--drop-failed-break-5", action="store_true",
                    default=CONFIG.shortlist.drop_failed_break_5_close)
    ap.add_argument("--out-hb", default="output/cte_hb_shortlist.csv")
    ap.add_argument("--out-fc", default="output/cte_fc_shortlist.csv")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    def _load(path: str, name: str) -> pd.DataFrame | None:
        if not Path(path).exists():
            print(f"⚠ {name} preds not found at {path} — skipping")
            return None
        return pd.read_parquet(path)

    hb = _load(args.hb_preds, "HB")
    fc = _load(args.fc_preds, "FC")

    if hb is not None:
        hb_sl = _build_line_shortlist(
            hb, "hb", args.top_k_hb, args.min_score,
            args.drop_failed_break_5, args.target,
        )
        Path(args.out_hb).parent.mkdir(parents=True, exist_ok=True)
        hb_sl.to_csv(args.out_hb, index=False)
        print(f"[WRITE] {args.out_hb}  rows={len(hb_sl)}")

        # Summary: per-day avg, runner rate within shortlist vs within HB test set
        if args.target in hb_sl.columns:
            base = hb[args.target].dropna().mean() if args.target in hb.columns else float("nan")
            hit = hb_sl[args.target].dropna().mean()
            print(f"  HB shortlist {args.target} rate: {hit:.2%}  "
                  f"(vs HB test-set base {base:.2%})  "
                  f"lift={(hit/base):.2f}x" if base and not np.isnan(base) else "")

    if fc is not None:
        fc_sl = _build_line_shortlist(
            fc, "fc", args.top_k_fc, args.min_score,
            args.drop_failed_break_5, args.target,
        )
        Path(args.out_fc).parent.mkdir(parents=True, exist_ok=True)
        fc_sl.to_csv(args.out_fc, index=False)
        print(f"[WRITE] {args.out_fc}  rows={len(fc_sl)}")

        if args.target in fc_sl.columns:
            base = fc[args.target].dropna().mean() if args.target in fc.columns else float("nan")
            hit = fc_sl[args.target].dropna().mean()
            print(f"  FC shortlist {args.target} rate: {hit:.2%}  "
                  f"(vs FC test-set base {base:.2%})  "
                  f"lift={(hit/base):.2f}x" if base and not np.isnan(base) else "")

    return 0


if __name__ == "__main__":
    sys.exit(main())
