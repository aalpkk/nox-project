"""mb_scanner heuristic ranker — per-family top-N by composite score.

Scoring lives in `mb_scanner/rank.py` so the HTML report tool can share it.

Usage:
    python tools/mb_scanner_rank.py [--top 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.rank import OUT_DIR, rank_all
from mb_scanner.schema import FAMILIES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    ranked = rank_all(out_dir=OUT_DIR)

    summary_cols = [
        "ticker", "score", "signal_state", "quartet_rank", "n_active_quartets",
        "ll_bar_date", "lh_bar_date", "hl_bar_date", "hh_bar_date",
        "zone_high", "zone_low", "zone_age_bars",
        "bos_distance_atr", "retest_kind", "retest_depth_atr",
        "asof_close", "also_fires_in",
    ]

    for fam in FAMILIES:
        df = ranked.get(fam)
        if df is None or df.empty:
            print(f"\n=== {fam} | (no active fresh quartets) ===")
            continue
        target = OUT_DIR / f"mb_scanner_rank_{fam}.parquet"
        df.to_parquet(target, index=False)
        avail = [c for c in summary_cols if c in df.columns]
        print()
        print(f"=== {fam} | N(rank)={len(df)}  TOP {min(args.top, len(df))} ===")
        print(df[avail].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
