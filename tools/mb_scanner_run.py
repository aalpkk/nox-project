"""mb_scanner runner — strict MSS-validated MB/BB scan over 5h/1d/1w/1M.

Usage:
    python tools/mb_scanner_run.py [--asof "2026-04-30 18:00"] [--families mb_5h mb_1d ...]
                                   [--tickers MGROS HDFGS] [--min-coverage 0.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.engine import _PARAMS as FAM_PARAMS, scan
from mb_scanner.schema import FAMILIES


def _summarize(df: pd.DataFrame, family: str) -> None:
    if df.empty:
        print(f"[{family}] 0 rows.")
        return
    print()
    print(f"=== {family} | N={len(df)} ===")
    print()
    print("  state distribution:")
    print(df["signal_state"].value_counts().to_string())

    print()
    print("  retest_kind distribution (non-empty):")
    rk = df["retest_kind"].fillna("")
    rk = rk[rk != ""]
    if len(rk):
        print(rk.value_counts().to_string())
    else:
        print("    (none)")

    print()
    cols = [
        "ticker", "signal_state", "quartet_rank", "n_active_quartets",
        "ll_bar_date", "lh_bar_date", "hl_bar_date", "hh_bar_date",
        "zone_high", "zone_low", "zone_age_bars",
        "bos_distance_atr", "retest_kind", "retest_depth_atr", "asof_close",
    ]
    avail = [c for c in cols if c in df.columns]
    print("  TOP 15 by zone freshness (ascending zone_age_bars), states != extended:")
    pri = df[df["signal_state"].isin(["above_mb", "mitigation_touch", "retest_bounce"])]
    if not pri.empty:
        top = pri.sort_values(["signal_state", "zone_age_bars"]).head(15)[avail]
        print(top.to_string(index=False))
    else:
        print("    (no above_mb / mitigation_touch / retest_bounce rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp (Europe/Istanbul). Defaults to last bar in each panel.")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES),
                    help=f"Families to run; default = all {list(FAMILIES)}")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Subset tickers (default: full universe).")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="Per-ticker coverage_pct floor (default 0.0 = include all).")
    args = ap.parse_args()

    print(f"[run] asof={args.asof or 'latest'}  families={args.families}  "
          f"tickers={'all' if args.tickers is None else args.tickers}")

    t0 = time.time()
    out = scan(
        families=args.families,
        tickers=args.tickers,
        asof=args.asof,
        min_coverage=args.min_coverage,
        write_parquet=True,
    )
    elapsed = time.time() - t0
    total_rows = sum(len(df) for df in out.values())
    print(f"\n[run] done in {elapsed:.1f}s  total_rows={total_rows}")

    for fam in args.families:
        df = out.get(fam)
        params = FAM_PARAMS[fam]
        print(f"\n[fam {fam}]  freq={params.frequency}  pivot_n={params.pivot_n}  "
              f"max_quartet_span={params.max_quartet_span_bars}  "
              f"max_zone_age={params.max_zone_age_bars}")
        _summarize(df if df is not None else pd.DataFrame(), fam)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
