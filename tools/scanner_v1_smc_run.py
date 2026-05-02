"""SMC families snapshot scan — full 607 universe at latest asof.

Runs mitigation_block + breaker_block on the locked nox_intraday_v1 master and
writes one parquet per family. Sanity output: rows per state, top-20 by
rule_score, distribution of zone_age / retest_kind. Production validation
(3y backtest, PF/EF gates) is a separate session — this is the SMOKE/SANITY
pass only.
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

from data import intraday_1h
from scanner.engine import scan


OUT_DIR = REPO / "output"
FAMILIES = ("mitigation_block", "breaker_block")


def _summarize(out: pd.DataFrame, family: str) -> None:
    fam_rows = out[out["setup_family"] == family]
    if fam_rows.empty:
        print(f"[{family}] 0 rows.")
        return
    print(f"\n=== {family} ===")
    print(f"  N rows: {len(fam_rows)}")
    print()
    print("  state distribution:")
    print(fam_rows["signal_state"].value_counts().to_string())
    print()
    print("  retest_kind distribution:")
    print(fam_rows["family__retest_kind"].fillna("").replace("", "<empty>").value_counts().to_string())
    print()
    print("  zone_age_bars (median / p25 / p75 / max):")
    age = fam_rows["family__zone_age_bars"].astype(float)
    print(f"    median={age.median():.0f}  p25={age.quantile(0.25):.0f}  "
          f"p75={age.quantile(0.75):.0f}  max={age.max():.0f}")
    print()
    print("  rule_score (median / p75 / max):")
    sc = fam_rows["rule_score"].astype(float)
    print(f"    median={sc.median():.2f}  p75={sc.quantile(0.75):.2f}  max={sc.max():.2f}")
    print()
    cols = ["ticker", "signal_state", "rule_score", "family__zone_age_bars",
            "family__bos_distance_atr", "family__retest_kind",
            "family__zone_width_atr", "common__rs_pctile_252",
            "common__close_vs_sma20"]
    available = [c for c in cols if c in fam_rows.columns]
    top = fam_rows.sort_values("rule_score", ascending=False).head(20)[available]
    print("  top-20 by rule_score:")
    print(top.to_string(index=False, float_format="%.3f"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp (Europe/Istanbul); defaults to last 1h bar.")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="Per-ticker coverage_pct floor (default 0.0 = all).")
    ap.add_argument("--out-prefix", default="scanner_v1_smc",
                    help="Output filename prefix (default scanner_v1_smc).")
    args = ap.parse_args()

    try:
        intraday_1h.verify_dataset()
    except FileNotFoundError as e:
        print(f"[warn] {e}")
        print("[warn] proceeding anyway — manifest absence does not block reading the master parquet.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fam in FAMILIES:
        t0 = time.time()
        out_path = OUT_DIR / f"{args.out_prefix}_{fam}.parquet"
        print(f"\n[run] family={fam} asof={args.asof or 'latest'} → {out_path.name}")
        out = scan(
            tickers=None,
            families=(fam,),
            asof=args.asof,
            min_coverage=args.min_coverage,
            out_path=out_path,
        )
        elapsed = time.time() - t0
        print(f"[run] {fam}: {len(out)} rows in {elapsed:.1f}s")
        _summarize(out, fam)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
