"""mb_scanner US POC smoke runner.

End-to-end pipeline validation: load US 1h bars from the new master parquet,
run mb_scanner daily/weekly/monthly families (mb_5h skipped — BIST session
hardcoded), print per-family summary and write per-family parquets to
output/mb_scanner_us_<family>.parquet.

Skips mb_scanner.engine.scan() because that wrapper imports BIST adapter
directly. Reuses lower-level engine functions (_detect_for_panel,
_resolve_asof) with US bars instead.

Usage:
    python tools/mb_scanner_us_poc.py
        [--families mb_1d mb_1w mb_1M]
        [--tickers AAPL MSFT NVDA GOOGL AMZN]
        [--asof "2026-05-21"]

Pre-req: tools/extfeed_pull_us.py --mode bootstrap (master parquet exists).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import intraday_1h_us  # noqa: E402
from mb_scanner.engine import _PARAMS as FAM_PARAMS, _detect_for_panel, _resolve_asof  # noqa: E402
from mb_scanner.resample import per_ticker_panel, to_daily, to_monthly, to_weekly  # noqa: E402
from mb_scanner.schema import OUTPUT_COLUMNS  # noqa: E402

OUT_DIR = REPO / "output"

# 5h is BIST-specific. Caller may still pass it; we filter it out with a warning.
US_SUPPORTED_FAMILIES = ("mb_1d", "mb_1w", "mb_1M", "bb_1d", "bb_1w", "bb_1M")


def _summarize(df: pd.DataFrame, family: str) -> None:
    if df.empty:
        print(f"[{family}] 0 rows.")
        return
    print()
    print(f"=== {family} | N={len(df)} ===")
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
    pri = df[df["signal_state"].isin(["above_mb", "mitigation_touch", "retest_bounce"])]
    print(f"  TOP 15 actionable (above_mb/mitigation_touch/retest_bounce), "
          f"by zone freshness:")
    if not pri.empty:
        top = pri.sort_values(["signal_state", "zone_age_bars"]).head(15)[avail]
        print(top.to_string(index=False))
    else:
        print("    (no actionable rows — all extended or no quartets)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=["mb_1d", "mb_1w", "mb_1M"])
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Subset tickers (default = all in master).")
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp (America/New_York). Default = last bar.")
    args = ap.parse_args()

    # Sanitize families: drop 5h with warning.
    fam_keys = []
    for f in args.families:
        if f not in FAM_PARAMS:
            print(f"!! unknown family {f}", file=sys.stderr)
            return 1
        if f not in US_SUPPORTED_FAMILIES:
            print(f"!! skipping {f}: 5h families not supported on US (BIST "
                  f"session hardcoded).", file=sys.stderr)
            continue
        fam_keys.append(f)
    if not fam_keys:
        print("!! no supported families to scan.", file=sys.stderr)
        return 1

    print(f"[run] families={fam_keys} tickers={args.tickers or 'all'} "
          f"asof={args.asof or 'latest'}", flush=True)

    t0 = time.time()
    bars = intraday_1h_us.load_intraday(
        tickers=args.tickers, start=None, end=None, min_coverage=0.0,
    )
    if bars.empty:
        print("!! no bars in master.", file=sys.stderr)
        return 2
    n_tickers = bars["ticker"].nunique()
    print(f"[load] {len(bars):,} rows / {n_tickers} tickers "
          f"({time.time() - t0:.1f}s)", flush=True)

    needed_freqs = {FAM_PARAMS[f].frequency for f in fam_keys}
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    t1 = time.time()
    if "1d" in needed_freqs:
        panels["1d"] = per_ticker_panel(to_daily(bars), "date")
    if "1w" in needed_freqs:
        panels["1w"] = per_ticker_panel(to_weekly(bars), "week_end")
    if "1M" in needed_freqs:
        panels["1M"] = per_ticker_panel(to_monthly(bars), "month_end")
    print(f"[resample] {time.time() - t1:.1f}s", flush=True)

    asof_ts = pd.Timestamp(args.asof) if args.asof else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in fam_keys:
        fam = FAM_PARAMS[f]
        panel_map = panels[fam.frequency]
        t2 = time.time()
        rows: list[dict] = []
        for ticker, df in panel_map.items():
            asof_idx = _resolve_asof(df, asof_ts)
            if asof_idx < 0:
                continue
            r = _detect_for_panel(
                ticker=ticker, fam=fam, df=df, asof_idx=asof_idx,
            )
            if r:
                rows.extend(r)
        out_df = (pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
                  if rows else pd.DataFrame(columns=list(OUTPUT_COLUMNS)))
        target = OUT_DIR / f"mb_scanner_us_{f}.parquet"
        out_df.to_parquet(target, index=False)
        print(f"[fam {f}]  freq={fam.frequency}  pivot_n={fam.pivot_n}  "
              f"max_zone_age={fam.max_zone_age_bars}  ({time.time() - t2:.1f}s)")
        _summarize(out_df, f)
        print(f"  → wrote {target}")

    print(f"\n[run] done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
