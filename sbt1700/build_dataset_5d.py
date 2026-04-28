"""Build the SBT-1700 / 5d-potential dataset on the nox_intraday_v1 bundle.

End-to-end pipeline:
  1. Load 1h master via data.intraday_1h.load_intraday() with
     min_coverage=0.95 (per-ticker eligibility).
  2. daily_resample_full(1h panel) → full-day daily OHLCV per ticker
     (used for prior indicators + forward 5d window). Single source of
     truth: the same 1h master powers both T's truncated bar and the
     forward window.
  3. aggregator_1h.aggregate_truncated_bars_1h → 17:00-truncated daily
     bar per (ticker, signal_date), n_bars and intraday_coverage.
  4. signals.detect_candidates(expected_bars=8) on the patched panel.
  5. features.build_features(expected_bars=8).
  6. labels_5d.attach_labels_5d (forward 5d MFE / close-return labels;
     no E3 simulator).
  7. Coverage policy: keep rows with intraday_coverage ≥ 7/8 (= 0.875).
     Rows with NaN labels (tail-of-panel, fwd_n_bars < 5) are kept in
     parquet and filtered later by the trainer.
  8. Persist output/sbt_1700_dataset_5d_intraday_v1.parquet + meta JSON.

CLI:
    python -m sbt1700.build_dataset_5d \\
        --start 2023-11-15 --end 2026-04-24 \\
        --min-coverage 0.95 \\
        --out output/sbt_1700_dataset_5d_intraday_v1.parquet \\
        --validate

Notes
-----
* The locked split (sbt1700.splits) operates on the ``date`` column of
  the persisted parquet; this builder produces every row before the
  TEST_END date and lets the trainer respect the lock.
* The TEST slice (2026-01-01..2026-04-24) is generated here without
  loading it for training. Coverage of TEST in the parquet is informational.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import intraday_1h
from sbt1700.aggregator_1h import aggregate_truncated_bars_1h, daily_resample_full
from sbt1700.config import (
    EXPECTED_BARS_1H,
    SCHEMA_VERSION,
)
from sbt1700.features import build_features
from sbt1700.labels_5d import attach_labels_5d, label_columns_5d
from sbt1700.signals import detect_candidates


# Per-day 1h coverage threshold: 7/8 = 0.875 (allow one missing bar).
COVERAGE_DROP_THRESHOLD_1H = 7.0 / EXPECTED_BARS_1H

# Dataset schema bump for the 5d-potential variant.
SCHEMA_VERSION_5D = f"{SCHEMA_VERSION}-5d-intraday_v1"

DEFAULT_OUT = Path("output/sbt_1700_dataset_5d_intraday_v1.parquet")


def _filter_daterange(df: pd.DataFrame, start: str | None, end: str | None,
                      date_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if start:
        out = out[out[date_col] >= pd.Timestamp(start)]
    if end:
        out = out[out[date_col] <= pd.Timestamp(end)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build SBT-1700 / 5d-potential dataset on nox_intraday_v1."
    )
    ap.add_argument("--start", type=str, default=None,
                    help="signal-date floor (default: nox_intraday_v1 MIN_DATE)")
    ap.add_argument("--end", type=str, default=None,
                    help="signal-date ceiling")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="per-ticker coverage_pct floor for eligible_tickers")
    ap.add_argument("--tickers", type=str, nargs="*", default=None,
                    help="optional explicit ticker list (smoke test)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--verify-bundle", action="store_true",
                    help="run intraday_1h.verify_dataset() before loading")
    args = ap.parse_args()

    print(f"[sbt1700-5d] schema={SCHEMA_VERSION_5D}")
    if args.verify_bundle:
        intraday_1h.verify_dataset()
        print("[sbt1700-5d] bundle verify: ok")

    print(f"[sbt1700-5d] loading 1h master "
          f"(min_coverage={args.min_coverage}, tickers={args.tickers!r})")
    bars = intraday_1h.load_intraday(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        min_coverage=args.min_coverage,
    )
    print(f"[sbt1700-5d] 1h bars: {bars.shape}, "
          f"{bars['ticker'].nunique()} tickers, "
          f"{bars['ts_istanbul'].min()} → {bars['ts_istanbul'].max()}")

    print("[sbt1700-5d] daily resample (full day, no cutoff)…")
    daily_master = daily_resample_full(bars)
    print(f"[sbt1700-5d] daily master: {daily_master.shape}, "
          f"{daily_master['ticker'].nunique()} tickers")

    print("[sbt1700-5d] aggregate 17:00-truncated bars…")
    truncated = aggregate_truncated_bars_1h(bars)
    truncated = _filter_daterange(truncated, args.start, args.end, "signal_date")
    print(f"[sbt1700-5d] truncated bars: {truncated.shape} "
          f"(expected {EXPECTED_BARS_1H}/pair)")

    print("[sbt1700-5d] detect SBT-1700 signals…")
    candidates = detect_candidates(daily_master, truncated, expected_bars=EXPECTED_BARS_1H)
    print(f"[sbt1700-5d] candidates: {len(candidates)}")

    if candidates.empty:
        print("[sbt1700-5d] no candidates — writing empty panel.")
        empty = pd.DataFrame(columns=["ticker", "date"])
        empty.to_parquet(args.out, index=False)
        return 0

    print("[sbt1700-5d] build 17:00-aware features…")
    features = build_features(daily_master, candidates, expected_bars=EXPECTED_BARS_1H)
    print(f"[sbt1700-5d] feature panel: {features.shape}")

    print("[sbt1700-5d] attach 5d forward labels…")
    panel = attach_labels_5d(daily_master, features)
    print(f"[sbt1700-5d] labelled panel: {panel.shape}")

    # Coverage policy (per-day intraday completeness).
    if "intraday_coverage" in panel.columns:
        before = len(panel)
        panel = panel[panel["intraday_coverage"] >= COVERAGE_DROP_THRESHOLD_1H].reset_index(drop=True)
        dropped = before - len(panel)
        print(f"[sbt1700-5d] dropped {dropped} rows below per-day coverage "
              f"{COVERAGE_DROP_THRESHOLD_1H:.3%}")

    panel["schema_version"] = SCHEMA_VERSION_5D

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    print(f"[sbt1700-5d] wrote {args.out} ({len(panel)} rows)")

    meta_path = args.out.with_suffix(".meta.json")
    label_cols = label_columns_5d()
    meta = {
        "schema_version": SCHEMA_VERSION_5D,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset_source": intraday_1h.DATASET_VERSION,
        "rows": int(len(panel)),
        "tickers": int(panel["ticker"].nunique()) if not panel.empty else 0,
        "date_min": str(panel["date"].min().date()) if not panel.empty else None,
        "date_max": str(panel["date"].max().date()) if not panel.empty else None,
        "expected_bars_per_pair": EXPECTED_BARS_1H,
        "per_day_coverage_drop_threshold": COVERAGE_DROP_THRESHOLD_1H,
        "horizon_days": 5,
        "min_fwd_bars": 5,
        "label_columns": label_cols,
        "feature_columns": [
            c for c in panel.columns
            if c not in ("ticker", "date", "schema_version")
            and c not in label_cols
        ],
        "n_rows_with_labels": int(panel["mfe_5d_R"].notna().sum()) if "mfe_5d_R" in panel.columns else 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[sbt1700-5d] wrote {meta_path}")

    if args.validate:
        # Lightweight inline summary (no external markdown writer here).
        n_with = int(panel["mfe_5d_R"].notna().sum())
        print(f"[sbt1700-5d] rows with non-null mfe_5d_R: {n_with}/{len(panel)}")
        if n_with > 0:
            sub = panel.dropna(subset=["mfe_5d_R"])
            print(f"[sbt1700-5d] mfe_5d_R: mean={sub['mfe_5d_R'].mean():.3f}, "
                  f"median={sub['mfe_5d_R'].median():.3f}, "
                  f"p90={sub['mfe_5d_R'].quantile(0.9):.3f}")
            print(f"[sbt1700-5d] hit_1R_5d rate: {sub['hit_1R_5d'].mean():.3%}")
            print(f"[sbt1700-5d] hit_2R_5d rate: {sub['hit_2R_5d'].mean():.3%}")
            print(f"[sbt1700-5d] close_positive_5d rate: {sub['close_positive_5d'].mean():.3%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
