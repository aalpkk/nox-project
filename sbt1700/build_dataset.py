"""Build the SBT-1700 dataset.

End-to-end pipeline:
  1. Load 15m intraday cache(s).
  2. Aggregate to 17:00-truncated daily bars (cutoff=16:45 TR).
  3. Detect 17:00 SBT signal candidates over the daily master with T patched.
  4. Compute 17:00-aware features (no lookahead, no EOD close of T).
  5. Run E3 execution over forward daily bars to attach realized-R labels.
  6. Apply coverage policy: drop intraday_coverage < 0.80, flag 0.80–0.95.
  7. Persist output/sbt_1700_dataset.parquet + run validation report.

CLI:
  python -m sbt1700.build_dataset \\
      --master output/ohlcv_10y_fintables_master.parquet \\
      --intraday output/nyxexp_intraday_15m_matriks.parquet \\
                 output/sbt1700_intraday_15m.parquet \\
      --start 2023-11-15 --end 2026-04-08 \\
      --out output/sbt_1700_dataset.parquet \\
      --validate
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sbt1700.aggregator import aggregate_truncated_bars
from sbt1700.config import (
    SCHEMA_VERSION,
    COVERAGE_DROP_THRESHOLD,
    EXPECTED_BARS_PER_PAIR,
)
from sbt1700.execution import params_dict as e3_params_dict
from sbt1700.features import build_features
from sbt1700.labels import attach_labels, label_columns
from sbt1700.signals import detect_candidates


DEFAULT_MASTER = Path("output/ohlcv_10y_fintables_master.parquet")
DEFAULT_OUT = Path("output/sbt_1700_dataset.parquet")
DEFAULT_INTRADAY = [
    Path("output/nyxexp_intraday_15m_matriks.parquet"),
    Path("output/sbt1700_intraday_15m.parquet"),
]


def load_intraday(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        if p and p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    return df.drop_duplicates(subset=["ticker", "signal_date", "bar_ts"])


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name != "Date":
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"}).set_index("Date")
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def filter_daterange(df: pd.DataFrame, start: str | None, end: str | None,
                     date_col: str = "signal_date") -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if start:
        out = out[out[date_col] >= pd.Timestamp(start)]
    if end:
        out = out[out[date_col] <= pd.Timestamp(end)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SBT-1700 dataset.")
    ap.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--intraday", type=Path, nargs="+", default=DEFAULT_INTRADAY)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--validate", action="store_true",
                    help="Also write validation markdown next to the parquet.")
    args = ap.parse_args()

    print(f"[sbt1700] schema={SCHEMA_VERSION} master={args.master}")
    master = load_master(args.master)
    print(f"[sbt1700] daily master: {master.shape}, "
          f"{master['ticker'].nunique()} tickers, "
          f"{master.index.min().date()} → {master.index.max().date()}")

    intraday = load_intraday(args.intraday)
    intraday = filter_daterange(intraday, args.start, args.end, "signal_date")
    print(f"[sbt1700] intraday 15m: {intraday.shape} rows after filter")

    bars = aggregate_truncated_bars(intraday)
    print(f"[sbt1700] truncated bars: {bars.shape} (expected {EXPECTED_BARS_PER_PAIR}/pair)")

    candidates = detect_candidates(master, bars)
    print(f"[sbt1700] SBT-1700 candidates: {len(candidates)}")

    features = build_features(master, candidates)
    print(f"[sbt1700] feature panel: {features.shape}")

    panel = attach_labels(master, features)
    print(f"[sbt1700] labelled panel: {panel.shape}")

    # Coverage policy
    if "intraday_coverage" in panel.columns:
        before = len(panel)
        panel = panel[panel["intraday_coverage"] >= COVERAGE_DROP_THRESHOLD].reset_index(drop=True)
        dropped = before - len(panel)
        print(f"[sbt1700] dropped {dropped} rows below coverage {COVERAGE_DROP_THRESHOLD:.0%}")

    panel["schema_version"] = SCHEMA_VERSION

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    print(f"[sbt1700] wrote {args.out} ({len(panel)} rows)")

    meta_path = args.out.with_suffix(".meta.json")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "rows": int(len(panel)),
        "tickers": int(panel["ticker"].nunique()) if not panel.empty else 0,
        "date_min": str(panel["date"].min().date()) if not panel.empty else None,
        "date_max": str(panel["date"].max().date()) if not panel.empty else None,
        "expected_bars_per_pair": EXPECTED_BARS_PER_PAIR,
        "coverage_drop_threshold": COVERAGE_DROP_THRESHOLD,
        "e3_params": e3_params_dict(),
        "label_columns": label_columns(),
        "feature_columns": [c for c in panel.columns
                            if c not in ("ticker", "date", "schema_version")
                            and c not in label_columns()],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[sbt1700] wrote {meta_path}")

    if args.validate:
        from sbt1700.validate_dataset import write_validation_report
        report_path = args.out.with_name(args.out.stem + "_validation.md")
        write_validation_report(panel, meta, report_path)
        print(f"[sbt1700] wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
