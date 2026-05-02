"""ema_context Phase 0 orchestrator — descriptive feature health layer.

Spec: memory/ema_context_phase0_spec.md (LOCKED 2026-05-03).
Single authorized run; tag breakpoints frozen post-distribution audit.

Outputs:
  output/ema_context_daily.parquet         — full feature + tag table
  output/ema_context_daily_metadata.json   — frozen breakpoints + run meta
  output/ema_context_phase0_audit.md       — sanity/audit report

Acceptance (any FAIL → non-zero exit):
  - feature NaN rate < 2% per feature (post warm-up of 50 bars)
  - ATR vs non-ATR pair correlation <= 0.95 for all 4 paired feats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import intraday_1h  # noqa: E402
from ema_context import (  # noqa: E402
    compute_ema_features,
    EMA_PERIODS,
    assign_tags,
    fit_breakpoints,
    BREAKPOINTS_VERSION,
)

OUT_DIR = ROOT / "output"
PARQUET_OUT = OUT_DIR / "ema_context_daily.parquet"
META_OUT = OUT_DIR / "ema_context_daily_metadata.json"
AUDIT_OUT = OUT_DIR / "ema_context_phase0_audit.md"

DATA_SOURCE_USED = "extfeed_intraday_1h_3y_master.parquet [via intraday_1h.daily_resample]"
SOURCE_PRIORITY_HIT = 1  # horizontal_base canonical adapter

WARMUP_BARS = 50
NAN_RATE_THRESHOLD = 0.02
PAIR_CORR_ABORT_THRESHOLD = 0.95

NUMERIC_FEATURES = [
    "ema_stack_width_atr",
    "ema_stack_width_pct",
    "ema_distance_21_atr",
    "ema_distance_21_pct",
    "ema21_slope_5",
    "ema50_slope_10",
]
ATR_NONATR_PAIRS = [
    ("ema_stack_width_atr", "ema_stack_width_pct"),
    ("ema_distance_21_atr", "ema_distance_21_pct"),
]


def _drop_warmup(features: pd.DataFrame) -> pd.DataFrame:
    """Drop first WARMUP_BARS per ticker for NaN/correlation audit."""
    parts = []
    for _, g in features.groupby("ticker", sort=False):
        gs = g.sort_values("date")
        if len(gs) > WARMUP_BARS:
            parts.append(gs.iloc[WARMUP_BARS:])
    return pd.concat(parts, ignore_index=True) if parts else features.iloc[0:0]


def _nan_rates(features_postwarm: pd.DataFrame) -> dict[str, float]:
    n = len(features_postwarm)
    if n == 0:
        return {f: float("nan") for f in NUMERIC_FEATURES}
    return {f: float(features_postwarm[f].isna().sum()) / n for f in NUMERIC_FEATURES}


def _correlation_matrix(features_postwarm: pd.DataFrame) -> pd.DataFrame:
    return features_postwarm[NUMERIC_FEATURES].corr()


def _state_regime_crosstab(tagged: pd.DataFrame) -> pd.DataFrame:
    if "regime" not in tagged.columns:
        return pd.DataFrame()
    return pd.crosstab(tagged["ema_stack_state"], tagged["regime"], dropna=False)


def _attach_regime(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join RDP v1 regime labels by date."""
    regime = intraday_1h.load_regime()[["date", "regime"]].copy()
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    out = daily_features.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(regime, on="date", how="left")
    return out


def _format_audit(
    *,
    coverage_attempted: int,
    coverage_covered: int,
    n_rows: int,
    n_rows_postwarm: int,
    asof_min: pd.Timestamp,
    asof_max: pd.Timestamp,
    nan_rates: dict,
    corr_matrix: pd.DataFrame,
    crosstab: pd.DataFrame,
    breakpoints: dict,
    pair_corr_violations: list,
    nan_violations: list,
) -> str:
    lines = []
    lines.append("# ema_context Phase 0 audit\n")
    lines.append(f"Run date: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append(f"Spec: memory/ema_context_phase0_spec.md (LOCKED 2026-05-03)")
    lines.append(f"Data source: `{DATA_SOURCE_USED}`")
    lines.append(f"Source priority hit: {SOURCE_PRIORITY_HIT}")
    lines.append(f"EMA periods: {EMA_PERIODS}")
    lines.append(f"Breakpoints version: {breakpoints['version']}\n")

    lines.append("## Coverage")
    pct = (coverage_covered / coverage_attempted * 100.0) if coverage_attempted else 0.0
    lines.append(f"- Universe attempted: {coverage_attempted}")
    lines.append(f"- Universe covered:   {coverage_covered} ({pct:.1f}%)")
    lines.append(f"- Total rows (raw):    {n_rows:,}")
    lines.append(f"- Rows (post-warmup {WARMUP_BARS}d): {n_rows_postwarm:,}")
    lines.append(f"- Date range: {asof_min.date()} → {asof_max.date()}\n")

    lines.append("## Frozen tag breakpoints")
    for k, v in breakpoints.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## NaN rate per feature (post-warmup)")
    lines.append("| feature | nan_rate | threshold | status |")
    lines.append("|---|---|---|---|")
    for feat, rate in nan_rates.items():
        status = "PASS" if rate < NAN_RATE_THRESHOLD else "FAIL"
        lines.append(f"| `{feat}` | {rate:.4f} | <{NAN_RATE_THRESHOLD} | {status} |")
    lines.append("")

    lines.append("## ATR ↔ non-ATR pair correlation (G7 trap audit)")
    lines.append("| atr_feat | nonatr_feat | corr | abort_if | status |")
    lines.append("|---|---|---|---|---|")
    for atr_f, na_f in ATR_NONATR_PAIRS:
        c = corr_matrix.loc[atr_f, na_f]
        status = "PASS" if abs(c) <= PAIR_CORR_ABORT_THRESHOLD else "FAIL_SPEC_REVIEW"
        lines.append(f"| `{atr_f}` | `{na_f}` | {c:.4f} | >{PAIR_CORR_ABORT_THRESHOLD} | {status} |")
    lines.append("")

    lines.append("## Full feature correlation matrix")
    lines.append("```")
    lines.append(corr_matrix.round(3).to_string())
    lines.append("```\n")

    lines.append("## ema_stack_state × RDP regime cross-tab")
    if crosstab.empty:
        lines.append("_No regime labels available._")
    else:
        lines.append("```")
        lines.append(crosstab.to_string())
        lines.append("```")
    lines.append("")

    lines.append("## Acceptance verdict")
    if not nan_violations and not pair_corr_violations:
        lines.append("**ALL PASS** — Phase 0 acceptance met.")
    else:
        lines.append("**FAIL** — investigation required:")
        for v in nan_violations:
            lines.append(f"- NaN: `{v[0]}` rate {v[1]:.4f} >= {NAN_RATE_THRESHOLD}")
        for v in pair_corr_violations:
            lines.append(f"- CORR (SPEC REVIEW): `{v[0]}` ↔ `{v[1]}` = {v[2]:.4f} > {PAIR_CORR_ABORT_THRESHOLD}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[ema_context] loading 1h master via intraday_1h adapter …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    daily = intraday_1h.daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    coverage_attempted = bars["ticker"].nunique()
    coverage_covered = daily["ticker"].nunique()
    print(f"  bars: {len(bars):,}  daily: {len(daily):,}  "
          f"tickers: {coverage_covered}", flush=True)

    print(f"[ema_context] computing features (EMA {EMA_PERIODS}) …", flush=True)
    features = compute_ema_features(daily)
    features["data_source_used"] = DATA_SOURCE_USED
    features["source_priority_hit"] = SOURCE_PRIORITY_HIT
    print(f"  rows: {len(features):,}", flush=True)

    print(f"[ema_context] joining RDP v1 regime …", flush=True)
    features = _attach_regime(features)

    print(f"[ema_context] feature audit (post-warmup={WARMUP_BARS}) …", flush=True)
    postwarm = _drop_warmup(features)
    nan_rates = _nan_rates(postwarm)
    corr_matrix = _correlation_matrix(postwarm)

    nan_violations = [(f, r) for f, r in nan_rates.items() if r >= NAN_RATE_THRESHOLD]
    pair_corr_violations = []
    for atr_f, na_f in ATR_NONATR_PAIRS:
        c = float(corr_matrix.loc[atr_f, na_f])
        if abs(c) > PAIR_CORR_ABORT_THRESHOLD:
            pair_corr_violations.append((atr_f, na_f, c))

    print(f"[ema_context] fitting tag breakpoints (frozen {BREAKPOINTS_VERSION}) …", flush=True)
    breakpoints = fit_breakpoints(postwarm)

    print(f"[ema_context] assigning tags …", flush=True)
    tagged = assign_tags(features, breakpoints)
    crosstab = _state_regime_crosstab(tagged)

    print(f"[ema_context] writing parquet → {PARQUET_OUT.name} …", flush=True)
    tagged.to_parquet(PARQUET_OUT, index=False)

    meta = {
        "phase0_run_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "spec": "memory/ema_context_phase0_spec.md",
        "data_source_used": DATA_SOURCE_USED,
        "source_priority_hit": SOURCE_PRIORITY_HIT,
        "ema_periods": list(EMA_PERIODS),
        "universe_attempted": int(coverage_attempted),
        "universe_covered": int(coverage_covered),
        "n_rows": int(len(tagged)),
        "n_rows_postwarm": int(len(postwarm)),
        "tag_breakpoints": breakpoints,
        "warmup_bars": WARMUP_BARS,
        "acceptance": {
            "nan_threshold": NAN_RATE_THRESHOLD,
            "pair_corr_abort_threshold": PAIR_CORR_ABORT_THRESHOLD,
            "nan_rates": nan_rates,
            "atr_nonatr_pair_corr": {
                f"{a}__{b}": float(corr_matrix.loc[a, b]) for a, b in ATR_NONATR_PAIRS
            },
            "nan_violations": nan_violations,
            "pair_corr_violations": [
                {"atr": a, "non_atr": b, "corr": c} for a, b, c in pair_corr_violations
            ],
            "all_pass": not nan_violations and not pair_corr_violations,
        },
    }
    META_OUT.write_text(json.dumps(meta, indent=2, default=float))

    audit = _format_audit(
        coverage_attempted=coverage_attempted,
        coverage_covered=coverage_covered,
        n_rows=len(tagged),
        n_rows_postwarm=len(postwarm),
        asof_min=tagged["date"].min(),
        asof_max=tagged["date"].max(),
        nan_rates=nan_rates,
        corr_matrix=corr_matrix,
        crosstab=crosstab,
        breakpoints=breakpoints,
        pair_corr_violations=pair_corr_violations,
        nan_violations=nan_violations,
    )
    AUDIT_OUT.write_text(audit)

    elapsed = time.time() - t0
    print(f"[ema_context] done in {elapsed:.1f}s", flush=True)
    print(f"  parquet: {PARQUET_OUT}", flush=True)
    print(f"  metadata: {META_OUT}", flush=True)
    print(f"  audit:    {AUDIT_OUT}", flush=True)

    if nan_violations or pair_corr_violations:
        print(f"\n[FAIL] Phase 0 acceptance violation:", flush=True)
        for v in nan_violations:
            print(f"  - NaN rate `{v[0]}` = {v[1]:.4f} >= {NAN_RATE_THRESHOLD}", flush=True)
        for v in pair_corr_violations:
            print(f"  - SPEC REVIEW pair corr `{v[0]}` ↔ `{v[1]}` = {v[2]:.4f}", flush=True)
        return 1
    print(f"\n[PASS] Phase 0 acceptance met.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
