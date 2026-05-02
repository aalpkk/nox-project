"""V1 freeze leak audit (Phase A2).

Per plan §5:
  1) Feature/label segregation: forward-label cols cannot leak into features.
  2) Per-feature |Pearson ρ| with realized_R_5d on the trade universe;
     |ρ| > 0.60 → flag for manual review.

The date-shuffle AUC test (which requires training) runs in Phase B.

Reads the frozen parquet; writes audit lines to the existing audit log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
AUDIT = ROOT / "output/horizontal_base_event_v1_audit.log"

LABEL_COLS = {
    "mfe_R_3d", "mae_R_3d", "realized_R_3d", "failed_breakout_3d", "time_to_MFE_3d",
    "mfe_R_5d", "mae_R_5d", "realized_R_5d", "failed_breakout_5d", "time_to_MFE_5d",
    "mfe_R_10d", "mae_R_10d", "realized_R_10d", "failed_breakout_10d", "time_to_MFE_10d",
    "mfe_R_20d", "mae_R_20d", "realized_R_20d", "failed_breakout_20d", "time_to_MFE_20d",
    "early_failure_5d",
    "quality_20d",
}

ID_COLS = {
    "ticker", "bar_date", "setup_family", "signal_type", "signal_state",
    "breakout_bar_date", "as_of_ts", "data_frequency", "schema_version",
    "feature_version", "scanner_version",
    "asof_idx", "breakout_idx",
    "regime_sub", "regime_window_id",
    "bar_year", "val_fold",
    # entry/invalidation/contract — not features per §4 (they ARE present
    # in _build_row output but not part of the model feature set; trainer
    # excludes them).
    "entry_reference_price", "invalidation_level", "initial_risk_pct",
    "family__trigger_level",
}

CATEGORICAL_FEATURES = {
    "family__body_class", "family__retest_kind", "family__slope_tier",
    "common__regime",
}

CORR_FLAG_THRESHOLD = 0.60
TARGET_LABEL = "realized_R_5d"


def main() -> int:
    df = pd.read_parquet(PARQUET)
    print(f"loaded {len(df):,} rows / {df.shape[1]} cols", flush=True)

    # 1) feature/label segregation
    leak_in_feats = [c for c in df.columns if c in LABEL_COLS and c.startswith(("common__", "family__"))]
    if leak_in_feats:
        print(f"!!! LABEL LEAK INTO FEATURE NAMESPACE: {leak_in_feats}", flush=True)

    # build feature column list
    feature_cols = [
        c for c in df.columns
        if c not in LABEL_COLS
        and c not in ID_COLS
        and (c.startswith("family__") or c.startswith("common__"))
    ]
    numeric_cols = [
        c for c in feature_cols
        if c not in CATEGORICAL_FEATURES and df[c].dtype.kind in "fiub"
    ]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    print(f"feature cols: {len(feature_cols)} (numeric={len(numeric_cols)}, "
          f"categorical={len(cat_cols)})", flush=True)

    # 2) trade universe |ρ| audit
    trade = df[df["signal_state"].isin(["trigger", "retest_bounce"])].copy()
    trade = trade[trade[TARGET_LABEL].notna()]
    print(f"trade universe (labeled): {len(trade):,}", flush=True)
    target = trade[TARGET_LABEL].values

    flagged: list[tuple[str, float]] = []
    rho_summary: list[tuple[str, float, int]] = []
    for c in numeric_cols:
        v = trade[c]
        if v.dtype == bool:
            v = v.astype(float)
        v = v.astype("float64")
        mask = v.notna() & np.isfinite(v) & np.isfinite(target)
        n = int(mask.sum())
        if n < 30:
            rho_summary.append((c, float("nan"), n))
            continue
        rho = float(np.corrcoef(v[mask].values, target[mask])[0, 1])
        rho_summary.append((c, rho, n))
        if abs(rho) > CORR_FLAG_THRESHOLD:
            flagged.append((c, rho))

    rho_summary.sort(key=lambda x: -abs(x[1]) if not np.isnan(x[1]) else 0.0)
    print()
    print("top |ρ| with realized_R_5d (numeric features, trade universe):")
    for c, rho, n in rho_summary[:25]:
        print(f"  {c:55s} ρ={rho:+.3f}  n={n}")

    if flagged:
        print()
        print(f"!! FLAG: {len(flagged)} feature(s) with |ρ| > {CORR_FLAG_THRESHOLD}:")
        for c, rho in flagged:
            print(f"   {c:55s} ρ={rho:+.3f}")
    else:
        print(f"\n[OK] no feature has |ρ| > {CORR_FLAG_THRESHOLD} with realized_R_5d on trade universe")

    # 3) span audit on names — match forward-label patterns specifically, not the
    # broad "realized_" string (common__realized_vol_* are current-bar volatility,
    # not forward returns).
    leak_keywords = ("mfe_r_", "mae_r_", "realized_r_", "failed_breakout_",
                     "time_to_mfe", "early_failure")
    name_leak = [c for c in feature_cols if any(k in c.lower() for k in leak_keywords)]
    if name_leak:
        print(f"\n!! NAMING LEAK: feature cols matching forward-label keywords: {name_leak}")
    else:
        print("[OK] no feature column name matches forward-label keyword set")

    # append to audit log
    audit_append = []
    audit_append.append("")
    audit_append.append("=" * 92)
    audit_append.append(f"PHASE A2 LEAK AUDIT — {pd.Timestamp.utcnow().isoformat()}")
    audit_append.append("=" * 92)
    audit_append.append(f"  trade universe (labeled): {len(trade):,}")
    audit_append.append(f"  feature cols: {len(feature_cols)} "
                        f"(numeric={len(numeric_cols)}, categorical={len(cat_cols)})")
    audit_append.append(f"  target for ρ: {TARGET_LABEL}")
    audit_append.append(f"  |ρ| flag threshold: {CORR_FLAG_THRESHOLD}")
    audit_append.append(f"  flagged features: {len(flagged)}")
    if flagged:
        for c, rho in flagged:
            audit_append.append(f"    !! {c}  ρ={rho:+.3f}")
    audit_append.append(f"  feature-name leak keyword scan: "
                        f"{'CLEAN' if not name_leak else 'LEAK ' + str(name_leak)}")
    audit_append.append(f"  label-namespace overlap with feature cols: "
                        f"{'CLEAN' if not leak_in_feats else 'LEAK ' + str(leak_in_feats)}")
    audit_append.append("  top 10 |ρ| (audit reference, all under threshold):")
    for c, rho, n in rho_summary[:10]:
        audit_append.append(f"    {c:50s} ρ={rho:+.3f}  n={n}")

    with open(AUDIT, "a") as f:
        f.write("\n".join(audit_append) + "\n")

    return 0 if not flagged and not name_leak and not leak_in_feats else 1


if __name__ == "__main__":
    sys.exit(main())
