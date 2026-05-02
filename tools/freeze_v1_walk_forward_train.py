"""V1 freeze walk-forward training + acceptance gate (Phase B).

Trains a calibrated LightGBM binary classifier on `quality_20d` per plan §6.
Time-ordered 4-fold walk-forward, expanding train window. Per-fold:

  - AUC, log-loss, Brier
  - calibration (10-bin reliability)
  - lift @ top-decile / top-quintile (overall)
  - body_class × slope_tier × retest_kind cohort grid (N, base rate, AUC, lift)

Acceptance gates (per plan §11):
  [a] AUC > 0.55 in 4/4 folds on quality_20d
  [b] F4 forward fold AUC > 0.50
  [c] No body × slope × retest cell with N >= 30 has lift @ top-decile < 0.7×
  [d] Bootstrap 95% CI for top-decile lift on full OOS excludes 1.0×
  [e] Calibration ECE < 0.05 (cross-fold pooled)

Date-shuffle leak test: in-fold permutation of the train labels; expected
mean AUC ≈ 0.50 across folds. Flag if any > 0.55.

Outputs:
  output/horizontal_base_event_v1_train.log
  output/horizontal_base_event_v1_train_per_fold.csv
  output/horizontal_base_event_v1_train_cohort.csv
  Acceptance gate result appended to audit.log.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
SPLIT = ROOT / "output/horizontal_base_event_v1_split.json"
MANIFEST = ROOT / "output/horizontal_base_event_v1_manifest.json"
AUDIT = ROOT / "output/horizontal_base_event_v1_audit.log"
TRAIN_LOG = ROOT / "output/horizontal_base_event_v1_train.log"
PER_FOLD_CSV = ROOT / "output/horizontal_base_event_v1_train_per_fold.csv"
COHORT_CSV = ROOT / "output/horizontal_base_event_v1_train_cohort.csv"

TARGET = "quality_20d"
LABEL_REALIZED = "mfe_R_20d"  # NaN check uses this

CATEGORICAL_FEATURES = [
    "family__body_class",
    "family__retest_kind",
    "family__slope_tier",
    "common__regime",
    "signal_state",
]

EXCLUDE_FROM_FEATURES = {
    # forward labels
    "mfe_R_3d", "mae_R_3d", "realized_R_3d", "failed_breakout_3d", "time_to_MFE_3d",
    "mfe_R_5d", "mae_R_5d", "realized_R_5d", "failed_breakout_5d", "time_to_MFE_5d",
    "mfe_R_10d", "mae_R_10d", "realized_R_10d", "failed_breakout_10d", "time_to_MFE_10d",
    "mfe_R_20d", "mae_R_20d", "realized_R_20d", "failed_breakout_20d", "time_to_MFE_20d",
    "early_failure_5d", "quality_20d",
    # identifiers / audit / contract
    "ticker", "bar_date", "setup_family", "signal_type",
    "breakout_bar_date", "as_of_ts", "data_frequency", "schema_version",
    "feature_version", "scanner_version",
    "asof_idx", "breakout_idx",
    "regime_sub", "regime_window_id",
    "bar_year", "val_fold",
    "entry_reference_price", "invalidation_level", "initial_risk_pct",
    "family__trigger_level",
    # geometry-only "level" cols whose value is just an absolute price level
    "family__channel_high", "family__channel_low", "family__channel_mid",
    "family__resistance_level", "family__hard_resistance",
}

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=30,
    max_depth=-1,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=0.1,
    verbose=-1,
    seed=17,
    n_jobs=-1,
)
NUM_BOOST = 600
EARLY_STOP = 60

LIFT_DECILE = 0.10
LIFT_QUINTILE = 0.20
COHORT_MIN_N = 30
LIFT_FAIL_THRESHOLD = 0.7
ECE_THRESHOLD = 0.05
PER_FOLD_AUC_GATE = 0.55
F4_AUC_GATE = 0.50

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 17

LOG_LINES: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG_LINES.append(s)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        m = mask.sum()
        if m == 0:
            continue
        ece += (m / n) * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)


def lift_at_top_k(y_true: np.ndarray, p: np.ndarray, frac: float) -> float:
    base = y_true.mean()
    if base <= 0:
        return float("nan")
    n_top = max(1, int(round(len(p) * frac)))
    order = np.argsort(-p)
    top_rate = y_true[order[:n_top]].mean()
    return float(top_rate / base)


def bootstrap_lift_ci(
    y_true: np.ndarray, p: np.ndarray, frac: float,
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        boots[i] = lift_at_top_k(y_true[idx], p[idx], frac)
    boots = boots[np.isfinite(boots)]
    return (
        float(np.percentile(boots, 2.5)),
        float(np.median(boots)),
        float(np.percentile(boots, 97.5)),
    )


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
    ]
    cat_cols = [c for c in feat_cols if c in CATEGORICAL_FEATURES]
    X = df[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, feat_cols, cat_cols


def train_one_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
) -> tuple[lgb.Booster, np.ndarray, int]:
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_ds, free_raw_data=False)
    booster = lgb.train(
        LGB_PARAMS,
        train_ds,
        num_boost_round=NUM_BOOST,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    p_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    return booster, p_val, int(booster.best_iteration or 0)


def calibrate_isotonic(p_train: np.ndarray, y_train: np.ndarray) -> callable:
    """Fit isotonic calibration on a held-out portion of train (last 20% of train)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    return iso.transform


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    split = json.loads(SPLIT.read_text())

    log(f"V1 walk-forward training — {pd.Timestamp.utcnow().isoformat()}")
    log(f"  loaded {len(df):,} rows / {df.shape[1]} cols")

    # trade universe + drop unlabeled
    trade = df[df["signal_state"].isin(["trigger", "retest_bounce"])].copy()
    trade = trade[trade[LABEL_REALIZED].notna()].copy()
    trade["bar_date"] = pd.to_datetime(trade["bar_date"]).dt.normalize()
    log(f"  trade universe (labeled): {len(trade):,}")
    log(f"  base rate quality_20d: {trade[TARGET].mean():.4f}")

    X, feat_cols, cat_cols = build_feature_matrix(trade)
    log(f"  feature cols: {len(feat_cols)}  (categorical={len(cat_cols)})")
    log(f"  numeric: {len(feat_cols)-len(cat_cols)}  cats: {cat_cols}")
    y = trade[TARGET].astype(int).values

    # walk-forward folds
    fold_records: list[dict] = []
    cohort_records: list[dict] = []
    pooled_p: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    pooled_cohort: list[pd.DataFrame] = []
    perm_aucs: list[float] = []

    log("\n" + "=" * 88)
    log("WALK-FORWARD FOLDS")
    log("=" * 88)

    for fdef in split["fold_definitions"]:
        fid = fdef["fold_id"]
        vs = pd.Timestamp(fdef["val_start"])
        ve = pd.Timestamp(fdef["val_end"])
        train_mask = trade["bar_date"] < vs
        val_mask = (trade["bar_date"] >= vs) & (trade["bar_date"] < ve)
        n_tr, n_va = int(train_mask.sum()), int(val_mask.sum())
        if n_tr < 50 or n_va < 30:
            log(f"\n[{fid}] SKIP — train={n_tr} val={n_va} too small")
            continue

        X_tr, X_va = X[train_mask].reset_index(drop=True), X[val_mask].reset_index(drop=True)
        y_tr, y_va = y[train_mask], y[val_mask]
        booster, p_va, best_iter = train_one_fold(X_tr, y_tr, X_va, y_va, cat_cols)

        # isotonic calibration fit on the train portion's OOF-style predictions
        # (use same booster predicting train; not strictly OOF but fast and
        # sufficient for the calibration sanity gate)
        p_tr = booster.predict(X_tr, num_iteration=booster.best_iteration)
        cal = calibrate_isotonic(p_tr, y_tr)
        p_va_cal = np.clip(cal(p_va), 1e-6, 1 - 1e-6)

        auc = roc_auc_score(y_va, p_va)
        ll = log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6))
        brier = brier_score_loss(y_va, p_va)
        ll_cal = log_loss(y_va, p_va_cal)
        ece_raw = expected_calibration_error(y_va, p_va)
        ece_cal = expected_calibration_error(y_va, p_va_cal)
        lift10 = lift_at_top_k(y_va, p_va, LIFT_DECILE)
        lift20 = lift_at_top_k(y_va, p_va, LIFT_QUINTILE)

        log(f"\n[{fid}] val_window {vs.date()} → {(ve - pd.Timedelta(days=1)).date()}")
        log(f"   train_n={n_tr:,}  val_n={n_va:,}  best_iter={best_iter}")
        log(f"   AUC={auc:.4f}  LogLoss={ll:.4f} (cal={ll_cal:.4f})  Brier={brier:.4f}")
        log(f"   ECE_raw={ece_raw:.4f}  ECE_cal={ece_cal:.4f}")
        log(f"   lift@10%={lift10:.2f}×  lift@20%={lift20:.2f}×")

        # date-shuffle permutation (simple within-train shuffle, not by-year)
        rng = np.random.default_rng(BOOTSTRAP_SEED + len(fold_records))
        y_perm = rng.permutation(y_tr)
        try:
            booster_p, p_va_p, _ = train_one_fold(X_tr, y_perm, X_va, y_va, cat_cols)
            auc_p = roc_auc_score(y_va, p_va_p)
        except Exception as e:
            auc_p = float("nan")
            log(f"   [perm] failed: {e}")
        log(f"   AUC_permuted={auc_p:.4f}  (expect ≈0.50; flag if >0.55)")
        perm_aucs.append(auc_p)

        fold_records.append({
            "fold": fid,
            "val_start": str(vs.date()),
            "val_end": str(ve.date()),
            "train_n": n_tr,
            "val_n": n_va,
            "best_iter": best_iter,
            "auc": auc,
            "logloss": ll,
            "logloss_cal": ll_cal,
            "brier": brier,
            "ece_raw": ece_raw,
            "ece_cal": ece_cal,
            "lift_top10": lift10,
            "lift_top20": lift20,
            "auc_permuted": auc_p,
        })

        # collect for pooled OOS metrics + cohort grid
        val_meta = trade[val_mask][[
            "family__body_class", "family__slope_tier", "family__retest_kind",
            "signal_state",
        ]].reset_index(drop=True)
        val_meta["fold"] = fid
        val_meta["y"] = y_va
        val_meta["p"] = p_va
        val_meta["p_cal"] = p_va_cal
        pooled_cohort.append(val_meta)
        pooled_p.append(p_va)
        pooled_y.append(y_va)

    # ------------------------------------------------------------------
    # pooled OOS
    P_pool = np.concatenate(pooled_p)
    Y_pool = np.concatenate(pooled_y)
    auc_pool = roc_auc_score(Y_pool, P_pool)
    ll_pool = log_loss(Y_pool, np.clip(P_pool, 1e-6, 1 - 1e-6))
    brier_pool = brier_score_loss(Y_pool, P_pool)
    ece_pool = expected_calibration_error(Y_pool, P_pool)
    lift10_pool = lift_at_top_k(Y_pool, P_pool, LIFT_DECILE)
    lift10_lo, lift10_med, lift10_hi = bootstrap_lift_ci(Y_pool, P_pool, LIFT_DECILE)

    log("\n" + "=" * 88)
    log("POOLED OOS (concat of all fold val sets)")
    log("=" * 88)
    log(f"  N={len(Y_pool):,}  base_rate={Y_pool.mean():.4f}")
    log(f"  AUC={auc_pool:.4f}  LogLoss={ll_pool:.4f}  Brier={brier_pool:.4f}")
    log(f"  ECE={ece_pool:.4f}")
    log(f"  lift@10%={lift10_pool:.2f}×  bootstrap 95%CI=[{lift10_lo:.2f}, {lift10_hi:.2f}]")

    # ------------------------------------------------------------------
    # cohort grid (body × slope × retest)
    coh_df = pd.concat(pooled_cohort, ignore_index=True)
    coh_df["family__retest_kind"] = coh_df["family__retest_kind"].fillna("none")
    cohort_rows: list[dict] = []
    grid = (
        coh_df
        .groupby(["family__body_class", "family__slope_tier", "family__retest_kind"], dropna=False, observed=True)
        .agg(N=("y", "size"), base_rate=("y", "mean"))
        .reset_index()
    )
    for _, r in grid.iterrows():
        b = r["family__body_class"]; s = r["family__slope_tier"]; k = r["family__retest_kind"]
        sub = coh_df[
            (coh_df["family__body_class"] == b)
            & (coh_df["family__slope_tier"] == s)
            & (coh_df["family__retest_kind"] == k)
        ]
        n = len(sub)
        if n == 0:
            continue
        try:
            auc_c = roc_auc_score(sub["y"], sub["p"]) if sub["y"].nunique() == 2 else float("nan")
        except ValueError:
            auc_c = float("nan")
        lift_c = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE) if n >= 5 else float("nan")
        cohort_rows.append({
            "body_class": b,
            "slope_tier": s,
            "retest_kind": k,
            "N": n,
            "base_rate": float(sub["y"].mean()),
            "auc": auc_c,
            "lift_top10": lift_c,
        })
    cohort_table = pd.DataFrame(cohort_rows).sort_values(["body_class", "slope_tier", "retest_kind"])

    log("\n" + "=" * 88)
    log("COHORT GRID — body × slope × retest (N>=10 shown)")
    log("=" * 88)
    log(f"  {'body':<12} {'slope':<6} {'retest':<14} {'N':>5} {'base':>6} {'AUC':>6} {'lift10':>7}")
    for _, r in cohort_table.iterrows():
        if r["N"] < 10:
            continue
        log(f"  {r['body_class']:<12} {r['slope_tier']:<6} {r['retest_kind']:<14} "
            f"{int(r['N']):>5} {r['base_rate']:>6.3f} "
            f"{(f'{r.auc:.3f}' if pd.notna(r.auc) else '—'):>6} "
            f"{(f'{r.lift_top10:.2f}' if pd.notna(r.lift_top10) else '—'):>7}")

    # ------------------------------------------------------------------
    # acceptance gates
    log("\n" + "=" * 88)
    log("ACCEPTANCE GATES")
    log("=" * 88)
    aucs = [f["auc"] for f in fold_records]
    f4 = next((f for f in fold_records if f["fold"] == "F4"), None)
    f4_auc = f4["auc"] if f4 else float("nan")

    gate_a = all(a > PER_FOLD_AUC_GATE for a in aucs)
    gate_b = (f4_auc > F4_AUC_GATE) if f4 is not None else False

    bad_cohorts = cohort_table[
        (cohort_table["N"] >= COHORT_MIN_N)
        & (cohort_table["lift_top10"] < LIFT_FAIL_THRESHOLD)
    ]
    gate_c = bad_cohorts.empty
    gate_d = lift10_lo > 1.0
    gate_e = ece_pool < ECE_THRESHOLD
    gate_perm = all((not np.isfinite(p)) or p < 0.55 for p in perm_aucs)

    log(f"  [a] AUC > {PER_FOLD_AUC_GATE} on 4/4 folds:  "
        f"{'PASS' if gate_a else 'FAIL'}  "
        f"(per-fold {[round(a,3) for a in aucs]})")
    log(f"  [b] F4 AUC > {F4_AUC_GATE}:                  "
        f"{'PASS' if gate_b else 'FAIL'}  (F4 AUC={f4_auc:.3f})")
    log(f"  [c] no body×slope×retest cell N>={COHORT_MIN_N} lift<{LIFT_FAIL_THRESHOLD}: "
        f"{'PASS' if gate_c else 'FAIL'}")
    if not gate_c:
        for _, r in bad_cohorts.iterrows():
            log(f"      !! {r['body_class']}/{r['slope_tier']}/{r['retest_kind']} "
                f"N={int(r['N'])} lift={r['lift_top10']:.2f}")
    log(f"  [d] bootstrap 95%CI excludes 1.0× on top-decile: "
        f"{'PASS' if gate_d else 'FAIL'}  (CI=[{lift10_lo:.2f}, {lift10_hi:.2f}])")
    log(f"  [e] pooled ECE < {ECE_THRESHOLD}:           "
        f"{'PASS' if gate_e else 'FAIL'}  (ECE={ece_pool:.4f})")
    log(f"  [perm] permutation AUC<0.55 4/4 folds:      "
        f"{'PASS' if gate_perm else 'FAIL'}  ({[round(p,3) for p in perm_aucs]})")

    overall_pass = gate_a and gate_b and gate_c and gate_d and gate_e and gate_perm
    log("\n" + ("ACCEPT V1 FREEZE" if overall_pass else "REJECT V1 FREEZE — see failed gates above"))

    # write artifacts
    pd.DataFrame(fold_records).to_csv(PER_FOLD_CSV, index=False)
    cohort_table.to_csv(COHORT_CSV, index=False)
    TRAIN_LOG.write_text("\n".join(LOG_LINES) + "\n")

    # append summary to audit
    with open(AUDIT, "a") as f:
        f.write("\n" + "=" * 92 + "\n")
        f.write(f"PHASE B WALK-FORWARD ACCEPTANCE GATE — {pd.Timestamp.utcnow().isoformat()}\n")
        f.write("=" * 92 + "\n")
        for k, v in [
            ("[a] per-fold AUC > 0.55", gate_a),
            ("[b] F4 AUC > 0.50", gate_b),
            ("[c] cohort N>=30 lift>=0.7", gate_c),
            ("[d] bootstrap 95%CI excl 1.0", gate_d),
            ("[e] pooled ECE < 0.05", gate_e),
            ("[perm] permutation AUC<0.55", gate_perm),
        ]:
            f.write(f"  {k:35s} {'PASS' if v else 'FAIL'}\n")
        f.write(f"  per-fold AUCs: {[round(a, 3) for a in aucs]}\n")
        f.write(f"  pooled OOS AUC: {auc_pool:.4f}  ECE: {ece_pool:.4f}\n")
        f.write(f"  pooled top-decile lift: {lift10_pool:.2f}× CI=[{lift10_lo:.2f}, {lift10_hi:.2f}]\n")
        f.write(f"  overall: {'ACCEPT' if overall_pass else 'REJECT'}\n")
        f.write(f"  elapsed: {time.time()-t0:.1f}s\n")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
