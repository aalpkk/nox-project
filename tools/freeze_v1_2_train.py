"""V1_2 freeze walk-forward training + acceptance gates (LOCKED SPEC).

Implements the locked v1_2 pre-registration spec at
`memory/scanner_v1_2_extreme_ranker_spec.md`.

Differences from v1 (`freeze_v1_walk_forward_train.py`):

  - Universe: trigger ∪ retest_bounce AND family__breakout_age <= 5  (N=1,854)
  - Target:   y = (mfe_pct_5d >= 5%)  with mfe_pct_5d = mfe_R_5d * atr_pct * 100
              (PCT-based — structural defense vs v1's R-based denominator leak)
  - Features: v1 features + RDP regime (joined from regime_labels_daily_rdp_v1.csv)
              + index_ret_5d/20d + vwap10_vs_vwap52 (already in parquet).
              Legacy `common__regime` (SMA-cross) DROPPED.
  - Folds:    4-fold WF, identical fold boundaries to v1's split.json.
  - Gates:    All evaluated on POOLED OOS (per-fold AUC advisory only):
              G1 pooled AUC > 0.62, lower 95% CI > 0.58
              G2 pooled top-decile lift on broad target ≥ 1.8×, CI ≥ 1.3×
              G3 pooled top-decile lift on narrow gate (≥10%/10d) ≥ 1.5×, CI ≥ 1.0×
              G4 2026Q2 sub-pool AUC ≥ 0.50  (deferred if N<30)
              G5 multi-seed (10) permutation: perm_dist mean+1·SE < model AUC − 0.05
              G6 cell floor: every body_class × signal_state × retest_kind cell
                 with N≥30 AND N×base≥1.0 has top-decile lift ≥ 0.7×
              G7 single-feature dominance: full-model OOS AUC > best-numeric-
                 single-feature OOS AUC + 0.03   (mb_scanner trap defense)

Anti-rescue: any G1-G7 fail → REJECT; no post-hoc tweaks.

Outputs:
  output/horizontal_base_event_v1_2_train.log
  output/horizontal_base_event_v1_2_predictions.parquet
  output/horizontal_base_event_v1_2_train_per_fold.csv
  output/horizontal_base_event_v1_2_train_cohort.csv
  output/horizontal_base_event_v1_2_perm_distribution.csv
  output/horizontal_base_event_v1_2_single_feature_auc.csv
  output/horizontal_base_event_v1_2_acceptance.md

Mode:
  --dry-run        only validate inputs (manifest, universe count, fold positives,
                   feature joins) and print the integrity report; no training.
  (default)        full single-fire execution against locked spec.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
SPLIT = ROOT / "output/horizontal_base_event_v1_split.json"
RDP_CSV = ROOT / "output/regime_labels_daily_rdp_v1.csv"

OUT_LOG = ROOT / "output/horizontal_base_event_v1_2_train.log"
OUT_PRED = ROOT / "output/horizontal_base_event_v1_2_predictions.parquet"
OUT_PER_FOLD = ROOT / "output/horizontal_base_event_v1_2_train_per_fold.csv"
OUT_COHORT = ROOT / "output/horizontal_base_event_v1_2_train_cohort.csv"
OUT_PERM = ROOT / "output/horizontal_base_event_v1_2_perm_distribution.csv"
OUT_SF_AUC = ROOT / "output/horizontal_base_event_v1_2_single_feature_auc.csv"
OUT_ACCEPT = ROOT / "output/horizontal_base_event_v1_2_acceptance.md"

# ---------- spec constants (DO NOT change without re-locking spec) ------------
TARGET_HORIZON_DAYS = 5
TARGET_THRESHOLD_PCT = 5.0
NARROW_GATE_HORIZON_DAYS = 10
NARROW_GATE_THRESHOLD_PCT = 10.0

UNIVERSE_AGE_MAX = 5
UNIVERSE_SIGNALS = ("trigger", "retest_bounce")

LIFT_DECILE = 0.10
COHORT_MIN_N = 30
COHORT_LIFT_FLOOR = 0.7
N_BASE_PRODUCT_FLOOR = 1.0

# Gate thresholds
G1_AUC_THRESHOLD = 0.62
G1_CI_LOWER_THRESHOLD = 0.58
G2_LIFT_THRESHOLD = 1.8
G2_CI_LOWER = 1.3
G3_LIFT_THRESHOLD = 1.5
G3_CI_LOWER = 1.0
G4_SUB_AUC_THRESHOLD = 0.50
G4_MIN_N = 30
G5_PERM_SEEDS = 10
G5_PERM_MARGIN = 0.05
G7_DOMINANCE_MARGIN = 0.03

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 17
PRIMARY_SEED = 42

# 2026Q2 sub-pool window
Q2_2026_START = pd.Timestamp("2026-04-01")
Q2_2026_END = pd.Timestamp("2026-06-30")

# ---------- feature spec ------------------------------------------------------
CATEGORICAL_FEATURES = [
    "family__body_class",
    "family__retest_kind",
    "family__slope_tier",
    "regime_rdp",          # joined from RDP CSV (replaces common__regime)
    "signal_state",
]

EXCLUDE_FROM_FEATURES = {
    # forward labels
    "mfe_R_3d", "mae_R_3d", "realized_R_3d", "failed_breakout_3d", "time_to_MFE_3d",
    "mfe_R_5d", "mae_R_5d", "realized_R_5d", "failed_breakout_5d", "time_to_MFE_5d",
    "mfe_R_10d", "mae_R_10d", "realized_R_10d", "failed_breakout_10d", "time_to_MFE_10d",
    "mfe_R_20d", "mae_R_20d", "realized_R_20d", "failed_breakout_20d", "time_to_MFE_20d",
    "early_failure_5d", "quality_20d",
    # post-target derived (label leakage)
    "mfe_pct_5d", "mfe_pct_10d", "mfe_pct_20d",
    "y_train", "y_narrow",
    # identifiers / audit / contract
    "ticker", "bar_date", "setup_family", "signal_type",
    "breakout_bar_date", "as_of_ts", "data_frequency", "schema_version",
    "feature_version", "scanner_version",
    "asof_idx", "breakout_idx",
    "regime_sub", "regime_window_id",
    "bar_year", "val_fold",
    "entry_reference_price", "invalidation_level", "initial_risk_pct",
    "family__trigger_level",
    # geometry-only "level" cols
    "family__channel_high", "family__channel_low", "family__channel_mid",
    "family__resistance_level", "family__hard_resistance",
    # legacy regime (we replace with RDP-joined)
    "common__regime",
}

# ---------- LightGBM HPs (locked, identical to v1) ----------------------------
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
    seed=PRIMARY_SEED,
    n_jobs=-1,
)
NUM_BOOST = 600
EARLY_STOP = 60

LOG_LINES: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG_LINES.append(s)


# ---------- data prep ---------------------------------------------------------
def load_and_prepare() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(PARQUET)
    split = json.loads(SPLIT.read_text())

    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.normalize()

    # %-based MFE columns (target uses these)
    atr_x100 = df["common__atr_pct"] * 100.0
    df["mfe_pct_5d"] = df["mfe_R_5d"] * atr_x100
    df["mfe_pct_10d"] = df["mfe_R_10d"] * atr_x100
    df["mfe_pct_20d"] = df["mfe_R_20d"] * atr_x100

    # Universe filter (Spec §1)
    keep = df["signal_state"].isin(UNIVERSE_SIGNALS)
    keep &= df["family__breakout_age"] <= UNIVERSE_AGE_MAX
    keep &= df["mfe_R_5d"].notna()
    keep &= df["mfe_R_10d"].notna()  # need narrow gate label too

    trade = df[keep].copy()

    # Targets
    trade["y_train"] = (trade["mfe_pct_5d"] >= TARGET_THRESHOLD_PCT).astype(int)
    trade["y_narrow"] = (trade["mfe_pct_10d"] >= NARROW_GATE_THRESHOLD_PCT).astype(int)

    # RDP regime join
    rdp = pd.read_csv(RDP_CSV)
    rdp["date"] = pd.to_datetime(rdp["date"]).dt.normalize()
    rdp_map = rdp[["date", "regime"]].rename(
        columns={"date": "bar_date", "regime": "regime_rdp"}
    )
    n_before = len(trade)
    trade = trade.merge(rdp_map, on="bar_date", how="left")
    n_after = len(trade)
    assert n_before == n_after, f"join changed row count: {n_before} → {n_after}"

    # missing regime → "neutral" (RDP labels start 2023-01-19, our universe starts 2023-01-02)
    miss = trade["regime_rdp"].isna()
    if miss.any():
        log(f"  RDP regime missing on {miss.sum()} rows (pre-2023-01-19 warmup) — filling 'neutral'")
        trade.loc[miss, "regime_rdp"] = "neutral"
    trade["regime_rdp"] = trade["regime_rdp"].astype("category")

    return trade, split


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    cat_cols = [c for c in feat_cols if c in CATEGORICAL_FEATURES]
    X = df[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, feat_cols, cat_cols


# ---------- core training -----------------------------------------------------
def train_one_fold(X_tr, y_tr, X_va, y_va, cat_cols, seed=PRIMARY_SEED):
    params = dict(LGB_PARAMS)
    params["seed"] = seed
    train_ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    val_ds = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=train_ds, free_raw_data=False)
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=NUM_BOOST,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, p_va, int(booster.best_iteration or 0)


def lift_at_top(y, p, frac=LIFT_DECILE):
    base = y.mean()
    if base <= 0:
        return float("nan")
    n_top = max(1, int(round(len(p) * frac)))
    order = np.argsort(-p)
    return float(y[order[:n_top]].mean() / base)


def bootstrap_ci(y, p, frac=LIFT_DECILE, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        boots[i] = lift_at_top(y[idx], p[idx], frac)
    boots = boots[np.isfinite(boots)]
    return float(np.percentile(boots, 2.5)), float(np.median(boots)), float(np.percentile(boots, 97.5))


def bootstrap_auc_ci(y, p, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        try:
            boots[i] = roc_auc_score(y[idx], p[idx])
        except ValueError:
            boots[i] = np.nan
    boots = boots[np.isfinite(boots)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------- G7 single-feature dominance ---------------------------------------
def best_single_feature_auc(X_pool: pd.DataFrame, y_pool: np.ndarray,
                            feat_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, float, str]:
    """For each numeric feature: pooled OOS AUC of using the feature value
    directly as a score (taking max with negation). Categoricals excluded from
    G7 — denominator/single-feature traps are a numeric-feature pathology.
    Returns (table, max_auc, max_feat)."""
    rows = []
    for c in feat_cols:
        if c in cat_cols:
            continue
        x = X_pool[c]
        try:
            x = pd.to_numeric(x, errors="coerce")
        except Exception:
            continue
        x = x.values.astype(np.float64)
        if not np.isfinite(x).any():
            continue
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        if np.unique(x).size < 5:
            # not enough variation
            continue
        try:
            a_pos = roc_auc_score(y_pool, x)
            a_neg = roc_auc_score(y_pool, -x)
        except ValueError:
            continue
        a = max(a_pos, a_neg)
        sign = "+" if a_pos >= a_neg else "-"
        rows.append({"feature": c, "auc_signed": a, "direction": sign})
    df = pd.DataFrame(rows).sort_values("auc_signed", ascending=False).reset_index(drop=True)
    if df.empty:
        return df, 0.0, "<none>"
    return df, float(df.iloc[0]["auc_signed"]), str(df.iloc[0]["feature"])


# ---------- dry-run integrity report ------------------------------------------
def dry_run(trade: pd.DataFrame, split: dict, feat_cols: list[str], cat_cols: list[str]) -> int:
    log("=" * 88)
    log("V1_2 DRY-RUN INTEGRITY CHECKS")
    log("=" * 88)
    log(f"event parquet:   {PARQUET}")
    log(f"split json:      {SPLIT}")
    log(f"rdp regime csv:  {RDP_CSV}")

    log(f"\nUniverse (trigger ∪ retest_bounce ∧ breakout_age ≤ {UNIVERSE_AGE_MAX}):")
    log(f"  N = {len(trade):,}    expected ≈ 1,854 (spec §1)")

    sig_counts = trade["signal_state"].value_counts().to_dict()
    log(f"  by signal_state: {sig_counts}")

    age_counts = trade["family__breakout_age"].value_counts().sort_index().to_dict()
    log(f"  by breakout_age: {age_counts}")

    log(f"\nTarget: mfe_pct_5d ≥ {TARGET_THRESHOLD_PCT}%  ( = mfe_R_5d × atr_pct × 100 )")
    pos = int(trade["y_train"].sum())
    base = trade["y_train"].mean()
    log(f"  positives: {pos:,}  base rate: {base*100:.2f}%   (spec table: 204 / 11.00%)")

    log(f"\nNarrow gate target (G3): mfe_pct_10d ≥ {NARROW_GATE_THRESHOLD_PCT}%")
    pos_n = int(trade["y_narrow"].sum())
    base_n = trade["y_narrow"].mean()
    log(f"  positives: {pos_n:,}  base rate: {base_n*100:.2f}%")

    log(f"\nPer-fold VAL positives (broad target):")
    folds = split["fold_definitions"]
    for f in folds:
        s, e = pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"])
        m = (trade["bar_date"] >= s) & (trade["bar_date"] < e)
        p = int((m & (trade["y_train"] == 1)).sum())
        n = int(m.sum())
        log(f"  {f['fold_id']}  {s.date()} → {e.date()}   N={n:>4}  pos={p:>3}")

    log(f"\nFeature matrix:")
    log(f"  total features: {len(feat_cols)}")
    log(f"  categorical:    {len(cat_cols)}  → {cat_cols}")
    log(f"  numeric:        {len(feat_cols) - len(cat_cols)}")

    new_feats = [
        "common__index_ret_5d",
        "common__index_ret_20d",
        "common__vwap10_vs_vwap52",
        "regime_rdp",
    ]
    log(f"\nKey discovery-driven features present?")
    for c in new_feats:
        present = c in feat_cols
        log(f"  {c:<35}  {'✓' if present else '✗ MISSING'}")

    log(f"\nLeakage check — forward labels in feature list?")
    leak_cols = [c for c in feat_cols if any(s in c for s in ["mfe_R_", "mae_R_", "realized_R_", "failed_breakout_", "mfe_pct_"])]
    if leak_cols:
        log(f"  FAIL: leak cols present: {leak_cols}")
        return 1
    log(f"  PASS — none in feature list")

    log(f"\nRDP regime distribution in universe:")
    log(f"  {trade['regime_rdp'].value_counts().to_dict()}")

    log(f"\n2026Q2 sub-pool size (G4 deferment check):")
    q2 = trade[(trade["bar_date"] >= Q2_2026_START) & (trade["bar_date"] <= Q2_2026_END)]
    log(f"  N = {len(q2)}    threshold for non-deferred: ≥ {G4_MIN_N}")

    log(f"\n" + "=" * 88)
    log("Dry-run integrity OK. Re-run without --dry-run to execute single fire.")
    log("=" * 88)
    return 0


# ---------- main --------------------------------------------------------------
def main(dry_run_only: bool = False) -> int:
    t0 = time.time()
    log(f"V1_2 walk-forward training — {pd.Timestamp.utcnow().isoformat()}")
    log(f"  spec: memory/scanner_v1_2_extreme_ranker_spec.md (LOCKED)")
    log(f"  primary seed: {PRIMARY_SEED}")
    log("")

    trade, split = load_and_prepare()
    X, feat_cols, cat_cols = build_feature_matrix(trade)

    if dry_run_only:
        rc = dry_run(trade, split, feat_cols, cat_cols)
        OUT_LOG.write_text("\n".join(LOG_LINES) + "\n")
        return rc

    # full integrity preview before fire
    dry_run(trade, split, feat_cols, cat_cols)

    y = trade["y_train"].astype(int).values
    y_narrow = trade["y_narrow"].astype(int).values

    log("\n" + "=" * 88)
    log("WALK-FORWARD FOLDS — primary fit")
    log("=" * 88)

    fold_records: list[dict] = []
    pooled = {"p": [], "y": [], "y_narrow": [], "fold_id": [], "row_idx": []}

    for fdef in split["fold_definitions"]:
        fid = fdef["fold_id"]
        vs = pd.Timestamp(fdef["val_start"])
        ve = pd.Timestamp(fdef["val_end"])
        train_mask = (trade["bar_date"] < vs).values
        val_mask = ((trade["bar_date"] >= vs) & (trade["bar_date"] < ve)).values
        n_tr, n_va = int(train_mask.sum()), int(val_mask.sum())
        if n_tr < 50 or n_va < 30:
            log(f"\n[{fid}] SKIP — train={n_tr} val={n_va} too small")
            continue

        X_tr = X.iloc[train_mask].reset_index(drop=True)
        X_va = X.iloc[val_mask].reset_index(drop=True)
        y_tr, y_va = y[train_mask], y[val_mask]
        y_va_narrow = y_narrow[val_mask]

        booster, p_va, best_iter = train_one_fold(X_tr, y_tr, X_va, y_va, cat_cols, seed=PRIMARY_SEED)

        try:
            auc = roc_auc_score(y_va, p_va)
        except ValueError:
            auc = float("nan")
        ll = log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6)) if y_va.sum() > 0 else float("nan")
        brier = brier_score_loss(y_va, p_va)
        lift10 = lift_at_top(y_va, p_va, LIFT_DECILE)

        log(f"\n[{fid}] val_window {vs.date()} → {(ve - pd.Timedelta(days=1)).date()}")
        log(f"   train_n={n_tr:,}  val_n={n_va:,}  best_iter={best_iter}")
        log(f"   AUC={auc:.4f}  LogLoss={ll:.4f}  Brier={brier:.4f}  lift@10%={lift10:.2f}×")

        fold_records.append({
            "fold": fid,
            "val_start": str(vs.date()),
            "val_end": str(ve.date()),
            "train_n": n_tr,
            "val_n": n_va,
            "best_iter": best_iter,
            "auc": auc,
            "logloss": ll,
            "brier": brier,
            "lift_top10_broad": lift10,
            "n_pos_broad": int(y_va.sum()),
            "n_pos_narrow": int(y_va_narrow.sum()),
        })

        pooled["p"].extend(p_va.tolist())
        pooled["y"].extend(y_va.tolist())
        pooled["y_narrow"].extend(y_va_narrow.tolist())
        pooled["fold_id"].extend([fid] * len(p_va))
        # capture original event-row indices (within trade)
        original_idx = np.flatnonzero(val_mask)
        pooled["row_idx"].extend(original_idx.tolist())

    if not pooled["p"]:
        log("FATAL: no folds produced predictions"); return 1

    # ----- pooled OOS metrics -----
    P = np.array(pooled["p"])
    Y = np.array(pooled["y"])
    Y_n = np.array(pooled["y_narrow"])
    F = np.array(pooled["fold_id"])
    R = np.array(pooled["row_idx"])

    log("\n" + "=" * 88)
    log("POOLED OOS METRICS")
    log("=" * 88)
    auc_pool = roc_auc_score(Y, P)
    auc_lo, auc_hi = bootstrap_auc_ci(Y, P)
    base_pool = Y.mean()
    base_narrow = Y_n.mean()
    lift_b = lift_at_top(Y, P, LIFT_DECILE)
    lb_lo, lb_med, lb_hi = bootstrap_ci(Y, P, LIFT_DECILE)
    lift_n = lift_at_top(Y_n, P, LIFT_DECILE)
    ln_lo, ln_med, ln_hi = bootstrap_ci(Y_n, P, LIFT_DECILE)

    log(f"  N_pooled = {len(Y):,}   broad pos = {int(Y.sum())} ({base_pool*100:.2f}%)   narrow pos = {int(Y_n.sum())} ({base_narrow*100:.2f}%)")
    log(f"  pooled AUC = {auc_pool:.4f}    bootstrap 95% CI = [{auc_lo:.4f}, {auc_hi:.4f}]")
    log(f"  pooled top-decile lift (broad ≥5%/5d):  {lift_b:.2f}×   CI [{lb_lo:.2f}, {lb_hi:.2f}]")
    log(f"  pooled top-decile lift (narrow ≥10%/10d): {lift_n:.2f}×   CI [{ln_lo:.2f}, {ln_hi:.2f}]")

    # ----- G4: 2026Q2 sub-pool -----
    bar_dates = trade.iloc[R]["bar_date"].values
    bar_dates = pd.to_datetime(bar_dates)
    q2_mask = (bar_dates >= Q2_2026_START) & (bar_dates <= Q2_2026_END)
    q2_n = int(q2_mask.sum())
    q2_pos = int(Y[q2_mask].sum()) if q2_n > 0 else 0
    if q2_n >= G4_MIN_N and q2_pos > 0 and q2_pos < q2_n:
        try:
            q2_auc = roc_auc_score(Y[q2_mask], P[q2_mask])
        except ValueError:
            q2_auc = float("nan")
        q4_deferred = False
    else:
        q2_auc = float("nan")
        q4_deferred = True
    log(f"\n  2026Q2 sub-pool: N={q2_n} pos={q2_pos}   AUC={'deferred' if q4_deferred else f'{q2_auc:.4f}'}")

    # ----- G5: multi-seed permutation -----
    log(f"\n" + "=" * 88)
    log(f"G5: PERMUTATION DISTRIBUTION ({G5_PERM_SEEDS} seeds)")
    log(f"=" * 88)
    perm_aucs = []
    perm_records = []
    for seed_i in range(1, G5_PERM_SEEDS + 1):
        rng = np.random.default_rng(1000 + seed_i)
        # permute labels within trade universe; refit per fold; compute pooled AUC
        y_perm = rng.permutation(y).copy()
        pp_p, pp_y = [], []
        for fdef in split["fold_definitions"]:
            vs = pd.Timestamp(fdef["val_start"]); ve = pd.Timestamp(fdef["val_end"])
            tr_m = (trade["bar_date"] < vs).values
            va_m = ((trade["bar_date"] >= vs) & (trade["bar_date"] < ve)).values
            if tr_m.sum() < 50 or va_m.sum() < 30:
                continue
            X_tr = X.iloc[tr_m].reset_index(drop=True)
            X_va = X.iloc[va_m].reset_index(drop=True)
            y_tr_p = y_perm[tr_m]
            y_va_p = y_perm[va_m]
            try:
                _, p_va_p, _ = train_one_fold(X_tr, y_tr_p, X_va, y_va_p, cat_cols, seed=PRIMARY_SEED)
            except Exception as e:
                log(f"  perm seed {seed_i}: fold fit failed: {e}")
                continue
            pp_p.extend(p_va_p.tolist())
            # CRITICAL: evaluate against TRUE y, not the permuted y
            pp_y.extend(y[va_m].tolist())
        if not pp_p:
            continue
        try:
            auc_p = roc_auc_score(pp_y, pp_p)
        except ValueError:
            auc_p = float("nan")
        perm_aucs.append(auc_p)
        perm_records.append({"seed": seed_i, "pooled_auc": auc_p})
        log(f"  seed {seed_i:>2}: pooled OOS AUC (vs true labels) = {auc_p:.4f}")

    perm_arr = np.array([a for a in perm_aucs if np.isfinite(a)])
    perm_mean = float(perm_arr.mean()) if len(perm_arr) else float("nan")
    perm_se = float(perm_arr.std(ddof=1) / np.sqrt(len(perm_arr))) if len(perm_arr) > 1 else float("nan")
    log(f"\n  perm mean = {perm_mean:.4f}   SE = {perm_se:.4f}   mean+1·SE = {perm_mean + perm_se:.4f}")
    log(f"  model AUC − 0.05 = {auc_pool - G5_PERM_MARGIN:.4f}    (G5 requires perm mean+SE < this)")

    # ----- G6: cell floor -----
    log(f"\n" + "=" * 88)
    log("G6: COHORT GRID — body × signal_state × retest_kind")
    log(f"=" * 88)

    pred_df = pd.DataFrame({
        "row_idx": R,
        "fold_id": F,
        "y_broad": Y,
        "y_narrow": Y_n,
        "p": P,
    })
    meta = trade.iloc[R][[
        "ticker", "bar_date", "signal_state",
        "family__body_class", "family__retest_kind", "family__slope_tier",
        "regime_rdp",
    ]].reset_index(drop=True)
    pred_df = pd.concat([pred_df.reset_index(drop=True), meta], axis=1)
    pred_df["family__retest_kind"] = pred_df["family__retest_kind"].fillna("none")

    cohort_rows = []
    bad_cells = []
    grid = (
        pred_df.groupby(["family__body_class", "signal_state", "family__retest_kind"], observed=True, dropna=False)
        .size().reset_index(name="N")
    )
    for _, r in grid.iterrows():
        b, s, k = r["family__body_class"], r["signal_state"], r["family__retest_kind"]
        sub = pred_df[
            (pred_df["family__body_class"] == b)
            & (pred_df["signal_state"] == s)
            & (pred_df["family__retest_kind"] == k)
        ]
        n = len(sub)
        base_c = sub["y_broad"].mean() if n else 0.0
        nb = n * base_c
        if n >= 5 and base_c > 0:
            l = lift_at_top(sub["y_broad"].values, sub["p"].values, LIFT_DECILE)
        else:
            l = float("nan")
        try:
            auc_c = roc_auc_score(sub["y_broad"], sub["p"]) if sub["y_broad"].nunique() == 2 else float("nan")
        except ValueError:
            auc_c = float("nan")
        diagnostic_only = nb < N_BASE_PRODUCT_FLOOR
        cohort_rows.append({
            "body_class": b, "signal_state": s, "retest_kind": k,
            "N": n, "base_rate": base_c, "n_times_base": nb,
            "auc": auc_c, "lift_top10": l,
            "diagnostic_only": diagnostic_only,
        })
        if (n >= COHORT_MIN_N) and (not diagnostic_only) and np.isfinite(l) and (l < COHORT_LIFT_FLOOR):
            bad_cells.append((b, s, k, n, l, base_c))
    cohort_df = pd.DataFrame(cohort_rows).sort_values(["body_class", "signal_state", "retest_kind"])
    log(f"  {'body':<12} {'signal':<14} {'retest':<14} {'N':>5} {'base':>6} {'AUC':>6} {'lift':>6} {'diag':>5}")
    for _, r in cohort_df.iterrows():
        if r["N"] < 10:
            continue
        log(f"  {str(r['body_class']):<12} {str(r['signal_state']):<14} {str(r['retest_kind']):<14} "
            f"{int(r['N']):>5} {r['base_rate']:>6.3f} "
            f"{(f'{r.auc:.3f}' if pd.notna(r.auc) else '—'):>6} "
            f"{(f'{r.lift_top10:.2f}' if pd.notna(r.lift_top10) else '—'):>6} "
            f"{'✓' if r['diagnostic_only'] else '·':>5}")

    # ----- G7: single-feature dominance -----
    log(f"\n" + "=" * 88)
    log("G7: SINGLE-FEATURE DOMINANCE (mb_scanner trap defense)")
    log(f"=" * 88)
    # use pooled OOS rows' feature matrix (so we evaluate on same OOS predictions)
    X_pool = X.iloc[R].reset_index(drop=True)
    sf_df, best_sf_auc, best_sf_feat = best_single_feature_auc(X_pool, Y, feat_cols, cat_cols)
    log(f"  best single numeric feature: {best_sf_feat}    AUC={best_sf_auc:.4f}")
    log(f"  full model OOS AUC:           {auc_pool:.4f}")
    log(f"  margin (model − best):        {auc_pool - best_sf_auc:+.4f}    G7 requires ≥ +{G7_DOMINANCE_MARGIN}")
    log(f"\n  top 10 single-feature AUCs:")
    for _, r in sf_df.head(10).iterrows():
        log(f"    {str(r['feature']):<40}  {r['auc_signed']:.4f}  ({r['direction']})")

    # ===== gate verdicts =====
    log(f"\n" + "=" * 88)
    log("ACCEPTANCE GATES — verdict")
    log("=" * 88)

    g1 = (auc_pool > G1_AUC_THRESHOLD) and (auc_lo > G1_CI_LOWER_THRESHOLD)
    g2 = (lift_b >= G2_LIFT_THRESHOLD) and (lb_lo >= G2_CI_LOWER)
    g3 = (lift_n >= G3_LIFT_THRESHOLD) and (ln_lo >= G3_CI_LOWER)
    if q4_deferred:
        g4 = "deferred"
    else:
        g4 = q2_auc >= G4_SUB_AUC_THRESHOLD
    g5 = (np.isfinite(perm_mean) and np.isfinite(perm_se)
          and (perm_mean + perm_se) < (auc_pool - G5_PERM_MARGIN))
    g6 = len(bad_cells) == 0
    g7 = (auc_pool - best_sf_auc) >= G7_DOMINANCE_MARGIN

    def fmt(v):
        if isinstance(v, str): return v.upper()
        return "PASS" if v else "FAIL"

    log(f"  G1 pooled AUC > {G1_AUC_THRESHOLD} & CI_lo > {G1_CI_LOWER_THRESHOLD}:                 "
        f"{fmt(g1)}  (AUC={auc_pool:.4f} CI=[{auc_lo:.4f},{auc_hi:.4f}])")
    log(f"  G2 broad lift ≥ {G2_LIFT_THRESHOLD}× & CI_lo ≥ {G2_CI_LOWER}×:                     "
        f"{fmt(g2)}  (lift={lift_b:.2f}× CI=[{lb_lo:.2f},{lb_hi:.2f}])")
    log(f"  G3 narrow lift ≥ {G3_LIFT_THRESHOLD}× & CI_lo ≥ {G3_CI_LOWER}×:                    "
        f"{fmt(g3)}  (lift={lift_n:.2f}× CI=[{ln_lo:.2f},{ln_hi:.2f}])")
    log(f"  G4 2026Q2 sub-pool AUC ≥ {G4_SUB_AUC_THRESHOLD}:                       "
        f"{fmt(g4)}  ({'N<30 deferred' if q4_deferred else f'AUC={q2_auc:.4f}, N={q2_n}'})")
    log(f"  G5 permutation mean+SE < model AUC − {G5_PERM_MARGIN}:                "
        f"{fmt(g5)}  (perm={perm_mean:.4f}±{perm_se:.4f})")
    log(f"  G6 cell floor (lift ≥ {COHORT_LIFT_FLOOR} for N≥{COHORT_MIN_N} ∧ N×base≥{N_BASE_PRODUCT_FLOOR}):  "
        f"{fmt(g6)}")
    if not g6:
        for b, s, k, n, l, br in bad_cells:
            log(f"      !! {b}/{s}/{k}  N={n} base={br:.3f} lift={l:.2f}")
    log(f"  G7 single-feature dominance defense (margin ≥ {G7_DOMINANCE_MARGIN}):  "
        f"{fmt(g7)}  (model − best_sf = {auc_pool - best_sf_auc:+.4f}, sf={best_sf_feat})")

    # G4 deferred is not a free pass per spec §5.G4 — only excludes G4 from gating
    pass_set = [g1, g2, g3, g5, g6, g7]
    if g4 == "deferred":
        log(f"\n  Note: G4 deferred (N<30); v1_2 acceptance hinges on G1/G2/G3/G5/G6/G7.")
        all_pass = all(pass_set)
    else:
        all_pass = all(pass_set + [g4])

    verdict = "ACCEPT" if all_pass else "REJECT"
    log(f"\n  ============================================")
    log(f"  v1_2 VERDICT: {verdict}")
    log(f"  ============================================")
    log(f"  Note: ACCEPT does NOT promote to live (Q4 lock — research verdict only).")
    log(f"  total elapsed: {time.time()-t0:.1f}s")

    # ----- write artifacts -----
    pd.DataFrame(fold_records).to_csv(OUT_PER_FOLD, index=False)
    cohort_df.to_csv(OUT_COHORT, index=False)
    pd.DataFrame(perm_records).to_csv(OUT_PERM, index=False)
    sf_df.to_csv(OUT_SF_AUC, index=False)
    pred_df.to_parquet(OUT_PRED, index=False)
    OUT_LOG.write_text("\n".join(LOG_LINES) + "\n")

    # acceptance.md
    write_acceptance_md(
        all_pass=all_pass, verdict=verdict,
        auc_pool=auc_pool, auc_lo=auc_lo, auc_hi=auc_hi,
        base_pool=base_pool, base_narrow=base_narrow,
        lift_b=lift_b, lb_lo=lb_lo, lb_hi=lb_hi,
        lift_n=lift_n, ln_lo=ln_lo, ln_hi=ln_hi,
        q4_deferred=q4_deferred, q2_auc=q2_auc, q2_n=q2_n, q2_pos=q2_pos,
        perm_mean=perm_mean, perm_se=perm_se,
        bad_cells=bad_cells,
        best_sf_feat=best_sf_feat, best_sf_auc=best_sf_auc,
        gates={"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7},
        fold_records=fold_records,
        elapsed=time.time() - t0,
    )

    return 0 if all_pass else 1


def write_acceptance_md(**kw) -> None:
    g = kw["gates"]
    def f(v):
        if isinstance(v, str): return v.upper()
        return "**PASS**" if v else "**FAIL**"

    lines = []
    lines.append(f"# V1_2 Extreme-Mover Ranker — Acceptance Report")
    lines.append("")
    lines.append(f"**Verdict: {kw['verdict']}**")
    lines.append("")
    lines.append(f"- Spec: `memory/scanner_v1_2_extreme_ranker_spec.md` (LOCKED)")
    lines.append(f"- Run: {pd.Timestamp.utcnow().isoformat()}   elapsed: {kw['elapsed']:.1f}s")
    lines.append(f"- Pre-registration discipline: single fire, anti-rescue active.")
    lines.append("")
    lines.append("## Pooled OOS metrics")
    lines.append("")
    lines.append(f"- Pooled AUC: **{kw['auc_pool']:.4f}**   bootstrap 95% CI = [{kw['auc_lo']:.4f}, {kw['auc_hi']:.4f}]")
    lines.append(f"- Broad target base rate (≥5%/5d): {kw['base_pool']*100:.2f}%")
    lines.append(f"- Narrow gate base rate (≥10%/10d): {kw['base_narrow']*100:.2f}%")
    lines.append(f"- Top-decile lift (broad): {kw['lift_b']:.2f}×   CI [{kw['lb_lo']:.2f}, {kw['lb_hi']:.2f}]")
    lines.append(f"- Top-decile lift (narrow): {kw['lift_n']:.2f}×   CI [{kw['ln_lo']:.2f}, {kw['ln_hi']:.2f}]")
    lines.append("")
    lines.append("## Per-fold results (advisory)")
    lines.append("")
    lines.append("| fold | window | train_n | val_n | best_iter | AUC | lift_top10 | n_pos_broad |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in kw["fold_records"]:
        lines.append(f"| {r['fold']} | {r['val_start']}→{r['val_end']} | {r['train_n']} | {r['val_n']} | "
                     f"{r['best_iter']} | {r['auc']:.4f} | {r['lift_top10_broad']:.2f}× | {r['n_pos_broad']} |")
    lines.append("")
    lines.append("## Acceptance gates (pooled-OOS)")
    lines.append("")
    lines.append(f"- **G1** pooled AUC > 0.62 ∧ CI lower > 0.58 — {f(g['G1'])}")
    lines.append(f"- **G2** broad-target top-decile lift ≥ 1.8× ∧ CI lower ≥ 1.3× — {f(g['G2'])}")
    lines.append(f"- **G3** narrow-gate (≥10%/10d) top-decile lift ≥ 1.5× ∧ CI lower ≥ 1.0× — {f(g['G3'])}")
    if g['G4'] == "deferred":
        lines.append(f"- **G4** 2026Q2 sub-pool AUC ≥ 0.50 — DEFERRED (N={kw['q2_n']} < 30)")
    else:
        lines.append(f"- **G4** 2026Q2 sub-pool AUC ≥ 0.50 — {f(g['G4'])} (AUC={kw['q2_auc']:.4f}, N={kw['q2_n']})")
    lines.append(f"- **G5** permutation mean+1·SE < model AUC − 0.05 — {f(g['G5'])} (perm={kw['perm_mean']:.4f}±{kw['perm_se']:.4f})")
    lines.append(f"- **G6** cohort cell floor (lift ≥ 0.7× for N≥30 ∧ N×base≥1.0) — {f(g['G6'])}")
    if not g['G6']:
        for b, s, k, n, l, br in kw['bad_cells']:
            lines.append(f"  - violation: {b}/{s}/{k} N={n} base={br:.3f} lift={l:.2f}")
    lines.append(f"- **G7** single-feature dominance defense (margin ≥ 0.03) — {f(g['G7'])} "
                 f"(best_sf=`{kw['best_sf_feat']}` AUC={kw['best_sf_auc']:.4f}, "
                 f"margin={kw['auc_pool']-kw['best_sf_auc']:+.4f})")
    lines.append("")
    lines.append("## Closure path triggered")
    lines.append("")
    if kw["all_pass"]:
        lines.append("- All gates pass → extreme-mover hypothesis **supported as research verdict**.")
        lines.append("- **NO promotion** to live/paper/shadow per Q4 spec lock.")
    else:
        failed = [k for k, v in g.items() if v is False]
        lines.append(f"- Failed gates: {failed}")
        lines.append("- Anti-rescue clause active: v1_2 closes; no retrain / target-swap / feature-add rescue.")
        lines.append("- Successor (v1_3) requires fresh pre-registration distinct from 'v1_2 was close'.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Predictions: `output/horizontal_base_event_v1_2_predictions.parquet`")
    lines.append("- Per-fold metrics: `output/horizontal_base_event_v1_2_train_per_fold.csv`")
    lines.append("- Cohort grid: `output/horizontal_base_event_v1_2_train_cohort.csv`")
    lines.append("- Permutation distribution: `output/horizontal_base_event_v1_2_perm_distribution.csv`")
    lines.append("- Single-feature AUC table: `output/horizontal_base_event_v1_2_single_feature_auc.csv`")
    lines.append("- Training log: `output/horizontal_base_event_v1_2_train.log`")

    OUT_ACCEPT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="integrity checks only, no training")
    args = ap.parse_args()
    sys.exit(main(dry_run_only=args.dry_run))
