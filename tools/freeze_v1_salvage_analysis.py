"""V1 freeze salvage / error-analysis (Phase D).

Diagnoses the v1 acceptance gate REJECT (F4 AUC 0.539, F2 perm 0.551,
1 thin cohort cell). Replays the deterministic v1 trainer (seed=17, same
splits, same params) to capture per-event OOS predictions, then runs Q1-Q7
from `memory/scanner_v1_freeze_results.md`:

  Q1. F4 miss decomposition by cohort
  Q2. F2 permutation distribution (10 seeds)
  Q3. Thin cohort cell rule analysis
  Q4. Trigger-only vs retest-only per-fold AUC (two-head feasibility)
  Q5. quality_10d horizon retrain — does F4 recover?
  Q6. Per-year + per-regime AUC drift
  Q7. Top-decile concentration (tickers/dates)

The replay produces identical predictions to the original v1 train — only
the dump of per-event predictions is new instrumentation. Q5 retrains on a
different target as a horizon diagnostic, NOT as v1.5.

Outputs:
  output/horizontal_base_event_v1_predictions.parquet  (per-event OOS preds)
  output/horizontal_base_event_v1_salvage.md           (findings report)
  output/horizontal_base_event_v1_salvage.log          (full log)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
SPLIT = ROOT / "output/horizontal_base_event_v1_split.json"
PRED_PARQUET = ROOT / "output/horizontal_base_event_v1_predictions.parquet"
SALVAGE_MD = ROOT / "output/horizontal_base_event_v1_salvage.md"
SALVAGE_LOG = ROOT / "output/horizontal_base_event_v1_salvage.log"

# --------------------------------------------------------------------------
# Mirror trainer config exactly (must match freeze_v1_walk_forward_train.py)
# --------------------------------------------------------------------------
TARGET = "quality_20d"
LABEL_REALIZED = "mfe_R_20d"

CATEGORICAL_FEATURES = [
    "family__body_class",
    "family__retest_kind",
    "family__slope_tier",
    "common__regime",
    "signal_state",
]

EXCLUDE_FROM_FEATURES = {
    "mfe_R_3d", "mae_R_3d", "realized_R_3d", "failed_breakout_3d", "time_to_MFE_3d",
    "mfe_R_5d", "mae_R_5d", "realized_R_5d", "failed_breakout_5d", "time_to_MFE_5d",
    "mfe_R_10d", "mae_R_10d", "realized_R_10d", "failed_breakout_10d", "time_to_MFE_10d",
    "mfe_R_20d", "mae_R_20d", "realized_R_20d", "failed_breakout_20d", "time_to_MFE_20d",
    "early_failure_5d", "quality_20d",
    "ticker", "bar_date", "setup_family", "signal_type",
    "breakout_bar_date", "as_of_ts", "data_frequency", "schema_version",
    "feature_version", "scanner_version",
    "asof_idx", "breakout_idx",
    "regime_sub", "regime_window_id",
    "bar_year", "val_fold",
    "entry_reference_price", "invalidation_level", "initial_risk_pct",
    "family__trigger_level",
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
BOOTSTRAP_SEED = 17
LIFT_DECILE = 0.10
QUALITY_TH = 2.0  # for quality_10d

LOG_LINES: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG_LINES.append(s)


def lift_at_top_k(y_true: np.ndarray, p: np.ndarray, frac: float) -> float:
    base = y_true.mean()
    if base <= 0:
        return float("nan")
    n_top = max(1, int(round(len(p) * frac)))
    order = np.argsort(-p)
    top_rate = y_true[order[:n_top]].mean()
    return float(top_rate / base)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    cat_cols = [c for c in feat_cols if c in CATEGORICAL_FEATURES]
    X = df[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, feat_cols, cat_cols


def train_one_fold(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    cat_cols: list[str],
    params: dict | None = None,
) -> tuple[lgb.Booster, np.ndarray, int]:
    p = dict(LGB_PARAMS) if params is None else dict(params)
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_ds, free_raw_data=False)
    booster = lgb.train(
        p, train_ds, num_boost_round=NUM_BOOST,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    p_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    return booster, p_val, int(booster.best_iteration or 0)


# ==========================================================================
# PART A — replay trainer, capture per-event OOS predictions
# ==========================================================================
def replay_and_dump_predictions(trade: pd.DataFrame, split: dict) -> pd.DataFrame:
    log("=" * 88)
    log("PART A — replay deterministic v1 trainer, dump per-event OOS predictions")
    log("=" * 88)

    X, feat_cols, cat_cols = build_feature_matrix(trade)
    y = trade[TARGET].astype(int).values
    log(f"  feature cols: {len(feat_cols)} (cat={len(cat_cols)})")

    rows: list[pd.DataFrame] = []
    for fdef in split["fold_definitions"]:
        fid = fdef["fold_id"]
        vs = pd.Timestamp(fdef["val_start"])
        ve = pd.Timestamp(fdef["val_end"])
        train_mask = trade["bar_date"] < vs
        val_mask = (trade["bar_date"] >= vs) & (trade["bar_date"] < ve)

        X_tr, X_va = X[train_mask].reset_index(drop=True), X[val_mask].reset_index(drop=True)
        y_tr, y_va = y[train_mask], y[val_mask]
        booster, p_va, best_iter = train_one_fold(X_tr, y_tr, X_va, y_va, cat_cols)

        # match trainer's isotonic step (fit on booster's train predictions)
        p_tr = booster.predict(X_tr, num_iteration=booster.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p_tr, y_tr)
        p_va_cal = np.clip(iso.transform(p_va), 1e-6, 1 - 1e-6)

        auc = safe_auc(y_va, p_va)
        log(f"  [{fid}] train_n={int(train_mask.sum()):,} val_n={int(val_mask.sum()):,} "
            f"best_iter={best_iter} AUC={auc:.4f}")

        meta = trade[val_mask][[
            "ticker", "bar_date", "signal_state", "family__body_class",
            "family__retest_kind", "family__slope_tier", "common__regime",
            "bar_year", "mfe_R_10d", "mfe_R_20d", "quality_20d",
        ]].copy().reset_index(drop=True)
        meta["fold"] = fid
        meta["y"] = y_va
        meta["p"] = p_va
        meta["p_cal"] = p_va_cal
        rows.append(meta)

    preds = pd.concat(rows, ignore_index=True)
    # add quality_10d label for Q5 evaluation
    preds["quality_10d"] = (preds["mfe_R_10d"] >= QUALITY_TH).astype("Int8")
    preds["quarter"] = pd.PeriodIndex(preds["bar_date"], freq="Q").astype(str)

    preds.to_parquet(PRED_PARQUET, index=False)
    log(f"  per-event predictions saved: {PRED_PARQUET.name}  N={len(preds):,}")
    return preds


# ==========================================================================
# PART B — Q1: F4 miss decomposition
# ==========================================================================
def q1_f4_decomposition(preds: pd.DataFrame) -> dict:
    log("\n" + "=" * 88)
    log("Q1. F4 miss decomposition — what cohort drags F4 AUC below 0.55?")
    log("=" * 88)
    f4 = preds[preds["fold"] == "F4"].copy()
    base_auc = safe_auc(f4["y"].values, f4["p"].values)
    log(f"  F4 N={len(f4):,}  base_rate={f4['y'].mean():.4f}  AUC={base_auc:.4f}")

    findings: dict = {"f4_auc": base_auc, "by_dim": {}}
    for dim in ["signal_state", "family__body_class", "family__retest_kind",
                "family__slope_tier", "common__regime", "quarter"]:
        log(f"\n  by {dim}:")
        log(f"    {'value':<18} {'N':>5} {'base':>6} {'AUC':>7} {'lift10':>7}")
        rows = []
        for v, sub in f4.groupby(dim, dropna=False, observed=True):
            n = len(sub)
            if n < 5:
                continue
            auc = safe_auc(sub["y"].values, sub["p"].values)
            lift = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE) if n >= 10 else float("nan")
            rows.append({"value": str(v), "N": n, "base": sub["y"].mean(),
                         "auc": auc, "lift10": lift})
            log(f"    {str(v):<18} {n:>5} {sub['y'].mean():>6.3f} "
                f"{(f'{auc:.3f}' if pd.notna(auc) else '—'):>7} "
                f"{(f'{lift:.2f}' if pd.notna(lift) else '—'):>7}")
        findings["by_dim"][dim] = rows
    return findings


# ==========================================================================
# PART C — Q2: F2 multi-seed permutation
# ==========================================================================
def q2_f2_perm_distribution(trade: pd.DataFrame, split: dict, n_seeds: int = 10) -> dict:
    log("\n" + "=" * 88)
    log(f"Q2. F2 permutation distribution — {n_seeds} seeds")
    log("=" * 88)
    X, _, cat_cols = build_feature_matrix(trade)
    y = trade[TARGET].astype(int).values

    fdef = next(f for f in split["fold_definitions"] if f["fold_id"] == "F2")
    vs = pd.Timestamp(fdef["val_start"]); ve = pd.Timestamp(fdef["val_end"])
    train_mask = trade["bar_date"] < vs
    val_mask = (trade["bar_date"] >= vs) & (trade["bar_date"] < ve)
    X_tr, X_va = X[train_mask].reset_index(drop=True), X[val_mask].reset_index(drop=True)
    y_tr, y_va = y[train_mask], y[val_mask]

    perm_aucs: list[float] = []
    log(f"  F2 train_n={len(y_tr):,} val_n={len(y_va):,} base_rate(val)={y_va.mean():.4f}")
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        y_perm = rng.permutation(y_tr)
        try:
            _, p_va_p, _ = train_one_fold(X_tr, y_perm, X_va, y_va, cat_cols)
            auc_p = safe_auc(y_va, p_va_p)
        except Exception as e:
            auc_p = float("nan")
            log(f"   seed {s}: failed {e}")
        perm_aucs.append(auc_p)
        log(f"   seed {1000+s}: perm_AUC={auc_p:.4f}")

    arr = np.array([a for a in perm_aucs if np.isfinite(a)])
    summary = {
        "seeds": list(range(1000, 1000 + n_seeds)),
        "perm_aucs": perm_aucs,
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "p25": float(np.percentile(arr, 25)) if len(arr) else float("nan"),
        "p75": float(np.percentile(arr, 75)) if len(arr) else float("nan"),
        "max": float(arr.max()) if len(arr) else float("nan"),
        "min": float(arr.min()) if len(arr) else float("nan"),
        "n_above_055": int((arr > 0.55).sum()),
        "n_total": len(arr),
        "v1_single_seed": 0.5507,  # from train log
    }
    log(f"\n  summary:  mean={summary['mean']:.4f}  median={summary['median']:.4f}  "
        f"p25={summary['p25']:.4f}  p75={summary['p75']:.4f}  max={summary['max']:.4f}")
    log(f"  v1 single-seed perm AUC was {summary['v1_single_seed']:.4f}; "
        f"# of {n_seeds} seeds above 0.55: {summary['n_above_055']}/{summary['n_total']}")
    return summary


# ==========================================================================
# PART D — Q3: thin cohort cell rule
# ==========================================================================
def q3_thin_cell(preds: pd.DataFrame) -> dict:
    log("\n" + "=" * 88)
    log("Q3. Thin cohort cell rule — is `lift<0.7` meaningful at base=0.033?")
    log("=" * 88)

    pooled = preds.copy()
    pooled["family__retest_kind"] = pooled["family__retest_kind"].fillna("none")
    grid = pooled.groupby(
        ["family__body_class", "family__slope_tier", "family__retest_kind"],
        dropna=False, observed=True,
    ).agg(N=("y", "size"), base=("y", "mean")).reset_index()

    rows = []
    log(f"  cells with N>=30 (existing gate threshold):")
    log(f"    {'cell':<40} {'N':>4} {'base':>6} {'AUC':>6} {'lift10':>7} {'expE_pos':>9}")
    for _, r in grid.iterrows():
        b, s, k, n, base = r["family__body_class"], r["family__slope_tier"], r["family__retest_kind"], int(r["N"]), float(r["base"])
        sub = pooled[
            (pooled["family__body_class"] == b)
            & (pooled["family__slope_tier"] == s)
            & (pooled["family__retest_kind"] == k)
        ]
        auc = safe_auc(sub["y"].values, sub["p"].values)
        lift = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE) if n >= 10 else float("nan")
        # expected positives in top-decile under the null:
        # = base * top_n  where top_n = max(1, round(N*0.10))
        top_n = max(1, int(round(n * 0.10)))
        exp_e = base * top_n
        rows.append({"cell": f"{b}/{s}/{k}", "N": n, "base": base, "auc": auc,
                     "lift10": lift, "expected_positives_in_top10": exp_e})
        if n >= 30:
            log(f"    {f'{b}/{s}/{k}':<40} {n:>4} {base:>6.3f} "
                f"{(f'{auc:.3f}' if pd.notna(auc) else '—'):>6} "
                f"{(f'{lift:.2f}' if pd.notna(lift) else '—'):>7} "
                f"{exp_e:>9.2f}")

    # the failed cell
    flagged = [r for r in rows if r["cell"] == "large_body/mild/no_touch"]
    if flagged:
        log(f"\n  FLAGGED CELL: large_body/mild/no_touch  N=30 base=0.033 "
            f"AUC={flagged[0]['auc']:.3f} lift={flagged[0]['lift10']:.2f}")
        log(f"  Top-decile of N=30 = 3 events. Expected positives at base 0.033 "
            f"= {flagged[0]['expected_positives_in_top10']:.2f}.")
        log(f"  Lift threshold 0.7× requires ≥ {0.7 * flagged[0]['expected_positives_in_top10']:.2f} "
            f"positives in top-decile; floor of integer count is 0 unless model picks ≥1.")
        log(f"  → AUC=0.690 means model RANKS the cell well; lift=0.0 is base-rate floor noise.")

    # what minimum-N rule excludes thin cells?
    log(f"\n  Cell N>=30 cells with expected_positives_in_top10 < 0.5 (i.e. lift gate is noise):")
    noise_cells = [r for r in rows if r["N"] >= 30 and r["expected_positives_in_top10"] < 0.5]
    for r in noise_cells:
        log(f"    {r['cell']}  N={r['N']} base={r['base']:.3f} expE={r['expected_positives_in_top10']:.2f}")
    log(f"  Suggested rule: N>=30 AND expected_positives_in_top10 >= 1.0 "
        f"(i.e. min N=30/base; e.g. base=0.033 needs N>=300 to make lift gate meaningful).")

    return {"rows": rows, "noise_cells": noise_cells}


# ==========================================================================
# PART E — Q4: trigger-only vs retest-only per-fold AUC
# ==========================================================================
def q4_two_head(preds: pd.DataFrame) -> dict:
    log("\n" + "=" * 88)
    log("Q4. Cohort heterogeneity — trigger vs retest per-fold AUC")
    log("=" * 88)
    out = {"per_fold": {}}
    log(f"  {'fold':<5} {'trig_n':>6} {'trig_AUC':>9} {'trig_lift':>9} "
        f"{'rtst_n':>6} {'rtst_AUC':>9} {'rtst_lift':>9}")
    for fid in ["F1", "F2", "F3", "F4"]:
        fsub = preds[preds["fold"] == fid]
        trig = fsub[fsub["signal_state"] == "trigger"]
        rtst = fsub[fsub["signal_state"] == "retest_bounce"]
        t_auc = safe_auc(trig["y"].values, trig["p"].values)
        r_auc = safe_auc(rtst["y"].values, rtst["p"].values)
        t_lift = lift_at_top_k(trig["y"].values, trig["p"].values, LIFT_DECILE) if len(trig) >= 10 else float("nan")
        r_lift = lift_at_top_k(rtst["y"].values, rtst["p"].values, LIFT_DECILE) if len(rtst) >= 10 else float("nan")
        log(f"  {fid:<5} {len(trig):>6} "
            f"{(f'{t_auc:.3f}' if pd.notna(t_auc) else '—'):>9} "
            f"{(f'{t_lift:.2f}' if pd.notna(t_lift) else '—'):>9} "
            f"{len(rtst):>6} "
            f"{(f'{r_auc:.3f}' if pd.notna(r_auc) else '—'):>9} "
            f"{(f'{r_lift:.2f}' if pd.notna(r_lift) else '—'):>9}")
        out["per_fold"][fid] = {
            "trigger_n": len(trig), "trigger_auc": t_auc, "trigger_lift10": t_lift,
            "retest_n": len(rtst), "retest_auc": r_auc, "retest_lift10": r_lift,
        }

    # is two-head viable?
    trig_aucs = [out["per_fold"][f]["trigger_auc"] for f in ["F1","F2","F3","F4"]]
    rtst_aucs = [out["per_fold"][f]["retest_auc"] for f in ["F1","F2","F3","F4"]]
    log(f"\n  trigger AUCs across folds: {[round(a,3) for a in trig_aucs]}")
    log(f"  retest  AUCs across folds: {[round(a,3) for a in rtst_aucs]}")
    log(f"  trigger min/4: {min(trig_aucs):.3f}  retest min/4: {min(rtst_aucs):.3f}")
    return out


# ==========================================================================
# PART F — Q5: quality_10d horizon retrain
# ==========================================================================
def q5_horizon_retrain(trade: pd.DataFrame, split: dict) -> dict:
    log("\n" + "=" * 88)
    log("Q5. quality_10d horizon retrain — does shorter horizon recover F4?")
    log("=" * 88)
    log("  NOTE: this retrains on a different target (q10d) for diagnosis only.")
    log("        It is NOT v1.5 and acceptance gates do NOT apply to it.")

    trade2 = trade.copy()
    trade2["quality_10d"] = (trade2["mfe_R_10d"] >= QUALITY_TH).astype(int)
    # drop events where mfe_R_10d is NaN
    keep = trade2["mfe_R_10d"].notna()
    trade2 = trade2[keep].reset_index(drop=True)
    log(f"  trade events with q10d label: {len(trade2):,}  pos_rate={trade2['quality_10d'].mean():.4f}")

    # rebuild features (same exclusion set; target column quality_10d not in
    # EXCLUDE_FROM_FEATURES so add it explicitly)
    EX10 = EXCLUDE_FROM_FEATURES | {"quality_10d"}
    feat_cols = [c for c in trade2.columns if c not in EX10]
    cat_cols = [c for c in feat_cols if c in CATEGORICAL_FEATURES]
    X = trade2[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = trade2["quality_10d"].astype(int).values

    out = {"per_fold": {}}
    log(f"\n  {'fold':<5} {'train_n':>8} {'val_n':>6} {'AUC':>6} {'lift10':>7}")
    for fdef in split["fold_definitions"]:
        fid = fdef["fold_id"]
        vs = pd.Timestamp(fdef["val_start"]); ve = pd.Timestamp(fdef["val_end"])
        train_mask = trade2["bar_date"] < vs
        val_mask = (trade2["bar_date"] >= vs) & (trade2["bar_date"] < ve)
        if int(val_mask.sum()) < 30:
            continue
        X_tr, X_va = X[train_mask].reset_index(drop=True), X[val_mask].reset_index(drop=True)
        y_tr, y_va = y[train_mask], y[val_mask]
        _, p_va, _ = train_one_fold(X_tr, y_tr, X_va, y_va, cat_cols)
        auc = safe_auc(y_va, p_va)
        lift = lift_at_top_k(y_va, p_va, LIFT_DECILE)
        log(f"  {fid:<5} {int(train_mask.sum()):>8} {int(val_mask.sum()):>6} "
            f"{auc:>6.3f} {lift:>7.2f}")
        out["per_fold"][fid] = {"train_n": int(train_mask.sum()), "val_n": int(val_mask.sum()),
                                "auc": auc, "lift10": lift}
    return out


# ==========================================================================
# PART G — Q6: per-year + per-regime AUC drift
# ==========================================================================
def q6_drift(preds: pd.DataFrame) -> dict:
    log("\n" + "=" * 88)
    log("Q6. Year + regime drift — does mid_body 2025-26 cooling appear in residuals?")
    log("=" * 88)

    out = {"by_year": {}, "by_regime": {}, "by_year_body": {}}

    # bar_year column may be int; ensure numeric
    preds = preds.copy()
    preds["bar_year"] = pd.to_numeric(preds["bar_year"], errors="coerce").astype("Int64")

    log(f"\n  by year (pooled across folds):")
    log(f"    {'year':<6} {'N':>5} {'base':>6} {'AUC':>6} {'lift10':>7}")
    for y, sub in preds.groupby("bar_year", dropna=True):
        n = len(sub)
        if n < 30:
            continue
        auc = safe_auc(sub["y"].values, sub["p"].values)
        lift = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE)
        log(f"    {int(y):<6} {n:>5} {sub['y'].mean():>6.3f} "
            f"{(f'{auc:.3f}' if pd.notna(auc) else '—'):>6} "
            f"{(f'{lift:.2f}' if pd.notna(lift) else '—'):>7}")
        out["by_year"][int(y)] = {"N": n, "base": float(sub["y"].mean()),
                                   "auc": auc, "lift10": lift}

    log(f"\n  by regime (pooled):")
    log(f"    {'regime':<8} {'N':>5} {'base':>6} {'AUC':>6} {'lift10':>7}")
    for r, sub in preds.groupby("common__regime", dropna=False, observed=True):
        n = len(sub)
        if n < 30:
            continue
        auc = safe_auc(sub["y"].values, sub["p"].values)
        lift = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE)
        log(f"    {str(r):<8} {n:>5} {sub['y'].mean():>6.3f} "
            f"{(f'{auc:.3f}' if pd.notna(auc) else '—'):>6} "
            f"{(f'{lift:.2f}' if pd.notna(lift) else '—'):>7}")
        out["by_regime"][str(r)] = {"N": n, "base": float(sub["y"].mean()),
                                     "auc": auc, "lift10": lift}

    log(f"\n  by year × body_class (the diag flag):")
    log(f"    {'year':<5} {'body':<11} {'N':>5} {'base':>6} {'AUC':>6} {'lift10':>7}")
    for (y, b), sub in preds.groupby(["bar_year", "family__body_class"],
                                     dropna=True, observed=True):
        n = len(sub)
        if n < 20:
            continue
        auc = safe_auc(sub["y"].values, sub["p"].values)
        lift = lift_at_top_k(sub["y"].values, sub["p"].values, LIFT_DECILE)
        log(f"    {int(y):<5} {str(b):<11} {n:>5} {sub['y'].mean():>6.3f} "
            f"{(f'{auc:.3f}' if pd.notna(auc) else '—'):>6} "
            f"{(f'{lift:.2f}' if pd.notna(lift) else '—'):>7}")
        out["by_year_body"][f"{int(y)}/{b}"] = {"N": n, "base": float(sub["y"].mean()),
                                                 "auc": auc, "lift10": lift}
    return out


# ==========================================================================
# PART H — Q7: top-decile concentration
# ==========================================================================
def q7_concentration(preds: pd.DataFrame) -> dict:
    log("\n" + "=" * 88)
    log("Q7. Top-decile concentration — how distributed is the edge?")
    log("=" * 88)

    out = {"per_fold": {}}
    log(f"  {'fold':<5} {'top10_N':>8} {'distinct_tickers':>17} "
        f"{'distinct_dates':>15} {'top1_ticker':>14} {'top1_share':>11}")
    for fid in ["F1", "F2", "F3", "F4"]:
        sub = preds[preds["fold"] == fid].copy()
        n_top = max(1, int(round(len(sub) * LIFT_DECILE)))
        top = sub.nlargest(n_top, "p")
        n_tk = top["ticker"].nunique()
        n_dt = top["bar_date"].nunique()
        if n_tk > 0:
            tc = top["ticker"].value_counts()
            top1 = tc.iloc[0]
            top1_tk = tc.index[0]
            share = top1 / len(top)
        else:
            top1 = 0; top1_tk = ""; share = 0.0
        log(f"  {fid:<5} {n_top:>8} {n_tk:>17} {n_dt:>15} "
            f"{top1_tk:>14} {share:>11.2%}")
        out["per_fold"][fid] = {
            "top10_N": n_top, "distinct_tickers": n_tk, "distinct_dates": n_dt,
            "top1_ticker": str(top1_tk), "top1_count": int(top1), "top1_share": float(share),
        }

    # pooled
    n_top_pool = max(1, int(round(len(preds) * LIFT_DECILE)))
    top_pool = preds.nlargest(n_top_pool, "p")
    log(f"\n  pooled top-decile: N={n_top_pool}  distinct_tickers={top_pool['ticker'].nunique()}  "
        f"distinct_dates={top_pool['bar_date'].nunique()}")
    out["pooled"] = {
        "top10_N": n_top_pool,
        "distinct_tickers": int(top_pool["ticker"].nunique()),
        "distinct_dates": int(top_pool["bar_date"].nunique()),
    }
    return out


# ==========================================================================
# Report writer
# ==========================================================================
def write_report(q1, q2, q3, q4, q5, q6, q7) -> None:
    md = []
    md.append("# V1.3.1 ML freeze v1 — Salvage / Error-Analysis Report")
    md.append(f"_Generated: {pd.Timestamp.utcnow().isoformat()}_")
    md.append("")
    md.append("Source: `tools/freeze_v1_salvage_analysis.py` against frozen v1 artifacts.")
    md.append("Replay was deterministic (seed=17, identical to original Phase B trainer).")
    md.append("Q5 retrain on `quality_10d` is diagnostic-only, NOT v1.5.")
    md.append("")

    # Q1
    md.append("## Q1. F4 miss decomposition")
    md.append(f"F4 overall AUC = **{q1['f4_auc']:.4f}** (gate threshold 0.55 → fails by Δ−{0.55-q1['f4_auc']:.3f}).")
    for dim, rows in q1["by_dim"].items():
        if not rows:
            continue
        md.append(f"\n**by `{dim}`:**\n")
        md.append("| value | N | base | AUC | lift@10 |")
        md.append("|---|---:|---:|---:|---:|")
        for r in rows:
            auc_s = f"{r['auc']:.3f}" if pd.notna(r["auc"]) else "—"
            lift_s = f"{r['lift10']:.2f}" if pd.notna(r["lift10"]) else "—"
            md.append(f"| {r['value']} | {r['N']} | {r['base']:.3f} | {auc_s} | {lift_s} |")

    # Q2
    md.append("\n## Q2. F2 permutation distribution (10 seeds)")
    md.append(f"Distribution of permutation-trained AUCs on F2 val (target permuted on train).")
    md.append("")
    md.append(f"- mean = **{q2['mean']:.4f}**")
    md.append(f"- median = **{q2['median']:.4f}**, p25 = {q2['p25']:.4f}, p75 = {q2['p75']:.4f}")
    md.append(f"- min = {q2['min']:.4f}, max = {q2['max']:.4f}")
    md.append(f"- # of seeds above the 0.55 gate: **{q2['n_above_055']} / {q2['n_total']}**")
    md.append(f"- v1 single-seed value (the failure): {q2['v1_single_seed']:.4f}")
    md.append("")
    md.append("| seed | perm AUC |")
    md.append("|---:|---:|")
    for s, a in zip(q2["seeds"], q2["perm_aucs"]):
        md.append(f"| {s} | {a:.4f} |")

    # Q3
    md.append("\n## Q3. Thin cohort cell rule")
    md.append("Cells with N≥30 and `expected_positives_in_top10 < 0.5` (i.e. lift gate is statistical noise):")
    md.append("")
    md.append("| cell | N | base | AUC | lift@10 | E[pos] in top10 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in q3["noise_cells"]:
        auc_s = f"{r['auc']:.3f}" if pd.notna(r["auc"]) else "—"
        lift_s = f"{r['lift10']:.2f}" if pd.notna(r["lift10"]) else "—"
        md.append(f"| {r['cell']} | {r['N']} | {r['base']:.3f} | {auc_s} | "
                  f"{lift_s} | {r['expected_positives_in_top10']:.2f} |")
    md.append("")
    md.append("Suggested replacement rule: `N >= 30 AND expected_positives_in_top10 >= 1.0`. "
              "At base=0.033 this requires N>=303 — i.e. the thin-base cells should be **excluded** "
              "from the cohort lift gate, not assessed by it.")

    # Q4
    md.append("\n## Q4. Two-head feasibility (trigger vs retest per-fold)")
    md.append("| fold | trigger_n | trigger_AUC | trigger_lift | retest_n | retest_AUC | retest_lift |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for fid in ["F1", "F2", "F3", "F4"]:
        d = q4["per_fold"][fid]
        md.append(f"| {fid} | {d['trigger_n']} | "
                  f"{(f'{d['trigger_auc']:.3f}' if pd.notna(d['trigger_auc']) else '—')} | "
                  f"{(f'{d['trigger_lift10']:.2f}' if pd.notna(d['trigger_lift10']) else '—')} | "
                  f"{d['retest_n']} | "
                  f"{(f'{d['retest_auc']:.3f}' if pd.notna(d['retest_auc']) else '—')} | "
                  f"{(f'{d['retest_lift10']:.2f}' if pd.notna(d['retest_lift10']) else '—')} |")

    # Q5
    md.append("\n## Q5. quality_10d horizon retrain (diagnostic, not v1.5)")
    md.append("Same folds, same features, same params, target swapped to `quality_10d` (MFE_R_10d ≥ 2.0).")
    md.append("")
    md.append("| fold | train_n | val_n | AUC | lift@10 |")
    md.append("|---|---:|---:|---:|---:|")
    for fid in ["F1", "F2", "F3", "F4"]:
        if fid not in q5["per_fold"]:
            continue
        d = q5["per_fold"][fid]
        md.append(f"| {fid} | {d['train_n']} | {d['val_n']} | "
                  f"{d['auc']:.3f} | {d['lift10']:.2f} |")

    # Q6
    md.append("\n## Q6. Year + regime drift")
    md.append("\n**by year (pooled):**\n")
    md.append("| year | N | base | AUC | lift@10 |")
    md.append("|---:|---:|---:|---:|---:|")
    for y, d in sorted(q6["by_year"].items()):
        auc_s = f"{d['auc']:.3f}" if pd.notna(d["auc"]) else "—"
        lift_s = f"{d['lift10']:.2f}" if pd.notna(d["lift10"]) else "—"
        md.append(f"| {y} | {d['N']} | {d['base']:.3f} | {auc_s} | {lift_s} |")
    md.append("\n**by regime (pooled):**\n")
    md.append("| regime | N | base | AUC | lift@10 |")
    md.append("|---|---:|---:|---:|---:|")
    for r, d in q6["by_regime"].items():
        auc_s = f"{d['auc']:.3f}" if pd.notna(d["auc"]) else "—"
        lift_s = f"{d['lift10']:.2f}" if pd.notna(d["lift10"]) else "—"
        md.append(f"| {r} | {d['N']} | {d['base']:.3f} | {auc_s} | {lift_s} |")
    md.append("\n**by year × body_class (mid_body cooling check):**\n")
    md.append("| year/body | N | base | AUC | lift@10 |")
    md.append("|---|---:|---:|---:|---:|")
    for k, d in sorted(q6["by_year_body"].items()):
        auc_s = f"{d['auc']:.3f}" if pd.notna(d["auc"]) else "—"
        lift_s = f"{d['lift10']:.2f}" if pd.notna(d["lift10"]) else "—"
        md.append(f"| {k} | {d['N']} | {d['base']:.3f} | {auc_s} | {lift_s} |")

    # Q7
    md.append("\n## Q7. Top-decile concentration")
    md.append("| fold | top10_N | distinct_tickers | distinct_dates | top1_ticker | top1_share |")
    md.append("|---|---:|---:|---:|---|---:|")
    for fid in ["F1", "F2", "F3", "F4"]:
        d = q7["per_fold"][fid]
        md.append(f"| {fid} | {d['top10_N']} | {d['distinct_tickers']} | "
                  f"{d['distinct_dates']} | {d['top1_ticker']} | {d['top1_share']:.2%} |")
    md.append(f"\nPooled top-decile: N={q7['pooled']['top10_N']}, "
              f"distinct tickers={q7['pooled']['distinct_tickers']}, "
              f"distinct dates={q7['pooled']['distinct_dates']}.")

    SALVAGE_MD.write_text("\n".join(md) + "\n")
    log(f"\nReport written to {SALVAGE_MD.name}")


# ==========================================================================
def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    split = json.loads(SPLIT.read_text())
    log(f"V1 salvage / error-analysis — {pd.Timestamp.utcnow().isoformat()}")
    log(f"  loaded parquet: {len(df):,} rows / {df.shape[1]} cols")

    trade = df[df["signal_state"].isin(["trigger", "retest_bounce"])].copy()
    trade = trade[trade[LABEL_REALIZED].notna()].copy()
    trade["bar_date"] = pd.to_datetime(trade["bar_date"]).dt.normalize()
    log(f"  trade universe (q20d-labeled): {len(trade):,}")

    preds = replay_and_dump_predictions(trade, split)
    q1 = q1_f4_decomposition(preds)
    q2 = q2_f2_perm_distribution(trade, split, n_seeds=10)
    q3 = q3_thin_cell(preds)
    q4 = q4_two_head(preds)
    q5 = q5_horizon_retrain(trade, split)
    q6 = q6_drift(preds)
    q7 = q7_concentration(preds)

    write_report(q1, q2, q3, q4, q5, q6, q7)
    SALVAGE_LOG.write_text("\n".join(LOG_LINES) + "\n")
    log(f"\nelapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
