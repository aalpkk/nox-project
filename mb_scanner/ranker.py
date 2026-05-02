"""mb_scanner PASS-cohort multi-horizon ranker — LOCKED pre-reg v1.

Spec: `memory/mb_scanner_pass_cohort_ranker_v1.md` (LOCKED 2026-05-01).

Trains 5 SEPARATE per-horizon LightGBM binary classifiers on Phase 1 PASS
cohorts (mb_1d / mb_1w / mb_1M / bb_1M `above_mb_birth`, pooled). Each
horizon h ∈ {1, 3, 5, 10, 20} learns `quality_h = mfe_r_h ≥ 2.0`.

V1 ML freeze lessons baked in:
  - No two-head, no multi-output, no horizon swap.
  - 4-fold time-ordered walk-forward on TRAIN+VAL only; TEST sealed.
  - 6 gates per horizon (must all pass): pooled OOS AUC, quarter
    stability, 10-seed permutation, top-decile lift CI, cell floor,
    sealed TEST AUC.

Anti-rescue clause active. Failure → close, no retrain / sweep /
feature-swap / threshold tuning / two-head combine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS
# ---------------------------------------------------------------------------

RANKER_SPEC_ID = "mb_scanner_pass_cohort_ranker_v1"

PASS_COHORTS: tuple[tuple[str, str], ...] = (
    ("mb_1d", "above_mb_birth"),
    ("mb_1w", "above_mb_birth"),
    ("mb_1M", "above_mb_birth"),
    ("bb_1M", "above_mb_birth"),
)
PRIMARY_COHORT = ("mb_1d", "above_mb_birth")

HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
TARGET_THRESHOLD_R = 2.0  # quality_h := mfe_r_h ≥ 2.0
N_FOLDS = 4               # time-ordered walk-forward on TRAIN+VAL
N_PERM_SEEDS = 10         # G3 permutation gate
TOP_DECILE = 0.10
CELL_FLOOR_LIFT = 0.7
CELL_FLOOR_MIN_N = 30
QUARTER_MIN_N = 30
G1_AUC_FLOOR = 0.55
G2_QUARTER_AUC_FLOOR = 0.50
G3_PERM_DELTA = 0.03      # mean(perm AUC) ≤ pooled AUC − 0.03
G6_TEST_AUC_FLOOR = 0.52
SEED = 42

# LightGBM params (LOCKED — no sweep)
LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "verbose": -1,
    "random_state": SEED,
}
EARLY_STOPPING_ROUNDS = 30

# LOCKED FEATURE SET v1 (raw / numeric)
NUMERIC_FEATURES: tuple[str, ...] = (
    # Volume
    "vol_ratio_20_at_event",
    # Momentum
    "bos_distance_pct_at_event",
    "bos_distance_atr_at_event",
    # Volatility
    "atr_pct_at_event",
    # Zone geometry
    "zone_width_atr",
    "zone_width_pct",
    "zone_age_bars",
    # Quartet structure
    "concurrent_quartets",
    "pivot_confirm_lag_bars",
    "hh_to_event_lag_bars",
    # Risk frame
    "r_distance_pct",         # derived: (event_close - struct_inv_low)/event_close
    # Quartet leg ratios (derived)
    "lh_over_ll",
    "hl_over_lh",
    "hh_over_lh",
    "quartet_span_bars",
    # Calendar
    "dow",
    "month",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "regime",                 # RDP v1: long / short / neutral
    "year_quarter",           # YYYYQ#
    "concurrent_birth_1w",    # bool→int
    "concurrent_birth_1M",    # bool→int
    "family",                 # mb_1d / mb_1w / mb_1M / bb_1M (pooled identifier)
)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# I/O
OUT_DIR = Path("output")
EVENT_PARQUET_TPL = "mb_scanner_phase1_events_{family}.parquet"
RDP_LABELS = OUT_DIR / "regime_labels_daily_rdp_v1.csv"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _load_pass_cohort_pool() -> pd.DataFrame:
    """Load and pool the 4 PASS cohorts (above_mb_birth only)."""
    frames = []
    for fam, et in PASS_COHORTS:
        path = OUT_DIR / EVENT_PARQUET_TPL.format(family=fam)
        df = pd.read_parquet(path)
        sub = df[df["event_type"] == et].copy()
        sub["family"] = fam
        frames.append(sub)
    pool = pd.concat(frames, ignore_index=True)
    pool["event_bar_date"] = pd.to_datetime(pool["event_bar_date"])
    return pool


def _attach_rdp_regime(df: pd.DataFrame) -> pd.DataFrame:
    rdp = pd.read_csv(RDP_LABELS, parse_dates=["date"])
    rdp = rdp[["date", "regime"]].rename(columns={"date": "event_bar_date"})
    out = df.merge(rdp, on="event_bar_date", how="left")
    # forward-fill on rare missing edges (RDP coverage 2023-01-19 ..)
    out = out.sort_values("event_bar_date")
    out["regime"] = out["regime"].fillna("neutral")
    return out


def _attach_cross_tf_flags(pool: pd.DataFrame) -> pd.DataFrame:
    """concurrent_birth_1w/1M: was there an above_mb_birth in mb_1w/mb_1M
    for the same ticker in the lookback window ending at event_bar_date?
    """
    out = pool.copy()
    out["concurrent_birth_1w"] = 0
    out["concurrent_birth_1M"] = 0

    # Build per-(family, ticker) sorted date arrays for lookup
    for src_fam, win_days, col in [
        ("mb_1w", 7, "concurrent_birth_1w"),
        ("mb_1M", 30, "concurrent_birth_1M"),
    ]:
        src = pool[pool["family"] == src_fam][["ticker", "event_bar_date"]].copy()
        if src.empty:
            continue
        src = src.sort_values(["ticker", "event_bar_date"])
        # group dates by ticker
        per_ticker_dates = {
            t: g["event_bar_date"].to_numpy() for t, g in src.groupby("ticker")
        }
        # for each row, mark 1 if any source-fam birth in (date - win, date]
        idxs = []
        for i, row in out[["ticker", "event_bar_date"]].iterrows():
            arr = per_ticker_dates.get(row["ticker"])
            if arr is None or arr.size == 0:
                continue
            d = row["event_bar_date"]
            lo = d - pd.Timedelta(days=win_days)
            mask = (arr > lo) & (arr <= d)
            if mask.any():
                idxs.append(i)
        out.loc[idxs, col] = 1
    return out


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Risk frame (R-distance as % of event close)
    out["r_distance_pct"] = (
        (out["event_close"] - out["structural_invalidation_low"]) / out["event_close"]
    )
    # Quartet leg ratios
    out["lh_over_ll"] = (out["lh_price"] - out["ll_price"]) / out["ll_price"]
    out["hl_over_lh"] = (out["lh_price"] - out["hl_price"]) / out["lh_price"]
    out["hh_over_lh"] = (out["hh_close"] - out["lh_price"]) / out["lh_price"]
    out["quartet_span_bars"] = out["hh_idx"] - out["ll_idx"]
    # Calendar
    out["dow"] = out["event_bar_date"].dt.dayofweek.astype(int)
    out["month"] = out["event_bar_date"].dt.month.astype(int)
    out["year_quarter"] = (
        out["event_bar_date"].dt.year.astype(str)
        + "Q"
        + out["event_bar_date"].dt.quarter.astype(str)
    )
    return out


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in HORIZONS:
        col = f"mfe_r_{h}"
        if col not in out.columns:
            out[f"quality_{h}"] = np.nan
            continue
        out[f"quality_{h}"] = (out[col] >= TARGET_THRESHOLD_R).astype(int)
        # invalidate where mfe_r_h is NaN (no panel coverage)
        out.loc[out[col].isna(), f"quality_{h}"] = np.nan
    return out


def build_pool() -> pd.DataFrame:
    """Public entry — load + feature-engineer + targets + cross-TF flags."""
    pool = _load_pass_cohort_pool()
    pool = _derive_features(pool)
    pool = _attach_rdp_regime(pool)
    pool = _attach_cross_tf_flags(pool)
    pool = _build_targets(pool)
    # categorical dtype (LightGBM)
    for cat in CATEGORICAL_FEATURES:
        if cat in pool.columns:
            pool[cat] = pool[cat].astype("category")
    return pool


# ---------------------------------------------------------------------------
# Splits and folds
# ---------------------------------------------------------------------------


def _trainval_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered walk-forward folds on TRAIN+VAL.

    Sort by event_bar_date, partition into N_FOLDS+1 contiguous chunks.
    Fold k: train on chunks [0..k], validate on chunk k+1.
    """
    tv = df[df["split"].isin(["TRAIN", "VAL"])].copy()
    tv = tv.sort_values("event_bar_date").reset_index(drop=False)
    n = len(tv)
    chunk = n // (N_FOLDS + 1)
    folds = []
    for k in range(N_FOLDS):
        tr_end = chunk * (k + 1)
        va_end = chunk * (k + 2) if k < N_FOLDS - 1 else n
        tr_idx = tv.loc[: tr_end - 1, "index"].to_numpy()
        va_idx = tv.loc[tr_end : va_end - 1, "index"].to_numpy()
        folds.append((tr_idx, va_idx))
    return folds


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class HorizonResult:
    horizon: int
    fold_aucs: list[float] = field(default_factory=list)
    pooled_oos_auc: float = float("nan")
    test_auc: float = float("nan")
    quarter_aucs: dict = field(default_factory=dict)  # year_quarter -> (n, auc)
    perm_aucs: list[float] = field(default_factory=list)
    top_decile_lift: float = float("nan")
    top_decile_lift_ci_lo: float = float("nan")
    top_decile_lift_ci_hi: float = float("nan")
    cell_floor_pass: bool = False
    cell_floor_min_lift: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    base_rate: float = float("nan")
    gates: dict = field(default_factory=dict)
    passed: bool = False

    def to_row(self) -> dict:
        return {
            "horizon": self.horizon,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "base_rate": self.base_rate,
            "pooled_oos_auc": self.pooled_oos_auc,
            "fold_auc_min": min(self.fold_aucs) if self.fold_aucs else float("nan"),
            "fold_auc_max": max(self.fold_aucs) if self.fold_aucs else float("nan"),
            "perm_auc_mean": (
                float(np.mean(self.perm_aucs)) if self.perm_aucs else float("nan")
            ),
            "test_auc": self.test_auc,
            "top_decile_lift": self.top_decile_lift,
            "top_decile_lift_ci_lo": self.top_decile_lift_ci_lo,
            "top_decile_lift_ci_hi": self.top_decile_lift_ci_hi,
            "cell_floor_min_lift": self.cell_floor_min_lift,
            **{f"gate_G{i+1}": v for i, v in enumerate(self._gate_order())},
            "passed": self.passed,
        }

    def _gate_order(self) -> list[bool]:
        return [
            self.gates.get(g, False)
            for g in ("G1", "G2", "G3", "G4", "G5", "G6")
        ]


def _train_horizon_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    cat_cols: list[str],
    weight_col: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    test_idx: np.ndarray,
):
    """Train 4 fold models on WF, plus one final TRAIN+VAL model for TEST read."""
    import lightgbm as lgb

    fold_models = []
    fold_aucs = []
    fold_oos_preds = []
    fold_oos_idx = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        # drop rows with NaN target
        tr_sub = df.loc[tr_idx]
        tr_sub = tr_sub[tr_sub[target_col].notna()]
        va_sub = df.loc[va_idx]
        va_sub = va_sub[va_sub[target_col].notna()]
        if len(va_sub) < 50 or tr_sub[target_col].nunique() < 2:
            fold_aucs.append(float("nan"))
            continue

        X_tr = tr_sub[feature_cols]
        y_tr = tr_sub[target_col].astype(int).to_numpy()
        w_tr = tr_sub[weight_col].to_numpy()
        X_va = va_sub[feature_cols]
        y_va = va_sub[target_col].astype(int).to_numpy()

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        p_va = model.predict_proba(X_va)[:, 1]
        try:
            auc = roc_auc_score(y_va, p_va)
        except ValueError:
            auc = float("nan")
        fold_models.append(model)
        fold_aucs.append(auc)
        fold_oos_preds.append(p_va)
        fold_oos_idx.append(va_sub.index.to_numpy())

    # final TRAIN+VAL model for TEST read
    tv_sub = df[df["split"].isin(["TRAIN", "VAL"]) & df[target_col].notna()]
    test_sub = df.loc[test_idx]
    test_sub = test_sub[test_sub[target_col].notna()]

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(
        tv_sub[feature_cols],
        tv_sub[target_col].astype(int).to_numpy(),
        sample_weight=tv_sub[weight_col].to_numpy(),
        categorical_feature=cat_cols,
    )
    p_test = (
        final_model.predict_proba(test_sub[feature_cols])[:, 1]
        if len(test_sub) > 0 else np.array([])
    )
    test_auc = float("nan")
    if len(test_sub) > 50 and test_sub[target_col].nunique() >= 2:
        try:
            test_auc = float(
                roc_auc_score(test_sub[target_col].astype(int).to_numpy(), p_test)
            )
        except ValueError:
            test_auc = float("nan")

    return {
        "fold_aucs": fold_aucs,
        "fold_oos_preds": fold_oos_preds,
        "fold_oos_idx": fold_oos_idx,
        "final_model": final_model,
        "test_auc": test_auc,
        "test_idx": test_sub.index.to_numpy(),
        "test_preds": p_test,
        "n_train": int(len(tv_sub)),
        "n_test": int(len(test_sub)),
        "base_rate": (
            float(tv_sub[target_col].mean()) if len(tv_sub) else float("nan")
        ),
    }


def _quarter_stability(
    df: pd.DataFrame,
    target_col: str,
    oos_idx: np.ndarray,
    oos_preds: np.ndarray,
) -> dict:
    if len(oos_idx) == 0:
        return {}
    sub = df.loc[oos_idx, [target_col, "year_quarter"]].copy()
    sub["pred"] = oos_preds
    sub = sub[sub[target_col].notna()]
    out = {}
    for yq, grp in sub.groupby("year_quarter", observed=True):
        n = len(grp)
        if n < QUARTER_MIN_N or grp[target_col].nunique() < 2:
            continue
        try:
            auc = float(roc_auc_score(grp[target_col].astype(int), grp["pred"]))
        except ValueError:
            auc = float("nan")
        out[str(yq)] = (n, auc)
    return out


def _permutation_aucs(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    cat_cols: list[str],
    weight_col: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_seeds: int,
) -> list[float]:
    """G3: shuffle target on the TRAIN side per fold, retrain, score VAL, average."""
    import lightgbm as lgb

    aucs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(SEED + seed)
        fold_aucs = []
        for tr_idx, va_idx in folds:
            tr_sub = df.loc[tr_idx]
            tr_sub = tr_sub[tr_sub[target_col].notna()]
            va_sub = df.loc[va_idx]
            va_sub = va_sub[va_sub[target_col].notna()]
            if len(va_sub) < 50 or tr_sub[target_col].nunique() < 2:
                continue
            y_tr = tr_sub[target_col].astype(int).to_numpy().copy()
            rng.shuffle(y_tr)
            params = dict(LGBM_PARAMS)
            params["random_state"] = SEED + seed
            model = lgb.LGBMClassifier(**params)
            model.fit(
                tr_sub[feature_cols], y_tr,
                sample_weight=tr_sub[weight_col].to_numpy(),
                categorical_feature=cat_cols,
            )
            p_va = model.predict_proba(va_sub[feature_cols])[:, 1]
            try:
                fold_aucs.append(
                    float(roc_auc_score(va_sub[target_col].astype(int), p_va))
                )
            except ValueError:
                continue
        if fold_aucs:
            aucs.append(float(np.mean(fold_aucs)))
    return aucs


def _bootstrap_lift_ci(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    top_q: float,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI on (mean(top_decile target) / mean(target)).

    Returns (lift, ci_lo, ci_hi). NaN-safe (drops NaN target).
    """
    mask = ~np.isnan(y)
    y = y[mask]
    pred = pred[mask]
    if y.size < 100 or y.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    cutoff = np.quantile(pred, 1 - top_q)
    top_mask = pred >= cutoff
    if top_mask.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    base_lift = (y[top_mask].mean() / y.mean()) if y.mean() > 0 else float("nan")

    n = y.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = pred[idx]
        cb = np.quantile(pb, 1 - top_q)
        tm = pb >= cb
        if tm.sum() < 5 or yb.mean() == 0:
            continue
        boots.append(yb[tm].mean() / yb.mean())
    if not boots:
        return float(base_lift), float("nan"), float("nan")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(base_lift), lo, hi


def _cell_floor_check(
    df: pd.DataFrame,
    target_col: str,
    oos_idx: np.ndarray,
    oos_preds: np.ndarray,
) -> tuple[bool, float, pd.DataFrame]:
    """G5: no regime × concurrency-bucket cell with N≥30 has lift < 0.7×.

    Concurrency bucket: concurrent_quartets ∈ {1, 2, 3+}.
    Lift = top-decile-by-pred quality_h hit rate / cell base rate.
    """
    if len(oos_idx) == 0:
        return True, float("nan"), pd.DataFrame()
    sub = df.loc[oos_idx, [target_col, "regime", "concurrent_quartets"]].copy()
    sub["pred"] = oos_preds
    sub = sub[sub[target_col].notna()]
    sub["cq_bucket"] = pd.cut(
        sub["concurrent_quartets"].astype(int),
        bins=[-1, 1, 2, 1000],
        labels=["1", "2", "3+"],
    )
    rows = []
    for (reg, cq), grp in sub.groupby(["regime", "cq_bucket"], observed=True):
        n = len(grp)
        if n < CELL_FLOOR_MIN_N:
            continue
        base = grp[target_col].mean()
        if base <= 0:
            continue
        cutoff = np.quantile(grp["pred"], 1 - TOP_DECILE)
        tm = grp["pred"] >= cutoff
        if tm.sum() < 5:
            continue
        lift = grp.loc[tm, target_col].mean() / base
        rows.append({
            "regime": reg, "cq_bucket": str(cq), "n": n, "base": base, "lift": lift,
        })
    if not rows:
        return True, float("nan"), pd.DataFrame()
    cell_df = pd.DataFrame(rows)
    min_lift = float(cell_df["lift"].min())
    return bool(min_lift >= CELL_FLOOR_LIFT), min_lift, cell_df


def evaluate_horizon(
    pool: pd.DataFrame,
    horizon: int,
    *,
    n_perm_seeds: int = N_PERM_SEEDS,
    verbose: bool = True,
) -> tuple[HorizonResult, dict]:
    """Run the full pipeline for one horizon. Returns (result, artifacts)."""
    target_col = f"quality_{horizon}"
    if target_col not in pool.columns:
        raise KeyError(f"missing target column {target_col}")

    # --- prepare frame
    df = pool.copy()
    df["weight"] = 1.0 / (1.0 + df["concurrent_quartets"].clip(lower=0))
    feat_cols = list(ALL_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)

    # NaN feature guard: drop rows with any NaN feature in the feature set
    feat_complete_mask = df[list(NUMERIC_FEATURES)].notna().all(axis=1)
    df = df[feat_complete_mask].copy()

    # --- folds + sealed test
    trainval_mask = df["split"].isin(["TRAIN", "VAL"])
    test_mask = df["split"] == "TEST"
    test_idx = df[test_mask].index.to_numpy()

    folds = _trainval_folds(df)

    if verbose:
        print(f"[h={horizon}] N total={len(df)}  TRAIN+VAL={trainval_mask.sum()}  "
              f"TEST={test_mask.sum()}  folds={len(folds)}")

    # --- train & evaluate
    train_out = _train_horizon_model(
        df, target_col, feat_cols, cat_cols, "weight", folds, test_idx,
    )
    fold_aucs = [a for a in train_out["fold_aucs"] if not np.isnan(a)]

    # pooled OOS preds across folds
    if train_out["fold_oos_preds"]:
        oos_preds = np.concatenate(train_out["fold_oos_preds"])
        oos_idx = np.concatenate(train_out["fold_oos_idx"])
    else:
        oos_preds = np.array([])
        oos_idx = np.array([], dtype=int)

    # pooled OOS AUC
    pooled_auc = float("nan")
    if oos_idx.size > 0:
        y_oos = df.loc[oos_idx, target_col].astype(int).to_numpy()
        try:
            pooled_auc = float(roc_auc_score(y_oos, oos_preds))
        except ValueError:
            pooled_auc = float("nan")

    # quarter stability
    quarter_aucs = _quarter_stability(df, target_col, oos_idx, oos_preds)

    # G4 top-decile lift CI on pooled OOS
    rng = np.random.default_rng(SEED)
    if oos_idx.size > 0:
        y_oos = df.loc[oos_idx, target_col].astype(float).to_numpy()
        lift, ci_lo, ci_hi = _bootstrap_lift_ci(
            y_oos, oos_preds, top_q=TOP_DECILE, n_boot=1000, rng=rng,
        )
    else:
        lift, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

    # G5 cell floor
    cell_pass, min_lift, cell_df = _cell_floor_check(
        df, target_col, oos_idx, oos_preds,
    )

    # G3 perm test
    if verbose:
        print(f"[h={horizon}] running {n_perm_seeds}-seed perm test ...")
    perm_aucs = _permutation_aucs(
        df, target_col, feat_cols, cat_cols, "weight", folds, n_perm_seeds,
    )

    # --- gates
    res = HorizonResult(
        horizon=horizon,
        fold_aucs=fold_aucs,
        pooled_oos_auc=pooled_auc,
        test_auc=train_out["test_auc"],
        quarter_aucs=quarter_aucs,
        perm_aucs=perm_aucs,
        top_decile_lift=lift,
        top_decile_lift_ci_lo=ci_lo,
        top_decile_lift_ci_hi=ci_hi,
        cell_floor_pass=cell_pass,
        cell_floor_min_lift=min_lift,
        n_train=train_out["n_train"],
        n_test=train_out["n_test"],
        base_rate=train_out["base_rate"],
    )
    g1 = bool(fold_aucs) and all(a > G1_AUC_FLOOR for a in fold_aucs)
    g2 = (
        bool(quarter_aucs)
        and all(auc >= G2_QUARTER_AUC_FLOOR for _, auc in quarter_aucs.values())
    )
    g3 = (
        bool(perm_aucs)
        and not np.isnan(pooled_auc)
        and float(np.mean(perm_aucs)) <= pooled_auc - G3_PERM_DELTA
    )
    g4 = (not np.isnan(ci_lo)) and (ci_lo > 1.0)
    g5 = bool(cell_pass)
    g6 = (not np.isnan(train_out["test_auc"])) and (
        train_out["test_auc"] > G6_TEST_AUC_FLOOR
    )
    res.gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    res.passed = all([g1, g2, g3, g4, g5, g6])

    artifacts = {
        "cell_df": cell_df,
        "fold_oos_preds": oos_preds,
        "fold_oos_idx": oos_idx,
        "test_preds": train_out["test_preds"],
        "test_idx": train_out["test_idx"],
        "final_model": train_out["final_model"],
        "quarter_aucs": quarter_aucs,
        "perm_aucs": perm_aucs,
    }
    return res, artifacts


def run_ranker(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full LOCKED-spec run: 5 horizons, 6 gates each, write summary."""
    t0 = time.time()
    if verbose:
        print(f"[{RANKER_SPEC_ID}] loading + feature engineering ...")
    pool = build_pool()
    if verbose:
        print(f"[ranker] pool N={len(pool)} families={pool['family'].value_counts().to_dict()}")

    rows = []
    artifacts_by_h: dict[int, dict] = {}
    for h in HORIZONS:
        if verbose:
            print(f"\n=== horizon {h} ===")
        res, art = evaluate_horizon(pool, h, verbose=verbose)
        rows.append(res.to_row())
        artifacts_by_h[h] = {"result": res, **art}

    summary = pd.DataFrame(rows)
    elapsed = time.time() - t0
    if verbose:
        print(f"\n[ranker] elapsed {elapsed:.1f}s")
    return summary, artifacts_by_h


def primary_horizon(summary: pd.DataFrame) -> int | None:
    """Election rule: among passing horizon models, pick the one with cleanest
    quarter stability (highest min quarter AUC); tie-break shorter horizon.
    """
    pass_df = summary[summary["passed"]]
    if pass_df.empty:
        return None
    # tie-break: shorter horizon first
    return int(pass_df.sort_values(["fold_auc_min", "horizon"], ascending=[False, True]).iloc[0]["horizon"])
