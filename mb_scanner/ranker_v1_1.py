"""mb_scanner PASS-cohort multi-horizon ranker — LOCKED pre-reg v1.1.

Spec: `memory/mb_scanner_pass_cohort_ranker_v1_1.md` (LOCKED 2026-05-02).

Successor to v1 (PASS-DEGENERATE). v1's gates passed by literal logic,
but `r_distance_pct` was both the target's denominator AND a feature,
making the problem mechanical not predictive.

v1.1 changes (locked):
  - Pre-filter universe to per-cohort tradeable risk band:
      mb_1d r_distance_pct ∈ [2%, 10%]
      mb_1w r_distance_pct ∈ [8%, 25%]
      mb_1M, bb_1M DROPPED (too small post-band)
  - Target: absolute mfe_h ≥ T_h (NOT mfe_r — denominator removed)
  - Dual T_h schedule (aggressive + moderate) trained in parallel
    → 5 horizons × 2 schedules = 10 horizon-models
  - DROP 4 risk-proxy features: r_distance_pct, atr_pct_at_event,
    zone_width_atr, zone_width_pct
  - Same 6 gates / WF / sample weight / model

Anti-rescue: failure → v1.2 (Path C, separate pre-reg), no v1.1 tweak.
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

RANKER_SPEC_ID = "mb_scanner_pass_cohort_ranker_v1_1"

# Universe in scope (mb_1M / bb_1M DROPPED post-band)
PASS_COHORTS: tuple[tuple[str, str], ...] = (
    ("mb_1d", "above_mb_birth"),
    ("mb_1w", "above_mb_birth"),
)

# LOCKED per-cohort tradeable risk bands
TRADEABLE_BANDS: dict[str, tuple[float, float]] = {
    "mb_1d": (0.02, 0.10),
    "mb_1w": (0.08, 0.25),
}

HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)

# LOCKED dual T_h schedules (mfe_pct_h = mfe_h / event_close)
T_SCHEDULES: dict[str, dict[int, float]] = {
    "aggressive": {1: 0.04, 3: 0.07, 5: 0.10, 10: 0.15, 20: 0.22},
    "moderate":   {1: 0.03, 3: 0.06, 5: 0.08, 10: 0.12, 20: 0.18},
}

N_FOLDS = 4
N_PERM_SEEDS = 10
TOP_DECILE = 0.10
CELL_FLOOR_LIFT = 0.7
CELL_FLOOR_MIN_N = 30
QUARTER_MIN_N = 30
G1_AUC_FLOOR = 0.55
G2_QUARTER_AUC_FLOOR = 0.50
G3_PERM_DELTA = 0.03
G6_TEST_AUC_FLOOR = 0.52
SEED = 42

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

# v1.1 LOCKED feature set (4 risk-proxies REMOVED vs v1)
NUMERIC_FEATURES: tuple[str, ...] = (
    # Volume
    "vol_ratio_20_at_event",
    # Momentum
    "bos_distance_pct_at_event",
    "bos_distance_atr_at_event",
    # Zone (width DROPPED; only age kept)
    "zone_age_bars",
    # Quartet structure
    "concurrent_quartets",
    "pivot_confirm_lag_bars",
    "hh_to_event_lag_bars",
    # Quartet leg ratios
    "lh_over_ll",
    "hl_over_lh",
    "hh_over_lh",
    "quartet_span_bars",
    # Calendar
    "dow",
    "month",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "regime",
    "year_quarter",
    "concurrent_birth_1w",
    "concurrent_birth_1M",
    "family",
)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features explicitly DROPPED vs v1 (kept here as reference, not used)
DROPPED_RISK_PROXY_FEATURES: tuple[str, ...] = (
    "r_distance_pct",
    "atr_pct_at_event",
    "zone_width_atr",
    "zone_width_pct",
)

OUT_DIR = Path("output")
EVENT_PARQUET_TPL = "mb_scanner_phase1_events_{family}.parquet"
RDP_LABELS = OUT_DIR / "regime_labels_daily_rdp_v1.csv"


# ---------------------------------------------------------------------------
# Feature engineering (v1.1)
# ---------------------------------------------------------------------------


def _load_pool() -> pd.DataFrame:
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
    out = out.sort_values("event_bar_date")
    out["regime"] = out["regime"].fillna("neutral")
    return out


def _attach_cross_tf_flags(pool: pd.DataFrame) -> pd.DataFrame:
    """concurrent_birth_1w/1M lookback flags (preserved from v1)."""
    out = pool.copy()
    out["concurrent_birth_1w"] = 0
    out["concurrent_birth_1M"] = 0

    # Cross-TF lookups need the FULL universe (not just pooled-2). Re-read.
    # We use phase1 mb_1w / mb_1M parquets as event sources for the lookback.
    for src_fam, win_days, col in [
        ("mb_1w", 7, "concurrent_birth_1w"),
        ("mb_1M", 30, "concurrent_birth_1M"),
    ]:
        path = OUT_DIR / EVENT_PARQUET_TPL.format(family=src_fam)
        if not path.exists():
            continue
        src = pd.read_parquet(path)
        src = src[src["event_type"] == "above_mb_birth"][
            ["ticker", "event_bar_date"]
        ].copy()
        src["event_bar_date"] = pd.to_datetime(src["event_bar_date"])
        if src.empty:
            continue
        per_ticker_dates = {
            t: g["event_bar_date"].sort_values().to_numpy()
            for t, g in src.groupby("ticker")
        }
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
    # r_distance_pct kept ONLY for band filter — NOT in ALL_FEATURES.
    # Derivation matches v1; filter logic in _apply_tradeable_band.
    out["r_distance_pct"] = (
        (out["event_close"] - out["structural_invalidation_low"])
        / out["event_close"]
    )
    out["lh_over_ll"] = (out["lh_price"] - out["ll_price"]) / out["ll_price"]
    out["hl_over_lh"] = (out["lh_price"] - out["hl_price"]) / out["lh_price"]
    out["hh_over_lh"] = (out["hh_close"] - out["lh_price"]) / out["lh_price"]
    out["quartet_span_bars"] = out["hh_idx"] - out["ll_idx"]
    out["dow"] = out["event_bar_date"].dt.dayofweek.astype(int)
    out["month"] = out["event_bar_date"].dt.month.astype(int)
    out["year_quarter"] = (
        out["event_bar_date"].dt.year.astype(str)
        + "Q"
        + out["event_bar_date"].dt.quarter.astype(str)
    )
    return out


def _apply_tradeable_band(df: pd.DataFrame) -> pd.DataFrame:
    """LOCKED v1.1 per-cohort risk-band filter."""
    keep = pd.Series(False, index=df.index)
    for fam, (lo, hi) in TRADEABLE_BANDS.items():
        m = (df["family"] == fam) & (
            df["r_distance_pct"] >= lo
        ) & (df["r_distance_pct"] <= hi)
        keep = keep | m
    return df[keep].copy().reset_index(drop=True)


def _build_targets_absolute(df: pd.DataFrame) -> pd.DataFrame:
    """`mfe_pct_h = mfe_h / event_close`, then quality_<schedule>_<h> bool."""
    out = df.copy()
    for h in HORIZONS:
        col = f"mfe_{h}"
        out[f"mfe_pct_{h}"] = out[col] / out["event_close"]
    for sched, T in T_SCHEDULES.items():
        for h in HORIZONS:
            mfe_pct = out[f"mfe_pct_{h}"]
            tgt = (mfe_pct >= T[h]).astype(float)
            tgt[mfe_pct.isna()] = np.nan
            out[f"quality_{sched}_{h}"] = tgt
    return out


def build_pool() -> pd.DataFrame:
    pool = _load_pool()
    pool = _derive_features(pool)
    pool = _attach_rdp_regime(pool)
    pool = _attach_cross_tf_flags(pool)
    pool = _apply_tradeable_band(pool)         # LOCKED band filter
    pool = _build_targets_absolute(pool)        # absolute mfe_h ≥ T_h
    for cat in CATEGORICAL_FEATURES:
        if cat in pool.columns:
            pool[cat] = pool[cat].astype("category")
    return pool


# ---------------------------------------------------------------------------
# Folds (identical to v1)
# ---------------------------------------------------------------------------


def _trainval_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
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
# Cell result + training (modeled on v1)
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    schedule: str
    horizon: int
    fold_aucs: list[float] = field(default_factory=list)
    pooled_oos_auc: float = float("nan")
    test_auc: float = float("nan")
    quarter_aucs: dict = field(default_factory=dict)
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
            "schedule": self.schedule,
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
            "gate_G1": self.gates.get("G1", False),
            "gate_G2": self.gates.get("G2", False),
            "gate_G3": self.gates.get("G3", False),
            "gate_G4": self.gates.get("G4", False),
            "gate_G5": self.gates.get("G5", False),
            "gate_G6": self.gates.get("G6", False),
            "passed": self.passed,
        }


def _train_cell(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    cat_cols: list[str],
    weight_col: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    test_idx: np.ndarray,
):
    import lightgbm as lgb
    fold_aucs, fold_oos_preds, fold_oos_idx = [], [], []
    for tr_idx, va_idx in folds:
        tr_sub = df.loc[tr_idx]
        tr_sub = tr_sub[tr_sub[target_col].notna()]
        va_sub = df.loc[va_idx]
        va_sub = va_sub[va_sub[target_col].notna()]
        if len(va_sub) < 50 or tr_sub[target_col].nunique() < 2:
            fold_aucs.append(float("nan"))
            continue
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            tr_sub[feature_cols], tr_sub[target_col].astype(int).to_numpy(),
            sample_weight=tr_sub[weight_col].to_numpy(),
            eval_set=[(va_sub[feature_cols], va_sub[target_col].astype(int).to_numpy())],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        p_va = model.predict_proba(va_sub[feature_cols])[:, 1]
        try:
            auc = roc_auc_score(va_sub[target_col].astype(int).to_numpy(), p_va)
        except ValueError:
            auc = float("nan")
        fold_aucs.append(auc)
        fold_oos_preds.append(p_va)
        fold_oos_idx.append(va_sub.index.to_numpy())

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


def _quarter_stability(df, target_col, oos_idx, oos_preds):
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


def _permutation_aucs(df, target_col, feat_cols, cat_cols, weight_col, folds, n_seeds):
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
            params = dict(LGBM_PARAMS); params["random_state"] = SEED + seed
            model = lgb.LGBMClassifier(**params)
            model.fit(
                tr_sub[feat_cols], y_tr,
                sample_weight=tr_sub[weight_col].to_numpy(),
                categorical_feature=cat_cols,
            )
            p_va = model.predict_proba(va_sub[feat_cols])[:, 1]
            try:
                fold_aucs.append(
                    float(roc_auc_score(va_sub[target_col].astype(int), p_va))
                )
            except ValueError:
                continue
        if fold_aucs:
            aucs.append(float(np.mean(fold_aucs)))
    return aucs


def _bootstrap_lift_ci(y, pred, *, top_q, n_boot, rng, alpha=0.05):
    mask = ~np.isnan(y)
    y = y[mask]; pred = pred[mask]
    if y.size < 100 or y.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    cutoff = np.quantile(pred, 1 - top_q)
    top_mask = pred >= cutoff
    if top_mask.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    base_lift = y[top_mask].mean() / y.mean() if y.mean() > 0 else float("nan")
    n = y.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]; pb = pred[idx]
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


def _cell_floor_check(df, target_col, oos_idx, oos_preds):
    if len(oos_idx) == 0:
        return True, float("nan"), pd.DataFrame()
    sub = df.loc[oos_idx, [target_col, "regime", "concurrent_quartets"]].copy()
    sub["pred"] = oos_preds
    sub = sub[sub[target_col].notna()]
    sub["cq_bucket"] = pd.cut(
        sub["concurrent_quartets"].astype(int),
        bins=[-1, 1, 2, 1000], labels=["1", "2", "3+"],
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
        rows.append({"regime": reg, "cq_bucket": str(cq), "n": n,
                     "base": base, "lift": lift})
    if not rows:
        return True, float("nan"), pd.DataFrame()
    cell_df = pd.DataFrame(rows)
    min_lift = float(cell_df["lift"].min())
    return bool(min_lift >= CELL_FLOOR_LIFT), min_lift, cell_df


def evaluate_cell(pool, schedule: str, horizon: int, *, n_perm_seeds=N_PERM_SEEDS,
                  verbose=True):
    target_col = f"quality_{schedule}_{horizon}"
    if target_col not in pool.columns:
        raise KeyError(f"missing target column {target_col}")

    df = pool.copy()
    df["weight"] = 1.0 / (1.0 + df["concurrent_quartets"].clip(lower=0))
    feat_cols = list(ALL_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)

    feat_complete_mask = df[list(NUMERIC_FEATURES)].notna().all(axis=1)
    df = df[feat_complete_mask].copy()

    trainval_mask = df["split"].isin(["TRAIN", "VAL"])
    test_mask = df["split"] == "TEST"
    test_idx = df[test_mask].index.to_numpy()

    folds = _trainval_folds(df)

    if verbose:
        print(f"[{schedule}/h={horizon}] N={len(df)} TV={trainval_mask.sum()} "
              f"TEST={test_mask.sum()} folds={len(folds)}")

    train_out = _train_cell(df, target_col, feat_cols, cat_cols, "weight",
                            folds, test_idx)
    fold_aucs = [a for a in train_out["fold_aucs"] if not np.isnan(a)]

    if train_out["fold_oos_preds"]:
        oos_preds = np.concatenate(train_out["fold_oos_preds"])
        oos_idx = np.concatenate(train_out["fold_oos_idx"])
    else:
        oos_preds = np.array([]); oos_idx = np.array([], dtype=int)

    pooled_auc = float("nan")
    if oos_idx.size > 0:
        y_oos = df.loc[oos_idx, target_col].astype(int).to_numpy()
        try:
            pooled_auc = float(roc_auc_score(y_oos, oos_preds))
        except ValueError:
            pooled_auc = float("nan")

    quarter_aucs = _quarter_stability(df, target_col, oos_idx, oos_preds)

    rng = np.random.default_rng(SEED)
    if oos_idx.size > 0:
        y_oos = df.loc[oos_idx, target_col].astype(float).to_numpy()
        lift, ci_lo, ci_hi = _bootstrap_lift_ci(
            y_oos, oos_preds, top_q=TOP_DECILE, n_boot=1000, rng=rng,
        )
    else:
        lift, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

    cell_pass, min_lift, cell_df = _cell_floor_check(
        df, target_col, oos_idx, oos_preds,
    )

    if verbose:
        print(f"[{schedule}/h={horizon}] running {n_perm_seeds}-seed perm test ...")
    perm_aucs = _permutation_aucs(df, target_col, feat_cols, cat_cols,
                                  "weight", folds, n_perm_seeds)

    res = CellResult(
        schedule=schedule, horizon=horizon,
        fold_aucs=fold_aucs, pooled_oos_auc=pooled_auc,
        test_auc=train_out["test_auc"], quarter_aucs=quarter_aucs,
        perm_aucs=perm_aucs, top_decile_lift=lift,
        top_decile_lift_ci_lo=ci_lo, top_decile_lift_ci_hi=ci_hi,
        cell_floor_pass=cell_pass, cell_floor_min_lift=min_lift,
        n_train=train_out["n_train"], n_test=train_out["n_test"],
        base_rate=train_out["base_rate"],
    )
    g1 = bool(fold_aucs) and all(a > G1_AUC_FLOOR for a in fold_aucs)
    g2 = (bool(quarter_aucs) and all(auc >= G2_QUARTER_AUC_FLOOR
                                     for _, auc in quarter_aucs.values()))
    g3 = (bool(perm_aucs) and not np.isnan(pooled_auc)
          and float(np.mean(perm_aucs)) <= pooled_auc - G3_PERM_DELTA)
    g4 = (not np.isnan(ci_lo)) and (ci_lo > 1.0)
    g5 = bool(cell_pass)
    g6 = (not np.isnan(train_out["test_auc"])) and (
        train_out["test_auc"] > G6_TEST_AUC_FLOOR
    )
    res.gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    res.passed = all([g1, g2, g3, g4, g5, g6])

    artifacts = {
        "cell_df": cell_df, "fold_oos_preds": oos_preds,
        "fold_oos_idx": oos_idx, "test_preds": train_out["test_preds"],
        "test_idx": train_out["test_idx"], "final_model": train_out["final_model"],
        "quarter_aucs": quarter_aucs, "perm_aucs": perm_aucs,
    }
    return res, artifacts


def run_ranker(verbose=True):
    """Full v1.1 LOCKED run: 2 schedules × 5 horizons = 10 cells."""
    t0 = time.time()
    if verbose:
        print(f"[{RANKER_SPEC_ID}] loading + feature engineering ...")
    pool = build_pool()
    if verbose:
        fams = pool["family"].value_counts().to_dict()
        print(f"[ranker] post-band pool N={len(pool)} families={fams}")

    rows = []
    artifacts_by_cell = {}
    for sched in T_SCHEDULES:
        for h in HORIZONS:
            if verbose:
                print(f"\n=== {sched} / horizon {h} ===")
            res, art = evaluate_cell(pool, sched, h, verbose=verbose)
            rows.append(res.to_row())
            artifacts_by_cell[(sched, h)] = {"result": res, **art}

    summary = pd.DataFrame(rows)
    elapsed = time.time() - t0
    if verbose:
        print(f"\n[ranker] elapsed {elapsed:.1f}s")
    return summary, artifacts_by_cell
