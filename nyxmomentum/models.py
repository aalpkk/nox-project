"""
Three models for cross-sectional ranking, each exposing the same
walk-forward interface:

  M0 : handcrafted composite (no fit — rule-based ranker, baseline bar)
  M1 : linear/ridge regressor on L2 (per-date z-scored features)
  M2 : LightGBM regressor on L2 (tree ensemble)

TRAINING PROTOCOL (locked 2026-04-21, per Step 4 directive):
  • Strict walk-forward, expanding train, NO random split
  • Features z-scored cross-sectionally per rebalance_date independently for
    every row — per-date stats use ONLY that date's eligible names (no
    lookahead into train from test or vice versa since each row consumes
    only its own date's cross-section)
  • Train on rows where eligible=True AND target present (not NaN)
  • Early stopping for LightGBM via the fold's validation window only;
    final test fold never sees val-set tuning
  • No aggressive hyperparameter search — CONFIG.model defaults only

FEATURE GOVERNANCE:
  Every model declares its feature set up-front via feature_list(). The
  runner emits a manifest table showing which columns are:
    • train_feature  (goes into X)
    • overlay_input  (enters overlay only, not X)
    • diagnostic     (never in X, never in overlay)
  Silently promoting an ex-ante proxy into X is forbidden without an
  explicit flag in the manifest (directive #7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

from .baselines import handcrafted_composite_score
from .features import FEATURE_COLUMNS
from .walk_forward import FoldSplit


# ── Helpers ───────────────────────────────────────────────────────────────────

def cs_zscore_columns(df: pd.DataFrame,
                      feat_cols: list[str]) -> pd.DataFrame:
    """Per-rebalance-date cross-sectional z-score of each column. NaN → NaN
    (caller decides whether to fill). Returns a copy with feat_cols replaced."""
    out = df.copy()

    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / sd

    for c in feat_cols:
        out[c] = df.groupby("rebalance_date", sort=False)[c].transform(_z)
    return out


# ── Base + model descriptors ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    name: str                     # "M0", "M1", "M2"
    family: str                   # "handcrafted" | "ridge" | "lightgbm"
    features: tuple[str, ...]     # model-feature column names
    description: str


def _make_eligible_train_frame(df: pd.DataFrame,
                                feat_cols: list[str],
                                target_col: str) -> pd.DataFrame:
    """Keep only eligible rows with non-null target. NaN features allowed for
    LightGBM; caller decides whether to fill for ridge."""
    mask = df["eligible"].astype(bool) & df[target_col].notna()
    return df.loc[mask].copy()


# ── M0 : handcrafted composite (no fit) ───────────────────────────────────────

def predict_m0(panel: pd.DataFrame) -> pd.Series:
    """Apply the current _HANDCRAFTED_WEIGHTS per-date composite to every row."""
    return handcrafted_composite_score(panel).astype(float)


def run_m0(panel: pd.DataFrame, folds: list[FoldSplit]) -> pd.DataFrame:
    """M0 has no fit — it predicts on every test row using the live composite.
    Returns long-format predictions over union of test folds."""
    preds = predict_m0(panel)
    out = panel[["ticker", "rebalance_date", "eligible"]].copy()
    out["prediction"] = preds.values

    test_mask = np.zeros(len(panel), dtype=bool)
    for f in folds:
        test_mask |= f.test_mask(panel["rebalance_date"]).values
    out = out.loc[test_mask].copy()

    # fold-id tag for per-fold diagnostics
    out["fold_id"] = ""
    for f in folds:
        m = f.test_mask(out["rebalance_date"])
        out.loc[m, "fold_id"] = f.fold_id
    return out.reset_index(drop=True)


# ── M1 : ridge on per-date z-scored features ─────────────────────────────────

def run_m1(panel: pd.DataFrame,
           folds: list[FoldSplit],
           features: Iterable[str],
           target_col: str = "l2_excess_vs_universe_median",
           alpha: float = 1.0) -> pd.DataFrame:
    """Fit ridge per fold on train window, predict on test window. Missing
    per-date z-scores filled with 0 (median of the cross-section)."""
    feat_cols = list(features)
    z_panel = cs_zscore_columns(panel, feat_cols)
    z_panel[feat_cols] = z_panel[feat_cols].fillna(0.0)

    rows: list[pd.DataFrame] = []
    for f in folds:
        tr_mask = f.train_mask(z_panel["rebalance_date"])
        te_mask = f.test_mask(z_panel["rebalance_date"])
        train = _make_eligible_train_frame(z_panel.loc[tr_mask], feat_cols, target_col)
        if train.empty:
            continue
        X_tr = train[feat_cols].values
        y_tr = train[target_col].values.astype(float)
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_tr, y_tr)

        test = z_panel.loc[te_mask, ["ticker", "rebalance_date", "eligible", *feat_cols]].copy()
        X_te = test[feat_cols].values
        test["prediction"] = model.predict(X_te)
        test["fold_id"] = f.fold_id
        rows.append(test[["ticker", "rebalance_date", "eligible", "prediction", "fold_id"]])

    if not rows:
        return pd.DataFrame(columns=["ticker", "rebalance_date", "eligible",
                                     "prediction", "fold_id"])
    return pd.concat(rows, ignore_index=True)


# ── M2 : LightGBM regressor ───────────────────────────────────────────────────

def run_m2(panel: pd.DataFrame,
           folds: list[FoldSplit],
           features: Iterable[str],
           target_col: str = "l2_excess_vs_universe_median",
           num_leaves: int = 31,
           learning_rate: float = 0.05,
           n_estimators: int = 500,
           min_data_in_leaf: int = 200,
           feature_fraction: float = 0.8,
           bagging_fraction: float = 0.8,
           early_stopping_rounds: int = 50,
           random_state: int = 42) -> pd.DataFrame:
    """LightGBM regressor with the fold's val window for early stopping. The
    test window is NEVER seen during training. Per-date z-scored features are
    passed as-is (LightGBM is scale-invariant, but z-scoring removes date-
    level drift that would otherwise split on calendar context)."""
    import lightgbm as lgb

    feat_cols = list(features)
    z_panel = cs_zscore_columns(panel, feat_cols)

    rows: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for f in folds:
        tr_mask = f.train_mask(z_panel["rebalance_date"])
        va_mask = f.val_mask(z_panel["rebalance_date"])
        te_mask = f.test_mask(z_panel["rebalance_date"])

        train = _make_eligible_train_frame(z_panel.loc[tr_mask], feat_cols, target_col)
        val = _make_eligible_train_frame(z_panel.loc[va_mask], feat_cols, target_col)
        if train.empty:
            continue

        X_tr = train[feat_cols].values
        y_tr = train[target_col].values.astype(float)
        eval_sets = []
        callbacks = []
        if not val.empty:
            X_va = val[feat_cols].values
            y_va = val[target_col].values.astype(float)
            eval_sets = [(X_va, y_va)]
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_data_in_leaf=min_data_in_leaf,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=eval_sets if eval_sets else None,
            callbacks=callbacks,
        )

        test = z_panel.loc[te_mask, ["ticker", "rebalance_date", "eligible", *feat_cols]].copy()
        X_te = test[feat_cols].values
        test["prediction"] = model.predict(X_te)
        test["fold_id"] = f.fold_id
        rows.append(test[["ticker", "rebalance_date", "eligible", "prediction", "fold_id"]])

        imp = pd.DataFrame({
            "fold_id": f.fold_id,
            "feature": feat_cols,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        })
        importances.append(imp)

    if not rows:
        return pd.DataFrame(columns=["ticker", "rebalance_date", "eligible",
                                     "prediction", "fold_id"])
    preds = pd.concat(rows, ignore_index=True)
    preds.attrs["importances"] = (
        pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    )
    return preds


# ── Governance manifest ───────────────────────────────────────────────────────

def feature_governance_manifest(model_features: dict[str, Iterable[str]],
                                 overlay_inputs: Iterable[str],
                                 diagnostic_columns: Iterable[str],
                                 ) -> pd.DataFrame:
    """Explicit who-is-where table. One row per unique column, columns indicate
    which model(s) consume it, whether overlay also uses it, and diagnostic
    status. A column used both as train_feature AND overlay_input is flagged
    with `dual_use=True` — that is the thing the user wants to see out loud."""
    all_cols: set[str] = set()
    for cols in model_features.values():
        all_cols.update(cols)
    all_cols.update(overlay_inputs)
    all_cols.update(diagnostic_columns)

    rows: list[dict] = []
    for col in sorted(all_cols):
        row = {"column": col}
        for mname, cols in model_features.items():
            row[f"in_{mname}"] = col in set(cols)
        row["overlay_input"] = col in set(overlay_inputs)
        row["diagnostic_only"] = col in set(diagnostic_columns)
        in_any_model = any(row.get(f"in_{m}", False) for m in model_features.keys())
        row["dual_use_model_and_overlay"] = in_any_model and row["overlay_input"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Factory for the three canonical specs ─────────────────────────────────────

def default_model_specs() -> tuple[ModelSpec, ...]:
    all_feats = tuple(FEATURE_COLUMNS)
    return (
        ModelSpec(
            name="M0",
            family="handcrafted",
            features=tuple(),  # composite reaches into features directly
            description="Handcrafted composite baseline (recent_extreme_21d dropped).",
        ),
        ModelSpec(
            name="M1",
            family="ridge",
            features=all_feats,
            description="Ridge on per-date z-scored features. Linear ranker.",
        ),
        ModelSpec(
            name="M2",
            family="lightgbm",
            features=all_feats,
            description="LightGBM regressor on L2. Tree ensemble for nonlinearity check.",
        ),
    )
