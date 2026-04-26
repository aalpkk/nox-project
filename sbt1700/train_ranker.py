"""SBT-1700 RESET — locked LightGBM regression ranker.

Methodology rules baked into this module:

- Single model class (LightGBM regression on `realized_R_net`).
- Hyperparameters are *locked* in `LGBM_REGRESSION_PARAMS`. There is no
  sweep, no random search, no per-fold tuning. Validation/test scores
  must reflect a single model decision, not a search over models.
- Feature blacklist (`FEATURE_BLACKLIST`) is enforced before fitting.
  Label-derived columns, identifiers, and obviously trade-shape
  parameters (entry_px / stop_px / tp_px / atr_1700) are dropped so
  the ranker cannot learn to "predict R from R".
- Date-based walk-forward only. No row-shuffled folds.
- Training data is whatever the caller passes — the reset orchestrator
  is responsible for confining this to the train split. This module
  does not load splits itself, on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import numpy as np
import pandas as pd

import lightgbm as lgb


LGBM_REGRESSION_PARAMS: dict = dict(
    objective="regression",
    metric="rmse",
    num_leaves=31,
    learning_rate=0.05,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbose=-1,
    seed=17,
)
N_ESTIMATORS: int = 200

# Columns that must never enter the feature matrix. Mix of (a) labels
# emitted by build_dataset / re-simulation, (b) per-trade parameters
# trivially derivable from labels, (c) identifiers / raw date strings.
FEATURE_BLACKLIST: frozenset[str] = frozenset({
    # E3 / multi-exit labels
    "realized_R_gross", "realized_R_net", "win_label",
    "tp_hit", "sl_hit", "timeout_hit", "partial_hit",
    "exit_reason", "exit_date", "bars_held",
    "entry_px", "stop_px", "tp_px", "partial_px",
    "initial_R_price", "exit_px", "cost_R",
    "exit_variant",
    # ATR used for stop sizing — feature only if discovery decides;
    # excluded by default to keep the ranker from learning entry_px ratios.
    "atr_1700",
    # Identifiers / metadata
    "ticker", "date", "schema_version",
    "intraday_coverage", "missing_bar_count", "n_bars_1700",
    "last_bar_ts_tr",
    # Forward-bar projections that should never exist in features
    "next_open", "next_high", "next_low", "next_close",
    "future_high", "future_low", "future_close",
})

DEFAULT_LABEL_COL: str = "realized_R_net"
N_WF_FOLDS: int = 3
TOP_DECILE_FRAC: float = 0.10


@dataclass
class TrainArtifacts:
    model: lgb.Booster
    feature_cols: list[str]
    n_train: int
    importance: pd.DataFrame


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns minus the blacklist. Stable order."""
    cols: list[str] = []
    for c in df.columns:
        if c in FEATURE_BLACKLIST:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def assert_no_blacklist_leak(feature_cols: Iterable[str]) -> None:
    leaked = sorted(set(feature_cols) & FEATURE_BLACKLIST)
    if leaked:
        raise RuntimeError(
            f"feature blacklist breach: {leaked}. The ranker would learn "
            "from label-derived columns. Refusing to fit."
        )


def fit_model(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    feature_cols: list[str] | None = None,
) -> TrainArtifacts:
    if feature_cols is None:
        feature_cols = select_feature_columns(df)
    assert_no_blacklist_leak(feature_cols)
    sub = df.dropna(subset=[label_col]).copy()
    if sub.empty:
        raise ValueError(f"no rows with non-null {label_col!r}")
    X = sub[feature_cols]
    y = sub[label_col].astype(float)
    train_set = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(
        params=LGBM_REGRESSION_PARAMS,
        train_set=train_set,
        num_boost_round=N_ESTIMATORS,
    )
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    return TrainArtifacts(
        model=booster,
        feature_cols=feature_cols,
        n_train=len(sub),
        importance=importance,
    )


def _date_chunks(dates: pd.Series, k: int) -> list[pd.Series]:
    s = dates.sort_values().reset_index(drop=True)
    return [pd.Series(c).reset_index(drop=True) for c in np.array_split(s, k)]


def _evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    yt = pd.Series(np.asarray(y_true)).reset_index(drop=True)
    yp = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    rho_value = float("nan")
    if len(yt) >= 5:
        rho_value = float(yt.rank().corr(yp.rank()))
    n = len(yt)
    if n == 0:
        return dict(spearman_rho=rho_value, all_avg_R=float("nan"),
                    top_decile_n=0, top_decile_avg_R=float("nan"),
                    top_decile_PF=float("nan"), top_decile_WR=float("nan"))
    top_n = max(1, int(round(TOP_DECILE_FRAC * n)))
    order = yp.sort_values(ascending=False).index
    top_idx = order[:top_n]
    top_R = yt.iloc[top_idx]
    wins = top_R[top_R > 0].sum()
    losses = -top_R[top_R < 0].sum()
    pf = float("inf") if losses == 0 and wins > 0 else (
        float(wins / losses) if losses > 0 else float("nan"))
    return dict(
        spearman_rho=rho_value,
        all_avg_R=float(y_true.mean()),
        top_decile_n=int(top_n),
        top_decile_avg_R=float(top_R.mean()),
        top_decile_PF=pf,
        top_decile_WR=float((top_R > 0).mean()),
    )


def walk_forward(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    k: int = N_WF_FOLDS,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """k-fold expanding-window walk-forward by signal date."""
    if "date" not in df.columns:
        raise KeyError("walk_forward requires a 'date' column")
    sub = df.dropna(subset=[label_col]).copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = select_feature_columns(sub)
    assert_no_blacklist_leak(feature_cols)

    unique_dates = pd.Series(sub["date"].drop_duplicates().sort_values().values)
    if len(unique_dates) < k + 1:
        raise ValueError(
            f"need at least {k+1} unique signal dates for {k}-fold WF; "
            f"got {len(unique_dates)}"
        )
    chunks = _date_chunks(unique_dates, k + 1)  # k+1 chunks => k folds

    rows: list[dict] = []
    for i in range(1, len(chunks)):
        train_dates = pd.concat(chunks[:i], ignore_index=True)
        val_dates = chunks[i]
        train_df = sub[sub["date"].isin(set(train_dates))]
        val_df = sub[sub["date"].isin(set(val_dates))]
        if train_df.empty or val_df.empty:
            rows.append(dict(fold=f"fold_{i}", n_train=len(train_df),
                             n_val=len(val_df), spearman_rho=float("nan"),
                             all_avg_R=float("nan"), top_decile_n=0,
                             top_decile_avg_R=float("nan"),
                             top_decile_PF=float("nan"),
                             top_decile_WR=float("nan")))
            continue
        artifacts = fit_model(train_df, label_col=label_col,
                              feature_cols=feature_cols)
        preds = artifacts.model.predict(val_df[feature_cols])
        metrics = _evaluate_predictions(val_df[label_col], pd.Series(preds))
        rows.append(dict(
            fold=f"fold_{i}",
            n_train=len(train_df),
            n_val=len(val_df),
            **metrics,
        ))
    return pd.DataFrame(rows)


def save_artifacts(
    artifacts: TrainArtifacts,
    out_dir: str | Path,
    tag: str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"sbt_1700_reset_model_{tag}.txt"
    imp_path = out_dir / f"sbt_1700_reset_importance_{tag}.csv"
    feat_path = out_dir / f"sbt_1700_reset_features_{tag}.json"
    artifacts.model.save_model(str(model_path))
    artifacts.importance.to_csv(imp_path, index=False)
    feat_path.write_text(json.dumps({
        "feature_cols": artifacts.feature_cols,
        "n_train": artifacts.n_train,
        "label_col": DEFAULT_LABEL_COL,
        "lgbm_params": LGBM_REGRESSION_PARAMS,
        "n_estimators": N_ESTIMATORS,
        "blacklist_size": len(FEATURE_BLACKLIST),
    }, indent=2))
    return dict(
        model_path=str(model_path),
        importance_path=str(imp_path),
        feature_manifest_path=str(feat_path),
    )
