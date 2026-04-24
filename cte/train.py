"""
CTE train — single-head LGBM classifier (walk-forward).

Default target: runner_{primary_target_h} (LabelParams.primary_target, default
"runner_15"). Sıkı trigger → düşük base rate (~24%) → LGBM küçük ama
hedefli etki için yeterli.

3 fold expanding walk-forward (CONFIG.split donmuş):
  fold1: train→2023-04-30  val 2023-05-15..2023-10-31  test 2023-11-15..2024-04-30
  fold2: train→2024-04-30  val 2024-05-15..2024-10-31  test 2024-11-15..2025-04-30
  fold3: train→2025-04-30  val 2025-05-15..2025-10-31  test 2025-11-15..2026-04-30

Baselines:
  - random            : uniform(0,1)
  - score_compression : compression_score percentile rank (per-date)

Output:
  output/cte_preds_v1.parquet
    ticker, date, fold_assigned, setup_type, <target>,
    score_model, score_random, score_compression, + all label + key feature cols
  output/cte_importance_v1.csv   feature importance per fold (gain)

Kullanım:
  python -m cte.train                       # target=runner_15, features=v1
  python -m cte.train --target runner_10    # daha kısa horizon
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

import lightgbm as lgb

from cte.config import CONFIG
from cte.features import FEATURES_V1


@dataclass(frozen=True)
class LGBMParams:
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_child_samples: int = 20  # CTE trigger sayısı az (~10k), küçük leaves
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 100
    seed: int = 17


def _per_date_rank(s: pd.Series, dates: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank per date (NaN preserves)."""
    df = pd.DataFrame({"s": s, "d": dates})
    return df.groupby("d")["s"].rank(method="average", pct=True, na_option="keep")


def train_fold(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    fold_name: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    params: LGBMParams,
) -> tuple[lgb.Booster, pd.DataFrame, pd.Series]:
    d = pd.to_datetime(data["date"])
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    vs = pd.Timestamp(val_start)
    ve = pd.Timestamp(val_end)
    qs = pd.Timestamp(test_start)
    qe = pd.Timestamp(test_end)

    tr_mask = (d >= ts) & (d <= te) & data[target_col].notna()
    va_mask = (d >= vs) & (d <= ve) & data[target_col].notna()
    qt_mask = (d >= qs) & (d <= qe) & data[target_col].notna()

    tr = data.loc[tr_mask].copy()
    va = data.loc[va_mask].copy()
    qt = data.loc[qt_mask].copy()

    X_tr = tr[feature_cols]
    y_tr = tr[target_col].astype(int)
    X_va = va[feature_cols]
    y_va = va[target_col].astype(int)
    X_qt = qt[feature_cols]

    print(
        f"  [{fold_name}] train N={len(tr):,}  val N={len(va):,}  test N={len(qt):,}  "
        f"(pos tr/va/qt: {y_tr.mean()*100:.1f}% / {y_va.mean()*100:.1f}% / "
        f"{qt[target_col].mean()*100:.1f}%)"
    )

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        max_depth=params.max_depth,
        min_child_samples=params.min_child_samples,
        feature_fraction=params.feature_fraction,
        bagging_fraction=params.bagging_fraction,
        bagging_freq=params.bagging_freq,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=params.seed,
        verbose=-1,
        n_jobs=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = clf.best_iteration_
    print(f"      best_iter={best_iter}")

    scores = clf.predict_proba(X_qt, num_iteration=best_iter)[:, 1]
    qt["score_model"] = scores
    qt["fold_assigned"] = fold_name
    rng = np.random.default_rng(params.seed + hash(fold_name) % 1000)
    qt["score_random"] = rng.uniform(0, 1, size=len(qt))
    if "compression_score" in qt.columns:
        qt["score_compression"] = _per_date_rank(qt["compression_score"], qt["date"]).values
    else:
        qt["score_compression"] = np.nan

    imp = pd.Series(
        clf.booster_.feature_importance(importance_type="gain"),
        index=feature_cols,
        name=fold_name,
    )
    return clf.booster_, qt, imp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument(
        "--target",
        default=CONFIG.label.primary_target,
        help="binary label column (default from CONFIG.label.primary_target)",
    )
    ap.add_argument("--out-preds", default="output/cte_preds_v1.parquet")
    ap.add_argument("--out-imp", default="output/cte_importance_v1.csv")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}  →  python -m cte.dataset")
        return 2

    print("═══ CTE Train v1 ═══")
    print(f"Dataset: {args.dataset}")

    df = pd.read_parquet(args.dataset)
    print(f"  shape: {df.shape}")
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [c for c in FEATURES_V1 if c in df.columns]
    missing = [c for c in FEATURES_V1 if c not in df.columns]
    if missing:
        print(f"⚠ missing feature columns (dropped): {missing}")
    print(f"  feature count: {len(feature_cols)}")
    print(f"  target: {args.target}")
    if args.target not in df.columns:
        print(f"❌ target {args.target} not in dataset columns")
        return 3

    params = LGBMParams()
    split = CONFIG.split

    all_test = []
    all_imp: list[pd.Series] = []
    for fs in split.folds:
        print(
            f"\n[FOLD {fs.name}] train→{fs.train_end}  "
            f"val {fs.val_start}→{fs.val_end}  "
            f"test {fs.test_start}→{fs.test_end}"
        )
        _, qt, imp = train_fold(
            df, feature_cols, args.target,
            fs.name,
            train_start=split.train_start,
            train_end=fs.train_end,
            val_start=fs.val_start, val_end=fs.val_end,
            test_start=fs.test_start, test_end=fs.test_end,
            params=params,
        )
        all_test.append(qt)
        all_imp.append(imp)

    preds = pd.concat(all_test, ignore_index=True)
    preds = preds.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Keep useful columns
    label_family = [
        "hold_3_close", "hold_5_close", "hold_3_struct", "hold_5_struct",
        "failed_break_3_close", "failed_break_5_close",
        "failed_break_3_struct", "failed_break_5_struct",
        "mfe_10_atr", "mae_10_atr", "spike_rejected_10",
        "mfe_15_atr", "mae_15_atr", "spike_rejected_15",
        "mfe_20_atr", "mae_20_atr", "spike_rejected_20",
        "expansion_score_10", "expansion_score_15", "expansion_score_20",
        "runner_10", "runner_15", "runner_20",
        "breakout_level_struct", "breakout_level_close", "atr_ref",
    ]
    keep = (
        ["ticker", "date", "fold_assigned", "setup_type",
         "trigger_hb", "trigger_fc", "close"]
        + [c for c in label_family if c in preds.columns]
        + ["score_model", "score_random", "score_compression"]
        + [c for c in feature_cols if c in preds.columns]
    )
    keep = [c for c in dict.fromkeys(keep) if c in preds.columns]
    preds = preds[keep]

    Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.out_preds)
    print(f"\n[WRITE] {args.out_preds}  shape={preds.shape}")

    imp_df = pd.concat(all_imp, axis=1)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean", ascending=False)
    imp_df.to_csv(args.out_imp)
    print(f"[WRITE] {args.out_imp}")
    print("\n[TOP-15 FEATURE IMPORTANCE (mean gain across folds)]")
    for f, r in imp_df.head(15).iterrows():
        print(
            f"  {f:<32}  "
            + "  ".join(f"{fs.name}={r[fs.name]:>7.0f}" for fs in split.folds)
            + f"  mean={r['mean']:>7.0f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
