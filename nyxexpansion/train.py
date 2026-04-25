"""
nyxexpansion v1 — LightGBM train (walk-forward).

Primary target: L3 cont_10 (binary).
3 fold expanding walk-forward — SplitParams donmuş, hyperparam tuning split'i
değiştirmez.

Per fold:
  - Train = [train_start .. fold.train_end]
  - Val   = [fold.val_start .. fold.val_end]        → early stopping
  - Test  = [fold.test_start .. fold.test_end]      → OOS report

Reference baselines (aynı fold pencereleri):
  - random: uniform [0,1) skor
  - rank_baseline: (breadth_ad_20d + rs_rank_cs_today + rvol z-score) normalize

Output:
  output/nyxexp_train_v1.parquet
    kolonlar: ticker, date, fold, split, cont_10, cont_10_struct,
              score_model, score_random, score_rank, + key panel fields.
  output/nyxexp_importance_v1.csv
    feature importance per fold (gain).

Kullanım:
  python -m nyxexpansion.train
  python -m nyxexpansion.train --features core26   (default)
  python -m nyxexpansion.train --features extended48  (ileride)
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

from nyxexpansion.config import CONFIG
from nyxexpansion.features import CORE_FEATURES


@dataclass(frozen=True)
class LGBMParams:
    """Modest baseline — v1. Tuning val'de yapılır, test'e dokunulmaz."""
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_child_samples: int = 40
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 100
    seed: int = 17


def _normalize_series(s: pd.Series) -> pd.Series:
    """Per-date z-score → probit-ish [0,1] via min-max rescale of rank."""
    return s.rank(method='average', pct=True, na_option='keep')


def build_rank_baseline(df: pd.DataFrame) -> pd.Series:
    """Basit rank baseline: date-bazında cross-sectional rank ortalaması.

    Fac: rs_rank_cs_today (zaten rank) + rvol_today + breadth_ad_20d
         (breadth per-date sabit, yine de rescale yapılabilir).
    Eksik değer olursa o hisse cross-sectional'a girmez → NaN.
    """
    parts = []
    for f in ['rs_rank_cs_today', 'rvol_today', 'breadth_ad_20d']:
        if f not in df.columns:
            continue
        # per-date rank
        r = df.groupby('date')[f].rank(method='average', pct=True, na_option='keep')
        parts.append(r)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1, skipna=True)


def train_fold(
    data: pd.DataFrame,
    feature_cols: list[str],
    fold_name: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    train_start: str,
    target_col: str,
    params: LGBMParams,
) -> tuple[lgb.Booster, pd.DataFrame, pd.Series]:
    """Tek fold train+val+test. Döndürür:
       booster, enriched test_rows, feature_importance (Series)."""
    d = pd.to_datetime(data['date'])
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

    print(f"  [{fold_name}] train N={len(tr):,}  val N={len(va):,}  "
          f"test N={len(qt):,}  (target pos tr/va/qt: "
          f"{y_tr.mean()*100:.1f}% / {y_va.mean()*100:.1f}% / "
          f"{qt[target_col].mean()*100:.1f}%)")

    clf = lgb.LGBMClassifier(
        objective='binary',
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
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = clf.best_iteration_
    print(f"      best_iter={best_iter}")

    # Predict test
    scores = clf.predict_proba(X_qt, num_iteration=best_iter)[:, 1]
    qt['score_model'] = scores
    qt['fold_assigned'] = fold_name
    # Baselines — aynı test setinde
    rng = np.random.default_rng(params.seed + hash(fold_name) % 1000)
    qt['score_random'] = rng.uniform(0, 1, size=len(qt))
    qt['score_rank'] = build_rank_baseline(qt).values

    # Feature importance (gain)
    imp = pd.Series(
        clf.booster_.feature_importance(importance_type='gain'),
        index=feature_cols, name=fold_name,
    )

    return clf.booster_, qt, imp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default="output/nyxexp_dataset_v1.parquet")
    ap.add_argument('--features', default='core26',
                    choices=['core26'])  # extended48 ileride
    ap.add_argument('--target', default=f'cont_{CONFIG.label.primary_h}',
                    help='Primary target — binary label')
    ap.add_argument('--out-preds', default="output/nyxexp_train_v1.parquet")
    ap.add_argument('--out-imp', default="output/nyxexp_importance_v1.csv")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}  →  python -m nyxexpansion.dataset")
        return 2

    print(f"═══ nyxexpansion Train v1 ═══")
    print(f"Dataset: {args.dataset}")

    df = pd.read_parquet(args.dataset)
    print(f"  shape: {df.shape}")
    df['date'] = pd.to_datetime(df['date'])

    feature_cols = CORE_FEATURES
    # Feature sanity
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"❌ Feature eksik: {missing}")
        return 3
    print(f"  feature count: {len(feature_cols)}  ({args.features})")
    print(f"  target: {args.target}")

    params = LGBMParams()
    split = CONFIG.split

    # Train per fold
    all_test = []
    all_imp: list[pd.Series] = []
    for fs in split.folds:
        print(f"\n[FOLD {fs.name}] train→{fs.train_end}  val {fs.val_start}→{fs.val_end}  "
              f"test {fs.test_start}→{fs.test_end}")
        booster, qt, imp = train_fold(
            df, feature_cols, fs.name,
            train_end=fs.train_end,
            val_start=fs.val_start, val_end=fs.val_end,
            test_start=fs.test_start, test_end=fs.test_end,
            train_start=split.train_start,
            target_col=args.target,
            params=params,
        )
        all_test.append(qt)
        all_imp.append(imp)

    # Combine
    preds = pd.concat(all_test, ignore_index=True)
    preds = preds.sort_values(['date', 'ticker']).reset_index(drop=True)

    # Keep useful columns
    keep = (['ticker', 'date', 'fold_assigned',
             args.target, f'cont_{CONFIG.label.primary_h}_struct',
             'mfe_mae_ratio_win', f'mfe_{CONFIG.label.primary_h}',
             f'mae_{CONFIG.label.primary_h}', f'expansion_score_{CONFIG.label.primary_h}',
             'score_model', 'score_random', 'score_rank']
            + [f for f in feature_cols if f in preds.columns]
            + (['xu_regime'] if 'xu_regime' in preds.columns else []))
    keep = list(dict.fromkeys(keep))
    preds = preds[keep]

    Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.out_preds)
    print(f"\n[WRITE] {args.out_preds}  shape={preds.shape}")

    imp_df = pd.concat(all_imp, axis=1)
    imp_df['mean'] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values('mean', ascending=False)
    imp_df.to_csv(args.out_imp)
    print(f"[WRITE] {args.out_imp}")
    print(f"\n[TOP-10 FEATURE IMPORTANCE (mean gain across folds)]")
    for f, r in imp_df.head(10).iterrows():
        print(f"  {f:<32}  "
              + "  ".join(f"{fs.name}={r[fs.name]:>7.0f}" for fs in split.folds)
              + f"  mean={r['mean']:>7.0f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
