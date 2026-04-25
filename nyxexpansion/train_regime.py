"""
nyxexpansion v1 — regime-conditional LightGBM (Aşama 1).

İki ayrı model per fold:
  - model_up:    xu_regime == 'uptrend' sample'ları
  - model_nonup: xu_regime in {'range', 'downtrend'} (+ 'unknown' küçük bir azınlık
                                                     → nonup'a dahil)

Inference: signal_date'in o günkü xu_regime'i → hangi modele route edileceğini
belirler. xu_regime aynı-seans XU100 EMA21/55 → leakage-safe.

Çıktı: output/nyxexp_train_regime_v1.parquet — aynı format, `model_kind`
kolonu eklemli ('up' / 'nonup'). `score_model` router çıktısıdır.
Baseline'lar (random, rank) aynı kalır.

Split DEĞİŞMEZ — SplitParams donmuş.

Kullanım:
    python -m nyxexpansion.train_regime
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

import lightgbm as lgb

from nyxexpansion.config import CONFIG
from nyxexpansion.features import (
    CORE_FEATURES, CORE_FEATURES_UP, CORE_FEATURES_NONUP,
)
from nyxexpansion.train import LGBMParams, build_rank_baseline


UP_REGIMES = {'uptrend'}
NONUP_REGIMES = {'range', 'downtrend', 'unknown'}


def _regime_mask(s: pd.Series, kind: str) -> pd.Series:
    """kind = 'up' veya 'nonup'."""
    r = s.fillna('unknown')
    if kind == 'up':
        return r.isin(UP_REGIMES)
    return r.isin(NONUP_REGIMES)


def _fit_one(
    tr: pd.DataFrame, va: pd.DataFrame, qt: pd.DataFrame,
    feature_cols: list[str], target_col: str, params: LGBMParams,
    tag: str,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int] | None:
    """Tek bir regime modeli. Val veya train çok küçükse None döner."""
    if len(tr) < 200 or len(va) < 50:
        print(f"      [{tag}] yetersiz örnek: tr={len(tr)}, va={len(va)} → atlandı")
        return None

    X_tr = tr[feature_cols]; y_tr = tr[target_col].astype(int)
    X_va = va[feature_cols]; y_va = va[target_col].astype(int)
    X_qt = qt[feature_cols]

    print(f"      [{tag}] tr={len(tr):,} (pos {y_tr.mean()*100:.1f}%)  "
          f"va={len(va):,} (pos {y_va.mean()*100:.1f}%)  "
          f"qt={len(qt):,} (pos {qt[target_col].mean()*100:.1f}%)")

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
    print(f"      [{tag}] best_iter={best_iter}")
    scores = clf.predict_proba(X_qt, num_iteration=best_iter)[:, 1] if len(qt) else np.array([])
    return clf, scores, best_iter


def train_fold_regime(
    data: pd.DataFrame, fold_name: str,
    train_start: str, train_end: str, val_start: str, val_end: str,
    test_start: str, test_end: str,
    target_col: str, params: LGBMParams,
    up_features: list[str] | None = None,
    nonup_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Asymmetric per-regime feature set desteği.

    up_features: UP regime modelinde kullanılacak feature listesi.
                 None → CORE_FEATURES_UP (34 = 33 core + chase_score_soft)
    nonup_features: NONUP regime modelinde kullanılacak feature listesi.
                    None → CORE_FEATURES_NONUP (26, J block yok)
    """
    if up_features is None:
        up_features = CORE_FEATURES_UP
    if nonup_features is None:
        nonup_features = CORE_FEATURES_NONUP

    d = pd.to_datetime(data['date'])
    tr = data[(d >= pd.Timestamp(train_start)) & (d <= pd.Timestamp(train_end))
              & data[target_col].notna()].copy()
    va = data[(d >= pd.Timestamp(val_start)) & (d <= pd.Timestamp(val_end))
              & data[target_col].notna()].copy()
    qt = data[(d >= pd.Timestamp(test_start)) & (d <= pd.Timestamp(test_end))
              & data[target_col].notna()].copy()

    imp_all: dict[str, pd.Series] = {}
    # Per-regime fit
    results: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    regime_features = {'up': up_features, 'nonup': nonup_features}
    for kind in ['up', 'nonup']:
        feats_k = regime_features[kind]
        tr_k = tr[_regime_mask(tr['xu_regime'], kind)]
        va_k = va[_regime_mask(va['xu_regime'], kind)]
        qt_k = qt[_regime_mask(qt['xu_regime'], kind)].copy()
        res = _fit_one(tr_k, va_k, qt_k, feats_k, target_col, params,
                       tag=f"{fold_name}/{kind}")
        if res is None:
            # Fallback: global model skoru kullanılmayacak, scores NaN
            qt_k['score_model'] = np.nan
            qt_k['model_kind'] = kind
            results[kind] = (qt_k, np.full(len(qt_k), np.nan))
            continue
        clf, scores, best_iter = res
        qt_k['score_model'] = scores
        qt_k['model_kind'] = kind
        results[kind] = (qt_k, scores)
        # importance
        imp_all[kind] = pd.Series(
            clf.booster_.feature_importance(importance_type='gain'),
            index=feats_k, name=f"{fold_name}_{kind}",
        )

    # Router merge — qt_k'ları concat
    merged_test = pd.concat([results['up'][0], results['nonup'][0]], ignore_index=False)
    merged_test = merged_test.sort_values(['date', 'ticker']).reset_index(drop=True)
    merged_test['fold_assigned'] = fold_name

    # Baseline'lar — aynı test (tümü) setinde
    rng = np.random.default_rng(params.seed + hash(fold_name) % 1000)
    merged_test['score_random'] = rng.uniform(0, 1, size=len(merged_test))
    merged_test['score_rank'] = build_rank_baseline(merged_test).values

    return merged_test, imp_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default="output/nyxexp_dataset_v3.parquet")
    ap.add_argument('--target', default=f'cont_{CONFIG.label.primary_h}')
    ap.add_argument('--out-preds', default="output/nyxexp_train_regime_v3.parquet")
    ap.add_argument('--out-imp', default="output/nyxexp_importance_regime_v3.csv")
    args = ap.parse_args()

    os.chdir(str(_ROOT))
    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}")
        return 2

    up_feats = CORE_FEATURES_UP
    nonup_feats = CORE_FEATURES_NONUP
    print(f"═══ nyxexpansion Train REGIME v3 (asymmetric) ═══")
    print(f"  UP features    = {len(up_feats)}  (= CORE_V2 {len(CORE_FEATURES)} + chase_score_soft)")
    print(f"  NONUP features = {len(nonup_feats)}  (= CORE_V1, no J block)")

    df = pd.read_parquet(args.dataset)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  shape={df.shape}  target={args.target}")

    needed = sorted(set(up_feats) | set(nonup_feats))
    missing = [f for f in needed if f not in df.columns]
    if missing:
        print(f"❌ eksik feature: {missing}")
        return 3

    params = LGBMParams()
    split = CONFIG.split

    all_preds: list[pd.DataFrame] = []
    all_imp: list[pd.Series] = []
    for fs in split.folds:
        print(f"\n[FOLD {fs.name}]  "
              f"tr→{fs.train_end} · val {fs.val_start}→{fs.val_end} · "
              f"test {fs.test_start}→{fs.test_end}")
        preds, imp_dict = train_fold_regime(
            df, fs.name,
            train_start=split.train_start, train_end=fs.train_end,
            val_start=fs.val_start, val_end=fs.val_end,
            test_start=fs.test_start, test_end=fs.test_end,
            target_col=args.target, params=params,
            up_features=up_feats, nonup_features=nonup_feats,
        )
        all_preds.append(preds)
        for kind, imp in imp_dict.items():
            all_imp.append(imp)

    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.sort_values(['date', 'ticker']).reset_index(drop=True)

    union_feats = sorted(set(up_feats) | set(nonup_feats))
    keep = (['ticker', 'date', 'fold_assigned', 'model_kind', 'xu_regime',
             args.target, f'cont_{CONFIG.label.primary_h}_struct',
             'mfe_mae_ratio_win',
             f'mfe_{CONFIG.label.primary_h}', f'mae_{CONFIG.label.primary_h}',
             f'expansion_score_{CONFIG.label.primary_h}',
             'score_model', 'score_random', 'score_rank']
            + [f for f in union_feats if f in preds.columns])
    keep = list(dict.fromkeys(keep))
    preds = preds[keep]
    Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.out_preds)
    print(f"\n[WRITE] {args.out_preds}  shape={preds.shape}")

    imp_df = pd.concat(all_imp, axis=1) if all_imp else pd.DataFrame()
    if not imp_df.empty:
        # Split up/nonup columns
        up_cols = [c for c in imp_df.columns if c.endswith('_up')]
        non_cols = [c for c in imp_df.columns if c.endswith('_nonup')]
        imp_df['up_mean'] = imp_df[up_cols].mean(axis=1) if up_cols else np.nan
        imp_df['nonup_mean'] = imp_df[non_cols].mean(axis=1) if non_cols else np.nan
        imp_df = imp_df.sort_values('up_mean', ascending=False)
        imp_df.to_csv(args.out_imp)
        print(f"[WRITE] {args.out_imp}")
        print(f"\n[TOP-10 gain — UP model]")
        for f, r in imp_df.head(10).iterrows():
            up_str = "  ".join(f"{c.split('_',1)[0]}={r[c]:>6.0f}" for c in up_cols)
            print(f"  {f:<32}  {up_str}  μ={r['up_mean']:>6.0f}")
        imp_df_n = imp_df.sort_values('nonup_mean', ascending=False)
        print(f"\n[TOP-10 gain — NONUP model]")
        for f, r in imp_df_n.head(10).iterrows():
            n_str = "  ".join(f"{c.split('_',1)[0]}={r[c]:>6.0f}" for c in non_cols)
            print(f"  {f:<32}  {n_str}  μ={r['nonup_mean']:>6.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
