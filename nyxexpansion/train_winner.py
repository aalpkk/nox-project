"""
nyxexpansion v1 — winner size regressor (two-head mimari Aşama 2).

Amaç:
  Primary target L3 (continuation_10) KALIR.
  Bu ek regressor L3=1 alt-kümesinde, "winner ne kadar büyük olur" öğrenir.
  Inference sırasında TÜM adaylar için winner_R_pred üretir.

Hedef: winner_R = mfe_10 * close_0 / atr_14  (R-multiple)
  - L3 base rate ~32% → subset küçük ama yeterli (~5000+)
  - Fold-aware winsorize @p99 → outlier dominance'ı kırar
  - LightGBM regression_l1 (MAE loss) → outlier robust

Regime split: classifier ile aynı — UP modeli vs NONUP modeli.
Feature sets: classifier ile aynı — CORE_FEATURES_UP vs CORE_FEATURES_NONUP.
Split/folds: classifier ile aynı — CONFIG.split.

Leakage:
  - winner_R label window t+1..t+10 (L3 ile aynı), model OOS test'i görmez
  - xu_regime aynı-seans BIST, trigger A leakage-safe
  - Winsorize sadece TRAIN+VAL dağılımından, test/qt'ye uygulanmaz

Çıktı: `output/nyxexp_train_winner_v3.parquet`
  kolon: `winner_R_pred`, `winner_R_true` (eval için).

Kullanım:
    python -m nyxexpansion.train_winner
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
from nyxexpansion.features import CORE_FEATURES_UP, CORE_FEATURES_NONUP
from nyxexpansion.train import LGBMParams

UP_REGIMES = {'uptrend'}
NONUP_REGIMES = {'range', 'downtrend', 'unknown'}


def _regime_mask(s: pd.Series, kind: str) -> pd.Series:
    r = s.fillna('unknown')
    return r.isin(UP_REGIMES) if kind == 'up' else r.isin(NONUP_REGIMES)


def _add_winner_R(df: pd.DataFrame, h: int = 10) -> pd.DataFrame:
    """winner_R = mfe_h * close_0 / atr_14 (R-multiple).

    atr_14/close_0 NaN veya ≤0 ise winner_R = NaN (train/eval'den dışlanır).
    """
    df = df.copy()
    mfe_col = f'mfe_{h}'
    denom = df['atr_14']
    invalid = (denom <= 0) | denom.isna() | df[mfe_col].isna() | df['close_0'].isna()
    df['winner_R'] = np.where(
        invalid, np.nan,
        df[mfe_col] * df['close_0'] / denom.replace(0, np.nan),
    )
    return df


def _fit_regressor(
    tr: pd.DataFrame, va: pd.DataFrame, qt: pd.DataFrame,
    feature_cols: list[str], target_col: str, params: LGBMParams,
    tag: str,
) -> tuple[lgb.LGBMRegressor, np.ndarray, int] | None:
    """Regressor fit + predict all qt (not just L3=1)."""
    if len(tr) < 100 or len(va) < 30:
        print(f"      [{tag}] yetersiz örnek: tr={len(tr)}, va={len(va)} → atlandı")
        return None

    X_tr = tr[feature_cols]; y_tr = tr[target_col].astype(float)
    X_va = va[feature_cols]; y_va = va[target_col].astype(float)
    X_qt = qt[feature_cols]

    print(f"      [{tag}] tr={len(tr):,} (μ={y_tr.mean():.2f} med={y_tr.median():.2f})  "
          f"va={len(va):,} (μ={y_va.mean():.2f})  qt_full={len(qt):,}")

    reg = lgb.LGBMRegressor(
        objective='regression_l1',  # MAE — outlier robust
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
    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='l1',
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = reg.best_iteration_
    print(f"      [{tag}] best_iter={best_iter}")
    preds = reg.predict(X_qt, num_iteration=best_iter) if len(qt) else np.array([])
    return reg, preds, best_iter


def train_fold_winner(
    data: pd.DataFrame, fold_name: str,
    train_start: str, train_end: str, val_start: str, val_end: str,
    test_start: str, test_end: str,
    l3_col: str = 'cont_10', target_col: str = 'winner_R',
    params: LGBMParams | None = None,
    up_features: list[str] | None = None,
    nonup_features: list[str] | None = None,
    winsorize_q: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Two-head Aşama 2 — L3=1 subset'te winner_R regressor.

    Test set (qt): TÜM adaylar (L3 label olmadan da); tahmin tüm adaylara yapılır.
    """
    if up_features is None:
        up_features = CORE_FEATURES_UP
    if nonup_features is None:
        nonup_features = CORE_FEATURES_NONUP
    if params is None:
        params = LGBMParams()

    d = pd.to_datetime(data['date'])
    # Train/val: yalnız L3=1 VE target notna
    l3_mask = data[l3_col] == 1
    tr_base = (d >= pd.Timestamp(train_start)) & (d <= pd.Timestamp(train_end))
    va_base = (d >= pd.Timestamp(val_start)) & (d <= pd.Timestamp(val_end))
    qt_base = (d >= pd.Timestamp(test_start)) & (d <= pd.Timestamp(test_end))
    target_ok = data[target_col].notna() & np.isfinite(data[target_col])

    tr = data[tr_base & l3_mask & target_ok].copy()
    va = data[va_base & l3_mask & target_ok].copy()
    # Test: TÜM adaylar (L3 filtresi yok — tahmin her aday için gerekli)
    qt = data[qt_base].copy()

    # Winsorize TRAIN+VAL dağılımından (global p99 — her rejim için ayrı)
    # qt'ye winsorize UYGULAMA — bunu biz tahmin ediyoruz
    imp_all: dict[str, pd.Series] = {}
    results: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    regime_features = {'up': up_features, 'nonup': nonup_features}
    for kind in ['up', 'nonup']:
        feats_k = regime_features[kind]
        tr_k = tr[_regime_mask(tr['xu_regime'], kind)].copy()
        va_k = va[_regime_mask(va['xu_regime'], kind)].copy()
        qt_k = qt[_regime_mask(qt['xu_regime'], kind)].copy()

        # Winsorize training target @p99 (train+val birleştirerek)
        comb = pd.concat([tr_k[target_col], va_k[target_col]])
        if len(comb) < 50:
            print(f"      [{fold_name}/{kind}] yetersiz L3=1: tr+va={len(comb)} → skip")
            qt_k['winner_R_pred'] = np.nan
            qt_k['model_kind'] = kind
            results[kind] = (qt_k, np.full(len(qt_k), np.nan))
            continue
        hi = comb.quantile(winsorize_q)
        lo = comb.quantile(1 - winsorize_q)  # çift taraflı — kısa negatif tail koru
        tr_k[target_col] = tr_k[target_col].clip(lower=lo, upper=hi)
        va_k[target_col] = va_k[target_col].clip(lower=lo, upper=hi)
        print(f"      [{fold_name}/{kind}] winsorize @[{lo:.2f}, {hi:.2f}]  "
              f"L3=1 tr={len(tr_k)} va={len(va_k)}")

        res = _fit_regressor(tr_k, va_k, qt_k, feats_k, target_col, params,
                             tag=f"{fold_name}/{kind}")
        if res is None:
            qt_k['winner_R_pred'] = np.nan
            qt_k['model_kind'] = kind
            results[kind] = (qt_k, np.full(len(qt_k), np.nan))
            continue
        reg, preds, best_iter = res
        qt_k['winner_R_pred'] = preds
        qt_k['model_kind'] = kind
        results[kind] = (qt_k, preds)
        imp_all[kind] = pd.Series(
            reg.booster_.feature_importance(importance_type='gain'),
            index=feats_k, name=f"{fold_name}_{kind}",
        )

    merged = pd.concat([results['up'][0], results['nonup'][0]], ignore_index=False)
    merged = merged.sort_values(['date', 'ticker']).reset_index(drop=True)
    merged['fold_assigned'] = fold_name
    return merged, imp_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default="output/nyxexp_dataset_v3.parquet")
    ap.add_argument('--out-preds', default="output/nyxexp_train_winner_v3.parquet")
    ap.add_argument('--out-imp', default="output/nyxexp_importance_winner_v3.csv")
    ap.add_argument('--winsorize-q', type=float, default=0.99)
    args = ap.parse_args()

    os.chdir(str(_ROOT))
    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}")
        return 2

    print(f"═══ nyxexpansion Train WINNER v3 (two-head Aşama 2) ═══")
    print(f"[LEAKAGE]")
    print(f"  winner_R = mfe_10 * close_0 / atr_14 (t..t+10 label, OOS test dokunulmaz)")
    print(f"  Winsorize p{args.winsorize_q*100:.0f} sadece TRAIN+VAL'den; qt'ye uygulanmaz")
    print(f"  L3=1 filtresi TRAIN/VAL'de; qt=tüm adaylar (tahmin her biri için)")

    df = pd.read_parquet(args.dataset)
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n  shape={df.shape}")

    # winner_R target
    df = _add_winner_R(df, h=CONFIG.label.primary_h)
    l3_col = f'cont_{CONFIG.label.primary_h}'
    l3_pos = df[l3_col].sum()
    w_ok = df['winner_R'].notna().sum()
    print(f"  L3=1: {int(l3_pos):,} / {len(df):,}  ({l3_pos/len(df)*100:.1f}%)")
    print(f"  winner_R valid: {w_ok:,}")

    up_feats = CORE_FEATURES_UP
    nonup_feats = CORE_FEATURES_NONUP
    needed = sorted(set(up_feats) | set(nonup_feats))
    missing = [f for f in needed if f not in df.columns]
    if missing:
        print(f"❌ eksik feature: {missing}")
        return 3

    params = LGBMParams()
    split = CONFIG.split
    all_preds = []
    all_imp = []
    for fs in split.folds:
        print(f"\n[FOLD {fs.name}]  "
              f"tr→{fs.train_end} · val {fs.val_start}→{fs.val_end} · "
              f"test {fs.test_start}→{fs.test_end}")
        preds, imp_dict = train_fold_winner(
            df, fs.name,
            train_start=split.train_start, train_end=fs.train_end,
            val_start=fs.val_start, val_end=fs.val_end,
            test_start=fs.test_start, test_end=fs.test_end,
            l3_col=l3_col, target_col='winner_R',
            params=params,
            up_features=up_feats, nonup_features=nonup_feats,
            winsorize_q=args.winsorize_q,
        )
        all_preds.append(preds)
        for kind, imp in imp_dict.items():
            all_imp.append(imp)

    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.sort_values(['date', 'ticker']).reset_index(drop=True)

    h = CONFIG.label.primary_h
    keep = ['ticker', 'date', 'fold_assigned', 'model_kind', 'xu_regime',
            l3_col, f'cont_{h}_struct', f'mfe_{h}', f'mae_{h}',
            f'expansion_score_{h}', 'winner_R', 'winner_R_pred']
    keep = [c for c in keep if c in preds.columns]
    preds = preds[keep]
    Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.out_preds)
    print(f"\n[WRITE] {args.out_preds}  shape={preds.shape}")

    imp_df = pd.concat(all_imp, axis=1) if all_imp else pd.DataFrame()
    if not imp_df.empty:
        up_cols = [c for c in imp_df.columns if c.endswith('_up')]
        non_cols = [c for c in imp_df.columns if c.endswith('_nonup')]
        imp_df['up_mean'] = imp_df[up_cols].mean(axis=1) if up_cols else np.nan
        imp_df['nonup_mean'] = imp_df[non_cols].mean(axis=1) if non_cols else np.nan
        imp_df = imp_df.sort_values('up_mean', ascending=False)
        imp_df.to_csv(args.out_imp)
        print(f"[WRITE] {args.out_imp}")
        print(f"\n[TOP-10 gain — UP winner model]")
        for f, r in imp_df.head(10).iterrows():
            print(f"  {f:<32}  μ_up={r['up_mean']:>6.0f}  μ_nonup={r['nonup_mean']:>6.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
