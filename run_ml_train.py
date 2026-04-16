#!/usr/bin/env python3
"""
ML Model Training + Evaluation
Kullanım:
    python run_ml_train.py                              # Default: 4 target
    python run_ml_train.py --target up_1g               # Tek target
    python run_ml_train.py --target all                  # 4 target (default)
    python run_ml_train.py --feature-group primitives    # Ablation: sadece primitives
    python run_ml_train.py --tune                       # Optuna hyperparameter tuning
    python run_ml_train.py --dataset output/ml_dataset.parquet  # Farklı dataset
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from ml.dataset import get_feature_columns, time_split, walk_forward_split, dataset_summary
from ml.evaluate import (
    full_evaluation, feature_importance_report, shap_importance,
    classification_metrics, regression_metrics, trading_metrics,
)

# ═══════════════════════════════════════════
# FEATURE GROUP TANIMLARI (Ablation)
# ═══════════════════════════════════════════

FEATURE_GROUPS = {
    'all': None,  # Tüm feature'lar
    'primitives': {
        'include_prefixes': [
            'returns_', 'close_position', 'gap_pct', 'ema21_dist', 'ema55_dist',
            'adx_', 'plus_di', 'minus_di', 'di_spread', 'ema_trend', 'supertrend',
            'pmax_dir', 'phase_above', 'htf_',
            'rsi_', 'macd_', 'wt1', 'wt2', 'wt_bullish', 'squeeze_',
            'atr_', 'bb_', 'drawdown_', 'daily_move_atr',
            'vol_ratio_', 'cmf_', 'obv_', 'rvol', 'mfi_',
            'swing_bias', 'bos_age', 'choch_age', 'structure_break', 'higher_low', 'near_40high',
        ],
        'description': 'Ham teknik primitives (screener türevleri hariç)',
    },
    'primitives_macro': {
        'include_prefixes': [
            'returns_', 'close_position', 'gap_pct', 'ema21_dist', 'ema55_dist',
            'adx_', 'plus_di', 'minus_di', 'di_spread', 'ema_trend', 'supertrend',
            'pmax_dir', 'phase_above', 'htf_',
            'rsi_', 'macd_', 'wt1', 'wt2', 'wt_bullish', 'squeeze_',
            'atr_', 'bb_', 'drawdown_', 'daily_move_atr',
            'vol_ratio_', 'cmf_', 'obv_', 'rvol', 'mfi_',
            'swing_bias', 'bos_age', 'choch_age', 'structure_break', 'higher_low', 'near_40high',
            'vix', 'dxy_', 'usdtry_', 'xu100_', 'spy_', 'macro_risk',
        ],
        'description': 'Primitives + makro kontekst',
    },
    'screener_derived': {
        'include_prefixes': [
            'q_', 'rs_', 'br_', 'rg_', 'regime_score', 'trend_score', 'entry_score',
            'oe_score', 'exit_stage', 'days_in_trade', 'gate_open', 'sell_severity',
            'rt_', 'pb_', 'is_tavan', 'tavan_', 'hit_tavan', 'close_to_high',
            'recent_tavan', 'vol_change_vs_prev', 'consecutive_green', 'pivot_delta',
        ],
        'description': 'Screener türev skorları (composites)',
    },
    'no_macro': {
        'exclude_prefixes': [
            'vix', 'dxy_', 'usdtry_', 'xu100_', 'spy_', 'macro_risk',
        ],
        'description': 'Tüm feature\'lar - makro',
    },
    'no_meta_scores': {
        'exclude_prefixes': [
            'regime_score', 'trend_score', 'entry_score', 'oe_score', 'exit_stage',
            'days_in_trade', 'q_total', 'br_score', 'rg_score', 'sell_severity',
        ],
        'description': 'Tüm feature\'lar - composite meta skorlar',
    },
}


def filter_features_by_group(all_feat_cols, group_name):
    """Feature group'a göre kolon filtrele."""
    if group_name == 'all' or group_name not in FEATURE_GROUPS:
        return all_feat_cols

    group = FEATURE_GROUPS[group_name]
    if group is None:
        return all_feat_cols

    if 'include_prefixes' in group and 'exclude_prefixes' not in group:
        prefixes = tuple(group['include_prefixes'])
        return [c for c in all_feat_cols if c.startswith(prefixes)]
    elif 'exclude_prefixes' in group:
        prefixes = tuple(group['exclude_prefixes'])
        return [c for c in all_feat_cols if not c.startswith(prefixes)]
    return all_feat_cols


# ═══════════════════════════════════════════
# MODEL EĞİTİMİ
# ═══════════════════════════════════════════

def train_lgbm_classifier(X_train, y_train, X_val, y_val, params=None):
    """LightGBM Classifier eğit."""
    import lightgbm as lgb

    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_lgbm_regressor(X_train, y_train, X_val, y_val, params=None):
    """LightGBM Regressor eğit."""
    import lightgbm as lgb

    default_params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_logistic_baseline(X_train, y_train):
    """Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_mlp_classifier(X_train, y_train, X_val, y_val):
    """MLP Classifier eğit (sklearn — BIST-scale veri için uygun küçük ağ)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,           # L2 regularization (BIST'te overfit riski yüksek)
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_mlp_regressor(X_train, y_train, X_val, y_val):
    """MLP Regressor eğit."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ═══════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════

def tune_lgbm(X_train, y_train, X_val, y_val, target_type='classification', n_trials=50):
    """Optuna ile LightGBM hyperparameter tuning."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [!] optuna yüklü değil, default parametreler kullanılıyor.")
        return {}

    import lightgbm as lgb

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'verbose': -1,
            'n_jobs': -1,
        }

        if target_type == 'classification':
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            proba = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import average_precision_score
            return average_precision_score(y_val, proba)
        else:
            params['objective'] = 'regression'
            params['metric'] = 'mae'
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            pred = model.predict(X_val)
            corr = np.corrcoef(y_val, pred)[0, 1]
            return corr

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ═══════════════════════════════════════════

def walk_forward_evaluate(dataset, target_col, feat_cols, model_type='lgbm_clf'):
    """Walk-forward cross-validation."""
    results = []

    for fold, train_mask, val_mask in walk_forward_split(dataset):
        train = dataset[train_mask]
        val = dataset[val_mask]

        X_train = train[feat_cols].values
        y_train = train[target_col].dropna().reindex(train.index).values
        X_val = val[feat_cols].values
        y_val = val[target_col].dropna().reindex(val.index).values

        # Remove NaN targets
        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_val, y_val = X_val[val_valid], y_val[val_valid]

        if len(X_train) < 1000 or len(X_val) < 100:
            continue

        dates = dataset.index.get_level_values('date')
        train_dates = dates[train_mask]
        val_dates = dates[val_mask]

        print(f"\n  Fold {fold}: train {train_dates.min().strftime('%Y-%m')} → "
              f"{train_dates.max().strftime('%Y-%m')} ({len(X_train):,}), "
              f"val {val_dates.min().strftime('%Y-%m')} → "
              f"{val_dates.max().strftime('%Y-%m')} ({len(X_val):,})")

        if model_type == 'lgbm_clf':
            model = train_lgbm_classifier(X_train, y_train, X_val, y_val)
            proba = model.predict_proba(X_val)[:, 1]
            cm = classification_metrics(y_val, proba)
            # Get return for trading metrics
            ret_col = 'ret_1g' if '1g' in target_col else 'ret_3g'
            val_ret = val[ret_col].values[val_valid] if ret_col in val.columns else None
            tm = trading_metrics(val_ret, proba) if val_ret is not None else {}
            results.append({
                'fold': fold, 'type': 'lgbm_clf', **cm,
                'top_wr': tm.get('top_decile_wr', 0),
                'spread': tm.get('spread_wr', 0),
            })
            print(f"    PR-AUC: {cm['pr_auc']:.3f} | F1: {cm['f1']:.3f} | "
                  f"Top WR: {tm.get('top_decile_wr', 0):.1%}")

        elif model_type == 'lgbm_reg':
            model = train_lgbm_regressor(X_train, y_train, X_val, y_val)
            pred = model.predict(X_val)
            rm = regression_metrics(y_val, pred)
            # Predicted return'ü ranking score olarak trading metrics'e ver
            tm = trading_metrics(y_val, pred) if len(y_val) >= 100 else {}
            results.append({
                'fold': fold, 'type': 'lgbm_reg', **rm,
                'top_wr': tm.get('top_10pct_wr', 0),
                'spread': tm.get('spread_wr', 0),
                'spearman': tm.get('spearman_corr', 0),
            })
            print(f"    MAE: {rm['mae']:.3f} | Corr: {rm['corr']:.3f} | "
                  f"Top10%WR: {tm.get('top_10pct_wr', 0):.1%}")

        elif model_type == 'lr':
            pipe = train_logistic_baseline(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            cm = classification_metrics(y_val, proba)
            results.append({'fold': fold, 'type': 'lr', **cm})
            print(f"    PR-AUC: {cm['pr_auc']:.3f} | F1: {cm['f1']:.3f}")

    return results


# ═══════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ML Model Training + Evaluation")
    parser.add_argument("--dataset", default="output/ml_dataset.parquet",
                        help="Input parquet")
    parser.add_argument("--target", default="all",
                        help="Target: all, up_1g, up_3g, ret_1g, ret_3g")
    parser.add_argument("--feature-group", default="all",
                        choices=list(FEATURE_GROUPS.keys()),
                        help="Feature group (ablation): " +
                             ", ".join(f"{k}" for k in FEATURE_GROUPS.keys()))
    parser.add_argument("--tune", action="store_true",
                        help="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trial sayısı")
    parser.add_argument("--output-dir", default="output/ml_models",
                        help="Model ve rapor çıktı dizini")
    parser.add_argument("--shap", action="store_true",
                        help="SHAP analizi yap")
    parser.add_argument("--no-walkforward", action="store_true",
                        help="Walk-forward CV atla (hızlı test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("ML MODEL TRAINING")
    print("="*60)
    t0 = time.time()

    # 1. Dataset yükle
    print(f"\n  Dataset: {args.dataset}")
    dataset = pd.read_parquet(args.dataset)
    dataset_summary(dataset)

    # 2. Feature group filtresi (ablation)
    all_feat_cols = get_feature_columns(dataset)
    feat_cols = filter_features_by_group(all_feat_cols, args.feature_group)
    group_desc = FEATURE_GROUPS.get(args.feature_group, {})
    if isinstance(group_desc, dict) and 'description' in group_desc:
        print(f"\n  Feature Group: {args.feature_group} — {group_desc['description']}")
    print(f"  Feature sayısı: {len(feat_cols)} / {len(all_feat_cols)} toplam")

    # 3. Simple split
    train_mask, val_mask, test_mask = time_split(dataset)
    print(f"\n  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    train = dataset[train_mask]
    val = dataset[val_mask]
    test = dataset[test_mask]

    # 4. Target belirleme
    ALL_TARGETS = ['up_1g', 'ret_1g', 'up_3g', 'ret_3g']
    if args.target == 'all':
        targets_to_train = ALL_TARGETS
    else:
        targets_to_train = [args.target]

    all_results = {}

    for target_col in targets_to_train:
        is_clf = target_col.startswith('up_')
        ret_col = target_col.replace('up_', 'ret_')

        print(f"\n{'='*60}")
        print(f"  TARGET: {target_col} ({'Classification' if is_clf else 'Regression'})")
        print(f"{'='*60}")

        # Prepare data
        X_train = train[feat_cols].values
        y_train = train[target_col].values
        X_val = val[feat_cols].values
        y_val = val[target_col].values
        X_test = test[feat_cols].values
        y_test = test[target_col].values

        # Remove NaN targets
        tr_valid = ~np.isnan(y_train)
        va_valid = ~np.isnan(y_val)
        te_valid = ~np.isnan(y_test)

        X_tr, y_tr = X_train[tr_valid], y_train[tr_valid]
        X_va, y_va = X_val[va_valid], y_val[va_valid]
        X_te, y_te = X_test[te_valid], y_test[te_valid]

        print(f"  Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

        if is_clf:
            # ── LightGBM Classifier ──
            print(f"\n  ── LightGBM Classifier ──")
            lgbm_params = {}
            if args.tune:
                print("  Tuning...")
                lgbm_params = tune_lgbm(X_tr, y_tr, X_va, y_va,
                                        target_type='classification',
                                        n_trials=args.n_trials)
                print(f"  Best params: {lgbm_params}")

            model = train_lgbm_classifier(X_tr, y_tr, X_va, y_va, params=lgbm_params)

            # Validation evaluation
            va_proba = model.predict_proba(X_va)[:, 1]
            va_ret = val[ret_col].values[va_valid] if ret_col in val.columns else None
            va_regime = val['regime_score'].values[va_valid] if 'regime_score' in val.columns else None
            print("\n  --- Validation ---")
            va_results = full_evaluation(y_va, va_ret, va_proba,
                                         label=f"LightGBM {target_col} (Val)",
                                         regime_values=va_regime)

            # Test evaluation
            if len(X_te) > 100:
                te_proba = model.predict_proba(X_te)[:, 1]
                te_ret = test[ret_col].values[te_valid] if ret_col in test.columns else None
                te_regime = test['regime_score'].values[te_valid] if 'regime_score' in test.columns else None
                print("\n  --- Test ---")
                te_results = full_evaluation(y_te, te_ret, te_proba,
                                             label=f"LightGBM {target_col} (Test)",
                                             regime_values=te_regime)
            else:
                te_results = None

            # Feature importance
            print("\n  --- Feature Importance (Gain) ---")
            fi = feature_importance_report(model, feat_cols, 'gain', 20)
            print(fi[['rank', 'feature', 'importance', 'pct']].to_string(index=False))

            # SHAP
            if args.shap:
                print("\n  --- SHAP ---")
                shap_df = shap_importance(model, X_va, feat_cols)
                if not shap_df.empty:
                    print(shap_df.head(20).to_string(index=False))
                    shap_df.to_csv(f"{args.output_dir}/shap_{target_col}.csv", index=False)

            # Save model
            model.booster_.save_model(f"{args.output_dir}/lgbm_{target_col}.txt")
            fi.to_csv(f"{args.output_dir}/importance_{target_col}.csv", index=False)

            # ── MLP Ensemble ──
            print(f"\n  ── MLP Classifier (ensemble) ──")
            mlp_model = train_mlp_classifier(X_tr, y_tr, X_va, y_va)
            mlp_proba = mlp_model.predict_proba(X_va)[:, 1]
            mlp_results = full_evaluation(y_va, va_ret, mlp_proba,
                                          label=f"MLP {target_col} (Val)")

            # Ensemble: 0.7 LightGBM + 0.3 MLP
            ens_proba = 0.7 * va_proba + 0.3 * mlp_proba
            print(f"\n  ── Ensemble (0.7 LGBM + 0.3 MLP) ──")
            ens_results = full_evaluation(y_va, va_ret, ens_proba,
                                          label=f"Ensemble {target_col} (Val)")

            # Save MLP
            import joblib
            joblib.dump(mlp_model, f"{args.output_dir}/mlp_{target_col}.pkl")

            # ── Logistic Regression Baseline ──
            print(f"\n  ── Logistic Regression Baseline ──")
            lr_model = train_logistic_baseline(X_tr, y_tr)
            lr_proba = lr_model.predict_proba(X_va)[:, 1]
            lr_results = full_evaluation(y_va, va_ret, lr_proba,
                                         label=f"LR Baseline {target_col} (Val)")

            all_results[target_col] = {
                'lgbm_val': va_results,
                'lgbm_test': te_results,
                'mlp_val': mlp_results,
                'ensemble_val': ens_results,
                'lr_val': lr_results,
                'params': lgbm_params,
            }

        else:
            # ── LightGBM Regressor ──
            print(f"\n  ── LightGBM Regressor ──")
            lgbm_params = {}
            if args.tune:
                print("  Tuning...")
                lgbm_params = tune_lgbm(X_tr, y_tr, X_va, y_va,
                                        target_type='regression',
                                        n_trials=args.n_trials)
                print(f"  Best params: {lgbm_params}")

            model = train_lgbm_regressor(X_tr, y_tr, X_va, y_va, params=lgbm_params)

            # Validation: regression + trading metrics (predicted return = ranking score)
            va_pred = model.predict(X_va)
            va_regime = val['regime_score'].values[va_valid] if 'regime_score' in val.columns else None
            print("\n  --- Validation ---")
            va_results = full_evaluation(
                None, y_va, va_pred,
                label=f"LightGBM {target_col} (Val)",
                regime_values=va_regime,
            )

            # Test evaluation
            te_results = None
            if len(X_te) > 100:
                te_pred = model.predict(X_te)
                te_regime = test['regime_score'].values[te_valid] if 'regime_score' in test.columns else None
                print("\n  --- Test ---")
                te_results = full_evaluation(
                    None, y_te, te_pred,
                    label=f"LightGBM {target_col} (Test)",
                    regime_values=te_regime,
                )

            # Feature importance
            print("\n  --- Feature Importance (Gain) ---")
            fi = feature_importance_report(model, feat_cols, 'gain', 20)
            print(fi[['rank', 'feature', 'importance', 'pct']].to_string(index=False))

            # SHAP
            if args.shap:
                print("\n  --- SHAP ---")
                shap_df = shap_importance(model, X_va, feat_cols)
                if not shap_df.empty:
                    print(shap_df.head(20).to_string(index=False))
                    shap_df.to_csv(f"{args.output_dir}/shap_{target_col}.csv", index=False)

            fi.to_csv(f"{args.output_dir}/importance_{target_col}.csv", index=False)
            model.booster_.save_model(f"{args.output_dir}/lgbm_{target_col}.txt")

            all_results[target_col] = {
                'lgbm_val': va_results,
                'lgbm_test': te_results,
                'params': lgbm_params,
            }

        # ── Walk-Forward CV ──
        if not args.no_walkforward:
            print(f"\n  ── Walk-Forward CV ({target_col}) ──")
            model_type = 'lgbm_clf' if is_clf else 'lgbm_reg'
            wf_results = walk_forward_evaluate(dataset, target_col, feat_cols, model_type)
            if wf_results:
                wf_df = pd.DataFrame(wf_results)
                print(f"\n  Walk-Forward Özet:")
                if is_clf:
                    print(f"    Mean PR-AUC: {wf_df['pr_auc'].mean():.3f} ± {wf_df['pr_auc'].std():.3f}")
                    print(f"    Mean F1:     {wf_df['f1'].mean():.3f} ± {wf_df['f1'].std():.3f}")
                    if 'top_wr' in wf_df.columns:
                        print(f"    Mean Top WR: {wf_df['top_wr'].mean():.1%} ± {wf_df['top_wr'].std():.1%}")
                else:
                    print(f"    Mean MAE:    {wf_df['mae'].mean():.3f} ± {wf_df['mae'].std():.3f}")
                    print(f"    Mean Corr:   {wf_df['corr'].mean():.3f} ± {wf_df['corr'].std():.3f}")
                    if 'top_wr' in wf_df.columns:
                        print(f"    Mean Top WR: {wf_df['top_wr'].mean():.1%} ± {wf_df['top_wr'].std():.1%}")
                wf_df.to_csv(f"{args.output_dir}/wf_{target_col}.csv", index=False)

    # ── COMPARATIVE SUMMARY ──
    print(f"\n{'='*60}")
    print(f"  KARŞILAŞTIRMA ÖZETİ")
    print(f"{'='*60}")
    print(f"  Feature Group: {args.feature_group} ({len(feat_cols)} feature)")
    print(f"\n  {'Target':<10} {'Type':<6} {'Top10%WR':>9} {'Top10%Med':>10} {'Spread':>8} "
          f"{'Spearman':>9} {'Mono':>6}")
    print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*6}")

    for target_col, res in all_results.items():
        lgbm_val = res.get('lgbm_val', {})
        tm = lgbm_val.get('trading', {}) if lgbm_val else {}
        if not tm or 'error' in tm:
            print(f"  {target_col:<10} {'clf' if target_col.startswith('up_') else 'reg':<6}"
                  f"    n/a       n/a      n/a       n/a    n/a")
            continue

        ttype = 'clf' if target_col.startswith('up_') else 'reg'
        t10_wr = tm.get('top_10pct_wr', 0)
        t10_med = tm.get('top_10pct_med_ret', 0)
        spread = tm.get('spread_wr', 0)
        spearman = tm.get('spearman_corr', 0)
        mono = tm.get('monotonicity', 0)

        print(f"  {target_col:<10} {ttype:<6} {t10_wr:>8.1%} {t10_med:>+9.2f}% "
              f"{spread:>+7.1%} {spearman:>+8.3f} {mono:>+5.3f}")

    # Test sonuçları varsa ayrıca göster
    has_test = any(r.get('lgbm_test') for r in all_results.values())
    if has_test:
        print(f"\n  TEST SET:")
        print(f"  {'Target':<10} {'Type':<6} {'Top10%WR':>9} {'Top10%Med':>10} {'Spread':>8} "
              f"{'Spearman':>9} {'Mono':>6}")
        print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*6}")
        for target_col, res in all_results.items():
            lgbm_test = res.get('lgbm_test', {})
            tm = lgbm_test.get('trading', {}) if lgbm_test else {}
            if not tm or 'error' in tm:
                continue
            ttype = 'clf' if target_col.startswith('up_') else 'reg'
            print(f"  {target_col:<10} {ttype:<6} {tm.get('top_10pct_wr', 0):>8.1%} "
                  f"{tm.get('top_10pct_med_ret', 0):>+9.2f}% "
                  f"{tm.get('spread_wr', 0):>+7.1%} "
                  f"{tm.get('spearman_corr', 0):>+8.3f} "
                  f"{tm.get('monotonicity', 0):>+5.3f}")

    # Save results JSON
    results_path = f"{args.output_dir}/results_{args.feature_group}.json"
    try:
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {}
            for mk, mv in v.items():
                if isinstance(mv, dict):
                    serializable[k][mk] = {
                        kk: vv for kk, vv in mv.items()
                        if not isinstance(vv, (list, np.ndarray))
                    }
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Sonuçlar: {results_path}")
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI — Süre: {time.time()-t0:.0f} saniye")
    print(f"  Çıktılar: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
