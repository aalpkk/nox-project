"""
ML Evaluation — Trading-Specific Metrikleri
Classification + regression + trading performans değerlendirme.
Decile, regime-segment, ranking kalitesi, median return.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
)


# ═══════════════════════════════════════════
# CLASSIFICATION METRİKLERİ
# ═══════════════════════════════════════════

def classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Standart classification metrikleri.

    Args:
        y_true: binary target (0/1)
        y_pred_proba: predicted probability of class 1
        threshold: classification threshold

    Returns:
        dict — metric name: value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_t = y_true[valid]
    y_p = y_pred_proba[valid]
    y_pred = (y_p >= threshold).astype(int)

    metrics = {
        'n': len(y_t),
        'base_rate': float(y_t.mean()),
        'precision': precision_score(y_t, y_pred, zero_division=0),
        'recall': recall_score(y_t, y_pred, zero_division=0),
        'f1': f1_score(y_t, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else 0,
        'roc_auc': roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else 0,
    }
    return metrics


# ═══════════════════════════════════════════
# REGRESSION METRİKLERİ
# ═══════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """Standart regression metrikleri."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]

    return {
        'n': len(y_t),
        'mae': mean_absolute_error(y_t, y_p),
        'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
        'corr': np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 2 else 0,
        'mean_true': float(y_t.mean()),
        'mean_pred': float(y_p.mean()),
    }


# ═══════════════════════════════════════════
# TRADING-SPESİFİK METRİKLER
# ═══════════════════════════════════════════

def trading_metrics(y_true_ret, y_pred_proba, n_deciles=10):
    """
    Trading-specific metrikleri: decile analysis.

    Args:
        y_true_ret: gerçek forward return (%)
        y_pred_proba: model predicted probability/score
        n_deciles: kaç gruba böl

    Returns:
        dict — summary metrics + decile detail
    """
    y_true_ret = np.asarray(y_true_ret, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    valid = ~(np.isnan(y_true_ret) | np.isnan(y_pred_proba))
    ret = pd.Series(y_true_ret[valid])
    prob = pd.Series(y_pred_proba[valid])

    if len(ret) < 100:
        return {'error': 'insufficient_data', 'n': len(ret)}

    # Decile analysis
    try:
        decile = pd.qcut(prob, n_deciles, labels=False, duplicates='drop')
    except ValueError:
        decile = pd.cut(prob, n_deciles, labels=False)

    decile_stats = []
    for d in sorted(decile.dropna().unique()):
        mask = decile == d
        d_ret = ret[mask]
        decile_stats.append({
            'decile': int(d),
            'n': int(mask.sum()),
            'mean_ret': float(d_ret.mean()),
            'med_ret': float(d_ret.median()),
            'wr': float((d_ret > 0).mean()),
            'std': float(d_ret.std()),
            'mean_score': float(prob[mask].mean()),
        })

    df_dec = pd.DataFrame(decile_stats)

    # Top decile (en yüksek skor)
    top = df_dec.iloc[-1] if len(df_dec) > 0 else {}
    bottom = df_dec.iloc[0] if len(df_dec) > 0 else {}

    # Monotonicity: WR decile'lar arası tutarlı artıyor mu?
    if len(df_dec) >= 3:
        wr_values = df_dec['wr'].values
        mono_score = np.corrcoef(np.arange(len(wr_values)), wr_values)[0, 1]
    else:
        mono_score = 0.0

    # Top 10% median return
    top_10_mask = prob >= prob.quantile(0.9)
    top_10_ret = ret[top_10_mask]
    bot_10_mask = prob <= prob.quantile(0.1)
    bot_10_ret = ret[bot_10_mask]

    # Ranking quality: Spearman correlation (score sıralaması vs return sıralaması)
    spearman_corr, spearman_p = spearmanr(prob, ret)

    return {
        'n': len(ret),
        'top_decile_wr': float(top.get('wr', 0)),
        'top_decile_mean_ret': float(top.get('mean_ret', 0)),
        'top_decile_med_ret': float(top.get('med_ret', 0)),
        'top_decile_n': int(top.get('n', 0)),
        'bottom_decile_wr': float(bottom.get('wr', 0)),
        'bottom_decile_mean_ret': float(bottom.get('mean_ret', 0)),
        'spread_wr': float(top.get('wr', 0)) - float(bottom.get('wr', 0)),
        'spread_ret': float(top.get('mean_ret', 0)) - float(bottom.get('mean_ret', 0)),
        'monotonicity': float(mono_score),
        'spearman_corr': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        'top_10pct_wr': float((top_10_ret > 0).mean()) if len(top_10_ret) > 0 else 0,
        'top_10pct_med_ret': float(top_10_ret.median()) if len(top_10_ret) > 0 else 0,
        'top_10pct_n': int(top_10_mask.sum()),
        'bot_10pct_wr': float((bot_10_ret > 0).mean()) if len(bot_10_ret) > 0 else 0,
        'bot_10pct_med_ret': float(bot_10_ret.median()) if len(bot_10_ret) > 0 else 0,
        'deciles': decile_stats,
    }


def regime_segment_metrics(y_true_ret, y_pred_proba, regime_values, n_top_pct=10):
    """
    Rejim segmentlerine göre trading performansı.

    Args:
        y_true_ret: gerçek forward return (%)
        y_pred_proba: model score
        regime_values: regime_score (0=CHOPPY, 1=EARLY, 2=TREND, 3=FULL_TREND)
        n_top_pct: top N% seçim (default: top 10%)

    Returns:
        list of dicts — per-regime metrics
    """
    y_true_ret = np.asarray(y_true_ret, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    regime_values = np.asarray(regime_values, dtype=float)
    valid = ~(np.isnan(y_true_ret) | np.isnan(y_pred_proba) | np.isnan(regime_values))
    ret = pd.Series(y_true_ret[valid])
    prob = pd.Series(y_pred_proba[valid])
    regime = pd.Series(regime_values[valid])

    regime_names = {0: 'CHOPPY', 1: 'EARLY', 2: 'TREND', 3: 'FULL_TREND'}
    results = []

    for r_val in sorted(regime.unique()):
        mask = regime == r_val
        n = mask.sum()
        if n < 20:
            continue
        seg_ret = ret[mask]
        seg_prob = prob[mask]

        # Global top N% threshold uygulanırsa bu segment kaçını yakalar?
        global_threshold = prob.quantile(1 - n_top_pct / 100)
        top_mask = seg_prob >= global_threshold
        top_n = top_mask.sum()
        top_ret = seg_ret[top_mask]

        results.append({
            'regime': regime_names.get(int(r_val), f'R{int(r_val)}'),
            'regime_val': int(r_val),
            'n_total': int(n),
            'base_wr': float((seg_ret > 0).mean()),
            'base_mean_ret': float(seg_ret.mean()),
            'top_n': int(top_n),
            'top_wr': float((top_ret > 0).mean()) if top_n > 10 else None,
            'top_mean_ret': float(top_ret.mean()) if top_n > 10 else None,
            'top_med_ret': float(top_ret.median()) if top_n > 10 else None,
            'coverage_pct': float(top_n / n * 100) if n > 0 else 0,
        })

    return results


def hit_rate_by_band(y_true_ret, y_pred_proba, bands=None):
    """
    Score band'larına göre hit rate analizi.

    Args:
        y_true_ret: gerçek forward return (%)
        y_pred_proba: model score
        bands: list of (low, high, label) tuples

    Returns:
        list of dicts
    """
    if bands is None:
        bands = [
            (0.0, 0.3, '0-30%'),
            (0.3, 0.4, '30-40%'),
            (0.4, 0.5, '40-50%'),
            (0.5, 0.6, '50-60%'),
            (0.6, 0.7, '60-70%'),
            (0.7, 0.8, '70-80%'),
            (0.8, 1.0, '80-100%'),
        ]

    y_true_ret = np.asarray(y_true_ret, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    valid = ~(np.isnan(y_true_ret) | np.isnan(y_pred_proba))
    ret = pd.Series(y_true_ret[valid])
    prob = pd.Series(y_pred_proba[valid])

    results = []
    for lo, hi, label in bands:
        mask = (prob >= lo) & (prob < hi)
        n = mask.sum()
        if n > 0:
            band_ret = ret[mask]
            results.append({
                'band': label,
                'n': int(n),
                'wr': float((band_ret > 0).mean()),
                'mean_ret': float(band_ret.mean()),
                'med_ret': float(band_ret.median()),
            })
    return results


def calibration_analysis(y_true, y_pred_proba, n_bins=10):
    """
    Calibration: predicted probability vs actual frequency.

    Returns:
        list of dicts {bin_center, predicted_prob, actual_freq, n}
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_t = y_true[valid]
    y_p = y_pred_proba[valid]

    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    for i in range(n_bins):
        mask = (y_p >= bins[i]) & (y_p < bins[i+1])
        n = mask.sum()
        if n > 0:
            results.append({
                'bin_center': (bins[i] + bins[i+1]) / 2,
                'predicted_prob': float(y_p[mask].mean()),
                'actual_freq': float(y_t[mask].mean()),
                'n': int(n),
            })
    return results


# ═══════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════

def feature_importance_report(model, feature_names, importance_type='gain', top_n=30):
    """
    LightGBM feature importance raporu.

    Args:
        model: trained LightGBM model
        feature_names: list of feature names
        importance_type: 'gain' or 'split'
        top_n: top N feature

    Returns:
        DataFrame — feature, importance, rank
    """
    if hasattr(model, 'booster_'):
        imp = model.booster_.feature_importance(importance_type=importance_type)
    elif hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    else:
        imp = model.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': imp,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['pct'] = df['importance'] / df['importance'].sum() * 100
    return df.head(top_n)


def shap_importance(model, X, feature_names=None, max_samples=5000):
    """
    SHAP-based feature importance.

    Args:
        model: trained model
        X: feature matrix (numpy or DataFrame)
        feature_names: kolon isimleri
        max_samples: SHAP hesaplama için max satır

    Returns:
        DataFrame — feature, mean_abs_shap, rank
    """
    try:
        import shap
    except ImportError:
        print("  [!] shap paketi yüklü değil, atlanıyor.")
        return pd.DataFrame()

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Binary classification: shap_values might be list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    mean_abs = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f'f{i}' for i in range(len(mean_abs))]

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df


# ═══════════════════════════════════════════
# FULL EVALUATION RAPORU
# ═══════════════════════════════════════════

def full_evaluation(y_true_binary, y_true_ret, y_pred_proba, label="Model",
                    regime_values=None):
    """
    Tam değerlendirme raporu yazdır.

    Args:
        y_true_binary: binary target (0/1) — None if regression-only
        y_true_ret: forward return (%)
        y_pred_proba: predicted probability or score
        label: model adı
        regime_values: optional regime_score array for segment analysis
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    cm = None
    cal = None

    # Classification (skip for regression targets)
    if y_true_binary is not None:
        cm = classification_metrics(
            np.array(y_true_binary, dtype=float),
            np.array(y_pred_proba, dtype=float)
        )
        print(f"\n  Classification (N={cm['n']:,}):")
        print(f"    Base Rate:  {cm['base_rate']:.1%}")
        print(f"    Precision:  {cm['precision']:.1%}")
        print(f"    Recall:     {cm['recall']:.1%}")
        print(f"    F1:         {cm['f1']:.3f}")
        print(f"    PR-AUC:     {cm['pr_auc']:.3f}")
        print(f"    ROC-AUC:    {cm['roc_auc']:.3f}")

    # Trading
    tm = trading_metrics(
        np.array(y_true_ret, dtype=float),
        np.array(y_pred_proba, dtype=float)
    )
    if 'error' not in tm:
        print(f"\n  Trading Metrics:")
        print(f"    Top Decile WR:     {tm['top_decile_wr']:.1%} (N={tm['top_decile_n']})")
        print(f"    Top Decile Ret:    {tm['top_decile_mean_ret']:+.2f}%")
        print(f"    Top Decile MedRet: {tm['top_decile_med_ret']:+.2f}%")
        print(f"    Bottom Decile WR:  {tm['bottom_decile_wr']:.1%}")
        print(f"    Spread (WR):       {tm['spread_wr']:+.1%}")
        print(f"    Spread (Ret):      {tm['spread_ret']:+.2f}%")
        print(f"    Monotonicity:      {tm['monotonicity']:.3f}")
        print(f"    Ranking (Spearman):{tm['spearman_corr']:+.3f} (p={tm['spearman_p']:.4f})")
        print(f"\n  Top/Bottom 10%:")
        print(f"    Top 10% WR:     {tm['top_10pct_wr']:.1%} (N={tm['top_10pct_n']}, "
              f"med={tm['top_10pct_med_ret']:+.2f}%)")
        print(f"    Bot 10% WR:     {tm['bot_10pct_wr']:.1%} "
              f"(med={tm['bot_10pct_med_ret']:+.2f}%)")
        print(f"    Separation:     {tm['top_10pct_wr'] - tm['bot_10pct_wr']:+.1%} WR | "
              f"{tm['top_10pct_med_ret'] - tm['bot_10pct_med_ret']:+.2f}% medRet")

    # Hit rate by band
    hr = hit_rate_by_band(
        np.array(y_true_ret, dtype=float),
        np.array(y_pred_proba, dtype=float)
    )
    if hr:
        print(f"\n  Hit Rate by Score Band:")
        print(f"    {'Band':<10} {'N':>6} {'WR':>6} {'MeanRet':>8} {'MedRet':>8}")
        for row in hr:
            print(f"    {row['band']:<10} {row['n']:>6} {row['wr']:>5.1%} "
                  f"{row['mean_ret']:>+7.2f}% {row['med_ret']:>+7.2f}%")

    # Regime segment analysis
    rsm = None
    if regime_values is not None:
        rsm = regime_segment_metrics(
            np.array(y_true_ret, dtype=float),
            np.array(y_pred_proba, dtype=float),
            np.array(regime_values, dtype=float)
        )
        if rsm:
            print(f"\n  Regime Segments (top 10% seçim):")
            print(f"    {'Regime':<12} {'N':>6} {'BaseWR':>7} {'TopN':>5} {'TopWR':>6} {'TopMed':>7}")
            for row in rsm:
                tw = f"{row['top_wr']:.1%}" if row['top_wr'] is not None else "  n/a"
                tm_str = f"{row['top_med_ret']:+.2f}%" if row['top_med_ret'] is not None else "   n/a"
                print(f"    {row['regime']:<12} {row['n_total']:>6} {row['base_wr']:>6.1%} "
                      f"{row['top_n']:>5} {tw:>6} {tm_str:>7}")

    # Calibration (classification only)
    if y_true_binary is not None:
        cal = calibration_analysis(
            np.array(y_true_binary, dtype=float),
            np.array(y_pred_proba, dtype=float)
        )
        if cal:
            print(f"\n  Calibration:")
            print(f"    {'Predicted':>9} {'Actual':>8} {'N':>6}")
            for row in cal:
                print(f"    {row['predicted_prob']:>8.1%} {row['actual_freq']:>7.1%} {row['n']:>6}")

    print(f"\n{'='*60}")
    return {
        'classification': cm, 'trading': tm, 'hit_rate': hr,
        'calibration': cal, 'regime_segments': rsm,
    }
