"""
Alpha Pipeline — Portföy Optimizasyonu (Aşama 4-5)
====================================================
Aşama 4: Markowitz Mean-Variance Optimization
Aşama 5: Maximum Sharpe Ratio Selection
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha.config import (
    COV_LOOKBACK_DAYS, MIN_STOCKS, MAX_STOCKS,
    WEIGHT_MIN, WEIGHT_MAX, RISK_FREE_RATE,
    TARGET_PORTFOLIO_SIZE, SCORE_TILT_FACTOR,
)


# ═══════════════════════════════════════════
# Ledoit-Wolf Shrinkage
# ═══════════════════════════════════════════

def _ledoit_wolf_shrink(sample_cov: np.ndarray, n_obs: int) -> np.ndarray:
    """Ledoit-Wolf lineer shrinkage — hedef: ölçeklenmiş birim matris.

    Kovaryans matrisini n_stocks/n_obs oranı yüksek olduğunda stabilize eder.
    """
    p = sample_cov.shape[0]
    if p <= 1 or n_obs <= 1:
        return sample_cov

    # Hedef: scaled identity (ortalama varyans × I)
    trace_s = np.trace(sample_cov)
    mu = trace_s / p
    target = mu * np.eye(p)

    # Shrinkage yoğunluğu (analitik formül)
    delta = sample_cov - target
    delta_sq_sum = np.sum(delta ** 2)

    # Basitleştirilmiş shrinkage katsayısı
    shrinkage = min(1.0, max(0.0, delta_sq_sum / (n_obs * delta_sq_sum + 1e-10)))
    # Pratikte: p/n büyüdükçe shrinkage → 1'e yaklaşır
    shrinkage = min(1.0, (p / n_obs) * 0.5)

    return (1 - shrinkage) * sample_cov + shrinkage * target


# ═══════════════════════════════════════════
# Return & Covariance Hesaplama
# ═══════════════════════════════════════════

def _compute_return_stats(price_data: dict, tickers: list,
                          as_of_idx: int = -1,
                          lookback: int = COV_LOOKBACK_DAYS,
                          ) -> tuple:
    """Beklenen getiri (μ) ve kovaryans matrisi (Σ) hesapla.

    Args:
        price_data: {ticker: DataFrame(OHLCV)}
        tickers: analiz edilecek hisseler
        as_of_idx: veriyi bu indekse kadar kes (-1 = son)
        lookback: geri bakış penceresi (gün)

    Returns:
        (mu: np.array, cov: np.array, valid_tickers: list)
    """
    # Ortak tarih indeksi oluştur
    close_dict = {}
    for t in tickers:
        df = price_data.get(t)
        if df is None:
            continue
        c = df['Close'].iloc[:as_of_idx] if as_of_idx != -1 else df['Close']
        if len(c) < lookback:
            continue
        close_dict[t] = c.iloc[-lookback:]

    if len(close_dict) < MIN_STOCKS:
        return None, None, []

    # DataFrame'e çevir, ortak tarihleri al
    close_df = pd.DataFrame(close_dict)
    close_df = close_df.dropna(axis=1, how='any')  # eksik günleri olan hisseleri çıkar

    valid_tickers = list(close_df.columns)
    if len(valid_tickers) < MIN_STOCKS:
        return None, None, []

    # Log returns
    log_returns = np.log(close_df / close_df.shift(1)).dropna()
    n_obs = len(log_returns)

    if n_obs < 30:
        return None, None, []

    # Yıllıklandırılmış μ ve Σ
    mu = log_returns.mean().values * 252
    sample_cov = log_returns.cov().values * 252

    # Ledoit-Wolf shrinkage
    cov = _ledoit_wolf_shrink(sample_cov, n_obs)

    return mu, cov, valid_tickers


# ═══════════════════════════════════════════
# AŞAMA 4: Markowitz Optimizasyonu
# ═══════════════════════════════════════════

def stage4_markowitz(price_data: dict, candidates: list,
                     as_of_idx: int = -1) -> dict:
    """Markowitz mean-variance optimizasyonu.

    Args:
        price_data: {ticker: DataFrame}
        candidates: scan_universe çıktısı (passed=True olanlar)
        as_of_idx: veri kesim noktası

    Returns:
        dict: {tickers, mu, cov, n_candidates, valid}
    """
    tickers = [c['ticker'] for c in candidates if c.get('passed', False)]

    if len(tickers) > MAX_STOCKS:
        # Composite skora göre ilk N'i al
        sorted_cands = sorted(candidates, key=lambda x: x['composite_score'], reverse=True)
        tickers = [c['ticker'] for c in sorted_cands if c.get('passed')][:MAX_STOCKS]

    mu, cov, valid_tickers = _compute_return_stats(
        price_data, tickers, as_of_idx=as_of_idx
    )

    if mu is None:
        return {'tickers': [], 'mu': None, 'cov': None, 'n_candidates': 0, 'valid': False}

    return {
        'tickers': valid_tickers,
        'mu': mu,
        'cov': cov,
        'n_candidates': len(valid_tickers),
        'valid': True,
    }


# ═══════════════════════════════════════════
# AŞAMA 5: Maximum Sharpe Seçimi
# ═══════════════════════════════════════════

def stage5_max_sharpe(markowitz_result: dict,
                      candidate_scores: dict = None) -> dict:
    """Maximum Sharpe Ratio portföy seçimi.

    Args:
        markowitz_result: stage4 çıktısı
        candidate_scores: {ticker: composite_score} — signal tilt için

    Returns:
        {weights, expected_return, expected_risk, sharpe_ratio, n_stocks, tickers}
    """
    empty = {
        'weights': {}, 'expected_return': 0.0, 'expected_risk': 0.0,
        'sharpe_ratio': 0.0, 'n_stocks': 0, 'tickers': [],
    }

    if not markowitz_result.get('valid'):
        return empty

    tickers = markowitz_result['tickers']
    mu = markowitz_result['mu'].copy()
    cov = markowitz_result['cov']
    n = len(tickers)
    rf = RISK_FREE_RATE

    if n < MIN_STOCKS:
        return empty

    # Score-weighted tilt: composite skor → beklenen getiriye küçük bonus
    if candidate_scores:
        for i, t in enumerate(tickers):
            s = candidate_scores.get(t, 50)
            mu[i] += SCORE_TILT_FACTOR * s / 100

    # Optimizasyon: maximize Sharpe = minimize -Sharpe
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 1e10
        return -(port_ret - rf) / port_vol

    # Kısıtlar
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(WEIGHT_MIN, WEIGHT_MAX)] * n

    # Başlangıç: eşit ağırlık
    w0 = np.ones(n) / n

    try:
        opt = minimize(neg_sharpe, w0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-12})

        if not opt.success:
            # Fallback: eşit ağırlık
            w_opt = np.ones(n) / n
        else:
            w_opt = opt.x
    except Exception:
        w_opt = np.ones(n) / n

    # 2-step pruning: WEIGHT_MIN altını sıfırla, yeniden optimize et
    mask = w_opt >= WEIGHT_MIN
    if mask.sum() < MIN_STOCKS:
        # Çok az hisse kaldıysa top N'i tut
        top_idx = np.argsort(w_opt)[-TARGET_PORTFOLIO_SIZE:]
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True

    if mask.sum() < n:
        # Kalan hisselerle yeniden optimize et
        sub_tickers = [tickers[i] for i in range(n) if mask[i]]
        sub_mu = mu[mask]
        sub_cov = cov[np.ix_(mask, mask)]
        n_sub = len(sub_tickers)

        def neg_sharpe_sub(w):
            pr = w @ sub_mu
            pv = np.sqrt(w @ sub_cov @ w)
            if pv < 1e-10:
                return 1e10
            return -(pr - rf) / pv

        constraints_sub = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds_sub = [(WEIGHT_MIN, WEIGHT_MAX)] * n_sub
        w0_sub = np.ones(n_sub) / n_sub

        try:
            opt2 = minimize(neg_sharpe_sub, w0_sub, method='SLSQP',
                            bounds=bounds_sub, constraints=constraints_sub,
                            options={'maxiter': 1000, 'ftol': 1e-12})
            if opt2.success:
                w_final = opt2.x
            else:
                w_final = w0_sub
        except Exception:
            w_final = w0_sub

        final_tickers = sub_tickers
        final_mu = sub_mu
        final_cov = sub_cov
    else:
        w_final = w_opt
        final_tickers = tickers
        final_mu = mu
        final_cov = cov

    # Sonuç hesapla
    port_ret = float(w_final @ final_mu)
    port_risk = float(np.sqrt(w_final @ final_cov @ w_final))
    sharpe = (port_ret - rf) / port_risk if port_risk > 1e-10 else 0.0

    weights = {t: round(float(w), 4) for t, w in zip(final_tickers, w_final) if w >= 0.01}

    return {
        'weights': weights,
        'expected_return': round(port_ret * 100, 2),
        'expected_risk': round(port_risk * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'n_stocks': len(weights),
        'tickers': list(weights.keys()),
    }


# ═══════════════════════════════════════════
# Kolaylık: Stage 4 + 5 tek çağrı
# ═══════════════════════════════════════════

def build_portfolio(price_data: dict, candidates: list,
                    as_of_idx: int = -1) -> dict:
    """Aşama 4 + 5'i tek çağrıda çalıştır.

    Args:
        price_data: {ticker: DataFrame}
        candidates: scan_universe çıktısı
        as_of_idx: veri kesim noktası

    Returns:
        stage5 çıktısı (weights, sharpe, risk, return)
    """
    passed = [c for c in candidates if c.get('passed', False)]

    if len(passed) < MIN_STOCKS:
        return {
            'weights': {}, 'expected_return': 0.0, 'expected_risk': 0.0,
            'sharpe_ratio': 0.0, 'n_stocks': 0, 'tickers': [],
        }

    mkw = stage4_markowitz(price_data, passed, as_of_idx=as_of_idx)

    scores = {c['ticker']: c['composite_score'] for c in passed}

    return stage5_max_sharpe(mkw, candidate_scores=scores)
