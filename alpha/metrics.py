"""
Alpha Pipeline — Performans Metrikleri
=======================================
Portföy vs benchmark karşılaştırması.
"""

import numpy as np
import pandas as pd

from alpha.config import RISK_FREE_RATE


def compute_alpha_metrics(equity_curve: list, initial_capital: float) -> dict:
    """Kapsamlı alpha metrikleri hesapla.

    Args:
        equity_curve: backtest çıktısı (list of dicts with date, equity, benchmark)
        initial_capital: başlangıç sermayesi

    Returns:
        dict: tüm metrikler
    """
    if not equity_curve or len(equity_curve) < 5:
        return _empty_metrics()

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    equity = df['equity']
    benchmark = df['benchmark'].dropna()
    n_days = len(equity)

    # Günlük getiriler
    port_returns = equity.pct_change().dropna()
    bm_returns = benchmark.pct_change().dropna() if len(benchmark) > 1 else pd.Series(dtype=float)

    # Ortak tarihler
    common = port_returns.index.intersection(bm_returns.index)
    pr = port_returns.loc[common] if len(common) > 0 else port_returns
    br = bm_returns.loc[common] if len(common) > 0 else pd.Series(dtype=float)

    # ── Toplam & Yıllık Getiri ──
    final_eq = float(equity.iloc[-1])
    total_return = (final_eq / initial_capital - 1) * 100
    years = n_days / 252
    annual_return = ((final_eq / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0

    bm_final = float(benchmark.iloc[-1]) if len(benchmark) > 0 else initial_capital
    bm_total = (bm_final / initial_capital - 1) * 100
    bm_annual = ((bm_final / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0

    # ── Alpha ──
    alpha = annual_return - bm_annual

    # ── Beta & Jensen's Alpha ──
    if len(br) > 10:
        cov_pb = np.cov(pr.values, br.values)
        beta = cov_pb[0, 1] / cov_pb[1, 1] if cov_pb[1, 1] > 1e-10 else 1.0
        jensens = annual_return - (RISK_FREE_RATE * 100 + beta * (bm_annual - RISK_FREE_RATE * 100))
    else:
        beta = 1.0
        jensens = alpha

    # ── Volatilite ──
    ann_vol = float(pr.std() * np.sqrt(252) * 100) if len(pr) > 1 else 0

    # ── Sharpe Ratio ──
    rf_daily = RISK_FREE_RATE / 252
    excess = pr - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 1e-10 else 0

    # ── Sortino Ratio ──
    downside = pr[pr < rf_daily] - rf_daily
    downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else ann_vol
    sortino = (annual_return / 100 - RISK_FREE_RATE) / downside_std if downside_std > 1e-10 else 0

    # ── Information Ratio ──
    if len(br) > 10:
        active = pr - br
        tracking_error = float(active.std() * np.sqrt(252))
        info_ratio = float(active.mean() * 252 / tracking_error) if tracking_error > 1e-10 else 0
    else:
        info_ratio = 0
        tracking_error = 0

    # ── Max Drawdown ──
    peak = equity.expanding().max()
    dd = (equity / peak - 1) * 100
    max_dd = float(dd.min())

    # DD süresi
    in_dd = dd < 0
    dd_duration = 0
    current_dd_len = 0
    for v in in_dd:
        if v:
            current_dd_len += 1
            dd_duration = max(dd_duration, current_dd_len)
        else:
            current_dd_len = 0

    # ── Calmar Ratio ──
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # ── Aylık istatistikler ──
    monthly = equity.resample('ME').last().pct_change().dropna() * 100
    win_months = (monthly > 0).sum()
    total_months = len(monthly)
    win_month_pct = win_months / total_months * 100 if total_months > 0 else 0

    return {
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'benchmark_total': round(bm_total, 2),
        'benchmark_annual': round(bm_annual, 2),
        'alpha': round(alpha, 2),
        'beta': round(beta, 3),
        'jensens_alpha': round(jensens, 2),
        'ann_volatility': round(ann_vol, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3),
        'information_ratio': round(info_ratio, 3),
        'tracking_error': round(tracking_error * 100, 2) if tracking_error else 0,
        'max_drawdown': round(max_dd, 2),
        'max_dd_duration_days': dd_duration,
        'calmar_ratio': round(calmar, 3),
        'win_month_pct': round(win_month_pct, 1),
        'n_months': total_months,
        'n_trading_days': n_days,
        'years': round(years, 2),
    }


def compute_monthly_returns(equity_curve: list) -> pd.DataFrame:
    """Aylık getiri heatmap verisi.

    Returns:
        DataFrame: yıl (index) × ay (columns), değerler % getiri
    """
    if not equity_curve:
        return pd.DataFrame()

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    monthly = df['equity'].resample('ME').last().pct_change() * 100
    monthly = monthly.dropna()

    table = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values,
    })

    pivot = table.pivot_table(values='return', index='year', columns='month', aggfunc='first')
    pivot.columns = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz',
                     'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara'][:len(pivot.columns)]
    return pivot


def compute_trade_stats(trades: list) -> dict:
    """Trade bazlı istatistikler.

    Args:
        trades: TradeRecord listesi (dataclass)

    Returns:
        dict: win_rate, avg_win, avg_loss, profit_factor, avg_hold
    """
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'avg_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'avg_hold': 0}

    pnls = [t.pnl_pct for t in trades]
    holds = [t.hold_days for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    pf = sum_wins / sum_losses if sum_losses > 0 else float('inf')

    # Çıkış nedeni dağılımı
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        'n_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'avg_hold': round(np.mean(holds), 1),
        'exit_reasons': reasons,
    }


def _empty_metrics():
    return {
        'total_return': 0, 'annual_return': 0, 'benchmark_total': 0,
        'benchmark_annual': 0, 'alpha': 0, 'beta': 1.0, 'jensens_alpha': 0,
        'ann_volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
        'information_ratio': 0, 'tracking_error': 0, 'max_drawdown': 0,
        'max_dd_duration_days': 0, 'calmar_ratio': 0, 'win_month_pct': 0,
        'n_months': 0, 'n_trading_days': 0, 'years': 0,
    }
