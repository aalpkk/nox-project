"""
Performance attribution — trade seti üzerinden WR, getiri, Sharpe, IR, DD, per-bucket/regime.

Walking skeleton: closed_trades list[dict] → özet istatistikler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_equity_curve(
    trades: list[dict],
    all_data: dict[str, pd.DataFrame],
    *,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Mark-to-market günlük equity curve.

    Trade'ler paralel çalışır — her gün açık pozisyonların kontribüsyon'larının
    toplamı = portföy günlük getirisi. size_pct initial_capital'in %'si olarak
    yorumlanır (fixed-fractional).

    Returns:
        pd.Series indexed by date → equity (TL)
    """
    if not trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(trades).copy()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])

    start = df['entry_date'].min()
    end = df['exit_date'].max()

    all_dates = set()
    for t in df['ticker'].unique():
        sub = all_data.get(t)
        if sub is None:
            continue
        idx = pd.to_datetime(sub.index)
        mask = (idx >= start) & (idx <= end)
        all_dates.update(idx[mask].tolist())
    if not all_dates:
        return pd.Series(dtype=float)

    date_idx = pd.DatetimeIndex(sorted(all_dates))
    portfolio_daily_ret = pd.Series(0.0, index=date_idx)

    for _, trade in df.iterrows():
        ticker = trade['ticker']
        sub = all_data.get(ticker)
        if sub is None:
            continue
        close = sub['Close'] if 'Close' in sub.columns else sub['close']
        close.index = pd.to_datetime(close.index)

        mask = (close.index >= trade['entry_date']) & (close.index <= trade['exit_date'])
        trade_px = close[mask]
        if len(trade_px) < 2:
            continue
        daily_ret = trade_px.pct_change().fillna(0.0)
        size_frac = float(trade['size_pct']) / 100.0
        contribution = daily_ret * size_frac
        portfolio_daily_ret = portfolio_daily_ret.add(
            contribution.reindex(date_idx).fillna(0.0), fill_value=0.0,
        )

    equity_curve = initial_capital * (1.0 + portfolio_daily_ret).cumprod()
    return equity_curve


def performance_summary(trades: list[dict], *,
                        initial_capital: float = 100_000.0,
                        annual_trading_days: int = 250,
                        xu100_ret_series: pd.Series | None = None,
                        all_data: dict[str, pd.DataFrame] | None = None) -> dict:
    """
    Blended (tüm bucket'lar) özet.

    all_data verilirse: günlük mark-to-market equity curve hesaplanır (paralel
    pozisyonları doğru yönetir). Aksi halde trade-level additive proxy kullanılır.
    """
    if not trades:
        return {
            'n_trades': 0, 'win_rate': 0.0, 'avg_return_pct': 0.0,
            'total_return_pct': 0.0, 'sharpe': 0.0, 'max_dd_pct': 0.0,
            'profit_factor': 0.0,
        }

    df = pd.DataFrame(trades).copy()
    df = df.sort_values('exit_date').reset_index(drop=True)
    df['return_pct'] = df['return_pct'].astype(float)
    df['size_pct'] = df['size_pct'].astype(float)
    # size_pct initial capital'in %'si olarak yorumla (fixed-fractional)
    df['abs_pnl_tl'] = initial_capital * df['size_pct'] / 100.0 * df['return_pct'] / 100.0
    df['contrib_pct'] = df['size_pct'] / 100.0 * df['return_pct']

    wins = df[df['return_pct'] > 0]
    losses = df[df['return_pct'] <= 0]
    total_win_tl = wins['abs_pnl_tl'].sum() if len(wins) else 0.0
    total_loss_tl = abs(losses['abs_pnl_tl'].sum()) if len(losses) else 1e-9

    # Daily MTM equity curve (paralel positions) — all_data varsa kullan
    if all_data is not None:
        equity_curve = compute_daily_equity_curve(
            trades, all_data, initial_capital=initial_capital,
        )
        daily_ret = equity_curve.pct_change().dropna() if len(equity_curve) > 1 else pd.Series(dtype=float)
    else:
        equity_curve = initial_capital + df['abs_pnl_tl'].cumsum()
        equity_curve.index = pd.to_datetime(df['exit_date'])
        daily_ret = pd.Series(dtype=float)

    if len(equity_curve) > 0:
        peak = equity_curve.cummax()
        dd = (equity_curve / peak - 1.0) * 100.0
        max_dd = float(dd.min())
        final_equity = float(equity_curve.iloc[-1])
        total_ret = float(final_equity / initial_capital - 1.0) * 100.0
    else:
        max_dd = 0.0
        final_equity = initial_capital
        total_ret = 0.0

    # Sharpe / Sortino
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        mean_d = daily_ret.mean()
        std_d = daily_ret.std()
        sharpe = float(mean_d / std_d * np.sqrt(annual_trading_days))
        downside = daily_ret[daily_ret < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(mean_d / downside.std() * np.sqrt(annual_trading_days))
        else:
            sortino = float('inf') if mean_d > 0 else 0.0
    elif len(df) > 1:
        # Trade-level proxy (all_data yoksa)
        trades_per_year_est = len(df) * annual_trading_days / max(
            (pd.to_datetime(df['exit_date']).max() -
             pd.to_datetime(df['exit_date']).min()).days, 1
        )
        mean_ct = df['contrib_pct'].mean()
        std_ct = df['contrib_pct'].std()
        sharpe = float(mean_ct / std_ct * np.sqrt(trades_per_year_est)) if std_ct > 0 else 0.0
        neg = df.loc[df['contrib_pct'] < 0, 'contrib_pct']
        sortino = float(mean_ct / neg.std() * np.sqrt(trades_per_year_est)) if len(neg) > 1 and neg.std() > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Annualized return
    if len(equity_curve) > 1:
        start_date = pd.to_datetime(equity_curve.index[0])
        end_date = pd.to_datetime(equity_curve.index[-1])
        years = max((end_date - start_date).days / 365.25, 1e-9)
        cagr = float((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr = 0.0

    summary = {
        'n_trades': len(df),
        'win_rate': float(len(wins) / len(df)),
        'avg_return_pct': float(df['return_pct'].mean()),
        'med_return_pct': float(df['return_pct'].median()),
        'total_return_pct': total_ret,
        'cagr_pct': cagr,
        'final_equity': final_equity,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd_pct': max_dd,
        'profit_factor': float(total_win_tl / total_loss_tl) if total_loss_tl > 0 else float('inf'),
        'avg_winner_pct': float(wins['return_pct'].mean()) if len(wins) else 0.0,
        'avg_loser_pct': float(losses['return_pct'].mean()) if len(losses) else 0.0,
    }

    # XU100 benchmark varsa alpha hesapla
    if xu100_ret_series is not None and not xu100_ret_series.empty:
        start = pd.to_datetime(df['entry_date']).min()
        end = pd.to_datetime(df['exit_date']).max()
        xu_window = xu100_ret_series.loc[start:end]
        if len(xu_window) > 0:
            xu_total = float((xu_window + 1).prod() - 1) * 100.0
            summary['xu100_return_pct'] = xu_total
            summary['alpha_vs_xu100_pct'] = summary['total_return_pct'] - xu_total

            # Annualized XU100 CAGR + alpha_cagr
            xu_years = max((end - start).days / 365.25, 1e-9)
            xu_cagr = float((1 + xu_total / 100.0) ** (1.0 / xu_years) - 1.0) * 100.0
            summary['xu100_cagr_pct'] = xu_cagr
            summary['alpha_cagr_pct'] = cagr - xu_cagr

    return summary


def per_bucket_attribution(trades: list[dict]) -> pd.DataFrame:
    """Her bucket için ayrı WR/getiri tablosu."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    agg = df.groupby('bucket').apply(lambda g: pd.Series({
        'n': len(g),
        'win_rate': (g['return_pct'] > 0).mean(),
        'avg_return_pct': g['return_pct'].mean(),
        'total_pnl_pct': g['pnl_pct_capital'].sum(),
        'reason_breakdown': g['reason'].value_counts(normalize=True).to_dict(),
    })).reset_index()
    return agg


def per_regime_attribution(trades: list[dict],
                           regime_series: pd.Series) -> pd.DataFrame:
    """Her rejim bucket'ı (up/down, low/high vol) içinde performance."""
    if not trades or regime_series is None or regime_series.empty:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['regime'] = df['entry_date'].map(
        lambda d: regime_series.asof(d) if pd.notna(d) else None
    )
    if df['regime'].isna().all():
        return pd.DataFrame()
    agg = df.groupby('regime').apply(lambda g: pd.Series({
        'n': len(g),
        'win_rate': (g['return_pct'] > 0).mean(),
        'avg_return_pct': g['return_pct'].mean(),
    })).reset_index()
    return agg
