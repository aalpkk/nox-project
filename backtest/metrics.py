"""
NOX Backtest — Metrics
Trade listesinden performans metrikleri hesapla.
"""
import numpy as np
import pandas as pd
from backtest.config import FILTER_TESTS, REGIMES, REGIME_GROUPS, SIGNAL_GROUPS
from backtest.engine import apply_filter


def calc_metrics(trades):
    """Trade listesinden temel metrikler."""
    if not trades:
        return {
            'n_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'avg_rr_realized': 0, 'avg_hold_days': 0,
            'max_win': 0, 'max_loss': 0, 'avg_mae': 0, 'avg_mfe': 0,
            'tp_pct': 0, 'stop_pct': 0, 'timeout_pct': 0,
            'sharpe': 0, 'expectancy': 0,
        }

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(trades)
    win_rate = len(wins) / n * 100 if n > 0 else 0
    avg_pnl = np.mean(pnls)
    total_pnl = sum(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    hold_days = [t.get('hold_days', 0) for t in trades]
    maes = [t.get('mae', 0) for t in trades]
    mfes = [t.get('mfe', 0) for t in trades]

    exits = [t.get('exit_reason', '') for t in trades]
    tp_pct = exits.count('TP') / n * 100
    stop_pct = exits.count('STOP') / n * 100
    timeout_pct = exits.count('TIMEOUT') / n * 100

    # Sharpe (günlük PnL bazında yaklaşık)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252 / max(np.mean(hold_days), 1)) if std_pnl > 0 else 0

    # Expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    return {
        'n_trades': n,
        'win_rate': round(win_rate, 1),
        'avg_pnl': round(avg_pnl, 2),
        'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_rr_realized': round(avg_win / abs(avg_loss), 2) if avg_loss != 0 else 0,
        'avg_hold_days': round(np.mean(hold_days), 1),
        'max_win': round(max(pnls), 2),
        'max_loss': round(min(pnls), 2),
        'avg_mae': round(np.mean(maes), 2),
        'avg_mfe': round(np.mean(mfes), 2),
        'tp_pct': round(tp_pct, 1),
        'stop_pct': round(stop_pct, 1),
        'timeout_pct': round(timeout_pct, 1),
        'sharpe': round(sharpe, 2),
        'expectancy': round(expectancy, 2),
    }


def build_matrix(all_trades):
    """
    Filtre × Rejim × Sinyal matrisi oluştur.

    Returns:
        matrix: dict[filter_name][regime_group][signal_group] → metrics
        best_filter: en iyi filtre kombinasyonu (expectancy bazlı)
    """
    matrix = {}

    for filter_name, filter_cfg in FILTER_TESTS.items():
        matrix[filter_name] = {}

        # Filtre uygula
        filtered = [t for t in all_trades if apply_filter(t, filter_cfg)]

        # Her rejim grubu için
        for regime_group, regime_list in REGIME_GROUPS.items():
            matrix[filter_name][regime_group] = {}

            regime_trades = [t for t in filtered
                           if t.get('regime_period', 'unknown') in regime_list
                           or regime_group == 'all']

            # Her sinyal grubu için
            for sig_group, sig_list in SIGNAL_GROUPS.items():
                sig_trades = [t for t in regime_trades if t.get('signal', '') in sig_list]
                matrix[filter_name][regime_group][sig_group] = calc_metrics(sig_trades)

            # Tüm sinyaller birlikte
            matrix[filter_name][regime_group]['all_signals'] = calc_metrics(regime_trades)

    # En iyi filtre bul — PF × Sharpe skor, min %10 trade şartı
    total_n = len(all_trades)
    min_n = max(100, total_n * 0.10)  # en az %10 veya 100 trade
    best_filter = None
    best_score = -999
    for fname, fdata in matrix.items():
        m = fdata.get('all', {}).get('all_signals', {})
        n = m.get('n_trades', 0)
        pf = m.get('profit_factor', 0)
        sharpe = m.get('sharpe', 0)
        if n >= min_n and pf > 1.0:
            score = pf * sharpe  # PF × Sharpe composite
            if score > best_score:
                best_score = score
                best_filter = fname

    if best_filter is None:
        best_filter = 'baseline'

    return matrix, best_filter


def matrix_to_dataframe(matrix, signal_group='all_signals'):
    """
    Matrisi okunabilir DataFrame'e çevir.
    Rows: filtreler, Columns: rejim grupları × metrikler
    """
    rows = []
    for fname, fdata in matrix.items():
        row = {'filter': fname, 'desc': FILTER_TESTS[fname]['desc']}
        for rgroup in ['all', 'bull', 'bear', 'sideways']:
            m = fdata.get(rgroup, {}).get(signal_group, {})
            prefix = rgroup
            row[f'{prefix}_n'] = m.get('n_trades', 0)
            row[f'{prefix}_wr'] = m.get('win_rate', 0)
            row[f'{prefix}_pf'] = m.get('profit_factor', 0)
            row[f'{prefix}_exp'] = m.get('expectancy', 0)
            row[f'{prefix}_avg_pnl'] = m.get('avg_pnl', 0)
            row[f'{prefix}_sharpe'] = m.get('sharpe', 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def signal_breakdown(all_trades, filter_name='baseline'):
    """Her sinyal tipi için ayrı metrikler."""
    filter_cfg = FILTER_TESTS.get(filter_name, FILTER_TESTS['baseline'])
    filtered = [t for t in all_trades if apply_filter(t, filter_cfg)]

    breakdown = {}
    signal_types = set(t.get('signal', '') for t in filtered)
    for sig in sorted(signal_types):
        sig_trades = [t for t in filtered if t.get('signal', '') == sig]
        breakdown[sig] = calc_metrics(sig_trades)

    return breakdown


def regime_breakdown(all_trades, filter_name='balanced'):
    """Her rejim dönemi için ayrı metrikler."""
    filter_cfg = FILTER_TESTS.get(filter_name, FILTER_TESTS['baseline'])
    filtered = [t for t in all_trades if apply_filter(t, filter_cfg)]

    breakdown = {}
    for regime_name, regime_cfg in REGIMES.items():
        r_trades = [t for t in filtered if t.get('regime_period', '') == regime_name]
        if r_trades:
            m = calc_metrics(r_trades)
            m['label'] = regime_cfg['label']
            m['type'] = regime_cfg['type']
            m['period'] = f"{regime_cfg['start']} → {regime_cfg['end']}"
            breakdown[regime_name] = m

    return breakdown
