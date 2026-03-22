#!/usr/bin/env python3
"""
NOX Divergence — Portfolio Backtesti
======================================
Kural-spesifik holding politikalarıyla portföy simülasyonu.

3 exit stratejisi:
  - FIXED:    Sabit horizon çıkış (WR-optimal)
  - TRAILING: Min-hold + trailing stop + max hold
  - PARTIAL:  Partial TP (%50) + runner (C, D_BUY için)

Kullanım:
    python run_portfolio_divergence.py                        # Son CSV'den
    python run_portfolio_divergence.py --csv path/to/csv      # Belirli CSV
    python run_portfolio_divergence.py --capital 200000       # Farklı sermaye
    python run_portfolio_divergence.py --max-pos 5            # Max 5 eşzamanlı
"""

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from run_divergence import _classify_rule, RULE_DEFS, RULE_ORDER
from run_holding_analysis import _load_csv_signals, _to_lower_cols, GROUPS, GROUP_LABELS


# =============================================================================
# SABITLER
# =============================================================================

INITIAL_CAPITAL = 100_000
POSITION_SIZE_PCT = 10       # her pozisyon sermayenin %10'u
MAX_CONCURRENT = 10          # max eşzamanlı pozisyon
COMMISSION_PCT = 0.2         # çift yön komisyon (%)

# Kural-spesifik exit parametreleri
RULE_PARAMS = {
    'B':      {'fixed': 5,  'min_hold': 3, 'max_hold': 7,  'trail_start': 3, 'trail_pct': 3.0},
    'C':      {'fixed': 5,  'min_hold': 3, 'max_hold': 10, 'trail_start': 3, 'trail_pct': 3.0,
               'partial_at': 5, 'runner_max': 10},
    'D_BUY':  {'fixed': 10, 'min_hold': 3, 'max_hold': 10, 'trail_start': 3, 'trail_pct': 3.0},
    'D_SELL': {'fixed': 3,  'min_hold': 2, 'max_hold': 5,  'trail_start': 2, 'trail_pct': 2.5},
    'E':      {'fixed': 5,  'min_hold': 3, 'max_hold': 5,  'trail_start': 3, 'trail_pct': 3.0},
}

# Kural önceliği (entry sıralaması)
RULE_PRIORITY = {'B': 0, 'D': 1, 'C': 2, 'E': 3}

EXIT_STRATEGIES = ['FIXED', 'TRAILING', 'PARTIAL']

# C kuralı daraltma filtreleri
C_FILTER_CONFIGS = {
    'none': {},  # filtre yok
    'moderate': {    # OBV/CHOPPY/SETUP çıkar, MFI sadece FULL_TREND+RR>=1
        'exclude_div_types': ['OBV_HIDDEN'],
        'exclude_regimes': [0],       # CHOPPY
        'exclude_states': ['SETUP'],
        'mfi_filter': {'require_rr_gte': 1.0, 'require_regime_in': [3]},
    },
    'aggressive': {  # MFI_HIDDEN tamamen çıkar
        'exclude_div_types': ['OBV_HIDDEN', 'MFI_HIDDEN'],
        'exclude_regimes': [0],
        'exclude_states': ['SETUP'],
    },
}

# TRAILING parametre sweep grid
TRAIL_SWEEP_GRID = {
    'B':      {'trail_pct': [2.5, 3.0, 4.0, 5.0],      'trail_start': [3, 4, 5]},
    'C':      {'trail_pct': [3.0, 4.0, 5.0, 6.0],      'trail_start': [3, 4, 5]},
    'D_BUY':  {'trail_pct': [3.0, 4.0, 5.0, 6.0, 8.0], 'trail_start': [3, 5, 7]},
    'D_SELL': {'trail_pct': [None, 2.5, 3.0],           'trail_start': [2]},
    'E':      {'trail_pct': [2.5, 3.0, 4.0, 5.0],      'trail_start': [2, 3]},
}


# =============================================================================
# C FILTRE FONKSIYONLARI
# =============================================================================

def _filter_c_signals(signals, level):
    """C sinyallerini filtrele. Diğer kurallar geçer.
    Returns: (filtered_signals, stats_dict)
    """
    cfg = C_FILTER_CONFIGS.get(level, {})
    if not cfg:
        return signals, {'level': level, 'before': len(signals),
                         'after': len(signals), 'c_before': 0, 'c_after': 0,
                         'drop_reasons': {}}

    c_before = sum(1 for s in signals if s['group'] == 'C')
    filtered = []
    drop_reasons = defaultdict(int)

    for s in signals:
        if s['group'] != 'C':
            filtered.append(s)
            continue

        div_type = s.get('div_type', '')
        regime = s.get('regime', -1)
        if isinstance(regime, str):
            try:
                regime = int(regime)
            except (ValueError, TypeError):
                regime = -1
        state = s.get('state', '')
        rr = s.get('rr_ratio', 0) or s.get('rr', 0) or 0

        dropped = False

        # Exclude div types
        if div_type in cfg.get('exclude_div_types', []):
            drop_reasons[f'div_type={div_type}'] += 1
            dropped = True

        # Exclude regimes
        if not dropped and regime in cfg.get('exclude_regimes', []):
            drop_reasons[f'regime={regime}'] += 1
            dropped = True

        # Exclude states
        if not dropped and state in cfg.get('exclude_states', []):
            drop_reasons[f'state={state}'] += 1
            dropped = True

        # MFI özel filtre (moderate seviye)
        if not dropped and 'mfi_filter' in cfg and div_type == 'MFI_HIDDEN':
            mfi_cfg = cfg['mfi_filter']
            rr_ok = rr >= mfi_cfg.get('require_rr_gte', 0)
            regime_ok = regime in mfi_cfg.get('require_regime_in', [])
            if not (rr_ok and regime_ok):
                drop_reasons['MFI_filter'] += 1
                dropped = True

        if not dropped:
            filtered.append(s)

    c_after = sum(1 for s in filtered if s['group'] == 'C')
    stats = {
        'level': level,
        'before': len(signals),
        'after': len(filtered),
        'c_before': c_before,
        'c_after': c_after,
        'drop_reasons': dict(drop_reasons),
    }
    return filtered, stats


def _print_filter_stats(level, stats):
    """C filtre istatistiklerini yazdır."""
    dropped = stats['c_before'] - stats['c_after']
    print(f"\n  C Filtre [{level.upper()}]: "
          f"{stats['c_before']} → {stats['c_after']} C sinyal "
          f"({dropped} elendi, %{dropped / stats['c_before'] * 100:.0f})" if stats['c_before'] > 0
          else f"\n  C Filtre [{level.upper()}]: C sinyal yok")
    print(f"  Toplam: {stats['before']} → {stats['after']} sinyal")
    if stats['drop_reasons']:
        for reason, n in sorted(stats['drop_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: -{n}")


# =============================================================================
# VERI YAPILARI
# =============================================================================

@dataclass
class Position:
    ticker: str
    rule: str              # B/C/D/E
    group: str             # B/C/D_BUY/D_SELL/E
    direction: str         # BUY/SELL
    entry_date: object     # pd.Timestamp
    entry_price: float
    size: float            # TL
    shares: float          # lot (float for partial)
    trail_peak: float      # trailing stop peak
    partial_closed: bool = False
    original_shares: float = 0.0

    def __post_init__(self):
        if self.original_shares == 0.0:
            self.original_shares = self.shares


@dataclass
class ClosedTrade:
    ticker: str
    rule: str
    group: str
    direction: str
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_tl: float
    exit_reason: str       # fixed/trail/max_hold/partial
    hold_days: int
    shares: float
    mfe: float = 0.0      # max favorable excursion (%)
    mae: float = 0.0      # max adverse excursion (%)


@dataclass
class EquitySnapshot:
    date: object
    equity: float
    daily_return: float
    drawdown: float
    open_positions: int
    closed_today: int


# =============================================================================
# PORTFOY SIMULATORU
# =============================================================================

class PortfolioSimulator:
    def __init__(self, initial_capital, position_size_pct, max_concurrent,
                 commission_pct, exit_strategy, rule_params):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_concurrent = max_concurrent
        self.commission_pct = commission_pct
        self.exit_strategy = exit_strategy      # FIXED / TRAILING / PARTIAL
        self.rule_params = rule_params

        self.cash = initial_capital
        self.positions = []                     # list[Position]
        self.closed_trades = []                 # list[ClosedTrade]
        self.equity_curve = []                  # list[EquitySnapshot]
        self.skipped_count = 0                  # kapasite veya duplicate nedeniyle atlanan

        # MFE/MAE tracking: position_id -> {mfe, mae}
        self._excursions = {}

    def _get_params(self, group):
        return self.rule_params.get(group, self.rule_params.get('C'))

    def _calc_pnl_pct(self, direction, entry_price, exit_price):
        if direction == 'BUY':
            return (exit_price / entry_price - 1) * 100
        else:  # SELL
            return (1 - exit_price / entry_price) * 100

    def _update_excursion(self, pos, high, low, close):
        """MFE/MAE güncelle."""
        pid = id(pos)
        if pid not in self._excursions:
            self._excursions[pid] = {'mfe': 0.0, 'mae': 0.0}

        if pos.direction == 'BUY':
            fav = (high / pos.entry_price - 1) * 100
            adv = (1 - low / pos.entry_price) * 100
        else:
            fav = (1 - low / pos.entry_price) * 100
            adv = (high / pos.entry_price - 1) * 100

        self._excursions[pid]['mfe'] = max(self._excursions[pid]['mfe'], fav)
        self._excursions[pid]['mae'] = max(self._excursions[pid]['mae'], adv)

    def _get_excursion(self, pos):
        pid = id(pos)
        ex = self._excursions.pop(pid, {'mfe': 0.0, 'mae': 0.0})
        return ex['mfe'], ex['mae']

    def _close_position(self, pos, exit_date, exit_price, exit_reason, shares=None):
        """Pozisyonu kapat, ClosedTrade oluştur."""
        if shares is None:
            shares = pos.shares

        pnl_pct_raw = self._calc_pnl_pct(pos.direction, pos.entry_price, exit_price)
        commission = self.commission_pct * 2  # çift yön
        pnl_pct = pnl_pct_raw - commission

        pnl_tl = shares * pos.entry_price * pnl_pct / 100
        self.cash += shares * exit_price  # satış geliri

        hold_days = (exit_date - pos.entry_date).days
        # iş günü yaklaşımı (5/7)
        hold_bdays = max(1, int(hold_days * 5 / 7))

        mfe, mae = 0.0, 0.0
        if shares == pos.shares:
            # tam çıkış — excursion al
            mfe, mae = self._get_excursion(pos)

        trade = ClosedTrade(
            ticker=pos.ticker,
            rule=pos.rule,
            group=pos.group,
            direction=pos.direction,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=round(exit_price, 4),
            pnl_pct=round(pnl_pct, 4),
            pnl_tl=round(pnl_tl, 2),
            exit_reason=exit_reason,
            hold_days=hold_bdays,
            shares=shares,
            mfe=round(mfe, 4),
            mae=round(mae, 4),
        )
        self.closed_trades.append(trade)
        return trade

    def check_exits(self, current_date, price_data):
        """Açık pozisyonları kontrol et, çıkış koşullarını uygula."""
        closed_today = []
        remaining = []

        for pos in self.positions:
            ticker = pos.ticker
            if ticker not in price_data:
                remaining.append(pos)
                continue

            df = price_data[ticker]
            # current_date'e en yakın bar
            mask = df.index <= current_date
            if not mask.any():
                remaining.append(pos)
                continue

            idx = mask.sum() - 1
            close_price = df['close'].iloc[idx]
            high_price = df['high'].iloc[idx]
            low_price = df['low'].iloc[idx]
            actual_date = df.index[idx]

            hold_days = (actual_date - pos.entry_date).days
            hold_bdays = max(0, int(hold_days * 5 / 7))

            # MFE/MAE güncelle
            self._update_excursion(pos, high_price, low_price, close_price)

            params = self._get_params(pos.group)
            closed = False

            if self.exit_strategy == 'FIXED':
                if hold_bdays >= params['fixed']:
                    self._close_position(pos, actual_date, close_price, 'fixed')
                    closed_today.append(pos)
                    closed = True

            elif self.exit_strategy == 'TRAILING':
                # trail_disabled → fixed horizon kullan
                if params.get('trail_disabled'):
                    if hold_bdays >= params['fixed']:
                        self._close_position(pos, actual_date, close_price, 'fixed_override')
                        closed_today.append(pos)
                        closed = True
                # Max hold zorla çıkış
                elif hold_bdays >= params['max_hold']:
                    self._close_position(pos, actual_date, close_price, 'max_hold')
                    closed_today.append(pos)
                    closed = True
                # Trailing stop (min_hold sonrası)
                elif hold_bdays >= params['trail_start']:
                    # Peak güncelle
                    if pos.direction == 'BUY':
                        pos.trail_peak = max(pos.trail_peak, high_price)
                        stop_price = pos.trail_peak * (1 - params['trail_pct'] / 100)
                        if close_price < stop_price:
                            self._close_position(pos, actual_date, close_price, 'trail')
                            closed_today.append(pos)
                            closed = True
                    else:  # SELL
                        pos.trail_peak = min(pos.trail_peak, low_price)
                        stop_price = pos.trail_peak * (1 + params['trail_pct'] / 100)
                        if close_price > stop_price:
                            self._close_position(pos, actual_date, close_price, 'trail')
                            closed_today.append(pos)
                            closed = True

            elif self.exit_strategy == 'PARTIAL':
                partial_at = params.get('partial_at')
                runner_max = params.get('runner_max')

                # Bu kural partial desteklemiyor → TRAILING gibi davran
                if partial_at is None:
                    # trail_disabled → fixed horizon kullan
                    if params.get('trail_disabled'):
                        if hold_bdays >= params['fixed']:
                            self._close_position(pos, actual_date, close_price, 'fixed_override')
                            closed_today.append(pos)
                            closed = True
                    # Max hold
                    elif hold_bdays >= params['max_hold']:
                        self._close_position(pos, actual_date, close_price, 'max_hold')
                        closed_today.append(pos)
                        closed = True
                    elif hold_bdays >= params['trail_start']:
                        if pos.direction == 'BUY':
                            pos.trail_peak = max(pos.trail_peak, high_price)
                            stop_price = pos.trail_peak * (1 - params['trail_pct'] / 100)
                            if close_price < stop_price:
                                self._close_position(pos, actual_date, close_price, 'trail')
                                closed_today.append(pos)
                                closed = True
                        else:
                            pos.trail_peak = min(pos.trail_peak, low_price)
                            stop_price = pos.trail_peak * (1 + params['trail_pct'] / 100)
                            if close_price > stop_price:
                                self._close_position(pos, actual_date, close_price, 'trail')
                                closed_today.append(pos)
                                closed = True
                else:
                    # Partial TP: partial_at günde yarısını kapat
                    if hold_bdays >= partial_at and not pos.partial_closed:
                        partial_shares = pos.shares / 2
                        self._close_position(pos, actual_date, close_price, 'partial',
                                             shares=partial_shares)
                        pos.shares -= partial_shares
                        pos.partial_closed = True
                        # Runner devam ediyor, trail peak reset
                        if pos.direction == 'BUY':
                            pos.trail_peak = high_price
                        else:
                            pos.trail_peak = low_price

                    # Runner max hold
                    if pos.partial_closed and hold_bdays >= runner_max:
                        self._close_position(pos, actual_date, close_price, 'runner_max')
                        closed_today.append(pos)
                        closed = True
                    # Runner trailing (partial sonrası)
                    elif pos.partial_closed and hold_bdays >= partial_at:
                        if pos.direction == 'BUY':
                            pos.trail_peak = max(pos.trail_peak, high_price)
                            stop_price = pos.trail_peak * (1 - params['trail_pct'] / 100)
                            if close_price < stop_price:
                                self._close_position(pos, actual_date, close_price, 'runner_trail')
                                closed_today.append(pos)
                                closed = True
                        else:
                            pos.trail_peak = min(pos.trail_peak, low_price)
                            stop_price = pos.trail_peak * (1 + params['trail_pct'] / 100)
                            if close_price > stop_price:
                                self._close_position(pos, actual_date, close_price, 'runner_trail')
                                closed_today.append(pos)
                                closed = True
                    # Partial öncesi max hold (henüz partial yapılmamış ama max_hold dolmuş)
                    elif not pos.partial_closed and hold_bdays >= params['max_hold']:
                        self._close_position(pos, actual_date, close_price, 'max_hold')
                        closed_today.append(pos)
                        closed = True

            if not closed:
                remaining.append(pos)

        self.positions = remaining
        return len(closed_today)

    def try_open(self, signal, price_data, current_date):
        """Yeni pozisyon aç (kapasite ve duplicate kontrolü)."""
        ticker = signal['ticker']

        # Kapasite kontrolü
        if len(self.positions) >= self.max_concurrent:
            self.skipped_count += 1
            return False

        # Aynı ticker zaten açıksa skip
        if any(p.ticker == ticker for p in self.positions):
            self.skipped_count += 1
            return False

        if ticker not in price_data:
            self.skipped_count += 1
            return False

        df = price_data[ticker]
        # entry_date'e en yakın bar (sonraki iş günü)
        entry_date_ts = pd.Timestamp(str(signal['entry_date'])[:10])
        idx_locs = df.index.searchsorted(entry_date_ts)
        if idx_locs >= len(df):
            self.skipped_count += 1
            return False

        bar_idx = idx_locs
        entry_price = df['close'].iloc[bar_idx]
        actual_entry_date = df.index[bar_idx]

        if entry_price <= 0:
            self.skipped_count += 1
            return False

        # Pozisyon büyüklüğü: mevcut equity'nin %X'i
        equity = self._calc_equity(price_data, current_date)
        pos_size = equity * self.position_size_pct / 100
        shares = pos_size / entry_price

        if shares <= 0 or pos_size < 100:
            self.skipped_count += 1
            return False

        # Cash'ten düş
        cost = shares * entry_price
        self.cash -= cost

        # Trail peak başlangıcı
        direction = signal['direction']
        if direction == 'BUY':
            trail_peak = entry_price
        else:
            trail_peak = entry_price

        pos = Position(
            ticker=ticker,
            rule=signal['rule'],
            group=signal['group'],
            direction=direction,
            entry_date=actual_entry_date,
            entry_price=entry_price,
            size=pos_size,
            shares=shares,
            trail_peak=trail_peak,
        )
        self.positions.append(pos)
        return True

    def _calc_equity(self, price_data, current_date):
        """Mevcut equity hesapla (cash + açık pozisyon MTM)."""
        equity = self.cash
        for pos in self.positions:
            ticker = pos.ticker
            if ticker not in price_data:
                equity += pos.shares * pos.entry_price
                continue

            df = price_data[ticker]
            mask = df.index <= current_date
            if not mask.any():
                equity += pos.shares * pos.entry_price
                continue

            close_price = df['close'].iloc[mask.sum() - 1]

            if pos.direction == 'BUY':
                mtm = pos.shares * close_price
            else:
                # Short: 2 × entry - current (short P&L)
                mtm = pos.shares * (2 * pos.entry_price - close_price)

            equity += mtm
        return equity

    def daily_snapshot(self, current_date, price_data, closed_today_count):
        """Günlük equity snapshot."""
        equity = self._calc_equity(price_data, current_date)

        # Daily return
        if self.equity_curve:
            prev_eq = self.equity_curve[-1].equity
            daily_ret = (equity / prev_eq - 1) * 100 if prev_eq > 0 else 0.0
        else:
            daily_ret = (equity / self.initial_capital - 1) * 100

        # Drawdown
        peak_equity = max(
            self.initial_capital,
            max((s.equity for s in self.equity_curve), default=self.initial_capital)
        )
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0

        snap = EquitySnapshot(
            date=current_date,
            equity=round(equity, 2),
            daily_return=round(daily_ret, 4),
            drawdown=round(dd, 4),
            open_positions=len(self.positions),
            closed_today=closed_today_count,
        )
        self.equity_curve.append(snap)
        return snap

    def run(self, signals, price_data):
        """Tam simülasyon çalıştır."""
        # Sinyalleri entry_date sırala
        signals_sorted = sorted(signals, key=lambda s: str(s['entry_date'])[:10])

        # Tüm tarihleri topla (sinyallerden + fiyat verisinden)
        all_dates = set()
        for s in signals_sorted:
            all_dates.add(pd.Timestamp(str(s['entry_date'])[:10]))

        for ticker, df in price_data.items():
            for dt in df.index:
                all_dates.add(dt)

        trading_days = sorted(all_dates)
        if not trading_days:
            return

        # Sinyal index: tarih -> sinyaller
        sig_by_date = defaultdict(list)
        for s in signals_sorted:
            dt = pd.Timestamp(str(s['entry_date'])[:10])
            sig_by_date[dt].append(s)

        # Günlük döngü
        for day in trading_days:
            # 1. EXIT: Açık pozisyonları kontrol
            closed_count = self.check_exits(day, price_data)

            # 2. ENTRY: Yeni sinyaller
            day_signals = sig_by_date.get(day, [])
            if day_signals:
                # Kural önceliğine göre sırala
                day_signals.sort(key=lambda s: (
                    RULE_PRIORITY.get(s['rule'], 9),
                    -s.get('quality', 0)
                ))
                for sig in day_signals:
                    self.try_open(sig, price_data, day)

            # 3. SNAPSHOT
            if self.positions or self.closed_trades or self.equity_curve:
                self.daily_snapshot(day, price_data, closed_count)

        # Simülasyon sonu: kalan açık pozisyonları son fiyattan kapat
        if self.positions:
            last_day = trading_days[-1]
            for pos in list(self.positions):
                ticker = pos.ticker
                if ticker in price_data:
                    df = price_data[ticker]
                    close_price = df['close'].iloc[-1]
                    self._close_position(pos, last_day, close_price, 'end_of_sim')
            self.positions = []


# =============================================================================
# METRIKLERI HESAPLAMA
# =============================================================================

def _calc_strategy_metrics(trades):
    """Trade listesinden strateji metrikleri."""
    if not trades:
        return {
            'n_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'med_pnl': 0,
            'profit_factor': 0, 'sharpe': 0, 'max_dd': 0, 'avg_hold': 0,
            'total_pnl_tl': 0,
        }

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(trades)
    wr = len(wins) / n * 100

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    holds = [t.hold_days for t in trades]
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1
    avg_hold = np.mean(holds) if holds else 1
    sharpe = (np.mean(pnls) / std_pnl) * np.sqrt(252 / max(avg_hold, 1)) if std_pnl > 0 else 0

    total_tl = sum(t.pnl_tl for t in trades)

    return {
        'n_trades': n,
        'win_rate': round(wr, 1),
        'avg_pnl': round(np.mean(pnls), 2),
        'med_pnl': round(np.median(pnls), 2),
        'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 2),
        'avg_hold': round(avg_hold, 1),
        'total_pnl_tl': round(total_tl, 0),
    }


def _calc_equity_stats(equity_curve, initial_capital):
    """Equity curve istatistikleri."""
    if not equity_curve:
        return {}

    equities = [s.equity for s in equity_curve]
    daily_rets = [s.daily_return for s in equity_curve if s.daily_return != 0]

    final_eq = equities[-1]
    total_ret = (final_eq / initial_capital - 1) * 100

    # Yıllık getiri (252 iş günü)
    n_days = len(equity_curve)
    if n_days > 1:
        annual_ret = ((final_eq / initial_capital) ** (252 / n_days) - 1) * 100
    else:
        annual_ret = 0

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    dds = []
    for eq in equities:
        peak = max(peak, eq)
        dd = (eq / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd
        dds.append(dd)

    avg_dd = np.mean([d for d in dds if d < 0]) if any(d < 0 for d in dds) else 0

    # Kârlı gün oranı
    profitable_days = sum(1 for r in daily_rets if r > 0)
    profitable_pct = profitable_days / len(daily_rets) * 100 if daily_rets else 0

    return {
        'final_equity': round(final_eq, 0),
        'total_return': round(total_ret, 1),
        'annual_return': round(annual_ret, 1),
        'max_drawdown': round(max_dd, 1),
        'avg_drawdown': round(avg_dd, 1),
        'profitable_day_pct': round(profitable_pct, 1),
        'n_days': n_days,
    }


# =============================================================================
# KONSOL CIKTI
# =============================================================================

def _print_strategy_comparison(results):
    """Strateji karşılaştırma tablosu."""
    strats = [s for s in EXIT_STRATEGIES if s in results]
    w = 100
    print(f"\n{'═' * w}")
    print(f"  STRATEJI KARSILASTIRMA")
    print(f"{'═' * w}")

    header = f"  {'':>20}"
    for strat in strats:
        header += f"  {strat:>14}"
    print(header)
    print(f"  {'─' * 20}" + "  ".join(f"{'─' * 14}" for _ in strats))

    metrics_rows = [
        ('N trades', 'n_trades', '{:>14d}'),
        ('Win Rate', 'win_rate', '{:>13.1f}%'),
        ('Avg PnL', 'avg_pnl', '{:>+13.2f}%'),
        ('Med PnL', 'med_pnl', '{:>+13.2f}%'),
        ('Profit Factor', 'profit_factor', '{:>14.2f}'),
        ('Sharpe (ann)', 'sharpe', '{:>14.2f}'),
        ('Avg Hold Days', 'avg_hold', '{:>14.1f}'),
        ('Total PnL (TL)', 'total_pnl_tl', '{:>13,.0f}₺'),
    ]

    for label, key, fmt in metrics_rows:
        line = f"  {label:>20}"
        for strat in strats:
            m = results[strat]['metrics']
            val = m.get(key, 0)
            line += f"  {fmt.format(val)}"
        print(line)

    # Equity stats
    print()
    eq_rows = [
        ('Final Equity', 'final_equity', '{:>13,.0f}₺'),
        ('Total Return', 'total_return', '{:>+13.1f}%'),
        ('Annual Return', 'annual_return', '{:>+13.1f}%'),
        ('Max Drawdown', 'max_drawdown', '{:>13.1f}%'),
        ('Avg Drawdown', 'avg_drawdown', '{:>13.1f}%'),
        ('Profitable Days', 'profitable_day_pct', '{:>13.1f}%'),
    ]

    for label, key, fmt in eq_rows:
        line = f"  {label:>20}"
        for strat in strats:
            eq = results[strat]['equity_stats']
            val = eq.get(key, 0)
            line += f"  {fmt.format(val)}"
        print(line)

    print(f"{'═' * w}")


def _print_rule_breakdown(results):
    """Kural × Strateji cross-tab."""
    strats = [s for s in EXIT_STRATEGIES if s in results]
    w = 100
    print(f"\n{'═' * w}")
    print(f"  KURAL x STRATEJI KIRILIM")
    print(f"{'═' * w}")

    for strat in strats:
        trades = results[strat]['trades']
        print(f"\n  ◆ {strat}")
        print(f"  {'Grup':>25}  {'N':>5}  {'WR':>7}  {'Avg':>8}  {'Med':>8}  {'PF':>6}  {'Hold':>5}  {'TotalTL':>10}")
        print(f"  {'─' * 25}  {'─' * 5}  {'─' * 7}  {'─' * 8}  {'─' * 8}  {'─' * 6}  {'─' * 5}  {'─' * 10}")

        for g in GROUPS:
            g_trades = [t for t in trades if t.group == g]
            if not g_trades:
                continue
            m = _calc_strategy_metrics(g_trades)
            label = GROUP_LABELS.get(g, g)
            if len(label) > 25:
                label = label[:25]
            total_tl = sum(t.pnl_tl for t in g_trades)
            print(f"  {label:>25}  {m['n_trades']:>5}  {m['win_rate']:>6.1f}%  "
                  f"{m['avg_pnl']:>+7.2f}%  {m['med_pnl']:>+7.2f}%  {m['profit_factor']:>6.2f}  "
                  f"{m['avg_hold']:>5.1f}  {total_tl:>+10,.0f}")

        # Toplam
        m = results[strat]['metrics']
        total_tl = sum(t.pnl_tl for t in trades)
        print(f"  {'TOPLAM':>25}  {m['n_trades']:>5}  {m['win_rate']:>6.1f}%  "
              f"{m['avg_pnl']:>+7.2f}%  {m['med_pnl']:>+7.2f}%  {m['profit_factor']:>6.2f}  "
              f"{m['avg_hold']:>5.1f}  {total_tl:>+10,.0f}")

    print(f"\n{'═' * w}")


def _print_exit_reason_breakdown(results):
    """Çıkış nedeni dağılımı."""
    strats = [s for s in EXIT_STRATEGIES if s in results]
    w = 100
    print(f"\n{'═' * w}")
    print(f"  CIKIS NEDENI DAGILIMI")
    print(f"{'═' * w}")

    for strat in strats:
        trades = results[strat]['trades']
        reason_counts = defaultdict(int)
        reason_pnls = defaultdict(list)
        for t in trades:
            reason_counts[t.exit_reason] += 1
            reason_pnls[t.exit_reason].append(t.pnl_pct)

        print(f"\n  ◆ {strat}")
        print(f"  {'Neden':>15}  {'N':>5}  {'%':>6}  {'Avg PnL':>9}  {'WR':>6}")
        print(f"  {'─' * 15}  {'─' * 5}  {'─' * 6}  {'─' * 9}  {'─' * 6}")

        for reason in sorted(reason_counts.keys()):
            n = reason_counts[reason]
            pct = n / len(trades) * 100
            pnls = reason_pnls[reason]
            avg_p = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"  {reason:>15}  {n:>5}  {pct:>5.1f}%  {avg_p:>+8.2f}%  {wr:>5.1f}%")

    print(f"\n{'═' * w}")


def _print_capacity_info(results):
    """Kapasite bilgisi."""
    strats = [s for s in EXIT_STRATEGIES if s in results]
    print(f"\n  KAPASITE BILGISI:")
    for strat in strats:
        sim = results[strat]['simulator']
        eq_curve = sim.equity_curve
        max_open = max((s.open_positions for s in eq_curve), default=0)
        avg_open = np.mean([s.open_positions for s in eq_curve]) if eq_curve else 0
        print(f"    {strat:>10}: Atlanan={sim.skipped_count}  "
              f"Max acik={max_open}  Ort acik={avg_open:.1f}")


# =============================================================================
# CSV KAYIT
# =============================================================================

def _save_trade_csv(results, output_dir):
    """Trade listelerini CSV'ye kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    for strat in [s for s in EXIT_STRATEGIES if s in results]:
        trades = results[strat]['trades']
        if not trades:
            continue

        rows = []
        for t in trades:
            rows.append({
                'ticker': t.ticker,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'rule': t.rule,
                'group': t.group,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct,
                'pnl_tl': t.pnl_tl,
                'exit_reason': t.exit_reason,
                'hold_days': t.hold_days,
                'shares': round(t.shares, 2),
                'mfe': t.mfe,
                'mae': t.mae,
            })

        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f'portfolio_divergence_{strat.lower()}_{date_str}.csv')
        df.to_csv(path, index=False)
        print(f"  Trade CSV: {path} ({len(rows)} trade)")


def _save_equity_csv(results, output_dir):
    """Equity curve CSV'ye kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    for strat in [s for s in EXIT_STRATEGIES if s in results]:
        curve = results[strat]['simulator'].equity_curve
        if not curve:
            continue

        rows = []
        for s in curve:
            rows.append({
                'date': s.date,
                'equity': s.equity,
                'daily_return': s.daily_return,
                'drawdown': s.drawdown,
                'open_positions': s.open_positions,
                'closed_today': s.closed_today,
            })

        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f'portfolio_equity_{strat.lower()}_{date_str}.csv')
        df.to_csv(path, index=False)
        print(f"  Equity CSV: {path} ({len(rows)} gün)")


# =============================================================================
# TRAILING PARAMETRE SWEEP
# =============================================================================


def _build_sweep_params(rule_group, trail_pct, trail_start):
    """RULE_PARAMS kopyası oluştur, hedef kuralın trail parametrelerini değiştir."""
    params = copy.deepcopy(RULE_PARAMS)
    if trail_pct is None:
        # Trail devre dışı → fixed horizon kullan
        params[rule_group]['trail_disabled'] = True
    else:
        params[rule_group]['trail_pct'] = trail_pct
        params[rule_group]['trail_start'] = trail_start
        params[rule_group].pop('trail_disabled', None)
    return params


def _run_single_sweep(signals, price_data, rule_group, trail_pct, trail_start, args):
    """Tek combo için TRAILING simülasyonu çalıştır.
    Returns: dict with metrics + exit_reason breakdown
    """
    rule_params = _build_sweep_params(rule_group, trail_pct, trail_start)

    sim = PortfolioSimulator(
        initial_capital=args.capital,
        position_size_pct=args.pos_size,
        max_concurrent=args.max_pos,
        commission_pct=args.commission,
        exit_strategy='TRAILING',
        rule_params=rule_params,
    )
    sim.run(signals, price_data)

    # Kural bazlı metrikleri hesapla
    rule_trades = [t for t in sim.closed_trades if t.group == rule_group]
    all_metrics = _calc_strategy_metrics(sim.closed_trades)
    rule_metrics = _calc_strategy_metrics(rule_trades)
    eq_stats = _calc_equity_stats(sim.equity_curve, args.capital)

    # Exit reason breakdown (sadece hedef kural)
    trail_trades = [t for t in rule_trades if t.exit_reason == 'trail']
    maxhold_trades = [t for t in rule_trades if t.exit_reason == 'max_hold']
    fixed_trades = [t for t in rule_trades if t.exit_reason == 'fixed_override']

    trail_n = len(trail_trades)
    trail_wr = (sum(1 for t in trail_trades if t.pnl_pct > 0) / trail_n * 100) if trail_n > 0 else 0
    maxhold_n = len(maxhold_trades)
    maxhold_wr = (sum(1 for t in maxhold_trades if t.pnl_pct > 0) / maxhold_n * 100) if maxhold_n > 0 else 0
    fixed_n = len(fixed_trades)
    fixed_wr = (sum(1 for t in fixed_trades if t.pnl_pct > 0) / fixed_n * 100) if fixed_n > 0 else 0

    return {
        'trail_pct': trail_pct,
        'trail_start': trail_start,
        'rule_n': rule_metrics['n_trades'],
        'rule_wr': rule_metrics['win_rate'],
        'rule_avg': rule_metrics['avg_pnl'],
        'rule_pf': rule_metrics['profit_factor'],
        'rule_tl': rule_metrics['total_pnl_tl'],
        'all_n': all_metrics['n_trades'],
        'all_wr': all_metrics['win_rate'],
        'all_pf': all_metrics['profit_factor'],
        'all_sharpe': all_metrics['sharpe'],
        'all_tl': all_metrics['total_pnl_tl'],
        'final_equity': eq_stats.get('final_equity', 0),
        'max_dd': eq_stats.get('max_drawdown', 0),
        'trail_n': trail_n,
        'trail_wr': round(trail_wr, 1),
        'maxhold_n': maxhold_n,
        'maxhold_wr': round(maxhold_wr, 1),
        'fixed_n': fixed_n,
        'fixed_wr': round(fixed_wr, 1),
    }


def _print_sweep_table(rule_group, results):
    """Tek kural için sweep sonuç tablosu."""
    w = 130
    print(f"\n  {'─' * w}")
    print(f"  ◆ {rule_group} — Parametre Sweep ({len(results)} combo)")
    print(f"  {'─' * w}")
    print(f"  {'Trail%':>7} {'Start':>5} │ {'N':>4} {'WR':>6} {'Avg':>7} {'PF':>5} "
          f"{'RuleTL':>9} │ {'AllWR':>6} {'AllPF':>5} {'Sharpe':>6} {'Equity':>9} "
          f"│ {'Tr_N':>4} {'Tr_WR':>5} {'MH_N':>4} {'MH_WR':>5} {'Fx_N':>4} {'Fx_WR':>5}")
    print(f"  {'─' * 7} {'─' * 5} {'─' * 1} {'─' * 4} {'─' * 6} {'─' * 7} {'─' * 5} "
          f"{'─' * 9} {'─' * 1} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 9} "
          f"{'─' * 1} {'─' * 4} {'─' * 5} {'─' * 4} {'─' * 5} {'─' * 4} {'─' * 5}")

    for r in results:
        tp = f"{r['trail_pct']:.1f}" if r['trail_pct'] is not None else "OFF"
        print(f"  {tp:>7} {r['trail_start']:>5} │ "
              f"{r['rule_n']:>4} {r['rule_wr']:>5.1f}% {r['rule_avg']:>+6.2f}% {r['rule_pf']:>5.2f} "
              f"{r['rule_tl']:>+9,.0f} │ "
              f"{r['all_wr']:>5.1f}% {r['all_pf']:>5.2f} {r['all_sharpe']:>6.2f} "
              f"{r['final_equity']:>9,.0f} │ "
              f"{r['trail_n']:>4} {r['trail_wr']:>4.0f}% "
              f"{r['maxhold_n']:>4} {r['maxhold_wr']:>4.0f}% "
              f"{r['fixed_n']:>4} {r['fixed_wr']:>4.0f}%")


def _pick_best(results):
    """Composite skor ile en iyi combo seç.
    Score = 0.4×WR + 0.3×PF + 0.3×Sharpe (normalize edilmiş)
    """
    if not results:
        return None

    # Normalize
    wrs = [r['all_wr'] for r in results]
    pfs = [r['all_pf'] for r in results]
    sharpes = [r['all_sharpe'] for r in results]

    wr_min, wr_max = min(wrs), max(wrs)
    pf_min, pf_max = min(pfs), max(pfs)
    sh_min, sh_max = min(sharpes), max(sharpes)

    wr_range = wr_max - wr_min if wr_max != wr_min else 1
    pf_range = pf_max - pf_min if pf_max != pf_min else 1
    sh_range = sh_max - sh_min if sh_max != sh_min else 1

    best_score = -1
    best_r = results[0]
    for r in results:
        n_wr = (r['all_wr'] - wr_min) / wr_range
        n_pf = (r['all_pf'] - pf_min) / pf_range
        n_sh = (r['all_sharpe'] - sh_min) / sh_range
        score = 0.4 * n_wr + 0.3 * n_pf + 0.3 * n_sh
        if score > best_score:
            best_score = score
            best_r = r

    return best_r


def _run_parameter_sweep(signals, price_data, args):
    """Per-rule trailing parametre sweep → best pick → combined final."""
    w = 130
    print(f"\n{'═' * w}")
    print(f"  TRAILING PARAMETRE SWEEP")
    print(f"{'═' * w}")

    all_sweep_results = {}
    best_params = {}

    # Baseline: mevcut default parametreler
    print(f"\n  Baseline (mevcut defaults) hesaplanıyor...")
    baseline_sim = PortfolioSimulator(
        initial_capital=args.capital,
        position_size_pct=args.pos_size,
        max_concurrent=args.max_pos,
        commission_pct=args.commission,
        exit_strategy='TRAILING',
        rule_params=RULE_PARAMS,
    )
    baseline_sim.run(signals, price_data)
    baseline_metrics = _calc_strategy_metrics(baseline_sim.closed_trades)
    baseline_eq = _calc_equity_stats(baseline_sim.equity_curve, args.capital)
    print(f"    Baseline: WR {baseline_metrics['win_rate']:.1f}%, "
          f"PF {baseline_metrics['profit_factor']:.2f}, "
          f"Sharpe {baseline_metrics['sharpe']:.2f}, "
          f"Equity {baseline_eq.get('final_equity', 0):,.0f}")

    # Per-rule sweep
    for rule_group in ['B', 'C', 'D_BUY', 'D_SELL', 'E']:
        grid = TRAIL_SWEEP_GRID.get(rule_group, {})
        trail_pcts = grid.get('trail_pct', [3.0])
        trail_starts = grid.get('trail_start', [3])

        combos = [(tp, ts) for tp in trail_pcts for ts in trail_starts]
        print(f"\n  {rule_group}: {len(combos)} combo test ediliyor...")

        results = []
        for tp, ts in combos:
            r = _run_single_sweep(signals, price_data, rule_group, tp, ts, args)
            results.append(r)

        all_sweep_results[rule_group] = results
        _print_sweep_table(rule_group, results)

        # En iyiyi seç
        best = _pick_best(results)
        if best:
            best_params[rule_group] = {
                'trail_pct': best['trail_pct'],
                'trail_start': best['trail_start'],
            }
            tp_str = f"{best['trail_pct']:.1f}%" if best['trail_pct'] is not None else "OFF"
            print(f"  → EN IYI: trail={tp_str}, start={best['trail_start']} "
                  f"(WR {best['all_wr']:.1f}%, PF {best['all_pf']:.2f}, "
                  f"Sharpe {best['all_sharpe']:.2f})")

    # Combined final: tüm kuralların en iyi parametreleriyle
    print(f"\n{'─' * w}")
    print(f"  COMBINED FINAL — Tüm kuralların en iyi parametreleriyle")
    print(f"{'─' * w}")

    optimized_params = copy.deepcopy(RULE_PARAMS)
    for rule_group, bp in best_params.items():
        if bp['trail_pct'] is None:
            optimized_params[rule_group]['trail_disabled'] = True
        else:
            optimized_params[rule_group]['trail_pct'] = bp['trail_pct']
            optimized_params[rule_group]['trail_start'] = bp['trail_start']
            optimized_params[rule_group].pop('trail_disabled', None)

    # Parametre özeti
    print(f"\n  Optimize edilmiş parametreler:")
    for rule_group in ['B', 'C', 'D_BUY', 'D_SELL', 'E']:
        p = optimized_params[rule_group]
        if p.get('trail_disabled'):
            print(f"    {rule_group:>7}: trail=OFF (fixed={p['fixed']}G)")
        else:
            print(f"    {rule_group:>7}: trail={p['trail_pct']:.1f}%, start={p['trail_start']}")

    final_sim = PortfolioSimulator(
        initial_capital=args.capital,
        position_size_pct=args.pos_size,
        max_concurrent=args.max_pos,
        commission_pct=args.commission,
        exit_strategy='TRAILING',
        rule_params=optimized_params,
    )
    final_sim.run(signals, price_data)
    final_metrics = _calc_strategy_metrics(final_sim.closed_trades)
    final_eq = _calc_equity_stats(final_sim.equity_curve, args.capital)

    # Baseline vs Optimized karşılaştırma
    print(f"\n  {'':>20}  {'BASELINE':>14}  {'OPTIMIZED':>14}  {'DELTA':>10}")
    print(f"  {'─' * 20}  {'─' * 14}  {'─' * 14}  {'─' * 10}")

    rows = [
        ('N trades', baseline_metrics['n_trades'], final_metrics['n_trades'], 'd'),
        ('Win Rate', baseline_metrics['win_rate'], final_metrics['win_rate'], '%'),
        ('Avg PnL', baseline_metrics['avg_pnl'], final_metrics['avg_pnl'], '%'),
        ('Profit Factor', baseline_metrics['profit_factor'], final_metrics['profit_factor'], 'f'),
        ('Sharpe', baseline_metrics['sharpe'], final_metrics['sharpe'], 'f'),
        ('Total TL', baseline_metrics['total_pnl_tl'], final_metrics['total_pnl_tl'], 'tl'),
        ('Final Equity', baseline_eq.get('final_equity', 0), final_eq.get('final_equity', 0), 'tl'),
        ('Max DD', baseline_eq.get('max_drawdown', 0), final_eq.get('max_drawdown', 0), '%'),
    ]

    for label, base, opt, fmt in rows:
        delta = opt - base
        if fmt == 'd':
            print(f"  {label:>20}  {base:>14d}  {opt:>14d}  {delta:>+10d}")
        elif fmt == '%':
            print(f"  {label:>20}  {base:>13.1f}%  {opt:>13.1f}%  {delta:>+9.1f}%")
        elif fmt == 'f':
            print(f"  {label:>20}  {base:>14.2f}  {opt:>14.2f}  {delta:>+10.2f}")
        elif fmt == 'tl':
            print(f"  {label:>20}  {base:>13,.0f}₺  {opt:>13,.0f}₺  {delta:>+9,.0f}₺")

    # Kural kırılımı
    print(f"\n  Kural bazlı karşılaştırma (OPTIMIZED):")
    print(f"  {'Grup':>10}  {'N':>4}  {'WR':>6}  {'Avg':>7}  {'PF':>5}  {'TotalTL':>10}")
    print(f"  {'─' * 10}  {'─' * 4}  {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 10}")
    for g in GROUPS:
        g_trades = [t for t in final_sim.closed_trades if t.group == g]
        if not g_trades:
            continue
        m = _calc_strategy_metrics(g_trades)
        total_tl = sum(t.pnl_tl for t in g_trades)
        print(f"  {g:>10}  {m['n_trades']:>4}  {m['win_rate']:>5.1f}%  "
              f"{m['avg_pnl']:>+6.2f}%  {m['profit_factor']:>5.2f}  {total_tl:>+10,.0f}")

    print(f"\n{'═' * w}")

    return all_sweep_results


def _save_sweep_csv(all_results, output_dir):
    """Sweep sonuçlarını CSV'ye kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    rows = []
    for rule_group, results in all_results.items():
        for r in results:
            row = {'rule_group': rule_group}
            row.update(r)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f'portfolio_sweep_trail_{date_str}.csv')
        df.to_csv(path, index=False)
        print(f"  Sweep CSV: {path} ({len(rows)} combo)")


# =============================================================================
# FIYAT VERISI YUKLEME
# =============================================================================

def _load_price_data(signals, period='2y'):
    """Tüm ticker'lar için fiyat verisi yükle."""
    tickers_needed = sorted(set(s['ticker'] for s in signals))
    print(f"\n  {len(tickers_needed)} ticker icin fiyat verisi yukleniyor (period={period})...")

    t0 = time.time()
    all_data = data_mod.fetch_data(tickers_needed, period=period)
    print(f"  {len(all_data)} ticker yuklendi ({time.time() - t0:.1f}s)")

    # Lowercase dönüşüm
    price_data = {}
    for ticker, df in all_data.items():
        price_data[ticker] = _to_lower_cols(df)

    return price_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Divergence Portfolio Backtesti")
    parser.add_argument('--csv', default=None, help='Backtest CSV dosyası (default: en son)')
    parser.add_argument('--period', default='2y', help='Veri periyodu (default: 2y)')
    parser.add_argument('--output', default='output', help='Çıktı dizini')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                        help=f'Başlangıç sermayesi (default: {INITIAL_CAPITAL:,.0f})')
    parser.add_argument('--pos-size', type=float, default=POSITION_SIZE_PCT,
                        help=f'Pozisyon büyüklüğü %% (default: {POSITION_SIZE_PCT})')
    parser.add_argument('--max-pos', type=int, default=MAX_CONCURRENT,
                        help=f'Max eşzamanlı pozisyon (default: {MAX_CONCURRENT})')
    parser.add_argument('--commission', type=float, default=COMMISSION_PCT,
                        help=f'Komisyon %% çift yön (default: {COMMISSION_PCT})')
    parser.add_argument('--save-csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--strategy', default=None,
                        help='Tek strateji çalıştır (FIXED/TRAILING/PARTIAL)')
    # C filtre argümanları
    parser.add_argument('--c-filter', default='none',
                        choices=list(C_FILTER_CONFIGS.keys()),
                        help='C kuralı daraltma seviyesi (default: none)')
    parser.add_argument('--c-filter-compare', action='store_true',
                        help='3 C filtre seviyesini yan yana karşılaştır (FIXED ile)')
    # Sweep argümanları
    parser.add_argument('--optimize', action='store_true',
                        help='TRAILING parametre sweep çalıştır')
    parser.add_argument('--sweep-csv', action='store_true',
                        help='Sweep sonuçlarını CSV\'ye kaydet')
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. CSV bul / yükle ──────────────────────────────────────────────────
    csv_path = args.csv
    if csv_path is None:
        import glob
        candidates = sorted(glob.glob(os.path.join(args.output, 'backtest_divergence_*.csv')))
        if not candidates:
            print("  HATA: Backtest CSV bulunamadı! Önce run_backtest_divergence.py çalıştırın.")
            sys.exit(1)
        csv_path = candidates[-1]

    w = 100
    print(f"\n{'═' * w}")
    print(f"  NOX DIVERGENCE — PORTFOLIO BACKTESTI")
    print(f"  Sermaye: {args.capital:,.0f} TL | Poz: %{args.pos_size} | "
          f"Max: {args.max_pos} | Komisyon: %{args.commission}")
    print(f"{'═' * w}")

    signals = _load_csv_signals(csv_path)
    if not signals:
        print("  HATA: Kural eşleşen sinyal yok!")
        sys.exit(1)

    # ── 2. Sinyal doğrulama + fiyat verisi ─────────────────────────────────
    price_data = _load_price_data(signals, period=args.period)

    valid_signals = []
    for s in signals:
        ticker = s['ticker']
        if ticker not in price_data:
            continue
        entry_str = str(s['entry_date'])[:10]
        try:
            entry_ts = pd.Timestamp(entry_str)
        except Exception:
            continue
        df = price_data[ticker]
        if len(df) < 2:
            continue
        last_date = df.index[-1]
        if entry_ts > last_date:
            continue
        valid_signals.append(s)

    print(f"  Geçerli sinyal: {len(valid_signals)} / {len(signals)}")

    # ── 3. C filtre karşılaştırma modu ─────────────────────────────────────
    if args.c_filter_compare:
        print(f"\n{'═' * w}")
        print(f"  C FILTRE KARSILASTIRMA (FIXED strateji ile)")
        print(f"{'═' * w}")

        for level in ['none', 'moderate', 'aggressive']:
            filtered, stats = _filter_c_signals(valid_signals, level)
            _print_filter_stats(level, stats)

            sim = PortfolioSimulator(
                initial_capital=args.capital,
                position_size_pct=args.pos_size,
                max_concurrent=args.max_pos,
                commission_pct=args.commission,
                exit_strategy='FIXED',
                rule_params=RULE_PARAMS,
            )
            sim.run(filtered, price_data)
            metrics = _calc_strategy_metrics(sim.closed_trades)
            eq_stats = _calc_equity_stats(sim.equity_curve, args.capital)

            print(f"    → N={metrics['n_trades']}, WR {metrics['win_rate']:.1f}%, "
                  f"Avg {metrics['avg_pnl']:+.2f}%, PF {metrics['profit_factor']:.2f}, "
                  f"Sharpe {metrics['sharpe']:.2f}, "
                  f"Equity {eq_stats.get('final_equity', 0):,.0f}₺")

            # C kural kırılımı
            c_trades = [t for t in sim.closed_trades if t.group == 'C']
            if c_trades:
                c_m = _calc_strategy_metrics(c_trades)
                c_tl = sum(t.pnl_tl for t in c_trades)
                print(f"    → C: N={c_m['n_trades']}, WR {c_m['win_rate']:.1f}%, "
                      f"PF {c_m['profit_factor']:.2f}, TL {c_tl:+,.0f}")

        print(f"\n  Toplam süre: {time.time() - t0:.1f}s")
        return

    # ── 4. C filtre uygula ─────────────────────────────────────────────────
    if args.c_filter != 'none':
        valid_signals, filter_stats = _filter_c_signals(valid_signals, args.c_filter)
        _print_filter_stats(args.c_filter, filter_stats)

    # ── 5. Optimize modu ───────────────────────────────────────────────────
    if args.optimize:
        sweep_results = _run_parameter_sweep(valid_signals, price_data, args)
        if args.sweep_csv:
            _save_sweep_csv(sweep_results, args.output)
        print(f"\n  Toplam süre: {time.time() - t0:.1f}s")
        return

    # ── 6. Stratejileri belirle ────────────────────────────────────────────
    strategies = EXIT_STRATEGIES
    if args.strategy:
        strat = args.strategy.upper()
        if strat not in EXIT_STRATEGIES:
            print(f"  HATA: Geçersiz strateji: {strat}. Seçenekler: {EXIT_STRATEGIES}")
            sys.exit(1)
        strategies = [strat]

    # ── 7. Her strateji için simülasyon ───────────────────────────────────
    results = {}
    for strat in strategies:
        print(f"\n  Strateji: {strat} çalıştırılıyor...")
        t1 = time.time()

        sim = PortfolioSimulator(
            initial_capital=args.capital,
            position_size_pct=args.pos_size,
            max_concurrent=args.max_pos,
            commission_pct=args.commission,
            exit_strategy=strat,
            rule_params=RULE_PARAMS,
        )

        sim.run(valid_signals, price_data)

        metrics = _calc_strategy_metrics(sim.closed_trades)
        eq_stats = _calc_equity_stats(sim.equity_curve, args.capital)

        results[strat] = {
            'simulator': sim,
            'trades': sim.closed_trades,
            'metrics': metrics,
            'equity_stats': eq_stats,
        }

        print(f"    {strat}: {metrics['n_trades']} trade, "
              f"WR {metrics['win_rate']:.1f}%, "
              f"Avg {metrics['avg_pnl']:+.2f}%, "
              f"PF {metrics['profit_factor']:.2f}, "
              f"Atlanan {sim.skipped_count} "
              f"({time.time() - t1:.1f}s)")

    # ── 8. Raporlama ──────────────────────────────────────────────────────
    if len(strategies) > 1:
        _print_strategy_comparison(results)

    _print_rule_breakdown(results)
    _print_exit_reason_breakdown(results)
    _print_capacity_info(results)

    # ── 9. CSV kaydet ─────────────────────────────────────────────────────
    if args.save_csv:
        _save_trade_csv(results, args.output)
        _save_equity_csv(results, args.output)

    print(f"\n  Toplam süre: {time.time() - t0:.1f}s")
    print(f"  Portfolio backtesti tamamlandı.")


if __name__ == '__main__':
    main()
