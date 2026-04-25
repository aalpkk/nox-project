"""
Trigger A — Clean Breakout.

Panel anchor: bir (ticker, date) satır yalnızca şu koşullar AND True ise True:
  1. close > max(high over prior `lookback_high` bars, excluding today)
  2. volume / sma(volume, rvol_win) >= rvol_min
  3. (close - low) / max(high - low, tiny) >= close_loc_min

Tüm karşılaştırmalar o bar'ın verisiyle yapılır → leakage yok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyxexpansion.config import TriggerAParams


def _rolling_prior_high(high: pd.Series, n: int) -> pd.Series:
    """max(high[t-n..t-1]). shift(1) ile today hariç tutulur."""
    return high.rolling(n, min_periods=n).max().shift(1)


def compute_trigger_a(df: pd.DataFrame, params: TriggerAParams | None = None) -> pd.DataFrame:
    """Tek-ticker OHLCV → trigger flag + yardımcı kolonlar.

    Args:
        df: DatetimeIndex, kolonlar Open/High/Low/Close/Volume.
        params: None ise default TriggerAParams().

    Returns:
        DataFrame with columns: trigger_a, prior_high_20, rvol, close_loc,
                                trigger_level (= prior_high_20 + tiny).
        Leading NaN rows (bootstrap period) drop edilmez; caller filtreler.
    """
    if params is None:
        params = TriggerAParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    vol = df['Volume'].astype(float)

    prior_high = _rolling_prior_high(high, params.lookback_high)
    vol_sma = vol.rolling(params.rvol_win, min_periods=params.rvol_win).mean()
    rvol = vol / vol_sma.replace(0.0, np.nan)

    bar_range = (high - low).replace(0.0, np.nan)
    close_loc = (close - low) / bar_range

    cond_break = close > prior_high
    cond_rvol = rvol >= params.rvol_min
    cond_loc = close_loc >= params.close_loc_min
    trig = (cond_break & cond_rvol & cond_loc).fillna(False)

    out = pd.DataFrame({
        'prior_high_20': prior_high,
        'rvol': rvol,
        'close_loc': close_loc,
        'trigger_level': prior_high,   # entry reference
        'trigger_a': trig.astype(bool),
    }, index=df.index)
    return out


def compute_trigger_a_panel(
    data_by_ticker: dict[str, pd.DataFrame],
    params: TriggerAParams | None = None,
) -> pd.DataFrame:
    """Çok-ticker → uzun panel (ticker, date) satırları (trigger True olanlar).

    Args:
        data_by_ticker: {ticker: OHLCV DataFrame}.
        params: TriggerAParams or None.

    Returns:
        Long DataFrame with columns:
            ticker, date, close, prior_high_20, rvol, close_loc, trigger_level.
        Yalnızca trigger_a True olan satırlar dahildir. `date` DatetimeIndex değil,
        kolondur — çoklu ticker'a uygun düz panel.
    """
    if params is None:
        params = TriggerAParams()

    rows: list[pd.DataFrame] = []
    for ticker, df in data_by_ticker.items():
        if df is None or df.empty or len(df) < params.lookback_high + params.rvol_win:
            continue
        feat = compute_trigger_a(df, params)
        mask = feat['trigger_a']
        if not mask.any():
            continue
        sub = feat.loc[mask, ['prior_high_20', 'rvol', 'close_loc', 'trigger_level']].copy()
        sub['close'] = df.loc[mask, 'Close'].astype(float)
        sub['ticker'] = ticker
        sub.index.name = 'date'
        sub = sub.reset_index()
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=['ticker', 'date', 'close', 'prior_high_20',
                                     'rvol', 'close_loc', 'trigger_level'])

    panel = pd.concat(rows, ignore_index=True)
    panel = panel[['ticker', 'date', 'close', 'prior_high_20',
                   'rvol', 'close_loc', 'trigger_level']]
    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)
    return panel
