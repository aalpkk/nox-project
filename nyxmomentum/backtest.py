"""
Cost-aware backtest.

Execution model (BacktestConfig.execution):
  next_open         — weights known at rebalance_date close, trades execute at
                      next session open. Default. Leakage-safe.
  rebalance_close   — trades execute at rebalance_date close. Overstates
                      performance unless matched by feature cutoff; use for
                      sensitivity only.

Costs:
  commission_bps per one-way TL notional
  slippage_bps   per one-way TL notional

Outputs: daily NAV series, per-rebalance trade log, turnover series,
gross vs net return curves.
"""
from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


def run_backtest(weights: pd.DataFrame,
                 panel: dict[str, pd.DataFrame],
                 rebalance_dates: pd.DatetimeIndex,
                 benchmark: pd.DataFrame | None,
                 config: BacktestConfig | None = None) -> dict:
    """
    Returns dict with:
      nav_gross, nav_net   — pd.Series (daily)
      trades                — pd.DataFrame (one row per buy/sell leg)
      turnover_series       — pd.Series indexed by rebalance_date
      by_period             — pd.DataFrame (per-period returns, gross/net)
    """
    raise NotImplementedError("Step 5")
