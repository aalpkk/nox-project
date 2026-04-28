"""
Evaluation dashboards.

Portfolio metrics: CAGR, ann vol, Sharpe, Sortino, max DD, Calmar, hit rate,
turnover, exposure, tail ratio.

Relative metrics: alpha vs XU100, information ratio, rolling alpha,
excess-return curve.

Ranking metrics: decile spread, top-minus-bottom, per-date Spearman IC,
mean rank correlation, top-N hit rate.

Stability slices: bull / range / weak market segmentation, per-sector breakdown,
liquidity bucket breakdown, microcap-included vs excluded, subperiod tables.

Standard comparisons (spec §17):
  - Classic 12-1 vs Momentum+
  - Momentum+ vs ML score
  - gross vs net
  - full universe vs liquidity-filtered
  - equal vs capped weight
  - exhaustion penalty on/off
  - monthly vs weekly rebalance
"""
from __future__ import annotations

import pandas as pd


def portfolio_metrics(nav: pd.Series, benchmark_nav: pd.Series | None = None) -> dict:
    raise NotImplementedError("Step 6")


def ranking_metrics(scores: pd.DataFrame, labels: pd.DataFrame) -> dict:
    raise NotImplementedError("Step 6")


def decile_report(scores: pd.DataFrame, labels: pd.DataFrame,
                  n_buckets: int = 10) -> pd.DataFrame:
    raise NotImplementedError("Step 6")


def stability_slices(nav: pd.Series, regime_series: pd.Series) -> pd.DataFrame:
    raise NotImplementedError("Step 6")
