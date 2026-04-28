"""
Score → portfolio weights.

Rules (from PortfolioConfig):
  - top_n OR top_quantile (never both — validated)
  - weighting = equal | score (with caps)
  - max_weight_per_stock, max_weight_per_sector — applied after selection,
    with residual redistributed to unconstrained names
  - hold_until_next_rebalance = True → no intra-period trading

Output: wide DataFrame indexed by rebalance_date, columns = ticker, values =
target weight (0 if not held). Weights sum to ≤ 1 per date (cash = 1 − Σ).
"""
from __future__ import annotations

import pandas as pd

from .config import PortfolioConfig


def build_portfolio(scores: pd.DataFrame,
                    eligible_panel: pd.DataFrame,
                    sector_map: dict[str, str] | None,
                    config: PortfolioConfig | None = None) -> pd.DataFrame:
    """Wide weight matrix (rebalance_date × ticker)."""
    raise NotImplementedError("Step 5")


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance date, as a fraction of portfolio NAV."""
    raise NotImplementedError("Step 5")
