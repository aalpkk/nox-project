"""
Production scoring entrypoint — one call per rebalance date.

Wraps universe → features → {baseline | trained model} → score, and returns a
sorted DataFrame ready for portfolio selection and display. Model artifacts
loaded lazily from Config.paths.artifacts.

Output columns:
  ticker, date, score, rank_overall, rank_within_sector,
  prob_outperform, expected_excess_return,
  liquidity_warning, overheat_warning
"""
from __future__ import annotations

import pandas as pd

from .config import Config


class MomentumScorer:
    """Thin orchestration layer — no new logic, only wiring."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def score(self, rebalance_date: pd.Timestamp,
              panel: dict[str, pd.DataFrame],
              xu100: pd.DataFrame | None = None,
              sector_map: dict[str, str] | None = None,
              macro: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        raise NotImplementedError("Step 7")
