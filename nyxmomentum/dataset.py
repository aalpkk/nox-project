"""
Panel assembly + walk-forward split assignment.

Join feature frame with label frame on (ticker, rebalance_date); attach fold
tag per SplitConfig. Persist to parquet for reuse across modelling runs.
Splits are date-anchored and frozen — hyperparameter search must not alter
them (spec §12).
"""
from __future__ import annotations

import pandas as pd

from .config import Config, SplitConfig


def build_dataset(features: pd.DataFrame,
                  labels: pd.DataFrame,
                  config: Config | None = None) -> pd.DataFrame:
    """
    Inner-join features and labels on (ticker, rebalance_date). Attaches
    a 'fold' column (fold1..foldN or 'none') and a 'split' column
    (train | val | test | embargo).
    """
    raise NotImplementedError("Step 4")


def assign_fold_tag(rebalance_dates: pd.Series,
                    config: SplitConfig) -> pd.DataFrame:
    """DataFrame indexed as input with columns ['fold', 'split']."""
    raise NotImplementedError("Step 4")


def persist_parquet(panel: pd.DataFrame, path: str) -> None:
    """Save with a short README alongside describing schema + produced_at."""
    raise NotImplementedError("Step 4")
