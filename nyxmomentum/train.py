"""
Model training tracks.

  rank_composite   weighted linear of standardized features (no fitting)
  lightgbm_reg     LightGBM regressor on L2 forward_excess_return
  lightgbm_cls     LightGBM classifier on L3 outperform_binary
                   (optional Platt / isotonic calibration)

Walk-forward expanding: train on cumulative history up to fold.train_end,
validate on fold.val_*, select iteration via val IC (for regressor) or
val log-loss (for classifier), test on fold.test_*. Test predictions are
concatenated across folds to produce OOS scores.
"""
from __future__ import annotations

import pandas as pd

from .config import ModelConfig


def train_and_predict(dataset: pd.DataFrame,
                      feature_cols: list[str],
                      label_col: str,
                      config: ModelConfig | None = None) -> pd.DataFrame:
    """OOS score frame — (ticker, rebalance_date, score, ...)."""
    raise NotImplementedError("Step 4")


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    raise NotImplementedError("Step 4")
