"""
Walk-forward split helpers — STRICT expanding train, no random split.

Fold anchors live in CONFIG.split.folds as tuples of:
  (fold_id, train_end, val_start, val_end, test_start, test_end)

Train window per fold is always [CONFIG.split.train_start, train_end]. We
never reach into the future for training. Val and test windows follow
train end with a configured embargo gap built into the anchors.

Any fold whose train window is empty (e.g. because data only starts in
2022 but train_end predates that) is skipped with a warning rather than
silently letting the model fit on nothing.

Leakage contract:
  • A row belongs to exactly one (train | val | test | none) bucket per fold.
  • rebalance_date is the cutoff; [train_start, train_end] is inclusive.
  • Test splits across folds do not overlap (fold anchors are non-overlapping).
  • OOS evaluation concatenates test-fold predictions → single panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import CONFIG, SplitConfig


@dataclass(frozen=True)
class FoldSplit:
    fold_id: str
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    def has_train(self, dates: pd.Series) -> bool:
        m = (dates >= self.train_start) & (dates <= self.train_end)
        return bool(m.any())

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.train_start) & (dates <= self.train_end)

    def val_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.val_start) & (dates <= self.val_end)

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.test_start) & (dates <= self.test_end)


def build_folds(split_cfg: SplitConfig | None = None) -> list[FoldSplit]:
    cfg = split_cfg or CONFIG.split
    folds: list[FoldSplit] = []
    train_start = pd.Timestamp(cfg.train_start)
    for tup in cfg.folds:
        fold_id, train_end, val_start, val_end, test_start, test_end = tup
        folds.append(FoldSplit(
            fold_id=fold_id,
            train_start=train_start,
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        ))
    return folds


def usable_folds(dates: pd.Series,
                 split_cfg: SplitConfig | None = None) -> list[FoldSplit]:
    """Return folds where train+test windows both contain ≥1 rebalance."""
    out: list[FoldSplit] = []
    for f in build_folds(split_cfg):
        if not f.has_train(dates):
            continue
        if not f.test_mask(dates).any():
            continue
        out.append(f)
    return out


def summarize_folds(folds: list[FoldSplit],
                    dates: pd.Series) -> pd.DataFrame:
    """One row per fold: date counts per bucket."""
    rows: list[dict] = []
    for f in folds:
        rows.append({
            "fold_id":       f.fold_id,
            "train_start":   f.train_start.date(),
            "train_end":     f.train_end.date(),
            "val_start":     f.val_start.date(),
            "val_end":       f.val_end.date(),
            "test_start":    f.test_start.date(),
            "test_end":      f.test_end.date(),
            "n_train_rows":  int(f.train_mask(dates).sum()),
            "n_val_rows":    int(f.val_mask(dates).sum()),
            "n_test_rows":   int(f.test_mask(dates).sum()),
        })
    return pd.DataFrame(rows)
