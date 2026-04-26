"""CI guard for the SBT-1700 RESET test-period lock.

Failing this test is not a bug — it's a methodology breach. The whole
point of the reset is that the test slice (2026-01-01 → 2026-04-24)
cannot be queried during discovery / validation / debugging. If a code
change makes `load_split(..., phase="test")` return rows without
explicit `allow_test=True`, the lock is broken and the reset is no
longer credible.

These tests build a tiny synthetic dataset spanning all three phases
so they don't depend on the production parquet existing in CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sbt1700.splits import (
    PHASES,
    SPLITS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    TestLockError,
    load_split,
    split_counts,
    write_split_manifest,
)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    rows = [
        {"ticker": "A", "date": "2024-02-01", "x": 1.0, "realized_R_net": 0.5},
        {"ticker": "B", "date": "2025-03-15", "x": 2.0, "realized_R_net": -0.2},
        {"ticker": "C", "date": "2025-09-15", "x": 3.0, "realized_R_net": 1.1},
        {"ticker": "D", "date": "2026-02-10", "x": 4.0, "realized_R_net": -0.7},
        {"ticker": "E", "date": "2026-04-20", "x": 5.0, "realized_R_net": 0.3},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    p = tmp_path / "synthetic.parquet"
    df.to_parquet(p)
    return p


def test_phases_constant_matches_splits():
    assert set(PHASES) == set(SPLITS.keys())


def test_train_split_only_returns_train_dates(tiny_dataset: Path):
    df = load_split(tiny_dataset, "train")
    assert not df.empty
    assert df["date"].min() >= TRAIN_START
    assert df["date"].max() <= TRAIN_END
    assert set(df["ticker"]) == {"A", "B"}


def test_validation_split_only_returns_validation_dates(tiny_dataset: Path):
    df = load_split(tiny_dataset, "validation")
    assert df["date"].min() >= VAL_START
    assert df["date"].max() <= VAL_END
    assert set(df["ticker"]) == {"C"}


def test_test_split_locked_by_default(tiny_dataset: Path):
    """The whole point of this PR — must raise without unlock."""
    with pytest.raises(TestLockError):
        load_split(tiny_dataset, "test")


def test_test_split_locked_with_allow_test_false_explicit(tiny_dataset: Path):
    with pytest.raises(TestLockError):
        load_split(tiny_dataset, "test", allow_test=False)


def test_test_split_unlocks_with_explicit_flag(tiny_dataset: Path):
    df = load_split(tiny_dataset, "test", allow_test=True)
    assert df["date"].min() >= TEST_START
    assert df["date"].max() <= TEST_END
    assert set(df["ticker"]) == {"D", "E"}


def test_unknown_phase_rejected(tiny_dataset: Path):
    with pytest.raises(ValueError):
        load_split(tiny_dataset, "holdout")


def test_split_counts_does_not_unlock_test(tiny_dataset: Path):
    """Counts can be reported without leaking row contents."""
    counts = split_counts(tiny_dataset)
    assert counts == {"train": 2, "validation": 1, "test": 2}


def test_manifest_does_not_leak_test_n_rows_field_at_top_level(
    tiny_dataset: Path, tmp_path: Path
):
    out = tmp_path / "manifest.json"
    manifest = write_split_manifest(tiny_dataset, out)
    test_block = manifest["splits"]["test"]
    # The user-facing n_rows must be the LOCKED placeholder, not an integer.
    assert isinstance(test_block["n_rows"], str)
    assert "LOCKED" in test_block["n_rows"]
    # The integer count is allowed in a separate field for manifest integrity,
    # but the primary n_rows slot stays locked-by-default.
    assert isinstance(test_block["n_rows_under_lock"], int)
