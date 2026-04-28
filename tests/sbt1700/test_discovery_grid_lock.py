"""CI guard for the SBT-1700 RESET `discovery_grid` phase.

Failing this test is a methodology breach. The grid-discovery phase
must read the train split only — never validation, never test. If a
refactor accidentally widens its read scope, this test fails before
the reset can ship.

The probe monkey-patches `load_split` to record every (path, phase,
allow_test) call made during `phase_discovery_grid`. The assertions:

    1. load_split is called at least once with phase='train'.
    2. load_split is never called with phase='validation' or 'test'.
    3. load_split is never called with allow_test=True.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from sbt1700 import reset_pipeline


@pytest.fixture
def tiny_panel(tmp_path: Path) -> Path:
    """Synthetic dataset spanning train, val, test windows."""
    rows = [
        # train rows — must be the only ones discovery_grid touches
        {"ticker": "A", "date": "2024-02-01", "close_1700": 10.0, "atr14_prior": 0.5,
         "feat_x": 1.0, "realized_R_net": 0.5},
        {"ticker": "B", "date": "2025-03-15", "close_1700": 20.0, "atr14_prior": 1.0,
         "feat_x": 2.0, "realized_R_net": -0.2},
        # val + test rows — discovery_grid must NOT read these
        {"ticker": "C", "date": "2025-09-15", "close_1700": 30.0, "atr14_prior": 1.5,
         "feat_x": 3.0, "realized_R_net": 1.1},
        {"ticker": "D", "date": "2026-02-10", "close_1700": 40.0, "atr14_prior": 2.0,
         "feat_x": 4.0, "realized_R_net": -0.7},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    p = tmp_path / "panel.parquet"
    df.to_parquet(p)
    return p


@pytest.fixture
def tiny_master(tmp_path: Path) -> Path:
    """Daily master with enough forward bars for the simulator to act on."""
    dates = pd.date_range("2024-01-01", "2026-04-25", freq="B")
    frames = []
    for tk, base in [("A", 10.0), ("B", 20.0), ("C", 30.0), ("D", 40.0)]:
        frames.append(pd.DataFrame({
            "ticker": tk,
            "Open": base, "High": base * 1.05, "Low": base * 0.95, "Close": base,
        }, index=dates))
    df = pd.concat(frames)
    df.index.name = "date"
    p = tmp_path / "master.parquet"
    df.to_parquet(p)
    return p


def test_discovery_grid_reads_only_train_split(
    monkeypatch: pytest.MonkeyPatch, tiny_panel: Path, tiny_master: Path,
    tmp_path: Path,
):
    calls: list[dict] = []
    real_load_split = reset_pipeline.load_split

    def spy_load_split(path, phase, allow_test=False):
        calls.append({"phase": phase, "allow_test": allow_test})
        return real_load_split(path, phase, allow_test=allow_test)

    monkeypatch.setattr(reset_pipeline, "load_split", spy_load_split)

    args = argparse.Namespace(
        dataset=tiny_panel,
        master=tiny_master,
        out_dir=tmp_path / "out",
        min_n_for_ranker=40,
    )
    reset_pipeline.phase_discovery_grid(args)

    phases_seen = [c["phase"] for c in calls]
    assert "train" in phases_seen, (
        f"discovery_grid must read the train split; saw {phases_seen!r}"
    )
    forbidden = {"validation", "test"}
    leaked = [c for c in calls if c["phase"] in forbidden]
    assert not leaked, (
        f"discovery_grid leaked into non-train splits: {leaked!r}"
    )
    unlocks = [c for c in calls if c["allow_test"]]
    assert not unlocks, (
        f"discovery_grid must never set allow_test=True; saw {unlocks!r}"
    )


def test_discovery_grid_writes_spec_filenames(
    tiny_panel: Path, tiny_master: Path, tmp_path: Path,
):
    """The three locked output filenames must appear in out_dir."""
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        dataset=tiny_panel,
        master=tiny_master,
        out_dir=out_dir,
        min_n_for_ranker=40,
    )
    reset_pipeline.phase_discovery_grid(args)

    for name in (
        "sbt_1700_exit_discovery_grid.csv",
        "sbt_1700_exit_discovery_family_summary.csv",
        "sbt_1700_exit_discovery_recommended_validation_exits.md",
    ):
        assert (out_dir / name).exists(), f"missing spec output: {name}"
