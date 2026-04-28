"""Compatibility tests for the v2 (F0..F8) exit name wiring.

The reset pipeline must accept v2 carry/final names produced by the
discovery_grid phase without losing the test-split lock or the legacy
E3..E7 path.

Scope:
    * resolver accepts the three PR #3 carry names
    * resolver rejects unknown names with a clear ValueError
    * `phase_validation` argument validation accepts F-names without
      ever loading the test split
    * `phase_final_test` accepts F-names but still requires
      unlock_test=true (lock behavior unchanged)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from sbt1700 import reset_pipeline
from sbt1700.exit_grid import is_v2_name, resolve_exit_spec
from sbt1700.exits_v2 import ExitConfigV2
from sbt1700.splits import TestLockError


PR3_CARRIES = (
    "F0_no_partial_trend_sl1.5_p0_ema10_h40",
    "F2_partial_with_breakeven_sl1.5_p33at1R_be1.5R_atr2.0_h40",
    "F8c_lock_after_1_5R_sl1.5_p0_lockL3_h20",
)


# ---------- resolver ---------------------------------------------------------

@pytest.mark.parametrize("name", PR3_CARRIES)
def test_resolver_accepts_pr3_carries(name: str):
    cfg = resolve_exit_spec(name)
    assert isinstance(cfg, ExitConfigV2)
    assert cfg.name == name
    assert is_v2_name(name)


def test_resolver_rejects_unknown_with_clear_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_exit_spec("F99_does_not_exist")
    msg = str(excinfo.value)
    assert "unknown v2 exit name" in msg
    assert "F99_does_not_exist" in msg
    assert "F0" in msg  # available family prefixes surfaced


def test_resolver_rejects_legacy_name_via_v2_path():
    """E3..E7 names belong to the legacy simulator. The v2 resolver must
    not silently return a legacy spec — that would mask a dispatch bug."""
    with pytest.raises(ValueError):
        resolve_exit_spec("E3_baseline")
    assert not is_v2_name("E3_baseline")


# ---------- phase_validation arg validation (no test access) -----------------

def _empty_dataset(tmp_path: Path) -> Path:
    """Minimal dataset that satisfies load_split's schema. Validation is
    expected to fail later because the panel is empty, but only after
    the carry-name check has succeeded."""
    df = pd.DataFrame({
        "ticker": ["A"],
        "date": [pd.Timestamp("2024-02-01")],
        "split": ["train"],
        "close_1700": [10.0],
        "atr14_prior": [0.5],
    })
    p = tmp_path / "ds.parquet"
    df.to_parquet(p)
    return p


def _empty_master(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
    df = pd.DataFrame({
        "ticker": "A", "Open": 10.0, "High": 10.5, "Low": 9.5, "Close": 10.0,
    }, index=dates)
    df.index.name = "date"
    p = tmp_path / "master.parquet"
    df.to_parquet(p)
    return p


def test_phase_validation_accepts_f_carries_without_test_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """F-name carries must pass the carry-name check. We monkey-patch
    load_split to fail fast on test access and return empties on
    train/validation, so we observe ONLY the name-check verdict."""
    accesses: list[str] = []

    def spy_load_split(path, phase, allow_test=False):
        accesses.append(phase)
        if phase == "test" or allow_test:
            raise AssertionError(
                f"phase_validation must not touch test split "
                f"(phase={phase!r}, allow_test={allow_test!r})"
            )
        # Return empty frame to short-circuit the rest of the phase
        # AFTER the carry-name check has already succeeded.
        return pd.DataFrame()

    monkeypatch.setattr(reset_pipeline, "load_split", spy_load_split)

    args = argparse.Namespace(
        dataset=_empty_dataset(tmp_path),
        master=_empty_master(tmp_path),
        out_dir=tmp_path / "out",
        carry=",".join(PR3_CARRIES),
        min_n_for_ranker=40,
    )

    # Empty splits make `phase_validation` raise RuntimeError after
    # the name check passes — that's what we want here.
    with pytest.raises(RuntimeError, match="train or validation split is empty"):
        reset_pipeline.phase_validation(args)

    assert "test" not in accesses, (
        f"phase_validation must never read test split; saw {accesses!r}"
    )


def test_phase_validation_rejects_unknown_carry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    args = argparse.Namespace(
        dataset=_empty_dataset(tmp_path),
        master=_empty_master(tmp_path),
        out_dir=tmp_path / "out",
        carry="F99_does_not_exist",
        min_n_for_ranker=40,
    )
    with pytest.raises(SystemExit, match="unknown exit"):
        reset_pipeline.phase_validation(args)


# ---------- phase_final_test lock behavior unchanged -------------------------

def test_phase_final_test_accepts_f_name_but_still_requires_unlock(
    tmp_path: Path,
):
    """F-name --exit must pass the name check but the test-split lock
    is unchanged: unlock_test=False -> TestLockError (raised before any
    name validation runs, so this is a separate gate)."""
    args = argparse.Namespace(
        dataset=_empty_dataset(tmp_path),
        master=_empty_master(tmp_path),
        out_dir=tmp_path / "out",
        exit=PR3_CARRIES[2],
        unlock_test=False,
    )
    with pytest.raises(TestLockError):
        reset_pipeline.phase_final_test(args)


def test_phase_final_test_unknown_exit_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Even with unlock_test=True, an unknown name must be rejected
    by the resolver gate before any test data is touched."""
    accesses: list[str] = []

    def spy_load_split(path, phase, allow_test=False):
        accesses.append(phase)
        return pd.DataFrame()

    monkeypatch.setattr(reset_pipeline, "load_split", spy_load_split)

    args = argparse.Namespace(
        dataset=_empty_dataset(tmp_path),
        master=_empty_master(tmp_path),
        out_dir=tmp_path / "out",
        exit="F99_does_not_exist",
        unlock_test=True,
    )
    with pytest.raises(SystemExit, match="unknown"):
        reset_pipeline.phase_final_test(args)
    # Lock semantics: name check fires before any test load attempts.
    assert "test" not in accesses, (
        f"final_test name-rejection must not touch test split first; saw {accesses!r}"
    )
