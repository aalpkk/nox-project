"""SBT-1700 RESET — locked train / validation / test split.

The split bounds are decided up front and the test slice is gated
behind an explicit `--unlock-test` to make accidental contamination
impossible during discovery and validation.

Hard rule (do not silently change):
    train      : 2024-01-16 → 2025-06-30   — all discovery work happens here
    validation : 2025-07-01 → 2025-12-31   — at most 2 carried candidates
    test       : 2026-01-01 → 2026-04-24   — locked, one-shot readout

If `phase == "test"` and `allow_test` is False, `load_split` raises
`TestLockError`. The CI guard (tests/sbt1700/test_splits_lock.py)
verifies this so the lock cannot be regressed silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Locked bounds — change requires SCHEMA_VERSION bump and a written rationale.
TRAIN_START = pd.Timestamp("2024-01-16")
TRAIN_END   = pd.Timestamp("2025-06-30")
VAL_START   = pd.Timestamp("2025-07-01")
VAL_END     = pd.Timestamp("2025-12-31")
TEST_START  = pd.Timestamp("2026-01-01")
TEST_END    = pd.Timestamp("2026-04-24")

SPLIT_SCHEMA_VERSION = 1

CONTAMINATION_NOTICE = (
    "Previous PR-2 E7 result was contaminated by same-dataset exit "
    "selection and is excluded from this reset."
)

PHASES = ("train", "validation", "test")


class TestLockError(RuntimeError):
    """Raised when the test split is requested without `allow_test=True`."""
    __test__ = False  # tell pytest this is not a test class


@dataclass(frozen=True)
class SplitBounds:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, dates: pd.Series) -> pd.Series:
        d = pd.to_datetime(dates)
        return (d >= self.start) & (d <= self.end)


SPLITS: dict[str, SplitBounds] = {
    "train":      SplitBounds("train",      TRAIN_START, TRAIN_END),
    "validation": SplitBounds("validation", VAL_START,   VAL_END),
    "test":       SplitBounds("test",       TEST_START,  TEST_END),
}


def load_split(
    dataset_path: str | Path,
    phase: str,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Read the dataset and return rows belonging to `phase`.

    The test split is hard-locked: requesting it without
    `allow_test=True` raises `TestLockError` regardless of dataset
    contents. This function is the single chokepoint for split access;
    every consumer in the reset pipeline routes through it.
    """
    if phase not in SPLITS:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    if phase == "test" and not allow_test:
        raise TestLockError(
            "Test split (2026-01-01 → 2026-04-24) is locked. Pass "
            "allow_test=True (CLI: --unlock-test) only for the final "
            "one-shot readout. " + CONTAMINATION_NOTICE
        )
    df = pd.read_parquet(dataset_path)
    if "date" not in df.columns:
        raise KeyError("dataset is missing the 'date' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    bounds = SPLITS[phase]
    out = df[bounds.mask(df["date"])].reset_index(drop=True)
    return out


def split_counts(dataset_path: str | Path) -> dict[str, int]:
    """Count rows per split without unlocking the test slice."""
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    return {name: int(b.mask(df["date"]).sum()) for name, b in SPLITS.items()}


def dataset_fingerprint(dataset_path: str | Path) -> str:
    """SHA-256 (first 16 hex chars) of the dataset bytes for the manifest."""
    h = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_split_manifest(
    dataset_path: str | Path,
    out_path: str | Path,
) -> dict:
    """Write the split manifest JSON. Test row count is *not* leaked.

    The manifest binds a specific dataset file (sha-256 prefix) to a
    set of split bounds. Re-running discovery/validation against a
    different dataset file should fail manifest verification.
    """
    counts = split_counts(dataset_path)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256_first16": dataset_fingerprint(dataset_path),
        "splits": {
            "train": {
                "start": TRAIN_START.strftime("%Y-%m-%d"),
                "end":   TRAIN_END.strftime("%Y-%m-%d"),
                "n_rows": counts["train"],
            },
            "validation": {
                "start": VAL_START.strftime("%Y-%m-%d"),
                "end":   VAL_END.strftime("%Y-%m-%d"),
                "n_rows": counts["validation"],
            },
            "test": {
                "start": TEST_START.strftime("%Y-%m-%d"),
                "end":   TEST_END.strftime("%Y-%m-%d"),
                "n_rows": "LOCKED — query gated behind --unlock-test",
                "n_rows_under_lock": counts["test"],
            },
        },
        "lock_rule": (
            "Test slice cannot be loaded without --unlock-test. "
            "Final-test readout is one-shot; subsequent rule changes "
            "require a fresh test set or a written re-contamination note."
        ),
        "contamination_notice": CONTAMINATION_NOTICE,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(
    manifest_path: str | Path,
    dataset_path: str | Path,
) -> None:
    """Raise if the manifest's dataset fingerprint does not match the file."""
    manifest = json.loads(Path(manifest_path).read_text())
    expected = manifest.get("dataset_sha256_first16")
    actual = dataset_fingerprint(dataset_path)
    if expected != actual:
        raise RuntimeError(
            f"dataset fingerprint mismatch: manifest expects {expected!r} "
            f"but {dataset_path} hashes to {actual!r}. Re-run the "
            "manifest step with the intended dataset."
        )
