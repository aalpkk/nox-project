"""SBT-1700 — Smart Breakout Targets at 17:00 TR (last completed 15m bar).

Independent of the legacy SBT ML / bucket system. The candidate ranker
is trained from scratch on a 17:00-truncated feature set with realized-R
labels from a frozen E3 execution rule.

See sbt1700/README.md for the locked design decisions.
"""

from sbt1700.config import (
    BAR_CUTOFF_HH,
    BAR_CUTOFF_MM,
    EXPECTED_BARS_PER_PAIR,
    COVERAGE_DROP_THRESHOLD,
    E3,
)

__all__ = [
    "BAR_CUTOFF_HH",
    "BAR_CUTOFF_MM",
    "EXPECTED_BARS_PER_PAIR",
    "COVERAGE_DROP_THRESHOLD",
    "E3",
]
