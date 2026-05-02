"""n-bar fractal pivot finder with confirmation-lag semantics.

Close-only: a bar i is a swing-high pivot iff `close[i]` is the strict
maximum over the window `close[i-n .. i+n]`; symmetric for swing-low.
Wicks AND opens are intentionally ignored — only the closing print marks
a pivot. This matches the conventional close-line zigzag traders read
visually (e.g. a red bar with very high open does not get treated as a
swing-high just because its open exceeds neighbors).

Confirmation requires bar `i+n` to exist — i.e. the pivot at index i is
*only* known to a live observer at bar `i+n`. For an asof bar `end_idx`,
this returns indices `i` such that `i+n <= end_idx`. Strict equality is
enforced (no ties allowed at the pivot bar) to avoid ambiguous pivots in
flat regions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_pivots(
    df: pd.DataFrame,
    n: int,
    end_idx: int,
) -> tuple[list[int], list[int]]:
    """Return (swing_high_indices, swing_low_indices) confirmed at bar end_idx.

    Close-only: uses close[i] for both swing highs and swing lows.

    Parameters
    ----------
    df : DataFrame with column 'close'.
    n : fractal half-window. A pivot at bar i is confirmed at i+n.
    end_idx : last visible bar index. Pivots with i+n > end_idx are excluded.
    """
    highs: list[int] = []
    lows: list[int] = []
    c = df["close"].to_numpy()
    last_confirmable = end_idx - n
    for i in range(n, last_confirmable + 1):
        win = c[i - n: i + n + 1]
        if c[i] >= win.max() and (win == c[i]).sum() == 1:
            highs.append(i)
        if c[i] <= win.min() and (win == c[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def alternating_pivots(
    df: pd.DataFrame,
    n: int,
    end_idx: int,
) -> list[tuple[str, int]]:
    """Merge swing highs/lows into a strictly alternating sequence (zig-zag).

    Returns a list of (kind, idx) tuples sorted by idx, where kind ∈ {'H','L'}
    and consecutive elements alternate. When two same-kind pivots occur
    consecutively (no opposite pivot between them), keeps the more extreme
    one (higher close H, lower close L). This collapses noise so quartet
    detection sees a clean LL→LH→HL→HH skeleton candidates.
    """
    sh, sl = find_pivots(df, n, end_idx)
    pivots: list[tuple[str, int]] = (
        [("H", i) for i in sh] + [("L", i) for i in sl]
    )
    pivots.sort(key=lambda x: x[1])
    if not pivots:
        return []

    c = df["close"].to_numpy()
    cleaned: list[tuple[str, int]] = []
    for kind, idx in pivots:
        if not cleaned or cleaned[-1][0] != kind:
            cleaned.append((kind, idx))
            continue
        # same kind as last — keep the more extreme close
        prev_kind, prev_idx = cleaned[-1]
        if kind == "H":
            if c[idx] > c[prev_idx]:
                cleaned[-1] = (kind, idx)
        else:
            if c[idx] < c[prev_idx]:
                cleaned[-1] = (kind, idx)
    return cleaned
