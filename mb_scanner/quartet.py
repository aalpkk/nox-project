"""LL → LH → HL → HH quartet detector (close-only).

Given confirmed close-pivots (via `pivots.alternating_pivots`), find
chronologically-ordered triplets (LL, LH, HL) such that:

  - LL, LH, HL are confirmed close pivots (lag-confirmed by end_idx)
  - LL (low) → LH (high) → HL (low) chronologically
  - For 'mb' mode: close[HL] > close[LL]   (failed-bounce reversal)
  - For 'bb' mode: close[HL] < close[LL]   (sweep below first low)

Then HH = the first bar after HL (and ≤ end_idx) with close > close[LH].
The BoS reference is the LH close — consistent with how LH was identified.
HH is the BoS confirmation bar; it is NOT required to itself be a pivot.

All valid quartets at as-of are returned by `find_all_quartets`. When
nested structure exists (a wide LL→LH→HL→HH macro setup with one or more
narrower quartets inside it), each is reported independently — they may
all be simultaneously active. `find_latest_quartet` returns the one with
the most recent HH (BoS) — convenient for single-row consumers.

Each quartet's full span (LL → HH) must fit within `max_quartet_span_bars`
to keep stale structures out.

Output `*_price` fields are close-based so downstream consumers see the
same reference levels used in the selection logic.
"""
from __future__ import annotations

import pandas as pd

from .pivots import alternating_pivots


def find_all_quartets(
    df: pd.DataFrame,
    *,
    n: int,
    end_idx: int,
    mode: str = "mb",
    max_quartet_span_bars: int = 200,
) -> list[dict]:
    """Return all valid quartets at end_idx, sorted by hh_idx ascending.

    Output dict keys per quartet: ll_idx, ll_price, lh_idx, lh_price,
    hl_idx, hl_price, hh_idx, hh_close. Prices are close-based.

    Close-only alternating pivots produce a zigzag L,H,L,H,... so
    consecutive (L,H,L) windows naturally enumerate non-overlapping
    candidate triplets; a wider macro quartet can coexist with one or
    more narrower nested quartets in the same as-of snapshot.
    """
    if mode not in ("mb", "bb"):
        raise ValueError(f"unknown mode: {mode!r}")
    pivots = alternating_pivots(df, n, end_idx)
    if len(pivots) < 3:
        return []

    c = df["close"].to_numpy()

    candidates: list[dict] = []
    for i in range(len(pivots) - 2):
        if pivots[i][0] != "L" or pivots[i + 1][0] != "H" or pivots[i + 2][0] != "L":
            continue
        ll_idx = pivots[i][1]
        lh_idx = pivots[i + 1][1]
        hl_idx = pivots[i + 2][1]
        # Reject degenerate quartets where two opposite-kind pivots collapse
        # onto the same bar.
        if not (ll_idx < lh_idx < hl_idx):
            continue

        if mode == "mb":
            if not (c[hl_idx] > c[ll_idx]):
                continue
        else:  # bb sweep
            if not (c[hl_idx] < c[ll_idx]):
                continue

        # BoS: first bar in (hl_idx, end_idx] closing above LH close.
        lh_close = c[lh_idx]
        hh_idx = -1
        for j in range(hl_idx + 1, end_idx + 1):
            if c[j] > lh_close:
                hh_idx = j
                break
        if hh_idx < 0:
            continue

        if hh_idx - ll_idx > max_quartet_span_bars:
            continue

        candidates.append({
            "ll_idx": ll_idx, "ll_price": float(c[ll_idx]),
            "lh_idx": lh_idx, "lh_price": float(c[lh_idx]),
            "hl_idx": hl_idx, "hl_price": float(c[hl_idx]),
            "hh_idx": hh_idx, "hh_close": float(c[hh_idx]),
        })

    candidates.sort(key=lambda q: q["hh_idx"])
    return candidates


def find_latest_quartet(
    df: pd.DataFrame,
    *,
    n: int,
    end_idx: int,
    mode: str = "mb",
    max_quartet_span_bars: int = 200,
) -> dict | None:
    """Return the most-recent-HH quartet, or None. Thin wrapper."""
    qs = find_all_quartets(
        df, n=n, end_idx=end_idx, mode=mode,
        max_quartet_span_bars=max_quartet_span_bars,
    )
    if not qs:
        return None
    return qs[-1]  # already sorted ascending by hh_idx
