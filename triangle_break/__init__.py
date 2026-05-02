"""triangle_break — descriptive scanner for converging-line up-breaks.

v0 spec (locked 2026-05-02):
  - Same panel + pivot/OLS primitives as channel_break (close-only n=2,
    4 TFs, alternating ≥2/2)
  - Routing: parallelism > 0.25 (NOT a parallel channel)
  - Subtypes accepted: ascending, symmetric, descending (long-only up-break)
  - Subtypes rejected: expanding, ambiguous
  - Convergence gates:
      bars_to_apex > 0           (lines haven't crossed yet — not expired)
      width_contraction < 0.85   (≥15% width compression vs first pivot)
  - Up-break gate: same as channel (close > upper × 1.005, high ≥ upper,
    vol_ratio ≥ 1.3, range_pos ≥ 0.60, body_atr ≥ 0.35)
  - States: trigger / extended (1-5 bar lookback) / pre_breakout

Descriptive layer only — no acceptance gate, no ML, no rank model.
"""
from .schema import (  # noqa: F401
    FAMILIES,
    OUTPUT_COLUMNS,
    TRIANGLE_BREAK_VERSION,
    TriangleParams,
    empty_row,
)
