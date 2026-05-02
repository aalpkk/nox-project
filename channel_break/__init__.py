"""channel_break — descriptive scanner for parallel-channel up-breaks.

v0 spec (locked 2026-05-02):
  - Close-only pivots (n=2 fractal), 4 TFs (5h/1d/1w/1M)
  - OLS fit upper (pivot highs) + lower (pivot lows)
  - Parallelism gate |Δslope|/mean(|slope|) ≤ 0.25 → fail routes to
    pending_triangle parquet (triangle workstream)
  - Width %3-25, alternating swing-touches ≥2/2 (3+/3+ → tier_a)
  - Slope class: asc / desc / flat (flat allowed; horizontal_base
    distinguishes itself via narrow-box compression, channel via
    oscillation)
  - Long-only up-break: close > upper × 1.005 ∧ high ≥ upper
  - Gates: vol_ratio_20 ≥ 1.3 ∧ range_pos ≥ 0.60 ∧ body_atr ≥ 0.35
  - States: trigger / extended (1-5 bar lookback) / pre_breakout

Descriptive layer only — no acceptance gate, no ML, no rank model.
Mirrors mb_scanner architecture (engine + html_report).
"""
from .schema import (  # noqa: F401
    CHANNEL_BREAK_VERSION,
    FAMILIES,
    OUTPUT_COLUMNS,
    PENDING_TRIANGLE_COLUMNS,
    ChannelParams,
    empty_row,
    empty_pending_row,
)
