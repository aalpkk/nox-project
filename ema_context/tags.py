"""EMA context tag assignment — Phase 0 LOCKED.

Tag breakpoint policy (memory/ema_context_phase0_spec.md):
- Phase 0 first run computes breakpoints from observed distribution
  (compression p33, extension |distance| p90, reclaim 3-bar window).
- Pinned to BREAKPOINTS_VERSION="v0.0" in parquet metadata.
- DONDURULUR — Phase 0 sonrası iterate yok.
"""
from __future__ import annotations

import pandas as pd

BREAKPOINTS_VERSION: str = "v0.0"

DEFAULT_COMPRESSION_PERCENTILE: float = 0.33
DEFAULT_EXTENSION_PERCENTILE: float = 0.90
DEFAULT_RECLAIM_WINDOW_BARS: int = 3


def fit_breakpoints(
    features: pd.DataFrame,
    *,
    compression_pct: float = DEFAULT_COMPRESSION_PERCENTILE,
    extension_pct: float = DEFAULT_EXTENSION_PERCENTILE,
) -> dict:
    """Compute tag breakpoints from observed feature distribution (first run)."""
    sw = features["ema_stack_width_atr"].dropna()
    dist = features["ema_distance_21_atr"].dropna().abs()

    return {
        "version": BREAKPOINTS_VERSION,
        "compression_tight_max": float(sw.quantile(compression_pct)),
        "compression_percentile": compression_pct,
        "overextended_min_abs": float(dist.quantile(extension_pct)),
        "extension_percentile": extension_pct,
        "reclaim_window_bars": DEFAULT_RECLAIM_WINDOW_BARS,
    }


def _reclaim_fresh_per_ticker(features: pd.DataFrame, window: int) -> pd.Series:
    """True if ema21_reclaim fired within last `window` bars (inclusive of today)."""
    out = pd.Series(False, index=features.index)
    for _, idx in features.groupby("ticker", sort=False).indices.items():
        sub = features.loc[idx].sort_values("date")
        rec = sub["ema21_reclaim"].fillna(False).astype(bool)
        rolled = rec.rolling(window, min_periods=1).max().astype(bool)
        out.loc[sub.index] = rolled.values
    return out


def assign_tags(features: pd.DataFrame, breakpoints: dict) -> pd.DataFrame:
    """Apply tags using frozen breakpoints. Returns features + tag cols."""
    out = features.copy()

    out["tag_compression"] = "compression_normal"
    out.loc[
        out["ema_stack_width_atr"] <= breakpoints["compression_tight_max"],
        "tag_compression",
    ] = "compression_tight"

    out["tag_extension"] = "aligned"
    out.loc[
        out["ema_distance_21_atr"].abs() >= breakpoints["overextended_min_abs"],
        "tag_extension",
    ] = "overextended"

    fresh = _reclaim_fresh_per_ticker(out, breakpoints["reclaim_window_bars"])
    out["tag_reclaim"] = "no_reclaim"
    out.loc[fresh, "tag_reclaim"] = "reclaim_fresh"

    stack_map = {"bull": "bull_stack", "bear": "bear_stack", "mixed": "mixed_stack"}
    out["tag_stack_state"] = out["ema_stack_state"].map(stack_map)

    out["tag_breakpoints_version"] = breakpoints["version"]
    return out
