"""Zone identification + state classification for MB/BB families.

Zone construction (canonical bullish ICT mitigation/breaker):

    Walk backward from LH (inclusive). The first bar with close > open is
    the zone *origin* — its body span [open, close] is the zone box.

    Rationale: this is the last bullish bar that pushed price up to make
    the failed swing high (LH). After HH (= close > LH), the supply that
    formed near LH is "broken" and is expected to act as demand on a
    pullback. The bullish-body interpretation matches the user's strict
    spec ("LH'den düşüşü başlatan son bullish candle body").

State machine (post-HH, evaluated bar-by-bar in (hh_idx, asof_idx]):

    above_mb         — no bar overlapped the zone box (price stayed above
                       zone after HH)
    mitigation_touch — ≥1 bar overlapped the zone, no qualifying bullish
                       reclaim yet
    retest_bounce    — asof_idx is the first bar to bullishly reclaim the
                       zone (close > zone_high + body/range/vol gates)
    extended         — a retest_bounce happened earlier; asof is post-bounce

    none             — invalidated (a close in (hh_idx, asof_idx] dropped
                       below the structural low - buffer*ATR) — no row.

Invalidation low:
    MB: structural low = HL (the higher low; if price closes below it, the
        "higher low" thesis is dead).
    BB: structural low = HL (in BB this is the swept low L2; below it means
        the sweep failed to hold).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# default reclaim gates — TF-specific overrides supplied via classify_state args
DEFAULT_RETEST_BODY_ATR_MULT = 0.40
DEFAULT_RETEST_RANGE_POS_MIN = 0.55
DEFAULT_RETEST_VOL_RATIO_MIN = 1.0
DEFAULT_INVALIDATION_BUFFER_ATR = 0.30


def find_zone(df: pd.DataFrame, lh_idx: int) -> Optional[dict]:
    """Locate the bullish-body zone by walking back from LH (inclusive).

    Returns dict with origin_idx, zone_high, zone_low, or None if no
    bullish bar exists at-or-before LH within a small lookback (16 bars).
    """
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lookback_floor = max(0, lh_idx - 16)
    for j in range(lh_idx, lookback_floor - 1, -1):
        if c[j] > o[j]:
            body_high = max(o[j], c[j])
            body_low = min(o[j], c[j])
            if body_high > body_low:
                return {
                    "origin_idx": int(j),
                    "zone_high": float(body_high),
                    "zone_low": float(body_low),
                    "zone_high_wick": float(h[j]),
                }
    return None


def is_invalidated(
    df: pd.DataFrame,
    *,
    hh_idx: int,
    asof_idx: int,
    structural_low: float,
    atr: pd.Series,
    buffer_atr: float = DEFAULT_INVALIDATION_BUFFER_ATR,
) -> bool:
    """True if any close in (hh_idx, asof_idx] drops below structural_low − buffer·ATR."""
    c = df["close"].to_numpy()
    a = atr.to_numpy()
    for i in range(hh_idx + 1, asof_idx + 1):
        ai = a[i] if not math.isnan(a[i]) else 0.0
        if ai <= 0:
            continue
        if c[i] < structural_low - buffer_atr * ai:
            return True
    return False


def is_retest_bounce_bar(
    df: pd.DataFrame,
    *,
    idx: int,
    zone_high: float,
    atr: pd.Series,
    vol_sma: pd.Series,
    body_atr_mult: float = DEFAULT_RETEST_BODY_ATR_MULT,
    range_pos_min: float = DEFAULT_RETEST_RANGE_POS_MIN,
    vol_ratio_min: float = DEFAULT_RETEST_VOL_RATIO_MIN,
) -> bool:
    """Bullish reclaim gate: close>zone_high, body up, body/range/vol thresholds."""
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    a = float(atr.iat[idx]) if pd.notna(atr.iat[idx]) else 0.0
    v = float(df["volume"].iat[idx])
    vs = float(vol_sma.iat[idx]) if pd.notna(vol_sma.iat[idx]) else 0.0
    if a <= 0 or vs <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    return (
        c > zone_high
        and c > o
        and range_pos >= range_pos_min
        and body > a * body_atr_mult
        and v > vs * vol_ratio_min
    )


def classify_state(
    df: pd.DataFrame,
    *,
    hh_idx: int,
    asof_idx: int,
    zone_high: float,
    zone_low: float,
    atr: pd.Series,
    vol_sma: pd.Series,
    retest_kwargs: Optional[dict] = None,
) -> tuple[str, Optional[int], int, float]:
    """Return (state, retest_idx, touches_into_zone, deepest_low_after_hh).

    state ∈ {above_mb, mitigation_touch, retest_bounce, extended}.
    """
    retest_kwargs = retest_kwargs or {}
    if asof_idx <= hh_idx:
        # asof IS the BoS bar — no post-BoS data yet.
        return "above_mb", None, 0, float("inf")

    l_arr = df["low"].to_numpy()
    h_arr = df["high"].to_numpy()

    touches = 0
    deepest_low = float("inf")
    first_retest: Optional[int] = None
    for i in range(hh_idx + 1, asof_idx + 1):
        li = l_arr[i]
        hi = h_arr[i]
        if li < deepest_low:
            deepest_low = float(li)
        # zone overlap: bar's range box intersects zone box
        overlap = (li <= zone_high) and (hi >= zone_low)
        if overlap:
            touches += 1
            if first_retest is None and is_retest_bounce_bar(
                df, idx=i, zone_high=zone_high, atr=atr,
                vol_sma=vol_sma, **retest_kwargs,
            ):
                first_retest = i

    if first_retest is None:
        if touches == 0:
            return "above_mb", None, 0, deepest_low
        return "mitigation_touch", None, touches, deepest_low

    if first_retest == asof_idx:
        return "retest_bounce", first_retest, touches, deepest_low
    return "extended", first_retest, touches, deepest_low


def retest_kind_label(retest_depth_atr: float) -> str:
    """Categorize how deeply the zone was tested.

    >= 0.30 ATR penetration (zone_high - deepest_low) → deep_touch
    >= 0.0  but shallower                              → shallow_touch
    < 0.0   (price never reached zone_high)            → no_touch
    """
    if not (isinstance(retest_depth_atr, float) and math.isfinite(retest_depth_atr)):
        return ""
    if retest_depth_atr >= 0.30:
        return "deep_touch"
    if retest_depth_atr >= 0.0:
        return "shallow_touch"
    return "no_touch"
