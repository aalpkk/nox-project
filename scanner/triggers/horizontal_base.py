"""Horizontal base / box breakout trigger (daily bars).

Snapshot mode: emits ONE row per ticker representing its current state at
``asof`` — pre_breakout / trigger / extended — relative to the most recent
squeeze episode.

V1.2 (2026-04-29) — daily-discipline upgrade:
    * Box uses DUAL resistance: `robust = q95(high)` and `hard = max(high)`.
      A confirmed breakout requires `close > robust*1.005 AND high >= hard`,
      so single wicks don't define the trigger but the print also can't claim
      the line while still inside it.
    * Box width hard-rejected at >25%; horizontality reject at slope >0.15%/day
      on both the close-line AND the resistance-line (avoids ascending-channel
      masquerade).
    * Trigger gate tightened: vol_ratio>=1.3, range_pos>=0.60, close>SMA20,
      close>VWAP20 OR close>VWAP60.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..schema import (
    FEATURE_VERSION,
    SCANNER_VERSION,
    SCHEMA_VERSION,
)


FAMILY = "horizontal_base"

# ------------------------------------------------------------------ params V1.2
BB_LENGTH = 20
BB_MULT = 2.0
BB_WIDTH_THRESH = 0.65
ATR_LENGTH = 14
ATR_SQUEEZE_LENGTH = 10
ATR_SMA_LENGTH = 20
ATR_SQUEEZE_RATIO = 0.95           # V1.2.2: NO LONGER A GATE. ATR ratio + decline
                                   # are scoring features only. The squeeze gate is
                                   # BB-width alone — see _compute_indicators.
MIN_SQUEEZE_BARS = 5
MAX_SQUEEZE_BARS = 40
# V1.3.1: trigger gate body floor lowered from 0.65 → 0.35 after the body
# ablation showed body_atr ∈ [0.35, 0.65) carries PF_5d 2.19 (vs 1.40 strict).
# The old 0.65 line is preserved as a TAG boundary in BODY_STRICT_THRESH —
# anything below 0.35 is still rejected (doji/weak-body never validated).
IMPULSE_ATR_MULT = 0.35
BODY_STRICT_THRESH = 0.65   # tag boundary, NOT a gate
BODY_LARGE_THRESH = 1.05    # tag boundary, NOT a gate
VOL_SMA_LENGTH = 20
ATR_SL_MULT = 0.30
TRADING_DAYS_PER_WEEK = 5

# Dual resistance — robust (q95) vs hard (max). Breakout must clear both.
RESISTANCE_ROBUST_QUANTILE = 0.95
SUPPORT_QUANTILE = 0.10
BREAKOUT_BUFFER = 0.005           # close > robust * (1 + buffer)

# Confirmed-breakout trigger gate (V1.2.3 daily, vol 1.3 mode)
TRIG_VOL_RATIO_MIN = 1.3
TRIG_RANGE_POS_MIN = 0.60

# Horizontality — slope expressed as fraction-of-price PER DAY.
# V1.2.3: cap relaxed to 0.5%/d, tier tags emitted in scoring:
#   |slope| <= 0.15%/d → ideal (no tag)
#   0.15..0.30%/d     → "loose_base" / "loose_resistance"
#   0.30..0.50%/d     → "very_loose_base" / "very_loose_resistance"
#   > 0.50%/d         → reject
SLOPE_REJECT_PER_DAY = 0.0050
SLOPE_LOOSE_PER_DAY  = 0.0015
SLOPE_VERY_LOOSE_PER_DAY = 0.0030

# Channel width — fraction of price; >25% rejected as horizontal_base
WIDTH_PCT_REJECT = 0.25

# Pre-breakout watchlist gate
PRE_BREAKOUT_PROXIMITY_ATR = 1.5
PRE_BREAKOUT_LOOKBACK_AFTER_SQUEEZE = 7
PRE_BREAKOUT_VOL_RATIO_MAX = 0.95

# Extended classification
MAX_EXTENDED_AGE_BARS = 10
EXTENDED_HOLD_ATR_BUFFER = 0.25

# Retest_bounce (T2) — first bullish reclaim after a pullback toward the broken
# resistance, within the same squeeze cycle. T1 still emits as "trigger"; T2
# emits as "retest_bounce" on the bar it confirms. Same setup family — this is
# event multiplication, not a new geometry. (V1.2.9, 2026-04-30)
MAX_RETEST_AGE = 8
RETEST_TOLERANCE_ATR = 0.50    # how close low must come to (or below) box_top
RETEST_VOL_RATIO_MIN = 1.1     # relaxed vs trigger 1.3
RETEST_BODY_ATR_MULT = 0.40    # T2 reclaim floor; stricter than V1.3.1 trigger 0.35
RETEST_RANGE_POS_MIN = 0.55    # relaxed vs trigger 0.60

# Squeeze relevance — older squeeze episodes are not actionable today
MAX_SQUEEZE_AGE_BARS = 30


# ------------------------------------------------------------------ helpers

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _anchored_vwap(df: pd.DataFrame, anchor_idx: int, end_idx: int) -> float:
    """Cumulative VWAP from anchor_idx through end_idx (inclusive)."""
    sub = df.iloc[anchor_idx: end_idx + 1]
    v = sub["volume"]
    if sub.empty or v.sum() <= 0:
        return float("nan")
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    return float((typical * v).sum() / v.sum())


def _ytd_vwap(df: pd.DataFrame, asof_idx: int) -> float:
    """VWAP from Jan 1 of asof's year through asof_idx."""
    asof_ts = pd.Timestamp(df.index[asof_idx])
    year_start = pd.Timestamp(year=asof_ts.year, month=1, day=1)
    ts_idx = pd.DatetimeIndex(df.index)
    mask = (ts_idx >= year_start) & (ts_idx <= asof_ts)
    sub = df.loc[mask]
    if sub.empty or sub["volume"].sum() <= 0:
        return float("nan")
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    return float((typical * sub["volume"]).sum() / sub["volume"].sum())


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    sma = c.rolling(BB_LENGTH).mean()
    std = c.rolling(BB_LENGTH).std()
    df["bb_width"] = (sma + BB_MULT * std - (sma - BB_MULT * std)) / sma
    df["bb_width_sma"] = df["bb_width"].rolling(BB_LENGTH).mean()

    df["atr_sq"] = _atr(df, ATR_SQUEEZE_LENGTH)
    df["atr_sq_sma"] = df["atr_sq"].rolling(ATR_SMA_LENGTH).mean()
    df["atr_14"] = _atr(df, ATR_LENGTH)
    df["atr_pctile_120"] = df["atr_14"].rolling(120, min_periods=20).rank(pct=True)
    # ATR-decline diagnostic features (V1.2.1, NOT a gate). Two anchors so
    # a downstream score can decide which signal is more stable.
    df["atr_decline_5_pct"] = df["atr_sq"] / df["atr_sq"].shift(5) - 1.0
    df["atr_slope_10"] = (df["atr_sq"] - df["atr_sq"].shift(10)) / (
        df["atr_sq"].shift(10).replace(0, np.nan) * 10.0
    )

    df["vol_sma"] = df["volume"].rolling(VOL_SMA_LENGTH).mean()
    df["body"] = (df["close"] - df["open"]).abs()
    df["sma20"] = c.rolling(20).mean()
    df["sma20_slope"] = (df["sma20"] - df["sma20"].shift(5)) / (df["sma20"].shift(5) + 1e-12)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    vsum = df["volume"].rolling
    df["vwap10"] = pv.rolling(10).sum() / vsum(10).sum().replace(0, np.nan)
    df["vwap20"] = pv.rolling(20).sum() / vsum(20).sum().replace(0, np.nan)
    df["vwap52"] = pv.rolling(52).sum() / vsum(52).sum().replace(0, np.nan)
    df["vwap60"] = pv.rolling(60).sum() / vsum(60).sum().replace(0, np.nan)

    # V1.2.2 — gate is BB-width only. ATR signals (sq_atr / atr_decline) live
    # alongside as quality features but no longer veto a candidate squeeze.
    df["sq_bb"] = df["bb_width"] < df["bb_width_sma"] * BB_WIDTH_THRESH
    df["sq_atr"] = df["atr_sq"] < df["atr_sq_sma"] * ATR_SQUEEZE_RATIO   # diagnostic
    df["squeeze"] = df["sq_bb"].fillna(False)

    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    df["rv120_pctile"] = df["rv20"].rolling(120, min_periods=20).rank(pct=True)

    df["ret_1w"] = c.pct_change(5)
    df["ret_4w"] = c.pct_change(20)
    df["ret_12w"] = c.pct_change(60)

    df["high_252"] = df["high"].rolling(252, min_periods=20).max()
    df["false_break_below_20"] = (
        (df["close"] < df["low"].rolling(20).min().shift(1))
        .rolling(60, min_periods=1).sum()
    )
    df["up_day"] = (df["close"] > df["open"]).astype(int)
    df["down_day"] = (df["close"] < df["open"]).astype(int)
    return df


def _find_squeeze_runs(squeeze: pd.Series) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    n = len(squeeze)
    in_sq = False
    s = 0
    for i in range(n):
        v = bool(squeeze.iat[i])
        if v and not in_sq:
            s = i
            in_sq = True
        elif not v and in_sq:
            runs.append((s, i - 1, i - s))
            in_sq = False
    if in_sq:
        runs.append((s, n - 1, n - s))
    return [(s, e, L) for (s, e, L) in runs if L >= MIN_SQUEEZE_BARS]


def _box_from_squeeze(df: pd.DataFrame, sq_s: int, sq_e: int) -> tuple[float, float, float]:
    """Return (robust_resistance, hard_resistance, support).

    robust = q95(high) — wick-robust line that the breakout body must clear.
    hard   = max(high) — every prior bar's high; co-confirm so the print
             can't claim the line while still inside the channel.
    support = q10(low).
    """
    sub = df.iloc[sq_s: sq_e + 1]
    robust = float(sub["high"].quantile(RESISTANCE_ROBUST_QUANTILE))
    hard = float(sub["high"].max())
    support = float(sub["low"].quantile(SUPPORT_QUANTILE))
    return robust, hard, support


def _line_fit(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 3 or not np.isfinite(y).all():
        return (0.0, 0.0)
    x = np.arange(len(y), dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return (0.0, 0.0)
    slope = ((x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    yhat = slope * x + intercept
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (float(slope / (ym + 1e-12)), float(r2))


def _trailing_pctile(s: pd.Series, idx: int, window: int) -> Optional[float]:
    lo = max(0, idx - window + 1)
    seg = s.iloc[lo: idx + 1].dropna()
    if len(seg) < 20:
        return None
    return float((seg.iloc[-1] >= seg).mean())


def _is_breakout_bar(
    df: pd.DataFrame,
    idx: int,
    robust_resistance: float,
    hard_resistance: float,
) -> bool:
    """V1.2 confirmed-breakout daily gate.

    close > robust * (1 + BREAKOUT_BUFFER)   — clears the wick-robust line
    high  >= hard                            — and the print isn't an isolated wick
    range_position >= 0.60                   — closed in the upper part of its range
    body decisive (vs ATR_sq impulse mult)
    vol_ratio_20d >= 1.3                     — confirmed expansion
    close > SMA20                            — trend filter
    close > VWAP20 OR close > VWAP60         — multi-anchor flow filter
    """
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    vwap20 = float(df["vwap20"].iat[idx]) if pd.notna(df["vwap20"].iat[idx]) else 0.0
    vwap60 = float(df["vwap60"].iat[idx]) if pd.notna(df["vwap60"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0 or sma20 <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    vwap_ok = (vwap20 > 0 and c > vwap20) or (vwap60 > 0 and c > vwap60)
    return (
        c > robust_resistance * (1.0 + BREAKOUT_BUFFER)
        and h >= hard_resistance
        and c > o
        and range_pos >= TRIG_RANGE_POS_MIN
        and body > atr * IMPULSE_ATR_MULT
        and vol > vol_sma * TRIG_VOL_RATIO_MIN
        and c > sma20
        and vwap_ok
    )


def _is_retest_bounce_bar(
    df: pd.DataFrame,
    idx: int,
    box_top: float,
) -> bool:
    """Bullish reclaim gate for T2 (retest_bounce).

    Looser than _is_breakout_bar — same setup, but already past initial
    confirmation, so we accept slightly weaker body/vol/range_pos. Pullback
    evidence is checked separately (see _find_retest_bounce_idx).
    """
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0 or sma20 <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    return (
        c > box_top * (1.0 + BREAKOUT_BUFFER)
        and c > o
        and range_pos >= RETEST_RANGE_POS_MIN
        and body > atr * RETEST_BODY_ATR_MULT
        and vol > vol_sma * RETEST_VOL_RATIO_MIN
        and c > sma20
    )


def _find_retest_bounce_idx(
    df: pd.DataFrame,
    breakout_idx: int,
    asof_idx: int,
    box_top: float,
) -> Optional[int]:
    """First retest_bounce bar in (T1, asof], or None.

    A bar i qualifies when:
      - some bar j ∈ (T1, i] has low within RETEST_TOLERANCE_ATR ATRs of box_top
        (a real pullback toward the broken resistance, possibly an intra-bar
        wick on i itself);
      - i passes _is_retest_bounce_bar (bullish reclaim of box_top).
    """
    end = min(asof_idx, breakout_idx + MAX_RETEST_AGE)
    deepest_low = float("inf")
    for i in range(breakout_idx + 1, end + 1):
        l_i = float(df["low"].iat[i])
        if l_i < deepest_low:
            deepest_low = l_i
        atr = float(df["atr_sq"].iat[i]) if pd.notna(df["atr_sq"].iat[i]) else 0.0
        if atr <= 0:
            continue
        retest_depth_atr = (box_top - deepest_low) / atr
        if retest_depth_atr < -RETEST_TOLERANCE_ATR:
            continue
        if not _is_retest_bounce_bar(df, i, box_top):
            continue
        return i
    return None


def _classify_state(
    df: pd.DataFrame,
    asof_idx: int,
    sq_s: int,
    sq_e: int,
    box_top: float,
    hard_resistance: float,
) -> tuple[str, Optional[int]]:
    """Return (state, breakout_idx). state ∈ {pre_breakout, trigger,
    retest_bounce, extended, none}.

    `box_top` here is the robust resistance (q95 high). The hard resistance
    (max high) is a co-confirm only — it does not alter pre/extended logic.
    `breakout_idx` always refers to T1 (the original breakout); for
    retest_bounce we still need T1 to anchor box geometry and breakout-age.
    """
    breakout_idx: Optional[int] = None
    for i in range(sq_e + 1, asof_idx + 1):
        if _is_breakout_bar(df, i, box_top, hard_resistance):
            breakout_idx = i
            break

    if breakout_idx is not None:
        c_now = float(df["close"].iat[asof_idx])
        age = asof_idx - breakout_idx
        if age == 0:
            return "trigger", breakout_idx
        # T2 detection — earliest retest_bounce candidate in (T1, asof].
        # If asof is exactly that candidate, emit retest_bounce; if earlier
        # candidates exist (already past), this asof is in extended phase.
        t2_idx = _find_retest_bounce_idx(df, breakout_idx, asof_idx, box_top)
        if t2_idx == asof_idx:
            return "retest_bounce", breakout_idx
        if age > MAX_EXTENDED_AGE_BARS:
            return "none", None
        # Validate the breakout is STILL holding: close must be meaningfully above
        # box_top (not just barely, not back inside). Otherwise the breakout has
        # failed and the asof bar is no longer an actionable extended-state row.
        atr = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if c_now <= box_top + EXTENDED_HOLD_ATR_BUFFER * atr:
            return "none", None
        return "extended", breakout_idx

    if asof_idx - sq_e > PRE_BREAKOUT_LOOKBACK_AFTER_SQUEEZE:
        return "none", None

    c_now = float(df["close"].iat[asof_idx])
    atr = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
    sma20 = float(df["sma20"].iat[asof_idx]) if pd.notna(df["sma20"].iat[asof_idx]) else 0.0
    if atr <= 0 or sma20 <= 0:
        return "none", None

    proximity_ok = (box_top - c_now) <= PRE_BREAKOUT_PROXIMITY_ATR * atr
    trend_ok = c_now > sma20

    win_lo = max(0, asof_idx - 9)
    sub = df.iloc[win_lo: asof_idx + 1]
    vol_sma = float(df["vol_sma"].iat[asof_idx]) if pd.notna(df["vol_sma"].iat[asof_idx]) else 0.0
    cur_vol = float(df["volume"].iat[asof_idx])
    drying = vol_sma > 0 and cur_vol < vol_sma * PRE_BREAKOUT_VOL_RATIO_MAX
    accumulating = int(sub["up_day"].sum()) > int(sub["down_day"].sum())

    if proximity_ok and trend_ok and (drying or accumulating):
        return "pre_breakout", None
    return "none", None


def _build_row(
    *,
    state: str,
    ticker: str,
    df: pd.DataFrame,
    asof_idx: int,
    breakout_idx: Optional[int],
    sq_start: int,
    sq_end: int,
    box_top: float,
    box_bot: float,
    hard_resistance: float,
    direction: int,
) -> dict:
    """Build schema-compliant row.

    Mechanics fields use the bar most relevant to the state:
      pre_breakout : asof bar (today's snapshot)
      trigger      : asof bar (which equals breakout bar by classification)
      extended     : breakout bar (preserves the original event quality)
    Geometry fields always come from the squeeze run.
    extension_from_trigger / entry_distance_to_trigger always use asof close.
    """
    feat_idx = breakout_idx if (state == "extended" and breakout_idx is not None) else asof_idx
    bo = df.iloc[feat_idx]
    asof_bar = df.iloc[asof_idx]

    asof_ts_raw = pd.Timestamp(df.index[asof_idx])
    asof_ts = (asof_ts_raw.tz_localize("Europe/Istanbul")
               if asof_ts_raw.tz is None else asof_ts_raw)
    bar_date = (asof_ts_raw.normalize() if asof_ts_raw.tz is None
                else asof_ts_raw.tz_convert("Europe/Istanbul").normalize().tz_localize(None))
    breakout_bar_date = (
        pd.Timestamp(df.index[breakout_idx]).normalize()
        if breakout_idx is not None else pd.NaT
    )

    entry = float(bo["close"])
    c_now = float(asof_bar["close"])
    atr_14 = float(bo["atr_14"]) if pd.notna(bo["atr_14"]) else float("nan")
    atr_sq = float(bo["atr_sq"]) if pd.notna(bo["atr_sq"]) else atr_14
    invalidation = box_bot - ATR_SL_MULT * atr_sq if direction > 0 else box_top + ATR_SL_MULT * atr_sq
    initial_risk_pct = ((entry - invalidation) / entry if direction > 0
                        else (invalidation - entry) / entry)

    box_range = box_top - box_bot
    high_d = float(bo["high"])
    low_d = float(bo["low"])
    open_d = float(bo["open"])
    rng_d = high_d - low_d
    range_position = (entry - low_d) / rng_d if rng_d > 0 else 0.5
    body_pct = abs(entry - open_d) / open_d if open_d > 0 else 0.0
    vol_ratio = (float(bo["volume"] / bo["vol_sma"])
                 if (pd.notna(bo["vol_sma"]) and bo["vol_sma"] > 0) else float("nan"))

    prev_close = float(df["close"].iat[feat_idx - 1]) if feat_idx > 0 else open_d
    gap_pct = (open_d - prev_close) / prev_close if prev_close > 0 else 0.0
    gap_atr = (open_d - prev_close) / atr_sq if atr_sq and atr_sq > 0 else float("nan")
    day_return = (entry / prev_close - 1.0) if prev_close > 0 else 0.0
    day_return_atr = (entry - prev_close) / atr_sq if atr_sq and atr_sq > 0 else float("nan")

    base_window = df.iloc[sq_start: sq_end + 1]
    inside_close = base_window["close"].between(box_bot, box_top, inclusive="both").mean()
    base_slope, base_r2 = _line_fit(base_window["close"].values)
    res_slope, _ = _line_fit(base_window["high"].values)
    upper_touch = int((base_window["high"] >= box_top - 0.10 * (atr_sq or 0)).sum())
    lower_touch = int((base_window["low"] <= box_bot + 0.10 * (atr_sq or 0)).sum())
    # ATR-compression aggregate over the base window — mean(atr_sq / atr_sq_sma).
    # Lower = tighter range volatility = higher squeeze quality.
    atr_ratio_base = base_window["atr_sq"] / base_window["atr_sq_sma"].replace(0, np.nan)
    atr_ratio_mean = float(atr_ratio_base.mean()) if atr_ratio_base.notna().any() else float("nan")

    width_pct = box_range / entry if entry > 0 else float("nan")
    width_atr = box_range / atr_sq if atr_sq and atr_sq > 0 else float("nan")
    width_pctile = _trailing_pctile(df["bb_width"], feat_idx, 252)

    # --- retest_bounce features ----------------------------------------------
    # breakout_age: bars from T1 to asof (0 for trigger; positive for
    # retest_bounce/extended; NaN/0 when there's no T1 yet — pre_breakout).
    if breakout_idx is not None:
        breakout_age = int(asof_idx - breakout_idx)
    else:
        breakout_age = 0  # Int16 column; sentinel 0 reads cleanly when state==pre_breakout
    retest_depth_atr = float("nan")
    retest_close_position = float("nan")
    retest_vol_pattern = float("nan")
    retest_kind = ""
    if state == "retest_bounce" and breakout_idx is not None:
        atr_t2 = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_t2 > 0:
            deepest_low = float("inf")
            for j in range(breakout_idx + 1, asof_idx + 1):
                l_j = float(df["low"].iat[j])
                if l_j < deepest_low:
                    deepest_low = l_j
            if np.isfinite(deepest_low):
                retest_depth_atr = (box_top - deepest_low) / atr_t2
            retest_close_position = (c_now - box_top) / atr_t2
        # vol pattern: T2 vol vs mean vol in (T1, T2) — exclusive both ends.
        # Tells us if the bounce had a volume spike against a quieter pullback.
        if asof_idx > breakout_idx + 1:
            mid_vol = df["volume"].iloc[breakout_idx + 1: asof_idx].mean()
            t2_vol = float(df["volume"].iat[asof_idx])
            if pd.notna(mid_vol) and mid_vol > 0:
                retest_vol_pattern = t2_vol / float(mid_vol)
        # taxonomy — coarse 3-bucket split. depth sign convention:
        #   depth > 0   → low went below box_top (real retest)
        #   depth <= 0  → low stayed above box_top (drift-up continuation)
        # 3y validation showed deep_touch R_20d=+0.61 vs no_touch R_20d=+0.09
        # — same family event but materially different forward path. Don't gate;
        # tag for ML and downstream slicing.
        if pd.notna(retest_depth_atr):
            if retest_depth_atr >= 0.30:
                retest_kind = "deep_touch"
            elif retest_depth_atr >= 0.0:
                retest_kind = "shallow_touch"
            else:
                retest_kind = "no_touch"

    # body_atr / body_class — trigger-bar geometry of the breakout body.
    # T1 (breakout_idx) is the relevant bar: trigger uses T1=asof, retest_bounce
    # and extended carry T1 from earlier in the cycle. pre_breakout has no T1.
    body_atr_val = float("nan")
    body_class = ""
    if breakout_idx is not None:
        t1_atr_sq = float(df["atr_sq"].iat[breakout_idx]) if pd.notna(df["atr_sq"].iat[breakout_idx]) else 0.0
        if t1_atr_sq > 0:
            t1_body = abs(float(df["close"].iat[breakout_idx]) - float(df["open"].iat[breakout_idx]))
            body_atr_val = t1_body / t1_atr_sq
            if body_atr_val >= BODY_LARGE_THRESH:
                body_class = "large_body"
            elif body_atr_val >= BODY_STRICT_THRESH:
                body_class = "strict_body"
            elif body_atr_val >= IMPULSE_ATR_MULT:
                body_class = "mid_body"

    trigger_lvl = box_top if direction > 0 else box_bot
    breakout_pct = (entry - trigger_lvl) / trigger_lvl if trigger_lvl > 0 else 0.0
    breakout_atr_v = ((entry - trigger_lvl) / atr_sq
                      if atr_sq and atr_sq > 0 else float("nan"))
    extension_from_trigger = (c_now / trigger_lvl - 1.0) if trigger_lvl > 0 else 0.0
    entry_distance_to_trigger = ((trigger_lvl - c_now) / max(c_now, 1e-9))

    vwap_ytd = _ytd_vwap(df, asof_idx)
    vwap_base = _anchored_vwap(df, sq_start, asof_idx)

    row: dict = {
        # identity
        "ticker": ticker,
        "bar_date": bar_date,
        "setup_family": FAMILY,
        "signal_type": "long_breakout" if direction > 0 else "short_breakdown",
        "signal_state": state,
        "breakout_bar_date": breakout_bar_date,
        # audit
        "as_of_ts": asof_ts,
        "data_frequency": "1d",
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "scanner_version": SCANNER_VERSION,
        # contract
        "family__trigger_level": np.float32(trigger_lvl),
        "entry_reference_price": np.float32(entry),
        "invalidation_level": np.float32(invalidation),
        "initial_risk_pct": np.float32(initial_risk_pct),
        # common — mechanics
        "common__breakout_pct": np.float32(breakout_pct),
        "common__breakout_atr": np.float32(breakout_atr_v) if pd.notna(breakout_atr_v) else np.float32("nan"),
        "common__range_position": np.float32(range_position),
        "common__body_pct": np.float32(body_pct),
        "common__volume_ratio_20": np.float32(vol_ratio),
        "common__extension_from_trigger": np.float32(extension_from_trigger),
        "common__entry_distance_to_trigger": np.float32(entry_distance_to_trigger),
        "common__gap_pct": np.float32(gap_pct),
        "common__gap_atr": np.float32(gap_atr),
        "common__day_return": np.float32(day_return),
        "common__day_return_atr": np.float32(day_return_atr) if pd.notna(day_return_atr) else np.float32("nan"),
        # common — vol/liq
        "common__atr_14": np.float32(atr_14),
        "common__atr_pct": np.float32(atr_14 / entry) if entry > 0 else np.float32("nan"),
        "common__realized_vol_20": np.float32(bo["rv20"]) if pd.notna(bo["rv20"]) else np.float32("nan"),
        "common__realized_vol_pctile_120": np.float32(bo["rv120_pctile"])
            if pd.notna(bo["rv120_pctile"]) else np.float32("nan"),
        "common__volume": np.float64(bo["volume"]),
        "common__turnover": np.float64(bo["volume"] * entry),
        "common__liquidity_score": np.float32(np.log1p(bo["volume"] * entry)),
        "common__risk_pct_score": np.float32(initial_risk_pct),
        # common — trend / VWAP (multi-anchor)
        "common__close_vs_sma20": np.float32(entry / bo["sma20"] - 1.0) if pd.notna(bo["sma20"]) else np.float32("nan"),
        "common__sma20_slope": np.float32(bo["sma20_slope"]) if pd.notna(bo["sma20_slope"]) else np.float32("nan"),
        "common__close_vs_vwap10": np.float32(entry / bo["vwap10"] - 1.0) if pd.notna(bo["vwap10"]) else np.float32("nan"),
        "common__close_vs_vwap52": np.float32(entry / bo["vwap52"] - 1.0) if pd.notna(bo["vwap52"]) else np.float32("nan"),
        "common__close_vs_vwap_ytd": np.float32(c_now / vwap_ytd - 1.0)
            if (vwap_ytd == vwap_ytd and vwap_ytd > 0) else np.float32("nan"),
        "common__close_vs_vwap_base": np.float32(c_now / vwap_base - 1.0)
            if (vwap_base == vwap_base and vwap_base > 0) else np.float32("nan"),
        "common__vwap10_vs_vwap52": np.float32(bo["vwap10"] / bo["vwap52"] - 1.0)
            if (pd.notna(bo["vwap10"]) and pd.notna(bo["vwap52"]) and bo["vwap52"] > 0) else np.float32("nan"),
        "common__extension_from_sma20": np.float32(entry / bo["sma20"] - 1.0) if pd.notna(bo["sma20"]) else np.float32("nan"),
        "common__extension_from_vwap52": np.float32(entry / bo["vwap52"] - 1.0) if pd.notna(bo["vwap52"]) else np.float32("nan"),
        # common — momentum / RS / regime (engine fills cross-sectional)
        "common__ret_1w": np.float32(bo["ret_1w"]) if pd.notna(bo["ret_1w"]) else np.float32("nan"),
        "common__ret_4w": np.float32(bo["ret_4w"]) if pd.notna(bo["ret_4w"]) else np.float32("nan"),
        "common__ret_12w": np.float32(bo["ret_12w"]) if pd.notna(bo["ret_12w"]) else np.float32("nan"),
        "common__rs_20d": np.float32("nan"),
        "common__rs_60d": np.float32("nan"),
        "common__rs_pctile_120": np.float32("nan"),
        "common__rs_pctile_252": np.float32("nan"),
        "common__rs_dist_to_252_high": np.float32(entry / bo["high_252"] - 1.0)
            if (pd.notna(bo["high_252"]) and bo["high_252"] > 0) else np.float32("nan"),
        "common__sector_rs_20d": np.float32("nan"),
        "common__market_trend_score": np.float32("nan"),
        "common__market_breadth_pct_above_sma20": np.float32("nan"),
        "common__market_vol_regime": np.float32("nan"),
        "common__index_ret_5d": np.float32("nan"),
        "common__index_ret_20d": np.float32("nan"),
        # family — geometry
        "family__base_lookback_weeks": np.int16(round((sq_end - sq_start + 1) / TRADING_DAYS_PER_WEEK)),
        "family__channel_high": np.float32(box_top),
        "family__channel_low": np.float32(box_bot),
        "family__channel_mid": np.float32((box_top + box_bot) / 2),
        "family__channel_width_pct": np.float32(width_pct),
        "family__channel_width_atr": np.float32(width_atr),
        "family__channel_width_pctile_252": np.float32(width_pctile)
            if width_pctile is not None else np.float32("nan"),
        "family__base_duration_weeks": np.float32((sq_end - sq_start + 1) / TRADING_DAYS_PER_WEEK),
        "family__upper_touch_count": np.int16(upper_touch),
        "family__lower_touch_count": np.int16(lower_touch),
        "family__touch_balance": np.float32(
            (upper_touch - lower_touch) / max(upper_touch + lower_touch, 1)
        ),
        "family__close_position_in_base": np.float32(
            (c_now - box_bot) / box_range if box_range > 0 else 0.5
        ),
        "family__prebreakout_distance_to_high": np.float32(
            (box_top - float(df["close"].iat[sq_end])) / box_top if box_top > 0 else float("nan")
        ),
        "family__bars_since_base_start": np.int16(asof_idx - sq_start),
        "family__volume_dryup_ratio": np.float32(
            base_window["volume"].mean() / df["vol_sma"].iat[sq_end]
        ) if (pd.notna(df["vol_sma"].iat[sq_end]) and df["vol_sma"].iat[sq_end] > 0) else np.float32("nan"),
        "family__volume_dryup_pctile": np.float32("nan"),
        "family__false_break_below_count": np.int16(
            int(bo["false_break_below_20"]) if pd.notna(bo["false_break_below_20"]) else 0
        ),
        "family__inside_base_close_ratio": np.float32(inside_close),
        "family__base_slope": np.float32(base_slope),
        "family__base_r2": np.float32(base_r2),
        "family__resistance_slope": np.float32(res_slope),
        "family__hard_resistance": np.float32(hard_resistance),
        "family__atr_decline_5_pct": np.float32(bo["atr_decline_5_pct"])
            if pd.notna(bo["atr_decline_5_pct"]) else np.float32("nan"),
        "family__atr_slope_10": np.float32(bo["atr_slope_10"])
            if pd.notna(bo["atr_slope_10"]) else np.float32("nan"),
        "family__atr_ratio_mean_base": np.float32(atr_ratio_mean),
        "family__days_since_last_pivot": np.int16(asof_idx - sq_end),
        # retest_bounce — populated only when state == retest_bounce; NaN/0 elsewhere.
        # breakout_age is also meaningful for extended (=asof - T1).
        "family__breakout_age": np.int16(breakout_age),
        "family__retest_depth_atr": np.float32(retest_depth_atr) if pd.notna(retest_depth_atr) else np.float32("nan"),
        "family__retest_close_position": np.float32(retest_close_position) if pd.notna(retest_close_position) else np.float32("nan"),
        "family__retest_vol_pattern": np.float32(retest_vol_pattern) if pd.notna(retest_vol_pattern) else np.float32("nan"),
        "family__retest_kind": retest_kind,
        "family__body_atr": np.float32(body_atr_val) if pd.notna(body_atr_val) else np.float32("nan"),
        "family__body_class": body_class,
    }
    return row


def detect(daily_df: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> list[dict]:
    """Snapshot detection: 0 or 1 row per ticker for the latest squeeze episode.

    Parameters
    ----------
    daily_df : DataFrame indexed by date with columns ['open','high','low','close','volume']
        and a 'ticker' attr (`df.attrs['ticker']`).
    asof : as-of bar; defaults to the last bar in daily_df.

    Returns
    -------
    list[dict] : 0 or 1 row with `signal_state` ∈ {pre_breakout, trigger, extended}.
    """
    ticker = daily_df.attrs.get("ticker", "?")
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)

    if asof is None:
        asof_idx = len(df) - 1
    else:
        asof_ts = pd.Timestamp(asof).normalize()
        ts_idx = pd.DatetimeIndex(df.index).normalize()
        mask = ts_idx <= asof_ts
        if not mask.any():
            return []
        asof_idx = int(np.flatnonzero(mask)[-1])

    if asof_idx < BB_LENGTH * 2:
        return []

    runs = _find_squeeze_runs(df["squeeze"].iloc[: asof_idx + 1])
    if not runs:
        return []
    sq_s, sq_e, _ = runs[-1]
    # Squeeze must be recent enough to be actionable today.
    if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
        return []

    base_window = df.iloc[sq_s: sq_e + 1]
    # Close-line slope (per-day, fraction of price)
    base_slope, _ = _line_fit(base_window["close"].values)
    if abs(base_slope) > SLOPE_REJECT_PER_DAY:
        return []
    # Resistance-line slope — rejects parallel-ascending channels masquerading
    # as horizontal bases. (Pure-resistance-flat + rising lows would be an
    # ascending_triangle; that family is not implemented yet, so it is dropped.)
    res_slope, _ = _line_fit(base_window["high"].values)
    if abs(res_slope) > SLOPE_REJECT_PER_DAY:
        return []

    box_top, hard_resistance, box_bot = _box_from_squeeze(df, sq_s, sq_e)
    if not (box_top > box_bot > 0):
        return []

    # Width hard-reject (>25% of price)
    asof_close = float(df["close"].iat[asof_idx])
    width_pct = (box_top - box_bot) / asof_close if asof_close > 0 else float("inf")
    if width_pct > WIDTH_PCT_REJECT:
        return []

    state, breakout_idx = _classify_state(
        df, asof_idx, sq_s, sq_e, box_top, hard_resistance,
    )
    if state == "none":
        return []

    row = _build_row(
        state=state, ticker=ticker, df=df, asof_idx=asof_idx,
        breakout_idx=breakout_idx, sq_start=sq_s, sq_end=sq_e,
        box_top=box_top, box_bot=box_bot,
        hard_resistance=hard_resistance, direction=+1,
    )
    return [row]
