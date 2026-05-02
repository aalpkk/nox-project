"""Bullish Mitigation Block (ICT/SMC) trigger — 1h bars.

Strict Inner-Circle-Trader bullish MB definition:

  1. Identify a swing-high → swing-low impulse-down leg via n-bar fractal pivots.
  2. The LAST bearish bar (close < open) inside that leg is the MB *origin*.
     The MB *zone* is the open/close span of that bar (zone_kind='body'),
     optionally extended to the wick (zone_kind='body_plus_wick').
  3. Confirm a Break-of-Structure: within MAX_BOS_LAG bars after the impulse
     low, some bar closes above the swing-high that started the impulse-down.
     That bar's index = BoS_idx; zone is now *armed*.
  4. Walk forward to as-of:
        * `zone_armed`        — no bar in (BoS, asof] has overlapped the zone.
        * `mitigation_touch`  — ≥1 bar overlapped the zone but no bullish
                                reclaim has confirmed yet.
        * `retest_bounce`     — as-of bar is the first bar to bullishly reclaim
                                the zone after a touch (close>zone_high, body
                                up, range_pos and vol gates).
        * `extended`          — a retest_bounce already happened earlier; as-of
                                is post-bounce and zone has held.
  5. Invalidation: if any close in (BoS, asof] falls below
     `zone_low - INVALIDATION_BUFFER_ATR * atr`, the zone is structurally void
     and we DO NOT emit a row — even if the asof bar would otherwise classify.

Snapshot mode (V1.4.0): one row per ticker for the most recent armed,
non-invalidated zone within MAX_ZONE_AGE_BARS of asof. If multiple zones are
armed, the most recent BoS wins.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..schema import (
    FEATURE_VERSION,
    SCANNER_VERSION,
    SCHEMA_VERSION,
)


FAMILY = "mitigation_block"

# ---------------------------------------------------------------- params V1.4.0
PIVOT_N = 2                       # fractal half-window; pivot confirms at i+n
IMPULSE_LEG_MIN_BARS = 4
IMPULSE_LEG_MAX_BARS = 30
IMPULSE_LEG_MIN_ATR = 1.0         # leg drop must be ≥ 1 ATR (filters noise)
MAX_BOS_LAG = 25                  # bars after swing_low for BoS to confirm
MAX_ZONE_AGE_BARS = 120           # 1h: ≈ 3 trading weeks; older = stale
RETEST_BODY_ATR_MULT = 0.40
RETEST_RANGE_POS_MIN = 0.55
RETEST_VOL_RATIO_MIN = 1.0        # modest — bounce volume often subdued
INVALIDATION_BUFFER_ATR = 0.30
ATR_LENGTH = 14
ATR_SQUEEZE_LENGTH = 10
ATR_SMA_LENGTH = 20
VOL_SMA_LENGTH = 20
SMA_LENGTH = 20
MIN_HISTORY = max(60, IMPULSE_LEG_MAX_BARS + MAX_BOS_LAG + MAX_ZONE_AGE_BARS // 2)


# ---------------------------------------------------------------- helpers

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["atr_sq"] = _atr(df, ATR_SQUEEZE_LENGTH)
    df["atr_sq_sma"] = df["atr_sq"].rolling(ATR_SMA_LENGTH).mean()
    df["atr_14"] = _atr(df, ATR_LENGTH)
    df["atr_pctile_120"] = df["atr_14"].rolling(120, min_periods=20).rank(pct=True)
    df["vol_sma"] = df["volume"].rolling(VOL_SMA_LENGTH).mean()
    df["sma20"] = c.rolling(SMA_LENGTH).mean()
    df["sma20_slope"] = (df["sma20"] - df["sma20"].shift(5)) / (df["sma20"].shift(5) + 1e-12)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    df["vwap10"] = pv.rolling(10).sum() / df["volume"].rolling(10).sum().replace(0, np.nan)
    df["vwap20"] = pv.rolling(20).sum() / df["volume"].rolling(20).sum().replace(0, np.nan)
    df["vwap52"] = pv.rolling(52).sum() / df["volume"].rolling(52).sum().replace(0, np.nan)

    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    df["rv120_pctile"] = df["rv20"].rolling(120, min_periods=20).rank(pct=True)

    df["ret_1w"] = c.pct_change(5)
    df["ret_4w"] = c.pct_change(20)
    df["ret_12w"] = c.pct_change(60)
    df["high_252"] = df["high"].rolling(252, min_periods=20).max()
    return df


def _find_pivots(df: pd.DataFrame, n: int, end_idx: int) -> tuple[list[int], list[int]]:
    """Return (swing_high_indices, swing_low_indices) confirmed by bar end_idx.

    A pivot at bar i is confirmed at bar i+n. We require i+n <= end_idx.
    """
    highs: list[int] = []
    lows: list[int] = []
    h = df["high"].values
    l = df["low"].values
    last_confirmable = end_idx - n
    for i in range(n, last_confirmable + 1):
        win_h = h[i - n: i + n + 1]
        win_l = l[i - n: i + n + 1]
        if h[i] >= win_h.max() and (win_h == h[i]).sum() == 1:
            highs.append(i)
        if l[i] <= win_l.min() and (win_l == l[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def _find_mb_candidates(
    df: pd.DataFrame,
    swing_highs: list[int],
    swing_lows: list[int],
    asof_idx: int,
) -> list[dict]:
    """Find all impulse-down legs and their bullish-MB candidates.

    Returns a list of dicts with: origin_idx, swing_high_idx, swing_high_price,
    swing_low_idx, swing_low_price, zone_high (body), zone_low (body),
    zone_high_wick, zone_low_wick.
    """
    if not swing_highs or not swing_lows:
        return []
    cands: list[dict] = []
    sl_arr = np.array(swing_lows)
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr_sq"].values
    for sh_idx in swing_highs:
        # find first swing_low strictly after sh_idx
        pos = sl_arr.searchsorted(sh_idx, side="right")
        if pos >= len(sl_arr):
            continue
        sl_idx = int(sl_arr[pos])
        leg_len = sl_idx - sh_idx
        if leg_len < IMPULSE_LEG_MIN_BARS or leg_len > IMPULSE_LEG_MAX_BARS:
            continue
        # leg drop must be material
        a = atr[sl_idx] if not math.isnan(atr[sl_idx]) else 0.0
        if a <= 0:
            continue
        leg_drop_atr = (h[sh_idx] - l[sl_idx]) / a
        if leg_drop_atr < IMPULSE_LEG_MIN_ATR:
            continue
        # find last bearish bar in [sh_idx, sl_idx] (inclusive)
        origin_idx = -1
        for j in range(sl_idx, sh_idx - 1, -1):
            if c[j] < o[j]:
                origin_idx = j
                break
        if origin_idx < 0:
            continue  # no bearish bar in leg — atypical, skip
        body_high = max(o[origin_idx], c[origin_idx])
        body_low = min(o[origin_idx], c[origin_idx])
        if body_high <= body_low:
            continue
        cands.append({
            "origin_idx": origin_idx,
            "swing_high_idx": sh_idx,
            "swing_high_price": float(h[sh_idx]),
            "swing_low_idx": sl_idx,
            "swing_low_price": float(l[sl_idx]),
            "zone_high_body": float(body_high),
            "zone_low_body": float(body_low),
            "zone_high_wick": float(h[origin_idx]),
            "zone_low_wick": float(l[origin_idx]),
            "leg_drop_atr": float(leg_drop_atr),
        })
    return cands


def _confirm_bos(
    df: pd.DataFrame,
    cand: dict,
    asof_idx: int,
) -> Optional[int]:
    """Return BoS_idx (first bar after swing_low closing > swing_high_price),
    within MAX_BOS_LAG bars and at or before asof. None if not confirmed.
    """
    sh_price = cand["swing_high_price"]
    sl_idx = cand["swing_low_idx"]
    end = min(asof_idx, sl_idx + MAX_BOS_LAG)
    c = df["close"].values
    for i in range(sl_idx + 1, end + 1):
        if c[i] > sh_price:
            return i
    return None


def _is_invalidated(
    df: pd.DataFrame,
    bos_idx: int,
    asof_idx: int,
    zone_low: float,
) -> bool:
    """True if any close in (BoS, asof] dropped below zone_low - buffer*ATR."""
    c = df["close"].values
    atr = df["atr_sq"].values
    for i in range(bos_idx + 1, asof_idx + 1):
        a = atr[i] if not math.isnan(atr[i]) else 0.0
        if a <= 0:
            continue
        if c[i] < zone_low - INVALIDATION_BUFFER_ATR * a:
            return True
    return False


def _is_retest_bounce_bar(
    df: pd.DataFrame,
    idx: int,
    zone_high: float,
) -> bool:
    """Bullish reclaim gate (T2). Mirrors horizontal_base retest gate."""
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    return (
        c > zone_high
        and c > o
        and range_pos >= RETEST_RANGE_POS_MIN
        and body > atr * RETEST_BODY_ATR_MULT
        and vol > vol_sma * RETEST_VOL_RATIO_MIN
    )


def _classify_state(
    df: pd.DataFrame,
    cand: dict,
    bos_idx: int,
    asof_idx: int,
    zone_high: float,
    zone_low: float,
) -> tuple[str, Optional[int], int, float]:
    """Return (state, retest_idx, touches_into_zone, deepest_low_after_bos).

    state ∈ {zone_armed, mitigation_touch, retest_bounce, extended, none}.
    retest_idx = idx of the first qualifying T2 bar in (BoS, asof], or None.
    touches_into_zone = count of bars i ∈ (BoS, asof] with low_i ≤ zone_high
                        AND high_i ≥ zone_low (overlap with the zone box).
    deepest_low_after_bos = min low in (BoS, asof] (= +inf if no bars).
    """
    if asof_idx <= bos_idx:
        # as-of is exactly the BoS bar — zone freshly armed, no post-BoS bars yet
        return "zone_armed", None, 0, float("inf")

    l_arr = df["low"].values
    h_arr = df["high"].values

    touches = 0
    deepest_low = float("inf")
    first_retest = None
    for i in range(bos_idx + 1, asof_idx + 1):
        li = l_arr[i]
        hi = h_arr[i]
        if li < deepest_low:
            deepest_low = float(li)
        # zone overlap: bar's range intersects the zone box
        overlap = (li <= zone_high) and (hi >= zone_low)
        if overlap:
            touches += 1
            # only consider retest_bounce ON or AFTER a touch bar (or on the
            # touch bar itself if the bar wicks down then closes back above)
            if first_retest is None and _is_retest_bounce_bar(df, i, zone_high):
                first_retest = i

    if first_retest is None:
        # never qualified — zone_armed if no touches, else mitigation_touch
        if touches == 0:
            return "zone_armed", None, touches, deepest_low
        return "mitigation_touch", None, touches, deepest_low

    if first_retest == asof_idx:
        return "retest_bounce", first_retest, touches, deepest_low
    return "extended", first_retest, touches, deepest_low


# ---------------------------------------------------------------- row builder

def _safe(v, fallback=float("nan")):
    return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else fallback


def _build_row(
    *,
    state: str,
    ticker: str,
    df: pd.DataFrame,
    asof_idx: int,
    cand: dict,
    bos_idx: int,
    retest_idx: Optional[int],
    zone_high: float,
    zone_low: float,
    zone_kind: str,
    touches: int,
    deepest_low_after_bos: float,
) -> dict:
    feat_idx = retest_idx if (state == "retest_bounce" and retest_idx is not None) else asof_idx
    bo = df.iloc[feat_idx]
    asof_bar = df.iloc[asof_idx]

    asof_ts_raw = pd.Timestamp(df.index[asof_idx])
    asof_ts = (asof_ts_raw.tz_localize("Europe/Istanbul")
               if asof_ts_raw.tz is None else asof_ts_raw)
    bar_date_normalized = (asof_ts_raw.normalize() if asof_ts_raw.tz is None
                           else asof_ts_raw.tz_convert("Europe/Istanbul").normalize().tz_localize(None))
    bos_bar_date = pd.Timestamp(df.index[bos_idx])
    if bos_bar_date.tz is not None:
        bos_bar_date = bos_bar_date.tz_convert("Europe/Istanbul").normalize().tz_localize(None)
    else:
        bos_bar_date = bos_bar_date.normalize()

    entry = float(bo["close"])
    c_now = float(asof_bar["close"])
    atr_14 = float(bo["atr_14"]) if pd.notna(bo["atr_14"]) else float("nan")
    atr_sq = float(bo["atr_sq"]) if pd.notna(bo["atr_sq"]) else atr_14
    invalidation = zone_low - INVALIDATION_BUFFER_ATR * (atr_sq if atr_sq > 0 else 0.0)
    initial_risk_pct = (entry - invalidation) / entry if entry > 0 else float("nan")

    # zone geometry
    zone_width = zone_high - zone_low
    zone_width_pct = zone_width / entry if entry > 0 else float("nan")
    zone_width_atr = zone_width / atr_sq if atr_sq > 0 else float("nan")

    # mechanics on feat bar
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
    gap_atr = (open_d - prev_close) / atr_sq if atr_sq > 0 else float("nan")
    day_return = (entry / prev_close - 1.0) if prev_close > 0 else 0.0
    day_return_atr = (entry - prev_close) / atr_sq if atr_sq > 0 else float("nan")

    # BoS / impulse provenance
    bos_atr = float(df["atr_sq"].iat[bos_idx]) if pd.notna(df["atr_sq"].iat[bos_idx]) else atr_sq
    bos_close = float(df["close"].iat[bos_idx])
    bos_distance_atr = ((bos_close - cand["swing_high_price"]) / bos_atr
                        if bos_atr > 0 else float("nan"))
    impulse_leg_bars = int(bos_idx - cand["origin_idx"])
    leg_high = float(df["high"].iloc[cand["origin_idx"]: bos_idx + 1].max())
    leg_low = float(df["low"].iloc[cand["origin_idx"]: bos_idx + 1].min())
    impulse_leg_atr = (leg_high - leg_low) / bos_atr if bos_atr > 0 else float("nan")

    # retest features
    retest_depth_atr = float("nan")
    retest_close_position = float("nan")
    retest_vol_pattern = float("nan")
    retest_kind = ""
    if math.isfinite(deepest_low_after_bos):
        atr_now = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_now > 0:
            retest_depth_atr = (zone_high - deepest_low_after_bos) / atr_now
    if state == "retest_bounce" and retest_idx is not None:
        atr_t2 = float(df["atr_sq"].iat[retest_idx]) if pd.notna(df["atr_sq"].iat[retest_idx]) else 0.0
        if atr_t2 > 0:
            retest_close_position = (float(df["close"].iat[retest_idx]) - zone_high) / atr_t2
        if retest_idx > bos_idx + 1:
            mid_vol = df["volume"].iloc[bos_idx + 1: retest_idx].mean()
            t2_vol = float(df["volume"].iat[retest_idx])
            if pd.notna(mid_vol) and mid_vol > 0:
                retest_vol_pattern = t2_vol / float(mid_vol)
    if pd.notna(retest_depth_atr):
        if retest_depth_atr >= 0.30:
            retest_kind = "deep_touch"
        elif retest_depth_atr >= 0.0:
            retest_kind = "shallow_touch"
        else:
            retest_kind = "no_touch"

    # contract — for MB, trigger_level = zone_high (the level that flipped to support)
    trigger_lvl = zone_high
    breakout_pct = (entry - trigger_lvl) / trigger_lvl if trigger_lvl > 0 else 0.0
    breakout_atr_v = ((entry - trigger_lvl) / atr_sq if atr_sq > 0 else float("nan"))
    extension_from_trigger = (c_now / trigger_lvl - 1.0) if trigger_lvl > 0 else 0.0
    entry_distance_to_trigger = (trigger_lvl - c_now) / max(c_now, 1e-9)

    row: dict = {
        # identity
        "ticker": ticker,
        "bar_date": bar_date_normalized,
        "setup_family": FAMILY,
        "signal_type": "long_mitigation",
        "signal_state": state,
        "breakout_bar_date": bos_bar_date,
        # audit
        "as_of_ts": asof_ts,
        "data_frequency": "1h",
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
        "common__volume_ratio_20": np.float32(vol_ratio) if pd.notna(vol_ratio) else np.float32("nan"),
        "common__extension_from_trigger": np.float32(extension_from_trigger),
        "common__entry_distance_to_trigger": np.float32(entry_distance_to_trigger),
        "common__gap_pct": np.float32(gap_pct),
        "common__gap_atr": np.float32(gap_atr) if pd.notna(gap_atr) else np.float32("nan"),
        "common__day_return": np.float32(day_return),
        "common__day_return_atr": np.float32(day_return_atr) if pd.notna(day_return_atr) else np.float32("nan"),
        # common — vol/liq
        "common__atr_14": np.float32(atr_14),
        "common__atr_pct": np.float32(atr_14 / entry) if entry > 0 and pd.notna(atr_14) else np.float32("nan"),
        "common__realized_vol_20": np.float32(bo["rv20"]) if pd.notna(bo["rv20"]) else np.float32("nan"),
        "common__realized_vol_pctile_120": np.float32(bo["rv120_pctile"])
            if pd.notna(bo["rv120_pctile"]) else np.float32("nan"),
        "common__volume": np.float64(bo["volume"]),
        "common__turnover": np.float64(bo["volume"] * entry),
        "common__liquidity_score": np.float32(np.log1p(bo["volume"] * entry)),
        "common__risk_pct_score": np.float32(initial_risk_pct),
        # common — trend / VWAP
        "common__close_vs_sma20": np.float32(entry / bo["sma20"] - 1.0) if pd.notna(bo["sma20"]) else np.float32("nan"),
        "common__sma20_slope": np.float32(bo["sma20_slope"]) if pd.notna(bo["sma20_slope"]) else np.float32("nan"),
        "common__close_vs_vwap10": np.float32(entry / bo["vwap10"] - 1.0) if pd.notna(bo["vwap10"]) else np.float32("nan"),
        "common__close_vs_vwap52": np.float32(entry / bo["vwap52"] - 1.0) if pd.notna(bo["vwap52"]) else np.float32("nan"),
        "common__close_vs_vwap_ytd": np.float32("nan"),
        "common__close_vs_vwap_base": np.float32("nan"),
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
        # family
        "family__zone_high": np.float32(zone_high),
        "family__zone_low": np.float32(zone_low),
        "family__zone_width_pct": np.float32(zone_width_pct) if pd.notna(zone_width_pct) else np.float32("nan"),
        "family__zone_width_atr": np.float32(zone_width_atr) if pd.notna(zone_width_atr) else np.float32("nan"),
        "family__zone_kind": zone_kind,
        "family__zone_age_bars": np.int16(asof_idx - cand["origin_idx"]),
        "family__bos_level": np.float32(cand["swing_high_price"]),
        "family__bos_distance_atr": np.float32(bos_distance_atr) if pd.notna(bos_distance_atr) else np.float32("nan"),
        "family__impulse_leg_bars": np.int16(impulse_leg_bars),
        "family__impulse_leg_atr": np.float32(impulse_leg_atr) if pd.notna(impulse_leg_atr) else np.float32("nan"),
        "family__leg_swing_low": np.float32(cand["swing_low_price"]),
        "family__touches_into_zone": np.int16(touches),
        "family__retest_depth_atr": np.float32(retest_depth_atr) if pd.notna(retest_depth_atr) else np.float32("nan"),
        "family__retest_close_position": np.float32(retest_close_position) if pd.notna(retest_close_position) else np.float32("nan"),
        "family__retest_vol_pattern": np.float32(retest_vol_pattern) if pd.notna(retest_vol_pattern) else np.float32("nan"),
        "family__retest_kind": retest_kind,
    }
    return row


# ---------------------------------------------------------------- main entry

def detect(intraday_df: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> list[dict]:
    """Snapshot detection: 0 or 1 row per ticker for the most recent armed,
    non-invalidated bullish MB zone within MAX_ZONE_AGE_BARS of asof.

    Parameters
    ----------
    intraday_df : DataFrame indexed by 1h timestamps with columns
                  ['open','high','low','close','volume'] and
                  `df.attrs['ticker']` set.
    asof : as-of timestamp; defaults to last bar in intraday_df.
    """
    ticker = intraday_df.attrs.get("ticker", "?")
    if len(intraday_df) < MIN_HISTORY:
        return []
    df = _compute_indicators(intraday_df)

    if asof is None:
        asof_idx = len(df) - 1
    else:
        asof_ts = pd.Timestamp(asof)
        ts_idx = pd.DatetimeIndex(df.index)
        if asof_ts.tz is None and ts_idx.tz is not None:
            asof_ts = asof_ts.tz_localize(ts_idx.tz)
        elif asof_ts.tz is not None and ts_idx.tz is None:
            asof_ts = asof_ts.tz_convert("Europe/Istanbul").tz_localize(None)
        mask = ts_idx <= asof_ts
        if not mask.any():
            return []
        asof_idx = int(np.flatnonzero(mask)[-1])

    if asof_idx < MIN_HISTORY:
        return []

    swing_highs, swing_lows = _find_pivots(df, PIVOT_N, asof_idx)
    if not swing_highs or not swing_lows:
        return []

    candidates = _find_mb_candidates(df, swing_highs, swing_lows, asof_idx)
    if not candidates:
        return []

    # Confirm BoS, age-filter, invalidation-filter; pick most recent BoS.
    armed: list[dict] = []
    for cand in candidates:
        bos_idx = _confirm_bos(df, cand, asof_idx)
        if bos_idx is None:
            continue
        # age = bars from origin (zone formation) to asof
        if asof_idx - cand["origin_idx"] > MAX_ZONE_AGE_BARS:
            continue
        # canonical zone = body
        zone_high = cand["zone_high_body"]
        zone_low = cand["zone_low_body"]
        if _is_invalidated(df, bos_idx, asof_idx, zone_low):
            continue
        cand_armed = dict(cand)
        cand_armed["bos_idx"] = bos_idx
        cand_armed["zone_high"] = zone_high
        cand_armed["zone_low"] = zone_low
        cand_armed["zone_kind"] = "body"
        armed.append(cand_armed)

    if not armed:
        return []

    best = max(armed, key=lambda c: c["bos_idx"])
    state, retest_idx, touches, deepest_low = _classify_state(
        df, best, best["bos_idx"], asof_idx, best["zone_high"], best["zone_low"],
    )
    if state == "none":
        return []

    row = _build_row(
        state=state, ticker=ticker, df=df, asof_idx=asof_idx,
        cand=best, bos_idx=best["bos_idx"], retest_idx=retest_idx,
        zone_high=best["zone_high"], zone_low=best["zone_low"],
        zone_kind=best["zone_kind"],
        touches=touches, deepest_low_after_bos=deepest_low,
    )
    return [row]
