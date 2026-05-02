"""Family-specific rule scoring.

Each scorer is a pure function of one row dict + the family weight prior. It
returns (rule_score, common_score, family_score, signal_tags). Weights are
NEVER tuned on backtest — they are frozen priors from `scanner.schema.RULE_WEIGHTS`.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from .schema import FAMILIES, RULE_WEIGHTS


def _safe(x, default=0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(v) else v


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------- horizontal_base

def _score_horizontal_base(row: dict) -> tuple[float, float, float, list[str]]:
    """State-aware scorer.

    Trigger     : full breakout-quality scoring (volume, body, range_position).
    Pre-breakout: replaces breakout_quality with proximity_quality (distance
                  to box top, vol drying), removes extension_penalty.
    Extended    : same breakout-quality (snapshot of the original breakout bar)
                  plus an age penalty proportional to bars_since_trigger and
                  extension above the trigger.
    """
    w = RULE_WEIGHTS["horizontal_base"]
    state = row.get("signal_state", "trigger")
    tags: list[str] = [f"state:{state}"]

    # ---- shared geometry / context ----------------------------------------
    width_atr = _safe(row.get("family__channel_width_atr"), 5.0)
    width_pct = _safe(row.get("family__channel_width_pct"), 0.20)
    base_tight = _clip01(1.0 - width_atr / 5.0) * 0.6 + _clip01(1.0 - width_pct / 0.20) * 0.4
    if base_tight > 0.7:
        tags.append("base_tight")

    # Volatility-contraction quality (V1.2.2): ATR is no longer a gate; instead
    # its compression + decline contribute to a separate score. Lower atr_ratio
    # (range tighter than its own SMA) and negative atr_decline boost score.
    atr_ratio = _safe(row.get("family__atr_ratio_mean_base"), 1.0)
    atr_decline = _safe(row.get("family__atr_decline_5_pct"), 0.0)
    atr_compression = _clip01((0.95 - atr_ratio) / 0.30)        # 0.95→0, 0.65→1
    atr_decline_q = _clip01(-atr_decline / 0.10)                # -10%→1, 0→0
    bb_tightness_q = _clip01((0.65 - width_pct / 0.30))         # rough BB-tight proxy
    vol_contraction = (
        0.50 * bb_tightness_q
        + 0.30 * atr_compression
        + 0.20 * atr_decline_q
    )
    if atr_compression > 0.5 and atr_decline_q > 0.3:
        tags.append("vol_collapse")

    upper = _safe(row.get("family__upper_touch_count"))
    lower = _safe(row.get("family__lower_touch_count"))
    balance = abs(_safe(row.get("family__touch_balance")))
    touch_q = _clip01(min(upper, 4) / 4.0) * 0.5 + _clip01(min(lower, 3) / 3.0) * 0.3 + (1.0 - _clip01(balance)) * 0.2
    if upper >= 2 and lower >= 1:
        tags.append("touch_ok")

    base_slope_abs = abs(_safe(row.get("family__base_slope")))
    res_slope_abs = abs(_safe(row.get("family__resistance_slope")))
    if base_slope_abs <= 0.0015:
        tags.append("flat_base")
    elif base_slope_abs <= 0.0030:
        tags.append("loose_base")
    elif base_slope_abs <= 0.0050:
        tags.append("very_loose_base")
    if res_slope_abs > 0.0015 and res_slope_abs <= 0.0030:
        tags.append("loose_resistance")
    elif res_slope_abs > 0.0030 and res_slope_abs <= 0.0050:
        tags.append("very_loose_resistance")

    cs20 = _safe(row.get("common__close_vs_sma20"))
    cv52 = _safe(row.get("common__close_vs_vwap52"))
    cv_ytd = _safe(row.get("common__close_vs_vwap_ytd"))
    s20s = _safe(row.get("common__sma20_slope"))
    trend = (
        _clip01(cs20 / 0.05) * 0.35
        + _clip01(cv52 / 0.10) * 0.25
        + _clip01(cv_ytd / 0.10) * 0.20
        + _clip01(s20s / 0.02) * 0.20
    )
    if cs20 > 0 and cv52 > 0 and cv_ytd > 0:
        tags.append("trend_reclaim")

    rs_p = _safe(row.get("common__rs_pctile_252"), 0.5)
    rs_score = _clip01(rs_p)
    if rs_p > 0.80:
        tags.append("rs_top20")

    liq = _safe(row.get("common__liquidity_score"))
    risk = _safe(row.get("initial_risk_pct"), 0.10)
    liq_score = _clip01((liq - 12.0) / 6.0) * 0.6 + _clip01(1.0 - risk / 0.10) * 0.4

    # ---- state-specific quality + penalty ---------------------------------
    bo_atr = _safe(row.get("common__breakout_atr"))
    rng_pos = _safe(row.get("common__range_position"), 0.5)
    body_pct = _safe(row.get("common__body_pct"))
    vr = _safe(row.get("common__volume_ratio_20"))
    dryup = _safe(row.get("family__volume_dryup_ratio"), 1.0)
    ext = _safe(row.get("common__extension_from_trigger"))

    if state == "pre_breakout":
        # proximity_quality replaces breakout_quality. Range_position close
        # to box top + vol drying + base tight = better watchlist.
        close_pos = _safe(row.get("family__close_position_in_base"), 0.5)
        dist_top = _safe(row.get("family__prebreakout_distance_to_high"), 0.10)
        proximity_q = (
            _clip01(close_pos) * 0.45
            + _clip01(1.0 - dist_top / 0.05) * 0.35
            + _clip01(1.0 - dryup) * 0.20
        )
        bo_quality = proximity_q
        vol_conf = _clip01(1.0 - dryup) * 0.6 + _clip01((vr - 0.7) / 0.6) * 0.4
        ext_pen = 0.0
        if dist_top < 0.02:
            tags.append("at_resistance")
        if dryup < 0.85:
            tags.append("volume_dryup_pre")

    elif state == "extended":
        bo_quality = (
            _clip01(bo_atr / 1.5) * 0.4
            + _clip01(rng_pos) * 0.3
            + _clip01(body_pct / 0.04) * 0.3
        )
        vol_conf = _clip01((vr - 1.0) / 1.5) * 0.7 + _clip01(1.0 - dryup) * 0.3
        if bo_atr > 0.5 and rng_pos > 0.7:
            tags.append("breakout_confirmed")
        if vr >= 1.8:
            tags.append("volume_expansion")
        # age + extension stacking penalty
        ext_pen = _clip01((ext - 0.02) / 0.05) * 0.6 + _clip01((ext - 0.05) / 0.05) * 0.4
        tags.append("late_chase")

    else:  # trigger
        bo_quality = (
            _clip01(bo_atr / 1.5) * 0.4
            + _clip01(rng_pos) * 0.3
            + _clip01(body_pct / 0.04) * 0.3
        )
        vol_conf = _clip01((vr - 1.0) / 1.5) * 0.7 + _clip01(1.0 - dryup) * 0.3
        if bo_atr > 0.5 and rng_pos > 0.7:
            tags.append("breakout_confirmed")
        if vr >= 1.8:
            tags.append("volume_expansion")
        ext_pen = _clip01((ext - 0.02) / 0.05)
        if ext > 0.04:
            tags.append("late_chase")

    family_score = (
        w["base_tightness_score"] * base_tight
        + w["volatility_contraction_score"] * vol_contraction
        + w["touch_quality_score"] * touch_q
        + w["breakout_quality_score"] * bo_quality
        + w["volume_confirmation_score"] * vol_conf
        + w["trend_reclaim_score"] * trend
        + w["relative_strength_score"] * rs_score
        + w["liquidity_risk_score"] * liq_score
        + w["extension_penalty"] * ext_pen
    )
    common_score = (
        w["breakout_quality_score"] * bo_quality
        + w["volume_confirmation_score"] * vol_conf
        + w["trend_reclaim_score"] * trend
        + w["relative_strength_score"] * rs_score
    )
    rule_score = family_score
    return rule_score, common_score, family_score, tags


# ---------------------------------------------------------------- ICT/SMC shared

def _ict_zone_quality(row: dict) -> float:
    """Tight zone + young zone = better quality."""
    width_atr = _safe(row.get("family__zone_width_atr"), 2.0)
    age = _safe(row.get("family__zone_age_bars"), 120.0)
    tight = _clip01((1.5 - width_atr) / 1.0)        # 0.5 ATR→1, 1.5 ATR→0
    fresh = _clip01((100.0 - age) / 100.0)          # 0 bars→1, 100 bars→0
    return tight * 0.6 + fresh * 0.4


def _ict_bos_quality(row: dict) -> float:
    bos_d = _safe(row.get("family__bos_distance_atr"))
    leg = _safe(row.get("family__impulse_leg_atr"))
    return _clip01(bos_d / 1.5) * 0.6 + _clip01(leg / 3.0) * 0.4


def _ict_retest_quality(row: dict, state: str) -> float:
    if state == "zone_armed":
        return 0.0  # forward-looking — no realized retest yet
    if state == "mitigation_touch":
        depth = _safe(row.get("family__retest_depth_atr"))
        return _clip01(depth / 0.50) * 0.5
    if state == "retest_bounce":
        depth = _safe(row.get("family__retest_depth_atr"))
        close_pos = _safe(row.get("family__retest_close_position"))
        body_ok = _clip01(depth / 0.30) * 0.4
        reclaim_ok = _clip01(close_pos / 0.20) * 0.4
        kind = row.get("family__retest_kind", "")
        bonus = 0.2 if kind == "deep_touch" else 0.1 if kind == "shallow_touch" else 0.0
        return body_ok + reclaim_ok + bonus
    if state == "extended":
        return 0.5  # already past; age + ext penalties handle staleness
    return 0.0


def _ict_vol_confirmation(row: dict, state: str) -> float:
    vr = _safe(row.get("common__volume_ratio_20"))
    vp = _safe(row.get("family__retest_vol_pattern"), 1.0)
    if state == "retest_bounce":
        return _clip01((vr - 1.0) / 1.0) * 0.5 + _clip01((vp - 1.0) / 0.5) * 0.5
    if state == "mitigation_touch":
        # subdued volume on the touch reads as no-panic absorption
        return _clip01((1.2 - vr) / 0.5) * 0.5
    return _clip01((vr - 1.0) / 1.0)


def _ict_trend_alignment(row: dict) -> float:
    cs20 = _safe(row.get("common__close_vs_sma20"))
    cv52 = _safe(row.get("common__close_vs_vwap52"))
    s20s = _safe(row.get("common__sma20_slope"))
    return (
        _clip01(cs20 / 0.05) * 0.40
        + _clip01(cv52 / 0.10) * 0.30
        + _clip01(s20s / 0.02) * 0.30
    )


def _ict_rs(row: dict) -> float:
    return _clip01(_safe(row.get("common__rs_pctile_252"), 0.5))


def _ict_liq_risk(row: dict) -> float:
    liq = _safe(row.get("common__liquidity_score"))
    risk = _safe(row.get("initial_risk_pct"), 0.10)
    return _clip01((liq - 12.0) / 6.0) * 0.6 + _clip01(1.0 - risk / 0.10) * 0.4


def _ict_stale_penalty(row: dict) -> float:
    age = _safe(row.get("family__zone_age_bars"))
    return _clip01((age - 60.0) / 60.0)        # 60 bars→0, 120 bars→1


def _ict_extension_penalty(row: dict, state: str) -> float:
    if state in ("zone_armed", "mitigation_touch"):
        return 0.0
    ext = _safe(row.get("common__extension_from_trigger"))
    return _clip01((ext - 0.02) / 0.05)


# ---------------------------------------------------------------- mitigation_block

def _score_mitigation_block(row: dict) -> tuple[float, float, float, list[str]]:
    w = RULE_WEIGHTS["mitigation_block"]
    state = row.get("signal_state", "zone_armed")
    tags: list[str] = [f"state:{state}"]

    zq = _ict_zone_quality(row)
    bq = _ict_bos_quality(row)
    rq = _ict_retest_quality(row, state)
    vq = _ict_vol_confirmation(row, state)
    tq = _ict_trend_alignment(row)
    rs = _ict_rs(row)
    lq = _ict_liq_risk(row)
    stale = _ict_stale_penalty(row)
    ext = _ict_extension_penalty(row, state)

    if zq > 0.6:
        tags.append("zone_tight")
    if bq > 0.6:
        tags.append("bos_strong")
    kind = row.get("family__retest_kind", "")
    if kind:
        tags.append(f"retest:{kind}")
    if state == "zone_armed":
        tags.append("watchlist")
    if state == "retest_bounce":
        tags.append("actionable")

    family_score = (
        w["zone_quality_score"] * zq
        + w["bos_quality_score"] * bq
        + w["retest_quality_score"] * rq
        + w["volume_confirmation_score"] * vq
        + w["trend_alignment_score"] * tq
        + w["relative_strength_score"] * rs
        + w["liquidity_risk_score"] * lq
        + w["stale_zone_penalty"] * stale
        + w["extension_penalty"] * ext
    )
    common_score = (
        w["volume_confirmation_score"] * vq
        + w["trend_alignment_score"] * tq
        + w["relative_strength_score"] * rs
    )
    return family_score, common_score, family_score, tags


# ---------------------------------------------------------------- breaker_block

def _score_breaker_block(row: dict) -> tuple[float, float, float, list[str]]:
    w = RULE_WEIGHTS["breaker_block"]
    state = row.get("signal_state", "zone_armed")
    tags: list[str] = [f"state:{state}"]

    zq = _ict_zone_quality(row)
    sweep_d = _safe(row.get("family__sweep_depth_atr"))
    sq = _clip01(sweep_d / 1.0)                  # 0 ATR→0, 1 ATR→1
    bq = _ict_bos_quality(row)
    rq = _ict_retest_quality(row, state)
    vq = _ict_vol_confirmation(row, state)
    tq = _ict_trend_alignment(row)
    rs = _ict_rs(row)
    lq = _ict_liq_risk(row)
    stale = _ict_stale_penalty(row)
    ext = _ict_extension_penalty(row, state)

    if sq > 0.6:
        tags.append("sweep_clear")
    if zq > 0.6:
        tags.append("zone_tight")
    if bq > 0.6:
        tags.append("bos_strong")
    kind = row.get("family__retest_kind", "")
    if kind:
        tags.append(f"retest:{kind}")
    if state == "zone_armed":
        tags.append("watchlist")
    if state == "retest_bounce":
        tags.append("actionable")

    family_score = (
        w["zone_quality_score"] * zq
        + w["sweep_quality_score"] * sq
        + w["bos_quality_score"] * bq
        + w["retest_quality_score"] * rq
        + w["volume_confirmation_score"] * vq
        + w["trend_alignment_score"] * tq
        + w["relative_strength_score"] * rs
        + w["liquidity_risk_score"] * lq
        + w["stale_zone_penalty"] * stale
        + w["extension_penalty"] * ext
    )
    common_score = (
        w["volume_confirmation_score"] * vq
        + w["trend_alignment_score"] * tq
        + w["relative_strength_score"] * rs
    )
    return family_score, common_score, family_score, tags


# ---------------------------------------------------------------- registry

SCORERS: dict[str, Callable[[dict], tuple[float, float, float, list[str]]]] = {
    "horizontal_base": _score_horizontal_base,
    "mitigation_block": _score_mitigation_block,
    "breaker_block": _score_breaker_block,
}


def score_row(row: dict) -> dict:
    """Mutates row in place: sets rule_score / common_score / family_score / signal_tags."""
    fam = row.get("setup_family")
    if fam not in SCORERS:
        # other families implemented later; emit zeros so schema stays satisfied
        row.setdefault("rule_score", np.float32(0.0))
        row.setdefault("common_score", np.float32(0.0))
        row.setdefault("family_score", np.float32(0.0))
        row.setdefault("signal_tags", json.dumps([]))
        row.setdefault("signal_reason_text", "")
        return row
    rule, common, fam_s, tags = SCORERS[fam](row)
    row["rule_score"] = np.float32(rule)
    row["common_score"] = np.float32(common)
    row["family_score"] = np.float32(fam_s)
    row["signal_tags"] = json.dumps(tags)
    row["signal_reason_text"] = " · ".join(tags) if tags else ""
    return row
