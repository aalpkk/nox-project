"""Decision Engine v0 — hard-rule action layer (deterministic, in-order).

Locked spec: memory/decision_engine_v0_spec.md §Hard rules.

Rules evaluated in numeric order; first match sets `final_action`. No
tie-breaking by score, no learned weights.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import (
    ACTION_COLUMNS,
    EXTENSION_CAP_ATR,
    LIQUIDITY_FLOOR_PCTILE,
    MAX_RISK_ATR,
    MIT_TOUCH_STATES,
)


def _liquidity_floor(events: pd.DataFrame) -> float | None:
    """Configurable floor; default = bottom 5%-ile across events that report a score.

    Returns None when too few events to set a floor (rule 9 then no-ops).
    """
    if "liquidity_score" not in events.columns:
        return None
    scored = events["liquidity_score"].dropna()
    if len(scored) < 20:
        return None
    return float(np.percentile(scored, LIQUIDITY_FLOOR_PCTILE * 100))


def _is_null(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _evaluate_row(row: pd.Series, *, liq_floor: float | None) -> tuple[str, list[str]]:
    """Return (final_action, reason_codes) for one event row.

    Spec rules 1-17, in order. Strength-context (Rule 16) preempts the
    no_entry_ref / no_stop_ref guards for nyxmomentum-style context-only
    events — those don't claim to be actionable trades; the spec example
    explicitly lands them at WATCHLIST.
    """
    reasons: list[str] = []
    phase = (row.get("phase") or "").strip()
    state = (row.get("state") or "").strip()
    family = (row.get("family") or "").strip()
    higher_tf = row.get("higher_tf_context") or ""

    risk_atr = row.get("risk_atr")
    risk_pct = row.get("risk_pct")
    extension_atr = row.get("extension_atr")
    liquidity = row.get("liquidity_score")
    regime = (row.get("regime") or "").strip()

    # Rule 16 preemption: strength_context never claims a stop/entry
    if phase == "strength_context":
        reasons.append("strength_context_only")
        reasons.append("fill_realism_unresolved")
        return "WATCHLIST", reasons

    # Rule 7 preemption: explicit exit_warning
    if phase == "exit_warning":
        reasons.append("nox_sat_conflict")
        reasons.append("fill_realism_unresolved")
        return "EXIT_WARNING", reasons

    # Rule 1: stop_ref null
    if _is_null(row.get("stop_ref")):
        reasons.append("no_stop_ref")
        reasons.append("fill_realism_unresolved")
        return "AVOID", reasons

    # Rule 2: entry_ref null
    if _is_null(row.get("entry_ref")):
        reasons.append("no_entry_ref")
        reasons.append("fill_realism_unresolved")
        return "AVOID", reasons

    # Rule 3: risk too wide
    if not _is_null(risk_atr) and float(risk_atr) > MAX_RISK_ATR:
        reasons.append("risk_too_wide")
        reasons.append("fill_realism_unresolved")
        return "AVOID", reasons

    # Rule 4: extended phase → AVOID late chase
    if phase == "extended":
        reasons.append("late_chase")
        reasons.append("avoid_late")
        reasons.append("fill_realism_unresolved")
        return "AVOID", reasons

    # Rule 5: standalone nox_rt_daily/pivot_al
    if family == "nox_rt_daily__pivot_al":
        reasons.append("nox_rt_daily_weak_standalone")
        reasons.append("fill_realism_unresolved")
        # higher_tf supportive → context-only WATCHLIST per spec
        if higher_tf and "weekly" in higher_tf.lower():
            return "WATCHLIST", reasons
        return "AVOID", reasons

    # Rule 6: state == mit_touch_first (incl. live alias 'mitigation_touch')
    if state in MIT_TOUCH_STATES:
        reasons.append("mit_touch_context_only")
        reasons.append("fill_realism_unresolved")
        return "WAIT_RETEST", reasons

    # Rule 8: early_setup / retest_pending
    if phase == "early_setup":
        reasons.append("pre_breakout")
        reasons.append("fill_realism_unresolved")
        return "WAIT_TRIGGER", reasons
    if phase == "retest_pending":
        reasons.append("zone_armed")
        reasons.append("fill_realism_unresolved")
        return "WAIT_TRIGGER", reasons

    # Rule 9: liquidity below floor
    liquidity_low_flag = False
    if (
        liq_floor is not None
        and not _is_null(liquidity)
        and float(liquidity) < liq_floor
    ):
        reasons.append("low_liquidity_below_floor")
        reasons.append("fill_realism_unresolved")
        return "AVOID", reasons
    if not _is_null(liquidity):
        # descriptive-only on the way through
        reasons.append("liquidity_ok")

    # ─── from here, gate-passing branches; collect positives ─────────────
    risk_ok = (not _is_null(risk_atr)) and float(risk_atr) <= MAX_RISK_ATR

    # Regime check — TREND/long/full_trend supportive; CHOPPY/neutral mismatch
    # downgrades by one tier rather than AVOID-ing.
    regime_supportive = regime in ("long",)
    regime_neutral = regime in ("neutral", "unknown", "")
    regime_short = regime in ("short",)

    def _tag_regime(rs: list[str]) -> None:
        if regime_supportive:
            rs.append("regime_ok")
        elif regime_short and family.startswith(("nox_", "horizontal_base", "nyx")):
            rs.append("regime_mismatch")

    # Rule 10: retest
    if phase == "retest" and risk_ok:
        reasons.append("retest_confirmed")
        reasons.append("risk_ok")
        _tag_regime(reasons)
        reasons.append("fill_realism_unresolved")
        if regime_short and not regime_supportive:
            return "WAIT_RETEST", reasons
        return "TRADEABLE", reasons

    # Rule 11/12: trigger
    if phase == "trigger":
        if risk_ok and (
            _is_null(extension_atr) or float(extension_atr) <= EXTENSION_CAP_ATR
        ):
            reasons.append("clean_trigger")
            reasons.append("risk_ok")
            _tag_regime(reasons)
            reasons.append("fill_realism_unresolved")
            if regime_short:
                return "WAIT_RETEST", reasons
            return "TRADEABLE", reasons
        if (
            risk_ok
            and not _is_null(extension_atr)
            and float(extension_atr) > EXTENSION_CAP_ATR
        ):
            reasons.append("extension_high")
            reasons.append("fill_realism_unresolved")
            return "WAIT_RETEST", reasons
        # risk fails but not too wide (was caught in rule 3) — fall through
        reasons.append("fill_realism_unresolved")
        return "WATCHLIST", reasons

    # Rule 13: reversal
    if phase == "reversal" and risk_ok:
        reasons.append("reversal_confirmed")
        reasons.append("risk_ok")
        _tag_regime(reasons)
        reasons.append("fill_realism_unresolved")
        return "TRADEABLE", reasons

    # Rule 14: accepted_continuation (H1 cohort)
    if phase == "accepted_continuation" and risk_ok:
        reasons.append("accepted_horizon_h1_20d")
        reasons.append("risk_ok")
        _tag_regime(reasons)
        reasons.append("fill_realism_unresolved")
        return "TRADEABLE", reasons

    # Rule 15: generic continuation (no accepted prior)
    if phase == "continuation" and risk_ok:
        reasons.append("continuation_no_prior")
        reasons.append("fill_realism_unresolved")
        return "WATCHLIST", reasons

    # Rule 17: unmatched default
    reasons.append("unmatched_default")
    reasons.append("fill_realism_unresolved")
    return "WATCHLIST", reasons


def apply_actions(events: pd.DataFrame) -> pd.DataFrame:
    """Run hard-rule table over every event; return decision_actions DataFrame."""
    if events.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)

    liq_floor = _liquidity_floor(events)

    actions = []
    reasons = []
    for _, row in events.iterrows():
        action, codes = _evaluate_row(row, liq_floor=liq_floor)
        # merge in any reason_candidates already attached upstream
        prior = list(row.get("reason_candidates") or [])
        merged = list(dict.fromkeys(prior + codes))
        actions.append(action)
        reasons.append(merged)

    out = events.copy()
    out["final_action"] = actions
    out["reason_codes"] = reasons

    return out[
        [c for c in ACTION_COLUMNS if c in out.columns]
    ].copy()


__all__ = ["apply_actions"]
