"""Decision Engine v0 — state → phase mapping.

Locked spec: memory/decision_engine_v0_spec.md §Taxonomy / §Hard rules.

`accepted_continuation` is reserved for cohorts with a pre-registered PASS
horizon prior (currently: mb_5h__above_mb_birth, mb_1d__above_mb_birth).
The accepted_continuation assignment lives in handoffs.py — phase.py emits
generic `continuation` and handoffs upgrades it.
"""

from __future__ import annotations

from .schema import MIT_TOUCH_STATES, RETEST_BOUNCE_STATES


def map_phase(*, source: str, family: str, state: str) -> str:
    """Return phase label per locked taxonomy.

    Inputs are source-native (e.g. mb_scanner state='mitigation_touch').
    Returns generic `continuation` for above_mb cohorts; the
    accepted_continuation upgrade for H1 families is applied in handoffs.py.
    """
    s = (state or "").strip()

    # mb_scanner / bb_scanner ─────────────────────────────────────────────
    if source == "mb_scanner":
        if s == "above_mb":
            # generic continuation; handoffs.py promotes H1 cohorts to
            # accepted_continuation by canonical family lookup.
            return "continuation"
        if s in MIT_TOUCH_STATES:
            return "retest_pending"
        if s in RETEST_BOUNCE_STATES:
            return "retest"
        if s == "extended":
            return "extended"
        return "continuation"

    # horizontal_base ─────────────────────────────────────────────────────
    if source == "horizontal_base":
        if s == "trigger":
            return "trigger"
        if s in RETEST_BOUNCE_STATES:
            return "retest"
        if s == "pre_breakout":
            return "early_setup"
        if s == "extended":
            return "extended"
        return "continuation"

    # nyxexpansion ────────────────────────────────────────────────────────
    if source == "nyxexpansion":
        # adapter only emits trigger-day rows
        return "trigger"

    # nyxmomentum ─────────────────────────────────────────────────────────
    if source == "nyxmomentum":
        return "strength_context"

    # nox_rt_daily ────────────────────────────────────────────────────────
    if source == "nox_rt_daily":
        # PIVOT_AL standalone — Rule 5 will downgrade. phase=trigger
        return "trigger"

    # nox_weekly ──────────────────────────────────────────────────────────
    if source == "nox_weekly":
        return "trigger"

    return "continuation"


__all__ = ["map_phase"]
