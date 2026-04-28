"""SBT-1700 — F0..F8 controlled exit-discovery grid.

Returns ~142 hand-curated variants spread over 12 families. The full
Cartesian of the design dimensions is intentionally avoided because the
spec calls for "around 40-120 variants maximum, each interpretable".

Family taxonomy
---------------
F0_no_partial_trend
    No partial, no breakeven. Tests pure trend-exit families against
    initial-SL choice.

F1_partial_no_breakeven
    Single partial (33% / 50% at +1R / +2R). No breakeven. Tests whether
    locking part of the position helps or hurts expectancy without the
    breakeven confound.

F2_partial_with_breakeven
    Partial + breakeven, where breakeven activates at MFE_R ≥ {1, 1.5, 2}.
    Tests whether breakeven cuts winners short.

F3_stepout
    Two-leg partial: 33% at +1R AND 33% at +2R, runner 34%. Optional
    breakeven after +2R. Half the variants delay runner activation
    until MFE_R ≥ +1R to test whether late activation matters when the
    runner is small.

F4_structure
    Structure-aware exits: close < box_top, close < box_mid, EMA10, or
    ATR trail 2.0, with partial off or 50%@+1R.

F5_mfe_capture (extension)
    MFE-giveback exits with explicit activation gates. Two flavours:
    F5a uses activation_R = 0.5 (start trailing once +0.5R is hit),
    F5b uses activation_R = 1.0. Crossed with sl ∈ {1.5, 2.0} ATR,
    partial ∈ {off, 33%@+1R}, giveback ∈ {0.20, 0.30, 0.40}, hold = 40.
    Designed to fill slot-3 (MFE-capture / low-giveback) which the
    F0–F4 grammar leaves unresolved.

F6_tight_trail (extension)
    ATR trail with tight k ∈ {0.75, 1.0, 1.25} (sub-2.0 territory).
    sl ∈ {1.5, 2.0}, partial ∈ {off, 33%@+1R}, hold = 40. Tests the
    capture/giveback edge of the trail spectrum.

F7_structure_reclaim (extension)
    Exit on structure-top reclaim (close < box_top), optional EMA10
    expressed as a sibling-variant fallback (not a runtime composite).
    sl ∈ {1.5, 2.0}, partial ∈ {off, 33%@+1R}, hold = 40. Distinct
    from F4 in: 33%@+1R partial sizing instead of 50%, dedicated
    family slot for structure-pair experiments.

F8a_lock_1R, F8b_lock_1R_tighter, F8c_lock_after_1_5R,
F8d_hybrid_lock_plus_trend (capture-focused extension)
    Profit-lock ladder. Stop ratchets up at MFE milestones:
      F8a: (+1R→entry+0.25R), (+2R→entry+1.0R),  (+3R→entry+2.0R)
      F8b: (+1R→entry+0.4R),  (+2R→entry+1.2R),  (+3R→entry+2.2R)
      F8c: (+1.5R→entry+0.75R), (+2.5R→entry+1.5R), (+3.5R→entry+2.5R)
      F8d: F8a ladder + EMA10 trend exit (close < EMA10).
    Each sub-family crosses sl ∈ {1.5, 2.0} × partial ∈ {off, 33%@+1R}
    × hold ∈ {20, 40} → 8 variants × 4 sub-families = 32. Same-bar
    pessimism preserved by end-of-bar arming (the bar that triggers a
    new rung uses the OLD floor for its own SL check).

Stable variant id format
------------------------
`<F#>_sl<sl>_<partial><BE><activation>_<trend>_h<max_hold>`

Examples:
    F0_sl1.5_p0_atr1.5_h40
    F1_sl2.0_p33at1R_mfe35_h40
    F2_sl1.5_p50at1R_be1.5R_atr2.0_h40
    F3_sl1.5_p33at1R+33at2R_be2R_a1R_atr2.0_h40
    F4_sl2.0_p0_struct_top_h40
    F5_sl1.5_p33at1R_a0.5R_mfe30_h40
    F6_sl2.0_p0_atr0.75_h40
    F7_sl1.5_p33at1R_struct_top_h40
    F8a_sl1.5_p0_lockL1_h40
    F8d_sl2.0_p33at1R_lockL1+ema10_h20
"""
from __future__ import annotations

from typing import Iterable

from sbt1700.exits_v2 import (
    ExitConfigV2,
    TREND_ATR_TRAIL,
    TREND_EMA,
    TREND_MFE_GIVEBACK,
    TREND_STRUCTURE_MID,
    TREND_STRUCTURE_TOP,
)


# Short labels for variant naming.
def _sl_tag(sl: float) -> str:
    return f"sl{sl:.1f}"


def _hold_tag(h: int) -> str:
    return f"h{h}"


def _partial_tag(size: float, R: float) -> str:
    if size <= 0:
        return "p0"
    return f"p{int(round(size * 100))}at{_R_short(R)}"


def _stepout_tag(s1: float, R1: float, s2: float, R2: float) -> str:
    return f"p{int(round(s1*100))}at{_R_short(R1)}+{int(round(s2*100))}at{_R_short(R2)}"


def _be_tag(be) -> str:
    if be is None:
        return ""
    return f"_be{_R_short(be)}"


def _act_tag(act: float) -> str:
    if act <= 0:
        return ""
    return f"_a{_R_short(act)}"


def _R_short(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}R"
    return f"{x:.1f}R"


def _trend_tag(kind: str, k=None, g=None, ema=None) -> str:
    if kind == TREND_ATR_TRAIL:
        return f"atr{k:.1f}"
    if kind == TREND_MFE_GIVEBACK:
        return f"mfe{int(round(g*100))}"
    if kind == TREND_EMA:
        return f"ema{ema}"
    if kind == TREND_STRUCTURE_TOP:
        return "struct_top"
    if kind == TREND_STRUCTURE_MID:
        return "struct_mid"
    return kind


def _make_name(family: str, sl: float, partial_part: str, be, activation,
               trend_part: str, hold: int) -> str:
    return (
        f"{family}_{_sl_tag(sl)}_{partial_part}{_be_tag(be)}{_act_tag(activation)}"
        f"_{trend_part}_{_hold_tag(hold)}"
    )


# ---------- F0: no partial, no breakeven --------------------------------------

def _build_F0() -> list[ExitConfigV2]:
    family = "F0_no_partial_trend"
    out: list[ExitConfigV2] = []
    plan = [
        # (sl, trend_kind, k, g, ema, max_hold)
        (1.5, TREND_ATR_TRAIL, 1.5, None, None, 20),
        (1.5, TREND_ATR_TRAIL, 1.5, None, None, 40),
        (1.5, TREND_ATR_TRAIL, 2.0, None, None, 20),
        (1.5, TREND_MFE_GIVEBACK, None, 0.35, None, 40),
        (1.5, TREND_MFE_GIVEBACK, None, 0.45, None, 40),
        (1.5, TREND_EMA, None, None, 10, 40),
        (2.0, TREND_ATR_TRAIL, 1.5, None, None, 20),
        (2.0, TREND_ATR_TRAIL, 1.5, None, None, 40),
        (2.0, TREND_ATR_TRAIL, 2.0, None, None, 40),
        (2.0, TREND_MFE_GIVEBACK, None, 0.35, None, 40),
        (2.0, TREND_MFE_GIVEBACK, None, 0.45, None, 40),
        (2.0, TREND_EMA, None, None, 10, 40),
    ]
    for sl, kind, k, g, ema, hold in plan:
        trend = _trend_tag(kind, k, g, ema)
        name = _make_name(family, sl, "p0", None, 0.0, trend, hold)
        out.append(ExitConfigV2(
            name=name, family=family,
            initial_sl_atr=sl,
            partial_size=0.0, partial_R=0.0,
            partial2_size=0.0, partial2_R=0.0,
            breakeven_R=None, activation_R=0.0,
            trend_kind=kind, trend_atr_k=k, trend_giveback=g,
            trend_ema_period=ema, max_hold_bars=hold,
        ))
    return out


# ---------- F1: partial, no breakeven -----------------------------------------

def _build_F1() -> list[ExitConfigV2]:
    family = "F1_partial_no_breakeven"
    out: list[ExitConfigV2] = []
    plan = [
        # (sl, partial_size, partial_R, kind, k, g, hold)
        (1.5, 0.33, 1.0, TREND_ATR_TRAIL, 1.5, None, 40),
        (2.0, 0.33, 1.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.33, 1.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.33, 1.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (1.5, 0.50, 1.0, TREND_ATR_TRAIL, 1.5, None, 40),
        (2.0, 0.50, 1.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 1.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.50, 1.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (1.5, 0.33, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 0.33, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.33, 2.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (2.0, 0.33, 2.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (1.5, 0.50, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 0.50, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 2.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (2.0, 0.50, 2.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
    ]
    for sl, ps, pR, kind, k, g, hold in plan:
        trend = _trend_tag(kind, k, g, None)
        name = _make_name(family, sl, _partial_tag(ps, pR), None, 0.0, trend, hold)
        out.append(ExitConfigV2(
            name=name, family=family,
            initial_sl_atr=sl,
            partial_size=ps, partial_R=pR,
            partial2_size=0.0, partial2_R=0.0,
            breakeven_R=None, activation_R=0.0,
            trend_kind=kind, trend_atr_k=k, trend_giveback=g,
            trend_ema_period=None, max_hold_bars=hold,
        ))
    return out


# ---------- F2: partial + breakeven -------------------------------------------

def _build_F2() -> list[ExitConfigV2]:
    family = "F2_partial_with_breakeven"
    out: list[ExitConfigV2] = []
    plan = [
        # (sl, partial_size, partial_R, breakeven_R, kind, k, g, hold)
        (1.5, 0.33, 1.0, 1.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.33, 1.0, 1.5, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.33, 1.0, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.33, 1.0, 1.5, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.33, 1.0, 1.5, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.33, 1.0, 2.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (1.5, 0.50, 1.0, 1.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 1.0, 1.5, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 1.0, 2.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 1.0, 1.5, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.50, 1.0, 1.5, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.50, 1.0, 2.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (2.0, 0.33, 1.5, 1.5, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 0.50, 1.5, 1.5, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 0.33, 1.5, 2.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 0.50, 1.5, 2.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (1.5, 0.33, 1.0, 1.5, TREND_ATR_TRAIL, 1.5, None, 20),
        (1.5, 0.50, 1.0, 1.5, TREND_ATR_TRAIL, 1.5, None, 20),
    ]
    for sl, ps, pR, be, kind, k, g, hold in plan:
        trend = _trend_tag(kind, k, g, None)
        name = _make_name(family, sl, _partial_tag(ps, pR), be, 0.0, trend, hold)
        out.append(ExitConfigV2(
            name=name, family=family,
            initial_sl_atr=sl,
            partial_size=ps, partial_R=pR,
            partial2_size=0.0, partial2_R=0.0,
            breakeven_R=be, activation_R=0.0,
            trend_kind=kind, trend_atr_k=k, trend_giveback=g,
            trend_ema_period=None, max_hold_bars=hold,
        ))
    return out


# ---------- F3: stepout (two partial legs) ------------------------------------

def _build_F3() -> list[ExitConfigV2]:
    family = "F3_stepout"
    out: list[ExitConfigV2] = []
    # 33% at +1R AND 33% at +2R, runner 34%. SL fixed at 1.5 ATR.
    sl = 1.5
    plan = [
        # (breakeven_R, activation_R, kind, k, g, hold)
        (None, 0.0, TREND_ATR_TRAIL, 2.0, None, 20),
        (None, 0.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (None, 0.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (None, 0.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
        (2.0, 1.0, TREND_ATR_TRAIL, 2.0, None, 20),
        (2.0, 1.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 1.0, TREND_MFE_GIVEBACK, None, 0.35, 40),
        (2.0, 1.0, TREND_MFE_GIVEBACK, None, 0.45, 40),
    ]
    for be, act, kind, k, g, hold in plan:
        trend = _trend_tag(kind, k, g, None)
        partial_part = _stepout_tag(0.33, 1.0, 0.33, 2.0)
        name = _make_name(family, sl, partial_part, be, act, trend, hold)
        out.append(ExitConfigV2(
            name=name, family=family,
            initial_sl_atr=sl,
            partial_size=0.33, partial_R=1.0,
            partial2_size=0.33, partial2_R=2.0,
            breakeven_R=be, activation_R=act,
            trend_kind=kind, trend_atr_k=k, trend_giveback=g,
            trend_ema_period=None, max_hold_bars=hold,
        ))
    return out


# ---------- F4: structure / EMA / ATR distinct --------------------------------

def _build_F4() -> list[ExitConfigV2]:
    family = "F4_structure"
    out: list[ExitConfigV2] = []
    plan = [
        # (sl, partial_size, partial_R, kind, k, ema, hold)
        (1.5, 0.0, 0.0, TREND_STRUCTURE_TOP, None, None, 40),
        (2.0, 0.0, 0.0, TREND_STRUCTURE_TOP, None, None, 40),
        (1.5, 0.0, 0.0, TREND_STRUCTURE_MID, None, None, 40),
        (2.0, 0.0, 0.0, TREND_STRUCTURE_MID, None, None, 40),
        (1.5, 0.0, 0.0, TREND_EMA, None, 10, 40),
        (2.0, 0.0, 0.0, TREND_EMA, None, 10, 40),
        (1.5, 0.0, 0.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (2.0, 0.0, 0.0, TREND_ATR_TRAIL, 2.0, None, 40),
        (1.5, 0.50, 1.0, TREND_STRUCTURE_TOP, None, None, 40),
        (2.0, 0.50, 1.0, TREND_STRUCTURE_TOP, None, None, 40),
        (1.5, 0.50, 1.0, TREND_EMA, None, 10, 40),
        (2.0, 0.50, 1.0, TREND_EMA, None, 10, 40),
    ]
    for sl, ps, pR, kind, k, ema, hold in plan:
        trend = _trend_tag(kind, k, None, ema)
        name = _make_name(family, sl, _partial_tag(ps, pR), None, 0.0, trend, hold)
        out.append(ExitConfigV2(
            name=name, family=family,
            initial_sl_atr=sl,
            partial_size=ps, partial_R=pR,
            partial2_size=0.0, partial2_R=0.0,
            breakeven_R=None, activation_R=0.0,
            trend_kind=kind, trend_atr_k=k, trend_giveback=None,
            trend_ema_period=ema, max_hold_bars=hold,
        ))
    return out


# ---------- F5: MFE-capture (giveback with explicit activation) ---------------

def _build_F5() -> list[ExitConfigV2]:
    """MFE-giveback exits with activation gate.

    Cartesian over: activation_R ∈ {0.5, 1.0} × sl ∈ {1.5, 2.0}
    × partial ∈ {off, 33%@+1R} × giveback ∈ {0.20, 0.30, 0.40} × hold=40.
    BE dropped (priority cut) and max_hold pinned to 40 (priority cut).
    => 24 variants.
    """
    family = "F5_mfe_capture"
    out: list[ExitConfigV2] = []
    activations = [0.5, 1.0]
    sls = [1.5, 2.0]
    partials = [(0.0, 0.0), (0.33, 1.0)]  # (size, R)
    givebacks = [0.20, 0.30, 0.40]
    hold = 40
    for act in activations:
        for sl in sls:
            for ps, pR in partials:
                for g in givebacks:
                    trend = _trend_tag(TREND_MFE_GIVEBACK, k=None, g=g)
                    name = _make_name(family, sl, _partial_tag(ps, pR),
                                      None, act, trend, hold)
                    out.append(ExitConfigV2(
                        name=name, family=family,
                        initial_sl_atr=sl,
                        partial_size=ps, partial_R=pR,
                        partial2_size=0.0, partial2_R=0.0,
                        breakeven_R=None, activation_R=act,
                        trend_kind=TREND_MFE_GIVEBACK,
                        trend_atr_k=None, trend_giveback=g,
                        trend_ema_period=None, max_hold_bars=hold,
                    ))
    return out


# ---------- F6: tight ATR trail (sub-2.0 k) -----------------------------------

def _build_F6() -> list[ExitConfigV2]:
    """Tight ATR trail.

    Cartesian over: k ∈ {0.75, 1.0, 1.25} × sl ∈ {1.5, 2.0}
    × partial ∈ {off, 33%@+1R} × hold=40.
    => 12 variants.
    """
    family = "F6_tight_trail"
    out: list[ExitConfigV2] = []
    ks = [0.75, 1.0, 1.25]
    sls = [1.5, 2.0]
    partials = [(0.0, 0.0), (0.33, 1.0)]
    hold = 40
    for k in ks:
        for sl in sls:
            for ps, pR in partials:
                trend = _trend_tag(TREND_ATR_TRAIL, k=k)
                name = _make_name(family, sl, _partial_tag(ps, pR),
                                  None, 0.0, trend, hold)
                out.append(ExitConfigV2(
                    name=name, family=family,
                    initial_sl_atr=sl,
                    partial_size=ps, partial_R=pR,
                    partial2_size=0.0, partial2_R=0.0,
                    breakeven_R=None, activation_R=0.0,
                    trend_kind=TREND_ATR_TRAIL, trend_atr_k=k,
                    trend_giveback=None, trend_ema_period=None,
                    max_hold_bars=hold,
                ))
    return out


# ---------- F7: structure reclaim (+ EMA10 fallback as sibling variant) -------

def _build_F7() -> list[ExitConfigV2]:
    """Structure reclaim (close < box_top), with EMA10 as a sibling-variant
    "fallback" rather than a runtime composite.

    Each cell of (sl × partial × hold) gets two siblings: one using
    `structure_top`, one using `ema10`. Operator picks via validation
    which "fallback" form to lock.

    Cartesian: trend ∈ {struct_top, ema10} × sl ∈ {1.5, 2.0}
    × partial ∈ {off, 33%@+1R} × hold=40.
    => 8 variants.
    """
    family = "F7_structure_reclaim"
    out: list[ExitConfigV2] = []
    sls = [1.5, 2.0]
    partials = [(0.0, 0.0), (0.33, 1.0)]
    hold = 40
    # Structure top (primary)
    for sl in sls:
        for ps, pR in partials:
            trend = _trend_tag(TREND_STRUCTURE_TOP)
            name = _make_name(family, sl, _partial_tag(ps, pR),
                              None, 0.0, trend, hold)
            out.append(ExitConfigV2(
                name=name, family=family,
                initial_sl_atr=sl,
                partial_size=ps, partial_R=pR,
                partial2_size=0.0, partial2_R=0.0,
                breakeven_R=None, activation_R=0.0,
                trend_kind=TREND_STRUCTURE_TOP,
                trend_atr_k=None, trend_giveback=None,
                trend_ema_period=None, max_hold_bars=hold,
            ))
    # EMA10 (sibling fallback)
    for sl in sls:
        for ps, pR in partials:
            trend = _trend_tag(TREND_EMA, ema=10)
            name = _make_name(family, sl, _partial_tag(ps, pR),
                              None, 0.0, trend, hold)
            out.append(ExitConfigV2(
                name=name, family=family,
                initial_sl_atr=sl,
                partial_size=ps, partial_R=pR,
                partial2_size=0.0, partial2_R=0.0,
                breakeven_R=None, activation_R=0.0,
                trend_kind=TREND_EMA, trend_atr_k=None,
                trend_giveback=None, trend_ema_period=10,
                max_hold_bars=hold,
            ))
    return out


# ---------- F8: profit-lock ladder (capture-focused extension) ----------------

# Ladder constants — kept module-level so audits can see them at a glance.
_LADDER_L1 = ((1.0, 0.25), (2.0, 1.0), (3.0, 2.0))   # F8a, F8d
_LADDER_L2 = ((1.0, 0.40), (2.0, 1.20), (3.0, 2.20))  # F8b
_LADDER_L3 = ((1.5, 0.75), (2.5, 1.50), (3.5, 2.50))  # F8c


def _build_F8_subfamily(family: str, ladder: tuple, ladder_tag: str,
                        with_ema_trend: bool) -> list[ExitConfigV2]:
    """Build one F8 sub-family. 8 variants per call.

    Cartesian over: sl ∈ {1.5, 2.0} × partial ∈ {off, 33%@+1R}
    × hold ∈ {20, 40}.
    """
    out: list[ExitConfigV2] = []
    sls = [1.5, 2.0]
    partials = [(0.0, 0.0), (0.33, 1.0)]
    holds = [20, 40]
    for sl in sls:
        for ps, pR in partials:
            for hold in holds:
                if with_ema_trend:
                    trend_part = f"{ladder_tag}+ema10"
                    name = _make_name(family, sl, _partial_tag(ps, pR),
                                      None, 0.0, trend_part, hold)
                    out.append(ExitConfigV2(
                        name=name, family=family,
                        initial_sl_atr=sl,
                        partial_size=ps, partial_R=pR,
                        partial2_size=0.0, partial2_R=0.0,
                        breakeven_R=None, activation_R=0.0,
                        trend_kind=TREND_EMA, trend_atr_k=None,
                        trend_giveback=None, trend_ema_period=10,
                        max_hold_bars=hold,
                        profit_lock_ladder=ladder,
                    ))
                else:
                    trend_part = ladder_tag
                    name = _make_name(family, sl, _partial_tag(ps, pR),
                                      None, 0.0, trend_part, hold)
                    out.append(ExitConfigV2(
                        name=name, family=family,
                        initial_sl_atr=sl,
                        partial_size=ps, partial_R=pR,
                        partial2_size=0.0, partial2_R=0.0,
                        breakeven_R=None, activation_R=0.0,
                        max_hold_bars=hold,
                        profit_lock_ladder=ladder,
                    ))
    return out


def _build_F8a() -> list[ExitConfigV2]:
    return _build_F8_subfamily("F8a_lock_1R", _LADDER_L1, "lockL1",
                               with_ema_trend=False)


def _build_F8b() -> list[ExitConfigV2]:
    return _build_F8_subfamily("F8b_lock_1R_tighter", _LADDER_L2, "lockL2",
                               with_ema_trend=False)


def _build_F8c() -> list[ExitConfigV2]:
    return _build_F8_subfamily("F8c_lock_after_1_5R", _LADDER_L3, "lockL3",
                               with_ema_trend=False)


def _build_F8d() -> list[ExitConfigV2]:
    return _build_F8_subfamily("F8d_hybrid_lock_plus_trend", _LADDER_L1,
                               "lockL1", with_ema_trend=True)


# ---------- public ------------------------------------------------------------

def build_grid_v2() -> list[ExitConfigV2]:
    """Return the full F0..F8 grid (~142 variants).

    Order is deterministic; names are unique. The grid is hand-curated,
    not Cartesian — expanding it is a deliberate decision that should
    bump the discovery output suffix and be recorded in the reset
    methodology note.
    """
    grid = (
        _build_F0() + _build_F1() + _build_F2() + _build_F3()
        + _build_F4() + _build_F5() + _build_F6() + _build_F7()
        + _build_F8a() + _build_F8b() + _build_F8c() + _build_F8d()
    )
    names = [c.name for c in grid]
    if len(set(names)) != len(names):
        dups = [n for n in names if names.count(n) > 1]
        raise RuntimeError(f"duplicate variant names in grid: {sorted(set(dups))}")
    return grid


def grid_summary() -> dict[str, int]:
    """Variant count per family — useful for sanity output."""
    grid = build_grid_v2()
    counts: dict[str, int] = {}
    for cfg in grid:
        counts[cfg.family] = counts.get(cfg.family, 0) + 1
    counts["TOTAL"] = sum(v for k, v in counts.items() if k != "TOTAL")
    return counts


# ---------- public name resolver ---------------------------------------------

# Cached name -> ExitConfigV2 lookup. Built lazily so `import`-ing this module
# stays cheap; rebuilt only when `_LOOKUP_V2` is None (e.g. test reset).
_LOOKUP_V2: dict[str, ExitConfigV2] | None = None


def _v2_lookup() -> dict[str, ExitConfigV2]:
    global _LOOKUP_V2
    if _LOOKUP_V2 is None:
        _LOOKUP_V2 = {cfg.name: cfg for cfg in build_grid_v2()}
    return _LOOKUP_V2


def resolve_exit_spec(name: str) -> ExitConfigV2:
    """Resolve a v2 (F-prefixed) exit name to its `ExitConfigV2`.

    Legacy E3..E7 names are NOT v2 specs — they live in `sbt1700.exits`
    and use a different simulator entry point. Callers that need to
    accept either form should branch on the prefix and dispatch to the
    appropriate simulator.
    """
    lookup = _v2_lookup()
    cfg = lookup.get(name)
    if cfg is not None:
        return cfg
    families = sorted({n.split("_", 1)[0] for n in lookup.keys()})
    raise ValueError(
        f"unknown v2 exit name: {name!r}. "
        f"Available v2 family prefixes: {families}. "
        f"For legacy E3..E7 names use sbt1700.exits.simulate_exit instead."
    )


def is_v2_name(name: str) -> bool:
    """True iff `name` is a known v2 grid variant."""
    return name in _v2_lookup()


if __name__ == "__main__":
    s = grid_summary()
    for k, v in s.items():
        print(f"  {k:<28} {v}")
