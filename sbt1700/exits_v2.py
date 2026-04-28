"""SBT-1700 — exits_v2: controlled exit-discovery simulator.

This module is independent of the legacy E3..E7 simulator in `exits.py`
(kept for backward reference). It supports the families F0..F4 used by
the train-only discovery grid:

Primitives:
- initial stop      : k_sl * ATR_1700               (k_sl ∈ {1.0, 1.5, 2.0})
- partial leg 1     : at +pa_R, locks pa_size       (pa_size ∈ {0, 0.33, 0.50})
- partial leg 2     : at +pb_R, locks pb_size       (used by stepout family)
- breakeven         : after MFE_R ≥ be_R, stop ← entry
- runner activation : trend rules silent until MFE_R ≥ act_R
- trend exit family : one of
    * atr_trail     : exit on close < highest_close_since_entry − k * ATR_1700
    * mfe_giveback  : exit when (MFE_R − current_R_close) / MFE_R ≥ g, MFE_R > 0
    * ema           : exit on close < EMA(period) of bootstrapped daily closes
    * structure_top : exit on close < box_top
    * structure_mid : exit on close < (box_top + box_bottom) / 2
- max hold          : timeout exit at close of bar `max_hold_bars`

Conventions (locked):
- LONG only.
- R = k_sl * ATR_1700, fixed at entry.
- Forward window starts at T+1; T's close_1700 is the entry price; T's
  daily bar is never consumed for label evaluation.
- Same-bar precedence: SL fills before partial / TP / trail (worst case).
  Partial + SL same bar: partial locks at its level, runner takes SL.
- Trend rules evaluated on close ONLY; trail uses highest forward close.
- Activation gates trend rules only — initial stop and partial fire
  regardless of activation.
- EMA bootstrap: seed EMA state with the mean of the last `period` prior
  daily closes (`daily_master`, dates < entry_date), so EMA is well-
  defined from forward bar 1.
- Costs: 5 bps slippage + 5 bps commission per side, applied as a
  constant cost_R over the round-trip notional. Identical contract to
  the legacy simulator.

This module deliberately exposes no defaults beyond the primitive
container. Family construction lives in `sbt1700/exit_grid.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


SLIPPAGE_BPS_PER_SIDE = 5.0
COMMISSION_BPS_PER_SIDE = 5.0

# Trend exit kinds — kept as string constants to keep dataclass hashable.
TREND_NONE = "none"
TREND_ATR_TRAIL = "atr_trail"
TREND_MFE_GIVEBACK = "mfe_giveback"
TREND_EMA = "ema"
TREND_STRUCTURE_TOP = "structure_top"
TREND_STRUCTURE_MID = "structure_mid"
VALID_TREND_KINDS = {
    TREND_NONE, TREND_ATR_TRAIL, TREND_MFE_GIVEBACK, TREND_EMA,
    TREND_STRUCTURE_TOP, TREND_STRUCTURE_MID,
}


@dataclass(frozen=True)
class ExitConfigV2:
    """Single exit variant. Fields are interpretable in isolation.

    `name` and `family` are metadata used by the grid + selection
    layer. The simulator does not interpret them.

    `profit_lock_ladder` is an ordered tuple of (mfe_threshold_R,
    locked_stop_R) rungs. When MFE crosses a rung's threshold by the
    end of a bar, the runner stop ratchets up to entry + locked_R * R
    for all subsequent bars. Rungs are monotonic — a triggered rung
    cannot be un-armed and lock_R must be non-decreasing across the
    ladder. Same-bar pessimism is preserved by arming at end-of-bar
    (the SL check on this bar uses the previous bar's floor). Set to
    None or empty tuple to disable.
    """
    name: str
    family: str
    initial_sl_atr: float
    partial_size: float = 0.0
    partial_R: float = 0.0
    partial2_size: float = 0.0
    partial2_R: float = 0.0
    breakeven_R: Optional[float] = None
    activation_R: float = 0.0
    trend_kind: str = TREND_NONE
    trend_atr_k: Optional[float] = None
    trend_giveback: Optional[float] = None  # fraction in [0, 1], e.g. 0.35
    trend_ema_period: Optional[int] = None
    max_hold_bars: int = 40
    profit_lock_ladder: Optional[tuple] = None

    def __post_init__(self):
        if self.trend_kind not in VALID_TREND_KINDS:
            raise ValueError(f"unknown trend_kind {self.trend_kind!r}")
        if self.trend_kind == TREND_ATR_TRAIL and self.trend_atr_k is None:
            raise ValueError("trend_atr_k required for atr_trail")
        if self.trend_kind == TREND_MFE_GIVEBACK and self.trend_giveback is None:
            raise ValueError("trend_giveback required for mfe_giveback")
        if self.trend_kind == TREND_EMA and self.trend_ema_period is None:
            raise ValueError("trend_ema_period required for ema")
        if self.partial_size < 0 or self.partial_size > 1:
            raise ValueError(f"partial_size out of range: {self.partial_size}")
        if self.partial2_size < 0 or self.partial2_size > 1:
            raise ValueError(f"partial2_size out of range: {self.partial2_size}")
        if self.partial_size + self.partial2_size > 1.0 + 1e-9:
            raise ValueError("partial_size + partial2_size must be ≤ 1")
        if self.initial_sl_atr <= 0:
            raise ValueError("initial_sl_atr must be > 0")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be > 0")
        if self.profit_lock_ladder is not None and len(self.profit_lock_ladder) > 0:
            prev_th = -float("inf")
            prev_lock = -float("inf")
            for rung in self.profit_lock_ladder:
                if not (isinstance(rung, tuple) and len(rung) == 2):
                    raise ValueError(
                        f"profit_lock_ladder rung must be (mfe_R, lock_R), "
                        f"got {rung!r}")
                mfe_th, lock_R = float(rung[0]), float(rung[1])
                if mfe_th <= 0:
                    raise ValueError(
                        f"profit_lock rung mfe_R must be > 0, got {mfe_th}")
                if lock_R >= mfe_th:
                    raise ValueError(
                        f"profit_lock rung lock_R ({lock_R}) must be < mfe_R "
                        f"({mfe_th}) — locking at-or-above the trigger is "
                        "degenerate")
                if mfe_th <= prev_th:
                    raise ValueError(
                        "profit_lock_ladder mfe_R thresholds must be strictly "
                        "increasing")
                if lock_R < prev_lock:
                    raise ValueError(
                        "profit_lock_ladder lock_R must be non-decreasing "
                        "(monotonic ratchet)")
                prev_th, prev_lock = mfe_th, lock_R


def _bad_label(cfg: ExitConfigV2, reason: str, **extra) -> dict:
    base = {
        "exit_variant": cfg.name,
        "exit_family": cfg.family,
        "realized_R_gross": np.nan,
        "realized_R_net": np.nan,
        "win_label": False,
        "initial_stop_hit": False,
        "breakeven_stop_hit": False,
        "profit_lock_stop_hit": False,
        "partial_hit": False,
        "partial2_hit": False,
        "trend_exit_hit": False,
        "max_hold_hit": False,
        "exit_reason": reason,
        "bars_held": 0,
        "MFE_R": np.nan,
        "giveback_R": np.nan,
        "captured_MFE_ratio": np.nan,
        "entry_px": np.nan,
        "stop_px": np.nan,
        "partial_px": np.nan,
        "partial2_px": np.nan,
        "atr_1700": np.nan,
        "initial_R_price": np.nan,
        "exit_px": np.nan,
        "exit_date": pd.NaT,
        "cost_R": np.nan,
    }
    base.update(extra)
    return base


def _bootstrap_ema(prior_closes: np.ndarray, period: int) -> float:
    """Seed EMA with the mean of last `period` prior closes."""
    if prior_closes.size == 0:
        return float("nan")
    take = prior_closes[-period:]
    return float(np.mean(take))


def _ema_step(prev_ema: float, x: float, period: int) -> float:
    if not np.isfinite(prev_ema):
        return float(x)
    alpha = 2.0 / (period + 1.0)
    return float(prev_ema + alpha * (x - prev_ema))


def simulate_exit_v2(
    cfg: ExitConfigV2,
    entry_date: pd.Timestamp,
    entry_px: float,
    atr_1700: float,
    forward_ohlc: pd.DataFrame,
    prior_closes: np.ndarray,
    box_top: Optional[float] = None,
    box_bottom: Optional[float] = None,
) -> dict:
    """Run a single (variant, signal) simulation.

    Parameters
    ----------
    cfg : ExitConfigV2
    entry_date : pd.Timestamp
        T (the signal date). Forward bars are dates strictly greater.
    entry_px : float
        17:00 truncated close used as the long entry price.
    atr_1700 : float
        ATR(14) on EOD-complete prior daily bars (matches legacy field
        `atr14_prior`).
    forward_ohlc : pd.DataFrame
        Daily OHLC indexed by date for the same ticker. Caller filters
        to dates > entry_date if desired; this function re-filters
        defensively and truncates to `cfg.max_hold_bars`.
    prior_closes : np.ndarray
        Daily closes prior to entry_date (chronological). Used only to
        bootstrap the EMA trend rule.
    box_top, box_bottom : Optional[float]
        Required only for structure_top / structure_mid trend kinds.
    """
    if not (np.isfinite(entry_px) and np.isfinite(atr_1700) and atr_1700 > 0):
        return _bad_label(cfg, "invalid_entry_or_atr",
                          entry_px=float(entry_px) if np.isfinite(entry_px) else np.nan,
                          atr_1700=float(atr_1700) if np.isfinite(atr_1700) else np.nan)

    R = cfg.initial_sl_atr * atr_1700
    if R <= 0:
        return _bad_label(cfg, "nonpositive_R", entry_px=float(entry_px), atr_1700=float(atr_1700))

    initial_stop_px = entry_px - R
    partial_px = entry_px + cfg.partial_R * R if cfg.partial_size > 0 else float("nan")
    partial2_px = entry_px + cfg.partial2_R * R if cfg.partial2_size > 0 else float("nan")
    breakeven_px = entry_px  # if breakeven activates

    # Structure trend: precompute level required by trend_kind.
    structure_level = None
    if cfg.trend_kind == TREND_STRUCTURE_TOP:
        if box_top is None or not np.isfinite(box_top):
            return _bad_label(cfg, "missing_box_top",
                              entry_px=float(entry_px), atr_1700=float(atr_1700),
                              stop_px=float(initial_stop_px), initial_R_price=float(R))
        structure_level = float(box_top)
    elif cfg.trend_kind == TREND_STRUCTURE_MID:
        if (box_top is None or box_bottom is None or
                not np.isfinite(box_top) or not np.isfinite(box_bottom)):
            return _bad_label(cfg, "missing_box_mid",
                              entry_px=float(entry_px), atr_1700=float(atr_1700),
                              stop_px=float(initial_stop_px), initial_R_price=float(R))
        structure_level = 0.5 * (float(box_top) + float(box_bottom))

    fwd = forward_ohlc[forward_ohlc.index > entry_date].head(cfg.max_hold_bars)
    if fwd.empty:
        return _bad_label(cfg, "no_forward_bars",
                          entry_px=float(entry_px), atr_1700=float(atr_1700),
                          stop_px=float(initial_stop_px), initial_R_price=float(R))

    # EMA bootstrap (only if needed).
    ema_state = float("nan")
    if cfg.trend_kind == TREND_EMA:
        ema_state = _bootstrap_ema(np.asarray(prior_closes, dtype=float), cfg.trend_ema_period)
        # If we have no prior history at all, EMA is undefined → treat as bad row.
        if not np.isfinite(ema_state):
            return _bad_label(cfg, "missing_ema_seed",
                              entry_px=float(entry_px), atr_1700=float(atr_1700),
                              stop_px=float(initial_stop_px), initial_R_price=float(R))

    # State machine.
    locked_R = 0.0
    runner_size = 1.0
    breakeven_active = False
    runner_active = (cfg.activation_R == 0.0)
    partial_filled = (cfg.partial_size == 0.0)
    partial2_filled = (cfg.partial2_size == 0.0)
    initial_stop_hit = False
    breakeven_stop_hit = False
    profit_lock_stop_hit = False
    partial_hit = False
    partial2_hit = False
    trend_exit_hit = False
    max_hold_hit = False
    # Ratcheted profit-lock floor; armed end-of-bar so the SL check
    # within a bar uses the floor as it stood at the prior bar's close.
    lock_floor_px = -float("inf")
    has_lock_ladder = bool(cfg.profit_lock_ladder)

    max_high = entry_px       # MFE tracked from forward highs
    max_close = entry_px      # for ATR trail (highest forward close)
    exit_px: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    bars_held = 0
    realized_R_gross = float("nan")
    exit_reason = "open"

    for i, (dt, row) in enumerate(fwd.iterrows(), 1):
        bars_held = i
        hi = float(row["High"])
        lo = float(row["Low"])
        cl = float(row["Close"])

        max_high = max(max_high, hi)
        # Effective stop reflects breakeven and profit-lock that activated on a PRIOR bar.
        effective_stop = initial_stop_px
        if breakeven_active:
            effective_stop = max(effective_stop, breakeven_px)
        if lock_floor_px > -float("inf"):
            effective_stop = max(effective_stop, lock_floor_px)

        hit_sl = lo <= effective_stop
        hit_partial = (not partial_filled) and (hi >= partial_px) if cfg.partial_size > 0 else False
        hit_partial2 = (not partial2_filled) and (hi >= partial2_px) if cfg.partial2_size > 0 else False

        # 1) Worst-case same-bar SL: SL takes precedence over partial/trend on this bar.
        if hit_sl:
            # Locked partial legs already locked from prior bars (locked_R).
            # No new partial fills this bar (worst case).
            runner_R = (effective_stop - entry_px) / R
            realized_R_gross = locked_R + runner_size * runner_R
            exit_px = float(effective_stop)
            exit_date = dt
            # Attribute the SL to its highest contributor:
            #   profit_lock_stop > breakeven_stop > initial_stop.
            # Profit-lock floors are >= breakeven_R (locks ride above entry);
            # we tag profit_lock_stop when the lock floor pinned the stop
            # strictly above breakeven_px.
            lock_pin = (lock_floor_px > -float("inf") and
                        effective_stop >= lock_floor_px - 1e-12 and
                        lock_floor_px > breakeven_px + 1e-12)
            if lock_pin:
                profit_lock_stop_hit = True
                exit_reason = "profit_lock_stop"
            elif breakeven_active and effective_stop >= entry_px - 1e-12:
                breakeven_stop_hit = True
                exit_reason = "breakeven_stop"
            else:
                initial_stop_hit = True
                exit_reason = "initial_stop"
            break

        # 2) Partial fills (in order: leg 1 then leg 2 if ordered by R level).
        # Determine fill order if both possible this bar.
        partial_events: list[tuple[float, str]] = []
        if hit_partial:
            partial_events.append((cfg.partial_R, "p1"))
        if hit_partial2:
            partial_events.append((cfg.partial2_R, "p2"))
        partial_events.sort(key=lambda t: t[0])
        for _, tag in partial_events:
            if tag == "p1":
                locked_R += cfg.partial_size * cfg.partial_R
                runner_size -= cfg.partial_size
                partial_filled = True
                partial_hit = True
            else:
                locked_R += cfg.partial2_size * cfg.partial2_R
                runner_size -= cfg.partial2_size
                partial2_filled = True
                partial2_hit = True
        # Numerical clean-up.
        if runner_size < 1e-9:
            runner_size = 0.0

        # If runner is fully closed by partials, exit at last fill price (use the higher partial level).
        if runner_size == 0.0:
            last_R = max((ev[0] for ev in partial_events), default=0.0)
            exit_px = float(entry_px + last_R * R)
            realized_R_gross = locked_R  # runner contribution is 0
            exit_date = dt
            exit_reason = "fully_partialled_out"
            break

        # 3) Trend exit (only if runner_active was True at start of this bar).
        if runner_active and cfg.trend_kind != TREND_NONE:
            # Update trail close BEFORE evaluating trail rule.
            new_max_close = max(max_close, cl)
            ema_now = ema_state
            if cfg.trend_kind == TREND_EMA:
                ema_now = _ema_step(ema_state, cl, cfg.trend_ema_period)

            trend_fire = False
            if cfg.trend_kind == TREND_ATR_TRAIL:
                trail_level = new_max_close - cfg.trend_atr_k * atr_1700
                if cl < trail_level:
                    trend_fire = True
            elif cfg.trend_kind == TREND_MFE_GIVEBACK:
                mfe_R_now = (max_high - entry_px) / R
                cur_R = (cl - entry_px) / R
                if mfe_R_now > 0:
                    if (mfe_R_now - cur_R) / mfe_R_now >= cfg.trend_giveback:
                        trend_fire = True
            elif cfg.trend_kind == TREND_EMA:
                if cl < ema_now:
                    trend_fire = True
            elif cfg.trend_kind == TREND_STRUCTURE_TOP:
                if cl < structure_level:
                    trend_fire = True
            elif cfg.trend_kind == TREND_STRUCTURE_MID:
                if cl < structure_level:
                    trend_fire = True

            # Commit state updates.
            max_close = new_max_close
            ema_state = ema_now

            if trend_fire:
                runner_R = (cl - entry_px) / R
                realized_R_gross = locked_R + runner_size * runner_R
                exit_px = float(cl)
                exit_date = dt
                trend_exit_hit = True
                exit_reason = f"trend_{cfg.trend_kind}"
                break
        else:
            # Even when runner is inactive, keep ATR-trail high-water and EMA state warm
            # so that on activation we don't reset them.
            max_close = max(max_close, cl)
            if cfg.trend_kind == TREND_EMA:
                ema_state = _ema_step(ema_state, cl, cfg.trend_ema_period)

        # 4) End-of-bar: arm breakeven, runner activation, profit-lock for NEXT bar.
        cur_max_R = (max_high - entry_px) / R
        if cfg.breakeven_R is not None and not breakeven_active:
            if cur_max_R >= cfg.breakeven_R:
                breakeven_active = True
        if not runner_active and cur_max_R >= cfg.activation_R:
            runner_active = True
        if has_lock_ladder:
            for mfe_th, lock_R in cfg.profit_lock_ladder:
                if cur_max_R >= mfe_th:
                    new_floor = entry_px + lock_R * R
                    if new_floor > lock_floor_px:
                        lock_floor_px = new_floor

    if exit_px is None:
        # Max-hold timeout exit at close of last bar.
        max_hold_hit = True
        last_dt = fwd.index[-1]
        last_close = float(fwd.loc[last_dt, "Close"])
        runner_R = (last_close - entry_px) / R
        realized_R_gross = locked_R + runner_size * runner_R
        exit_px = float(last_close)
        exit_date = last_dt
        exit_reason = "max_hold"

    # MFE / giveback diagnostics (in R).
    mfe_R = (max_high - entry_px) / R
    realized_R = realized_R_gross
    giveback_R = max(0.0, mfe_R - realized_R) if np.isfinite(realized_R) else float("nan")
    captured_ratio = (realized_R / mfe_R) if (np.isfinite(realized_R) and mfe_R > 1e-9) else float("nan")

    cost_bps = (SLIPPAGE_BPS_PER_SIDE + COMMISSION_BPS_PER_SIDE) * 2
    cost_pct = cost_bps / 1e4
    cost_R = (entry_px * cost_pct) / R
    realized_R_net = realized_R_gross - cost_R

    return {
        "exit_variant": cfg.name,
        "exit_family": cfg.family,
        "realized_R_gross": float(realized_R_gross),
        "realized_R_net": float(realized_R_net),
        "win_label": bool(realized_R_net > 0),
        "initial_stop_hit": bool(initial_stop_hit),
        "breakeven_stop_hit": bool(breakeven_stop_hit),
        "profit_lock_stop_hit": bool(profit_lock_stop_hit),
        "partial_hit": bool(partial_hit),
        "partial2_hit": bool(partial2_hit),
        "trend_exit_hit": bool(trend_exit_hit),
        "max_hold_hit": bool(max_hold_hit),
        "exit_reason": exit_reason,
        "bars_held": int(bars_held),
        "MFE_R": float(mfe_R),
        "giveback_R": float(giveback_R) if np.isfinite(giveback_R) else float("nan"),
        "captured_MFE_ratio": float(captured_ratio) if np.isfinite(captured_ratio) else float("nan"),
        "entry_px": float(entry_px),
        "stop_px": float(initial_stop_px),
        "partial_px": float(partial_px) if np.isfinite(partial_px) else float("nan"),
        "partial2_px": float(partial2_px) if np.isfinite(partial2_px) else float("nan"),
        "atr_1700": float(atr_1700),
        "initial_R_price": float(R),
        "exit_px": float(exit_px),
        "exit_date": pd.Timestamp(exit_date),
        "cost_R": float(cost_R),
    }
