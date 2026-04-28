"""Multi-exit simulator for SBT-1700 edge diagnosis (PR-2).

Five exit variants are evaluated on the same entry signal so we can
separate the question 'is the entry edge real?' from the question
'what execution rule extracts that edge?'.

Variants:
    E3_baseline   — TP=1R, SL=0.30 ATR, timeout=5, no partial, no trail
    E4_wider_stop — TP=1R, SL=0.60 ATR, timeout=5, no partial, no trail
    E5_symmetric  — TP=1R, SL=1.00 ATR, timeout=5, no partial, no trail
    E6_time_exit  — no TP,  SL=0.60 ATR, timeout=5 close exit
    E7_partial    — 50% off at +1R, runner exits at SL=0.60 ATR or timeout

R is defined per variant as the SL distance (entry - sl_px). For E6
(no TP) R is also the SL distance, used to express realized return in
risk units.

Costs: 5 bps slippage + 5 bps commission per side, applied symmetrically
to round-trip notional. Partial fills do not change total round-trip
slippage on full notional, so cost_R is identical across variants.

Forward-window contract is the same as E3:
- exit window starts at T+1
- worst-case-fill on same-bar tag (SL first)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


SLIPPAGE_BPS_PER_SIDE = 5.0
COMMISSION_BPS_PER_SIDE = 5.0


@dataclass(frozen=True)
class ExitParams:
    name: str
    sl_atr_mult: float
    has_tp: bool
    tp_R_mult: float = 1.0
    timeout_bars: int = 5
    has_partial: bool = False
    partial_R_mult: float = 1.0
    partial_size: float = 0.5
    worst_case_same_bar: bool = True


EXIT_VARIANTS: dict[str, ExitParams] = {
    "E3_baseline":   ExitParams(name="E3_baseline",   sl_atr_mult=0.30, has_tp=True,  tp_R_mult=1.0),
    "E4_wider_stop": ExitParams(name="E4_wider_stop", sl_atr_mult=0.60, has_tp=True,  tp_R_mult=1.0),
    "E5_symmetric":  ExitParams(name="E5_symmetric",  sl_atr_mult=1.00, has_tp=True,  tp_R_mult=1.0),
    "E6_time_exit":  ExitParams(name="E6_time_exit",  sl_atr_mult=0.60, has_tp=False),
    "E7_partial":    ExitParams(name="E7_partial",    sl_atr_mult=0.60, has_tp=False,
                                has_partial=True, partial_R_mult=1.0, partial_size=0.5),
}


def variant_names() -> list[str]:
    return list(EXIT_VARIANTS.keys())


def simulate_exit(
    variant: str,
    entry_date: pd.Timestamp,
    entry_px: float,
    atr_1700: float,
    forward_ohlc: pd.DataFrame,
) -> dict:
    if variant not in EXIT_VARIANTS:
        raise KeyError(f"unknown exit variant: {variant}")
    p = EXIT_VARIANTS[variant]

    if not (np.isfinite(entry_px) and np.isfinite(atr_1700) and atr_1700 > 0):
        return _bad_label(p.name, "invalid_entry_or_atr", entry_px, atr_1700)

    sl_distance = p.sl_atr_mult * atr_1700
    stop_px = entry_px - sl_distance
    initial_R = sl_distance  # R is always the SL distance
    if initial_R <= 0:
        return _bad_label(p.name, "nonpositive_R", entry_px, atr_1700)

    tp_px = (entry_px + p.tp_R_mult * initial_R) if p.has_tp else np.nan
    partial_px = (entry_px + p.partial_R_mult * initial_R) if p.has_partial else np.nan

    fwd = forward_ohlc[forward_ohlc.index > entry_date].head(p.timeout_bars)
    if fwd.empty:
        return _bad_label(p.name, "no_forward_bars", entry_px, atr_1700,
                          stop_px=stop_px, tp_px=tp_px)

    # State for partial-runner accounting.
    partial_filled = False
    partial_R = 0.0          # locked R from the partial leg
    runner_size = 1.0        # fraction of position still open

    tp_hit = sl_hit = timeout_hit = partial_hit = False
    exit_reason = "open"
    exit_px: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    bars_held = 0
    realized_R_gross = np.nan

    for i, (dt, row) in enumerate(fwd.iterrows(), 1):
        bars_held = i
        hi = float(row["High"])
        lo = float(row["Low"])
        cl = float(row["Close"])

        hit_sl = lo <= stop_px

        if p.has_partial and not partial_filled:
            hit_partial = hi >= partial_px
            if hit_partial and hit_sl:
                if p.worst_case_same_bar:
                    # Partial fills at +1R, runner stops at -1R same bar.
                    partial_R = p.partial_size * p.partial_R_mult
                    runner_R = (1.0 - p.partial_size) * (-1.0)
                    realized_R_gross = partial_R + runner_R
                    partial_filled = True
                    partial_hit = True
                    sl_hit = True
                    exit_px = stop_px  # nominal — actual partial fill is partial_px
                    exit_reason = "partial_then_sl_same_bar"
                    exit_date = dt
                    break
                else:
                    partial_R = p.partial_size * p.partial_R_mult
                    runner_R = (1.0 - p.partial_size) * p.partial_R_mult
                    realized_R_gross = partial_R + runner_R
                    partial_filled = True
                    partial_hit = True
                    tp_hit = True
                    exit_px = partial_px
                    exit_reason = "partial_tp_same_bar_best_case"
                    exit_date = dt
                    break
            if hit_partial:
                partial_filled = True
                partial_hit = True
                partial_R = p.partial_size * p.partial_R_mult
                runner_size = 1.0 - p.partial_size
                # continue checking SL/timeout on remaining bars
                continue
            if hit_sl:
                sl_hit = True
                realized_R_gross = -1.0  # full size at SL
                exit_px = stop_px
                exit_reason = "sl"
                exit_date = dt
                break
            continue  # neither hit → next bar

        # Variants without partial OR runner phase after partial fill.
        if p.has_tp:
            hit_tp = hi >= tp_px
            if hit_tp and hit_sl:
                if p.worst_case_same_bar:
                    sl_hit = True
                    runner_R = -1.0
                    realized_R_gross = partial_R + runner_size * runner_R
                    exit_px = stop_px
                    exit_reason = "sl_same_bar_worst_case" if not partial_filled else "runner_sl_same_bar"
                    exit_date = dt
                    break
                else:
                    tp_hit = True
                    runner_R = p.tp_R_mult
                    realized_R_gross = partial_R + runner_size * runner_R
                    exit_px = tp_px
                    exit_reason = "tp_same_bar_best_case"
                    exit_date = dt
                    break
            if hit_tp:
                tp_hit = True
                runner_R = p.tp_R_mult
                realized_R_gross = partial_R + runner_size * runner_R
                exit_px = tp_px
                exit_reason = "tp"
                exit_date = dt
                break
            if hit_sl:
                sl_hit = True
                runner_R = -1.0
                realized_R_gross = partial_R + runner_size * runner_R
                exit_px = stop_px
                exit_reason = "runner_sl" if partial_filled else "sl"
                exit_date = dt
                break
        else:
            # E6 / E7 runner phase: only SL or timeout.
            if hit_sl:
                sl_hit = True
                runner_R = -1.0
                realized_R_gross = partial_R + runner_size * runner_R
                exit_px = stop_px
                exit_reason = "runner_sl" if partial_filled else "sl"
                exit_date = dt
                break

    if exit_px is None:
        # Timeout exit at the close of the last bar in the window.
        timeout_hit = True
        last_dt = fwd.index[-1]
        last_close = float(fwd.loc[last_dt, "Close"])
        runner_R = (last_close - entry_px) / initial_R
        realized_R_gross = partial_R + runner_size * runner_R
        exit_px = last_close
        exit_reason = "runner_timeout" if partial_filled else "timeout"
        exit_date = last_dt

    cost_bps = (SLIPPAGE_BPS_PER_SIDE + COMMISSION_BPS_PER_SIDE) * 2
    cost_pct = cost_bps / 1e4
    cost_R = (entry_px * cost_pct) / initial_R
    realized_R_net = realized_R_gross - cost_R

    return {
        "exit_variant": p.name,
        "realized_R_gross": float(realized_R_gross),
        "realized_R_net": float(realized_R_net),
        "win_label": bool(realized_R_net > 0),
        "tp_hit": bool(tp_hit),
        "sl_hit": bool(sl_hit),
        "timeout_hit": bool(timeout_hit),
        "partial_hit": bool(partial_hit),
        "exit_reason": exit_reason,
        "bars_held": int(bars_held),
        "entry_px": float(entry_px),
        "stop_px": float(stop_px),
        "tp_px": float(tp_px) if np.isfinite(tp_px) else np.nan,
        "partial_px": float(partial_px) if np.isfinite(partial_px) else np.nan,
        "atr_1700": float(atr_1700),
        "initial_R_price": float(initial_R),
        "exit_px": float(exit_px),
        "exit_date": pd.Timestamp(exit_date),
        "cost_R": float(cost_R),
    }


def _bad_label(variant: str, reason: str, entry_px: float, atr_1700: float, **extra) -> dict:
    base = {
        "exit_variant": variant,
        "realized_R_gross": np.nan,
        "realized_R_net": np.nan,
        "win_label": False,
        "tp_hit": False,
        "sl_hit": False,
        "timeout_hit": False,
        "partial_hit": False,
        "exit_reason": reason,
        "bars_held": 0,
        "entry_px": float(entry_px) if np.isfinite(entry_px) else np.nan,
        "stop_px": np.nan,
        "tp_px": np.nan,
        "partial_px": np.nan,
        "atr_1700": float(atr_1700) if np.isfinite(atr_1700) else np.nan,
        "initial_R_price": np.nan,
        "exit_px": np.nan,
        "exit_date": pd.NaT,
        "cost_R": np.nan,
    }
    base.update(extra)
    return base
