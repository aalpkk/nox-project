"""Frozen E3 execution simulator for SBT-1700.

Per-trade rule (LONG only, phase 1):
    entry_px = close_1700                     (last completed 15m close ≤ 16:45 TR)
    stop_px  = entry_px - 0.30 * atr_1700     (atr from prior daily bars only)
    tp_px    = entry_px + 1.00 * (entry_px - stop_px)
    timeout  = 5 daily bars

The exit window starts on T+1 (the entry day's post-17:00 OHLC is not
available daily-only, and using daily High/Low of T would peek at prior
session data — conservative choice is to skip T entirely).

If both High >= tp and Low <= stop touch on the SAME daily bar,
worst-case-fill is assumed: SL first. (Configurable in E3Params.)

Returns a dict per (ticker, signal_date):
    realized_R_gross, realized_R_net, win_label,
    tp_hit, sl_hit, timeout_hit, exit_reason, bars_held,
    entry_px, stop_px, tp_px, atr_1700, exit_px, exit_date.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from sbt1700.config import E3, E3Params


def simulate_e3(
    entry_date: pd.Timestamp,
    entry_px: float,
    atr_1700: float,
    forward_ohlc: pd.DataFrame,
    params: E3Params = E3,
) -> dict:
    """Simulate one E3 trade.

    Args:
        entry_date: T (signal date). The exit window starts at T+1.
        entry_px: 17:00-truncated close.
        atr_1700: ATR(14) computed from prior daily bars only (excludes T).
        forward_ohlc: daily OHLC indexed by date, must contain rows for at
            least T+1 .. T+timeout_bars in calendar order. Columns:
            Open, High, Low, Close.

    Output: dict with realized_R_gross/net, hit flags, exit_reason, etc.
    """
    if not (np.isfinite(entry_px) and np.isfinite(atr_1700) and atr_1700 > 0):
        return _bad_label("invalid_entry_or_atr", entry_px, atr_1700)

    stop_px = entry_px - params.sl_atr_mult * atr_1700
    initial_R = entry_px - stop_px
    if initial_R <= 0:
        return _bad_label("nonpositive_R", entry_px, atr_1700)

    tp_px = entry_px + params.tp_R_mult * initial_R

    # Forward-only window: rows AFTER T.
    fwd = forward_ohlc[forward_ohlc.index > entry_date].head(params.timeout_bars)
    if fwd.empty:
        return _bad_label("no_forward_bars", entry_px, atr_1700,
                          stop_px=stop_px, tp_px=tp_px)

    exit_px: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_reason: str = "open"
    tp_hit = sl_hit = timeout_hit = False
    bars_held = 0

    for i, (dt, row) in enumerate(fwd.iterrows(), 1):
        bars_held = i
        hi = float(row["High"])
        lo = float(row["Low"])
        cl = float(row["Close"])

        hit_tp = hi >= tp_px
        hit_sl = lo <= stop_px

        if hit_tp and hit_sl:
            # Both touched the same bar. Conservative: SL first.
            if params.worst_case_same_bar:
                sl_hit = True
                exit_px = stop_px
                exit_reason = "sl_same_bar_worst_case"
                exit_date = dt
                break
            else:
                tp_hit = True
                exit_px = tp_px
                exit_reason = "tp_same_bar_best_case"
                exit_date = dt
                break
        if hit_tp:
            tp_hit = True
            exit_px = tp_px
            exit_reason = "tp"
            exit_date = dt
            break
        if hit_sl:
            sl_hit = True
            exit_px = stop_px
            exit_reason = "sl"
            exit_date = dt
            break

    if exit_px is None:
        # Timeout: exit at the close of the last bar in the window.
        timeout_hit = True
        last_dt = fwd.index[-1]
        exit_px = float(fwd.loc[last_dt, "Close"])
        exit_reason = "timeout"
        exit_date = last_dt

    realized_R_gross = (exit_px - entry_px) / initial_R

    # Net of slippage + commission (both sides).
    cost_bps = (params.slippage_bps_per_side + params.commission_bps_per_side) * 2
    cost_pct = cost_bps / 1e4
    cost_R = (entry_px * cost_pct) / initial_R
    realized_R_net = realized_R_gross - cost_R

    return {
        "realized_R_gross": float(realized_R_gross),
        "realized_R_net": float(realized_R_net),
        "win_label": bool(realized_R_net > 0),
        "tp_hit": bool(tp_hit),
        "sl_hit": bool(sl_hit),
        "timeout_hit": bool(timeout_hit),
        "exit_reason": exit_reason,
        "bars_held": int(bars_held),
        "entry_px": float(entry_px),
        "stop_px": float(stop_px),
        "tp_px": float(tp_px),
        "atr_1700": float(atr_1700),
        "initial_R_price": float(initial_R),
        "exit_px": float(exit_px),
        "exit_date": pd.Timestamp(exit_date),
        "cost_R": float(cost_R),
    }


def _bad_label(reason: str, entry_px: float, atr_1700: float, **extra) -> dict:
    base = {
        "realized_R_gross": np.nan,
        "realized_R_net": np.nan,
        "win_label": False,
        "tp_hit": False,
        "sl_hit": False,
        "timeout_hit": False,
        "exit_reason": reason,
        "bars_held": 0,
        "entry_px": float(entry_px) if np.isfinite(entry_px) else np.nan,
        "stop_px": np.nan,
        "tp_px": np.nan,
        "atr_1700": float(atr_1700) if np.isfinite(atr_1700) else np.nan,
        "initial_R_price": np.nan,
        "exit_px": np.nan,
        "exit_date": pd.NaT,
        "cost_R": np.nan,
    }
    base.update(extra)
    return base


def params_dict() -> dict:
    """E3 params as a dict — embed in dataset metadata for provenance."""
    return asdict(E3)
