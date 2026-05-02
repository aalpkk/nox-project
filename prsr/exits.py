"""Stop / time-stop / TP-appendix exit machinery.

For each candidate at fire date T, simulate forward bars t ∈ [T+1, T+N].

Primary exit rule:
  - if low[t] <= initial_stop: exit at initial_stop on bar t (assume fillable)
  - if t == T + time_stop_horizon: exit at close[t]

Side variants (5/20-bar time stops, TP 2R) reuse the same forward path
and produce additional exit columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C


def simulate_trade(
    forward: pd.DataFrame,
    entry_price: float,
    initial_stop: float,
    horizon: int,
) -> tuple[float, str, int]:
    """Walk forward bars; return (exit_price, exit_kind, holding_days).

    exit_kind ∈ {'stop', 'time'}.
    forward must be sorted ascending by date and have columns low, close.
    """
    if len(forward) == 0:
        return (np.nan, "no_forward", 0)
    n = min(horizon, len(forward))
    for i in range(n):
        bar = forward.iloc[i]
        if bar["low"] <= initial_stop:
            return (initial_stop, "stop", i + 1)
    last = forward.iloc[n - 1]
    return (float(last["close"]), "time", n)


def simulate_tp(
    forward: pd.DataFrame,
    entry_price: float,
    initial_stop: float,
    tp_price: float,
    horizon: int,
) -> tuple[float, str, int]:
    """TP variant: take profit at tp_price if high[t] >= tp before stop or time."""
    if len(forward) == 0:
        return (np.nan, "no_forward", 0)
    n = min(horizon, len(forward))
    for i in range(n):
        bar = forward.iloc[i]
        if bar["low"] <= initial_stop and bar["high"] >= tp_price:
            # ambiguous — assume worst-case stop fills first
            return (initial_stop, "stop_tp_amb", i + 1)
        if bar["low"] <= initial_stop:
            return (initial_stop, "stop", i + 1)
        if bar["high"] >= tp_price:
            return (tp_price, "tp", i + 1)
    last = forward.iloc[n - 1]
    return (float(last["close"]), "time", n)


def evaluate_trade(
    panel_for_ticker: pd.DataFrame,
    fire_idx: int,
    entry_mode: str,
    pattern_low: float,
    atr20_at_T: float,
) -> dict | None:
    """Compute primary + side exits for a single (ticker, fire_date, entry_mode) triple.

    panel_for_ticker is the chronologically sorted daily slice for one ticker.
    fire_idx is the integer index in that slice corresponding to fire date T.

    Returns dict with primary + 5/20 + tp side columns, or None if entry impossible.
    """
    if fire_idx + 1 >= len(panel_for_ticker):
        return None
    if entry_mode == "open_T1":
        entry_row = panel_for_ticker.iloc[fire_idx + 1]
        entry_price = float(entry_row["open"])
        forward = panel_for_ticker.iloc[fire_idx + 1 :].reset_index(drop=True)
        # for open_T1, the entry bar IS forward[0] — stop check applies from bar 0 onwards
        # using its own low (could fill same day if open opens below stop)
        # convention: the entry bar's low is checked too, since stop is intraday
    elif entry_mode == "close_T":
        entry_row = panel_for_ticker.iloc[fire_idx]
        entry_price = float(entry_row["close"])
        forward = panel_for_ticker.iloc[fire_idx + 1 :].reset_index(drop=True)
    else:
        raise ValueError(f"unknown entry_mode {entry_mode}")

    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    if not np.isfinite(pattern_low) or not np.isfinite(atr20_at_T) or atr20_at_T <= 0:
        return None

    initial_stop = pattern_low - C.INITIAL_STOP_ATR_MULT * atr20_at_T
    if initial_stop >= entry_price:
        return None  # invalid stop above entry; skip
    risk_per_unit = entry_price - initial_stop
    tp_price = entry_price + C.TP_R_MULTIPLE * risk_per_unit

    primary_exit, primary_kind, primary_days = simulate_trade(
        forward, entry_price, initial_stop, C.TIME_STOP_PRIMARY
    )
    side_exits = {}
    for h in C.TIME_STOP_SIDE:
        ex, kind, days = simulate_trade(forward, entry_price, initial_stop, h)
        side_exits[f"exit_t{h}"] = ex
        side_exits[f"exit_t{h}_kind"] = kind
        side_exits[f"exit_t{h}_days"] = days
    tp_exit, tp_kind, tp_days = simulate_tp(
        forward, entry_price, initial_stop, tp_price, C.TIME_STOP_PRIMARY
    )

    def ret(exit_price: float) -> float:
        if not np.isfinite(exit_price) or entry_price <= 0:
            return np.nan
        return (exit_price - entry_price) / entry_price

    def r_mult(exit_price: float) -> float:
        if not np.isfinite(exit_price) or risk_per_unit <= 0:
            return np.nan
        return (exit_price - entry_price) / risk_per_unit

    return {
        "entry_mode": entry_mode,
        "entry_price": entry_price,
        "initial_stop": initial_stop,
        "risk_per_unit": risk_per_unit,
        "tp_price": tp_price,
        # primary
        "exit_primary": primary_exit,
        "exit_primary_kind": primary_kind,
        "exit_primary_days": primary_days,
        "ret_primary": ret(primary_exit),
        "r_primary": r_mult(primary_exit),
        # side time stops
        **side_exits,
        "ret_t5": ret(side_exits["exit_t5"]),
        "r_t5": r_mult(side_exits["exit_t5"]),
        "ret_t20": ret(side_exits["exit_t20"]),
        "r_t20": r_mult(side_exits["exit_t20"]),
        # TP appendix
        "exit_tp": tp_exit,
        "exit_tp_kind": tp_kind,
        "exit_tp_days": tp_days,
        "ret_tp": ret(tp_exit),
        "r_tp": r_mult(tp_exit),
    }
