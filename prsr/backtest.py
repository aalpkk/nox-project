"""End-to-end orchestration.

Pipeline:
  1. universe.build_universe — daily panel + tier classification
  2. regime.add_regime_features + attach_cross_sectional_thresholds
  3. patterns.add_pattern_features + detect_pattern_a/b
  4. oscillator.add_oscillator_features (DIAGNOSTIC ONLY)
  5. candidate gate: tier==core ∧ regime_pass ∧ (A ∨ B)
  6. evaluate_trade per (candidate, entry_mode)
  7. random_baseline build
  8. aggregation, drawdown, concentration, watchable list
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C
from prsr import universe, regime, patterns, oscillator
from prsr.exits import evaluate_trade
from prsr.random_baseline import build_random_baseline


def build_full_panel() -> pd.DataFrame:
    """Daily panel + universe tiers + regime + pattern + oscillator features."""
    panel = universe.build_universe()
    panel = regime.add_regime_features(panel)
    panel = regime.attach_cross_sectional_thresholds(panel)
    panel = patterns.add_pattern_features(panel)
    panel = oscillator.add_oscillator_features(panel)

    a = patterns.detect_pattern_a(panel)
    b = patterns.detect_pattern_b(panel)
    panel = patterns.attach_pattern_low(panel, a, b)

    panel["pattern_pass"] = a | b
    panel["regime_pass"] = regime.regime_pass(panel)
    panel["universe_pass"] = panel["tier"].eq("core")
    panel["candidate"] = panel["universe_pass"] & panel["regime_pass"] & panel["pattern_pass"]
    panel["osc_confirmed"] = oscillator.osc_confirmed(panel)

    return panel


def evaluate_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    """Build per-trade rows for every PRSR candidate × entry_mode."""
    panel_sorted = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = {t: g.reset_index(drop=True) for t, g in panel_sorted.groupby("ticker", sort=False)}
    cand = panel_sorted[panel_sorted["candidate"]].copy()

    rows = []
    for _, row in cand.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        tg = by_ticker[ticker]
        mask = tg["date"] == date
        if not mask.any():
            continue
        fire_idx = int(np.flatnonzero(mask.values)[0])
        pl = float(row["pattern_low"])
        atr_T = float(row["atr20"])
        for entry_mode in ("open_T1", "close_T"):
            tr = evaluate_trade(tg, fire_idx, entry_mode, pl, atr_T)
            if tr is None:
                continue
            tr["ticker"] = ticker
            tr["fire_date"] = date
            tr["pattern_kind"] = row["pattern_kind"]
            tr["pattern_low"] = pl
            tr["atr20"] = atr_T
            tr["osc_confirmed"] = bool(row["osc_confirmed"])
            tr["source"] = "prsr"
            rows.append(tr)
    return pd.DataFrame(rows)


def aggregate_metrics(trades: pd.DataFrame, ret_col: str = "ret_primary") -> dict:
    """PF_proxy, realized_med, win_rate, n, MaxDD proxy."""
    if len(trades) == 0:
        return {"n": 0, "pf": np.nan, "realized_med": np.nan, "win_rate": np.nan, "max_dd": np.nan}
    r = trades[ret_col].dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    pf = float(pos / neg) if neg > 0 else float("inf")
    realized_med = float(r.median())
    win_rate = float((r > 0).mean())
    # MaxDD proxy: chronological cumulative equity over fire_date
    sorted_t = trades.sort_values("fire_date")
    eq = sorted_t[ret_col].fillna(0).cumsum()
    peak = eq.cummax()
    dd = (eq - peak).min()
    max_dd = float(dd) if np.isfinite(dd) else np.nan
    return {
        "n": int(len(r)),
        "pf": pf,
        "realized_med": realized_med,
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


def aggregate_yearly(trades: pd.DataFrame, ret_col: str = "ret_primary") -> pd.DataFrame:
    out = trades.copy()
    out["year"] = pd.to_datetime(out["fire_date"]).dt.year
    rows = []
    for y, g in out.groupby("year"):
        m = aggregate_metrics(g, ret_col)
        m["year"] = int(y)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("year")


def top5_date_share(trades: pd.DataFrame, ret_col: str = "ret_primary") -> float:
    if len(trades) == 0:
        return float("nan")
    by_date = trades.groupby("fire_date")[ret_col].sum()
    by_date_abs = by_date.abs()
    total = by_date_abs.sum()
    if total <= 0:
        return float("nan")
    top5 = by_date_abs.sort_values(ascending=False).head(5).sum()
    return float(top5 / total)


def chronological_drawdown(trades: pd.DataFrame, ret_col: str = "ret_primary") -> pd.DataFrame:
    out = trades.sort_values("fire_date").copy()
    out["equity"] = out[ret_col].fillna(0).cumsum()
    out["peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak"]
    return out[["fire_date", "ticker", "ret_primary" if ret_col == "ret_primary" else ret_col,
                "equity", "peak", "drawdown"]]
