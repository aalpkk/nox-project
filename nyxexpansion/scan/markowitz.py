"""
Combinatorial 4-stock max-Sharpe Markowitz portfolio for nyxexpansion scan.

Spec (user-locked 2026-04-24):
  - Universe: top-15 scan candidates (winR-sorted, risk_bucket != severe)
  - 4-stock subsets of these 15  → C(15,4) = 1365 combos
  - 60-day daily returns (log)
  - Long-only, weight ∈ [0.10, 0.50], sum(w)=1
  - Objective: max Sharpe
  - Output: best combo's weights + expected return/risk/Sharpe
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha.portfolio import _ledoit_wolf_shrink

LOOKBACK_DAYS = 60
WEIGHT_MIN = 0.10
WEIGHT_MAX = 0.50
PORTFOLIO_SIZE = 4
RISK_FREE_RATE = 0.0  # simpler — raw Sharpe, rf=0

OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")


def _load_returns(tickers: list[str], as_of_date: pd.Timestamp,
                  lookback: int = LOOKBACK_DAYS) -> tuple[pd.DataFrame, list[str]]:
    """Load daily log returns for `tickers` ending at `as_of_date`, last `lookback` bars.

    Returns (log_returns_df, kept_tickers). Tickers with <lookback bars or any NaN drop out.
    """
    if not OHLCV_PATH.exists():
        return pd.DataFrame(), []

    oh = pd.read_parquet(OHLCV_PATH)
    if oh.index.name:
        oh = oh.reset_index()
    if "Date" not in oh.columns:
        oh["Date"] = pd.to_datetime(oh.iloc[:, 0])
    oh["Date"] = pd.to_datetime(oh["Date"])

    closes = {}
    for t in tickers:
        g = oh[(oh["ticker"] == t) & (oh["Date"] <= as_of_date)].sort_values("Date")
        if len(g) < lookback + 2:
            continue
        closes[t] = g["Close"].iloc[-(lookback + 1):].values

    if len(closes) < PORTFOLIO_SIZE:
        return pd.DataFrame(), []

    # Align by position (same trading days per ticker — assume common calendar)
    df = pd.DataFrame(closes)
    lr = np.log(df / df.shift(1)).dropna()
    if len(lr) < lookback - 5:
        return pd.DataFrame(), []
    return lr, list(df.columns)


def _optimize_combo(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve max-Sharpe for a single 4-asset combo. Returns (weights, sharpe)."""
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 1e10
        return -(port_ret - RISK_FREE_RATE) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(WEIGHT_MIN, WEIGHT_MAX)] * n
    w0 = np.ones(n) / n

    try:
        res = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if res.success and np.isfinite(res.fun):
            w = res.x
            sharpe = -float(res.fun)
            return w, sharpe
    except Exception:
        pass
    return w0, -float("inf")


def combinatorial_max_sharpe(
    candidate_tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
    portfolio_size: int = PORTFOLIO_SIZE,
) -> dict:
    """Enumerate 4-stock subsets of `candidate_tickers`, return best max-Sharpe combo.

    Returns dict with keys:
      tickers, weights, expected_return (annualized %), expected_risk (annualized %),
      sharpe, per_stock_stats (per-ticker 60d mean/vol annualized),
      universe_used (tickers that had data), combos_evaluated
    """
    lr, kept = _load_returns(candidate_tickers, as_of_date, lookback=lookback)
    empty = {
        "tickers": [], "weights": {}, "expected_return": 0.0, "expected_risk": 0.0,
        "sharpe": 0.0, "per_stock_stats": {}, "universe_used": kept, "combos_evaluated": 0,
        "error": None,
    }
    if lr.empty or len(kept) < portfolio_size:
        empty["error"] = f"insufficient data (have {len(kept)} tickers with {lookback}d history)"
        return empty

    # Annualized μ and Σ
    mu_all = lr.mean().values * 252
    cov_all_sample = lr.cov().values * 252
    n_obs = len(lr)
    cov_all = _ledoit_wolf_shrink(cov_all_sample, n_obs)

    # Per-stock stats for HTML
    per = {}
    for i, t in enumerate(kept):
        per[t] = {
            "mean_ann_pct": round(mu_all[i] * 100, 2),
            "vol_ann_pct": round(float(np.sqrt(cov_all[i, i])) * 100, 2),
            "last_return_pct": round(lr[t].iloc[-1] * 100, 2),
            "obs": int(n_obs),
        }

    # Enumerate all size-`portfolio_size` combos
    best = {"sharpe": -float("inf"), "w": None, "idx": None}
    combos_evaluated = 0
    for combo in combinations(range(len(kept)), portfolio_size):
        combos_evaluated += 1
        idx = list(combo)
        sub_mu = mu_all[idx]
        sub_cov = cov_all[np.ix_(idx, idx)]
        w, sharpe = _optimize_combo(sub_mu, sub_cov)
        if sharpe > best["sharpe"]:
            best = {"sharpe": sharpe, "w": w, "idx": idx}

    if best["w"] is None:
        empty["error"] = "no feasible combination"
        empty["combos_evaluated"] = combos_evaluated
        return empty

    sel_tickers = [kept[i] for i in best["idx"]]
    weights = {t: round(float(w), 4) for t, w in zip(sel_tickers, best["w"])}
    sub_mu = mu_all[best["idx"]]
    sub_cov = cov_all[np.ix_(best["idx"], best["idx"])]
    w_arr = best["w"]
    port_ret = float(w_arr @ sub_mu)
    port_risk = float(np.sqrt(w_arr @ sub_cov @ w_arr))
    sharpe = (port_ret - RISK_FREE_RATE) / port_risk if port_risk > 1e-10 else 0.0

    return {
        "tickers": sel_tickers,
        "weights": weights,
        "expected_return": round(port_ret * 100, 2),
        "expected_risk": round(port_risk * 100, 2),
        "sharpe": round(sharpe, 3),
        "per_stock_stats": per,
        "universe_used": kept,
        "combos_evaluated": combos_evaluated,
        "lookback_days": lookback,
        "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
        "error": None,
    }
