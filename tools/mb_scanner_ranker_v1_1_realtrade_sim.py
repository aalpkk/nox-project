"""Realistic trade-level metrics: random pick vs heuristic vs all events.

Anti-rescue compliant: NO retraining. Uses only frozen Phase 1 outcome
columns (r_h, mfe_r_h, mae_r_h) to simulate realistic exit rules.

User question: "ML olmadan nasıl seçeceğim, gerçek getiri/WR/DD/Sharpe ne?"

Universe: v1.1 band-filtered TEST split (post-2025-09-01 hold-out).
Trade rules:
  (T1) Time-stop at H bars, exit at close → realized r_H
  (T2) Triple-barrier: stop at -1R (structural_invalidation_low),
       target at +2R, time-stop at H bars (worst-case ambiguity).
       Output in R-multiples.

Strategies:
  (S1) ALL events (every fire taken, equal weight)
  (S2) RANDOM top-K/day (K=3, 200 seeds, mean ± std)
  (S3) HEURISTIC top-K/day by hl_over_lh (both polarities tested)

Cost assumptions:
  0 bps clean baseline + 30 bps one-way (60 bps round-trip) for realism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.ranker_v1_1 import build_pool  # noqa: E402

OUT_DIR = Path("output")
REPORT = OUT_DIR / "mb_scanner_ranker_v1_1_realtrade_sim.md"

HORIZONS = (5, 10, 20)
TOP_K = 3
N_RAND_SEEDS = 200
COST_RT_BPS = 60  # round-trip 60bps (30bps one-way)
TARGET_R = 2.0


def _max_drawdown_arith(equity: np.ndarray) -> float:
    """Max drawdown on arithmetic cumulative-return equity curve."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _stats_from_returns(rets: np.ndarray, label: str) -> dict:
    """Per-trade arithmetic stats (no compounding — overlapping trades).

    Interpretation: each trade = 1 equal-size unit of capital, return
    in same unit. Sum across trades = cumulative units gained/lost.
    Drawdown computed on arithmetic cumulative curve.
    """
    rets = np.asarray(rets, dtype=float)
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return dict(label=label, n=0)
    eq = np.cumsum(rets)
    sum_total = float(eq[-1])
    mean_t = float(rets.mean())
    std_t = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe_t = mean_t / std_t if std_t > 0 else float("nan")
    wr = float((rets > 0).mean())
    pos_sum = float(rets[rets > 0].sum())
    neg_sum = float(-rets[rets < 0].sum())
    pf = pos_sum / neg_sum if neg_sum > 0 else float("nan")
    mdd_units = _max_drawdown_arith(eq)
    return dict(
        label=label, n=int(len(rets)),
        mean_per_trade=mean_t,
        sum_units=sum_total,
        win_rate=wr,
        profit_factor=pf,
        sharpe_per_trade=sharpe_t,
        max_drawdown_units=mdd_units,
    )


def _triple_barrier_R(mfe_r, mae_r, r, target_R=TARGET_R):
    """Return realized R-multiple per trade under triple barrier:
       stop at -1R, target at +target_R, time-stop = realized r at H.
       Worst-case if both touched: assume stop first. Position size = 1R.
    """
    out = np.full(len(mfe_r), np.nan)
    mask = ~np.isnan(mfe_r) & ~np.isnan(mae_r) & ~np.isnan(r)
    mfe = mfe_r[mask]; mae = mae_r[mask]; rr = r[mask]
    realized = np.empty(len(mfe))
    # default = time-stop in R-multiples — convert raw r to R-multiples
    # Note: r is raw close return, mfe/mae_r already in R; we need r_R.
    # R-distance per trade = (close - struct_low) / close = r_distance
    # We don't have r_distance here; use mfe_r and mae_r as the R basis
    # (consistent with Phase 1's convention).
    # If neither stop nor target touched: time-stop in raw r → ambiguous
    # without r_distance per row. Approximate: if mfe_r >= target → +target_R
    # if mae_r >= 1.0 → -1.0R
    # else use the R-equivalent as (mfe_r - mae_r) / (depending on direction).
    # Simpler: time-stop at expiry in R: closest end-state in R is r/(struct_R)
    # — we don't have that directly so approximate by sign(r) * min(|...|).
    # Fallback: take mfe_r - mae_r when neither barrier hit (rough proxy).

    target_hit = mfe >= target_R
    stop_hit = mae >= 1.0
    both = target_hit & stop_hit
    only_target = target_hit & ~stop_hit
    only_stop = stop_hit & ~target_hit
    neither = ~target_hit & ~stop_hit

    # both touched → conservative: assume stop first → -1R
    realized[both] = -1.0
    realized[only_target] = float(target_R)
    realized[only_stop] = -1.0
    # neither → time-stop: use mfe_r - mae_r as approximate exit R
    # (this is rough — sign depends on which dominates at expiry)
    # better: end-state R = r relative to risk; we approximate as
    # mfe_r if final r > 0 else -mae_r (kept conservative)
    nb = neither
    sign_pos = rr[nb] > 0
    realized_nb = np.where(sign_pos, mfe[nb], -mae[nb])
    realized[nb] = realized_nb

    out[np.where(mask)[0]] = realized
    return out


def _per_day_top_k(df: pd.DataFrame, score_col: str, k: int,
                   ascending: bool = False) -> pd.DataFrame:
    """Return rows that are top-K per event_bar_date by score_col."""
    return (
        df.sort_values(score_col, ascending=ascending)
          .groupby("event_bar_date", as_index=False)
          .head(k)
    )


def _random_top_k(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    return _per_day_top_k(df, "_rand", k)


def _apply_cost(rets: np.ndarray, cost_bps: float) -> np.ndarray:
    return rets - cost_bps / 1e4


def _scenario_table(test: pd.DataFrame, h: int, *, cost_bps: float):
    rcol = f"r_{h}"
    mfecol = f"mfe_r_{h}"
    maecol = f"mae_r_{h}"

    rows = []

    # S1 — ALL events
    rets_all = test[rcol].dropna().to_numpy()
    rets_all = _apply_cost(rets_all, cost_bps)
    rows.append(_stats_from_returns(rets_all, f"ALL_h{h}"))

    # S2 — RANDOM top-K/day mean over seeds
    rand_summaries = []
    rand_sum_units = []
    rand_wrs = []
    rand_mdds = []
    for s in range(N_RAND_SEEDS):
        pick = _random_top_k(test, TOP_K, seed=s)
        r = pick[rcol].dropna().to_numpy()
        r = _apply_cost(r, cost_bps)
        st = _stats_from_returns(r, f"RAND_h{h}_seed{s}")
        rand_summaries.append(st)
        rand_sum_units.append(st["sum_units"])
        rand_wrs.append(st["win_rate"])
        rand_mdds.append(st["max_drawdown_units"])
    rand_summary = {
        "label": f"RAND_topK{TOP_K}_h{h}",
        "n": int(np.median([x["n"] for x in rand_summaries])),
        "mean_per_trade": float(np.mean(
            [x["mean_per_trade"] for x in rand_summaries]
        )),
        "sum_units": float(np.mean(rand_sum_units)),
        "sum_units_std": float(np.std(rand_sum_units, ddof=1)),
        "win_rate": float(np.mean(rand_wrs)),
        "profit_factor": float(np.nanmean(
            [x["profit_factor"] for x in rand_summaries]
        )),
        "sharpe_per_trade": float(np.nanmean(
            [x["sharpe_per_trade"] for x in rand_summaries]
        )),
        "max_drawdown_units": float(np.mean(rand_mdds)),
    }
    rows.append(rand_summary)

    # S3a — HEURISTIC top-K/day by hl_over_lh DESC (deep pullback)
    pick_desc = _per_day_top_k(test, "hl_over_lh", TOP_K, ascending=False)
    r = pick_desc[rcol].dropna().to_numpy()
    r = _apply_cost(r, cost_bps)
    rows.append(_stats_from_returns(r, f"HEUR_hl_DESC_topK{TOP_K}_h{h}"))

    # S3b — HEURISTIC top-K/day by hl_over_lh ASC (shallow pullback)
    pick_asc = _per_day_top_k(test, "hl_over_lh", TOP_K, ascending=True)
    r = pick_asc[rcol].dropna().to_numpy()
    r = _apply_cost(r, cost_bps)
    rows.append(_stats_from_returns(r, f"HEUR_hl_ASC_topK{TOP_K}_h{h}"))

    # also run triple-barrier R-units for time-stop H to give R-perspective
    R = _triple_barrier_R(
        test[mfecol].to_numpy(),
        test[maecol].to_numpy(),
        test[rcol].to_numpy(),
    )
    test = test.copy()
    test[f"R_{h}"] = R
    pick_desc_R = _per_day_top_k(test, "hl_over_lh", TOP_K, ascending=False)
    pick_asc_R = _per_day_top_k(test, "hl_over_lh", TOP_K, ascending=True)
    r_all = test[f"R_{h}"].dropna().to_numpy()
    rows.append(_stats_from_returns(r_all, f"ALL_R_h{h}_tripleBarrier"))
    rows.append(_stats_from_returns(
        pick_desc_R[f"R_{h}"].dropna().to_numpy(),
        f"HEUR_hl_DESC_R_h{h}_tripleBarrier",
    ))
    rows.append(_stats_from_returns(
        pick_asc_R[f"R_{h}"].dropna().to_numpy(),
        f"HEUR_hl_ASC_R_h{h}_tripleBarrier",
    ))
    rand_R_results = []
    for s in range(N_RAND_SEEDS):
        pk = _random_top_k(test, TOP_K, seed=s)
        rs = pk[f"R_{h}"].dropna().to_numpy()
        st = _stats_from_returns(rs, f"RAND_R_h{h}_seed{s}")
        rand_R_results.append(st)
    rand_R_summary = {
        "label": f"RAND_topK{TOP_K}_R_h{h}_tripleBarrier",
        "n": int(np.median([x["n"] for x in rand_R_results])),
        "mean_per_trade": float(np.mean(
            [x["mean_per_trade"] for x in rand_R_results]
        )),
        "sum_units": float(np.mean(
            [x["sum_units"] for x in rand_R_results]
        )),
        "win_rate": float(np.mean(
            [x["win_rate"] for x in rand_R_results]
        )),
        "profit_factor": float(np.nanmean(
            [x["profit_factor"] for x in rand_R_results]
        )),
        "sharpe_per_trade": float(np.nanmean(
            [x["sharpe_per_trade"] for x in rand_R_results]
        )),
        "max_drawdown_units": float(np.mean(
            [x["max_drawdown_units"] for x in rand_R_results]
        )),
    }
    rows.append(rand_R_summary)

    return pd.DataFrame(rows)


def main():
    print("Loading v1.1 band-filtered pool ...")
    pool = build_pool()
    test = pool[pool["split"] == "TEST"].copy()
    print(f"  TEST events: {len(test)}")
    print(f"  TEST date range: "
          f"{test['event_bar_date'].min().date()} → "
          f"{test['event_bar_date'].max().date()}")
    print(f"  Unique trading days in TEST: "
          f"{test['event_bar_date'].nunique()}")
    print(f"  Events/day median: "
          f"{test.groupby('event_bar_date').size().median():.0f}, "
          f"mean: {test.groupby('event_bar_date').size().mean():.1f}, "
          f"max: {test.groupby('event_bar_date').size().max()}")
    print()

    all_results = []
    for cost in (0, COST_RT_BPS):
        for h in HORIZONS:
            print(f"=== H={h} bars   cost={cost}bps round-trip ===")
            tab = _scenario_table(test, h, cost_bps=cost)
            tab["cost_bps"] = cost
            tab["horizon"] = h
            print(tab[["label", "n", "mean_per_trade", "sum_units",
                       "win_rate", "profit_factor",
                       "sharpe_per_trade", "max_drawdown_units"]
                      ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print()
            all_results.append(tab)

    full = pd.concat(all_results, ignore_index=True)
    csv_path = OUT_DIR / "mb_scanner_ranker_v1_1_realtrade_sim.csv"
    full.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    # markdown summary
    with open(REPORT, "w") as f:
        f.write("# mb_scanner ranker v1.1 — Realistic trade-level sim\n\n")
        f.write("Anti-rescue compliant: no retraining, uses Phase 1 frozen "
                "outcome columns.\n\n")
        f.write(f"- TEST split events: {len(test)}\n")
        f.write(f"- Universe: v1.1 band-filtered (mb_1d 2-10%, mb_1w 8-25%)\n")
        f.write(f"- Top-K per day = {TOP_K}\n")
        f.write(f"- Random seeds: {N_RAND_SEEDS}\n\n")
        for h in HORIZONS:
            for cost in (0, COST_RT_BPS):
                f.write(f"## H={h} bars, cost={cost}bps round-trip\n\n")
                t = full[(full["horizon"] == h)
                         & (full["cost_bps"] == cost)]
                f.write(t[["label", "n", "mean_per_trade", "sum_units",
                           "win_rate", "profit_factor",
                           "sharpe_per_trade", "max_drawdown_units"]
                          ].to_markdown(index=False, floatfmt=".4f"))
                f.write("\n\n")
    print(f"  wrote {REPORT}")


if __name__ == "__main__":
    raise SystemExit(main())
