"""HW Overlay v1 metric calculator.

Spec: memory/hw_overlay_v1_spec.md.

Metrics per (scanner × family × dot_class_slice):
  n_events, n_traded, entry_fill_rate
  WR (dual-condition: realized_R >= +0.10 AND holding <= 10 days)
  WR_random_p05/p50/p95, lift_vs_random
  mean_realized_R, median_realized_R
  mfe_capture (median realized_R / mfe_R)
  Sharpe_per_trade, Sortino_per_trade
  MaxDD_per_trade_sequence (cumulative-$1, equal-weight, entry-date order)
  mean_holding_period
  pct_HW_exit, pct_time_stop

3-column dot-class split: pooled (AL ∪ AL_OS), AL_only, AL_OS_only.

Random baseline: ×500 shuffles per scanner cohort. For each shuffle, replace
event_date with random date from HW range (uniform over universe trading days),
keep ticker frequency stable. Apply same HW overlay. Compute WR per shuffle.

Output: output/hw_overlay_v1_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.hw_overlay_v1_build import (
    DEFAULT_HW_CSV, DEFAULT_MASTER, WAIT_N_BY_SCANNER, DEFAULT_WAIT_N,
    build_trade, daily_panel_from_master, hw_events_by_ticker,
)

DEFAULT_EVENTS = "output/hw_overlay_v1_events.parquet"
DEFAULT_OUT = "output/hw_overlay_v1_metrics.csv"
N_SHUFFLES = 500
RNG_SEED = 20260502


def _slice_metrics(trades: pd.DataFrame, n_events: int) -> dict:
    if trades.empty:
        return {
            "n_events": n_events, "n_traded": 0, "entry_fill_rate": 0.0,
            "WR": np.nan, "mean_realized_R": np.nan, "median_realized_R": np.nan,
            "mfe_capture_median": np.nan,
            "Sharpe_per_trade": np.nan, "Sortino_per_trade": np.nan,
            "MaxDD_cumR": np.nan, "worst_trade_R": np.nan,
            "mean_holding_period": np.nan,
            "pct_HW_exit": np.nan, "pct_time_stop": np.nan,
            "WR_armA_aligned": np.nan, "mean_R_armA_aligned": np.nan,
            "WR_armC": np.nan, "mean_R_armC": np.nan,
            "WR_armD": np.nan, "mean_R_armD": np.nan, "mean_cycles_armD": np.nan,
        }
    n = len(trades)
    R = trades["realized_R"].values
    wins = trades["is_win"].sum()
    sharpe = (R.mean() / R.std(ddof=1)) * math.sqrt(252) if R.std(ddof=1) > 0 else np.nan
    downside = R[R < 0]
    sortino = (R.mean() / downside.std(ddof=1)) * math.sqrt(252) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    # cumulative-R additive sequence by entry-date order (avoids cumprod blow-up over many trades)
    seq = trades.sort_values("entry_date")
    eq = seq["realized_R"].cumsum()
    peak = eq.cummax()
    dd_abs = (eq - peak).min()  # most negative cumulative drop, in R-units
    # also worst single-trade R (a more interpretable per-trade tail metric)
    worst_trade_R = float(R.min())
    mfe_cap = (trades["realized_R"] / trades["mfe_R"]).replace([np.inf, -np.inf], np.nan).dropna()
    n_HW_exit = trades["exit_signal_kind"].isin(["SAT", "SAT_OB"]).sum()
    n_ts = (trades["exit_signal_kind"] == "time_stop").sum()
    # Arm A on the same trades (aligned slice — same events, scanner-alone outcome)
    armA_R_series = trades["armA_realized_R"].dropna() if "armA_realized_R" in trades else pd.Series(dtype=float)
    if len(armA_R_series) > 0:
        wr_armA_aligned = float((armA_R_series >= 0.10).mean())
        mean_R_armA_aligned = float(armA_R_series.mean())
    else:
        wr_armA_aligned = np.nan
        mean_R_armA_aligned = np.nan
    # Arm C on the same trades (HW entry, +10d fixed exit, no SAT)
    armC_R_series = trades["armC_realized_R"].dropna() if "armC_realized_R" in trades else pd.Series(dtype=float)
    if len(armC_R_series) > 0:
        wr_armC = float((armC_R_series >= 0.10).mean())
        mean_R_armC = float(armC_R_series.mean())
    else:
        wr_armC = np.nan
        mean_R_armC = np.nan
    # Arm D on the same trades (multi-cycle within 10d, compound R)
    armD_R_series = trades["armD_realized_R"].dropna() if "armD_realized_R" in trades else pd.Series(dtype=float)
    if len(armD_R_series) > 0:
        wr_armD = float((armD_R_series >= 0.10).mean())
        mean_R_armD = float(armD_R_series.mean())
        mean_cycles_armD = float(trades["armD_n_cycles"].mean()) if "armD_n_cycles" in trades else np.nan
    else:
        wr_armD = np.nan
        mean_R_armD = np.nan
        mean_cycles_armD = np.nan
    return {
        "n_events": n_events, "n_traded": n,
        "entry_fill_rate": n / n_events if n_events > 0 else np.nan,
        "WR": wins / n,
        "mean_realized_R": float(R.mean()),
        "median_realized_R": float(np.median(R)),
        "mfe_capture_median": float(mfe_cap.median()) if len(mfe_cap) > 0 else np.nan,
        "Sharpe_per_trade": sharpe,
        "Sortino_per_trade": sortino,
        "MaxDD_cumR": float(dd_abs),
        "worst_trade_R": worst_trade_R,
        "mean_holding_period": float(trades["holding_days"].mean()),
        "pct_HW_exit": n_HW_exit / n,
        "pct_time_stop": n_ts / n,
        "WR_armA_aligned": wr_armA_aligned,
        "mean_R_armA_aligned": mean_R_armA_aligned,
        "WR_armC": wr_armC,
        "mean_R_armC": mean_R_armC,
        "WR_armD": wr_armD,
        "mean_R_armD": mean_R_armD,
        "mean_cycles_armD": mean_cycles_armD,
    }


def _build_cache(
    panel: dict, hw_by_ticker: dict, wait_n_values: list[int],
) -> dict[int, np.ndarray]:
    """Precompute per (ticker, date, wait_n) → (entry_filled, is_win) flags.

    Returns dict[wait_n] → ndarray of dtype:
      [('ticker', 'U16'), ('date_idx', 'i8'), ('filled', 'i1'), ('win', 'i1')]
    Each ndarray row corresponds to one (ticker, date) tuple.
    """
    cache: dict[int, list] = {wn: [] for wn in wait_n_values}
    n_total = sum(len(g) for g in panel.values())
    print(f"[cache] precomputing {n_total:,} (ticker,date) tuples × {len(wait_n_values)} wait_n values …")
    for tk, daily in panel.items():
        hw = hw_by_ticker.get(tk, pd.DataFrame())
        for di, date in enumerate(daily["date"]):
            ev = pd.Series({"ticker": tk, "event_date": pd.Timestamp(date)})
            for wn in wait_n_values:
                tr = build_trade(ev, daily, hw, wn)
                if tr is None:
                    cache[wn].append((tk, di, 0, 0))
                else:
                    cache[wn].append((tk, di, int(tr["entry_filled"]), int(tr["is_win"])))
    out = {}
    for wn, rows in cache.items():
        out[wn] = np.array(rows, dtype=[("ticker", "U16"), ("date_idx", "i8"), ("filled", "i1"), ("win", "i1")])
        print(f"[cache] wait_n={wn}: {len(rows):,} rows; filled_rate={out[wn]['filled'].mean():.4f}; raw_WR={out[wn]['win'].sum()/max(out[wn]['filled'].sum(),1):.4f}")
    return out


def _random_wr_dist(
    cohort_size: int,
    cache_arr: np.ndarray,
    n_shuffles: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if cohort_size == 0 or len(cache_arr) == 0:
        return (np.nan, np.nan, np.nan)
    n_keys = len(cache_arr)
    wrs = np.empty(n_shuffles, dtype=np.float64)
    filled_arr = cache_arr["filled"]
    win_arr = cache_arr["win"]
    for s in range(n_shuffles):
        idxs = rng.integers(0, n_keys, size=cohort_size)
        f = filled_arr[idxs]
        w = win_arr[idxs]
        traded = int(f.sum())
        wrs[s] = (w.sum() / traded) if traded > 0 else 0.0
    return (float(np.quantile(wrs, 0.05)), float(np.quantile(wrs, 0.50)), float(np.quantile(wrs, 0.95)))


def run(
    events_path: str = DEFAULT_EVENTS,
    hw_csv: str = DEFAULT_HW_CSV,
    master_path: str = DEFAULT_MASTER,
    out_path: str = DEFAULT_OUT,
    n_shuffles: int = N_SHUFFLES,
) -> pd.DataFrame:
    print(f"[load] {events_path}")
    df = pd.read_parquet(events_path)
    print(f"[load] {len(df):,} rows; filled={df['entry_filled'].sum():,}")

    print("[panel] building random baseline universe …")
    panel = daily_panel_from_master(master_path)
    hw_by_ticker = hw_events_by_ticker(hw_csv)
    wait_ns = sorted(set(list(WAIT_N_BY_SCANNER.values()) + [DEFAULT_WAIT_N]))
    cache = _build_cache(panel, hw_by_ticker, wait_ns)

    rng = np.random.default_rng(RNG_SEED)

    rows = []
    cohorts = df.groupby(["scanner", "family"])
    for (scanner, family), g in cohorts:
        wait_n = WAIT_N_BY_SCANNER.get(scanner, DEFAULT_WAIT_N)
        n_events = len(g)
        # Cohort-level Arm A on ALL events (filled + unfilled): scanner-alone baseline
        armA_all = g["armA_realized_R"].dropna() if "armA_realized_R" in g else pd.Series(dtype=float)
        if len(armA_all) > 0:
            wr_armA_all = float((armA_all >= 0.10).mean())
            mean_R_armA_all = float(armA_all.mean())
            n_armA_all = int(len(armA_all))
        else:
            wr_armA_all = np.nan
            mean_R_armA_all = np.nan
            n_armA_all = 0
        traded = g[g["entry_filled"]]
        for slice_name, slice_df in [
            ("pooled", traded),
            ("AL_only", traded[traded["entry_signal_kind"] == "AL"]),
            ("AL_OS_only", traded[traded["entry_signal_kind"] == "AL_OS"]),
        ]:
            m = _slice_metrics(slice_df, n_events)
            m["scanner"] = scanner
            m["family"] = family
            m["slice"] = slice_name
            m["wait_n"] = wait_n
            m["WR_armA_all_events"] = wr_armA_all
            m["mean_R_armA_all_events"] = mean_R_armA_all
            m["n_armA_all_events"] = n_armA_all
            # lift columns (HW timing on aligned slice; HW total vs all-events baseline)
            wr_b = m["WR"]
            wr_b_ok = isinstance(wr_b, float) and not np.isnan(wr_b)
            wr_aligned = m["WR_armA_aligned"]
            aligned_ok = isinstance(wr_aligned, float) and not np.isnan(wr_aligned) and wr_aligned > 0
            all_ok = isinstance(wr_armA_all, float) and not np.isnan(wr_armA_all) and wr_armA_all > 0
            m["lift_HW_timing"] = (wr_b / wr_aligned) if (wr_b_ok and aligned_ok) else np.nan
            m["lift_HW_total"] = (wr_b / wr_armA_all) if (wr_b_ok and all_ok) else np.nan
            m["delta_WR_vs_aligned"] = (wr_b - wr_aligned) if (wr_b_ok and isinstance(wr_aligned, float) and not np.isnan(wr_aligned)) else np.nan
            m["delta_WR_vs_all"] = (wr_b - wr_armA_all) if (wr_b_ok and isinstance(wr_armA_all, float) and not np.isnan(wr_armA_all)) else np.nan
            # Arm C decomposition: HW filter only (entry on AL, no SAT exit, +10d fixed)
            wr_c = m["WR_armC"]
            wr_c_ok = isinstance(wr_c, float) and not np.isnan(wr_c)
            m["lift_HW_filter_only"] = (wr_c / wr_armA_all) if (wr_c_ok and all_ok) else np.nan
            m["lift_HW_filter_aligned"] = (wr_c / wr_aligned) if (wr_c_ok and aligned_ok) else np.nan
            # Cost of HW SAT: B vs C (same entry, different exit). >1 = SAT helps; <1 = SAT hurts
            m["lift_satexit_vs_fixed"] = (wr_b / wr_c) if (wr_b_ok and wr_c_ok and wr_c > 0) else np.nan
            # Arm D vs others
            wr_d = m["WR_armD"]
            wr_d_ok = isinstance(wr_d, float) and not np.isnan(wr_d)
            m["lift_multicycle_vs_B"] = (wr_d / wr_b) if (wr_d_ok and wr_b_ok and wr_b > 0) else np.nan
            m["lift_multicycle_vs_A_all"] = (wr_d / wr_armA_all) if (wr_d_ok and all_ok) else np.nan
            rows.append(m)
        # Random baseline computed once per cohort (pooled trades)
        p05, p50, p95 = _random_wr_dist(
            cohort_size=n_events,
            cache_arr=cache[wait_n],
            n_shuffles=n_shuffles,
            rng=rng,
        )
        for r in rows[-3:]:
            r["WR_random_p05"] = p05
            r["WR_random_p50"] = p50
            r["WR_random_p95"] = p95
            if r["WR"] is not None and not (isinstance(r["WR"], float) and np.isnan(r["WR"])) and not np.isnan(p50) and p50 > 0:
                r["lift_vs_random"] = r["WR"] / p50
            else:
                r["lift_vs_random"] = np.nan
        wr_obs = rows[-3]["WR"]
        wr_str = f"{wr_obs:.4f}" if isinstance(wr_obs, float) and not np.isnan(wr_obs) else "NaN"
        wa_str = f"{wr_armA_all:.4f}" if not np.isnan(wr_armA_all) else "NaN"
        p50_str = f"{p50:.4f}" if not np.isnan(p50) else "NaN"
        print(f"[cohort] {scanner}/{family} N={n_events} pooled WR={wr_str} armA_all={wa_str} rnd_p50={p50_str}")

    out = pd.DataFrame(rows)
    cols_order = [
        "scanner", "family", "slice", "wait_n",
        "n_events", "n_traded", "entry_fill_rate",
        "WR", "WR_random_p05", "WR_random_p50", "WR_random_p95", "lift_vs_random",
        "WR_armA_all_events", "WR_armA_aligned", "WR_armC", "WR_armD",
        "lift_HW_total", "lift_HW_timing",
        "lift_HW_filter_only", "lift_HW_filter_aligned", "lift_satexit_vs_fixed",
        "lift_multicycle_vs_B", "lift_multicycle_vs_A_all",
        "delta_WR_vs_all", "delta_WR_vs_aligned",
        "mean_realized_R", "median_realized_R",
        "mean_R_armA_all_events", "mean_R_armA_aligned", "mean_R_armC", "mean_R_armD", "mean_cycles_armD",
        "mfe_capture_median",
        "Sharpe_per_trade", "Sortino_per_trade", "MaxDD_cumR", "worst_trade_R",
        "mean_holding_period", "pct_HW_exit", "pct_time_stop",
        "n_armA_all_events",
    ]
    out = out[cols_order]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {len(out):,} rows → {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=DEFAULT_EVENTS)
    ap.add_argument("--hw-csv", default=DEFAULT_HW_CSV)
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    args = ap.parse_args()
    run(args.events, args.hw_csv, args.master, args.out, args.n_shuffles)


if __name__ == "__main__":
    main()
