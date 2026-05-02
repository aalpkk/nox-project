"""hwo_mtf_v1 — single authorized pre-reg run.

Spec: hwo_mtf_v1/SPEC.md (LOCKED 2026-05-01,
hash b8f42e1642652ec5b7efde400268b274d5ee75272330fd8934a6882873d0e9cc).

This script executes ONE backtest of HW big-dot multi-TF turning points on
BIST 607 q60 Core, K∈{5,10,20}, primary K=10, primary trigger T_1d∧5h.

Outputs:
    output/hwo_mtf_v1_report.md
    output/hwo_mtf_v1_per_trade.csv
    output/hwo_mtf_v1_random_baseline.csv
    output/hwo_mtf_v1_aggregate.csv
    output/hwo_mtf_v1_verdict.json

NO HTML, NO ML, NO live cron, NO param sweep. Acceptance verdict is final;
salvage / error-analysis is a separate thread.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mb_scanner.resample import to_5h, to_weekly
from oscmatrix.components.hyperwave import compute_hyperwave
from prsr import config as PC
from prsr.universe import build_universe

# ---- LOCKED PARAMETERS (mirror SPEC; do NOT mutate) ------------------------
SPEC_HASH = "b8f42e1642652ec5b7efde400268b274d5ee75272330fd8934a6882873d0e9cc"
SEED = 0x4857_4F4D_5446_3031  # ASCII "HWO MTF01" packed
START_DATE = pd.Timestamp(PC.START_DATE)  # 2023-01-02
END_DATE = pd.Timestamp(PC.END_DATE)      # 2026-04-29
HW_LEN = 7
HW_SIG = 3
HORIZONS = (5, 10, 20)
PRIMARY_K = 10
CONFL_5H_WINDOW_DAYS = 3   # T_5h within {D-2, D-1, D} inclusive
CONFL_1W_WINDOW_DAYS = 5   # T_1w within ≤ 5 trading-day prior of D
RANDOM_DRAWS = 100
COVERAGE_FLOOR = 0.80      # per-TF per-year aggregate

# Acceptance gates (LOCKED §9)
G1_PF_MIN = 1.87
G2_RANK_PCT_MIN = 0.52
G3_KS_PVAL_MAX = 0.05
G4_N_MIN = 30

OUT_DIR = Path("output")
OUT_REPORT = OUT_DIR / "hwo_mtf_v1_report.md"
OUT_PER_TRADE = OUT_DIR / "hwo_mtf_v1_per_trade.csv"
OUT_RANDOM = OUT_DIR / "hwo_mtf_v1_random_baseline.csv"
OUT_AGGREGATE = OUT_DIR / "hwo_mtf_v1_aggregate.csv"
OUT_VERDICT = OUT_DIR / "hwo_mtf_v1_verdict.json"
OUT_COVERAGE = OUT_DIR / "hwo_mtf_v1_coverage.csv"
OUT_MANIFEST = OUT_DIR / "hwo_mtf_v1_manifest.json"


# ---- Utilities --------------------------------------------------------------

def pf_proxy(returns: np.ndarray) -> float:
    """sum(positive) / |sum(negative)|; same definition baseline_decomp_v1."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def trading_days_index(daily_panel: pd.DataFrame) -> pd.DatetimeIndex:
    """Distinct sorted trading dates across the universe."""
    return pd.DatetimeIndex(sorted(daily_panel["date"].unique()))


# ---- HW per-TF compute ------------------------------------------------------

def compute_hw_events(panel: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Compute HW + os_hwo_up per ticker on a long-format OHLC panel.

    Returns DF with cols [ticker, ts, close, hyperwave, signal, os_hwo_up,
    ob_hwo_down]. ts column name = ts_col.
    """
    out_frames = []
    for ticker, g in panel.groupby("ticker", observed=True):
        g = g.sort_values(ts_col).reset_index(drop=True)
        if len(g) < HW_LEN + 2 * HW_SIG:
            continue
        hw = compute_hyperwave(g[["close"]], length=HW_LEN, sig_len=HW_SIG)
        merged = pd.concat(
            [
                g[[ts_col, "close"]].reset_index(drop=True),
                hw.reset_index(drop=True),
            ],
            axis=1,
        )
        merged["ticker"] = ticker
        out_frames.append(merged)
    if not out_frames:
        return pd.DataFrame(
            columns=["ticker", ts_col, "close", "hyperwave", "signal",
                     "hwo_up", "hwo_down", "os_hwo_up", "ob_hwo_down"]
        )
    return pd.concat(out_frames, ignore_index=True)


# ---- Coverage check ---------------------------------------------------------

def coverage_check(daily: pd.DataFrame, weekly: pd.DataFrame, fivh: pd.DataFrame) -> pd.DataFrame:
    """Per-TF per-year aggregate coverage. Returns long-form coverage table.

    Coverage = observed_total_bars / (max_bars_any_ticker_had × active_tickers).
    'Active' = ticker with ≥1 bar of that TF in that year. The denominator
    uses max-observed bars per year so partial years (e.g. 2026 Jan–Apr)
    calibrate correctly without a hardcoded full-year expected count.
    """
    rows = []
    daily = daily.assign(year=daily["date"].dt.year)
    weekly = weekly.assign(year=pd.to_datetime(weekly["week_end"]).dt.year)
    fivh = fivh.assign(year=pd.to_datetime(fivh["ts_istanbul"]).dt.year)

    for tf, df in [("1d", daily), ("1w", weekly), ("5h", fivh)]:
        years = sorted(df["year"].unique())
        for yr in years:
            sub = df[df["year"] == yr]
            per_ticker = sub.groupby("ticker", observed=True).size()
            if per_ticker.empty:
                continue
            n_active = int(per_ticker.size)
            max_bars = int(per_ticker.max())
            n_bars = int(per_ticker.sum())
            denom = n_active * max_bars
            cov = n_bars / denom if denom > 0 else 0.0
            rows.append({"tf": tf, "year": int(yr), "active_tickers": n_active,
                         "bars": n_bars, "max_bars_observed": max_bars,
                         "coverage": round(cov, 4)})
    return pd.DataFrame(rows)


# ---- Trigger event tables ---------------------------------------------------

def fires_1d(events_1d: pd.DataFrame) -> pd.DataFrame:
    """T_1d: os_hwo_up_1d on day D. Returns (ticker, fire_date)."""
    e = events_1d[events_1d["os_hwo_up"]].copy()
    e = e.rename(columns={"date": "fire_date"})
    return e[["ticker", "fire_date"]].drop_duplicates().reset_index(drop=True)


def fires_1w(events_1w: pd.DataFrame) -> pd.DataFrame:
    """T_1w: os_hwo_up_1w on week W. fire_week = Friday-end label."""
    e = events_1w[events_1w["os_hwo_up"]].copy()
    e = e.rename(columns={"week_end": "fire_week"})
    return e[["ticker", "fire_week"]].drop_duplicates().reset_index(drop=True)


def fires_5h(events_5h: pd.DataFrame) -> pd.DataFrame:
    """T_5h: os_hwo_up_5h on (date, bar). Returns ticker, fire_date_5h, bar."""
    e = events_5h[events_5h["os_hwo_up"]].copy()
    ts = pd.to_datetime(e["ts_istanbul"])
    e["fire_date_5h"] = ts.dt.tz_convert("Europe/Istanbul").dt.normalize().dt.tz_localize(None)
    e["bar"] = np.where(ts.dt.tz_convert("Europe/Istanbul").dt.hour == 9, "AM", "PM")
    return e[["ticker", "fire_date_5h", "bar"]].drop_duplicates().reset_index(drop=True)


def confluence_1d_5h(f1d: pd.DataFrame, f5h: pd.DataFrame, td_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Pair T_1d on D with T_5h within trading-day window {D-2, D-1, D}.

    Returns DF: ticker, fire_date (=D), bar_5h_origin (AM|PM|both),
                 fire_date_5h (closest 5h fire date in window).
    """
    if f1d.empty or f5h.empty:
        return pd.DataFrame(columns=["ticker", "fire_date", "fire_date_5h", "bar_5h_origin"])
    pos = pd.Series(np.arange(len(td_idx)), index=td_idx)
    out_rows = []
    f5h_g = f5h.groupby("ticker", sort=False)
    for tk, g_1d in f1d.groupby("ticker", sort=False):
        if tk not in f5h_g.groups:
            continue
        g_5h = f5h_g.get_group(tk).copy()
        g_5h["pos"] = g_5h["fire_date_5h"].map(pos)
        g_5h = g_5h.dropna(subset=["pos"])
        if g_5h.empty:
            continue
        for d in g_1d["fire_date"]:
            if d not in pos.index:
                continue
            pd_ = pos[d]
            window_mask = (g_5h["pos"] >= pd_ - 2) & (g_5h["pos"] <= pd_)
            in_win = g_5h[window_mask]
            if in_win.empty:
                continue
            bars = sorted(in_win["bar"].unique().tolist())
            origin = "+".join(bars)
            closest_5h = in_win.iloc[in_win["pos"].argmax()]["fire_date_5h"]
            out_rows.append({
                "ticker": tk, "fire_date": d,
                "fire_date_5h": closest_5h, "bar_5h_origin": origin,
            })
    return pd.DataFrame(out_rows)


def confluence_1d_1w(f1d: pd.DataFrame, f1w: pd.DataFrame, td_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Pair T_1d on D with T_1w whose fire_week (Friday) is within ≤5 trading days prior of D."""
    if f1d.empty or f1w.empty:
        return pd.DataFrame(columns=["ticker", "fire_date", "fire_week"])
    pos = pd.Series(np.arange(len(td_idx)), index=td_idx)
    # Map fire_week (Friday) to nearest trading-day position (≤ that Friday)
    out_rows = []
    f1w_g = f1w.groupby("ticker", sort=False)
    for tk, g_1d in f1d.groupby("ticker", sort=False):
        if tk not in f1w_g.groups:
            continue
        g_1w = f1w_g.get_group(tk).copy()
        # Convert fire_week (W-FRI label) to td position by finding the largest td <= fire_week
        g_1w["pos"] = g_1w["fire_week"].apply(
            lambda w: int(pos[pos.index <= w].iloc[-1]) if (pos.index <= w).any() else np.nan
        )
        g_1w = g_1w.dropna(subset=["pos"])
        if g_1w.empty:
            continue
        for d in g_1d["fire_date"]:
            if d not in pos.index:
                continue
            pd_ = pos[d]
            window_mask = (g_1w["pos"] >= pd_ - 5) & (g_1w["pos"] <= pd_)
            in_win = g_1w[window_mask]
            if in_win.empty:
                continue
            closest = in_win.iloc[in_win["pos"].argmax()]["fire_week"]
            out_rows.append({"ticker": tk, "fire_date": d, "fire_week": closest})
    return pd.DataFrame(out_rows)


# ---- Trade construction ----------------------------------------------------

def build_trade_table(daily: pd.DataFrame, fires: pd.DataFrame,
                      trigger: str, fire_date_col: str,
                      core_membership: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, fire_date), compute entry/exit and rank_pct per K.

    Daily must be sorted (ticker, date). Adds columns next_open and
    fwd_close_K for each K in HORIZONS. Filters fires to those where
    ticker tier='core' on fire_date.
    """
    if fires.empty:
        return pd.DataFrame()

    # Build per-ticker forward-looking columns
    d = daily.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker", sort=False)
    d["next_open"] = g["open"].shift(-1)
    for K in HORIZONS:
        d[f"fwd_close_{K}"] = g["close"].shift(-K)

    fires = fires.rename(columns={fire_date_col: "fire_date"})
    fires["trigger"] = trigger

    # Filter by core membership on fire date
    cm = core_membership.set_index(["ticker", "date"])["is_core"]
    fires["is_core"] = fires.set_index(["ticker", "fire_date"]).index.map(cm)
    fires = fires[fires["is_core"].fillna(False)].drop(columns=["is_core"])

    if fires.empty:
        return pd.DataFrame()

    # Merge entry/exit prices
    cols_keep = ["ticker", "date", "open", "close", "next_open"] + [f"fwd_close_{K}" for K in HORIZONS]
    merged = fires.merge(
        d[cols_keep].rename(columns={"date": "fire_date"}),
        on=["ticker", "fire_date"], how="left",
    )

    # Compute returns per K
    for K in HORIZONS:
        merged[f"ret_open_{K}"] = (merged[f"fwd_close_{K}"] / merged["next_open"]) - 1.0

    return merged


def attach_rank_pct(trades: pd.DataFrame, daily: pd.DataFrame,
                    core_membership: pd.DataFrame) -> pd.DataFrame:
    """For each fire (date, K), compute rank_pct within same-date Core cohort.

    rank_pct = fraction of Core peers (entering on same fire_date and held same K)
    with strictly LOWER ret_open_K. Higher rank_pct = better.
    """
    if trades.empty:
        return trades

    d = daily.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker", sort=False)
    d["next_open"] = g["open"].shift(-1)
    for K in HORIZONS:
        d[f"fwd_close_{K}"] = g["close"].shift(-K)
        d[f"ret_open_{K}"] = (d[f"fwd_close_{K}"] / d["next_open"]) - 1.0

    cm = core_membership.set_index(["ticker", "date"])["is_core"]
    d_idx = d.set_index(["ticker", "date"])
    d_idx["is_core"] = cm
    d = d_idx.reset_index()
    d_core = d[d["is_core"].fillna(False)]

    out = trades.copy()
    for K in HORIZONS:
        col = f"ret_open_{K}"
        cohort = d_core[["date", "ticker", col]].rename(columns={"date": "fire_date"})
        # for each fire (fire_date, ticker, ret), rank vs same-date Core
        rp = []
        cohort_g = cohort.groupby("fire_date")
        for _, row in out.iterrows():
            fd = row["fire_date"]
            r = row[col]
            if pd.isna(r) or fd not in cohort_g.groups:
                rp.append(np.nan)
                continue
            peer = cohort_g.get_group(fd)[col].dropna()
            if peer.empty:
                rp.append(np.nan)
                continue
            rp.append(float((peer < r).sum() / len(peer)))
        out[f"rank_pct_{K}"] = rp
    return out


# ---- Random baseline -------------------------------------------------------

def random_baseline_pf(primary_trades: pd.DataFrame, daily: pd.DataFrame,
                        core_membership: pd.DataFrame, K: int,
                        n_draws: int = RANDOM_DRAWS, seed: int = SEED) -> dict:
    """For each fire date in primary, sample N(D) random Core tickers; compute PF per draw."""
    if primary_trades.empty:
        return {"K": K, "n_draws": n_draws, "mean_pf": float("nan"),
                "p05": float("nan"), "p95": float("nan"), "draws": []}

    rng = np.random.default_rng(seed)

    # Per-date target N
    fire_n = primary_trades.groupby("fire_date").size().to_dict()

    d = daily.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker", sort=False)
    d["next_open"] = g["open"].shift(-1)
    d[f"fwd_close_{K}"] = g["close"].shift(-K)
    d[f"ret_open_{K}"] = (d[f"fwd_close_{K}"] / d["next_open"]) - 1.0

    cm = core_membership.set_index(["ticker", "date"])["is_core"]
    d_idx = d.set_index(["ticker", "date"])
    d_idx["is_core"] = cm
    d = d_idx.reset_index()
    d_core = d[d["is_core"].fillna(False)]
    by_date = d_core.groupby("date")

    pfs = []
    for draw in range(n_draws):
        rets = []
        for fd, n in fire_n.items():
            if fd not in by_date.groups:
                continue
            pool = by_date.get_group(fd)[f"ret_open_{K}"].dropna().values
            if pool.size == 0:
                continue
            n_take = min(int(n), pool.size)
            picks = rng.choice(pool, size=n_take, replace=False)
            rets.extend(picks.tolist())
        pfs.append(pf_proxy(np.array(rets)))

    pfs_arr = np.array(pfs, dtype=float)
    return {
        "K": K, "n_draws": n_draws,
        "mean_pf": float(np.nanmean(pfs_arr)),
        "p05": float(np.nanpercentile(pfs_arr, 5)),
        "p95": float(np.nanpercentile(pfs_arr, 95)),
        "draws": pfs_arr.tolist(),
    }


# ---- Verdict ---------------------------------------------------------------

@dataclass
class Verdict:
    spec_hash: str
    seed_hex: str
    primary_trigger: str
    primary_K: int
    n_fires: int
    pf_primary: float
    rank_pct_mean: float
    ks_pvalue: float
    random_pf_mean: float
    g1_pass: bool
    g2_pass: bool
    g3_pass: bool
    g4_pass: bool
    accepted: bool
    note: str
    timestamp: str


def evaluate_gates(primary_trades: pd.DataFrame, random_pf_mean: float) -> Verdict:
    K = PRIMARY_K
    rcol = f"ret_open_{K}"
    rank_col = f"rank_pct_{K}"
    n = int(primary_trades[rcol].notna().sum())
    pf = pf_proxy(primary_trades[rcol].dropna().values) if n > 0 else float("nan")
    rank_mean = float(primary_trades[rank_col].dropna().mean()) if n > 0 else float("nan")

    if n > 0 and primary_trades[rank_col].notna().sum() > 0:
        ks_stat, ks_p = stats.kstest(primary_trades[rank_col].dropna().values, "uniform")
    else:
        ks_p = float("nan")

    g1 = bool((not math.isnan(pf)) and (pf >= G1_PF_MIN))
    g2 = bool((not math.isnan(rank_mean)) and (rank_mean >= G2_RANK_PCT_MIN))
    g3 = bool((not math.isnan(ks_p)) and (ks_p < G3_KS_PVAL_MAX))
    g4 = bool(n >= G4_N_MIN)
    accepted = bool(g1 and g2 and g3 and g4)

    fail_reasons = []
    if not g1:
        fail_reasons.append(f"G1 PF {pf:.3f} < {G1_PF_MIN}")
    if not g2:
        fail_reasons.append(f"G2 rank_pct {rank_mean:.3f} < {G2_RANK_PCT_MIN}")
    if not g3:
        fail_reasons.append(f"G3 KS p {ks_p:.4f} ≥ {G3_KS_PVAL_MAX}")
    if not g4:
        fail_reasons.append(f"G4 N {n} < {G4_N_MIN}")
    note = "all gates pass" if accepted else " | ".join(fail_reasons)

    return Verdict(
        spec_hash=SPEC_HASH, seed_hex=hex(SEED),
        primary_trigger="T_1d∧5h", primary_K=K,
        n_fires=n, pf_primary=float(pf), rank_pct_mean=rank_mean,
        ks_pvalue=float(ks_p) if not math.isnan(ks_p) else float("nan"),
        random_pf_mean=float(random_pf_mean),
        g1_pass=g1, g2_pass=g2, g3_pass=g3, g4_pass=g4,
        accepted=accepted, note=note,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---- Main ------------------------------------------------------------------

def main() -> None:
    print(f"[hwo_mtf_v1] spec_hash={SPEC_HASH[:16]}…  seed={hex(SEED)}")
    print(f"[window]    {START_DATE.date()} → {END_DATE.date()}")

    # 1. Universe (daily panel + tier classification)
    print("[1/9] building universe (q60 Core)…")
    universe = build_universe()
    universe = universe[(universe["date"] >= START_DATE) & (universe["date"] <= END_DATE)].copy()
    universe["is_core"] = universe["tier"] == "core"
    n_active = universe["ticker"].nunique()
    print(f"      tickers={n_active}  rows={len(universe):,}")

    # 2. Daily panel for HW (close only matters)
    daily = universe[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()
    core_membership = universe[["ticker", "date", "is_core"]].copy()
    td_idx = trading_days_index(daily)
    print(f"      trading days: {len(td_idx)}")

    # 3. Build 1w panel (via mb_scanner.resample.to_weekly on 1h master)
    print("[2/9] building 1w panel…")
    bars_1h = pd.read_parquet(PC.MASTER_PARQUET)
    bdate = bars_1h["ts_istanbul"].dt.tz_convert("Europe/Istanbul").dt.normalize().dt.tz_localize(None)
    bars_1h = bars_1h[(bdate >= START_DATE) & (bdate <= END_DATE)].copy()
    weekly = to_weekly(bars_1h)
    print(f"      weekly bars: {len(weekly):,}  tickers: {weekly['ticker'].nunique()}")

    # 4. Build 5h panel (n_bars==5 only)
    print("[3/9] building 5h panel (n_bars==5)…")
    fivh = to_5h(bars_1h)
    fivh = fivh[fivh["n_bars"] == 5].reset_index(drop=True)
    fivh_date = fivh["ts_istanbul"].dt.tz_convert("Europe/Istanbul").dt.normalize().dt.tz_localize(None)
    fivh = fivh[(fivh_date >= START_DATE) & (fivh_date <= END_DATE)].reset_index(drop=True)
    print(f"      5h bars: {len(fivh):,}  tickers: {fivh['ticker'].nunique()}")

    # 5. Coverage check
    print("[4/9] coverage check (≥80% per-TF per-year)…")
    cov = coverage_check(daily, weekly, fivh)
    cov.to_csv(OUT_COVERAGE, index=False)
    print(cov.to_string(index=False))
    if (cov["coverage"] < COVERAGE_FLOOR).any():
        offenders = cov[cov["coverage"] < COVERAGE_FLOOR]
        print("\n[ABORT] coverage floor breached:")
        print(offenders.to_string(index=False))
        return

    # 6. HW per TF
    print("[5/9] HW per-TF…")
    events_1d = compute_hw_events(daily, ts_col="date")
    events_1w = compute_hw_events(weekly, ts_col="week_end")
    events_5h = compute_hw_events(fivh, ts_col="ts_istanbul")
    print(f"      1d HW rows: {len(events_1d):,}  os_hwo_up: {int(events_1d['os_hwo_up'].sum())}")
    print(f"      1w HW rows: {len(events_1w):,}  os_hwo_up: {int(events_1w['os_hwo_up'].sum())}")
    print(f"      5h HW rows: {len(events_5h):,}  os_hwo_up: {int(events_5h['os_hwo_up'].sum())}")

    # 7. Trigger event tables
    print("[6/9] trigger fires…")
    f_1d = fires_1d(events_1d)
    f_1w = fires_1w(events_1w)
    f_5h = fires_5h(events_5h)
    print(f"      T_1d:  {len(f_1d):,}  T_1w: {len(f_1w):,}  T_5h: {len(f_5h):,}")

    primary = confluence_1d_5h(f_1d, f_5h, td_idx)
    secondary_1d_1w = confluence_1d_1w(f_1d, f_1w, td_idx)
    print(f"      T_1d∧5h (primary): {len(primary):,}")
    print(f"      T_1d∧1w: {len(secondary_1d_1w):,}")

    # 8. Trade tables (per-trigger)
    print("[7/9] building trades + ranks per trigger × K…")
    trade_specs = [
        ("T_1d", f_1d, "fire_date"),
        ("T_1w", f_1w.assign(fire_date=lambda x: x["fire_week"]), "fire_date"),  # entry next-open after Friday
        ("T_5h", f_5h.assign(fire_date=lambda x: x["fire_date_5h"]).drop_duplicates(["ticker", "fire_date"]), "fire_date"),
        ("T_1d∧5h", primary, "fire_date"),
        ("T_1d∧1w", secondary_1d_1w, "fire_date"),
    ]
    all_trades = []
    for trig, fires, fcol in trade_specs:
        if fires.empty:
            continue
        tt = build_trade_table(daily, fires.copy(), trig, fcol, core_membership)
        if tt.empty:
            continue
        tt = attach_rank_pct(tt, daily, core_membership)
        all_trades.append(tt)
        print(f"      {trig}: rows={len(tt)}  N(K=10)={int(tt[f'ret_open_{PRIMARY_K}'].notna().sum())}")

    if not all_trades:
        print("[ABORT] no trades produced from any trigger")
        return

    per_trade = pd.concat(all_trades, ignore_index=True)
    per_trade.to_csv(OUT_PER_TRADE, index=False)
    print(f"      wrote {OUT_PER_TRADE} ({len(per_trade)} rows)")

    # 9. Aggregate per trigger × K
    print("[8/9] aggregate metrics…")
    agg_rows = []
    for trig in ["T_1d", "T_1w", "T_5h", "T_1d∧5h", "T_1d∧1w"]:
        sub = per_trade[per_trade["trigger"] == trig]
        for K in HORIZONS:
            r = sub[f"ret_open_{K}"].dropna()
            rp = sub[f"rank_pct_{K}"].dropna()
            if len(r) == 0:
                agg_rows.append({"trigger": trig, "K": K, "N": 0, "PF": float("nan"),
                                 "rank_pct_mean": float("nan"), "ks_p": float("nan")})
                continue
            pf = pf_proxy(r.values)
            rp_mean = float(rp.mean()) if len(rp) else float("nan")
            ks_p = float(stats.kstest(rp.values, "uniform").pvalue) if len(rp) else float("nan")
            agg_rows.append({"trigger": trig, "K": K, "N": int(len(r)), "PF": pf,
                             "rank_pct_mean": rp_mean, "ks_p": ks_p})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_AGGREGATE, index=False)
    print(agg.to_string(index=False))

    # Random baseline (primary trigger only, all K reported, gate uses K=10)
    print("[random baseline] 100 draws per K…")
    rb_rows = []
    primary_trades = per_trade[per_trade["trigger"] == "T_1d∧5h"].copy()
    for K in HORIZONS:
        rb = random_baseline_pf(primary_trades, daily, core_membership, K=K)
        rb_rows.append({"K": K, "n_draws": rb["n_draws"], "mean_pf": rb["mean_pf"],
                        "p05": rb["p05"], "p95": rb["p95"]})
        # write per-draw separately
    rb_df = pd.DataFrame(rb_rows)
    rb_df.to_csv(OUT_RANDOM, index=False)
    print(rb_df.to_string(index=False))

    rb_K10 = next(r for r in rb_rows if r["K"] == PRIMARY_K)

    # 10. Verdict
    print("[9/9] gate evaluation…")
    verdict = evaluate_gates(primary_trades, rb_K10["mean_pf"])
    OUT_VERDICT.write_text(json.dumps(asdict(verdict), indent=2))
    print(f"      verdict: {'ACCEPTED' if verdict.accepted else 'REJECTED'}")
    print(f"      {verdict.note}")
    print(f"      wrote {OUT_VERDICT}")

    # 11. Manifest (cross-TF timing audit per §13)
    primary[["ticker", "fire_date", "fire_date_5h", "bar_5h_origin"]].to_json(
        OUT_MANIFEST, orient="records", date_format="iso", indent=2,
    )

    # 12. Report
    write_report(verdict, agg, cov, rb_rows, primary, len(f_1d), len(f_1w), len(f_5h))


def write_report(verdict: Verdict, agg: pd.DataFrame, cov: pd.DataFrame,
                  rb_rows, primary_pairs, n_1d: int, n_1w: int, n_5h: int) -> None:
    lines = [
        "# hwo_mtf_v1 — Single Authorized Run Report",
        "",
        f"- spec_hash: `{SPEC_HASH}`",
        f"- seed: `{hex(SEED)}` (\"HWO MTF01\")",
        f"- window: {START_DATE.date()} → {END_DATE.date()}",
        f"- HW params: length={HW_LEN}, sig_len={HW_SIG} (close source)",
        f"- horizons: K ∈ {list(HORIZONS)}, primary K={PRIMARY_K}",
        f"- random baseline: {RANDOM_DRAWS} draws, same-date same-N from q60 Core",
        f"- timestamp: {verdict.timestamp}",
        "",
        "## VERDICT",
        f"**{'CLOSED_ACCEPTED' if verdict.accepted else 'CLOSED_REJECTED'}** — {verdict.note}",
        "",
        "| Gate | Metric | Threshold | Value | Pass |",
        "|---|---|---|---|---|",
        f"| G1 | PF (T_1d∧5h, K={PRIMARY_K}) | ≥ {G1_PF_MIN} | {verdict.pf_primary:.3f} | {'✓' if verdict.g1_pass else '✗'} |",
        f"| G2 | rank_pct mean | ≥ {G2_RANK_PCT_MIN} | {verdict.rank_pct_mean:.3f} | {'✓' if verdict.g2_pass else '✗'} |",
        f"| G3 | KS p-value vs uniform | < {G3_KS_PVAL_MAX} | {verdict.ks_pvalue:.4f} | {'✓' if verdict.g3_pass else '✗'} |",
        f"| G4 | N fires | ≥ {G4_N_MIN} | {verdict.n_fires} | {'✓' if verdict.g4_pass else '✗'} |",
        "",
        f"Random baseline mean PF (K={PRIMARY_K}): **{verdict.random_pf_mean:.3f}** "
        f"(primary {verdict.pf_primary:.3f} → lift {verdict.pf_primary - verdict.random_pf_mean:+.3f})",
        "",
        "## Coverage check (per-TF per-year, floor 80%)",
        cov.to_markdown(index=False),
        "",
        "## Trigger fire counts",
        f"- T_1d (single daily HWO Up + HW<20): {n_1d}",
        f"- T_1w (single weekly):                {n_1w}",
        f"- T_5h (single 5h):                    {n_5h}",
        f"- T_1d∧5h (PRIMARY, ≤3 trading-day):  {len(primary_pairs)}",
        "",
        "## Aggregate metrics per trigger × K",
        agg.to_markdown(index=False),
        "",
        "## Random baseline summary (primary trigger, per K)",
        pd.DataFrame(rb_rows).to_markdown(index=False),
        "",
        "## Closure",
        ("This thread is now CLOSED_ACCEPTED. Production / paper-trade is a SEPARATE "
         "pre-registered thread per SPEC §12." if verdict.accepted else
         "This thread is now CLOSED_REJECTED. No rescue allowed; no parameter sweep, "
         "no horizon swap, no window carve-out (SPEC §12). A wholly different "
         "hypothesis is permitted in a new pre-reg thread."),
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"      wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
