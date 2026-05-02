#!/usr/bin/env python3
"""baseline_decomp v1 — single diagnostic run.

Spec: baseline_decomp/SPEC.md (frozen pre-registration). All thresholds
below are locked at spec time. Do not tune.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prsr import universe as U  # noqa: E402

# === Locked config (spec) ===========================================
SPEC_VERSION = "1.0.0"
K_HORIZON = 10
WINDOW_START = pd.Timestamp("2023-01-02")
WINDOW_END = pd.Timestamp("2026-04-29")

RANDOM_DISCLOSURE = {
    "AL_F6": {
        "seed": 17,
        "sampling": (
            "Same-date 1-per-fire from q60-Core panel excluding actual fires; "
            "n_fires(date) draws per date; producer "
            "tools/oscmatrix_phase4a_core_whitelist_backtest.py."
        ),
    },
    "PRSR": {
        "seed": 17,
        "sampling": (
            "Same-date same-N from Tradeable Core excluding PRSR fires; "
            "min(n_fires(date), |eligible(date)|) draws per date; "
            "producer prsr/random_baseline.py."
        ),
    },
}

REBOUND_RET_THR = 0.01
REBOUND_DD_THR = -0.02
REBOUND_DD_LOOKBACK = 5

REBOUND_FRACTION_VERDICT = 0.60
RANK_PCT_NEG_MAX = 0.50
RANK_PCT_FLAT_LOW, RANK_PCT_FLAT_HIGH = 0.48, 0.52
RANK_PCT_POS_MIN = 0.55
KS_P_THRESHOLD = 0.05
TIMING_DOMINATED_RATIO = 2.0
SELECTION_LEG_PF_NULL_MAX = 1.10
SELECTION_LEG_PF_SUBSET_MIN = 1.30
CLUSTER_LIFT_MIN = 1.5
CLUSTER_CHI2_P_MAX = 0.05

# === IO =============================================================
OUT_DIR = ROOT / "output"
EW_CACHE = OUT_DIR / "baseline_decomp_ew_core_daily.parquet"
OUT_REPORT = OUT_DIR / "baseline_decomp_v1_report.md"
OUT_DATE = OUT_DIR / "baseline_decomp_v1_date_attribution.csv"
OUT_RANK = OUT_DIR / "baseline_decomp_v1_rank_pct.csv"
OUT_DECOMP = OUT_DIR / "baseline_decomp_v1_decomposition.csv"
OUT_EWS = OUT_DIR / "baseline_decomp_v1_ew_core_strategy.csv"

ALF6_TRADES_CSV = OUT_DIR / "oscmatrix_phase4a_per_trade.csv"
ALF6_RANDOM_CSV = OUT_DIR / "oscmatrix_phase4a_random_baseline.csv"
PRSR_TRADES_CSV = OUT_DIR / "prsr_v1_per_trade.csv"
XU100_PARQUET = OUT_DIR / "xu100_extfeed_daily.parquet"


# === Helpers ========================================================
def pf_proxy(rets: np.ndarray) -> float:
    rets = np.asarray(rets, dtype=float)
    rets = rets[~np.isnan(rets)]
    pos = rets[rets > 0].sum()
    neg = -rets[rets < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def gini_abs(values: np.ndarray) -> float:
    v = np.abs(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if v.sum() == 0 or len(v) == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def cumshare_at(values: np.ndarray, frac: float) -> float:
    """Top-frac share of |values| (signed values keep sign in PnL sum)."""
    v = np.asarray(values, dtype=float)
    v_sorted = np.sort(v)[::-1]
    k = max(1, int(round(len(v_sorted) * frac)))
    total = v_sorted.sum()
    if total == 0:
        return float("nan")
    return float(v_sorted[:k].sum() / total)


# === Step 1: Universe + EW Core daily series ========================
def build_panel_with_forward() -> pd.DataFrame:
    """Daily panel + tier + forward returns for K=1..K_HORIZON."""
    print("[1/7] building daily panel and q60 Core tier …", flush=True)
    panel = U.build_universe()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    panel["ret_d"] = g["close"].pct_change()
    panel["open_t1"] = g["open"].shift(-1)
    for x in range(1, K_HORIZON + 1):
        panel[f"close_t{x}"] = g["close"].shift(-x)
        panel[f"ret_open_{x}"] = panel[f"close_t{x}"] / panel["open_t1"] - 1.0
    return panel


def ew_core_daily(panel: pd.DataFrame) -> pd.DataFrame:
    print("[2/7] computing EW Core daily return + rebound flag …", flush=True)
    core_daily = (
        panel[panel["tier"] == "core"]
        .groupby("date", as_index=False)["ret_d"]
        .mean()
        .rename(columns={"ret_d": "ew_core_ret_d"})
    )
    core_daily = core_daily.sort_values("date").reset_index(drop=True)
    core_daily["ew_core_idx"] = (1.0 + core_daily["ew_core_ret_d"].fillna(0.0)).cumprod()
    prior_max = (
        core_daily["ew_core_idx"]
        .shift(1)
        .rolling(REBOUND_DD_LOOKBACK)
        .max()
    )
    core_daily["prior_dd_5d"] = core_daily["ew_core_idx"] / prior_max - 1.0
    core_daily["rebound_day"] = (
        (core_daily["ew_core_ret_d"] >= REBOUND_RET_THR)
        & (core_daily["prior_dd_5d"] <= REBOUND_DD_THR)
    ).fillna(False)
    core_daily.to_parquet(EW_CACHE)
    return core_daily


def xu100_aux(core_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Load XU100 daily; degrade to auxiliary if missing/misaligned."""
    print("[3/7] loading XU100 (auxiliary) …", flush=True)
    info: dict = {"available": False, "fallback_reason": None, "missing_days": None}
    if not XU100_PARQUET.exists():
        info["fallback_reason"] = f"{XU100_PARQUET.name} not present"
        return pd.DataFrame(), info

    raw = pd.read_parquet(XU100_PARQUET)
    if not isinstance(raw.index, pd.RangeIndex) and raw.index.name in (
        "Date", "date", "ts_istanbul", "datetime", "timestamp",
    ):
        raw = raw.reset_index()
    date_col = next(
        (
            c for c in ("date", "Date", "ts_istanbul", "datetime", "timestamp")
            if c in raw.columns
        ),
        None,
    )
    if date_col is None:
        info["fallback_reason"] = "no recognisable date column in XU100 parquet"
        return pd.DataFrame(), info
    close_col = next(
        (c for c in ("close", "Close", "xu100_close", "value") if c in raw.columns),
        None,
    )
    if close_col is None:
        info["fallback_reason"] = "no close column in XU100 parquet"
        return pd.DataFrame(), info

    raw = raw[[date_col, close_col]].copy()
    raw[date_col] = pd.to_datetime(raw[date_col])
    if raw[date_col].dt.tz is not None:
        raw[date_col] = raw[date_col].dt.tz_localize(None)
    raw["date"] = raw[date_col].dt.normalize()
    raw = raw.rename(columns={close_col: "xu100_close"}).drop(columns=[date_col])
    raw = raw[(raw["date"] >= WINDOW_START) & (raw["date"] <= WINDOW_END)]
    raw = raw.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    raw["xu100_ret_d"] = raw["xu100_close"].pct_change()

    core_dates = set(core_daily["date"])
    xu_dates = set(raw["date"])
    missing = sorted(core_dates - xu_dates)
    info["missing_days"] = len(missing)
    if len(missing) > 5:
        info["fallback_reason"] = (
            f"{len(missing)} EW-Core trading days missing in XU100 (>5 threshold)"
        )
        return raw, info
    info["available"] = True
    return raw, info


# === Step 2: Load signal + random pools =============================
def load_alf6_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[4/7] loading AL_F6 (open_T1, K=10) signal + random …", flush=True)
    sig = pd.read_csv(ALF6_TRADES_CSV)
    sig = sig[(sig["entry_mode"] == "open_T1") & (sig["K"] == K_HORIZON)].copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    sig = sig.rename(columns={"realized_pct": "ret_actual"})
    sig["pool"] = "AL_F6_signal"

    rnd = pd.read_csv(ALF6_RANDOM_CSV)
    rnd = rnd[(rnd["entry_mode"] == "open_T1") & (rnd["K"] == K_HORIZON)].copy()
    rnd["date"] = pd.to_datetime(rnd["date"]).dt.normalize()
    rnd = rnd.rename(columns={"realized_pct": "ret_actual"})
    rnd["pool"] = "AL_F6_random"

    keep = ["pool", "ticker", "date", "ret_actual"]
    return sig[keep].reset_index(drop=True), rnd[keep].reset_index(drop=True)


def load_prsr_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[5/7] loading PRSR (open_T1, primary) signal + random …", flush=True)
    df = pd.read_csv(PRSR_TRADES_CSV)
    df = df[df["entry_mode"] == "open_T1"].copy()
    df["date"] = pd.to_datetime(df["fire_date"]).dt.normalize()
    df = df.rename(
        columns={"ret_primary": "ret_actual", "exit_primary_days": "hold_days"}
    )
    sig = df[df["source"] == "prsr"][["ticker", "date", "ret_actual", "hold_days"]].copy()
    rnd = df[df["source"] == "random"][["ticker", "date", "ret_actual", "hold_days"]].copy()
    sig["pool"] = "PRSR_signal"
    rnd["pool"] = "PRSR_random"
    keep = ["pool", "ticker", "date", "ret_actual", "hold_days"]
    return sig[keep].reset_index(drop=True), rnd[keep].reset_index(drop=True)


# === Step 3: Forward-K joiner =======================================
def attach_forward_open_K(pools: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add ret_open_K (K=10) per-trade and same-date EW Core ret_open_K mean."""
    fwd_lookup = panel[["ticker", "date", f"ret_open_{K_HORIZON}"]].rename(
        columns={f"ret_open_{K_HORIZON}": "ret_open_K_signal"}
    )
    out = pools.merge(fwd_lookup, on=["ticker", "date"], how="left")

    core_only = panel[panel["tier"] == "core"]
    cohort = core_only.groupby("date", as_index=False).agg(
        ret_ewcore_K=(f"ret_open_{K_HORIZON}", "mean"),
        core_count=(f"ret_open_{K_HORIZON}", "count"),
    )
    out = out.merge(cohort, on="date", how="left")
    return out


# === M1: Date attribution ===========================================
def date_attribution(pool: pd.DataFrame, label: str) -> dict:
    g = pool.groupby("date")["ret_actual"].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"sum": f"pnl_{label}", "count": f"n_{label}"})
    return {"per_date": g, "total_pnl": g[f"pnl_{label}"].sum()}


def lorenz_summary(pnl_series: np.ndarray) -> dict:
    n = len(pnl_series)
    if n == 0:
        return {"n_dates": 0}
    sorted_pnl = np.sort(pnl_series)[::-1]
    total = sorted_pnl.sum()

    def topk_share(k: int) -> float:
        if total == 0:
            return float("nan")
        return float(sorted_pnl[: min(k, n)].sum() / total)

    def topfrac_share(frac: float) -> float:
        return cumshare_at(pnl_series, frac)

    return {
        "n_dates": int(n),
        "total_pnl": float(total),
        "top5_share": topk_share(5),
        "top10_share": topk_share(10),
        "top20_share": topk_share(20),
        "top_decile_share": topfrac_share(0.10),
        "top_quintile_share": topfrac_share(0.20),
        "gini_abs": gini_abs(pnl_series),
    }


# === M3: Cluster ====================================================
def cluster_lift(
    pool: pd.DataFrame, all_dates: pd.DataFrame, label: str
) -> dict:
    """Lift of pool fires on rebound days vs base rate."""
    fires_per_date = pool.groupby("date").size().rename(f"fires_{label}")
    merged = all_dates.merge(fires_per_date, on="date", how="left").fillna(
        {f"fires_{label}": 0}
    )
    merged[f"fired_{label}"] = merged[f"fires_{label}"] > 0
    R = int(merged["rebound_day"].sum())
    NR = int((~merged["rebound_day"]).sum())
    fired_total = int(merged[f"fired_{label}"].sum())
    fired_on_rebound = int(
        (merged[f"fired_{label}"] & merged["rebound_day"]).sum()
    )
    base_rate = fired_total / (R + NR) if (R + NR) > 0 else float("nan")
    rebound_rate = fired_on_rebound / R if R > 0 else float("nan")
    lift = (
        rebound_rate / base_rate
        if base_rate and not np.isnan(base_rate) and base_rate > 0
        else float("nan")
    )

    a = fired_on_rebound
    b = fired_total - fired_on_rebound
    c = R - fired_on_rebound
    d = NR - b
    table = np.array([[a, b], [c, d]])
    chi2, p, _, _ = sst.chi2_contingency(table) if (table.min() >= 0 and table.sum() > 0) else (np.nan, np.nan, None, None)
    return {
        "label": label,
        "R": R,
        "NR": NR,
        "fired_total": fired_total,
        "fired_on_rebound": fired_on_rebound,
        "base_rate": base_rate,
        "rebound_rate": rebound_rate,
        "lift": lift,
        "chi2": float(chi2) if chi2 is not None else float("nan"),
        "p_value": float(p) if p is not None else float("nan"),
    }


# === M4: Cross-sectional rank =======================================
def per_trade_rank_pct(pool: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Compute percentile rank of selected ticker's ret_open_K within
    the same-date core(d) ret_open_K distribution.
    """
    core_only = panel[panel["tier"] == "core"][
        ["ticker", "date", f"ret_open_{K_HORIZON}"]
    ].dropna(subset=[f"ret_open_{K_HORIZON}"])
    by_date = {
        d: g[f"ret_open_{K_HORIZON}"].values
        for d, g in core_only.groupby("date")
    }

    pool = pool.copy()
    pool["ret_open_K_signal"] = pool["ret_open_K_signal"].astype(float)
    rank_pct = np.full(len(pool), np.nan)
    cohort_n = np.zeros(len(pool), dtype=int)
    for i, row in enumerate(pool.itertuples(index=False)):
        d = row.date
        r = row.ret_open_K_signal
        if pd.isna(r):
            continue
        v = by_date.get(d)
        if v is None or len(v) == 0:
            continue
        rank_pct[i] = float((v <= r).sum() / len(v))
        cohort_n[i] = int(len(v))
    pool["rank_pct"] = rank_pct
    pool["cohort_n"] = cohort_n
    return pool


def rank_summary(pool_with_rank: pd.DataFrame, label: str) -> dict:
    s = pool_with_rank["rank_pct"].dropna()
    if len(s) == 0:
        return {"label": label, "n": 0}
    ks_stat, ks_p = sst.kstest(s.values, "uniform")
    return {
        "label": label,
        "n": int(len(s)),
        "mean_rank_pct": float(s.mean()),
        "median_rank_pct": float(s.median()),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


# === M5: Mechanical EW rebound strategy =============================
def mechanical_ew_rebound(panel: pd.DataFrame, core_daily: pd.DataFrame) -> pd.DataFrame:
    """For each rebound trigger date d: equal-weight Core, hold K bars."""
    triggers = core_daily[core_daily["rebound_day"]].copy()
    core_only = panel[panel["tier"] == "core"][
        ["ticker", "date", f"ret_open_{K_HORIZON}"]
    ]
    cohort_mean = (
        core_only.groupby("date", as_index=False)
        .agg(ew_core_K_ret=(f"ret_open_{K_HORIZON}", "mean"))
    )
    triggers = triggers.merge(cohort_mean, on="date", how="left")
    triggers = triggers.dropna(subset=["ew_core_K_ret"]).reset_index(drop=True)
    return triggers[
        [
            "date",
            "ew_core_ret_d",
            "prior_dd_5d",
            "ew_core_K_ret",
        ]
    ]


# === M6: Additive decomposition =====================================
def decompose(pool_attached: pd.DataFrame) -> pd.DataFrame:
    out = pool_attached.copy()
    out["market_leg"] = out["ret_ewcore_K"]
    out["selection_residual"] = out["ret_actual"] - out["market_leg"]
    return out


# === Verdict logic ==================================================
def rank_verdict(mean_rp: float, ks_p: float) -> str:
    if np.isnan(mean_rp):
        return "no_data"
    if mean_rp >= RANK_PCT_POS_MIN and ks_p < KS_P_THRESHOLD:
        return "selection-positive"
    if mean_rp < RANK_PCT_NEG_MAX and ks_p < KS_P_THRESHOLD:
        return "selection-negative"
    if RANK_PCT_FLAT_LOW <= mean_rp <= RANK_PCT_FLAT_HIGH and ks_p >= KS_P_THRESHOLD:
        return "selection-flat"
    return "selection-mixed"


def cluster_verdict(lift: float, p: float) -> str:
    if np.isnan(lift) or np.isnan(p):
        return "no_data"
    if lift >= CLUSTER_LIFT_MIN and p < CLUSTER_CHI2_P_MAX:
        return "cluster-positive"
    return "cluster-negative"


def timing_verdict(
    decomp_alf6: pd.DataFrame, decomp_prsr: pd.DataFrame
) -> tuple[str, dict]:
    """Q6: timing-dominated / selection-positive on subset / null."""
    info = {}
    for label, df in [("AL_F6", decomp_alf6), ("PRSR", decomp_prsr)]:
        m_sum = float(df["market_leg"].sum())
        s_sum = float(df["selection_residual"].sum())
        s_pf = pf_proxy(df["selection_residual"].values)
        info[label] = {
            "market_sum": m_sum,
            "selection_sum": s_sum,
            "selection_pf": s_pf,
            "abs_ratio_market_over_selection": (
                abs(m_sum) / abs(s_sum) if s_sum != 0 else float("inf")
            ),
        }
    timing_dominated = (
        info["AL_F6"]["abs_ratio_market_over_selection"] >= TIMING_DOMINATED_RATIO
        and info["PRSR"]["abs_ratio_market_over_selection"] >= TIMING_DOMINATED_RATIO
        and info["AL_F6"]["selection_pf"] < SELECTION_LEG_PF_NULL_MAX
        and info["PRSR"]["selection_pf"] < SELECTION_LEG_PF_NULL_MAX
    )
    if timing_dominated:
        return "timing-dominated", info
    if (
        info["AL_F6"]["selection_pf"] >= SELECTION_LEG_PF_SUBSET_MIN
        or info["PRSR"]["selection_pf"] >= SELECTION_LEG_PF_SUBSET_MIN
    ):
        return "selection-positive-on-subset", info
    return "null", info


# === Decision tree ==================================================
def decision_tree(
    rebound_top20_frac_alf6: float,
    rebound_top20_frac_prsr: float,
    rank_v_alf6: str,
    rank_v_prsr: str,
    cluster_v_alf6: str,
    cluster_v_prsr: str,
    timing_v: str,
) -> tuple[str, str]:
    rebound_avg = np.nanmean([rebound_top20_frac_alf6, rebound_top20_frac_prsr])

    def both(v_a, v_b, target):
        return v_a == target and v_b == target

    if rebound_avg >= REBOUND_FRACTION_VERDICT and timing_v == "timing-dominated":
        return (
            "market-regime/timing model thread",
            "Q2 rebound fraction ≥ 0.60 AND Q6 timing-dominated, both signals.",
        )
    if rank_v_alf6 == "selection-positive" or rank_v_prsr == "selection-positive":
        if both(cluster_v_alf6, cluster_v_prsr, "cluster-positive"):
            return (
                "date/regime conditional filter thread",
                "Selection-positive on a subset AND signals cluster on rebound days.",
            )
    if both(rank_v_alf6, rank_v_prsr, "selection-flat") and timing_v == "null":
        return (
            "early-reversal scanner family CLOSED",
            "No bucket distinguishes signal from random; partial-fit default.",
        )
    return (
        "early-reversal scanner family CLOSED",
        "Partial-fit default: signature does not match a non-CLOSED row exactly.",
    )


# === Cross-signal conflict tagging ==================================
def cross_signal(label_a: str, label_b: str) -> str:
    if label_a == label_b:
        return f"finding ({label_a})"
    return f"signal-specific (AL_F6={label_a}, PRSR={label_b})"


# === Report =========================================================
def render_report(ctx: dict) -> str:
    md: list[str] = []
    md.append("# Core q60 Random Rebound Baseline — Decomposition v1\n")
    md.append(
        f"Spec: `baseline_decomp/SPEC.md` v{SPEC_VERSION}. "
        f"Window {WINDOW_START.date()} → {WINDOW_END.date()}. "
        f"K_HORIZON={K_HORIZON} bars, entry=open_T1.\n"
    )

    md.append("## Random-baseline disclosure\n")
    md.append("| pool | seed | sampling |")
    md.append("|---|---|---|")
    for k, v in RANDOM_DISCLOSURE.items():
        md.append(f"| {k} | {v['seed']} | {v['sampling']} |")
    md.append("")

    md.append("## Pool sizes (after open_T1, K=10 / primary filter)\n")
    md.append("| pool | n_trades | unique_dates | unique_tickers | total_pnl |")
    md.append("|---|---|---|---|---|")
    for label, df in ctx["pools"].items():
        md.append(
            f"| {label} | {len(df)} | {df['date'].nunique()} | "
            f"{df['ticker'].nunique()} | {df['ret_actual'].sum():+.4f} |"
        )
    md.append("")

    md.append("## Q1. Date attribution\n")
    md.append(
        "Per-date PnL concentration. `top5/10/20_share` = sum of top-K date "
        "PnLs / total PnL. `gini_abs` = Gini on |per-date PnL|.\n"
    )
    md.append(
        "| pool | n_dates | total_pnl | top5 | top10 | top20 | top_decile | top_quintile | gini |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    for label, s in ctx["lorenz"].items():
        md.append(
            f"| {label} | {s['n_dates']} | {s['total_pnl']:+.4f} | "
            f"{s['top5_share']:.3f} | {s['top10_share']:.3f} | "
            f"{s['top20_share']:.3f} | {s['top_decile_share']:.3f} | "
            f"{s['top_quintile_share']:.3f} | {s['gini_abs']:.3f} |"
        )
    md.append("")

    xu = ctx["xu100_info"]
    md.append("## Q2. Rebound-day classification\n")
    if not xu["available"]:
        md.append(
            f"**XU100 graceful degradation triggered.** Reason: {xu['fallback_reason']}. "
            "Primary verdict driven by EW Core only. XU100 omitted from tables below.\n"
        )
    else:
        md.append(
            f"XU100 coverage check: missing trading days inside window = {xu['missing_days']}. "
            "XU100 retained as auxiliary alongside EW Core.\n"
        )
    md.append(
        f"Rebound def (locked): `EW Core ret_d ≥ {REBOUND_RET_THR:+.2%}` AND "
        f"`EW Core prior_dd_5d ≤ {REBOUND_DD_THR:+.2%}`.\n"
    )
    cd = ctx["core_daily"]
    R = int(cd["rebound_day"].sum())
    total = int(len(cd))
    md.append(
        f"Base rate: {R}/{total} dates = {R/total:.3f} rebound days in window.\n"
    )
    md.append("Top-20 random PnL date rebound classification:\n")
    md.append(
        "| pool | rebound_share_top20 | median_ew_core_ret_top20 | median_xu100_ret_top20 |"
    )
    md.append("|---|---|---|---|")
    for label, s in ctx["top20_rebound"].items():
        md.append(
            f"| {label} | {s['rebound_share']:.3f} | "
            f"{s['median_ew_core_ret']:+.4f} | "
            f"{s['median_xu100_ret']:+.4f} |"
            if not np.isnan(s.get("median_xu100_ret", np.nan))
            else f"| {label} | {s['rebound_share']:.3f} | "
                 f"{s['median_ew_core_ret']:+.4f} | n/a |"
        )
    md.append("")

    md.append("## Q3. Signal clustering on rebound days\n")
    md.append(
        "| pool | R | NR | fired_dates | fired_on_rebound | base_rate | "
        "rebound_rate | lift | chi2_p | verdict |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, c in ctx["clusters"].items():
        v = cluster_verdict(c["lift"], c["p_value"])
        md.append(
            f"| {label} | {c['R']} | {c['NR']} | {c['fired_total']} | "
            f"{c['fired_on_rebound']} | {c['base_rate']:.3f} | "
            f"{c['rebound_rate']:.3f} | {c['lift']:.2f} | "
            f"{c['p_value']:.4f} | {v} |"
        )
    md.append("")
    md.append(
        f"Cross-signal cluster verdict: **{ctx['cross_cluster']}**.\n"
    )

    md.append("## Q4. Cross-sectional rank percentile\n")
    md.append(
        "rank_pct ∈ [0,1], 1.0 = best in same-date Core cohort over K=10. "
        "Random pools should be ≈ uniform (sanity check).\n"
    )
    md.append("| pool | n | mean_rank | median_rank | KS_stat | KS_p | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for label, s in ctx["rank_summaries"].items():
        if s["n"] == 0:
            md.append(f"| {label} | 0 | n/a | n/a | n/a | n/a | no_data |")
            continue
        v = rank_verdict(s["mean_rank_pct"], s["ks_p"])
        md.append(
            f"| {label} | {s['n']} | {s['mean_rank_pct']:.4f} | "
            f"{s['median_rank_pct']:.4f} | {s['ks_stat']:.4f} | "
            f"{s['ks_p']:.4f} | {v} |"
        )
    md.append("")
    md.append(
        f"Cross-signal selection verdict: **{ctx['cross_rank']}**.\n"
    )

    md.append("## Q5. Mechanical EW Core rebound strategy\n")
    ew_s = ctx["ew_strategy_summary"]
    md.append(
        f"Trigger: EW Core ret_d ≥ {REBOUND_RET_THR:+.2%} AND prior_dd_5d ≤ "
        f"{REBOUND_DD_THR:+.2%}. Hold K=10 bars on equal-weight Core.\n"
    )
    md.append("| metric | mechanical_EW_rebound | AL_F6_signal | PRSR_signal |")
    md.append("|---|---|---|---|")
    for k in ("n", "pf", "win_rate", "mean", "median", "max_dd"):
        md.append(
            f"| {k} | {ew_s['mechanical'].get(k, float('nan')):.4f} | "
            f"{ew_s['AL_F6'].get(k, float('nan')):.4f} | "
            f"{ew_s['PRSR'].get(k, float('nan')):.4f} |"
        )
    md.append("")

    md.append("## Q6. Additive decomposition (selection vs market leg)\n")
    md.append(
        f"`ret_actual = market_leg + selection_residual`, "
        f"market_leg = same-date EW Core ret over K=10.\n"
    )
    md.append("| pool | n | market_sum | selection_sum | selection_pf | |market|/|sel| |")
    md.append("|---|---|---|---|---|---|")
    for label, info in ctx["timing_info"].items():
        md.append(
            f"| {label} | {ctx['decomp_n'][label]} | "
            f"{info['market_sum']:+.4f} | {info['selection_sum']:+.4f} | "
            f"{info['selection_pf']:.4f} | "
            f"{info['abs_ratio_market_over_selection']:.2f} |"
        )
    md.append("")
    md.append(f"Q6 verdict: **{ctx['timing_verdict']}**.\n")

    md.append("## Decision tree outcome\n")
    md.append(f"- next_thread: **{ctx['next_thread']}**")
    md.append(f"- reason: {ctx['next_thread_reason']}")
    md.append("")
    md.append("Decision-tree inputs (for audit):")
    md.append(f"- rebound_top20_frac AL_F6: {ctx['top20_rebound']['AL_F6_random']['rebound_share']:.3f}")
    md.append(f"- rebound_top20_frac PRSR: {ctx['top20_rebound']['PRSR_random']['rebound_share']:.3f}")
    md.append(f"- rank verdict AL_F6: {ctx['rank_verdicts']['AL_F6_signal']}")
    md.append(f"- rank verdict PRSR: {ctx['rank_verdicts']['PRSR_signal']}")
    md.append(f"- cluster verdict AL_F6: {ctx['cluster_verdicts']['AL_F6_signal']}")
    md.append(f"- cluster verdict PRSR: {ctx['cluster_verdicts']['PRSR_signal']}")
    md.append(f"- timing verdict: {ctx['timing_verdict']}")
    md.append("")

    md.append("## Cross-signal conflict tags\n")
    md.append(f"- rank: {ctx['cross_rank']}")
    md.append(f"- cluster: {ctx['cross_cluster']}")
    md.append("")

    md.append("## Artifacts\n")
    md.append(f"- `{OUT_DATE.relative_to(ROOT)}` — per-date PnL + rebound + fire counts")
    md.append(f"- `{OUT_RANK.relative_to(ROOT)}` — per-trade rank_pct (all 4 pools)")
    md.append(f"- `{OUT_DECOMP.relative_to(ROOT)}` — per-trade decomposition")
    md.append(f"- `{OUT_EWS.relative_to(ROOT)}` — mechanical EW rebound trades")
    md.append(f"- `{EW_CACHE.relative_to(ROOT)}` — EW Core daily series cache")
    md.append("")

    return "\n".join(md)


# === Main ===========================================================
def main() -> None:
    panel = build_panel_with_forward()
    core_daily = ew_core_daily(panel)
    xu100, xu_info = xu100_aux(core_daily)

    alf6_sig, alf6_rnd = load_alf6_pools()
    prsr_sig, prsr_rnd = load_prsr_pools()
    pools = {
        "AL_F6_signal": alf6_sig,
        "AL_F6_random": alf6_rnd,
        "PRSR_signal": prsr_sig,
        "PRSR_random": prsr_rnd,
    }

    print("[6/7] computing M1–M6 …", flush=True)
    pools_attached = {
        label: attach_forward_open_K(p, panel) for label, p in pools.items()
    }

    # M1
    lorenz = {
        label: lorenz_summary(
            df.groupby("date")["ret_actual"].sum().values
        )
        for label, df in pools.items()
    }

    # M2 (top-20 random rebound classification)
    cd_keys = core_daily[["date", "ew_core_ret_d", "prior_dd_5d", "rebound_day"]]
    if xu_info["available"]:
        cd_keys = cd_keys.merge(
            xu100[["date", "xu100_ret_d"]], on="date", how="left"
        )
    else:
        cd_keys = cd_keys.assign(xu100_ret_d=np.nan)

    top20_rebound = {}
    for label in ("AL_F6_random", "PRSR_random", "AL_F6_signal", "PRSR_signal"):
        per_date = (
            pools[label]
            .groupby("date")["ret_actual"]
            .sum()
            .reset_index(name="pnl")
        )
        top20 = per_date.sort_values("pnl", ascending=False).head(20)
        top20_join = top20.merge(cd_keys, on="date", how="left")
        rb_share = float(top20_join["rebound_day"].fillna(False).mean()) if len(top20_join) else float("nan")
        top20_rebound[label] = {
            "rebound_share": rb_share,
            "median_ew_core_ret": float(top20_join["ew_core_ret_d"].median()),
            "median_xu100_ret": float(top20_join["xu100_ret_d"].median()) if xu_info["available"] else float("nan"),
            "n_top": int(len(top20_join)),
        }

    # Per-date attribution CSV
    date_attr = cd_keys.copy()
    for label, p in pools.items():
        per = p.groupby("date")["ret_actual"].agg(["sum", "count"])
        per = per.rename(columns={"sum": f"pnl_{label}", "count": f"n_{label}"})
        date_attr = date_attr.merge(per, on="date", how="left")
    for c in date_attr.columns:
        if c.startswith("pnl_") or c.startswith("n_"):
            date_attr[c] = date_attr[c].fillna(0)
    date_attr.to_csv(OUT_DATE, index=False)

    # M3: cluster
    all_dates_df = cd_keys[["date", "rebound_day"]].copy()
    clusters = {
        label: cluster_lift(pools[label], all_dates_df, label)
        for label in ("AL_F6_signal", "AL_F6_random", "PRSR_signal", "PRSR_random")
    }

    # M4: rank percentile
    pools_ranked = {
        label: per_trade_rank_pct(p, panel) for label, p in pools_attached.items()
    }
    rank_summaries = {
        label: rank_summary(p, label) for label, p in pools_ranked.items()
    }

    # Combined rank CSV
    rank_combined = pd.concat(
        [
            p.assign(pool=label)[
                ["pool", "ticker", "date", "ret_open_K_signal",
                 "ret_ewcore_K", "rank_pct", "cohort_n"]
            ]
            for label, p in pools_ranked.items()
        ],
        ignore_index=True,
    )
    rank_combined.to_csv(OUT_RANK, index=False)

    # M5: mechanical EW rebound + side-by-side summary
    ew_strategy = mechanical_ew_rebound(panel, core_daily)
    ew_strategy.to_csv(OUT_EWS, index=False)

    def summarise_returns(rets: np.ndarray) -> dict:
        rets = np.asarray(rets, dtype=float)
        rets = rets[~np.isnan(rets)]
        if len(rets) == 0:
            return {"n": 0, "pf": float("nan"), "win_rate": float("nan"),
                    "mean": float("nan"), "median": float("nan"), "max_dd": float("nan")}
        cum = np.cumsum(rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak)
        return {
            "n": len(rets),
            "pf": pf_proxy(rets),
            "win_rate": float((rets > 0).mean()),
            "mean": float(rets.mean()),
            "median": float(np.median(rets)),
            "max_dd": float(dd.min()),
        }

    ew_strategy_summary = {
        "mechanical": summarise_returns(ew_strategy["ew_core_K_ret"].values),
        "AL_F6": summarise_returns(alf6_sig["ret_actual"].values),
        "PRSR": summarise_returns(prsr_sig["ret_actual"].values),
    }

    # M6: decompose (signal pools only — random decomposition is sanity)
    decomp_alf6 = decompose(pools_attached["AL_F6_signal"]).dropna(
        subset=["ret_actual", "market_leg"]
    )
    decomp_prsr = decompose(pools_attached["PRSR_signal"]).dropna(
        subset=["ret_actual", "market_leg"]
    )
    decomp_alf6_rnd = decompose(pools_attached["AL_F6_random"]).dropna(
        subset=["ret_actual", "market_leg"]
    )
    decomp_prsr_rnd = decompose(pools_attached["PRSR_random"]).dropna(
        subset=["ret_actual", "market_leg"]
    )

    decomp_combined = pd.concat(
        [
            d.assign(pool=lab)[
                ["pool", "ticker", "date", "ret_actual", "market_leg",
                 "selection_residual"]
            ]
            for lab, d in (
                ("AL_F6_signal", decomp_alf6),
                ("AL_F6_random", decomp_alf6_rnd),
                ("PRSR_signal", decomp_prsr),
                ("PRSR_random", decomp_prsr_rnd),
            )
        ],
        ignore_index=True,
    )
    decomp_combined.to_csv(OUT_DECOMP, index=False)

    timing_v, timing_info = timing_verdict(decomp_alf6, decomp_prsr)

    # Verdicts
    rank_verdicts = {
        label: rank_verdict(
            rank_summaries[label].get("mean_rank_pct", float("nan")),
            rank_summaries[label].get("ks_p", float("nan")),
        )
        if rank_summaries[label].get("n", 0) > 0
        else "no_data"
        for label in pools_ranked
    }
    cluster_verdicts = {
        label: cluster_verdict(c["lift"], c["p_value"])
        for label, c in clusters.items()
    }

    # Cross-signal conflict tags
    cross_rank = cross_signal(
        rank_verdicts["AL_F6_signal"], rank_verdicts["PRSR_signal"]
    )
    cross_cluster = cross_signal(
        cluster_verdicts["AL_F6_signal"], cluster_verdicts["PRSR_signal"]
    )

    next_thread, next_reason = decision_tree(
        top20_rebound["AL_F6_random"]["rebound_share"],
        top20_rebound["PRSR_random"]["rebound_share"],
        rank_verdicts["AL_F6_signal"],
        rank_verdicts["PRSR_signal"],
        cluster_verdicts["AL_F6_signal"],
        cluster_verdicts["PRSR_signal"],
        timing_v,
    )

    print("[7/7] rendering report …", flush=True)
    ctx = {
        "pools": pools,
        "lorenz": lorenz,
        "core_daily": core_daily,
        "xu100_info": xu_info,
        "top20_rebound": top20_rebound,
        "clusters": clusters,
        "rank_summaries": rank_summaries,
        "rank_verdicts": rank_verdicts,
        "cluster_verdicts": cluster_verdicts,
        "cross_rank": cross_rank,
        "cross_cluster": cross_cluster,
        "ew_strategy_summary": ew_strategy_summary,
        "timing_verdict": timing_v,
        "timing_info": timing_info,
        "decomp_n": {
            "AL_F6": len(decomp_alf6),
            "PRSR": len(decomp_prsr),
        },
        "next_thread": next_thread,
        "next_thread_reason": next_reason,
    }
    md = render_report(ctx)
    OUT_REPORT.write_text(md, encoding="utf-8")
    print(f"\nReport written: {OUT_REPORT}")
    print(f"  next_thread: {next_thread}")
    print(f"  reason: {next_reason}")


if __name__ == "__main__":
    main()
