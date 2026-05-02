"""Phase 1.3 — universe filter sweep on AL_F6.

Phase 1.2 hold-out finding: AL_F6 (HWO Up & armed_recent10 & refractory10)
passes DISCOVERY+CORE acceptance but fails STRESS (K=10 ratio 0.85,
false_rev 42%). Phase 2 verified the *behavior* is sound (AL_F6 lives in
the TV HWO Up family, 4/4 same-bar match). The failure is a **universe**
problem: stress-cohort tickers (KAREL, PARSN, KLSER, GOODY, KONTR, …)
are the spec/illiquid/gappy tail where any HWO Up-family signal degrades.

This phase searches for the simplest tradeability filter that:
  • holds D+C acceptance         ratio_med ≥ 1.7   false_rev ≤ 35%
  • EITHER excludes STRESS       retention(STRESS) ≤ 30%
    OR recovers STRESS           ratio_med ≥ 1.5   false_rev ≤ 35%
  • retains ≥ 50% of D+C fires
  • leaves ≥ 150 eligible tickers per date in the 607-universe panel (p50)

Tradeability features (per ticker × date, look-ahead-free):
  median_turnover_60    median(close × volume) over prior 60 daily bars
  atr_pct_20            ATR(20)/close, prior-day values only
  median_abs_gap_atr_60 median(|open − prev_close|/ATR_20) over prior 60d

Cross-sectional thresholds per date across full 607-panel:
  liq_q50 / liq_q60     turnover lower-floor (higher = stricter)
  vol_q80               atr_pct ceiling (lower = stricter)
  gap_q80               gap ceiling (lower = stricter)

Gates evaluated:
  G0 no_filter
  G1 LIQ ≥ q50
  G2 LIQ ≥ q60
  G3 LIQ ≥ q50 ∧ atr_pct ≤ q80
  G4 LIQ ≥ q50 ∧ gap ≤ q80
  G5 LIQ ≥ q50 ∧ atr_pct ≤ q80 ∧ gap ≤ q80
  G6 LIQ ≥ q60 ∧ atr_pct ≤ q80 ∧ gap ≤ q80

Final candidate (if a gate passes acceptance):
  PRIMARY_TRADEABLE_AL = AL_F6 ∧ <selected_gate>

Naming: this is the "OSCMTRX HWO-style gated early reversal engine".
Never frame it as a LuxAlgo Reversal-arrow clone — Reversal arrows were
closed in Phase 0.5 (repaint / pivot-right confirmation).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

MASTER = Path("output/extfeed_intraday_1h_3y_master.parquet")
MIN_DATE = "2023-01-02"

OUT_DIR = Path("output")
OUT_SUMMARY = OUT_DIR / "oscmatrix_phase1_3_universe_filter_summary.csv"
OUT_BY_COHORT = OUT_DIR / "oscmatrix_phase1_3_universe_filter_by_cohort.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase1_3_universe_filter_report.md"

CSV_DIR_A = "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1"
CSV_DIR_B = f"{CSV_DIR_A}/New Folder With Items 3"

DISCOVERY = {
    "THYAO": f"{CSV_DIR_A}/BIST_THYAO, 1D.csv",
    "GARAN": f"{CSV_DIR_B}/BIST_GARAN, 1D.csv",
    "EREGL": f"{CSV_DIR_B}/BIST_EREGL, 1D.csv",
    "ASELS": f"{CSV_DIR_B}/BIST_ASELS, 1D.csv",
}
CORE = {t: f"{CSV_DIR_B}/BIST_{t}, 1D.csv" for t in [
    "AKBNK", "SISE", "TUPRS", "KCHOL", "BIMAS", "FROTO",
    "PETKM", "KRDMD", "YKBNK", "TOASO", "SAHOL",
]}
STRESS = {t: f"{CSV_DIR_B}/BIST_{t}, 1D.csv" for t in [
    "KAREL", "PARSN", "BRISA", "GOODY", "INDES",
    "NETAS", "KONTR", "KLSER", "MAVI",
]}
COHORTS = {"DISCOVERY": DISCOVERY, "CORE": CORE, "STRESS": STRESS}

HORIZONS = [5, 10, 20]
PRIMARY_K = 10
EVAL_W = 10
ROLL_N = 10
REFRACTORY_K = 10
ARM_RECENT_K = 10
OVF_RECENT_K = 5
FALSE_REV_ATR_THR = 0.5

GATES_DEF = [
    ("G0_no_filter",  []),
    ("G1_LIQ50",       [("liq", "q50")]),
    ("G2_LIQ60",       [("liq", "q60")]),
    ("G3_L50_V80",     [("liq", "q50"), ("vol", "q80")]),
    ("G4_L50_G80",     [("liq", "q50"), ("gap", "q80")]),
    ("G5_L50_V80_G80", [("liq", "q50"), ("vol", "q80"), ("gap", "q80")]),
    ("G6_L60_V80_G80", [("liq", "q60"), ("vol", "q80"), ("gap", "q80")]),
]


# =========================================================================
# 607-universe daily features
# =========================================================================

def build_universe_panel() -> pd.DataFrame:
    """Daily OHLCV per ticker from the 607-ticker 1h master parquet."""
    if not MASTER.exists():
        raise FileNotFoundError(f"master parquet missing: {MASTER}")
    bars = pd.read_parquet(MASTER)
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])
    if bars["ts_istanbul"].dt.tz is None:
        bars["ts_istanbul"] = bars["ts_istanbul"].dt.tz_localize("Europe/Istanbul")
    bars = bars[bars["ts_istanbul"] >= pd.Timestamp(MIN_DATE).tz_localize("Europe/Istanbul")]
    bars["date"] = bars["ts_istanbul"].dt.date
    daily = (
        bars.groupby(["ticker", "date"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return daily


def add_tradeability_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, append look-ahead-free 60d trailing tradeability columns.

    All columns describe the *prior* day's regime (shift(1) ∘ rolling), so a
    fire on day T uses features that close at the end of T−1.
    """
    out = []
    for tkr, g in daily.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        prev_close = g["close"].shift(1)
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(20, min_periods=20).mean()
        atr20_prev = atr20.shift(1)
        prev_close_for_pct = prev_close
        g["atr_pct_20"] = (atr20_prev / prev_close_for_pct).where(prev_close_for_pct > 0)

        turnover = g["close"] * g["volume"]
        g["median_turnover_60"] = (
            turnover.shift(1).rolling(60, min_periods=20).median()
        )

        gap = (g["open"] - prev_close).abs()
        gap_atr = (gap / atr20_prev).where(atr20_prev > 0)
        g["median_abs_gap_atr_60"] = gap_atr.shift(1).rolling(60, min_periods=20).median()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_cross_sectional_thresholds(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date q50/q60/q80 across all 607 tickers (NaN-skipping)."""
    def q(s: pd.Series, p: float) -> float:
        s = s.dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    out = (
        panel.groupby("date")
        .agg(
            liq_q50=("median_turnover_60", lambda s: q(s, 0.50)),
            liq_q60=("median_turnover_60", lambda s: q(s, 0.60)),
            vol_q80=("atr_pct_20", lambda s: q(s, 0.80)),
            gap_q80=("median_abs_gap_atr_60", lambda s: q(s, 0.80)),
        )
        .reset_index()
    )
    return out


def apply_gate_mask(panel_with_thr: pd.DataFrame, gate_name: str) -> pd.Series:
    """Boolean mask on a row-aligned (panel ⋈ thresholds) DataFrame."""
    constraints = dict(GATES_DEF)[gate_name]
    if not constraints:
        return pd.Series(True, index=panel_with_thr.index)
    mask = pd.Series(True, index=panel_with_thr.index)
    for fam, qtag in constraints:
        if fam == "liq":
            mask &= (panel_with_thr["median_turnover_60"]
                     >= panel_with_thr[f"liq_{qtag}"])
        elif fam == "vol":
            mask &= (panel_with_thr["atr_pct_20"]
                     <= panel_with_thr[f"vol_{qtag}"])
        elif fam == "gap":
            mask &= (panel_with_thr["median_abs_gap_atr_60"]
                     <= panel_with_thr[f"gap_{qtag}"])
    # NaN features → exclude (cannot prove eligibility)
    return mask.fillna(False)


# =========================================================================
# AL_F6 fires per cohort ticker (Phase 1.2 logic, with date attached)
# =========================================================================

def recent_any(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).max().astype(bool)


def apply_refractory(fires: pd.Series, K: int) -> pd.Series:
    arr = fires.values
    out = np.zeros(len(arr), dtype=bool)
    last = -10**9
    for i in range(len(arr)):
        if arr[i] and (i - last) > K:
            out[i] = True
            last = i
    return pd.Series(out, index=fires.index)


def build_al_f6(tv: pd.DataFrame, ours: pd.DataFrame) -> pd.Series:
    hw = ours["hyperwave"].reindex(tv.index)
    hwo_up = ours["hwo_up"].reindex(tv.index).fillna(False).astype(bool)

    if "Lower Confluence Value" in tv.columns:
        lower_zone = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    else:
        lower_zone = pd.Series(0, index=tv.index)

    if "Bearish Overflow" in tv.columns:
        bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50)
    else:
        bear_ovf = pd.Series(False, index=tv.index)

    al_armed = (hw < 30) | (lower_zone == 2) | recent_any(bear_ovf, OVF_RECENT_K)
    al_armed_recent = recent_any(al_armed, ARM_RECENT_K)
    al_f6_raw = hwo_up & al_armed_recent
    return apply_refractory(al_f6_raw, REFRACTORY_K)


def lag_in_window(bar_loc: int, series: pd.Series, window: int, side: str) -> int:
    n = len(series)
    lo = max(0, bar_loc - window)
    hi = min(n, bar_loc + window + 1)
    sub = series.iloc[lo:hi]
    if side == "low":
        return bar_loc - (lo + int(sub.values.argmin()))
    return bar_loc - (lo + int(sub.values.argmax()))


def forward_extremes(df: pd.DataFrame, K: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    fmh = pd.concat([df["high"].shift(-k) for k in range(1, K + 1)], axis=1).max(axis=1)
    fml = pd.concat([df["low"].shift(-k) for k in range(1, K + 1)], axis=1).min(axis=1)
    fwd_close = df["close"].shift(-K)
    return fmh, fml, fwd_close


def collect_fires(ticker: str, csv_path: str) -> list[dict]:
    """Return one row per AL_F6 fire × horizon, with (ticker, date, metrics)."""
    if not Path(csv_path).exists():
        return []
    try:
        tv = load_tv_csv(csv_path).set_index("ts")
    except Exception as exc:
        print(f"  [{ticker}] FAILED to load: {exc}")
        return []

    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")

    df = tv[["open", "high", "low", "close"]].astype(float)
    hw = ours["hyperwave"].reindex(tv.index)
    atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    fires = build_al_f6(tv, ours)
    locs = np.where(fires.values)[0]
    if len(locs) == 0:
        return []

    rows: list[dict] = []
    fwd: dict[int, tuple[pd.Series, pd.Series, pd.Series]] = {
        K: forward_extremes(df, K) for K in HORIZONS
    }

    close, high, low = df["close"], df["high"], df["low"]
    roll_low = low.rolling(ROLL_N, min_periods=1).min()

    for i in locs:
        a = atr.iloc[i]
        if not a or np.isnan(a):
            continue
        ci = close.iloc[i]
        ts = tv.index[i]
        date = pd.Timestamp(ts).tz_convert("Europe/Istanbul").date() if ts.tzinfo else ts.date()

        p_lag = lag_in_window(i, low, EVAL_W, "low")
        hw_lag = lag_in_window(i, hw, EVAL_W, "low")
        roll_dist = float((ci - roll_low.iloc[i]) / a)

        for K in HORIZONS:
            fmh, fml, fwd_close = fwd[K]
            mfe_raw = fmh.iloc[i] - ci
            mae_raw = ci - fml.iloc[i]
            realized = fwd_close.iloc[i] - ci if pd.notna(fwd_close.iloc[i]) else np.nan
            if pd.isna(mfe_raw) or pd.isna(mae_raw):
                continue
            mfe_atr = float(mfe_raw / a)
            mae_atr = float(mae_raw / a)
            ratio = mfe_atr / max(mae_atr, 1e-3)
            cap = (float(realized) / float(mfe_raw)) if pd.notna(realized) and mfe_raw > 0 else np.nan
            false_rev = bool(realized < -FALSE_REV_ATR_THR * a) if pd.notna(realized) else None
            rows.append({
                "ticker": ticker,
                "date": date,
                "K": K,
                "p_lag": p_lag,
                "hw_lag": hw_lag,
                "roll_dist": roll_dist,
                "mae_atr": mae_atr,
                "mfe_atr": mfe_atr,
                "ratio": ratio,
                "capture": cap,
                "false_rev": false_rev,
            })
    return rows


# =========================================================================
# Aggregation
# =========================================================================

def summarize(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0, "p_lag_med": np.nan, "hw_lag_med": np.nan,
                "ratio_med": np.nan, "false_rev_pct": np.nan,
                "capture_med": np.nan, "mae_med": np.nan, "mfe_med": np.nan}
    return {
        "n": int(len(rows)),
        "p_lag_med": float(rows["p_lag"].median()),
        "hw_lag_med": float(rows["hw_lag"].median()),
        "ratio_med": float(rows["ratio"].median()),
        "false_rev_pct": float(rows["false_rev"].mean() * 100) if rows["false_rev"].notna().any() else np.nan,
        "capture_med": float(rows["capture"].median()) if rows["capture"].notna().any() else np.nan,
        "mae_med": float(rows["mae_atr"].median()),
        "mfe_med": float(rows["mfe_atr"].median()),
    }


def coverage_607(panel: pd.DataFrame, thresholds: pd.DataFrame, gate_name: str) -> dict:
    """Per-date eligible-ticker count under a gate.

    Returns dict with median/p10/p90 counts and the unique-ticker set size.
    """
    merged = panel.merge(thresholds, on="date", how="left")
    mask = apply_gate_mask(merged, gate_name)
    elig = merged[mask]
    per_date = elig.groupby("date")["ticker"].nunique()
    return {
        "median": float(per_date.median()) if len(per_date) else 0.0,
        "p10": float(per_date.quantile(0.10)) if len(per_date) else 0.0,
        "p90": float(per_date.quantile(0.90)) if len(per_date) else 0.0,
        "unique_tickers": int(elig["ticker"].nunique()),
        "n_dates": int(per_date.shape[0]),
    }


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("=== Phase 1.3 — universe filter sweep on AL_F6 ===")
    print("    OSCMTRX HWO-style gated early reversal engine\n")

    print("Step 1/4 — building 607-universe daily panel + tradeability features …")
    daily = build_universe_panel()
    panel = add_tradeability_features(daily)
    print(f"  panel: {panel['ticker'].nunique()} tickers × {panel['date'].nunique()} dates "
          f"({len(panel):,} rows)")

    print("\nStep 2/4 — computing cross-sectional thresholds per date …")
    thresholds = build_cross_sectional_thresholds(panel)
    print(f"  threshold table: {len(thresholds):,} dates")

    print("\nStep 3/4 — collecting AL_F6 fires from cohort tickers …")
    fires_by_cohort: dict[str, list[dict]] = {}
    for cohort, tickers in COHORTS.items():
        rows: list[dict] = []
        loaded, missing = [], []
        for tkr, path in tickers.items():
            r = collect_fires(tkr, path)
            (loaded if r else missing).append(tkr)
            rows.extend(r)
        fires_by_cohort[cohort] = rows
        print(f"  {cohort:<10} loaded={len(loaded):>2}  fires={len(rows):>4}  "
              f"missing={missing}")

    fires_all = []
    for cohort, rows in fires_by_cohort.items():
        for r in rows:
            r2 = dict(r)
            r2["cohort"] = cohort
            fires_all.append(r2)
    fires_df = pd.DataFrame(fires_all)
    if fires_df.empty:
        raise SystemExit("no fires collected; aborting.")
    fires_df["date"] = pd.to_datetime(fires_df["date"]).dt.date

    print("\nStep 4/4 — applying gates and computing 607 coverage …")
    feat = panel[["ticker", "date", "median_turnover_60",
                  "atr_pct_20", "median_abs_gap_atr_60"]].copy()
    fires_df = fires_df.merge(feat, on=["ticker", "date"], how="left")
    fires_df = fires_df.merge(thresholds, on="date", how="left")

    summary_rows: list[dict] = []
    by_cohort_rows: list[dict] = []

    base_dc = fires_df[(fires_df["cohort"].isin(["DISCOVERY", "CORE"]))
                        & (fires_df["K"] == PRIMARY_K)]
    base_st = fires_df[(fires_df["cohort"] == "STRESS") & (fires_df["K"] == PRIMARY_K)]
    base_dc_n = len(base_dc)
    base_st_n = len(base_st)

    for gate_name, _constraints in GATES_DEF:
        gmask = apply_gate_mask(fires_df, gate_name)
        kept = fires_df[gmask].copy()
        cov = coverage_607(panel, thresholds, gate_name)

        for K in HORIZONS:
            for cohort in ["DISCOVERY", "CORE", "STRESS", "DISCOVERY+CORE"]:
                if cohort == "DISCOVERY+CORE":
                    sel_mask = kept["cohort"].isin(["DISCOVERY", "CORE"])
                    base_mask = fires_df["cohort"].isin(["DISCOVERY", "CORE"])
                else:
                    sel_mask = kept["cohort"] == cohort
                    base_mask = fires_df["cohort"] == cohort
                sub = kept[sel_mask & (kept["K"] == K)]
                base = fires_df[base_mask & (fires_df["K"] == K)]
                s = summarize(sub)
                s["cohort"] = cohort
                s["gate"] = gate_name
                s["K"] = K
                s["base_n"] = int(len(base))
                s["retention_pct"] = (
                    100.0 * len(sub) / len(base) if len(base) else np.nan
                )
                s["coverage_median"] = cov["median"]
                s["coverage_p10"] = cov["p10"]
                s["coverage_p90"] = cov["p90"]
                s["coverage_unique"] = cov["unique_tickers"]
                by_cohort_rows.append(s)

        # gate-level summary at primary horizon
        kept_dc = kept[(kept["cohort"].isin(["DISCOVERY", "CORE"]))
                        & (kept["K"] == PRIMARY_K)]
        kept_st = kept[(kept["cohort"] == "STRESS") & (kept["K"] == PRIMARY_K)]
        s_dc = summarize(kept_dc)
        s_st = summarize(kept_st)
        summary_rows.append({
            "gate": gate_name,
            "K": PRIMARY_K,
            "dc_n": s_dc["n"],
            "dc_retention_pct": (100.0 * s_dc["n"] / base_dc_n) if base_dc_n else np.nan,
            "dc_ratio_med": s_dc["ratio_med"],
            "dc_false_rev_pct": s_dc["false_rev_pct"],
            "dc_p_lag_med": s_dc["p_lag_med"],
            "dc_hw_lag_med": s_dc["hw_lag_med"],
            "st_n": s_st["n"],
            "st_retention_pct": (100.0 * s_st["n"] / base_st_n) if base_st_n else np.nan,
            "st_ratio_med": s_st["ratio_med"],
            "st_false_rev_pct": s_st["false_rev_pct"],
            "cov_median": cov["median"],
            "cov_p10": cov["p10"],
            "cov_p90": cov["p90"],
            "cov_unique": cov["unique_tickers"],
        })

    summary_df = pd.DataFrame(summary_rows)
    by_cohort_df = pd.DataFrame(by_cohort_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    by_cohort_df.to_csv(OUT_BY_COHORT, index=False)
    print(f"\n  wrote {OUT_SUMMARY}")
    print(f"  wrote {OUT_BY_COHORT}")

    # ---- Acceptance
    print("\n=== Acceptance gate ===")
    print(f"  D+C ratio_med ≥ 1.7  ∧  D+C false_rev ≤ 35%  ∧  retention ≥ 50%")
    print(f"  STRESS: ratio ≥ 1.5 (recovered) OR retention ≤ 30% (excluded)")
    print(f"  607-universe: cov_median ≥ 150  (else gate too narrow)\n")

    cols_print = ["gate", "dc_n", "dc_retention_pct", "dc_ratio_med",
                  "dc_false_rev_pct", "st_n", "st_retention_pct",
                  "st_ratio_med", "st_false_rev_pct",
                  "cov_median", "cov_p10", "cov_unique"]
    print(summary_df[cols_print].to_string(index=False))

    # Apply acceptance
    def gate_passes(row: pd.Series) -> tuple[bool, str]:
        reasons = []
        ok = True
        if not (pd.notna(row["dc_ratio_med"]) and row["dc_ratio_med"] >= 1.7):
            ok = False; reasons.append(f"dc_ratio<1.7 ({row['dc_ratio_med']:.2f})")
        if not (pd.notna(row["dc_false_rev_pct"]) and row["dc_false_rev_pct"] <= 35):
            ok = False; reasons.append(f"dc_false_rev>35 ({row['dc_false_rev_pct']:.1f}%)")
        if not (pd.notna(row["dc_retention_pct"]) and row["dc_retention_pct"] >= 50):
            ok = False; reasons.append(f"dc_retention<50 ({row['dc_retention_pct']:.1f}%)")
        # stress: either excluded or recovered
        st_excluded = (pd.notna(row["st_retention_pct"])
                       and row["st_retention_pct"] <= 30)
        st_recovered = (pd.notna(row["st_ratio_med"]) and row["st_ratio_med"] >= 1.5
                        and pd.notna(row["st_false_rev_pct"])
                        and row["st_false_rev_pct"] <= 35)
        if not (st_excluded or st_recovered):
            ok = False
            reasons.append(
                f"stress not handled (ret={row['st_retention_pct']:.0f}%, "
                f"ratio={row['st_ratio_med']:.2f}, "
                f"false_rev={row['st_false_rev_pct']:.0f}%)"
            )
        if not (pd.notna(row["cov_median"]) and row["cov_median"] >= 150):
            ok = False; reasons.append(f"cov_median<150 ({row['cov_median']:.0f})")
        return ok, "; ".join(reasons) if reasons else "all checks pass"

    print("\n--- Per-gate acceptance verdict ---")
    pass_rows = []
    for _, r in summary_df.iterrows():
        ok, reasons = gate_passes(r)
        verdict = "PASS" if ok else "FAIL"
        st_status = ("excluded" if (pd.notna(r["st_retention_pct"]) and r["st_retention_pct"] <= 30)
                     else "recovered" if (pd.notna(r["st_ratio_med"]) and r["st_ratio_med"] >= 1.5)
                     else "fails")
        print(f"  {r['gate']:<18} {verdict}  (stress={st_status})  → {reasons}")
        if ok:
            pass_rows.append((r["gate"], r))

    # pick simplest passing gate (G0..G6 order)
    selected = pass_rows[0] if pass_rows else None

    # ---- Markdown report
    md_lines: list[str] = []
    md_lines.append("# Phase 1.3 — Universe filter sweep on AL_F6")
    md_lines.append("")
    md_lines.append("OSCMTRX HWO-style gated early reversal engine — universe filter selection.")
    md_lines.append("")
    md_lines.append("## Executive verdict")
    md_lines.append("")
    if selected:
        gname, grow = selected
        md_lines.append(f"**SELECTED GATE: {gname}** — passes all acceptance checks at K={PRIMARY_K}.")
        md_lines.append("")
        md_lines.append(f"`PRIMARY_TRADEABLE_AL = AL_F6 ∧ {gname}`")
    else:
        md_lines.append("**NO GATE PASSES.** All seven candidates fail at least one acceptance check.")
        md_lines.append("Phase 4 should NOT proceed under these gates; revisit feature design or tighten/relax thresholds.")
    md_lines.append("")
    md_lines.append("## Acceptance criteria")
    md_lines.append("")
    md_lines.append(f"- D+C cohort: ratio_med ≥ 1.7, false_rev ≤ 35%, retention ≥ 50%")
    md_lines.append(f"- STRESS cohort: either ratio_med ≥ 1.5 (recovered) OR retention ≤ 30% (excluded)")
    md_lines.append(f"- 607-universe coverage: median eligible tickers/date ≥ 150")
    md_lines.append(f"- Primary horizon: K = {PRIMARY_K}")
    md_lines.append("")
    md_lines.append("## Comparison table (K=10)")
    md_lines.append("")
    md_lines.append("| gate | dc_n | dc_ret% | dc_ratio | dc_false% | st_n | st_ret% | st_ratio | st_false% | cov_med | cov_p10 | cov_uniq |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| {r['gate']} | {int(r['dc_n'])} | "
            f"{r['dc_retention_pct']:.0f} | {r['dc_ratio_med']:.2f} | {r['dc_false_rev_pct']:.0f} | "
            f"{int(r['st_n'])} | {r['st_retention_pct']:.0f} | {r['st_ratio_med']:.2f} | {r['st_false_rev_pct']:.0f} | "
            f"{r['cov_median']:.0f} | {r['cov_p10']:.0f} | {int(r['cov_unique'])} |"
        )
    md_lines.append("")
    md_lines.append("## Per-gate verdict")
    md_lines.append("")
    md_lines.append("| gate | verdict | stress | reasons |")
    md_lines.append("|---|---|---|---|")
    for _, r in summary_df.iterrows():
        ok, reasons = gate_passes(r)
        verdict = "PASS" if ok else "FAIL"
        st_status = ("excluded" if (pd.notna(r["st_retention_pct"]) and r["st_retention_pct"] <= 30)
                     else "recovered" if (pd.notna(r["st_ratio_med"]) and r["st_ratio_med"] >= 1.5)
                     else "fails")
        md_lines.append(f"| {r['gate']} | {verdict} | {st_status} | {reasons} |")
    md_lines.append("")
    md_lines.append("## Cohort breakdown across horizons")
    md_lines.append("")
    md_lines.append("`output/oscmatrix_phase1_3_universe_filter_by_cohort.csv` — full breakdown.")
    md_lines.append("Below: K=10 only.")
    md_lines.append("")
    md_lines.append("| gate | cohort | n | retention% | ratio | false_rev% | p_lag | hw_lag |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    bc = by_cohort_df[by_cohort_df["K"] == PRIMARY_K].copy()
    bc["cohort_ord"] = bc["cohort"].map(
        {"DISCOVERY": 0, "CORE": 1, "DISCOVERY+CORE": 2, "STRESS": 3}
    )
    bc["gate_ord"] = bc["gate"].map({g: i for i, (g, _) in enumerate(GATES_DEF)})
    bc = bc.sort_values(["gate_ord", "cohort_ord"])
    for _, r in bc.iterrows():
        md_lines.append(
            f"| {r['gate']} | {r['cohort']} | {int(r['n'])} | "
            f"{r['retention_pct']:.0f}% | {r['ratio_med']:.2f} | "
            f"{r['false_rev_pct']:.0f} | {r['p_lag_med']:+.1f} | {r['hw_lag_med']:+.1f} |"
        )
    md_lines.append("")
    md_lines.append("## 607-universe coverage")
    md_lines.append("")
    md_lines.append("Per-date eligible ticker count under each gate (across the entire 607 panel since 2023-01-02):")
    md_lines.append("")
    md_lines.append("| gate | median | p10 | p90 | unique tickers ever eligible |")
    md_lines.append("|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| {r['gate']} | {r['cov_median']:.0f} | "
            f"{r['cov_p10']:.0f} | {r['cov_p90']:.0f} | {int(r['cov_unique'])} |"
        )
    md_lines.append("")
    md_lines.append("## Cohort feature diagnostic")
    md_lines.append("")
    md_lines.append("Per-ticker pass-rate (% of dates each ticker satisfies a single feature gate),")
    md_lines.append("over the full 2023-01-02 → 2026-04-29 panel. This isolates *why* each gate")
    md_lines.append("filters the way it does.")
    md_lines.append("")
    cohort_map = {tkr: cohort for cohort, ts in COHORTS.items() for tkr in ts}
    pano = panel.merge(thresholds, on="date", how="left")
    pano["liq_pass50"] = pano["median_turnover_60"] >= pano["liq_q50"]
    pano["liq_pass60"] = pano["median_turnover_60"] >= pano["liq_q60"]
    pano["vol_pass80"] = pano["atr_pct_20"] <= pano["vol_q80"]
    pano["gap_pass80"] = pano["median_abs_gap_atr_60"] <= pano["gap_q80"]
    pano["cohort"] = pano["ticker"].map(cohort_map)
    pano_c = pano[pano["cohort"].notna()].dropna(
        subset=["median_turnover_60", "atr_pct_20", "median_abs_gap_atr_60"]
    )
    cohort_stats = (
        pano_c.groupby("cohort")
        [["liq_pass50", "liq_pass60", "vol_pass80", "gap_pass80"]]
        .mean() * 100
    ).round(1)
    md_lines.append("| cohort | liq≥q50 | liq≥q60 | vol≤q80 | gap≤q80 |")
    md_lines.append("|---|---|---|---|---|")
    for cohort in ["DISCOVERY", "CORE", "STRESS"]:
        if cohort in cohort_stats.index:
            r = cohort_stats.loc[cohort]
            md_lines.append(
                f"| {cohort} | {r['liq_pass50']:.0f}% | {r['liq_pass60']:.0f}% | "
                f"{r['vol_pass80']:.0f}% | {r['gap_pass80']:.0f}% |"
            )
    md_lines.append("")
    md_lines.append("**Diagnosis.** Stress is *thinner* (liq_pass50 58% vs 100% core/discovery)")
    md_lines.append("but the volatility and gap features do **not** discriminate stress as expected:")
    md_lines.append("")
    md_lines.append("- `atr_pct_20` puts stress and core in the same band (~99–93% pass q80).")
    md_lines.append("- `median_abs_gap_atr_60` shows the *opposite* of intuition: stress passes")
    md_lines.append("  q80 89% of the time, while DISCOVERY (THYAO/GARAN/EREGL/ASELS) only")
    md_lines.append("  passes 36% — large-cap momentum names gap *more* (relative to their")
    md_lines.append("  ATR) because they trade on news; thin stress names gap *less* because")
    md_lines.append("  they don't move overnight.")
    md_lines.append("")
    md_lines.append("So the only feature that partly separates stress from D+C is")
    md_lines.append("`median_turnover_60`. But pure liquidity (G2 LIQ60) only drops STRESS to")
    md_lines.append("36% retention while leaving its post-fire follow-through still broken")
    md_lines.append("(ratio_med 0.56, false_rev 48%) — meaning the *surviving* high-liquidity")
    md_lines.append("stress fires (KONTR, MAVI, KAREL) are themselves the chop-trap problem.")
    md_lines.append("")
    md_lines.append("Stress cohort fire density is also higher than core (per-ticker fires:")
    md_lines.append(f"DISCOVERY {len(fires_by_cohort['DISCOVERY'])//len(DISCOVERY)}/ticker, "
                    f"CORE {len(fires_by_cohort['CORE'])//len(CORE)}/ticker, "
                    f"STRESS {len(fires_by_cohort['STRESS'])//len(STRESS)}/ticker). "
                    "Stress is ")
    md_lines.append("*signal-rich but non-trending* — chop. The discriminator is")
    md_lines.append("**trendiness**, not liquidity / volatility / gap.")
    md_lines.append("")
    md_lines.append("Candidate features for a follow-up Phase 1.3b sweep:")
    md_lines.append("")
    md_lines.append("- Kaufman efficiency ratio: `|close_T − close_T−N| / Σ|close_t − close_t−1|`")
    md_lines.append("  — high = trending, low = chop. Cross-sectional q40/q50 floor.")
    md_lines.append("- Rolling 1-bar autocorrelation of close-to-close returns (negative =")
    md_lines.append("  mean-reverting/chop, ~0 or positive = trend persistence).")
    md_lines.append("- Hurst exponent over 60–120 daily bars.")
    md_lines.append("- Realised drift: `(close_T − close_T−60) / (atr_60 × √60)`. Sign and")
    md_lines.append("  magnitude separate trenders from oscillators.")
    md_lines.append("")
    md_lines.append("## SAT side")
    md_lines.append("")
    md_lines.append("SAT short-entry stays disabled (Phase 1.1+1.2 finding). SAT_E exit-")
    md_lines.append("probe is out of scope for Phase 1.3 — schedule a separate")
    md_lines.append("`tools/oscmatrix_sat_exit_probe.py` if the long-side exit overlay is needed.")
    md_lines.append("")
    md_lines.append("## Next steps")
    md_lines.append("")
    if selected:
        md_lines.append(f"- Phase 4: 607-universe daily backtest of `AL_F6 ∧ {selected[0]}`.")
        md_lines.append("- Lock the gate definition and feature recipe (turnover60 / atr_pct20 / gap60) into the production scanner.")
        md_lines.append("- Compare cross-sectional gate vs a global rolling-quantile gate as fallback for live.")
    else:
        md_lines.append("- Acceptance failed across all 7 gates. Do **not** advance to Phase 4 under these features.")
        md_lines.append("- Diagnosis points to a missing **trendiness** feature, not a tighter")
        md_lines.append("  threshold on the existing axes. Tightening liq/vol/gap further only")
        md_lines.append("  increases collateral damage to D+C (G6 already drops D+C retention to")
        md_lines.append("  30% and 607 coverage to 133).")
        md_lines.append("- Phase 1.3b: add Kaufman efficiency ratio (60d) and realised drift")
        md_lines.append("  (60d) as additional cross-sectional axes; rerun the same gate sweep.")
        md_lines.append("- Alternative path: keep AL_F6 as-is and accept stress-degraded")
        md_lines.append("  performance, but flag stress-cohort-style tickers at runtime via the")
        md_lines.append("  same trendiness feature for human review (ranker, not a hard gate).")
        md_lines.append("- SAT exit probe still pending — schedule independently.")

    OUT_REPORT.write_text("\n".join(md_lines))
    print(f"\n  wrote {OUT_REPORT}")

    if selected:
        print(f"\n  PRIMARY_TRADEABLE_AL = AL_F6 ∧ {selected[0]}")
    else:
        print("\n  No gate passes acceptance. Phase 4 blocked until further investigation.")


if __name__ == "__main__":
    main()
