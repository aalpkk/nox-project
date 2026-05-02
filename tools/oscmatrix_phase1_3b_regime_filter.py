"""Phase 1.3b — efficiency/drift regime filter sweep on AL_F6.

Phase 1.3 verdict: liq/vol/gap features cannot discriminate stress from
D+C without collateral damage. Diagnosis: stress is signal-rich-but-non-
trending (chop). The correct discriminator is **trend efficiency**, not
tradeability.

This phase tests two repaint-safe regime features (computed only from past
data, shift(1) discipline so a fire on day T uses through-day-T−1 closes):

  er60        = |close − close[t−60]| / Σ|close.diff()|_60
                Kaufman efficiency ratio (KER). 1 = pure trend, 0 = pure chop.
  drift60     = close / close[t−60] − 1
  abs_drift60 = |drift60|        # accepts up- AND recovery-from-down trends

LIQ (`median_turnover_60`) is carried over from Phase 1.3 only as a
composite axis, never on its own.

Cross-sectional thresholds per date across the full 607-panel (q50, q60).

Gates (only objective rules; AL_F6 logic untouched):
  R0 no_filter
  R1 ER60        ≥ q50
  R2 ER60        ≥ q60
  R3 abs_drift60 ≥ q50
  R4 abs_drift60 ≥ q60
  R5 ER60 ≥ q50  ∧ LIQ ≥ q50
  R6 ER60 ≥ q60  ∧ LIQ ≥ q50
  R7 ER60 ≥ q50  ∧ abs_drift60 ≥ q50
  R8 ER60 ≥ q60  ∧ abs_drift60 ≥ q50 ∧ LIQ ≥ q50
  R9 ER60 ≥ q50  ∧ drift60 > −0.20      (diagnostic only — falling-knife
                                          guard; not selectable as winner)

Acceptance (must hold for the gate to be selected):
  D+C @ K=5  AND  D+C @ K=10:   ratio_med ≥ 1.7   ∧   false_rev_pct ≤ 35
  STRESS:                       retention ≤ 30%   OR   ratio_med ≥ 1.3
  607-universe:                 cov_median ≥ 150  (hard)
  D+C retention ≥ 50%           (preferred, soft warning if violated)
  Financial rationale:          operator must endorse the simplest passing gate.

If no R-gate (R0–R8) passes:
  AL_F6 remains a Core-style universe candidate ONLY (BIST100-like
  whitelist), NOT a 607-universe scanner candidate. Phase 4 then splits
  into (A) Core-style large-universe scan and (B) full-607 diagnostic only.

Naming: this is the "OSCMTRX HWO-style gated early reversal engine".
Do not frame it as a LuxAlgo Reversal-arrow clone.
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
OUT_SUMMARY = OUT_DIR / "oscmatrix_phase1_3b_regime_filter_summary.csv"
OUT_BY_COHORT = OUT_DIR / "oscmatrix_phase1_3b_regime_filter_by_cohort.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase1_3b_regime_filter_report.md"

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

# Gate constraint families:
#   ("er",       "qNN")    er60        ≥ er60_qNN
#   ("absdrift", "qNN")    abs_drift60 ≥ absdrift60_qNN
#   ("liq",      "qNN")    median_turnover_60 ≥ liq_qNN
#   ("drift_min", value)   drift60 > value          (constant, not quantile)
GATES_DEF = [
    ("R0_no_filter",          []),
    ("R1_ER50",                [("er",       "q50")]),
    ("R2_ER60",                [("er",       "q60")]),
    ("R3_AD50",                [("absdrift", "q50")]),
    ("R4_AD60",                [("absdrift", "q60")]),
    ("R5_ER50_LIQ50",          [("er", "q50"), ("liq", "q50")]),
    ("R6_ER60_LIQ50",          [("er", "q60"), ("liq", "q50")]),
    ("R7_ER50_AD50",           [("er", "q50"), ("absdrift", "q50")]),
    ("R8_ER60_AD50_LIQ50",     [("er", "q60"), ("absdrift", "q50"), ("liq", "q50")]),
    ("R9_ER50_DRIFTGUARD_DIAG", [("er", "q50"), ("drift_min", -0.20)]),  # diagnostic
]
DIAGNOSTIC_GATES = {"R9_ER50_DRIFTGUARD_DIAG"}


# =========================================================================
# 607-universe daily features
# =========================================================================

def build_universe_panel() -> pd.DataFrame:
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


def add_regime_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, append ER60, drift60, abs_drift60, median_turnover_60.

    All features are shift(1)'ed: a row dated T encodes regime through T−1.
    """
    out = []
    for tkr, g in daily.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        c = g["close"]
        diff_abs = c.diff().abs()
        denom = diff_abs.rolling(60, min_periods=30).sum()
        numer = (c - c.shift(60)).abs()
        er60_raw = (numer / denom).where(denom > 0)
        g["er60"] = er60_raw.shift(1)

        drift60_raw = c / c.shift(60) - 1
        g["drift60"] = drift60_raw.shift(1)
        g["abs_drift60"] = drift60_raw.abs().shift(1)

        turnover = c * g["volume"]
        g["median_turnover_60"] = (
            turnover.shift(1).rolling(60, min_periods=20).median()
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_cross_sectional_thresholds(panel: pd.DataFrame) -> pd.DataFrame:
    def q(s: pd.Series, p: float) -> float:
        s = s.dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    out = (
        panel.groupby("date")
        .agg(
            er60_q50=("er60", lambda s: q(s, 0.50)),
            er60_q60=("er60", lambda s: q(s, 0.60)),
            absdrift60_q50=("abs_drift60", lambda s: q(s, 0.50)),
            absdrift60_q60=("abs_drift60", lambda s: q(s, 0.60)),
            liq_q50=("median_turnover_60", lambda s: q(s, 0.50)),
        )
        .reset_index()
    )
    return out


def apply_gate_mask(panel_with_thr: pd.DataFrame, gate_name: str) -> pd.Series:
    constraints = dict(GATES_DEF)[gate_name]
    if not constraints:
        return pd.Series(True, index=panel_with_thr.index)
    mask = pd.Series(True, index=panel_with_thr.index)
    for fam, q in constraints:
        if fam == "er":
            mask &= panel_with_thr["er60"] >= panel_with_thr[f"er60_{q}"]
        elif fam == "absdrift":
            mask &= panel_with_thr["abs_drift60"] >= panel_with_thr[f"absdrift60_{q}"]
        elif fam == "liq":
            mask &= panel_with_thr["median_turnover_60"] >= panel_with_thr[f"liq_{q}"]
        elif fam == "drift_min":
            mask &= panel_with_thr["drift60"] > q
    return mask.fillna(False)


# =========================================================================
# AL_F6 fire collection (untouched signal logic)
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

def gate_passes(row: pd.Series) -> tuple[bool, bool, list[str]]:
    """Return (selectable, retention_warn, reasons).

    selectable: True if hard acceptance gates pass.
    retention_warn: True if D+C retention < 50% (soft, doesn't block).
    """
    reasons: list[str] = []
    ok = True
    for K in [5, 10]:
        rk = row.get(f"dc_ratio_k{K}", np.nan)
        fk = row.get(f"dc_false_k{K}", np.nan)
        if not (pd.notna(rk) and rk >= 1.7):
            ok = False; reasons.append(f"dc_ratio_k{K}<1.7 ({rk:.2f})")
        if not (pd.notna(fk) and fk <= 35):
            ok = False; reasons.append(f"dc_false_k{K}>35 ({fk:.0f}%)")
    st_excluded = (pd.notna(row["st_retention_pct"])
                   and row["st_retention_pct"] <= 30)
    st_recovered = (pd.notna(row["st_ratio_k10"]) and row["st_ratio_k10"] >= 1.3)
    if not (st_excluded or st_recovered):
        ok = False
        reasons.append(
            f"stress not handled (ret={row['st_retention_pct']:.0f}%, "
            f"ratio_k10={row['st_ratio_k10']:.2f})"
        )
    if not (pd.notna(row["cov_median"]) and row["cov_median"] >= 150):
        ok = False; reasons.append(f"cov_median<150 ({row['cov_median']:.0f})")
    retention_warn = (pd.notna(row["dc_retention_pct"])
                      and row["dc_retention_pct"] < 50)
    if retention_warn:
        reasons.append(f"WARN dc_retention<50 ({row['dc_retention_pct']:.0f}%)")
    return ok, retention_warn, reasons


def main() -> None:
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("=== Phase 1.3b — efficiency/drift regime filter sweep on AL_F6 ===")
    print("    OSCMTRX HWO-style gated early reversal engine\n")

    print("Step 1/4 — building 607-universe daily panel + regime features …")
    daily = build_universe_panel()
    panel = add_regime_features(daily)
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

    feat = panel[["ticker", "date", "er60", "drift60", "abs_drift60",
                  "median_turnover_60"]].copy()
    fires_df = fires_df.merge(feat, on=["ticker", "date"], how="left")
    fires_df = fires_df.merge(thresholds, on="date", how="left")

    print("\nStep 4/4 — applying gates and computing 607 coverage …")

    summary_rows: list[dict] = []
    by_cohort_rows: list[dict] = []

    base_dc_n = {K: int(((fires_df["cohort"].isin(["DISCOVERY", "CORE"]))
                          & (fires_df["K"] == K)).sum())
                  for K in HORIZONS}
    base_st_n = {K: int(((fires_df["cohort"] == "STRESS")
                          & (fires_df["K"] == K)).sum())
                  for K in HORIZONS}

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

        # gate-level summary across both critical horizons
        row = {"gate": gate_name}
        for K in [5, 10]:
            kdc = kept[(kept["cohort"].isin(["DISCOVERY", "CORE"])) & (kept["K"] == K)]
            kst = kept[(kept["cohort"] == "STRESS") & (kept["K"] == K)]
            sdc = summarize(kdc)
            sst = summarize(kst)
            row[f"dc_n_k{K}"] = sdc["n"]
            row[f"dc_ratio_k{K}"] = sdc["ratio_med"]
            row[f"dc_false_k{K}"] = sdc["false_rev_pct"]
            row[f"st_n_k{K}"] = sst["n"]
            row[f"st_ratio_k{K}"] = sst["ratio_med"]
            row[f"st_false_k{K}"] = sst["false_rev_pct"]
        # canonical retention figures from K=10
        row["dc_retention_pct"] = (
            100.0 * row["dc_n_k10"] / base_dc_n[10] if base_dc_n[10] else np.nan
        )
        row["st_retention_pct"] = (
            100.0 * row["st_n_k10"] / base_st_n[10] if base_st_n[10] else np.nan
        )
        row["cov_median"] = cov["median"]
        row["cov_p10"] = cov["p10"]
        row["cov_p90"] = cov["p90"]
        row["cov_unique"] = cov["unique_tickers"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    by_cohort_df = pd.DataFrame(by_cohort_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    by_cohort_df.to_csv(OUT_BY_COHORT, index=False)
    print(f"\n  wrote {OUT_SUMMARY}")
    print(f"  wrote {OUT_BY_COHORT}")

    # ---- Acceptance
    print("\n=== Acceptance gate ===")
    print("  D+C K=5 AND K=10:    ratio_med ≥ 1.7   AND   false_rev ≤ 35%")
    print("  STRESS:              retention ≤ 30%   OR    ratio_med ≥ 1.3 (K=10)")
    print("  607-universe:        cov_median ≥ 150 (hard)")
    print("  D+C retention ≥ 50%  (preferred — soft warn only)\n")

    cols = ["gate",
            "dc_n_k5", "dc_ratio_k5", "dc_false_k5",
            "dc_n_k10", "dc_ratio_k10", "dc_false_k10",
            "dc_retention_pct",
            "st_n_k10", "st_retention_pct", "st_ratio_k10", "st_false_k10",
            "cov_median", "cov_p10", "cov_unique"]
    print(summary_df[cols].to_string(index=False))

    print("\n--- Per-gate acceptance verdict ---")
    pass_rows: list[tuple[str, pd.Series, bool]] = []
    for _, r in summary_df.iterrows():
        ok, ret_warn, reasons = gate_passes(r)
        if r["gate"] in DIAGNOSTIC_GATES:
            tag = "DIAG"
        else:
            tag = "PASS" if ok else "FAIL"
        st_status = ("excluded" if (pd.notna(r["st_retention_pct"]) and r["st_retention_pct"] <= 30)
                     else "recovered" if (pd.notna(r["st_ratio_k10"]) and r["st_ratio_k10"] >= 1.3)
                     else "fails")
        print(f"  {r['gate']:<28} {tag:<5}  (stress={st_status})  → "
              f"{'; '.join(reasons) if reasons else 'all checks pass'}")
        if ok and r["gate"] not in DIAGNOSTIC_GATES:
            pass_rows.append((r["gate"], r, ret_warn))

    selected = pass_rows[0] if pass_rows else None

    # ---- Markdown report
    md: list[str] = []
    md.append("# Phase 1.3b — Efficiency/drift regime filter sweep on AL_F6")
    md.append("")
    md.append("OSCMTRX HWO-style gated early reversal engine — regime filter selection.")
    md.append("Phase 1.3 found tradeability features cannot discriminate stress from D+C")
    md.append("without collateral damage; this phase tests trend-efficiency features instead.")
    md.append("")
    md.append("## Executive verdict")
    md.append("")
    if selected:
        gname, grow, ret_warn = selected
        md.append(f"**SELECTED GATE: {gname}** — passes all hard acceptance checks.")
        if ret_warn:
            md.append("")
            md.append(f"⚠ D+C retention {grow['dc_retention_pct']:.0f}% is below the 50% preferred")
            md.append("threshold. Soft warning — operator must endorse the trade-off.")
        md.append("")
        md.append(f"`PRIMARY_TRADEABLE_AL = AL_F6 ∧ {gname}`")
        md.append("")
        md.append("**Financial rationale (operator review required).**")
        md.append("Stress underperformance comes from chop, not illiquidity. The selected")
        md.append("gate filters by trend efficiency / drift, which is the diagnostic-")
        md.append("matched axis. R5+ (LIQ) gates also keep illiquid tail off the production")
        md.append("scanner. Operator must endorse the simplest passing gate.")
    else:
        md.append("**NO R-GATE PASSES.** R0–R8 all fail at least one hard acceptance check.")
        md.append("")
        md.append("Per the original spec: AL_F6 is now treated as a **Core-style universe")
        md.append("candidate only** (BIST100-like whitelist), NOT a 607-universe scanner")
        md.append("candidate. Phase 4 splits into:")
        md.append("")
        md.append("- **A** Core-style large-universe scan (whitelist of D+C-class tickers)")
        md.append("- **B** Full-607 diagnostic-only run (no live deployment)")
    md.append("")
    md.append("## Acceptance criteria")
    md.append("")
    md.append("- D+C K=5 **and** K=10:  `ratio_med ≥ 1.7` ∧ `false_rev ≤ 35%`")
    md.append("- STRESS: `retention ≤ 30%` (excluded) **or** `ratio_med(K=10) ≥ 1.3` (recovered)")
    md.append("- 607-universe: `cov_median ≥ 150` (hard)")
    md.append("- D+C retention ≥ 50% (preferred, soft warning if violated)")
    md.append("- Financial rationale endorsed by operator")
    md.append("")
    md.append("## Comparison table")
    md.append("")
    md.append("| gate | dc_n_k5 | dc_R_k5 | dc_F_k5 | dc_n_k10 | dc_R_k10 | dc_F_k10 | dc_ret% | st_n_k10 | st_ret% | st_R_k10 | cov_med | cov_p10 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md.append(
            f"| {r['gate']} | "
            f"{int(r['dc_n_k5'])} | {r['dc_ratio_k5']:.2f} | {r['dc_false_k5']:.0f} | "
            f"{int(r['dc_n_k10'])} | {r['dc_ratio_k10']:.2f} | {r['dc_false_k10']:.0f} | "
            f"{r['dc_retention_pct']:.0f} | "
            f"{int(r['st_n_k10'])} | {r['st_retention_pct']:.0f} | {r['st_ratio_k10']:.2f} | "
            f"{r['cov_median']:.0f} | {r['cov_p10']:.0f} |"
        )
    md.append("")
    md.append("## Per-gate verdict")
    md.append("")
    md.append("| gate | verdict | stress | reasons |")
    md.append("|---|---|---|---|")
    for _, r in summary_df.iterrows():
        ok, ret_warn, reasons = gate_passes(r)
        if r["gate"] in DIAGNOSTIC_GATES:
            verdict = "DIAG"
        else:
            verdict = "PASS" if ok else "FAIL"
        st_status = ("excluded" if (pd.notna(r["st_retention_pct"]) and r["st_retention_pct"] <= 30)
                     else "recovered" if (pd.notna(r["st_ratio_k10"]) and r["st_ratio_k10"] >= 1.3)
                     else "fails")
        md.append(f"| {r['gate']} | {verdict} | {st_status} | "
                  f"{'; '.join(reasons) if reasons else 'all checks pass'} |")
    md.append("")
    md.append("## Cohort breakdown across horizons (K=5/10/20)")
    md.append("")
    md.append("`output/oscmatrix_phase1_3b_regime_filter_by_cohort.csv` — full breakdown.")
    md.append("Below: K=5 and K=10 only.")
    md.append("")
    md.append("| gate | cohort | K | n | retention% | ratio | false_rev% | p_lag | hw_lag |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    bc = by_cohort_df[by_cohort_df["K"].isin([5, 10])].copy()
    bc["cohort_ord"] = bc["cohort"].map(
        {"DISCOVERY": 0, "CORE": 1, "DISCOVERY+CORE": 2, "STRESS": 3}
    )
    bc["gate_ord"] = bc["gate"].map({g: i for i, (g, _) in enumerate(GATES_DEF)})
    bc = bc.sort_values(["gate_ord", "cohort_ord", "K"])
    for _, r in bc.iterrows():
        md.append(
            f"| {r['gate']} | {r['cohort']} | {int(r['K'])} | {int(r['n'])} | "
            f"{r['retention_pct']:.0f}% | {r['ratio_med']:.2f} | "
            f"{r['false_rev_pct']:.0f} | {r['p_lag_med']:+.1f} | {r['hw_lag_med']:+.1f} |"
        )
    md.append("")
    md.append("## 607-universe coverage")
    md.append("")
    md.append("| gate | median | p10 | p90 | unique tickers ever eligible |")
    md.append("|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md.append(
            f"| {r['gate']} | {r['cov_median']:.0f} | "
            f"{r['cov_p10']:.0f} | {r['cov_p90']:.0f} | {int(r['cov_unique'])} |"
        )
    md.append("")
    md.append("## Cohort feature diagnostic — ER60 / abs_drift60")
    md.append("")
    md.append("Per-ticker median ER60 and abs_drift60 (full 2023-01-02 → 2026-04-29 panel).")
    md.append("Confirms whether the new features actually separate stress from D+C.")
    md.append("")
    cohort_map = {tkr: cohort for cohort, ts in COHORTS.items() for tkr in ts}
    pano = panel.copy()
    pano["cohort"] = pano["ticker"].map(cohort_map)
    pano_c = pano[pano["cohort"].notna()].dropna(subset=["er60", "abs_drift60"])
    per_ticker = (
        pano_c.groupby(["cohort", "ticker"])
        [["er60", "abs_drift60"]].median()
    )
    md.append("| cohort | ticker | er60_med | abs_drift60_med |")
    md.append("|---|---|---|---|")
    for (cohort, ticker), r in per_ticker.iterrows():
        md.append(f"| {cohort} | {ticker} | {r['er60']:.3f} | {r['abs_drift60']:.3f} |")
    md.append("")
    md.append("Cohort-level aggregates (median across tickers and dates):")
    md.append("")
    cohort_stats = pano_c.groupby("cohort")[["er60", "abs_drift60"]].median().round(3)
    md.append("| cohort | er60_med | abs_drift60_med |")
    md.append("|---|---|---|")
    for cohort in ["DISCOVERY", "CORE", "STRESS"]:
        if cohort in cohort_stats.index:
            r = cohort_stats.loc[cohort]
            md.append(f"| {cohort} | {r['er60']:.3f} | {r['abs_drift60']:.3f} |")
    md.append("")
    md.append("## SAT side")
    md.append("")
    md.append("SAT short-entry stays disabled. SAT_E exit-probe is out of scope here —")
    md.append("schedule a separate `tools/oscmatrix_sat_exit_probe.py` if needed.")
    md.append("")
    md.append("## Next steps")
    md.append("")
    if selected:
        gname, grow, ret_warn = selected
        md.append(f"- Phase 4: 607-universe daily backtest of `AL_F6 ∧ {gname}`.")
        md.append("- Lock the regime feature recipe (er60 / abs_drift60 / median_turnover_60)")
        md.append("  and the gate definition into the production scanner.")
        if ret_warn:
            md.append(f"- Operator must endorse the {grow['dc_retention_pct']:.0f}% D+C retention")
            md.append("  trade-off (soft warning); reduces D+C signal density to gain stress")
            md.append("  exclusion.")
    else:
        md.append("- AL_F6 = Core-style universe candidate only. Phase 4 splits into:")
        md.append("  - **(A)** large-universe scan over a Core-class whitelist (BIST100-")
        md.append("    or large-cap-discovery-style); production candidate.")
        md.append("  - **(B)** full-607 diagnostic-only run; not live.")
        md.append("- Universe-filter research thread is closed. Re-opening requires either")
        md.append("  a new feature axis or a label/horizon redesign.")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\n  wrote {OUT_REPORT}")

    if selected:
        gname, _, ret_warn = selected
        print(f"\n  PRIMARY_TRADEABLE_AL = AL_F6 ∧ {gname}"
              + ("  (with retention-warn)" if ret_warn else ""))
    else:
        print("\n  No R-gate passes acceptance.")
        print("  → AL_F6 = Core-style universe candidate only. Phase 4 must split into")
        print("    (A) Core-style whitelist scan and (B) full-607 diagnostic-only run.")


if __name__ == "__main__":
    main()
