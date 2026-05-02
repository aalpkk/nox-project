"""Phase 0.6 — component fidelity audit (overflow / HWO / AL_F6).

Phase 4A is BLOCKED until this audit confirms compute_all reproduces the
component values used in Phase 1.2 acceptance. The risk: the AL_F6 that
passed Phase 1.2 was built with `OURS hwo_up + TV lower_zone + TV bear_ovf`.
On the 607-universe panel only OHLCV is available, so AL_F6 must use
`OURS hwo_up + OURS lower_zone + OURS bear_ovf`. If those swaps materially
change AL_F6, Phase 4 would test a *different* signal than Phase 1.x.

Audit scope (only AL_F6's dependencies):
  hwo_up / hwo_down                     event signal
  lower_zone / upper_zone               confluence meter (== TV's
                                        Lower/Upper Confluence Value)
  bullish_overflow / bearish_overflow   overflow channel
  hyperwave                             scalar (numeric corr only)

Out of scope (would open a separate rabbit hole):
  Money Flow drift, confluence meter precision beyond zone class,
  reversal arrows, signal channel fidelity.

AL_F6 reconstructions compared:
  TV-built  = OURS hwo_up ∧ TV lower_zone ∧ TV bear_ovf_recent5 ∧ ref10
              (this is the rule that passed Phase 1.2 acceptance)
  OURS-built = OURS hwo_up ∧ OURS lower_zone ∧ OURS bear_ovf_recent5 ∧ ref10
              (this is what 607-universe Phase 4 would actually run)

Decision tree:
  Case 1 — Phase 4A may proceed with OURS components.
    AL_F6 overlap (TV-built vs OURS-built, ±1 bar) ≥ 90%
    AND Phase 1.2 D+C acceptance preserved with OURS-only:
        K=5/10 ratio_med ≥ 1.7  ∧  false_rev_pct ≤ 35%
  Case 2 — compute_all fix REQUIRED before Phase 4.
    overlap < 90% OR D+C acceptance fails.
  Case 3 — HWO-only fix.
    HWO event F1 ≤ 0.85 alone explains AL_F6 drift; lower_zone /
    overflow disagreements are minor.
  Case 4 — overflow-only fix or note.
    HWO is fine; bear_ovf event divergence is large but AL_F6 overlap
    holds and metrics pass (because hw<30 or lower_zone==2 cover armed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

OUT_DIR = Path("output")
OUT_PER_TICKER_COMP = OUT_DIR / "oscmatrix_phase0_6_per_ticker_components.csv"
OUT_PER_TICKER_AL_F6 = OUT_DIR / "oscmatrix_phase0_6_per_ticker_al_f6.csv"
OUT_COHORT_METRICS = OUT_DIR / "oscmatrix_phase0_6_cohort_al_f6_metrics.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase0_6_audit_report.md"

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
EVAL_W = 10
ROLL_N = 10
REFRACTORY_K = 10
ARM_RECENT_K = 10
OVF_RECENT_K = 5
FALSE_REV_ATR_THR = 0.5

OVERLAP_TOLS = [0, 1, 2]


# =========================================================================
# Component-level audit primitives
# =========================================================================

def event_overlap(a_events: np.ndarray, b_events: np.ndarray, tol: int) -> dict:
    """Symmetric ±tol bar event overlap.

    Returns:
      precision : (ours hits with TV match) / ours total      [TV as reference]
      recall    : (TV hits with ours match) / TV total
      f1        : harmonic mean
      jaccard   : symmetric overlap fraction
    Treating `a` = TV (reference) and `b` = ours.
    """
    if a_events.size == 0 and b_events.size == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                "jaccard": 1.0, "n_a": 0, "n_b": 0}
    matched_b_to_a = sum(
        np.any(np.abs(a_events - i) <= tol) for i in b_events
    ) if a_events.size else 0
    matched_a_to_b = sum(
        np.any(np.abs(b_events - i) <= tol) for i in a_events
    ) if b_events.size else 0
    precision = (matched_b_to_a / b_events.size) if b_events.size else np.nan
    recall = (matched_a_to_b / a_events.size) if a_events.size else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else np.nan)
    jaccard = ((matched_a_to_b + matched_b_to_a)
               / (a_events.size + b_events.size))
    return {"precision": precision, "recall": recall, "f1": f1,
            "jaccard": jaccard, "n_a": int(a_events.size),
            "n_b": int(b_events.size)}


def numeric_corr(tv: pd.Series, ours: pd.Series) -> float:
    df = pd.DataFrame({"a": tv, "b": ours}).dropna()
    if len(df) < 10:
        return np.nan
    return float(df["a"].corr(df["b"]))


# =========================================================================
# Per-ticker audit
# =========================================================================

def audit_ticker(ticker: str, csv_path: str) -> tuple[dict, dict, list[dict]]:
    """Return (component_row, al_f6_row, fire_rows) for one ticker."""
    if not Path(csv_path).exists():
        return {}, {}, []
    tv = load_tv_csv(csv_path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")

    idx = tv.index
    ours_a = ours.reindex(idx)

    # ---- numeric corr: hyperwave ----
    hw_corr = numeric_corr(tv["HyperWave"], ours_a["hyperwave"]) if "HyperWave" in tv.columns else np.nan

    # ---- zone class agreement ----
    zone_pairs = []
    for tv_col, our_col in [("Lower Confluence Value", "lower_zone"),
                             ("Upper Confluence Value", "upper_zone")]:
        if tv_col in tv.columns and our_col in ours_a.columns:
            tv_z = tv[tv_col].astype("Int64").fillna(0).astype(int)
            our_z = ours_a[our_col].astype("Int64").fillna(0).astype(int)
            agree_pct = float((tv_z == our_z).mean() * 100)
            # event = zone == 2 (full confluence)
            tv_ev = np.where(tv_z.values == 2)[0]
            our_ev = np.where(our_z.values == 2)[0]
            ev = event_overlap(tv_ev, our_ev, tol=1)
            zone_pairs.append({
                "name": our_col,
                "agree_pct": agree_pct,
                "tv_zone2_n": int((tv_z == 2).sum()),
                "ours_zone2_n": int((our_z == 2).sum()),
                "zone2_f1_pm1": ev["f1"],
            })
        else:
            zone_pairs.append({"name": our_col, "agree_pct": np.nan,
                               "tv_zone2_n": 0, "ours_zone2_n": 0,
                               "zone2_f1_pm1": np.nan})

    # ---- overflow audit ----
    ovf_metrics = {}
    for tag, tv_col, our_col in [("bull", "Bullish Overflow", "bullish_overflow"),
                                   ("bear", "Bearish Overflow", "bearish_overflow")]:
        if tv_col in tv.columns and our_col in ours_a.columns:
            tv_v = tv[tv_col]
            our_v = ours_a[our_col]
            tv_active = (tv_v.fillna(50) != 50)
            our_active = (our_v.fillna(50) != 50)
            ovf_metrics[f"{tag}_tv_active_n"] = int(tv_active.sum())
            ovf_metrics[f"{tag}_ours_active_n"] = int(our_active.sum())
            # numeric correlation only on TV-active rows
            mask = tv_active | our_active
            if mask.sum() >= 10:
                ovf_metrics[f"{tag}_value_corr_active"] = float(
                    pd.DataFrame({"a": tv_v, "b": our_v})[mask]
                    .dropna().corr().iloc[0, 1] if mask.sum() else np.nan
                )
            else:
                ovf_metrics[f"{tag}_value_corr_active"] = np.nan
            tv_ev = np.where(tv_active.values)[0]
            our_ev = np.where(our_active.values)[0]
            for tol in OVERLAP_TOLS:
                eo = event_overlap(tv_ev, our_ev, tol=tol)
                ovf_metrics[f"{tag}_f1_pm{tol}"] = eo["f1"]
                ovf_metrics[f"{tag}_jaccard_pm{tol}"] = eo["jaccard"]

    # ---- HWO audit ----
    # OURS hwo_up returns FULL cross set (covers both small and big AL dots).
    # TV exposes them in two columns; combine for the right truth set.
    hwo_metrics = {}
    for tag, tv_cols, our_col in [
        ("hwo_up", ["HWO Up", "Oversold HWO Up"], "hwo_up"),
        ("hwo_down", ["HWO Down", "Overbought HWO Down"], "hwo_down"),
    ]:
        if our_col in ours_a.columns and any(c in tv.columns for c in tv_cols):
            tv_active = pd.Series(False, index=idx)
            for c in tv_cols:
                if c in tv.columns:
                    tv_active = tv_active | tv[c].notna()
            our_active = ours_a[our_col].fillna(False).astype(bool)
            tv_ev = np.where(tv_active.values)[0]
            our_ev = np.where(our_active.values)[0]
            hwo_metrics[f"{tag}_tv_n"] = int(len(tv_ev))
            hwo_metrics[f"{tag}_ours_n"] = int(len(our_ev))
            for tol in OVERLAP_TOLS:
                eo = event_overlap(tv_ev, our_ev, tol=tol)
                hwo_metrics[f"{tag}_precision_pm{tol}"] = eo["precision"]
                hwo_metrics[f"{tag}_recall_pm{tol}"] = eo["recall"]
                hwo_metrics[f"{tag}_f1_pm{tol}"] = eo["f1"]

    component_row = {
        "ticker": ticker,
        "n_bars": len(idx),
        "hw_corr": hw_corr,
        **{f"zone_{z['name']}_{k}": z[k] for z in zone_pairs
            for k in ["agree_pct", "tv_zone2_n", "ours_zone2_n", "zone2_f1_pm1"]},
        **ovf_metrics,
        **hwo_metrics,
    }

    # ---- AL_F6 reconstruction ----
    df = tv[["open", "high", "low", "close"]].astype(float)
    hw = ours_a["hyperwave"]
    atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()

    hwo_up_ours = ours_a["hwo_up"].fillna(False).astype(bool)

    # TV-built components (Phase 1.2 ground truth)
    if "Lower Confluence Value" in tv.columns:
        lower_zone_tv = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    else:
        lower_zone_tv = pd.Series(0, index=idx)
    if "Bearish Overflow" in tv.columns:
        bear_ovf_tv = (tv["Bearish Overflow"].fillna(50) != 50)
    else:
        bear_ovf_tv = pd.Series(False, index=idx)

    # OURS-built components (607 production)
    lower_zone_ours = ours_a["lower_zone"].astype("Int64").fillna(0).astype(int)
    bear_ovf_ours = (ours_a["bearish_overflow"].fillna(50) != 50)

    al_f6_tv = build_al_f6(hwo_up_ours, hw, lower_zone_tv, bear_ovf_tv)
    al_f6_ours = build_al_f6(hwo_up_ours, hw, lower_zone_ours, bear_ovf_ours)

    al_f6_tv_idx = np.where(al_f6_tv.values)[0]
    al_f6_ours_idx = np.where(al_f6_ours.values)[0]

    al_f6_row = {
        "ticker": ticker,
        "tv_n": int(len(al_f6_tv_idx)),
        "ours_n": int(len(al_f6_ours_idx)),
    }
    for tol in OVERLAP_TOLS:
        eo = event_overlap(al_f6_tv_idx, al_f6_ours_idx, tol=tol)
        al_f6_row[f"precision_pm{tol}"] = eo["precision"]
        al_f6_row[f"recall_pm{tol}"] = eo["recall"]
        al_f6_row[f"f1_pm{tol}"] = eo["f1"]
        al_f6_row[f"jaccard_pm{tol}"] = eo["jaccard"]

    # ---- pivot-proximity samples for Phase 1.2 metrics, OURS-only ----
    fire_rows: list[dict] = []
    if len(al_f6_ours_idx):
        fwd = {K: forward_extremes(df, K) for K in HORIZONS}
        close, low = df["close"], df["low"]
        for i in al_f6_ours_idx:
            a = atr.iloc[i]
            if not a or np.isnan(a):
                continue
            ci = close.iloc[i]
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
                false_rev = bool(realized < -FALSE_REV_ATR_THR * a) if pd.notna(realized) else None
                fire_rows.append({
                    "ticker": ticker, "K": K,
                    "p_lag": lag_in_window(i, low, EVAL_W, "low"),
                    "hw_lag": lag_in_window(i, hw, EVAL_W, "low"),
                    "mae_atr": mae_atr, "mfe_atr": mfe_atr, "ratio": ratio,
                    "false_rev": false_rev,
                })
    return component_row, al_f6_row, fire_rows


# =========================================================================
# AL_F6 + helpers (mirrors Phase 1.2)
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


def build_al_f6(hwo_up: pd.Series, hw: pd.Series,
                lower_zone: pd.Series, bear_ovf: pd.Series) -> pd.Series:
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


# =========================================================================
# Cohort summary (Phase 1.2 acceptance, OURS-only)
# =========================================================================

def cohort_metrics_from_fires(fires: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for K in HORIZONS:
        sub = fires[fires["K"] == K]
        if sub.empty:
            rows.append({"K": K, "n": 0, "ratio_med": np.nan,
                         "false_rev_pct": np.nan, "p_lag_med": np.nan,
                         "hw_lag_med": np.nan})
            continue
        rows.append({
            "K": K,
            "n": int(len(sub)),
            "ratio_med": float(sub["ratio"].median()),
            "false_rev_pct": float(sub["false_rev"].mean() * 100)
                if sub["false_rev"].notna().any() else np.nan,
            "p_lag_med": float(sub["p_lag"].median()),
            "hw_lag_med": float(sub["hw_lag"].median()),
        })
    return pd.DataFrame(rows)


# =========================================================================
# Main
# =========================================================================

def fmt_pct(x: float, dp: int = 1) -> str:
    return f"{x:.{dp}f}" if pd.notna(x) else "—"


def main() -> None:
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("=== Phase 0.6 — component fidelity audit ===")
    print("    (overflow / HWO / AL_F6 reconstruction)\n")

    component_rows: list[dict] = []
    al_f6_rows: list[dict] = []
    all_fires: list[dict] = []
    cohort_of: dict[str, str] = {}

    for cohort, tickers in COHORTS.items():
        for tkr, path in tickers.items():
            cohort_of[tkr] = cohort
            comp, al, fires = audit_ticker(tkr, path)
            if not comp:
                print(f"  {cohort:<10} {tkr:<8} MISSING")
                continue
            comp["cohort"] = cohort
            al["cohort"] = cohort
            for f in fires:
                f["cohort"] = cohort
            component_rows.append(comp)
            al_f6_rows.append(al)
            all_fires.extend(fires)
            print(f"  {cohort:<10} {tkr:<8} bars={comp['n_bars']:>4}  "
                  f"hwo_up TV={comp['hwo_up_tv_n']}/OURS={comp['hwo_up_ours_n']} "
                  f"f1±1={fmt_pct(comp['hwo_up_f1_pm1'], 2)}  "
                  f"bear_ovf TV={comp['bear_tv_active_n']}/OURS={comp['bear_ours_active_n']}  "
                  f"AL_F6 TV={al['tv_n']}/OURS={al['ours_n']} f1±1={fmt_pct(al['f1_pm1'], 2)}")

    components_df = pd.DataFrame(component_rows)
    al_f6_df = pd.DataFrame(al_f6_rows)
    fires_df = pd.DataFrame(all_fires)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    components_df.to_csv(OUT_PER_TICKER_COMP, index=False)
    al_f6_df.to_csv(OUT_PER_TICKER_AL_F6, index=False)
    print(f"\n  wrote {OUT_PER_TICKER_COMP}")
    print(f"  wrote {OUT_PER_TICKER_AL_F6}")

    # ---- aggregate component metrics ----
    print("\n=== Component-level aggregates (mean across 24 tickers) ===")
    comp_agg = {
        "hyperwave_corr_mean": float(components_df["hw_corr"].mean()),
        "lower_zone_agree_mean": float(components_df["zone_lower_zone_agree_pct"].mean()),
        "upper_zone_agree_mean": float(components_df["zone_upper_zone_agree_pct"].mean()),
        "lower_zone2_f1_mean": float(components_df["zone_lower_zone_zone2_f1_pm1"].mean()),
        "bull_ovf_active_TV_total": int(components_df["bull_tv_active_n"].sum()),
        "bull_ovf_active_OURS_total": int(components_df["bull_ours_active_n"].sum()),
        "bear_ovf_active_TV_total": int(components_df["bear_tv_active_n"].sum()),
        "bear_ovf_active_OURS_total": int(components_df["bear_ours_active_n"].sum()),
        "bull_ovf_f1_pm1_mean": float(components_df["bull_f1_pm1"].mean()),
        "bear_ovf_f1_pm1_mean": float(components_df["bear_f1_pm1"].mean()),
        "hwo_up_TV_total": int(components_df["hwo_up_tv_n"].sum()),
        "hwo_up_OURS_total": int(components_df["hwo_up_ours_n"].sum()),
        "hwo_up_f1_pm1_mean": float(components_df["hwo_up_f1_pm1"].mean()),
        "hwo_up_precision_pm1_mean": float(components_df["hwo_up_precision_pm1"].mean()),
        "hwo_up_recall_pm1_mean": float(components_df["hwo_up_recall_pm1"].mean()),
        "hwo_down_TV_total": int(components_df["hwo_down_tv_n"].sum()),
        "hwo_down_OURS_total": int(components_df["hwo_down_ours_n"].sum()),
        "hwo_down_f1_pm1_mean": float(components_df["hwo_down_f1_pm1"].mean()),
    }
    for k, v in comp_agg.items():
        print(f"  {k:<35} {v}")

    # ---- AL_F6 aggregate ----
    print("\n=== AL_F6 reconstruction overlap (TV-built vs OURS-built) ===")
    al_f6_agg = {
        "tv_n_total": int(al_f6_df["tv_n"].sum()),
        "ours_n_total": int(al_f6_df["ours_n"].sum()),
        "f1_pm1_mean": float(al_f6_df["f1_pm1"].mean()),
        "f1_pm2_mean": float(al_f6_df["f1_pm2"].mean()),
        "jaccard_pm1_mean": float(al_f6_df["jaccard_pm1"].mean()),
        "jaccard_pm2_mean": float(al_f6_df["jaccard_pm2"].mean()),
    }
    for k, v in al_f6_agg.items():
        print(f"  {k:<25} {v}")

    # ---- Phase 1.2 acceptance metrics OURS-only ----
    print("\n=== Phase 1.2 acceptance metrics (OURS-built AL_F6 only) ===")
    cohort_metric_rows = []
    for cohort_name in ["DISCOVERY", "CORE", "STRESS", "DISCOVERY+CORE"]:
        if cohort_name == "DISCOVERY+CORE":
            sub = fires_df[fires_df["cohort"].isin(["DISCOVERY", "CORE"])]
        else:
            sub = fires_df[fires_df["cohort"] == cohort_name]
        m = cohort_metrics_from_fires(sub)
        for _, r in m.iterrows():
            r2 = r.to_dict()
            r2["cohort"] = cohort_name
            cohort_metric_rows.append(r2)
    cohort_metrics_df = pd.DataFrame(cohort_metric_rows)
    cohort_metrics_df.to_csv(OUT_COHORT_METRICS, index=False)
    print(f"  wrote {OUT_COHORT_METRICS}\n")

    cm = cohort_metrics_df.copy()
    cm["cohort_ord"] = cm["cohort"].map(
        {"DISCOVERY": 0, "CORE": 1, "DISCOVERY+CORE": 2, "STRESS": 3}
    )
    cm = cm.sort_values(["cohort_ord", "K"]).drop(columns=["cohort_ord"])
    print(cm.to_string(index=False))

    # ---- Decision ----
    print("\n=== Decision ===")
    f1_pm1_overall = al_f6_agg["f1_pm1_mean"]
    al_f6_overlap_ok = f1_pm1_overall >= 0.90

    dc = cohort_metrics_df[cohort_metrics_df["cohort"] == "DISCOVERY+CORE"]
    dc_k5 = dc[dc["K"] == 5].iloc[0] if not dc[dc["K"] == 5].empty else None
    dc_k10 = dc[dc["K"] == 10].iloc[0] if not dc[dc["K"] == 10].empty else None

    def metric_pass(row, ratio_min=1.7, false_max=35.0) -> tuple[bool, str]:
        if row is None:
            return False, "no row"
        ok_ratio = pd.notna(row["ratio_med"]) and row["ratio_med"] >= ratio_min
        ok_false = pd.notna(row["false_rev_pct"]) and row["false_rev_pct"] <= false_max
        return ok_ratio and ok_false, (
            f"ratio={row['ratio_med']:.2f}{' ✓' if ok_ratio else ' ✗'}, "
            f"false_rev={row['false_rev_pct']:.0f}%{' ✓' if ok_false else ' ✗'}"
        )

    k5_ok, k5_str = metric_pass(dc_k5)
    k10_ok, k10_str = metric_pass(dc_k10)
    metrics_ok = k5_ok and k10_ok

    # diagnose dominant component drift
    hwo_f1 = comp_agg["hwo_up_f1_pm1_mean"]
    bear_ovf_f1 = comp_agg["bear_ovf_f1_pm1_mean"]
    bear_active_ratio = (
        comp_agg["bear_ovf_active_OURS_total"]
        / max(comp_agg["bear_ovf_active_TV_total"], 1)
    )

    case = None
    if al_f6_overlap_ok and metrics_ok:
        case = 1
    elif hwo_f1 <= 0.85 and not metrics_ok:
        case = 3
    elif (pd.isna(bear_ovf_f1) or bear_ovf_f1 < 0.5) and al_f6_overlap_ok and metrics_ok:
        case = 4
    else:
        case = 2

    print(f"  AL_F6 overlap ±1 (mean F1): {f1_pm1_overall:.3f}  "
          f"({'≥0.90 ✓' if al_f6_overlap_ok else '<0.90 ✗'})")
    print(f"  D+C K=5  acceptance: {k5_str}")
    print(f"  D+C K=10 acceptance: {k10_str}")
    print(f"  HWO Up F1 ±1: {hwo_f1:.3f}  bear_ovf F1 ±1: {bear_ovf_f1:.3f}  "
          f"bear_active OURS/TV ratio: {bear_active_ratio:.2f}")
    print(f"\n  → CASE {case}")
    case_text = {
        1: "Phase 4A may proceed with OURS components.",
        2: "compute_all fix REQUIRED before Phase 4. AL_F6 overlap or D+C metrics fail.",
        3: "HWO-only fix required. HWO event F1 ≤ 0.85 alone explains the drift.",
        4: "Overflow-only divergence; AL_F6 overlap and metrics still pass. "
           "Note as known limitation; Phase 4A may proceed.",
    }
    print(f"     {case_text[case]}")

    # ---- Markdown report ----
    md: list[str] = []
    md.append("# Phase 0.6 — Component fidelity audit (overflow / HWO / AL_F6)")
    md.append("")
    md.append("Pre-Phase-4 audit. Confirms whether `compute_all` reproduces the components")
    md.append("AL_F6 was accepted on in Phase 1.2.")
    md.append("")
    md.append(f"**CASE {case}** — {case_text[case]}")
    md.append("")
    md.append("## Decision criteria")
    md.append("")
    md.append("- **Case 1**  AL_F6 overlap ±1 (mean F1) ≥ 0.90 **and** D+C K=5/10 acceptance preserved (ratio ≥ 1.7, false_rev ≤ 35%).")
    md.append("- **Case 2**  Acceptance fails OR overlap < 0.90 with no isolated dominant cause.")
    md.append("- **Case 3**  HWO Up F1 ≤ 0.85 alone explains the drift.")
    md.append("- **Case 4**  Overflow event divergence is large but AL_F6 overlap and acceptance pass (hw<30 / lower_zone==2 cover armed).")
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append(f"- AL_F6 overlap ±1 (mean F1 across 24 tickers):  **{f1_pm1_overall:.3f}**")
    md.append(f"- AL_F6 overlap ±2 (mean F1):  {al_f6_agg['f1_pm2_mean']:.3f}")
    md.append(f"- D+C K=5  acceptance: {k5_str}")
    md.append(f"- D+C K=10 acceptance: {k10_str}")
    md.append("")
    md.append("## Component-level aggregates")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| HyperWave numeric corr (mean) | {comp_agg['hyperwave_corr_mean']:.4f} |")
    md.append(f"| lower_zone class agreement % (mean) | {comp_agg['lower_zone_agree_mean']:.1f}% |")
    md.append(f"| upper_zone class agreement % (mean) | {comp_agg['upper_zone_agree_mean']:.1f}% |")
    md.append(f"| lower_zone == 2 event F1 ±1 (mean) | {comp_agg['lower_zone2_f1_mean']:.3f} |")
    md.append(f"| HWO Up TV/OURS total fires | {comp_agg['hwo_up_TV_total']} / {comp_agg['hwo_up_OURS_total']} |")
    md.append(f"| HWO Up F1 ±1 / precision / recall (mean) | {comp_agg['hwo_up_f1_pm1_mean']:.3f} / {comp_agg['hwo_up_precision_pm1_mean']:.3f} / {comp_agg['hwo_up_recall_pm1_mean']:.3f} |")
    md.append(f"| HWO Down TV/OURS total fires | {comp_agg['hwo_down_TV_total']} / {comp_agg['hwo_down_OURS_total']} |")
    md.append(f"| HWO Down F1 ±1 (mean) | {comp_agg['hwo_down_f1_pm1_mean']:.3f} |")
    md.append(f"| Bullish overflow active TV/OURS total | {comp_agg['bull_ovf_active_TV_total']} / {comp_agg['bull_ovf_active_OURS_total']} |")
    md.append(f"| Bearish overflow active TV/OURS total | {comp_agg['bear_ovf_active_TV_total']} / {comp_agg['bear_ovf_active_OURS_total']} |")
    md.append(f"| Bull overflow F1 ±1 (mean) | {comp_agg['bull_ovf_f1_pm1_mean']:.3f} |")
    md.append(f"| Bear overflow F1 ±1 (mean) | {comp_agg['bear_ovf_f1_pm1_mean']:.3f} |")
    md.append("")
    md.append("## Per-ticker component fidelity")
    md.append("")
    md.append("| cohort | ticker | bars | hw_corr | lz_agree% | uz_agree% | HWO Up TV/OURS F1±1 | HWO Down TV/OURS F1±1 | bull_ovf TV/OURS F1±1 | bear_ovf TV/OURS F1±1 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    cohort_ord = {"DISCOVERY": 0, "CORE": 1, "STRESS": 2}
    components_df["cohort_ord"] = components_df["cohort"].map(cohort_ord)
    for _, r in components_df.sort_values(["cohort_ord", "ticker"]).iterrows():
        md.append(
            f"| {r['cohort']} | {r['ticker']} | {int(r['n_bars'])} | "
            f"{r['hw_corr']:.3f} | {r['zone_lower_zone_agree_pct']:.0f}% | "
            f"{r['zone_upper_zone_agree_pct']:.0f}% | "
            f"{int(r['hwo_up_tv_n'])}/{int(r['hwo_up_ours_n'])} {r['hwo_up_f1_pm1']:.2f} | "
            f"{int(r['hwo_down_tv_n'])}/{int(r['hwo_down_ours_n'])} {r['hwo_down_f1_pm1']:.2f} | "
            f"{int(r['bull_tv_active_n'])}/{int(r['bull_ours_active_n'])} {fmt_pct(r['bull_f1_pm1'], 2)} | "
            f"{int(r['bear_tv_active_n'])}/{int(r['bear_ours_active_n'])} {fmt_pct(r['bear_f1_pm1'], 2)} |"
        )
    md.append("")
    md.append("## AL_F6 reconstruction (TV-built vs OURS-built) per ticker")
    md.append("")
    md.append("| cohort | ticker | TV n | OURS n | F1 ±0 | F1 ±1 | F1 ±2 |")
    md.append("|---|---|---|---|---|---|---|")
    al_f6_df["cohort_ord"] = al_f6_df["cohort"].map(cohort_ord)
    for _, r in al_f6_df.sort_values(["cohort_ord", "ticker"]).iterrows():
        md.append(
            f"| {r['cohort']} | {r['ticker']} | {int(r['tv_n'])} | {int(r['ours_n'])} | "
            f"{fmt_pct(r['f1_pm0'], 2)} | {fmt_pct(r['f1_pm1'], 2)} | "
            f"{fmt_pct(r['f1_pm2'], 2)} |"
        )
    md.append("")
    md.append("## Phase 1.2 acceptance — OURS-built AL_F6 only")
    md.append("")
    md.append("Re-run of Phase 1.2 acceptance metrics, but with AL_F6 constructed using")
    md.append("`compute_all` outputs only (no TV CSV component reads).")
    md.append("")
    md.append("| cohort | K | n | ratio_med | false_rev% | p_lag | hw_lag |")
    md.append("|---|---|---|---|---|---|---|")
    cm2 = cohort_metrics_df.copy()
    cm2["cohort_ord"] = cm2["cohort"].map(
        {"DISCOVERY": 0, "CORE": 1, "DISCOVERY+CORE": 2, "STRESS": 3}
    )
    for _, r in cm2.sort_values(["cohort_ord", "K"]).iterrows():
        md.append(
            f"| {r['cohort']} | {int(r['K'])} | {int(r['n'])} | "
            f"{r['ratio_med']:.2f} | {fmt_pct(r['false_rev_pct'], 1)} | "
            f"{fmt_pct(r['p_lag_med'], 1)} | {fmt_pct(r['hw_lag_med'], 1)} |"
        )
    md.append("")
    md.append("## Diagnosis")
    md.append("")
    md.append("- HyperWave numeric correlation is the floor of the audit — if it's not")
    md.append("  ≈1.0, every downstream component is suspect.")
    md.append("- HWO Up over-firing with high recall but low precision = our event rule")
    md.append("  triggers on more bars than TV; tightening cross-up condition would")
    md.append("  reduce false fires.")
    md.append("- Overflow active count near-zero in OURS while TV has many active bars")
    md.append("  indicates the overflow channel is not implemented or stuck at 50.")
    md.append("- AL_F6 overlap can survive HWO/overflow drift if `hw<30 ∨ lower_zone==2`")
    md.append("  already covers the armed path on most fires.")
    md.append("")
    md.append("## Next steps")
    md.append("")
    if case == 1:
        md.append("- Phase 4A unblocked. Proceed with OURS-only AL_F6 on the Core-class")
        md.append("  whitelist universe (`median_turnover_60 ≥ q60` ∧ `history ≥ 500`).")
        md.append("- Document component drift as a known limitation in Phase 4 reports.")
    elif case == 2:
        md.append("- Phase 4A blocked. Fix `compute_all`: rebuild `bearish_overflow` /")
        md.append("  `bullish_overflow` per LuxAlgo definition, audit `hwo_up` event rule.")
        md.append("- After fix, re-run this audit; require Case 1 before Phase 4A start.")
        md.append("- Once Phase 0.6 reaches Case 1, also re-run Phase 1.2 metrics with")
        md.append("  OURS-only components to refresh acceptance — small-set numbers may shift.")
    elif case == 3:
        md.append("- Fix `hwo_up` event rule (cross-up condition + zone gate). Overflow")
        md.append("  fix can wait. Re-run audit, require Case 1.")
    elif case == 4:
        md.append("- Phase 4A may proceed. Document the dead overflow channel as a known")
        md.append("  limitation. Optional follow-up: rebuild overflow component for full")
        md.append("  parity with TV and re-run audit later.")
    md.append("")
    md.append("## Out of scope (do NOT open)")
    md.append("- Money Flow drift, confluence meter precision beyond zone class, reversal")
    md.append("  arrows, signal channel fidelity. These are separate rabbit holes.")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\n  wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
