"""Phase 0.6.3 — close vs ohlc4 robustness probe (Yol 3).

Phase 0.6.2 fix swapped HyperWave source close→ohlc4 (RMSE 7.53→4.97 vs TV).
Side effect: D+C K=5/10 ratio dropped 1.72/2.20 → 1.32/1.31 (FAIL acceptance);
STRESS less broken (0.83 → 1.24/1.19). Cohort separation compressed.

Question this probe answers BEFORE choosing close vs ohlc4 in production:
  Is the close-source AL_F6 edge real (genuine alpha picked up by drift in HW)
  or a fidelity artifact (close-source HW lags TV → AL_F6 fires on bars that
  happen to land on lows for unrelated reasons)?

Method: build AL_F6 twice with all-else-equal:
  variant A:  close-source HyperWave   (Phase 1.x acceptance config)
  variant B:  ohlc4-source HyperWave   (Phase 0.6.2 fix, current production)
Money-flow side is identical (HW-source-independent). bear_ovf is the dead
stub (constant 50.0) → no effect. lower_zone differs by source via HW>50/<50.

Per-fire classification:
  BOTH         : same bar fired in both A and B
  CLOSE_ONLY   : fire only in A
  OHLC4_ONLY   : fire only in B
Also secondary view at ±1 and ±2 bar tolerance.

Per-cohort × K=5/10 metrics: n, p_lag_med, hw_lag_med, mae_atr_med,
mfe_atr_med, ratio_med (= MFE/MAE), false_rev_pct, mfe_capture_med.

Plus timing relation: close-before-ohlc4 1-2, close-after 1-2, exact.

Decision interpretation:
  CASE A  CLOSE_ONLY ratio strong + OHLC4_ONLY weak → close edge is real;
          ohlc4 fix breaks it. Action: revert to close, redesign separately.
  CASE B  BOTH overlap dominates and both have strong ratio → fidelity vs
          edge are aligned; ohlc4 keeps. Drop in metrics likely cohort
          re-mix (composition of fires shifted). Action: keep ohlc4.
  CASE C  OHLC4_ONLY strong + CLOSE_ONLY weak → ohlc4 picks better fires;
          close was lucky on small-set. Action: keep ohlc4, doc upgrade.
  CASE D  Neither cohort shows strong ratio in any class → AL_F6 itself
          weak, source choice is noise. Action: rule redesign needed.

Output:
  output/oscmatrix_phase0_6_3_per_fire.csv
  output/oscmatrix_phase0_6_3_class_metrics.csv
  output/oscmatrix_phase0_6_3_timing.csv
  output/oscmatrix_phase0_6_3_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix.components.money_flow import compute_money_flow  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

OUT_DIR = Path("output")
OUT_PER_FIRE = OUT_DIR / "oscmatrix_phase0_6_3_per_fire.csv"
OUT_CLASS_METRICS = OUT_DIR / "oscmatrix_phase0_6_3_class_metrics.csv"
OUT_TIMING = OUT_DIR / "oscmatrix_phase0_6_3_timing.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase0_6_3_report.md"

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

HORIZONS = [5, 10]
EVAL_W = 10
ROLL_N = 10
REFRACTORY_K = 10
ARM_RECENT_K = 10
OVF_RECENT_K = 5
FALSE_REV_ATR_THR = 0.5

HW_LEN = 7
HW_SIG_LEN = 3
OS_THRESHOLD = 20.0


# =========================================================================
# Inline HW (so we can compute it twice with different sources without
# touching production hyperwave.py)
# =========================================================================

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    rs = _rma(up, length) / _rma(dn, length).replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_hw_inline(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "close":
        src = df["close"]
    elif source == "ohlc4":
        src = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    else:
        raise ValueError(f"unknown source: {source!r}")
    rsi = _rsi(src, HW_LEN)
    hw = rsi.rolling(HW_SIG_LEN, min_periods=HW_SIG_LEN).mean()
    sig = hw.rolling(HW_SIG_LEN, min_periods=HW_SIG_LEN).mean()
    hw_prev = hw.shift(1)
    sig_prev = sig.shift(1)
    cross_up = (hw_prev <= sig_prev) & (hw > sig)
    return pd.DataFrame({"hyperwave": hw, "signal": sig, "hwo_up": cross_up},
                        index=df.index)


# =========================================================================
# AL_F6 helpers (mirror Phase 0.6 / 1.2)
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


def lower_zone_from_components(hw: pd.Series, mf: pd.Series) -> pd.Series:
    hw_bear = (hw < 50).astype(int)
    mf_bear = (mf < 50).astype(int)
    return hw_bear + mf_bear


def lag_in_window(bar_loc: int, series: pd.Series, window: int, side: str) -> int:
    n = len(series)
    lo = max(0, bar_loc - window)
    hi = min(n, bar_loc + window + 1)
    sub = series.iloc[lo:hi]
    if side == "low":
        return bar_loc - (lo + int(sub.values.argmin()))
    return bar_loc - (lo + int(sub.values.argmax()))


def forward_extremes(df: pd.DataFrame, K: int):
    fmh = pd.concat([df["high"].shift(-k) for k in range(1, K + 1)], axis=1).max(axis=1)
    fml = pd.concat([df["low"].shift(-k) for k in range(1, K + 1)], axis=1).min(axis=1)
    fwd_close = df["close"].shift(-K)
    return fmh, fml, fwd_close


# =========================================================================
# Per-ticker probe
# =========================================================================

def probe_ticker(ticker: str, csv_path: str) -> tuple[list[dict], dict]:
    """Returns (per-fire rows, timing-summary dict)."""
    if not Path(csv_path).exists():
        return [], {}
    tv = load_tv_csv(csv_path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())

    df = tv[["open", "high", "low", "close"]].astype(float)
    atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()

    # Money flow is HW-source-independent; compute once.
    mf = compute_money_flow(ohlcv, length=35, smooth=6).reindex(df.index)
    money_flow = mf["money_flow"]
    # bear_ovf is the dead stub (constant 50.0) → no effect.
    bear_ovf = pd.Series(False, index=df.index)

    # Two HW variants
    hw_close = compute_hw_inline(ohlcv, "close").reindex(df.index)
    hw_ohlc4 = compute_hw_inline(ohlcv, "ohlc4").reindex(df.index)

    lz_close = lower_zone_from_components(hw_close["hyperwave"], money_flow)
    lz_ohlc4 = lower_zone_from_components(hw_ohlc4["hyperwave"], money_flow)

    al_f6_close = build_al_f6(hw_close["hwo_up"].fillna(False).astype(bool),
                              hw_close["hyperwave"], lz_close, bear_ovf)
    al_f6_ohlc4 = build_al_f6(hw_ohlc4["hwo_up"].fillna(False).astype(bool),
                              hw_ohlc4["hyperwave"], lz_ohlc4, bear_ovf)

    idx_close = np.where(al_f6_close.values)[0]
    idx_ohlc4 = np.where(al_f6_ohlc4.values)[0]
    set_close = set(idx_close.tolist())
    set_ohlc4 = set(idx_ohlc4.tolist())

    # Same-bar exact classification
    both_exact = set_close & set_ohlc4
    close_only_strict = set_close - set_ohlc4
    ohlc4_only_strict = set_ohlc4 - set_close

    # ±1 / ±2 secondary classification (matched on either side counts)
    def has_match(i: int, others: np.ndarray, tol: int) -> bool:
        if others.size == 0:
            return False
        return bool(np.any(np.abs(others - i) <= tol))

    others_for_close = idx_ohlc4
    others_for_ohlc4 = idx_close

    # Timing relation per close fire
    timing = {"exact": 0, "close_before_ohlc4_1_2": 0, "close_after_ohlc4_1_2": 0,
              "unmatched_close": 0, "unmatched_ohlc4": 0,
              "n_close": int(len(idx_close)), "n_ohlc4": int(len(idx_ohlc4))}
    for i in idx_close:
        if others_for_close.size == 0:
            timing["unmatched_close"] += 1
            continue
        diffs = others_for_close - i
        nearest = diffs[np.argmin(np.abs(diffs))]
        if nearest == 0:
            timing["exact"] += 1
        elif -2 <= nearest <= -1:
            # ohlc4 fire is BEFORE close fire (close after ohlc4)
            timing["close_after_ohlc4_1_2"] += 1
        elif 1 <= nearest <= 2:
            # ohlc4 fire is AFTER close fire (close before ohlc4)
            timing["close_before_ohlc4_1_2"] += 1
        else:
            timing["unmatched_close"] += 1
    for i in idx_ohlc4:
        if others_for_ohlc4.size == 0:
            timing["unmatched_ohlc4"] += 1
            continue
        diffs = others_for_ohlc4 - i
        nearest = diffs[np.argmin(np.abs(diffs))]
        if abs(nearest) > 2:
            timing["unmatched_ohlc4"] += 1

    # Per-fire metrics rows. Per fire we attach pivot-proximity info under
    # whichever variant produced that fire (close or ohlc4) and tag the
    # source_class. For BOTH (same-bar) we record once (use close-variant
    # HW for hw_lag — they fire on same bar so it doesn't matter much).
    fires = []
    fwd = {K: forward_extremes(df, K) for K in HORIZONS}
    close_p = df["close"]
    low_p = df["low"]

    def add_fire(i: int, source_class: str, hw_used: pd.Series):
        a = atr.iloc[i]
        if pd.isna(a) or a == 0:
            return
        ci = close_p.iloc[i]
        for K in HORIZONS:
            fmh, fml, fwd_close = fwd[K]
            mfe_raw = fmh.iloc[i] - ci
            mae_raw = ci - fml.iloc[i]
            realized = (fwd_close.iloc[i] - ci) if pd.notna(fwd_close.iloc[i]) else np.nan
            if pd.isna(mfe_raw) or pd.isna(mae_raw):
                continue
            mfe_atr = float(mfe_raw / a)
            mae_atr = float(mae_raw / a)
            ratio = mfe_atr / max(mae_atr, 1e-3)
            false_rev = bool(realized < -FALSE_REV_ATR_THR * a) if pd.notna(realized) else None
            capture = (float(realized / mfe_raw) if (pd.notna(realized) and mfe_raw > 0)
                       else np.nan)
            fires.append({
                "ticker": ticker,
                "ts": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                "bar_idx": int(i),
                "K": K,
                "source_class": source_class,
                "p_lag": lag_in_window(i, low_p, EVAL_W, "low"),
                "hw_lag": lag_in_window(i, hw_used, EVAL_W, "low"),
                "mae_atr": mae_atr,
                "mfe_atr": mfe_atr,
                "ratio": ratio,
                "false_rev": false_rev,
                "mfe_capture": capture,
            })

    # Same-bar exact membership decides the strict class.
    for i in idx_close:
        if i in both_exact:
            add_fire(i, "BOTH", hw_close["hyperwave"])
        else:
            # secondary: does close have a near match in ohlc4?
            has_pm1 = has_match(i, others_for_close, 1)
            has_pm2 = has_match(i, others_for_close, 2)
            tag = "CLOSE_ONLY"
            if has_pm1:
                tag = "CLOSE_NEAR_OHLC4_PM1"
            elif has_pm2:
                tag = "CLOSE_NEAR_OHLC4_PM2"
            add_fire(i, tag, hw_close["hyperwave"])
    for i in idx_ohlc4:
        if i in both_exact:
            continue  # already added under close pass
        has_pm1 = has_match(i, others_for_ohlc4, 1)
        has_pm2 = has_match(i, others_for_ohlc4, 2)
        tag = "OHLC4_ONLY"
        if has_pm1:
            tag = "OHLC4_NEAR_CLOSE_PM1"
        elif has_pm2:
            tag = "OHLC4_NEAR_CLOSE_PM2"
        add_fire(i, tag, hw_ohlc4["hyperwave"])

    timing["both_exact"] = int(len(both_exact))
    timing["close_only_strict"] = int(len(close_only_strict))
    timing["ohlc4_only_strict"] = int(len(ohlc4_only_strict))

    return fires, timing


# =========================================================================
# Aggregate metrics
# =========================================================================

CLASS_BUCKETS = {
    "BOTH": ["BOTH"],
    "CLOSE_ONLY": ["CLOSE_ONLY", "CLOSE_NEAR_OHLC4_PM1", "CLOSE_NEAR_OHLC4_PM2"],
    "OHLC4_ONLY": ["OHLC4_ONLY", "OHLC4_NEAR_CLOSE_PM1", "OHLC4_NEAR_CLOSE_PM2"],
    "CLOSE_ONLY_STRICT": ["CLOSE_ONLY"],
    "OHLC4_ONLY_STRICT": ["OHLC4_ONLY"],
    "ALL_CLOSE": ["BOTH", "CLOSE_ONLY", "CLOSE_NEAR_OHLC4_PM1", "CLOSE_NEAR_OHLC4_PM2"],
    "ALL_OHLC4": ["BOTH", "OHLC4_ONLY", "OHLC4_NEAR_CLOSE_PM1", "OHLC4_NEAR_CLOSE_PM2"],
}


def class_metrics(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"n": 0, "ratio_med": np.nan, "false_rev_pct": np.nan,
                "p_lag_med": np.nan, "hw_lag_med": np.nan,
                "mae_atr_med": np.nan, "mfe_atr_med": np.nan,
                "mfe_capture_med": np.nan, "uniq_dates": 0}
    return {
        "n": int(len(sub)),
        "ratio_med": float(sub["ratio"].median()),
        "false_rev_pct": float(sub["false_rev"].mean() * 100)
            if sub["false_rev"].notna().any() else np.nan,
        "p_lag_med": float(sub["p_lag"].median()),
        "hw_lag_med": float(sub["hw_lag"].median()),
        "mae_atr_med": float(sub["mae_atr"].median()),
        "mfe_atr_med": float(sub["mfe_atr"].median()),
        "mfe_capture_med": float(sub["mfe_capture"].dropna().median())
            if sub["mfe_capture"].notna().any() else np.nan,
        "uniq_dates": int(sub["ts"].nunique()),
    }


def fmt(x, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if pd.notna(x) else "—"


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("=== Phase 0.6.3 — close vs ohlc4 robustness probe (Yol 3) ===\n")

    all_fires: list[dict] = []
    timing_rows: list[dict] = []
    cohort_of: dict[str, str] = {}

    for cohort, tickers in COHORTS.items():
        for tkr, path in tickers.items():
            cohort_of[tkr] = cohort
            fires, timing = probe_ticker(tkr, path)
            if not fires and not timing:
                print(f"  {cohort:<10} {tkr:<8} MISSING")
                continue
            for f in fires:
                f["cohort"] = cohort
            all_fires.extend(fires)
            timing["ticker"] = tkr
            timing["cohort"] = cohort
            timing_rows.append(timing)
            n_close = timing.get("n_close", 0)
            n_ohlc4 = timing.get("n_ohlc4", 0)
            both = timing.get("both_exact", 0)
            print(f"  {cohort:<10} {tkr:<8} close={n_close:>3} ohlc4={n_ohlc4:>3} "
                  f"both_exact={both:>3} close_only={timing['close_only_strict']:>3} "
                  f"ohlc4_only={timing['ohlc4_only_strict']:>3} "
                  f"exact={timing['exact']:>3} c_before={timing['close_before_ohlc4_1_2']:>2} "
                  f"c_after={timing['close_after_ohlc4_1_2']:>2}")

    fires_df = pd.DataFrame(all_fires)
    timing_df = pd.DataFrame(timing_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fires_df.to_csv(OUT_PER_FIRE, index=False)
    timing_df.to_csv(OUT_TIMING, index=False)
    print(f"\n  wrote {OUT_PER_FIRE}  ({len(fires_df)} fire rows)")
    print(f"  wrote {OUT_TIMING}     ({len(timing_df)} ticker rows)")

    # Aggregate per cohort × class × K
    cohorts = ["DISCOVERY", "CORE", "DISCOVERY+CORE", "STRESS", "ALL"]
    metric_rows = []
    for cohort_name in cohorts:
        if cohort_name == "DISCOVERY+CORE":
            sub_c = fires_df[fires_df["cohort"].isin(["DISCOVERY", "CORE"])]
        elif cohort_name == "ALL":
            sub_c = fires_df
        else:
            sub_c = fires_df[fires_df["cohort"] == cohort_name]
        for class_label, tags in CLASS_BUCKETS.items():
            sub_cls = sub_c[sub_c["source_class"].isin(tags)]
            for K in HORIZONS:
                m = class_metrics(sub_cls[sub_cls["K"] == K])
                metric_rows.append({
                    "cohort": cohort_name,
                    "class": class_label,
                    "K": K,
                    **m,
                })
    class_df = pd.DataFrame(metric_rows)
    class_df.to_csv(OUT_CLASS_METRICS, index=False)
    print(f"  wrote {OUT_CLASS_METRICS} ({len(class_df)} rows)")

    # Console summary: D+C and STRESS, K=5/10, key buckets
    print("\n=== D+C and STRESS, key buckets ===")
    print(class_df[
        class_df["cohort"].isin(["DISCOVERY+CORE", "STRESS"])
        & class_df["class"].isin(["BOTH", "ALL_CLOSE", "ALL_OHLC4",
                                    "CLOSE_ONLY", "OHLC4_ONLY",
                                    "CLOSE_ONLY_STRICT", "OHLC4_ONLY_STRICT"])
    ][["cohort", "class", "K", "n", "ratio_med", "false_rev_pct",
        "p_lag_med", "hw_lag_med", "mae_atr_med", "mfe_atr_med",
        "mfe_capture_med"]].to_string(index=False))

    # Timing aggregate
    print("\n=== Timing relation (totals across 24 tickers) ===")
    totals = {
        "n_close": int(timing_df["n_close"].sum()),
        "n_ohlc4": int(timing_df["n_ohlc4"].sum()),
        "both_exact": int(timing_df["both_exact"].sum()),
        "close_only_strict": int(timing_df["close_only_strict"].sum()),
        "ohlc4_only_strict": int(timing_df["ohlc4_only_strict"].sum()),
        "exact_match_count": int(timing_df["exact"].sum()),
        "close_before_ohlc4_1_2": int(timing_df["close_before_ohlc4_1_2"].sum()),
        "close_after_ohlc4_1_2": int(timing_df["close_after_ohlc4_1_2"].sum()),
        "unmatched_close": int(timing_df["unmatched_close"].sum()),
        "unmatched_ohlc4": int(timing_df["unmatched_ohlc4"].sum()),
    }
    for k, v in totals.items():
        print(f"  {k:<28} {v}")

    # Decision interpretation
    print("\n=== Decision interpretation ===")
    dc = class_df[class_df["cohort"] == "DISCOVERY+CORE"]

    def get(cls, k):
        row = dc[(dc["class"] == cls) & (dc["K"] == k)]
        return row.iloc[0] if not row.empty else None

    rows = {
        "BOTH_K10": get("BOTH", 10),
        "CLOSE_ONLY_K10": get("CLOSE_ONLY", 10),
        "OHLC4_ONLY_K10": get("OHLC4_ONLY", 10),
        "BOTH_K5": get("BOTH", 5),
        "CLOSE_ONLY_K5": get("CLOSE_ONLY", 5),
        "OHLC4_ONLY_K5": get("OHLC4_ONLY", 5),
    }

    def ratio(r):
        return float(r["ratio_med"]) if (r is not None and pd.notna(r["ratio_med"])) else np.nan

    bp_close_strong = (ratio(rows["CLOSE_ONLY_K10"]) >= 1.7) and (ratio(rows["OHLC4_ONLY_K10"]) < 1.5)
    bp_ohlc4_strong = (ratio(rows["OHLC4_ONLY_K10"]) >= 1.7) and (ratio(rows["CLOSE_ONLY_K10"]) < 1.5)
    bp_both_strong = (ratio(rows["BOTH_K10"]) >= 1.7)
    bp_none_strong = (ratio(rows["BOTH_K10"]) < 1.5
                      and ratio(rows["CLOSE_ONLY_K10"]) < 1.5
                      and ratio(rows["OHLC4_ONLY_K10"]) < 1.5)

    case = None
    if bp_close_strong and not bp_ohlc4_strong:
        case = "A — close-only carries the edge"
    elif bp_ohlc4_strong and not bp_close_strong:
        case = "C — ohlc4-only carries the edge"
    elif bp_both_strong:
        case = "B — overlap is what works; source choice is fidelity, not edge"
    elif bp_none_strong:
        case = "D — no strong cohort under any source; AL_F6 weak"
    else:
        case = "MIXED — none of A/B/C/D crisp; inspect class table"

    print(f"  D+C K=10 BOTH ratio       : {fmt(ratio(rows['BOTH_K10']))}  (n={int(rows['BOTH_K10']['n']) if rows['BOTH_K10'] is not None else 0})")
    print(f"  D+C K=10 CLOSE_ONLY ratio : {fmt(ratio(rows['CLOSE_ONLY_K10']))}  (n={int(rows['CLOSE_ONLY_K10']['n']) if rows['CLOSE_ONLY_K10'] is not None else 0})")
    print(f"  D+C K=10 OHLC4_ONLY ratio : {fmt(ratio(rows['OHLC4_ONLY_K10']))}  (n={int(rows['OHLC4_ONLY_K10']['n']) if rows['OHLC4_ONLY_K10'] is not None else 0})")
    print(f"  CASE: {case}")

    # ---- Markdown report ----
    md: list[str] = []
    md.append("# Phase 0.6.3 — close vs ohlc4 robustness probe (Yol 3)")
    md.append("")
    md.append("OSCMTRX HWO-style gated early reversal engine — pre-Phase-4 audit.")
    md.append("")
    md.append("Phase 0.6.2 fix (HyperWave source close→ohlc4) improved numeric fidelity")
    md.append("vs TV (RMSE 7.53→4.97, corr 0.967→0.976) but compressed D+C K=5/10 ratio")
    md.append("from 1.72/2.20 → 1.32/1.31. This probe tests whether the close-source edge")
    md.append("is real alpha or a fidelity artifact, by classifying each AL_F6 fire into")
    md.append("BOTH / CLOSE_ONLY / OHLC4_ONLY and measuring pivot-proximity per class.")
    md.append("")
    md.append(f"**CASE:** {case}")
    md.append("")
    md.append("## Acceptance reading")
    md.append("")
    md.append("- Strong: ratio_med ≥ 1.7 AND false_rev ≤ 35%")
    md.append("- Weak: ratio_med < 1.5")
    md.append("- A class is \"the edge\" if its ratio is strong while the alternative is weak.")
    md.append("")
    md.append("## Headline (Discovery+Core, K=10)")
    md.append("")
    md.append("| class | n | ratio_med | false_rev% | p_lag | hw_lag | mfe_capture |")
    md.append("|---|---|---|---|---|---|---|")
    for cls in ["BOTH", "CLOSE_ONLY", "OHLC4_ONLY", "ALL_CLOSE", "ALL_OHLC4"]:
        r = dc[(dc["class"] == cls) & (dc["K"] == 10)]
        if r.empty:
            continue
        r = r.iloc[0]
        md.append(
            f"| {cls} | {int(r['n'])} | {fmt(r['ratio_med'])} | "
            f"{fmt(r['false_rev_pct'], 1)} | {fmt(r['p_lag_med'], 1)} | "
            f"{fmt(r['hw_lag_med'], 1)} | {fmt(r['mfe_capture_med'])} |"
        )
    md.append("")
    md.append("## Full class table (all cohorts × class × K)")
    md.append("")
    md.append("| cohort | class | K | n | ratio_med | false_rev% | p_lag | hw_lag | mae_atr | mfe_atr | mfe_capture | uniq_dates |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cohort_ord = {"DISCOVERY": 0, "CORE": 1, "DISCOVERY+CORE": 2, "STRESS": 3, "ALL": 4}
    class_df["cohort_ord"] = class_df["cohort"].map(cohort_ord)
    class_df["class_ord"] = class_df["class"].map(
        {c: i for i, c in enumerate(CLASS_BUCKETS.keys())}
    )
    for _, r in class_df.sort_values(["cohort_ord", "class_ord", "K"]).iterrows():
        md.append(
            f"| {r['cohort']} | {r['class']} | {int(r['K'])} | {int(r['n'])} | "
            f"{fmt(r['ratio_med'])} | {fmt(r['false_rev_pct'], 1)} | "
            f"{fmt(r['p_lag_med'], 1)} | {fmt(r['hw_lag_med'], 1)} | "
            f"{fmt(r['mae_atr_med'])} | {fmt(r['mfe_atr_med'])} | "
            f"{fmt(r['mfe_capture_med'])} | {int(r['uniq_dates'])} |"
        )
    md.append("")
    md.append("## Timing relation (close vs ohlc4 fire ordering)")
    md.append("")
    md.append("| metric | total | %_of_close |")
    md.append("|---|---|---|")
    nc = max(totals["n_close"], 1)
    for k in ["n_close", "n_ohlc4", "both_exact", "close_only_strict",
              "ohlc4_only_strict", "exact_match_count",
              "close_before_ohlc4_1_2", "close_after_ohlc4_1_2",
              "unmatched_close", "unmatched_ohlc4"]:
        v = totals[k]
        pct = (v / nc * 100) if nc else 0
        md.append(f"| {k} | {v} | {pct:.1f}% |")
    md.append("")
    md.append("## Per-ticker timing breakdown")
    md.append("")
    md.append("| cohort | ticker | close | ohlc4 | both | c_only | o_only | exact | c_before | c_after |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    timing_df["cohort_ord"] = timing_df["cohort"].map({"DISCOVERY": 0, "CORE": 1, "STRESS": 2})
    for _, r in timing_df.sort_values(["cohort_ord", "ticker"]).iterrows():
        md.append(
            f"| {r['cohort']} | {r['ticker']} | {int(r['n_close'])} | "
            f"{int(r['n_ohlc4'])} | {int(r['both_exact'])} | "
            f"{int(r['close_only_strict'])} | {int(r['ohlc4_only_strict'])} | "
            f"{int(r['exact'])} | {int(r['close_before_ohlc4_1_2'])} | "
            f"{int(r['close_after_ohlc4_1_2'])} |"
        )
    md.append("")
    md.append("## Interpretation matrix")
    md.append("")
    md.append("- **CASE A** CLOSE_ONLY strong + OHLC4_ONLY weak → close edge is real alpha;")
    md.append("  ohlc4 fix loses fires that mattered. Action: revert to close, leave the")
    md.append("  fidelity gap as a known limitation, redesign as a separate workstream.")
    md.append("- **CASE B** BOTH dominates and is strong → fidelity and edge align; D+C")
    md.append("  ratio drop is composition (the fires lost are bad ones, those gained are")
    md.append("  bad ones, the kept ones are still good). Action: keep ohlc4, document.")
    md.append("- **CASE C** OHLC4_ONLY strong + CLOSE_ONLY weak → ohlc4 actually picks")
    md.append("  better fires; close was lucky on the small set. Action: keep ohlc4 as")
    md.append("  upgrade, redo Phase 1.x acceptance under ohlc4 ground truth.")
    md.append("- **CASE D** No class strong → AL_F6 itself is weak under either source;")
    md.append("  source choice is noise. Action: rule redesign needed; do NOT proceed to")
    md.append("  Phase 4 with current AL_F6 specification.")
    md.append("")
    md.append("## Constraints (do NOT do)")
    md.append("")
    md.append("- Do not change AL_F6 thresholds during this probe.")
    md.append("- Do not tune refractory / arm window / ovf recent / OS threshold.")
    md.append("- Do not rebuild the dead overflow channel; it is constant 50 in both runs")
    md.append("  and has zero effect on AL_F6 here.")
    md.append("- Do not start Phase 4 backtests until the close-vs-ohlc4 decision is made.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- `{OUT_PER_FIRE}` — per-fire metrics with source_class tag")
    md.append(f"- `{OUT_CLASS_METRICS}` — cohort × class × K aggregate")
    md.append(f"- `{OUT_TIMING}` — per-ticker timing relation")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\n  wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
