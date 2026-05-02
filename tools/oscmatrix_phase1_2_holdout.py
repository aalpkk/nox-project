"""Phase 1.2 — small-set hold-out validation. NO new grid, NO param changes.

Re-applies the four shortlisted Phase 1.1 rules (AL_F4, AL_F6, SAT_B, SAT_E)
to three disjoint cohorts:

  DISCOVERY  : 4 tickers used for component fit + rule selection
  CORE       : 11 tickers (sector-diverse, mature large-caps)
  STRESS     : 9 tickers (volatile/mid-cap/spec)

If AL_F6 holds MFE/MAE>1.5, false_rev<35%, p_lag in [0,3], hw_lag in [1,2]
and is K=5/K=10 consistent in CORE and STRESS, AL_F6 is treated as
small-set final and Phase 2 may begin. Otherwise, AL_F6 = overfit candidate
and AL_F4 (or a broader HWO-rule) is reconsidered.

SAT short-side stays disabled. SAT_E carried only as a weak exit-candidate
benchmark.

Missing CSVs (informational only): KOZAL (Core), LINK (Stress).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

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
    # KOZAL: CSV not present
]}

STRESS = {t: f"{CSV_DIR_B}/BIST_{t}, 1D.csv" for t in [
    "KAREL", "PARSN", "BRISA", "GOODY", "INDES",
    "NETAS", "KONTR", "KLSER", "MAVI",
    # LINK: CSV not present
]}

COHORTS = {"DISCOVERY": DISCOVERY, "CORE": CORE, "STRESS": STRESS}

HORIZONS = [5, 10, 20]
EVAL_W = 10
ROLL_N = 10
REFRACTORY_K = 10
ARM_RECENT_K = 10
OVF_RECENT_K = 5
FALSE_REV_ATR_THR = 0.5
SAT_E_HIGH_DIST_CAP_ATR = 0.75


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


def build_inputs(tv: pd.DataFrame, ours: pd.DataFrame, df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    inp = pd.DataFrame(index=tv.index)
    hw = ours["hyperwave"].reindex(tv.index)
    inp["hw"] = hw
    inp["hw_lt_30"] = hw < 30
    inp["hw_gt_70"] = hw > 70
    inp["hwo_up"] = ours["hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    inp["hwo_down"] = ours["hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    inp["ob_hwo_down"] = ours["ob_hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    if "Lower Confluence Value" in tv.columns:
        inp["lower_zone"] = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
        inp["upper_zone"] = tv["Upper Confluence Value"].astype("Int64").fillna(0).astype(int)
    else:
        inp["lower_zone"] = 0
        inp["upper_zone"] = 0
    if "Bullish Overflow" in tv.columns:
        bull_ovf = (tv["Bullish Overflow"].fillna(50) != 50)
        bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50)
    else:
        bull_ovf = pd.Series(False, index=tv.index)
        bear_ovf = pd.Series(False, index=tv.index)
    inp["bull_ovf_recent"] = recent_any(bull_ovf, OVF_RECENT_K)
    inp["bear_ovf_recent"] = recent_any(bear_ovf, OVF_RECENT_K)
    roll_high = df["high"].rolling(ROLL_N, min_periods=1).max()
    inp["roll_high_dist_atr"] = (roll_high - df["close"]) / atr
    return inp


def build_rules(inp: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    al_armed = inp["hw_lt_30"] | (inp["lower_zone"] == 2) | inp["bear_ovf_recent"]
    sat_armed = inp["hw_gt_70"] | (inp["upper_zone"] == 2) | inp["bull_ovf_recent"]
    al_armed_recent = recent_any(al_armed, ARM_RECENT_K)
    sat_armed_recent = recent_any(sat_armed, ARM_RECENT_K)

    al_f4 = inp["hwo_up"] & (inp["lower_zone"] >= 1)
    al_f3_raw = inp["hwo_up"] & al_armed_recent
    al_f6 = apply_refractory(al_f3_raw, REFRACTORY_K)

    sat_b = inp["hwo_down"] & inp["bull_ovf_recent"]
    sat_e_pre = inp["hwo_down"] & sat_armed_recent
    high_dist_ok = inp["roll_high_dist_atr"].fillna(np.inf) <= SAT_E_HIGH_DIST_CAP_ATR
    sat_e = sat_e_pre & high_dist_ok

    return {
        "AL_F4": ("AL", al_f4),
        "AL_F6": ("AL", al_f6),
        "SAT_B": ("SAT", sat_b),
        "SAT_E": ("SAT", sat_e),
    }


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


def collect_samples(side: str, fires: pd.Series, df: pd.DataFrame, hw: pd.Series, atr: pd.Series,
                    inp: pd.DataFrame, K: int) -> list[dict]:
    locs = np.where(fires.values)[0]
    if len(locs) == 0:
        return []
    close, high, low = df["close"], df["high"], df["low"]
    fmh, fml, fwd_close = forward_extremes(df, K)
    roll_low = low.rolling(ROLL_N, min_periods=1).min()
    samples = []
    for i in locs:
        a = atr.iloc[i]
        if not a or np.isnan(a):
            continue
        ci = close.iloc[i]
        if side == "AL":
            p_lag = lag_in_window(i, low, EVAL_W, "low")
            hw_lag = lag_in_window(i, hw, EVAL_W, "low")
            roll_dist = float((ci - roll_low.iloc[i]) / a)
            mfe_raw = fmh.iloc[i] - ci
            mae_raw = ci - fml.iloc[i]
            realized = fwd_close.iloc[i] - ci if pd.notna(fwd_close.iloc[i]) else np.nan
        else:
            p_lag = lag_in_window(i, high, EVAL_W, "high")
            hw_lag = lag_in_window(i, hw, EVAL_W, "high")
            roll_dist = float(inp["roll_high_dist_atr"].iloc[i]) if pd.notna(inp["roll_high_dist_atr"].iloc[i]) else np.nan
            mfe_raw = ci - fml.iloc[i]
            mae_raw = fmh.iloc[i] - ci
            realized = ci - fwd_close.iloc[i] if pd.notna(fwd_close.iloc[i]) else np.nan
        if pd.isna(mfe_raw) or pd.isna(mae_raw):
            continue
        mfe_atr = float(mfe_raw / a)
        mae_atr = float(mae_raw / a)
        ratio = mfe_atr / max(mae_atr, 1e-3)
        cap = (float(realized) / float(mfe_raw)) if pd.notna(realized) and mfe_raw > 0 else np.nan
        false_rev = bool(realized < -FALSE_REV_ATR_THR * a) if pd.notna(realized) else None
        samples.append({
            "p_lag": p_lag, "hw_lag": hw_lag, "roll_dist": roll_dist,
            "mae_atr": mae_atr, "mfe_atr": mfe_atr, "ratio": ratio,
            "capture": cap, "false_rev": false_rev,
        })
    return samples


def summarize_samples(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0, "p_lag_med": np.nan, "hw_lag_med": np.nan, "roll_dist_med": np.nan,
                "mae_med": np.nan, "mfe_med": np.nan, "ratio_med": np.nan,
                "capture_med": np.nan, "false_rev_pct": np.nan}
    df = pd.DataFrame(samples)
    return {
        "n": len(df),
        "p_lag_med": float(df["p_lag"].median()),
        "hw_lag_med": float(df["hw_lag"].median()),
        "roll_dist_med": float(df["roll_dist"].median()) if df["roll_dist"].notna().any() else np.nan,
        "mae_med": float(df["mae_atr"].median()),
        "mfe_med": float(df["mfe_atr"].median()),
        "ratio_med": float(df["ratio"].median()),
        "capture_med": float(df["capture"].median()) if df["capture"].notna().any() else np.nan,
        "false_rev_pct": float(df["false_rev"].mean() * 100) if df["false_rev"].notna().any() else np.nan,
    }


def run_cohort(name: str, ticker_paths: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    pooled: dict[tuple[str, int], list[dict]] = {}
    rule_side: dict[str, str] = {}
    coverage: dict[str, int] = {}
    for ticker, path in ticker_paths.items():
        if not Path(path).exists():
            print(f"  [{name}] missing CSV: {ticker} → {path}")
            continue
        try:
            tv = load_tv_csv(path).set_index("ts")
        except Exception as exc:
            print(f"  [{name}] FAILED to load {ticker}: {exc}")
            continue
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        df = tv[["open", "high", "low", "close"]].astype(float)
        hw = ours["hyperwave"].reindex(tv.index)
        atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
        inp = build_inputs(tv, ours, df, atr)
        rules = build_rules(inp)
        coverage[ticker] = len(tv)
        for rname, (side, fires) in rules.items():
            rule_side[rname] = side
            for K in HORIZONS:
                pooled.setdefault((rname, K), []).extend(
                    collect_samples(side, fires, df, hw, atr, inp, K)
                )
    rows = []
    for (rname, K), samples in pooled.items():
        s = summarize_samples(samples)
        rows.append({"cohort": name, "rule": rname, "side": rule_side[rname], "K": K, **s})
    return pd.DataFrame(rows), coverage


def main() -> None:
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("=== Phase 1.2 — small-set hold-out validation ===")
    print(f"  Rules under test: AL_F4, AL_F6, SAT_B, SAT_E (no parameter changes)")
    print(f"  Horizons: {HORIZONS}\n")

    all_rows = []
    for name, paths in COHORTS.items():
        print(f"--- Loading cohort {name} ({len(paths)} tickers) ---")
        df, cov = run_cohort(name, paths)
        present = [t for t in paths if t in cov]
        missing = [t for t in paths if t not in cov]
        print(f"  loaded: {present}")
        if missing:
            print(f"  missing: {missing}")
        all_rows.append(df)
        print()

    rep = pd.concat(all_rows, ignore_index=True)

    cols = ["cohort", "rule", "K", "n", "p_lag_med", "hw_lag_med", "roll_dist_med",
            "mae_med", "mfe_med", "ratio_med", "capture_med", "false_rev_pct"]

    for side in ["AL", "SAT"]:
        print(f"\n--- {side} side, all cohorts ---")
        sub = rep[rep["side"] == side].copy()
        sub["cohort_ord"] = sub["cohort"].map({"DISCOVERY": 0, "CORE": 1, "STRESS": 2})
        sub = sub.sort_values(["rule", "cohort_ord", "K"]).drop(columns=["cohort_ord"])
        print(sub[cols].to_string(index=False))

    # Acceptance gate for AL_F6 — primary candidate
    print("\n\n=== AL_F6 acceptance gate ===")
    print("  Per cohort × K, check:")
    print("    G1: ratio_med ≥ 1.5            G2: false_rev_pct ≤ 35")
    print("    G3: 0 ≤ p_lag_med ≤ 3          G4: 1 ≤ hw_lag_med ≤ 2")
    print("    G5: K=5 and K=10 both pass G1+G2 (consistency)\n")

    al_f6 = rep[rep["rule"] == "AL_F6"].copy()
    for cohort in ["DISCOVERY", "CORE", "STRESS"]:
        sub = al_f6[al_f6["cohort"] == cohort].sort_values("K")
        if sub.empty:
            print(f"  {cohort}: no data")
            continue
        gates = []
        for _, r in sub.iterrows():
            g1 = bool(r["ratio_med"] >= 1.5) if pd.notna(r["ratio_med"]) else False
            g2 = bool(r["false_rev_pct"] <= 35) if pd.notna(r["false_rev_pct"]) else False
            g3 = bool(0 <= r["p_lag_med"] <= 3) if pd.notna(r["p_lag_med"]) else False
            g4 = bool(1 <= r["hw_lag_med"] <= 2) if pd.notna(r["hw_lag_med"]) else False
            mark = lambda b: "✓" if b else "✗"
            gates.append((int(r["K"]), int(r["n"]), g1, g2, g3, g4))
            print(f"  {cohort} K={int(r['K']):>2}  n={int(r['n']):>3}  "
                  f"ratio={r['ratio_med']:.2f}{mark(g1)}  "
                  f"false_rev={r['false_rev_pct']:.1f}%{mark(g2)}  "
                  f"p_lag={r['p_lag_med']:+.1f}{mark(g3)}  "
                  f"hw_lag={r['hw_lag_med']:+.1f}{mark(g4)}")
        # G5 consistency
        try:
            k5 = next(g for g in gates if g[0] == 5)
            k10 = next(g for g in gates if g[0] == 10)
            g5 = k5[2] and k5[3] and k10[2] and k10[3]
            print(f"  {cohort} G5 (K=5 & K=10 both pass G1+G2): {'PASS' if g5 else 'FAIL'}")
        except StopIteration:
            print(f"  {cohort} G5: insufficient horizons")
        print()


if __name__ == "__main__":
    main()
