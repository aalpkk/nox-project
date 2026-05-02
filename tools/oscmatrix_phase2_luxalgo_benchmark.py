"""Phase 2 — LuxAlgo behavior benchmark, same pivot-proximity table for all.

Scope: 4 DISCOVERY tickers (only set with full TV ground truth).

Side-by-side, AL: TV HWO Up / TV OS-HWO Up / TV Rev Up + / TV Rev Up - /
ours HWO Up / ours OS-HWO Up / AL_F4 / AL_F6.

SAT: TV HWO Down / TV OB-HWO Down / TV Rev Down + / TV Rev Down - /
ours HWO Down / ours OB-HWO Down / SAT_B / SAT_E.

The "is_repaint_suspect" column flags TV Reversal arrows (Phase 0 evidence:
74-84% negative price-pivot lag). Read those rows as "look-ahead reference",
not honest live signal.

Goal: see whether AL_F6 sits on TV HWO Up's curve (same family of behavior)
or has drifted into a different signal class.
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


def collect_samples(side: str, fires: pd.Series, df: pd.DataFrame, hw: pd.Series, atr: pd.Series, K: int) -> list[dict]:
    locs = np.where(fires.values)[0]
    if len(locs) == 0:
        return []
    close, high, low = df["close"], df["high"], df["low"]
    fmh, fml, fwd_close = forward_extremes(df, K)
    roll_low = low.rolling(ROLL_N, min_periods=1).min()
    roll_high = high.rolling(ROLL_N, min_periods=1).max()
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
            roll_dist = float((roll_high.iloc[i] - ci) / a)
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


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0, "p_lag_med": np.nan, "hw_lag_med": np.nan, "roll_dist_med": np.nan,
                "mae_med": np.nan, "mfe_med": np.nan, "ratio_med": np.nan,
                "capture_med": np.nan, "false_rev_pct": np.nan}
    df = pd.DataFrame(samples)
    return {
        "n": len(df),
        "p_lag_med": float(df["p_lag"].median()),
        "hw_lag_med": float(df["hw_lag"].median()),
        "roll_dist_med": float(df["roll_dist"].median()),
        "mae_med": float(df["mae_atr"].median()),
        "mfe_med": float(df["mfe_atr"].median()),
        "ratio_med": float(df["ratio"].median()),
        "capture_med": float(df["capture"].median()) if df["capture"].notna().any() else np.nan,
        "false_rev_pct": float(df["false_rev"].mean() * 100) if df["false_rev"].notna().any() else np.nan,
    }


def build_signals(tv: pd.DataFrame, ours: pd.DataFrame) -> dict[str, tuple[str, str, pd.Series]]:
    """Return {label: (side, source_tag, fire_series)}."""
    hw = ours["hyperwave"].reindex(tv.index)
    bull_ovf = (tv["Bullish Overflow"].fillna(50) != 50)
    bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50)
    bull_ovf_recent = recent_any(bull_ovf, OVF_RECENT_K)
    bear_ovf_recent = recent_any(bear_ovf, OVF_RECENT_K)
    lower_zone = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    upper_zone = tv["Upper Confluence Value"].astype("Int64").fillna(0).astype(int)

    al_armed = (hw < 30) | (lower_zone == 2) | bear_ovf_recent
    sat_armed = (hw > 70) | (upper_zone == 2) | bull_ovf_recent
    al_armed_rec = recent_any(al_armed, ARM_RECENT_K)
    sat_armed_rec = recent_any(sat_armed, ARM_RECENT_K)

    def tv_fires(col):
        return tv[col].notna() if col in tv.columns else pd.Series(False, index=tv.index)

    ours_hwo_up = ours["hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    ours_hwo_dn = ours["hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    ours_os_up = ours["os_hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    ours_ob_dn = ours["ob_hwo_down"].reindex(tv.index).fillna(False).astype(bool)

    al_f4 = ours_hwo_up & (lower_zone >= 1)
    al_f6 = apply_refractory(ours_hwo_up & al_armed_rec, REFRACTORY_K)
    sat_b = ours_hwo_dn & bull_ovf_recent
    # SAT_E uses live-feasible high distance cap
    df_close = tv["close"].astype(float)
    df_high = tv["high"].astype(float)
    atr = (tv["high"] - tv["low"]).rolling(14, min_periods=14).mean()
    roll_high_dist = (df_high.rolling(ROLL_N, min_periods=1).max() - df_close) / atr
    sat_e = ours_hwo_dn & sat_armed_rec & (roll_high_dist.fillna(np.inf) <= SAT_E_HIGH_DIST_CAP_ATR)

    sigs = {
        # AL — TV reference (look-ahead/repaint-suspect for arrows)
        "TV  HWO Up":               ("AL", "TV-dot",        tv_fires("HWO Up")),
        "TV  Oversold HWO Up":      ("AL", "TV-dot",        tv_fires("Oversold HWO Up")),
        "TV  Reversal Up +":        ("AL", "TV-arrow*",     tv_fires("Reversal Up +")),
        "TV  Reversal Up -":        ("AL", "TV-arrow*",     tv_fires("Reversal Up -")),
        # AL — ours
        "OURS HWO Up":              ("AL", "ours-dot",      ours_hwo_up),
        "OURS Oversold HWO Up":     ("AL", "ours-dot",      ours_os_up),
        "OURS AL_F4 (HWO Up & lz>=1)":            ("AL", "ours-rule", al_f4),
        "OURS AL_F6 (HWO Up & arm10 & ref10)":    ("AL", "ours-rule", al_f6),

        # SAT — TV reference
        "TV  HWO Down":             ("SAT", "TV-dot",       tv_fires("HWO Down")),
        "TV  Overbought HWO Down":  ("SAT", "TV-dot",       tv_fires("Overbought HWO Down")),
        "TV  Reversal Down +":      ("SAT", "TV-arrow*",    tv_fires("Reversal Down +")),
        "TV  Reversal Down -":      ("SAT", "TV-arrow*",    tv_fires("Reversal Down -")),
        # SAT — ours
        "OURS HWO Down":            ("SAT", "ours-dot",     ours_hwo_dn),
        "OURS Overbought HWO Down": ("SAT", "ours-dot",     ours_ob_dn),
        "OURS SAT_B (HWO Dn & bull_ovf_recent5)": ("SAT", "ours-rule", sat_b),
        "OURS SAT_E (HWO Dn & sat_arm10 & high_dist≤0.75)": ("SAT", "ours-rule", sat_e),
    }
    return sigs


def main() -> None:
    pooled: dict[tuple[str, int], list[dict]] = {}
    sig_meta: dict[str, tuple[str, str]] = {}
    sig_order: list[str] = []

    for ticker, path in DISCOVERY.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        df = tv[["open", "high", "low", "close"]].astype(float)
        hw = ours["hyperwave"].reindex(tv.index)
        atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
        sigs = build_signals(tv, ours)
        for label, (side, tag, fires) in sigs.items():
            if label not in sig_meta:
                sig_meta[label] = (side, tag)
                sig_order.append(label)
            for K in HORIZONS:
                pooled.setdefault((label, K), []).extend(
                    collect_samples(side, fires, df, hw, atr, K)
                )

    rows = []
    for label in sig_order:
        side, tag = sig_meta[label]
        for K in HORIZONS:
            s = summarize(pooled.get((label, K), []))
            rows.append({"signal": label, "src": tag, "side": side, "K": K, **s})
    rep = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.max_colwidth", 60)
    print("=== Phase 2 — LuxAlgo behavior benchmark (4 DISCOVERY tickers) ===")
    print("  src tag legend: TV-dot, TV-arrow* (* = repaint-suspect),")
    print("                  ours-dot, ours-rule")
    print(f"  Horizons: {HORIZONS}\n")

    cols = ["signal", "src", "K", "n", "p_lag_med", "hw_lag_med", "roll_dist_med",
            "mae_med", "mfe_med", "ratio_med", "capture_med", "false_rev_pct"]

    print("--- AL side ---")
    al = rep[rep["side"] == "AL"][cols]
    print(al.to_string(index=False))

    print("\n--- SAT side ---")
    sat = rep[rep["side"] == "SAT"][cols]
    print(sat.to_string(index=False))

    # AL_F6 vs TV HWO Up — same-family check
    print("\n\n=== AL_F6 ↔ TV HWO Up similarity check (K=10) ===")
    print("Same-family if: |Δ p_lag|≤1, |Δ hw_lag|≤1, ratio within 2× of TV HWO Up,")
    print("                 false_rev_pct within ±15pp.\n")
    base = rep[(rep["signal"] == "TV  HWO Up") & (rep["K"] == 10)].iloc[0]
    targets = [
        "OURS HWO Up",
        "OURS AL_F4 (HWO Up & lz>=1)",
        "OURS AL_F6 (HWO Up & arm10 & ref10)",
    ]
    print(f"  TV HWO Up  K=10:  n={int(base['n'])}  p_lag={base['p_lag_med']:+.1f}  hw_lag={base['hw_lag_med']:+.1f}  "
          f"ratio={base['ratio_med']:.2f}  false_rev={base['false_rev_pct']:.1f}%  cap={base['capture_med']:.2f}")
    print()
    for t in targets:
        r = rep[(rep["signal"] == t) & (rep["K"] == 10)]
        if r.empty:
            continue
        r = r.iloc[0]
        dpl = abs(r["p_lag_med"] - base["p_lag_med"])
        dhl = abs(r["hw_lag_med"] - base["hw_lag_med"])
        ratio_ok = (r["ratio_med"] <= 2 * base["ratio_med"]) and (r["ratio_med"] >= base["ratio_med"] / 2)
        fr_ok = abs(r["false_rev_pct"] - base["false_rev_pct"]) <= 15
        m = lambda b: "✓" if b else "✗"
        print(f"  {t:<40}  n={int(r['n']):>3}  Δp_lag={dpl:.1f}{m(dpl<=1)}  "
              f"Δhw_lag={dhl:.1f}{m(dhl<=1)}  ratio={r['ratio_med']:.2f}{m(ratio_ok)}  "
              f"false_rev={r['false_rev_pct']:.1f}%{m(fr_ok)}")


if __name__ == "__main__":
    main()
