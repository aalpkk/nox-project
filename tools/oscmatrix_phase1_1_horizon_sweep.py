"""Phase 1.1 — horizon sweep + SAT redesign + rolling-distance cap.

Builds on Phase 1. AL_F2/F5 dropped (F2 redundant: os_hwo_up ⊂ hwo_up; F5 too
sparse + ratio outlier). SAT redesigned: vanilla `HWO Down & sat_armed` and
`HWO Down & uz>=1 & HW>60` removed (MFE/MAE < 1 in Phase 1). Replacement
candidates per spec:

  AL:
    AL_F4 = HWO Up & lower_zone>=1
    AL_F6 = HWO Up & armed_recent10 & refractory10

  SAT:
    SAT_A  = OB HWO Down
    SAT_B  = HWO Down & bull_ovf_recent5
    SAT_C  = SAT_A | SAT_B
    SAT_D  = HWO Down & bull_ovf_recent5 & refractory10
    SAT_E  = HWO Down & sat_armed_recent10 & (rolling_high_dist_atr <= 0.75)

Each scored at K ∈ {5, 10, 20}. Pooled per-signal samples across 4 tickers
(unweighted; medians correct). Selection priority (per spec):
  (1) false_rev% low  (2) MAE low  (3) MFE/MAE > 1.5
  (4) capture pos & meaningful  (5) n adequate  (6) p_lag aligned
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_PATHS = {
    "THYAO": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/BIST_THYAO, 1D.csv",
    "GARAN": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_ASELS, 1D.csv",
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


def build_inputs(tv: pd.DataFrame, ours: pd.DataFrame, df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    inp = pd.DataFrame(index=tv.index)
    hw = ours["hyperwave"].reindex(tv.index)
    inp["hw"] = hw
    inp["hw_lt_30"] = hw < 30
    inp["hw_gt_70"] = hw > 70
    inp["hwo_up"] = ours["hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    inp["hwo_down"] = ours["hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    inp["ob_hwo_down"] = ours["ob_hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    inp["lower_zone"] = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    inp["upper_zone"] = tv["Upper Confluence Value"].astype("Int64").fillna(0).astype(int)
    bull_ovf = (tv["Bullish Overflow"].fillna(50) != 50)
    bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50)
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

    rules: dict[str, tuple[str, pd.Series]] = {}

    # AL
    rules["AL_F4: HWO Up & lz>=1"] = ("AL", inp["hwo_up"] & (inp["lower_zone"] >= 1))
    al_f3_raw = inp["hwo_up"] & al_armed_recent
    rules["AL_F6: HWO Up & arm_rec10 & refract10"] = ("AL", apply_refractory(al_f3_raw, REFRACTORY_K))

    # SAT
    sat_a = inp["ob_hwo_down"]
    sat_b = inp["hwo_down"] & inp["bull_ovf_recent"]
    sat_c = sat_a | sat_b
    sat_d = apply_refractory(sat_b, REFRACTORY_K)
    sat_e_pre = inp["hwo_down"] & sat_armed_recent
    high_dist_ok = inp["roll_high_dist_atr"].fillna(np.inf) <= SAT_E_HIGH_DIST_CAP_ATR
    sat_e = sat_e_pre & high_dist_ok

    rules["SAT_A: OB HWO Down"] = ("SAT", sat_a)
    rules["SAT_B: HWO Down & bull_ovf_recent5"] = ("SAT", sat_b)
    rules["SAT_C: A | B"] = ("SAT", sat_c)
    rules["SAT_D: B & refractory10"] = ("SAT", sat_d)
    rules[f"SAT_E: HWO Down & sat_arm_rec10 & high_dist<={SAT_E_HIGH_DIST_CAP_ATR}"] = ("SAT", sat_e)

    return rules


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
    samples = []
    for i in locs:
        a = atr.iloc[i]
        if not a or np.isnan(a):
            continue
        ci = close.iloc[i]
        if side == "AL":
            p_lag = lag_in_window(i, low, EVAL_W, "low")
            hw_lag = lag_in_window(i, hw, EVAL_W, "low")
            roll_dist = inp["roll_high_dist_atr"].iloc[i]  # n/a for AL — use low-dist instead
            roll_low = low.rolling(ROLL_N, min_periods=1).min()
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
        return {"n": 0}
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


def main() -> None:
    tickers_loaded = []
    panels = []
    for ticker, path in CSV_PATHS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        df = tv[["open", "high", "low", "close"]].astype(float)
        hw = ours["hyperwave"].reindex(tv.index)
        atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
        inp = build_inputs(tv, ours, df, atr)
        rules = build_rules(inp)
        panels.append((ticker, df, hw, atr, inp, rules))
        tickers_loaded.append(ticker)

    # Pool samples across tickers per (rule, K)
    rule_order: list[str] = []
    side_map: dict[str, str] = {}
    pooled: dict[tuple[str, int], list[dict]] = {}
    for ticker, df, hw, atr, inp, rules in panels:
        for rname, (side, fires) in rules.items():
            if rname not in side_map:
                rule_order.append(rname)
                side_map[rname] = side
            for K in HORIZONS:
                pooled.setdefault((rname, K), []).extend(
                    collect_samples(side, fires, df, hw, atr, inp, K)
                )

    rows = []
    for rname in rule_order:
        for K in HORIZONS:
            s = summarize_samples(pooled.get((rname, K), []))
            rows.append({"rule": rname, "side": side_map[rname], "K": K, **s})
    rep = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"=== Phase 1.1 — horizon sweep across {len(tickers_loaded)} tickers ({', '.join(tickers_loaded)}) ===")
    print(f"  K∈{HORIZONS}  eval_W={EVAL_W}  roll_N={ROLL_N}  refractory={REFRACTORY_K}")
    print(f"  arm_recent={ARM_RECENT_K}  ovf_recent={OVF_RECENT_K}  sat_E_high_dist_cap={SAT_E_HIGH_DIST_CAP_ATR} ATR")
    print()

    cols = ["rule", "K", "n", "p_lag_med", "hw_lag_med", "roll_dist_med",
            "mae_med", "mfe_med", "ratio_med", "capture_med", "false_rev_pct"]
    print("--- AL side ---")
    print(rep[rep["side"] == "AL"][cols].to_string(index=False))
    print()
    print("--- SAT side ---")
    print(rep[rep["side"] == "SAT"][cols].to_string(index=False))
    print()

    # Selection scoring per priority order
    print("--- Selection scoring (priority: false_rev↓ → MAE↓ → ratio↑ → capture↑ → n) ---")
    for side in ["AL", "SAT"]:
        sub = rep[rep["side"] == side].copy()
        # score: lower better for false_rev_pct, mae_med; higher better for ratio_med, capture_med
        sub["score"] = (
            -sub["false_rev_pct"].fillna(100)
            - sub["mae_med"].fillna(10) * 20
            + sub["ratio_med"].clip(upper=10).fillna(0) * 20
            + sub["capture_med"].fillna(-1) * 20
        )
        sub = sub.sort_values("score", ascending=False)
        print(f"\n{side} ranking (top 6):")
        print(sub[cols + ["score"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
