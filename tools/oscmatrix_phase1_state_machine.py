"""Phase 1 — armed/fire trigger family on HW dots, repaint-safe.

Six AL-side and six SAT-side rules per Phase 1 spec. Reversal arrows are
NOT a target. SAT vanilla HWO Down is never standalone; only as gated
filter (F2/F3/F4) or refractory'd flavor (F5/F6).

Live-feasible rule inputs (no future bars):
  hw, hwo_up, hwo_down, os_hwo_up, ob_hwo_down  (oscmatrix.compute_all)
  upper_zone, lower_zone                         (TV ground truth, doc rule)
  bull_overflow / bear_overflow recent K         (TV ground truth)
  rolling N-bar low/high distance                (live-feasible cap proxy)

Eval-only metrics (look back/forward; NOT used to filter triggers):
  price_pivot_lag, hw_pivot_lag, MFE_K, MAE_K, MFE/MAE, capture_ratio,
  false_reversal_rate.

Output: AL and SAT tables with per-rule pooled medians across 4 tickers.
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

EVAL_W = 10              # pivot lookback/forward window for diagnostics
FWD_K = 10               # MFE/MAE forward horizon (bars)
ROLL_N = 10              # rolling distance window for live-feasible cap
REFRACTORY_K = 10        # bars after a fire before next fire allowed
ARM_RECENT_K = 10        # arm-recent window
OVF_RECENT_K = 5         # overflow recent window
FALSE_REV_ATR_THR = 0.5  # adverse close[t+K]−close[t] threshold (ATR)


def recent_any(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).max().astype(bool)


def apply_refractory(fires: pd.Series, K: int) -> pd.Series:
    arr = fires.values
    out = np.zeros(len(arr), dtype=bool)
    last_fire = -10**9
    for i in range(len(arr)):
        if arr[i] and (i - last_fire) > K:
            out[i] = True
            last_fire = i
    return pd.Series(out, index=fires.index)


def build_inputs(tv: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    inp = pd.DataFrame(index=tv.index)
    hw = ours["hyperwave"].reindex(tv.index)
    inp["hw"] = hw
    inp["hw_lt_30"] = hw < 30
    inp["hw_gt_60"] = hw > 60
    inp["hw_gt_70"] = hw > 70
    inp["hwo_up"] = ours["hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    inp["hwo_down"] = ours["hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    inp["os_hwo_up"] = ours["os_hwo_up"].reindex(tv.index).fillna(False).astype(bool)
    inp["ob_hwo_down"] = ours["ob_hwo_down"].reindex(tv.index).fillna(False).astype(bool)
    inp["upper_zone"] = tv["Upper Confluence Value"].astype("Int64").fillna(0).astype(int)
    inp["lower_zone"] = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    bull_ovf = (tv["Bullish Overflow"].fillna(50) != 50)
    bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50)
    inp["bull_ovf_recent"] = recent_any(bull_ovf, OVF_RECENT_K)
    inp["bear_ovf_recent"] = recent_any(bear_ovf, OVF_RECENT_K)
    return inp


def build_rules(inp: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    al_armed = inp["hw_lt_30"] | (inp["lower_zone"] == 2) | inp["bear_ovf_recent"]
    sat_armed = inp["hw_gt_70"] | (inp["upper_zone"] == 2) | inp["bull_ovf_recent"]
    al_armed_recent = recent_any(al_armed, ARM_RECENT_K)
    sat_armed_recent = recent_any(sat_armed, ARM_RECENT_K)

    rules: dict[str, tuple[str, pd.Series]] = {}

    rules["AL_F1: HWO Up"] = ("AL", inp["hwo_up"])
    rules["AL_F2: HWO Up | OS HWO Up"] = ("AL", inp["hwo_up"] | inp["os_hwo_up"])
    rules["AL_F3: HWO Up & armed_recent10"] = ("AL", inp["hwo_up"] & al_armed_recent)
    rules["AL_F4: HWO Up & lower_zone>=1"] = ("AL", inp["hwo_up"] & (inp["lower_zone"] >= 1))
    rules["AL_F5: HWO Up & bear_ovf_recent5"] = ("AL", inp["hwo_up"] & inp["bear_ovf_recent"])
    al_f3_raw = inp["hwo_up"] & al_armed_recent
    rules["AL_F6: F3 & refractory10"] = ("AL", apply_refractory(al_f3_raw, REFRACTORY_K))

    rules["SAT_F1: OB HWO Down"] = ("SAT", inp["ob_hwo_down"])
    rules["SAT_F2: HWO Down & sat_armed_recent10"] = ("SAT", inp["hwo_down"] & sat_armed_recent)
    rules["SAT_F3: HWO Down & uz>=1 & HW>60"] = ("SAT", inp["hwo_down"] & (inp["upper_zone"] >= 1) & inp["hw_gt_60"])
    rules["SAT_F4: HWO Down & bull_ovf_recent5"] = ("SAT", inp["hwo_down"] & inp["bull_ovf_recent"])
    sat_f2_raw = inp["hwo_down"] & sat_armed_recent
    sat_f5 = apply_refractory(sat_f2_raw, REFRACTORY_K)
    rules["SAT_F5: F2 & refractory10"] = ("SAT", sat_f5)
    rules["SAT_F6: OB HWO Down | F5"] = ("SAT", inp["ob_hwo_down"] | sat_f5)

    return rules


def lag_in_window(bar_loc: int, series: pd.Series, window: int, side: str) -> int:
    n = len(series)
    lo = max(0, bar_loc - window)
    hi = min(n, bar_loc + window + 1)
    sub = series.iloc[lo:hi]
    if side == "low":
        ext_offset = int(sub.values.argmin())
    else:
        ext_offset = int(sub.values.argmax())
    return bar_loc - (lo + ext_offset)


def forward_extremes(df: pd.DataFrame, K: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    fmh = pd.concat([df["high"].shift(-k) for k in range(1, K + 1)], axis=1).max(axis=1)
    fml = pd.concat([df["low"].shift(-k) for k in range(1, K + 1)], axis=1).min(axis=1)
    fwd_close = df["close"].shift(-K)
    return fmh, fml, fwd_close


def score_rule(side: str, fires: pd.Series, df: pd.DataFrame, hw: pd.Series, atr: pd.Series) -> dict | None:
    locs = np.where(fires.values)[0]
    if len(locs) == 0:
        return None
    close, high, low = df["close"], df["high"], df["low"]
    fmh, fml, fwd_close = forward_extremes(df, FWD_K)
    roll_low = low.rolling(ROLL_N, min_periods=1).min()
    roll_high = high.rolling(ROLL_N, min_periods=1).max()

    p_lags, hw_lags, roll_dists, maes, mfes, ratios, captures, false_revs = [], [], [], [], [], [], [], []
    for i in locs:
        a = atr.iloc[i]
        if not a or np.isnan(a):
            continue
        ci = close.iloc[i]
        if side == "AL":
            p_lag = lag_in_window(i, low, EVAL_W, "low")
            hw_lag = lag_in_window(i, hw, EVAL_W, "low")
            roll_dist = (ci - roll_low.iloc[i]) / a
            mfe_raw = fmh.iloc[i] - ci
            mae_raw = ci - fml.iloc[i]
            realized = fwd_close.iloc[i] - ci
        else:
            p_lag = lag_in_window(i, high, EVAL_W, "high")
            hw_lag = lag_in_window(i, hw, EVAL_W, "high")
            roll_dist = (roll_high.iloc[i] - ci) / a
            mfe_raw = ci - fml.iloc[i]
            mae_raw = fmh.iloc[i] - ci
            realized = ci - fwd_close.iloc[i]

        if pd.isna(mfe_raw) or pd.isna(mae_raw):
            continue
        mfe_atr = float(mfe_raw / a)
        mae_atr = float(mae_raw / a)
        ratio = mfe_atr / max(mae_atr, 1e-6)
        cap = (float(realized) / float(mfe_raw)) if pd.notna(realized) and mfe_raw > 0 else np.nan
        false_rev = (realized < -FALSE_REV_ATR_THR * a) if pd.notna(realized) else None

        p_lags.append(p_lag)
        hw_lags.append(hw_lag)
        roll_dists.append(float(roll_dist))
        mfes.append(mfe_atr)
        maes.append(mae_atr)
        ratios.append(ratio)
        if not np.isnan(cap):
            captures.append(cap)
        if false_rev is not None:
            false_revs.append(bool(false_rev))

    if not p_lags:
        return None
    return {
        "n": len(p_lags),
        "p_lag_med": float(np.median(p_lags)),
        "hw_lag_med": float(np.median(hw_lags)),
        "roll_dist_atr_med": float(np.median(roll_dists)),
        "mae10_atr_med": float(np.median(maes)),
        "mfe10_atr_med": float(np.median(mfes)),
        "mfe_mae_med": float(np.median(ratios)),
        "capture_med": float(np.median(captures)) if captures else np.nan,
        "false_rev_rate": float(np.mean(false_revs) * 100) if false_revs else np.nan,
    }


def main() -> None:
    rule_results: dict[str, list[dict]] = {}
    rule_side: dict[str, str] = {}
    for ticker, path in CSV_PATHS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        df = tv[["open", "high", "low", "close"]].astype(float)
        hw = ours["hyperwave"].reindex(tv.index)
        atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
        inp = build_inputs(tv, ours)
        rules = build_rules(inp)
        for rname, (side, fires) in rules.items():
            r = score_rule(side, fires, df, hw, atr)
            if r is None:
                continue
            r["ticker"] = ticker
            rule_results.setdefault(rname, []).append(r)
            rule_side[rname] = side

    rows = []
    for rname, lst in rule_results.items():
        total_n = sum(d["n"] for d in lst)
        if total_n == 0:
            continue

        def wavg(key: str) -> float:
            vals = [(d[key], d["n"]) for d in lst if pd.notna(d.get(key, np.nan))]
            if not vals:
                return np.nan
            num = sum(v * w for v, w in vals)
            den = sum(w for _, w in vals)
            return num / den if den else np.nan

        rows.append({
            "rule": rname,
            "side": rule_side[rname],
            "n": total_n,
            "p_lag_med": wavg("p_lag_med"),
            "hw_lag_med": wavg("hw_lag_med"),
            "roll_dist_atr_med": wavg("roll_dist_atr_med"),
            "mae10_atr_med": wavg("mae10_atr_med"),
            "mfe10_atr_med": wavg("mfe10_atr_med"),
            "mfe_mae_med": wavg("mfe_mae_med"),
            "capture_med": wavg("capture_med"),
            "false_rev_rate": wavg("false_rev_rate"),
        })
    rep = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("=== Phase 1 — armed/fire trigger family scoring (4-ticker pooled) ===")
    print(f"  fwd_K={FWD_K}  eval_W={EVAL_W}  roll_N={ROLL_N}  refractory={REFRACTORY_K}")
    print(f"  arm_recent={ARM_RECENT_K}  ovf_recent={OVF_RECENT_K}  false_rev_thr={FALSE_REV_ATR_THR} ATR")
    print(f"  AL_ARMED = HW<30 | lz==2 | bear_ovf_recent5")
    print(f"  SAT_ARMED = HW>70 | uz==2 | bull_ovf_recent5\n")

    cols = ["rule", "n", "p_lag_med", "hw_lag_med", "roll_dist_atr_med",
            "mae10_atr_med", "mfe10_atr_med", "mfe_mae_med", "capture_med", "false_rev_rate"]
    print("--- AL side (long, expect close > entry) ---")
    al = rep[rep["side"] == "AL"][cols]
    print(al.to_string(index=False))
    print()
    print("--- SAT side (short, expect close < entry) ---")
    sat = rep[rep["side"] == "SAT"][cols]
    print(sat.to_string(index=False))
    print()

    print("--- Selection criteria reminder ---")
    print("  pick: high n, low |p_lag|, low roll_dist_atr, high mfe_mae, low false_rev_rate")
    print("  positive p_lag/hw_lag = signal AFTER pivot (live-feasible)")


if __name__ == "__main__":
    main()
