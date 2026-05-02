"""Layer B — rule extraction (NOT classifier) for Reversal Up/Down arrows.

Tests user-spec rules v1/v2/v3 plus probe-driven refinements R1/R2 for major (+)
arrows. Reports precision/recall/F1 vs TV ground-truth Reversal Up/Down +/-
columns with ±1 bar tolerance.

User spec:
  v1 = HW cross up AND HW < 50
  v2 = v1 AND LowerZone >= 1
  v3 = v2 AND (bull_div_recent OR bear_overflow_active)

Probe-driven (major-arrow-targeted):
  R_up_plus  = HW < 30 AND lower_zone >= 1 AND bear_overflow_recent_5
  R_up_minus = HW < 50 AND lower_zone >= 1 AND not(bear_overflow_recent_5 AND HW < 30)
  R_dn_plus  = HW > 70 AND upper_zone >= 1 AND bull_overflow_recent_5
  R_dn_minus = HW > 50 AND upper_zone >= 1 AND not(bull_overflow_recent_5 AND HW > 70)
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

TOLERANCE = 1


def recent_any(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).max().astype(bool)


def build_inputs(tv: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=tv.index)
    hw = ours["hyperwave"]
    out["hw"] = hw
    out["hw_lt_50"] = hw < 50
    out["hw_lt_30"] = hw < 30
    out["hw_gt_50"] = hw > 50
    out["hw_gt_70"] = hw > 70
    out["hwo_up"] = ours["hwo_up"].fillna(False).astype(bool)
    out["hwo_down"] = ours["hwo_down"].fillna(False).astype(bool)
    # Use TV ground-truth zones to isolate rule-quality from our zone-error.
    out["upper_zone"] = tv["Upper Confluence Value"].astype("Int64").fillna(0).astype(int)
    out["lower_zone"] = tv["Lower Confluence Value"].astype("Int64").fillna(0).astype(int)
    bull_ovf = (tv["Bullish Overflow"].fillna(50) != 50).astype(bool)
    bear_ovf = (tv["Bearish Overflow"].fillna(50) != 50).astype(bool)
    out["bull_ovf_recent_5"] = recent_any(bull_ovf, 5)
    out["bear_ovf_recent_5"] = recent_any(bear_ovf, 5)
    out["bull_ovf_active"] = bull_ovf
    out["bear_ovf_active"] = bear_ovf
    return out


def evaluate_rule(fires: pd.Series, truth_idx: pd.Index, all_idx: pd.Index, tolerance: int = TOLERANCE) -> dict:
    fires = fires.fillna(False).astype(bool)
    our_fires = list(all_idx[fires.values])
    tv_fires = list(truth_idx)
    tp = 0
    matched_t = set()
    matched_o = set()
    locs_o = {ts: all_idx.get_loc(ts) for ts in our_fires}
    locs_t = {ts: all_idx.get_loc(ts) for ts in tv_fires}
    for ts_o, lo in locs_o.items():
        if ts_o in matched_o:
            continue
        for ts_t, lt in locs_t.items():
            if ts_t in matched_t:
                continue
            if abs(lo - lt) <= tolerance:
                tp += 1
                matched_o.add(ts_o)
                matched_t.add(ts_t)
                break
    fp = len(our_fires) - len(matched_o)
    fn = len(tv_fires) - len(matched_t)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and (prec + rec)) else float("nan")
    return {"n_rule": len(our_fires), "n_tv": len(tv_fires), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def run_rules(inp: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    rules: dict[str, pd.Series] = {}

    # User spec, baseline progression
    v1 = inp["hwo_up"] & inp["hw_lt_50"]
    v2 = v1 & (inp["lower_zone"] >= 1)
    v3 = v2 & inp["bear_ovf_recent_5"]   # divergence proxy too sparse → use overflow as exhaustion proxy

    # Probe-driven major-arrow-targeted
    R_up_plus = (inp["hw_lt_30"]) & (inp["lower_zone"] >= 1) & inp["bear_ovf_recent_5"]
    R_up_plus_strict = R_up_plus & (inp["lower_zone"] == 2)
    R_up_minus = inp["hw_lt_50"] & (inp["lower_zone"] >= 1) & ~R_up_plus

    R_dn_plus = (inp["hw_gt_70"]) & (inp["upper_zone"] >= 1) & inp["bull_ovf_recent_5"]
    R_dn_plus_strict = R_dn_plus & (inp["upper_zone"] == 2)
    R_dn_minus = inp["hw_gt_50"] & (inp["upper_zone"] >= 1) & ~R_dn_plus

    rules["v1_up: hwo_up & hw<50"] = v1
    rules["v2_up: v1 & lz>=1"] = v2
    rules["v3_up: v2 & bear_ovf_5"] = v3
    rules["R_up_plus: hw<30 & lz>=1 & bear_ovf_5"] = R_up_plus
    rules["R_up_plus_strict: ...& lz==2"] = R_up_plus_strict
    rules["R_up_minus: hw<50 & lz>=1 & !plus"] = R_up_minus
    rules["R_dn_plus: hw>70 & uz>=1 & bull_ovf_5"] = R_dn_plus
    rules["R_dn_plus_strict: ...& uz==2"] = R_dn_plus_strict
    rules["R_dn_minus: hw>50 & uz>=1 & !plus"] = R_dn_minus

    truths = {
        "Reversal Up +": tv.index[tv["Reversal Up +"].notna()],
        "Reversal Up -": tv.index[tv["Reversal Up -"].notna()],
        "Reversal Down +": tv.index[tv["Reversal Down +"].notna()],
        "Reversal Down -": tv.index[tv["Reversal Down -"].notna()],
    }

    rows = []
    for rname, rseries in rules.items():
        for tname, tidx in truths.items():
            d = evaluate_rule(rseries, tidx, inp.index)
            d["rule"] = rname
            d["target"] = tname
            rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for ticker, path in CSV_PATHS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        inp = build_inputs(tv, ours)
        df = run_rules(inp, tv)
        df["ticker"] = ticker
        all_rows.append(df)
    big = pd.concat(all_rows, ignore_index=True)

    # Aggregate across tickers per rule × target
    agg = big.groupby(["rule", "target"]).agg(
        n_rule=("n_rule", "sum"),
        n_tv=("n_tv", "sum"),
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
    ).reset_index()
    agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"]).replace(0, np.nan)
    agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"]).replace(0, np.nan)
    agg["f1"] = 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("=== Aggregate (4-ticker pooled, ±1 bar) ===")
    cols = ["rule", "target", "n_rule", "n_tv", "tp", "fp", "fn", "precision", "recall", "f1"]
    # Show each rule against its primary target (the one matching its kind)
    primary = {
        "v1_up: hwo_up & hw<50": "Reversal Up -",
        "v2_up: v1 & lz>=1": "Reversal Up -",
        "v3_up: v2 & bear_ovf_5": "Reversal Up -",
        "R_up_plus: hw<30 & lz>=1 & bear_ovf_5": "Reversal Up +",
        "R_up_plus_strict: ...& lz==2": "Reversal Up +",
        "R_up_minus: hw<50 & lz>=1 & !plus": "Reversal Up -",
        "R_dn_plus: hw>70 & uz>=1 & bull_ovf_5": "Reversal Down +",
        "R_dn_plus_strict: ...& uz==2": "Reversal Down +",
        "R_dn_minus: hw>50 & uz>=1 & !plus": "Reversal Down -",
    }
    print("\n--- PRIMARY target match per rule ---")
    pri = agg[agg.apply(lambda r: primary.get(r["rule"]) == r["target"], axis=1)]
    print(pri[cols].sort_values("f1", ascending=False, na_position="last").to_string(index=False))

    print("\n--- Full rule × target matrix (precision) ---")
    pivot = agg.pivot(index="rule", columns="target", values="precision")
    print(pivot.to_string())
    print("\n--- Full rule × target matrix (recall) ---")
    pivot = agg.pivot(index="rule", columns="target", values="recall")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
