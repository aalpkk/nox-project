"""Layer A validator — HW dots + Confluence Zones vs TV CSV ground truth.

For dot columns (HWO Up / HWO Down / Oversold / Overbought) we score by
timestamp match: a TV non-null cell == a fire on that bar. We allow ±k bar
tolerance because our HW (corr 0.967 vs TV) can shift a cross by 1 bar.

For zone columns (Upper/Lower Confluence Value, discrete {0,1,2}) we sweep
the candidate rule set and pick the highest exact-match rate per ticker.
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

DOT_PAIRS = [
    ("HWO Up", "hwo_up"),
    ("HWO Down", "hwo_down"),
    ("Oversold HWO Up", "os_hwo_up"),
    ("Overbought HWO Down", "ob_hwo_down"),
]

ZONE_PAIRS = [
    ("Upper Confluence Value", "upper_zone"),
    ("Lower Confluence Value", "lower_zone"),
]


def score_dots(tv: pd.DataFrame, ours: pd.DataFrame, tolerance: int = 1) -> list[dict]:
    rows = []
    for tv_col, our_col in DOT_PAIRS:
        if tv_col not in tv.columns or our_col not in ours.columns:
            rows.append({"col": tv_col, "tv_n": 0, "ours_n": 0, "tp": 0, "fp": 0, "fn": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan})
            continue
        tv_idx = tv[tv_col].notna()
        ours_idx = ours[our_col].fillna(False).astype(bool)
        merged = pd.DataFrame({"tv": tv_idx.values, "ours": ours_idx.reindex(tv.index, fill_value=False).values}, index=tv.index)
        tv_fires = merged.index[merged["tv"]]
        our_fires = merged.index[merged["ours"]]
        if tolerance == 0:
            tp_set = set(tv_fires) & set(our_fires)
            fp = len(set(our_fires) - set(tv_fires))
            fn = len(set(tv_fires) - set(our_fires))
            tp = len(tp_set)
        else:
            tp = 0
            matched_tv = set()
            matched_our = set()
            tv_pos = pd.Index(tv_fires)
            our_pos = pd.Index(our_fires)
            tv_locs = {ts: tv.index.get_loc(ts) for ts in tv_pos}
            our_locs = {ts: tv.index.get_loc(ts) for ts in our_pos}
            for ts_o, loc_o in our_locs.items():
                if ts_o in matched_our:
                    continue
                for ts_t, loc_t in tv_locs.items():
                    if ts_t in matched_tv:
                        continue
                    if abs(loc_o - loc_t) <= tolerance:
                        tp += 1
                        matched_tv.add(ts_t)
                        matched_our.add(ts_o)
                        break
            fp = len(our_pos) - len(matched_our)
            fn = len(tv_pos) - len(matched_tv)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
        rows.append({"col": tv_col, "tv_n": len(tv_fires), "ours_n": len(our_fires),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    return rows


def score_zones(tv: pd.DataFrame, ours: pd.DataFrame) -> list[dict]:
    rows = []
    for tv_col, our_col in ZONE_PAIRS:
        if tv_col not in tv.columns or our_col not in ours.columns:
            rows.append({"col": tv_col, "n": 0, "exact": np.nan, "off_by_1": np.nan,
                         "tv_dist": "", "ours_dist": ""})
            continue
        merged = pd.DataFrame({"tv": tv[tv_col], "ours": ours[our_col]}, index=tv.index).dropna()
        if merged.empty:
            rows.append({"col": tv_col, "n": 0, "exact": np.nan, "off_by_1": np.nan,
                         "tv_dist": "", "ours_dist": ""})
            continue
        merged["tv"] = merged["tv"].astype(int)
        merged["ours"] = merged["ours"].astype(int)
        diff = (merged["tv"] - merged["ours"]).abs()
        rows.append({
            "col": tv_col,
            "n": len(merged),
            "exact": float((diff == 0).mean()),
            "off_by_1": float((diff <= 1).mean()),
            "tv_dist": dict(merged["tv"].value_counts().sort_index()),
            "ours_dist": dict(merged["ours"].value_counts().sort_index()),
        })
    return rows


def run_one(ticker: str, csv_path: str, zone_rule: str) -> tuple[list[dict], list[dict]]:
    tv = load_tv_csv(csv_path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule=zone_rule)
    dot_rows = score_dots(tv, ours, tolerance=1)
    zone_rows = score_zones(tv, ours)
    for r in dot_rows + zone_rows:
        r["ticker"] = ticker
        r["zone_rule"] = zone_rule
    return dot_rows, zone_rows


def main() -> None:
    print("=== Layer A validation: HW dots + Confluence Zones ===\n")

    # Dots first (rule-independent)
    all_dots = []
    for ticker, path in CSV_PATHS.items():
        dot_rows, _ = run_one(ticker, path, zone_rule="above_50")
        all_dots.extend(dot_rows)
    dot_df = pd.DataFrame(all_dots)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("--- HW dot triggers (±1 bar tolerance) ---")
    print(dot_df[["ticker", "col", "tv_n", "ours_n", "tp", "fp", "fn", "precision", "recall", "f1"]].to_string(index=False))
    print()

    # Zones — sweep 3 rules
    for rule in ["above_50", "above_signal", "vs_threshold"]:
        all_zones = []
        for ticker, path in CSV_PATHS.items():
            _, z = run_one(ticker, path, zone_rule=rule)
            all_zones.extend(z)
        z_df = pd.DataFrame(all_zones)
        print(f"--- Confluence Zones, rule={rule!r} ---")
        cols = ["ticker", "col", "n", "exact", "off_by_1", "tv_dist", "ours_dist"]
        print(z_df[cols].to_string(index=False))
        print(f"  → avg exact-match: {z_df['exact'].mean():.3f}")
        print()


if __name__ == "__main__":
    main()
