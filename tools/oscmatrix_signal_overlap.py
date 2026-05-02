"""Signal overlap diagnostic: candidate AL rules vs TV Reversal Up arrows.

We don't (yet) reproduce TV's Confluence Value column — it's the strongest
single discriminator at fire moments (93% of fires) but stateful in a way the
public formulae don't expose. This harness tests rules built from what we DO
have (HW, MF, thresholds) and reports precision/recall against TV's Reversal Up
+/- columns with a small ±k bar tolerance window.

Goal: sanity-check that our components carry enough signal to track TV's
arrows. The reversal-inspect run gave us the firing-context fingerprint:
  - HW < 50 at 88.4% of fires (oversold-recovering)
  - TV Lower Confluence > 0.5 at 93% of fires (we can't compute this yet)
  - MF < 50 only 39.5% — MF position is NOT a strong gate
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_DIR = "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1"
CSVS = {
    "THYAO": f"{CSV_DIR}/BIST_THYAO, 1D.csv",
    "GARAN": f"{CSV_DIR}/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": f"{CSV_DIR}/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": f"{CSV_DIR}/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

TOLERANCE_BARS = 2  # ±2-day window for "match"


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """True at bar t when a crosses up through b (a[t-1] <= b[t-1] and a[t] > b[t])."""
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    return (prev_a <= prev_b) & (a > b)


def _cross_up_value(a: pd.Series, level: float) -> pd.Series:
    return (a.shift(1) <= level) & (a > level)


def build_candidates(comp: pd.DataFrame) -> dict[str, pd.Series]:
    mf = comp["money_flow"]
    hw = comp["hyperwave"]
    sig = hw.rolling(3, min_periods=3).mean()  # HW signal proxy
    upper = comp["upper_threshold"]
    lower = comp["lower_threshold"]

    return {
        # v0: HW crosses up through 50 (oversold→neutral transition)
        "v0_hw_cross_50": _cross_up_value(hw, 50.0),
        # v1: HW crosses up through its own signal while in oversold zone
        "v1_hw_cross_sig_below50": _cross_up(hw, sig) & (hw < 50),
        # v2: MF crosses up through lower_threshold (bear-zone exit)
        "v2_mf_cross_lower": _cross_up(mf, lower),
        # v3: HW oversold-recovery AND MF rising
        "v3_hw_recover_mf_rising": _cross_up(hw, sig) & (hw < 50) & (mf > mf.shift(1)),
        # v4: HW cross 50 OR MF cross lower (union)
        "v4_union": _cross_up_value(hw, 50.0) | _cross_up(mf, lower),
        # v5: stricter — HW recovery + MF above lower (both sides aligned)
        "v5_hw_recover_and_mf_above_lower": _cross_up(hw, sig) & (hw < 50) & (mf > lower),
    }


def score_overlap(
    tv_fires: pd.DatetimeIndex,
    rule_fires: pd.DatetimeIndex,
    tolerance: int,
    bar_index: pd.DatetimeIndex,
) -> dict[str, float]:
    """Match each rule fire to nearest TV fire within ±tolerance bars."""
    if len(rule_fires) == 0:
        return {"n_rule": 0, "n_tv": len(tv_fires), "tp": 0, "fp": 0,
                "fn": len(tv_fires), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tv_pos = bar_index.get_indexer(tv_fires)
    rule_pos = bar_index.get_indexer(rule_fires)
    tv_set = set(tv_pos)

    matched_tv = set()
    tp = 0
    for rp in rule_pos:
        # find any tv fire within tolerance
        hit = None
        for offset in range(-tolerance, tolerance + 1):
            if (rp + offset) in tv_set and (rp + offset) not in matched_tv:
                hit = rp + offset
                break
        if hit is not None:
            matched_tv.add(hit)
            tp += 1

    fp = len(rule_pos) - tp
    fn = len(tv_pos) - len(matched_tv)
    prec = tp / max(len(rule_pos), 1)
    rec = tp / max(len(tv_pos), 1)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "n_rule": int(len(rule_pos)),
        "n_tv": int(len(tv_pos)),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
    }


def evaluate_ticker(ticker: str, path: str) -> pd.DataFrame:
    tv = load_tv_csv(path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    comp = compute_all(ohlcv, DEFAULT_PARAMS)

    j = tv.join(comp).dropna(subset=["money_flow", "hyperwave"])

    rev_mask = j["Reversal Up +"].notna() | j["Reversal Up -"].notna()
    tv_fires = j.index[rev_mask]

    candidates = build_candidates(j)
    rows = []
    for name, mask in candidates.items():
        rule_fires = j.index[mask.fillna(False)]
        scores = score_overlap(tv_fires, rule_fires, TOLERANCE_BARS, j.index)
        rows.append({"ticker": ticker, "rule": name, **scores})
    return pd.DataFrame(rows)


def main() -> None:
    parts = [evaluate_ticker(t, p) for t, p in CSVS.items()]
    df = pd.concat(parts, ignore_index=True)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    print(f"=== Per-ticker overlap (tolerance ±{TOLERANCE_BARS} bars) ===")
    print(df.to_string(index=False))

    print("\n=== Aggregate per rule (sum across tickers) ===")
    agg = df.groupby("rule").agg(
        n_rule=("n_rule", "sum"),
        n_tv=("n_tv", "sum"),
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
    ).reset_index()
    agg["precision"] = agg["tp"] / agg["n_rule"].replace(0, np.nan)
    agg["recall"] = agg["tp"] / agg["n_tv"].replace(0, np.nan)
    agg["f1"] = 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"]).replace(0, np.nan)
    agg = agg.sort_values("f1", ascending=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
