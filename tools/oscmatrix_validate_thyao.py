"""Diff oscmatrix output against TradingView THYAO 1D CSV export."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import diff_components, load_tv_csv, to_ohlcv  # noqa: E402

CSV_PATH = (
    "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/"
    "Downloads1/BIST_THYAO, 1D.csv"
)


def main() -> None:
    tv = load_tv_csv(CSV_PATH)
    print(f"Loaded TV CSV: {len(tv)} rows  {tv['ts'].min()}  →  {tv['ts'].max()}")
    print(f"OSCMTRX-relevant columns present: {sorted(c for c in tv.columns if c in {'Money Flow','Upper Money Flow Threshold','Lower Money Flow Threshold','Bullish Overflow','Bearish Overflow','HyperWave'})}")

    ohlcv = to_ohlcv(tv)
    ours = compute_all(ohlcv, DEFAULT_PARAMS)

    diff = diff_components(tv, ours)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== Component diff (TV ground truth − ours) ===")
    print(diff.to_string(index=False))

    # spot sample
    tv_idx = tv.set_index("ts")
    sample_ts = tv_idx.index[100]
    print(f"\n=== Sample row {sample_ts} ===")
    for tv_col, our_col in [
        ("Money Flow", "money_flow"),
        ("Upper Money Flow Threshold", "upper_threshold"),
        ("Lower Money Flow Threshold", "lower_threshold"),
        ("HyperWave", "hyperwave"),
    ]:
        if tv_col in tv_idx.columns and our_col in ours.columns:
            print(f"  {tv_col:<35} TV={tv_idx.loc[sample_ts, tv_col]:>10.4f}   ours={ours.loc[sample_ts, our_col]:>10.4f}")


if __name__ == "__main__":
    main()
