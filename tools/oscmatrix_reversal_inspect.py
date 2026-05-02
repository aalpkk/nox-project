"""Inspect what our component values look like on bars where TV fires Reversal Up."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def inspect(ticker: str, path: str) -> pd.DataFrame:
    tv = load_tv_csv(path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS)

    j = tv.join(ours)

    # Bars where TV fires reversal up (either + or -)
    rev_mask = j["Reversal Up +"].notna() | j["Reversal Up -"].notna()
    fires = j[rev_mask].copy()
    fires["ticker"] = ticker
    fires["kind"] = fires.apply(
        lambda r: "+_only" if pd.notna(r["Reversal Up +"]) and pd.isna(r["Reversal Up -"])
        else ("-_only" if pd.notna(r["Reversal Up -"]) and pd.isna(r["Reversal Up +"])
        else "both"),
        axis=1,
    )
    cols = ["ticker", "kind", "money_flow", "lower_threshold", "upper_threshold",
            "hyperwave", "Bullish Overflow", "Bearish Overflow",
            "Upper Confluence Value", "Lower Confluence Value", "Confluence Meter Value"]
    return fires[cols].reset_index()


def main() -> None:
    parts = []
    for ticker, path in CSVS.items():
        parts.append(inspect(ticker, path))
    df = pd.concat(parts, ignore_index=True)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)

    print(f"=== TV Reversal-Up firings across 4 tickers (n={len(df)}) ===\n")
    print(df.to_string(index=False))

    print("\n=== Aggregate stats at fire moments ===")
    num_cols = ["money_flow", "lower_threshold", "upper_threshold", "hyperwave",
                "Lower Confluence Value", "Confluence Meter Value"]
    print(df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).to_string())

    # Key derived: MF vs lower_threshold
    df["mf_above_lower"] = df["money_flow"] > df["lower_threshold"]
    df["mf_below_50"] = df["money_flow"] < 50
    df["hw_below_50"] = df["hyperwave"] < 50
    df["lower_conf_active"] = df["Lower Confluence Value"] > 0.5
    print("\n=== Boolean states at fire moments ===")
    for c in ["mf_above_lower", "mf_below_50", "hw_below_50", "lower_conf_active"]:
        print(f"  {c}: {df[c].mean():.1%} of fires")


if __name__ == "__main__":
    main()
