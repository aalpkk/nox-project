"""TradingView CSV ground-truth diff harness.

TV exports columns matching OSCMTRX components verbatim. Map them to our
internal column names and emit per-component RMSE/max-abs/corr.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TV_TO_OURS: dict[str, str] = {
    "Money Flow": "money_flow",
    "Upper Money Flow Threshold": "upper_threshold",
    "Lower Money Flow Threshold": "lower_threshold",
    "Bullish Overflow": "bullish_overflow",
    "Bearish Overflow": "bearish_overflow",
    "HyperWave": "hyperwave",
}


def load_tv_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Europe/Istanbul")
    return df


def to_ohlcv(tv: pd.DataFrame) -> pd.DataFrame:
    return (
        tv[["ts", "open", "high", "low", "close", "Volume"]]
        .rename(columns={"Volume": "volume"})
        .set_index("ts")
    )


def diff_components(tv: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    if "ts" in tv.columns and tv.index.name != "ts":
        tv = tv.set_index("ts")
    rows = []
    for tv_col, our_col in TV_TO_OURS.items():
        if tv_col not in tv.columns or our_col not in ours.columns:
            rows.append({"component": tv_col, "n": 0, "rmse": None, "max_abs": None, "corr": None, "note": "missing"})
            continue
        merged = pd.DataFrame({"a": tv[tv_col], "b": ours[our_col]}).dropna()
        if merged.empty:
            rows.append({"component": tv_col, "n": 0, "rmse": None, "max_abs": None, "corr": None, "note": "no overlap"})
            continue
        d = merged["a"] - merged["b"]
        rows.append(
            {
                "component": tv_col,
                "n": len(merged),
                "rmse": float((d**2).mean() ** 0.5),
                "max_abs": float(d.abs().max()),
                "corr": float(merged["a"].corr(merged["b"])),
                "note": "",
            }
        )
    return pd.DataFrame(rows)
