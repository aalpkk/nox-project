"""Daily-only seed for SBT-1700 signal day detection.

The 17:00 truncation requires 15m intraday data, which Matriks fetches
one (ticker, date) call at a time. To avoid fetching all
ticker × trading-day combinations (millions), we first identify the
candidate (ticker, date) pairs where SBT *might* fire — using the
EOD-complete daily bar — and only fetch 15m for those.

This is intentionally a SUPERSET of the 17:00 candidate set: a daily-bar
breakout is necessary but not sufficient for a 17:00 breakout (the 17:00
close + intraday volume must also clear the gates). The 17:00-truncated
re-detection (sbt1700.signals) prunes false positives.

CLI:
  python -m sbt1700.signal_seed \\
      --master output/ohlcv_10y_fintables_master.parquet \\
      --start 2023-11-15 --end 2026-04-30 \\
      --out output/sbt1700_signal_seed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbt1700.config import (
    SBT_BB_LENGTH,
    SBT_BB_MULT,
    SBT_BB_WIDTH_THRESH,
    SBT_ATR_LENGTH,
    SBT_ATR_SMA_LENGTH,
    SBT_ATR_SQUEEZE_RATIO,
    SBT_MIN_SQUEEZE_BARS,
    SBT_MAX_SQUEEZE_BARS,
    SBT_IMPULSE_ATR_MULT,
    SBT_VOL_SMA_LENGTH,
    SBT_VOL_MULT,
    SBT_HTF_EMA_LENGTH,
)


def _seed_for_ticker(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    df = daily.sort_index().copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    sma = c.rolling(SBT_BB_LENGTH).mean()
    std = c.rolling(SBT_BB_LENGTH).std()
    bbw = (sma + SBT_BB_MULT * std - (sma - SBT_BB_MULT * std)) / sma
    bbw_sma = bbw.rolling(SBT_BB_LENGTH).mean()

    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(SBT_ATR_LENGTH).mean()
    atr_sma = atr.rolling(SBT_ATR_SMA_LENGTH).mean()
    vol_sma = v.rolling(SBT_VOL_SMA_LENGTH).mean()
    htf_ema = c.ewm(span=SBT_HTF_EMA_LENGTH, adjust=False).mean()

    sq_bb = bbw < bbw_sma * SBT_BB_WIDTH_THRESH
    sq_atr_flag = atr < atr_sma * SBT_ATR_SQUEEZE_RATIO
    squeeze = (sq_bb & sq_atr_flag).fillna(False)

    runs = np.zeros(len(squeeze), dtype=int)
    cnt = 0
    for i, s in enumerate(squeeze.values):
        cnt = cnt + 1 if s else 0
        runs[i] = cnt

    df["sq_run"] = runs
    df["sq_run_prev"] = pd.Series(runs, index=df.index).shift(1).fillna(0).astype(int)
    df["atr_prev"] = atr.shift(1)
    df["vol_sma_prev"] = vol_sma.shift(1)
    df["htf_ema_prev"] = htf_ema.shift(1)

    rows: list[dict] = []
    for i in range(1, len(df)):
        sq_prev = int(df["sq_run_prev"].iat[i])
        if not (SBT_MIN_SQUEEZE_BARS <= sq_prev <= SBT_MAX_SQUEEZE_BARS):
            continue

        atr_prev = df["atr_prev"].iat[i]
        vsma = df["vol_sma_prev"].iat[i]
        htf = df["htf_ema_prev"].iat[i]
        if not (np.isfinite(atr_prev) and atr_prev > 0
                and np.isfinite(vsma) and vsma > 0
                and np.isfinite(htf)):
            continue

        sq_start = i - sq_prev
        if sq_start < 0:
            continue
        box_top = float(df["High"].iloc[sq_start:i].max())
        box_bottom = float(df["Low"].iloc[sq_start:i].min())
        if not (box_top > box_bottom > 0):
            continue

        c_eod = float(df["Close"].iat[i])
        v_eod = float(df["Volume"].iat[i])

        # EOD-bar gates (loose): same shape as 17:00 gates but on EOD.
        # We only require the EOD breakout — that's the necessary
        # superset. Volume gate is full-day (not pace-scaled), since
        # we're seeding from the EOD bar.
        if c_eod <= box_top + SBT_IMPULSE_ATR_MULT * atr_prev:
            continue
        if v_eod < SBT_VOL_MULT * vsma:
            continue
        if c_eod <= htf:
            continue

        rows.append({
            "ticker": daily.attrs.get("ticker", ""),
            "date": pd.Timestamp(df.index[i]).date(),
            "squeeze_run_prev": sq_prev,
            "atr_prev": float(atr_prev),
            "box_top": box_top,
            "box_bottom": box_bottom,
            "close_eod": c_eod,
            "vol_eod": v_eod,
        })
    return pd.DataFrame(rows)


def build_seed(daily_master: pd.DataFrame,
               start: str | None = None,
               end: str | None = None) -> pd.DataFrame:
    if daily_master.empty:
        return pd.DataFrame()
    df = daily_master.copy()
    if df.index.name != "Date":
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    chunks: list[pd.DataFrame] = []
    for tk, sub in df.groupby("ticker"):
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        sub.attrs["ticker"] = tk
        s = _seed_for_ticker(sub)
        if not s.empty:
            chunks.append(s)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path,
                    default=Path("output/ohlcv_10y_fintables_master.parquet"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("output/sbt1700_signal_seed.csv"))
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    if master.index.name != "Date":
        if "Date" in master.columns:
            master = master.set_index("Date")
    master.index = pd.to_datetime(master.index).normalize()

    print(f"[seed] master: {master.shape}, "
          f"{master['ticker'].nunique()} tickers")
    seed = build_seed(master, start=args.start, end=args.end)
    print(f"[seed] candidate (ticker, date) pairs: {len(seed)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seed[["ticker", "date"]].to_csv(args.out, index=False)
    print(f"[seed] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
