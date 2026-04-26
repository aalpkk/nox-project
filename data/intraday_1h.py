"""nox_intraday_v1 — adapter (single truth source for intraday backtest data).

Tüm pipeline'lar (v4C, sbt1700, cte, HB-BRK-v2, ignito, nyxalpha) bu adapter
üzerinden okur. Master parquet, coverage csv, regime labels ve splits.json
**immutable**; pipeline'lar yazmaz, sadece kendi namespace'lerine output üretir
(örn. `output/nyxexp_v4C_intraday_v1_*`).

Bundle (v1 contract — output/dataset_manifest.json sha256 ile pin'li):
  output/extfeed_intraday_1h_3y_master.parquet
    ticker, ts_utc, ts_istanbul, open, high, low, close, volume
  output/extfeed_intraday_coverage.csv
    ticker, n_bars, first_ts, last_ts, span_days, coverage_pct, …
  output/regime_labels_daily.csv
    date, close, regime, sub_regime, window_id, label_source
  output/dataset_splits.json
    train/val/test boundaries + per-split regime breakdown

Regenerate (version bump zorunlu — v1 → v2):
  python tools/define_regime_manual.py
  python tools/define_dataset_splits.py
  python tools/build_dataset_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_VERSION = "nox_intraday_v1"

MASTER = Path("output/extfeed_intraday_1h_3y_master.parquet")
COVERAGE = Path("output/extfeed_intraday_coverage.csv")
REGIME = Path("output/regime_labels_daily.csv")
SPLITS = Path("output/dataset_splits.json")
MANIFEST = Path("output/dataset_manifest.json")

# 3y design window starts 2023-01-02 (first BIST trading day of 2023).
# TV gave a sparse 2021/2022 tail for 17 tickers (1×523 + 16×~252 bars) —
# noise, not a usable evren. Adapter clamps to MIN_DATE by default.
MIN_DATE = "2023-01-02"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"manifest missing: {MANIFEST} — run tools/build_dataset_manifest.py")
    return json.loads(MANIFEST.read_text())


def verify_dataset(strict: bool = False) -> dict:
    """Verify the bundle against the manifest.

    strict=False: cheap check (size match per artifact). Default.
    strict=True : sha256 recompute per artifact (slow on master parquet).

    Returns dict {artifact_key: "ok" | reason}. Raises if any mismatch.
    """
    m = load_manifest()
    results: dict[str, str] = {}
    for a in m["artifacts"]:
        p = Path(a["path"])
        if not p.exists():
            raise FileNotFoundError(f"{a['key']}: missing {p}")
        size = p.stat().st_size
        if size != a["size_bytes"]:
            raise ValueError(
                f"{a['key']}: size {size} != manifest {a['size_bytes']}; "
                f"bundle drifted, regenerate manifest or restore artifact."
            )
        if strict:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != a["sha256"]:
                raise ValueError(
                    f"{a['key']}: sha256 mismatch — file modified after manifest build."
                )
        results[a["key"]] = "ok"
    return results


def load_coverage() -> pd.DataFrame:
    if not COVERAGE.exists():
        raise FileNotFoundError(f"coverage csv missing: {COVERAGE}")
    return pd.read_csv(COVERAGE)


def load_regime() -> pd.DataFrame:
    if not REGIME.exists():
        raise FileNotFoundError(f"regime csv missing: {REGIME}")
    df = pd.read_csv(REGIME)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def eligible_tickers(min_coverage: float = 0.0) -> list[str]:
    """Tickers whose coverage_pct >= threshold."""
    cov = load_coverage()
    return cov.loc[cov["coverage_pct"] >= min_coverage, "ticker"].tolist()


def load_intraday(
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_coverage: float = 0.0,
    with_regime: bool = False,
) -> pd.DataFrame:
    """Load intraday 1h bars with optional filters.

    Parameters
    ----------
    tickers : list of ticker codes (e.g. ["THYAO", "ASELS"]). None = all.
    start, end : inclusive date filters on ts_istanbul (str ISO date or Timestamp).
    min_coverage : per-ticker coverage_pct floor (0.0 = no filter).
                   Tipik kullanım: 0.95 (training/backtest), 0.50 (broader),
                   0.0 (live scan, recent IPOs dahil).
    with_regime : True ise XU100 regime sütununu (bull/bear/choppy/warmup) join eder.

    Returns
    -------
    DataFrame: ticker, ts_utc, ts_istanbul, open, high, low, close, volume
               (+ regime, regime_date if with_regime=True)
               sorted by (ticker, ts_istanbul).
    """
    if not MASTER.exists():
        raise FileNotFoundError(f"master parquet missing: {MASTER}")

    bars = pd.read_parquet(MASTER)
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])

    # Default lower bound: clamp to 3y design window unless caller overrides.
    effective_start = start if start is not None else MIN_DATE

    if min_coverage > 0:
        keep = set(eligible_tickers(min_coverage))
        bars = bars[bars["ticker"].isin(keep)]

    if tickers is not None:
        wanted = set(tickers)
        bars = bars[bars["ticker"].isin(wanted)]

    start_ts = pd.Timestamp(effective_start)
    if start_ts.tz is None and bars["ts_istanbul"].dt.tz is not None:
        start_ts = start_ts.tz_localize(bars["ts_istanbul"].dt.tz)
    bars = bars[bars["ts_istanbul"] >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tz is None and bars["ts_istanbul"].dt.tz is not None:
            end_ts = end_ts.tz_localize(bars["ts_istanbul"].dt.tz)
        if pd.Timestamp(end).normalize() == pd.Timestamp(end):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        bars = bars[bars["ts_istanbul"] <= end_ts]

    bars = bars.sort_values(["ticker", "ts_istanbul"]).reset_index(drop=True)

    if with_regime:
        regime = load_regime()[["date", "regime"]]
        bars["regime_date"] = bars["ts_istanbul"].dt.date
        bars = bars.merge(regime, left_on="regime_date", right_on="date", how="left")
        bars = bars.drop(columns=["date"])
        bars["regime"] = bars["regime"].fillna("warmup")

    return bars


def daily_resample(
    df: pd.DataFrame,
    *,
    tz: str = "Europe/Istanbul",
) -> pd.DataFrame:
    """1h → daily OHLCV per ticker."""
    if df.empty:
        return df
    df = df.copy()
    if df["ts_istanbul"].dt.tz is None:
        df["ts_istanbul"] = df["ts_istanbul"].dt.tz_localize(tz)
    df["date"] = df["ts_istanbul"].dt.date
    agg = (
        df.groupby(["ticker", "date"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
    )
    return agg


def load_splits() -> dict:
    if not SPLITS.exists():
        raise FileNotFoundError(f"splits missing: {SPLITS} — run tools/define_dataset_splits.py")
    return json.loads(SPLITS.read_text())


def signal_entry_prices(
    pairs: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "signal_date",
    hours_tr: tuple[int, ...] = (16, 17),
) -> pd.DataFrame:
    """Per (ticker, date), return open + close prices of selected TR-hour 1h bars.

    Open-time bar convention (verified): bar timestamped HH:00 TR closes at
    (HH+1):00 TR. So:
      hour 16 close → price at 17:00 TR  (== "17:00 bar close" entry; primary)
      hour 17 open  → first print at 17:00 TR  (fallback if h16 bar missing,
                                                e.g. illiquid stock no trades 16-17)
      hour 17 close → price at 18:00 TR  (full-day close, == daily close)

    Returns a wide DataFrame with columns:
      ticker, signal_date, price_h{HH}_open, price_h{HH}_close, … (one pair per requested hour)
    Rows are left-joinable onto `pairs`.
    """
    if pairs.empty:
        return pairs.copy()

    bars = pd.read_parquet(MASTER, columns=["ticker", "ts_istanbul", "open", "close"])
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])
    bars["date"] = bars["ts_istanbul"].dt.date
    bars["hh"] = bars["ts_istanbul"].dt.hour

    wanted = pairs[[ticker_col, date_col]].drop_duplicates().rename(
        columns={ticker_col: "ticker", date_col: "signal_date"}
    )
    wanted["signal_date"] = pd.to_datetime(wanted["signal_date"]).dt.date

    sub = bars[bars["hh"].isin(hours_tr)].merge(
        wanted, left_on=["ticker", "date"], right_on=["ticker", "signal_date"],
        how="inner",
    )

    out = wanted.copy()
    for hh in hours_tr:
        cols = {"open": f"price_h{hh:02d}_open", "close": f"price_h{hh:02d}_close"}
        per_hour = (sub[sub["hh"] == hh][["ticker", "signal_date", "open", "close"]]
                    .rename(columns=cols))
        out = out.merge(per_hour, on=["ticker", "signal_date"], how="left")
    return out


__all__ = [
    "DATASET_VERSION",
    "MIN_DATE",
    "MASTER",
    "COVERAGE",
    "REGIME",
    "SPLITS",
    "MANIFEST",
    "load_manifest",
    "verify_dataset",
    "load_splits",
    "load_coverage",
    "load_regime",
    "eligible_tickers",
    "load_intraday",
    "daily_resample",
    "signal_entry_prices",
]
