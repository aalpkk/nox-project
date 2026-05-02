"""mb_scanner orchestrator.

Loads the locked nox_intraday_v1 1h master, resamples to 5h/1d/1w/1M
panels, detects all simultaneously-active (LL, LH, HL, HH) quartets per
family/timeframe, classifies state for each independently, emits 0..N
rows per (ticker, family) at as-of (one per active quartet — wide and
nested narrow setups are both reported).

Per-family TF params (pivot_n, max_quartet_span_bars, max_zone_age_bars)
are tuned by hand to produce comparable lookback windows across frequencies:

  5h: ~6 weeks structure / ~4 weeks zone freshness (TV "5 saatlik" aligned)
  1d: ~4 months structure / ~3 months zone freshness
  1w: ~7 months structure / ~5 months zone freshness
  1M: ~1 year structure / ~8 months zone freshness
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import math
import numpy as np
import pandas as pd

from data import intraday_1h

from .pivots import alternating_pivots  # noqa: F401  (re-exported)
from .quartet import find_all_quartets, find_latest_quartet  # noqa: F401
from .resample import per_ticker_panel, to_5h, to_daily, to_monthly, to_weekly
from .schema import OUTPUT_COLUMNS, VERSION as SCHEMA_VERSION, empty_row
from .zones import (
    DEFAULT_INVALIDATION_BUFFER_ATR,
    classify_state,
    find_zone,
    is_invalidated,
    retest_kind_label,
)


@dataclass(frozen=True)
class FamilyParams:
    family: str
    mode: Literal["mb", "bb"]
    frequency: Literal["5h", "1d", "1w", "1M"]
    pivot_n: int
    max_quartet_span_bars: int
    max_zone_age_bars: int


_PARAMS: dict[str, FamilyParams] = {
    "mb_5h": FamilyParams("mb_5h", "mb", "5h", 2, 120, 80),
    "mb_1d": FamilyParams("mb_1d", "mb", "1d", 2, 80, 60),
    "mb_1w": FamilyParams("mb_1w", "mb", "1w", 2, 30, 20),
    "mb_1M": FamilyParams("mb_1M", "mb", "1M", 2, 12, 8),
    "bb_5h": FamilyParams("bb_5h", "bb", "5h", 2, 120, 80),
    "bb_1d": FamilyParams("bb_1d", "bb", "1d", 2, 80, 60),
    "bb_1w": FamilyParams("bb_1w", "bb", "1w", 2, 30, 20),
    "bb_1M": FamilyParams("bb_1M", "bb", "1M", 2, 12, 8),
}

OUT_DIR = Path("output")


# ---------------------------------------------------------------- indicators

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _vol_sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"].rolling(n).mean()


def _sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["close"].rolling(n).mean()


# ---------------------------------------------------------------- core

def _bar_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp (tz-aware or naive) to a naive date Timestamp."""
    if ts.tz is not None:
        return ts.tz_convert("Europe/Istanbul").normalize().tz_localize(None)
    return ts.normalize()


def _build_row(
    *,
    ticker: str,
    fam: FamilyParams,
    df: pd.DataFrame,
    asof_idx: int,
    quartet: dict,
    zone: dict,
    state: str,
    retest_idx: Optional[int],
    touches: int,
    deepest_low: float,
    atr: pd.Series,
    vol_sma: pd.Series,
    sma20: pd.Series,
    quartet_rank: int = 0,
    n_active_quartets: int = 1,
) -> dict:
    asof_ts = pd.Timestamp(df.index[asof_idx])
    asof_close = float(df["close"].iat[asof_idx])
    asof_volume = float(df["volume"].iat[asof_idx])

    atr_now = float(atr.iat[asof_idx]) if pd.notna(atr.iat[asof_idx]) else float("nan")
    vol_sma_now = float(vol_sma.iat[asof_idx]) if pd.notna(vol_sma.iat[asof_idx]) else float("nan")
    sma_now = float(sma20.iat[asof_idx]) if pd.notna(sma20.iat[asof_idx]) else float("nan")

    zone_high = zone["zone_high"]
    zone_low = zone["zone_low"]
    zone_width = zone_high - zone_low
    zone_width_pct = zone_width / asof_close if asof_close > 0 else float("nan")
    zone_width_atr = zone_width / atr_now if atr_now and atr_now > 0 else float("nan")
    zone_age_bars = int(asof_idx - zone["origin_idx"])

    lh_price = quartet["lh_price"]
    bos_distance_pct = (asof_close - lh_price) / lh_price if lh_price > 0 else float("nan")
    bos_distance_atr = (asof_close - lh_price) / atr_now if atr_now and atr_now > 0 else float("nan")

    retest_depth_atr = float("nan")
    if math.isfinite(deepest_low) and atr_now and atr_now > 0:
        retest_depth_atr = (zone_high - deepest_low) / atr_now
    rkind = retest_kind_label(retest_depth_atr) if math.isfinite(retest_depth_atr) else ""

    structural_low = quartet["hl_price"]
    invalidation = structural_low - DEFAULT_INVALIDATION_BUFFER_ATR * (atr_now if atr_now and atr_now > 0 else 0.0)
    initial_risk_pct = (asof_close - invalidation) / asof_close if asof_close > 0 else float("nan")

    vol_ratio = (asof_volume / vol_sma_now) if vol_sma_now and vol_sma_now > 0 else float("nan")
    close_vs_sma20 = (asof_close / sma_now - 1.0) if sma_now and sma_now > 0 else float("nan")

    row = empty_row()
    row.update({
        "ticker": ticker,
        "setup_family": fam.family,
        "data_frequency": fam.frequency,
        "signal_state": state,
        "as_of_ts": asof_ts,
        "bar_date": _bar_date(asof_ts),
        "quartet_rank": int(quartet_rank),
        "n_active_quartets": int(n_active_quartets),
        "ll_idx": int(quartet["ll_idx"]),
        "ll_bar_date": _bar_date(pd.Timestamp(df.index[quartet["ll_idx"]])),
        "ll_price": float(quartet["ll_price"]),
        "lh_idx": int(quartet["lh_idx"]),
        "lh_bar_date": _bar_date(pd.Timestamp(df.index[quartet["lh_idx"]])),
        "lh_price": float(quartet["lh_price"]),
        "hl_idx": int(quartet["hl_idx"]),
        "hl_bar_date": _bar_date(pd.Timestamp(df.index[quartet["hl_idx"]])),
        "hl_price": float(quartet["hl_price"]),
        "hh_idx": int(quartet["hh_idx"]),
        "hh_bar_date": _bar_date(pd.Timestamp(df.index[quartet["hh_idx"]])),
        "hh_close": float(quartet["hh_close"]),
        "zone_origin_idx": int(zone["origin_idx"]),
        "zone_origin_bar_date": _bar_date(pd.Timestamp(df.index[zone["origin_idx"]])),
        "zone_high": float(zone_high),
        "zone_low": float(zone_low),
        "zone_width_pct": float(zone_width_pct) if pd.notna(zone_width_pct) else None,
        "zone_width_atr": float(zone_width_atr) if pd.notna(zone_width_atr) else None,
        "zone_age_bars": zone_age_bars,
        "touches_into_zone": int(touches),
        "deepest_low_after_hh": float(deepest_low) if math.isfinite(deepest_low) else None,
        "retest_depth_atr": float(retest_depth_atr) if math.isfinite(retest_depth_atr) else None,
        "retest_kind": rkind,
        "retest_idx": int(retest_idx) if retest_idx is not None else None,
        "retest_bar_date": (_bar_date(pd.Timestamp(df.index[retest_idx])) if retest_idx is not None else None),
        "asof_close": asof_close,
        "asof_volume": asof_volume,
        "bos_distance_pct": float(bos_distance_pct) if pd.notna(bos_distance_pct) else None,
        "bos_distance_atr": float(bos_distance_atr) if pd.notna(bos_distance_atr) else None,
        "atr_14": float(atr_now) if pd.notna(atr_now) else None,
        "atr_pct": float(atr_now / asof_close) if (atr_now and asof_close > 0) else None,
        "vol_ratio_20": float(vol_ratio) if pd.notna(vol_ratio) else None,
        "close_vs_sma20": float(close_vs_sma20) if pd.notna(close_vs_sma20) else None,
        "structural_invalidation_low": float(invalidation),
        "initial_risk_pct": float(initial_risk_pct) if pd.notna(initial_risk_pct) else None,
        "schema_version": SCHEMA_VERSION,
    })
    return row


def _detect_for_panel(
    *,
    ticker: str,
    fam: FamilyParams,
    df: pd.DataFrame,
    asof_idx: int,
) -> list[dict]:
    """Run quartet+zone+state pipeline on one ticker's resampled panel.

    Returns 0..N rows — one per simultaneously-active quartet that passes
    the zone-freshness and invalidation gates.
    """
    if asof_idx < max(fam.pivot_n + 4, 30):
        return []

    quartets = find_all_quartets(
        df, n=fam.pivot_n, end_idx=asof_idx, mode=fam.mode,
        max_quartet_span_bars=fam.max_quartet_span_bars,
    )
    if not quartets:
        return []

    atr = _atr(df, n=14)
    vol_sma = _vol_sma(df, n=20)
    sma20 = _sma(df, n=20)

    survivors: list[tuple[dict, dict, str, Optional[int], int, float]] = []
    for q in quartets:
        zone = find_zone(df, q["lh_idx"])
        if zone is None:
            continue
        if asof_idx - zone["origin_idx"] > fam.max_zone_age_bars:
            continue
        if is_invalidated(
            df, hh_idx=q["hh_idx"], asof_idx=asof_idx,
            structural_low=q["hl_price"], atr=atr,
        ):
            continue
        state, retest_idx, touches, deepest_low = classify_state(
            df, hh_idx=q["hh_idx"], asof_idx=asof_idx,
            zone_high=zone["zone_high"], zone_low=zone["zone_low"],
            atr=atr, vol_sma=vol_sma,
        )
        survivors.append((q, zone, state, retest_idx, touches, deepest_low))

    if not survivors:
        return []

    n_active = len(survivors)
    rows: list[dict] = []
    for rank, (q, zone, state, retest_idx, touches, deepest_low) in enumerate(survivors):
        rows.append(_build_row(
            ticker=ticker, fam=fam, df=df, asof_idx=asof_idx,
            quartet=q, zone=zone, state=state,
            retest_idx=retest_idx, touches=touches, deepest_low=deepest_low,
            atr=atr, vol_sma=vol_sma, sma20=sma20,
            quartet_rank=rank, n_active_quartets=n_active,
        ))
    return rows


def _resolve_asof(
    df: pd.DataFrame,
    asof: Optional[pd.Timestamp],
) -> int:
    """Locate latest bar idx with index ≤ asof. Returns -1 if none."""
    if asof is None:
        return len(df) - 1
    asof_ts = pd.Timestamp(asof)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None and asof_ts.tz is None:
        asof_ts = asof_ts.tz_localize(idx.tz)
    elif idx.tz is None and asof_ts.tz is not None:
        asof_ts = asof_ts.tz_convert("Europe/Istanbul").tz_localize(None)
    mask = idx <= asof_ts
    if not mask.any():
        return -1
    return int(np.flatnonzero(mask)[-1])


# ---------------------------------------------------------------- public

def scan(
    *,
    families: Iterable[str],
    tickers: Iterable[str] | None = None,
    asof: str | pd.Timestamp | None = None,
    min_coverage: float = 0.0,
    out_dir: str | Path = OUT_DIR,
    write_parquet: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run mb_scanner across requested families.

    Returns {family: DataFrame}. If write_parquet=True, also writes one parquet
    per family to `out_dir / mb_scanner_<family>.parquet`.
    """
    fam_keys = list(families)
    bad = [f for f in fam_keys if f not in _PARAMS]
    if bad:
        raise ValueError(f"unknown families: {bad}; valid={list(_PARAMS)}")

    bars = intraday_1h.load_intraday(
        tickers=list(tickers) if tickers is not None else None,
        start=None, end=None, min_coverage=min_coverage,
    )
    if bars.empty:
        return {f: pd.DataFrame(columns=OUTPUT_COLUMNS) for f in fam_keys}

    # Resample once per frequency on demand.
    needed_freqs = {_PARAMS[f].frequency for f in fam_keys}
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    if "5h" in needed_freqs:
        panels["5h"] = per_ticker_panel(to_5h(bars), "ts_istanbul")
    if "1d" in needed_freqs:
        panels["1d"] = per_ticker_panel(to_daily(bars), "date")
    if "1w" in needed_freqs:
        panels["1w"] = per_ticker_panel(to_weekly(bars), "week_end")
    if "1M" in needed_freqs:
        panels["1M"] = per_ticker_panel(to_monthly(bars), "month_end")

    asof_ts = pd.Timestamp(asof) if asof is not None else None

    results: dict[str, list[dict]] = {f: [] for f in fam_keys}
    for f in fam_keys:
        fam = _PARAMS[f]
        panel_map = panels[fam.frequency]
        for ticker, df in panel_map.items():
            asof_idx = _resolve_asof(df, asof_ts)
            if asof_idx < 0:
                continue
            rows = _detect_for_panel(
                ticker=ticker, fam=fam, df=df, asof_idx=asof_idx,
            )
            if rows:
                results[f].extend(rows)

    out_map: dict[str, pd.DataFrame] = {}
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for f in fam_keys:
        rows = results[f]
        df = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)) if rows else pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        out_map[f] = df
        if write_parquet:
            target = out_dir_p / f"mb_scanner_{f}.parquet"
            df.to_parquet(target, index=False)
    return out_map
