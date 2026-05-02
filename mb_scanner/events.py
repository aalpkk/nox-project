"""Event-log extraction for mb_scanner backtest (Phase 0).

The main `engine.scan()` emits one row per (ticker, family, active quartet)
at a single as-of timestamp. For backtest we need every state-transition
event in history — i.e. when each quartet was *born* (HH BoS confirmed
with pivot lag), when it first *touched* its zone, and when (if ever) it
produced a *retest_bounce* reclaim.

Output: `output/mb_scanner_events_<family>.parquet` with rows like

    above_mb_birth          → entry bar = max(hh_idx, hl_idx + pivot_n)
    mit_touch_first         → entry bar = first overlap bar after birth
    retest_bounce_first     → entry bar = first reclaim-gate-pass bar

Each row contains the full quartet structure plus event-bar features
(close, ATR, vol_ratio, BoS distance, zone age) so the Phase 1 outcome
generator can join directly without re-touching the engine.

Confirmation lag rationale: pivots are close-only fractals confirmed at
i+n. The HL pivot for a quartet is only visible at HL_idx + pivot_n.
If HH (BoS) fires earlier (HH < HL+n), a live trader could not have
identified the quartet at HH; the earliest *tradeable* birth bar is
max(HH, HL+n). This guards Phase 1 against same-bar look-ahead.

Quartets that are invalidated (close < HL − 0.30·ATR) before mit_touch /
retest events suppress those later events. Quartets aged out beyond
`max_zone_age_bars` also suppress.

Public entry: `extract_events(families, ...) -> dict[str, DataFrame]`.
Driver: `tools/mb_scanner_events_run.py`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data import intraday_1h

from .engine import _PARAMS as FAM_PARAMS, _atr, _bar_date, _sma, _vol_sma, FamilyParams
from .quartet import find_all_quartets
from .resample import per_ticker_panel, to_5h, to_daily, to_monthly, to_weekly
from .zones import (
    DEFAULT_INVALIDATION_BUFFER_ATR,
    find_zone,
    is_retest_bounce_bar,
    retest_kind_label,
)

EVENT_SCHEMA_VERSION = "1.0.0"

EVENT_TYPES = ("above_mb_birth", "mit_touch_first", "retest_bounce_first")

EVENT_OUTPUT_COLUMNS = (
    # identity
    "ticker", "setup_family", "data_frequency", "mode",
    "quartet_id",
    "event_type",
    "event_idx", "event_bar_date", "event_ts",
    # quartet structure
    "ll_idx", "ll_bar_date", "ll_price",
    "lh_idx", "lh_bar_date", "lh_price",
    "hl_idx", "hl_bar_date", "hl_price",
    "hh_idx", "hh_bar_date", "hh_close",
    # zone
    "zone_origin_idx", "zone_origin_bar_date",
    "zone_high", "zone_low",
    "zone_width_pct", "zone_width_atr",
    "zone_age_bars",
    # event-bar context
    "event_close", "event_volume",
    "atr_at_event", "atr_pct_at_event",
    "vol_ratio_20_at_event",
    "bos_distance_pct_at_event", "bos_distance_atr_at_event",
    # retest details (mit_touch_first / retest_bounce_first only)
    "retest_depth_atr", "retest_kind_at_event",
    # cross-quartet population (concurrent quartets for same ticker/family at event_idx)
    "concurrent_quartets",
    # lag diagnostics
    "hh_to_event_lag_bars",
    "pivot_confirm_lag_bars",
    # invalidation level
    "structural_invalidation_low",
    # provenance
    "schema_version",
)


def _qid(ticker: str, q: dict) -> str:
    return f"{ticker}|{q['ll_idx']}|{q['lh_idx']}|{q['hl_idx']}|{q['hh_idx']}"


def _event_row(
    *,
    ticker: str,
    fam: FamilyParams,
    df: pd.DataFrame,
    quartet: dict,
    zone: dict,
    event_type: str,
    event_idx: int,
    atr: pd.Series,
    vol_sma: pd.Series,
    deepest_low_at_event: float,
    concurrent_quartets: int,
) -> dict:
    event_ts = pd.Timestamp(df.index[event_idx])
    event_close = float(df["close"].iat[event_idx])
    event_volume = float(df["volume"].iat[event_idx])
    atr_now = float(atr.iat[event_idx]) if pd.notna(atr.iat[event_idx]) else float("nan")
    vol_sma_now = float(vol_sma.iat[event_idx]) if pd.notna(vol_sma.iat[event_idx]) else float("nan")

    zone_high = zone["zone_high"]
    zone_low = zone["zone_low"]
    zone_width = zone_high - zone_low
    zone_width_pct = zone_width / event_close if event_close > 0 else float("nan")
    zone_width_atr = zone_width / atr_now if atr_now and atr_now > 0 else float("nan")
    zone_age_bars = int(event_idx - zone["origin_idx"])

    lh_price = float(quartet["lh_price"])
    bos_distance_pct = (event_close - lh_price) / lh_price if lh_price > 0 else float("nan")
    bos_distance_atr = (event_close - lh_price) / atr_now if atr_now and atr_now > 0 else float("nan")

    retest_depth_atr = float("nan")
    rkind = ""
    if event_type in ("mit_touch_first", "retest_bounce_first"):
        if math.isfinite(deepest_low_at_event) and atr_now and atr_now > 0:
            retest_depth_atr = (zone_high - deepest_low_at_event) / atr_now
            rkind = retest_kind_label(retest_depth_atr)

    structural_low = float(quartet["hl_price"])
    invalidation = structural_low - DEFAULT_INVALIDATION_BUFFER_ATR * (
        atr_now if atr_now and atr_now > 0 else 0.0
    )

    vol_ratio = (event_volume / vol_sma_now) if vol_sma_now and vol_sma_now > 0 else float("nan")

    hh_idx = int(quartet["hh_idx"])
    hl_idx = int(quartet["hl_idx"])
    hh_to_event = int(event_idx - hh_idx)
    pivot_confirm_lag = int(max(0, hl_idx + fam.pivot_n - hh_idx))

    row: dict = {col: None for col in EVENT_OUTPUT_COLUMNS}
    row.update({
        "ticker": ticker,
        "setup_family": fam.family,
        "data_frequency": fam.frequency,
        "mode": fam.mode,
        "quartet_id": _qid(ticker, quartet),
        "event_type": event_type,
        "event_idx": int(event_idx),
        "event_bar_date": _bar_date(event_ts),
        "event_ts": event_ts,
        "ll_idx": int(quartet["ll_idx"]),
        "ll_bar_date": _bar_date(pd.Timestamp(df.index[quartet["ll_idx"]])),
        "ll_price": float(quartet["ll_price"]),
        "lh_idx": int(quartet["lh_idx"]),
        "lh_bar_date": _bar_date(pd.Timestamp(df.index[quartet["lh_idx"]])),
        "lh_price": float(quartet["lh_price"]),
        "hl_idx": hl_idx,
        "hl_bar_date": _bar_date(pd.Timestamp(df.index[hl_idx])),
        "hl_price": structural_low,
        "hh_idx": hh_idx,
        "hh_bar_date": _bar_date(pd.Timestamp(df.index[hh_idx])),
        "hh_close": float(quartet["hh_close"]),
        "zone_origin_idx": int(zone["origin_idx"]),
        "zone_origin_bar_date": _bar_date(pd.Timestamp(df.index[zone["origin_idx"]])),
        "zone_high": float(zone_high),
        "zone_low": float(zone_low),
        "zone_width_pct": float(zone_width_pct) if pd.notna(zone_width_pct) else None,
        "zone_width_atr": float(zone_width_atr) if pd.notna(zone_width_atr) else None,
        "zone_age_bars": zone_age_bars,
        "event_close": event_close,
        "event_volume": event_volume,
        "atr_at_event": float(atr_now) if pd.notna(atr_now) else None,
        "atr_pct_at_event": float(atr_now / event_close) if (atr_now and event_close > 0) else None,
        "vol_ratio_20_at_event": float(vol_ratio) if pd.notna(vol_ratio) else None,
        "bos_distance_pct_at_event": float(bos_distance_pct) if pd.notna(bos_distance_pct) else None,
        "bos_distance_atr_at_event": float(bos_distance_atr) if pd.notna(bos_distance_atr) else None,
        "retest_depth_atr": float(retest_depth_atr) if math.isfinite(retest_depth_atr) else None,
        "retest_kind_at_event": rkind,
        "concurrent_quartets": int(concurrent_quartets),
        "hh_to_event_lag_bars": hh_to_event,
        "pivot_confirm_lag_bars": pivot_confirm_lag,
        "structural_invalidation_low": float(invalidation),
        "schema_version": EVENT_SCHEMA_VERSION,
    })
    return row


def _walk_quartet(
    *,
    df: pd.DataFrame,
    fam: FamilyParams,
    q: dict,
    zone: dict,
    atr: pd.Series,
    vol_sma: pd.Series,
) -> tuple[Optional[int], Optional[int], Optional[int], float]:
    """Walk forward from quartet birth, return (birth_idx, mit_touch_idx,
    retest_bounce_idx, deepest_low_at_first_retest_or_touch).

    `birth_idx` accounts for HL pivot confirmation lag — earliest tradeable
    bar is max(HH, HL + pivot_n). If birth_idx is past panel end → None.
    """
    n_bars = len(df)
    hh_idx = int(q["hh_idx"])
    hl_idx = int(q["hl_idx"])
    birth_idx = max(hh_idx, hl_idx + fam.pivot_n)
    if birth_idx >= n_bars:
        return None, None, None, float("inf")

    age_cap = min(hh_idx + fam.max_zone_age_bars, n_bars - 1)
    if birth_idx > age_cap:
        return birth_idx, None, None, float("inf")

    structural_low = float(q["hl_price"])
    a_arr = atr.to_numpy()
    c_arr = df["close"].to_numpy()
    l_arr = df["low"].to_numpy()
    h_arr = df["high"].to_numpy()

    zone_high = zone["zone_high"]
    zone_low = zone["zone_low"]
    buffer = DEFAULT_INVALIDATION_BUFFER_ATR

    deepest_low = float("inf")
    first_touch: Optional[int] = None
    first_retest: Optional[int] = None

    # Walk strictly AFTER birth_idx (the birth bar is the BoS confirmation
    # itself; touches/retests must be subsequent bars).
    for i in range(birth_idx + 1, age_cap + 1):
        # invalidation
        ai = a_arr[i] if not math.isnan(a_arr[i]) else 0.0
        if ai > 0 and c_arr[i] < structural_low - buffer * ai:
            break
        # deepest low tracker
        if l_arr[i] < deepest_low:
            deepest_low = float(l_arr[i])
        # zone overlap
        overlap = (l_arr[i] <= zone_high) and (h_arr[i] >= zone_low)
        if overlap:
            if first_touch is None:
                first_touch = i
            if first_retest is None and is_retest_bounce_bar(
                df, idx=i, zone_high=zone_high, atr=atr, vol_sma=vol_sma,
            ):
                first_retest = i
                break  # we only need the first reclaim

    return birth_idx, first_touch, first_retest, deepest_low


def _events_for_panel(
    *,
    ticker: str,
    fam: FamilyParams,
    df: pd.DataFrame,
) -> list[dict]:
    """Extract all events for one ticker on its resampled panel."""
    n_bars = len(df)
    if n_bars < max(fam.pivot_n + 4, 30):
        return []

    quartets = find_all_quartets(
        df, n=fam.pivot_n, end_idx=n_bars - 1, mode=fam.mode,
        max_quartet_span_bars=fam.max_quartet_span_bars,
    )
    if not quartets:
        return []

    atr = _atr(df, n=14)
    vol_sma = _vol_sma(df, n=20)

    # First pass: collect each quartet's events without concurrency counts.
    pending: list[tuple[dict, dict, str, int, float]] = []  # (q, zone, event_type, event_idx, deepest_low_at_event)
    for q in quartets:
        zone = find_zone(df, q["lh_idx"])
        if zone is None:
            continue
        birth_idx, first_touch, first_retest, deepest_low = _walk_quartet(
            df=df, fam=fam, q=q, zone=zone, atr=atr, vol_sma=vol_sma,
        )
        if birth_idx is None:
            continue
        # birth event always emitted (we have a tradeable bar)
        pending.append((q, zone, "above_mb_birth", birth_idx, float("inf")))
        # mit_touch_first only emitted when state was actually `mitigation_touch`
        # at some point — i.e., first overlap bar precedes any reclaim. If
        # first_touch == first_retest, the engine state went directly
        # above_mb → retest_bounce, never through mitigation_touch.
        if first_touch is not None and (first_retest is None or first_touch < first_retest):
            mt_low = float(df["low"].iat[first_touch])
            mt_deepest = min(mt_low, deepest_low) if math.isfinite(deepest_low) else mt_low
            pending.append((q, zone, "mit_touch_first", first_touch, mt_deepest))
        if first_retest is not None:
            pending.append((q, zone, "retest_bounce_first", first_retest, deepest_low))

    if not pending:
        return []

    # Second pass: compute concurrent_quartets per event_idx (count quartets
    # whose alive window [birth_idx, age_cap] covers event_idx).
    quartet_windows: list[tuple[int, int]] = []
    for q, zone, _, _, _ in pending:
        birth = max(int(q["hh_idx"]), int(q["hl_idx"]) + fam.pivot_n)
        age_cap = min(int(q["hh_idx"]) + fam.max_zone_age_bars, n_bars - 1)
        if birth <= age_cap:
            quartet_windows.append((birth, age_cap))

    rows: list[dict] = []
    for q, zone, event_type, event_idx, deepest_low_evt in pending:
        concurrent = sum(1 for b, c in quartet_windows if b <= event_idx <= c)
        rows.append(_event_row(
            ticker=ticker, fam=fam, df=df,
            quartet=q, zone=zone,
            event_type=event_type, event_idx=event_idx,
            atr=atr, vol_sma=vol_sma,
            deepest_low_at_event=deepest_low_evt,
            concurrent_quartets=concurrent,
        ))
    return rows


def extract_events(
    *,
    families: Iterable[str],
    tickers: Iterable[str] | None = None,
    min_coverage: float = 0.0,
    out_dir: str | Path = Path("output"),
    write_parquet: bool = True,
) -> dict[str, pd.DataFrame]:
    """Extract historical event log per family.

    Returns {family: DataFrame}. If write_parquet=True, writes one parquet
    per family to `out_dir / mb_scanner_events_<family>.parquet`.
    """
    fam_keys = list(families)
    bad = [f for f in fam_keys if f not in FAM_PARAMS]
    if bad:
        raise ValueError(f"unknown families: {bad}; valid={list(FAM_PARAMS)}")

    bars = intraday_1h.load_intraday(
        tickers=list(tickers) if tickers is not None else None,
        start=None, end=None, min_coverage=min_coverage,
    )
    if bars.empty:
        return {f: pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS)) for f in fam_keys}

    needed_freqs = {FAM_PARAMS[f].frequency for f in fam_keys}
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    if "5h" in needed_freqs:
        panels["5h"] = per_ticker_panel(to_5h(bars), "ts_istanbul")
    if "1d" in needed_freqs:
        panels["1d"] = per_ticker_panel(to_daily(bars), "date")
    if "1w" in needed_freqs:
        panels["1w"] = per_ticker_panel(to_weekly(bars), "week_end")
    if "1M" in needed_freqs:
        panels["1M"] = per_ticker_panel(to_monthly(bars), "month_end")

    results: dict[str, list[dict]] = {f: [] for f in fam_keys}
    for f in fam_keys:
        fam = FAM_PARAMS[f]
        panel_map = panels[fam.frequency]
        for ticker, df in panel_map.items():
            rows = _events_for_panel(ticker=ticker, fam=fam, df=df)
            if rows:
                results[f].extend(rows)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_map: dict[str, pd.DataFrame] = {}
    for f in fam_keys:
        rows = results[f]
        df = pd.DataFrame(rows, columns=list(EVENT_OUTPUT_COLUMNS)) if rows else pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS))
        out_map[f] = df
        if write_parquet:
            target = out_dir_p / f"mb_scanner_events_{f}.parquet"
            df.to_parquet(target, index=False)
    return out_map
