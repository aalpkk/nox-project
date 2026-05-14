"""Decision Engine v0 — scanner output → normalized events.

Six adapters, one shared schema. Reads source files, returns DataFrame
matching schema.EVENT_COLUMNS. No scoring, no ranking.

Locked spec: memory/decision_engine_v0_spec.md §Normalized event schema.
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .phase import map_phase
from .risk import derive_risk
from .schema import EVENT_COLUMNS, canonical_family

# ─── helpers ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def _empty_event() -> dict:
    return {col: None for col in EVENT_COLUMNS}


def _to_date(x):
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


def _val(row, key, default=None):
    if key not in row.index:
        return default
    v = row[key]
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _load_regime() -> pd.DataFrame:
    path = OUTPUT / "regime_labels_daily_rdp_v1.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.sort_values("date")[["date", "regime"]]


def _attach_regime(events: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Backward-as-of carry; tag stale_days when event_date > max(regime.date)."""
    if events.empty:
        events["regime"] = pd.Series(dtype="object")
        events["regime_stale_days"] = pd.Series(dtype="object")
        return events
    out = events.drop(columns=[c for c in ("regime", "regime_stale_days") if c in events.columns]).copy()
    out["date_dt"] = pd.to_datetime(out["date"])
    rg = regime.rename(columns={"date": "regime_date"}).copy()
    rg["regime_date_dt"] = pd.to_datetime(rg["regime_date"])
    rg = rg.sort_values("regime_date_dt")
    out = out.sort_values("date_dt")
    merged = pd.merge_asof(
        out, rg, left_on="date_dt", right_on="regime_date_dt", direction="backward"
    )
    merged["regime_stale_days"] = (
        merged["date_dt"] - merged["regime_date_dt"]
    ).dt.days
    merged.loc[merged["regime"].isna(), "regime_stale_days"] = None
    merged["regime"] = merged["regime"].fillna("unknown")
    merged = merged.drop(columns=["regime_date", "regime_date_dt", "date_dt"])
    return merged


# ─── adapter: mb_scanner ──────────────────────────────────────────────────

_MB_FAMILIES = [
    ("mb_5h", "5h"),
    ("mb_1d", "1d"),
    ("mb_1w", "1w"),
    ("mb_1M", "1M"),
    ("bb_5h", "5h"),
    ("bb_1d", "1d"),
    ("bb_1w", "1w"),
    ("bb_1M", "1M"),
]


def adapt_mb_scanner() -> pd.DataFrame:
    rows = []
    for fam, tf in _MB_FAMILIES:
        path = OUTPUT / f"mb_scanner_{fam}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            state = str(_val(r, "signal_state", ""))
            family_key = f"{fam}__{state}"  # e.g. mb_5h__above_mb
            entry_ref = _val(r, "asof_close")
            stop_ref = _val(r, "structural_invalidation_low")
            atr = _val(r, "atr_14")
            risk_pct, risk_atr = derive_risk(
                entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
            )
            ev = _empty_event()
            ev.update(
                date=_to_date(_val(r, "bar_date")),
                ticker=str(_val(r, "ticker", "")),
                source="mb_scanner",
                family=canonical_family(family_key),
                state=state,
                phase=map_phase(source="mb_scanner", family=family_key, state=state),
                timeframe=tf,
                direction="long",
                raw_signal_present=True,
                entry_ref=float(entry_ref) if entry_ref is not None else None,
                stop_ref=float(stop_ref) if stop_ref is not None else None,
                risk_pct=risk_pct,
                risk_atr=risk_atr,
                extension_atr=_val(r, "bos_distance_atr"),
                liquidity_score=None,  # mb_scanner doesn't emit liquidity_score
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=None,
                fill_assumption="unresolved",
                bar_timestamp=_val(r, "as_of_ts"),
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── adapter: horizontal_base ─────────────────────────────────────────────


def _latest_horizontal_base() -> Path | None:
    paths = sorted(OUTPUT.glob("horizontal_base_live_*.parquet"))
    return paths[-1] if paths else None


def adapt_horizontal_base() -> pd.DataFrame:
    path = _latest_horizontal_base()
    if path is None:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        state = str(_val(r, "signal_state", ""))
        family_key = f"horizontal_base__{state}"
        entry_ref = _val(r, "entry_reference_price") or _val(r, "family__trigger_level")
        stop_ref = _val(r, "invalidation_level")
        atr = _val(r, "common__atr_14")
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
        )
        ev = _empty_event()
        ev.update(
            date=_to_date(_val(r, "bar_date")),
            ticker=str(_val(r, "ticker", "")),
            source="horizontal_base",
            family=family_key,
            state=state,
            phase=map_phase(source="horizontal_base", family=family_key, state=state),
            timeframe="1d",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if entry_ref is not None else None,
            stop_ref=float(stop_ref) if stop_ref is not None else None,
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            extension_atr=_val(r, "common__extension_from_trigger"),
            liquidity_score=_val(r, "common__liquidity_score"),
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=None,
            fill_assumption="unresolved",
            bar_timestamp=_val(r, "as_of_ts"),
        )
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── adapter: nyxexpansion (derive triggers from dataset) ─────────────────


def adapt_nyxexpansion() -> pd.DataFrame:
    path = OUTPUT / "nyxexp_dataset_v4.parquet"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    latest = df["date"].max()
    today = df[df["date"] == latest].copy()
    fired = today[
        (today["close"] > today["prior_high_20"])
        & (today["rvol"] >= 1.5)
        & (today["close_loc"] >= 0.70)
    ]
    rows = []
    for _, r in fired.iterrows():
        entry_ref = _val(r, "close")
        atr = _val(r, "atr_14")
        ets = _val(r, "entry_to_stop_atr")
        stop_ref = None
        if entry_ref is not None and atr is not None and ets is not None:
            try:
                stop_ref = float(entry_ref) - float(atr) * float(ets)
            except (TypeError, ValueError):
                stop_ref = None
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
        )
        ev = _empty_event()
        ev.update(
            date=_to_date(_val(r, "date")),
            ticker=str(_val(r, "ticker", "")),
            source="nyxexpansion",
            family="nyxexpansion__triggerA",
            state="trigger",
            phase="trigger",
            timeframe="1d",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if entry_ref is not None else None,
            stop_ref=stop_ref,
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            extension_atr=_val(r, "dist_above_trigger_atr"),
            liquidity_score=None,
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=None,
            fill_assumption="unresolved",
        )
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── adapter: nyxmomentum ─────────────────────────────────────────────────


def _latest_nyxmomentum_date() -> str | None:
    live_dir = OUTPUT / "nyxmomentum" / "live"
    if not live_dir.exists():
        return None
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_(m0|v5)_picks\.csv$")
    dates = []
    for f in live_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            dates.append(m.group(1))
    if not dates:
        return None
    return max(dates)


def adapt_nyxmomentum() -> pd.DataFrame:
    asof = _latest_nyxmomentum_date()
    if asof is None:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for variant in ("m0", "v5"):
        p = OUTPUT / "nyxmomentum" / "live" / f"{asof}_{variant}_picks.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            ev = _empty_event()
            ev.update(
                date=_to_date(asof),
                ticker=str(_val(r, "ticker", "")),
                source="nyxmomentum",
                family=f"nyxmomentum__{variant}",
                state="strength_top_decile",
                phase="strength_context",
                timeframe="1d",
                direction="long",
                raw_signal_present=True,
                entry_ref=None,  # no entry/stop in picks csv
                stop_ref=None,
                risk_pct=None,
                risk_atr=None,
                extension_atr=None,
                liquidity_score=None,
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=_val(r, "score"),  # passed-through, NOT consumed
                fill_assumption="unresolved",
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── adapter: nox_rt_daily ────────────────────────────────────────────────


def _latest_nox_v3_csv(prefix: str) -> Path | None:
    paths = sorted(OUTPUT.glob(f"{prefix}_*.csv"))
    return paths[-1] if paths else None


def adapt_nox_rt_daily() -> pd.DataFrame:
    path = _latest_nox_v3_csv("nox_v3_signals")
    if path is None:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    # keep only the latest signal_date in the file (live-style snapshot)
    df = pd.read_csv(path, parse_dates=["pivot_date", "signal_date"])
    df = df[df["signal"] == "PIVOT_AL"]
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    latest = df["signal_date"].max()
    today = df[df["signal_date"] == latest]
    rows = []
    for _, r in today.iterrows():
        entry_ref = _val(r, "close")
        stop_ref = _val(r, "pivot_price")
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=None
        )
        ev = _empty_event()
        ev.update(
            date=_to_date(latest),
            ticker=str(_val(r, "ticker", "")),
            source="nox_rt_daily",
            family="nox_rt_daily__pivot_al",
            state="pivot_al",
            phase="trigger",
            timeframe="1d",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if entry_ref is not None else None,
            stop_ref=float(stop_ref) if stop_ref is not None else None,
            risk_pct=risk_pct,
            risk_atr=risk_atr,  # no ATR available here
            extension_atr=_val(r, "delta_pct"),  # %; not ATR — leave as nullable proxy
            liquidity_score=None,
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=_val(r, "rg_score"),
            fill_assumption="unresolved",
        )
        # delta_pct is in %, not ATR — null it to avoid mis-typed extension gate
        ev["extension_atr"] = None
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── adapter: nox_weekly ──────────────────────────────────────────────────


def adapt_nox_weekly() -> pd.DataFrame:
    path = _latest_nox_v3_csv("nox_v3_signals_weekly")
    if path is None:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_csv(path, parse_dates=["pivot_date", "signal_date"])
    df = df[df["signal"] == "PIVOT_AL"]
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    latest = df["signal_date"].max()
    today = df[df["signal_date"] == latest]
    rows = []
    for _, r in today.iterrows():
        entry_ref = _val(r, "close")
        stop_ref = _val(r, "pivot_price")
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=None
        )
        ev = _empty_event()
        ev.update(
            date=_to_date(latest),
            ticker=str(_val(r, "ticker", "")),
            source="nox_weekly",
            family="nox_weekly__weekly_pivot_al",
            state="pivot_al",
            phase="trigger",
            timeframe="1w",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if entry_ref is not None else None,
            stop_ref=float(stop_ref) if stop_ref is not None else None,
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            extension_atr=None,
            liquidity_score=None,
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=_val(r, "rg_score"),
            fill_assumption="unresolved",
        )
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


# ─── orchestrator ─────────────────────────────────────────────────────────


def build_events() -> pd.DataFrame:
    parts = [
        adapt_mb_scanner(),
        adapt_horizontal_base(),
        adapt_nyxexpansion(),
        adapt_nyxmomentum(),
        adapt_nox_rt_daily(),
        adapt_nox_weekly(),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    # regime backward-as-of carry
    regime = _load_regime()
    df = _attach_regime(df, regime)

    # ensure schema column order + presence
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EVENT_COLUMNS].copy()


__all__ = [
    "build_events",
    "adapt_mb_scanner",
    "adapt_horizontal_base",
    "adapt_nyxexpansion",
    "adapt_nyxmomentum",
    "adapt_nox_rt_daily",
    "adapt_nox_weekly",
]
