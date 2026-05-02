"""Scanner event adapters → unified roster.

Each adapter returns a DataFrame with at least:
  ticker, event_date, scanner, family, slice_tags

`scanner` is the top-level scanner name (e.g., "horizontal_base").
`family` is the within-scanner sub-cohort (e.g., "trigger", "retest_bounce").
`slice_tags` is a JSON-string dict of any extra cohort tags (body_class, retest_kind, …).

event_date is normalized to a pandas Timestamp at midnight (no tz).

Coverage filter: events outside HW source window (2023-04-27 → 2026-04-30) are dropped.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd

HW_RANGE_START = pd.Timestamp("2023-04-27")
HW_RANGE_END = pd.Timestamp("2026-04-30")


def _normalize(df: pd.DataFrame, scanner: str) -> pd.DataFrame:
    df = df.copy()
    df["scanner"] = scanner
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize().dt.tz_localize(None)
    df = df[(df["event_date"] >= HW_RANGE_START) & (df["event_date"] <= HW_RANGE_END)]
    if "slice_tags" not in df.columns:
        df["slice_tags"] = "{}"
    df["slice_tags"] = df["slice_tags"].apply(
        lambda v: v if isinstance(v, str) else json.dumps(v or {}, sort_keys=True)
    )
    return df[["ticker", "event_date", "scanner", "family", "slice_tags"]].reset_index(drop=True)


# ---------- 1. horizontal_base ----------
def horizontal_base(path: str = "output/horizontal_base_event_v1.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = df[df["signal_state"].isin(["trigger", "retest_bounce"])].copy()
    keep["event_date"] = keep["bar_date"]
    keep["family"] = keep["signal_state"]
    tag_cols = [c for c in ["body_class", "retest_kind", "tier"] if c in keep.columns]
    if tag_cols:
        keep["slice_tags"] = keep[tag_cols].apply(
            lambda r: json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, sort_keys=True),
            axis=1,
        )
    return _normalize(keep, "horizontal_base")


# ---------- 2. nyxmomentum ----------
def nyxmomentum(picks_dir: str = "output/nyxmomentum/live") -> pd.DataFrame:
    rows = []
    for envelope in ("v5", "m0"):
        for f in sorted(glob.glob(os.path.join(picks_dir, f"*_{envelope}_picks.csv"))):
            stem = os.path.basename(f).split("_")[0]  # YYYY-MM-DD
            try:
                rb_date = pd.Timestamp(stem)
            except Exception:
                continue
            df = pd.read_csv(f)
            for _, r in df.iterrows():
                rows.append({
                    "ticker": r["ticker"],
                    "event_date": rb_date,
                    "family": envelope,
                    "slice_tags": json.dumps({"rank": int(r["rank"])}, sort_keys=True),
                })
    if not rows:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    df = pd.DataFrame(rows)
    return _normalize(df, "nyxmomentum")


# ---------- 3. nyxexpansion winmag (TriggerA) ----------
def nyxexpansion(path: str = "output/nyxexp_dataset_v4.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    trig = df[
        (df["close"] > df["prior_high_20"])
        & (df["rvol"] >= 1.5)
        & (df["close_loc"] >= 0.70)
    ].copy()
    trig["event_date"] = trig["date"]
    trig["family"] = "triggerA"
    tag_cols = [c for c in ["xu_regime", "split"] if c in trig.columns]
    if tag_cols:
        trig["slice_tags"] = trig[tag_cols].apply(
            lambda r: json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, sort_keys=True),
            axis=1,
        )
    return _normalize(trig, "nyxexpansion")


# ---------- 4. mb_scanner (8 family) ----------
_MB_FAMILIES = [
    ("mb", "5h"), ("mb", "1d"), ("mb", "1w"), ("mb", "1M"),
    ("bb", "5h"), ("bb", "1d"), ("bb", "1w"), ("bb", "1M"),
]


def mb_scanner(events_dir: str = "output") -> pd.DataFrame:
    parts = []
    for mode, tf in _MB_FAMILIES:
        f = os.path.join(events_dir, f"mb_scanner_events_{mode}_{tf}.parquet")
        if not Path(f).exists():
            continue
        df = pd.read_parquet(f)
        df = df.copy()
        df["event_date"] = pd.to_datetime(df["event_bar_date"])
        df["family"] = f"{mode}_{tf}__{df['event_type']}"
        df["family"] = df.apply(lambda r: f"{mode}_{tf}__{r['event_type']}", axis=1)
        tag_cols = [c for c in ["retest_kind_at_event", "quartet_id"] if c in df.columns]
        if tag_cols:
            df["slice_tags"] = df[tag_cols].apply(
                lambda r: json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, sort_keys=True),
                axis=1,
            )
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    df = pd.concat(parts, ignore_index=True)
    return _normalize(df, "mb_scanner")


# ---------- 5. NOX RT daily ----------
def nox_rt_daily(out_glob: str = "output/nox_v3_signals_2*.csv") -> pd.DataFrame:
    files = [f for f in sorted(glob.glob(out_glob)) if "weekly" not in f]
    if not files:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty or "signal" not in d.columns:
            continue
        d = d[d["signal"] == "PIVOT_AL"].copy()
        if d.empty:
            continue
        d["event_date"] = pd.to_datetime(d["signal_date"])
        d["family"] = "pivot_al"
        tag_cols = [c for c in ["severity", "wl_status", "tb_stage", "tb_prep"] if c in d.columns]
        if tag_cols:
            d["slice_tags"] = d[tag_cols].apply(
                lambda r: json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, sort_keys=True),
                axis=1,
            )
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["ticker", "event_date", "family"]).reset_index(drop=True)
    return _normalize(df, "nox_rt_daily")


# ---------- 6. NOX Weekly ----------
def nox_weekly(out_glob: str = "output/nox_v3_signals_weekly_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(out_glob))
    if not files:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty or "signal" not in d.columns:
            continue
        d = d[d["signal"] == "PIVOT_AL"].copy()
        if d.empty:
            continue
        d["event_date"] = pd.to_datetime(d["signal_date"])
        d["family"] = "weekly_pivot_al"
        tag_cols = [c for c in ["trigger_type", "severity", "wl_status", "tb_stage", "tb_prep"] if c in d.columns]
        if tag_cols:
            d["slice_tags"] = d[tag_cols].apply(
                lambda r: json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, sort_keys=True),
                axis=1,
            )
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ticker", "event_date", "scanner", "family", "slice_tags"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["ticker", "event_date", "family"]).reset_index(drop=True)
    return _normalize(df, "nox_weekly")


# ---------- aggregator ----------
def build_all() -> pd.DataFrame:
    parts = [
        horizontal_base(),
        nyxmomentum(),
        nyxexpansion(),
        mb_scanner(),
        nox_rt_daily(),
        nox_weekly(),
    ]
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    roster = build_all()
    print(f"total events: {len(roster):,}")
    print(roster.groupby(["scanner", "family"]).size().to_string())
