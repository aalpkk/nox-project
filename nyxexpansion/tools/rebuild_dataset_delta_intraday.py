"""Append today's trigger panel to ``nyxexp_dataset_v4.parquet`` using **17:00 TR
truncated intraday bars** instead of yfinance EOD.

Live counterpart to ``rebuild_dataset_delta.py``: runs at 17:00 TR (BIST still
open) so the live scan job can produce actionable signals BEFORE the 18:00
close. Trigger detection runs against a synthetic daily bar built from the
10:00-17:00 TR 15m window — at decision time only those bars are observable.

Pipeline phase 0' — must run before ``scan_latest`` on the live cron path.

Steps:
  1. Detect target date (CLI ``--date`` or Europe/Istanbul today).
  2. Fetch intraday 15m bars for the FULL master universe via
     ``fetch_intraday_layered`` (Fintables → extfeed → Matriks → yfinance).
  3. Aggregate to truncated daily OHLCV per ticker via
     ``aggregate_truncated_bars`` (10:15..17:00 TR window, 28 bars).
  4. Coverage gate: require ≥ ``--min-coverage`` of universe; otherwise abort
     (early-close day, holiday, system outage).
  5. Patch master OHLCV with truncated bar at target_date (drop-then-append,
     idempotent — the 18:05 cron later overwrites with EOD final bar).
  6. Recompute ``compute_trigger_a_panel`` + ``build_feature_panel``
     + ``compute_labels_on_panel`` over the full panel; pull target_date slice.
  7. Append target_date rows to v4 (drop existing first, idempotent).

Labels at target_date will be NaN (no forward bars yet); this is expected and
matches the EOD pipeline's behaviour for the most recent date.

CLI:
    python -m nyxexpansion.tools.rebuild_dataset_delta_intraday        # today TR
    python -m nyxexpansion.tools.rebuild_dataset_delta_intraday --date 2026-04-24
    python -m nyxexpansion.tools.rebuild_dataset_delta_intraday --skip-fetch  # use cached intraday master
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from nyxexpansion.config import CONFIG  # noqa: E402
from nyxexpansion.trigger import compute_trigger_a_panel  # noqa: E402
from nyxexpansion.features import build_feature_panel  # noqa: E402
from nyxexpansion.labels import compute_labels_on_panel  # noqa: E402
from nyxexpansion.retention.truncate import aggregate_truncated_bars  # noqa: E402
from nyxexpansion.intraday.fetch_layered import fetch_intraday_layered  # noqa: E402
from nyxexpansion.tools.presmoke import (  # noqa: E402
    load_ohlcv_cache, load_xu100, classify_xu100_regime,
)

MASTER_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
V4_PATH = Path("output/nyxexp_dataset_v4.parquet")
INTRADAY_MASTER_PATH = Path("output/nyxexp_intraday_master.parquet")
DEFAULT_MIN_COVERAGE = 0.5

INTRADAY_CACHE_COLUMNS = (
    "ticker", "signal_date", "bar_ts", "date",
    "open", "high", "low", "close", "volume", "quantity",
    "bars_source",
)


def _today_istanbul() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=None)).tz_localize(
        "UTC").tz_convert("Europe/Istanbul").normalize().tz_localize(None)


def _load_intraday_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(INTRADAY_CACHE_COLUMNS))
    df = pd.read_parquet(path)
    for c in INTRADAY_CACHE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _drop_target_from_cache(cache: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    if cache.empty:
        return cache
    target_date_obj = target.date()
    sd = pd.to_datetime(cache["signal_date"]).dt.date
    return cache[sd != target_date_obj].copy()


def _persist_intraday(cache: pd.DataFrame, new_rows: list[dict],
                      out_path: Path) -> pd.DataFrame:
    if not new_rows:
        if not cache.empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(out_path, index=False)
        return cache
    new_df = pd.DataFrame(new_rows)
    for c in INTRADAY_CACHE_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = pd.NA
    new_df = new_df[list(INTRADAY_CACHE_COLUMNS)]
    combined = pd.concat([cache, new_df], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return combined


def _fetch_intraday_full_universe(
    universe: list[str],
    target: pd.Timestamp,
    intraday_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the 4-tier fetcher for full universe and persist to intraday master.

    Returns (refreshed_cache, source_breakdown).
    """
    cache = _load_intraday_cache(intraday_path)
    cache = _drop_target_from_cache(cache, target)

    print(f"  [fetch] launching layered fetcher for {len(universe)} tickers")
    t0 = time.time()
    results = fetch_intraday_layered(universe, target)

    new_rows: list[dict] = []
    breakdown: dict[str, int] = {}
    for tk, res in results.items():
        new_rows.extend(res.rows)
        key = res.bars_source or "missing"
        breakdown[key] = breakdown.get(key, 0) + 1

    cache = _persist_intraday(cache, new_rows, intraday_path)
    elapsed = time.time() - t0
    print(f"  [fetch] {elapsed:.1f}s — {sum(breakdown.values())} tickers, "
          f"breakdown: {breakdown}")
    return cache, breakdown


def _patch_master(master: pd.DataFrame, truncated: pd.DataFrame,
                  target: pd.Timestamp) -> pd.DataFrame:
    """Drop target_date rows from master, append truncated bar, sort.

    ``truncated`` columns: ticker, signal_date, Open, High, Low, Close, Volume,
    n_bars, last_bar_ts_tr (output of aggregate_truncated_bars).
    Master columns: Open, High, Low, Close, Volume, ticker, Adj Close.
    """
    keep = master[master.index != target].copy()

    delta = truncated.copy()
    delta = delta[delta["signal_date"] == target]
    if delta.empty:
        return master

    delta = delta.set_index(pd.DatetimeIndex(
        [pd.Timestamp(target)] * len(delta), name=master.index.name or "Date"
    ))
    delta = delta.drop(columns=["signal_date"])
    if "Adj Close" not in delta.columns:
        delta["Adj Close"] = np.nan
    for col in master.columns:
        if col not in delta.columns:
            delta[col] = np.nan
    delta = delta[list(master.columns)]

    out = pd.concat([keep, delta]).sort_index()
    out.index.name = master.index.name or "Date"
    return out


def _build_target_panel(target: pd.Timestamp, xu_close,
                        master_path: Path) -> pd.DataFrame:
    """Run trigger → features → labels for full panel; return target_date slice.

    Mirror of rebuild_dataset_delta._build_target_panel — same call chain so the
    output schema is bit-identical to the EOD path. Labels at target will be
    NaN since no forward bars exist yet.
    """
    data = load_ohlcv_cache(str(master_path))
    print(f"  reload master as dict-of-DataFrames: {len(data)} tickers")

    panel = compute_trigger_a_panel(data, CONFIG.trigger)
    panel["date"] = pd.to_datetime(panel["date"])
    target_panel = panel[panel["date"] == target].copy()
    print(f"  triggers @ {target.date()} (truncated bar): {len(target_panel)}")
    if target_panel.empty:
        return pd.DataFrame()

    panel_f = build_feature_panel(panel, data, xu100_close=xu_close)
    panel_f["date"] = pd.to_datetime(panel_f["date"])
    target_f = panel_f[panel_f["date"] == target].copy()
    print(f"  features @ {target.date()}: {len(target_f)}")

    labeled = compute_labels_on_panel(panel, data, CONFIG.label)
    labeled["date"] = pd.to_datetime(labeled["date"])
    h = CONFIG.label.primary_h
    keep_cols = [
        "ticker", "date",
        f"atr_{CONFIG.label.atr_window}", "close_0",
        *[f"mfe_{hh}" for hh in CONFIG.label.horizons],
        *[f"mae_{hh}" for hh in CONFIG.label.horizons],
        "mfe_mae_ratio_raw", "mfe_mae_ratio_win",
        "follow_through_3", f"cont_{h}", f"expansion_score_{h}",
        f"cont_{h}_struct", "risk_unit_struct_pct",
    ]
    keep_cols = [c for c in keep_cols if c in labeled.columns]
    labels_small = labeled[keep_cols]

    merged = target_f.merge(labels_small, on=["ticker", "date"], how="left",
                            suffixes=("", "_lbl"))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="target date YYYY-MM-DD (boş = bugün TR)")
    ap.add_argument("--master", default=str(MASTER_PATH))
    ap.add_argument("--v4", default=str(V4_PATH))
    ap.add_argument("--intraday", default=str(INTRADAY_MASTER_PATH),
                    help="intraday 15m master cache (output/input of fetch_layered)")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help="truncated-bar coverage / universe altında pasif çık")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="skip layered fetch step; use existing intraday cache")
    args = ap.parse_args()

    if args.date:
        target = pd.Timestamp(args.date).normalize()
    else:
        target = _today_istanbul()

    master_path = Path(args.master)
    v4_path = Path(args.v4)
    intraday_path = Path(args.intraday)

    print("═══ rebuild dataset delta — INTRADAY (17:00 TR truncated) ═══")
    print(f"  target   : {target.date()}")
    print(f"  master   : {master_path}")
    print(f"  v4       : {v4_path}")
    print(f"  intraday : {intraday_path}")
    print(f"  fetch    : {'SKIP' if args.skip_fetch else 'fetch_layered (4-tier)'}")
    print()

    if target.weekday() >= 5:
        print(f"⚠ {target.date()} weekend — skipping")
        return 0

    print("[1/7] load master")
    master = pd.read_parquet(master_path)
    if master.index.name != "Date":
        master.index.name = "Date"
    print(f"  master: {master.shape}  max date: {master.index.max().date()}")
    universe = sorted(master["ticker"].astype(str).unique().tolist())
    print(f"  universe: {len(universe)}")

    print("\n[2/7] intraday data acquisition")
    if args.skip_fetch:
        print("  --skip-fetch set; reading existing cache only")
        intraday_cache = _load_intraday_cache(intraday_path)
    else:
        intraday_cache, _bd = _fetch_intraday_full_universe(
            universe, target, intraday_path,
        )

    print("\n[3/7] aggregate 17:00 TR truncated daily bars")
    target_intraday = intraday_cache[
        pd.to_datetime(intraday_cache["signal_date"]).dt.normalize() == target
    ].copy()
    if target_intraday.empty:
        print(f"⚠ no intraday rows for {target.date()} — aborting")
        return 0
    truncated = aggregate_truncated_bars(target_intraday)
    truncated = truncated[truncated["signal_date"] == target].copy()
    coverage = len(truncated) / max(len(universe), 1)
    print(f"  truncated bars @ {target.date()}: {len(truncated)}/{len(universe)} "
          f"({coverage:.1%})")
    if coverage < args.min_coverage:
        print(f"⚠ coverage < {args.min_coverage:.0%} — aborting "
              f"(early close / holiday / fetcher outage)")
        return 0
    expected_bars = truncated["n_bars"].median() if "n_bars" in truncated.columns else None
    if expected_bars is not None:
        print(f"  bars/ticker median: {int(expected_bars)} (28 = full 10:15..17:00)")

    print("\n[4/7] patch master OHLCV (idempotent — 18:05 EOD will overwrite)")
    patched = _patch_master(master, truncated, target)
    print(f"  master after patch: {patched.shape}  max: {patched.index.max().date()}")
    patched.to_parquet(master_path)
    print(f"  ✓ {master_path}")

    print("\n[5/7] XU100 + regime")
    xu = load_xu100(refresh=True, period="6y")
    xu_close = xu["Close"] if xu is not None and not xu.empty else None
    if xu_close is not None:
        print(f"  xu100: {len(xu_close)} bars  max: {xu_close.index.max().date()}")

    print("\n[6/7] rebuild trigger + features + labels (truncated bar)")
    merged = _build_target_panel(target, xu_close, master_path)
    if merged.empty:
        print(f"  ℹ no triggers @ {target.date()} (truncated) — v4 unchanged")
        return 0
    if xu is not None and not xu.empty:
        regime = classify_xu100_regime(xu)
        merged["xu_regime"] = pd.to_datetime(merged["date"]).map(regime).values
        print(f"  regime: {merged['xu_regime'].iloc[0] if len(merged) else 'N/A'}")
    merged["fold"] = "fold3"
    merged["split"] = "test"

    print("\n[7/7] append to v4 (idempotent)")
    v4 = pd.read_parquet(v4_path)
    v4["date"] = pd.to_datetime(v4["date"])
    print(f"  v4 before: {v4.shape}  max: {v4['date'].max().date()}")
    v4_keep = v4[v4["date"] != target].copy()

    missing_in_merged = set(v4.columns) - set(merged.columns)
    extra_in_merged = set(merged.columns) - set(v4.columns)
    if missing_in_merged:
        print(f"  ⚠ missing in merged: {sorted(missing_in_merged)[:8]}"
              f"{'...' if len(missing_in_merged) > 8 else ''}")
    if extra_in_merged:
        print(f"  ⚠ extra in merged: {sorted(extra_in_merged)[:8]}"
              f"{'...' if len(extra_in_merged) > 8 else ''}")
    for c in missing_in_merged:
        merged[c] = np.nan
    merged = merged[list(v4.columns)]

    new_v4 = pd.concat([v4_keep, merged], ignore_index=True)
    new_v4 = new_v4.sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"  v4 after: {new_v4.shape}  max: {new_v4['date'].max().date()}")
    new_v4.to_parquet(v4_path)
    print(f"  ✓ {v4_path}")
    print(f"\n  triggers appended @ {target.date()} (truncated): {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
