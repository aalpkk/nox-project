"""Append today's trigger panel to ``nyxexp_dataset_v4.parquet`` (idempotent).

Pipeline phase 0 — runs before ``scan_latest`` so the daily candidate ranker
has rows to score for the current trading day.

Steps:
  1. Detect target date (CLI ``--date`` or Europe/Istanbul today).
  2. Bulk-fetch yfinance EOD for every ticker already in the master OHLCV.
  3. Skip if coverage < threshold (weekend, holiday, pre-close).
  4. Patch master parquet for target_date (drop-then-append, idempotent).
  5. Recompute ``compute_trigger_a_panel`` + ``build_feature_panel``
     + ``compute_labels_on_panel`` over the full panel; pull target_date slice.
  6. Append target_date rows to v4 (drop existing first, idempotent).

CLI:
    python -m nyxexpansion.tools.rebuild_dataset_delta            # today
    python -m nyxexpansion.tools.rebuild_dataset_delta --date 2026-04-24
    python -m nyxexpansion.tools.rebuild_dataset_delta --min-coverage 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
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
from nyxexpansion.tools.presmoke import (  # noqa: E402
    load_ohlcv_cache, load_xu100, classify_xu100_regime,
)

MASTER_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
V4_PATH = Path("output/nyxexp_dataset_v4.parquet")
DEFAULT_MIN_COVERAGE = 0.5


def _today_istanbul() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=None)).tz_localize(
        "UTC").tz_convert("Europe/Istanbul").normalize().tz_localize(None)


def _suffix_is(ticker: str) -> str:
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def _strip_is(symbol: str) -> str:
    return symbol[:-3] if symbol.endswith(".IS") else symbol


def _fetch_yf_eod(tickers: list[str], target: pd.Timestamp) -> pd.DataFrame:
    """Bulk yfinance download for one trading day → long-form ticker/Date frame."""
    import yfinance as yf

    symbols = [_suffix_is(t) for t in tickers]
    start = target.strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  yf.download: {len(symbols)} symbols  {start}..{end}")
    raw = yf.download(
        symbols, start=start, end=end, interval="1d",
        group_by="ticker", auto_adjust=False, progress=False, threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows = []
    for sym in symbols:
        try:
            sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        if sub is None or sub.empty:
            continue
        for idx, row in sub.iterrows():
            close = row.get("Close")
            if pd.isna(close):
                continue
            rows.append({
                "Date": pd.Timestamp(idx).normalize(),
                "ticker": _strip_is(sym),
                "Open": float(row.get("Open")) if pd.notna(row.get("Open")) else np.nan,
                "High": float(row.get("High")) if pd.notna(row.get("High")) else np.nan,
                "Low": float(row.get("Low")) if pd.notna(row.get("Low")) else np.nan,
                "Close": float(close),
                "Volume": float(row.get("Volume")) if pd.notna(row.get("Volume")) else 0.0,
                "Adj Close": np.nan,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Date")
    return df


def _patch_master(master: pd.DataFrame, delta: pd.DataFrame,
                  target: pd.Timestamp) -> pd.DataFrame:
    """Drop target_date rows from master, append delta, sort."""
    keep = master[master.index != target].copy()
    delta_target = delta[delta.index == target].copy()
    if delta_target.empty:
        return master
    # Align columns to master schema
    for col in master.columns:
        if col not in delta_target.columns:
            delta_target[col] = np.nan
    delta_target = delta_target[list(master.columns)]
    out = pd.concat([keep, delta_target]).sort_index()
    out.index.name = master.index.name or "Date"
    return out


def _build_target_panel(master: pd.DataFrame, target: pd.Timestamp,
                        xu_close) -> pd.DataFrame:
    """Run trigger → features → labels for full panel; return target_date slice."""
    data = load_ohlcv_cache(str(MASTER_PATH))
    print(f"  reload master as dict-of-DataFrames: {len(data)} tickers")

    panel = compute_trigger_a_panel(data, CONFIG.trigger)
    panel["date"] = pd.to_datetime(panel["date"])
    target_panel = panel[panel["date"] == target].copy()
    print(f"  triggers @ {target.date()}: {len(target_panel)}")
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
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help="yf coverage altında pasif çık (default 0.5)")
    args = ap.parse_args()

    if args.date:
        target = pd.Timestamp(args.date).normalize()
    else:
        target = _today_istanbul()

    master_path = Path(args.master)
    v4_path = Path(args.v4)

    print("═══ rebuild dataset delta ═══")
    print(f"  target  : {target.date()}")
    print(f"  master  : {master_path}")
    print(f"  v4      : {v4_path}")
    print()

    if target.weekday() >= 5:
        print(f"⚠ {target.date()} weekend — skipping")
        return 0

    print("[1/6] load master")
    master = pd.read_parquet(master_path)
    if master.index.name != "Date":
        master.index.name = "Date"
    print(f"  master: {master.shape}  max date: {master.index.max().date()}")
    universe = sorted(master["ticker"].astype(str).unique().tolist())
    print(f"  universe: {len(universe)}")

    print("[2/6] yfinance EOD bulk fetch")
    delta = _fetch_yf_eod(universe, target)
    delta_target = delta[delta.index == target] if not delta.empty else pd.DataFrame()
    coverage = len(delta_target) / max(len(universe), 1)
    print(f"  delta @ {target.date()}: {len(delta_target)}/{len(universe)} "
          f"({coverage:.1%})")
    if coverage < args.min_coverage:
        print(f"⚠ coverage < {args.min_coverage:.0%} (likely pre-close or holiday) — skipping")
        return 0

    print("[3/6] patch master (idempotent)")
    patched = _patch_master(master, delta, target)
    print(f"  master after patch: {patched.shape}  max: {patched.index.max().date()}")
    patched.to_parquet(master_path)
    print(f"  ✓ {master_path}")

    print("[4/6] XU100 + regime")
    xu = load_xu100(refresh=True, period="6y")
    xu_close = xu["Close"] if xu is not None and not xu.empty else None
    if xu_close is not None:
        print(f"  xu100: {len(xu_close)} bars  max: {xu_close.index.max().date()}")

    print("[5/6] rebuild trigger + features + labels")
    merged = _build_target_panel(patched, target, xu_close)
    if merged.empty:
        print(f"  ℹ no triggers @ {target.date()} — v4 unchanged")
        return 0
    if xu is not None and not xu.empty:
        regime = classify_xu100_regime(xu)
        merged["xu_regime"] = pd.to_datetime(merged["date"]).map(regime).values
        print(f"  regime: {merged['xu_regime'].iloc[0] if len(merged) else 'N/A'}")
    merged["fold"] = "fold3"
    merged["split"] = "test"

    print("[6/6] append to v4 (idempotent)")
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
    print(f"\n  triggers appended @ {target.date()}: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
