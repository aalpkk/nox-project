"""
CTE dataset builder.

Master parquet (607 tickers) + XU100 cache → cte_dataset_v1.parquet:
  ticker, date, close, setup_type, trigger_hb, trigger_fc, trigger_cte
  + FEATURES_V1 columns
  + label columns from cte.labels (hold/failed_break/runner/expansion)
  + fold assignment (train/val/test for 3 walk-forward folds)

Leakage guards (aggregated):
  - Structure / compression: [t-W, t-1] (shift(1) in each computation).
  - First-break: bar t's own close > boundary_t is embedded in trigger; the
    break counts use only prior bars.
  - Label: forward-looking by design; features don't read labels.
  - XU100 Block E: `.shift(1)` on trend score & returns.
  - Fold split: strict embargo via label_horizon_bars gap between train_end
    and val_start (SplitParams encodes "-05-15" start = 15 days after end;
    max runner horizon=20 bars → need embargo ≥ 20 business days = safe).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cte.config import CONFIG, Config, FoldSpec
from cte.features import FEATURES_V1, FeatureParams, enrich_with_block_e
from cte.labels import compute_labels
from cte.structure import compute_structure
from cte.trigger import compute_trigger


LABEL_COLS: tuple[str, ...] = (
    "breakout_level_struct", "breakout_level_close", "atr_ref",
    "hold_3_close", "hold_5_close", "hold_3_struct", "hold_5_struct",
    "failed_break_3_close", "failed_break_5_close",
    "failed_break_3_struct", "failed_break_5_struct",
    "mfe_10_atr", "mae_10_atr", "spike_rejected_10",
    "mfe_15_atr", "mae_15_atr", "spike_rejected_15",
    "mfe_20_atr", "mae_20_atr", "spike_rejected_20",
    "expansion_score_10", "expansion_score_15", "expansion_score_20",
    "runner_10", "runner_15", "runner_20",
    "primary_target",
)


def _assign_fold(date: pd.Timestamp, folds: Iterable[FoldSpec]) -> str:
    """Map a date to 'fold{i}_{split}' or ''."""
    ts = pd.Timestamp(date)
    for f in folds:
        if ts <= pd.Timestamp(f.train_end):
            return f"{f.name}_train"
        if pd.Timestamp(f.val_start) <= ts <= pd.Timestamp(f.val_end):
            return f"{f.name}_val"
        if pd.Timestamp(f.test_start) <= ts <= pd.Timestamp(f.test_end):
            return f"{f.name}_test"
    return ""


def _build_single_ticker(
    ticker: str,
    df: pd.DataFrame,
    xu100_close: pd.Series | None,
    cfg: Config,
    feat_params: FeatureParams,
) -> pd.DataFrame | None:
    """Run trigger + labels + Block E enrichment for one ticker."""
    if df is None or df.empty or len(df) < cfg.data.min_bars_per_ticker:
        return None

    st = compute_structure(df, comp=cfg.compression, hb=cfg.hb, fc=cfg.fc)
    tr = compute_trigger(
        df,
        comp=cfg.compression,
        hb=cfg.hb,
        fc=cfg.fc,
        dry=cfg.dryup,
        fb=cfg.firstness,
        bar=cfg.bar,
    )
    lb = compute_labels(df, st, tr, params=cfg.label)
    enriched = enrich_with_block_e(
        df, tr, xu100_close=xu100_close, params=feat_params,
    )

    mask = enriched["trigger_cte"].fillna(False)
    if not mask.any():
        return None

    feature_cols = [c for c in FEATURES_V1 if c in enriched.columns]
    meta_cols = ["trigger_hb", "trigger_fc", "trigger_cte", "setup_type"]
    label_cols = [c for c in LABEL_COLS if c in lb.columns]

    trig_rows = enriched.loc[mask].copy()
    trig_rows = trig_rows[meta_cols + feature_cols]
    trig_rows = trig_rows.join(lb.loc[mask, label_cols])
    trig_rows["close"] = df.loc[mask, "Close"].astype(float)
    trig_rows["ticker"] = ticker
    trig_rows.index.name = "date"
    return trig_rows.reset_index()


def build_dataset(
    ohlcv_by_ticker: dict[str, pd.DataFrame] | None = None,
    xu100_close: pd.Series | None = None,
    cfg: Config | None = None,
    feat_params: FeatureParams | None = None,
    master_parquet: str | Path | None = None,
    xu100_parquet: str | Path = "output/xu100_cache.parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the CTE panel dataset.

    Two usage modes:
      (a) Pass `ohlcv_by_ticker` and `xu100_close` directly — preferred for
          tests / custom runs.
      (b) Pass `master_parquet` path — loads `output/ohlcv_10y_fintables_master.parquet`
          (schema: Open/High/Low/Close/Volume + 'ticker') and splits per-ticker.

    Returns long DataFrame sorted by (date, ticker) with a 'fold' column.
    """
    if cfg is None:
        cfg = CONFIG
    if feat_params is None:
        feat_params = FeatureParams()

    if ohlcv_by_ticker is None:
        if master_parquet is None:
            master_parquet = cfg.data.yf_cache_path
        if verbose:
            print(f"[cte.dataset] loading {master_parquet}")
        master = pd.read_parquet(master_parquet)
        if "ticker" not in master.columns:
            raise ValueError(
                f"master parquet {master_parquet} missing 'ticker' column",
            )
        ohlcv_by_ticker = {
            tk: sub.sort_index()
            for tk, sub in master.groupby("ticker")
        }
        if verbose:
            print(f"[cte.dataset] {len(ohlcv_by_ticker)} tickers split")

    if xu100_close is None and Path(xu100_parquet).exists():
        xu100_df = pd.read_parquet(xu100_parquet)
        xu100_close = xu100_df["Close"].astype(float)
        if verbose:
            print(f"[cte.dataset] XU100 series: {len(xu100_close)} bars "
                  f"{xu100_close.index.min().date()} → {xu100_close.index.max().date()}")

    rows: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (ticker, sub) in enumerate(ohlcv_by_ticker.items(), start=1):
            out = _build_single_ticker(ticker, sub, xu100_close, cfg, feat_params)
            if out is not None:
                rows.append(out)
            if verbose and i % 50 == 0:
                print(f"[cte.dataset] processed {i}/{len(ohlcv_by_ticker)} "
                      f"— accumulated {sum(len(r) for r in rows)} triggers")

    if not rows:
        if verbose:
            print("[cte.dataset] no triggers found")
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)

    # Front-order columns
    front = ["ticker", "date", "close",
             "trigger_hb", "trigger_fc", "trigger_cte", "setup_type"]
    rest = [c for c in panel.columns if c not in front]
    panel = panel[front + rest]
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Fold assignment
    panel["fold"] = panel["date"].map(
        lambda d: _assign_fold(d, cfg.split.folds),
    )

    if verbose:
        by_fold = panel["fold"].value_counts().to_dict()
        print(f"[cte.dataset] total triggers: {len(panel)}")
        print(f"[cte.dataset] fold distribution:")
        for k in sorted(by_fold, key=lambda x: (x == "", x)):
            print(f"    {k:20s}: {by_fold[k]}")

    return panel


if __name__ == "__main__":
    out = build_dataset()
    if not out.empty:
        out_path = Path("output/cte_dataset_v1.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path)
        print(f"[WRITE] {out_path}  shape={out.shape}")
