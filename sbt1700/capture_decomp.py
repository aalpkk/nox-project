"""SBT-1700 — capture decomposition by forward-path type.

For a small set of carried exit variants, run the simulator on TRAIN,
tag every trade with its forward-path type (`parabolic`, `spike_fade`,
`clean`), and aggregate the six diagnostic metrics per (variant, path)
cell:

    n
    avg_MFE_R
    avg_realized_R
    captured_MFE_ratio_cohort  = Σ realized_R / Σ MFE_R
    avg_giveback_R
    avg_hold_bars
    trend_exit_rate

The output answers the question:

    "Of the %77 giveback we observed, where does it come from?"

The classifier is exit-agnostic (see `path_type.py`), so the path tag is
the same across variants for a given (ticker, date). Differences in
per-path capture between variants therefore reflect *exit behaviour
on the same path*, not selection drift.

Hard contract:
    * Reads ONLY the panel passed in (caller is `reset_pipeline.phase_capture_decomp`,
      which loads TRAIN via `splits.load_split`). Validation/test never
      touched here.
    * No model fitting, no ranker work; this is descriptive.
    * Outputs CSVs only — `sbt_1700_capture_decomp_train.csv` (per-cell)
      and `sbt_1700_capture_decomp_paths.csv` (per-path cohort summary
      independent of variant).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sbt1700.exits_v2 import ExitConfigV2
from sbt1700.exit_discovery import simulate_variant_on_panel, _by_ticker_master
from sbt1700.path_type import (
    PATH_PARABOLIC,
    PATH_SPIKE_FADE,
    PATH_CLEAN,
    PATH_UNKNOWN,
    classify_panel,
)


PATH_ORDER = (PATH_PARABOLIC, PATH_SPIKE_FADE, PATH_CLEAN, "ALL")


# ---------- aggregate per (variant, path) -----------------------------------

def _cell_metrics(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "n": 0,
            "avg_MFE_R": float("nan"),
            "avg_realized_R": float("nan"),
            "captured_MFE_ratio_cohort": float("nan"),
            "avg_giveback_R": float("nan"),
            "avg_hold_bars": float("nan"),
            "trend_exit_rate": float("nan"),
            "initial_stop_rate": float("nan"),
            "max_hold_rate": float("nan"),
            "wr": float("nan"),
        }
    realized = sub["realized_R_net"].to_numpy(dtype=float)
    mfe = sub["MFE_R"].to_numpy(dtype=float)
    sum_mfe = np.nansum(mfe)
    cap_cohort = (
        float(np.nansum(realized) / sum_mfe) if sum_mfe > 1e-9 else float("nan")
    )
    return {
        "n": int(len(sub)),
        "avg_MFE_R": float(np.nanmean(mfe)),
        "avg_realized_R": float(np.nanmean(realized)),
        "captured_MFE_ratio_cohort": cap_cohort,
        "avg_giveback_R": float(np.nanmean(sub["giveback_R"].to_numpy(dtype=float))),
        "avg_hold_bars": float(sub["bars_held"].mean()),
        "trend_exit_rate": float(sub["trend_exit_hit"].mean()),
        "initial_stop_rate": float(sub["initial_stop_hit"].mean()),
        "max_hold_rate": float(sub["max_hold_hit"].mean()),
        "wr": float((realized > 0).mean()),
    }


def decompose_variant(
    cfg: ExitConfigV2,
    panel: pd.DataFrame,
    by_ticker: dict[str, pd.DataFrame],
    path_tags: pd.DataFrame,
) -> pd.DataFrame:
    """Run the simulator on `panel`, join path tags, return per-path rows."""
    trades = simulate_variant_on_panel(cfg, panel, by_ticker)
    if trades.empty:
        return pd.DataFrame(columns=[
            "exit_variant", "exit_family", "trend_kind", "path_type",
            "n", "avg_MFE_R", "avg_realized_R", "captured_MFE_ratio_cohort",
            "avg_giveback_R", "avg_hold_bars", "trend_exit_rate",
            "initial_stop_rate", "max_hold_rate", "wr",
        ])
    trades["date"] = pd.to_datetime(trades["date"])
    tagged = trades.merge(
        path_tags[["ticker", "date", "path_type"]],
        on=["ticker", "date"],
        how="left",
    )
    tagged["path_type"] = tagged["path_type"].fillna(PATH_UNKNOWN)
    rows: list[dict] = []
    for path in (PATH_PARABOLIC, PATH_SPIKE_FADE, PATH_CLEAN):
        sub = tagged[(tagged["path_type"] == path)
                     & tagged["realized_R_net"].notna()]
        rows.append({
            "exit_variant": cfg.name,
            "exit_family": cfg.family,
            "trend_kind": cfg.trend_kind,
            "path_type": path,
            **_cell_metrics(sub),
        })
    # ALL row — same variant across all paths (excluding unknown).
    sub_all = tagged[tagged["path_type"].isin([PATH_PARABOLIC, PATH_SPIKE_FADE, PATH_CLEAN])
                     & tagged["realized_R_net"].notna()]
    rows.append({
        "exit_variant": cfg.name,
        "exit_family": cfg.family,
        "trend_kind": cfg.trend_kind,
        "path_type": "ALL",
        **_cell_metrics(sub_all),
    })
    return pd.DataFrame(rows)


def path_cohort_summary(path_tags: pd.DataFrame) -> pd.DataFrame:
    """Variant-agnostic per-path cohort stats — how big each bucket is."""
    valid = path_tags[path_tags["path_type"] != PATH_UNKNOWN]
    rows: list[dict] = []
    n_total = int(len(valid))
    for path in (PATH_PARABOLIC, PATH_SPIKE_FADE, PATH_CLEAN):
        sub = valid[valid["path_type"] == path]
        n = int(len(sub))
        rows.append({
            "path_type": path,
            "n": n,
            "share": (n / n_total) if n_total > 0 else float("nan"),
            "avg_mfe_R_path": float(sub["mfe_R_path"].mean()) if n > 0 else float("nan"),
            "avg_bars_to_mfe": float(sub["bars_to_mfe"].mean()) if n > 0 else float("nan"),
            "avg_mfe_R_first3": float(sub["mfe_R_first3"].mean()) if n > 0 else float("nan"),
            "avg_post_mfe_giveback_pct": float(sub["post_mfe_giveback_pct"].mean())
                                          if n > 0 else float("nan"),
        })
    rows.append({
        "path_type": "ALL",
        "n": n_total,
        "share": 1.0 if n_total > 0 else float("nan"),
        "avg_mfe_R_path": float(valid["mfe_R_path"].mean()) if n_total > 0 else float("nan"),
        "avg_bars_to_mfe": float(valid["bars_to_mfe"].mean()) if n_total > 0 else float("nan"),
        "avg_mfe_R_first3": float(valid["mfe_R_first3"].mean()) if n_total > 0 else float("nan"),
        "avg_post_mfe_giveback_pct": float(valid["post_mfe_giveback_pct"].mean())
                                      if n_total > 0 else float("nan"),
    })
    return pd.DataFrame(rows)


# ---------- top-level driver --------------------------------------------------

def run_capture_decomp(
    panel: pd.DataFrame,
    daily_master: pd.DataFrame,
    carried_variants: Iterable[ExitConfigV2],
    out_dir: Path,
    suffix: str = "train",
) -> dict:
    """Tag the panel, decompose each carried variant, write CSVs.

    Returns a small summary dict (paths to the written files + counts).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ticker = _by_ticker_master(daily_master)

    path_tags = classify_panel(panel, daily_master)
    paths_csv = out_dir / f"sbt_1700_capture_decomp_paths_{suffix}.csv"
    path_tags.to_csv(paths_csv, index=False)

    cohort_summary = path_cohort_summary(path_tags)
    cohort_csv = out_dir / f"sbt_1700_capture_decomp_cohort_{suffix}.csv"
    cohort_summary.to_csv(cohort_csv, index=False)

    rows: list[pd.DataFrame] = []
    for cfg in carried_variants:
        rows.append(decompose_variant(cfg, panel, by_ticker, path_tags))
    decomp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    decomp_csv = out_dir / f"sbt_1700_capture_decomp_{suffix}.csv"
    decomp.to_csv(decomp_csv, index=False)

    return {
        "paths_csv": str(paths_csv),
        "cohort_csv": str(cohort_csv),
        "decomp_csv": str(decomp_csv),
        "n_panel": int(len(panel)),
        "n_path_unknown": int((path_tags["path_type"] == PATH_UNKNOWN).sum()),
        "n_variants": len(rows),
    }
