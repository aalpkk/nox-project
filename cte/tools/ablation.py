"""
CTE ablation — trigger geometry × label family × feature set grid.

Her deney:
  (cfg_override) → dataset build → train → eval
  summary: N_triggers, base_rate, score_model_p@10/20, lift@10, PF_proxy,
           fold_rho, per-setup pass rates.

Matrix eksenleri (ilk pass, alınabilir çiftler):
  structure_window : (15, 20, 25)
  bar_min_return   : (0.03, 0.05, 0.07)
  bar_min_rvol     : (1.3, 1.5, 1.8)
  max_prior_attempts: (0, 1, 2)     # firstness hard filter
  dryup_enabled    : (True, False)
  setup_family     : ("hb_only", "fc_only", "both")
  target           : ("runner_10", "runner_15", "runner_20")

Full grid çok büyük; önce --mode=quick ile 10 konfigurasyon denenir, en iyi
"liftin" edge sinyallerini kullanıp --mode=sweep ile spesifik ekseni genişlet.

Kullanım:
  python -m cte.tools.ablation --mode quick
  python -m cte.tools.ablation --mode sweep --axis bar_min_return
  python -m cte.tools.ablation --mode custom --override 'bar.min_return_1d=0.03' \
      --override 'firstness.max_prior_attempts=2'
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.config import (
    CONFIG,
    BreakoutBarParams,
    CompressionParams,
    Config,
    DryupParams,
    FallingChannelParams,
    FirstBreakParams,
    HorizontalBaseParams,
    LabelParams,
)
from cte.dataset import build_dataset
from cte.features import FEATURES_V1
from cte.train import LGBMParams, train_fold


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted overrides like 'bar.min_return_1d=0.03'."""
    groups = {
        "compression": cfg.compression,
        "hb": cfg.hb,
        "fc": cfg.fc,
        "dryup": cfg.dryup,
        "firstness": cfg.firstness,
        "bar": cfg.bar,
        "label": cfg.label,
        "data": cfg.data,
        "split": cfg.split,
    }
    updated = dict(groups)
    for dotted, value in overrides.items():
        group_name, attr = dotted.split(".", 1)
        if group_name not in groups:
            raise KeyError(f"unknown config group: {group_name}")
        current = updated[group_name]
        updated[group_name] = replace(current, **{attr: value})
    return Config(**updated)


def _evaluate(preds: pd.DataFrame, target: str) -> dict[str, float]:
    mask = preds[target].notna()
    if mask.sum() == 0:
        return {}
    y = preds.loc[mask, target]
    s = preds.loc[mask, "score_model"]
    base = float(y.mean())
    rho = float(s.rank().corr(y.rank())) if len(y) > 10 else np.nan

    def p_at(k_frac: float) -> float:
        k = max(1, int(round(len(s) * k_frac)))
        top_idx = s.nlargest(k).index
        return float(y.loc[top_idx].mean())

    p10 = p_at(0.10)
    p20 = p_at(0.20)
    lift10 = p10 / base if base > 0 else np.nan
    # PF proxy
    h = target.split("_")[-1]
    mfe_col = f"mfe_{h}_atr"
    mae_col = f"mae_{h}_atr"
    pf = np.nan
    wr = np.nan
    if mfe_col in preds.columns and mae_col in preds.columns:
        full = preds.loc[mask].copy()
        mfe_arr = np.minimum(full[mfe_col].values, 3.0)
        mae_arr = -np.minimum(full[mae_col].values, 1.5)
        realised = np.where(full[target].values == 1, mfe_arr, mae_arr)
        realised = realised[~np.isnan(realised)]
        pos = realised[realised > 0].sum()
        neg = -realised[realised < 0].sum()
        pf = float(pos / neg) if neg > 0 else float("inf")
        wr = float((realised > 0).mean())

    return {
        "n": int(mask.sum()),
        "base_rate": base,
        "rho": rho,
        "p@10": p10,
        "p@20": p20,
        "lift@10": lift10,
        "pf_proxy": pf,
        "wr_proxy": wr,
    }


def _run_experiment(
    name: str,
    overrides: dict[str, Any],
    target: str,
    setup_family: str,
    panel_master: dict[str, pd.DataFrame] | None,
    xu100_close: pd.Series | None,
) -> dict[str, Any]:
    cfg = _apply_overrides(CONFIG, overrides)
    # Build dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = build_dataset(
            ohlcv_by_ticker=panel_master,
            xu100_close=xu100_close,
            cfg=cfg,
            verbose=False,
        )
    if ds.empty:
        return {"name": name, "status": "empty", "overrides": overrides}

    # Setup filter
    if setup_family == "hb_only":
        ds = ds[ds["setup_type"] == "hb"]
    elif setup_family == "fc_only":
        ds = ds[ds["setup_type"] == "fc"]
    # "both" → keep all

    if ds.empty or ds[target].notna().sum() < 100:
        return {"name": name, "status": "too_small",
                "n": int(ds[target].notna().sum()), "overrides": overrides}

    # Train WF
    feature_cols = [c for c in FEATURES_V1 if c in ds.columns]
    ds["date"] = pd.to_datetime(ds["date"])
    params = LGBMParams()

    all_test = []
    for fs in cfg.split.folds:
        try:
            _, qt, _ = train_fold(
                ds, feature_cols, target, fs.name,
                train_start=cfg.split.train_start,
                train_end=fs.train_end,
                val_start=fs.val_start, val_end=fs.val_end,
                test_start=fs.test_start, test_end=fs.test_end,
                params=params,
            )
            all_test.append(qt)
        except Exception as e:
            print(f"  [{name}] fold {fs.name} failed: {e}")

    if not all_test:
        return {"name": name, "status": "train_failed", "overrides": overrides}

    preds = pd.concat(all_test, ignore_index=True)
    metrics = _evaluate(preds, target)
    return {
        "name": name,
        "status": "ok",
        "overrides": overrides,
        "target": target,
        "setup_family": setup_family,
        **metrics,
    }


def _quick_matrix() -> list[tuple[str, dict[str, Any], str, str]]:
    """10 konfigurasyon — en yüksek-signal aksen noktaları."""
    base_target = "runner_15"
    return [
        ("baseline_v1", {}, base_target, "both"),
        ("looser_return_3pct", {"bar.min_return_1d": 0.03}, base_target, "both"),
        ("stricter_return_7pct", {"bar.min_return_1d": 0.07}, base_target, "both"),
        ("looser_rvol_1p3", {"bar.min_rvol": 1.3}, base_target, "both"),
        ("no_firstness_hard", {"firstness.max_prior_attempts": 3}, base_target, "both"),
        ("strict_firstness", {"firstness.max_prior_attempts": 0}, base_target, "both"),
        ("short_horizon", {}, "runner_10", "both"),
        ("long_horizon", {}, "runner_20", "both"),
        ("hb_only", {}, base_target, "hb_only"),
        ("fc_only", {}, base_target, "fc_only"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "custom"], default="quick")
    ap.add_argument("--override", action="append", default=[],
                    help="Custom override: 'group.attr=value'")
    ap.add_argument("--target", default="runner_15")
    ap.add_argument("--setup", default="both",
                    choices=["both", "hb_only", "fc_only"])
    ap.add_argument("--out", default="output/cte_ablation_v1.csv")
    ap.add_argument("--master", default=CONFIG.data.yf_cache_path)
    ap.add_argument("--xu100", default="output/xu100_cache.parquet")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    # Load once
    print(f"[ablation] loading master from {args.master}")
    master = pd.read_parquet(args.master)
    panel = {tk: sub.sort_index() for tk, sub in master.groupby("ticker")}
    xu100_close = None
    if Path(args.xu100).exists():
        xu100_close = pd.read_parquet(args.xu100)["Close"].astype(float)
    print(f"  {len(panel)} tickers, XU100 bars={len(xu100_close) if xu100_close is not None else 0}")

    # Build experiment list
    if args.mode == "quick":
        experiments = _quick_matrix()
    else:
        overrides = {}
        for s in args.override:
            k, v = s.split("=", 1)
            try:
                val: Any = float(v)
                if val == int(val):
                    val = int(val)
            except ValueError:
                val = v
            overrides[k] = val
        experiments = [("custom", overrides, args.target, args.setup)]

    # Run each
    results: list[dict[str, Any]] = []
    for i, (name, ovr, target, setup) in enumerate(experiments, start=1):
        print(f"\n[{i}/{len(experiments)}] {name}  target={target}  setup={setup}")
        print(f"    overrides={ovr}")
        res = _run_experiment(name, ovr, target, setup, panel, xu100_close)
        print(f"    → status={res.get('status')}  "
              f"n={res.get('n')}  base={res.get('base_rate')}  "
              f"lift@10={res.get('lift@10')}  PF={res.get('pf_proxy')}")
        results.append(res)

    # Save
    summary = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\n[WRITE] {args.out}")

    # Print sorted by lift@10
    print("\n─── Sorted by lift@10 (desc) ───")
    ok = summary[summary["status"] == "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values("lift@10", ascending=False)
        cols = ["name", "setup_family", "target", "n", "base_rate",
                "rho", "p@10", "lift@10", "pf_proxy"]
        cols = [c for c in cols if c in ok.columns]
        print(ok[cols].to_string(
            index=False,
            formatters={
                "base_rate": "{:.1%}".format,
                "rho": "{:+.3f}".format,
                "p@10": "{:.1%}".format,
                "lift@10": "{:.2f}x".format,
                "pf_proxy": "{:.2f}".format,
            },
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
