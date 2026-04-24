"""
CTE HB seed-bag test (uzman oturumu: cte_hb_v2).

Motivation
----------
HB single-seed tune (output/cte_hb_tune_sweep.csv) showed baseline lift@10 is
**seed-fragile**: 1.14x (seed 17) / 1.14x (23) / 0.69x (42) / 0.91x (101),
seed-avg ≈ 0.97x. With N_test=185 (top-10% = 18 rows), one flip is ±5.5pp.
Single-seed "1.14x" is partly a lottery.

Solution test: train same config across multiple seeds, **average test scores
per row**, eval the averaged score. This reduces variance without changing
contract, dataset, features, split, or shared train/line code. The only
thing "new" is inference-time score aggregation.

Allowed scope (feedback_cte_session_split):
  - cte/train_hb.py, cte/tools/*        ← OK
  - cte/line.py, cte/features.py, etc.  ← UNTOUCHED

Output
------
  output/cte_hb_seed_bag.csv
    One row per (config × bag_size) with mean lift@10, rho, p@10,
    PF_top30, and per-fold lift@10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.config import CONFIG
from cte.contract import CONTRACT_PATH, verify_contract
from cte.features import FEATURES_V1
from cte.line import eval_line, train_line
from cte.train import LGBMParams


@dataclass
class BagConfig:
    name: str
    target: str = "runner_15"
    mode: str = "mixed"
    features: tuple[str, ...] | None = None
    lgbm_overrides: dict = field(default_factory=dict)
    seeds: tuple[int, ...] = (17, 23, 42, 101, 7, 99, 3, 71, 5, 11, 29, 53)
    note: str = ""


# Candidate configs (from hb_tune single-seed results + controls)
DEFAULT_CONFIGS: tuple[BagConfig, ...] = (
    BagConfig(name="baseline_bag",
              note="mcs=20, 35 feats, runner_15"),
    BagConfig(name="h20_bag", target="runner_20",
              note="runner_20 horizon"),
    BagConfig(name="h10_bag", target="runner_10",
              note="runner_10 horizon"),
    BagConfig(name="mcs10_bag",
              lgbm_overrides={"min_child_samples": 10},
              note="lower min_child (hb_tune best single-seed)"),
    BagConfig(name="mcs40_bag",
              lgbm_overrides={"min_child_samples": 40},
              note="higher min_child"),
    BagConfig(name="leaves15_bag",
              lgbm_overrides={"num_leaves": 15},
              note="shallower trees"),
    BagConfig(name="ff100_bag",
              lgbm_overrides={"feature_fraction": 1.0},
              note="no feature subsampling (all seeds see all feats)"),
)


def _params_with_overrides(seed: int, overrides: dict) -> LGBMParams:
    base = asdict(LGBMParams())
    base.update(overrides)
    base["seed"] = seed
    return LGBMParams(**base)


def _train_one(df: pd.DataFrame, cfg: BagConfig, seed: int, split) -> pd.DataFrame:
    feats_src = list(cfg.features) if cfg.features else list(FEATURES_V1)
    feats = [c for c in feats_src if c in df.columns]
    params = _params_with_overrides(seed, cfg.lgbm_overrides)
    res = train_line(df=df, line="hb", mode=cfg.mode, target=cfg.target,
                     feature_cols=feats, split=split, params=params)
    return res.preds  # contains score_model for this seed


def run_bag(df: pd.DataFrame, cfg: BagConfig, split, sizes: tuple[int, ...]) -> list[dict]:
    """Train each seed once, then evaluate the average score across the first
    k seeds for k in sizes. Returns one summary row per bag size."""
    print(f"\n▶ config={cfg.name}  target={cfg.target}  mode={cfg.mode}  "
          f"seeds={cfg.seeds}  lgbm={cfg.lgbm_overrides or '{}'}")

    seed_preds: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        print(f"  [seed={seed}]")
        p = _train_one(df, cfg, seed, split)
        if p.empty:
            print(f"    (empty preds for seed {seed}, skipping)")
            continue
        p = p[["ticker", "date", "fold_assigned", "setup_type",
               "score_model", "score_compression", "score_random",
               cfg.target, f"mfe_{cfg.target.split('_')[-1]}_atr",
               f"mae_{cfg.target.split('_')[-1]}_atr"]].copy()
        p["seed"] = seed
        seed_preds.append(p)

    if not seed_preds:
        return [{"config": cfg.name, "bag_size": 0, "note": "all seeds empty"}]

    # Align on (ticker, date, fold_assigned) — stable across seeds.
    # score_compression is deterministic per-fold (rank of compression_score)
    # but score_random varies per seed, so exclude it from merge key.
    rows = []
    merged = None
    key = ["ticker", "date", "fold_assigned"]
    score_cols = []
    for i, p in enumerate(seed_preds):
        col = f"score_seed_{p['seed'].iloc[0]}"
        sub = p.rename(columns={"score_model": col}).drop(columns=["seed"])
        score_cols.append(col)
        if merged is None:
            # first seed: carry all non-score columns
            keep_cols = [c for c in sub.columns
                         if c not in ("score_random",)]
            merged = sub[keep_cols].copy()
        else:
            merged = merged.merge(sub[key + [col]], on=key, how="inner")

    if merged is None or merged.empty:
        return [{"config": cfg.name, "bag_size": 0, "note": "merge failed"}]

    # Eval each bag size (mean of first k seed columns)
    for k in sizes:
        if k > len(score_cols):
            continue
        cols_k = score_cols[:k]
        bagged = merged.copy()
        bagged["score_model"] = bagged[cols_k].mean(axis=1)
        evald = eval_line(bagged, target=cfg.target, line="hb")
        row = _summarize(cfg, evald, k, len(bagged))
        rows.append(row)

    return rows


def _g(d, *path, default=float("nan")):
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def _summarize(cfg: BagConfig, e: dict, bag_size: int, n_rows: int) -> dict:
    over = e.get("overall", {})
    m = over.get("scores", {}).get("score_model", {})
    pf_o = over.get("pf_proxy_overall", {})
    pf_t = over.get("pf_proxy_top30", {})
    byf = e.get("by_fold", {})
    lift_by_fold = {f: _g(byf, f, "scores", "score_model", "lift@10")
                    for f in ("fold1", "fold2", "fold3")}
    return {
        "config": cfg.name,
        "target": cfg.target,
        "mode": cfg.mode,
        "bag_size": bag_size,
        "n_rows": n_rows,
        "base": over.get("base", float("nan")),
        "rho": m.get("rho", float("nan")),
        "p@10": m.get("p@10", float("nan")),
        "lift@10": m.get("lift@10", float("nan")),
        "lift@20": m.get("lift@20", float("nan")),
        "lift@10_fold1": lift_by_fold["fold1"],
        "lift@10_fold2": lift_by_fold["fold2"],
        "lift@10_fold3": lift_by_fold["fold3"],
        "PF_top30": pf_t.get("PF_proxy", float("nan")),
        "WR_top30": pf_t.get("WR_proxy", float("nan")),
        "PF_overall": pf_o.get("PF_proxy", float("nan")),
        "n_seeds_avail": len(cfg.seeds),
        "lgbm_overrides": json.dumps(cfg.lgbm_overrides, sort_keys=True),
        "note": cfg.note,
    }


def _print_table(rows: list[dict]) -> None:
    cols = ["config", "target", "bag_size", "n_rows", "base",
            "rho", "p@10", "lift@10", "lift@20",
            "lift@10_fold1", "lift@10_fold2", "lift@10_fold3",
            "PF_top30", "WR_top30", "note"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    for c in ["base", "rho", "p@10", "WR_top30"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    for c in [c for c in df.columns if c.startswith("lift@") or c.startswith("PF_")]:
        df[c] = df[c].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    print("\n═══ HB seed-bag summary ═══")
    print(df.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--out-csv", default="output/cte_hb_seed_bag.csv")
    ap.add_argument("--sizes", nargs="*", type=int, default=[1, 3, 5, 8],
                    help="Bag sizes to evaluate (subset of seeds). Default 1/3/5/8.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run configs with these names")
    ap.add_argument("--ignore-contract", action="store_true")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}")
        return 2

    if args.ignore_contract:
        print("⚠ --ignore-contract: HB seed-bag contract atladı")
    else:
        verify_contract(
            dataset_path=args.dataset,
            contract_path=str(CONTRACT_PATH),
            raise_on_mismatch=True,
            verbose=True,
            line="hb",
        )

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[DATA] {args.dataset}  shape={df.shape}")

    configs = DEFAULT_CONFIGS
    if args.only:
        only_set = set(args.only)
        configs = tuple(c for c in configs if c.name in only_set)
        if not configs:
            print(f"❌ No configs match --only {args.only}")
            return 3

    all_rows = []
    for cfg in configs:
        try:
            rows = run_bag(df, cfg, CONFIG.split, tuple(args.sizes))
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ✗ config {cfg.name} failed: {e}")
            all_rows.append({"config": cfg.name, "bag_size": 0,
                             "note": f"FAILED: {e}"})

    rows_df = pd.DataFrame(all_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(args.out_csv, index=False)
    print(f"\n[WRITE] {args.out_csv}  n_rows={len(all_rows)}")
    _print_table(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
