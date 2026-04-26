"""Apply v2 ranker (with categories + age) on VAL — top-K lift vs baselines.

Mirrors rank_apply.py but uses v2 weights and expands features first.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener_combo.data_prep import split_bounds
from screener_combo.rank_discovery_v2 import expand_features
from screener_combo.rank_apply import (
    build_ranker_score, topk_metrics, _DEDUP_DROP, GATES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--top-k", type=int, nargs="*", default=[3, 5, 10])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    weights_path = out_dir / f"screener_combo_v1_ranker_weights_v2_{args.tag}_h{args.horizon}.csv"
    if not triggers_path.exists() or not weights_path.exists():
        raise FileNotFoundError("missing scan or v2 ranker output")

    table = pd.read_parquet(triggers_path)
    table["date"] = pd.to_datetime(table["date"])
    table, _ = expand_features(table)
    weights_full = pd.read_csv(weights_path)

    bounds = split_bounds()
    val_lo, val_hi = bounds["val"]
    val = table[(table.date >= val_lo) & (table.date <= val_hi)].copy()
    print(f"VAL rows: {len(val):,}  ({val_lo.date()} → {val_hi.date()})")

    rng = np.random.default_rng(42)
    rows = []
    for gate in GATES:
        gw = weights_full[weights_full.gate == gate]
        if gw.empty:
            print(f"  {gate}: no v2 ranker weights — skip")
            continue
        gv = val[val[gate]].copy()
        if gv.empty:
            print(f"  {gate}: no VAL trigger rows — skip")
            continue
        gv["rank_score"] = build_ranker_score(gv, gw, gate)

        for k in args.top_k:
            mr = topk_metrics(gv, "rank_score", args.horizon, k, ascending=False)
            mr.update({"gate": gate, "top_k": k, "method": "ranker_v2"})
            rows.append(mr)

            seed_pfs, seed_means = [], []
            for s in range(args.seeds):
                gv["rand_score"] = rng.random(len(gv))
                m = topk_metrics(gv, "rand_score", args.horizon, k, ascending=False)
                if m.get("days", 0) > 0:
                    seed_pfs.append(m.get("PF_daily", np.nan))
                    seed_means.append(m.get("mean_daily_R_%", np.nan))
            if seed_pfs:
                rows.append({
                    "gate": gate, "top_k": k, "method": f"random_avg_seeds={args.seeds}",
                    "days": int(gv.date.nunique()),
                    "mean_daily_R_%": float(np.nanmean(seed_means)),
                    "median_daily_R_%": np.nan,
                    "hit_daily_%": np.nan,
                    "PF_daily": (float(np.nanmean([p for p in seed_pfs if not np.isinf(p)]))
                                 if any(not np.isinf(p) for p in seed_pfs) else np.nan),
                    "cum_R_%": np.nan,
                })

    out_df = pd.DataFrame(rows)
    out_df = out_df[["gate", "top_k", "method", "days", "mean_daily_R_%",
                     "median_daily_R_%", "hit_daily_%", "PF_daily", "cum_R_%"]]
    out_path = out_dir / f"screener_combo_v1_rank_val_v2_{args.tag}_h{args.horizon}.csv"
    out_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  → {out_path}")

    for gate in GATES:
        sub = out_df[out_df.gate == gate]
        if sub.empty:
            continue
        print(f"\n=== {gate} | VAL | h={args.horizon} (v2) ===")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
