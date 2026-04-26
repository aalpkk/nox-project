"""Combined daily basket: union of top-3 from each gate's v2 ranker.

Per day: take top-3 by v2 ranker score within each gate (RT/NW/AS),
union → up-to-9 unique tickers per day → equal-weight portfolio return.

Comparison rows:
  combined_v2          — union of v2-ranker top-3 across 3 gates
  combined_random      — union of random top-3 across 3 gates (avg over seeds)
  combined_all_triggers— union of ALL trigger rows (any gate fired) — unranked baseline
  per-gate top-3       — for reference

Output:
  output/screener_combo_v1_combined_basket_{tag}_h{H}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener_combo.data_prep import split_bounds
from screener_combo.rank_discovery_v2 import expand_features
from screener_combo.rank_apply import build_ranker_score, GATES


def _daily_metrics(daily: pd.Series) -> dict:
    if daily.empty:
        return {"days": 0}
    pos = daily[daily > 0].sum()
    neg = -daily[daily < 0].sum()
    pf = float(pos / neg) if neg > 0 else float("inf") if pos > 0 else float("nan")
    return {
        "days": int(daily.shape[0]),
        "mean_daily_R_%": float(daily.mean()) * 100,
        "median_daily_R_%": float(daily.median()) * 100,
        "hit_daily_%": float((daily > 0).mean()) * 100,
        "PF_daily": pf,
        "cum_R_%": float(daily.sum()) * 100,
        "avg_basket_size": np.nan,
    }


def _basket_daily(picks: pd.DataFrame, h: int) -> tuple[pd.Series, float]:
    """picks: rows with ticker/date/fwd_R_h. Dedup per (date, ticker), avg per day."""
    r_col = f"fwd_R_{h}"
    p = picks[["date", "ticker", r_col]].dropna(subset=[r_col])
    p = p.drop_duplicates(["date", "ticker"])
    if p.empty:
        return pd.Series(dtype=float), 0.0
    daily = p.groupby("date")[r_col].mean()
    avg_size = p.groupby("date").size().mean()
    return daily, float(avg_size)


def _topk_per_gate(gate_rows: pd.DataFrame, score_col: str, h: int, k: int) -> pd.DataFrame:
    """Return per-day top-k rows for one gate by score_col (desc)."""
    r_col = f"fwd_R_{h}"
    if score_col not in gate_rows.columns:
        return gate_rows.iloc[0:0]
    g = gate_rows.dropna(subset=[score_col, r_col])
    if g.empty:
        return g
    return (
        g.sort_values(["date", score_col], ascending=[True, False])
        .groupby("date")
        .head(k)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
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

    h = args.horizon
    k = args.top_k
    r_col = f"fwd_R_{h}"

    rng = np.random.default_rng(42)
    rows = []

    # Per-gate v2 ranker top-k pick frames + per-gate metrics
    per_gate_picks = {}
    per_gate_random_picks = {}
    for gate in GATES:
        gw = weights_full[weights_full.gate == gate]
        if gw.empty:
            print(f"  {gate}: no v2 ranker weights — skip")
            per_gate_picks[gate] = None
            continue
        gv = val[val[gate]].copy()
        if gv.empty:
            print(f"  {gate}: no VAL trigger rows — skip")
            per_gate_picks[gate] = None
            continue
        gv["rank_score"] = build_ranker_score(gv, gw, gate)
        picks = _topk_per_gate(gv, "rank_score", h, k)
        per_gate_picks[gate] = picks
        # per-gate solo daily
        daily, _ = _basket_daily(picks, h)
        m = _daily_metrics(daily)
        m.update({"basket": f"{gate}_top{k}", "method": "ranker_v2"})
        rows.append(m)

        # Per-gate RANDOM top-k for fair comparison in combined random
        gv["rand_score"] = rng.random(len(gv))
        per_gate_random_picks[gate] = _topk_per_gate(gv, "rand_score", h, k)

    # ===== Combined v2 ranker basket (union top-3 per gate) =====
    valid = [p for p in per_gate_picks.values() if p is not None and not p.empty]
    if valid:
        combined = pd.concat(valid, ignore_index=True)
        daily, avg_size = _basket_daily(combined, h)
        m = _daily_metrics(daily)
        m.update({"basket": f"combined_top{k}", "method": "ranker_v2",
                  "avg_basket_size": float(avg_size)})
        rows.append(m)

    # ===== Combined RANDOM basket (avg over seeds) =====
    seed_metrics = []
    for s in range(args.seeds):
        seed_picks = []
        for gate in GATES:
            gw = weights_full[weights_full.gate == gate]
            if gw.empty:
                continue
            gv = val[val[gate]].copy()
            if gv.empty:
                continue
            gv["rand_score"] = rng.random(len(gv))
            seed_picks.append(_topk_per_gate(gv, "rand_score", h, k))
        if not seed_picks:
            continue
        combined_rand = pd.concat(seed_picks, ignore_index=True)
        daily, avg_size = _basket_daily(combined_rand, h)
        seed_metrics.append((_daily_metrics(daily), avg_size))
    if seed_metrics:
        means = [m[0]["mean_daily_R_%"] for m in seed_metrics if m[0].get("days", 0)]
        pfs = [m[0]["PF_daily"] for m in seed_metrics
               if m[0].get("days", 0) and not np.isinf(m[0]["PF_daily"])]
        cums = [m[0]["cum_R_%"] for m in seed_metrics if m[0].get("days", 0)]
        sizes = [m[1] for m in seed_metrics]
        rows.append({
            "basket": f"combined_top{k}", "method": f"random_avg_seeds={args.seeds}",
            "days": seed_metrics[0][0].get("days", 0),
            "mean_daily_R_%": float(np.nanmean(means)),
            "median_daily_R_%": np.nan,
            "hit_daily_%": np.nan,
            "PF_daily": float(np.nanmean(pfs)) if pfs else np.nan,
            "cum_R_%": float(np.nanmean(cums)),
            "avg_basket_size": float(np.nanmean(sizes)),
        })

    # ===== Combined all-triggers (no ranker, all 3 gates' rows pooled) =====
    any_trig = val[val[GATES].any(axis=1)].copy()
    daily, avg_size = _basket_daily(any_trig, h)
    m = _daily_metrics(daily)
    m.update({"basket": "combined_all_triggers", "method": "no_ranker",
              "avg_basket_size": float(avg_size)})
    rows.append(m)

    out_df = pd.DataFrame(rows)
    cols = ["basket", "method", "days", "avg_basket_size",
            "mean_daily_R_%", "median_daily_R_%", "hit_daily_%",
            "PF_daily", "cum_R_%"]
    out_df = out_df[[c for c in cols if c in out_df.columns]]
    out_path = out_dir / f"screener_combo_v1_combined_basket_{args.tag}_h{h}.csv"
    out_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  → {out_path}")
    print()
    print(f"=== Combined daily basket | VAL | h={h} | top-{k} per gate ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
