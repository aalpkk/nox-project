"""Apply discovered ranker on VAL — top-K lift vs alphabetical baseline.

Lift verdict per (gate, top_K, horizon):
   PF_rank_top_K   vs   PF_alpha_top_K   vs   PF_random_top_K (mean over seeds)

Output:
  output/screener_combo_v1_rank_val_{tag}_h{H}_top{K}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener_combo.data_prep import split_bounds


GATES = ["regime_trig", "weekly_trig", "alsat_trig"]


# Features that are duplicates of each other (same value in pipeline).
# Drop one when constructing the ranker so we don't double-count.
_DEDUP_DROP = {"drawdown_20d_pct"}  # keep dist_20d_high_pct


def _zscore_winsorized(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_ranker_score(table: pd.DataFrame, weights: pd.DataFrame, gate: str) -> pd.Series:
    """Return ranker score (higher = better expected R) for the gate's rows.

    weights: subset of ranker_weights for `gate` (cols: feature, rho, weight).
    """
    feats = weights[~weights.feature.isin(_DEDUP_DROP)].copy()
    if feats.empty:
        return pd.Series(np.nan, index=table.index)

    z_sum = pd.Series(0.0, index=table.index)
    n_terms = 0
    for _, row in feats.iterrows():
        f = row["feature"]
        if f not in table.columns:
            continue
        z = _zscore_winsorized(table[f])
        # weight = sign(rho); use |rho| as relative importance
        z_sum = z_sum.add(np.sign(row["rho"]) * abs(row["rho"]) * z.fillna(0), fill_value=0)
        n_terms += 1
    if n_terms == 0:
        return pd.Series(np.nan, index=table.index)
    return z_sum


def topk_metrics(rows: pd.DataFrame, score_col: str, horizon: int, top_k: int,
                 ascending: bool = False) -> dict:
    """Per-day top-K by `score_col`, equal-weight average return at horizon."""
    r_col = f"fwd_R_{horizon}"
    rows = rows[[score_col, r_col, "ticker", "date"]].dropna(subset=[r_col]).copy()
    if rows.empty:
        return {"days": 0}
    if ascending:
        rows[score_col] = -rows[score_col]
    daily = (
        rows.sort_values(["date", score_col], ascending=[True, False])
        .groupby("date")
        .head(top_k)
        .groupby("date")[r_col]
        .mean()
    )
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--top-k", type=int, nargs="*", default=[3, 5, 10])
    ap.add_argument("--seeds", type=int, default=5)  # for random baseline
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    weights_path = out_dir / f"screener_combo_v1_ranker_weights_{args.tag}_h{args.horizon}.csv"
    if not triggers_path.exists() or not weights_path.exists():
        raise FileNotFoundError("missing scan or ranker output")
    table = pd.read_parquet(triggers_path)
    table["date"] = pd.to_datetime(table["date"])
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
            print(f"  {gate}: no ranker weights — skip")
            continue
        gv = val[val[gate]].copy()
        if gv.empty:
            print(f"  {gate}: no VAL trigger rows — skip")
            continue
        gv["rank_score"] = build_ranker_score(gv, gw, gate)
        gv["alpha_score"] = -gv["ticker"].map(lambda x: ord(x[0]))  # alphabetical (negative for desc-pick = ascending)

        for k in args.top_k:
            # ranker
            mr = topk_metrics(gv, "rank_score", args.horizon, k, ascending=False)
            mr.update({"gate": gate, "top_k": k, "method": "ranker"})
            rows.append(mr)

            # alphabetical baseline
            ma = topk_metrics(gv.assign(alpha_idx=range(len(gv))),
                              "alpha_idx", args.horizon, k, ascending=True)
            ma.update({"gate": gate, "top_k": k, "method": "alphabetical"})
            rows.append(ma)

            # random baseline (avg over seeds)
            seed_pfs, seed_means = [], []
            for s in range(args.seeds):
                gv["rand_score"] = rng.random(len(gv))
                m = topk_metrics(gv, "rand_score", args.horizon, k, ascending=False)
                if "PF_daily" in m and m["days"] > 0:
                    seed_pfs.append(m.get("PF_daily", np.nan))
                    seed_means.append(m.get("mean_daily_R_%", np.nan))
            if seed_pfs:
                rows.append({
                    "gate": gate, "top_k": k, "method": f"random_avg_seeds={args.seeds}",
                    "days": int(gv.date.nunique()),
                    "mean_daily_R_%": float(np.nanmean(seed_means)),
                    "median_daily_R_%": np.nan,
                    "hit_daily_%": np.nan,
                    "PF_daily": float(np.nanmean([p for p in seed_pfs if not np.isinf(p)])) if any(not np.isinf(p) for p in seed_pfs) else np.nan,
                    "cum_R_%": np.nan,
                })

    out_df = pd.DataFrame(rows)
    out_df = out_df[["gate", "top_k", "method", "days", "mean_daily_R_%",
                     "median_daily_R_%", "hit_daily_%", "PF_daily", "cum_R_%"]]
    out_path = out_dir / f"screener_combo_v1_rank_val_{args.tag}_h{args.horizon}.csv"
    out_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  → {out_path}")

    # Pretty-print per gate
    for gate in GATES:
        sub = out_df[out_df.gate == gate]
        if sub.empty:
            continue
        print(f"\n=== {gate} | VAL | h={args.horizon} ===")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
