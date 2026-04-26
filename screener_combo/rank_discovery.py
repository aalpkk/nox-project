"""Data-driven ranker discovery on TRAIN.

For each gate cohort, compute Spearman rho between each candidate feature and
fwd_R_h20 ON TRAIN ONLY. Features with |rho| above a threshold and adequate
sample size become ranker components. Composite ranker score:

    ranker_score = sum_i sign(rho_i) * z(feature_i)

clipped + standardized so that top-K selection is invariant to absolute scale.

Output:
  output/screener_combo_v1_ranker_weights_{tag}.csv  — gate × feature × rho × n_used
  print summary
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from screener_combo.features import FEATURE_COLS
from screener_combo.data_prep import split_bounds


GATES = ["regime_trig", "weekly_trig", "alsat_trig"]
HORIZONS = [5, 10, 20]


def _drop_redundant_in_gate(rho_df: pd.DataFrame, gate: str) -> pd.DataFrame:
    """Avoid double-counting features that are essentially the gate's own trigger
    rule (would create circularity / overfit on validation).

    Heuristic: each gate's natural fingerprint feature gets a soft demote.
    """
    return rho_df  # no-op for now; documented for future tightening


def discover(table: pd.DataFrame, tag: str = "trainval", min_n: int = 200,
             rho_threshold: float = 0.05) -> pd.DataFrame:
    """For each (gate, feature, horizon), compute Spearman rho on TRAIN.

    Returns a long-form DataFrame:
       gate, feature, horizon, rho, p_value, n_used
    """
    bounds = split_bounds()
    train_lo, train_hi = bounds["train"]
    table["date"] = pd.to_datetime(table["date"])
    train = table[(table.date >= train_lo) & (table.date <= train_hi)]

    rows = []
    for gate in GATES:
        sub = train[train[gate]]
        n_gate = len(sub)
        for h in HORIZONS:
            r_col = f"fwd_R_{h}"
            if r_col not in sub.columns:
                continue
            for feat in FEATURE_COLS:
                if feat not in sub.columns:
                    continue
                pair = sub[[feat, r_col]].dropna()
                if len(pair) < min_n:
                    rows.append({
                        "gate": gate, "feature": feat, "horizon": h,
                        "rho": np.nan, "p_value": np.nan, "n_used": len(pair),
                        "n_gate": n_gate,
                    })
                    continue
                # Drop infinities
                pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < min_n:
                    continue
                try:
                    rho, p = spearmanr(pair[feat], pair[r_col])
                except Exception:
                    rho, p = np.nan, np.nan
                rows.append({
                    "gate": gate, "feature": feat, "horizon": h,
                    "rho": float(rho), "p_value": float(p), "n_used": len(pair),
                    "n_gate": n_gate,
                })
    rho_df = pd.DataFrame(rows)
    return rho_df


def select_predictive(rho_df: pd.DataFrame, horizon: int = 20,
                      rho_min: float = 0.05, top_n: int = 8) -> dict[str, pd.DataFrame]:
    """Per gate, return predictive features for given horizon.

    Selection rule: |rho| >= rho_min, p < 0.10, then keep top_n by |rho|.
    """
    out = {}
    sub = rho_df[rho_df.horizon == horizon].copy()
    sub["abs_rho"] = sub["rho"].abs()
    for gate in GATES:
        gsub = sub[(sub.gate == gate) & sub.rho.notna()].copy()
        keep = gsub[(gsub.abs_rho >= rho_min) & (gsub.p_value < 0.10)]
        keep = keep.sort_values("abs_rho", ascending=False).head(top_n)
        out[gate] = keep[["feature", "rho", "p_value", "n_used"]].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--rho-min", type=float, default=0.05)
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    if not triggers_path.exists():
        raise FileNotFoundError(f"missing: {triggers_path} — run screener_combo.run first")
    table = pd.read_parquet(triggers_path)
    print(f"loaded triggers: {len(table):,} rows, {table.shape[1]} cols")

    rho_df = discover(table, tag=args.tag, rho_threshold=args.rho_min)
    rho_path = out_dir / f"screener_combo_v1_rho_train_{args.tag}.csv"
    rho_df.to_csv(rho_path, index=False, float_format="%.4f")
    print(f"  → {rho_path}")

    preds = select_predictive(rho_df, horizon=args.horizon, rho_min=args.rho_min, top_n=args.top_n)
    print()
    print(f"=== TRAIN-discovered predictive features (horizon={args.horizon}, |rho| >= {args.rho_min}) ===")
    for gate, df in preds.items():
        gate_n = int(rho_df[(rho_df.gate == gate) & (rho_df.horizon == args.horizon)].n_gate.max() or 0)
        print(f"\n[{gate}] gate N (train) = {gate_n}")
        if df.empty:
            print("  (no feature passed threshold)")
            continue
        print(df.to_string(index=False))

    # Save predictive list as ranker weights
    rows = []
    for gate, df in preds.items():
        for _, r in df.iterrows():
            rows.append({"gate": gate, "feature": r.feature,
                         "rho": r.rho, "weight": np.sign(r.rho),
                         "n_used": int(r.n_used)})
    rk = pd.DataFrame(rows)
    rk_path = out_dir / f"screener_combo_v1_ranker_weights_{args.tag}_h{args.horizon}.csv"
    rk.to_csv(rk_path, index=False, float_format="%.4f")
    print(f"\n  → {rk_path}")


if __name__ == "__main__":
    main()
