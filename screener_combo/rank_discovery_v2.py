"""Ranker v2 discovery — adds category dummies + age features.

For each gate, expand the feature set with:
  - Category dummies (RT: subtype/tier; AS: subtype/decision; NOX: dw_type)
  - Signal-age scalars (days_since_RT, days_since_NW, days_since_AS, days_since_ND)
  - is_fresh_<gate> (≤1 bar) flags

Compute Spearman rho per (gate, feature, horizon) on TRAIN, save weights.

Output:
  output/screener_combo_v1_rho_train_v2_{tag}.csv
  output/screener_combo_v1_ranker_weights_v2_{tag}_h{H}.csv
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


def expand_features(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add category dummies + age features to the trigger table.
    Returns (table_with_new_cols, list_of_new_feature_names).
    """
    new_cols = []
    df = table.copy()

    # ---- RT subtype/tier dummies ----
    for sub in ["DONUS", "SICRAMA"]:
        col = f"rt_is_{sub}"
        df[col] = (df["rt_subtype"] == sub).astype(int)
        new_cols.append(col)
    for tier in ["ALTIN", "GUMUS", "BRONZ", "NORMAL", "ELE"]:
        col = f"rt_tier_is_{tier}"
        df[col] = (df["rt_tier"] == tier).astype(int)
        new_cols.append(col)
    # ordinal tier (ALTIN=4 > GUMUS=3 > BRONZ=2 > NORMAL=1 > ELE=0)
    tier_map = {"ALTIN": 4, "GUMUS": 3, "BRONZ": 2, "NORMAL": 1, "ELE": 0, "": np.nan}
    df["rt_tier_ord"] = df["rt_tier"].map(tier_map)
    new_cols.append("rt_tier_ord")

    # ---- AS subtype dummies ----
    for sub in ["CMB+", "CMB", "GUCLU", "ZAYIF", "DONUS", "ERKEN",
                "PB", "SQ", "MR", "BILESEN"]:
        col = f"as_is_{sub.replace('+', 'P')}"
        df[col] = (df["as_subtype"] == sub).astype(int)
        new_cols.append(col)
    # AS decision (ordinal: AL=2, IZLE=1, ATLA=0, "-" = NaN)
    dec_map = {"AL": 2, "IZLE": 1, "ATLA": 0, "-": np.nan, "": np.nan}
    df["as_decision_ord"] = df["as_decision"].map(dec_map)
    new_cols.append("as_decision_ord")

    # ---- NOX D/W/DW dummies ----
    for typ in ["D", "W", "DW"]:
        col = f"nox_is_{typ}"
        df[col] = (df["nox_dw_type"] == typ).astype(int)
        new_cols.append(col)

    # ---- RT GIR (entry_score 0-4) / OK (oe_score 0-4) ----
    if "rt_entry_score" in df.columns:
        new_cols.append("rt_entry_score")
        df["rt_gir_eq0"] = (df["rt_entry_score"] == 0).astype(int)
        df["rt_gir_ge3"] = (df["rt_entry_score"] >= 3).astype(int)
        new_cols += ["rt_gir_eq0", "rt_gir_ge3"]
    if "rt_oe_score" in df.columns:
        new_cols.append("rt_oe_score")
        df["rt_ok_eq0"] = (df["rt_oe_score"] == 0).astype(int)
        df["rt_ok_ge2"] = (df["rt_oe_score"] >= 2).astype(int)
        new_cols += ["rt_ok_eq0", "rt_ok_ge2"]
    if "rt_entry_score" in df.columns and "rt_oe_score" in df.columns:
        df["rt_gir_ge3_x_ok_eq0"] = (
            (df["rt_entry_score"] >= 3) & (df["rt_oe_score"] == 0)
        ).astype(int)
        new_cols.append("rt_gir_ge3_x_ok_eq0")

    # ---- Age features ----
    for c in ["days_since_RT", "days_since_NW", "days_since_AS", "days_since_ND"]:
        if c in df.columns:
            new_cols.append(c)
            # is_fresh: fired today or yesterday
            fresh_col = f"is_fresh_{c.split('_')[-1]}"
            df[fresh_col] = (df[c] <= 1).astype(int)
            new_cols.append(fresh_col)

    return df, new_cols


def discover(table: pd.DataFrame, extra_cols: list[str], min_n: int = 200) -> pd.DataFrame:
    bounds = split_bounds()
    train_lo, train_hi = bounds["train"]
    table["date"] = pd.to_datetime(table["date"])
    train = table[(table.date >= train_lo) & (table.date <= train_hi)]

    feat_set = list(FEATURE_COLS) + extra_cols
    rows = []
    for gate in GATES:
        sub = train[train[gate]]
        n_gate = len(sub)
        for h in HORIZONS:
            r_col = f"fwd_R_{h}"
            if r_col not in sub.columns:
                continue
            for feat in feat_set:
                if feat not in sub.columns:
                    continue
                pair = sub[[feat, r_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < min_n or pair[feat].nunique() < 2:
                    rows.append({
                        "gate": gate, "feature": feat, "horizon": h,
                        "rho": np.nan, "p_value": np.nan, "n_used": len(pair),
                        "n_gate": n_gate,
                    })
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
    return pd.DataFrame(rows)


def select_predictive(rho_df: pd.DataFrame, horizon: int = 20,
                      rho_min: float = 0.05, top_n: int = 10) -> dict[str, pd.DataFrame]:
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
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    if not triggers_path.exists():
        raise FileNotFoundError(f"missing: {triggers_path}")
    table = pd.read_parquet(triggers_path)
    print(f"loaded triggers: {len(table):,} rows, {table.shape[1]} cols")

    table, extra = expand_features(table)
    print(f"  added {len(extra)} category/age features")

    rho_df = discover(table, extra)
    rho_path = out_dir / f"screener_combo_v1_rho_train_v2_{args.tag}.csv"
    rho_df.to_csv(rho_path, index=False, float_format="%.4f")
    print(f"  → {rho_path}")

    preds = select_predictive(rho_df, horizon=args.horizon, rho_min=args.rho_min,
                              top_n=args.top_n)
    print()
    print(f"=== TRAIN-discovered features v2 (h={args.horizon}, |rho|≥{args.rho_min}) ===")
    for gate, df in preds.items():
        gate_n = int(rho_df[(rho_df.gate == gate) & (rho_df.horizon == args.horizon)]
                     .n_gate.max() or 0)
        print(f"\n[{gate}] gate N (train) = {gate_n}")
        if df.empty:
            print("  (no feature passed threshold)")
            continue
        print(df.to_string(index=False))

    rows = []
    for gate, df in preds.items():
        for _, r in df.iterrows():
            rows.append({"gate": gate, "feature": r.feature, "rho": r.rho,
                         "weight": np.sign(r.rho), "n_used": int(r.n_used)})
    rk = pd.DataFrame(rows)
    rk_path = out_dir / f"screener_combo_v1_ranker_weights_v2_{args.tag}_h{args.horizon}.csv"
    rk.to_csv(rk_path, index=False, float_format="%.4f")
    print(f"\n  → {rk_path}")


if __name__ == "__main__":
    main()
