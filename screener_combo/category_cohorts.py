"""Per-category cohort metrics on TRAIN.

Hypothesis: each signal carries internal sub-types (RT: DONUS/SICRAMA × tier,
AS: 10 sub-categories, NOX: D/W/DW) that may have very different forward-R
profiles. Compare cohort metrics per (gate, sub-category, horizon).

Output:
  output/screener_combo_v1_category_cohorts_{tag}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener_combo.data_prep import split_bounds


HORIZONS = [5, 10, 20]


def _safe_pf(r: pd.Series) -> float:
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def _cohort_stats(rows: pd.DataFrame, h: int) -> dict:
    col = f"fwd_R_{h}"
    r = rows[col].dropna()
    if r.empty:
        return {"n": 0}
    return {
        "n": int(len(r)),
        "mean_R_%": float(r.mean()) * 100,
        "median_R_%": float(r.median()) * 100,
        "hit_%": float((r > 0).mean()) * 100,
        "PF": _safe_pf(r),
        "p10_%": float(r.quantile(0.10)) * 100,
        "p90_%": float(r.quantile(0.90)) * 100,
    }


def per_category(table: pd.DataFrame) -> pd.DataFrame:
    """Cohort metrics per (gate, category, horizon) on TRAIN."""
    bounds = split_bounds()
    train_lo, train_hi = bounds["train"]
    table["date"] = pd.to_datetime(table["date"])
    train = table[(table.date >= train_lo) & (table.date <= train_hi)]

    out = []

    # ---- RT subtype × tier (only on regime_trig=True) ----
    rt = train[train.regime_trig]
    for sub in ["DONUS", "SICRAMA"]:
        rows = rt[rt.rt_subtype == sub]
        for h in HORIZONS:
            d = _cohort_stats(rows, h)
            d.update({"gate": "regime_trig", "category": f"subtype={sub}", "horizon": h})
            out.append(d)
    # DONUS tiers
    rtd = rt[rt.rt_subtype == "DONUS"]
    for tier in ["ALTIN", "GUMUS", "BRONZ", "NORMAL", "ELE"]:
        rows = rtd[rtd.rt_tier == tier]
        for h in HORIZONS:
            d = _cohort_stats(rows, h)
            d.update({"gate": "regime_trig", "category": f"DONUS_tier={tier}", "horizon": h})
            out.append(d)

    # ---- AS subtype (only on alsat_trig=True, i.e. decision == AL) ----
    asat = train[train.alsat_trig]
    for sub in ["CMB+", "CMB", "GUCLU", "ZAYIF", "DONUS", "ERKEN", "PB", "SQ", "MR", "BILESEN"]:
        rows = asat[asat.as_subtype == sub]
        for h in HORIZONS:
            d = _cohort_stats(rows, h)
            d.update({"gate": "alsat_trig", "category": f"subtype={sub}", "horizon": h})
            out.append(d)
    # AS decision tier (any sub-component, not just AL) — let user see IZLE/ATLA cohorts too
    for dec in ["AL", "IZLE", "ATLA"]:
        rows = train[train.as_decision == dec]
        for h in HORIZONS:
            d = _cohort_stats(rows, h)
            d.update({"gate": "as_anytrig", "category": f"decision={dec}", "horizon": h})
            out.append(d)

    # ---- NOX D vs W vs DW (any nox fire) ----
    nx = train[train.nox_dw_type != ""]
    for typ in ["D", "W", "DW"]:
        rows = nx[nx.nox_dw_type == typ]
        for h in HORIZONS:
            d = _cohort_stats(rows, h)
            d.update({"gate": "nox_any", "category": f"nox_type={typ}", "horizon": h})
            out.append(d)

    # ---- Cross-gate freshness: AS triggers split by RT age ----
    if "days_since_RT" in train.columns:
        as_with_rt = train[train.alsat_trig & train.days_since_RT.notna()]
        for label, mask in [
            ("RT_age=0",       as_with_rt.days_since_RT == 0),
            ("RT_age=1-3",    (as_with_rt.days_since_RT >= 1) & (as_with_rt.days_since_RT <= 3)),
            ("RT_age=4-10",   (as_with_rt.days_since_RT >= 4) & (as_with_rt.days_since_RT <= 10)),
            ("RT_age>10",     as_with_rt.days_since_RT > 10),
        ]:
            rows = as_with_rt[mask]
            for h in HORIZONS:
                d = _cohort_stats(rows, h)
                d.update({"gate": "alsat_trig", "category": label, "horizon": h})
                out.append(d)

    df = pd.DataFrame(out)
    return df[["gate", "category", "horizon", "n", "mean_R_%", "median_R_%",
               "hit_%", "PF", "p10_%", "p90_%"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    table = pd.read_parquet(triggers_path)
    print(f"loaded triggers: {len(table):,} rows")

    df = per_category(table)
    out_path = out_dir / f"screener_combo_v1_category_cohorts_{args.tag}.csv"
    df.to_csv(out_path, index=False, float_format="%.2f")
    print(f"  → {out_path}")
    print()

    # Pretty print per gate at horizon=20
    h = 20
    for gate in ["regime_trig", "alsat_trig", "as_anytrig", "nox_any"]:
        sub = df[(df.gate == gate) & (df.horizon == h)]
        if sub.empty:
            continue
        print(f"\n=== {gate} | h={h} (TRAIN) ===")
        print(sub[["category", "n", "mean_R_%", "median_R_%", "hit_%", "PF"]].to_string(index=False))


if __name__ == "__main__":
    main()
