"""Category-only edge validation on VAL.

Test whether the high-PF categories discovered on TRAIN survive on VAL —
WITHOUT any ranker. Pure category filter → cohort metrics.

For each candidate:
  N_train, PF_train, mean_R_train_%   (reference)
  N_val,   PF_val,   mean_R_val_%, hit_val_%

Lift expressed against same-gate baseline (boolean trigger only).

Output:
  output/screener_combo_v1_category_validate_{tag}_h{H}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from screener_combo.data_prep import split_bounds


def _safe_pf(r: pd.Series) -> float:
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def _stats(rows: pd.DataFrame, h: int) -> dict:
    col = f"fwd_R_{h}"
    r = rows[col].dropna()
    if r.empty:
        return {"n": 0, "PF": np.nan, "mean_R_%": np.nan,
                "median_R_%": np.nan, "hit_%": np.nan}
    return {
        "n": int(len(r)),
        "PF": _safe_pf(r),
        "mean_R_%": float(r.mean()) * 100,
        "median_R_%": float(r.median()) * 100,
        "hit_%": float((r > 0).mean()) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path("output")
    triggers_path = out_dir / f"screener_combo_v1_triggers_{args.tag}.parquet"
    table = pd.read_parquet(triggers_path)
    table["date"] = pd.to_datetime(table["date"])
    bounds = split_bounds()
    tr_lo, tr_hi = bounds["train"]
    vl_lo, vl_hi = bounds["val"]
    train = table[(table.date >= tr_lo) & (table.date <= tr_hi)]
    val = table[(table.date >= vl_lo) & (table.date <= vl_hi)]
    print(f"TRAIN rows: {len(train):,}  VAL rows: {len(val):,}")

    h = args.horizon
    rows = []

    # Helper: emit a row for a given filter
    def emit(name, gate, train_mask, val_mask):
        st_tr = _stats(train[train_mask], h)
        st_vl = _stats(val[val_mask], h)
        rows.append({
            "filter": name, "gate": gate,
            "N_train": st_tr["n"], "PF_train": st_tr["PF"],
            "mean_R_train_%": st_tr["mean_R_%"], "hit_train_%": st_tr["hit_%"],
            "N_val": st_vl["n"], "PF_val": st_vl["PF"],
            "mean_R_val_%": st_vl["mean_R_%"], "median_R_val_%": st_vl["median_R_%"],
            "hit_val_%": st_vl["hit_%"],
        })

    # ===== Baselines (full gate, boolean) =====
    emit("BASE: regime_trig",  "regime_trig", train.regime_trig, val.regime_trig)
    emit("BASE: weekly_trig",  "weekly_trig", train.weekly_trig, val.weekly_trig)
    emit("BASE: alsat_trig",   "alsat_trig",  train.alsat_trig,  val.alsat_trig)
    emit("BASE: nox_any",      "nox_any",     train.nox_dw_type != "", val.nox_dw_type != "")

    # ===== RT DONUS tiers =====
    for tier in ["ALTIN", "GUMUS", "BRONZ", "NORMAL", "ELE"]:
        m_tr = train.regime_trig & (train.rt_subtype == "DONUS") & (train.rt_tier == tier)
        m_vl = val.regime_trig   & (val.rt_subtype   == "DONUS") & (val.rt_tier   == tier)
        emit(f"RT DONUS_tier={tier}", "regime_trig", m_tr, m_vl)
    # RT subtype only
    for sub in ["DONUS", "SICRAMA"]:
        emit(f"RT subtype={sub}", "regime_trig",
             train.regime_trig & (train.rt_subtype == sub),
             val.regime_trig   & (val.rt_subtype   == sub))

    # ===== AS subtype =====
    for sub in ["GUCLU", "ZAYIF", "DONUS", "PB", "SQ", "MR", "BILESEN"]:
        emit(f"AS subtype={sub}", "alsat_trig",
             train.alsat_trig & (train.as_subtype == sub),
             val.alsat_trig   & (val.as_subtype   == sub))
    # AS decision (any sub-component fired, not just AL)
    for dec in ["AL", "IZLE", "ATLA"]:
        emit(f"AS decision={dec}", "as_anytrig",
             train.as_decision == dec,
             val.as_decision   == dec)

    # ===== NOX type =====
    for typ in ["D", "W", "DW"]:
        emit(f"NOX type={typ}", "nox_any",
             train.nox_dw_type == typ, val.nox_dw_type == typ)

    # ===== AS × RT-age =====
    age_buckets = [
        ("RT_age=0",     0, 0),
        ("RT_age=1-3",   1, 3),
        ("RT_age=4-10",  4, 10),
        ("RT_age>10",   11, 99999),
    ]
    for label, lo, hi in age_buckets:
        m_tr = train.alsat_trig & train.days_since_RT.between(lo, hi)
        m_vl = val.alsat_trig   & val.days_since_RT.between(lo, hi)
        emit(f"AS × {label}", "alsat_trig", m_tr, m_vl)

    # ===== RT × GIR (entry_score 0-4) — only on RT trigger days =====
    for g in [0, 1, 2, 3, 4]:
        emit(f"RT GIR={g}", "regime_trig",
             train.regime_trig & (train.rt_entry_score == g),
             val.regime_trig   & (val.rt_entry_score   == g))
    # Coarse buckets
    emit("RT GIR>=3", "regime_trig",
         train.regime_trig & (train.rt_entry_score >= 3),
         val.regime_trig   & (val.rt_entry_score   >= 3))
    emit("RT GIR<=2", "regime_trig",
         train.regime_trig & (train.rt_entry_score <= 2),
         val.regime_trig   & (val.rt_entry_score   <= 2))

    # ===== RT × OK (oe_score 0-4) — clean entry vs overextended =====
    for o in [0, 1, 2, 3, 4]:
        emit(f"RT OK={o}", "regime_trig",
             train.regime_trig & (train.rt_oe_score == o),
             val.regime_trig   & (val.rt_oe_score   == o))
    emit("RT OK=0 (clean)", "regime_trig",
         train.regime_trig & (train.rt_oe_score == 0),
         val.regime_trig   & (val.rt_oe_score   == 0))
    emit("RT OK>=2 (extended)", "regime_trig",
         train.regime_trig & (train.rt_oe_score >= 2),
         val.regime_trig   & (val.rt_oe_score   >= 2))

    # ===== RT × GIR × OK joint (golden entry hypothesis) =====
    emit("RT GIR=4 × OK=0 (golden)", "regime_trig",
         train.regime_trig & (train.rt_entry_score == 4) & (train.rt_oe_score == 0),
         val.regime_trig   & (val.rt_entry_score   == 4) & (val.rt_oe_score   == 0))
    emit("RT GIR>=3 × OK=0", "regime_trig",
         train.regime_trig & (train.rt_entry_score >= 3) & (train.rt_oe_score == 0),
         val.regime_trig   & (val.rt_entry_score   >= 3) & (val.rt_oe_score   == 0))
    emit("RT GIR<=1 × OK>=2 (worst)", "regime_trig",
         train.regime_trig & (train.rt_entry_score <= 1) & (train.rt_oe_score >= 2),
         val.regime_trig   & (val.rt_entry_score   <= 1) & (val.rt_oe_score   >= 2))

    # ===== Compound winners (TRAIN-best categories combined) =====
    # AS-ZAYIF × RT_age>10
    emit("AS=ZAYIF × RT_age>10", "alsat_trig",
         train.alsat_trig & (train.as_subtype == "ZAYIF") & (train.days_since_RT > 10),
         val.alsat_trig   & (val.as_subtype   == "ZAYIF") & (val.days_since_RT   > 10))
    # AS-IZLE-or-ATLA (since AL is worst) — treat as a non-AL high-PF mode
    emit("AS decision=IZLE|ATLA", "as_anytrig",
         train.as_decision.isin(["IZLE", "ATLA"]),
         val.as_decision.isin(["IZLE", "ATLA"]))
    # Pure NOX-W (no AS gate) — biggest single-feature lift
    emit("NOX W only", "nox_any",
         train.nox_dw_type == "W",
         val.nox_dw_type   == "W")

    df = pd.DataFrame(rows)

    # Add lift columns vs gate baseline
    base_pf = {r["gate"]: r["PF_val"] for r in rows if r["filter"].startswith("BASE:")}
    df["PF_val_vs_base"] = df.apply(
        lambda r: (r["PF_val"] - base_pf.get(r["gate"], np.nan))
        if r["gate"] in base_pf else np.nan, axis=1
    )

    out_path = out_dir / f"screener_combo_v1_category_validate_{args.tag}_h{args.horizon}.csv"
    df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  → {out_path}")
    print()
    cols = ["filter", "N_train", "PF_train", "N_val", "PF_val",
            "mean_R_val_%", "hit_val_%", "PF_val_vs_base"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
