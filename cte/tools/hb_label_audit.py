"""
CTE HB runner_{10,15,20} label audit.

Audits whether the HB seed-bag "runner_20 is better" finding (cte_hb_v2) is
a genuine ranking improvement or a base-rate artifact (looser label →
more positives → cheaper lift@10).

Method
------
1. Read production preds. For each row, read the three runner labels.
2. Cross-tabulate: runner_15 vs runner_20 (and runner_10 vs runner_15).
   Count pos/neg/pos-only/neg-only rows per fold.
3. For each horizon, re-train using the *existing* contract-compliant
   train_line() call, then eval on the same test pool. Compare
   lift@10 / rho / PF_top30 cleanly.
4. Also compute the model-vs-compression decile agreement per horizon.

No dataset change, no production artifact overwrite. Writes only diag outputs.

Outputs
-------
  output/cte_hb_diag_label_audit.csv      — per-horizon overall + fold metrics
  output/cte_hb_diag_label_crosstab.csv   — pairwise label transitions
  memory/cte_hb_label_audit.md
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
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


HORIZONS = ("runner_10", "runner_15", "runner_20")
MFE_FOR = {"runner_10": ("mfe_10_atr", "mae_10_atr"),
           "runner_15": ("mfe_15_atr", "mae_15_atr"),
           "runner_20": ("mfe_20_atr", "mae_20_atr")}


def _label_crosstab(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    sub = df[[a, b]].dropna()
    if sub.empty:
        return pd.DataFrame()
    ct = pd.crosstab(sub[a].astype(int), sub[b].astype(int),
                     rownames=[a], colnames=[b], margins=True, margins_name="all")
    return ct


def _label_rates(df: pd.DataFrame, fold: str | None = None) -> dict:
    if fold is not None:
        df = df[df["fold_assigned"] == fold]
    out = {"fold": fold or "all", "n": len(df)}
    for h in HORIZONS:
        if h not in df.columns:
            continue
        valid = df[h].dropna()
        out[f"{h}_base"] = float(valid.mean()) if len(valid) else float("nan")
        out[f"{h}_n_pos"] = int(valid.sum()) if len(valid) else 0
        out[f"{h}_n_valid"] = int(len(valid))
    return out


def _horizon_eval(df_train: pd.DataFrame, target: str) -> dict:
    """Train one horizon with baseline config (seed=17, mcs=20, 35 feats,
    mixed), then eval on its test pool. Returns overall + per-fold metrics."""
    feats = [c for c in FEATURES_V1 if c in df_train.columns]
    res = train_line(df=df_train, line="hb", mode="mixed", target=target,
                     feature_cols=feats, split=CONFIG.split, params=LGBMParams())
    e = eval_line(res.preds, target=target, line="hb")
    over = e.get("overall", {})
    m = over.get("scores", {}).get("score_model", {})
    c = over.get("scores", {}).get("score_compression", {})
    out = {
        "horizon": target,
        "n_test": over.get("n", 0),
        "base": over.get("base", float("nan")),
        "model_rho": m.get("rho", float("nan")),
        "model_lift10": m.get("lift@10", float("nan")),
        "model_lift20": m.get("lift@20", float("nan")),
        "comp_lift10": c.get("lift@10", float("nan")),
        "comp_rho": c.get("rho", float("nan")),
        "pf_overall": over.get("pf_proxy_overall", {}).get("PF_proxy", float("nan")),
        "pf_top30": over.get("pf_proxy_top30", {}).get("PF_proxy", float("nan")),
        "wr_top30": over.get("pf_proxy_top30", {}).get("WR_proxy", float("nan")),
    }
    byf = e.get("by_fold", {})
    for fold in ("fold1", "fold2", "fold3"):
        b = byf.get(fold, {})
        mm = b.get("scores", {}).get("score_model", {})
        out[f"{fold}_base"] = b.get("base", float("nan"))
        out[f"{fold}_lift10"] = mm.get("lift@10", float("nan"))
        out[f"{fold}_rho"] = mm.get("rho", float("nan"))
    return out, res.preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--preds", default="output/cte_hb_preds_v1.parquet",
                    help="Used for label crosstab on the HB candidate pool")
    ap.add_argument("--out-csv", default="output/cte_hb_diag_label_audit.csv")
    ap.add_argument("--out-crosstab",
                    default="output/cte_hb_diag_label_crosstab.csv")
    ap.add_argument("--out-md", default=str(
        Path.home() / ".claude/projects/-Users-alpkarakasli-Projects-nox-project/memory/cte_hb_label_audit.md"))
    ap.add_argument("--ignore-contract", action="store_true")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not args.ignore_contract:
        verify_contract(
            dataset_path=args.dataset,
            contract_path=str(CONTRACT_PATH),
            raise_on_mismatch=True, verbose=True, line="hb",
        )

    ds = pd.read_parquet(args.dataset)
    ds["date"] = pd.to_datetime(ds["date"])
    print(f"[DATA] dataset shape={ds.shape}")

    # ── Part A: label base rates + crosstab on HB candidate pool ──
    hb_pool = ds[ds["trigger_hb"].astype(bool)].copy()
    print(f"[POOL] HB candidates in dataset: {len(hb_pool)}")

    # Fold-assignment comes from preds; enrich pool so fold slicing is consistent.
    preds = pd.read_parquet(args.preds)
    preds["date"] = pd.to_datetime(preds["date"])
    pool_enriched = hb_pool.merge(
        preds[["ticker", "date", "fold_assigned"]],
        on=["ticker", "date"], how="left")
    pool_test = pool_enriched[pool_enriched["fold_assigned"].notna()].copy()
    print(f"[POOL] matched to test folds: {len(pool_test)} / {len(hb_pool)}")

    # Rates per fold
    rates = [_label_rates(pool_test)]
    for fold in ("fold1", "fold2", "fold3"):
        rates.append(_label_rates(pool_test, fold))
    rates_df = pd.DataFrame(rates)
    print("\n═══ Label base rates per fold (HB test pool) ═══")
    print(rates_df.to_string(index=False))

    # Crosstab
    cts = {}
    for a, b in [("runner_10", "runner_15"), ("runner_15", "runner_20")]:
        ct = _label_crosstab(pool_test, a, b)
        cts[f"{a}_vs_{b}"] = ct
        print(f"\n═══ Crosstab {a} vs {b} ═══")
        print(ct)

    # Combined crosstab flat csv
    flat_rows = []
    for name, ct in cts.items():
        if ct.empty:
            continue
        for r in ct.index:
            for c in ct.columns:
                flat_rows.append({"pair": name, "a_value": r, "b_value": c,
                                  "count": int(ct.loc[r, c])})
    pd.DataFrame(flat_rows).to_csv(args.out_crosstab, index=False)
    print(f"\n[WRITE] {args.out_crosstab}")

    # ── Part B: retrain + eval per horizon (no artifact write) ──
    print("\n═══ Per-horizon retrain + eval ═══")
    eval_rows = []
    for h in HORIZONS:
        if h not in ds.columns:
            print(f"  {h} not in dataset, skipping")
            continue
        out, _preds = _horizon_eval(ds, h)
        eval_rows.append(out)
        print(f"  {h}: N={out['n_test']}  base={out['base']:.3f}  "
              f"lift@10={out['model_lift10']:.2f}x  rho={out['model_rho']:+.3f}  "
              f"PF_t30={out['pf_top30']:.2f}  "
              f"fold123 lift@10={out['fold1_lift10']:.2f}/{out['fold2_lift10']:.2f}/"
              f"{out['fold3_lift10']:.2f}")

    eval_df = pd.DataFrame(eval_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(args.out_csv, index=False)
    print(f"\n[WRITE] {args.out_csv}")

    # ── Part C: markdown report ──
    md = []
    md.append("---")
    md.append("name: HB label-horizon audit")
    md.append("description: HB runner_10/15/20 label tabanı + pairwise transitions + horizon başı train/eval. runner_20 lift artışının base-rate artifact mı yoksa gerçek rank ilerleyişi mi olduğuna karar verdirir.")
    md.append("type: project")
    md.append("---")
    md.append(f"Contract v5 altında, dataset `{args.dataset}`. Production artifact YAZILMADI.")
    md.append("")
    md.append("## 1. Label base rates (HB test pool)")
    md.append("")
    md.append("| fold | N | runner_10 base / n_pos | runner_15 base / n_pos | runner_20 base / n_pos |")
    md.append("|---|---:|---:|---:|---:|")
    for r in rates:
        def _fmt(h):
            b = r.get(f"{h}_base", float("nan"))
            n = r.get(f"{h}_n_pos", 0)
            v = r.get(f"{h}_n_valid", 0)
            if not np.isfinite(b):
                return "—"
            return f"{b:.3f} ({n}/{v})"
        md.append(f"| {r['fold']} | {r['n']} | {_fmt('runner_10')} | "
                  f"{_fmt('runner_15')} | {_fmt('runner_20')} |")
    md.append("")
    md.append("## 2. Label transitions (pairwise crosstab)")
    for name, ct in cts.items():
        md.append(f"")
        md.append(f"**{name}**")
        md.append("")
        md.append(ct.to_markdown() if not ct.empty else "(empty)")
    md.append("")
    md.append("## 3. Per-horizon retrain (baseline config)")
    md.append("")
    md.append("Config: mcs=20, 35 feats, mixed mode, seed=17. Same split, same feature set.")
    md.append("")
    md.append("| horizon | N | base | model_lift@10 | model_rho | PF_top30 | WR_top30 | fold1 lift | fold2 lift | fold3 lift | comp_lift@10 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in eval_rows:
        md.append(
            f"| {r['horizon']} | {r['n_test']} | {r['base']:.3f} | "
            f"{r['model_lift10']:.2f}x | {r['model_rho']:+.3f} | "
            f"{r['pf_top30']:.2f} | {r['wr_top30']:.3f} | "
            f"{r['fold1_lift10']:.2f}x | {r['fold2_lift10']:.2f}x | "
            f"{r['fold3_lift10']:.2f}x | {r['comp_lift10']:.2f}x |"
        )
    md.append("")

    # Interpretation helper: is runner_20 better PURELY on rho when base rate
    # is renormalised? lift@10 mechanically changes with base; rho does not.
    md.append("## 4. Rank-quality vs base-rate artifact")
    md.append("")
    md.append("`lift@10 = p@10 / base` — so base rate changes mechanically shift lift.")
    md.append("`rho` is rank-only and does NOT depend on base rate. Better cross-horizon")
    md.append("comparison.")
    md.append("")
    md.append("- Baseline (runner_15) rho: see row above.")
    md.append("- runner_10 rho vs runner_15 rho: same rows, different label assignment.")
    md.append("- runner_20 rho vs runner_15 rho: same rows, different label assignment.")
    md.append("")
    md.append("If runner_20's rho > runner_15's rho, the model ranks HB candidates better")
    md.append("when the reward horizon is longer — that's a genuine signal-shape finding,")
    md.append("not a base-rate trick.")
    md.append("")
    md.append("If runner_20's lift@10 > runner_15's lift@10 but rho is similar/worse,")
    md.append("the apparent 'improvement' comes from more positives in the top-10 bucket")
    md.append("— base-rate artifact.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print(f"\n[WRITE] {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
