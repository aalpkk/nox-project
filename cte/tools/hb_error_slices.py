"""
CTE HB error-slice diagnostic.

Joins HB production preds with the underlying feature dataset, then reports
how model edge (lift@10, rho, PF_top30) and compression edge behave across
structural slices:

  - compression_score tertile
  - breakout_vol_ratio tertile
  - bar_rvol tertile
  - hb_width_atr tertile
  - hb_close_density_atr tertile
  - rs_20 tertile
  - above_ma20 / above_ma50 (binary)
  - fold × setup_type
  - year / quarter

Goal: tell whether HB model works only inside a specific slice (regime
scoped) or everywhere. If model only wins in one slice, the aggregate
1.14x is a selective edge. If compression wins in different slices, that's
an ensemble hint.

Outputs
-------
  output/cte_hb_diag_slices.csv
  memory/cte_hb_diag_slices.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))


SLICE_FEATURES_TERTILE = (
    "compression_score",
    "breakout_vol_ratio",
    "bar_rvol",
    "hb_width_atr",
    "hb_close_density_atr",
    "rs_20",
    "bar_return_1d",
    "bb_width_pctile",
    "dryup_ratio_3_20",
    "bar_body_pct_range",
)

SLICE_FEATURES_BINARY = (
    "above_ma20",
    "above_ma50",
)


def _spearman(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return float("nan")
    return float(a[mask].rank().corr(b[mask].rank()))


def _precision_at(score, target, k_frac):
    mask = score.notna() & target.notna()
    s, y = score[mask], target[mask]
    if len(s) == 0:
        return float("nan")
    k = max(1, int(round(len(s) * k_frac)))
    top = s.nlargest(k).index
    return float(y.loc[top].mean())


def _pf_top30(df, score, target, mfe, mae):
    valid = df[df[score].notna() & df[target].notna() & df[mfe].notna() & df[mae].notna()]
    if len(valid) < 5:
        return float("nan"), float("nan"), 0
    cut = valid[score].quantile(0.70)
    top = valid[valid[score] >= cut]
    if len(top) == 0:
        return float("nan"), float("nan"), 0
    y = top[target].astype(int).values
    gain = np.minimum(top[mfe].values, 3.0)
    loss = -np.minimum(top[mae].values, 1.5)
    realised = np.where(y == 1, gain, loss)
    pos = realised[realised > 0].sum()
    neg = -realised[realised < 0].sum()
    pf = float(pos / neg) if neg > 0 else float("inf")
    wr = float((realised > 0).mean())
    return pf, wr, int(len(top))


def _slice_row(sub, name, value, target, mfe, mae):
    if len(sub) < 8:
        return None
    base = float(sub[target].mean()) if sub[target].notna().any() else float("nan")
    if not np.isfinite(base) or base == 0:
        return None
    # model
    m_rho = _spearman(sub["score_model"], sub[target])
    m_p10 = _precision_at(sub["score_model"], sub[target], 0.10)
    m_lift = m_p10 / base if base > 0 else float("nan")
    m_pf, m_wr, m_n = _pf_top30(sub, "score_model", target, mfe, mae)
    # compression
    c_rho = _spearman(sub["score_compression"], sub[target])
    c_p10 = _precision_at(sub["score_compression"], sub[target], 0.10)
    c_lift = c_p10 / base if base > 0 else float("nan")
    c_pf, c_wr, c_n = _pf_top30(sub, "score_compression", target, mfe, mae)
    # avg mfe/mae
    avg_mfe = float(sub[mfe].mean()) if mfe in sub.columns else float("nan")
    avg_mae = float(sub[mae].mean()) if mae in sub.columns else float("nan")
    return {
        "slice_name": name,
        "slice_value": value,
        "n": int(len(sub)),
        "pos_rate": base,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "model_rho": m_rho,
        "model_p10": m_p10,
        "model_lift10": m_lift,
        "model_pf_top30": m_pf,
        "comp_rho": c_rho,
        "comp_p10": c_p10,
        "comp_lift10": c_lift,
        "comp_pf_top30": c_pf,
        "lift_delta": (m_lift - c_lift) if np.isfinite(m_lift) and np.isfinite(c_lift) else float("nan"),
        "rho_delta": (m_rho - c_rho) if np.isfinite(m_rho) and np.isfinite(c_rho) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="output/cte_hb_preds_v1.parquet")
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--target", default="runner_15")
    ap.add_argument("--mfe", default="mfe_15_atr")
    ap.add_argument("--mae", default="mae_15_atr")
    ap.add_argument("--out-csv", default="output/cte_hb_diag_slices.csv")
    ap.add_argument("--out-md", default=str(
        Path.home() / ".claude/projects/-Users-alpkarakasli-Projects-nox-project/memory/cte_hb_diag_slices.md"))
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    p = pd.read_parquet(args.preds)
    p["date"] = pd.to_datetime(p["date"])
    ds = pd.read_parquet(args.dataset)
    ds["date"] = pd.to_datetime(ds["date"])

    join_cols = ["ticker", "date"] + [c for c in (
        list(SLICE_FEATURES_TERTILE) + list(SLICE_FEATURES_BINARY)
    ) if c in ds.columns and c not in p.columns]
    ds_join = ds[join_cols].drop_duplicates(["ticker", "date"])
    enriched = p.merge(ds_join, on=["ticker", "date"], how="left")
    missing = enriched[join_cols[2:]].isna().sum()
    print(f"[JOIN] preds shape={p.shape} → enriched shape={enriched.shape}; "
          f"missing per feature: {missing.to_dict()}")

    rows: list[dict] = []

    # Tertile slices on numeric features
    for feat in SLICE_FEATURES_TERTILE:
        if feat not in enriched.columns:
            continue
        valid = enriched[enriched[feat].notna()]
        if len(valid) < 30:
            continue
        try:
            valid = valid.copy()
            valid["__bucket"] = pd.qcut(valid[feat], q=3,
                                         labels=["q1_low", "q2_mid", "q3_high"],
                                         duplicates="drop")
        except Exception:
            continue
        for bkt, sub in valid.groupby("__bucket", observed=False):
            r = _slice_row(sub, feat, str(bkt), args.target, args.mfe, args.mae)
            if r:
                rows.append(r)

    # Binary slices
    for feat in SLICE_FEATURES_BINARY:
        if feat not in enriched.columns:
            continue
        valid = enriched[enriched[feat].notna()]
        for val, sub in valid.groupby(feat):
            r = _slice_row(sub, feat, str(int(val)), args.target, args.mfe, args.mae)
            if r:
                rows.append(r)

    # fold × setup_type
    for (fold, st), sub in enriched.groupby(["fold_assigned", "setup_type"]):
        r = _slice_row(sub, "fold_x_setup", f"{fold}/{st}", args.target, args.mfe, args.mae)
        if r:
            rows.append(r)

    # Year (and quarter) slices
    enriched["year"] = enriched["date"].dt.year.astype(int)
    enriched["yq"] = enriched["date"].dt.year.astype(str) + "Q" + enriched["date"].dt.quarter.astype(str)
    for yr, sub in enriched.groupby("year"):
        r = _slice_row(sub, "year", str(int(yr)), args.target, args.mfe, args.mae)
        if r:
            rows.append(r)
    for yq, sub in enriched.groupby("yq"):
        r = _slice_row(sub, "year_quarter", str(yq), args.target, args.mfe, args.mae)
        if r:
            rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ No slice rows produced (all slices too small)")
        return 0

    out = out.sort_values(["slice_name", "slice_value"]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}  rows={len(out)}")

    # Pretty print per slice family
    for feat in sorted(out["slice_name"].unique()):
        sub = out[out["slice_name"] == feat]
        print(f"\n── {feat} ──")
        cols = ["slice_value", "n", "pos_rate", "model_lift10", "comp_lift10",
                "lift_delta", "model_rho", "comp_rho", "model_pf_top30", "comp_pf_top30"]
        d = sub[cols].copy()
        for c in ("pos_rate",):
            d[c] = d[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        for c in ("model_rho", "comp_rho", "rho_delta"):
            if c in d.columns:
                d[c] = d[c].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
        for c in ("model_lift10", "comp_lift10", "lift_delta",
                  "model_pf_top30", "comp_pf_top30"):
            if c in d.columns:
                d[c] = d[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        print(d.to_string(index=False))

    # ── Markdown report ──
    md = []
    md.append("---")
    md.append("name: HB error-slice diagnostic")
    md.append("description: HB production preds × feature tertile/binary/time slices. Model vs compression lift@10, rho, PF_top30.")
    md.append("type: project")
    md.append("---")
    md.append(f"Kaynak: `{args.preds}` enriched from `{args.dataset}`. No retraining.")
    md.append("")
    md.append("## Slice results")
    md.append("")
    md.append("| slice | value | N | pos | model_lift@10 | comp_lift@10 | Δlift | model_rho | comp_rho | model_PF_t30 | comp_PF_t30 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in out.iterrows():
        md.append(
            f"| {r['slice_name']} | {r['slice_value']} | {int(r['n'])} | "
            f"{r['pos_rate']:.3f} | "
            f"{r['model_lift10']:.2f} | {r['comp_lift10']:.2f} | "
            f"{r['lift_delta']:+.2f} | "
            f"{r['model_rho']:+.3f} | {r['comp_rho']:+.3f} | "
            f"{r['model_pf_top30']:.2f} | {r['comp_pf_top30']:.2f} |"
        )
    md.append("")
    # Sorted highlights
    model_wins = out[out["lift_delta"] > 0.5].sort_values("lift_delta", ascending=False).head(8)
    comp_wins = out[out["lift_delta"] < -0.5].sort_values("lift_delta").head(8)
    md.append("## Slices where model dominates compression (Δlift > 0.5)")
    for _, r in model_wins.iterrows():
        md.append(f"- **{r['slice_name']}={r['slice_value']}** N={int(r['n'])}: "
                  f"model {r['model_lift10']:.2f}x vs comp {r['comp_lift10']:.2f}x "
                  f"(Δ {r['lift_delta']:+.2f}), model rho {r['model_rho']:+.3f}")
    md.append("")
    md.append("## Slices where compression dominates model (Δlift < -0.5)")
    for _, r in comp_wins.iterrows():
        md.append(f"- **{r['slice_name']}={r['slice_value']}** N={int(r['n'])}: "
                  f"comp {r['comp_lift10']:.2f}x vs model {r['model_lift10']:.2f}x "
                  f"(Δ {r['lift_delta']:+.2f}), comp rho {r['comp_rho']:+.3f}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print(f"\n[WRITE] {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
