"""
CTE HB baseline diagnostic — audit on existing production artifact.

Reads output/cte_hb_preds_v1.parquet (seed=17 baseline, runner_15, mixed) and
produces a structural readout that makes the "1.14x lift@10" figure
comparable to its true behaviour per fold, per decile, and relative to the
compression heuristic. No retraining, no model changes.

Questions answered:
  1. Where does the aggregate 1.14x come from — is it carried by one fold?
  2. Is score_model monotone across score_model deciles, or noisy bumps?
  3. How often does score_model agree with score_compression at the top?
  4. What does score_compression alone deliver? (If HB model ≈ compression,
     ML adds nothing.)

Outputs
-------
  output/cte_hb_diag_baseline.csv
  memory/cte_hb_diag_baseline.md
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


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return float("nan")
    return float(a[mask].rank().corr(b[mask].rank()))


def _precision_at(score: pd.Series, target: pd.Series, k_frac: float) -> tuple[int, float]:
    mask = score.notna() & target.notna()
    s, y = score[mask], target[mask]
    if len(s) == 0:
        return 0, float("nan")
    k = max(1, int(round(len(s) * k_frac)))
    top = s.nlargest(k).index
    return k, float(y.loc[top].mean())


def _pf_ratio(mfe: pd.Series, mae: pd.Series, target: pd.Series) -> dict:
    """Proxy PF: gain capped at 3R for winners, loss capped at 1.5R for losers."""
    mask = mfe.notna() & mae.notna() & target.notna()
    if mask.sum() == 0:
        return {"n": 0, "pf": float("nan"), "wr": float("nan"), "avgR": float("nan"),
                "sum_gain": float("nan"), "sum_loss": float("nan")}
    y = target[mask].astype(int).values
    gain = np.minimum(mfe[mask].values, 3.0)
    loss = -np.minimum(mae[mask].values, 1.5)
    realised = np.where(y == 1, gain, loss)
    pos = realised[realised > 0].sum()
    neg = -realised[realised < 0].sum()
    return {
        "n": int(mask.sum()),
        "pf": float(pos / neg) if neg > 0 else float("inf"),
        "wr": float((realised > 0).mean()),
        "avgR": float(realised.mean()),
        "sum_gain": float(pos),
        "sum_loss": float(neg),
    }


def _metrics_for_score(df: pd.DataFrame, score_col: str, target: str,
                        mfe: str, mae: str) -> dict:
    base = float(df[target].dropna().mean()) if df[target].notna().any() else float("nan")
    rho = _spearman(df[score_col], df[target])
    _, p10 = _precision_at(df[score_col], df[target], 0.10)
    _, p20 = _precision_at(df[score_col], df[target], 0.20)
    _, p30 = _precision_at(df[score_col], df[target], 0.30)
    # PF_top30
    valid = df[df[score_col].notna() & df[target].notna()]
    if len(valid) >= 10:
        cut = valid[score_col].quantile(0.70)
        top30 = valid[valid[score_col] >= cut]
    else:
        top30 = valid
    pf_top30 = _pf_ratio(top30[mfe], top30[mae], top30[target])
    pf_overall = _pf_ratio(df[mfe], df[mae], df[target])
    return {
        "n": int(df[target].notna().sum()),
        "base": base,
        "rho": rho,
        "p@10": p10, "p@20": p20, "p@30": p30,
        "lift@10": p10 / base if base and base > 0 else float("nan"),
        "lift@20": p20 / base if base and base > 0 else float("nan"),
        "lift@30": p30 / base if base and base > 0 else float("nan"),
        "pf_overall": pf_overall.get("pf", float("nan")),
        "wr_overall": pf_overall.get("wr", float("nan")),
        "pf_top30": pf_top30.get("pf", float("nan")),
        "wr_top30": pf_top30.get("wr", float("nan")),
        "pf_top30_n": pf_top30.get("n", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-10 overlap between two scores
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_idx(df: pd.DataFrame, col: str, k_frac: float) -> set:
    mask = df[col].notna()
    sub = df.loc[mask]
    if sub.empty:
        return set()
    k = max(1, int(round(len(sub) * k_frac)))
    return set(sub[col].nlargest(k).index)


def _overlap(df: pd.DataFrame, a: str, b: str, k_frac: float = 0.10) -> dict:
    ia, ib = _top_k_idx(df, a, k_frac), _top_k_idx(df, b, k_frac)
    if not ia or not ib:
        return {"a_top": len(ia), "b_top": len(ib), "intersect": 0,
                "pct_of_a": float("nan"), "pct_of_b": float("nan")}
    inter = ia & ib
    return {
        "a_top": len(ia), "b_top": len(ib), "intersect": len(inter),
        "pct_of_a": len(inter) / len(ia) if ia else float("nan"),
        "pct_of_b": len(inter) / len(ib) if ib else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decile table
# ─────────────────────────────────────────────────────────────────────────────

def _decile_table(df: pd.DataFrame, score: str, target: str,
                   mfe: str, mae: str) -> pd.DataFrame:
    mask = df[score].notna() & df[target].notna()
    sub = df.loc[mask, [score, target, mfe, mae]].copy()
    if len(sub) < 20:
        return pd.DataFrame()
    try:
        sub["q"] = pd.qcut(sub[score], q=10, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for q, g in sub.groupby("q"):
        y = g[target].astype(int).values
        gain = np.minimum(g[mfe].values, 3.0)
        loss = -np.minimum(g[mae].values, 1.5)
        realised = np.where(y == 1, gain, loss)
        pos_sum = realised[realised > 0].sum()
        neg_sum = -realised[realised < 0].sum()
        rows.append({
            "q": int(q),
            "n": int(len(g)),
            "pos_rate": float(y.mean()),
            "avg_mfe": float(g[mfe].mean()),
            "avg_mae": float(g[mae].mean()),
            "sum_gain": float(pos_sum),
            "sum_loss": float(neg_sum),
            "pf": float(pos_sum / neg_sum) if neg_sum > 0 else float("inf"),
            "avgR": float(realised.mean()),
        })
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="output/cte_hb_preds_v1.parquet")
    ap.add_argument("--target", default="runner_15")
    ap.add_argument("--mfe", default="mfe_15_atr")
    ap.add_argument("--mae", default="mae_15_atr")
    ap.add_argument("--out-csv", default="output/cte_hb_diag_baseline.csv")
    ap.add_argument("--out-md", default=str(
        Path.home() / ".claude/projects/-Users-alpkarakasli-Projects-nox-project/memory/cte_hb_diag_baseline.md"))
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    p = pd.read_parquet(args.preds)
    p["date"] = pd.to_datetime(p["date"])
    print(f"[DATA] {args.preds}  shape={p.shape}")

    rows: list[dict] = []

    # Overall: score_model + score_compression + score_random
    for sc in ("score_model", "score_compression", "score_random"):
        if sc not in p.columns:
            continue
        m = _metrics_for_score(p, sc, args.target, args.mfe, args.mae)
        m["scope"] = "overall"
        m["slice"] = "ALL"
        m["score"] = sc
        rows.append(m)

    # Per-fold
    for fold, fg in p.groupby("fold_assigned"):
        for sc in ("score_model", "score_compression", "score_random"):
            if sc not in fg.columns:
                continue
            m = _metrics_for_score(fg, sc, args.target, args.mfe, args.mae)
            m["scope"] = "fold"
            m["slice"] = str(fold)
            m["score"] = sc
            rows.append(m)

    # Per setup_type
    if "setup_type" in p.columns:
        for st, sg in p.groupby("setup_type"):
            if len(sg) < 10:
                continue
            for sc in ("score_model", "score_compression", "score_random"):
                m = _metrics_for_score(sg, sc, args.target, args.mfe, args.mae)
                m["scope"] = "setup_type"
                m["slice"] = str(st)
                m["score"] = sc
                rows.append(m)

    summary_df = pd.DataFrame(rows)
    col_order = ["scope", "slice", "score", "n", "base", "rho",
                 "p@10", "p@20", "p@30",
                 "lift@10", "lift@20", "lift@30",
                 "pf_overall", "wr_overall",
                 "pf_top30", "wr_top30", "pf_top30_n"]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}  rows={len(summary_df)}")

    # Decile table for model + compression
    dec_model = _decile_table(p, "score_model", args.target, args.mfe, args.mae)
    dec_comp = _decile_table(p, "score_compression", args.target, args.mfe, args.mae)

    # Overlap top-10% (and 20%) model vs compression
    overlap_overall_10 = _overlap(p, "score_model", "score_compression", 0.10)
    overlap_overall_20 = _overlap(p, "score_model", "score_compression", 0.20)
    overlap_by_fold = {}
    for fold, fg in p.groupby("fold_assigned"):
        overlap_by_fold[str(fold)] = {
            "top10": _overlap(fg, "score_model", "score_compression", 0.10),
            "top20": _overlap(fg, "score_model", "score_compression", 0.20),
        }

    # ── Pretty print ──
    def _fmt_row(r: dict) -> str:
        return (f"  {r['scope']:<10} {r['slice']:<8} {r['score']:<18} "
                f"N={r['n']:>3} base={r.get('base', float('nan')):.3f} "
                f"rho={r['rho']:+.3f} "
                f"p@10={r['p@10']:.3f} lift@10={r['lift@10']:.2f}x  "
                f"p@20={r['p@20']:.3f} lift@20={r['lift@20']:.2f}x  "
                f"PF_o={r['pf_overall']:.2f} PF_t30={r['pf_top30']:.2f}")

    print("\n═══ OVERALL ═══")
    for r in rows:
        if r["scope"] == "overall":
            print(_fmt_row(r))
    print("\n═══ PER FOLD ═══")
    for r in rows:
        if r["scope"] == "fold":
            print(_fmt_row(r))
    print("\n═══ PER SETUP_TYPE ═══")
    for r in rows:
        if r["scope"] == "setup_type":
            print(_fmt_row(r))

    print("\n═══ score_model DECILES ═══")
    if not dec_model.empty:
        print(dec_model.to_string(index=False))
    print("\n═══ score_compression DECILES ═══")
    if not dec_comp.empty:
        print(dec_comp.to_string(index=False))

    print("\n═══ TOP-10% OVERLAP (model vs compression) ═══")
    print(f"  overall: a_top={overlap_overall_10['a_top']} b_top={overlap_overall_10['b_top']} "
          f"inter={overlap_overall_10['intersect']} "
          f"pct_model={overlap_overall_10['pct_of_a']:.0%} "
          f"pct_comp={overlap_overall_10['pct_of_b']:.0%}")
    for fold, ov in overlap_by_fold.items():
        print(f"  {fold}  top10: inter={ov['top10']['intersect']}/"
              f"{ov['top10']['a_top']} ({ov['top10']['pct_of_a']:.0%})"
              f"   top20: inter={ov['top20']['intersect']}/"
              f"{ov['top20']['a_top']} ({ov['top20']['pct_of_a']:.0%})")

    # ── Markdown report ──
    md = []
    md.append("---")
    md.append("name: HB baseline diagnostic")
    md.append("description: cte_hb_v3 oturumu baseline audit — production artifact üstünde metric breakdown, decile, overlap. Hiç retrain yok, hiç model değişikliği yok.")
    md.append("type: project")
    md.append("---")
    md.append(f"Kaynak: `{args.preds}` (seed=17, runner_15, mcs=20, mixed). Üretim değişikliği YAPILMADI.")
    md.append("")
    md.append("## 1. Overall")
    for r in rows:
        if r["scope"] == "overall":
            md.append(f"- `{r['score']}` N={r['n']} base={r['base']:.3f} "
                      f"rho={r['rho']:+.3f} lift@10={r['lift@10']:.2f}x "
                      f"lift@20={r['lift@20']:.2f}x lift@30={r['lift@30']:.2f}x "
                      f"PF_overall={r['pf_overall']:.2f} PF_top30={r['pf_top30']:.2f}")
    md.append("")
    md.append("## 2. Per-fold")
    md.append("| fold | score | N | base | rho | lift@10 | lift@20 | PF_top30 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["scope"] == "fold":
            md.append(f"| {r['slice']} | {r['score']} | {r['n']} | {r['base']:.3f} | "
                      f"{r['rho']:+.3f} | {r['lift@10']:.2f}x | "
                      f"{r['lift@20']:.2f}x | {r['pf_top30']:.2f} |")
    md.append("")
    md.append("## 3. Per setup_type")
    md.append("| setup | score | N | base | rho | lift@10 | PF_top30 |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["scope"] == "setup_type":
            md.append(f"| {r['slice']} | {r['score']} | {r['n']} | {r['base']:.3f} | "
                      f"{r['rho']:+.3f} | {r['lift@10']:.2f}x | {r['pf_top30']:.2f} |")
    md.append("")
    md.append("## 4. score_model decile table")
    if not dec_model.empty:
        md.append("| q | n | pos_rate | avg_mfe | avg_mae | sum_gain | sum_loss | pf | avgR |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in dec_model.iterrows():
            md.append(f"| {int(row['q'])} | {int(row['n'])} | {row['pos_rate']:.3f} | "
                      f"{row['avg_mfe']:+.3f} | {row['avg_mae']:+.3f} | "
                      f"{row['sum_gain']:.2f} | {row['sum_loss']:.2f} | "
                      f"{row['pf']:.2f} | {row['avgR']:+.2f} |")
    md.append("")
    md.append("## 5. score_compression decile table")
    if not dec_comp.empty:
        md.append("| q | n | pos_rate | avg_mfe | avg_mae | pf | avgR |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in dec_comp.iterrows():
            md.append(f"| {int(row['q'])} | {int(row['n'])} | {row['pos_rate']:.3f} | "
                      f"{row['avg_mfe']:+.3f} | {row['avg_mae']:+.3f} | "
                      f"{row['pf']:.2f} | {row['avgR']:+.2f} |")
    md.append("")
    md.append("## 6. Model vs Compression top-10/20% overlap")
    md.append(f"- overall top-10: intersect={overlap_overall_10['intersect']} / "
              f"model_top={overlap_overall_10['a_top']} ({overlap_overall_10['pct_of_a']:.0%}) "
              f"/ comp_top={overlap_overall_10['b_top']} ({overlap_overall_10['pct_of_b']:.0%})")
    md.append(f"- overall top-20: intersect={overlap_overall_20['intersect']} / "
              f"model_top={overlap_overall_20['a_top']} ({overlap_overall_20['pct_of_a']:.0%})")
    for fold, ov in overlap_by_fold.items():
        md.append(f"- {fold} top-10: inter={ov['top10']['intersect']}/"
                  f"{ov['top10']['a_top']} ({ov['top10']['pct_of_a']:.0%}) | "
                  f"top-20: inter={ov['top20']['intersect']}/"
                  f"{ov['top20']['a_top']} ({ov['top20']['pct_of_a']:.0%})")
    md.append("")

    out_md_path = Path(args.out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md))
    print(f"\n[WRITE] {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
