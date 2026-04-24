"""
CTE eval — trading-metric report from output/cte_preds_v1.parquet.

Metrikler:
  - Base rate (pos rate) vs precision at top-K% by score
  - Spearman rho(score, target)
  - Per-fold + per-setup_type breakdown
  - Profit factor / win rate / MFE / MAE on close-entry assumption
    (exit at t+h, or earlier if failed_break hits)
  - Bucket table: score decile → runner rate

Kullanım:
  python -m cte.tools.eval
  python -m cte.tools.eval --preds output/cte_preds_v1.parquet --target runner_15
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


def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return np.nan
    return float(a[mask].rank().corr(b[mask].rank()))


def _precision_at_topk(score: pd.Series, target: pd.Series, k_frac: float) -> tuple[int, float]:
    mask = score.notna() & target.notna()
    s = score[mask]
    y = target[mask]
    if len(s) == 0:
        return 0, np.nan
    k = max(1, int(round(len(s) * k_frac)))
    top = s.nlargest(k).index
    p = y.loc[top].mean()
    return k, float(p)


def _decile_table(score: pd.Series, target: pd.Series) -> pd.DataFrame:
    mask = score.notna() & target.notna()
    if mask.sum() < 20:
        return pd.DataFrame()
    s = score[mask]
    y = target[mask]
    qs = pd.qcut(s, q=10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"q": qs, "y": y}).groupby("q").agg(
        n=("y", "size"),
        pos=("y", "sum"),
        rate=("y", "mean"),
    )
    return tab


def _report_scores(
    df: pd.DataFrame,
    target: str,
    scores: list[str],
    title: str,
) -> None:
    print(f"\n{'─' * 68}\n{title}\n{'─' * 68}")
    n_total = df[target].notna().sum()
    base = df[target].dropna().mean()
    print(f"  N={n_total}  base_rate={base:.2%}")

    rows = []
    for sc in scores:
        if sc not in df.columns:
            continue
        rho = _spearman(df[sc], df[target])
        _, p10 = _precision_at_topk(df[sc], df[target], 0.10)
        _, p20 = _precision_at_topk(df[sc], df[target], 0.20)
        _, p30 = _precision_at_topk(df[sc], df[target], 0.30)
        lift10 = (p10 / base) if base > 0 else np.nan
        rows.append(
            {
                "score": sc, "rho": rho,
                "p@10%": p10, "p@20%": p20, "p@30%": p30,
                "lift@10%": lift10,
            }
        )
    rep = pd.DataFrame(rows)
    if not rep.empty:
        print(
            rep.to_string(
                index=False,
                formatters={
                    "rho": "{:+.3f}".format,
                    "p@10%": "{:.1%}".format,
                    "p@20%": "{:.1%}".format,
                    "p@30%": "{:.1%}".format,
                    "lift@10%": "{:.2f}x".format,
                },
            )
        )


def _profit_factor_close_entry(df: pd.DataFrame, target: str, mfe: str, mae: str) -> dict:
    """Simplified trade metrics: entry=close, exit after max runner horizon.

    TP anlamında yaklaşık: runner_{h}=1 → sürede ~runner_mfe_atr MFE kaydedildi
    ama gerçek kapanış R MFE değil; burada MFE/MAE'den geçici realised Karlaş
    tahmini:
      realised_r = min(mfe, 3.0) - (mae if failed_break else 0)
    Basit ama v1 için yeterli. Eval'in gerçek versiyonu sonra.
    """
    mask = df[target].notna() & df[mfe].notna() & df[mae].notna()
    sub = df.loc[mask].copy()
    if sub.empty:
        return {"n": 0}

    # Simple proxy: gain = min(mfe, 3.0), loss = -min(mae, 1.5)
    gain = np.minimum(sub[mfe].values, 3.0)
    loss = -np.minimum(sub[mae].values, 1.5)
    # Runner=1 → realised = gain; else realised = loss (pulled back)
    realised = np.where(sub[target].values == 1, gain, loss)
    pos_sum = realised[realised > 0].sum()
    neg_sum = -realised[realised < 0].sum()
    pf = pos_sum / neg_sum if neg_sum > 0 else np.inf
    wr = float((realised > 0).mean())
    return {
        "n": int(len(sub)),
        "avg_R": float(realised.mean()),
        "PF_proxy": float(pf),
        "WR_proxy": wr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="output/cte_preds_v1.parquet")
    ap.add_argument("--target", default="runner_15")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.preds).exists():
        print(f"❌ Preds yok: {args.preds}")
        return 2

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    scores = [c for c in ["score_model", "score_compression", "score_random"] if c in df.columns]

    print(f"═══ CTE Eval ═══  target={args.target}")
    print(f"Preds: {args.preds}  shape={df.shape}")

    # Overall
    _report_scores(df, args.target, scores, "OVERALL")

    # Per fold
    for fold in df["fold_assigned"].dropna().unique():
        sub = df[df["fold_assigned"] == fold]
        _report_scores(sub, args.target, scores, f"FOLD {fold}")

    # Per setup_type
    for st in ["hb", "fc", "both"]:
        sub = df[df["setup_type"] == st]
        if len(sub) < 20:
            continue
        _report_scores(sub, args.target, scores, f"SETUP={st}")

    # Decile of score_model
    if "score_model" in df.columns:
        tab = _decile_table(df["score_model"], df[args.target])
        if not tab.empty:
            print("\n─────────── SCORE_MODEL DECILES ───────────")
            print(tab.to_string(formatters={"rate": "{:.1%}".format}))

    # Profit factor proxy (runner_15 horizon)
    h = args.target.split("_")[-1]
    mfe = f"mfe_{h}_atr"
    mae = f"mae_{h}_atr"
    if mfe in df.columns and mae in df.columns:
        print(f"\n─────────── PROFIT-FACTOR PROXY (overall) ───────────")
        print(_profit_factor_close_entry(df, args.target, mfe, mae))
        # top 30% by model
        if "score_model" in df.columns:
            s = df["score_model"]
            top = df.loc[s >= s.quantile(0.70)]
            print("  top 30% by score_model:")
            print("  ", _profit_factor_close_entry(top, args.target, mfe, mae))

    return 0


if __name__ == "__main__":
    sys.exit(main())
