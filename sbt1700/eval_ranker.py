"""SBT-1700 RESET — phase-aware ranker evaluation.

This module computes both the raw cohort metrics (PF, WR, avg_R, etc.)
and the ranked-output metrics (rho, top-decile bucket) given a trained
LightGBM regression model and a labeled DataFrame.

It does *not* load splits itself; the reset orchestrator routes splits
through `sbt1700.splits.load_split` so the test-period lock has a single
chokepoint. Pass already-filtered DataFrames in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import lightgbm as lgb


TOP_DECILE_FRAC: float = 0.10


@dataclass
class CohortMetrics:
    n: int
    wr: float
    pf: float
    avg_R: float
    median_R: float
    total_R: float
    tp_pct: float
    sl_pct: float
    timeout_pct: float
    partial_pct: float
    avg_bars_held: float
    top5_R_share: float
    bot5_R_share: float


def cohort_metrics(df: pd.DataFrame, label_col: str = "realized_R_net") -> CohortMetrics:
    sub = df.dropna(subset=[label_col]).copy()
    n = len(sub)
    if n == 0:
        nan = float("nan")
        return CohortMetrics(0, nan, nan, nan, nan, 0.0,
                             nan, nan, nan, nan, nan, nan, nan)
    R = sub[label_col].astype(float)
    wins = R[R > 0].sum()
    losses = -R[R < 0].sum()
    pf = float("inf") if (losses == 0 and wins > 0) else (
        float(wins / losses) if losses > 0 else float("nan"))
    total = float(R.sum())
    R_sorted_desc = R.sort_values(ascending=False)
    R_sorted_asc = R.sort_values(ascending=True)
    top5 = R_sorted_desc.head(5).sum()
    bot5 = R_sorted_asc.head(5).sum()

    def _pct(col: str) -> float:
        if col not in sub.columns:
            return float("nan")
        return float(sub[col].astype(bool).mean())

    bars_held = (sub["bars_held"].astype(float).mean()
                 if "bars_held" in sub.columns else float("nan"))
    return CohortMetrics(
        n=n,
        wr=float((R > 0).mean()),
        pf=pf,
        avg_R=float(R.mean()),
        median_R=float(R.median()),
        total_R=total,
        tp_pct=_pct("tp_hit"),
        sl_pct=_pct("sl_hit"),
        timeout_pct=_pct("timeout_hit"),
        partial_pct=_pct("partial_hit"),
        avg_bars_held=bars_held,
        top5_R_share=float(top5 / total) if total != 0 else float("nan"),
        bot5_R_share=float(bot5 / total) if total != 0 else float("nan"),
    )


def by_year_summary(df: pd.DataFrame, label_col: str = "realized_R_net") -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[label_col]).copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["year"] = sub["date"].dt.year
    rows: list[dict] = []
    for y, g in sub.groupby("year"):
        m = cohort_metrics(g, label_col=label_col)
        rows.append({"year": int(y), "N": m.n, "WR": m.wr, "PF": m.pf,
                     "avg_R": m.avg_R, "median_R": m.median_R, "total_R": m.total_R})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def concentration_summary(df: pd.DataFrame, label_col: str = "realized_R_net") -> pd.DataFrame:
    if "ticker" not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[label_col]).copy()
    if sub.empty:
        return pd.DataFrame()
    g = (sub.groupby("ticker")[label_col]
         .agg(["count", "sum", "mean"])
         .rename(columns={"count": "N", "sum": "total_R", "mean": "avg_R"})
         .sort_values("total_R", ascending=False))
    return g.reset_index().head(10)


def rank_metrics(
    df: pd.DataFrame,
    score: pd.Series,
    label_col: str = "realized_R_net",
) -> dict:
    """Spearman rho + top-decile bucket on the predicted score."""
    sub = df.dropna(subset=[label_col]).copy()
    s = score.loc[sub.index] if isinstance(score.index, pd.RangeIndex) else score.iloc[:len(sub)]
    if len(sub) == 0:
        return dict(spearman_rho=float("nan"), top_decile_n=0,
                    top_decile_avg_R=float("nan"),
                    top_decile_PF=float("nan"),
                    top_decile_WR=float("nan"))
    R = sub[label_col].astype(float).reset_index(drop=True)
    s = pd.Series(np.asarray(s)).reset_index(drop=True).iloc[:len(R)]
    rho = float(R.rank().corr(s.rank())) if len(R) >= 5 else float("nan")
    n = len(R)
    top_n = max(1, int(round(TOP_DECILE_FRAC * n)))
    top_idx = s.sort_values(ascending=False).head(top_n).index
    top_R = R.iloc[top_idx]
    wins = top_R[top_R > 0].sum()
    losses = -top_R[top_R < 0].sum()
    pf = float("inf") if (losses == 0 and wins > 0) else (
        float(wins / losses) if losses > 0 else float("nan"))
    return dict(
        spearman_rho=rho,
        top_decile_n=int(top_n),
        top_decile_avg_R=float(top_R.mean()),
        top_decile_PF=pf,
        top_decile_WR=float((top_R > 0).mean()),
    )


def predict_with_model(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(model.predict(df[feature_cols]), index=df.index)


def evaluate(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "realized_R_net",
) -> dict:
    """Headline summary of a single phase-split eval."""
    sub = df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    score = predict_with_model(model, sub, feature_cols)
    cohort = cohort_metrics(sub, label_col=label_col)
    rmet = rank_metrics(sub, score, label_col=label_col) if len(sub) else {}
    return {
        "cohort": cohort.__dict__,
        "ranker": rmet,
        "n": len(sub),
        "score_min": float(score.min()) if len(score) else float("nan"),
        "score_max": float(score.max()) if len(score) else float("nan"),
    }


def save_trade_list(
    df: pd.DataFrame,
    score: Optional[pd.Series],
    out_path: str | Path,
    label_col: str = "realized_R_net",
) -> None:
    cols = [c for c in ["ticker", "date", "exit_variant", "entry_px", "stop_px",
                        "tp_px", "exit_px", "exit_date", "exit_reason",
                        "bars_held", label_col, "realized_R_gross",
                        "tp_hit", "sl_hit", "timeout_hit", "partial_hit"]
            if c in df.columns]
    out = df[cols].copy()
    if score is not None and len(score) == len(out):
        out["model_score"] = score.values
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
