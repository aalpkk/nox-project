"""Run all exit variants on the SBT-1700 dataset and compute raw cohort
diagnostics per variant. No ranker, no filtering — pure execution-rule
sensitivity on the same entry signal.

Outputs:
    output/sbt_1700_exit_matrix_trades.parquet  — per (ticker, date, variant) row
    output/sbt_1700_exit_matrix.csv             — variant-level summary
    output/sbt_1700_exit_matrix.md              — human-readable report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbt1700.exits import EXIT_VARIANTS, simulate_exit, variant_names


def _load_master(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    if df.index.name != "Date":
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"}).set_index("Date")
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    return {tk: g[["Open", "High", "Low", "Close"]].sort_index()
            for tk, g in df.groupby("ticker")}


def _profit_factor(rs: pd.Series) -> float:
    rs = rs.dropna()
    wins = rs[rs > 0].sum()
    losses = -rs[rs < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _cohort_summary(rows: pd.DataFrame, variant: str) -> dict:
    valid = rows.dropna(subset=["realized_R_net"])
    n = len(valid)
    if n == 0:
        return {"exit_variant": variant, "N": 0}

    R = valid["realized_R_net"]
    by_year = (
        valid.assign(year=valid["date"].dt.year)
             .groupby("year")["realized_R_net"]
             .agg(["count", "mean", "median", _profit_factor,
                   lambda x: float((x > 0).mean())])
             .rename(columns={"count": "n", "mean": "avg_R", "median": "med_R",
                              "_profit_factor": "PF", "<lambda_0>": "WR"})
             .to_dict(orient="index")
    )
    # ATR-normalized vol bucket: terciles of atr14_prior / close_1700.
    if "atr14_prior" in valid.columns and "close_1700" in valid.columns:
        rel_atr = valid["atr14_prior"] / valid["close_1700"]
        try:
            buckets = pd.qcut(rel_atr, 3, labels=["low_vol", "mid_vol", "high_vol"])
            vol_summary = (
                valid.assign(bucket=buckets)
                     .groupby("bucket", observed=True)["realized_R_net"]
                     .agg(["count", "mean", _profit_factor,
                           lambda x: float((x > 0).mean())])
                     .rename(columns={"count": "n", "mean": "avg_R",
                                      "_profit_factor": "PF",
                                      "<lambda_0>": "WR"})
                     .to_dict(orient="index")
            )
        except ValueError:
            vol_summary = {}
    else:
        vol_summary = {}

    top_tickers = valid["ticker"].value_counts().head(5).to_dict()
    top5_R = valid["realized_R_net"].nlargest(5).sum()
    bot5_R = valid["realized_R_net"].nsmallest(5).sum()
    total_R = valid["realized_R_net"].sum()

    return {
        "exit_variant": variant,
        "N": int(n),
        "WR_net": float((R > 0).mean()),
        "PF_net": _profit_factor(R),
        "avg_R_net": float(R.mean()),
        "median_R_net": float(R.median()),
        "tp_pct": float(valid["tp_hit"].mean()),
        "sl_pct": float(valid["sl_hit"].mean()),
        "timeout_pct": float(valid["timeout_hit"].mean()),
        "partial_pct": float(valid.get("partial_hit", pd.Series([False]*n)).mean()),
        "avg_bars_held": float(valid["bars_held"].mean()),
        "total_R_net": float(total_R),
        "top5_R_share": float(top5_R / total_R) if total_R != 0 else float("nan"),
        "bot5_R_share": float(bot5_R / total_R) if total_R != 0 else float("nan"),
        "by_year": by_year,
        "vol_buckets": vol_summary,
        "top_tickers": top_tickers,
    }


def run_exit_matrix(
    dataset: pd.DataFrame,
    daily_master: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (trades_df, summaries) where trades_df is a long table
    ticker × date × exit_variant and summaries is a list of per-variant
    dicts suitable for CSV/markdown rendering.
    """
    trades: list[dict] = []
    for r in dataset.itertuples(index=False):
        tk = r.ticker
        T = pd.Timestamp(r.date).normalize()
        entry_px = float(r.close_1700)
        atr_1700 = float(r.atr14_prior)
        sub = daily_master.get(tk)
        if sub is None:
            continue
        for v in variant_names():
            res = simulate_exit(v, T, entry_px, atr_1700, sub)
            res["ticker"] = tk
            res["date"] = T
            res["atr14_prior"] = atr_1700
            res["close_1700"] = entry_px
            trades.append(res)

    trades_df = pd.DataFrame(trades)
    summaries = [_cohort_summary(trades_df[trades_df["exit_variant"] == v], v)
                 for v in variant_names()]
    return trades_df, summaries


def to_summary_csv(summaries: list[dict]) -> pd.DataFrame:
    flat = []
    for s in summaries:
        if s.get("N", 0) == 0:
            flat.append({"exit_variant": s["exit_variant"], "N": 0})
            continue
        flat.append({k: s[k] for k in [
            "exit_variant", "N", "WR_net", "PF_net", "avg_R_net", "median_R_net",
            "tp_pct", "sl_pct", "timeout_pct", "partial_pct",
            "avg_bars_held", "total_R_net", "top5_R_share", "bot5_R_share",
        ]})
    return pd.DataFrame(flat)


def to_markdown(summaries: list[dict]) -> str:
    lines = ["# SBT-1700 — Exit Matrix (raw cohort)",
             "",
             "_All variants applied to the same entry signal cohort. No ranker, no filter._",
             ""]
    # Headline table.
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| variant | N | WR_net | PF_net | avg_R | med_R | tp% | sl% | to% | partial% | avg_bars | total_R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        if s.get("N", 0) == 0:
            lines.append(f"| {s['exit_variant']} | 0 | – | – | – | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {s['exit_variant']} | {s['N']} | {s['WR_net']:.3f} | {s['PF_net']:.3f} | "
            f"{s['avg_R_net']:+.3f} | {s['median_R_net']:+.3f} | "
            f"{s['tp_pct']:.3f} | {s['sl_pct']:.3f} | {s['timeout_pct']:.3f} | {s['partial_pct']:.3f} | "
            f"{s['avg_bars_held']:.2f} | {s['total_R_net']:+.2f} |"
        )
    lines.append("")

    # Per-year tables.
    lines.append("## By-year breakdown")
    lines.append("")
    for s in summaries:
        if s.get("N", 0) == 0 or not s.get("by_year"):
            continue
        lines.append(f"### {s['exit_variant']}")
        lines.append("")
        lines.append("| year | n | avg_R | med_R | PF | WR |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for yr, m in sorted(s["by_year"].items()):
            lines.append(f"| {yr} | {m['n']} | {m['avg_R']:+.3f} | {m['med_R']:+.3f} | "
                         f"{m['PF']:.3f} | {m['WR']:.3f} |")
        lines.append("")

    # Volatility buckets.
    lines.append("## Volatility-bucket breakdown (terciles of atr14_prior / close_1700)")
    lines.append("")
    for s in summaries:
        if s.get("N", 0) == 0 or not s.get("vol_buckets"):
            continue
        lines.append(f"### {s['exit_variant']}")
        lines.append("")
        lines.append("| bucket | n | avg_R | PF | WR |")
        lines.append("|---|---:|---:|---:|---:|")
        for bk in ("low_vol", "mid_vol", "high_vol"):
            m = s["vol_buckets"].get(bk)
            if m is None:
                continue
            lines.append(f"| {bk} | {m['n']} | {m['avg_R']:+.3f} | {m['PF']:.3f} | {m['WR']:.3f} |")
        lines.append("")

    # Concentration.
    lines.append("## Ticker concentration (top-5 by row count)")
    lines.append("")
    for s in summaries:
        if s.get("N", 0) == 0:
            continue
        tt = s["top_tickers"]
        lines.append(f"- **{s['exit_variant']}** — {tt}, top-5 R share: {s['top5_R_share']:+.3f}, "
                     f"bottom-5 R share: {s['bot5_R_share']:+.3f}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SBT-1700 multi-exit matrix.")
    ap.add_argument("--dataset", type=Path, default=Path("output/sbt_1700_dataset.parquet"))
    ap.add_argument("--master", type=Path, default=Path("output/ohlcv_10y_fintables_master.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    args = ap.parse_args()

    dataset = pd.read_parquet(args.dataset)
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    print(f"[exit_matrix] dataset: {dataset.shape}")

    master = _load_master(args.master)
    print(f"[exit_matrix] master: {len(master)} tickers")

    trades_df, summaries = run_exit_matrix(dataset, master)
    print(f"[exit_matrix] trades: {trades_df.shape}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_dir / "sbt_1700_exit_matrix_trades.parquet"
    csv_path = args.out_dir / "sbt_1700_exit_matrix.csv"
    md_path = args.out_dir / "sbt_1700_exit_matrix.md"

    trades_df.to_parquet(trades_path, index=False)
    to_summary_csv(summaries).to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(summaries))

    print(f"[exit_matrix] wrote {trades_path}")
    print(f"[exit_matrix] wrote {csv_path}")
    print(f"[exit_matrix] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
