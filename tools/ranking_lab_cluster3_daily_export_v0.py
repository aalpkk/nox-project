"""Ranking Lab v0 — cluster3 daily picks exporter (advisor contract).

Reads the cluster3 paper-snapshot ledger, selects the archetype matches LOCKED
for a given as-of date, joins the entry reference price from the features panel,
and emits the daily CSV (+ a compact HTML) that the portfolio advisor's
`scanner_reader` ingests from the GitHub Actions artifact.

cluster3 is a LONG defensive archetype (paper / weekly research line) — every
row is an AL candidate; `fast_rank_early` is the quality score.

Inputs:
  - output/ranking_lab_cluster3_paper_snapshots_v0.parquet
  - output/ranking_lab_features_v0.parquet   (entry_ref join)

Outputs:
  - output/cluster3_signals_<YYYYMMDD>.csv
  - output/cluster3_signals_<YYYYMMDD>.html

Usage:
  python tools/ranking_lab_cluster3_daily_export_v0.py --asof YYYY-MM-DD
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
SNAPSHOTS = OUT / "ranking_lab_cluster3_paper_snapshots_v0.parquet"
FEATURES = OUT / "ranking_lab_features_v0.parquet"

# Columns written to the CSV consumed by agent/scanner_reader._parse_cluster3
CSV_COLS = [
    "ticker", "signal_date", "source", "family", "close",
    "fast_rank_early", "p3_8", "p5_10", "p10_15",
    "ret_5d", "price_vs_60d_high", "atr_pct", "cluster_assigned",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD (snapshot as-of date)")
    a = ap.parse_args()
    asof = pd.Timestamp(a.asof).normalize()
    tag = asof.strftime("%Y%m%d")
    csv_path = OUT / f"cluster3_signals_{tag}.csv"
    html_path = OUT / f"cluster3_signals_{tag}.html"

    if not SNAPSHOTS.exists():
        print(f"[export] no snapshot ledger ({SNAPSHOTS}); writing empty CSV")
        pd.DataFrame(columns=CSV_COLS).to_csv(csv_path, index=False)
        html_path.write_text("<html><body><p>cluster3: no ledger</p></body></html>")
        return 0

    snap = pd.read_parquet(SNAPSHOTS)
    snap["snapshot_asof_date"] = pd.to_datetime(snap["snapshot_asof_date"]).dt.normalize()
    snap["signal_date"] = pd.to_datetime(snap["signal_date"]).dt.normalize()
    today = snap[snap["snapshot_asof_date"] == asof].copy()
    print(f"[export] asof={asof.date()}  archetype picks={len(today)}")

    if today.empty:
        pd.DataFrame(columns=CSV_COLS).to_csv(csv_path, index=False)
        html_path.write_text(
            f"<html><body><h3>cluster3 — {asof.date()}</h3>"
            "<p>No archetype matches for this as-of date (paper / weekly line).</p></body></html>")
        print(f"[export] wrote empty {csv_path.name}")
        return 0

    # Entry reference price from the features panel (entry_ref is dropped from the
    # model but retained in the panel; cluster3 is long, so it is the AL anchor).
    feat = pd.read_parquet(FEATURES, columns=["ticker", "signal_date", "family", "entry_ref"])
    feat["signal_date"] = pd.to_datetime(feat["signal_date"]).dt.normalize()
    feat = feat.drop_duplicates(["ticker", "signal_date", "family"], keep="first")
    today = today.merge(feat, on=["ticker", "signal_date", "family"], how="left")
    today["close"] = today["entry_ref"]

    out = today.reindex(columns=CSV_COLS)
    out = out.sort_values("fast_rank_early", ascending=False).reset_index(drop=True)
    out.to_csv(csv_path, index=False)
    print(f"[export] wrote {csv_path}  ({len(out)} rows)")

    # Compact HTML (parity with other scanners' GH-Pages artifact)
    rows_html = "\n".join(
        f"<tr><td>{r.ticker}</td><td>{r.source}/{r.family}</td>"
        f"<td>{r.close:.2f}</td><td>{r.fast_rank_early:.3f}</td>"
        f"<td>{r.ret_5d:+.1%}</td><td>{r.price_vs_60d_high:+.1%}</td></tr>"
        for r in out.itertuples(index=False)
    )
    html_path.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:system-ui;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}</style></head><body>"
        f"<h3>cluster3 archetype — {asof.date()} "
        "<small>(paper / haftalık — bilgi)</small></h3>"
        "<table><tr><th>Ticker</th><th>src/family</th><th>entry_ref</th>"
        "<th>fast_rank_early</th><th>ret_5d</th><th>vs60dH</th></tr>"
        f"{rows_html}</table></body></html>")
    print(f"[export] wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
