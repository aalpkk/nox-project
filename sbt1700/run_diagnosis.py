"""Single-shot orchestrator: exit_matrix → ranker_diagnosis → edge_diagnosis.

CLI:
  python -m sbt1700.run_diagnosis \\
      --dataset output/sbt_1700_dataset.parquet \\
      --master output/ohlcv_10y_fintables_master.parquet \\
      --out-dir output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sbt1700.exit_matrix import (
    _load_master,
    run_exit_matrix,
    to_markdown,
    to_summary_csv,
)
from sbt1700.ranker_diagnosis import run_ranker_diagnosis
from sbt1700.edge_diagnosis import write_diagnosis


def main() -> int:
    ap = argparse.ArgumentParser(description="SBT-1700 PR-2 edge diagnosis pipeline.")
    ap.add_argument("--dataset", type=Path, default=Path("output/sbt_1700_dataset.parquet"))
    ap.add_argument("--master", type=Path, default=Path("output/ohlcv_10y_fintables_master.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.dataset)
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    print(f"[diagnosis] dataset: {dataset.shape}")

    master = _load_master(args.master)
    print(f"[diagnosis] master: {len(master)} tickers")

    trades_df, summaries = run_exit_matrix(dataset, master)
    print(f"[diagnosis] exit-matrix trades: {trades_df.shape}")

    trades_path = args.out_dir / "sbt_1700_exit_matrix_trades.parquet"
    summary_csv = args.out_dir / "sbt_1700_exit_matrix.csv"
    summary_md = args.out_dir / "sbt_1700_exit_matrix.md"
    ranker_csv = args.out_dir / "sbt_1700_ranker_by_exit.csv"
    diag_md = args.out_dir / "sbt_1700_edge_diagnosis.md"

    trades_df.to_parquet(trades_path, index=False)
    to_summary_csv(summaries).to_csv(summary_csv, index=False)
    summary_md.write_text(to_markdown(summaries))
    print(f"[diagnosis] wrote {trades_path}")
    print(f"[diagnosis] wrote {summary_csv}")
    print(f"[diagnosis] wrote {summary_md}")

    ranker = run_ranker_diagnosis(dataset, trades_df)
    ranker.to_csv(ranker_csv, index=False)
    print(f"[diagnosis] wrote {ranker_csv} ({len(ranker)} fold rows)")

    write_diagnosis(summary_csv, ranker_csv, diag_md)
    print(f"[diagnosis] wrote {diag_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
