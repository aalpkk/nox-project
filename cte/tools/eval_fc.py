"""
CTE eval_fc — per-line eval for the FC pipeline.

Reports score_model vs score_compression vs score_random on the FC preds file
only. No cross-line "overall" here — that concept no longer exists at the
model layer.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.line import eval_line, format_eval_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="output/cte_fc_preds_v1.parquet")
    ap.add_argument("--target", default="runner_15")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    os.chdir(str(_ROOT))
    if not Path(args.preds).exists():
        print(f"❌ FC preds yok: {args.preds}  →  python -m cte.train_fc")
        return 2

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    print(f"FC preds: {args.preds}  shape={df.shape}")

    evald = eval_line(df, target=args.target, line="fc")
    print(format_eval_report(evald))

    if args.out_json:
        import json
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(evald, f, indent=2, default=str)
        print(f"\n[WRITE] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
