"""
CTE train_fc — FC (falling channel) production line.

Default mode: pure (train ONLY on setup_type=='fc' rows; still score all
trigger_fc==True rows including "both"). Rationale: v2b ablation showed
FC-head pure (1.89x lift@10) beats FC-head mixed (1.35x) — "both" rows
dilute the narrower FC geometry during training.

Kullanım:
  python -m cte.train_fc                      # default mode=pure
  python -m cte.train_fc --mode mixed         # research variant
  python -m cte.train_fc --target runner_10

Outputs:
  output/cte_fc_preds_v1.parquet
  output/cte_fc_importance_v1.csv

Production entrypoint. No cross-line combining; HB line is run separately via
cte.train_hb. Optional portfolio merge is in cte.tools.portfolio_merge.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from dataclasses import asdict

from cte.config import CONFIG
from cte.contract import CONTRACT_PATH, verify_contract
from cte.features import FEATURES_V1
from cte.line import train_line, save_line_artifacts
from cte.train import LGBMParams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--mode", choices=["pure", "mixed"],
                    default=CONFIG.line.fc_mode_default)
    ap.add_argument("--target", default=CONFIG.label.primary_target)
    ap.add_argument("--out-preds", default="output/cte_fc_preds_v1.parquet")
    ap.add_argument("--out-imp", default="output/cte_fc_importance_v1.csv")
    ap.add_argument("--contract", default=str(CONTRACT_PATH))
    ap.add_argument("--ignore-contract", action="store_true",
                    help="Contract doğrulamasını atla (tavsiye edilmez)")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset} → python -m cte.dataset")
        return 2

    if args.ignore_contract:
        print("⚠ --ignore-contract: CTE contract doğrulaması atlandı (FC uzman)")
    else:
        verify_contract(
            dataset_path=args.dataset,
            contract_path=args.contract,
            raise_on_mismatch=True,
            verbose=True,
            line="fc",
        )

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"])
    print(f"═══ CTE FC-line v1 ═══  dataset={args.dataset}  shape={df.shape}")

    feature_cols = [c for c in FEATURES_V1 if c in df.columns]
    missing = [c for c in FEATURES_V1 if c not in df.columns]
    if missing:
        print(f"⚠ missing features (dropped): {missing}")
    print(f"  features={len(feature_cols)}  target={args.target}  mode={args.mode}")

    if args.target not in df.columns:
        print(f"❌ target {args.target} not in dataset columns")
        return 3

    lgbm_kwargs = asdict(LGBMParams())
    lgbm_kwargs["min_child_samples"] = CONFIG.line.fc_lgbm_min_child_samples
    fc_params = LGBMParams(**lgbm_kwargs)
    print(f"  FC-local LGBM override: min_child_samples={fc_params.min_child_samples}")

    result = train_line(
        df=df,
        line="fc",
        mode=args.mode,
        target=args.target,
        feature_cols=feature_cols,
        split=CONFIG.split,
        params=fc_params,
    )
    save_line_artifacts(result, args.out_preds, args.out_imp)

    if not result.importance.empty:
        print("\n[FC-line top 10 features by mean gain]")
        print(result.importance.head(10)[["mean"]].to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
