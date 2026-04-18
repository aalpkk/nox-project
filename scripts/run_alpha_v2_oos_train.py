"""
Alpha v2 — OOS train helper.

Labels dataset'ini tarih kesiti ile böler, train kısmında Model B eğitir,
test kısmında OOS predictions üretir. Sonra save_models ile OOS-only model
kaydeder — smoke backtest'inde test döneme uygulanır.

Çalıştırma:
    python scripts/run_alpha_v2_oos_train.py \\
        --labels output/alpha_v2_labels_gen6.parquet \\
        --train-frac 0.70 \\
        --out output/alpha_v2_model_b_gen6_oos.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from alpha_v2.layer4_edge.model_b import (  # noqa: E402
    train_model_b, save_models, FEATURE_COLS_DEFAULT,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='output/alpha_v2_labels_gen6.parquet')
    ap.add_argument('--train-frac', type=float, default=0.70)
    ap.add_argument('--out', default='output/alpha_v2_model_b_gen6_oos.pkl')
    args = ap.parse_args()

    df = pd.read_parquet(args.labels).sort_values('date').reset_index(drop=True)
    cutoff_idx = int(len(df) * args.train_frac)
    cutoff_date = pd.to_datetime(df.iloc[cutoff_idx]['date'])
    train_df = df.iloc[:cutoff_idx].copy()
    print(f"📥 {len(df)} label  |  cutoff {cutoff_date.date()}  "
          f"|  train {len(train_df)}  test {len(df) - cutoff_idx}")

    models = {}
    for bucket in sorted(train_df['bucket'].unique()):
        sub = train_df[train_df['bucket'] == bucket]
        if len(sub) < 200:
            print(f"\n⚠ {bucket}: {len(sub)} < 200; atlandı")
            continue

        print(f"\n═══ {bucket.upper()} train={len(sub)} ═══")
        try:
            result = train_model_b(
                train_df, bucket,
                feature_cols=FEATURE_COLS_DEFAULT,
                target_col='pd_hit', n_folds=5, embargo_bars=10,
            )
        except Exception as e:
            print(f"  ❌ {e}")
            continue
        print(f"  CV AUC: {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
        print(f"  Top-3: {', '.join(result.feature_importance.head(3)['feature'].tolist())}")
        models[bucket] = result

    if not models:
        print("❌ Hiç model eğitilemedi")
        return 1

    save_models(models, args.out)
    print(f"\n💾 {len(models)} OOS model → {args.out}")
    print(f"ℹ️  Test cutoff: trades entry_date > {cutoff_date.date()} = OOS")
    # Save cutoff for downstream
    with open(Path(args.out).with_suffix('.cutoff'), 'w') as f:
        f.write(str(cutoff_date.date()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
