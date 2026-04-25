"""Fit + persist the retention surrogate (UP + NONUP heads) from preds_v4C.

Producer side of the artifact loaded by ``nyxexpansion.retention.surrogate.load``.

Run:
    python -m nyxexpansion.retention.train_surrogate

Output:
    output/nyxexp_retention_surrogate_v1.pkl   (binary bundle)
    output/nyxexp_retention_surrogate_v1.json  (human-readable metadata)

Acceptance gate (mirrors research surrogate at
``nyxexpansion/research/truncated_17_00_sensitivity.py``):

- UP head Spearman rho on hold-out ≥ 0.88 (research: 0.9028)
- NONUP head Spearman rho on hold-out ≥ 0.88 (research: 0.9017)

If either fails, the script aborts WITHOUT writing the artifact so a stale
bundle is never picked up by the live scan.

Hyperparameters and split protocol kept bit-identical to the research
sensitivity script: 80/20 random split with seed=17, LGBMRegressor with
the locked params below, early stopping at 40 rounds on validation L1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nyxexpansion.features import CORE_FEATURES_NONUP, CORE_FEATURES_UP
from nyxexpansion.retention import TRUNCATED_FEATURE_SCHEMA_VERSION
from nyxexpansion.retention.surrogate import (
    SURROGATE_MODEL_VERSION,
    write_sidecar,
)


PREDS_PATH = Path("output/nyxexp_preds_v4C.parquet")
ARTIFACT_PATH = Path(f"output/nyxexp_retention_surrogate_{SURROGATE_MODEL_VERSION}.pkl")

ACCEPTANCE_RHO = 0.88

REGIME_SPLIT_RULE = (
    "preds_v4C.model_kind ∈ {'up', 'nonup'} — UP head uses CORE_FEATURES_UP "
    "(V1 + J + chase_score_soft); NONUP head uses CORE_FEATURES_V1 "
    "(no J block, no chase). xu_regime determines model_kind upstream in v4C."
)

LGBM_PARAMS: dict = dict(
    objective="regression_l1",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=17,
    verbose=-1,
    n_jobs=-1,
)

SPLIT_SEED = 17
TRAIN_FRAC = 0.8
EARLY_STOP_ROUNDS = 40


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_cols(cols: list[str]) -> str:
    return hashlib.sha256("|".join(cols).encode("utf-8")).hexdigest()


def _fit_one_head(
    preds: pd.DataFrame,
    *,
    regime: str,
    feature_cols_canonical: list[str],
    label: str,
) -> dict:
    sub = preds[preds["model_kind"] == regime].copy()
    feat_cols = [c for c in feature_cols_canonical if c in sub.columns]
    missing = sorted(set(feature_cols_canonical) - set(feat_cols))
    if missing:
        raise RuntimeError(
            f"[{label}] preds_v4C is missing canonical feature columns: {missing}"
        )

    X = sub[feat_cols].astype(float)
    y = sub["winner_R_pred"].astype(float)
    n = len(X)
    if n < 100:
        raise RuntimeError(
            f"[{label}] only {n} rows for regime={regime!r}; need ≥100 to fit"
        )

    rng = np.random.default_rng(SPLIT_SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * TRAIN_FRAC)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    reg = lgb.LGBMRegressor(**LGBM_PARAMS)
    reg.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    yhat_va = reg.predict(X.iloc[va_idx], num_iteration=reg.best_iteration_)
    rho, _ = spearmanr(y.iloc[va_idx].values, yhat_va)

    print(
        f"  [{label}] n_train={len(tr_idx)} n_val={len(va_idx)} "
        f"best_iter={reg.best_iteration_} rho={rho:.4f}"
    )
    return {
        "feature_columns": feat_cols,
        "feature_columns_hash": _sha256_cols(feat_cols),
        "model": reg,
        "best_iteration": int(reg.best_iteration_) if reg.best_iteration_ else None,
        "validation_rho": float(rho),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=PREDS_PATH,
                    help="Path to nyxexp_preds_v4C.parquet")
    ap.add_argument("--out", type=Path, default=ARTIFACT_PATH,
                    help="Output artifact pickle path")
    ap.add_argument("--acceptance-rho", type=float, default=ACCEPTANCE_RHO,
                    help="Minimum hold-out Spearman rho per head")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Retention surrogate train — model_version={SURROGATE_MODEL_VERSION}, "
          f"schema={TRUNCATED_FEATURE_SCHEMA_VERSION}")
    print("=" * 70)
    print(f"  preds: {args.preds}")
    print(f"  out  : {args.out}")
    print(f"  acceptance rho: {args.acceptance_rho}")

    if not args.preds.exists():
        print(f"ERROR: preds parquet not found: {args.preds}", file=sys.stderr)
        return 2

    preds = pd.read_parquet(args.preds)
    if "model_kind" not in preds.columns:
        print("ERROR: preds parquet missing 'model_kind' column", file=sys.stderr)
        return 2
    if "winner_R_pred" not in preds.columns:
        print("ERROR: preds parquet missing 'winner_R_pred' column", file=sys.stderr)
        return 2

    print(f"\n[1/3] Loaded preds: rows={len(preds):,}  "
          f"up={int((preds.model_kind=='up').sum())}  "
          f"nonup={int((preds.model_kind=='nonup').sum())}")

    print("\n[2/3] Fitting heads…")
    up_bundle = _fit_one_head(
        preds, regime="up",
        feature_cols_canonical=CORE_FEATURES_UP, label="UP",
    )
    nu_bundle = _fit_one_head(
        preds, regime="nonup",
        feature_cols_canonical=CORE_FEATURES_NONUP, label="NONUP",
    )

    if up_bundle["validation_rho"] < args.acceptance_rho:
        print(f"\nFAIL: UP rho {up_bundle['validation_rho']:.4f} "
              f"< acceptance {args.acceptance_rho}", file=sys.stderr)
        return 3
    if nu_bundle["validation_rho"] < args.acceptance_rho:
        print(f"\nFAIL: NONUP rho {nu_bundle['validation_rho']:.4f} "
              f"< acceptance {args.acceptance_rho}", file=sys.stderr)
        return 3

    print("\n[3/3] Persisting artifact…")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_version": SURROGATE_MODEL_VERSION,
        "truncated_feature_schema_version": TRUNCATED_FEATURE_SCHEMA_VERSION,
        "training_dataset_path": str(args.preds),
        "training_dataset_hash": _sha256_file(args.preds),
        "training_dataset_mtime": dt.datetime.fromtimestamp(
            args.preds.stat().st_mtime, tz=dt.timezone.utc,
        ).isoformat(),
        "created_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "regime_split_rule": REGIME_SPLIT_RULE,
        "lgbm_params": LGBM_PARAMS,
        "split": {
            "seed": SPLIT_SEED,
            "train_frac": TRAIN_FRAC,
            "early_stop_rounds": EARLY_STOP_ROUNDS,
        },
        "acceptance_rho": float(args.acceptance_rho),
        "up": up_bundle,
        "nonup": nu_bundle,
    }
    with args.out.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    sidecar = write_sidecar(args.out, bundle)

    print(f"  pickle:  {args.out}  ({args.out.stat().st_size:,} bytes)")
    print(f"  sidecar: {sidecar}")
    print(f"  UP rho:    {up_bundle['validation_rho']:.4f}  "
          f"feats={len(up_bundle['feature_columns'])}")
    print(f"  NONUP rho: {nu_bundle['validation_rho']:.4f}  "
          f"feats={len(nu_bundle['feature_columns'])}")
    print("\nOK — artifact persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
