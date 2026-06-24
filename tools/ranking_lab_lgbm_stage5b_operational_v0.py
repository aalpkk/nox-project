"""Ranking Lab v0 — Stage 5b OPERATIONAL frozen scorer (freeze + score).

Purpose
-------
The research script `ranking_lab_lgbm_v0_stage5b.py` RE-TRAINS LightGBM
walk-forward on every run. That is correct for offline evaluation but unusable
in a daily CI close-chain (heavy, and "frozen model" in name only). This module
splits the operational path off without touching the research script:

  * `freeze` — replicate the Stage 5b **F5 forward fold** exactly (train on all
    labeled history through the F5 cutoff, deterministic seed=42) for the three
    SHORT heads (y3_8 / y5_10 / y10_15) and persist the boosters + the exact
    preprocessing needed to reproduce their inputs (train-fold p1/p99 winsorize
    caps + categorical level order). One-time; the bundle is committed.

  * `score` — load the frozen boosters and score the FORWARD event set
    (signal_date > cutoff) with the identical feature pipeline, emit
    `fast_rank_early` (= 0.40·p3_8 + 0.35·p5_10 + 0.25·p10_15) and upsert the
    forward rows into `ranking_lab_fast_rank_scores_v0.parquet`. No training.

Why F5: Stage 5b's F5 fold is *already* the operational forward scorer — train
≤ cutoff, predict the fresh post-cutoff events strictly out-of-sample. Freezing
its boosters reproduces the same scores the cluster3 archetype model was frozen
against, so fresh candidates get a TRUE fast_rank_early (not a median fill).

Inputs (both modes):
  - output/ranking_lab_features_v0.parquet

freeze outputs:
  - output/ranking_lab_lgbm_stage5b_frozen_v0/booster_{y3_8,y5_10,y10_15}.txt
  - output/ranking_lab_lgbm_stage5b_frozen_v0/meta.json

score outputs:
  - output/ranking_lab_fast_rank_scores_v0.parquet  (forward rows upserted)

Usage:
  python tools/ranking_lab_lgbm_stage5b_operational_v0.py freeze [--force] [--stamp YYYY-MM-DD]
  python tools/ranking_lab_lgbm_stage5b_operational_v0.py score [--asof YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "tools"))

# Reuse the EXACT research-script constants + feature pipeline (no re-implementation).
import ranking_lab_lgbm_v0_stage5b as s5b  # noqa: E402

FEAT_PATH = OUT / "ranking_lab_features_v0.parquet"
SCORES_PATH = OUT / "ranking_lab_fast_rank_scores_v0.parquet"
FROZEN_DIR = OUT / "ranking_lab_lgbm_stage5b_frozen_v0"
META_PATH = FROZEN_DIR / "meta.json"

# SHORT heads only — the three that define fast_rank_early. Tail heads are not
# used by cluster3 / fast_rank scoring (excluded from FAST_RANK by ONAY).
SHORT_TARGETS = [
    ("y3_8", "mfe_pct_3d", 0.08),
    ("y5_10", "mfe_pct_5d", 0.10),
    ("y10_15", "mfe_pct_10d", 0.15),
]
HEAD_TO_PCOL = {"y3_8": "p3_8", "y5_10": "p5_10", "y10_15": "p10_15"}
# fast_rank_early weights (verbatim from ranking_lab_fast_rank_v0.py)
FR_EARLY_W = {"p3_8": 0.40, "p5_10": 0.35, "p10_15": 0.25}


def _load_features() -> pd.DataFrame:
    feats = pd.read_parquet(FEAT_PATH)
    feats["signal_date"] = pd.to_datetime(feats["signal_date"]).dt.normalize()
    for b in s5b.RANK_BASES + s5b.AUDIT_RANK_BASES:
        assert b in feats.columns, f"missing rank-base column: {b}"
    feats = s5b._add_rank_features(feats)
    missing = [c for c in s5b.FEATURE_COLS if c not in feats.columns]
    assert not missing, f"missing feature columns: {missing}"
    return feats


# --------------------------------------------------------------------------- freeze
def cmd_freeze(force: bool, stamp: str | None) -> int:
    if META_PATH.exists() and not force:
        print(f"[freeze] frozen bundle exists: {META_PATH} (use --force to refit)")
        return 0
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    feats = _load_features()

    # F5 = forward fold: train on ALL labeled history through cutoff (te).
    f5 = next(f for f in s5b.FOLDS if f[0] == "F5")
    _, ts, te, _, _ = f5
    cutoff = pd.Timestamp(te)
    train_raw = feats[(feats["signal_date"] >= pd.Timestamp(ts))
                      & (feats["signal_date"] <= cutoff)].copy()
    print(f"[freeze] F5 train window {ts} → {te}  rows={len(train_raw):,}")

    # Winsorize caps computed from TRAIN ONLY, then applied (identical to research).
    caps = s5b._winsorize_caps(train_raw)
    train = s5b._apply_winsorize(train_raw, caps)

    # Categorical level order captured from the winsorized train build, so frozen
    # scoring reconstructs identical LightGBM category codes.
    X_train = s5b._build_X(train)
    cat_levels = {c: list(map(str, X_train[c].cat.categories)) for c in s5b.CATEGORICAL_FEATURES}

    heads_meta = []
    for name, lcol, thr in SHORT_TARGETS:
        # _train_one expects already-winsorized train; test_df only used for a
        # throwaway predict (we keep the booster). Pass a 1-row slice.
        _, _, fit_info, booster = s5b._train_one(train, train.iloc[:1], lcol, thr, caps)
        best_iter = int(fit_info["best_iteration"])
        booster.save_model(str(FROZEN_DIR / f"booster_{name}.txt"),
                           num_iteration=best_iter)
        heads_meta.append({"name": name, "label_col": lcol, "threshold": thr,
                           "best_iteration": best_iter,
                           "n_train_used": fit_info["n_train_used"]})
        print(f"[freeze]   {name}: best_iter={best_iter} "
              f"n_train_used={fit_info['n_train_used']:,} → booster_{name}.txt")

    meta = {
        "version": "STAGE5B_OPERATIONAL_v0",
        "frozen_at": stamp or "unset",
        "source_research_script": "ranking_lab_lgbm_v0_stage5b.py",
        "fold": "F5",
        "cutoff": str(cutoff.date()),
        "lgbm_params": s5b.LGBM_PARAMS,
        "numeric_features": s5b.NUMERIC_FEATURES,
        "categorical_features": s5b.CATEGORICAL_FEATURES,
        "feature_cols": s5b.FEATURE_COLS,
        "winsorize_caps": {c: [float(lo), float(hi)] for c, (lo, hi) in caps.items()},
        "categorical_levels": cat_levels,
        "fast_rank_early_weights": FR_EARLY_W,
        "heads": heads_meta,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[freeze] wrote {META_PATH}")
    return 0


# ---------------------------------------------------------------------------- score
def _build_X_frozen(forward_w: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    """Build the model matrix with categoricals pinned to the frozen level order
    so LightGBM category codes match training exactly. Unseen levels → NaN code
    (treated as missing by the booster), identical to a real live-scoring day."""
    X = s5b._build_X(forward_w)  # sets categories from data + numeric coercion
    for c in s5b.CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(object), categories=cat_levels[c])
    return X


def _empty_scores_frame() -> pd.DataFrame:
    cols = ["ticker", "signal_date", "source", "family", "fold",
            "p3_8", "p5_10", "p10_15", "fast_rank_early"]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def cmd_score(asof: str | None) -> int:
    if not META_PATH.exists():
        raise SystemExit(f"[score] frozen bundle missing: {META_PATH} — run `freeze` first")
    meta = json.loads(META_PATH.read_text())
    cutoff = pd.Timestamp(meta["cutoff"])
    caps = {c: (float(lo), float(hi)) for c, (lo, hi) in meta["winsorize_caps"].items()}
    cat_levels = meta["categorical_levels"]

    feats = _load_features()
    fwd = feats[feats["signal_date"] > cutoff].copy()
    if asof:
        fwd = fwd[fwd["signal_date"] <= pd.Timestamp(asof).normalize()].copy()
    print(f"[score] forward set: signal_date > {cutoff.date()}"
          + (f" and ≤ {asof}" if asof else "") + f"  rows={len(fwd):,}")

    if fwd.empty:
        out = _empty_scores_frame()
    else:
        fwd_w = s5b._apply_winsorize(fwd, caps)
        X = _build_X_frozen(fwd_w, cat_levels)
        out = fwd[["ticker", "signal_date", "source", "family"]].copy()
        out["fold"] = "F5_OP"
        for h in meta["heads"]:
            pcol = HEAD_TO_PCOL[h["name"]]
            booster = lgb.Booster(model_file=str(FROZEN_DIR / f"booster_{h['name']}.txt"))
            out[pcol] = booster.predict(X, num_iteration=h["best_iteration"])
        w = meta["fast_rank_early_weights"]
        out["fast_rank_early"] = (w["p3_8"] * out["p3_8"]
                                  + w["p5_10"] * out["p5_10"]
                                  + w["p10_15"] * out["p10_15"])
        print(f"[score] scored {len(out):,} forward rows "
              f"(fast_rank_early μ={out['fast_rank_early'].mean():.4f} "
              f"p90={out['fast_rank_early'].quantile(0.90):.4f})")

    # Upsert: keep historical base rows ≤ cutoff (out-of-fold research scores
    # the cluster3 model was frozen against), replace the entire forward window.
    if SCORES_PATH.exists():
        base = pd.read_parquet(SCORES_PATH)
        base["signal_date"] = pd.to_datetime(base["signal_date"]).dt.normalize()
        keep = base[base["signal_date"] <= cutoff].copy()
        combined = pd.concat([keep, out], ignore_index=True)
        print(f"[score] upsert: kept {len(keep):,} base rows ≤ {cutoff.date()} + "
              f"{len(out):,} forward rows → {len(combined):,}")
    else:
        combined = out
        print(f"[score] no base scores file — writing forward-only ({len(combined):,} rows)")

    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    combined.to_parquet(SCORES_PATH, index=False)
    print(f"[score] wrote {SCORES_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5b operational frozen scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("freeze", help="train + persist F5 boosters (one-time)")
    pf.add_argument("--force", action="store_true")
    pf.add_argument("--stamp", default=None, help="freeze date stamp (YYYY-MM-DD)")
    ps = sub.add_parser("score", help="score forward events with frozen boosters")
    ps.add_argument("--asof", default=None, help="YYYY-MM-DD (default: all forward)")
    a = ap.parse_args()
    if a.cmd == "freeze":
        return cmd_freeze(force=a.force, stamp=a.stamp)
    if a.cmd == "score":
        return cmd_score(asof=a.asof)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
