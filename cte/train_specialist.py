"""
[LEGACY / RESEARCH — superseded by cte.train_hb + cte.train_fc (2026-04-23)]

Kept for apples-to-apples comparisons with v1/v2/v2a/v2b ablations. For
production, use the separated HB and FC lines:
  - cte.train_hb        (default mode=mixed)
  - cte.train_fc        (default mode=pure)
  - cte.tools.eval_hb, cte.tools.eval_fc
  - cte.tools.shortlist
  - cte.tools.portfolio_merge

The "single universal ranker" combine step implemented here is NO LONGER
production default. See memory/cte_v2_specialist.md for why (cross-line
score-scale calibration mismatch).

───────────────────────────────────────────────────────────────────────────

CTE train v2 — specialist heads (HB-head + FC-head), walk-forward.

v1 ablation showed:
  - combined head (all setups): lift@10 = 0.74x (no edge)
  - fc-only subset:             lift@10 = 1.89x (strong edge)
  - hb-only subset:             lift@10 = 0.91x (marginal)

Hypothesis: feature-target relationship differs between HB and FC geometries;
a single ranker can't learn both. Train two specialist heads and combine at
scoring time by setup membership.

Membership convention:
  - setup_type == "hb"   → hb-head only
  - setup_type == "fc"   → fc-head only
  - setup_type == "both" → average of the two heads' per-fold scores
Training set per head: rows where the corresponding trigger_hb / trigger_fc is True.
  → hb-head trains on {hb, both} rows
  → fc-head trains on {fc, both} rows

Output:
  output/cte_preds_v2.parquet         (unified test preds with score_model v2)
  output/cte_importance_v2_hb.csv     (per-fold gain, hb head)
  output/cte_importance_v2_fc.csv     (per-fold gain, fc head)

Kullanım:
  python -m cte.train_specialist
  python -m cte.train_specialist --target runner_15
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

import lightgbm as lgb

from cte.config import CONFIG
from cte.features import FEATURES_V1
from cte.train import LGBMParams, _per_date_rank


def _train_fold_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    fold_name: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    params: LGBMParams,
) -> tuple[lgb.Booster, pd.DataFrame, pd.Series]:
    """Like cte.train.train_fold but takes separate train_df and test_df.
    train_df is filtered by date + target-not-null; test_df is filtered by
    test date range only. Lets pure-specialist training score "both" rows
    at inference even though they were excluded from training.
    """
    td = pd.to_datetime(train_df["date"])
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    vs = pd.Timestamp(val_start)
    ve = pd.Timestamp(val_end)

    tr_mask = (td >= ts) & (td <= te) & train_df[target_col].notna()
    va_mask = (td >= vs) & (td <= ve) & train_df[target_col].notna()
    tr = train_df.loc[tr_mask].copy()
    va = train_df.loc[va_mask].copy()

    qd = pd.to_datetime(test_df["date"])
    qs = pd.Timestamp(test_start)
    qe = pd.Timestamp(test_end)
    qt_mask = (qd >= qs) & (qd <= qe) & test_df[target_col].notna()
    qt = test_df.loc[qt_mask].copy()

    X_tr = tr[feature_cols]; y_tr = tr[target_col].astype(int)
    X_va = va[feature_cols]; y_va = va[target_col].astype(int)
    X_qt = qt[feature_cols]

    print(f"  [{fold_name}] train N={len(tr):,}  val N={len(va):,}  test N={len(qt):,}  "
          f"(pos tr/va/qt: {y_tr.mean()*100:.1f}% / {y_va.mean()*100:.1f}% / "
          f"{qt[target_col].mean()*100:.1f}%)")

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        max_depth=params.max_depth,
        min_child_samples=params.min_child_samples,
        feature_fraction=params.feature_fraction,
        bagging_fraction=params.bagging_fraction,
        bagging_freq=params.bagging_freq,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=params.seed,
        verbose=-1,
        n_jobs=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = clf.best_iteration_
    print(f"      best_iter={best_iter}")

    scores = clf.predict_proba(X_qt, num_iteration=best_iter)[:, 1]
    qt["score_model"] = scores
    qt["fold_assigned"] = fold_name
    rng = np.random.default_rng(params.seed + hash(fold_name) % 1000)
    qt["score_random"] = rng.uniform(0, 1, size=len(qt))
    if "compression_score" in qt.columns:
        qt["score_compression"] = _per_date_rank(qt["compression_score"], qt["date"]).values
    else:
        qt["score_compression"] = np.nan

    imp = pd.Series(
        clf.booster_.feature_importance(importance_type="gain"),
        index=feature_cols,
        name=fold_name,
    )
    return clf.booster_, qt, imp


def _train_head(
    df: pd.DataFrame,
    head: str,
    feature_cols: list[str],
    target: str,
    params: LGBMParams,
    pure: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one specialist head across all folds. Returns (test_preds, importance).

    pure=True: train only on geometry-pure rows (setup_type == head), but still
    score *all* rows where the head's trigger fires (including setup_type=="both").
    "both" rows get a score at combine-time from both heads.
    """
    trigger_col = f"trigger_{head}"
    all_members = df[df[trigger_col].astype(bool)].copy()
    train_pool = (
        all_members[all_members["setup_type"] == head].copy()
        if pure
        else all_members.copy()
    )
    mode_tag = "pure" if pure else "mixed"
    print(
        f"\n━━━ {head.upper()}-head ({mode_tag})  "
        f"N_train_pool={len(train_pool):,}  N_test_pool={len(all_members):,}  "
        f"pos_rate(train)={train_pool[target].dropna().mean()*100:.1f}% ━━━"
    )

    all_test = []
    all_imp: list[pd.Series] = []
    split = CONFIG.split
    for fs in split.folds:
        print(f"  [FOLD {fs.name}]")
        try:
            _, qt, imp = _train_fold_split(
                train_pool, all_members, feature_cols, target, fs.name,
                train_start=split.train_start,
                train_end=fs.train_end,
                val_start=fs.val_start, val_end=fs.val_end,
                test_start=fs.test_start, test_end=fs.test_end,
                params=params,
            )
            qt["head"] = head
            all_test.append(qt)
            all_imp.append(imp.rename(f"{fs.name}"))
        except Exception as e:
            print(f"    fold {fs.name} failed: {e}")

    test_df = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    imp_df = pd.concat(all_imp, axis=1) if all_imp else pd.DataFrame()
    if not imp_df.empty:
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)
    return test_df, imp_df


def _combine_heads(hb_pred: pd.DataFrame, fc_pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """Unify test-row predictions across heads.

    Strategy: build a union index of (ticker, date) pairs seen by either head.
    For each row, attach score_hb (if hb-head saw it), score_fc (if fc-head
    saw it), and a shared metadata slice (setup_type, fold, compression score,
    target, etc.) taken from whichever head has it (they agree since both
    read from the same dataset).

    score_model logic:
      setup_type == "hb"    → score_hb
      setup_type == "fc"    → score_fc
      setup_type == "both"  → mean(score_hb, score_fc), fall back to whichever exists
    """
    if hb_pred.empty and fc_pred.empty:
        return pd.DataFrame()

    key = ["ticker", "date"]

    # Shared metadata columns (same across heads for a given key).
    shared_cols = [
        "setup_type", "fold_assigned", "trigger_hb", "trigger_fc",
        "score_compression", "score_random", "close", target,
    ]
    # Also carry label-family cols if present (for downstream PF proxy).
    label_family = [
        "hold_3_close", "hold_5_close", "failed_break_5_close",
        "mfe_10_atr", "mae_10_atr", "spike_rejected_10",
        "mfe_15_atr", "mae_15_atr", "spike_rejected_15",
        "mfe_20_atr", "mae_20_atr", "spike_rejected_20",
        "runner_10", "runner_15", "runner_20",
        "breakout_level_struct", "breakout_level_close",
    ]

    def _select(pred: pd.DataFrame, score_name: str) -> pd.DataFrame:
        if pred.empty:
            return pd.DataFrame(columns=key + [score_name])
        out = pred[key].copy()
        out[score_name] = pred["score_model"].values
        for c in shared_cols + label_family:
            if c in pred.columns and c not in out.columns:
                out[c] = pred[c].values
        return out

    hb = _select(hb_pred, "score_hb")
    fc = _select(fc_pred, "score_fc")

    if hb.empty:
        merged = fc.copy()
        merged["score_hb"] = np.nan
    elif fc.empty:
        merged = hb.copy()
        merged["score_fc"] = np.nan
    else:
        merged = hb.merge(fc, on=key, how="outer", suffixes=("", "_fc"))
        # Coalesce shared metadata columns from fc side into primary side.
        for col in shared_cols + label_family:
            alt = f"{col}_fc"
            if alt in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[alt])
                else:
                    merged[col] = merged[alt]
                merged = merged.drop(columns=[alt])

    # Combine score_model per setup_type.
    st = merged.get("setup_type")
    sh = merged.get("score_hb")
    sf = merged.get("score_fc")
    score_model = np.full(len(merged), np.nan, dtype=float)
    if st is not None:
        both_mask = (st == "both").values
        hb_mask = (st == "hb").values
        fc_mask = (st == "fc").values
        sh_arr = sh.values.astype(float) if sh is not None else np.full(len(merged), np.nan)
        sf_arr = sf.values.astype(float) if sf is not None else np.full(len(merged), np.nan)

        # both → mean of available scores
        both_mean = np.nanmean(np.vstack([sh_arr, sf_arr]), axis=0)
        score_model = np.where(both_mask, both_mean, score_model)
        # hb → score_hb
        score_model = np.where(hb_mask, sh_arr, score_model)
        # fc → score_fc
        score_model = np.where(fc_mask, sf_arr, score_model)
    merged["score_model"] = score_model
    return merged


def _eval_summary(preds: pd.DataFrame, target: str, title: str) -> None:
    print(f"\n──── {title} ────")
    mask = preds[target].notna() & preds["score_model"].notna()
    df = preds.loc[mask]
    if df.empty:
        print("  (no rows)")
        return
    n = len(df)
    base = df[target].mean()
    rho = float(df["score_model"].rank().corr(df[target].rank()))
    k = max(1, int(round(n * 0.10)))
    top = df.nlargest(k, "score_model")
    p10 = float(top[target].mean())
    lift = p10 / base if base > 0 else np.nan
    print(f"  N={n}  base={base:.2%}  rho={rho:+.3f}  p@10={p10:.2%}  lift@10={lift:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--target", default=CONFIG.label.primary_target)
    ap.add_argument("--out-preds", default=None,
                    help="Default: output/cte_preds_v2.parquet or _v2a if --pure")
    ap.add_argument("--out-imp-hb", default=None)
    ap.add_argument("--out-imp-fc", default=None)
    ap.add_argument("--pure", action="store_true",
                    help="Drop setup_type=='both' rows from each head's training set")
    ap.add_argument("--hb-pure", action="store_true",
                    help="HB-head drops 'both' rows (if absent and --pure not set, HB-head is mixed)")
    ap.add_argument("--fc-pure", action="store_true",
                    help="FC-head drops 'both' rows")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    hb_pure = args.pure or args.hb_pure
    fc_pure = args.pure or args.fc_pure
    if args.pure:
        variant = "v2a"
    elif fc_pure and not hb_pure:
        variant = "v2b_fcpure"
    elif hb_pure and not fc_pure:
        variant = "v2b_hbpure"
    else:
        variant = "v2"
    out_preds = args.out_preds or f"output/cte_preds_{variant}.parquet"
    out_imp_hb = args.out_imp_hb or f"output/cte_importance_{variant}_hb.csv"
    out_imp_fc = args.out_imp_fc or f"output/cte_importance_{variant}_fc.csv"

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}")
        return 2

    print("═══ CTE Train v2 — specialist heads ═══")
    df = pd.read_parquet(args.dataset)
    print(f"  shape: {df.shape}")
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [c for c in FEATURES_V1 if c in df.columns]
    print(f"  features: {len(feature_cols)}  target: {args.target}")
    if args.target not in df.columns:
        print(f"❌ target not in columns")
        return 3

    # Sanity: trigger_hb/trigger_fc exist
    for col in ("trigger_hb", "trigger_fc", "setup_type"):
        if col not in df.columns:
            print(f"❌ {col} missing in dataset")
            return 4

    params = LGBMParams()
    hb_pred, hb_imp = _train_head(df, "hb", feature_cols, args.target, params, pure=hb_pure)
    fc_pred, fc_imp = _train_head(df, "fc", feature_cols, args.target, params, pure=fc_pure)

    # Combine
    merged = _combine_heads(hb_pred, fc_pred, args.target)

    # Write
    Path(out_preds).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_preds)
    print(f"\n[WRITE] {out_preds}  shape={merged.shape}")
    hb_imp.to_csv(out_imp_hb)
    fc_imp.to_csv(out_imp_fc)
    print(f"[WRITE] {out_imp_hb}")
    print(f"[WRITE] {out_imp_fc}")

    # Eval overall + per-setup_type
    _eval_summary(merged, args.target, "v2 OVERALL (combined heads)")
    for st in ("hb", "fc", "both"):
        sub = merged[merged["setup_type"] == st]
        _eval_summary(sub, args.target, f"v2 setup_type={st}")

    # Per-fold
    for fold in sorted(merged["fold_assigned"].dropna().unique()):
        sub = merged[merged["fold_assigned"] == fold]
        _eval_summary(sub, args.target, f"v2 FOLD {fold}")

    # Top feature importance per head
    if not hb_imp.empty:
        print("\n[HB-head top 10 features by mean gain]")
        print(hb_imp.head(10)[["mean"]].to_string())
    if not fc_imp.empty:
        print("\n[FC-head top 10 features by mean gain]")
        print(fc_imp.head(10)[["mean"]].to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
