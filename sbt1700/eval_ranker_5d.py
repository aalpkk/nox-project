"""Train + evaluate the 5d-potential setup-quality ranker.

This module is the validation gate for the SBT-1700 / 5d ranker.
It is strictly TRAIN-fit + VAL-evaluated; the TEST split (locked behind
sbt1700.splits.load_split allow_test) is never touched here.

Pipeline:
    1. Load TRAIN + VAL via sbt1700.splits.load_split (test stays locked).
    2. Filter rows to those with non-null primary label (mfe_5d_R).
    3. Build feature matrix using locked LGBM_REGRESSION_PARAMS (seed=17),
       extending sbt1700.train_ranker.FEATURE_BLACKLIST with the 5d-label
       and forward-derived columns so they cannot enter the model.
    4. Fit one model per target on TRAIN; predict on VAL.
    5. Emit metrics per spec:
         - Spearman rho(score, target)
         - top decile / top quintile metrics:
             avg/median mfe_5d_R, avg ret_5d_close_R,
             hit_1R_5d, hit_2R_5d, close_positive_5d
         - bottom decile metrics
         - monotonicity (5 score buckets)
         - lift vs raw VAL cohort
         - ticker / date concentration
         - feature importance
    6. Apply acceptance gate.

Outputs (under output/):
    sbt_1700_5d_label_summary_train.csv
    sbt_1700_5d_ranker_validation.csv
    sbt_1700_5d_ranker_preds_val.parquet
    sbt_1700_5d_ranker_feature_importance.csv
    sbt_1700_5d_ranker_report.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

from sbt1700 import splits as splits_mod
from sbt1700.train_ranker import (
    FEATURE_BLACKLIST,
    LGBM_REGRESSION_PARAMS,
    N_ESTIMATORS,
    select_feature_columns,
    assert_no_blacklist_leak,
)


# Extend the legacy blacklist with 5d-label and forward-derived columns
# so they cannot leak into the feature matrix as numeric inputs.
LABEL_BLACKLIST_5D = frozenset({
    # Primary / secondary / helper labels
    "mfe_5d_R", "ret_5d_close_R",
    "mfe_5d_pct", "ret_5d_close_pct",
    "hit_1R_5d", "hit_2R_5d", "close_positive_5d",
    # Forward-window primitives (must not appear as features)
    "fwd_high_5d", "fwd_close_5d", "fwd_n_bars",
})

FEATURE_BLACKLIST_5D: frozenset[str] = FEATURE_BLACKLIST | LABEL_BLACKLIST_5D


PRIMARY_LABEL = "mfe_5d_R"
SECONDARY_LABEL = "ret_5d_close_R"
CLASSIFIER_LABEL = "hit_1R_5d"

LGBM_CLASSIFIER_PARAMS: dict = dict(LGBM_REGRESSION_PARAMS) | dict(
    objective="binary",
    metric="binary_logloss",
)


OUTPUT_DIR = Path("output")
OUT_LABEL_SUMMARY = OUTPUT_DIR / "sbt_1700_5d_label_summary_train.csv"
OUT_VAL_METRICS   = OUTPUT_DIR / "sbt_1700_5d_ranker_validation.csv"
OUT_VAL_PREDS     = OUTPUT_DIR / "sbt_1700_5d_ranker_preds_val.parquet"
OUT_FEAT_IMP      = OUTPUT_DIR / "sbt_1700_5d_ranker_feature_importance.csv"
OUT_REPORT        = OUTPUT_DIR / "sbt_1700_5d_ranker_report.md"


# ── Acceptance gate ───────────────────────────────────────────────────
ACCEPT_RHO_MIN = 0.0           # rho(primary) > 0
ACCEPT_TOP_QUINTILE_MUST_BEAT_COHORT_AVG = True
ACCEPT_TOP_QUINTILE_MUST_BEAT_COHORT_HIT1R = True
ACCEPT_BOTTOM_LT_TOP = True


@dataclass
class TrainedHead:
    name: str
    target: str
    booster: lgb.Booster
    feature_cols: list[str]
    n_train: int
    importance: pd.DataFrame
    is_classifier: bool


# ── Feature matrix ────────────────────────────────────────────────────

def _select_features_5d(df: pd.DataFrame) -> list[str]:
    """Numeric columns minus FEATURE_BLACKLIST_5D, stable order."""
    cols: list[str] = []
    for c in df.columns:
        if c in FEATURE_BLACKLIST_5D:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def _assert_no_leak_5d(feature_cols) -> None:
    leaked = sorted(set(feature_cols) & FEATURE_BLACKLIST_5D)
    if leaked:
        raise RuntimeError(
            f"feature blacklist breach (5d): {leaked}. "
            "Refusing to fit — would learn from labels or forward bars."
        )


# ── Trainer ────────────────────────────────────────────────────────────

def _fit_head(
    train: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    is_classifier: bool,
) -> TrainedHead:
    sub = train.dropna(subset=[target]).copy()
    if sub.empty:
        raise ValueError(f"no train rows with non-null {target!r}")
    X = sub[feature_cols]
    y = sub[target].astype(float if not is_classifier else int)
    params = LGBM_CLASSIFIER_PARAMS if is_classifier else LGBM_REGRESSION_PARAMS
    train_set = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(params=params, train_set=train_set, num_boost_round=N_ESTIMATORS)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    return TrainedHead(
        name=("clf_" if is_classifier else "reg_") + target,
        target=target,
        booster=booster,
        feature_cols=feature_cols,
        n_train=len(sub),
        importance=importance,
        is_classifier=is_classifier,
    )


# ── Metrics ────────────────────────────────────────────────────────────

_LABEL_REPORT_COLS = [
    "mfe_5d_R", "ret_5d_close_R",
    "hit_1R_5d", "hit_2R_5d", "close_positive_5d",
]


def _summarise_cohort(df: pd.DataFrame, label_cols=_LABEL_REPORT_COLS) -> dict:
    out: dict = {"n": int(len(df))}
    for c in label_cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if c.startswith("hit_") or c.startswith("close_positive"):
            out[f"{c}_rate"] = float(s.mean())
        else:
            out[f"{c}_mean"] = float(s.mean())
            out[f"{c}_median"] = float(s.median())
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1).dropna()
    if len(s) < 5:
        return float("nan")
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))


def _slice(df: pd.DataFrame, score: pd.Series, frac_lo: float, frac_hi: float) -> pd.DataFrame:
    """Return rows with score-rank percentile in [frac_lo, frac_hi]."""
    n = len(df)
    if n == 0:
        return df.iloc[0:0]
    order = np.argsort(-score.values, kind="mergesort")  # high score first
    lo = int(round(frac_lo * n))
    hi = int(round(frac_hi * n))
    hi = max(hi, lo + 1)
    sel = order[lo:hi]
    return df.iloc[sel].copy()


def _bucket_metrics(df: pd.DataFrame, score: pd.Series, n_buckets: int = 5) -> pd.DataFrame:
    """Score-bucket monotonicity table. Bucket 1 = highest score."""
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    order = np.argsort(-score.values, kind="mergesort")
    rows: list[dict] = []
    for k in range(n_buckets):
        lo = int(round(k * n / n_buckets))
        hi = int(round((k + 1) * n / n_buckets))
        hi = max(hi, lo + 1)
        sel = order[lo:hi]
        sub = df.iloc[sel]
        row = {"bucket": k + 1, "n": int(len(sub)),
               "score_min": float(score.iloc[sel].min()),
               "score_max": float(score.iloc[sel].max())}
        row.update(_summarise_cohort(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _concentration(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"ticker_top1_share": float("nan"),
                "ticker_top5_share": float("nan"),
                "date_top1_share": float("nan"),
                "date_top5_share": float("nan")}
    n = len(df)
    tk = df["ticker"].value_counts(normalize=True)
    dt = df["date"].astype(str).value_counts(normalize=True)
    return {
        "ticker_top1_share": float(tk.iloc[0]) if not tk.empty else float("nan"),
        "ticker_top5_share": float(tk.head(5).sum()) if not tk.empty else float("nan"),
        "date_top1_share": float(dt.iloc[0]) if not dt.empty else float("nan"),
        "date_top5_share": float(dt.head(5).sum()) if not dt.empty else float("nan"),
        "n_unique_tickers": int(df["ticker"].nunique()),
        "n_unique_dates": int(df["date"].nunique()),
        "n_rows": n,
    }


# ── Evaluator ──────────────────────────────────────────────────────────

def _evaluate_head(
    head: TrainedHead,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, dict, pd.DataFrame, dict, dict]:
    """Run head on val_df, return (preds, summary, buckets, top20, bot10)."""
    score = pd.Series(
        head.booster.predict(val_df[feature_cols]),
        index=val_df.index,
        name=f"score_{head.name}",
    )

    rho_primary = _spearman(score, val_df[PRIMARY_LABEL])
    rho_secondary = _spearman(score, val_df[SECONDARY_LABEL])
    rho_hit1r = _spearman(score, val_df[CLASSIFIER_LABEL].astype(float))

    cohort = _summarise_cohort(val_df)
    top20 = _slice(val_df, score, 0.0, 0.20)
    top10 = _slice(val_df, score, 0.0, 0.10)
    bot20 = _slice(val_df, score, 0.80, 1.0)
    bot10 = _slice(val_df, score, 0.90, 1.0)

    summary = {
        "head": head.name,
        "target": head.target,
        "is_classifier": head.is_classifier,
        "n_train": int(head.n_train),
        "n_val": int(len(val_df)),
        "spearman_rho_primary": rho_primary,
        "spearman_rho_secondary": rho_secondary,
        "spearman_rho_hit1r": rho_hit1r,
    }
    summary.update({f"cohort_{k}": v for k, v in cohort.items()})
    summary.update({f"top20_{k}": v for k, v in _summarise_cohort(top20).items()})
    summary.update({f"top10_{k}": v for k, v in _summarise_cohort(top10).items()})
    summary.update({f"bot20_{k}": v for k, v in _summarise_cohort(bot20).items()})
    summary.update({f"bot10_{k}": v for k, v in _summarise_cohort(bot10).items()})
    summary.update({f"conc_top20_{k}": v for k, v in _concentration(top20).items()})

    buckets = _bucket_metrics(val_df, score, n_buckets=5)
    buckets.insert(0, "head", head.name)

    return score, summary, buckets, _summarise_cohort(top20), _summarise_cohort(bot20)


# ── Acceptance gate ────────────────────────────────────────────────────

def _check_acceptance(summary: dict) -> tuple[str, list[str]]:
    """Return (verdict, reasons[]). verdict ∈ {PASS, FAIL}."""
    fails: list[str] = []

    rho = summary.get("spearman_rho_primary")
    if rho is None or not np.isfinite(rho) or rho <= ACCEPT_RHO_MIN:
        fails.append(f"rho_primary={rho!r} not > {ACCEPT_RHO_MIN}")

    cohort_avg = summary.get("cohort_mfe_5d_R_mean")
    top20_avg = summary.get("top20_mfe_5d_R_mean")
    if (ACCEPT_TOP_QUINTILE_MUST_BEAT_COHORT_AVG
            and (cohort_avg is None or top20_avg is None
                 or top20_avg <= cohort_avg)):
        fails.append(
            f"top20_mfe_5d_R_mean={top20_avg!r} not > cohort {cohort_avg!r}"
        )

    cohort_hit = summary.get("cohort_hit_1R_5d_rate")
    top20_hit = summary.get("top20_hit_1R_5d_rate")
    if (ACCEPT_TOP_QUINTILE_MUST_BEAT_COHORT_HIT1R
            and (cohort_hit is None or top20_hit is None
                 or top20_hit <= cohort_hit)):
        fails.append(
            f"top20_hit_1R_5d_rate={top20_hit!r} not > cohort {cohort_hit!r}"
        )

    bot_avg = summary.get("bot20_mfe_5d_R_mean")
    if (ACCEPT_BOTTOM_LT_TOP and (bot_avg is None or top20_avg is None
                                  or bot_avg >= top20_avg)):
        fails.append(
            f"bot20_mfe_5d_R_mean={bot_avg!r} not < top20 {top20_avg!r}"
        )

    return ("PASS" if not fails else "FAIL", fails)


# ── Report writer ──────────────────────────────────────────────────────

def _write_report(
    out_path: Path,
    train_summary: dict,
    val_summary_primary: dict,
    val_summary_secondary: dict,
    val_summary_clf: dict | None,
    buckets_primary: pd.DataFrame,
    importance_primary: pd.DataFrame,
    verdict: str,
    fails: list[str],
    feature_cols: list[str],
    extras: dict,
) -> None:
    L: list[str] = []
    L.append("# SBT-1700 / 5d-Potential Setup-Quality Ranker — Validation Report")
    L.append("")
    L.append(f"Dataset: `{extras.get('dataset_path')}`  ")
    L.append(f"Schema:  `{extras.get('schema_version')}`  ")
    L.append(f"Built:   `{extras.get('built_at')}`  ")
    L.append(f"Source:  `{extras.get('dataset_source')}`")
    L.append("")
    L.append("## Splits (locked; test untouched)")
    L.append("")
    L.append("| phase | start | end | n_rows |")
    L.append("|---|---|---|---|")
    for ph, info in extras.get("splits", {}).items():
        nr = info.get("n_rows", "—")
        L.append(f"| {ph} | {info['start']} | {info['end']} | {nr} |")
    L.append("")
    L.append("## TRAIN cohort label summary")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    for k, v in train_summary.items():
        L.append(f"| {k} | {v} |")
    L.append("")

    def _sec(title: str, summary: dict | None):
        if summary is None:
            return
        L.append(f"## VAL — head `{summary.get('head')}` (target: `{summary.get('target')}`)")
        L.append("")
        rho = summary.get("spearman_rho_primary")
        rho_s = summary.get("spearman_rho_secondary")
        rho_h = summary.get("spearman_rho_hit1r")
        L.append(f"- Spearman rho(score, mfe_5d_R) = **{rho:.4f}**")
        L.append(f"- Spearman rho(score, ret_5d_close_R) = {rho_s:.4f}")
        L.append(f"- Spearman rho(score, hit_1R_5d) = {rho_h:.4f}")
        L.append(f"- n_train = {summary.get('n_train')}, n_val = {summary.get('n_val')}")
        L.append("")
        L.append("| cohort | n | mfe_5d_R mean | mfe_5d_R median | ret_5d_close_R mean | hit_1R | hit_2R | close+ |")
        L.append("|---|---|---|---|---|---|---|---|")
        for tag in ("cohort", "top20", "top10", "bot20", "bot10"):
            n = summary.get(f"{tag}_n", "")
            r = lambda key, default="—": summary.get(f"{tag}_{key}", default)
            L.append(
                f"| {tag} | {n} | "
                f"{r('mfe_5d_R_mean'):} | {r('mfe_5d_R_median')} | "
                f"{r('ret_5d_close_R_mean')} | "
                f"{r('hit_1R_5d_rate')} | {r('hit_2R_5d_rate')} | "
                f"{r('close_positive_5d_rate')} |"
            )
        L.append("")
        c_top20_t1 = summary.get("conc_top20_ticker_top1_share")
        c_top20_t5 = summary.get("conc_top20_ticker_top5_share")
        c_top20_d1 = summary.get("conc_top20_date_top1_share")
        c_top20_d5 = summary.get("conc_top20_date_top5_share")
        L.append(f"- top20 concentration: ticker top1 {c_top20_t1}, top5 {c_top20_t5}; "
                 f"date top1 {c_top20_d1}, top5 {c_top20_d5}")
        L.append("")

    _sec("primary regression", val_summary_primary)
    _sec("secondary regression", val_summary_secondary)
    _sec("classifier (optional)", val_summary_clf)

    L.append("## Score-bucket monotonicity (primary head)")
    L.append("")
    if not buckets_primary.empty:
        L.append(buckets_primary.to_markdown(index=False, floatfmt=".4f"))
        L.append("")

    L.append("## Top 20 features by gain (primary head)")
    L.append("")
    L.append(importance_primary.head(20).to_markdown(index=False, floatfmt=".2f"))
    L.append("")

    L.append("## Acceptance gate")
    L.append("")
    L.append(f"**Verdict: {verdict}**")
    if fails:
        L.append("")
        L.append("Failures:")
        for f in fails:
            L.append(f"- {f}")
    L.append("")
    L.append("Rules applied:")
    L.append(f"- rho(score, mfe_5d_R) > {ACCEPT_RHO_MIN}")
    L.append("- top quintile mfe_5d_R mean > cohort mean")
    L.append("- top quintile hit_1R_5d rate > cohort rate")
    L.append("- bottom quintile mfe_5d_R mean < top quintile mean")
    L.append("")
    L.append(f"**Test split is LOCKED. This run did not load TEST rows.**")
    out_path.write_text("\n".join(L))


# ── CLI driver ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train + evaluate the 5d-potential SBT-1700 ranker (TRAIN+VAL)."
    )
    ap.add_argument("--dataset", type=Path,
                    default=Path("output/sbt_1700_dataset_5d_intraday_v1.parquet"))
    ap.add_argument("--with-classifier", action="store_true",
                    help="also fit a binary head on hit_1R_5d.")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset missing: {args.dataset}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[eval-5d] dataset = {args.dataset}")
    train = splits_mod.load_split(args.dataset, "train")
    val = splits_mod.load_split(args.dataset, "validation")
    print(f"[eval-5d] TRAIN rows = {len(train)}, VAL rows = {len(val)}")

    # Drop rows where the primary label is NaN (tail-of-panel etc.).
    train = train.dropna(subset=[PRIMARY_LABEL]).reset_index(drop=True)
    val = val.dropna(subset=[PRIMARY_LABEL]).reset_index(drop=True)
    print(f"[eval-5d] after dropna({PRIMARY_LABEL}): "
          f"TRAIN={len(train)}, VAL={len(val)}")

    feature_cols = _select_features_5d(train)
    _assert_no_leak_5d(feature_cols)
    print(f"[eval-5d] {len(feature_cols)} features (blacklist guarded)")

    # ── TRAIN cohort label summary ────────────────────────────────────
    train_summary = _summarise_cohort(train)
    pd.DataFrame([train_summary]).to_csv(OUT_LABEL_SUMMARY, index=False)
    print(f"[eval-5d] wrote {OUT_LABEL_SUMMARY}")

    # ── Fit heads ─────────────────────────────────────────────────────
    print("[eval-5d] fitting primary head (mfe_5d_R)…")
    head_primary = _fit_head(train, PRIMARY_LABEL, feature_cols, is_classifier=False)
    print("[eval-5d] fitting secondary head (ret_5d_close_R)…")
    head_secondary = _fit_head(train, SECONDARY_LABEL, feature_cols, is_classifier=False)
    head_clf = None
    if args.with_classifier:
        print("[eval-5d] fitting classifier head (hit_1R_5d)…")
        head_clf = _fit_head(train, CLASSIFIER_LABEL, feature_cols, is_classifier=True)

    # ── Evaluate on VAL ───────────────────────────────────────────────
    print("[eval-5d] scoring VAL with all heads…")
    score_p, sum_p, buckets_p, _, _ = _evaluate_head(head_primary, val, feature_cols)
    score_s, sum_s, _, _, _ = _evaluate_head(head_secondary, val, feature_cols)
    score_c, sum_c = (None, None)
    if head_clf is not None:
        score_c, sum_c, _, _, _ = _evaluate_head(head_clf, val, feature_cols)

    # ── Persist preds + per-head metrics CSV ──────────────────────────
    preds = val[["ticker", "date", PRIMARY_LABEL, SECONDARY_LABEL,
                 CLASSIFIER_LABEL, "hit_2R_5d", "close_positive_5d",
                 "intraday_coverage"]].copy()
    preds["score_primary"] = score_p.values
    preds["score_secondary"] = score_s.values
    if score_c is not None:
        preds["score_classifier"] = score_c.values
    preds.to_parquet(OUT_VAL_PREDS, index=False)
    print(f"[eval-5d] wrote {OUT_VAL_PREDS}")

    metrics_rows = [sum_p, sum_s] + ([sum_c] if sum_c else [])
    pd.DataFrame(metrics_rows).to_csv(OUT_VAL_METRICS, index=False)
    print(f"[eval-5d] wrote {OUT_VAL_METRICS}")

    head_primary.importance.to_csv(OUT_FEAT_IMP, index=False)
    print(f"[eval-5d] wrote {OUT_FEAT_IMP}")

    # ── Acceptance gate (on primary head) ─────────────────────────────
    verdict, fails = _check_acceptance(sum_p)
    print(f"[eval-5d] acceptance: {verdict}")
    for f in fails:
        print(f"[eval-5d]   FAIL: {f}")

    # ── Report ────────────────────────────────────────────────────────
    meta_path = args.dataset.with_suffix(".meta.json")
    extras: dict = {"dataset_path": str(args.dataset)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        extras.update({
            "schema_version": meta.get("schema_version"),
            "built_at": meta.get("built_at"),
            "dataset_source": meta.get("dataset_source"),
        })
    counts = splits_mod.split_counts(args.dataset)
    extras["splits"] = {
        "train": {"start": str(splits_mod.TRAIN_START.date()),
                  "end": str(splits_mod.TRAIN_END.date()),
                  "n_rows": counts["train"]},
        "validation": {"start": str(splits_mod.VAL_START.date()),
                       "end": str(splits_mod.VAL_END.date()),
                       "n_rows": counts["validation"]},
        "test": {"start": str(splits_mod.TEST_START.date()),
                 "end": str(splits_mod.TEST_END.date()),
                 "n_rows": "LOCKED"},
    }

    _write_report(
        OUT_REPORT,
        train_summary=train_summary,
        val_summary_primary=sum_p,
        val_summary_secondary=sum_s,
        val_summary_clf=sum_c,
        buckets_primary=buckets_p,
        importance_primary=head_primary.importance,
        verdict=verdict,
        fails=fails,
        feature_cols=feature_cols,
        extras=extras,
    )
    print(f"[eval-5d] wrote {OUT_REPORT}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
