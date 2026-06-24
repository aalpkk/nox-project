"""Ranking Lab v0 — Stage 5b de-scaled LightGBM rerun.

Drops absolute-scale predictors (entry_ref, atr, atr_14_daily, liquidity_proxy)
that Stage 5 importance analysis flagged as drivers of small/mid-cap ticker
concentration. Adds date-relative rank features (rank-pct within each
signal_date over the panel candidate set) and winsorizes the remaining raw
numerics at train-fold p1/p99 (caps applied to val/test). Categorical features
unchanged.

Evaluation reports BOTH raw top-K and ticker-capped top-K diagnostics. Cap is
eval-only (max 3 rows per ticker across pooled top-K, max 1 per ticker per
signal_date); the model itself does not see ticker.

Inputs:
  - output/ranking_lab_features_v0.parquet

Outputs:
  - output/ranking_lab_lgbm_predictions_v0_stage5b.parquet
  - output/ranking_lab_lgbm_metrics_v0_stage5b.csv
  - output/ranking_lab_lgbm_feature_importance_v0_stage5b.csv
  - output/ranking_lab_lgbm_calibration_v0_stage5b.csv
  - output/ranking_lab_lgbm_report_v0_stage5b.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

FEAT_PATH = OUT / "ranking_lab_features_v0.parquet"

DST_PRED = OUT / "ranking_lab_lgbm_predictions_v0_stage5b.parquet"
DST_METRICS = OUT / "ranking_lab_lgbm_metrics_v0_stage5b.csv"
DST_IMP = OUT / "ranking_lab_lgbm_feature_importance_v0_stage5b.csv"
DST_CALIB = OUT / "ranking_lab_lgbm_calibration_v0_stage5b.csv"
DST_MD = OUT / "ranking_lab_lgbm_report_v0_stage5b.md"

FOLDS = [
    ("F1", "2023-01-13", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("F2", "2023-01-13", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("F3", "2023-01-13", "2025-06-30", "2025-07-01", "2025-12-31"),
    ("F4", "2023-01-13", "2025-12-31", "2026-01-01", "2026-05-12"),
    # F5 = FORWARD scoring fold (added 2026-06-04): train on ALL labeled history
    # through 2026-05-12, predict the fresh post-cutoff events (2026-05-13 → latest)
    # strictly out-of-sample. Forward test labels are NaN (windows not closed) so
    # these rows get predictions but are excluded from eval metrics by design.
    # F1-F4 reproduce identically; F5 is purely additive (no lookahead).
    ("F5", "2023-01-13", "2026-05-12", "2026-05-13", "2026-06-30"),
]

CORE_TARGETS = [
    ("y3_8", "mfe_pct_3d", 0.08),
    ("y5_10", "mfe_pct_5d", 0.10),
    ("y10_15", "mfe_pct_10d", 0.15),
    ("y30_30", "mfe_pct_30d", 0.30),
    ("y30_50", "mfe_pct_30d", 0.50),
    ("y60_80", "mfe_pct_60d", 0.80),
]
AUDIT_TARGETS = [
    ("y30_80", "mfe_pct_30d", 0.80),
    ("y60_100", "mfe_pct_60d", 1.00),
]
ALL_TARGETS = CORE_TARGETS + AUDIT_TARGETS

# Stage 4 rule baseline lifts and Stage 5 raw ML lifts (for the comparison table)
RULE_BASELINE_LIFT = {
    "y3_8": 1.3849, "y5_10": 1.3116, "y10_15": 1.2084,
    "y30_30": 1.0563, "y30_50": 1.0666, "y60_80": 1.0265,
    "y30_80": 0.9866, "y60_100": 0.9104,
}
STAGE5_ML_LIFT = {
    "y3_8": 2.4527, "y5_10": 1.9738, "y10_15": 1.7203,
    "y30_30": 1.1743, "y30_50": 1.4801, "y60_80": 1.3370,
    "y30_80": 2.0118, "y60_100": 1.3258,
}
STAGE5_TOP20_MAX_TICKER = {
    "y3_8": 0.20, "y5_10": 0.40, "y10_15": 0.40,
    "y30_30": 0.40, "y30_50": 0.30, "y60_80": 0.40,
    "y30_80": 0.45, "y60_100": 0.85,
}

# Predictor whitelist for Stage 5b
# RAW numeric: dropped entry_ref, atr, atr_14_daily, liquidity_proxy (absolute scale).
# Kept normalized only. xu100_atr_pct is a market-wide regime scalar (varies only by date) — kept.
NUMERIC_FEATURES_RAW = [
    "atr_pct", "atr_ratio_daily", "xu100_atr_pct",
    "vol_z_20", "ret_1d", "ret_5d", "gap_pct",
    "price_vs_20d_high", "price_vs_60d_high",
    "event_multiplicity", "n_concurrent_sources",
    "n_concurrent_families", "n_concurrent_timeframes",
]

# Date-relative rank features (computed below from the panel; rank-pct within signal_date)
RANK_BASES = [
    "liquidity_proxy",   # the bare absolute is dropped but its date-rank is fine
    "atr_pct",
    "vol_z_20",
    "ret_1d",
    "ret_5d",
    "price_vs_20d_high",
    "price_vs_60d_high",
]
RANK_FEATURES = [f"{b}_rank_pct_by_date" for b in RANK_BASES]

# Audit-only (computed but EXCLUDED from features by ONAY): entry_ref_rank_pct_by_date
AUDIT_RANK_BASES = ["entry_ref"]
AUDIT_RANK_FEATURES = [f"{b}_rank_pct_by_date" for b in AUDIT_RANK_BASES]

NUMERIC_FEATURES = NUMERIC_FEATURES_RAW + RANK_FEATURES
CATEGORICAL_FEATURES = ["source", "family", "setup_label", "signal_state", "timeframe"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DROPPED_FROM_STAGE5 = ["entry_ref", "atr", "atr_14_daily", "liquidity_proxy"]

FORBIDDEN_PATTERNS = (
    "mfe_pct_", "realized_pct_", "src_native_payload",
    "fold", "ticker", "signal_date", "side",
)

# LGBM params (same predeclared values as Stage 5; not tuned)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
N_ESTIMATORS_MAX = 500
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.20

# Winsorize bounds (train-fold p1/p99 on the raw numerics only; rank features in [0,1])
WINSORIZE_COLS = [c for c in NUMERIC_FEATURES_RAW if c != "xu100_atr_pct"]  # market scalar — skip
# Note: xu100_atr_pct is bounded by index ATR/close; no extreme per-stock outliers.

# Ticker cap (eval only): no model knowledge of ticker
EVAL_CAP_MAX_PER_TICKER = 3   # in pooled top-K
EVAL_CAP_MAX_PER_TICKER_DATE = 1


def _add_rank_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Add per-date rank-pct features. Within each signal_date, rank candidates and
    normalize to [0,1] (method=average so ties get mid-rank). NaN inputs → NaN ranks.

    LEAKAGE NOTE: rank uses ONLY same-signal_date rows — no forward or past universe
    information leaks across dates. Equivalent to live-scoring where the same-date
    candidate set is known by close.
    """
    out = feats.copy()
    grp = out.groupby("signal_date", sort=False)
    for base in RANK_BASES + AUDIT_RANK_BASES:
        if base not in out.columns:
            raise ValueError(f"rank base column missing: {base}")
        col = f"{base}_rank_pct_by_date"
        out[col] = grp[base].rank(pct=True, method="average")
    return out


def _winsorize_caps(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute p1/p99 caps from train fold for WINSORIZE_COLS."""
    caps: dict[str, tuple[float, float]] = {}
    for c in WINSORIZE_COLS:
        s = pd.to_numeric(train_df[c], errors="coerce")
        lo = float(np.nanpercentile(s.values, 1.0))
        hi = float(np.nanpercentile(s.values, 99.0))
        caps[c] = (lo, hi)
    return caps


def _apply_winsorize(df: pd.DataFrame, caps: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip numeric columns to provided caps. Leaves NaN alone."""
    out = df.copy()
    for c, (lo, hi) in caps.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=lo, upper=hi)
    return out


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X["signal_state"] = X["signal_state"].fillna("untagged")
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _split_train_val(train_df: pd.DataFrame, val_frac: float = VAL_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values("signal_date").reset_index(drop=True)
    n = len(train_df)
    cut = int(n * (1.0 - val_frac))
    tr = train_df.iloc[:cut].reset_index(drop=True)
    va = train_df.iloc[cut:].reset_index(drop=True)
    return tr, va


def _topk_block(df: pd.DataFrame, score_col: str, target_col: str, threshold: float, k: int) -> dict:
    s = df.sort_values(score_col, ascending=False).head(k)
    mfe = pd.to_numeric(s[target_col], errors="coerce")
    valid = s[s[target_col].notna()]
    pos_total = int((pd.to_numeric(df[target_col], errors="coerce") >= threshold).sum())
    n_pos = int((mfe >= threshold).sum())
    cap = float(n_pos) / pos_total if pos_total > 0 else float("nan")
    prec = float(n_pos) / max(len(valid), 1) if len(valid) > 0 else float("nan")
    return {
        "k": k,
        "n_kept": len(s),
        "topK_pos": n_pos,
        "precision": prec,
        "capture": cap,
        "mean_mfe": float(mfe.mean()) if mfe.notna().any() else float("nan"),
        "p50_mfe": float(mfe.median()) if mfe.notna().any() else float("nan"),
        "p90_mfe": float(mfe.quantile(0.90)) if mfe.notna().any() else float("nan"),
        "max_mfe": float(mfe.max()) if mfe.notna().any() else float("nan"),
    }


def _ticker_capped_topk(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Walk sorted-by-score rows; accept until per-ticker and per-(ticker,date) caps fill.

    Caps: <= EVAL_CAP_MAX_PER_TICKER total per ticker AND
          <= EVAL_CAP_MAX_PER_TICKER_DATE per (ticker, signal_date).
    Returns at most k rows (fewer only if dataframe is exhausted)."""
    sub = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    per_ticker: dict[str, int] = {}
    per_tkr_date: dict[tuple[str, object], int] = {}
    out_idx: list[int] = []
    for i, row in enumerate(sub.itertuples(index=False)):
        tkr = getattr(row, "ticker")
        dt = getattr(row, "signal_date")
        if per_ticker.get(tkr, 0) >= EVAL_CAP_MAX_PER_TICKER:
            continue
        key = (tkr, dt)
        if per_tkr_date.get(key, 0) >= EVAL_CAP_MAX_PER_TICKER_DATE:
            continue
        out_idx.append(i)
        per_ticker[tkr] = per_ticker.get(tkr, 0) + 1
        per_tkr_date[key] = per_tkr_date.get(key, 0) + 1
        if len(out_idx) >= k:
            break
    return sub.iloc[out_idx].reset_index(drop=True)


def _capped_topk_block(df: pd.DataFrame, score_col: str, target_col: str, threshold: float, k: int) -> dict:
    top = _ticker_capped_topk(df, score_col, k)
    mfe = pd.to_numeric(top[target_col], errors="coerce")
    valid = top[top[target_col].notna()]
    pos_total = int((pd.to_numeric(df[target_col], errors="coerce") >= threshold).sum())
    n_pos = int((mfe >= threshold).sum())
    cap = float(n_pos) / pos_total if pos_total > 0 else float("nan")
    prec = float(n_pos) / max(len(valid), 1) if len(valid) > 0 else float("nan")
    tk = top["ticker"].value_counts()
    return {
        "k": k,
        "n_kept": len(top),
        "topK_pos": n_pos,
        "precision": prec,
        "capture": cap,
        "mean_mfe": float(mfe.mean()) if mfe.notna().any() else float("nan"),
        "unique_tickers": int(tk.size),
        "max_ticker_share": float(tk.iloc[0] / len(top)) if len(top) > 0 else float("nan"),
    }


def _concentration(df: pd.DataFrame, score_col: str, k: int = 20) -> dict:
    s = df.sort_values(score_col, ascending=False).head(k)
    n = len(s)
    if n == 0:
        return {"top20_n": 0,
                "top20_unique_tickers": 0, "top20_max_ticker_share": float("nan"),
                "top20_unique_dates": 0, "top20_max_date_share": float("nan"),
                "top20_top_source_family": "—", "top20_max_source_family_share": float("nan"),
                "top20_top_ticker": "—"}
    tk = s["ticker"].value_counts()
    dt = s["signal_date"].astype(str).value_counts()
    sf = (s["source"].astype(str) + "/" + s["family"].astype(str)).value_counts()
    return {
        "top20_n": n,
        "top20_unique_tickers": int(tk.size),
        "top20_top_ticker": str(tk.index[0]),
        "top20_max_ticker_share": float(tk.iloc[0] / n),
        "top20_unique_dates": int(dt.size),
        "top20_max_date_share": float(dt.iloc[0] / n),
        "top20_top_source_family": str(sf.index[0]),
        "top20_max_source_family_share": float(sf.iloc[0] / n),
    }


def _train_one(train_df: pd.DataFrame, test_df: pd.DataFrame,
                label_col: str, threshold: float,
                caps: dict[str, tuple[float, float]]
                ) -> tuple[np.ndarray, pd.DataFrame, dict, lgb.Booster]:
    """Train + predict for one fold + one target with pre-applied winsorize caps."""
    train_df = train_df.copy()
    train_df["_y"] = (pd.to_numeric(train_df[label_col], errors="coerce") >= threshold).astype("Int64")
    n_pre = len(train_df)
    mask_lab = train_df[label_col].notna()
    train_df = train_df.loc[mask_lab].reset_index(drop=True)
    n_post = len(train_df)

    tr_df, va_df = _split_train_val(train_df)
    # Caps already applied at the fold level before this call
    X_tr = _build_X(tr_df)
    y_tr = tr_df["_y"].astype(int).values
    X_va = _build_X(va_df)
    y_va = va_df["_y"].astype(int).values
    X_te = _build_X(test_df)

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    val_set = lgb.Dataset(X_va, label=y_va, categorical_feature=CATEGORICAL_FEATURES, reference=train_set, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        params=LGBM_PARAMS,
        train_set=train_set,
        num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )

    test_preds = booster.predict(X_te, num_iteration=booster.best_iteration)
    fit_info = {
        "n_train_rows": n_post,
        "n_train_dropped_nan": n_pre - n_post,
        "n_train_used": len(tr_df),
        "n_val": len(va_df),
        "best_iteration": int(booster.best_iteration) if booster.best_iteration else int(N_ESTIMATORS_MAX),
        "train_pos": int((y_tr == 1).sum()),
        "val_pos": int((y_va == 1).sum()),
    }

    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    feat_names = booster.feature_name()
    imp = pd.DataFrame({
        "feature": feat_names,
        "gain": gain.astype(float),
        "split": split.astype(int),
    })
    return test_preds, imp, fit_info, booster


def _metrics_block(test_df: pd.DataFrame, preds: np.ndarray, label_col: str, threshold: float,
                    fold: str, target: str) -> tuple[dict, pd.DataFrame, dict, dict]:
    """Compute per-(fold,target) RAW metrics + capped metrics + calibration deciles."""
    df = test_df.copy()
    df["_score"] = preds
    y_true_cont = pd.to_numeric(df[label_col], errors="coerce")
    df = df[y_true_cont.notna()].reset_index(drop=True)
    y_true_cont = pd.to_numeric(df[label_col], errors="coerce")
    y_true = (y_true_cont >= threshold).astype(int).values
    scores = df["_score"].values
    n = len(df)
    n_pos = int(y_true.sum())
    base_rate = n_pos / n if n > 0 else float("nan")

    auc = roc_auc_score(y_true, scores) if 0 < n_pos < n else float("nan")
    pr_auc = average_precision_score(y_true, scores) if 0 < n_pos < n else float("nan")
    brier = brier_score_loss(y_true, scores) if n > 0 else float("nan")
    try:
        ll = log_loss(y_true, np.clip(scores, 1e-7, 1 - 1e-7))
    except Exception:
        ll = float("nan")

    k_dec = max(int(np.ceil(n * 0.10)), 1) if n > 0 else 0
    k_5 = max(int(np.ceil(n * 0.05)), 1) if n > 0 else 0
    sorted_df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    top_dec = sorted_df.head(k_dec)
    top_5 = sorted_df.head(k_5)
    top_dec_pos = int((pd.to_numeric(top_dec[label_col], errors="coerce") >= threshold).sum())
    top_5_pos = int((pd.to_numeric(top_5[label_col], errors="coerce") >= threshold).sum())
    prec_dec = top_dec_pos / k_dec if k_dec > 0 else float("nan")
    prec_5 = top_5_pos / k_5 if k_5 > 0 else float("nan")
    lift_dec = prec_dec / base_rate if base_rate and not np.isnan(base_rate) and base_rate > 0 else float("nan")
    lift_5 = prec_5 / base_rate if base_rate and not np.isnan(base_rate) and base_rate > 0 else float("nan")

    cap20 = _topk_block(df, "_score", label_col, threshold, 20)
    cap50 = _topk_block(df, "_score", label_col, threshold, 50)
    cap100 = _topk_block(df, "_score", label_col, threshold, 100)

    # Ticker-capped variants (eval only)
    cap_t20 = _capped_topk_block(df, "_score", label_col, threshold, 20)
    cap_t50 = _capped_topk_block(df, "_score", label_col, threshold, 50)
    cap_t100 = _capped_topk_block(df, "_score", label_col, threshold, 100)

    top50 = sorted_df.head(50)
    mfe30 = pd.to_numeric(top50["mfe_pct_30d"], errors="coerce")
    mfe60 = pd.to_numeric(top50["mfe_pct_60d"], errors="coerce")
    conc = _concentration(df, "_score", k=20)

    # Calibration deciles
    df2 = df.copy()
    if n > 0:
        df2["_decile"] = pd.qcut(df2["_score"].rank(method="first"), q=10, labels=False, duplicates="drop")
    else:
        df2["_decile"] = np.nan
    calib_rows = []
    for d, grp in df2.groupby("_decile"):
        y_g = (pd.to_numeric(grp[label_col], errors="coerce") >= threshold).astype(int)
        s_g = grp["_score"]
        calib_rows.append({
            "fold": fold, "target": target,
            "decile": int(d) if not np.isnan(d) else None,
            "n": int(len(grp)),
            "mean_score": float(s_g.mean()) if len(grp) else float("nan"),
            "actual_rate": float(y_g.mean()) if len(grp) else float("nan"),
        })
    calib_df = pd.DataFrame(calib_rows)

    row = {
        "fold": fold, "target": target,
        "label_col": label_col, "threshold": threshold,
        "n_test": n, "n_positive": n_pos, "base_rate": base_rate,
        "auc": auc, "pr_auc": pr_auc, "brier": brier, "logloss": ll,
        "top_decile_k": k_dec,
        "top_decile_precision": prec_dec,
        "top_decile_lift": lift_dec,
        "top_5pct_k": k_5,
        "top_5pct_precision": prec_5,
        "top_5pct_lift": lift_5,
        "precision_top20": cap20["precision"], "capture_top20": cap20["capture"],
        "precision_top50": cap50["precision"], "capture_top50": cap50["capture"],
        "precision_top100": cap100["precision"], "capture_top100": cap100["capture"],
        # Capped variants
        "capped_precision_top20": cap_t20["precision"], "capped_capture_top20": cap_t20["capture"],
        "capped_precision_top50": cap_t50["precision"], "capped_capture_top50": cap_t50["capture"],
        "capped_precision_top100": cap_t100["precision"], "capped_capture_top100": cap_t100["capture"],
        "capped_top50_unique_tickers": cap_t50["unique_tickers"],
        "capped_top50_max_ticker_share": cap_t50["max_ticker_share"],
        "top50_mfe30_mean": float(mfe30.mean()) if mfe30.notna().any() else float("nan"),
        "top50_mfe60_mean": float(mfe60.mean()) if mfe60.notna().any() else float("nan"),
        "top20_unique_tickers": conc["top20_unique_tickers"],
        "top20_top_ticker": conc["top20_top_ticker"],
        "top20_max_ticker_share": conc["top20_max_ticker_share"],
        "top20_unique_dates": conc["top20_unique_dates"],
        "top20_max_date_share": conc["top20_max_date_share"],
        "top20_top_source_family": conc["top20_top_source_family"],
        "top20_max_source_family_share": conc["top20_max_source_family_share"],
    }
    return row, calib_df, conc, {"raw_top50": cap50, "capped_top50": cap_t50}


def main() -> None:
    print(f"[lgbm-5b] loading features: {FEAT_PATH}")
    feats = pd.read_parquet(FEAT_PATH)
    feats["signal_date"] = pd.to_datetime(feats["signal_date"]).dt.normalize()
    print(f"  rows={len(feats):,}  cols={feats.shape[1]}  tickers={feats.ticker.nunique()}")

    # Verify rank-base columns exist
    for b in RANK_BASES + AUDIT_RANK_BASES:
        assert b in feats.columns, f"missing rank-base column: {b}"

    print("[lgbm-5b] computing per-date rank-pct features...")
    feats = _add_rank_features(feats)
    print(f"  added: {RANK_FEATURES + AUDIT_RANK_FEATURES}")
    # Sanity: ranks bounded
    for c in RANK_FEATURES + AUDIT_RANK_FEATURES:
        s = pd.to_numeric(feats[c], errors="coerce")
        lo = float(s.min()); hi = float(s.max()); nz = int(s.notna().sum())
        print(f"    {c}: min={lo:.4f} max={hi:.4f} non-null={nz:,}")
        assert lo >= 0.0 and hi <= 1.0, f"rank out of [0,1]: {c}"

    # Drop-from-feature-set sanity
    print(f"[lgbm-5b] dropped from Stage 5 ML features (raw absolute scale): {DROPPED_FROM_STAGE5}")
    print(f"[lgbm-5b] audit-only rank features (NOT in feature whitelist): {AUDIT_RANK_FEATURES}")

    # Forbidden patterns check
    for fc in FEATURE_COLS:
        for pat in FORBIDDEN_PATTERNS:
            assert not fc.startswith(pat), f"forbidden pattern {pat} in FEATURE_COLS: {fc}"
    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    assert not missing, f"missing feature columns: {missing}"

    # Audit head sparsity gate
    pos_counts_audit: dict[str, dict[str, int]] = {}
    for tname, lcol, thr in AUDIT_TARGETS:
        pos_counts_audit[tname] = {}
        for fold_name, ts, te, vs, ve in FOLDS:
            # Forward fold F5 has no matured long-horizon labels (windows still open);
            # its structural sparsity must NOT gate audit-head training. Skip it here.
            if fold_name == "F5":
                continue
            test = feats[(feats["signal_date"] >= pd.Timestamp(vs)) & (feats["signal_date"] <= pd.Timestamp(ve))]
            pos = int((pd.to_numeric(test[lcol], errors="coerce") >= thr).sum())
            pos_counts_audit[tname][fold_name] = pos
    audit_to_train: list[str] = []
    for tname in pos_counts_audit:
        min_pos = min(pos_counts_audit[tname].values())
        if min_pos >= 100:
            audit_to_train.append(tname)
            print(f"  audit head {tname}: min fold pos={min_pos} >= 100 → train")
        else:
            print(f"  audit head {tname}: min fold pos={min_pos} < 100 → SKIP")

    targets_to_run = [t for t in ALL_TARGETS if t[0] in {x[0] for x in CORE_TARGETS} or t[0] in audit_to_train]

    all_rows: list[dict] = []
    all_calib: list[pd.DataFrame] = []
    all_imp: list[pd.DataFrame] = []
    all_preds: list[pd.DataFrame] = []
    all_caps: list[dict] = []

    for fold_name, ts, te, vs, ve in FOLDS:
        ts_d, te_d = pd.Timestamp(ts), pd.Timestamp(te)
        vs_d, ve_d = pd.Timestamp(vs), pd.Timestamp(ve)
        train_raw = feats[(feats["signal_date"] >= ts_d) & (feats["signal_date"] <= te_d)].copy()
        test_raw = feats[(feats["signal_date"] >= vs_d) & (feats["signal_date"] <= ve_d)].copy()
        # Compute winsorize caps from TRAIN ONLY then apply to train/test
        caps = _winsorize_caps(train_raw)
        train = _apply_winsorize(train_raw, caps)
        test = _apply_winsorize(test_raw, caps)
        cap_row = {"fold": fold_name}
        for c, (lo, hi) in caps.items():
            cap_row[f"{c}_p1"] = lo
            cap_row[f"{c}_p99"] = hi
        all_caps.append(cap_row)
        print(f"\n[lgbm-5b][{fold_name}] train={len(train):,} test={len(test):,}  "
              f"train_range={ts_d.date()}→{te_d.date()}  test_range={vs_d.date()}→{ve_d.date()}")
        # Print a few sample caps for visibility
        sample = ["atr_pct", "ret_5d", "vol_z_20"]
        for c in sample:
            if c in caps:
                lo, hi = caps[c]
                print(f"    cap[{c}]: p1={lo:.4f}  p99={hi:.4f}")

        for tname, lcol, thr in targets_to_run:
            print(f"  [lgbm-5b][{fold_name}][{tname}] training (label={lcol} >= {thr}) ...")
            preds, imp, fit_info, booster = _train_one(train, test, lcol, thr, caps)
            row, calib_df, _, _ = _metrics_block(test, preds, lcol, thr, fold_name, tname)
            row.update({f"fit_{k}": v for k, v in fit_info.items()})
            all_rows.append(row)
            calib_df["fold"] = fold_name
            calib_df["target"] = tname
            all_calib.append(calib_df)
            imp_df = imp.copy()
            imp_df["fold"] = fold_name
            imp_df["target"] = tname
            all_imp.append(imp_df)
            pred_df = test[["ticker", "signal_date", "source", "family"]].copy()
            pred_df["fold"] = fold_name
            pred_df["target"] = tname
            pred_df["score"] = preds
            pred_df["label_col"] = lcol
            pred_df["threshold"] = thr
            pred_df["mfe_pct_3d"] = pd.to_numeric(test["mfe_pct_3d"], errors="coerce")
            pred_df["mfe_pct_5d"] = pd.to_numeric(test["mfe_pct_5d"], errors="coerce")
            pred_df["mfe_pct_10d"] = pd.to_numeric(test["mfe_pct_10d"], errors="coerce")
            pred_df["mfe_pct_30d"] = pd.to_numeric(test["mfe_pct_30d"], errors="coerce")
            pred_df["mfe_pct_60d"] = pd.to_numeric(test["mfe_pct_60d"], errors="coerce")
            pred_df["y_true"] = (pd.to_numeric(test[lcol], errors="coerce") >= thr).astype("Int64")
            all_preds.append(pred_df)
            best_iter = fit_info["best_iteration"]
            print(f"    n_train_used={fit_info['n_train_used']:,} val={fit_info['n_val']:,}  "
                  f"best_iter={best_iter}  test n={row['n_test']}  base={row['base_rate']:.4f}  "
                  f"AUC={row['auc']:.4f}  lift_dec={row['top_decile_lift']:.3f}  "
                  f"top_ticker={row['top20_top_ticker']}@{row['top20_max_ticker_share']:.2f}  "
                  f"capped_top50_prec={row['capped_precision_top50']:.4f}")

    metrics_df = pd.DataFrame(all_rows)
    calib_df = pd.concat(all_calib, ignore_index=True)
    imp_df = pd.concat(all_imp, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    # Pooled (ALL folds) metrics
    pooled_rows: list[dict] = []
    for tname, lcol, thr in targets_to_run:
        sub = preds_df[preds_df["target"] == tname].copy()
        sub = sub[sub["y_true"].notna()].reset_index(drop=True)
        sub["_score"] = sub["score"].values
        # Reuse metrics block on the pooled frame
        row, _, _, _ = _metrics_block(sub.rename(columns={"score": "_score_keep"}).assign(_score=sub["_score"].values),
                                       sub["_score"].values, lcol, thr, "ALL", tname)
        # NB: _metrics_block returns the pooled row directly; just rewrite fold/target
        row["fold"] = "ALL"; row["target"] = tname
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)

    metrics_all = pd.concat([metrics_df, pooled_df], ignore_index=True, sort=False)
    metrics_all.to_csv(DST_METRICS, index=False)
    print(f"\n[lgbm-5b] metrics → {DST_METRICS}  ({len(metrics_all)} rows)")

    preds_df.to_parquet(DST_PRED, index=False)
    print(f"[lgbm-5b] predictions → {DST_PRED}  ({len(preds_df):,} rows)")

    imp_agg = (
        imp_df.groupby(["target", "feature"], sort=False)
        .agg(mean_gain=("gain", "mean"), sum_split=("split", "sum"), n_folds=("gain", "count"))
        .reset_index()
        .sort_values(["target", "mean_gain"], ascending=[True, False])
    )
    imp_agg.to_csv(DST_IMP, index=False)
    calib_df.to_csv(DST_CALIB, index=False)
    print(f"[lgbm-5b] importance → {DST_IMP}  ({len(imp_agg)} rows)")
    print(f"[lgbm-5b] calibration → {DST_CALIB}  ({len(calib_df)} rows)")

    caps_df = pd.DataFrame(all_caps)
    DST_CAPS = OUT / "ranking_lab_lgbm_winsorize_caps_v0_stage5b.csv"
    caps_df.to_csv(DST_CAPS, index=False)
    print(f"[lgbm-5b] winsorize caps → {DST_CAPS}")

    # ----- Report -----
    lines: list[str] = []
    lines.append("# Ranking Lab v0 — Stage 5b De-scaled LightGBM Report")
    lines.append("")
    lines.append(f"- Input: `{FEAT_PATH.name}`")
    lines.append(f"- Folds: {[f[0] for f in FOLDS]}")
    lines.append(f"- Targets trained: {[t[0] for t in targets_to_run]}")
    lines.append(f"- Audit heads sparsity check: y30_80 min_fold_pos={min(pos_counts_audit['y30_80'].values())}; "
                 f"y60_100 min_fold_pos={min(pos_counts_audit['y60_100'].values())}")
    lines.append("")

    lines.append("## 0) Stage 5b changes vs Stage 5")
    lines.append("")
    lines.append("**Dropped from ML feature set (Stage 5 → 5b):** " + ", ".join(DROPPED_FROM_STAGE5))
    lines.append("")
    lines.append("**Added per-date rank features (rank-pct within signal_date):** " + ", ".join(RANK_FEATURES))
    lines.append("")
    lines.append("**Audit-only (NOT in feature whitelist):** " + ", ".join(AUDIT_RANK_FEATURES))
    lines.append("")
    lines.append("**Train-fold p1/p99 winsorize on raw numerics** (caps applied to val + test).")
    lines.append("")
    lines.append(f"**Eval ticker cap** (eval-only, model unaware): "
                 f"max {EVAL_CAP_MAX_PER_TICKER} rows per ticker in pooled top-K, "
                 f"max {EVAL_CAP_MAX_PER_TICKER_DATE} per (ticker, signal_date). Reports both RAW and CAPPED.")
    lines.append("")

    lines.append("## 1) Model spec")
    lines.append("")
    lines.append("LightGBM binary (same predeclared hyperparams as Stage 5, no retuning):")
    for k, v in LGBM_PARAMS.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- num_boost_round (max): {N_ESTIMATORS_MAX}")
    lines.append(f"- early_stopping_rounds: {EARLY_STOPPING_ROUNDS}  (val=last {VAL_FRACTION:.0%} of train period)")
    lines.append("")

    lines.append("## 2) Feature set (Stage 5b)")
    lines.append("")
    lines.append("**Numeric raw (kept):** " + ", ".join(NUMERIC_FEATURES_RAW))
    lines.append("")
    lines.append("**Numeric rank-pct by date (new):** " + ", ".join(RANK_FEATURES))
    lines.append("")
    lines.append("**Categorical (LGBM native):** " + ", ".join(CATEGORICAL_FEATURES))
    lines.append("")
    lines.append("**Forbidden patterns:** " + ", ".join(FORBIDDEN_PATTERNS))
    lines.append("")

    lines.append("## 3) Per-fold winsorize caps")
    lines.append("")
    lines.append(caps_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 4) Per-fold metrics (core targets)")
    lines.append("")
    cols_core = ["fold", "target", "n_test", "n_positive", "base_rate", "auc",
                 "top_decile_precision", "top_decile_lift",
                 "precision_top50", "capture_top50",
                 "capped_precision_top50", "capped_capture_top50",
                 "capped_top50_unique_tickers",
                 "top20_top_ticker", "top20_max_ticker_share",
                 "top20_max_source_family_share"]
    core_names = [t[0] for t in CORE_TARGETS]
    core_metrics = metrics_df[metrics_df["target"].isin(core_names)][cols_core].copy()
    lines.append(core_metrics.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 5) Per-fold metrics (audit heads)")
    lines.append("")
    audit_names = [t[0] for t in AUDIT_TARGETS if t[0] in {x[0] for x in targets_to_run}]
    if audit_names:
        audit_metrics = metrics_df[metrics_df["target"].isin(audit_names)][cols_core].copy()
        lines.append(audit_metrics.to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("(none trained)")
    lines.append("")

    lines.append("## 6) Pooled (ALL folds) metrics")
    lines.append("")
    lines.append(pooled_df[cols_core].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 7) Stage 5 vs Stage 5b comparison (pooled)")
    lines.append("")
    cmp_rows = []
    for tname, lcol, thr in ALL_TARGETS:
        row = pooled_df[pooled_df["target"] == tname]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        cmp_rows.append({
            "target": tname,
            "base_rate": r["base_rate"],
            "rule_lift": RULE_BASELINE_LIFT.get(tname, float("nan")),
            "stage5_raw_lift": STAGE5_ML_LIFT.get(tname, float("nan")),
            "stage5b_raw_lift": r["top_decile_lift"],
            "stage5b_capped_top50_prec": r["capped_precision_top50"],
            "stage5b_top20_unique": r["top20_unique_tickers"],
            "stage5b_top20_max_ticker_share": r["top20_max_ticker_share"],
            "stage5_top20_max_ticker_share": STAGE5_TOP20_MAX_TICKER.get(tname, float("nan")),
            "stage5b_top20_max_sf_share": r["top20_max_source_family_share"],
            "stage5b_capture_top50": r["capture_top50"],
            "stage5b_capped_capture_top50": r["capped_capture_top50"],
        })
    cmp_df = pd.DataFrame(cmp_rows)
    lines.append(cmp_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 8) Top-20 concentration (pooled, per target)")
    lines.append("")
    conc_cols = ["target", "top20_unique_tickers", "top20_top_ticker", "top20_max_ticker_share",
                 "top20_unique_dates", "top20_max_date_share",
                 "top20_top_source_family", "top20_max_source_family_share"]
    lines.append(pooled_df[conc_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 9) Top-20 feature importance per target (mean gain across folds)")
    lines.append("")
    for tname in [t[0] for t in targets_to_run]:
        sub = imp_agg[imp_agg["target"] == tname].head(20).copy()
        lines.append(f"### {tname}")
        lines.append("")
        lines.append(sub[["feature", "mean_gain", "sum_split", "n_folds"]].to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

    lines.append("## 10) Calibration deciles (pooled across folds, mean of per-fold rows)")
    lines.append("")
    cal_agg = (
        calib_df.groupby(["target", "decile"], sort=False)
        .agg(n=("n", "sum"), mean_score=("mean_score", "mean"), actual_rate=("actual_rate", "mean"))
        .reset_index()
        .sort_values(["target", "decile"])
    )
    for tname in [t[0] for t in targets_to_run]:
        sub = cal_agg[cal_agg["target"] == tname]
        lines.append(f"### {tname}")
        lines.append("")
        lines.append(sub[["decile", "n", "mean_score", "actual_rate"]].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    # ----- Verdict -----
    def _ml_lift(tname: str) -> float:
        row = pooled_df[pooled_df["target"] == tname]
        return float(row["top_decile_lift"].iloc[0]) if len(row) else float("nan")

    def _max_share(field: str) -> float:
        return float(pooled_df[field].max()) if len(pooled_df) else float("nan")

    lift = {t: _ml_lift(t) for t in [x[0] for x in targets_to_run]}
    max_ticker_share = _max_share("top20_max_ticker_share")
    max_date_share = _max_share("top20_max_date_share")
    max_sf_share = _max_share("top20_max_source_family_share")

    # Feature importance: are top-3 features by mean gain all rank-relative? (sanity)
    raw_abs_in_top3: list[str] = []
    for tname in [t[0] for t in targets_to_run]:
        top3 = imp_agg[imp_agg["target"] == tname].head(3)["feature"].tolist()
        for f in top3:
            if f in {"entry_ref", "atr", "atr_14_daily", "liquidity_proxy"}:
                raw_abs_in_top3.append(f"{tname}:{f}")
    # (Should be empty by construction — those features are not in FEATURE_COLS.)

    # Per ONAY thresholds
    concentration_kill = False
    conc_notes: list[str] = []
    if max_ticker_share > 0.50:
        concentration_kill = True
        conc_notes.append(f"top20 single-ticker peaked at {max_ticker_share:.0%}.")
    if max_date_share > 0.50:
        concentration_kill = True
        conc_notes.append(f"top20 single-date peaked at {max_date_share:.0%}.")
    if max_sf_share > 0.50:
        concentration_kill = True
        conc_notes.append(f"top20 single-source/family peaked at {max_sf_share:.0%}.")

    # Tail thresholds
    y30_50_pass = lift.get("y30_50", float("nan")) >= 1.30
    y60_80_pass = lift.get("y60_80", float("nan")) >= 1.20

    # Short heads
    short_strong = (lift.get("y3_8", 0) > RULE_BASELINE_LIFT["y3_8"]
                    and lift.get("y5_10", 0) > RULE_BASELINE_LIFT["y5_10"]
                    and lift.get("y10_15", 0) > RULE_BASELINE_LIFT["y10_15"])

    if concentration_kill:
        verdict = "STOP_CONCENTRATION"
    elif y30_50_pass and y60_80_pass:
        verdict = "PASS_TAIL_PROMISING"
    elif short_strong and not (y30_50_pass or y60_80_pass):
        verdict = "PASS_SHORT_ONLY"
    else:
        verdict = "WEAK_DO_NOT_USE"

    lines.append(f"## VERDICT: **{verdict}**")
    lines.append("")
    lines.append("### Pooled lift summary")
    lines.append("")
    lines.append("| target | base | rule | stage5_raw | stage5b_raw | Δ(5b-rule) | Δ(5b-5) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tname in [x[0] for x in targets_to_run]:
        rule = RULE_BASELINE_LIFT.get(tname, float("nan"))
        s5 = STAGE5_ML_LIFT.get(tname, float("nan"))
        s5b = lift.get(tname, float("nan"))
        base = pooled_df[pooled_df["target"] == tname]["base_rate"].iloc[0]
        lines.append(f"| {tname} | {base:.4f} | {rule:.3f}× | {s5:.3f}× | {s5b:.3f}× | {s5b - rule:+.3f} | {s5b - s5:+.3f} |")
    lines.append("")
    lines.append("### Concentration")
    lines.append("")
    lines.append(f"- top20 max single-ticker share across targets: **{max_ticker_share:.2%}**")
    lines.append(f"- top20 max single-date share: {max_date_share:.2%}")
    lines.append(f"- top20 max source/family share: {max_sf_share:.2%}")
    for nt in conc_notes:
        lines.append(f"- {nt}")
    lines.append("")
    lines.append("### Pass criteria checks")
    lines.append("")
    lines.append(f"- y30_50 ≥ 1.30× ? **{y30_50_pass}** (ml={lift.get('y30_50', float('nan')):.3f}×)")
    lines.append(f"- y60_80 ≥ 1.20× ? **{y60_80_pass}** (ml={lift.get('y60_80', float('nan')):.3f}×)")
    lines.append(f"- short heads remain above rule ? **{short_strong}**")
    lines.append(f"- top20 dominant ticker <50% ? **{max_ticker_share < 0.50}**")
    lines.append(f"- raw absolute-scale features in top-3 importance ? **{len(raw_abs_in_top3) > 0}** "
                 f"({raw_abs_in_top3 if raw_abs_in_top3 else 'none — by construction'})")
    lines.append("")

    DST_MD.write_text("\n".join(lines))
    print(f"\n[lgbm-5b] wrote {DST_MD}")
    print(f"[lgbm-5b] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
