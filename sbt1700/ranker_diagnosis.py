"""Per-exit LightGBM ranker diagnosis (PR-2).

For each exit variant we train a small LightGBM regressor on
``realized_R_net`` using only 17:00-knowable features. We then evaluate
on a 3-fold walk-forward split (date-based, no overlap), plus a 2026
holdout if N permits.

This is *diagnosis*, not deployment:
- no hyperparameter sweep
- conservative defaults (seed=17, num_leaves=31, lr=0.05, n_estimators=200)
- top-decile metrics report only — no acceptance gate / live wiring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sbt1700.exits import variant_names


# Columns that must NOT be used as features (label or post-trade info).
NON_FEATURE_COLS = {
    "ticker", "date", "schema_version",
    # Frozen E3 label cols carried in dataset:
    "realized_R_gross", "realized_R_net", "win_label",
    "tp_hit", "sl_hit", "timeout_hit", "exit_reason",
    "bars_held", "entry_px", "stop_px", "tp_px",
    "atr_1700", "initial_R_price", "exit_px", "exit_date", "cost_R",
    # Per-variant trade output cols (when joining trades_df):
    "exit_variant", "partial_hit", "partial_px",
}


LGBM_PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=31,
    learning_rate=0.05,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbose=-1,
    seed=17,
)
N_ESTIMATORS = 200


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _profit_factor(rs: np.ndarray) -> float:
    rs = rs[~np.isnan(rs)]
    wins = rs[rs > 0].sum()
    losses = -rs[rs < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def date_walk_forward_folds(dates: pd.Series, k: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Returns k expanding folds defined by (train_end, val_start, val_end).
    Train = dates <= train_end. Val = train_end < dates <= val_end.
    Splits the unique signal dates into k contiguous chunks for val.
    """
    uniq = pd.Series(sorted(dates.unique()))
    if len(uniq) < (k + 1) * 2:
        return []
    # First chunk reserved as initial train; remaining chunks become val.
    # Use k+1 chunks so the first becomes train and the rest are val folds.
    chunks = np.array_split(uniq, k + 1)
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(1, k + 1):
        prior = pd.concat([pd.Series(c).reset_index(drop=True) for c in chunks[:i]], ignore_index=True)
        val = pd.Series(chunks[i]).reset_index(drop=True)
        train_end = prior.iloc[-1]
        val_start = val.iloc[0]
        val_end = val.iloc[-1]
        folds.append((pd.Timestamp(train_end), pd.Timestamp(val_start), pd.Timestamp(val_end)))
    return folds


def _train_score(train: pd.DataFrame, val: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, lgb.Booster]:
    train_R = train["realized_R_net"].astype(float)
    valid_mask = train_R.notna()
    Xtr = train.loc[valid_mask, feats].astype(float).values
    ytr = train_R[valid_mask].values
    Xva = val[feats].astype(float).values
    dtrain = lgb.Dataset(Xtr, label=ytr)
    booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=N_ESTIMATORS)
    preds = booster.predict(Xva, num_iteration=booster.best_iteration or N_ESTIMATORS)
    return preds, booster


def evaluate_fold(val_df: pd.DataFrame, preds: np.ndarray, top_decile: float = 0.10) -> dict:
    R = val_df["realized_R_net"].astype(float).values
    valid = ~np.isnan(R) & ~np.isnan(preds)
    R = R[valid]
    P = preds[valid]
    n = len(R)
    if n == 0:
        return {"n": 0}
    rho, p_rho = spearmanr(P, R)
    # Top decile
    k = max(1, int(round(top_decile * n)))
    top_idx = np.argsort(P)[::-1][:k]
    top_R = R[top_idx]
    return {
        "n": int(n),
        "spearman_rho": float(rho) if rho == rho else float("nan"),
        "spearman_p": float(p_rho) if p_rho == p_rho else float("nan"),
        "all_avg_R": float(R.mean()),
        "all_PF": _profit_factor(R),
        "all_WR": float((R > 0).mean()),
        "top_decile_n": int(k),
        "top_decile_avg_R": float(top_R.mean()),
        "top_decile_PF": _profit_factor(top_R),
        "top_decile_WR": float((top_R > 0).mean()),
        "top_decile_total_R": float(top_R.sum()),
    }


def diagnose_one_variant(
    feature_panel: pd.DataFrame,
    trades_for_variant: pd.DataFrame,
    variant: str,
) -> list[dict]:
    """Return a list of fold-level result dicts for one exit variant."""
    # Join feature panel with the variant's per-row R_net.
    df = feature_panel.merge(
        trades_for_variant[["ticker", "date", "realized_R_net"]],
        on=["ticker", "date"],
        how="inner",
        suffixes=("", "_drop"),
    )
    if "realized_R_net_drop" in df.columns:
        df = df.drop(columns=["realized_R_net"]).rename(columns={"realized_R_net_drop": "realized_R_net"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["realized_R_net"]).sort_values("date").reset_index(drop=True)

    feats = feature_columns(df)
    folds = date_walk_forward_folds(df["date"], k=3)
    if not folds:
        return [{"exit_variant": variant, "fold": "insufficient_dates", "n": int(len(df))}]

    rows: list[dict] = []
    for fi, (train_end, val_start, val_end) in enumerate(folds, 1):
        train = df[df["date"] <= train_end]
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
        if len(train) < 50 or len(val) < 20:
            rows.append({
                "exit_variant": variant, "fold": f"fold_{fi}",
                "train_end": str(train_end.date()),
                "val_start": str(val_start.date()), "val_end": str(val_end.date()),
                "n_train": int(len(train)), "n_val": int(len(val)),
                "skipped": "too_small",
            })
            continue
        preds, _ = _train_score(train, val, feats)
        m = evaluate_fold(val, preds)
        m.update({
            "exit_variant": variant,
            "fold": f"fold_{fi}",
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_features": len(feats),
        })
        rows.append(m)

    # 2026 holdout (separate from walk-forward).
    train_holdout = df[df["date"].dt.year < 2026]
    val_holdout = df[df["date"].dt.year == 2026]
    if len(train_holdout) >= 100 and len(val_holdout) >= 30:
        preds, _ = _train_score(train_holdout, val_holdout, feats)
        m = evaluate_fold(val_holdout, preds)
        m.update({
            "exit_variant": variant,
            "fold": "holdout_2026",
            "train_end": str(train_holdout["date"].max().date()),
            "val_start": str(val_holdout["date"].min().date()),
            "val_end": str(val_holdout["date"].max().date()),
            "n_train": int(len(train_holdout)),
            "n_val": int(len(val_holdout)),
            "n_features": len(feats),
        })
        rows.append(m)

    return rows


def run_ranker_diagnosis(
    dataset: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    out: list[dict] = []
    for v in variant_names():
        sub = trades_df[trades_df["exit_variant"] == v]
        if sub.empty:
            continue
        rows = diagnose_one_variant(dataset, sub, v)
        out.extend(rows)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="SBT-1700 per-exit ranker diagnosis.")
    ap.add_argument("--dataset", type=Path, default=Path("output/sbt_1700_dataset.parquet"))
    ap.add_argument("--trades", type=Path, default=Path("output/sbt_1700_exit_matrix_trades.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    args = ap.parse_args()

    dataset = pd.read_parquet(args.dataset)
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    trades = pd.read_parquet(args.trades)
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    print(f"[ranker] dataset {dataset.shape} | trades {trades.shape}")

    results = run_ranker_diagnosis(dataset, trades)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "sbt_1700_ranker_by_exit.csv"
    results.to_csv(out_path, index=False)
    print(f"[ranker] wrote {out_path} ({len(results)} fold rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
