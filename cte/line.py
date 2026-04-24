"""
CTE line — shared training / evaluation helper for HB and FC pipelines.

Design choice (2026-04-23):
CTE no longer has a single universal ranker. HB and FC are separate
production lines. This module is the shared core: each line calls
``train_line`` with its own trigger column and mode. There is no cross-line
score combine here — that is the job of ``cte.tools.portfolio_merge``.

Training rules per line:
  - mode == "pure":  train on rows where setup_type == line_name.
                     Score at inference on ALL trigger_{line} == True rows
                     (so "both" rows still get a score even though they
                     weren't in training).
  - mode == "mixed": train + score on all trigger_{line} == True rows.

This matches the v2/v2a/v2b ablation in train_specialist.py but without
the cross-head combining step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import lightgbm as lgb

from cte.config import CONFIG, SplitParams
from cte.features import FEATURES_V1
from cte.train import LGBMParams, _per_date_rank


LINE_TO_TRIGGER = {"hb": "trigger_hb", "fc": "trigger_fc"}


@dataclass
class LineResult:
    preds: pd.DataFrame            # per-row test scores, all folds
    importance: pd.DataFrame       # feature importance (per-fold + mean)
    line: str                      # "hb" | "fc"
    mode: str                      # "pure" | "mixed"
    target: str


def _fold_train(
    train_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.Series]:
    td = pd.to_datetime(train_pool["date"])
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    vs, ve = pd.Timestamp(val_start), pd.Timestamp(val_end)

    tr_mask = (td >= ts) & (td <= te) & train_pool[target_col].notna()
    va_mask = (td >= vs) & (td <= ve) & train_pool[target_col].notna()
    tr = train_pool.loc[tr_mask].copy()
    va = train_pool.loc[va_mask].copy()

    qd = pd.to_datetime(test_pool["date"])
    qs, qe = pd.Timestamp(test_start), pd.Timestamp(test_end)
    qt_mask = (qd >= qs) & (qd <= qe) & test_pool[target_col].notna()
    qt = test_pool.loc[qt_mask].copy()

    X_tr, y_tr = tr[feature_cols], tr[target_col].astype(int)
    X_va, y_va = va[feature_cols], va[target_col].astype(int)
    X_qt = qt[feature_cols]

    print(
        f"  [{fold_name}] train N={len(tr):,}  val N={len(va):,}  test N={len(qt):,}  "
        f"(pos tr/va/qt: {y_tr.mean()*100:.1f}% / {y_va.mean()*100:.1f}% / "
        f"{qt[target_col].mean()*100:.1f}%)"
    )

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
    return qt, imp


def train_line(
    df: pd.DataFrame,
    line: Literal["hb", "fc"],
    mode: Literal["pure", "mixed"],
    target: str,
    feature_cols: list[str] | None = None,
    split: SplitParams | None = None,
    params: LGBMParams | None = None,
) -> LineResult:
    """Train one CTE line (HB or FC). Returns predictions + importance.

    No cross-line combining here. Portfolio-level merge is a separate step.
    """
    if line not in ("hb", "fc"):
        raise ValueError(f"line must be 'hb' or 'fc', got {line!r}")
    if mode not in ("pure", "mixed"):
        raise ValueError(f"mode must be 'pure' or 'mixed', got {mode!r}")

    split = split or CONFIG.split
    params = params or LGBMParams()
    feature_cols = feature_cols or [c for c in FEATURES_V1 if c in df.columns]

    trigger_col = LINE_TO_TRIGGER[line]
    all_members = df[df[trigger_col].astype(bool)].copy()
    train_pool = (
        all_members[all_members["setup_type"] == line].copy()
        if mode == "pure" else all_members.copy()
    )

    print(
        f"\n━━━ {line.upper()}-line ({mode})  "
        f"N_train_pool={len(train_pool):,}  N_test_pool={len(all_members):,}  "
        f"pos_rate(train)={train_pool[target].dropna().mean()*100:.1f}% ━━━"
    )

    all_test: list[pd.DataFrame] = []
    all_imp: list[pd.Series] = []
    for fs in split.folds:
        print(f"  [FOLD {fs.name}]")
        try:
            qt, imp = _fold_train(
                train_pool, all_members, feature_cols, target, fs.name,
                train_start=split.train_start,
                train_end=fs.train_end,
                val_start=fs.val_start, val_end=fs.val_end,
                test_start=fs.test_start, test_end=fs.test_end,
                params=params,
            )
            qt["line"] = line
            qt["line_mode"] = mode
            all_test.append(qt)
            all_imp.append(imp)
        except Exception as e:
            print(f"    fold {fs.name} failed: {e}")

    preds = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    preds = preds.sort_values(["date", "ticker"]).reset_index(drop=True) if not preds.empty else preds

    imp_df = pd.concat(all_imp, axis=1) if all_imp else pd.DataFrame()
    if not imp_df.empty:
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)

    return LineResult(preds=preds, importance=imp_df, line=line, mode=mode, target=target)


# ═════════════════════════════════════════════════════════════════════════════
# Per-line evaluation — single-setup; base rate is line-specific, not universal
# ═════════════════════════════════════════════════════════════════════════════

def _precision_at(score: pd.Series, target: pd.Series, k_frac: float) -> tuple[int, float]:
    mask = score.notna() & target.notna()
    s, y = score[mask], target[mask]
    if len(s) == 0:
        return 0, float("nan")
    k = max(1, int(round(len(s) * k_frac)))
    top = s.nlargest(k).index
    return k, float(y.loc[top].mean())


def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return float("nan")
    return float(a[mask].rank().corr(b[mask].rank()))


def _pf_proxy(df: pd.DataFrame, target: str, mfe: str, mae: str) -> dict:
    if mfe not in df.columns or mae not in df.columns:
        return {}
    mask = df[target].notna() & df[mfe].notna() & df[mae].notna()
    sub = df.loc[mask]
    if sub.empty:
        return {"n": 0}
    gain = np.minimum(sub[mfe].values, 3.0)
    loss = -np.minimum(sub[mae].values, 1.5)
    realised = np.where(sub[target].values == 1, gain, loss)
    pos_sum = realised[realised > 0].sum()
    neg_sum = -realised[realised < 0].sum()
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")
    return {
        "n": int(len(sub)),
        "avg_R": float(realised.mean()),
        "PF_proxy": float(pf),
        "WR_proxy": float((realised > 0).mean()),
    }


def eval_line(
    preds: pd.DataFrame,
    target: str,
    line: str,
    scores_to_compare: tuple[str, ...] = ("score_model", "score_compression", "score_random"),
) -> dict:
    """Per-line evaluation. Reports metrics within the line's own base rate.

    Returns a dict with:
      - overall: {score: {rho, p@10/20/30, lift@10/20/30, pf_proxy, wr_proxy}}
      - by_fold: same structure per fold
      - by_setup: same structure per setup_type
      - top_decile_rate: runner rate at top 10% by score_model
    """
    out: dict = {"line": line, "target": target}

    def _slice_metrics(df: pd.DataFrame, tag: str) -> dict:
        if df.empty or df[target].notna().sum() == 0:
            return {"n": 0, "base": float("nan")}
        base = float(df[target].dropna().mean())
        block = {"n": int(df[target].notna().sum()), "base": base, "scores": {}}
        for sc in scores_to_compare:
            if sc not in df.columns:
                continue
            rho = _spearman(df[sc], df[target])
            _, p10 = _precision_at(df[sc], df[target], 0.10)
            _, p20 = _precision_at(df[sc], df[target], 0.20)
            _, p30 = _precision_at(df[sc], df[target], 0.30)
            block["scores"][sc] = {
                "rho": rho,
                "p@10": p10, "p@20": p20, "p@30": p30,
                "lift@10": p10 / base if base > 0 else float("nan"),
                "lift@20": p20 / base if base > 0 else float("nan"),
                "lift@30": p30 / base if base > 0 else float("nan"),
            }
        # PF proxy (on primary target horizon)
        h = target.split("_")[-1]
        pf = _pf_proxy(df, target, f"mfe_{h}_atr", f"mae_{h}_atr")
        block["pf_proxy_overall"] = pf
        # PF proxy on top 30% by score_model
        if "score_model" in df.columns:
            top30_cut = df["score_model"].quantile(0.70)
            top30 = df[df["score_model"] >= top30_cut]
            block["pf_proxy_top30"] = _pf_proxy(top30, target, f"mfe_{h}_atr", f"mae_{h}_atr")
        return block

    out["overall"] = _slice_metrics(preds, "overall")

    by_fold = {}
    for fold in sorted(preds["fold_assigned"].dropna().unique()):
        by_fold[fold] = _slice_metrics(preds[preds["fold_assigned"] == fold], fold)
    out["by_fold"] = by_fold

    by_setup = {}
    if "setup_type" in preds.columns:
        for st in preds["setup_type"].dropna().unique():
            sub = preds[preds["setup_type"] == st]
            if len(sub) < 10:
                continue
            by_setup[st] = _slice_metrics(sub, st)
    out["by_setup"] = by_setup

    # Decile monotonicity
    if "score_model" in preds.columns:
        mask = preds["score_model"].notna() & preds[target].notna()
        if mask.sum() >= 20:
            s = preds.loc[mask, "score_model"]
            y = preds.loc[mask, target]
            try:
                qs = pd.qcut(s, q=10, labels=False, duplicates="drop")
                tab = (
                    pd.DataFrame({"q": qs, "y": y})
                    .groupby("q")
                    .agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean"))
                    .reset_index()
                )
                out["deciles"] = tab.to_dict(orient="records")
            except Exception:
                out["deciles"] = []
    return out


def format_eval_report(evald: dict) -> str:
    """Pretty-print an eval_line() result for CLI."""
    lines = [f"═══ CTE {evald['line'].upper()}-line eval ═══  target={evald['target']}"]

    def _fmt_block(tag: str, block: dict) -> list[str]:
        if not block or "scores" not in block or block.get("n", 0) == 0:
            return [f"  {tag}: (no rows)"]
        rows = [f"  {tag}: N={block['n']}  base={block['base']:.2%}"]
        for sc, m in block["scores"].items():
            rows.append(
                f"    {sc:<20} rho={m['rho']:+.3f}  "
                f"p@10={m['p@10']:.1%}  p@20={m['p@20']:.1%}  "
                f"lift@10={m['lift@10']:.2f}x  lift@20={m['lift@20']:.2f}x"
            )
        pf_o = block.get("pf_proxy_overall", {})
        if pf_o.get("n"):
            rows.append(
                f"    PF_proxy overall: PF={pf_o.get('PF_proxy', float('nan')):.2f}  "
                f"WR={pf_o.get('WR_proxy', float('nan')):.2%}  "
                f"avgR={pf_o.get('avg_R', float('nan')):+.2f}"
            )
        pf_t = block.get("pf_proxy_top30", {})
        if pf_t.get("n"):
            rows.append(
                f"    PF_proxy top30%:  PF={pf_t.get('PF_proxy', float('nan')):.2f}  "
                f"WR={pf_t.get('WR_proxy', float('nan')):.2%}  "
                f"avgR={pf_t.get('avg_R', float('nan')):+.2f}"
            )
        return rows

    lines += ["", "── OVERALL ──"] + _fmt_block("overall", evald.get("overall", {}))

    lines += ["", "── PER FOLD ──"]
    for fold, block in evald.get("by_fold", {}).items():
        lines += _fmt_block(fold, block)

    lines += ["", "── PER SETUP_TYPE ──"]
    for st, block in evald.get("by_setup", {}).items():
        lines += _fmt_block(f"setup={st}", block)

    deciles = evald.get("deciles") or []
    if deciles:
        lines += ["", "── score_model DECILES ──"]
        for row in deciles:
            lines.append(
                f"    q{int(row['q']):<2} n={int(row['n']):>3}  "
                f"pos={int(row['pos']):>3}  rate={row['rate']:.1%}"
            )
    return "\n".join(lines)


def save_line_artifacts(
    result: LineResult,
    preds_path: str | Path,
    importance_path: str | Path,
) -> None:
    preds_path = Path(preds_path)
    importance_path = Path(importance_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep diagnostic columns in preds
    keep = [
        "ticker", "date", "fold_assigned", "line", "line_mode",
        "setup_type", "trigger_hb", "trigger_fc", "close",
        "score_model", "score_random", "score_compression",
    ]
    label_family = [
        "hold_3_close", "hold_5_close", "hold_3_struct", "hold_5_struct",
        "failed_break_3_close", "failed_break_5_close",
        "failed_break_3_struct", "failed_break_5_struct",
        "mfe_10_atr", "mae_10_atr", "spike_rejected_10",
        "mfe_15_atr", "mae_15_atr", "spike_rejected_15",
        "mfe_20_atr", "mae_20_atr", "spike_rejected_20",
        "expansion_score_10", "expansion_score_15", "expansion_score_20",
        "runner_10", "runner_15", "runner_20",
        "breakout_level_struct", "breakout_level_close", "atr_ref",
        "compression_score", "bar_return_1d", "breakout_vol_ratio",
    ]
    cols = [c for c in dict.fromkeys(keep + label_family) if c in result.preds.columns]
    result.preds[cols].to_parquet(preds_path)
    result.importance.to_csv(importance_path)
    print(f"[WRITE] {preds_path}  shape={result.preds[cols].shape}")
    print(f"[WRITE] {importance_path}")
