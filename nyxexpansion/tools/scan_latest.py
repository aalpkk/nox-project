"""
v4C ile canlı BIST tarama — son trigger bar(lar)ında tahmin üret.

Akış:
  1. Mevcut dataset v3'ü yükle (trigger + features + labels)
  2. target_date için XU100 regime'ini refresh edip dataset'e patch'le
  3. fold3 gibi eğitim penceresi ile clf (L3) + winner_R regressor eğit
  4. target_date rows üzerinde tahmin yap (label NaN olabilir — qt.notna filtresi yok)
  5. v4C score = winner_R_pred  → range veto + per-fold rank normalize
  6. Top picks'i yazdır

Leakage: train/val window target_date'ten ≥15 takvim günü önce biter; label horizon=10 bar < 15g.

Kullanım:
    python -m nyxexpansion.tools.scan_latest --date 2026-04-20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

import lightgbm as lgb

from nyxexpansion.config import CONFIG
from nyxexpansion.features import CORE_FEATURES_UP, CORE_FEATURES_NONUP
from nyxexpansion.train import LGBMParams
from nyxexpansion.train_winner import _add_winner_R
from nyxexpansion.tools.presmoke import load_xu100, classify_xu100_regime


UP = {'uptrend'}
NONUP = {'range', 'downtrend', 'unknown'}


def _regime_mask(s, kind):
    r = s.fillna('unknown')
    return r.isin(UP) if kind == 'up' else r.isin(NONUP)


def patch_regime_for_date(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """xu_regime None olanları refresh XU100'den doldur."""
    xu = load_xu100(refresh=False, period='2y')
    reg = classify_xu100_regime(xu)
    # Normalize index
    reg.index = pd.to_datetime(reg.index).normalize()
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    d_norm = df['date'].dt.normalize()
    mapped = d_norm.map(reg)
    # Fill missing
    df['xu_regime'] = df['xu_regime'].where(
        df['xu_regime'].notna() & (df['xu_regime'] != 'None'),
        mapped,
    )
    return df


def train_clf(
    df: pd.DataFrame, qt: pd.DataFrame,
    train_start, train_end, val_start, val_end,
    target='cont_10', params: LGBMParams | None = None,
) -> pd.Series:
    """Classifier (L3) — UP vs NONUP routed."""
    if params is None:
        params = LGBMParams()
    d = pd.to_datetime(df['date'])
    tgt_ok = df[target].notna()
    tr = df[(d >= pd.Timestamp(train_start)) & (d <= pd.Timestamp(train_end)) & tgt_ok]
    va = df[(d >= pd.Timestamp(val_start)) & (d <= pd.Timestamp(val_end)) & tgt_ok]
    print(f"  [CLF] tr={len(tr):,} val={len(va):,} qt={len(qt):,}")

    out = pd.Series(np.nan, index=qt.index, dtype=float)
    feats = {'up': CORE_FEATURES_UP, 'nonup': CORE_FEATURES_NONUP}
    for kind in ['up', 'nonup']:
        tr_k = tr[_regime_mask(tr['xu_regime'], kind)]
        va_k = va[_regime_mask(va['xu_regime'], kind)]
        qt_k = qt[_regime_mask(qt['xu_regime'], kind)]
        if len(tr_k) < 200:
            print(f"    [{kind}] yetersiz train: tr={len(tr_k)} → skip")
            continue
        X_tr = tr_k[feats[kind]]; y_tr = tr_k[target].astype(int)
        has_val = len(va_k) >= 20
        clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=params.n_estimators, learning_rate=params.learning_rate,
            num_leaves=params.num_leaves, max_depth=params.max_depth,
            min_child_samples=params.min_child_samples,
            feature_fraction=params.feature_fraction,
            bagging_fraction=params.bagging_fraction, bagging_freq=params.bagging_freq,
            reg_alpha=params.reg_alpha, reg_lambda=params.reg_lambda,
            random_state=params.seed, verbose=-1, n_jobs=-1,
        )
        if has_val:
            X_va = va_k[feats[kind]]; y_va = va_k[target].astype(int)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='binary_logloss',
                    callbacks=[lgb.early_stopping(params.early_stopping_rounds, verbose=False),
                               lgb.log_evaluation(0)])
            bi = clf.best_iteration_
        else:
            clf.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])
            bi = None
        if len(qt_k):
            scores = clf.predict_proba(qt_k[feats[kind]], num_iteration=bi)[:, 1]
            out.loc[qt_k.index] = scores
        print(f"    [{kind}] tr={len(tr_k)} val={len(va_k)} best_iter={bi} "
              f"qt_scored={len(qt_k)}")
    return out


def train_winner(
    df: pd.DataFrame, qt: pd.DataFrame,
    train_start, train_end, val_start, val_end,
    l3_col='cont_10', target='winner_R', params: LGBMParams | None = None,
    winsorize_q=0.99,
) -> pd.Series:
    """Winner regressor — L3=1 subset; qt tüm adaylar."""
    if params is None:
        params = LGBMParams()
    d = pd.to_datetime(df['date'])
    l3m = df[l3_col] == 1
    tgt_ok = df[target].notna() & np.isfinite(df[target])
    tr = df[(d >= pd.Timestamp(train_start)) & (d <= pd.Timestamp(train_end)) & l3m & tgt_ok]
    va = df[(d >= pd.Timestamp(val_start)) & (d <= pd.Timestamp(val_end)) & l3m & tgt_ok]
    print(f"  [WIN] L3=1 tr={len(tr):,} val={len(va):,} qt={len(qt):,}")

    out = pd.Series(np.nan, index=qt.index, dtype=float)
    feats = {'up': CORE_FEATURES_UP, 'nonup': CORE_FEATURES_NONUP}
    for kind in ['up', 'nonup']:
        tr_k = tr[_regime_mask(tr['xu_regime'], kind)].copy()
        va_k = va[_regime_mask(va['xu_regime'], kind)].copy()
        qt_k = qt[_regime_mask(qt['xu_regime'], kind)]
        if len(tr_k) < 50:
            print(f"    [{kind}] yetersiz L3=1 train: {len(tr_k)} → skip")
            continue
        comb = pd.concat([tr_k[target], va_k[target]])
        hi = comb.quantile(winsorize_q); lo = comb.quantile(1 - winsorize_q)
        tr_k[target] = tr_k[target].clip(lower=lo, upper=hi)
        if len(va_k):
            va_k[target] = va_k[target].clip(lower=lo, upper=hi)
        X_tr = tr_k[feats[kind]]; y_tr = tr_k[target].astype(float)
        has_val = len(va_k) >= 10
        reg = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=params.n_estimators, learning_rate=params.learning_rate,
            num_leaves=params.num_leaves, max_depth=params.max_depth,
            min_child_samples=params.min_child_samples,
            feature_fraction=params.feature_fraction,
            bagging_fraction=params.bagging_fraction, bagging_freq=params.bagging_freq,
            reg_alpha=params.reg_alpha, reg_lambda=params.reg_lambda,
            random_state=params.seed, verbose=-1, n_jobs=-1,
        )
        if has_val:
            X_va = va_k[feats[kind]]; y_va = va_k[target].astype(float)
            reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='l1',
                    callbacks=[lgb.early_stopping(params.early_stopping_rounds, verbose=False),
                               lgb.log_evaluation(0)])
            bi = reg.best_iteration_
        else:
            reg.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])
            bi = None
        if len(qt_k):
            out.loc[qt_k.index] = reg.predict(qt_k[feats[kind]], num_iteration=bi)
        print(f"    [{kind}] tr={len(tr_k)} val={len(va_k)} best_iter={bi} "
              f"qt_scored={len(qt_k)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='output/nyxexp_dataset_v3.parquet')
    ap.add_argument('--date', default=None, help='target date YYYY-MM-DD (None = latest)')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--also-last-n-days', type=int, default=3,
                    help='include last N trading days in scoring window')
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    df = pd.read_parquet(args.dataset)
    df['date'] = pd.to_datetime(df['date'])
    print(f"═══ nyxexpansion SCAN — dataset: {args.dataset}")
    print(f"  dataset range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Patch regime (some rows may be None if XU100 was stale)
    df = patch_regime_for_date(df, df['date'].max())

    # winner_R target
    df = _add_winner_R(df, h=CONFIG.label.primary_h)

    # Target date
    if args.date is None:
        target_dt = df['date'].max()
    else:
        target_dt = pd.Timestamp(args.date)
    print(f"  TARGET DATE: {target_dt.date()}")

    # qt scope = last N trading days (inclusive target)
    all_dates_sorted = sorted(df['date'].unique())
    idx = all_dates_sorted.index(target_dt) if target_dt in all_dates_sorted else len(all_dates_sorted) - 1
    lookback_start = all_dates_sorted[max(0, idx - args.also_last_n_days + 1)]
    qt = df[(df['date'] >= lookback_start) & (df['date'] <= target_dt)].copy()
    print(f"  scoring window: {lookback_start.date()} → {target_dt.date()}  N={len(qt)}")
    if len(qt) == 0:
        print(f"❌ no signals in scoring window")
        return 1
    qt_daily = qt.groupby(qt['date'].dt.date).size().to_dict()
    print(f"  per-day: {qt_daily}")

    # Training window: everything up to target_dt - 15 days (embargo for 10-bar label)
    # Val: 6-ay penceresi hemen embargo öncesi, train: bundan ötesi
    embargo_days = 15
    val_end = target_dt - pd.Timedelta(days=embargo_days)
    val_start = val_end - pd.Timedelta(days=180)
    train_end = val_start - pd.Timedelta(days=embargo_days)
    train_start = CONFIG.split.train_start
    print(f"\n  train: {train_start} → {train_end.date()}  "
          f"val: {val_start.date()} → {val_end.date()}  "
          f"embargo={embargo_days}d to qt")

    # ═══ Train + score ═══
    print(f"\n[1/2 classifier]")
    p_l3 = train_clf(df, qt, train_start, train_end.strftime('%Y-%m-%d'),
                     val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'))
    qt['p_l3'] = p_l3

    print(f"\n[2/2 winner regressor]")
    wR = train_winner(df, qt, train_start, train_end.strftime('%Y-%m-%d'),
                      val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'))
    qt['winner_R_pred'] = wR

    # ═══ v4C score + filters ═══
    qt['score_v4C'] = qt['winner_R_pred']

    # Range veto
    print(f"\n[FILTERS]")
    before = len(qt)
    vet = qt[qt['xu_regime'] != 'range'].copy()
    print(f"  range veto:   {before} → {len(vet)}  (removed {before-len(vet)})")

    # Drop NaN score (e.g. regime hiç kurulamamış)
    vet = vet[vet['score_v4C'].notna()]
    print(f"  valid score:  {len(vet)}")

    # Per-day percentile (fold_assigned yerine date)
    vet['score_pct'] = vet.groupby(vet['date'].dt.date)['score_v4C'].rank(pct=True)

    # Sort
    vet = vet.sort_values(['date', 'score_v4C'], ascending=[False, False])

    # ═══ Execution metadata tagging ═══
    from nyxexpansion.tools.candidate_tags import tag_candidates, print_tag_summary
    from nyxexpansion.tools.execution_risk_score import score_candidates
    vet = tag_candidates(vet)
    vet = score_candidates(vet)
    print_tag_summary(vet)
    print(f"\n─── RISK_BUCKET DAĞILIMI ──────────────────────────")
    rbdist = vet['risk_bucket'].value_counts()
    for b in ['clean', 'mild', 'elevated', 'severe']:
        n = int(rbdist.get(b, 0))
        print(f"  {b:<10}  N={n:>3}  ({n/len(vet)*100:5.1f}%)")

    # Top picks
    print(f"\n═══ TOP-{args.top} v4C picks — {lookback_start.date()} → {target_dt.date()} ═══")
    print(f"  {'date':<10}  {'ticker':<7}  {'regime':<10}  "
          f"{'p_l3':>6}  {'winR':>6}  {'pct':>5}  "
          f"{'exec_tag':<18}  {'risk':<9}  {'rscr':>5}  "
          f"stretch/ext/mom/room")
    top = vet.head(args.top)
    for _, r in top.iterrows():
        print(f"  {r['date'].date()!s:<10}  {r['ticker']:<7}  "
              f"{r['xu_regime']:<10}  "
              f"{r['p_l3']:>6.3f}  {r['winner_R_pred']:>6.2f}  "
              f"{r['score_pct']:>5.2f}  "
              f"{r['exec_tag']:<18}  "
              f"{r['risk_bucket']:<9}  "
              f"{r['execution_risk_score']:>5.1f}  "
              f"{r['stretch_rating'][:4]}/{r['extension_rating'][:4]}/"
              f"{r['momentum_intensity'][:4]}/{r['upside_room'][:4]}")

    # Per-date top picks (tag etiketli)
    print(f"\n═══ TOP-3 per day (+ exec_tag) ═══")
    for dt, g in vet.groupby(vet['date'].dt.date):
        print(f"\n  {dt}  (N={len(g)}, tags: {dict(g['exec_tag'].value_counts())})")
        for _, r in g.head(3).iterrows():
            print(f"    {r['ticker']:<7} {r['xu_regime']:<10} "
                  f"p_l3={r['p_l3']:.3f}  winR={r['winner_R_pred']:.2f}  "
                  f"pct={r['score_pct']:.2f}  "
                  f"[{r['exec_tag']}]")

    # Per-tag top picks (kullanıcı için direct okunur)
    print(f"\n═══ PER-TAG TOP-5 (target date {target_dt.date()}) ═══")
    target_sigs = vet[vet['date'] == target_dt]
    for tag in ['clean_watch', 'extended_watch', 'special_handling']:
        sub = target_sigs[target_sigs['exec_tag'] == tag]
        if sub.empty:
            continue
        print(f"\n  [{tag}]  N={len(sub)}")
        for _, r in sub.head(5).iterrows():
            print(f"    {r['ticker']:<7}  winR={r['winner_R_pred']:.2f}  "
                  f"pct={r['score_pct']:.2f}  "
                  f"[{r['risk_bucket']} / rs={r['execution_risk_score']:.1f}]  "
                  f"s={r['stretch_rating']}  e={r['extension_rating']}  "
                  f"m={r['momentum_intensity']}  r={r['upside_room']}")

    # Save
    out_path = f"output/nyxexp_scan_{target_dt.strftime('%Y%m%d')}.parquet"
    vet.to_parquet(out_path)
    print(f"\n[WRITE] {out_path}  ({len(vet)} signals, with exec_tag)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
