"""
nyxexpansion v1 — OOS evaluation (per-fold).

Girdi: output/nyxexp_train_v1.parquet (train.py çıktısı).
Rapor:
  1. Per-fold pos rate + trigger density + regime distribution
  2. Decile hit rate (L3 cont_10 ve P2 cont_10_struct)
  3. Top-decile MFE/MAE ortalama
  4. Model vs random vs rank baseline (top-decile lift)
  5. L3 → P2 gap (model top-decile'da structural label da pozitif mi?)
  6. Regime segmenti (uptrend / range / downtrend — eğer xu_regime varsa)
  7. Decile mean raw L1 mfe_mae_ratio_win (eval-only)

Kullanım:
    python -m nyxexpansion.evaluate
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

from nyxexpansion.config import CONFIG


def _fmt_pct(x: float, width: int = 6) -> str:
    return f"{x*100:{width}.1f}%" if np.isfinite(x) else "   NaN"


def _decile(s: pd.Series, q: int = 10) -> pd.Series:
    """Per-score decile label 0..q-1. NaN → NaN."""
    try:
        return pd.qcut(s, q, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series(np.nan, index=s.index)


def _decile_by_fold(df: pd.DataFrame, score_col: str, q: int = 10) -> pd.Series:
    """Her fold içinde ayrı qcut — fold'lar arası score dağılımı farkını nötralize et.
    Test pencereleri farklı base pos rate'e sahip; cross-fold qcut fold3'ü top-D'ye
    bastırabilir. Fold içi rank → merged yorum daha sağlıklı.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby('fold_assigned'):
        d = _decile(g[score_col], q=q)
        out.loc[g.index] = d.values
    return out


def per_fold_summary(df: pd.DataFrame, target: str, target_struct: str) -> None:
    """Fold başına pos rate + trigger density + model top decile."""
    print(f"\n═══ PER-FOLD OOS ═══")
    print(f"  {'fold':<7} {'N':>5} {'days':>5} {'sig/gün':>8} "
          f"{'L3 pos':>7} {'P2 pos':>7} {'L3 D10':>7} {'L3 lift':>8} "
          f"{'P2 D10':>7} {'P2 lift':>8}")
    for fold, g in df.groupby('fold_assigned'):
        g = g.copy()
        base_l3 = g[target].mean() if target in g.columns else np.nan
        base_p2 = g[target_struct].mean() if target_struct in g.columns else np.nan
        n_days = g['date'].dt.date.nunique()
        g['_dec'] = _decile(g['score_model'])
        top = g[g['_dec'] == g['_dec'].max()]
        l3_top = top[target].mean() if target in top.columns else np.nan
        p2_top = top[target_struct].mean() if target_struct in top.columns else np.nan
        lift_l3 = l3_top / base_l3 if base_l3 and np.isfinite(base_l3) else np.nan
        lift_p2 = p2_top / base_p2 if base_p2 and np.isfinite(base_p2) else np.nan
        print(f"  {fold:<7} {len(g):>5} {n_days:>5} {len(g)/max(n_days,1):>8.1f} "
              f"{_fmt_pct(base_l3):>7} {_fmt_pct(base_p2):>7} "
              f"{_fmt_pct(l3_top):>7} {lift_l3:>7.2f}x "
              f"{_fmt_pct(p2_top):>7} {lift_p2:>7.2f}x")


def decile_table(df: pd.DataFrame, score_col: str, target: str,
                 mfe_col: str, mae_col: str, label: str) -> None:
    """Decile-by-decile: N, pos rate, mean MFE, mean MAE."""
    g = df.copy()
    g['_dec'] = _decile_by_fold(g, score_col)
    g = g.dropna(subset=['_dec'])
    if len(g) == 0:
        print(f"  ({label}) veri yok")
        return
    rows = []
    for dec, sub in g.groupby('_dec'):
        rows.append({
            'dec': int(dec),
            'N': len(sub),
            'pos': sub[target].mean() if target in sub.columns else np.nan,
            'mfe_mean': sub[mfe_col].mean() if mfe_col in sub.columns else np.nan,
            'mae_mean': sub[mae_col].mean() if mae_col in sub.columns else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values('dec')
    print(f"\n[{label}]  score = {score_col}")
    print(f"  {'dec':>3} {'N':>5} {'pos':>7} {'MFE':>7} {'MAE':>7}")
    for _, r in tab.iterrows():
        print(f"  {int(r['dec']):>3} {int(r['N']):>5} "
              f"{_fmt_pct(r['pos']):>7} "
              f"{_fmt_pct(r['mfe_mean']):>7} {_fmt_pct(r['mae_mean']):>7}")


def top_decile_compare(df: pd.DataFrame, target: str) -> None:
    """Model vs random vs rank baseline — top decile lift."""
    print(f"\n═══ MODEL vs BASELINES (top-decile, all folds merged) ═══")
    base = df[target].mean()
    print(f"  overall pos:  {_fmt_pct(base)}")
    print(f"  {'score':<14} {'topD pos':>10} {'lift':>8} {'Q10/Q1':>8}")
    for s in ['score_model', 'score_rank', 'score_random']:
        if s not in df.columns:
            continue
        d = df.copy()
        d['_dec'] = _decile_by_fold(d, s)
        d = d.dropna(subset=['_dec'])
        if len(d) == 0:
            continue
        top = d[d['_dec'] == d['_dec'].max()][target].mean()
        bot = d[d['_dec'] == d['_dec'].min()][target].mean()
        lift = top / base if base else np.nan
        q_ratio = top / bot if bot and np.isfinite(bot) else np.nan
        print(f"  {s:<14} {_fmt_pct(top):>10} {lift:>7.2f}x "
              f"{q_ratio:>7.2f}x")


def regime_breakdown(df: pd.DataFrame, target: str) -> None:
    """Regime × model top decile pos."""
    if 'xu_regime' not in df.columns:
        return
    print(f"\n═══ REJİM × MODEL TOP DECILE ═══")
    d = df.copy()
    d['_dec'] = _decile_by_fold(d, 'score_model')
    d = d.dropna(subset=['_dec'])
    print(f"  {'regime':<12} {'N':>5} {'base pos':>10} {'topD pos':>10} {'lift':>8}")
    for reg, g in d.groupby(d['xu_regime'].fillna('unknown')):
        base = g[target].mean()
        top = g[g['_dec'] == g['_dec'].max()][target].mean()
        lift = top / base if base and np.isfinite(base) else np.nan
        print(f"  {reg:<12} {len(g):>5} {_fmt_pct(base):>10} "
              f"{_fmt_pct(top):>10} {lift:>7.2f}x")


def l3_p2_gap(df: pd.DataFrame, target: str, target_struct: str) -> None:
    """Model top decile L3=1 olanların P2=1 oranı — structural consistency."""
    print(f"\n═══ L3 → P2 GAP (model top decile'da L3=1 kümesinde P2=1) ═══")
    d = df.copy()
    d['_dec'] = _decile_by_fold(d, 'score_model')
    d = d.dropna(subset=['_dec'])
    top = d[d['_dec'] == d['_dec'].max()]
    if target_struct not in top.columns:
        print("  P2 yok.")
        return
    # Top decile L3 pos rate
    p_l3 = top[target].mean()
    p_p2 = top[target_struct].mean()
    only_l3 = top.loc[top[target] == 1, target_struct].mean()
    print(f"  Top-decile N: {len(top)}")
    print(f"  L3 pos   = {_fmt_pct(p_l3)}")
    print(f"  P2 pos   = {_fmt_pct(p_p2)}")
    print(f"  L3=1 iken P2=1 oranı: {_fmt_pct(only_l3)}")
    print(f"  (L3 model'i buluyorsa ama P2 düşükse structural risk yüksek demek)")


def score_bucket_mfe_mae(df: pd.DataFrame, mfe_col: str, mae_col: str) -> None:
    """Score bucket (deciles) başına mean MFE, MAE, L4 expansion skoru."""
    d = df.copy()
    d['_dec'] = _decile_by_fold(d, 'score_model')
    d = d.dropna(subset=['_dec'])
    print(f"\n═══ SCORE BUCKET × MFE / MAE (model) ═══")
    print(f"  {'dec':>3} {'N':>5} {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8}")
    for dec, g in d.groupby('_dec'):
        mfe = g[mfe_col].mean() if mfe_col in g.columns else np.nan
        mae = g[mae_col].mean() if mae_col in g.columns else np.nan
        ratio = mfe / mae if mae and np.isfinite(mae) else np.nan
        print(f"  {int(dec):>3} {len(g):>5} {_fmt_pct(mfe):>7} "
              f"{_fmt_pct(mae):>7} {ratio:>7.2f}")


def decile_mean_l1(df: pd.DataFrame) -> None:
    """Decile mean L1 mfe_mae_ratio_win."""
    if 'mfe_mae_ratio_win' not in df.columns:
        return
    d = df.copy()
    d['_dec'] = _decile_by_fold(d, 'score_model')
    d = d.dropna(subset=['_dec'])
    print(f"\n═══ DECILE × L1 mfe_mae_ratio_win (eval-only) ═══")
    print(f"  {'dec':>3} {'N':>5} {'ratio mean':>11} {'p50':>6}")
    for dec, g in d.groupby('_dec'):
        v = g['mfe_mae_ratio_win'].dropna()
        if not len(v):
            continue
        print(f"  {int(dec):>3} {len(g):>5} {v.mean():>11.2f} {v.median():>6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default="output/nyxexp_train_v1.parquet")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.preds).exists():
        print(f"❌ Preds yok: {args.preds}  →  python -m nyxexpansion.train")
        return 2

    h = CONFIG.label.primary_h
    target = f'cont_{h}'
    target_struct = f'cont_{h}_struct'
    mfe_col = f'mfe_{h}'
    mae_col = f'mae_{h}'

    df = pd.read_parquet(args.preds)
    df['date'] = pd.to_datetime(df['date'])

    print(f"═══ nyxexpansion Evaluate v1 ═══")
    print(f"Preds: {args.preds}  shape={df.shape}")
    print(f"Target: {target}  |  Struct: {target_struct}")

    # 1. Per-fold summary
    per_fold_summary(df, target, target_struct)

    # 2. Decile (L3, model score)
    decile_table(df, 'score_model', target, mfe_col, mae_col,
                 label="MODEL SCORE → L3 cont_10 decile")

    # 3. Decile (P2, model score)
    decile_table(df, 'score_model', target_struct, mfe_col, mae_col,
                 label="MODEL SCORE → P2 cont_10_struct decile")

    # 4. Baseline compare
    top_decile_compare(df, target)

    # 5. Regime segment
    regime_breakdown(df, target)

    # 6. L3 vs P2 gap
    l3_p2_gap(df, target, target_struct)

    # 7. Score bucket × MFE/MAE
    score_bucket_mfe_mae(df, mfe_col, mae_col)

    # 8. Decile mean L1
    decile_mean_l1(df)

    return 0


if __name__ == '__main__':
    sys.exit(main())
