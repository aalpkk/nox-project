#!/usr/bin/env python3
"""
Feature audit for the per-ticker output of ml/features.py.

Builds a multi-ticker panel, runs compute_all_features, optionally runs
compute_cross_sectional_features when available, then reports:

  * NaN ratio per feature (top-30 worst)
  * inf / -inf counts (any non-zero)
  * Constant / near-constant features (std==0 or unique<=2 ignoring NaN)
  * High-correlation pairs (|corr| > 0.95) within numeric features
  * Group count from feature_registry
  * known_at distribution
  * Cross-check with F0 / F1 / F2 whitelists
  * Features in panel but missing from registry (UNKNOWN)

Output:
  - Console summary
  - Markdown report at output/feature_audit_v0.md

Usage:
  python tools/feature_audit.py \\
      --ohlcv output/ohlcv_6y_fintables.parquet \\
      --xu100 output/xu100_extfeed_daily.parquet \\
      --sector-map tools/sector_map.json \\
      --tickers GARAN,AKBNK,TUPRS,EREGL,BIMAS,SAHOL,SISE,YKBNK,KOZAL,FROTO \\
      --tail-bars 200 \\
      --out output/feature_audit_v0.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import compute_all_features  # noqa: E402

try:
    from ml.features import compute_cross_sectional_features  # noqa: E402
    _HAS_CS = True
except ImportError:
    compute_cross_sectional_features = None
    _HAS_CS = False

from ml.feature_registry import (  # noqa: E402
    build_feature_manifest,
    F0_CORE_EXISTING,
    F1_CONTEXT_ADDED,
    F2_FULL_EXPERIMENTAL,
    T_CLOSE_FEATURES,
    T1_OPEN_FEATURES,
    EXCLUDE_FEATURES,
)


DEFAULT_TICKERS = [
    'GARAN', 'AKBNK', 'TUPRS', 'EREGL', 'BIMAS',
    'SAHOL', 'SISE', 'YKBNK', 'KOZAL', 'FROTO',
]


def build_panel(ohlcv_path, xu_path, tickers, tail_bars):
    p = pd.read_parquet(ohlcv_path)
    if 'ticker' not in p.columns:
        raise ValueError(f"{ohlcv_path}: expected 'ticker' column")
    if not isinstance(p.index, pd.DatetimeIndex):
        if 'Date' in p.columns:
            p = p.set_index('Date')
        else:
            raise ValueError(f"{ohlcv_path}: expected DatetimeIndex or 'Date' column")

    xu = pd.read_parquet(xu_path) if xu_path else None

    panels = []
    skipped = []
    for t in tickers:
        g = p[p['ticker'] == t].copy()
        g = g.sort_index()
        if len(g) < 80:
            skipped.append((t, len(g)))
            continue
        f = compute_all_features(g, xu_df=xu)
        if tail_bars and tail_bars > 0:
            f = f.tail(tail_bars).copy()
        f['ticker'] = t
        f = f.reset_index()
        idx_name = f.columns[0]
        f = f.rename(columns={idx_name: 'date'})
        panels.append(f)

    if not panels:
        raise RuntimeError(f"No tickers produced a panel (skipped={skipped})")
    panel = pd.concat(panels, ignore_index=True)
    return panel, skipped


def audit_nan(panel, top_n=30):
    n = len(panel)
    nan_ratio = panel.isna().sum() / max(n, 1)
    nan_ratio = nan_ratio.sort_values(ascending=False)
    return nan_ratio, nan_ratio.head(top_n)


def audit_inf(panel):
    num = panel.select_dtypes(include=[np.number])
    pos = np.isposinf(num).sum()
    neg = np.isneginf(num).sum()
    pos = pos[pos > 0]
    neg = neg[neg > 0]
    return pos, neg


def audit_constant(panel, top_n=30):
    num = panel.select_dtypes(include=[np.number])
    nunique = num.nunique(dropna=True)
    std = num.std(skipna=True)
    constant = nunique[nunique <= 1].index.tolist()
    near_constant = nunique[(nunique > 1) & (nunique <= 2)].index.tolist()
    zero_std = std[std == 0].index.tolist()
    out = pd.DataFrame({
        'nunique': nunique,
        'std': std,
    }).sort_values('nunique')
    return constant, near_constant, zero_std, out.head(top_n)


def audit_correlation(panel, threshold=0.95, top_n=30):
    num = panel.select_dtypes(include=[np.number])
    num = num.loc[:, num.std(skipna=True).fillna(0) > 0]
    if num.shape[0] > 20000:
        num = num.sample(n=20000, random_state=42)
    corr = num.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.at[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append((a, b, v))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs[:top_n], len(pairs)


def write_report(out_path, panel, manifest, summary):
    lines = []
    lines.append("# Feature audit — v0")
    lines.append("")
    lines.append(f"- panel rows: **{summary['panel_rows']:,}**")
    lines.append(f"- panel columns: **{summary['panel_cols']:,}**")
    lines.append(f"- tickers: **{summary['n_tickers']}** ({', '.join(summary['tickers'])})")
    if summary['skipped']:
        sk = ', '.join(f"{t}({n})" for t, n in summary['skipped'])
        lines.append(f"- skipped (insufficient bars): {sk}")
    lines.append(f"- date range: **{summary['date_min']} → {summary['date_max']}**")
    if summary.get('cs_skipped'):
        lines.append(
            "- cross-sectional aggregator: **not available** "
            "(compute_cross_sectional_features not yet shipped)"
        )
    lines.append("")

    lines.append("## Registry coverage")
    lines.append("")
    lines.append(f"- features classified: **{summary['classified']}** / {summary['panel_cols']}")
    lines.append(f"- UNKNOWN (not in registry, not date/ticker): **{summary['unknown_count']}**")
    if summary['unknown_features']:
        lines.append("  - sample: " + ", ".join(summary['unknown_features'][:20]))
    lines.append("")
    lines.append("### Group distribution")
    lines.append("")
    lines.append("| group | count |")
    lines.append("|---|---:|")
    for g, c in summary['group_counts'].items():
        lines.append(f"| {g} | {c} |")
    lines.append("")
    lines.append("### `known_at` distribution")
    lines.append("")
    lines.append("| known_at | count |")
    lines.append("|---|---:|")
    for k, c in summary['known_at_counts'].items():
        lines.append(f"| {k} | {c} |")
    lines.append("")

    lines.append("### Whitelist coverage (manifest vs. panel)")
    lines.append("")
    lines.append("| whitelist | defined | present | missing |")
    lines.append("|---|---:|---:|---:|")
    for name, (defined, present, missing) in summary['whitelist'].items():
        lines.append(f"| {name} | {defined} | {present} | {missing} |")
    lines.append("")

    lines.append("## NaN / inf")
    lines.append("")
    lines.append(f"- features with NaN ratio > 0.50: **{summary['nan_gt_50']}**")
    lines.append(f"- features with NaN ratio > 0.90: **{summary['nan_gt_90']}**")
    lines.append("")
    lines.append("### Top-30 NaN ratio")
    lines.append("")
    lines.append("| feature | nan_ratio |")
    lines.append("|---|---:|")
    for k, v in summary['top_nan'].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    if summary['inf_pos'] or summary['inf_neg']:
        lines.append("### Infinities (any non-zero count)")
        lines.append("")
        lines.append("| feature | +inf | -inf |")
        lines.append("|---|---:|---:|")
        all_inf = set(summary['inf_pos']) | set(summary['inf_neg'])
        for k in sorted(all_inf):
            lines.append(
                f"| {k} | {summary['inf_pos'].get(k, 0)} | {summary['inf_neg'].get(k, 0)} |"
            )
        lines.append("")
    else:
        lines.append("### Infinities")
        lines.append("")
        lines.append("- none.")
        lines.append("")

    lines.append("## Constant / near-constant")
    lines.append("")
    lines.append(f"- exactly constant (nunique<=1): **{len(summary['constant'])}**")
    if summary['constant']:
        lines.append("  - " + ", ".join(summary['constant'][:30]))
    lines.append(f"- near-constant (nunique==2): **{len(summary['near_constant'])}**")
    if summary['near_constant']:
        lines.append("  - " + ", ".join(summary['near_constant'][:30]))
    lines.append(f"- zero-std numeric: **{len(summary['zero_std'])}**")
    if summary['zero_std']:
        lines.append("  - " + ", ".join(summary['zero_std'][:30]))
    lines.append("")

    lines.append("## High-correlation pairs (|corr| ≥ 0.95)")
    lines.append("")
    lines.append(f"- total pairs above threshold: **{summary['n_corr_pairs']}**")
    lines.append("")
    if summary['top_corr']:
        lines.append("| feature_a | feature_b | corr |")
        lines.append("|---|---|---:|")
        for a, b, v in summary['top_corr']:
            lines.append(f"| {a} | {b} | {v:+.3f} |")
    else:
        lines.append("- none above threshold.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv', default='output/ohlcv_6y_fintables.parquet')
    ap.add_argument('--xu100', default='output/xu100_extfeed_daily.parquet')
    ap.add_argument('--sector-map', default='tools/sector_map.json')
    ap.add_argument('--tickers', default=','.join(DEFAULT_TICKERS))
    ap.add_argument('--tail-bars', type=int, default=200)
    ap.add_argument('--out', default='output/feature_audit_v0.md')
    ap.add_argument('--corr-threshold', type=float, default=0.95)
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    print(f"[audit] panel: {args.ohlcv}", flush=True)
    print(f"[audit] xu100: {args.xu100}", flush=True)
    print(f"[audit] tickers ({len(tickers)}): {', '.join(tickers)}", flush=True)
    print(f"[audit] tail_bars: {args.tail_bars}", flush=True)

    panel, skipped = build_panel(args.ohlcv, args.xu100, tickers, args.tail_bars)
    print(f"[audit] pre-CS panel: {panel.shape}", flush=True)

    cs_skipped = False
    if _HAS_CS:
        panel = compute_cross_sectional_features(panel, sector_map_path=args.sector_map)
        print(f"[audit] post-CS panel: {panel.shape}", flush=True)
    else:
        cs_skipped = True
        print(
            "[audit] compute_cross_sectional_features not available — "
            "skipping CS aggregator step.",
            flush=True,
        )

    manifest = build_feature_manifest(list(panel.columns))
    classified = (manifest['group'] != 'UNKNOWN').sum()
    unknown_mask = manifest['group'] == 'UNKNOWN'
    real_unknown = manifest[unknown_mask & ~manifest['feature_name'].isin(['date', 'ticker'])]

    nan_ratio_full, top_nan = audit_nan(panel, top_n=30)
    nan_gt_50 = int((nan_ratio_full > 0.50).sum())
    nan_gt_90 = int((nan_ratio_full > 0.90).sum())
    inf_pos, inf_neg = audit_inf(panel)
    constant, near_constant, zero_std, _ = audit_constant(panel, top_n=30)
    top_corr, n_corr_pairs = audit_correlation(
        panel, threshold=args.corr_threshold, top_n=30
    )

    cols = set(panel.columns)
    whitelist = {
        'F0_CORE_EXISTING':     (len(F0_CORE_EXISTING),     len(F0_CORE_EXISTING & cols),     len(F0_CORE_EXISTING - cols)),
        'F1_CONTEXT_ADDED':     (len(F1_CONTEXT_ADDED),     len(F1_CONTEXT_ADDED & cols),     len(F1_CONTEXT_ADDED - cols)),
        'F2_FULL_EXPERIMENTAL': (len(F2_FULL_EXPERIMENTAL), len(F2_FULL_EXPERIMENTAL & cols), len(F2_FULL_EXPERIMENTAL - cols)),
        'T_CLOSE_FEATURES':     (len(T_CLOSE_FEATURES),     len(T_CLOSE_FEATURES & cols),     len(T_CLOSE_FEATURES - cols)),
        'T1_OPEN_FEATURES':     (len(T1_OPEN_FEATURES),     len(T1_OPEN_FEATURES & cols),     len(T1_OPEN_FEATURES - cols)),
        'EXCLUDE_FEATURES':     (len(EXCLUDE_FEATURES),     len(EXCLUDE_FEATURES & cols),     len(EXCLUDE_FEATURES - cols)),
    }

    summary = {
        'panel_rows': len(panel),
        'panel_cols': panel.shape[1],
        'n_tickers': len(tickers) - len(skipped),
        'tickers': [t for t in tickers if t not in {x for x, _ in skipped}],
        'skipped': skipped,
        'date_min': str(panel['date'].min().date()) if 'date' in panel.columns else 'N/A',
        'date_max': str(panel['date'].max().date()) if 'date' in panel.columns else 'N/A',
        'cs_skipped': cs_skipped,
        'classified': int(classified),
        'unknown_count': len(real_unknown),
        'unknown_features': real_unknown['feature_name'].tolist(),
        'group_counts': manifest['group'].value_counts().to_dict(),
        'known_at_counts': manifest['known_at'].value_counts().to_dict(),
        'whitelist': whitelist,
        'nan_gt_50': nan_gt_50,
        'nan_gt_90': nan_gt_90,
        'top_nan': top_nan.to_dict(),
        'inf_pos': inf_pos.to_dict(),
        'inf_neg': inf_neg.to_dict(),
        'constant': constant,
        'near_constant': near_constant,
        'zero_std': zero_std,
        'top_corr': top_corr,
        'n_corr_pairs': n_corr_pairs,
    }

    out_path = Path(args.out)
    write_report(out_path, panel, manifest, summary)

    print()
    print(f"[audit] classified: {classified} / {panel.shape[1]}  unknown: {len(real_unknown)}")
    print(f"[audit] NaN ratio > 0.50: {nan_gt_50}   > 0.90: {nan_gt_90}")
    print(f"[audit] constant: {len(constant)}   near-constant: {len(near_constant)}   zero-std: {len(zero_std)}")
    print(f"[audit] |corr|≥{args.corr_threshold} pairs: {n_corr_pairs}")
    print(f"[audit] report: {out_path}")


if __name__ == '__main__':
    main()
