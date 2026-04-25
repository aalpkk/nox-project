"""
nyxexpansion dataset builder.

Trigger A signal panel + core 26 features + L3/P2/L1 labels + fold tag.
Output: `output/nyxexp_dataset_v1.parquet` — single long panel.

Leakage kuralları:
  - Feature'lar signal_date bar'ına kadar (today inclusive ama forward YOK).
  - Label pencereleri [t+1..t+h] → explicit forward.
  - Train/val/test assignment: SplitParams'tan tek seferlik etiketleme,
    hyperparam tuning sırasında DEĞİŞMEZ.
  - Embargo val_start−train_end ≥ 15 takvim günü (10 işgünü + weekend buffer),
    label horizon 10 bar için yeterli.

Kullanım:
    python -m nyxexpansion.dataset  [--cache path]  [--min-bars 80]
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

from nyxexpansion.config import CONFIG, FoldSpec, SplitParams
from nyxexpansion.trigger import compute_trigger_a_panel
from nyxexpansion.labels import compute_labels_on_panel
from nyxexpansion.features import build_feature_panel, CORE_FEATURES
from nyxexpansion.tools.presmoke import (
    load_ohlcv_cache, load_xu100, classify_xu100_regime,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fold assignment
# ═════════════════════════════════════════════════════════════════════════════

def assign_fold_tag(date: pd.Series, split: SplitParams) -> pd.DataFrame:
    """Her satıra (fold_name, split_name) etiketi.

    Dönen DataFrame: kolonlar ['fold', 'split']. Aynı index/uzunluk.
    Bir satır birden fazla fold'un test'ine düşemez (fold pencereleri çakışmaz).
    Val pencerleri de çakışmaz. Train expanding — bir satır 3 fold'un
    train'ine de düşebilir. Aşağıda satıra ATANAN fold = ilk denk geldiği
    test pencere'dir; değilse ilk denk geldiği val pencere; değilse tüm
    fold'larda train.

    fold = 'fold1' | 'fold2' | 'fold3' | 'none' (hiçbirine uymuyor — embargo bölgesi)
    split = 'train' | 'val' | 'test' | 'embargo'
    """
    d = pd.to_datetime(date)
    fold_out = pd.Series('none', index=d.index, dtype=object)
    split_out = pd.Series('embargo', index=d.index, dtype=object)

    # Test penceresi öncelikli (en kıymetli alan, çakışmaz)
    for fs in split.folds:
        ts = pd.Timestamp(fs.test_start)
        te = pd.Timestamp(fs.test_end)
        mask = (d >= ts) & (d <= te)
        fold_out.loc[mask] = fs.name
        split_out.loc[mask] = 'test'

    # Val penceresi (test dışındaki satırlar için)
    untagged = split_out == 'embargo'
    for fs in split.folds:
        vs = pd.Timestamp(fs.val_start)
        ve = pd.Timestamp(fs.val_end)
        mask = untagged & (d >= vs) & (d <= ve)
        fold_out.loc[mask] = fs.name
        split_out.loc[mask] = 'val'
        untagged = split_out == 'embargo'

    # Train: train_start .. train_end — her fold için AYNI satırı farklı fold'un
    # train'ine dahil etmek istiyoruz → train satırları fold bazında ayrı tutulur.
    # Bunun için 'train' etiketini fold3 (en geniş) için atıyoruz; fold filtrelemesi
    # için eğitim sırasında split.folds.<train_end'e göre filtre uygulanır.
    train_start = pd.Timestamp(split.train_start)
    # En geniş fold'un train_end'ini al (fold3)
    latest_te = max(pd.Timestamp(fs.train_end) for fs in split.folds)
    train_mask = untagged & (d >= train_start) & (d <= latest_te)
    split_out.loc[train_mask] = 'train'
    # fold_out train için 'none' kalır — fold filtre train_end'e göre yapılır.

    return pd.DataFrame({'fold': fold_out.values, 'split': split_out.values},
                        index=d.index)


# ═════════════════════════════════════════════════════════════════════════════
# Build
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(
    cache_path: str,
    min_bars: int,
    xu100_refresh: bool = False,
    out_path: str = "output/nyxexp_dataset_v1.parquet",
) -> pd.DataFrame:
    """Ana pipeline: data → trigger → features → labels → fold assignment → parquet."""
    # 1. Load data
    print(f"[DATA] {cache_path}")
    data = load_ohlcv_cache(cache_path)
    usable = {t: d for t, d in data.items() if len(d) >= min_bars}
    print(f"  Ticker: {len(usable)} / {len(data)} (≥{min_bars} bar)")
    date_min = min(d.index.min() for d in usable.values())
    date_max = max(d.index.max() for d in usable.values())
    print(f"  Tarih: {date_min.date()} → {date_max.date()}")

    # 2. Trigger
    print(f"[TRIGGER A]")
    panel = compute_trigger_a_panel(usable, CONFIG.trigger)
    print(f"  Sinyal: {len(panel):,}")
    if panel.empty:
        raise RuntimeError("Hiç sinyal yok — cache/min_bars kontrol et")

    # 3. XU100 (regime + rs için)
    xu = load_xu100(refresh=xu100_refresh, period="6y")
    xu_close = xu['Close'] if xu is not None and not xu.empty else None
    if xu_close is not None:
        print(f"  XU100: {len(xu_close)} gün ({xu_close.index.min().date()} → "
              f"{xu_close.index.max().date()})")

    # 4. Features
    print(f"[FEATURES] core {len(CORE_FEATURES)} hesaplanıyor...")
    panel_f = build_feature_panel(panel, usable, xu100_close=xu_close)
    print(f"  Feature panel shape: {panel_f.shape}")

    # 5. Labels
    print(f"[LABELS]")
    labeled = compute_labels_on_panel(panel, usable, CONFIG.label)
    h = CONFIG.label.primary_h
    keep_cols = [
        'ticker', 'date',
        f'atr_{CONFIG.label.atr_window}', 'close_0',
        *[f'mfe_{hh}' for hh in CONFIG.label.horizons],
        *[f'mae_{hh}' for hh in CONFIG.label.horizons],
        'mfe_mae_ratio_raw', 'mfe_mae_ratio_win',
        'follow_through_3', f'cont_{h}', f'expansion_score_{h}',
        f'cont_{h}_struct', 'risk_unit_struct_pct',
    ]
    keep_cols = [c for c in keep_cols if c in labeled.columns]
    labels_small = labeled[keep_cols]

    merged = panel_f.merge(labels_small, on=['ticker', 'date'], how='left',
                           suffixes=('', '_lbl'))
    print(f"  Merged shape: {merged.shape}")

    # 6. Regime (analiz kolonu — model'e girmez doğrudan; xu100_trend_score girer)
    if xu is not None and not xu.empty:
        regime = classify_xu100_regime(xu)
        merged['xu_regime'] = pd.to_datetime(merged['date']).map(regime).values

    # 7. Fold assignment
    fold_df = assign_fold_tag(merged['date'], CONFIG.split)
    merged['fold'] = fold_df['fold'].values
    merged['split'] = fold_df['split'].values

    # 8. Sort & save
    merged = merged.sort_values(['date', 'ticker']).reset_index(drop=True)

    # Quick summary
    print(f"\n[SPLIT SUMMARY]")
    print(f"  {'split':<10} {'N':>6} {'cont_10 pos':>13} {'P2 pos':>10}")
    for sp_name in ['train', 'val', 'test', 'embargo']:
        s = merged[merged['split'] == sp_name]
        if len(s) == 0:
            continue
        pos_l3 = s[f'cont_{h}'].dropna().mean() if f'cont_{h}' in s.columns else np.nan
        pos_p2 = (s[f'cont_{h}_struct'].dropna().mean()
                  if f'cont_{h}_struct' in s.columns else np.nan)
        pos_l3_str = f"{pos_l3*100:>5.1f}%" if np.isfinite(pos_l3) else '   NA'
        pos_p2_str = f"{pos_p2*100:>5.1f}%" if np.isfinite(pos_p2) else '   NA'
        print(f"  {sp_name:<10} {len(s):>6,} {pos_l3_str:>13} {pos_p2_str:>10}")

    print(f"\n[SPLIT × FOLD] (test penceresi)")
    tst = merged[merged['split'] == 'test']
    for fs in CONFIG.split.folds:
        sf = tst[tst['fold'] == fs.name]
        if len(sf) == 0:
            continue
        pos = sf[f'cont_{h}'].dropna().mean() if f'cont_{h}' in sf.columns else np.nan
        d0 = sf['date'].min()
        d1 = sf['date'].max()
        print(f"  {fs.name}  N={len(sf):>5,}  {d0.date()} → {d1.date()}  "
              f"cont_{h} pos={pos*100:.1f}%")

    # NaN rate on core features
    print(f"\n[CORE FEATURE NaN%]")
    for f in CORE_FEATURES:
        if f in merged.columns:
            nan_pct = merged[f].isna().mean()
            if nan_pct > 0.10:
                print(f"  ⚠️  {f:<32} {nan_pct*100:>5.1f}%")

    # Save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(f"\n[WRITE] {out_path}  ({out.stat().st_size / 1e6:.1f} MB)")

    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CONFIG.data.yf_cache_path)
    ap.add_argument('--min-bars', type=int, default=CONFIG.data.min_bars_per_ticker)
    ap.add_argument('--refresh-xu100', action='store_true')
    ap.add_argument('--out', default="output/nyxexp_dataset_v1.parquet")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.cache).exists():
        print(f"❌ Cache yok: {args.cache}")
        return 2

    build_dataset(
        cache_path=args.cache,
        min_bars=args.min_bars,
        xu100_refresh=args.refresh_xu100,
        out_path=args.out,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
