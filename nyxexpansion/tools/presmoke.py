"""
Step 0 — Pre-smoke for nyxexpansion.

Amaç: feature/model yazmadan ÖNCE trigger'ın sağlıklı mı görmek.
- Universe ve tarih aralığı
- Toplam sinyal sayısı + per-ticker dağılım
- Rejime göre sinyal yoğunluğu (XU100 trend state)
- Raw 3/5/7/10 bar forward MFE + MAE dağılımı
- L3 cont_10 (primary label) pozitif oranı — feature yazmadan önce baseline

Makro YOK, feature YOK. Yalnızca trigger + forward path.

Kullanım:
    python -m nyxexpansion.tools.presmoke
    python -m nyxexpansion.tools.presmoke --refresh-xu100    # XU100'ü yfinance'ten çek
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Proje kökü path'e ekle
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from nyxexpansion.config import CONFIG
from nyxexpansion.trigger import compute_trigger_a_panel
from nyxexpansion.labels import compute_labels_on_panel


def load_ohlcv_cache(path: str) -> dict[str, pd.DataFrame]:
    """yf_price_cache.parquet → {ticker: OHLCV DataFrame (DatetimeIndex)}."""
    df = pd.read_parquet(path)
    if 'ticker' not in df.columns:
        raise RuntimeError(f"'{path}' içinde 'ticker' kolonu yok. Kolonlar: {list(df.columns)}")
    out: dict[str, pd.DataFrame] = {}
    # df.index → Date
    for ticker, sub in df.groupby('ticker', sort=False):
        s = sub.drop(columns=['ticker']).copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        # Tekrarlanan tarih varsa son kaydı al
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep='last')]
        out[ticker] = s
    return out


def load_xu100(refresh: bool = False, cache_path: str = "output/xu100_cache.parquet",
               period: str = "2y") -> pd.DataFrame | None:
    """XU100 Close (ffill'li DatetimeIndex)."""
    p = Path(cache_path)
    if p.exists() and not refresh:
        try:
            xu = pd.read_parquet(cache_path)
            if 'Close' in xu.columns and len(xu) > 50:
                xu.index = pd.to_datetime(xu.index)
                return xu
        except Exception:
            pass
    try:
        import yfinance as yf
        raw = yf.download("XU100.IS", period=period, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw[['Open', 'High', 'Low', 'Close', 'Volume']].to_parquet(cache_path)
        return raw
    except Exception as e:
        print(f"  [!] XU100 fetch hata: {e}")
        return None


def classify_xu100_regime(xu: pd.DataFrame) -> pd.Series:
    """Basit 3-sınıf rejim: uptrend / range / downtrend (EMA21 + EMA55 slope)."""
    close = xu['Close'].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    slope21 = ema21.pct_change(10)
    regime = pd.Series('range', index=close.index, dtype=object)
    up_mask = (close > ema21) & (ema21 > ema55) & (slope21 > 0)
    dn_mask = (close < ema21) & (ema21 < ema55) & (slope21 < 0)
    regime.loc[up_mask] = 'uptrend'
    regime.loc[dn_mask] = 'downtrend'
    return regime


def _fmt_pct(x: float) -> str:
    return f"{x*100:>6.1f}%" if np.isfinite(x) else "   NaN"


def _describe(series: pd.Series, name: str) -> str:
    s = series.dropna()
    if len(s) == 0:
        return f"  {name:<20} N=0"
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    return (
        f"  {name:<20} N={len(s):>5}  "
        f"mean={_fmt_pct(s.mean())}  "
        f"p10={_fmt_pct(q[0])}  p25={_fmt_pct(q[1])}  "
        f"p50={_fmt_pct(q[2])}  p75={_fmt_pct(q[3])}  p90={_fmt_pct(q[4])}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CONFIG.data.yf_cache_path)
    ap.add_argument('--refresh-xu100', action='store_true')
    ap.add_argument('--min-bars', type=int, default=CONFIG.data.min_bars_per_ticker)
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    cache_path = args.cache
    if not Path(cache_path).exists():
        print(f"❌ Cache yok: {cache_path}")
        return 2

    print(f"═══ nyxexpansion Step 0 Pre-smoke ═══")
    print(f"Cache: {cache_path}")

    data = load_ohlcv_cache(cache_path)
    # Min bar filtresi
    usable = {t: d for t, d in data.items() if len(d) >= args.min_bars}
    total_tickers = len(data)
    usable_n = len(usable)

    # Tarih aralığı
    all_dates = pd.Index([])
    for d in usable.values():
        all_dates = all_dates.union(d.index)
    all_dates = pd.DatetimeIndex(all_dates).sort_values()
    date_min = all_dates.min().strftime('%Y-%m-%d') if len(all_dates) else 'NA'
    date_max = all_dates.max().strftime('%Y-%m-%d') if len(all_dates) else 'NA'
    trading_days = len(all_dates)

    print(f"\n[UNIVERSE]")
    print(f"  Toplam ticker:    {total_tickers}")
    print(f"  Kullanılabilir:   {usable_n}  (≥{args.min_bars} bar)")
    print(f"  Tarih aralığı:    {date_min} → {date_max}")
    print(f"  Trading günü:     {trading_days}")

    # Trigger A
    print(f"\n[TRIGGER A]  close>prior20H AND rvol≥{CONFIG.trigger.rvol_min} "
          f"AND close_loc≥{CONFIG.trigger.close_loc_min}")
    panel = compute_trigger_a_panel(usable, CONFIG.trigger)
    n_sig = len(panel)
    if n_sig == 0:
        print("  ❌ Hiç sinyal yok.")
        return 1

    n_unique_tickers = panel['ticker'].nunique()
    sig_per_day = n_sig / max(trading_days, 1)
    sig_per_ticker = n_sig / max(n_unique_tickers, 1)
    print(f"  Toplam sinyal:          {n_sig:,}")
    print(f"  Sinyal veren ticker:    {n_unique_tickers} / {usable_n} ({100*n_unique_tickers/usable_n:.0f}%)")
    print(f"  Sinyal / gün (ort):     {sig_per_day:.1f}")
    print(f"  Sinyal / ticker (ort):  {sig_per_ticker:.1f}")

    # Per-ticker dist
    per_t = panel.groupby('ticker').size().sort_values(ascending=False)
    print(f"\n  Per-ticker sinyal dağılımı:")
    print(f"    p10 / p50 / p90 / max : "
          f"{int(per_t.quantile(0.1))} / {int(per_t.quantile(0.5))} / "
          f"{int(per_t.quantile(0.9))} / {int(per_t.max())}")
    print(f"    Top 10:  " + ", ".join(
        f"{t}({n})" for t, n in per_t.head(10).items()
    ))

    # Aylık sinyal yoğunluğu
    panel['year_month'] = panel['date'].dt.to_period('M').astype(str)
    monthly = panel.groupby('year_month').size()
    print(f"\n  Aylık sinyal yoğunluğu:")
    for ym, n in monthly.items():
        bar = '█' * int(round(n / max(monthly.max(), 1) * 30))
        print(f"    {ym}  {n:>4}  {bar}")

    # XU100 regime
    print(f"\n[XU100 REGIME]")
    xu = load_xu100(refresh=args.refresh_xu100)
    if xu is None or xu.empty:
        print("  ⚠️ XU100 yüklenemedi, rejim segmenti atlanıyor.")
    else:
        regime = classify_xu100_regime(xu)
        # Panel'e join (signal_date → regime)
        panel_r = panel.merge(
            regime.rename('xu_regime').to_frame(),
            left_on='date', right_index=True, how='left',
        )
        reg_counts = panel_r['xu_regime'].fillna('unknown').value_counts()
        print(f"  Rejim başına sinyal sayısı:")
        for r, n in reg_counts.items():
            print(f"    {r:<10} {n:>5}  ({100*n/len(panel_r):.1f}%)")
        # Rejime göre forward MFE kıyası için panel_r sakla
    panel = panel.drop(columns=['year_month'], errors='ignore')

    # Labels
    print(f"\n[LABELS] MFE/MAE + L3 cont_10 (primary) hesaplanıyor...")
    labeled = compute_labels_on_panel(panel, usable, CONFIG.label)

    horizons = CONFIG.label.horizons
    print(f"\n[RAW FORWARD MFE — signal_date başından]")
    for h in horizons:
        col = f'mfe_{h}'
        if col in labeled.columns:
            print(_describe(labeled[col], col))

    print(f"\n[RAW FORWARD MAE]")
    for h in horizons:
        col = f'mae_{h}'
        if col in labeled.columns:
            print(_describe(labeled[col], col))

    # L3 cont_{primary_h}
    h = CONFIG.label.primary_h
    col = f'cont_{h}'
    if col in labeled.columns:
        v = labeled[col].dropna()
        pos = v.mean() if len(v) else float('nan')
        print(f"\n[L3 cont_{h}]  (MFE≥{CONFIG.label.cont_mfe_atr_mult}×ATR AND "
              f"MAE≤{CONFIG.label.cont_mae_atr_mult}×ATR)")
        print(f"  N valid:        {len(v):,} / {len(labeled):,}")
        print(f"  Pozitif oran:   {pos*100:.1f}%  (primary training target)")

    # L3 struct
    col_s = f'cont_{h}_struct'
    if col_s in labeled.columns:
        vs = labeled[col_s].dropna()
        pos_s = vs.mean() if len(vs) else float('nan')
        print(f"\n[P2 cont_{h}_struct]  (research — structural risk_unit)")
        print(f"  N valid:        {len(vs):,}")
        print(f"  Pozitif oran:   {pos_s*100:.1f}%")

    # L2 follow_through_3
    if 'follow_through_3' in labeled.columns:
        v2 = labeled['follow_through_3'].dropna()
        pos2 = v2.mean() if len(v2) else float('nan')
        print(f"\n[L2 follow_through_3]  (eval-only — noisy)")
        print(f"  N valid:        {len(v2):,}")
        print(f"  Pozitif oran:   {pos2*100:.1f}%")

    # L1 mfe_mae_ratio
    if 'mfe_mae_ratio_raw' in labeled.columns:
        raw = labeled['mfe_mae_ratio_raw'].replace([np.inf, -np.inf], np.nan).dropna()
        win = labeled['mfe_mae_ratio_win'].dropna()
        print(f"\n[L1 mfe_mae_ratio]  (h={h})")
        if len(raw):
            q = raw.quantile([0.1, 0.5, 0.9, 0.99]).values
            print(f"  raw         N={len(raw):>5}  "
                  f"mean={raw.mean():.2f}  p50={q[1]:.2f}  p90={q[2]:.2f}  p99={q[3]:.2f}  "
                  f"max={raw.max():.2f}")
        if len(win):
            q = win.quantile([0.1, 0.5, 0.9]).values
            print(f"  winsor@{CONFIG.label.mfe_mae_ratio_clip:.1f} N={len(win):>5}  "
                  f"mean={win.mean():.2f}  p50={q[1]:.2f}  p90={q[2]:.2f}")

    # Regime × cont
    if xu is not None and not xu.empty and f'cont_{h}' in labeled.columns:
        regime = classify_xu100_regime(xu)
        lbl_r = labeled.merge(regime.rename('xu_regime').to_frame(),
                              left_on='date', right_index=True, how='left')
        grp = lbl_r.groupby(lbl_r['xu_regime'].fillna('unknown'))[f'cont_{h}'].agg(['count', 'mean'])
        print(f"\n[REJİM × cont_{h} pozitif oranı]")
        for r, row in grp.iterrows():
            cnt = int(row['count'])
            mean_v = row['mean']
            mean_str = f"{mean_v*100:>5.1f}%" if pd.notna(mean_v) else "  NaN"
            print(f"    {r:<10} N={cnt:>5}  pos_rate={mean_str}")

    print(f"\n═══ Step 0 bitti ═══")
    return 0


if __name__ == '__main__':
    sys.exit(main())
