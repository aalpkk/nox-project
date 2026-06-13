"""
Sektör/piyasa endeksleri günlük CLOSE master'ı — İş Yatırım IndexHistoricalAll.

Endeks başına TEK REST isteğiyle tam tarihçe çekilir (close-only; endekslerde
gerçek işlem hacmi yok, OHLC gerekirse Fintables MCP'den 300-satır sayfalı
alınabilir). Nokta çapraz doğrulama Fintables `endeks_mumlar_gunluk_gh` ile
yapıldı: XBANK 2005-03-08 close 510.67 / 2005-03-09 close 503.17 birebir.

Çıktı: output/sector_index_daily_master.parquet  (kolonlar: date, code, close)

Kullanım:
    python tools/sector_index_master_build.py [--lookback-days 8200] [--groups sektor,piyasa,tematik,katilim]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from agent.sector_regime import _ALL_INDICES, _fetch_isy_single

OUT_PATH = os.path.join(ROOT, 'output', 'sector_index_daily_master.parquet')

# Fintables endeks_mumlar_gunluk_gh ile nokta teyit (kod, tarih, kapanış)
CROSS_CHECK = [
    ('XBANK', '2005-03-08', 510.67),
    ('XBANK', '2005-03-09', 503.17),
]


def _splice_rebase(s: pd.Series, code: str, thresh: float = 0.5) -> pd.Series:
    """Tek günde |getiri|>thresh sıçramalarını rebase/birim artefaktı sayıp
    önceki segmenti sıçrama oranıyla ölçekler (sıçrama günü getirisi 0 olur).

    Bilinen vaka: İŞY XBANK/XUMAL/XFINK/XILTM/XSPOR 2010-07-01→2011-01-02
    segmenti ÷100 birimde geliyor (2010-07-01'de −%99, 2011-01-02'de +%10000).
    Gerçek endeks günlük hareketi hiçbir dönemde ±%50'ye yaklaşmadığı için
    eşik güvenli.
    """
    import numpy as np
    adj = s.copy()
    i = 1
    while i < len(adj):
        r = adj.iloc[i] / adj.iloc[i - 1]
        if abs(r - 1) > thresh:
            f = 10.0 ** round(np.log10(r))   # birim faktörü (×100, ÷100 ...)
            if f != 1.0:
                adj.iloc[i:] = adj.iloc[i:] / f
                print(f"  ⚙ {code}: {adj.index[i].date()} birim düzeltme ÷{f:g} "
                      f"(ham oran {r:.4g}, korunan gün getirisi {r/f - 1:+.2%})")
            else:
                print(f"  ⚠️ {code}: {adj.index[i].date()} |getiri|>%50 ama 10-kuvveti değil ({r:.4g}) — dokunulmadı")
        i += 1
    return adj


def build(lookback_days: int, groups: list) -> pd.DataFrame:
    codes = [c for g in groups for c in _ALL_INDICES[g]]
    frames, failed = [], []
    for i, code in enumerate(codes, 1):
        df = _fetch_isy_single(code, lookback_days=lookback_days, timeout=30)
        if df is None or df.empty:
            failed.append(code)
            print(f"  ⚠️ {code}: veri yok/çekilemedi")
            continue
        df = df.sort_index()
        df['close'] = _splice_rebase(df['close'], code)
        out = df.reset_index().rename(columns={'date': 'date'})
        out['code'] = code
        frames.append(out[['date', 'code', 'close']])
        print(f"  ✓ {i:2d}/{len(codes)} {code}: {len(out)} bar  {out['date'].min().date()} → {out['date'].max().date()}")
        time.sleep(0.2)
    if failed:
        print(f"  ⚠️ Çekilemeyen endeksler: {', '.join(failed)}")
    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates(['date', 'code']).sort_values(['code', 'date']).reset_index(drop=True)
    return master


def cross_check(master: pd.DataFrame) -> None:
    for code, d, want in CROSS_CHECK:
        row = master[(master['code'] == code) & (master['date'] == pd.Timestamp(d))]
        if row.empty:
            print(f"  ⚠️ cross-check {code} {d}: satır yok")
            continue
        got = float(row['close'].iloc[0])
        ok = abs(got - want) / want < 0.001
        print(f"  {'✓' if ok else '🔴'} cross-check {code} {d}: İŞY {got:.2f} vs Fintables {want:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookback-days', type=int, default=8200)
    ap.add_argument('--groups', default='piyasa,sektor,tematik')
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(',') if g.strip()]
    for g in groups:
        if g not in _ALL_INDICES:
            raise SystemExit(f"bilinmeyen grup: {g} (geçerli: {list(_ALL_INDICES)})")

    print(f"Endeks master build — gruplar: {groups}, lookback {args.lookback_days}g")
    master = build(args.lookback_days, groups)
    cross_check(master)
    master.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ {OUT_PATH}: {len(master)} satır, {master['code'].nunique()} endeks, "
          f"{master['date'].min().date()} → {master['date'].max().date()}")


if __name__ == '__main__':
    main()
