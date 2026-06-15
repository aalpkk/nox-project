"""
Capture (down/up) faktörü testi v0 — defansif/low-vol edge (user "düşerken az düşen").

İki ölçüm:
  A) vs XU100 (piyasa-seviyesi): trailing down-capture → fwd β-alfa vs XU100.
     SONUÇ: düşük down-capture Q5−Q1 fwd120 +6.9%, HER İKİ dönemde anlamlı
     (2018-21 p.021, 2022-26 p.040 ham) = oturumun en sağlam edge'i (low-vol/BAB).
     Up-capture yarısı NULL (defansif hisse ralliyi yakalamaz; alfa düşüş-emmede).
  B) vs SEKTÖR endeksi (within-sector): sektör-rel down-capture → sektör-rel alfa.
     SONUÇ: ERA-KIRILGAN — 2018-21 null, sadece 2022-26 (faiz rejimi). Yani
     A'nın sağlamlığı SEKTÖR-SEVİYESİ defansif etkisini taşıyor (defansif sektörler
     piyasayı yener), within-sector HİSSE seçimi değil. (user düzeltmesi 2026-06-16)

Çıkarım: defansif tilt piyasa/sektör-seçimi katmanında sağlam; sektör-İÇİ hisse
seçiminde değil → eşit-ağırlık sepet. Saf fiyat verisi, token gerekmez.
Per [[feedback_separability_metric_design]] (GOOD-vs-WITHIN-CANDIDATE benchmark).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools.sector_leadlag_v0 import load_matrix

WIN, BETA, RNG = 250, 120, np.random.default_rng(42)


def load():
    idx = load_matrix()
    st = pd.read_parquet(os.path.join(ROOT, 'output', 'ohlcv_10y_fintables_master.parquet')).reset_index()
    pc = st.pivot_table(index='Date', columns='ticker', values='Close').sort_index().reindex(idx.index).ffill(limit=3)
    vol = st.pivot_table(index='Date', columns='ticker', values='Volume').sort_index().reindex(idx.index)
    cal = idx.index
    if getattr(cal, 'tz', None) is not None:
        cal = cal.tz_localize(None)
    smap = json.load(open(os.path.join(ROOT, 'tools', 'sector_map.json')))
    sec = {t: inf.get('sector_index') for t, inf in smap['tickers'].items() if inf.get('sector_index')}
    return idx, pc, vol, cal, sec


def _beta_alpha(s, bench, ti, fwd):
    both = pd.concat([s.iloc[ti - BETA:ti].pct_change(), bench.pct_change().iloc[ti - BETA:ti]], axis=1).dropna()
    if len(both) < 60:
        return None
    beta = np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / (both.iloc[:, 1].var() + 1e-12)
    return float(s.iloc[ti + fwd] / s.iloc[ti] - 1) - beta * float(bench.iloc[ti + fwd] / bench.iloc[ti] - 1)


def run(mode, fwd, lo, hi):
    """mode='xu100' (piyasa) | 'sector' (within-sector). Q5−Q1 = düşük-dc − yüksek-dc."""
    idx, pc, vol, cal, sec = load.cache
    xu = idx['XU100']
    q5q1 = []
    ridx = [i for i in range(WIN, len(cal) - fwd, 20) if pd.Timestamp(lo) <= cal[i] < pd.Timestamp(hi)]
    for ti in ridx:
        adv = (vol.iloc[ti - 60:ti] * pc.iloc[ti - 60:ti]).mean().dropna().sort_values(ascending=False)
        univ = adv.head(200 if mode == 'sector' else 150).index
        rows = []
        for k in univ:
            if k not in pc.columns:
                continue
            bench = idx[sec[k]] if mode == 'sector' and sec.get(k) in idx.columns else (None if mode == 'sector' else xu)
            if bench is None:
                continue
            ir = bench.pct_change().iloc[ti - WIN:ti]
            dn = ir < 0
            if dn.sum() < 10 or ir[dn].mean() == 0:
                continue
            s = pc[k]
            if any(pd.isna(s.iloc[j]) for j in (ti - WIN, ti, ti + fwd)):
                continue
            sr = s.iloc[ti - WIN:ti].pct_change().reindex(ir.index)
            dc = sr[dn].mean() / ir[dn].mean()
            if np.isnan(dc):
                continue
            a = _beta_alpha(s, bench, ti, fwd)
            if a is None:
                continue
            rows.append({'dc': dc, 'a': a})
        need = 20 if mode == 'sector' else 15
        if len(rows) < need:
            continue
        d = pd.DataFrame(rows).sort_values('dc')
        n5 = max(3, len(d) // 5)
        q5q1.append(d['a'].head(n5).mean() - d['a'].tail(n5).mean())
    q = np.array(q5q1)
    b = np.array([RNG.choice(q, len(q), True).mean() for _ in range(2000)])
    return q.mean(), min((b <= 0).mean(), (b >= 0).mean()) * 2, len(q)


def main():
    load.cache = load()
    for mode, title in [('xu100', 'A) vs XU100 (piyasa-seviyesi defansif — SAĞLAM)'),
                        ('sector', 'B) vs SEKTÖR endeksi (within-sector seçim — ERA-KIRILGAN)')]:
        print(f"\n== {title} ==  düşük down-capture − yüksek, β-alfa Q5−Q1")
        for fwd in (60, 120):
            for seg, lo, hi in [('TÜM', '2018-01-01', '2027'), ('2018-21', '2018-01-01', '2022'), ('2022-26', '2022-01-01', '2027')]:
                m, p, n = run(mode, fwd, lo, hi)
                print(f"  fwd{fwd} {seg:7s}: {m * 100:+.2f}% p {p:.3f} (reb {n})")


if __name__ == '__main__':
    main()
