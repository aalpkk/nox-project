"""
Kalite faktörü testi v0 — ROE/marj/kaldıraç → forward β-alfa (point-in-time).

User "temel analiz?" → kâr-momentumu FAIL'inden sonra klasik temel analizin
KALİTE ayağı. Value (F/K, PD/DD) fiyat+pay-sayısı ister (yok); kalite oranları
`hisse_finansal_tablolari_finansal_oranlari`'da (ROE=Özkaynak Karlılığı, Net Kar
Marjı, Kaldıraç Oranı), parent'tan yayın tarihiyle point-in-time.

Beklenti (baştan): kalite faktörü YAVAŞ (edge yıllar ölçeğinde, çoğu fiyatlanmış)
→ 60g ufukta kesitsel sinyal zayıf/null olabilir. Yine de β-düşülmüş, kesitsel,
GOOD-vs-FAILED (Q5-Q1) ile dürüstçe ölçülür. Token: keychain (Bash wrapper).

Pull → output/quality_pit_panel.parquet (cache); test forward fwd60/120.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools.sector_leadlag_v0 import load_matrix

PANEL = os.path.join(ROOT, 'output', 'quality_pit_panel.parquet')
ORANLAR = {'Özkaynak Karlılığı': 'roe', 'Net Kar Marjı': 'nmarj', 'Kaldıraç Oranı': 'kaldirac'}
CHUNK, RNG, BETA = 8, np.random.default_rng(42), 120


def build():
    from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient, _parse_markdown_table
    univ = pd.read_csv(os.path.join(ROOT, 'output', '_fundmom_liquid100.csv'))['ticker'].tolist()
    cli = FintablesMCPClient()
    oran_in = ", ".join(f"'{o}'" for o in ORANLAR)
    case = " ".join(f"MAX(CASE WHEN o.oran='{o}' THEN o.deger END) AS {c}," for o, c in ORANLAR.items())
    rows = []
    for i in range(0, len(univ), CHUNK):
        ch = univ[i:i + CHUNK]
        in_c = ", ".join(f"'{c}'" for c in ch)
        sql = (f"SELECT o.hisse_senedi_kodu AS kod, o.yil, o.ay, MAX(f.yayinlanma_tarihi_utc) AS pub, {case[:-1]}"
               " FROM hisse_finansal_tablolari_finansal_oranlari o"
               " JOIN hisse_finansal_tablolari f ON f.hisse_senedi_kodu=o.hisse_senedi_kodu AND f.yil=o.yil AND f.ay=o.ay"
               f" WHERE o.hisse_senedi_kodu IN ({in_c}) AND o.oran IN ({oran_in}) AND o.ay IN (3,6,9,12) AND o.yil>=2017"
               " GROUP BY o.hisse_senedi_kodu, o.yil, o.ay ORDER BY o.hisse_senedi_kodu, o.yil, o.ay LIMIT 300")
        p = cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "kalite faktör panel"})
        if isinstance(p, dict):
            for r in _parse_markdown_table(p.get("table") or ""):
                rows.append(r)
        print(f"  chunk {i//CHUNK+1}: +{len(rows)} toplam")
    df = pd.DataFrame(rows)
    for c in ORANLAR.values():
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['yil'] = df['yil'].astype(int); df['ay'] = df['ay'].astype(int)
    df['pub'] = pd.to_datetime(df['pub'], errors='coerce')
    df = df.dropna(subset=['pub']).drop_duplicates(['kod', 'yil', 'ay']).sort_values(['kod', 'yil', 'ay'])
    df.to_parquet(PANEL, index=False)
    print(f"✓ {PANEL}: {len(df)} satır, {df['kod'].nunique()} hisse")
    return df


def latest_at(g, t):
    a = g[g['pub'] <= t]
    return a.iloc[-1] if len(a) else None


def test(feat, sign, label):
    idx = load_matrix(); xu = idx['XU100']
    pan = pd.read_parquet(PANEL); pan['pub'] = pd.to_datetime(pan['pub'])
    if getattr(pan['pub'].dt, 'tz', None) is not None: pan['pub'] = pan['pub'].dt.tz_localize(None)
    st = pd.read_parquet(os.path.join(ROOT, 'output', 'ohlcv_10y_fintables_master.parquet')).reset_index()
    pc = st.pivot_table(index='Date', columns='ticker', values='Close').sort_index().reindex(idx.index).ffill(limit=3)
    cal = idx.index
    if getattr(cal, 'tz', None) is not None: cal = cal.tz_localize(None)
    by = {k: g for k, g in pan.groupby('kod')}
    for fwd in (60, 120):
        rhos, q5q1 = [], []
        ridx = [i for i in range(BETA, len(cal) - fwd, 20) if cal[i] >= pd.Timestamp('2018-01-01')]
        for ti in ridx:
            T = cal[ti]; rows = []
            ib = xu.pct_change().iloc[ti - BETA:ti]; ifwd = float(xu.iloc[ti + fwd] / xu.iloc[ti] - 1)
            for k, g in by.items():
                if k not in pc.columns: continue
                r = latest_at(g, T)
                if r is None or not np.isfinite(r[feat]): continue
                s = pc[k]
                if any(pd.isna(s.iloc[j]) for j in (ti - BETA, ti, ti + fwd)): continue
                sb = s.iloc[ti - BETA:ti].pct_change(); both = pd.concat([sb, ib], axis=1).dropna()
                if len(both) < 60: continue
                beta = np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / (both.iloc[:, 1].var() + 1e-12)
                alpha = float(s.iloc[ti + fwd] / s.iloc[ti] - 1) - beta * ifwd
                rows.append({'f': sign * float(r[feat]), 'a': alpha})
            if len(rows) >= 12:
                d = pd.DataFrame(rows); rho, _ = spearmanr(d['f'], d['a'])
                if rho == rho: rhos.append(rho)
                ds = d.sort_values('f'); n5 = max(2, len(d) // 5)
                q5q1.append(ds['a'].tail(n5).mean() - ds['a'].head(n5).mean())
        rhos, q5q1 = np.array(rhos), np.array(q5q1)
        br = np.array([RNG.choice(rhos, len(rhos), True).mean() for _ in range(2000)])
        bq = np.array([RNG.choice(q5q1, len(q5q1), True).mean() for _ in range(2000)])
        pr = min((br <= 0).mean(), (br >= 0).mean()) * 2; pq = min((bq <= 0).mean(), (bq >= 0).mean()) * 2
        print(f"  {label} fwd{fwd}: rho {rhos.mean():+.3f} p {pr:.3f} | Q5−Q1 {np.mean(q5q1)*100:+.2f}% p {pq:.3f} (reb {len(rhos)})")


def main():
    if not os.path.exists(PANEL):
        if not os.environ.get('FINTABLES_MCP_TOKEN'):
            raise SystemExit("panel yok + FINTABLES_MCP_TOKEN yok")
        build()
    print("== Kalite faktörü → fwd β-alfa (point-in-time, yüksek=iyi) ==")
    test('roe', +1, 'ROE')
    test('nmarj', +1, 'Net Marj')
    test('kaldirac', -1, 'Düşük Kaldıraç')


if __name__ == '__main__':
    main()
