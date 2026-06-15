"""
Value faktörü testi v0 — kazanç-getirisi E/P → forward β-alfa (point-in-time).

User "F/K, PD/DD yok mu": hazır çarpan tablosu yok ama EPS (Hisse Başına Kar)
oran tablosunda → F/K=fiyat/EPS hesaplanabilir. Negatif kâr için E/P=EPS/fiyat
(yüksek=ucuz=value). Point-in-time: EPS son yayınlanmış (pub<=t), fiyat=t.

Inflation E/P oranında iptal (TL EPS / TL fiyat). Beklenti: value faktörü BIST'te
güçlü olabilir (literatürde sağlam) — momentum/kaliteden farklı, dürüstçe ölçülür.
Token: keychain (Bash wrapper).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.sector_leadlag_v0 import load_matrix

PANEL = os.path.join(ROOT, 'output', 'eps_pit_panel.parquet')
CHUNK, RNG, BETA = 8, np.random.default_rng(42), 120


def build():
    from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient, _parse_markdown_table
    univ = pd.read_csv(os.path.join(ROOT, 'output', '_fundmom_liquid100.csv'))['ticker'].tolist()
    cli = FintablesMCPClient(); rows = []
    for i in range(0, len(univ), CHUNK):
        in_c = ", ".join(f"'{c}'" for c in univ[i:i+CHUNK])
        sql = ("SELECT o.hisse_senedi_kodu AS kod, o.yil, o.ay, MAX(f.yayinlanma_tarihi_utc) AS pub,"
               " MAX(o.deger) AS eps FROM hisse_finansal_tablolari_finansal_oranlari o"
               " JOIN hisse_finansal_tablolari f ON f.hisse_senedi_kodu=o.hisse_senedi_kodu AND f.yil=o.yil AND f.ay=o.ay"
               f" WHERE o.hisse_senedi_kodu IN ({in_c}) AND o.oran='Hisse Başına Kar' AND o.ay IN (3,6,9,12) AND o.yil>=2017"
               " GROUP BY o.hisse_senedi_kodu, o.yil, o.ay ORDER BY o.hisse_senedi_kodu, o.yil, o.ay LIMIT 300")
        p = cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "EPS value panel"})
        if isinstance(p, dict):
            rows += _parse_markdown_table(p.get("table") or "")
    df = pd.DataFrame(rows)
    df['eps'] = pd.to_numeric(df['eps'], errors='coerce')
    df['yil'] = df['yil'].astype(int); df['ay'] = df['ay'].astype(int)
    df['pub'] = pd.to_datetime(df['pub'], errors='coerce')
    df = df.dropna(subset=['pub', 'eps']).drop_duplicates(['kod','yil','ay']).sort_values(['kod','yil','ay'])
    df.to_parquet(PANEL, index=False)
    print(f"✓ {PANEL}: {len(df)} satır, {df['kod'].nunique()} hisse")
    return df


def main():
    if not os.path.exists(PANEL):
        if not os.environ.get('FINTABLES_MCP_TOKEN'): raise SystemExit("panel+token yok")
        build()
    idx = load_matrix(); xu = idx['XU100']
    pan = pd.read_parquet(PANEL); pan['pub'] = pd.to_datetime(pan['pub'])
    if getattr(pan['pub'].dt,'tz',None) is not None: pan['pub'] = pan['pub'].dt.tz_localize(None)
    st = pd.read_parquet(os.path.join(ROOT,'output','ohlcv_10y_fintables_master.parquet')).reset_index()
    pc = st.pivot_table(index='Date',columns='ticker',values='Close').sort_index().reindex(idx.index).ffill(limit=3)
    cal = idx.index
    if getattr(cal,'tz',None) is not None: cal = cal.tz_localize(None)
    by = {k:g for k,g in pan.groupby('kod')}
    # F/K akıl-sağlığı: son rebalansta medyan
    print("== Value faktörü E/P=EPS/fiyat (yüksek=ucuz) → fwd β-alfa (point-in-time) ==")
    for fwd in (60,120):
        for seg,lo,hi in [('TÜM','2018-01-01','2027'),('2018-21','2018-01-01','2022'),('2022-26','2022-01-01','2027')]:
            rhos,q5q1,fk_med = [],[],[]
            ridx=[i for i in range(BETA,len(cal)-fwd,20) if pd.Timestamp(lo)<=cal[i]<pd.Timestamp(hi)]
            for ti in ridx:
                T=cal[ti]; rows=[]
                ib=xu.pct_change().iloc[ti-BETA:ti]; ifwd=float(xu.iloc[ti+fwd]/xu.iloc[ti]-1)
                for k,g in by.items():
                    if k not in pc.columns: continue
                    a=g[g['pub']<=T]
                    if not len(a): continue
                    eps=a.iloc[-1]['eps']; s=pc[k]
                    if any(pd.isna(s.iloc[j]) for j in (ti-BETA,ti,ti+fwd)) or s.iloc[ti]<=0: continue
                    ep=eps/float(s.iloc[ti])
                    both=pd.concat([s.iloc[ti-BETA:ti].pct_change(),ib],axis=1).dropna()
                    if len(both)<60: continue
                    beta=np.cov(both.iloc[:,0],both.iloc[:,1])[0,1]/(both.iloc[:,1].var()+1e-12)
                    rows.append({'ep':ep,'a':float(s.iloc[ti+fwd]/s.iloc[ti]-1)-beta*ifwd,'fk':(1/ep if ep>0 else np.nan)})
                if len(rows)<12: continue
                d=pd.DataFrame(rows); rho,_=spearmanr(d['ep'],d['a'])
                if rho==rho: rhos.append(rho)
                ds=d.sort_values('ep'); n5=max(2,len(d)//5)
                q5q1.append(ds['a'].tail(n5).mean()-ds['a'].head(n5).mean())  # ucuz−pahalı
                fk_med.append(d['fk'].median())
            rhos,q5q1=np.array(rhos),np.array(q5q1)
            br=np.array([RNG.choice(rhos,len(rhos),True).mean() for _ in range(2000)])
            bq=np.array([RNG.choice(q5q1,len(q5q1),True).mean() for _ in range(2000)])
            pr=min((br<=0).mean(),(br>=0).mean())*2; pq=min((bq<=0).mean(),(bq>=0).mean())*2
            extra=f" | medyan F/K {np.nanmedian(fk_med):.1f}" if seg=='TÜM' and fwd==60 else ""
            print(f"  fwd{fwd} {seg:7s}: rho {rhos.mean():+.3f} p {pr:.3f} | ucuz−pahalı Q5−Q1 {np.mean(q5q1)*100:+.2f}% p {pq:.3f} (reb {len(rhos)}){extra}")
        print()


if __name__ == '__main__':
    main()
