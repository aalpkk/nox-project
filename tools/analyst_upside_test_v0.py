import os,sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.sector_leadlag_v0 import load_matrix
from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient,_parse_markdown_table
RNG=np.random.default_rng(42); BETA=120
cli=FintablesMCPClient(); rows=[]
wins=[]
for y in range(2022,2027): wins+=[(f'{y}-01-01',f'{y}-06-30'),(f'{y}-07-01',f'{y}-12-31')]
for a,b in wins:
    sql=("SELECT hisse_senedi_kodu AS kod, yayin_tarihi_europe_istanbul AS tarih, hedef_fiyat AS hf"
         " FROM hisse_senedi_araci_kurum_hedef_fiyatlari"
         f" WHERE yayin_tarihi_europe_istanbul BETWEEN DATE '{a}' AND DATE '{b}' AND hedef_fiyat>0"
         " ORDER BY kod, tarih LIMIT 300")
    p=cli.call_tool("veri_sorgula",{"sql":sql,"purpose":"analist hedef seviye"})
    if isinstance(p,dict): rows+=_parse_markdown_table(p.get("table") or "")
df=pd.DataFrame(rows); df['tarih']=pd.to_datetime(df['tarih']); df['hf']=pd.to_numeric(df['hf'],errors='coerce')
df=df.dropna(subset=['tarih','hf']).sort_values(['kod','tarih'])
print(f"çekilen: {len(df)} hedef, {df['kod'].nunique()} hisse, {df['tarih'].dt.year.min()}-{df['tarih'].dt.year.max()}")
idx=load_matrix(); xu=idx['XU100']
st=pd.read_parquet('output/ohlcv_10y_fintables_master.parquet').reset_index()
pc=st.pivot_table(index='Date',columns='ticker',values='Close').sort_index().reindex(idx.index).ffill(limit=3)
cal=idx.index
if getattr(cal,'tz',None) is not None: cal=cal.tz_localize(None)
if getattr(df['tarih'].dt,'tz',None) is not None: df['tarih']=df['tarih'].dt.tz_localize(None)
by={k:g for k,g in df.groupby('kod')}
for fwd in (60,120):
    rhos,q5q1=[],[]
    ridx=[i for i in range(BETA,len(cal)-fwd,20) if cal[i]>=pd.Timestamp('2022-06-01')]
    for ti in ridx:
        T=cal[ti]; ib=xu.pct_change().iloc[ti-BETA:ti]; ifwd=float(xu.iloc[ti+fwd]/xu.iloc[ti]-1); rows2=[]
        for k,g in by.items():
            if k not in pc.columns: continue
            rec=g[(g['tarih']<=T)&(g['tarih']>=T-pd.Timedelta(days=180))]
            if not len(rec): continue
            tgt=rec['hf'].iloc[-1]; s=pc[k]
            if any(pd.isna(s.iloc[j]) for j in (ti-BETA,ti,ti+fwd)) or s.iloc[ti]<=0: continue
            up=tgt/float(s.iloc[ti])-1
            both=pd.concat([s.iloc[ti-BETA:ti].pct_change(),ib],axis=1).dropna()
            if len(both)<60: continue
            beta=np.cov(both.iloc[:,0],both.iloc[:,1])[0,1]/(both.iloc[:,1].var()+1e-12)
            rows2.append({'up':up,'a':float(s.iloc[ti+fwd]/s.iloc[ti]-1)-beta*ifwd})
        if len(rows2)<12: continue
        d=pd.DataFrame(rows2); rho,_=spearmanr(d['up'],d['a'])
        if rho==rho: rhos.append(rho)
        ds=d.sort_values('up'); n5=max(2,len(d)//5)
        q5q1.append(ds['a'].tail(n5).mean()-ds['a'].head(n5).mean())
    rhos,q5q1=np.array(rhos),np.array(q5q1)
    if len(rhos)<8: print(f"  fwd{fwd}: yetersiz reb {len(rhos)}"); continue
    bq=np.array([RNG.choice(q5q1,len(q5q1),True).mean() for _ in range(2000)])
    br=np.array([RNG.choice(rhos,len(rhos),True).mean() for _ in range(2000)])
    print(f"  fwd{fwd}: rho {rhos.mean():+.3f} p {min((br<=0).mean(),(br>=0).mean())*2:.3f} | yüksek-upside Q5−Q1 {np.mean(q5q1)*100:+.2f}% p {min((bq<=0).mean(),(bq>=0).mean())*2:.3f} (reb {len(rhos)})")
