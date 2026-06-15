import os,sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.sector_leadlag_v0 import load_matrix
from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient,_parse_markdown_table
RNG=np.random.default_rng(42); BETA=120
SC={'endeks_ustu':2,'al':1,'tut':0,'endekse_paralel':0,'endeks_alti':-1,'sat':-2}
cli=FintablesMCPClient(); rows=[]
wins=[]
for y in range(2022,2027): wins+=[(f'{y}-01-01',f'{y}-06-30'),(f'{y}-07-01',f'{y}-12-31')]
for a,b in wins:
    sql=("SELECT hisse_senedi_kodu AS kod, yayin_tarihi_europe_istanbul AS tarih, tavsiye,"
         " CASE WHEN model_portfoyde THEN 1 ELSE 0 END AS mp"
         " FROM hisse_senedi_araci_kurum_hedef_fiyatlari"
         f" WHERE yayin_tarihi_europe_istanbul BETWEEN DATE '{a}' AND DATE '{b}'"
         " ORDER BY kod, tarih LIMIT 300")
    p=cli.call_tool("veri_sorgula",{"sql":sql,"purpose":"analist favori"})
    if isinstance(p,dict): rows+=_parse_markdown_table(p.get("table") or "")
df=pd.DataFrame(rows); df['tarih']=pd.to_datetime(df['tarih'])
df['sc']=df['tavsiye'].map(SC); df['mp']=pd.to_numeric(df['mp'],errors='coerce')
df=df.dropna(subset=['tarih']).sort_values(['kod','tarih'])
print(f"çekilen: {len(df)}, {df['kod'].nunique()} hisse")
idx=load_matrix(); xu=idx['XU100']
st=pd.read_parquet('output/ohlcv_10y_fintables_master.parquet').reset_index()
pc=st.pivot_table(index='Date',columns='ticker',values='Close').sort_index().reindex(idx.index).ffill(limit=3)
cal=idx.index
if getattr(cal,'tz',None) is not None: cal=cal.tz_localize(None)
if getattr(df['tarih'].dt,'tz',None) is not None: df['tarih']=df['tarih'].dt.tz_localize(None)
by={k:g for k,g in df.groupby('kod')}

def alpha(k,ti,fwd):
    s=pc[k]
    if any(pd.isna(s.iloc[j]) for j in (ti-BETA,ti,ti+fwd)): return None
    both=pd.concat([s.iloc[ti-BETA:ti].pct_change(),xu.pct_change().iloc[ti-BETA:ti]],axis=1).dropna()
    if len(both)<60: return None
    beta=np.cov(both.iloc[:,0],both.iloc[:,1])[0,1]/(both.iloc[:,1].var()+1e-12)
    return float(s.iloc[ti+fwd]/s.iloc[ti]-1)-beta*float(xu.iloc[ti+fwd]/xu.iloc[ti]-1)

for fwd in (60,120):
    rec_q,mp_diff=[],[]
    ridx=[i for i in range(BETA,len(cal)-fwd,20) if cal[i]>=pd.Timestamp('2022-06-01')]
    for ti in ridx:
        T=cal[ti]; rrows=[]
        for k,g in by.items():
            if k not in pc.columns: continue
            r=g[(g['tarih']<=T)&(g['tarih']>=T-pd.Timedelta(days=180))]
            if not len(r): continue
            a=alpha(k,ti,fwd)
            if a is None: continue
            rrows.append({'sc':r['sc'].mean(),'mp':r['mp'].max(),'a':a})
        if len(rrows)<12: continue
        d=pd.DataFrame(rrows)
        ds=d.sort_values('sc'); n5=max(2,len(d)//5)
        rec_q.append(ds['a'].tail(n5).mean()-ds['a'].head(n5).mean())  # yüksek tavsiye − düşük
        if (d['mp']==1).sum()>=3 and (d['mp']==0).sum()>=3:
            mp_diff.append(d.loc[d['mp']==1,'a'].mean()-d.loc[d['mp']==0,'a'].mean())
    def rep(arr,lbl):
        a=np.array(arr)
        if len(a)<8: print(f"  {lbl} fwd{fwd}: yetersiz ({len(a)})"); return
        b=np.array([RNG.choice(a,len(a),True).mean() for _ in range(2000)])
        print(f"  {lbl} fwd{fwd}: {a.mean()*100:+.2f}% p {min((b<=0).mean(),(b>=0).mean())*2:.3f} (reb {len(a)})")
    rep(rec_q,'tavsiye Q5−Q1 (yüksek−düşük)')
    rep(mp_diff,'model-portföy − değil       ')
