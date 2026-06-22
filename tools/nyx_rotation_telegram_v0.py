"""
Nyx Rotasyon Günlük Telegram Raporu v0 — 4 bilgi:
  1) YENİ DÖNEN (ARMED)         : düşüş kapısı geçmiş, dönüş bekleyen/yeni tetiklenen sektör
  2) SONRAKİ LİDER ADAYI        : (1)'den, tarihsel anatomi rölvesine göre erken-lider olanlar
                                  (sector_leadlag_v0_ranks.csv early_post nrank düşük)
  3) LİDER                      : göreli güçte öne geçmiş sektör (rel20 vs XU100 > eşik)
  4) DOWN-CAPTURE AL ADAYLARI   : (2)+(3) sektörlerindeki düşük down-capture (defansif) hisseler
                                  — oturumun sağlam edge'i (vs XU100, low-vol/BAB)

Sektör verisi: İŞ Yatırım canlı (auth yok). Hisse: output/ohlcv_10y_fintables_master.parquet.
Telegram: core.reports.send_telegram (TG_BOT_TOKEN/TG_CHAT_ID env — GitHub Actions secret).
Kullanım: python tools/nyx_rotation_telegram_v0.py [--notify]
"""
import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from agent.sector_regime import _ALL_INDICES

DD_ARM, DD_LOOK, BOUNCE = -0.10, 120, 0.03
WIN = 250                  # down-capture trailing penceresi
REL_LEAD = 0.015           # LİDER eşiği: 20g rel > +1.5%
FRESH_BARS = 8             # yeni-tetik "taze" sınırı
ANAT_LEADER = 0.43         # early_post nrank < bu → tarihsel erken-lider
ADV_MIN = 50e6             # likidite (TL/gün)
TOPN_PER_SEC = 2
SECTORS = _ALL_INDICES['sektor']


def sector_state(v):
    """Tek sektör close dizisinde durum makinesi → (state, run_min, trig_idx_last)."""
    n = len(v)
    state, run_min, trig = 'IDLE', None, None
    for t in range(DD_LOOK, n):
        if state == 'IDLE':
            if v[t] / v[t - DD_LOOK:t].max() - 1 <= DD_ARM:
                state, run_min, trig = 'ARMED', v[t], None
        else:
            run_min = min(run_min, v[t])
            if v[t] > v[t - DD_LOOK:t].max():
                state, run_min, trig = 'IDLE', None, None
            elif trig is not None and v[t] < run_min:
                trig = None
            elif trig is None and v[t] >= run_min * (1 + BOUNCE):
                trig = t
    return state, run_min, trig


def anatomy_priority():
    """sector_leadlag ranks → sektör başına early_post ortalama nrank (düşük=erken lider)."""
    p = os.path.join(ROOT, 'output', 'sector_leadlag_v0_ranks.csv')
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    ep = df[df['window'] == 'early_post'].copy()
    ep['nrank'] = (ep['rank'] - 1) / (ep['n_avail'] - 1)
    return ep.groupby('code')['nrank'].mean().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notify', action='store_true')
    args = ap.parse_args()

    # 1) sektör + XU100: extfeed_index master (TradingView, restore ile günlük taze;
    #    hisselerle TEK-KAYNAK, simetrik 1h→günlük resample). İŞY + Fintables intraday
    #    karışımı BIRAKILDI — iki-kaynak hizalaması nan/gecikme üretiyordu.
    ief = pd.read_parquet(os.path.join(ROOT, 'output', 'extfeed_index_1h_master.parquet'),
                          columns=['ticker', 'ts_istanbul', 'close'])
    ief['gun'] = pd.to_datetime(ief['ts_istanbul']).dt.tz_localize(None).dt.normalize()
    ig = ief.sort_values('ts_istanbul').groupby(['ticker', 'gun'])['close'].last().reset_index()
    idx = ig.pivot(index='gun', columns='ticker', values='close').sort_index()
    idx = idx.dropna(subset=['XU100']).ffill()
    xu = idx['XU100']
    asof = idx.index[-1].date()
    print(f"  extfeed_index 1h→günlük: {idx.shape[1]} endeks, son {asof}")
    anat = anatomy_priority()

    xur_full = xu.pct_change()
    dn_full = xur_full.iloc[-WIN:] < 0
    xur_dn_mean = xur_full.iloc[-WIN:][dn_full].mean()
    rows = []
    for c in SECTORS:
        if c not in idx.columns:
            continue
        s = idx[c].dropna()
        v = s.values
        st, run_min, trig = sector_state(v)
        rel20 = float(s.iloc[-1] / s.iloc[-21] - 1) - float(xu.iloc[-1] / xu.iloc[-21] - 1)
        rel10 = float(s.iloc[-1] / s.iloc[-11] - 1) - float(xu.iloc[-1] / xu.iloc[-11] - 1)
        dipten = (v[-1] / run_min - 1) if run_min else np.nan
        yas = (len(v) - 1 - trig) if trig is not None else np.nan
        # sektör endeksinin down-capture'ı vs XU100 (trailing 250g; düşük=defansif sektör)
        sret = idx[c].pct_change().iloc[-WIN:]
        sec_dc = float(sret[dn_full].mean() / xur_dn_mean) if xur_dn_mean else np.nan
        rows.append({'kod': c, 'state': st, 'rel20': rel20, 'rel10': rel10,
                     'dipten': dipten, 'yas': yas, 'anat': anat.get(c, np.nan), 'dc': sec_dc})
    sdf = pd.DataFrame(rows)

    # sınıflandırma (öncelik: LİDER > YENİ DÖNEN)
    def classify(r):
        if r['rel20'] > REL_LEAD and r['state'] != 'IDLE':
            return 'LIDER'
        if r['state'] == 'ARMED' or (r['state'] != 'IDLE' and r['yas'] == r['yas'] and r['yas'] <= FRESH_BARS):
            return 'YENI'
        return 'DIGER'
    sdf['bucket'] = sdf.apply(classify, axis=1)
    sdf['sonraki_lider'] = (sdf['bucket'] == 'YENI') & (sdf['anat'] < ANAT_LEADER)

    liders = sdf[sdf['bucket'] == 'LIDER'].sort_values('rel20', ascending=False)
    yeni = sdf[sdf['bucket'] == 'YENI'].sort_values('anat')
    # ④ iki havuz: OLGUN (lider + anatomi-erken sonraki-lider) ve ERKEN (kalan ARMED/yeni dönen)
    mature_secs = set(liders['kod']) | set(sdf[sdf['sonraki_lider']]['kod'])
    early_secs = set(yeni['kod']) - mature_secs
    target_secs = mature_secs | early_secs

    # 4) down-capture AL adayları (hedef sektörler)
    # VERİ: extfeed 1h master (CI'da restore-extfeed-master ile GÜNLÜK taze; tek-kaynak,
    # tüm BIST scanner'larıyla aynı). ohlcv master + Fintables tail BIRAKILDI — onlar
    # repo-vintage'a bağımlıydı (runner'da bayat → liquid150 evreni kayıyor + 300-cap
    # tail eksik seri). extfeed → her gün son bar=kapanış, gün-toplam=hacim.
    smap = json.load(open(os.path.join(ROOT, 'tools', 'sector_map.json')))
    tick_secs = {t: (inf.get('all_sector_indexes') or []) for t, inf in smap['tickers'].items()}
    ef = pd.read_parquet(os.path.join(ROOT, 'output', 'extfeed_intraday_1h_3y_master.parquet'),
                         columns=['ticker', 'ts_istanbul', 'close', 'volume'])
    ef['gun'] = pd.to_datetime(ef['ts_istanbul']).dt.tz_localize(None).dt.normalize()
    g = ef.sort_values('ts_istanbul').groupby(['ticker', 'gun']).agg(
        close=('close', 'last'), volume=('volume', 'sum')).reset_index()
    pc = g.pivot(index='gun', columns='ticker', values='close').sort_index()
    vol = g.pivot(index='gun', columns='ticker', values='volume').sort_index()
    print(f"  extfeed 1h→günlük: {pc.shape[1]} hisse, son {pc.index[-1].date()}")
    # GLOBAL likit-150 (taze extfeed ADV)
    adv_all = (vol.iloc[-60:] * pc.iloc[-60:]).mean().dropna()
    liquid150 = set(adv_all.sort_values(ascending=False).head(150).index)
    sec_of = {}  # ticker → hedef sektör (likit-150 ∩ hedef sektör; dedup)
    for t, ss in tick_secs.items():
        hit = [s for s in ss if s in target_secs]
        if hit and t in liquid150 and t in pc.columns:
            sec_of[t] = hit[0]
    # XU100 (idx) hisse takvimine hizala. 🔴 common index VARLIĞA bakar değere değil;
    # XU100'ün NaN olduğu gün (İŞY/Fintables append boşluğu) common'a girerse xuc son=NaN
    # → tüm rel20 nan olur. O yüzden common'ı XU100-DOLU günlere kısıtla.
    xu_al = xu.copy(); xu_al.index = pd.to_datetime(xu_al.index)
    xu_al = xu_al[xu_al.notna()]
    common = pc.index.intersection(xu_al.index)
    pcc, xuc = pc.loc[common], xu_al.loc[common]
    # ④ down-capture hisse verisinin EFEKTİF son tarihi (tail başarısızsa master'da donar);
    # sektör asof'undan geri kalmışsa Fintables tail/token bayat demektir → mesajda uyar.
    stock_asof = pcc.index[-1].date()
    stale_days = (asof - stock_asof).days
    xur = xuc.pct_change()
    dn = xur < 0
    cand = []
    for m, sec in sec_of.items():
        if m not in pcc.columns:
            continue
        s = pcc[m]
        if s.iloc[-WIN:].isna().mean() > 0.2 or pd.isna(s.iloc[-1]):
            continue
        sr = s.pct_change().iloc[-WIN:]
        d2 = dn.iloc[-WIN:]
        if d2.sum() < 10 or xur.iloc[-WIN:][d2].mean() == 0:
            continue
        dc = sr[d2].mean() / xur.iloc[-WIN:][d2].mean()
        if np.isnan(dc) or dc < 0 or dc >= 0.85:   # defansif bandı
            continue
        # rel20 NaN-safe: hisse/XU100'ün son GEÇERLİ değerinden 20 işlem günü geri
        sv, xv = s.dropna(), xuc.dropna()
        r20 = (float(sv.iloc[-1] / sv.iloc[-21] - 1) - float(xv.iloc[-1] / xv.iloc[-21] - 1)
               if len(sv) >= 21 and len(xv) >= 21 else float('nan'))
        cand.append({'kod': m, 'sec': sec, 'dc': dc, 'rel20': r20,
                     'bucket': 'olgun' if sec in mature_secs else 'erken'})

    # mesaj
    L = [f"🧭 <b>Nyx Rotasyon — {asof}</b>", ""]
    if stale_days > 4:   # hafta sonu toleransı; üstü = extfeed bayat
        L.insert(1, f"⚠️ <b>BAYAT VERİ:</b> ④ hisse fiyatları {stock_asof} tarihinde donmuş "
                    f"({stale_days}g geride). extfeed master güncel değil → CI'da "
                    f"<code>restore-extfeed-master</code> step'i çalışmamış ya da "
                    f"master-data-pull bayat (lokal koşuda normaldir).")
    def _dc(x):
        return f"dc {x:.2f}" if x == x else "dc —"
    L.append("<b>① LİDER (öne geçmiş)</b>  [dc düşük=defansif lider]")
    L += [f"  {r.kod}: rel20 {r.rel20*100:+.1f}% · {_dc(r.dc)} · dipten {r.dipten*100:+.0f}%" for r in liders.itertuples()] or ["  (yok)"]
    L.append("\n<b>② SONRAKİ LİDER ADAYI (ARMED + anatomi)</b>")
    sl = sdf[sdf['sonraki_lider']].sort_values('anat')
    L += [f"  {r.kod}: anatomi-erken {r.anat:.2f} · {_dc(r.dc)} · rel20 {r.rel20*100:+.1f}%" for r in sl.itertuples()] or ["  (yok)"]
    L.append("\n<b>③ YENİ DÖNEN (ARMED/taze)</b>")
    L += [f"  {r.kod}: {r.state}{f' yaş{int(r.yas)}' if r.yas==r.yas else ''} · {_dc(r.dc)} · rel20 {r.rel20*100:+.1f}%" for r in yeni.itertuples()] or ["  (yok)"]
    olgun = sorted([c for c in cand if c['bucket'] == 'olgun'], key=lambda x: x['dc'])
    # ERKEN: düşen-bıçak guard (rel20 ≥ −%10) — sürekli çöken sahte-defansifleri ele
    erken = sorted([c for c in cand if c['bucket'] == 'erken' and c['rel20'] >= -0.10], key=lambda x: x['dc'])
    L.append("\n<b>④a DEFANSİF AL — OLGUN sektörler</b> (lider/sonraki-lider)")
    L += [f"  {c['kod']} ({c['sec']}): dc {c['dc']:.2f} · rel20 {c['rel20']*100:+.1f}%" for c in olgun[:8]] or ["  (yok)"]
    L.append("\n<b>④b DEFANSİF AL — ERKEN sektörler</b> (yeni dönen ARMED, düşen-bıçak elendi)")
    L += [f"  {c['kod']} ({c['sec']}): dc {c['dc']:.2f} · rel20 {c['rel20']*100:+.1f}%" for c in erken[:8]] or ["  (yok)"]
    L.append("\n<i>down-capture vs XU100 (düşük=defansif, low-vol edge); ④b erken=daha riskli; rotasyon betimsel. Geçersiz: XU100 düşüş kapısı altı.</i>")
    msg = "\n".join(L)
    print(msg)

    if args.notify:
        from core.reports import send_telegram
        send_telegram(msg)
        print("\n✓ Telegram gönderildi")


if __name__ == '__main__':
    main()
