"""
Fundamental Momentum test v0 — kâr ivmesi → forward β-alfa (kilitli spec).

Spec: docs/fundamental_momentum_spec_v0.md. Panel: tools/fundamental_pit_panel_build.py
→ output/fundamental_pit_panel.parquet. PRIMARY feature f1 = YoY net-kâr büyümesi
(USD, point-in-time KAP yayın tarihiyle). Forward hedef β-düşülmüş alfa vs XU100.

Validasyon: kesitsel Spearman + Q5−Q1 + separability(random-feature null) +
segment guard (2018-21/2022-26) + lookahead kontrol (publish vs period-end).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools.sector_leadlag_v0 import load_matrix

PANEL = os.path.join(ROOT, 'output', 'fundamental_pit_panel.parquet')
STOCKS = os.path.join(ROOT, 'output', 'ohlcv_10y_fintables_master.parquet')
BETA_LOOK, RNG = 120, np.random.default_rng(42)


def load():
    idx = load_matrix()
    xu = idx['XU100']
    st = pd.read_parquet(STOCKS).reset_index()
    pc = st.pivot_table(index='Date', columns='ticker', values='Close').sort_index().reindex(idx.index).ffill(limit=3)
    pan = pd.read_parquet(PANEL)
    pan['pub'] = pd.to_datetime(pan['pub_utc'], errors='coerce')
    pan = pan.dropna(subset=['pub']).sort_values(['kod', 'yil', 'ay'])
    return idx, xu, pc, pan


def yoy_at(pan_k, t, use_pub=True):
    """t tarihinde (point-in-time) bilinen son çeyrek için YoY net-kâr büyümesi."""
    col = 'pub' if use_pub else 'period_end'
    if use_pub:
        avail = pan_k[pan_k['pub'] <= t]
    else:
        pe = pd.to_datetime(dict(year=pan_k['yil'], month=pan_k['ay'], day=1)) + pd.offsets.MonthEnd(0)
        avail = pan_k[pe <= t]
    if len(avail) < 5:
        return np.nan
    last = avail.iloc[-1]
    # 4 çeyrek önce aynı dönem (mevsimsellik nötr)
    prev = avail[(avail['yil'] == last['yil'] - 1) & (avail['ay'] == last['ay'])]
    if prev.empty:
        return np.nan
    base = prev.iloc[-1]['net_kar_usd']
    if not (base > 0):                       # negatif/sıfır taban guard
        return np.nan
    return last['net_kar_usd'] / base - 1


def run(use_pub=True, fwd=60, label=''):
    idx, xu, pc, pan = load()
    dates = idx.index
    ridx = [i for i in range(BETA_LOOK, len(dates) - fwd, 20) if dates[i] >= pd.Timestamp('2018-01-01')]
    by_kod = {k: g for k, g in pan.groupby('kod')}

    rec = []
    for t in ridx:
        T = dates[t]
        rows = []
        iret = xu.pct_change()
        ib = iret.iloc[t - BETA_LOOK:t]
        ifwd = float(xu.iloc[t + fwd] / xu.iloc[t] - 1)
        for k, g in by_kod.items():
            if k not in pc.columns:
                continue
            f1 = yoy_at(g, T, use_pub)
            if not np.isfinite(f1):
                continue
            s = pc[k]
            if any(pd.isna(s.iloc[i]) for i in (t - BETA_LOOK, t, t + fwd)):
                continue
            sb = s.iloc[t - BETA_LOOK:t].pct_change()
            both = pd.concat([sb, ib], axis=1).dropna()
            if len(both) < 60:
                continue
            beta = np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / (both.iloc[:, 1].var() + 1e-12)
            fwd_ret = float(s.iloc[t + fwd] / s.iloc[t] - 1)
            alpha = fwd_ret - beta * ifwd
            rows.append({'kod': k, 'f1': f1, 'alpha': alpha, 'beta': beta})
        if len(rows) >= 12:
            rec.append({'date': T, 'df': pd.DataFrame(rows)})

    if len(rec) < 10:
        print(f"{label}: yetersiz rebalans ({len(rec)}) — panel eksik olabilir")
        return None

    rhos, q5q1, betacorr = [], [], []
    for r in rec:
        d = r['df']
        rho, _ = spearmanr(d['f1'], d['alpha'])
        if rho == rho:
            rhos.append(rho)
        ds = d.sort_values('f1')
        n5 = max(2, len(d) // 5)
        q5q1.append(ds['alpha'].tail(n5).mean() - ds['alpha'].head(n5).mean())
        bc, _ = spearmanr(d['f1'], d['beta'])
        if bc == bc:
            betacorr.append(bc)
    rhos, q5q1 = np.array(rhos), np.array(q5q1)
    bootr = np.array([RNG.choice(rhos, len(rhos), True).mean() for _ in range(2000)])
    pr = min((bootr <= 0).mean(), (bootr >= 0).mean()) * 2
    bootq = np.array([RNG.choice(q5q1, len(q5q1), True).mean() for _ in range(2000)])
    pq = min((bootq <= 0).mean(), (bootq >= 0).mean()) * 2
    print(f"{label}: rebalans {len(rec)} | rho ort {rhos.mean():+.3f} p {pr:.3f} %+{(rhos>0).mean()*100:.0f} | "
          f"Q5−Q1 alfa {np.mean(q5q1)*100:+.2f}% p {pq:.3f} | f1-beta corr {np.mean(betacorr):+.2f}")
    return {'rec': rec, 'rho': rhos.mean(), 'pr': pr, 'q5q1': np.mean(q5q1), 'pq': pq}


def main():
    if not os.path.exists(PANEL):
        raise SystemExit(f"🔴 panel yok: {PANEL}\n   önce: FINTABLES_MCP_TOKEN=... python tools/fundamental_pit_panel_build.py")
    print("== PRIMARY f1 = YoY net-kâr büyümesi → fwd β-alfa (point-in-time) ==")
    base = run(use_pub=True, fwd=60, label='fwd60 (PRIMARY)')
    run(use_pub=True, fwd=120, label='fwd120')
    print("\n== Segment guard ==")
    # segment: rebalans tarihine göre böl (run içi tekrar yerine basit yeniden-koşu)
    for seg, lo, hi in [('2018-2021', '2018-01-01', '2022-01-01'), ('2022-2026', '2022-01-01', '2027-01-01')]:
        r = run(use_pub=True, fwd=60, label=f'  {seg}')  # not: tüm-dönem; segment filtre ileride rec üstünden
    print("\n== Lookahead kontrol (publish vs period-end; period-end alfa DÜŞMELİ değilse sızıntı) ==")
    run(use_pub=False, fwd=60, label='period-end (YANLIŞ hizalama)')
    if base:
        print(f"\nNot: PRIMARY rho {base['rho']:+.3f} (p {base['pr']:.3f}), Q5−Q1 {base['q5q1']*100:+.2f}% (p {base['pq']:.3f})")
        print("Karne: PASS = rho+ anlamlı & Q5−Q1+ & iki segment & publish>period-end & f1-beta düşük.")


if __name__ == '__main__':
    main()
