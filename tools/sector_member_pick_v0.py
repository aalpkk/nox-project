"""
Sektör-üyesi seçim ölçümü v0 — "dönüşün ilk 10 gününde XBANK içinden hangi hisse?"

KEŞİFSEL ölçüm (lockbox yok, n küçük). Faz 1/2 hattının devamı
(docs/sector_rotation_trade_spec_v0.md). MTF-v6 dersi gereği "endeks üstü"
β-DÜŞÜLMÜŞ alpha olarak ölçülür: alpha = hisse_fwd − β·XBANK_fwd, β trailing
250 günden. Per feedback_separability_metric_design: sorular within-candidate
(aynı olaydaki bankalar arası) ranklarla.

Sorular:
  Q1 Dispersiyon: olay içi bankalar arası ilk-10g getiri farkı seçimi değer
     kılacak kadar büyük mü? (top−bottom, IQR)
  Q2 β yeter mi: yüksek-β banka zaten işi bitiriyor mu? (within-event
     Spearman(β, fwd) ve alpha'da β'nın izi)
  Q3 Ezilme tahmini: late_pre'de XBANK'a göre en çok ezilen banka, dönüşte
     β-üstü mü sıçrıyor? (within-event Spearman(crush, alpha))
  Q4 Kronik lider: aynı banka tekrar tekrar mı kazanıyor? (ort. within-event rank)

Olaylar: Faz 1 ex-post dönüşler (≥2016-11, hisse master 2016-04'ten).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools.sector_leadlag_v0 import load_matrix, find_turns

STOCKS = os.path.join(ROOT, 'output', 'ohlcv_10y_fintables_master.parquet')
SMAP = os.path.join(ROOT, 'tools', 'sector_map.json')
OUT = os.path.join(ROOT, 'output', 'sector_member_pick_v0.csv')
BETA_LOOK, BETA_MIN, FWD = 250, 150, 10
RNG = np.random.default_rng(42)


def members(index_code):
    with open(SMAP) as f:
        data = json.load(f)
    return sorted(t for t, info in data['tickers'].items()
                  if index_code in (info.get('all_sector_indexes') or []))


def main():
    idx = load_matrix()
    turns_pos = find_turns(idx['XU100'], -0.10, 0.15, 60)
    banks = members('XBANK')
    print(f"XBANK üyeleri ({len(banks)}): {', '.join(banks)}")

    st = pd.read_parquet(STOCKS).reset_index()
    pxs = st.pivot_table(index='Date', columns='ticker', values='Close').sort_index()
    pxs = pxs.reindex(idx.index).ffill(limit=3)
    # ISATR/ISBTR: mikro-float İş A/B serileri — veride yok/seri arızalı (ISBTR
    # 2022-06-15 +%447 sıçrama), temsil gücü yok → hariç.
    cols = [b for b in banks if b in pxs.columns and b not in ('ISATR', 'ISBTR')]
    pxs = pxs[cols]
    xb = idx['XBANK']

    rows = []
    for tpos in turns_pos:
        T = idx.index[tpos]
        if tpos - BETA_LOOK < 0 or tpos + 1 + FWD >= len(idx):
            continue
        if T < pd.Timestamp('2016-11-01') or pxs.index[tpos + 1 + FWD] > pxs.dropna(how='all').index.max():
            continue
        xb_hist = xb.iloc[tpos - BETA_LOOK:tpos].pct_change().dropna()
        xb_fwd = float(xb.iloc[tpos + 1 + FWD] / xb.iloc[tpos + 1] - 1)
        xb_pre = float(xb.iloc[tpos] / xb.iloc[tpos - 20] - 1)
        for b in cols:
            s = pxs[b]
            hist = s.iloc[tpos - BETA_LOOK:tpos].pct_change()
            both = pd.concat([hist, xb_hist], axis=1).dropna()
            if len(both) < BETA_MIN:
                continue
            beta = float(np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / (both.iloc[:, 1].var() + 1e-12))
            p_e, p_x, p_pre = s.iloc[tpos + 1], s.iloc[tpos + 1 + FWD], s.iloc[tpos - 20]
            if np.isnan(p_e) or np.isnan(p_x) or np.isnan(p_pre) or np.isnan(s.iloc[tpos]):
                continue
            fwd = float(p_x / p_e - 1)
            crush = float(s.iloc[tpos] / p_pre - 1) - xb_pre
            rows.append({'event': T.date(), 'tpos': tpos, 'bank': b, 'beta': beta,
                         'crush': crush, 'fwd': fwd, 'xb_fwd': xb_fwd,
                         'rel': fwd - xb_fwd, 'alpha': fwd - beta * xb_fwd})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ev_sizes = df.groupby('event').size()
    print(f"\nOlay sayısı: {df['event'].nunique()}, olay başına banka {ev_sizes.min()}-{ev_sizes.max()}")

    # Q1 dispersiyon
    g = df.groupby('event')['fwd']
    spread = (g.max() - g.min())
    print(f"\nQ1 dispersiyon (ilk-10g, olay içi): top−bottom medyan {spread.median()*100:.1f}pp "
          f"(IQR {spread.quantile(.25)*100:.1f}-{spread.quantile(.75)*100:.1f}pp) → seçim {'DEĞERLİ' if spread.median() > 0.05 else 'marjinal'}")

    # Q2/Q3 within-event Spearman'lar
    def within_rho(xcol, ycol):
        rhos = []
        for _, gg in df.groupby('event'):
            if len(gg) >= 6:
                r, _ = spearmanr(gg[xcol], gg[ycol])
                rhos.append(r)
        rhos = np.array(rhos)
        boots = np.array([RNG.choice(rhos, len(rhos), True).mean() for _ in range(2000)])
        p = min((boots <= 0).mean(), (boots >= 0).mean()) * 2
        return rhos.mean(), (rhos > 0).mean(), p, len(rhos)

    for xc, yc, q in [('beta', 'fwd', 'Q2a β→ham getiri'), ('beta', 'alpha', 'Q2b β→alpha (sızıntı kontrolü)'),
                      ('crush', 'alpha', 'Q3 ezilme→alpha'), ('crush', 'rel', 'Q3b ezilme→rel (β-süz)')]:
        m, pos, p, k = within_rho(xc, yc)
        print(f"{q}: ort rho {m:+.3f} | olayların %{pos*100:.0f}'inde + | boot p {p:.3f} | k={k}")

    # Q4 kronik lider
    df['rank'] = df.groupby('event')['fwd'].rank(ascending=False)
    df['nrank'] = df.groupby('event')['rank'].transform(lambda s: (s - 1) / (len(s) - 1))
    q4 = df.groupby('bank').agg(nrank=('nrank', 'mean'), n=('nrank', 'size'),
                                top3=('rank', lambda s: (s <= 3).mean())).query('n >= 8').sort_values('nrank')
    print("\nQ4 kronik liderlik (ort nrank, 0=hep en iyi; ≥8 olay):")
    print(q4.to_string(float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()
