"""
Sektör lead-lag anatomi ölçümü v0 — "XU100 dönüşlerinde sıralama var mı?"

Kullanıcı tezi (2026-06-13): BIST yükselişlerinden ÖNCE testere-yatayda
XTUMY/XHARZ görece iyi gider; BIST100 dönerken ilk XBANK liderlik eder.

Bu script TRADE SİNYALİ TEST ETMEZ. XU100 dip-dönüş olayları EX-POST (geleceği
görerek) tanımlanır; soru "dönüş anatomisinde tutarlı sektör sıralaması var mı"
sorusudur. Tradable dedektör (H-B) ayrı pre-spec'in işi.

Olay tanımı (primary, pre-stated):
  - pivot low: close == min(close[t-10 .. t+10])
  - düşüş kapısı: close / max(close[t-120 .. t-1]) - 1 <= -0.10
  - ileri kapı:   max(close[t+1 .. t+60]) / close[t] - 1 >= +0.15
  - dedup: 40 bar içindeki olaylardan en düşük close'lu kalır
  Alt (loose) varyant: dd<=-0.08, fwd>=+0.10 @ 90 bar — sayım/istikrar kontrolü.

Pencereler (XU100 işlem-günü ofsetleri, getiri = endeks − XU100 göreli):
  chop      [T-60, T-20]   (tez: XTUMY/XHARZ burada iyi)
  late_pre  [T-20, T]
  early_post[T,    T+10]   (tez: XBANK burada lider)
  mid_post  [T+10, T+40]

Aday seti: 27 sektör endeksi + XTUMY + XHARZ (o tarihte verisi olanlar arasında
rank; 1 = en iyi göreli getiri).

Null/placebo (per feedback_separability_metric_design): aynı pencere/rank
hesabı, dönüşlerden ±20 bar uzak rastgele tarihlerde; N_olay'lık 2000 bootstrap
çekilişiyle gözlenen ortalama rank'in yüzdeliği. "Dönüşte XBANK lider" iddiası
ancak placebo dağılımından ayrışıyorsa gerçek.

Çıktılar:
  output/sector_leadlag_v0_events.csv  (olay listesi)
  output/sector_leadlag_v0_ranks.csv   (olay × pencere × endeks long-form)
  stdout rapor (era split 2018 öncesi/sonrası + güncel snapshot)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from agent.sector_regime import _ALL_INDICES

MASTER = os.path.join(ROOT, 'output', 'sector_index_daily_master.parquet')
EV_OUT = os.path.join(ROOT, 'output', 'sector_leadlag_v0_events.csv')
RK_OUT = os.path.join(ROOT, 'output', 'sector_leadlag_v0_ranks.csv')

CAND = list(_ALL_INDICES['sektor']) + ['XTUMY', 'XHARZ']
FOCUS = ['XBANK', 'XTUMY', 'XHARZ']
WINDOWS = {'chop': (-60, -20), 'late_pre': (-20, 0), 'early_post': (0, 10), 'mid_post': (10, 40)}
PIVOT_W, DD_LOOK, RNG = 10, 120, np.random.default_rng(42)
N_BOOT = 2000


def load_matrix():
    m = pd.read_parquet(MASTER)
    px = m.pivot_table(index='date', columns='code', values='close').sort_index()
    px = px.loc[px['XU100'].notna()]          # XU100 işlem takvimi
    px = px.ffill(limit=5)
    return px


def find_turns(xu: pd.Series, dd_gate: float, fwd_gate: float, fwd_bars: int):
    c = xu.values
    n = len(c)
    events = []
    for t in range(DD_LOOK, n - fwd_bars):
        lo, hi = max(0, t - PIVOT_W), min(n, t + PIVOT_W + 1)
        if c[t] != c[lo:hi].min():
            continue
        peak = c[t - DD_LOOK:t].max()
        if c[t] / peak - 1 > dd_gate:
            continue
        if c[t + 1:t + 1 + fwd_bars].max() / c[t] - 1 < fwd_gate:
            continue
        events.append(t)
    # dedup: 40 bar içinde en düşük close
    dedup = []
    for t in events:
        if dedup and t - dedup[-1] < 40:
            if c[t] < c[dedup[-1]]:
                dedup[-1] = t
            continue
        dedup.append(t)
    return dedup


def window_relrets(px: pd.DataFrame, t: int, a: int, b: int):
    """t pozisyonu etrafında [t+a, t+b] göreli getirileri (Series, cand only)."""
    i0, i1 = t + a, t + b
    if i0 < 0 or i1 >= len(px):
        return None
    p0, p1 = px.iloc[i0], px.iloc[i1]
    ret = p1 / p0 - 1
    rel = ret - ret['XU100']
    rel = rel.reindex(CAND).dropna()
    return rel if len(rel) >= 10 else None


def event_table(px: pd.DataFrame, positions: list):
    rows = []
    for t in positions:
        for wname, (a, b) in WINDOWS.items():
            rel = window_relrets(px, t, a, b)
            if rel is None:
                continue
            ranks = rel.rank(ascending=False)  # 1 = en iyi
            for code in rel.index:
                rows.append({'date': px.index[t], 'pos': t, 'window': wname, 'code': code,
                             'rel_ret': rel[code], 'rank': ranks[code], 'n_avail': len(rel)})
    return pd.DataFrame(rows)


def norm_rank(df):
    """rank'i 0-1'e normalize et (0 = en iyi) — değişen aday sayısına dayanıklı."""
    return (df['rank'] - 1) / (df['n_avail'] - 1)


def summarize(rk: pd.DataFrame, label: str):
    print(f"\n— {label} (olay sayısı: {rk['pos'].nunique()}) —")
    rk = rk.copy()
    rk['nrank'] = norm_rank(rk)
    for wname in WINDOWS:
        sub = rk[rk['window'] == wname]
        if sub.empty:
            continue
        g = sub.groupby('code').agg(nrank=('nrank', 'mean'), rel=('rel_ret', 'mean'),
                                    top5=('rank', lambda s: (s <= 5).mean()), n=('rank', 'size'))
        g = g.sort_values('nrank')
        focus = g.loc[[c for c in FOCUS if c in g.index]]
        top3 = ', '.join(f"{c}({g.loc[c,'nrank']:.2f})" for c in g.index[:3])
        print(f"  {wname:10s} en-iyi-3: {top3}")
        for c, r in focus.iterrows():
            print(f"     {c:6s} nrank {r['nrank']:.3f}  rel {r['rel']*100:+.2f}%  top5 {r['top5']*100:.0f}%  (n={int(r['n'])})")


def placebo_pctile(px, turn_pos, rk_events):
    """Gözlenen focus-sektör ortalama nrank'inin placebo bootstrap yüzdeliği.
    Düşük yüzdelik = dönüşlerde placebo'dan sistematik İYİ rank."""
    n_ev = len(turn_pos)
    lo_need = max(-min(a for a, _ in WINDOWS.values()), DD_LOOK)
    hi_need = max(b for _, b in WINDOWS.values())
    banned = set()
    for t in turn_pos:
        banned.update(range(t - 20, t + 21))
    pool = [t for t in range(lo_need, len(px) - hi_need) if t not in banned]

    rk_events = rk_events.copy()
    rk_events['nrank'] = norm_rank(rk_events)
    obs = rk_events.groupby(['code', 'window'])['nrank'].mean()

    # placebo havuzunda her tarih için focus nrank'leri önceden hesapla
    cache = {}
    sample_dates = RNG.choice(pool, size=min(len(pool), 1500), replace=False)
    for t in sample_dates:
        for wname, (a, b) in WINDOWS.items():
            rel = window_relrets(px, int(t), a, b)
            if rel is None:
                continue
            ranks = rel.rank(ascending=False)
            nr = (ranks - 1) / (len(rel) - 1)
            for c in FOCUS:
                if c in nr.index:
                    cache.setdefault((c, wname), []).append(nr[c])

    print(f"\n— Placebo bootstrap (havuz {len(sample_dates)} tarih × {N_BOOT} çekiliş, olay başına n={n_ev}) —")
    print("  yüzdelik = gözlenen ort. nrank'in placebo dağılımındaki yeri; <5 = anlamlı LİDERLİK, >95 = anlamlı geride")
    for (c, wname), vals in sorted(cache.items(), key=lambda kv: (kv[0][0], list(WINDOWS).index(kv[0][1]))):
        if (c, wname) not in obs.index:
            continue
        vals = np.array(vals)
        boots = np.array([RNG.choice(vals, size=n_ev, replace=True).mean() for _ in range(N_BOOT)])
        pct = (boots < obs[(c, wname)]).mean() * 100
        print(f"  {c:6s} {wname:10s} gözlenen {obs[(c, wname)]:.3f} | placebo med {np.median(boots):.3f} | pctile {pct:5.1f}")


def continuous_leadlag(px: pd.DataFrame):
    """Keşifsel: 20g göreli getiri → XU100 ileri-20g getirisi Spearman."""
    from scipy.stats import spearmanr
    xu = px['XU100']
    fwd = xu.shift(-20) / xu - 1
    print("\n— Keşifsel sürekli lead-lag: corr( rel_ret_20g(t), XU100 fwd_20g(t) ), 5g adım —")
    rows = []
    for c in FOCUS + ['XUSIN', 'XHOLD', 'XUMAL']:
        if c not in px.columns:
            continue
        rel = (px[c] / px[c].shift(20) - 1) - (xu / xu.shift(20) - 1)
        df = pd.DataFrame({'rel': rel, 'fwd': fwd}).dropna().iloc[::5]
        for era, sub in [('tüm', df), ('<2018', df[df.index < '2018-01-01']), ('>=2018', df[df.index >= '2018-01-01'])]:
            if len(sub) < 50:
                continue
            rho, p = spearmanr(sub['rel'], sub['fwd'])
            rows.append((c, era, rho, p, len(sub)))
    for c, era, rho, p, n in rows:
        print(f"  {c:6s} {era:6s} rho {rho:+.3f}  p {p:.3f}  n {n}")


def snapshot(px: pd.DataFrame):
    print("\n— Güncel snapshot (göreli getiri vs XU100, son N işlem günü) —")
    for nd in (10, 20, 60):
        rel = (px.iloc[-1] / px.iloc[-1 - nd] - 1) - (px['XU100'].iloc[-1] / px['XU100'].iloc[-1 - nd] - 1)
        rel = rel.reindex(CAND).dropna().sort_values(ascending=False)
        focus = ', '.join(f"{c}: {rel[c]*100:+.1f}% (#{list(rel.index).index(c)+1}/{len(rel)})" for c in FOCUS if c in rel.index)
        print(f"  {nd:3d}g  top3: {', '.join(f'{c} {rel[c]*100:+.1f}%' for c in rel.index[:3])} | {focus}")


def main():
    px = load_matrix()
    xu = px['XU100']
    print(f"Matris: {px.shape[0]} gün × {px.shape[1]} endeks, {px.index.min().date()} → {px.index.max().date()}")

    turns = find_turns(xu, -0.10, 0.15, 60)
    loose = find_turns(xu, -0.08, 0.10, 90)
    print(f"XU100 dönüş olayları — primary: {len(turns)}, loose: {len(loose)}")
    print("  primary tarihler:", ', '.join(str(px.index[t].date()) for t in turns))

    rk = event_table(px, turns)
    ev = pd.DataFrame({'date': [px.index[t] for t in turns], 'pos': turns,
                       'close': [xu.iloc[t] for t in turns]})
    ev.to_csv(EV_OUT, index=False)
    rk.to_csv(RK_OUT, index=False)

    summarize(rk, 'PRIMARY — tüm olaylar')
    cut = px.index[rk['pos']].year if False else None
    rk_dates = rk.merge(ev[['pos', 'date']].rename(columns={'date': 'ev_date'}), on='pos')
    summarize(rk_dates[rk_dates['ev_date'] < '2018-01-01'], 'era <2018')
    summarize(rk_dates[rk_dates['ev_date'] >= '2018-01-01'], 'era >=2018')

    rk_loose = event_table(px, loose)
    summarize(rk_loose, 'LOOSE varyant (istikrar kontrolü)')

    placebo_pctile(px, turns, rk)
    continuous_leadlag(px)
    snapshot(px)


if __name__ == '__main__':
    main()
