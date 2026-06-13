"""
Sektör Rotasyon Trade Sistemi v0 — nedensel dedektör + rotasyon takvimi backtest.

Spec: docs/sector_rotation_trade_spec_v0.md (KİLİTLİ — parametreler oynanmaz).
Faz 1 anatomi: tools/sector_leadlag_v0.py. Veri: sector_index_daily_master.

Durum makinesi (tamamen geçmişe bakan):
  IDLE → ARMED:  XU100 / max(son 120 bar) − 1 ≤ −10%
  ARMED → TETİK: close ≥ run_min × (1+bounce) [VE XBANK 5g rel > 0]
  giriş t+1 close; STOP: XU100 < run_min → t+1 close tasfiye; zaman çıkışı +40 bar.
Rotasyon: [0,+10] XBANK → (+10,+25] XHOLD → (+25,+40] XMESY+XELKT+XFINK eşit.
Maliyet: taraf başına 15 bps. Null'lar: random-same-TIM (zamanlama),
XU100-tut aynı tarih (rotasyon alfası), random-sektör aynı tarih.
Segment guard: tetik <2018 / ≥2018 ayrı karne.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from tools.sector_leadlag_v0 import load_matrix, find_turns, CAND

COST = 0.0015          # taraf başına
DD_ARM, DD_LOOK = -0.10, 120
HOLD, LEG1, LEG2 = 40, 10, 25
COOLDOWN = 10
LEG_ASSETS = [('XBANK',), ('XHOLD',), ('XMESY', 'XELKT', 'XFINK')]
N_NULL = 1000
RNG = np.random.default_rng(42)
TR_OUT = os.path.join(ROOT, 'output', 'sector_rotation_trade_v0_trades.csv')


def detect(px, bounce, confirm):
    xu = px['XU100'].values
    xb = px['XBANK'].values
    n = len(xu)
    state, run_min, peak = 'IDLE', None, None
    block_until = -1
    trades = []
    t = DD_LOOK
    while t < n - 2:
        if state == 'IDLE':
            peak = xu[t - DD_LOOK:t].max()
            if xu[t] / peak - 1 <= DD_ARM:
                state, run_min = 'ARMED', xu[t]
        elif state == 'ARMED':
            run_min = min(run_min, xu[t])
            if xu[t] > xu[t - DD_LOOK:t].max():
                state = 'IDLE'
            elif t >= block_until and xu[t] >= run_min * (1 + bounce):
                ok = True
                if confirm and t >= 5:
                    ok = (xb[t] / xb[t - 5] - xu[t] / xu[t - 5]) > 0
                if ok:
                    trades.append({'trig': t, 'entry': t + 1, 'run_min': run_min})
                    # trade süresi boyunca ilerle (stop benzetimi sim'de; burada blokla)
                    t_exit = simulate_exit(xu, t + 1, run_min)
                    block_until = t_exit + COOLDOWN
                    t = t_exit
        t += 1
    return trades


def simulate_exit(xu, e, run_min):
    """Çıkış pozisyonu: stop (close<run_min → t+1) veya e+HOLD."""
    last = min(e + HOLD, len(xu) - 1)
    for d in range(e + 1, last + 1):
        if xu[d] < run_min:
            return min(d + 1, len(xu) - 1)
    return last


def leg_asset(offset):
    if offset <= LEG1:
        return LEG_ASSETS[0]
    if offset <= LEG2:
        return LEG_ASSETS[1]
    return LEG_ASSETS[2]


def trade_returns(px, tr, rotation=True, asset_override=None):
    """Günlük getiri listesi + taraf sayısı. asset_override: leg→tuple dict (null için)."""
    xu = px['XU100'].values
    e = tr['entry']
    x_exit = simulate_exit(xu, e, tr['run_min'])
    stopped = x_exit < min(e + HOLD, len(xu) - 1) or (
        x_exit == min(e + HOLD, len(xu) - 1) and any(xu[d] < tr['run_min'] for d in range(e + 1, x_exit + 1)))
    rets, sides, prev_asset = [], 1, None
    for d in range(e + 1, x_exit + 1):
        off = d - e
        if rotation:
            assets = asset_override[leg_asset(off)] if asset_override else leg_asset(off)
        else:
            assets = ('XU100',)
        if prev_asset is not None and assets != prev_asset:
            sides += 2
        prev_asset = assets
        rs = []
        for a in assets:
            s = px[a].values
            if not np.isnan(s[d]) and not np.isnan(s[d - 1]):
                rs.append(s[d] / s[d - 1] - 1)
        rets.append(np.mean(rs) if rs else 0.0)
    sides += 1
    return rets, sides, x_exit, stopped


def trade_net(rets, sides):
    return float(np.prod([1 + r for r in rets]) * (1 - COST * sides) - 1)


def run_cell(px, bounce, confirm, turns, verbose=False):
    trades = detect(px, bounce, confirm)
    dates = px.index
    rows = []
    for tr in trades:
        rets, sides, x_exit, stopped = trade_returns(px, tr, rotation=True)
        xrets, xsides, _, _ = trade_returns(px, tr, rotation=False)
        caught = any(0 <= tr['trig'] - tp <= HOLD for tp in turns)
        lag = min((tr['trig'] - tp for tp in turns if 0 <= tr['trig'] - tp <= HOLD), default=None)
        rows.append({'trig_date': dates[tr['trig']], 'entry': tr['entry'], 'exit': x_exit,
                     'len': x_exit - tr['entry'], 'stopped': stopped, 'caught': caught, 'lag': lag,
                     'net_rot': trade_net(rets, sides), 'net_xu': trade_net(xrets, xsides)})
    df = pd.DataFrame(rows)
    return df, trades


def timing_null(px, lens, n_draw=N_NULL):
    """Aynı uzunluklarda rastgele örtüşmesiz XU100 pencereleri — ort. net getiri dağılımı."""
    xu = px['XU100'].values
    n = len(xu)
    means = []
    for _ in range(n_draw):
        used, vals = [], []
        for L in lens:
            for _try in range(50):
                s = int(RNG.integers(DD_LOOK, n - L - 1))
                if all(s + L < a or s > b for a, b in used):
                    used.append((s, s + L))
                    vals.append(xu[s + L] / xu[s] * (1 - COST * 2) - 1)
                    break
        means.append(np.mean(vals))
    return np.array(means)


def sector_null(px, trades, n_draw=N_NULL):
    """Aynı tarihler, leg varlıkları rastgele sektör — ort. net getiri dağılımı."""
    means = []
    for _ in range(n_draw):
        override = {la: tuple(RNG.choice(CAND, size=len(la), replace=False)) for la in LEG_ASSETS}
        vals = []
        for tr in trades:
            rets, sides, _, _ = trade_returns(px, tr, rotation=True, asset_override=override)
            vals.append(trade_net(rets, sides))
        means.append(np.mean(vals))
    return np.array(means)


def paired_boot(diff, n=2000):
    diff = np.asarray(diff)
    boots = np.array([RNG.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)])
    return float((boots <= 0).mean())


def seg_report(px, df, trades, label):
    if df.empty:
        print(f"  {label}: trade yok")
        return None
    catch = df['caught'].mean()
    lag = df.loc[df['lag'].notna(), 'lag']
    print(f"  {label}: n={len(df)}  net_rot ort {df['net_rot'].mean()*100:+.2f}% med {df['net_rot'].median()*100:+.2f}% "
          f"win {(df['net_rot']>0).mean()*100:.0f}%  stop {(df['stopped']).mean()*100:.0f}%  "
          f"dönüş-yakalama {catch*100:.0f}% (medyan gecikme {lag.median() if len(lag) else float('nan'):.0f} bar)")
    tn = timing_null(px, df['len'].tolist())
    xu_mean = df['net_xu'].mean()
    t_pct = float((tn < xu_mean).mean() * 100)
    sn = sector_null(px, trades)
    s_pct = float((sn < df['net_rot'].mean()).mean() * 100)
    p_pair = paired_boot(df['net_rot'] - df['net_xu'])
    rot_minus_xu = (df['net_rot'] - df['net_xu']).mean()
    print(f"     ZAMANLAMA: XU100-tut ort {xu_mean*100:+.2f}% | random-pencere p95 {np.percentile(tn,95)*100:+.2f}% | pctile {t_pct:.0f} → {'GEÇTİ' if t_pct>=95 else 'KALDI'}")
    print(f"     ROTASYON : rot−XU100 ort {rot_minus_xu*100:+.2f}pp (eşli boot p {p_pair:.3f}) | random-sektör pctile {s_pct:.0f} → {'GEÇTİ' if (p_pair<0.05 and s_pct>=95) else 'KALDI'}")
    return {'t_pct': t_pct, 's_pct': s_pct, 'p_pair': p_pair, 'n': len(df)}


def main():
    px = load_matrix()
    xu = px['XU100']
    turns = find_turns(xu, -0.10, 0.15, 60)
    print(f"Matris {px.shape}, ex-post dönüş {len(turns)} (yakalama kıyası için)")

    cells = [(0.03, True), (0.03, False), (0.05, True), (0.05, False)]
    for bounce, confirm in cells:
        primary = (bounce, confirm) == (0.03, True)
        tag = f"bounce {bounce*100:.0f}% confirm {'ON' if confirm else 'OFF'}" + ("  ← PRIMARY" if primary else "")
        print(f"\n══ {tag} ══")
        df, trades = run_cell(px, bounce, confirm, turns)
        if df.empty:
            print("  trade yok"); continue
        if primary:
            df.to_csv(TR_OUT, index=False)
        tr_by_pos = {t['entry']: t for t in trades}
        segA = df[df['trig_date'] < '2018-01-01']
        segB = df[df['trig_date'] >= '2018-01-01']
        trA = [tr_by_pos[e] for e in segA['entry']]
        trB = [tr_by_pos[e] for e in segB['entry']]
        seg_report(px, segA, trA, 'SegA <2018')
        seg_report(px, segB, trB, 'SegB ≥2018')

        # günlük seri (TIM/Sharpe/MaxDD) — sadece bilgi amaçlı
        daily = pd.Series(0.0, index=px.index)
        for tr in trades:
            rets, sides, x_exit, _ = trade_returns(px, tr, rotation=True)
            cost_d = COST * sides / max(len(rets), 1)
            for j, d in enumerate(range(tr['entry'] + 1, x_exit + 1)):
                daily.iloc[d] += rets[j] - cost_d
        eq = (1 + daily).cumprod()
        tim = (daily != 0).mean()
        sharpe = daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
        mdd = (eq / eq.cummax() - 1).min()
        bh = xu.iloc[-1] / xu.iloc[DD_LOOK] - 1
        print(f"  seri: TIM {tim*100:.0f}%  Sharpe {sharpe:.2f}  MaxDD {mdd*100:.1f}%  toplam {eq.iloc[-1]-1:+.1%}  (XU100 B&H {bh:+.0%}, TRY faizi 0 varsayımı)")


if __name__ == '__main__':
    main()
