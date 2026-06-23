"""
Sektör Rotasyon Monitör v0 — PAPER-FORWARD takip + Telegram + kilitli fren.

Backtest karnesi FAIL/era-fragile (docs/sector_rotation_trade_spec_v0.md,
tools/sector_rotation_trade_v0.py): zamanlama becerisi sadece 2022+ döneminde,
rotasyon overlay'i (XBANK→XHOLD→ikinci dalga) ise 2018+ iki yarıda da stabil
(+3pp/trade). Bu monitör durum makinesini İŞY canlı verisiyle her gün yeniden
oynatır; ARM/TRIGGER/LEG/EXIT/FREN olaylarını Telegram'a bildirir, paper-ledger
biriktirir. CANLI PARA SİSTEMİ DEĞİL — karar desteği + taze-veri doğrulama.

Fren (spec §Fren, 2026-06-13 kilitli): son 4 trade 4'ü STOP veya son 8 trade
ort. net < 0 → FREN ON (tetikler SUPPRESSED); fren sonrası son 4 paper trade
ort. net > 0 → FREN OFF.

Kullanım:
    python tools/sector_rotation_monitor_v0.py                # durum raporu
    python tools/sector_rotation_monitor_v0.py --log          # + log satırı
    python tools/sector_rotation_monitor_v0.py --log --notify # + Telegram (yeni olay varsa)
    python tools/sector_rotation_monitor_v0.py --digest       # tam durumu her halükarda yolla

İki hücre şeffaf izlenir (PRIMARY confirm-ON + keşifsel confirm-OFF); hücre
seçimi sonuca göre değiştirilmez.
"""
import argparse
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

# NOX: İŞY (1 seans gecikmeli) BIRAKILDI → extfeed_index_1h_master (nyx_rotation ile
# TEK-KAYNAK; CI'da extfeed_index_pull --delta ile günlük taze → güncel seansa ulaşır).
from tools.sector_rotation_trade_v0 import (COOLDOWN, COST, DD_ARM, DD_LOOK,
                                            HOLD, LEG1, LEG2, leg_asset)

CODES = ['XU100', 'XBANK', 'XHOLD', 'XMESY', 'XELKT', 'XFINK']
BOUNCE = 0.03
LOOKBACK_DAYS = 1700   # ~4.5 yıl bar → fren için yeterli trade tarihi
LOG = os.path.join(ROOT, 'output', 'sector_rotation_monitor_log.csv')
BRAKE_STOPS, BRAKE_MEANN, BRAKE_LIFT = 4, 8, 4


_INDEX_DAILY = None


def _load_index_daily():
    """extfeed_index_1h_master → günlük endeks close paneli (date-index, ticker-kolon).
    nyx_rotation ile birebir aynı 1h→günlük resample. Bir kez yüklenir (cache)."""
    global _INDEX_DAILY
    if _INDEX_DAILY is None:
        ief = pd.read_parquet(os.path.join(ROOT, 'output', 'extfeed_index_1h_master.parquet'),
                              columns=['ticker', 'ts_istanbul', 'close'])
        ief['gun'] = pd.to_datetime(ief['ts_istanbul']).dt.tz_localize(None).dt.normalize()
        ig = (ief.sort_values('ts_istanbul')
              .groupby(['ticker', 'gun'])['close'].last().reset_index())
        _INDEX_DAILY = ig.pivot(index='gun', columns='ticker', values='close').sort_index().ffill()
    return _INDEX_DAILY


def _index_close(code):
    """Tek endeks günlük close serisi (yoksa None) — _fetch_isy_single(c)['close'] yerine."""
    idx = _load_index_daily()
    return idx[code].dropna() if code in idx.columns else None


def fetch():
    idx = _load_index_daily()
    if 'XU100' not in idx.columns:
        raise SystemExit("extfeed_index_1h_master'da XU100 yok")
    keep = [c for c in CODES if c in idx.columns]
    return idx[keep].dropna(subset=['XU100'])


def leg_daily_ret(px, d, off):
    rs = []
    for a in leg_asset(max(off, 1)):
        s = px[a].values
        if not np.isnan(s[d]) and not np.isnan(s[d - 1]):
            rs.append(s[d] / s[d - 1] - 1)
    return float(np.mean(rs)) if rs else 0.0


def close_trade_net(px, entry, x_exit):
    rets, sides, prev = [], 1, None
    for d in range(entry + 1, x_exit + 1):
        assets = leg_asset(d - entry)
        if prev is not None and assets != prev:
            sides += 2
        prev = assets
        rets.append(leg_daily_ret(px, d, d - entry))
    sides += 1
    return float(np.prod([1 + r for r in rets]) * (1 - COST * sides) - 1)


def replay(px, confirm):
    """Durum makinesi + paper-ledger + fren. Döner: durum dict + olay listesi."""
    xu, xb = px['XU100'].values, px['XBANK'].values
    dates = px.index
    n = len(xu)
    state, run_min, block_until = 'IDLE', None, -1
    open_trade = None
    closed, events = [], []
    brake_on, closed_since_brake = False, 0

    def brake_check():
        nonlocal brake_on, closed_since_brake
        if not brake_on:
            tail4 = closed[-BRAKE_STOPS:]
            cond_stop = len(tail4) == BRAKE_STOPS and all(t['stopped'] for t in tail4)
            tail8 = closed[-BRAKE_MEANN:]
            cond_mean = len(tail8) == BRAKE_MEANN and np.mean([t['net'] for t in tail8]) < 0
            if cond_stop or cond_mean:
                brake_on, closed_since_brake = True, 0
                return 'BRAKE_ON'
        else:
            if closed_since_brake >= BRAKE_LIFT and np.mean([t['net'] for t in closed[-BRAKE_LIFT:]]) > 0:
                brake_on = False
                return 'BRAKE_OFF'
        return None

    for t in range(DD_LOOK, n):
        if open_trade is not None:
            e = open_trade['entry']
            off = t - e
            if off in (LEG1 + 1, LEG2 + 1):
                events.append((dates[t], 'LEG_SWITCH', '+'.join(leg_asset(off)), open_trade.get('suppressed', False)))
            if xu[t] < open_trade['run_min'] or off >= HOLD:
                stopped = xu[t] < open_trade['run_min']
                x_exit = min(t + 1, n - 1)
                net = close_trade_net(px, e, x_exit)
                closed.append({'entry_d': dates[e], 'exit_d': dates[x_exit], 'net': net, 'stopped': stopped})
                closed_since_brake += 1
                events.append((dates[t], 'EXIT_STOP' if stopped else 'EXIT_TIME',
                               f"net {net*100:+.1f}%", open_trade.get('suppressed', False)))
                ev = brake_check()
                if ev:
                    events.append((dates[t], ev, '', False))
                block_until = t + COOLDOWN
                open_trade, state = None, 'ARMED'
            continue
        if state == 'IDLE':
            if xu[t] / xu[t - DD_LOOK:t].max() - 1 <= DD_ARM:
                state, run_min = 'ARMED', xu[t]
                events.append((dates[t], 'ARM', f"dd {xu[t]/xu[t-DD_LOOK:t].max()-1:+.1%}", False))
        elif state == 'ARMED':
            run_min = min(run_min, xu[t])
            if xu[t] > xu[t - DD_LOOK:t].max():
                state, run_min = 'IDLE', None
                events.append((dates[t], 'DISARM', 'yeni 120g zirve', False))
            elif t >= block_until and xu[t] >= run_min * (1 + BOUNCE):
                ok = (xb[t] / xb[t - 5] - xu[t] / xu[t - 5]) > 0 if confirm else True
                if ok:
                    open_trade = {'entry': t + 1 if t + 1 < n else t, 'trig': t,
                                  'run_min': run_min, 'suppressed': brake_on}
                    events.append((dates[t], 'TRIGGER' + ('_SUPPRESSED' if brake_on else ''),
                                   f"dip {run_min:,.0f} +%{BOUNCE*100:.0f}", brake_on))

    return {'state': 'IN_TRADE' if open_trade else state, 'run_min': run_min,
            'trade': open_trade, 'closed': closed, 'events': events,
            'brake': brake_on, 'block_until': block_until, 'n': n}


def fmt_status(px, st, confirm, label):
    xu = px['XU100']
    last_c = float(xu.iloc[-1])
    L = [f"[{label}] durum: {st['state']}{'  🛑 FREN ON' if st['brake'] else ''}"]
    if st['state'] == 'ARMED':
        need = st['run_min'] * (1 + BOUNCE)
        met = last_c >= need
        L.append(f"  dip {st['run_min']:,.0f} | tetik {need:,.0f} "
                 f"({'eşik SAĞLANMIŞ' if met else f'{(need/last_c-1)*100:+.1f}% uzakta'})")
        cd = st['block_until'] - (st['n'] - 1)
        if met and cd > 0:
            L.append(f"  tetik-yasağı {cd} bar daha")
        if confirm:
            rel5 = float(px['XBANK'].iloc[-1] / px['XBANK'].iloc[-6] - xu.iloc[-1] / xu.iloc[-6])
            L.append(f"  XBANK 5g rel {rel5*100:+.2f}% ({'konfirm HAZIR' if rel5 > 0 else 'konfirm bekliyor'})")
    elif st['state'] == 'IN_TRADE':
        tr = st['trade']
        off = st['n'] - 1 - tr['entry']
        assets = '+'.join(leg_asset(max(off, 1)))
        sup = ' (SUPPRESSED — fren)' if tr.get('suppressed') else ''
        L.append(f"  giriş {px.index[min(tr['entry'], st['n']-1)].date()} offset {off}/{HOLD} | leg: {assets}{sup}")
        L.append(f"  stop: XU100 < {tr['run_min']:,.0f} | sonraki geçiş offset "
                 f"{LEG1 if off < LEG1 else (LEG2 if off < LEG2 else HOLD)}")
    cl = st['closed']
    if cl:
        t8 = cl[-BRAKE_MEANN:]
        L.append(f"  paper-ledger: {len(cl)} kapanan, son{len(t8)} ort {np.mean([t['net'] for t in t8])*100:+.2f}%, "
                 f"son {cl[-1]['exit_d'].date()} net {cl[-1]['net']*100:+.1f}%")
    return L


SCAN_CODES = ['XUSIN', 'XGIDA', 'XKMYA', 'XMADN', 'XMANA', 'XMESY', 'XKAGT', 'XTAST',
              'XTEKS', 'XUHIZ', 'XELKT', 'XILTM', 'XINSA', 'XTCRT', 'XTRZM', 'XULAS',
              'XUMAL', 'XBANK', 'XSGRT', 'XFINK', 'XHOLD', 'XGMYO', 'XYORT', 'XUTEK',
              'XBLSM', 'XTUMY', 'XHARZ', 'XU100']
SCAN_FRESH = 5  # ≤N bar önceki tetik "taze" sayılır


def sector_scan(lookback_days=600):
    """Her sektörün KENDİ dibine aynı makine (fiyat-only) — bilgi amaçlı,
    backtest'i yapılmadı; tetik tazeliği >90 bar ise bayat sayılır."""
    rows = []
    for c in SCAN_CODES:
        s = _index_close(c)
        if s is None or len(s) < DD_LOOK + 10:
            continue
        v = s.values
        state, run_min, trig_i = 'IDLE', None, None
        for t in range(DD_LOOK, len(v)):
            if state == 'IDLE':
                if v[t] / v[t - DD_LOOK:t].max() - 1 <= DD_ARM:
                    state, run_min, trig_i = 'ARMED', v[t], None
            else:
                run_min = min(run_min, v[t])
                if v[t] > v[t - DD_LOOK:t].max():
                    state, run_min, trig_i = 'IDLE', None, None
                elif trig_i is not None and v[t] < run_min:
                    trig_i = None
                elif trig_i is None and v[t] >= run_min * (1 + BOUNCE):
                    trig_i = t
        last = len(v) - 1
        dd = v[last] / v[last - DD_LOOK:last].max() - 1
        rows.append({'kod': c, 'dd': dd,
                     'durum': ('TETİK' if trig_i is not None else state if state == 'ARMED' else 'TREND'),
                     'dipten': (v[last] / run_min - 1) if run_min else np.nan,
                     'yas': (last - trig_i) if trig_i is not None else np.nan})
    return pd.DataFrame(rows).set_index('kod')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--notify', action='store_true')
    ap.add_argument('--digest', action='store_true')
    ap.add_argument('--no-scan', action='store_true', help='sektör kendi-dibi taramasını atla')
    args = ap.parse_args()

    px = fetch()
    last_bar = px.index[-1].date()
    print(f"İŞY canlı: {len(px)} bar, son {last_bar}")

    prev_bar = None
    if os.path.exists(LOG):
        try:
            prev_bar = pd.to_datetime(pd.read_csv(LOG)['bar_date']).max().date()
        except Exception:
            prev_bar = None

    all_lines, new_events = [], []
    states = {}

    # Banka-ateşleme günü (keşifsel, 2026-06-13 ölçümü): XBANK 1g rel ≥ +3pp
    # → XU100 ileri 20-40 bar tarihsel olarak bazın çok üstü (pctile 99-100,
    # eşik 3/4/5pp'de sağlam, n=52/21/11, hepsi 2018+). Enstrüman endeks/geniş
    # taraf; XBANK'ın kendisi sıçrama sonrası rel olarak SÖNÜYOR (−1pp 10-20g).
    ign = float((px['XBANK'].iloc[-1] / px['XBANK'].iloc[-2] - 1)
                - (px['XU100'].iloc[-1] / px['XU100'].iloc[-2] - 1))
    if ign >= 0.03:
        new_events.append(f"[bilgi] {last_bar} BANK_IGNITION — XBANK 1g rel {ign*100:+.1f}pp "
                          f"(tarihsel: XU100 fwd20-40 baz üstü; XBANK kovalamak değil endeks/rotasyon)")
    for confirm, label in [(True, 'PRIMARY confirm-ON'), (False, 'keşifsel confirm-OFF')]:
        st = replay(px, confirm)
        states[label] = st
        lines = fmt_status(px, st, confirm, label)
        all_lines += lines + ['']
        print('\n'.join(lines))
        cutoff = prev_bar if prev_bar is not None else px.index[-1].date() - pd.Timedelta(days=1)
        for d, ev, detail, suppressed in st['events']:
            if d.date() > cutoff:
                new_events.append(f"[{label}] {d.date()} {ev} {detail}")

    if new_events:
        print("\nYENİ OLAYLAR (önceki logdan beri):")
        for e in new_events:
            print(' ', e)

    scan_lines = []
    if not args.no_scan:
        sc = sector_scan()
        fresh = sc[(sc['durum'] == 'TETİK') & (sc['yas'] <= SCAN_FRESH)]
        if not fresh.empty:
            scan_lines.append("Taze sektör kendi-dip tetikleri (bilgi amaçlı, backtest'siz):")
            for k, r in fresh.iterrows():
                scan_lines.append(f"  {k}: dipten {r['dipten']*100:+.1f}%, {int(r['yas'])} bar önce (dd {r['dd']*100:+.1f}%)")
        if args.digest:
            scan_lines.append("Sektör taraması (dd120g | durum | dipten | tetik yaşı):")
            for k, r in sc.sort_values('yas').iterrows():
                ya = f"{int(r['yas'])}b" if r['yas'] == r['yas'] else '—'
                dt = f"{r['dipten']*100:+.1f}%" if r['dipten'] == r['dipten'] else '—'
                scan_lines.append(f"  {k:6s} {r['dd']*100:+5.1f}% | {r['durum']:5s} | {dt:>7s} | {ya}")
        if scan_lines:
            print('\n' + '\n'.join(scan_lines))

    if args.notify and (new_events or args.digest):
        from core.reports import send_telegram
        head = f"🧭 Sektör Rotasyon Monitör — {last_bar} (PAPER, canlı değil)\n"
        body = (('\n'.join('• ' + e for e in new_events) + '\n\n') if new_events else '') + '\n'.join(all_lines)
        if scan_lines:
            body += '\n' + '\n'.join(scan_lines)
        send_telegram(head + body)
        print("✓ Telegram gönderildi" if (new_events or args.digest) else "")

    if args.log:
        row = {'run_ts': datetime.now().isoformat(timespec='seconds'), 'bar_date': str(last_bar),
               'xu100': float(px['XU100'].iloc[-1]),
               'state_on': states['PRIMARY confirm-ON']['state'],
               'brake_on': states['PRIMARY confirm-ON']['brake'],
               'state_off': states['keşifsel confirm-OFF']['state'],
               'brake_off': states['keşifsel confirm-OFF']['brake'],
               'new_events': '; '.join(new_events)}
        hdr = not os.path.exists(LOG)
        pd.DataFrame([row]).to_csv(LOG, mode='a', header=hdr, index=False)
        print(f"✓ log: {LOG}")


if __name__ == '__main__':
    main()
