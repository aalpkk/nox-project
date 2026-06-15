"""
Analist hedef-fiyat revizyon drift'i v0 — erken-sinyal testi (user "erken açıklama").

Fundamental-momentum FAIL'inin (yayın gecikmesi alfayı öldürüyor) ardından:
alfa period-end→publish penceresinde → ERKEN kamu sinyali aranıyor. En nicel
erken kanal = analist hedef-fiyat/tavsiye revizyonları (resmi finansaldan önce,
tarihli). Veri İNCE (1885 kayıt, 2022+, ~3 rev/hisse/yıl) → ilk-bakış, robust değil.

Test: yukarı-revizyon olayı (kurum kendi önceki hedefini yükseltti VEYA tavsiye
upgrade) → forward 20/60g getiri vs XU100. Aşağı-revizyon + baz ile kıyas.
Point-in-time: olay = yayin_tarihi; forward o tarihten.

Token: keychain (Bash wrapper FINTABLES_MCP_TOKEN verir).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from tools.sector_leadlag_v0 import load_matrix


def pull():
    from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient, _parse_markdown_table
    cli = FintablesMCPClient()
    # 6-aylık pencerelerle çek (her pencere <300 satır)
    wins = []
    for y in range(2022, 2027):
        wins += [(f'{y}-01-01', f'{y}-06-30'), (f'{y}-07-01', f'{y}-12-31')]
    rows = []
    for a, b in wins:
        sql = ("SELECT hisse_senedi_kodu AS kod, araci_kurum_kodu AS kurum,"
               " yayin_tarihi_europe_istanbul AS tarih, tavsiye, hedef_fiyat"
               " FROM hisse_senedi_araci_kurum_hedef_fiyatlari"
               f" WHERE yayin_tarihi_europe_istanbul BETWEEN DATE '{a}' AND DATE '{b}'"
               " AND hedef_fiyat > 0 ORDER BY kod, kurum, tarih LIMIT 300")
        p = cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "analist revizyon drift"})
        if isinstance(p, dict):
            for r in _parse_markdown_table(p.get("table") or ""):
                rows.append(r)
    df = pd.DataFrame(rows)
    df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')
    df['hedef_fiyat'] = pd.to_numeric(df['hedef_fiyat'], errors='coerce')
    return df.dropna(subset=['tarih', 'hedef_fiyat']).sort_values(['kod', 'kurum', 'tarih'])


def events(df):
    """Kurum kendi önceki hedefini değiştirdi → yön. İlk gözlem nötr (atlanır)."""
    out = []
    for (kod, kur), g in df.groupby(['kod', 'kurum']):
        g = g.sort_values('tarih')
        prev = None
        for _, r in g.iterrows():
            if prev is not None and prev > 0:
                chg = r['hedef_fiyat'] / prev - 1
                if abs(chg) > 0.02:                      # min %2 revizyon
                    out.append({'kod': kod, 'tarih': r['tarih'], 'chg': chg,
                                'yon': 'up' if chg > 0 else 'down'})
            prev = r['hedef_fiyat']
    return pd.DataFrame(out)


def test(ev):
    idx = load_matrix(); xu = idx['XU100']
    st = pd.read_parquet(os.path.join(ROOT, 'output', 'ohlcv_10y_fintables_master.parquet')).reset_index()
    pc = st.pivot_table(index='Date', columns='ticker', values='Close').sort_index().reindex(idx.index).ffill(limit=3)
    cal = idx.index
    if getattr(cal, 'tz', None) is not None:
        cal = cal.tz_localize(None)
    ev = ev.copy()
    if getattr(ev['tarih'].dt, 'tz', None) is not None:
        ev['tarih'] = ev['tarih'].dt.tz_localize(None)
    RNG = np.random.default_rng(42)

    def fwd_rel(kod, t, h):
        if kod not in pc.columns: return np.nan
        t = pd.Timestamp(t)
        if t.tz is not None: t = t.tz_localize(None)
        pos = cal.searchsorted(t)
        if pos + h >= len(cal) or pos < 1: return np.nan
        s = pc[kod]
        if pd.isna(s.iloc[pos]) or pd.isna(s.iloc[pos + h]): return np.nan
        return float(s.iloc[pos + h] / s.iloc[pos] - 1) - float(xu.iloc[pos + h] / xu.iloc[pos] - 1)

    for h in (20, 60):
        ev[f'r{h}'] = [fwd_rel(r['kod'], r['tarih'], h) for _, r in ev.iterrows()]
    print(f"olay: {len(ev)} ({(ev['yon']=='up').sum()} up / {(ev['yon']=='down').sum()} down), "
          f"{ev['kod'].nunique()} hisse, {ev['tarih'].dt.year.min()}-{ev['tarih'].dt.year.max()}")
    for h in (20, 60):
        up = ev.loc[ev['yon'] == 'up', f'r{h}'].dropna()
        dn = ev.loc[ev['yon'] == 'down', f'r{h}'].dropna()
        def boot(x):
            x = x.values
            b = np.array([RNG.choice(x, len(x), True).mean() for _ in range(2000)])
            return min((b <= 0).mean(), (b >= 0).mean()) * 2
        print(f"  fwd{h}g vs XU100: UP ort {up.mean()*100:+.2f}% (med {up.median()*100:+.2f}, win {(up>0).mean()*100:.0f}%, p {boot(up):.3f}, n{len(up)}) | "
              f"DOWN ort {dn.mean()*100:+.2f}% (p {boot(dn):.3f}, n{len(dn)}) | UP−DOWN {(up.mean()-dn.mean())*100:+.2f}pp")


def main():
    if not os.environ.get('FINTABLES_MCP_TOKEN'):
        raise SystemExit("FINTABLES_MCP_TOKEN gerekli")
    df = pull()
    print(f"çekilen kayıt: {len(df)}, {df['kod'].nunique()} hisse")
    ev = events(df)
    if ev.empty:
        print("revizyon olayı yok"); return
    test(ev)


if __name__ == '__main__':
    main()
