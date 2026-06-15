"""
Fundamental point-in-time panel build — likit evren çeyreklik USD finansallar.

Spec: docs/fundamental_momentum_spec_v0.md (kilitli). Sektör-rotasyon hattının
yön-değiştiren bulgusu: liderler (ASTOR/ASELS) fiyatla değil kâr-ivmesiyle
bulunuyor. Bu script Fintables MCP'den (fund_flow_monitor ile aynı istemci,
FINTABLES_MCP_TOKEN gerektirir — interaktif chat yerine CI/lokal token'la koşar)
likit evren için çeyreklik USD gelir-tablosu kalemlerini KAP yayın tarihiyle
(point-in-time) çeker.

🔴 BANKA/SİGORTA: gelir_tablosu_kalemleri'nde 'Dönem Karı (Zararı)' BOŞ döner
(farklı şablon). Bunlar ayrı kalem-adıyla ('Net Dönem Karı (Zararı)') ikinci
geçişte çekilir; eşleşmezse atlanır (spec §açık riskler).

Kullanım:
    FINTABLES_MCP_TOKEN=... python tools/fundamental_pit_panel_build.py \
        [--universe output/_fundmom_liquid100.csv] [--from-year 2017]
Çıktı: output/fundamental_pit_panel.parquet
       (kod, yil, ay, pub_utc, net_kar_usd, brut_usd, ciro_usd, sablon)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

OUT = os.path.join(ROOT, 'output', 'fundamental_pit_panel.parquet')
CHUNK = 8                 # 8 hisse × ~37 çeyrek ≈ 296 satır < 300 cap
_MCP_RETRY, _MCP_WAIT = 4, 65

# default (gelir_tablosu) ve banka şablonu kalem adları
NET_DEFAULT = "'Dönem Karı (Zararı)'"
NET_BANK = "'Net Dönem Karı (Zararı)', 'Dönem Net Kar/Zararı', 'Konsolide Net Dönem Karı/Zararı'"
BRUT = "'Brüt Kar (Zarar)'"
CIRO = "'Satışlar', 'Hasılat'"


def _call(cli, sql):
    import time
    last = None
    for k in range(_MCP_RETRY):
        try:
            return cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "fundamental PIT panel build"})
        except Exception as e:  # noqa: BLE001
            last = e
            if "429" in str(e) and k < _MCP_RETRY - 1:
                print(f"  ⏳ 429 — {_MCP_WAIT}s bekle ({k+1})"); time.sleep(_MCP_WAIT); continue
            raise
    raise last


def _pivot_sql(codes, net_kalem, from_year, with_other=True):
    in_codes = ", ".join(f"'{c}'" for c in codes)
    other = (f", MAX(CASE WHEN g.kalem={BRUT} THEN g.usd_donemsel END) AS brut,"
             f" MAX(CASE WHEN g.kalem IN ({CIRO}) THEN g.usd_donemsel END) AS ciro") if with_other else ""
    return (
        "SELECT g.hisse_senedi_kodu AS kod, g.yil, g.ay,"
        " MAX(f.yayinlanma_tarihi_utc) AS pub,"
        f" MAX(CASE WHEN g.kalem IN ({net_kalem}) THEN g.usd_donemsel END) AS net_kar{other}"
        " FROM hisse_finansal_tablolari_gelir_tablosu_kalemleri g"
        " JOIN hisse_finansal_tablolari f ON f.hisse_senedi_kodu=g.hisse_senedi_kodu"
        " AND f.yil=g.yil AND f.ay=g.ay"
        f" WHERE g.hisse_senedi_kodu IN ({in_codes}) AND g.ay IN (3,6,9,12) AND g.yil>={from_year}"
        " GROUP BY g.hisse_senedi_kodu, g.yil, g.ay"
        " ORDER BY g.hisse_senedi_kodu, g.yil, g.ay LIMIT 300"
    )


def _parse(payload):
    from nyxexpansion.intraday.fetchers.fintables import _parse_markdown_table
    rows = []
    for r in _parse_markdown_table(payload.get("table") or ""):
        rows.append(r)
    return rows


def _f(x):
    try:
        s = str(x).strip()
        return float(s) if s and s.lower() not in ("none", "null", "nan") else float("nan")
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', default=os.path.join(ROOT, 'output', '_fundmom_liquid100.csv'))
    ap.add_argument('--from-year', type=int, default=2017)
    args = ap.parse_args()

    if not os.environ.get("FINTABLES_MCP_TOKEN"):
        raise SystemExit("🔴 FINTABLES_MCP_TOKEN gerekli (CI secret / lokal env). "
                         "Interaktif chat'te bu script koşmaz — orada veri_sorgula ile elle çekilir.")
    from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient

    univ = pd.read_csv(args.universe)['ticker'].tolist()
    chunks = [univ[i:i + CHUNK] for i in range(0, len(univ), CHUNK)]
    cli = FintablesMCPClient()
    out = []
    need_bank = []
    for i, ch in enumerate(chunks):
        payload = _call(cli, _pivot_sql(ch, NET_DEFAULT, args.from_year))
        rows = _parse(payload) if isinstance(payload, dict) else []
        got_codes = set()
        for r in rows:
            nk = _f(r.get('net_kar'))
            out.append({'kod': r.get('kod'), 'yil': int(r['yil']), 'ay': int(r['ay']),
                        'pub_utc': str(r.get('pub', ''))[:19], 'net_kar_usd': nk,
                        'brut_usd': _f(r.get('brut')), 'ciro_usd': _f(r.get('ciro')),
                        'sablon': 'default'})
            if nk == nk:
                got_codes.add(r.get('kod'))
        # net_kar tamamen boş gelen kodlar = banka/sigorta şablonu → 2. geçiş
        empties = [c for c in ch if c not in got_codes]
        need_bank.extend(empties)
        print(f"  chunk {i+1}/{len(chunks)}: {len(rows)} satır, default-boş {len(empties)}")

    # Banka/sigorta 2. geçiş (net kâr farklı kalem adı, brut/ciro yok)
    if need_bank:
        bchunks = [need_bank[i:i + CHUNK] for i in range(0, len(need_bank), CHUNK)]
        for i, ch in enumerate(bchunks):
            payload = _call(cli, _pivot_sql(ch, NET_BANK, args.from_year, with_other=False))
            rows = _parse(payload) if isinstance(payload, dict) else []
            for r in rows:
                nk = _f(r.get('net_kar'))
                if nk != nk:
                    continue
                out.append({'kod': r.get('kod'), 'yil': int(r['yil']), 'ay': int(r['ay']),
                            'pub_utc': str(r.get('pub', ''))[:19], 'net_kar_usd': nk,
                            'brut_usd': float('nan'), 'ciro_usd': float('nan'), 'sablon': 'bank'})
            print(f"  banka-chunk {i+1}/{len(bchunks)}: {len(rows)} satır")

    df = pd.DataFrame(out).drop_duplicates(['kod', 'yil', 'ay']).sort_values(['kod', 'yil', 'ay'])
    df = df[df['net_kar_usd'].notna()].reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"\n✓ {OUT}: {len(df)} satır, {df['kod'].nunique()} hisse "
          f"({(df['sablon']=='bank').sum()} banka-satırı), {df['yil'].min()}-{df['yil'].max()}")


if __name__ == '__main__':
    main()
