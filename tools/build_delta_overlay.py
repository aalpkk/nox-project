#!/usr/bin/env python3
"""XU100 delta barlarını birleştirip doğrulayan overlay üretici.

`output/_delta/xu100_g*.csv` (Fintables `mumlar_gunluk_gh` çekimi, elle
aktarılmış) + XU100 endeks barları → tek overlay parquet.

NEDEN AYRI DOSYA, MASTER'A YAZILMIYOR: delta yalnızca 100 XU100 hissesini
kapsıyor; 607 hisselik master'a kısmi ekleme yapmak PIT evren kurulumunu
(çeyreklik hacim sıralaması tüm evreni gerektirir) ve tüm backtest'leri
sessizce bozardı. Overlay yalnızca ml_rank_scan tarafından bellekte uygulanır.

DOĞRULAMA: her ticker tam 9 bar taşımalı; 07-14 kapanışından ilk delta barına
geçişte |getiri| eşiği aşarsa aktarım hatası şüphesiyle işaretlenir (BIST günlük
marj ±%10 — tavan/taban dışı hareket bu bandı zorlamaz, ancak sermaye
artırımı/bedelsiz gerçek olabilir; o yüzden uyarı, hata değil).

Kullanım:
  python tools/build_delta_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DELTA_DIR = ROOT / "output" / "_delta"
OHLCV = ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
OUT_STOCK = DELTA_DIR / "xu100_overlay.parquet"
OUT_INDEX = DELTA_DIR / "xu100_index_overlay.parquet"

# zaman_utc D 21:00Z → Istanbul günü D+1 (doğrulandı: 2026-04-23T21:00Z ↔ 04-24)
DAY_MAP = {
    1: "2026-07-16", 2: "2026-07-17", 3: "2026-07-20", 4: "2026-07-21",
    5: "2026-07-22", 6: "2026-07-23", 7: "2026-07-24", 8: "2026-07-27",
    9: "2026-07-28", 10: "2026-07-29",
}
# 2026-07-15 Demokrasi ve Milli Birlik Günü — BIST kapalı, master 07-14'te bitiyor.

INDEX_ROWS = [
    (1, 14029.16, 14254.55, 14029.16, 14251.29),
    (2, 14145.75, 14153.74, 13962.64, 13981.05),
    (3, 13936.16, 14193.99, 13910.52, 14070.98),
    (4, 14038.17, 14127.80, 13916.08, 13974.14),
    (5, 14010.38, 14143.66, 13929.87, 14138.85),
    (6, 14150.52, 14228.02, 14031.89, 14077.67),
    (7, 14053.08, 14188.63, 13875.58, 13943.87),
    (8, 14028.40, 14085.39, 13773.98, 13774.77),
    (9, 13786.47, 13872.92, 13635.39, 13687.86),
    (10, 13692.31, 13721.90, 13449.02, 13501.55),
]

GAP_WARN_PCT = 12.0


def main() -> int:
    parts = [pd.read_csv(p) for p in sorted(DELTA_DIR.glob("xu100_g*.csv"))]
    if not parts:
        print("delta CSV yok", file=sys.stderr)
        return 1
    d = pd.concat(parts, ignore_index=True)
    d["Date"] = pd.to_datetime(d["d"].map(DAY_MAP))
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                          "close": "Close", "volume": "Volume"})
    d = d[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]]

    # ── doğrulama 1: her ticker tam 9 bar ──
    cnt = d.groupby("ticker").size()
    bad = cnt[cnt != len(DAY_MAP)]
    if len(bad):
        print(f"[HATA] eksik/fazla bar: {bad.to_dict()}", file=sys.stderr)
        return 1

    # ── doğrulama 2: OHLC iç tutarlılık ──
    inv = d[(d["High"] < d["Low"]) | (d["High"] < d["Open"]) | (d["High"] < d["Close"])
            | (d["Low"] > d["Open"]) | (d["Low"] > d["Close"])]
    if len(inv):
        print(f"[HATA] OHLC tutarsız {len(inv)} satır:\n{inv}", file=sys.stderr)
        return 1

    # ── doğrulama 3: master'ın son barına bağlanma (aktarım hatası dedektörü) ──
    m = pd.read_parquet(OHLCV)
    last = m[m.index == m.index.max()].set_index("ticker")["Close"]
    first = d[d["Date"] == pd.Timestamp(DAY_MAP[1])].set_index("ticker")["Close"]
    common = first.index.intersection(last.index)
    jump = ((first[common] / last[common] - 1) * 100).sort_values()
    flagged = jump[jump.abs() > GAP_WARN_PCT]

    print(f"delta: {len(d):,} satır · {d['ticker'].nunique()} ticker · "
          f"{d['Date'].min().date()} → {d['Date'].max().date()}")
    print(f"master son bar: {m.index.max().date()} · bağlanan ticker: {len(common)}")
    print(f"geçiş getirisi: min {jump.min():+.1f}% · medyan {jump.median():+.1f}% "
          f"· max {jump.max():+.1f}%")
    if len(flagged):
        print(f"[UYARI] |geçiş| > %{GAP_WARN_PCT} ({len(flagged)}): "
              + ", ".join(f"{t} {v:+.1f}%" for t, v in flagged.items()))
        print("   → sermaye artırımı/bedelsiz olabilir; aktarım hatası da olabilir. Kontrol et.")

    d.set_index("Date").to_parquet(OUT_STOCK)

    ix = pd.DataFrame(INDEX_ROWS, columns=["d", "Open", "High", "Low", "Close"])
    ix["Date"] = pd.to_datetime(ix["d"].map(DAY_MAP))
    ix["Volume"] = float("nan")
    ix = ix.drop(columns=["d"]).set_index("Date").sort_index()
    ix.to_parquet(OUT_INDEX)

    print(f"\n  → {OUT_STOCK}  ({len(d):,})")
    print(f"  → {OUT_INDEX}  ({len(ix)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
