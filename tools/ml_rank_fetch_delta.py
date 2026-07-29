#!/usr/bin/env python3
"""ML-Rank için OHLCV delta çekici — CI'da elle aktarımın yerini alır.

Master parquet (`ohlcv_10y_fintables_master.parquet`) commit'lendiği tarihte
donuyor; ML-Rank ise son kapanışa ihtiyaç duyuyor. Bu script eksik günleri
YALNIZCA XU100 evreni için repo'nun kendi Fintables fetcher'ıyla çeker ve
`build_delta_overlay.py` ile aynı formatta bir overlay parquet üretir.

MASTER'A YAZILMAZ. Delta 100 hisseyi kapsıyor; 607 hisselik master'a kısmi
ekleme PIT evren kurulumunu (çeyreklik hacim sıralaması tüm evreni gerektirir)
ve backtest'leri sessizce bozar.

KOLON UYARISI: `nyxexpansion.daily.fetchers.fintables` satırlarındaki `volume`
alanı TL HACMİDİR (hacim_15_dk_gecikmeli). Master'ın `Volume` kolonu ise LOT
sayısıdır (islem_adedi) — 2026-04-24 GARAN barında birebir doğrulandı. Bu yüzden
burada `quantity` alanı kullanılır. Karıştırmak likidite kapısını ve vol20_tl'yi
~fiyat katı kadar şişirirdi.

Gerekli ortam değişkeni: FINTABLES_MCP_TOKEN

Kullanım:
  python tools/ml_rank_fetch_delta.py                 # master sonrası → bugün
  python tools/ml_rank_fetch_delta.py --end 2026-07-29
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.index_universe import CURRENT  # noqa: E402

OHLCV = ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
XU_CACHE = ROOT / "output" / "xu100_cache.parquet"
OUT_DIR = ROOT / "output" / "_delta"
OUT_STOCK = OUT_DIR / "xu100_overlay.parquet"
OUT_INDEX = OUT_DIR / "xu100_index_overlay.parquet"

GAP_WARN_PCT = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (varsayılan: bugün TR)")
    ap.add_argument("--universe", default="XU100")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = pd.read_parquet(OHLCV)
    m.index = pd.to_datetime(m.index)
    start = (m.index.max() + timedelta(days=1)).date()
    end = pd.Timestamp(args.end).date() if args.end else \
        pd.Timestamp.now(tz="Europe/Istanbul").date()
    if end < start:
        print(f"master zaten güncel ({m.index.max().date()}); delta gerekmiyor")
        return 0

    tickers = sorted(CURRENT[args.universe])
    print(f"delta penceresi: {start} → {end} · {len(tickers)} ticker")

    from nyxexpansion.daily.fetchers.fintables import fetch_range
    rows = fetch_range(tickers, start, end)
    if not rows:
        print("[HATA] fetcher boş döndü", file=sys.stderr)
        return 1

    d = pd.DataFrame(rows)
    # quantity = lot (master ile aynı ölçek); volume alanı TL hacmidir → KULLANMA
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                          "close": "Close", "quantity": "Volume", "date": "Date"})
    d["Date"] = pd.to_datetime(d["Date"])
    d = d[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]].dropna()
    d = d[d["Date"] > m.index.max()].drop_duplicates(["Date", "ticker"])

    # ── doğrulamalar (build_delta_overlay.py ile aynı) ──
    inv = d[(d["High"] < d["Low"]) | (d["High"] < d["Open"]) | (d["High"] < d["Close"])
            | (d["Low"] > d["Open"]) | (d["Low"] > d["Close"])]
    if len(inv):
        print(f"[HATA] OHLC tutarsız {len(inv)} satır", file=sys.stderr)
        return 1

    n_days = d["Date"].nunique()
    cnt = d.groupby("ticker").size()
    if len(cnt) and (cnt != n_days).any():
        eksik = cnt[cnt != n_days]
        print(f"[UYARI] bar sayısı tutarsız ({n_days} seans bekleniyor): {eksik.to_dict()}")

    last_close = m[m.index == m.index.max()].set_index("ticker")["Close"]
    first_day = d["Date"].min()
    first = d[d["Date"] == first_day].set_index("ticker")["Close"]
    common = first.index.intersection(last_close.index)
    jump = ((first[common] / last_close[common] - 1) * 100)
    flagged = jump[jump.abs() > GAP_WARN_PCT]

    print(f"delta: {len(d):,} satır · {d['ticker'].nunique()} ticker · "
          f"{d['Date'].min().date()} → {d['Date'].max().date()} ({n_days} seans)")
    print(f"geçiş getirisi: min {jump.min():+.1f}% · medyan {jump.median():+.1f}% "
          f"· max {jump.max():+.1f}%")
    if len(flagged):
        print(f"[UYARI] |geçiş| > %{GAP_WARN_PCT} ({len(flagged)}): "
              + ", ".join(f"{t} {v:+.1f}%" for t, v in flagged.items()))

    d.set_index("Date").to_parquet(OUT_STOCK)
    print(f"  → {OUT_STOCK}")

    # ── XU100 endeksi (features için bench) ──
    try:
        from nyxexpansion.intraday.fetchers.fintables import FintablesMCPClient
        cli = FintablesMCPClient()
        s_utc = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d 21:00:00")
        e_utc = pd.Timestamp(end).strftime("%Y-%m-%d 21:00:00")
        sql = ("SELECT zaman_utc, acilis, yuksek, dusuk, kapanis "
               "FROM endeks_mumlar_gunluk_gh WHERE kod = 'XU100' "
               f"AND zaman_utc >= TIMESTAMP '{s_utc}' AND zaman_utc <= TIMESTAMP '{e_utc}' "
               "ORDER BY zaman_utc LIMIT 300")
        payload = cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "ml_rank XU100 delta"})
        from nyxexpansion.intraday.fetchers.fintables import _parse_markdown_table, _coerce_float
        if isinstance(payload, str):
            import json as _json
            payload = _json.loads(payload)
        raw = _parse_markdown_table((payload or {}).get("table") or "")
        ix = pd.DataFrame([{
            "Date": pd.Timestamp(r["zaman_utc"].replace("Z", "+00:00"))
                      .tz_convert("Europe/Istanbul").normalize().tz_localize(None),
            "Open": _coerce_float(r["acilis"]), "High": _coerce_float(r["yuksek"]),
            "Low": _coerce_float(r["dusuk"]), "Close": _coerce_float(r["kapanis"]),
            "Volume": float("nan"),
        } for r in raw])
        xu = pd.read_parquet(XU_CACHE)
        ix = ix[ix["Date"] > xu.index.max()].set_index("Date").sort_index()
        ix.to_parquet(OUT_INDEX)
        print(f"  → {OUT_INDEX}  ({len(ix)} bar)")
    except Exception as e:
        print(f"[UYARI] XU100 endeks delta alınamadı: {type(e).__name__}: {e}")
        print("   → tarayıcı endeksi eski kapanışla çalışır (RS/makro feature'ları bayat)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
