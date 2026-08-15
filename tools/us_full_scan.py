#!/usr/bin/env python3
"""NOX US FULL SCAN — tüm ABD evreni (~3000 likit sembol) taranır.

Nasdaq Trader sembol listesinden evren kurulur, yfinance toplu indirme ile tek
seferde 6 aylık OHLCV çekilir, movers + kurulum sınıflandırması yapılır.
(Evren geneli tek atımlık tarama olduğu için kaynak yfinance bulk'tır; izleme
evreninin extfeed'li hassas taraması tools/us_watch_scan.py'de.)

Çıktılar: output/usdata/full_scan.md (Claude'un okuduğu) + full_universe.csv
Yayın: nox-signals/usdata/full_scan.md

Filtre: fiyat>$3, ort. günlük dolar hacmi >$20M (likidite çöpü elenir).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.us_watch_scan import publish_to_signals

OUT_DIR = REPO / "output" / "usdata"
MIN_PRICE, MIN_DOLLAR_VOL = 3.0, 20e6
BATCH = 400


def universe():
    urls = ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]
    syms = set()
    for u in urls:
        try:
            txt = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
            df = pd.read_csv(io.StringIO(txt), sep="|")
            col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            etf = df.get("ETF", pd.Series(["N"] * len(df)))
            test = df.get("Test Issue", pd.Series(["N"] * len(df)))
            ok = df[(etf != "Y") & (test != "Y")][col].dropna()
            syms |= {s.strip() for s in ok if s.strip().isalpha() and len(s.strip()) <= 5}
        except Exception as e:
            print(f"universe {u}: {e}", file=sys.stderr)
    return sorted(syms)


def fetch(syms):
    frames = {}
    for i in range(0, len(syms), BATCH):
        chunk = syms[i:i + BATCH]
        try:
            df = yf.download(chunk, period="6mo", interval="1d", group_by="ticker",
                             auto_adjust=False, threads=True, progress=False)
            for s in chunk:
                try:
                    d = df[s].dropna()
                    if len(d) > 60:
                        frames[s] = d
                except Exception:
                    pass
        except Exception as e:
            print(f"batch {i}: {e}", file=sys.stderr)
    return frames


def metrics(s, d):
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    dv = float((c * v).tail(20).mean())
    if last < MIN_PRICE or dv < MIN_DOLLAR_VOL:
        return None
    sma20, sma50 = float(c.tail(20).mean()), float(c.tail(50).mean())
    hi20, lo20 = float(h.tail(20).max()), float(l.tail(20).min())
    hi252 = float(h.max())
    return {"sym": s, "last": round(last, 2), "chg": round(100 * (last / prev - 1), 2),
            "chg5": round(100 * (last / float(c.iloc[-6]) - 1), 2),
            "chg20": round(100 * (last / float(c.iloc[-21]) - 1), 2),
            "ext20": round(100 * (last / sma20 - 1), 1),
            "off_hi": round(100 * (last / hi252 - 1), 1),
            "volx": round(float(v.iloc[-1] / max(v.tail(20).mean(), 1)), 2),
            "sma20": round(sma20, 2), "sma50": round(sma50, 2),
            "hi20": round(hi20, 2), "lo20": round(lo20, 2),
            "dvol_m": round(dv / 1e6)}


def setup(m):
    t = []
    if m["chg"] > 5 and m["volx"] > 1.5:
        t.append("PATLAMA_1G")
    if m["chg5"] > 8 and abs(m["chg"]) < 2.5 and m["volx"] < 1.3:
        t.append("SINDIRME")
    if m["last"] >= m["hi20"] * 0.995 and m["chg20"] > 5:
        t.append("20G_ZIRVE")
    if abs(m["ext20"]) < 2.5 and m["sma20"] > m["sma50"] and m["chg20"] > 0:
        t.append("SMA20_TEST")
    if m["chg"] < -5 and m["volx"] > 1.5:
        t.append("COKUS_1G")
    if m["ext20"] > 18:
        t.append("UZAMIS")
    return t


def main() -> int:
    syms = universe()
    print(f"evren: {len(syms)}", flush=True)
    if not syms:
        return 1
    data = fetch(syms)
    print(f"veri gelen: {len(data)}", flush=True)
    rows = []
    for s, d in data.items():
        try:
            m = metrics(s, d)
        except Exception:
            continue
        if m:
            m["tags"] = "|".join(setup(m)) or "-"
            rows.append(m)
    if not rows:
        return 1
    df = pd.DataFrame(rows).sort_values("chg", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "full_universe.csv", index=False)

    def tbl(sub, title, n=25):
        if sub.empty:
            return []
        out = [f"## {title}", "",
               "| Sem | Son | 1g% | 5g% | 20g% | SMA20fark | 52hZ% | HacimX | $Hacim(M) | Kurulum |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.head(n).iterrows():
            out.append(f"| **{r['sym']}** | {r['last']} | {r['chg']} | {r['chg5']} | {r['chg20']} | "
                       f"{r['ext20']}% | {r['off_hi']} | {r['volx']} | {r['dvol_m']} | {r['tags']} |")
        return out + [""]

    o = [f"# NOX US FULL SCAN — {pd.Timestamp.utcnow():%Y-%m-%d %H:%M}Z",
         f"Taranan: {len(df)} likit sembol (fiyat>${MIN_PRICE}, $hacim>{MIN_DOLLAR_VOL / 1e6:.0f}M)", ""]
    o += tbl(df[df.chg > 0], "EN COK YUKSELENLER")
    o += tbl(df.sort_values("chg"), "EN COK DUSENLER")
    o += tbl(df[df.tags.str.contains("SINDIRME")].sort_values("chg5", ascending=False),
             "SINDIRME (bolge+onay adayi)")
    o += tbl(df[df.tags.str.contains("20G_ZIRVE")].sort_values("volx", ascending=False),
             "20 GUN ZIRVESI")
    o += tbl(df[df.tags.str.contains("SMA20_TEST")].sort_values("chg20", ascending=False),
             "SMA20 TESTI (pullback)")
    md = "\n".join(o)
    (OUT_DIR / "full_scan.md").write_text(md, encoding="utf-8")
    print(f"full_scan.md yazildi — {len(df)} sembol", flush=True)
    publish_to_signals("usdata/full_scan.md", md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
