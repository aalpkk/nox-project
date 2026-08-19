#!/usr/bin/env python3
"""NOX US WATCH — izleme evreni taraması (extfeed birincil / yfinance yedek).

config/us_watch_universe.yaml'daki sektör-etiketli evren için günlük (~300 bar)
+ saatlik (hourly_subset, 200 bar) OHLCV çekilir, tek dosyalık tarama özeti
üretilir ve nox-signals'a yayınlanır.

Veri kaynağı: extfeed TV WS (INTRADAY_SID/SIGN) — BIST extfeed ile aynı hat.
Extfeed başarısız olursa sembol bazında yfinance'a düşer.

Çıktılar (output/usdata/):
  summary.json            — tüm evrenin metrik + kurulum etiketi özeti (Claude tek fetch ile okur)
  us_scan.md              — sınıflandırılmış aday listesi (insan okunur)
  {SYM}_1d.csv, {SYM}_1h.csv — ham barlar (CI artifact)

Yayın (GH_TOKEN + GH_PAGES_REPO): nox-signals/usdata/summary.json + us_scan.md
  https://raw.githubusercontent.com/aalpkk/nox-signals/main/usdata/summary.json

Kurulum etiketleri (Alp'in kural seti):
  BREAKOUT_1G : 20g zirvesi bugün kırıldı, hacim>1.5x — b-tipi giriş adayı (kovalanmaz, 15dk onay)
  SINDIRME    : 5g'de >%7 sıçrama + bugün |%|<2, hacim sönük — bölge+onay adayı (EN İYİ)
  BOLGEDE     : SMA20'ye %2 içinde geri çekilmiş, trend yukarı (SMA20>SMA50) — pullback tetiği
  TABAN       : 52h zirveden -%25 altı ama 20g dibinin %5 üstü — ters dönüş izleme
  UZAMIS      : SMA20'nin %15+ üstü — KOVALAMA YOK
  ZAYIF       : SMA20<SMA50 ve 20g dibine yakın — dosya kapalı
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:  # lokal koşularda repo/.env ve bir üst klasördeki .env otomatik yüklenir
    from dotenv import load_dotenv
    for _p in (REPO / ".env", REPO.parent / ".env"):
        load_dotenv(_p)
except ImportError:
    pass

OUT_DIR = REPO / "output" / "usdata"

PRIORITY = ["SINDIRME", "BREAKOUT_1G", "BOLGEDE", "TABAN", "NOTR", "UZAMIS", "ZAYIF"]


def load_universe():
    cfg = yaml.safe_load((REPO / "config" / "us_watch_universe.yaml").read_text())
    hourly = set(cfg.pop("hourly_subset", []))
    universe = {s: grp for grp, syms in cfg.items() for s in syms}
    return universe, hourly


def exchange_map() -> dict:
    """NASDAQ Trader listed dosyalarından ticker → TV borsa öneki (NASDAQ/NYSE/AMEX)."""
    try:
        from markets.us.data import _fetch_us_exchange_map
        return _fetch_us_exchange_map()
    except Exception as e:
        print(f"⚠️ exchange map alınamadı ({e}) — hepsi NASDAQ varsayılacak", flush=True)
        return {}


def fetch_ext(sym: str, exch: str, tf: str, n: int, auth):
    from markets.extfeed import fetch_bars
    df = fetch_bars(f"{exch}:{sym}", tf, n_bars=n, auth=auth)
    if df is None or df.empty:
        raise RuntimeError("empty")
    df = df.copy()
    ny = df["time"].dt.tz_convert("America/New_York")
    df["date"] = ny.dt.strftime("%Y-%m-%d" if tf == "D" else "%Y-%m-%d %H:%M")
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_yf(sym: str, tf: str, n: int):
    import yfinance as yf
    per, iv = ("2y", "1d") if tf == "D" else ("60d", "1h")
    df = yf.Ticker(sym).history(period=per, interval=iv, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("empty")
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    dcol = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={dcol: "date"})
    df["date"] = df["date"].astype(str).str[:(10 if tf == "D" else 16)]
    return df[["date", "open", "high", "low", "close", "volume"]].tail(n)


def get(sym: str, exch: str, tf: str, n: int, auth):
    errs = []
    if auth is not None:
        try:
            return fetch_ext(sym, exch, tf, n, auth), "ext"
        except Exception as e:
            errs.append(f"ext:{e}")
    try:
        return fetch_yf(sym, tf, n), "yf"
    except Exception as e:
        errs.append(f"yf:{e}")
    raise RuntimeError(" | ".join(errs)[:160])


def summarize(grp: str, df: pd.DataFrame) -> dict:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return {
        "group": grp,
        "last_close": round(last, 2), "prev_close": round(prev, 2),
        "chg_pct": round(100 * (last / prev - 1), 2),
        "chg_5d_pct": round(100 * (last / float(c.iloc[-6]) - 1), 2) if len(c) > 6 else None,
        "chg_20d_pct": round(100 * (last / float(c.iloc[-21]) - 1), 2) if len(c) > 21 else None,
        "sma20": round(float(c.tail(20).mean()), 2), "sma50": round(float(c.tail(50).mean()), 2),
        "hi20": round(float(h.tail(20).max()), 2), "lo20": round(float(l.tail(20).min()), 2),
        "hi252": round(float(h.tail(252).max()), 2),
        "off_hi252_pct": round(100 * (last / float(h.tail(252).max()) - 1), 2),
        "atr14": round(float(tr.tail(14).mean()), 2),
        "vol_ratio20": round(float(v.iloc[-1] / max(v.tail(20).mean(), 1)), 2),
        "last_bar_date": str(df["date"].iloc[-1]),
    }


def classify(d: dict):
    last, s20, s50 = d["last_close"], d["sma20"], d["sma50"]
    hi20, lo20 = d["hi20"], d["lo20"]
    chg, chg5 = d["chg_pct"], d.get("chg_5d_pct") or 0
    vr, off_hi = d["vol_ratio20"], d["off_hi252_pct"]
    ext = 100 * (last / s20 - 1)
    tags = []
    if last >= hi20 * 0.999 and chg > 2 and vr > 1.5:
        tags.append("BREAKOUT_1G")
    if chg5 > 7 and abs(chg) < 2 and vr < 1.2:
        tags.append("SINDIRME")
    if abs(ext) < 2 and s20 > s50 and chg5 > -8:
        tags.append("BOLGEDE")
    if off_hi < -25 and last > lo20 * 1.05:
        tags.append("TABAN")
    if ext > 15:
        tags.append("UZAMIS")
    if s20 < s50 and last < lo20 * 1.05:
        tags.append("ZAYIF")
    return tags or ["NOTR"], round(ext, 1)


def session_state_ny() -> str:
    now = pd.Timestamp.now(tz="America/New_York")
    if now.dayofweek >= 5:
        return "weekend"
    t = now.hour * 60 + now.minute
    if t < 9 * 60 + 30:
        return "premarket"
    if t < 16 * 60:
        return "in_session"  # dikkat: son günlük bar KISMİ (canlı bar)
    return "post_close"


def publish_to_signals(rel_path: str, text: str) -> bool:
    """nox-signals reposuna dosya PUT (contents API). GH_TOKEN + GH_PAGES_REPO gerekli."""
    import requests
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")
    if not token or not repo:
        print("⚠️ GH_TOKEN/GH_PAGES_REPO tanımsız — yayın atlandı", flush=True)
        return False
    url = f"https://api.github.com/repos/{repo}/contents/{rel_path}"
    hdr = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    sha = None
    try:
        r = requests.get(url, headers=hdr, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass
    payload = {"message": f"usdata: {rel_path} {dt.datetime.utcnow():%Y-%m-%d %H:%M}Z",
               "branch": "main",
               "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
    if sha:
        payload["sha"] = sha
    for attempt in range(3):
        try:
            r = requests.put(url, headers=hdr, json=payload, timeout=30)
            if r.status_code in (200, 201):
                print(f"✅ yayınlandı: {repo}/{rel_path}", flush=True)
                return True
            if r.status_code >= 500 and attempt < 2:
                time.sleep(3)
                continue
            print(f"⚠️ yayın hatası {r.status_code}: {r.text[:150]}", flush=True)
            return False
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            print(f"⚠️ yayın hatası: {e}", flush=True)
    return False


def build_md(symbols: dict, meta: dict) -> str:
    rows = []
    for sym, d in symbols.items():
        rank = min(PRIORITY.index(t) for t in d["tags"])
        rows.append((rank, sym, d))
    rows.sort(key=lambda r: (r[0], -abs(r[2]["chg_pct"])))
    out = [f"# NOX US tarama — {meta['generated_utc']}Z ({meta['symbols_ok']} sembol, "
           f"NY seans: {meta['session_state_ny']})", ""]
    if meta["session_state_ny"] == "in_session":
        out += ["> Seans içi koşu: son günlük bar KISMİ — 1g% ve HacimX gün sonu değildir.", ""]
    cur = None
    for rank, sym, d in rows:
        head = PRIORITY[rank]
        if head != cur:
            cur = head
            out += ["", f"## {head}", "",
                    "| Sem | Grup | Son | 1g% | 5g% | 20g% | SMA20fark | 52hZirve% | HacimX | Stop öneri (20g dip) | Kaynak |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
        out.append(f"| **{sym}** | {d['group']} | {d['last_close']} | {d['chg_pct']} | "
                   f"{d.get('chg_5d_pct')} | {d.get('chg_20d_pct')} | {d['ext20_pct']}% | "
                   f"{d['off_hi252_pct']} | {d['vol_ratio20']} | {d['lo20']} | {d['source']} |")
    if meta["errors"]:
        out += ["", "## Veri hatası", "",
                ", ".join(f"{k}({v[:60]})" for k, v in meta["errors"].items())]
    return "\n".join(out)


def main() -> int:
    universe, hourly = load_universe()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    auth = None
    try:
        from markets.extfeed import auth_from_env
        auth = auth_from_env()
        _ = auth.token()
        print("[auth] extfeed JWT alındı", flush=True)
    except Exception as e:
        print(f"⚠️ extfeed auth yok ({e}) — tümü yfinance'tan çekilecek", flush=True)

    exch = exchange_map()
    out, errors, t0 = {}, {}, time.time()
    for i, (sym, grp) in enumerate(universe.items(), 1):
        ex = exch.get(sym, "NASDAQ")
        try:
            df, src = get(sym, ex, "D", 300, auth)
            df.to_csv(OUT_DIR / f"{sym}_1d.csv", index=False)
            rec = summarize(grp, df) | {"source": src}
            rec["tags"], rec["ext20_pct"] = classify(rec)
            out[sym] = rec
            if sym in hourly:
                try:
                    dfh, _ = get(sym, ex, "60", 200, auth)
                    dfh.to_csv(OUT_DIR / f"{sym}_1h.csv", index=False)
                except Exception as e:
                    errors[f"{sym}_1h"] = str(e)[:160]
            time.sleep(0.3)
        except Exception as e:
            errors[sym] = str(e)[:160]
        if i % 10 == 0:
            print(f"  [{i}/{len(universe)}] ok={len(out)} err={len(errors)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = {"generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "session_state_ny": session_state_ny(),
            "symbols_ok": len(out), "errors": errors,
            "source_counts": pd.Series([d["source"] for d in out.values()])
                               .value_counts().to_dict() if out else {}}
    summary = json.dumps({"meta": meta, "symbols": out}, indent=1)
    (OUT_DIR / "summary.json").write_text(summary)
    md = build_md(out, meta)
    (OUT_DIR / "us_scan.md").write_text(md, encoding="utf-8")
    print(f"OK {len(out)} sembol, {len(errors)} hata, kaynak={meta['source_counts']}", flush=True)

    publish_to_signals("usdata/summary.json", summary)
    publish_to_signals("usdata/us_scan.md", md)

    if len(out) < len(universe) * 0.6:
        print("!! %60 eşiği altında sembol — FAIL", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
