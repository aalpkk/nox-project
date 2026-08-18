#!/usr/bin/env python3
"""NOX US PREMARKET SCAN — izleme evreni premarket taraması (yfinance prepost 5m).

config/us_watch_universe.yaml evreni için bugünün premarket (04:00-09:30 ET)
5 dakikalık barlarından: son premarket fiyat, önceki seans kapanışına göre gap %
ve premarket aralık çıkarılır.

Notlar:
- extfeed TV WS hattı normal seans barı verir; premarket kaynağı yfinance
  prepost'tur (bilinçli tercih — BIST-kritik TV session'ına dokunulmaz).
- Önceki kapanış 5m verisinin kendisinden alınır (önceki seansın 09:30-16:00
  son barı) — Yahoo'nun günlük barı premarket saatlerinde NaN Close verebiliyor.
- Yahoo premarket bar hacmi güvenilir değil (çoğunlukla 0) — hacim raporlanmaz.

Çıktılar: output/usdata/premarket_scan.md + premarket.json
Yayın: nox-signals/usdata/premarket_scan.md + premarket.json
Cron: hafta içi 12:50 UTC (TR 15:50, NY 08:50 — açılışa 40 dk).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.us_watch_scan import load_universe, publish_to_signals

OUT_DIR = REPO / "output" / "usdata"
GAP_FLAG = 2.0  # |gap%| esigi


def main() -> int:
    import yfinance as yf
    universe, _ = load_universe()
    syms = list(universe)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    now_ny = pd.Timestamp.now(tz="America/New_York")
    today = now_ny.normalize()
    pm_start = today + pd.Timedelta(hours=4)
    pm_end = today + pd.Timedelta(hours=9, minutes=30)

    # 5 gunluk 5m prepost: hem bugunun premarket'i hem onceki seansin
    # regular-hours kapanisi buradan cikar (gunluk bar NaN glitch'inden bagimsiz).
    m5 = yf.download(syms, period="5d", interval="5m", prepost=True,
                     group_by="ticker", threads=True, progress=False,
                     auto_adjust=False)

    rows, errors = {}, {}
    for s in syms:
        try:
            b_all = m5[s].dropna(subset=["Close"])
            # onceki seans kapanisi: bugunden onceki, regular saatlerdeki son bar
            prev = b_all[b_all.index < today]
            prev = prev[(prev.index.hour * 60 + prev.index.minute >= 9 * 60 + 30)
                        & (prev.index.hour < 16)]
            if prev.empty:
                raise RuntimeError("onceki seans bari yok")
            prev_close = float(prev["Close"].iloc[-1])
            prev_date = str(prev.index[-1])[:10]

            b = b_all[(b_all.index >= pm_start) & (b_all.index < pm_end)]
            if b.empty:
                rows[s] = {"group": universe[s], "prev_close": round(prev_close, 2),
                           "prev_close_date": prev_date, "pm_last": None,
                           "pm_gap_pct": None, "pm_bars": 0}
                continue
            pm_last = float(b["Close"].iloc[-1])
            rows[s] = {
                "group": universe[s],
                "prev_close": round(prev_close, 2), "prev_close_date": prev_date,
                "pm_last": round(pm_last, 2),
                "pm_gap_pct": round(100 * (pm_last / prev_close - 1), 2),
                "pm_hi": round(float(b["High"].max()), 2),
                "pm_lo": round(float(b["Low"].min()), 2),
                "pm_bars": int(len(b)),
                "pm_last_bar_ny": str(b.index[-1])[:16],
            }
        except Exception as e:
            errors[s] = str(e)[:120]

    meta = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "ny_time": str(now_ny)[:16], "premarket_window_ny": "04:00-09:30",
            "symbols_ok": len(rows), "errors": errors}
    js = json.dumps({"meta": meta, "symbols": rows}, indent=1)
    (OUT_DIR / "premarket.json").write_text(js)

    traded = {s: r for s, r in rows.items() if r.get("pm_gap_pct") is not None}
    srt = sorted(traded.items(), key=lambda kv: -abs(kv[1]["pm_gap_pct"]))
    out = [f"# NOX US PREMARKET — {meta['ny_time']} ET ({len(traded)}/{len(rows)} sembolde islem)",
           "",
           f"Gap referansi: onceki seans ({srt[0][1]['prev_close_date'] if srt else '?'}) kapanisi. "
           f"|gap| >= {GAP_FLAG}% isaretli. Yahoo premarket hacim vermez.", "",
           "| Sem | Grup | Onceki kapanis | PM son | GAP% | PM aralik | Son bar (ET) | Bayrak |",
           "|---|---|---|---|---|---|---|---|"]
    for s, r in srt:
        flag = ""
        if r["pm_gap_pct"] >= GAP_FLAG:
            flag = "GAP_UP"
        elif r["pm_gap_pct"] <= -GAP_FLAG:
            flag = "GAP_DOWN"
        out.append(f"| **{s}** | {r['group']} | {r['prev_close']} | {r['pm_last']} | "
                   f"{r['pm_gap_pct']} | {r['pm_lo']}-{r['pm_hi']} | "
                   f"{r['pm_last_bar_ny'][11:]} | {flag} |")
    quiet = [s for s, r in rows.items() if r.get("pm_gap_pct") is None]
    if quiet:
        out += ["", f"Premarket islem yok: {', '.join(quiet)}"]
    if errors:
        out += ["", "## Veri hatasi", ", ".join(f"{k}({v[:40]})" for k, v in errors.items())]
    md = "\n".join(out)
    (OUT_DIR / "premarket_scan.md").write_text(md, encoding="utf-8")
    print(f"OK {len(traded)}/{len(rows)} sembol premarket, {len(errors)} hata", flush=True)

    publish_to_signals("usdata/premarket.json", js)
    publish_to_signals("usdata/premarket_scan.md", md)
    return 0 if len(rows) >= len(syms) * 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())
