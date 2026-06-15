# -*- coding: utf-8 -*-
"""
Portföy Takas/AKD Advisor — günlük, GitHub cron + Telegram
==========================================================
portfolio-advisor.yaml'daki her pozisyon için:
  💰 Fiyat/KZ : EXTFEED 1h master son barı (canlı-ish; Matriks DEĞİL) → K/Z vs maliyet
  📊 Akış     : Matriks AKD — günlük top alıcı/satıcı + key_buyer 3A birikimi
  🔄 ICE      : Matriks — son günler net akış + top3 alıcı yoğunluğu
  📈 Yoğunlaşma: Matriks settlement top3 %
  Alarm: 🔴 EL DÖNÜŞÜ (key_buyer günlük net satıcı) / 🟠 ICE 2 gün ardışık negatif

Fiyat extfeed'den (output/extfeed_intraday_1h_3y_master.parquet); takas Matriks'ten.
CI'da extfeed master'ı restore-extfeed-master action tazeler. EKDMR gibi extfeed
evreninde olmayan semboller "extfeed'de yok" notuyla geçilir.

Run: set -a; source .env; set +a; PYTHONPATH=. python tools/portfolio_takas_advisor.py
Requires: MATRIKS_API_KEY, MATRIKS_CLIENT_ID  (+ TG_BOT_TOKEN/TG_CHAT_ID)
"""
import os
import re
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

from agent.matriks_client import MatriksClient
from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day
from core.reports import send_telegram

PORTFOLIO = "portfolio-advisor.yaml"
EXTFEED = Path("output/extfeed_intraday_1h_3y_master.parquet")
RETAIL = ("MIDAS",)
STALE_HOURS = 30  # extfeed son barı bundan eskiyse ⚠️ bayat


def fmt(n):
    return f"{n:+,.0f}"


def load_extfeed_prices(tickers):
    """Her ticker için extfeed 1h master'ın SON bar close'u + bar zamanı."""
    if not EXTFEED.exists():
        return {}, None
    df = pd.read_parquet(EXTFEED, columns=["ticker", "ts_istanbul", "close"])
    df = df[df["ticker"].isin(tickers)]
    out = {}
    for t, g in df.groupby("ticker"):
        last = g.sort_values("ts_istanbul").iloc[-1]
        out[t] = {"price": float(last["close"]), "ts": pd.Timestamp(last["ts_istanbul"])}
    master_max = pd.Timestamp(df["ts_istanbul"].max()) if len(df) else None
    return out, master_max


def main():
    if not os.environ.get("MATRIKS_API_KEY"):
        print("ERROR: MATRIKS_API_KEY yok", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(open(PORTFOLIO, encoding="utf-8"))
    positions = cfg.get("positions", [])
    tickers = [p["ticker"] for p in positions]
    client = MatriksClient()

    px_map, master_max = load_extfeed_prices(tickers)
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    master_note = ""
    if master_max is not None:
        age_h = (now_tr - master_max.to_pydatetime()).total_seconds() / 3600
        if age_h > STALE_HOURS:
            master_note = f"  ⚠️ extfeed bayat (son bar {master_max:%m-%d %H:%M})"
    else:
        master_note = "  ⚠️ extfeed master yok"

    lines = [f"💼 <b>Portföy Takas/AKD — {date.today().isoformat()}</b>{master_note}"]
    tot_alerts = 0

    for pos in positions:
        t = pos["ticker"]
        cost = pos.get("cost")
        house = (pos.get("house") or "").upper()
        keys = [k.upper() for k in (pos.get("key_buyers") or [])]
        try:
            # --- EXTFEED fiyat / K-Z ---
            pinfo = px_map.get(t)
            if pinfo:
                px = pinfo["price"]
                kz = (px / cost - 1) if cost else None
                px_txt = f"{px:g}"
                kz_txt = f"{kz:+.1%}" if kz is not None else "?"
            else:
                px = None
                px_txt = "extfeed'de yok"
                kz_txt = "?"

            # --- Matriks akış (G/H/A/3A) ---
            flows = client.get_institutional_flow_periods(t, top=10)
            kur = flows_to_takas_data(flows, t).get(t, {}).get("kurumlar", []) if flows else []

            def find(subs):
                return [k for k in kur if any(s in k.get("Aracı Kurum", "").upper() for s in subs)]

            if not keys and kur:  # fallback: en büyük 3A pozitif, ev/perakende hariç
                cand = [k for k in kur if k.get("3 Aylık Fark", 0) > 0
                        and not (house and house.split()[0] in k.get("Aracı Kurum", "").upper())
                        and not any(r in k.get("Aracı Kurum", "").upper() for r in RETAIL)]
                if cand:
                    keys = [max(cand, key=lambda x: x.get("3 Aylık Fark", 0))["Aracı Kurum"].upper()]

            alerts, key_txt = [], []
            found_keys = find(keys)
            agg_g = sum(k.get("Günlük Fark", 0) for k in found_keys)
            agg_q = sum(max(k.get("3 Aylık Fark", 0), 0) for k in found_keys)
            for k in found_keys:
                g = k.get("Günlük Fark", 0); q = k.get("3 Aylık Fark", 0)
                key_txt.append(f"{k.get('Aracı Kurum','?')[:14]} G{fmt(g)}/3A{fmt(q)}")
            # EL DÖNÜŞÜ = toplayıcı GRUBUN toplamı net satıcı (tek ismin offset'lenen
            # düşük günü tetiklemez; grup >%10 kümülatifi bir günde geri verirse)
            if found_keys and agg_g < -100_000 and agg_g < -0.10 * agg_q:
                alerts.append(f"🔴 EL DÖNÜŞÜ: key_buyer grubu net satıcı "
                              f"(toplam G{fmt(agg_g)} / 3A{fmt(agg_q)})")

            buyers = sorted(kur, key=lambda x: -x.get("Günlük Fark", 0))[:1]
            sellers = sorted(kur, key=lambda x: x.get("Günlük Fark", 0))[:1]
            top_b = f"{buyers[0].get('Aracı Kurum','?')[:12]} {fmt(buyers[0].get('Günlük Fark',0))}" if buyers else "-"
            top_s = f"{sellers[0].get('Aracı Kurum','?')[:12]} {fmt(sellers[0].get('Günlük Fark',0))}" if sellers else "-"

            # --- Matriks ICE ---
            hist = client.get_daily_flow_history(t, days=4, top=10)
            nets = []
            for ds in sorted(hist.keys()):
                d = flow_to_takas_history_day(hist[ds], t, ds)
                td = d.get(ds, {}).get(t, {})
                nets.append((ds, td.get("net_tip", {}).get("diger", 0), td.get("top3_alici_pct", 0)))
            if len(nets) >= 2 and nets[-1][1] < 0 and nets[-2][1] < 0:
                alerts.append(f"🟠 ÇIKIŞ SERİSİ: {nets[-2][1]/1e6:+.1f}M→{nets[-1][1]/1e6:+.1f}M")
            ice_txt = "  ".join(f"{d[-5:]}:{v/1e6:+.1f}M/{p:.0f}%" for d, v, p in nets[-3:])

            conc = ""
            try:
                s = client.get_settlement(t, top=5)
                m = re.search(r"Top 3:\s*([\d.]+)%", (s or {}).get("analysis", ""))
                if m:
                    conc = f"  yoğ.top3 {m.group(1)}%"
            except Exception:
                pass

            mark = "🔴" if any(a.startswith("🔴") for a in alerts) else ("🟠" if alerts else "🟢")
            lines.append(f"\n{mark} <b>{t}</b>  {px_txt}  K/Z {kz_txt}{conc}")
            if key_txt:
                lines.append(f"   key: {' | '.join(key_txt)}")
            lines.append(f"   bugün alıcı: {top_b} | satıcı: {top_s}")
            if ice_txt:
                lines.append(f"   ICE: {ice_txt}")
            for a in alerts:
                lines.append(f"   {a}")
            tot_alerts += len(alerts)
        except Exception as e:
            lines.append(f"\n⚪ <b>{t}</b>: hata {e}")

    lines.append(f"\n{'🔔 '+str(tot_alerts)+' alarm' if tot_alerts else '✅ alarm yok'}")
    lines.append("📋 KAP el-kontrol: ortak pay-satış / tedbir-VBTS / %5 eşik → kap.org.tr")
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
