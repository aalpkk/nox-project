# -*- coding: utf-8 -*-
"""
IPO-RBR Takas Watch — hafta içi akşam: izleme listesinde el dönüşü var mı?
==========================================================================
data/ipo_rbr_watch.json listesi için Matriks AKD çeker, üç alarmı kontrol eder:
  🔴 EL DÖNÜŞÜ : ana toplayıcı (key_buyers, ev-dışı) günlük net SATICI
                 (G < -100K ve < -%5 × kümülatif 3A birikimi)
  🟠 ÇIKIŞ SERİSİ: ICE net akış 2 ardışık gün negatif
  ℹ️ EV bilgisi : arz aracı kurumu akışı ayrıca raporlanır (sinyal sayılmaz)
Sonucu Telegram'a yollar (TG_BOT_TOKEN/TG_CHAT_ID; yoksa konsola).
KAP hatırlatması: ortak pay-satış / tedbir bildirimi kontrolü mesaja eklenir.

Run: PYTHONPATH=. python tools/ipo_rbr_takas_watch.py
Requires: MATRIKS_API_KEY, MATRIKS_CLIENT_ID
"""
import json
import os
import sys
from datetime import date

from agent.matriks_client import MatriksClient
from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day
from core.reports import send_telegram

WATCH = "data/ipo_rbr_watch.json"


def main():
    if not os.environ.get("MATRIKS_API_KEY"):
        print("ERROR: MATRIKS_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    cfg = json.load(open(WATCH, encoding="utf-8"))
    client = MatriksClient()

    lines = [f"📊 <b>IPO-RBR Takas Watch — {date.today().isoformat()}</b>"]
    alerts = 0
    for item in cfg["tickers"]:
        t = item["ticker"]
        house = item.get("house", "")
        keys = item.get("key_buyers", [])
        try:
            flows = client.get_institutional_flow_periods(t, top=10)
            if not flows:
                lines.append(f"⚪ {t}: veri yok")
                continue
            takas = flows_to_takas_data(flows, t)
            kur = takas.get(t, {}).get("kurumlar", [])

            def find(subs):
                return [k for k in kur
                        if any(s in k.get("Aracı Kurum", "").upper() for s in subs)]

            tick_alerts, info = [], []
            for k in find([s.upper() for s in keys]):
                g = k.get("Günlük Fark", 0)
                q3 = k.get("3 Aylık Fark", 0)
                nm = k.get("Aracı Kurum", "?")[:18]
                info.append(f"{nm} G={g:+,.0f} 3A={q3:+,.0f}")
                if g < -100_000 and g < -0.05 * max(q3, 0):
                    tick_alerts.append(f"🔴 EL DÖNÜŞÜ: {nm} G={g:+,.0f}")
            for k in find([house.upper().split()[0]] if house else []):
                info.append(f"[EV] {k.get('Aracı Kurum','?')[:14]} G={k.get('Günlük Fark',0):+,.0f}")

            hist = client.get_daily_flow_history(t, days=3, top=10)
            nets = []
            for ds in sorted(hist.keys()):
                day = flow_to_takas_history_day(hist[ds], t, ds)
                td = day.get(ds, {}).get(t, {})
                nets.append((ds, td.get("net_tip", {}).get("diger", 0)))
            if len(nets) >= 2 and nets[-1][1] < 0 and nets[-2][1] < 0:
                tick_alerts.append(
                    f"🟠 ÇIKIŞ SERİSİ: {nets[-2][1]/1e6:+.1f}M → {nets[-1][1]/1e6:+.1f}M")

            ice_txt = " ".join(f"{d[-5:]}:{v/1e6:+.1f}M" for d, v in nets[-2:])
            mark = "🔴" if any(a.startswith("🔴") for a in tick_alerts) else (
                "🟠" if tick_alerts else "🟢")
            lines.append(f"{mark} <b>{t}</b> {ice_txt} | " + " | ".join(info[:3]))
            for a in tick_alerts:
                lines.append(f"   {a}")
            alerts += len(tick_alerts)
        except Exception as e:
            lines.append(f"⚪ {t}: hata {e}")

    lines.append("")
    lines.append("📋 KAP el-kontrolü: ATATR ortak pay-satış bildirimi / "
                 "tüm liste için tedbir-VBTS duyurusu → kap.org.tr")
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)
    print(f"\n{alerts} alarm.")


if __name__ == "__main__":
    main()
