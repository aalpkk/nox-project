"""
NOX Advisor — takas/AKD katmanı (Matriks).

Elde tutulan her pozisyon için aracı-kurum takas akışı: key-buyer durumu,
bugünkü alıcı/satıcı, ICE çıkış-serisi, yoğunluk + 'EL DÖNÜŞÜ/ÇIKIŞ SERİSİ'
alarmları. tools/portfolio_takas_advisor.py mantığının advisor'a taşınmış hali.

GRACEFUL: Matriks API hata verirse status=MATRIKS_ERROR + mesaj döner (advisor
ÇÖKMEZ); render bu mesajı HTML'de kırmızı banner yapar ki kullanıcı güncellesin.
Anahtar yoksa status=DISABLED (Matriks kapalı).
"""
import os

RETAIL = ("MIDAS",)


def _fmt(n):
    try:
        return f"{n:+,.0f}"
    except Exception:
        return "?"


def load_takas(tickers, key_buyers=None):
    """tickers: elde tutulan hisseler. key_buyers: {ticker: [kurum adları]} (ops).
    Döner: {status, error, per_ticker:{t:{...}}, n_alerts}."""
    if not os.environ.get("MATRIKS_API_KEY"):
        return {"status": "DISABLED", "per_ticker": {}, "n_alerts": 0,
                "note": "MATRIKS_API_KEY yok — takas atlandı"}
    key_buyers = key_buyers or {}
    try:
        from agent.matriks_client import MatriksClient
        from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day
        client = MatriksClient()
    except Exception as e:
        return {"status": "MATRIKS_ERROR", "error": f"istemci kurulamadı: {e}",
                "per_ticker": {}, "n_alerts": 0}

    per, n_alerts, hard_err = {}, 0, None
    for t in tickers:
        keys = [k.upper() for k in key_buyers.get(t, [])]
        try:
            flows = client.get_institutional_flow_periods(t, top=10)
            kur = flows_to_takas_data(flows, t).get(t, {}).get("kurumlar", []) if flows else []
            if not kur:
                per[t] = {"durum": "veri yok"}
                continue

            # key-buyer yoksa: en büyük pozitif 3-aylık (perakende hariç) fallback
            if not keys:
                cand = [k for k in kur if k.get("3 Aylık Fark", 0) > 0
                        and not any(r in k.get("Aracı Kurum", "").upper() for r in RETAIL)]
                if cand:
                    keys = [max(cand, key=lambda x: x.get("3 Aylık Fark", 0))["Aracı Kurum"].upper()]

            alerts, key_txt = [], []
            for k in [x for x in kur if any(s in x.get("Aracı Kurum", "").upper() for s in keys)]:
                g = k.get("Günlük Fark", 0); q = k.get("3 Aylık Fark", 0)
                nm = k.get("Aracı Kurum", "?")[:14]
                key_txt.append(f"{nm} G{_fmt(g)}/3A{_fmt(q)}")
                if g < -100_000 and g < -0.05 * max(q, 0):
                    alerts.append(f"EL DÖNÜŞÜ: {nm} bugün net satıcı (G{_fmt(g)})")

            buyers = sorted(kur, key=lambda x: -x.get("Günlük Fark", 0))[:1]
            sellers = sorted(kur, key=lambda x: x.get("Günlük Fark", 0))[:1]
            top_b = (f"{buyers[0].get('Aracı Kurum','?')[:12]} {_fmt(buyers[0].get('Günlük Fark',0))}"
                     if buyers else "-")
            top_s = (f"{sellers[0].get('Aracı Kurum','?')[:12]} {_fmt(sellers[0].get('Günlük Fark',0))}"
                     if sellers else "-")

            # ICE çıkış-serisi (son 4 gün net 'diğer')
            ice_txt = ""
            try:
                hist = client.get_daily_flow_history(t, days=4, top=10)
                nets = []
                for ds in sorted(hist.keys()):
                    d = flow_to_takas_history_day(hist[ds], t, ds)
                    td = d.get(ds, {}).get(t, {})
                    nets.append((ds, td.get("net_tip", {}).get("diger", 0),
                                 td.get("top3_alici_pct", 0)))
                if len(nets) >= 2 and nets[-1][1] < 0 and nets[-2][1] < 0:
                    alerts.append(f"ÇIKIŞ SERİSİ: {nets[-2][1]/1e6:+.1f}M→{nets[-1][1]/1e6:+.1f}M")
                ice_txt = "  ".join(f"{d[-5:]}:{v/1e6:+.1f}M/{p:.0f}%" for d, v, p in nets[-3:])
            except Exception:
                pass

            per[t] = {"key": key_txt, "top_buyer": top_b, "top_seller": top_s,
                      "ice": ice_txt, "alerts": alerts,
                      "mark": ("🔴" if any("EL DÖNÜŞÜ" in a for a in alerts)
                               else "🟠" if alerts else "🟢")}
            n_alerts += len(alerts)
        except Exception as e:
            # tek ticker hatası → o ticker boş, ama API tamamen ölmüşse yakala
            per[t] = {"durum": f"hata: {str(e)[:60]}"}
            hard_err = str(e)

    # tüm ticker'lar hata verdiyse = API ölü → MATRIKS_ERROR
    if hard_err and all("hata" in v.get("durum", "") for v in per.values() if "durum" in v) \
            and not any("mark" in v for v in per.values()):
        return {"status": "MATRIKS_ERROR", "error": hard_err,
                "per_ticker": per, "n_alerts": 0}
    return {"status": "OK", "per_ticker": per, "n_alerts": n_alerts}
