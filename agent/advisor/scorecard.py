"""
NOX Advisor — skorkart: dünkü öneri vs bugünkü gerçekleşen.

Bugünkü advisory yayınlanmadan ÖNCE, private repodaki advisory_latest (=önceki
rapor) bugünün fiyatlarıyla skorlanır ve nox-private/advisor/scorecard.json'a
append edilir. Ölçüm naif d+1 yaklaşımıdır (kapanış≈dolum varsayımı, fillability
caveat'i sabit) — advisor'ın değer katıp katmadığının kaba göstergesi.
"""
import datetime

from agent.advisor.portfolio import gh_get_json, gh_put_json

SCORECARD_PATH = "advisor/scorecard.json"
MAX_ENTRIES = 500


def score_previous(prev_advisory, prices):
    """Önceki advisory'yi bugünkü fiyatlarla skorla (saf fonksiyon).

    BUY adayı: entry_ref → şimdiki fiyat (alınsaydı getirisi).
    SELL/TRIM: öneri günü son fiyat → şimdiki fiyat (satılsaydı kaçınılan hareket;
    negatifse öneri isabetli).
    """
    if not prev_advisory:
        return None
    items = []
    for b in prev_advisory.get("buy_candidates", []):
        now = prices.get(b["ticker"])
        if now and b.get("entry_ref"):
            items.append({"ticker": b["ticker"], "kind": "buy",
                          "ref_price": b["entry_ref"], "price_now": now,
                          "ret_pct": round(100.0 * (now / b["entry_ref"] - 1.0), 2)})
    for r in prev_advisory.get("position_recommendations", []):
        if r.get("action") in ("SELL", "TRIM") and r.get("last"):
            now = prices.get(r["ticker"])
            if now:
                items.append({"ticker": r["ticker"], "kind": r["action"].lower(),
                              "ref_price": r["last"], "price_now": now,
                              "ret_pct": round(100.0 * (now / r["last"] - 1.0), 2)})
    if not items:
        return None
    buys = [i["ret_pct"] for i in items if i["kind"] == "buy"]
    sells = [i["ret_pct"] for i in items if i["kind"] in ("sell", "trim")]
    return {
        "prev_asof": prev_advisory.get("asof"),
        "prev_mode": prev_advisory.get("mode"),
        "scored_at": datetime.date.today().isoformat(),
        "items": items,
        "buy_mean_ret_pct": round(sum(buys) / len(buys), 2) if buys else None,
        "sell_avoided_mean_pct": round(sum(sells) / len(sells), 2) if sells else None,
    }


def update_scorecard(entry):
    """Skorkart girdisini private repoya append et (best-effort)."""
    if not entry:
        return None
    import os
    repo = os.environ.get("NOX_PORTFOLIO_REPO", "")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    try:
        obj, sha = gh_get_json(repo, SCORECARD_PATH)
        history = obj.get("entries", []) if obj else []
        # aynı prev_asof tekrar skorlanmasın (dispatch yeniden-koşuları)
        if any(e.get("prev_asof") == entry["prev_asof"] for e in history):
            return None
        history.append(entry)
        history = history[-MAX_ENTRIES:]
        return gh_put_json(repo, SCORECARD_PATH, {"entries": history},
                           f"scorecard {entry['prev_asof']}", sha=sha)
    except Exception as e:
        print(f"⚠️ scorecard güncelleme hatası: {e}")
        return None


def format_scorecard_line(entry):
    """Telegram raporu için tek satır 'dünkü isabet' özeti."""
    if not entry:
        return None
    parts = [f"📈 Önceki rapor ({entry['prev_asof']}):"]
    if entry.get("buy_mean_ret_pct") is not None:
        n = sum(1 for i in entry["items"] if i["kind"] == "buy")
        parts.append(f"AL adayları ort. {entry['buy_mean_ret_pct']:+.1f}% ({n})")
    if entry.get("sell_avoided_mean_pct") is not None:
        n = sum(1 for i in entry["items"] if i["kind"] in ("sell", "trim"))
        parts.append(f"SAT/AZALT sonrası hareket ort. {entry['sell_avoided_mean_pct']:+.1f}% ({n})")
    return " · ".join(parts)
