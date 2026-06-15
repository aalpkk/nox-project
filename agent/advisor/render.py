"""
NOX Advisor — rapor render: Telegram (HTML parse mode) + bağımsız HTML dosyası.

Gizlilik: HTML public GH Pages'e GİTMEZ — Telegram'a document olarak gönderilir.
"""
import os
from pathlib import Path

ADVISOR_DIR = Path(os.environ.get("NOX_OUTPUT_DIR", "output")) / "advisor"

_ACTION_EMOJI = {"HOLD": "🟡", "TRIM": "🟠", "SELL": "🔴"}
_STATUS_EMOJI = {"OK": "✅", "NO_DE_DAY": "⚪", "STALE": "⚠️", "UNAVAILABLE": "❌"}


def _fmt_tl(v):
    return f"{v:,.0f}".replace(",", ".") if v is not None else "—"


def render_telegram_tr(advisory):
    a = advisory
    ps = a["portfolio_summary"]
    mode_tag = " <i>(kural-tabanlı fallback)</i>" if a["mode"] == "deterministic_fallback" else ""

    lines = [
        f"🤖 <b>NOX Portföy Danışmanı — {a['asof']}</b>{mode_tag}",
        f"💼 Varlık: <b>{_fmt_tl(ps['equity_tl'])} TL</b> · Nakit: {_fmt_tl(ps['cash_tl'])} TL "
        f"(%{100 - ps['invested_pct']:.0f}) · Açık risk: %{ps['open_risk_pct']:.1f}",
        "",
    ]

    # girdi durumu — sorun varsa yüksek sesle
    bad = {k: v for k, v in a["inputs_status"].items() if v not in ("OK", "NO_DE_DAY")}
    if a["inputs_status"].get("decision_engine") == "NO_DE_DAY":
        lines.append("⚪ Bugün NO_DE_DAY — DE v1 yeni watchlist üretmedi.")
    if bad:
        lines.append("⚠️ <b>Girdi sorunları:</b> " +
                     ", ".join(f"{k}={v}" for k, v in bad.items()))
        lines.append("")

    rot = a.get("sector_rotation")
    if rot:
        fren = "fren AÇIK ⛔" if (rot.get("brake_primary") or rot.get("brake_explore")) else "fren kapalı"
        stale = " (BAYAT)" if rot.get("status") == "STALE" else ""
        lines.append(f"🔄 Rotasyon monitörü{stale} ({rot.get('bar_date')}): "
                     f"primer={rot.get('state_primary_confirm_on')} · "
                     f"keşif={rot.get('state_explore_confirm_off')} · {fren} <i>(paper)</i>")
        if rot.get("last_bank_ignition"):
            lines.append(f"   ⚡ {rot['last_bank_ignition']}")
        lines.append("")

    hw = a.get("hw_obos")
    if hw:
        stale = " (BAYAT)" if hw.get("status") == "STALE" else ""
        lines.append(f"〰️ HW dönüş genişliği{stale} ({hw.get('scan_date')}): "
                     f"↓{hw.get('n_sat_ob', 0)} tepe / ↑{hw.get('n_al_os', 0)} dip "
                     f"<i>(betimsel, edge yok)</i>")
        lines.append("")

    # pozisyonlar
    if a["position_recommendations"]:
        lines.append("📊 <b>Pozisyonlar</b>")
        for r in sorted(a["position_recommendations"],
                        key=lambda x: ("SELL", "TRIM", "HOLD").index(x["action"])):
            em = _ACTION_EMOJI.get(r["action"], "•")
            flag_txt = f" [{','.join(r['flags'])}]" if r["flags"] else ""
            pnl = f"{r['pnl_pct']:+.1f}%" if r["pnl_pct"] is not None else "—"
            lines.append(f"{em} <b>{r['ticker']}</b> {r['action']} ({r['confidence']}) · "
                         f"%{r['weight_pct']:.1f} ağırlık · PnL {pnl}{flag_txt}")
            if r["rationale_tr"]:
                lines.append(f"   <i>{r['rationale_tr']}</i>")
    else:
        lines.append("📊 Açık pozisyon yok.")
    lines.append("")

    # Trident Tier-1 durumu (backtest BİRİNCİL sinyali — her zaman raporla)
    n_trident = sum(1 for b in a["buy_candidates"] if b.get("trident_tier1"))
    if n_trident:
        trd_names = [b["ticker"] for b in a["buy_candidates"] if b.get("trident_tier1")]
        lines.append(f"🔱 <b>Trident Tier-1: {n_trident} aktif</b> — {', '.join(trd_names[:10])} "
                     "<i>(backtest birincil: temiz-koşucu +6.7)</i>")
    else:
        lines.append("🔱 <i>Trident Tier-1: bugün aktif aday yok (G4 rejim-yukarı kapısı; "
                     "RISK_OFF'ta pasif — normal).</i>")
    lines.append("")

    # AL adayları
    if a["buy_candidates"]:
        lines.append("🟢 <b>AL Adayları</b> <i>(DE v1, paper-track tek-rejim)</i>")
        for b in a["buy_candidates"]:
            add_tag = " (EKLEME)" if b["action"] == "ADD" else ""
            trid = "🔱 " if b.get("trident_tier1") else ""
            conf_bits = []
            if b.get("trident_tier1"):
                conf_bits.append("TRİDENT-T1")
            if (b.get("n_cells") or 1) > 1:
                conf_bits.append(f"DE {b['n_cells']} hücre")
            if b.get("context_lists"):
                conf_bits.append("+" + ",".join(b["context_lists"][:5]))
            conf_txt = f" · 🔗 {' '.join(conf_bits)}" if conf_bits else ""
            lines.append(
                f"🟢 {trid}<b>{b['ticker']}</b>{add_tag} {b['section']} · "
                f"{b['suggested_qty']} adet @ {b['entry_ref']:.2f} · "
                f"stop {b['stop_ref']:.2f} · risk {_fmt_tl(b['risk_tl'])} TL{conf_txt}"
            )
            if b["rationale_tr"]:
                lines.append(f"   <i>{b['rationale_tr']}</i>")
    else:
        lines.append("🟢 Bugün korkulukları geçen AL adayı yok.")

    if a.get("skipped_candidates"):
        sk = a["skipped_candidates"]
        # kalite sırasının başındaki (nakit/limit duvarına takılan) güçlü adaylar
        strong = [s for s in sk if s.get("section") == "EXECUTABLE"][:5]
        if strong:
            lines.append("👀 <b>Nakit/limit duvarına takılan güçlü adaylar</b> "
                         "<i>(boyut yok — bilgi)</i>")
            for s in strong:
                cell = f" ({s['n_cells']} hücre)" if (s.get("n_cells") or 1) > 1 else ""
                ctx = f" +{','.join(s['context_lists'][:4])}" if s.get("context_lists") else ""
                refs = (f" giriş~{s['entry_ref']:.2f} stop~{s['stop_ref']:.2f}"
                        if s.get("entry_ref") and s.get("stop_ref") else "")
                lines.append(f"• {s['ticker']}{cell}{ctx}{refs} — {s['status']}")
        lines.append(f"⛔ Toplam elenen: {len(sk)} aday")
    lines.append("")

    if a.get("scorecard_prev"):
        from agent.advisor.scorecard import format_scorecard_line
        sc = format_scorecard_line(a["scorecard_prev"])
        if sc:
            lines.append(sc)
            lines.append("")
    if a.get("narrative_tr"):
        lines.append(f"🧭 {a['narrative_tr']}")
    if a.get("guardrail_log"):
        lines.append(f"🛡 Korkuluk: {len(a['guardrail_log'])} müdahale")
    lines.append(f"\n⚠️ <i>{a['disclaimer_tr']}</i>")
    return "\n".join(lines)


def render_portfolio_tr(pre, source="?", rev=None):
    """Bot /portfoy çıktısı — pre_check sonucundan canlı özet."""
    lines = [
        "💼 <b>Portföy</b>",
        f"Varlık: <b>{_fmt_tl(pre['equity_tl'])} TL</b> · Nakit: {_fmt_tl(pre['cash_tl'])} TL · "
        f"Yatırımda %{pre['invested_pct']:.1f}",
        f"Açık risk: %{pre['risk']['existing_open_risk_pct']:.1f} "
        f"(tavan %{pre['risk']['cap_pct']})",
        "",
    ]
    if not pre["positions"]:
        lines.append("Açık pozisyon yok. /poz_al ile ekle.")
    for p in sorted(pre["positions"], key=lambda x: -x["weight_pct"]):
        pnl = f"{p['pnl_pct']:+.1f}%" if p["last"] is not None else "fiyat yok"
        flag_txt = f" ⚠️{','.join(p['flags'])}" if p["flags"] else ""
        stop_txt = f" · stop {p['stop']}" if p.get("stop") else ""
        lines.append(f"• <b>{p['ticker']}</b> {p['qty']}×{p['avg_cost']:.2f} → "
                     f"{p['last'] if p['last'] is not None else '—'} ({pnl}) · "
                     f"%{p['weight_pct']:.1f}{stop_txt}{flag_txt}")
    lines.append(f"\n<i>kaynak: {source} · rev {rev[:8] if rev else 'lokal'}</i>")
    return "\n".join(lines)


def render_html(advisory):
    """Basit bağımsız HTML raporu — Telegram document için."""
    a = advisory
    ps = a["portfolio_summary"]

    def rows(items, cols):
        out = ""
        for it in items:
            out += "<tr>" + "".join(f"<td>{it.get(c, '')}</td>" for c in cols) + "</tr>"
        return out

    pos_rows = rows(a["position_recommendations"],
                    ["ticker", "action", "confidence", "qty", "avg_cost", "last",
                     "weight_pct", "pnl_pct", "flags", "rationale_tr"])
    buy_rows = rows(a["buy_candidates"],
                    ["ticker", "section", "suggested_qty", "entry_ref", "stop_ref",
                     "risk_tl", "rationale_tr"])
    inputs = "".join(f"<li>{k}: {_STATUS_EMOJI.get(v, '')} {v}</li>"
                     for k, v in a["inputs_status"].items())
    guard = "".join(f"<li>{g}</li>" for g in a.get("guardrail_log", []))

    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<title>NOX Danışman {a['asof']}</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#e6edf3;margin:24px}}
h1,h2{{color:#58a6ff}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #30363d;padding:6px 10px;font-size:13px;text-align:left}}
th{{background:#161b22}} .meta{{color:#8b949e;font-size:13px}}
.disclaimer{{color:#f0883e;font-size:12px;margin-top:24px}}
</style></head><body>
<h1>NOX Portföy Danışmanı — {a['asof']}</h1>
<p class="meta">mod: {a['mode']} · model: {a.get('model') or '—'} · portföy rev: {a.get('portfolio_rev') or 'lokal'}</p>
<p>Varlık <b>{_fmt_tl(ps['equity_tl'])} TL</b> · Nakit {_fmt_tl(ps['cash_tl'])} TL ·
Yatırımda %{ps['invested_pct']:.1f} · Açık risk %{ps['open_risk_pct']:.1f}
(tavan %{a['risk_summary']['cap_pct']})</p>
<h2>Girdi durumu</h2><ul>{inputs}</ul>
<h2>Pozisyon önerileri</h2>
<table><tr><th>Hisse</th><th>Aksiyon</th><th>Güven</th><th>Adet</th><th>Maliyet</th>
<th>Son</th><th>Ağırlık %</th><th>PnL %</th><th>Flag</th><th>Gerekçe</th></tr>{pos_rows}</table>
<h2>AL adayları</h2>
<table><tr><th>Hisse</th><th>Bölüm</th><th>Adet</th><th>Giriş</th><th>Stop</th>
<th>Risk TL</th><th>Gerekçe</th></tr>{buy_rows}</table>
<h2>Genel değerlendirme</h2><p>{a.get('narrative_tr', '')}</p>
<h2>Korkuluk müdahaleleri</h2><ul>{guard or '<li>yok</li>'}</ul>
<p class="disclaimer">⚠️ {a['disclaimer_tr']}</p>
</body></html>"""
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"advisor_report_{a['asof']}.html"
    path.write_text(html, encoding="utf-8")
    return path
