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

    # Trident durumu (backtest BİRİNCİL — her zaman raporla). trident_geo = G4'süz
    # geometri (rejim-bağımsız, RISK_OFF'ta da fırlar); trident_tier1 = G4-dahil teyit.
    geo = [b["ticker"] for b in a["buy_candidates"] if b.get("trident_geo")]
    t1 = set(b["ticker"] for b in a["buy_candidates"] if b.get("trident_tier1"))
    if geo:
        marked = [f"{t}{'✓G4' if t in t1 else ''}" for t in geo[:12]]
        lines.append(f"🔱 <b>Trident geo (G4'süz): {len(geo)} aday</b> — {', '.join(marked)} "
                     "<i>(D∈20-30 ∧ SIL-derin; backtest birincil +6.7; ✓G4=rejim teyitli)</i>")
    else:
        lines.append("🔱 <i>Trident: bugün geometri adayı yok (D∈20-30 ∧ SIL-derin koşulu).</i>")
    lines.append("")

    # AL adayları
    if a["buy_candidates"]:
        lines.append("🟢 <b>AL Adayları</b> <i>(DE v1, paper-track tek-rejim)</i>")
        for b in a["buy_candidates"]:
            add_tag = " (EKLEME)" if b["action"] == "ADD" else ""
            trid = "🔱 " if b.get("trident_geo") else ""
            conf_bits = []
            if b.get("trident_geo"):
                conf_bits.append("TRİDENT" + ("✓G4" if b.get("trident_tier1") else "-geo"))
            nc = b.get("n_cells") or 1
            if nc > 1:
                conf_bits.append(f"{nc} hücre konfluens")  # NOT timeframe — kaç sinyal birlikte
            if b.get("context_lists"):
                conf_bits.append("+" + ",".join(b["context_lists"][:5]))
            conf_txt = f" · 🔗 {' '.join(conf_bits)}" if conf_bits else ""
            # tf = GERÇEK zaman dilimi (5h/1d/1w/1mo); state = setup tipi
            tf = b.get("timeframe") or "?"
            st = (b.get("state") or "").replace("_", " ")[:18]
            lines.append(
                f"🟢 {trid}<b>{b['ticker']}</b>{add_tag} [{tf}{' '+st if st else ''}] · "
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
    # takas/AKD özeti + Matriks hata uyarısı
    tk = a.get("takas") or {}
    if tk.get("status") == "MATRIKS_ERROR":
        lines.append(f"🔴 <b>MATRİKS HATASI</b> — takas güncellenemedi: {tk.get('error','?')} "
                     "(oturumu yenile, HTML'de detay)")
        lines.append("")
    elif tk.get("status") == "OK":
        alarmlı = [f"{info['mark']} {t}" for t, info in tk.get("per_ticker", {}).items()
                   if info.get("alerts")]
        if alarmlı:
            lines.append("💼 <b>Takas alarmı:</b> " + ", ".join(alarmlı) +
                         " <i>(detay HTML)</i>")
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
                    ["ticker", "timeframe", "state", "n_cells", "section",
                     "suggested_qty", "entry_ref", "stop_ref", "risk_tl", "rationale_tr"])
    inputs = "".join(f"<li>{k}: {_STATUS_EMOJI.get(v, '')} {v}</li>"
                     for k, v in a["inputs_status"].items())
    guard = "".join(f"<li>{g}</li>" for g in a.get("guardrail_log", []))

    # ── TAKAS/AKD bölümü + Matriks hata banner'ı ──
    tk = a.get("takas") or {}
    takas_banner = ""
    if tk.get("status") == "MATRIKS_ERROR":
        takas_banner = (f'<div style="background:#5a1a1a;border:1px solid #f85149;color:#ffb3ab;'
                        f'padding:12px;border-radius:6px;margin:12px 0;font-weight:bold">'
                        f'⚠️ MATRİKS API HATASI — takas verisi güncellenemedi: '
                        f'{tk.get("error","?")}<br>→ Matriks oturumunu/anahtarını yenile, '
                        f'takas bilgisi bu raporda EKSİK.</div>')
    elif tk.get("status") == "DISABLED":
        takas_banner = ('<div style="background:#3a3a1a;border:1px solid #d29922;color:#e3d18a;'
                        'padding:10px;border-radius:6px;margin:12px 0">'
                        'ℹ️ Takas kapalı (MATRIKS_API_KEY yok).</div>')
    takas_rows = ""
    for tkr, info in (tk.get("per_ticker") or {}).items():
        if "mark" not in info:
            takas_rows += (f"<tr><td>{tkr}</td><td colspan='4'>{info.get('durum','—')}</td></tr>")
            continue
        al = "<br>".join(info.get("alerts", [])) or "—"
        takas_rows += (f"<tr><td>{info['mark']} {tkr}</td>"
                       f"<td>{' | '.join(info.get('key', [])) or '—'}</td>"
                       f"<td>{info.get('top_buyer', '—')}</td>"
                       f"<td>{info.get('top_seller', '—')}</td>"
                       f"<td>{info.get('ice', '—')}</td>"
                       f"<td style='color:#f85149'>{al}</td></tr>")
    takas_html = ""
    if takas_banner or takas_rows:
        n_al = tk.get("n_alerts", 0)
        takas_html = (f"<h2>Takas / AKD (Matriks){' — '+str(n_al)+' alarm' if n_al else ''}</h2>"
                      f"{takas_banner}")
        if takas_rows:
            takas_html += ("<table><tr><th>Hisse</th><th>Key kurum (G/3A)</th>"
                           "<th>Bugün alıcı</th><th>Bugün satıcı</th><th>ICE (3g)</th>"
                           "<th>Alarm</th></tr>" + takas_rows + "</table>")

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
{takas_html}
<h2>AL adayları</h2>
<table><tr><th>Hisse</th><th>TF (5h/1d/1w)</th><th>Setup</th><th>Hücre</th><th>Bölüm</th><th>Adet</th><th>Giriş</th><th>Stop</th>
<th>Risk TL</th><th>Gerekçe</th></tr>{buy_rows}</table>
<h2>Genel değerlendirme</h2><p>{a.get('narrative_tr', '')}</p>
<h2>Korkuluk müdahaleleri</h2><ul>{guard or '<li>yok</li>'}</ul>
<p class="disclaimer">⚠️ {a['disclaimer_tr']}</p>
</body></html>"""
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"advisor_report_{a['asof']}.html"
    path.write_text(html, encoding="utf-8")
    return path
