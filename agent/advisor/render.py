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


def _full_confluence(b):
    """Bir adayın TÜM sinyal çakışması (AL kararı olmasa da): DE-tarafı families
    (mb/bb çoklu-TF birth/retest, triangle_break, hb, paper — TAZE) + çapraz tarayıcılar
    (RT/sbt/nox_v3/alsat/tavan — latest_signals snapshot'ından) + cluster3/trident/HW."""
    parts = []
    # DE-tarafı families → okunur kod (kaynak[tf'ler])
    fams = (b.get("families") or "").split(";") if b.get("families") else []
    de = {}
    for f in fams:
        if "__" not in f:
            continue
        head, state = f.split("__", 1)
        # head: mb_5h / bb_1d / tr_1d / hb / line_tr / paper...
        bits = head.split("_")
        tf = next((x for x in bits if x in ("5h", "1d", "1w", "1mo")), "")
        if head.startswith("tr"):
            key = "triangle"
        elif head.startswith(("mb", "bb")):
            key = "mb-birth" if "above_mb_birth" in state else \
                  ("mb-retest" if "retest" in state else "mb-mit")
        elif head.startswith("hb"):
            key = "horizontal_base"
        elif "paper" in head:
            key = "paper"
        else:
            key = head
        de.setdefault(key, set())
        if tf:
            de[key].add(tf)
    for k, tfs in de.items():
        tford = sorted(tfs, key=lambda t: {"5h": 0, "1d": 1, "1w": 2, "1mo": 3}.get(t, 9))
        parts.append(f"{k}[{','.join(tford)}]" if tford else k)
    # çapraz tarayıcılar (context)
    for c in (b.get("context_lists") or []):
        parts.append(c)
    # özel hatlar
    if b.get("weekly_lead"):
        tf = "+".join(b.get("weekly_lead_tf") or [])
        parts.append(f"1w-LİDER✓({tf})" if tf else "1w-LİDER✓")
    if b.get("trident_geo"):
        parts.append("TRİDENT" + ("✓G4" if b.get("trident_tier1") else ""))
    return parts


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
            # TÜM sinyal çakışması (AL kararı olmasa da) — kırpmadan
            conf = _full_confluence(b)
            mtf = b.get("mtf_birth") or []
            mtf_txt = f" · çoklu-TF birth: {'+'.join(mtf)}" if len(mtf) > 1 else ""
            conf_txt = f"\n   🔗 çakışma: {', '.join(conf)}" if conf else ""
            tf = b.get("timeframe") or "?"
            st = (b.get("state") or "").replace("_", " ")[:18]
            lines.append(
                f"🟢 {trid}<b>{b['ticker']}</b>{add_tag} [{tf}{' '+st if st else ''}] · "
                f"{b['suggested_qty']} adet @ {b['entry_ref']:.2f} · "
                f"stop {b['stop_ref']:.2f} · risk {_fmt_tl(b['risk_tl'])} TL{mtf_txt}{conf_txt}"
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

    # HAFTALIK-LİDER çakışma — aday-DIŞI (al kararı değil, bilgi): son kapanmış
    # haftalık barda mb_1w above + aynı hafta günlük/5h above. DE adayı olmasa da raporla.
    wlw = a.get("weekly_lead_watch") or []
    if wlw:
        bar = a.get("weekly_lead_bar") or "?"
        lines.append(f"📐 <b>Haftalık-lider çakışma</b> <i>(1w birth {bar} + günlük/5h, "
                     f"aday-dışı, bilgi)</i>")
        for w in wlw:
            tf = "+".join(w.get("tf") or [])
            lines.append(f"• <b>{w['ticker']}</b> 1w+{tf}")
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
            cells = ""
            for c in cols:
                v = it.get(c, "")
                cells += f"<td><b>{v}</b></td>" if c == "ticker" else f"<td>{v}</td>"
            out += f"<tr>{cells}</tr>"
        return out

    pos_rows = rows(a["position_recommendations"],
                    ["ticker", "action", "confidence", "qty", "avg_cost", "last",
                     "weight_pct", "pnl_pct", "flags", "rationale_tr"])
    # her adaya TAM çakışma string'i + formatlı sayılar (HTML kolonu için);
    # advisory dict'ini MUTASYON YAPMA — Telegram render aynı dict'i sonra kullanıyor.
    def _r2(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return x
    def _ri(x):
        try:
            return f"{float(x):,.0f}"
        except Exception:
            return x
    buy_view = []
    for b in a["buy_candidates"]:
        mtf = b.get("mtf_birth") or []
        mtf_p = f"çoklu-TF birth:{'+'.join(mtf)}; " if len(mtf) > 1 else ""
        buy_view.append({**b,
                         "confluence": mtf_p + ", ".join(_full_confluence(b)),
                         "entry_ref": _r2(b.get("entry_ref")),
                         "stop_ref": _r2(b.get("stop_ref")),
                         "risk_tl": _ri(b.get("risk_tl")),
                         "suggested_qty": _ri(b.get("suggested_qty"))})
    buy_rows = rows(buy_view,
                    ["ticker", "timeframe", "state", "confluence", "section",
                     "suggested_qty", "entry_ref", "stop_ref", "risk_tl"])
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
    # haftalık-lider çakışma (aday-dışı) — HTML bölümü (NOX temalı)
    wlw = a.get("weekly_lead_watch") or []
    weekly_lead_html = ""
    if wlw:
        bar = a.get("weekly_lead_bar") or "?"
        wl_rows = "".join(
            f"<tr><td><b>{w['ticker']}</b></td>"
            f"<td><span class='tag tag-w'>1w</span> + {'+'.join(w.get('tf') or [])}</td></tr>"
            for w in wlw)
        weekly_lead_html = (
            f"<h2 class='sec'>📐 Haftalık-lider çakışma <span class='sec-sub'>aday-dışı · bilgi</span></h2>"
            f"<p class='note'>Son kapanmış haftalık barda (<b>{bar}</b>) <code>mb_1w above_mb_birth</code> + "
            f"aynı hafta günlük/5h <code>above_mb_birth</code>. DE adayı DEĞİL — çapraz-TF yapı hizalanması.</p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>Çakışan TF</th></tr></thead>"
            f"<tbody>{wl_rows}</tbody></table></div>")

    takas_html = ""
    if takas_banner or takas_rows:
        n_al = tk.get("n_alerts", 0)
        takas_html = (f"<h2 class='sec'>Takas / AKD <span class='sec-sub'>Matriks"
                      f"{' · '+str(n_al)+' alarm' if n_al else ''}</span></h2>{takas_banner}")
        if takas_rows:
            takas_html += ("<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th>"
                           "<th>Key kurum (G/3A)</th><th>Bugün alıcı</th><th>Bugün satıcı</th>"
                           "<th>ICE (3g)</th><th>Alarm</th></tr></thead><tbody>"
                           + takas_rows + "</tbody></table></div>")

    from core.reports import _NOX_CSS
    mode_tr = ("kural-tabanlı" if a["mode"] == "deterministic_fallback" else "LLM")
    inv = ps["invested_pct"]
    risk_pct = ps["open_risk_pct"]
    cap = a["risk_summary"]["cap_pct"]

    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nyx Portföy Tavsiye Ekranı — {a['asof']}</title>
<style>
{_NOX_CSS}
.adv-wrap {{ position:relative; z-index:1; max-width:1180px; margin:0 auto; padding:20px 18px 48px; }}
.adv-header {{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap;
  gap:12px; padding:18px 0 16px; margin-bottom:18px; border-bottom:1px solid var(--border-subtle); }}
.adv-logo {{ font-family:var(--font-display); font-size:1.7rem; font-weight:800; letter-spacing:-0.03em;
  background:linear-gradient(135deg,#c9a96e,#e8dcc8,#a8876a); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; background-clip:text; line-height:1.05; }}
.adv-logo .sub {{ display:block; font-size:0.42em; font-weight:600; letter-spacing:0.18em;
  text-transform:uppercase; -webkit-text-fill-color:var(--text-muted); margin-top:3px; }}
.adv-meta {{ text-align:right; font-size:0.78rem; color:var(--text-muted); font-family:var(--font-mono); line-height:1.6; }}
.adv-meta b {{ color:var(--nox-gold); }}
.adv-stats {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:22px; }}
.adv-stat {{ background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:14px;
  padding:10px 16px; min-width:130px; }}
.adv-stat .lbl {{ font-size:0.62rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; }}
.adv-stat .val {{ font-family:var(--font-mono); font-size:1.05rem; font-weight:700; color:var(--text-primary); margin-top:2px; }}
.adv-stat .val.gold {{ color:var(--nox-gold); }}
.adv-stat .val.warn {{ color:var(--nox-red); }}
h2.sec {{ font-family:var(--font-display); font-size:1.1rem; font-weight:700; color:var(--text-primary);
  margin:26px 0 10px; display:flex; align-items:baseline; gap:10px; }}
h2.sec .sec-sub {{ font-size:0.7rem; font-weight:500; color:var(--text-muted); text-transform:uppercase;
  letter-spacing:0.06em; font-family:var(--font-mono); }}
.note {{ font-size:0.76rem; color:var(--text-secondary); line-height:1.5; margin-bottom:10px; }}
.note code, td code {{ font-family:var(--font-mono); color:var(--nox-gold); font-size:0.92em; }}
ul.inputs {{ list-style:none; display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }}
ul.inputs li {{ background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:20px;
  padding:5px 12px; font-size:0.72rem; font-family:var(--font-mono); color:var(--text-secondary); }}
ul.guard {{ font-size:0.76rem; color:var(--text-secondary); line-height:1.6; padding-left:18px; }}
.tag {{ display:inline-block; padding:1px 6px; border-radius:4px; font-size:0.66rem; font-weight:600;
  font-family:var(--font-mono); background:var(--nox-gold-dim); color:var(--nox-gold); }}
.tag-w {{ background:rgba(138,122,158,0.14); color:#a89ec0; }}
.narrative {{ background:var(--bg-card); border:1px solid var(--border-subtle); border-left:3px solid var(--nox-gold);
  border-radius:10px; padding:14px 18px; font-size:0.85rem; line-height:1.6; color:var(--text-primary); }}
.disclaimer {{ color:var(--nox-orange); font-size:0.72rem; margin-top:28px; line-height:1.5;
  border-top:1px solid var(--border-subtle); padding-top:14px; }}
td b {{ color:var(--text-primary); }}
</style></head><body>
<div class="aurora-bg"><div class="aurora-layer aurora-layer-1"></div>
<div class="aurora-layer aurora-layer-2"></div><div class="aurora-layer aurora-layer-3"></div></div>
<div class="mesh-overlay"></div>
<div class="adv-wrap">
  <header class="adv-header">
    <div class="adv-logo">NYX<span class="sub">Portföy Tavsiye Ekranı</span></div>
    <div class="adv-meta"><b>{a['asof']}</b> · mod: {mode_tr}<br>
      model: {a.get('model') or '—'} · rev: {a.get('portfolio_rev') or 'lokal'}</div>
  </header>

  <div class="adv-stats">
    <div class="adv-stat"><div class="lbl">Varlık</div><div class="val gold">{_fmt_tl(ps['equity_tl'])} ₺</div></div>
    <div class="adv-stat"><div class="lbl">Nakit</div><div class="val">{_fmt_tl(ps['cash_tl'])} ₺</div></div>
    <div class="adv-stat"><div class="lbl">Yatırımda</div><div class="val">%{inv:.1f}</div></div>
    <div class="adv-stat"><div class="lbl">Açık risk</div><div class="val{' warn' if risk_pct > cap else ''}">%{risk_pct:.1f} <span style="font-size:0.6rem;color:var(--text-muted)">/ %{cap}</span></div></div>
  </div>

  <h2 class="sec">Girdi durumu</h2>
  <ul class="inputs">{inputs}</ul>

  <h2 class="sec">Pozisyon önerileri</h2>
  <div class="nox-table-wrap"><table><thead><tr><th>Hisse</th><th>Aksiyon</th><th>Güven</th>
  <th>Adet</th><th>Maliyet</th><th>Son</th><th>Ağırlık %</th><th>PnL %</th><th>Flag</th><th>Gerekçe</th>
  </tr></thead><tbody>{pos_rows}</tbody></table></div>

  {takas_html}

  <h2 class="sec">AL adayları <span class="sec-sub">DE v1 · paper-track tek-rejim</span></h2>
  <div class="nox-table-wrap"><table><thead><tr><th>Hisse</th><th>TF</th><th>Setup</th>
  <th>🔗 Tüm çakışma</th><th>Bölüm</th><th>Adet</th><th>Giriş</th><th>Stop</th><th>Risk ₺</th>
  </tr></thead><tbody>{buy_rows}</tbody></table></div>

  {weekly_lead_html}

  <h2 class="sec">Genel değerlendirme</h2>
  <div class="narrative">{a.get('narrative_tr', '') or '—'}</div>

  <h2 class="sec">Korkuluk müdahaleleri</h2>
  <ul class="guard">{guard or '<li>yok</li>'}</ul>

  <p class="disclaimer">⚠️ {a['disclaimer_tr']}</p>
</div>
</body></html>"""
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"advisor_report_{a['asof']}.html"
    path.write_text(html, encoding="utf-8")
    return path
