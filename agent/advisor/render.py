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


def _tv(ticker):
    """Ticker'ı TradingView grafik linkine çevir (HTML; .tv-link _NOX_CSS'te)."""
    t = str(ticker)
    return (f'<a class="tv-link" target="_blank" '
            f'href="https://www.tradingview.com/chart/?symbol=BIST:{t}">{t}</a>')


def _sec(title, body, sub="", open=True):
    """HTML bölümünü açılır-kapanır <details> dropdown olarak sar (kullanıcı isteği)."""
    op = " open" if open else ""
    subh = f" <span class='sec-sub'>{sub}</span>" if sub else ""
    return (f"<details class='sec-d'{op}><summary>{title}{subh}</summary>"
            f"<div class='sec-body'>{body}</div></details>")


_MACRO_CAT_ORDER = ["BIST", "US", "Kripto", "Emtia", "FX", "Faiz"]


def _rdp_text(rdp):
    """XU100 RDP rejim satırı (BİLGİ). long DIŞI = risk-off/SAT uyarısı."""
    if not rdp or rdp.get("status") == "UNAVAILABLE" or not rdp.get("regime"):
        return None
    reg = rdp.get("regime")
    stale = " (BAYAT)" if rdp.get("status") == "STALE" else ""
    if rdp.get("is_sell"):
        return (f"🔴 <b>XU100 RDP: {reg.upper()}</b>{stale} ({rdp.get('date')}) — "
                f"risk-OFF/SAT rejimi <i>(RDP overlay, canlı kapı kapalı, bilgi)</i>")
    return (f"🟢 XU100 RDP: {reg}{stale} ({rdp.get('date')}) — risk-ON "
            f"<i>(RDP overlay, bilgi)</i>")


def _macro_groups(snapshot):
    """category → [satır dict] (sıralı). reversal/trend etiketli."""
    by = {}
    for m in snapshot or []:
        by.setdefault(m.get("category", "?"), []).append(m)
    out = []
    for cat in _MACRO_CAT_ORDER + [c for c in by if c not in _MACRO_CAT_ORDER]:
        if cat in by:
            out.append((cat, by[cat]))
    return out


def _macro_inst_txt(m):
    """Tek enstrüman özet metni: ad chg1d/chg5d trend [dip↗/tepe↘]."""
    def pct(v):
        return f"{v:+.1f}%" if isinstance(v, (int, float)) else "—"
    rev = m.get("reversal") or ""
    rev_tag = f" <b>{rev}</b>" if rev else ""
    rsi = m.get("rsi")
    rsi_t = f" RSI{rsi:.0f}" if isinstance(rsi, (int, float)) else ""
    return (f"{m.get('name')}: {pct(m.get('chg_1d'))} (5g {pct(m.get('chg_5d'))}) "
            f"· {m.get('trend', '?')}{rsi_t}{rev_tag}")


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
    # çapraz tarayıcılar (context) — 'tavan' = bist-tavan-SCREENER üyeliği
    # (tavan adayı/devam/seri), "bugün kilitlendi" DEĞİL → yanıltmamak için netleştir.
    _ctx_lbl = {"tavan": "tavan-aday(scr)", "tavan_kandidat": "tavan-kandidat(scr)"}
    for c in (b.get("context_lists") or []):
        parts.append(_ctx_lbl.get(c, c))
    # özel hatlar
    if b.get("weekly_lead"):
        tf = "+".join(b.get("weekly_lead_tf") or [])
        parts.append(f"1w-LİDER✓({tf})" if tf else "1w-LİDER✓")
    if b.get("trident_geo"):
        parts.append("TRİDENT" + ("✓G4" if b.get("trident_tier1") else ""))
    # DE'nin KENDİ trident sınıfı (geo'dan ayrı; SIL_DEEP/WEEKLY_BIRTH_ACTIVE/
    # D_PCT_30PLUS/TIER1) — advisor geo'su 0 olsa bile DE'nin etiketlediğini gösterir.
    for t in (b.get("trident_tag") or "").split(";"):
        t = t.strip()
        if t and f"🔱{t}" not in parts:
            parts.append(f"🔱{t}")
    wf = b.get("trident_weekly_family") or ""
    if wf and not b.get("weekly_lead"):  # 1w-LİDER✓ zaten varsa tekrar etme
        parts.append(f"1w-birth({wf})")
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

    # TARAMA TAZELİĞİ — bayat kaynak varsa yüksek sesle uyar (kullanıcı endişesi)
    stale_fr = [x for x in (a.get("freshness") or []) if x.get("stale")]
    if stale_fr:
        lines.append("🕒 <b>BAYAT tarama:</b> " +
                     ", ".join(f"{x['kaynak']}({x.get('tarih')})" for x in stale_fr) +
                     " <i>(asof'tan eski — dikkat)</i>")
        lines.append("")

    # XU100 RDP rejimi (risk-ON/OFF) — en üstte, makro pusula
    rdp_t = _rdp_text(a.get("xu100_rdp"))
    if rdp_t:
        lines.append(rdp_t)
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
        # ÖNE ÇIKAN SEKTÖRLER + o sektördeki DE adayların
        fav = rot.get("favorable_sectors") or []
        sec_l = {s["kod"]: s for s in (rot.get("sectors") or [])}
        if fav:
            def _sdesc(k):
                s = sec_l.get(k, {})
                dp = s.get("dipten_pct")
                return f"{k}({s.get('durum','?')}{f' dipten+{dp}%' if dp else ''})"
            lines.append("   📈 öne çıkan sektörler: " + ", ".join(_sdesc(k) for k in fav))
            rp = a.get("rotation_picks") or {}
            if rp:
                for si, info in rp.items():
                    dcm = info.get("down_capture") or {}
                    tk_txt = ", ".join(
                        f"{t}{f'(dc{dcm[t]:.2f})' if isinstance(dcm.get(t), (int, float)) else ''}"
                        for t in info["tickers"])
                    lines.append(f"      → <b>{si}</b>: {tk_txt}")
                lines.append("   <i>(sektör/zaman seçimi doğrulanmış; sektör-içi sıra = DÜŞÜK "
                             "down-capture/defansif önce — vs-XU100 +alfa, both-period ✓)</i>")
        # LİDER sektörlerde defansif hisseler (TÜM likit üye, dc sıralı)
        sdc = a.get("sector_dc_picks") or {}
        if sdc:
            lines.append("   📉 <b>sektör durumu + defansif (düşük dc) hisseler:</b>")
            for si, info in sdc.items():
                rs = info.get("rows") or []
                tk = ", ".join(f"{r['ticker']}{'🎯' if r.get('in_de') else ''}(dc{r['dc']:+.2f})"
                               for r in rs[:6])
                lines.append(f"      <b>{si}</b> [{info.get('durum')}]: {tk}")
            else:
                lines.append("   <i>(bu sektörlerde bugün DE adayı yok)</i>")
        lines.append("")

    # GENİŞ MAKRO — NASDAQ/S&P, layer1 kripto, XAU/XAG/metaller, FX, faiz (BİLGİ)
    msnap = a.get("macro_snapshot") or []
    if msnap:
        lines.append("🌍 <b>Makro</b> <i>(trend + dipten/tepeden dönüş, bilgi)</i>")
        for cat, items in _macro_groups(msnap):
            rev = [m for m in items if m.get("reversal")]
            head = ", ".join(_macro_inst_txt(m).replace(" <b>", " ").replace("</b>", "")
                             for m in items[:6])
            lines.append(f"• <b>{cat}</b>: {head}")
            for m in rev:
                lines.append(f"    {m.get('reversal')} {m.get('name')}")
        lines.append("")

    hw = a.get("hw_obos")
    if hw:
        stale = " (BAYAT)" if hw.get("status") == "STALE" else ""
        lines.append(f"〰️ HW dönüş genişliği{stale} ({hw.get('scan_date')}): "
                     f"↓{hw.get('n_sat_ob', 0)} tepe / ↑{hw.get('n_al_os', 0)} dip "
                     f"<i>(betimsel, edge yok)</i>")
        lines.append("")

    # Tavan V1 canlı tarama (tavan-scan-live — izleme bağlamı, AL adayı DEĞİL)
    tv = a.get("tavan_lock") or {}
    if tv.get("status") in ("OK", "STALE") and (tv.get("picks") or []):
        st = " BAYAT" if tv.get("status") == "STALE" else ""
        tks = []
        for p in (tv["picks"] or [])[:10]:
            t = p.get("ticker") or "?"
            ml = p.get("ml_s")
            tav = "🟥" if p.get("at_tavan") else ""
            tks.append(f"{t}{tav}{f'({ml:.2f})' if isinstance(ml,(int,float)) else ''}")
        lines.append(f"🔒 <b>Tavan V1 canlı</b>{st} <i>(scan-live, izleme bağlamı)</i>: " + ", ".join(tks))
        lines.append("")

    # cluster3 açık adaylar (arketip forward-paper — bilgi)
    c3o = a.get("cluster3_open") or []
    if c3o:
        names = [f"{c['ticker']}{'🎯' if c.get('in_de') else ''}" for c in c3o[:12]]
        lines.append(f"🧬 <b>cluster3 açık</b> <i>(paper, bilgi)</i>: " + ", ".join(names))
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

    # Trident durumu. geo = advisor G4'süz geometri (D∈20-30 ∧ SIL-derin ∧ BoS);
    # tier1 = DE G4-dahil. AYRICA DE'nin kendi trident-tag'i (SIL_DEEP/WEEKLY_BIRTH_
    # ACTIVE/D_PCT_30PLUS) raporlanır — geo 0 olsa da. Buy + bütçe-blocked HEPSİ taranır.
    _all_cand = a["buy_candidates"] + (a.get("skipped_candidates") or [])
    geo = [b["ticker"] for b in _all_cand if b.get("trident_geo")]
    t1 = set(b["ticker"] for b in _all_cand if b.get("trident_tier1"))
    if geo:
        marked = [f"{t}{'✓G4' if t in t1 else ''}" for t in geo[:12]]
        lines.append(f"🔱 <b>Trident geo (G4'süz): {len(geo)} aday</b> — {', '.join(marked)} "
                     "<i>(D∈20-30 ∧ SIL-derin; backtest birincil +6.7; ✓G4=rejim teyitli)</i>")
    else:
        lines.append("🔱 <i>Trident geo: bugün geometri adayı yok (D∈20-30 ∧ SIL-derin); "
                     "tier1 G4 rejim-yukarı ister — düşen tapede kapalı.</i>")
    tagged = [b for b in _all_cand if (b.get("trident_tag") or "").strip()]
    if tagged:
        from collections import Counter
        tcnt = Counter(t.strip() for b in tagged
                       for t in (b.get("trident_tag") or "").split(";") if t.strip())
        tag_sum = ", ".join(f"{k}×{v}" for k, v in tcnt.most_common())
        ex = ", ".join(b["ticker"] for b in tagged[:12])
        lines.append(f"🔱 <b>DE trident-tag: {len(tagged)} isim</b> ({tag_sum}) — {ex}")
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
            dc = b.get("down_capture")
            dc_txt = f" · dc {dc:+.2f}" if isinstance(dc, (int, float)) else ""
            _ti = ((a.get("takas") or {}).get("per_ticker") or {}).get(b["ticker"]) or {}
            tk_txt = (f" · takas {_ti['mark']}{'⚠️' if _ti.get('alerts') else ''}"
                      if _ti.get("mark") else "")
            lines.append(
                f"🟢 {trid}<b>{b['ticker']}</b>{add_tag} [{tf}{' '+st if st else ''}] · "
                f"{b['suggested_qty']} adet @ {b['entry_ref']:.2f} · "
                f"stop {b['stop_ref']:.2f} · risk {_fmt_tl(b['risk_tl'])} TL{dc_txt}{tk_txt}{mtf_txt}{conf_txt}"
            )
            if b["rationale_tr"]:
                lines.append(f"   <i>{b['rationale_tr']}</i>")
    else:
        lines.append("🟢 Bugün korkulukları geçen AL adayı yok.")

    ce = a.get("chased_excluded") or []
    if ce:
        lst = ", ".join("{}(+{:.0f}%)".format(c["ticker"], c["runup_pct"]) for c in ce[:10])
        lines.append(f"⛔ <b>Kovalama-filtresi:</b> {len(ce)} aday elendi "
                     f"(yakın +%20 koşmuş, weekly birth yok): {lst}")
        lines.append("")

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
                cells += f"<td><b>{_tv(v)}</b></td>" if c == "ticker" else f"<td>{v}</td>"
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
    # takas per_ticker (pozisyon+aday) → AL adaylarına takas işareti/özeti
    _tkp = (a.get("takas") or {}).get("per_ticker") or {}
    def _takas_cell(tkr):
        info = _tkp.get(tkr)
        if not info or "mark" not in info:
            return info.get("durum", "—") if info else "—"
        al = " · ".join(info.get("alerts", []))
        return f"{info['mark']} {info.get('top_buyer','')}{(' | '+al) if al else ''}"
    buy_view = []
    for b in a["buy_candidates"]:
        mtf = b.get("mtf_birth") or []
        mtf_p = f"çoklu-TF birth:{'+'.join(mtf)}; " if len(mtf) > 1 else ""
        dc = b.get("down_capture")
        buy_view.append({**b,
                         "confluence": mtf_p + ", ".join(_full_confluence(b)),
                         "dc": f"{dc:+.2f}" if isinstance(dc, (int, float)) else "—",
                         "takas": _takas_cell(b["ticker"]),
                         "entry_ref": _r2(b.get("entry_ref")),
                         "stop_ref": _r2(b.get("stop_ref")),
                         "risk_tl": _ri(b.get("risk_tl")),
                         "suggested_qty": _ri(b.get("suggested_qty"))})
    buy_rows = rows(buy_view,
                    ["ticker", "timeframe", "state", "confluence", "dc", "takas", "section",
                     "suggested_qty", "entry_ref", "stop_ref", "risk_tl"])

    # BÜTÇE-KISITSIZ TAM LİSTE: buy + skipped (nakit/limit duvarına takılanlar) —
    # portföy bütçesi olmasaymış gibi TÜM DE adayları. Altta <details> ile açılır.
    skipped = a.get("skipped_candidates") or []
    full_items = []
    for b in a["buy_candidates"]:
        full_items.append({**b, "_durum": "ALINABİLİR"})
    for s in skipped:
        full_items.append({**s, "_durum": f"⛔ {s.get('status', 'bütçe-dışı')}"})
    full_rows = ""
    for it in full_items:
        mtf = it.get("mtf_birth") or []
        mtf_p = f"çoklu-TF birth:{'+'.join(mtf)}; " if len(mtf) > 1 else ""
        conf = mtf_p + ", ".join(_full_confluence(it))
        dc = it.get("down_capture")
        dc_t = f"{dc:+.2f}" if isinstance(dc, (int, float)) else "—"
        full_rows += (
            f"<tr><td><b>{_tv(it['ticker'])}</b></td>"
            f"<td>{it.get('timeframe', '')}</td><td>{it.get('state', '')}</td>"
            f"<td>{conf}</td><td>{dc_t}</td><td>{it.get('section', '')}</td>"
            f"<td>{_r2(it.get('entry_ref'))}</td><td>{_r2(it.get('stop_ref'))}</td>"
            f"<td>{it.get('_durum')}</td></tr>")
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
    _cand_set = set(tk.get("candidate_tickers") or [])
    _pos_set = set(p.get("ticker") for p in a.get("position_recommendations", []))
    def _rol(tkr):  # rol etiketi: pozisyon / AL adayı
        if tkr in _pos_set and tkr in _cand_set:
            return " <span class='tag'>poz+aday</span>"
        if tkr in _cand_set:
            return " <span class='tag tag-w'>AL adayı</span>"
        return ""
    takas_rows = ""
    for tkr, info in (tk.get("per_ticker") or {}).items():
        if "mark" not in info:
            takas_rows += (f"<tr><td>{_tv(tkr)}{_rol(tkr)}</td><td colspan='4'>{info.get('durum','—')}</td></tr>")
            continue
        al = "<br>".join(info.get("alerts", [])) or "—"
        takas_rows += (f"<tr><td>{info['mark']} {_tv(tkr)}{_rol(tkr)}</td>"
                       f"<td>{' | '.join(info.get('key', [])) or '—'}</td>"
                       f"<td>{info.get('top_buyer', '—')}</td>"
                       f"<td>{info.get('top_seller', '—')}</td>"
                       f"<td>{info.get('ice', '—')}</td>"
                       f"<td style='color:#f85149'>{al}</td></tr>")
    # ── MAKRO & REJİM HTML bölümü (RDP + sektör rotasyon + geniş makro) ──
    rdp = a.get("xu100_rdp") or {}
    rdp_html = ""
    if rdp.get("regime"):
        sell = rdp.get("is_sell")
        color = "#f85149" if sell else "#7a9e7a"
        tag = "risk-OFF / SAT" if sell else "risk-ON"
        st = " · BAYAT" if rdp.get("status") == "STALE" else ""
        rdp_html = (f"<div style='border-left:3px solid {color};background:var(--bg-card);"
                    f"border:1px solid var(--border-subtle);border-radius:10px;padding:10px 14px;"
                    f"margin:8px 0'><b style='color:{color}'>XU100 RDP: {rdp['regime'].upper()}</b> "
                    f"— {tag} <span class='meta'>({rdp.get('date')}{st} · RDP overlay, canlı kapı "
                    f"kapalı, bilgi)</span></div>")
    rot = a.get("sector_rotation") or {}
    rot_html = ""
    if rot.get("bar_date"):
        fren = "fren AÇIK ⛔" if (rot.get("brake_primary") or rot.get("brake_explore")) else "fren kapalı"
        st = " · BAYAT" if rot.get("status") == "STALE" else ""
        ign = (f"<br>⚡ {rot['last_bank_ignition']}" if rot.get("last_bank_ignition") else "")
        rot_html = (f"<p class='note'>🔄 <b>Sektör rotasyon monitörü</b> ({rot.get('bar_date')}{st}): "
                    f"primer=<b>{rot.get('state_primary_confirm_on')}</b> · "
                    f"keşif=<b>{rot.get('state_explore_confirm_off')}</b> · {fren} "
                    f"<i>(paper-forward, canlı kapı kapalı)</i>{ign}</p>")
        fav = rot.get("favorable_sectors") or []
        sec_l = {s["kod"]: s for s in (rot.get("sectors") or [])}
        rp = a.get("rotation_picks") or {}
        if fav:
            pick_rows = ""
            for k in fav:
                s = sec_l.get(k, {})
                dp = s.get("dipten_pct")
                info = rp.get(k) or {}
                tks = info.get("tickers") or []
                dcm = info.get("down_capture") or {}
                if tks:
                    tlinks = ", ".join(
                        _tv(t) + (f"<span class='meta'> dc{dcm[t]:.2f}</span>"
                                  if isinstance(dcm.get(t), (int, float)) else "")
                        for t in tks)
                else:
                    tlinks = "<span class='meta'>bugün DE adayı yok</span>"
                pick_rows += (f"<tr><td><b>{k}</b></td>"
                              f"<td>{s.get('durum','?')}{f' · {dp:+.1f}%/5g' if dp is not None else ''}</td>"
                              f"<td>{tlinks}</td></tr>")
            rot_html += (
                "<p class='note'>📈 <b>Rotasyonda öne çıkan sektörler → o sektördeki DE adayların</b> "
                "<i>(sektör/zaman seçimi DOĞRULANMIŞ; sektör-içi sıra = DÜŞÜK down-capture/defansif "
                "önce — vs-XU100 +alfa her iki dönem ✓; dc düşük = düşerken az düşen)</i></p>"
                "<div class='nox-table-wrap'><table><thead><tr><th>Sektör</th><th>Durum</th>"
                f"<th>DE adayların</th></tr></thead><tbody>{pick_rows}</tbody></table></div>")
    hw = a.get("hw_obos") or {}
    hw_html = ""
    if hw.get("scan_date"):
        st = " · BAYAT" if hw.get("status") == "STALE" else ""
        hw_html = (f"<p class='note'>〰️ <b>HW dönüş genişliği</b> ({hw.get('scan_date')}{st}): "
                   f"↓{hw.get('n_sat_ob', 0)} tepe / ↑{hw.get('n_al_os', 0)} dip "
                   f"<i>(betimsel, edge yok)</i></p>")
    msnap = a.get("macro_snapshot") or []
    macro_table = ""
    if msnap:
        def _pct(v):
            return f"{v:+.1f}%" if isinstance(v, (int, float)) else "—"
        mrows = ""
        for cat, items in _macro_groups(msnap):
            for i, m in enumerate(items):
                rev = m.get("reversal") or ""
                rev_c = "#7a9e7a" if rev == "dip↗" else ("#f85149" if rev == "tepe↘" else "")
                rev_h = f"<span style='color:{rev_c};font-weight:700'>{rev}</span>" if rev else ""
                c1 = m.get("chg_1d"); c1c = "#7a9e7a" if (c1 or 0) >= 0 else "#f85149"
                cat_cell = (f"<td rowspan='{len(items)}' style='vertical-align:top'>"
                            f"<b>{cat}</b></td>") if i == 0 else ""
                rsi = m.get("rsi")
                mrows += (f"<tr>{cat_cell}<td>{m.get('name')}</td>"
                          f"<td>{m.get('price') if m.get('price') is not None else '—'}</td>"
                          f"<td style='color:{c1c}'>{_pct(c1)}</td><td>{_pct(m.get('chg_5d'))}</td>"
                          f"<td>{_pct(m.get('chg_1m'))}</td>"
                          f"<td>{f'{rsi:.0f}' if isinstance(rsi,(int,float)) else '—'}</td>"
                          f"<td>{m.get('trend','?')}</td><td>{rev_h}</td></tr>")
        macro_table = (
            "<div class='nox-table-wrap'><table><thead><tr><th>Kategori</th><th>Enstrüman</th>"
            "<th>Fiyat</th><th>1g</th><th>5g</th><th>1a</th><th>RSI</th><th>Trend</th><th>Dönüş</th>"
            f"</tr></thead><tbody>{mrows}</tbody></table></div>")
    # "Makro & Rejim" bölümü: RDP/sektör/hw paragrafları VEYA makro tablosu varsa render et
    macro_html = ""
    if rdp_html or rot_html or hw_html or macro_table:
        macro_html = _sec("🌍 Makro & Rejim", f"{rdp_html}{rot_html}{hw_html}{macro_table}",
                          sub="trend + dipten/tepeden dönüş · bilgi")

    # haftalık-lider çakışma (aday-dışı) — HTML bölümü (NOX temalı)
    wlw = a.get("weekly_lead_watch") or []
    weekly_lead_html = ""
    if wlw:
        bar = a.get("weekly_lead_bar") or "?"
        wl_rows = "".join(
            f"<tr><td><b>{_tv(w['ticker'])}</b></td>"
            f"<td><span class='tag tag-w'>1w</span> + {'+'.join(w.get('tf') or [])}</td></tr>"
            for w in wlw)
        weekly_lead_html = _sec(
            "📐 Haftalık-lider çakışma",
            f"<p class='note'>Son kapanmış haftalık barda (<b>{bar}</b>) <code>mb_1w above_mb_birth</code> + "
            f"aynı hafta günlük/5h <code>above_mb_birth</code>. DE adayı DEĞİL — çapraz-TF yapı hizalanması.</p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>Çakışan TF</th></tr></thead>"
            f"<tbody>{wl_rows}</tbody></table></div>",
            sub="aday-dışı · bilgi")

    takas_html = ""
    if takas_banner or takas_rows:
        n_al = tk.get("n_alerts", 0)
        tk_body = takas_banner
        if takas_rows:
            tk_body += ("<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th>"
                        "<th>Key kurum (G/3A)</th><th>Bugün alıcı</th><th>Bugün satıcı</th>"
                        "<th>ICE (3g)</th><th>Alarm</th></tr></thead><tbody>"
                        + takas_rows + "</tbody></table></div>")
        takas_html = _sec("💼 Takas / AKD", tk_body,
                          sub=f"Matriks{' · '+str(n_al)+' alarm' if n_al else ''}")

    # ── TAVAN V1 CANLI (tavan-scan-live) bölümü — tavan izleme bağlamı, AL adayı DEĞİL ──
    tv = a.get("tavan_lock") or {}
    tavan_html = ""
    tpicks = tv.get("picks") or []
    if tv.get("status") in ("OK", "STALE") and tpicks:
        def _pf(x, fmt):
            return fmt.format(x) if isinstance(x, (int, float)) else (x if x else "—")
        st = " · BAYAT" if tv.get("status") == "STALE" else ""
        trows = ""
        for p in tpicks[:30]:
            tkr = p.get("ticker") or "?"
            ml = p.get("ml_s")
            v1 = "✓" if p.get("v1_candidate") else ""
            tav = "🟥tavanda" if p.get("at_tavan") else ("vurdu" if p.get("hit_tavan") else "")
            ml_c = "#7a9e7a" if (isinstance(ml, (int, float)) and ml >= 0.65) else "var(--text-primary)"
            trows += (f"<tr><td><b>{_tv(str(tkr))}</b></td>"
                      f"<td style='color:{ml_c}'>{_pf(ml, '{:.2f}')}</td><td>{v1}</td>"
                      f"<td>{_pf(p.get('close'), '{:.2f}')}</td>"
                      f"<td>{_pf(p.get('pct_from_prev'), '{:+.1f}%')}</td><td>{tav}</td></tr>")
        tavan_html = _sec(
            "🔒 Tavan V1 canlı tarama",
            f"<p class='note'>Tavan V1 canlı aday taraması (<code>tavan-scan-live</code>, "
            f"scan {tv.get('scan_asof','?')}{st}). ml_s = V1 kalite (≥0.65 yeşil), ✓=v1_candidate, "
            f"🟥=şu an tavanda. AL adayı DEĞİL — tavan izleme bağlamı (paper tek-rejim). "
            f"Çıkış: <code>{tv.get('exit_rules','—')}</code></p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>ml_s</th><th>v1</th>"
            f"<th>Kapanış</th><th>Δ önceki</th><th>Tavan</th></tr></thead>"
            f"<tbody>{trows}</tbody></table></div>",
            sub=f"{len(tpicks)} aday · izleme bağlamı")

    # ── LİDER/ARMED SEKTÖRLERDE DEFANSİF HİSSELER (düşük down-capture) ──
    sdc = a.get("sector_dc_picks") or {}
    sector_dc_html = ""
    if sdc:
        _dem = {"TETİK": "🟢TETİK", "ARMED": "🟠ARMED", "LİDER": "🔵LİDER"}
        blocks = ""
        for si, info in sdc.items():
            rows = info.get("rows") or []
            durum = info.get("durum") or "?"
            cells = "".join(
                f"<tr><td><b>{_tv(r['ticker'])}</b>{' 🎯DE' if r.get('in_de') else ''}</td>"
                f"<td style=\"color:{'#7a9e7a' if r['dc'] < 0.7 else 'var(--text-primary)'}\">{r['dc']:+.2f}</td>"
                f"<td>{r['adv']/1e6:.0f}M₺</td></tr>" for r in rows)
            blocks += (f"<p class='note' style='margin-top:8px'><b>{si}</b> "
                       f"<span class='meta'>{_dem.get(durum, durum)}</span></p>"
                       f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th>"
                       f"<th>dc</th><th>likidite</th></tr></thead><tbody>{cells}</tbody></table></div>")
        sector_dc_html = _sec(
            "📉 Lider sektörlerde defansif hisseler",
            f"<p class='note'>ARMED/LİDER sektörlerin TÜM likit üyeleri, DÜŞÜK down-capture "
            f"(düşerken az düşen=defansif) önce sıralı. dc&lt;0.7 yeşil. 🎯DE=aynı zamanda DE "
            f"adayı. Doğrulanmış edge: düşük dc vs-XU100 +alfa her iki dönem ✓ (lider sektör/zaman "
            f"seçimi + sektör-içi defansif).</p>{blocks}",
            sub="down-capture · doğrulanmış defansif faktör")

    # ── TARAMA TAZELİĞİ paneli (kullanıcı: dahil edilen taramalar güncel olmayabiliyor) ──
    fr = a.get("freshness") or []
    fresh_html = ""
    if fr:
        n_stale = sum(1 for x in fr if x.get("stale"))
        frows = "".join(
            f"<tr><td>{x['kaynak']}</td><td>{x.get('tarih') or '—'}</td>"
            f"<td>{'⚠️ BAYAT' if x.get('stale') else '✓ güncel'}</td></tr>"
            for x in fr)
        fresh_html = _sec(
            "📅 Tarama tazeliği",
            f"<p class='note'>Advisor'a giren her kaynağın son veri tarihi. asof <b>{a['asof']}</b>'tan "
            f"≥1 İŞGÜNÜ eski = BAYAT (hafta sonu hariç). {n_stale} bayat / {len(fr)} kaynak.</p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Kaynak</th><th>Son tarih</th>"
            f"<th>Durum</th></tr></thead><tbody>{frows}</tbody></table></div>",
            sub=f"{n_stale} bayat / {len(fr)} kaynak", open=bool(n_stale))

    # ── KOVALAMA-FİLTRESİ ile elenenler (yakın +%20 koşmuş, weekly birth yok) ──
    ce = a.get("chased_excluded") or []
    chased_html = ""
    if ce:
        crows = "".join(f"<tr><td><b>{_tv(c['ticker'])}</b></td>"
                        f"<td style='color:#f85149'>+{c['runup_pct']:.0f}%</td>"
                        f"<td>{c.get('section', '')}</td></tr>" for c in ce)
        chased_html = _sec(
            "⛔ Kovalama-filtresi — elenen adaylar",
            f"<p class='note'>Yakın zamanda <b>≥+%20 koşmuş</b> VE weekly birth (mb_1w "
            f"above) OLMAYAN adaylar aday-havuzundan çıkarıldı (kullanıcı kuralı — uzamış "
            f"hisseyi haftalık yapı desteği olmadan kovalama). Yakın-koşu extfeed'den.</p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>Yakın koşu</th>"
            f"<th>DE bölüm</th></tr></thead><tbody>{crows}</tbody></table></div>",
            sub=f"{len(ce)} elendi · bilgi", open=False)

    # ── cluster3 STANDALONE bölümü (açık arketip-paper havuzu — örtüşmese de) ──
    c3o = a.get("cluster3_open") or []
    cluster3_html = ""
    if c3o:
        c3rows = "".join(
            f"<tr><td><b>{_tv(c['ticker'])}</b>{' 🎯DE' if c.get('in_de') else ''}</td>"
            f"<td>{c.get('signal_date') or '—'}</td></tr>" for c in c3o)
        cluster3_html = _sec(
            "🧬 cluster3 açık adaylar",
            f"<p class='note'>cluster3 arketip forward-paper havuzu (son 30g açık, henüz "
            f"realize olmamış). 🎯DE = aynı zamanda bugünkü DE adayı. Backtest'te zayıf/negatif "
            f"edge → bilgi/paper-track.</p>"
            f"<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>Sinyal tarihi</th>"
            f"</tr></thead><tbody>{c3rows}</tbody></table></div>",
            sub=f"{len(c3o)} açık aday · bilgi", open=False)

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
details.fulllist {{ margin-top:26px; border:1px solid var(--border-subtle); border-radius:12px;
  background:var(--bg-card); padding:0 4px; }}
details.fulllist > summary {{ cursor:pointer; list-style:none; padding:14px 16px; font-family:var(--font-display);
  font-weight:700; font-size:1.0rem; color:var(--nox-gold); user-select:none; }}
details.fulllist > summary::-webkit-details-marker {{ display:none; }}
details.fulllist > summary:hover {{ color:var(--text-primary); }}
details.fulllist[open] > summary {{ border-bottom:1px solid var(--border-subtle); }}
details.fulllist .note {{ padding:10px 16px 0; }}
/* açılır-kapanır bölüm başlıkları (tüm section'lar dropdown) */
details.sec-d {{ margin:14px 0; border:1px solid var(--border-subtle); border-radius:12px;
  background:rgba(13,13,16,0.4); overflow:hidden; }}
details.sec-d > summary {{ cursor:pointer; list-style:none; padding:12px 16px; font-family:var(--font-display);
  font-weight:700; font-size:1.08rem; color:var(--text-primary); user-select:none;
  display:flex; align-items:baseline; gap:10px; }}
details.sec-d > summary::-webkit-details-marker {{ display:none; }}
details.sec-d > summary::before {{ content:'▸'; color:var(--nox-gold); font-size:0.8em; transition:transform .15s; }}
details.sec-d[open] > summary::before {{ transform:rotate(90deg); }}
details.sec-d > summary:hover {{ color:var(--nox-gold); }}
details.sec-d[open] > summary {{ border-bottom:1px solid var(--border-subtle); }}
details.sec-d .sec-body {{ padding:12px 16px; }}
details.sec-d .sec-body > .nox-table-wrap {{ margin:0; }}
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

  {macro_html}

  {_sec("📋 Girdi durumu", f"<ul class='inputs'>{inputs}</ul>")}

  {fresh_html}

  {_sec("📊 Pozisyon önerileri",
        "<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>Aksiyon</th><th>Güven</th>"
        "<th>Adet</th><th>Maliyet</th><th>Son</th><th>Ağırlık %</th><th>PnL %</th><th>Flag</th><th>Gerekçe</th>"
        f"</tr></thead><tbody>{pos_rows}</tbody></table></div>")}

  {takas_html}

  {_sec("🟢 AL adayları",
        "<div class='nox-table-wrap'><table><thead><tr><th>Hisse</th><th>TF</th><th>Setup</th>"
        "<th>🔗 Tüm çakışma</th><th>dc</th><th>💼 Takas</th><th>Bölüm</th><th>Adet</th><th>Giriş</th><th>Stop</th><th>Risk ₺</th>"
        f"</tr></thead><tbody>{buy_rows}</tbody></table></div>",
        sub="DE v1 · paper-track · dc=down-capture (düşük=defansif) · takas=Matriks akış")}

  {chased_html}

  {weekly_lead_html}

  {sector_dc_html}

  {tavan_html}

  {cluster3_html}

  {_sec("🧭 Genel değerlendirme", f"<div class='narrative'>{a.get('narrative_tr', '') or '—'}</div>")}

  {_sec("🛡 Korkuluk müdahaleleri", f"<ul class='guard'>{guard or '<li>yok</li>'}</ul>", open=False)}

  <details class="fulllist">
    <summary>📋 Bütçe kısıtı olmasa — TÜM liste ({len(full_items)}) ▾</summary>
    <p class="note">Portföy nakit/limit kısıtı dikkate alınmadan TÜM DE adayları
    (alınabilir + bütçe duvarına takılanlar). Durum kolonu hangisinin bütçeyle
    elendiğini gösterir.</p>
    <div class="nox-table-wrap"><table><thead><tr><th>Hisse</th><th>TF</th><th>Setup</th>
    <th>🔗 Tüm çakışma</th><th>dc</th><th>Bölüm</th><th>Giriş</th><th>Stop</th><th>Durum</th>
    </tr></thead><tbody>{full_rows}</tbody></table></div>
  </details>

  <p class="disclaimer">⚠️ {a['disclaimer_tr']}</p>
</div>
</body></html>"""
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"advisor_report_{a['asof']}.html"
    path.write_text(html, encoding="utf-8")
    return path
