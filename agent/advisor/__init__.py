"""
NOX Advisor — günlük AI portföy danışmanı.

Akış: portföy → doğrulanmış sinyaller (DE v1 / tavan lock / cluster-3) →
bağlam (tarayıcılar + makro) → canlı fiyat → korkuluk ön-hesap →
context pack (persist) → tek LLM çağrısı (hata→fallback) → post-validasyon →
advisory persist + Telegram + HTML.

Kullanım: python -m agent.advisor --asof 2026-06-13 --notify
"""
import datetime

from zoneinfo import ZoneInfo


def resolve_asof(asof=None):
    if asof:
        return asof
    return datetime.datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat()


def run_advisor(asof=None, notify=False, dry_run=False, use_llm=True,
                portfolio_path=None, publish_latest=True, llm_mode="auto"):
    from agent.advisor import signals, guardrails, context_pack, synthesis, render, scorecard
    from agent.advisor.portfolio import Portfolio, publish_advisory_latest, fetch_advisory_latest

    asof = resolve_asof(asof)
    print(f"🤖 NOX Advisor — asof={asof}")

    # 1) portföy
    pf = Portfolio.load(path=portfolio_path)
    print(f"   portföy: {pf.source} rev={pf.rev or '—'} · "
          f"{len(pf.tickers())} pozisyon · nakit {pf.data['cash_tl']:,.0f} TL")

    # 2) doğrulanmış sinyaller + önceki advisory (skorkart için, yayından ÖNCE)
    validated = signals.load_validated_signals(asof)
    de = validated["decision_engine"]
    print(f"   DE v1: {de['status']} ({len(de['buy_rows'])} aday) · "
          f"tavan: {validated['tavan_lock']['status']} · "
          f"cluster3: {validated['cluster3']['status']}")
    try:
        prev_advisory = fetch_advisory_latest()
    except Exception:
        prev_advisory = None

    # 3) canlı fiyat (pozisyonlar + adaylar + önceki rapor ticker'ları)
    tickers = set(pf.tickers()) | {r["ticker"] for r in de["buy_rows"]}
    if prev_advisory and prev_advisory.get("asof") != asof:
        tickers |= {b["ticker"] for b in prev_advisory.get("buy_candidates", [])}
        tickers |= {r["ticker"] for r in prev_advisory.get("position_recommendations", [])}
    tickers = sorted(tickers)
    prices = signals.fetch_prices(tickers)
    print(f"   fiyat: {len(prices)}/{len(tickers)} ticker")

    # 3b) skorkart: önceki rapor bugünkü fiyatlarla (aynı gün yeniden-koşuda atla)
    score_entry = None
    if prev_advisory and prev_advisory.get("asof") != asof:
        score_entry = scorecard.score_previous(prev_advisory, prices)
        if score_entry:
            print(f"   skorkart: {scorecard.format_scorecard_line(score_entry)}")

    # 4) bağlam
    context = signals.load_context_signals(asof, tickers_of_interest=tickers)
    macro = signals.load_macro()

    from agent.advisor import takas as takas_mod  # 4c'de değil — AL adaylarını da kapsamak için sentez SONRASI yüklenir

    # 4d) KOVALAMA FİLTRESİ (kullanıcı kuralı): yakın zamanda ≥+%20 koşmuş hisseyi
    #     weekly birth (mb_1w above_mb_birth) YOKSA aday-havuzundan ÇIKAR — LLM hiç görmesin.
    #     Yakın-koşu extfeed'den; weekly-birth mb_birth_xtf'ten (son kapanmış haftalık bar).
    chased_excluded = []
    try:
        wbirth = set((signals.mb_birth_xtf(asof).get("per_ticker") or {}).keys())  # weekly above olanlar
        de_tk = [r["ticker"] for r in de["buy_rows"]]
        runup = signals.fetch_runup(de_tk, asof)
        kept = []
        for r in de["buy_rows"]:
            t = r["ticker"]
            ru = runup.get(t)
            if ru is not None and ru >= 0.20 and t not in wbirth:
                chased_excluded.append({"ticker": t, "runup_pct": round(ru * 100, 1),
                                        "section": r.get("section")})
            else:
                kept.append(r)
        de["buy_rows"] = kept
        if chased_excluded:
            _lst = ", ".join("{}(+{:.0f}%)".format(c["ticker"], c["runup_pct"])
                             for c in chased_excluded[:8])
            print(f"   kovalama-filtresi: {len(chased_excluded)} aday ELENDİ "
                  f"(+%20 koşmuş, weekly birth yok): {_lst}")
    except Exception as e:
        print(f"   kovalama-filtresi HATA: {e}")

    # 5) korkuluk ön-hesap + pack — bağlam örtüşmesi kabul SIRASINI güçlendirir;
    #    HW dönüş betimsel pozisyon-rengi (SAT_OB yumuşak 'tepe' flag'i)
    hw = validated["hw_obos"]
    pre = guardrails.pre_check(pf.data, prices, de["buy_rows"],
                               context_hits=context.get("per_ticker", {}),
                               hw_per_ticker=hw.get("per_ticker", {}))
    pack = context_pack.build_context_pack(asof, pf, prices, validated, context, macro, pre)
    pack_path = context_pack.persist_pack(pack)
    print(f"   pack: {pack_path}")

    # 6) sentez (LLM → fallback) + post-validasyon
    advisory = synthesis.build_advisory(pack, use_llm=use_llm and not dry_run,
                                        llm_mode=llm_mode)
    if score_entry:
        advisory["scorecard_prev"] = score_entry
    advisory["chased_excluded"] = chased_excluded  # kovalama-filtresiyle elenenler (rapor notu)

    # 4c→6) takas/AKD — pozisyonlar + AL ADAYLARI (kullanıcı isteği) için Matriks akışı.
    #     Sentez sonrası: admit edilen adaylar belli; pozisyon ∪ aday (az isim) → makul
    #     Matriks çağrısı. Aday takas işaretleri render'da AL tablosuna da yansır.
    cand_tk = [b["ticker"] for b in advisory.get("buy_candidates", [])]
    takas_tk = list(dict.fromkeys(pf.tickers() + cand_tk))  # sıra korunur, tekrarsız
    takas = takas_mod.load_takas(takas_tk)
    takas["candidate_tickers"] = cand_tk  # render: pozisyon mu aday mı ayrımı
    print(f"   takas: {takas['status']} ({len(takas_tk)} isim: {len(pf.tickers())} poz + {len(cand_tk)} aday)" +
          (f" · {takas.get('n_alerts', 0)} alarm" if takas['status'] == 'OK'
           else f" — {takas.get('note') or takas.get('error', '')}"))
    advisory["takas"] = takas  # pozisyon+aday takas özeti + Matriks durum (render banner)
    advisory["inputs_status"]["takas"] = takas["status"]

    # 6b) HAFTALIK-LİDER lead-lag (mb_scanner_events parquet'ten, DE CSV'de görünmez):
    #     önceki tamamlanmış haftalık barda mb_1w above (LEAD) → içinde bulunulan
    #     (sonraki) hafta günlük/5h above (LAG). Aynı hafta çakışması DEĞİL.
    #     Adayları zenginleştir (1w bileşenini mtf_birth'e ekle) + aday-DIŞI
    #     çakışmaları ayrı 'haftalık-lider izleme' listesine koy (al kararı olmasa da raporla).
    try:
        xtf = signals.mb_birth_xtf(asof)
        per = xtf.get("per_ticker", {})
        cand_tk = set()
        for b in advisory.get("buy_candidates", []):
            cand_tk.add(b["ticker"])
            info = per.get(b["ticker"])
            if info and info.get("weekly_lead"):
                b["weekly_lead"] = True
                b["weekly_lead_tf"] = info["tf"]
                mtf = list(b.get("mtf_birth") or [])
                if "1w" not in mtf:
                    mtf.append("1w")
                b["mtf_birth"] = sorted(
                    mtf, key=lambda t: {"5h": 0, "1d": 1, "1w": 2, "1mo": 3}.get(t, 9))
        advisory["weekly_lead_watch"] = sorted(
            ({"ticker": t, "weekly_bar": v["weekly_bar"], "tf": v["tf"]}
             for t, v in per.items() if v.get("weekly_lead") and t not in cand_tk),
            key=lambda x: x["ticker"])
        advisory["weekly_lead_bar"] = xtf.get("weekly_bar")
        print(f"   haftalık-lider çakışma: {sum(1 for v in per.values() if v['weekly_lead'])} "
              f"toplam ({len(advisory['weekly_lead_watch'])} aday-dışı, bar={xtf.get('weekly_bar')})")
    except Exception as e:
        print(f"   haftalık-lider çakışma HATA: {e}")
        advisory["weekly_lead_watch"] = []

    # 6c) ÇAPRAZ SİNYAL ÇAKIŞMA → context_lists (→ _full_confluence raporda gösterir):
    #     (a) cluster3 (açık arketip-paper havuzu örtüşmesi)
    #     (b) triangle_break + horizontal_break (DE family'sinde YOK, event parquet'ten)
    try:
        all_tk = [b["ticker"] for b in advisory.get("buy_candidates", [])]
        c3 = validated.get("cluster3", {})
        c3_open = {str(c["ticker"]).upper() for c in (c3.get("open_candidates") or [])}
        xsig = signals.cross_signal_membership(asof, all_tk)  # {tkr: [triangle_break, horizontal_break]}
        n_c3 = n_tr = n_hb = 0
        for b in advisory.get("buy_candidates", []):
            cl = list(b.get("context_lists") or [])
            tags = []
            if b["ticker"] in c3_open:
                tags.append("cluster3"); n_c3 += 1
            for tg in xsig.get(b["ticker"], []):
                tags.append(tg)
                n_tr += (tg == "triangle_break"); n_hb += (tg == "horizontal_break")
            for tg in tags:
                if tg not in cl:
                    cl.append(tg)
            b["context_lists"] = cl
        # cluster3 STANDALONE liste (örtüşmese de raporla — aday-dışı c3 açık havuzu)
        cand_set = set(all_tk)
        advisory["cluster3_open"] = sorted(
            ({"ticker": str(c["ticker"]).upper(),
              "signal_date": c.get("signal_date"),
              "in_de": str(c["ticker"]).upper() in cand_set}
             for c in (c3.get("open_candidates") or [])),
            key=lambda x: (not x["in_de"], x["ticker"]))[:40]
        print(f"   çapraz çakışma: cluster3={n_c3} triangle={n_tr} horizontal={n_hb} "
              f"(açık c3 havuzu {len(c3_open)})")
    except Exception as e:
        print(f"   çapraz çakışma HATA: {e}")
        advisory["cluster3_open"] = []

    # 6c2) SEKTÖR ROTASYON operasyonelleştirme: öne çıkan sektörler → o sektördeki
    #      DE adayların. Sektör/zaman seçimi DOĞRULANMIŞ edge; sektör-İÇİ hisse seçimi
    #      robust DEĞİL → eşit-ağırlık/bilgi (PR #81/#86 anatomi + #109 faktör araştırması).
    try:
        smap = signals.load_sector_map()
        strength = signals.load_sector_strength(asof)  # BİRİNCİL Fintables, FALLBACK İŞY
        sectors = strength.get("sectors") or []
        fav = set(strength.get("favorable_sectors") or [])
        sec_lookup = {s["kod"]: s for s in sectors}
        picks = {}  # sector_index → [ticker...]
        for b in advisory.get("buy_candidates", []):
            si, sname = smap.get(b["ticker"], (None, None))
            b["sector_index"] = si
            b["sector_name"] = sname
            if si and si in fav:
                b["sector_favorable"] = True
                picks.setdefault(si, []).append(b["ticker"])
        # DOWN-CAPTURE faktörü (doğrulanmış defansif +alfa): adaylar + LİDER/ARMED
        # sektörlerin TÜM üyeleri (kullanıcı isteği). Tek XU100 çekimi paylaşılır.
        all_cands = advisory.get("buy_candidates", []) + (advisory.get("skipped_candidates") or [])
        cand_tk = [b["ticker"] for b in all_cands]
        members_by_sec = signals.sector_members(fav)         # lider sektör → üyeler
        leader_members = sorted({t for ms in members_by_sec.values() for t in ms})
        dcap = signals.compute_down_capture(sorted(set(cand_tk) | set(leader_members)), asof)
        for b in all_cands:
            info = dcap.get(b["ticker"])
            b["down_capture"] = info["dc"] if info else None
        # LİDER SEKTÖRLERDE DEFANSİF HİSSELER: üyeleri likidite eler + dc'ye göre sırala
        cand_set = set(cand_tk)
        sec_durum = {s["kod"]: s.get("durum") for s in sectors}  # her sektör TEK durum
        ADV_MIN = 5_000_000  # likidite tabanı (TL/gün, ~60g ort.)
        sector_dc_picks = {}
        for si in sorted(fav, key=lambda k: {"TETİK": 0, "ARMED": 1, "LİDER": 2}.get(sec_durum.get(k), 9)):
            rows = []
            for t in members_by_sec.get(si, []):
                d = dcap.get(t)
                if d and d.get("adv", 0) >= ADV_MIN:
                    rows.append({"ticker": t, "dc": d["dc"], "adv": round(d["adv"]),
                                 "in_de": t in cand_set})
            rows.sort(key=lambda x: x["dc"])            # düşük dc = defansif önce
            if rows:
                sector_dc_picks[si] = {"durum": sec_durum.get(si), "rows": rows[:8]}
        advisory["sector_dc_picks"] = sector_dc_picks
        advisory.setdefault("sector_rotation", {})
        advisory["sector_rotation"]["sectors"] = sectors
        advisory["sector_rotation"]["favorable_sectors"] = sorted(fav)
        advisory["sector_rotation"]["strength_source"] = strength.get("source")
        advisory["rotation_picks"] = {
            si: {"tickers": sorted(tk, key=lambda t: (dcap.get(t, {}).get("dc") if dcap.get(t) else 9e9)),
                 "durum": sec_lookup.get(si, {}).get("durum"),
                 "dipten_pct": sec_lookup.get(si, {}).get("dipten_pct"),
                 "down_capture": {t: (dcap.get(t, {}).get("dc")) for t in tk}}
            for si, tk in sorted(picks.items())}
        print(f"   sektör rotasyon [{strength.get('source')}]: öne çıkan {sorted(fav)} → "
              f"eşleşen aday {sum(len(v) for v in picks.values())} ({len(picks)} sektörde, "
              f"{len(sectors)} sektör taranmış) · down-capture {len(dcap)} hisse")
    except Exception as e:
        print(f"   sektör rotasyon eşleme HATA: {e}")
        advisory["rotation_picks"] = {}

    # 6d) XU100 RDP rejimi (long/flat/short) + geniş makro snapshot + tavan lock — BİLGİ
    advisory["xu100_rdp"] = signals.load_xu100_rdp(asof)
    advisory["macro_snapshot"] = (macro or {}).get("snapshot", [])
    advisory["tavan_lock"] = signals.load_tavan_scan_live(asof)  # tavan-scan-live (kullanıcı: lock yerine bu)
    _tsl = advisory["tavan_lock"]
    print(f"   tavan-scan-live: {_tsl.get('status')} ({len(_tsl.get('picks') or [])} aday, "
          f"scan={_tsl.get('scan_asof')})")
    rdp = advisory["xu100_rdp"]
    print(f"   XU100 RDP: {rdp.get('regime')} ({rdp.get('date')}, {rdp.get('status')})"
          f"{' ⚠️SAT/risk-off' if rdp.get('is_sell') else ''}")

    # 6e) TARAMA TAZELİĞİ paneli (kullanıcı endişesi: dahil edilen taramalar güncel
    #     olmayabiliyor). Her kaynağın tarihi + BAYAT işareti (asof'tan >3 gün eski).
    def _stale(d, weekly=False):  # işgünü-bazlı: daily 1 gün=BAYAT; haftalık 1 hafta=BAYAT
        return signals._busday_stale(d, asof, 5 if weekly else 0)
    fr = []
    de_block = validated.get("decision_engine", {})
    fr.append({"kaynak": "DE v1 watchlist", "tarih": de_block.get("asof"),
               "stale": de_block.get("status") == "STALE" or _stale(de_block.get("asof"))})
    for scr, d in sorted((context.get("scan_dates") or {}).items()):
        fr.append({"kaynak": f"ctx:{scr}", "tarih": d,
                   "stale": scr in (context.get("stale_scanners") or [])})
    for lbl, d, wk in [("tavan-scan-live", (advisory.get("tavan_lock") or {}).get("scan_asof"), False),
                       ("hw-obos", (validated.get("hw_obos") or {}).get("scan_date"), False),
                       ("sektör monitör", (advisory.get("sector_rotation") or {}).get("bar_date"), False),
                       ("XU100 RDP", rdp.get("date"), False),
                       ("haftalık-lider bar", advisory.get("weekly_lead_bar"), True),  # son Cuma normal
                       ("cluster3", (validated.get("cluster3") or {}).get("last_signal_date"), True)]:
        if d:
            fr.append({"kaynak": lbl, "tarih": str(d)[:10], "stale": _stale(d, wk)})
    advisory["freshness"] = fr
    advisory["context_scan_dates"] = context.get("scan_dates") or {}
    n_stale = sum(1 for x in fr if x["stale"])
    print(f"   tarama tazeliği: {len(fr)} kaynak, {n_stale} BAYAT "
          f"({', '.join(x['kaynak'] for x in fr if x['stale']) or 'hepsi güncel'})")

    adv_path = context_pack.persist_advisory(advisory)
    print(f"   advisory: {adv_path} (mode={advisory['mode']})")

    # 7) render + teslimat
    html_path = render.render_html(advisory)
    msg = render.render_telegram_tr(advisory)
    if dry_run:
        print("─" * 60)
        print(msg)
        print("─" * 60)
        print(f"(dry-run — Telegram gönderilmedi; HTML: {html_path})")
        return advisory

    if notify:
        from core.reports import send_telegram, send_telegram_document
        send_telegram(msg)
        send_telegram_document(str(html_path))
    if publish_latest:
        publish_advisory_latest(advisory)   # bot /danis bunu okur
        # HTML'i public GH Pages'e de yayınla (kullanıcı onayı 2026-07-14;
        # rapor portföyü içerir — GH_PAGES_REPO tanımlıysa çalışır).
        from agent.advisor.portfolio import publish_html_to_pages
        publish_html_to_pages(html_path, advisory.get("asof", "latest"))
        scorecard.update_scorecard(score_entry)
    return advisory
