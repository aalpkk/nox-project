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

    # 4c) takas/AKD — elde tutulan pozisyonlar için Matriks akışı (graceful;
    #     API hata verirse status=MATRIKS_ERROR, raporda kırmızı uyarı → kullanıcı güncellesin)
    from agent.advisor import takas as takas_mod
    takas = takas_mod.load_takas(pf.tickers())
    print(f"   takas: {takas['status']}" +
          (f" ({takas.get('n_alerts', 0)} alarm)" if takas['status'] == 'OK'
           else f" — {takas.get('note') or takas.get('error', '')}"))

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
    advisory["takas"] = takas  # pozisyon takas özeti + Matriks durum (render banner)
    advisory["inputs_status"]["takas"] = takas["status"]

    # 6b) HAFTALIK-LİDER çakışma (mb_scanner_events parquet'ten, DE CSV'de görünmez):
    #     son kapanmış haftalık barda mb_1w above + aynı hafta günlük/5h above.
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

    # 6c) cluster3 ÇAKIŞMA: DE adayı aynı zamanda açık cluster3 arketip-paper adayı mı?
    #     context_lists'e "cluster3" ekle → _full_confluence raporda gösterir.
    try:
        c3 = validated.get("cluster3", {})
        c3_open = {str(c["ticker"]).upper() for c in (c3.get("open_candidates") or [])}
        n_hit = 0
        for b in advisory.get("buy_candidates", []):
            if b["ticker"] in c3_open:
                cl = list(b.get("context_lists") or [])
                if "cluster3" not in cl:
                    cl.append("cluster3")
                b["context_lists"] = cl
                n_hit += 1
        print(f"   cluster3 çakışma: {n_hit} aday (açık c3 havuzu {len(c3_open)})")
    except Exception as e:
        print(f"   cluster3 çakışma HATA: {e}")

    # 6d) XU100 RDP rejimi (long/flat/short) + geniş makro snapshot — BİLGİ
    advisory["xu100_rdp"] = signals.load_xu100_rdp(asof)
    advisory["macro_snapshot"] = (macro or {}).get("snapshot", [])
    rdp = advisory["xu100_rdp"]
    print(f"   XU100 RDP: {rdp.get('regime')} ({rdp.get('date')}, {rdp.get('status')})"
          f"{' ⚠️SAT/risk-off' if rdp.get('is_sell') else ''}")

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
        scorecard.update_scorecard(score_entry)
    return advisory
