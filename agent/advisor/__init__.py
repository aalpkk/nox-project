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
                portfolio_path=None, publish_latest=True):
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

    # 5) korkuluk ön-hesap + pack
    pre = guardrails.pre_check(pf.data, prices, de["buy_rows"])
    pack = context_pack.build_context_pack(asof, pf, prices, validated, context, macro, pre)
    pack_path = context_pack.persist_pack(pack)
    print(f"   pack: {pack_path}")

    # 6) sentez (LLM → fallback) + post-validasyon
    advisory = synthesis.build_advisory(pack, use_llm=use_llm and not dry_run)
    if score_entry:
        advisory["scorecard_prev"] = score_entry
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
