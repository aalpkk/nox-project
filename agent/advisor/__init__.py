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


def _log_context_membership(asof, context):
    """Günün bağlam-tarayıcı üyeliğini parquet'e yaz (idempotent, asof başına 1)."""
    import pandas as pd
    from pathlib import Path
    per = context.get("per_ticker") or {}
    if not per:
        return
    out = Path("output/advisor/context_log")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{asof}.parquet"
    if path.exists():
        return
    rows = [{"asof": asof, "ticker": t, "scanner": s}
            for t, scanners in per.items() for s in scanners]
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"   context-log: {len(rows)} üyelik → {path.name}")


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

    # 4b) forward-log: bağlam tarayıcı üyeliğini günlük kaydet → ileride
    #     (alsat dahil, replay-edilemeyenler için) kendi MFE/MAE backtest'i.
    #     Geçmiş iddiası yok, paper-track veri biriktirme.
    if not dry_run:
        try:
            _log_context_membership(asof, context)
        except Exception as e:
            print(f"   ⚠️ context forward-log atlandı: {e}")

    # 5) korkuluk ön-hesap + pack — bağlam örtüşmesi kabul SIRASINI güçlendirir;
    #    HW dönüş betimsel pozisyon-rengi (SAT_OB yumuşak 'tepe' flag'i);
    #    backtest-doğrulanmış seçim ağırlıkları (PROCEED ise) sıralamayı yönetir
    hw = validated["hw_obos"]
    sel_w = signals.load_selection_weights()
    cand_tickers = [r["ticker"] for r in de["buy_rows"]]
    panel_feats = signals.load_panel_features(asof, cand_tickers) if sel_w else {}
    if sel_w:
        print(f"   seçim ağırlıkları (backtest PROCEED): {sel_w} · "
              f"panel-feature {len(panel_feats)}/{len(cand_tickers)} ticker")
    pre = guardrails.pre_check(pf.data, prices, de["buy_rows"],
                               context_hits=context.get("per_ticker", {}),
                               hw_per_ticker=hw.get("per_ticker", {}),
                               panel_features=panel_feats, selection_weights=sel_w)
    pack = context_pack.build_context_pack(asof, pf, prices, validated, context, macro, pre)
    pack_path = context_pack.persist_pack(pack)
    print(f"   pack: {pack_path}")

    # 6) sentez (LLM → fallback) + post-validasyon
    advisory = synthesis.build_advisory(pack, use_llm=use_llm and not dry_run,
                                        llm_mode=llm_mode)
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
