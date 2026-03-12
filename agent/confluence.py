"""
NOX Agent — Çakışma (Confluence) Skoru
Çoklu tarama sonuçlarına dayalı puan sistemi.
Forward test WR verilerine göre kalibrasyon.

Puan tablosu (nox_agent_prompt.md Section 4):
  +3: NW PIVOT_AL + WL=HAZIR (WR %77.8)
  +2: NW PIVOT_AL + WL=İZLE veya ZONE_ONLY+İZLE
  +2: RT FULL_TREND/TREND + TAZE window
  +2: AL/SAT DÖNÜŞ + Q≥85
  +2: H+AL veya H+PB badge
  +1: HC2 tetik bonusu
  +1: AL/SAT DÖNÜŞ/CMB Q≥60 (veya karar=AL)
  +1: Tavan skor ≥50
  +2: Kademe S/A<0.50 (çok güçlü alıcı)
  +1: Kademe S/A<0.80 (güçlü alıcı konfirmasyonu)
  +1: Divergence AL + NW AL overlap
  ℹ️: Kademe S/A>1.50 (bilgi notu, puan kesme yok)
  -1: CMF negatif
  -2: OE ≥ 4
  -1: OE = 3
  -2: NW PIVOT_SAT
  >= 5 puan: Güçlü AL adayı
  3-4: AL
  1-2: İzle
  0: Nötr
  < 0: Kaçın

NOT: nox_v3_daily (WR %41.6) confluence'a katılmaz — gürültü üretir.
     Günlük NW sadece bilgi amaçlı scanner_reader'da tutulur.
"""


def _deduplicate_signals(ticker_signals):
    """Her screener'dan sadece en son tarihli sinyali al."""
    by_screener = {}
    for s in ticker_signals:
        scr = s['screener']
        by_screener.setdefault(scr, []).append(s)

    result = []
    for scr, sigs in by_screener.items():
        latest_date = max(s.get('csv_date', '') for s in sigs)
        latest = [s for s in sigs if s.get('csv_date', '') == latest_date]
        result.extend(latest)
    return result


def calc_confluence_score(ticker, signals, macro_regime=None):
    """
    Belirli bir ticker için çakışma skoru hesapla.

    Returns:
        dict: {ticker, score, details[], recommendation, source_count, signals}
    """
    ticker = ticker.upper().strip()
    candidates = [ticker, f"{ticker}.IS", ticker.replace('.IS', '')]

    all_ticker_signals = [s for s in signals if s['ticker'] in candidates]
    if not all_ticker_signals:
        return {
            "ticker": ticker,
            "score": 0,
            "details": ["Hiç sinyal bulunamadı"],
            "recommendation": "VERİ_YOK",
            "source_count": 0,
            "signals": [],
        }

    ticker_signals = _deduplicate_signals(all_ticker_signals)

    score = 0
    details = []
    sources = set()

    # ══════════════════════════════════════════════════════════════
    # 1. NOX v3 Weekly (WR %66.9+) — ANA yapısal sinyal
    # ══════════════════════════════════════════════════════════════
    nw_signals = [s for s in ticker_signals if s['screener'] == 'nox_v3_weekly']
    nw_al = [s for s in nw_signals if s['direction'] == 'AL']
    nw_sat = [s for s in nw_signals if s['direction'] == 'SAT']

    for s in nw_al:
        sources.add('nox_v3_weekly')
        sig_type = s.get('signal_type', '')
        wl = s.get('wl_status', '')
        trigger = s.get('trigger_type', '')
        fresh = s.get('fresh', '')

        if sig_type == 'PIVOT_AL' and wl == 'HAZIR':
            score += 3
            details.append("+3 NW PIVOT_AL + WL=HAZIR (WR %77.8)")
        elif sig_type == 'PIVOT_AL' and wl in ('İZLE', 'IZLE'):
            score += 2
            details.append(f"+2 NW PIVOT_AL WL=İZLE [{trigger or '?'}]")
        elif sig_type == 'PIVOT_AL':
            score += 2
            wl_info = f" WL={wl}" if wl else ""
            details.append(f"+2 NW PIVOT_AL{wl_info} [{trigger or '?'}]")
        elif sig_type == 'ZONE_ONLY' and wl in ('İZLE', 'IZLE'):
            score += 2
            details.append("+2 NW ZONE_ONLY + WL=İZLE (WR %68.9)")
        elif sig_type == 'ZONE_ONLY':
            score += 1
            details.append("+1 NW ZONE_ONLY")
        elif sig_type == 'ADAY':
            score += 1
            details.append("+1 NW ADAY (onaysız pivot)")

        # Tetik tipi bonusu
        if trigger == 'HC2':
            score += 1
            details.append("+1 HC2 tetik (WR %68.3)")

        # Fresh bilgi notu
        if fresh == 'BUGUN' and s.get('gate') == 'OK':
            details.append("⚡ BUGÜN fresh + Gate OK")

    for s in nw_sat:
        sources.add('nox_v3_weekly')
        score -= 2
        details.append("-2 NW PIVOT_SAT")

    # ══════════════════════════════════════════════════════════════
    # nox_v3_daily KASITLI OLARAK DAHIL EDİLMİYOR
    # WR %41.6 — gürültü üretir, confluence'a katkısı yok.
    # scanner_reader'da ayrı screener olarak tutulur (bilgi amaçlı).
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    # 2. AL/SAT Screener — sinyal tipi + Q skoru ağırlıklı
    # ══════════════════════════════════════════════════════════════
    alsat_signals = [s for s in ticker_signals if s['screener'] == 'alsat']
    for s in alsat_signals:
        sources.add('alsat')
        sig_type = s.get('signal_type', '')
        q = s.get('quality') or 0
        karar = s.get('karar', '')
        oe = s.get('oe')

        # DÖNÜŞ Q≥85 = en güçlü AL/SAT sinyali
        if sig_type in ('DÖNÜŞ', 'DONUS', 'DONUŞ') and q >= 85:
            score += 2
            details.append(f"+2 AL/SAT DÖNÜŞ Q={q} (en yüksek WR)")
        elif sig_type in ('DÖNÜŞ', 'DONUS', 'DONUŞ') and q >= 60:
            score += 1
            details.append(f"+1 AL/SAT DÖNÜŞ Q={q}")
        elif sig_type == 'CMB' and q >= 80:
            score += 1
            details.append(f"+1 AL/SAT CMB Q={q}")
        elif sig_type in ('GUCLU', 'GÜÇLÜ') and karar == 'AL':
            score += 1
            details.append(f"+1 AL/SAT GÜÇLÜ karar=AL")
        # ERKEN, ZAYIF, PB, BİLEŞEN → 0 puan (zayıf sinyaller)
        elif karar == 'AL' and sig_type not in ('ERKEN', 'ZAYIF'):
            score += 1
            details.append(f"+1 AL/SAT {sig_type} karar=AL")
        # ERKEN/ZAYIF → sources'a eklenir ama puan yok
        elif karar == 'AL':
            sources.add('alsat')
            details.append(f"0 AL/SAT {sig_type} (zayıf, puan yok)")

        # OE penaltı
        if oe is not None and oe >= 3:
            penalty = -2 if oe >= 4 else -1
            score += penalty
            details.append(f"{penalty:+d} AL/SAT OE={oe}")

    # ══════════════════════════════════════════════════════════════
    # 3. Rejim v3 (bist-tavan-screener)
    # ══════════════════════════════════════════════════════════════
    rt_signals = [s for s in ticker_signals if s['screener'] == 'rejim_v3']
    for s in rt_signals:
        sources.add('rejim_v3')
        sig_type = s.get('signal_type', '')
        if sig_type in ('FULL_TREND', 'TREND'):
            score += 2
            details.append(f"+2 RT {sig_type}")
        elif sig_type in ('GUCLU', 'GÜÇLÜ', 'CMB'):
            score += 1
            details.append(f"+1 RT {sig_type}")
        # ERKEN, ZAYIF, PB, MR, DONUS, BILESEN → 0 puan
        # (fazla gürültü üretiyordu)

        oe = s.get('oe')
        if oe is not None and oe >= 3:
            penalty = -2 if oe >= 4 else -1
            score += penalty
            details.append(f"{penalty:+d} RT OE={oe}")

    # ══════════════════════════════════════════════════════════════
    # 4. Regime Transition — badge'ler burada (H+AL, H+PB)
    # ══════════════════════════════════════════════════════════════
    rt2_signals = [s for s in ticker_signals if s['screener'] == 'regime_transition']
    rt2_al = [s for s in rt2_signals if s['direction'] == 'AL']
    for s in rt2_al:
        sources.add('regime_transition')
        q = s.get('quality')
        window = s.get('entry_window', '')
        badge = s.get('badge', '')
        cmf = s.get('cmf')

        # Badge: H+AL / H+PB — tarihsel en yüksek WR setup
        if badge in ('H+AL', 'H+PB'):
            score += 2
            details.append(f"+2 {badge} badge (tarihsel en yüksek WR)")

        # TAZE window + yüksek entry score (badge yoksa)
        if window == 'TAZE' and q is not None and q >= 3 and not badge:
            score += 2
            details.append(f"+2 RT Transition TAZE skor={q}")
        elif q is not None and q >= 3 and not badge:
            score += 1
            details.append(f"+1 RT Transition skor={q} [{window}]")

        # CMF penaltı
        if cmf is not None and cmf < 0:
            score -= 1
            details.append(f"-1 CMF negatif ({cmf:.3f})")

        # OE penaltı
        oe = s.get('oe')
        if oe is not None and oe >= 3:
            penalty = -2 if oe >= 4 else -1
            score += penalty
            details.append(f"{penalty:+d} RT OE={oe}")

    # ══════════════════════════════════════════════════════════════
    # 5. Tavan Scanner
    # ══════════════════════════════════════════════════════════════
    tavan_signals = [s for s in ticker_signals if s['screener'] == 'tavan']
    for s in tavan_signals:
        if s['direction'] != 'AL':
            continue
        sources.add('tavan')
        skor = s.get('skor', 0)
        streak = s.get('streak', 0)
        vol_ratio = s.get('volume_ratio', 1.0)

        if skor >= 60 and vol_ratio < 1.0:
            score += 2
            details.append(f"+2 Tavan skor={skor} düşük hacim (WR %84.6)")
        elif skor >= 50:
            score += 1
            vol_tag = " düşük hacim" if vol_ratio < 1.0 else ""
            details.append(f"+1 Tavan skor={skor} streak={streak}{vol_tag}")

    # ══════════════════════════════════════════════════════════════
    # 6. Kademe S/A — KONFİRMASYON katmanı (eleme aracı DEĞİL)
    # ══════════════════════════════════════════════════════════════
    kademe_signals = [s for s in ticker_signals if s['screener'] == 'kademe']
    for s in kademe_signals:
        sa = s.get('sa_ratio', 1.0)
        if sa < 0.50 and not s.get('kilitli'):
            sources.add('kademe')
            score += 2
            details.append(f"+2 Kademe S/A={sa} (çok güçlü alıcı)")
        elif sa < 0.80 and not s.get('kilitli'):
            sources.add('kademe')
            score += 1
            details.append(f"+1 Kademe S/A={sa} (güçlü alıcı)")
        elif s.get('kilitli'):
            sources.add('kademe')
            score += 1
            details.append("+1 Kademe kilitli tavan")
        elif sa > 1.50:
            # Konfirmasyon: bilgi notu, puan kesme yok
            sources.add('kademe')
            details.append(f"ℹ️ Kademe S/A={sa} (satıcı baskısı — bilgi notu)")

    # ══════════════════════════════════════════════════════════════
    # 7. Divergence — PASİF (geliştirme bekliyor)
    # ══════════════════════════════════════════════════════════════
    # div_signals = [s for s in ticker_signals if s['screener'] == 'divergence']
    # div_al = [s for s in div_signals if s['direction'] == 'AL']
    # div_sat = [s for s in div_signals if s['direction'] == 'SAT']
    # if div_al and nw_al:
    #     sources.add('divergence')
    #     score += 1
    #     details.append("+1 Divergence AL + NW AL uyumu")
    # if div_sat and nw_sat:
    #     sources.add('divergence')
    #     score += 1
    #     details.append("+1 Divergence SAT + NW SAT uyumu")

    # ══════════════════════════════════════════════════════════════
    # 8. Makro rejim etkisi
    # ══════════════════════════════════════════════════════════════
    if macro_regime:
        regime = macro_regime.get("regime", "NÖTR")
        if regime in ("RISK_ON", "GÜÇLÜ_RISK_ON") and nw_al:
            score += 1
            details.append(f"+1 Makro {regime}")
        elif regime in ("RISK_OFF", "GÜÇLÜ_RISK_OFF") and nw_al:
            score -= 1
            details.append(f"-1 Makro {regime} (dikkat)")

    # ══════════════════════════════════════════════════════════════
    # Çakışma özet + tavsiye
    # ══════════════════════════════════════════════════════════════
    source_count = len(sources)
    if source_count >= 3:
        details.insert(0, f"⬡ {source_count} kaynak çakışması")

    if score >= 5:
        recommendation = "GÜÇLÜ_AL"
    elif score >= 3:
        recommendation = "AL"
    elif score >= 1:
        recommendation = "İZLE"
    elif score == 0:
        recommendation = "NÖTR"
    else:
        recommendation = "KAÇIN"

    return {
        "ticker": ticker,
        "score": score,
        "details": details,
        "recommendation": recommendation,
        "source_count": source_count,
        "signals": ticker_signals,
    }


def calc_all_confluence(signals, macro_regime=None, min_score=1):
    """Tüm ticker'lar için çakışma skoru hesapla ve sırala."""
    tickers = set(s['ticker'] for s in signals)
    results = []
    for ticker in tickers:
        result = calc_confluence_score(ticker, signals, macro_regime)
        if result['score'] >= min_score:
            results.append(result)

    results.sort(key=lambda x: (x['score'], x['source_count']), reverse=True)
    return results


def format_confluence_summary(results, top_n=15):
    """Çakışma sonuçlarını Telegram formatında döndür."""
    if not results:
        return "Çakışma sinyali bulunamadı."

    lines = [f"<b>⬡ Çakışma Analizi — {len(results)} hisse</b>", ""]

    rec_emoji = {
        "GÜÇLÜ_AL": "🟢🟢",
        "AL": "🟢",
        "İZLE": "🟡",
        "NÖTR": "⚪",
        "KAÇIN": "🔴",
    }

    for r in results[:top_n]:
        emoji = rec_emoji.get(r['recommendation'], '')
        src = r.get('source_count', 0)
        lines.append(
            f"{emoji} <b>{r['ticker']}</b> — Skor: {r['score']} ({r['recommendation']}) [{src} kaynak]")
        for d in r['details'][:5]:
            lines.append(f"  {d}")
        lines.append("")

    if len(results) > top_n:
        lines.append(f"... ve {len(results) - top_n} hisse daha")

    return "\n".join(lines)
