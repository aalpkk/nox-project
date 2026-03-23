"""
NOX Agent — Çakışma (Confluence) Skoru v2
3 katmanlı skorlama: Yön Uyumu → Kalite Ağırlıklı Puan → Rejim Çarpanı.

Katman A: Yön uyumu — AL+SAT çelişki tespiti (puanlama öncesi)
Katman B: Kalite ağırlıklı puanlama — her kaynak horizon'a ayrılır (yapısal/taktik)
Katman C: Rejim çarpanı — makro rejime göre final skor ölçeklenir

Etiketler:
  TRADEABLE: final≥5, structural≥3, yön uyumlu
  TAKTİK: final≥3, structural<2 (kısa horizon ağırlıklı)
  İZLE: final≥3 veya final≥1 (setup var, teyit bekle)
  BEKLE: çelişki var veya kalite düşük
  ELE: sinyal zayıf

SM/ICE: Skor etkisi YOK — sadece bilgi notu.
SAT sinyalleri: Skor etkisi YOK — sadece çelişki flag.
OE/CMF: Bilgi notu — skor etkisi YOK.
"""


# Yapısal kaynaklar: swing/3G-5G horizon
_STRUCTURAL = {'nox_v3_weekly', 'regime_transition', 'alsat'}
# Taktik kaynaklar: 1G burst horizon
_TACTICAL = {'tavan', 'kademe'}

# Rejim çarpanı — final_score = round(raw_score * multiplier)
_REGIME_MULT = {
    'GÜÇLÜ_RISK_ON': 1.2,
    'RISK_ON': 1.1,
    'NÖTR': 1.0,
    'TRANSITION': 0.9,
    'RISK_OFF': 0.7,
    'GÜÇLÜ_RISK_OFF': 0.5,
}


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


def calc_confluence_score(ticker, signals, macro_regime=None, mkk_data=None,
                          sms_scores=None, ice_results=None):
    """
    Belirli bir ticker için 3 katmanlı çakışma skoru hesapla.

    Args:
        ticker: Hisse kodu
        signals: Tüm sinyaller listesi
        macro_regime: Makro rejim dict (opsiyonel)
        mkk_data: MKK verisi dict (opsiyonel)
        sms_scores: SMS skorları dict (opsiyonel, sadece bilgi notu)
        ice_results: ICE sonuçları dict (opsiyonel, sadece bilgi notu)

    Returns:
        dict: {ticker, score, raw_score, structural_score, tactical_score,
               has_conflict, conflict_sources, details, recommendation,
               source_count, signals}
    """
    ticker = ticker.upper().strip()
    candidates = [ticker, f"{ticker}.IS", ticker.replace('.IS', '')]

    all_ticker_signals = [s for s in signals if s['ticker'] in candidates]
    if not all_ticker_signals:
        return {
            "ticker": ticker,
            "score": 0,
            "raw_score": 0,
            "structural_score": 0,
            "tactical_score": 0,
            "has_conflict": False,
            "conflict_sources": [],
            "details": ["Hiç sinyal bulunamadı"],
            "recommendation": "VERİ_YOK",
            "source_count": 0,
            "signals": [],
        }

    ticker_signals = _deduplicate_signals(all_ticker_signals)

    # ══════════════════════════════════════════════════════════════
    # KATMAN A: Yön Uyumu — çelişki tespiti (puanlama öncesi)
    # ══════════════════════════════════════════════════════════════
    al_sources = set()
    sat_sources = set()
    for s in ticker_signals:
        direction = s.get('direction', '')
        karar = s.get('karar', '')
        scr = s['screener']
        if direction == 'AL' or karar == 'AL':
            al_sources.add(scr)
        if direction == 'SAT' or karar == 'SAT':
            sat_sources.add(scr)

    has_conflict = bool(al_sources) and bool(sat_sources)
    conflict_note = ""
    if has_conflict:
        conflict_note = f"⚠️ Çelişki: AL={sorted(al_sources)} vs SAT={sorted(sat_sources)}"

    # ══════════════════════════════════════════════════════════════
    # KATMAN B: Kalite Ağırlıklı Puanlama
    # Yapısal (structural) ve taktik (tactical) ayrı toplanır
    # ══════════════════════════════════════════════════════════════
    structural_score = 0
    tactical_score = 0
    details = []
    sources = set()

    def _add(pts, screener, detail):
        nonlocal structural_score, tactical_score
        if screener in _STRUCTURAL:
            structural_score += pts
        elif screener in _TACTICAL:
            tactical_score += pts
        else:
            structural_score += pts  # default: yapısal
        if detail:
            details.append(detail)

    # ── 1. NOX v3 Weekly (yapısal) ──
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
            if trigger == 'HC2':
                _add(4, 'nox_v3_weekly', "+4 NW PIVOT_AL + HAZIR + HC2 (WR %77.8)")
            else:
                _add(3, 'nox_v3_weekly', f"+3 NW PIVOT_AL + HAZIR [{trigger or '?'}]")
        elif sig_type == 'PIVOT_AL' and wl in ('İZLE', 'IZLE'):
            if trigger == 'EMA_R':
                _add(2, 'nox_v3_weekly', f"+2 NW PIVOT_AL İZLE EMA_R (WR %57.8)")
            else:
                _add(2, 'nox_v3_weekly', f"+2 NW PIVOT_AL İZLE [{trigger or '?'}]")
        elif sig_type == 'PIVOT_AL':
            wl_info = f" WL={wl}" if wl else ""
            if wl in ('BEKLE',) or not wl:
                _add(0, 'nox_v3_weekly', f"0 NW PIVOT_AL{wl_info} (düşük kalite)")
            else:
                _add(2, 'nox_v3_weekly', f"+2 NW PIVOT_AL{wl_info} [{trigger or '?'}]")
        elif sig_type == 'ZONE_ONLY' and wl in ('İZLE', 'IZLE'):
            _add(1, 'nox_v3_weekly', "+1 NW ZONE_ONLY + İZLE (tetik yok)")
        elif sig_type == 'ZONE_ONLY':
            _add(0, 'nox_v3_weekly', "0 NW ZONE_ONLY (tetik yok, düşür)")
        elif sig_type == 'ADAY':
            _add(0, 'nox_v3_weekly', "0 NW ADAY (onaysız pivot)")
        # BEKLE ve diğer WL → 0 puan

        # Fresh bilgi notu
        if fresh == 'BUGUN' and s.get('gate') == 'OK':
            details.append("⚡ BUGÜN fresh + Gate OK")

    # SAT → skor etkisi yok, sadece çelişki flag'inde kullanılır
    for s in nw_sat:
        sources.add('nox_v3_weekly')
        details.append("ℹ️ NW PIVOT_SAT (skor etkisi yok, çelişki flag)")

    # ══════════════════════════════════════════════════════════════
    # nox_v3_daily KASITLI OLARAK DAHIL EDİLMİYOR
    # WR %41.6 — gürültü üretir, confluence'a katkısı yok.
    # ══════════════════════════════════════════════════════════════

    # ── 2. AL/SAT Screener (yapısal) ──
    alsat_signals = [s for s in ticker_signals if s['screener'] == 'alsat']
    for s in alsat_signals:
        sources.add('alsat')
        sig_type = s.get('signal_type', '')
        q = s.get('quality') or 0
        karar = s.get('karar', '')
        oe = s.get('oe')

        if sig_type == 'CMB' and q >= 80:
            _add(3, 'alsat', f"+3 AL/SAT CMB Q={q}")
        elif sig_type in ('GUCLU', 'GÜÇLÜ') and karar == 'AL':
            _add(2, 'alsat', f"+2 AL/SAT GÜÇLÜ karar=AL")
        elif sig_type in ('DÖNÜŞ', 'DONUS', 'DONUŞ') and q >= 85:
            _add(2, 'alsat', f"+2 AL/SAT DÖNÜŞ Q={q}")
        elif sig_type in ('DÖNÜŞ', 'DONUS', 'DONUŞ') and q >= 60:
            _add(1, 'alsat', f"+1 AL/SAT DÖNÜŞ Q={q}")
        elif sig_type in ('BILESEN', 'BİLEŞEN') and q >= 70:
            macd = s.get('macd', 0) or 0
            if macd > 0:
                _add(1, 'alsat', f"+1 AL/SAT BİLEŞEN Q={q} MACD>0")
            else:
                _add(0, 'alsat', f"0 AL/SAT BİLEŞEN Q={q} MACD≤0 (kalite gate)")
        elif karar == 'AL' and sig_type not in ('ERKEN', 'ZAYIF'):
            _add(0, 'alsat', f"0 AL/SAT {sig_type} (kalite gate)")
        elif karar == 'AL':
            details.append(f"0 AL/SAT {sig_type} (zayıf, puan yok)")

        # OE → bilgi notu (skor etkisi yok)
        if oe is not None and oe != '' and int(oe or 0) >= 3:
            details.append(f"ℹ️ AL/SAT OE={oe} (bilgi notu)")

    # ── 3. Rejim v3 (bist-tavan-screener) ──
    rt_signals = [s for s in ticker_signals if s['screener'] == 'rejim_v3']
    for s in rt_signals:
        sources.add('rejim_v3')
        sig_type = s.get('signal_type', '')
        if sig_type in ('FULL_TREND', 'TREND'):
            _add(2, 'regime_transition', f"+2 RT {sig_type}")
        elif sig_type in ('GUCLU', 'GÜÇLÜ', 'CMB'):
            _add(1, 'regime_transition', f"+1 RT {sig_type}")

        oe = s.get('oe')
        if oe is not None and oe != '' and int(oe or 0) >= 3:
            details.append(f"ℹ️ RT OE={oe} (bilgi notu)")

    # ── 4. Regime Transition — badge'ler (yapısal) ──
    rt2_signals = [s for s in ticker_signals if s['screener'] == 'regime_transition']
    rt2_al = [s for s in rt2_signals if s['direction'] == 'AL']
    for s in rt2_al:
        sources.add('regime_transition')
        q = s.get('quality')
        window = s.get('entry_window', '')
        badge = s.get('badge', '')
        cmf = s.get('cmf')

        # Badge: H+PB TAZE → +3, H+PB other → +2, H+AL → +2
        if badge == 'H+PB' and window == 'TAZE':
            _add(3, 'regime_transition', f"+3 H+PB TAZE (en yüksek WR)")
        elif badge in ('H+AL', 'H+PB'):
            _add(2, 'regime_transition', f"+2 {badge} badge")

        # TAZE entry≥3 no-badge → +1
        if window == 'TAZE' and q is not None and q >= 3 and not badge:
            _add(1, 'regime_transition', f"+1 RT Transition TAZE skor={q} (no-badge)")
        elif q is not None and q >= 3 and not badge:
            _add(0, 'regime_transition', f"0 RT Transition skor={q} [{window}] (no-badge)")

        # Hacim-donus tier bonusu/penaltisi (backtest: 3y, ALTIN 5G %75, GUMUS %71)
        vol_tier = s.get('vol_tier', '')
        if vol_tier == 'ALTIN':
            _add(2, 'regime_transition', "+2 RT Hacim ALTIN (ATR≤3% Part=3 OE≤1)")
        elif vol_tier == 'GUMUS':
            _add(1, 'regime_transition', "+1 RT Hacim GÜMÜŞ (ATR≤3% Part=3)")
        elif vol_tier == 'BRONZ':
            details.append("ℹ️ RT Hacim BRONZ (ATR≤4% Part≥3)")
        elif vol_tier == 'ELE':
            _add(-1, 'regime_transition', "-1 RT Hacim ELE (kötü hacim profili)")

        # CMF → bilgi notu (skor etkisi yok)
        if cmf is not None and cmf < 0:
            details.append(f"ℹ️ CMF negatif ({cmf:.3f}) (bilgi notu)")

        # OE → bilgi notu
        oe = s.get('oe')
        if oe is not None and oe != '' and int(oe or 0) >= 3:
            details.append(f"ℹ️ RT OE={oe} (bilgi notu)")

    # ── 5. Tavan Scanner (taktik) ──
    tavan_signals = [s for s in ticker_signals if s['screener'] == 'tavan']
    for s in tavan_signals:
        if s['direction'] != 'AL':
            continue
        sources.add('tavan')
        skor = s.get('skor', 0)
        vol_ratio = s.get('volume_ratio', 1.0)

        if skor >= 50 and vol_ratio < 3.0:
            _add(2, 'tavan', f"+2 Tavan skor={skor} vol={vol_ratio:.1f}x (kilitli)")
        elif skor >= 50:
            _add(1, 'tavan', f"+1 Tavan skor={skor} vol={vol_ratio:.1f}x")
        # skor<50 → 0 puan

    # ── 6. Kademe S/A (taktik) ──
    kademe_signals = [s for s in ticker_signals if s['screener'] == 'kademe']
    for s in kademe_signals:
        sa = s.get('sa_ratio', 1.0)
        if sa < 0.50 and not s.get('kilitli'):
            sources.add('kademe')
            _add(2, 'kademe', f"+2 Kademe S/A={sa} (çok güçlü alıcı)")
        elif sa < 0.80 and not s.get('kilitli'):
            sources.add('kademe')
            _add(1, 'kademe', f"+1 Kademe S/A={sa} (güçlü alıcı)")
        elif s.get('kilitli'):
            sources.add('kademe')
            _add(1, 'kademe', "+1 Kademe kilitli tavan")
        elif sa > 1.50:
            sources.add('kademe')
            details.append(f"ℹ️ Kademe S/A={sa} (satıcı baskısı — bilgi notu)")

    # ── 6b. MKK Yatırımcı Dağılımı ──
    if mkk_data:
        mkk_ticker = mkk_data.get(ticker)
        if mkk_ticker:
            bireysel_pct = mkk_ticker.get('bireysel_pct', 100) or 100
            bireysel_fark = mkk_ticker.get('bireysel_fark_1g', 0) or 0

            if bireysel_pct < 30:
                sources.add('mkk')
                _add(1, 'nox_v3_weekly', f"+1 MKK kurumsal ağırlıklı (bireysel %{bireysel_pct:.1f})")
            if bireysel_fark < -0.5:
                sources.add('mkk')
                _add(1, 'nox_v3_weekly', f"+1 MKK kurumsal birikim (bireysel {bireysel_fark:+.2f}% günlük)")

    # ── 6c. SM/ICE — skor etkisi YOK, sadece bilgi notu ──
    if ice_results:
        ice = ice_results.get(ticker)
        if ice:
            details.append(f"ℹ️ ICE ×{ice.multiplier:.2f}{ice.icon} (bilgi — skor etkisi yok)")
    elif sms_scores:
        sms = sms_scores.get(ticker)
        if sms:
            sms_val = sms.score if hasattr(sms, 'score') else sms
            details.append(f"ℹ️ SMS {sms_val} (bilgi — skor etkisi yok)")

    # ══════════════════════════════════════════════════════════════
    # KATMAN C: Rejim Çarpanı
    # ══════════════════════════════════════════════════════════════
    raw_score = structural_score + tactical_score
    regime_mult = 1.0
    if macro_regime:
        regime = macro_regime.get("regime", "NÖTR")
        regime_mult = _REGIME_MULT.get(regime, 1.0)
        if regime_mult != 1.0:
            details.append(f"×{regime_mult:.1f} Makro {regime}")

    final_score = round(raw_score * regime_mult)

    # ══════════════════════════════════════════════════════════════
    # Çakışma özet + tavsiye
    # ══════════════════════════════════════════════════════════════
    source_count = len(sources)
    if source_count >= 3:
        details.insert(0, f"⬡ {source_count} kaynak çakışması")
    if conflict_note:
        details.insert(0, conflict_note)

    # Yeni etiketler
    if has_conflict:
        recommendation = "BEKLE"
    elif final_score >= 5 and structural_score >= 3:
        recommendation = "TRADEABLE"
    elif final_score >= 3:
        if structural_score < 2:
            recommendation = "TAKTİK"
        else:
            recommendation = "İZLE"
    elif final_score >= 1:
        recommendation = "İZLE"
    else:
        recommendation = "ELE"

    return {
        "ticker": ticker,
        "score": final_score,
        "raw_score": raw_score,
        "structural_score": structural_score,
        "tactical_score": tactical_score,
        "has_conflict": has_conflict,
        "conflict_sources": sorted(sat_sources),
        "details": details,
        "recommendation": recommendation,
        "source_count": source_count,
        "signals": ticker_signals,
    }


def calc_all_confluence(signals, macro_regime=None, min_score=1, mkk_data=None,
                        sms_scores=None, ice_results=None):
    """Tüm ticker'lar için çakışma skoru hesapla ve sırala."""
    tickers = set(s['ticker'] for s in signals)
    results = []
    for ticker in tickers:
        result = calc_confluence_score(ticker, signals, macro_regime, mkk_data,
                                       sms_scores, ice_results)
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
        "TRADEABLE": "🟢🟢",
        "TAKTİK": "🔵",
        "İZLE": "🟡",
        "BEKLE": "🟠",
        "ELE": "🔴",
    }

    for r in results[:top_n]:
        emoji = rec_emoji.get(r['recommendation'], '')
        src = r.get('source_count', 0)
        struct = r.get('structural_score', 0)
        tact = r.get('tactical_score', 0)
        conflict = " ⚠️ÇELİŞKİ" if r.get('has_conflict') else ""
        lines.append(
            f"{emoji} <b>{r['ticker']}</b> — Skor: {r['score']} ({r['recommendation']}) "
            f"[{src} kaynak Y:{struct} T:{tact}]{conflict}")
        for d in r['details'][:5]:
            lines.append(f"  {d}")
        lines.append("")

    if len(results) > top_n:
        lines.append(f"... ve {len(results) - top_n} hisse daha")

    return "\n".join(lines)
