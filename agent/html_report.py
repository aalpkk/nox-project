"""
NOX Agent — Brifing HTML Raporu
Mevcut _NOX_CSS temasını kullanır.
"""
import json
import os
import re
from datetime import datetime, timezone, timedelta

from core.reports import _NOX_CSS, _sanitize

_TZ_TR = timezone(timedelta(hours=3))

_TV_BASE = 'https://www.tradingview.com/chart/?symbol=BIST:'
_TICKER_EXCLUDE = {
    'BIST', 'HAZIR', 'IZLE', 'BEKLE', 'DEVAM', 'TAZE', 'ZONE', 'PIVOT',
    'BADGE', 'TREND', 'FULL', 'CHOPPY', 'CMF', 'ADX', 'RSI', 'EMA', 'SMA',
    'ATR', 'VOL', 'ACIL', 'TAVAN', 'SINYAL', 'KAYNAK', 'YUKARI', 'ASAGI',
    'NOTR', 'RISK', 'KILITLI', 'SETUP', 'TRADE', 'ENTRY',
    'STOP', 'HTML', 'HTTP', 'HTTPS', 'OZEL', 'GENEL', 'DONUSU', 'DONUS',
    'DALGA', 'HAFTA', 'GUNLUK', 'MAX', 'MIN', 'NOT', 'SAT', 'SKOR',
    'DEGER', 'CIKIS', 'GIRIS', 'NEGATIF', 'POZITIF', 'REJiM', 'REJIM',
    'ONERI', 'UYARI', 'ALTIN', 'DOVIZ', 'FAIZ', 'EMTIA',
}

_LIST_SHORT = {'alsat': 'NYX-Tek', 'tavan': 'NYX9', 'nw': 'NYX-Dip', 'rt': 'NYX-Trend', 'sbt': 'NYX-Krlm'}
_LIST_LABELS = {
    'alsat': 'NYX Teknik Sinyal',
    'tavan': 'NYX 9',
    'nw': 'NYX Dip Pivot',
    'rt': 'NYX Trend',
    'sbt': 'NYX Kırılım',
}
_LIST_ICONS = {
    'alsat': '',
    'tavan': '',
    'nw': '',
    'rt': '',
    'sbt': '',
}


def _linkify_tickers(html_text):
    """HTML metninde BIST ticker isimlerini TradingView linklerine cevir."""
    def _replace(m):
        ticker = m.group(1)
        if ticker in _TICKER_EXCLUDE or len(ticker) < 3:
            return ticker
        return (f'<a href="{_TV_BASE}{ticker}" target="_blank" '
                f'class="tv-link" title="{ticker} — TradingView">{ticker}</a>')

    # HTML tag'lerini ayir, sadece metin kisimlarini isle
    parts = re.split(r'(<[^>]+>)', html_text)
    for i, part in enumerate(parts):
        if not part.startswith('<'):
            parts[i] = re.sub(r'\b([A-Z]{3,6})\b', _replace, part)
    return ''.join(parts)


def _prepare_lists_json(lists_dict, max_per_list=15):
    """4 listenin her birinden top N sinyali JSON-serializable dict'e cevir."""
    result = {}
    for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
        items = lists_dict.get(key, [])
        entries = []
        for ticker, score, reasons, sig in items[:max_per_list]:
            entry = {
                'ticker': ticker,
                'score': score,
                'reasons': reasons,
                'signal_type': sig.get('signal_type', '') if isinstance(sig, dict) else '',
            }
            if isinstance(sig, dict):
                # Dual ML skorları
                if sig.get('ml_score') is not None:
                    entry['ml_score'] = sig['ml_score']
                if sig.get('ml_score_short') is not None:
                    entry['ml_score_short'] = sig['ml_score_short']
                if sig.get('ml_score_swing') is not None:
                    entry['ml_score_swing'] = sig['ml_score_swing']
                # ML effect
                if sig.get('ml_effect'):
                    entry['ml_effect'] = sig['ml_effect']
                # SBT bucket + gate
                if sig.get('sbt_bucket'):
                    entry['sbt_bucket'] = sig['sbt_bucket']
                if sig.get('gate_penalty'):
                    entry['gate_penalty'] = sig['gate_penalty']
                if sig.get('_rule_score') is not None:
                    entry['rule_score'] = sig['_rule_score']
                # Vol tier (RT hacim-donus)
                if sig.get('vol_tier'):
                    entry['vol_tier'] = sig['vol_tier']
                    entry['vol_tier_icon'] = sig.get('vol_tier_icon', '')
                # Sektör regime
                if sig.get('sector_index'):
                    entry['sector_index'] = sig['sector_index']
                    entry['sector_regime'] = sig.get('sector_regime', '')
                # Breakout ML
                if sig.get('breakout_master') is not None:
                    entry['breakout_master'] = sig['breakout_master']
                if sig.get('breakout_fusion') is not None:
                    entry['breakout_fusion'] = sig['breakout_fusion']
                if sig.get('breakout_tier'):
                    entry['breakout_tier'] = sig['breakout_tier']
                # ICE kurumsal veriler
                if sig.get('ice_mult') is not None:
                    entry['ice_mult'] = sig['ice_mult']
                    entry['ice_icon'] = sig.get('ice_icon', '')
                if sig.get('cost_ratio'):
                    entry['cost_ratio'] = sig['cost_ratio']
                    entry['cost_value'] = sig.get('cost_value', '')
                if sig.get('streak_days'):
                    entry['streak_days'] = sig['streak_days']
                    entry['streak_momentum'] = sig.get('streak_momentum', '')
                if sig.get('position_change_pct') is not None:
                    entry['position_change_pct'] = sig['position_change_pct']
            entries.append(entry)
        result[key] = {
            'label': _LIST_LABELS.get(key, key),
            'icon': _LIST_ICONS.get(key, ''),
            'short': _LIST_SHORT.get(key, key),
            'total': len(items),
            'items': entries,
        }
    return result


def _prepare_overlap_json(lists_dict, max_per_group=15):
    """Tier1'den overlap_count bazli gruplama."""
    tier1 = lists_dict.get('tier1', [])
    groups = {}  # overlap_count -> items
    for ticker, quality, reasons, meta in tier1:
        oc = meta.get('overlap_count', 2) if isinstance(meta, dict) else 2
        entry = {
            'ticker': ticker,
            'quality': quality,
            'reasons': reasons,
            'overlap_count': oc,
            'in_lists': meta.get('in_lists', []) if isinstance(meta, dict) else [],
            'relaxed': meta.get('relaxed', False) if isinstance(meta, dict) else False,
        }
        if isinstance(meta, dict):
            if meta.get('ml_score') is not None:
                entry['ml_score'] = meta['ml_score']
            if meta.get('ml_score_short') is not None:
                entry['ml_score_short'] = meta['ml_score_short']
            if meta.get('ml_score_swing') is not None:
                entry['ml_score_swing'] = meta['ml_score_swing']
            if meta.get('ml_effect'):
                entry['ml_effect'] = meta['ml_effect']
            if meta.get('sbt_bucket'):
                entry['sbt_bucket'] = meta['sbt_bucket']
            if meta.get('gate_penalty'):
                entry['gate_penalty'] = meta['gate_penalty']
            # Vol tier (RT hacim-donus)
            if meta.get('vol_tier'):
                entry['vol_tier'] = meta['vol_tier']
                entry['vol_tier_icon'] = meta.get('vol_tier_icon', '')
            # Sektör regime
            if meta.get('sector_index'):
                entry['sector_index'] = meta['sector_index']
                entry['sector_regime'] = meta.get('sector_regime', '')
            # Breakout ML
            if meta.get('breakout_master') is not None:
                entry['breakout_master'] = meta['breakout_master']
            if meta.get('breakout_fusion') is not None:
                entry['breakout_fusion'] = meta['breakout_fusion']
            if meta.get('breakout_tier'):
                entry['breakout_tier'] = meta['breakout_tier']
            if meta.get('brk_ml_s') is not None:
                entry['brk_ml_s'] = meta['brk_ml_s']
            if meta.get('brk_avoid'):
                entry['brk_avoid'] = True
            # ICE kurumsal veriler
            if meta.get('ice_mult') is not None:
                entry['ice_mult'] = meta['ice_mult']
                entry['ice_icon'] = meta.get('ice_icon', '')
            if meta.get('cost_ratio'):
                entry['cost_ratio'] = meta['cost_ratio']
                entry['cost_value'] = meta.get('cost_value', '')
            if meta.get('streak_days'):
                entry['streak_days'] = meta['streak_days']
                entry['streak_momentum'] = meta.get('streak_momentum', '')
            if meta.get('position_change_pct') is not None:
                entry['position_change_pct'] = meta['position_change_pct']
        groups.setdefault(oc, []).append(entry)
    # Her grubun ilk max_per_group'unu al, buyuk overlap_count once
    result = []
    for oc in sorted(groups.keys(), reverse=True):
        result.append({
            'overlap_count': oc,
            'label': f'{oc} Cakisma',
            'items': groups[oc][:max_per_group],
            'total': len(groups[oc]),
        })
    return result


def _prepare_macro_detail(macro_data):
    """Makro detay: rejim sinyalleri, kategori rejimleri, enstruman grid."""
    if not macro_data:
        return {'signals': [], 'categories': {}, 'instruments': []}
    return {
        'signals': macro_data.get('signals', []),
        'categories': _sanitize(macro_data.get('category_regimes', {})),
        'instruments': _sanitize(macro_data.get('snapshot', [])),
    }


def _prepare_shortlist_json(lists_dict, max_items=15):
    """Tier1 + Tier2A + Tier2B shortlist verisini JSON-serializable dict'e cevir."""
    result = {}
    for key, label, icon in [
        ('tier1', 'Tier 1 — Çakışmalar', '🔥'),
        ('tier2a', 'Tier 2A — Tactical ⚡1G', '⚡'),
        ('tier2b', 'Tier 2B — Swing-Lite', '⭐'),
    ]:
        items = lists_dict.get(key, [])
        entries = []
        for ticker, score, reasons, meta in items[:max_items]:
            entry = {
                'ticker': ticker,
                'score': score,
                'reasons': reasons,
            }
            if isinstance(meta, dict):
                entry['in_lists'] = meta.get('in_lists', [])
                if meta.get('ml_score_short') is not None:
                    entry['ml_score_short'] = meta['ml_score_short']
                if meta.get('ml_score_swing') is not None:
                    entry['ml_score_swing'] = meta['ml_score_swing']
                if meta.get('ml_effect'):
                    entry['ml_effect'] = meta['ml_effect']
                if meta.get('sbt_bucket'):
                    entry['sbt_bucket'] = meta['sbt_bucket']
                if meta.get('sector_index'):
                    entry['sector_index'] = meta['sector_index']
                    entry['sector_regime'] = meta.get('sector_regime', '')
                # Breakout ML
                if meta.get('breakout_master') is not None:
                    entry['breakout_master'] = meta['breakout_master']
                if meta.get('breakout_fusion') is not None:
                    entry['breakout_fusion'] = meta['breakout_fusion']
                if meta.get('breakout_tier'):
                    entry['breakout_tier'] = meta['breakout_tier']
                if meta.get('brk_ml_s') is not None:
                    entry['brk_ml_s'] = meta['brk_ml_s']
                if meta.get('brk_avoid'):
                    entry['brk_avoid'] = True
                # ICE kurumsal veriler
                if meta.get('ice_mult') is not None:
                    entry['ice_mult'] = meta['ice_mult']
                    entry['ice_icon'] = meta.get('ice_icon', '')
                if meta.get('cost_ratio'):
                    entry['cost_ratio'] = meta['cost_ratio']
                    entry['cost_value'] = meta.get('cost_value', '')
                if meta.get('streak_days'):
                    entry['streak_days'] = meta['streak_days']
                    entry['streak_momentum'] = meta.get('streak_momentum', '')
                if meta.get('position_change_pct') is not None:
                    entry['position_change_pct'] = meta['position_change_pct']
                # Vol tier
                if meta.get('vol_tier'):
                    entry['vol_tier'] = meta['vol_tier']
                    entry['vol_tier_icon'] = meta.get('vol_tier_icon', '')
                # Gate
                if meta.get('gate_penalty'):
                    entry['gate_penalty'] = meta['gate_penalty']
            entries.append(entry)
        result[key] = {
            'label': label,
            'icon': icon,
            'total': len(items),
            'items': entries,
        }
    return result


def _prepare_sector_summary(lists_dict):
    """Tüm BIST endeksleri regime özeti — gruplu, AL sinyali bazlı."""
    from agent.sector_regime import _ALL_INDICES

    all_regimes = lists_dict.get('_index_regimes', {})
    if not all_regimes:
        return {}

    group_labels = {
        'piyasa': '📈 Piyasa',
        'sektor': '🏭 Sektör',
        'tematik': '🏷 Tematik',
        'katilim': '☪️ Katılım',
    }

    groups = []
    for group, codes in _ALL_INDICES.items():
        group_data = {c: all_regimes[c] for c in codes if c in all_regimes}
        if not group_data:
            continue
        al = sorted(k for k, v in group_data.items() if v.get('in_trade', False))
        pasif = sorted(k for k, v in group_data.items() if not v.get('in_trade', False))
        groups.append({
            'key': group,
            'label': group_labels.get(group, group),
            'al': al,
            'pasif': pasif,
            'total': len(group_data),
        })

    return {
        'groups': groups,
        'total': len(all_regimes),
    }


def generate_briefing_html(briefing_text, macro_data, confluence_results,
                           signal_summary, lists_dict=None, news_items=None,
                           shortlist_tickers=None, sat_tickers=None,
                           limit_order_text=None):
    """
    Brifing HTML raporu olustur.

    Args:
        briefing_text: Claude'un urettigi brifing metni
        macro_data: assess_macro_regime() sonucu
        confluence_results: calc_all_confluence() sonucu
        signal_summary: summarize_signals() sonucu
        lists_dict: _compute_4_lists() sonucu (optional)
        news_items: fetch_market_news() sonucu (optional)
        shortlist_tickers: set of tickers in shortlist (optional)
        sat_tickers: set of tickers with SAT signals (optional)
    """
    now = datetime.now(_TZ_TR).strftime('%d.%m.%Y %H:%M')
    regime = macro_data.get("regime", "N/A") if macro_data else "N/A"
    risk_score = macro_data.get("risk_score", 0) if macro_data else 0

    # Regime renk
    regime_colors = {
        "GÜÇLÜ_RISK_ON": "#7a9e7a",
        "RISK_ON": "#7a9e7a",
        "NÖTR": "#8a8580",
        "RISK_OFF": "#9e5a5a",
        "GÜÇLÜ_RISK_OFF": "#9e5a5a",
    }
    regime_color = regime_colors.get(regime, "#a1a1aa")

    # JSON veriler
    lists_json = json.dumps(
        _sanitize(_prepare_lists_json(lists_dict)) if lists_dict else {},
        ensure_ascii=False)
    overlap_json = json.dumps(
        _sanitize(_prepare_overlap_json(lists_dict)) if lists_dict else [],
        ensure_ascii=False)
    macro_detail_json = json.dumps(
        _sanitize(_prepare_macro_detail(macro_data)),
        ensure_ascii=False)
    macro_json = json.dumps(
        _sanitize(macro_data.get("snapshot", []) if macro_data else []),
        ensure_ascii=False)
    confluence_json = json.dumps(
        _sanitize(confluence_results[:30] if confluence_results else []),
        ensure_ascii=False)
    news_json = json.dumps(
        _sanitize(news_items or []),
        ensure_ascii=False)
    shortlist_set_json = json.dumps(
        list(shortlist_tickers) if shortlist_tickers else [],
        ensure_ascii=False)
    sat_set_json = json.dumps(
        list(sat_tickers) if sat_tickers else [],
        ensure_ascii=False)

    # Shortlist (tier1/2a/2b) + sector summary
    shortlist_json = json.dumps(
        _sanitize(_prepare_shortlist_json(lists_dict)) if lists_dict else {},
        ensure_ascii=False)
    sector_summary_json = json.dumps(
        _sanitize(_prepare_sector_summary(lists_dict)) if lists_dict else {},
        ensure_ascii=False)

    # ML rerank + filtered data
    rerank_json = json.dumps(
        _sanitize(lists_dict.get('_ml_rank_changes', [])) if lists_dict else [],
        ensure_ascii=False)
    filtered_json = json.dumps(
        _sanitize(lists_dict.get('_ml_filtered', [])) if lists_dict else [],
        ensure_ascii=False)

    # Scanner base URL'leri (HTML'deki JS linkleri için)
    _nox_base = os.environ.get("GH_PAGES_BASE_URL", "https://aalpkk.github.io/nox-signals").rstrip("/")
    _bist_base = os.environ.get("BIST_PAGES_BASE_URL", "https://aalpkk.github.io/bist-signals").rstrip("/")

    # ML Güçlü listesi (≥0.50)
    _LIST_SHORT_HTML = {'alsat': 'NYX-Tek', 'tavan': 'NYX9', 'nw': 'NYX-Dip', 'rt': 'NYX-Trend', 'sbt': 'NYX-Krlm'}
    ml_strong_items = []
    if lists_dict:
        _ml_seen_html = set()
        for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for ticker, score, reasons, sig in lists_dict.get(key, []):
                if ticker in _ml_seen_html or not isinstance(sig, dict):
                    continue
                s_val = sig.get('ml_score_short')
                w_val = sig.get('ml_score_swing')
                if s_val is None and w_val is None:
                    continue
                best = max(s_val or 0, w_val or 0)
                if best >= 0.50:
                    _ml_seen_html.add(ticker)
                    src = []
                    for ln in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
                        for t2, _, _, _ in lists_dict.get(ln, []):
                            if t2 == ticker:
                                src.append(_LIST_SHORT_HTML[ln])
                                break
                    ml_strong_items.append({
                        'ticker': ticker,
                        'ml_short': round(s_val * 100) if s_val else 0,
                        'ml_swing': round(w_val * 100) if w_val else 0,
                        'best': round(best * 100),
                        'sources': src,
                    })
        ml_strong_items.sort(key=lambda x: -x['best'])
    ml_strong_json = json.dumps(_sanitize(ml_strong_items), ensure_ascii=False)

    # Breakout ML alerts
    breakout_alerts = lists_dict.get('_breakout_alerts', []) if lists_dict else []
    breakout_ml_items = []
    for ba in breakout_alerts[:15]:
        breakout_ml_items.append({
            'ticker': ba.get('ticker', ''),
            'tavan_prob': round(ba.get('tavan_prob', 0) * 100) if ba.get('tavan_prob') else 0,
            'rally_prob': round(ba.get('rally_prob', 0) * 100) if ba.get('rally_prob') else 0,
            'master': round(ba.get('breakout_master', 0) * 100) if ba.get('breakout_master') else 0,
            'fusion': round(ba.get('fusion_score', 0) * 100) if ba.get('fusion_score') else 0,
            'ml_s': round(ba.get('ml_s_score', 0) * 100) if ba.get('ml_s_score') else 0,
            'tier': ba.get('tier', ''),
            'in_shortlist': ba.get('in_shortlist', ''),
        })
    breakout_ml_json = json.dumps(_sanitize(breakout_ml_items), ensure_ascii=False)

    # Limit Order TP sinyalleri
    limit_tp_items = lists_dict.get('_limit_tp', []) if lists_dict else []
    limit_tp_json = json.dumps(_sanitize(limit_tp_items), ensure_ascii=False)

    # Limit order AI metnini HTML'e çevir
    limit_order_html = ""
    if limit_order_text:
        limit_order_html = _linkify_tickers(_markdown_to_html(limit_order_text))

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYX Brifing — {now}</title>
<style>
{_NOX_CSS}

.briefing-container {{
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
    padding: 0 1.5rem 2rem;
}}

/* ─── KATMAN 1: DURUM (sticky top bar) ─── */
.nox-status-bar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(6,6,8,0.55);
    backdrop-filter: blur(24px) saturate(1.3);
    -webkit-backdrop-filter: blur(24px) saturate(1.3);
    border-bottom: 1px solid rgba(201,169,110,0.08);
    padding: 0.6rem 1.5rem;
    margin: 0 -1.5rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}}
.nox-status-bar .logo {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.15rem;
    white-space: nowrap;
}}
.nox-status-bar .logo .nox-text {{
    font-family: var(--font-brand);
    font-size: 6rem;
    color: #fff;
    letter-spacing: 0.06em;
    line-height: 0.85;
}}
.nox-status-bar .logo .brief-text {{
    font-family: var(--font-handwrite);
    font-size: 1.3rem;
    color: #fff;
    margin-left: 0.15rem;
    position: relative;
    top: -0.1rem;
}}
.nox-status-bar .regime-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 1rem;
    font-weight: 600;
    font-size: 0.72rem;
    border: 1px solid;
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.nox-status-bar .macro-pills {{
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    flex: 1;
}}
.nox-status-bar .mpill {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    white-space: nowrap;
}}
.nox-status-bar .mpill b {{ font-weight: 600; }}
.nox-status-bar .meta-right {{
    font-size: 0.7rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    white-space: nowrap;
    margin-left: auto;
}}

/* ─── KATMAN 1.5: Endeks uyarıları ─── */
.nox-index-warn {{
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
    padding: 0.5rem 0.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
}}

/* ─── KATMAN 2: AKSİYON KARTLARI ─── */
.action-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}}
.action-card {{
    background: rgba(199,189,190,0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: none;
    border-radius: 18px;
    padding: 0.9rem 1.1rem;
    border-left: 3px solid transparent;
    transition: all 0.25s ease;
    animation: cardFadeIn 0.3s ease-out both;
    position: relative;
    overflow: hidden;
}}
.action-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(184,149,110,0.15), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}}
.action-card:hover {{
    background: rgba(199,189,190,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(184,149,110,0.03);
    transform: translateY(-1px);
}}
.action-card:hover::before {{ opacity: 1; }}
.action-card.tier1 {{ border-left: 3px solid rgba(201,169,110,0.4); }}
.action-card.tier2a {{ border-left: 3px solid rgba(184,149,110,0.25); }}
.action-card.tier2b {{ border-left: 3px solid rgba(138,122,158,0.25); }}
@keyframes cardFadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.action-card .card-head {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}}
.action-card .card-head .ticker {{
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 1.15rem;
    color: #fff;
}}
.action-card .card-head .tier-tag {{
    font-size: 0.6rem;
    font-family: var(--font-mono);
    padding: 0.1rem 0.35rem;
    border-radius: 0.25rem;
    font-weight: 600;
    text-transform: uppercase;
}}
.action-card .card-head .tier-tag.t1 {{ background: rgba(201,169,110,0.12); color: #c9a96e; }}
.action-card .card-head .tier-tag.t2a {{ background: rgba(168,135,106,0.12); color: #a8876a; }}
.action-card .card-head .tier-tag.t2b {{ background: rgba(138,122,158,0.12); color: #8a7a9e; }}
.action-card .card-head .score-pill {{
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: var(--nox-gold-dim);
    color: var(--nox-gold);
    margin-left: auto;
}}
.action-card .card-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-bottom: 0.4rem;
}}
.action-card .card-reasons {{
    font-size: 0.75rem;
    color: var(--text-secondary);
    line-height: 1.4;
    margin-bottom: 0.4rem;
}}
.action-card .card-entry {{
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding-top: 0.4rem;
    border-top: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 0.72rem;
}}
.action-card .card-entry .e-label {{ color: var(--text-muted); }}
.action-card .card-entry .e-limit {{ color: var(--nox-gold); font-weight: 600; }}
.action-card .card-entry .e-sl {{ color: var(--nox-red); font-weight: 600; }}
.action-card .card-entry .e-tp {{ color: var(--nox-green); font-weight: 600; }}
.tier-group-label {{
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 1.2rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.tier-group-label:first-child {{ margin-top: 0; }}
.tier-group-label .cnt {{
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}}

/* ─── KATMAN 3: RADAR ─── */
.radar-section {{
    background: rgba(199,189,190,0.07);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: none;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}}
.radar-section .radar-header {{
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.radar-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.82rem;
}}
.radar-item:last-child {{ border-bottom: none; }}
.radar-item .rank {{ color: var(--text-muted); font-family: var(--font-mono); font-size: 0.72rem; min-width: 1.2rem; }}
.radar-item .ticker {{ font-family: 'Inter', sans-serif; font-weight: 900; font-size: 0.92rem; min-width: 4rem; }}
.radar-item .tag {{
    font-size: 0.65rem;
    padding: 0.08rem 0.3rem;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-weight: 600;
}}
.radar-item .tag.ml {{ background: rgba(192,132,252,0.15); color: var(--nox-purple); }}
.radar-item .tag.brk {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.radar-item .tag.ltp {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}

/* ─── KATMAN 4: DETAY (collapse) ─── */
.detail-block {{
    margin-bottom: 0.75rem;
}}
.detail-block > summary {{
    font-family: var(--font-display);
    color: var(--nox-gold);
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    padding: 0.6rem 0.8rem;
    background: rgba(199,189,190,0.07);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: none;
    border-radius: 16px;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.2s;
    list-style: none;
}}
.detail-block > summary::-webkit-details-marker {{ display: none; }}
.detail-block > summary::before {{
    content: '▸';
    font-size: 0.8rem;
    color: var(--text-muted);
    transition: transform 0.2s;
}}
.detail-block[open] > summary::before {{ transform: rotate(90deg); }}
.detail-block > summary:hover {{
    background: rgba(199,189,190,0.10);
}}
.detail-block > summary .det-count {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-left: auto;
}}
.detail-block .detail-body {{
    padding: 0.8rem;
    border: none;
    border-radius: 0 0 16px 16px;
    background: rgba(6,7,9,0.6);
}}

/* ─── TABS (sinyal listeleri) ─── */
.nox-tabs {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.75rem;
    overflow-x: auto;
}}
.nox-tab {{
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: color 0.15s, border-color 0.15s;
    background: none;
    border-top: none; border-left: none; border-right: none;
    font-family: var(--font-display);
}}
.nox-tab:hover {{ color: var(--text-secondary); }}
.nox-tab.active {{
    color: var(--nox-gold);
    border-bottom-color: var(--nox-gold);
}}
.nox-tab-panel {{ display: none; }}
.nox-tab-panel.active {{ display: block; }}

/* ─── SECTION TITLE (katman başlıkları) ─── */
.layer-title {{
    font-family: var(--font-display);
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid transparent;
    border-image: linear-gradient(90deg, rgba(201,169,110,0.2), transparent 60%) 1;
}}

/* ─── FOOTER ─── */
.nox-footer {{
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.35);
    font-family: var(--font-mono);
    letter-spacing: 0.1em;
}}

/* ─── GLOSSARY ─── */
.glossary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
}}
.glossary-group {{
    padding-bottom: 0.5rem;
}}
.glossary-cat {{
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--nox-gold);
    margin-bottom: 0.4rem;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid rgba(201,169,110,0.1);
}}
.glossary-row {{
    display: flex;
    gap: 0.6rem;
    padding: 0.2rem 0;
    font-size: 0.78rem;
    line-height: 1.4;
}}
.glossary-key {{
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--text-primary);
    min-width: 5.5rem;
    flex-shrink: 0;
    font-size: 0.72rem;
}}
.glossary-val {{
    color: var(--text-secondary);
    font-size: 0.74rem;
}}

/* SECTION TITLES (legacy — used in detail blocks) */
.section-title {{
    font-family: var(--font-display);
    color: var(--nox-cyan);
    font-size: 1.1rem;
    margin: 2.5rem 0 1rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
}}

/* SIGNAL LISTS — 2x2 grid */
.signal-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}}
@media (max-width: 700px) {{
    .signal-grid {{ grid-template-columns: 1fr; }}
}}
.signal-section {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem;
}}
.signal-section .list-header {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
}}
.signal-section .list-header .count {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-left: auto;
}}
.signal-card {{
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}}
.signal-card:last-child {{ border-bottom: none; }}
.signal-card .rank {{
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    min-width: 1.2rem;
}}
.signal-card .ticker {{
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 0.92rem;
    min-width: 4rem;
}}
.signal-card .reasons {{
    color: var(--text-secondary);
    font-size: 0.78rem;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.signal-card .score-pill {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: var(--nox-cyan-dim);
    color: var(--nox-cyan);
    white-space: nowrap;
}}
.ml-badge {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    white-space: nowrap;
}}
.ml-badge.ml-strong {{ background: rgba(201,169,110,0.15); color: var(--nox-gold); }}
.ml-badge.ml-mid {{ background: rgba(122,143,165,0.15); color: var(--nox-blue); }}
.ml-badge.ml-weak {{ background: rgba(158,90,90,0.15); color: var(--nox-red); }}

/* DUAL ML BADGE */
.ml-dual {{
    display: inline-flex;
    align-items: center;
    gap: 0.15rem;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    white-space: nowrap;
}}
.ml-dual .ml-s, .ml-dual .ml-w {{
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
}}
.ml-dual .ml-s {{ border-right: 1px solid var(--border-subtle); }}

/* SBT BADGE */
.sbt-badge {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    white-space: nowrap;
}}
.sbt-badge.sbt-ap {{ background: rgba(122,158,122,0.2); color: var(--nox-green); }}
.sbt-badge.sbt-a {{ background: rgba(122,143,165,0.2); color: var(--nox-blue); }}
.sbt-badge.sbt-b {{ background: rgba(138,133,128,0.12); color: var(--text-secondary); }}
.sbt-badge.sbt-x {{ background: rgba(158,90,90,0.15); color: var(--nox-red); }}

/* VOL TIER BADGE (Hacim-donus) */
.vol-tier {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
    white-space: nowrap;
}}
.vol-tier.vt-altin {{ background: rgba(201,169,110,0.15); color: var(--nox-gold); }}
.vol-tier.vt-gumus {{ background: rgba(192,192,192,0.2); color: #c0c0c0; }}
.vol-tier.vt-bronz {{ background: rgba(205,127,50,0.2); color: #cd7f32; }}

/* SECTOR BADGE */
.sector-badge {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    white-space: nowrap;
}}
.sector-badge.sector-ok {{ background: rgba(122,158,122,0.15); color: var(--nox-green); }}
.sector-badge.sector-warn {{ background: rgba(168,135,106,0.15); color: var(--nox-orange); }}

/* GATE TAG */
.gate-tag {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.08rem 0.25rem;
    border-radius: 3px;
    background: rgba(158,90,90,0.12);
    color: var(--nox-red);
    white-space: nowrap;
}}
.brk-avoid {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    background: rgba(158,90,90,0.18);
    color: var(--nox-red);
    font-weight: 600;
    white-space: nowrap;
}}

/* RERANK SECTION */
.rerank-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border-subtle);
}}
.rerank-item:last-child {{ border-bottom: none; }}
.rerank-delta {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
}}
.rerank-delta.up {{ color: var(--nox-green); }}
.rerank-delta.down {{ color: var(--nox-red); }}

/* FILTERED SECTION */
.filtered-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-muted);
}}
.filtered-item:last-child {{ border-bottom: none; }}
.filtered-item .reason {{ font-style: italic; }}

/* OVERLAP SECTION */
.overlap-section {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}}
.overlap-section .group-header {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
    font-size: 0.9rem;
}}
.overlap-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.75rem;
    font-family: var(--font-mono);
}}
.overlap-badge.b4 {{ background: rgba(158,90,90,0.2); color: var(--nox-red); }}
.overlap-badge.b3 {{ background: rgba(168,135,106,0.2); color: var(--nox-orange); }}
.overlap-badge.b2 {{ background: rgba(201,169,110,0.15); color: var(--nox-gold); }}
.overlap-item {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}}
.overlap-item:last-child {{ border-bottom: none; }}
.overlap-item .ticker {{ font-family: 'Inter', sans-serif; font-weight: 900; font-size: 0.92rem; min-width: 4rem; }}
.overlap-item .lists-tag {{
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: var(--bg-elevated);
    color: var(--text-secondary);
    font-family: var(--font-mono);
}}
.overlap-item .quality {{
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--nox-orange);
    margin-left: auto;
}}
.overlap-item .reasons-text {{
    font-size: 0.75rem;
    color: var(--text-muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* MACRO */
.macro-signals {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
}}
.regime-signal {{
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}}
.category-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.6rem;
    margin-bottom: 1rem;
}}
.category-regime {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.7rem 1rem;
}}
.category-regime .cat-name {{
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.category-regime .cat-regime {{
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 0.2rem;
}}
.category-regime .cat-score {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
}}
.macro-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0;
}}
.macro-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}}
.macro-card .name {{
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.macro-card .price {{
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
}}
.macro-card .change {{
    font-family: var(--font-mono);
    font-size: 0.8rem;
}}
.macro-card .detail {{
    font-size: 0.7rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-top: 0.2rem;
}}

/* NEWS */
.news-section {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}}
.news-item {{
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}}
.news-item:last-child {{ border-bottom: none; }}
.news-item a {{
    color: var(--text-primary);
    text-decoration: none;
    font-weight: 500;
}}
.news-item a:hover {{ color: var(--nox-cyan); }}
.news-item .news-meta {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}}

/* AI ANALYSIS */
.briefing-text {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
}}
.briefing-text h2 {{
    color: var(--nox-cyan);
    font-size: 1.1rem;
    margin-top: 1.2rem;
}}
.briefing-text strong {{
    color: var(--nox-orange);
}}

/* CONFLUENCE TABLE */
.confluence-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin: 1rem 0;
}}
.confluence-table th {{
    background: var(--bg-elevated);
    color: var(--text-secondary);
    text-align: left;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-subtle);
}}
.confluence-table td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}}
.confluence-table tr:hover {{
    background: var(--bg-hover);
}}
.score-badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
    font-family: var(--font-mono);
}}
.rec-badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
}}
.tv-link {{
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 0.95rem;
    text-decoration: none;
    transition: opacity 0.15s;
}}
.tv-link:hover {{
    opacity: 0.75;
    color: var(--nox-gold);
}}
.shortlist-badge {{
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 500;
}}
.shortlist-badge.in-list {{
    background: rgba(74,222,128,0.15);
    color: #4ade80;
}}
.shortlist-badge.conflict {{
    background: rgba(248,113,113,0.15);
    color: #f87171;
}}
.shortlist-badge.not-in {{
    color: var(--text-muted);
}}
</style>
</head>
<body>
<div class="aurora-bg">
    <div class="aurora-layer aurora-layer-1"></div>
    <div class="aurora-layer aurora-layer-2"></div>
    <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>
<div class="briefing-container">

    <!-- ══════ KATMAN 1: DURUM (sticky) ══════ -->
    <div class="nox-status-bar" id="statusBar">
        <span class="logo"><span class="nox-text">NYX</span><span class="brief-text">brief</span></span>
        <span class="regime-pill" style="color:{regime_color};border-color:{regime_color}">{regime}</span>
        <span class="macro-pills" id="macroPills"></span>
        <span class="meta-right">{now}</span>
    </div>
    <div class="nox-index-warn" id="indexWarn" style="display:none"></div>

    <!-- ══════ KATMAN 2: AKSİYON ══════ -->
    <div class="layer-title">Aksiyon — Shortlist</div>
    <div id="actionCards"></div>

    <!-- ══════ KATMAN 3: RADAR ══════ -->
    <div class="layer-title">Radar — Keşif</div>
    <div id="radarContainer"></div>

    <!-- ══════ KATMAN 4: DETAY ══════ -->
    <div class="layer-title">Detay</div>

    <!-- 4a: Sinyal Listeleri (tab'lı) -->
    <details class="detail-block" id="signalListsBlock">
        <summary>Sinyal Listeleri <span class="det-count" id="signalListsCount"></span></summary>
        <div class="detail-body">
            <div class="nox-tabs" id="signalTabs"></div>
            <div id="signalTabPanels"></div>
        </div>
    </details>

    <!-- 4b: AI Analiz -->
    <details class="detail-block">
        <summary>AI Analiz</summary>
        <div class="detail-body briefing-text">{limit_order_html if limit_order_html else '<p style="color:var(--text-muted)">AI giriş stratejisi üretilmedi (--no-ai)</p>'}</div>
    </details>

    <!-- 4c: ML Rerank + Filtered -->
    <details class="detail-block" id="mlDetailBlock">
        <summary>ML Detay <span class="det-count" id="mlDetailCount"></span></summary>
        <div class="detail-body">
            <div style="font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">Rerank Değişimi</div>
            <div id="rerankContainer"></div>
            <div style="font-weight:600;font-size:0.85rem;margin:1rem 0 0.5rem">Filtreyle Elenenler</div>
            <div id="filteredContainer"></div>
        </div>
    </details>

    <!-- 4d: Makro Detay -->
    <details class="detail-block">
        <summary>Makro Detay</summary>
        <div class="detail-body">
            <div class="macro-signals" id="macroSignals"></div>
            <div class="category-grid" id="categoryGrid"></div>
            <div class="macro-grid" id="macroGrid"></div>
        </div>
    </details>

    <!-- 4e: Backtest -->
    <details class="detail-block">
        <summary>Backtest Sonuçları</summary>
        <div class="detail-body">

        <h3 style="color:var(--nox-cyan);font-size:0.9rem;margin:0 0 0.5rem">Özet Sıralama (5G WR)</h3>
        <div style="overflow-x:auto">
        <table class="confluence-table" style="font-size:0.75rem">
            <thead><tr><th>#</th><th>Strateji</th><th>N</th><th>5G WR</th><th>5G ORT</th><th>5G MED</th></tr></thead>
            <tbody>
            <tr><td>1</td><td>OL rt+tavan</td><td>26</td><td style="color:#7a9e7a;font-weight:600">71.4%</td><td>+9.04%</td><td>+8.82%</td></tr>
            <tr><td>2</td><td>Rejim (RT)</td><td>102</td><td style="color:#7a9e7a;font-weight:600">66.7%</td><td>+4.55%</td><td>+2.63%</td></tr>
            <tr><td>3</td><td>ML≥0.50</td><td>200</td><td style="color:#7a9e7a;font-weight:600">64.5%</td><td>+4.04%</td><td>+2.92%</td></tr>
            <tr><td>4</td><td>ML≥0.55</td><td>103</td><td style="color:#7a9e7a;font-weight:600">63.1%</td><td>+4.11%</td><td>+3.20%</td></tr>
            <tr><td>5</td><td>OL nw+tavan</td><td>16</td><td style="color:#7a9e7a">61.5%</td><td>+5.93%</td><td>+4.14%</td></tr>
            <tr><td>6</td><td>OL 3+ liste</td><td>5</td><td style="color:#7a9e7a">60.0%</td><td>+5.34%</td><td>+4.14%</td></tr>
            <tr><td>7</td><td>NW Elmas</td><td>1140</td><td style="color:#c9a96e">59.6%</td><td>+2.16%</td><td>+1.05%</td></tr>
            <tr><td>8</td><td>AL/SAT</td><td>29</td><td style="color:#c9a96e">58.6%</td><td>+4.74%</td><td>+2.60%</td></tr>
            <tr><td>9</td><td>Tier2</td><td>180</td><td style="color:#c9a96e">58.5%</td><td>+3.71%</td><td>+2.20%</td></tr>
            <tr><td>10</td><td>T+Y (Taze+Yeni)</td><td>63</td><td style="color:#c9a96e">58.2%</td><td>+4.50%</td><td>+5.62%</td></tr>
            </tbody>
        </table>
        </div>

        <details style="margin-top:1rem">
            <summary style="color:var(--text-secondary);font-size:0.8rem;cursor:pointer">Shortlist Backtest Detay (20 gün, N=733)</summary>
            <div style="overflow-x:auto;margin-top:0.5rem">
            <table class="confluence-table" style="font-size:0.7rem">
                <thead><tr><th>Strateji</th><th>N</th><th>1G ORT</th><th>1G WR</th><th>3G ORT</th><th>3G WR</th><th>5G ORT</th><th>5G WR</th></tr></thead>
                <tbody>
                <tr><td>OL rt+tavan</td><td>26</td><td>4.57</td><td>70.8</td><td>8.32</td><td>77.3</td><td>9.04</td><td style="color:#7a9e7a">71.4</td></tr>
                <tr><td>OL 3+ liste</td><td>5</td><td>4.76</td><td>100.0</td><td>6.29</td><td>80.0</td><td>5.34</td><td>60.0</td></tr>
                <tr><td>T+Y (Taze+Yeni)</td><td>63</td><td>3.36</td><td>62.3</td><td>6.81</td><td>68.4</td><td>4.50</td><td>58.2</td></tr>
                <tr><td>OL nw+tavan</td><td>16</td><td>3.19</td><td>68.8</td><td>5.50</td><td>71.4</td><td>5.93</td><td>61.5</td></tr>
                <tr><td>Tier1</td><td>80</td><td>2.73</td><td>61.5</td><td>5.62</td><td>64.4</td><td>3.78</td><td>53.6</td></tr>
                <tr><td>Rejim (RT)</td><td>102</td><td>0.68</td><td>55.4</td><td>3.75</td><td>58.3</td><td>4.55</td><td style="color:#7a9e7a">66.7</td></tr>
                <tr><td>Tier2</td><td>180</td><td>1.69</td><td>58.1</td><td>2.76</td><td>54.4</td><td>3.71</td><td>58.5</td></tr>
                <tr><td>AL/SAT</td><td>29</td><td>0.54</td><td>44.8</td><td>2.65</td><td>55.2</td><td>4.74</td><td>58.6</td></tr>
                <tr><td>Tavan</td><td>180</td><td>0.72</td><td>44.0</td><td>0.90</td><td>48.9</td><td>1.88</td><td>54.2</td></tr>
                <tr><td>NW Pivot</td><td>162</td><td>0.27</td><td>55.8</td><td>0.52</td><td>50.0</td><td>0.15</td><td style="color:#9e5a5a">45.1</td></tr>
                <tr style="font-weight:600;border-top:2px solid var(--border-dim)"><td>GENEL</td><td>733</td><td>1.07</td><td>53.5</td><td>2.15</td><td>53.5</td><td>2.50</td><td>54.1</td></tr>
                </tbody>
            </table>
            </div>
        </details>

        <details style="margin-top:0.75rem">
            <summary style="color:var(--text-secondary);font-size:0.8rem;cursor:pointer">ML Filtreli (Shortlist üzeri)</summary>
            <div style="overflow-x:auto;margin-top:0.5rem">
            <table class="confluence-table" style="font-size:0.7rem">
                <thead><tr><th>ML Eşik</th><th>N</th><th>1G ORT</th><th>1G WR</th><th>3G ORT</th><th>3G WR</th><th>5G ORT</th><th>5G WR</th></tr></thead>
                <tbody>
                <tr><td>Baseline</td><td>505-649</td><td>1.07</td><td>53.5</td><td>2.15</td><td>53.5</td><td>2.50</td><td>54.1</td></tr>
                <tr><td>ML≥0.45</td><td>378-481</td><td>1.57</td><td>56.3</td><td>2.50</td><td>54.4</td><td>2.98</td><td>56.6</td></tr>
                <tr><td style="color:var(--nox-cyan)">ML≥0.50</td><td>200-254</td><td>2.32</td><td>63.0</td><td>3.52</td><td>61.0</td><td>4.04</td><td style="color:#7a9e7a;font-weight:600">64.5</td></tr>
                <tr><td>ML≥0.55</td><td>103-129</td><td>3.43</td><td>63.6</td><td>4.65</td><td>63.1</td><td>4.11</td><td style="color:#7a9e7a">63.1</td></tr>
                </tbody>
            </table>
            </div>
        </details>

        <details style="margin-top:0.75rem">
            <summary style="color:var(--text-secondary);font-size:0.8rem;cursor:pointer">Uzun Vadeli Backtestler (Tüm BIST, 2-3 yıl)</summary>
            <div style="overflow-x:auto;margin-top:0.5rem">
            <table class="confluence-table" style="font-size:0.7rem">
                <thead><tr><th>Strateji</th><th>N</th><th>1G ORT</th><th>1G WR</th><th>3G ORT</th><th>3G WR</th><th>5G ORT</th><th>5G WR</th></tr></thead>
                <tbody>
                <tr><td>NW Elmas</td><td>1140</td><td>0.41</td><td>51.1</td><td>1.17</td><td>53.5</td><td>2.16</td><td style="color:#c9a96e">59.6</td></tr>
                <tr><td>RT Volume (sıçrama)</td><td>4793</td><td>0.49</td><td>50.3</td><td>1.13</td><td>51.6</td><td>1.58</td><td>52.7</td></tr>
                <tr><td>RT Volume (tümü)</td><td>6953</td><td>0.50</td><td>49.9</td><td>1.03</td><td>50.8</td><td>1.53</td><td>52.4</td></tr>
                <tr><td>SBT Breakout</td><td>2061</td><td>0.39</td><td>50.9</td><td>0.48</td><td>50.1</td><td>0.65</td><td>50.3</td></tr>
                <tr><td>AL/SAT Dönüş</td><td>8897</td><td>0.15</td><td>46.5</td><td>0.34</td><td>45.7</td><td>0.51</td><td style="color:#9e5a5a">46.8</td></tr>
                <tr><td>Divergence BUY</td><td>27125</td><td>0.14</td><td>47.3</td><td>0.77</td><td>47.6</td><td>0.69</td><td style="color:#9e5a5a">48.8</td></tr>
                </tbody>
            </table>
            </div>
        </details>

        <div style="margin-top:0.75rem;font-size:0.7rem;color:var(--text-muted);line-height:1.5">
            <b>Çıkarımlar:</b> OL rt+tavan en iyi getiri/risk (medyan +8.82%) |
            ML≥0.50 en iyi N/WR dengesi (N=200, %64.5) |
            1G momentum: ML≥0.55 (ORT %3.43) |
            Medyan negatif = tehlike: AL/SAT Dönüş, Divergence
        </div>
        </div>
    </details>

    <!-- 4f: Haberler -->
    <details class="detail-block">
        <summary>Piyasa Haberleri</summary>
        <div class="detail-body" id="newsContainer"></div>
    </details>

    <!-- 4g: Çakışma Tablosu (en sonda) -->
    <details class="detail-block">
        <summary>Çakışma Tablosu <span class="det-count" id="confluenceCount"></span></summary>
        <div class="detail-body" style="overflow-x:auto">
            <table class="confluence-table" id="confluenceTable">
                <thead>
                    <tr>
                        <th>Hisse</th>
                        <th>Skor</th>
                        <th>Yapısal</th>
                        <th>Taktik</th>
                        <th>Sonuç</th>
                        <th>Durum</th>
                        <th>Detaylar</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </details>

    <!-- 4h: Sözlük -->
    <details class="detail-block">
        <summary>Kısaltmalar ve Terimler</summary>
        <div class="detail-body">
            <div class="glossary-grid">
                <div class="glossary-group">
                    <div class="glossary-cat">Sinyal Kaynakları</div>
                    <div class="glossary-row"><span class="glossary-key">NYX-Tek</span><span class="glossary-val">NYX Teknik Sinyal — teknik dönüş sinyalleri (RSI, MACD, hacim)</span></div>
                    <div class="glossary-row"><span class="glossary-key">NYX9</span><span class="glossary-val">NYX 9 — tavan/kilitli tavan adayları, streak takibi</span></div>
                    <div class="glossary-row"><span class="glossary-key">NYX-Dip</span><span class="glossary-val">NYX Dip Pivot — haftalık Elmas pivot kırılım sinyalleri</span></div>
                    <div class="glossary-row"><span class="glossary-key">NYX-Trend</span><span class="glossary-val">NYX Trend — rejim geçiş sinyalleri (trend başlangıcı)</span></div>
                    <div class="glossary-row"><span class="glossary-key">NYX-Krlm</span><span class="glossary-val">NYX Kırılım — birikim→kırılım pattern tespiti</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">Tier Sistemi</div>
                    <div class="glossary-row"><span class="glossary-key">T1</span><span class="glossary-val">Tier 1 — 2+ listede çakışan, en güçlü sinyaller</span></div>
                    <div class="glossary-row"><span class="glossary-key">T2A</span><span class="glossary-val">Tier 2A — Taktik (1G momentum), kısa vadeli fırsatlar</span></div>
                    <div class="glossary-row"><span class="glossary-key">T2B</span><span class="glossary-val">Tier 2B — Swing-Lite, orta vadeli izleme listesi</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">ML Skorları</div>
                    <div class="glossary-row"><span class="glossary-key">S##</span><span class="glossary-val">ML Short skoru — 1 günlük yükseliş olasılığı (%)</span></div>
                    <div class="glossary-row"><span class="glossary-key">W##</span><span class="glossary-val">ML Swing skoru — 3 günlük yükseliş olasılığı (%)</span></div>
                    <div class="glossary-row"><span class="glossary-key">🟢 ≥60</span><span class="glossary-val">Güçlü ML sinyali</span></div>
                    <div class="glossary-row"><span class="glossary-key">🟡 40-59</span><span class="glossary-val">Orta ML sinyali</span></div>
                    <div class="glossary-row"><span class="glossary-key">🔴 &lt;40</span><span class="glossary-val">Zayıf ML sinyali</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">SBT Bucket</div>
                    <div class="glossary-row"><span class="glossary-key">SBT:A+</span><span class="glossary-val">En güçlü breakout adayı</span></div>
                    <div class="glossary-row"><span class="glossary-key">SBT:A</span><span class="glossary-val">Güçlü breakout adayı</span></div>
                    <div class="glossary-row"><span class="glossary-key">SBT:B</span><span class="glossary-val">Orta — izlemeye devam</span></div>
                    <div class="glossary-row"><span class="glossary-key">SBT:X</span><span class="glossary-val">Zayıf/bozulmuş setup → gate cezası</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">ICE (Kurumsal Teyit)</div>
                    <div class="glossary-row"><span class="glossary-key">ICE×1.15</span><span class="glossary-val">Kurumsal çarpan — ≥1.15 güçlü kurumsal destek</span></div>
                    <div class="glossary-row"><span class="glossary-key">r=0.95</span><span class="glossary-val">Maliyet avantajı oranı — SM broker ortalama maliyeti / fiyat. &lt;1 = kurumsal karda</span></div>
                    <div class="glossary-row"><span class="glossary-key">SM4g💪</span><span class="glossary-val">Smart Money streak — 4 gün üst üste net alım, güçlü momentum</span></div>
                    <div class="glossary-row"><span class="glossary-key">Δ+2.3%</span><span class="glossary-val">Pozisyon değişimi — kurumsal net alım yüzdesi (son gün)</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">Breakout ML</div>
                    <div class="glossary-row"><span class="glossary-key">BRK🎯T5</span><span class="glossary-val">Birikim→Breakout Top 5 — en yüksek kırılım olasılığı</span></div>
                    <div class="glossary-row"><span class="glossary-key">BRK⚡T10</span><span class="glossary-val">Birikim→Breakout Top 6-10 — izle</span></div>
                    <div class="glossary-row"><span class="glossary-key">F##</span><span class="glossary-val">Fusion skoru — 0.4×Master + 0.6×ML_S birleşik tahmin</span></div>
                    <div class="glossary-row"><span class="glossary-key">⛔ALMA</span><span class="glossary-val">ML tarafından elenen (zayıf breakout + zayıf ML)</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">Hacim & Sektör</div>
                    <div class="glossary-row"><span class="glossary-key">🥇ALTIN</span><span class="glossary-val">Altın hacim tier — ortalama hacmin 3x+ sıçrama</span></div>
                    <div class="glossary-row"><span class="glossary-key">🥈GUMUS</span><span class="glossary-val">Gümüş hacim tier — ortalama hacmin 2-3x sıçrama</span></div>
                    <div class="glossary-row"><span class="glossary-key">✅XUHIZ</span><span class="glossary-val">Sektör endeksi AL — endeks rejimi yükseliş trendinde</span></div>
                    <div class="glossary-row"><span class="glossary-key">⚠️XBANK↓</span><span class="glossary-val">Sektör endeksi PASİF — endeks rejimi düşüş/nötr</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">Giriş Stratejisi</div>
                    <div class="glossary-row"><span class="glossary-key">Limit</span><span class="glossary-val">Limit emir fiyatı (kapanışın -%1.5 altı)</span></div>
                    <div class="glossary-row"><span class="glossary-key">TP</span><span class="glossary-val">Take Profit — hedef fiyat (%4 yukarı)</span></div>
                    <div class="glossary-row"><span class="glossary-key">streak</span><span class="glossary-val">Ardışık tavan/güçlü gün sayısı</span></div>
                    <div class="glossary-row"><span class="glossary-key">##p</span><span class="glossary-val">Rule engine puanı — sinyal güç skoru</span></div>
                </div>
                <div class="glossary-group">
                    <div class="glossary-cat">Makro</div>
                    <div class="glossary-row"><span class="glossary-key">RISK_ON</span><span class="glossary-val">Piyasa olumlu — alım için uygun ortam</span></div>
                    <div class="glossary-row"><span class="glossary-key">NÖTR</span><span class="glossary-val">Kararsız — seçici ol</span></div>
                    <div class="glossary-row"><span class="glossary-key">RISK_OFF</span><span class="glossary-val">Piyasa olumsuz — defansif kal</span></div>
                </div>
            </div>
        </div>
    </details>

    <div class="nox-footer">by AAK. YTD.</div>
</div>

<script>
const TV = 'https://www.tradingview.com/chart/?symbol=BIST:';
const LISTS = {lists_json};
const OVERLAP = {overlap_json};
const MD = {macro_detail_json};
const MACRO = {macro_json};
const CONF = {confluence_json};
const NEWS = {news_json};
const SL_SET = new Set({shortlist_set_json});
const SAT_SET = new Set({sat_set_json});
const RERANK = {rerank_json};
const FILTERED = {filtered_json};
const ML_S = {ml_strong_json};
const BRK_ML = {breakout_ml_json};
const LTP = {limit_tp_json};
const SL = {shortlist_json};
const SEC = {sector_summary_json};
const SCAN = {{
    'alsat': '{_bist_base}/',
    'tavan': '{_bist_base}/tavan.html',
    'nw': '{_nox_base}/nox_v3_weekly.html',
    'rt': '{_nox_base}/regime_transition.html',
    'sbt': '{_nox_base}/smart_breakout.html',
}};

// ── Utility: Badge render helpers ──
function tvL(t) {{ return `<a href="${{TV}}${{t}}" target="_blank" class="tv-link">${{t}}</a>`; }}
function mlBadge(item) {{
    const mk = (v,p) => {{
        if(v==null) return '';
        const n=Math.round(v*100), c=v>=0.60?'ml-strong':v>=0.40?'ml-mid':'ml-weak';
        return `<span class="ml-${{p}} ${{c}}">${{p.toUpperCase()}}${{n}}</span>`;
    }};
    if(item.ml_score_short!=null||item.ml_score_swing!=null)
        return `<span class="ml-dual">${{mk(item.ml_score_short,'s')}}${{mk(item.ml_score_swing,'w')}}</span>`;
    return '';
}}
function sbtBadge(item) {{
    if(!item.sbt_bucket) return '';
    const c={{'A+':'sbt-ap','A':'sbt-a','B':'sbt-b','C':'sbt-b','X':'sbt-x'}}[item.sbt_bucket]||'';
    return `<span class="sbt-badge ${{c}}">SBT:${{item.sbt_bucket}}</span>`;
}}
function sectorBadge(item) {{
    if(!item.sector_index) return '';
    return item.sector_regime==='AL'
        ? `<span class="sector-badge sector-ok">✅${{item.sector_index}}</span>`
        : `<span class="sector-badge sector-warn">⚠️${{item.sector_index}}↓</span>`;
}}
function brkBadge(item) {{
    if(item.brk_avoid) return `<span class="brk-avoid">⛔ALMA</span>`;
    if(item.breakout_tier==='top5') {{
        const f=item.breakout_fusion?Math.round(item.breakout_fusion*100):'';
        return `<span class="ml-badge ml-strong" style="font-size:0.65rem">BRK🎯T5${{f?'·F'+f:''}}</span>`;
    }}
    if(item.breakout_tier==='top10') {{
        const f=item.breakout_fusion?Math.round(item.breakout_fusion*100):'';
        return `<span class="ml-badge ml-mid" style="font-size:0.65rem">BRK⚡T10${{f?'·F'+f:''}}</span>`;
    }}
    return '';
}}
function iceBadge(item) {{
    if(item.ice_mult==null) return '';
    const c=item.ice_mult>=1.15?'#7a9e7a':item.ice_mult>=1.02?'#c9a96e':item.ice_mult>=0.90?'#8a8580':'#9e5a5a';
    let x='';
    if(item.streak_days>=3) {{ const m=item.streak_momentum==='GÜÇLÜ'?'💪':''; x+=` SM${{item.streak_days}}g${{m}}`; }}
    if(item.position_change_pct!=null&&Math.abs(item.position_change_pct)>=0.5) {{ x+=` Δ${{item.position_change_pct>0?'+':''}}${{item.position_change_pct.toFixed(1)}}%`; }}
    if(item.cost_ratio) x+=` r=${{item.cost_ratio.toFixed(2)}}`;
    return `<span style="background:${{c}}18;color:${{c}};padding:0.1rem 0.35rem;border-radius:0.25rem;font-size:0.65rem;font-weight:600;white-space:nowrap">ICE×${{item.ice_mult.toFixed(2)}}${{x}}</span>`;
}}
function volBadge(item) {{
    if(!item.vol_tier||item.vol_tier==='NORMAL') return '';
    const c={{'ALTIN':'vt-altin','GUMUS':'vt-gumus','BRONZ':'vt-bronz'}}[item.vol_tier]||'';
    return `<span class="vol-tier ${{c}}">${{item.vol_tier_icon||''}}${{item.vol_tier}}</span>`;
}}
function filterReasons(reasons) {{
    return (reasons||[]).filter(r=>!r.startsWith('🤖')&&!r.startsWith('SBT:')&&!r.startsWith('✅')&&!(r.startsWith('⚠️')&&!r.startsWith('⚠️taban')));
}}

// ══════ KATMAN 1: Makro pills (sticky bar) ══════
(function() {{
    const pills = document.getElementById('macroPills');
    // Key macro instruments for pills
    const keyNames = ['XU100', 'USDTRY', 'SPY', 'VIX'];
    const keyItems = MACRO.filter(m => keyNames.some(k => (m.name||'').toUpperCase().includes(k)));
    keyItems.forEach(item => {{
        if(!item.price) return;
        const chg = item.chg_1d != null ? item.chg_1d.toFixed(1) : '-';
        const arrow = item.chg_1d > 0 ? '↑' : item.chg_1d < 0 ? '↓' : '→';
        const color = item.chg_1d > 0 ? 'var(--nox-green)' : item.chg_1d < 0 ? 'var(--nox-red)' : 'var(--text-muted)';
        pills.innerHTML += `<span class="mpill"><b style="color:${{color}}">${{arrow}}</b> ${{item.name}} <b style="color:${{color}}">${{chg}}%</b></span>`;
    }});
    // Add AL count from categories
    const cats = MD.categories || {{}};
    const bistCat = cats['BIST'];
    if(bistCat) {{
        pills.innerHTML += `<span class="mpill">BIST <b>${{bistCat.regime}}</b></span>`;
    }}
}})();

// ── Index warnings (sadece ⚠️ olanlar) ──
(function() {{
    const container = document.getElementById('indexWarn');
    if(!SEC || !SEC.groups || SEC.groups.length===0) return;
    let warns = [];
    SEC.groups.forEach(g => {{
        if(g.pasif && g.pasif.length > 0) {{
            g.pasif.forEach(s => warns.push(s));
        }}
    }});
    if(warns.length === 0) return;
    container.style.display = 'flex';
    const alCount = SEC.groups.reduce((s,g) => s + (g.al?g.al.length:0), 0);
    container.innerHTML = `<span style="font-size:0.75rem;color:var(--text-muted);margin-right:0.3rem">⚠️ Pasif endeksler (${{warns.length}}/${{SEC.total}}):</span>` +
        warns.map(s => `<span class="sector-badge sector-warn" style="font-size:0.7rem">${{s}}</span>`).join(' ');
}})();

// ══════ KATMAN 2: AKSİYON KARTLARI ══════
(function() {{
    const container = document.getElementById('actionCards');
    if(!SL || Object.keys(SL).length===0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Shortlist verisi yok</div>';
        return;
    }}
    // Build limit TP lookup: ticker -> entry data
    const ltpMap = {{}};
    (LTP||[]).forEach(ltp => {{ ltpMap[ltp.ticker] = ltp; }});

    let html = '';
    const tierConf = {{
        tier1:  {{ cls:'tier1', tag:'T1', tagCls:'t1' }},
        tier2a: {{ cls:'tier2a', tag:'T2A', tagCls:'t2a' }},
        tier2b: {{ cls:'tier2b', tag:'T2B', tagCls:'t2b' }},
    }};
    let cardIdx = 0;
    ['tier1','tier2a','tier2b'].forEach(key => {{
        const list = SL[key];
        if(!list || list.items.length===0) return;
        const tc = tierConf[key];
        html += `<div class="tier-group-label">${{list.icon}} ${{list.label}} <span class="cnt">${{list.total}}</span></div>`;
        html += '<div class="action-grid">';
        list.items.forEach((item,i) => {{
            const delay = (cardIdx * 0.04).toFixed(2);
            cardIdx++;
            const reasons = filterReasons(item.reasons).slice(0,3).join(' · ');
            const listsTag = (item.in_lists||[]).map(l=>({{alsat:'NYX-Tek',tavan:'NYX9',nw:'NYX-Dip',rt:'NYX-Trend',sbt:'NYX-Krlm'}})[l]||l).join('+');
            // Entry info from limit TP
            const ltp = ltpMap[item.ticker];
            let entryHtml = '';
            if(ltp) {{
                entryHtml = `<div class="card-entry">
                    <span><span class="e-label">Limit</span> <span class="e-limit">${{ltp.limit_price.toFixed(2)}}</span></span>
                    <span><span class="e-label">TP</span> <span class="e-tp">${{ltp.tp_price.toFixed(2)}}</span></span>
                    <span class="e-tp">+%${{ltp.net_pct.toFixed(1)}}</span>
                    <span style="color:var(--text-muted)">streak=${{ltp.streak}}</span>
                </div>`;
            }}
            html += `<div class="action-card ${{tc.cls}}" style="animation-delay:${{delay}}s">
                <div class="card-head">
                    ${{tvL(item.ticker)}}
                    <span class="tier-tag ${{tc.tagCls}}">${{tc.tag}}</span>
                    ${{listsTag ? `<span style="font-size:0.65rem;color:var(--text-muted);font-family:var(--font-mono)">${{listsTag}}</span>` : ''}}
                    <span class="score-pill">${{item.score}}p</span>
                </div>
                <div class="card-badges">
                    ${{mlBadge(item)}}${{sbtBadge(item)}}${{sectorBadge(item)}}${{brkBadge(item)}}${{iceBadge(item)}}${{volBadge(item)}}
                </div>
                <div class="card-reasons">${{reasons}}</div>
                ${{entryHtml}}
            </div>`;
        }});
        html += '</div>';
    }});
    container.innerHTML = html;
}})();

// ══════ KATMAN 3: RADAR ══════
(function() {{
    const container = document.getElementById('radarContainer');
    // Combine ML Güçlü + Breakout ML + Limit TP (exclude those already in shortlist)
    const slTickers = new Set();
    ['tier1','tier2a','tier2b'].forEach(k => {{
        if(SL[k]) SL[k].items.forEach(it => slTickers.add(it.ticker));
    }});

    let items = [];

    // ML Güçlü (shortlist'te olmayanlar)
    (ML_S||[]).forEach(m => {{
        if(slTickers.has(m.ticker)) return;
        items.push({{ticker:m.ticker, type:'ml', sort:m.best,
            html:`<span class="tag ml">ML S${{m.ml_short}}·W${{m.ml_swing}}</span><span style="font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono)">${{m.sources.join('+')}}</span>`}});
    }});
    // Breakout ML (shortlist'te olmayanlar)
    (BRK_ML||[]).forEach(b => {{
        if(slTickers.has(b.ticker) || items.some(x=>x.ticker===b.ticker)) return;
        const fC = b.fusion>=80?'var(--nox-green)':b.fusion>=60?'#c9a96e':'var(--text-secondary)';
        items.push({{ticker:b.ticker, type:'brk', sort:b.fusion,
            html:`<span class="tag brk">BRK F${{b.fusion}}</span><span style="font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono)">TVN:${{b.tavan_prob}} S:${{b.ml_s}}</span>`}});
    }});
    // Limit TP (shortlist'te olmayanlar)
    (LTP||[]).forEach(ltp => {{
        if(slTickers.has(ltp.ticker) || items.some(x=>x.ticker===ltp.ticker)) return;
        items.push({{ticker:ltp.ticker, type:'ltp', sort:90,
            html:`<span class="tag ltp">LTP +%${{ltp.net_pct.toFixed(1)}}</span><span style="font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono)">streak=${{ltp.streak}} S${{ltp.ml_s}}</span>`}});
    }});

    items.sort((a,b) => b.sort - a.sort);

    if(items.length===0) {{
        container.innerHTML = '<div class="radar-section"><div style="color:var(--text-muted);font-size:0.85rem">Radar boş — tüm güçlü sinyaller shortlistte</div></div>';
        return;
    }}
    let html = '<div class="radar-section"><div class="radar-header">🔭 Keşif <span style="font-size:0.7rem;color:var(--text-muted);font-weight:400">shortlist dışı güçlü sinyaller</span></div>';
    items.slice(0,15).forEach((it,i) => {{
        html += `<div class="radar-item">
            <span class="rank">${{i+1}}</span>
            <span class="ticker">${{tvL(it.ticker)}}</span>
            ${{it.html}}
        </div>`;
    }});
    html += '</div>';
    container.innerHTML = html;
}})();

// ══════ KATMAN 4a: Sinyal Listeleri (tab'lı) ══════
(function() {{
    const tabsEl = document.getElementById('signalTabs');
    const panelsEl = document.getElementById('signalTabPanels');
    const order = ['alsat','tavan','nw','rt','sbt'];
    let totalSignals = 0;
    order.forEach((key,idx) => {{
        const list = LISTS[key];
        if(!list) return;
        totalSignals += list.total;
        const isFirst = idx === 0;
        // Tab button
        tabsEl.innerHTML += `<button class="nox-tab${{isFirst?' active':''}}" data-tab="${{key}}">${{list.icon}} ${{list.short}} <span style="font-size:0.65rem;color:var(--text-muted)">${{list.total}}</span></button>`;
        // Tab panel
        const scanUrl = SCAN[key] || '#';
        let html = `<div class="nox-tab-panel${{isFirst?' active':''}}" data-tab="${{key}}">`;
        html += `<div style="margin-bottom:0.5rem;font-size:0.8rem"><a href="${{scanUrl}}" target="_blank" style="color:var(--nox-cyan);text-decoration:none">${{list.label}} →</a></div>`;
        if(list.items.length===0) {{
            html += '<div style="color:var(--text-muted);font-size:0.8rem">Sinyal yok</div>';
        }}
        list.items.forEach((item,i) => {{
            const reasons = filterReasons(item.reasons).slice(0,4).join(' ');
            // Gate tag
            let gateTag = '';
            if(item.gate_penalty>=99) gateTag='<span class="gate-tag">HARD</span>';
            else if(item.gate_penalty>=1) gateTag='<span class="gate-tag">soft</span>';
            html += `<div class="signal-card">
                <span class="rank">${{i+1}}</span>
                <a href="${{TV}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                <span class="reasons">${{reasons}}</span>
                ${{volBadge(item)}}${{sectorBadge(item)}}${{sbtBadge(item)}}${{mlBadge(item)}}${{brkBadge(item)}}${{gateTag}}${{iceBadge(item)}}
                <span class="score-pill">${{item.score}}p</span>
            </div>`;
        }});
        html += '</div>';
        panelsEl.innerHTML += html;
    }});
    document.getElementById('signalListsCount').textContent = totalSignals + ' sinyal';
    // Tab switch logic
    tabsEl.addEventListener('click', e => {{
        const btn = e.target.closest('.nox-tab');
        if(!btn) return;
        const key = btn.dataset.tab;
        tabsEl.querySelectorAll('.nox-tab').forEach(b => b.classList.toggle('active', b.dataset.tab===key));
        panelsEl.querySelectorAll('.nox-tab-panel').forEach(p => p.classList.toggle('active', p.dataset.tab===key));
    }});
}})();

// ══════ KATMAN 4c: ML Rerank + Filtered ══════
(function() {{
    let cnt = 0;
    // Rerank
    const rc = document.getElementById('rerankContainer');
    if(!RERANK || RERANK.length===0) {{
        rc.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem">Rerank verisi yok</div>';
    }} else {{
        cnt += RERANK.length;
        const ups = RERANK.filter(r=>r.delta>0).sort((a,b)=>b.delta-a.delta);
        const downs = RERANK.filter(r=>r.delta<0).sort((a,b)=>a.delta-b.delta);
        let h = '';
        if(ups.length) {{
            h += '<div style="font-size:0.78rem;color:var(--nox-green);margin-bottom:0.3rem">↑ Yükselenler</div>';
            ups.slice(0,8).forEach(r => {{
                h += `<div class="rerank-item">${{tvL(r.ticker)}} <span class="lists-tag">${{r.list_tag}}</span> <span style="font-size:0.78rem;color:var(--text-muted)">${{r.old_rank}}→${{r.new_rank}}</span> <span class="rerank-delta up">+${{r.delta}}</span></div>`;
            }});
        }}
        if(downs.length) {{
            h += '<div style="font-size:0.78rem;color:var(--nox-red);margin:0.5rem 0 0.3rem">↓ Düşenler</div>';
            downs.slice(0,8).forEach(r => {{
                h += `<div class="rerank-item">${{tvL(r.ticker)}} <span class="lists-tag">${{r.list_tag}}</span> <span style="font-size:0.78rem;color:var(--text-muted)">${{r.old_rank}}→${{r.new_rank}}</span> <span class="rerank-delta down">${{r.delta}}</span></div>`;
            }});
        }}
        rc.innerHTML = h;
    }}
    // Filtered
    const fc = document.getElementById('filteredContainer');
    const LS = {{'alsat':'NYX-Tek','tavan':'NYX9','nw':'NYX-Dip','rt':'NYX-Trend','sbt':'NYX-Krlm','tier2a':'T2A','tier2b':'T2B','tier2':'T2'}};
    if(!FILTERED || FILTERED.length===0) {{
        fc.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem">Elenen sinyal yok</div>';
    }} else {{
        cnt += FILTERED.length;
        let h = `<div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:0.4rem">Rule güçlü ama ML zayıf — ${{FILTERED.length}} hisse</div>`;
        FILTERED.slice(0,10).forEach(item => {{
            h += `<div class="filtered-item">${{tvL(item.ticker)}} <span class="lists-tag">${{LS[item.list]||item.list}}</span> <span style="font-family:var(--font-mono);font-size:0.72rem">rule:${{item.rule_score}}p</span> <span class="reason">${{item.reason}}</span></div>`;
        }});
        fc.innerHTML = h;
    }}
    document.getElementById('mlDetailCount').textContent = cnt ? cnt + ' kayıt' : '';
}})();

// ══════ KATMAN 4d: Makro Detay ══════
(function() {{
    const sigC = document.getElementById('macroSignals');
    if(MD.signals && MD.signals.length>0) {{
        MD.signals.forEach(sig => {{
            const el = document.createElement('div');
            el.className = 'regime-signal';
            el.textContent = sig;
            sigC.appendChild(el);
        }});
    }} else {{
        sigC.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem">Makro veri yok</div>';
    }}
    const catGrid = document.getElementById('categoryGrid');
    const catOrder = ['BIST','US','FX','Emtia','Kripto','Faiz'];
    const rc = {{'GÜÇLÜ_YUKARI':'var(--nox-green)','YUKARI':'var(--nox-green)','NÖTR':'var(--text-muted)','AŞAĞI':'var(--nox-red)','GÜÇLÜ_AŞAĞI':'var(--nox-red)'}};
    const cats = MD.categories || {{}};
    catOrder.forEach(cat => {{
        const d = cats[cat]; if(!d) return;
        const card = document.createElement('div');
        card.className = 'category-regime';
        card.innerHTML = `<div class="cat-name">${{cat}}</div><div class="cat-regime" style="color:${{rc[d.regime]||'var(--text-secondary)'}}">${{d.regime}}</div><div class="cat-score">skor: ${{d.score}}</div>`;
        catGrid.appendChild(card);
    }});
    const mg = document.getElementById('macroGrid');
    MACRO.forEach(item => {{
        if(!item.price) return;
        const c1 = item.chg_1d!=null?item.chg_1d.toFixed(1):'-';
        const c5 = item.chg_5d!=null?item.chg_5d.toFixed(1):'-';
        const cm = item.chg_1m!=null?item.chg_1m.toFixed(1):'-';
        const clr = item.chg_1d>0?'var(--nox-green)':item.chg_1d<0?'var(--nox-red)':'var(--text-muted)';
        const t = item.trend==='UP'?'↑':item.trend==='DOWN'?'↓':'→';
        const ema = item.above_ema21?'<span style="color:var(--nox-green)">EMA↑</span>':'<span style="color:var(--nox-red)">EMA↓</span>';
        const card = document.createElement('div');
        card.className = 'macro-card';
        card.innerHTML = `<div class="name">${{item.name}}</div><div class="price">${{item.price.toLocaleString('tr-TR')}}</div><div class="change" style="color:${{clr}}">${{t}} 1G:${{c1}}% · 5G:${{c5}}%</div><div class="detail">1A:${{cm}}% ${{ema}}</div>`;
        mg.appendChild(card);
    }});
}})();

// ══════ KATMAN 4f: Haberler ══════
(function() {{
    const c = document.getElementById('newsContainer');
    if(!NEWS || NEWS.length===0) {{ c.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Haber bulunamadı</div>'; return; }}
    let h = '';
    NEWS.forEach(item => {{
        let ds = '';
        if(item.pub_date) {{ try {{ const d=new Date(item.pub_date); ds=d.toLocaleDateString('tr-TR')+' '+d.toLocaleTimeString('tr-TR',{{hour:'2-digit',minute:'2-digit'}}); }} catch(e){{ ds=item.pub_date; }} }}
        h += `<div class="news-item"><a href="${{item.link||'#'}}" target="_blank">${{item.title||''}}</a><div class="news-meta">${{item.source||''}}${{item.source&&ds?' — ':''}}${{ds}}</div></div>`;
    }});
    c.innerHTML = h;
}})();

// ══════ KATMAN 4g: Çakışma Tablosu ══════
(function() {{
    const tbody = document.querySelector('#confluenceTable tbody');
    const recC = {{'TRADEABLE':'#7a9e7a','TAKTİK':'#7a8fa5','İZLE':'#c9a96e','BEKLE':'#a8876a','ELE':'#9e5a5a','VERİ_YOK':'#555250'}};
    document.getElementById('confluenceCount').textContent = CONF.length ? CONF.length + ' hisse' : '';
    CONF.forEach(item => {{
        const tr = document.createElement('tr');
        const sc = item.score>=5?'#7a9e7a':item.score>=3?'#c9a96e':item.score<=0?'#9e5a5a':'#8a8580';
        const hc = item.has_conflict||(SAT_SET.has(item.ticker)&&item.recommendation!=='BEKLE');
        const dr = hc?'BEKLE':item.recommendation;
        const rc = recC[dr]||'#8a8580';
        const det = (item.details||[]).slice(0,3).join('<br>');
        let dur = '<span class="shortlist-badge not-in">—</span>';
        if(hc) dur='<span class="shortlist-badge conflict">ÇELİŞKİ</span>';
        else if(SL_SET.has(item.ticker)) dur='<span class="shortlist-badge in-list">SHORTLIST</span>';
        tr.innerHTML = `<td>${{tvL(item.ticker)}}</td><td><span class="score-badge" style="background:${{sc}}20;color:${{sc}}">${{item.score}}</span></td><td style="font-family:var(--font-mono);font-size:0.85rem;color:var(--text-secondary)">${{item.structural_score||0}}</td><td style="font-family:var(--font-mono);font-size:0.85rem;color:var(--text-secondary)">${{item.tactical_score||0}}</td><td><span class="rec-badge" style="background:${{rc}}20;color:${{rc}}">${{dr}}</span></td><td>${{dur}}</td><td style="font-size:0.8rem;color:var(--text-secondary)">${{det}}</td>`;
        tbody.appendChild(tr);
    }});
}})();
</script>
</body>
</html>"""
    return html


def _markdown_to_html(text):
    """Basit markdown -> HTML donusumu."""
    if not text:
        return ""
    lines = text.split('\n')
    result = []
    for line in lines:
        # Headers
        if line.startswith('## '):
            result.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith('### '):
            result.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith('# '):
            result.append(f"<h2>{line[2:]}</h2>")
        # Bold
        elif '**' in line:
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            result.append(f"<p>{line}</p>")
        # List items
        elif line.startswith('- '):
            result.append(f"<li>{line[2:]}</li>")
        elif line.strip() == '':
            result.append('<br>')
        else:
            result.append(f"<p>{line}</p>")
    return '\n'.join(result)
