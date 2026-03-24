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

_LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}
_LIST_LABELS = {
    'alsat': 'AL/SAT Tarama',
    'tavan': 'Tavan Tarayici',
    'nw': 'NW Pivot AL',
    'rt': 'Regime Transition',
    'sbt': 'SBT Breakout',
}
_LIST_ICONS = {
    'alsat': '\U0001f4cb',  # clipboard
    'tavan': '\U0001f53a',  # up triangle
    'nw': '\U0001f4ca',     # chart
    'rt': '\u26a1',         # lightning
    'sbt': '\U0001f680',    # rocket
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
        "GÜÇLÜ_RISK_ON": "#4ade80",
        "RISK_ON": "#4ade80",
        "NÖTR": "#a1a1aa",
        "RISK_OFF": "#f87171",
        "GÜÇLÜ_RISK_OFF": "#f87171",
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
    _LIST_SHORT_HTML = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}
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

    # Limit order AI metnini HTML'e çevir
    limit_order_html = ""
    if limit_order_text:
        limit_order_html = _linkify_tickers(_markdown_to_html(limit_order_text))

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX Brifing — {now}</title>
<style>
{_NOX_CSS}

.briefing-container {{
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1.5rem;
}}

/* HEADER */
.briefing-header {{
    text-align: center;
    margin-bottom: 2rem;
}}
.briefing-header h1 {{
    font-family: var(--font-display);
    font-size: 1.8rem;
    color: var(--nox-cyan);
    margin: 0;
}}
.briefing-header .subtitle {{
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}}
.regime-badge {{
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 1rem;
    margin: 1rem 0;
    border: 1px solid;
}}

/* SECTION TITLES */
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
    font-weight: 600;
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
.ml-badge.ml-strong {{ background: rgba(250,204,21,0.15); color: #facc15; }}
.ml-badge.ml-mid {{ background: rgba(59,130,246,0.15); color: #3b82f6; }}
.ml-badge.ml-weak {{ background: rgba(239,68,68,0.15); color: #ef4444; }}

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
.sbt-badge.sbt-ap {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
.sbt-badge.sbt-a {{ background: rgba(96,165,250,0.2); color: #60a5fa; }}
.sbt-badge.sbt-b {{ background: rgba(161,161,170,0.12); color: #a1a1aa; }}
.sbt-badge.sbt-x {{ background: rgba(248,113,113,0.15); color: #f87171; }}

/* VOL TIER BADGE (Hacim-donus) */
.vol-tier {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
    white-space: nowrap;
}}
.vol-tier.vt-altin {{ background: rgba(250,204,21,0.2); color: #facc15; }}
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
.sector-badge.sector-ok {{ background: rgba(74,222,128,0.15); color: #4ade80; }}
.sector-badge.sector-warn {{ background: rgba(251,146,60,0.15); color: #fb923c; }}

/* GATE TAG */
.gate-tag {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.08rem 0.25rem;
    border-radius: 3px;
    background: rgba(248,113,113,0.12);
    color: #f87171;
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
.rerank-delta.up {{ color: #4ade80; }}
.rerank-delta.down {{ color: #f87171; }}

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
.overlap-badge.b4 {{ background: rgba(248,113,113,0.2); color: #f87171; }}
.overlap-badge.b3 {{ background: rgba(251,146,60,0.2); color: #fb923c; }}
.overlap-badge.b2 {{ background: rgba(250,204,21,0.2); color: #facc15; }}
.overlap-item {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}}
.overlap-item:last-child {{ border-bottom: none; }}
.overlap-item .ticker {{ font-weight: 600; min-width: 4rem; }}
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
    color: var(--nox-cyan);
    text-decoration: none;
    border-bottom: 1px dotted var(--nox-cyan);
    transition: opacity 0.15s;
}}
.tv-link:hover {{
    opacity: 0.7;
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
<div class="briefing-container">

    <!-- HEADER -->
    <div class="briefing-header">
        <h1>⬡ NOX Brifing</h1>
        <div class="subtitle">{now} · Otomatik piyasa analizi</div>
        <div class="regime-badge" style="color:{regime_color}; border-color:{regime_color}">
            {regime} (Risk Skoru: {risk_score})
        </div>
    </div>

    <!-- SHORTLIST -->
    <h2 class="section-title">⬡ Shortlist — Öncelikli Hisseler</h2>
    <div id="shortlistContainer"></div>

    <!-- SECTOR SUMMARY -->
    <div id="sectorSummary" style="margin-bottom:1.5rem"></div>

    <!-- SIGNAL LISTS -->
    <h2 class="section-title">📊 Sinyal Listeleri</h2>
    <div class="signal-grid" id="signalGrid"></div>

    <!-- OVERLAP -->
    <h2 class="section-title">🔥 Çapraz Çakışmalar</h2>
    <div id="overlapContainer"></div>

    <!-- ML RERANK -->
    <h2 class="section-title">🧠 ML Rerank Değişimi</h2>
    <div id="rerankContainer"></div>

    <!-- ML FILTERED -->
    <h2 class="section-title">⚠️ Filtreyle Elenenler</h2>
    <div id="filteredContainer"></div>

    <!-- ML GÜÇLÜ -->
    <h2 class="section-title">🤖 ML Güçlü (skor≥50)</h2>
    <div id="mlStrongContainer"></div>

    <!-- LIMIT ORDER STRATEGY (AI) -->
    <h2 class="section-title">💰 Giriş Stratejisi</h2>
    <div id="limitOrderContainer" class="briefing-text">{limit_order_html if limit_order_html else '<p style="color:var(--text-muted)">AI giriş stratejisi üretilmedi (--no-ai)</p>'}</div>

    <!-- MACRO -->
    <h2 class="section-title">🌍 Makro Durum</h2>
    <div class="macro-signals" id="macroSignals"></div>
    <div class="category-grid" id="categoryGrid"></div>
    <div class="macro-grid" id="macroGrid"></div>

    <!-- NEWS -->
    <h2 class="section-title">📰 Piyasa Haberleri</h2>
    <div id="newsContainer"></div>

    <!-- CONFLUENCE TABLE -->
    <h2 class="section-title">⬡ Çakışma Tablosu</h2>
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

<script>
const TV_BASE = 'https://www.tradingview.com/chart/?symbol=BIST:';
const LISTS = {lists_json};
const OVERLAP = {overlap_json};
const MACRO_DETAIL = {macro_detail_json};
const MACRO = {macro_json};
const CONFLUENCE = {confluence_json};
const NEWS = {news_json};
const SHORTLIST_SET = new Set({shortlist_set_json});
const SAT_SET = new Set({sat_set_json});
const RERANK_DATA = {rerank_json};
const FILTERED_DATA = {filtered_json};
const ML_STRONG = {ml_strong_json};
const SHORTLIST = {shortlist_json};
const SECTOR_SUMMARY = {sector_summary_json};

// ── Shortlist (Tier1/2A/2B) ──
(function() {{
    const container = document.getElementById('shortlistContainer');
    if (!SHORTLIST || Object.keys(SHORTLIST).length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Shortlist verisi yok</div>';
        return;
    }}
    const TV_BASE = '{_TV_BASE}';
    let html = '';
    ['tier1', 'tier2a', 'tier2b'].forEach(key => {{
        const list = SHORTLIST[key];
        if (!list || list.items.length === 0) return;
        html += `<div style="margin-bottom:1rem">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                <span style="font-size:1.1rem">${{list.icon}}</span>
                <span style="font-weight:600;color:var(--text-primary)">${{list.label}}</span>
                <span style="color:var(--text-muted);font-size:0.75rem">${{list.total}} hisse</span>
            </div>`;
        list.items.forEach((item, i) => {{
            const reasons = item.reasons.filter(r => !r.startsWith('🤖') && !r.startsWith('SBT:') && !r.startsWith('✅') && !r.startsWith('⚠️')).slice(0, 3).join(' ');
            const listsTag = (item.in_lists || []).map(l => ({{alsat:'AS',tavan:'TVN',nw:'NW',rt:'RT',sbt:'SBT'}})[l] || l).join('+');
            // ML badge
            let mlBadge = '';
            if (item.ml_score_short != null || item.ml_score_swing != null) {{
                const mkB = (val, pfx) => {{
                    if (val == null) return '';
                    const p = Math.round(val * 100);
                    const c = val >= 0.60 ? 'ml-strong' : val >= 0.40 ? 'ml-mid' : 'ml-weak';
                    return `<span class="ml-${{pfx}} ${{c}}">${{pfx.toUpperCase()}}${{p}}</span>`;
                }};
                mlBadge = `<span class="ml-dual">${{mkB(item.ml_score_short,'s')}}${{mkB(item.ml_score_swing,'w')}}</span>`;
            }}
            // SBT badge
            let sbtBadge = '';
            if (item.sbt_bucket) {{
                const sbtCls = {{'A+':'sbt-ap','A':'sbt-a','B':'sbt-b','C':'sbt-b','X':'sbt-x'}}[item.sbt_bucket] || '';
                sbtBadge = `<span class="sbt-badge ${{sbtCls}}">SBT:${{item.sbt_bucket}}</span>`;
            }}
            // Sector badge
            let sectorBadge = '';
            if (item.sector_index) {{
                if (item.sector_regime === 'AL') {{
                    sectorBadge = `<span class="sector-badge sector-ok">✅${{item.sector_index}}</span>`;
                }} else {{
                    sectorBadge = `<span class="sector-badge sector-warn">⚠️${{item.sector_index}}↓</span>`;
                }}
            }}
            html += `<div class="signal-card">
                <span class="rank">${{i+1}}</span>
                <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                ${{listsTag ? `<span class="lists-tag">${{listsTag}}</span>` : ''}}
                <span class="reasons">${{reasons}}</span>
                ${{sectorBadge}}${{sbtBadge}}${{mlBadge}}
                <span class="score-pill">${{item.score}}p</span>
            </div>`;
        }});
        html += '</div>';
    }});
    container.innerHTML = html;
}})();

// ── Endeks & Sektör Özeti ──
(function() {{
    const container = document.getElementById('sectorSummary');
    if (!SECTOR_SUMMARY || !SECTOR_SUMMARY.groups || SECTOR_SUMMARY.groups.length === 0) return;
    let html = '<div style="padding:0.5rem 0.8rem;background:var(--bg-card);border-radius:8px;border:1px solid var(--border-subtle)">';
    html += '<div style="font-weight:600;margin-bottom:0.4rem;font-size:0.85rem">📊 Endeks & Sektör Durumu <small style="color:var(--text-muted)">(' + SECTOR_SUMMARY.total + ' endeks)</small></div>';
    SECTOR_SUMMARY.groups.forEach(g => {{
        const alCount = g.al ? g.al.length : 0;
        html += '<div style="font-size:0.78rem;margin-bottom:0.25rem">';
        html += `<span style="font-weight:500">${{g.label}}</span> <small style="color:var(--text-muted)">${{alCount}}/${{g.total}} AL</small> `;
        if (g.al) g.al.forEach(s => {{ html += `<span class="sector-badge sector-ok">✅${{s}}</span> `; }});
        if (g.pasif) g.pasif.forEach(s => {{ html += `<span class="sector-badge sector-warn">⚠️${{s}}</span> `; }});
        html += '</div>';
    }});
    html += '</div>';
    container.innerHTML = html;
}})();

// ── Sinyal Listeleri (2x2 grid) ──
(function() {{
    const grid = document.getElementById('signalGrid');
    const order = ['alsat', 'tavan', 'nw', 'rt', 'sbt'];
    const scanUrls = {{
        'alsat': '{_bist_base}/',
        'tavan': '{_bist_base}/tavan.html',
        'nw': '{_nox_base}/nox_v3_weekly.html',
        'rt': '{_nox_base}/regime_transition.html',
        'sbt': '{_nox_base}/smart_breakout.html',
    }};
    order.forEach(key => {{
        const list = LISTS[key];
        if (!list) return;
        const sec = document.createElement('div');
        sec.className = 'signal-section';
        const scanUrl = scanUrls[key] || '#';
        let html = `<div class="list-header">
            <span><a href="${{scanUrl}}" target="_blank" style="color:inherit;text-decoration:none">${{list.icon}} ${{list.label}}</a></span>
            <span class="count">${{list.total}} sinyal</span>
        </div>`;
        if (list.items.length === 0) {{
            html += '<div style="color:var(--text-muted);font-size:0.8rem">Sinyal yok</div>';
        }}
        list.items.forEach((item, i) => {{
            // Filter out ML/SBT/sector badges from reasons (rendered separately)
            const reasons = item.reasons.filter(r => !r.startsWith('🤖') && !r.startsWith('SBT:') && !r.startsWith('✅') && !r.startsWith('⚠️')).slice(0, 4).join(' ');
            // Dual ML badge
            let mlBadge = '';
            if (item.ml_score_short != null || item.ml_score_swing != null) {{
                const mkBadge = (val, prefix) => {{
                    if (val == null) return '';
                    const pct = Math.round(val * 100);
                    const cls = val >= 0.60 ? 'ml-strong' : val >= 0.40 ? 'ml-mid' : 'ml-weak';
                    return `<span class="ml-${{prefix}} ${{cls}}">${{prefix.toUpperCase()}}${{pct}}</span>`;
                }};
                mlBadge = `<span class="ml-dual">${{mkBadge(item.ml_score_short,'s')}}${{mkBadge(item.ml_score_swing,'w')}}</span>`;
            }} else if (item.ml_score != null) {{
                const mlPct = Math.round(item.ml_score * 100);
                const mlCls = item.ml_score >= 0.60 ? 'ml-strong' : item.ml_score >= 0.40 ? 'ml-mid' : 'ml-weak';
                mlBadge = `<span class="ml-badge ${{mlCls}}">ML${{mlPct}}</span>`;
            }}
            // SBT badge
            let sbtBadge = '';
            if (item.sbt_bucket) {{
                const sbtCls = {{'A+':'sbt-ap','A':'sbt-a','B':'sbt-b','C':'sbt-b','X':'sbt-x'}}[item.sbt_bucket] || '';
                sbtBadge = `<span class="sbt-badge ${{sbtCls}}">SBT:${{item.sbt_bucket}}</span>`;
            }}
            // Gate tag
            let gateTag = '';
            if (item.gate_penalty >= 99) {{
                gateTag = '<span class="gate-tag">HARD</span>';
            }} else if (item.gate_penalty >= 1) {{
                gateTag = '<span class="gate-tag">soft</span>';
            }}
            // Vol tier badge (RT hacim-donus)
            let volBadge = '';
            if (item.vol_tier && item.vol_tier !== 'NORMAL') {{
                const vtCls = {{'ALTIN':'vt-altin','GUMUS':'vt-gumus','BRONZ':'vt-bronz'}}[item.vol_tier] || '';
                const vtIcon = item.vol_tier_icon || '';
                volBadge = `<span class="vol-tier ${{vtCls}}">${{vtIcon}}${{item.vol_tier}}</span>`;
            }}
            // Sector badge
            let sectorBadge = '';
            if (item.sector_index) {{
                if (item.sector_regime === 'AL') {{
                    sectorBadge = `<span class="sector-badge sector-ok">✅${{item.sector_index}}</span>`;
                }} else {{
                    sectorBadge = `<span class="sector-badge sector-warn">⚠️${{item.sector_index}}↓</span>`;
                }}
            }}
            html += `<div class="signal-card">
                <span class="rank">${{i+1}}</span>
                <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                <span class="reasons">${{reasons}}</span>
                ${{volBadge}}${{sectorBadge}}${{sbtBadge}}${{mlBadge}}${{gateTag}}
                <span class="score-pill">${{item.score}}p</span>
            </div>`;
        }});
        sec.innerHTML = html;
        grid.appendChild(sec);
    }});
}})();

// ── Çapraz Çakışmalar ──
(function() {{
    const container = document.getElementById('overlapContainer');
    if (!OVERLAP || OVERLAP.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Çakışma bulunamadı</div>';
        return;
    }}
    const badgeClass = {{4: 'b4', 3: 'b3', 2: 'b2'}};
    OVERLAP.forEach(group => {{
        const sec = document.createElement('div');
        sec.className = 'overlap-section';
        const bc = badgeClass[group.overlap_count] || 'b2';
        let html = `<div class="group-header">
            <span class="overlap-badge ${{bc}}">${{group.overlap_count}}x</span>
            <span>${{group.label}}</span>
            <span style="color:var(--text-muted);font-size:0.75rem;margin-left:auto">${{group.total}} hisse</span>
        </div>`;
        group.items.forEach(item => {{
            const listsTag = (item.in_lists || []).map(l => {{
                const short = {{'alsat':'AS','tavan':'TVN','nw':'NW','rt':'RT','sbt':'SBT'}}[l] || l;
                return short;
            }}).join('+');
            const relaxed = item.relaxed ? ' [RT↓]' : '';
            const reason0 = item.reasons && item.reasons.length > 0
                ? item.reasons.filter(r => !r.startsWith('🤖') && !r.startsWith('SBT:') && !r.startsWith('✅') && !r.startsWith('⚠️'))[0] || ''
                : '';
            // Dual ML badge
            let mlBadge = '';
            if (item.ml_score_short != null || item.ml_score_swing != null) {{
                const mkB = (val, pfx) => {{
                    if (val == null) return '';
                    const p = Math.round(val * 100);
                    const c = val >= 0.60 ? 'ml-strong' : val >= 0.40 ? 'ml-mid' : 'ml-weak';
                    return `<span class="ml-${{pfx}} ${{c}}">${{pfx.toUpperCase()}}${{p}}</span>`;
                }};
                mlBadge = `<span class="ml-dual">${{mkB(item.ml_score_short,'s')}}${{mkB(item.ml_score_swing,'w')}}</span>`;
            }} else if (item.ml_score != null) {{
                const mlPct = Math.round(item.ml_score * 100);
                const mlCls = item.ml_score >= 0.60 ? 'ml-strong' : item.ml_score >= 0.40 ? 'ml-mid' : 'ml-weak';
                mlBadge = `<span class="ml-badge ${{mlCls}}">ML${{mlPct}}</span>`;
            }}
            // SBT badge
            let sbtBadge = '';
            if (item.sbt_bucket) {{
                const sbtCls = {{'A+':'sbt-ap','A':'sbt-a','B':'sbt-b','C':'sbt-b','X':'sbt-x'}}[item.sbt_bucket] || '';
                sbtBadge = `<span class="sbt-badge ${{sbtCls}}">SBT:${{item.sbt_bucket}}</span>`;
            }}
            // Vol tier badge
            let volBadge = '';
            if (item.vol_tier && item.vol_tier !== 'NORMAL') {{
                const vtCls = {{'ALTIN':'vt-altin','GUMUS':'vt-gumus','BRONZ':'vt-bronz'}}[item.vol_tier] || '';
                const vtIcon = item.vol_tier_icon || '';
                volBadge = `<span class="vol-tier ${{vtCls}}">${{vtIcon}}${{item.vol_tier}}</span>`;
            }}
            // Sector badge
            let sectorBadge = '';
            if (item.sector_index) {{
                if (item.sector_regime === 'AL') {{
                    sectorBadge = `<span class="sector-badge sector-ok">✅${{item.sector_index}}</span>`;
                }} else {{
                    sectorBadge = `<span class="sector-badge sector-warn">⚠️${{item.sector_index}}↓</span>`;
                }}
            }}
            html += `<div class="overlap-item">
                <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                <span class="lists-tag">${{listsTag}}${{relaxed}}</span>
                <span class="reasons-text">${{reason0}}</span>
                ${{volBadge}}${{sectorBadge}}${{sbtBadge}}${{mlBadge}}
                <span class="quality">${{item.quality}}p</span>
            </div>`;
        }});
        sec.innerHTML = html;
        container.appendChild(sec);
    }});
}})();

// ── ML Rerank Değişimi ──
(function() {{
    const container = document.getElementById('rerankContainer');
    if (!RERANK_DATA || RERANK_DATA.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">ML rerank verisi yok</div>';
        return;
    }}
    const sec = document.createElement('div');
    sec.className = 'overlap-section';
    const ups = RERANK_DATA.filter(r => r.delta > 0).sort((a,b) => b.delta - a.delta);
    const downs = RERANK_DATA.filter(r => r.delta < 0).sort((a,b) => a.delta - b.delta);
    let html = '';
    if (ups.length > 0) {{
        html += '<div style="font-size:0.8rem;color:var(--nox-green);margin-bottom:0.3rem">↑ Yükselenler</div>';
        ups.slice(0, 8).forEach(r => {{
            html += `<div class="rerank-item">
                <a href="${{TV_BASE}}${{r.ticker}}" target="_blank" class="tv-link" style="font-weight:600;min-width:4rem">${{r.ticker}}</a>
                <span class="lists-tag">${{r.list_tag}}</span>
                <span style="font-size:0.8rem;color:var(--text-muted)">${{r.old_rank}}→${{r.new_rank}}</span>
                <span class="rerank-delta up">+${{r.delta}}</span>
            </div>`;
        }});
    }}
    if (downs.length > 0) {{
        html += '<div style="font-size:0.8rem;color:var(--nox-red);margin:0.5rem 0 0.3rem">↓ Düşenler</div>';
        downs.slice(0, 8).forEach(r => {{
            html += `<div class="rerank-item">
                <a href="${{TV_BASE}}${{r.ticker}}" target="_blank" class="tv-link" style="font-weight:600;min-width:4rem">${{r.ticker}}</a>
                <span class="lists-tag">${{r.list_tag}}</span>
                <span style="font-size:0.8rem;color:var(--text-muted)">${{r.old_rank}}→${{r.new_rank}}</span>
                <span class="rerank-delta down">${{r.delta}}</span>
            </div>`;
        }});
    }}
    sec.innerHTML = html;
    container.appendChild(sec);
}})();

// ── Filtreyle Elenenler ──
(function() {{
    const container = document.getElementById('filteredContainer');
    if (!FILTERED_DATA || FILTERED_DATA.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Elenen sinyal yok</div>';
        return;
    }}
    const listShort = {{'alsat':'AS','tavan':'TVN','nw':'NW','rt':'RT','sbt':'SBT','tier2a':'T2A','tier2b':'T2B','tier2':'T2'}};
    const sec = document.createElement('div');
    sec.className = 'overlap-section';
    let html = `<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.5rem">Rule güçlü ama ML zayıf — ${{FILTERED_DATA.length}} hisse elendi</div>`;
    FILTERED_DATA.slice(0, 10).forEach(item => {{
        const tag = listShort[item.list] || item.list;
        html += `<div class="filtered-item">
            <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link" style="font-weight:600;min-width:4rem">${{item.ticker}}</a>
            <span class="lists-tag">${{tag}}</span>
            <span style="font-family:var(--font-mono);font-size:0.75rem">rule:${{item.rule_score}}p</span>
            <span class="reason">${{item.reason}}</span>
        </div>`;
    }});
    sec.innerHTML = html;
    container.appendChild(sec);
}})();

// ── ML Güçlü ──
(function() {{
    const container = document.getElementById('mlStrongContainer');
    if (!ML_STRONG || ML_STRONG.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">ML≥50 sinyal yok</div>';
        return;
    }}
    const sec = document.createElement('div');
    sec.className = 'overlap-section';
    let html = `<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.5rem">ML skor ≥50 — ${{ML_STRONG.length}} hisse</div>`;
    ML_STRONG.slice(0, 15).forEach((item, idx) => {{
        const srcStr = item.sources ? item.sources.join('+') : '';
        const sColor = item.ml_short >= 55 ? 'var(--nox-cyan)' : item.ml_short >= 50 ? '#eab308' : 'var(--text-muted)';
        const wColor = item.ml_swing >= 55 ? 'var(--nox-cyan)' : item.ml_swing >= 50 ? '#eab308' : 'var(--text-muted)';
        html += `<div class="filtered-item">
            <span style="color:var(--text-muted);font-size:0.75rem;min-width:1.2rem">${{idx+1}}.</span>
            <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link" style="font-weight:600;min-width:4rem">${{item.ticker}}</a>
            <span style="font-family:var(--font-mono);font-size:0.8rem">S<span style="color:${{sColor}}">${{item.ml_short}}</span>·W<span style="color:${{wColor}}">${{item.ml_swing}}</span></span>
            <span class="lists-tag">${{srcStr}}</span>
        </div>`;
    }});
    sec.innerHTML = html;
    container.appendChild(sec);
}})();

// ── Makro Detay ──
(function() {{
    // Rejim sinyalleri
    const sigContainer = document.getElementById('macroSignals');
    if (MACRO_DETAIL.signals && MACRO_DETAIL.signals.length > 0) {{
        MACRO_DETAIL.signals.forEach(sig => {{
            const el = document.createElement('div');
            el.className = 'regime-signal';
            el.textContent = sig;
            sigContainer.appendChild(el);
        }});
    }} else {{
        sigContainer.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem">Makro veri yok</div>';
    }}

    // Kategori rejimleri
    const catGrid = document.getElementById('categoryGrid');
    const catOrder = ['BIST', 'US', 'FX', 'Emtia', 'Kripto', 'Faiz'];
    const regimeColors = {{
        'GÜÇLÜ_YUKARI': 'var(--nox-green)', 'YUKARI': 'var(--nox-green)',
        'NÖTR': 'var(--text-muted)',
        'AŞAĞI': 'var(--nox-red)', 'GÜÇLÜ_AŞAĞI': 'var(--nox-red)',
    }};
    const cats = MACRO_DETAIL.categories || {{}};
    catOrder.forEach(cat => {{
        const data = cats[cat];
        if (!data) return;
        const card = document.createElement('div');
        card.className = 'category-regime';
        const color = regimeColors[data.regime] || 'var(--text-secondary)';
        card.innerHTML = `
            <div class="cat-name">${{cat}}</div>
            <div class="cat-regime" style="color:${{color}}">${{data.regime}}</div>
            <div class="cat-score">skor: ${{data.score}}</div>
        `;
        catGrid.appendChild(card);
    }});

    // Enstruman grid
    const macroGrid = document.getElementById('macroGrid');
    MACRO.forEach(item => {{
        if (!item.price) return;
        const chg1d = item.chg_1d != null ? item.chg_1d.toFixed(1) : '-';
        const chg5d = item.chg_5d != null ? item.chg_5d.toFixed(1) : '-';
        const chg1m = item.chg_1m != null ? item.chg_1m.toFixed(1) : '-';
        const color = item.chg_1d > 0 ? 'var(--nox-green)' : item.chg_1d < 0 ? 'var(--nox-red)' : 'var(--text-muted)';
        const trend = item.trend === 'UP' ? '↑' : item.trend === 'DOWN' ? '↓' : '→';
        const emaTag = item.above_ema21 ? '<span style="color:var(--nox-green)">EMA↑</span>' : '<span style="color:var(--nox-red)">EMA↓</span>';
        const card = document.createElement('div');
        card.className = 'macro-card';
        card.innerHTML = `
            <div class="name">${{item.name}}</div>
            <div class="price">${{item.price.toLocaleString('tr-TR')}}</div>
            <div class="change" style="color:${{color}}">${{trend}} 1G:${{chg1d}}% · 5G:${{chg5d}}%</div>
            <div class="detail">1A:${{chg1m}}% ${{emaTag}}</div>
        `;
        macroGrid.appendChild(card);
    }});
}})();

// ── Piyasa Haberleri ──
(function() {{
    const container = document.getElementById('newsContainer');
    if (!NEWS || NEWS.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">Haber bulunamadı</div>';
        return;
    }}
    const sec = document.createElement('div');
    sec.className = 'news-section';
    let html = '';
    NEWS.forEach(item => {{
        const title = item.title || '';
        const link = item.link || '#';
        const source = item.source || '';
        const pubDate = item.pub_date || '';
        // Basit tarih parse
        let dateStr = '';
        if (pubDate) {{
            try {{
                const d = new Date(pubDate);
                dateStr = d.toLocaleDateString('tr-TR') + ' ' + d.toLocaleTimeString('tr-TR', {{hour:'2-digit', minute:'2-digit'}});
            }} catch(e) {{
                dateStr = pubDate;
            }}
        }}
        html += `<div class="news-item">
            <a href="${{link}}" target="_blank">${{title}}</a>
            <div class="news-meta">${{source}}${{source && dateStr ? ' — ' : ''}}${{dateStr}}</div>
        </div>`;
    }});
    sec.innerHTML = html;
    container.appendChild(sec);
}})();

// ── Çakışma Tablosu ──
(function() {{
    const tbody = document.querySelector('#confluenceTable tbody');
    const recColors = {{
        'TRADEABLE': '#4ade80',
        'TAKTİK': '#60a5fa',
        'İZLE': '#fbbf24',
        'BEKLE': '#fb923c',
        'ELE': '#f87171',
        'VERİ_YOK': '#71717a',
    }};
    CONFLUENCE.forEach(item => {{
        const tr = document.createElement('tr');
        const scoreColor = item.score >= 5 ? '#4ade80' : item.score >= 3 ? '#fbbf24' : item.score <= 0 ? '#f87171' : '#a1a1aa';
        const hasConflict = item.has_conflict || (SAT_SET.has(item.ticker) && item.recommendation !== 'BEKLE');
        const displayRec = hasConflict ? 'BEKLE' : item.recommendation;
        const recColor = recColors[displayRec] || '#a1a1aa';
        const details = (item.details || []).slice(0, 3).join('<br>');
        const structScore = item.structural_score || 0;
        const tactScore = item.tactical_score || 0;
        // Durum badge
        let durumHtml = '<span class="shortlist-badge not-in">—</span>';
        if (hasConflict) {{
            durumHtml = '<span class="shortlist-badge conflict">ÇELİŞKİ</span>';
        }} else if (SHORTLIST_SET.has(item.ticker)) {{
            durumHtml = '<span class="shortlist-badge in-list">SHORTLIST</span>';
        }}
        tr.innerHTML = `
            <td><a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link"><b>${{item.ticker}}</b></a></td>
            <td><span class="score-badge" style="background:${{scoreColor}}20;color:${{scoreColor}}">${{item.score}}</span></td>
            <td style="font-family:var(--font-mono);font-size:0.85rem;color:var(--text-secondary)">${{structScore}}</td>
            <td style="font-family:var(--font-mono);font-size:0.85rem;color:var(--text-secondary)">${{tactScore}}</td>
            <td><span class="rec-badge" style="background:${{recColor}}20;color:${{recColor}}">${{displayRec}}</span></td>
            <td>${{durumHtml}}</td>
            <td style="font-size:0.8rem;color:var(--text-secondary)">${{details}}</td>
        `;
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
