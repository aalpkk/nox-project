"""
NOX Agent — Brifing HTML Raporu
Mevcut _NOX_CSS temasını kullanır.
"""
import json
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

_LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}
_LIST_LABELS = {
    'alsat': 'AL/SAT Tarama',
    'tavan': 'Tavan Tarayici',
    'nw': 'NW Pivot AL',
    'rt': 'Regime Transition',
}
_LIST_ICONS = {
    'alsat': '\U0001f4cb',  # clipboard
    'tavan': '\U0001f53a',  # up triangle
    'nw': '\U0001f4ca',     # chart
    'rt': '\u26a1',         # lightning
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


def _prepare_lists_json(lists_dict, max_per_list=5):
    """4 listenin her birinden top N sinyali JSON-serializable dict'e cevir."""
    result = {}
    for key in ('alsat', 'tavan', 'nw', 'rt'):
        items = lists_dict.get(key, [])
        entries = []
        for ticker, score, reasons, sig in items[:max_per_list]:
            entries.append({
                'ticker': ticker,
                'score': score,
                'reasons': reasons,
                'signal_type': sig.get('signal_type', '') if isinstance(sig, dict) else '',
            })
        result[key] = {
            'label': _LIST_LABELS.get(key, key),
            'icon': _LIST_ICONS.get(key, ''),
            'short': _LIST_SHORT.get(key, key),
            'total': len(items),
            'items': entries,
        }
    return result


def _prepare_overlap_json(lists_dict, max_per_group=5):
    """Tier1'den overlap_count bazli gruplama."""
    tier1 = lists_dict.get('tier1', [])
    groups = {}  # overlap_count -> items
    for ticker, quality, reasons, meta in tier1:
        oc = meta.get('overlap_count', 2) if isinstance(meta, dict) else 2
        groups.setdefault(oc, []).append({
            'ticker': ticker,
            'quality': quality,
            'reasons': reasons,
            'overlap_count': oc,
            'in_lists': meta.get('in_lists', []) if isinstance(meta, dict) else [],
            'relaxed': meta.get('relaxed', False) if isinstance(meta, dict) else False,
        })
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


def generate_briefing_html(briefing_text, macro_data, confluence_results,
                           signal_summary, lists_dict=None, news_items=None):
    """
    Brifing HTML raporu olustur.

    Args:
        briefing_text: Claude'un urettigi brifing metni
        macro_data: assess_macro_regime() sonucu
        confluence_results: calc_all_confluence() sonucu
        signal_summary: summarize_signals() sonucu
        lists_dict: _compute_4_lists() sonucu (optional)
        news_items: fetch_market_news() sonucu (optional)
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

    # Brifing metnini HTML'e cevir + ticker linkleri
    briefing_html = _linkify_tickers(_markdown_to_html(briefing_text))

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

    <!-- SIGNAL LISTS -->
    <h2 class="section-title">📊 Sinyal Listeleri</h2>
    <div class="signal-grid" id="signalGrid"></div>

    <!-- OVERLAP -->
    <h2 class="section-title">🔥 Çapraz Çakışmalar</h2>
    <div id="overlapContainer"></div>

    <!-- MACRO -->
    <h2 class="section-title">🌍 Makro Durum</h2>
    <div class="macro-signals" id="macroSignals"></div>
    <div class="category-grid" id="categoryGrid"></div>
    <div class="macro-grid" id="macroGrid"></div>

    <!-- NEWS -->
    <h2 class="section-title">📰 Piyasa Haberleri</h2>
    <div id="newsContainer"></div>

    <!-- AI ANALYSIS -->
    <h2 class="section-title">🤖 AI Analiz</h2>
    <div class="briefing-text">
        {briefing_html}
    </div>

    <!-- CONFLUENCE TABLE -->
    <h2 class="section-title">⬡ Çakışma Tablosu</h2>
    <table class="confluence-table" id="confluenceTable">
        <thead>
            <tr>
                <th>Hisse</th>
                <th>Skor</th>
                <th>Tavsiye</th>
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

// ── Sinyal Listeleri (2x2 grid) ──
(function() {{
    const grid = document.getElementById('signalGrid');
    const order = ['alsat', 'tavan', 'nw', 'rt'];
    order.forEach(key => {{
        const list = LISTS[key];
        if (!list) return;
        const sec = document.createElement('div');
        sec.className = 'signal-section';
        let html = `<div class="list-header">
            <span>${{list.icon}} ${{list.label}}</span>
            <span class="count">${{list.total}} sinyal</span>
        </div>`;
        if (list.items.length === 0) {{
            html += '<div style="color:var(--text-muted);font-size:0.8rem">Sinyal yok</div>';
        }}
        list.items.forEach((item, i) => {{
            const reasons = item.reasons.slice(0, 4).join(' ');
            html += `<div class="signal-card">
                <span class="rank">${{i+1}}</span>
                <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                <span class="reasons">${{reasons}}</span>
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
                const short = {{'alsat':'AS','tavan':'TVN','nw':'NW','rt':'RT'}}[l] || l;
                return short;
            }}).join('+');
            const relaxed = item.relaxed ? ' [RT↓]' : '';
            const reason0 = item.reasons && item.reasons.length > 0 ? item.reasons[0] : '';
            html += `<div class="overlap-item">
                <a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link ticker">${{item.ticker}}</a>
                <span class="lists-tag">${{listsTag}}${{relaxed}}</span>
                <span class="reasons-text">${{reason0}}</span>
                <span class="quality">${{item.quality}}p</span>
            </div>`;
        }});
        sec.innerHTML = html;
        container.appendChild(sec);
    }});
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
        'GÜÇLÜ_AL': '#4ade80', 'AL': '#4ade80', 'İZLE': '#fbbf24',
        'NÖTR': '#a1a1aa', 'KAÇIN': '#f87171', 'VERİ_YOK': '#71717a'
    }};
    CONFLUENCE.forEach(item => {{
        const tr = document.createElement('tr');
        const scoreColor = item.score >= 5 ? '#4ade80' : item.score >= 3 ? '#fbbf24' : item.score <= 0 ? '#f87171' : '#a1a1aa';
        const recColor = recColors[item.recommendation] || '#a1a1aa';
        const details = (item.details || []).slice(0, 3).join('<br>');
        tr.innerHTML = `
            <td><a href="${{TV_BASE}}${{item.ticker}}" target="_blank" class="tv-link"><b>${{item.ticker}}</b></a></td>
            <td><span class="score-badge" style="background:${{scoreColor}}20;color:${{scoreColor}}">${{item.score}}</span></td>
            <td><span class="rec-badge" style="background:${{recColor}}20;color:${{recColor}}">${{item.recommendation}}</span></td>
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
