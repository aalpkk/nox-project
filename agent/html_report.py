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


def _linkify_tickers(html_text):
    """HTML metninde BIST ticker isimlerini TradingView linklerine çevir."""
    def _replace(m):
        ticker = m.group(1)
        if ticker in _TICKER_EXCLUDE or len(ticker) < 3:
            return ticker
        return (f'<a href="{_TV_BASE}{ticker}" target="_blank" '
                f'class="tv-link" title="{ticker} — TradingView">{ticker}</a>')

    # HTML tag'lerini ayır, sadece metin kısımlarını işle
    parts = re.split(r'(<[^>]+>)', html_text)
    for i, part in enumerate(parts):
        if not part.startswith('<'):
            parts[i] = re.sub(r'\b([A-Z]{3,6})\b', _replace, part)
    return ''.join(parts)


def generate_briefing_html(briefing_text, macro_data, confluence_results,
                           signal_summary):
    """
    Brifing HTML raporu oluştur.

    Args:
        briefing_text: Claude'un ürettiği brifing metni
        macro_data: assess_macro_regime() sonucu
        confluence_results: calc_all_confluence() sonucu
        signal_summary: summarize_signals() sonucu
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

    # Makro enstrüman tablosu JSON
    macro_json = json.dumps(
        _sanitize(macro_data.get("snapshot", []) if macro_data else []),
        ensure_ascii=False)
    confluence_json = json.dumps(
        _sanitize(confluence_results[:30] if confluence_results else []),
        ensure_ascii=False)
    summary_json = json.dumps(
        _sanitize(signal_summary) if signal_summary else {},
        ensure_ascii=False)

    # Brifing metnini HTML'e çevir (markdown-like) + ticker linkleri
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
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1.5rem;
}}
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
.section-title {{
    font-family: var(--font-display);
    color: var(--nox-cyan);
    font-size: 1.1rem;
    margin: 2rem 0 1rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
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
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.macro-card .price {{
    font-family: var(--font-mono);
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
}}
.macro-card .change {{
    font-family: var(--font-mono);
    font-size: 0.85rem;
}}
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
    <div class="briefing-header">
        <h1>⬡ NOX Brifing</h1>
        <div class="subtitle">{now} · Otomatik piyasa analizi</div>
        <div class="regime-badge" style="color:{regime_color}; border-color:{regime_color}">
            {regime} (Risk Skoru: {risk_score})
        </div>
    </div>

    <div class="briefing-text">
        {briefing_html}
    </div>

    <h2 class="section-title">🌍 Makro Enstrümanlar</h2>
    <div class="macro-grid" id="macroGrid"></div>

    <h2 class="section-title">⬡ Çakışma Analizi</h2>
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
const MACRO = {macro_json};
const CONFLUENCE = {confluence_json};

// Makro grid
const macroGrid = document.getElementById('macroGrid');
MACRO.forEach(item => {{
    if (!item.price) return;
    const chg1d = item.chg_1d != null ? item.chg_1d.toFixed(1) : '-';
    const chg5d = item.chg_5d != null ? item.chg_5d.toFixed(1) : '-';
    const color = item.chg_1d > 0 ? 'var(--nox-green)' : item.chg_1d < 0 ? 'var(--nox-red)' : 'var(--text-muted)';
    const trend = item.trend === 'UP' ? '↑' : item.trend === 'DOWN' ? '↓' : '→';
    const card = document.createElement('div');
    card.className = 'macro-card';
    card.innerHTML = `
        <div class="name">${{item.name}}</div>
        <div class="price">${{item.price.toLocaleString('tr-TR')}}</div>
        <div class="change" style="color:${{color}}">${{trend}} 1G:${{chg1d}}% · 5G:${{chg5d}}%</div>
    `;
    macroGrid.appendChild(card);
}});

// Çakışma tablosu
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
        <td><a href="https://www.tradingview.com/chart/?symbol=BIST:${{item.ticker}}" target="_blank" class="tv-link"><b>${{item.ticker}}</b></a></td>
        <td><span class="score-badge" style="background:${{scoreColor}}20;color:${{scoreColor}}">${{item.score}}</span></td>
        <td><span class="rec-badge" style="background:${{recColor}}20;color:${{recColor}}">${{item.recommendation}}</span></td>
        <td style="font-size:0.8rem;color:var(--text-secondary)">${{details}}</td>
    `;
    tbody.appendChild(tr);
}});
</script>
</body>
</html>"""
    return html


def _markdown_to_html(text):
    """Basit markdown → HTML dönüşümü."""
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
            import re
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
