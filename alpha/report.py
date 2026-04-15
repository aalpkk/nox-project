"""
Alpha Pipeline — HTML Rapor Üreticisi
======================================
Self-contained HTML: equity curve, drawdown, heatmap, trade log.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


def _color_for_pct(val: float) -> str:
    """Yüzde değere göre renk."""
    if val > 0:
        intensity = min(1.0, abs(val) / 15)
        return f'rgba(122,158,122,{0.2 + intensity * 0.6})'
    elif val < 0:
        intensity = min(1.0, abs(val) / 15)
        return f'rgba(158,90,90,{0.2 + intensity * 0.6})'
    return 'transparent'


def _fmt(val, suffix='%'):
    """Sayı formatlama."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val:+.1f}{suffix}' if suffix == '%' else f'{val:.2f}'


def generate_alpha_report(backtest_result: dict, metrics: dict,
                          monthly_returns: pd.DataFrame,
                          trade_stats: dict,
                          output_dir: str = 'output') -> str:
    """HTML rapor üret.

    Returns: dosya yolu
    """
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    filepath = os.path.join(output_dir, f'alpha_pipeline_{today}.html')

    eq_curve = backtest_result.get('equity_curve', [])
    trades = backtest_result.get('trades', [])
    rebalances = backtest_result.get('rebalance_events', [])
    funnel = backtest_result.get('stage_funnel', [])

    # Equity curve JS data
    eq_dates = [e['date'].strftime('%Y-%m-%d') if hasattr(e['date'], 'strftime') else str(e['date']) for e in eq_curve]
    eq_values = [e['equity'] for e in eq_curve]
    bm_values = [e.get('benchmark') or 'null' for e in eq_curve]
    dd_values = [e.get('dd_pct', 0) for e in eq_curve]

    # Monthly heatmap HTML
    heatmap_html = _build_heatmap(monthly_returns)

    # Trade log HTML
    trade_html = _build_trade_table(trades)

    # Rebalance log HTML
    rebal_html = _build_rebalance_table(rebalances)

    # Funnel HTML
    funnel_html = _build_funnel(funnel)

    # Metrik kartları
    m = metrics
    ts = trade_stats

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYX Alpha Pipeline — BIST</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg: #060709; --card: #0d0d10; --elevated: #141417;
  --border: #1e1e23; --text: #e8e4dc; --dim: #8a8580; --muted: #555250;
  --gold: #c9a96e; --green: #7a9e7a; --red: #9e5a5a; --blue: #7a8fa5;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text);
        min-height:100vh; padding:20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:28px; color:var(--gold); margin-bottom:6px; }}
.subtitle {{ color:var(--dim); font-size:14px; margin-bottom:24px; }}

/* Cards */
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
          gap:12px; margin-bottom:28px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
         padding:16px; text-align:center; }}
.card .label {{ font-size:11px; color:var(--dim); text-transform:uppercase;
                letter-spacing:1px; margin-bottom:6px; }}
.card .value {{ font-size:22px; font-weight:700; }}
.card .value.pos {{ color:var(--green); }}
.card .value.neg {{ color:var(--red); }}
.card .value.neutral {{ color:var(--gold); }}

/* Chart */
.chart-box {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
              padding:20px; margin-bottom:20px; }}
.chart-box h3 {{ font-size:14px; color:var(--gold); margin-bottom:12px; }}
canvas {{ width:100%; height:280px; }}

/* Tables */
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ background:var(--elevated); color:var(--dim); font-weight:600; padding:8px 10px;
      text-align:left; border-bottom:1px solid var(--border); }}
td {{ padding:6px 10px; border-bottom:1px solid var(--border); color:var(--text); }}
tr:hover td {{ background:var(--elevated); }}
.pnl-pos {{ color:var(--green); font-weight:600; }}
.pnl-neg {{ color:var(--red); font-weight:600; }}

/* Heatmap */
.heatmap {{ display:grid; gap:2px; }}
.heatmap-cell {{ padding:6px 4px; text-align:center; font-size:11px;
                 font-family:'JetBrains Mono',monospace; border-radius:4px; }}
.heatmap-header {{ font-weight:700; color:var(--dim); }}

/* Funnel */
.funnel {{ display:flex; gap:8px; align-items:flex-end; margin:16px 0; }}
.funnel-bar {{ background:var(--gold); border-radius:4px 4px 0 0; min-width:80px;
               text-align:center; padding:4px; font-size:11px; color:var(--bg); font-weight:600; }}
.funnel-label {{ font-size:10px; color:var(--dim); text-align:center; margin-top:4px; }}

.section {{ margin-bottom:28px; }}
.section h2 {{ font-size:16px; color:var(--gold); margin-bottom:12px;
               padding-bottom:6px; border-bottom:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">

<h1>NYX Alpha Pipeline</h1>
<p class="subtitle">BIST 5-Aşamalı Yatırım Sistemi &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<!-- Summary Cards -->
<div class="cards">
  <div class="card">
    <div class="label">Toplam Getiri</div>
    <div class="value {'pos' if m['total_return']>0 else 'neg'}">{m['total_return']:+.1f}%</div>
  </div>
  <div class="card">
    <div class="label">XU100</div>
    <div class="value {'pos' if m['benchmark_total']>0 else 'neg'}">{m['benchmark_total']:+.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Alpha</div>
    <div class="value {'pos' if m['alpha']>0 else 'neg'}">{m['alpha']:+.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Sharpe</div>
    <div class="value neutral">{m['sharpe_ratio']:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Sortino</div>
    <div class="value neutral">{m['sortino_ratio']:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Max DD</div>
    <div class="value neg">{m['max_drawdown']:.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Info Ratio</div>
    <div class="value neutral">{m['information_ratio']:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value {'pos' if ts.get('win_rate',0)>50 else 'neg'}">{ts.get('win_rate',0):.0f}%</div>
  </div>
</div>

<!-- Equity Curve (Canvas) -->
<div class="chart-box">
  <h3>Equity Curve vs XU100</h3>
  <canvas id="eqChart"></canvas>
</div>

<!-- Drawdown -->
<div class="chart-box">
  <h3>Drawdown</h3>
  <canvas id="ddChart"></canvas>
</div>

<!-- Monthly Heatmap -->
<div class="section">
  <h2>Aylik Getiri Haritasi</h2>
  {heatmap_html}
</div>

<!-- Stage Funnel -->
<div class="section">
  <h2>Asama Hunisi (Ortalama)</h2>
  {funnel_html}
</div>

<!-- Trade Stats -->
<div class="section">
  <h2>Trade Istatistikleri</h2>
  <div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr));">
    <div class="card"><div class="label">Toplam Trade</div><div class="value neutral">{ts.get('n_trades',0)}</div></div>
    <div class="card"><div class="label">Ort. PnL</div><div class="value {'pos' if ts.get('avg_pnl',0)>0 else 'neg'}">{ts.get('avg_pnl',0):+.1f}%</div></div>
    <div class="card"><div class="label">Ort. Kazanc</div><div class="value pos">{ts.get('avg_win',0):+.1f}%</div></div>
    <div class="card"><div class="label">Ort. Kayip</div><div class="value neg">{ts.get('avg_loss',0):.1f}%</div></div>
    <div class="card"><div class="label">Profit Factor</div><div class="value neutral">{ts.get('profit_factor',0):.2f}</div></div>
    <div class="card"><div class="label">Ort. Tutma</div><div class="value neutral">{ts.get('avg_hold',0):.0f} gun</div></div>
  </div>
</div>

<!-- Rebalance Log -->
<div class="section">
  <h2>Rebalance Log</h2>
  {rebal_html}
</div>

<!-- Trade Log -->
<div class="section">
  <h2>Trade Log (Son 50)</h2>
  {trade_html}
</div>

<p style="color:var(--muted);font-size:11px;margin-top:40px;text-align:center;">
  NYX Alpha Pipeline &mdash; {m.get('years',0):.1f} yil, {m.get('n_trading_days',0)} gun, Beta: {m.get('beta',1):.2f}, Jensen Alpha: {m.get('jensens_alpha',0):+.1f}%
</p>

</div>

<script>
// Minimal canvas chart renderer
const dates = {eq_dates};
const equity = {eq_values};
const benchmark = [{','.join(str(v) for v in bm_values)}];
const drawdown = {dd_values};

function drawLine(ctx, data, color, w, h, yMin, yMax) {{
  if (!data.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i=0; i<data.length; i++) {{
    if (data[i] === null) continue;
    const x = (i / (data.length-1)) * w;
    const y = h - ((data[i] - yMin) / (yMax - yMin)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
}}

function renderChart(id, series, colors) {{
  const canvas = document.getElementById(id);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;

  let allVals = [];
  series.forEach(s => s.forEach(v => {{ if (v !== null && !isNaN(v)) allVals.push(v); }}));
  const yMin = Math.min(...allVals) * 0.98;
  const yMax = Math.max(...allVals) * 1.02;

  // Grid
  ctx.strokeStyle = '#1e1e23'; ctx.lineWidth = 0.5;
  for (let i=0; i<5; i++) {{
    const y = (i/4) * h;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
  }}

  series.forEach((s, i) => drawLine(ctx, s, colors[i], w, h, yMin, yMax));

  // Labels
  ctx.font = '10px JetBrains Mono';
  ctx.fillStyle = '#8a8580';
  ctx.fillText(yMax.toLocaleString('tr-TR', {{maximumFractionDigits:0}}), 4, 14);
  ctx.fillText(yMin.toLocaleString('tr-TR', {{maximumFractionDigits:0}}), 4, h - 4);
}}

renderChart('eqChart', [equity, benchmark], ['#c9a96e', '#555250']);
renderChart('ddChart', [drawdown], ['#9e5a5a']);
</script>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return filepath


def _build_heatmap(monthly_returns: pd.DataFrame) -> str:
    if monthly_returns.empty:
        return '<p style="color:var(--dim);">Yeterli veri yok</p>'

    months = list(monthly_returns.columns)
    rows_html = '<div class="heatmap" style="grid-template-columns: 60px ' + ' '.join(['1fr'] * len(months)) + ';">'

    # Header
    rows_html += '<div class="heatmap-cell heatmap-header">Yil</div>'
    for m in months:
        rows_html += f'<div class="heatmap-cell heatmap-header">{m}</div>'

    # Data rows
    for year in monthly_returns.index:
        rows_html += f'<div class="heatmap-cell heatmap-header">{year}</div>'
        for m in months:
            val = monthly_returns.loc[year, m] if m in monthly_returns.columns else np.nan
            if pd.isna(val):
                rows_html += '<div class="heatmap-cell" style="background:transparent;">—</div>'
            else:
                bg = _color_for_pct(val)
                rows_html += f'<div class="heatmap-cell" style="background:{bg};">{val:+.1f}</div>'

    rows_html += '</div>'
    return rows_html


def _build_trade_table(trades: list) -> str:
    if not trades:
        return '<p style="color:var(--dim);">Trade yok</p>'

    # Son 50 trade
    recent = sorted(trades, key=lambda t: t.exit_date, reverse=True)[:50]

    rows = ''
    for t in recent:
        pnl_cls = 'pnl-pos' if t.pnl_pct > 0 else 'pnl-neg'
        entry_str = t.entry_date.strftime('%m/%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:5]
        exit_str = t.exit_date.strftime('%m/%d') if hasattr(t.exit_date, 'strftime') else str(t.exit_date)[:5]
        rows += f'''<tr>
          <td>{t.ticker}</td>
          <td>{entry_str}</td><td>{exit_str}</td>
          <td>{t.entry_price:.2f}</td><td>{t.exit_price:.2f}</td>
          <td class="{pnl_cls}">{t.pnl_pct:+.1f}%</td>
          <td>{t.exit_reason}</td><td>{t.hold_days}g</td>
          <td>{t.weight*100:.0f}%</td>
        </tr>'''

    return f'''<table>
      <thead><tr><th>Hisse</th><th>Giris</th><th>Cikis</th><th>Giris₺</th><th>Cikis₺</th>
                 <th>PnL</th><th>Neden</th><th>Sure</th><th>Agirlik</th></tr></thead>
      <tbody>{rows}</tbody></table>'''


def _build_rebalance_table(rebalances: list) -> str:
    if not rebalances:
        return '<p style="color:var(--dim);">Rebalance yok</p>'

    rows = ''
    for r in rebalances:
        date_str = r.date.strftime('%Y-%m-%d') if hasattr(r.date, 'strftime') else str(r.date)
        new_str = ', '.join(r.new_tickers[:5])
        if len(r.new_tickers) > 5:
            new_str += f' +{len(r.new_tickers)-5}'
        rows += f'''<tr>
          <td>{date_str}</td>
          <td>{r.candidates_total}</td><td>{r.candidates_passed}</td><td>{r.final_stocks}</td>
          <td>{r.turnover*100:.0f}%</td><td>{r.sharpe:.2f}</td>
          <td style="font-size:10px;">{new_str}</td>
        </tr>'''

    return f'''<table>
      <thead><tr><th>Tarih</th><th>Aday</th><th>Gecen</th><th>Final</th>
                 <th>Turnover</th><th>Sharpe</th><th>Portfoy</th></tr></thead>
      <tbody>{rows}</tbody></table>'''


def generate_live_scan_report(candidates: list, portfolio: dict = None,
                              output_dir: str = 'output') -> str:
    """Live scan HTML raporu üret."""
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    filepath = os.path.join(output_dir, f'alpha_scan_{today}.html')

    # Aday tablosu
    rows = ''
    for i, c in enumerate(candidates):
        ml3g = f"{c['ml_3g']:.2f}" if c.get('ml_3g') else '—'
        stop_color = 'var(--green)' if c['stop_pct'] <= 7 else ('var(--gold)' if c['stop_pct'] <= 10 else 'var(--red)')
        rows += f'''<tr>
          <td style="font-weight:600;">{i+1}</td>
          <td style="font-weight:700;"><a href="https://www.tradingview.com/chart/?symbol=BIST:{c['ticker']}" target="_blank" style="color:var(--gold);text-decoration:none;">{c['ticker']}</a></td>
          <td>{c['ml_1g']:.2f}</td>
          <td>{ml3g}</td>
          <td style="font-weight:600;">{c['composite']:.1f}</td>
          <td>{c['adx']:.1f}</td>
          <td class="{'pnl-pos' if c['cmf']>0 else 'pnl-neg'}">{c['cmf']:+.3f}</td>
          <td>{c['rsi']:.1f}</td>
          <td style="font-weight:600;">{c['close']:.2f}</td>
          <td style="color:var(--red);font-weight:600;">{c['stop']:.2f}</td>
          <td style="color:{stop_color};font-weight:600;">{c['stop_pct']:.1f}%</td>
          <td style="color:var(--green);font-weight:600;">{c['trail_target']:.2f}</td>
        </tr>'''

    # Portföy tablosu
    portfolio_html = ''
    if portfolio and portfolio.get('n_stocks', 0) > 0:
        p_rows = ''
        for t, w in sorted(portfolio['weights'].items(), key=lambda x: -x[1]):
            ticker = t.replace('.IS', '')
            bar_w = max(4, w * 500)
            p_rows += f'''<tr>
              <td style="font-weight:700;"><a href="https://www.tradingview.com/chart/?symbol=BIST:{ticker}" target="_blank" style="color:var(--gold);text-decoration:none;">{ticker}</a></td>
              <td>{w*100:.1f}%</td>
              <td><div style="background:var(--gold);height:16px;width:{bar_w}px;border-radius:3px;opacity:0.7;"></div></td>
            </tr>'''
        portfolio_html = f'''
        <div class="section">
          <h2>Onerilen Portfoy</h2>
          <div class="cards" style="grid-template-columns:repeat(3,1fr);margin-bottom:16px;">
            <div class="card"><div class="label">Sharpe</div><div class="value neutral">{portfolio.get('sharpe_ratio',0):.2f}</div></div>
            <div class="card"><div class="label">Beklenen Getiri</div><div class="value pos">{portfolio.get('expected_return',0):+.1f}%</div></div>
            <div class="card"><div class="label">Beklenen Risk</div><div class="value neg">{portfolio.get('expected_risk',0):.1f}%</div></div>
          </div>
          <table><thead><tr><th>Hisse</th><th>Agirlik</th><th></th></tr></thead>
          <tbody>{p_rows}</tbody></table>
        </div>'''

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYX Alpha Scan — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg: #060709; --card: #0d0d10; --elevated: #141417;
  --border: #1e1e23; --text: #e8e4dc; --dim: #8a8580; --muted: #555250;
  --gold: #c9a96e; --green: #7a9e7a; --red: #9e5a5a; --blue: #7a8fa5;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text);
        min-height:100vh; padding:20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:28px; color:var(--gold); margin-bottom:6px; }}
.subtitle {{ color:var(--dim); font-size:14px; margin-bottom:24px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
          gap:12px; margin-bottom:28px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
         padding:16px; text-align:center; }}
.card .label {{ font-size:11px; color:var(--dim); text-transform:uppercase;
                letter-spacing:1px; margin-bottom:6px; }}
.card .value {{ font-size:22px; font-weight:700; }}
.card .value.pos {{ color:var(--green); }}
.card .value.neg {{ color:var(--red); }}
.card .value.neutral {{ color:var(--gold); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:var(--elevated); color:var(--dim); font-weight:600; padding:10px 12px;
      text-align:left; border-bottom:1px solid var(--border); position:sticky; top:0;
      cursor:pointer; user-select:none; }}
th:hover {{ color:var(--gold); }}
th.sort-asc::after {{ content:' ▲'; font-size:10px; }}
th.sort-desc::after {{ content:' ▼'; font-size:10px; }}
td {{ padding:8px 12px; border-bottom:1px solid var(--border); color:var(--text);
      font-family:'JetBrains Mono',monospace; font-size:12px; }}
tr:hover td {{ background:var(--elevated); }}
.pnl-pos {{ color:var(--green); font-weight:600; }}
.pnl-neg {{ color:var(--red); font-weight:600; }}
.section {{ margin-bottom:28px; }}
.section h2 {{ font-size:16px; color:var(--gold); margin-bottom:12px;
               padding-bottom:6px; border-bottom:1px solid var(--border); }}
.legend {{ display:flex; gap:20px; margin:16px 0; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:6px; font-size:12px; color:var(--dim); }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; }}
</style>
</head>
<body>
<div class="container">

<h1>NYX Alpha Scan</h1>
<p class="subtitle">Gunluk ML Tarama &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash; {len(candidates)} aday</p>

<div class="cards">
  <div class="card"><div class="label">Toplam Aday</div><div class="value neutral">{len(candidates)}</div></div>
  <div class="card"><div class="label">Ort. ML Skor</div><div class="value neutral">{np.mean([c['ml_1g'] for c in candidates]):.2f}</div></div>
  <div class="card"><div class="label">Ort. Stop%</div><div class="value neg">{np.mean([c['stop_pct'] for c in candidates]):.1f}%</div></div>
  <div class="card"><div class="label">Min Stop%</div><div class="value pos">{min(c['stop_pct'] for c in candidates):.1f}%</div></div>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--gold);"></div>ML 1g: 1 gunluk yukaridaysa</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--green);"></div>ML 3g: 3 gunluk swing</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--red);"></div>Stop: Entry - 2xATR</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--blue);"></div>Trail: Entry + 1.5xATR</div>
</div>

<div class="section">
  <h2>Aday Listesi</h2>
  <table>
    <thead><tr>
      <th>#</th><th>Hisse</th><th>ML 1g</th><th>ML 3g</th><th>Skor</th>
      <th>ADX</th><th>CMF</th><th>RSI</th><th>Fiyat</th>
      <th>Stop</th><th>Stop%</th><th>Trail</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

{portfolio_html}

<p style="color:var(--muted);font-size:11px;margin-top:40px;text-align:center;">
  NYX Alpha Pipeline &mdash; ML + Cok Asamali Cikis Stratejisi &mdash; Backtest: Sharpe 1.05, Alpha +131%, DD -19%
</p>

</div>
<script>
document.querySelectorAll('.section table').forEach(table => {{
  const thead = table.querySelector('thead');
  if(!thead) return;
  const ths = thead.querySelectorAll('th');
  ths.forEach((th, colIdx) => {{
    th.addEventListener('click', () => {{
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const cur = th.classList.contains('sort-asc') ? 'asc' : (th.classList.contains('sort-desc') ? 'desc' : 'none');
      ths.forEach(h => h.classList.remove('sort-asc','sort-desc'));
      const dir = cur === 'asc' ? 'desc' : 'asc';
      th.classList.add('sort-' + dir);
      rows.sort((a, b) => {{
        let av = a.cells[colIdx]?.textContent.replace('%','').replace('—','').trim() || '';
        let bv = b.cells[colIdx]?.textContent.replace('%','').replace('—','').trim() || '';
        const an = parseFloat(av), bn = parseFloat(bv);
        if(!isNaN(an) && !isNaN(bn)) return dir==='asc' ? an-bn : bn-an;
        return dir==='asc' ? av.localeCompare(bv,'tr') : bv.localeCompare(av,'tr');
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}});
</script>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    return filepath


def _build_funnel(funnel: list) -> str:
    if not funnel:
        return '<p style="color:var(--dim);">Veri yok</p>'

    # Ortalamaları al
    totals = [f[1] for f in funnel]
    passeds = [f[2] for f in funnel]
    finals = [f[3] for f in funnel]

    avg_total = np.mean(totals)
    avg_passed = np.mean(passeds)
    avg_final = np.mean(finals)

    max_val = max(avg_total, 1)

    return f'''<div class="funnel">
      <div>
        <div class="funnel-bar" style="height:{max(20, avg_total/max_val*150)}px;">{avg_total:.0f}</div>
        <div class="funnel-label">Momentum<br>(Asama 1-2)</div>
      </div>
      <div style="color:var(--dim);font-size:20px;align-self:center;">→</div>
      <div>
        <div class="funnel-bar" style="height:{max(20, avg_passed/max_val*150)}px;background:var(--green);">{avg_passed:.0f}</div>
        <div class="funnel-label">Onaylanan<br>(Asama 3)</div>
      </div>
      <div style="color:var(--dim);font-size:20px;align-self:center;">→</div>
      <div>
        <div class="funnel-bar" style="height:{max(20, avg_final/max_val*150)}px;background:var(--blue);">{avg_final:.0f}</div>
        <div class="funnel-label">Portfoy<br>(Asama 4-5)</div>
      </div>
    </div>'''
