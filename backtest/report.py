"""
NOX Backtest — Report
Backtest sonuçlarını HTML rapor olarak oluştur.
"""
import os
import pandas as pd
from datetime import datetime
from backtest.config import FILTER_TESTS, REGIMES, REGIME_GROUPS, SIGNAL_GROUPS
from backtest.engine import apply_filter
from backtest.metrics import (
    build_matrix, matrix_to_dataframe, calc_metrics,
    signal_breakdown, regime_breakdown,
)


def _color_wr(wr):
    if wr >= 60: return '#22c55e'
    if wr >= 50: return '#eab308'
    if wr >= 40: return '#f97316'
    return '#ef4444'

def _color_pf(pf):
    if pf >= 2.0: return '#22c55e'
    if pf >= 1.5: return '#84cc16'
    if pf >= 1.0: return '#eab308'
    return '#ef4444'

def _color_exp(exp):
    if exp >= 1.0: return '#22c55e'
    if exp >= 0.5: return '#84cc16'
    if exp >= 0: return '#eab308'
    return '#ef4444'


def generate_backtest_html(all_trades, mode='trend', output_dir='.', elite=False):
    """Ana backtest HTML raporu oluştur."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    matrix, best_filter = build_matrix(all_trades)

    # Signal breakdown (baseline)
    sig_bk = signal_breakdown(all_trades, 'baseline')
    sig_bk_balanced = signal_breakdown(all_trades, 'balanced')

    # Regime breakdown
    reg_bk = regime_breakdown(all_trades, 'balanced')

    # Matrix DataFrame
    df_matrix = matrix_to_dataframe(matrix, 'all_signals')

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX Backtest — BIST {mode.upper()}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a0a; color: #e5e5e5; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; padding: 20px; }}
.header {{ text-align: center; padding: 30px 0; border-bottom: 1px solid #262626; margin-bottom: 30px; }}
.header h1 {{ font-size: 28px; color: #f5f5f5; letter-spacing: 4px; }}
.header .sub {{ color: #737373; font-size: 12px; margin-top: 8px; }}
.best-badge {{ display: inline-block; background: #22c55e20; border: 1px solid #22c55e; color: #22c55e; padding: 4px 12px; border-radius: 4px; font-size: 11px; margin-top: 10px; }}
.section {{ margin-bottom: 40px; }}
.section h2 {{ font-size: 16px; color: #a3a3a3; margin-bottom: 15px; border-left: 3px solid #525252; padding-left: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: #0f0f0f; border-radius: 8px; overflow: hidden; }}
th {{ background: #1a1a1a; color: #a3a3a3; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; padding: 10px 8px; text-align: center; border-bottom: 1px solid #262626; }}
td {{ padding: 8px; text-align: center; border-bottom: 1px solid #1a1a1a; font-size: 12px; }}
tr:hover {{ background: #1a1a1a; }}
tr.best {{ background: #22c55e10; }}
.metric {{ font-weight: bold; }}
.subtitle {{ color: #737373; font-size: 11px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
.card {{ background: #111; border: 1px solid #262626; border-radius: 8px; padding: 16px; }}
.card h3 {{ font-size: 13px; color: #d4d4d4; margin-bottom: 10px; }}
.stat {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1a1a; }}
.stat-label {{ color: #737373; }}
.stat-value {{ font-weight: bold; }}
.regime-tag {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold; }}
.bull {{ background: #22c55e20; color: #22c55e; }}
.bear {{ background: #ef444420; color: #ef4444; }}
.sideways {{ background: #eab30820; color: #eab308; }}
.footer {{ text-align: center; padding: 20px; color: #404040; font-size: 11px; margin-top: 40px; border-top: 1px solid #1a1a1a; }}
</style>
</head>
<body>

<div class="header">
  <h1>N O X — B A C K T E S T</h1>
  <div class="sub">BIST · {mode.upper()} · 10 Yıl · {now}</div>
  <div class="sub">{len(all_trades):,} trade analiz edildi</div>
  {"<div class='best-badge'>🏆 En İyi Filtre: " + best_filter + " (" + FILTER_TESTS.get(best_filter, {}).get('desc', '') + ")</div>" if best_filter else ""}
</div>

<!-- SECTION 1: FILTER × REGIME MATRIX -->
<div class="section">
  <h2>Filtre × Rejim Matrisi</h2>
  <div style="overflow-x: auto;">
  <table>
    <tr>
      <th rowspan="2">Filtre</th>
      <th colspan="4">TÜM DÖNEM</th>
      <th colspan="4">🐂 BOĞA</th>
      <th colspan="4">🐻 AYI</th>
      <th colspan="4">➡️ YATAY</th>
    </tr>
    <tr>
      <th>N</th><th>WR%</th><th>PF</th><th>Exp</th>
      <th>N</th><th>WR%</th><th>PF</th><th>Exp</th>
      <th>N</th><th>WR%</th><th>PF</th><th>Exp</th>
      <th>N</th><th>WR%</th><th>PF</th><th>Exp</th>
    </tr>
"""

    for _, row in df_matrix.iterrows():
        is_best = row['filter'] == best_filter
        cls = ' class="best"' if is_best else ''
        badge = ' 🏆' if is_best else ''
        html += f"    <tr{cls}>\n"
        html += f"      <td style='text-align:left'>{row['desc']}{badge}</td>\n"
        for rg in ['all', 'bull', 'bear', 'sideways']:
            n = int(row[f'{rg}_n'])
            wr = row[f'{rg}_wr']
            pf = row[f'{rg}_pf']
            exp = row[f'{rg}_exp']
            html += f"      <td>{n}</td>"
            html += f"<td class='metric' style='color:{_color_wr(wr)}'>{wr:.1f}</td>"
            html += f"<td class='metric' style='color:{_color_pf(pf)}'>{pf:.2f}</td>"
            html += f"<td class='metric' style='color:{_color_exp(exp)}'>{exp:.2f}</td>\n"
        html += "    </tr>\n"

    html += """  </table>
  </div>
</div>

<!-- SECTION 2: SIGNAL BREAKDOWN -->
<div class="section">
  <h2>Sinyal Tipi Performansı (Filtre: Baseline)</h2>
  <table>
    <tr>
      <th>Sinyal</th><th>N</th><th>WR%</th><th>Avg PnL%</th><th>PF</th>
      <th>Avg Win%</th><th>Avg Loss%</th><th>Avg R:R</th><th>Avg Hold</th>
      <th>TP%</th><th>Stop%</th><th>Timeout%</th><th>Sharpe</th><th>Exp</th>
    </tr>
"""

    for sig, m in sorted(sig_bk.items(), key=lambda x: -x[1].get('expectancy', 0)):
        if m['n_trades'] == 0:
            continue
        html += f"""    <tr>
      <td style='text-align:left; font-weight:bold'>{sig}</td>
      <td>{m['n_trades']}</td>
      <td class='metric' style='color:{_color_wr(m["win_rate"])}'>{m['win_rate']:.1f}</td>
      <td class='metric'>{m['avg_pnl']:.2f}</td>
      <td class='metric' style='color:{_color_pf(m["profit_factor"])}'>{m['profit_factor']:.2f}</td>
      <td style='color:#22c55e'>{m['avg_win']:.2f}</td>
      <td style='color:#ef4444'>{m['avg_loss']:.2f}</td>
      <td>{m['avg_rr_realized']:.2f}</td>
      <td>{m['avg_hold_days']:.1f}d</td>
      <td>{m['tp_pct']:.0f}</td>
      <td>{m['stop_pct']:.0f}</td>
      <td>{m['timeout_pct']:.0f}</td>
      <td>{m['sharpe']:.2f}</td>
      <td class='metric' style='color:{_color_exp(m["expectancy"])}'>{m['expectancy']:.2f}</td>
    </tr>\n"""

    html += """  </table>
</div>

<!-- SECTION 3: REGIME PERIODS -->
<div class="section">
  <h2>Rejim Dönemi Performansı (Filtre: Balanced)</h2>
  <table>
    <tr>
      <th>Dönem</th><th>Tip</th><th>Tarih</th><th>N</th><th>WR%</th>
      <th>Avg PnL%</th><th>PF</th><th>Total PnL%</th><th>Exp</th>
    </tr>
"""

    for rname, m in sorted(reg_bk.items(), key=lambda x: x[1].get('period', '')):
        rtype = m.get('type', '')
        type_cls = rtype if rtype in ('bull', 'bear', 'sideways') else ''
        html += f"""    <tr>
      <td style='text-align:left'>{m['label']}</td>
      <td><span class='regime-tag {type_cls}'>{rtype.upper()}</span></td>
      <td class='subtitle'>{m['period']}</td>
      <td>{m['n_trades']}</td>
      <td class='metric' style='color:{_color_wr(m["win_rate"])}'>{m['win_rate']:.1f}</td>
      <td class='metric'>{m['avg_pnl']:.2f}</td>
      <td class='metric' style='color:{_color_pf(m["profit_factor"])}'>{m['profit_factor']:.2f}</td>
      <td class='metric'>{m['total_pnl']:.1f}</td>
      <td class='metric' style='color:{_color_exp(m["expectancy"])}'>{m['expectancy']:.2f}</td>
    </tr>\n"""

    html += f"""  </table>
</div>

<!-- SECTION 4: SUMMARY CARDS -->
<div class="section">
  <h2>Özet İstatistikler</h2>
  <div class="grid">
"""

    # Overall stats card
    overall = calc_metrics(all_trades)
    html += f"""    <div class="card">
      <h3>📊 Genel (Baseline)</h3>
      <div class="stat"><span class="stat-label">Toplam Trade</span><span class="stat-value">{overall['n_trades']:,}</span></div>
      <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value" style="color:{_color_wr(overall['win_rate'])}">{overall['win_rate']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Profit Factor</span><span class="stat-value" style="color:{_color_pf(overall['profit_factor'])}">{overall['profit_factor']:.2f}</span></div>
      <div class="stat"><span class="stat-label">Expectancy</span><span class="stat-value" style="color:{_color_exp(overall['expectancy'])}">{overall['expectancy']:.2f}%</span></div>
      <div class="stat"><span class="stat-label">Sharpe</span><span class="stat-value">{overall['sharpe']:.2f}</span></div>
      <div class="stat"><span class="stat-label">Avg Hold</span><span class="stat-value">{overall['avg_hold_days']:.1f} gün</span></div>
      <div class="stat"><span class="stat-label">Max Win</span><span class="stat-value" style="color:#22c55e">{overall['max_win']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Max Loss</span><span class="stat-value" style="color:#ef4444">{overall['max_loss']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Avg MAE</span><span class="stat-value">{overall['avg_mae']:.1f}%</span></div>
    </div>
"""

    # Best filter card
    if best_filter:
        bf_trades = [t for t in all_trades if apply_filter(t, FILTER_TESTS[best_filter])]
        bf = calc_metrics(bf_trades)
        html += f"""    <div class="card" style="border-color:#22c55e40">
      <h3>🏆 En İyi: {best_filter} ({FILTER_TESTS[best_filter]['desc']})</h3>
      <div class="stat"><span class="stat-label">Toplam Trade</span><span class="stat-value">{bf['n_trades']:,}</span></div>
      <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value" style="color:{_color_wr(bf['win_rate'])}">{bf['win_rate']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Profit Factor</span><span class="stat-value" style="color:{_color_pf(bf['profit_factor'])}">{bf['profit_factor']:.2f}</span></div>
      <div class="stat"><span class="stat-label">Expectancy</span><span class="stat-value" style="color:{_color_exp(bf['expectancy'])}">{bf['expectancy']:.2f}%</span></div>
      <div class="stat"><span class="stat-label">Sharpe</span><span class="stat-value">{bf['sharpe']:.2f}</span></div>
      <div class="stat"><span class="stat-label">Avg PnL</span><span class="stat-value">{bf['avg_pnl']:.2f}%</span></div>
    </div>
"""

    # Exit distribution card
    html += f"""    <div class="card">
      <h3>🚪 Çıkış Dağılımı (Baseline)</h3>
      <div class="stat"><span class="stat-label">TP Hit</span><span class="stat-value" style="color:#22c55e">{overall['tp_pct']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Stop Hit</span><span class="stat-value" style="color:#ef4444">{overall['stop_pct']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Timeout</span><span class="stat-value" style="color:#eab308">{overall['timeout_pct']:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Avg Win / Avg Loss</span><span class="stat-value">{overall['avg_rr_realized']:.2f}x</span></div>
    </div>
"""

    html += """  </div>
</div>

<div class="footer">
  NOX project / backtest · BIST · %s · Powered by NOX Engine
</div>

</body>
</html>""" % now

    # Save
    mode_suffix = f"{mode}_momentum" if elite else mode
    fname = f"nox_backtest_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fpath = os.path.join(output_dir, fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"📄 Backtest rapor: {fpath}")
    return fpath


def save_trades_csv(all_trades, output_dir='.', elite=False):
    """Tüm trade'leri CSV'ye kaydet."""
    if not all_trades:
        return None
    df = pd.DataFrame(all_trades)
    cols = ['ticker', 'signal', 'trade_mode', 'date', 'regime_period', 'regime_type',
            'entry_price', 'stop', 'tp', 'rr', 'quality', 'rs_score',
            'overext_score', 'cmf', 'atr_pctile', 'dist_ema20',
            'market_regime', 'weekly_st_up', 'pos_size',
            'exit_price', 'exit_date', 'exit_reason',
            'pnl_pct', 'hold_days', 'mae', 'mfe']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values('date')
    mode_suffix = "momentum" if elite else "core"
    fname = f"nox_backtest_trades_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    fpath = os.path.join(output_dir, fname)
    df.to_csv(fpath, index=False)
    print(f"📊 Trade CSV: {fpath}")
    return fpath
