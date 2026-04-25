"""HTML report for nyxexpansion daily scan + Markowitz 4-stock portfolio.

Visual aesthetic borrowed from the briefing report (NOX dark theme, aurora bg,
glassmorphism cards). Tickers link to TradingView (BIST: prefix).
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from core.reports import _NOX_CSS

_TV_BASE = "https://www.tradingview.com/chart/?symbol=BIST:"


def _fnum(v, fmt: str = ".2f", default: str = "—") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return default


def _ticker_link(ticker: str) -> str:
    t = html.escape(str(ticker))
    return (f'<a class="tv-link" href="{_TV_BASE}{t}" target="_blank" '
            f'rel="noopener" title="{t} — TradingView">{t}</a>')


def _bucket_chip(bucket: str) -> str:
    b = str(bucket)
    color_map = {
        "clean":    ("rgba(122,158,122,0.18)", "var(--nox-green)"),
        "mild":     ("rgba(122,143,165,0.18)", "var(--nox-blue)"),
        "elevated": ("rgba(184,149,110,0.18)", "var(--nox-copper)"),
        "severe":   ("rgba(158,90,90,0.20)",   "var(--nox-red)"),
    }
    bg, fg = color_map.get(b, ("rgba(85,82,80,0.18)", "var(--text-muted)"))
    return (f'<span class="reg-badge" style="background:{bg};color:{fg}">'
            f'{html.escape(b)}</span>')


def _tag_chip(tag: str) -> str:
    t = str(tag)
    color_map = {
        "clean_watch":      ("rgba(122,158,122,0.14)", "var(--nox-green)"),
        "extended_watch":   ("rgba(184,149,110,0.16)", "var(--nox-copper)"),
        "special_handling": ("rgba(138,122,158,0.16)", "var(--nox-purple)"),
    }
    bg, fg = color_map.get(t, ("rgba(85,82,80,0.14)", "var(--text-muted)"))
    return (f'<span class="kc-badge" style="background:{bg};color:{fg}">'
            f'{html.escape(t)}</span>')


def _retention_chip(pass_flag: bool, note: str) -> str:
    cls = "kc-hi" if pass_flag else "kc-lo"
    label = "PASS" if pass_flag else "DROP"
    return (f'<span class="kc-badge {cls}" title="{html.escape(note)}">'
            f'{label}</span>')


def _note_for_row(row) -> str:
    parts = []
    stretch = str(row.get("stretch_rating", ""))
    ext = str(row.get("extension_rating", ""))
    mom = str(row.get("momentum_intensity", ""))
    room = str(row.get("upside_room", ""))
    if ext == "very_extended":
        parts.append("very_extended")
    if stretch in ("high", "very_high"):
        parts.append(f"stretch={stretch}")
    if room == "tight":
        parts.append("tight_room")
    elif room == "above_52w":
        parts.append("above_52w")
    if mom == "strong":
        parts.append("strong_momo")
    if row.get("risk_bucket") == "severe":
        parts.append("SEVERE→exclude")
    return ", ".join(parts) if parts else "—"


def _retention_cell(d: dict) -> tuple[str, str, str]:
    rank = d.get("rank_1700_surrogate")
    score = d.get("score_1700_surrogate")
    note = d.get("timing_clean_note") or "—"
    rank_str = "—" if rank is None or pd.isna(rank) else str(int(rank))
    score_str = "—" if score is None or pd.isna(score) else f"{float(score):.2f}"
    return rank_str, score_str, str(note)


def _row_html(i: int, d: dict) -> str:
    note = _note_for_row(d)
    bucket = d.get("risk_bucket", "—")
    tag = d.get("exec_tag", "—")
    rank_str, score_str, ret_note = _retention_cell(d)
    ret_pass = bool(d.get("retention_pass", False))
    return (
        f"<tr>"
        f"<td class='num'>{i}</td>"
        f"<td>{_ticker_link(d.get('ticker', ''))}</td>"
        f"<td class='num'>{_fnum(d.get('winner_R_pred'))}</td>"
        f"<td class='num'>{_fnum(d.get('score_pct'))}</td>"
        f"<td class='num'>{score_str}</td>"
        f"<td class='num'>{rank_str}</td>"
        f"<td>{_tag_chip(tag)}</td>"
        f"<td>{_bucket_chip(bucket)}</td>"
        f"<td class='num'>{_fnum(d.get('execution_risk_score'), '.1f')}</td>"
        f"<td>{_retention_chip(ret_pass, ret_note)}</td>"
        f"<td class='detail-cell'>{html.escape(note)}</td>"
        f"</tr>"
    )


def _candidates_table(rows_html: list[str], table_id: str) -> str:
    body = "\n".join(rows_html) if rows_html else (
        "<tr><td colspan='11' style='text-align:center;color:var(--text-muted);"
        "padding:24px'>—</td></tr>"
    )
    return f"""
  <div class="nox-table-wrap">
    <table id="{table_id}">
      <thead><tr>
        <th>#</th><th>Ticker</th><th>winR</th><th>pct</th>
        <th>winR_1700</th><th>rank_1700</th>
        <th>exec_tag</th><th>risk_bucket</th><th>rscr</th>
        <th>retention</th><th>Not</th>
      </tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
  </div>
"""


def _portfolio_section(portfolio: dict, scan_df: pd.DataFrame) -> str:
    if not portfolio or not portfolio.get("weights") or portfolio.get("error"):
        err = portfolio.get("error") if portfolio else "Markowitz çalıştırılamadı"
        return f"""
        <section class="nox-card warn-card">
          <h2>💼 Markowitz 4'lü Portföy — ÜRETİLMEDİ</h2>
          <div class="card-body">Sebep: {html.escape(str(err))}</div>
        </section>
        """

    per = portfolio.get("per_stock_stats", {})
    sel = sorted(portfolio["weights"].items(), key=lambda x: -x[1])
    pf_rows = []
    for t, w in sel:
        stats = per.get(t, {})
        match = scan_df[scan_df["ticker"] == t]
        winR = match["winner_R_pred"].iloc[0] if not match.empty else None
        bucket = match["risk_bucket"].iloc[0] if not match.empty else "—"
        last_pct = stats.get("last_return_pct", 0.0)
        last_cls = "rs-pos" if last_pct >= 0 else "rs-neg"
        pf_rows.append(
            f"<tr>"
            f"<td>{_ticker_link(t)}</td>"
            f"<td class='num'><b>{w*100:.1f}%</b></td>"
            f"<td class='num'>{_fnum(winR)}</td>"
            f"<td>{_bucket_chip(bucket)}</td>"
            f"<td class='num'>{_fnum(stats.get('mean_ann_pct'), '+.1f')}%</td>"
            f"<td class='num'>{_fnum(stats.get('vol_ann_pct'), '.1f')}%</td>"
            f"<td class='num {last_cls}'>{_fnum(last_pct, '+.2f')}%</td>"
            f"</tr>"
        )

    wmin, wmax = portfolio.get("weight_bounds", [0.10, 0.50])
    kpis = (
        ("Sharpe",     f"{portfolio['sharpe']:.3f}"),
        ("μ (ann.)",   f"{portfolio['expected_return']:+.2f}%"),
        ("σ (ann.)",   f"{portfolio['expected_risk']:.2f}%"),
        ("Hisse",      f"{len(portfolio['weights'])}"),
        ("Lookback",   f"{portfolio.get('lookback_days', 60)}g"),
        ("w-aralığı",  f"[{wmin:.2f}, {wmax:.2f}]"),
        ("Combos",     f"{portfolio.get('combos_evaluated', 0):,}"),
    )
    kpi_html = "".join(
        f'<div class="kpi"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(v)}</span></div>'
        for k, v in kpis
    )

    n_universe = len(portfolio.get('universe_used', []))
    return f"""
    <section class="nox-card pf-card">
      <h2>💼 Markowitz 4'lü Portföy <span class="sub">— Max Sharpe (combinatorial)</span></h2>
      <div class="kpi-strip">{kpi_html}</div>
      <div class="nox-table-wrap">
        <table>
          <thead><tr>
            <th>Hisse</th><th>Ağırlık</th><th>winR</th><th>Bucket</th>
            <th>μ (60g ann.)</th><th>σ (60g ann.)</th><th>Son Gün</th>
          </tr></thead>
          <tbody>{''.join(pf_rows)}</tbody>
        </table>
      </div>
      <div class="card-footer">
        <b>Aday havuzu:</b> winR sıralı top-{n_universe} (risk_bucket ≠ severe) ·
        60g daily log-return, yıllıklandırılmış μ/Σ + Ledoit-Wolf shrinkage ·
        SLSQP, long-only, sum(w)=1.
      </div>
    </section>
    """


def _stat_chip(label: str, value, color_var: str = "var(--text-secondary)") -> str:
    return (
        f'<div class="nox-stat">'
        f'<span class="dot" style="background:{color_var}"></span>'
        f'<span>{html.escape(label)}</span>'
        f'<span class="cnt">{html.escape(str(value))}</span>'
        f'</div>'
    )


def render_html(
    scan_df: pd.DataFrame,
    portfolio: dict,
    target_date: pd.Timestamp,
    meta: dict,
) -> str:
    n_total = len(scan_df)
    n_severe = int((scan_df.get("risk_bucket") == "severe").sum())

    retention = meta.get("retention") or {}
    retention_enabled = bool(retention.get("enabled", False))

    if retention_enabled:
        tradeable_df = scan_df[scan_df.get("retention_pass") == True].copy()
        watchlist_df = scan_df[scan_df.get("retention_pass") != True].copy()
    else:
        tradeable_df = scan_df.iloc[0:0].copy()
        watchlist_df = scan_df.copy()

    tradeable_rows = [
        _row_html(i, dict(zip(scan_df.columns, r)))
        for i, r in enumerate(tradeable_df.itertuples(index=False, name=None), 1)
    ]
    watchlist_rows = [
        _row_html(i, dict(zip(scan_df.columns, r)))
        for i, r in enumerate(watchlist_df.itertuples(index=False, name=None), 1)
    ]

    universe_size = meta.get("universe_size", "?")
    dataset_path = meta.get("dataset_path", "?")
    regime_dist = meta.get("regime_dist", {}) or {}
    regime_str = ", ".join(f"{k}={v}" for k, v in regime_dist.items()) if regime_dist else "—"
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Stats strip ──
    stats_html = (
        _stat_chip("Evren", universe_size, "var(--nox-blue)")
        + _stat_chip("Trigger", n_total, "var(--nox-cyan)")
        + _stat_chip("Severe", n_severe, "var(--nox-red)")
        + _stat_chip("Rejim", regime_str, "var(--nox-copper)")
    )
    if retention_enabled:
        stats_html += (
            _stat_chip("PASS", retention.get("n_pass", 0), "var(--nox-green)")
            + _stat_chip("DROP", retention.get("n_drop", 0), "var(--nox-orange)")
        )

    # ── Warning banner (always-on) ──
    warn_banner = """
    <section class="nox-card banner warn-banner">
      <span class="banner-icon">⚠️</span>
      <div>
        <b>Daily candidate ranker — auto-entry DEĞİL.</b>
        Bu liste trigger'dan geçen adayları winR sırasıyla gösterir; pozisyon kararı için
        execution risk (bucket / very_extended / tight_room) ve canlı 17:30 dip/volume
        gözlemi ayrıca değerlendirilmelidir. Severe bucket hard-exclude edilmiştir.
      </div>
    </section>
    """

    # ── Retention banner ──
    if retention_enabled:
        rank_t = retention.get("rank_threshold", 10)
        n_pass = retention.get("n_pass", 0)
        n_drop = retention.get("n_drop", 0)
        n_unscored = retention.get("n_unscored", 0)
        notes = retention.get("notes", {}) or {}
        notes_str = ", ".join(f"{k}={v}" for k, v in notes.items()) or "—"
        sources = retention.get("source_breakdown", {}) or {}
        if sources:
            src_str = " · ".join(f"{k}={v}" for k, v in sorted(sources.items()))
            src_line = (
                f'<div class="banner-sub">Bugün veri kaynakları: '
                f'<code>{html.escape(src_str)}</code></div>'
            )
        else:
            src_line = ""
        retention_banner = f"""
        <section class="nox-card banner ret-banner">
          <span class="banner-icon">🛡</span>
          <div>
            <b>Timing-clean retention filter (17:00 TR top-{rank_t})</b> —
            PASS=<b>{n_pass}</b> · DROP=<b>{n_drop}</b> · unscored=<b>{n_unscored}</b>.
            Tradeable list = retention_pass=True; watchlist'te kalanlar gözlem amaçlı,
            17:30 proxy'de tradeable kabul edilmez.
            <div class="banner-sub">Notes: <code>{html.escape(notes_str)}</code></div>
            {src_line}
          </div>
        </section>
        """
    else:
        retention_banner = """
        <section class="nox-card banner ret-banner-off">
          <span class="banner-icon">⚠️</span>
          <div>
            <b>Timing-clean retention stage SKIPPED.</b> Hiçbir aday için 17:00
            truncated re-rank yapılmadı; aşağıdaki tüm satırlar Watchlist olarak
            işaretlendi. Tradeable list bu raporda BOŞ.
          </div>
        </section>
        """

    pf_section = _portfolio_section(portfolio, scan_df)

    return f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nyxexpansion v4C scan — {target_date.date()}</title>
<style>{_NOX_CSS}

/* ── scan-specific overrides ── */
.nox-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 18px;
  backdrop-filter: blur(8px);
}}
.nox-card h2 {{
  font-family: var(--font-display);
  font-size: 0.95rem; font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--text-primary);
  margin-bottom: 12px;
}}
.nox-card h2 .sub {{
  font-weight: 400; color: var(--text-muted); font-size: 0.85em;
  margin-left: 4px;
}}
.nox-card .card-body {{ font-size: 0.85rem; color: var(--text-secondary); }}
.nox-card .card-footer {{
  margin-top: 10px;
  font-size: 0.72rem; color: var(--text-muted);
  font-family: var(--font-mono);
  border-top: 1px solid var(--border-subtle);
  padding-top: 10px;
}}

/* Banners */
.banner {{
  display: flex; gap: 12px; align-items: flex-start;
  font-size: 0.82rem; line-height: 1.5;
  border-left: 3px solid var(--nox-gold);
  background: rgba(201,169,110,0.04);
}}
.banner b {{ color: var(--text-primary); }}
.banner .banner-icon {{ font-size: 1.2rem; flex-shrink: 0; line-height: 1.2; }}
.banner-sub {{ margin-top: 6px; font-size: 0.78rem; color: var(--text-muted); }}
.banner-sub code {{
  font-family: var(--font-mono); font-size: 0.78rem;
  background: var(--bg-elevated); padding: 1px 6px; border-radius: 4px;
  color: var(--text-secondary);
}}
.warn-banner {{ border-left-color: var(--nox-orange); background: rgba(168,135,106,0.05); }}
.ret-banner {{ border-left-color: var(--nox-green); background: rgba(122,158,122,0.05); }}
.ret-banner-off {{ border-left-color: var(--nox-red); background: rgba(158,90,90,0.05); }}
.warn-card {{ border-left: 3px solid var(--nox-red); }}

/* KPI strip in portfolio */
.kpi-strip {{
  display: flex; flex-wrap: wrap; gap: 18px;
  margin-bottom: 14px; padding: 10px 12px;
  background: var(--bg-elevated); border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
}}
.kpi {{ display: flex; flex-direction: column; gap: 2px; min-width: 75px; }}
.kpi .k {{ font-size: 0.62rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi .v {{ font-family: var(--font-mono); font-weight: 600;
  font-size: 0.88rem; color: var(--text-primary); }}

/* Section count in h2 */
.sec-count {{
  display: inline-block; margin-left: 6px;
  font-family: var(--font-mono); font-size: 0.78em;
  color: var(--nox-cyan); font-weight: 500;
}}

/* Make number cells slightly tighter */
td.num {{ text-align: right; font-variant-numeric: tabular-nums;
  font-family: var(--font-mono); }}
.detail-cell {{ white-space: normal; max-width: 320px; }}
</style></head>
<body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="nox-container">

  <header class="nox-header">
    <div class="nox-logo">
      nyxexpansion<span class="proj"></span>
      <span class="mode">v4C scan · {target_date.date()}</span>
    </div>
    <div class="nox-meta">
      Generated <b>{now_str}</b><br>
      Dataset: <code>{html.escape(str(dataset_path))}</code>
    </div>
  </header>

  <div class="nox-stats">{stats_html}</div>

  {warn_banner}
  {retention_banner}
  {pf_section}

  <section class="nox-card">
    <h2>✅ Tradeable Candidates<span class="sec-count">{len(tradeable_df)}</span></h2>
    <div class="card-body">retention_pass=True · 17:30 proxy için kabul edilen liste</div>
    {_candidates_table(tradeable_rows, "tradeable")}
  </section>

  <section class="nox-card">
    <h2>👁 Watchlist Only<span class="sec-count">{len(watchlist_df)}</span></h2>
    <div class="card-body">ranker'da çıktı ama timing-clean retention'da elendi · gözlem amaçlı, tradeable DEĞİL</div>
    {_candidates_table(watchlist_rows, "watchlist")}
  </section>

  <div class="nox-status">
    nyxexpansion daily scan · <b>{now_str}</b> · v4C ·
    dataset=<code>{html.escape(str(dataset_path))}</code>
  </div>

</div>
</body></html>"""
