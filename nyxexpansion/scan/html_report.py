"""HTML report for nyxexpansion daily scan + Markowitz 4-stock portfolio."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd


def _note_for_row(row) -> str:
    """Short note describing the execution picture of a candidate."""
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


def _bucket_class(bucket: str) -> str:
    return {
        "clean": "b-clean", "mild": "b-mild",
        "elevated": "b-elev", "severe": "b-sev",
    }.get(str(bucket), "")


def _tag_class(tag: str) -> str:
    return {
        "clean_watch": "t-clean", "extended_watch": "t-ext",
        "special_handling": "t-spec",
    }.get(str(tag), "")


def render_html(
    scan_df: pd.DataFrame,
    portfolio: dict,
    target_date: pd.Timestamp,
    meta: dict,
) -> str:
    """Render the scan + portfolio HTML.

    Args:
        scan_df: scan output (one row per candidate), already filtered to target_date
                 and sorted descending by winR.
        portfolio: combinatorial_max_sharpe output.
        target_date: the scan's target date.
        meta: dict with keys `dataset_path`, `n_total`, `regime_dist` (optional),
              `severe_excluded` (int), `universe_size`.
    """
    n_total = len(scan_df)
    n_severe = int((scan_df.get("risk_bucket") == "severe").sum())

    # ── Candidates table ─────────────────────────────────────────────────
    rows = []
    for i, r in enumerate(scan_df.itertuples(index=False), 1):
        d = r._asdict() if hasattr(r, "_asdict") else dict(r._asdict())
        note = _note_for_row(d)
        bucket = str(d.get("risk_bucket", "—"))
        tag = str(d.get("exec_tag", "—"))
        winR = d.get("winner_R_pred")
        pct = d.get("score_pct")
        rscr = d.get("execution_risk_score")
        rows.append(
            f"<tr>"
            f"<td class='num'>{i}</td>"
            f"<td class='tk'>{html.escape(str(d.get('ticker', '')))}</td>"
            f"<td class='num'>{winR:.2f}</td>"
            f"<td class='num'>{pct:.2f}</td>"
            f"<td><span class='tag {_tag_class(tag)}'>{html.escape(tag)}</span></td>"
            f"<td><span class='bucket {_bucket_class(bucket)}'>{html.escape(bucket)}</span></td>"
            f"<td class='num'>{rscr:.1f}</td>"
            f"<td class='note'>{html.escape(note)}</td>"
            f"</tr>"
        )
    cand_rows = "\n".join(rows)

    # ── Portfolio block ──────────────────────────────────────────────────
    pf_rows = ""
    pf_section = ""
    if portfolio and portfolio.get("weights") and not portfolio.get("error"):
        per = portfolio.get("per_stock_stats", {})
        sel = sorted(portfolio["weights"].items(), key=lambda x: -x[1])
        for t, w in sel:
            stats = per.get(t, {})
            mean_pct = stats.get("mean_ann_pct", 0.0)
            vol_pct = stats.get("vol_ann_pct", 0.0)
            last_pct = stats.get("last_return_pct", 0.0)
            match = scan_df[scan_df["ticker"] == t]
            winR = match["winner_R_pred"].iloc[0] if not match.empty else 0.0
            bucket = match["risk_bucket"].iloc[0] if not match.empty else "—"
            pf_rows += (
                f"<tr>"
                f"<td class='tk'>{html.escape(t)}</td>"
                f"<td class='num'><b>{w*100:.1f}%</b></td>"
                f"<td class='num'>{winR:.2f}</td>"
                f"<td><span class='bucket {_bucket_class(bucket)}'>{html.escape(str(bucket))}</span></td>"
                f"<td class='num'>{mean_pct:+.1f}%</td>"
                f"<td class='num'>{vol_pct:.1f}%</td>"
                f"<td class='num'>{last_pct:+.2f}%</td>"
                f"</tr>"
            )
        wmin, wmax = portfolio.get("weight_bounds", [0.10, 0.50])
        pf_section = f"""
        <div class="box">
          <h2>💼 Markowitz 4'lü Portföy — Max Sharpe (combinatorial)</h2>
          <div class="kpis">
            <div>Sharpe: <b>{portfolio['sharpe']:.3f}</b></div>
            <div>Beklenen Getiri (ann.): <b>{portfolio['expected_return']:+.2f}%</b></div>
            <div>Risk σ (ann.): <b>{portfolio['expected_risk']:.2f}%</b></div>
            <div>Hisse: <b>{len(portfolio['weights'])}</b></div>
            <div>Lookback: <b>{portfolio.get('lookback_days', 60)}g</b></div>
            <div>Ağırlık aralığı: <b>[{wmin:.2f}, {wmax:.2f}]</b></div>
            <div>Kombinasyon: <b>{portfolio.get('combos_evaluated', 0):,}</b></div>
          </div>
          <table class="data">
            <thead><tr><th>Hisse</th><th>Ağırlık</th><th>winR</th><th>Bucket</th>
              <th>μ (60g ann.)</th><th>σ (60g ann.)</th><th>Son Gün</th></tr></thead>
            <tbody>{pf_rows}</tbody>
          </table>
          <div class="footer-note">
            <b>Aday havuzu:</b> winR sıralı top-{len(portfolio.get('universe_used', []))} (risk_bucket ≠ severe) ·
            Girdi: 60 günlük daily log-return, yıllıklandırılmış μ/Σ + Ledoit-Wolf shrinkage ·
            Optimizer: SLSQP, long-only, sum(w)=1.
          </div>
        </div>
        """
    else:
        err = portfolio.get("error") if portfolio else "Markowitz çalıştırılamadı"
        pf_section = f"""
        <div class="box warn">
          <h2>💼 Markowitz 4'lü Portföy — ÜRETİLMEDİ</h2>
          <div>Sebep: {html.escape(str(err))}</div>
        </div>
        """

    # ── Header block ─────────────────────────────────────────────────────
    universe_size = meta.get("universe_size", "?")
    dataset_path = meta.get("dataset_path", "?")
    regime_dist = meta.get("regime_dist", {})
    regime_str = ", ".join(f"{k}={v}" for k, v in regime_dist.items()) if regime_dist else "—"
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    warn_banner = """
    <div class="banner">
      ⚠️ <b>Daily candidate ranker — auto-entry DEĞİL.</b>
      Bu liste trigger'dan geçen adayları winR sırasıyla gösterir; pozisyon kararı için
      execution risk (bucket / very_extended / tight_room) ve canlı 17:30 dip/volume
      gözlemi ayrıca değerlendirilmelidir. Severe bucket hard-exclude edilmiştir.
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="utf-8">
<title>nyxexpansion v4C scan — {target_date.date()}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; background: #fafafa; color: #222; max-width: 1180px; }}
h1 {{ margin: 0 0 4px; }}
.meta {{ color: #666; margin-bottom: 16px; font-size: 0.92em; }}
.banner {{ background: #fff8e1; border-left: 3px solid #f9a825; padding: 10px 14px;
           margin: 12px 0 18px; font-size: 0.92em; color: #5f4a00; border-radius: 4px; }}
.box {{ background: white; padding: 16px 22px; border-radius: 8px; margin-bottom: 20px;
       box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.box.warn {{ background: #ffebee; }}
h2 {{ margin-top: 0; font-size: 1.12em; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 22px; font-size: 0.93em; margin-bottom: 14px; color: #333; }}
table.data {{ width: 100%; border-collapse: collapse; font-size: 0.89em; }}
table.data th {{ text-align: left; background: #f0f0f0; padding: 7px 9px; font-weight: 600; }}
table.data td {{ padding: 5px 9px; border-bottom: 1px solid #eee; }}
table.data tr:hover {{ background: #f7f7f7; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.tk {{ font-weight: 600; }}
td.note {{ color: #666; font-size: 0.88em; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.82em; }}
.t-clean {{ background: #e8f5e9; color: #1b5e20; }}
.t-ext {{ background: #fff3e0; color: #e65100; }}
.t-spec {{ background: #f3e5f5; color: #6a1b9a; }}
.bucket {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.82em; font-weight: 500; }}
.b-clean {{ background: #e3f2fd; color: #0d47a1; }}
.b-mild {{ background: #f0f4c3; color: #33691e; }}
.b-elev {{ background: #ffe0b2; color: #e65100; }}
.b-sev {{ background: #ffcdd2; color: #b71c1c; }}
.footer-note {{ color: #777; font-size: 0.85em; margin-top: 10px; }}
.footer {{ color: #888; font-size: 0.82em; margin-top: 16px; }}
</style></head><body>
<h1>nyxexpansion v4C scan · {target_date.date()}</h1>
<div class="meta">
  Tarama tarihi: <b>{target_date.date()}</b> ·
  Evren: <b>{universe_size}</b> ticker ·
  Trigger: <b>{n_total}</b> sinyal ·
  Rejim: <b>{regime_str}</b> ·
  Severe exclude: <b>{n_severe}</b> ·
  Dataset: <code>{html.escape(str(dataset_path))}</code>
</div>
{warn_banner}
{pf_section}
<div class="box">
  <h2>📋 Tüm Çıkan Hisseler ({n_total})</h2>
  <table class="data">
    <thead><tr>
      <th>#</th><th>Ticker</th><th>winR</th><th>pct</th>
      <th>exec_tag</th><th>risk_bucket</th><th>rscr</th><th>Not</th>
    </tr></thead>
    <tbody>
{cand_rows}
    </tbody>
  </table>
</div>
<div class="footer">nyxexpansion daily scan · {now_str} · v4C · dataset={html.escape(str(dataset_path))}</div>
</body></html>"""
