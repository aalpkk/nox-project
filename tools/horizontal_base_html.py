"""Render horizontal_base_live_<asof>.csv to a single HTML — briefing aesthetic.

Pairs with tools/horizontal_base_scan_live.py. Reads:
    output/horizontal_base_live_<asof>.csv

Emits:
    output/horizontal_base_scan_<asof>.html  (+ _latest.html with --also-latest)

DESCRIPTIVE — V1.4.0/FEATURE 1.3.0 horizontal_base events with derived
tier (A/B/C) and tradeable flags. NO trade-edge claim — pre-registered
ML lines v1/v1_2 REJECTED. Trader uses tier+context to decide; rules-
based picker backtest gives in-sample reference numbers only.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.reports import _NOX_CSS

TIER_COLOR = {
    "A": "var(--nox-green)",
    "B": "var(--nox-cyan)",
    "C": "var(--nox-orange)",
}
TIER_LABEL = {
    "A": "Tier A",
    "B": "Tier B",
    "C": "Tier C",
}
TIER_HINT = {
    "A": "trigger·mid_body ∪ retest·deep_touch",
    "B": "trigger·strict_body ∪ retest·shallow_touch",
    "C": "trigger·large_body ∪ retest·no_touch ∪ extended",
}

STATE_LABEL = {
    "trigger": "Trigger",
    "retest_bounce": "Retest",
    "extended": "Extended",
}
STATE_COLOR = {
    "trigger": "var(--nox-green)",
    "retest_bounce": "var(--nox-cyan)",
    "extended": "var(--nox-orange)",
}

REGIME_COLOR = {
    "long": "var(--nox-green)",
    "neutral": "var(--text-muted)",
    "short": "var(--nox-red)",
}

_LOCAL_CSS = """
.briefing-container {
  position: relative; z-index: 1;
  max-width: 1280px; margin: 0 auto;
  padding: 0 1.4rem 2rem;
}
.hb-tab-nav {
  display: flex; gap: 0.6rem; flex-wrap: wrap;
  margin: 0.8rem 0 1.2rem 0;
  position: sticky; top: 0; z-index: 5;
  background: var(--bg, #0c0d10); padding: 0.5rem 0;
  border-bottom: 1px solid var(--border-dim);
}
.hb-tab {
  display: inline-flex; align-items: center; gap: 0.45rem;
  padding: 0.4rem 0.8rem; border: 1px solid var(--border-dim);
  border-radius: 999px; font-size: 0.78rem; cursor: pointer;
  background: var(--bg-elevated); color: var(--text-secondary);
  text-decoration: none;
}
.hb-tab:hover { border-color: var(--nox-cyan); color: var(--nox-cyan); }
.hb-tab .cnt {
  font-family: var(--font-mono); font-weight: 700; font-size: 0.78rem;
  color: var(--text-primary);
}
.hb-tab .dot { width: 8px; height: 8px; border-radius: 50%; }
.hb-card {
  background: var(--bg-elevated); border: 1px solid var(--border-dim);
  border-radius: 10px; padding: 1rem 1.1rem; margin: 1rem 0;
  scroll-margin-top: 5rem;
}
.hb-card h2 {
  font-size: 0.95rem; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--text-secondary);
  margin: 0 0 0.7rem 0;
  display: flex; align-items: center; gap: 0.6rem;
}
.hb-card h2 .tag {
  font-size: 0.7rem; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--border-dim); color: var(--text-muted);
  letter-spacing: 0.05em;
}
.hb-card h3 {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--text-muted);
  margin: 1rem 0 0.5rem 0;
}
.hb-note {
  font-size: 0.78rem; color: var(--text-muted);
  line-height: 1.5; margin: 0 0 0.8rem 0;
}
table.nox-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.nox-table th, table.nox-table td {
  padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border-dim);
}
table.nox-table th {
  text-align: left; font-weight: 600; letter-spacing: 0.03em;
  color: var(--text-secondary); text-transform: uppercase; font-size: 0.72rem;
}
table.nox-table tbody tr:hover { background: var(--bg-hover); }
tr.tradeable-row { background: rgba(34,197,94,0.06); }
tr.tradeable-row:hover { background: rgba(34,197,94,0.10); }
.pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
}
.pill-tradeable {
  background: var(--nox-green); color: #0a0a0a;
  font-size: 0.65rem; padding: 0 5px;
}
.empty-row td {
  text-align: center; opacity: 0.6; padding: 1.1rem 0.6rem;
}
.hb-footer {
  margin-top: 1.4rem; font-size: 0.72rem; color: var(--text-muted);
  line-height: 1.5;
}
.hb-footer code { color: var(--text-secondary); }
"""

TABLE_HEADER = (
    "<thead><tr>"
    "<th>Sembol</th>"
    "<th>State</th>"
    "<th>Body</th>"
    "<th>Retest</th>"
    "<th style='text-align:right'>Age</th>"
    "<th>Regime</th>"
    "<th style='text-align:right'>Day %</th>"
    "<th style='text-align:right'>ATR %</th>"
    "<th style='text-align:right'>BO %</th>"
    "<th>Trade</th>"
    "</tr></thead>"
)


def _pct(x, default="—"):
    if x is None or pd.isna(x):
        return default
    return f"{x*100:+.2f}%"


def _atrpct(x, default="—"):
    if x is None or pd.isna(x):
        return default
    return f"{x*100:.2f}%"


def _state_pill(ss: str) -> str:
    label = STATE_LABEL.get(ss, ss)
    color = STATE_COLOR.get(ss, "var(--text-muted)")
    return f"<span class='pill' style='color:{color};border:1px solid {color}'>{html.escape(label)}</span>"


def _regime_pill(regime) -> str:
    if regime is None or pd.isna(regime) or regime == "":
        return "<span style='color:var(--text-muted)'>—</span>"
    color = REGIME_COLOR.get(regime, "var(--text-muted)")
    return f"<span style='color:{color}'>{html.escape(str(regime))}</span>"


def _row(r: pd.Series) -> str:
    body = str(r.get("family__body_class", "") or "")
    retest = str(r.get("family__retest_kind", "") or "")
    if not body or body == "nan":
        body = "—"
    if not retest or retest == "nan":
        retest = "—"
    age = r.get("family__breakout_age")
    age_str = "—" if age is None or pd.isna(age) else f"{int(age)}"
    trd_a = bool(r.get("tradeable_a", False))
    trd_ab = bool(r.get("tradeable_ab", False))
    if trd_a:
        trd_pill = "<span class='pill pill-tradeable'>A</span>"
    elif trd_ab:
        trd_pill = "<span class='pill pill-tradeable' style='background:var(--nox-cyan)'>AB</span>"
    else:
        trd_pill = "<span style='color:var(--text-muted)'>—</span>"
    cls = "tradeable-row" if trd_a else ""
    return (
        f"<tr class='{cls}'>"
        f"<td><b>{html.escape(str(r['ticker']))}</b></td>"
        f"<td>{_state_pill(str(r['signal_state']))}</td>"
        f"<td>{html.escape(body)}</td>"
        f"<td>{html.escape(retest)}</td>"
        f"<td style='text-align:right'>{age_str}</td>"
        f"<td>{_regime_pill(r.get('common__regime'))}</td>"
        f"<td style='text-align:right'>{_pct(r.get('common__day_return'))}</td>"
        f"<td style='text-align:right'>{_atrpct(r.get('common__atr_pct'))}</td>"
        f"<td style='text-align:right'>{_pct(r.get('common__breakout_pct'))}</td>"
        f"<td>{trd_pill}</td>"
        "</tr>"
    )


def _table(rows_html: str, empty_msg: str) -> str:
    body = rows_html if rows_html else f"<tr class='empty-row'><td colspan='10'>{empty_msg}</td></tr>"
    return f"<table class='nox-table'>{TABLE_HEADER}<tbody>{body}</tbody></table>"


def _tier_section(tier: str, df: pd.DataFrame) -> tuple[str, dict]:
    sub = df[df["tier"] == tier].copy() if not df.empty else df
    n = len(sub)
    n_trig = int((sub["signal_state"] == "trigger").sum()) if n else 0
    n_ret = int((sub["signal_state"] == "retest_bounce").sum()) if n else 0
    n_ext = int((sub["signal_state"] == "extended").sum()) if n else 0
    n_trd_a = int(sub["tradeable_a"].sum()) if n else 0
    n_trd_ab = int(sub["tradeable_ab"].sum()) if n else 0

    rows_html = "".join(_row(r) for _, r in sub.iterrows())
    color = TIER_COLOR[tier]

    body = f"""
<section class="hb-card" id="tier-{tier}">
  <h2>
    <span style="color:{color};">●</span>
    {TIER_LABEL[tier]}
    <span style="color:var(--text-muted);font-weight:400;font-size:0.78rem;">· {TIER_HINT[tier]}</span>
    <span class="tag">N {n}</span>
    <span class="tag">trig {n_trig} / retest {n_ret} / ext {n_ext}</span>
    <span class="tag">tradeable A {n_trd_a} · AB {n_trd_ab}</span>
  </h2>
  {_table(rows_html, f'Tier {tier} eventi yok')}
</section>
"""
    summary = {
        "n_total": n,
        "n_trigger": n_trig,
        "n_retest_bounce": n_ret,
        "n_extended": n_ext,
        "n_tradeable_a": n_trd_a,
        "n_tradeable_ab": n_trd_ab,
        "tradeable_a_tickers": (
            sub.loc[sub["tradeable_a"], "ticker"].astype(str).tolist() if n else []
        ),
        "tradeable_ab_tickers": (
            sub.loc[sub["tradeable_ab"] & ~sub["tradeable_a"], "ticker"].astype(str).tolist() if n else []
        ),
    }
    return body, summary


def render(asof: str, in_dir: Path) -> tuple[str, dict]:
    csv_path = in_dir / f"horizontal_base_live_{asof}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    sections: list[str] = []
    summaries: dict[str, dict] = {}
    for tier in ["A", "B", "C"]:
        body, summary = _tier_section(tier, df)
        sections.append(body)
        summaries[tier] = summary

    n_total = len(df)
    n_trd_a = int(df["tradeable_a"].sum()) if n_total else 0
    n_trd_ab = int(df["tradeable_ab"].sum()) if n_total else 0
    n_trig = int((df["signal_state"] == "trigger").sum()) if n_total else 0
    n_ret = int((df["signal_state"] == "retest_bounce").sum()) if n_total else 0
    n_ext = int((df["signal_state"] == "extended").sum()) if n_total else 0

    nav_html = '<div class="hb-tab-nav">' + "".join(
        f'<a class="hb-tab" href="#tier-{tier}">'
        f'<span class="dot" style="background:{TIER_COLOR[tier]}"></span>'
        f'{TIER_LABEL[tier]} <span class="cnt">{summaries[tier]["n_total"]}</span>'
        f'</a>'
        for tier in ["A", "B", "C"]
    ) + '</div>'

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    picks_json = {
        "schema_version": 1,
        "system": "horizontal_base_live",
        "scanner_version": "1.4.0",
        "feature_version": "1.3.0",
        "asof": asof,
        "n_total": n_total,
        "n_trigger": n_trig,
        "n_retest_bounce": n_ret,
        "n_extended": n_ext,
        "n_tradeable_a": n_trd_a,
        "n_tradeable_ab": n_trd_ab,
        "tier_summaries": summaries,
    }
    picks_marker = (
        '<script id="horizontal-base-live-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    body = f"""
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="briefing-container">

  <div class="nox-header" style="margin-bottom:1.2rem;">
    <div class="nox-logo">
      HORIZONTAL BASE<span class="proj"> · LIVE TARAMA</span>
      <span class="mode">scanner v1.4.0 · feature 1.3.0 · trigger ∪ retest_bounce ∪ extended · tier A/B/C</span>
    </div>
    <div class="nox-meta">
      As-of: <b>{html.escape(asof)}</b><br>
      N <b>{n_total}</b> · trigger <b>{n_trig}</b> · retest <b>{n_ret}</b> · extended <b>{n_ext}</b><br>
      tradeable A <b>{n_trd_a}</b> · AB <b>{n_trd_ab}</b><br>
      generated {html.escape(now_str)}
    </div>
  </div>

  <div class="nox-stats">
""" + "".join(
        f'<div class="nox-stat"><span class="dot" style="background:{TIER_COLOR[tier]};"></span>'
        f'<span>{TIER_LABEL[tier]}</span><span class="cnt">{summaries[tier]["n_total"]}</span></div>'
        for tier in ["A", "B", "C"]
    ) + f"""
  </div>

  {nav_html}

  <p class="hb-note">
    <b>Tradeable filter:</b> signal_state ∈ {{trigger, retest_bounce}} ∧ breakout_age ≤ 5
    ∧ regime ∈ {{long, neutral}}. Yeşil zemin = <code>tradeable_a</code> (Tier A geçenler);
    AB rozeti = Tier B'den eklenenler. Extended (cycle yaşlı) <b>tradeable değil</b>;
    referans amaçlı listelenir.
  </p>

  {"".join(sections)}

  <div class="hb-footer">
    Pipeline: <code>tools/horizontal_base_scan_live.py</code> ·
    render: <code>tools/horizontal_base_html.py</code>.<br>
    Veri: <code>output/extfeed_intraday_1h_3y_master.parquet</code> (BIST 607 evren, 3y, 1h→günlük resample).
    Regime: RDP v1 (mult=8 sw=8%) · <code>output/regime_labels_daily_rdp_v1.csv</code> backward as-of.<br>
    Tier: <code>tools/scanner_v1_event_quality_diag.py</code> PF tag'lerinden türetildi
    (A: trigger·mid_body ∪ retest·deep_touch · B: trigger·strict_body ∪ retest·shallow_touch ·
    C: trigger·large_body ∪ retest·no_touch).<br>
    <b>Bu bir tarama çıktısıdır</b> — descriptive event log, <b>trade-edge iddiası YOK</b>.
    Pre-registered ML lines <code>v1</code> + <code>v1_2</code> REJECTED;
    rules-based ranker backtest in-sample reference: tier-A K=3 portfolio
    Sharpe ≈ 2.0 / MaxDD ≈ −5% (NOT pre-registered, NOT OOS).
  </div>

</div>
"""

    page = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX Horizontal Base · {html.escape(asof)}</title>
<style>
{_NOX_CSS}
{_LOCAL_CSS}
</style>
</head>
<body>
{picks_marker}
{body}
</body>
</html>
"""
    return page, picks_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD; matches CSV filename")
    ap.add_argument("--in-dir", default="output")
    ap.add_argument("--out", default=None,
                    help="default: output/horizontal_base_scan_<asof>.html")
    ap.add_argument("--also-latest", action="store_true",
                    help="also write output/horizontal_base_scan_latest.html")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else in_dir / f"horizontal_base_scan_{args.asof}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page, _ = render(args.asof, in_dir)
    out_path.write_text(page, encoding="utf-8")
    print(f"[write] {out_path}")

    if args.also_latest:
        latest = out_path.parent / "horizontal_base_scan_latest.html"
        latest.write_text(page, encoding="utf-8")
        print(f"[write] {latest}")


if __name__ == "__main__":
    main()
