"""HW Overlay v1 descriptive HTML panel.

Renders metrics CSV as a briefing-aesthetic panel: per-scanner sticky tabs,
per-family table with 3-column dot-class split (pooled / AL / AL_OS),
random-baseline percentile band shown next to observed WR.

Output: output/hw_overlay_v1_panel.html
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

DEFAULT_METRICS = "output/hw_overlay_v1_metrics.csv"
DEFAULT_OUT = "output/hw_overlay_v1_panel.html"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_num(v, dp: int = 3) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{dp}f}"


def _fmt_int(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{int(v)}"


def _fmt_lift(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.2f}×"


def _lift_cell(v) -> str:
    if pd.isna(v):
        return "—"
    if v >= 1.10:
        color = "#1a8a3a"
    elif v >= 1.00:
        color = "#5a8a3a"
    elif v >= 0.90:
        color = "#888"
    else:
        color = "#a4231e"
    return f"<span style='color:{color};font-weight:600'>{v:.2f}×</span>"


def _wr_cell(wr, p05, p50, p95) -> str:
    if pd.isna(wr):
        return "—"
    color = "#888"
    if not pd.isna(p95) and wr > p95:
        color = "#1a8a3a"
    elif not pd.isna(p05) and wr < p05:
        color = "#a4231e"
    band = ""
    if not pd.isna(p50):
        band = f"<div class='band'>rand p05/p50/p95: {_fmt_pct(p05)} / {_fmt_pct(p50)} / {_fmt_pct(p95)}</div>"
    return f"<div style='color:{color};font-weight:600'>{_fmt_pct(wr)}</div>{band}"


def render(metrics_csv: str = DEFAULT_METRICS, out_path: str = DEFAULT_OUT) -> str:
    df = pd.read_csv(metrics_csv)
    scanners = sorted(df["scanner"].unique())

    parts = []
    parts.append("""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>HW Overlay v1 — descriptive panel</title>
<style>
  :root { --bg:#fafafa; --card:#fff; --border:#e3e3e3; --muted:#666; --ink:#222; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); margin: 0; padding: 0; }
  header { padding: 22px 28px 12px; border-bottom: 1px solid var(--border); background: var(--card); position: sticky; top: 0; z-index: 5; }
  header h1 { margin: 0 0 4px; font-size: 20px; font-weight: 700; }
  header .meta { color: var(--muted); font-size: 13px; }
  .scope-banner { background: #fff8e6; border: 1px solid #f0c040; padding: 10px 14px; margin: 12px 28px 0; border-radius: 4px; font-size: 13px; color: #5a4400; }
  nav.tabs { padding: 8px 28px 0; background: var(--card); border-bottom: 1px solid var(--border); position: sticky; top: 60px; z-index: 4; display: flex; gap: 4px; flex-wrap: wrap; }
  nav.tabs a { padding: 8px 14px; border: 1px solid var(--border); border-bottom: none; border-radius: 4px 4px 0 0; text-decoration: none; color: var(--ink); font-size: 13px; font-weight: 500; background: #f0f0f0; }
  nav.tabs a:hover { background: #e8e8e8; }
  nav.tabs a.active { background: var(--card); border-color: var(--border); border-bottom: 1px solid var(--card); margin-bottom: -1px; color: #1a4dbf; font-weight: 600; }
  main { padding: 24px 28px; }
  section.scanner { margin-bottom: 36px; scroll-margin-top: 110px; }
  section h2 { margin: 0 0 8px; font-size: 17px; font-weight: 700; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; background: var(--card); margin: 12px 0 8px; font-size: 12.5px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); border-radius: 4px; overflow: hidden; }
  th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid #f0f0f0; }
  th { background: #f6f6f6; font-weight: 600; color: #444; text-align: right; }
  th.l, td.l { text-align: left; }
  td.fam { font-weight: 600; }
  .slice-tag { display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 2px; background: #eef; color: #336; margin-left: 4px; }
  .slice-tag.AL_only { background: #e8f0ff; color: #2050a0; }
  .slice-tag.AL_OS_only { background: #ffe8e8; color: #a04020; }
  .band { font-size: 10.5px; color: var(--muted); margin-top: 2px; }
  .legend { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .small { color: var(--muted); font-size: 11.5px; }
</style>
</head><body>""")

    parts.append('<header>')
    parts.append('<h1>HW Overlay v1 — descriptive panel</h1>')
    parts.append(f'<div class="meta">{len(df):,} cohort × slice rows · scanners: {", ".join(scanners)} · WR = realized_R ≥ +10% AND holding ≤ 10d</div>')
    parts.append('</header>')

    parts.append('<div class="scope-banner">')
    parts.append('<b>Scope:</b> descriptive metric overlay only. NO acceptance gates, NO promotion path. ')
    parts.append('Big-dot (AL_OS) inclusion is for measurement transparency; any lift > random p95 is a measurement, not a green light to re-open closed lines.')
    parts.append('</div>')

    parts.append('<nav class="tabs">')
    for i, s in enumerate(scanners):
        active = ' class="active"' if i == 0 else ''
        parts.append(f'<a href="#sc-{s}"{active}>{s}</a>')
    parts.append('</nav>')

    parts.append('<main>')
    for s in scanners:
        sub = df[df["scanner"] == s].sort_values(["family", "slice"])
        families = sub["family"].unique()
        parts.append(f'<section id="sc-{s}" class="scanner">')
        parts.append(f'<h2>{html.escape(s)}</h2>')
        parts.append('<table>')
        parts.append('<thead><tr>'
                     '<th class="l">family</th>'
                     '<th class="l">slice</th>'
                     '<th>n_ev</th><th>n_tr</th><th>fill</th>'
                     '<th>WR<br/>(HW=B)</th>'
                     '<th>WR_C<br/>(HW+10d)</th>'
                     '<th>WR_D<br/>(multi-cyc)</th>'
                     '<th>WR_A<br/>all</th><th>WR_A<br/>aligned</th>'
                     '<th>lift_HW<br/>total</th><th>lift_HW<br/>timing</th>'
                     '<th>lift_D<br/>vs A_all</th>'
                     '<th>lift_filter<br/>aligned</th><th>lift_SAT<br/>vs fixed</th>'
                     '<th>lift<br/>vs rand</th>'
                     '<th>mean R</th><th>mean R<br/>arm A</th><th>med R</th><th>MFE cap</th>'
                     '<th>Sharpe</th><th>Sortino</th><th>MaxDD<br/>(cumR)</th><th>worst<br/>trade</th>'
                     '<th>hold</th><th>%HW exit</th>'
                     '</tr></thead><tbody>')
        for fam in families:
            fam_df = sub[sub["family"] == fam]
            for _, r in fam_df.iterrows():
                slice_class = r["slice"]
                fam_cell = f'<td class="l fam">{html.escape(str(fam))}</td>' if r["slice"] == "pooled" else '<td class="l"></td>'
                slice_cell = f'<span class="slice-tag {slice_class}">{r["slice"]}</span>'
                parts.append('<tr>')
                parts.append(fam_cell)
                parts.append(f'<td class="l">{slice_cell}</td>')
                parts.append(f'<td>{_fmt_int(r["n_events"])}</td>')
                parts.append(f'<td>{_fmt_int(r["n_traded"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["entry_fill_rate"])}</td>')
                parts.append(f'<td>{_wr_cell(r["WR"], r["WR_random_p05"], r["WR_random_p50"], r["WR_random_p95"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["WR_armC"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["WR_armD"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["WR_armA_all_events"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["WR_armA_aligned"])}</td>')
                parts.append(f'<td>{_lift_cell(r["lift_HW_total"])}</td>')
                parts.append(f'<td>{_lift_cell(r["lift_HW_timing"])}</td>')
                parts.append(f'<td>{_lift_cell(r["lift_multicycle_vs_A_all"])}</td>')
                parts.append(f'<td>{_lift_cell(r["lift_HW_filter_aligned"])}</td>')
                parts.append(f'<td>{_lift_cell(r["lift_satexit_vs_fixed"])}</td>')
                parts.append(f'<td>{_fmt_lift(r["lift_vs_random"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["mean_realized_R"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["mean_R_armA_all_events"])}</td>')
                parts.append(f'<td>{_fmt_pct(r["median_realized_R"])}</td>')
                parts.append(f'<td>{_fmt_num(r["mfe_capture_median"], 2)}</td>')
                parts.append(f'<td>{_fmt_num(r["Sharpe_per_trade"], 2)}</td>')
                parts.append(f'<td>{_fmt_num(r["Sortino_per_trade"], 2)}</td>')
                parts.append(f'<td>{_fmt_num(r["MaxDD_cumR"], 2)}</td>')
                parts.append(f'<td>{_fmt_pct(r["worst_trade_R"])}</td>')
                parts.append(f'<td>{_fmt_num(r["mean_holding_period"], 1)}</td>')
                parts.append(f'<td>{_fmt_pct(r["pct_HW_exit"])}</td>')
                parts.append('</tr>')
        parts.append('</tbody></table>')
        parts.append(f'<div class="legend">family wait_n: {sub["wait_n"].iloc[0]} trading days · WR color: green &gt; rand p95, red &lt; rand p05, gray within band · '
                     '<b>WR_A all</b> = scanner-alone (enter event close, hold 10d) on ALL events · '
                     '<b>WR_A aligned</b> = scanner-alone on the same events HW filled in this slice · '
                     '<b>lift_HW total</b> = WR / WR_A_all (filter+timing combined) · '
                     '<b>lift_HW timing</b> = WR / WR_A_aligned (timing only, same events) · '
                     '<b>WR_C</b> = HW entry, +10d fixed exit (no SAT) · '
                     '<b>lift_filter aligned</b> = WR_C / WR_A_aligned (HW filter+entry-delay only) · '
                     '<b>lift_SAT vs fixed</b> = WR / WR_C (does HW SAT help vs fixed +10d?) · '
                     '<b>WR_D</b> = multi-cycle in 10d (AL→enter, SAT→exit, repeat, force-close at +10d, compound R) · '
                     '<b>lift_D vs A_all</b> = WR_D / WR_A_all (multi-cycle vs scanner-alone)</div>')
        parts.append('</section>')

    parts.append('</main>')
    parts.append("""<script>
const tabs = document.querySelectorAll('nav.tabs a');
tabs.forEach(t => t.addEventListener('click', e => {
  tabs.forEach(x => x.classList.remove('active'));
  t.classList.add('active');
}));
</script></body></html>""")

    html_str = "\n".join(parts)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html_str, encoding="utf-8")
    print(f"[html] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=DEFAULT_METRICS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    render(args.metrics, args.out)


if __name__ == "__main__":
    main()
