"""Build pre-open HTML report from validator output and publish.

Reads the validator output CSV (per-ticker PASS/WATCH/REJECT + pre-market
features), emits a single self-contained HTML report (no external CDNs),
pushes it to GH Pages via the Contents API, and sends a Telegram message
with the URL.

Usage:
    python tools/preopen_report_html.py \\
        --validated  output/preopen_validated_us_2026_05_26.csv \\
        [--candidates output/tclose_candidates_us_mb_1d_2026_05_22.csv] \\
        [--out-html  output/preopen_us_2026_05_26.html] \\
        [--publish-name preopen_us_2026_05_26.html] \\
        [--send-telegram]

Env vars (read from process env or .env via wrapper):
    GH_TOKEN          GitHub API token (contents:write on GH_PAGES_REPO)
    GH_PAGES_REPO     "<owner>/<name>" of the pages repo
    GH_PAGES_BASE_URL optional override (defaults to
                      https://<owner>.github.io/<name>)
    TG_BOT_TOKEN      Telegram bot token
    TG_CHAT_ID        Telegram chat id

Exit codes: 0 success, 1 missing input, 2 publish failed.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import urllib.request
import urllib.parse


# ───────────── HTML rendering ─────────────

def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:+.2f}%"


def _fmt_num(v, dp=2) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{dp}f}"


def _fmt_int(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{int(v):,}"


def _badge_for(status: str) -> str:
    s = (status or "").upper()
    if "PASS" in s:
        return ("<span style='background:#1f7a3a;color:#fff;padding:2px 8px;"
                "border-radius:4px;font-size:11px;font-weight:600'>PASS</span>")
    if "REJECT" in s:
        return ("<span style='background:#b71c1c;color:#fff;padding:2px 8px;"
                "border-radius:4px;font-size:11px;font-weight:600'>REJECT</span>")
    if "WATCH" in s:
        return ("<span style='background:#b78400;color:#fff;padding:2px 8px;"
                "border-radius:4px;font-size:11px;font-weight:600'>WATCH</span>")
    return f"<span style='color:#666'>{status or '—'}</span>"


def _gap_color(gap_pct) -> str:
    if pd.isna(gap_pct):
        return "color:#888"
    if gap_pct >= 0.03:
        return "color:#1f7a3a;font-weight:600"
    if gap_pct >= 0.005:
        return "color:#1f7a3a"
    if gap_pct <= -0.03:
        return "color:#b71c1c;font-weight:600"
    if gap_pct <= -0.005:
        return "color:#b71c1c"
    return "color:#444"


def build_html(
    df: pd.DataFrame,
    *,
    snapshot_date: str,
    candidates_path: str,
    n_pass: int,
    n_watch: int,
    n_reject: int,
) -> str:
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    generated = now_tr.strftime("%Y-%m-%d %H:%M TR")

    # Sort: REJECT first (most urgent to skip), then WATCH by gap_pct desc, then PASS by gap_pct desc.
    def _rank(s):
        s = (s or "").upper()
        if "REJECT" in s:
            return 0
        if "WATCH" in s:
            return 1
        return 2

    d = df.copy()
    d["_rank"] = d["preopen_status"].apply(_rank)
    d = d.sort_values(["_rank", "gap_pct"], ascending=[True, False]).drop(columns="_rank")

    rows_html = []
    for _, r in d.iterrows():
        tv_symbol = f"NASDAQ:{r['ticker']}"  # display fallback
        tv_link = (f"<a href='https://www.tradingview.com/symbols/{r['ticker']}/' "
                   f"target='_blank' style='color:#1976d2;text-decoration:none'>"
                   f"{r['ticker']}</a>")
        gap_style = _gap_color(r.get("gap_pct"))
        rows_html.append(f"""
        <tr>
          <td>{_badge_for(r.get('preopen_status'))}</td>
          <td style='font-weight:600'>{tv_link}</td>
          <td>{_fmt_num(r.get('close_t'))}</td>
          <td>{_fmt_num(r.get('bos_level'))}</td>
          <td>{_fmt_num(r.get('stop_ref'))}</td>
          <td>{_fmt_num(r.get('atr_14'))}</td>
          <td>{_fmt_num(r.get('pm_last'))}</td>
          <td style='{gap_style}'>{_fmt_pct(r.get('gap_pct'))}</td>
          <td>{_fmt_num(r.get('gap_atr'))}</td>
          <td>{_fmt_int(r.get('pm_volume'))}</td>
          <td>{_fmt_pct(r.get('pm_volume_ratio'))}</td>
          <td>{_fmt_num(r.get('setup_score'), 2) if 'setup_score' in d.columns else '—'}</td>
          <td style='color:#666;font-size:11px'>{(r.get('preopen_reasons') or '') if pd.notna(r.get('preopen_reasons')) else ''}</td>
        </tr>""")

    table_html = "\n".join(rows_html)

    summary_chips = (
        f"<span style='background:#1f7a3a;color:#fff;padding:4px 10px;border-radius:4px;margin-right:6px'>PASS {n_pass}</span>"
        f"<span style='background:#b78400;color:#fff;padding:4px 10px;border-radius:4px;margin-right:6px'>WATCH {n_watch}</span>"
        f"<span style='background:#b71c1c;color:#fff;padding:4px 10px;border-radius:4px'>REJECT {n_reject}</span>"
    )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>preopen US {snapshot_date}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 24px;
    background: #fafafa;
    color: #222;
  }}
  h1 {{ font-size: 22px; margin: 0 0 6px 0; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .summary {{ margin: 12px 0 18px; font-size: 13px; }}
  table {{
    border-collapse: collapse;
    background: #fff;
    width: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    font-size: 13px;
  }}
  th {{
    background: #34495e;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 500;
    border-bottom: 2px solid #2c3e50;
    position: sticky;
    top: 0;
  }}
  td {{
    padding: 7px 10px;
    border-bottom: 1px solid #eee;
    vertical-align: middle;
  }}
  tr:hover td {{ background: #f5f9ff; }}
  .footer {{ margin-top: 18px; color: #888; font-size: 11px; }}
</style>
</head>
<body>
  <h1>📋 preopen US — {snapshot_date}</h1>
  <div class="meta">candidates: <code>{Path(candidates_path).name}</code> · generated {generated}</div>
  <div class="summary">{summary_chips}</div>
  <table>
    <thead>
      <tr>
        <th>status</th>
        <th>ticker</th>
        <th>close T</th>
        <th>BoS (LH)</th>
        <th>stop</th>
        <th>ATR</th>
        <th>pm_last</th>
        <th>gap%</th>
        <th>gap_atr</th>
        <th>pm_vol</th>
        <th>vol/avg20</th>
        <th>score</th>
        <th>reasons</th>
      </tr>
    </thead>
    <tbody>
{table_html}
    </tbody>
  </table>
  <div class="footer">
    Source: mb_scanner US fresh BoS T-close candidates + TV WS pre-market snapshot · validator v0 ·
    gap_pct=(pm_last/close_t−1), gap_atr=(pm_last−close_t)/atr_14, vol/avg20=pm_volume/avg_volume_20
  </div>
</body>
</html>
"""


# ───────────── publish ─────────────

def push_to_gh_pages(html: str, filename: str) -> Optional[str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")
    if not token or not repo:
        print("[gh-pages] GH_TOKEN/GH_PAGES_REPO not set — skip publish.")
        return None

    api_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "preopen-report",
    }
    content_b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")

    # Probe for existing sha
    sha = None
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 200:
                import json as _json
                sha = _json.loads(r.read()).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[gh-pages] probe HTTP {e.code} (continuing)")
    except Exception as e:
        print(f"[gh-pages] probe error: {e} (continuing)")

    now_tr = datetime.now(timezone(timedelta(hours=3)))
    payload = {
        "message": f"preopen US — {now_tr.strftime('%Y-%m-%d %H:%M TR')}",
        "content": content_b64,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha

    import json as _json
    req = urllib.request.Request(
        api_url, method="PUT", headers={**headers, "Content-Type": "application/json"},
        data=_json.dumps(payload).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = _json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[gh-pages] PUT failed HTTP {e.code}: {e.read()[:200]}")
        return None
    except Exception as e:
        print(f"[gh-pages] PUT error: {e}")
        return None

    base = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base:
        owner, name = repo.split("/")
        base = f"https://{owner}.github.io/{name}"
    url = f"{base}/{filename}"
    print(f"[gh-pages] published: {url}")
    return url


def send_telegram(url: str, *, snapshot_date: str, n_pass: int, n_watch: int,
                  n_reject: int, n_total: int, top_movers: list[dict]) -> bool:
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat:
        print("[tg] TG_BOT_TOKEN/TG_CHAT_ID missing — skip.")
        return False

    lines = [
        f"<b>📋 preopen US — {snapshot_date}</b>",
        f"PASS={n_pass} · WATCH={n_watch} · REJECT={n_reject}  (N={n_total})",
    ]
    if top_movers:
        lines.append("")
        lines.append("<b>Top movers</b>")
        for m in top_movers[:5]:
            gap = m.get("gap_pct")
            gap_s = f"{gap*100:+.2f}%" if pd.notna(gap) else "—"
            lines.append(f"  {m['ticker']:6s}  {gap_s:>8s}  {m.get('preopen_status','')}")
    if url:
        lines.append("")
        lines.append(f"<a href=\"{url}\">📄 full report</a>")

    msg = "\n".join(lines)

    api = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": "false",
    }).encode()
    req = urllib.request.Request(api, data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[tg] sent ({r.status})")
            return True
    except Exception as e:
        print(f"[tg] send failed: {e}")
        return False


# ───────────── main ─────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", required=True,
                    help="Validator output CSV (preopen_validated_us_*.csv)")
    ap.add_argument("--candidates", default=None,
                    help="Original T-close candidates CSV (for footer reference)")
    ap.add_argument("--out-html", default=None,
                    help="Local HTML path (default = derived from validated filename)")
    ap.add_argument("--publish-name", default=None,
                    help="Filename on GH Pages (default = same as out-html basename)")
    ap.add_argument("--send-telegram", action="store_true",
                    help="Send Telegram message with link.")
    ap.add_argument("--no-publish", action="store_true",
                    help="Skip GH Pages push (write local file only).")
    args = ap.parse_args()

    val_path = Path(args.validated)
    if not val_path.exists():
        print(f"!! validated file missing: {val_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(val_path)
    if "preopen_status" not in df.columns:
        print("!! preopen_status column missing in validated CSV.", file=sys.stderr)
        return 1

    # Derive snapshot_date from filename, e.g. preopen_validated_us_2026_05_26.csv
    stem = val_path.stem  # preopen_validated_us_2026_05_26
    parts = stem.rsplit("_", 3)
    snapshot_date = (
        "_".join(parts[-3:]).replace("_", "-") if len(parts) >= 3 else "unknown"
    )

    counts = df["preopen_status"].value_counts().to_dict()
    n_pass = int(counts.get("PASS_PREOPEN", 0))
    n_watch = int(counts.get("WATCH_OPEN", 0))
    n_reject = int(counts.get("REJECT_PREOPEN", 0))
    n_total = len(df)

    html = build_html(
        df,
        snapshot_date=snapshot_date,
        candidates_path=args.candidates or "(unknown)",
        n_pass=n_pass, n_watch=n_watch, n_reject=n_reject,
    )

    out_html = Path(args.out_html) if args.out_html else (
        val_path.parent / f"preopen_us_{snapshot_date.replace('-', '_')}.html"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[html] wrote {out_html} ({len(html):,} bytes)")

    url = None
    if not args.no_publish:
        publish_name = args.publish_name or out_html.name
        url = push_to_gh_pages(html, publish_name)

    if args.send_telegram:
        # Top movers by absolute gap (REJECT > WATCH > PASS, then |gap|)
        d = df.copy()
        d["abs_gap"] = d["gap_pct"].abs()
        d = d.sort_values("abs_gap", ascending=False)
        movers = d.head(5).to_dict("records")
        send_telegram(
            url=url or "",
            snapshot_date=snapshot_date,
            n_pass=n_pass, n_watch=n_watch, n_reject=n_reject,
            n_total=n_total, top_movers=movers,
        )

    return 0 if (args.no_publish or url) else 2


if __name__ == "__main__":
    sys.exit(main())
