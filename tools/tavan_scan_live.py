"""Tavan V1 Live Intraday Scanner — paper/live entry candidate list.

V1 pre-registered + WF-validated config (2026-05-21):
  H=25 / Trail 2% / SL-10% / ML_S>=0.65 / D1-FILTER

Bu script SADECE giriş adayı listesi üretir. Tarama saatlerinde (17:00/17:30 TR)
tetiklenir; akşam close'ta (18:00 TR) trader manuel emir verir.

Mekanik:
  1. Universe = output/extfeed_intraday_coverage.csv
  2. Her ticker için daily-D bar pull (~120 bar; bugünün partial bar'ı dahil).
  3. Tavan tespit: today's close ~ prev_close × 1.10 (±0.5% tolerans)
     - is_currently_at_tavan: |today_close - prev_close*1.10| / prev_close <= 0.005
     - hit_tavan_intraday  : today_high >= prev_close * 1.10
  4. Aday set: is_currently_at_tavan == True
  5. ML_S compute: ml.scorer.MLScorer.score_tickers (compute_all_features 46-feat)
  6. V1 filter: ml_s >= 0.65
  7. Output: parquet/CSV/HTML; push to gh-pages (opt-in).

CLI:
  python tools/tavan_scan_live.py [--limit N] [--n-bars 120] [--asof YYYY-MM-DD]
                                  [--scan-tag 17_00] [--push]

Outputs:
  output/tavan_scan_live_panel_<stamp>.parquet
  output/tavan_scan_live_candidates_<stamp>.csv
  output/tavan_scan_live_<stamp>.html
  output/tavan_scan_live_latest.html

NOTE: backtest is_tavan == close@tavan (T-close exact). Live'da tolerans
proxy kullanıyoruz çünkü tam close henüz oluşmamış; trader bunu kabul etmiş.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from markets.extfeed import auth_from_env, fetch_bars  # noqa: E402

COVERAGE = ROOT / "output/extfeed_intraday_coverage.csv"
OUT_DIR = ROOT / "output"

TAVAN_LIMIT_PCT = 0.10           # BIST daily price band +10%
TAVAN_TOLERANCE = 0.005          # ±0.5% from limit-up considered "at tavan"
ML_S_THRESHOLD = 0.65            # V1 filter
N_BARS_DEFAULT = 120


def load_universe(limit: int | None = None) -> list[str]:
    cov = pd.read_csv(COVERAGE)
    tickers = sorted(cov["ticker"].dropna().unique().tolist())
    return tickers[:limit] if limit else tickers


def pull_daily(auth, code: str, n_bars: int) -> pd.DataFrame:
    """Daily-D bars; latest row is today's partial bar during market hours."""
    df = fetch_bars(f"BIST:{code}", "D", n_bars, auth=auth, timeout_s=20)
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["time"].dt.date)
    out = out[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
    return out


def detect_tavan_state(daily: pd.DataFrame) -> dict | None:
    """Inspect today's row vs yesterday's close for tavan state."""
    if len(daily) < 2:
        return None
    today = daily.iloc[-1]
    prev = daily.iloc[-2]
    prev_close = float(prev["close"])
    if prev_close <= 0 or not np.isfinite(prev_close):
        return None
    today_high = float(today["high"])
    today_close = float(today["close"])
    tavan_target = prev_close * (1.0 + TAVAN_LIMIT_PCT)
    distance_close = (today_close - tavan_target) / prev_close
    return {
        "prev_close":              prev_close,
        "today_open":              float(today["open"]),
        "today_high":              today_high,
        "today_low":                float(today["low"]),
        "today_close":             today_close,
        "today_volume":            float(today["volume"]),
        "tavan_target":            tavan_target,
        "pct_from_prev_close":     (today_close - prev_close) / prev_close * 100.0,
        "pct_from_tavan_target":   distance_close * 100.0,
        "is_currently_at_tavan":   abs(distance_close) <= TAVAN_TOLERANCE,
        "hit_tavan_intraday":      today_high >= tavan_target * (1.0 - 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap universe (debug)")
    ap.add_argument("--n-bars", type=int, default=N_BARS_DEFAULT)
    ap.add_argument("--asof", default="", help="Pin asof (YYYY-MM-DD); blank = latest")
    ap.add_argument("--scan-tag", default="", help="Label for output stamp (e.g. 17_00)")
    ap.add_argument("--push", action="store_true", help="Push HTML to gh-pages")
    args = ap.parse_args()

    universe = load_universe(args.limit)
    print(f"Universe: {len(universe)} tickers")

    auth = auth_from_env()
    time.sleep(0.4)

    # --- pass 1: pull daily, detect tavan candidates ---
    daily_cache: dict[str, pd.DataFrame] = {}
    tavan_rows: list[dict] = []
    failures: list[str] = []
    t0 = time.time()

    for i, code in enumerate(universe, 1):
        try:
            d = pull_daily(auth, code, args.n_bars)
        except Exception as e:
            failures.append(f"{code}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(0.4)
            continue
        if d.empty or len(d) < 80:
            time.sleep(0.4)
            continue

        if args.asof:
            d = d[d.index <= pd.Timestamp(args.asof)]
            if len(d) < 2:
                time.sleep(0.4)
                continue

        st = detect_tavan_state(d)
        if st is None:
            time.sleep(0.4)
            continue

        if st["is_currently_at_tavan"] or st["hit_tavan_intraday"]:
            row = {"ticker": code, **st, "asof": d.index.max()}
            tavan_rows.append(row)
            daily_cache[code] = d

        if i % 50 == 0 or i == len(universe):
            elapsed = (time.time() - t0) / 60
            print(f"  [{i:4d}/{len(universe)}] tavan_hits={len(tavan_rows)} "
                  f"elapsed={elapsed:.1f}m")
        time.sleep(0.3)

    elapsed_m = (time.time() - t0) / 60
    print(f"\n  total pull elapsed: {elapsed_m:.1f} min")
    print(f"  tavan candidates  : {len(tavan_rows)}")
    print(f"  failures          : {len(failures)}")

    if not tavan_rows:
        print("  no tavan hits today — empty list")
        _write_outputs(pd.DataFrame(), args.scan_tag, args.asof, args.push,
                       n_universe=len(universe))
        return 0

    # --- pass 2: ML_S compute ---
    print("\n  pulling XU100 + scoring ML_S...")
    try:
        xu = fetch_bars("BIST:XU100", "D", args.n_bars, auth=auth, timeout_s=30)
        xu["date"] = pd.to_datetime(xu["time"].dt.date)
        xu = xu[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
        if args.asof:
            xu = xu[xu.index <= pd.Timestamp(args.asof)]
        # Uppercase for compute_all_features
        xu.columns = [c.title() for c in xu.columns]
    except Exception as e:
        print(f"  [WARN] XU100 fetch failed: {e}  — ml_s set to NaN for all")
        xu = None

    from ml.scorer import MLScorer  # noqa: E402
    from ml.features import compute_all_features  # noqa: E402
    scorer = MLScorer()
    scorer._load_models()

    price_data = {}
    for code, d in daily_cache.items():
        df = d.copy()
        df.columns = [c.title() for c in df.columns]
        price_data[code] = df

    scores: dict[str, dict] = {}
    score_skip_reason: dict[str, str] = {}
    if not scorer.loaded:
        print("  [WARN] MLScorer not loaded — ml_s set to NaN")
    else:
        for tk in daily_cache.keys():
            df = price_data.get(tk)
            if df is None or len(df) < 80:
                score_skip_reason[tk] = f"insufficient_history ({0 if df is None else len(df)} bars)"
                continue
            try:
                feats = compute_all_features(df, xu_df=xu)
            except Exception as e:
                score_skip_reason[tk] = f"feature_compute_error: {type(e).__name__}: {str(e)[:80]}"
                continue
            if feats.empty:
                score_skip_reason[tk] = "feature_compute_empty"
                continue
            row = feats.iloc[-1]
            vec = scorer._make_feature_vector(row)
            if vec is None:
                missing = [c for c in scorer._feat_cols if c not in row.index or pd.isna(row.get(c))]
                score_skip_reason[tk] = f"feature_vector_sparse (missing {len(missing)}/{len(scorer._feat_cols)})"
                continue
            try:
                pred_a = float(scorer._universe_1g.predict(vec.reshape(1, -1))[0])
                pred_b = None
                if scorer._reranker_1g is not None:
                    pred_b = float(scorer._reranker_1g.predict(vec.reshape(1, -1))[0])
                scores[tk] = {"ml_score": pred_a, "ml_a": pred_a, "ml_b": pred_b}
            except Exception as e:
                score_skip_reason[tk] = f"predict_error: {type(e).__name__}: {str(e)[:80]}"
        if score_skip_reason:
            print(f"\n  [skipped {len(score_skip_reason)}/{len(daily_cache)} from ML_S]:")
            for tk, reason in score_skip_reason.items():
                print(f"    {tk}: {reason}")

    df = pd.DataFrame(tavan_rows)
    df["ml_s"] = df["ticker"].map(lambda t: scores.get(t, {}).get("ml_score", np.nan))
    df["ml_s_ge_065"] = df["ml_s"] >= ML_S_THRESHOLD
    df["v1_candidate"] = df["is_currently_at_tavan"] & df["ml_s_ge_065"]
    df["scan_tag"] = args.scan_tag or "live"
    df["scan_run_utc"] = datetime.utcnow().isoformat(timespec="seconds")

    df = df.sort_values(["v1_candidate", "ml_s"], ascending=[False, False]).reset_index(drop=True)

    print(f"\n  ML_S scored        : {df['ml_s'].notna().sum()}/{len(df)}")
    print(f"  V1 CANDIDATES (ml_s>=0.65 & at-tavan): {int(df['v1_candidate'].sum())}")

    _write_outputs(df, args.scan_tag, args.asof, args.push, n_universe=len(universe))
    return 0


def _write_outputs(df: pd.DataFrame, scan_tag: str, asof: str, push: bool,
                   n_universe: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asof_stamp = (pd.Timestamp(asof).strftime("%Y%m%d") if asof
                  else datetime.utcnow().strftime("%Y%m%d"))
    tag = f"_{scan_tag}" if scan_tag else ""
    stamp = f"{asof_stamp}{tag}"

    out_parquet = OUT_DIR / f"tavan_scan_live_panel_{stamp}.parquet"
    out_csv = OUT_DIR / f"tavan_scan_live_candidates_{stamp}.csv"
    out_html = OUT_DIR / f"tavan_scan_live_{stamp}.html"
    out_latest = OUT_DIR / "tavan_scan_live_latest.html"

    if df.empty:
        df = pd.DataFrame(columns=["ticker", "ml_s", "v1_candidate"])
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"  ✓ panel  → {out_parquet}")
    print(f"  ✓ csv    → {out_csv}")

    html = _render_html(df, scan_tag, asof_stamp, n_universe)
    out_html.write_text(html, encoding="utf-8")
    out_latest.write_text(html, encoding="utf-8")
    print(f"  ✓ html   → {out_html}")

    if push:
        _maybe_push(html, stamp)


def _render_html(df: pd.DataFrame, scan_tag: str, asof_stamp: str, n_universe: int) -> str:
    """NOX-themed report — briefing aesthetic (aurora bg, gold accent)."""
    n_total = len(df)
    n_v1 = int(df["v1_candidate"].sum()) if "v1_candidate" in df.columns and not df.empty else 0
    n_scored = int(df["ml_s"].notna().sum()) if "ml_s" in df.columns and not df.empty else 0
    title_text = f"Tavan V1 Live Scan — {asof_stamp}"
    if scan_tag:
        title_text += f" · {scan_tag}"

    rows_html = ""
    if not df.empty:
        for _, r in df.iterrows():
            is_v1 = bool(r.get("v1_candidate"))
            ml_s = r.get("ml_s")
            ml_s_str = f"{ml_s:.3f}" if pd.notna(ml_s) else "—"
            ml_cls = ""
            if pd.notna(ml_s):
                if ml_s >= 0.65: ml_cls = "ml-hi"
                elif ml_s >= 0.50: ml_cls = "ml-mid"
                else: ml_cls = "ml-lo"
            pct_prev = r.get("pct_from_prev_close", float('nan'))
            pct_tgt = r.get("pct_from_tavan_target", float('nan'))
            badge = '<span class="badge-v1">V1</span>' if is_v1 else ''
            tr_cls = "row-v1" if is_v1 else ""
            ticker = r.get("ticker", "")
            tv_link = (f'<a class="tv-link" href="https://tr.tradingview.com/chart/?symbol=BIST%3A{ticker}" '
                       f'target="_blank">{ticker}</a>')
            rows_html += (
                f'<tr class="{tr_cls}">'
                f'<td class="tk">{tv_link} {badge}</td>'
                f'<td>{r.get("prev_close",""):.2f}</td>'
                f'<td>{r.get("today_close",""):.2f}</td>'
                f'<td class="num">{pct_prev:+.2f}%</td>'
                f'<td class="num muted">{pct_tgt:+.2f}%</td>'
                f'<td>{"✓" if r.get("is_currently_at_tavan") else ""}</td>'
                f'<td>{"✓" if r.get("hit_tavan_intraday") else ""}</td>'
                f'<td class="{ml_cls}">{ml_s_str}</td>'
                f'<td>{"<b>✓</b>" if is_v1 else ""}</td>'
                f'</tr>'
            )
        table_html = f"""
        <div class="nox-table-wrap">
          <table>
            <thead><tr>
              <th>Ticker</th><th>Prev Close</th><th>Today Close</th>
              <th>%Δ prev</th><th>%Δ target</th>
              <th>At Tavan</th><th>Intraday Hit</th>
              <th>ML_S</th><th>V1</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""
    else:
        table_html = '<div class="empty-state">Bugün tavan tespit edilmedi.</div>'

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_text}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik+Glitch&family=Homemade+Apple&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
:root {{
  --bg-primary: #060709;
  --bg-card: #0d0d10;
  --bg-elevated: #141417;
  --bg-hover: #1b1b1f;
  --border-subtle: #1e1e23;
  --border-dim: #2c2b30;
  --text-primary: #e8e4dc;
  --text-secondary: #8a8580;
  --text-muted: #555250;
  --nox-gold: #c9a96e;
  --nox-gold-dim: rgba(201,169,110,0.10);
  --nox-cyan: #c9a96e;
  --nox-cyan-dim: rgba(201,169,110,0.10);
  --nox-green: #7a9e7a;
  --nox-red: #9e5a5a;
  --nox-blue: #7a8fa5;
  --font-display: 'DM Sans', sans-serif;
  --font-brand: 'Rubik Glitch', cursive;
  --font-handwrite: 'Homemade Apple', cursive;
  --font-mono: 'JetBrains Mono', monospace;
  --radius: 14px;
  --radius-sm: 8px;
}}
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: var(--font-display);
  background: #060709;
  color: var(--text-primary);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}
.aurora-bg {{ position: fixed; top:0; left:0; width:100%; height:100%; z-index:0; overflow:hidden; pointer-events:none; }}
.aurora-layer {{ position:absolute; width:200%; height:200%; top:-50%; left:-50%; opacity:1; }}
.aurora-layer-1 {{
  background:
    radial-gradient(ellipse 35% 25% at 8% 12%, rgba(207,199,196,0.35) 0%, transparent 50%),
    radial-gradient(ellipse 25% 20% at 88% 78%, rgba(32,30,33,0.6) 0%, transparent 50%);
  animation: aurora-drift 28s ease-in-out infinite;
}}
.aurora-layer-2 {{
  background:
    radial-gradient(ellipse 30% 22% at 78% 8%, rgba(207,199,196,0.2) 0%, transparent 50%),
    radial-gradient(ellipse 22% 22% at 12% 88%, rgba(32,30,33,0.5) 0%, transparent 50%);
  animation: aurora-drift 35s ease-in-out infinite reverse;
}}
.aurora-layer-3 {{
  background: radial-gradient(ellipse 25% 15% at 45% 45%, rgba(184,149,110,0.12) 0%, transparent 40%);
  animation: aurora-pulse 22s ease-in-out infinite;
}}
.mesh-overlay {{
  position: fixed; top:0; left:0; width:100%; height:100%; z-index:0; pointer-events:none;
  background-image:
    radial-gradient(circle 400px at 12% 18%, rgba(207,199,196,0.08) 0%, transparent 50%),
    radial-gradient(circle 350px at 88% 72%, rgba(32,30,33,0.2) 0%, transparent 50%);
  filter: blur(40px);
}}
@keyframes aurora-drift {{
  0%, 100% {{ transform: translate(0,0) rotate(0deg); }}
  25% {{ transform: translate(4%, -4%) rotate(3deg); }}
  50% {{ transform: translate(-4%, 4%) rotate(-3deg); }}
  75% {{ transform: translate(2%, 2%) rotate(2deg); }}
}}
@keyframes aurora-pulse {{
  0%, 100% {{ opacity: 0.3; transform: scale(1); }}
  50% {{ opacity: 0.6; transform: scale(1.1); }}
}}
.nox-container {{ position:relative; z-index:1; max-width:1280px; margin:0 auto; padding:24px 20px; }}

.nox-header {{
  display:flex; align-items:flex-end; justify-content:space-between;
  padding:24px 0 18px; margin-bottom:24px;
  border-bottom:1px solid var(--border-subtle);
}}
.nox-logo {{
  font-family: var(--font-display);
  font-size: 1.6rem; font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #c9a96e, #e8dcc8, #a8876a);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.nox-logo .proj {{ font-weight:400; -webkit-text-fill-color: var(--text-secondary); font-size:0.7em; margin-left:6px; letter-spacing:0.02em; }}
.nox-logo .mode {{ display:block; font-size:0.45em; font-weight:500; letter-spacing:0.15em;
  text-transform:uppercase; -webkit-text-fill-color: var(--text-muted); margin-top:4px; }}
.nox-meta {{ text-align:right; font-size:0.78rem; color:var(--text-muted); font-family: var(--font-mono); line-height:1.6; }}
.nox-meta b {{ color: var(--nox-gold); }}

.nox-stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
.nox-stat {{
  display:flex; align-items:center; gap:8px;
  background: var(--bg-card); border:1px solid var(--border-subtle);
  border-radius:20px; padding:8px 16px;
  font-size:0.8rem; font-weight:500;
}}
.nox-stat .lbl {{ color: var(--text-secondary); text-transform:uppercase; letter-spacing:0.06em; font-size:0.66rem; }}
.nox-stat .cnt {{ font-family: var(--font-mono); font-weight:700; font-size:0.95rem; color: var(--text-primary); }}
.nox-stat.gold .cnt {{ color: var(--nox-gold); }}

.config-card {{
  background: var(--bg-card); border:1px solid var(--border-subtle);
  border-left:3px solid var(--nox-gold);
  border-radius: var(--radius); padding:14px 18px; margin-bottom:20px;
  font-size:0.82rem; color: var(--text-secondary); line-height:1.7;
}}
.config-card b {{ color: var(--text-primary); }}
.config-card .gold {{ color: var(--nox-gold); }}

.nox-table-wrap {{ overflow-x:auto; border-radius: var(--radius); border:1px solid var(--border-subtle); background: var(--bg-card); }}
table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
thead {{ position:sticky; top:0; z-index:10; }}
th {{
  background: var(--bg-elevated); color: var(--text-muted);
  font-weight:600; font-size:0.66rem;
  text-transform:uppercase; letter-spacing:0.08em;
  padding:12px 10px; text-align:left;
  border-bottom:1px solid var(--border-subtle);
  white-space:nowrap;
}}
td {{ padding:10px; border-bottom:1px solid rgba(39,39,42,0.5); white-space:nowrap;
     font-family: var(--font-mono); font-size:0.78rem; color: var(--text-primary); }}
td.tk {{ font-family: var(--font-display); font-weight:600; }}
td.num {{ font-feature-settings: 'tnum'; }}
td.muted {{ color: var(--text-muted); }}
td.ml-hi {{ color: var(--nox-green); font-weight:700; }}
td.ml-mid {{ color: var(--nox-blue); }}
td.ml-lo {{ color: var(--text-muted); }}
tr {{ transition: background 0.1s; }}
tr:hover {{ background: var(--bg-hover); }}
tr.row-v1 {{ background: var(--nox-gold-dim); }}
tr.row-v1:hover {{ background: rgba(201,169,110,0.18); }}

.tv-link {{ color: var(--text-primary); text-decoration:none; transition: color 0.15s; }}
.tv-link:hover {{ color: var(--nox-gold); }}

.badge-v1 {{
  display:inline-block; padding:2px 7px; border-radius:8px;
  font-size:0.6rem; font-weight:700; font-family: var(--font-mono);
  background: var(--nox-gold-dim); color: var(--nox-gold);
  border:1px solid rgba(201,169,110,0.3);
  margin-left:4px; letter-spacing:0.05em;
}}

.empty-state {{
  background: var(--bg-card); border:1px solid var(--border-subtle);
  border-radius: var(--radius); padding:48px; text-align:center;
  color: var(--text-muted); font-family: var(--font-mono); font-size:0.85rem;
}}

.nox-status {{
  text-align:center; padding:20px 0 8px; margin-top:24px;
  font-size:0.7rem; color: var(--text-muted);
  font-family: var(--font-mono);
  border-top:1px solid var(--border-subtle);
}}
.nox-status b {{ color: var(--nox-gold); }}

@media (max-width: 768px) {{
  .nox-header {{ flex-direction:column; gap:12px; align-items:flex-start; }}
  .nox-meta {{ text-align:left; }}
  table {{ font-size:0.72rem; }}
  td, th {{ padding:6px 5px; }}
  .nox-logo {{ font-size:1.3rem; }}
}}
</style>
</head>
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
      TAVAN<span class="proj">/V1</span>
      <span class="mode">Live Intraday Scan</span>
    </div>
    <div class="nox-meta">
      <div>asof <b>{asof_stamp}</b>{f' · <b>{scan_tag}</b>' if scan_tag else ''}</div>
      <div>universe <b>{n_universe}</b> · scored <b>{n_scored}/{n_total}</b></div>
    </div>
  </header>

  <div class="nox-stats">
    <div class="nox-stat"><span class="lbl">Universe</span><span class="cnt">{n_universe}</span></div>
    <div class="nox-stat"><span class="lbl">Tavan Hits</span><span class="cnt">{n_total}</span></div>
    <div class="nox-stat"><span class="lbl">ML_S Scored</span><span class="cnt">{n_scored}</span></div>
    <div class="nox-stat gold"><span class="lbl">V1 Candidates</span><span class="cnt">{n_v1}</span></div>
  </div>

  <div class="config-card">
    <div><b>V1 Config</b> <span class="gold">(WF-validated 2026-05-21)</span> · H=25 · Trail 2% · SL-10% · ML_S≥0.65 · D1-FILTER</div>
    <div><b>Entry</b>: akşam close (18:00 TR) · <b>Tavan tespit</b>: today_close ≈ prev_close × 1.10 (±0.5%)</div>
    <div><b>OOS perf</b>: WR <span class="gold">91.4%</span> · mean <span class="gold">+11.0%</span> · PF 21.2 · p5 -5.2% · ~459 trade/yıl</div>
  </div>

  {table_html}

  <div class="nox-status">
    NOX · tavan v1 scanner · generated <b>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</b>
  </div>
</div>
</body>
</html>"""


def _maybe_push(html: str, stamp: str) -> None:
    try:
        from core.reports import push_html_to_github
    except ImportError as e:
        print(f"  push skipped: cannot import push_html_to_github ({e})")
        return
    push_html_to_github(html, f"tavan_scan_live_{stamp}.html", stamp)
    push_html_to_github(html, "tavan_scan_live_latest.html", stamp)


if __name__ == "__main__":
    sys.exit(main())
