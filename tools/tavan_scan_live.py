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
    scorer = MLScorer()

    price_data = {}
    for code, d in daily_cache.items():
        df = d.copy()
        df.columns = [c.title() for c in df.columns]
        price_data[code] = df

    if scorer.loaded:
        scores = scorer.score_tickers(list(daily_cache.keys()), price_data, xu)
    else:
        print("  [WARN] MLScorer not loaded — ml_s set to NaN")
        scores = {}

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
    """Minimal NOX-themed report."""
    n_total = len(df)
    n_v1 = int(df["v1_candidate"].sum()) if "v1_candidate" in df.columns and not df.empty else 0
    title = f"Tavan V1 Live Scan — {asof_stamp}" + (f" ({scan_tag})" if scan_tag else "")

    if df.empty:
        body = "<p class='empty'>Bugün tavan tespit edilmedi.</p>"
    else:
        cols = ["ticker", "prev_close", "today_close", "pct_from_prev_close",
                "pct_from_tavan_target", "is_currently_at_tavan",
                "hit_tavan_intraday", "ml_s", "v1_candidate"]
        cols = [c for c in cols if c in df.columns]
        view = df[cols].copy()
        for c in ("prev_close", "today_close"):
            if c in view.columns:
                view[c] = view[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        for c in ("pct_from_prev_close", "pct_from_tavan_target"):
            if c in view.columns:
                view[c] = view[c].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        if "ml_s" in view.columns:
            view["ml_s"] = view["ml_s"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        for c in ("is_currently_at_tavan", "hit_tavan_intraday", "v1_candidate"):
            if c in view.columns:
                view[c] = view[c].map(lambda x: "✓" if x else "")
        body = "<h2>Adaylar</h2>" + view.to_html(index=False, classes="tbl", escape=False)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#0b0f15; color:#e8edf3; font-family: -apple-system, sans-serif; padding:24px; }}
  h1 {{ color:#7dd3fc; margin-bottom:4px; }}
  .meta {{ color:#94a3b8; font-size:13px; margin-bottom:20px; }}
  .empty {{ color:#94a3b8; padding:40px; text-align:center; }}
  .tbl {{ border-collapse: collapse; margin-top: 12px; }}
  .tbl th, .tbl td {{ border:1px solid #233; padding:6px 10px; font-size:13px; }}
  .tbl th {{ background:#1a2332; color:#7dd3fc; }}
  .tbl tr:nth-child(even) {{ background:#0f1622; }}
  .config {{ background:#1a2332; padding:12px; border-left:3px solid #7dd3fc;
             margin:16px 0; font-size:13px; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="meta">Universe: {n_universe} ticker · tavan hits: {n_total} · V1 candidates: <b>{n_v1}</b></div>
<div class="config">
  <b>V1 Config</b> (WF-validated 2026-05-21): H=25 / Trail 2% / SL-10% / ML_S≥0.65 / D1-FILTER<br>
  <b>Entry</b>: akşam close (18:00 TR) · <b>Tavan tespit</b>: today_close ≈ prev_close × 1.10 (±0.5%)<br>
  <b>OOS perf</b>: WR 91.4% · mean +11.0% · PF 21.2 · p5 -5.2% · ~459 trade/yıl
</div>
{body}
</body></html>"""


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
