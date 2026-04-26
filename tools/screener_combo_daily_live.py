"""Screener Combo daily live runner — extfeed daily-D pull + 9-list.

Pipeline:
  1. Universe = output/extfeed_intraday_coverage.csv tickers (607)
  2. extfeed timeframe="D" pull, ~180 bars per ticker (rolling window for features)
  3. extfeed pull BIST:XU100 daily-D (benchmark for rs_score)
  4. For each ticker: weekly resample → run RT/NW/AS signals → emit today's row
  5. expand_features (v2 categories + ages + GIR/OK)
  6. Apply pre-trained v2 ranker weights → score
  7. Per-gate top-3, union → 9-list (deduped)

Output:
  output/screener_combo_v1_live_panel_<YYYYMMDD>.parquet  — today's trigger rows + features
  output/screener_combo_v1_live_basket_<YYYYMMDD>.csv    — 9-list with metadata
  output/screener_combo_v1_live_pergate_<YYYYMMDD>.csv   — per-gate top-3 detail
  output/screener_combo_v1_live_<YYYYMMDD>.html          — NOX-themed report
  output/screener_combo_v1_live_latest.html              — convenience alias
  output/_screener_combo_live.log

Usage:
  python tools/screener_combo_daily_live.py [--limit N] [--n-bars N] [--asof YYYY-MM-DD] [--push]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars
from screener_combo.signals import (
    regime_transition_rich,
    nox_rich,
    alsat_rich,
)
from screener_combo.features import compute_feature_panel, FEATURE_COLS
from screener_combo.scan import _days_since_true, _weekly_from_daily
from screener_combo.rank_discovery_v2 import expand_features, GATES
from screener_combo.rank_apply import build_ranker_score
from screener_combo.render_html import render_html


COVERAGE = Path("output/extfeed_intraday_coverage.csv")
WEIGHTS_PATH = Path("output/screener_combo_v1_ranker_weights_v2_trainval_h20.csv")

OUT_PANEL_TPL = "output/screener_combo_v1_live_panel_{stamp}.parquet"
OUT_BASKET_TPL = "output/screener_combo_v1_live_basket_{stamp}.csv"
OUT_PERGATE_TPL = "output/screener_combo_v1_live_pergate_{stamp}.csv"
OUT_LOG = Path("output/_screener_combo_live.log")


def load_universe() -> list[str]:
    cov = pd.read_csv(COVERAGE)
    return sorted(cov["ticker"].dropna().unique().tolist())


def pull_one(auth, symbol: str, n: int) -> pd.DataFrame:
    df = fetch_bars(symbol, "D", n, auth=auth, timeout_s=20)
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["time"].dt.date)
    return out[["date", "open", "high", "low", "close", "volume"]]


def build_today_row(daily: pd.DataFrame, weekly: pd.DataFrame, bench: pd.Series,
                    today: pd.Timestamp) -> pd.DataFrame | None:
    """Return single-row DataFrame for `today` with gates + categories + features.
    None if today not present in daily index or warmup insufficient.
    """
    if today not in daily.index:
        return None
    try:
        rt_df = regime_transition_rich(daily, weekly)
        nw_df = nox_rich(daily, weekly)
        as_df = alsat_rich(daily, weekly, bench)
        feats = compute_feature_panel(daily, weekly, bench)
    except Exception:
        return None

    if today not in rt_df.index:
        return None

    rt_trig = rt_df["al_signal"].astype(bool)
    nw_trig = nw_df["nox_w_trig"].astype(bool)
    as_trig = as_df["al_signal"].astype(bool)
    nox_d_trig = nw_df["nox_d_trig"].astype(bool)

    df = pd.DataFrame({
        "regime_trig": rt_trig,
        "weekly_trig": nw_trig,
        "alsat_trig":  as_trig,
        "rt_subtype":  rt_df["rt_subtype"],
        "rt_tier":     rt_df["rt_tier"],
        "rt_entry_score": rt_df["rt_entry_score"],
        "rt_oe_score":    rt_df["rt_oe_score"],
        "nox_d_trig":  nox_d_trig,
        "nox_dw_type": nw_df["nox_dw_type"],
        "as_subtype":  as_df["as_subtype"],
        "as_decision": as_df["as_decision"],
        "days_since_RT": _days_since_true(rt_trig),
        "days_since_NW": _days_since_true(nw_trig),
        "days_since_AS": _days_since_true(as_trig),
        "days_since_ND": _days_since_true(nox_d_trig),
    }, index=daily.index)
    df = df.join(feats[FEATURE_COLS])

    row = df.loc[[today]].copy()
    if not row[["regime_trig", "weekly_trig", "alsat_trig"]].any(axis=1).item():
        return None
    return row.reset_index().rename(columns={"index": "date"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size for debug")
    ap.add_argument("--n-bars", type=int, default=200,
                    help="daily bars to pull per ticker (default 200 ~ 9 months)")
    ap.add_argument("--asof", default=None,
                    help="pin 'today' (YYYY-MM-DD); default = latest date present in pulled data")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--push", action="store_true",
                    help="push HTML to gh-pages (requires GH_TOKEN + GH_PAGES_REPO env)")
    args = ap.parse_args()

    if not WEIGHTS_PATH.exists():
        print(f"!! ranker weights missing: {WEIGHTS_PATH}", file=sys.stderr)
        return 1
    weights_full = pd.read_csv(WEIGHTS_PATH)

    universe = load_universe()
    if args.limit:
        universe = universe[: args.limit]
    print(f"  universe   : {len(universe)} tickers")
    print(f"  n_bars     : {args.n_bars} daily")

    auth = auth_from_env()
    auth.token()
    print("  ✓ JWT acquired")

    # Benchmark: BIST:XU100 daily-D
    print("  pulling XU100 ...")
    try:
        xu = pull_one(auth, "BIST:XU100", args.n_bars)
        if xu.empty:
            raise RuntimeError("XU100 empty")
        bench_close = xu.set_index("date")["close"]
        print(f"  XU100 bars : {len(bench_close)}  ({bench_close.index.min().date()} → {bench_close.index.max().date()})")
    except Exception as e:
        print(f"  XU100 FAIL : {e}", file=sys.stderr)
        return 1
    time.sleep(0.4)

    # Per-ticker pull + signal extract
    rows: list[pd.DataFrame] = []
    failures: list[str] = []
    log_lines: list[str] = []
    t0 = time.time()
    n_have_data = 0
    n_today_signal = 0

    for i, code in enumerate(universe, 1):
        symbol = f"BIST:{code}"
        try:
            d = pull_one(auth, symbol, args.n_bars)
        except Exception as e:
            failures.append(f"{code}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(0.4)
            continue
        if d.empty or len(d) < 80:
            time.sleep(0.4)
            continue
        n_have_data += 1
        d = d.set_index("date").sort_index()

        # if asof not given, use the max date across all data we've seen so far;
        # we'll infer today after the loop. For now collect raw + try signal latest.
        try:
            w = _weekly_from_daily(d)
            if len(w) < 12:
                time.sleep(0.4)
                continue
        except Exception:
            time.sleep(0.4)
            continue

        latest = d.index.max()
        today = pd.Timestamp(args.asof) if args.asof else latest
        row = build_today_row(d, w, bench_close, today)
        if row is not None:
            row.insert(0, "ticker", code)
            rows.append(row)
            n_today_signal += 1

        if i % 50 == 0 or i == len(universe):
            elapsed = time.time() - t0
            msg = (f"  [{i:4d}/{len(universe)}] {code:6s} "
                   f"have_data={n_have_data} today_sig={n_today_signal} "
                   f"elapsed={elapsed/60:.1f}m")
            print(msg)
            log_lines.append(msg)
        time.sleep(0.4)

    elapsed = time.time() - t0
    print()
    print(f"  total elapsed : {elapsed/60:.1f} min")
    print(f"  have data     : {n_have_data}/{len(universe)}")
    print(f"  signal today  : {n_today_signal}")
    print(f"  failures      : {len(failures)}")

    if not rows:
        print("  no signals fired today — empty basket")
        # still render an empty-state HTML so the daily artifact is consistent
        empty_basket = pd.DataFrame(columns=["ticker", "gate", "rank_in_gate", "rank_score"])
        empty_pergate = pd.DataFrame(columns=["gate", "rank_in_gate", "ticker"])
        stamp = (pd.Timestamp(args.asof).strftime("%Y%m%d") if args.asof
                 else datetime.utcnow().strftime("%Y%m%d"))
        out_html = Path(f"output/screener_combo_v1_live_{stamp}.html")
        latest_html = Path("output/screener_combo_v1_live_latest.html")
        html = render_html(
            empty_basket, empty_pergate,
            asof=pd.Timestamp(args.asof) if args.asof else pd.Timestamp.utcnow(),
            n_universe=len(universe), n_today_signal=0,
        )
        out_html.write_text(html, encoding="utf-8")
        latest_html.write_text(html, encoding="utf-8")
        print(f"  ✓ html (empty) → {out_html}")
        if args.push:
            _maybe_push(html, stamp)
        OUT_LOG.write_text("\n".join(log_lines + [f"FAIL: {f}" for f in failures]))
        return 0

    panel = pd.concat(rows, ignore_index=True)
    asof = panel["date"].max()
    stamp = pd.Timestamp(asof).strftime("%Y%m%d")
    out_panel = Path(OUT_PANEL_TPL.format(stamp=stamp))
    out_basket = Path(OUT_BASKET_TPL.format(stamp=stamp))
    out_pergate = Path(OUT_PERGATE_TPL.format(stamp=stamp))
    out_panel.parent.mkdir(parents=True, exist_ok=True)

    panel["date"] = pd.to_datetime(panel["date"])
    panel.to_parquet(out_panel, index=False)
    print(f"  ✓ panel    → {out_panel}  ({len(panel)} rows)")

    # Expand features (categories + ages + GIR/OK + dummies) and score per gate
    panel_x, _ = expand_features(panel)
    pergate_rows = []
    union_rows = []
    seen = set()  # (ticker, gate_label) for union dedupe info

    for gate in GATES:
        gw = weights_full[weights_full.gate == gate]
        sub = panel_x[panel_x[gate]].copy()
        if sub.empty or gw.empty:
            print(f"  {gate}: 0 fires today")
            continue
        sub["rank_score"] = build_ranker_score(sub, gw, gate)
        top = (sub.sort_values("rank_score", ascending=False)
               .head(args.top_k)
               .copy())
        top["gate"] = gate
        top["rank_in_gate"] = range(1, len(top) + 1)
        pergate_rows.append(top)
        for _, r in top.iterrows():
            union_rows.append({
                "ticker": r["ticker"],
                "gate": gate,
                "rank_in_gate": int(r["rank_in_gate"]),
                "rank_score": float(r["rank_score"]) if pd.notna(r["rank_score"]) else np.nan,
                "rt_subtype": r.get("rt_subtype", ""),
                "rt_tier": r.get("rt_tier", ""),
                "rt_gir": int(r["rt_entry_score"]) if pd.notna(r.get("rt_entry_score")) else None,
                "rt_ok":  int(r["rt_oe_score"])    if pd.notna(r.get("rt_oe_score")) else None,
                "as_subtype": r.get("as_subtype", ""),
                "as_decision": r.get("as_decision", ""),
                "nox_dw_type": r.get("nox_dw_type", ""),
                "close": float(r["close"]) if "close" in r and pd.notna(r["close"]) else np.nan,
                "atr_pct": float(r["atr_pct"]) if pd.notna(r.get("atr_pct")) else np.nan,
                "rs_score": float(r["rs_score"]) if pd.notna(r.get("rs_score")) else np.nan,
            })

    if pergate_rows:
        pergate = pd.concat(pergate_rows, ignore_index=True)
        cols = ["gate", "rank_in_gate", "ticker", "rank_score",
                "rt_subtype", "rt_tier", "rt_entry_score", "rt_oe_score",
                "as_subtype", "as_decision", "nox_dw_type"]
        pergate[[c for c in cols if c in pergate.columns]].to_csv(
            out_pergate, index=False, float_format="%.4f"
        )
        print(f"  ✓ per-gate → {out_pergate}")

    if union_rows:
        u = pd.DataFrame(union_rows)
        # dedupe by ticker, keeping the first (highest-ranked) gate occurrence
        u_dedup = u.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
        u_dedup["asof"] = pd.Timestamp(asof).date().isoformat()
        u_dedup.to_csv(out_basket, index=False, float_format="%.4f")
        print(f"  ✓ basket   → {out_basket}  ({len(u_dedup)} unique tickers, {len(u)} gate-picks)")
        print()
        print(f"=== Combined 9-list (asof {pd.Timestamp(asof).date()}) ===")
        print(u_dedup[["ticker", "gate", "rank_in_gate", "rank_score",
                       "rt_subtype", "rt_tier", "rt_gir", "rt_ok",
                       "as_subtype", "as_decision", "nox_dw_type"]].to_string(index=False))

    # Render HTML
    bk_df = pd.DataFrame(union_rows).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True) \
            if union_rows else pd.DataFrame(columns=["ticker", "gate", "rank_in_gate"])
    pg_df = pd.concat(pergate_rows, ignore_index=True) if pergate_rows else pd.DataFrame()
    out_html = Path(f"output/screener_combo_v1_live_{stamp}.html")
    latest_html = Path("output/screener_combo_v1_live_latest.html")
    html = render_html(
        bk_df, pg_df,
        asof=pd.Timestamp(asof),
        n_universe=len(universe), n_today_signal=n_today_signal,
    )
    out_html.write_text(html, encoding="utf-8")
    latest_html.write_text(html, encoding="utf-8")
    print(f"  ✓ html     → {out_html}")
    if args.push:
        _maybe_push(html, stamp)

    OUT_LOG.write_text("\n".join(log_lines + [f"FAIL: {f}" for f in failures]))
    return 0


def _maybe_push(html: str, stamp: str) -> None:
    try:
        from core.reports import push_html_to_github
    except Exception as e:
        print(f"  push skipped: cannot import push_html_to_github ({e})")
        return
    # daily-stamped + latest convenience filename on gh-pages
    push_html_to_github(html, f"screener_combo_{stamp}.html", stamp)
    push_html_to_github(html, "screener_combo_latest.html", stamp)


if __name__ == "__main__":
    sys.exit(main())
