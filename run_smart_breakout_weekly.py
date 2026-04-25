#!/usr/bin/env python3
"""Smart Breakout Targets — Haftalık (W-FRI) tarayıcı.

Reuses calc_indicators() and scan_single() from run_smart_breakout.py with
weekly-resampled bars. ML scoring is disabled (model is daily-feature trained).

Data path (3-tier layered, fintables -> matriks -> yfinance):
  1. Load fintables-verified daily panel (output/ohlcv_6y_fintables.parquet)
  2. PATCH missing recent days for panel tickers via layered fetcher
  3. BACKFILL universe-only tickers (full 6y) via layered fetcher
  4. Resample W-FRI; truncate to --as-of (default last Friday)
  5. Run scan_single() per ticker → HTML report

The layered fetcher lives at ``nyxexpansion/daily/fetch_layered.py`` and is
shared by every BIST scanner that needs daily OHLCV.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_smart_breakout import (  # noqa: E402
    BB_LENGTH,
    ATR_SMA_LENGTH,
    calc_indicators,
    scan_single,
)
from nyxexpansion.daily.fetch_layered import pull_panel  # noqa: E402

PANEL_PARQUET = ROOT / "output" / "ohlcv_6y_fintables.parquet"
SYMBOLS_FILE = ROOT / "tools" / "bist_symbols.txt"
OUT_HTML = ROOT / "output" / "smart_breakout_weekly.html"

DEFAULT_AS_OF = "2026-04-24"
PATCH_LOOKBACK_DAYS = 14
BACKFILL_PERIOD = "6y"


def _load_symbols() -> list[str]:
    return [s.strip() for s in SYMBOLS_FILE.read_text().splitlines() if s.strip()]


def _load_panel() -> pd.DataFrame:
    """Load the cached fintables-verified panel; empty frame if missing.

    Empty frame triggers a full backfill (every universe symbol pulled via the
    layered fetcher). On GitHub Actions the parquet is restored from
    actions/cache so subsequent runs only patch the delta.
    """
    if not PANEL_PARQUET.exists():
        print(f"  [load] panel parquet not found at {PANEL_PARQUET} — full bootstrap")
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "ticker"])
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty
    df = pd.read_parquet(PANEL_PARQUET)
    df.index = pd.to_datetime(df.index)
    return df


def _save_panel(panel: pd.DataFrame) -> None:
    if panel is None or panel.empty:
        return
    PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_index().to_parquet(PANEL_PARQUET)
    print(f"  [save] panel parquet → {PANEL_PARQUET} ({len(panel):,} bars)")


def _layered_pull(tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp,
                   *, label: str) -> tuple[pd.DataFrame, dict]:
    """Layered (fintables -> matriks -> yfinance) pull for a date window."""
    if not tickers:
        return pd.DataFrame(), {"rows": 0, "ticker_counts": {}, "notes": {}}
    print(f"  [{label}] layered pull {start_date.date()} → {end_date.date()} for {len(tickers)} ticker")
    panel, summary = pull_panel(tickers, start_date.date(), end_date.date(), quiet=False)
    if panel.empty:
        print(f"  [{label}] no rows returned (counts={summary.get('ticker_counts', {})})")
        return panel, summary
    counts = summary.get("ticker_counts", {})
    breakdown = " ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(f"  [{label}] {len(panel):,} bars  ·  {breakdown}")
    return panel, summary


def _patch_panel(panel: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Append missing recent days for tickers already in the panel."""
    if panel.empty:
        return panel, {"patched_rows": 0, "patch_window": None, "ticker_counts": {}}
    panel_max = panel.index.max().normalize()
    if panel_max >= as_of:
        return panel, {"patched_rows": 0, "patch_window": None, "ticker_counts": {}}
    start_date = (panel_max + pd.Timedelta(days=1))
    panel_tickers = sorted(panel["ticker"].unique().tolist())
    new_df, summary = _layered_pull(panel_tickers, start_date, as_of, label="patch")
    if new_df.empty:
        return panel, {"patched_rows": 0,
                        "patch_window": (start_date.strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d")),
                        "ticker_counts": summary.get("ticker_counts", {})}
    new_df = new_df[new_df.index <= as_of]
    add = new_df[["Open", "High", "Low", "Close", "Volume", "ticker"]]
    combined = pd.concat([panel, add]).sort_index()
    combined = (
        combined.reset_index()
        .drop_duplicates(subset=["Date", "ticker"], keep="last")
        .set_index("Date")
        .sort_index()
    )
    return combined, {
        "patched_rows": len(add),
        "patch_window": (start_date.strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d")),
        "ticker_counts": summary.get("ticker_counts", {}),
    }


def _backfill_missing(panel: pd.DataFrame, all_symbols: list[str], as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Pull full history for symbols absent from the panel."""
    have = set(panel["ticker"].unique())
    missing = sorted(set(all_symbols) - have)
    if not missing:
        return panel, {"backfilled_tickers": 0, "missing": [], "ticker_counts": {}}
    start_date = (as_of - pd.Timedelta(days=365 * 6)).normalize()
    add_df, summary = _layered_pull(missing, start_date, as_of, label="backfill")
    if add_df.empty:
        return panel, {"backfilled_tickers": 0, "missing": missing,
                        "ticker_counts": summary.get("ticker_counts", {})}
    add_df = add_df[add_df.index <= as_of]
    add = add_df[["Open", "High", "Low", "Close", "Volume", "ticker"]]
    filled = sorted(add["ticker"].unique().tolist())
    still_missing = sorted(set(missing) - set(filled))
    combined = pd.concat([panel, add]).sort_index()
    return combined, {
        "backfilled_tickers": len(filled),
        "filled": filled,
        "missing": still_missing,
        "ticker_counts": summary.get("ticker_counts", {}),
    }


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to W-FRI bars (label = Friday close)."""
    agg = (
        daily.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    return agg


def _build_weekly_panel(panel: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Per-ticker weekly OHLCV truncated at as_of (inclusive of week containing as_of)."""
    weekly: dict[str, pd.DataFrame] = {}
    panel = panel[panel.index <= as_of]
    for ticker, group in panel.groupby("ticker"):
        sub = group[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index()
        if sub.empty:
            continue
        wk = _to_weekly(sub)
        wk = wk[wk.index <= (as_of + pd.Timedelta(days=7))]
        wk = wk[wk.index <= pd.Timestamp(as_of) + pd.tseries.offsets.Week(weekday=4)]
        if len(wk) < (BB_LENGTH + ATR_SMA_LENGTH + 10):
            continue
        weekly[ticker] = wk
    return weekly


def _scan_weekly(weekly: dict[str, pd.DataFrame], use_vol: bool, use_htf: bool, max_bo_age: int) -> dict[str, list]:
    results = {"SQUEEZE": [], "BOX": [], "BREAKOUT": []}
    skipped = 0
    for ticker, wk in weekly.items():
        try:
            res = scan_single(wk, use_vol, use_htf)
        except Exception:
            skipped += 1
            continue
        if res is None:
            continue
        res["ticker"] = ticker
        status = res["status"]
        if status == "BREAKOUT":
            if res["bars_ago"] <= max_bo_age and res["trade_status"] == "OPEN":
                results["BREAKOUT"].append(res)
        elif status == "BOX":
            results["BOX"].append(res)
        elif status == "SQUEEZE":
            results["SQUEEZE"].append(res)
    if skipped:
        print(f"  [scan] {skipped} ticker scan_single error (skipped)")
    return results


def _generate_html_weekly(results: dict, meta: dict) -> str:
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    date_str = now_tr.strftime("%d.%m.%Y %H:%M")
    as_of_str = pd.Timestamp(meta["as_of"]).strftime("%d.%m.%Y")
    breakouts = sorted(results["BREAKOUT"], key=lambda x: (x.get("strength", 0), -x.get("bars_ago", 99)), reverse=True)
    boxes = sorted(results["BOX"], key=lambda x: x.get("squeeze_bars", 0), reverse=True)
    squeezes = sorted(results["SQUEEZE"], key=lambda x: x["squeeze_bars"], reverse=True)
    TV = "https://www.tradingview.com/chart/?symbol=BIST:"

    def _tk(t: str) -> str:
        return f'<a href="{TV}{t}" target="_blank" class="tk">{t}</a>'

    def _bo_rows() -> str:
        rows = []
        for r in breakouts:
            cls = "long" if r["dir"] == "LONG" else "short"
            tp_marks = ""
            if r["tp1_hit"]:
                tp_marks += " ①"
            if r["tp2_hit"]:
                tp_marks += " ②"
            if r["tp3_hit"]:
                tp_marks += " ③"
            s = r["strength"]
            s_cls = "strong" if s >= 4 else "medium" if s >= 2 else "normal"
            rows.append(
                f'<tr class="{cls}">'
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f'<td class="{cls}">{r["dir"]}</td>'
                f'<td class="{s_cls}">{s}/4</td>'
                f"<td>{r['entry']:.2f}</td>"
                f"<td>{r['sl']:.2f}</td>"
                f"<td>{r['tp1']:.2f}</td>"
                f"<td>{r['tp2']:.2f}</td>"
                f"<td>{r['tp3']:.2f}</td>"
                f"<td>{r['trade_status']}{tp_marks}</td>"
                f"<td>{r['bars_ago']}H</td>"
                f"</tr>"
            )
        return "\n".join(rows) or '<tr><td colspan="10" class="empty">— hiç breakout sinyali yok —</td></tr>'

    def _box_rows() -> str:
        rows = []
        for r in boxes[:60]:
            dist_top = (r["box_top"] - r["close"]) / r["close"] * 100
            dist_bot = (r["close"] - r["box_bot"]) / r["close"] * 100
            closer = f"↑{dist_top:.1f}%" if dist_top < dist_bot else f"↓{dist_bot:.1f}%"
            rows.append(
                f"<tr>"
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f"<td>{r['close']:.2f}</td>"
                f"<td>{r['box_top']:.2f}</td>"
                f"<td>{r['box_bot']:.2f}</td>"
                f"<td>{closer}</td>"
                f"<td>{r['squeeze_bars']}</td>"
                f"<td>{r['bars_since']}H</td>"
                f"</tr>"
            )
        return "\n".join(rows) or '<tr><td colspan="7" class="empty">— —</td></tr>'

    def _sq_rows() -> str:
        rows = []
        for r in squeezes[:60]:
            rows.append(
                f"<tr>"
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f"<td>{r['close']:.2f}</td>"
                f"<td>{r['squeeze_bars']}</td>"
                f"</tr>"
            )
        return "\n".join(rows) or '<tr><td colspan="3" class="empty">— —</td></tr>'

    universe_n = meta.get("universe_n", "?")
    scanned_n = meta.get("scanned_n", "?")
    patch_info = meta.get("patch_info", {})
    backfill_info = meta.get("backfill_info", {})
    patch_rows = patch_info.get("patched_rows", 0)
    bf_n = backfill_info.get("backfilled_tickers", 0)
    bf_missing = backfill_info.get("missing", []) or []
    bf_missing_html = ", ".join(bf_missing) if bf_missing else "—"

    def _tier_badges(counts: dict) -> str:
        if not counts:
            return ""
        parts = []
        for src in ("fintables_d", "matriks_d", "yfinance_d", "missing"):
            n = counts.get(src, 0)
            if n:
                parts.append(f'<span class="tag">{src}</span> {n}')
        return " &nbsp; ".join(parts)

    patch_badges = _tier_badges(patch_info.get("ticker_counts", {}))
    bf_badges = _tier_badges(backfill_info.get("ticker_counts", {}))

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>Smart Breakout — Haftalık (W-FRI) — {as_of_str}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
body{{background:#0d1117;color:#c9d1d9;padding:20px;line-height:1.55}}
h1{{color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:10px;margin-bottom:8px}}
h2{{color:#79c0ff;margin-top:30px;margin-bottom:12px;border-left:3px solid #58a6ff;padding-left:10px}}
.meta{{color:#8b949e;font-size:13px;margin-bottom:18px}}
.meta b{{color:#c9d1d9}}
table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:6px;overflow:hidden;font-size:13px}}
th{{background:#21262d;color:#79c0ff;padding:9px 8px;text-align:left;font-weight:600;border-bottom:1px solid #30363d}}
td{{padding:7px 8px;border-bottom:1px solid #21262d}}
tr.long td.long{{color:#3fb950;font-weight:700}}
tr.short td.short{{color:#f85149;font-weight:700}}
.tk{{color:#58a6ff;text-decoration:none;font-weight:600}}
.tk:hover{{text-decoration:underline}}
.strong{{color:#3fb950;font-weight:700}}
.medium{{color:#d29922;font-weight:600}}
.normal{{color:#8b949e}}
.empty{{text-align:center;color:#8b949e;font-style:italic;padding:18px}}
.banner{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 16px;margin-bottom:18px;color:#8b949e;font-size:12.5px}}
.banner b{{color:#c9d1d9}}
.banner .tag{{display:inline-block;padding:2px 8px;border-radius:10px;background:#21262d;color:#79c0ff;margin-right:6px;font-size:11px}}
</style></head><body>
<h1>📊 Smart Breakout — Haftalık (W-FRI)</h1>
<div class="meta">As-of: <b>{as_of_str}</b> (Cuma kapanışı) · Rapor üretim: {date_str}</div>
<div class="banner">
  <span class="tag">universe</span> {universe_n} sembol &nbsp;
  <span class="tag">tarandı</span> {scanned_n} ticker (≥80 hafta veri) &nbsp;
  <span class="tag">patch</span> {patch_rows} bar &nbsp;
  <span class="tag">backfill</span> {bf_n} ticker
  <br><br>
  <b>Parametreler (W-FRI bar):</b>
  BB({BB_LENGTH}w) ratio &lt; 0.80 · ATR({ATR_SMA_LENGTH//2}w) &lt; 1.0×SMA{ATR_SMA_LENGTH}w ·
  squeeze 5–40 hafta · vol filtre ON · HTF EMA50w OFF · ML scoring DISABLED (model = daily features)
  <br><br>
  <b>Veri yolu (3-tier layered, fintables → matriks → yfinance):</b>
  <br>&nbsp;&nbsp;patch tier dağılımı: {patch_badges or "—"}
  <br>&nbsp;&nbsp;backfill tier dağılımı: {bf_badges or "—"}
  {f"<br><b>Backfill missing:</b> {bf_missing_html}" if bf_missing else ""}
</div>

<h2>🎯 BREAKOUT — açık sinyaller (son {meta.get('max_bo_age',5)} hafta)</h2>
<table>
<thead><tr><th>Sembol</th><th>Yön</th><th>Güç</th><th>Giriş</th><th>SL</th><th>TP1</th><th>TP2</th><th>TP3</th><th>Durum</th><th>Eski</th></tr></thead>
<tbody>{_bo_rows()}</tbody>
</table>

<h2>📦 BOX — kırılma bekleyen kutular (en sıkı 60)</h2>
<table>
<thead><tr><th>Sembol</th><th>Fiyat</th><th>Tepe</th><th>Taban</th><th>Yakın kenar</th><th>Sıkışma (h)</th><th>Eski</th></tr></thead>
<tbody>{_box_rows()}</tbody>
</table>

<h2>🌀 SQUEEZE — hâlâ sıkışmada (en uzun 60)</h2>
<table>
<thead><tr><th>Sembol</th><th>Fiyat</th><th>Süre (h)</th></tr></thead>
<tbody>{_sq_rows()}</tbody>
</table>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=DEFAULT_AS_OF, help="haftalık close tarihi (YYYY-MM-DD, Cuma)")
    ap.add_argument("--max-bo-age", type=int, default=5, help="kaç hafta önceye kadar BREAKOUT göster")
    ap.add_argument("--no-vol", action="store_true", help="hacim filtresini kapat")
    ap.add_argument("--htf", action="store_true", help="HTF EMA50w filtresini aç")
    ap.add_argument("--out", default=str(OUT_HTML))
    args = ap.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize()
    if as_of.weekday() != 4:
        print(f"⚠ as-of {as_of.date()} Cuma değil (weekday={as_of.weekday()}); haftalık resample yine de yürüyecek")

    print(f"═══ Smart Breakout — Haftalık (W-FRI) ═══")
    print(f"  as-of  : {as_of.date()}")
    print(f"  panel  : {PANEL_PARQUET}")
    print()

    print(f"  [load] panel parquet")
    panel = _load_panel()
    print(f"  [load] {len(panel):,} bar, {panel['ticker'].nunique()} ticker, {panel.index.min().date()} → {panel.index.max().date()}")

    panel, patch_info = _patch_panel(panel, as_of)
    symbols = _load_symbols()
    panel, backfill_info = _backfill_missing(panel, symbols, as_of)
    _save_panel(panel)

    print(f"  [resample] daily → W-FRI per ticker")
    weekly = _build_weekly_panel(panel, as_of)
    print(f"  [resample] {len(weekly)} ticker scannable (≥{BB_LENGTH+ATR_SMA_LENGTH+10} weekly bars)")

    print(f"  [scan] running scan_single() per ticker")
    results = _scan_weekly(weekly, use_vol=(not args.no_vol), use_htf=args.htf, max_bo_age=args.max_bo_age)

    print()
    print(f"═══ summary ═══")
    print(f"  BREAKOUT : {len(results['BREAKOUT'])}")
    print(f"  BOX      : {len(results['BOX'])}")
    print(f"  SQUEEZE  : {len(results['SQUEEZE'])}")
    print()

    meta = {
        "as_of": as_of,
        "universe_n": len(symbols),
        "scanned_n": len(weekly),
        "patch_info": patch_info,
        "backfill_info": backfill_info,
        "max_bo_age": args.max_bo_age,
    }
    html = _generate_html_weekly(results, meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"📄 HTML rapor: {out_path}")

    if results["BREAKOUT"]:
        print()
        print("Top 10 BREAKOUT:")
        for r in sorted(results["BREAKOUT"], key=lambda x: (x.get("strength", 0), -x.get("bars_ago", 99)), reverse=True)[:10]:
            print(f"  {r['ticker']:<8} {r['dir']:<5} s={r['strength']}/4 entry={r['entry']:.2f} sl={r['sl']:.2f} tp1={r['tp1']:.2f} status={r['trade_status']} {r['bars_ago']}H")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
