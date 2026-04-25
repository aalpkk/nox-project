"""Layered daily-bar fetcher: fintables -> extfeed -> matriks -> yfinance.

Daily counterpart of ``nyxexpansion.intraday.fetch_layered``. Used by every
GitHub Actions BIST scan that needs daily OHLCV. Tier order (per ticker):

1. ``fintables_d`` — batch SQL via Fintables MCP HTTP (auto-paged under the
                     300-row cap). No per-ticker rate limit; serves full
                     history. Slow on 6-year × full-universe bootstraps but
                     covers anything the cheaper tiers miss.
2. ``extfeed_d``   — TradingView WS daily bars (cookie-auth). One call per
                     ticker covers ~5000 bars. Fast and complete.
3. ``matriks_d``   — per-ticker historicalData. Live but server caps allBars
                     at 60 daily rows; only useful for short patches.
4. ``yfinance_d``  — last resort batch download. Free, but flaky for the
                     newer BIST listings.

Each row carries a ``bars_source`` tag so downstream code can surface which
tier served each ticker. Callers who want a flat pandas frame can use the
helper :func:`pull_panel`.

CLI (smoke probe):

    python -m nyxexpansion.daily.fetch_layered \\
        --tickers GARAN,THYAO --start 2026-04-22 --end 2026-04-25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nyxexpansion.daily.fetchers import (  # noqa: E402
    AuthError,
    DAILY_BAR_COLUMNS,
    FetchResult,
)
from nyxexpansion.daily.fetchers import fintables as fin  # noqa: E402
from nyxexpansion.daily.fetchers import extfeed_d as ext  # noqa: E402
from nyxexpansion.daily.fetchers import matriks as mat  # noqa: E402
from nyxexpansion.daily.fetchers import yfinance_d as yfd  # noqa: E402


def fetch_daily_layered(
    tickers: list[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    skip_fintables: bool = False,
    skip_extfeed: bool = False,
    skip_matriks: bool = False,
    skip_yfinance: bool = False,
    quiet: bool = False,
) -> dict[str, FetchResult]:
    """Run the four tiers in sequence; return per-ticker outcome map."""
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    pending = list(dict.fromkeys(tickers))  # dedupe, preserve order
    results: dict[str, FetchResult] = {}

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    # Tier 1 — Fintables
    if pending and not skip_fintables:
        if not os.environ.get("FINTABLES_MCP_TOKEN"):
            _log("  [layered-d] FINTABLES_MCP_TOKEN unset — skipping fintables tier")
        else:
            _log(f"  [layered-d] tier 1 fintables ({len(pending)} ticker, {s}..{e})")
            try:
                fin_results = fin.fetch_per_ticker(pending, s, e)
            except AuthError as exc:
                _log(f"  [layered-d] fintables auth error: {exc}")
                fin_results = {}
            except Exception as exc:
                _log(f"  [layered-d] fintables tier failed: {type(exc).__name__}: {exc}")
                fin_results = {}
            served = []
            for tk, res in fin_results.items():
                if res.note == "ok":
                    results[tk] = res
                    served.append(tk)
            for tk in served:
                if tk in pending:
                    pending.remove(tk)
            _log(f"  [layered-d] fintables served {len(served)}/{len(fin_results)}; "
                 f"{len(pending)} → extfeed")

    # Tier 2 — Extfeed (TradingView WS daily)
    if pending and not skip_extfeed:
        if not (os.environ.get("INTRADAY_SID") and os.environ.get("INTRADAY_SIGN")):
            _log("  [layered-d] INTRADAY_SID/SIGN unset — skipping extfeed tier")
        else:
            _log(f"  [layered-d] tier 2 extfeed ({len(pending)} ticker, {s}..{e})")
            try:
                ext_results = ext.fetch_per_ticker(pending, s, e)
            except Exception as exc:
                _log(f"  [layered-d] extfeed tier failed: {type(exc).__name__}: {exc}")
                ext_results = {}
            served = []
            for tk, res in ext_results.items():
                if res.note == "ok":
                    results[tk] = res
                    served.append(tk)
            for tk in served:
                if tk in pending:
                    pending.remove(tk)
            _log(f"  [layered-d] extfeed served {len(served)}/{len(ext_results)}; "
                 f"{len(pending)} → matriks")

    # Tier 3 — Matriks
    if pending and not skip_matriks:
        if not os.environ.get("MATRIKS_API_KEY"):
            _log("  [layered-d] MATRIKS_API_KEY unset — skipping matriks tier")
        else:
            _log(f"  [layered-d] tier 3 matriks ({len(pending)} ticker)")
            try:
                mat_results = mat.fetch_per_ticker(pending, s, e)
            except Exception as exc:
                _log(f"  [layered-d] matriks tier failed: {type(exc).__name__}: {exc}")
                mat_results = {}
            served = []
            for tk, res in mat_results.items():
                if res.note == "ok":
                    results[tk] = res
                    served.append(tk)
            for tk in served:
                if tk in pending:
                    pending.remove(tk)
            _log(f"  [layered-d] matriks served {len(served)}/{len(mat_results)}; "
                 f"{len(pending)} → yfinance")

    # Tier 4 — yfinance
    if pending and not skip_yfinance:
        _log(f"  [layered-d] tier 4 yfinance ({len(pending)} ticker)")
        try:
            yf_results = yfd.fetch_per_ticker(pending, s, e)
        except Exception as exc:
            _log(f"  [layered-d] yfinance tier failed: {type(exc).__name__}: {exc}")
            yf_results = {}
        served = []
        for tk, res in yf_results.items():
            if res.note == "ok":
                results[tk] = res
                served.append(tk)
        for tk in served:
            if tk in pending:
                pending.remove(tk)
        _log(f"  [layered-d] yfinance served {len(served)}/{len(yf_results)}; "
             f"{len(pending)} unresolved")

    for tk in pending:
        results.setdefault(tk, FetchResult(
            ticker=tk, start_date=s, end_date=e,
            bars_source=None, rows=[], note="all_failed",
            detail="all tiers failed",
        ))
    return results


def results_to_frame(results: dict[str, FetchResult]) -> pd.DataFrame:
    """Flatten per-ticker FetchResults into one canonical bar frame."""
    all_rows: list[dict] = []
    for res in results.values():
        all_rows.extend(res.rows)
    if not all_rows:
        return pd.DataFrame(columns=list(DAILY_BAR_COLUMNS))
    df = pd.DataFrame(all_rows)
    for c in DAILY_BAR_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[list(DAILY_BAR_COLUMNS)]


def pull_panel(
    tickers: list[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    skip_fintables: bool = False,
    skip_extfeed: bool = False,
    skip_matriks: bool = False,
    skip_yfinance: bool = False,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """High-level helper: returns (panel_df indexed by Date, summary dict).

    Panel shape: index=Date (datetime64[ns], naive), columns
    Open/High/Low/Close/Volume/ticker (yfinance-compatible). This is the
    same format scanners (run_smart_breakout etc.) already consume.
    """
    results = fetch_daily_layered(
        tickers, start_date, end_date,
        skip_fintables=skip_fintables,
        skip_extfeed=skip_extfeed,
        skip_matriks=skip_matriks,
        skip_yfinance=skip_yfinance,
        quiet=quiet,
    )
    raw = results_to_frame(results)

    counts = {"fintables_d": 0, "extfeed_d": 0, "matriks_d": 0, "yfinance_d": 0, "missing": 0}
    notes: dict[str, int] = {}
    for tk, res in results.items():
        if res.bars_source:
            counts[res.bars_source] = counts.get(res.bars_source, 0) + 1
        else:
            counts["missing"] += 1
        notes[res.note] = notes.get(res.note, 0) + 1

    if raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "ticker"]), {
            "ticker_counts": counts, "notes": notes, "rows": 0,
        }

    panel = raw.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).copy()
    panel["Date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index("Date").sort_index()
    panel = panel[["Open", "High", "Low", "Close", "Volume", "ticker", "bars_source"]]
    return panel, {
        "ticker_counts": counts, "notes": notes, "rows": len(panel),
    }


def _summarize(results: dict[str, FetchResult]) -> dict:
    counts = {"fintables_d": 0, "extfeed_d": 0, "matriks_d": 0, "yfinance_d": 0, "missing": 0}
    notes: dict[str, int] = {}
    rows = 0
    for tk, res in results.items():
        rows += len(res.rows)
        if res.bars_source:
            counts[res.bars_source] = counts.get(res.bars_source, 0) + 1
        else:
            counts["missing"] += 1
        notes[res.note] = notes.get(res.note, 0) + 1
    return {"ticker_counts": counts, "notes": notes, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma-separated ticker list")
    ap.add_argument("--start", required=True, help="start date YYYY-MM-DD (inclusive, Istanbul)")
    ap.add_argument("--end", required=True, help="end date YYYY-MM-DD (inclusive, Istanbul)")
    ap.add_argument("--skip-fintables", action="store_true")
    ap.add_argument("--skip-extfeed", action="store_true")
    ap.add_argument("--skip-matriks", action="store_true")
    ap.add_argument("--skip-yfinance", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("❌ no tickers")
        return 2

    print(f"═══ layered daily fetch ═══")
    print(f"  range   : {args.start} .. {args.end}")
    print(f"  tickers : {len(tickers)}")
    print()

    t0 = time.time()
    results = fetch_daily_layered(
        tickers, args.start, args.end,
        skip_fintables=args.skip_fintables,
        skip_extfeed=args.skip_extfeed,
        skip_matriks=args.skip_matriks,
        skip_yfinance=args.skip_yfinance,
    )
    elapsed = time.time() - t0

    summary = _summarize(results)
    print()
    print(f"═══ summary ({elapsed:.1f}s) ═══")
    for src, n in summary["ticker_counts"].items():
        print(f"  {src:<14} {n}")
    print(f"  notes: {summary['notes']}")
    print(f"  total rows: {summary['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
