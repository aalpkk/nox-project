"""Layered live intraday fetcher: Fintables → Matriks → yfinance 1h.

CLI entry point used by the daily scan workflow before the retention stage:

    python -m nyxexpansion.intraday.fetch_layered \\
        --tickers-from output/nyxexp_scan_<YYYYMMDD>.parquet \\
        --date YYYY-MM-DD \\
        --out output/nyxexp_intraday_master.parquet

Tier order (per ticker):
1. ``fintables_15m``  — batch SQL via Fintables MCP HTTP (single call for all
                        tickers); 15-min delayed but no per-ticker rate limit.
2. ``matriks_15m``    — per-ticker historicalData call; live, but rate-limited
                        and prone to 503/429.
3. ``yfinance_1h``    — last resort, 1h granularity. Reduced sub-hourly
                        resolution; truncated daily aggregate still well-defined.

Each row in the output parquet carries a ``bars_source`` tag so downstream
stages (retention, HTML banner, log writer) can surface which tier served
each candidate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nyxexpansion.intraday.fetchers import (  # noqa: E402
    AuthError,
    FetchResult,
)
from nyxexpansion.intraday.fetchers import fintables as fin  # noqa: E402
from nyxexpansion.intraday.fetchers import matriks as mat  # noqa: E402
from nyxexpansion.intraday.fetchers import yfinance_h as yfh  # noqa: E402

DEFAULT_OUT = Path("output/nyxexp_intraday_master.parquet")

CACHE_COLUMNS = (
    "ticker", "signal_date", "bar_ts", "date",
    "open", "high", "low", "close", "volume", "quantity",
    "bars_source",
)


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    df = pd.read_parquet(path)
    for c in CACHE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _persist(cache_df: pd.DataFrame, new_rows: list[dict], out_path: Path) -> pd.DataFrame:
    if not new_rows:
        return cache_df
    new_df = pd.DataFrame(new_rows)
    for c in CACHE_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = pd.NA
    new_df = new_df[list(CACHE_COLUMNS)]
    combined = pd.concat([cache_df, new_df], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return combined


def _drop_today(cache_df: pd.DataFrame, target_date) -> pd.DataFrame:
    """Remove any rows for ``target_date`` so a fresh fetch overwrites them."""
    if cache_df.empty:
        return cache_df
    target_date_obj = pd.Timestamp(target_date).date()
    sd = pd.to_datetime(cache_df["signal_date"]).dt.date
    return cache_df[sd != target_date_obj].copy()


def fetch_intraday_layered(
    tickers: list[str],
    target_date,
    *,
    skip_fintables: bool = False,
    skip_matriks: bool = False,
    skip_yfinance: bool = False,
) -> dict[str, FetchResult]:
    """Run the three tiers in sequence; return per-ticker outcome map."""
    target = pd.Timestamp(target_date).normalize()
    pending = list(tickers)
    results: dict[str, FetchResult] = {}

    # Tier 1 — Fintables batch
    if pending and not skip_fintables:
        if not os.environ.get("FINTABLES_MCP_TOKEN"):
            print("  [layered] FINTABLES_MCP_TOKEN unset — skipping fintables tier")
        else:
            print(f"  [layered] tier 1 fintables ({len(pending)} ticker batch)")
            try:
                fin_results = fin.fetch_per_ticker(pending, target)
            except AuthError as exc:
                print(f"  [layered] fintables auth error: {exc}")
                fin_results = {}
            except Exception as exc:
                print(f"  [layered] fintables tier failed: {type(exc).__name__}: {exc}")
                fin_results = {}
            served = []
            for tk, res in fin_results.items():
                if res.note == "ok":
                    results[tk] = res
                    served.append(tk)
            for tk in served:
                if tk in pending:
                    pending.remove(tk)
            print(f"  [layered] fintables served {len(served)}/{len(fin_results)}; "
                  f"{len(pending)} → matriks")

    # Tier 2 — Matriks per-ticker
    if pending and not skip_matriks:
        if not os.environ.get("MATRIKS_API_KEY"):
            print("  [layered] MATRIKS_API_KEY unset — skipping matriks tier")
        else:
            print(f"  [layered] tier 2 matriks ({len(pending)} ticker)")
            try:
                mat_results = mat.fetch_per_ticker(pending, target)
            except Exception as exc:
                print(f"  [layered] matriks tier failed: {type(exc).__name__}: {exc}")
                mat_results = {}
            served = []
            for tk, res in mat_results.items():
                if res.note == "ok":
                    results[tk] = res
                    served.append(tk)
            for tk in served:
                if tk in pending:
                    pending.remove(tk)
            print(f"  [layered] matriks served {len(served)}/{len(mat_results)}; "
                  f"{len(pending)} → yfinance")

    # Tier 3 — yfinance 1h
    if pending and not skip_yfinance:
        print(f"  [layered] tier 3 yfinance 1h ({len(pending)} ticker)")
        try:
            yf_results = yfh.fetch_per_ticker(pending, target)
        except Exception as exc:
            print(f"  [layered] yfinance tier failed: {type(exc).__name__}: {exc}")
            yf_results = {}
        served = []
        for tk, res in yf_results.items():
            if res.note == "ok":
                results[tk] = res
                served.append(tk)
        for tk in served:
            if tk in pending:
                pending.remove(tk)
        print(f"  [layered] yfinance served {len(served)}/{len(yf_results)}; "
              f"{len(pending)} unresolved")

    # Final unresolved → record as empty
    target_date_obj = target.date()
    for tk in pending:
        results.setdefault(tk, FetchResult(
            ticker=tk, signal_date=target_date_obj,
            bars_source=None, rows=[], note="all_failed",
            detail="all three tiers failed",
        ))
    return results


def _summarize(results: dict[str, FetchResult]) -> dict:
    counts = {"fintables_15m": 0, "matriks_15m": 0, "yfinance_1h": 0,
              "missing": 0}
    notes: dict[str, int] = {}
    for tk, res in results.items():
        if res.bars_source:
            counts[res.bars_source] = counts.get(res.bars_source, 0) + 1
        else:
            counts["missing"] += 1
        notes[res.note] = notes.get(res.note, 0) + 1
    return {"counts": counts, "notes": notes}


def _load_tickers_from_scan(scan_path: Path, target_date) -> list[str]:
    df = pd.read_parquet(scan_path)
    df["date"] = pd.to_datetime(df["date"])
    target_norm = pd.Timestamp(target_date).normalize()
    panel = df[df["date"].dt.normalize() == target_norm]
    return sorted(panel["ticker"].astype(str).unique().tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="target date YYYY-MM-DD")
    ap.add_argument("--tickers-from", default=None,
                    help="parquet path; pulls unique tickers for target date")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated ticker list (alternative to --tickers-from)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-fintables", action="store_true")
    ap.add_argument("--skip-matriks", action="store_true")
    ap.add_argument("--skip-yfinance", action="store_true")
    args = ap.parse_args()

    target = pd.Timestamp(args.date).normalize()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_from:
        tickers = _load_tickers_from_scan(Path(args.tickers_from), target)
    else:
        print("❌ either --tickers or --tickers-from required")
        return 2
    if not tickers:
        print(f"⚠ no tickers for {target.date()} — nothing to fetch")
        return 0

    out_path = Path(args.out)
    print(f"═══ layered intraday fetch ═══")
    print(f"  target  : {target.date()}")
    print(f"  tickers : {len(tickers)}")
    print(f"  cache   : {out_path}")
    print()

    cache = _load_cache(out_path)
    cache = _drop_today(cache, target)

    t0 = time.time()
    results = fetch_intraday_layered(
        tickers, target,
        skip_fintables=args.skip_fintables,
        skip_matriks=args.skip_matriks,
        skip_yfinance=args.skip_yfinance,
    )

    new_rows = []
    for res in results.values():
        new_rows.extend(res.rows)
    cache = _persist(cache, new_rows, out_path)

    summary = _summarize(results)
    elapsed = time.time() - t0
    print()
    print(f"═══ summary ({elapsed:.1f}s) ═══")
    for src, n in summary["counts"].items():
        print(f"  {src:<14} {n}")
    print()
    print(f"  notes: {summary['notes']}")
    print(f"  cache rows after persist: {len(cache):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
