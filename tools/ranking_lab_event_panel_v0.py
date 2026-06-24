"""Ranking Lab v0 — Stage 1 event panel.

Builds `output/ranking_lab_event_panel_v0.parquet` as the single source of truth
for the right-tail-runner-ranker research lab.

Design contract (per ONAY 2026-05-12):
  - Use migrated clean consumer outputs only:
        mb_scanner, horizontal_base, triangle_break, paper_execution_v0
                                                       (line_e + line_tr)
  - Compute forward MFE / realized labels at horizons [3, 5, 10, 20, 30, 60]
    using the SAME algorithm, SAME daily-bar repertoire, SAME resample
    convention, SAME anchor convention as the existing unified-events builder
    (`tools/decision_engine_lab_unified_events_3y.py`).
  - Drop rows whose signal_date > per-ticker max bar date in the extfeed
    master (future-dated monthly anchors, etc.).
  - Run equality audit vs the existing 5-horizon unified panel for the shared
    horizons [5, 10, 20, 30, 60].
  - Emit `output/ranking_lab_base_rates_v0.csv` with the ONAY-specified
    binary base rates (y3_8, y5_10, y10_15, y30_30, y30_50, y60_80) +
    audit-only y30_80 / y60_100 if sample supports.

This script DOES NOT modify the existing unified panel or its builder; it
imports the 4 in-scope source loaders and re-implements only the forward
MFE/ATR computation locally so the horizon list can be extended.

Outputs:
    output/ranking_lab_event_panel_v0.parquet
    output/ranking_lab_event_panel_v0.md
    output/ranking_lab_base_rates_v0.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
TOOLS = ROOT / "tools"

sys.path.insert(0, str(TOOLS))
import decision_engine_lab_unified_events_3y as u3y  # noqa: E402

EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"

DST_PARQUET = OUT / "ranking_lab_event_panel_v0.parquet"
DST_MD = OUT / "ranking_lab_event_panel_v0.md"
DST_BASE_RATES = OUT / "ranking_lab_base_rates_v0.csv"

OLD_PANEL = OUT / "decision_engine_lab_unified_events_3y.parquet"

WINDOWS = [3, 5, 10, 20, 30, 60]
OLD_WINDOWS = [5, 10, 20, 30, 60]

ALLOWED_SOURCES = {"mb_scanner", "horizontal_base", "triangle_break", "paper_execution_v0"}

UNIFIED_COLS = [
    "ticker", "signal_date", "source", "family", "setup_label",
    "signal_state", "side", "entry_ref", "atr", "atr_14_daily",
] + [f"mfe_pct_{w}d" for w in WINDOWS] + [f"realized_pct_{w}d" for w in WINDOWS] + [
    "src_native_payload",
]

BASE_RATE_TARGETS = [
    ("y3_8", "mfe_pct_3d", 0.08),
    ("y5_10", "mfe_pct_5d", 0.10),
    ("y10_15", "mfe_pct_10d", 0.15),
    ("y30_30", "mfe_pct_30d", 0.30),
    ("y30_50", "mfe_pct_30d", 0.50),
    ("y60_80", "mfe_pct_60d", 0.80),
    # audit-only (per ONAY: if sample supports)
    ("y30_80", "mfe_pct_30d", 0.80),
    ("y60_100", "mfe_pct_60d", 1.00),
]


def _compute_forward_and_atr(daily: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date), compute forward MFE / realized at WINDOWS + ATR(14).

    Algorithm verbatim from decision_engine_lab_unified_events_3y._compute_forward_and_atr,
    with WINDOWS = [3, 5, 10, 20, 30, 60] instead of [5, 10, 20, 30, 60].

    Anchor convention: close[i] of signal_date; forward window = high/close over
    days i+1 ... i+w (inclusive). Future-dated bars (i+w >= n) yield NaN.
    """
    parts = []
    n_tk = daily.ticker.nunique()
    print(f"[ranking_lab_panel] computing forward+atr per ticker (n={n_tk}) windows={WINDOWS} ...")
    for k, (tkr, grp) in enumerate(daily.groupby("ticker", sort=False), 1):
        g = grp.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        prev_close = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = np.full(n, np.nan)
        period = 14
        if n >= period:
            atr[period - 1] = np.nanmean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        out = pd.DataFrame({"ticker": tkr, "date": g["date"]})
        out["atr_14_daily"] = atr
        for w in WINDOWS:
            mfe = np.full(n, np.nan)
            real = np.full(n, np.nan)
            limit = n - w
            for i in range(limit):
                lo = i + 1
                hi = i + w + 1
                mfe[i] = high[lo:hi].max() / close[i] - 1.0
                real[i] = close[i + w] / close[i] - 1.0
            out[f"mfe_pct_{w}d"] = mfe
            out[f"realized_pct_{w}d"] = real
        parts.append(out)
        if k % 100 == 0:
            print(f"  {k}/{n_tk} tickers processed")
    ref = pd.concat(parts, ignore_index=True)
    print(f"[ranking_lab_panel] reference rows: {len(ref):,}")
    return ref


def _load_in_scope_sources() -> pd.DataFrame:
    """Load events from the 4 in-scope migrated consumer sources."""
    parts: list[pd.DataFrame] = []

    print("[ranking_lab_panel] loading in-scope sources only ...")
    # mb_scanner: 8 family parquets
    for tf in ("5h", "1d", "1w", "1M"):
        for mode in ("mb", "bb"):
            family = f"{mode}_{tf}"
            p = OUT / f"mb_scanner_events_{family}.parquet"
            df = u3y._load_mb(p, family)
            print(f"  mb_scanner/{family}: {len(df):,} rows")
            parts.append(df)

    # horizontal_base
    df = u3y._load_hb(OUT / "horizontal_base_event_v1.parquet")
    print(f"  horizontal_base/hb: {len(df):,} rows")
    parts.append(df)

    # triangle_break
    df = u3y._load_triangle_break(OUT / "decision_engine_lab_triangle_break_events_3y.parquet")
    print(f"  triangle_break: {len(df):,} rows")
    parts.append(df)

    # paper_execution_v0 / line_e
    df = u3y._load_paper_line_e(OUT / "paper_execution_v0_trades.parquet")
    print(f"  paper_execution_v0/line_e: {len(df):,} rows")
    parts.append(df)

    # paper_execution_v0 / line_tr
    df = u3y._load_paper_line_tr(OUT / "paper_execution_v0_trigger_retest_trades.parquet")
    print(f"  paper_execution_v0/line_tr: {len(df):,} rows")
    parts.append(df)

    unified = pd.concat([p for p in parts if p is not None and len(p)],
                        ignore_index=True, sort=False)
    unified["signal_date"] = pd.to_datetime(unified["signal_date"]).dt.normalize()
    unified = unified.dropna(subset=["ticker", "signal_date"]).reset_index(drop=True)
    print(f"[ranking_lab_panel] stacked in-scope events: {len(unified):,} rows")
    # Sanity check: all sources must be in ALLOWED_SOURCES
    bad = set(unified["source"].unique()) - ALLOWED_SOURCES
    assert not bad, f"unexpected sources: {bad}"
    return unified


def _drop_future_dated(unified: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows whose signal_date > per-ticker max bar date in extfeed daily.

    Returns (kept_panel, dropped_df) so the audit can report excluded rows by
    source/family/timeframe.
    """
    per_t_max = daily.groupby("ticker")["date"].max().reset_index().rename(columns={"date": "_max_bar_date"})
    per_t_max["_max_bar_date"] = pd.to_datetime(per_t_max["_max_bar_date"])
    m = unified.merge(per_t_max, on="ticker", how="left")
    is_future = m["signal_date"] > m["_max_bar_date"]
    # Also flag events where ticker has no bars at all (cannot compute labels)
    has_no_bars = m["_max_bar_date"].isna()
    drop_mask = is_future | has_no_bars
    kept = m.loc[~drop_mask].drop(columns=["_max_bar_date"]).reset_index(drop=True)
    dropped = m.loc[drop_mask, ["ticker", "signal_date", "source", "family", "_max_bar_date"]].reset_index(drop=True)
    n_future = int(is_future.sum())
    n_no_bars = int(has_no_bars.sum())
    print(f"[ranking_lab_panel] dropped future-dated rows: {n_future:,}  ticker-without-bars: {n_no_bars:,}")
    return kept, dropped


def _equality_audit(new_panel: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    """Compare new mfe_pct_{w}d to old panel on (ticker, signal_date, source, family).

    Old panel covers 9 sources; we restrict the join to the in-scope subset.

    The old unified panel is a dev-time research artifact (not committed / absent
    in CI). The audit is a label-equality SANITY check, not required to build the
    panel — when the old panel is missing, skip it and return a zero placeholder.
    """
    if not old_path.exists():
        print(f"[ranking_lab_panel][audit] SKIP — old panel absent ({old_path.name}); "
              "equality audit is dev-only, not required to produce the panel")
        audit = pd.DataFrame(columns=["horizon_d", "n_compared", "n_equal",
                                      "n_diff", "max_abs_diff", "equal_pct"])
        audit.attrs["n_only_old"] = 0
        audit.attrs["n_only_new"] = 0
        audit.attrs["n_both"] = 0
        audit.attrs["skipped"] = True
        return audit
    old = pq.read_table(old_path).to_pandas()
    old["signal_date"] = pd.to_datetime(old["signal_date"]).dt.normalize()
    old = old[old["source"].isin(ALLOWED_SOURCES)].copy()
    print(f"[ranking_lab_panel][audit] old panel in-scope rows: {len(old):,}")

    keys = ["ticker", "signal_date", "source", "family"]
    cols_old = keys + [f"mfe_pct_{w}d" for w in OLD_WINDOWS]
    cols_new = keys + [f"mfe_pct_{w}d" for w in OLD_WINDOWS]
    old_sub = old[cols_old].rename(columns={c: c + "_old" for c in cols_old if c not in keys})
    new_sub = new_panel[cols_new].rename(columns={c: c + "_new" for c in cols_new if c not in keys})

    # NB: same (ticker, signal_date, source, family) can have multiple rows
    # (e.g. several HB events on the same bar with different signal_state).
    # We compare ROW-COUNT alignment after sorting + adding row index within group.
    old_sub = old_sub.sort_values(keys).reset_index(drop=True)
    new_sub = new_sub.sort_values(keys).reset_index(drop=True)
    old_sub["_rk"] = old_sub.groupby(keys).cumcount()
    new_sub["_rk"] = new_sub.groupby(keys).cumcount()
    merged = old_sub.merge(new_sub, on=keys + ["_rk"], how="outer", indicator=True)

    rows = []
    for w in OLD_WINDOWS:
        a = merged[f"mfe_pct_{w}d_old"]
        b = merged[f"mfe_pct_{w}d_new"]
        both = a.notna() & b.notna()
        if both.sum() == 0:
            rows.append({"horizon_d": w, "n_compared": 0, "n_equal": 0,
                         "n_diff": 0, "max_abs_diff": float("nan"),
                         "equal_pct": float("nan")})
            continue
        diff = (a[both] - b[both]).abs()
        max_abs = float(diff.max())
        n_equal = int((diff <= 1e-9).sum())
        n_diff = int((diff > 1e-9).sum())
        rows.append({"horizon_d": w, "n_compared": int(both.sum()),
                     "n_equal": n_equal, "n_diff": n_diff,
                     "max_abs_diff": max_abs,
                     "equal_pct": float(n_equal) / float(both.sum())})

    audit = pd.DataFrame(rows)
    n_only_old = int((merged["_merge"] == "left_only").sum())
    n_only_new = int((merged["_merge"] == "right_only").sum())
    audit.attrs["n_only_old"] = n_only_old
    audit.attrs["n_only_new"] = n_only_new
    audit.attrs["n_both"] = int((merged["_merge"] == "both").sum())
    print(f"[ranking_lab_panel][audit] keys both={audit.attrs['n_both']:,}  only_old={n_only_old:,}  only_new={n_only_new:,}")
    return audit


def _base_rates(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col, thr in BASE_RATE_TARGETS:
        s = panel[col].astype(float)
        denom = int(s.notna().sum())
        if denom == 0:
            rows.append({"target": name, "horizon": col, "threshold": thr,
                         "n_eligible": 0, "n_hit": 0, "base_rate": float("nan")})
            continue
        hit = int((s >= thr).sum())
        rows.append({"target": name, "horizon": col, "threshold": thr,
                     "n_eligible": denom, "n_hit": hit,
                     "base_rate": hit / denom})
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[ranking_lab_panel] horizons={WINDOWS}  allowed_sources={sorted(ALLOWED_SOURCES)}")

    daily = u3y._load_daily_bars()
    print(f"[ranking_lab_panel] daily bars: {len(daily):,}  tickers={daily.ticker.nunique()}  "
          f"range={daily.date.min().date()} → {daily.date.max().date()}")

    ref = _compute_forward_and_atr(daily)
    ref = ref.rename(columns={"date": "signal_date"})

    unified = _load_in_scope_sources()
    n_pre = len(unified)
    unified, dropped = _drop_future_dated(unified, daily)
    n_post = len(unified)
    n_dropped_future = n_pre - n_post

    print(f"[ranking_lab_panel] joining forward returns + atr_14_daily ...")
    unified = unified.merge(ref, on=["ticker", "signal_date"], how="left")
    unified = unified[UNIFIED_COLS]
    unified["atr"] = pd.to_numeric(unified["atr"], errors="coerce")
    unified["entry_ref"] = pd.to_numeric(unified["entry_ref"], errors="coerce")

    print(f"[ranking_lab_panel] writing panel → {DST_PARQUET}")
    unified.to_parquet(DST_PARQUET, index=False)

    # ----- equality audit
    audit = _equality_audit(unified, OLD_PANEL)

    # ----- base rates
    base_rates = _base_rates(unified)
    base_rates.to_csv(DST_BASE_RATES, index=False)
    print(f"[ranking_lab_panel] base-rates → {DST_BASE_RATES}")

    # ----- md report
    lines: list[str] = []
    lines.append("# Ranking Lab v0 — Event Panel (Stage 1)")
    lines.append("")
    lines.append(f"- Output: `{DST_PARQUET.name}`  ({DST_PARQUET.stat().st_size:,} B)")
    lines.append(f"- Rows: {len(unified):,}   tickers: {unified['ticker'].nunique()}")
    lines.append(f"- Date range: {unified['signal_date'].min().date()} → "
                 f"{unified['signal_date'].max().date()}")
    lines.append(f"- Sources: {sorted(unified['source'].unique())}")
    lines.append(f"- Forward horizons (recomputed): {WINDOWS}")
    lines.append(f"- Anchor convention: close[i] of signal_date; forward window = high/close over days i+1 .. i+w (inclusive).")
    lines.append("")

    lines.append("## A) Source/family counts (new v0 panel)")
    lines.append("")
    grp = unified.groupby(["source", "family"]).size().rename("n").reset_index()
    grp["pct"] = grp["n"] / grp["n"].sum()
    lines.append(grp.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## B) Future-dated / no-bar rows excluded")
    lines.append("")
    lines.append(f"- total excluded: **{n_dropped_future:,}** (pre={n_pre:,} → post={n_post:,})")
    lines.append("")
    if len(dropped):
        drp_grp = dropped.groupby(["source", "family"]).size().rename("n_excluded").reset_index()
        drp_grp = drp_grp.sort_values("n_excluded", ascending=False)
        lines.append(drp_grp.to_markdown(index=False))
        # Date breakdown
        lines.append("")
        lines.append("### Excluded rows: top signal_dates")
        date_grp = dropped.groupby(["signal_date"]).size().rename("n").reset_index().sort_values("n", ascending=False).head(15)
        lines.append(date_grp.to_markdown(index=False))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## C) Label equality audit vs old panel (shared horizons)")
    lines.append("")
    if audit.attrs.get("skipped"):
        lines.append("- SKIPPED — old unified panel absent (dev-only sanity check, not run in CI).")
        lines.append("")
    else:
        lines.append(f"- old panel in-scope (mb+HB+triangle+paper) keys: **both={audit.attrs['n_both']:,}**, "
                     f"only_old={audit.attrs['n_only_old']:,}, only_new={audit.attrs['n_only_new']:,}")
        lines.append("")
        lines.append(audit.to_markdown(index=False, floatfmt=".6f"))
        lines.append("")

    lines.append("## D) Base rates (Stage 2 target distribution)")
    lines.append("")
    lines.append(base_rates.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")

    lines.append("## E) Forward MFE percentile pool (overall)")
    lines.append("")
    pct_rows = []
    for w in WINDOWS:
        col = f"mfe_pct_{w}d"
        s = unified[col].dropna()
        row = {"horizon_d": w, "n": int(len(s)),
               "mean": float(s.mean()) if len(s) else float("nan"),
               "p50": float(s.quantile(0.50)) if len(s) else float("nan"),
               "p90": float(s.quantile(0.90)) if len(s) else float("nan"),
               "p95": float(s.quantile(0.95)) if len(s) else float("nan"),
               "p99": float(s.quantile(0.99)) if len(s) else float("nan"),
               }
        pct_rows.append(row)
    lines.append(pd.DataFrame(pct_rows).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    DST_MD.write_text("\n".join(lines))
    print(f"[ranking_lab_panel] wrote {DST_MD}")
    print(f"[ranking_lab_panel] DONE  rows={len(unified):,}")


if __name__ == "__main__":
    main()
