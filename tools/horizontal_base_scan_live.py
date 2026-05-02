"""Live descriptive scanner for V1.4.0 horizontal_base events.

Daily scan emitting today's events (signal_state ∈ {trigger,
retest_bounce, extended}) with derived `tier` (A/B/C) tags and
`tradeable_a` / `tradeable_ab` flags.

Tier classification (sourced from `event_quality_diag` PFs, 2026-04-30):
  A: trigger ∧ body_class=mid_body  ∪  retest_bounce ∧ retest_kind=deep_touch
  B: trigger ∧ body_class=strict_body ∪ retest_bounce ∧ retest_kind=shallow_touch
  C: trigger ∧ body_class=large_body ∪ retest_bounce ∧ retest_kind=no_touch

Tradeable flag (universe filter):
  signal_state ∈ {trigger, retest_bounce} ∧ breakout_age ≤ 5
  ∧ regime ∈ {long, neutral} ∧ tier ∈ <set>

NOT a trade-edge claim. Pre-registered ML lines v1/v1_2 REJECTED;
this output is descriptive cohort tagging on top of the production
scanner. Trader uses tier+context to decide; rules-based picker
backtest (in-sample, NOT pre-reg'd) gives reference numbers in
`output/horizontal_base_rules_v1_backtest.md`.

Usage:
  python -m tools.horizontal_base_scan_live [--asof YYYY-MM-DD] [--out-dir output]

Outputs (per asof):
  output/horizontal_base_live_<asof>.csv     -- today's events full schema + tier
  output/horizontal_base_live_<asof>.parquet -- same, parquet
  output/horizontal_base_live_latest.csv     -- copy of latest run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import intraday_1h  # noqa: E402
from scanner.context import build_market_context, fill_cross_sectional  # noqa: E402
from scanner.schema import FEATURE_VERSION, SCANNER_VERSION  # noqa: E402

# Reuse builder primitives to keep live scan output schema-aligned with
# the V1 freeze parquet (offline) — single source of truth.
from tools.build_horizontal_base_event_v1 import (  # noqa: E402
    REGIME_CSV,
    _emit_rows_for_ticker,
    _patch_load_regime_for_index,
    _restore_load_regime,
    _slope_tier,
)

OUT_DIR_DEFAULT = ROOT / "output"


# ----------------------------------------------------------------------------
# tier + tradeable derivation


def tier_of(row) -> str:
    bc = row.get("family__body_class", "") or ""
    rk = row.get("family__retest_kind", "") or ""
    ss = row.get("signal_state", "") or ""
    if (ss == "trigger" and bc == "mid_body") or (ss == "retest_bounce" and rk == "deep_touch"):
        return "A"
    if (ss == "trigger" and bc == "strict_body") or (ss == "retest_bounce" and rk == "shallow_touch"):
        return "B"
    return "C"


def _tradeable_flag(row, tiers: tuple[str, ...]) -> bool:
    if row.get("signal_state") not in ("trigger", "retest_bounce"):
        return False
    age = row.get("family__breakout_age")
    if age is None or pd.isna(age) or age > 5:
        return False
    regime = row.get("common__regime")
    if regime not in ("long", "neutral"):
        return False
    return row.get("tier") in tiers


# ----------------------------------------------------------------------------
# scan pipeline


def scan(asof: pd.Timestamp | None = None, out_dir: Path = OUT_DIR_DEFAULT) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scan] V1.4.0 horizontal_base live scan — "
          f"SCANNER={SCANNER_VERSION} FEATURE={FEATURE_VERSION}", flush=True)

    t_load = time.time()
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    daily = intraday_1h.daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    print(f"[scan] daily: {len(daily):,} rows / {daily['ticker'].nunique()} tickers "
          f"(load {time.time()-t_load:.1f}s)", flush=True)

    if asof is None:
        asof = daily["date"].max()
    asof = pd.Timestamp(asof).normalize()
    asof_str = asof.strftime("%Y-%m-%d")
    print(f"[scan] asof = {asof_str}", flush=True)

    # Market context with RDP-patched regime source
    print(f"[scan] build_market_context (RDP-patched) …", flush=True)
    original = _patch_load_regime_for_index()
    try:
        market_df, rs_df = build_market_context(daily)
    finally:
        _restore_load_regime(original)

    # Per-ticker scan via shared builder primitive
    tickers = sorted(daily["ticker"].unique())
    all_rows: list[dict] = []
    t_scan = time.time()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        sub.attrs["ticker"] = str(t)
        all_rows.extend(_emit_rows_for_ticker(sub, str(t)))
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t_scan:.1f}s "
                  f"events={len(all_rows):,}", flush=True)
    print(f"[scan] elapsed: {time.time()-t_scan:.1f}s — total events: {len(all_rows):,}",
          flush=True)

    if not all_rows:
        print("[scan] no events emitted", flush=True)
        return pd.DataFrame()

    rows = pd.DataFrame(all_rows)
    rows["bar_date"] = pd.to_datetime(rows["bar_date"]).dt.normalize()

    # Cross-sectional fill (RS, index, breadth, market_trend, market_vol)
    cross_sec_cols = [
        "common__rs_20d", "common__rs_60d", "common__rs_pctile_120",
        "common__rs_pctile_252", "common__sector_rs_20d",
        "common__index_ret_5d", "common__index_ret_20d",
        "common__market_trend_score", "common__market_vol_regime",
        "common__market_breadth_pct_above_sma20",
    ]
    for c in cross_sec_cols:
        if c not in rows.columns:
            rows[c] = np.float32("nan")
    enriched: list[dict] = []
    for _, r in rows.iterrows():
        d = dict(r)
        fill_cross_sectional(d, market_df, rs_df)
        enriched.append(d)
    rows = pd.DataFrame(enriched)

    # RDP regime backward as-of
    rdp = pd.read_csv(REGIME_CSV)
    rdp["date"] = pd.to_datetime(rdp["date"]).dt.normalize()
    rdp = rdp.sort_values("date").drop_duplicates("date", keep="last")
    rdp_lookup = rdp.set_index("date")[["regime", "sub_regime", "window_id"]].sort_index()
    rows = rows.sort_values("bar_date")
    rows["common__regime"] = pd.Series(
        rdp_lookup["regime"].reindex(rows["bar_date"].values, method="ffill").values,
        index=rows.index,
    )
    rows["regime_sub"] = pd.Series(
        rdp_lookup["sub_regime"].reindex(rows["bar_date"].values, method="ffill").values,
        index=rows.index,
    )
    rows["regime_window_id"] = "rdp_v1_mult8_atr14_sw8"

    # Slope tier
    rows["family__abs_base_slope"] = rows["family__base_slope"].abs().astype("float32")
    rows["family__abs_resistance_slope"] = rows["family__resistance_slope"].abs().astype("float32")
    rows["family__slope_tier"] = [
        _slope_tier(a, b)
        for a, b in zip(rows["family__abs_base_slope"], rows["family__abs_resistance_slope"])
    ]

    # Filter to asof
    today = rows[rows["bar_date"] == asof].copy().reset_index(drop=True)
    if today.empty:
        # Try most recent date <= asof in case asof is not a trading day
        max_date = rows.loc[rows["bar_date"] <= asof, "bar_date"].max()
        if pd.notna(max_date) and max_date != asof:
            print(f"[scan] no events on {asof_str}, falling back to last "
                  f"available bar {max_date.date()}", flush=True)
            asof = max_date
            asof_str = asof.strftime("%Y-%m-%d")
            today = rows[rows["bar_date"] == asof].copy().reset_index(drop=True)
        if today.empty:
            print(f"[scan] no events on or before {asof_str}", flush=True)
            return today

    # Tier + tradeable flags
    today["tier"] = today.apply(tier_of, axis=1)
    today["tradeable_a"] = today.apply(lambda r: _tradeable_flag(r, ("A",)), axis=1)
    today["tradeable_ab"] = today.apply(lambda r: _tradeable_flag(r, ("A", "B")), axis=1)

    # Sort: tier A first, then B, then C; within tier day_return DESC
    today["_tier_rank"] = today["tier"].map({"A": 0, "B": 1, "C": 2})
    today = today.sort_values(
        ["_tier_rank", "common__day_return"], ascending=[True, False]
    ).drop(columns="_tier_rank").reset_index(drop=True)

    # Reorder convenient leading columns for readability
    lead_cols = [
        "ticker", "bar_date", "signal_state", "tier",
        "tradeable_a", "tradeable_ab",
        "family__body_class", "family__retest_kind", "family__breakout_age",
        "common__regime", "common__day_return", "common__atr_pct",
    ]
    lead = [c for c in lead_cols if c in today.columns]
    other = [c for c in today.columns if c not in lead]
    today = today[lead + other]

    # Output
    out_csv = out_dir / f"horizontal_base_live_{asof_str}.csv"
    out_parquet = out_dir / f"horizontal_base_live_{asof_str}.parquet"
    out_csv_latest = out_dir / "horizontal_base_live_latest.csv"
    today.to_csv(out_csv, index=False)
    today.to_parquet(out_parquet, index=False)
    today.to_csv(out_csv_latest, index=False)

    n_total = len(today)
    n_a = int((today["tier"] == "A").sum())
    n_b = int((today["tier"] == "B").sum())
    n_c = int((today["tier"] == "C").sum())
    n_trig = int((today["signal_state"] == "trigger").sum())
    n_ret = int((today["signal_state"] == "retest_bounce").sum())
    n_ext = int((today["signal_state"] == "extended").sum())
    n_trd_a = int(today["tradeable_a"].sum())
    n_trd_ab = int(today["tradeable_ab"].sum())
    print(f"[scan] {asof_str}: {n_total} events "
          f"(trigger={n_trig} retest_bounce={n_ret} extended={n_ext})", flush=True)
    print(f"[scan] tier A={n_a} B={n_b} C={n_c} | "
          f"tradeable_a={n_trd_a} tradeable_ab={n_trd_ab}", flush=True)
    print(f"[scan] wrote: {out_csv.name}, {out_parquet.name}, {out_csv_latest.name}",
          flush=True)
    return today


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="YYYY-MM-DD (default = max bar date in master)")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof).normalize() if args.asof else None
    scan(asof=asof, out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
