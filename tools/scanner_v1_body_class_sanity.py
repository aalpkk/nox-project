"""V1.3.1 horizontal_base post-change sanity — verifies body-class expansion.

Runs production state classification per (ticker, asof) using the V1.3.1
scanner internals (IMPULSE_ATR_MULT = 0.35) and confirms:

  1. Total trigger events ≈ 1,413
       — matches body_ablation A1 cohort exactly (per-asof, first-break = asof)
  2. body_class distribution among triggers:
       mid_body     ≈ 156
       strict_body  ≈ 350   (body_strict_eligible 1257 − retest/ext leakage)
       large_body   ≈ 907
  3. strict-compatible subset (body_class != mid_body) ≈ 1,257
       — matches body_strict_eligible from ablation, i.e. the cohort that
         would have fired under legacy V1.2.9 strict gate.

This is a one-shot acceptance check, not part of the live pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import intraday_1h
from scanner.triggers.horizontal_base import (
    BB_LENGTH,
    BODY_LARGE_THRESH,
    BODY_STRICT_THRESH,
    IMPULSE_ATR_MULT,
    MAX_SQUEEZE_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _classify_state,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)
from scanner.schema import FEATURE_VERSION, SCANNER_VERSION


def _classify_body(body_atr: float) -> str:
    if not (body_atr == body_atr):  # NaN
        return ""
    if body_atr >= BODY_LARGE_THRESH:
        return "large_body"
    if body_atr >= BODY_STRICT_THRESH:
        return "strict_body"
    if body_atr >= IMPULSE_ATR_MULT:
        return "mid_body"
    return ""


def scan_ticker(daily_df: pd.DataFrame, ticker: str) -> list[dict]:
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return []

    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])
    n = len(df)
    cycle_struct: dict = {}
    out: list[dict] = []

    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue
        if pos not in cycle_struct:
            base = df.iloc[sq_s: sq_e + 1]
            b_slope, _ = _line_fit(base["close"].values)
            r_slope, _ = _line_fit(base["high"].values)
            if abs(b_slope) > SLOPE_REJECT_PER_DAY or abs(r_slope) > SLOPE_REJECT_PER_DAY:
                cycle_struct[pos] = None
            else:
                box_top, hard, box_bot = _box_from_squeeze(df, sq_s, sq_e)
                if not (box_top > box_bot > 0):
                    cycle_struct[pos] = None
                else:
                    cycle_struct[pos] = (box_top, hard, box_bot)
        cs = cycle_struct[pos]
        if cs is None:
            continue
        box_top, hard, box_bot = cs

        asof_close = float(df["close"].iat[asof_idx])
        if asof_close <= 0:
            continue
        if (box_top - box_bot) / asof_close > WIDTH_PCT_REJECT:
            continue

        state, breakout_idx = _classify_state(df, asof_idx, sq_s, sq_e, box_top, hard)
        if state == "none":
            continue
        if state != "trigger":
            out.append({"ticker": ticker, "date": df.index[asof_idx],
                        "state": state, "body_atr": float("nan"),
                        "body_class": ""})
            continue

        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        body = abs(float(df["close"].iat[asof_idx]) - float(df["open"].iat[asof_idx]))
        body_atr = body / atr_sq if atr_sq > 0 else float("nan")
        out.append({
            "ticker": ticker,
            "date": df.index[asof_idx],
            "state": "trigger",
            "body_atr": float(body_atr),
            "body_class": _classify_body(body_atr),
        })

    return out


def main() -> int:
    t0 = time.time()
    print(f"V1.3.1 body-class sanity (SCANNER={SCANNER_VERSION} FEATURE={FEATURE_VERSION})", flush=True)
    print(f"  IMPULSE_ATR_MULT={IMPULSE_ATR_MULT}  "
          f"BODY_STRICT_THRESH={BODY_STRICT_THRESH}  "
          f"BODY_LARGE_THRESH={BODY_LARGE_THRESH}", flush=True)

    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in "
          f"{time.time()-t0:.1f}s", flush=True)

    print("[2/3] daily resample …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    print("[3/3] historical state-scan per ticker …", flush=True)
    t2 = time.time()
    all_rows: list[dict] = []
    tickers = daily["ticker"].unique()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        all_rows.extend(scan_ticker(sub_df, str(t)))
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s "
                  f"events_so_far={len(all_rows):,}", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_rows:
        print("!! no events", flush=True)
        return 1

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    state_counts = df["state"].value_counts()
    triggers = df[df["state"] == "trigger"].copy()

    print()
    print("=" * 78)
    print("[A] STATE TOTALS  (V1.3.1 — body floor 0.35)")
    print("=" * 78)
    for s in ("trigger", "retest_bounce", "extended", "pre_breakout"):
        n = int(state_counts.get(s, 0))
        print(f"  {s:<14s} {n:>6,d}")

    print()
    print("=" * 78)
    print("[B] body_class DISTRIBUTION  (state == trigger)")
    print("=" * 78)
    by_class = triggers["body_class"].value_counts()
    for cls in ("mid_body", "strict_body", "large_body"):
        n = int(by_class.get(cls, 0))
        print(f"  {cls:<14s} {n:>6,d}")
    misc = int((~triggers["body_class"].isin(["mid_body", "strict_body", "large_body"])).sum())
    print(f"  {'(other/empty)':<14s} {misc:>6,d}  (should be 0)")

    print()
    print("=" * 78)
    print("[C] ACCEPTANCE GATES  (vs body_ablation A1 cohort)")
    print("=" * 78)
    n_trig = int(state_counts.get("trigger", 0))
    n_mid  = int(by_class.get("mid_body", 0))
    n_str  = int(by_class.get("strict_body", 0))
    n_lrg  = int(by_class.get("large_body", 0))
    n_strict_compat = n_str + n_lrg

    def gate(label: str, got: int, expect: int, tol: int):
        ok = abs(got - expect) <= tol
        flag = "OK" if ok else "!! FAIL"
        print(f"  {label:<48s} got={got:>5,d}  expected≈{expect:,d}  (±{tol})  {flag}")

    gate("trigger N (≈ A1 cohort)",            n_trig,         1413, 5)
    gate("mid_body N",                         n_mid,          156,  5)
    gate("strict_body + large_body (legacy)",  n_strict_compat, 1257, 5)
    gate("large_body N",                       n_lrg,          907,  5)

    print()
    print("=" * 78)
    print("[D] body_atr quantiles (state == trigger)")
    print("=" * 78)
    qs = triggers["body_atr"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    for q, v in qs.items():
        print(f"  q{int(q*100):02d}  {v:.3f}")
    print(f"  min  {triggers['body_atr'].min():.3f}  (must be ≥ {IMPULSE_ATR_MULT})")
    print(f"  max  {triggers['body_atr'].max():.3f}")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
