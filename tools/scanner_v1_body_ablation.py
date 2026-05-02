"""Strict horizontal_base body-only ablation: A0 (body≥0.65) vs A1 (body≥0.35).

User directive (2026-04-30) after Family B (squeeze_breakout_loose) validation:
Family B did not graduate, but the loose run surfaced that strict-rejected
events with body_atr ∈ [0.35, 0.65) carry PF_5d 2.06. That is a strict-gate
finding, not a loose-family finding. This tool re-tests the body gate
*inside* strict horizontal_base before any production change is considered.

A0 = current strict V1.2.9 trigger, IMPULSE_ATR_MULT = 0.65
A1 = same gates EXCEPT IMPULSE_ATR_MULT = 0.35
All other gates identical:
  vol≥1.30, range_pos≥0.60, slope≤0.15%/d, width≤25%, BB squeeze 0.65×,
  close > robust*(1+0.005), high >= hard, close>open, close>SMA20,
  close>VWAP20 OR close>VWAP60.

Per cycle (ticker, sq_s, sq_e) we emit at most:
  - one A0 event (first strict-trigger bar after squeeze, body≥0.65)
  - one A1 event (first relaxed-trigger bar, body≥0.35)

A0 ⊂ A1 by bar count (relaxation is monotone), but A0_idx may differ from
A1_idx — relaxed gate can fire EARLIER in the same cycle. Cycle-shift
diagnostic captures this.

Cohort tags within A1:
  body_strict_eligible: body_atr ≥ 0.65 at trigger bar (overlaps A0 set
                        if cycle-shift = 0; differs if shift < 0)
  body_mid_only:        0.35 ≤ body_atr < 0.65 (the newly added)

Reports:
  [A] N delta A0 → A1
  [B] head-to-head: A0 / A1 / body_mid_only / body_strict_eligible (in A1)
  [C] cycle-shift histogram (A1_idx - A0_idx)
  [D] same-cycle paired comparison (cycles with both A0 and A1 events)
  [E] body_atr fine buckets (0.35-0.45, 0.45-0.55, 0.55-0.65, 0.65-0.85, 0.85+)
  [F] per-year stability (2023 / 2024 / 2025 / 2026)
  [G] lookahead audit + spot-check
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
    ATR_SL_MULT,
    BB_LENGTH,
    BREAKOUT_BUFFER,
    MAX_SQUEEZE_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    TRIG_RANGE_POS_MIN,
    TRIG_VOL_RATIO_MIN,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)


HORIZONS = [3, 5, 10, 20]
FAILED_BREAK_BUFFER_ATR = 0.20
A0_BODY_MULT = 0.65   # current strict gate
A1_BODY_MULT = 0.35   # ablation gate
OUT_CSV = "output/scanner_v1_body_ablation_events.csv"


def _is_strict_trigger_bar_at_body(
    df: pd.DataFrame,
    idx: int,
    robust_resistance: float,
    hard_resistance: float,
    body_mult: float,
) -> tuple[bool, float]:
    """Replica of horizontal_base._is_breakout_bar with parameterised body gate.

    Returns (passes, body_atr). body_atr returned even when the bar fails so
    the cohort tagging is consistent.
    """
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    vwap20 = float(df["vwap20"].iat[idx]) if pd.notna(df["vwap20"].iat[idx]) else 0.0
    vwap60 = float(df["vwap60"].iat[idx]) if pd.notna(df["vwap60"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0 or sma20 <= 0:
        return False, float("nan")
    body = abs(c - o)
    body_atr = body / atr
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    vwap_ok = (vwap20 > 0 and c > vwap20) or (vwap60 > 0 and c > vwap60)
    passes = (
        c > robust_resistance * (1.0 + BREAKOUT_BUFFER)
        and h >= hard_resistance
        and c > o
        and range_pos >= TRIG_RANGE_POS_MIN
        and body > atr * body_mult
        and vol > vol_sma * TRIG_VOL_RATIO_MIN
        and c > sma20
        and vwap_ok
    )
    return passes, body_atr


def _forward_labels(
    df: pd.DataFrame,
    trigger_idx: int,
    entry: float,
    R_per_share: float,
    box_top: float,
    atr_sq: float,
) -> dict:
    n = len(df)
    out: dict = {}
    below_threshold = box_top - FAILED_BREAK_BUFFER_ATR * atr_sq
    for h in HORIZONS:
        end_h = min(trigger_idx + h, n - 1)
        if end_h <= trigger_idx:
            out[f"mfe_R_{h}d"] = float("nan")
            out[f"mae_R_{h}d"] = float("nan")
            out[f"realized_R_{h}d"] = float("nan")
            out[f"failed_breakout_{h}d"] = False
            continue
        fwd = df.iloc[trigger_idx + 1: end_h + 1]
        mh = float(fwd["high"].max())
        ml = float(fwd["low"].min())
        ch = float(fwd["close"].iloc[-1])
        out[f"mfe_R_{h}d"] = (mh - entry) / R_per_share
        out[f"mae_R_{h}d"] = (ml - entry) / R_per_share
        out[f"realized_R_{h}d"] = (ch - entry) / R_per_share
        out[f"failed_breakout_{h}d"] = bool((fwd["close"] < below_threshold).any())
    mae5 = out.get("mae_R_5d")
    out["early_failure_5d"] = bool(mae5 <= -1.0) if pd.notna(mae5) else False
    return out


def scan_ticker(daily_df: pd.DataFrame, ticker: str) -> list[dict]:
    """Per-ticker per-asof scan — matches production horizontal_base.detect()
    semantics exactly. For each asof_idx, find the latest squeeze run, run
    structural pre-filters, then under each body threshold check whether asof
    IS the first breakout bar after sq_e.

    Emits at most one A0 row and one A1 row per (ticker, asof). cycle_id ties
    rows from the same squeeze run together for paired analysis.
    """
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return []
    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])
    n = len(df)
    out: list[dict] = []

    # Per-cycle gate cache (slope, box, struct pre-filter result).
    # Key: cycle_id; value: (b_slope, r_slope, box_top, hard, box_bot) or None.
    cycle_struct: dict[int, tuple[float, float, float, float, float] | None] = {}

    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue
        if asof_idx <= sq_e:
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
                    cycle_struct[pos] = (b_slope, r_slope, box_top, hard, box_bot)

        cs = cycle_struct[pos]
        if cs is None:
            continue
        b_slope, r_slope, box_top, hard, box_bot = cs

        # Production width semantics: width / close[asof_idx], NOT close[sq_e].
        asof_close = float(df["close"].iat[asof_idx])
        if asof_close <= 0:
            continue
        width_pct = (box_top - box_bot) / asof_close
        if width_pct > WIDTH_PCT_REJECT:
            continue

        # For each body threshold: find first breakout bar in (sq_e, asof_idx].
        # If first breakout == asof_idx, this asof IS a trigger event.
        cycle_id = f"{ticker}__{sq_s}_{sq_e}"
        for cohort_label, body_mult in (("A0", A0_BODY_MULT), ("A1", A1_BODY_MULT)):
            first_break = None
            body_at_first = float("nan")
            for i in range(sq_e + 1, asof_idx + 1):
                ok, body_atr = _is_strict_trigger_bar_at_body(
                    df, i, box_top, hard, body_mult,
                )
                if ok:
                    first_break = i
                    body_at_first = body_atr
                    break
            if first_break is None or first_break != asof_idx:
                continue

            atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
            if atr_sq <= 0:
                continue
            entry = asof_close
            invalidation = box_bot - ATR_SL_MULT * atr_sq
            R_per_share = entry - invalidation
            if R_per_share <= 0:
                continue

            row = {
                "cohort": cohort_label,
                "cycle_id": cycle_id,
                "ticker": ticker,
                "date": df.index[asof_idx],
                "trigger_idx": int(asof_idx),
                "sq_s": int(sq_s),
                "sq_e": int(sq_e),
                "age_from_sqe": int(asof_idx - sq_e),
                "box_top": float(box_top),
                "box_bot": float(box_bot),
                "atr_sq": atr_sq,
                "entry": entry,
                "R_per_share": float(R_per_share),
                "body_atr": float(body_at_first),
                "width_pct": float(width_pct),
                "base_slope": float(b_slope),
                "res_slope": float(r_slope),
            }
            row.update(_forward_labels(df, asof_idx, entry, R_per_share, box_top, atr_sq))
            out.append(row)

    return out


def _summary_row(label: str, sub: pd.DataFrame) -> dict:
    r5 = sub["realized_R_5d"].dropna()
    r20 = sub["realized_R_20d"].dropna()
    mfe20 = sub["mfe_R_20d"].dropna()
    pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
    pf20 = (r20[r20 > 0].sum() / -r20[r20 < 0].sum()) if (r20 < 0).any() else float("inf")
    return {
        "cohort": label,
        "N": len(sub),
        "R_5d_mean": round(r5.mean(), 3) if len(r5) else float("nan"),
        "R_5d_med": round(r5.median(), 3) if len(r5) else float("nan"),
        "R_20d_mean": round(r20.mean(), 3) if len(r20) else float("nan"),
        "MFE_5d_mean": round(sub["mfe_R_5d"].mean(), 3),
        "MFE_20d_mean": round(mfe20.mean(), 3) if len(mfe20) else float("nan"),
        "win5d%": round((r5 > 0).mean() * 100, 1) if len(r5) else float("nan"),
        "EF_5d%": round(sub["early_failure_5d"].mean() * 100, 1),
        "FB_5d%": round(sub["failed_breakout_5d"].mean() * 100, 1),
        "PF_5d": round(pf5, 2),
        "PF_20d": round(pf20, 2),
    }


def main():
    t0 = time.time()
    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in "
          f"{time.time()-t0:.1f}s", flush=True)

    print("[2/3] daily resample …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    print("[3/3] per-ticker per-asof body ablation scan …", flush=True)
    t2 = time.time()
    all_events: list[dict] = []
    tickers = daily["ticker"].unique()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        ev = scan_ticker(sub_df, str(t))
        if ev:
            all_events.extend(ev)
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s "
                  f"events={len(all_events):,}", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_events:
        print("!! no events")
        return 1

    df = pd.DataFrame(all_events)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["year"] = df["date"].dt.year
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df):,} events to {OUT_CSV}\n")

    a0 = df[df["cohort"] == "A0"].copy()
    a1 = df[df["cohort"] == "A1"].copy()
    a1["body_class"] = np.where(
        a1["body_atr"] >= A0_BODY_MULT, "body_strict_eligible", "body_mid_only"
    )

    # ===== A. N delta =======================================================
    print("=" * 78)
    print("[A] N DELTA  (A0 → A1, per-asof scan, dedup by cycle)")
    print("=" * 78)
    print(f"  A0 (body ≥ 0.65) cycles fired: {len(a0):>5,}")
    print(f"  A1 (body ≥ 0.35) cycles fired: {len(a1):>5,}")
    if len(a0):
        delta = len(a1) - len(a0)
        print(f"  delta:                          {delta:+,d}  ({delta/len(a0)*100:+.1f}%)")
    n_strict_eligible = int((a1["body_class"] == "body_strict_eligible").sum())
    n_mid_only = int((a1["body_class"] == "body_mid_only").sum())
    print(f"\n  within A1:")
    print(f"    body_strict_eligible (≥0.65 at trigger): {n_strict_eligible:>5,}")
    print(f"    body_mid_only (0.35–0.65 at trigger):    {n_mid_only:>5,}")

    # ===== B. Head-to-head =================================================
    print("\n" + "=" * 78)
    print("[B] HEAD-TO-HEAD COHORTS")
    print("=" * 78)
    rows = [
        _summary_row("A0 (body≥0.65)", a0),
        _summary_row("A1 (body≥0.35) ALL", a1),
        _summary_row("A1: body_strict_eligible", a1[a1["body_class"] == "body_strict_eligible"]),
        _summary_row("A1: body_mid_only",        a1[a1["body_class"] == "body_mid_only"]),
    ]
    print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== C. Cycle-shift histogram ========================================
    print("\n" + "=" * 78)
    print("[C] CYCLE-SHIFT HISTOGRAM  (A1_idx - A0_idx, per cycle that fires both)")
    print("=" * 78)
    a0_pivot = a0.set_index("cycle_id")["trigger_idx"]
    a1_pivot = a1.set_index("cycle_id")["trigger_idx"]
    paired_cycles = a0_pivot.index.intersection(a1_pivot.index)
    shifts = (a1_pivot.loc[paired_cycles] - a0_pivot.loc[paired_cycles]).astype(int)
    print(f"  cycles with both A0 and A1 events: {len(shifts):,}")
    print(f"  cycles with A0 only (no A1):       {len(a0) - len(shifts):,}  (should be 0)")
    print(f"  cycles with A1 only (no A0):       {len(a1) - len(shifts):,}  "
          f"(= cycles where strict never fires but relaxed does)")
    if len(shifts):
        print(f"\n  shift = A1_idx - A0_idx (negative = relaxed fires earlier):")
        print(f"    min={shifts.min()}  max={shifts.max()}  "
              f"mean={shifts.mean():+.2f}  median={int(shifts.median())}")
        # Explicit value counts for clarity (avoids pd.cut interval-label confusion).
        vc = shifts.value_counts().sort_index()
        for shift_val, c in vc.items():
            print(f"    shift = {int(shift_val):>+3d}  {c:>5,d}  "
                  f"({c/len(shifts)*100:>5.1f}%)")

    # ===== D. Same-cycle paired comparison =================================
    print("\n" + "=" * 78)
    print("[D] SAME-CYCLE PAIRED COMPARISON  (cycles that fire under BOTH gates)")
    print("=" * 78)
    if len(shifts):
        a0_p = a0[a0["cycle_id"].isin(paired_cycles)].set_index("cycle_id")
        a1_p = a1[a1["cycle_id"].isin(paired_cycles)].set_index("cycle_id")
        rows = []
        for h in HORIZONS:
            d_real = a1_p[f"realized_R_{h}d"] - a0_p[f"realized_R_{h}d"]
            d_mfe  = a1_p[f"mfe_R_{h}d"]      - a0_p[f"mfe_R_{h}d"]
            rows.append({
                "horizon": f"{h}d",
                "Δ realized_R mean":  round(d_real.mean(), 3),
                "Δ MFE_R mean":       round(d_mfe.mean(), 3),
                "A1 better (real)":   int((d_real > 0).sum()),
                "A0 better (real)":   int((d_real < 0).sum()),
                "tied":               int((d_real == 0).sum()),
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n  interpretation: if A1 fires earlier and Δ realized > 0, the "
              "early-fire trade beats the late-fire trade on the SAME cycle —")
        print("  i.e. body_atr in [0.35, 0.65) was fine and the strict gate was "
              "discarding usable signal.")

    # ===== E. body_atr fine buckets (within A1) ============================
    print("\n" + "=" * 78)
    print("[E] body_atr FINE BUCKETS  (A1 events)")
    print("=" * 78)
    bins = [0.35, 0.45, 0.55, 0.65, 0.85, 1.05, np.inf]
    labels = ["0.35–0.45", "0.45–0.55", "0.55–0.65", "0.65–0.85", "0.85–1.05", "≥1.05"]
    a1["body_bucket"] = pd.cut(a1["body_atr"], bins=bins, labels=labels,
                                include_lowest=True)
    rows = []
    for b in labels:
        sub = a1[a1["body_bucket"] == b]
        if len(sub) == 0:
            continue
        rows.append(_summary_row(f"body {b}", sub))
    print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== F. Per-year stability ===========================================
    print("\n" + "=" * 78)
    print("[F] PER-YEAR STABILITY")
    print("=" * 78)
    for yr in sorted(df["year"].unique()):
        rows = []
        for label, sub in (
            ("A0",                        a0[a0["year"] == yr]),
            ("A1 ALL",                    a1[a1["year"] == yr]),
            ("A1: body_strict_eligible",  a1[(a1["year"] == yr) & (a1["body_class"] == "body_strict_eligible")]),
            ("A1: body_mid_only",         a1[(a1["year"] == yr) & (a1["body_class"] == "body_mid_only")]),
        ):
            if len(sub) == 0:
                continue
            rows.append(_summary_row(label, sub))
        if rows:
            print(f"\n  --- year {yr}  (events: A0={len(a0[a0['year']==yr])} "
                  f"A1={len(a1[a1['year']==yr])}) ---")
            print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== G. Lookahead audit ==============================================
    print("\n" + "=" * 78)
    print("[G] LOOKAHEAD AUDIT")
    print("=" * 78)
    print("  trigger detection: per-asof first-break scan in (sq_e, asof_idx], emit only if first_break==asof_idx (production semantics).")
    print("  bar i decision uses: high[i]/low[i]/open[i]/close[i] + atr_sq[i] + "
          "vol_sma[i] + sma20[i] + vwap20[i] + vwap60[i].")
    print("  all bar-i indicators computed from data ≤ i. Daily close-entry safe.")
    print("  forward labels strictly df.iloc[trigger_idx+1 : trigger_idx+h+1].")
    valid = df[df["mfe_R_5d"].notna()]
    bad = valid[
        (valid["realized_R_5d"] < valid["mae_R_5d"] - 1e-6) |
        (valid["realized_R_5d"] > valid["mfe_R_5d"] + 1e-6)
    ]
    print(f"  spot-check: realized_5d outside [MAE_5d, MFE_5d]: {len(bad):,}  "
          f"(should be 0)")

    # cross-check: A0 events here should match retest_validate trigger N=1,353
    print(f"\n  baseline cross-check: A0 N here vs retest_validate trigger N=1,353")
    print(f"    A0 N here: {len(a0):,}")
    if abs(len(a0) - 1353) > 5:
        print(f"    !! divergence > 5 — investigate")
    else:
        print(f"    OK (matches retest_validate per-asof scan within 5)")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
