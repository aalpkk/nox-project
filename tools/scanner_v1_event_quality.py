"""V1.3.1 event-quality diagnostic — cohort × horizon report.

Pre-ML-freeze cohort map under live production gates. Not a gate test.

Cohorts (rows):
    trigger_overall
    trigger_mid_body         (body_atr ∈ [0.35, 0.65))
    trigger_strict_body      (body_atr ∈ [0.65, 1.05))
    trigger_large_body       (body_atr ≥ 1.05)
    retest_bounce_overall
    retest_deep_touch        (retest_depth_atr ≥ 0.30)
    retest_shallow_touch     (retest_depth_atr ∈ [0.0, 0.30))
    retest_no_touch          (retest_depth_atr < 0.0 — drift-up reclaim)
    extended_reference

Horizons:
    3d / 5d / 10d / 20d

Metrics per cell:
    N, R_mean, MFE_R_mean, MAE_R_mean, win%, PF, EF%, FB%, time_to_MFE

Audits:
    [F] year × cohort stability (R_5d / PF_5d / N)
    [G] ticker / date concentration

Forward labels:
    entry        = close at asof_idx (the bar emitting the row)
    R_per_share  = entry - (box_bot - 0.30 * atr_sq_at_asof)
    fwd window   = df.iloc[asof_idx + 1 : asof_idx + h + 1]
    EF           = MAE_5d ≤ -1R
    FB           = within fwd window, close < box_top - 0.20 * atr_sq
    time_to_MFE  = bars from asof to MFE bar (1..h); NaN if window empty

Same forward-label semantics applied to all cohorts including extended,
which makes "extended_reference" interpretable as 'what would happen if
you re-entered an already-extended bar today' — useful as control.
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

HORIZONS = [3, 5, 10, 20]
FAILED_BREAK_BUFFER_ATR = 0.20
OUT_CSV = "output/scanner_v1_event_quality_events.csv"


def _classify_body(body_atr: float) -> str:
    if not (body_atr == body_atr):
        return ""
    if body_atr >= BODY_LARGE_THRESH:
        return "large_body"
    if body_atr >= BODY_STRICT_THRESH:
        return "strict_body"
    if body_atr >= IMPULSE_ATR_MULT:
        return "mid_body"
    return ""


def _classify_retest(depth_atr: float) -> str:
    if not (depth_atr == depth_atr):
        return ""
    if depth_atr >= 0.30:
        return "deep_touch"
    if depth_atr >= 0.0:
        return "shallow_touch"
    return "no_touch"


def _forward_labels(
    df: pd.DataFrame,
    asof_idx: int,
    entry: float,
    R_per_share: float,
    box_top: float,
    atr_sq: float,
) -> dict:
    n = len(df)
    out: dict = {}
    below_threshold = box_top - FAILED_BREAK_BUFFER_ATR * atr_sq
    for h in HORIZONS:
        end_h = min(asof_idx + h, n - 1)
        if end_h <= asof_idx:
            out[f"mfe_R_{h}d"] = float("nan")
            out[f"mae_R_{h}d"] = float("nan")
            out[f"realized_R_{h}d"] = float("nan")
            out[f"failed_breakout_{h}d"] = False
            out[f"time_to_MFE_{h}d"] = float("nan")
            continue
        fwd = df.iloc[asof_idx + 1: end_h + 1]
        highs = fwd["high"].values
        lows = fwd["low"].values
        mh = float(highs.max())
        ml = float(lows.min())
        ch = float(fwd["close"].iloc[-1])
        mfe_idx_local = int(np.argmax(highs))  # 0-based within fwd
        out[f"mfe_R_{h}d"] = (mh - entry) / R_per_share
        out[f"mae_R_{h}d"] = (ml - entry) / R_per_share
        out[f"realized_R_{h}d"] = (ch - entry) / R_per_share
        out[f"failed_breakout_{h}d"] = bool((fwd["close"] < below_threshold).any())
        out[f"time_to_MFE_{h}d"] = float(mfe_idx_local + 1)  # 1..h
    mae5 = out.get("mae_R_5d")
    out["early_failure_5d"] = bool(mae5 <= -1.0) if pd.notna(mae5) else False
    return out


def _retest_depth_atr(df: pd.DataFrame, breakout_idx: int, asof_idx: int,
                      box_top: float) -> float:
    if breakout_idx is None or asof_idx <= breakout_idx:
        return float("nan")
    atr_t2 = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
    if atr_t2 <= 0:
        return float("nan")
    deepest_low = float("inf")
    for j in range(breakout_idx + 1, asof_idx + 1):
        l_j = float(df["low"].iat[j])
        if l_j < deepest_low:
            deepest_low = l_j
    if not np.isfinite(deepest_low):
        return float("nan")
    return (box_top - deepest_low) / atr_t2


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
        if state == "none" or state == "pre_breakout":
            continue

        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_sq <= 0:
            continue
        entry = asof_close
        invalidation = box_bot - ATR_SL_MULT * atr_sq
        R_per_share = entry - invalidation
        if R_per_share <= 0:
            continue

        # body_class on T1 (breakout bar)
        body_class = ""
        body_atr_val = float("nan")
        if breakout_idx is not None:
            t1_atr = float(df["atr_sq"].iat[breakout_idx]) if pd.notna(df["atr_sq"].iat[breakout_idx]) else 0.0
            if t1_atr > 0:
                t1_body = abs(float(df["close"].iat[breakout_idx]) - float(df["open"].iat[breakout_idx]))
                body_atr_val = t1_body / t1_atr
                body_class = _classify_body(body_atr_val)

        retest_kind = ""
        retest_depth = float("nan")
        if state == "retest_bounce":
            retest_depth = _retest_depth_atr(df, breakout_idx, asof_idx, box_top)
            retest_kind = _classify_retest(retest_depth)

        row = {
            "ticker": ticker,
            "date": df.index[asof_idx],
            "state": state,
            "body_class": body_class,
            "body_atr": float(body_atr_val) if pd.notna(body_atr_val) else float("nan"),
            "retest_kind": retest_kind,
            "retest_depth_atr": float(retest_depth) if pd.notna(retest_depth) else float("nan"),
            "asof_idx": int(asof_idx),
            "breakout_idx": int(breakout_idx) if breakout_idx is not None else -1,
            "entry": entry,
            "R_per_share": float(R_per_share),
            "box_top": float(box_top),
            "box_bot": float(box_bot),
            "atr_sq": atr_sq,
        }
        row.update(_forward_labels(df, asof_idx, entry, R_per_share, box_top, atr_sq))
        out.append(row)

    return out


def _summary_row(label: str, sub: pd.DataFrame, h: int) -> dict:
    r = sub[f"realized_R_{h}d"].dropna()
    mfe = sub[f"mfe_R_{h}d"].dropna()
    mae = sub[f"mae_R_{h}d"].dropna()
    pf = (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf")
    ef_col = "early_failure_5d" if h <= 5 else "early_failure_5d"  # only computed at 5d
    fb = sub[f"failed_breakout_{h}d"].mean() * 100 if len(sub) else float("nan")
    ttm = sub[f"time_to_MFE_{h}d"].dropna()
    return {
        "cohort": label,
        "N": len(sub),
        "R_mean": round(r.mean(), 3) if len(r) else float("nan"),
        "MFE_R_mean": round(mfe.mean(), 3) if len(mfe) else float("nan"),
        "MAE_R_mean": round(mae.mean(), 3) if len(mae) else float("nan"),
        "win%": round((r > 0).mean() * 100, 1) if len(r) else float("nan"),
        "PF": round(pf, 2),
        "EF_5d%": round(sub["early_failure_5d"].mean() * 100, 1),  # 5d-anchored, repeated for all h
        "FB%": round(fb, 1),
        "ttMFE": round(ttm.mean(), 2) if len(ttm) else float("nan"),
    }


def _build_cohort_table(events: pd.DataFrame, h: int) -> pd.DataFrame:
    cohorts: list[tuple[str, pd.DataFrame]] = []
    trig = events[events["state"] == "trigger"]
    cohorts.append(("trigger_overall", trig))
    cohorts.append(("  trigger_mid_body",    trig[trig["body_class"] == "mid_body"]))
    cohorts.append(("  trigger_strict_body", trig[trig["body_class"] == "strict_body"]))
    cohorts.append(("  trigger_large_body",  trig[trig["body_class"] == "large_body"]))
    rb = events[events["state"] == "retest_bounce"]
    cohorts.append(("retest_bounce_overall", rb))
    cohorts.append(("  retest_deep_touch",    rb[rb["retest_kind"] == "deep_touch"]))
    cohorts.append(("  retest_shallow_touch", rb[rb["retest_kind"] == "shallow_touch"]))
    cohorts.append(("  retest_no_touch",      rb[rb["retest_kind"] == "no_touch"]))
    cohorts.append(("extended_reference", events[events["state"] == "extended"]))

    rows = [_summary_row(label, sub, h) for label, sub in cohorts]
    return pd.DataFrame(rows).set_index("cohort")


def main() -> int:
    t0 = time.time()
    print(f"V1.3.1 event-quality diagnostic "
          f"(SCANNER={SCANNER_VERSION} FEATURE={FEATURE_VERSION})", flush=True)
    print(f"  body floor={IMPULSE_ATR_MULT}  "
          f"strict_thresh={BODY_STRICT_THRESH}  large_thresh={BODY_LARGE_THRESH}",
          flush=True)

    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in "
          f"{time.time()-t0:.1f}s", flush=True)

    print("[2/3] daily resample …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    print("[3/3] per-ticker event scan + forward labels …", flush=True)
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
    df["year"] = df["date"].dt.year
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df):,} events to {OUT_CSV}\n")

    # ===== A. State totals ===================================================
    print("=" * 92)
    print("[A] STATE / COHORT TOTALS")
    print("=" * 92)
    print(df["state"].value_counts().to_string())
    print()
    print("body_class within trigger:")
    print(df[df["state"] == "trigger"]["body_class"].value_counts().to_string())
    print()
    print("retest_kind within retest_bounce:")
    print(df[df["state"] == "retest_bounce"]["retest_kind"].value_counts().to_string())

    # ===== B-E. Cohort × horizon =============================================
    section_letters = {3: "B", 5: "C", 10: "D", 20: "E"}
    for h in HORIZONS:
        print()
        print("=" * 92)
        print(f"[{section_letters[h]}] COHORT QUALITY @ {h}d HORIZON")
        print("=" * 92)
        tbl = _build_cohort_table(df, h)
        print(tbl.to_string())

    # ===== F. Year × cohort stability ========================================
    print()
    print("=" * 92)
    print("[F] YEAR × COHORT STABILITY  (5d horizon: N / R_mean / PF / win%)")
    print("=" * 92)
    cohort_specs: list[tuple[str, pd.Series]] = [
        ("trigger_mid_body",    (df["state"].eq("trigger") & df["body_class"].eq("mid_body"))),
        ("trigger_strict_body", (df["state"].eq("trigger") & df["body_class"].eq("strict_body"))),
        ("trigger_large_body",  (df["state"].eq("trigger") & df["body_class"].eq("large_body"))),
        ("retest_deep_touch",    (df["state"].eq("retest_bounce") & df["retest_kind"].eq("deep_touch"))),
        ("retest_shallow_touch", (df["state"].eq("retest_bounce") & df["retest_kind"].eq("shallow_touch"))),
        ("retest_no_touch",      (df["state"].eq("retest_bounce") & df["retest_kind"].eq("no_touch"))),
    ]
    years = sorted(df["year"].unique())
    print(f"  {'cohort':<24s}  " +
          "  ".join(f"{y:>22d}" for y in years))
    print(f"  {'':<24s}  " +
          "  ".join(f"{'N/Rm/PF/win%':>22s}" for _ in years))
    for name, mask in cohort_specs:
        sub = df[mask]
        cells = []
        for y in years:
            sy = sub[sub["year"] == y]
            r = sy["realized_R_5d"].dropna()
            n = len(sy)
            if n == 0:
                cells.append(f"{'-':>22s}")
                continue
            rm = r.mean() if len(r) else float("nan")
            pf = (r[r>0].sum() / -r[r<0].sum()) if (r < 0).any() else float("inf")
            wp = (r > 0).mean() * 100 if len(r) else float("nan")
            cells.append(f"{n:>4d}/{rm:>+5.2f}/{pf:>4.2f}/{wp:>4.1f}")
        print(f"  {name:<24s}  " + "  ".join(f"{c:>22s}" for c in cells))

    # ===== G. Concentration =================================================
    print()
    print("=" * 92)
    print("[G] CONCENTRATION AUDIT")
    print("=" * 92)
    trig = df[df["state"] == "trigger"]
    rb   = df[df["state"] == "retest_bounce"]

    print()
    print("  top tickers by trigger count:")
    top_t = trig["ticker"].value_counts().head(15)
    for tk, c in top_t.items():
        pct = c / len(trig) * 100
        print(f"    {tk:<12s} {c:>4d}  ({pct:4.1f}% of triggers)")
    print(f"  total triggers: {len(trig):,}  unique tickers: {trig['ticker'].nunique()}")

    print()
    print("  date concentration — days with most simultaneous triggers:")
    by_day = trig.groupby("date").size().sort_values(ascending=False).head(10)
    for d, c in by_day.items():
        print(f"    {d.date()}  {c:>3d} triggers")
    print(f"  trigger days: {trig['date'].nunique()}  med/day: "
          f"{trig.groupby('date').size().median():.1f}  "
          f"p95/day: {int(trig.groupby('date').size().quantile(0.95))}")

    print()
    print("  retest concentration:")
    top_rb = rb["ticker"].value_counts().head(10)
    for tk, c in top_rb.items():
        print(f"    {tk:<12s} {c:>3d}  ({c/len(rb)*100:4.1f}%)")
    print(f"  total retest_bounce: {len(rb):,}  unique tickers: {rb['ticker'].nunique()}")

    # ===== H. Sanity =========================================================
    print()
    print("=" * 92)
    print("[H] SANITY")
    print("=" * 92)
    valid = df[df["mfe_R_5d"].notna()]
    bad = valid[
        (valid["realized_R_5d"] < valid["mae_R_5d"] - 1e-6) |
        (valid["realized_R_5d"] > valid["mfe_R_5d"] + 1e-6)
    ]
    print(f"  realized_5d outside [MAE_5d, MFE_5d]: {len(bad):,}  (should be 0)")
    print(f"  body_atr min: {trig['body_atr'].min():.3f}  (should be ≥ {IMPULSE_ATR_MULT})")
    print(f"  trigger N: {len(trig):,}  (V1.3.1 expected ≈ 1,413)")
    print(f"  retest N:  {len(rb):,}")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
