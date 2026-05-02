"""V1.2.9 retest_bounce — full 607-universe validation.

Acceptance gate before Family B (squeeze_breakout_loose) work begins.

Sections:
  [A] State counts (trigger / retest_bounce / pre / extended) over 3y
  [B] Cycle-level dedup (cycle = ticker + sq_e):
       - events per cycle
       - co-occurrence T1 vs T2 in same cycle
       - orphan retests (T2 without preceding T1 in same cycle)
  [C] Label diagnostic per state:
       - MFE_R / MAE_R / realized_R at H ∈ {3,5,10,20}d
       - early_failure_5d (MAE_5d ≤ -1R)
       - failed_breakout_Hd (close back below box_top - 0.20*atr_sq within H)
       - PF proxy (sum positive realized_R / sum abs negative)
       - win-rate
  [D] Retest_bounce slices:
       - retest_depth_atr buckets
       - breakout_age buckets
       - retest_close_position buckets
       - retest_vol_pattern buckets
  [E] Head-to-head: trigger vs retest_bounce vs extended
  [F] Lookahead audit
       - confirms _find_retest_bounce_idx upper bound = asof_idx (no future)
       - confirms forward labels start at asof_idx+1

R definition (constant per event):
   invalidation = box_bot - 0.30 * atr_sq           (matches scanner)
   R_per_share  = entry - invalidation
   MFE_R_h      = (max_high in (asof, asof+h]   - entry) / R_per_share
   MAE_R_h      = (min_low  in (asof, asof+h]   - entry) / R_per_share
   realized_R_h = (close at min(asof+h, n-1)    - entry) / R_per_share
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
    MAX_SQUEEZE_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _classify_state,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)


HORIZONS = [3, 5, 10, 20]
FAILED_BREAK_BUFFER_ATR = 0.20  # close < box_top - 0.20*atr_sq → decisively back inside


def scan_ticker(daily_df: pd.DataFrame, ticker: str) -> list[dict]:
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return []
    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])

    out: list[dict] = []
    n = len(df)
    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue

        base = df.iloc[sq_s: sq_e + 1]
        b_slope, _ = _line_fit(base["close"].values)
        if abs(b_slope) > SLOPE_REJECT_PER_DAY:
            continue
        r_slope, _ = _line_fit(base["high"].values)
        if abs(r_slope) > SLOPE_REJECT_PER_DAY:
            continue

        box_top, hard, box_bot = _box_from_squeeze(df, sq_s, sq_e)
        if not (box_top > box_bot > 0):
            continue
        c_asof = float(df["close"].iat[asof_idx])
        if c_asof <= 0:
            continue
        width_pct = (box_top - box_bot) / c_asof
        if width_pct > WIDTH_PCT_REJECT:
            continue

        state, breakout_idx = _classify_state(
            df, asof_idx, sq_s, sq_e, box_top, hard,
        )
        if state == "none":
            continue

        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_sq <= 0:
            continue
        entry = c_asof
        invalidation = box_bot - ATR_SL_MULT * atr_sq
        R_per_share = entry - invalidation
        if R_per_share <= 0:
            continue

        breakout_age = int(asof_idx - breakout_idx) if breakout_idx is not None else -1
        retest_depth_atr = float("nan")
        retest_close_position = float("nan")
        retest_vol_pattern = float("nan")
        if state == "retest_bounce" and breakout_idx is not None:
            deepest_low = float("inf")
            for j in range(breakout_idx + 1, asof_idx + 1):
                lj = float(df["low"].iat[j])
                if lj < deepest_low:
                    deepest_low = lj
            if np.isfinite(deepest_low):
                retest_depth_atr = (box_top - deepest_low) / atr_sq
            retest_close_position = (entry - box_top) / atr_sq
            if asof_idx > breakout_idx + 1:
                mid_vol = df["volume"].iloc[breakout_idx + 1: asof_idx].mean()
                if pd.notna(mid_vol) and mid_vol > 0:
                    retest_vol_pattern = float(df["volume"].iat[asof_idx]) / float(mid_vol)

        rec = {
            "ticker": ticker,
            "date": df.index[asof_idx],
            "state": state,
            "asof_idx": int(asof_idx),
            "breakout_idx": int(breakout_idx) if breakout_idx is not None else -1,
            "sq_s": int(sq_s),
            "sq_e": int(sq_e),
            "breakout_age": breakout_age,
            "box_top": float(box_top),
            "box_bot": float(box_bot),
            "atr_sq": atr_sq,
            "entry": entry,
            "R_per_share": float(R_per_share),
            "retest_depth_atr": retest_depth_atr,
            "retest_close_position": retest_close_position,
            "retest_vol_pattern": retest_vol_pattern,
        }

        # Forward labels — strictly (asof_idx, asof_idx+h]; never include current bar.
        below_threshold = box_top - FAILED_BREAK_BUFFER_ATR * atr_sq
        for h in HORIZONS:
            end_h = min(asof_idx + h, n - 1)
            if end_h <= asof_idx:
                rec[f"mfe_R_{h}d"] = float("nan")
                rec[f"mae_R_{h}d"] = float("nan")
                rec[f"realized_R_{h}d"] = float("nan")
                rec[f"failed_breakout_{h}d"] = False
                continue
            sub = df.iloc[asof_idx + 1: end_h + 1]
            mh = float(sub["high"].max())
            ml = float(sub["low"].min())
            ch = float(sub["close"].iloc[-1])
            rec[f"mfe_R_{h}d"] = (mh - entry) / R_per_share
            rec[f"mae_R_{h}d"] = (ml - entry) / R_per_share
            rec[f"realized_R_{h}d"] = (ch - entry) / R_per_share
            rec[f"failed_breakout_{h}d"] = bool((sub["close"] < below_threshold).any())

        mae5 = rec.get("mae_R_5d")
        rec["early_failure_5d"] = bool(mae5 <= -1.0) if pd.notna(mae5) else False

        out.append(rec)
    return out


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

    print("[3/3] per-ticker scan + forward labels …", flush=True)
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
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_events:
        print("!! no events")
        return 1

    df = pd.DataFrame(all_events)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out_csv = "output/scanner_v1_retest_validate_events.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {len(df):,} events to {out_csv}\n")

    # ===== A. State counts ===================================================
    print("=" * 78)
    print("[A] STATE COUNTS  (full 3y, 607-universe)")
    print("=" * 78)
    sc = df["state"].value_counts()
    for state, n in sc.items():
        print(f"  {state:<16s}  {n:>7,d}")
    print(f"\n  unique tickers: {df['ticker'].nunique()}")
    print(f"  date range:     {df['date'].min().date()} → {df['date'].max().date()}")
    n_days = df['date'].nunique()
    print(f"  trading days:   {n_days}")

    # ===== B. Cycle-level dedup ==============================================
    print("\n" + "=" * 78)
    print("[B] CYCLE-LEVEL ANALYSIS  (cycle key = ticker + sq_e)")
    print("=" * 78)
    df["cycle_key"] = df["ticker"] + "__" + df["sq_e"].astype(str)
    n_cycles = df["cycle_key"].nunique()
    events_per_cycle = df.groupby("cycle_key").size()
    print(f"  total cycles touched: {n_cycles:,}")
    print(f"  events / cycle: median={int(events_per_cycle.median())} "
          f"mean={events_per_cycle.mean():.2f} max={int(events_per_cycle.max())}")

    cycles_with_T1 = set(df[df["state"] == "trigger"]["cycle_key"])
    cycles_with_T2 = set(df[df["state"] == "retest_bounce"]["cycle_key"])
    cycles_with_pre = set(df[df["state"] == "pre_breakout"]["cycle_key"])
    cycles_with_ext = set(df[df["state"] == "extended"]["cycle_key"])

    print(f"\n  cycles with T1 (trigger):       {len(cycles_with_T1):>5,}")
    print(f"  cycles with T2 (retest_bounce): {len(cycles_with_T2):>5,}")
    print(f"  cycles with extended:           {len(cycles_with_ext):>5,}")
    print(f"  cycles with pre_breakout:       {len(cycles_with_pre):>5,}")
    print(f"\n  T2 ∩ T1 (T2 with prior T1 same cycle):  {len(cycles_with_T1 & cycles_with_T2):>5,}")
    orphan_T2 = cycles_with_T2 - cycles_with_T1
    print(f"  T2 \\ T1 (orphan retests w/o T1):        {len(orphan_T2):>5,}")
    if orphan_T2:
        # An orphan T2 means: in this cycle, T2 was emitted but no T1. That can
        # happen if T1 fell outside the scan window or the squeeze window cut
        # off T1 entirely. Worth flagging.
        print("    (orphans typically arise when T1 happened before BB_LENGTH*2 "
              "or after the analysis window)")

    # cycles with T2 but the T1 was before our scan start
    multi_event_cycles = events_per_cycle[events_per_cycle > 1]
    print(f"\n  cycles with >1 event:           {len(multi_event_cycles):>5,}")
    print(f"  cycles with >1 event %:         {len(multi_event_cycles)/max(n_cycles,1)*100:>5.1f}%")

    # ===== C. Label diagnostic per state =====================================
    print("\n" + "=" * 78)
    print("[C] LABEL DIAGNOSTIC PER STATE")
    print("=" * 78)
    for state in ["trigger", "retest_bounce", "extended", "pre_breakout"]:
        sub = df[df["state"] == state]
        if len(sub) == 0:
            continue
        print(f"\n  --- state = {state}  (N = {len(sub):,}) ---")
        for h in HORIZONS:
            mfe = sub[f"mfe_R_{h}d"].dropna()
            mae = sub[f"mae_R_{h}d"].dropna()
            real = sub[f"realized_R_{h}d"].dropna()
            fb = sub[f"failed_breakout_{h}d"].dropna()
            print(f"    {h:>2d}d:  MFE_R med={mfe.median():>+5.2f} mean={mfe.mean():>+5.2f}  "
                  f"MAE_R med={mae.median():>+5.2f}  "
                  f"R_realized mean={real.mean():>+5.2f}  "
                  f"failed_break={fb.mean()*100:>4.1f}%")
        ef = sub["early_failure_5d"]
        print(f"    early_failure_5d (MAE_5d ≤ -1R): {ef.mean()*100:.1f}%")
        r5 = sub["realized_R_5d"].dropna()
        pos = r5[r5 > 0].sum()
        neg = -r5[r5 < 0].sum()
        pf5 = pos / neg if neg > 0 else float("inf")
        print(f"    PF proxy (realized_R_5d): {pf5:.2f}  win%={(r5 > 0).mean()*100:.1f}")
        r20 = sub["realized_R_20d"].dropna()
        pos20 = r20[r20 > 0].sum()
        neg20 = -r20[r20 < 0].sum()
        pf20 = pos20 / neg20 if neg20 > 0 else float("inf")
        print(f"    PF proxy (realized_R_20d): {pf20:.2f}  win%={(r20 > 0).mean()*100:.1f}")

    # ===== D. Retest_bounce slices ===========================================
    print("\n" + "=" * 78)
    print("[D] RETEST_BOUNCE SLICES")
    print("=" * 78)
    rb = df[df["state"] == "retest_bounce"].copy()
    if len(rb) == 0:
        print("  !! no retest_bounce events — Family A failed to fire on full universe")
    else:
        # depth bucket: signed positive = low went below box_top
        rb["depth_bucket"] = pd.cut(
            rb["retest_depth_atr"],
            bins=[-np.inf, -0.10, 0.10, 0.30, np.inf],
            labels=["no_touch", "kiss", "shallow", "deep"],
        )
        print("\n  retest_depth_atr buckets (signed: + = below box_top):")
        for b, s in rb.groupby("depth_bucket", observed=False):
            if len(s) == 0:
                continue
            r5 = s["realized_R_5d"].dropna()
            r20 = s["realized_R_20d"].dropna()
            ef = s["early_failure_5d"]
            pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
            print(f"    {str(b):<10s}  N={len(s):>5,d}  R_5d mean={r5.mean():>+5.2f}  "
                  f"R_20d mean={r20.mean():>+5.2f}  PF5={pf5:>4.2f}  EF%={ef.mean()*100:>4.1f}")

        rb["age_bucket"] = pd.cut(rb["breakout_age"], bins=[0, 2, 4, 6, 8],
                                   include_lowest=True)
        print("\n  breakout_age buckets:")
        for b, s in rb.groupby("age_bucket", observed=False):
            if len(s) == 0:
                continue
            r5 = s["realized_R_5d"].dropna()
            ef = s["early_failure_5d"]
            print(f"    age {str(b):<10s}  N={len(s):>5,d}  R_5d mean={r5.mean():>+5.2f}  "
                  f"EF%={ef.mean()*100:>4.1f}")

        rb["cp_bucket"] = pd.cut(
            rb["retest_close_position"],
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["barely(<0.5)", "modest(0.5-1)", "decisive(1-2)", "extended(>2)"],
        )
        print("\n  retest_close_position buckets (close above box_top in ATRs):")
        for b, s in rb.groupby("cp_bucket", observed=False):
            if len(s) == 0:
                continue
            r5 = s["realized_R_5d"].dropna()
            r20 = s["realized_R_20d"].dropna()
            print(f"    {str(b):<18s}  N={len(s):>5,d}  R_5d mean={r5.mean():>+5.2f}  "
                  f"R_20d mean={r20.mean():>+5.2f}")

        rb_vp = rb[rb["retest_vol_pattern"].notna()].copy()
        if len(rb_vp):
            rb_vp["vp_bucket"] = pd.cut(
                rb_vp["retest_vol_pattern"], bins=[0, 1.0, 1.5, 2.5, np.inf],
                labels=["quiet(<1)", "normal(1-1.5)", "spike(1.5-2.5)", "explosive(>2.5)"],
            )
            print("\n  retest_vol_pattern buckets (T2 vol vs pullback mean):")
            for b, s in rb_vp.groupby("vp_bucket", observed=False):
                if len(s) == 0:
                    continue
                r5 = s["realized_R_5d"].dropna()
                ef = s["early_failure_5d"]
                print(f"    {str(b):<18s}  N={len(s):>5,d}  R_5d mean={r5.mean():>+5.2f}  "
                      f"EF%={ef.mean()*100:>4.1f}")

    # ===== E. Head-to-head ===================================================
    print("\n" + "=" * 78)
    print("[E] HEAD-TO-HEAD COMPARISON")
    print("=" * 78)
    rows = []
    for state in ["trigger", "retest_bounce", "extended"]:
        sub = df[df["state"] == state]
        if len(sub) == 0:
            continue
        r5 = sub["realized_R_5d"].dropna()
        r20 = sub["realized_R_20d"].dropna()
        pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
        pf20 = (r20[r20 > 0].sum() / -r20[r20 < 0].sum()) if (r20 < 0).any() else float("inf")
        rows.append({
            "state": state,
            "N": len(sub),
            "R_5d_mean": round(r5.mean(), 3),
            "R_5d_med": round(r5.median(), 3),
            "R_20d_mean": round(r20.mean(), 3),
            "MFE_5d_mean": round(sub["mfe_R_5d"].mean(), 3),
            "MFE_20d_mean": round(sub["mfe_R_20d"].mean(), 3),
            "win5d%": round((r5 > 0).mean() * 100, 1),
            "EF_5d%": round(sub["early_failure_5d"].mean() * 100, 1),
            "FB_5d%": round(sub["failed_breakout_5d"].mean() * 100, 1),
            "PF_5d": round(pf5, 2),
            "PF_20d": round(pf20, 2),
        })
    sumdf = pd.DataFrame(rows).set_index("state")
    print(sumdf.to_string())

    # ===== F. Lookahead audit ================================================
    print("\n" + "=" * 78)
    print("[F] LOOKAHEAD AUDIT  (sanity)")
    print("=" * 78)
    print("  retest detection (_find_retest_bounce_idx):")
    print("    - search range:        breakout_idx+1 .. min(asof_idx, breakout_idx+8)")
    print("    - upper bound:         asof_idx (NEVER > asof_idx)")
    print("    - bar i decision uses: low[i] (intra-bar) + close[i] (bar-close).")
    print("    - both visible at bar-close — daily close-entry semantics OK.")
    print("    - intraday-entry claim: NONE. Do NOT use T2 for intraday-fill backtests.")
    print("\n  forward labels:")
    print("    - window strictly:    df.iloc[asof_idx+1 : asof_idx+h+1]")
    print("    - current bar (asof_idx) excluded from MFE/MAE/realized.")
    invalid = df[df["mfe_R_5d"].notna()]
    # Spot-check: realized_R_5d should be in [MAE_5d, MFE_5d] always.
    bad = invalid[
        (invalid["realized_R_5d"] < invalid["mae_R_5d"] - 1e-6) |
        (invalid["realized_R_5d"] > invalid["mfe_R_5d"] + 1e-6)
    ]
    print(f"  spot-check: realized_5d outside [MAE_5d, MFE_5d]: {len(bad):,}  "
          f"(should be 0)")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
