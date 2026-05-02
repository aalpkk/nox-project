"""V1.2.4 horizontal_base — per-gate rejection diagnostic.

For every "candidate bar" (one that has a valid squeeze base behind it AND is
within the actionable post-squeeze window AND passes structural validations),
evaluate each trigger-gate component INDEPENDENTLY (not short-circuited) and
tally pass/fail counts.

Goal: identify which gate is the binding constraint on the trigger pool.

Output: per-gate pass-rate, plus joint marginals (e.g. how many pass all
EXCEPT vol; how many pass all EXCEPT range_pos).
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
    BREAKOUT_BUFFER,
    IMPULSE_ATR_MULT,
    MAX_EXTENDED_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    TRIG_RANGE_POS_MIN,
    TRIG_VOL_RATIO_MIN,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)


GATE_NAMES = [
    "breakout",     # close > robust * (1 + buffer)
    "hard",         # high >= hard_resistance
    "up_day",       # close > open
    "range_pos",    # range_pos >= 0.60
    "body",         # body > atr_sq * IMPULSE_ATR_MULT
    "vol",          # vol > vol_sma * TRIG_VOL_RATIO_MIN
    "sma20",        # close > SMA20
    "vwap",         # close > VWAP20 OR close > VWAP60
]


def gate_eval(df: pd.DataFrame, idx: int, robust: float, hard: float) -> dict:
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
        return None  # candidate not well-defined
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    return {
        "breakout":  c > robust * (1.0 + BREAKOUT_BUFFER),
        "hard":      h >= hard,
        "up_day":    c > o,
        "range_pos": range_pos >= TRIG_RANGE_POS_MIN,
        "body":      body > atr * IMPULSE_ATR_MULT,
        "vol":       vol > vol_sma * TRIG_VOL_RATIO_MIN,
        "sma20":     c > sma20,
        "vwap":      (vwap20 > 0 and c > vwap20) or (vwap60 > 0 and c > vwap60),
    }


def candidates_for_ticker(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """For one ticker: enumerate every candidate (asof_idx where structural
    pipeline survives), evaluate gates."""
    if len(daily_df) < BB_LENGTH * 3:
        return pd.DataFrame()
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return pd.DataFrame()

    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])

    rows = []
    n = len(df)
    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        # actionable window: post-squeeze, within MAX_EXTENDED_AGE_BARS
        if asof_idx <= sq_e:
            continue
        if asof_idx - sq_e > MAX_EXTENDED_AGE_BARS:
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

        gates = gate_eval(df, asof_idx, box_top, hard)
        if gates is None:
            continue
        gates["ticker"] = ticker
        gates["date"] = df.index[asof_idx]
        gates["age_after_squeeze"] = int(asof_idx - sq_e)
        rows.append(gates)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(
        tickers=None, start=None, end=None, min_coverage=0.0,
    )
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in {time.time()-t0:.1f}s", flush=True)

    print("[2/3] resampling to daily …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    print("[3/3] per-ticker gate diagnostic …", flush=True)
    t2 = time.time()
    all_rows = []
    tickers = daily["ticker"].unique()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        sub = candidates_for_ticker(sub_df, str(t))
        if not sub.empty:
            all_rows.append(sub)
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_rows:
        print("!! no candidates found", flush=True)
        return 1
    cands = pd.concat(all_rows, ignore_index=True)
    n = len(cands)
    print()
    print("=" * 78)
    print("PER-GATE REJECTION DIAGNOSTIC (V1.2.4 horizontal_base)")
    print("=" * 78)
    print(f"candidates (post structural-pipeline): N = {n:,}")
    print(f"  age=0 (sq_e+1): {(cands['age_after_squeeze']==1).sum():,}  "
          f"age≤5: {(cands['age_after_squeeze']<=5).sum():,}  "
          f"age≤10: {(cands['age_after_squeeze']<=10).sum():,}")
    print()

    # 1) Per-gate independent pass rate
    print("[A] independent pass rate (each gate evaluated alone on N candidates):")
    print(f"    {'gate':<12s} {'pass':>10s} {'rate':>8s}")
    rates = {}
    for g in GATE_NAMES:
        passed = int(cands[g].sum())
        rate = passed / n
        rates[g] = rate
        print(f"    {g:<12s} {passed:>10,d} {rate*100:>7.1f}%")

    # 2) "All-but-one" — how many candidates pass every gate EXCEPT the named one
    print()
    print("[B] all-but-one analysis (pass every gate EXCEPT the named one — i.e.")
    print("    the named gate is solely responsible for blocking the trigger):")
    print(f"    {'blocking gate':<14s} {'count':>8s} {'% of N':>8s} "
          f"{'% of misses':>13s}")
    miss_total = (~cands[GATE_NAMES].all(axis=1)).sum()
    for g in GATE_NAMES:
        others = [x for x in GATE_NAMES if x != g]
        only_g_blocks = (cands[others].all(axis=1) & ~cands[g]).sum()
        print(f"    {g:<14s} {only_g_blocks:>8,d} {only_g_blocks/n*100:>7.2f}% "
              f"{only_g_blocks/max(1,miss_total)*100:>12.1f}%")

    # 3) Pass-all (= would fire as trigger). Should equal historical trigger count.
    pass_all = int(cands[GATE_NAMES].all(axis=1).sum())
    print()
    print(f"[C] pass-all (would fire as trigger): {pass_all:,}  "
          f"({pass_all/n*100:.1f}% of N)")
    print(f"    expected ≈ historical trigger emissions (V1.2.4 ≈ 1,408)")

    # 4) Pair-wise blocker — which 2-gate combos block most often
    print()
    print("[D] top 2-gate blocker pairs (both gates fail simultaneously, others may also):")
    pair_counts = []
    for i_, g1 in enumerate(GATE_NAMES):
        for g2 in GATE_NAMES[i_+1:]:
            both_fail = (~cands[g1] & ~cands[g2]).sum()
            pair_counts.append((g1, g2, int(both_fail)))
    pair_counts.sort(key=lambda x: -x[2])
    for g1, g2, c in pair_counts[:8]:
        print(f"    {g1+'/'+g2:<22s} {c:>8,d}  ({c/n*100:.1f}% of N)")

    # 5) Marginal lift if each gate is dropped (all-but-one becomes a pass)
    print()
    print("[E] hypothetical pool size if a single gate is removed:")
    print(f"    baseline triggers (all 8 gates): {pass_all:,}")
    for g in GATE_NAMES:
        others = [x for x in GATE_NAMES if x != g]
        new_n = int(cands[others].all(axis=1).sum())
        delta = new_n - pass_all
        delta_pct = delta / max(1, pass_all) * 100
        print(f"    drop {g:<10s}: {new_n:>6,d}  (+{delta:,}  +{delta_pct:.1f}%)")

    out_path = "output/scanner_v1_gate_diagnostic.csv"
    summary = pd.DataFrame({
        "gate": GATE_NAMES,
        "indep_pass_rate": [rates[g] for g in GATE_NAMES],
        "indep_pass_count": [int(cands[g].sum()) for g in GATE_NAMES],
        "candidates_total": n,
    })
    summary.to_csv(out_path, index=False)
    print(f"\nwrote: {out_path}")
    print(f"total elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
