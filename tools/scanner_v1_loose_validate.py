"""squeeze_breakout_loose_v0_diagnostic — full 607-universe validation.

Family B is DIAGNOSTIC, NOT production. This tool decides — via the 5
acceptance criteria locked in design — whether it can graduate to a real
family or whether it stays in research namespace.

Sections:
  [A] State counts (loose-trigger only — diagnostic family is snapshot)
  [B] Strict-overlap split:
       - cohort_A (loose ∩ strict) — should reproduce strict V1.2.9 metrics
       - cohort_B (loose minus strict) — the non-strict additions (the question)
  [C] Width buckets:    ≤15%, 15–25%, 25–35%, >35%
  [D] Slope buckets:    ≤0.15%/d, 0.15–0.30, 0.30–0.50, >0.50%/d
  [E] Strict reject-reason tag frequency on cohort_B
  [F] Head-to-head vs strict trigger / retest_bounce / extended baselines
       (loaded from retest_validate_events.csv if present)
  [G] Lookahead audit + spot-check

Acceptance criteria (all 5 must pass for graduation):
  1. N anlamlı artmalı vs (strict trigger + retest_bounce) baseline.
  2. early_failure_5d strict trigger / retest_bounce'tan felaket kötü olmamalı.
  3. Top bucket (or strict_overlap=True cohort) full-loose'tan belirgin
     iyi ayrışmalı (edge ölçülebilir mi?).
  4. Edge yalnızca kirli bucket'lardan (width>35% / slope>0.50%/d) gelmemeli.
  5. strict_overlap=False cohort'ta da makul performans olmalı.

R definition (constant per event, matches retest_validate):
   invalidation = box_bot - 0.30 * atr_sq
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
    RESISTANCE_ROBUST_QUANTILE,
    SUPPORT_QUANTILE,
    _line_fit,
)
from scanner.triggers.squeeze_breakout_loose import (
    LOOSE_LIQUIDITY_MIN_TL,
    LOOSE_RISK_PCT_MAX,
    _compute_indicators_loose,
    _find_squeeze_runs,
    _is_loose_breakout_bar,
    _strict_audit_eval,
)


HORIZONS = [3, 5, 10, 20]
FAILED_BREAK_BUFFER_ATR = 0.20
RETEST_EVENTS_CSV = "output/scanner_v1_retest_validate_events.csv"
LOOSE_EVENTS_CSV = "output/scanner_v1_loose_validate_events.csv"


def scan_ticker_loose(daily_df: pd.DataFrame, ticker: str) -> list[dict]:
    """Per-ticker loose-trigger snapshot scan with forward labels."""
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators_loose(daily_df)
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
        if asof_idx <= sq_e:
            continue
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue

        sub = df.iloc[sq_s: sq_e + 1]
        hard = float(sub["high"].max())
        box_bot = float(sub["low"].quantile(SUPPORT_QUANTILE))
        box_top_robust = float(sub["high"].quantile(RESISTANCE_ROBUST_QUANTILE))
        if not (hard > box_bot > 0):
            continue

        # Trigger snapshot: asof must be the FIRST loose breakout in (sq_e, asof].
        first_break: int | None = None
        for i in range(sq_e + 1, asof_idx + 1):
            if _is_loose_breakout_bar(df, i, hard):
                first_break = i
                break
        if first_break is None or first_break != asof_idx:
            continue

        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_sq <= 0:
            continue
        entry = float(df["close"].iat[asof_idx])
        if entry <= 0:
            continue
        invalidation = box_bot - ATR_SL_MULT * atr_sq
        R_per_share = entry - invalidation
        if R_per_share <= 0:
            continue

        # Loose family hard gates.
        risk_pct = R_per_share / entry
        if not (0 < risk_pct <= LOOSE_RISK_PCT_MAX):
            continue
        turnover = entry * float(df["volume"].iat[asof_idx])
        if turnover < LOOSE_LIQUIDITY_MIN_TL:
            continue

        # Audit: width + slope + strict-overlap + reject reasons.
        width_pct = (hard - box_bot) / entry
        b_slope, _ = _line_fit(sub["close"].values)
        r_slope, _ = _line_fit(sub["high"].values)
        slope_pct_per_day = max(abs(b_slope), abs(r_slope))

        strict_pass, strict_reasons = _strict_audit_eval(
            df, asof_idx, box_top_robust, hard,
            b_slope, r_slope, width_pct,
        )

        rec = {
            "ticker": ticker,
            "date": df.index[asof_idx],
            "state": "trigger",
            "asof_idx": int(asof_idx),
            "sq_s": int(sq_s),
            "sq_e": int(sq_e),
            "box_top_robust": float(box_top_robust),
            "hard_resistance": float(hard),
            "box_bot": float(box_bot),
            "atr_sq": atr_sq,
            "entry": entry,
            "R_per_share": float(R_per_share),
            "risk_pct": float(risk_pct),
            "turnover": float(turnover),
            "width_pct": float(width_pct),
            "slope_pct_per_day": float(slope_pct_per_day),
            "base_slope": float(b_slope),
            "res_slope": float(r_slope),
            "strict_overlap_flag": bool(strict_pass),
            "strict_reject_reason_tags": ",".join(strict_reasons) if strict_reasons else "",
        }

        below_threshold = hard - FAILED_BREAK_BUFFER_ATR * atr_sq
        for h in HORIZONS:
            end_h = min(asof_idx + h, n - 1)
            if end_h <= asof_idx:
                rec[f"mfe_R_{h}d"] = float("nan")
                rec[f"mae_R_{h}d"] = float("nan")
                rec[f"realized_R_{h}d"] = float("nan")
                rec[f"failed_breakout_{h}d"] = False
                continue
            fwd = df.iloc[asof_idx + 1: end_h + 1]
            mh = float(fwd["high"].max())
            ml = float(fwd["low"].min())
            ch = float(fwd["close"].iloc[-1])
            rec[f"mfe_R_{h}d"] = (mh - entry) / R_per_share
            rec[f"mae_R_{h}d"] = (ml - entry) / R_per_share
            rec[f"realized_R_{h}d"] = (ch - entry) / R_per_share
            rec[f"failed_breakout_{h}d"] = bool((fwd["close"] < below_threshold).any())

        mae5 = rec.get("mae_R_5d")
        rec["early_failure_5d"] = bool(mae5 <= -1.0) if pd.notna(mae5) else False
        out.append(rec)
    return out


def _summary_row(label: str, sub: pd.DataFrame) -> dict:
    r5 = sub["realized_R_5d"].dropna()
    r20 = sub["realized_R_20d"].dropna()
    pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
    pf20 = (r20[r20 > 0].sum() / -r20[r20 < 0].sum()) if (r20 < 0).any() else float("inf")
    return {
        "cohort": label,
        "N": len(sub),
        "R_5d_mean": round(r5.mean(), 3) if len(r5) else float("nan"),
        "R_5d_med": round(r5.median(), 3) if len(r5) else float("nan"),
        "R_20d_mean": round(r20.mean(), 3) if len(r20) else float("nan"),
        "MFE_5d_mean": round(sub["mfe_R_5d"].mean(), 3),
        "MFE_20d_mean": round(sub["mfe_R_20d"].mean(), 3),
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

    print("[3/3] per-ticker loose scan + forward labels …", flush=True)
    t2 = time.time()
    all_events: list[dict] = []
    tickers = daily["ticker"].unique()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        ev = scan_ticker_loose(sub_df, str(t))
        if ev:
            all_events.extend(ev)
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s "
                  f"events_so_far={len(all_events):,}", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_events:
        print("!! no loose events — Family B did not fire on full universe")
        return 1

    df = pd.DataFrame(all_events)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.to_csv(LOOSE_EVENTS_CSV, index=False)
    print(f"\nwrote {len(df):,} loose events to {LOOSE_EVENTS_CSV}\n")

    # ===== A. State counts ===================================================
    print("=" * 78)
    print("[A] LOOSE FAMILY — TOP-LEVEL COUNTS  (3y, 607-universe)")
    print("=" * 78)
    print(f"  N events:            {len(df):,}")
    print(f"  unique tickers:      {df['ticker'].nunique()}")
    print(f"  date range:          {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  trading days:        {df['date'].nunique()}")
    print(f"  events / day median: {df.groupby('date').size().median():.1f}")
    print(f"  events / day mean:   {df.groupby('date').size().mean():.2f}")

    # Reference baselines from retest_validate (if present).
    ref_trigger_n = ref_retest_n = ref_extended_n = None
    ref_trigger_ef = ref_retest_ef = None
    ref_trigger_pf5 = ref_retest_pf5 = None
    ref_df = None
    if Path(RETEST_EVENTS_CSV).exists():
        ref_df = pd.read_csv(RETEST_EVENTS_CSV)
        for state, target in [("trigger", "trigger"), ("retest_bounce", "retest"),
                               ("extended", "extended")]:
            sub = ref_df[ref_df["state"] == state]
            n = len(sub)
            if target == "trigger":
                ref_trigger_n = n
                ref_trigger_ef = sub["early_failure_5d"].mean() * 100 if n else None
                r5 = sub["realized_R_5d"].dropna()
                ref_trigger_pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
            elif target == "retest":
                ref_retest_n = n
                ref_retest_ef = sub["early_failure_5d"].mean() * 100 if n else None
                r5 = sub["realized_R_5d"].dropna()
                ref_retest_pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
            else:
                ref_extended_n = n
        baseline_n = (ref_trigger_n or 0) + (ref_retest_n or 0)
        print(f"\n  reference (strict trigger + retest_bounce): N={baseline_n:,} "
              f"(trigger {ref_trigger_n}, retest {ref_retest_n})")
        if baseline_n > 0:
            uplift = (len(df) - baseline_n) / baseline_n * 100
            print(f"  loose family delta vs reference: {len(df) - baseline_n:+,d} "
                  f"({uplift:+.1f}%)")
    else:
        print(f"\n  (reference {RETEST_EVENTS_CSV} not found — head-to-head [F] skipped)")

    # ===== B. Strict-overlap split ==========================================
    print("\n" + "=" * 78)
    print("[B] STRICT-OVERLAP SPLIT  (loose ∩ strict   vs   loose \\ strict)")
    print("=" * 78)
    rows = [
        _summary_row("ALL_LOOSE",     df),
        _summary_row("strict_overlap=True  (cohort A)", df[df["strict_overlap_flag"]]),
        _summary_row("strict_overlap=False (cohort B)", df[~df["strict_overlap_flag"]]),
    ]
    sumdf = pd.DataFrame(rows).set_index("cohort")
    print(sumdf.to_string())
    n_overlap = int(df["strict_overlap_flag"].sum())
    n_only = len(df) - n_overlap
    print(f"\n  cohort A (loose ∩ strict): {n_overlap:,}  ({n_overlap/len(df)*100:.1f}%)")
    print(f"  cohort B (loose \\ strict): {n_only:,}  ({n_only/len(df)*100:.1f}%)")

    # ===== C. Width buckets =================================================
    print("\n" + "=" * 78)
    print("[C] WIDTH BUCKETS  (channel width / entry close)")
    print("=" * 78)
    width_bins = [0.0, 0.15, 0.25, 0.35, np.inf]
    width_labels = ["≤15%", "15–25%", "25–35%", ">35%"]
    df["width_bucket"] = pd.cut(df["width_pct"], bins=width_bins,
                                 labels=width_labels, include_lowest=True)
    rows = []
    for b in width_labels:
        sub = df[df["width_bucket"] == b]
        if len(sub) == 0:
            continue
        rows.append(_summary_row(f"width {b}", sub))
    print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== D. Slope buckets =================================================
    print("\n" + "=" * 78)
    print("[D] SLOPE BUCKETS  (max(|base_slope|, |res_slope|), per-day fraction)")
    print("=" * 78)
    slope_bins = [0.0, 0.0015, 0.0030, 0.0050, np.inf]  # 0.15%/d, 0.30%/d, 0.50%/d
    slope_labels = ["≤0.15%/d", "0.15–0.30", "0.30–0.50", ">0.50%/d"]
    df["slope_bucket"] = pd.cut(df["slope_pct_per_day"], bins=slope_bins,
                                 labels=slope_labels, include_lowest=True)
    rows = []
    for b in slope_labels:
        sub = df[df["slope_bucket"] == b]
        if len(sub) == 0:
            continue
        rows.append(_summary_row(f"slope {b}", sub))
    print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== E. Reject-reason tag frequency on cohort B =======================
    print("\n" + "=" * 78)
    print("[E] STRICT REJECT-REASON TAG FREQUENCY  (cohort B = loose \\ strict)")
    print("=" * 78)
    cohort_b = df[~df["strict_overlap_flag"]]
    if len(cohort_b) == 0:
        print("  cohort B empty — every loose trigger also passed strict gates.")
    else:
        tag_counts: dict[str, int] = {}
        for s in cohort_b["strict_reject_reason_tags"]:
            if not isinstance(s, str) or not s:
                continue
            for tag in s.split(","):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not tag_counts:
            print("  no reject reasons recorded (unexpected — investigate)")
        else:
            total = len(cohort_b)
            print(f"  cohort B size: {total:,}")
            print(f"  {'tag':<24s}  {'N':>7s}  {'% cohort B':>10s}")
            for tag, c in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {tag:<24s}  {c:>7,d}  {c/total*100:>9.1f}%")

            # Per-tag forward performance: events that fail ONLY this tag.
            print("\n  per-tag PF_5d (events whose strict-reject set CONTAINS this tag):")
            print(f"  {'tag':<24s}  {'N':>5s}  {'R_5d_mean':>10s}  {'PF_5d':>6s}  {'EF%':>5s}")
            for tag, _ in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
                m = cohort_b["strict_reject_reason_tags"].fillna("").str.contains(
                    rf"(^|,){tag}(,|$)", regex=True, na=False
                )
                sub = cohort_b[m]
                if len(sub) == 0:
                    continue
                r5 = sub["realized_R_5d"].dropna()
                pf5 = (r5[r5 > 0].sum() / -r5[r5 < 0].sum()) if (r5 < 0).any() else float("inf")
                print(f"  {tag:<24s}  {len(sub):>5,d}  {r5.mean():>+10.3f}  "
                      f"{pf5:>6.2f}  {sub['early_failure_5d'].mean()*100:>4.1f}")

    # ===== F. Head-to-head vs strict trigger / retest / extended ============
    if ref_df is not None:
        print("\n" + "=" * 78)
        print("[F] HEAD-TO-HEAD  (this loose run vs strict reference cohorts)")
        print("=" * 78)
        rows = []
        rows.append(_summary_row("strict_trigger (ref)",  ref_df[ref_df["state"] == "trigger"]))
        rows.append(_summary_row("retest_bounce (ref)",   ref_df[ref_df["state"] == "retest_bounce"]))
        rows.append(_summary_row("strict_extended (ref)", ref_df[ref_df["state"] == "extended"]))
        rows.append(_summary_row("loose_trigger ALL",     df))
        rows.append(_summary_row("loose ∩ strict",        df[df["strict_overlap_flag"]]))
        rows.append(_summary_row("loose \\ strict",       df[~df["strict_overlap_flag"]]))
        print(pd.DataFrame(rows).set_index("cohort").to_string())

    # ===== G. Lookahead audit ===============================================
    print("\n" + "=" * 78)
    print("[G] LOOKAHEAD AUDIT")
    print("=" * 78)
    print("  loose detection:")
    print("    - first-breakout search range: (sq_e, asof_idx], snapshot = asof_idx only")
    print("    - bar i decision uses: high[i] / low[i] / open[i] / close[i] (intra-bar + close).")
    print("    - close-entry semantics OK; intraday-fill claim NONE.")
    print("\n  forward labels:")
    print("    - window strictly: df.iloc[asof_idx+1 : asof_idx+h+1]")
    print("    - current bar (asof_idx) excluded from MFE/MAE/realized.")
    valid = df[df["mfe_R_5d"].notna()]
    bad = valid[
        (valid["realized_R_5d"] < valid["mae_R_5d"] - 1e-6) |
        (valid["realized_R_5d"] > valid["mfe_R_5d"] + 1e-6)
    ]
    print(f"  spot-check: realized_5d outside [MAE_5d, MFE_5d]: {len(bad):,}  "
          f"(should be 0)")

    # ===== Acceptance summary ===============================================
    print("\n" + "=" * 78)
    print("ACCEPTANCE GATE SUMMARY  (mechanical — final call is the user's)")
    print("=" * 78)
    base_n = (ref_trigger_n or 0) + (ref_retest_n or 0)
    n_loose = len(df)
    cohort_b_df = df[~df["strict_overlap_flag"]]
    r5_b = cohort_b_df["realized_R_5d"].dropna()
    pf5_b = (r5_b[r5_b > 0].sum() / -r5_b[r5_b < 0].sum()) if (r5_b < 0).any() else float("inf")
    ef_b = cohort_b_df["early_failure_5d"].mean() * 100 if len(cohort_b_df) else float("nan")

    crit_msgs = []
    crit_msgs.append(
        f"  1. N delta vs (trigger+retest): {n_loose - base_n:+,d}  "
        f"({(n_loose - base_n)/max(base_n,1)*100:+.1f}%)"
        if base_n else "  1. N delta: reference missing"
    )
    if ref_trigger_ef is not None:
        ef_loose = df["early_failure_5d"].mean() * 100
        crit_msgs.append(
            f"  2. EF_5d  loose={ef_loose:.1f}%  vs strict_trigger={ref_trigger_ef:.1f}% "
            f"retest={ref_retest_ef:.1f}%"
        )
    crit_msgs.append(
        "  3. top-bucket separation: see [C] / [D] — manual judgement"
    )
    # crit 4: edge concentration check — clean buckets share of positives
    clean_mask = (df["width_pct"] <= 0.35) & (df["slope_pct_per_day"] <= 0.0050)
    r5_all = df["realized_R_5d"].dropna()
    r5_clean = df[clean_mask]["realized_R_5d"].dropna()
    pf5_all = (r5_all[r5_all > 0].sum() / -r5_all[r5_all < 0].sum()) if (r5_all < 0).any() else float("inf")
    pf5_clean = (r5_clean[r5_clean > 0].sum() / -r5_clean[r5_clean < 0].sum()) if (r5_clean < 0).any() else float("inf")
    crit_msgs.append(
        f"  4. Clean-bucket (width≤35% AND slope≤0.50%/d) PF_5d={pf5_clean:.2f} "
        f"(N={len(r5_clean):,})  vs ALL PF_5d={pf5_all:.2f} (N={len(r5_all):,})  "
        f"— edge should hold here, not just dirty tails"
    )
    crit_msgs.append(
        f"  5. cohort B (loose \\ strict) PF_5d={pf5_b:.2f} EF_5d={ef_b:.1f}% "
        f"N={len(cohort_b_df):,}"
    )
    print("\n".join(crit_msgs))

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
