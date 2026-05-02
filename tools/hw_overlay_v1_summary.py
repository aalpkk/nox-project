"""Quick sanity summary for HW Overlay v1 outputs.

Reads events parquet + metrics CSV, prints headline stats.
"""
from __future__ import annotations

import pandas as pd

EVENTS = "output/hw_overlay_v1_events.parquet"
METRICS = "output/hw_overlay_v1_metrics.csv"


def main() -> None:
    e = pd.read_parquet(EVENTS)
    m = pd.read_csv(METRICS)

    filled = e[e["entry_filled"]]
    print(f"=== Events ===")
    print(f"  total roster:       {len(e):,}")
    print(f"  filled (entered):   {len(filled):,}  ({len(filled)/len(e):.1%})")
    print(f"  scanners:           {e['scanner'].nunique()}")
    print(f"  cohorts (sc×fam):   {e.groupby(['scanner','family']).ngroups}")
    print()

    print("=== Filled trade outcomes — Arm B (HW overlay) ===")
    print(f"  WR (≥10% & ≤10d):   {filled['is_win'].mean():.1%}")
    print(f"  mean realized R:    {filled['realized_R'].mean():.2%}")
    print(f"  median realized R:  {filled['realized_R'].median():.2%}")
    print(f"  mean holding days:  {filled['holding_days'].mean():.2f}")
    print(f"  exit dist: SAT={int((filled['exit_signal_kind']=='SAT').sum()):,} "
          f"SAT_OB={int((filled['exit_signal_kind']=='SAT_OB').sum()):,} "
          f"time_stop={int((filled['exit_signal_kind']=='time_stop').sum()):,}")
    print(f"  entry kind: AL={int((filled['entry_signal_kind']=='AL').sum()):,} "
          f"AL_OS={int((filled['entry_signal_kind']=='AL_OS').sum()):,}")
    print()

    if "armA_realized_R" in e.columns:
        armA = e["armA_realized_R"].dropna()
        if len(armA) > 0:
            armA_wr = (armA >= 0.10).mean()
            print("=== Arm A (scanner-alone, enter event close, hold 10d) — overall ===")
            print(f"  N (all events):     {len(armA):,}")
            print(f"  WR_A (≥10% in 10d): {armA_wr:.1%}")
            print(f"  mean R_A:           {armA.mean():.2%}")
            print(f"  median R_A:         {armA.median():.2%}")
            print()

    if "armC_realized_R" in e.columns:
        armC = filled["armC_realized_R"].dropna()
        if len(armC) > 0:
            armC_wr = (armC >= 0.10).mean()
            print("=== Arm C (HW entry, +10d fixed exit, NO SAT) — filled trades only ===")
            print(f"  N (filled):         {len(armC):,}")
            print(f"  WR_C (≥10% in 10d): {armC_wr:.1%}")
            print(f"  mean R_C:           {armC.mean():.2%}")
            print(f"  median R_C:         {armC.median():.2%}")
            print()

    if "armD_realized_R" in e.columns:
        armD = filled["armD_realized_R"].dropna()
        if len(armD) > 0:
            armD_wr = (armD >= 0.10).mean()
            print("=== Arm D (multi-cycle in 10d window: AL→enter, SAT→exit, repeat, force-close at +10d) ===")
            print(f"  N (filled):         {len(armD):,}")
            print(f"  WR_D (cumR ≥ +10%): {armD_wr:.1%}")
            print(f"  mean cumR_D:        {armD.mean():.2%}")
            print(f"  median cumR_D:      {armD.median():.2%}")
            print(f"  mean cycles/event:  {filled['armD_n_cycles'].mean():.2f}")
            cyc_dist = filled['armD_n_cycles'].value_counts().sort_index().to_dict()
            print(f"  cycles distribution: {dict(list(cyc_dist.items())[:6])}")
            print()

    pool = m[(m["slice"] == "pooled") & (m["n_traded"] >= 50)].copy()
    cols = ["scanner", "family", "n_traded", "entry_fill_rate",
            "WR", "WR_armC", "WR_armD", "WR_armA_all_events", "WR_armA_aligned",
            "lift_HW_total", "lift_multicycle_vs_B", "lift_multicycle_vs_A_all",
            "lift_HW_filter_aligned", "lift_satexit_vs_fixed",
            "mean_realized_R", "mean_R_armC", "mean_R_armD", "mean_cycles_armD"]

    print("=== Top 12 by WR_armD (multi-cycle, n_traded ≥ 50) ===")
    print(pool.sort_values("WR_armD", ascending=False).head(12)[cols].to_string(index=False))
    print()

    print("=== Top 12 by lift_multicycle_vs_A_all (multi-cycle vs scanner-alone) ===")
    print(pool.sort_values("lift_multicycle_vs_A_all", ascending=False).head(12)[cols].to_string(index=False))
    print()

    print("=== Bottom 8 by lift_multicycle_vs_A_all ===")
    print(pool.sort_values("lift_multicycle_vs_A_all", ascending=True).head(8)[cols].to_string(index=False))
    print()

    print("=== Big-dot (AL_OS) only slices with n_traded ≥ 10 ===")
    big = m[(m["slice"] == "AL_OS_only") & (m["n_traded"] >= 10)].sort_values("lift_HW_total", ascending=False)
    if len(big) == 0:
        print("  no AL_OS-only slices with n ≥ 10")
    else:
        print(big[cols].to_string(index=False))


if __name__ == "__main__":
    main()
