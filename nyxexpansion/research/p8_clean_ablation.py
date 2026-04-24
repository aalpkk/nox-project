"""
P8 + aggressive clean overlay — research-only ablation.

Background
  - Prior overlay study (2026-04-23): P8 (two-stage 15%@+1.5ATR + 15%@+3.0ATR)
    remains topline winner. OL_CLEAN validated diagnosis direction on clean
    bucket (median capture +0.099) but realized impact was tiny (+0.03pp).
  - User call: retest the clean-overlay idea with a more aggressive dose on
    top of P8 base. Two interpretations of "P8 + aggressive clean":

      P8_CLEAN_STACK
        Non-clean: P8 (+1.5 ATR 15% → +3.0 ATR 15%)
        Clean:    +0.7 ATR 25% → +1.5 ATR 15% → +3.0 ATR 15%   (3 stages)
        Net clean reduction before final runner: 55% (45% runner)

      P8_CLEAN_REPLACE
        Non-clean: P8 (+1.5 ATR 15% → +3.0 ATR 15%)
        Clean:    +0.7 ATR 25% → +3.0 ATR 15%                    (first P8 stage replaced)
        Net clean reduction: 40% (60% runner)

  - Test against P0 (control) and P8 (current champion).

Mechanics unchanged: 111 locked baseline, 17:30 entry, 15 bps slippage, SWING
exit engine, BE after first partial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nyxexpansion.research.path_type_overlays import (  # noqa: E402
    OverlaySpec,
    _simulate_overlay,
    _load_panel,
    _agg,
)

LOCKED_PATH = Path("output/_locked111_signals.parquet")
OUT_SUMMARY = Path("output/nyxexp_p8_clean_ablation.csv")
OUT_TRADES = Path("output/nyxexp_p8_clean_tradelevel.parquet")

SLIPPAGE_BPS = 15


SPECS: tuple[OverlaySpec, ...] = (
    OverlaySpec(
        id="P0",
        label="CONTROL — swing default (+1.5 ATR / 40% / BE, trail 2.5 ATR)",
        base_partials=(("atr", 1.5, 0.40),),
    ),
    OverlaySpec(
        id="P8",
        label="P8 two-stage (15% @+1.5 ATR + 15% @+3.0 ATR, BE)",
        base_partials=(("atr", 1.5, 0.15), ("atr", 3.0, 0.15)),
    ),
    OverlaySpec(
        id="P8_CLEAN_STACK",
        label="P8 everywhere + clean extra +0.7 ATR 25% (clean: 25+15+15 = 55%, 45% runner)",
        base_partials=(("atr", 1.5, 0.15), ("atr", 3.0, 0.15)),
        clean_override_partials=(
            ("atr", 0.7, 0.25),
            ("atr", 1.5, 0.15),
            ("atr", 3.0, 0.15),
        ),
    ),
    OverlaySpec(
        id="P8_CLEAN_REPLACE",
        label="Clean: +0.7 ATR 25% → +3.0 ATR 15% (40%, 60% runner); non-clean: P8",
        base_partials=(("atr", 1.5, 0.15), ("atr", 3.0, 0.15)),
        clean_override_partials=(
            ("atr", 0.7, 0.25),
            ("atr", 3.0, 0.15),
        ),
    ),
)


def run():
    locked = pd.read_parquet(LOCKED_PATH)
    panel = _load_panel(locked["ticker"].unique())
    print(f"Loaded locked baseline: {len(locked)} signals")
    print(f"Bucket dist: {locked.risk_bucket.value_counts().to_dict()}")
    print()

    rows = []
    for spec in SPECS:
        for _, r in locked.iterrows():
            t = r.ticker
            d = pd.to_datetime(r.signal_date)
            if t not in panel:
                continue
            df_t, atr_t = panel[t]
            matches = df_t.index[df_t["Date"] == d]
            if len(matches) == 0:
                continue
            entry_idx = int(matches[0])
            if entry_idx >= len(df_t) - 1:
                continue
            ctx = {"risk_bucket": r.risk_bucket, "upside_room_52w_atr": r.upside_room_52w_atr}
            sim = _simulate_overlay(df_t, entry_idx, atr_t, spec, ctx)

            adj = float(r.price_18_00) / float(r.price_17_30) if r.price_17_30 > 0 else 1.0
            ret17 = ((1 + sim["blended_gross_ret_pct"] / 100.0) * adj - 1.0) * 100.0
            mfe17 = ((1 + sim["mfe_close_pct"] / 100.0) * adj - 1.0) * 100.0
            mae17 = ((1 + sim["mae_close_pct"] / 100.0) * adj - 1.0) * 100.0
            ret_net = ret17 - SLIPPAGE_BPS / 100.0

            rows.append({
                "spec_id": spec.id,
                "spec_label": spec.label,
                "ticker": t,
                "signal_date": r.signal_date,
                "fold": r.fold,
                "risk_bucket": r.risk_bucket,
                "reason": sim["reason"],
                "bars_held": sim["bars_held"],
                "partial_count": sim["partial_count"],
                "partial_keys": ",".join(sim["partial_keys"]) if sim["partial_keys"] else "",
                "ret_17_30_net_pct": ret_net,
                "mfe_17_30_pct": mfe17,
                "mae_17_30_pct": mae17,
                "giveback_pct": mfe17 - ret_net,
            })

    df = pd.DataFrame(rows)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_TRADES, index=False)
    print(f"✅ trade-level output: {OUT_TRADES} ({len(df)} rows)")

    # ── Spec summary ────────────────────────────────────────────────────
    summ_rows = []
    for spec in SPECS:
        g = df[df.spec_id == spec.id]
        row = {"spec_id": spec.id, "spec_label": spec.label}
        row.update(_agg(g))
        summ_rows.append(row)
    summary = pd.DataFrame(summ_rows)

    p0 = summary.loc[summary.spec_id == "P0"].iloc[0]
    p8 = summary.loc[summary.spec_id == "P8"].iloc[0]
    for col in ["PF", "avg%", "WR%", "MaxDD%", "total%", "avg_giveback%", "winner_capture_med"]:
        base_p0 = p0[col]
        base_p8 = p8[col]
        summary["Δ_P0_" + col] = summary[col].apply(
            lambda v: round(float(v) - float(base_p0), 3)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v))
               and base_p0 is not None and isinstance(base_p0, (int, float)) and np.isfinite(float(base_p0))
            else None
        )
        summary["Δ_P8_" + col] = summary[col].apply(
            lambda v: round(float(v) - float(base_p8), 3)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v))
               and base_p8 is not None and isinstance(base_p8, (int, float)) and np.isfinite(float(base_p8))
            else None
        )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"✅ summary: {OUT_SUMMARY}")
    print()

    # ── Console: overall ────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("═" * 140)
    print("P8 + AGGRESSIVE CLEAN — OVERALL (111 trades, 17:30 entry @ 15 bps)")
    print("═" * 140)
    cols = ["spec_id", "N", "PF", "avg%", "WR%", "MaxDD%", "total%", "avg_mfe%",
            "avg_giveback%", "winner_capture_med", "partial_share%",
            "Δ_P8_PF", "Δ_P8_MaxDD%", "Δ_P8_total%"]
    print(summary[cols].to_string(index=False))
    print()

    # ── Per-bucket breakdown ────────────────────────────────────────────
    print("═" * 120)
    print("PER-BUCKET IMPACT (key cohort: clean — the one overlay actually touches)")
    print("═" * 120)
    df["winner_capture"] = np.where(
        df["mfe_17_30_pct"] > 0, df["ret_17_30_net_pct"] / df["mfe_17_30_pct"], np.nan
    )
    for bucket in ["clean", "mild", "elevated"]:
        sub = df[df["risk_bucket"] == bucket]
        n = len(sub[sub.spec_id == "P0"])
        print(f"\n[{bucket}]  N={n}")
        print(f"  {'spec':<20} {'avg_real%':>10} {'avg_gb%':>9} {'avg_mfe%':>9} {'PF':>6} {'WR%':>5} {'win_cap_med':>12} {'Δreal_vs_P8':>12}")
        p8_real = sub[sub.spec_id == "P8"]["ret_17_30_net_pct"].mean()
        for s in [sp.id for sp in SPECS]:
            srows = sub[sub.spec_id == s]
            if len(srows) == 0:
                continue
            rets = srows["ret_17_30_net_pct"]
            wins = rets[rets > 0].sum()
            loss = rets[rets < 0].sum()
            pf = wins / abs(loss) if loss < 0 else float("inf")
            wr = (rets > 0).mean() * 100
            winners_only = srows[srows["mfe_17_30_pct"] > 0]
            wcap_med = winners_only["winner_capture"].median() if len(winners_only) else float("nan")
            dv8 = rets.mean() - p8_real
            print(f"  {s:<20} {rets.mean():>+9.2f}% {srows['giveback_pct'].mean():>8.2f}% {srows['mfe_17_30_pct'].mean():>8.2f}% {pf:>6.2f} {wr:>5.1f} {wcap_med:>12.3f}  {dv8:>+10.2f}pp")

    # ── Clean top-quartile MFE (inversion cohort) ───────────────────────
    print("\n" + "═" * 120)
    print("CLEAN TOP-QUARTILE MFE — THE INVERSION COHORT")
    print("═" * 120)
    clean = df[df["risk_bucket"] == "clean"]
    mfe_q3 = clean[clean.spec_id == "P0"]["mfe_17_30_pct"].quantile(0.75)
    print(f"clean MFE 75th pct = {mfe_q3:.2f}%")
    print(f"\n  {'spec':<20} {'N':>3} {'avg_real%':>10} {'avg_gb%':>9} {'avg_mfe%':>9} {'avg_cap':>9} {'med_cap':>9}")
    for s in [sp.id for sp in SPECS]:
        srows = clean[(clean.spec_id == s) & (clean["mfe_17_30_pct"] >= mfe_q3)]
        if len(srows) == 0:
            continue
        print(f"  {s:<20} {len(srows):>3} {srows['ret_17_30_net_pct'].mean():>+9.2f}% "
              f"{srows['giveback_pct'].mean():>8.2f}% {srows['mfe_17_30_pct'].mean():>8.2f}% "
              f"{srows['winner_capture'].mean():>9.3f} {srows['winner_capture'].median():>9.3f}")

    # ── Per-fold stability ─────────────────────────────────────────────
    print("\n" + "═" * 120)
    print("PER-FOLD STABILITY (PF and total% per fold for each spec)")
    print("═" * 120)
    for s in [sp.id for sp in SPECS]:
        parts = []
        for f in sorted(df["fold"].unique()):
            sub = df[(df.spec_id == s) & (df["fold"] == f)]
            r = sub["ret_17_30_net_pct"] / 100.0
            pf = (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf")
            eq = (1 + r.fillna(0)).cumprod()
            tot = (eq.iloc[-1] - 1) * 100 if len(eq) else 0
            parts.append(f"fold{f}: PF={pf:>4.2f} tot={tot:>+7.1f}%")
        print(f"  {s:<20}  " + "  ".join(parts))

    # ── Decision check vs P8 ───────────────────────────────────────────
    print("\n" + "═" * 100)
    print("DECISION CHECK vs P8 (not vs P0): do we beat the current champion?")
    print("═" * 100)
    p8_row = summary.loc[summary.spec_id == "P8"].iloc[0]
    for _, row in summary.iterrows():
        if row.spec_id in ("P0", "P8"):
            continue
        d_pf = row["Δ_P8_PF"] or 0
        d_dd = row["Δ_P8_MaxDD%"] or 0
        d_tot = row["Δ_P8_total%"] or 0
        pf_ok = d_pf >= 0
        dd_ok = d_dd >= -1.5  # tighter than P0 gate — we're above a high bar
        tot_ok = d_tot >= 0
        verdict = "PASS" if (pf_ok and dd_ok and tot_ok) else "FAIL"
        flags = "".join(("✓" if x else "✗") for x in (pf_ok, dd_ok, tot_ok))
        print(f"  {row.spec_id:<20}  {verdict}  [PF/DD/tot:{flags}]  ΔPF={d_pf:+.2f}  ΔDD={d_dd:+.1f}pp  Δtot={d_tot:+.1f}pp")


if __name__ == "__main__":
    run()
