"""HW Overlay v1 — Arm comparison (C / E / F5 / F10 / G5 / G10).

Reads `output/hw_overlay_v1_events.parquet` and computes per-arm metrics on
filled trades. Each arm shares the Arm B entry (first bullish HW within wait
window). Exit policies differ:
  - C:    fixed +10d hold (NO HW SAT)
  - E:    SAT_OB only (ignore SAT), force-close at +10d
  - F5:   any SAT/SAT_OB gated by running MFE ≥ +5%, force-close at +10d
  - F10:  any SAT/SAT_OB gated by running MFE ≥ +10%, force-close at +10d
  - G5:   SAT_OB only gated by running MFE ≥ +5%, force-close at +10d
  - G10:  SAT_OB only gated by running MFE ≥ +10%, force-close at +10d

For each arm reports: n_trades, win_rate, mean/median R, profit_factor,
mean/median MFE, mean/median giveback, capture mean/median, exit-reason
distribution, top-decile-MFE stats (right-tail capture).

Outputs:
  output/hw_overlay_v1_arm_compare.csv     — overall (one row per arm)
  output/hw_overlay_v1_arm_cohort.csv      — per scanner×family×arm (n≥50)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EVENTS = "output/hw_overlay_v1_events.parquet"
DEFAULT_OVERALL = "output/hw_overlay_v1_arm_compare.csv"
DEFAULT_COHORT = "output/hw_overlay_v1_arm_cohort.csv"

# (arm_key, R-col, MFE-col, exit-kind-col, is-win-col)
ARM_COLS: list[tuple[str, str, str | None, str | None, str]] = [
    # Existing baseline arms in the events parquet
    ("B",   "realized_R",       "mfe_R",       "exit_signal_kind", "is_win"),
    ("C",   "armC_realized_R",  None,          None,               "armC_is_win"),
    ("E",   "armE_realized_R",  "armE_mfe_R",  "armE_exit_kind",   "armE_is_win"),
    ("F5",  "armF5_realized_R", "armF5_mfe_R", "armF5_exit_kind",  "armF5_is_win"),
    ("F10", "armF10_realized_R","armF10_mfe_R","armF10_exit_kind", "armF10_is_win"),
    ("G5",  "armG5_realized_R", "armG5_mfe_R", "armG5_exit_kind",  "armG5_is_win"),
    ("G10", "armG10_realized_R","armG10_mfe_R","armG10_exit_kind", "armG10_is_win"),
]


def _profit_factor(r: np.ndarray) -> float:
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def _arm_metrics(df: pd.DataFrame, arm_key: str, r_col: str,
                 mfe_col: str | None, ek_col: str | None,
                 iw_col: str) -> dict:
    s = df.dropna(subset=[r_col]).copy()
    n = len(s)
    if n == 0:
        return {"arm": arm_key, "n_trades": 0}
    r = s[r_col].astype(float).to_numpy()
    wr = float(s[iw_col].mean()) if iw_col in s.columns else float((r >= 0.10).mean())
    pf = _profit_factor(r)
    out = {
        "arm": arm_key,
        "n_trades": n,
        "win_rate": wr,
        "mean_R": float(r.mean()),
        "median_R": float(np.median(r)),
        "profit_factor": pf,
    }
    # MFE-derived metrics
    if mfe_col is not None and mfe_col in s.columns:
        mfe = s[mfe_col].astype(float).to_numpy()
        out["mean_MFE"] = float(np.nanmean(mfe))
        out["median_MFE"] = float(np.nanmedian(mfe))
        gb = mfe - r
        out["mean_giveback"] = float(np.nanmean(gb))
        out["median_giveback"] = float(np.nanmedian(gb))
        # capture: realized / MFE (skip MFE ≤ 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cap = np.where(mfe > 0, r / mfe, np.nan)
        cap = cap[np.isfinite(cap)]
        out["capture_mean"] = float(np.mean(cap)) if len(cap) > 0 else float("nan")
        out["capture_median"] = float(np.median(cap)) if len(cap) > 0 else float("nan")
        # right-tail: top decile by MFE
        if n >= 10:
            thr = float(np.nanquantile(mfe, 0.90))
            mask = mfe >= thr
            r_top = r[mask]
            mfe_top = mfe[mask]
            if len(r_top) > 0:
                out["topdec_n"] = int(len(r_top))
                out["topdec_mean_R"] = float(np.mean(r_top))
                out["topdec_mean_MFE"] = float(np.mean(mfe_top))
                with np.errstate(divide="ignore", invalid="ignore"):
                    cap_top = np.where(mfe_top > 0, r_top / mfe_top, np.nan)
                cap_top = cap_top[np.isfinite(cap_top)]
                out["topdec_capture_mean"] = float(np.mean(cap_top)) if len(cap_top) > 0 else float("nan")
                out["topdec_giveback_mean"] = float(np.mean(mfe_top - r_top))
            else:
                out["topdec_n"] = 0
        else:
            out["topdec_n"] = 0
    else:
        # Arm C has no MFE/exit-kind tracking; skip those fields
        for k in ("mean_MFE", "median_MFE", "mean_giveback", "median_giveback",
                  "capture_mean", "capture_median",
                  "topdec_n", "topdec_mean_R", "topdec_mean_MFE",
                  "topdec_capture_mean", "topdec_giveback_mean"):
            out[k] = float("nan") if k != "topdec_n" else 0
    # Exit-reason distribution
    if ek_col is not None and ek_col in s.columns:
        vc = s[ek_col].astype(str).value_counts()
        out["n_exit_SAT"] = int(vc.get("SAT", 0))
        out["n_exit_SAT_OB"] = int(vc.get("SAT_OB", 0))
        out["n_exit_force_close"] = int(vc.get("force_close", 0)) + int(vc.get("time_stop", 0))
    elif arm_key == "C":
        out["n_exit_SAT"] = 0
        out["n_exit_SAT_OB"] = 0
        out["n_exit_force_close"] = n  # all C trades exit at +10d
    else:
        out["n_exit_SAT"] = 0
        out["n_exit_SAT_OB"] = 0
        out["n_exit_force_close"] = 0
    return out


def compute_overall(df: pd.DataFrame) -> pd.DataFrame:
    filled = df[df["entry_filled"]].copy()
    rows = [_arm_metrics(filled, k, rc, mc, ec, iw) for (k, rc, mc, ec, iw) in ARM_COLS]
    return pd.DataFrame(rows)


def compute_cohort(df: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    filled = df[df["entry_filled"]].copy()
    rows = []
    for (scanner, family), g in filled.groupby(["scanner", "family"]):
        if len(g) < min_n:
            continue
        for (k, rc, mc, ec, iw) in ARM_COLS:
            m = _arm_metrics(g, k, rc, mc, ec, iw)
            m["scanner"] = scanner
            m["family"] = family
            m["cohort_n_filled"] = len(g)
            rows.append(m)
    return pd.DataFrame(rows)


def _fmt(x, w=8, dec=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"{'NaN':>{w}}"
    if isinstance(x, float):
        return f"{x:>{w}.{dec}f}"
    return f"{x:>{w}}"


def print_overall(over: pd.DataFrame) -> None:
    cols = [
        ("arm", 5, "s"), ("n_trades", 9, "d"),
        ("win_rate", 8, "f"), ("profit_factor", 7, "f"),
        ("mean_R", 8, "f"), ("median_R", 8, "f"),
        ("mean_MFE", 8, "f"), ("median_MFE", 9, "f"),
        ("mean_giveback", 9, "f"), ("median_giveback", 9, "f"),
        ("capture_mean", 8, "f"), ("capture_median", 8, "f"),
        ("topdec_mean_R", 9, "f"), ("topdec_capture_mean", 9, "f"),
        ("topdec_giveback_mean", 9, "f"),
        ("n_exit_SAT", 7, "d"), ("n_exit_SAT_OB", 7, "d"),
        ("n_exit_force_close", 9, "d"),
    ]
    head = "  ".join(f"{c[0]:>{c[1]}}" for c in cols)
    print(head)
    for _, row in over.iterrows():
        cells = []
        for name, w, t in cols:
            v = row.get(name, float("nan"))
            if t == "d":
                try:
                    cells.append(f"{int(v):>{w}d}")
                except (ValueError, TypeError):
                    cells.append(f"{'NaN':>{w}}")
            elif t == "f":
                cells.append(_fmt(v, w=w, dec=4))
            else:
                cells.append(f"{str(v):>{w}}")
        print("  ".join(cells))


def apply_decision_rules(over: pd.DataFrame) -> list[str]:
    """Apply user-locked 8-rule interpretation, return summary lines."""
    by = {r.arm: r for r in over.itertuples(index=False)}
    if "C" not in by or "B" not in by:
        return ["[!] missing baseline arms B or C"]
    out = []
    C = by["C"]; B = by["B"]
    out.append(f"Baselines — B: WR {B.win_rate:.3f}  PF {B.profit_factor:.3f}  meanR {B.mean_R:+.3%}")
    out.append(f"            C: WR {C.win_rate:.3f}  PF {C.profit_factor:.3f}  meanR {C.mean_R:+.3%}")
    out.append("")

    def diff(arm, attr):
        return getattr(by[arm], attr) - getattr(C, attr)

    # Rule 1: any arm > C on PF AND mean_R AND not lower WR (capture without giving up frequency)
    cands_pf_mr = [k for k in ("E", "F5", "F10", "G5", "G10")
                   if (by[k].profit_factor > C.profit_factor)
                   and (by[k].mean_R > C.mean_R)
                   and (by[k].win_rate >= C.win_rate - 0.005)]
    out.append(f"R1 candidates dominating C on PF + mean_R (WR not down >0.5pp): {cands_pf_mr or 'none'}")

    # Rule 2: capture_mean > 0.55 — arm protects realized vs MFE meaningfully
    cands_cap = [k for k in ("E", "F5", "F10", "G5", "G10")
                 if not math.isnan(getattr(by[k], "capture_mean", float("nan")))
                 and getattr(by[k], "capture_mean") > 0.55]
    out.append(f"R2 capture_mean > 0.55: {cands_cap or 'none'}  "
               f"(C capture undefined — fixed exit, no MFE tracking)")

    # Rule 3: top-decile capture_mean > 0.55 means arm specifically rescues right-tail trades
    cands_tdc = [k for k in ("E", "F5", "F10", "G5", "G10")
                 if not math.isnan(getattr(by[k], "topdec_capture_mean", float("nan")))
                 and getattr(by[k], "topdec_capture_mean") > 0.55]
    out.append(f"R3 right-tail capture_mean > 0.55: {cands_tdc or 'none'}")

    # Rule 4: SAT exits should be < 30% of total — gated arms should rarely fire
    cands_gate = []
    for k in ("E", "F5", "F10", "G5", "G10"):
        ek = by[k]
        n_t = ek.n_trades
        if n_t == 0:
            continue
        n_sat_total = ek.n_exit_SAT + ek.n_exit_SAT_OB
        frac = n_sat_total / n_t
        if frac < 0.30:
            cands_gate.append((k, frac))
    out.append(f"R4 (gate firing rate): SAT-trigger frac < 30%: "
               f"{', '.join(f'{k}={f:.1%}' for k,f in cands_gate) or 'none'}")

    # Rule 5: no arm should have lower mean_R than B AND lower WR than C (clear loser)
    losers = [k for k in ("E", "F5", "F10", "G5", "G10")
              if by[k].mean_R < B.mean_R and by[k].win_rate < C.win_rate - 0.005]
    out.append(f"R5 clear losers (mean_R<B AND WR<C): {losers or 'none'}")

    # Rule 6: G > F at same threshold ⇒ SAT_OB carries info that plain SAT doesn't
    pairs = [("F5", "G5"), ("F10", "G10")]
    rs = []
    for f, g in pairs:
        df_pf = by[g].profit_factor - by[f].profit_factor
        df_mr = by[g].mean_R - by[f].mean_R
        rs.append(f"{g}vs{f}: ΔPF={df_pf:+.3f}  ΔmeanR={df_mr:+.3%}")
    out.append(f"R6 SAT_OB-only vs any-bear: {' | '.join(rs)}")

    # Rule 7: F10 > F5 (G10 > G5) ⇒ higher MFE gate is better signal selectivity
    pairs2 = [("F5", "F10"), ("G5", "G10")]
    rs2 = []
    for low, high in pairs2:
        rs2.append(f"{high}vs{low}: ΔPF={by[high].profit_factor-by[low].profit_factor:+.3f}  "
                   f"ΔmeanR={by[high].mean_R-by[low].mean_R:+.3%}  "
                   f"ΔWR={by[high].win_rate-by[low].win_rate:+.3%}")
    out.append(f"R7 higher-MFE gate effect: {' | '.join(rs2)}")

    # Rule 8: at least one arm should have giveback_mean noticeably below C's giveback
    # C has no MFE-tracked giveback — compare mean_giveback across arms; smaller = better
    out.append("R8 giveback ranking (mean_giveback ascending — smaller = arm protects more upside):")
    g_rows = [(k, getattr(by[k], "mean_giveback", float("nan")))
              for k in ("B", "E", "F5", "F10", "G5", "G10")]
    g_rows = [(k, v) for k, v in g_rows if not (isinstance(v, float) and math.isnan(v))]
    g_rows.sort(key=lambda x: x[1])
    out.append("    " + " | ".join(f"{k}: {v:+.3%}" for k, v in g_rows))

    return out


def run(events_path: str = DEFAULT_EVENTS,
        overall_out: str = DEFAULT_OVERALL,
        cohort_out: str = DEFAULT_COHORT) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[load] {events_path}")
    df = pd.read_parquet(events_path)
    n_filled = int(df["entry_filled"].sum())
    print(f"[load] {len(df):,} rows, filled {n_filled:,}")

    over = compute_overall(df)
    coh = compute_cohort(df, min_n=50)

    Path(overall_out).parent.mkdir(parents=True, exist_ok=True)
    over.to_csv(overall_out, index=False)
    coh.to_csv(cohort_out, index=False)
    print(f"[done] overall → {overall_out}")
    print(f"[done] cohort  → {cohort_out}  ({len(coh):,} rows)")

    print()
    print("=== Arm comparison — pooled across all filled trades ===")
    print_overall(over)
    print()
    print("=== Decision-rule interpretation ===")
    for line in apply_decision_rules(over):
        print(line)

    return over, coh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=DEFAULT_EVENTS)
    ap.add_argument("--overall-out", default=DEFAULT_OVERALL)
    ap.add_argument("--cohort-out", default=DEFAULT_COHORT)
    args = ap.parse_args()
    run(args.events, args.overall_out, args.cohort_out)


if __name__ == "__main__":
    main()
