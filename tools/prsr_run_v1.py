"""PRSR v1 — single pre-registered run.

Spec: prsr/SPEC.md. Frozen. No tweaking after run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prsr import config as C
from prsr import backtest


def primary_filter(trades: pd.DataFrame) -> pd.DataFrame:
    """The primary acceptance pool: open_T1 entries, all PRSR candidates."""
    return trades[(trades["source"] == "prsr") & (trades["entry_mode"] == "open_T1")].copy()


def random_filter(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[(trades["source"] == "random") & (trades["entry_mode"] == "open_T1")].copy()


def acceptance_check(prsr_t1: pd.DataFrame, rand_t1: pd.DataFrame, close_t_only_t1: pd.DataFrame) -> dict:
    """Return per-criterion pass/fail dict + overall verdict."""
    p = backtest.aggregate_metrics(prsr_t1, "ret_primary")
    r = backtest.aggregate_metrics(rand_t1, "ret_primary")
    py = backtest.aggregate_yearly(prsr_t1, "ret_primary")
    ry = backtest.aggregate_yearly(rand_t1, "ret_primary")
    yearly_compare = py.merge(ry, on="year", how="outer", suffixes=("_prsr", "_rand")).fillna({"pf_prsr": 0, "pf_rand": 0})
    yearly_beat = int((yearly_compare["pf_prsr"] > yearly_compare["pf_rand"]).sum())
    yearly_total = int(len(yearly_compare))
    top5 = backtest.top5_date_share(prsr_t1, "ret_primary")

    pf_threshold = max(C.ACCEPT_PF_MIN, r["pf"] + C.ACCEPT_PF_OVER_RANDOM if np.isfinite(r["pf"]) else C.ACCEPT_PF_MIN)
    realized_threshold = (
        r["realized_med"] + C.ACCEPT_REALIZED_MED_OVER_RANDOM_PP
        if np.isfinite(r["realized_med"]) else C.ACCEPT_REALIZED_MED_OVER_RANDOM_PP
    )
    max_dd_threshold = r["max_dd"] if np.isfinite(r["max_dd"]) else float("-inf")

    crit = []
    crit.append({
        "name": "PF_proxy",
        "threshold": f">= {C.ACCEPT_PF_MIN} AND >= random ({r['pf']:.3f}) + {C.ACCEPT_PF_OVER_RANDOM}",
        "computed": f"{p['pf']:.3f}",
        "pass": np.isfinite(p["pf"]) and p["pf"] >= pf_threshold,
    })
    crit.append({
        "name": "realized_med",
        "threshold": f">= random ({r['realized_med']*100:.3f}%) + {C.ACCEPT_REALIZED_MED_OVER_RANDOM_PP*100:.2f}pp",
        "computed": f"{p['realized_med']*100:.3f}%",
        "pass": np.isfinite(p["realized_med"]) and p["realized_med"] >= realized_threshold,
    })
    crit.append({
        "name": "yearly_beat_random",
        "threshold": f">= {C.ACCEPT_YEARLY_BEAT_RANDOM_MIN} of {C.ACCEPT_YEARLY_BEAT_RANDOM_OF}",
        "computed": f"{yearly_beat} of {yearly_total}",
        "pass": yearly_beat >= C.ACCEPT_YEARLY_BEAT_RANDOM_MIN,
    })
    crit.append({
        "name": "top5_date_share",
        "threshold": f"<= {C.ACCEPT_TOP5_DATE_SHARE_MAX}",
        "computed": f"{top5:.3f}" if np.isfinite(top5) else "nan",
        "pass": np.isfinite(top5) and top5 <= C.ACCEPT_TOP5_DATE_SHARE_MAX,
    })
    crit.append({
        "name": "N_primary",
        "threshold": f">= {C.ACCEPT_N_MIN}",
        "computed": f"{p['n']}",
        "pass": p["n"] >= C.ACCEPT_N_MIN,
    })
    crit.append({
        "name": "max_dd",
        "threshold": f">= random ({r['max_dd']:.4f})",
        "computed": f"{p['max_dd']:.4f}",
        "pass": np.isfinite(p["max_dd"]) and p["max_dd"] >= max_dd_threshold,
    })

    overall_pass = all(c["pass"] for c in crit)
    return {
        "criteria": crit,
        "verdict": "PASS" if overall_pass else "REJECTED",
        "prsr_metrics": p,
        "random_metrics": r,
        "yearly_compare": yearly_compare,
        "top5": top5,
    }


def write_report(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    accept: dict,
    out_path: str,
) -> None:
    p = accept["prsr_metrics"]
    r = accept["random_metrics"]
    yc = accept["yearly_compare"]
    crit = accept["criteria"]

    prsr_all = trades[trades["source"] == "prsr"]
    osc_subset = prsr_all[(prsr_all["entry_mode"] == "open_T1") & prsr_all["osc_confirmed"]]
    osc_metrics = backtest.aggregate_metrics(osc_subset, "ret_primary")
    close_t_metrics = backtest.aggregate_metrics(
        prsr_all[prsr_all["entry_mode"] == "close_T"], "ret_primary"
    )
    tp_metrics = backtest.aggregate_metrics(
        prsr_all[prsr_all["entry_mode"] == "open_T1"], "ret_tp"
    )

    n_core_dates = panel[panel["tier"] == "core"]["date"].nunique()
    n_core_tickers = panel[panel["tier"] == "core"]["ticker"].nunique()
    n_candidates = int(panel["candidate"].sum())
    n_fire_dates = panel.loc[panel["candidate"], "date"].nunique()
    n_watchable = panel[panel["tier"] == "watchable"][["ticker", "date"]].drop_duplicates().shape[0]

    lines = []
    L = lines.append
    L(f"# PRSR v1 — Pre-Registered Run-1 Report")
    L("")
    L(f"Verdict: **{accept['verdict']}**")
    L("")
    L("Engine: PRSR v1 — price-structure-first reversal engine.")
    L("Spec: `prsr/SPEC.md` (frozen pre-registration).")
    L("Predecessor: OSCMTRX thread CLOSED_REJECTED 2026-04-30.")
    L("")
    L("## Primary verdict — PRSR all candidates, open_T1, ret_primary")
    L("")
    L("| metric | threshold | computed | pass |")
    L("|---|---|---|---|")
    for c in crit:
        L(f"| {c['name']} | {c['threshold']} | {c['computed']} | {'✓' if c['pass'] else '✗'} |")
    L("")
    L(f"Overall: **{accept['verdict']}**")
    L("")
    L("## Run configuration")
    L("")
    L(f"- Window: {C.START_DATE} → {C.END_DATE}")
    L(f"- Universe: history_bars ≥ {C.HISTORY_BARS_CORE} ∧ median_turnover_60 ≥ q{int(C.LIQUIDITY_QUANTILE*100)}")
    L(f"- Patterns enabled: A (failed breakdown), B (spring/reclaim). C/D disabled.")
    L(f"- Regime: ER60 ≥ q{int(C.ER_QUANTILE*100)} ∧ ATR%20 ≤ q{int(C.ATR_PCT_QUANTILE*100)}")
    L(f"- Entry primary: open_T1 (close_T diagnostic only)")
    L(f"- Initial stop: pattern_low − {C.INITIAL_STOP_ATR_MULT} × ATR20")
    L(f"- Time stop primary: {C.TIME_STOP_PRIMARY} bars (5/20 side analysis)")
    L(f"- Random baseline seed: {C.BASELINE_SEED}")
    L("")
    L("## Universe / candidate counts")
    L("")
    L("| count | value |")
    L("|---|---|")
    L(f"| Tradeable Core unique tickers | {n_core_tickers} |")
    L(f"| Tradeable Core unique dates | {n_core_dates} |")
    L(f"| PRSR candidates (universe ∧ regime ∧ (A∨B)) | {n_candidates} |")
    L(f"| PRSR unique fire dates | {n_fire_dates} |")
    L(f"| Watchable (250–499 bars) (ticker,date) pairs | {n_watchable} |")
    L("")
    L("## Headline metrics — primary pool (open_T1, ret_primary)")
    L("")
    L("| pool | n | PF | realized_med | win_rate | max_dd |")
    L("|---|---|---|---|---|---|")
    L(f"| PRSR all candidates | {p['n']} | {p['pf']:.3f} | {p['realized_med']*100:.3f}% | {p['win_rate']:.3f} | {p['max_dd']:.4f} |")
    L(f"| Random baseline | {r['n']} | {r['pf']:.3f} | {r['realized_med']*100:.3f}% | {r['win_rate']:.3f} | {r['max_dd']:.4f} |")
    L("")
    L("## Per-year (open_T1, ret_primary)")
    L("")
    L("| year | n_prsr | PF_prsr | realized_med_prsr | n_rand | PF_rand | realized_med_rand |")
    L("|---|---|---|---|---|---|---|")
    for _, row in yc.iterrows():
        L(
            f"| {int(row['year'])} | {int(row.get('n_prsr', 0))} | {row.get('pf_prsr', float('nan')):.3f} | "
            f"{row.get('realized_med_prsr', float('nan'))*100:.3f}% | {int(row.get('n_rand', 0))} | "
            f"{row.get('pf_rand', float('nan')):.3f} | {row.get('realized_med_rand', float('nan'))*100:.3f}% |"
        )
    L("")
    L(f"top-5 fire-date PnL share (PRSR primary): **{accept['top5']:.3f}**")
    L("")
    L("## Diagnostic only (NOT primary verdict)")
    L("")
    L("These pools are reported but cannot rescue a failing primary verdict.")
    L("")
    L("| diagnostic pool | n | PF | realized_med | win_rate |")
    L("|---|---|---|---|---|")
    L(f"| osc_confirmed subset (open_T1) | {osc_metrics['n']} | {osc_metrics['pf']:.3f} | {osc_metrics['realized_med']*100:.3f}% | {osc_metrics['win_rate']:.3f} |")
    L(f"| close_T optimistic (PRSR all candidates) | {close_t_metrics['n']} | {close_t_metrics['pf']:.3f} | {close_t_metrics['realized_med']*100:.3f}% | {close_t_metrics['win_rate']:.3f} |")
    L(f"| TP 2R appendix (open_T1, ret_tp) | {tp_metrics['n']} | {tp_metrics['pf']:.3f} | {tp_metrics['realized_med']*100:.3f}% | {tp_metrics['win_rate']:.3f} |")
    L(f"| Watchable IPO list (250–499 bars, no trades) | (ticker,date) pairs: {n_watchable} | — | — | — |")
    L("")
    L("## Spec discipline")
    L("")
    L("- Acceptance computed on open_T1 + PRSR all-candidates only.")
    L("- close_T optimistic, osc_confirmed subset, TP 2R, HWO Down partial: diagnostic / appendix.")
    L("- Watchable tier produced no trades; only watchlist CSV.")
    L("- No post-run parameter tweak. Spec is frozen.")
    L("")
    if accept["verdict"] == "PASS":
        L("## Next allowed step")
        L("")
        L("v1 PASSES. Per spec: v1.1 may enable patterns C and D and re-run with the same acceptance numbers, same window, same baseline.")
    else:
        L("## Next allowed step")
        L("")
        L("v1 REJECTED. Per spec do-not list: do not relax thresholds, do not enable C/D as rescue, do not promote close_T or oscillator subset to primary.")
        L("Allowed: open a new pre-registered thread (different module, different hypothesis, own acceptance set).")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    print("[prsr] building daily panel + features…", flush=True)
    panel = backtest.build_full_panel()
    n_total_pairs = len(panel)
    n_core = int(panel["tier"].eq("core").sum())
    n_watchable = int(panel["tier"].eq("watchable").sum())
    n_ipo = int(panel["tier"].eq("ipo").sum())
    n_candidates = int(panel["candidate"].sum())
    print(f"[prsr] panel rows: {n_total_pairs:,} | core: {n_core:,} | watchable: {n_watchable:,} | ipo: {n_ipo:,}", flush=True)
    print(f"[prsr] candidates: {n_candidates:,}", flush=True)

    print("[prsr] evaluating PRSR candidates…", flush=True)
    prsr_trades = backtest.evaluate_candidates(panel)
    print(f"[prsr] PRSR per-trade rows: {len(prsr_trades):,}", flush=True)

    print("[prsr] building random baseline…", flush=True)
    from prsr.random_baseline import build_random_baseline
    fires = panel.loc[panel["candidate"], ["ticker", "date", "pattern_low", "atr20"]].copy()
    base_trades = build_random_baseline(panel, fires)
    print(f"[prsr] random per-trade rows: {len(base_trades):,}", flush=True)

    all_trades = pd.concat([prsr_trades, base_trades], ignore_index=True)
    all_trades.to_csv(C.OUT_PER_TRADE, index=False)
    base_trades.to_csv(C.OUT_RANDOM, index=False)

    # Watchable list
    watch = panel[panel["tier"] == "watchable"][["ticker", "date", "history_bars"]]
    watch.to_csv(C.OUT_WATCHABLE, index=False)

    # Concentration
    conc = (
        panel.loc[panel["candidate"]]
        .groupby("ticker")
        .size()
        .reset_index(name="fires")
        .sort_values("fires", ascending=False)
    )
    conc.to_csv(C.OUT_CONCENTRATION, index=False)

    # Acceptance
    prsr_t1 = primary_filter(all_trades)
    rand_t1 = random_filter(all_trades)
    close_t_only = all_trades[(all_trades["source"] == "prsr") & (all_trades["entry_mode"] == "close_T")]

    accept = acceptance_check(prsr_t1, rand_t1, close_t_only)

    # Aggregate CSV
    agg_rows = []
    agg_rows.append({"pool": "prsr_open_T1_primary", **backtest.aggregate_metrics(prsr_t1, "ret_primary")})
    agg_rows.append({"pool": "random_open_T1_primary", **backtest.aggregate_metrics(rand_t1, "ret_primary")})
    agg_rows.append({"pool": "prsr_close_T_diagnostic", **backtest.aggregate_metrics(close_t_only, "ret_primary")})
    osc_subset = all_trades[(all_trades["source"] == "prsr") & (all_trades["entry_mode"] == "open_T1") & all_trades["osc_confirmed"]]
    agg_rows.append({"pool": "prsr_osc_confirmed_diagnostic", **backtest.aggregate_metrics(osc_subset, "ret_primary")})
    agg_rows.append({"pool": "prsr_open_T1_tp2r_appendix", **backtest.aggregate_metrics(prsr_t1, "ret_tp")})
    agg_rows.append({"pool": "prsr_open_T1_t5_side", **backtest.aggregate_metrics(prsr_t1, "ret_t5")})
    agg_rows.append({"pool": "prsr_open_T1_t20_side", **backtest.aggregate_metrics(prsr_t1, "ret_t20")})
    pd.DataFrame(agg_rows).to_csv(C.OUT_AGGREGATE, index=False)

    # Drawdown
    dd = backtest.chronological_drawdown(prsr_t1, "ret_primary")
    dd.to_csv(C.OUT_DRAWDOWN, index=False)

    # Osc subset CSV
    osc_subset.to_csv(C.OUT_OSC_SUBSET, index=False)

    # Report
    write_report(panel, all_trades, accept, C.OUT_REPORT)

    elapsed = time.time() - t0
    print("", flush=True)
    print(f"[prsr] done in {elapsed:.1f}s", flush=True)
    print(f"[prsr] verdict: {accept['verdict']}", flush=True)
    print(f"[prsr] PRSR primary: n={accept['prsr_metrics']['n']} PF={accept['prsr_metrics']['pf']:.3f} "
          f"realized_med={accept['prsr_metrics']['realized_med']*100:.3f}%", flush=True)
    print(f"[prsr] RANDOM primary: n={accept['random_metrics']['n']} PF={accept['random_metrics']['pf']:.3f} "
          f"realized_med={accept['random_metrics']['realized_med']*100:.3f}%", flush=True)
    print(f"[prsr] report: {C.OUT_REPORT}", flush=True)
    return 0 if accept["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
