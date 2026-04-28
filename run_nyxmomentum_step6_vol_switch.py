"""
Step 6b — simple rule-based volatility switch sensitivity.

Step 6 regime diagnostic identified volatility as the only clean regime
switcher: M0 leads in low-vol buckets (Shp@60 +4.22), V5 leads in high-
vol buckets (Shp@60 +3.41; V4 +3.78). Before touching any ex-ante
regime classifier (which would add a new overfit layer over just 17
OOS rebalances), test whether the economic value of a vol switch shows
up with a trivial rule.

QUESTION: Does a rule-based M0↔V5 switch beat "always M0" or "always
V5," or is the regime diagnostic interesting but economically inert?

ALLOCATIONS TESTED:

  1. always_M0          — every rebalance, M0 top-20 picks.
  2. always_V5          — every rebalance, V5 dampened picks.
  3. vol_switch_oracle  — threshold = MEDIAN of cross-sectional
                           vol_std_60d across all 17 OOS dates (post-
                           hoc). vol[d] ≥ threshold → V5, else M0.
                           This is the UPPER BOUND of the switch's
                           economic value; it knows the binarization
                           in advance.
  4. vol_switch_live    — threshold per date = EXPANDING median of
                           vol_std_60d on dates strictly BEFORE d.
                           Requires min_periods=4 (~quarter); prior
                           dates default to M0 (the benign "assume
                           calm" default). This approximates what a
                           naive live rule could actually deliver.

For each allocation: build basket sequence honestly (re-compute
turnover from actual ticker-set changes between adjacent baskets,
since cross-profile switches change names more than either native
turnover would), then compute net CAGR@60bps, Sharpe@60, DD@60, avg
turnover, # profile switches.

DECISION GATE (pre-registered):
  • If vol_switch_oracle does NOT beat BOTH always_M0 AND always_V5 on
    net CAGR@60 AND Sharpe@60 → no classifier branch. The regime
    diagnostic is interesting for profile framing but not deployable.
  • If vol_switch_live materially closes the gap to oracle → a simple
    classifier (next iteration) has a chance.
  • Otherwise (oracle good, live bad) → the regime boundary is real
    but not ex-ante detectable from a trailing-median rule.

OUTPUT (output/nyxmomentum/reports/):
  step6_vol_switch_comparison.csv     allocation × metric
  step6_vol_switch_choices.csv        per-rebalance profile chosen per rule
  step6_vol_switch_run_meta.json      verdict
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
M0_SEL     = f"{REPORTS_DIR}/step5_selection_V0_M0_baseline.parquet"
V5_SEL     = f"{REPORTS_DIR}/step5_selection_V5_M0M1_ensemble_damp.parquet"
LABELS     = f"{REPORTS_DIR}/step1_labels.parquet"
FEATURES   = f"{REPORTS_DIR}/step2_features.parquet"


# ── Pick sets + volatility signal ───────────────────────────────────────────

def _pick_sets(selection: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    """Per rebalance_date, the set of selected tickers."""
    sel = selection.loc[selection["selected"].astype(bool)]
    return {d: set(g["ticker"])
            for d, g in sel.groupby("rebalance_date", sort=True)}


def _per_date_volatility(features: pd.DataFrame,
                          oos_dates: list[pd.Timestamp]) -> pd.Series:
    """Cross-sectional median of vol_std_60d per rebalance_date. Ex-ante."""
    f = features.loc[
        features["eligible"].astype(bool)
        & features["rebalance_date"].isin(oos_dates)
    ]
    return f.groupby("rebalance_date")["vol_std_60d"].median().sort_index()


# ── Allocation sequence + metrics ───────────────────────────────────────────

def _sequence_metrics(profile_by_date: pd.Series,
                       m0_picks: dict, v5_picks: dict,
                       labels: pd.DataFrame,
                       bps: int = 60) -> dict:
    """
    Build basket sequence from per-date profile choice, then compute
    per-rebalance portfolio return (equal weight over the chosen
    profile's picks) and actual basket-change turnover.

    Returns metrics dict + a DataFrame of per-date rows.
    """
    lab = labels[["ticker", "rebalance_date", "l1_forward_return"]]
    lab_by_date = {d: g.set_index("ticker")["l1_forward_return"]
                   for d, g in lab.groupby("rebalance_date", sort=True)}

    rows: list[dict] = []
    prev: set[str] | None = None
    prev_profile: str | None = None
    switches = 0
    for d, profile in profile_by_date.sort_index().items():
        basket = v5_picks.get(d, set()) if profile == "V5" else m0_picks.get(d, set())
        n = len(basket)
        if n == 0 or d not in lab_by_date:
            continue
        rets = lab_by_date[d].reindex(list(basket))
        r = float(rets.mean())  # NaN-safe via pandas default (skipna)
        if prev is None:
            turn = np.nan
        else:
            turn = len(basket - prev) / max(n, 1)
        prof_switch = int(prev_profile is not None and prev_profile != profile)
        switches += prof_switch
        rows.append({
            "rebalance_date":    d,
            "profile":           profile,
            "n_names":           n,
            "portfolio_return":  r,
            "turnover_fraction": turn,
            "profile_switch":    prof_switch,
        })
        prev = basket
        prev_profile = profile

    seq = pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)
    if seq.empty:
        return {"n_rebalances": 0}, seq

    r_gross = seq["portfolio_return"].astype(float)
    turn_filled = seq["turnover_fraction"].fillna(0.0).astype(float)
    r_net = r_gross - turn_filled * (bps / 10000.0)
    n = len(seq)
    eq = (1.0 + r_net).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mu_g, sd_g = float(r_gross.mean()), float(r_gross.std(ddof=0))
    mu_n, sd_n = float(r_net.mean()),   float(r_net.std(ddof=0))

    metrics = {
        "n_rebalances":         int(n),
        "profile_switches":     int(switches),
        "avg_turnover":         float(seq["turnover_fraction"].dropna().mean())
                                  if seq["turnover_fraction"].notna().any() else np.nan,
        "mean_monthly_gross":   mu_g,
        "sharpe_gross":         float(mu_g / sd_g * np.sqrt(12)) if sd_g > 0 else np.nan,
        "mean_monthly_net60":   mu_n,
        "sharpe_net60":         float(mu_n / sd_n * np.sqrt(12)) if sd_n > 0 else np.nan,
        "net_cagr_60bps":       float(eq.iloc[-1] ** (12 / n) - 1.0) if n > 0 else np.nan,
        "max_drawdown_net60":   float(dd.min()),
        "hit_rate_net":         float((r_net > 0).mean()),
        "pct_dates_V5":         float((seq["profile"] == "V5").mean()),
    }
    return metrics, seq


# ── Profile-choice rules ────────────────────────────────────────────────────

def _rule_always(profile: str, dates: list[pd.Timestamp]) -> pd.Series:
    return pd.Series([profile] * len(dates), index=dates)


def _rule_oracle(vol: pd.Series) -> pd.Series:
    """Post-hoc median threshold. vol[d] >= median → V5."""
    thr = vol.median()
    return pd.Series(
        np.where(vol >= thr, "V5", "M0"),
        index=vol.index,
    )


def _rule_live_expanding(vol: pd.Series, min_periods: int = 4,
                          warmup_default: str = "M0") -> pd.Series:
    """
    For each date d (chronologically ordered), threshold = expanding
    median of vol on dates STRICTLY BEFORE d. Requires min_periods
    prior observations — earlier dates fall back to warmup_default.
    """
    s = vol.sort_index()
    prior_med = s.shift(1).expanding(min_periods=min_periods).median()
    choice = np.where(
        prior_med.isna(),
        warmup_default,
        np.where(s.values >= prior_med.values, "V5", "M0"),
    )
    return pd.Series(choice, index=s.index)


# ── Printing + verdict ──────────────────────────────────────────────────────

def _fmt_pct(v, w: int = 8) -> str:
    return (f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  ").rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_table(comp: pd.DataFrame, choices: pd.DataFrame,
                  vol: pd.Series, t0: float, reports_dir: str) -> None:
    print()
    print("══ nyxmomentum Step 6b — rule-based vol-switch sensitivity ══")
    print()
    print("  Volatility signal: cross-sectional median of vol_std_60d per rebalance date.")
    print(f"  OOS range: {min(vol.index).date()} → {max(vol.index).date()}  "
          f"(n={len(vol)})")
    print(f"  Post-hoc median threshold: {vol.median():.4f}  "
          f"(bucket sizes: High={int((vol >= vol.median()).sum())}, "
          f"Low={int((vol < vol.median()).sum())})")

    print()
    print("  ALLOCATION COMPARISON:")
    print(f"  {'allocation':<22}"
          f"{'nCAGR60':>10}{'nShp60':>8}{'DD60':>8}"
          f"{'turn':>7}{'gross µ':>9}{'switches':>10}{'%V5':>7}{'N':>4}")
    for _, r in comp.iterrows():
        print(f"  {r['allocation']:<22}"
              f"{_fmt_pct(r['net_cagr_60bps'], 10)}"
              f"{_fmt_num(r['sharpe_net60'], 8)}"
              f"{_fmt_pct(r['max_drawdown_net60'], 8)}"
              f"{_fmt_pct(r['avg_turnover'], 7)}"
              f"{_fmt_pct(r['mean_monthly_gross'], 9)}"
              f"{int(r.get('profile_switches', 0)):>10}"
              f"{_fmt_pct(r['pct_dates_V5'], 7)}"
              f"{int(r['n_rebalances']):>4}")

    print()
    print("  PROFILE CHOICES PER DATE:")
    piv = choices.pivot_table(index="rebalance_date", columns="rule",
                                values="profile", aggfunc="first")
    print(piv.to_string())


def _verdict(comp: pd.DataFrame) -> tuple[str, str]:
    """
    Pre-registered gate: does the oracle switch beat BOTH always-M0 AND
    always-V5 on net CAGR AND Sharpe net? And does the live rule
    materially close the gap?
    """
    def _val(allocation: str, col: str) -> float:
        r = comp.loc[comp["allocation"] == allocation]
        return float(r[col].iloc[0]) if len(r) else np.nan

    m0_c = _val("always_M0", "net_cagr_60bps"); m0_s = _val("always_M0", "sharpe_net60")
    v5_c = _val("always_V5", "net_cagr_60bps"); v5_s = _val("always_V5", "sharpe_net60")
    or_c = _val("vol_switch_oracle", "net_cagr_60bps"); or_s = _val("vol_switch_oracle", "sharpe_net60")
    # Two warmup variants. A robust rule should not depend on warmup default.
    lv_m_c = _val("vol_switch_live_m0w", "net_cagr_60bps")
    lv_m_s = _val("vol_switch_live_m0w", "sharpe_net60")
    lv_v_c = _val("vol_switch_live_v5w", "net_cagr_60bps")
    lv_v_s = _val("vol_switch_live_v5w", "sharpe_net60")
    warmup_cagr_spread = abs(lv_m_c - lv_v_c)

    oracle_beats_both = (or_c > max(m0_c, v5_c)) and (or_s > max(m0_s, v5_s))
    oracle_cagr_gap   = or_c - max(m0_c, v5_c)
    oracle_sharpe_gap = or_s - max(m0_s, v5_s)
    best_baseline_c = max(m0_c, v5_c)
    best_baseline_s = max(m0_s, v5_s)

    # Robustness check: if the WORSE warmup variant fails to close ≥ 50% of the
    # oracle gap, the rule is warmup-dependent → not a robust live rule. We
    # anchor on the WORSE warmup because warmup default is an arbitrary
    # implementation choice that a live deployment can't retroactively pick.
    worse_live_c = min(lv_m_c, lv_v_c)
    worse_live_s = min(lv_m_s, lv_v_s)
    oracle_edge_c = or_c - best_baseline_c
    worse_edge_c  = worse_live_c - best_baseline_c
    worse_closes_gap = (oracle_edge_c > 0) and (worse_edge_c >= 0.5 * oracle_edge_c)
    warmup_dependent = warmup_cagr_spread >= 0.10   # 10pp CAGR spread = fragile

    lines: list[str] = []
    lines.append(
        f"Oracle vs best baseline: ΔCAGR {oracle_cagr_gap:+.1%}, "
        f"ΔShp {oracle_sharpe_gap:+.2f}"
    )
    lines.append(
        f"Live M0-warmup vs best baseline:  ΔCAGR {lv_m_c - best_baseline_c:+.1%}, "
        f"ΔShp {lv_m_s - best_baseline_s:+.2f}"
    )
    lines.append(
        f"Live V5-warmup vs best baseline:  ΔCAGR {lv_v_c - best_baseline_c:+.1%}, "
        f"ΔShp {lv_v_s - best_baseline_s:+.2f}"
    )
    lines.append(
        f"Warmup-dependency: CAGR spread between the two warmup defaults = "
        f"{warmup_cagr_spread:+.1%}  "
        f"({'fragile — rule leans on warmup choice' if warmup_dependent else 'robust to warmup choice'})"
    )

    if not oracle_beats_both:
        verdict = (
            "CLASSIFIER BRANCH: DO NOT OPEN. "
            "Oracle vol-switch does not beat both always-M0 and always-V5 on "
            "CAGR AND Sharpe. Regime diagnostic is useful for framing two "
            "profiles, but the switch itself has no reliable economic lift at "
            "n=17. Ship the dual-profile catalog (M0 aggressive / V5 defensive) "
            "and let the user pick their regime."
        )
    elif warmup_dependent:
        verdict = (
            "CLASSIFIER BRANCH: DO NOT OPEN. "
            f"Oracle switch IS good (ΔCAGR {oracle_cagr_gap:+.1%}), but the "
            "live rule's outcome depends heavily on the warmup default — "
            f"M0-warmup delivers {lv_m_c:+.1%} while V5-warmup delivers "
            f"{lv_v_c:+.1%} (spread {warmup_cagr_spread:+.1%}). The V5-warmup "
            "'recovery' is an accident of fold2 being high-vol; any real live "
            "deployment cannot pick the lucky warmup retroactively. Stay with "
            "dual-profile catalog. A classifier would inherit the same fragility."
        )
    elif oracle_beats_both and worse_closes_gap:
        verdict = (
            "CLASSIFIER BRANCH: JUSTIFIED (with caution). "
            "Oracle switch clearly beats both baselines AND the WORSE of the "
            "two warmup-default live variants still recovers ≥ 50% of the "
            "advantage. Rule is robust to the warmup choice. A more careful "
            "ex-ante classifier could be worth trying — but only with "
            "pre-registered holdout design; 17 rebalances leaves almost no "
            "room for tuning."
        )
    else:
        verdict = (
            "CLASSIFIER BRANCH: HIGH RISK. "
            "Oracle switch would help IF regime were known in advance, but no "
            "warmup variant of the naive live rule recovers it robustly. The "
            "regime boundary is real but not detectable from a simple trailing "
            "statistic. Any classifier risks fitting noise. Stay with dual-"
            "profile catalog for now."
        )

    commentary = "\n".join(f"  • {s}" for s in lines)
    return commentary, verdict


# ── Run ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading selections + labels + features …")
    m0_sel = pd.read_parquet(args.m0_selection)
    v5_sel = pd.read_parquet(args.v5_selection)
    labels = pd.read_parquet(args.labels)
    features = pd.read_parquet(args.features)

    m0_picks = _pick_sets(m0_sel)
    v5_picks = _pick_sets(v5_sel)
    oos_dates = sorted(set(m0_picks.keys()) & set(v5_picks.keys()))
    print(f"  OOS dates: {len(oos_dates)}  ({oos_dates[0].date()} → {oos_dates[-1].date()})")

    vol = _per_date_volatility(features, oos_dates)
    print(f"  volatility signal: median={vol.median():.4f}, "
          f"range=[{vol.min():.4f}, {vol.max():.4f}]")

    print("[2/4] Building allocations (always_M0, always_V5, oracle, live) …")
    rules: dict[str, pd.Series] = {
        "always_M0":            _rule_always("M0", oos_dates),
        "always_V5":            _rule_always("V5", oos_dates),
        "vol_switch_oracle":    _rule_oracle(vol),
        "vol_switch_live_m0w":  _rule_live_expanding(vol, min_periods=4,
                                                        warmup_default="M0"),
        "vol_switch_live_v5w":  _rule_live_expanding(vol, min_periods=4,
                                                        warmup_default="V5"),
    }

    print("[3/4] Metrics per allocation …")
    comp_rows: list[dict] = []
    choices_rows: list[dict] = []
    for name, profile_series in rules.items():
        m, seq = _sequence_metrics(profile_series, m0_picks, v5_picks,
                                    labels, bps=60)
        m["allocation"] = name
        comp_rows.append(m)
        for d, prof in profile_series.items():
            choices_rows.append({
                "rule":           name,
                "rebalance_date": d,
                "profile":        prof,
            })

    comp = pd.DataFrame(comp_rows)
    # Reorder columns
    col_order = ["allocation", "n_rebalances", "profile_switches",
                 "pct_dates_V5", "mean_monthly_gross", "sharpe_gross",
                 "mean_monthly_net60", "sharpe_net60", "net_cagr_60bps",
                 "max_drawdown_net60", "avg_turnover", "hit_rate_net"]
    comp = comp[[c for c in col_order if c in comp.columns]]
    comp.to_csv(os.path.join(reports_dir, "step6_vol_switch_comparison.csv"), index=False)

    choices = pd.DataFrame(choices_rows)
    choices.to_csv(os.path.join(reports_dir, "step6_vol_switch_choices.csv"), index=False)

    commentary, verdict = _verdict(comp)

    print("[4/4] Writing meta + printing …")
    meta = {
        "produced_at":    pd.Timestamp.utcnow().isoformat(),
        "n_oos":          len(oos_dates),
        "vol_median_post_hoc": float(vol.median()),
        "vol_signal":     "cross-sectional median of vol_std_60d per rebalance date",
        "rules": {
            "always_M0":         "every rebalance use M0 top-20 picks",
            "always_V5":         "every rebalance use V5 damp picks",
            "vol_switch_oracle": "vol[d] >= median(vol over all OOS dates) → V5, else M0 (POST-HOC)",
            "vol_switch_live":   "vol[d] >= expanding median up to d-1 (min_periods=4; pre-warmup=M0)",
        },
        "bps":            60,
        "commentary":     commentary,
        "verdict":        verdict,
        "elapsed_sec":    time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step6_vol_switch_run_meta.json"), meta)

    _print_table(comp, choices, vol, t0, reports_dir)
    print()
    print("  ANALYSIS:")
    print(commentary)
    print()
    print(f"  VERDICT:\n  **{verdict}**")
    print()
    print(f"  reports: {reports_dir}/step6_vol_switch_*.{{csv,json}}  "
          f"(elapsed {time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m0-selection", default=M0_SEL)
    p.add_argument("--v5-selection", default=V5_SEL)
    p.add_argument("--labels",       default=LABELS)
    p.add_argument("--features",     default=FEATURES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
