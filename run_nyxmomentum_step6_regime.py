"""
Step 6 — regime-switch diagnostic.

Step 5.5 per-fold readout surfaced the real question: V4 ensemble has
stable *ordering* across regimes but inconsistent *net CAGR* advantage —
M0 wins fold3 and fold4, V4 wins fold2 (and V4's aggregate edge is
driven by M0's fold2 collapse).

Instead of asking "which model is better," this runner asks "WHEN is
each model better?" It partitions rebalance dates into regime buckets
from four ex-ante indicators and measures how M0 / V4 / V5 perform in
each.

REGIME INDICATORS (ex-ante, known at rebalance_date close):

  1. market_trend_state       — cross-sectional median of mom_63d.
                                High = broad market rising; Low = falling.
  2. breadth_state            — fraction of eligible tickers with
                                px_over_ma50 > 1.0.
                                High = broad participation; Low = narrow.
  3. volatility_state         — cross-sectional median of vol_std_60d.
                                High = turbulent; Low = calm.
  4. dispersion_state         — cross-sectional std of M0 prediction score.
                                High = scores well-separated (signal strong);
                                Low = scores clustered (signal weak).

  Each indicator binarized by its own median across the 17 rebalance
  dates → High / Low buckets (~8/9 samples each).

METRICS PER (regime_var, bucket, variant):
  n_rebalances, mean_monthly_gross, mean_monthly_net60,
  sharpe_gross, sharpe_net60, hit_rate (frac > 0).

OUTPUT (output/nyxmomentum/reports/):
  step6_regime_indicators.csv         per rebalance_date state values + buckets
  step6_regime_by_variant.csv         long format metrics per cell
  step6_regime_run_meta.json          verdict + narrative

NOTE: no parameter tuning this round. V4/V5 weights and dampener locked
to Step 5 spec. Aggregate CAGR is no longer the headline — we are
diagnosing regime dependence, not promoting a single winner.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.dampener import DampenerConfig, apply_dampener
from nyxmomentum.ensemble import rank_blend
from nyxmomentum.evaluation import top_n_portfolio_returns
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
PREDS_M0   = f"{REPORTS_DIR}/step4_predictions_M0.parquet"
PREDS_M1   = f"{REPORTS_DIR}/step4_predictions_M1.parquet"
LABELS     = f"{REPORTS_DIR}/step1_labels.parquet"
FEATURES   = f"{REPORTS_DIR}/step2_features.parquet"

V5_DAMP = DampenerConfig(n_enter=20, n_exit=30, smoothing_alpha=0.5)


# ── Portfolio builders (Step 5.5 locked) ────────────────────────────────────

def _v4_preds(preds_m0: pd.DataFrame, preds_m1: pd.DataFrame) -> pd.DataFrame:
    return rank_blend(preds_m0, preds_m1, weight_a=0.5, weight_b=0.5,
                       name_a="M0", name_b="M1")


def _strict_topn_portfolio(preds: pd.DataFrame, labels: pd.DataFrame,
                            top_n: int) -> pd.DataFrame:
    if "fold_id" not in preds.columns:
        preds = preds.assign(fold_id="unknown")
    return top_n_portfolio_returns(preds, labels, top_n=top_n, eligible_only=True)


# ── Regime indicators ───────────────────────────────────────────────────────

def _compute_regime_indicators(features: pd.DataFrame,
                                preds_m0: pd.DataFrame) -> pd.DataFrame:
    """
    Per rebalance_date, compute four ex-ante regime scalars over the
    eligible universe. All inputs are known at rebalance close.
    """
    f = features.loc[features["eligible"].astype(bool)].copy()
    # Market trend state: cross-sectional median of mom_63d
    trend = f.groupby("rebalance_date")["mom_63d"].median().rename("market_trend")

    # Breadth state: fraction above MA50. `px_over_ma50` in the features
    # panel is stored as (price / MA50) − 1 (centered near zero), so the
    # "above MA50" cutoff is > 0, not > 1.
    f["_above_ma50"] = (f["px_over_ma50"] > 0.0).astype(float)
    breadth = f.groupby("rebalance_date")["_above_ma50"].mean().rename("breadth")

    # Volatility state: cross-sectional median vol_std_60d
    vol = f.groupby("rebalance_date")["vol_std_60d"].median().rename("volatility")

    # Dispersion state: cross-sectional std of M0 prediction score
    pm = preds_m0.loc[preds_m0["eligible"].astype(bool)]
    disp = pm.groupby("rebalance_date")["prediction"].std(ddof=0).rename("dispersion")

    df = pd.concat([trend, breadth, vol, disp], axis=1).reset_index()
    # Binarize each by its own median across all OOS rebalance dates
    for col in ["market_trend", "breadth", "volatility", "dispersion"]:
        med = df[col].median()
        df[f"{col}_bucket"] = np.where(df[col] >= med, "High", "Low")
        df[f"{col}_median"] = med
    return df


# ── Cell metrics ────────────────────────────────────────────────────────────

def _cell_metrics(port: pd.DataFrame, bps: int) -> dict:
    """Simple per-rebalance mean returns + annualized Sharpe. Net = r_gross − turnover·bps."""
    if port.empty:
        return {"n_rebalances": 0}
    r_gross = port["portfolio_return"].astype(float)
    turn = port["turnover_fraction"].fillna(0.0).astype(float)
    r_net = r_gross - turn * (bps / 10000.0)
    n = len(port)
    mu_g = float(r_gross.mean()); sd_g = float(r_gross.std(ddof=0))
    mu_n = float(r_net.mean());   sd_n = float(r_net.std(ddof=0))
    eq_n = (1.0 + r_net).cumprod()
    peak = eq_n.cummax()
    dd = eq_n / peak - 1.0
    return {
        "n_rebalances":       int(n),
        "mean_monthly_gross": mu_g,
        "mean_monthly_net":   mu_n,
        "sharpe_gross":       float(mu_g / sd_g * np.sqrt(12)) if sd_g > 0 else np.nan,
        "sharpe_net":         float(mu_n / sd_n * np.sqrt(12)) if sd_n > 0 else np.nan,
        "hit_rate_net":       float((r_net > 0).mean()),
        "max_dd_net":         float(dd.min()) if n else np.nan,
    }


def _regime_by_variant_table(regime: pd.DataFrame,
                              portfolios: dict[str, pd.DataFrame],
                              bps: int = 60) -> pd.DataFrame:
    """Long-format: one row per (regime_var, bucket, variant)."""
    rows: list[dict] = []
    for rvar in ["market_trend", "breadth", "volatility", "dispersion"]:
        bkt = f"{rvar}_bucket"
        keep = regime[["rebalance_date", bkt, rvar]]
        for bucket_val in ["High", "Low"]:
            dates = keep.loc[keep[bkt] == bucket_val, "rebalance_date"]
            if dates.empty:
                continue
            for vname, vport in portfolios.items():
                p = vport.loc[vport["rebalance_date"].isin(dates)]
                m = _cell_metrics(p, bps=bps)
                m.update({
                    "regime_var": rvar,
                    "bucket":     bucket_val,
                    "variant":    vname,
                    "bps_round_trip": bps,
                    "indicator_range": f"{keep.loc[keep[bkt] == bucket_val, rvar].min():.3f} "
                                        f"→ {keep.loc[keep[bkt] == bucket_val, rvar].max():.3f}",
                })
                rows.append(m)
    return pd.DataFrame(rows)


# ── Printing ────────────────────────────────────────────────────────────────

def _fmt_pct(v, w: int = 8) -> str:
    return (f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  ").rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_regime_table(reg: pd.DataFrame, rbv: pd.DataFrame) -> None:
    print()
    print("══ nyxmomentum Step 6 — regime-switch diagnostic ══")
    print()
    print(f"  Rebalances: {len(reg)} OOS  (fold2+fold3+fold4)")
    print("  Regime indicators (median split):")
    for rvar in ["market_trend", "breadth", "volatility", "dispersion"]:
        med = reg[f"{rvar}_median"].iloc[0]
        nH = (reg[f"{rvar}_bucket"] == "High").sum()
        nL = (reg[f"{rvar}_bucket"] == "Low").sum()
        print(f"    {rvar:<14} median={med:+.4f}  |  High: n={nH}  Low: n={nL}")

    for rvar in ["market_trend", "breadth", "volatility", "dispersion"]:
        print()
        print(f"  ── {rvar.upper()} ─────────────────────────")
        print(f"  {'bucket':<8}{'variant':<24}"
              f"{'n':>4}{'µ_net':>10}{'shp_net':>9}{'hit':>7}{'DD_net':>9}{'shp_gross':>11}")
        sub = rbv.loc[rbv["regime_var"] == rvar].sort_values(["bucket", "variant"])
        for _, r in sub.iterrows():
            print(f"  {r['bucket']:<8}{r['variant']:<24}"
                  f"{int(r.get('n_rebalances', 0)):>4}"
                  f"{_fmt_pct(r.get('mean_monthly_net'), 10)}"
                  f"{_fmt_num(r.get('sharpe_net'), 9)}"
                  f"{_fmt_pct(r.get('hit_rate_net'), 7)}"
                  f"{_fmt_pct(r.get('max_dd_net'), 9)}"
                  f"{_fmt_num(r.get('sharpe_gross'), 11)}")


def _separation_table(rbv: pd.DataFrame) -> pd.DataFrame:
    """
    For each regime_var, compute per-variant (High−Low) deltas on
    mean_monthly_net and sharpe_net. Large |delta| = variant is
    regime-sensitive. Cross-variant delta-of-delta surfaces the
    "which variant prefers which regime" pattern.
    """
    rows: list[dict] = []
    for rvar, gvar in rbv.groupby("regime_var", sort=False):
        for variant, gv in gvar.groupby("variant", sort=False):
            hi = gv.loc[gv["bucket"] == "High"]
            lo = gv.loc[gv["bucket"] == "Low"]
            if hi.empty or lo.empty:
                continue
            h_mu = float(hi["mean_monthly_net"].iloc[0])
            l_mu = float(lo["mean_monthly_net"].iloc[0])
            h_sh = float(hi["sharpe_net"].iloc[0])
            l_sh = float(lo["sharpe_net"].iloc[0])
            rows.append({
                "regime_var":           rvar,
                "variant":              variant,
                "mu_net_high":          h_mu,
                "mu_net_low":           l_mu,
                "mu_net_delta_HmL":     h_mu - l_mu,
                "sharpe_net_high":      h_sh,
                "sharpe_net_low":       l_sh,
                "sharpe_net_delta_HmL": h_sh - l_sh,
            })
    return pd.DataFrame(rows)


def _winner_by_cell(rbv: pd.DataFrame) -> pd.DataFrame:
    """For each (regime_var, bucket), identify the variant with highest sharpe_net."""
    rows: list[dict] = []
    for (rvar, bkt), g in rbv.groupby(["regime_var", "bucket"], sort=False):
        sub = g.dropna(subset=["sharpe_net"])
        if sub.empty:
            continue
        idx = sub["sharpe_net"].idxmax()
        winner = sub.loc[idx]
        rows.append({
            "regime_var":   rvar,
            "bucket":       bkt,
            "winner":       winner["variant"],
            "winner_sharpe_net": float(winner["sharpe_net"]),
            "winner_mu_net":     float(winner["mean_monthly_net"]),
            "n_rebalances":      int(winner["n_rebalances"]),
        })
    return pd.DataFrame(rows)


def _commentary(rbv: pd.DataFrame, sep: pd.DataFrame, winners: pd.DataFrame) -> tuple[str, str]:
    """
    Produce Turkish narrative + production-shortlist verdict.

    Per-cell sharpe_net is read as the "which profile earns more risk-
    adjusted return in this regime" signal. V4 and V5 use the same
    ensemble scores — V5 just applies the dampener — so when the
    ensemble side wins a cell we interpret it as M0-aggressive losing
    to ensemble-based selection.

    Verdict thresholds (pre-registered):
      "regime-switched allocation candidate" — at least one regime_var where:
        • M0 is the winner in one bucket AND an ensemble variant
          (V4 or V5) is the winner in the other, AND
        • the sharpe_net gap (M0_best_bucket − M0_worst_bucket) is ≥ 1.0 AND
        • the ensemble variant's gap in the opposite direction is ≥ 1.0.
      "mixed — dual profile" — no clean split; recommend M0 aggressive +
          V5 defensive as parallel profiles, let capital allocator choose.
      "no separation" — variant-to-variant sharpe_net gap < 0.5 in every
          cell.
    """
    lines: list[str] = []
    clean_split_candidates: list[tuple[str, float, float]] = []

    def _bucket_val(rvar: str, bucket: str, variant: str, col: str) -> float:
        r = rbv.loc[(rbv["regime_var"] == rvar) &
                    (rbv["bucket"] == bucket) &
                    (rbv["variant"] == variant)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    # Per-regime narrative + switch detection
    for rvar in ["market_trend", "breadth", "volatility", "dispersion"]:
        sub = rbv.loc[rbv["regime_var"] == rvar]
        if sub.empty:
            continue
        w_hi = winners.loc[(winners["regime_var"] == rvar) & (winners["bucket"] == "High")]
        w_lo = winners.loc[(winners["regime_var"] == rvar) & (winners["bucket"] == "Low")]
        w_hi_name = str(w_hi["winner"].iloc[0]) if len(w_hi) else "—"
        w_lo_name = str(w_lo["winner"].iloc[0]) if len(w_lo) else "—"
        w_hi_sh = float(w_hi["winner_sharpe_net"].iloc[0]) if len(w_hi) else np.nan
        w_lo_sh = float(w_lo["winner_sharpe_net"].iloc[0]) if len(w_lo) else np.nan

        # "Clean" = M0 wins one bucket, ensemble wins the other, AND both
        # variants show ≥ 1.0 Sharpe swing between buckets on the opposite
        # direction (i.e. the regime genuinely flips which variant is better,
        # not just noise at the boundary).
        winners_set = {w_hi_name, w_lo_name}
        ensemble_set = {"V4_ensemble_strict", "V5_ensemble_damp"}
        if ("M0_reference" in winners_set and (winners_set & ensemble_set)):
            m0_gap = abs(_bucket_val(rvar, "High", "M0_reference", "sharpe_net")
                          - _bucket_val(rvar, "Low", "M0_reference", "sharpe_net"))
            ens_variant = next(iter(winners_set & ensemble_set))
            ens_gap = abs(_bucket_val(rvar, "High", ens_variant, "sharpe_net")
                           - _bucket_val(rvar, "Low", ens_variant, "sharpe_net"))
            # Sign check: M0 advantage must be in the bucket opposite to
            # ensemble's advantage.
            m0_hi_minus_lo = (_bucket_val(rvar, "High", "M0_reference", "sharpe_net")
                               - _bucket_val(rvar, "Low", "M0_reference", "sharpe_net"))
            ens_hi_minus_lo = (_bucket_val(rvar, "High", ens_variant, "sharpe_net")
                                - _bucket_val(rvar, "Low", ens_variant, "sharpe_net"))
            opposite_sign = (m0_hi_minus_lo * ens_hi_minus_lo) < 0
            if m0_gap >= 1.0 and ens_gap >= 1.0 and opposite_sign:
                clean_split_candidates.append((rvar, m0_gap, ens_gap))

        lines.append(
            f"{rvar}: High→{w_hi_name} (Shp {w_hi_sh:+.2f}) | "
            f"Low→{w_lo_name} (Shp {w_lo_sh:+.2f})"
        )

    # Top regime sensitivity per variant
    for variant in ["M0_reference", "V4_ensemble_strict", "V5_ensemble_damp"]:
        ss = sep.loc[sep["variant"] == variant].copy()
        if ss.empty:
            continue
        ss["_abs_mu_delta"] = ss["mu_net_delta_HmL"].abs()
        top = ss.sort_values("_abs_mu_delta", ascending=False).iloc[0]
        lines.append(
            f"{variant} en rejim-duyarlı eksen: {top['regime_var']} "
            f"(µ_net Δ = {top['mu_net_delta_HmL']:+.1%})"
        )

    # Verdict + actionable recommendation
    m0_wins = (winners["winner"] == "M0_reference").sum()
    v5_wins = (winners["winner"] == "V5_ensemble_damp").sum()
    v4_wins = (winners["winner"] == "V4_ensemble_strict").sum()

    if clean_split_candidates:
        # Rank by total gap size → prefer strongest switcher
        clean_split_candidates.sort(key=lambda t: t[1] + t[2], reverse=True)
        best_rvar, m0_gap, ens_gap = clean_split_candidates[0]
        # Which bucket favors which profile? Check on volatility specifically.
        m0_high_better = (_bucket_val(best_rvar, "High", "M0_reference", "sharpe_net")
                          > _bucket_val(best_rvar, "Low", "M0_reference", "sharpe_net"))
        m0_bucket = "High" if m0_high_better else "Low"
        ens_bucket = "Low" if m0_high_better else "High"
        verdict = (
            f"regime-switched candidate on {best_rvar}: "
            f"M0 in {best_rvar}={m0_bucket}, V5 in {best_rvar}={ens_bucket} "
            f"(M0 ΔShp {m0_gap:+.1f}, ensemble ΔShp {ens_gap:+.1f}). "
            f"Live switching requires robust ex-ante classifier for {best_rvar} — "
            f"dual profile M0+V5 remains the safe default."
        )
        lines.append(
            f"Rejim-switcher adayı: **{best_rvar}** "
            f"(M0 {best_rvar}={m0_bucket} tarafında hakim, V5 {best_rvar}={ens_bucket} tarafında hakim)"
        )
    elif m0_wins >= 1 and (v5_wins + v4_wins) >= 1:
        verdict = (
            f"mixed — no clean regime switch detected, dual profile recommended: "
            f"M0 aggressive + V5 defensive as parallel tracks "
            f"(cells won — M0: {m0_wins}, V4: {v4_wins}, V5: {v5_wins}). "
            f"V4 arada bir profil değil, V5'in turnover'lı versiyonu gibi — "
            f"promosyon shortlist'te V4 için ayrı bir gerekçe aramaya gerek yok."
        )
    else:
        verdict = (
            "no clear separation — per-cell sharpe gaps within noise. "
            "Keep M0 as single benchmark until another OOS fold arrives."
        )

    commentary = "\n".join(f"  • {s}" for s in lines)
    return commentary, verdict


# ── Run ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/5] Loading …")
    preds_m0 = pd.read_parquet(args.preds_m0)
    preds_m1 = pd.read_parquet(args.preds_m1)
    labels   = pd.read_parquet(args.labels)
    features = pd.read_parquet(args.features)
    print(f"  preds_M0={len(preds_m0):,}  preds_M1={len(preds_m1):,}  "
          f"labels={len(labels):,}  features={len(features):,}")

    print("[2/5] Building M0 / V4 / V5 portfolios …")
    v4_preds = _v4_preds(preds_m0, preds_m1)
    v4_port  = _strict_topn_portfolio(v4_preds, labels, top_n=args.top_n)
    _, v5_port = apply_dampener(v4_preds, labels, V5_DAMP)
    fold_map = v4_preds.groupby("rebalance_date")["fold_id"].first()
    v5_port = v5_port.assign(fold_id=v5_port["rebalance_date"].map(fold_map))
    m0_port  = _strict_topn_portfolio(preds_m0, labels, top_n=args.top_n)

    # Align all portfolios to the same OOS rebalance dates
    oos_dates = sorted(v4_port["rebalance_date"].unique())
    portfolios = {
        "M0_reference":       m0_port.loc[m0_port["rebalance_date"].isin(oos_dates)],
        "V4_ensemble_strict": v4_port,
        "V5_ensemble_damp":   v5_port,
    }
    print(f"  OOS rebalances: {len(oos_dates)}  "
          f"({min(oos_dates).date()} → {max(oos_dates).date()})")

    print("[3/5] Regime indicators …")
    # Restrict features to OOS rebalance dates
    feat_oos = features.loc[features["rebalance_date"].isin(oos_dates)]
    pm0_oos  = preds_m0.loc[preds_m0["rebalance_date"].isin(oos_dates)]
    regime = _compute_regime_indicators(feat_oos, pm0_oos)
    regime.to_csv(os.path.join(reports_dir, "step6_regime_indicators.csv"), index=False)

    print("[4/5] Per-cell metrics …")
    rbv = _regime_by_variant_table(regime, portfolios, bps=60)
    rbv.to_csv(os.path.join(reports_dir, "step6_regime_by_variant.csv"), index=False)

    sep = _separation_table(rbv)
    winners = _winner_by_cell(rbv)
    sep.to_csv(os.path.join(reports_dir, "step6_regime_separation.csv"), index=False)
    winners.to_csv(os.path.join(reports_dir, "step6_regime_winners.csv"), index=False)

    commentary, verdict = _commentary(rbv, sep, winners)

    print("[5/5] Writing meta + printing …")
    meta = {
        "produced_at":   pd.Timestamp.utcnow().isoformat(),
        "top_n":         args.top_n,
        "bps_net":       60,
        "v5_config":     {"n_enter": V5_DAMP.n_enter, "n_exit": V5_DAMP.n_exit,
                          "smoothing_alpha": V5_DAMP.smoothing_alpha},
        "weights_locked": {"w_M0": 0.5, "w_M1": 0.5},
        "n_oos_rebalances": len(oos_dates),
        "regime_indicator_medians": {
            col: float(regime[f"{col}_median"].iloc[0])
            for col in ["market_trend", "breadth", "volatility", "dispersion"]
        },
        "verdict":       verdict,
        "commentary":    commentary,
        "elapsed_sec":   time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step6_regime_run_meta.json"), meta)

    _print_regime_table(regime, rbv)
    print()
    print("  WINNER BY CELL (max sharpe_net@60bps):")
    print(winners.to_string(index=False))
    print()
    print("  SEPARATION TABLE (variant × regime_var High−Low deltas):")
    print(sep[[
        "regime_var", "variant", "mu_net_high", "mu_net_low",
        "mu_net_delta_HmL", "sharpe_net_high", "sharpe_net_low",
        "sharpe_net_delta_HmL",
    ]].round(3).to_string(index=False))
    print()
    print("  COMMENTARY:")
    print(commentary)
    print()
    print(f"  VERDICT: **{verdict}**")
    print()
    print(f"  reports: {reports_dir}/step6_regime_*.{{csv,json}}"
          f"  (elapsed {time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-m0", default=PREDS_M0)
    p.add_argument("--preds-m1", default=PREDS_M1)
    p.add_argument("--labels",   default=LABELS)
    p.add_argument("--features", default=FEATURES)
    p.add_argument("--top-n",    type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
