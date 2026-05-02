"""
Rules-based ranker backtest — V1.3.1 horizontal_base.

NO ML. Fixed rules from prior conversation:
  * Universe: signal_state ∈ {trigger, retest_bounce} ∧ breakout_age ≤ 5
              ∧ regime ∈ {long, neutral}
  * Tier A: mid_body trigger ∪ retest_bounce/deep_touch
  * Tier B: strict_body trigger ∪ retest_bounce/shallow_touch
  * Tier C: large_body trigger ∪ retest_bounce/no_touch
  * Sort within tier: common__day_return DESC
  * K = 3 picks per bar_date (greedy fill A→B→C)

Trade model:
  * Entry: close of bar_date
  * Exit: close 5 trading days later
  * Return: realized_R_5d × atr_pct × 100 (already in dataset as MFE-style realized)
  * TC: 30 bps round-trip
  * Equity: equal-weight, sequential compounding (trade-stream view)

NOT pre-registered. NOT OOS. Rules came from full-period descriptives;
this is in-sample sanity backtest. Numbers are upper-bound; live edge
likely lower.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

DATA = "output/horizontal_base_event_v1.parquet"
OUT_MD = "output/horizontal_base_rules_v1_backtest.md"

K = 3
TC_PCT = 0.30  # round-trip
RNG_SEED = 42

def tier_of(row):
    bc = row["family__body_class"]
    rk = row["family__retest_kind"]
    ss = row["signal_state"]
    if (ss == "trigger" and bc == "mid_body") or (ss == "retest_bounce" and rk == "deep_touch"):
        return "A"
    if (ss == "trigger" and bc == "strict_body") or (ss == "retest_bounce" and rk == "shallow_touch"):
        return "B"
    return "C"

def load_universe():
    df = pd.read_parquet(DATA)
    m = (df["signal_state"].isin(["trigger","retest_bounce"])
         & (df["family__breakout_age"] <= 5)
         & df["mfe_R_5d"].notna()
         & df["realized_R_5d"].notna()
         & df["common__regime"].isin(["long","neutral"]))
    u = df.loc[m].copy().reset_index(drop=True)
    u["ret_pct_5d"] = u["realized_R_5d"] * u["common__atr_pct"] * 100.0
    u["tier"] = u.apply(tier_of, axis=1)
    return u

def pick_rules(u, k=K):
    rows = []
    for bd, grp in u.groupby("bar_date"):
        out = []
        for t in ["A","B","C"]:
            sub = grp[grp["tier"] == t].sort_values("common__day_return", ascending=False)
            n_take = k - len(out)
            if n_take <= 0: break
            out.extend(sub.head(n_take).to_dict("records"))
        rows.extend(out)
    return pd.DataFrame(rows)

def pick_random(u, k=K, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for bd, grp in u.groupby("bar_date"):
        n = min(k, len(grp))
        idx = rng.choice(len(grp), size=n, replace=False)
        rows.extend(grp.iloc[idx].to_dict("records"))
    return pd.DataFrame(rows)

def pick_by_feature(u, feat, k=K, ascending=False):
    rows = []
    for bd, grp in u.groupby("bar_date"):
        sub = grp.sort_values(feat, ascending=ascending)
        rows.extend(sub.head(k).to_dict("records"))
    return pd.DataFrame(rows)

def metrics(picks, name):
    if picks.empty:
        return {"name": name, "n": 0}
    p = picks.sort_values("bar_date").reset_index(drop=True)
    p["ret_net"] = p["ret_pct_5d"] - TC_PCT
    r = p["ret_net"].values
    n = len(r)
    years = (p["bar_date"].max() - p["bar_date"].min()).days / 365.25
    tpy = n / years if years > 0 else np.nan
    wr = (r > 0).mean()
    mean_r = r.mean()
    std_r = r.std(ddof=1) if n > 1 else np.nan
    sharpe = (mean_r / std_r) * np.sqrt(tpy) if std_r and not np.isnan(std_r) else np.nan
    downside = r[r < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_r / dstd) * np.sqrt(tpy) if dstd and not np.isnan(dstd) else np.nan
    # PF
    gain = r[r > 0].sum()
    loss = -r[r < 0].sum()
    pf = gain / loss if loss > 0 else np.nan
    # equity curve (sequential compounding, single-bankroll)
    eq = (1 + r/100).cumprod()
    total_ret = (eq[-1] - 1) * 100
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1) * 100
    max_dd = dd.min()
    cagr = (eq[-1] ** (1/years) - 1) * 100 if years > 0 else np.nan
    return {
        "name": name, "n": n, "years": years, "trades_per_year": tpy,
        "WR": wr, "mean_pct": mean_r, "std_pct": std_r,
        "Sharpe": sharpe, "Sortino": sortino, "PF": pf,
        "TotalReturn_pct": total_ret, "CAGR_pct": cagr, "MaxDD_pct": max_dd,
    }

def fmt_row(m):
    if m.get("n", 0) == 0:
        return f"{m['name']:30s}  no trades"
    return (f"{m['name']:30s}  n={m['n']:5d}  WR={m['WR']:.4f}  "
            f"mean={m['mean_pct']:+.3f}%  Sharpe={m['Sharpe']:5.2f}  "
            f"Sortino={m['Sortino']:5.2f}  PF={m['PF']:4.2f}  "
            f"TotalRet={m['TotalReturn_pct']:+8.1f}%  CAGR={m['CAGR_pct']:+5.1f}%  "
            f"MaxDD={m['MaxDD_pct']:+.1f}%")

def by_tier_breakdown(picks):
    out = {}
    for t in ["A","B","C"]:
        sub = picks[picks["tier"] == t]
        if sub.empty: continue
        out[t] = metrics(sub, f"  rules[Tier {t}]")
    return out

def by_year_breakdown(picks):
    out = {}
    p = picks.copy()
    p["year"] = p["bar_date"].dt.year
    for y, sub in p.groupby("year"):
        out[int(y)] = metrics(sub, f"  rules[{y}]")
    return out

def main():
    u = load_universe()
    print(f"universe N = {len(u)} | bar_dates = {u['bar_date'].nunique()} | "
          f"period = {u['bar_date'].min().date()} → {u['bar_date'].max().date()}")
    print(f"K = {K}, TC = {TC_PCT} bps round-trip\n")

    # Strategies
    rules = pick_rules(u, K)
    rand = pick_random(u, K)
    by_atr = pick_by_feature(u, "common__atr_pct", K, ascending=False)
    by_dr = pick_by_feature(u, "common__day_return", K, ascending=False)
    all_ev = u.copy()  # treat each event as 1 trade

    strategies = [
        ("rules-based (tier+day_return)", rules),
        ("baseline: random K=3", rand),
        ("baseline: top-3 atr_pct", by_atr),
        ("baseline: top-3 day_return", by_dr),
        ("baseline: all events EW", all_ev),
    ]
    rows = [metrics(p, name) for name, p in strategies]

    print("=" * 78)
    print("HEADLINE METRICS")
    print("=" * 78)
    for r in rows:
        print(fmt_row(r))

    print()
    print("=" * 78)
    print("RULES — BREAKDOWNS")
    print("=" * 78)
    print(f"\nBy tier:")
    for t, m in by_tier_breakdown(rules).items():
        print(fmt_row(m))
    print(f"\nBy year:")
    for y, m in sorted(by_year_breakdown(rules).items()):
        print(fmt_row(m))

    # write report
    Path("output").mkdir(exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("# Rules-based ranker backtest — V1.3.1 horizontal_base\n\n")
        f.write("**Status:** in-sample, NOT pre-registered, NOT OOS. Rules derived from "
                "full-period descriptives (`event_quality_diag` + `v1_3_within_bin_diagnostic`). "
                "Numbers are an upper-bound; live edge likely lower.\n\n")
        f.write(f"**Universe:** N={len(u)}, period {u['bar_date'].min().date()} → {u['bar_date'].max().date()}\n\n")
        f.write(f"**Rules:** Tier A→B→C greedy fill, K={K}/day, sort by `day_return` within tier, "
                f"5-day hold, TC={TC_PCT}bps round-trip, equal-weight sequential compounding.\n\n")
        f.write("## Headline\n\n")
        f.write("| strategy | N | WR | mean/trade | Sharpe | Sortino | PF | Total ret | CAGR | Max DD |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get("n",0) == 0:
                f.write(f"| {r['name']} | 0 | — | — | — | — | — | — | — | — |\n")
                continue
            f.write(f"| {r['name']} | {r['n']} | {r['WR']:.4f} | {r['mean_pct']:+.3f}% | "
                    f"{r['Sharpe']:.2f} | {r['Sortino']:.2f} | {r['PF']:.2f} | "
                    f"{r['TotalReturn_pct']:+.1f}% | {r['CAGR_pct']:+.2f}% | {r['MaxDD_pct']:+.1f}% |\n")
        f.write("\n## Rules — by tier\n\n")
        f.write("| tier | N | WR | mean | Sharpe | Sortino | PF | Total ret | Max DD |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for t, m in by_tier_breakdown(rules).items():
            f.write(f"| {t} | {m['n']} | {m['WR']:.4f} | {m['mean_pct']:+.3f}% | "
                    f"{m['Sharpe']:.2f} | {m['Sortino']:.2f} | {m['PF']:.2f} | "
                    f"{m['TotalReturn_pct']:+.1f}% | {m['MaxDD_pct']:+.1f}% |\n")
        f.write("\n## Rules — by calendar year\n\n")
        f.write("| year | N | WR | mean | Sharpe | Sortino | PF | Total ret | Max DD |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for y, m in sorted(by_year_breakdown(rules).items()):
            f.write(f"| {y} | {m['n']} | {m['WR']:.4f} | {m['mean_pct']:+.3f}% | "
                    f"{m['Sharpe']:.2f} | {m['Sortino']:.2f} | {m['PF']:.2f} | "
                    f"{m['TotalReturn_pct']:+.1f}% | {m['MaxDD_pct']:+.1f}% |\n")
        f.write("\n## Caveats\n\n")
        f.write("- Sequential compounding assumes single-bankroll trade-stream — "
                "real portfolio with K=3/day × 5d overlap would have ~15 concurrent positions. "
                "True portfolio Sharpe ≠ trade-level Sharpe; daily-MTM portfolio backtest is a separate exercise.\n")
        f.write("- TC=30bps approximates BIST commission+spread for a 5d swing; not modeled: "
                "slippage, partial fills, halts, dividends.\n")
        f.write("- Tier boundaries from full-period `event_quality_diag` PFs; "
                "sort feature (day_return) from full-period within-bin diagnostic. "
                "Walk-forward verification (refit tier order each fold) was NOT done — "
                "warranted before live commitment.\n")
    print(f"\nReport: {OUT_MD}")

if __name__ == "__main__":
    main()
