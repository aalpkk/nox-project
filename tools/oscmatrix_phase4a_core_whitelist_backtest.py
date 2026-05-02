"""Phase 4A — Core-class whitelist backtest of close-source AL_F6.

OSCMTRX-inspired close-source HWO early reversal engine.
NOT a LuxAlgo OSCMTRX bit-exact clone.

Decision (Phase 0.6.3 close-vs-ohlc4 robustness probe):
  HyperWave source = close. ohlc4 stays as TV-fidelity diagnostic only.

Universe (Core-class objective whitelist):
  median_turnover_60 ≥ cross-sectional q60      (per-date, look-ahead-free)
  history_bars ≥ 500                             (prior daily bars only)

Signal:
  AL_F6 = HWO_up ∧ armed_recent10 ∧ refractory10
  with all components from compute_all(close-source HW).

Entry:
  primary   : signal at close(T) → enter at open(T+1)
  secondary : signal at close(T) → enter at close(T)   (optimistic)

SAT short-entry: disabled (Phase 1.1+1.2 finding).

Window: 2023-01-02 → 2026-04-29 (full panel).

Per-trade pool (no top-K cap). Outputs:
  per-trade CSV (one row per fire × horizon × entry-mode)
  aggregate CSV (cohort × year × horizon × entry-mode)
  same-date random baseline CSV (fire-date-matched random sample on
                                   the same eligible universe)
  ticker / date concentration CSV
  drawdown CSV (chronological cumulative equity)
  report markdown

This is the production candidate; Phase 4B (full-607 diagnostic) is a
separate run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402

MASTER = Path("output/extfeed_intraday_1h_3y_master.parquet")
MIN_DATE = "2023-01-02"
MAX_DATE = "2026-04-29"

OUT_DIR = Path("output")
OUT_PER_TRADE = OUT_DIR / "oscmatrix_phase4a_per_trade.csv"
OUT_AGG = OUT_DIR / "oscmatrix_phase4a_aggregate.csv"
OUT_BASELINE = OUT_DIR / "oscmatrix_phase4a_random_baseline.csv"
OUT_CONCENTRATION = OUT_DIR / "oscmatrix_phase4a_concentration.csv"
OUT_DD = OUT_DIR / "oscmatrix_phase4a_drawdown.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase4a_report.md"

# AL_F6 hyperparams (frozen from Phase 1.x)
REFRACTORY_K = 10
ARM_RECENT_K = 10
OVF_RECENT_K = 5
FALSE_REV_ATR_THR = 0.5
HORIZONS = [5, 10, 20]

# Universe gate
HISTORY_FLOOR = 500
LIQ_QUANTILE = 0.60

# Random baseline
N_BASELINE_DRAWS = 1
BASELINE_SEED = 17


# =========================================================================
# Daily panel from 1h master
# =========================================================================

def build_daily_panel() -> pd.DataFrame:
    print(f"  loading {MASTER} …")
    bars = pd.read_parquet(MASTER)
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])
    if bars["ts_istanbul"].dt.tz is None:
        bars["ts_istanbul"] = bars["ts_istanbul"].dt.tz_localize("Europe/Istanbul")
    min_ts = pd.Timestamp(MIN_DATE).tz_localize("Europe/Istanbul")
    max_ts = (pd.Timestamp(MAX_DATE) + pd.Timedelta(days=1)).tz_localize("Europe/Istanbul")
    bars = bars[(bars["ts_istanbul"] >= min_ts) & (bars["ts_istanbul"] < max_ts)]
    bars["date"] = bars["ts_istanbul"].dt.date
    daily = (
        bars.groupby(["ticker", "date"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"  daily panel: {len(daily):,} rows, {daily['ticker'].nunique()} tickers, "
          f"{daily['date'].nunique()} dates")
    return daily


def add_universe_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Append look-ahead-free universe features (per ticker).

    median_turnover_60 : median(close × volume) over prior 60 daily bars
                         (NOT including current day).
    history_bars       : count of prior daily bars (excludes today).
    """
    out = []
    for tkr, g in daily.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        turnover = g["close"] * g["volume"]
        g["median_turnover_60"] = (
            turnover.shift(1).rolling(60, min_periods=20).median()
        )
        g["history_bars"] = np.arange(len(g))  # bars *before* today
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_q60_thresholds(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional q60 of `median_turnover_60` per date."""
    def q60(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.quantile(LIQ_QUANTILE)) if len(s) else np.nan
    out = (
        panel.groupby("date")
        .agg(liq_q60=("median_turnover_60", q60))
        .reset_index()
    )
    return out


def attach_eligibility(panel: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    p = panel.merge(thresholds, on="date", how="left")
    p["eligible_core"] = (
        (p["median_turnover_60"] >= p["liq_q60"])
        & (p["history_bars"] >= HISTORY_FLOOR)
    ).fillna(False)
    return p


# =========================================================================
# Signal — AL_F6 with close-source HW
# =========================================================================

def recent_any(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).max().astype(bool)


def apply_refractory(fires: pd.Series, K: int) -> pd.Series:
    arr = fires.values
    out = np.zeros(len(arr), dtype=bool)
    last = -10**9
    for i in range(len(arr)):
        if arr[i] and (i - last) > K:
            out[i] = True
            last = i
    return pd.Series(out, index=fires.index)


def build_al_f6(g: pd.DataFrame, ours: pd.DataFrame) -> pd.Series:
    hw = ours["hyperwave"].reindex(g.index)
    hwo_up = ours["hwo_up"].reindex(g.index).fillna(False).astype(bool)
    lower_zone = ours["lower_zone"].astype("Int64").fillna(0).astype(int).reindex(g.index).fillna(0).astype(int)
    bear_ovf = (ours["bearish_overflow"].fillna(50) != 50).reindex(g.index).fillna(False).astype(bool)
    al_armed = (hw < 30) | (lower_zone == 2) | recent_any(bear_ovf, OVF_RECENT_K)
    al_armed_recent = recent_any(al_armed, ARM_RECENT_K)
    al_f6_raw = hwo_up & al_armed_recent
    return apply_refractory(al_f6_raw, REFRACTORY_K)


# =========================================================================
# Forward returns + MFE/MAE
# =========================================================================

def compute_forward_metrics(g: pd.DataFrame, fire_idx: np.ndarray) -> pd.DataFrame:
    """Per fire × horizon × entry-mode metrics."""
    if len(fire_idx) == 0:
        return pd.DataFrame()
    open_p = g["open"].values
    high_p = g["high"].values
    low_p = g["low"].values
    close_p = g["close"].values
    n = len(g)

    # ATR(14) from daily bars
    prev_c = pd.Series(close_p).shift(1)
    tr = pd.concat([
        pd.Series(high_p - low_p),
        (pd.Series(high_p) - prev_c).abs(),
        (pd.Series(low_p) - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean().values

    rows = []
    for i in fire_idx:
        a = atr[i] if i < len(atr) else np.nan
        if not np.isfinite(a) or a == 0:
            continue
        for entry_mode, entry_price, fwd_lo, fwd_hi in [
            ("close_T",  close_p[i],  i + 1, None),
            ("open_T1",  open_p[i + 1] if i + 1 < n else np.nan, i + 1, None),
        ]:
            if not np.isfinite(entry_price):
                continue
            for K in HORIZONS:
                hi = min(n, i + K + 1)
                if hi <= i + 1:
                    continue
                # forward window strictly T+1..T+K
                fmh = float(np.max(high_p[i + 1:hi]))
                fml = float(np.min(low_p[i + 1:hi]))
                # realized close — for primary use close at T+K; for secondary same.
                exit_idx = i + K
                if exit_idx >= n:
                    realized = np.nan
                    exit_price = np.nan
                else:
                    exit_price = close_p[exit_idx]
                    realized = exit_price - entry_price
                mfe = fmh - entry_price
                mae = entry_price - fml
                mfe_atr = mfe / a
                mae_atr = mae / a
                ratio = mfe_atr / max(mae_atr, 1e-3)
                false_rev = (realized < -FALSE_REV_ATR_THR * a) if np.isfinite(realized) else None
                pct = (realized / entry_price) if (np.isfinite(realized) and entry_price > 0) else np.nan
                capture = (realized / mfe) if (np.isfinite(realized) and mfe > 0) else np.nan
                rows.append({
                    "fire_idx": int(i),
                    "K": K,
                    "entry_mode": entry_mode,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price) if np.isfinite(exit_price) else np.nan,
                    "realized": float(realized) if np.isfinite(realized) else np.nan,
                    "realized_pct": float(pct) if np.isfinite(pct) else np.nan,
                    "mfe_atr": float(mfe_atr),
                    "mae_atr": float(mae_atr),
                    "ratio": float(ratio),
                    "false_rev": false_rev,
                    "mfe_capture": float(capture) if np.isfinite(capture) else np.nan,
                    "atr": float(a),
                })
    return pd.DataFrame(rows)


# =========================================================================
# Per-ticker pipeline
# =========================================================================

def run_ticker(g: pd.DataFrame, eligibility: pd.Series) -> pd.DataFrame:
    """Returns per-fire forward metrics with date attached, filtered to
    eligible_core==True at the fire bar.
    """
    df = g[["open", "high", "low", "close", "volume"]].astype(float).copy()
    df.index = pd.RangeIndex(len(df))
    if len(df) < HISTORY_FLOOR:
        return pd.DataFrame()
    ours = compute_all(df, DEFAULT_PARAMS, zone_rule="above_50")
    al_f6 = build_al_f6(df, ours)
    fire_idx = np.where(al_f6.values)[0]
    if len(fire_idx) == 0:
        return pd.DataFrame()
    elig = eligibility.reset_index(drop=True).values.astype(bool)
    fire_idx = np.array([i for i in fire_idx if i < len(elig) and elig[i]])
    if len(fire_idx) == 0:
        return pd.DataFrame()
    fwd = compute_forward_metrics(df, fire_idx)
    if fwd.empty:
        return fwd
    dates = g["date"].reset_index(drop=True)
    fwd["date"] = fwd["fire_idx"].map(lambda i: dates.iloc[i])
    fwd["close_at_fire"] = fwd["fire_idx"].map(lambda i: float(df["close"].iloc[i]))
    return fwd


# =========================================================================
# Aggregation helpers
# =========================================================================

def aggregate_metrics(fires: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if fires.empty:
        return pd.DataFrame()
    for keys, sub in fires.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        wins = sub["realized"].dropna() > 0
        pos_sum = float(sub.loc[sub["realized"] > 0, "realized"].sum())
        neg_sum = float(-sub.loc[sub["realized"] < 0, "realized"].sum())
        pf = (pos_sum / neg_sum) if neg_sum > 0 else np.nan
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": int(len(sub)),
            "uniq_dates": int(sub["date"].nunique()),
            "uniq_tickers": int(sub["ticker"].nunique()) if "ticker" in sub.columns else None,
            "ratio_med": float(sub["ratio"].median()),
            "mfe_atr_med": float(sub["mfe_atr"].median()),
            "mae_atr_med": float(sub["mae_atr"].median()),
            "realized_med": float(sub["realized_pct"].dropna().median()) if sub["realized_pct"].notna().any() else np.nan,
            "realized_mean": float(sub["realized_pct"].dropna().mean()) if sub["realized_pct"].notna().any() else np.nan,
            "win_rate": float(wins.mean()) if len(wins) else np.nan,
            "false_rev_pct": float(sub["false_rev"].dropna().astype(float).mean() * 100)
                if sub["false_rev"].notna().any() else np.nan,
            "mfe_capture_med": float(sub["mfe_capture"].dropna().median())
                if sub["mfe_capture"].notna().any() else np.nan,
            "pf_proxy": pf,
            "pos_sum_pct": pos_sum,
            "neg_sum_pct": neg_sum,
        })
    return pd.DataFrame(rows)


def chrono_drawdown(fires: pd.DataFrame, K: int, entry_mode: str) -> pd.DataFrame:
    """Cumulative-equity drawdown proxy: chronological sum of realized_pct
    (per fire), assuming 1 unit per trade.  Each trade closes K days after
    the fire so we group by exit-date for fair sequencing.
    """
    sub = fires[(fires["K"] == K) & (fires["entry_mode"] == entry_mode)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.dropna(subset=["realized_pct"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["exit_date"] = sub["date"] + pd.to_timedelta(K, unit="D")
    sub = sub.sort_values("exit_date").reset_index(drop=True)
    sub["equity"] = sub["realized_pct"].cumsum()
    sub["peak"] = sub["equity"].cummax()
    sub["dd"] = sub["equity"] - sub["peak"]
    return sub[["date", "exit_date", "ticker", "K", "entry_mode",
                "realized_pct", "equity", "peak", "dd"]]


# =========================================================================
# Same-date random baseline
# =========================================================================

def random_baseline(eligible_panel: pd.DataFrame, fires: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    """For each fire date with K_SET, sample one eligible (ticker, date)
    pair *not* fired on AL_F6 that day, compute the same horizon metrics.
    Result: a null distribution matched to fire-date eligibility at the
    same cardinality.
    """
    if fires.empty:
        return pd.DataFrame()
    actual_pairs = set((r.ticker, r.date) for r in fires.itertuples())
    fire_dates = fires.groupby("date").size()  # fires per date
    elig_by_date = (
        eligible_panel[eligible_panel["eligible_core"]]
        .groupby("date")["ticker"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for d, n_fires in fire_dates.items():
        candidates = [t for t in elig_by_date.get(d, [])
                      if (t, d) not in actual_pairs]
        if not candidates:
            continue
        n_pick = min(n_fires, len(candidates))
        picks = rng.choice(candidates, size=n_pick, replace=False)
        for t in picks:
            rows.append({"date": d, "ticker": t})
    if not rows:
        return pd.DataFrame()
    pairs = pd.DataFrame(rows)

    # For each picked (ticker, date) pair, compute forward metrics from
    # that ticker's daily series. We'll re-build per-ticker daily indexed
    # forward windows rather than re-running compute_all.
    full = eligible_panel
    full_idx = full.set_index(["ticker", "date"]).sort_index()
    out_rows = []
    for tkr, sub in pairs.groupby("ticker"):
        try:
            tg = full_idx.xs(tkr, level="ticker").reset_index()
        except KeyError:
            continue
        tg = tg.reset_index(drop=True)
        date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(tg["date"].values)}
        for d in sub["date"].values:
            i = date_to_pos.get(pd.Timestamp(d))
            if i is None:
                continue
            df_local = tg[["open", "high", "low", "close"]].astype(float)
            high = df_local["high"].values
            low = df_local["low"].values
            close = df_local["close"].values
            open_ = df_local["open"].values
            n = len(df_local)
            prev_c = pd.Series(close).shift(1)
            tr = pd.concat([
                pd.Series(high - low),
                (pd.Series(high) - prev_c).abs(),
                (pd.Series(low) - prev_c).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean().values
            a = atr[i] if i < len(atr) else np.nan
            if not np.isfinite(a) or a == 0:
                continue
            for entry_mode, entry_price in [
                ("close_T", close[i]),
                ("open_T1", open_[i + 1] if i + 1 < n else np.nan),
            ]:
                if not np.isfinite(entry_price):
                    continue
                for K in HORIZONS:
                    hi = min(n, i + K + 1)
                    if hi <= i + 1:
                        continue
                    fmh = float(np.max(high[i + 1:hi]))
                    fml = float(np.min(low[i + 1:hi]))
                    exit_idx = i + K
                    realized = (close[exit_idx] - entry_price) if exit_idx < n else np.nan
                    pct = (realized / entry_price) if (np.isfinite(realized) and entry_price > 0) else np.nan
                    out_rows.append({
                        "ticker": tkr,
                        "date": d,
                        "K": K,
                        "entry_mode": entry_mode,
                        "realized_pct": float(pct) if np.isfinite(pct) else np.nan,
                        "ratio": (fmh - entry_price) / max(entry_price - fml, 1e-9 * entry_price),
                        "mfe_atr": (fmh - entry_price) / a,
                        "mae_atr": (entry_price - fml) / a,
                        "false_rev": (np.isfinite(realized) and realized < -FALSE_REV_ATR_THR * a),
                    })
    return pd.DataFrame(out_rows)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("=== Phase 4A — Core-class whitelist backtest ===")
    print("    OSCMTRX-inspired close-source HWO early reversal engine\n")

    daily = build_daily_panel()
    daily = add_universe_features(daily)
    thresholds = build_q60_thresholds(daily)
    panel = attach_eligibility(daily, thresholds)
    print(f"  cross-sectional q60 + history floor → "
          f"eligible per-date median = {int(panel.groupby('date')['eligible_core'].sum().median())}")

    print("\n  computing AL_F6 per ticker …")
    all_fires = []
    skip_n = 0
    n_tickers = panel["ticker"].nunique()
    for k, (tkr, g) in enumerate(panel.groupby("ticker", sort=False), start=1):
        if k % 100 == 0:
            print(f"    [{k}/{n_tickers}] processed; fires so far={len(all_fires)}")
        g = g.reset_index(drop=True)
        elig = g["eligible_core"]
        out = run_ticker(g, elig)
        if out.empty:
            skip_n += 1
            continue
        out["ticker"] = tkr
        out["year"] = pd.to_datetime(out["date"]).dt.year
        all_fires.append(out)
    if not all_fires:
        print("  no fires — abort.")
        return
    fires = pd.concat(all_fires, ignore_index=True)
    fires["date"] = pd.to_datetime(fires["date"])
    print(f"  total fires: {fires.groupby(['ticker', 'date']).ngroups} unique (ticker, date) "
          f"× {len(HORIZONS)*2} (K × entry_mode) = {len(fires)} per-trade rows. "
          f"(skipped {skip_n} tickers)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fires.to_csv(OUT_PER_TRADE, index=False)
    print(f"  wrote {OUT_PER_TRADE}")

    # Aggregate: by year × K × entry_mode
    agg_year = aggregate_metrics(fires, ["year", "entry_mode", "K"])
    agg_all = aggregate_metrics(fires.assign(year_all="ALL"),
                                ["year_all", "entry_mode", "K"]).rename(columns={"year_all": "year"})
    agg = pd.concat([agg_all, agg_year], ignore_index=True)
    agg.to_csv(OUT_AGG, index=False)
    print(f"  wrote {OUT_AGG}")

    # Concentration
    by_ticker = (fires.groupby(["ticker"]).size().reset_index(name="fires_total")
                 .sort_values("fires_total", ascending=False))
    by_date = (fires.groupby(["date"]).size().reset_index(name="fires_total")
               .sort_values("fires_total", ascending=False))
    by_ticker.head(50).to_csv(OUT_CONCENTRATION.with_suffix(".by_ticker_top50.csv"), index=False)
    by_date.head(50).to_csv(OUT_CONCENTRATION.with_suffix(".by_date_top50.csv"), index=False)
    by_ticker.to_csv(OUT_CONCENTRATION, index=False)
    print(f"  wrote {OUT_CONCENTRATION} (per-ticker counts)")

    # Drawdown chronological per (K, entry_mode)
    dd_rows = []
    dd_summary_rows = []
    for K in HORIZONS:
        for em in ["close_T", "open_T1"]:
            ddf = chrono_drawdown(fires, K, em)
            if not ddf.empty:
                ddf["K"] = K
                ddf["entry_mode"] = em
                dd_rows.append(ddf)
                dd_summary_rows.append({
                    "K": K, "entry_mode": em,
                    "n_trades": int(len(ddf)),
                    "final_equity_pct": float(ddf["equity"].iloc[-1]),
                    "max_dd_pct": float(ddf["dd"].min()),
                })
    if dd_rows:
        dd_all = pd.concat(dd_rows, ignore_index=True)
        dd_all.to_csv(OUT_DD, index=False)
        print(f"  wrote {OUT_DD}")

    # Same-date random baseline
    print("\n  building same-date random baseline …")
    rng = np.random.default_rng(BASELINE_SEED)
    baseline = random_baseline(panel, fires.drop_duplicates(["ticker", "date"]), rng)
    if not baseline.empty:
        baseline["year"] = pd.to_datetime(baseline["date"]).dt.year
        baseline.to_csv(OUT_BASELINE, index=False)
        baseline_agg = (
            baseline.groupby(["entry_mode", "K"], dropna=False)
            .agg(n=("realized_pct", "count"),
                 ratio_med=("ratio", "median"),
                 mfe_atr_med=("mfe_atr", "median"),
                 mae_atr_med=("mae_atr", "median"),
                 realized_med=("realized_pct", "median"),
                 win_rate=("realized_pct", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
                 false_rev_pct=("false_rev", lambda s: float(s.dropna().astype(float).mean() * 100) if s.notna().any() else np.nan))
            .reset_index()
        )
        print(f"  wrote {OUT_BASELINE} ({len(baseline)} baseline trades)")
    else:
        baseline_agg = pd.DataFrame()

    # ---- Console summary ----
    print("\n=== Aggregate (overall, primary entry=open_T1) ===")
    summ = agg[(agg["year"] == "ALL") & (agg["entry_mode"] == "open_T1")][
        ["K", "n", "uniq_dates", "ratio_med", "realized_med",
         "win_rate", "false_rev_pct", "mfe_capture_med", "pf_proxy"]
    ]
    print(summ.to_string(index=False))

    print("\n=== Aggregate (overall, secondary entry=close_T) ===")
    summ2 = agg[(agg["year"] == "ALL") & (agg["entry_mode"] == "close_T")][
        ["K", "n", "uniq_dates", "ratio_med", "realized_med",
         "win_rate", "false_rev_pct", "mfe_capture_med", "pf_proxy"]
    ]
    print(summ2.to_string(index=False))

    print("\n=== Per-year (open_T1, K=10) ===")
    py = agg[(agg["year"] != "ALL") & (agg["entry_mode"] == "open_T1") & (agg["K"] == 10)]
    print(py[["year", "n", "uniq_dates", "ratio_med", "realized_med",
              "win_rate", "false_rev_pct", "pf_proxy"]].to_string(index=False))

    if not baseline_agg.empty:
        print("\n=== Same-date random baseline ===")
        print(baseline_agg.to_string(index=False))

    if dd_summary_rows:
        print("\n=== Drawdown summary ===")
        print(pd.DataFrame(dd_summary_rows).to_string(index=False))

    # ---- Markdown report ----
    md: list[str] = []
    md.append("# Phase 4A — Core-class whitelist backtest")
    md.append("")
    md.append("OSCMTRX-inspired close-source HWO early reversal engine.")
    md.append("**Not** a LuxAlgo OSCMTRX bit-exact clone.")
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append("- Window: 2023-01-02 → 2026-04-29 (full panel)")
    md.append(f"- Universe (Core-class): median_turnover_60 ≥ cross-sectional q{int(LIQ_QUANTILE*100)} ∧ history_bars ≥ {HISTORY_FLOOR}")
    md.append(f"- Eligible per-date median: {int(panel.groupby('date')['eligible_core'].sum().median())}")
    md.append("- Signal: AL_F6 = HWO_up ∧ armed_recent10 ∧ refractory10 (close-source HW)")
    md.append("- Entry primary: signal at close(T) → enter at open(T+1)")
    md.append("- Entry secondary: signal at close(T) → enter at close(T) (optimistic)")
    md.append("- SAT short-entry: disabled")
    md.append(f"- Horizons: {HORIZONS}")
    md.append("")
    md.append("## Headline (entry=open_T1, primary)")
    md.append("")
    md.append("| K | n | uniq_dates | ratio_med | realized_med | win_rate | false_rev% | capture | PF |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summ.iterrows():
        md.append(
            f"| {int(r['K'])} | {int(r['n'])} | {int(r['uniq_dates'])} | "
            f"{r['ratio_med']:.2f} | {r['realized_med']:.4f} | "
            f"{r['win_rate']:.3f} | {r['false_rev_pct']:.1f} | "
            f"{r['mfe_capture_med']:.3f} | {r['pf_proxy']:.2f} |"
        )
    md.append("")
    md.append("## Headline (entry=close_T, secondary/optimistic)")
    md.append("")
    md.append("| K | n | uniq_dates | ratio_med | realized_med | win_rate | false_rev% | capture | PF |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summ2.iterrows():
        md.append(
            f"| {int(r['K'])} | {int(r['n'])} | {int(r['uniq_dates'])} | "
            f"{r['ratio_med']:.2f} | {r['realized_med']:.4f} | "
            f"{r['win_rate']:.3f} | {r['false_rev_pct']:.1f} | "
            f"{r['mfe_capture_med']:.3f} | {r['pf_proxy']:.2f} |"
        )
    md.append("")
    md.append("## Per-year (entry=open_T1)")
    md.append("")
    md.append("| year | K | n | uniq_dates | ratio_med | realized_med | win_rate | false_rev% | PF |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    py_full = agg[(agg["year"] != "ALL") & (agg["entry_mode"] == "open_T1")]
    py_full = py_full.sort_values(["year", "K"])
    for _, r in py_full.iterrows():
        md.append(
            f"| {r['year']} | {int(r['K'])} | {int(r['n'])} | {int(r['uniq_dates'])} | "
            f"{r['ratio_med']:.2f} | {r['realized_med']:.4f} | "
            f"{r['win_rate']:.3f} | {r['false_rev_pct']:.1f} | "
            f"{r['pf_proxy']:.2f} |"
        )
    md.append("")
    if not baseline_agg.empty:
        md.append("## Same-date random baseline")
        md.append("")
        md.append("Randomly sampled (ticker, date) pairs from the same eligible universe on")
        md.append("each fire date (no AL_F6 fire there). Read these rows as the null.")
        md.append("")
        md.append("| entry_mode | K | n | ratio_med | realized_med | win_rate | false_rev% |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in baseline_agg.sort_values(["entry_mode", "K"]).iterrows():
            md.append(
                f"| {r['entry_mode']} | {int(r['K'])} | {int(r['n'])} | "
                f"{r['ratio_med']:.2f} | {r['realized_med']:.4f} | "
                f"{r['win_rate']:.3f} | {r['false_rev_pct']:.1f} |"
            )
        md.append("")
    if dd_summary_rows:
        md.append("## Drawdown proxy (chronological cumulative equity, 1 unit/trade)")
        md.append("")
        md.append("| K | entry_mode | n_trades | final_equity | max_dd |")
        md.append("|---|---|---|---|---|")
        for r in dd_summary_rows:
            md.append(
                f"| {r['K']} | {r['entry_mode']} | {r['n_trades']} | "
                f"{r['final_equity_pct']:.4f} | {r['max_dd_pct']:.4f} |"
            )
        md.append("")
    md.append("## Concentration")
    md.append("")
    md.append(f"- Unique tickers firing: {fires['ticker'].nunique()}")
    md.append(f"- Unique fire dates: {fires['date'].nunique()}")
    md.append(f"- Top-50 ticker counts: `{OUT_CONCENTRATION.with_suffix('.by_ticker_top50.csv')}`")
    md.append(f"- Top-50 dense fire dates: `{OUT_CONCENTRATION.with_suffix('.by_date_top50.csv')}`")
    md.append("")
    md.append("Top-10 most-firing tickers:")
    md.append("")
    md.append("| ticker | fires |")
    md.append("|---|---|")
    for _, r in by_ticker.head(10).iterrows():
        md.append(f"| {r['ticker']} | {int(r['fires_total']) // (len(HORIZONS) * 2)} |")
    md.append("")
    md.append("Top-10 densest fire dates:")
    md.append("")
    md.append("| date | fires |")
    md.append("|---|---|")
    for _, r in by_date.head(10).iterrows():
        md.append(f"| {r['date'].date() if hasattr(r['date'], 'date') else r['date']} | "
                  f"{int(r['fires_total']) // (len(HORIZONS) * 2)} |")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- `{OUT_PER_TRADE}` — per fire × K × entry_mode")
    md.append(f"- `{OUT_AGG}` — aggregate (overall + per-year)")
    md.append(f"- `{OUT_BASELINE}` — same-date random baseline trades")
    md.append(f"- `{OUT_CONCENTRATION}` — per-ticker fire totals")
    md.append(f"- `{OUT_DD}` — chronological drawdown rows")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\n  wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
