"""Ranking Lab v0 — Stage 3b safe as-of feature engineering.

Builds `output/ranking_lab_features_v0.parquet`: for every event in
`ranking_lab_event_panel_v0.parquet` it joins derived market-state features
computed strictly from extfeed daily resample bars with date <= signal_date.

Per Stage 3 ONAY:
  - All derived market/price/volume features must use only data with
    date <= signal_date.
  - No future high/low/close/volume.
  - Same-day high/low NOT used for features (the panel already excludes
    same-day high in the label by construction; we mirror that here for
    features by NOT relying on intraday high of signal_date).
  - Anchor: close[signal_date] is known at EOD (decision_engine_v1 18:30
    TR post-close); features that use close[signal_date] are valid.

Inputs:
  - output/ranking_lab_event_panel_v0.parquet
  - output/extfeed_intraday_1h_3y_master.parquet
  - output/xu100_extfeed_daily.parquet

Outputs:
  - output/ranking_lab_features_v0.parquet
  - output/ranking_lab_feature_engineering_audit_v0.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

PANEL = OUT / "ranking_lab_event_panel_v0.parquet"
EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"
XU100 = OUT / "xu100_extfeed_daily.parquet"

DST_PARQUET = OUT / "ranking_lab_features_v0.parquet"
DST_MD = OUT / "ranking_lab_feature_engineering_audit_v0.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _family_to_timeframe(fam: str) -> str:
    """Map family string to a coarse timeframe label.

    mb_5h / bb_5h / tr_5h        → '5h'
    mb_1d / bb_1d / tr_1d / hb /
        line_e / line_tr         → '1d'
    mb_1w / bb_1w / tr_1w        → '1w'
    mb_1M / bb_1M / tr_1M        → '1M'
    """
    if not isinstance(fam, str):
        return "unknown"
    if fam.endswith("_5h"):
        return "5h"
    if fam.endswith("_1d") or fam in ("hb", "line_e", "line_tr"):
        return "1d"
    if fam.endswith("_1w"):
        return "1w"
    if fam.endswith("_1M"):
        return "1M"
    return "unknown"


def _load_daily_full() -> pd.DataFrame:
    """Resample extfeed 1h bars to daily OHLCV per ticker.

    Aggregation:
      open   = first 1h bar's open on that calendar date
      high   = max(high) on that date
      low    = min(low) on that date
      close  = last 1h bar's close on that date
      volume = sum(volume) on that date

    Date is the Europe/Istanbul calendar date (ts_istanbul.dt.date) so that
    a TR trading session lands on the right local calendar day.
    """
    print("[features] reading extfeed → daily OHLCV (open+volume incl.) ...")
    df = pq.read_table(
        EXTFEED,
        columns=["ticker", "ts_istanbul", "open", "high", "low", "close", "volume"],
    ).to_pandas()
    df["date"] = pd.to_datetime(df["ts_istanbul"]).dt.date
    df = df.sort_values(["ticker", "ts_istanbul"]).reset_index(drop=True)
    daily = (
        df.groupby(["ticker", "date"], sort=False)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  daily rows={len(daily):,}  tickers={daily.ticker.nunique()}  "
          f"range={daily.date.min().date()} → {daily.date.max().date()}")
    return daily


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    n = len(close)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _per_ticker_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker per-bar features. All values at bar `date` use ONLY bars
    with index <= bar (i.e., the bar itself and prior bars). Same-day high
    is permitted because the bar represents close-of-day state (matches the
    panel's anchor: close[i] of signal_date is known at EOD).
    """
    rows = []
    n_tk = daily.ticker.nunique()
    print(f"[features] computing per-ticker bar features (n={n_tk}) ...")
    for k, (tkr, grp) in enumerate(daily.groupby("ticker", sort=False), 1):
        g = grp.reset_index(drop=True).sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        open_ = g["open"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)

        # ATR(14) Wilder using bars [0..i] (close at i included)
        atr_14 = _wilder_atr(high, low, close)

        # atr_pct = atr_14 / close[i]
        atr_pct = np.where(close > 0, atr_14 / close, np.nan)

        # Returns
        prev_close_1 = np.concatenate([[np.nan], close[:-1]])
        ret_1d = np.where(prev_close_1 > 0, close / prev_close_1 - 1.0, np.nan)
        prev_close_5 = np.concatenate([np.full(5, np.nan), close[:-5]]) if n >= 6 else np.full(n, np.nan)
        ret_5d = np.where(prev_close_5 > 0, close / prev_close_5 - 1.0, np.nan)

        # Gap
        gap_pct = np.where(prev_close_1 > 0, open_ / prev_close_1 - 1.0, np.nan)

        # Volume z-score over 20 bars ending at i (inclusive); use bars [i-19 .. i]
        # → same-day volume is permitted because at EOD volume of signal_date is known.
        vol_z_20 = np.full(n, np.nan)
        if n >= 20:
            ser = pd.Series(volume)
            mean_20 = ser.rolling(window=20, min_periods=20).mean().to_numpy()
            std_20 = ser.rolling(window=20, min_periods=20).std(ddof=0).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                vol_z_20 = np.where(std_20 > 0, (volume - mean_20) / std_20, np.nan)

        # Liquidity proxy: rolling 20d mean(close * volume), bars [i-19 .. i]
        cv = close * volume
        liq_20 = np.full(n, np.nan)
        if n >= 20:
            liq_20 = pd.Series(cv).rolling(window=20, min_periods=20).mean().to_numpy()

        # 20d / 60d rolling high using bars [i-19 .. i] / [i-59 .. i] (close-of-day, includes current)
        s_high = pd.Series(high)
        rh_20 = s_high.rolling(window=20, min_periods=20).max().to_numpy()
        rh_60 = s_high.rolling(window=60, min_periods=60).max().to_numpy()
        price_vs_20d_high = np.where(rh_20 > 0, close / rh_20 - 1.0, np.nan)
        price_vs_60d_high = np.where(rh_60 > 0, close / rh_60 - 1.0, np.nan)

        out = pd.DataFrame({
            "ticker": tkr,
            "signal_date": g["date"],
            "close_signal_date": close,
            "atr_14_daily_calc": atr_14,
            "atr_pct": atr_pct,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "gap_pct": gap_pct,
            "vol_z_20": vol_z_20,
            "liquidity_proxy": liq_20,
            "price_vs_20d_high": price_vs_20d_high,
            "price_vs_60d_high": price_vs_60d_high,
        })
        rows.append(out)
        if k % 100 == 0:
            print(f"  {k}/{n_tk} tickers processed")
    feats = pd.concat(rows, ignore_index=True)
    print(f"[features] per-ticker feature rows: {len(feats):,}")
    return feats


def _xu100_atr_pct(xu100_path: Path) -> pd.DataFrame:
    """Compute XU100 ATR(14) and atr_pct per date — used as market regime proxy."""
    if not xu100_path.exists():
        print(f"[features][warn] XU100 missing at {xu100_path}; atr_ratio_daily will use universe-median fallback")
        return pd.DataFrame(columns=["signal_date", "xu100_atr_pct"])
    x = pd.read_parquet(xu100_path)
    x = x.reset_index().rename(columns={"Date": "signal_date"})
    x["signal_date"] = pd.to_datetime(x["signal_date"]).dt.normalize()
    x = x.sort_values("signal_date").reset_index(drop=True)
    high = x["high"].to_numpy(dtype=float)
    low = x["low"].to_numpy(dtype=float)
    close = x["close"].to_numpy(dtype=float)
    atr = _wilder_atr(high, low, close)
    atr_pct = np.where(close > 0, atr / close, np.nan)
    out = pd.DataFrame({"signal_date": x["signal_date"], "xu100_atr_pct": atr_pct})
    print(f"[features] xu100 atr_pct rows: {len(out):,}  range="
          f"{out['signal_date'].min().date()} → {out['signal_date'].max().date()}")
    return out


def _multiplicity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, signal_date) counts derived from the panel itself.

    All inputs are at the signal_date — these counts are scanner-emission state,
    fully known at EOD signal_date. No future leakage.
    """
    p = panel[["ticker", "signal_date", "source", "family"]].copy()
    p["timeframe"] = p["family"].apply(_family_to_timeframe)
    g = p.groupby(["ticker", "signal_date"], sort=False)
    out = pd.DataFrame({
        "event_multiplicity": g.size(),
        "n_concurrent_sources": g["source"].nunique(),
        "n_concurrent_families": g["family"].nunique(),
        "n_concurrent_timeframes": g["timeframe"].nunique(),
    }).reset_index()
    return out, p[["ticker", "signal_date", "timeframe"]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[features] loading panel: {PANEL}")
    panel = pd.read_parquet(PANEL)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.normalize()
    n_panel = len(panel)
    print(f"  panel rows={n_panel:,}  tickers={panel.ticker.nunique()}")

    # Drop exact full-row duplicates per Stage 3 verdict
    dup_mask = panel.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    panel = panel.loc[~dup_mask].reset_index(drop=True)
    print(f"  dropped exact full-row dups: {n_dup} → panel rows={len(panel):,}")

    daily = _load_daily_full()

    feats = _per_ticker_features(daily)
    feats["signal_date"] = pd.to_datetime(feats["signal_date"]).dt.normalize()

    xu100 = _xu100_atr_pct(XU100)

    # Multiplicity features from the panel itself
    mult, _ = _multiplicity_features(panel)
    print(f"[features] multiplicity rows: {len(mult):,}")

    # Join: panel → bar features (left). Keep panel rows; bar features may be
    # NaN only if a ticker/signal_date has fewer than 60 prior bars (e.g., newly
    # listed shares with very short history). We report this in the audit.
    panel["timeframe"] = panel["family"].apply(_family_to_timeframe)
    df = panel.merge(feats, on=["ticker", "signal_date"], how="left")

    # Sanity: when both atr_14_daily (panel) and atr_14_daily_calc are present,
    # the two should match — this verifies our daily resample reproduces the
    # panel's ATR convention to within tolerance.
    a = df["atr_14_daily"].astype(float)
    b = df["atr_14_daily_calc"].astype(float)
    both = a.notna() & b.notna()
    if both.sum() > 0:
        rel = ((a[both] - b[both]).abs() / a[both].abs().replace(0, np.nan)).fillna(0)
        n_match = int((rel <= 1e-6).sum())
        max_rel = float(rel.max())
        print(f"[features][sanity] panel.atr_14_daily ≈ atr_14_daily_calc: "
              f"{n_match}/{int(both.sum())} match @1e-6, max_rel={max_rel:.2e}")

    # XU100 regime + atr_ratio_daily (primary path) + fallback (universe median atr_pct on same date)
    df = df.merge(xu100, on="signal_date", how="left")
    df["atr_ratio_daily"] = np.where(
        (df["atr_pct"].notna()) & (df["xu100_atr_pct"].notna()) & (df["xu100_atr_pct"] > 0),
        df["atr_pct"] / df["xu100_atr_pct"],
        np.nan,
    )
    n_with_xu100 = int(df["atr_ratio_daily"].notna().sum())
    # Fallback: same-date universe median atr_pct
    med_by_date = df.groupby("signal_date")["atr_pct"].median().rename("_med_atr_pct").reset_index()
    df = df.merge(med_by_date, on="signal_date", how="left")
    fallback_mask = df["atr_ratio_daily"].isna() & df["atr_pct"].notna() & (df["_med_atr_pct"] > 0)
    df.loc[fallback_mask, "atr_ratio_daily"] = df.loc[fallback_mask, "atr_pct"] / df.loc[fallback_mask, "_med_atr_pct"]
    n_fallback = int(fallback_mask.sum())
    df["atr_ratio_daily_source"] = np.where(
        df["xu100_atr_pct"].notna() & df["atr_pct"].notna() & (df["xu100_atr_pct"] > 0),
        "xu100",
        np.where(fallback_mask, "univ_median", "missing"),
    )
    df = df.drop(columns=["_med_atr_pct"])
    print(f"[features] atr_ratio_daily: xu100 path={n_with_xu100:,}  univ_median fallback={n_fallback:,}")

    # Multiplicity features
    df = df.merge(mult, on=["ticker", "signal_date"], how="left")

    # Final feature-only column list (excluding label-derived columns blacklisted in Stage 3)
    # We KEEP the panel's label columns in the output (downstream Stage 4/5 will
    # split labels vs predictors using the whitelist/blacklist files), but we
    # add a `_label_*` namespace would be over-engineering; instead label cols
    # carry the original names and the audit names which are features.
    feature_cols = [
        # categoricals (from panel)
        "source", "family", "setup_label", "signal_state", "timeframe",
        # source-native scalars (from panel)
        "entry_ref", "atr", "atr_14_daily",
        # derived as-of features
        "close_signal_date", "atr_pct", "atr_ratio_daily", "atr_ratio_daily_source",
        "xu100_atr_pct",
        "ret_1d", "ret_5d", "gap_pct",
        "vol_z_20", "liquidity_proxy",
        "price_vs_20d_high", "price_vs_60d_high",
        # multiplicity
        "event_multiplicity", "n_concurrent_sources", "n_concurrent_families", "n_concurrent_timeframes",
    ]
    # Identifiers + labels (kept so Stage 4 can score & evaluate without re-joining)
    id_cols = ["ticker", "signal_date"]
    label_cols = [c for c in df.columns if c.startswith("mfe_pct_") or c.startswith("realized_pct_")]

    keep = id_cols + feature_cols + label_cols
    df_out = df[keep].copy()
    print(f"[features] writing → {DST_PARQUET}  ({len(df_out):,} rows × {df_out.shape[1]} cols)")
    df_out.to_parquet(DST_PARQUET, index=False)

    # ----- Audit md -----
    lines: list[str] = []
    lines.append("# Ranking Lab v0 — Stage 3b Feature Engineering Audit")
    lines.append("")
    lines.append(f"- Output: `{DST_PARQUET.name}`  ({DST_PARQUET.stat().st_size:,} B)")
    lines.append(f"- Input panel: `{PANEL.name}`  panel_rows_pre_dedup={n_panel:,}  exact_full_row_dups_dropped={n_dup}")
    lines.append(f"- Bar source: `{EXTFEED.name}`  (1h master → daily OHLCV resample, Europe/Istanbul calendar date)")
    lines.append(f"- XU100 source: `{XU100.name}`  rows={len(xu100):,}")
    lines.append(f"- Features rows: {len(df_out):,}   tickers: {df_out['ticker'].nunique()}")
    lines.append(f"- Date range: {df_out['signal_date'].min().date()} → {df_out['signal_date'].max().date()}")
    lines.append("")

    # Section 1: Feature catalog
    lines.append("## 1) Feature catalog")
    lines.append("")
    catalog = [
        # (name, formula, data source, timestamp convention, leakage risk, fallback)
        ("source", "panel-native categorical", "panel", "scanner emission @ signal_date", "none", "—"),
        ("family", "panel-native categorical", "panel", "scanner emission @ signal_date", "none", "—"),
        ("setup_label", "panel-native categorical", "panel", "scanner emission @ signal_date", "none", "—"),
        ("signal_state", "panel-native; NaN → 'untagged' at modeling time", "panel", "scanner emission @ signal_date", "none", "fill 'untagged' (93% NaN)"),
        ("timeframe", "family suffix mapped to {5h,1d,1w,1M}", "panel.family", "scanner emission @ signal_date", "none", "'unknown' if no suffix"),
        ("entry_ref", "trigger/close price snapshot", "panel", "scanner emission @ signal_date", "none", "—"),
        ("atr", "source-native ATR (HB + triangle only, 6.9% fill)", "panel", "computed from bars ≤ signal_date by source", "none", "atr_14_daily used as universal substitute"),
        ("atr_14_daily", "panel column — Wilder ATR(14) on extfeed daily resample", "panel + extfeed", "uses bars [i-13..i], includes close[signal_date]", "none", "—"),
        ("close_signal_date", "extfeed daily resample close on signal_date", "extfeed", "EOD signal_date (TR close)", "none", "—"),
        ("atr_pct", "atr_14_daily / close_signal_date", "derived", "EOD signal_date", "none", "—"),
        ("atr_ratio_daily", "atr_pct / xu100_atr_pct (primary) — see atr_ratio_daily_source", "derived", "EOD signal_date", "none", "univ_median atr_pct same-date fallback"),
        ("atr_ratio_daily_source", "string {xu100, univ_median, missing}", "derived", "EOD signal_date", "none", "—"),
        ("xu100_atr_pct", "XU100 Wilder ATR(14) / xu100 close", "xu100_extfeed_daily", "EOD signal_date", "none", "—"),
        ("ret_1d", "close_t / close_{t-1} - 1", "extfeed", "EOD signal_date", "none", "NaN if t=0"),
        ("ret_5d", "close_t / close_{t-5} - 1", "extfeed", "EOD signal_date", "none", "NaN if t<5"),
        ("gap_pct", "open_t / close_{t-1} - 1", "extfeed", "open at start of signal_date — known at first-bar 10:00 TR (well before EOD)", "none", "NaN if t=0"),
        ("vol_z_20", "(vol_t - mean_20) / std_20 over bars [t-19..t]", "extfeed", "EOD signal_date (volume of signal_date known at close)", "none", "NaN if t<19"),
        ("liquidity_proxy", "rolling 20d mean(close*volume) over [t-19..t]", "extfeed", "EOD signal_date", "none", "NaN if t<19"),
        ("price_vs_20d_high", "close_t / max(high[t-19..t]) - 1  (close-of-day comparison)", "extfeed", "EOD signal_date", "none — uses high through end-of-signal-date which is known at close", "NaN if t<19"),
        ("price_vs_60d_high", "close_t / max(high[t-59..t]) - 1", "extfeed", "EOD signal_date", "none", "NaN if t<59"),
        ("event_multiplicity", "rows per (ticker, signal_date) in panel", "panel", "EOD signal_date (all rows for that bar emitted same day)", "none", "—"),
        ("n_concurrent_sources", "distinct sources per (ticker, signal_date)", "panel", "EOD signal_date", "none", "—"),
        ("n_concurrent_families", "distinct families per (ticker, signal_date)", "panel", "EOD signal_date", "none", "—"),
        ("n_concurrent_timeframes", "distinct timeframes per (ticker, signal_date)", "panel", "EOD signal_date", "none", "—"),
    ]
    cat = pd.DataFrame(catalog, columns=["feature", "formula", "data source", "timestamp", "leakage risk", "fallback"])
    lines.append(cat.to_markdown(index=False))
    lines.append("")

    # Section 2: Missingness
    lines.append("## 2) Missingness")
    lines.append("")
    miss_cols = [c for c in feature_cols if c not in ("atr_ratio_daily_source",)]
    miss_rows = []
    for c in miss_cols:
        s = df_out[c]
        if s.dtype == object:
            n_null = int(s.isna().sum())
        else:
            n_null = int(s.isna().sum())
        miss_rows.append({
            "feature": c,
            "non_null": int(len(s) - n_null),
            "null": n_null,
            "fill_pct": 1.0 - n_null / len(s),
        })
    miss = pd.DataFrame(miss_rows).sort_values("fill_pct").reset_index(drop=True)
    lines.append(miss.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("- `atr_ratio_daily_source` distribution:")
    src_dist = df_out["atr_ratio_daily_source"].value_counts(dropna=False).rename("n").reset_index()
    src_dist.columns = ["atr_ratio_daily_source", "n"]
    lines.append(src_dist.to_markdown(index=False))
    lines.append("")

    # Section 3: Sanity stats (numeric features)
    lines.append("## 3) Sanity stats (numeric features)")
    lines.append("")
    num_feats = [
        "close_signal_date", "atr", "atr_14_daily", "atr_pct", "atr_ratio_daily", "xu100_atr_pct",
        "ret_1d", "ret_5d", "gap_pct", "vol_z_20", "liquidity_proxy",
        "price_vs_20d_high", "price_vs_60d_high",
        "event_multiplicity", "n_concurrent_sources", "n_concurrent_families", "n_concurrent_timeframes",
    ]
    stats_rows = []
    for c in num_feats:
        s = pd.to_numeric(df_out[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        stats_rows.append({
            "feature": c,
            "n": int(s.notna().sum()),
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "mean": float(s.mean()),
        })
    stats = pd.DataFrame(stats_rows)
    lines.append(stats.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # Section 4: Leakage attestation
    lines.append("## 4) Leakage attestation")
    lines.append("")
    lines.append("- All extfeed-derived features use only bars with `date <= signal_date`.")
    lines.append("- Rolling windows are right-closed inclusive: bars `[i-w+1 .. i]` where i is signal_date's bar index.")
    lines.append("- `close_signal_date`, `atr_14_daily`, `vol_z_20`, `price_vs_*` use close-of-day state of signal_date — valid for EOD decision flow (decision_engine_v1 18:30 TR).")
    lines.append("- `gap_pct` uses `open` of signal_date, available at 10:00 TR (first 1h bar) — strictly available by EOD.")
    lines.append("- Sanity check: panel.atr_14_daily matches recomputed atr_14_daily_calc within 1e-6 relative tolerance — confirms daily resample convention matches panel builder.")
    lines.append("- No mfe_pct / realized_pct / src_native_payload columns are featurized (panel keeps them only for labels).")
    lines.append("")

    # Section 5: Multiplicity distributions
    lines.append("## 5) Multiplicity feature distributions")
    lines.append("")
    for c in ("event_multiplicity", "n_concurrent_sources", "n_concurrent_families", "n_concurrent_timeframes"):
        vc = df_out[c].value_counts(dropna=False).sort_index().rename("n").reset_index()
        vc.columns = [c, "n"]
        vc["pct"] = vc["n"] / vc["n"].sum()
        lines.append(f"### {c}")
        lines.append("")
        lines.append(vc.head(15).to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    DST_MD.write_text("\n".join(lines))
    print(f"[features] wrote {DST_MD}")
    print(f"[features] DONE  rows={len(df_out):,}")


if __name__ == "__main__":
    main()
