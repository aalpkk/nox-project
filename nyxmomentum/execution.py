"""
Ex-ante execution proxies + realized fillability diagnostics.

**These are daily-OHLCV-derived proxies — NOT intraday truth.**
We do not measure true opening-auction volume, true fill rate, or true
quoted spread. The columns approximate execution friction using only
information that is strictly knowable at rebalance_date close.

Two role classes, kept separate on purpose:

  ex_ante         Computed from trailing windows ending on rebalance_date.
                  Safe to feed into the portfolio risk overlay. Columns are
                  prefixed 'proxy_' to make the approximation visible in
                  every downstream table.

  diagnostic_only Observed on day t+1 (first trading day after rebalance).
                  NEVER enters selection or overlay — doing so would leak
                  the future. Emitted only for evaluation (e.g. "how often
                  did our top-10 hit a limit-open the day we bought?").

If Matriks intraday is enabled in V2, the ex_ante proxies can be replaced
with true measurements; the diagnostic_only columns can be replaced with
real fill-rate / spread metrics at execution time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExecutionProxyConfig, EXECUTION_COLUMN_ROLES


EX_ANTE_COLS = [c for c, r in EXECUTION_COLUMN_ROLES.items() if r == "ex_ante"]
DIAGNOSTIC_COLS = [c for c, r in EXECUTION_COLUMN_ROLES.items() if r == "diagnostic_only"]


# ── Primitives ────────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    return _true_range(df).rolling(window, min_periods=window).mean()


# ── Per-ticker rolling proxy series ───────────────────────────────────────────

def compute_proxy_timeseries(df: pd.DataFrame,
                             config: ExecutionProxyConfig) -> pd.DataFrame:
    """
    Compute ex-ante proxy columns as daily time-series on a single ticker's
    full history. Downstream code indexes this by rebalance_date.

    All rolling windows are TRAILING — value at row t uses data up through t.
    """
    prev_close = df["Close"].shift(1)
    gap_abs = (df["Open"] - prev_close).abs()
    gap_pct = (df["Open"] - prev_close) / prev_close

    atr = _atr(df, config.atr_window)
    # Avoid division blow-up when ATR is 0 on very early bars / illiquid runs
    atr_safe = atr.replace(0.0, np.nan)

    open_dislocation = (gap_abs / atr_safe).rolling(
        config.short_lookback, min_periods=config.short_lookback
    ).median()

    stale_flag = ((df["Open"] == df["High"]) & (df["Open"] == df["Low"])).astype(float)
    stale_freq = stale_flag.rolling(
        config.long_lookback, min_periods=config.long_lookback
    ).mean()

    limit_open_flag = (gap_pct.abs() > config.limit_threshold).astype(float)
    limit_open_freq = limit_open_flag.rolling(
        config.long_lookback, min_periods=config.long_lookback
    ).mean()

    range_ratio = ((df["High"] - df["Low"]) / df["Close"]).replace([np.inf, -np.inf], np.nan)
    range_ratio_med = range_ratio.rolling(
        config.short_lookback, min_periods=config.short_lookback
    ).median()

    gap_dispersion = gap_pct.rolling(
        config.short_lookback, min_periods=config.short_lookback
    ).std()

    out = pd.DataFrame({
        "proxy_open_dislocation_20d":   open_dislocation,
        "proxy_stale_open_freq_60d":    stale_freq,
        "proxy_limit_open_freq_60d":    limit_open_freq,
        "proxy_daily_range_ratio_20d":  range_ratio_med,
        "proxy_gap_dispersion_20d":     gap_dispersion,
    }, index=df.index)
    return out


def compute_realized_timeseries(df: pd.DataFrame,
                                config: ExecutionProxyConfig) -> pd.DataFrame:
    """
    Diagnostic-only series. At row t, record what actually happened at t+1.
    **NEVER** feed back into selection/overlay — doing so leaks the future.

    Semantics (NaN propagation matters — "unknown" ≠ "did not trade"):

      realized_t1_has_open     1.0  if t+1 Open is finite and > 0
                               0.0  if t+1 row exists but Open is 0 / non-finite
                               NaN  if there is no t+1 row at all (series end)

      realized_t1_has_volume   1.0  if t+1 Volume is finite
                               0.0  if t+1 row exists but Volume is non-finite
                               NaN  if there is no t+1 row at all

      realized_t1_traded       1.0  if t+1 Volume > 0
                               0.0  if t+1 Volume == 0 (halt / no fills)
                               NaN  if Volume is unknown (series end or NaN)

      realized_next_open_gap   (t+1 Open − t Close) / t Close
                               NaN if either side is unknown

      realized_t1_limit_open   1.0 if |gap| > realized_gap_flag_threshold
                               0.0 if gap is measurable and within threshold
                               NaN if gap is unknown

    Having `has_open` and `has_volume` separate lets the user tell apart
    "we genuinely couldn't observe t+1" from "t+1 existed but was a halt".
    """
    # Detect existence of a t+1 row at all. Because the Series is shifted
    # in-place, the only guaranteed "no t+1" marker is the terminal row —
    # but intermediate NaNs in source data can also appear, so rely on
    # whether ANY of the t+1 OHLCV fields are observed.
    next_open = df["Open"].shift(-1)
    next_volume = df["Volume"].shift(-1)
    close_t = df["Close"]

    # A row "has t+1" if the shifted series produced any non-null field.
    # In well-formed daily OHLCV, Open and Volume are both NaN only at the
    # terminal row. We treat has_t1 as: any of them is finite.
    has_t1 = next_open.notna() | next_volume.notna()

    # has_open: open is finite and strictly positive.
    # - finite & > 0 → 1
    # - finite & ≤ 0 → 0 (degenerate)
    # - non-finite but has_t1 → 0 (t+1 existed, no open printed)
    # - no t+1 at all → NaN
    open_finite = np.isfinite(next_open.values)
    has_open_num = np.where(open_finite & (next_open.values > 0), 1.0, 0.0)
    has_open = pd.Series(has_open_num, index=df.index).where(has_t1, np.nan)

    # has_volume: volume is finite (may be 0).
    vol_finite = np.isfinite(next_volume.values)
    has_volume_num = np.where(vol_finite, 1.0, 0.0)
    has_volume = pd.Series(has_volume_num, index=df.index).where(has_t1, np.nan)

    # traded: Volume > 0. Only meaningful when Volume is known.
    traded_num = np.where(vol_finite & (next_volume.values > 0), 1.0, 0.0)
    traded = pd.Series(traded_num, index=df.index)
    traded = traded.where(pd.Series(vol_finite, index=df.index), np.nan)

    # Gap: only defined when both Close[t] and Open[t+1] are known & > 0.
    gap = (next_open - close_t) / close_t
    gap = gap.replace([np.inf, -np.inf], np.nan)
    gap = gap.where(close_t > 0, np.nan)

    # limit_open: defined only where gap is defined (NaN propagates).
    abs_gap = gap.abs()
    limit_open = pd.Series(np.nan, index=df.index)
    defined = abs_gap.notna()
    limit_open.loc[defined] = (abs_gap[defined] > config.realized_gap_flag_threshold).astype(float)

    out = pd.DataFrame({
        "realized_t1_has_open":    has_open,
        "realized_t1_has_volume":  has_volume,
        "realized_t1_traded":      traded,
        "realized_next_open_gap":  gap,
        "realized_t1_limit_open":  limit_open,
    }, index=df.index)
    return out


# ── Panel extension ───────────────────────────────────────────────────────────

def extend_universe_panel(universe_panel: pd.DataFrame,
                          panel: dict[str, pd.DataFrame],
                          config: ExecutionProxyConfig | None = None) -> pd.DataFrame:
    """
    Attach proxy + realized columns to the universe panel returned by
    universe.build_universe_panel(). Input panel is indexed rowwise by
    (ticker, rebalance_date); we look each pair up in the pre-computed
    per-ticker timeseries.

    Uses asof() so a rebalance_date falling on a halted/missing trading day
    picks up the most recent available value (and NaN if none exist).
    """
    cfg = config or ExecutionProxyConfig()

    # Pre-compute per-ticker proxy + realized series once
    proxies_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker, df in panel.items():
        if df is None or len(df) == 0:
            continue
        px = compute_proxy_timeseries(df, cfg)
        if cfg.emit_realized_diagnostics:
            rx = compute_realized_timeseries(df, cfg)
            px = pd.concat([px, rx], axis=1)
        proxies_by_ticker[ticker] = px

    rows: list[dict] = []
    empty_row = {c: None for c in (EX_ANTE_COLS + DIAGNOSTIC_COLS)}
    for _, r in universe_panel.iterrows():
        ticker = r["ticker"]
        rd = pd.Timestamp(r["rebalance_date"])
        px = proxies_by_ticker.get(ticker)
        if px is None or px.empty or rd < px.index[0]:
            rows.append(empty_row.copy())
            continue
        # asof returns the row whose index is the greatest ≤ rd
        lookup = px.asof(rd)
        if isinstance(lookup, pd.Series):
            row_dict = {k: (float(v) if pd.notna(v) else None) for k, v in lookup.items()}
        else:
            row_dict = empty_row.copy()
        rows.append(row_dict)

    add = pd.DataFrame(rows, index=universe_panel.index)
    # Preserve column order: original panel cols then ex_ante then diagnostic
    ordered_new = [c for c in EX_ANTE_COLS + DIAGNOSTIC_COLS if c in add.columns]
    add = add[ordered_new]
    return pd.concat([universe_panel, add], axis=1)


# ── Coverage report ───────────────────────────────────────────────────────────

def proxy_coverage_report(extended_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column NaN coverage + distribution. Separates ex_ante from
    diagnostic_only rows explicitly in the 'role' column so the user can
    verify only ex_ante columns end up in the overlay.
    """
    rows: list[dict] = []
    for col, role in EXECUTION_COLUMN_ROLES.items():
        if col not in extended_panel.columns:
            continue
        s = pd.to_numeric(extended_panel[col], errors="coerce")
        if len(s) == 0:
            continue
        nn = s.dropna()
        rows.append({
            "column": col,
            "role": role,
            "n_total": int(len(s)),
            "n_nonnull": int(len(nn)),
            "coverage": float(len(nn) / len(s)) if len(s) else 0.0,
            "mean":   float(nn.mean())   if len(nn) else None,
            "median": float(nn.median()) if len(nn) else None,
            "q25":    float(nn.quantile(0.25)) if len(nn) else None,
            "q75":    float(nn.quantile(0.75)) if len(nn) else None,
            "min":    float(nn.min())    if len(nn) else None,
            "max":    float(nn.max())    if len(nn) else None,
        })
    return pd.DataFrame(rows)


def column_role_manifest() -> pd.DataFrame:
    """Export the ex-ante vs diagnostic-only contract as a DataFrame for
    inclusion in reports. Makes the separation machine-readable."""
    return pd.DataFrame([
        {"column": c, "role": r, "enters_selection_overlay": r == "ex_ante"}
        for c, r in EXECUTION_COLUMN_ROLES.items()
    ])


# ── Ex-ante proxy pairwise correlation ────────────────────────────────────────

def ex_ante_correlation_matrix(extended_panel: pd.DataFrame,
                               method: str = "spearman") -> pd.DataFrame:
    """
    Pairwise correlation across the ex-ante proxies on the eligible subset of
    rows. Spearman by default — proxies are rank/fraction-like and not
    Gaussian. Restricting to eligible rows avoids letting low-quality / thin
    names dominate the off-diagonal structure.
    """
    cols = [c for c in EX_ANTE_COLS if c in extended_panel.columns]
    if not cols:
        return pd.DataFrame()
    sub = extended_panel
    if "eligible" in sub.columns:
        sub = sub.loc[sub["eligible"].astype(bool)]
    return sub[cols].corr(method=method)


def ex_ante_high_corr_pairs(corr: pd.DataFrame,
                            threshold: float = 0.7) -> pd.DataFrame:
    """
    Flatten a correlation matrix to unique off-diagonal pairs with
    |r| ≥ threshold. Surfaces proxies that would double-count inside the
    overlay's weighted-sum if given independent weights.
    """
    columns = ["col_a", "col_b", "corr", "abs_corr"]
    if corr.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({
                    "col_a": a,
                    "col_b": b,
                    "corr": float(r),
                    "abs_corr": float(abs(r)),
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
