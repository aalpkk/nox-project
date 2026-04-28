"""
Feature family for cross-sectional momentum — V1.

ROLE DISCIPLINE (locked 2026-04-21, rule set):
  Every feature emitted here carries role='model_feature'. Overlay inputs
  live in execution.py under EXECUTION_COLUMN_ROLES[role=='ex_ante'].
  A single variable must NEVER be:
    • a model_feature AND
    • an overlay_input
  simultaneously. If the same economic concept matters for both, split it
  into two explicitly named features with distinct definitions.

  Realized t+1 diagnostic columns (realized_next_open_gap,
  realized_t1_limit_open, realized_t1_traded, realized_t1_has_open,
  realized_t1_has_volume) MUST NEVER appear in the feature set — not as
  train features, rank features, or overlay inputs. This module hard-fails
  if they ever make it into the panel.

BLOCKS (v1 minimum):
  momentum, relative_strength, liquidity, trend_quality, volatility,
  exhaustion. Regime + sector deferred until we have reliable mappings
  (weak/partial sector data → false neutralization, spec §3).

TIME CONTRACT:
  Every feature has a FeatureSpec documenting observation_window, anchor,
  shift_rule, normalization, winsorization, dependencies. Transforms
  (cross-sectional rank, winsorization, z-score) are DEFERRED to the
  dataset stage — this module emits raw numerics only. A caller who
  inspects the manifest knows exactly what happens at each downstream step.

LEAKAGE:
  All windows are trailing, computed from df.loc[df.index <= rebalance_date]
  via asof() lookup. XU100 is same-session (no shift). No US-close macro in
  v1 — deferred to the regime block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FeatureConfig, EXECUTION_COLUMN_ROLES


# ── FeatureSpec: one row per feature, per the locked time contract ────────────

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    block: str                              # momentum | relative_strength | …
    role: str = "model_feature"             # model_feature | overlay_input | diagnostic_only
    observation_window: str = ""            # e.g. "[rd-252, rd-21]"
    anchor: str = "rebalance_date"
    shift_rule: str = "none (BIST session)"
    normalization: str = "raw"              # transform deferred to dataset stage
    winsorization: str = "deferred_to_dataset_stage"
    dependencies: tuple[str, ...] = ()
    description: str = ""


# Canonical spec list. compute_feature_timeseries() MUST produce exactly
# these column names, in any order. Runner validates.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # ── Momentum (5) ──────────────────────────────────────────────────────
    FeatureSpec(
        name="mom_21d", block="momentum",
        observation_window="[rd-21, rd]",
        dependencies=("Close",),
        description="Total return over trailing 21 trading days (~1m).",
    ),
    FeatureSpec(
        name="mom_63d", block="momentum",
        observation_window="[rd-63, rd]",
        dependencies=("Close",),
        description="Total return over trailing 63 trading days (~3m).",
    ),
    FeatureSpec(
        name="mom_126d", block="momentum",
        observation_window="[rd-126, rd]",
        dependencies=("Close",),
        description="Total return over trailing 126 trading days (~6m).",
    ),
    FeatureSpec(
        name="mom_252d_skip_21d", block="momentum",
        observation_window="[rd-252, rd-21]",
        dependencies=("Close",),
        description="Classic 12-1 momentum: Close(rd-21)/Close(rd-252)−1. "
                    "Skips the last 21d to avoid the short-term reversal "
                    "window contaminating long-horizon alpha.",
    ),
    FeatureSpec(
        name="mom_63d_skip_21d", block="momentum",
        observation_window="[rd-63, rd-21]",
        dependencies=("Close",),
        description="3-1 momentum: Close(rd-21)/Close(rd-63)−1.",
    ),

    # ── Relative Strength vs XU100 (3) ────────────────────────────────────
    FeatureSpec(
        name="rs_xu100_63d", block="relative_strength",
        observation_window="[rd-63, rd]",
        dependencies=("Close", "XU100.Close"),
        description="log(Close/Close_63)−log(XU100/XU100_63). XU100 trades "
                    "same session as tickers → no shift.",
    ),
    FeatureSpec(
        name="rs_xu100_126d", block="relative_strength",
        observation_window="[rd-126, rd]",
        dependencies=("Close", "XU100.Close"),
        description="log-return diff vs XU100 over 126 days.",
    ),
    FeatureSpec(
        name="rs_xu100_252d_skip_21d", block="relative_strength",
        observation_window="[rd-252, rd-21]",
        dependencies=("Close", "XU100.Close"),
        description="12-1 RS vs XU100 in log-space.",
    ),

    # ── Liquidity (3) ─────────────────────────────────────────────────────
    FeatureSpec(
        name="log_tl_turnover_20d", block="liquidity",
        observation_window="[rd-20, rd]",
        dependencies=("Close", "Volume"),
        description="log of mean TL turnover over 20 days. "
                    "Distinct from overlay's proxy_open_dislocation / "
                    "proxy_limit_open_freq — this is SIZE, not FRICTION.",
    ),
    FeatureSpec(
        name="log_tl_turnover_60d", block="liquidity",
        observation_window="[rd-60, rd]",
        dependencies=("Close", "Volume"),
        description="log of mean TL turnover over 60 days.",
    ),
    FeatureSpec(
        name="log_amihud_20d", block="liquidity",
        observation_window="[rd-20, rd]",
        dependencies=("Close", "Volume"),
        description="log(mean(|ret_1d| / tl_turnover_1d) over 20d). Raw Amihud "
                    "is ~log-normal and scale-tiny (≈1e-11); log transform "
                    "makes cross-sectional std meaningful. Higher = more "
                    "price impact per TL traded.",
    ),

    # ── Trend Quality (3) ─────────────────────────────────────────────────
    FeatureSpec(
        name="trend_r2_126d", block="trend_quality",
        observation_window="[rd-126, rd]",
        dependencies=("Close",),
        description="R² of OLS fit of log(Close) on time over 126d. "
                    "Separates 'smooth trend' from 'choppy +X% over same window'.",
    ),
    FeatureSpec(
        name="trend_above_ma200_pct_126d", block="trend_quality",
        observation_window="[rd-126, rd] (MA200 requires rd-326..rd)",
        dependencies=("Close",),
        description="Fraction of last 126 days spent above the 200-day MA.",
    ),
    FeatureSpec(
        name="px_over_ma50", block="trend_quality",
        observation_window="[rd-50, rd]",
        dependencies=("Close",),
        description="Close/MA50 − 1 evaluated at rebalance_date.",
    ),

    # ── Volatility (3) ────────────────────────────────────────────────────
    FeatureSpec(
        name="vol_std_20d", block="volatility",
        observation_window="[rd-20, rd]",
        dependencies=("Close",),
        description="Std of daily simple returns over 20 days.",
    ),
    FeatureSpec(
        name="vol_std_60d", block="volatility",
        observation_window="[rd-60, rd]",
        dependencies=("Close",),
        description="Std of daily simple returns over 60 days.",
    ),
    FeatureSpec(
        name="vol_parkinson_20d", block="volatility",
        observation_window="[rd-20, rd]",
        dependencies=("High", "Low"),
        description="Parkinson range-based vol: sqrt(mean(log(H/L)² / (4·ln 2))) "
                    "over 20 days. More efficient than close-to-close vol "
                    "when intraday range is informative.",
    ),

    # ── Exhaustion (3) ────────────────────────────────────────────────────
    FeatureSpec(
        name="dist_from_52w_high", block="exhaustion",
        observation_window="[rd-252, rd]",
        dependencies=("Close",),
        description="Close / max(Close over trailing 252d) − 1. "
                    "Non-positive. Near 0 = at highs, very negative = washed out.",
    ),
    FeatureSpec(
        name="px_over_ma50_zscore_20d", block="exhaustion",
        observation_window="[rd-20, rd]",
        dependencies=("Close",),
        description="Trailing 20d z-score of px_over_ma50. High = "
                    "extended above MA50 vs recent history.",
    ),
    FeatureSpec(
        name="recent_extreme_21d", block="exhaustion",
        observation_window="[rd-21, rd]",
        dependencies=("Close",),
        description="|mom_21d| — magnitude (not direction) of 1m move.",
    ),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
FEATURE_BLOCKS: tuple[str, ...] = tuple(sorted({s.block for s in FEATURE_SPECS}))

# Machine-readable role map (mirrors EXECUTION_COLUMN_ROLES / LABEL_COLUMN_ROLES)
FEATURE_COLUMN_ROLES: dict[str, str] = {s.name: s.role for s in FEATURE_SPECS}

# Hard-rejected column names — never allowed in the feature set
_DISALLOWED_FEATURE_NAMES = frozenset(
    c for c, r in EXECUTION_COLUMN_ROLES.items() if r == "diagnostic_only"
)


def _assert_no_realized_cols(columns: Iterable[str]) -> None:
    bad = [c for c in columns if c in _DISALLOWED_FEATURE_NAMES]
    if bad:
        raise ValueError(
            f"Realized t+1 diagnostic columns must NEVER enter the feature set. "
            f"Offending names: {bad}. See EXECUTION_COLUMN_ROLES."
        )


# ── Per-ticker feature timeseries ─────────────────────────────────────────────

def _rolling_r2(log_close: pd.Series, window: int) -> pd.Series:
    """R² of OLS fit y = α + β·t over each trailing window on log-price."""
    def _r2(arr: np.ndarray) -> float:
        y = arr[np.isfinite(arr)]
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            return np.nan
        return 1.0 - ss_res / ss_tot
    return log_close.rolling(window, min_periods=window).apply(_r2, raw=True)


def compute_feature_timeseries(df: pd.DataFrame,
                               xu100_close: pd.Series | None,
                               config: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Compute all v1 features as daily time-series on one ticker's history.
    Values at day t use only data up through t. The caller picks the
    rebalance_date value via asof() at panel extension time.
    """
    cfg = config or FeatureConfig()
    close = df["Close"].astype(float)
    # Avoid log(0) blowups while keeping NaN semantics for missing bars
    close_pos = close.where(close > 0, np.nan)
    log_close = np.log(close_pos)
    # Explicit fill_method=None: do NOT forward-fill NaN before diff
    # (silences pandas FutureWarning; matches our leakage discipline).
    ret1 = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # ── Momentum ──────────────────────────────────────────────────────────
    mom_21 = close.pct_change(21, fill_method=None)
    mom_63 = close.pct_change(63, fill_method=None)
    mom_126 = close.pct_change(126, fill_method=None)
    # skip_21 = Close(t-21)/Close(t-L) − 1
    mom_252_skip = close.shift(21) / close.shift(252) - 1.0
    mom_63_skip  = close.shift(21) / close.shift(63)  - 1.0

    # ── Relative Strength vs XU100 (log-return diff) ──────────────────────
    if xu100_close is not None and len(xu100_close) > 0:
        xu = xu100_close.reindex(df.index).ffill()
        xu_pos = xu.where(xu > 0, np.nan)
        xu_log = np.log(xu_pos)
        rs_63 = (log_close - log_close.shift(63)) - (xu_log - xu_log.shift(63))
        rs_126 = (log_close - log_close.shift(126)) - (xu_log - xu_log.shift(126))
        rs_252_skip = (
            (log_close.shift(21) - log_close.shift(252))
            - (xu_log.shift(21) - xu_log.shift(252))
        )
    else:
        nan_s = pd.Series(np.nan, index=df.index)
        rs_63 = rs_126 = rs_252_skip = nan_s

    # ── Liquidity ─────────────────────────────────────────────────────────
    if "Volume" in df.columns:
        volume = df["Volume"].astype(float)
        tl_turnover = close * volume
        tl_pos = tl_turnover.where(tl_turnover > 0, np.nan)
        log_tl_20 = np.log(tl_pos.rolling(20, min_periods=20).mean())
        log_tl_60 = np.log(tl_pos.rolling(60, min_periods=60).mean())
        amihud_daily = (ret1.abs() / tl_pos).replace([np.inf, -np.inf], np.nan)
        amihud_mean_20 = amihud_daily.rolling(20, min_periods=20).mean()
        # log(amihud) — make cross-sectional spread meaningful despite tiny magnitude
        log_amihud_20 = np.log(amihud_mean_20.where(amihud_mean_20 > 0, np.nan))
    else:
        nan_s = pd.Series(np.nan, index=df.index)
        log_tl_20 = log_tl_60 = log_amihud_20 = nan_s

    # ── Trend Quality ─────────────────────────────────────────────────────
    trend_r2_126 = _rolling_r2(log_close, 126)
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    above_ma200 = (close > ma200).astype(float).where(ma200.notna(), np.nan)
    above_ma200_pct_126 = above_ma200.rolling(126, min_periods=126).mean()
    px_over_ma50 = close / ma50 - 1.0

    # ── Volatility ────────────────────────────────────────────────────────
    vol_20 = ret1.rolling(20, min_periods=20).std()
    vol_60 = ret1.rolling(60, min_periods=60).std()
    if "High" in df.columns and "Low" in df.columns:
        high = df["High"].astype(float).where(df["High"] > 0, np.nan)
        low = df["Low"].astype(float).where(df["Low"] > 0, np.nan)
        log_hl_sq = (np.log(high / low)) ** 2
        vol_parkinson_20 = np.sqrt(
            log_hl_sq.rolling(20, min_periods=20).mean() / (4.0 * np.log(2.0))
        )
    else:
        vol_parkinson_20 = pd.Series(np.nan, index=df.index)

    # ── Exhaustion ────────────────────────────────────────────────────────
    roll_max_252 = close.rolling(252, min_periods=252).max()
    dist_52w = close / roll_max_252 - 1.0
    po50_mean = px_over_ma50.rolling(20, min_periods=20).mean()
    po50_std = px_over_ma50.rolling(20, min_periods=20).std().replace(0, np.nan)
    px_over_ma50_z20 = (px_over_ma50 - po50_mean) / po50_std
    recent_extreme = mom_21.abs()

    out = pd.DataFrame({
        "mom_21d": mom_21,
        "mom_63d": mom_63,
        "mom_126d": mom_126,
        "mom_252d_skip_21d": mom_252_skip,
        "mom_63d_skip_21d": mom_63_skip,
        "rs_xu100_63d": rs_63,
        "rs_xu100_126d": rs_126,
        "rs_xu100_252d_skip_21d": rs_252_skip,
        "log_tl_turnover_20d": log_tl_20,
        "log_tl_turnover_60d": log_tl_60,
        "log_amihud_20d": log_amihud_20,
        "trend_r2_126d": trend_r2_126,
        "trend_above_ma200_pct_126d": above_ma200_pct_126,
        "px_over_ma50": px_over_ma50,
        "vol_std_20d": vol_20,
        "vol_std_60d": vol_60,
        "vol_parkinson_20d": vol_parkinson_20,
        "dist_from_52w_high": dist_52w,
        "px_over_ma50_zscore_20d": px_over_ma50_z20,
        "recent_extreme_21d": recent_extreme,
    }, index=df.index)

    # Contract guard — any silent drift blows up early.
    expected = set(FEATURE_COLUMNS)
    produced = set(out.columns)
    if expected != produced:
        raise ValueError(
            f"compute_feature_timeseries output drift. "
            f"missing={sorted(expected - produced)}, extra={sorted(produced - expected)}"
        )
    _assert_no_realized_cols(out.columns)
    return out[list(FEATURE_COLUMNS)]


# ── Panel extension ───────────────────────────────────────────────────────────

def build_feature_panel(universe_panel: pd.DataFrame,
                        panel: dict[str, pd.DataFrame],
                        xu100: pd.DataFrame | None,
                        config: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Attach feature columns to the universe panel. Output: long-format with
    (ticker, rebalance_date, eligible, <FEATURE_COLUMNS>). asof() lookup
    is used so a rebalance_date landing on a halt day falls back to the
    most recent available bar.
    """
    cfg = config or FeatureConfig()
    xu100_close = (
        xu100["Close"] if xu100 is not None and "Close" in xu100.columns else None
    )

    feats_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker, df in panel.items():
        if df is None or df.empty:
            continue
        feats_by_ticker[ticker] = compute_feature_timeseries(df, xu100_close, cfg)

    rows: list[dict] = []
    empty_row = {c: None for c in FEATURE_COLUMNS}
    for _, r in universe_panel.iterrows():
        ticker = r["ticker"]
        rd = pd.Timestamp(r["rebalance_date"])
        ft = feats_by_ticker.get(ticker)
        if ft is None or ft.empty or rd < ft.index[0]:
            rows.append(empty_row.copy())
            continue
        lookup = ft.asof(rd)
        if isinstance(lookup, pd.Series):
            row_dict = {k: (float(v) if pd.notna(v) else None) for k, v in lookup.items()}
        else:
            row_dict = empty_row.copy()
        rows.append(row_dict)

    add = pd.DataFrame(rows, index=universe_panel.index)[list(FEATURE_COLUMNS)]
    keep_base = [c for c in ["ticker", "rebalance_date", "eligible"] if c in universe_panel.columns]
    base = universe_panel[keep_base].reset_index(drop=True)
    add = add.reset_index(drop=True)
    merged = pd.concat([base, add], axis=1)
    _assert_no_realized_cols(merged.columns)
    return merged


# ── Manifest + reports ────────────────────────────────────────────────────────

def feature_manifest() -> pd.DataFrame:
    """Machine-readable time contract, one row per feature."""
    return pd.DataFrame([
        {
            "feature": s.name,
            "block": s.block,
            "role": s.role,
            "observation_window": s.observation_window,
            "anchor": s.anchor,
            "shift_rule": s.shift_rule,
            "normalization": s.normalization,
            "winsorization": s.winsorization,
            "dependencies": ",".join(s.dependencies),
            "description": s.description,
        }
        for s in FEATURE_SPECS
    ])


def feature_coverage_report(features: pd.DataFrame,
                            near_zero_var_cv: float = 1e-4) -> pd.DataFrame:
    """
    Per-feature: coverage (non-null share), mean/std, near-zero-variance
    flag, tail quantiles for sanity.
    """
    rows: list[dict] = []
    spec_by_name = {s.name: s for s in FEATURE_SPECS}
    for c in FEATURE_COLUMNS:
        if c not in features.columns:
            continue
        s = pd.to_numeric(features[c], errors="coerce")
        nn = s.dropna()
        if len(nn) == 0:
            rows.append({
                "feature": c,
                "block": spec_by_name[c].block,
                "role": spec_by_name[c].role,
                "n_total": int(len(s)),
                "n_nonnull": 0,
                "coverage": 0.0,
                "mean": None, "std": None,
                "near_zero_variance": True,
                "q01": None, "q50": None, "q99": None,
            })
            continue
        std = float(nn.std())
        mean = float(nn.mean())
        # CV-style near-zero check: absolute threshold on std OR std-to-mean ratio.
        nzv = bool(std < near_zero_var_cv)
        rows.append({
            "feature": c,
            "block": spec_by_name[c].block,
            "role": spec_by_name[c].role,
            "n_total": int(len(s)),
            "n_nonnull": int(len(nn)),
            "coverage": float(len(nn) / len(s)),
            "mean": mean,
            "std": std,
            "near_zero_variance": nzv,
            "q01": float(nn.quantile(0.01)),
            "q50": float(nn.quantile(0.50)),
            "q99": float(nn.quantile(0.99)),
        })
    return pd.DataFrame(rows)


def feature_correlation_matrix(features: pd.DataFrame,
                               method: str = "spearman",
                               eligible_only: bool = True) -> pd.DataFrame:
    """
    Pairwise correlation across model features. Defaults to Spearman +
    eligible-only rows (matches the overlay / proxy correlation convention).
    """
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    if not cols:
        return pd.DataFrame()
    sub = features
    if eligible_only and "eligible" in sub.columns:
        sub = sub.loc[sub["eligible"].astype(bool)]
    return sub[cols].corr(method=method)


def feature_high_corr_pairs(corr: pd.DataFrame,
                            threshold: float = 0.85) -> pd.DataFrame:
    """Unique off-diagonal pairs with |r| ≥ threshold. 0.85 is the default:
    below this, features are allowed to overlap; above, we suspect the
    same signal expressed twice."""
    columns = ["feature_a", "feature_b", "block_a", "block_b", "corr", "abs_corr"]
    if corr.empty:
        return pd.DataFrame(columns=columns)
    block_by_name = {s.name: s.block for s in FEATURE_SPECS}
    rows: list[dict] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "block_a": block_by_name.get(a, ""),
                    "block_b": block_by_name.get(b, ""),
                    "corr": float(r),
                    "abs_corr": float(abs(r)),
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def feature_ic_summary(features: pd.DataFrame,
                       labels: pd.DataFrame,
                       target_col: str = "l2_excess_vs_universe_median",
                       min_rows_per_date: int = 20) -> pd.DataFrame:
    """
    Per-feature Spearman information coefficient vs the training target,
    computed per rebalance_date then aggregated.

    ⚠ SANITY ONLY. Do NOT use this table to select features — doing so
    optimizes on the same target the model will later learn, which leaks
    evaluation into selection. Look at the distribution; don't pick off it.
    """
    merged = features.merge(
        labels[["ticker", "rebalance_date", target_col]],
        on=["ticker", "rebalance_date"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for c in FEATURE_COLUMNS:
        if c not in merged.columns:
            continue
        def _ic(g: pd.DataFrame) -> float:
            x = g[c]; y = g[target_col]
            m = x.notna() & y.notna()
            if m.sum() < min_rows_per_date:
                return np.nan
            return x[m].corr(y[m], method="spearman")
        ics = merged.groupby("rebalance_date", sort=False).apply(_ic, include_groups=False)
        ics = pd.to_numeric(ics, errors="coerce").dropna()
        if len(ics) == 0:
            rows.append({"feature": c, "mean_ic": None, "ic_std": None,
                         "ic_t": None, "n_dates": 0, "hit_rate": None})
            continue
        mu = float(ics.mean())
        sd = float(ics.std())
        t = float(mu / (sd / np.sqrt(len(ics)))) if sd > 0 else 0.0
        hit = float((np.sign(ics) == np.sign(mu)).mean()) if mu != 0 else 0.5
        rows.append({
            "feature": c,
            "mean_ic": mu,
            "ic_std": sd,
            "ic_t": t,
            "n_dates": int(len(ics)),
            "hit_rate": hit,
        })
    return pd.DataFrame(rows)
