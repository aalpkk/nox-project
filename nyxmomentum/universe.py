"""
Per-rebalance tradability gate.

Runs a small battery of filters at each rebalance date and returns a
long-format panel — one row per (ticker, rebalance_date) — with eligibility
plus diagnostic columns. Downstream modules (features, labels, portfolio)
consume only rows where eligible == True.

Filters (all configurable via UniverseConfig):

  insufficient_history   — fewer than min_history_days trading bars before date
  recent_ipo             — first trading date within exclude_recent_ipo_days
  low_price              — close < min_price_tl
  low_liquidity          — 20d avg TL turnover (close * volume) < min_tl_volume_20d
  sparse_volume          — too few non-NaN volume days in 252d lookback
  too_many_missing       — NaN close ratio in 252d lookback > max_missing_ratio
  too_many_limits        — |ret| > limit_move_threshold frequency over the last
                           limit_move_lookback days exceeds max_limit_move_freq

Volume semantics: yfinance BIST delivers Volume in shares, Close in TL. TL
turnover is close * volume. If your data source differs, adapt upstream.

Leakage: only data with index <= rebalance_date is consulted. All rolling
windows are trailing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import UniverseConfig


REQUIRED_COLS = ("Close", "Volume")

# Filter column order used in diagnostic output
DIAG_COLS = [
    "ticker", "rebalance_date", "eligible", "reason",
    "close", "tl_volume_20d", "history_days", "ipo_age_days",
    "missing_ratio_252", "turnover_days_252",
    "limit_move_freq_60d", "zero_return_day_freq_252",
]


def _ensure_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in REQUIRED_COLS)


def check_membership(ticker: str,
                     df: pd.DataFrame,
                     rebalance_date: pd.Timestamp,
                     config: UniverseConfig) -> dict:
    """
    Apply filters to one (ticker, rebalance_date). Returns a diagnostic dict.

    The returned dict has every key in DIAG_COLS. On hard failures (empty
    frame, missing columns, future-start-date) eligible=False and reason
    captures the specific fault.
    """
    rebalance_date = pd.Timestamp(rebalance_date)

    row = {k: None for k in DIAG_COLS}
    row["ticker"] = ticker
    row["rebalance_date"] = rebalance_date

    if df is None or len(df) == 0:
        row["eligible"] = False
        row["reason"] = "no_data"
        return row
    if not _ensure_cols(df):
        row["eligible"] = False
        row["reason"] = "missing_columns"
        return row

    past = df.loc[df.index <= rebalance_date]
    if len(past) == 0 or past.index[0] > rebalance_date:
        row["eligible"] = False
        row["reason"] = "no_data_before_date"
        return row

    close = past["Close"]
    vol = past["Volume"]

    close_last = close.iloc[-1]
    first_date = past.index[0]
    ipo_age_days = int((rebalance_date - first_date).days)
    history_days = int(len(past))

    lookback_252 = past.tail(252)
    missing_ratio = float(lookback_252["Close"].isna().mean())
    turnover_days = int(lookback_252["Volume"].notna().sum())

    lookback_60 = past.tail(config.limit_move_lookback)
    rets_60 = lookback_60["Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    limit_move_freq = float((rets_60.abs() > config.limit_move_threshold).mean()) if len(rets_60) else 0.0

    lookback_20 = past.tail(20)
    tl_series = (lookback_20["Close"] * lookback_20["Volume"]).replace([np.inf, -np.inf], np.nan).dropna()
    tl_volume_20d = float(tl_series.mean()) if len(tl_series) else 0.0

    zero_ret_freq = 0.0
    rets_252 = lookback_252["Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets_252):
        zero_ret_freq = float((rets_252 == 0).mean())

    row.update({
        "close": float(close_last) if pd.notna(close_last) else None,
        "tl_volume_20d": tl_volume_20d,
        "history_days": history_days,
        "ipo_age_days": ipo_age_days,
        "missing_ratio_252": missing_ratio,
        "turnover_days_252": turnover_days,
        "limit_move_freq_60d": limit_move_freq,
        "zero_return_day_freq_252": zero_ret_freq,
    })

    reasons: list[str] = []
    if history_days < config.min_history_days:
        reasons.append("insufficient_history")
    if ipo_age_days < config.exclude_recent_ipo_days:
        reasons.append("recent_ipo")
    if row["close"] is None or row["close"] < config.min_price_tl:
        reasons.append("low_price")
    if tl_volume_20d < config.min_tl_volume_20d:
        reasons.append("low_liquidity")
    if turnover_days < config.min_turnover_days_available:
        reasons.append("sparse_volume")
    if missing_ratio > config.max_missing_ratio:
        reasons.append("too_many_missing")
    if limit_move_freq > config.max_limit_move_freq:
        reasons.append("too_many_limits")

    row["eligible"] = len(reasons) == 0
    row["reason"] = "ok" if not reasons else ",".join(reasons)
    return row


def build_universe_panel(panel: dict[str, pd.DataFrame],
                         rebalance_dates: pd.DatetimeIndex,
                         config: UniverseConfig | None = None) -> pd.DataFrame:
    """
    Long-format membership panel.

    Columns: ticker, rebalance_date, eligible, reason, close, tl_volume_20d,
             history_days, ipo_age_days, missing_ratio_252, turnover_days_252,
             limit_move_freq_60d, zero_return_day_freq_252.
    """
    cfg = config or UniverseConfig()
    rows: list[dict] = []
    for date in rebalance_dates:
        date = pd.Timestamp(date)
        for ticker, df in panel.items():
            if df is None or len(df) == 0:
                continue
            # Skip tickers that haven't started trading at all by this date
            if df.index[0] > date:
                continue
            rows.append(check_membership(ticker, df, date, cfg))

    out = pd.DataFrame(rows, columns=DIAG_COLS)
    if len(out):
        out["rebalance_date"] = pd.to_datetime(out["rebalance_date"])
    return out


def universe_summary(panel_long: pd.DataFrame, top_k_reasons: int = 5) -> pd.DataFrame:
    """
    Per-rebalance-date rollup: counts, eligibility rate, top rejection reasons.
    """
    if panel_long.empty:
        return pd.DataFrame(columns=[
            "rebalance_date", "total_considered", "eligible", "rejected",
            "eligibility_rate", "top_rejections"
        ])

    def _agg(g: pd.DataFrame) -> pd.Series:
        total = int(len(g))
        eligible = int(g["eligible"].sum())
        rejected = g.loc[~g["eligible"], "reason"]
        exploded = rejected.dropna().str.split(",").explode()
        exploded = exploded[exploded != "ok"]
        top = exploded.value_counts().head(top_k_reasons).to_dict()
        return pd.Series({
            "total_considered": total,
            "eligible": eligible,
            "rejected": total - eligible,
            "eligibility_rate": eligible / total if total else 0.0,
            "top_rejections": top,
        })

    summary = (
        panel_long.groupby("rebalance_date", group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return summary


def reason_histogram(panel_long: pd.DataFrame) -> pd.DataFrame:
    """Global rejection reason counts across all (ticker, date) rows."""
    if panel_long.empty:
        return pd.DataFrame(columns=["reason", "count"])
    rejected = panel_long.loc[~panel_long["eligible"], "reason"].dropna()
    exploded = rejected.str.split(",").explode()
    exploded = exploded[exploded != "ok"]
    counts = exploded.value_counts().reset_index()
    counts.columns = ["reason", "count"]
    return counts


def eligible_tickers_on(panel_long: pd.DataFrame,
                        rebalance_date: pd.Timestamp) -> list[str]:
    """Convenience selector used by downstream stages."""
    ts = pd.Timestamp(rebalance_date)
    mask = (panel_long["rebalance_date"] == ts) & panel_long["eligible"]
    return panel_long.loc[mask, "ticker"].tolist()
