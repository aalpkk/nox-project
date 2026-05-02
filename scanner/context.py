"""Cross-sectional / market context for scanner rows.

Builds two artifacts once per scan run:

  market_df : DataFrame indexed by date with universe-wide context
              (index returns, breadth, vol regime, regime label).
  rs_df     : DataFrame indexed by (ticker, date) with relative-strength
              metrics computed against the index and cross-sectionally.

Engine looks up these tables after each per-ticker `detect()` to fill the
`common__rs_*`, `common__index_*`, `common__market_*` schema columns. Sector
RS is left null in V1 (no clean sector mapping in the bundle).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data import intraday_1h


SECTOR_MAP_PATH = Path("tools/sector_map.json")


def _sector_map() -> dict[str, str]:
    """ticker -> sector key (KAP source).

    Prefers KAP `sector_index` code (e.g. 'XBANK', 'XKMYA') over the free-text
    `sector` field — the index code is populated more consistently and gives
    a stabler grouping. Falls back to text sector, then 'Unknown'.
    """
    if not SECTOR_MAP_PATH.exists():
        return {}
    raw = json.loads(SECTOR_MAP_PATH.read_text())
    tickers = raw.get("tickers", {})
    out: dict[str, str] = {}
    for t, rec in tickers.items():
        if not isinstance(rec, dict):
            continue
        key = rec.get("sector_index") or rec.get("sector") or "Unknown"
        out[t] = str(key)
    return out


def _index_daily() -> pd.DataFrame:
    """Pull XU100 daily close from regime_labels_daily.csv (label source)."""
    reg = intraday_1h.load_regime()
    df = reg[["date", "close", "regime"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return df


def build_market_context(daily_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (market_df, rs_df).

    Parameters
    ----------
    daily_panel : long-format DataFrame with columns
                  ['ticker','date','open','high','low','close','volume'].

    Returns
    -------
    market_df  : indexed by date.
    rs_df      : indexed by (ticker, date).
    """
    if daily_panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel = daily_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # ---- index frame -------------------------------------------------------
    idx = _index_daily()
    idx = idx.rename(columns={"close": "index_close", "regime": "regime_label"})
    idx["index_ret_5d"] = idx["index_close"].pct_change(5)
    idx["index_ret_20d"] = idx["index_close"].pct_change(20)
    idx["index_ret_60d"] = idx["index_close"].pct_change(60)
    idx["index_sma20"] = idx["index_close"].rolling(20).mean()
    idx["index_sma200"] = idx["index_close"].rolling(200).mean()
    idx["market_trend_score"] = (
        np.sign(idx["index_close"] - idx["index_sma20"]).fillna(0) * 0.5
        + np.sign(idx["index_close"] - idx["index_sma200"]).fillna(0) * 0.5
    )
    idx["index_rv20"] = idx["index_close"].pct_change().rolling(20).std() * np.sqrt(252)
    idx["market_vol_regime"] = idx["index_rv20"].rolling(252, min_periods=20).rank(pct=True)

    # ---- breadth -----------------------------------------------------------
    wide_close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    wide_sma20 = wide_close.rolling(20).mean()
    breadth = (wide_close > wide_sma20).sum(axis=1) / wide_close.notna().sum(axis=1)
    breadth.name = "market_breadth_pct_above_sma20"

    market_df = idx.join(breadth, how="left")
    market_df.index = market_df.index.normalize()

    # ---- per-ticker RS -----------------------------------------------------
    ret20 = wide_close.pct_change(20, fill_method=None)
    ret60 = wide_close.pct_change(60, fill_method=None)
    ret120 = wide_close.pct_change(120, fill_method=None)
    ret252 = wide_close.pct_change(252, fill_method=None)

    # rs vs index — log-difference: ticker_ret - index_ret on aligned dates.
    # ffill across holiday gaps (e.g. ticker traded on a day the index didn't);
    # never bfill (would leak forward).
    idx_ret20 = idx["index_ret_20d"].reindex(ret20.index).ffill()
    idx_ret60 = idx["index_ret_60d"].reindex(ret60.index).ffill()
    rs_20 = ret20.sub(idx_ret20, axis=0)
    rs_60 = ret60.sub(idx_ret60, axis=0)

    # rs_pctile_120/252 = cross-sectional percentile of ret_120/ret_252 per date
    rs_pctile_120 = ret120.rank(axis=1, pct=True)
    rs_pctile_252 = ret252.rank(axis=1, pct=True)

    # sector RS: ticker_ret_20d - sector_mean_ret_20d on the same date.
    # sector_mean is per-sector cross-sectional mean (groupby on columns).
    sectors = _sector_map()
    sec_per_col = pd.Series(
        {t: sectors.get(t, "Unknown") for t in wide_close.columns},
        name="sector",
    )
    sector_mean_20 = ret20.T.groupby(sec_per_col).transform("mean").T
    sector_rs_20 = ret20.sub(sector_mean_20)

    rs_df = (
        pd.concat({
            "common__rs_20d": rs_20,
            "common__rs_60d": rs_60,
            "common__rs_pctile_120": rs_pctile_120,
            "common__rs_pctile_252": rs_pctile_252,
            "common__sector_rs_20d": sector_rs_20,
        }, axis=1)
        .stack(level=1, future_stack=True)
        .swaplevel(0, 1)
        .sort_index()
    )
    rs_df.index = rs_df.index.set_names(["ticker", "date"])
    rs_df.index = rs_df.index.set_levels(
        [rs_df.index.levels[0], rs_df.index.levels[1].normalize()],
    )
    return market_df, rs_df


def _asof_lookup(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the largest date in idx that is <= ts, or None."""
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return idx[pos]


def fill_cross_sectional(
    row: dict,
    market_df: pd.DataFrame,
    rs_df: pd.DataFrame,
) -> dict:
    """Patch row with cross-sectional features. Mutates and returns row.

    Uses backward as-of lookup on bar_date so holidays / missing index days
    fall back to the prior trading day's context.
    """
    ticker = row.get("ticker")
    bd = pd.Timestamp(row.get("bar_date")).normalize()

    if not market_df.empty:
        ts = _asof_lookup(market_df.index, bd)
        if ts is not None:
            m = market_df.loc[ts]
            row["common__index_ret_5d"] = _f32(m.get("index_ret_5d"))
            row["common__index_ret_20d"] = _f32(m.get("index_ret_20d"))
            row["common__market_trend_score"] = _f32(m.get("market_trend_score"))
            row["common__market_vol_regime"] = _f32(m.get("market_vol_regime"))
            row["common__market_breadth_pct_above_sma20"] = _f32(m.get("market_breadth_pct_above_sma20"))

    if not rs_df.empty and ticker in rs_df.index.get_level_values("ticker"):
        per_t = rs_df.xs(ticker, level="ticker")
        ts = _asof_lookup(per_t.index, bd)
        if ts is not None:
            r = per_t.loc[ts]
            row["common__rs_20d"] = _f32(r.get("common__rs_20d"))
            row["common__rs_60d"] = _f32(r.get("common__rs_60d"))
            row["common__rs_pctile_120"] = _f32(r.get("common__rs_pctile_120"))
            row["common__rs_pctile_252"] = _f32(r.get("common__rs_pctile_252"))
            row["common__sector_rs_20d"] = _f32(r.get("common__sector_rs_20d"))
    return row


def _f32(x) -> Optional[np.float32]:
    if x is None:
        return np.float32("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.float32("nan")
    if not np.isfinite(v):
        return np.float32("nan")
    return np.float32(v)
