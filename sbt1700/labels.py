"""Label generation for SBT-1700: run E3 over each candidate and
attach realized-R / hit-flag columns.

The exit window for E3 starts at T+1 and uses up to ``timeout_bars``
forward daily bars. We never read T's EOD close.
"""

from __future__ import annotations

import pandas as pd

from sbt1700.execution import simulate_e3
from sbt1700.config import E3


_LABEL_COLS = [
    "realized_R_gross", "realized_R_net", "win_label",
    "tp_hit", "sl_hit", "timeout_hit",
    "exit_reason", "bars_held",
    "entry_px", "stop_px", "tp_px", "atr_1700",
    "initial_R_price", "exit_px", "exit_date", "cost_R",
]


def attach_labels(
    daily_master: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Run E3 per (ticker, date) and append label columns to feature_panel.

    Args:
        daily_master: long table indexed by Date with ``ticker`` column and
            OHLC. The forward window pulls High/Low/Close from T+1 onwards.
        feature_panel: rows with at least ticker, date, close_1700, atr14_prior.

    Returns feature_panel with _LABEL_COLS appended.
    """
    if feature_panel.empty:
        return feature_panel

    daily_master = daily_master.sort_index()
    by_ticker = {
        tk: g[["Open", "High", "Low", "Close"]].sort_index()
        for tk, g in daily_master.groupby("ticker")
    }

    out_rows: list[dict] = []
    for r in feature_panel.itertuples(index=False):
        sub = by_ticker.get(r.ticker)
        if sub is None or sub.empty:
            out_rows.append({c: None for c in _LABEL_COLS})
            continue

        T = pd.Timestamp(r.date).normalize()
        atr_1700 = float(r.atr14_prior)  # ATR from EOD-complete prior bars
        entry_px = float(r.close_1700)

        result = simulate_e3(
            entry_date=T,
            entry_px=entry_px,
            atr_1700=atr_1700,
            forward_ohlc=sub,
            params=E3,
        )
        out_rows.append({c: result.get(c) for c in _LABEL_COLS})

    label_df = pd.DataFrame(out_rows, index=feature_panel.index)
    return pd.concat([feature_panel.reset_index(drop=True),
                      label_df.reset_index(drop=True)], axis=1)


def label_columns() -> list[str]:
    return list(_LABEL_COLS)
