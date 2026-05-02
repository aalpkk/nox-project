"""Same-date, same-eligible-universe, same-N random baseline.

For each fire date with N PRSR primary candidates, sample N tickers from
the Tradeable Core cohort on that date that did NOT fire any PRSR
candidate. Apply identical entry / stop / time-stop machinery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C
from prsr.exits import evaluate_trade


def build_random_baseline(
    panel: pd.DataFrame,
    primary_fires: pd.DataFrame,
    seed: int = C.BASELINE_SEED,
) -> pd.DataFrame:
    """Build per-trade rows for the random baseline.

    panel must contain (ticker, date, open, high, low, close, volume,
    pattern_low, atr20, tier).
    primary_fires must contain (ticker, date, pattern_low, atr20).
    """
    rng = np.random.default_rng(seed)

    fire_dates = primary_fires["date"].value_counts().sort_index()
    panel_sorted = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = {t: g.reset_index(drop=True) for t, g in panel_sorted.groupby("ticker", sort=False)}

    fire_set = set(zip(primary_fires["ticker"], primary_fires["date"]))

    rows = []
    for date, n_fires in fire_dates.items():
        same_day_core = panel_sorted[
            (panel_sorted["date"] == date) & (panel_sorted["tier"] == "core")
        ]
        eligible = same_day_core[
            ~same_day_core.apply(lambda r: (r["ticker"], r["date"]) in fire_set, axis=1)
        ]
        if len(eligible) == 0:
            continue
        # same-N sampling
        n_take = min(int(n_fires), len(eligible))
        idxs = rng.choice(len(eligible), size=n_take, replace=False)
        picks = eligible.iloc[idxs]

        # Synthesize a pseudo-pattern for stop computation:
        #   pattern_low = min(low[t-5..t]) — same construction PRSR pattern B uses,
        #   so the stop logic is comparable. atr20 from the panel.
        for _, row in picks.iterrows():
            ticker = row["ticker"]
            tg = by_ticker.get(ticker)
            if tg is None:
                continue
            mask = tg["date"] == date
            if not mask.any():
                continue
            fire_idx = int(np.flatnonzero(mask.values)[0])
            if fire_idx < 5:
                continue
            window_low = float(tg["low"].iloc[max(0, fire_idx - 5) : fire_idx + 1].min())
            atr_T = float(tg["atr20"].iloc[fire_idx])
            for entry_mode in ("open_T1", "close_T"):
                tr = evaluate_trade(tg, fire_idx, entry_mode, window_low, atr_T)
                if tr is None:
                    continue
                tr["ticker"] = ticker
                tr["fire_date"] = date
                tr["pattern_low"] = window_low
                tr["atr20"] = atr_T
                tr["pattern_kind"] = "random"
                tr["source"] = "random"
                rows.append(tr)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
