"""Confluence Zones (Upper / Lower) — discrete {0, 1, 2}.

Per LuxAlgo docs:
  "When both the Hyper Wave and Money Flow oscillators are indicating an
   uptrend the upper zone is colored with a darker green. If only one of the
   oscillators is indicating an uptrend then the confluence zone will have a
   brighter green."

Mapped to numerical values:
  2 = strong confluence (both agree)
  1 = weak confluence   (only one agrees)
  0 = no confluence     (neither agrees)

"Indicating an uptrend" is ambiguous (cross-vs-signal? above midline?). We
expose three candidate rules so the validator can pick the one that matches
the TV "Upper/Lower Confluence Value" column bit-exact.

  rule = "above_signal" : HW > signal              ;  MF > MF.shift(smooth_proxy)
  rule = "above_50"     : HW > 50                  ;  MF > 50
  rule = "vs_threshold" : HW > 50                  ;  MF > upper_threshold (Lower: < lower_threshold)
"""
from __future__ import annotations

import pandas as pd


def compute_zones(
    mf: pd.DataFrame,
    hw: pd.DataFrame,
    *,
    rule: str = "above_50",
) -> pd.DataFrame:
    money_flow = mf["money_flow"]
    upper_thr = mf.get("upper_threshold")
    lower_thr = mf.get("lower_threshold")
    hyperwave = hw["hyperwave"]
    signal = hw["signal"]

    if rule == "above_signal":
        hw_bull = hyperwave > signal
        hw_bear = hyperwave < signal
        mf_bull = money_flow > money_flow.shift(1)
        mf_bear = money_flow < money_flow.shift(1)
    elif rule == "above_50":
        hw_bull = hyperwave > 50
        hw_bear = hyperwave < 50
        mf_bull = money_flow > 50
        mf_bear = money_flow < 50
    elif rule == "vs_threshold":
        hw_bull = hyperwave > 50
        hw_bear = hyperwave < 50
        if upper_thr is None or lower_thr is None:
            raise ValueError("vs_threshold rule requires mf with upper/lower thresholds")
        mf_bull = money_flow > upper_thr
        mf_bear = money_flow < lower_thr
    else:
        raise ValueError(f"unknown rule: {rule!r}")

    upper_zone = hw_bull.astype(int) + mf_bull.astype(int)
    lower_zone = hw_bear.astype(int) + mf_bear.astype(int)

    return pd.DataFrame(
        {
            "upper_zone": upper_zone,
            "lower_zone": lower_zone,
        },
        index=mf.index,
    )
