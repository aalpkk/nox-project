from __future__ import annotations

import pandas as pd

from .components.confluence import compute_zones
from .components.hyperwave import compute_hyperwave
from .components.money_flow import compute_money_flow
from .params import DEFAULT_PARAMS, OSCMatrixParams


def compute_all(
    df: pd.DataFrame,
    params: OSCMatrixParams = DEFAULT_PARAMS,
    *,
    zone_rule: str = "above_50",
) -> pd.DataFrame:
    """Run all oscmatrix components on an OHLCV DataFrame.

    df: index=timestamp (or date), columns include open, high, low, close, volume.
    Returns a DataFrame indexed by df.index with all component series joined.
    """
    mf = compute_money_flow(df, length=params.mf_len, smooth=params.mf_smooth)
    hw = compute_hyperwave(df, length=params.hw_len, sig_len=params.hw_sig_len)
    zones = compute_zones(mf, hw, rule=zone_rule)
    return mf.join(hw).join(zones)
