"""Local close-source HW / HWO computation for PRSR diagnostic tag.

CRITICAL: This module returns a diagnostic tag (osc_confirmed) and feature
columns. It MUST NOT be used in the primary candidate gate. The closed
oscmatrix/ archive is not imported from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    rs = _rma(up, length) / _rma(dn, length).replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _hw_columns(close: pd.Series) -> pd.DataFrame:
    rsi = _rsi(close, C.HW_RSI_LENGTH)
    hw = rsi.rolling(C.HW_SIG_LEN, min_periods=C.HW_SIG_LEN).mean()
    sig = hw.rolling(C.HW_SIG_LEN, min_periods=C.HW_SIG_LEN).mean()
    hw_prev = hw.shift(1)
    sig_prev = sig.shift(1)
    hwo_up = (hw_prev <= sig_prev) & (hw > sig)
    hwo_down = (hw_prev >= sig_prev) & (hw < sig)
    return pd.DataFrame(
        {
            "hw": hw,
            "hw_signal": sig,
            "hwo_up": hwo_up.fillna(False),
            "hwo_down": hwo_down.fillna(False),
        }
    )


def add_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker close-source HW + dot triggers.

    Diagnostic only. Calling code must not insert these into the primary
    candidate gate.
    """
    out = df.copy()
    g = out.groupby("ticker", sort=False, group_keys=False)
    parts = g["close"].apply(_hw_columns)
    parts = parts.reset_index(level=0, drop=True)
    out = pd.concat([out, parts], axis=1)
    g2 = out.groupby("ticker", sort=False, group_keys=False)
    out["hwo_up_recent"] = g2["hwo_up"].transform(
        lambda s: s.rolling(C.HWO_UP_LOOKBACK).max() > 0
    ).fillna(False)
    out["hw_lt_50"] = (out["hw"] < C.HW_LT_50).fillna(False)
    out["lower_zone"] = (out["hw"] < C.HW_LOWER_ZONE_THRESHOLD).fillna(False)
    out["hw_slope_pos"] = (
        g2["hw"].transform(lambda s: s - s.shift(C.HW_SLOPE_BARS)).gt(0).fillna(False)
    )
    return out


def osc_confirmed(df: pd.DataFrame) -> pd.Series:
    """(HWO Up recent ∨ HW < 50) ∧ (HW slope > 0 ∨ lower_zone).

    Diagnostic tag — never gate.
    """
    a = df["hwo_up_recent"] | df["hw_lt_50"]
    b = df["hw_slope_pos"] | df["lower_zone"]
    return (a & b).fillna(False)
