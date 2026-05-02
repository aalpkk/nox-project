"""HyperWave oscillator + Signal line + dot triggers.

Source = close (TRADING-EDGE variant; not the highest-fidelity TV match).

Phase 0.6.3 close-vs-ohlc4 robustness probe (2026-04-30) showed that
`ohlc4` improves TV HW numeric fidelity (RMSE 7.53→4.97, corr 0.967→0.976)
but destroys the AL_F6 trading edge and the D+C vs Stress separation:

  D+C K=10  CLOSE_ONLY ratio 2.89, false_rev 30%, p_lag 0
  D+C K=10  OHLC4_ONLY ratio 1.28, false_rev 39%
  STRESS K=10 CLOSE_ONLY ratio 0.70  /  OHLC4_ONLY ratio 1.33

The close-source "fidelity gap" behaves as a useful smoothing/lag filter:
HWO Up under close fires ~1 bar later, landing on the actual price low
(p_lag=0) for D+C reversals while remaining noisy on STRESS chop. Under
ohlc4 the cohort separation collapses (D+C edge ~= STRESS edge).

Therefore close is the production source. ohlc4 stays available as a
diagnostic / TV-fidelity variant only. Engine name:
  OSCMTRX-inspired close-source HWO early reversal engine
NOT a "LuxAlgo OSCMTRX bit-exact clone".

Dot triggers (per LuxAlgo small/big-dot semantics; truth-set audited via
TV `HWO Up ∪ Oversold HWO Up`, NOT `HWO Up` alone):
  hwo_up        : HW crosses above signal               (full cross, both dots)
  hwo_down      : HW crosses below signal               (full cross, both dots)
  os_hwo_up     : hwo_up AND HW < 20    (oversold turning point, big dot)
  ob_hwo_down   : hwo_down AND HW > 80  (overbought turning point, big dot)

`hwo_up` mirrors TV `HWO Up ∪ Oversold HWO Up` (small + big AL dots together).
`hwo_down` mirrors TV `HWO Down ∪ Overbought HWO Down`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OS_THRESHOLD = 20.0
OB_THRESHOLD = 80.0


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    rs = _rma(up, length) / _rma(dn, length).replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_hyperwave(
    df: pd.DataFrame,
    *,
    length: int = 7,
    sig_len: int = 3,
) -> pd.DataFrame:
    src = df["close"]
    rsi = _rsi(src, length)
    hyperwave = rsi.rolling(sig_len, min_periods=sig_len).mean()
    signal = hyperwave.rolling(sig_len, min_periods=sig_len).mean()

    hw_prev = hyperwave.shift(1)
    sig_prev = signal.shift(1)
    cross_up = (hw_prev <= sig_prev) & (hyperwave > signal)
    cross_dn = (hw_prev >= sig_prev) & (hyperwave < signal)

    hwo_up = cross_up
    hwo_down = cross_dn
    os_hwo_up = cross_up & (hyperwave < OS_THRESHOLD)
    ob_hwo_down = cross_dn & (hyperwave > OB_THRESHOLD)

    return pd.DataFrame(
        {
            "hyperwave": hyperwave,
            "signal": signal,
            "hwo_up": hwo_up,
            "hwo_down": hwo_down,
            "os_hwo_up": os_hwo_up,
            "ob_hwo_down": ob_hwo_down,
        },
        index=df.index,
    )
