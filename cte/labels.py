"""
CTE labels — early-validation + runner/expansion aileleri.

Dual breakout_level semantics (kullanıcı onaylı):
  - _struct : boundary = max(hb_upper, fc_upper_last) at trigger bar
              → "yapı ihlali" semantiği (compression kutusu kırıldı mı?)
  - _close  : boundary = close[t] (entry price)
              → "trade PnL" semantiği (giriş zararda mı?)

Her horizon için iki versiyon üretilir:
  hold_h_struct, hold_h_close
  failed_break_h_struct, failed_break_h_close

Runner family (yalnız _close ref'li — trade PnL anlamlı):
  runner_h = 1 iff
      (MFE_h / ATR) >= runner_mfe_atr              # yeterli uzama
      AND (MAE_h / ATR) <= runner_max_mae_atr      # çok derin çekilmemiş
      AND hold_{runner_min_hold_h}_close == 1      # erken ölmemiş
      AND NOT spike_rejected_h                     # h anında peak'ten ciddi düşmemiş

  expansion_score_h = MFE_h / ATR  (continuous version)

Leakage
-------
Tüm label'lar t+1..t+h'den türetilir — gerçek forward-looking, ama bu
label'ın doğası gereği. Feature-side leakage YOK (label'a bakarak feature
çıkarmıyoruz).

ATR referansı `[t-14, t-1]` ile normalize — t barına ait aşırı geniş bar
denominator'ü şişirmesin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import LabelParams
from cte.structure import _atr_prior


def _forward_window_frame(s: pd.Series, max_h: int) -> np.ndarray:
    """n x max_h matrix: M[t, i-1] = s[t+i] for i in 1..max_h (NaN past end)."""
    n = len(s)
    arr = np.full((n, max_h), np.nan)
    vals = s.to_numpy()
    for i in range(1, max_h + 1):
        arr[: n - i, i - 1] = vals[i:]
    return arr


def compute_labels(
    df: pd.DataFrame,
    structure_frame: pd.DataFrame,
    trigger_frame: pd.DataFrame,
    params: LabelParams | None = None,
) -> pd.DataFrame:
    """Per-bar label frame; populated only at trigger bars.

    Parameters
    ----------
    df : OHLCV (Open, High, Low, Close, Volume) indexed by date.
    structure_frame : output of cte.structure.compute_structure. Used to derive
        breakout_level_struct = max(hb_upper, fc_upper_last).
    trigger_frame : output of cte.trigger.compute_trigger. Used as mask
        (labels only valid at trigger_cte=True rows).
    params : LabelParams.

    Returns
    -------
    DataFrame indexed like df with columns:
        breakout_level_struct, breakout_level_close,
        atr_ref,
        hold_{h}_struct, hold_{h}_close for h in early_horizons
        failed_break_{h}_struct, failed_break_{h}_close for h in early_horizons
        mfe_{h}_atr, mae_{h}_atr, spike_rejected_{h} for h in runner_horizons
        runner_{h} for h in runner_horizons
        expansion_score_{h} for h in runner_horizons
        primary_target  (= runner_{primary_target_h})
    """
    if params is None:
        params = LabelParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    atr = _atr_prior(df, params.atr_window).rename("atr_ref")

    # Boundary sources
    hb_upper = structure_frame.get("hb_upper")
    fc_upper_last = structure_frame.get("fc_upper_last")
    boundary_struct = pd.concat(
        [
            hb_upper if hb_upper is not None else pd.Series(np.nan, index=df.index),
            fc_upper_last if fc_upper_last is not None else pd.Series(np.nan, index=df.index),
        ],
        axis=1,
    ).max(axis=1)

    # Trigger mask — boundary only defined for trigger rows
    trig_mask = trigger_frame.get("trigger_cte", pd.Series(False, index=df.index)).fillna(False)
    boundary_struct = boundary_struct.where(trig_mask)

    # Forward window for lookups
    all_horizons = tuple(sorted(set(params.early_horizons) | set(params.runner_horizons)))
    max_h = max(all_horizons)

    fwd_close = _forward_window_frame(close, max_h)  # n x max_h
    fwd_high = _forward_window_frame(df["High"].astype(float), max_h)
    fwd_low = _forward_window_frame(df["Low"].astype(float), max_h)

    # Forward lookups at end of data naturally produce all-NaN slices; we want
    # NaN outputs (not runtime warnings).
    import warnings as _warnings

    out: dict[str, pd.Series] = {
        "breakout_level_struct": boundary_struct,
        "breakout_level_close": close.where(trig_mask),
        "atr_ref": atr,
    }

    # Early-validation: hold_h / failed_break_h for both breakout_level variants
    close_v = close.to_numpy()
    boundary_v = boundary_struct.to_numpy()
    atr_v = atr.to_numpy()
    n = len(close)

    for h in params.early_horizons:
        idx_col = h - 1
        close_h = fwd_close[:, idx_col]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            min_close = np.nanmin(fwd_close[:, :h], axis=1)

        # _close semantics
        hold_close = np.full(n, np.nan)
        fail_close = np.full(n, np.nan)
        valid_close = (~np.isnan(close_h)) & (~np.isnan(atr_v)) & trig_mask.to_numpy()
        hold_close[valid_close] = (close_h[valid_close] > close_v[valid_close]).astype(float)
        thresh_close = close_v - params.failback_atr_frac * atr_v
        fail_close[valid_close] = (
            min_close[valid_close] < thresh_close[valid_close]
        ).astype(float)

        # _struct semantics
        hold_struct = np.full(n, np.nan)
        fail_struct = np.full(n, np.nan)
        valid_struct = valid_close & (~np.isnan(boundary_v))
        hold_struct[valid_struct] = (close_h[valid_struct] > boundary_v[valid_struct]).astype(float)
        thresh_struct = boundary_v - params.failback_atr_frac * atr_v
        fail_struct[valid_struct] = (
            min_close[valid_struct] < thresh_struct[valid_struct]
        ).astype(float)

        out[f"hold_{h}_close"] = pd.Series(hold_close, index=df.index)
        out[f"hold_{h}_struct"] = pd.Series(hold_struct, index=df.index)
        out[f"failed_break_{h}_close"] = pd.Series(fail_close, index=df.index)
        out[f"failed_break_{h}_struct"] = pd.Series(fail_struct, index=df.index)

    # Runner family — _close refs (trade PnL semantics)
    min_hold_h = params.runner_min_hold_h
    min_hold_col = min_hold_h - 1
    close_min_hold = fwd_close[:, min_hold_col] if max_h >= min_hold_h else None

    for h in params.runner_horizons:
        idx_col = h - 1
        close_h = fwd_close[:, idx_col]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            peak_h = np.nanmax(fwd_high[:, :h], axis=1)
            trough_h = np.nanmin(fwd_low[:, :h], axis=1)

        mfe = (peak_h - close_v) / atr_v
        mae = (close_v - trough_h) / atr_v
        mfe_series = pd.Series(mfe, index=df.index).where(trig_mask)
        mae_series = pd.Series(mae, index=df.index).where(trig_mask)
        out[f"mfe_{h}_atr"] = mfe_series
        out[f"mae_{h}_atr"] = mae_series
        out[f"expansion_score_{h}"] = mfe_series

        # Spike-reject: close_h / peak_h ratio
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = close_h / peak_h
        spike = np.full(n, np.nan)
        valid = (~np.isnan(close_h)) & (~np.isnan(peak_h)) & trig_mask.to_numpy()
        spike[valid] = (ratio[valid] < params.spike_reject_close_peak_ratio).astype(float)
        out[f"spike_rejected_{h}"] = pd.Series(spike, index=df.index)

        # Runner: all conditions. Require complete information — if close[t+h]
        # is missing (end of data), spike is NaN → runner is NaN (unknown),
        # not 0.
        runner = np.full(n, np.nan)
        if close_min_hold is not None:
            hold_min = (close_min_hold > close_v).astype(float)
            runner_valid = (
                (~np.isnan(mfe))
                & (~np.isnan(mae))
                & (~np.isnan(close_min_hold))
                & (~np.isnan(close_h))
                & (~np.isnan(spike))
                & trig_mask.to_numpy()
            )
            runner[runner_valid] = (
                (mfe[runner_valid] >= params.runner_mfe_atr)
                & (mae[runner_valid] <= params.runner_max_mae_atr)
                & (hold_min[runner_valid] == 1)
                & (spike[runner_valid] == 0)
            ).astype(float)
        out[f"runner_{h}"] = pd.Series(runner, index=df.index)

    result = pd.DataFrame(out, index=df.index)

    # Primary target convenience column
    if params.primary_target in result.columns:
        result["primary_target"] = result[params.primary_target]

    return result
