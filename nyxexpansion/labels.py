"""
MFE/MAE + continuation labels.

Her (ticker, signal_date) satırı için t=0 = signal bar. Forward path
[t+1 .. t+h] üzerinde MFE/MAE hesaplanır. Bu forward bilgi label'dır,
feature olarak KULLANILMAZ.

Labels:
  L1 mfe_mae_ratio (raw + winsorized@clip)  — regression target
  L2 follow_through_3                         — binary, eval-only (noisy)
  L3 cont_{h}                                 — binary, PRIMARY train target
       (MFE_h ≥ cont_mfe_atr_mult × ATR0) AND (MAE_h ≤ cont_mae_atr_mult × ATR0)
  L4 expansion_score                          — regression, eval-only
       (MFE_h × close / ATR0) − λ × (MAE_h × close / ATR0)
  P2 cont_{h}_struct                          — research label
       risk_unit = close0 − max(swing_low_{struct_swing_lookback}, trigger_level − buffer×ATR0)
       pozitif: (MFE_h × close0 ≥ 2 × risk_unit) AND (MAE_h × close0 ≤ 1 × risk_unit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyxexpansion.config import LabelParams


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder ATR approx: SMA of True Range."""
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _forward_mfe_mae(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t=[0..N-1] için t+1..t+h penceresi üstünde MFE, MAE, close_h (hepsi fraction).

    MFE = max(high[t+1..t+h]) / close[t] - 1
    MAE = 1 - min(low[t+1..t+h]) / close[t]
    close_h = close[t+h]

    Pencere dolmuyorsa NaN.
    """
    n = len(close)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    cls_h = np.full(n, np.nan)
    for t in range(n - h):
        c0 = close[t]
        if c0 <= 0 or not np.isfinite(c0):
            continue
        hi = high[t + 1: t + 1 + h]
        lo = low[t + 1: t + 1 + h]
        if len(hi) < h:
            continue
        mfe[t] = hi.max() / c0 - 1.0
        mae[t] = 1.0 - lo.min() / c0
        cls_h[t] = close[t + h]
    return mfe, mae, cls_h


def _forward_min_low_window(low: np.ndarray, close: np.ndarray,
                            h_min: int, h_max: int) -> np.ndarray:
    """Erken drawdown için [t+h_min .. t+h_max] min(low). NaN if pencere eksik."""
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(n - h_max):
        seg = low[t + h_min: t + h_max + 1]
        if len(seg) < (h_max - h_min + 1):
            continue
        out[t] = seg.min()
    return out


def _swing_low(low: pd.Series, lookback: int) -> pd.Series:
    """Son `lookback` barın min low'u (bugün dahil değil)."""
    return low.rolling(lookback, min_periods=lookback).min().shift(1)


def compute_labels_for_ticker(
    df: pd.DataFrame,
    trigger_level: pd.Series | None = None,
    params: LabelParams | None = None,
) -> pd.DataFrame:
    """Tek ticker için tüm barlara labellar hesapla (her bar = potansiyel signal_date).

    Args:
        df: DatetimeIndex, Open/High/Low/Close/Volume.
        trigger_level: P2 için — o bar'ın trigger seviyesi (prior_high_20). None ise
                       P2 atlanır (NaN döner). Aynı index.
        params: LabelParams or None.

    Returns:
        DataFrame (aynı index) kolonlar:
          atr_{window}, close_0
          mfe_{h}, mae_{h} for each h in horizons
          mfe_mae_ratio_raw, mfe_mae_ratio_win  (h=primary_h üzerinden)
          follow_through_3  (L2, eval-only)
          cont_{primary_h}  (L3, PRIMARY)
          expansion_score_{primary_h}  (L4)
          cont_{primary_h}_struct  (P2, research; trigger_level None ise NaN)
    """
    if params is None:
        params = LabelParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)
    atr = _atr(df, params.atr_window)
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    out[f'atr_{params.atr_window}'] = atr
    out['close_0'] = close

    h_arr = np.asarray(high.values, dtype=float)
    l_arr = np.asarray(low.values, dtype=float)
    c_arr = np.asarray(close.values, dtype=float)

    for h in params.horizons:
        mfe, mae, _ = _forward_mfe_mae(h_arr, l_arr, c_arr, h)
        out[f'mfe_{h}'] = mfe
        out[f'mae_{h}'] = mae

    # L1 mfe_mae_ratio (primary_h bazlı)
    h = params.primary_h
    mfe_h = out[f'mfe_{h}']
    mae_h = out[f'mae_{h}']
    atr_frac = atr / close.replace(0.0, np.nan)   # atr/close ≈ atr_pct
    denom_floor = (params.mfe_mae_mae_floor_atr_frac * atr_frac).clip(lower=0.0)
    denom = np.maximum(mae_h.values, denom_floor.values)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_raw = np.where(denom > 0, mfe_h.values / denom, np.nan)
    out['mfe_mae_ratio_raw'] = ratio_raw
    out['mfe_mae_ratio_win'] = np.clip(ratio_raw, 0.0, params.mfe_mae_ratio_clip)

    # L2 follow_through_3 (eval-only)
    if 3 in params.horizons:
        # close[t+3] >= close[t]*(1+gain)  AND  min(low[t+1..t+3]) >= close[t]*(1-dd)
        c3_idx = close.shift(-3)
        cond_close = c3_idx >= close * (1.0 + params.ft3_min_close_gain)
        min_low_1_3 = _forward_min_low_window(l_arr, c_arr, 1, 3)
        cond_dd = min_low_1_3 >= (close.values * (1.0 - params.ft3_max_drawdown))
        ft3 = cond_close.values & cond_dd
        # Pencere dolmuyorsa NaN
        valid = ~np.isnan(min_low_1_3) & cond_close.notna().values
        ft3_out = np.where(valid, ft3.astype(float), np.nan)
        out['follow_through_3'] = ft3_out
    else:
        out['follow_through_3'] = np.nan

    # L3 cont_{h} — PRIMARY
    # MFE_h × close0 ≥ mult × ATR0 ≡ MFE_h ≥ mult × (ATR0/close0) = mult × atr_frac
    cont_up = mfe_h.values >= (params.cont_mfe_atr_mult * atr_frac.values)
    cont_dn = mae_h.values <= (params.cont_mae_atr_mult * atr_frac.values)
    cont_valid = ~(np.isnan(mfe_h.values) | np.isnan(mae_h.values) | np.isnan(atr_frac.values))
    cont = np.where(cont_valid, (cont_up & cont_dn).astype(float), np.nan)
    out[f'cont_{h}'] = cont

    # L4 expansion_score_{h}
    atr_frac_safe = np.where(atr_frac.values > 0, atr_frac.values, np.nan)
    with np.errstate(invalid='ignore'):
        exp_score = (mfe_h.values / atr_frac_safe) - params.expansion_lambda * (
            mae_h.values / atr_frac_safe
        )
    out[f'expansion_score_{h}'] = np.where(cont_valid, exp_score, np.nan)

    # P2 cont_{h}_struct — research
    if trigger_level is not None:
        sw_low = _swing_low(low, params.struct_swing_lookback)
        # invalidation_level = max(swing_low, trigger_level − buffer × ATR0)
        buf = params.struct_trigger_buffer_atr * atr
        trig_floor = trigger_level - buf
        invalidation = pd.concat([sw_low, trig_floor], axis=1).max(axis=1)
        risk_unit_abs = (close - invalidation).clip(lower=0.0)
        # Pozitif: MFE_h × close0 ≥ 2 × risk_unit  AND  MAE_h × close0 ≤ 1 × risk_unit
        mfe_abs = mfe_h.values * close.values
        mae_abs = mae_h.values * close.values
        cont_s_up = mfe_abs >= (params.cont_mfe_atr_mult * risk_unit_abs.values)
        cont_s_dn = mae_abs <= (params.cont_mae_atr_mult * risk_unit_abs.values)
        struct_valid = cont_valid & (risk_unit_abs.values > 0)
        cont_s = np.where(struct_valid, (cont_s_up & cont_s_dn).astype(float), np.nan)
        out[f'cont_{h}_struct'] = cont_s
        out['risk_unit_struct_pct'] = np.where(
            struct_valid, risk_unit_abs.values / close.values, np.nan,
        )
    else:
        out[f'cont_{h}_struct'] = np.nan
        out['risk_unit_struct_pct'] = np.nan

    return out


def compute_labels_on_panel(
    panel: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    params: LabelParams | None = None,
) -> pd.DataFrame:
    """`trigger.compute_trigger_a_panel` çıktısı + OHLCV → label'ları panel'e join'ler.

    Args:
        panel: ['ticker', 'date', 'close', 'prior_high_20', 'rvol', 'close_loc',
                'trigger_level'] long DataFrame.
        data_by_ticker: {ticker: OHLCV DataFrame}.

    Returns:
        panel + label kolonları eklenmiş DataFrame.
    """
    if params is None:
        params = LabelParams()

    if panel.empty:
        return panel.copy()

    pieces: list[pd.DataFrame] = []
    for ticker, sub in panel.groupby('ticker', sort=False):
        df = data_by_ticker.get(ticker)
        if df is None or df.empty:
            continue
        # trigger_level serisi: o ticker'ın TÜM barları için — labels fonksiyonu
        # bar-başına hesap yapar; burada df içi trigger_level'ı biz üretmiyoruz,
        # sadece panel'den o signal_date'teki trigger_level'ı geçeriz.
        # Ama compute_labels_for_ticker aynı index'te trigger_level bekler.
        # Panel'deki trigger_level sadece trigger bar'ları için var; geri kalan bar'lar
        # için P2'yi NaN yapacağız. Bunu basitleştirmek için her ticker'ın TÜM index'i
        # için trigger_level = prior_high(20) hesaplayıp besleyelim.
        from nyxexpansion.trigger import _rolling_prior_high
        from nyxexpansion.config import TriggerAParams as _TP
        _tp = _TP()
        trig_level_full = _rolling_prior_high(df['High'].astype(float), _tp.lookback_high)

        lbl = compute_labels_for_ticker(df, trigger_level=trig_level_full, params=params)
        lbl = lbl.reset_index().rename(columns={'Date': 'date', df.index.name or 'index': 'date'})
        # Merge signal_date rows only
        sub_dates = sub[['ticker', 'date']].copy()
        if 'date' not in lbl.columns:
            lbl = lbl.rename(columns={lbl.columns[0]: 'date'})
        lbl['ticker'] = ticker
        merged = sub.merge(lbl, on=['ticker', 'date'], how='left', suffixes=('', '_lbl'))
        pieces.append(merged)

    if not pieces:
        return panel.copy()

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(['date', 'ticker']).reset_index(drop=True)
    return out
