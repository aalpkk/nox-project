"""
Three daily-bar signals harmonized to a common interface.

Each signal returns a Series[bool] indexed by date with True on trigger days.

  regime_transition_signal  — local scan_regime_transition AL transitions
  nox_weekly_signal         — local compute_nox_v3 on weekly, projected to daily
  alsat_signal              — vectorized port of bist_rejim_v3 + alsat AL gate
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from markets.bist.regime_transition import (
    scan_regime_transition,
    compute_trade_state,
)
from markets.bist.nox_v3_signals import compute_nox_v3, _pine_rsi


# ============================================================
# Helpers — shared indicators used by AS port
# ============================================================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _rma(_true_range(df), n)


def _adx_with_slope(df: pd.DataFrame, length: int = 14, slope_len: int = 3):
    """ADX series + slope series (per-bar)."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up.fillna(0)
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn.fillna(0)
    tr = _true_range(df)
    atr = _rma(tr, length)
    pdi = 100 * _rma(plus_dm, length) / atr.replace(0, np.nan)
    mdi = 100 * _rma(minus_dm, length) / atr.replace(0, np.nan)
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    adx = _rma(dx, length).fillna(0)
    slope = (adx - adx.shift(slope_len)) / slope_len
    return adx, slope, pdi, mdi


def _supertrend_dir(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend direction (+1 long, -1 short)."""
    atr = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n = len(df)
    up = upper.copy().to_numpy()
    dn = lower.copy().to_numpy()
    cl = df["close"].to_numpy()
    direction = np.ones(n, dtype=int)
    for i in range(1, n):
        if cl[i - 1] <= up[i - 1]:
            up[i] = min(up[i], up[i - 1])
        if cl[i - 1] >= dn[i - 1]:
            dn[i] = max(dn[i], dn[i - 1])
        if direction[i - 1] == 1 and cl[i] < dn[i - 1]:
            direction[i] = -1
        elif direction[i - 1] == -1 and cl[i] > up[i - 1]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)


def _wavetrend(df: pd.DataFrame, ch_len: int = 10, avg_len: int = 21, ma_len: int = 4):
    """WaveTrend (LazyBear)."""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = _ema(ap, ch_len)
    d = _ema((ap - esa).abs(), ch_len)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    tci = _ema(ci, avg_len)
    wt1 = tci
    wt2 = _sma(wt1, ma_len)
    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    return wt1, wt2, cross_up


def _pmax_long(df: pd.DataFrame, atr_len: int = 10, mult: float = 3.0, ma_len: int = 10) -> pd.Series:
    """Simplified PMAX long-state (true if trend up)."""
    atr = _atr(df, atr_len)
    src = _ema(df["close"], ma_len)
    upper = src + mult * atr
    lower = src - mult * atr
    n = len(df)
    cl = df["close"].to_numpy()
    up = upper.to_numpy()
    dn = lower.to_numpy()
    long = np.ones(n, dtype=bool)
    for i in range(1, n):
        if long[i - 1] and cl[i] < dn[i - 1]:
            long[i] = False
        elif (not long[i - 1]) and cl[i] > up[i - 1]:
            long[i] = True
        else:
            long[i] = long[i - 1]
    return pd.Series(long, index=df.index)


def _pivot_swing_break(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """SMC-lite: structural break above prior swing high.

    Approximation of `recent_bos & swing_bias==1` from regime_v3:
    True if close > rolling-max(high, lookback).shift(1).
    """
    swing_high = df["high"].rolling(lookback).max().shift(1)
    return (df["close"] > swing_high).fillna(False)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return macd - sig


def _rs_score(close: pd.Series, bench_close: pd.Series, n1: int = 21, n2: int = 63) -> pd.Series:
    """RS score: weighted relative price change vs benchmark.
    Series-wise version of regime_v3 rs_score (0.6*N1 + 0.4*N2 spread, in %).
    """
    aligned_b = bench_close.reindex(close.index).ffill()
    sp1 = (close / close.shift(n1) - 1.0) * 100
    sp2 = (close / close.shift(n2) - 1.0) * 100
    bp1 = (aligned_b / aligned_b.shift(n1) - 1.0) * 100
    bp2 = (aligned_b / aligned_b.shift(n2) - 1.0) * 100
    return (sp1 - bp1) * 0.6 + (sp2 - bp2) * 0.4


def _quality_score(df: pd.DataFrame) -> pd.Series:
    """regime_v3 Quality: rvol(25) + clv(25) + wick(25) + range_atr(25)."""
    h, l, c, o, v = df["high"], df["low"], df["close"], df["open"], df["volume"]
    vol_sma20 = _sma(v, 20)
    rvol = v / vol_sma20.replace(0, np.nan)
    candle_range = (h - l).replace(0, np.nan)
    clv = (c - l) / candle_range
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    wick_ratio = upper_wick / candle_range
    atr_val = _atr(df, 14)
    range_atr = candle_range / atr_val.replace(0, np.nan)

    rvol_s = np.where(rvol >= 2, 25, np.where(rvol >= 1.5, 20, np.where(rvol >= 1, 10, 0)))
    clv_s = np.where(clv >= 0.75, 25, np.where(clv >= 0.5, 15, np.where(clv >= 0.25, 5, 0)))
    wick_s = np.where(wick_ratio <= 0.15, 25, np.where(wick_ratio <= 0.3, 15, np.where(wick_ratio <= 0.5, 5, 0)))
    range_s = np.where(range_atr >= 1.2, 25, np.where(range_atr >= 0.8, 15, np.where(range_atr >= 0.5, 5, 0)))
    q = pd.Series(rvol_s + clv_s + wick_s + range_s, index=df.index)
    return q


# ============================================================
# Signal #1 — Regime Transition (local, series-based)
# ============================================================

def regime_transition_signal(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.Series:
    """Trigger Series[bool]: True on AL transition day (sticky-AL start)."""
    rt = scan_regime_transition(daily, weekly_df=weekly)
    state = compute_trade_state(rt["regime"], rt["close"], rt["ema21"])
    return state["al_signal"].astype(bool)


def regime_transition_rich(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Categorical breakdown for RT AL signals.

    Returns DataFrame indexed by daily.index with columns:
      al_signal  — bool (same as regime_transition_signal)
      rt_subtype — "DONUS" (CHOPPY/GRI → TREND/FULL) | "SICRAMA" (within-trend) | ""
      rt_tier    — "ALTIN" | "GUMUS" | "BRONZ" | "ELE" | "NORMAL" | ""  (DONUS only)
      rt_part    — participation_score (0-3)
      rt_atr_pct — ATR / close × 100
    """
    rt = scan_regime_transition(daily, weekly_df=weekly)
    regime = rt["regime"]
    state = compute_trade_state(regime, rt["close"], rt["ema21"])
    al = state["al_signal"].astype(bool)

    # DONUS = previous regime ∈ {CHOPPY=0, GRI_BOLGE=1} AND current ∈ {TREND=2, FULL_TREND=3}
    prev = regime.shift(1).fillna(0)
    is_donus = al & prev.isin([0, 1]) & regime.isin([2, 3])
    is_sicrama = al & ~is_donus

    subtype = pd.Series("", index=daily.index, dtype=object)
    subtype[is_donus] = "DONUS"
    subtype[is_sicrama] = "SICRAMA"

    # Tier classification — only meaningful for DONUS
    atr = _atr(daily, 14)
    atr_pct = (atr / daily["close"]) * 100.0
    rvol = daily["volume"] / _sma(daily["volume"], 20).replace(0, np.nan)

    # Participation score (rt_part) and CMF — already computed in features.py path,
    # mirror inline here to avoid circular import
    from markets.bist.regime_transition import compute_participation_score
    part = compute_participation_score(daily)
    part_score = part["participation_score"]
    cmf = part["cmf"]

    # OE score — series-wise simplified (RSI > 80 OR mom5 > 8 OR ema_dist > 5)
    rsi14 = _pine_rsi(daily["close"], 14)
    mom5 = (daily["close"] / daily["close"].shift(5) - 1) * 100
    ema21 = _ema(daily["close"], 21)
    ema_dist = (daily["close"] - ema21) / ema21 * 100
    oe = ((rsi14 > 80).astype(int) + (mom5 > 8).astype(int) + (ema_dist > 5).astype(int)).fillna(0)

    tier = pd.Series("", index=daily.index, dtype=object)
    # ELE: bad volume profile
    bad_vol = (atr_pct > 5) | (cmf < -0.1) | (rvol > 5)
    tier[is_donus & bad_vol] = "ELE"
    # ALTIN
    altin = (atr_pct <= 3) & (part_score == 3) & (oe <= 1)
    tier[is_donus & altin & ~bad_vol] = "ALTIN"
    # GUMUS
    gumus = (atr_pct <= 3) & (part_score == 3) & ~altin
    tier[is_donus & gumus & ~bad_vol] = "GUMUS"
    # BRONZ
    bronz = (atr_pct <= 4) & (part_score >= 3) & ~altin & ~gumus
    tier[is_donus & bronz & ~bad_vol] = "BRONZ"
    # NORMAL = remaining DONUS
    tier[is_donus & (tier == "")] = "NORMAL"

    # Entry score (GIR 0-4) — 4 bileşen, series-wise replica of run_regime_transition.py
    adx, adx_slope, _, _ = _adx_with_slope(daily, 14, 3)
    gir_vol_low = (atr_pct < 3.0).astype(int)        # 1. Vol düşük
    gir_early   = (adx_slope < 0).astype(int)        # 2. Erken giriş (ADX henüz dönmemiş)
    gir_room    = (regime <= 2).astype(int)          # 3. Büyüme odası
    gir_no_pump = (rvol < 2.0).astype(int)           # 4. Pump yok
    rt_entry_score = (gir_vol_low + gir_early + gir_room + gir_no_pump).fillna(0).astype(int)

    # OE score (OK 0-4) — 4 bileşen
    bb20_mid = _sma(daily["close"], 20)
    bb20_std = daily["close"].rolling(20).std()
    bb20_upper = bb20_mid + 2.0 * bb20_std
    ok_rsi      = (rsi14 > 80).astype(int)
    ok_bb       = (daily["close"] > bb20_upper).astype(int)
    ok_mom      = (mom5 > 8).astype(int)
    ok_ema_far  = (ema_dist > 5).astype(int)
    rt_oe_score = (ok_rsi + ok_bb + ok_mom + ok_ema_far).fillna(0).astype(int)

    return pd.DataFrame({
        "al_signal": al,
        "rt_subtype": subtype,
        "rt_tier": tier,
        "rt_part": part_score,
        "rt_atr_pct": atr_pct,
        "rt_trend_score":  rt["trend_score"].astype(int),
        "rt_part_score":   part_score.astype(int),
        "rt_exp_score":    rt["expansion_score"].astype(int) if "expansion_score" in rt else pd.Series(0, index=daily.index),
        "rt_regime":       regime.astype(int),
        "rt_entry_score":  rt_entry_score,        # GIR 0-4
        "rt_oe_score":     rt_oe_score,            # OK 0-4
    }, index=daily.index)


# ============================================================
# Signal #2 — NOX v3 Weekly pivot_buy projected to daily
# ============================================================

def nox_weekly_signal(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.Series:
    """Trigger Series[bool]: True on the daily bar that LANDS on a confirmed
    weekly pivot_buy bar.

    The `pivot_buy` is True at the bar that confirms the pivot low (i.e. lookback
    bars after the actual low). We map each weekly bar's pivot_buy boolean onto
    the LAST trading day of that week so the signal lights up T+0 of the daily
    cohort using only information available by close of that Friday.
    """
    nox = compute_nox_v3(weekly)
    pb = nox["pivot_buy"].astype(bool)
    if pb.empty:
        return pd.Series(False, index=daily.index)
    # Map weekly Friday index → corresponding daily date (last day in week ≤ Friday)
    out = pd.Series(False, index=daily.index)
    for w_date, flag in pb.items():
        if not flag:
            continue
        # Take all daily bars in week ending w_date
        in_week = daily.index[(daily.index <= w_date)]
        if len(in_week) == 0:
            continue
        # Mark the last daily bar in that week
        last_day = in_week.max()
        # Constrain: last_day must fall within (w_date - 6 days, w_date]
        if (w_date - last_day).days <= 6:
            out.loc[last_day] = True
    return out


def nox_rich(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Categorical NOX with daily + weekly pivot_buy + combined D+W.

    Columns:
      nox_w_trig   — weekly pivot_buy projected to last day of that week
      nox_d_trig   — daily pivot_buy on that bar
      nox_dw_type  — "DW" | "D" | "W" | ""  (DW = both fire on same day; supersedes individual)
    """
    w_trig = nox_weekly_signal(daily, weekly)

    # Daily nox (pivot_buy on daily bars). Don't require gate (default).
    nox_d = compute_nox_v3(daily)
    d_trig = nox_d["pivot_buy"].astype(bool).reindex(daily.index, fill_value=False)

    typ = pd.Series("", index=daily.index, dtype=object)
    both = w_trig & d_trig
    only_d = d_trig & ~w_trig
    only_w = w_trig & ~d_trig
    typ[both] = "DW"
    typ[only_d] = "D"
    typ[only_w] = "W"

    return pd.DataFrame({
        "nox_w_trig": w_trig,
        "nox_d_trig": d_trig,
        "nox_dw_type": typ,
    }, index=daily.index)


# ============================================================
# Signal #3 — AS (AL/SAT) — vectorized port of bist_rejim_v3 + classify
# ============================================================

# Constants (mirror bist_rejim_v3)
_ADX_LEN = 14
_ADX_TREND = 20.0
_ADX_CHOPPY = 15.0
_ADX_SLOPE_THRESH = 0.5
_EMA_FAST = 21
_EMA_SLOW = 55
_RVOL_THRESH = 1.5
_QUAL_TREND = 40.0
_QUAL_GRI = 60.0
_RS_THRESHOLD = 0.0
_MR_RSI_THRESH = 20.0


def _alsat_signal_components(daily: pd.DataFrame, weekly: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Produce per-bar component series needed by AS classify rule."""
    c = daily["close"]
    h = daily["high"]
    l = daily["low"]
    o = daily["open"]
    v = daily["volume"]

    ema_f = _ema(c, _EMA_FAST)
    ema_s = _ema(c, _EMA_SLOW)
    ema_trend_up = ema_f > ema_s

    st_dir = _supertrend_dir(daily, 10, 3.0)
    st_up = st_dir == 1

    # Weekly trend (project down to daily)
    if len(weekly) >= 20:
        w_ema_f = _ema(weekly["close"], _EMA_FAST)
        w_ema_s = _ema(weekly["close"], _EMA_SLOW)
        w_trend_up = (w_ema_f > w_ema_s).reindex(daily.index, method="ffill").fillna(False)
        w_adx, w_adx_slope, _, _ = _adx_with_slope(weekly, _ADX_LEN, 3)
        w_adx_d = w_adx.reindex(daily.index, method="ffill").fillna(0)
        w_rising = (w_adx_slope > _ADX_SLOPE_THRESH).reindex(daily.index, method="ffill").fillna(False)
    else:
        w_trend_up = pd.Series(False, index=daily.index)
        w_adx_d = pd.Series(0.0, index=daily.index)
        w_rising = pd.Series(False, index=daily.index)

    adx, adx_slope, _, _ = _adx_with_slope(daily, _ADX_LEN, 3)
    adx_rising = adx_slope > _ADX_SLOPE_THRESH

    trend_count = ema_trend_up.astype(int) + st_up.astype(int) + w_trend_up.astype(int)
    confirmed_trend_up = (trend_count >= 2) & (c > ema_s)

    # HTF regime tier
    htf_r = pd.Series(-1, index=daily.index)
    htf_r = htf_r.where(~((w_adx_d > _ADX_TREND) & w_rising), 2)
    htf_r = htf_r.where(~((w_adx_d > _ADX_TREND) & ~w_rising), 1)
    # Replace any cells that are 1 yet not actually in tier-1: redefine cleanly
    htf_tier = pd.Series(-1, index=daily.index)
    htf_tier[(w_adx_d > _ADX_TREND) & w_rising] = 2
    htf_tier[(w_adx_d > _ADX_TREND) & ~w_rising] = 1
    htf_tier[(w_adx_d <= _ADX_TREND) & (w_adx_d > _ADX_CHOPPY)] = 0
    htf_tier[w_adx_d <= _ADX_CHOPPY] = -1

    daily_confirm = (adx > _ADX_CHOPPY) & adx_rising

    regime = pd.Series(0, index=daily.index)
    regime[~confirmed_trend_up] = 0
    regime[confirmed_trend_up & (htf_tier == 2) & daily_confirm] = 3
    regime[confirmed_trend_up & (htf_tier >= 1) & ~((htf_tier == 2) & daily_confirm)] = 2
    regime[confirmed_trend_up & (htf_tier == 0)] = 1

    # WaveTrend
    wt1, wt2, wt_cross_up = _wavetrend(daily)
    wt_recent = wt_cross_up.rolling(3).max().fillna(False).astype(bool)
    wt_bullish = wt1 > wt2

    # PMAX
    pmax_long = _pmax_long(daily)

    # SMC-lite (BOS/CHoCH proxy via rolling swing-high break)
    bos_5 = _pivot_swing_break(daily, lookback=20)
    bos_tight = bos_5.rolling(5).max().fillna(False).astype(bool)
    choch_tight = bos_tight  # simplification — same proxy

    # Squeeze
    SQ_LEN = 20
    sq_basis = _sma(c, SQ_LEN)
    sq_dev = c.rolling(SQ_LEN).std() * 2.0
    sq_rng = _sma(_true_range(daily), SQ_LEN)
    bb_lower_inner = sq_basis - sq_dev
    bb_upper_inner = sq_basis + sq_dev
    kc_lower = sq_basis - 1.5 * sq_rng
    kc_upper = sq_basis + 1.5 * sq_rng
    sqz_on = (bb_lower_inner > kc_lower) & (bb_upper_inner < kc_upper)
    hh = h.rolling(SQ_LEN).max()
    ll = l.rolling(SQ_LEN).min()
    sq_mid = (hh + ll) / 2
    sq_mom_src = c - (sq_mid + sq_basis) / 2
    sq_mom = sq_mom_src.rolling(SQ_LEN).mean()
    sq_release = (~sqz_on) & sqz_on.shift(1).fillna(False) & (sq_mom > 0) & (sq_mom > sq_mom.shift(1))
    sq_release_recent = sq_release.rolling(3).max().fillna(False).astype(bool)

    atr = _atr(daily, 14)
    atr_ma = _sma(atr, 20)
    atr_expanding_now = atr > atr_ma * 1.05
    atr_expanding = atr_expanding_now.rolling(3).max().fillna(False).astype(bool)
    trend_sq = sq_release_recent | ((sq_mom > 0) & (sq_mom > sq_mom.shift(1)) & ~sqz_on)

    # BB / MR
    bb_basis = _sma(c, 20)
    bb_dev_val = c.rolling(20).std() * 2.0
    bb_lower = bb_basis - bb_dev_val
    donch_lower = l.rolling(20).min()
    rsi_short = _pine_rsi(c, 2)

    # Quality
    quality = _quality_score(daily)
    q_pass_trend = quality >= _QUAL_TREND
    q_pass_gri = quality >= _QUAL_GRI

    # Volume gate
    vol_sma20 = _sma(v, 20)
    rvol = v / vol_sma20.replace(0, np.nan)
    vol_high = v > vol_sma20 * _RVOL_THRESH
    avg_turnover = vol_sma20 * c
    vol_gate = avg_turnover >= 500_000

    # RS
    rs = _rs_score(c, bench)
    rs_pass = rs > _RS_THRESHOLD

    # PB sub-conditions
    pb_rsi = _pine_rsi(c, 5)
    pb_dipped = ((pb_rsi < 40) & (pb_rsi > 20)).rolling(5).max().fillna(False).astype(bool)
    pb_vol_dry = (v < vol_sma20 * 0.8).rolling(5).max().fillna(False).astype(bool)
    cross_up_ema_f = (c > ema_f) & (c.shift(1) <= ema_f.shift(1))
    pb_reclaim = cross_up_ema_f.rolling(3).max().fillna(False).astype(bool)
    pb_qual = q_pass_gri.where(regime == 1, q_pass_trend)
    pullback = confirmed_trend_up & st_up & rs_pass & pb_dipped & pb_vol_dry & pb_reclaim & pb_qual

    # SQ expansion
    sq_exp = (regime >= 1) & confirmed_trend_up & rs_pass & sq_release_recent & atr_expanding & (c > sq_basis) & q_pass_gri

    # MR
    mr_bb = (l <= bb_lower) & (c > bb_lower)
    mr_donch = (l <= donch_lower.shift(1)) & (c > donch_lower.shift(1))
    mean_rev = (regime <= 1) & (mr_bb | mr_donch) & (rsi_short < _MR_RSI_THRESH) & (rsi_short > rsi_short.shift(1)) & (c > ema_s * 0.90)

    # DONUS
    ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
    recent_e55 = ema55_cross.rolling(10).max().fillna(False).astype(bool)
    recent_wt_cross = wt_cross_up.rolling(10).max().fillna(False).astype(bool)
    ema55_dist_pct = (c - ema_s) / ema_s * 100
    approaching = (ema55_dist_pct.between(-3, 3)) & (c > c.shift(3)) & (c.shift(3) > c.shift(6))
    donus = (recent_e55 & recent_wt_cross & rs_pass & (c > ema_s)) | (recent_wt_cross & rs_pass & approaching & vol_high)

    # ERKEN
    sw_hl = h.rolling(20).max().shift(1)
    struct_break = c > sw_hl
    mom_up5 = (c / c.shift(5) - 1) * 100
    green_cnt = ((c > o).astype(int).rolling(5).sum())
    adx_turn = (adx > adx.shift(1)) & (adx.shift(1) > adx.shift(2))
    early_rsi = _pine_rsi(c, 14)
    highest5 = c.rolling(5).max()
    early_struct = struct_break & (v > vol_sma20 * 1.2) & adx_turn
    early_mom = (mom_up5 > 5.0) & (c >= highest5) & (green_cnt >= 3) & (v > vol_sma20 * 1.2)
    early = (regime <= 1) & (early_struct | early_mom) & (early_rsi < 75)

    # COMBO
    combo_base = (wt_cross_up | wt_recent) & wt_bullish & pmax_long
    combo_plus = combo_base & choch_tight
    combo_bos = combo_base & bos_tight & ~choch_tight

    # GUCLU / ZAYIF
    strong = (regime >= 2) & ema_trend_up & st_up & trend_sq & vol_high & rs_pass & q_pass_trend
    weak = (regime >= 2) & ema_trend_up & st_up & trend_sq & ~vol_high & rs_pass & q_pass_trend

    # BILESEN fallback
    has_wt = wt_recent | wt_cross_up
    active = has_wt.astype(int) + pmax_long.astype(int) + bos_tight.astype(int)
    bilesen = (active >= 2) & rs_pass & ~combo_plus & ~combo_bos & ~strong & ~weak & ~donus & ~early & ~pullback & ~sq_exp & ~mean_rev

    # MACD hist
    macd_h = _macd_hist(c)

    return pd.DataFrame({
        "regime": regime,
        "rs": rs,
        "rs_pass": rs_pass,
        "quality": quality,
        "macd_pos": macd_h > 0,
        "vol_gate": vol_gate,
        # Signal categories (priority order: combo_plus > combo_bos > strong > weak > donus > early > pullback > sq_exp > mean_rev > bilesen)
        "combo_plus": combo_plus,
        "combo_bos": combo_bos,
        "strong": strong,
        "weak": weak,
        "donus": donus,
        "early": early,
        "pullback": pullback,
        "sq_exp": sq_exp,
        "mean_rev": mean_rev,
        "bilesen": bilesen,
    })


def _alsat_classify_row(row) -> str:
    """Per-bar AL/IZLE/ATLA decision; returns 'AL', 'IZLE', 'ATLA', or '-' (no signal)."""
    if not row["vol_gate"]:
        return "-"
    rs = row["rs"]
    q = row["quality"]
    macd_pos = bool(row["macd_pos"])

    # Priority — only the highest-priority True wins
    if row["combo_plus"]:
        return "ATLA"  # CMB+ always ATLA
    if row["combo_bos"]:
        if q >= 80:
            return "AL"
        if q >= 70 and macd_pos:
            return "IZLE"
        return "ATLA"
    if row["strong"]:
        if rs < 0:
            return "ATLA"
        if 30 <= rs <= 60:
            return "AL"
        return "IZLE"
    if row["weak"]:
        if 20 <= rs <= 60 and macd_pos and q >= 50:
            return "AL"
        if macd_pos and q >= 40:
            return "IZLE"
        return "ATLA"
    if row["donus"]:
        return "AL"
    if row["early"]:
        return "ATLA"  # ERKEN always ATLA
    if row["pullback"]:
        if rs > 0 and q >= 50:
            return "AL"
        return "IZLE"
    if row["sq_exp"]:
        if rs > 0 and q >= 50:
            return "AL"
        return "IZLE"
    if row["mean_rev"]:
        if rs > 0:
            return "AL"
        return "IZLE"
    if row["bilesen"]:
        if q >= 70 and macd_pos:
            return "AL"
        if q >= 50 and macd_pos:
            return "IZLE"
        return "ATLA"
    return "-"


def alsat_signal(daily: pd.DataFrame, weekly: pd.DataFrame, bench: pd.Series) -> pd.Series:
    """Trigger Series[bool]: True on bars where AS classify returns AL."""
    comp = _alsat_signal_components(daily, weekly, bench)
    decision = comp.apply(_alsat_classify_row, axis=1)
    return decision == "AL"


# AS sub-category priority (highest first → first match wins)
_AS_SUBTYPE_ORDER = [
    ("CMB+",    "combo_plus"),
    ("CMB",     "combo_bos"),
    ("GUCLU",   "strong"),
    ("ZAYIF",   "weak"),
    ("DONUS",   "donus"),
    ("ERKEN",   "early"),
    ("PB",      "pullback"),
    ("SQ",      "sq_exp"),
    ("MR",      "mean_rev"),
    ("BILESEN", "bilesen"),
]


def alsat_rich(daily: pd.DataFrame, weekly: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """AS with sub-category label + classify decision.

    Columns:
      al_signal     — bool (decision == AL, same as alsat_signal)
      as_subtype    — "CMB+"|"CMB"|"GUCLU"|"ZAYIF"|"DONUS"|"ERKEN"|"PB"|"SQ"|"MR"|"BILESEN"|""
      as_decision   — "AL"|"IZLE"|"ATLA"|"-"
    Note: as_subtype is set on every bar where ANY sub-component fires (priority order),
    not just AL bars — this lets us study lift per sub-type independent of the AL gate.
    """
    comp = _alsat_signal_components(daily, weekly, bench)

    subtype = pd.Series("", index=daily.index, dtype=object)
    for label, col in _AS_SUBTYPE_ORDER:
        mask = comp[col].astype(bool) & (subtype == "")
        subtype[mask] = label

    decision = comp.apply(_alsat_classify_row, axis=1).astype(str)

    return pd.DataFrame({
        "al_signal": (decision == "AL"),
        "as_subtype": subtype,
        "as_decision": decision,
    }, index=daily.index)
