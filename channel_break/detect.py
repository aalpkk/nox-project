"""Channel detection: pivot fits, parallelism, alternating-touch gates.

Pure detection — no I/O, no resampling. Operates on a per-ticker panel
with a DatetimeIndex and OHLCV columns.

Public surface:
    detect(df, asof_idx, ticker, fam_name, params)
        → {"type": "channel"|"pending_triangle", "row": dict} | None
    fit_geometry(df, asof_idx, params)
        → geometry dict | None  (raw OLS fit + pivots + market context)
    scan_breakout_state(df, asof_idx, geom, params)
        → (signal_state, breakout_idx, breakout_age, breakout_close_over_upper_pct)

`fit_geometry` and `scan_breakout_state` are reused by the triangle_break
package; they expose the geometric primitives without committing to the
channel-vs-triangle routing decision.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from mb_scanner.pivots import alternating_pivots, find_pivots

from .schema import (
    CHANNEL_BREAK_VERSION,
    ChannelParams,
    empty_pending_row,
    empty_row,
)


# ---------------------------------------------------------------- helpers

def _bar_date(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tz is not None:
        return ts.tz_convert("Europe/Istanbul").normalize().tz_localize(None)
    return ts.normalize()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _vol_sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"].rolling(n).mean()


def _sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["close"].rolling(n).mean()


def _fit_line(idxs: list[int], prices: list[float]) -> tuple[float, float]:
    if len(idxs) < 2:
        raise ValueError("need ≥2 points")
    x = np.asarray(idxs, dtype=float)
    y = np.asarray(prices, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _line_at(slope: float, intercept: float, idx: int) -> float:
    return slope * idx + intercept


def _fit_containing_line(
    pivot_idxs: list[int],
    closes: np.ndarray,
    window_lo: int,
    window_hi: int,
    *,
    side: str,
    tol_pct: float = 0.005,
    max_violations: int = 0,
) -> Optional[tuple[float, float, list[int]]]:
    """Find a line through ≥2 pivots that bounds all closes in window.

    For side="upper": close[k] ≤ line(k) × (1+tol_pct) for every
    k ∈ [window_lo, window_hi], with at most max_violations exceptions.
    For side="lower": close[k] ≥ line(k) × (1-tol_pct), same tolerance.

    Among valid lines, picks the one with most pivot touches
    (|close[p] − line(p)| / close[p] ≤ tol_pct). Tie-break: most recent
    last touch.

    Returns (slope, intercept, sorted_touch_idxs) or None.
    """
    if len(pivot_idxs) < 2 or side not in ("upper", "lower"):
        return None

    is_upper = side == "upper"
    best: tuple[tuple[int, int], float, float, list[int]] | None = None

    for ii in range(len(pivot_idxs)):
        i = pivot_idxs[ii]
        ci = float(closes[i])
        for jj in range(ii + 1, len(pivot_idxs)):
            j = pivot_idxs[jj]
            cj = float(closes[j])
            if j == i:
                continue
            s = (cj - ci) / (j - i)
            b = ci - s * i

            n_violations = 0
            valid = True
            for k in range(window_lo, window_hi + 1):
                line_k = s * k + b
                if line_k <= 0:
                    continue
                ck = float(closes[k])
                if is_upper:
                    if ck > line_k * (1.0 + tol_pct):
                        n_violations += 1
                        if n_violations > max_violations:
                            valid = False
                            break
                else:
                    if ck < line_k * (1.0 - tol_pct):
                        n_violations += 1
                        if n_violations > max_violations:
                            valid = False
                            break
            if not valid:
                continue

            touches: list[int] = []
            for p in pivot_idxs:
                line_p = s * p + b
                cp = float(closes[p])
                if line_p > 0 and abs(cp - line_p) / cp <= tol_pct:
                    touches.append(p)
            if len(touches) < 2:
                continue
            score = (len(touches), max(touches))
            if best is None or score > best[0]:
                best = (score, s, b, touches)

    if best is None:
        return None
    _, s, b, touches = best
    return s, b, sorted(touches)


def _gates_at(
    df: pd.DataFrame,
    idx: int,
    atr: pd.Series,
    vol_sma: pd.Series,
    *,
    vol_ratio_min: float,
    range_pos_min: float,
    body_atr_min: float,
) -> tuple[bool, dict]:
    o = float(df["open"].iat[idx])
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    v = float(df["volume"].iat[idx])

    atr_now = float(atr.iat[idx]) if pd.notna(atr.iat[idx]) else float("nan")
    vsma = float(vol_sma.iat[idx]) if pd.notna(vol_sma.iat[idx]) else float("nan")

    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    body_atr = abs(c - o) / atr_now if atr_now and atr_now > 0 else float("nan")
    vol_ratio = v / vsma if vsma and vsma > 0 else float("nan")

    gates_pass = (
        math.isfinite(vol_ratio) and vol_ratio >= vol_ratio_min
        and math.isfinite(range_pos) and range_pos >= range_pos_min
        and math.isfinite(body_atr) and body_atr >= body_atr_min
    )

    return gates_pass, {
        "vol_ratio_20": vol_ratio,
        "range_pos": range_pos,
        "body_atr": body_atr,
        "atr_14": atr_now,
        "atr_pct": (atr_now / c) if (atr_now and c > 0) else float("nan"),
    }


def _slope_to_pct_per_bar(slope_price: float, mean_close: float) -> float:
    if mean_close <= 0:
        return 0.0
    return slope_price / mean_close * 100.0


def _classify_slope(mean_slope_pct: float, flat_threshold: float) -> str:
    if abs(mean_slope_pct) < flat_threshold:
        return "flat"
    return "asc" if mean_slope_pct > 0 else "desc"


def _triangle_kind(s_u_pct: float, s_l_pct: float, flat_thr: float = 0.05) -> str:
    """Classify converging/diverging fit when parallelism fails.

    flat_thr is a small absolute %/bar floor — a slope below this is
    treated as flat-ish for triangle classification purposes.
    """
    u_flat = abs(s_u_pct) < flat_thr
    l_flat = abs(s_l_pct) < flat_thr
    if u_flat and s_l_pct > flat_thr:
        return "ascending"
    if l_flat and s_u_pct < -flat_thr:
        return "descending"
    if s_u_pct < -flat_thr and s_l_pct > flat_thr:
        return "symmetric"
    if s_u_pct > flat_thr and s_l_pct < -flat_thr:
        return "expanding"
    if s_u_pct > s_l_pct + flat_thr:
        return "expanding"
    if s_l_pct > s_u_pct + flat_thr:
        return "symmetric"
    return "ambiguous"


# ---------------------------------------------------------------- geometry

def fit_geometry(
    df: pd.DataFrame,
    asof_idx: int,
    params: ChannelParams,
) -> Optional[dict]:
    """Compute pivot OLS fit + market context at asof.

    Shared by channel_break and triangle_break. Routing decisions
    (parallelism gate, width gate, subtype filter) live in callers.

    Returns None when the panel is too short or alternating-touch
    minimums fail. Returns a dict on success — callers MUST decide
    whether the geometry warrants a row.
    """
    n_min = max(params.lookback_bars, params.atr_n + params.pivot_n + 2)
    if asof_idx < n_min:
        return None

    lb_start = max(0, asof_idx - params.lookback_bars)

    sh, sl = find_pivots(df, params.pivot_n, asof_idx)
    sh = [i for i in sh if lb_start <= i <= asof_idx]
    sl = [i for i in sl if lb_start <= i <= asof_idx]

    if len(sh) < 2 or len(sl) < 2:
        return None

    pivots_all = alternating_pivots(df, params.pivot_n, asof_idx)
    pivots_win = [(k, i) for (k, i) in pivots_all if lb_start <= i <= asof_idx]
    if len(pivots_win) < 4:
        return None

    n_h_alt = sum(1 for (k, _) in pivots_win if k == "H")
    n_l_alt = sum(1 for (k, _) in pivots_win if k == "L")
    if n_h_alt < 2 or n_l_alt < 2:
        return None

    h_idxs_all = [i for (k, i) in pivots_win if k == "H"]
    l_idxs_all = [i for (k, i) in pivots_win if k == "L"]

    c = df["close"].to_numpy()

    # ATR-relative containment tolerance (v0.3). Single tol per setup,
    # anchored at asof so low-vol tickers get tight bands and high-vol get
    # wider. Floored / capped to keep extremes sane.
    _atr_series_pre = _atr(df, n=params.atr_n)
    atr_at_asof = float(_atr_series_pre.iat[asof_idx]) if pd.notna(_atr_series_pre.iat[asof_idx]) else float("nan")
    close_at_asof_for_tol = float(c[asof_idx])
    if not (atr_at_asof and atr_at_asof > 0 and close_at_asof_for_tol > 0):
        return None
    raw_tol = params.containment_tol_atr_k * atr_at_asof / close_at_asof_for_tol
    tol_used = max(params.containment_tol_min, min(params.containment_tol_max, raw_tol))

    # Try shrinking the fit window from asof down by up to extended_lookback_bars.
    # The smallest excluded suffix produces the tightest valid containment;
    # bars in (fit_window_hi..asof_idx] become breakout candidates handled
    # by scan_breakout_state. A downside violation in that suffix → reject
    # (long-only pattern; lower-line break is out-of-scope).
    upper_fit = lower_fit = None
    fit_window_hi = -1
    for k in range(0, params.extended_lookback_bars + 1):
        win_hi = asof_idx - k
        if win_hi <= lb_start:
            break
        h_pool = [i for i in h_idxs_all if i <= win_hi]
        l_pool = [i for i in l_idxs_all if i <= win_hi]
        if len(h_pool) < 2 or len(l_pool) < 2:
            continue
        u_fit = _fit_containing_line(
            h_pool, c, lb_start, win_hi,
            side="upper",
            tol_pct=tol_used,
            max_violations=params.max_line_violations,
        )
        l_fit = _fit_containing_line(
            l_pool, c, lb_start, win_hi,
            side="lower",
            tol_pct=tol_used,
            max_violations=params.max_line_violations,
        )
        if u_fit is None or l_fit is None:
            continue
        s_l_cand, b_l_cand, _ = l_fit
        downside = False
        for j in range(win_hi + 1, asof_idx + 1):
            line_l_j = s_l_cand * j + b_l_cand
            if line_l_j > 0 and float(c[j]) < line_l_j * (1.0 - tol_used):
                downside = True
                break
        if downside:
            return None  # downside break, not our long-only pattern
        upper_fit, lower_fit, fit_window_hi = u_fit, l_fit, win_hi
        break

    if upper_fit is None or lower_fit is None:
        return None

    s_u, b_u, h_idxs = upper_fit
    s_l, b_l, l_idxs = lower_fit

    asof_close = float(c[asof_idx])
    upper_at_asof = _line_at(s_u, b_u, asof_idx)
    lower_at_asof = _line_at(s_l, b_l, asof_idx)

    if lower_at_asof >= upper_at_asof:
        return None  # topology guard — lines crossed at asof

    width_pct = (upper_at_asof - lower_at_asof) / asof_close

    mean_close_win = float(np.mean(c[lb_start:asof_idx + 1]))
    s_u_pct = _slope_to_pct_per_bar(s_u, mean_close_win)
    s_l_pct = _slope_to_pct_per_bar(s_l, mean_close_win)
    mean_slope_pct = (s_u_pct + s_l_pct) / 2.0

    if max(abs(s_u_pct), abs(s_l_pct)) < 0.05:
        parallelism = 0.0
    else:
        mean_abs = (abs(s_u_pct) + abs(s_l_pct)) / 2.0
        parallelism = abs(s_u_pct - s_l_pct) / mean_abs if mean_abs > 0 else float("inf")

    res_u = [abs(c[i] - _line_at(s_u, b_u, i)) for i in h_idxs]
    res_l = [abs(c[i] - _line_at(s_l, b_l, i)) for i in l_idxs]
    max_res = max(max(res_u), max(res_l))
    width_now = upper_at_asof - lower_at_asof
    fit_max_residual_pct = (max_res / width_now) if width_now > 0 else float("inf")

    # fit_quality from touch counts (containment fit residuals are bounded
    # by tol_pct, so OLS-style residual classification is no longer useful)
    n_h, n_l = len(h_idxs), len(l_idxs)
    if n_h >= 3 and n_l >= 3:
        fit_quality = "tight"
    elif n_h >= 3 or n_l >= 3:
        fit_quality = "loose"
    else:
        fit_quality = "rough"

    first_pivot_idx = min(h_idxs[0], l_idxs[0])
    last_pivot_idx = max(h_idxs[-1], l_idxs[-1])

    atr = _atr(df, n=params.atr_n)
    vol_sma = _vol_sma(df, n=params.vol_sma_n)
    sma20 = _sma(df, n=params.sma_n)

    asof_open = float(df["open"].iat[asof_idx])
    asof_high = float(df["high"].iat[asof_idx])
    asof_low = float(df["low"].iat[asof_idx])
    asof_volume = float(df["volume"].iat[asof_idx])
    sma_now = float(sma20.iat[asof_idx]) if pd.notna(sma20.iat[asof_idx]) else float("nan")
    close_vs_sma20 = (asof_close / sma_now - 1.0) if sma_now > 0 else float("nan")

    gates_now_pass, gate_vals = _gates_at(
        df, asof_idx, atr, vol_sma,
        vol_ratio_min=params.vol_ratio_min,
        range_pos_min=params.range_pos_min,
        body_atr_min=params.body_atr_min,
    )

    asof_ts = pd.Timestamp(df.index[asof_idx])
    bar_date = _bar_date(asof_ts)

    return {
        "h_idxs": h_idxs,
        "l_idxs": l_idxs,
        "n_pivots_upper": len(h_idxs),
        "n_pivots_lower": len(l_idxs),
        "n_swing_touches": len(h_idxs) + len(l_idxs),
        "tier_a": (len(h_idxs) >= 3 and len(l_idxs) >= 3),
        "fit_window_hi": fit_window_hi,
        "s_u": s_u, "b_u": b_u, "s_l": s_l, "b_l": b_l,
        "s_u_pct": s_u_pct, "s_l_pct": s_l_pct,
        "mean_slope_pct": mean_slope_pct,
        "upper_at_asof": upper_at_asof,
        "lower_at_asof": lower_at_asof,
        "channel_width_pct": width_pct,
        "parallelism": parallelism,
        "fit_max_residual_pct": fit_max_residual_pct,
        "fit_quality": fit_quality,
        "first_pivot_idx": first_pivot_idx,
        "last_pivot_idx": last_pivot_idx,
        "channel_age_bars": int(asof_idx - first_pivot_idx),
        "asof_close": asof_close,
        "asof_open": asof_open,
        "asof_high": asof_high,
        "asof_low": asof_low,
        "asof_volume": asof_volume,
        "atr_14": gate_vals["atr_14"],
        "atr_pct": gate_vals["atr_pct"],
        "vol_ratio_20": gate_vals["vol_ratio_20"],
        "range_pos": gate_vals["range_pos"],
        "body_atr": gate_vals["body_atr"],
        "close_vs_sma20": close_vs_sma20,
        "asof_ts": asof_ts,
        "bar_date": bar_date,
        "gates_now_pass": gates_now_pass,
        # Series for caller-side extended scan
        "_atr_series": atr,
        "_vol_sma_series": vol_sma,
    }


def scan_breakout_state(
    df: pd.DataFrame,
    asof_idx: int,
    geom: dict,
    params: ChannelParams,
) -> tuple[Optional[str], Optional[int], Optional[int], Optional[float]]:
    """State machine: trigger / extended / pre_breakout / None.

    Returns (signal_state, breakout_idx, breakout_age_bars, breakout_close_over_upper_pct).
    None signal_state means the geometry exists but no actionable state at asof.
    """
    s_u, b_u = geom["s_u"], geom["b_u"]
    upper_at_asof = geom["upper_at_asof"]
    asof_close = geom["asof_close"]
    asof_high = geom["asof_high"]
    atr = geom["_atr_series"]
    vol_sma = geom["_vol_sma_series"]

    break_now = (
        asof_close > upper_at_asof * (1.0 + params.breakout_close_pct)
        and asof_high >= upper_at_asof
    )

    signal_state: Optional[str] = None
    breakout_idx: Optional[int] = None
    breakout_age: Optional[int] = None

    if break_now and geom["gates_now_pass"]:
        signal_state = "trigger"
        breakout_idx = asof_idx
        breakout_age = 0
    else:
        for back in range(1, params.extended_lookback_bars + 1):
            j = asof_idx - back
            if j < 0:
                break
            upper_at_j = _line_at(s_u, b_u, j)
            cj = float(df["close"].iat[j])
            hj = float(df["high"].iat[j])
            broke_at_j = (
                cj > upper_at_j * (1.0 + params.breakout_close_pct)
                and hj >= upper_at_j
            )
            if not broke_at_j:
                continue
            gates_at_j_pass, _ = _gates_at(
                df, j, atr, vol_sma,
                vol_ratio_min=params.vol_ratio_min,
                range_pos_min=params.range_pos_min,
                body_atr_min=params.body_atr_min,
            )
            if gates_at_j_pass:
                signal_state = "extended"
                breakout_idx = j
                breakout_age = back
                break

        if signal_state is None:
            pre_low = upper_at_asof * (1.0 - params.pre_breakout_close_pct)
            pre_high = upper_at_asof * (1.0 + params.breakout_close_pct)
            if pre_low <= asof_close <= pre_high:
                signal_state = "pre_breakout"

    breakout_close_over_upper_pct: Optional[float] = None
    if breakout_idx is not None:
        upper_at_break = _line_at(s_u, b_u, breakout_idx)
        bc = float(df["close"].iat[breakout_idx])
        if upper_at_break > 0:
            breakout_close_over_upper_pct = bc / upper_at_break - 1.0

    return signal_state, breakout_idx, breakout_age, breakout_close_over_upper_pct


# ---------------------------------------------------------------- core

def detect(
    df: pd.DataFrame,
    asof_idx: int,
    ticker: str,
    fam_name: str,
    params: ChannelParams,
) -> Optional[dict]:
    """Detect channel at asof_idx. Returns {type, row} dict or None."""
    geom = fit_geometry(df, asof_idx, params)
    if geom is None:
        return None

    base_meta = {
        "ticker": ticker,
        "setup_family": fam_name,
        "data_frequency": params.frequency,
        "as_of_ts": geom["asof_ts"],
        "bar_date": geom["bar_date"],
        "lookback_bars": params.lookback_bars,
        "n_pivots_upper": geom["n_pivots_upper"],
        "n_pivots_lower": geom["n_pivots_lower"],
        "n_swing_touches": geom["n_swing_touches"],
        "tier_a": geom["tier_a"],
        "upper_slope_pct_per_bar": geom["s_u_pct"],
        "lower_slope_pct_per_bar": geom["s_l_pct"],
        "mean_slope_pct_per_bar": geom["mean_slope_pct"],
        "upper_at_asof": geom["upper_at_asof"],
        "lower_at_asof": geom["lower_at_asof"],
        "channel_width_pct": geom["channel_width_pct"],
        "parallelism": geom["parallelism"],
        "fit_max_residual_pct": geom["fit_max_residual_pct"],
        "fit_quality": geom["fit_quality"],
        "channel_age_bars": geom["channel_age_bars"],
        "first_pivot_idx": int(geom["first_pivot_idx"]),
        "first_pivot_bar_date": _bar_date(pd.Timestamp(df.index[geom["first_pivot_idx"]])),
        "last_pivot_idx": int(geom["last_pivot_idx"]),
        "last_pivot_bar_date": _bar_date(pd.Timestamp(df.index[geom["last_pivot_idx"]])),
        "asof_close": geom["asof_close"],
        "asof_open": geom["asof_open"],
        "asof_high": geom["asof_high"],
        "asof_low": geom["asof_low"],
        "asof_volume": geom["asof_volume"],
        "atr_14": geom["atr_14"],
        "atr_pct": geom["atr_pct"],
        "vol_ratio_20": geom["vol_ratio_20"],
        "range_pos": geom["range_pos"],
        "body_atr": geom["body_atr"],
        "close_vs_sma20": geom["close_vs_sma20"],
        "schema_version": CHANNEL_BREAK_VERSION,
    }

    # ---- pending_triangle branch ----
    if geom["parallelism"] > params.parallelism_max:
        row = empty_pending_row()
        row.update(base_meta)
        row["triangle_kind_hint"] = _triangle_kind(geom["s_u_pct"], geom["s_l_pct"])
        return {"type": "pending_triangle", "row": row}

    # ---- channel branch ----
    if (geom["channel_width_pct"] < params.width_min_pct
            or geom["channel_width_pct"] > params.width_max_pct):
        return None

    slope_class = _classify_slope(geom["mean_slope_pct"], params.flat_slope_pct_per_bar)

    signal_state, breakout_idx, breakout_age, brk_over_upper = scan_breakout_state(
        df, asof_idx, geom, params,
    )
    if signal_state is None:
        return None

    breakout_bar_date = None
    if breakout_idx is not None:
        breakout_bar_date = _bar_date(pd.Timestamp(df.index[breakout_idx]))

    row = empty_row()
    row.update(base_meta)
    row.update({
        "signal_state": signal_state,
        "slope_class": slope_class,
        "direction_tag": f"{slope_class}_up_break",
        "breakout_idx": int(breakout_idx) if breakout_idx is not None else None,
        "breakout_bar_date": breakout_bar_date,
        "breakout_age_bars": breakout_age,
        "breakout_close_over_upper_pct": brk_over_upper,
    })
    return {"type": "channel", "row": row}
