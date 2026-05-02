"""squeeze_breakout_loose_v0_diagnostic — Family B (DIAGNOSTIC, NOT PRODUCTION).

Permissive squeeze-breakout detector built to test whether a smart-style
loose cohort carries genuine alpha vs noise from dirty (high-width / steep
slope) bases. Validation outcome decides whether it graduates to a real
family. Until then:
    - column / family name carry "_v0_diagnostic" suffix
    - downstream artefacts MUST namespace separately
    - rule weights are all zero (no scoring)
    - acceptance criteria locked at agent design (memory file).

GATE (vs LEGACY V1.2.9 horizontal_base — frozen reference for the
strict_reject audit; current production strict is V1.3.1 with body 0.35):
    BB squeeze threshold     | legacy 0.65  | loose 0.80   ← wider consolidation OK
    body / atr_sq            | legacy 0.65  | loose 0.35   (V1.3.1 strict also 0.35)
    vol / vol_sma            | legacy 1.30  | loose 1.50   ← actually higher
    range_pos                | legacy 0.60  | loose 0.50
    breakout level           | robust q95+hard with buffer | hard max only with buffer
    close > sma20            | required     | required (kept)
    close > vwap20|60        | required     | dropped
    slope cap                | 0.15%/d both | NO CAP (audit only)
    width hard reject        | 25%          | NO CAP (audit only)
    initial_risk_pct         | unbounded    | <= 12%       ← new: bound left tail
    liquidity (turnover)     | unbounded    | >= 500K TL   ← new: tradability floor
    state machine            | pre/trig/retest/ext | TRIGGER ONLY (snapshot)

Audit columns emitted on every row:
    family__width_pct, family__slope_pct_per_day,
    family__strict_overlap_flag, family__strict_reject_reason_tags
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..schema import FEATURE_VERSION, SCANNER_VERSION, SCHEMA_VERSION
from .horizontal_base import (
    ATR_SL_MULT,
    BB_LENGTH,
    BB_MULT,
    BREAKOUT_BUFFER,
    MAX_SQUEEZE_AGE_BARS,
    MIN_SQUEEZE_BARS,
    RESISTANCE_ROBUST_QUANTILE,
    SLOPE_REJECT_PER_DAY as STRICT_SLOPE_REJECT_PER_DAY,
    SUPPORT_QUANTILE,
    TRADING_DAYS_PER_WEEK,
    TRIG_RANGE_POS_MIN as STRICT_TRIG_RANGE_POS_MIN,
    TRIG_VOL_RATIO_MIN as STRICT_TRIG_VOL_RATIO_MIN,
    WIDTH_PCT_REJECT as STRICT_WIDTH_PCT_REJECT,
    _atr,
    _box_from_squeeze,
    _line_fit,
)


# Legacy V1.2.9 strict trigger body floor — pinned locally so the diagnostic
# family's "would V1.2.9 strict reject this?" audit semantics are preserved
# even after horizontal_base lowered IMPULSE_ATR_MULT to 0.35 in V1.3.1.
# DO NOT change this — it is a frozen reference, not a gate this family applies.
_LEGACY_STRICT_IMPULSE_ATR_MULT = 0.65


FAMILY = "squeeze_breakout_loose_v0_diagnostic"

# ------------------------------------------------------------------ params Family B
LOOSE_BB_WIDTH_THRESH = 0.80     # vs strict 0.65 — wider consolidations qualify
LOOSE_BODY_ATR_MULT = 0.35       # vs strict 0.65
LOOSE_VOL_RATIO_MIN = 1.50       # vs strict 1.30 (intentionally higher: smart-style)
LOOSE_RANGE_POS_MIN = 0.50       # vs strict 0.60
LOOSE_RISK_PCT_MAX = 0.12        # left-tail bound: max 12% initial_risk_pct
LOOSE_LIQUIDITY_MIN_TL = 500_000  # turnover_TL floor (matches nyxexp live cohort)

# Squeeze relevance — same as strict.
# (No slope cap, no width hard reject — audit-only.)


# ------------------------------------------------------------------ helpers
def _compute_indicators_loose(df: pd.DataFrame) -> pd.DataFrame:
    """Same indicators as strict but with looser BB squeeze threshold."""
    df = df.copy()
    c = df["close"]
    sma = c.rolling(BB_LENGTH).mean()
    std = c.rolling(BB_LENGTH).std()
    df["bb_width"] = (sma + BB_MULT * std - (sma - BB_MULT * std)) / sma
    df["bb_width_sma"] = df["bb_width"].rolling(BB_LENGTH).mean()

    df["atr_sq"] = _atr(df, 10)
    df["atr_14"] = _atr(df, 14)

    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["sma20"] = c.rolling(20).mean()

    # Loose squeeze gate: bb_width < bb_width_sma * LOOSE_BB_WIDTH_THRESH
    df["sq_bb"] = df["bb_width"] < df["bb_width_sma"] * LOOSE_BB_WIDTH_THRESH
    df["squeeze"] = df["sq_bb"].fillna(False)
    return df


def _find_squeeze_runs(squeeze: pd.Series) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    n = len(squeeze)
    in_sq = False
    s = 0
    for i in range(n):
        v = bool(squeeze.iat[i])
        if v and not in_sq:
            s = i
            in_sq = True
        elif not v and in_sq:
            runs.append((s, i - 1, i - s))
            in_sq = False
    if in_sq:
        runs.append((s, n - 1, n - s))
    return [(s, e, L) for (s, e, L) in runs if L >= MIN_SQUEEZE_BARS]


def _is_loose_breakout_bar(
    df: pd.DataFrame,
    idx: int,
    hard_resistance: float,
) -> bool:
    """Loose breakout gate. Same close>level+buffer geometry as strict but
    relaxed body/range_pos and tighter vol; no VWAP filter."""
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0 or sma20 <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    return (
        c > hard_resistance * (1.0 + BREAKOUT_BUFFER)
        and h >= hard_resistance
        and c > o
        and range_pos >= LOOSE_RANGE_POS_MIN
        and body > atr * LOOSE_BODY_ATR_MULT
        and vol > vol_sma * LOOSE_VOL_RATIO_MIN
        and c > sma20
    )


def _strict_audit_eval(
    df: pd.DataFrame,
    idx: int,
    box_top_robust: float,
    hard_resistance: float,
    base_slope: float,
    res_slope: float,
    width_pct: float,
) -> tuple[bool, list[str]]:
    """Run the FULL strict V1.2.9 horizontal_base gate independently and return
    (would_pass, reject_reason_tags). Uses same per-bar inputs as strict
    _is_breakout_bar, plus the structural pre-filters (slope/width).
    """
    reasons: list[str] = []

    # Structural pre-filters (strict only — loose accepts these).
    if abs(base_slope) > STRICT_SLOPE_REJECT_PER_DAY:
        reasons.append("base_slope_steep")
    if abs(res_slope) > STRICT_SLOPE_REJECT_PER_DAY:
        reasons.append("res_slope_steep")
    if width_pct > STRICT_WIDTH_PCT_REJECT:
        reasons.append("width_too_wide")

    # Per-bar gates.
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5

    # robust > 0 implied by hard_resistance > 0
    if not (c > box_top_robust * (1.0 + BREAKOUT_BUFFER)):
        reasons.append("close_below_robust")
    if not (h >= hard_resistance):
        reasons.append("high_below_hard")
    if not (c > o):
        reasons.append("not_up_day")
    if not (range_pos >= STRICT_TRIG_RANGE_POS_MIN):
        reasons.append("range_pos_low")
    if atr <= 0 or not (body > atr * _LEGACY_STRICT_IMPULSE_ATR_MULT):
        reasons.append("body_low")
    if vol_sma <= 0 or not (vol > vol_sma * STRICT_TRIG_VOL_RATIO_MIN):
        reasons.append("vol_low")
    if sma20 <= 0 or not (c > sma20):
        reasons.append("sma20_below")
    # VWAP gate — strict requires close > vwap20 OR close > vwap60. Loose
    # doesn't compute these. Recompute on the fly.
    pv = (df["high"] + df["low"] + df["close"]) / 3.0 * df["volume"]
    vsum20 = df["volume"].rolling(20).sum()
    vsum60 = df["volume"].rolling(60).sum()
    vwap20_i = float((pv.rolling(20).sum() / vsum20.replace(0, np.nan)).iat[idx])
    vwap60_i = float((pv.rolling(60).sum() / vsum60.replace(0, np.nan)).iat[idx])
    vwap_ok = ((not np.isnan(vwap20_i)) and vwap20_i > 0 and c > vwap20_i) or \
              ((not np.isnan(vwap60_i)) and vwap60_i > 0 and c > vwap60_i)
    if not vwap_ok:
        reasons.append("vwap_below")

    return (len(reasons) == 0, reasons)


def _build_row_loose(
    *,
    ticker: str,
    df: pd.DataFrame,
    asof_idx: int,
    sq_start: int,
    sq_end: int,
    hard_resistance: float,
    box_top_robust: float,
    box_bot: float,
    base_slope: float,
    res_slope: float,
) -> dict:
    bo = df.iloc[asof_idx]
    asof_ts_raw = pd.Timestamp(df.index[asof_idx])
    asof_ts = (asof_ts_raw.tz_localize("Europe/Istanbul")
               if asof_ts_raw.tz is None else asof_ts_raw)
    bar_date = (asof_ts_raw.normalize() if asof_ts_raw.tz is None
                else asof_ts_raw.tz_convert("Europe/Istanbul").normalize().tz_localize(None))

    entry = float(bo["close"])
    atr_sq = float(bo["atr_sq"]) if pd.notna(bo["atr_sq"]) else 0.0
    invalidation = box_bot - ATR_SL_MULT * atr_sq
    initial_risk_pct = (entry - invalidation) / entry if entry > 0 else 0.0

    box_range = hard_resistance - box_bot
    width_pct = box_range / entry if entry > 0 else 0.0
    width_atr = box_range / atr_sq if atr_sq > 0 else float("nan")

    high_d = float(bo["high"])
    low_d = float(bo["low"])
    open_d = float(bo["open"])
    rng_d = high_d - low_d
    range_position = (entry - low_d) / rng_d if rng_d > 0 else 0.5
    body_pct = abs(entry - open_d) / open_d if open_d > 0 else 0.0
    vol_ratio = (float(bo["volume"] / bo["vol_sma"])
                 if (pd.notna(bo["vol_sma"]) and bo["vol_sma"] > 0) else float("nan"))
    breakout_pct = (entry - hard_resistance) / hard_resistance if hard_resistance > 0 else 0.0
    breakout_atr_v = (entry - hard_resistance) / atr_sq if atr_sq > 0 else float("nan")

    slope_pct_per_day = max(abs(base_slope), abs(res_slope))

    strict_pass, strict_reasons = _strict_audit_eval(
        df, asof_idx, box_top_robust, hard_resistance,
        base_slope, res_slope, width_pct,
    )

    return {
        # identity
        "ticker": ticker,
        "bar_date": bar_date,
        "setup_family": FAMILY,
        "signal_type": "long_breakout",
        "signal_state": "trigger",
        "breakout_bar_date": bar_date,
        # audit
        "as_of_ts": asof_ts,
        "data_frequency": "1d",
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "scanner_version": SCANNER_VERSION,
        # contract
        "family__trigger_level": np.float32(hard_resistance),
        "entry_reference_price": np.float32(entry),
        "invalidation_level": np.float32(invalidation),
        "initial_risk_pct": np.float32(initial_risk_pct),
        # common (subset — diagnostic family, not full schema population)
        "common__breakout_pct": np.float32(breakout_pct),
        "common__breakout_atr": np.float32(breakout_atr_v) if pd.notna(breakout_atr_v) else np.float32("nan"),
        "common__range_position": np.float32(range_position),
        "common__body_pct": np.float32(body_pct),
        "common__volume_ratio_20": np.float32(vol_ratio),
        "common__atr_14": np.float32(bo["atr_14"]) if pd.notna(bo["atr_14"]) else np.float32("nan"),
        "common__atr_pct": np.float32(bo["atr_14"] / entry) if (pd.notna(bo["atr_14"]) and entry > 0) else np.float32("nan"),
        "common__volume": np.float64(bo["volume"]),
        "common__turnover": np.float64(bo["volume"] * entry),
        "common__liquidity_score": np.float32(np.log1p(bo["volume"] * entry)),
        "common__close_vs_sma20": np.float32(entry / bo["sma20"] - 1.0) if pd.notna(bo["sma20"]) else np.float32("nan"),
        # family (geometry)
        "family__channel_high": np.float32(hard_resistance),
        "family__channel_low": np.float32(box_bot),
        "family__channel_width_pct": np.float32(width_pct),
        "family__channel_width_atr": np.float32(width_atr),
        "family__base_duration_weeks": np.float32((sq_end - sq_start + 1) / TRADING_DAYS_PER_WEEK),
        "family__base_slope": np.float32(base_slope),
        "family__resistance_slope": np.float32(res_slope),
        # family (audit — required by user before family graduates)
        "family__width_pct": np.float32(width_pct),
        "family__slope_pct_per_day": np.float32(slope_pct_per_day),
        "family__strict_overlap_flag": bool(strict_pass),
        "family__strict_reject_reason_tags": ",".join(strict_reasons) if strict_reasons else "",
    }


def detect(daily_df: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> list[dict]:
    """Loose-family snapshot detection. Trigger-only (no pre/extended/retest)."""
    ticker = daily_df.attrs.get("ticker", "?")
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators_loose(daily_df)

    if asof is None:
        asof_idx = len(df) - 1
    else:
        asof_ts = pd.Timestamp(asof).normalize()
        ts_idx = pd.DatetimeIndex(df.index).normalize()
        mask = ts_idx <= asof_ts
        if not mask.any():
            return []
        asof_idx = int(np.flatnonzero(mask)[-1])

    if asof_idx < BB_LENGTH * 2:
        return []

    runs = _find_squeeze_runs(df["squeeze"].iloc[: asof_idx + 1])
    if not runs:
        return []
    sq_s, sq_e, _ = runs[-1]
    if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
        return []
    if asof_idx <= sq_e:
        return []  # asof inside squeeze — no breakout possible yet

    # We need both robust q95 (for strict-overlap audit) and hard max (for loose
    # trigger). Replicate _box_from_squeeze inline — looser bb threshold means
    # squeeze runs differ from strict but the high-stat math is the same.
    sub = df.iloc[sq_s: sq_e + 1]
    hard_resistance = float(sub["high"].max())
    box_bot = float(sub["low"].quantile(SUPPORT_QUANTILE))
    box_top_robust = float(sub["high"].quantile(RESISTANCE_ROBUST_QUANTILE))
    if not (hard_resistance > box_bot > 0):
        return []

    # Loose trigger ONLY fires on the FIRST breakout bar in (sq_e, asof].
    breakout_idx: Optional[int] = None
    for i in range(sq_e + 1, asof_idx + 1):
        if _is_loose_breakout_bar(df, i, hard_resistance):
            breakout_idx = i
            break
    if breakout_idx is None or breakout_idx != asof_idx:
        return []  # diagnostic = trigger snapshot only; ignore extended bars

    # Slopes (audit only, no cap).
    base_slope, _ = _line_fit(sub["close"].values)
    res_slope, _ = _line_fit(sub["high"].values)

    # initial_risk_pct cap (gate)
    entry = float(df["close"].iat[asof_idx])
    atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
    invalidation = box_bot - ATR_SL_MULT * atr_sq
    risk_pct = (entry - invalidation) / entry if entry > 0 else float("inf")
    if not (0 < risk_pct <= LOOSE_RISK_PCT_MAX):
        return []

    # liquidity (gate)
    turnover = entry * float(df["volume"].iat[asof_idx])
    if turnover < LOOSE_LIQUIDITY_MIN_TL:
        return []

    row = _build_row_loose(
        ticker=ticker, df=df, asof_idx=asof_idx,
        sq_start=sq_s, sq_end=sq_e,
        hard_resistance=hard_resistance, box_top_robust=box_top_robust,
        box_bot=box_bot, base_slope=base_slope, res_slope=res_slope,
    )
    return [row]
