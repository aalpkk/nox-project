"""
Partial-TP ablation on the locked 17:30 WinMag baseline (111 signals).

Research-only — does NOT touch production config or selection/ranker.

Pipeline:
  1. Load 111 locked signals (reachable ∧ liq≥500K ∧ risk_bucket ≠ severe from yf399).
  2. For each signal, re-simulate the SWING exit on daily OHLCV with:
       - per-bar MFE / MAE tracking from entry_idx+1 through exit
       - configurable partial-TP spec (triggers + fractions + gate)
  3. Apply 17:30 adjustment (price_18:00 / price_17:30) to all returns/MFE/MAE.
  4. Emit:
       Part 1 — per-trade capture audit CSV
       Part 2 — per-trade × per-spec parquet + spec-summary CSV

Specs:
  P0  control (swing default: +1.5 ATR / sell 40% / move-BE)
  P1  +2.0 ATR / sell 25%
  P2  +3.0 ATR / sell 25%
  P3  +2.0 ATR / sell 15%
  P4  +2.0 ATR / sell 33%
  P5  liquidity zone (52w-high proxy = entry + upside_room_52w_atr × ATR) / sell 25%
  P6  contextual — partial only on bucket ∈ {elevated, mild}
  P7  contextual — partial only when above_52w or tight room (upside_room ≤ 2 ATR)
  P8  two-stage — 15% @ +1.5 ATR, then 15% @ +3.0 ATR

Constraints:
  - No change to selection, ranker, filter set. Same 111 signals always.
  - Stop/trail/time-stop logic identical across specs (only partial varies).
  - 15 bps slippage applied (matches locked baseline convention).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exits.config import SWING, CostModel, DEFAULT_COST_MODEL, round_trip_cost  # noqa: E402
from exits.structure import last_swing_low  # noqa: E402


LOCKED_SIG_PATH = Path("output/_locked111_signals.parquet")
OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
OUT_AUDIT = Path("output/nyxexp_exit_capture_audit.csv")
OUT_ABL = Path("output/nyxexp_partial_tp_ablation.csv")
OUT_TRADE = Path("output/nyxexp_partial_tp_tradelevel.parquet")

SLIPPAGE_BPS = 15
EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════
# Partial-TP spec
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Trigger:
    kind: str        # 'atr' | 'R' | 'liq_zone'
    value: float     # ATR multiple, R multiple, or (unused if liq_zone — target read from ctx)
    fraction: float  # 0..1


@dataclass(frozen=True)
class Spec:
    id: str
    label: str
    triggers: tuple[Trigger, ...]
    gate_label: str = "all"
    # gate: fn(ctx_dict) -> bool. None => always apply.
    gate: Callable[[dict], bool] | None = None
    move_be_after_first_partial: bool = True


def _gate_bucket_extended_elevated(ctx: dict) -> bool:
    return ctx.get("risk_bucket") in ("elevated", "mild")


def _gate_above52w_or_tight_room(ctx: dict) -> bool:
    room = ctx.get("upside_room_52w_atr")
    if room is None or pd.isna(room):
        return False
    # tight room = ≤ 2 ATR from 52w-high (or above_52w implied when room ≤ 0)
    return float(room) <= 2.0


SPECS: tuple[Spec, ...] = (
    Spec(id="P0", label="CONTROL (swing default: +1.5 ATR, 40%, BE)",
         triggers=(Trigger("atr", 1.5, 0.40),)),
    Spec(id="P1", label="+2.0 ATR / sell 25% / BE",
         triggers=(Trigger("atr", 2.0, 0.25),)),
    Spec(id="P2", label="+3.0 ATR / sell 25% / BE",
         triggers=(Trigger("atr", 3.0, 0.25),)),
    Spec(id="P3", label="+2.0 ATR / sell 15% / BE",
         triggers=(Trigger("atr", 2.0, 0.15),)),
    Spec(id="P4", label="+2.0 ATR / sell 33% / BE",
         triggers=(Trigger("atr", 2.0, 0.33),)),
    Spec(id="P5", label="Liquidity zone (52w-high proxy) / sell 25% / BE",
         triggers=(Trigger("liq_zone", 0.0, 0.25),)),
    Spec(id="P6", label="Contextual (elevated/mild only): +2.0 ATR / 25% / BE",
         triggers=(Trigger("atr", 2.0, 0.25),),
         gate_label="bucket∈{elevated,mild}",
         gate=_gate_bucket_extended_elevated),
    Spec(id="P7", label="Contextual (above52w/tight): +2.0 ATR / 25% / BE",
         triggers=(Trigger("atr", 2.0, 0.25),),
         gate_label="upside_room_52w≤2ATR",
         gate=_gate_above52w_or_tight_room),
    Spec(id="P8", label="Two-stage: 15% @ +1.5 ATR, 15% @ +3.0 ATR, BE after first",
         triggers=(Trigger("atr", 1.5, 0.15), Trigger("atr", 3.0, 0.15))),
)


# ══════════════════════════════════════════════════════════════════════
# Core math utilities (mirrors exits/simulator.py, extended)
# ══════════════════════════════════════════════════════════════════════

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr[:period - 1] = atr[period - 1]
    return atr


@dataclass
class TradeRecord:
    ticker: str
    signal_date: object
    fold: str
    spec_id: str
    spec_label: str
    gate_pass: bool
    entry_idx: int
    entry_price_close: float         # daily close at signal_date (engine entry)
    price_17_30: float
    price_18_00: float
    adj_mult: float                  # price_18_00 / price_17_30
    exit_idx: int
    exit_price: float
    bars_held: int
    reason: str
    partial_count: int
    partial_prices: tuple
    partial_fractions: tuple
    # Close-entry components
    close_gross_ret_pct: float       # blended gross (partials + remainder) from close entry
    close_final_only_ret_pct: float  # final segment alone (remainder)
    # 17:30-entry components
    ret_17_30_net_pct: float         # 15 bps slippage applied
    mfe_17_30_pct: float
    mae_17_30_pct: float
    capture_ratio: float             # ret_17_30_net / max(mfe_17_30, eps)
    giveback_pct: float              # mfe_17_30 - ret_17_30_net
    risk_bucket: str
    upside_room_52w_atr: float


def _simulate_with_mfe_mae(
    df_ohlcv: pd.DataFrame,
    entry_idx: int,
    atr_series: np.ndarray,
    spec: Spec,
    ctx: dict,
    cost_cfg: CostModel = DEFAULT_COST_MODEL,
) -> dict:
    """
    Simulate SWING exit from daily close entry with custom partial-TP spec.

    Returns dict with: exit_idx, exit_price_raw (pre-slippage), exit_price (post-slip),
    reason, bars_held, partial_price_list, partial_frac_list, mfe_close, mae_close,
    final_gross_ret_pct (close entry, pre-slippage, pre-costs), blended_gross_ret_pct.
    """
    cfg = SWING
    o = df_ohlcv["Open"].to_numpy(float)
    h = df_ohlcv["High"].to_numpy(float)
    l = df_ohlcv["Low"].to_numpy(float)
    c = df_ohlcv["Close"].to_numpy(float)
    n = len(c)

    entry_price = float(c[entry_idx])
    entry_atr = float(atr_series[entry_idx]) if atr_series[entry_idx] > 0 else max(1e-6, entry_price * 0.02)

    # Initial stop: -0.5 ATR
    initial_stop = entry_price - cfg.initial_stop_atr * entry_atr
    # Structure stop
    sw = last_swing_low(l, entry_idx, pivot_strength=cfg.pivot_strength, lookback=cfg.structure_lookback)
    structure_stop = sw.price if sw else None
    current_stop = initial_stop
    if structure_stop is not None:
        current_stop = max(current_stop, structure_stop)
    R_dist = max(1e-6, entry_price - initial_stop)

    gate_pass = (spec.gate is None) or spec.gate(ctx)

    # Partial targets (price levels) — compute up front
    partial_targets: list[tuple[float, float]] = []  # (price, fraction)
    if gate_pass:
        for trig in spec.triggers:
            if trig.kind == "atr":
                target = entry_price + trig.value * entry_atr
            elif trig.kind == "R":
                target = entry_price + trig.value * R_dist
            elif trig.kind == "liq_zone":
                room_atr = ctx.get("upside_room_52w_atr")
                if room_atr is None or pd.isna(room_atr) or float(room_atr) <= 0.1:
                    # no meaningful liquidity zone above entry → no partial
                    continue
                target = entry_price + float(room_atr) * entry_atr
            else:
                continue
            partial_targets.append((float(target), float(trig.fraction)))

    partials_done: list[tuple[int, float, float]] = []  # (idx, price, fraction)
    highest_since_entry = entry_price
    trail_activated = False
    be_moved = False

    mfe_high = entry_price  # tracked from entry_idx+1 onward (intraday high)
    mae_low = entry_price   # tracked likewise

    exit_idx = entry_idx
    exit_price_raw = entry_price
    reason = "open"

    for i in range(entry_idx + 1, min(entry_idx + 1 + cfg.time_stop_bars, n)):
        bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
        bars_held = i - entry_idx
        # MFE / MAE update (intraday)
        mfe_high = max(mfe_high, bar_h)
        mae_low = min(mae_low, bar_l)

        # 0. Gap stop
        if bar_o <= current_stop:
            exit_idx = i; exit_price_raw = bar_o; reason = "stop"
            break

        sl_hit = bar_l <= current_stop
        if sl_hit:
            exit_idx = i; exit_price_raw = current_stop; reason = "stop"
            break

        # Failed breakout
        fb_threshold = entry_price * (1 - cfg.failed_breakout_threshold_pct / 100.0)
        if cfg.use_failed_breakout and bars_held <= cfg.failed_breakout_bars and bar_c < fb_threshold:
            exit_idx = i; exit_price_raw = bar_c; reason = "failed_breakout"
            break

        # Partial triggers (check all not-yet-taken)
        remaining = [t for t in partial_targets
                     if not any(abs(p[1] - t[0]) < EPS and abs(p[2] - t[1]) < EPS for p in partials_done)]
        for tgt_price, frac in remaining:
            if bar_h >= tgt_price:
                partials_done.append((i, tgt_price, frac))
                if spec.move_be_after_first_partial and not be_moved:
                    current_stop = max(current_stop, entry_price)
                    be_moved = True

        # BE on partial (already handled above)

        # ATR trail
        if cfg.use_atr_trail:
            highest_since_entry = max(highest_since_entry, bar_h)
            activation_price = entry_price + cfg.trail_activate_atr * entry_atr
            if not trail_activated and highest_since_entry >= activation_price:
                trail_activated = True
            if trail_activated:
                trail_stop = highest_since_entry - cfg.trail_atr_mult * entry_atr
                current_stop = max(current_stop, trail_stop)

        # Structure trail
        if cfg.use_structure:
            new_sw = last_swing_low(l, i, pivot_strength=cfg.pivot_strength, lookback=cfg.structure_lookback)
            if new_sw is not None and new_sw.price > current_stop:
                current_stop = new_sw.price
    else:
        final_idx = min(entry_idx + cfg.time_stop_bars, n - 1)
        exit_idx = final_idx
        exit_price_raw = float(c[final_idx])
        reason = "time"

    # Slippage on exit leg (same reason mapping as production)
    if reason in ("stop", "structure", "trail"):
        slip = cost_cfg.exit_slippage_stop_pct / 100.0
    elif reason in ("time", "failed_breakout"):
        slip = cost_cfg.exit_slippage_time_pct / 100.0
    elif reason == "fixed_tp":
        slip = cost_cfg.exit_slippage_tp_pct / 100.0
    else:
        slip = 0.0
    exit_price = exit_price_raw * (1.0 - slip)

    final_gross_ret = (exit_price / entry_price - 1.0) * 100.0

    # Blend: partials get TP slippage; remainder gets the final reason's slippage
    part_slip = cost_cfg.exit_slippage_tp_pct / 100.0
    total_partial_frac = sum(p[2] for p in partials_done)
    part_component = 0.0
    part_price_list = []
    part_frac_list = []
    for _, p_price, p_frac in partials_done:
        p_exit = p_price * (1.0 - part_slip)
        p_ret = (p_exit / entry_price - 1.0) * 100.0
        part_component += p_frac * p_ret
        part_price_list.append(p_price)
        part_frac_list.append(p_frac)

    remainder_frac = max(0.0, 1.0 - total_partial_frac)
    blended_gross = part_component + remainder_frac * final_gross_ret

    mfe_close = (mfe_high / entry_price - 1.0) * 100.0
    mae_close = (mae_low / entry_price - 1.0) * 100.0

    return {
        "exit_idx": int(exit_idx),
        "exit_price": float(exit_price),
        "exit_price_raw": float(exit_price_raw),
        "reason": reason,
        "bars_held": int(exit_idx - entry_idx),
        "blended_gross_ret_pct": float(blended_gross),
        "final_only_ret_pct": float((exit_price / entry_price - 1.0) * 100.0),  # remainder before fraction
        "mfe_close_pct": float(mfe_close),
        "mae_close_pct": float(mae_close),
        "partial_prices": tuple(part_price_list),
        "partial_fractions": tuple(part_frac_list),
        "gate_pass": bool(gate_pass),
    }


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def _load_ohlcv_panel(tickers: Sequence[str]) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Load OHLCV per ticker, precompute ATR."""
    oh = pd.read_parquet(OHLCV_PATH)
    oh = oh[oh["ticker"].isin(list(set(tickers)))].copy()
    oh = oh.reset_index().rename(columns={oh.index.name or "Date": "Date"})
    if "Date" not in oh.columns:
        oh["Date"] = pd.to_datetime(oh.iloc[:, 0])
    oh["Date"] = pd.to_datetime(oh["Date"])
    panel: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for t, g in oh.groupby("ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        atr = _wilder_atr(g["High"].to_numpy(float), g["Low"].to_numpy(float), g["Close"].to_numpy(float))
        panel[t] = (g, atr)
    return panel


def _apply_17_30(close_ret_pct: float, mfe_close_pct: float, mae_close_pct: float,
                 adj_mult: float, slippage_bps: int) -> tuple[float, float, float]:
    """Convert close-entry returns/MFE/MAE to 17:30-entry basis, apply slippage."""
    ret_17_30 = ((1 + close_ret_pct / 100.0) * adj_mult - 1.0) * 100.0
    mfe_17_30 = ((1 + mfe_close_pct / 100.0) * adj_mult - 1.0) * 100.0
    mae_17_30 = ((1 + mae_close_pct / 100.0) * adj_mult - 1.0) * 100.0
    ret_net = ret_17_30 - slippage_bps / 100.0  # bps -> pct
    return ret_net, mfe_17_30, mae_17_30


def run() -> int:
    locked = pd.read_parquet(LOCKED_SIG_PATH)
    print(f"Loaded locked baseline: {len(locked)} signals")
    print(f"Bucket dist: {locked.risk_bucket.value_counts().to_dict()}")
    print()

    panel = _load_ohlcv_panel(locked["ticker"].unique())
    missing_ticker = [t for t in locked["ticker"].unique() if t not in panel]
    if missing_ticker:
        print(f"⚠ OHLCV missing for {len(missing_ticker)} tickers: {missing_ticker[:10]}")

    trades: list[dict] = []
    skipped = 0

    for spec in SPECS:
        for _, row in locked.iterrows():
            t = row.ticker
            d = pd.to_datetime(row.signal_date)
            if t not in panel:
                skipped += 1
                continue
            df_t, atr_t = panel[t]
            matches = df_t.index[df_t["Date"] == d]
            if len(matches) == 0:
                skipped += 1
                continue
            entry_idx = int(matches[0])
            if entry_idx >= len(df_t) - 1:
                skipped += 1
                continue

            ctx = {
                "risk_bucket": row.risk_bucket,
                "upside_room_52w_atr": row.upside_room_52w_atr,
            }

            sim = _simulate_with_mfe_mae(df_t, entry_idx, atr_t, spec, ctx)

            price_17_30 = float(row.price_17_30)
            price_18_00 = float(row.price_18_00)
            adj = price_18_00 / price_17_30 if price_17_30 > 0 else 1.0

            ret_net, mfe_17, mae_17 = _apply_17_30(
                sim["blended_gross_ret_pct"],
                sim["mfe_close_pct"],
                sim["mae_close_pct"],
                adj, SLIPPAGE_BPS,
            )
            cap = ret_net / max(mfe_17, EPS) if mfe_17 > EPS else np.nan
            giveback = mfe_17 - ret_net

            trades.append({
                "spec_id": spec.id,
                "spec_label": spec.label,
                "gate_label": spec.gate_label,
                "gate_pass": sim["gate_pass"],
                "ticker": t,
                "signal_date": row.signal_date,
                "fold": row.fold,
                "risk_bucket": row.risk_bucket,
                "upside_room_52w_atr": row.upside_room_52w_atr,
                "entry_idx": entry_idx,
                "entry_price_close": float(df_t.loc[entry_idx, "Close"]),
                "price_17_30": price_17_30,
                "price_18_00": price_18_00,
                "adj_mult": adj,
                "exit_idx": sim["exit_idx"],
                "exit_price": sim["exit_price"],
                "reason": sim["reason"],
                "bars_held": sim["bars_held"],
                "partial_count": len(sim["partial_prices"]),
                "partial_prices": sim["partial_prices"],
                "partial_fractions": sim["partial_fractions"],
                "close_gross_ret_pct": sim["blended_gross_ret_pct"],
                "close_final_only_ret_pct": sim["final_only_ret_pct"],
                "ret_17_30_net_pct": ret_net,
                "mfe_17_30_pct": mfe_17,
                "mae_17_30_pct": mae_17,
                "capture_ratio": cap,
                "giveback_pct": giveback,
            })

    df_trades = pd.DataFrame(trades)
    # Parquet can't serialize tuples of numpy floats uniformly → cast to lists/strings
    df_trades["partial_prices"] = df_trades["partial_prices"].apply(lambda x: list(x))
    df_trades["partial_fractions"] = df_trades["partial_fractions"].apply(lambda x: list(x))
    OUT_TRADE.parent.mkdir(parents=True, exist_ok=True)
    df_trades.to_parquet(OUT_TRADE, index=False)
    print(f"✅ Trade-level output: {OUT_TRADE}  ({len(df_trades)} rows)")

    # ── Part 1: capture audit (P0 only) ─────────────────────────────────
    p0 = df_trades[df_trades.spec_id == "P0"].copy()
    p0_audit = p0[[
        "ticker", "signal_date", "fold", "risk_bucket", "upside_room_52w_atr",
        "reason", "bars_held", "partial_count",
        "ret_17_30_net_pct", "mfe_17_30_pct", "mae_17_30_pct",
        "capture_ratio", "giveback_pct",
    ]].sort_values("ret_17_30_net_pct", ascending=False)
    p0_audit.to_csv(OUT_AUDIT, index=False)
    print(f"✅ Capture audit (Part 1): {OUT_AUDIT}  ({len(p0_audit)} rows)")

    # ── Part 2: ablation summary ───────────────────────────────────────
    def agg(g: pd.DataFrame) -> dict:
        r = g["ret_17_30_net_pct"] / 100.0
        mfe = g["mfe_17_30_pct"]
        gb = g["giveback_pct"]
        wins_mask = r > 0
        wins = int(wins_mask.sum())
        wr = wins / len(r) * 100 if len(r) else 0.0
        pf = (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.inf
        eq = (1 + r.fillna(0)).cumprod()
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100 if len(eq) else 0.0
        # capture ratio winners-only: realized/MFE, clipped to [0, 1]
        if wins > 0:
            win_realized = r[wins_mask] * 100
            win_mfe = mfe[wins_mask].clip(lower=EPS)
            cap_win = (win_realized / win_mfe).replace([np.inf, -np.inf], np.nan)
        else:
            cap_win = pd.Series([], dtype=float)
        return {
            "N": len(r),
            "PF": round(pf, 2) if np.isfinite(pf) else float("inf"),
            "avg%": round(r.mean() * 100, 2),
            "WR%": round(wr, 1),
            "MaxDD%": round(dd, 1),
            "total%": round((eq.iloc[-1] - 1) * 100, 1) if len(eq) else 0.0,
            "avg_winner%": round(r[wins_mask].mean() * 100, 2) if wins > 0 else 0.0,
            "avg_loser%": round(r[~wins_mask].mean() * 100, 2) if (~wins_mask).any() else 0.0,
            "avg_mfe%": round(mfe.mean(), 2),
            "avg_giveback%": round(gb.mean(), 2),
            "winner_capture_median": round(cap_win.median(), 3) if len(cap_win) else None,
            "winner_capture_mean": round(cap_win.mean(), 3) if len(cap_win) else None,
            "avg_bars_held": round(g["bars_held"].mean(), 2),
            "partial_share%": round((g["partial_count"] > 0).mean() * 100, 1),
            "reason_stop%": round((g["reason"] == "stop").mean() * 100, 1),
            "reason_time%": round((g["reason"] == "time").mean() * 100, 1),
            "reason_failbk%": round((g["reason"] == "failed_breakout").mean() * 100, 1),
        }

    rows = []
    for spec in SPECS:
        g = df_trades[df_trades.spec_id == spec.id]
        row = {"spec_id": spec.id, "spec_label": spec.label, "gate_label": spec.gate_label}
        row.update(agg(g))
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Deltas vs P0
    baseline_cols = ["PF", "avg%", "WR%", "MaxDD%", "total%", "avg_giveback%", "winner_capture_median"]
    p0_row = summary.loc[summary.spec_id == "P0"].iloc[0]
    for col in baseline_cols:
        base = p0_row[col]
        if base is None or (isinstance(base, float) and np.isnan(base)):
            summary["Δ_" + col] = None
            continue
        summary["Δ_" + col] = summary[col].apply(
            lambda v: round(float(v) - float(base), 3) if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v)) else None
        )

    OUT_ABL.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_ABL, index=False)
    print(f"✅ Ablation summary (Part 2): {OUT_ABL}")
    print()

    # ── Part 1 console: capture audit (P0 = locked behavior proxy) ────────
    print("═" * 110)
    print("PART 1 — CAPTURE AUDIT (P0 ≈ locked baseline, 111 trades, 17:30 entry @ 15 bps)")
    print("═" * 110)
    p0_metrics = summary.loc[summary.spec_id == "P0"].iloc[0]
    print(f"  N={p0_metrics['N']}  PF={p0_metrics['PF']}  avg={p0_metrics['avg%']:+.2f}%  "
          f"WR={p0_metrics['WR%']:.1f}%  DD={p0_metrics['MaxDD%']:+.1f}%  total={p0_metrics['total%']:+.1f}%")
    print(f"  avg MFE: {p0_metrics['avg_mfe%']:+.2f}%   avg realized: {p0_metrics['avg%']:+.2f}%   "
          f"avg giveback: {p0_metrics['avg_giveback%']:+.2f}pp")
    cap_med = p0_metrics["winner_capture_median"]
    cap_mean = p0_metrics["winner_capture_mean"]
    print(f"  Winners-only capture  median: {cap_med:.2f}   mean: {cap_mean:.2f}  "
          f"(1.00 = full capture, 0.50 = half MFE realized)")
    print()
    # Segment: by bucket
    print("  PER-BUCKET (P0):")
    for b, g in df_trades[df_trades.spec_id == "P0"].groupby("risk_bucket", observed=True):
        a = agg(g)
        cap = a["winner_capture_median"]
        cap_str = f"{cap:.2f}" if cap is not None else "n/a"
        print(f"    {b:<9} N={a['N']:>3}  PF={a['PF']:>5}  avg={a['avg%']:>+5.2f}%  "
              f"WR={a['WR%']:>5.1f}%  DD={a['MaxDD%']:>+6.1f}%  mfe={a['avg_mfe%']:>+5.2f}%  "
              f"giveback={a['avg_giveback%']:>5.2f}pp  winCap={cap_str}")
    print()
    print("  PER-FOLD (P0):")
    for f, g in df_trades[df_trades.spec_id == "P0"].groupby("fold", observed=True):
        a = agg(g)
        cap = a["winner_capture_median"]
        cap_str = f"{cap:.2f}" if cap is not None else "n/a"
        print(f"    {str(f):<9} N={a['N']:>3}  PF={a['PF']:>5}  avg={a['avg%']:>+5.2f}%  "
              f"WR={a['WR%']:>5.1f}%  DD={a['MaxDD%']:>+6.1f}%  mfe={a['avg_mfe%']:>+5.2f}%  "
              f"giveback={a['avg_giveback%']:>5.2f}pp  winCap={cap_str}")
    print()

    # ── Part 2 console: ablation summary ──────────────────────────────────
    print("═" * 130)
    print("PART 2 — ABLATION SUMMARY vs P0 (111 trades, 17:30 entry @ 15 bps)")
    print("═" * 130)
    cols = ["spec_id", "N", "PF", "avg%", "WR%", "MaxDD%", "total%",
            "avg_mfe%", "avg_giveback%", "winner_capture_median",
            "Δ_PF", "Δ_MaxDD%", "Δ_total%", "Δ_avg_giveback%"]
    print(summary[cols].to_string(index=False))
    print()

    # ── Part 4 decision helper ────────────────────────────────────────────
    print("═" * 110)
    print("DECISION RULE CHECK (vs P0)")
    print("═" * 110)
    print("Pass criteria: ΔPF ≥ 0, ΔMaxDD ≥ -3pp, Δtotal ≥ 0, winner tail not truncated (avg_winner% ≥ P0·0.90)")
    p0_winner_avg = float(p0_row["avg_winner%"])
    for _, row in summary.iterrows():
        if row.spec_id == "P0":
            continue
        pf_ok = (row["Δ_PF"] or 0) >= 0
        dd_ok = (row["Δ_MaxDD%"] or 0) >= -3.0
        tot_ok = (row["Δ_total%"] or 0) >= 0
        winner_ok = float(row["avg_winner%"]) >= 0.9 * p0_winner_avg
        verdict = "PASS" if all([pf_ok, dd_ok, tot_ok, winner_ok]) else "fail"
        flags = "".join([
            "✓" if pf_ok else "✗", "✓" if dd_ok else "✗",
            "✓" if tot_ok else "✗", "✓" if winner_ok else "✗",
        ])
        print(f"  {row.spec_id}: {verdict}  [PF/DD/total/winner: {flags}]  "
              f"ΔPF={row['Δ_PF']:+.2f} ΔDD={row['Δ_MaxDD%']:+.1f}pp "
              f"Δtotal={row['Δ_total%']:+.1f}pp avg_win={row['avg_winner%']:.2f}% (P0={p0_winner_avg:.2f}%)")

    # ── Per-fold stability for the decision-relevant specs ──────────────
    print()
    print("═" * 110)
    print("PER-FOLD STABILITY (does the winning spec hold up across folds?)")
    print("═" * 110)
    leaders = ["P0", "P3", "P8"]
    print(f"  Spec  Fold    N    PF    avg%   WR%    DD%   total%")
    print(f"  " + "─" * 55)
    for sid in leaders:
        sub = df_trades[df_trades.spec_id == sid]
        for f, g in sub.groupby("fold", observed=True):
            a = agg(g)
            print(f"  {sid}    {str(f):<8}{a['N']:>3}  {a['PF']:>5}  {a['avg%']:>+5.2f}  "
                  f"{a['WR%']:>5.1f}  {a['MaxDD%']:>+6.1f}  {a['total%']:>+7.1f}")
        print()

    # ── Check: how often does +3.0 ATR trigger fire (second stage of P8)? ──
    p8 = df_trades[df_trades.spec_id == "P8"]
    stage2_hits = p8["partial_count"] >= 2
    stage1_only = p8["partial_count"] == 1
    none_hit = p8["partial_count"] == 0
    print(f"P8 partial distribution:")
    print(f"  both stages hit  : {int(stage2_hits.sum())} ({stage2_hits.mean()*100:.1f}%)")
    print(f"  stage1 only      : {int(stage1_only.sum())} ({stage1_only.mean()*100:.1f}%)")
    print(f"  no partial       : {int(none_hit.sum())} ({none_hit.mean()*100:.1f}%)")
    # Capture delta on the subset where stage2 fires
    p0_matched = df_trades[(df_trades.spec_id == "P0") &
                           df_trades.set_index(["ticker","signal_date"]).index.isin(
                               p8[stage2_hits].set_index(["ticker","signal_date"]).index)]
    p8_stage2 = p8[stage2_hits]
    if len(p8_stage2) and len(p0_matched):
        print(f"  On {len(p8_stage2)} trades where both stages fire:")
        print(f"    P0 realized mean: {(p0_matched['ret_17_30_net_pct']).mean():+.2f}%  "
              f"P8 realized mean: {(p8_stage2['ret_17_30_net_pct']).mean():+.2f}%  "
              f"delta: {(p8_stage2['ret_17_30_net_pct'].mean() - p0_matched['ret_17_30_net_pct'].mean()):+.2f}pp")

    print()
    print(f"Skipped (missing OHLCV or last-bar): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
