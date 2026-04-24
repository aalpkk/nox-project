"""
Path-type-aware exit overlays — research only.

Builds three runtime-conditional overlays based on the winner-path diagnosis
(2026-04-23) and tests them against the locked 111 baseline:

  OL_PARA   convexity-triggered (parabolic) — when running MFE crosses 3 ATR,
            switch trail multiplier 2.5→1.5; at 4 ATR take 15% extra partial.
  OL_SPIKE  early-spike-fade — within first 2 bars, if MFE ≥ 1 ATR AND current
            bar closes red AND below prior close → take 25% partial + move BE.
  OL_CLEAN  clean-context only — replace default partial with +1.0 ATR / 33% / BE.

Compared to:
  P0   control (swing default: +1.5 ATR / 40% / BE, trail 2.5 ATR)
  P8   two-stage 15% @+1.5 ATR + 15% @+3.0 ATR / BE after first

Universe / filters / selection unchanged. 17:30 entry with 15 bps slippage.

Notes
- All overlays are realtime-detectable (no future leak).
- OL_PARA tightens trail mid-trade and adds a second partial if convexity fires.
- OL_SPIKE only acts in bars 1–2 of the trade (matches the early_spike_fade
  diagnosis: those trades peaked by bar 2).
- OL_CLEAN is preselected at entry by risk_bucket; non-clean trades fall back
  to the swing default behavior (P0 logic).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exits.config import SWING, DEFAULT_COST_MODEL  # noqa: E402
from exits.structure import last_swing_low  # noqa: E402

LOCKED_PATH = Path("output/_locked111_signals.parquet")
OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
OUT_SUMMARY = Path("output/nyxexp_path_overlay_ablation.csv")
OUT_TRADES = Path("output/nyxexp_path_overlay_tradelevel.parquet")

SLIPPAGE_BPS = 15
EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════
# Spec representation: a function(state) -> action(s)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OverlaySpec:
    id: str
    label: str
    # default partial schedule: list of (kind, value, fraction)
    #   kind ∈ {'atr','R'}; value = ATR multiple or R multiple; fraction ∈ [0,1]
    base_partials: tuple = ()
    # if set and ctx["risk_bucket"] == "clean", use these instead of base_partials
    clean_override_partials: tuple = ()
    # base trail multiplier (overrides cfg.trail_atr_mult when set)
    base_trail_mult: float | None = None
    # gate: only apply overlay extras if gate(ctx) is True
    gate: Callable[[dict], bool] | None = None
    # convexity-triggered tighter trail: when running MFE in ATR units >=
    # convexity_trail_at_atr → switch trail multiplier to convexity_trail_mult
    convexity_trail_at_atr: float | None = None
    convexity_trail_mult: float | None = None
    # convexity-triggered second partial: at MFE_atr >= convexity_partial_at_atr
    # take an extra fraction. Only applied if not already in base_partials.
    convexity_partial_at_atr: float | None = None
    convexity_partial_frac: float | None = None
    # early-spike momentum-loss: in first early_spike_bars, if running_MFE_atr ≥
    # early_spike_min_atr AND today's close < open AND close < prior_close →
    # take early_spike_frac partial + move BE
    early_spike_bars: int = 0
    early_spike_min_atr: float = 0.0
    early_spike_frac: float = 0.0


def _gate_clean(ctx: dict) -> bool:
    return ctx.get("risk_bucket") == "clean"


SPECS: tuple[OverlaySpec, ...] = (
    OverlaySpec(
        id="P0",
        label="CONTROL — swing default (+1.5 ATR / 40% / BE, trail 2.5 ATR)",
        base_partials=(("atr", 1.5, 0.40),),
    ),
    OverlaySpec(
        id="P8",
        label="P8 two-stage (15% @+1.5 ATR + 15% @+3.0 ATR, BE)",
        base_partials=(("atr", 1.5, 0.15), ("atr", 3.0, 0.15)),
    ),
    OverlaySpec(
        id="OL_PARA",
        label="Parabolic overlay — base P0 + tighter trail @MFE≥3 ATR + 15% partial @MFE≥4 ATR",
        base_partials=(("atr", 1.5, 0.40),),
        convexity_trail_at_atr=3.0,
        convexity_trail_mult=1.5,
        convexity_partial_at_atr=4.0,
        convexity_partial_frac=0.15,
    ),
    OverlaySpec(
        id="OL_SPIKE",
        label="Early-spike-fade overlay — base P0 + bar1-2 momentum-loss 25% partial",
        base_partials=(("atr", 1.5, 0.40),),
        early_spike_bars=2,
        early_spike_min_atr=1.0,
        early_spike_frac=0.25,
    ),
    OverlaySpec(
        id="OL_CLEAN",
        label="Clean-context overlay — clean: +1.0 ATR / 33% / BE; non-clean: P0",
        base_partials=(("atr", 1.5, 0.40),),  # default for non-clean
        clean_override_partials=(("atr", 1.0, 0.33),),
    ),
)


# ══════════════════════════════════════════════════════════════════════
# ATR + simulation
# ══════════════════════════════════════════════════════════════════════

def _wilder_atr(h, l, c, period=14):
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr[:period - 1] = atr[period - 1]
    return atr


def _simulate_overlay(
    df_ohlcv: pd.DataFrame,
    entry_idx: int,
    atr_series: np.ndarray,
    spec: OverlaySpec,
    ctx: dict,
) -> dict:
    cfg = SWING
    cost = DEFAULT_COST_MODEL
    o = df_ohlcv["Open"].to_numpy(float)
    h = df_ohlcv["High"].to_numpy(float)
    l = df_ohlcv["Low"].to_numpy(float)
    c = df_ohlcv["Close"].to_numpy(float)
    n = len(c)

    entry_price = float(c[entry_idx])
    entry_atr = float(atr_series[entry_idx]) if atr_series[entry_idx] > 0 else max(1e-6, entry_price * 0.02)

    initial_stop = entry_price - cfg.initial_stop_atr * entry_atr
    sw = last_swing_low(l, entry_idx, pivot_strength=cfg.pivot_strength, lookback=cfg.structure_lookback)
    if sw is not None:
        initial_stop = max(initial_stop, sw.price)
    current_stop = initial_stop
    R_dist = max(1e-6, entry_price - initial_stop)

    # ── Resolve overlay-aware base partials and trail ───────────────────
    if spec.clean_override_partials and ctx.get("risk_bucket") == "clean":
        base_partials = spec.clean_override_partials
    else:
        base_partials = spec.base_partials
    base_trail_mult = spec.base_trail_mult or cfg.trail_atr_mult

    # Pre-compute base partial price targets
    base_targets: list[tuple[float, float, str]] = []  # (price, frac, key)
    for kind, val, frac in base_partials:
        if kind == "atr":
            tgt = entry_price + val * entry_atr
        elif kind == "R":
            tgt = entry_price + val * R_dist
        else:
            continue
        base_targets.append((float(tgt), float(frac), f"base_{kind}_{val}"))

    partials_done: list[tuple[int, float, float, str]] = []  # (idx, price, frac, key)
    highest_since_entry = entry_price
    trail_activated = False
    be_moved = False
    convex_trail_active = False
    convex_partial_done = False
    spike_partial_done = False

    mfe_high = entry_price
    mae_low = entry_price

    exit_idx = entry_idx
    exit_price_raw = entry_price
    reason = "open"

    for i in range(entry_idx + 1, min(entry_idx + 1 + cfg.time_stop_bars, n)):
        bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
        bars_held = i - entry_idx
        prior_c = c[i - 1]

        mfe_high = max(mfe_high, bar_h)
        mae_low = min(mae_low, bar_l)
        running_mfe_atr = (mfe_high - entry_price) / entry_atr

        # 0. Gap below stop
        if bar_o <= current_stop:
            exit_idx = i; exit_price_raw = bar_o; reason = "stop"; break
        if bar_l <= current_stop:
            exit_idx = i; exit_price_raw = current_stop; reason = "stop"; break

        # Failed breakout
        fb_threshold = entry_price * (1 - cfg.failed_breakout_threshold_pct / 100.0)
        if cfg.use_failed_breakout and bars_held <= cfg.failed_breakout_bars and bar_c < fb_threshold:
            exit_idx = i; exit_price_raw = bar_c; reason = "failed_breakout"; break

        # ── BASE partial triggers (any not yet done) ───────────────────
        for tgt_price, frac, key in base_targets:
            already = any(k == key for *_, k in partials_done)
            if not already and bar_h >= tgt_price:
                partials_done.append((i, tgt_price, frac, key))
                if not be_moved:
                    current_stop = max(current_stop, entry_price)
                    be_moved = True

        # ── OL_PARA convexity-triggered second partial ─────────────────
        if (spec.convexity_partial_at_atr is not None
                and not convex_partial_done
                and running_mfe_atr >= spec.convexity_partial_at_atr):
            target = entry_price + spec.convexity_partial_at_atr * entry_atr
            if bar_h >= target:
                partials_done.append((i, target, float(spec.convexity_partial_frac), "convex"))
                convex_partial_done = True
                if not be_moved:
                    current_stop = max(current_stop, entry_price)
                    be_moved = True

        # ── OL_PARA convexity-triggered tighter trail ─────────────────
        if (spec.convexity_trail_at_atr is not None
                and not convex_trail_active
                and running_mfe_atr >= spec.convexity_trail_at_atr):
            convex_trail_active = True

        # ── OL_SPIKE early-spike momentum-loss partial ────────────────
        if (spec.early_spike_bars > 0
                and not spike_partial_done
                and bars_held <= spec.early_spike_bars
                and running_mfe_atr >= spec.early_spike_min_atr
                and bar_c < bar_o
                and bar_c < prior_c):
            target = bar_c  # exit at current close
            partials_done.append((i, target, float(spec.early_spike_frac), "spike"))
            spike_partial_done = True
            if not be_moved:
                current_stop = max(current_stop, entry_price)
                be_moved = True

        # ── ATR trail update ──────────────────────────────────────────
        if cfg.use_atr_trail:
            highest_since_entry = max(highest_since_entry, bar_h)
            activation_price = entry_price + cfg.trail_activate_atr * entry_atr
            if not trail_activated and highest_since_entry >= activation_price:
                trail_activated = True
            if trail_activated:
                trail_mult = (spec.convexity_trail_mult
                              if convex_trail_active and spec.convexity_trail_mult is not None
                              else base_trail_mult)
                trail_stop = highest_since_entry - trail_mult * entry_atr
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

    # Slippage on final exit leg
    if reason in ("stop", "structure", "trail"):
        slip = cost.exit_slippage_stop_pct / 100.0
    elif reason in ("time", "failed_breakout"):
        slip = cost.exit_slippage_time_pct / 100.0
    else:
        slip = 0.0
    exit_price = exit_price_raw * (1.0 - slip)
    final_gross = (exit_price / entry_price - 1.0) * 100.0

    # Blend partials (TP slip on partials, except spike-partial which fills at bar_c → use stop slip)
    part_component = 0.0
    total_partial_frac = 0.0
    for _, p_price, p_frac, key in partials_done:
        slip_pct = cost.exit_slippage_stop_pct / 100.0 if key == "spike" else cost.exit_slippage_tp_pct / 100.0
        p_exit = p_price * (1.0 - slip_pct)
        p_ret = (p_exit / entry_price - 1.0) * 100.0
        part_component += p_frac * p_ret
        total_partial_frac += p_frac
    remainder_frac = max(0.0, 1.0 - total_partial_frac)
    blended_gross = part_component + remainder_frac * final_gross

    return {
        "exit_idx": int(exit_idx),
        "exit_price": float(exit_price),
        "reason": reason,
        "bars_held": int(exit_idx - entry_idx),
        "blended_gross_ret_pct": float(blended_gross),
        "mfe_close_pct": float((mfe_high / entry_price - 1.0) * 100.0),
        "mae_close_pct": float((mae_low / entry_price - 1.0) * 100.0),
        "partial_count": len(partials_done),
        "partial_keys": tuple(k for *_, k in partials_done),
        "convex_trail_active": convex_trail_active,
        "convex_partial_done": convex_partial_done,
        "spike_partial_done": spike_partial_done,
    }


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def _load_panel(tickers):
    oh = pd.read_parquet(OHLCV_PATH)
    oh = oh[oh["ticker"].isin(list(set(tickers)))].copy()
    if oh.index.name:
        oh = oh.reset_index()
    if "Date" not in oh.columns:
        oh["Date"] = pd.to_datetime(oh.iloc[:, 0])
    oh["Date"] = pd.to_datetime(oh["Date"])
    out = {}
    for t, g in oh.groupby("ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        atr = _wilder_atr(g["High"].to_numpy(float), g["Low"].to_numpy(float), g["Close"].to_numpy(float))
        out[t] = (g, atr)
    return out


def _agg(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {}
    r = g["ret_17_30_net_pct"] / 100.0
    mfe = g["mfe_17_30_pct"]
    gb = g["giveback_pct"]
    wins_mask = r > 0
    wins = int(wins_mask.sum())
    pf = (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.inf
    eq = (1 + r.fillna(0)).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100 if len(eq) else 0.0
    if wins > 0:
        cap = (r[wins_mask] * 100 / mfe[wins_mask].clip(lower=EPS)).replace([np.inf, -np.inf], np.nan)
        win_med = float(cap.median())
        avg_win = float(r[wins_mask].mean() * 100)
    else:
        win_med = None
        avg_win = 0.0
    return {
        "N": n,
        "PF": round(pf, 2) if np.isfinite(pf) else float("inf"),
        "avg%": round(r.mean() * 100, 2),
        "WR%": round(wins / n * 100, 1),
        "MaxDD%": round(dd, 1),
        "total%": round((eq.iloc[-1] - 1) * 100, 1),
        "avg_winner%": round(avg_win, 2),
        "avg_mfe%": round(mfe.mean(), 2),
        "avg_giveback%": round(gb.mean(), 2),
        "winner_capture_med": round(win_med, 3) if win_med is not None else None,
        "avg_bars_held": round(g["bars_held"].mean(), 2),
        "partial_share%": round((g["partial_count"] > 0).mean() * 100, 1),
    }


def run():
    locked = pd.read_parquet(LOCKED_PATH)
    panel = _load_panel(locked["ticker"].unique())
    print(f"Loaded locked baseline: {len(locked)} signals")
    print(f"Bucket dist: {locked.risk_bucket.value_counts().to_dict()}")
    print()

    rows = []
    for spec in SPECS:
        for _, r in locked.iterrows():
            t = r.ticker
            d = pd.to_datetime(r.signal_date)
            if t not in panel:
                continue
            df_t, atr_t = panel[t]
            matches = df_t.index[df_t["Date"] == d]
            if len(matches) == 0:
                continue
            entry_idx = int(matches[0])
            if entry_idx >= len(df_t) - 1:
                continue
            ctx = {"risk_bucket": r.risk_bucket, "upside_room_52w_atr": r.upside_room_52w_atr}
            sim = _simulate_overlay(df_t, entry_idx, atr_t, spec, ctx)

            adj = float(r.price_18_00) / float(r.price_17_30) if r.price_17_30 > 0 else 1.0
            ret17 = ((1 + sim["blended_gross_ret_pct"] / 100.0) * adj - 1.0) * 100.0
            mfe17 = ((1 + sim["mfe_close_pct"] / 100.0) * adj - 1.0) * 100.0
            mae17 = ((1 + sim["mae_close_pct"] / 100.0) * adj - 1.0) * 100.0
            ret_net = ret17 - SLIPPAGE_BPS / 100.0

            rows.append({
                "spec_id": spec.id,
                "spec_label": spec.label,
                "ticker": t,
                "signal_date": r.signal_date,
                "fold": r.fold,
                "risk_bucket": r.risk_bucket,
                "reason": sim["reason"],
                "bars_held": sim["bars_held"],
                "partial_count": sim["partial_count"],
                "partial_keys": ",".join(sim["partial_keys"]) if sim["partial_keys"] else "",
                "convex_trail_active": sim["convex_trail_active"],
                "convex_partial_done": sim["convex_partial_done"],
                "spike_partial_done": sim["spike_partial_done"],
                "ret_17_30_net_pct": ret_net,
                "mfe_17_30_pct": mfe17,
                "mae_17_30_pct": mae17,
                "giveback_pct": mfe17 - ret_net,
            })

    df = pd.DataFrame(rows)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_TRADES, index=False)
    print(f"✅ trade-level output: {OUT_TRADES} ({len(df)} rows)")

    # ── Spec summary ────────────────────────────────────────────────────
    summ_rows = []
    for spec in SPECS:
        g = df[df.spec_id == spec.id]
        row = {"spec_id": spec.id, "spec_label": spec.label}
        row.update(_agg(g))
        summ_rows.append(row)
    summary = pd.DataFrame(summ_rows)
    p0 = summary.loc[summary.spec_id == "P0"].iloc[0]
    for col in ["PF", "avg%", "WR%", "MaxDD%", "total%", "avg_giveback%", "winner_capture_med"]:
        base = p0[col]
        if base is None or (isinstance(base, float) and np.isnan(base)):
            summary["Δ_" + col] = None
            continue
        summary["Δ_" + col] = summary[col].apply(
            lambda v: round(float(v) - float(base), 3)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v)) else None
        )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"✅ summary: {OUT_SUMMARY}")
    print()

    # Console
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("═" * 130)
    print("PATH-TYPE OVERLAY ABLATION vs P0 (control) — locked 111, 17:30 entry @ 15 bps")
    print("═" * 130)
    cols = ["spec_id", "N", "PF", "avg%", "WR%", "MaxDD%", "total%", "avg_mfe%",
            "avg_giveback%", "winner_capture_med", "avg_bars_held",
            "Δ_PF", "Δ_MaxDD%", "Δ_total%", "Δ_avg_giveback%"]
    print(summary[cols].to_string(index=False))
    print()

    # ── Decision check ────────────────────────────────────────────────
    print("═" * 100)
    print("DECISION CHECK (vs P0): ΔPF ≥ 0 · ΔDD ≥ −3pp · Δtotal ≥ 0 · winner tail ≥ 0.9·P0")
    print("═" * 100)
    p0_winner = float(p0["avg_winner%"])
    for _, row in summary.iterrows():
        if row.spec_id == "P0":
            continue
        pf_ok = (row["Δ_PF"] or 0) >= 0
        dd_ok = (row["Δ_MaxDD%"] or 0) >= -3.0
        tot_ok = (row["Δ_total%"] or 0) >= 0
        winner_ok = float(row["avg_winner%"]) >= 0.9 * p0_winner
        flags = "".join(["✓" if x else "✗" for x in (pf_ok, dd_ok, tot_ok, winner_ok)])
        verdict = "PASS" if all([pf_ok, dd_ok, tot_ok, winner_ok]) else "fail"
        print(f"  {row.spec_id:<10} {verdict}  [PF/DD/tot/win:{flags}]  "
              f"ΔPF={row['Δ_PF']:+.2f}  ΔDD={row['Δ_MaxDD%']:+.1f}pp  "
              f"Δtot={row['Δ_total%']:+.1f}pp  win={row['avg_winner%']:.2f}%")

    # ── Per-fold stability of the leaders ─────────────────────────────
    print()
    print("═" * 100)
    print("PER-FOLD STABILITY (PF and total% per fold for each spec)")
    print("═" * 100)
    for sid in [s.id for s in SPECS]:
        sub = df[df.spec_id == sid]
        line = f"  {sid:<10}"
        for f in ["fold1", "fold2", "fold3"]:
            g = sub[sub.fold == f]
            if not len(g):
                line += f"  {f}:n/a"
                continue
            r = g["ret_17_30_net_pct"] / 100
            pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
            tot = ((1 + r.fillna(0)).cumprod().iloc[-1] - 1) * 100
            line += f"  {f}: PF={pf:>4.2f} tot={tot:>+6.1f}%"
        print(line)

    # ── Targeted impact on flagged path-types ─────────────────────────
    # Use the diagnostic CSV to see which trades were classified parabolic
    # and check overlay impact on those specifically.
    diag_path = Path("output/nyxexp_winner_path_diagnostics.csv")
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        diag["signal_date"] = pd.to_datetime(diag["signal_date"]).dt.date
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
        merged = df.merge(diag[["ticker", "signal_date", "path_type"]],
                          on=["ticker", "signal_date"], how="left")
        print()
        print("═" * 110)
        print("TARGETED IMPACT — by diagnosed path-type (P0 vs each overlay)")
        print("═" * 110)
        for ptype in ["parabolic", "early_spike_fade", "single_bar", "other", "slow_grinder"]:
            sub = merged[merged.path_type == ptype]
            if not len(sub):
                continue
            n_trades = len(sub[sub.spec_id == "P0"])
            print(f"\n[{ptype}]  N={n_trades}")
            print(f"  {'spec':<10} {'avg_real%':>10} {'avg_gb%':>10} {'avg_mfe%':>10} {'win_cap':>9} "
                  f"{'reasonMix':>20}  Δreal_vs_P0")
            p0_real = sub[sub.spec_id == "P0"]["ret_17_30_net_pct"].mean()
            for sid in [s.id for s in SPECS]:
                ssub = sub[sub.spec_id == sid]
                if not len(ssub):
                    continue
                r = ssub["ret_17_30_net_pct"]
                mfe = ssub["mfe_17_30_pct"]
                gb = ssub["giveback_pct"]
                win_cap = (r[r > 0] / mfe[r > 0].clip(lower=EPS)).median() if (r > 0).any() else np.nan
                rmix = ssub["reason"].value_counts(normalize=True).to_dict()
                rmix_s = " ".join(f"{k}={v*100:.0f}" for k, v in rmix.items())
                delta = r.mean() - p0_real
                print(f"  {sid:<10} {r.mean():>+9.2f} {gb.mean():>9.2f} {mfe.mean():>9.2f} "
                      f"{win_cap:>9.2f} {rmix_s:>20}  {delta:>+10.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
