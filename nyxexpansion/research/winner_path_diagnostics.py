"""
Winner path-type diagnostics on the locked 111 WinMag 17:30 baseline.

Pure diagnostic — no production config changes, no new backtest.

Question answered:
  "Which kinds of winners are giving back the most profit, and how does the
   current exit engine fail each kind?"

Approach:
  1. For each locked-baseline trade, reconstruct the daily-bar path from entry
     (close of signal_date) through exit (engine-reported bars_held).
  2. Compute path-shape features:
       bars_to_MFE, pct_of_MFE_reached_by_bar{1,2,3},
       first/second_half_MFE_share, monotonicity, max_runup_slope,
       late_acceleration_flag, early_spike_flag, fade_after_early_spike,
       parabolicity_proxy.
  3. Apply a simple rule-based path-type classifier:
       single_bar / early_spike_fade / slow_grinder / parabolic / other
  4. Produce Tables A (by path-type), B (×context), C (×fold), D (winner
     concentration) — for all-trades / winners / top-Q / top-D.

All returns, MFE, MAE are on 17:30 entry basis with 15 bps slippage on realized.
Path-shape features use close-entry (daily) because engine operates on dailies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LOCKED_PATH = Path("output/_locked111_signals.parquet")
OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
BT_YF_PATH = Path("output/_oldv3_yf399/backtest_v4C.parquet")  # has xu_regime

OUT_TRADE = Path("output/nyxexp_winner_path_diagnostics.csv")
OUT_SUMMARY = Path("output/nyxexp_winner_path_summary.csv")
OUT_CONTEXT = Path("output/nyxexp_winner_path_by_context.csv")

SLIPPAGE_BPS = 15
EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════
# Path-shape computation (per-bar MFE from close entry)
# ══════════════════════════════════════════════════════════════════════

def _compute_path(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  entry_price: float) -> dict:
    """
    Given the sequence of daily bars from entry_idx+1 through exit_idx inclusive,
    compute path-shape metrics.

    highs/lows/closes are relative to entry bar (i.e. first bar is entry_idx+1).
    entry_price = close[entry_idx].
    """
    n = len(highs)
    if n == 0:
        return {"bars": 0}

    # Running MFE (based on intraday highs) and MAE (based on intraday lows)
    running_max_high = np.maximum.accumulate(highs)
    running_min_low = np.minimum.accumulate(lows)

    mfe_series = (running_max_high - entry_price) / entry_price * 100.0      # % above entry
    mae_series = (running_min_low - entry_price) / entry_price * 100.0       # % below entry (negative)

    final_mfe = float(mfe_series[-1])
    final_mae = float(mae_series[-1])

    # bars_to_MFE: 1-indexed bar number when running max was last updated to the final value
    idx_mfe = int(np.argmax(mfe_series))  # first occurrence of max
    bars_to_mfe = idx_mfe + 1

    # pct of MFE reached by bar 1, 2, 3
    def _pct_by(k: int) -> float | None:
        if k > n or final_mfe <= EPS:
            return None
        return float(mfe_series[k - 1] / final_mfe)

    pct_b1 = _pct_by(1)
    pct_b2 = _pct_by(2)
    pct_b3 = _pct_by(3)

    # Halves of the trade window
    if n >= 2:
        mid = n // 2
        first_half_max_mfe = float(mfe_series[:mid].max()) if mid > 0 else float(mfe_series[0])
        second_half_max_mfe = float(mfe_series[mid:].max())
    else:
        first_half_max_mfe = final_mfe
        second_half_max_mfe = final_mfe
    if final_mfe > EPS:
        first_half_share = first_half_max_mfe / final_mfe
        second_half_share = second_half_max_mfe / final_mfe
    else:
        first_half_share = None
        second_half_share = None

    # Max incremental run-up slope (per-bar delta MFE before MFE bar)
    if idx_mfe >= 1:
        inc = np.diff(mfe_series[: idx_mfe + 1])  # length idx_mfe
        max_runup_slope = float(inc.max()) if len(inc) else 0.0
    else:
        max_runup_slope = float(mfe_series[0])  # bar 1 alone

    # Monotonicity proxy = fraction of positive incremental bars before MFE
    if idx_mfe >= 1:
        pos_inc = np.sum(np.diff(mfe_series[: idx_mfe + 1]) > 0)
        monotonicity = pos_inc / idx_mfe
    else:
        monotonicity = 1.0  # single bar → treat as perfectly monotonic

    # Parabolicity proxy: max incremental slope in second half of run-up relative to first half
    if idx_mfe >= 2:
        half = max(1, idx_mfe // 2)
        inc = np.diff(mfe_series[: idx_mfe + 1])
        first_inc = inc[:half]
        second_inc = inc[half:]
        first_max = float(first_inc.max()) if len(first_inc) else 0.0
        second_max = float(second_inc.max()) if len(second_inc) else 0.0
        parabolicity = (second_max - first_max) if (first_max + second_max) > 0 else 0.0
    else:
        parabolicity = 0.0

    late_accel = parabolicity > 1.0   # second-half bar ran up >1pp more than first-half max
    early_spike = (bars_to_mfe <= 2) and (pct_b2 is not None and pct_b2 >= 0.70)

    # Fade after early spike = how much of MFE was given back by the final close
    final_close_ret_pct = (closes[-1] - entry_price) / entry_price * 100.0
    fade_after_early_spike = float(final_mfe - final_close_ret_pct) if early_spike else None

    return {
        "bars": n,
        "mfe_close_pct": final_mfe,
        "mae_close_pct": final_mae,
        "bars_to_MFE": bars_to_mfe,
        "pct_of_MFE_by_bar1": pct_b1,
        "pct_of_MFE_by_bar2": pct_b2,
        "pct_of_MFE_by_bar3": pct_b3,
        "first_half_MFE_share": first_half_share,
        "second_half_MFE_share": second_half_share,
        "max_runup_slope": max_runup_slope,
        "monotonicity": monotonicity,
        "parabolicity": parabolicity,
        "late_acceleration_flag": bool(late_accel),
        "early_spike_flag": bool(early_spike),
        "fade_after_early_spike_pct": fade_after_early_spike,
        "final_close_ret_close_pct": final_close_ret_pct,
    }


def _classify_path(row: dict) -> str:
    """Deterministic rule-based path-type assignment."""
    n = row["bars"]
    if n == 1:
        return "single_bar"

    final_mfe = row["mfe_close_pct"]
    # If MFE is essentially zero (no up move at all), classify as other
    if final_mfe <= 0.5:
        return "other"

    b2mfe = row["bars_to_MFE"]
    p2 = row["pct_of_MFE_by_bar2"] or 0.0
    fhs = row["first_half_MFE_share"] or 0.0
    shs = row["second_half_MFE_share"] or 0.0
    mono = row["monotonicity"] or 0.0
    late = row["late_acceleration_flag"]
    early = row["early_spike_flag"]
    fade = row["fade_after_early_spike_pct"] or 0.0

    # A) early_spike_fade — strong burst by bar 2, then fade
    if early and fade >= 3.0:
        return "early_spike_fade"

    # C) parabolic — late accel, second-half dominant
    if n >= 3 and late and shs >= 0.65:
        return "parabolic"

    # B) slow_grinder — multi-bar build, second half builds up, no spike
    if n >= 4 and b2mfe >= 3 and fhs < 0.6 and mono >= 0.5 and not early:
        return "slow_grinder"

    # else
    return "other"


# ══════════════════════════════════════════════════════════════════════
# Loader
# ══════════════════════════════════════════════════════════════════════

def _load_panel(tickers):
    oh = pd.read_parquet(OHLCV_PATH)
    oh = oh[oh["ticker"].isin(list(set(tickers)))].copy()
    if oh.index.name:
        oh = oh.reset_index()
    if "Date" not in oh.columns:
        oh["Date"] = pd.to_datetime(oh.iloc[:, 0])
    oh["Date"] = pd.to_datetime(oh["Date"])
    return {t: g.sort_values("Date").reset_index(drop=True) for t, g in oh.groupby("ticker", sort=False)}


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════

def build_trade_level() -> pd.DataFrame:
    locked = pd.read_parquet(LOCKED_PATH)
    # Merge in xu_regime from the original backtest
    eng = pd.read_parquet(BT_YF_PATH)[["ticker", "date", "xu_regime"]].copy()
    eng["signal_date"] = pd.to_datetime(eng["date"]).dt.date
    locked = locked.merge(eng[["ticker", "signal_date", "xu_regime"]],
                          on=["ticker", "signal_date"], how="left")

    panel = _load_panel(locked["ticker"].unique())
    rows = []
    for _, r in locked.iterrows():
        t = r.ticker
        d = pd.to_datetime(r.signal_date)
        bars_held = int(r.bars_held)
        if t not in panel:
            continue
        df_t = panel[t]
        matches = df_t.index[df_t["Date"] == d]
        if len(matches) == 0:
            continue
        entry_idx = int(matches[0])
        end_idx = min(entry_idx + bars_held, len(df_t) - 1)
        if end_idx <= entry_idx:
            continue
        entry_price = float(df_t.loc[entry_idx, "Close"])
        highs = df_t.loc[entry_idx + 1: end_idx, "High"].to_numpy(float)
        lows = df_t.loc[entry_idx + 1: end_idx, "Low"].to_numpy(float)
        closes = df_t.loc[entry_idx + 1: end_idx, "Close"].to_numpy(float)
        if len(highs) == 0:
            continue

        path = _compute_path(highs, lows, closes, entry_price)

        # 17:30 adjustment
        adj = float(r.price_18_00) / float(r.price_17_30) if r.price_17_30 > 0 else 1.0
        mfe17 = (1 + path["mfe_close_pct"] / 100.0) * adj - 1.0
        mae17 = (1 + path["mae_close_pct"] / 100.0) * adj - 1.0
        realized_close = float(r.gross_return)  # already cost- and slip-adjusted on exit leg
        realized17 = (1 + realized_close / 100.0) * adj - 1.0
        realized17_net_pct = realized17 * 100.0 - SLIPPAGE_BPS / 100.0

        row = {
            "ticker": t,
            "signal_date": r.signal_date,
            "fold": r.fold,
            "xu_regime": r.xu_regime,
            "risk_bucket": r.risk_bucket,
            "score": r.score,
            "upside_room_52w_atr": r.upside_room_52w_atr,
            "bars_held": bars_held,
            "reason": r.reason,
            "partial_taken": r.partial_taken,
            # returns on 17:30 basis
            "realized_17_30_pct": realized17_net_pct,
            "mfe_17_30_pct": mfe17 * 100.0,
            "mae_17_30_pct": mae17 * 100.0,
            "giveback_pct": mfe17 * 100.0 - realized17_net_pct,
            "realized_over_MFE": (realized17_net_pct / max(mfe17 * 100.0, EPS)
                                  if mfe17 * 100.0 > EPS else np.nan),
        }
        row.update(path)
        row["path_type"] = _classify_path(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _agg(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"N": 0}
    r = g["realized_17_30_pct"]
    mfe = g["mfe_17_30_pct"]
    mae = g["mae_17_30_pct"]
    gb = g["giveback_pct"]
    rom = g["realized_over_MFE"].replace([np.inf, -np.inf], np.nan)
    wins = (r > 0).sum()
    reasons = g["reason"].value_counts(normalize=True).to_dict()
    pf = (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.inf
    return {
        "N": n,
        "WR%": round(wins / n * 100, 1),
        "avg_realized%": round(r.mean(), 2),
        "avg_MFE%": round(mfe.mean(), 2),
        "avg_MAE%": round(mae.mean(), 2),
        "avg_giveback%": round(gb.mean(), 2),
        "mean_realized_over_MFE": round(rom.mean(), 3) if rom.notna().any() else None,
        "median_realized_over_MFE": round(rom.median(), 3) if rom.notna().any() else None,
        "avg_bars_held": round(g["bars_held"].mean(), 2),
        "PF": round(pf, 2) if np.isfinite(pf) else float("inf"),
        "total_giveback_pct": round(gb.sum(), 1),
        "reason_stop%": round(reasons.get("stop", 0) * 100, 1),
        "reason_time%": round(reasons.get("time", 0) * 100, 1),
        "reason_failbk%": round(reasons.get("failed_breakout", 0) * 100, 1),
    }


def run():
    df = build_trade_level()
    print(f"Built diagnostics for {len(df)} trades")
    print()
    print("Path-type distribution:")
    print(df["path_type"].value_counts().to_frame("N"))
    print()
    print("Exit reason by path-type:")
    print(pd.crosstab(df["path_type"], df["reason"]))
    print()

    OUT_TRADE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TRADE, index=False)
    print(f"✅ trade-level: {OUT_TRADE}")

    # ── Cohorts ─────────────────────────────────────────────────────────
    cohorts = {
        "all": df,
        "winners": df[df["realized_17_30_pct"] > 0],
        "top_quartile_MFE": df[df["mfe_17_30_pct"] >= df["mfe_17_30_pct"].quantile(0.75)],
        "top_decile_MFE": df[df["mfe_17_30_pct"] >= df["mfe_17_30_pct"].quantile(0.90)],
    }

    # Table A — by path-type × cohort
    rowsA = []
    for cname, cdf in cohorts.items():
        for p, g in cdf.groupby("path_type", observed=True):
            row = {"cohort": cname, "path_type": p}
            row.update(_agg(g))
            rowsA.append(row)
    tblA = pd.DataFrame(rowsA)

    # Table B — path-type × risk_bucket (winners only)
    rowsB = []
    for p, g1 in cohorts["winners"].groupby("path_type", observed=True):
        for b, g2 in g1.groupby("risk_bucket", observed=True):
            row = {"scope": "winners", "path_type": p, "risk_bucket": b}
            row.update(_agg(g2))
            rowsB.append(row)
    # also regime overlay
    for p, g1 in cohorts["winners"].groupby("path_type", observed=True):
        for reg, g2 in g1.groupby("xu_regime", observed=True):
            row = {"scope": "winners", "path_type": p, "regime": reg}
            row.update(_agg(g2))
            rowsB.append(row)
    tblB = pd.DataFrame(rowsB)

    # Table C — path-type × fold
    rowsC = []
    for p, g1 in cohorts["winners"].groupby("path_type", observed=True):
        for f, g2 in g1.groupby("fold", observed=True):
            row = {"scope": "winners", "path_type": p, "fold": f}
            row.update(_agg(g2))
            rowsC.append(row)
    tblC = pd.DataFrame(rowsC)

    # Table D — winner concentration
    rowsD = []
    for cname in ["top_quartile_MFE", "top_decile_MFE"]:
        cdf = cohorts[cname]
        total_gb = cdf["giveback_pct"].sum()
        for p, g in cdf.groupby("path_type", observed=True):
            share = len(g) / max(len(cdf), 1) * 100
            gb = g["giveback_pct"].sum()
            gb_share = gb / total_gb * 100 if total_gb != 0 else 0.0
            rowsD.append({
                "cohort": cname,
                "path_type": p,
                "N": len(g),
                "share_of_cohort_%": round(share, 1),
                "avg_giveback%": round(g["giveback_pct"].mean(), 2),
                "total_giveback%": round(gb, 1),
                "share_of_total_giveback_%": round(gb_share, 1),
            })
    tblD = pd.DataFrame(rowsD)

    # ── Print ───────────────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print()
    print("═" * 110)
    print("TABLE A — path-type summary × cohort")
    print("═" * 110)
    for cname, sub in tblA.groupby("cohort", sort=False):
        print(f"\n[{cname}]")
        print(sub.drop(columns=["cohort"]).to_string(index=False))

    print()
    print("═" * 110)
    print("TABLE B — path-type × context (winners only)")
    print("═" * 110)
    print("\n▸ by risk_bucket:")
    print(tblB[tblB["risk_bucket"].notna()].drop(columns=["regime"], errors="ignore").to_string(index=False))
    print("\n▸ by regime:")
    print(tblB[tblB.get("regime").notna() if "regime" in tblB.columns else False]
          .drop(columns=["risk_bucket"], errors="ignore").to_string(index=False))

    print()
    print("═" * 110)
    print("TABLE C — path-type × fold (winners only)")
    print("═" * 110)
    print(tblC.to_string(index=False))

    print()
    print("═" * 110)
    print("TABLE D — winner concentration (who contributes most giveback)")
    print("═" * 110)
    print(tblD.to_string(index=False))

    # Save
    # Collapse tables to one long summary CSV
    tblA["table"] = "A_path_type_cohort"
    tblB["table"] = "B_path_type_context"
    tblC["table"] = "C_path_type_fold"
    tblD["table"] = "D_winner_concentration"
    combined = pd.concat([tblA, tblB, tblC, tblD], ignore_index=True, sort=False)
    combined.to_csv(OUT_SUMMARY, index=False)
    print(f"\n✅ summary: {OUT_SUMMARY}")

    # Context CSV (tables B + D)
    pd.concat([tblB, tblD], ignore_index=True, sort=False).to_csv(OUT_CONTEXT, index=False)
    print(f"✅ context: {OUT_CONTEXT}")

    # ── Decision-question answers ───────────────────────────────────────
    print()
    print("═" * 110)
    print("DIAGNOSTIC ANSWERS")
    print("═" * 110)

    # 1. Which path-type leaves the most profit on the table?
    gb_rank = (df.groupby("path_type")
                 .agg(N=("giveback_pct", "size"),
                      total_gb=("giveback_pct", "sum"),
                      avg_gb=("giveback_pct", "mean"),
                      avg_mfe=("mfe_17_30_pct", "mean"))
                 .sort_values("total_gb", ascending=False))
    print("\nQ1 — Giveback ranking by path-type (all trades):")
    print(gb_rank.round(2))

    # 2. Main exit reason in the worst giveback group
    worst = gb_rank.index[0]
    worst_grp = df[df.path_type == worst]
    print(f"\nQ2/Q5 — Worst giveback group is '{worst}' (N={len(worst_grp)}, total_gb={worst_grp.giveback_pct.sum():.1f}pp)")
    print(f"        Exit reason mix in this group: {worst_grp.reason.value_counts(normalize=True).round(3).to_dict()}")

    # 3. clean vs elevated comparison within top-quartile MFE
    tq = cohorts["top_quartile_MFE"]
    print(f"\nQ3 — Top-quartile MFE cohort (N={len(tq)}) — giveback by bucket:")
    print(tq.groupby("risk_bucket", observed=True).agg(
        N=("giveback_pct", "size"),
        avg_mfe=("mfe_17_30_pct", "mean"),
        avg_realized=("realized_17_30_pct", "mean"),
        avg_giveback=("giveback_pct", "mean"),
        median_rom=("realized_over_MFE", "median"),
    ).round(2))

    # 4. Early/late exit character per path-type (avg bars_to_MFE vs bars_held)
    bcompare = (df.groupby("path_type")
                  .agg(N=("bars_held", "size"),
                       avg_bars_to_MFE=("bars_to_MFE", "mean"),
                       avg_bars_held=("bars_held", "mean"))
                  .round(2))
    bcompare["gap"] = (bcompare["avg_bars_to_MFE"] - bcompare["avg_bars_held"]).round(2)
    print(f"\nQ4 — Timing alignment (bars_to_MFE vs bars_held):")
    print(bcompare)
    print("    gap < 0 → MFE reached before exit (exit is ok-to-late); "
          "gap ≈ 0 → exit at peak; gap > 0 shouldn't happen (MFE capped at bars_held).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
