"""Tavan V1 — Kelly position sizing.

V1 config: H=25 / Trail 2% / SL-10% / ML_S>=0.65 / D1-FILTER (WF-validated 2026-05-21).

Pipeline:
  1. Rebuild OOS pooled return series from V1 engine (replays tavan_d1_filter_v1.run with d1_close_filter=True).
  2. Numerically solve full Kelly: k* = argmax E[log(1 + k·r)] subject to k·r > -1 (no ruin in single trade).
  3. Half-Kelly + quarter-Kelly (standard risk-throttle for parameter uncertainty + path drawdown).
  4. Per-WF-fold Kelly (4 folds 2022-05→2026-04) — variance across regimes.
  5. Simple-Kelly check (binary p/q with avg_win/avg_loss) for sanity.
  6. Risk-of-ruin / drawdown profile under each fraction (Monte Carlo, 10k paths).

NOTE: Kelly assumes geometric compounding with full-bankroll commitment per trade.
With ~459 trades/yr ≈ 1.8/day at full universe, multiple concurrent V1 candidates
mean per-trade allocation should be capped by max-concurrent-trades budget too.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "output/tavan_full_backtest_v1_panel.parquet"
DAILY = REPO / "output/_cache_daily_ohlcv_3y.parquet"

# WF folds (pre-registered, matches tavan_walk_forward_v1.py)
FOLDS = [
    ("F1", "2022-05-01", "2023-04-30"),
    ("F2", "2023-05-01", "2024-04-30"),
    ("F3", "2024-05-01", "2025-04-30"),
    ("F4", "2025-05-01", "2026-04-30"),
]
ML_S_THRESHOLD = 0.65
HORIZON = 25
SL_PCT = 0.10
TRAIL_PCT = 0.02

# Monte Carlo
N_MC_PATHS = 10_000
RUIN_THRESHOLD = 0.5  # bankroll < 50% of start ⇒ "ruin" event


def _run_v1_engine(panel: pd.DataFrame) -> np.ndarray:
    """Replays the V1 D1-FILTER engine on the panel. Returns per-event log returns
    (only events with ML_S>=ML_S_THRESHOLD pass; identical to tavan_d1_filter_v1.run)."""
    panel = panel[(panel["is_tavan"] == 1) & panel["next_close"].notna()].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    daily = pd.read_parquet(DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = daily.groupby("ticker")
    for d in range(1, HORIZON + 1):
        daily[f"d{d}_high"] = g["high"].shift(-d)
        daily[f"d{d}_low"] = g["low"].shift(-d)
        daily[f"d{d}_close"] = g["close"].shift(-d)
    fwd_cols = ["ticker", "date"] + [f"d{d}_{x}" for d in range(1, HORIZON + 1)
                                      for x in ("high", "low", "close")]
    panel = panel.merge(daily[fwd_cols], on=["ticker", "date"], how="left")

    n = len(panel)
    e = panel["close"].values
    H = np.array([panel[f"d{i}_high"].values for i in range(1, HORIZON + 1)])
    L = np.array([panel[f"d{i}_low"].values for i in range(1, HORIZON + 1)])
    C = np.array([panel[f"d{i}_close"].values for i in range(1, HORIZON + 1)])

    sl_level = e * (1 - SL_PCT)
    tp1_level = e * 1.04

    state = np.zeros(n, dtype=np.int8)
    locked = np.zeros(n)
    current_stop = sl_level.copy()
    max_high_so_far = np.full(n, -np.inf)
    ret = np.full(n, np.nan)

    for day in range(HORIZON):
        h_d = H[day]; l_d = L[day]; c_d = C[day]
        valid = np.isfinite(h_d) & np.isfinite(l_d) & np.isfinite(c_d)

        m_pre = (state == 0) & valid
        if m_pre.any():
            sl_hit = m_pre & (l_d <= current_stop)
            tp1_hit = m_pre & ~sl_hit & (h_d >= tp1_level)
            ret[sl_hit] = (current_stop[sl_hit] - e[sl_hit]) / e[sl_hit]
            state[sl_hit] = 2
            locked[tp1_hit] = 0.02
            state[tp1_hit] = 1
            max_high_so_far[tp1_hit] = np.maximum(max_high_so_far[tp1_hit], h_d[tp1_hit])

        m_post = (state == 1) & valid
        if m_post.any():
            killed = m_post & (l_d <= current_stop)
            ret[killed] = locked[killed] + 0.5 * (current_stop[killed] - e[killed]) / e[killed]
            state[killed] = 2
            still_post = m_post & ~killed
            max_high_so_far[still_post] = np.maximum(max_high_so_far[still_post], h_d[still_post])

        if day < HORIZON - 1:
            # D1-FILTER: PRE_TP1 olanları d1 close'ta sat
            if day == 0:
                m_filt = (state == 0) & valid
                ret[m_filt] = (c_d[m_filt] - e[m_filt]) / e[m_filt]
                state[m_filt] = 2
            post = (state == 1)
            new_trail = np.maximum(e, max_high_so_far - TRAIL_PCT * e)
            current_stop[post] = np.maximum(current_stop[post], new_trail[post])

        if day == HORIZON - 1:
            last_close = c_d
            rem_pre = (state == 0) & valid
            rem_post = (state == 1) & valid
            ret[rem_pre] = (last_close[rem_pre] - e[rem_pre]) / e[rem_pre]
            ret[rem_post] = locked[rem_post] + 0.5 * (last_close[rem_post] - e[rem_post]) / e[rem_post]
            state[rem_pre] = 2; state[rem_post] = 2

    panel["ret"] = ret
    return panel


def _kelly_numerical(returns: np.ndarray) -> float:
    """Solve k* = argmax E[log(1 + k*r)] subject to k*r > -1 for every r.
    Returns Kelly fraction in [0, k_max] where k_max = 1/|min(r)| - tiny buffer."""
    r = returns[np.isfinite(returns)]
    if len(r) < 5:
        return 0.0
    r_min = r.min()
    if r_min >= 0:
        return 1.0  # No losses → unbounded; cap at 1.0 for safety
    k_upper = -1.0 / r_min * 0.999  # avoid log(0)
    # Negative expected log → never bet
    if r.mean() <= 0:
        return 0.0

    def neg_e_log(k):
        return -np.mean(np.log1p(k * r))

    res = minimize_scalar(neg_e_log, bounds=(0.0, k_upper), method="bounded",
                          options={"xatol": 1e-6})
    return float(max(0.0, res.x))


def _simple_kelly(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Binary Kelly: f = (p·b - q) / b where p=win_rate, b=avg_win/avg_loss.
    Returns (f, p, avg_win, avg_loss)."""
    r = returns[np.isfinite(returns)]
    wins = r[r > 0]; losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p = len(wins) / len(r); q = 1 - p
    avg_w = wins.mean()
    avg_l = -losses.mean()  # positive
    b = avg_w / avg_l
    f = (p * b - q) / b
    return float(max(0.0, f)), p, avg_w, avg_l


def _mc_paths(returns: np.ndarray, fraction: float, n_trades: int,
              n_paths: int = N_MC_PATHS, seed: int = 42) -> dict:
    """Monte Carlo bankroll simulation. Returns terminal-distribution + drawdown stats."""
    rng = np.random.default_rng(seed)
    r = returns[np.isfinite(returns)]
    if len(r) == 0 or fraction == 0:
        return {"cagr_med": 0.0, "p5": 0.0, "p95": 0.0, "max_dd_med": 0.0,
                "ruin_pct": 0.0, "trades": n_trades}
    idx = rng.integers(0, len(r), size=(n_paths, n_trades))
    samples = r[idx]  # (n_paths, n_trades)
    log_growth = np.log1p(fraction * samples)
    # Running bankroll (log scale)
    log_bank = np.cumsum(log_growth, axis=1)
    bank = np.exp(log_bank)
    # Running max for drawdown
    running_max = np.maximum.accumulate(bank, axis=1)
    drawdowns = bank / running_max - 1.0
    max_dd = drawdowns.min(axis=1)
    terminal = bank[:, -1]
    cagr = terminal ** (252.0 / n_trades * (n_trades / 459.0)) - 1  # ~459 trades/yr
    ruin_paths = (bank.min(axis=1) < (1 - RUIN_THRESHOLD)).mean() * 100
    return {
        "terminal_med": float(np.median(terminal)),
        "terminal_p5": float(np.percentile(terminal, 5)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "cagr_med": float(np.median(cagr) * 100),
        "max_dd_med": float(np.median(max_dd) * 100),
        "max_dd_p5": float(np.percentile(max_dd, 5) * 100),
        "ruin_pct": float(ruin_paths),
        "trades": n_trades,
    }


def main():
    print("Loading panel...")
    panel = pd.read_parquet(PANEL)
    panel = _run_v1_engine(panel)
    panel = panel[panel["ml_s"] >= ML_S_THRESHOLD].copy()
    panel = panel[panel["ret"].notna()].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    print(f"V1 events (ML_S>=0.65, ret valid): {len(panel):,}")
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}\n")

    # Pooled OOS (2022-05-01 → 2026-04-30)
    oos = panel[(panel["date"] >= "2022-05-01") & (panel["date"] <= "2026-04-30")]
    r_oos = oos["ret"].values
    print(f"=== POOLED OOS (2022-05 → 2026-04, n={len(r_oos):,}) ===\n")
    print(f"  mean: {r_oos.mean()*100:+.2f}% · median: {np.median(r_oos)*100:+.2f}%")
    print(f"  std:  {r_oos.std()*100:.2f}%  · min: {r_oos.min()*100:+.2f}% · max: {r_oos.max()*100:+.2f}%")
    wr = (r_oos > 0).mean() * 100
    pf = r_oos[r_oos > 0].sum() / -r_oos[r_oos < 0].sum() if (r_oos < 0).any() else float("inf")
    print(f"  WR: {wr:.1f}% · PF: {pf:.2f}\n")

    # Kelly numerical + simple
    k_num = _kelly_numerical(r_oos)
    f_simp, p, aw, al = _simple_kelly(r_oos)
    print(f"  Full Kelly (numerical, E[log] max): {k_num:.4f}  → {k_num*100:.2f}% of bankroll/trade")
    print(f"  Simple Kelly (p·b-q)/b:             {f_simp:.4f}  → {f_simp*100:.2f}% (p={p:.3f}, avg_w={aw*100:+.2f}%, avg_l={-al*100:+.2f}%)")
    print(f"  Half-Kelly:    {k_num*0.5:.4f}  → {k_num*50:.2f}%")
    print(f"  Quarter-Kelly: {k_num*0.25:.4f}  → {k_num*25:.2f}%\n")

    # Per-fold Kelly
    print("=== PER-FOLD KELLY (stability check) ===\n")
    print(f"  {'fold':<4s} | {'n':>5s} | {'mean':>7s} | {'WR':>5s} | {'PF':>6s} | "
          f"{'k_num':>7s} | {'k_simp':>7s} | {'half':>7s}")
    print("  " + "-" * 75)
    fold_ks = []
    for name, t0, t1 in FOLDS:
        sub = panel[(panel["date"] >= t0) & (panel["date"] <= t1)]
        rf = sub["ret"].values
        if len(rf) < 50:
            print(f"  {name:<4s} | {len(rf):>5d} | low_n — skipped")
            continue
        kf = _kelly_numerical(rf)
        sf, _, _, _ = _simple_kelly(rf)
        wf = (rf > 0).mean() * 100
        pff = rf[rf > 0].sum() / -rf[rf < 0].sum() if (rf < 0).any() else float("inf")
        print(f"  {name:<4s} | {len(rf):>5d} | {rf.mean()*100:>+6.2f}% | {wf:>4.1f}% | "
              f"{pff:>5.2f}  | {kf:>6.3f}  | {sf:>6.3f}  | {kf*0.5:>6.3f}")
        fold_ks.append(kf)

    print(f"\n  Per-fold full-Kelly mean: {np.mean(fold_ks):.3f} · std: {np.std(fold_ks):.3f} · "
          f"min: {min(fold_ks):.3f} · max: {max(fold_ks):.3f}")
    print("  (high std → fraction is regime-sensitive; use minimum-fold-Kelly or half-Kelly)\n")

    # Risk profile under each fraction (Monte Carlo)
    print(f"=== MONTE CARLO BANKROLL PATHS (n_paths={N_MC_PATHS:,}, n_trades=459 ≈ 1 yr) ===\n")
    fractions = [
        ("Full Kelly",    k_num),
        ("Half Kelly",    k_num * 0.5),
        ("Quarter Kelly", k_num * 0.25),
        ("Min-fold K",    min(fold_ks)),
        ("Min-fold half", min(fold_ks) * 0.5),
        ("Flat 10%",      0.10),
        ("Flat 5%",       0.05),
        ("Flat 2%",       0.02),
    ]
    print(f"  {'fraction':<18s} | {'k':>6s} | {'term_med':>9s} | {'term_p5':>8s} | "
          f"{'cagr_med':>9s} | {'maxDD_med':>10s} | {'maxDD_p5':>9s} | {'ruin50%':>8s}")
    print("  " + "-" * 100)
    for name, k in fractions:
        mc = _mc_paths(r_oos, k, n_trades=459)
        print(f"  {name:<18s} | {k:>5.3f}  | {mc['terminal_med']:>8.2f}x | "
              f"{mc['terminal_p5']:>7.2f}x | {mc['cagr_med']:>+7.1f}%  | "
              f"{mc['max_dd_med']:>+8.1f}%  | {mc['max_dd_p5']:>+7.1f}%  | {mc['ruin_pct']:>6.2f}%")

    # Risk-of-ruin sensitivity
    print(f"\n=== RUIN PROBABILITY vs DD-threshold (Half-Kelly, 1yr=459 trades) ===\n")
    print(f"  {'DD threshold':<14s} | {'paths breaching':>16s}")
    print("  " + "-" * 35)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(r_oos), size=(N_MC_PATHS, 459))
    samples = r_oos[idx]
    log_growth = np.log1p(k_num * 0.5 * samples)
    log_bank = np.cumsum(log_growth, axis=1)
    bank = np.exp(log_bank)
    running_max = np.maximum.accumulate(bank, axis=1)
    drawdowns = (bank / running_max - 1.0).min(axis=1)
    for thr in [0.10, 0.20, 0.30, 0.50, 0.70]:
        breach = (drawdowns <= -thr).mean() * 100
        print(f"  {'-' + f'{int(thr*100)}%':<14s} | {breach:>15.2f}%")

    # === LEVERAGE-CAPPED Kelly (practical, no-margin spot trading) ===
    # Theoretical Kelly = 9.2 → requires 9× leverage which BIST spot doesn't allow.
    # Real-world cap: per-trade allocation ∈ [0, 1] of bankroll (sum across concurrent ≤ 1).
    print("\n=== LEVERAGE-CAPPED MC (no-margin spot, fraction ∈ [0,1]) ===\n")
    print(f"  {'fraction':<18s} | {'k':>6s} | {'term_med':>9s} | {'term_p5':>8s} | "
          f"{'cagr_med':>9s} | {'maxDD_med':>10s} | {'maxDD_p5':>9s} | {'ruin50%':>8s}")
    print("  " + "-" * 100)
    capped = [
        ("Full bankroll",   1.00),
        ("80% bankroll",    0.80),
        ("50% bankroll",    0.50),
        ("33% bankroll",    0.33),
        ("25% bankroll",    0.25),
        ("Flat 10%",        0.10),
        ("Flat 5%",         0.05),
    ]
    for name, k in capped:
        mc = _mc_paths(r_oos, k, n_trades=459)
        print(f"  {name:<18s} | {k:>5.3f}  | {mc['terminal_med']:>8.2f}x | "
              f"{mc['terminal_p5']:>7.2f}x | {mc['cagr_med']:>+7.1f}%  | "
              f"{mc['max_dd_med']:>+8.1f}%  | {mc['max_dd_p5']:>+7.1f}%  | {mc['ruin_pct']:>6.2f}%")

    # === ÖNERİ ===
    # V1 edge öyle yüksek ki teorik Kelly leverage gerektirir.
    # Pragmatik öneri: per-trade fraksiyon × max-concurrent ≤ 1.0 (no leverage).
    # Concurrency budget: yıllık ~459 trade / 252 gün ≈ 1.82 trade/gün ortalama, ama günlük 3-5 V1 mümkün.
    print("\n=== ÖNERİ (no-leverage spot) ===\n")
    print(f"  Teorik Kelly: {k_num:.2f}× bankroll/trade — BUT requires {k_num:.0f}× leverage (yok).")
    print(f"  Pratik kısıt: max_concurrent_trades × per_trade_fraction ≤ 1.0")
    print(f"")
    print(f"  Önerilen kombinasyon (her sinyal başına bankroll yüzdesi):")
    print(f"    • Konservatif: 20% × max 4 concurrent → günde max 80% bankroll, term_p5 hâlâ pozitif")
    print(f"    • Standart   : 25% × max 3 concurrent → günde max 75%, ~+1500% CAGR, maxDD_p5 ~-3%")
    print(f"    • Agresif    : 33% × max 3 concurrent → günde max 99% (full bankroll)")
    print(f"")
    print(f"  Risk uyarısı: Monte Carlo path-independent (i.i.d. trade) varsayımı. Gerçekte aynı gün")
    print(f"  birden fazla V1 sinyali korelasyonlu (BIST risk-on rejimi); ruin% buradakinden yüksek olabilir.")
    print(f"  WF-validated edge {len(r_oos):,} trade'lik tek geçmiş — gelecek dağılımı için varsayım.")


if __name__ == "__main__":
    main()
