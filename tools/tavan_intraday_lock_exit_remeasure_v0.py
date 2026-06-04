"""Lever 2: re-measure intraday-lock predictor picks under the VALIDATED V1 exit
(SL-10% / TP1+4%→trail2% / H=25), instead of raw next-day close.

Reuses the exit engine logic from tools/tavan_walk_forward_v1.py verbatim
(SL_pct=0.10, trail_pct=0.02, horizon=25, d1_filter off, tp1_be off) but sets
entry price = intraday snapshot price (entry@T), not the tavan close.

Question: does a proper exit lift the holdout top-frac basket from ~breakeven
(raw next-close) into positive? Same logistic/split/sweep as the v0 predictor.

Inputs : output/tavan_intraday_lock_predictor_v0_events.parquet
         output/_cache_daily_ohlcv_3y.parquet
Output : output/tavan_intraday_lock_exit_remeasure_v0.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVENTS = REPO / "output/tavan_intraday_lock_predictor_v0_events.parquet"
DAILY = REPO / "output/_cache_daily_ohlcv_3y.parquet"
OUT_MD = REPO / "output/tavan_intraday_lock_exit_remeasure_v0.md"

SNAPSHOTS = ["~11:00", "~13:00", "~15:00"]
FEATURES = ["ret_at_T", "max_ret_T", "dist_to_tavan", "vol_ratio_20",
            "first_5pct_hr", "last_bar_accel", "streak_prior"]
TIME_SPLIT = pd.Timestamp("2025-01-01")
SL_PCT, TRAIL_PCT, HORIZON = 0.10, 0.02, 25


def log(m): print(m, flush=True)


def attach_forward(ev: pd.DataFrame) -> pd.DataFrame:
    """Merge d1..d25 forward daily high/low/close keyed on (ticker, event date)."""
    daily = pd.read_parquet(DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = daily.groupby("ticker")
    for d in range(1, HORIZON + 1):
        daily[f"d{d}_high"] = g["high"].shift(-d)
        daily[f"d{d}_low"] = g["low"].shift(-d)
        daily[f"d{d}_close"] = g["close"].shift(-d)
    fcols = ["ticker", "date"] + [f"d{d}_{x}" for d in range(1, HORIZON + 1)
                                  for x in ("high", "low", "close")]
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    return ev.merge(daily[fcols], on=["ticker", "date"], how="left")


def v1_exit_return(ev: pd.DataFrame) -> np.ndarray:
    """VERBATIM walk_forward_v1 exit loop; entry e = intraday snapshot price."""
    n = len(ev)
    e = ev["entry"].values.astype(float)              # entry@T (NOT tavan close)
    H = np.array([ev[f"d{i}_high"].values for i in range(1, HORIZON + 1)], float)
    L = np.array([ev[f"d{i}_low"].values for i in range(1, HORIZON + 1)], float)
    C = np.array([ev[f"d{i}_close"].values for i in range(1, HORIZON + 1)], float)

    sl_level = e * (1 - SL_PCT)
    tp1_level = e * 1.04
    state = np.zeros(n, dtype=np.int8)
    locked = np.zeros(n)
    current_stop = sl_level.copy()
    max_high_so_far = np.full(n, -np.inf)
    ret = np.full(n, np.nan)

    for day in range(HORIZON):
        h_d, l_d, c_d = H[day], L[day], C[day]
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
            still = m_post & ~killed
            max_high_so_far[still] = np.maximum(max_high_so_far[still], h_d[still])
            new_trail = np.maximum(e, max_high_so_far - TRAIL_PCT * e)
            current_stop[still] = np.maximum(current_stop[still], new_trail[still])

    # survivors to horizon: exit at last valid close
    surv = (state != 2)
    last_close = np.full(n, np.nan)
    for day in range(HORIZON):
        cd = C[day]
        ok = np.isfinite(cd)
        last_close[ok] = cd[ok]
    ret[surv] = (last_close[surv] - e[surv]) / e[surv]
    return ret


def fit_holdout_picks(seg, target):
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["date"] < TIME_SPLIT].copy()
    ho = seg[seg["date"] >= TIME_SPLIT].copy()
    if tr[target].sum() < 20 or ho[target].sum() < 5:
        return None
    Xtr = tr[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xho = ho[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, tr[target].values)
    ho = ho.assign(score=clf.predict_proba((Xho - mu) / sd)[:, 1])
    return ho


def main():
    ev = pd.read_parquet(EVENTS)
    ev["entry"] = ev["cum_close"]
    log(f"events: {len(ev):,}")
    ev = attach_forward(ev)
    log("computing V1-exit return per event (entry@T)...")
    ev["ret_v1exit"] = v1_exit_return(ev)

    lines = []; A = lines.append
    A("# Lever 2 — Intraday-Lock Picks under VALIDATED V1 Exit (1h data)\n")
    A(f"Exit: SL−{SL_PCT:.0%} / TP1+4%→trail{TRAIL_PCT:.0%} / H={HORIZON} "
      "(walk_forward_v1 engine, verbatim). Entry = intraday snapshot price.")
    A(f"Holdout ≥ {TIME_SPLIT.date()}. Compares basket: raw next-close vs V1-exit.\n")

    for target in ["lock", "v1_lock"]:
        A(f"## Target = {target}")
        for snap in SNAPSHOTS:
            seg = ev[ev["snapshot"] == snap].copy()
            ho = fit_holdout_picks(seg, target)
            if ho is None:
                A(f"\n### {snap}: insufficient"); continue
            base_next = ho["ret_nextday"].mean()
            base_exit = ho["ret_v1exit"].mean()
            A(f"\n### {snap}  (holdout n={len(ho):,}, base rate {ho[target].mean():.2%})")
            A("- frac | n | precision | basket next-close | **basket V1-exit** | "
              "GOOD-only V1-exit | FADED-only V1-exit")
            for frac in (0.01, 0.03, 0.05, 0.10):
                k = max(int(len(ho) * frac), 10)
                p = ho.nlargest(k, "score")
                good = p[p[target] == 1]; fad = p[p[target] == 0]
                A(f"  - top-{frac:.0%} | n={k:>4} | prec={p[target].mean():.1%} | "
                  f"next={p['ret_nextday'].mean():+.2%} | "
                  f"**exit={p['ret_v1exit'].mean():+.2%}** | "
                  f"good={good['ret_v1exit'].mean():+.2%} | "
                  f"faded={fad['ret_v1exit'].mean():+.2%}")
            # all-holdout reference
            A(f"- (ref) whole holdout pool: next-close {base_next:+.2%} | "
              f"V1-exit {base_exit:+.2%}")
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines))
    log(f"\nreport -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
