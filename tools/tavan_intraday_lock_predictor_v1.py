"""Tavan intraday early-lock predictor v1 — ENRICHED FEATURES + lightgbm.

Builds on v0 (memory/tavan_v1_intraday_lock_predictor_v0_outcome.md). Goal: raise
PRECISION of finding v1_lock (tavan@close AND ml_s>=0.65) at 11/13/15:00, the
validated super-trade. Same lookahead discipline; ml_s is LABEL only.

NEW features (all pre-T, lookahead-safe, 1h-derivable):
  breadth_5pct   — cross-sectional: fraction of universe up >=+5% at T (strong tape)
  vol_accel      — last-bar volume / mean earlier-bar volume that day
  velocity_2bar  — ret_at_T minus ret two bars earlier (accelerating toward tavan?)
  pullback_hi    — max_ret_T - ret_at_T (small = holding near intraday high)
  open_gap       — (day open - prev_close)/prev_close
  dist_20dhigh   — (cum_close - prior-20d high)/prior-20d high
  tavan_freq_20  — # tavan in prior 20 days (lookahead-safe, shifted)
  + v0 set (ret_at_T, max_ret_T, dist_to_tavan, vol_ratio_20, first_5pct_hr,
            last_bar_accel, streak_prior)

Reports: logistic vs lightgbm, OLD vs NEW features, precision@top-frac on a
TEST split untouched for model/feature choice, plus V1-exit basket.

Out: output/tavan_intraday_lock_predictor_v1.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INTRA = REPO / "output/extfeed_intraday_1h_3y_master.parquet"
PANEL = REPO / "output/tavan_full_backtest_v1_panel.parquet"
DAILY = REPO / "output/_cache_daily_ohlcv_3y.parquet"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v1.md"

SNAPSHOTS = {"~11:00": 10, "~13:00": 12, "~15:00": 14}
TIME_SPLIT = pd.Timestamp("2025-01-01")
SL_PCT, TRAIL_PCT, HORIZON = 0.10, 0.02, 25
OLD = ["ret_at_T", "max_ret_T", "dist_to_tavan", "vol_ratio_20",
       "first_5pct_hr", "last_bar_accel", "streak_prior"]
NEW = OLD + ["breadth_5pct", "vol_accel", "velocity_2bar", "pullback_hi",
             "open_gap", "dist_20dhigh", "tavan_freq_20"]


def log(m): print(m, flush=True)


def load_intraday():
    m = pd.read_parquet(INTRA, columns=["ticker", "ts_istanbul", "open", "high",
                                        "low", "close", "volume"])
    m["ts"] = pd.to_datetime(m["ts_istanbul"])
    m["date"] = m["ts"].dt.normalize().dt.tz_localize(None)
    m["hr"] = m["ts"].dt.hour
    return m.sort_values(["ticker", "date", "hr"]).reset_index(drop=True)


def daily_frame(m):
    d = (m.groupby(["ticker", "date"])
           .agg(d_open=("open", "first"), d_high=("high", "max"),
                d_low=("low", "min"), d_close=("close", "last"))
           .reset_index().sort_values(["ticker", "date"]))
    d["prev_close"] = d.groupby("ticker")["d_close"].shift(1)
    d["next_close"] = d.groupby("ticker")["d_close"].shift(-1)
    d["hi20_prior"] = d.groupby("ticker")["d_high"].transform(
        lambda s: s.rolling(20, min_periods=5).max().shift(1))
    g = d.sort_values(["ticker", "date"]).groupby("ticker")
    for k in range(1, HORIZON + 1):
        d[f"d{k}_high"] = g["d_high"].shift(-k)
        d[f"d{k}_low"] = g["d_low"].shift(-k)
        d[f"d{k}_close"] = g["d_close"].shift(-k)
    return d


def cum_at(m, H, col_close="close"):
    """Cumulative aggregates from bars hr<=H per (ticker,date)."""
    sub = m[m["hr"] <= H]
    g = sub.groupby(["ticker", "date"])
    out = g.agg(cum_close=("close", "last"), cum_high=("high", "max"),
                vol_so_far=("volume", "sum"), nbar=("close", "size")).reset_index()
    lastbar = g.tail(1)[["ticker", "date", "open", "close", "volume"]].rename(
        columns={"open": "lb_open", "close": "lb_close", "volume": "lb_vol"})
    out = out.merge(lastbar, on=["ticker", "date"], how="left")
    return out


def build(m, d, panel_lab, H, label):
    pc = d[["ticker", "date", "prev_close", "d_open", "hi20_prior", "d_close",
            "next_close"] + [f"d{k}_{x}" for k in range(1, HORIZON + 1)
                             for x in ("high", "low", "close")]]
    cur = cum_at(m, H)
    m2 = cum_at(m, max(H - 2, 9))[["ticker", "date", "cum_close"]].rename(
        columns={"cum_close": "cum_close_m2"})
    feat = cur.merge(m2, on=["ticker", "date"], how="left").merge(
        pc, on=["ticker", "date"], how="left")
    feat = feat[(feat["prev_close"] > 0) & feat["prev_close"].notna()].copy()

    feat["ret_at_T"] = (feat["cum_close"] - feat["prev_close"]) / feat["prev_close"]
    feat["ret_m2"] = (feat["cum_close_m2"] - feat["prev_close"]) / feat["prev_close"]
    feat["max_ret_T"] = (feat["cum_high"] - feat["prev_close"]) / feat["prev_close"]
    feat["tavan_target"] = feat["prev_close"] * 1.10
    feat["dist_to_tavan"] = (feat["tavan_target"] - feat["cum_close"]) / feat["prev_close"]
    feat["last_bar_accel"] = (feat["lb_close"] - feat["lb_open"]) / feat["lb_open"].replace(0, np.nan)
    feat["velocity_2bar"] = feat["ret_at_T"] - feat["ret_m2"]
    feat["pullback_hi"] = feat["max_ret_T"] - feat["ret_at_T"]
    feat["open_gap"] = (feat["d_open"] - feat["prev_close"]) / feat["prev_close"]
    feat["dist_20dhigh"] = (feat["cum_close"] - feat["hi20_prior"]) / feat["hi20_prior"].replace(0, np.nan)
    feat["vol_accel"] = feat["lb_vol"] / (feat["vol_so_far"] / feat["nbar"]).replace(0, np.nan)

    # first +5% hour (pre-T)
    sub = m[m["hr"] <= H].merge(pc[["ticker", "date", "prev_close"]], on=["ticker", "date"], how="left")
    fh = (sub[sub["high"] >= sub["prev_close"] * 1.05].groupby(["ticker", "date"])["hr"]
          .min().rename("first_5pct_hr").reset_index())
    feat = feat.merge(fh, on=["ticker", "date"], how="left")
    feat["first_5pct_hr"] = feat["first_5pct_hr"].fillna(99)

    # trailing-20d same-window volume ratio
    feat = feat.sort_values(["ticker", "date"])
    feat["vol20"] = feat.groupby("ticker")["vol_so_far"].transform(
        lambda s: s.rolling(20, min_periods=5).mean().shift(1))
    feat["vol_ratio_20"] = feat["vol_so_far"] / feat["vol20"].replace(0, np.nan)

    # cross-sectional breadth at T (strong-tape proxy) — over ALL tickers that day
    feat["breadth_5pct"] = feat.groupby("date")["ret_at_T"].transform(
        lambda s: (s >= 0.05).mean())

    # labels + streak_prior + tavan_freq_20
    feat = feat.merge(panel_lab, on=["ticker", "date"], how="left")
    feat = feat[feat["is_tavan"].notna()].copy()
    feat["lock"] = (feat["is_tavan"] == 1).astype(int)
    feat["v1_lock"] = ((feat["is_tavan"] == 1) & (feat["ml_s"] >= 0.65)).astype(int)

    pool = feat[(feat["ret_at_T"] >= 0.03) & (feat["ret_at_T"] <= 0.08)
                & (feat["max_ret_T"] < 0.09)].copy()
    pool["snapshot"] = label
    pool["entry"] = pool["cum_close"]
    pool["ret_nextday"] = (pool["next_close"] - pool["entry"]) / pool["entry"]
    pool["ret_v1exit"] = v1_exit(pool)
    return pool


def v1_exit(ev):
    n = len(ev)
    e = ev["entry"].values.astype(float)
    H = np.array([ev[f"d{i}_high"].values for i in range(1, HORIZON + 1)], float)
    L = np.array([ev[f"d{i}_low"].values for i in range(1, HORIZON + 1)], float)
    C = np.array([ev[f"d{i}_close"].values for i in range(1, HORIZON + 1)], float)
    sl = e * (1 - SL_PCT); tp1 = e * 1.04
    state = np.zeros(n, np.int8); locked = np.zeros(n)
    stop = sl.copy(); mx = np.full(n, -np.inf); ret = np.full(n, np.nan)
    for day in range(HORIZON):
        h, l, c = H[day], L[day], C[day]
        v = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        pre = (state == 0) & v
        if pre.any():
            slh = pre & (l <= stop); tph = pre & ~slh & (h >= tp1)
            ret[slh] = (stop[slh] - e[slh]) / e[slh]; state[slh] = 2
            locked[tph] = 0.02; state[tph] = 1
            mx[tph] = np.maximum(mx[tph], h[tph])
        post = (state == 1) & v
        if post.any():
            kil = post & (l <= stop)
            ret[kil] = locked[kil] + 0.5 * (stop[kil] - e[kil]) / e[kil]; state[kil] = 2
            still = post & ~kil
            mx[still] = np.maximum(mx[still], h[still])
            nt = np.maximum(e, mx - TRAIL_PCT * e); stop[still] = np.maximum(stop[still], nt[still])
    surv = state != 2
    last = np.full(n, np.nan)
    for day in range(HORIZON):
        ok = np.isfinite(C[day]); last[ok] = C[day][ok]
    ret[surv] = (last[surv] - e[surv]) / e[surv]
    return ret


def auc(y, s):
    y = np.asarray(y); s = pd.Series(s).astype(float)
    mm = s.notna().values
    y, s = y[mm], s[mm].values
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(seg, target, feats, model):
    tr = seg[seg["date"] < TIME_SPLIT].copy()
    te = seg[seg["date"] >= TIME_SPLIT].copy()
    if tr[target].sum() < 20 or te[target].sum() < 5: return None
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xte = te[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    if model == "logit":
        from sklearn.linear_model import LogisticRegression
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit((Xtr - mu) / sd, tr[target].values)
        te = te.assign(score=clf.predict_proba((Xte - mu) / sd)[:, 1])
    else:
        import lightgbm as lgb
        pos = tr[target].mean()
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                 num_leaves=31, min_child_samples=50,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=(1 - pos) / max(pos, 1e-6),
                                 verbosity=-1)
        clf.fit(Xtr, tr[target].values)
        te = te.assign(score=clf.predict_proba(Xte)[:, 1])
    rows = []
    for frac in (0.01, 0.03, 0.05):
        k = max(int(len(te) * frac), 10)
        p = te.nlargest(k, "score")
        rows.append((frac, k, p[target].mean(), p["ret_v1exit"].mean(), p["ret_nextday"].mean()))
    return dict(test_auc=auc(te[target].values, te["score"].values),
                base=te[target].mean(), sweep=rows)


def main():
    log("loading 1h master..."); m = load_intraday()
    d = daily_frame(m)
    panel = pd.read_parquet(PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    panel_lab = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]

    pools = []
    for label, H in SNAPSHOTS.items():
        log(f"building {label}..."); pools.append(build(m, d, panel_lab, H, label))
    allp = pd.concat(pools, ignore_index=True)

    lines = []; A = lines.append
    A("# Tavan Intraday Lock Predictor v1 — enriched features + lightgbm\n")
    A(f"Target **v1_lock** (primary). TEST holdout >= {TIME_SPLIT.date()} "
      "(untouched for model/feature choice). Basket = V1-exit (SL−10/TP1+4→trail2/H25).")
    A(f"NEW features: {[f for f in NEW if f not in OLD]}\n")

    for target in ["v1_lock", "lock"]:
        A(f"## Target = {target}")
        for label in SNAPSHOTS:
            seg = allp[allp["snapshot"] == label].copy()
            A(f"\n### {label}  (pool n={len(seg):,})")
            for mdl, fs, tag in [("logit", OLD, "logit/OLD"),
                                 ("logit", NEW, "logit/NEW"),
                                 ("lgbm", NEW, "lgbm/NEW")]:
                r = evaluate(seg, target, fs, mdl)
                if r is None: A(f"- {tag}: insufficient"); continue
                s1 = r["sweep"][0]  # top-1%
                A(f"- **{tag}**: testAUC={r['test_auc']:.3f} base={r['base']:.2%} | "
                  f"top-1% prec={s1[2]:.1%} (n={s1[1]}) exit={s1[3]:+.2%} | "
                  + " ".join(f"top-{int(f*100)}%={pr:.1%}/{ex:+.1%}"
                             for (f, k, pr, ex, nx) in r["sweep"][1:]))
        A("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
