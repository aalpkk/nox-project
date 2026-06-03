"""Tavan V1 — Intraday Early-Lock Predictor v0 (FEASIBILITY).

Pre-spec: memory/tavan_v1_intraday_lock_predictor_v0_prespec.md (LOCKED 2026-06-03).

Question: at snapshot T (~11:00/13:00/15:00) using ONLY pre-T 1h info, can we
predict end-of-day tavan lock?
  Label A = locks tavan@close today          (panel is_tavan==1, validated flag)
  Label B = Label A AND ml_s>=0.65           (V1-quality subset; ml_s = LABEL only)

Design (see pre-spec):
  - Snapshots = cumulative bars hr<=10 / <=12 / <=14 (lookahead-safe, end-of-bar).
  - Candidate pool at T = in-play & still fillable: ret_at_T in [+3%, +8%],
    not-yet-locked (max_ret_T < +9%). Names already >=+9% at T excluded (already
    locked/unfillable — catching them earlier is the point).
  - Features (pre-T, NO ml_s): ret_at_T, max_ret_T, dist_to_tavan, vol_ratio_20
    (vol-so-far vs trailing-20d same-window), first_5pct_hr, last_bar_accel,
    prior tavan_streak.
  - Separability: GOOD(lock) vs WITHIN-pool FADED. Random-same-size null = filter
    quality. TIME split train<=2024-12 / holdout 2025+.
  - Return re-measure: entry at T price, exit same-day close & next-day close;
    vs +8-11% T0-close baseline net of higher entry.

Outputs:
  output/tavan_intraday_lock_predictor_v0_events.parquet  (per candidate-day-snapshot)
  output/tavan_intraday_lock_predictor_v0.md              (human report)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INTRA = REPO / "output/extfeed_intraday_1h_3y_master.parquet"
PANEL = REPO / "output/tavan_full_backtest_v1_panel.parquet"
OUT_EVENTS = REPO / "output/tavan_intraday_lock_predictor_v0_events.parquet"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v0.md"

SNAPSHOTS = {"~11:00": 10, "~13:00": 12, "~15:00": 14}   # cum bars hr<=H
POOL_LO, POOL_HI = 0.03, 0.08      # in-play fillable band at T
LOCKED_AT_T = 0.09                 # >= this at T = already locked (excluded)
ML_S_THR = 0.65
TIME_SPLIT = pd.Timestamp("2025-01-01")
# NOTE: panel tavan_streak is SAME-DAY-INCLUSIVE → leaks the label (AUC=1.0).
# Use streak_prior (shift 1 per ticker) = streak coming INTO today (lookahead-safe).
FEATURES = ["ret_at_T", "max_ret_T", "dist_to_tavan", "vol_ratio_20",
            "first_5pct_hr", "last_bar_accel", "streak_prior"]
RNG_SEED = 12345


def log(m): print(m, flush=True)


def load_intraday() -> pd.DataFrame:
    m = pd.read_parquet(INTRA, columns=["ticker", "ts_istanbul", "open", "high",
                                        "low", "close", "volume"])
    m["ts"] = pd.to_datetime(m["ts_istanbul"])
    m["date"] = m["ts"].dt.normalize().dt.tz_localize(None)
    m["hr"] = m["ts"].dt.hour
    return m.sort_values(["ticker", "date", "hr"]).reset_index(drop=True)


def daily_frame(m: pd.DataFrame) -> pd.DataFrame:
    d = (m.groupby(["ticker", "date"])
           .agg(d_open=("open", "first"), d_high=("high", "max"),
                d_low=("low", "min"), d_close=("close", "last"))
           .reset_index().sort_values(["ticker", "date"]))
    d["prev_close"] = d.groupby("ticker")["d_close"].shift(1)
    d["next_close"] = d.groupby("ticker")["d_close"].shift(-1)
    return d


def snapshot_features(m: pd.DataFrame, H: int) -> pd.DataFrame:
    """Cumulative pre-T features from bars with hr<=H (lookahead-safe)."""
    sub = m[m["hr"] <= H]
    g = sub.groupby(["ticker", "date"])
    feat = g.agg(cum_close=("close", "last"),
                 cum_high=("high", "max"),
                 vol_so_far=("volume", "sum")).reset_index()
    # last-bar intrabar momentum at/just-before T
    last_bar = g.tail(1)[["ticker", "date", "open", "close"]].rename(
        columns={"open": "lb_open", "close": "lb_close"})
    feat = feat.merge(last_bar, on=["ticker", "date"], how="left")
    feat["last_bar_accel"] = (feat["lb_close"] - feat["lb_open"]) / feat["lb_open"].replace(0, np.nan)
    return feat


def first_cross_hr(m: pd.DataFrame, H: int, prev_close: pd.DataFrame,
                   mult: float) -> pd.DataFrame:
    """Hour of first bar (hr<=H) whose high>=prev_close*mult; NaN if none yet."""
    sub = m[m["hr"] <= H].merge(prev_close, on=["ticker", "date"], how="left")
    hit = sub[sub["high"] >= sub["prev_close"] * mult]
    fh = hit.groupby(["ticker", "date"])["hr"].min().rename("first_5pct_hr").reset_index()
    return fh


def build_snapshot(m, d, panel_lab, H, label):
    pc = d[["ticker", "date", "prev_close"]]
    feat = snapshot_features(m, H)
    feat = feat.merge(pc, on=["ticker", "date"], how="left")
    feat = feat[(feat["prev_close"] > 0) & feat["prev_close"].notna()].copy()

    feat["ret_at_T"] = (feat["cum_close"] - feat["prev_close"]) / feat["prev_close"]
    feat["max_ret_T"] = (feat["cum_high"] - feat["prev_close"]) / feat["prev_close"]
    feat["tavan_target"] = feat["prev_close"] * 1.10
    feat["dist_to_tavan"] = (feat["tavan_target"] - feat["cum_close"]) / feat["prev_close"]

    fh = first_cross_hr(m, H, pc, 1.05)
    feat = feat.merge(fh, on=["ticker", "date"], how="left")
    feat["first_5pct_hr"] = feat["first_5pct_hr"].fillna(99)  # not-yet-crossed sentinel

    # trailing-20d same-window volume ratio
    feat = feat.sort_values(["ticker", "date"])
    feat["vol20"] = (feat.groupby("ticker")["vol_so_far"]
                         .transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1)))
    feat["vol_ratio_20"] = feat["vol_so_far"] / feat["vol20"].replace(0, np.nan)

    # labels + daily outcome + streak
    feat = feat.merge(panel_lab, on=["ticker", "date"], how="left")
    feat = feat.merge(d[["ticker", "date", "d_close", "next_close"]],
                      on=["ticker", "date"], how="left")
    feat = feat[feat["is_tavan"].notna()].copy()            # labeled window only
    feat["lock"] = (feat["is_tavan"] == 1).astype(int)
    feat["v1_lock"] = ((feat["is_tavan"] == 1) & (feat["ml_s"] >= ML_S_THR)).astype(int)

    # candidate pool: in-play, fillable, not-yet-locked at T
    pool = feat[(feat["ret_at_T"] >= POOL_LO) & (feat["ret_at_T"] <= POOL_HI)
                & (feat["max_ret_T"] < LOCKED_AT_T)].copy()
    pool["snapshot"] = label
    pool["H"] = H
    # entry-at-T return re-measure
    pool["entry"] = pool["cum_close"]
    pool["ret_sameday"] = (pool["d_close"] - pool["entry"]) / pool["entry"]
    pool["ret_nextday"] = (pool["next_close"] - pool["entry"]) / pool["entry"]
    return pool


def null_p95(pool, target, n_pick, n_iter=2000):
    """Random same-size subsample lock-rate p95 (filter-quality null)."""
    y = pool[target].values
    base = y.mean()
    if n_pick < 1 or n_pick > len(y):
        return base, np.nan
    rng = np.random.default_rng(RNG_SEED)
    rates = np.array([y[rng.choice(len(y), n_pick, replace=False)].mean()
                      for _ in range(n_iter)])
    return base, np.percentile(rates, 95)


def fit_eval(seg, target, top_frac=0.10):
    """Train multi-feature logistic on TRAIN, evaluate top-frac picks on HOLDOUT.
    Returns holdout precision (lock-rate of picks), lift vs base, null p95, and
    realized entry@T next-day return of the picks (the actual strategy payoff)."""
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["date"] < TIME_SPLIT].copy()
    ho = seg[seg["date"] >= TIME_SPLIT].copy()
    if tr[target].sum() < 20 or len(ho) < 200 or ho[target].sum() < 5:
        return None
    Xtr = tr[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xho = ho[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit((Xtr - mu) / sd, tr[target].values)
    ho = ho.assign(score=clf.predict_proba((Xho - mu) / sd)[:, 1])
    auc = single_feat_auc(ho[target].values, ho["score"].values)
    base = ho[target].mean()
    coefs = dict(zip(FEATURES, np.round(clf.coef_[0], 3)))
    # selection-tightness sweep: does precision cross profitability as we tighten?
    sweep = []
    for frac in (0.01, 0.03, 0.05, 0.10):
        k = max(int(len(ho) * frac), 10)
        picks = ho.nlargest(k, "score")
        prec = picks[target].mean()
        _, p95 = null_p95(ho, target, k)
        sweep.append(dict(frac=frac, k=k, precision=prec,
                          lift=prec / base if base > 0 else np.nan,
                          beats_null=bool(prec > p95),
                          basket_nextday=picks["ret_nextday"].mean(),
                          basket_sameday=picks["ret_sameday"].mean()))
    return dict(n_ho=len(ho), base=base, holdout_auc=auc, coefs=coefs, sweep=sweep)


def single_feat_auc(y, x):
    """Rank-AUC of feature x for binary y (no sklearn dep)."""
    x = pd.Series(x).astype(float)
    m = x.notna() & pd.Series(y).notna().values
    x, yy = x[m].values, np.asarray(y)[m.values]
    n1, n0 = yy.sum(), (1 - yy).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().values
    auc = (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc


def topq_lift(seg, target, score):
    """Top-quartile-by-score lock rate vs pool base; + null p95."""
    s = seg.dropna(subset=[score])
    if len(s) < 40:
        return None
    thr = s[score].quantile(0.75)
    top = s[s[score] >= thr]
    base = s[target].mean()
    top_rate = top[target].mean()
    _, p95 = null_p95(s, target, len(top))
    return dict(n_pool=len(s), base=base, n_top=len(top), top_rate=top_rate,
                lift=top_rate / base if base > 0 else np.nan, null_p95=p95,
                beats_null=bool(top_rate > p95) if np.isfinite(p95) else None)


def main():
    log("loading intraday 1h master...")
    m = load_intraday()
    log(f"  rows={len(m):,} tickers={m.ticker.nunique()} "
        f"dates {m.date.min().date()}..{m.date.max().date()}")
    d = daily_frame(m)
    panel = pd.read_parquet(PANEL, columns=["date", "ticker", "is_tavan", "ml_s",
                                            "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["ticker", "date"])
    # lookahead-safe: streak coming INTO today = yesterday's (inclusive) streak
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel_lab = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior"]]

    pools = []
    for label, H in SNAPSHOTS.items():
        log(f"building snapshot {label} (hr<={H})...")
        pools.append(build_snapshot(m, d, panel_lab, H, label))
    allp = pd.concat(pools, ignore_index=True)
    allp.to_parquet(OUT_EVENTS, index=False)
    log(f"  pool events written: {len(allp):,} -> {OUT_EVENTS.name}")

    # ---- report ----
    lines = []
    A = lines.append
    A("# Tavan V1 — Intraday Early-Lock Predictor v0 (FEASIBILITY)\n")
    A(f"Pre-spec LOCKED. Snapshots {list(SNAPSHOTS)}. Pool band ret_at_T "
      f"[{POOL_LO:.0%},{POOL_HI:.0%}], excl already-locked (>={LOCKED_AT_T:.0%} at T).")
    A(f"Labeled window (intraday ∩ panel): {allp.date.min().date()}..{allp.date.max().date()}. "
      f"Time split holdout >= {TIME_SPLIT.date()}.\n")

    # data-granularity caveat: intra-bar open->+10% locks (never in fillable pool)
    locks_all = allp[allp["snapshot"] == "~11:00"]  # any snapshot pool
    A("## 0. Coverage / granularity")
    for label in SNAPSHOTS:
        seg = allp[allp["snapshot"] == label]
        A(f"- {label}: pool n={len(seg):,} | lock base={seg['lock'].mean():.1%} "
          f"| v1_lock base={seg['v1_lock'].mean():.2%}")
    A("")

    for target in ["lock", "v1_lock"]:
        A(f"## Target = {target}  (logistic on {FEATURES}, top-10% holdout picks)")
        for label in SNAPSHOTS:
            seg = allp[allp["snapshot"] == label].copy()
            tr = seg[seg["date"] < TIME_SPLIT]
            ho = seg[seg["date"] >= TIME_SPLIT]
            A(f"\n### {label}  (pool n={len(seg):,}; train {len(tr):,} / holdout {len(ho):,})")
            aucs = {f: single_feat_auc(tr[target].values, tr[f].values) for f in FEATURES}
            A("- single-feat AUC (train): "
              + ", ".join(f"{f}={a:.3f}" for f, a in sorted(aucs.items(),
                          key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else 0))))
            r = fit_eval(seg, target)
            if r is None:
                A("- predictor: insufficient positives/holdout")
            else:
                A(f"- HOLDOUT base={r['base']:.2%} | model AUC={r['holdout_auc']:.3f}")
                A("- selection sweep (holdout): "
                  "frac | n | precision | lift | beats_null | basket next-day | same-day")
                for s in r["sweep"]:
                    A(f"    - top-{s['frac']:.0%}: n={s['k']:>4} | prec={s['precision']:.1%} "
                      f"| {s['lift']:.1f}x | {s['beats_null']} | "
                      f"basket_next={s['basket_nextday']:+.2%} | same={s['basket_sameday']:+.2%}")
                A(f"- coefs: {r['coefs']}")
            lk = seg[seg[target] == 1]; fd = seg[seg[target] == 0]
            A(f"- ceiling (perfect pick) entry@T → next-day: GOOD mean={lk['ret_nextday'].mean():.2%} "
              f"(n={len(lk)}) vs FADED {fd['ret_nextday'].mean():.2%}")
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"report -> {OUT_MD}")
    log("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
