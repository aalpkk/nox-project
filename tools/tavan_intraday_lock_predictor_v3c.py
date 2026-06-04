"""v3c — do the SCREENER's structural/context features raise precision?

The user's point: the tavan screener (ml.features.compute_all_features, ~46-feat
ml_s model) computes STRUCTURAL features that define a good tavan — RS, trend stack,
base/squeeze quality, distance from highs, ATR regime, liquidity rank. My intraday
predictor only had crude intraday-trajectory + streak/freq. Those structural features,
shifted to PRIOR close (known at T), are lookahead-safe and far more promising than
the bar-shape features (v3b, redundant).

This computes a focused structural set DIRECTLY from the daily cache (cheap; avoids
the 969-line compute_all_features), all as-of prior close (shift 1 per ticker), and
tests base NEW vs NEW+STRUCT on the same 3-fold (basket=ret_v1exit, sel@VAL-cov 1%).

LEAKAGE GUARD: features are prior-day; ml_s (label) uses TODAY's bar. Prior-day RS/
trend correlating with today's ml_s-quality lock is legitimate signal, not leakage.
Watch AUC — a jump to ~1.0 would flag a leak.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import tavan_intraday_lock_predictor_v1 as v1   # noqa: E402
import tavan_intraday_lock_predictor_v2 as v2   # noqa: E402

DAILY = REPO / "output/_cache_daily_ohlcv_3y.parquet"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v3c.md"
TARGET, RET = "v1_lock", "ret_v1exit"
STRUCT = ["st_ret20", "st_rs20_cs", "st_dist_20dhigh", "st_dist_252high",
          "st_sma50_dist", "st_sma200_dist", "st_atr_rank60", "st_squeeze",
          "st_liq_cs", "st_chc", "st_clp_prev", "prev_ret"]


def log(m): print(m, flush=True)


def build_struct():
    """Focused structural features, ALL as-of prior close (shift 1)."""
    d = pd.read_parquet(DAILY)[["ticker", "date", "open", "high", "low", "close", "volume"]]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = d.groupby("ticker")
    c = d["close"]
    # returns / momentum
    d["ret20"] = g["close"].transform(lambda s: s / s.shift(20) - 1)
    d["ret60"] = g["close"].transform(lambda s: s / s.shift(60) - 1)
    d["prev_ret"] = g["close"].transform(lambda s: s.shift(1) / s.shift(2) - 1)
    # distance from rolling highs
    d["hi20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=5).max())
    d["hi252"] = g["high"].transform(lambda s: s.rolling(252, min_periods=40).max())
    d["st_dist_20dhigh"] = (c - d["hi20"]) / d["hi20"]
    d["st_dist_252high"] = (c - d["hi252"]) / d["hi252"]
    # trend stack
    d["sma50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    d["sma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=40).mean())
    d["st_sma50_dist"] = (c - d["sma50"]) / d["sma50"]
    d["st_sma200_dist"] = (c - d["sma200"]) / d["sma200"]
    # ATR + regime
    tr = pd.concat([(d["high"] - d["low"]),
                    (d["high"] - g["close"].shift(1)).abs(),
                    (d["low"] - g["close"].shift(1)).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.groupby(d["ticker"]).transform(lambda s: s.rolling(14, min_periods=5).mean())
    d["atr_pct"] = d["atr14"] / c
    d["atr5"] = tr.groupby(d["ticker"]).transform(lambda s: s.rolling(5, min_periods=2).mean())
    d["st_squeeze"] = (d["atr5"] / c) / d["atr_pct"].replace(0, np.nan)   # <1 = contracting
    d["st_atr_rank60"] = d.groupby("ticker")["atr_pct"].transform(
        lambda s: s.rolling(60, min_periods=15).rank(pct=True))
    # liquidity + close position + consecutive higher closes
    d["tl20"] = g.apply(lambda x: (x["close"] * x["volume"]).rolling(20, min_periods=5).median()
                        ).reset_index(level=0, drop=True)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["clp"] = ((d["close"] - d["low"]) / rng).fillna(0.5)
    up = (d["close"] > g["close"].shift(1)).astype(int)
    d["chc"] = up.groupby(d["ticker"]).transform(
        lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1))
    d["st_ret20"] = d["ret20"]
    d["st_clp_prev"] = d["clp"]
    d["st_chc"] = d["chc"]
    # cross-sectional ranks per date (RS proxy + liquidity pctile) — uses same-day
    # cross-section of PAST returns, fine; then we shift the whole row by 1 below
    d["st_rs20_cs"] = d.groupby("date")["ret20"].rank(pct=True)
    d["st_liq_cs"] = d.groupby("date")["tl20"].rank(pct=True)
    # SHIFT every structural feature by 1 (as-of prior close = known at T today)
    cols = STRUCT
    for col in cols:
        d[col] = d.groupby("ticker")[col].shift(1)
    return d[["ticker", "date"] + cols]


def fit_eval(seg, feats, val_cov=0.01):
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["split"] == "train"]; val = seg[seg["split"] == "val"]; te = seg[seg["split"] == "test"]
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, tr[TARGET].values)
    sc = lambda part: clf.predict_proba(
        (part[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values - mu) / sd)[:, 1]
    vt = val.assign(s=sc(val)); tt = te.assign(s=sc(te))
    tau = np.quantile(vt["s"], 1 - val_cov)
    p = tt[tt["s"] >= tau]
    return dict(auc=v1.auc(te[TARGET].values, sc(te)), n=len(p),
                prec=p[TARGET].mean(), mean=p[RET].mean(), med=p[RET].median(),
                win=(p[RET] > 0).mean(), coef=dict(zip(feats, np.round(clf.coef_[0], 2))))


def main():
    log("loading..."); m = v1.load_intraday(); d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize(); panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    pl = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]
    log("building structural features..."); st = build_struct()

    lines = []; A = lines.append
    A("# v3c — screener structural features (prior-day, lookahead-safe) for precision\n")
    A(f"STRUCT (as-of prior close): {STRUCT}\n")
    A("basket=ret_v1exit, selection@VAL-cov 1%, 3-fold TRAIN/VAL/TEST.\n")

    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg.merge(st, on=["ticker", "date"], how="left")
        seg["split"] = np.where(seg["date"] < v2.TRAIN_END, "train",
                                np.where(seg["date"] < v2.VAL_END, "val", "test"))
        tr = seg[seg["split"] == "train"]
        A(f"## {label}  (test base={seg[seg.split=='test'][TARGET].mean():.2%})")
        sa = {f: v1.auc(tr[TARGET].values, tr[f].fillna(tr[f].median()).values) for f in STRUCT}
        A("- struct single-feat TRAIN AUC: " + ", ".join(
            f"{f}={a:.3f}" for f, a in sorted(sa.items(), key=lambda kv: -abs((kv[1] or .5) - .5))))
        base = fit_eval(seg, v1.NEW)
        full = fit_eval(seg, v1.NEW + STRUCT)
        A(f"- **base NEW**       : AUC={base['auc']:.3f} | n={base['n']} prec={base['prec']*100:.0f}% "
          f"mean={base['mean']*100:+.1f}% med={base['med']*100:+.1f}% win={base['win']*100:.0f}%")
        A(f"- **NEW+STRUCT**     : AUC={full['auc']:.3f} | n={full['n']} prec={full['prec']*100:.0f}% "
          f"mean={full['mean']*100:+.1f}% med={full['med']*100:+.1f}% win={full['win']*100:.0f}%")
        # struct-only model (is the structural context alone predictive?)
        so = fit_eval(seg, STRUCT)
        A(f"- STRUCT-only        : AUC={so['auc']:.3f} | n={so['n']} prec={so['prec']*100:.0f}% "
          f"mean={so['mean']*100:+.1f}%")
        top = sorted(full["coef"].items(), key=lambda kv: -abs(kv[1]))[:6]
        A(f"- top |coef| (NEW+STRUCT): {top}")
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
