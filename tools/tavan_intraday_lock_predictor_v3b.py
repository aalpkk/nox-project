"""v3b — hunt for EXTRA 1h-derivable features that raise precision (before 15m).

Adds 6 theory-motivated features, all strictly pre-T and 1h/daily-derivable:
  buy_pressure      mean((close-low)/(high-low)) over bars<=T — where each bar closes
                    (demand strength; lockers close bars at highs)
  intraday_mae      worst pullback from running intraday high over bars<=T
                    (clean march to tavan vs choppy; lower = cleaner)
  up_bar_frac       fraction of bars<=T closing above their open
  liq_tl_20         log trailing-20d median daily TL turnover (Audit#4: thin names
                    lock; liquidity should be NEGATIVELY related to lock)
  prev_ret          prior-day return (momentum context)
  sector_breadth    fraction of same-sektor_id tickers up>=5% at T (theme clustering)

Evaluation: 3-fold (TRAIN/VAL/TEST as v2/v3), basket=ret_v1exit, selection at
VAL-coverage 1%. Reports base NEW vs NEW+EXTRA on TEST (precision, mean, median),
each extra's single-feature TRAIN AUC, and a LOFO drop (which extras carry weight).
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

SECTOR = REPO / "output/bist_sector_map.csv"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v3b.md"
TARGET, RET = "v1_lock", "ret_v1exit"
EXTRA = ["buy_pressure", "intraday_mae", "up_bar_frac", "liq_tl_20", "prev_ret", "sector_breadth"]


def log(m): print(m, flush=True)


def extra_intraday(m, H):
    """Per (ticker,date) bar-shape features from bars hr<=H (pre-T)."""
    sub = m[m["hr"] <= H].sort_values(["ticker", "date", "hr"]).copy()
    rng = (sub["high"] - sub["low"]).replace(0, np.nan)
    sub["bp"] = ((sub["close"] - sub["low"]) / rng).fillna(0.5)
    sub["up"] = (sub["close"] > sub["open"]).astype(float)
    grp = sub.groupby(["ticker", "date"], sort=False)
    sub["run_high"] = grp["high"].cummax()
    sub["dd"] = (sub["low"] - sub["run_high"]) / sub["run_high"]
    agg = grp.agg(buy_pressure=("bp", "mean"), up_bar_frac=("up", "mean"),
                  intraday_mae=("dd", "min")).reset_index()
    return agg


def extra_daily(m):
    """Trailing daily liquidity + prev-day return (shifted, pre-T)."""
    dv = (m.groupby(["ticker", "date"])
            .agg(d_close=("close", "last"), d_vol=("volume", "sum")).reset_index()
            .sort_values(["ticker", "date"]))
    dv["tl"] = dv["d_close"] * dv["d_vol"]
    dv["liq_tl_20"] = np.log1p(dv.groupby("ticker")["tl"].transform(
        lambda s: s.rolling(20, min_periods=5).median().shift(1)))
    pc = dv.groupby("ticker")["d_close"].shift(1)
    ppc = dv.groupby("ticker")["d_close"].shift(2)
    dv["prev_ret"] = (pc - ppc) / ppc
    return dv[["ticker", "date", "liq_tl_20", "prev_ret"]]


def sector_breadth(m, d, H, sec):
    """Frac of same-sektor tickers up>=5% at T (full universe, not just pool)."""
    cur = v1.cum_at(m, H)[["ticker", "date", "cum_close"]]
    cur = cur.merge(d[["ticker", "date", "prev_close"]], on=["ticker", "date"], how="left")
    cur = cur[(cur["prev_close"] > 0) & cur["prev_close"].notna()].copy()
    cur["ret_at_T"] = (cur["cum_close"] - cur["prev_close"]) / cur["prev_close"]
    cur = cur.merge(sec, on="ticker", how="left")
    cur["is5"] = (cur["ret_at_T"] >= 0.05).astype(float)
    sb = (cur.groupby(["date", "sektor_id"])["is5"].mean()
          .rename("sector_breadth").reset_index())
    cur = cur.merge(sb, on=["date", "sektor_id"], how="left")
    return cur[["ticker", "date", "sector_breadth"]]


def fit_eval(seg, feats, val_cov=0.01):
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["split"] == "train"]; val = seg[seg["split"] == "val"]; te = seg[seg["split"] == "test"]
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, tr[TARGET].values)
    sc = lambda part: clf.predict_proba(
        (part[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values - mu) / sd)[:, 1]
    vt = val.assign(s=sc(val)); tt = te.assign(s=sc(te))
    tau = np.quantile(vt["s"], 1 - val_cov)
    p = tt[tt["s"] >= tau]
    return dict(auc=v1.auc(te[TARGET].values, sc(te)), n=len(p),
                prec=p[TARGET].mean(), mean=p[RET].mean(), med=p[RET].median(),
                win=(p[RET] > 0).mean())


def main():
    log("loading..."); m = v1.load_intraday(); d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize(); panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    pl = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]
    sec = pd.read_csv(SECTOR)
    daily_x = extra_daily(m)

    lines = []; A = lines.append
    A("# v3b — extra 1h features for precision (basket=ret_v1exit, sel@VAL-cov 1%)\n")
    A(f"EXTRA: {EXTRA}\n")

    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg.merge(extra_intraday(m, H), on=["ticker", "date"], how="left")
        seg = seg.merge(daily_x, on=["ticker", "date"], how="left")
        seg = seg.merge(sector_breadth(m, d, H, sec), on=["ticker", "date"], how="left")
        seg["split"] = np.where(seg["date"] < v2.TRAIN_END, "train",
                                np.where(seg["date"] < v2.VAL_END, "val", "test"))
        tr = seg[seg["split"] == "train"]
        A(f"## {label}  (test base={seg[seg.split=='test'][TARGET].mean():.2%})")
        # single-feature TRAIN AUC for each extra
        sa = {f: v1.auc(tr[TARGET].values, tr[f].fillna(tr[f].median()).values) for f in EXTRA}
        A("- extra single-feat TRAIN AUC: " + ", ".join(
            f"{f}={a:.3f}" for f, a in sorted(sa.items(), key=lambda kv: -abs((kv[1] or .5) - .5))))
        base = fit_eval(seg, v1.NEW)
        full = fit_eval(seg, v1.NEW + EXTRA)
        A(f"- **base NEW**   : AUC={base['auc']:.3f} | n={base['n']} prec={base['prec']*100:.0f}% "
          f"mean={base['mean']*100:+.1f}% med={base['med']*100:+.1f}% win={base['win']*100:.0f}%")
        A(f"- **NEW+EXTRA**  : AUC={full['auc']:.3f} | n={full['n']} prec={full['prec']*100:.0f}% "
          f"mean={full['mean']*100:+.1f}% med={full['med']*100:+.1f}% win={full['win']*100:.0f}%")
        # LOFO: drop each extra from full, see precision/mean change
        lofo = []
        for f in EXTRA:
            r = fit_eval(seg, v1.NEW + [x for x in EXTRA if x != f])
            lofo.append((f, full["prec"] - r["prec"], full["mean"] - r["mean"]))
        A("- LOFO (Δprec / Δmean when removed): " + ", ".join(
            f"{f}={dp*100:+.0f}%/{dm*100:+.1f}%" for f, dp, dm in
            sorted(lofo, key=lambda x: -abs(x[1]))))
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
