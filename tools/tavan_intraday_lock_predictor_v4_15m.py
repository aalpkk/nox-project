"""v4 — 15m MICROSTRUCTURE probe: does finer intraday data raise lock precision?

Context: 1h is tapped out (v3b), structural features predict the ml_s quality-gate
not the lock (v3c). The genuinely hard, intraday-uncertain part is the LOCK. This
tests whether 15m microstructure (finer momentum, volume surge, buy-pressure,
ceiling-pressing, VWAP) cracks the lock-prediction precision ceiling that 1h hit.

HONEST LIMITATION: usable 15m density is ONLY 2025-01 → 2026-06 (~1.4y, single
bull-ish regime). So this is a THIN, single-regime, in-sample-ish FEASIBILITY probe
— it can RULE 15m OUT (no lift even here) or flag a LEAD (lift worth forward-tracking),
but cannot robustly validate. Split is temporal within 2025-2026.

Apples-to-apples: pool/labels/exit are the SAME 1h-defined rows (v1.build); only the
feature set differs — NEW (1h) vs FEATS15 (15m) vs NEW+15m. Targets: lock + v1_lock.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import tavan_intraday_lock_predictor_v1 as v1   # noqa: E402

M15 = REPO / "output/extfeed_intraday_15m_master.parquet"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v4_15m.md"
RET = "ret_v1exit"
SNAP_STR = {"~11:00": "11:00", "~13:00": "13:00", "~15:00": "15:00"}
# split within 15m-dense window (2025-2026)
TR_END = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2026-01-01")
FEATS15 = ["r15_1", "accel15", "vol_surge15", "buy_press15", "up_vol_frac15",
           "near_ceil15", "mae15", "vwap_dist15", "last45_slope"]


def log(m): print(m, flush=True)


def load15():
    df = pd.read_parquet(M15)
    t = pd.to_datetime(df["time"])
    df["date"] = t.dt.normalize().dt.tz_localize(None)
    df["hm"] = t.dt.strftime("%H:%M")
    df = df[df["date"] >= pd.Timestamp("2024-12-01")]      # dense window only
    return df.sort_values(["ticker", "date", "hm"]).reset_index(drop=True)


def feats15(m15, T_str, pc):
    """15m microstructure from bars time-of-day <= T_str (pre-T), per ticker-date."""
    sub = m15[m15["hm"] <= T_str].copy()
    sub = sub.merge(pc, on=["ticker", "date"], how="left")
    sub = sub[(sub["prev_close"] > 0) & sub["prev_close"].notna()]
    g = sub.groupby(["ticker", "date"], sort=False)
    rng = (sub["high"] - sub["low"]).replace(0, np.nan)
    sub["bp"] = ((sub["close"] - sub["low"]) / rng).fillna(0.5)
    sub["pc_close"] = g["close"].shift(1)
    sub["bar_ret"] = sub["close"] / sub["pc_close"] - 1
    sub["upvol"] = np.where(sub["close"] > sub["open"], sub["volume"], 0.0)
    sub["run_high"] = g["high"].cummax()
    sub["dd"] = (sub["low"] - sub["run_high"]) / sub["run_high"]
    sub["tp"] = (sub["high"] + sub["low"] + sub["close"]) / 3
    sub["tpv"] = sub["tp"] * sub["volume"]
    sub["near_ceil"] = (sub["high"] >= sub["prev_close"] * 1.085).astype(float)
    agg = g.agg(
        cum_close=("close", "last"), nbar15=("close", "size"),
        buy_press15=("bp", "mean"), up_vol=("upvol", "sum"), tot_vol=("volume", "sum"),
        mae15=("dd", "min"), near_ceil15=("near_ceil", "mean"),
        tpv_sum=("tpv", "sum"), prev_close=("prev_close", "first"),
    ).reset_index()
    agg["up_vol_frac15"] = agg["up_vol"] / agg["tot_vol"].replace(0, np.nan)
    agg["vwap_dist15"] = (agg["cum_close"] - agg["tpv_sum"] / agg["tot_vol"].replace(0, np.nan)) \
        / (agg["tpv_sum"] / agg["tot_vol"].replace(0, np.nan))
    # last-bar features: take tail bars per group
    tail = sub.groupby(["ticker", "date"]).tail(4).copy()
    tg = tail.groupby(["ticker", "date"], sort=False)
    last = tg.agg(r15_1=("bar_ret", "last"),
                  last45_slope=("bar_ret", lambda s: s.tail(3).sum())).reset_index()
    # accel = last bar ret - the bar-ret before it
    def _accel(s):
        v = s.dropna().values
        return (v[-1] - v[-2]) if len(v) >= 2 else 0.0
    acc = tg["bar_ret"].apply(_accel).rename("accel15").reset_index()
    out = agg.merge(last, on=["ticker", "date"], how="left").merge(acc, on=["ticker", "date"], how="left")
    keep = ["ticker", "date"] + FEATS15
    out["vol_surge15"] = np.nan  # filled below from nbar split
    # vol_surge: last-2-bars vol / mean earlier-bars vol
    l2 = sub.groupby(["ticker", "date"]).tail(2).groupby(["ticker", "date"])["volume"].sum().rename("v_last2")
    out = out.merge(l2, on=["ticker", "date"], how="left")
    out["vol_surge15"] = out["v_last2"] / ((out["tot_vol"] - out["v_last2"]) /
                                           (out["nbar15"] - 2).clip(lower=1)).replace(0, np.nan)
    return out[keep]


def fit_eval(seg, feats, target, val_cov=0.01):
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["split"] == "train"]; val = seg[seg["split"] == "val"]; te = seg[seg["split"] == "test"]
    if tr[target].sum() < 15 or te[target].sum() < 5:
        return None
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, tr[target].values)
    sc = lambda part: clf.predict_proba(
        (part[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values - mu) / sd)[:, 1]
    vt = val.assign(s=sc(val)); tt = te.assign(s=sc(te))
    tau = np.quantile(vt["s"], 1 - val_cov)
    p = tt[tt["s"] >= tau]
    return dict(auc=v1.auc(te[target].values, sc(te)), n=len(p),
                prec=p[target].mean() if len(p) else np.nan,
                mean=p[RET].mean() if len(p) else np.nan,
                med=p[RET].median() if len(p) else np.nan,
                win=(p[RET] > 0).mean() if len(p) else np.nan)


def main():
    log("loading 1h + daily..."); m = v1.load_intraday(); d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize(); panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    pl = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]
    log("loading 15m..."); m15 = load15()
    pc = d[["ticker", "date", "prev_close"]].copy()

    lines = []; A = lines.append
    A("# v4 — 15m microstructure probe (lock precision vs 1h). THIN single-regime (2025-2026).\n")
    A(f"Split: train<{TR_END.date()} / val<{VAL_END.date()} / test>= {VAL_END.date()}. "
      f"FEATS15={FEATS15}\n")

    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg[seg["date"] >= pd.Timestamp("2025-01-01")]            # 15m-dense
        f15 = feats15(m15, SNAP_STR[label], pc)
        seg = seg.merge(f15, on=["ticker", "date"], how="inner")
        seg["split"] = np.where(seg["date"] < TR_END, "train",
                                np.where(seg["date"] < VAL_END, "val", "test"))
        ntr = (seg["split"] == "train").sum(); nte = (seg["split"] == "test").sum()
        A(f"## {label}  (rows {len(seg):,}; train {ntr:,}/test {nte:,})")
        for target in ["lock", "v1_lock"]:
            te = seg[seg["split"] == "test"]
            A(f"### target={target} (test base={te[target].mean():.2%}, pos={int(te[target].sum())})")
            for tag, fs in [("1h NEW", v1.NEW), ("15m", FEATS15), ("1h+15m", v1.NEW + FEATS15)]:
                r = fit_eval(seg, fs, target)
                if r is None:
                    A(f"- {tag}: insufficient"); continue
                A(f"- **{tag:7s}**: AUC={r['auc']:.3f} | top-1% n={r['n']} prec={r['prec']*100:.0f}% "
                  f"mean={r['mean']*100:+.1f}% med={r['med']*100:+.1f}% win={r['win']*100:.0f}%")
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
