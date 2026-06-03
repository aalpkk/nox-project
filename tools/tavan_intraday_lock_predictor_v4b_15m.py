"""v4b — enrich the 15m feature set for LOCK prediction (overfit-guarded).

Adds 7 theory-motivated lock-FORMATION microstructure features on top of v4's 9.
A lock = price pinned at +10% with sellers drying up. Signatures, all pre-T:
  vol_dryup15      mean vol(last 2 bars) / max vol(earlier bars)   (absorption: LOW=drying)
  range_compress15 mean range(last 3) / mean range(earlier)        (pinned near ceiling)
  close_near_hi15  frac of bars closing >=99% of running intraday high (holding gains)
  upper_wick15     mean upper-wick frac of last 3 bars             (rejection at highs: HIGH=bad)
  dist_from_hi15   (cum_close - cum_high)/cum_high at T            (closing off highs = weakening)
  consec_hh15      consecutive higher-high bars ending at T        (relentless ascent)
  accel_late15     ret(last 3 bars) - ret(prior 3 bars)            (speeding up into T)

OVERFIT GUARD (thin single-regime 2025-2026 test): the verdict is CONSISTENCY across
the 3 snapshots, not one cell. A real feature helps lock in the same direction at
11/13/15:00; a flip-flopper is noise. Reports single-feat TRAIN AUC + LOFO + base
(1h+15m) vs +EXTRA on the operating point. Target = lock (reliable; v1_lock too thin).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import tavan_intraday_lock_predictor_v1 as v1   # noqa: E402
import tavan_intraday_lock_predictor_v4_15m as v4  # noqa: E402

OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v4b_15m.md"
RET = "ret_v1exit"
EXTRA15 = ["vol_dryup15", "range_compress15", "close_near_hi15", "upper_wick15",
           "dist_from_hi15", "consec_hh15", "accel_late15"]


def log(m): print(m, flush=True)


def feats15_extra(m15, T_str, pc):
    sub = m15[m15["hm"] <= T_str].copy()
    sub = sub.merge(pc, on=["ticker", "date"], how="left")
    sub = sub[(sub["prev_close"] > 0) & sub["prev_close"].notna()]
    sub = sub.sort_values(["ticker", "date", "hm"])
    g = sub.groupby(["ticker", "date"], sort=False)
    sub["nfe"] = g.cumcount(ascending=False)            # 0 = last bar
    rng = (sub["high"] - sub["low"]).replace(0, np.nan)
    sub["bar_range"] = rng / sub["close"]
    sub["uw"] = ((sub["high"] - sub["close"]) / rng).fillna(0.0)
    sub["run_high"] = g["high"].cummax()
    sub["near_hi"] = (sub["close"] >= 0.99 * sub["run_high"]).astype(float)
    sub["bar_ret"] = sub["close"] / g["close"].shift(1) - 1
    sub["hh"] = (sub["high"] > g["high"].shift(1)).astype(float)

    base = g.agg(cum_close=("close", "last"), cum_high=("high", "max"),
                 close_near_hi15=("near_hi", "mean")).reset_index()
    base["dist_from_hi15"] = (base["cum_close"] - base["cum_high"]) / base["cum_high"]

    def cmean(mask, col, name):
        return sub[mask].groupby(["ticker", "date"])[col].mean().rename(name)

    def csum(mask, col, name):
        return sub[mask].groupby(["ticker", "date"])[col].sum().rename(name)

    uw3 = cmean(sub["nfe"] <= 2, "uw", "upper_wick15")
    r_last3 = cmean(sub["nfe"] <= 2, "bar_range", "_rl3")
    r_earl = cmean(sub["nfe"] > 2, "bar_range", "_re")
    v_last2 = sub[sub["nfe"] <= 1].groupby(["ticker", "date"])["volume"].mean().rename("_vl2")
    v_earlmax = sub[sub["nfe"] > 1].groupby(["ticker", "date"])["volume"].max().rename("_vem")
    ret_last3 = csum(sub["nfe"] <= 2, "bar_ret", "_rr3")
    ret_prev3 = csum((sub["nfe"] >= 3) & (sub["nfe"] <= 5), "bar_ret", "_rp3")
    # consecutive higher-highs ending at last bar: trailing run of hh==1
    def trail(s):
        v = s.values
        c = 0
        for x in v[::-1]:
            if x == 1:
                c += 1
            else:
                break
        return c
    consec = g["hh"].apply(trail).rename("consec_hh15")

    out = base.merge(uw3, on=["ticker", "date"], how="left") \
        .merge(r_last3, on=["ticker", "date"], how="left").merge(r_earl, on=["ticker", "date"], how="left") \
        .merge(v_last2, on=["ticker", "date"], how="left").merge(v_earlmax, on=["ticker", "date"], how="left") \
        .merge(ret_last3, on=["ticker", "date"], how="left").merge(ret_prev3, on=["ticker", "date"], how="left") \
        .merge(consec, on=["ticker", "date"], how="left")
    out["range_compress15"] = out["_rl3"] / out["_re"].replace(0, np.nan)
    out["vol_dryup15"] = out["_vl2"] / out["_vem"].replace(0, np.nan)
    out["accel_late15"] = out["_rr3"].fillna(0) - out["_rp3"].fillna(0)
    return out[["ticker", "date"] + EXTRA15]


def main():
    log("loading..."); m = v1.load_intraday(); d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize(); panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    pl = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]
    m15 = v4.load15(); pc = d[["ticker", "date", "prev_close"]].copy()

    BASE = v1.NEW + v4.FEATS15
    lines = []; A = lines.append
    A("# v4b — enriched 15m features for LOCK (overfit-guard = consistency across snapshots)\n")
    A(f"EXTRA15={EXTRA15}\nBASE=1h NEW + v4 FEATS15 (target=lock, basket=ret_v1exit, top-1%)\n")

    summary = []
    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg[seg["date"] >= pd.Timestamp("2025-01-01")]
        seg = seg.merge(v4.feats15(m15, v4.SNAP_STR[label], pc), on=["ticker", "date"], how="inner")
        seg = seg.merge(feats15_extra(m15, v4.SNAP_STR[label], pc), on=["ticker", "date"], how="inner")
        seg["split"] = np.where(seg["date"] < v4.TR_END, "train",
                                np.where(seg["date"] < v4.VAL_END, "val", "test"))
        tr = seg[seg["split"] == "train"]
        A(f"## {label}")
        sa = {f: v1.auc(tr["lock"].values, tr[f].fillna(tr[f].median()).values) for f in EXTRA15}
        A("- extra single-feat TRAIN AUC: " + ", ".join(
            f"{f.replace('15','')}={a:.3f}" for f, a in sorted(sa.items(), key=lambda kv: -abs((kv[1] or .5) - .5))))
        b = v4.fit_eval(seg, BASE, "lock"); f = v4.fit_eval(seg, BASE + EXTRA15, "lock")
        if b and f:
            A(f"- **BASE**      : AUC={b['auc']:.3f} n={b['n']} prec={b['prec']*100:.0f}% "
              f"mean={b['mean']*100:+.1f}% med={b['med']*100:+.1f}% win={b['win']*100:.0f}%")
            A(f"- **BASE+EXTRA**: AUC={f['auc']:.3f} n={f['n']} prec={f['prec']*100:.0f}% "
              f"mean={f['mean']*100:+.1f}% med={f['med']*100:+.1f}% win={f['win']*100:.0f}%")
            summary.append((label, b, f))
            lofo = []
            for x in EXTRA15:
                r = v4.fit_eval(seg, BASE + [y for y in EXTRA15 if y != x], "lock")
                if r:
                    lofo.append((x, f["prec"] - r["prec"], f["mean"] - r["mean"]))
            A("- LOFO (Δprec/Δmean removed): " + ", ".join(
                f"{x.replace('15','')}={dp*100:+.0f}%/{dm*100:+.1f}%" for x, dp, dm in
                sorted(lofo, key=lambda z: -abs(z[1]))))
        A("")

    A("## Consistency verdict (overfit guard)")
    if summary:
        dauc = np.mean([f["auc"] - b["auc"] for _, b, f in summary])
        dmean = np.mean([f["mean"] - b["mean"] for _, b, f in summary])
        dprec = np.mean([f["prec"] - b["prec"] for _, b, f in summary])
        npos = sum((f["mean"] - b["mean"]) > 0 for _, b, f in summary)
        A(f"- avg ΔAUC={dauc:+.3f}, Δmean={dmean*100:+.1f}pp, Δprec={dprec*100:+.0f}pp; "
          f"mean improved in {npos}/3 snapshots")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
