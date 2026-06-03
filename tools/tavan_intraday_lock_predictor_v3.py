"""Tavan intraday lock predictor v3 — PRECISION selection under the PROPER V1 exit.

Correction over v2: v2 used run-it (−15%/H25) which whipsaws winners into a tail
lottery (median −15%). Under the validated V1 exit (SL−10/TP1+4→trail2/H25) the
v1_lock hits are +7% median / ~98% win (the real super-trade), and the top-1%
BASKET median is already positive (+5%); only the MEAN is thin (+1-2%) because
faders' −10% stops drag it. So eliminating faders (the user's instinct) IS the
lever: tighten selection → precision up → basket mean climbs toward the +7% ceiling,
median stays positive. This tool measures exactly that, under V1-exit.

Design: model logit/NEW on TRAIN; selection threshold chosen by VAL coverage target
(never on TEST); TEST reported untouched. Basket return = ret_v1exit. We sweep a
VAL-coverage grid (tighten → looser) to trace the precision/mean tradeoff, plus the
VAL-optimal max-mean point, plus a regime split.
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

REGIME = REPO / "output/regime_labels_daily_rdp_v1.csv"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v3.md"
TARGET = "v1_lock"
FEATS = v1.NEW
RET = "ret_v1exit"                       # the PROPER exit (median-positive)
COV_GRID = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]  # VAL coverage targets (tight→loose)


def log(m): print(m, flush=True)


def stats(picks, ndays):
    if len(picks) == 0:
        return dict(n=0, cov=0, prec=np.nan, fad=np.nan, mean=np.nan,
                    med=np.nan, win=np.nan, good=np.nan, faded=np.nan)
    r = picks[RET]
    g = picks[picks[TARGET] == 1][RET]
    f = picks[picks[TARGET] == 0][RET]
    return dict(n=len(picks), cov=picks["date"].dt.normalize().nunique(),
                prec=picks[TARGET].mean(), fad=1 - picks[TARGET].mean(),
                mean=r.mean(), med=r.median(), win=(r > 0).mean(),
                good=g.mean() if len(g) else np.nan,
                faded=f.mean() if len(f) else np.nan)


def main():
    log("loading 1h master..."); m = v1.load_intraday()
    d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize(); panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    pl = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]
    reg = pd.read_csv(REGIME); reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
    reg = reg[["date", "regime"]]

    lines = []; A = lines.append
    A("# Tavan Intraday Lock Predictor v3 — precision selection under V1-exit\n")
    A(f"Target **{TARGET}**, basket return = **{RET}** (SL−10/TP1+4→trail2/H25). "
      f"Model logit/NEW on TRAIN(<{v2.TRAIN_END.date()}); selection τ set by VAL "
      f"coverage target([{v2.TRAIN_END.date()},{v2.VAL_END.date()})); "
      f"TEST(>={v2.VAL_END.date()}) untouched.")
    A("Question: does tightening selection (eliminating faders) lift basket MEAN "
      "toward the v1-hit ceiling (~+7%) while keeping median positive?\n")

    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg.merge(reg, on="date", how="left")
        seg = v2.fit_score(seg)
        if seg is None:
            A(f"## {label}: insufficient\n"); continue
        val = seg[seg["split"] == "val"].copy()
        test = seg[seg["split"] == "test"].copy()
        ndays = test["date"].dt.normalize().nunique()
        # ceiling: v1-hit return under V1-exit
        hit = test[test[TARGET] == 1][RET]
        A(f"## {label}  (test n={len(test):,}, base={test[TARGET].mean():.2%}, "
          f"AUC={v1.auc(test[TARGET].values, test['score'].values):.3f})")
        A(f"- **ceiling** (perfect v1-pick, V1-exit): mean={hit.mean():+.1%} "
          f"median={hit.median():+.1%} win={(hit>0).mean():.0%}")
        A("\n| VAL-cov | τ | test n | days/​{0} | prec | fader% | **mean** | median | win% | GOOD | FADED |".format(ndays))
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        best = None
        for cov in COV_GRID:
            tau = np.quantile(val["score"], 1 - cov)        # VAL coverage target → τ
            p = test[test["score"] >= tau]
            s = stats(p, ndays)
            A(f"| {cov:.1%} | {tau:.3f} | {s['n']} | {s['cov']}/{ndays} | "
              f"{s['prec']*100:.0f}% | {s['fad']*100:.0f}% | **{s['mean']*100:+.1f}%** | "
              f"{s['med']*100:+.1f}% | {s['win']*100:.0f}% | {s['good']*100:+.1f}% | {s['faded']*100:+.1f}% |")
            # VAL-optimal (max VAL mean under V1-exit, min 8 picks)
            vp = val[val["score"] >= tau]
            if len(vp) >= 8:
                vm = vp[RET].mean()
                if best is None or vm > best[1]:
                    best = (tau, vm, cov)
        # regime split at VAL-optimal τ
        if best is not None:
            tau = best[0]; p = test[test["score"] >= tau]
            A(f"\n**VAL-optimal τ={tau:.3f} (val-cov {best[2]:.1%}):** "
              f"test n={len(p)} prec={p[TARGET].mean()*100:.0f}% mean={p[RET].mean()*100:+.1f}% "
              f"median={p[RET].median()*100:+.1f}% win={(p[RET]>0).mean()*100:.0f}%")
            parts = []
            for rg in ["long", "neutral", "short"]:
                sub = p[p["regime"] == rg]
                parts.append(f"{rg} n={len(sub)} mean={sub[RET].mean()*100:+.1f}%/win={(sub[RET]>0).mean()*100:.0f}%"
                             if len(sub) else f"{rg} n=0")
            A(f"- regime: {'; '.join(parts)}")
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
