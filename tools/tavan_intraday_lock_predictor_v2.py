"""Tavan intraday lock predictor v2 — SELECTION METHOD (eliminate faders) + regime split.

Builds on v1 (memory/tavan_v1_intraday_lock_predictor_v0_outcome.md + v1 enriched
features). Reuses v1's build()/feature machinery verbatim (imported), adds:

  * runit_exit  — wide disaster stop (-15%) + 25d hold, NO TP/trail. This is the
                  exit that revealed the v1_lock right tail (TP/trail chopped it).
  * 3-fold TEMPORAL split: TRAIN (model) / VAL (pick selection params) / TEST (report).
                  Selection threshold is chosen on VAL, NEVER on TEST (overfit guard).
  * Selection methods, compared on TEST:
        - top-k%        (fixed budget; the v0/v1 rule — coverage-oriented)
        - prob-thresh   (fire iff P>=tau*, tau* = argmax VAL basket return w/ min coverage)
        - EV-thresh     (EV = p*GOOD_val + (1-p)*FADED_val; fire iff EV>=cut*, cut* on VAL)
  * Regime split of TEST picks (RDP long/neutral/short) — is the edge bull-only?

Question the user posed: instead of top-%, find a method that ELIMINATES FADERS.
The classifier already separates GOOD vs FADED (AUC ~0.93); top-% is the wrong
operating rule (forces a fixed basket). A VAL-tuned probability/EV threshold with
abstention targets precision instead of coverage.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tavan_intraday_lock_predictor_v1 as v1  # noqa: E402

REGIME = REPO / "output/regime_labels_daily_rdp_v1.csv"
OUT_MD = REPO / "output/tavan_intraday_lock_predictor_v2.md"

TRAIN_END = pd.Timestamp("2024-01-01")   # < : train (fit model)
VAL_END = pd.Timestamp("2025-01-01")     # [TRAIN_END, VAL_END): val (pick thresholds)
                                         # >= VAL_END : test (report, untouched)
DISASTER = 0.15                          # run-it wide stop
TARGET = "v1_lock"
FEATS = v1.NEW
MIN_COV = 0.004                          # min fraction of VAL rows a threshold must pick
                                         # (guards against 1-2 trade overfit thresholds)


def log(m): print(m, flush=True)


def runit_exit(ev, disaster=DISASTER, horizon=v1.HORIZON):
    """Wide disaster stop + hold to horizon. Entry = intraday snapshot price."""
    n = len(ev)
    e = ev["entry"].values.astype(float)
    L = np.array([ev[f"d{i}_low"].values for i in range(1, horizon + 1)], float)
    C = np.array([ev[f"d{i}_close"].values for i in range(1, horizon + 1)], float)
    stop = e * (1 - disaster)
    ret = np.full(n, np.nan)
    done = np.zeros(n, bool)
    last = np.full(n, np.nan)
    for day in range(horizon):
        l, c = L[day], C[day]
        ok = np.isfinite(c)
        last[ok] = c[ok]
        hit = (~done) & np.isfinite(l) & (l <= stop)
        ret[hit] = -disaster
        done[hit] = True
    surv = ~done
    ret[surv] = (last[surv] - e[surv]) / e[surv]
    return ret


def fit_score(seg):
    """Fit logit/NEW on TRAIN, return seg with score + split tags (val/test)."""
    from sklearn.linear_model import LogisticRegression
    tr = seg[seg["date"] < TRAIN_END]
    if tr[TARGET].sum() < 20:
        return None
    Xtr = tr[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, tr[TARGET].values)
    X = seg[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    seg = seg.assign(score=clf.predict_proba((X - mu) / sd)[:, 1])
    seg["split"] = np.where(seg["date"] < TRAIN_END, "train",
                            np.where(seg["date"] < VAL_END, "val", "test"))
    return seg


def basket_stats(picks):
    """Summary of a selected basket (run-it returns)."""
    if len(picks) == 0:
        return dict(n=0, prec=np.nan, fader_pct=np.nan, basket=np.nan,
                    med=np.nan, good=np.nan, faded=np.nan, ndays=0)
    g = picks[picks[TARGET] == 1]
    f = picks[picks[TARGET] == 0]
    return dict(n=len(picks), prec=picks[TARGET].mean(),
                fader_pct=1 - picks[TARGET].mean(),
                basket=picks["ret_runit"].mean(), med=picks["ret_runit"].median(),
                good=g["ret_runit"].mean() if len(g) else np.nan,
                faded=f["ret_runit"].mean() if len(f) else np.nan,
                ndays=picks["date"].dt.normalize().nunique())


def pick_tau(val):
    """tau* = score cutoff maximizing VAL mean basket return, with min coverage."""
    cand = np.quantile(val["score"], np.linspace(0.80, 0.999, 60))
    best, best_tau = -np.inf, None
    floor = max(int(len(val) * MIN_COV), 8)
    for tau in cand:
        p = val[val["score"] >= tau]
        if len(p) < floor:
            continue
        m = p["ret_runit"].mean()
        if m > best:
            best, best_tau = m, tau
    return best_tau


def pick_ev_cut(val):
    """EV(p)=p*GOOD_val+(1-p)*FADED_val; cut* maximizes VAL mean basket w/ min cov."""
    g = val[val[TARGET] == 1]["ret_runit"].mean()
    f = val[val[TARGET] == 0]["ret_runit"].mean()
    val = val.assign(ev=val["score"] * g + (1 - val["score"]) * f)
    cand = np.quantile(val["ev"], np.linspace(0.80, 0.999, 60))
    best, best_cut = -np.inf, None
    floor = max(int(len(val) * MIN_COV), 8)
    for cut in cand:
        p = val[val["ev"] >= cut]
        if len(p) < floor:
            continue
        m = p["ret_runit"].mean()
        if m > best:
            best, best_cut = m, cut
    return best_cut, g, f


def main():
    log("loading 1h master..."); m = v1.load_intraday()
    d = v1.daily_frame(m)
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan", "ml_s", "tavan_streak"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["ticker", "date"])
    panel["streak_prior"] = panel.groupby("ticker")["tavan_streak"].shift(1).fillna(0)
    panel["tavan_freq_20"] = panel.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    panel_lab = panel[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]

    reg = pd.read_csv(REGIME)
    reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
    reg = reg[["date", "regime"]]

    pools = []
    for label, H in v1.SNAPSHOTS.items():
        log(f"building {label}..."); pools.append(v1.build(m, d, panel_lab, H, label))
    allp = pd.concat(pools, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"]).dt.normalize()
    allp["ret_runit"] = runit_exit(allp)
    allp = allp.merge(reg, on="date", how="left")

    lines = []; A = lines.append
    A("# Tavan Intraday Lock Predictor v2 — selection method (eliminate faders) + regime\n")
    A(f"Target **{TARGET}**. Model logit/NEW on TRAIN(<{TRAIN_END.date()}); "
      f"thresholds tuned on VAL([{TRAIN_END.date()},{VAL_END.date()})); "
      f"TEST(>={VAL_END.date()}) reported untouched. Exit = run-it (disaster −{DISASTER:.0%} + H{v1.HORIZON} hold).")
    A("Methods: **top-k%** (fixed budget) vs **prob-thresh** (P≥τ*, abstain) vs "
      "**EV-thresh** (EV≥cut*). τ*/cut* = argmax VAL basket return, min-coverage guarded.\n")

    for label in v1.SNAPSHOTS:
        seg = fit_score(allp[allp["snapshot"] == label].copy())
        if seg is None:
            A(f"## {label}: insufficient train positives\n"); continue
        val = seg[seg["split"] == "val"].copy()
        test = seg[seg["split"] == "test"].copy()
        A(f"## {label}  (val n={len(val):,} / test n={len(test):,}; "
          f"test base={test[TARGET].mean():.2%}, AUC={v1.auc(test[TARGET].values, test['score'].values):.3f})")
        nday = test["date"].dt.normalize().nunique()

        # baselines: top-1% / top-3%
        methods = {}
        for frac in (0.01, 0.03):
            k = max(int(len(test) * frac), 10)
            methods[f"top-{int(frac*100)}%"] = test.nlargest(k, "score")
        # prob-threshold tuned on VAL
        tau = pick_tau(val)
        if tau is not None:
            methods[f"prob τ={tau:.3f}"] = test[test["score"] >= tau]
        # EV-threshold tuned on VAL
        cut, gv, fv = pick_ev_cut(val)
        if cut is not None:
            ev_test = test["score"] * gv + (1 - test["score"]) * fv
            methods[f"EV cut={cut:+.3f}"] = test[ev_test.values >= cut]

        A("\n| method | n | days/​{0} | prec | fader% | basket | median | GOOD | FADED |".format(nday))
        A("|---|---|---|---|---|---|---|---|---|")
        for name, picks in methods.items():
            s = basket_stats(picks)
            A(f"| {name} | {s['n']} | {s['ndays']}/{nday} | "
              f"{s['prec']*100:.0f}% | {s['fader_pct']*100:.0f}% | "
              f"{s['basket']*100:+.1f}% | {s['med']*100:+.1f}% | "
              f"{s['good']*100:+.1f}% | {s['faded']*100:+.1f}% |")

        # regime split of the prob-threshold picks (the fader-eliminating method)
        if tau is not None:
            pp = methods[f"prob τ={tau:.3f}"]
            A(f"\n**Regime split (prob τ picks, test):** ")
            parts = []
            for rg in ["long", "neutral", "short"]:
                sub = pp[pp["regime"] == rg]
                if len(sub):
                    parts.append(f"{rg} n={len(sub)} prec={sub[TARGET].mean()*100:.0f}% "
                                 f"basket={sub['ret_runit'].mean()*100:+.1f}%")
                else:
                    parts.append(f"{rg} n=0")
            A("; ".join(parts))
        A("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines)); log(f"\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
