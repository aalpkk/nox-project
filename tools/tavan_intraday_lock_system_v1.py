"""Tavan Intraday Lock SYSTEM v1 — frozen live apparatus (PAPER-TRACK).

Outcome: memory/tavan_intraday_lock_system_outcome.md.
Package: enter EARLY (11:00) on top lock-model picks at the +3-8% price (fillable,
pre-queue) → exit with the standard V1-exit (SL−10/TP1+4 sell-half/2%-trail/BE/H25).
Model = logistic on 1h NEW + 15m FEATS15, target = LOCK (quality is structural/pre-open,
handled separately; lock is the hard intraday part 15m helps with).

⚠️ SINGLE-REGIME LEAD (trained 2025-2026 bull). PAPER-TRACK ONLY — not validated live.

Subcommands:
  freeze              train final model on ALL labeled data >= 2025-01, save artifacts
  scan --asof DATE [--snapshot 11:00] [--topk N]
                      produce the candidate list + entry/exit plan for that day
                      (reads the master parquets; run the daily pulls first so today's
                       bars up to the snapshot are present)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import tavan_intraday_lock_predictor_v1 as v1   # noqa: E402
import tavan_intraday_lock_predictor_v4_15m as v4  # noqa: E402

MODEL = REPO / "output/tavan_intraday_lock_system_v1_model.json"
TRACKER = REPO / "output/tavan_intraday_lock_system_tracker.csv"
BASE = v1.NEW + v4.FEATS15
TRAIN_FROM = pd.Timestamp("2025-01-01")
SL_PCT, TP1_PCT, TP1_LOCK, TRAIL = 0.10, 0.04, 0.02, 0.02


def log(m): print(m, flush=True)


def _ohlcv_lab(d):
    """Label/streak features derived from OHLCV (close>=prev*1.095=tavan), NOT the
    panel — so the pool builds for ANY date incl. live (no today's-label dependency).
    streak_prior/tavan_freq_20 use only PRIOR days (shifted), so they are pre-T safe."""
    x = d[["ticker", "date", "prev_close", "d_close"]].copy().sort_values(["ticker", "date"])
    x["is_tavan"] = ((x["d_close"] >= x["prev_close"] * 1.095) & x["prev_close"].notna()).astype(float)
    run = x.groupby("ticker")["is_tavan"].transform(
        lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1))
    x["streak_prior"] = run.groupby(x["ticker"]).shift(1).fillna(0)
    x["tavan_freq_20"] = x.groupby("ticker")["is_tavan"].transform(
        lambda s: s.rolling(20, min_periods=1).sum().shift(1)).fillna(0)
    x["ml_s"] = np.nan
    return x[["ticker", "date", "is_tavan", "ml_s", "streak_prior", "tavan_freq_20"]]


def _pools():
    """Build the +3-8% pool with 1h NEW + 15m FEATS15, all snapshots. Features are
    self-contained (OHLCV+15m, no panel) so scan runs live; freeze adds the real label."""
    m = v1.load_intraday(); d = v1.daily_frame(m)
    pl = _ohlcv_lab(d)
    m15 = v4.load15(); pc = d[["ticker", "date", "prev_close"]].copy()
    out = {}
    for label, H in v1.SNAPSHOTS.items():
        seg = v1.build(m, d, pl, H, label)
        seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
        seg = seg.merge(v4.feats15(m15, v4.SNAP_STR[label], pc), on=["ticker", "date"], how="left")
        out[label] = seg
    return out


def freeze():
    from sklearn.linear_model import LogisticRegression
    pools = _pools()
    # accurate training label from the panel (historical only); features are OHLCV-derived
    panel = pd.read_parquet(v1.PANEL, columns=["date", "ticker", "is_tavan"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.rename(columns={"is_tavan": "lock_true"})
    art = {"base": BASE, "trained_from": str(TRAIN_FROM.date()), "snapshots": {}}
    for label, seg in pools.items():
        seg = seg.merge(panel, on=["ticker", "date"], how="left")
        seg["lock"] = seg["lock_true"]
        tr = seg[(seg["date"] >= TRAIN_FROM) & seg["lock"].notna()]
        X = tr[BASE].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        mu, sd = X.mean(0), X.std(0) + 1e-9
        clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit((X - mu) / sd, tr["lock"].values)
        sc = clf.predict_proba((X - mu) / sd)[:, 1]
        art["snapshots"][label] = dict(
            coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
            mu=mu.tolist(), sd=sd.tolist(),
            thr_top1=float(np.quantile(sc, 0.99)), thr_top3=float(np.quantile(sc, 0.97)),
            n_train=int(len(tr)), lock_rate=float(tr["lock"].mean()))
        log(f"  {label}: trained n={len(tr):,} lock-rate={tr['lock'].mean():.1%} "
            f"thr_top1={art['snapshots'][label]['thr_top1']:.3f}")
    MODEL.write_text(json.dumps(art, indent=2))
    log(f"-> saved {MODEL}")


def _score(seg, a):
    X = seg[BASE].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    z = (X - np.array(a["mu"])) / np.array(a["sd"])
    lin = z @ np.array(a["coef"]) + a["intercept"]
    return 1 / (1 + np.exp(-lin))


def scan(asof, snapshot, topk, telegram=False):
    if not MODEL.exists():
        log("No frozen model — run: python tools/tavan_intraday_lock_system_v1.py freeze"); return
    art = json.loads(MODEL.read_text())
    label = {"11:00": "~11:00", "13:00": "~13:00", "15:00": "~15:00"}[snapshot]
    a = art["snapshots"][label]
    asof = pd.Timestamp(asof).normalize()
    seg = _pools()[label]
    day = seg[seg["date"] == asof].copy()
    if day.empty:
        log(f"No pool rows for {asof.date()} @ {snapshot} (no +3-8% candidates, or data not pulled). "); return
    day["P_lock"] = _score(day, a)
    day = day.sort_values("P_lock", ascending=False)
    thr = a["thr_top1"]
    log(f"\n=== TAVAN INTRADAY LOCK SCAN — {asof.date()} @ {snapshot} ===")
    log(f"(model trained {art['trained_from']}+, lock-rate {a['lock_rate']:.1%}; "
        f"high-conviction P_lock>={thr:.3f}) — PAPER-TRACK\n")
    log(f"{'tk':<8}{'P_lock':>7}  {'now%':>6}  {'entry':>8}  {'SL-10%':>8}  {'TP1+4%':>8}  conv")
    picks = []
    for _, r in day.head(max(topk, 8)).iterrows():
        e = r["entry"]; conv = "**HIGH**" if r["P_lock"] >= thr else ""
        log(f"{r['ticker']:<8}{r['P_lock']*100:>6.1f}%  {r['ret_at_T']*100:>5.1f}%  "
            f"{e:>8.2f}  {e*(1-SL_PCT):>8.2f}  {e*(1+TP1_PCT):>8.2f}  {conv}")
        if r["P_lock"] >= thr:
            picks.append(r["ticker"])
    log(f"\nHIGH-conviction picks (P_lock>={thr:.3f}): {picks if picks else 'NONE today'}")
    log("EXIT PLAN per pick: SL=−10% | at +4% sell HALF (locks +2%) | trail remaining 2% "
        "below high, floor=breakeven | hard exit 25 trading days.")

    if telegram:
        hi = day[day["P_lock"] >= thr]
        lines_tg = [f"🎯 TAVAN LOCK {asof.date()} @{snapshot} (PAPER)"]
        if len(hi):
            for _, r in hi.iterrows():
                e = r["entry"]
                lines_tg.append(f"• {r['ticker']}  P={r['P_lock']*100:.0f}%  al≈{e:.2f}  "
                                f"SL {e*(1-SL_PCT):.2f}  TP1 {e*(1+TP1_PCT):.2f}")
            lines_tg.append("Exit: +4%→yarı sat, kalan %2 trail (BE taban), SL −10%, H25.")
        else:
            top = day.iloc[0]
            lines_tg.append(f"HIGH yok. En yüksek: {top['ticker']} P={top['P_lock']*100:.0f}%")
        try:
            from core.reports import send_telegram
            send_telegram("\n".join(lines_tg)); log("[telegram sent]")
        except Exception as e:
            log(f"[telegram failed: {e}]")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sp = sub.add_parser("scan")
    sp.add_argument("--asof", required=True)
    sp.add_argument("--snapshot", default="11:00", choices=["11:00", "13:00", "15:00"])
    sp.add_argument("--topk", type=int, default=12)
    sp.add_argument("--telegram", action="store_true")
    args = ap.parse_args()
    if args.cmd == "freeze":
        freeze()
    else:
        scan(args.asof, args.snapshot, args.topk, args.telegram)


if __name__ == "__main__":
    main()
