#!/usr/bin/env python3
"""Gün kalitesi: düşük skorlu günlerde işlem yapmamak işe yarıyor mu?

Soru: ML-Rank top-N listesinin skor SEVİYESİ günden güne değişiyor. Skorun
düşük olduğu günlerde seçicilik de kayboluyor mu — yani "bugün zayıf, atla"
kuralı kurulabilir mi?

Bu §6.6'daki "mutlak eşik kullanma" sonucuyla ÇELİŞMEZ: orada eşik hangi
HİSSENİN seçileceğini belirliyordu; burada sorulan, o günün TOPLU sinyal
gücünün ileri getiriyi öngörüp öngörmediği. Yine de aynı tuzak geçerli —
mutlak skor durağan olmadığından kural mutlak sayı üzerine değil, GENİŞLEYEN
PENCERE YÜZDEBİRLİĞİ üzerine kurulur (o güne kadarki geçmişle karşılaştırma,
look-ahead yok).

Kullanım:
  python tools/index_bt_ml_dayquality.py --topn 4 --outdir output/index_bt/ml_dayq
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.index_bt_engine import Evaluator                    # noqa: E402
from tools.index_bt_ml_threshold import block_bootstrap_mean   # noqa: E402
from alpha.config import MIN_VOLUME_TL, MIN_DATA_DAYS          # noqa: E402

PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"
MIN_HIST_DAYS = 250      # genişleyen pencere yüzdebirliği için asgari geçmiş


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--today-score", type=float, default=None,
                    help="bugünkü top-N ortalama ml_1g (tarihsel dağılımda yerini bulmak için)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    H, N = args.horizon, args.topn

    ev = Evaluator(args.universe, "pit", args.start, args.end)
    p = pd.read_parquet(PANEL)
    d = p.index.get_level_values("date")
    p = p.loc[(d >= pd.Timestamp(args.start)) & (d <= pd.Timestamp(args.end))].reset_index()
    p = p[(p["n_bars"] >= MIN_DATA_DAYS) & (p["n_feats"] >= 5)
          & (p["vol20_tl"] >= MIN_VOLUME_TL) & p["vol20_tl"].notna()]
    p = ev.restrict(p)

    fm = ev.fm.reindex(pd.MultiIndex.from_frame(p[["date", "ticker"]])).reset_index(drop=True)
    p = p.reset_index(drop=True)
    p["r"] = fm[f"fwd{H}"].to_numpy()
    p["bad"] = fm[f"bad{H}"].fillna(False).astype(bool).to_numpy()
    p = p[~p["bad"] & p["r"].notna() & p["ml_1g"].notna()]
    p["rk"] = p.groupby("date")["ml_1g"].rank(ascending=False, method="first")
    sel = p[p["rk"] <= N].copy()

    # günlük özet: top-N ortalama skoru ve o günün top-N getirisi
    day = sel.groupby("date").agg(day_score=("ml_1g", "mean"),
                                  day_top1=("ml_1g", "max"),
                                  day_ret=("r", "mean"), n=("r", "size")).reset_index()
    print(f"{len(day)} seans · top-{N} ortalama skoru: "
          f"min {day.day_score.min():.4f} · medyan {day.day_score.median():.4f} · "
          f"max {day.day_score.max():.4f}")
    print(f"  p10={day.day_score.quantile(.10):.4f}  p25={day.day_score.quantile(.25):.4f}  "
          f"p75={day.day_score.quantile(.75):.4f}  p90={day.day_score.quantile(.90):.4f}")

    out = {"universe": args.universe, "topn": N, "horizon": H,
           "window": {"start": args.start, "end": args.end},
           "day_score_quantiles": {q: round(float(day.day_score.quantile(q)), 4)
                                   for q in (.05, .10, .25, .50, .75, .90, .95)}}

    # ── bugünün yeri ──
    if args.today_score is not None:
        pct = float((day.day_score < args.today_score).mean() * 100)
        out["today"] = {"score": args.today_score, "percentile": round(pct, 1)}
        print(f"\nBUGÜN top-{N} ortalama skoru {args.today_score:.4f} → "
              f"tarihsel yüzdebirlik {pct:.1f}")

    # ── skor kovalarına göre ileri getiri ──
    print(f"\n── gün skoru kovalarına göre top-{N} getirisi ({H}g)")
    print(f"{'kova':>10s} {'gün':>5s} {'skor ort':>9s} {'getiri':>8s} {'GA alt':>8s} "
          f"{'GA üst':>8s} {'medyan':>8s}")
    day["q"] = pd.qcut(day["day_score"], 5, labels=["Q1 (en düşük)", "Q2", "Q3", "Q4", "Q5 (en yüksek)"])
    merged = sel.merge(day[["date", "q", "day_score"]], on="date")
    buckets = []
    for q, g in merged.groupby("q", observed=True):
        obs, lo, hi = block_bootstrap_mean(g, "r")
        rec = {"bucket": str(q), "n_days": int(g["date"].nunique()), "n": int(len(g)),
               "score_mean": float(g["day_score"].mean()),
               "mean": obs, "ci_lo": lo, "ci_hi": hi, "median": float(g["r"].median())}
        buckets.append(rec)
        print(f"{str(q):>10s} {rec['n_days']:5d} {rec['score_mean']:9.4f} {obs:8.2f} "
              f"{lo:8.2f} {hi:8.2f} {rec['median']:8.2f}")
    out["score_buckets"] = buckets

    # ── genişleyen pencere yüzdebirliği ile "zayıf gün" kuralı (look-ahead yok) ──
    print(f"\n── genişleyen-pencere yüzdebirliğine göre eleme kuralı")
    day = day.sort_values("date").reset_index(drop=True)
    exp_pct = np.full(len(day), np.nan)
    vals = day["day_score"].to_numpy()
    for i in range(MIN_HIST_DAYS, len(day)):
        exp_pct[i] = (vals[:i] < vals[i]).mean() * 100
    day["exp_pct"] = exp_pct
    merged2 = sel.merge(day[["date", "exp_pct"]], on="date")
    merged2 = merged2[merged2["exp_pct"].notna()]

    print(f"{'kural':>22s} {'gün':>5s} {'n':>6s} {'getiri':>8s} {'GA alt':>8s} {'GA üst':>8s}")
    rules = []
    for thr in (0, 10, 20, 25, 33, 50):
        g = merged2[merged2["exp_pct"] >= thr]
        obs, lo, hi = block_bootstrap_mean(g, "r")
        rnd = ev._random_baseline(g.groupby("date").size())[f"h{H}"]
        rec = {"skip_below_pct": thr, "n_days": int(g["date"].nunique()), "n": int(len(g)),
               "mean": obs, "ci_lo": lo, "ci_hi": hi,
               "rnd_mean": rnd["mean_mean"], "rnd_p95": rnd["mean_p95"],
               "edge": obs - rnd["mean_mean"]}
        rules.append(rec)
        lab = "eleme yok" if thr == 0 else f"alt %{thr} atla"
        print(f"{lab:>22s} {rec['n_days']:5d} {rec['n']:6,d} {obs:8.2f} {lo:8.2f} {hi:8.2f} "
              f"| rnd {rnd['mean_mean']:5.2f} fark {rec['edge']:+5.2f}")
    out["skip_rules"] = rules

    (outdir / f"results_top{N}.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results_top{N}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
