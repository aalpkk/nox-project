#!/usr/bin/env python3
"""ML eşik duyarlılığı teşhisi — 0.70'teki kırılma gürültü mü, yapı mı?

§6'daki bulgu: `ml_1g >= 0.65` güçlü (+4.16), `>= 0.70` çöküyor (+1.76, işlem
medyanı −6.17). Bir eşiği "seçmeden" önce bu kırılmanın kaynağını ayırmak gerek.

Dört ayrı teşhis:

  A. AYRIK kova (disjoint bucket). Kümülatif eşikler iç içe geçtiği için
     ">=0.65 iyi, >=0.70 kötü" ifadesi tek başına yanıltıcı: >=0.65'in içindeki
     tüm değeri [0.65,0.70) dilimi taşıyor olabilir. Ayrık kovalar bunu ayırır.

  B. Blok bootstrap güven aralığı. Aynı gün içindeki sinyaller çapraz-kesitte
     korele; 5 günlük örtüşen pencereler de zaman serisinde korele. Bu yüzden
     GÜN bazında yeniden örnekleme yapılır (i.i.d. sinyal varsayımı değil).

  C. Dağılım kayması. `ml_1g` mutlak bir eşiktir ama skorun dağılımı zamanla
     kayarsa aynı sayı farklı şey ifade eder. Yıl bazında eşik-üstü sinyal
     sayısı ve skor yüzdebirlikleri buna bakar.

  D. Kovalama (chase) hipotezi. En yüksek skorlar zaten patlamış hisselere mi
     düşüyor? Sinyal günü getirisi ve EMA21 uzaklığı kova bazında raporlanır.

Ayrıca çapraz-kesit ALTERNATİFİ ölçülür: mutlak eşik yerine gün içi top-N /
top-q%. Bu, gün başına sinyal sayısını sabitler ve dağılım kaymasına bağışıktır.

Kullanım:
  python tools/index_bt_ml_threshold.py --universe XU100 --outdir output/index_bt/ml_thr_xu100
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

from tools.index_bt_engine import Evaluator  # noqa: E402
from alpha.config import MIN_VOLUME_TL, MIN_DATA_DAYS  # noqa: E402

PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"

BUCKETS = [(0.40, 0.48), (0.48, 0.55), (0.55, 0.60),
           (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
CUM_THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]
TOPN_GRID = [1, 2, 3, 4, 6, 8, 12, 20]

SEGMENTS = [("model_ici", "2021-07-01", "2025-06-30"),
            ("val_2025H2", "2025-07-01", "2025-12-31"),
            ("oos_2026", "2026-01-01", "2026-07-14")]


def block_bootstrap_mean(df: pd.DataFrame, col: str, n_boot: int = 2000,
                         seed: int = 7) -> tuple[float, float, float]:
    """Gün-blokları yeniden örneklenerek ortalama için %95 GA.

    Aynı gün içindeki sinyaller korele olduğundan sinyal seviyesinde değil GÜN
    seviyesinde örnekleme yapılır (blok = bir seans).
    """
    if not len(df):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby("date")]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if len(groups) < 5:
        return (float(np.concatenate(groups).mean()) if groups else np.nan, np.nan, np.nan)
    k = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        means[b] = np.concatenate([groups[i] for i in idx]).mean()
    obs = float(np.concatenate(groups).mean())
    return obs, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit")
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    H = args.horizon
    ev = Evaluator(args.universe, args.mode, args.start, args.end, n_random=20)

    panel = pd.read_parquet(PANEL)
    panel = panel.loc[(panel.index.get_level_values("date") >= pd.Timestamp(args.start))
                      & (panel.index.get_level_values("date") <= pd.Timestamp(args.end))]
    p = panel.reset_index()
    p = p[(p["n_bars"] >= MIN_DATA_DAYS) & (p["n_feats"] >= 5)
          & (p["vol20_tl"] >= MIN_VOLUME_TL) & p["vol20_tl"].notna()]
    p = ev.restrict(p)

    fm = ev.fm.reindex(pd.MultiIndex.from_frame(p[["date", "ticker"]])).reset_index(drop=True)
    p = p.reset_index(drop=True)
    p["r"] = fm[f"fwd{H}"].to_numpy()
    p["bad"] = fm[f"bad{H}"].fillna(False).astype(bool).to_numpy()
    p = p[~p["bad"] & p["r"].notna() & p["ml_1g"].notna()].copy()
    p["sig_day_ret"] = p.groupby("ticker")["close"].pct_change() * 100
    print(f"evren {args.universe}: {len(p):,} likit evren-gün · "
          f"{p['date'].nunique()} seans · ufuk {H}g")

    out = {"universe": args.universe, "horizon": H,
           "window": {"start": args.start, "end": args.end}}

    # ── A + B: ayrık kovalar + blok bootstrap ──
    print(f"\n── A/B. AYRIK ml_1g kovaları ({H}g getiri, %95 blok-bootstrap GA)")
    print(f"{'kova':>14s} {'n':>7s} {'gün':>5s} {'ort':>7s} {'GA alt':>7s} {'GA üst':>7s} "
          f"{'med':>7s} {'sinyalgünü ret':>14s} {'EMA21 uzak':>10s}")
    rows = []
    for lo, hi in BUCKETS:
        b = p[(p["ml_1g"] >= lo) & (p["ml_1g"] < hi)]
        obs, ci_lo, ci_hi = block_bootstrap_mean(b, "r")
        rec = {"bucket": f"[{lo:.2f},{hi:.2f})", "n": int(len(b)),
               "n_days": int(b["date"].nunique()),
               "mean": obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "median": float(b["r"].median()) if len(b) else np.nan,
               "sig_day_ret_mean": float(b["sig_day_ret"].mean()) if len(b) else np.nan}
        rows.append(rec)
        print(f"{rec['bucket']:>14s} {rec['n']:7,d} {rec['n_days']:5d} {obs:7.2f} "
              f"{ci_lo:7.2f} {ci_hi:7.2f} {rec['median']:7.2f} "
              f"{rec['sig_day_ret_mean']:14.2f} {'':>10s}")
    out["disjoint_buckets"] = rows

    # ── kümülatif eşik profili (plato araması) ──
    print(f"\n── kümülatif eşik profili (plato var mı?)")
    print(f"{'eşik':>6s} {'n':>7s} {'ort':>7s} {'GA alt':>7s} {'GA üst':>7s} {'med':>7s} {'gün/yıl':>8s}")
    cum = []
    n_years = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days / 365.25
    for thr in CUM_THRESHOLDS:
        b = p[p["ml_1g"] >= thr]
        obs, ci_lo, ci_hi = block_bootstrap_mean(b, "r")
        rec = {"thr": thr, "n": int(len(b)), "mean": obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "median": float(b["r"].median()) if len(b) else np.nan,
               "signals_per_year": round(len(b) / n_years, 1)}
        cum.append(rec)
        print(f"{thr:6.2f} {rec['n']:7,d} {obs:7.2f} {ci_lo:7.2f} {ci_hi:7.2f} "
              f"{rec['median']:7.2f} {rec['signals_per_year']:8.1f}")
    out["cumulative_thresholds"] = cum

    # ── C: dağılım kayması ──
    print(f"\n── C. dağılım kayması — yıl bazında ml_1g yüzdebirlikleri ve eşik-üstü sayı")
    yr = p.groupby(p["date"].dt.year)
    drift = []
    print(f"{'yıl':>5s} {'evren-gün':>10s} {'p50':>6s} {'p90':>6s} {'p99':>6s} "
          f"{'>=0.60':>7s} {'>=0.65':>7s} {'>=0.70':>7s}")
    for y, g in yr:
        rec = {"year": int(y), "n": int(len(g)),
               "p50": float(g["ml_1g"].quantile(0.50)),
               "p90": float(g["ml_1g"].quantile(0.90)),
               "p99": float(g["ml_1g"].quantile(0.99)),
               "n_ge_060": int((g["ml_1g"] >= 0.60).sum()),
               "n_ge_065": int((g["ml_1g"] >= 0.65).sum()),
               "n_ge_070": int((g["ml_1g"] >= 0.70).sum())}
        drift.append(rec)
        print(f"{y:5d} {rec['n']:10,d} {rec['p50']:6.3f} {rec['p90']:6.3f} {rec['p99']:6.3f} "
              f"{rec['n_ge_060']:7,d} {rec['n_ge_065']:7,d} {rec['n_ge_070']:7,d}")
    out["distribution_drift"] = drift

    # ── çapraz-kesit alternatifi: gün içi top-N ──
    print(f"\n── çapraz-kesit alternatifi: gün içi ml_1g top-N (dağılım kaymasına bağışık)")
    print(f"{'N':>4s} {'n':>7s} {'ort':>7s} {'GA alt':>7s} {'GA üst':>7s} {'med':>7s} "
          f"{'RND':>6s} {'RNDp95':>7s} {'✔':>2s}")
    p = p.sort_values(["date", "ml_1g"], ascending=[True, False])
    p["rank_in_day"] = p.groupby("date")["ml_1g"].rank(ascending=False, method="first")
    topn = []
    for N in TOPN_GRID:
        b = p[p["rank_in_day"] <= N]
        obs, ci_lo, ci_hi = block_bootstrap_mean(b, "r")
        rnd = ev._random_baseline(b.groupby("date").size())[f"h{H}"]
        rec = {"N": N, "n": int(len(b)), "mean": obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "median": float(b["r"].median()),
               "rnd_mean": rnd["mean_mean"], "rnd_p95": rnd["mean_p95"],
               "beats_p95": bool(obs > rnd["mean_p95"])}
        topn.append(rec)
        print(f"{N:4d} {rec['n']:7,d} {obs:7.2f} {ci_lo:7.2f} {ci_hi:7.2f} {rec['median']:7.2f} "
              f"{rnd['mean_mean']:6.2f} {rnd['mean_p95']:7.2f} {'✔' if rec['beats_p95'] else '':>2s}")
    out["cross_sectional_topN"] = topn

    # ── segment bazında hem eşik hem top-N ──
    print(f"\n── dönem kırılımı (eşik vs top-N)")
    seg_out = {}
    for name, s, e in SEGMENTS:
        st, en = pd.Timestamp(s), pd.Timestamp(e)
        sub = p[(p["date"] >= st) & (p["date"] <= en)]
        blk = {}
        print(f"  [{name}]")
        for label, sel in ([(f"ml>={t}", sub[sub["ml_1g"] >= t]) for t in (0.60, 0.65, 0.70)]
                           + [(f"top{N}", sub[sub["rank_in_day"] <= N]) for N in (4, 8)]):
            obs, ci_lo, ci_hi = block_bootstrap_mean(sel, "r", n_boot=800)
            blk[label] = {"n": int(len(sel)), "mean": obs, "ci_lo": ci_lo, "ci_hi": ci_hi}
            print(f"    {label:10s} n={len(sel):6,d} ort={obs:7.2f} GA=[{ci_lo:6.2f},{ci_hi:6.2f}]")
        seg_out[name] = blk
    out["segments"] = seg_out

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
