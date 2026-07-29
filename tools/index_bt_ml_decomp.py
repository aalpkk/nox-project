#!/usr/bin/env python3
"""ML katkısının ayrıştırılması — alpha ve tavan_v1 aynı skoru mu satıyor?

Soru: BIST-100 testinden sağ çıkan iki hat (alpha top-4 ve tavan_v1) aynı
MLScorer `up_1g universe` çıktısına dayanıyor (alpha eşiği 0.48, tavan_v1
eşiği 0.65). Dolayısıyla bunlar İKİ bağımsız kanıt değil, TEK modelin iki
farklı olay kümesine uygulanmış hâli olabilir.

Bu araç üç şeyi ölçer:
  1. Çıplak ML eşiği — hiçbir tarama mantığı olmadan `ml_1g >= X` kohortu.
     Tarama katmanları ML'in üstüne değer katıyor mu?
  2. ml_1g desilleri — skorun kendisi monoton mu?
  3. Model-içi / model-dışı ayrımı — modeller ~2025-06'ya kadar eğitildi;
     eşik kohortlarının kenarı OOS'ta duruyor mu?
  4. Örtüşme — tavan_v1 sinyalleri zaten alpha adayı mı?

Kullanım:
  python tools/index_bt_ml_decomp.py --universe XU100 --outdir output/index_bt/ml_decomp_xu100
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

from tools.index_bt_engine import Evaluator, load_prices_long  # noqa: E402
from tools.index_bt_misc import tavan_signals                  # noqa: E402
from alpha.config import (                                     # noqa: E402
    ML_SCORE_THRESHOLD, ML_SWING_THRESHOLD, ML_SLOPE_MIN,
    CONFIRMATION_MIN_SCORE, MIN_VOLUME_TL, MIN_DATA_DAYS, ML_COMPOSITE_WEIGHT,
)

PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"

# Modeller ~2025-06'ya kadar eğitildi (alpha backtest raporunun segment adları).
SEGMENTS = [
    ("model_ici_2021H2_2025H1", "2021-07-01", "2025-06-30"),
    ("val_2025H2", "2025-07-01", "2025-12-31"),
    ("oos_2026", "2026-01-01", "2026-07-14"),
]


def alpha_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    """scripts/alpha_scan_backtest_sim.py::make_candidates ile aynı kapılar."""
    p = panel
    eligible = ((p["n_bars"] >= MIN_DATA_DAYS) & (p["n_feats"] >= 5)
                & (p["vol20_tl"] >= MIN_VOLUME_TL) & p["vol20_tl"].notna())
    slope_ok = p["ml_1g_lag5"].isna() | ((p["ml_1g"] - p["ml_1g_lag5"]) >= ML_SLOPE_MIN)
    cand = (eligible
            & p["ml_1g"].notna() & (p["ml_1g"] >= ML_SCORE_THRESHOLD)
            & (p["ml_3g"].isna() | (p["ml_3g"] >= ML_SWING_THRESHOLD))
            & slope_ok & (p["conf_score"] >= CONFIRMATION_MIN_SCORE))
    out = p.copy()
    out["eligible"] = eligible
    out["is_candidate"] = cand
    ml_avg = np.where(out["ml_3g"].isna(), out["ml_1g"], (out["ml_1g"] + out["ml_3g"]) / 2.0)
    out["composite"] = np.minimum(
        100.0, ml_avg * 100 * ML_COMPOSITE_WEIGHT
        + out["conf_score"] * 10 * (1 - ML_COMPOSITE_WEIGHT)).round(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit")
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(args.universe, args.mode, args.start, args.end)
    print(f"evren {args.universe}/{args.mode}: {len(ev.tickers)} ticker, {len(ev.uni):,} evren-gün")

    panel = pd.read_parquet(PANEL)
    panel = panel.loc[(panel.index.get_level_values("date") >= pd.Timestamp(args.start))
                      & (panel.index.get_level_values("date") <= pd.Timestamp(args.end))]
    cand = alpha_candidates(panel).reset_index()
    cand = ev.restrict(cand)
    cand["rank_in_day"] = cand[cand["is_candidate"]].groupby("date")["composite"] \
                              .rank(ascending=False, method="first")
    print(f"panel (evren içi): {len(cand):,} · aday {int(cand.is_candidate.sum()):,}")

    tv = ev.restrict(tavan_signals())
    tv_v1 = tv[tv["ml_s"].fillna(-1) >= 0.65]
    print(f"tavan {len(tv):,} · tavan_v1 {len(tv_v1):,}")

    out = {"universe": args.universe, "mode": args.mode,
           "window": {"start": args.start, "end": args.end}, "cohorts": {}, "segments": {}}

    def run(name, pairs, store=out["cohorts"]):
        pairs = pairs[["date", "ticker"]].drop_duplicates()
        if len(pairs) < 5:
            store[name] = {"n_signals": len(pairs)}
            return
        blk = ev.layer_a(pairs)
        tr = ev.layer_b(pairs)
        blk["trades"] = ev.trade_stats(tr)
        store[name] = blk
        s = blk["signal_h5"]; r = blk["random"]["h5"]
        flag = "*" if s["mean"] > r["mean_p95"] else " "
        print(f"  {name:34s}{flag}n={s['n']:6,d} ort={s['mean']:6.2f} rnd={r['mean_mean']:6.2f} "
              f"p95={r['mean_p95']:6.2f} | t={blk['trades'].get('mean',0):6.2f} "
              f"med={blk['trades'].get('median',0):6.2f} PF={blk['trades'].get('pf',0):5.2f}")

    # ── 1. çıplak ML eşikleri (hiçbir tarama mantığı yok) ──
    print("\n── 1. çıplak ML eşiği (sadece likidite kapısı + ml_1g >= X)")
    liq = cand[cand["eligible"]]
    for thr in (0.48, 0.55, 0.60, 0.65, 0.70):
        run(f"ml_1g>={thr:.2f}_ciplak", liq[liq["ml_1g"] >= thr])

    # ── 2. tarama katmanlarının ML üstüne katkısı ──
    print("\n── 2. tarama katmanları (aynı ML eşiğinde)")
    run("alpha_aday (ml>=0.48 + kapılar)", cand[cand["is_candidate"]])
    run("alpha_top4", cand[cand["rank_in_day"] <= 4])
    run("tavan_v1 (tavan + ml>=0.65)", tv_v1)
    hi = liq[liq["ml_1g"] >= 0.65]
    run("ml>=0.65 + tavan DEĞİL", hi.merge(tv_v1[["date", "ticker"]].assign(_t=1),
                                           on=["date", "ticker"], how="left")
                                     .query("_t.isna()"))

    # ── 3. ml_1g desilleri (evren içi, tarama yok) ──
    print("\n── 3. ml_1g desilleri (5g ortalama)")
    li = pd.MultiIndex.from_frame(liq[["date", "ticker"]].drop_duplicates())
    fm = ev.fm.reindex(li)
    dec = pd.DataFrame({"ml": liq.drop_duplicates(["date", "ticker"])["ml_1g"].to_numpy(),
                        "r5": fm["fwd5"].to_numpy()}).dropna()
    q = pd.qcut(dec["ml"], 10, labels=False, duplicates="drop")
    out["ml_deciles_h5"] = {str(int(k)): round(float(g["r5"].mean()), 3)
                            for k, g in dec.groupby(q)}
    print("  ", out["ml_deciles_h5"])

    # ── 4. model-içi / model-dışı ──
    print("\n── 4. dönem kırılımı (modeller ~2025-06'ya kadar eğitildi)")
    for seg, s, e in SEGMENTS:
        st, en = pd.Timestamp(s), pd.Timestamp(e)
        out["segments"][seg] = {}
        print(f"  [{seg}]")
        for nm, df in (("alpha_top4", cand[cand["rank_in_day"] <= 4]),
                       ("tavan_v1", tv_v1),
                       ("ml>=0.65_ciplak", liq[liq["ml_1g"] >= 0.65])):
            sub = df[(df["date"] >= st) & (df["date"] <= en)]
            run(f"    {nm}", sub, store=out["segments"][seg])

    # ── 5. örtüşme ──
    print("\n── 5. örtüşme")
    key = lambda d: set(zip(d["date"], d["ticker"]))
    a_cand, a_top4, t_v1 = key(cand[cand["is_candidate"]]), key(cand[cand["rank_in_day"] <= 4]), key(tv_v1)
    ov = {
        "tavan_v1_n": len(t_v1),
        "tavan_v1_alpha_adayi_da": len(t_v1 & a_cand),
        "tavan_v1_alpha_top4_de": len(t_v1 & a_top4),
        "alpha_top4_n": len(a_top4),
        "alpha_top4_tavan_v1_de": len(a_top4 & t_v1),
    }
    ov["tavan_v1_alpha_aday_pct"] = round(100 * ov["tavan_v1_alpha_adayi_da"] / max(1, len(t_v1)), 1)
    ov["alpha_top4_tavan_pct"] = round(100 * ov["alpha_top4_tavan_v1_de"] / max(1, len(a_top4)), 1)
    out["overlap"] = ov
    print("  ", json.dumps(ov, ensure_ascii=False))

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
