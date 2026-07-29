#!/usr/bin/env python3
"""ML top-N tarayıcısı hangi evrende koşmalı? — XU100 vs XU100\\XU030 vs XU030.

Soru: BIST-30 isimlerinde edge bulunamadığına göre, tarayıcı evreninden onları
çıkarmak (yani XU100'ün "orta 70"inde koşmak) daha mı iyi?

İki farklı test, ikisi de gerekli:

  1. AYRI EVREN. Her evrende BAĞIMSIZ top-N seçimi + o evrenin kendi rastgele
     tabanı. "XU100'de top-4" ile "orta-70'te top-4" farklı hisseler seçer.

  2. KOŞULLU AYRIŞTIRMA. XU100 top-N seçiminin İÇİNDE, seçilen hisse BIST-30
     üyesi mi değil mi diye ayır. Bu, evreni değiştirmeden "BIST-30 isimleri
     seçildiklerinde kötü mü performans veriyor" sorusunu yanıtlar — ki asıl
     merak edilen bu.

Güven aralıkları gün-bloklu bootstrap.

Kullanım:
  python tools/index_bt_ml_universe.py --topn 4 --outdir output/index_bt/ml_universe
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

from tools.index_bt_engine import Evaluator                       # noqa: E402
from tools.index_bt_ml_threshold import block_bootstrap_mean      # noqa: E402
from tools.index_universe import load_membership                  # noqa: E402
from alpha.config import MIN_VOLUME_TL, MIN_DATA_DAYS             # noqa: E402

PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"
SEGMENTS = [("model_ici", "2021-07-01", "2025-06-30"),
            ("val_2025H2", "2025-07-01", "2025-12-31"),
            ("oos_2026", "2026-01-01", "2026-07-14")]


def load_panel(start, end):
    p = pd.read_parquet(PANEL)
    d = p.index.get_level_values("date")
    p = p.loc[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index()
    return p[(p["n_bars"] >= MIN_DATA_DAYS) & (p["n_feats"] >= 5)
             & (p["vol20_tl"] >= MIN_VOLUME_TL) & p["vol20_tl"].notna()]


def evaluate(ev, panel, topn, H, label, n_random=20):
    """Evren içinde top-N seç, ileri getiri + rastgele taban."""
    p = ev.restrict(panel).copy()
    fm = ev.fm.reindex(pd.MultiIndex.from_frame(p[["date", "ticker"]])).reset_index(drop=True)
    p = p.reset_index(drop=True)
    p["r"] = fm[f"fwd{H}"].to_numpy()
    p["bad"] = fm[f"bad{H}"].fillna(False).astype(bool).to_numpy()
    p = p[~p["bad"] & p["r"].notna() & p["ml_1g"].notna()]
    p["rk"] = p.groupby("date")["ml_1g"].rank(ascending=False, method="first")
    sel = p[p["rk"] <= topn]
    obs, lo, hi = block_bootstrap_mean(sel, "r")
    rnd = ev._random_baseline(sel.groupby("date").size())[f"h{H}"]
    return {"label": label, "n": int(len(sel)), "n_days": int(sel["date"].nunique()),
            "mean": obs, "ci_lo": lo, "ci_hi": hi,
            "median": float(sel["r"].median()),
            "rnd_mean": rnd["mean_mean"], "rnd_p95": rnd["mean_p95"],
            "beats_p95": bool(obs > rnd["mean_p95"])}, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    H, N = args.horizon, args.topn
    panel = load_panel(args.start, args.end)
    out = {"topn": N, "horizon": H, "window": {"start": args.start, "end": args.end}}

    # ── 1. ayrı evrenler ──
    print(f"── 1. AYRI EVREN — her evrende bağımsız top-{N} seçimi ({H}g)")
    print(f"{'evren':>14s} {'n':>6s} {'gün':>5s} {'ort':>7s} {'GA alt':>7s} {'GA üst':>7s} "
          f"{'med':>6s} {'RND':>6s} {'RNDp95':>7s} {'✔':>2s}")
    sels = {}
    rows = []
    for uni, lab in (("XU100", "XU100 (100)"),
                     ("XU100EX30", "XU100\\XU030 (70)"),
                     ("XU030", "XU030 (30)")):
        ev = Evaluator(uni, "pit", args.start, args.end)
        rec, sel = evaluate(ev, panel, N, H, lab)
        sels[uni] = (ev, sel)
        rows.append(rec)
        print(f"{lab:>14s} {rec['n']:6,d} {rec['n_days']:5d} {rec['mean']:7.2f} "
              f"{rec['ci_lo']:7.2f} {rec['ci_hi']:7.2f} {rec['median']:6.2f} "
              f"{rec['rnd_mean']:6.2f} {rec['rnd_p95']:7.2f} {'✔' if rec['beats_p95'] else '':>2s}")
    out["separate_universes"] = rows

    # ── 2. koşullu ayrıştırma: XU100 top-N içinde BIST-30 üyesi olanlar ──
    print(f"\n── 2. KOŞULLU — XU100 top-{N} seçiminin içinde BIST-30 üyeliğine göre")
    memb = load_membership()
    k30 = set(zip(memb.loc[memb["in_xu030"], "date"], memb.loc[memb["in_xu030"], "ticker"]))
    _, sel100 = sels["XU100"]
    is30 = np.fromiter(((d, t) in k30 for d, t in zip(sel100["date"], sel100["ticker"])),
                       dtype=bool, count=len(sel100))
    sel100 = sel100.assign(is_xu030=is30)
    cond = []
    for flag, lab in ((True, "BIST-30 üyesi"), (False, "BIST-30 DIŞI")):
        g = sel100[sel100["is_xu030"] == flag]
        obs, lo, hi = block_bootstrap_mean(g, "r")
        rec = {"grup": lab, "n": int(len(g)), "pay_pct": round(100 * len(g) / len(sel100), 1),
               "mean": obs, "ci_lo": lo, "ci_hi": hi, "median": float(g["r"].median())}
        cond.append(rec)
        print(f"  {lab:>16s} n={rec['n']:5,d} (%{rec['pay_pct']:.1f}) ort={obs:7.2f} "
              f"GA=[{lo:6.2f},{hi:6.2f}] med={rec['median']:6.2f}")
    out["conditional_within_xu100"] = cond

    # ── 3. dönem kırılımı ──
    print(f"\n── 3. dönem kırılımı (ayrı evrenler)")
    seg = {}
    for name, s, e in SEGMENTS:
        st, en = pd.Timestamp(s), pd.Timestamp(e)
        seg[name] = {}
        print(f"  [{name}]")
        for uni, lab in (("XU100", "XU100"), ("XU100EX30", "XU100\\30"), ("XU030", "XU030")):
            _, sel = sels[uni]
            g = sel[(sel["date"] >= st) & (sel["date"] <= en)]
            obs, lo, hi = block_bootstrap_mean(g, "r", n_boot=800)
            seg[name][lab] = {"n": int(len(g)), "mean": obs, "ci_lo": lo, "ci_hi": hi}
            print(f"    {lab:10s} n={len(g):5,d} ort={obs:7.2f} GA=[{lo:6.2f},{hi:6.2f}]")
    out["segments"] = seg

    # ── 4. örtüşme: iki evrenin seçimleri ne kadar aynı ──
    _, s70 = sels["XU100EX30"]
    a = set(zip(sel100["date"], sel100["ticker"]))
    b = set(zip(s70["date"], s70["ticker"]))
    out["overlap"] = {"xu100_topN": len(a), "mid70_topN": len(b),
                      "ortak": len(a & b),
                      "ortak_pct_of_mid70": round(100 * len(a & b) / max(1, len(b)), 1)}
    print(f"\n── 4. örtüşme: XU100 top-{N} ∩ orta70 top-{N} = {len(a & b):,} "
          f"(orta70'in %{out['overlap']['ortak_pct_of_mid70']}'i)")

    (outdir / f"results_top{N}.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results_top{N}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
