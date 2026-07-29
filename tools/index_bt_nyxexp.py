#!/usr/bin/env python3
"""Nyxpansion (nyxexp) günlük tarama — endeks-kısıtlı tarihsel backtest.

Canlı tarama (nyxexpansion/tools/scan_latest.py) her gün için modeli YENİDEN
eğitiyor: train = CONFIG.split.train_start → (hedef − 195g), val = son 6 ay
(hedef − 195g … hedef − 15g), 15 günlük embargo (10-bar etiket ufkundan büyük).
Burada aynı protokol AYLIK adımlarla yürütülür: her ay için o ayın olayları
`qt`, eğitim penceresi ayın başına göre kurulur → tarihsel skorlar
look-ahead'siz üretilir.

Ölçülen kohortlar (hepsi XU100 / XU030 üyeliğine indirgenmiş):
  events   — taramanın ham tetik olayları (aday evreni)
  scored   — range-veto sonrası skorlanan adaylar (production filtresi)
  topN     — gün içi winner_R_pred sıralamasında ilk N (yayınlanan kart)

Kullanım:
  python tools/index_bt_nyxexp.py --universe XU100 --outdir output/index_bt/nyxexp_xu100
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

from nyxexpansion.config import CONFIG                       # noqa: E402
from nyxexpansion.train_winner import _add_winner_R          # noqa: E402
from nyxexpansion.tools.scan_latest import train_clf, train_winner  # noqa: E402
from tools.index_bt_engine import Evaluator                  # noqa: E402

DATASET = ROOT / "output" / "nyxexp_dataset_v4.parquet"
SCORE_CACHE = ROOT / "output" / "index_bt" / "_nyxexp_walkforward_scores.parquet"

EMBARGO_D = 15
VAL_WINDOW_D = 180


def walk_forward_scores(df: pd.DataFrame, score_start: str, score_end: str) -> pd.DataFrame:
    """Aylık yeniden eğitimle out-of-sample winner_R_pred / p_l3 üret."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    months = pd.period_range(pd.Timestamp(score_start), pd.Timestamp(score_end), freq="M")
    out = []
    for per in months:
        m_start, m_end = per.start_time, per.end_time
        qt = df[(df["date"] >= m_start) & (df["date"] <= m_end)]
        if not len(qt):
            continue
        val_end = m_start - pd.Timedelta(days=EMBARGO_D)
        val_start = val_end - pd.Timedelta(days=VAL_WINDOW_D)
        train_end = val_start - pd.Timedelta(days=EMBARGO_D)
        train_start = CONFIG.split.train_start
        if pd.Timestamp(train_start) >= train_end:
            continue
        print(f"── {per}  qt={len(qt):,}  train→{train_end.date()}  val {val_start.date()}→{val_end.date()}")
        try:
            p = train_clf(df, qt, train_start, train_end.strftime("%Y-%m-%d"),
                          val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d"))
            w = train_winner(df, qt, train_start, train_end.strftime("%Y-%m-%d"),
                             val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"   [atla] {type(e).__name__}: {e}")
            continue
        out.append(pd.DataFrame({
            "date": qt["date"].to_numpy(), "ticker": qt["ticker"].to_numpy(),
            "xu_regime": qt["xu_regime"].to_numpy(),
            "p_l3": p.to_numpy(), "winner_R_pred": w.to_numpy(),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit", choices=["pit", "current"])
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-04-24")
    ap.add_argument("--score-start", default="2022-01-01",
                    help="walk-forward skorlamanın başlangıcı (öncesi eğitim penceresi)")
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = _add_winner_R(df, h=CONFIG.label.primary_h)
    print(f"dataset: {len(df):,} olay · {df.date.min().date()} → {df.date.max().date()}")

    if SCORE_CACHE.exists():
        scores = pd.read_parquet(SCORE_CACHE)
        scores["date"] = pd.to_datetime(scores["date"])
        print(f"skor önbelleği: {len(scores):,} satır")
    else:
        scores = walk_forward_scores(df, args.score_start, args.end)
        scores.to_parquet(SCORE_CACHE, index=False)
        print(f"walk-forward skor: {len(scores):,} satır → {SCORE_CACHE}")

    ev = Evaluator(args.universe, args.mode, args.start, args.end)
    print(f"evren {args.universe}/{args.mode}: {len(ev.tickers)} ticker, {len(ev.uni):,} evren-gün")

    out = {"universe": args.universe, "mode": args.mode,
           "window": {"start": args.start, "end": args.end},
           "score_window": {"start": args.score_start, "end": args.end},
           "cohorts": {}}

    def run(name, pairs):
        pairs = pairs.drop_duplicates()
        if not len(pairs):
            out["cohorts"][name] = {"n_signals": 0}
            return
        print(f"── kohort {name}: {len(pairs):,}")
        blk = ev.layer_a(pairs)
        tr = ev.layer_b(pairs)
        blk["trades"] = ev.trade_stats(tr)
        if len(tr):
            tr.assign(cohort=name).to_parquet(outdir / f"trades_{name}.parquet", index=False)
        out["cohorts"][name] = blk

    # 1) ham tetik olayları (tüm pencere)
    ev_pairs = ev.restrict(df[["date", "ticker"]])
    run("events", ev_pairs)

    # 2) production filtreli skorlanmış adaylar (range veto + skor NaN değil)
    sc = scores[(scores["xu_regime"] != "range") & scores["winner_R_pred"].notna()].copy()
    sc_u = ev.restrict(sc)
    run("scored", sc_u[["date", "ticker"]])

    # 3) gün içi top-N (yayınlanan kart)
    if len(sc_u):
        sc_u = sc_u.sort_values(["date", "winner_R_pred"], ascending=[True, False])
        top = sc_u.groupby("date").head(args.topn)
        run(f"top{args.topn}", top[["date", "ticker"]])
        # skor desilleri (aday içi ayrıştırma, 5 günlük)
        si = pd.MultiIndex.from_frame(sc_u[["date", "ticker"]].drop_duplicates())
        fm = ev.fm.reindex(si)
        d5 = pd.DataFrame({"r": fm["fwd5"].to_numpy(),
                           "s": sc_u.drop_duplicates(["date", "ticker"])["winner_R_pred"].to_numpy()})
        d5 = d5.dropna()
        if len(d5) > 100:
            q = pd.qcut(d5["s"], 5, labels=False, duplicates="drop")
            out["score_quintiles_h5"] = {str(int(k)): float(g["r"].mean())
                                         for k, g in d5.groupby(q)}

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
