#!/usr/bin/env python3
"""Lindsay Trident + DE v1 tag'leri — endeks-kısıtlı backtest.

Advisor'ın Decision Engine v1 watchlist'inde gördüğü `trident_geo`,
`trident_tier1`, `trident_tag` alanlarının ürettiği kohortlar burada tarihsel
olarak yeniden kurulup ölçülür. Zincir:

  extfeed 1h master
    → tools/mb_scanner_events_run.py   (above_mb_birth olayları, aile başına)
    → tools/trident_probe_mb_3y.py     (A/B/C → D = B*C/A projeksiyonu + metrikler)
    → BURASI                            (kapılar + tag'ler + endeks kısıtı + ölçüm)

Kapılar (tools/decision_engine_v1_watchlist_generator.py PR-DE-3.18 ile birebir):
  G1  D_pct ∈ [20, 30)
  G2  structural_invalidation_pct_below_B ≤ son 365 günün %33'lük dilimi
  G3  bos_distance_atr_at_event         ≥ son 365 günün %67'lik dilimi
  G4  XU100 20 günlük getirisi > 0 (rejim yukarı)
  trident_geo   = G1 & G2 & G3          (G4'süz geometri — DE'de BİRİNCİL)
  trident_tier1 = trident_geo & G4

Tag'ler:
  D_PCT_30PLUS        D_pct ≥ 30
  SIL_DEEP            structural_invalidation_pct_below_B ≤ −10.0
  WEEKLY_BIRTH_ACTIVE aynı ticker'da bir ÖNCEKİ tamamlanmış haftada mb_1w/bb_1w
                      above_mb_birth (WEEKLY_CONFLUENCE_WEEKLY_FAMILIES)

Tag kapsamı üretimdeki gibi YALNIZCA `mb_1d` ailesidir (çapraz-TF yüzeyleme yok).

G2/G3 eşikleri **point-in-time**: her olay için yalnızca o olaydan KESİNLİKLE
önceki 365 günün mb_1d kohortu kullanılır (min n=30), üretimdeki trailing
tercile mantığıyla aynı.

Kullanım:
  python tools/index_bt_trident.py --universe XU100 --outdir output/index_bt/trident_xu100
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

from tools.index_bt_engine import Evaluator, XU_PATH  # noqa: E402

PROBE = ROOT / "output" / "trident_probe_mb_3y.parquet"

TIER1_FAMILY = "mb_1d"
D_PCT_LOW, D_PCT_HIGH = 20.0, 30.0
TRAILING_DAYS = 365
TRAILING_MIN_N = 30
SIL_DEEP_THRESHOLD = -10.0
D_PCT_30PLUS = 30.0
WEEKLY_FAMILIES = ("mb_1w", "bb_1w")


def _pit_terciles(ev: pd.DataFrame) -> pd.DataFrame:
    """Her olay tarihi için, o tarihten ÖNCEKİ 365 günün mb_1d kohortundan
    SIL p33 ve BoS p67. Aynı gün içindeki olaylar birbirini besleyemez."""
    ev = ev.sort_values("event_bar_date").reset_index(drop=True)
    dates = ev["event_bar_date"].to_numpy()
    sil = ev["structural_invalidation_pct_below_B"].to_numpy(dtype=float)
    bos = ev["bos_distance_atr_at_event"].to_numpy(dtype=float)

    uniq = pd.DatetimeIndex(sorted(pd.unique(dates)))
    lo_idx = np.searchsorted(dates, (uniq - pd.Timedelta(days=TRAILING_DAYS)).to_numpy(), side="left")
    hi_idx = np.searchsorted(dates, uniq.to_numpy(), side="left")   # strictly before

    p33 = {}
    p67 = {}
    for d, lo, hi in zip(uniq, lo_idx, hi_idx):
        n = hi - lo
        if n < TRAILING_MIN_N:
            p33[d], p67[d] = np.nan, np.nan
            continue
        s = sil[lo:hi]; b = bos[lo:hi]
        s = s[np.isfinite(s)]; b = b[np.isfinite(b)]
        p33[d] = float(np.quantile(s, 0.33)) if len(s) >= TRAILING_MIN_N else np.nan
        p67[d] = float(np.quantile(b, 0.67)) if len(b) >= TRAILING_MIN_N else np.nan

    ev["sil_p33"] = ev["event_bar_date"].map(p33)
    ev["bos_p67"] = ev["event_bar_date"].map(p67)
    return ev


def _xu100_regime() -> pd.Series:
    xu = pd.read_parquet(XU_PATH)["Close"]
    xu.index = pd.to_datetime(xu.index).tz_localize(None).normalize()
    xu = xu.sort_index()
    return (xu / xu.shift(20) - 1.0) > 0        # G4


def _weekly_birth_flags(probe: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    """(hafta_başlangıcı, ticker) — o haftadan SONRAKİ hafta için aktif sayılacak
    mb_1w/bb_1w above_mb_birth olayları."""
    w = probe[probe["family"].isin(WEEKLY_FAMILIES)].copy()
    w["week"] = w["event_bar_date"].dt.to_period("W-FRI")
    return {(p, t) for p, t in zip(w["week"], w["ticker"])}


def build_cohorts() -> pd.DataFrame:
    probe = pd.read_parquet(PROBE)
    probe["event_bar_date"] = pd.to_datetime(probe["event_bar_date"]).dt.normalize()

    weekly = _weekly_birth_flags(probe)

    ev = probe[probe["family"] == TIER1_FAMILY].copy()
    ev = _pit_terciles(ev)

    reg = _xu100_regime()
    ev["G4"] = ev["event_bar_date"].map(reg).fillna(False).astype(bool)

    d = ev["D_pct"].astype(float)
    ev["G1"] = (d >= D_PCT_LOW) & (d < D_PCT_HIGH)
    ev["G2"] = ev["structural_invalidation_pct_below_B"].astype(float) <= ev["sil_p33"]
    ev["G3"] = ev["bos_distance_atr_at_event"].astype(float) >= ev["bos_p67"]
    ev[["G2", "G3"]] = ev[["G2", "G3"]].fillna(False)

    ev["trident_geo"] = ev["G1"] & ev["G2"] & ev["G3"]
    ev["trident_tier1"] = ev["trident_geo"] & ev["G4"]
    ev["D_PCT_30PLUS"] = d >= D_PCT_30PLUS
    ev["SIL_DEEP"] = ev["structural_invalidation_pct_below_B"].astype(float) <= SIL_DEEP_THRESHOLD

    prev_week = ev["event_bar_date"].dt.to_period("W-FRI") - 1
    ev["WEEKLY_BIRTH_ACTIVE"] = [
        (p, t) in weekly for p, t in zip(prev_week, ev["ticker"])
    ]

    ev["mb_1d_birth"] = True
    ev = ev.rename(columns={"event_bar_date": "date"})
    return ev


COHORTS = ["mb_1d_birth", "trident_geo", "trident_tier1",
           "D_PCT_30PLUS", "SIL_DEEP", "WEEKLY_BIRTH_ACTIVE",
           "G1", "G2", "G3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit", choices=["pit", "current"])
    ap.add_argument("--start", default="2023-01-13", help="mb_scanner olay penceresi başlangıcı")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ev = build_cohorts()
    print(f"mb_1d above_mb_birth olayları: {len(ev):,} · "
          f"{ev['date'].min().date()} → {ev['date'].max().date()}")

    e = Evaluator(args.universe, args.mode, args.start, args.end)
    print(f"evren {args.universe}/{args.mode}: {len(e.tickers)} ticker, {len(e.uni):,} evren-gün")

    ev_u = e.restrict(ev)
    print(f"evrene indirgenmiş: {len(ev_u):,}")

    out = {"scan": "trident", "universe": args.universe, "mode": args.mode,
           "window": {"start": args.start, "end": args.end},
           "gate_params": {"D_pct": [D_PCT_LOW, D_PCT_HIGH],
                           "trailing_days": TRAILING_DAYS, "min_n": TRAILING_MIN_N,
                           "sil_deep": SIL_DEEP_THRESHOLD},
           "cohorts": {}}

    for name in COHORTS:
        pairs = ev_u.loc[ev_u[name].fillna(False), ["date", "ticker"]].drop_duplicates()
        if not len(pairs):
            out["cohorts"][name] = {"n_signals": 0}
            continue
        print(f"── kohort {name}: {len(pairs):,}")
        blk = e.layer_a(pairs)
        tr = e.layer_b(pairs)
        blk["trades"] = e.trade_stats(tr)
        if len(tr):
            tr.assign(cohort=name).to_parquet(outdir / f"trades_{name}.parquet", index=False)
        out["cohorts"][name] = blk

    # üretimdeki D-hedefine ulaşma oranı (probe metriği) — kısıtlı evrende
    sub = ev_u[ev_u["trident_geo"].fillna(False)]
    if len(sub):
        out["D_touch_rate_geo"] = {
            f"{h}d": float(sub[f"D_touched_{h}d"].mean())
            for h in (5, 10, 20, 30) if f"D_touched_{h}d" in sub.columns
        }
    base = ev_u
    out["D_touch_rate_all"] = {
        f"{h}d": float(base[f"D_touched_{h}d"].mean())
        for h in (5, 10, 20, 30) if f"D_touched_{h}d" in base.columns
    }

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
