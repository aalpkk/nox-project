#!/usr/bin/env python3
"""Tavan tarayıcı ve HyperWave OB/OS tarayıcı — endeks-kısıtlı backtest.

  tavan   — briefing'in "Tavan Scanner / Tavan Kandidat" listesi.
            Canlı mantık (tools/tavan_scan_live.py): gün kapanışı önceki
            kapanışın ~%110'u (±%0.5 tolerans) ise "tavanda", ardından
            V1 filtresi ML_S >= 0.65. ML_S = MLScorer up_1g universe skoru =
            alpha panelindeki `ml_1g` kolonu → panel_v2'den birebir okunur.
            Kohortlar: tavan (ham), tavan_v1 (ML filtreli).

  hwobos  — advisor'ın `load_hw_obos` girdisi. 1h master'dan TF'e resample
            edilip HyperWave (RSI7→SMA3, sinyal SMA3) big-dot dönüş noktaları.
            Kohort: her TF için AL_OS (aşırı satım dönüşü).
            NOT: modülün kendi SPEC'i bu taramayı "descriptive, NO trading-edge
            claim" olarak etiketliyor ve hwo_mtf_v1 ön-kayıtlı backtesti
            CLOSED_REJECTED; buradaki ölçüm bunu endeks evreninde teyit eder.

Kullanım:
  python tools/index_bt_misc.py --scan tavan  --universe XU100 --outdir ...
  python tools/index_bt_misc.py --scan hwobos --universe XU030 --outdir ...
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

PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"
EXTFEED_1H = ROOT / "output" / "extfeed_intraday_1h_3y_master.parquet"
HWO_CACHE = ROOT / "output" / "index_bt" / "_hwobos_events.parquet"
SBT_DATASET = ROOT / "output" / "sbt_1700_dataset_5d_intraday_v1.parquet"

TAVAN_TOL = 0.005          # tools/tavan_scan_live.py toleransı
TAVAN_RATIO = 1.10
ML_S_THRESHOLD = 0.65      # V1 pre-registered eşik


# ────────────────────────────── tavan ──────────────────────────────


def tavan_signals() -> pd.DataFrame:
    """(date, ticker, at_tavan, ml_s) — günlük barlardan tavan tespiti."""
    long = load_prices_long()
    long = long.sort_values(["ticker", "date"])
    prev = long.groupby("ticker")["close"].shift(1)
    ratio = long["close"] / prev
    at_tavan = (ratio - TAVAN_RATIO).abs() <= TAVAN_TOL
    sig = long.loc[at_tavan, ["date", "ticker"]].copy()

    panel = pd.read_parquet(PANEL, columns=["ml_1g"]).reset_index()
    panel = panel.rename(columns={"ml_1g": "ml_s"})
    sig = sig.merge(panel, on=["date", "ticker"], how="left")
    return sig


# ────────────────────────────── hyperwave ──────────────────────────────


def hwobos_events(tfs=("1d", "1w")) -> pd.DataFrame:
    from tools.hw_obos_scan import scan_ticker

    master = pd.read_parquet(EXTFEED_1H)
    parts = []
    tickers = sorted(master["ticker"].unique())
    for i, tk in enumerate(tickers, 1):
        sub = master[master["ticker"] == tk].sort_values("ts_istanbul")
        for tf in tfs:
            try:
                ev = scan_ticker(sub, tf, asof_ts=None)
            except Exception:
                continue
            if len(ev):
                parts.append(ev[["ticker", "tf", "ts", "kind"]])
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] {tk} · olay={sum(len(p) for p in parts):,}")
    if not parts:
        return pd.DataFrame(columns=["ticker", "tf", "ts", "kind"])
    ev = pd.concat(parts, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["ts"]).dt.tz_localize(None).dt.normalize()
    return ev


def _snap_to_sessions(sig: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Haftalık/aylık bar etiketlerini geçerli bir seans gününe hizala (<= etiket)."""
    s = sig.copy()
    idx = sessions.searchsorted(s["date"].to_numpy(), side="right") - 1
    ok = idx >= 0
    s = s[ok].copy()
    s["date"] = sessions[idx[ok]]
    return s


# ────────────────────────────── main ──────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, choices=["tavan", "hwobos", "sbt"])
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit", choices=["pit", "current"])
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-04-24")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    HWO_CACHE.parent.mkdir(parents=True, exist_ok=True)

    start = args.start
    if args.scan == "hwobos":
        # 1h master 2023-06'da başlıyor; HW ısınması için ilk 6 ay atılır
        start = max(start, "2024-01-01")

    ev = Evaluator(args.universe, args.mode, start, args.end)
    sessions = pd.DatetimeIndex(sorted(ev.universe["date"].unique()))
    print(f"evren {args.universe}/{args.mode}: {len(ev.tickers)} ticker, "
          f"{len(ev.uni):,} evren-gün, {len(sessions)} seans")

    out = {"scan": args.scan, "universe": args.universe, "mode": args.mode,
           "window": {"start": start, "end": args.end}, "cohorts": {}}

    def run(name, pairs):
        pairs = pairs[["date", "ticker"]].drop_duplicates()
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

    if args.scan == "tavan":
        sig = tavan_signals()
        sig_u = ev.restrict(sig)
        print(f"tavan sinyali (evren içi): {len(sig_u):,}")
        run("tavan", sig_u)
        v1 = sig_u[sig_u["ml_s"].fillna(-1) >= ML_S_THRESHOLD]
        out["ml_s_coverage"] = {"n_total": int(len(sig_u)),
                                "n_scored": int(sig_u["ml_s"].notna().sum()),
                                "n_v1": int(len(v1))}
        run("tavan_v1", v1)

    elif args.scan == "sbt":
        # SBT-1700 aday olayları (17:00 sıkışma-kırılım tespiti). E04 sıralama
        # katmanı model tabanlı ve TEST örneklemi zaten n=25 trade; endeks
        # kısıtında büsbütün eriyor → yalnızca aday kohortu ölçülür.
        d = pd.read_parquet(SBT_DATASET, columns=["ticker", "date"])
        d["date"] = pd.to_datetime(d["date"])
        run("sbt_candidates", ev.restrict(d))

    else:
        if HWO_CACHE.exists():
            evdf = pd.read_parquet(HWO_CACHE)
            evdf["date"] = pd.to_datetime(evdf["date"])
        else:
            print("── HyperWave olayları üretiliyor (1h master)")
            evdf = hwobos_events()
            evdf.to_parquet(HWO_CACHE, index=False)
        print(f"HW olay tablosu: {len(evdf):,}")
        for tf in sorted(evdf["tf"].unique()):
            al = evdf[(evdf["tf"] == tf) & (evdf["kind"] == "AL_OS")][["date", "ticker"]]
            al = _snap_to_sessions(al, sessions)
            run(f"AL_OS_{tf}", ev.restrict(al))

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
