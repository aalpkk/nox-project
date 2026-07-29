#!/usr/bin/env python3
"""screener_combo hattı (Regime Transition + NOX v3 + AL/SAT) — endeks-kısıtlı backtest.

Bu üç tarama hem briefing HTML'inin dahili tarayıcı havuzunu (regime_transition,
nox_v3_daily, nox_v3_weekly, alsat) hem de dış `screener_combo_latest.html`
listesini besler; üçü de screener_combo/signals.py içinde tarihsel olarak
yeniden üretilebilir hâlde olduğu için tek sürücüde ölçülür.

Kohortlar:
  RT_state  — regime_transition AL durumu (kalıcı durum bayrağı)
  RT_new    — AL durumuna GEÇİŞ barı (taramanın "yeni sinyal" satırı)
  NOXw      — nox_v3 haftalık pivot alım
  NOXd      — nox_v3 günlük pivot alım
  AS        — AL/SAT tarayıcı AL kararı
  vote2/3   — aynı gün RT_new/NOXw/AS kapılarından >=2 / 3 tanesi
  RT+AS     — kaskad (ikisi aynı gün)

Kullanım:
  python tools/index_bt_combo.py --universe XU100 --outdir output/index_bt/combo_xu100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener_combo.signals import regime_transition_rich, nox_rich, alsat_rich  # noqa: E402
from tools.index_bt_engine import Evaluator, load_prices_long, XU_PATH  # noqa: E402


def _weekly(d: pd.DataFrame) -> pd.DataFrame:
    return (d.resample("W-FRI")
             .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                  close=("close", "last"), volume=("volume", "sum"))
             .dropna())


def build_signals(tickers: set[str], min_bars: int = 120) -> pd.DataFrame:
    """Tüm geçmiş üzerinde sinyal tablosu (evren filtresi SONRA uygulanır).

    Göstergeler yalnızca geçmişe bakan rolling/ewm/shift işlemleridir; tam seri
    üzerinde bir kez hesaplayıp t barındaki değeri okumak, seriyi t'de kesmekle
    özdeştir → look-ahead yok.
    """
    long = load_prices_long()
    xu = pd.read_parquet(XU_PATH)["Close"]
    xu.index = pd.to_datetime(xu.index).tz_localize(None).normalize()

    rows = []
    tks = sorted(tickers)
    t0 = time.time()
    for i, tkr in enumerate(tks, 1):
        sub = long[long.ticker == tkr].set_index("date")[
            ["open", "high", "low", "close", "volume"]].sort_index().astype("float64")
        if len(sub) < min_bars:
            continue
        try:
            w = _weekly(sub)
            if len(w) < 12:
                continue
            rt = regime_transition_rich(sub, w)
            nw = nox_rich(sub, w)
            asg = alsat_rich(sub, w, xu)
        except Exception as e:
            print(f"  [atla] {tkr}: {type(e).__name__}: {e}")
            continue

        rt_state = rt["al_signal"].astype(bool)
        rt_new = rt_state & ~rt_state.shift(1, fill_value=False)
        df = pd.DataFrame({
            "RT_state": rt_state,
            "RT_new": rt_new,
            "NOXw": nw["nox_w_trig"].astype(bool),
            "NOXd": nw["nox_d_trig"].astype(bool),
            "AS": asg["al_signal"].astype(bool),
            "rt_tier": rt["rt_tier"],
            "as_subtype": asg["as_subtype"],
        }, index=sub.index)
        df = df[df[["RT_new", "NOXw", "NOXd", "AS"]].any(axis=1) | df["RT_state"]]
        df.index.name = "date"
        df = df.reset_index()
        df.insert(0, "ticker", tkr)
        rows.append(df)
        if i % 40 == 0:
            print(f"  [{i}/{len(tks)}] {tkr} · {time.time()-t0:.0f}s")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


COHORTS = ["RT_state", "RT_new", "NOXw", "NOXd", "AS"]


def add_combos(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.copy()
    votes = s["RT_new"].astype(int) + s["NOXw"].astype(int) + s["AS"].astype(int)
    s["vote2"] = votes >= 2
    s["vote3"] = votes >= 3
    s["RT_AS"] = s["RT_new"] & s["AS"]
    s["RT_NOXw"] = s["RT_new"] & s["NOXw"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="XU100")
    ap.add_argument("--mode", default="pit", choices=["pit", "current"])
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2026-04-24")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--signals-cache", default="output/index_bt/_combo_signals.parquet")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.signals_cache); cache.parent.mkdir(parents=True, exist_ok=True)

    print(f"── evren: {args.universe}/{args.mode}")
    ev = Evaluator(args.universe, args.mode, args.start, args.end, n_random=args.n_random)
    print(f"  {len(ev.tickers)} ticker, {len(ev.uni):,} evren-gün, "
          f"{ev.universe['date'].nunique()} seans")

    if cache.exists():
        sig = pd.read_parquet(cache)
        missing = ev.tickers - set(sig["ticker"].unique())
        if missing:
            print(f"── {len(missing)} eksik ticker için sinyal üretiliyor")
            sig = pd.concat([sig, build_signals(missing)], ignore_index=True)
            sig.to_parquet(cache, index=False)
    else:
        print("── sinyaller üretiliyor (tüm geçmiş)")
        sig = build_signals(ev.tickers)
        sig.to_parquet(cache, index=False)
    print(f"  sinyal tablosu: {len(sig):,} satır")

    sig = add_combos(sig)
    sig_u = ev.restrict(sig)
    print(f"  evrene indirgenmiş: {len(sig_u):,} satır")

    names = COHORTS + ["vote2", "vote3", "RT_AS", "RT_NOXw"]
    out = {"universe": args.universe, "mode": args.mode,
           "window": {"start": args.start, "end": args.end,
                      "n_sessions": int(ev.universe["date"].nunique()),
                      "n_tickers": len(ev.tickers)},
           "cohorts": {}}

    for name in names:
        pairs = sig_u.loc[sig_u[name].fillna(False), ["date", "ticker"]]
        if not len(pairs):
            out["cohorts"][name] = {"n_signals": 0}
            continue
        print(f"── kohort {name}: {len(pairs):,} sinyal")
        blk = ev.layer_a(pairs)
        tr = ev.layer_b(pairs)
        blk["trades"] = ev.trade_stats(tr)
        if len(tr):
            tr.assign(cohort=name).to_parquet(outdir / f"trades_{name}.parquet", index=False)
        out["cohorts"][name] = blk

    # trade seviyesinde rastgele kontrol — en dar kohortun günlük sayısıyla
    ref = sig_u.loc[sig_u["RT_new"].fillna(False)].groupby("date").size()
    print("── rastgele trade tabanı (RT_new günlük sayısıyla)")
    out["random_trade_baseline_rt_new"] = ev.random_trade_baseline(ref, n_rep=5)

    (outdir / "results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
