#!/usr/bin/env python3
"""BIST-100 / BIST-30 kısıtlı tarama backtest motoru — ortak katman.

Alpha scan backtestindeki (scripts/alpha_scan_backtest_sim.py) ölçüm
mimarisinin taramadan-bağımsız hâli. Her tarama için istenen tek şey
`(date, ticker)` çiftlerinden oluşan bir sinyal kohortu; motor şunları üretir:

  Katman A — sabit ufuk ileri getiriler (giriş t+1 açılış, çıkış t+h kapanış)
             · SIG   sinyal kohortu
             · EVREN aynı gün endekste olan TÜM hisseler (taban)
             · RND   aynı gün, aynı sayıda RASTGELE endeks üyesi (n tekrar)
             · REL   sinyalin o günkü evren medyanına göre farkı
  Katman B — üretim çıkış makinesiyle (2xATR stop, +2R BE, 1.5xATR trailing,
             %50 TP, 40 gün max hold) gerçek trade PnL'i.

RND kontrolü kritik: BIST-100 evreni zaten yükselen bir endeks olduğundan
"sinyal ortalaması pozitif" tek başına anlamsızdır; ölçülen şey aynı gün aynı
evrenden rastgele seçime göre ARTIŞ'tır.

Süreksizlik barları (|günlük getiri| > %20 → bedelsiz/bölünme/veri dikişi)
işaretlenir ve temiz kohort ayrıca raporlanır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from alpha_scan_backtest_sim import (          # noqa: E402
    build_forward_matrix, simulate_trade, ret_stats, HORIZONS,
)
from core.indicators import calc_atr           # noqa: E402
from tools.index_universe import load_membership, CURRENT, _norm  # noqa: E402

OHLCV_PATH = ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
XU_PATH = ROOT / "output" / "xu100_cache.parquet"


# ────────────────────────────── veri ──────────────────────────────


def load_prices_long() -> pd.DataFrame:
    """Uzun format günlük barlar: ticker, date, open/high/low/close/volume."""
    raw = pd.read_parquet(OHLCV_PATH).reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={raw.columns[0]: "date"})
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    return raw[keep].sort_values(["ticker", "date"]).reset_index(drop=True)


def universe_pairs(index: str, mode: str = "pit") -> pd.DataFrame:
    """(date, ticker) — evrende olan tüm gün/hisse çiftleri."""
    if index.lower() == "all":
        long = load_prices_long()
        return long[["date", "ticker"]]
    idx = _norm(index)
    if mode == "current":
        long = load_prices_long()
        keys = (set(CURRENT["XU100"]) - set(CURRENT["XU030"])) if idx == "XU100EX30" \
            else set(CURRENT[idx])
        return long.loc[long["ticker"].isin(keys), ["date", "ticker"]]
    memb = load_membership()
    if idx == "XU100EX30":
        sel = memb["in_xu100"] & ~memb["in_xu030"]
    else:
        sel = memb["in_xu100"] if idx == "XU100" else memb["in_xu030"]
    return memb.loc[sel, ["date", "ticker"]].reset_index(drop=True)


def prices_dict(long: pd.DataFrame, tickers: set[str] | None = None) -> dict:
    out = {}
    for tkr, g in long.groupby("ticker", sort=False):
        if tickers is not None and tkr not in tickers:
            continue
        d = g.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        d.columns = ["Open", "High", "Low", "Close", "Volume"]
        out[tkr] = d
    return out


# ────────────────────────────── değerlendirme ──────────────────────────────


class Evaluator:
    """Bir evren için ileri getiri matrisi + kohort ölçüm makinesi."""

    def __init__(self, index: str, mode: str = "pit",
                 start: str = "2021-07-01", end: str = "2026-04-24",
                 n_random: int = 20, seed: int = 42):
        self.index, self.mode = index, mode
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.n_random = n_random
        self.rng = np.random.default_rng(seed)

        long = load_prices_long()
        uni = universe_pairs(index, mode)
        uni = uni[(uni["date"] >= self.start) & (uni["date"] <= self.end)]
        self.universe = uni.drop_duplicates().reset_index(drop=True)
        self.tickers = set(self.universe["ticker"])

        self.prices = prices_dict(long, self.tickers)
        self.fm = build_forward_matrix(self.prices)          # (date,ticker) -> fwd*/bad*
        self.uni_idx = pd.MultiIndex.from_frame(
            self.universe.rename(columns={"date": "date", "ticker": "ticker"}))
        self.uni = self.fm.reindex(self.uni_idx).dropna(how="all")
        # reindex sonrası bad* kolonları object dtype'a düşüyor; `~mask` bozulmasın
        for h in HORIZONS:
            self.uni[f"bad{h}"] = self.uni[f"bad{h}"].fillna(False).astype(bool)

        self.atr = {t: calc_atr(df).to_numpy() for t, df in self.prices.items()}
        self.pos = {t: {d: i for i, d in enumerate(df.index)} for t, df in self.prices.items()}
        self._uni_counts = self.uni.groupby(level=0).size()

    # ── Katman A ──

    def _random_baseline(self, counts: pd.Series) -> dict:
        ug = self.uni
        dts = ug.index.get_level_values("date")
        k_row = pd.Series(dts.map(counts).fillna(0).to_numpy(), index=ug.index)
        acc = {h: [] for h in HORIZONS}
        for _ in range(self.n_random):
            r = pd.Series(self.rng.random(len(ug)), index=ug.index)
            rk = r.groupby(dts).rank(method="first")
            rp = ug[(rk <= k_row).to_numpy()]
            if not len(rp):
                continue
            for h in HORIZONS:
                clean = rp[~rp[f"bad{h}"].fillna(False).astype(bool)]
                acc[h].append(ret_stats(clean[f"fwd{h}"].to_numpy()))
        out = {}
        for h in HORIZONS:
            arr = [a for a in acc[h] if a.get("n")]
            if not arr:
                continue
            out[f"h{h}"] = {
                "n_reps": len(arr),
                "hit_mean": float(np.mean([a["hit"] for a in arr])),
                "mean_mean": float(np.mean([a["mean"] for a in arr])),
                "mean_p95": float(np.percentile([a["mean"] for a in arr], 95)),
                "median_mean": float(np.mean([a["median"] for a in arr])),
                "median_p95": float(np.percentile([a["median"] for a in arr], 95)),
            }
        return out

    def layer_a(self, sig_pairs: pd.DataFrame, with_random: bool = True) -> dict:
        """sig_pairs: date, ticker kolonlu DataFrame (evrene göre önceden filtrelenmiş)."""
        if not len(sig_pairs):
            return {"n_signals": 0}
        si = pd.MultiIndex.from_frame(sig_pairs[["date", "ticker"]].drop_duplicates())
        sg = self.fm.reindex(si).dropna(how="all")
        if not len(sg):
            return {"n_signals": 0}
        for h in HORIZONS:
            sg[f"bad{h}"] = sg[f"bad{h}"].fillna(False).astype(bool)
        blk = {
            "n_signals": int(len(sg)),
            "n_days": int(sg.index.get_level_values("date").nunique()),
            "signals_per_day_mean": float(sg.groupby(level=0).size().mean()),
            "n_tickers": int(sg.index.get_level_values("ticker").nunique()),
        }
        for h in HORIZONS:
            col, bcol = f"fwd{h}", f"bad{h}"
            sgc = sg[~sg[bcol].fillna(False)]
            ugc = self.uni[~self.uni[bcol].fillna(False)]
            blk[f"signal_h{h}"] = ret_stats(sgc[col].to_numpy())
            blk[f"universe_h{h}"] = ret_stats(ugc[col].to_numpy())
            med = ugc.groupby(level=0)[col].median()
            rel = sgc[col] - sgc.index.get_level_values("date").map(med)
            blk[f"relative_h{h}"] = ret_stats(rel.to_numpy())
        if with_random:
            blk["random"] = self._random_baseline(sg.groupby(level=0).size())
        return blk

    # ── Katman B ──

    def layer_b(self, sig_pairs: pd.DataFrame, conservative: bool = True) -> pd.DataFrame:
        recs = []
        for d, t in sig_pairs[["date", "ticker"]].drop_duplicates().itertuples(index=False):
            df = self.prices.get(t)
            if df is None:
                continue
            i = self.pos[t].get(d)
            if i is None:
                continue
            a = self.atr[t][i]
            rec = simulate_trade(df, i, float(a) if np.isfinite(a) else np.nan,
                                 self.atr[t], conservative=conservative)
            if rec is None:
                continue
            rec.update({"signal_date": d, "ticker": t})
            recs.append(rec)
        return pd.DataFrame(recs)

    def trade_stats(self, trades: pd.DataFrame) -> dict:
        if not len(trades):
            return {"n": 0}
        s = ret_stats(trades["pnl_pct"].to_numpy())
        s["avg_hold_bars"] = float(trades["hold_bars"].mean())
        s["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
        return s

    def random_trade_baseline(self, counts_by_day: pd.Series, n_rep: int = 5,
                              conservative: bool = True) -> dict:
        """Aynı gün, aynı sayıda rastgele endeks üyesine AYNI çıkış makinesi."""
        ug = self.uni
        dts = ug.index.get_level_values("date")
        k_row = pd.Series(dts.map(counts_by_day).fillna(0).to_numpy(), index=ug.index)
        stats = []
        for _ in range(n_rep):
            r = pd.Series(self.rng.random(len(ug)), index=ug.index)
            rk = r.groupby(dts).rank(method="first")
            pick = ug[(rk <= k_row).to_numpy()]
            if not len(pick):
                continue
            pairs = pd.DataFrame({"date": pick.index.get_level_values("date"),
                                  "ticker": pick.index.get_level_values("ticker")})
            tr = self.layer_b(pairs, conservative=conservative)
            if len(tr):
                stats.append(ret_stats(tr["pnl_pct"].to_numpy()))
        if not stats:
            return {}
        return {
            "n_reps": len(stats),
            "n_mean": float(np.mean([s["n"] for s in stats])),
            "hit_mean": float(np.mean([s["hit"] for s in stats])),
            "mean_mean": float(np.mean([s["mean"] for s in stats])),
            "mean_p95": float(np.percentile([s["mean"] for s in stats], 95)),
            "median_mean": float(np.mean([s["median"] for s in stats])),
            "pf_mean": float(np.mean([s["pf"] for s in stats])),
        }

    # ── yardımcı ──

    def restrict(self, sig: pd.DataFrame) -> pd.DataFrame:
        """Sinyal tablosunu evrene indirger (date+ticker eşleşmesi)."""
        u = self.universe.copy()
        u["_in"] = True
        m = sig.merge(u, on=["date", "ticker"], how="left")
        m = m[m["_in"].fillna(False)].drop(columns=["_in"])
        return m[(m["date"] >= self.start) & (m["date"] <= self.end)].reset_index(drop=True)
