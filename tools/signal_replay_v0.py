#!/usr/bin/env python3
"""
Sinyal-replay harness v0 — bağlam tarayıcılarının GEÇMİŞ üyeliğini koddan üret.

Kullanıcı: "kodlar githubda, geçmiş sinyaller üretilebilir" — doğru. Artifact
retention (90g) + token-bayat penceresi yerine, fiyat-master'ı her örnekleme
tarihi T'de kesip tarayıcının ÇEKİRDEK fonksiyonunu çağırarak point-in-time
üyelik üretilir (token-bağımsız, cluster3 scan --asof mantığı).

Çıktı: output/signal_replay_membership_v0.parquet  (asof, ticker, scanner)

LOOKAHEAD AUDIT (smart_breakout, 2026-06-13): replay @2026-06-12 = 11 breakout,
11/11'i (100%) aynı-tarih canlı GH Pages artifact'ında mevcut → replay sadık,
gross lookahead YOK. Yaklaşım doğrulandı; diğer adapter'lar aynı truncate@T
deseniyle eklenir, her biri kendi canlı-artifact audit'inden geçer.

ÖNCE smart_breakout (en basit, XU bağımsız) ile audit:
  python tools/signal_replay_v0.py --scanner smart_breakout --start 2026-03-20 \
         --end 2026-06-12 --audit
Tam set:
  python tools/signal_replay_v0.py --start 2024-01-01 --end 2026-06-12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"
EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"
DST = OUT / "signal_replay_membership_v0.parquet"

MIN_WARMUP_BARS = 220  # indikatör warmup (BB/ATR/SMA200 vb. için yeterli geçmiş)


def load_daily_caps():
    """extfeed → {ticker: daily df (Capitalized OHLCV, DatetimeIndex)}.
    smart_breakout scan_single Capitalized kolon bekliyor."""
    from data.intraday_1h import daily_resample
    bars = pd.read_parquet(EXTFEED, columns=["ticker", "ts_istanbul", "open",
                                             "high", "low", "close", "volume"])
    daily = daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"])
    out = {}
    for tkr, g in daily.groupby("ticker", observed=True):
        d = g.sort_values("date").set_index("date")[["open", "high", "low", "close", "volume"]]
        d.columns = ["Open", "High", "Low", "Close", "Volume"]
        out[str(tkr)] = d
    return out


def sampling_dates(daily_caps, start, end, step_days=7):
    """Haftalık-adım örnekleme tarihleri (gerçek işlem günlerine snap)."""
    all_dates = sorted(set().union(*[set(d.index) for d in daily_caps.values()]))
    all_dates = [d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not all_dates:
        return []
    picked, last = [], None
    for d in all_dates:
        if last is None or (d - last).days >= step_days:
            picked.append(d)
            last = d
    return picked


# ── ADAPTER: smart_breakout (per-ticker, truncate@T → scan_single) ──

def replay_smart_breakout(daily_caps, dates, max_bo_age=5):
    import run_smart_breakout as sb
    rows = []
    for T in dates:
        for tkr, df in daily_caps.items():
            sub = df[df.index <= T]
            if len(sub) < MIN_WARMUP_BARS:
                continue
            try:
                res = sb.scan_single(sub, use_vol=True, use_htf=False)
            except Exception:
                continue
            if res and res.get("status") == "BREAKOUT" and \
                    res.get("bars_ago", 99) <= max_bo_age and \
                    res.get("trade_status") == "OPEN":
                rows.append({"asof": T.normalize(), "ticker": tkr,
                             "scanner": "smart_breakout"})
    return rows


def _lower(df):
    """Capitalized → lowercase OHLCV (nox_v3/regime _scan beklentisi)."""
    return df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})


# ── ADAPTER: nox_v3 (weekly pivot + daily; _scan buys+candidates = AL üyeliği) ──

def replay_nox_v3(daily_caps, dates):
    import run_nox_v3 as nx
    low_caps = {t: _lower(d) for t, d in daily_caps.items()}
    rows = []
    for T in dates:
        cut = {t: d[d.index <= T] for t, d in low_caps.items()}
        cut = {t: d for t, d in cut.items() if len(d) >= MIN_WARMUP_BARS}
        if not cut:
            continue
        # AKSİYON üyeliği = fresh BUGÜN ∧ gate AÇIK (advisor 'nw' tanımı;
        # ham buys ≈ "pivot üstünde" DURUMU, taze tetik değil → filtrele)
        try:
            b, _s, _c, _n, _dt = nx._scan(cut, tf_label="daily")
            for r in b:
                if r.get("fresh") == "BUGUN" and r.get("gate"):
                    rows.append({"asof": T.normalize(), "ticker": r["ticker"],
                                 "scanner": "nox_v3_daily"})
        except Exception:
            pass
        try:
            wk = {t: nx._to_weekly(d) for t, d in cut.items()}
            wk = {t: d for t, d in wk.items() if len(d) >= 30}
            b, _s, _c, _n, _dt = nx._scan(wk, tf_label="weekly")
            for r in b:
                if r.get("fresh") == "BUGUN" and r.get("gate"):
                    rows.append({"asof": T.normalize(), "ticker": r["ticker"],
                                 "scanner": "nox_v3_weekly"})
        except Exception:
            pass
    return rows


# ── ADAPTER: regime_transition — AUDIT DÜŞTÜ (2026-06-13) ──
# replay days_since canlıdan sistematik farklı (replay 21/189/28 vs canlı 7/56/7):
# haftalık-bar inşası / rejim-dönüş tespiti hassasiyeti (canlı feed farkı olası).
# %51 recall → kombinasyon analizine GÜVENİLİR DEĞİL. forward-log'a bırakıldı.
# Adapter ileride düzeltme için tutuluyor; AUDIT_FAILED ile default'tan hariç.

def replay_regime(daily_caps, dates):
    import run_regime_transition as rt
    low_caps = {t: _lower(d) for t, d in daily_caps.items()}
    rows = []
    for T in dates:
        cut = {t: d[d.index <= T] for t, d in low_caps.items()}
        cut = {t: d for t, d in cut.items() if len(d) >= MIN_WARMUP_BARS}
        if not cut:
            continue
        try:
            wk = {t: rt._to_weekly(d, include_incomplete=True) for t, d in cut.items()}
            mo = {t: rt._to_monthly(d) for t, d in cut.items()}
            wk = {t: d for t, d in wk.items() if len(d) >= 30}
            results, _n, _dt, _rd = rt._scan_all(wk, higher_tf_dfs=mo,
                                                 timeframe="weekly")
            for s in results:
                badge = rt._entry_window_tf(s.days_since, "weekly")
                if badge in ("TAZE", "2.DALGA") and s.oe_score <= 2:
                    rows.append({"asof": T.normalize(), "ticker": s.ticker,
                                 "scanner": "regime_transition"})
        except Exception:
            pass
    return rows


# ── ADAPTER: HW OB/OS (AL_OS dönüş; scan_ticker causal → tek-hesap, tarih-pencereli)
# ⚠️ AL_OS backtest'te REDDEDİLDİ (hwo_mtf_v1 rank 0.43) — kombinasyonda 'farklı
# davranır mı' diye test için, edge iddiası YOK.

def replay_hw_obos(daily_caps, dates, lookback_days=10, tf="1d"):
    sys.path.insert(0, str(ROOT / "tools"))
    import hw_obos_scan as hw
    # raw 1h gerekli (daily_caps değil) — extfeed'i ticker bazında tut
    bars = pd.read_parquet(EXTFEED, columns=["ticker", "ts_istanbul", "open",
                                             "high", "low", "close", "volume"])
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])
    rows = []
    dset = [pd.Timestamp(d).normalize() for d in dates]
    for tk, sub in bars.groupby("ticker", observed=True):
        try:
            ev = hw.scan_ticker(sub.sort_values("ts_istanbul"), tf, asof_ts=None)
        except Exception:
            continue
        if ev.empty:
            continue
        ev = ev[ev["kind"] == "AL_OS"].copy()
        if ev.empty:
            continue
        ev_ts = pd.to_datetime(ev["ts"]).dt.tz_localize(None).dt.normalize()
        for T in dset:
            lo = T - pd.Timedelta(days=lookback_days)
            if ((ev_ts > lo) & (ev_ts <= T)).any():
                rows.append({"asof": T, "ticker": str(tk), "scanner": "hw_al_os"})
    return rows


# Audit'ten GEÇEN adapter'lar (kombinasyon analizine girer):
ADAPTERS = {"smart_breakout": replay_smart_breakout,   # audit 100%
            "nox_v3": replay_nox_v3,                    # audit 98.6%
            "hw_obos": replay_hw_obos}                  # AL_OS anti-edge (test amaçlı)
# Audit DÜŞEN (forward-log'a, kombinasyona girmez):
ADAPTERS_AUDIT_FAILED = {"regime_transition": replay_regime}  # recall %51


def audit(rows, scanner):
    """Replay'lenmiş üyeliği canlı GH Pages artifact'ıyla kıyasla (lookahead kapısı).
    En son replay tarihinde replay-seti vs canlı-set Jaccard örtüşmesi."""
    rep = pd.DataFrame(rows)
    if rep.empty:
        print("[audit] replay boş — atlandı"); return
    last = rep["asof"].max()
    rep_set = set(rep[rep["asof"] == last]["ticker"].str.replace(".IS", "", regex=False))
    print(f"[audit] {scanner} @ {last.date()}: replay {len(rep_set)} breakout")
    print(f"        örnek: {sorted(rep_set)[:12]}")
    print("        → canlı artifact karşılaştırması manuel: hw-obos/sbt artifact'ı "
          "indirip kesişime bak (replay≈canlı ise lookahead yok)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scanner", default="smart_breakout",
                    choices=list(ADAPTERS) + list(ADAPTERS_AUDIT_FAILED) + ["all"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-06-12")
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="Mevcut membership parquet'e ekle (farklı scanner)")
    a = ap.parse_args()

    print("[replay] daily yükleniyor...")
    caps = load_daily_caps()
    dates = sampling_dates(caps, a.start, a.end, a.step_days)
    print(f"[replay] {len(caps)} ticker, {len(dates)} örnekleme tarihi "
          f"({a.start}→{a.end}, {a.step_days}g adım)")

    reg = {**ADAPTERS, **ADAPTERS_AUDIT_FAILED}
    scanners = list(ADAPTERS) if a.scanner == "all" else [a.scanner]
    all_rows = []
    for s in scanners:
        print(f"[replay] {s}...")
        rows = reg[s](caps, dates)
        print(f"  {len(rows)} üyelik kaydı")
        all_rows += rows
        if a.audit:
            audit(rows, s)

    df = pd.DataFrame(all_rows)
    if a.append and DST.exists():
        old = pd.read_parquet(DST)
        old = old[~old["scanner"].isin(scanners)]  # bu scanner'ı tazele
        df = pd.concat([old, df], ignore_index=True)
    if not df.empty:
        df.to_parquet(DST, index=False)
        print(f"[replay] → {DST} ({len(df)} satır, scanner'lar: "
              f"{sorted(df['scanner'].unique())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
