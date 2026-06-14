#!/usr/bin/env python3
"""
Sinyal-KOMBİNASYON × MFE+MAE fırsat profili v0.

Kullanıcı isteği (net): "feature değil sinyal KOMBİNASYONLARI; capture bakmayalım
önce, MFE+MAE." → Her olayı (ticker, date) hangi sinyallerin BİRLİKTE tetiklediğine
göre grupla; her kombinasyonun MFE20+MAE20 fırsat profilini havuz-İÇİNDE ölç.
(Capture YOK — V1/realized burada kullanılmaz; fırsat-tavan + MAE-risk.)

Sinyal aileleri (audit-geçen derin-geçmiş):
  - DE-tarafı: mb_scanner / horizontal_base / triangle_break / paper_execution
    (ranking_lab_features_v0 'source' kolonu)
  - cluster3  (paper snapshots ledger)
  - nox_v3 (daily/weekly), smart_breakout  (signal_replay_membership_v0)
  Ertelenen (audit-fail / compute): regime_transition, alpha → forward-log.

Çıktı: output/signal_combination_mfemae_v0.md
Kullanım: python tools/signal_combination_mfemae_v0.py [--asof-min 2024-01-01] [--min-n 50]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"
PANEL = OUT / "ranking_lab_features_v0.parquet"
EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"
CLUSTER3 = OUT / "ranking_lab_cluster3_paper_snapshots_v0.parquet"
REPLAY = OUT / "signal_replay_membership_v0.parquet"
TRIDENT = OUT / "trident_probe_mb_3y.parquet"
DST_MD = OUT / "signal_combination_mfemae_v0.md"

FWD = 20
# fırsat profili eşikleri (havuz-içi yorumlanır; mutlak referans için):
CLEAN_MFE = 0.20    # "temiz koşucu" MFE eşiği
CLEAN_MAE = -0.08   # sığ MAE (cehennem-dönüşü değil)


def compute_mae(panel):
    """İleri FWD bar MAE = min(low)/entry − 1 (extfeed)."""
    from data.intraday_1h import daily_resample
    bars = pd.read_parquet(EXTFEED, columns=["ticker", "ts_istanbul", "open",
                                             "high", "low", "close", "volume"])
    daily = daily_resample(bars).sort_values(["ticker", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    g = daily.groupby("ticker")
    for d in range(1, FWD + 1):
        daily[f"d{d}_low"] = g["low"].shift(-d)
    fcols = ["ticker", "date"] + [f"d{d}_low" for d in range(1, FWD + 1)]
    ev = panel[["ticker", "signal_date", "close_signal_date"]].copy()
    ev["ticker"] = ev["ticker"].astype(str)
    ev["date"] = pd.to_datetime(ev["signal_date"]).dt.normalize()
    ev = ev.merge(daily[fcols], on=["ticker", "date"], how="left")
    L = np.array([ev[f"d{i}_low"].values for i in range(1, FWD + 1)], float)
    e = ev["close_signal_date"].values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.nanmin(L, axis=0) / e - 1.0
    return pd.Series(mae, index=panel.index).replace([np.inf, -np.inf], np.nan)


def build_trident_set():
    """Trident Tier-1 üyeliği (mb_1d above_mb_birth): G1 D_pct∈[20,30) ∧
    G2 SIL≤p33(derin) ∧ G3 BoS≥p67(yüksek). G4 XU100-rejim atlandı (piyasa-geneli).
    Tercile full-period (advisor trailing-365d; kombinasyon için yaklaşık).
    Trident = memory'de tek PASS_SELECTION hat → kombinasyonda kritik test."""
    if not TRIDENT.exists():
        return set()
    df = pd.read_parquet(TRIDENT)
    # mb_1d cohort
    fam = df.get("setup_family", df.get("family"))
    df = df[fam.astype(str).str.contains("mb_1d|mb.*1d", case=False, na=False) |
            df.get("family", pd.Series("", index=df.index)).astype(str).eq("mb_1d")].copy()
    if df.empty:
        df = pd.read_parquet(TRIDENT)  # cohort filtresi tutmazsa tüm above_mb_birth
    sil = df["structural_invalidation_pct_below_B"]
    bos = df["bos_distance_atr_at_event"]
    dpct = df["D_pct"]
    sil_t33 = sil.quantile(0.33)
    bos_t67 = bos.quantile(0.67)
    g1 = (dpct >= 20.0) & (dpct < 30.0)
    g2 = sil <= sil_t33
    g3 = bos >= bos_t67
    tier1 = df[g1 & g2 & g3]
    d = pd.to_datetime(tier1["event_bar_date"]).dt.normalize()
    return set(zip(tier1["ticker"].astype(str).str.upper(), d))


def build_membership(panel):
    """Her (ticker, date) için sinyal SETİ (frozenset). Panel = DE-tarafı satır
    bazında source; cluster3 + replay (nox_v3/smart_breakout) (ticker,date) join."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    df["tkr"] = df["ticker"].astype(str).str.upper()

    # DE-tarafı: (ticker,date) bazında hangi source'lar tetikledi
    de = df.groupby(["tkr", "date"])["source"].apply(lambda s: set(s.unique()))

    # cluster3
    c3 = set()
    if CLUSTER3.exists():
        c = pd.read_parquet(CLUSTER3)
        c["date"] = pd.to_datetime(c["signal_date"]).dt.normalize()
        c3 = set(zip(c["ticker"].astype(str).str.upper(), c["date"]))

    # replay (nox_v3_daily/weekly, smart_breakout) — en yakın ÖNCEKİ replay tarihine
    # (haftalık-adım): olay tarihinden ≤7g önceki replay tetiği "aynı dönem" sayılır
    rep_map = {}
    if REPLAY.exists():
        rep = pd.read_parquet(REPLAY)
        rep["asof"] = pd.to_datetime(rep["asof"]).dt.normalize()
        for (tkr, asof), grp in rep.groupby([rep["ticker"].astype(str).str.upper(), "asof"]):
            rep_map.setdefault(tkr, []).append((asof, set(grp["scanner"])))
        for t in rep_map:
            rep_map[t].sort()

    def replay_at(tkr, date):
        out = set()
        for asof, scs in rep_map.get(tkr, []):
            if 0 <= (date - asof).days <= 7:
                out |= scs
        return out

    trident = build_trident_set()

    sets = {}
    for (tkr, date), srcs in de.items():
        s = set(srcs)
        if (tkr, date) in c3:
            s.add("cluster3")
        if (tkr, date) in trident:
            s.add("trident_tier1")
        s |= replay_at(tkr, date)
        sets[(tkr, date)] = frozenset(s)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof-min", default="2024-01-01")
    ap.add_argument("--min-n", type=int, default=50, help="kombinasyon min olay")
    a = ap.parse_args()

    print("[komb] panel + MAE...")
    df = pd.read_parquet(PANEL)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df[(df["signal_date"] >= pd.Timestamp(a.asof_min)) & df["mfe_pct_20d"].notna()].copy()
    df["mae_pct_20d"] = compute_mae(df)
    df = df[df["mae_pct_20d"].notna()].copy()

    print("[komb] sinyal setleri...")
    sets = build_membership(df)
    df["tkr"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    df["sigset"] = [sets.get((t, d), frozenset()) for t, d in zip(df["tkr"], df["date"])]
    # (ticker,date) tekille — aynı olayın çok satırını tek sigset'e indir
    ev = df.sort_values("signal_date").drop_duplicates(["tkr", "date"]).copy()
    print(f"  {len(ev):,} benzersiz olay")

    # havuz baseline
    base_mfe = ev["mfe_pct_20d"].median()
    base_mae = ev["mae_pct_20d"].median()
    base_clean = ((ev["mfe_pct_20d"] >= CLEAN_MFE) & (ev["mae_pct_20d"] >= CLEAN_MAE)).mean()

    SIGNALS = ["mb_scanner", "horizontal_base", "triangle_break", "paper_execution",
               "cluster3", "trident_tier1", "nox_v3_daily", "nox_v3_weekly",
               "smart_breakout", "hw_al_os"]

    def profile(mask, label):
        sub = ev[mask]
        if len(sub) < a.min_n:
            return None
        return {"komb": label, "n": int(len(sub)),
                "mfe_med": round(float(sub["mfe_pct_20d"].median()), 3),
                "mae_med": round(float(sub["mae_pct_20d"].median()), 3),
                "clean_pct": round(float(((sub["mfe_pct_20d"] >= CLEAN_MFE) &
                                          (sub["mae_pct_20d"] >= CLEAN_MAE)).mean()), 3)}

    # tekil sinyaller + ikili/üçlü kombinasyonlar (yeterli n olanlar)
    rows = []
    for k in (1, 2, 3):
        for combo in combinations(SIGNALS, k):
            mask = ev["sigset"].apply(lambda s: all(c in s for c in combo))
            p = profile(mask, " + ".join(combo))
            if p:
                p["k"] = k
                rows.append(p)

    res = pd.DataFrame(rows)
    # temiz-koşucu oranına göre sırala (fırsat profili), havuz baseline'a kıyasla
    res["clean_lift"] = (res["clean_pct"] - base_clean).round(3)
    res = res.sort_values("clean_pct", ascending=False)

    lines = ["# Sinyal-Kombinasyon × MFE+MAE Fırsat Profili v0", ""]
    lines.append(f"- havuz: {len(ev):,} benzersiz olay, {a.asof_min}+ | ufuk {FWD}g")
    lines.append(f"- HAVUZ BASELINE: MFE_med {base_mfe:+.3f} | MAE_med {base_mae:+.3f} | "
                 f"temiz-koşucu %{base_clean*100:.1f}")
    lines.append(f"- 'temiz koşucu' = MFE≥{CLEAN_MFE:.0%} ∧ MAE≥{CLEAN_MAE:.0%} (capture YOK, fırsat-tavan)")
    lines.append(f"- ⚠️ replay'e giren: nox_v3/smart_breakout (audit-geçti); "
                 f"regime/alpha forward-log'da (henüz YOK)")
    lines.append("")
    lines.append("## En yüksek temiz-koşucu oranlı kombinasyonlar (min-n=%d)" % a.min_n)
    lines.append("| k | kombinasyon | n | MFE_med | MAE_med | temiz%% | baseline-lift |".replace("%%", "%"))
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in res.head(30).iterrows():
        lines.append(f"| {r['k']} | {r['komb']} | {r['n']} | {r['mfe_med']:+.3f} | "
                     f"{r['mae_med']:+.3f} | {r['clean_pct']*100:.1f} | {r['clean_lift']*100:+.1f} |")
    lines.append("")
    lines.append("## Tekil sinyaller (referans)")
    lines.append("| sinyal | n | MFE_med | MAE_med | temiz% | lift |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in res[res["k"] == 1].sort_values("clean_pct", ascending=False).iterrows():
        lines.append(f"| {r['komb']} | {r['n']} | {r['mfe_med']:+.3f} | {r['mae_med']:+.3f} | "
                     f"{r['clean_pct']*100:.1f} | {r['clean_lift']*100:+.1f} |")

    DST_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[komb] → {DST_MD}")
    print("\n".join(lines[:8]))
    print("... en iyi 5:")
    for _, r in res.head(5).iterrows():
        print(f"  {r['komb']}: n={r['n']} temiz%{r['clean_pct']*100:.1f} "
              f"(lift {r['clean_lift']*100:+.1f}) MFE{r['mfe_med']:+.2f}/MAE{r['mae_med']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
