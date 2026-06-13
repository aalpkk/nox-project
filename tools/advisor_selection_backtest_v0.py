#!/usr/bin/env python3
"""
Advisor aday-seçim backtest'i v0 — ÇEKİRDEK EDGE TESTİ (replay'siz).

Soru: advisor'ın AL adaylarını sıralarken kullandığı seçim feature'ları
(DE konfluens: n_concurrent_sources/families/timeframes, setup_label, family,
cluster3 üyeliği, risk metrikleri) gerçekten İYİ fırsat profilini KÖTÜ'den
ayrıştırıyor mu — havuz İÇİNDE (random evren değil), gerçekçi capture ile.

Kilitli disiplin (memory):
  - GOOD-vs-WITHIN-pool-FAILED, random-vs-değil [feedback_separability_metric_design]
  - MFE tek başına fantezi → MFE20+MAE20 JOINT fırsat profili + realized_pct20
    capture reality-check (panelde hazır) [tavan/cluster hatları]
  - verdict PROCEED_WEIGHTS / NO_EDGE_KEEP_HEURISTIC

Çıktı: output/advisor_selection_backtest_v0.md + _weights.json

Kullanım:
  python tools/advisor_selection_backtest_v0.py [--asof-min 2024-01-01] [--no-mae]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # data.intraday_1h vb. repo-kök importları için
OUT = ROOT / "output"
PANEL = OUT / "ranking_lab_features_v0.parquet"
EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"
CLUSTER3 = OUT / "ranking_lab_cluster3_paper_snapshots_v0.parquet"
DST_MD = OUT / "advisor_selection_backtest_v0.md"
DST_WEIGHTS = OUT / "advisor_selection_backtest_v0_weights.json"

FWD = 20                       # backtest ufku (gün)
GOOD_MFE_Q = 0.70              # GOOD: MFE havuz-içi p70 üstü
GOOD_MAE_FLOOR = -0.08         # GOOD: MAE −%8'den sığ (cehennem-dönüşü değil)
FAIL_MFE_Q = 0.30             # FAILED: MFE p30 altı VEYA derin MAE
FAIL_MAE_DEEP = -0.15
NULL_PERM = 1000

# Era split (veri penceresi 2024+; yıl-bazlı OOS işaret-tutarlılık)
ERAS = [("2024", "2024-01-01", "2025-01-01"),
        ("2025", "2025-01-01", "2026-01-01"),
        ("2026", "2026-01-01", "2027-01-01")]


V1_SL, V1_TRAIL, V1_H = 0.10, 0.02, 25


def _v1_exit_return(e, H, L, C):
    """VERBATIM tavan walk_forward_v1 exit (SL−10/TP1+4 yarı/trail2/H25).
    e: entry vec; H/L/C: (H25, n) ileri-bar dizileri."""
    n = len(e)
    sl_level = e * (1 - V1_SL)
    tp1_level = e * 1.04
    state = np.zeros(n, dtype=np.int8)
    locked = np.zeros(n)
    current_stop = sl_level.copy()
    max_high = np.full(n, -np.inf)
    ret = np.full(n, np.nan)
    for day in range(V1_H):
        h_d, l_d, c_d = H[day], L[day], C[day]
        valid = np.isfinite(h_d) & np.isfinite(l_d) & np.isfinite(c_d)
        m_pre = (state == 0) & valid
        if m_pre.any():
            sl_hit = m_pre & (l_d <= current_stop)
            tp1_hit = m_pre & ~sl_hit & (h_d >= tp1_level)
            ret[sl_hit] = (current_stop[sl_hit] - e[sl_hit]) / e[sl_hit]
            state[sl_hit] = 2
            locked[tp1_hit] = 0.02
            state[tp1_hit] = 1
            max_high[tp1_hit] = np.maximum(max_high[tp1_hit], h_d[tp1_hit])
        m_post = (state == 1) & valid
        if m_post.any():
            killed = m_post & (l_d <= current_stop)
            ret[killed] = locked[killed] + 0.5 * (current_stop[killed] - e[killed]) / e[killed]
            state[killed] = 2
            still = m_post & ~killed
            max_high[still] = np.maximum(max_high[still], h_d[still])
            new_trail = np.maximum(e, max_high - V1_TRAIL * e)
            current_stop[still] = np.maximum(current_stop[still], new_trail[still])
    surv = (state != 2)
    last_close = np.full(n, np.nan)
    for day in range(V1_H):
        ok = np.isfinite(C[day])
        last_close[ok] = C[day][ok]
    ret[surv] = (last_close[surv] - e[surv]) / e[surv]
    return ret


def compute_forward_outcomes(panel: pd.DataFrame):
    """Tek extfeed yüklemesinde MAE20 + V1-exit capture (gerçekçi çıkış).
    V1-capture = sistemin GERÇEK çıkışı; d20-hold (realized_pct_20d) panelde hazır
    ama strawman (round-trip'e izin verir). Kullanıcı uyarısı: capture neye göreyse
    feature ona göre fantezi/gerçek görünür → V1 ile yargıla."""
    print("  forward MAE + V1-exit capture (extfeed)...")
    from data.intraday_1h import daily_resample
    bars = pd.read_parquet(EXTFEED, columns=["ticker", "ts_istanbul", "open",
                                             "high", "low", "close", "volume"])
    daily = daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = daily.groupby("ticker")
    for d in range(1, V1_H + 1):
        daily[f"d{d}_high"] = g["high"].shift(-d)
        daily[f"d{d}_low"] = g["low"].shift(-d)
        daily[f"d{d}_close"] = g["close"].shift(-d)
    fcols = ["ticker", "date"] + [f"d{d}_{x}" for d in range(1, V1_H + 1)
                                  for x in ("high", "low", "close")]
    ev = panel[["ticker", "signal_date", "close_signal_date"]].copy()
    ev["ticker"] = ev["ticker"].astype(str)
    ev["date"] = pd.to_datetime(ev["signal_date"]).dt.normalize()
    ev = ev.merge(daily[fcols], on=["ticker", "date"], how="left")

    e = ev["close_signal_date"].values.astype(float)
    H = np.array([ev[f"d{i}_high"].values for i in range(1, V1_H + 1)], float)
    L = np.array([ev[f"d{i}_low"].values for i in range(1, V1_H + 1)], float)
    C = np.array([ev[f"d{i}_close"].values for i in range(1, V1_H + 1)], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.nanmin(L[:FWD], axis=0) / e - 1.0
        v1 = _v1_exit_return(e, H, L, C)
    return (pd.Series(mae, index=panel.index).replace([np.inf, -np.inf], np.nan),
            pd.Series(v1, index=panel.index))


def cluster3_membership() -> set:
    """(ticker, signal_date) cluster3 arketip üyeliği seti."""
    if not CLUSTER3.exists():
        return set()
    c = pd.read_parquet(CLUSTER3)
    c["signal_date"] = pd.to_datetime(c["signal_date"]).dt.normalize()
    return set(zip(c["ticker"].astype(str).str.upper(), c["signal_date"]))


def bucketize(df: pd.DataFrame) -> pd.Series:
    """Havuz-içi GOOD/FAILED/NEUTRAL — GERÇEKÇİ V1-exit capture üzerinden.
    (d20-hold strawman'di: round-trip'e izin verip oynak isimleri haksız cezaladı.
    V1-exit = sistemin gerçek çıkışı, trail koşucu MFE'sini round-trip'ten önce kilitler.
    Kullanıcı uyarısı: feature fantezi mi gerçek mi capture metoduna bağlı.)
    MAE ikincil risk-filtresi."""
    cap = df["v1_capture"]
    mae = df["mae_pct_20d"]
    hi, lo = cap.quantile(GOOD_MFE_Q), cap.quantile(FAIL_MFE_Q)
    out = pd.Series("NEUTRAL", index=df.index)
    out[(cap >= hi) & (mae >= GOOD_MAE_FLOOR)] = "GOOD"
    out[(cap <= lo) | (mae <= FAIL_MAE_DEEP)] = "FAILED"
    return out


def separability(df: pd.DataFrame, feat: str, rng) -> dict:
    """Feature yüksek-yarısı GOOD-oranı vs düşük-yarısı; null-permütasyon p."""
    sub = df[df[feat].notna() & df["bucket"].isin(["GOOD", "FAILED"])].copy()
    if len(sub) < 100:
        return {"feature": feat, "n": len(sub), "verdict": "THIN"}
    good = (sub["bucket"] == "GOOD").astype(float).values
    x = sub[feat].astype(float).values
    med = np.median(x)
    hi_mask = x > med
    if hi_mask.sum() < 30 or (~hi_mask).sum() < 30:
        # kategorik/az-varyans: korelasyon temelli
        gap = np.corrcoef(x, good)[0, 1] if x.std() > 0 else 0.0
        obs = gap
        null = np.array([np.corrcoef(rng.permutation(x), good)[0, 1]
                         for _ in range(NULL_PERM)])
    else:
        obs = good[hi_mask].mean() - good[~hi_mask].mean()
        null = np.empty(NULL_PERM)
        for i in range(NULL_PERM):
            p = rng.permutation(good)
            null[i] = p[hi_mask].mean() - p[~hi_mask].mean()
    p95 = np.quantile(null, 0.95)
    pval = (null >= obs).mean()
    return {"feature": feat, "n": int(len(sub)),
            "good_lift_hi_minus_lo": round(float(obs), 4),
            "null_p95": round(float(p95), 4), "p_value": round(float(pval), 4),
            "passes": bool(obs > p95 and pval < 0.05)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof-min", default="2024-01-01",
                    help="Bu tarihten önceki event'leri at (era-relevans)")
    ap.add_argument("--no-mae", action="store_true",
                    help="MAE hesabını atla (hızlı; realized_pct20 ile yaklaşık bucket)")
    a = ap.parse_args()
    rng = np.random.default_rng(0x5E1EC7)  # "SELECT"

    print("[backtest] panel yükleniyor...")
    df = pd.read_parquet(PANEL)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df[df["signal_date"] >= pd.Timestamp(a.asof_min)].copy()
    df = df[df["mfe_pct_20d"].notna()].copy()
    print(f"  {len(df):,} event ({df['signal_date'].min().date()}→{df['signal_date'].max().date()})")

    mae, v1 = compute_forward_outcomes(df)
    df["mae_pct_20d"] = mae
    df["v1_capture"] = v1
    df = df[df["mae_pct_20d"].notna() & df["v1_capture"].notna()].copy()
    print(f"  MAE+V1-capture'lı event: {len(df):,}")

    # cluster3 üyelik flag'i
    c3 = cluster3_membership()
    df["c3_member"] = [int((t, d.normalize()) in c3)
                       for t, d in zip(df["ticker"].astype(str).str.upper(),
                                       df["signal_date"])]

    df["bucket"] = bucketize(df)
    print(f"  bucket: {df['bucket'].value_counts().to_dict()}")

    feats = ["n_concurrent_sources", "n_concurrent_families", "n_concurrent_timeframes",
             "event_multiplicity", "c3_member", "atr_pct", "vol_z_20",
             "price_vs_20d_high", "price_vs_60d_high", "ret_5d"]

    # ── tüm-dönem + era OOS separability ──
    lines = ["# Advisor Aday-Seçim Backtest v0 (çekirdek edge, replay'siz)", ""]
    lines.append(f"- havuz: {len(df):,} event, {a.asof_min}+ | ufuk {FWD}g")
    lines.append(f"- GOOD={int((df.bucket=='GOOD').sum())} FAILED={int((df.bucket=='FAILED').sum())} "
                 f"NEUTRAL={int((df.bucket=='NEUTRAL').sum())}")
    lines.append(f"- 3 lens medyan: MFE20 {df['mfe_pct_20d'].median():+.3f} (fırsat-tavan) | "
                 f"V1-capture {df['v1_capture'].median():+.3f} (GERÇEKÇİ çıkış) | "
                 f"d20-hold {df['realized_pct_20d'].median():+.3f} (strawman) | "
                 f"MAE20 {df['mae_pct_20d'].median():+.3f}")
    lines.append("")
    lines.append("## Feature separability (tüm dönem, GOOD-vs-FAILED within-pool)")
    lines.append("| feature | n | GOOD-lift(hi−lo) | null_p95 | p | geçer |")
    lines.append("|---|---|---|---|---|---|")

    all_res = {}
    for f in feats:
        r = separability(df, f, rng)
        all_res[f] = r
        if r.get("verdict") == "THIN":
            lines.append(f"| {f} | {r['n']} | — | — | — | THIN |")
        else:
            lines.append(f"| {f} | {r['n']} | {r['good_lift_hi_minus_lo']:+.4f} | "
                         f"{r['null_p95']:.4f} | {r['p_value']:.3f} | "
                         f"{'✅' if r['passes'] else '❌'} |")

    # ── era-OOS tutarlılık (yalnızca tüm-dönem geçenler) ──
    lines += ["", "## Era-OOS işaret tutarlılığı (geçen feature'lar)"]
    passers = [f for f, r in all_res.items() if r.get("passes")]
    era_consistent = {}
    if passers:
        lines.append("| feature | " + " | ".join(e[0] for e in ERAS) + " | tutarlı |")
        lines.append("|---|" + "---|" * (len(ERAS) + 1))
        for f in passers:
            signs = []
            for _, lo, hi in ERAS:
                seg = df[(df["signal_date"] >= lo) & (df["signal_date"] < hi)]
                rr = separability(seg, f, rng)
                signs.append(rr.get("good_lift_hi_minus_lo", 0.0))
            consistent = all(s > 0 for s in signs) or all(s < 0 for s in signs)
            era_consistent[f] = consistent
            lines.append(f"| {f} | " + " | ".join(f"{s:+.3f}" for s in signs) +
                         f" | {'✅' if consistent else '❌'} |")
    else:
        lines.append("(tüm-dönem geçen feature yok)")

    # ── capture-kapısı: OOS-tutarlı feature realized getiriyi de İYİLEŞTİRMELİ
    #    (MFE-fantezi koruması; atr_pct burada elenir) ──
    def _capture_positive(f):
        try:
            q = pd.qcut(df[f].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
            return df.loc[q == "Q4", "v1_capture"].median() > \
                   df.loc[q == "Q1", "v1_capture"].median()
        except Exception:
            return False

    # ── verdict + weights ──
    final = [f for f in passers if era_consistent.get(f) and _capture_positive(f)]
    dropped_fantasy = [f for f in passers if era_consistent.get(f) and not _capture_positive(f)]
    verdict = "PROCEED_WEIGHTS" if final else "NO_EDGE_KEEP_HEURISTIC"
    lines += ["", f"## VERDICT: {verdict}"]
    if dropped_fantasy:
        lines.append(f"⚠️ OOS-tutarlı ama capture-NEGATİF (MFE-fantezi, elendi): {dropped_fantasy}")
    if final:
        # ağırlık = OOS-tutarlı GOOD-lift (pozitif normalize)
        raw = {f: abs(all_res[f]["good_lift_hi_minus_lo"]) for f in final}
        s = sum(raw.values()) or 1.0
        weights = {f: round(v / s, 4) for f, v in raw.items()}
        lines.append(f"OOS-doğrulanmış seçim feature'ları: {final}")
        lines.append(f"Normalize ağırlık: {weights}")
    else:
        weights = {}
        lines.append("Hiçbir feature havuz-içi GOOD/FAILED'i OOS-tutarlı ayrıştırmadı "
                     "→ mevcut sezgisel sıralama KORUNUR (eşit-ağırlık), advisor "
                     "raporuna dürüst not. (cluster3/sektör-RS NULL kalıbı.)")

    # capture reality-check — V1-exit (gerçekçi) üzerinden, 3-lens çeyrek karşılaştırma
    lines += ["", "## Capture reality-check (V1-exit gerçekçi çıkış)"]
    g = df[df.bucket == "GOOD"]; fl = df[df.bucket == "FAILED"]
    lines.append(f"- GOOD V1-capture medyan {g['v1_capture'].median():+.3f} | "
                 f"FAILED {fl['v1_capture'].median():+.3f} | "
                 f"gap {g['v1_capture'].median()-fl['v1_capture'].median():+.3f}")
    lines.append("")
    lines.append("### TÜM feature'lar — çeyrek Q4−Q1 (MFE-tavan vs V1-capture: fantezi tespiti)")
    lines.append("| feature | MFE Q4−Q1 | V1cap Q4−Q1 | d20 Q4−Q1 | V1-pozitif |")
    lines.append("|---|---|---|---|---|")
    for f in feats:
        try:
            q = pd.qcut(df[f].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
            def qd(col):
                return df.loc[q == "Q4", col].median() - df.loc[q == "Q1", col].median()
            mfe_d, v1_d, d20_d = qd("mfe_pct_20d"), qd("v1_capture"), qd("realized_pct_20d")
            lines.append(f"| {f} | {mfe_d:+.3f} | {v1_d:+.3f} | {d20_d:+.3f} | "
                         f"{'✅' if v1_d > 0 else '❌'} |")
        except Exception:
            lines.append(f"| {f} | — | — | — | ? |")

    DST_MD.write_text("\n".join(lines), encoding="utf-8")
    DST_WEIGHTS.write_text(json.dumps(
        {"verdict": verdict, "weights": weights,
         "asof_min": a.asof_min, "horizon": FWD,
         "note": "GOOD-vs-FAILED within-pool, OOS-tutarlı; MFE+MAE joint profil"},
        indent=2), encoding="utf-8")
    print(f"[backtest] VERDICT={verdict} | {DST_MD}")
    print("\n".join(lines[-12:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
