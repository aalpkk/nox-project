#!/usr/bin/env python3
r"""NYX ML-Rank — BIST-100 içi çapraz-kesit ML sıralama tarayıcısı.

Ne yapar: her seans, BIST-100 evreninde likidite kapısını geçen hisseleri
`MLScorer universe_up_1g` skoruna göre sıralar ve ilk N'i yayınlar.

NEDEN EŞİK DEĞİL SIRALAMA — `output/index_bt/BIST100_BIST30_TARAMA_BACKTEST.md`
§6.6'daki ölçüme dayanır:
  · Mutlak `ml_1g` eşiği DURAĞAN DEĞİL: 0.65 eşiği 2022'de 102, 2024'te 9 sinyal
    üretiyor (11 kat). Aynı sayı, farklı seçicilik.
  · Eşik profilinde plato yok; optimum verinin kenarında ve konumu istatistiksel
    olarak tanımlanabilir değil (0.65'te %95 GA [1.80, 6.41], 0.72'de [−3.78, 7.60]).
  · Gün içi top-N ise N'de monoton ve uçurumsuz: N=1..20'nin hepsi rastgele
    seçimin p95'ini geçiyor, güven aralıklarının hiçbiri sıfırı içermiyor.
  · Dönem dayanıklılığı: top-4 model-içi/val/OOS üçünde de sıfırın üstünde
    (+0.87 / +2.23 / +1.75), mutlak eşiklerin GA'sı en az bir dönemde sıfırı içeriyor.

N BİR İSTATİSTİK PARAMETRESİ DEĞİL, KAPASİTE PARAMETRESİDİR. Backtest argmax'ı
(N=1) seçilmez; N eşzamanlı taşınacak pozisyon sayısına eşitlenir. XU100/5 gün
beklenen seçicilik: N=1 +1.55 · N=2 +1.26 · N=4 +1.10 · N=8 +1.00 puan
(rastgele seçim tabanına göre sırasıyla +0.83 / +0.64 / +0.48 / +0.34).

KOVALAMA KORUMASI skor tavanıyla değil giriş kuralıyla: skor yükseldikçe sinyal
günü getirisi de yükseliyor (ml≥0.70 kovasında ortalama +%9.82 — yani "bugün
zaten tavana yakın kapatmış"). Bu yüzden skor kırpılmaz, sinyal günü getirisi
`--chase-max`'ı aşan satır elenir.

BIST-30 ÇIKARILMAZ. Ölçüm (§6.8): XU100 top-4 +1.10 vs XU100\XU030 top-4 +1.11 —
fark yok, seçimlerin %77'si zaten aynı, dönemler arası yön değiştiriyor. BIST-30
üyeliği çıktıda yalnızca BİLGİ olarak işaretlenir.

UYARI: BIST-30 evreninde TEK BAŞINA koşturulmamalıdır — 30 isim içinden 4 seçmek
rastgele seçimin p95'ini geçmiyor (+0.87 vs p95 0.90).

Kullanım:
  python tools/ml_rank_scan.py --topn 4
  python tools/ml_rank_scan.py --topn 4 --asof 2026-07-14 --outdir output
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha.config import MIN_VOLUME_TL, MIN_DATA_DAYS, POSITION_STOP_ATR_MULT  # noqa: E402
from core.indicators import calc_atr                                          # noqa: E402
from tools.index_universe import CURRENT, CURRENT_ASOF, load_membership       # noqa: E402

OHLCV = ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
XU_CACHE = ROOT / "output" / "xu100_cache.parquet"
PANEL = ROOT / "output" / "alpha_bt" / "panel_v2.parquet"
DAYSCORE_REF = ROOT / "output" / "ml_rank_dayscore_ref.csv"
TZ_TR = timezone(timedelta(hours=3))

CHASE_MAX_PCT = 7.0      # sinyal günü getirisi bunu aşarsa kovalama sayılır → elenir
MIN_BARS = 120
MODELS_DIR = ROOT / "ml" / "models"
LF_MODELS_DIR = ROOT / "output" / "_ml_models_lf"


def _lf_model_dir() -> str:
    """LightGBM `.txt` modelleri CRLF checkout'ta parse edilemiyor (Windows).

    `scripts/alpha_scan_backtest_panel.py` bunu ALPHA_BT_MODEL_DIR ile elle
    çözüyordu; burada LF'e normalize edilmiş bir kopya otomatik üretilir ve
    kaynak dosya değiştiğinde tazelenir. LF ortamında kopya birebir aynıdır.
    """
    LF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for src in MODELS_DIR.iterdir():
        if not src.is_file():
            continue
        dst = LF_MODELS_DIR / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        data = src.read_bytes()
        dst.write_bytes(data.replace(b"\r\n", b"\n") if src.suffix == ".txt" else data)
    return str(LF_MODELS_DIR)


def _day_quality(day_score: float, topn: int) -> dict:
    """Bugünün top-N ortalama skorunu tarihsel dağılımdaki yerine oturt.

    §6.9: gün skoru ileri getiriyi MONOTON öngörüyor — en düşük beşte birlik
    dilimde top-4'ün 5 günlük getirisi −0.11 (GA sıfırı içeriyor), en yüksekte
    +2.22. Yani zayıf günlerde liste yayınlanabilir ama işleme girmemek için
    ölçülmüş bir gerekçe var.

    Karşılaştırma MUTLAK SAYIYLA DEĞİL yüzdebirlikle yapılır — skorun kuyruğu
    yıllar arasında kayıyor (§6.6-C).
    """
    # Birincil kaynak: commit'lenmiş küçük referans dosyası (CI'da panel yok).
    if DAYSCORE_REF.exists():
        try:
            ref = pd.read_csv(DAYSCORE_REF)
            hist = ref.loc[ref["topn"] == topn, "day_score"]
            if len(hist) >= 250:
                pct = float((hist < day_score).mean() * 100)
                return {"available": True, "source": DAYSCORE_REF.name,
                        "day_score": round(day_score, 4), "percentile": round(pct, 1),
                        "hist_n_days": int(len(hist)),
                        "hist_median": round(float(hist.median()), 4),
                        "band": ("ZAYIF" if pct < 33 else "ORTA" if pct < 67 else "GÜÇLÜ")}
        except Exception:
            pass
    try:
        p = pd.read_parquet(PANEL, columns=["ml_1g", "vol20_tl", "n_bars", "n_feats"]).reset_index()
        memb = load_membership()
    except Exception:
        return {"available": False}
    # Tarihsel dağılım BUGÜNKÜ ile AYNI evrende kurulmalı: XU100 (PIT) üyeliği +
    # aynı likidite kapısı. Tüm 607'lik evrenden hesaplanan dağılımla
    # karşılaştırmak elmayla armut olur (medyan 0.67 vs 0.56).
    keys = set(zip(memb.loc[memb["in_xu100"], "date"], memb.loc[memb["in_xu100"], "ticker"]))
    p = p[(p["n_bars"] >= MIN_DATA_DAYS) & (p["n_feats"] >= 5)
          & (p["vol20_tl"] >= MIN_VOLUME_TL) & p["ml_1g"].notna()]
    in_uni = np.fromiter(((d, t) in keys for d, t in zip(p["date"], p["ticker"])),
                         dtype=bool, count=len(p))
    p = p[in_uni]
    p["rk"] = p.groupby("date")["ml_1g"].rank(ascending=False, method="first")
    hist = p[p["rk"] <= topn].groupby("date")["ml_1g"].mean()
    if len(hist) < 250:
        return {"available": False}
    pct = float((hist < day_score).mean() * 100)
    return {
        "available": True,
        "day_score": round(day_score, 4),
        "percentile": round(pct, 1),
        "hist_n_days": int(len(hist)),
        "hist_median": round(float(hist.median()), 4),
        "band": ("ZAYIF" if pct < 33 else "ORTA" if pct < 67 else "GÜÇLÜ"),
    }


def _universe(asof: pd.Timestamp, source: str) -> tuple[set[str], set[str], str]:
    """(xu100, xu030, kaynak etiketi)."""
    if source == "current":
        return set(CURRENT["XU100"]), set(CURRENT["XU030"]), f"matriks_snapshot_{CURRENT_ASOF}"
    memb = load_membership()
    last = memb[memb["date"] <= asof]["date"].max()
    sub = memb[memb["date"] == last]
    return (set(sub.loc[sub["in_xu100"], "ticker"]),
            set(sub.loc[sub["in_xu030"], "ticker"]),
            f"pit_{pd.Timestamp(last).date()}")


def scan(asof: str | None, topn: int, universe_source: str, chase_max: float,
         ohlcv_delta: str | None = None,
         xu_delta: str | None = None) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(OHLCV)
    raw.index = pd.to_datetime(raw.index)
    xu = pd.read_parquet(XU_CACHE)[["Open", "High", "Low", "Close", "Volume"]]

    # Overlay: master'ın bitişinden sonraki barlar ayrı dosyadan BELLEKTE eklenir.
    # Master'a yazılmaz — delta yalnızca XU100'ü kapsıyor, kısmi ekleme PIT evren
    # kurulumunu ve backtest'leri bozar (bkz. tools/build_delta_overlay.py).
    overlay_tag = None
    if ohlcv_delta:
        dl = pd.read_parquet(ohlcv_delta)
        dl.index = pd.to_datetime(dl.index)
        dl = dl[dl.index > raw.index.max()]
        if len(dl):
            overlay_tag = (f"{ohlcv_delta} · {len(dl):,} bar · "
                           f"{dl.index.min().date()}→{dl.index.max().date()}")
            raw = pd.concat([raw, dl[raw.columns.intersection(dl.columns)]]).sort_index()
    if xu_delta:
        xd = pd.read_parquet(xu_delta)
        xd.index = pd.to_datetime(xd.index)
        xd = xd[xd.index > xu.index.max()]
        if len(xd):
            xu = pd.concat([xu, xd.reindex(columns=xu.columns)]).sort_index()

    asof_ts = pd.Timestamp(asof) if asof else raw.index.max()
    raw = raw[raw.index <= asof_ts]
    last_session = raw.index.max()

    xu100, xu030, uni_tag = _universe(last_session, universe_source)
    xu = xu[xu.index <= last_session]

    from ml.scorer import MLScorer
    from ml.features import compute_all_features
    scorer = MLScorer(model_dir=_lf_model_dir())
    scorer._load_models()
    if not scorer.loaded:
        raise RuntimeError("MLScorer yüklenemedi — ml/models eksik veya bozuk")

    rows, skipped = [], {}
    for tkr in sorted(xu100 & set(raw["ticker"].unique())):
        df = raw[raw["ticker"] == tkr][["Open", "High", "Low", "Close", "Volume"]].sort_index()
        if len(df) < MIN_BARS:
            skipped[tkr] = f"yetersiz geçmiş ({len(df)} bar)"
            continue
        if df.index.max() != last_session:
            skipped[tkr] = f"son bar eksik ({df.index.max().date()})"
            continue

        vol20_tl = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1])
        if not np.isfinite(vol20_tl) or vol20_tl < MIN_VOLUME_TL:
            skipped[tkr] = f"likidite kapısı ({vol20_tl:,.0f} TL)"
            continue

        try:
            feats = compute_all_features(df, xu_df=xu)
        except Exception as e:
            skipped[tkr] = f"feature hatası: {type(e).__name__}"
            continue
        if feats.empty:
            skipped[tkr] = "feature boş"
            continue
        vec = scorer._make_feature_vector(feats.iloc[-1])
        if vec is None:
            skipped[tkr] = "feature vektörü seyrek"
            continue
        try:
            ml = float(scorer._universe_1g.predict(vec.reshape(1, -1))[0])
        except Exception as e:
            skipped[tkr] = f"predict hatası: {type(e).__name__}"
            continue

        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        atr = float(calc_atr(df).iloc[-1])
        rows.append({
            "ticker": tkr,
            "ml_1g": round(ml, 4),
            "close": round(close, 4),
            "sig_day_ret_pct": round((close / prev - 1) * 100, 2) if prev > 0 else np.nan,
            "atr14": round(atr, 4),
            "stop_2atr": round(close - POSITION_STOP_ATR_MULT * atr, 4) if np.isfinite(atr) else None,
            "stop_pct": round(-POSITION_STOP_ATR_MULT * atr / close * 100, 2)
                        if np.isfinite(atr) and close > 0 else None,
            "vol20_tl": int(vol20_tl),
            "is_xu030": tkr in xu030,
        })

    d = pd.DataFrame(rows).sort_values("ml_1g", ascending=False).reset_index(drop=True)
    if d.empty:
        return d, {"asof": str(last_session.date()), "n_scored": 0, "skipped": skipped}

    d["rank"] = np.arange(1, len(d) + 1)
    d["chase_flag"] = d["sig_day_ret_pct"] > chase_max
    # Kovalama elemesi SIRALAMADAN SONRA: elenen satırın yeri boş bırakılmaz,
    # bir sonraki aday yukarı çekilir (kapasite sabit kalsın).
    eligible = d[~d["chase_flag"]].copy()
    eligible["pick_rank"] = np.arange(1, len(eligible) + 1)
    d = d.merge(eligible[["ticker", "pick_rank"]], on="ticker", how="left")
    d["selected"] = d["pick_rank"].le(topn).fillna(False)

    meta = {
        "asof": str(last_session.date()),
        "universe": "XU100",
        "universe_source": uni_tag,
        "universe_size": len(xu100),
        "n_scored": int(len(d)),
        "n_skipped": len(skipped),
        "topn": topn,
        "chase_max_pct": chase_max,
        "n_chase_filtered": int(d["chase_flag"].sum()),
        "exit_machine": ("2xATR stop · +2R breakeven · +1.5xATR trailing · "
                         "önceki gün low kırılınca %50 TP · kalan yarı zirve−1.5xATR "
                         "trailing · 40 gün max hold"),
        "entry": "t+1 açılış",
        "backtest_ref": "output/index_bt/BIST100_BIST30_TARAMA_BACKTEST.md §6.6",
        "expected_edge_5d_pp": {"1": 1.55, "2": 1.26, "4": 1.10, "8": 1.00}.get(str(topn)),
        "generated_at_tr": datetime.now(TZ_TR).strftime("%Y-%m-%d %H:%M"),
        "overlay": overlay_tag,
        "day_quality": _day_quality(float(d.loc[d["selected"], "ml_1g"].mean()), topn),
        "skipped": skipped,
    }
    return d, meta


HTML_TPL = """<meta charset="utf-8">
<title>NYX ML-Rank — BIST-100 · {asof}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;max-width:1000px}}
 table{{border-collapse:collapse;width:100%;margin:12px 0}}
 th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}
 th{{background:#f4f4f4;text-align:center}} td:first-child,th:first-child{{text-align:left}}
 tr.sel{{background:#eaf6ea;font-weight:600}} tr.chase{{color:#999;text-decoration:line-through}}
 .note{{background:#fff8e1;border-left:4px solid #fbc02d;padding:10px 14px;margin:14px 0}}
 code{{background:#f4f4f4;padding:1px 4px}}
</style>
<h1>NYX ML-Rank — BIST-100</h1>
<p><b>{asof}</b> · evren {universe_size} hisse ({universe_source}) · skorlanan {n_scored}
 · top-{topn} · kovalama elemesi {n_chase_filtered}</p>
<div class="note">
<b>Yöntem.</b> Mutlak ML eşiği <i>kullanılmaz</i> — skorun kuyruk dağılımı yıllar
arasında 11 kat kayıyor. Bunun yerine gün içi çapraz-kesit sıralaması alınır.
<code>N</code> istatistik değil kapasite parametresidir (eşzamanlı pozisyon sayısı).
Beklenen 5 günlük seçicilik (rastgele seçim tabanına göre): N=4 → +0.48 puan.
Giriş t+1 açılış. Çıkış: {exit_machine}.
Sinyal günü getirisi %{chase_max_pct} üstü olanlar kovalama sayılıp elenir.
<b>Gün kalitesi</b> top-N ortalama skorunun tarihsel yüzdebirliğidir; en düşük
beşte birlik dilimde top-4'ün 5 günlük getirisi −0.11 (GA sıfırı içeriyor).
Dayanak: <code>{backtest_ref}</code>.
</div>
<table>
<tr><th>#</th><th>Hisse</th><th>ML</th><th>Fiyat</th><th>Gün %</th><th>ATR14</th>
<th>Stop (2×ATR)</th><th>Stop %</th><th>20g hacim (mn TL)</th><th>BIST-30</th></tr>
{rows}
</table>
<p style="color:#666;font-size:12px">Üretim {generated_at_tr} TR ·
paper-track, yatırım tavsiyesi değildir.</p>
<script id="mlrank-data" type="application/json">{payload}</script>
"""


def render_html(d: pd.DataFrame, meta: dict) -> str:
    tr = []
    for _, r in d.head(25).iterrows():
        cls = "sel" if r["selected"] else ("chase" if r["chase_flag"] else "")
        tr.append(
            f'<tr class="{cls}"><td>{int(r["rank"])}</td><td>{r["ticker"]}</td>'
            f'<td>{r["ml_1g"]:.4f}</td><td>{r["close"]:.2f}</td>'
            f'<td>{r["sig_day_ret_pct"]:+.2f}</td><td>{r["atr14"]:.3f}</td>'
            f'<td>{r["stop_2atr"]:.2f}</td><td>{r["stop_pct"]:.1f}</td>'
            f'<td>{r["vol20_tl"]/1e6:,.0f}</td><td>{"✓" if r["is_xu030"] else ""}</td></tr>')
    payload = json.dumps({"meta": {k: v for k, v in meta.items() if k != "skipped"},
                          "picks": d[d["selected"]].drop(columns=["chase_flag"])
                                    .to_dict("records")}, ensure_ascii=False)
    return HTML_TPL.format(rows="\n".join(tr), payload=payload, **meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=4,
                    help="kapasite parametresi = eşzamanlı pozisyon sayısı")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (varsayılan: master'ın son barı)")
    ap.add_argument("--universe-source", default="current", choices=["current", "pit"])
    ap.add_argument("--chase-max", type=float, default=CHASE_MAX_PCT)
    ap.add_argument("--ohlcv-delta", default=None,
                    help="master sonrası ek günlük barlar (bellekte overlay, master yazılmaz)")
    ap.add_argument("--xu-delta", default=None, help="XU100 endeksi için aynı overlay")
    ap.add_argument("--skip-below-pct", type=float, default=None,
                    help="gün skoru bu tarihsel yüzdebirliğin altındaysa listeyi ZAYIF "
                         "işaretle ve çıkış kodu 2 döndür (otomasyon için)")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    d, meta = scan(args.asof, args.topn, args.universe_source, args.chase_max,
                   ohlcv_delta=args.ohlcv_delta, xu_delta=args.xu_delta)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = meta["asof"].replace("-", "")

    if d.empty:
        print("aday yok"); return 1

    d.to_csv(outdir / f"ml_rank_scan_{stamp}.csv", index=False)
    (outdir / f"ml_rank_scan_{stamp}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    html = render_html(d, meta)
    (outdir / f"ml_rank_scan_{stamp}.html").write_text(html, encoding="utf-8")
    (outdir / "ml_rank_scan_latest.html").write_text(html, encoding="utf-8")

    sel = d[d["selected"]]
    print(f"── NYX ML-Rank · {meta['asof']} · evren {meta['universe_size']} "
          f"({meta['universe_source']}) · skorlanan {meta['n_scored']} "
          f"· atlanan {meta['n_skipped']}")
    if meta.get("overlay"):
        print(f"   overlay: {meta['overlay']}")
    dq = meta.get("day_quality") or {}
    if dq.get("available"):
        print(f"   gün kalitesi: top{args.topn} ort skor {dq['day_score']:.4f} → "
              f"tarihsel yüzdebirlik {dq['percentile']:.0f} [{dq['band']}] "
              f"(medyan gün {dq['hist_median']:.4f})")
    if meta["n_chase_filtered"]:
        ch = d[d["chase_flag"]].head(5)
        print(f"   kovalama elemesi ({meta['n_chase_filtered']}): "
              + ", ".join(f"{r.ticker} {r.sig_day_ret_pct:+.1f}%" for r in ch.itertuples()))
    print(f"\n   {'#':>2s} {'hisse':<7s} {'ML':>7s} {'fiyat':>9s} {'gün%':>6s} "
          f"{'stop':>9s} {'stop%':>6s} {'B30':>4s}")
    for r in sel.itertuples():
        print(f"   {int(r.pick_rank):2d} {r.ticker:<7s} {r.ml_1g:7.4f} {r.close:9.2f} "
              f"{r.sig_day_ret_pct:+6.2f} {r.stop_2atr:9.2f} {r.stop_pct:6.1f} "
              f"{'✓' if r.is_xu030 else '':>4s}")
    print(f"\n  → {outdir}/ml_rank_scan_{stamp}.{{csv,json,html}} + _latest.html")
    if args.skip_below_pct is not None and dq.get("available") \
            and dq["percentile"] < args.skip_below_pct:
        print(f"\n  ⚠ ZAYIF GÜN — yüzdebirlik {dq['percentile']:.0f} < "
              f"{args.skip_below_pct:.0f}; işleme girilmemesi önerilir (çıkış kodu 2)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
