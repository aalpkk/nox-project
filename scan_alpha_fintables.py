"""nyxalpha scan — Fintables-verified BIST universe (607 tickers).

Modes:
  --mode cache  → read output/ohlcv_6y_fintables.parquet (fast, historical)
  --mode live   → yfinance fetch for 607 fintables tickers (GHA, live eod)

Outputs:
  output/alpha_scan_fintables_{date}.csv   — all candidates
  output/alpha_scan_fintables_{date}.html  — dashboard with top-4 Markowitz portfolio

Usage:
  python scan_alpha_fintables.py [--mode cache|live] [--cutoff YYYY-MM-DD] [--push]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from alpha.config import (
    ML_SCORE_THRESHOLD, ML_SWING_THRESHOLD,
    ML_SLOPE_LOOKBACK, ML_SLOPE_MIN, ML_COMPOSITE_WEIGHT,
    CONFIRMATION_MIN_SCORE, MIN_VOLUME_TL,
)
from core.indicators import calc_atr

OHLCV_PATH = Path("output/ohlcv_6y_fintables.parquet")
# Tracked in git; also fall back to output/nyxmomentum path for dev machines
UNIVERSE_CSV_CANDIDATES = [
    Path("data/fintables_bist_equities_20260422.csv"),
    Path("output/nyxmomentum/universe/fintables_bist_equities_20260422.csv"),
]


def _load_panel_cache(cutoff: pd.Timestamp) -> dict:
    """Fintables parquet cache → {ticker.IS: df}."""
    df = pd.read_parquet(OHLCV_PATH)
    df = df[df.index <= cutoff]
    out = {}
    for tkr, sub in df.groupby("ticker"):
        sub = sub.drop(columns=["ticker"]).sort_index()
        if len(sub) < 80:
            continue
        out[f"{tkr}.IS"] = sub
    return out


def _load_panel_live(cutoff: pd.Timestamp) -> dict:
    """yfinance fetch for 607 fintables tickers."""
    from ml.dataset import fetch_all_data

    universe_csv = next((p for p in UNIVERSE_CSV_CANDIDATES if p.exists()), None)
    if universe_csv is None:
        raise FileNotFoundError(f"Fintables universe CSV missing: {UNIVERSE_CSV_CANDIDATES}")
    tickers = pd.read_csv(universe_csv)["kod"].tolist()
    print(f"  [LIVE] 607 fintables ticker yfinance fetch başlıyor...")
    data = fetch_all_data(tickers, period="1y", batch_size=50)
    out = {}
    for t, df in data.items():
        if df is None or df.empty:
            continue
        sub = df[df.index <= cutoff].sort_index()
        if len(sub) < 80:
            continue
        out[f"{t}.IS" if not t.endswith(".IS") else t] = sub[["Open", "High", "Low", "Close", "Volume"]]
    return out


def _load_xu(cutoff: pd.Timestamp) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        xu = yf.download("XU100.IS", period="2y", progress=False, auto_adjust=True)
        if xu is None or xu.empty:
            return None
        if isinstance(xu.columns, pd.MultiIndex):
            xu.columns = [c[0] if isinstance(c, tuple) else c for c in xu.columns]
        xu = xu[xu.index <= cutoff]
        return xu if not xu.empty else None
    except Exception as e:
        print(f"  [WARN] XU100 fetch: {e}")
        return None


def _build_markowitz_top4(all_data: dict, candidates: list) -> dict:
    """Top-4 candidate'a 4'lü Markowitz."""
    from alpha.portfolio import build_portfolio

    passed = candidates[:4]
    if len(passed) < 3:
        return {}
    bp_candidates = [{"ticker": c["ticker"].replace(".IS", ""),
                      "composite_score": c["composite"], "passed": True} for c in passed]
    price_bare = {t.replace(".IS", ""): df for t, df in all_data.items()}
    return build_portfolio(price_bare, bp_candidates)


def _render_html(candidates: list, portfolio: dict, cutoff: pd.Timestamp, meta: dict) -> str:
    """Dashboard HTML — all candidates table + Markowitz top-4."""
    n = len(candidates)
    pf_rows = ""
    if portfolio and portfolio.get("weights"):
        for t, w in sorted(portfolio["weights"].items(), key=lambda x: -x[1]):
            match = next((c for c in candidates if c["ticker"] == t), None)
            if match:
                pf_rows += (
                    f"<tr><td><b>{t}</b></td>"
                    f"<td>{w*100:.1f}%</td>"
                    f"<td>{match['composite']:.1f}</td>"
                    f"<td>{match['ml_1g']:.2f}</td>"
                    f"<td>{match['ml_3g']:.2f}</td>"
                    f"<td>{match['close']:,.2f}</td>"
                    f"<td>{match['stop']:,.2f}</td>"
                    f"<td>{match['stop_pct']:.1f}%</td>"
                    f"<td>{match['trail_target']:,.2f}</td></tr>"
                )
    cand_rows = ""
    for i, c in enumerate(candidates, 1):
        ml3g = f"{c['ml_3g']:.2f}" if c.get("ml_3g") else "—"
        cand_rows += (
            f"<tr><td>{i}</td><td><b>{c['ticker']}</b></td>"
            f"<td>{c['composite']:.1f}</td>"
            f"<td>{c['ml_1g']:.2f}</td><td>{ml3g}</td>"
            f"<td>{c['adx']:.1f}</td>"
            f"<td>{c['cmf']:+.3f}</td>"
            f"<td>{c['rsi']:.1f}</td>"
            f"<td>{c['close']:,.2f}</td>"
            f"<td>{c['stop']:,.2f}</td>"
            f"<td>{c['stop_pct']:.1f}%</td>"
            f"<td>{c['trail_target']:,.2f}</td></tr>"
        )

    pf_section = ""
    if portfolio and portfolio.get("weights"):
        pf_section = f"""
        <div class="box">
          <h2>🎯 Markowitz 4'lü Portföy (Max Sharpe)</h2>
          <div class="kpis">
            <div>Beklenen Getiri: <b>{portfolio['expected_return']:+.1f}%</b></div>
            <div>Risk (σ): <b>{portfolio['expected_risk']:.1f}%</b></div>
            <div>Sharpe: <b>{portfolio['sharpe_ratio']:.2f}</b></div>
            <div>Hisse: <b>{portfolio['n_stocks']}</b></div>
          </div>
          <table class="data">
            <thead><tr><th>Hisse</th><th>Ağırlık</th><th>Skor</th><th>ML1g</th><th>ML3g</th>
            <th>Fiyat</th><th>Stop</th><th>Stop%</th><th>Trail</th></tr></thead>
            <tbody>{pf_rows}</tbody>
          </table>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="utf-8"><title>nyxalpha scan — {cutoff.date()}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; color: #222; }}
h1 {{ margin: 0 0 4px; }}
.meta {{ color: #666; margin-bottom: 16px; font-size: 0.9em; }}
.box {{ background: white; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;
       box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
h2 {{ margin-top: 0; font-size: 1.1em; }}
.kpis {{ display: flex; gap: 24px; font-size: 0.95em; margin-bottom: 12px; color: #444; }}
table.data {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
table.data th {{ text-align: left; background: #f0f0f0; padding: 6px 8px; }}
table.data td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
table.data tr:hover {{ background: #f7f7f7; }}
.footer {{ color: #888; font-size: 0.8em; margin-top: 16px; }}
</style></head><body>
<h1>nyxalpha scan</h1>
<div class="meta">Kapanış: <b>{cutoff.date()}</b> · Evren: <b>{meta['n_universe']}</b> → Likit: <b>{meta['n_liquid']}</b> → Aday: <b>{n}</b> · Mode: {meta['mode']}</div>
{pf_section}
<div class="box">
  <h2>Tüm Adaylar ({n})</h2>
  <table class="data">
    <thead><tr><th>#</th><th>Hisse</th><th>Skor</th><th>ML1g</th><th>ML3g</th><th>ADX</th><th>CMF</th><th>RSI</th>
    <th>Fiyat</th><th>Stop</th><th>Stop%</th><th>Trail</th></tr></thead>
    <tbody>{cand_rows}</tbody>
  </table>
</div>
<div class="footer">nyxalpha · {datetime.now().strftime('%Y-%m-%d %H:%M')} · BE_SHIFT_R={os.environ.get('BE_SHIFT_R', '2.0')}</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cache", "live"], default="cache")
    ap.add_argument("--cutoff", type=str, default=None,
                    help="YYYY-MM-DD (default: parquet son bar veya bugün)")
    ap.add_argument("--push", action="store_true", help="GitHub Pages'e deploy")
    args = ap.parse_args()

    t_all = time.time()

    # Cutoff
    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    elif args.mode == "cache" and OHLCV_PATH.exists():
        cutoff = pd.read_parquet(OHLCV_PATH).index.max()
    else:
        cutoff = pd.Timestamp(datetime.now().date())
    print(f"NYX ALPHA SCAN — mode={args.mode} cutoff={cutoff.date()}")

    # Veri
    t0 = time.time()
    if args.mode == "cache":
        if not OHLCV_PATH.exists():
            print(f"❌ Cache parquet yok: {OHLCV_PATH}")
            return 1
        all_data = _load_panel_cache(cutoff)
    else:
        all_data = _load_panel_live(cutoff)
    print(f"  {len(all_data)} hisse (veri yüklü, {time.time()-t0:.1f}s)")
    if not all_data:
        return 1
    n_universe = len(all_data)
    sample_t = next(iter(all_data))
    print(f"  Veri son bar: {all_data[sample_t].index[-1].date()} (örnek {sample_t})")

    # Likidite
    t0 = time.time()
    filtered = {}
    for t, df in all_data.items():
        vol_tl = (df["Close"] * df["Volume"]).tail(20).mean()
        if pd.notna(vol_tl) and vol_tl >= MIN_VOLUME_TL:
            filtered[t] = df
    all_data = filtered
    n_liquid = len(all_data)
    print(f"  {n_liquid} hisse (likidite sonrası, {time.time()-t0:.1f}s)")

    # XU100 (RS feature için)
    xu_df = _load_xu(cutoff)
    if xu_df is not None:
        print(f"  XU100: {len(xu_df)} gün (son {xu_df.index[-1].date()})")
    else:
        print("  XU100: YOK → RS feature'ları default")

    # ML scan
    from ml.scorer import MLScorer
    from ml.features import compute_all_features
    from alpha.stages import stage3_confirmation

    ml_scorer = MLScorer()
    ml_scorer._load_models()
    if not ml_scorer.loaded:
        print("❌ ML modeller yüklenemedi")
        return 2

    t1 = time.time()
    ml_candidates = []
    errors = 0
    for ticker, df in all_data.items():
        if len(df) < 80:
            continue
        try:
            feats = compute_all_features(df, xu_df=xu_df)
            if feats.empty or len(feats) < 5:
                continue
            row = feats.iloc[-1]
            vec = ml_scorer._make_feature_vector(row)
            if vec is None:
                continue
            preds = ml_scorer._predict_all(vec)
            ml_1g = preds["ml_a_1g"]
            ml_3g = preds["ml_a_3g"]
            if ml_1g is None or ml_1g < ML_SCORE_THRESHOLD:
                continue
            if ml_3g is not None and ml_3g < ML_SWING_THRESHOLD:
                continue
            if len(feats) > ML_SLOPE_LOOKBACK:
                row_ago = feats.iloc[-1 - ML_SLOPE_LOOKBACK]
                vec_ago = ml_scorer._make_feature_vector(row_ago)
                if vec_ago is not None:
                    p_ago = ml_scorer._predict_all(vec_ago)
                    if p_ago["ml_a_1g"] is not None and (ml_1g - p_ago["ml_a_1g"]) < ML_SLOPE_MIN:
                        continue
            confirmation = stage3_confirmation(df)
            if confirmation["score"] < CONFIRMATION_MIN_SCORE:
                continue
            ml_avg = ml_1g if ml_3g is None else (ml_1g + ml_3g) / 2
            composite = ml_avg * 100 * ML_COMPOSITE_WEIGHT + confirmation["score"] * 10 * (1 - ML_COMPOSITE_WEIGHT)
            atr_val = 0.0
            if len(df) >= 20:
                _atr = calc_atr(df)
                if not pd.isna(_atr.iloc[-1]):
                    atr_val = float(_atr.iloc[-1])
            close_px = float(df["Close"].iloc[-1])
            stop_dist_pct = (atr_val * 2.0 / close_px * 100) if close_px > 0 else 0
            ml_candidates.append({
                "ticker": ticker.replace(".IS", ""),
                "ml_1g": ml_1g,
                "ml_3g": ml_3g,
                "composite": round(min(100, composite), 1),
                "adx": confirmation["adx"],
                "cmf": confirmation["cmf"],
                "rsi": confirmation["rsi"],
                "conf_score": confirmation["score"],
                "close": close_px,
                "atr": atr_val,
                "stop": round(close_px - 2.0 * atr_val, 2),
                "stop_pct": round(stop_dist_pct, 1),
                "trail_target": round(close_px + 1.5 * atr_val, 2),
            })
        except Exception:
            errors += 1
            continue
    ml_candidates.sort(key=lambda x: x["composite"], reverse=True)
    print(f"  {len(ml_candidates)} aday, {errors} hata ({time.time()-t1:.0f}s)")

    # Markowitz top 4
    portfolio = _build_markowitz_top4(all_data, ml_candidates)
    if portfolio and portfolio.get("weights"):
        print(f"  Markowitz 4'lü: Sharpe {portfolio['sharpe_ratio']:.2f} | "
              f"{portfolio['expected_return']:+.1f}% @ {portfolio['expected_risk']:.1f}% risk")
        for t, w in sorted(portfolio["weights"].items(), key=lambda x: -x[1]):
            print(f"    {t:<8} {w*100:>5.1f}%")
    else:
        print("  Markowitz portföy oluşturulamadı (<4 aday veya opt hatası)")

    # Kaydet
    out_csv = Path(f"output/alpha_scan_fintables_{cutoff.date()}.csv")
    out_html = Path(f"output/alpha_scan_fintables_{cutoff.date()}.html")
    pd.DataFrame(ml_candidates).to_csv(out_csv, index=False)

    meta = {"n_universe": n_universe, "n_liquid": n_liquid, "mode": args.mode}
    html = _render_html(ml_candidates, portfolio, cutoff, meta)
    out_html.write_text(html, encoding="utf-8")
    print(f"  CSV:  {out_csv}")
    print(f"  HTML: {out_html}")
    print(f"  Toplam süre: {time.time()-t_all:.0f}s")

    # GitHub Pages push
    if args.push:
        try:
            from core.reports import push_html_to_github
            url = push_html_to_github(html, "alpha_scan.html", cutoff.strftime("%Y-%m-%d"))
            if url:
                print(f"  Pages: {url}")
        except Exception as e:
            print(f"  ⚠️ Pages push: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
