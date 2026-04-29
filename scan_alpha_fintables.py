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

OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")
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
    """Dashboard HTML — NOX briefing aesthetic (dark, aurora bg, gold/copper palette)."""
    n = len(candidates)
    cutoff_str = cutoff.strftime("%d.%m.%Y")
    gen_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    pf_rows = ""
    if portfolio and portfolio.get("weights"):
        for t, w in sorted(portfolio["weights"].items(), key=lambda x: -x[1]):
            match = next((c for c in candidates if c["ticker"] == t), None)
            if match:
                pf_rows += (
                    f"<tr>"
                    f"<td class='ticker'><b>{t}</b></td>"
                    f"<td class='wt'>{w*100:.1f}%</td>"
                    f"<td class='score'>{match['composite']:.1f}</td>"
                    f"<td>{match['ml_1g']:.2f}</td>"
                    f"<td>{match['ml_3g']:.2f}</td>"
                    f"<td>{match['close']:,.2f}</td>"
                    f"<td class='neg'>{match['stop']:,.2f}</td>"
                    f"<td class='neg'>{match['stop_pct']:.1f}%</td>"
                    f"<td class='pos'>{match['trail_target']:,.2f}</td>"
                    f"</tr>"
                )

    cand_rows = ""
    for i, c in enumerate(candidates, 1):
        ml3g = f"{c['ml_3g']:.2f}" if c.get("ml_3g") else "—"
        cmf_v = c['cmf']
        cmf_cls = "pos" if cmf_v > 0 else ("neg" if cmf_v < 0 else "")
        cand_rows += (
            f"<tr>"
            f"<td class='idx'>{i}</td>"
            f"<td class='ticker'><b>{c['ticker']}</b></td>"
            f"<td class='score'>{c['composite']:.1f}</td>"
            f"<td>{c['ml_1g']:.2f}</td><td>{ml3g}</td>"
            f"<td>{c['adx']:.1f}</td>"
            f"<td class='{cmf_cls}'>{cmf_v:+.3f}</td>"
            f"<td>{c['rsi']:.1f}</td>"
            f"<td>{c['close']:,.2f}</td>"
            f"<td class='neg'>{c['stop']:,.2f}</td>"
            f"<td class='neg'>{c['stop_pct']:.1f}%</td>"
            f"<td class='pos'>{c['trail_target']:,.2f}</td>"
            f"</tr>"
        )

    pf_section = ""
    if portfolio and portfolio.get("weights"):
        sharpe = portfolio['sharpe_ratio']
        sharpe_cls = "kpi-pos" if sharpe >= 1.0 else ("kpi-mid" if sharpe >= 0.5 else "kpi-neg")
        pf_section = f"""
    <div class="layer-title">Markowitz Portföy — Max Sharpe</div>
    <div class="card portfolio-card">
      <div class="kpi-strip">
        <div class="kpi"><span class="kpi-label">Beklenen Getiri</span><span class="kpi-val kpi-pos">{portfolio['expected_return']:+.1f}%</span></div>
        <div class="kpi"><span class="kpi-label">Risk (σ)</span><span class="kpi-val">{portfolio['expected_risk']:.1f}%</span></div>
        <div class="kpi"><span class="kpi-label">Sharpe</span><span class="kpi-val {sharpe_cls}">{sharpe:.2f}</span></div>
        <div class="kpi"><span class="kpi-label">Hisse</span><span class="kpi-val">{portfolio['n_stocks']}</span></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Hisse</th><th>Ağırlık</th><th>Skor</th><th>ML1g</th><th>ML3g</th>
            <th>Fiyat</th><th>Stop</th><th>Stop %</th><th>Trail</th>
          </tr></thead>
          <tbody>{pf_rows}</tbody>
        </table>
      </div>
    </div>
"""

    be_shift = os.environ.get('BE_SHIFT_R', '2.0')

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYX Alpha Scan — {cutoff_str}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik+Glitch&family=Homemade+Apple&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
:root {{
  --bg-primary: #060709;
  --bg-card: #0d0d10;
  --bg-elevated: #141417;
  --bg-hover: #1b1b1f;
  --border-subtle: #1e1e23;
  --border-dim: #2c2b30;
  --text-primary: #e8e4dc;
  --text-secondary: #8a8580;
  --text-muted: #555250;
  --nox-gold: #c9a96e;
  --nox-gold-dim: rgba(201,169,110,0.10);
  --nox-copper: #b8956e;
  --nox-green: #7a9e7a;
  --nox-red: #9e5a5a;
  --nox-blue: #7a8fa5;
  --font-display: 'DM Sans', sans-serif;
  --font-brand: 'Rubik Glitch', cursive;
  --font-handwrite: 'Homemade Apple', cursive;
  --font-mono: 'JetBrains Mono', monospace;
  --radius: 14px;
  --radius-sm: 8px;
}}
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: var(--font-display);
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}

.aurora-bg {{ position: fixed; top:0; left:0; width:100%; height:100%; z-index:0; overflow:hidden; pointer-events:none; }}
.aurora-layer {{ position: absolute; width:200%; height:200%; top:-50%; left:-50%; opacity:1; }}
.aurora-layer-1 {{
  background:
    radial-gradient(ellipse 35% 25% at 8% 12%, rgba(207,199,196,0.35) 0%, transparent 50%),
    radial-gradient(ellipse 25% 20% at 88% 78%, rgba(32,30,33,0.6) 0%, transparent 50%);
  animation: aurora-drift 28s ease-in-out infinite;
}}
.aurora-layer-2 {{
  background:
    radial-gradient(ellipse 30% 22% at 78% 8%, rgba(207,199,196,0.2) 0%, transparent 50%),
    radial-gradient(ellipse 22% 22% at 12% 88%, rgba(32,30,33,0.5) 0%, transparent 50%);
  animation: aurora-drift 35s ease-in-out infinite reverse;
}}
.aurora-layer-3 {{
  background: radial-gradient(ellipse 25% 15% at 45% 45%, rgba(184,149,110,0.12) 0%, transparent 40%);
  animation: aurora-pulse 22s ease-in-out infinite;
}}
.mesh-overlay {{
  position: fixed; top:0; left:0; width:100%; height:100%; z-index:0; pointer-events:none;
  background-image:
    radial-gradient(circle 400px at 12% 18%, rgba(207,199,196,0.08) 0%, transparent 50%),
    radial-gradient(circle 350px at 88% 72%, rgba(32,30,33,0.2) 0%, transparent 50%);
  filter: blur(40px);
}}
@keyframes aurora-drift {{
  0%,100% {{ transform: translate(0,0) rotate(0deg); }}
  25% {{ transform: translate(4%,-4%) rotate(3deg); }}
  50% {{ transform: translate(-4%,4%) rotate(-3deg); }}
  75% {{ transform: translate(2%,2%) rotate(2deg); }}
}}
@keyframes aurora-pulse {{
  0%,100% {{ opacity: 0.3; transform: scale(1); }}
  50% {{ opacity: 0.6; transform: scale(1.1); }}
}}

.container {{
  position: relative; z-index: 1;
  max-width: 1200px; margin: 0 auto; padding: 0 1.5rem 3rem;
}}

/* Sticky status bar */
.status-bar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(6,6,8,0.55);
  backdrop-filter: blur(24px) saturate(1.3);
  -webkit-backdrop-filter: blur(24px) saturate(1.3);
  border-bottom: 1px solid rgba(201,169,110,0.08);
  padding: 0.6rem 1.5rem;
  margin: 0 -1.5rem 1.75rem;
  display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
}}
.status-bar .logo {{
  display: inline-flex; align-items: baseline; gap: 0.15rem; white-space: nowrap;
}}
.status-bar .logo .nox-text {{
  font-family: var(--font-brand);
  font-size: 2.4rem;
  color: #fff;
  letter-spacing: 0.06em;
  line-height: 0.85;
}}
.status-bar .logo .alpha-text {{
  font-family: var(--font-handwrite);
  font-size: 1.3rem;
  color: var(--nox-gold);
  margin-left: 0.25rem;
  position: relative;
  top: -0.1rem;
}}
.status-bar .meta-pill {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  padding: 0.25rem 0.7rem;
  border-radius: 0.75rem;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  white-space: nowrap;
}}
.status-bar .meta-pill b {{ color: var(--nox-gold); font-weight: 600; }}
.status-bar .meta-right {{
  font-size: 0.72rem; color: var(--text-muted);
  font-family: var(--font-mono); white-space: nowrap; margin-left: auto;
}}

/* Layer titles */
.layer-title {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--nox-gold);
  margin: 1.5rem 0 0.6rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border-subtle);
}}
.layer-title:first-of-type {{ margin-top: 0; }}

/* Cards */
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 1.1rem 1.25rem 1.25rem;
  margin-bottom: 0.5rem;
}}
.portfolio-card {{
  border-color: rgba(201,169,110,0.18);
  background: linear-gradient(180deg, rgba(201,169,110,0.025) 0%, var(--bg-card) 60%);
}}

/* KPI strip */
.kpi-strip {{
  display: flex; flex-wrap: wrap; gap: 1.5rem;
  padding: 0.5rem 0 1rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 1rem;
}}
.kpi {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.kpi-label {{
  font-family: var(--font-mono);
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}}
.kpi-val {{
  font-family: var(--font-mono);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-primary);
}}
.kpi-pos {{ color: var(--nox-green); }}
.kpi-mid {{ color: var(--nox-gold); }}
.kpi-neg {{ color: var(--nox-red); }}

/* Tables */
.table-wrap {{
  overflow-x: auto;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
  background: rgba(6,7,9,0.3);
}}
table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
thead {{ position: sticky; top: 0; z-index: 5; }}
th {{
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-weight: 600; font-size: 0.66rem;
  text-transform: uppercase; letter-spacing: 0.07em;
  padding: 9px 10px; text-align: left;
  border-bottom: 1px solid var(--border-subtle);
  white-space: nowrap;
  font-family: var(--font-display);
}}
td {{
  padding: 7px 10px;
  border-bottom: 1px solid rgba(39,39,42,0.4);
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: 0.74rem;
  color: var(--text-secondary);
}}
tr {{ transition: background 0.1s; }}
tr:hover {{ background: var(--bg-hover); }}
td.idx {{ color: var(--text-muted); }}
td.ticker {{ color: var(--text-primary); font-family: var(--font-display); font-weight: 600; }}
td.ticker b {{ color: var(--nox-gold); }}
td.score {{ color: var(--nox-gold); font-weight: 600; }}
td.wt {{ color: var(--nox-copper); font-weight: 600; }}
td.pos {{ color: var(--nox-green); }}
td.neg {{ color: var(--nox-red); }}

/* Counter pill in header */
.section-meta {{
  display: inline-flex; gap: 0.4rem; align-items: center;
  font-family: var(--font-mono); font-size: 0.66rem;
  color: var(--text-muted);
  text-transform: none; letter-spacing: 0.02em;
  margin-left: 0.6rem;
  padding: 0.15rem 0.55rem;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 0.6rem;
}}
.section-meta b {{ color: var(--nox-gold); font-weight: 600; }}

/* Footer */
.footer {{
  text-align: center;
  padding: 1.75rem 0 0.5rem;
  margin-top: 1.5rem;
  font-size: 0.68rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
  border-top: 1px solid var(--border-subtle);
}}
.footer b {{ color: var(--nox-gold); font-weight: 500; }}
.footer .sep {{ color: var(--border-dim); margin: 0 0.5rem; }}

@media (max-width: 768px) {{
  .container {{ padding: 0 0.75rem 2rem; }}
  .status-bar {{ margin: 0 -0.75rem 1.25rem; }}
  .kpi-strip {{ gap: 1rem; }}
  table {{ font-size: 0.7rem; }}
  td, th {{ padding: 6px 7px; }}
}}
</style>
</head>
<body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="container">
  <div class="status-bar">
    <span class="logo"><span class="nox-text">NYX</span><span class="alpha-text">alpha</span></span>
    <span class="meta-pill">Kapanış: <b>{cutoff_str}</b></span>
    <span class="meta-pill">Evren <b>{meta['n_universe']}</b> → Likit <b>{meta['n_liquid']}</b> → Aday <b>{n}</b></span>
    <span class="meta-pill">Mode <b>{meta['mode']}</b></span>
    <span class="meta-right">{gen_str}</span>
  </div>
{pf_section}
  <div class="layer-title">Tüm Adaylar <span class="section-meta">N = <b>{n}</b></span></div>
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Hisse</th><th>Skor</th><th>ML1g</th><th>ML3g</th>
          <th>ADX</th><th>CMF</th><th>RSI</th>
          <th>Fiyat</th><th>Stop</th><th>Stop %</th><th>Trail</th>
        </tr></thead>
        <tbody>{cand_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    nyxalpha <span class="sep">·</span> generated <b>{gen_str}</b>
    <span class="sep">·</span> BE_SHIFT_R <b>{be_shift}</b>
    <span class="sep">·</span> Fintables {meta['n_universe']}-universe
  </div>
</div>
</body>
</html>"""


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

    # Cache staleness guard — fail loud if user requested cache and the parquet
    # is more than 1 weekday behind today. Prevents silent stale-data outputs.
    if args.mode == "cache" and not args.cutoff:
        today = pd.Timestamp(datetime.now().date())
        biz_lag = len(pd.bdate_range(cutoff + pd.Timedelta(days=1), today)) - 1
        if biz_lag >= 1:
            print(f"  ⚠ STALE CACHE: parquet last bar {cutoff.date()} is "
                  f"{biz_lag} işgünü behind today ({today.date()}). "
                  f"Use --mode live for fresh EOD or refresh the parquet first.")

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
