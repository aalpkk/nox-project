"""triangle_break — overlay 3 ATR-fractions on ONE candlestick panel (v0.3).

Renders a single candle chart with three upper/lower line pairs drawn
in different shades, so the line drift between ATR-k values is obvious:
  - k 0.20 → pale lines
  - k 0.25 → mid lines
  - k 0.35 → bold lines

The actual tolerance shown in title is per-ticker: tol = clip(k×ATR/close,
floor, cap). Same k can produce different absolute tol on different
tickers depending on volatility.

Usage:
    python tools/triangle_break_tol_overlay.py --ticker INTEM --tf 5h
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from channel_break.detect import _line_at, fit_geometry
from data import intraday_1h
from mb_scanner.resample import per_ticker_panel, to_5h, to_daily, to_monthly, to_weekly
from triangle_break.schema import FAMILIES

OUT_DIR = Path("output")
TF_TO_FAM = {"5h": "tr_5h", "1d": "tr_1d", "1w": "tr_1w", "1M": "tr_1M"}
TF_RESAMPLER = {
    "5h": (to_5h, "ts_istanbul"),
    "1d": (to_daily, "date"),
    "1w": (to_weekly, "week_end"),
    "1M": (to_monthly, "month_end"),
}

KS = [0.20, 0.25, 0.35]

# Three-shade palette (low → bright)
UPPER_SHADES = ["#7a6a3a", "#b08840", "#f7c245"]   # gold pale → bright
LOWER_SHADES = ["#3a4a5a", "#5a7090", "#7da6d6"]   # blue pale → bright
LINE_WIDTHS  = [1.2, 1.6, 2.0]
LINE_DASH    = ["dot", "dash", "solid"]


def _panel(ticker: str, tf: str) -> pd.DataFrame:
    bars = intraday_1h.load_intraday(tickers=[ticker], min_coverage=0.0)
    if bars.empty:
        raise SystemExit(f"no bars for {ticker}")
    resampler, idx_col = TF_RESAMPLER[tf]
    panels = per_ticker_panel(resampler(bars), idx_col)
    df = panels.get(ticker)
    if df is None or df.empty:
        raise SystemExit(f"empty panel for {ticker} on {tf}")
    return df


def _classify_subtype(s_u: float, s_l: float, flat_th: float) -> str:
    u_flat = abs(s_u) < flat_th
    l_flat = abs(s_l) < flat_th
    if l_flat and s_u < 0:
        return "ascending"
    if u_flat and s_l > 0:
        return "descending"
    if s_u < 0 and s_l > 0:
        return "symmetric"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--tf", choices=list(TF_TO_FAM), default="1d")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fam = TF_TO_FAM[args.tf]
    base = FAMILIES[fam]
    df = _panel(args.ticker, args.tf)
    asof_idx = len(df) - 1
    n = len(df)

    # Run fit at each ATR-fraction k
    fits = []
    for k in KS:
        params = replace(base, containment_tol_atr_k=k)
        geom = fit_geometry(df, asof_idx, params)
        fits.append((k, params, geom))

    # Window: cover all fits' first_pivot back through asof
    firsts = [int(g["first_pivot_idx"]) for _, _, g in fits if g is not None]
    win_lo = max(0, min(firsts) - 4) if firsts else max(0, asof_idx - 60)
    win_hi = min(n - 1, asof_idx + 8)
    sub = df.iloc[win_lo:win_hi + 1]
    x = pd.to_datetime(sub.index)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x, open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"],
            increasing_line_color="#7a9e7a",
            decreasing_line_color="#a86a6a",
            increasing_fillcolor="#7a9e7a",
            decreasing_fillcolor="#a86a6a",
            line=dict(width=1), showlegend=False, name="",
        )
    )

    # Pre-compute ATR at asof for tol resolution display
    from channel_break.detect import _atr  # local import to keep module list lean
    atr_series = _atr(df, n=base.atr_n)
    atr_at = float(atr_series.iat[asof_idx]) if pd.notna(atr_series.iat[asof_idx]) else float("nan")
    close_at = float(df["close"].iat[asof_idx])

    summary_lines = []
    for idx, (k_val, params_k, geom) in enumerate(fits):
        raw = (k_val * atr_at / close_at) if (atr_at and close_at) else float("nan")
        eff = max(params_k.containment_tol_min, min(params_k.containment_tol_max, raw))
        clipped_tag = ""
        if abs(eff - raw) > 1e-9:
            clipped_tag = " (clipped)"
        if geom is None:
            summary_lines.append(
                f"<b>k={k_val:.2f}</b> · tol={eff*100:.2f}%{clipped_tag}: no fit"
            )
            continue
        first = int(geom["first_pivot_idx"])
        line_lo = first
        line_hi = asof_idx
        line_idxs = list(range(line_lo, line_hi + 1))
        line_x = pd.to_datetime(df.index[line_lo:line_hi + 1])
        upper = [_line_at(geom["s_u"], geom["b_u"], i) for i in line_idxs]
        lower = [_line_at(geom["s_l"], geom["b_l"], i) for i in line_idxs]

        tag = f"k={k_val:.2f} (tol {eff*100:.2f}%)"
        fig.add_trace(
            go.Scatter(
                x=line_x, y=upper, mode="lines",
                line=dict(color=UPPER_SHADES[idx], width=LINE_WIDTHS[idx], dash=LINE_DASH[idx]),
                name=f"upper @ {tag}",
                hovertemplate=f"upper @ {tag} · %{{y:.2f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_x, y=lower, mode="lines",
                line=dict(color=LOWER_SHADES[idx], width=LINE_WIDTHS[idx], dash=LINE_DASH[idx]),
                name=f"lower @ {tag}",
                hovertemplate=f"lower @ {tag} · %{{y:.2f}}<extra></extra>",
            )
        )

        s_u, s_l = float(geom["s_u"]), float(geom["s_l"])
        c_ref = close_at
        flat_th_abs = base.flat_slope_pct_per_bar / 100.0 * c_ref
        sub_label = _classify_subtype(s_u, s_l, flat_th_abs)
        n_h = len(geom["h_idxs"])
        n_l = len(geom["l_idxs"])
        slope_u_pct = (s_u / c_ref) * 100
        slope_l_pct = (s_l / c_ref) * 100
        first_date = pd.to_datetime(df.index[first]).strftime("%Y-%m-%d")

        summary_lines.append(
            f"<span style='color:{UPPER_SHADES[idx]};'>●</span>"
            f"<span style='color:{LOWER_SHADES[idx]};'>●</span> "
            f"<b>k={k_val:.2f}</b> · tol {eff*100:.2f}%{clipped_tag} · "
            f"{sub_label} · {n_h}H/{n_l}L · "
            f"slope U {slope_u_pct:+.3f}% / L {slope_l_pct:+.3f}% · "
            f"width {geom['channel_width_pct']*100:.1f}% · "
            f"first pivot {first_date}"
        )

    fig.add_vline(
        x=pd.to_datetime(df.index[asof_idx]),
        line=dict(color="#c7bdbe", width=1, dash="dot"),
    )

    summary_html = "<br>".join(summary_lines)
    fig.update_layout(
        height=620,
        title=dict(
            text=(
                f"<b>{args.ticker}</b> · {args.tf} · tolerance overlay<br>"
                f"<span style='font-size:11px;color:#9aa0a6;'>{summary_html}</span>"
            ),
            x=0, font=dict(size=14),
        ),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a", plot_bgcolor="#111214",
        font=dict(family="Inter, sans-serif", size=11, color="#c7bdbe"),
        margin=dict(l=40, r=20, t=160, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0,
            xanchor="right", x=1.0,
            bgcolor="rgba(20,20,20,0.6)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, showgrid=True,
                     gridcolor="rgba(199,189,190,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(199,189,190,0.06)", zeroline=False)

    out_path = Path(args.out) if args.out else OUT_DIR / f"triangle_break_{args.ticker}_{args.tf}_overlay.html"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"[overlay] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
