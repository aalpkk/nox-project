"""triangle_break — side-by-side plot of one ticker at 3 ATR-fractions (v0.3).

Renders the same ticker × tf at containment_tol_atr_k = 0.20 / 0.25 / 0.35,
so you can see what changes visually:
  - line position (line slides to find pivot pairs that satisfy tol)
  - touch count (band thickness widens with tol)
  - subtype label may flip (asc ↔ sym ↔ desc)

Effective tol per ticker = clip(k × ATR_14 / close, lo, hi) — annotated
in each panel.

Usage:
    python tools/triangle_break_tol_compare.py --ticker AGESA --tf 1d
    python tools/triangle_break_tol_compare.py --ticker INTEM --tf 5h
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        return "ascending"   # flat support, descending top => asc-triangle
    if u_flat and s_l > 0:
        return "descending"
    if s_u < 0 and s_l > 0:
        return "symmetric"
    return "other"


def _add_panel(
    fig: go.Figure,
    col: int,
    df: pd.DataFrame,
    geom: dict,
    asof_idx: int,
    tol: float,
    title: str,
) -> None:
    n = len(df)
    first = int(geom["first_pivot_idx"])
    pad_left = 4
    pad_right = 12
    win_lo = max(0, first - pad_left)
    win_hi = min(n - 1, asof_idx + pad_right)

    sub = df.iloc[win_lo:win_hi + 1]
    x = pd.to_datetime(sub.index)
    fig.add_trace(
        go.Candlestick(
            x=x, open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"],
            increasing_line_color="#7a9e7a",
            decreasing_line_color="#a86a6a",
            increasing_fillcolor="#7a9e7a",
            decreasing_fillcolor="#a86a6a",
            line=dict(width=1), showlegend=False, name="",
        ),
        row=1, col=col,
    )

    line_lo, line_hi = first, asof_idx
    line_idxs = list(range(line_lo, line_hi + 1))
    line_x = pd.to_datetime(df.index[line_lo:line_hi + 1])
    upper = np.array([_line_at(geom["s_u"], geom["b_u"], i) for i in line_idxs])
    lower = np.array([_line_at(geom["s_l"], geom["b_l"], i) for i in line_idxs])

    upper_band_top = upper * (1 + tol)
    lower_band_bot = lower * (1 - tol)

    # tolerance band shading (upper)
    fig.add_trace(
        go.Scatter(
            x=list(line_x) + list(line_x[::-1]),
            y=list(upper_band_top) + list(upper[::-1]),
            fill="toself",
            fillcolor="rgba(201,169,110,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip", name="",
        ),
        row=1, col=col,
    )
    # tolerance band shading (lower)
    fig.add_trace(
        go.Scatter(
            x=list(line_x) + list(line_x[::-1]),
            y=list(lower) + list(lower_band_bot[::-1]),
            fill="toself",
            fillcolor="rgba(122,143,165,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip", name="",
        ),
        row=1, col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=line_x, y=upper, mode="lines",
            line=dict(color="#c9a96e", width=1.6, dash="dash"),
            showlegend=False, name="upper",
            hovertemplate="upper · %{y:.2f}<extra></extra>",
        ),
        row=1, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=line_x, y=lower, mode="lines",
            line=dict(color="#7a8fa5", width=1.6, dash="dash"),
            showlegend=False, name="lower",
            hovertemplate="lower · %{y:.2f}<extra></extra>",
        ),
        row=1, col=col,
    )

    # Touch markers — bars whose close is within tol of either line
    closes = df["close"].to_numpy(dtype=float)
    touch_h, touch_l = [], []
    for i in line_idxs:
        c = closes[i]
        u = _line_at(geom["s_u"], geom["b_u"], i)
        l = _line_at(geom["s_l"], geom["b_l"], i)
        if abs(c - u) / c <= tol:
            touch_h.append(i)
        if abs(c - l) / c <= tol:
            touch_l.append(i)
    if touch_h:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(df.index[touch_h]),
                y=[float(closes[i]) for i in touch_h],
                mode="markers",
                marker=dict(symbol="triangle-down", size=8,
                            color="#c9a96e", line=dict(width=1, color="#0a0a0a")),
                showlegend=False, name="H touch",
                hovertemplate="H touch · %{y:.2f}<extra></extra>",
            ),
            row=1, col=col,
        )
    if touch_l:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(df.index[touch_l]),
                y=[float(closes[i]) for i in touch_l],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8,
                            color="#7a8fa5", line=dict(width=1, color="#0a0a0a")),
                showlegend=False, name="L touch",
                hovertemplate="L touch · %{y:.2f}<extra></extra>",
            ),
            row=1, col=col,
        )

    # Asof
    fig.add_vline(
        x=pd.to_datetime(df.index[asof_idx]),
        line=dict(color="#c7bdbe", width=1, dash="dot"),
        row=1, col=col,
    )

    # subtitle annotation
    fig.add_annotation(
        text=title,
        xref=f"x{col} domain" if col > 1 else "x domain",
        yref=f"y{col} domain" if col > 1 else "y domain",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        showarrow=False, align="left",
        font=dict(size=11, color="#c7bdbe"),
        bgcolor="rgba(20,20,20,0.7)",
        bordercolor="rgba(199,189,190,0.2)", borderwidth=1,
        row=1, col=col,
    )


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

    from channel_break.detect import _atr  # for per-ticker tol resolution
    atr_series = _atr(df, n=base.atr_n)
    atr_at = float(atr_series.iat[asof_idx]) if pd.notna(atr_series.iat[asof_idx]) else float("nan")
    close_at = float(df["close"].iat[asof_idx])

    def _eff_tol(k_val: float) -> float:
        raw = (k_val * atr_at / close_at) if (atr_at and close_at) else float("nan")
        return max(base.containment_tol_min, min(base.containment_tol_max, raw))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"k = {kv:.2f} → tol {_eff_tol(kv)*100:.2f}%" for kv in KS
        ],
        horizontal_spacing=0.05,
    )

    for col, k_val in enumerate(KS, start=1):
        params = replace(base, containment_tol_atr_k=k_val)
        geom = fit_geometry(df, asof_idx, params)
        eff_tol = _eff_tol(k_val)
        if geom is None:
            fig.add_annotation(
                text=f"no fit @ k={k_val:.2f} (tol {eff_tol*100:.2f}%)",
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=14, color="#a86a6a"),
                row=1, col=col,
            )
            continue
        s_u, s_l = float(geom["s_u"]), float(geom["s_l"])
        c_ref = close_at
        flat_th_abs = base.flat_slope_pct_per_bar / 100.0 * c_ref
        sub = _classify_subtype(s_u, s_l, flat_th_abs)
        n_h = len(geom["h_idxs"])
        n_l = len(geom["l_idxs"])
        title = (
            f"{sub} · {n_h}H/{n_l}L · "
            f"width {geom['channel_width_pct']*100:.1f}% · "
            f"contract {geom.get('width_contraction_ratio', 0)*100:.0f}%"
        )
        _add_panel(fig, col, df, geom, asof_idx, eff_tol, title)

    fig.update_layout(
        height=520,
        title=dict(
            text=(
                f"<b>{args.ticker}</b> · {args.tf} · "
                f"tolerance comparison "
                f"<span style='color:#9aa0a6;font-size:11px;'>"
                f"(shaded band = ±tol around line · "
                f"H ▼ / L ▲ markers = bars whose close lies inside tol)"
                f"</span>"
            ),
            x=0, font=dict(size=14),
        ),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a", plot_bgcolor="#111214",
        font=dict(family="Inter, sans-serif", size=11, color="#c7bdbe"),
        margin=dict(l=40, r=20, t=100, b=40),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].rangeslider = dict(visible=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(199,189,190,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(199,189,190,0.06)", zeroline=False)

    out_path = Path(args.out) if args.out else OUT_DIR / f"triangle_break_{args.ticker}_{args.tf}_tolcmp.html"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"[tolcmp] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
