"""triangle_break — interactive candle plots of detected triangles.

Reads `output/triangle_break_tr_<tf>.parquet` and re-runs `fit_geometry`
to recover raw line slopes, then renders a single self-contained HTML
with N candlestick panels:
  - candles
  - upper/lower OLS trendlines extending to apex
  - pivot markers (H ▼ red, L ▲ green)
  - asof vertical guide
  - breakout marker (if signal_state ∈ {trigger, extended})
  - apex point

Usage:
    python tools/triangle_break_plot.py                    # top-6 of 1d
    python tools/triangle_break_plot.py --tf 1w --top 8
    python tools/triangle_break_plot.py --ticker KRDMD --tf 1M
    python tools/triangle_break_plot.py --tickers DURKN EDIP EUPWR --tf 1d

Output: output/triangle_break_<tf>_inspect.html  (auto-opened with --open)
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from channel_break.detect import _line_at, fit_geometry
from data import intraday_1h
from mb_scanner.resample import per_ticker_panel, to_5h, to_daily, to_monthly, to_weekly
from triangle_break.html_report import _rank_triangles
from triangle_break.schema import FAMILIES

OUT_DIR = Path("output")

TF_TO_FAM = {"5h": "tr_5h", "1d": "tr_1d", "1w": "tr_1w", "1M": "tr_1M"}
TF_RESAMPLER = {
    "5h": (to_5h, "ts_istanbul"),
    "1d": (to_daily, "date"),
    "1w": (to_weekly, "week_end"),
    "1M": (to_monthly, "month_end"),
}


SUBTYPE_BADGE = {
    "ascending":  "▲ASC",
    "symmetric":  "◆SYM",
    "descending": "▼DESC",
}
STATE_BADGE = {
    "trigger":      "TRG",
    "pre_breakout": "PRE",
    "extended":     "EXT",
}


def _resolve_asof_idx(df: pd.DataFrame, asof_ts: pd.Timestamp) -> int:
    idx = pd.DatetimeIndex(df.index)
    asof_ts = pd.Timestamp(asof_ts)
    if idx.tz is not None and asof_ts.tz is None:
        asof_ts = asof_ts.tz_localize(idx.tz)
    elif idx.tz is None and asof_ts.tz is not None:
        asof_ts = asof_ts.tz_convert("Europe/Istanbul").tz_localize(None)
    mask = idx <= asof_ts
    if not mask.any():
        return -1
    return int(np.flatnonzero(mask)[-1])


def _select_rows(
    tf: str,
    *,
    tickers: list[str] | None,
    top: int,
) -> pd.DataFrame:
    fam = TF_TO_FAM[tf]
    p = OUT_DIR / f"triangle_break_{fam}.parquet"
    if not p.exists():
        raise SystemExit(
            f"missing {p}; run `tools/triangle_break_scan_live.py` first."
        )
    df = pd.read_parquet(p)
    if df.empty:
        raise SystemExit(f"{p} is empty.")
    if tickers:
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            raise SystemExit(f"none of {tickers} in {p}")
        return df.reset_index(drop=True)
    df = _rank_triangles(df).head(top).reset_index(drop=True)
    return df


def _panels_for_tickers(
    tickers: list[str], tf: str,
) -> dict[str, pd.DataFrame]:
    bars = intraday_1h.load_intraday(
        tickers=tickers, start=None, end=None, min_coverage=0.0,
    )
    if bars.empty:
        raise SystemExit(f"intraday_1h returned no bars for {tickers}")
    resampler, idx_col = TF_RESAMPLER[tf]
    return per_ticker_panel(resampler(bars), idx_col)


def _plot_one(
    fig: go.Figure,
    row_idx: int,
    col_idx: int,
    *,
    df: pd.DataFrame,
    geom: dict,
    meta: pd.Series,
    asof_idx: int,
    breakout_idx: int | None,
    apex_idx_ref: int,
) -> None:
    """Add one triangle panel into a subplot cell."""
    first = int(geom["first_pivot_idx"])
    bars_to_apex = int(meta.get("bars_to_apex") or 0)
    n = len(df)

    # Window: 3 bars before first_pivot → asof + min(15, bars_to_apex + 3)
    pad_left = 3
    pad_right = max(5, min(20, bars_to_apex + 3))
    win_lo = max(0, first - pad_left)
    win_hi = min(n - 1, asof_idx + pad_right)

    sub = df.iloc[win_lo:win_hi + 1]
    x = pd.to_datetime(sub.index)

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"],
            increasing_line_color="#7a9e7a",
            decreasing_line_color="#a86a6a",
            increasing_fillcolor="#7a9e7a",
            decreasing_fillcolor="#a86a6a",
            line=dict(width=1),
            showlegend=False,
            name="",
        ),
        row=row_idx, col=col_idx,
    )

    # Trendlines from first_pivot to apex (or window end if apex past window)
    line_lo = first
    line_hi = min(n - 1, max(asof_idx, apex_idx_ref))
    line_idxs = list(range(line_lo, line_hi + 1))
    line_x = pd.to_datetime(df.index[line_lo:line_hi + 1])
    upper_y = [_line_at(geom["s_u"], geom["b_u"], i) for i in line_idxs]
    lower_y = [_line_at(geom["s_l"], geom["b_l"], i) for i in line_idxs]

    fig.add_trace(
        go.Scatter(
            x=line_x, y=upper_y, mode="lines",
            line=dict(color="#c9a96e", width=1.5, dash="dash"),
            showlegend=False, name="upper",
            hovertemplate="upper · %{y:.2f}<extra></extra>",
        ),
        row=row_idx, col=col_idx,
    )
    fig.add_trace(
        go.Scatter(
            x=line_x, y=lower_y, mode="lines",
            line=dict(color="#7a8fa5", width=1.5, dash="dash"),
            showlegend=False, name="lower",
            hovertemplate="lower · %{y:.2f}<extra></extra>",
        ),
        row=row_idx, col=col_idx,
    )

    # Pivot markers
    h_idxs = [i for i in geom["h_idxs"] if win_lo <= i <= win_hi]
    l_idxs = [i for i in geom["l_idxs"] if win_lo <= i <= win_hi]
    if h_idxs:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(df.index[h_idxs]),
                y=[float(df["close"].iat[i]) for i in h_idxs],
                mode="markers",
                marker=dict(symbol="triangle-down", size=9,
                            color="#c9a96e", line=dict(width=1, color="#0a0a0a")),
                showlegend=False, name="H pivot",
                hovertemplate="H · %{y:.2f}<extra></extra>",
            ),
            row=row_idx, col=col_idx,
        )
    if l_idxs:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(df.index[l_idxs]),
                y=[float(df["close"].iat[i]) for i in l_idxs],
                mode="markers",
                marker=dict(symbol="triangle-up", size=9,
                            color="#7a8fa5", line=dict(width=1, color="#0a0a0a")),
                showlegend=False, name="L pivot",
                hovertemplate="L · %{y:.2f}<extra></extra>",
            ),
            row=row_idx, col=col_idx,
        )

    # Apex marker (if within or near window)
    apex_y_upper = _line_at(geom["s_u"], geom["b_u"], apex_idx_ref)
    if 0 <= apex_idx_ref < n and win_lo <= apex_idx_ref <= win_hi + 5:
        fig.add_trace(
            go.Scatter(
                x=[pd.to_datetime(df.index[min(apex_idx_ref, n-1)])],
                y=[apex_y_upper],
                mode="markers",
                marker=dict(symbol="x", size=11, color="#a86a6a",
                            line=dict(width=2)),
                showlegend=False, name="apex",
                hovertemplate=f"apex · {bars_to_apex} bar<extra></extra>",
            ),
            row=row_idx, col=col_idx,
        )

    # Asof vertical line (use vline via shape — must reference subplot axes)
    asof_x = pd.to_datetime(df.index[asof_idx])
    fig.add_vline(
        x=asof_x, line=dict(color="#c7bdbe", width=1, dash="dot"),
        row=row_idx, col=col_idx,
    )

    # Breakout marker
    if breakout_idx is not None and 0 <= int(breakout_idx) < n:
        bi = int(breakout_idx)
        if win_lo <= bi <= win_hi:
            fig.add_trace(
                go.Scatter(
                    x=[pd.to_datetime(df.index[bi])],
                    y=[float(df["close"].iat[bi])],
                    mode="markers",
                    marker=dict(symbol="star", size=14,
                                color="#7a9e7a",
                                line=dict(width=1, color="#0a0a0a")),
                    showlegend=False, name="breakout",
                    hovertemplate="breakout · %{y:.2f}<extra></extra>",
                ),
                row=row_idx, col=col_idx,
            )


def _build_figure(
    rows: pd.DataFrame, tf: str, asof_label: str,
) -> go.Figure:
    fam = TF_TO_FAM[tf]
    params = FAMILIES[fam]
    tickers = rows["ticker"].tolist()
    panels = _panels_for_tickers(tickers, tf)

    n = len(rows)
    cols = 2 if n > 1 else 1
    nrows = (n + cols - 1) // cols

    titles = []
    for i in range(n):
        r = rows.iloc[i]
        sub = SUBTYPE_BADGE.get(str(r["triangle_subtype"]), "?")
        st = STATE_BADGE.get(str(r["signal_state"]), "?")
        tier = "A" if bool(r.get("tier_a")) else "B"
        contract = float(r.get("width_contraction_ratio") or 0.0)
        b2a = int(r.get("bars_to_apex") or 0)
        titles.append(
            f"<b>{r['ticker']}</b> · {sub} · {st} · T{tier} · "
            f"contract {contract*100:.0f}% · →apex {b2a} bar"
        )

    fig = make_subplots(
        rows=nrows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for i in range(n):
        row = rows.iloc[i]
        ticker = str(row["ticker"])
        df = panels.get(ticker)
        if df is None or df.empty:
            continue
        asof_idx = _resolve_asof_idx(df, pd.Timestamp(row["as_of_ts"]))
        if asof_idx < 0:
            continue
        geom = fit_geometry(df, asof_idx, params)
        if geom is None:
            continue
        breakout_idx = row.get("breakout_idx")
        try:
            bi = int(breakout_idx) if breakout_idx is not None and pd.notna(breakout_idx) else None
        except (TypeError, ValueError):
            bi = None
        apex_idx_ref = int(row.get("apex_idx") or asof_idx)

        ridx = i // cols + 1
        cidx = i % cols + 1
        _plot_one(
            fig, ridx, cidx,
            df=df, geom=geom, meta=row,
            asof_idx=asof_idx,
            breakout_idx=bi,
            apex_idx_ref=apex_idx_ref,
        )

    fig.update_layout(
        height=380 * nrows,
        title=dict(
            text=(
                f"<b>NOX Triangle Break</b> — {tf} · asof {asof_label} · "
                f"{n} panel<br>"
                f"<span style='font-size:11px;color:#9aa0a6;'>"
                f"Üst (gold) ve alt (blue) trendline OLS · "
                f"H ▼ pivot · L ▲ pivot · ✕ apex · ★ breakout · "
                f"dotted vertical = asof</span>"
            ),
            x=0, font=dict(size=15),
        ),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#111214",
        font=dict(family="Inter, sans-serif", size=11, color="#c7bdbe"),
        margin=dict(l=40, r=20, t=120, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True, gridcolor="rgba(199,189,190,0.06)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(199,189,190,0.06)",
        zeroline=False,
    )
    # disable rangeslider on every candlestick subplot
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].rangeslider = dict(visible=False)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=list(TF_TO_FAM), default="1d")
    ap.add_argument("--top", type=int, default=6,
                    help="Top-N rows by rank (default 6).")
    ap.add_argument("--ticker", default=None,
                    help="Single ticker (overrides --top).")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Explicit ticker list (overrides --top).")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default output/triangle_break_<tf>_inspect.html).")
    ap.add_argument("--open", action="store_true",
                    help="Open output in default browser.")
    args = ap.parse_args()

    tickers = None
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = list(args.tickers)

    rows = _select_rows(args.tf, tickers=tickers, top=args.top)
    asof_label = pd.Timestamp(rows["as_of_ts"].max()).strftime("%Y-%m-%d")
    print(f"[plot] tf={args.tf}  n={len(rows)}  asof={asof_label}  "
          f"tickers={rows['ticker'].tolist()}")

    fig = _build_figure(rows, args.tf, asof_label)

    out_path = Path(args.out) if args.out else OUT_DIR / f"triangle_break_{args.tf}_inspect.html"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"[plot] wrote {out_path}  ({out_path.stat().st_size:,} bytes)")
    if args.open:
        webbrowser.open(f"file://{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
