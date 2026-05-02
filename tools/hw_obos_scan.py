"""HyperWave OB/OS big-dot scanner across {5h, 1d, 1w, 1mo} timeframes.

Detects big-dot turning points per LuxAlgo HyperWave semantics:
  AL_OS  = HW crosses ABOVE signal AND HW < 20  (oversold turning point)
  SAT_OB = HW crosses BELOW signal AND HW > 80  (overbought turning point)

Bar construction (mb_scanner.resample, 1h master → TF):
  5h  : TV-aligned, [09:00-13:00] morning + [14:00-18:00] afternoon
        (n_bars==5 clean sessions only)
  1d  : standard date groupby
  1w  : daily resampled W-FRI, label=Friday
  1mo : daily resampled BME, label=last business day

HW: SMA(RSI(close, 7), 3); signal: SMA(HW, 3). Source = close.

Output is descriptive (event log per TF). NO trading-edge claim.
Pre-registered backtest hwo_mtf_v1 CLOSED_REJECTED — see SPEC §12.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mb_scanner.resample import to_5h, to_daily, to_weekly, to_monthly
from oscmatrix.components.hyperwave import compute_hyperwave

DEFAULT_MASTER = "output/extfeed_intraday_1h_3y_master.parquet"

# Per-TF config: resampler fn, timestamp column on resampled DF,
# minimum bars before HW cross-detection is reliable, and default
# lookback (in TF bars) for the event log window.
TF_CONFIGS = {
    "5h":  {"resampler": to_5h,      "ts_col": "ts_istanbul", "min_bars": 30, "lookback_bars": 60,
            "require_n_bars": 5,  # TV-aligned clean session filter
            "lookback_days": 30},
    "1d":  {"resampler": to_daily,   "ts_col": "date",        "min_bars": 60, "lookback_bars": 60,
            "require_n_bars": None,
            "lookback_days": 60},
    "1w":  {"resampler": to_weekly,  "ts_col": "week_end",    "min_bars": 26, "lookback_bars": 26,
            "require_n_bars": None,
            "lookback_days": 365 // 2},
    "1mo": {"resampler": to_monthly, "ts_col": "month_end",   "min_bars": 12, "lookback_bars": 12,
            "require_n_bars": None,
            "lookback_days": 365 + 90},
}


def _bar_close_ts(tf: str, ts, tz: str = "Europe/Istanbul") -> pd.Timestamp:
    """Return the wall-clock close time of a bar labeled ts on TF.

    For 5h: 14:00 afternoon bar's last hourly bar is 18:00 (which carries the
    18:08-18:10 closing prints). Once the master has hourly bar 18:00, the 5h
    bar is effectively complete — so close = label + 4h (not +5h).
    For daily/weekly/monthly: BIST closes 18:00, so close = label-date 18:00.
    """
    t = pd.Timestamp(ts)
    if tf == "5h":
        if t.tz is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)
        return t + pd.Timedelta(hours=4)
    # daily / weekly / monthly: label is a date; close is 18:00 TR same date
    t_naive = t.tz_localize(None) if t.tz is not None else t
    return (t_naive.normalize() + pd.Timedelta(hours=18)).tz_localize(tz)


def scan_ticker(bars_1h: pd.DataFrame, tf: str, asof_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    cfg = TF_CONFIGS[tf]
    bars = cfg["resampler"](bars_1h)
    if bars.empty:
        return bars.iloc[0:0]
    if cfg["require_n_bars"] is not None and "n_bars" in bars.columns:
        bars = bars[bars["n_bars"] == cfg["require_n_bars"]].copy()
    # Drop trailing unclosed bar(s) so HW crosses don't repaint.
    if asof_ts is not None and not bars.empty:
        ts_col = cfg["ts_col"]
        close_times = bars[ts_col].apply(lambda t: _bar_close_ts(tf, t))
        if asof_ts.tz is None:
            asof_local = asof_ts.tz_localize("Europe/Istanbul")
        else:
            asof_local = asof_ts.tz_convert("Europe/Istanbul")
        bars = bars[close_times <= asof_local].copy()
    if len(bars) < cfg["min_bars"]:
        return bars.iloc[0:0]
    ts_col = cfg["ts_col"]
    panel = bars.set_index(ts_col)[["open", "high", "low", "close", "volume"]]
    hw = compute_hyperwave(panel, length=7, sig_len=3)
    joined = panel.join(hw)
    al = joined[joined["os_hwo_up"]].assign(kind="AL_OS")
    sa = joined[joined["ob_hwo_down"]].assign(kind="SAT_OB")
    events = pd.concat([al, sa]).sort_index()
    if events.empty:
        return events
    events = events.reset_index().rename(columns={ts_col: "ts"})
    events["ticker"] = bars["ticker"].iloc[0]
    events["tf"] = tf
    return events[["ticker", "tf", "ts", "kind", "hyperwave", "signal", "close", "volume"]]


def run_tf(
    master_df: pd.DataFrame,
    tf: str,
    asof_ts: pd.Timestamp,
    lookback_days: int | None,
    out_csv: str,
) -> pd.DataFrame:
    cfg = TF_CONFIGS[tf]
    parts: list[pd.DataFrame] = []
    tickers = sorted(master_df["ticker"].unique())
    for i, tk in enumerate(tickers, 1):
        sub = master_df[master_df["ticker"] == tk].sort_values("ts_istanbul")
        ev = scan_ticker(sub, tf, asof_ts=asof_ts)
        if not ev.empty:
            parts.append(ev)
        if i % 200 == 0:
            print(f"[{tf}] {i}/{len(tickers)} processed; events so far={sum(len(p) for p in parts)}")

    if not parts:
        print(f"[{tf}] no events")
        events = pd.DataFrame(columns=["ticker", "tf", "ts", "kind", "hyperwave", "signal", "close", "volume"])
    else:
        events = pd.concat(parts, ignore_index=True)
        events = events.sort_values(["ts", "ticker"]).reset_index(drop=True)

    lb_days = lookback_days if lookback_days is not None else cfg["lookback_days"]
    events_window = events
    if not events.empty:
        ts_series = pd.to_datetime(events["ts"])
        if hasattr(ts_series.dtype, "tz") and ts_series.dt.tz is not None:
            cutoff = asof_ts.tz_convert(ts_series.dt.tz) - pd.Timedelta(days=lb_days)
        else:
            cutoff = asof_ts.tz_localize(None) - pd.Timedelta(days=lb_days)
        events_window = events[ts_series >= cutoff].reset_index(drop=True)

    n_al = int((events_window["kind"] == "AL_OS").sum()) if not events_window.empty else 0
    n_sat = int((events_window["kind"] == "SAT_OB").sum()) if not events_window.empty else 0
    print(f"[{tf}] window {lb_days}d → {len(events_window)} events (AL_OS={n_al}, SAT_OB={n_sat})")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    events_window.to_csv(out_csv, index=False)
    print(f"[{tf}] wrote {out_csv}")
    return events_window


def run(
    master_path: str,
    tfs: list[str],
    asof: str | None,
    lookback_days: int | None,
    out_dir: str,
) -> dict[str, pd.DataFrame]:
    mp = pd.read_parquet(master_path)
    print(f"[load] master shape={mp.shape} tickers={mp['ticker'].nunique()}")

    if asof is None:
        asof_ts = pd.to_datetime(mp["ts_istanbul"]).max()
    else:
        asof_ts = pd.Timestamp(asof, tz="Europe/Istanbul")
    print(f"[asof] {asof_ts}")

    asof_str = asof_ts.strftime("%Y-%m-%d")
    out_dir_p = Path(out_dir)

    results: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        if tf not in TF_CONFIGS:
            raise ValueError(f"unknown tf={tf}; expected one of {list(TF_CONFIGS)}")
        out_csv = str(out_dir_p / f"hw_obos_{tf}_scan_{asof_str}.csv")
        results[tf] = run_tf(mp, tf, asof_ts, lookback_days, out_csv)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default = max ts in master")
    ap.add_argument("--tfs", nargs="+", default=["5h", "1d", "1w", "1mo"],
                    help="Subset of {5h, 1d, 1w, 1mo}; default = all four")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="Override per-TF default lookback window in calendar days")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    run(args.master, args.tfs, args.asof, args.lookback_days, args.out_dir)


if __name__ == "__main__":
    main()
