"""HyperWave full-cross scanner across {5h, 1d, 1w} timeframes.

Detects EVERY HW × signal crossing per LuxAlgo HyperWave semantics.
Emits 4 kinds (small + big dots), so downstream can combine with other
scanners for entry/exit composition:

  AL_OS  = HW crosses ABOVE signal AND HW < 20   (big dot, oversold turning)
  AL     = HW crosses ABOVE signal AND HW ≥ 20   (small dot, mid-zone bullish cross)
  SAT_OB = HW crosses BELOW signal AND HW > 80   (big dot, overbought turning)
  SAT    = HW crosses BELOW signal AND HW ≤ 80   (small dot, mid-zone bearish cross)

Sibling of tools/hw_obos_scan.py (which emits big-dots only). Kept separate
so the LIVE OBOS pipeline schema stays stable for its Telegram/HTML parser.

Bar construction (mb_scanner.resample, 1h master → TF):
  5h : TV-aligned, [09:00-13:00] morning + [14:00-18:00] afternoon
       (n_bars==5 clean sessions only)
  1d : standard date groupby
  1w : daily resampled W-FRI, label=Friday

HW: SMA(RSI(close, 7), 3); signal: SMA(HW, 3). Source = close.

Output is descriptive (event log per TF). NO trading-edge claim.
Pre-registered backtest hwo_mtf_v1 CLOSED_REJECTED — see SPEC §12.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mb_scanner.resample import to_5h, to_daily, to_weekly
from oscmatrix.components.hyperwave import compute_hyperwave

DEFAULT_MASTER = "output/extfeed_intraday_1h_3y_master.parquet"

TF_CONFIGS = {
    "5h": {"resampler": to_5h,     "ts_col": "ts_istanbul", "min_bars": 30,
           "require_n_bars": 5,    "lookback_days": 30},
    "1d": {"resampler": to_daily,  "ts_col": "date",        "min_bars": 60,
           "require_n_bars": None, "lookback_days": 60},
    "1w": {"resampler": to_weekly, "ts_col": "week_end",    "min_bars": 26,
           "require_n_bars": None, "lookback_days": 365 // 2},
}


def _bar_close_ts(tf: str, ts, tz: str = "Europe/Istanbul") -> pd.Timestamp:
    """Wall-clock close time of a bar labeled ts on TF."""
    t = pd.Timestamp(ts)
    if tf == "5h":
        if t.tz is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)
        return t + pd.Timedelta(hours=4)
    t_naive = t.tz_localize(None) if t.tz is not None else t
    return (t_naive.normalize() + pd.Timedelta(hours=18)).tz_localize(tz)


def _classify(row: pd.Series) -> tuple[str, str]:
    """Return (kind, dot_size) for a bar that has a cross flagged."""
    if row["hwo_up"]:
        if row["os_hwo_up"]:
            return "AL_OS", "big"
        return "AL", "small"
    if row["hwo_down"]:
        if row["ob_hwo_down"]:
            return "SAT_OB", "big"
        return "SAT", "small"
    return "", ""


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
    # Capture HW/signal previous values for downstream slope/transition use.
    joined["hyperwave_prev"] = joined["hyperwave"].shift(1)
    joined["signal_prev"] = joined["signal"].shift(1)

    cross_mask = joined["hwo_up"] | joined["hwo_down"]
    events = joined[cross_mask].copy()
    if events.empty:
        return events
    classified = events.apply(_classify, axis=1, result_type="expand")
    classified.columns = ["kind", "dot_size"]
    events = pd.concat([events, classified], axis=1)
    events = events.reset_index().rename(columns={ts_col: "ts"})
    events["ticker"] = bars["ticker"].iloc[0]
    events["tf"] = tf
    return events[[
        "ticker", "tf", "ts", "kind", "dot_size",
        "hyperwave", "signal", "hyperwave_prev", "signal_prev",
        "close", "volume",
    ]]


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
        events = pd.DataFrame(columns=[
            "ticker", "tf", "ts", "kind", "dot_size",
            "hyperwave", "signal", "hyperwave_prev", "signal_prev",
            "close", "volume",
        ])
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

    counts = events_window["kind"].value_counts().to_dict() if not events_window.empty else {}
    summary = " · ".join(f"{k}={counts.get(k, 0)}" for k in ("AL_OS", "AL", "SAT_OB", "SAT"))
    print(f"[{tf}] window {lb_days}d → {len(events_window)} events ({summary})")

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
        out_csv = str(out_dir_p / f"hwo_full_{tf}_scan_{asof_str}.csv")
        results[tf] = run_tf(mp, tf, asof_ts, lookback_days, out_csv)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default = max ts in master")
    ap.add_argument("--tfs", nargs="+", default=["5h", "1d", "1w"],
                    help="Subset of {5h, 1d, 1w}; default = all three")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="Override per-TF default lookback window in calendar days")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    run(args.master, args.tfs, args.asof, args.lookback_days, args.out_dir)


if __name__ == "__main__":
    main()
