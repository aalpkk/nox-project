"""Faz I — unified events parquet across all on-disk signal sources.

Stacks the following sources into a single (long-format) events table with a
common schema, then attaches forward MFE / realized returns at horizons
{5, 10, 20, 30, 60} trading days plus a daily Wilder ATR(14) reference.

Sources (all read-only, currently on disk):
    mb_scanner    : mb_scanner_events_{mb,bb}_{5h,1d,1w,1M}.parquet
    horizontal_base : horizontal_base_event_v1.parquet
    nox_rt_daily  : decision_engine_lab_regime_transition_events_3y.parquet
    nox_weekly    : decision_engine_lab_weekly_regime_transition_events_3y.parquet
    nyxmomentum   : decision_engine_lab_momentum_picks_3y.parquet
    paper_line_e  : paper_execution_v0_trades.parquet           (one row per daily replay tick)
    paper_line_tr : paper_execution_v0_trigger_retest_trades.parquet
    nyxalpha      : decision_engine_lab_alpha_events_3y.parquet
    smart_breakout: decision_engine_lab_smart_breakout_events_3y.parquet

Forward returns are computed once from the extfeed master (1h → daily resample)
using the same algorithm as `decision_engine_lab_universal_returns_3y.py`.

ATR(14) is the canonical daily Wilder ATR computed once per (ticker, date).

Output:
    output/decision_engine_lab_unified_events_3y.parquet
    output/decision_engine_lab_unified_events_3y.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"

DST_PARQUET = OUT / "decision_engine_lab_unified_events_3y.parquet"
DST_MD = OUT / "decision_engine_lab_unified_events_3y.md"

WINDOWS = [5, 10, 20, 30, 60]
HIT_THRESHOLDS = [0.05, 0.10, 0.20, 0.50, 1.00]
PCTS = [10, 25, 50, 75, 90, 95, 99]

UNIFIED_COLS = [
    "ticker",
    "signal_date",
    "source",
    "family",
    "setup_label",
    "signal_state",
    "side",
    "entry_ref",
    "atr",
    "atr_14_daily",
] + [f"mfe_pct_{w}d" for w in WINDOWS] + [f"realized_pct_{w}d" for w in WINDOWS] + [
    "src_native_payload",
]


# ---------------------------------------------------------------------------
# Daily bar reference (forward returns + ATR)
# ---------------------------------------------------------------------------

def _load_daily_bars() -> pd.DataFrame:
    print("[unified_events] reading extfeed → daily OHLC...")
    df = pq.read_table(
        EXTFEED, columns=["ticker", "ts_utc", "high", "low", "close"]
    ).to_pandas()
    df["date"] = pd.to_datetime(df["ts_utc"]).dt.date
    daily = (
        df.groupby(["ticker", "date"])
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  {len(daily):,} daily bars across {daily.ticker.nunique()} tickers")
    return daily


def _compute_forward_and_atr(daily: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date), compute forward MFE / realized and Wilder ATR(14)."""
    parts = []
    n_tk = daily.ticker.nunique()
    print(f"[unified_events] computing forward + atr per ticker (n={n_tk})...")
    max_w = max(WINDOWS)
    for k, (tkr, grp) in enumerate(daily.groupby("ticker", sort=False), 1):
        g = grp.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        # ATR (Wilder, period 14)
        prev_close = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = np.full(n, np.nan)
        period = 14
        if n >= period:
            atr[period - 1] = np.nanmean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        out = pd.DataFrame({"ticker": tkr, "date": g["date"]})
        out["atr_14_daily"] = atr
        for w in WINDOWS:
            mfe = np.full(n, np.nan)
            real = np.full(n, np.nan)
            limit = n - w
            for i in range(limit):
                lo = i + 1
                hi = i + w + 1
                mfe[i] = high[lo:hi].max() / close[i] - 1.0
                real[i] = close[i + w] / close[i] - 1.0
            out[f"mfe_pct_{w}d"] = mfe
            out[f"realized_pct_{w}d"] = real
        parts.append(out)
        if k % 100 == 0:
            print(f"  {k}/{n_tk} tickers processed")
    ref = pd.concat(parts, ignore_index=True)
    print(f"[unified_events] reference rows: {len(ref):,}")
    return ref


# ---------------------------------------------------------------------------
# Source loaders → unified rows
# ---------------------------------------------------------------------------

def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _new_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in UNIFIED_COLS})


def _payload(row: pd.Series, keys: list[str]) -> str:
    obj = {}
    for k in keys:
        v = row.get(k)
        if pd.isna(v):
            continue
        if hasattr(v, "isoformat"):
            obj[k] = v.isoformat()
        elif isinstance(v, (np.floating, np.integer)):
            obj[k] = v.item()
        else:
            obj[k] = v
    return json.dumps(obj, default=str, sort_keys=True)


def _load_mb(path: Path, family: str) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    if df.empty:
        return _new_frame()
    payload_keys = [
        "event_type", "mode", "event_idx", "zone_high", "zone_low",
        "zone_width_pct", "zone_width_atr", "zone_age_bars", "event_volume",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["event_bar_date"]),
        "source": "mb_scanner",
        "family": family,
        "setup_label": df["mode"].astype(str) + "__" + df["event_type"].astype(str),
        "signal_state": pd.NA,
        "side": "long",
        "entry_ref": df["event_close"].astype(float),
        "atr": np.nan,
        "src_native_payload": [
            _payload(r, payload_keys) for _, r in df.iterrows()
        ],
    })
    return out


def _load_hb(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    payload_keys = [
        "signal_type", "breakout_bar_date", "family__trigger_level",
        "invalidation_level", "initial_risk_pct",
        "common__breakout_pct", "common__breakout_atr",
        "common__volume_ratio_20", "common__atr_pct",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["bar_date"]),
        "source": "horizontal_base",
        "family": "hb",
        "setup_label": "hb__" + df["signal_state"].astype(str),
        "signal_state": df["signal_state"].astype(str),
        "side": "long",
        "entry_ref": df["entry_reference_price"].astype(float),
        "atr": df.get("common__atr_14", np.nan).astype(float)
              if "common__atr_14" in df.columns else np.nan,
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_regime(path: Path, source_name: str, family_default: str) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    payload_keys = [
        "regime", "regime_name", "prev_regime", "prev_regime_name",
        "trend_score", "participation_score", "expansion_score",
        "entry_score", "oe_score", "vol_low", "early_entry", "growth_room",
        "no_pump", "atr_pct", "adx", "adx_slope", "cmf", "rvol",
        "di_spread", "rsi", "execution_label", "timeframe",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["signal_date"]),
        "source": source_name,
        "family": df.get("family", family_default).astype(str),
        "setup_label": df.get("setup_label", "").astype(str),
        "signal_state": pd.NA,
        "side": df.get("side", "long").astype(str),
        "entry_ref": df["entry_ref"].astype(float),
        "atr": df.get("atr", np.nan).astype(float),
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_nyx(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    payload_keys = ["rank", "momentum_score", "momentum_weight",
                    "execution_label", "timeframe"]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["signal_date"]),
        "source": "nyxmomentum",
        "family": df["family"].astype(str),
        "setup_label": df.get("setup_label", "").astype(str),
        "signal_state": pd.NA,
        "side": df.get("side", "long").astype(str),
        "entry_ref": df["entry_ref"].astype(float),
        "atr": df.get("atr", np.nan).astype(float),
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_paper_line_e(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    payload_keys = [
        "asof_date", "lane_source", "preview_seen_17", "close_confirmed",
        "ema_tier", "earliness_score_pct", "invalidation_level",
        "risk_per_share", "entry_date", "skip_reason",
        "paper_valid_from", "paper_valid_until",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["signal_asof_date"]),
        "source": "paper_execution_v0",
        "family": "line_e",
        "setup_label": "line_e__" + df["signal_state"].astype(str),
        "signal_state": df["signal_state"].astype(str),
        "side": "long",
        "entry_ref": df["entry_reference_price"].astype(float),
        "atr": np.nan,
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_alpha(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    if df.empty:
        return _new_frame()
    payload_keys = [
        "momentum_type", "composite_score", "adx", "cmf", "rsi",
        "candle_pattern", "price_slope", "signal_slope",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["signal_date"]),
        "source": "nyxalpha",
        "family": "alpha",
        "setup_label": df["setup_label"].astype(str),
        "signal_state": pd.NA,
        "side": df.get("side", "long").astype(str),
        "entry_ref": df["entry_ref"].astype(float),
        "atr": df.get("atr", np.nan).astype(float),
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_smart_breakout(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    if df.empty:
        return _new_frame()
    payload_keys = [
        "box_top", "box_bot", "stop_ref", "risk_pct",
        "squeeze_bars", "bars_after_squeeze", "strength",
        "vol_ratio", "body_atr", "above_ema50",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["signal_date"]),
        "source": "smart_breakout",
        "family": "smart_breakout",
        "setup_label": df["setup_label"].astype(str),
        "signal_state": pd.NA,
        "side": df.get("side", "long").astype(str),
        "entry_ref": df["entry_ref"].astype(float),
        "atr": df.get("atr", np.nan).astype(float),
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


def _load_triangle_break(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    if df.empty:
        return _new_frame()
    payload_cols = [c for c in df.columns if c.startswith("_pl_")]
    payload_keys = [c[len("_pl_"):] for c in payload_cols]
    payload_view = df[payload_cols].rename(
        columns={c: c[len("_pl_"):] for c in payload_cols}
    )
    out = pd.DataFrame({
        "ticker": df["ticker"].astype(str),
        "signal_date": _coerce_date(df["signal_date"]),
        "source": "triangle_break",
        "family": df["family"].astype(str),
        "setup_label": df["setup_label"].astype(str),
        "signal_state": pd.NA,
        "side": df.get("side", "long").astype(str),
        "entry_ref": df["entry_ref"].astype(float),
        "atr": df["atr"].astype(float),
        "src_native_payload": [_payload(r, payload_keys)
                               for _, r in payload_view.iterrows()],
    })
    return out


def _load_paper_line_tr(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _new_frame()
    df = pq.read_table(path).to_pandas()
    payload_keys = [
        "lane_source", "preview_seen_17", "close_confirmed", "ema_tr_tier",
        "earliness_score_pct", "invalidation_level", "risk_per_share",
        "entry_date", "skip_reason",
        "paper_valid_from", "paper_valid_until",
    ]
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "signal_date": _coerce_date(df["asof_date"]),
        "source": "paper_execution_v0",
        "family": "line_tr",
        "setup_label": "line_tr__" + df["signal_state"].astype(str),
        "signal_state": df["signal_state"].astype(str),
        "side": "long",
        "entry_ref": df["entry_reference_price"].astype(float),
        "atr": np.nan,
        "src_native_payload": [_payload(r, payload_keys) for _, r in df.iterrows()],
    })
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _per_source_summary(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, fam in (
        unified[["source", "family"]].drop_duplicates().sort_values(["source", "family"]).itertuples(index=False)
    ):
        sub = unified[(unified["source"] == src) & (unified["family"] == fam)]
        row = {"source": src, "family": fam, "n": len(sub),
               "n_tickers": sub["ticker"].nunique(),
               "date_min": str(sub["signal_date"].min().date())
                           if len(sub) and sub["signal_date"].notna().any() else "",
               "date_max": str(sub["signal_date"].max().date())
                           if len(sub) and sub["signal_date"].notna().any() else "",
               "atr_pct_filled": float(sub["atr_14_daily"].notna().mean()) if len(sub) else 0.0,
               "mfe30_pct_filled": float(sub["mfe_pct_30d"].notna().mean()) if len(sub) else 0.0,
               }
        rows.append(row)
    return pd.DataFrame(rows)


def _mfe_dist_block(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        col = f"mfe_pct_{w}d"
        s = unified[col].astype(float).dropna()
        if len(s) == 0:
            continue
        row = {"horizon_d": w, "n": len(s),
               "mean": float(s.mean()), "std": float(s.std())}
        for q in PCTS:
            row[f"p{q}"] = float(s.quantile(q / 100.0))
        for thr in HIT_THRESHOLDS:
            row[f"≥{int(thr * 100)}%"] = float((s >= thr).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _per_source_mfe30(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, fam), sub in unified.groupby(["source", "family"], dropna=False):
        s = sub["mfe_pct_30d"].astype(float).dropna()
        if len(s) < 50:
            continue
        row = {"source": src, "family": fam, "n": len(s),
               "median": float(s.median()), "mean": float(s.mean())}
        for q in [25, 75, 90, 95, 99]:
            row[f"p{q}"] = float(s.quantile(q / 100.0))
        for thr in HIT_THRESHOLDS:
            row[f"≥{int(thr * 100)}%"] = float((s >= thr).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "family"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    daily = _load_daily_bars()
    ref = _compute_forward_and_atr(daily)
    ref = ref.rename(columns={"date": "signal_date"})

    parts: list[pd.DataFrame] = []

    print("[unified_events] loading sources...")
    for tf in ("5h", "1d", "1w", "1M"):
        for mode in ("mb", "bb"):
            family = f"{mode}_{tf}"
            p = OUT / f"mb_scanner_events_{family}.parquet"
            df = _load_mb(p, family)
            print(f"  mb_scanner/{family}: {len(df):,} rows")
            parts.append(df)

    parts.append(_print_load(_load_hb(OUT / "horizontal_base_event_v1.parquet"), "horizontal_base/hb"))
    parts.append(_print_load(_load_regime(
        OUT / "decision_engine_lab_regime_transition_events_3y.parquet",
        "nox_rt_daily", "regime_transition"), "nox_rt_daily"))
    parts.append(_print_load(_load_regime(
        OUT / "decision_engine_lab_weekly_regime_transition_events_3y.parquet",
        "nox_weekly", "weekly_regime_transition"), "nox_weekly"))
    parts.append(_print_load(_load_nyx(OUT / "decision_engine_lab_momentum_picks_3y.parquet"),
                             "nyxmomentum"))
    parts.append(_print_load(_load_paper_line_e(OUT / "paper_execution_v0_trades.parquet"),
                             "paper_line_e"))
    parts.append(_print_load(_load_paper_line_tr(OUT / "paper_execution_v0_trigger_retest_trades.parquet"),
                             "paper_line_tr"))
    parts.append(_print_load(_load_alpha(OUT / "decision_engine_lab_alpha_events_3y.parquet"),
                             "nyxalpha/alpha"))
    parts.append(_print_load(_load_smart_breakout(OUT / "decision_engine_lab_smart_breakout_events_3y.parquet"),
                             "smart_breakout"))
    parts.append(_print_load(_load_triangle_break(OUT / "decision_engine_lab_triangle_break_events_3y.parquet"),
                             "triangle_break"))

    unified = pd.concat([p for p in parts if p is not None and len(p)],
                        ignore_index=True, sort=False)
    print(f"[unified_events] stacked: {len(unified):,} rows")

    unified["signal_date"] = pd.to_datetime(unified["signal_date"]).dt.normalize()
    unified = unified.dropna(subset=["ticker", "signal_date"]).reset_index(drop=True)

    print(f"[unified_events] joining forward returns + atr_14_daily...")
    unified = unified.merge(ref, on=["ticker", "signal_date"], how="left")

    # Order columns
    unified = unified[UNIFIED_COLS]

    # Cast to consistent dtypes
    unified["atr"] = pd.to_numeric(unified["atr"], errors="coerce")
    unified["entry_ref"] = pd.to_numeric(unified["entry_ref"], errors="coerce")

    print(f"[unified_events] writing parquet → {DST_PARQUET}")
    unified.to_parquet(DST_PARQUET, index=False)

    # ---- markdown summary ----
    lines: list[str] = []
    lines.append("# Decision Engine Lab — Unified Events (Faz I, 3y)")
    lines.append("")
    lines.append(f"- Output: `{DST_PARQUET.name}`  ({DST_PARQUET.stat().st_size:,} B)")
    lines.append(f"- Total rows: {len(unified):,}  ({unified['ticker'].nunique()} tickers)")
    lines.append(f"- Date range: {unified['signal_date'].min().date()} → "
                 f"{unified['signal_date'].max().date()}")
    lines.append(f"- Sources: {sorted(unified['source'].unique())}")
    lines.append(f"- Forward return horizons: {WINDOWS} trading days")
    lines.append(f"- Reference atr: Wilder ATR(14) on extfeed daily resample (`atr_14_daily`)")
    lines.append(f"- Source-native ATR (where available) preserved in `atr` column")
    lines.append("")

    lines.append("## A) Per-source counts + horizon coverage")
    lines.append("")
    lines.append(_per_source_summary(unified).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## B) Forward MFE distribution — overall pool")
    lines.append("")
    lines.append(_mfe_dist_block(unified).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## C) Per-source MFE 30d ranking")
    lines.append("")
    lines.append(_per_source_mfe30(unified).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    DST_MD.write_text("\n".join(lines))
    print(f"[unified_events] wrote {DST_MD}")


def _print_load(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  {label}: {len(df):,} rows")
    return df


if __name__ == "__main__":
    main()
