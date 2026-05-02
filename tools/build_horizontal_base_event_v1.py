"""V1 ML freeze dataset builder for horizontal_base.

Reproduces the V1.3.1 event population from `scanner_v1_event_quality.py`,
emits one schema-compliant row per event via
`scanner.triggers.horizontal_base._build_row`, attaches RDP v1 regime,
slope_tier, same-date rank_pct overlays, and forward labels (incl. binary
`quality_20d`). Writes parquet + manifest + split + audit log.

Production scanner is NOT modified. RDP regime override is local to this
builder; legacy `regime_labels_daily.csv` reads downstream of
`scanner.context` are intentionally bypassed (see plan §0 (b) / §4g).

Inputs (all under output/):
  scanner_v1_event_quality_events.csv   -- canonical event manifest + labels
  regime_labels_daily_rdp_v1.csv         -- RDP v1 regime labels (date join)
  xu100_extfeed_daily.parquet            -- XU100 OHLC for market context
  extfeed_intraday_1h_3y_master.parquet  -- bar data via intraday_1h adapter

Outputs:
  output/horizontal_base_event_v1.parquet
  output/horizontal_base_event_v1_manifest.json
  output/horizontal_base_event_v1_split.json
  output/horizontal_base_event_v1_audit.log
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import intraday_1h  # noqa: E402
from scanner import context as scctx  # noqa: E402
from scanner.context import build_market_context, fill_cross_sectional  # noqa: E402
from scanner.schema import (  # noqa: E402
    FAMILIES,
    FEATURE_VERSION,
    SCANNER_VERSION,
    SCHEMA_VERSION,
    output_columns_for,
)
from scanner.triggers.horizontal_base import (  # noqa: E402
    ATR_SL_MULT,
    BB_LENGTH,
    MAX_SQUEEZE_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _build_row,
    _classify_state,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)

# ----------------------------------------------------------------------------
# paths

EVENTS_CSV = ROOT / "output/scanner_v1_event_quality_events.csv"
REGIME_CSV = ROOT / "output/regime_labels_daily_rdp_v1.csv"
XU100_PARQUET = ROOT / "output/xu100_extfeed_daily.parquet"
SCANNER_HORIZONTAL_BASE = ROOT / "scanner/triggers/horizontal_base.py"

OUT_PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
OUT_MANIFEST = ROOT / "output/horizontal_base_event_v1_manifest.json"
OUT_SPLIT = ROOT / "output/horizontal_base_event_v1_split.json"
OUT_AUDIT = ROOT / "output/horizontal_base_event_v1_audit.log"

# ----------------------------------------------------------------------------
# constants (from plan)

QUALITY_TH = 2.0  # MFE_R_20d >= 2.0 → quality_20d True

SLOPE_FLAT = 0.0015
SLOPE_MILD = 0.0030
SLOPE_LOOSE = SLOPE_REJECT_PER_DAY  # 0.0050

# Rank-pct overlay raw cols (computed within same-date all-ticker panel)
RANK_PCT_RAW = (
    "common__turnover",
    "common__volume_ratio_20",
    "common__atr_pct",
    "common__rs_20d",
    "common__rs_60d",
)

# Walk-forward fold definitions (val windows; train = all events with date < val_start)
FOLD_DEFS = [
    ("F1", "2024-01-01", "2024-04-01"),
    ("F2", "2024-07-01", "2025-01-01"),
    ("F3", "2025-01-01", "2025-07-01"),
    ("F4", "2026-01-01", "2027-01-01"),
]

DATASET_NAME = "horizontal_base_event_v1"

# ----------------------------------------------------------------------------
# helpers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slope_tier(abs_b: float, abs_r: float) -> str:
    a = abs_b if pd.notna(abs_b) else 0.0
    b = abs_r if pd.notna(abs_r) else 0.0
    m = max(a, b)
    if m <= SLOPE_FLAT:
        return "flat"
    if m <= SLOPE_MILD:
        return "mild"
    if m <= SLOPE_LOOSE:
        return "loose"
    return "rejected_diag"


def _val_fold_for_date(d: pd.Timestamp) -> str:
    for fid, vs, ve in FOLD_DEFS:
        if pd.Timestamp(vs) <= d < pd.Timestamp(ve):
            return fid
    return ""


def _patch_load_regime_for_index() -> callable:
    """Return original load_regime so callers can restore.

    Replaces `intraday_1h.load_regime` with one returning RDP regime joined
    onto the long-history XU100 OHLC from `xu100_extfeed_daily.parquet`. This
    lets `scanner.context.build_market_context` use the long XU100 series for
    cross-sectional context (index returns, breadth, RS) while categorical
    regime labels come from RDP v1.
    """
    original = intraday_1h.load_regime

    def _load_regime_rdp() -> pd.DataFrame:
        rdp = pd.read_csv(REGIME_CSV)
        rdp["date"] = pd.to_datetime(rdp["date"]).dt.date
        xu = pd.read_parquet(XU100_PARQUET)
        if xu.index.name == "Date":
            xu = xu.reset_index().rename(columns={"Date": "date"})
        xu["date"] = pd.to_datetime(xu["date"]).dt.date
        xu = xu[["date", "close"]].drop_duplicates("date").sort_values("date")
        merged = xu.merge(
            rdp[["date", "regime", "sub_regime", "window_id", "label_source"]],
            on="date",
            how="left",
        )
        # Pre-RDP rows (before 2023-01-19) get NaN regime — that's OK; events
        # start 2023-04-12 anyway. Fill identifier columns for downstream code
        # that may assume non-null label_source.
        merged["label_source"] = merged["label_source"].fillna("rdp_v1_warmup")
        merged["window_id"] = merged["window_id"].fillna("rdp_v1_warmup")
        return merged

    intraday_1h.load_regime = _load_regime_rdp
    return original


def _restore_load_regime(original: callable) -> None:
    intraday_1h.load_regime = original


# ----------------------------------------------------------------------------
# event loop (mirrors event_quality.scan_ticker but emits _build_row outputs)


def _emit_rows_for_ticker(daily_df: pd.DataFrame, ticker: str) -> list[dict]:
    """Emit one schema-compliant row per event matching the event_quality
    scan logic on this ticker. Forward labels are NOT included here — they
    are joined later from the canonical events CSV.
    """
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return []

    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])
    n = len(df)
    cycle_struct: dict = {}
    out: list[dict] = []

    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue
        if pos not in cycle_struct:
            base = df.iloc[sq_s: sq_e + 1]
            b_slope, _ = _line_fit(base["close"].values)
            r_slope, _ = _line_fit(base["high"].values)
            if abs(b_slope) > SLOPE_REJECT_PER_DAY or abs(r_slope) > SLOPE_REJECT_PER_DAY:
                cycle_struct[pos] = None
            else:
                box_top, hard, box_bot = _box_from_squeeze(df, sq_s, sq_e)
                if not (box_top > box_bot > 0):
                    cycle_struct[pos] = None
                else:
                    cycle_struct[pos] = (box_top, hard, box_bot)
        cs = cycle_struct[pos]
        if cs is None:
            continue
        box_top, hard, box_bot = cs
        asof_close = float(df["close"].iat[asof_idx])
        if asof_close <= 0:
            continue
        if (box_top - box_bot) / asof_close > WIDTH_PCT_REJECT:
            continue
        state, breakout_idx = _classify_state(df, asof_idx, sq_s, sq_e, box_top, hard)
        if state == "none" or state == "pre_breakout":
            continue
        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_sq <= 0:
            continue
        entry_close = asof_close
        invalidation = box_bot - ATR_SL_MULT * atr_sq
        R_per_share = entry_close - invalidation
        if R_per_share <= 0:
            continue

        row = _build_row(
            state=state,
            ticker=ticker,
            df=df,
            asof_idx=int(asof_idx),
            breakout_idx=int(breakout_idx) if breakout_idx is not None else None,
            sq_start=int(sq_s),
            sq_end=int(sq_e),
            box_top=float(box_top),
            box_bot=float(box_bot),
            hard_resistance=float(hard),
            direction=+1,
        )
        # Carry asof_idx / breakout_idx for cross-check vs events CSV.
        row["asof_idx"] = int(asof_idx)
        row["breakout_idx"] = int(breakout_idx) if breakout_idx is not None else -1
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# rank_pct overlays — same-date all-ticker cross-section


def _rank_pct_panel_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """Compute rank_pct overlays within same (year-month-day) panel of events.

    Plan §4e calls for "same-day all-ticker panel" — using the events panel
    (10,470 rows over ~700 unique dates) as the reference cross-section.
    Each event is ranked among ALL events occurring on its date (every state
    contributes — extended events make the panel ~6x denser per day than
    trade-only events would). This is audit-safe (close-after, no intraday
    information) and avoids re-running indicator math over the full 607x822
    daily grid.

    For the trade universe (1,956 events) this gives ~14 events/day median
    panel size — adequate for ML to learn rank position.

    Returns the events DataFrame with new <col>_rank_pct_date columns.
    """
    out = events.copy()
    out["_panel_date"] = pd.to_datetime(out["bar_date"]).dt.normalize()
    for raw in RANK_PCT_RAW:
        if raw not in out.columns:
            out[f"{raw}_rank_pct_date"] = np.float32("nan")
            continue
        out[f"{raw}_rank_pct_date"] = (
            out.groupby("_panel_date")[raw].rank(pct=True, method="average").astype("float32")
        )
    out = out.drop(columns="_panel_date")
    return out


# ----------------------------------------------------------------------------
# forward labels join


FWD_LABEL_COLS = [
    "mfe_R_3d", "mae_R_3d", "realized_R_3d", "failed_breakout_3d", "time_to_MFE_3d",
    "mfe_R_5d", "mae_R_5d", "realized_R_5d", "failed_breakout_5d", "time_to_MFE_5d",
    "mfe_R_10d", "mae_R_10d", "realized_R_10d", "failed_breakout_10d", "time_to_MFE_10d",
    "mfe_R_20d", "mae_R_20d", "realized_R_20d", "failed_breakout_20d", "time_to_MFE_20d",
    "early_failure_5d",
]


def _attach_labels(rows: pd.DataFrame, events_csv: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["_join_date"] = pd.to_datetime(rows["bar_date"]).dt.normalize()
    ev = events_csv.copy()
    ev["_join_date"] = pd.to_datetime(ev["date"]).dt.normalize()
    keep = ["ticker", "_join_date"] + FWD_LABEL_COLS
    merged = rows.merge(ev[keep], on=["ticker", "_join_date"], how="left", validate="one_to_one")
    merged = merged.drop(columns="_join_date")
    merged["quality_20d"] = (merged["mfe_R_20d"] >= QUALITY_TH).astype("Int8")
    return merged


# ----------------------------------------------------------------------------
# main


def main() -> int:
    t0 = time.time()
    print(f"V1 freeze builder — SCANNER={SCANNER_VERSION} FEATURE={FEATURE_VERSION} SCHEMA={SCHEMA_VERSION}",
          flush=True)
    print(f"  events:  {EVENTS_CSV.relative_to(ROOT)}", flush=True)
    print(f"  regime:  {REGIME_CSV.relative_to(ROOT)} (RDP v1 OFFLINE)", flush=True)
    print(f"  xu100:   {XU100_PARQUET.relative_to(ROOT)}", flush=True)

    # 1) Load events CSV (canonical event manifest with forward labels)
    events_csv = pd.read_csv(EVENTS_CSV)
    events_csv["date"] = pd.to_datetime(events_csv["date"]).dt.normalize()
    print(f"[1/8] events CSV: {len(events_csv):,} rows / "
          f"{events_csv['ticker'].nunique()} tickers / "
          f"{events_csv['date'].nunique()} unique dates", flush=True)

    # 2) Load 1h bars + daily resample
    print("[2/8] loading 1h master + daily resample …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    daily = intraday_1h.daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    print(f"  bars: {len(bars):,}  daily rows: {len(daily):,}  "
          f"tickers: {daily['ticker'].nunique()}", flush=True)

    # 3) Build market context (with RDP-overridden index source)
    print("[3/8] build_market_context (RDP-patched _index_daily) …", flush=True)
    original_load_regime = _patch_load_regime_for_index()
    try:
        market_df, rs_df = build_market_context(daily)
    finally:
        _restore_load_regime(original_load_regime)
    print(f"  market_df: {len(market_df):,} dates  rs_df: {len(rs_df):,} (ticker,date)",
          flush=True)

    # 4) Per-ticker event scan via _build_row
    print("[4/8] per-ticker event scan + _build_row …", flush=True)
    t1 = time.time()
    tickers = sorted(daily["ticker"].unique())
    all_rows: list[dict] = []
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        sub_df.attrs["ticker"] = str(t)
        all_rows.extend(_emit_rows_for_ticker(sub_df, str(t)))
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t1:.1f}s "
                  f"events={len(all_rows):,}", flush=True)
    print(f"  scan elapsed: {time.time()-t1:.1f}s  rows: {len(all_rows):,}", flush=True)

    rows = pd.DataFrame(all_rows)
    if "bar_date" not in rows.columns:
        raise RuntimeError("_build_row missing bar_date")
    rows["bar_date"] = pd.to_datetime(rows["bar_date"]).dt.normalize()

    # 5) Cross-sectional fill (RS, index, breadth, market_trend, market_vol)
    print("[5/8] fill_cross_sectional + RDP regime override …", flush=True)
    cross_sec_cols = [
        "common__rs_20d", "common__rs_60d", "common__rs_pctile_120",
        "common__rs_pctile_252", "common__sector_rs_20d",
        "common__index_ret_5d", "common__index_ret_20d",
        "common__market_trend_score", "common__market_vol_regime",
        "common__market_breadth_pct_above_sma20",
    ]
    for c in cross_sec_cols:
        if c not in rows.columns:
            rows[c] = np.float32("nan")
    # apply per-row
    enriched: list[dict] = []
    for _, r in rows.iterrows():
        d = dict(r)
        fill_cross_sectional(d, market_df, rs_df)
        enriched.append(d)
    rows = pd.DataFrame(enriched)

    # categorical regime from RDP CSV (date as-of backward)
    rdp = pd.read_csv(REGIME_CSV)
    rdp["date"] = pd.to_datetime(rdp["date"]).dt.normalize()
    rdp = rdp.sort_values("date").drop_duplicates("date", keep="last")
    rdp_lookup = rdp.set_index("date")[["regime", "sub_regime", "window_id"]]
    # backward as-of: each event's regime = regime at the most recent RDP date <= bar_date
    rows = rows.sort_values("bar_date")
    rdp_indexed = rdp_lookup.sort_index()
    rows["common__regime"] = pd.Series(
        rdp_indexed["regime"]
        .reindex(rows["bar_date"].values, method="ffill")
        .values,
        index=rows.index,
    )
    rows["regime_sub"] = pd.Series(
        rdp_indexed["sub_regime"]
        .reindex(rows["bar_date"].values, method="ffill")
        .values,
        index=rows.index,
    )
    rows["regime_window_id"] = "rdp_v1_mult8_atr14_sw8"
    print(f"  regime fill: {(rows['common__regime'].notna()).sum():,} / {len(rows):,}",
          flush=True)

    # 6) slope_tier (derived from base_slope/resistance_slope)
    rows["family__abs_base_slope"] = rows["family__base_slope"].abs().astype("float32")
    rows["family__abs_resistance_slope"] = rows["family__resistance_slope"].abs().astype("float32")
    rows["family__slope_tier"] = [
        _slope_tier(a, b)
        for a, b in zip(rows["family__abs_base_slope"], rows["family__abs_resistance_slope"])
    ]

    # 7) rank_pct overlays (same-date panel — uses events as the cross-section)
    rows = _rank_pct_panel_from_events(rows)

    # 8) attach forward labels + quality_20d
    rows = _attach_labels(rows, events_csv)

    # cross-check vs events CSV count
    n_expect = len(events_csv)
    n_built = len(rows)
    if n_built != n_expect:
        print(f"  !! WARN built={n_built} != events_csv={n_expect}", flush=True)

    # state cross-check
    print("  state counts (built):"); print(rows["signal_state"].value_counts().to_string())

    # fold assignment helper col (used in split.json + downstream trainer)
    rows["bar_year"] = rows["bar_date"].dt.year.astype("int16")
    rows["val_fold"] = rows["bar_date"].apply(_val_fold_for_date)

    # ------------------------------------------------------------------
    # write parquet
    print(f"[6/8] writing parquet → {OUT_PARQUET.relative_to(ROOT)}", flush=True)
    rows.to_parquet(OUT_PARQUET, index=False)
    parquet_sha = _sha256(OUT_PARQUET)
    print(f"  rows={len(rows):,}  cols={rows.shape[1]}  sha256={parquet_sha[:12]}…",
          flush=True)

    # ------------------------------------------------------------------
    # manifest
    print("[7/8] writing manifest + split", flush=True)
    label_dist = {
        "n_total": int(len(rows)),
        "n_trigger": int((rows["signal_state"] == "trigger").sum()),
        "n_retest_bounce": int((rows["signal_state"] == "retest_bounce").sum()),
        "n_extended": int((rows["signal_state"] == "extended").sum()),
        "quality_20d_pos": int(rows["quality_20d"].fillna(0).sum()),
        "quality_20d_total_labeled": int(rows["mfe_R_20d"].notna().sum()),
        "quality_20d_pos_rate_trade_only": float(
            rows.loc[rows["signal_state"].isin(["trigger", "retest_bounce"]),
                     "quality_20d"].fillna(0).mean()
        ),
    }
    manifest = {
        "dataset": DATASET_NAME,
        "version": "v1",
        "scanner_version": SCANNER_VERSION,
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "frozen_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(rows)),
        "feature_cols": int(rows.shape[1]),
        "input_artifacts": {
            "events_csv": str(EVENTS_CSV.relative_to(ROOT)),
            "events_csv_sha256": _sha256(EVENTS_CSV),
            "regime_artifact": str(REGIME_CSV.relative_to(ROOT)),
            "regime_artifact_sha256": _sha256(REGIME_CSV),
            "regime_window_id": "rdp_v1_mult8_atr14_sw8",
            "regime_source": "rdp_v1",
            "legacy_sma_cross": "deprecated, NOT used",
            "xu100_index_parquet": str(XU100_PARQUET.relative_to(ROOT)),
            "xu100_index_parquet_sha256": _sha256(XU100_PARQUET),
            "scanner_horizontal_base_sha256": _sha256(SCANNER_HORIZONTAL_BASE),
        },
        "label_horizon_unit": "trading_days",
        "label_start": "next_bar_after_asof",
        "primary_target": "quality_20d",
        "primary_target_definition": "MFE_R_20d >= 2.0",
        "label_dist": label_dist,
        "slope_tier_thresholds": {
            "flat": SLOPE_FLAT,
            "mild": SLOPE_MILD,
            "loose": SLOPE_LOOSE,
        },
        "rank_pct_columns": [f"{c}_rank_pct_date" for c in RANK_PCT_RAW],
        "rank_pct_panel_definition": "same-date events panel cross-section",
        "out_parquet": str(OUT_PARQUET.relative_to(ROOT)),
        "out_parquet_sha256": parquet_sha,
        "production_lock": {
            "horizontal_base_body_floor": 0.35,
            "horizontal_base_slope_cap_per_day": SLOPE_REJECT_PER_DAY,
            "retest_bounce_enabled": True,
            "rejected_diag_in_universe": False,
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    split = {
        "dataset": DATASET_NAME,
        "version": "v1",
        "fold_definitions": [
            {"fold_id": fid, "val_start": vs, "val_end": ve,
             "train_rule": f"all events with bar_date < {vs}"}
            for fid, vs, ve in FOLD_DEFS
        ],
        "fold_event_counts": {
            fid: {
                "train": int((rows["bar_date"] < pd.Timestamp(vs)).sum()),
                "val": int(((rows["bar_date"] >= pd.Timestamp(vs)) &
                            (rows["bar_date"] < pd.Timestamp(ve))).sum()),
                "trade_train": int(((rows["bar_date"] < pd.Timestamp(vs)) &
                                    rows["signal_state"].isin(["trigger", "retest_bounce"])).sum()),
                "trade_val": int(((rows["bar_date"] >= pd.Timestamp(vs)) &
                                   (rows["bar_date"] < pd.Timestamp(ve)) &
                                   rows["signal_state"].isin(["trigger", "retest_bounce"])).sum()),
            }
            for fid, vs, ve in FOLD_DEFS
        },
        "trade_universe_filter": "signal_state in ['trigger', 'retest_bounce']",
        "reference_only": "signal_state == 'extended'",
    }
    OUT_SPLIT.write_text(json.dumps(split, indent=2))

    # ------------------------------------------------------------------
    # audit log (Phase A audit lite — full audit in Phase A2)
    print("[8/8] audit log", flush=True)
    audit_lines = []
    audit_lines.append(f"V1 freeze builder audit — {pd.Timestamp.utcnow().isoformat()}")
    audit_lines.append(f"  rows: {len(rows):,}  cols: {rows.shape[1]}")
    audit_lines.append(f"  expected (events CSV): {n_expect:,}")
    audit_lines.append(f"  state counts: " + rows["signal_state"].value_counts().to_dict().__str__())
    audit_lines.append(f"  body_class counts: " + rows["family__body_class"].value_counts().to_dict().__str__())
    audit_lines.append(f"  slope_tier counts: " + rows["family__slope_tier"].value_counts().to_dict().__str__())
    audit_lines.append(f"  regime counts: " + rows["common__regime"].value_counts(dropna=False).to_dict().__str__())
    audit_lines.append(f"  bar_date range: {rows['bar_date'].min()} → {rows['bar_date'].max()}")
    audit_lines.append(f"  quality_20d positive (trade only): "
                       f"{int(rows.loc[rows['signal_state'].isin(['trigger', 'retest_bounce']), 'quality_20d'].fillna(0).sum()):,} "
                       f"/ {int(rows['signal_state'].isin(['trigger', 'retest_bounce']).sum()):,}")
    audit_lines.append("  fold trade event counts:")
    for fid, _, _ in FOLD_DEFS:
        audit_lines.append(f"    {fid}: train={split['fold_event_counts'][fid]['trade_train']:,} "
                           f"val={split['fold_event_counts'][fid]['trade_val']:,}")
    audit_lines.append(f"  artifacts:")
    audit_lines.append(f"    events_csv_sha256:        {manifest['input_artifacts']['events_csv_sha256'][:16]}…")
    audit_lines.append(f"    regime_artifact_sha256:   {manifest['input_artifacts']['regime_artifact_sha256'][:16]}…")
    audit_lines.append(f"    xu100_index_parquet_sha256: {manifest['input_artifacts']['xu100_index_parquet_sha256'][:16]}…")
    audit_lines.append(f"    scanner_horizontal_base_sha256: {manifest['input_artifacts']['scanner_horizontal_base_sha256'][:16]}…")
    audit_lines.append(f"    out_parquet_sha256:       {parquet_sha[:16]}…")
    audit_lines.append(f"  total elapsed: {time.time()-t0:.1f}s")
    OUT_AUDIT.write_text("\n".join(audit_lines) + "\n")
    print("\n".join(audit_lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
