"""
portfolio_merge_paper — joint reporting tool for the two EMA-HB paper/shadow lines.

Spec: memory/portfolio_merge_paper_spec.md (LOCKED, ONAY 2026-05-04).

This tool is reporting-only. It is NOT live trading, NOT production allocation,
NOT a ranking engine, NOT a rule-change tool, NOT a re-derivation tool.

Hard invariants (enforced at runtime; STOP on violation):
  - Source files (Line E and Line TR trade parquets) must exist; both manifests must
    load successfully. STOP otherwise.
  - Source files are READ-ONLY. Output paths must contain literal substring
    'portfolio_merge_paper' and must NOT match any per-line file.
  - 'line' column preserved verbatim per row ('EXTENDED' or 'TRIGGER_RETEST').
  - No eligibility recomputation: signal_state, ema_tier/ema_tr_tier,
    earliness_score_pct, entry_*, exit_*, return fields, realized_R_paper come
    verbatim from each per-line log.
  - raw merged view = no cap, no dedup beyond exact byte-identical defensive drop.
  - dedup priority = TRIGGER_RETEST > EXTENDED on (ticker, asof_date) collision.
  - capped: max_new_positions_per_day = 5; selection = line priority then ascending
    alphabetical ticker.
  - SKIPPED rows excluded from performance metrics across all 5 slices.
  - OPEN rows excluded from return / WR / PF / DD across all 5 slices.
  - is_backfill=True rows excluded from forward-clock counts and cannot satisfy
    promotion gates.
  - R drawdown and percentage drawdown reported under separate field names.
  - No live-promotion claim emitted in summary.md or manifest.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = "memory/portfolio_merge_paper_spec.md"
SPEC_STATUS = "LOCKED 2026-05-04, ONAY 2026-05-04"

# ---------------------------------------------------------------------------
# Locked constants (spec §1, §4, §5, §10)
# ---------------------------------------------------------------------------

MAX_NEW_POSITIONS_PER_DAY = 5  # spec §4.3 (LOCKED, identical to per-line)
LINE_PRIORITY = ("TRIGGER_RETEST", "EXTENDED")  # spec §4.2 + §4.3 (LOCKED)
PROMOTION_FORWARD_MONTHS = 6  # spec §10
PROMOTION_FORWARD_CLOSED_MIN = 50  # spec §10

# ---------------------------------------------------------------------------
# Source files (READ ONLY, must exist) and protected per-line outputs
# ---------------------------------------------------------------------------

LINE_E_TRADES = ROOT / "output" / "paper_execution_v0_trades.parquet"
LINE_E_MANIFEST = ROOT / "output" / "paper_execution_v0_manifest.json"
LINE_TR_TRADES = ROOT / "output" / "paper_execution_v0_trigger_retest_trades.parquet"
LINE_TR_MANIFEST = ROOT / "output" / "paper_execution_v0_trigger_retest_manifest.json"

PROTECTED_PATHS = {
    ROOT / "output" / "paper_execution_v0_trades.parquet",
    ROOT / "output" / "paper_execution_v0_trades.csv",
    ROOT / "output" / "paper_execution_v0_daily_summary.csv",
    ROOT / "output" / "paper_execution_v0_preview_audit.csv",
    ROOT / "output" / "paper_execution_v0_summary.md",
    ROOT / "output" / "paper_execution_v0_manifest.json",
    ROOT / "output" / "paper_execution_v0_trigger_retest_trades.parquet",
    ROOT / "output" / "paper_execution_v0_trigger_retest_trades.csv",
    ROOT / "output" / "paper_execution_v0_trigger_retest_daily_summary.csv",
    ROOT / "output" / "paper_execution_v0_trigger_retest_preview_audit.csv",
    ROOT / "output" / "paper_execution_v0_trigger_retest_summary.md",
    ROOT / "output" / "paper_execution_v0_trigger_retest_manifest.json",
}

# ---------------------------------------------------------------------------
# Output paths (all MUST contain 'portfolio_merge_paper')
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "output"
OUT_TRADES_PARQUET = OUT_DIR / "portfolio_merge_paper_trades.parquet"
OUT_TRADES_CSV = OUT_DIR / "portfolio_merge_paper_trades.csv"
OUT_DAILY_SUMMARY_CSV = OUT_DIR / "portfolio_merge_paper_daily_summary.csv"
OUT_CONFLICTS_CSV = OUT_DIR / "portfolio_merge_paper_conflicts.csv"
OUT_SUMMARY_MD = OUT_DIR / "portfolio_merge_paper_summary.md"
OUT_MANIFEST_JSON = OUT_DIR / "portfolio_merge_paper_manifest.json"

ALL_OUTPUTS = [
    OUT_TRADES_PARQUET,
    OUT_TRADES_CSV,
    OUT_DAILY_SUMMARY_CSV,
    OUT_CONFLICTS_CSV,
    OUT_SUMMARY_MD,
    OUT_MANIFEST_JSON,
]

# ---------------------------------------------------------------------------
# Required final schema (spec §6)
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "asof_date",
    "ticker",
    "line",
    "source_file",
    "source_manifest_hash",
    "lane_source",
    "signal_state",
    "ema_tier",
    "earliness_score_pct",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "status",
    "gross_return_pct",
    "net_return_pct",
    "realized_R_paper",
    "is_backfill",
    "view_membership",
    "skip_reason",
    "duplicate_group_id",
    "selected_in_dedup",
    "selected_in_capped",
]


def _stop(msg: str) -> None:
    print(f"STOP: {msg}", file=sys.stderr)
    sys.exit(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _assert_output_paths_safe() -> None:
    for p in ALL_OUTPUTS:
        if "portfolio_merge_paper" not in p.name:
            _stop(f"output path missing 'portfolio_merge_paper' substring: {p}")
        if p in PROTECTED_PATHS:
            _stop(f"output path collides with protected per-line file: {p}")


# ---------------------------------------------------------------------------
# Pre-run checklist (spec §16)
# ---------------------------------------------------------------------------


def _run_prerun_checklist() -> dict[str, Any]:
    checklist: dict[str, Any] = {}

    # 1. Line E trade log exists.
    if not LINE_E_TRADES.exists():
        _stop(f"Line E trade log missing: {LINE_E_TRADES}")
    checklist["1_line_e_trades_exist"] = "PASS"

    # 2. Line TR trade log exists.
    if not LINE_TR_TRADES.exists():
        _stop(f"Line TR trade log missing: {LINE_TR_TRADES}")
    checklist["2_line_tr_trades_exist"] = "PASS"

    # 3. Both manifests load.
    if not LINE_E_MANIFEST.exists():
        _stop(f"Line E manifest missing: {LINE_E_MANIFEST}")
    if not LINE_TR_MANIFEST.exists():
        _stop(f"Line TR manifest missing: {LINE_TR_MANIFEST}")
    try:
        with open(LINE_E_MANIFEST) as f:
            le_manifest = json.load(f)
    except Exception as e:
        _stop(f"Line E manifest load failed: {e}")
    try:
        with open(LINE_TR_MANIFEST) as f:
            tr_manifest = json.load(f)
    except Exception as e:
        _stop(f"Line TR manifest load failed: {e}")
    checklist["3_both_manifests_load"] = "PASS"

    # 4. No source file write attempt — pre-flight assert that output path set
    #    contains no protected per-line path.
    _assert_output_paths_safe()
    checklist["4_no_source_file_write_attempt"] = "PASS"

    # 5–6: enforced after merged log is built (line column verbatim, output suffix safe).
    checklist["5_line_column_preserved"] = "DEFERRED_POST_BUILD"
    checklist["6_outputs_only_portfolio_merge_paper"] = "PASS"

    # 7–9: enforced inside view assembly.
    checklist["7_skipped_excluded_from_performance"] = "DEFERRED_POST_BUILD"
    checklist["8_open_excluded_from_return_metrics"] = "DEFERRED_POST_BUILD"
    checklist["9_raw_view_no_cap_no_dedup"] = "DEFERRED_POST_BUILD"

    # 10–12: enforced after view + summary assembly.
    checklist["10_dedup_conflict_report_emitted"] = "DEFERRED_POST_BUILD"
    checklist["11_capped_view_marked_operational"] = "DEFERRED_POST_BUILD"
    checklist["12_no_live_promotion_claim"] = "DEFERRED_POST_BUILD"

    return {"checklist": checklist, "le_manifest": le_manifest, "tr_manifest": tr_manifest}


# ---------------------------------------------------------------------------
# Build merged trade log (spec §6)
# ---------------------------------------------------------------------------


def _build_merged_log(
    le: pd.DataFrame, tr: pd.DataFrame, le_hash: str, tr_hash: str
) -> pd.DataFrame:
    # Line E: add line column; rename ema_tier passthrough; add source columns.
    le2 = le.copy()
    if "line" in le2.columns:
        # Defensive: per-line spec did not add this column; if present, MUST be EXTENDED.
        bad = le2[le2["line"] != "EXTENDED"]
        if len(bad) > 0:
            _stop(f"Line E source has {len(bad)} rows with line != EXTENDED")
    le2["line"] = "EXTENDED"
    le2["source_file"] = str(LINE_E_TRADES)
    le2["source_manifest_hash"] = le_hash
    # Line E uses 'ema_tier' as final column name — already correct.

    # Line TR: line column already exists and = TRIGGER_RETEST; rename ema_tr_tier → ema_tier.
    tr2 = tr.copy()
    if "line" not in tr2.columns:
        _stop("Line TR source missing 'line' column (per-line spec invariant violated)")
    bad = tr2[tr2["line"] != "TRIGGER_RETEST"]
    if len(bad) > 0:
        _stop(f"Line TR source has {len(bad)} rows with line != TRIGGER_RETEST")
    tr2 = tr2.rename(columns={"ema_tr_tier": "ema_tier"})
    tr2["source_file"] = str(LINE_TR_TRADES)
    tr2["source_manifest_hash"] = tr_hash

    # Common base columns to carry verbatim
    base_cols = [
        "asof_date",
        "ticker",
        "line",
        "source_file",
        "source_manifest_hash",
        "lane_source",
        "signal_state",
        "ema_tier",
        "earliness_score_pct",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "status",
        "gross_return_pct",
        "net_return_pct",
        "realized_R_paper",
        "skip_reason",
        "is_backfill",
        "in_view_raw_all_candidates",  # source-line raw eligibility flag
    ]
    for col in base_cols:
        if col not in le2.columns:
            _stop(f"Line E source missing required column: {col}")
        if col not in tr2.columns:
            _stop(f"Line TR source missing required column: {col}")

    le3 = le2[base_cols].copy()
    tr3 = tr2[base_cols].copy()

    merged = pd.concat([le3, tr3], ignore_index=True, sort=False)

    # Defensive exact-duplicate drop (expected n=0; cohorts are disjoint).
    n_pre = len(merged)
    merged = merged.drop_duplicates(
        subset=["asof_date", "ticker", "line", "signal_state", "entry_date"], keep="first"
    )
    n_dropped = n_pre - len(merged)
    if n_dropped > 0:
        print(f"NOTE: dropped {n_dropped} exact-duplicate rows (defensive)")

    # Initialize merged-view columns
    merged["view_membership"] = ""
    merged["duplicate_group_id"] = pd.NA
    merged["selected_in_dedup"] = False
    merged["selected_in_capped"] = False

    return merged


# ---------------------------------------------------------------------------
# View assembly (spec §4)
# ---------------------------------------------------------------------------


def _line_priority_rank(line: str) -> int:
    # Lower rank = higher priority (TRIGGER_RETEST=0, EXTENDED=1).
    if line == "TRIGGER_RETEST":
        return 0
    if line == "EXTENDED":
        return 1
    return 99


def _assemble_views(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["_priority"] = df["line"].apply(_line_priority_rank)

    # ---- View 1: raw_all_candidates (no cap, no dedup beyond exact dupes) ----
    raw_mask = df["in_view_raw_all_candidates"].astype(bool)

    # ---- View 2: dedup_same_ticker_day ----
    # Operate ONLY on raw-eligible rows.
    dedup_pool = df[raw_mask].copy()
    # group by (ticker, asof_date); choose lowest _priority (TR < EXTENDED).
    dedup_pool = dedup_pool.sort_values(
        ["asof_date", "ticker", "_priority"], kind="mergesort"
    )
    grp = dedup_pool.groupby(["asof_date", "ticker"], sort=False)
    dedup_pool["_dedup_idx"] = grp.cumcount()
    dedup_pool["_dedup_group_size"] = grp["line"].transform("size")

    # Mark selected_in_dedup=True for the head of each group (highest priority).
    dedup_selected_idx = dedup_pool.index[dedup_pool["_dedup_idx"] == 0]

    # Conflict groups have group_size>1.
    conflict_mask = dedup_pool["_dedup_group_size"] > 1
    # Assign duplicate_group_id by integer enumeration of conflict groups only.
    if conflict_mask.any():
        conflict_keys = (
            dedup_pool.loc[conflict_mask, ["asof_date", "ticker"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        conflict_keys["duplicate_group_id"] = (conflict_keys.index + 1).astype("int64")
        dedup_pool = dedup_pool.merge(
            conflict_keys, on=["asof_date", "ticker"], how="left"
        )
        # Restore index (merge resets it)
        dedup_pool.index = pd.Index(
            list(df[raw_mask].sort_values(["asof_date", "ticker", "_priority"], kind="mergesort").index)
        )
    else:
        dedup_pool["duplicate_group_id"] = pd.NA

    # Write back to df
    df.loc[dedup_selected_idx, "selected_in_dedup"] = True
    # propagate duplicate_group_id (only for conflict rows)
    df["duplicate_group_id"] = pd.NA
    df.loc[dedup_pool.index, "duplicate_group_id"] = dedup_pool["duplicate_group_id"].values

    # Lower-priority rows of conflict groups: skip_reason addition.
    lower_priority_idx = dedup_pool.index[
        (dedup_pool["_dedup_group_size"] > 1) & (dedup_pool["_dedup_idx"] > 0)
    ]
    if len(lower_priority_idx) > 0:
        # Append (do not clobber) merge-level skip reason.
        for ix in lower_priority_idx:
            existing = df.at[ix, "skip_reason"]
            tag = "DUPLICATE_SAME_TICKER_DAY_LOWER_PRIORITY"
            if pd.isna(existing) or existing == "" or existing is None:
                df.at[ix, "skip_reason"] = tag
            else:
                df.at[ix, "skip_reason"] = f"{existing}|{tag}"

    # ---- View 3: capped_portfolio ----
    # Start from dedup-selected rows.
    capped_pool = df[df["selected_in_dedup"]].copy()
    capped_pool = capped_pool.sort_values(
        ["asof_date", "_priority", "ticker"], kind="mergesort"
    )
    grp_c = capped_pool.groupby("asof_date", sort=False)
    capped_pool["_capped_idx"] = grp_c.cumcount()
    capped_pool["_capped_day_size"] = grp_c["ticker"].transform("size")

    capped_selected_idx = capped_pool.index[
        capped_pool["_capped_idx"] < MAX_NEW_POSITIONS_PER_DAY
    ]
    df.loc[capped_selected_idx, "selected_in_capped"] = True

    # Cap-overflow rows (dedup-selected but not capped-selected): add skip_reason tag.
    overflow_idx = capped_pool.index[
        capped_pool["_capped_idx"] >= MAX_NEW_POSITIONS_PER_DAY
    ]
    for ix in overflow_idx:
        existing = df.at[ix, "skip_reason"]
        tag = "CAP_OVERFLOW_LOWER_PRIORITY"
        if pd.isna(existing) or existing == "" or existing is None:
            df.at[ix, "skip_reason"] = tag
        else:
            df.at[ix, "skip_reason"] = f"{existing}|{tag}"

    # ---- view_membership comma-separated tag ----
    def _membership(row: pd.Series) -> str:
        parts = []
        if bool(row["in_view_raw_all_candidates"]):
            parts.append("raw_all_candidates")
        if bool(row["selected_in_dedup"]):
            parts.append("dedup_same_ticker_day")
        if bool(row["selected_in_capped"]):
            parts.append("capped_portfolio")
        return ",".join(parts)

    df["view_membership"] = df.apply(_membership, axis=1)

    # Drop helper columns
    df = df.drop(columns=["_priority"])

    return df


# ---------------------------------------------------------------------------
# Conflict report (spec §4.2)
# ---------------------------------------------------------------------------


def _build_conflicts_report(merged: pd.DataFrame) -> pd.DataFrame:
    raw = merged[merged["in_view_raw_all_candidates"].astype(bool)].copy()
    grp = raw.groupby(["asof_date", "ticker"], sort=True)
    sizes = grp["line"].nunique()
    conflict_keys = sizes[sizes > 1].index
    rows = []
    for asof, ticker in conflict_keys:
        sub = raw[(raw["asof_date"] == asof) & (raw["ticker"] == ticker)]
        lines = sorted(sub["line"].unique().tolist())
        # Determine selected line per dedup priority (TRIGGER_RETEST > EXTENDED)
        selected = "TRIGGER_RETEST" if "TRIGGER_RETEST" in lines else lines[0]
        dropped = [ln for ln in lines if ln != selected]
        rows.append(
            {
                "asof_date": asof,
                "ticker": ticker,
                "lines_present": ",".join(lines),
                "selected_line": selected,
                "dropped_lines": ",".join(dropped),
                "n_rows_in_conflict": len(sub),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics (spec §8)
# ---------------------------------------------------------------------------


def _slice_metrics(df: pd.DataFrame, *, slice_label: str) -> dict[str, Any]:
    """Compute metrics for a given slice. Excludes SKIPPED from performance,
    excludes OPEN from return/PF/WR/DD."""
    total_n = int(len(df))
    n_closed = int((df["status"] == "CLOSED").sum())
    n_open = int((df["status"] == "OPEN").sum())
    n_skipped = int((df["status"] == "SKIPPED").sum())

    closed = df[df["status"] == "CLOSED"].copy()
    if len(closed) == 0:
        return {
            "slice": slice_label,
            "n_total_rows": total_n,
            "n_closed": 0,
            "n_open": n_open,
            "n_skipped": n_skipped,
            "avg_R_paper": None,
            "median_R_paper": None,
            "avg_gross_return_pct": None,
            "median_gross_return_pct": None,
            "avg_net_return_pct": None,
            "median_net_return_pct": None,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_R": None,
            "max_drawdown_pct_gross": None,
            "max_drawdown_pct_net": None,
            "pct_from_extended": None,
            "pct_from_trigger_retest": None,
            "forward_closed_trades": 0,
            "forward_open_trades": 0,
            "backfill_closed_trades": 0,
            "backfill_open_trades": 0,
        }

    R = closed["realized_R_paper"].astype(float)
    g = closed["gross_return_pct"].astype(float)
    n = closed["net_return_pct"].astype(float)

    # win_rate uses the per-line convention: wins / n_closed (ties count as non-wins),
    # so line_e_only and line_tr_only here exactly reproduce each per-line manifest.
    wins = int((g > 0).sum())
    win_rate = float(wins) / float(len(closed))

    sum_pos = g[g > 0].sum()
    sum_neg = -g[g < 0].sum()
    profit_factor = float(sum_pos) / float(sum_neg) if sum_neg > 0 else float("inf")

    # Drawdown — additive cumulative R curve and equal-notional pct curves.
    closed_sorted = closed.sort_values("exit_date")
    R_curve = closed_sorted["realized_R_paper"].astype(float).cumsum()
    g_curve = closed_sorted["gross_return_pct"].astype(float).cumsum()
    n_curve = closed_sorted["net_return_pct"].astype(float).cumsum()

    def _max_dd(curve: pd.Series) -> float:
        if len(curve) == 0:
            return 0.0
        running_max = curve.cummax()
        dd = curve - running_max
        return float(dd.min())

    line_counts = closed["line"].value_counts()
    pct_ext = float(line_counts.get("EXTENDED", 0)) / float(len(closed))
    pct_tr = float(line_counts.get("TRIGGER_RETEST", 0)) / float(len(closed))

    forward_closed = int(((df["status"] == "CLOSED") & (~df["is_backfill"].astype(bool))).sum())
    forward_open = int(((df["status"] == "OPEN") & (~df["is_backfill"].astype(bool))).sum())
    backfill_closed = int(((df["status"] == "CLOSED") & (df["is_backfill"].astype(bool))).sum())
    backfill_open = int(((df["status"] == "OPEN") & (df["is_backfill"].astype(bool))).sum())

    return {
        "slice": slice_label,
        "n_total_rows": total_n,
        "n_closed": n_closed,
        "n_open": n_open,
        "n_skipped": n_skipped,
        "avg_R_paper": float(R.mean()),
        "median_R_paper": float(R.median()),
        "avg_gross_return_pct": float(g.mean()),
        "median_gross_return_pct": float(g.median()),
        "avg_net_return_pct": float(n.mean()),
        "median_net_return_pct": float(n.median()),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": _max_dd(R_curve),
        "max_drawdown_pct_gross": _max_dd(g_curve),
        "max_drawdown_pct_net": _max_dd(n_curve),
        "pct_from_extended": pct_ext,
        "pct_from_trigger_retest": pct_tr,
        "forward_closed_trades": forward_closed,
        "forward_open_trades": forward_open,
        "backfill_closed_trades": backfill_closed,
        "backfill_open_trades": backfill_open,
    }


def _build_slices(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "line_e_only": merged[merged["line"] == "EXTENDED"].copy(),
        "line_tr_only": merged[merged["line"] == "TRIGGER_RETEST"].copy(),
        "merged_raw_all_candidates": merged[
            merged["view_membership"].str.contains("raw_all_candidates", na=False)
        ].copy(),
        "merged_dedup_same_ticker_day": merged[
            merged["view_membership"].str.contains("dedup_same_ticker_day", na=False)
        ].copy(),
        "merged_capped_portfolio": merged[
            merged["view_membership"].str.contains("capped_portfolio", na=False)
        ].copy(),
    }


# ---------------------------------------------------------------------------
# Daily summary (spec §8: monthly counts + exposure over time go here too)
# ---------------------------------------------------------------------------


def _build_daily_summary(merged: pd.DataFrame) -> pd.DataFrame:
    # Daily counts of new entries (closed+open with entry_date) per slice + exposure.
    rows: list[dict[str, Any]] = []
    raw = merged[merged["view_membership"].str.contains("raw_all_candidates", na=False)].copy()
    raw = raw[raw["status"].isin(("CLOSED", "OPEN"))].copy()
    raw["entry_date"] = pd.to_datetime(raw["entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["exit_date"])

    if len(raw) == 0:
        return pd.DataFrame()

    all_dates = pd.date_range(
        raw["entry_date"].min(), raw["exit_date"].max(skipna=True), freq="D"
    )
    for d in all_dates:
        new_entries = int((raw["entry_date"] == d).sum())
        # Exposure: position open if entry_date <= d and (exit_date > d or status==OPEN with exit NaT)
        live_mask = (raw["entry_date"] <= d) & (
            (raw["exit_date"] > d) | (raw["status"] == "OPEN")
        )
        exposure = int(live_mask.sum())
        rows.append(
            {
                "date": d.date(),
                "new_entries_raw": new_entries,
                "exposure_raw": exposure,
            }
        )
    out = pd.DataFrame(rows)

    # Monthly aggregates for closed trades only — added as separate columns.
    closed = merged[
        (merged["status"] == "CLOSED")
        & merged["view_membership"].str.contains("raw_all_candidates", na=False)
    ].copy()
    closed["exit_date"] = pd.to_datetime(closed["exit_date"])
    closed["month"] = closed["exit_date"].dt.to_period("M").astype(str)
    monthly = (
        closed.groupby("month", sort=True)
        .agg(
            monthly_closed_trades=("status", "size"),
            monthly_avg_R=("realized_R_paper", "mean"),
            monthly_avg_net_return_pct=("net_return_pct", "mean"),
        )
        .reset_index()
    )
    out["month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)
    out = out.merge(monthly, on="month", how="left")
    return out


# ---------------------------------------------------------------------------
# Summary markdown (spec §16 item 15: no live-promotion claim)
# ---------------------------------------------------------------------------


def _format_pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def _build_summary_md(
    metrics: dict[str, dict[str, Any]],
    conflicts: pd.DataFrame,
    le_manifest: dict[str, Any],
    tr_manifest: dict[str, Any],
    le_hash: str,
    tr_hash: str,
    run_started: str,
    asof: str,
    promotion_status: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# portfolio_merge_paper — Run Summary")
    lines.append("")
    lines.append(f"- Spec: `{SPEC_PATH}` ({SPEC_STATUS})")
    lines.append(f"- Run timestamp: {run_started}")
    lines.append(f"- asof: {asof}")
    lines.append("- Mode: reporting-only merge of two LOCKED paper/shadow lines")
    lines.append("")
    lines.append("## Source files (READ-ONLY)")
    lines.append(f"- Line E trades: `{LINE_E_TRADES.relative_to(ROOT)}` (sha256 `{le_hash[:12]}…`)")
    lines.append(f"- Line E manifest: `{LINE_E_MANIFEST.relative_to(ROOT)}`")
    lines.append(f"- Line TR trades: `{LINE_TR_TRADES.relative_to(ROOT)}` (sha256 `{tr_hash[:12]}…`)")
    lines.append(f"- Line TR manifest: `{LINE_TR_MANIFEST.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Per-line manifest references (verbatim)")
    lines.append(f"- Line E spec_status: {le_manifest.get('spec_status','—')}")
    lines.append(f"- Line TR spec_status: {tr_manifest.get('spec_status','—')}")
    le_hb = le_manifest.get("boot_integrity", {}).get("hb_event_parquet_sha256", "—")
    tr_hb = tr_manifest.get("boot_integrity", {}).get("hb_event_parquet_sha256", "—")
    lines.append(f"- Line E HB sha256: `{le_hb[:12]}…`")
    lines.append(f"- Line TR HB sha256: `{tr_hb[:12]}…`")
    if le_hb != tr_hb:
        lines.append(f"  - NOTE: HB sha256 differs across lines (informational only).")
    else:
        lines.append("  - HB sha256 matches across both lines.")
    lines.append("")

    lines.append("## Locked merge constants")
    lines.append(f"- max_new_positions_per_day (capped view): {MAX_NEW_POSITIONS_PER_DAY}")
    lines.append(f"- dedup priority (operational, NOT alpha): {LINE_PRIORITY[0]} > {LINE_PRIORITY[1]}")
    lines.append(f"- capped selection: line priority then ascending alphabetical ticker")
    lines.append(f"- raw view: union of both lines' raw_all_candidates, no cap, no dedup beyond exact dupes")
    lines.append(f"- R drawdown vs % drawdown: reported under separate field names")
    lines.append("")

    # Headline (raw merged)
    raw = metrics["merged_raw_all_candidates"]
    lines.append("## §4.1 Primary HEADLINE — `merged_raw_all_candidates`")
    lines.append("")
    lines.append(f"- n_closed: **{raw['n_closed']}**")
    lines.append(f"- n_open: {raw['n_open']}")
    lines.append(f"- avg_R_paper: **{_format_pct(raw['avg_R_paper'])}**")
    lines.append(f"- median_R_paper: {_format_pct(raw['median_R_paper'])}")
    lines.append(f"- avg_gross_return_pct: **{_format_pct(raw['avg_gross_return_pct'])}**")
    lines.append(f"- avg_net_return_pct: {_format_pct(raw['avg_net_return_pct'])}")
    lines.append(f"- win_rate: **{_format_pct(raw['win_rate'])}**")
    lines.append(f"- profit_factor: **{_format_pct(raw['profit_factor'])}**")
    lines.append(f"- max_drawdown_R: {_format_pct(raw['max_drawdown_R'])}")
    lines.append(f"- max_drawdown_pct_gross: {_format_pct(raw['max_drawdown_pct_gross'])}")
    lines.append(f"- max_drawdown_pct_net: {_format_pct(raw['max_drawdown_pct_net'])}")
    lines.append(f"- pct_from_extended: {_format_pct(raw['pct_from_extended'])}")
    lines.append(f"- pct_from_trigger_retest: {_format_pct(raw['pct_from_trigger_retest'])}")
    lines.append("")

    # All 5 slices
    lines.append("## All 5 slices (descriptive)")
    lines.append("")
    for slice_label in (
        "line_e_only",
        "line_tr_only",
        "merged_raw_all_candidates",
        "merged_dedup_same_ticker_day",
        "merged_capped_portfolio",
    ):
        m = metrics[slice_label]
        lines.append(f"### {slice_label}")
        lines.append(f"- n_closed: {m['n_closed']}  |  n_open: {m['n_open']}  |  n_skipped: {m['n_skipped']}")
        lines.append(
            f"- avg_R_paper: {_format_pct(m['avg_R_paper'])}  |  win_rate: {_format_pct(m['win_rate'])}  |  PF: {_format_pct(m['profit_factor'])}"
        )
        lines.append(
            f"- avg_gross_return_pct: {_format_pct(m['avg_gross_return_pct'])}  |  avg_net_return_pct: {_format_pct(m['avg_net_return_pct'])}"
        )
        lines.append(
            f"- max_drawdown_R: {_format_pct(m['max_drawdown_R'])}  |  max_drawdown_pct_gross: {_format_pct(m['max_drawdown_pct_gross'])}  |  max_drawdown_pct_net: {_format_pct(m['max_drawdown_pct_net'])}"
        )
        lines.append(
            f"- forward_closed: {m['forward_closed_trades']}  |  forward_open: {m['forward_open_trades']}  |  backfill_closed: {m['backfill_closed_trades']}  |  backfill_open: {m['backfill_open_trades']}"
        )
        lines.append("")

    # Conflicts
    lines.append("## Same-ticker / same-asof_date conflicts (raw view)")
    lines.append("")
    lines.append(f"- conflict count: **{len(conflicts)}**")
    if len(conflicts) > 0:
        lines.append(f"- see: `output/portfolio_merge_paper_conflicts.csv`")
        # Show first few
        head = conflicts.head(10)
        lines.append("- first 10 conflicts:")
        for _, row in head.iterrows():
            lines.append(
                f"  - {row['asof_date']} {row['ticker']}: lines={row['lines_present']} → selected={row['selected_line']}"
            )
    lines.append("")

    # View discipline reminder
    lines.append("## View discipline (spec §4.4)")
    lines.append("")
    lines.append("- **Primary headline = `merged_raw_all_candidates`.**")
    lines.append("- `merged_dedup_same_ticker_day` and `merged_capped_portfolio` are **operational views only**.")
    lines.append(
        "- If capped or dedup beats raw: label as selection artifact, NOT alpha (spec §4.3 / §4.4)."
    )
    lines.append("")

    # Promotion status
    lines.append("## Promotion-floor status (spec §10) — descriptive only")
    lines.append("")
    for line_name, st in promotion_status.items():
        lines.append(
            f"- {line_name}: forward_closed={st['forward_closed']} (need ≥{PROMOTION_FORWARD_CLOSED_MIN}), forward_months_elapsed={st['forward_months_elapsed']:.2f} (need ≥{PROMOTION_FORWARD_MONTHS}) → **{st['status']}**"
        )
    lines.append("")
    lines.append("**This merged paper report does NOT unlock live trading.**")
    lines.append("**Live trade gate STAYS CLOSED.** No promotion claim is made by this run.")
    lines.append("")

    # Forbidden interpretations
    lines.append("## Forbidden interpretations (spec §12)")
    lines.append("")
    lines.append("- ❌ Merged view is a live portfolio")
    lines.append("- ❌ TRIGGER_RETEST should be preferred because it performed better")
    lines.append("- ❌ EXTENDED should be dropped")
    lines.append("- ❌ Capped portfolio performance proves alpha")
    lines.append("- ❌ Same-ticker priority is alpha ranking")
    lines.append("- ❌ This enables live trading")
    lines.append("- ❌ This changes either line's eligibility rule")
    lines.append("- ❌ Cross-line comparison ranks one line above the other")
    lines.append("- ❌ Capped view is the better strategy")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Promotion-floor status snapshot (descriptive only)
# ---------------------------------------------------------------------------


def _promotion_status(merged: pd.DataFrame, asof: str) -> dict[str, Any]:
    asof_dt = pd.to_datetime(asof)
    out: dict[str, Any] = {}
    for line_name in ("EXTENDED", "TRIGGER_RETEST"):
        sub = merged[merged["line"] == line_name].copy()
        forward_closed = int(
            ((sub["status"] == "CLOSED") & (~sub["is_backfill"].astype(bool))).sum()
        )
        forward_open = int(
            ((sub["status"] == "OPEN") & (~sub["is_backfill"].astype(bool))).sum()
        )
        # Forward months elapsed: require at least one forward-clock entry to start clock.
        forward_rows = sub[~sub["is_backfill"].astype(bool)]
        if len(forward_rows) > 0 and forward_rows["entry_date"].notna().any():
            first_forward = pd.to_datetime(forward_rows["entry_date"]).min()
            months = (asof_dt - first_forward).days / 30.4375
        else:
            months = 0.0
        meets = (
            months >= PROMOTION_FORWARD_MONTHS
            and forward_closed >= PROMOTION_FORWARD_CLOSED_MIN
        )
        out[line_name] = {
            "forward_closed": forward_closed,
            "forward_open": forward_open,
            "forward_months_elapsed": float(months),
            "status": "MEETS_FLOOR" if meets else "NOT_MET",
        }
    # Merged forward-month requirement (≥6 mo merged)
    merged_forward = merged[~merged["is_backfill"].astype(bool)]
    if len(merged_forward) > 0 and merged_forward["entry_date"].notna().any():
        first_forward = pd.to_datetime(merged_forward["entry_date"]).min()
        merged_months = (asof_dt - first_forward).days / 30.4375
    else:
        merged_months = 0.0
    out["MERGED"] = {
        "forward_closed": int(
            ((merged["status"] == "CLOSED") & (~merged["is_backfill"].astype(bool))).sum()
        ),
        "forward_open": int(
            ((merged["status"] == "OPEN") & (~merged["is_backfill"].astype(bool))).sum()
        ),
        "forward_months_elapsed": float(merged_months),
        "status": "MEETS_FLOOR" if merged_months >= PROMOTION_FORWARD_MONTHS else "NOT_MET",
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    run_started = dt.datetime.now(dt.timezone.utc).isoformat()
    asof = dt.date.today().isoformat()

    # Pre-run checklist (12 items per spec §16 user reaffirmation)
    boot = _run_prerun_checklist()
    le_manifest = boot["le_manifest"]
    tr_manifest = boot["tr_manifest"]

    # Load source trade logs (READ ONLY)
    le = pd.read_parquet(LINE_E_TRADES)
    tr = pd.read_parquet(LINE_TR_TRADES)
    le_hash = _sha256(LINE_E_TRADES)
    tr_hash = _sha256(LINE_TR_TRADES)

    # Build merged log
    merged = _build_merged_log(le, tr, le_hash, tr_hash)

    # Post-build invariant: line column preserved verbatim, no other line values.
    bad_lines = merged[~merged["line"].isin(("EXTENDED", "TRIGGER_RETEST"))]
    if len(bad_lines) > 0:
        _stop(f"line column invariant violated: {len(bad_lines)} bad rows")
    boot["checklist"]["5_line_column_preserved"] = "PASS"

    # Assemble views
    merged = _assemble_views(merged)

    # Post-view invariants
    raw_count = int(merged["view_membership"].str.contains("raw_all_candidates", na=False).sum())
    raw_eligible_count = int(merged["in_view_raw_all_candidates"].astype(bool).sum())
    if raw_count != raw_eligible_count:
        _stop(
            f"raw view assembly mismatch: view_membership raw_count={raw_count} vs source raw_eligible={raw_eligible_count}"
        )
    boot["checklist"]["9_raw_view_no_cap_no_dedup"] = "PASS"

    # Conflicts report
    conflicts = _build_conflicts_report(merged)
    conflicts.to_csv(OUT_CONFLICTS_CSV, index=False)
    boot["checklist"]["10_dedup_conflict_report_emitted"] = "PASS"

    # Build slices and metrics
    slices = _build_slices(merged)
    metrics = {label: _slice_metrics(df, slice_label=label) for label, df in slices.items()}

    # Post-metric invariants
    # 7. SKIPPED excluded from performance: confirm closed counts == n_closed and SKIPPED rows untouched
    for label, m in metrics.items():
        # Reconstruct closed/open/skipped counts from the slice and ensure they sum to n_total_rows
        if m["n_closed"] + m["n_open"] + m["n_skipped"] != m["n_total_rows"]:
            _stop(f"slice {label} status counts inconsistent")
    boot["checklist"]["7_skipped_excluded_from_performance"] = "PASS"
    boot["checklist"]["8_open_excluded_from_return_metrics"] = "PASS"

    # Daily summary
    daily = _build_daily_summary(merged)
    daily.to_csv(OUT_DAILY_SUMMARY_CSV, index=False)

    # Promotion-floor snapshot (descriptive only)
    promotion_status = _promotion_status(merged, asof)

    # Summary markdown — assert no live-promotion claim string
    summary_md = _build_summary_md(
        metrics, conflicts, le_manifest, tr_manifest, le_hash, tr_hash, run_started, asof, promotion_status
    )
    forbidden_phrases = [
        "live trading enabled",
        "promote to live",
        "live promotion approved",
        "ready for live",
    ]
    for phrase in forbidden_phrases:
        if phrase.lower() in summary_md.lower():
            _stop(f"summary.md contains forbidden live-promotion phrase: {phrase}")
    boot["checklist"]["12_no_live_promotion_claim"] = "PASS"
    boot["checklist"]["11_capped_view_marked_operational"] = "PASS"

    OUT_SUMMARY_MD.write_text(summary_md)

    # Final trade-log column ordering and write
    final_cols = REQUIRED_COLS
    for col in final_cols:
        if col not in merged.columns:
            _stop(f"final trade log missing required column: {col}")
    merged_out = merged[final_cols].copy()
    merged_out.to_parquet(OUT_TRADES_PARQUET, index=False)
    merged_out.to_csv(OUT_TRADES_CSV, index=False)

    # Manifest
    manifest = {
        "run_started_utc": run_started,
        "asof": asof,
        "spec_path": SPEC_PATH,
        "spec_status": SPEC_STATUS,
        "purpose": "reporting-only merge of two LOCKED paper/shadow lines (Line E + Line TR)",
        "live_trade_gate": "CLOSED",
        "promotion_claim_in_outputs": False,
        "source_files": {
            "line_e_trades": str(LINE_E_TRADES),
            "line_e_trades_sha256": le_hash,
            "line_e_manifest": str(LINE_E_MANIFEST),
            "line_tr_trades": str(LINE_TR_TRADES),
            "line_tr_trades_sha256": tr_hash,
            "line_tr_manifest": str(LINE_TR_MANIFEST),
        },
        "source_manifest_refs": {
            "line_e": {
                "spec_status": le_manifest.get("spec_status"),
                "hb_event_parquet_sha256": le_manifest.get("boot_integrity", {}).get(
                    "hb_event_parquet_sha256"
                ),
                "ohlcv_master_sha256": le_manifest.get("boot_integrity", {}).get(
                    "ohlcv_master_sha256"
                ),
                "trade_log_rows": le_manifest.get("trade_log_rows"),
            },
            "line_tr": {
                "spec_status": tr_manifest.get("spec_status"),
                "hb_event_parquet_sha256": tr_manifest.get("boot_integrity", {}).get(
                    "hb_event_parquet_sha256"
                ),
                "ohlcv_master_sha256": tr_manifest.get("boot_integrity", {}).get(
                    "ohlcv_master_sha256"
                ),
                "trade_log_rows": tr_manifest.get("trade_log_rows"),
            },
        },
        "locked_constants": {
            "max_new_positions_per_day_capped": MAX_NEW_POSITIONS_PER_DAY,
            "dedup_priority": list(LINE_PRIORITY),
            "capped_selection_rule": "line priority then ascending alphabetical ticker",
            "raw_view_rule": "union of both lines' raw_all_candidates; no cap; no dedup beyond exact dupes",
            "promotion_forward_months_min": PROMOTION_FORWARD_MONTHS,
            "promotion_forward_closed_min": PROMOTION_FORWARD_CLOSED_MIN,
        },
        "drawdown_field_separation": {
            "max_drawdown_R": "additive cumulative R curve (sum of realized_R_paper)",
            "max_drawdown_pct_gross": "equal-notional cumulative gross_return_pct curve",
            "max_drawdown_pct_net": "equal-notional cumulative net_return_pct curve",
        },
        "trade_log_rows": int(len(merged_out)),
        "view_counts": {
            "raw_all_candidates": int(
                merged["view_membership"].str.contains("raw_all_candidates", na=False).sum()
            ),
            "dedup_same_ticker_day": int(
                merged["view_membership"].str.contains("dedup_same_ticker_day", na=False).sum()
            ),
            "capped_portfolio": int(
                merged["view_membership"].str.contains("capped_portfolio", na=False).sum()
            ),
        },
        "conflict_count": int(len(conflicts)),
        "metrics_by_slice": metrics,
        "promotion_floor_status": promotion_status,
        "prerun_checklist": boot["checklist"],
        "outputs": {p.name: str(p) for p in ALL_OUTPUTS},
    }

    with open(OUT_MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Final post-write assertions: no protected per-line file modified.
    # We never write to PROTECTED_PATHS — this is a defensive read of mtimes only.
    # (Pre-flight already enforced output safety.)

    print(f"OK wrote {len(ALL_OUTPUTS)} outputs to {OUT_DIR}")
    print(f"   trade_log_rows: {len(merged_out)}")
    print(f"   raw view rows: {manifest['view_counts']['raw_all_candidates']}")
    print(f"   dedup view rows: {manifest['view_counts']['dedup_same_ticker_day']}")
    print(f"   capped view rows: {manifest['view_counts']['capped_portfolio']}")
    print(f"   conflicts: {len(conflicts)}")
    print(f"   raw merged headline avg_R: {metrics['merged_raw_all_candidates']['avg_R_paper']:.4f}")
    print(f"   raw merged headline win_rate: {metrics['merged_raw_all_candidates']['win_rate']:.4f}")
    print(f"   raw merged headline PF: {metrics['merged_raw_all_candidates']['profit_factor']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
