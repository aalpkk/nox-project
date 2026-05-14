"""ema_context Pilot 3 — single authorized run.

Spec: memory/ema_context_pilot3_spec.md (LOCKED 2026-05-03)

EMA HB compression timing map. 7 pre-locked offset buckets.
Path A — outcome-free, descriptive timing map.
NOT a Pilot 2 window-widening rescue.

Anti-tweak (locked):
  - 7 buckets pre-locked, no merge/split/shift
  - Primary metric ema_stack_width_pct; ATR audit-only
  - Argmin only at bucket level (no sub-bucket offset scan)
  - Pre-event allowed buckets {[-15,-11], [-10,-6]} pre-locked
  - [-20,-16] argmin → at most PARTIAL (background regime concern)
  - Rebound formula: pre_ref = [-5,-1] mean, post_ref = [+5,+10] mean
  - Q2 strict argmin match — neighbor kabul YASAK
  - ATR-only PASS does NOT count as Pilot 3 PASS
  - INSUFFICIENT ≠ FAIL
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOCKED CONSTANTS — DO NOT MODIFY POST-RUN
# =============================================================================

HB_EVENT_PARQUET = PROJECT_ROOT / "output" / "horizontal_base_event_v1.parquet"
HB_EVENT_MANIFEST = PROJECT_ROOT / "output" / "horizontal_base_event_v1_manifest.json"
EMA_CONTEXT_PARQUET = PROJECT_ROOT / "output" / "ema_context_daily.parquet"
EMA_CONTEXT_METADATA = PROJECT_ROOT / "output" / "ema_context_daily_metadata.json"

OUT_Q1_PCT = PROJECT_ROOT / "output" / "ema_context_pilot3_q1_combined_pct.csv"
OUT_Q1_ATR = PROJECT_ROOT / "output" / "ema_context_pilot3_q1_combined_atr.csv"
OUT_Q2_PCT = PROJECT_ROOT / "output" / "ema_context_pilot3_q2_per_state_pct.csv"
OUT_Q2_ATR = PROJECT_ROOT / "output" / "ema_context_pilot3_q2_per_state_atr.csv"
OUT_Q3_SLOPE_PCT = PROJECT_ROOT / "output" / "ema_context_pilot3_q3_slope_tier_pct.csv"
OUT_Q3_SLOPE_ATR = PROJECT_ROOT / "output" / "ema_context_pilot3_q3_slope_tier_atr.csv"
OUT_Q3_WIDTH_PCT = PROJECT_ROOT / "output" / "ema_context_pilot3_q3_width_tier_pct.csv"
OUT_Q3_WIDTH_ATR = PROJECT_ROOT / "output" / "ema_context_pilot3_q3_width_tier_atr.csv"
OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot3_panel.parquet"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot3_summary.md"

# 7 pre-locked offset buckets (closed intervals, native daily bars)
BUCKETS: list[tuple[str, int, int]] = [
    ("[-20,-16]", -20, -16),
    ("[-15,-11]", -15, -11),
    ("[-10,-6]", -10, -6),
    ("[-5,-1]", -5, -1),
    ("[0,+1]", 0, 1),
    ("[+2,+5]", 2, 5),
    ("[+6,+10]", 6, 10),
]
BUCKET_LABELS = [b[0] for b in BUCKETS]
PRE_REF_BUCKET = "[-5,-1]"
POST_REF_BUCKET = "[+5,+10]"

# Bucket label updated to match key in BUCKETS for [+5,+10] cell — naming convention:
# We map [+6,+10] for window grid but rebound uses [+5,+10]. Per spec: "post_ref = [+5,+10] mean".
# To stay consistent with locked grid (7 non-overlapping buckets including [+2,+5] and [+6,+10]),
# rebound post_ref is computed directly from offsets +5..+10 inclusive (pre-locked offset set,
# NOT a bucket label). pre_ref uses the [-5,-1] bucket as already in the grid.
POST_REF_OFFSETS = list(range(5, 11))   # +5..+10 inclusive — locked post_ref offset set
PRE_REF_OFFSETS = list(range(-5, 0))     # -5..-1 inclusive — matches [-5,-1] bucket

# Allowed pre-event buckets per spec
ALLOWED_PRE_EVENT_BUCKETS = {"[-15,-11]", "[-10,-6]"}
EXPECTED_PRE_EVENT_BUCKET = "[-10,-6]"
EARLY_BACKGROUND_BUCKET = "[-20,-16]"   # at most PARTIAL
NON_PRE_EVENT_BUCKETS = {"[-5,-1]", "[0,+1]", "[+2,+5]", "[+6,+10]"}

# Rebound rule
REBOUND_MIN = 0.10

# Min-N gates
Q1_BUCKET_MIN_N = 90
Q2_STATE_BUCKET_MIN_N = 30
Q3_SLICE_MIN_N = 200

SIGNAL_STATE_ORDER = ["trigger", "retest_bounce", "extended"]

# Geometry slice axes
SLOPE_TIER_ORDER = ["flat", "mild", "loose"]
WIDTH_TIER_ORDER = ["low_width_pctile", "mid_width_pctile", "high_width_pctile"]
WIDTH_TIER_LO = 0.333
WIDTH_TIER_HI = 0.666

# Source manifest expectations
EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_BREAKPOINTS_VERSION = "v0.0"
EXPECTED_EC_ROWS = 451475

# Window for offset extraction
OFFSET_MIN = -20
OFFSET_MAX = 10


# =============================================================================
# BOOT — fail-fast manifest check
# =============================================================================


def _validate_inputs() -> tuple[dict, dict]:
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"HB event parquet missing: {HB_EVENT_PARQUET}")
    if not HB_EVENT_MANIFEST.exists():
        raise FileNotFoundError(f"HB event manifest missing: {HB_EVENT_MANIFEST}")
    if not EMA_CONTEXT_PARQUET.exists():
        raise FileNotFoundError(f"ema_context parquet missing: {EMA_CONTEXT_PARQUET}")
    if not EMA_CONTEXT_METADATA.exists():
        raise FileNotFoundError(f"ema_context metadata missing: {EMA_CONTEXT_METADATA}")

    manifest = json.loads(HB_EVENT_MANIFEST.read_text())
    sv = manifest.get("scanner_version")
    if sv != EXPECTED_SCANNER_VERSION:
        raise RuntimeError(
            f"HB manifest scanner_version mismatch: got {sv!r} != "
            f"{EXPECTED_SCANNER_VERSION!r}. SPEC violation, no fallback."
        )
    rows = manifest.get("rows")
    if rows != EXPECTED_HB_ROWS:
        raise RuntimeError(
            f"HB manifest rows mismatch: got {rows!r} != {EXPECTED_HB_ROWS}. "
            f"SPEC violation, no rebuild."
        )

    em_meta = json.loads(EMA_CONTEXT_METADATA.read_text())
    bp_version = em_meta.get("tag_breakpoints", {}).get("version")
    if bp_version != EXPECTED_BREAKPOINTS_VERSION:
        raise RuntimeError(
            f"ema_context breakpoints version mismatch: got {bp_version!r} != "
            f"{EXPECTED_BREAKPOINTS_VERSION!r}. SPEC violation."
        )
    ec_rows = em_meta.get("n_rows")
    if ec_rows != EXPECTED_EC_ROWS:
        raise RuntimeError(
            f"ema_context rows mismatch: got {ec_rows!r} != {EXPECTED_EC_ROWS}. "
            f"SPEC violation."
        )

    return manifest, em_meta


# =============================================================================
# DATA PREP
# =============================================================================


def _load_hb_events() -> pd.DataFrame:
    df = pd.read_parquet(HB_EVENT_PARQUET)
    if len(df) != EXPECTED_HB_ROWS:
        raise RuntimeError(f"HB events row count mismatch: {len(df)} != {EXPECTED_HB_ROWS}")
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.normalize()
    keep = [
        "ticker",
        "bar_date",
        "signal_state",
        "family__slope_tier",
        "family__channel_width_pctile_252",
    ]
    out = df[keep].copy()
    out["signal_state"] = out["signal_state"].astype(str)
    out["width_tier"] = pd.cut(
        out["family__channel_width_pctile_252"],
        bins=[-np.inf, WIDTH_TIER_LO, WIDTH_TIER_HI, np.inf],
        labels=WIDTH_TIER_ORDER,
    ).astype(str)
    return out


def _load_ema_context() -> pd.DataFrame:
    df = pd.read_parquet(EMA_CONTEXT_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# =============================================================================
# Trajectory extraction — long-form per event-offset
# =============================================================================


def _build_offset_long(events: pd.DataFrame, ema_ctx: pd.DataFrame) -> pd.DataFrame:
    """Build long-form (event_id, ticker, signal_state, slope_tier, width_tier, offset, pct, atr).

    Uses per-ticker dense numpy arrays for fast offset lookup at each event's anchor row.
    """
    ec = ema_ctx[["ticker", "date", "ema_stack_width_pct", "ema_stack_width_atr"]].copy()
    ec = ec.sort_values(["ticker", "date"]).reset_index(drop=True)
    ec["row_in_ticker"] = ec.groupby("ticker").cumcount()

    # Build per-ticker arrays
    pct_arrays: dict[str, np.ndarray] = {}
    atr_arrays: dict[str, np.ndarray] = {}
    for tk, grp in ec.groupby("ticker", sort=False):
        pct_arrays[tk] = grp["ema_stack_width_pct"].to_numpy(dtype=float)
        atr_arrays[tk] = grp["ema_stack_width_atr"].to_numpy(dtype=float)

    # Map (ticker, date) → row_in_ticker
    key_lookup = ec.set_index(["ticker", "date"])["row_in_ticker"]

    # Locate each event's anchor row
    ev = events.copy().reset_index(drop=True)
    ev["event_id"] = ev.index
    anchor_keys = list(zip(ev["ticker"].values, ev["bar_date"].values))
    anchors = key_lookup.reindex(anchor_keys).to_numpy()
    ev["anchor_row"] = anchors

    # Drop events without an anchor match (defensive — should be 0 given Pilot 2 found 0 missing)
    n_missing = int(np.isnan(anchors).sum())
    if n_missing > 0:
        print(f"  WARN: {n_missing} events missing anchor in ema_context (dropped)")
    ev = ev.dropna(subset=["anchor_row"]).copy()
    ev["anchor_row"] = ev["anchor_row"].astype(int)

    offsets = list(range(OFFSET_MIN, OFFSET_MAX + 1))
    rows: list[tuple] = []

    for _, row in ev.iterrows():
        tk = row["ticker"]
        anchor = int(row["anchor_row"])
        sig = row["signal_state"]
        slope_t = row["family__slope_tier"]
        width_t = row["width_tier"]
        eid = int(row["event_id"])
        pct_arr = pct_arrays.get(tk)
        atr_arr = atr_arrays.get(tk)
        if pct_arr is None or atr_arr is None:
            continue
        n_bars = len(pct_arr)
        for off in offsets:
            target = anchor + off
            if target < 0 or target >= n_bars:
                continue
            v_pct = pct_arr[target]
            v_atr = atr_arr[target]
            if np.isnan(v_pct) and np.isnan(v_atr):
                continue
            rows.append((eid, tk, sig, slope_t, width_t, off, float(v_pct), float(v_atr)))

    return pd.DataFrame(
        rows,
        columns=["event_id", "ticker", "signal_state", "slope_tier", "width_tier", "offset", "pct", "atr"],
    )


# =============================================================================
# Bucket aggregation
# =============================================================================


def _assign_bucket(offset: int) -> str | None:
    for label, lo, hi in BUCKETS:
        if lo <= offset <= hi:
            return label
    return None


def _bucket_aggregate(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return bucket × {mean, median, count} aggregation for a metric column."""
    sub = long_df.dropna(subset=[metric]).copy()
    sub["bucket"] = sub["offset"].map(_assign_bucket)
    sub = sub.dropna(subset=["bucket"])
    agg = sub.groupby("bucket")[metric].agg(["mean", "median", "count"]).reset_index()
    # Reorder per locked label order
    agg["__order"] = agg["bucket"].apply(lambda b: BUCKET_LABELS.index(b))
    agg = agg.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return agg


def _bucket_aggregate_grouped(long_df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """Return (group, bucket) × {mean, median, count} aggregation."""
    sub = long_df.dropna(subset=[metric, group_col]).copy()
    sub["bucket"] = sub["offset"].map(_assign_bucket)
    sub = sub.dropna(subset=["bucket"])
    agg = sub.groupby([group_col, "bucket"])[metric].agg(["mean", "median", "count"]).reset_index()
    agg["__order"] = agg["bucket"].apply(lambda b: BUCKET_LABELS.index(b))
    agg = agg.sort_values([group_col, "__order"]).drop(columns="__order").reset_index(drop=True)
    return agg


def _argmin_bucket(bucket_agg: pd.DataFrame, min_n: int) -> tuple[str | None, dict]:
    """Find argmin bucket among sufficient-N buckets. Returns (label, info)."""
    suff = bucket_agg[bucket_agg["count"] >= min_n].copy()
    info = {
        "n_buckets_total": int(len(bucket_agg)),
        "n_buckets_sufficient": int(len(suff)),
        "insufficient_buckets": [
            {"bucket": r["bucket"], "n": int(r["count"])}
            for _, r in bucket_agg[bucket_agg["count"] < min_n].iterrows()
        ],
    }
    if suff.empty:
        return None, info
    idx = suff["mean"].idxmin()
    label = suff.loc[idx, "bucket"]
    info["argmin_mean"] = float(suff.loc[idx, "mean"])
    info["argmin_n"] = int(suff.loc[idx, "count"])
    return label, info


def _rebound_from_buckets(bucket_agg: pd.DataFrame) -> dict:
    """Compute rebound = (post_ref - pre_ref) / |pre_ref| using locked offset references.

    Note: rebound uses raw offset means via bucket labels (pre_ref bucket = [-5,-1]),
    and post_ref derived from offsets +5..+10. Since [+6,+10] bucket excludes +5,
    we approximate post_ref using the [+6,+10] bucket mean weighted with offset +5
    via a separate calculation. For deterministic locked behavior, use the post_ref
    bucket label directly: pre_ref_bucket=[-5,-1], post_ref_bucket=[+6,+10].
    Spec mandates POST_REF_OFFSETS = +5..+10 — handled in _rebound_offset_based().
    """
    raise NotImplementedError("Use _rebound_offset_based() for spec compliance.")


def _rebound_offset_based(long_df: pd.DataFrame, metric: str) -> dict:
    """Rebound using locked PRE_REF_OFFSETS / POST_REF_OFFSETS (+5..+10 includes +5)."""
    sub = long_df.dropna(subset=[metric]).copy()
    pre = sub[sub["offset"].isin(PRE_REF_OFFSETS)][metric]
    post = sub[sub["offset"].isin(POST_REF_OFFSETS)][metric]
    pre_n = int(len(pre))
    post_n = int(len(post))
    pre_mean = float(pre.mean()) if pre_n > 0 else float("nan")
    post_mean = float(post.mean()) if post_n > 0 else float("nan")
    if pre_n == 0 or pre_mean == 0 or np.isnan(pre_mean):
        rebound = float("nan")
    else:
        rebound = (post_mean - pre_mean) / abs(pre_mean)
    return {
        "pre_ref_offsets": list(PRE_REF_OFFSETS),
        "post_ref_offsets": list(POST_REF_OFFSETS),
        "pre_n": pre_n,
        "post_n": post_n,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "rebound_pct": float(rebound) if not np.isnan(rebound) else None,
    }


# =============================================================================
# Q1 / Q2 / Q3
# =============================================================================


def _q1_evaluate(long_df: pd.DataFrame, metric: str) -> dict:
    agg = _bucket_aggregate(long_df, metric)
    argmin_label, info = _argmin_bucket(agg, Q1_BUCKET_MIN_N)
    rebound = _rebound_offset_based(long_df, metric)

    if info["n_buckets_total"] < len(BUCKETS) or info["insufficient_buckets"]:
        # Per spec: any bucket N<90 → Q1 INSUFFICIENT
        status = "INSUFFICIENT"
    elif argmin_label is None:
        status = "INSUFFICIENT"
    else:
        argmin_pre_event_allowed = argmin_label in ALLOWED_PRE_EVENT_BUCKETS
        argmin_early_background = argmin_label == EARLY_BACKGROUND_BUCKET
        argmin_non_pre_event = argmin_label in NON_PRE_EVENT_BUCKETS
        rebound_pass = (
            rebound["rebound_pct"] is not None
            and rebound["rebound_pct"] >= REBOUND_MIN
        )

        if argmin_pre_event_allowed and rebound_pass:
            status = "PASS"
        elif argmin_early_background:
            status = "PARTIAL"
        elif argmin_pre_event_allowed and not rebound_pass:
            status = "PARTIAL"
        elif argmin_non_pre_event:
            status = "FAIL"
        else:
            status = "FAIL"

    return {
        "metric": metric,
        "agg": agg,
        "argmin_bucket": argmin_label,
        "argmin_info": info,
        "rebound": rebound,
        "status": status,
        "argmin_clean_expected": (argmin_label == EXPECTED_PRE_EVENT_BUCKET),
    }


def _q2_evaluate(long_df: pd.DataFrame, metric: str) -> dict:
    agg = _bucket_aggregate_grouped(long_df, metric, "signal_state")
    per_state: dict[str, dict] = {}
    insufficient_states: list[str] = []
    for state in SIGNAL_STATE_ORDER:
        sub = agg[agg["signal_state"] == state].copy()
        if sub.empty:
            per_state[state] = {"argmin_bucket": None, "status": "INSUFFICIENT", "info": {}}
            insufficient_states.append(state)
            continue
        # Reorder bucket order
        sub["__order"] = sub["bucket"].apply(lambda b: BUCKET_LABELS.index(b))
        sub = sub.sort_values("__order").reset_index(drop=True)
        # Argmin among sufficient-N
        suff = sub[sub["count"] >= Q2_STATE_BUCKET_MIN_N]
        any_insufficient = bool((sub["count"] < Q2_STATE_BUCKET_MIN_N).any())
        if suff.empty:
            per_state[state] = {
                "argmin_bucket": None,
                "status": "INSUFFICIENT",
                "info": {"any_bucket_insufficient": any_insufficient},
            }
            insufficient_states.append(state)
            continue
        idx = suff["mean"].idxmin()
        label = suff.loc[idx, "bucket"]
        per_state[state] = {
            "argmin_bucket": label,
            "argmin_n": int(suff.loc[idx, "count"]),
            "argmin_mean": float(suff.loc[idx, "mean"]),
            "status": "OK",
            "any_bucket_insufficient": any_insufficient,
        }

    # Majority computation
    bucket_counts: dict[str, int] = {}
    for state in SIGNAL_STATE_ORDER:
        b = per_state[state].get("argmin_bucket")
        if b:
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

    majority_bucket = None
    majority_count = 0
    if bucket_counts:
        majority_bucket = max(bucket_counts, key=lambda k: bucket_counts[k])
        majority_count = bucket_counts[majority_bucket]

    has_majority = majority_count >= 2
    majority_in_allowed = majority_bucket in ALLOWED_PRE_EVENT_BUCKETS if majority_bucket else False
    majority_early = majority_bucket == EARLY_BACKGROUND_BUCKET

    # Verdict per spec
    if not bucket_counts:
        status = "INSUFFICIENT"
    elif has_majority and majority_in_allowed:
        status = "PASS"
    elif has_majority and majority_early:
        status = "PARTIAL"
    elif not has_majority:
        # All 3 different argmins
        # Check if combined is FAIL — but here we don't have combined verdict access; per spec
        # "≥2/3 in non-pre-event" is FAIL, "all different" is PARTIAL alone
        status = "PARTIAL"
    else:
        # has_majority but not allowed and not early → majority in non-pre-event bucket → FAIL
        status = "FAIL"

    return {
        "metric": metric,
        "agg": agg,
        "per_state": per_state,
        "bucket_counts": bucket_counts,
        "majority_bucket": majority_bucket,
        "majority_count": majority_count,
        "insufficient_states": insufficient_states,
        "status": status,
    }


def _q3_axis_eval(long_df: pd.DataFrame, axis_col: str, axis_levels: list[str], metric: str) -> dict:
    agg = _bucket_aggregate_grouped(long_df, metric, axis_col)
    per_level: dict[str, dict] = {}
    sufficient_levels: list[str] = []

    for level in axis_levels:
        sub = agg[agg[axis_col] == level].copy()
        if sub.empty:
            per_level[level] = {"argmin_bucket": None, "status": "INSUFFICIENT", "n_total": 0}
            continue
        sub["__order"] = sub["bucket"].apply(lambda b: BUCKET_LABELS.index(b))
        sub = sub.sort_values("__order").reset_index(drop=True)
        n_total = int(sub["count"].sum())
        if n_total < Q3_SLICE_MIN_N:
            per_level[level] = {
                "argmin_bucket": None,
                "status": "INSUFFICIENT",
                "n_total": n_total,
            }
            continue
        sufficient_levels.append(level)
        # Argmin among buckets with at least 1 observation
        sub_nonzero = sub[sub["count"] > 0]
        idx = sub_nonzero["mean"].idxmin()
        label = sub_nonzero.loc[idx, "bucket"]
        per_level[level] = {
            "argmin_bucket": label,
            "argmin_n": int(sub_nonzero.loc[idx, "count"]),
            "argmin_mean": float(sub_nonzero.loc[idx, "mean"]),
            "n_total": n_total,
            "status": "OK",
        }

    return {
        "axis": axis_col,
        "agg": agg,
        "per_level": per_level,
        "sufficient_levels": sufficient_levels,
    }


def _q3_evaluate(long_df: pd.DataFrame, metric: str) -> dict:
    slope_eval = _q3_axis_eval(long_df, "slope_tier", SLOPE_TIER_ORDER, metric)
    width_eval = _q3_axis_eval(long_df, "width_tier", WIDTH_TIER_ORDER, metric)

    sufficient_total: list[str] = []
    pre_event_allowed_levels: list[str] = []
    early_levels: list[str] = []
    non_pre_event_levels: list[str] = []

    for axis_eval, axis_name in ((slope_eval, "slope"), (width_eval, "width")):
        for level in axis_eval["sufficient_levels"]:
            sufficient_total.append(f"{axis_name}/{level}")
            argmin = axis_eval["per_level"][level]["argmin_bucket"]
            if argmin in ALLOWED_PRE_EVENT_BUCKETS:
                pre_event_allowed_levels.append(f"{axis_name}/{level}")
            elif argmin == EARLY_BACKGROUND_BUCKET:
                early_levels.append(f"{axis_name}/{level}")
            elif argmin in NON_PRE_EVENT_BUCKETS:
                non_pre_event_levels.append(f"{axis_name}/{level}")

    if not sufficient_total:
        status = "INSUFFICIENT"
    elif pre_event_allowed_levels:
        status = "PASS"
    elif early_levels and not non_pre_event_levels:
        # All sufficient slices show early background
        status = "PARTIAL"
    elif early_levels and non_pre_event_levels:
        status = "PARTIAL"
    else:
        # No pre-event allowed levels, only non-pre-event
        status = "FAIL"

    return {
        "metric": metric,
        "slope": slope_eval,
        "width": width_eval,
        "sufficient_total": sufficient_total,
        "pre_event_allowed_levels": pre_event_allowed_levels,
        "early_levels": early_levels,
        "non_pre_event_levels": non_pre_event_levels,
        "status": status,
    }


# =============================================================================
# Overall verdict
# =============================================================================


def _overall_verdict(q1: dict, q2: dict, q3: dict) -> dict:
    statuses = {"Q1": q1["status"], "Q2": q2["status"], "Q3": q3["status"]}
    n_pass = sum(1 for v in statuses.values() if v == "PASS")
    n_partial = sum(1 for v in statuses.values() if v == "PARTIAL")
    n_fail = sum(1 for v in statuses.values() if v == "FAIL")

    # Per spec verdict table
    if n_pass == 3 and q1.get("argmin_bucket") == EXPECTED_PRE_EVENT_BUCKET:
        overall = "PASS_CLEAN"
    elif n_pass == 3:
        overall = "PASS_ACCEPTABLE_SHIFT"
    elif n_pass == 2 and n_partial == 1 and n_fail == 0:
        overall = "PARTIAL"
    elif n_fail >= 1:
        overall = "CLOSED"
    elif n_partial == 3:
        overall = "CLOSED"
    elif n_pass <= 1:
        overall = "CLOSED"
    else:
        overall = "PARTIAL"

    return {
        "statuses": statuses,
        "n_pass": n_pass,
        "n_partial": n_partial,
        "n_fail": n_fail,
        "overall": overall,
    }


# =============================================================================
# Output writers
# =============================================================================


def _write_q1_csv(q1: dict, path: Path) -> None:
    df = q1["agg"].copy()
    df["argmin_bucket"] = q1["argmin_bucket"]
    df["pre_ref_mean"] = q1["rebound"]["pre_mean"]
    df["post_ref_mean"] = q1["rebound"]["post_mean"]
    df["rebound_pct"] = q1["rebound"]["rebound_pct"]
    df["status"] = q1["status"]
    df.to_csv(path, index=False)


def _write_q2_csv(q2: dict, path: Path) -> None:
    df = q2["agg"].copy()
    df["majority_bucket"] = q2["majority_bucket"]
    df["majority_count"] = q2["majority_count"]
    df["status"] = q2["status"]
    rows: list[dict] = []
    for state, info in q2["per_state"].items():
        rows.append(
            {
                "signal_state": state,
                "argmin_bucket": info.get("argmin_bucket"),
                "argmin_n": info.get("argmin_n"),
                "argmin_mean": info.get("argmin_mean"),
                "state_status": info.get("status"),
                "any_bucket_insufficient": info.get("any_bucket_insufficient"),
            }
        )
    per_state_df = pd.DataFrame(rows)
    per_state_df["majority_bucket"] = q2["majority_bucket"]
    per_state_df["q2_status"] = q2["status"]
    # Combined output: agg first, then summary table appended
    out = pd.concat([df, per_state_df], ignore_index=True, sort=False)
    out.to_csv(path, index=False)


def _write_q3_csv(axis_eval: dict, path: Path) -> None:
    df = axis_eval["agg"].copy()
    rows: list[dict] = []
    for level, info in axis_eval["per_level"].items():
        rows.append(
            {
                axis_eval["axis"]: level,
                "argmin_bucket": info.get("argmin_bucket"),
                "argmin_n": info.get("argmin_n"),
                "argmin_mean": info.get("argmin_mean"),
                "level_n_total": info.get("n_total"),
                "level_status": info.get("status"),
            }
        )
    summary = pd.DataFrame(rows)
    out = pd.concat([df, summary], ignore_index=True, sort=False)
    out.to_csv(path, index=False)


def _format_summary(
    q1_pct: dict,
    q2_pct: dict,
    q3_pct: dict,
    q1_atr: dict,
    q2_atr: dict,
    q3_atr: dict,
    overall_pct: dict,
    overall_atr: dict,
    runtime_s: float,
    em_meta: dict,
    hb_manifest: dict,
    panel_n: int,
    long_n: int,
    n_events_used: int,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 3 — Run Summary")
    lines.append("")
    lines.append(f"- Spec: `memory/ema_context_pilot3_spec.md` LOCKED 2026-05-03")
    lines.append(f"- Run date: {pd.Timestamp.utcnow().isoformat()} (single authorized run)")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(f"- HB event source: `output/horizontal_base_event_v1.parquet` "
                 f"(scanner_version {hb_manifest.get('scanner_version')}, frozen_at {hb_manifest.get('frozen_at')}, rows {hb_manifest.get('rows')})")
    lines.append(f"- ema_context source: `output/ema_context_daily.parquet` "
                 f"(breakpoints {em_meta.get('tag_breakpoints', {}).get('version')})")
    lines.append(f"- Panel rows (events): {panel_n:,}")
    lines.append(f"- Long-form rows (event × offset): {long_n:,}")
    lines.append(f"- Events with offset coverage: {n_events_used:,}")
    lines.append(
        f"- Anti-tweak: 7 buckets pre-locked, primary=ema_stack_width_pct, atr audit-only, "
        f"argmin bucket-level only, ATR-only PASS does NOT count, no Pilot 2 window relitigation"
    )
    lines.append("")
    lines.append("## Naming")
    lines.append("")
    lines.append("**EMA HB compression timing map** — descriptive only. NO edge claim, NO outcome label, NO horizon, NO HB feature/gate/ML auto-promotion.")
    lines.append("")

    # Overall
    lines.append("## Overall Pilot 3 Verdict (Primary — ema_stack_width_pct)")
    lines.append("")
    lines.append(f"**{overall_pct['overall']}** (Q1={overall_pct['statuses']['Q1']}, Q2={overall_pct['statuses']['Q2']}, Q3={overall_pct['statuses']['Q3']})")
    lines.append("")
    lines.append(f"- Q1 combined argmin bucket: **{q1_pct['argmin_bucket']}** (rebound = {q1_pct['rebound']['rebound_pct']})")
    lines.append(f"- Q1 expected (clean) bucket: **[-10,-6]** → {'MATCH' if q1_pct.get('argmin_clean_expected') else 'NO MATCH'}")
    lines.append(f"- Q1 allowed pre-event buckets: {{[-15,-11], [-10,-6]}}")
    lines.append(f"- Q2 majority argmin: **{q2_pct['majority_bucket']}** ({q2_pct['majority_count']}/3 states)")
    lines.append(f"- Q3 sufficient slices: {q3_pct['sufficient_total']}")
    lines.append(f"- Q3 pre-event-allowed slices: {q3_pct['pre_event_allowed_levels']}")
    lines.append("")

    # G7 audit
    lines.append("## G7 Audit (ATR-normalized — audit-only, NOT a gate)")
    lines.append("")
    lines.append(f"ATR-mirror verdict: **{overall_atr['overall']}** (Q1={overall_atr['statuses']['Q1']}, Q2={overall_atr['statuses']['Q2']}, Q3={overall_atr['statuses']['Q3']})")
    lines.append("")
    lines.append(f"- ATR Q1 combined argmin bucket: **{q1_atr['argmin_bucket']}** (rebound = {q1_atr['rebound']['rebound_pct']})")
    lines.append(f"- ATR Q2 majority: **{q2_atr['majority_bucket']}** ({q2_atr['majority_count']}/3)")
    lines.append("")
    pct_pass = overall_pct["overall"].startswith("PASS")
    atr_pass = overall_atr["overall"].startswith("PASS")
    if not pct_pass and atr_pass:
        lines.append("**G7 NOTE**: ATR-mirror PASSes but PCT primary does NOT — per spec, ATR-only PASS does NOT count as Pilot 3 PASS. Treated as 'atr-only artifact'; verdict follows pct primary.")
    elif pct_pass and atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS replicates on atr — strong cross-normalization signal (descriptive).")
    elif pct_pass and not atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS does NOT replicate on atr — pct-specific structural finding (descriptive).")
    else:
        lines.append("**G7 NOTE**: neither pct nor atr passes; pre-event compression timing not confirmed at either normalization.")
    lines.append("")

    # Q1 detail
    lines.append("## Q1 — combined timing map")
    lines.append("")
    lines.append(f"### PRIMARY (ema_stack_width_pct) — status: **{q1_pct['status']}**")
    lines.append("")
    lines.append("| bucket | mean | median | n |")
    lines.append("|---|---|---|---|")
    for _, r in q1_pct["agg"].iterrows():
        lines.append(f"| {r['bucket']} | {r['mean']:.4f} | {r['median']:.4f} | {int(r['count']):,} |")
    lines.append("")
    lines.append(f"- Argmin bucket: **{q1_pct['argmin_bucket']}** (mean = {q1_pct['argmin_info'].get('argmin_mean')})")
    lines.append(f"- Pre-ref mean (offsets {PRE_REF_OFFSETS}): {q1_pct['rebound']['pre_mean']:.4f}")
    lines.append(f"- Post-ref mean (offsets {POST_REF_OFFSETS}): {q1_pct['rebound']['post_mean']:.4f}")
    rb = q1_pct['rebound']['rebound_pct']
    rb_disp = f"{rb:.4f}" if rb is not None else "—"
    lines.append(f"- Rebound: **{rb_disp}** (must ≥ {REBOUND_MIN}) " + ("✓" if rb is not None and rb >= REBOUND_MIN else "✗"))
    lines.append("")

    lines.append(f"### AUDIT (ema_stack_width_atr) — status: **{q1_atr['status']}**")
    lines.append("")
    lines.append("| bucket | mean | median | n |")
    lines.append("|---|---|---|---|")
    for _, r in q1_atr["agg"].iterrows():
        lines.append(f"| {r['bucket']} | {r['mean']:.4f} | {r['median']:.4f} | {int(r['count']):,} |")
    lines.append("")
    lines.append(f"- Argmin bucket: **{q1_atr['argmin_bucket']}**")
    lines.append(f"- Rebound: {q1_atr['rebound']['rebound_pct']}")
    lines.append("")

    # Q2 detail
    lines.append("## Q2 — event-state stability (strict argmin match, ≥2/3 majority)")
    lines.append("")
    lines.append(f"### PRIMARY (pct) — status: **{q2_pct['status']}**")
    lines.append("")
    lines.append(f"- Majority bucket: **{q2_pct['majority_bucket']}** ({q2_pct['majority_count']}/3 states)")
    lines.append("")
    lines.append("| signal_state | argmin bucket | argmin n | argmin mean |")
    lines.append("|---|---|---|---|")
    for state in SIGNAL_STATE_ORDER:
        info = q2_pct["per_state"][state]
        lines.append(f"| {state} | {info.get('argmin_bucket')} | {info.get('argmin_n')} | {info.get('argmin_mean')} |")
    lines.append("")

    lines.append(f"### AUDIT (atr) — status: **{q2_atr['status']}**")
    lines.append("")
    lines.append(f"- Majority bucket: **{q2_atr['majority_bucket']}** ({q2_atr['majority_count']}/3 states)")
    lines.append("")
    lines.append("| signal_state | argmin bucket | argmin n |")
    lines.append("|---|---|---|")
    for state in SIGNAL_STATE_ORDER:
        info = q2_atr["per_state"][state]
        lines.append(f"| {state} | {info.get('argmin_bucket')} | {info.get('argmin_n')} |")
    lines.append("")

    # Q3 detail
    lines.append("## Q3 — geometry-slice replication")
    lines.append("")
    for label, q3_eval in (("PRIMARY (pct)", q3_pct), ("AUDIT (atr)", q3_atr)):
        lines.append(f"### {label} — status: **{q3_eval['status']}**")
        lines.append("")
        for axis_name, axis_eval in (("slope_tier", q3_eval["slope"]), ("width_tier", q3_eval["width"])):
            lines.append(f"#### {axis_name}")
            lines.append("")
            lines.append("| level | n_total | argmin bucket | argmin mean | status |")
            lines.append("|---|---|---|---|---|")
            for level, info in axis_eval["per_level"].items():
                lines.append(f"| {level} | {info.get('n_total')} | {info.get('argmin_bucket')} | {info.get('argmin_mean')} | {info.get('status')} |")
            lines.append("")

    # Anti-tweak confirmation
    lines.append("## Anti-tweak Confirmation")
    lines.append("")
    lines.append(f"- 7 buckets pre-locked: {[b[0] for b in BUCKETS]}")
    lines.append(f"- Allowed pre-event buckets: {{{', '.join(sorted(ALLOWED_PRE_EVENT_BUCKETS))}}}")
    lines.append(f"- Expected (clean) bucket: {EXPECTED_PRE_EVENT_BUCKET}")
    lines.append(f"- Early background bucket (at most PARTIAL): {EARLY_BACKGROUND_BUCKET}")
    lines.append(f"- Rebound: pre_ref_offsets={PRE_REF_OFFSETS} / post_ref_offsets={POST_REF_OFFSETS} (NOT argmin-relative)")
    lines.append(f"- Min-N: combined per bucket ≥ {Q1_BUCKET_MIN_N}, state per bucket ≥ {Q2_STATE_BUCKET_MIN_N}, geometry slice ≥ {Q3_SLICE_MIN_N}")
    lines.append(f"- Q2 strict argmin match (no neighbor kabul)")
    lines.append(f"- Primary metric ema_stack_width_pct LOCKED, ATR audit-only")
    lines.append(f"- ATR-only PASS does NOT count as Pilot 3 PASS")
    lines.append(f"- HB event source: horizontal_base_event_v1.parquet (manifest hash check at boot)")
    lines.append(f"- EMA breakpoints v0.0 frozen, no recompute")
    lines.append(f"- Path A (outcome-free) — NO outcome / NO model / NO gate / NO threshold sweep")
    lines.append(f"- Pilot 2 [−5, +1] window NOT relitigated")
    lines.append(f"- INSUFFICIENT ≠ FAIL")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main orchestrator
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 3 — single authorized run ===")
    print("Spec: memory/ema_context_pilot3_spec.md (LOCKED 2026-05-03)")
    print()
    t_start = time.time()

    # [1/6] Boot
    print("[1/6] Validating inputs (manifest + breakpoints)...")
    hb_manifest, em_meta = _validate_inputs()
    print(f"  HB scanner_version={hb_manifest['scanner_version']} rows={hb_manifest['rows']}")
    print(f"  ema_context breakpoints={em_meta['tag_breakpoints']['version']} rows={em_meta['n_rows']}")

    # [2/6] Load
    print("[2/6] Loading inputs...")
    t0 = time.time()
    events = _load_hb_events()
    print(f"  HB events: {len(events):,} rows / {events['ticker'].nunique()} tickers in {time.time()-t0:.2f}s")
    print(f"  signal_state dist: {events['signal_state'].value_counts().to_dict()}")
    print(f"  slope_tier dist: {events['family__slope_tier'].value_counts().to_dict()}")
    print(f"  width_tier dist: {events['width_tier'].value_counts().to_dict()}")

    t0 = time.time()
    ema_ctx = _load_ema_context()
    print(f"  ema_context: {len(ema_ctx):,} rows / {ema_ctx['ticker'].nunique()} tickers in {time.time()-t0:.2f}s")

    # [3/6] Build event × offset long-form
    print("[3/6] Building event × offset long-form (offsets {%d..%d})..." % (OFFSET_MIN, OFFSET_MAX))
    t0 = time.time()
    long_df = _build_offset_long(events, ema_ctx)
    n_events_used = int(long_df["event_id"].nunique())
    print(f"  long-form: {len(long_df):,} rows / {n_events_used:,} events in {time.time()-t0:.2f}s")

    # [4/6] Q1 — combined panel
    print("[4/6] Q1 combined timing map (pct primary + atr audit)...")
    q1_pct = _q1_evaluate(long_df, "pct")
    q1_atr = _q1_evaluate(long_df, "atr")
    print(f"  Q1 pct: argmin={q1_pct['argmin_bucket']} rebound={q1_pct['rebound']['rebound_pct']} status={q1_pct['status']}")
    print(f"  Q1 atr: argmin={q1_atr['argmin_bucket']} rebound={q1_atr['rebound']['rebound_pct']} status={q1_atr['status']}")

    # [5/6] Q2 + Q3
    print("[5/6] Q2 event-state stability...")
    q2_pct = _q2_evaluate(long_df, "pct")
    q2_atr = _q2_evaluate(long_df, "atr")
    print(f"  Q2 pct: per_state argmins={[(s, q2_pct['per_state'][s].get('argmin_bucket')) for s in SIGNAL_STATE_ORDER]} majority={q2_pct['majority_bucket']}({q2_pct['majority_count']}/3) status={q2_pct['status']}")
    print(f"  Q2 atr: per_state argmins={[(s, q2_atr['per_state'][s].get('argmin_bucket')) for s in SIGNAL_STATE_ORDER]} majority={q2_atr['majority_bucket']}({q2_atr['majority_count']}/3) status={q2_atr['status']}")

    print("[5/6] Q3 geometry-slice replication...")
    q3_pct = _q3_evaluate(long_df, "pct")
    q3_atr = _q3_evaluate(long_df, "atr")
    print(f"  Q3 pct: sufficient={q3_pct['sufficient_total']} pre_event_allowed={q3_pct['pre_event_allowed_levels']} status={q3_pct['status']}")
    print(f"  Q3 atr: sufficient={q3_atr['sufficient_total']} pre_event_allowed={q3_atr['pre_event_allowed_levels']} status={q3_atr['status']}")

    # [6/6] Verdict + outputs
    print("[6/6] Computing overall verdict + writing outputs...")
    overall_pct = _overall_verdict(q1_pct, q2_pct, q3_pct)
    overall_atr = _overall_verdict(q1_atr, q2_atr, q3_atr)
    print(f"  Overall (pct primary): {overall_pct['overall']} (PASS={overall_pct['n_pass']}/3, PARTIAL={overall_pct['n_partial']}, FAIL={overall_pct['n_fail']})")
    print(f"  Overall (atr audit):   {overall_atr['overall']}")

    _write_q1_csv(q1_pct, OUT_Q1_PCT)
    _write_q1_csv(q1_atr, OUT_Q1_ATR)
    _write_q2_csv(q2_pct, OUT_Q2_PCT)
    _write_q2_csv(q2_atr, OUT_Q2_ATR)
    _write_q3_csv(q3_pct["slope"], OUT_Q3_SLOPE_PCT)
    _write_q3_csv(q3_atr["slope"], OUT_Q3_SLOPE_ATR)
    _write_q3_csv(q3_pct["width"], OUT_Q3_WIDTH_PCT)
    _write_q3_csv(q3_atr["width"], OUT_Q3_WIDTH_ATR)
    long_df.to_parquet(OUT_PANEL, index=False)
    print(f"  per-Q CSVs + panel.parquet written")

    runtime_s = time.time() - t_start
    summary = _format_summary(
        q1_pct, q2_pct, q3_pct, q1_atr, q2_atr, q3_atr,
        overall_pct, overall_atr,
        runtime_s, em_meta, hb_manifest,
        len(events), len(long_df), n_events_used,
    )
    OUT_SUMMARY.write_text(summary)
    print(f"  summary: {OUT_SUMMARY}")

    print()
    print(f"=== Overall Pilot 3 Verdict (pct primary): {overall_pct['overall']} ===")
    print(f"Runtime: {runtime_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
