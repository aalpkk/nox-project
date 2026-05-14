"""ema_context Pilot 3 — FORWARD sibling (key-based remap + new-current emission).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Forward output contract (§3.4 P3-a):
  - Read locked pilot 3 panel verbatim (322,011 rows / 10,470 unique event_id).
  - Bridge each locked event_id N → archived HB row N → stable_event_key tuple.
  - Match against current HB. Three populations:
      matched_old:    locked event_id whose stable_event_key is present in current HB
                      → carry locked panel rows forward verbatim (research_frozen pct/atr)
      unmatched_old:  locked event_id whose stable_event_key is absent in current HB
                      → DROP from forward panel; record in unmapped_locked_events
      new_current:    stable_event_key present in current HB but not in archived HB
                      → compute fresh forward panel rows via pilot3._build_offset_long
                        on filtered new events with CURRENT ema_context_daily
  - Emit:
      output/ema_context_pilot3_panel_forward.parquet
      output/ema_context_pilot3_event_id_to_stable_key.parquet  (bridge sidecar)
  - Locked output/ema_context_pilot3_panel.parquet remains BYTE-EQUAL.

Hard rules (FORBIDDEN per §3.5):
  - No positional fallback for unmatched_old.
  - No synthetic event injection.
  - No modification of locked Pilot 3 outputs.
  - No re-tuple of stable_event_key at runtime (LOCKED to 5-tuple).

Returns dict on success with key_remap counts and unmapped_locked_events for manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import analytical kernel from locked pilot 3 (NOT modified, library-import only).
from tools.ema_context_pilot3 import (  # type: ignore
    _build_offset_long,
    WIDTH_TIER_LO,
    WIDTH_TIER_HI,
    WIDTH_TIER_ORDER,
)

# =============================================================================
# LOCKED CONSTANTS
# =============================================================================

CANDIDATE_KEY = ["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"]
RESEARCH_BASELINE_HB_ROWS = 10470
EXPECTED_LOCKED_HB_ARCHIVE_SHA = (
    "2eb8a9a5d68e7e4831158f6a3e97c8b74521591af55e29a9682aa3ae7107b818"
)
EXPECTED_LOCKED_PILOT3_PANEL_ROWS = 322011
EXPECTED_LOCKED_PILOT3_UNIQUE_EID = 10470

LOCKED_HB_ARCHIVE = (
    PROJECT_ROOT
    / "output"
    / "_archive"
    / "horizontal_base_event_v1__pre_refresh__asof_2026-04-29__sha256_2eb8a9a5.parquet"
)
LOCKED_PILOT3_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot3_panel.parquet"
CURRENT_HB = PROJECT_ROOT / "output" / "horizontal_base_event_v1.parquet"
EMA_CONTEXT_DAILY = PROJECT_ROOT / "output" / "ema_context_daily.parquet"

OUT_PANEL_FORWARD = PROJECT_ROOT / "output" / "ema_context_pilot3_panel_forward.parquet"
OUT_BRIDGE = PROJECT_ROOT / "output" / "ema_context_pilot3_event_id_to_stable_key.parquet"

# event_id namespace separation: new-current events get event_id offset by RESEARCH_BASELINE
# so 0..10469 = locked (matched_old preserved) and 10470+ = new_current. No collision.
NEW_CURRENT_EVENT_ID_OFFSET = RESEARCH_BASELINE_HB_ROWS


# =============================================================================
# Helpers
# =============================================================================


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CANDIDATE_KEY columns to a hashable canonical form for matching."""
    out = df.copy()
    out["bar_date"] = pd.to_datetime(out["bar_date"]).dt.normalize()
    out["breakout_bar_date"] = pd.to_datetime(out["breakout_bar_date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str)
    out["setup_family"] = out["setup_family"].astype(str)
    out["signal_type"] = out["signal_type"].astype(str)
    return out


def _key_tuples(df: pd.DataFrame) -> list[tuple]:
    return list(
        zip(
            df["ticker"].astype(str).values,
            df["bar_date"].values,
            df["setup_family"].astype(str).values,
            df["signal_type"].astype(str).values,
            df["breakout_bar_date"].values,
        )
    )


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    t0 = time.time()
    print("[Pilot 3 forward] Validating inputs (no row-count guard; key-based remap)...", flush=True)

    if not LOCKED_HB_ARCHIVE.exists():
        raise FileNotFoundError(f"Locked HB archive missing: {LOCKED_HB_ARCHIVE}")
    if not LOCKED_PILOT3_PANEL.exists():
        raise FileNotFoundError(f"Locked Pilot 3 panel missing: {LOCKED_PILOT3_PANEL}")
    if not CURRENT_HB.exists():
        raise FileNotFoundError(f"Current HB missing: {CURRENT_HB}")
    if not EMA_CONTEXT_DAILY.exists():
        raise FileNotFoundError(f"ema_context_daily missing: {EMA_CONTEXT_DAILY}")

    # Verify locked HB archive sha256 (anti-tamper)
    arch_sha = _sha256(LOCKED_HB_ARCHIVE)
    if arch_sha != EXPECTED_LOCKED_HB_ARCHIVE_SHA:
        raise RuntimeError(
            f"Locked HB archive sha256 mismatch:\n  got      {arch_sha}\n  "
            f"expected {EXPECTED_LOCKED_HB_ARCHIVE_SHA}\nSPEC §3.4 violation."
        )
    print(f"  Locked HB archive sha256 OK ({arch_sha[:16]}…)", flush=True)

    # ---- Load inputs ----
    print("[Pilot 3 forward] Loading inputs...", flush=True)
    locked_panel = pd.read_parquet(LOCKED_PILOT3_PANEL)
    if len(locked_panel) != EXPECTED_LOCKED_PILOT3_PANEL_ROWS:
        raise RuntimeError(
            f"Locked Pilot 3 panel rows mismatch: got {len(locked_panel)} != "
            f"{EXPECTED_LOCKED_PILOT3_PANEL_ROWS}. SPEC violation."
        )
    if locked_panel["event_id"].nunique() != EXPECTED_LOCKED_PILOT3_UNIQUE_EID:
        raise RuntimeError(
            f"Locked Pilot 3 panel unique event_id mismatch: "
            f"got {locked_panel['event_id'].nunique()} != {EXPECTED_LOCKED_PILOT3_UNIQUE_EID}."
        )

    locked_hb = pd.read_parquet(LOCKED_HB_ARCHIVE)
    if len(locked_hb) != RESEARCH_BASELINE_HB_ROWS:
        raise RuntimeError(
            f"Locked HB archive rows mismatch: got {len(locked_hb)} != "
            f"{RESEARCH_BASELINE_HB_ROWS}."
        )
    locked_hb = _normalize_keys(locked_hb).reset_index(drop=True)

    current_hb = pd.read_parquet(CURRENT_HB)
    current_hb = _normalize_keys(current_hb).reset_index(drop=True)
    print(
        f"  locked_panel rows={len(locked_panel):,} unique_eid={locked_panel['event_id'].nunique():,}",
        flush=True,
    )
    print(f"  locked_hb rows={len(locked_hb):,}", flush=True)
    print(f"  current_hb rows={len(current_hb):,}", flush=True)

    # ---- Stable_event_key uniqueness checks ----
    print("[Pilot 3 forward] stable_event_key uniqueness checks...", flush=True)
    dup_locked = int(locked_hb.duplicated(subset=CANDIDATE_KEY).sum())
    dup_current = int(current_hb.duplicated(subset=CANDIDATE_KEY).sum())
    if dup_locked > 0:
        raise RuntimeError(
            f"stable_key_duplicate_detected_in_locked_hb: {dup_locked} duplicates. "
            f"SPEC §3.5 invariant violated; would force re-tuple."
        )
    if dup_current > 0:
        raise RuntimeError(
            f"stable_key_duplicate_detected_in_current_hb: {dup_current} duplicates. "
            f"SPEC §3.5 invariant violated; would force re-tuple."
        )
    print(f"  duplicates: locked=0 / current=0", flush=True)

    # ---- Build bridge: locked event_id N → stable_event_key tuple ----
    print("[Pilot 3 forward] Building event_id → stable_event_key bridge...", flush=True)
    locked_keys = locked_hb[CANDIDATE_KEY].copy()
    locked_keys["event_id"] = np.arange(len(locked_keys), dtype=np.int64)

    current_key_tuples = set(_key_tuples(current_hb))
    locked_keys_tuples = _key_tuples(locked_keys)
    locked_keys["matched_in_current"] = [k in current_key_tuples for k in locked_keys_tuples]

    # Map current_hb position by stable_event_key for fast lookup
    current_pos_lookup: dict[tuple, int] = {
        k: i for i, k in enumerate(_key_tuples(current_hb))
    }
    locked_keys["current_hb_index"] = [
        current_pos_lookup.get(k) for k in locked_keys_tuples
    ]

    matched_event_ids = set(
        locked_keys.loc[locked_keys["matched_in_current"], "event_id"].astype(int).tolist()
    )
    unmatched_event_ids = sorted(
        locked_keys.loc[~locked_keys["matched_in_current"], "event_id"].astype(int).tolist()
    )
    n_matched = len(matched_event_ids)
    n_unmatched = len(unmatched_event_ids)
    print(f"  matched_old={n_matched:,} / unmatched_old={n_unmatched:,}", flush=True)

    # ---- Identify new-current events ----
    locked_key_tuples_set = set(locked_keys_tuples)
    current_key_tuples_list = _key_tuples(current_hb)
    is_new = np.array(
        [k not in locked_key_tuples_set for k in current_key_tuples_list], dtype=bool
    )
    n_new = int(is_new.sum())
    print(f"  new_current={n_new:,}", flush=True)

    # ---- §3.5 hard checks (sub-ceiling on unmapped_locked_count + Q8 deltas) ----
    UNMAPPED_SUBCEILING = max(10, int(np.ceil(0.001 * RESEARCH_BASELINE_HB_ROWS)))
    if n_unmatched > UNMAPPED_SUBCEILING:
        raise RuntimeError(
            f"emission_diff_exceeds_subceiling: unmatched_old={n_unmatched} > "
            f"sub-ceiling={UNMAPPED_SUBCEILING}. SPEC §6.4 HALT."
        )
    delta_rows = len(current_hb) - RESEARCH_BASELINE_HB_ROWS
    delta_pct = delta_rows / RESEARCH_BASELINE_HB_ROWS
    if delta_pct > 0.05 or delta_rows > 1000:
        raise RuntimeError(
            f"forward_delta_exceeds_q8_ceiling: delta_rows={delta_rows} delta_pct={delta_pct:.4f}. "
            f"SPEC §3.2 + §6.4 HALT."
        )
    print(
        f"  sub-ceiling check OK ({n_unmatched} ≤ {UNMAPPED_SUBCEILING}); "
        f"Q8 delta_rows={delta_rows} delta_pct={delta_pct:.4f}",
        flush=True,
    )

    # ---- Classify unmapped events with stable_event_key tuple + near-miss counts ----
    unmapped_records = []
    for eid in unmatched_event_ids:
        row = locked_hb.iloc[int(eid)]
        key_dict = {c: row[c] for c in CANDIDATE_KEY}
        key_dict_str = {
            c: (str(v) if not isinstance(v, pd.Timestamp) else v.isoformat()[:10])
            for c, v in key_dict.items()
        }
        # Near-miss counts
        same_ticker_same_breakout = int(
            (
                (current_hb["ticker"] == row["ticker"])
                & (current_hb["breakout_bar_date"] == row["breakout_bar_date"])
            ).sum()
        )
        same_ticker_same_bar_date = int(
            (
                (current_hb["ticker"] == row["ticker"])
                & (current_hb["bar_date"] == row["bar_date"])
            ).sum()
        )
        unmapped_records.append(
            {
                "old_event_id": int(eid),
                "stable_event_key": key_dict_str,
                "near_miss_same_ticker_same_breakout_count": same_ticker_same_breakout,
                "near_miss_same_ticker_same_bar_date_count": same_ticker_same_bar_date,
            }
        )

    # ---- Build matched-old forward panel block (verbatim from locked panel + key cols) ----
    print("[Pilot 3 forward] Building matched_old panel block...", flush=True)
    matched_panel = locked_panel[locked_panel["event_id"].isin(matched_event_ids)].copy()
    if matched_panel["event_id"].nunique() != n_matched:
        raise RuntimeError(
            f"matched_panel unique_eid mismatch: got {matched_panel['event_id'].nunique()} != "
            f"{n_matched}."
        )
    # Attach stable_event_key cols by joining with locked_keys
    matched_panel = matched_panel.merge(
        locked_keys[["event_id"] + CANDIDATE_KEY].rename(columns={"ticker": "ticker_key"}),
        on="event_id",
        how="left",
    )
    # Drop redundant ticker_key (locked_panel already has ticker; assert equal)
    if not (matched_panel["ticker"] == matched_panel["ticker_key"]).all():
        raise RuntimeError(
            "ticker mismatch between locked_panel and locked_hb_keys join — "
            "stable_event_key bridge is broken."
        )
    matched_panel = matched_panel.drop(columns=["ticker_key"])
    matched_panel["track"] = "matched_old"
    print(f"  matched_panel rows={len(matched_panel):,}", flush=True)

    # ---- Build new-current forward panel block via pilot3._build_offset_long ----
    print("[Pilot 3 forward] Building new_current panel block via _build_offset_long...", flush=True)
    new_hb = current_hb.loc[is_new].reset_index(drop=True).copy()
    if len(new_hb) != n_new:
        raise RuntimeError(f"new_hb size mismatch: {len(new_hb)} != {n_new}")

    # Prepare events DataFrame in the schema _build_offset_long + _load_hb_events expect
    # (ticker, bar_date, signal_state, family__slope_tier, width_tier).
    # Plus carry stable_event_key extra cols for re-attach after kernel.
    new_events = new_hb[
        [
            "ticker",
            "bar_date",
            "signal_state",
            "family__slope_tier",
            "family__channel_width_pctile_252",
            "setup_family",
            "signal_type",
            "breakout_bar_date",
        ]
    ].copy()
    new_events["signal_state"] = new_events["signal_state"].astype(str)
    new_events["width_tier"] = (
        pd.cut(
            new_events["family__channel_width_pctile_252"],
            bins=[-np.inf, WIDTH_TIER_LO, WIDTH_TIER_HI, np.inf],
            labels=WIDTH_TIER_ORDER,
        )
        .astype(str)
    )
    # Add forward-namespace event_id (positional within new_events)
    new_events = new_events.reset_index(drop=True)
    new_events["forward_pos_event_id"] = new_events.index.astype(np.int64)

    # Load current ema_context_daily for kernel
    ema_ctx = pd.read_parquet(EMA_CONTEXT_DAILY)
    ema_ctx["date"] = pd.to_datetime(ema_ctx["date"]).dt.normalize()

    new_long = _build_offset_long(new_events, ema_ctx)
    print(
        f"  new_long rows={len(new_long):,} unique_eid={new_long['event_id'].nunique()}",
        flush=True,
    )

    # Re-key new_long: kernel emits event_id = positional 0..N-1.
    # Map to forward namespace and attach stable_event_key cols.
    new_long = new_long.rename(columns={"event_id": "forward_pos_event_id"})
    new_long = new_long.merge(
        new_events[
            [
                "forward_pos_event_id",
                "bar_date",
                "setup_family",
                "signal_type",
                "breakout_bar_date",
            ]
        ],
        on="forward_pos_event_id",
        how="left",
    )
    new_long["event_id"] = (
        new_long["forward_pos_event_id"].astype(np.int64) + NEW_CURRENT_EVENT_ID_OFFSET
    )
    new_long = new_long.drop(columns=["forward_pos_event_id"])
    new_long["track"] = "new_current"

    # Reorder cols to match matched_panel
    new_long = new_long[matched_panel.columns]

    # ---- Concatenate ----
    print("[Pilot 3 forward] Concatenating matched_old + new_current...", flush=True)
    forward_panel = pd.concat([matched_panel, new_long], axis=0, ignore_index=True)
    forward_panel_unique_eid = int(forward_panel["event_id"].nunique())
    print(
        f"  forward_panel rows={len(forward_panel):,} unique_eid={forward_panel_unique_eid:,}",
        flush=True,
    )
    expected_unique_eid = n_matched + n_new
    if forward_panel_unique_eid != expected_unique_eid:
        raise RuntimeError(
            f"forward_panel unique_eid mismatch: got {forward_panel_unique_eid} != "
            f"{expected_unique_eid} (matched_old + new_current)."
        )

    # ---- Emit bridge sidecar ----
    print("[Pilot 3 forward] Emitting bridge sidecar...", flush=True)
    bridge = locked_keys[
        ["event_id"] + CANDIDATE_KEY + ["matched_in_current", "current_hb_index"]
    ].copy()
    bridge["event_id"] = bridge["event_id"].astype(np.int64)

    # ---- Atomic writes ----
    print("[Pilot 3 forward] Writing outputs (atomic)...", flush=True)

    def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)

    _atomic_write_parquet(forward_panel, OUT_PANEL_FORWARD)
    _atomic_write_parquet(bridge, OUT_BRIDGE)
    print(f"  Wrote: {OUT_PANEL_FORWARD.name} ({OUT_PANEL_FORWARD.stat().st_size:,} B)", flush=True)
    print(f"  Wrote: {OUT_BRIDGE.name} ({OUT_BRIDGE.stat().st_size:,} B)", flush=True)

    runtime_s = time.time() - t0
    print(f"[Pilot 3 forward] Done in {runtime_s:.1f}s", flush=True)

    # Emit summary JSON to stdout for orchestrator capture
    summary = {
        "tool": "ema_context_pilot3_forward",
        "runtime_s": round(runtime_s, 3),
        "research_baseline_HB_rows": RESEARCH_BASELINE_HB_ROWS,
        "current_HB_rows": int(len(current_hb)),
        "delta_rows": delta_rows,
        "delta_pct": float(delta_pct),
        "matched_old_event_count": n_matched,
        "unmatched_old_event_count": n_unmatched,
        "new_current_event_count": n_new,
        "duplicate_stable_event_key_count_locked": dup_locked,
        "duplicate_stable_event_key_count_current": dup_current,
        "unmapped_locked_count_subceiling": UNMAPPED_SUBCEILING,
        "unmapped_locked_events": unmapped_records,
        "forward_panel_rows": int(len(forward_panel)),
        "forward_panel_unique_event_id": forward_panel_unique_eid,
        "bridge_rows": int(len(bridge)),
        "out_panel_forward": str(OUT_PANEL_FORWARD.relative_to(PROJECT_ROOT)),
        "out_bridge": str(OUT_BRIDGE.relative_to(PROJECT_ROOT)),
    }
    print("PILOT3_FORWARD_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, default=str))
    print("PILOT3_FORWARD_SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
