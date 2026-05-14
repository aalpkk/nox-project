"""paper_execution_v0_trigger_retest — Line TR forward paper/shadow execution.

Spec: memory/paper_execution_v0_trigger_retest_spec.md (LOCKED 2026-05-04, ONAY 2026-05-04).

This is PAPER / SHADOW only. NOT live trading. NOT a live entry filter.
NOT a position sizing rule. NOT a ranking signal. NOT an EMA gate.

Eligibility (Line TR, EVENT_NEAR_TIER):
  signal_state in {trigger, retest_bounce}
  AND -5 <= earliness_score_pct <= -1.

NON_EVENT_NEAR (signal_state in {trigger, retest_bounce} AND > -1)
  → SKIPPED, skip_reason='NON_EVENT_NEAR' (warning, NOT contra-signal).
DROP_EARLY_DIAGNOSTIC (signal_state in {trigger, retest_bounce} AND < -5)
  → SKIPPED, skip_reason='DROP_EARLY_DIAGNOSTIC' (descriptive census, NOT trade rule).
OUT_OF_SCOPE (extended / NaN earliness / other states)
  → omitted from trade table (counted in census only).

Two-lane:
  --mode close_confirmed (Lane 2, canonical paper signal source)
  --mode preview         (Lane 1, operational pre-scan only — no perf accounting)

Entry: next trading-day open after close-confirmed signal day.
Exit:  +10 trading-day close. No TP / SL / trailing.

Two views via boolean flags:
  in_view_raw_all_candidates  — primary, no cap, all close-confirmed EVENT_NEAR_TIER
  in_view_capped_portfolio    — operational, max 5/day, alphabetical ticker

Frozen risk denominator: R_per_share = entry_reference_price - invalidation_level
(reused from HB event parquet so realized_R_paper denominator is comparable to
Pilot 7 historical realized_R_10d denominator). Numerator uses paper-trade prices
(next-open entry, +10d close exit).

Anti-tweak: spec is LOCKED end-to-end. Bug affecting eligibility/return → void
+ rerun + manifest note. No threshold/horizon/cost/selection sweep.

Filename + column invariants:
  - every output file is suffixed `_trigger_retest`
  - every trade-log row has `line == "TRIGGER_RETEST"`
  - tool refuses to write if any output path collides with a Line E
    `paper_execution_v0_*` artifact (no-suffix or other-suffix file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import intraday_1h  # noqa: E402

# ---------------------------------------------------------------------------
# LOCKED constants (spec §4 + §5 + §6)

EVENT_NEAR_LO = -5.0  # inclusive — spec §1.2 / §4.1
EVENT_NEAR_HI = -1.0  # inclusive — spec §1.2 / §4.1
DROP_BOUNDARY = -5.0  # < -5 = DROP_EARLY_DIAGNOSTIC — spec §1.2 / §4.2
HORIZON_TRADING_DAYS = 10  # spec §4.4
COST_BPS_ENTRY = 10.0
COST_BPS_EXIT = 10.0
COST_ROUNDTRIP_FRAC = (COST_BPS_ENTRY + COST_BPS_EXIT) / 10000.0  # 0.0020
MAX_NEW_POSITIONS_PER_DAY = 5  # spec §5.2

LINE_LABEL = "TRIGGER_RETEST"  # spec §7.2 invariant
TR_STATES = {"trigger", "retest_bounce"}

EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_HB_SHA256 = (
    "2eb8a9a5d68e7e4831158f6a3e97c8b74521591af55e29a9682aa3ae7107b818"
)

# Validity revision (LOCKED 2026-05-04 + ONAY §14 additive clarification).
SOURCE_LINE = LINE_LABEL  # "TRIGGER_RETEST"
VALIDITY_WINDOW_DEFAULT = "next_trading_day"
SIGNATURE_HEX_LEN = 16
WIDTH_ROUND_DECIMALS = 6

# ---------------------------------------------------------------------------
# paths

HB_PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
PILOT7_PANEL = ROOT / "output/ema_context_pilot7_panel.parquet"
PILOT7_GATES = ROOT / "output/ema_context_pilot7_gates_main.csv"
PILOT7_DROP_CENSUS = ROOT / "output/ema_context_pilot7_drop_diagnostic_census.csv"
OHLCV_MASTER = ROOT / "output/extfeed_intraday_1h_3y_master.parquet"
EMA_CONTEXT_DAILY = ROOT / "output/ema_context_daily.parquet"

# Forward mode (LOCKED 2026-05-06 paper_execution_v0_forward_run_spec.md):
# When --forward is set, swap PILOT7_PANEL to forward artifact and replace
# positional join + locked-HB sha gate with stable_event_key 5-tuple key-based
# merge (PILOT7_GATES / PILOT7_DROP_CENSUS stay locked-research per spec §6.3).
_FORWARD_MODE = False
PILOT7_PANEL_FORWARD = ROOT / "output/ema_context_pilot7_panel_forward.parquet"
STABLE_EVENT_KEY_NORM = ["ticker", "_bar_date_d", "setup_family", "signal_type", "_breakout_d"]
_FORWARD_MERGE_DIAG: dict = {}


def _hb_parquet() -> Path:
    return HB_PARQUET


def _pilot7_panel() -> Path:
    return PILOT7_PANEL_FORWARD if _FORWARD_MODE else PILOT7_PANEL


OUT_TRADES_PARQUET = ROOT / "output/paper_execution_v0_trigger_retest_trades.parquet"
OUT_TRADES_CSV = ROOT / "output/paper_execution_v0_trigger_retest_trades.csv"
OUT_DAILY_SUMMARY = ROOT / "output/paper_execution_v0_trigger_retest_daily_summary.csv"
OUT_PREVIEW_AUDIT = ROOT / "output/paper_execution_v0_trigger_retest_preview_audit.csv"
OUT_SUMMARY_MD = ROOT / "output/paper_execution_v0_trigger_retest_summary.md"
OUT_MANIFEST = ROOT / "output/paper_execution_v0_trigger_retest_manifest.json"

# Line E filenames — protected from overwrite (§14 item 16).
LINE_E_PROTECTED = {
    ROOT / "output/paper_execution_v0_trades.parquet",
    ROOT / "output/paper_execution_v0_trades.csv",
    ROOT / "output/paper_execution_v0_daily_summary.csv",
    ROOT / "output/paper_execution_v0_preview_audit.csv",
    ROOT / "output/paper_execution_v0_summary.md",
    ROOT / "output/paper_execution_v0_manifest.json",
}

# ---------------------------------------------------------------------------
# helpers


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stop(msg: str) -> None:
    print(f"[paper_execution_v0_trigger_retest] STOP — {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Validity revision helpers (LOCKED 2026-05-04 + §14 ONAY clarification).
# Spec: memory/paper_execution_v0_validity_revision_spec.md §3, §4, §5, §13, §14.
# Per spec §13.2 / §14.4 — duplicated inside this producer (no shared library).


def load_ema_daily_index(parquet_path: Path = EMA_CONTEXT_DAILY) -> dict:
    """Build (ticker, date_obj) -> ema_stack_width_pct lookup.

    Read-only consumption of `output/ema_context_daily.parquet`. NOT a recompute
    of EMA. NOT a mutation of the source. Per §14.2 ONAY clarification.
    """
    df = pd.read_parquet(parquet_path, columns=["ticker", "date", "ema_stack_width_pct"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    idx = {}
    for tk, d, w in zip(df["ticker"].values, df["date"].values, df["ema_stack_width_pct"].values):
        idx[(str(tk), d)] = float(w) if pd.notna(w) else None
    return idx


def trading_day_offset(cal: dict, ticker: str, base_date, offset_days: int):
    """Return the trading-day at `offset_days` from `base_date` on the per-ticker
    calendar. offset=0 → base_date itself (must be a trading day in cal). Negative
    offset → past; positive offset → future. None if walks off either end.
    """
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None
    dates = sub["date"].values
    base_np = np.datetime64(base_date)
    base_idx = np.searchsorted(dates, base_np, side="left")
    if base_idx >= len(dates) or dates[base_idx] != base_np:
        return None
    target_idx = base_idx + int(offset_days)
    if target_idx < 0 or target_idx >= len(dates):
        return None
    return sub.iloc[target_idx]["date"]


def next_trading_day(cal: dict, ticker: str, base_date):
    """Next trading-day strictly after `base_date` on the per-ticker calendar.
    None if calendar gap (no further trading day available).
    """
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None
    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(base_date), side="right")
    if idx >= len(dates):
        return None
    return sub.iloc[idx]["date"]


def compute_ema_signature_id(
    ticker: str,
    source_line: str,
    signal_state: str,
    signal_date_iso: str,
    argmin_offset_bar_iso_date: str,
    width_value: float,
) -> str:
    """Per §4.1 LOCKED formula. sha256-prefix over the six-tuple. Fail-fast on
    missing inputs is the caller's responsibility.
    """
    payload = (
        f"{ticker}|{source_line}|{signal_state}|{signal_date_iso}|"
        f"{argmin_offset_bar_iso_date}|{round(float(width_value), WIDTH_ROUND_DECIMALS)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:SIGNATURE_HEX_LEN]


def attach_validity_fields(
    ticker: str,
    source_line: str,
    signal_state: str,
    bar_date,
    earliness_score_pct,
    cal: dict,
    ema_daily_index: dict,
) -> dict:
    """Compute the five validity columns + signature for a forward-emitted candidate
    row. Returns dict with keys: paper_valid_from, paper_valid_until, paper_signal_age,
    paper_expired_flag, ema_signature_id, _argmin_date_iso, _width_at_argmin.

    Fail-fast (via _stop) on:
      - missing earliness_score_pct (signal_date-missing-class FAIL)
      - argmin date unresolvable (input-coverage FAIL)
      - width missing at (ticker, argmin date) (input-coverage FAIL)
      - next trading day unresolvable for paper_valid_until (calendar-gap FAIL)
    """
    if earliness_score_pct is None or pd.isna(earliness_score_pct):
        _stop(
            f"signal_date_unavailable: earliness_score_pct NULL for ticker={ticker} "
            f"signal_date={bar_date} (R1 signature requires non-null offset; "
            "no synthesized weaker fallback per spec §4.1)"
        )

    offset = int(earliness_score_pct)
    argmin_date = trading_day_offset(cal, ticker, bar_date, offset)
    if argmin_date is None:
        _stop(
            f"argmin_offset_bar unresolvable: ticker={ticker} signal_date={bar_date} "
            f"offset={offset} walks off ticker calendar (per §14.3)"
        )

    width = ema_daily_index.get((str(ticker), argmin_date))
    if width is None:
        _stop(
            f"width_value missing in ema_context_daily: ticker={ticker} "
            f"argmin_date={argmin_date} (per §14.3; no synthesized fallback)"
        )

    valid_until = next_trading_day(cal, ticker, bar_date)
    if valid_until is None:
        _stop(
            f"calendar-gap on paper_valid_until: ticker={ticker} "
            f"paper_valid_from={bar_date} has no next trading day in calendar "
            "(per §2.1.4 fail-fast)"
        )

    sig_id = compute_ema_signature_id(
        ticker=str(ticker),
        source_line=source_line,
        signal_state=str(signal_state),
        signal_date_iso=bar_date.isoformat() if hasattr(bar_date, "isoformat") else str(bar_date),
        argmin_offset_bar_iso_date=argmin_date.isoformat()
        if hasattr(argmin_date, "isoformat")
        else str(argmin_date),
        width_value=float(width),
    )

    return {
        "paper_valid_from": bar_date,
        "paper_valid_until": valid_until,
        "paper_signal_age": 0,
        "paper_expired_flag": False,
        "ema_signature_id": sig_id,
        "_argmin_date_iso": argmin_date.isoformat()
        if hasattr(argmin_date, "isoformat")
        else str(argmin_date),
        "_width_at_argmin": float(width),
    }


def apply_r1_r2_filter(
    candidate_signature_id: str,
    candidate_ticker: str,
    in_progress_log: list[dict],
    source_line: str,
):
    """Per §4.2 (R1) + §5.2 (R2) + §2.1.6 — independent filters; both can fire
    on the same candidate and counters increment independently. Returns
    `(allow, r1_fired, r2_fired)` where `allow = not (r1_fired or r2_fired)`.

    R1: same ema_signature_id present in any prior row (regardless of status /
    expired flag) → fired.
    R2: same (ticker, source_line) present in any prior row whose status is
    OPEN AND exit_date is None → fired. Per §5.3 R2 is per-line.
    """
    r1_fired = False
    r2_fired = False
    for prior in in_progress_log:
        if not r1_fired and prior.get("ema_signature_id") == candidate_signature_id:
            r1_fired = True
        if (
            not r2_fired
            and prior.get("ticker") == candidate_ticker
            and prior.get("line", source_line) == source_line
            and prior.get("status") == "OPEN"
            and prior.get("exit_date") is None
        ):
            r2_fired = True
        if r1_fired and r2_fired:
            break
    return (not (r1_fired or r2_fired)), r1_fired, r2_fired


def _assert_no_line_e_collision() -> None:
    """Spec §14 item 16: implementation MUST NOT modify, overwrite, or read-write
    Line E `paper_execution_v0_*` files. We assert each TR output path is in the
    `_trigger_retest` family and is not in the protected Line E set."""
    tr_outputs = {
        OUT_TRADES_PARQUET,
        OUT_TRADES_CSV,
        OUT_DAILY_SUMMARY,
        OUT_PREVIEW_AUDIT,
        OUT_SUMMARY_MD,
        OUT_MANIFEST,
    }
    for p in tr_outputs:
        if p in LINE_E_PROTECTED:
            _stop(f"line invariant violation — TR output collides with Line E: {p}")
        if "_trigger_retest" not in p.name:
            _stop(f"line invariant violation — TR output missing suffix: {p}")


# ---------------------------------------------------------------------------
# §6.2 / §6.3 boot integrity


def boot_integrity() -> dict:
    """Validates required artifacts (spec §6.1+§6.2+§6.3). Aborts via STOP on any
    failure. Returns dict of resolved paths + hashes for manifest emission.

    Forward mode (LOCKED 2026-05-06): the locked-research `EXPECTED_HB_SHA256`
    and `EXPECTED_HB_ROWS=10470` gates are skipped (HB is intentionally
    refreshed); HB↔panel parity replaces them. PILOT7_GATES /
    PILOT7_DROP_CENSUS paths are unchanged (spec §6.3).
    """
    info = {"checked_at": _utc_now(), "forward_mode": bool(_FORWARD_MODE)}

    hb_path = _hb_parquet()
    panel_path = _pilot7_panel()

    for label, path in [
        ("hb_event_parquet", hb_path),
        ("pilot7_panel", panel_path),
        ("pilot7_gates_main", PILOT7_GATES),
        ("pilot7_drop_diagnostic_census", PILOT7_DROP_CENSUS),
        ("ohlcv_master", OHLCV_MASTER),
        ("ema_context_daily", EMA_CONTEXT_DAILY),
    ]:
        if not path.exists():
            _stop(f"required artifact missing: {label} at {path}")
        info[label + "_path"] = str(path.relative_to(ROOT))
        info[label + "_size_bytes"] = path.stat().st_size

    # SHA256 hashes — required by spec §6.2(4) for manifest.
    info["hb_event_parquet_sha256"] = _file_hash(hb_path)
    info["pilot7_panel_sha256"] = _file_hash(panel_path)
    print("[paper_execution_v0_trigger_retest] hashing OHLCV master (one-time)...")
    info["ohlcv_master_sha256"] = _file_hash(OHLCV_MASTER)

    if not _FORWARD_MODE:
        # HB sha256 must match LOCKED expected (spec §6.1).
        if info["hb_event_parquet_sha256"] != EXPECTED_HB_SHA256:
            _stop(
                f"HB parquet sha256 mismatch — expected {EXPECTED_HB_SHA256[:16]}…, "
                f"observed {info['hb_event_parquet_sha256'][:16]}…"
            )

    # HB scanner_version + row count.
    hb = pd.read_parquet(hb_path, columns=["ticker", "scanner_version"])
    if (hb["scanner_version"] != EXPECTED_SCANNER_VERSION).any():
        _stop(
            f"hb scanner_version mismatch — expected {EXPECTED_SCANNER_VERSION}, "
            f"observed {hb['scanner_version'].unique().tolist()}"
        )
    if not _FORWARD_MODE and len(hb) != EXPECTED_HB_ROWS:
        _stop(f"hb row count mismatch — expected {EXPECTED_HB_ROWS}, got {len(hb)}")
    info["hb_scanner_version"] = EXPECTED_SCANNER_VERSION
    info["hb_event_rows"] = int(len(hb)) if _FORWARD_MODE else int(EXPECTED_HB_ROWS)

    # Pilot 7 panel must have required columns and matching row count.
    panel = pd.read_parquet(
        panel_path,
        columns=["ticker", "bar_date", "signal_state", "earliness_score_pct", "ema_tr_tier"],
    )
    if _FORWARD_MODE:
        if len(panel) != len(hb):
            _stop(
                f"forward HB↔pilot7-panel row mismatch hb={len(hb)} panel={len(panel)}"
            )
    elif len(panel) != EXPECTED_HB_ROWS:
        _stop(
            f"pilot7 panel row mismatch — expected {EXPECTED_HB_ROWS}, got {len(panel)}"
        )
    info["pilot7_panel_rows"] = int(len(panel))

    # OHLCV adapter is the 1h-resample (canonical Pilot 5/6/7 adapter).
    info["ohlcv_adapter"] = (
        "data.intraday_1h.daily_resample(extfeed_intraday_1h_3y_master.parquet) — "
        "same canonical adapter Pilot 5/6/7 historical labels traced through; "
        "adapter path explicitly recorded per spec §6.2 (no silent substitution)"
    )

    return info


# ---------------------------------------------------------------------------
# data loading + cohort tagging (spec §1.2 + cross-check vs Pilot 7 panel)


def load_signals() -> pd.DataFrame:
    """HB events filtered to signal_state ∈ {trigger, retest_bounce} joined
    to Pilot 7 panel; cohort tagged per spec §1.2.

    Default mode: positional join (locked-research, byte-equal).
    Forward mode (LOCKED 2026-05-06): key-based merge on stable_event_key
    5-tuple `(ticker, bar_date, setup_family, signal_type, breakout_bar_date)`
    with `validate="one_to_one"`.

    Cohort labels are re-derived from earliness_score_pct AND cross-checked
    against Pilot 7 panel's `ema_tr_tier`; mismatch → STOP (spec §10).
    """
    hb = pd.read_parquet(_hb_parquet())
    panel = pd.read_parquet(_pilot7_panel())

    if _FORWARD_MODE:
        if len(hb) != len(panel):
            _stop(
                f"forward HB↔pilot7-panel row count mismatch hb={len(hb)} panel={len(panel)}"
            )
        hb_keyed = hb.copy()
        hb_keyed["_bar_date_d"] = pd.to_datetime(hb_keyed["bar_date"]).dt.date
        hb_keyed["_breakout_d"] = pd.to_datetime(hb_keyed["breakout_bar_date"]).dt.date
        panel_keyed = panel.copy()
        panel_keyed["_bar_date_d"] = pd.to_datetime(panel_keyed["bar_date"]).dt.date
        panel_keyed["_breakout_d"] = pd.to_datetime(panel_keyed["breakout_bar_date"]).dt.date

        for label, df in (("hb", hb_keyed), ("pilot7_panel", panel_keyed)):
            dups = int(df.duplicated(subset=STABLE_EVENT_KEY_NORM).sum())
            if dups > 0:
                _stop(
                    f"forward: {label} has {dups} duplicate stable_event_key tuples; "
                    "spec §13.2 Q9 requires stable key duplicates = 0"
                )

        merged = hb_keyed.merge(
            panel_keyed[STABLE_EVENT_KEY_NORM + ["earliness_score_pct", "ema_tr_tier"]].rename(
                columns={"ema_tr_tier": "panel_ema_tr_tier"}
            ),
            on=STABLE_EVENT_KEY_NORM,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched_count = int((merged["_merge"] != "both").sum())
        if unmatched_count > 0:
            _stop(
                f"forward HB↔pilot7-panel key-merge unmatched: {unmatched_count} HB rows "
                "had no panel match on stable_event_key 5-tuple"
            )
        # Null earliness_score_pct / panel_ema_tr_tier are allowed: cohort filters
        # (signal_state ∈ {trigger, retest_bounce} ∧ -5 ≤ score ≤ -1) drop those
        # rows naturally. Halt is reserved for true key-mismatch only
        # (Remediation A', ONAY 2026-05-06).
        earliness_null_count = int(merged["earliness_score_pct"].isna().sum())
        panel_tier_null_count = int(merged["panel_ema_tr_tier"].isna().sum())
        null_mask = merged["earliness_score_pct"].isna()
        diag_by_signal_state: dict = {}
        diag_by_track: dict = {}
        if null_mask.any():
            if "signal_state" in merged.columns:
                diag_by_signal_state = {
                    str(k): int(v)
                    for k, v in merged.loc[null_mask, "signal_state"]
                    .value_counts(dropna=False)
                    .to_dict()
                    .items()
                }
            if "track" in panel_keyed.columns:
                tr = panel_keyed[STABLE_EVENT_KEY_NORM + ["track"]]
                tmp = merged.loc[null_mask, STABLE_EVENT_KEY_NORM].merge(
                    tr, on=STABLE_EVENT_KEY_NORM, how="left"
                )
                diag_by_track = {
                    str(k): int(v)
                    for k, v in tmp["track"].value_counts(dropna=False).to_dict().items()
                }
        _FORWARD_MERGE_DIAG.clear()
        _FORWARD_MERGE_DIAG.update(
            {
                "hb_rows": int(len(hb_keyed)),
                "panel_rows": int(len(panel_keyed)),
                "merge_indicator": {
                    str(k): int(v)
                    for k, v in merged["_merge"]
                    .astype(str)
                    .value_counts()
                    .to_dict()
                    .items()
                },
                "unmatched_count": unmatched_count,
                "earliness_null_count": earliness_null_count,
                "panel_ema_tr_tier_null_count": panel_tier_null_count,
                "null_distribution_by_signal_state": diag_by_signal_state,
                "null_distribution_by_track": diag_by_track,
            }
        )
        merged.drop(columns=["_merge"], inplace=True)

        out = merged[
            [
                "ticker",
                "bar_date",
                "signal_state",
                "entry_reference_price",
                "invalidation_level",
                "initial_risk_pct",
                "realized_R_10d",
                "common__atr_14",
            ]
        ].copy()
        out["earliness_score_pct"] = merged["earliness_score_pct"].values
        out["panel_ema_tr_tier"] = merged["panel_ema_tr_tier"].values
    else:
        if len(hb) != len(panel):
            _stop("HB ↔ Pilot 7 panel row count mismatch on positional join")
        if not (hb["ticker"].values == panel["ticker"].values).all():
            _stop("HB ↔ Pilot 7 panel ticker mismatch on positional join")
        if not (hb["signal_state"].values == panel["signal_state"].values).all():
            _stop("HB ↔ Pilot 7 panel signal_state mismatch on positional join")

        out = hb[
            [
                "ticker",
                "bar_date",
                "signal_state",
                "entry_reference_price",
                "invalidation_level",
                "initial_risk_pct",
                "realized_R_10d",
                "common__atr_14",
            ]
        ].copy()
        out["earliness_score_pct"] = panel["earliness_score_pct"].values
        out["panel_ema_tr_tier"] = panel["ema_tr_tier"].values

    # Re-derive ema_tr_tier from earliness_score_pct directly (spec §10 — do not
    # blindly copy panel labels in case panel changes; cross-check on mismatch).
    def _derive_tier(state: str, score) -> str:
        if state not in TR_STATES:
            return "OUT_OF_SCOPE"
        if pd.isna(score):
            return "OUT_OF_SCOPE"
        if score < DROP_BOUNDARY:
            return "DROP_EARLY_DIAGNOSTIC"
        if EVENT_NEAR_LO <= score <= EVENT_NEAR_HI:
            return "EVENT_NEAR_TIER"
        if score > EVENT_NEAR_HI:
            return "NON_EVENT_NEAR"
        return "OUT_OF_SCOPE"

    out["ema_tr_tier"] = [
        _derive_tier(s, e) for s, e in zip(out["signal_state"], out["earliness_score_pct"])
    ]

    # Cross-check vs panel labels — STOP on mismatch.
    diff = out[out["ema_tr_tier"] != out["panel_ema_tr_tier"]]
    if len(diff) > 0:
        cols = ["ticker", "bar_date", "signal_state", "earliness_score_pct",
                "ema_tr_tier", "panel_ema_tr_tier"]
        sample = diff[cols].head(5).to_dict("records")
        _stop(
            f"cohort cross-check failed — {len(diff)} rows where re-derived tier "
            f"differs from Pilot 7 panel `ema_tr_tier`. Sample: {sample}"
        )

    out["bar_date"] = pd.to_datetime(out["bar_date"]).dt.date

    return out


def load_daily_ohlcv() -> pd.DataFrame:
    """1h master → daily OHLCV via the canonical Pilot 5/6/7 adapter."""
    bars = pd.read_parquet(
        OHLCV_MASTER,
        columns=["ticker", "ts_istanbul", "open", "high", "low", "close", "volume"],
    )
    bars["ts_istanbul"] = pd.to_datetime(bars["ts_istanbul"])
    daily = intraday_1h.daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    return daily


# ---------------------------------------------------------------------------
# trading-day arithmetic per ticker


def build_ticker_calendar(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cal = {}
    for tk, sub in daily.groupby("ticker", sort=False):
        cal[tk] = sub.reset_index(drop=True)
    return cal


def next_open_lookup(cal, ticker, after):
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None, None
    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(after), side="right")
    if idx >= len(sub):
        return None, None
    return sub.iloc[idx]["date"], sub.iloc[idx]["open"]


def nth_trading_day_close(cal, ticker, start, n):
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None, None
    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(start), side="left")
    target = idx + n
    if target >= len(sub):
        return None, None
    return sub.iloc[target]["date"], sub.iloc[target]["close"]


# ---------------------------------------------------------------------------
# core trade-log build (Lane 2)


def build_trade_log(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    asof,
    ema_daily_index: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Trigger/retest scope only. Cohort routing per spec §4.2:
      EVENT_NEAR_TIER       → OPEN/CLOSED counted paper trade
      NON_EVENT_NEAR        → SKIPPED, skip_reason='NON_EVENT_NEAR'
      DROP_EARLY_DIAGNOSTIC → SKIPPED, skip_reason='DROP_EARLY_DIAGNOSTIC'
      OUT_OF_SCOPE          → omitted from trade table (census only).

    Validity revision (LOCKED + §14 ONAY):
      - For EVENT_NEAR_TIER candidates that would be emitted as OPEN/CLOSED, the
        five validity columns + ema_signature_id are computed via
        `attach_validity_fields(...)`. Fail-fast on missing inputs.
      - R1 (stale-signature anti-repeat) and R2 (open-duplicate guard) are
        applied via `apply_r1_r2_filter(...)`. Block counters are returned in
        `validity_stats` for manifest emission.
      - SKIPPED rows (NON_EVENT_NEAR / DROP_EARLY_DIAGNOSTIC / OUT_OF_SCOPE /
        next_open_unavailable) carry NULL validity columns and are NOT subject
        to R1/R2.
    """
    if ema_daily_index is None:
        _stop(
            "build_trade_log: ema_daily_index is required (validity revision "
            "LOCKED 2026-05-04 + §14 ONAY). Load via load_ema_daily_index()."
        )

    # Filter to trigger/retest universe (OUT_OF_SCOPE rows excluded from trade log).
    sig = signals[signals["signal_state"].isin(TR_STATES)].copy()

    # Deterministic iteration order — required for R1/R2 to be stable across
    # runs. Earlier (ticker, bar_date) wins; later duplicates are blocked.
    sig = sig.sort_values(["ticker", "bar_date"], kind="stable").reset_index(drop=True)

    cal = build_ticker_calendar(daily)

    # Validity counters per §2.1.7 / §13.3.
    r1_blocks = 0
    r2_blocks = 0
    signal_date_unavailable_halts = 0

    _NULL_VALIDITY = dict(
        paper_valid_from=None,
        paper_valid_until=None,
        paper_signal_age=None,
        paper_expired_flag=None,
        ema_signature_id=None,
    )

    rows: list[dict] = []
    for _, ev in sig.iterrows():
        ticker = ev["ticker"]
        bar_date = ev["bar_date"]
        tier = ev["ema_tr_tier"]
        earliness = ev["earliness_score_pct"]
        signal_state = ev["signal_state"]

        rps = (
            float(ev["entry_reference_price"]) - float(ev["invalidation_level"])
            if pd.notna(ev["entry_reference_price"])
            and pd.notna(ev["invalidation_level"])
            else np.nan
        )

        base = dict(
            asof_date=bar_date,
            ticker=ticker,
            line=LINE_LABEL,
            lane_source="close_confirmed",
            preview_seen_17=False,
            close_confirmed=True,
            signal_state=signal_state,
            ema_tr_tier=tier,
            earliness_score_pct=earliness,
            entry_reference_price=ev["entry_reference_price"],
            invalidation_level=ev["invalidation_level"],
            risk_per_share=rps,
        )

        # NON_EVENT_NEAR — SKIPPED, warning only; NULL validity, no R1/R2.
        if tier == "NON_EVENT_NEAR":
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="NON_EVENT_NEAR",
                ) | _NULL_VALIDITY
            )
            continue

        # DROP_EARLY_DIAGNOSTIC — SKIPPED, diagnostic census only, NO trade rule.
        if tier == "DROP_EARLY_DIAGNOSTIC":
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="DROP_EARLY_DIAGNOSTIC",
                ) | _NULL_VALIDITY
            )
            continue

        # OUT_OF_SCOPE in trigger/retest — defensive (NaN earliness only here);
        # tag as SKIPPED with reason rather than silently dropping.
        if tier == "OUT_OF_SCOPE":
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="OUT_OF_SCOPE",
                ) | _NULL_VALIDITY
            )
            continue

        # EVENT_NEAR_TIER — try to build paper trade.
        entry_date, entry_price = next_open_lookup(cal, ticker, bar_date)
        if entry_date is None or pd.isna(entry_price):
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="next_open_unavailable",
                ) | _NULL_VALIDITY
            )
            continue

        # Compute validity + signature for the candidate (fail-fast inside).
        validity = attach_validity_fields(
            ticker=str(ticker),
            source_line=SOURCE_LINE,
            signal_state=str(signal_state),
            bar_date=bar_date,
            earliness_score_pct=earliness,
            cal=cal,
            ema_daily_index=ema_daily_index,
        )

        # R1 + R2 against in-progress log. Per §2.1.6 R1 and R2 are independent
        # filters: either suffices to block, both can fire on the same candidate
        # and increment independently.
        allow, r1_fired, r2_fired = apply_r1_r2_filter(
            candidate_signature_id=validity["ema_signature_id"],
            candidate_ticker=str(ticker),
            in_progress_log=rows,
            source_line=SOURCE_LINE,
        )
        if r1_fired:
            r1_blocks += 1
        if r2_fired:
            r2_blocks += 1
        if not allow:
            continue

        validity_columns = {
            "paper_valid_from": validity["paper_valid_from"],
            "paper_valid_until": validity["paper_valid_until"],
            "paper_signal_age": validity["paper_signal_age"],
            "paper_expired_flag": validity["paper_expired_flag"],
            "ema_signature_id": validity["ema_signature_id"],
        }

        exit_date, exit_price = nth_trading_day_close(
            cal, ticker, entry_date, HORIZON_TRADING_DAYS
        )
        if exit_date is None or pd.isna(exit_price):
            rows.append(
                base | dict(
                    entry_date=entry_date,
                    entry_price=float(entry_price),
                    exit_date=None,
                    exit_price=np.nan,
                    status="OPEN",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason=None,
                ) | validity_columns
            )
            continue

        # CLOSED — both entry and exit observed.
        gross = float(exit_price) / float(entry_price) - 1.0
        net = gross - COST_ROUNDTRIP_FRAC
        realized_R_paper = (
            (float(exit_price) - float(entry_price)) / rps
            if rps and rps > 0
            else np.nan
        )

        rows.append(
            base | dict(
                entry_date=entry_date,
                entry_price=float(entry_price),
                exit_date=exit_date,
                exit_price=float(exit_price),
                status="CLOSED",
                gross_return_pct=gross,
                net_return_pct=net,
                realized_R_paper=realized_R_paper,
                skip_reason=None,
            ) | validity_columns
        )

    log = pd.DataFrame(rows)

    # Backfill flag — events whose +10d window is past asof are backfilled
    # historical observations; events whose entry is on/after asof are
    # forward-clock paper trades.
    log["is_backfill"] = log["entry_date"].apply(
        lambda d: bool(d is not None and d <= asof)
    )

    validity_stats = dict(
        validity_revision_applied=True,
        validity_window_default=VALIDITY_WINDOW_DEFAULT,
        r1_stale_signature_blocks_count=int(r1_blocks),
        r2_open_duplicate_blocks_count=int(r2_blocks),
        signal_date_unavailable_halts=int(signal_date_unavailable_halts),
    )

    return log, validity_stats


# ---------------------------------------------------------------------------
# §5 view materialization (boolean flags)


def materialize_views(log: pd.DataFrame) -> pd.DataFrame:
    in_raw = (log["ema_tr_tier"] == "EVENT_NEAR_TIER") & (
        log["status"].isin(["OPEN", "CLOSED"])
    )
    log["in_view_raw_all_candidates"] = in_raw

    in_capped = pd.Series(False, index=log.index)
    for asof_dt, idx in log[in_raw].groupby("asof_date").groups.items():
        sub = log.loc[idx].sort_values("ticker", kind="stable")
        keep = sub.head(MAX_NEW_POSITIONS_PER_DAY).index
        in_capped.loc[keep] = True
    log["in_view_capped_portfolio"] = in_capped

    return log


# ---------------------------------------------------------------------------
# §8 metrics


def compute_view_metrics(log: pd.DataFrame, view_col: str) -> dict:
    sub = log[log[view_col]].copy()
    closed = sub[sub["status"] == "CLOSED"]
    open_ct = int((sub["status"] == "OPEN").sum())

    if len(closed) == 0:
        return {
            "n_closed": 0,
            "n_open": open_ct,
            "avg_gross_return_pct": None,
            "median_gross_return_pct": None,
            "avg_net_return_pct": None,
            "median_net_return_pct": None,
            "avg_R_paper": None,
            "median_R_paper": None,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_gross": None,
        }

    gross = closed["gross_return_pct"].astype(float)
    net = closed["net_return_pct"].astype(float)
    R = closed["realized_R_paper"].astype(float)

    pos = gross[gross > 0].sum()
    neg = -gross[gross < 0].sum()
    pf = float(pos / neg) if neg > 0 else float("inf") if pos > 0 else None

    eq = (
        closed.sort_values(["exit_date", "ticker"])["gross_return_pct"]
        .astype(float)
        .cumsum()
        + 1.0
    )
    rolling_max = eq.cummax()
    dd = float((eq / rolling_max - 1).min()) if len(eq) else None

    return {
        "n_closed": int(len(closed)),
        "n_open": open_ct,
        "avg_gross_return_pct": float(gross.mean()),
        "median_gross_return_pct": float(gross.median()),
        "avg_net_return_pct": float(net.mean()),
        "median_net_return_pct": float(net.median()),
        "avg_R_paper": float(R.mean(skipna=True)) if R.notna().any() else None,
        "median_R_paper": float(R.median(skipna=True)) if R.notna().any() else None,
        "win_rate": float((gross > 0).mean()),
        "profit_factor": pf,
        "max_drawdown_gross": dd,
    }


def trigger_retest_split(log: pd.DataFrame) -> dict:
    """Descriptive trigger-only and retest_bounce-only summary on EVENT_NEAR_TIER
    closed trades (raw view). For monitoring drift only — NOT a retest-only
    claim (Pilot 7 G5 retest INSUFFICIENT)."""
    sub = log[
        (log["in_view_raw_all_candidates"]) & (log["status"] == "CLOSED")
    ].copy()
    out = {}
    for state in ("trigger", "retest_bounce"):
        s = sub[sub["signal_state"] == state]
        if len(s) == 0:
            out[state] = {"n_closed": 0}
            continue
        gross = s["gross_return_pct"].astype(float)
        R = s["realized_R_paper"].astype(float)
        pos = gross[gross > 0].sum()
        neg = -gross[gross < 0].sum()
        pf = float(pos / neg) if neg > 0 else float("inf") if pos > 0 else None
        out[state] = {
            "n_closed": int(len(s)),
            "avg_gross_return_pct": float(gross.mean()),
            "median_gross_return_pct": float(gross.median()),
            "avg_R_paper": float(R.mean(skipna=True)) if R.notna().any() else None,
            "win_rate": float((gross > 0).mean()),
            "profit_factor": pf,
        }
    return out


def daily_summary(log: pd.DataFrame) -> pd.DataFrame:
    g = log.groupby("asof_date", group_keys=False)
    out = pd.DataFrame(
        dict(
            n_candidates=g.size(),
            n_event_near=g.apply(
                lambda d: int((d["ema_tr_tier"] == "EVENT_NEAR_TIER").sum())
            ),
            n_non_event_near_skipped=g.apply(
                lambda d: int((d["ema_tr_tier"] == "NON_EVENT_NEAR").sum())
            ),
            n_drop_early_diagnostic_skipped=g.apply(
                lambda d: int((d["ema_tr_tier"] == "DROP_EARLY_DIAGNOSTIC").sum())
            ),
            n_open=g.apply(lambda d: int((d["status"] == "OPEN").sum())),
            n_closed=g.apply(lambda d: int((d["status"] == "CLOSED").sum())),
            n_skipped_no_open=g.apply(
                lambda d: int((d["skip_reason"] == "next_open_unavailable").sum())
            ),
            avg_gross_return_pct=g.apply(
                lambda d: float(
                    d.loc[d["status"] == "CLOSED", "gross_return_pct"].mean()
                )
                if (d["status"] == "CLOSED").any()
                else np.nan
            ),
        )
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# Lane 1 preview audit (operational only — no perf accounting)


def emit_preview_audit_stub() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "asof_date",
            "ticker",
            "preview_state",
            "close_state",
            "preview_ema_tr_tier",
            "close_ema_tr_tier",
            "preview_to_close_status",
        ]
    )


# ---------------------------------------------------------------------------
# summary md


def emit_summary_md(
    log: pd.DataFrame,
    raw_metrics: dict,
    capped_metrics: dict,
    tr_split: dict,
    manifest: dict,
    asof,
) -> str:
    n_total = len(log)
    n_event_near = int((log["ema_tr_tier"] == "EVENT_NEAR_TIER").sum())
    n_non_event = int((log["ema_tr_tier"] == "NON_EVENT_NEAR").sum())
    n_drop = int((log["ema_tr_tier"] == "DROP_EARLY_DIAGNOSTIC").sum())
    n_oos = int((log["ema_tr_tier"] == "OUT_OF_SCOPE").sum())
    n_open = int((log["status"] == "OPEN").sum())
    n_closed = int((log["status"] == "CLOSED").sum())
    n_skipped = int((log["status"] == "SKIPPED").sum())
    n_backfill = int(log["is_backfill"].sum())
    n_forward = n_total - n_backfill

    parts: list[str] = []
    parts.append("# paper_execution_v0_trigger_retest — Run Summary (Line TR)\n")
    parts.append(
        "- Spec: `memory/paper_execution_v0_trigger_retest_spec.md` LOCKED 2026-05-04, ONAY 2026-05-04\n"
    )
    parts.append(f"- Run timestamp: {manifest['run_started_utc']}\n")
    parts.append(f"- asof: {asof}\n")
    parts.append(
        f"- Mode: {manifest['mode']}  |  Line: **{LINE_LABEL}** (separate from Line E `paper_execution_v0`)\n"
    )
    parts.append(
        f"- Return-unit choice: {manifest['return_unit']}\n"
    )
    parts.append("\n## Cohort census (trigger ∪ retest_bounce universe)\n")
    parts.append(f"- Trade-log rows: {n_total}\n")
    parts.append(f"- EVENT_NEAR_TIER: {n_event_near}\n")
    parts.append(f"- NON_EVENT_NEAR (SKIPPED, warning only): {n_non_event}\n")
    parts.append(
        f"- DROP_EARLY_DIAGNOSTIC (SKIPPED, diagnostic census only, NO trade rule): {n_drop}\n"
    )
    parts.append(f"- OUT_OF_SCOPE (NaN earliness): {n_oos}\n")
    parts.append(
        f"- Status: CLOSED={n_closed} / OPEN={n_open} / SKIPPED={n_skipped}\n"
    )
    parts.append(f"- Backfilled historical: {n_backfill}  |  Forward-clock: {n_forward}\n")

    parts.append("\n## Lane discipline\n")
    parts.append(
        "- Lane 1 (17:00 preview) — operational pre-scan only, NO performance accounting.\n"
    )
    parts.append(
        "- Lane 2 (close-confirmed) — canonical paper signal source; **all** trade-log rows here are `lane_source=close_confirmed`.\n"
    )
    parts.append(
        "- Preview infra not yet wired for first emission; `preview_seen_17 = False` on all rows; preview_audit emitted as empty schema-only file.\n"
    )

    parts.append("\n## §5.1 Primary view — `raw_all_candidates` (HEADLINE)\n")
    parts.append(
        "\n*Every close-confirmed EVENT_NEAR_TIER event, no cap. Equal-notional shadow.*\n\n"
    )
    for k, v in raw_metrics.items():
        if isinstance(v, float):
            parts.append(f"- **{k}**: {v:.4f}\n")
        else:
            parts.append(f"- **{k}**: {v}\n")

    parts.append("\n## §5.2 Operational secondary view — `capped_portfolio`\n")
    parts.append(
        "\n*Operational triage only — NOT alpha ranking. max_new_positions_per_day=5, selection rule = ascending alphabetical ticker.*\n"
    )
    parts.append(
        "*If this view looks better than `raw_all_candidates`, that is selection artifact / sample-size effect — NOT a justification to cap in any future spec (spec §5.2).*\n\n"
    )
    for k, v in capped_metrics.items():
        if isinstance(v, float):
            parts.append(f"- {k}: {v:.4f}\n")
        else:
            parts.append(f"- {k}: {v}\n")

    parts.append(
        "\n## Trigger vs retest_bounce descriptive split (NOT a retest-only claim)\n"
    )
    parts.append(
        "\n*Pilot 7 G5 retest-only INSUFFICIENT — descriptive monitoring only, no promotion / no sub-tiering.*\n\n"
    )
    for state in ("trigger", "retest_bounce"):
        parts.append(f"- **{state}**:\n")
        for k, v in tr_split[state].items():
            if isinstance(v, float):
                parts.append(f"  - {k}: {v:.4f}\n")
            else:
                parts.append(f"  - {k}: {v}\n")

    parts.append("\n## Locked constants (spec §1.2 / §4 / §5 / §6)\n")
    parts.append(
        f"- event-near window: {EVENT_NEAR_LO} ≤ earliness_score_pct ≤ {EVENT_NEAR_HI} (LOCKED)\n"
    )
    parts.append(f"- drop boundary: earliness_score_pct < {DROP_BOUNDARY} → DROP_EARLY_DIAGNOSTIC (LOCKED)\n")
    parts.append(f"- horizon: {HORIZON_TRADING_DAYS} trading days (LOCKED)\n")
    parts.append(
        f"- cost: {COST_BPS_ENTRY} bps entry + {COST_BPS_EXIT} bps exit = "
        f"{COST_ROUNDTRIP_FRAC*100:.2f}% round-trip (LOCKED, percentage view only)\n"
    )
    parts.append(
        f"- max_new_positions_per_day: {MAX_NEW_POSITIONS_PER_DAY} (capped view, LOCKED)\n"
    )
    parts.append(
        "- selection rule (capped view): ascending alphabetical ticker (LOCKED, operational, not alpha)\n"
    )
    parts.append("- entry: next trading-day open after close-confirmed signal (LOCKED)\n")
    parts.append(
        f"- exit: close at +{HORIZON_TRADING_DAYS} trading days from entry (LOCKED)\n"
    )
    parts.append("- TP / SL / trailing: NONE (LOCKED)\n")
    parts.append(
        "- realized_R_paper: numerator = (paper_exit − paper_entry); denominator = "
        "frozen R_per_share = entry_reference_price − invalidation_level (reused from HB parquet). "
        "Paper R is denominator-comparable to Pilot 7 historical realized_R_10d "
        "but NOT numerator-identical (paper entry = next-open ≠ historical signal-day entry).\n"
    )
    parts.append(
        f"- line invariant: every trade-log row has `line == \"{LINE_LABEL}\"`; tool refuses to overwrite Line E `paper_execution_v0_*` files.\n"
    )

    parts.append("\n## Forbidden interpretations (spec §13.2)\n")
    parts.append("- ❌ EMA is a live buy signal\n")
    parts.append("- ❌ NON_EVENT_NEAR is a short / avoid signal\n")
    parts.append("- ❌ DROP_EARLY_DIAGNOSTIC is bad\n")
    parts.append("- ❌ Trigger/retest should be pooled with extended\n")
    parts.append("- ❌ Retest-only edge is proven\n")
    parts.append("- ❌ EMA should rank scanner output\n")
    parts.append("- ❌ This can be traded live now\n")
    parts.append("- ❌ 17:00 preview is an entry trigger\n")
    parts.append("- ❌ Pilot 7 historical result guarantees forward performance\n")
    parts.append("- ❌ Line TR > Line E or Line E > Line TR (cross-line comparison is OUT)\n")
    parts.append("- ❌ `capped_portfolio` is the better strategy\n")
    parts.append(
        "- ❌ Cascade to decision_engine / NOX / production HTML without per-target ONAY\n"
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["close_confirmed", "preview"],
        default="close_confirmed",
    )
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default = today TR)")
    ap.add_argument(
        "--forward",
        action="store_true",
        help="Forward mode (LOCKED 2026-05-06): use refreshed HB + "
             "ema_context_pilot7_panel_forward.parquet and stable_event_key "
             "5-tuple key-based merge. Locked-research path is unchanged when "
             "--forward is omitted.",
    )
    args = ap.parse_args()

    global _FORWARD_MODE
    _FORWARD_MODE = bool(args.forward)

    t0 = time.time()
    print(
        f"[paper_execution_v0_trigger_retest] mode={args.mode} forward={_FORWARD_MODE} starting..."
    )

    _assert_no_line_e_collision()
    print("[paper_execution_v0_trigger_retest] line invariant + no Line E collision OK")

    info = boot_integrity()
    print("[paper_execution_v0_trigger_retest] boot integrity OK")

    asof = (
        pd.Timestamp(args.asof).date()
        if args.asof
        else pd.Timestamp.now(tz="Europe/Istanbul").date()
    )

    if args.mode == "preview":
        print("[paper_execution_v0_trigger_retest] preview mode: writing empty audit stub")
        audit = emit_preview_audit_stub()
        audit.to_csv(OUT_PREVIEW_AUDIT, index=False)
        print(
            f"[paper_execution_v0_trigger_retest] wrote {OUT_PREVIEW_AUDIT.relative_to(ROOT)}"
        )
        print(f"[paper_execution_v0_trigger_retest] done in {time.time() - t0:.1f}s")
        return

    # Lane 2 — close confirmed
    signals = load_signals()
    print(
        f"[paper_execution_v0_trigger_retest] HB trigger/retest signals: {len(signals)}"
    )

    daily = load_daily_ohlcv()
    print(
        f"[paper_execution_v0_trigger_retest] daily OHLCV rows: {len(daily)} (resampled from 1h)"
    )

    ema_daily_index = load_ema_daily_index()
    print(
        f"[paper_execution_v0_trigger_retest] ema_context_daily index rows: {len(ema_daily_index)}"
    )

    log, validity_stats = build_trade_log(signals, daily, asof, ema_daily_index)
    log = materialize_views(log)
    print(f"[paper_execution_v0_trigger_retest] trade log rows: {len(log)}")

    # Lane purity invariant.
    bad_lane = log[log["lane_source"] != "close_confirmed"]
    if len(bad_lane) > 0:
        _stop(
            f"lane purity violation — {len(bad_lane)} rows with lane_source != close_confirmed"
        )

    # Line invariant.
    bad_line = log[log["line"] != LINE_LABEL]
    if len(bad_line) > 0:
        _stop(
            f"line invariant violation — {len(bad_line)} rows with line != {LINE_LABEL}"
        )

    raw_metrics = compute_view_metrics(log, "in_view_raw_all_candidates")
    capped_metrics = compute_view_metrics(log, "in_view_capped_portfolio")
    tr_split = trigger_retest_split(log)
    daily_sum = daily_summary(log)

    # Manifest
    manifest = {
        "run_started_utc": _utc_now(),
        "manifest_notes": [
            "First emission. Line TR (trigger/retest) shadow execution per "
            "memory/paper_execution_v0_trigger_retest_spec.md §1.2 + §4. Cohort "
            "labels re-derived from earliness_score_pct AND cross-checked against "
            "Pilot 7 panel `ema_tr_tier`; mismatch would have STOPped the run."
        ],
        "mode": args.mode,
        "asof": str(asof),
        "spec_path": "memory/paper_execution_v0_trigger_retest_spec.md",
        "spec_status": "LOCKED 2026-05-04, ONAY 2026-05-04",
        "line": LINE_LABEL,
        "view_choice": (
            "boolean_flags (in_view_raw_all_candidates / in_view_capped_portfolio)"
        ),
        "primary_view": "raw_all_candidates",
        "secondary_view": "capped_portfolio",
        "selection_rule_capped": (
            "ascending alphabetical ticker (operational, NOT alpha ranking)"
        ),
        "return_unit": (
            "realized_R_paper (numerator: paper-trade prices; denominator: "
            "frozen R_per_share = entry_reference_price - invalidation_level "
            "reused from HB parquet) + percentage gross/net"
        ),
        "cost_assumption_pct": (
            f"{COST_BPS_ENTRY}+{COST_BPS_EXIT} bps round-trip = {COST_ROUNDTRIP_FRAC}"
        ),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "event_near_window_pct": [EVENT_NEAR_LO, EVENT_NEAR_HI],
        "drop_boundary_pct": DROP_BOUNDARY,
        "max_new_positions_per_day": MAX_NEW_POSITIONS_PER_DAY,
        "ohlcv_pipeline": (
            "data.intraday_1h.daily_resample(extfeed_intraday_1h_3y_master.parquet) — "
            "same canonical adapter Pilot 5/6/7 historical labels traced through; "
            "explicitly recorded per spec §6.2 (no silent substitution)"
        ),
        "non_event_near_handling": (
            "SKIPPED, displayed in trade log, NOT contra-signal, NOT paper-traded"
        ),
        "drop_early_diagnostic_handling": (
            "SKIPPED, diagnostic census only, NOT paper-traded, NOT warning, "
            "NOT added back into baseline"
        ),
        "lane_purity_check": "passed (all rows lane_source=close_confirmed)",
        "line_invariant_check": f"passed (all rows line={LINE_LABEL})",
        "line_e_collision_check": "passed (no overlap with paper_execution_v0_* files)",
        "boot_integrity": info,
        "forward_merge_diag": dict(_FORWARD_MERGE_DIAG) if _FORWARD_MODE else None,
        "trade_log_rows": int(len(log)),
        "cohort_counts": {
            "EVENT_NEAR_TIER": int((log["ema_tr_tier"] == "EVENT_NEAR_TIER").sum()),
            "NON_EVENT_NEAR": int((log["ema_tr_tier"] == "NON_EVENT_NEAR").sum()),
            "DROP_EARLY_DIAGNOSTIC": int(
                (log["ema_tr_tier"] == "DROP_EARLY_DIAGNOSTIC").sum()
            ),
            "OUT_OF_SCOPE": int((log["ema_tr_tier"] == "OUT_OF_SCOPE").sum()),
        },
        "trigger_count": int((log["signal_state"] == "trigger").sum()),
        "retest_bounce_count": int((log["signal_state"] == "retest_bounce").sum()),
        "status_distribution": log["status"].value_counts().to_dict(),
        "primary_view_metrics": raw_metrics,
        "secondary_view_metrics": capped_metrics,
        "trigger_retest_descriptive_split": tr_split,
        # Validity revision (LOCKED 2026-05-04 + §14 ONAY) — five additive fields.
        **validity_stats,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(
        f"[paper_execution_v0_trigger_retest] wrote {OUT_MANIFEST.relative_to(ROOT)}"
    )

    log.to_parquet(OUT_TRADES_PARQUET, index=False)
    log.to_csv(OUT_TRADES_CSV, index=False)
    print(
        f"[paper_execution_v0_trigger_retest] wrote {OUT_TRADES_PARQUET.relative_to(ROOT)} ({len(log)} rows)"
    )
    print(f"[paper_execution_v0_trigger_retest] wrote {OUT_TRADES_CSV.relative_to(ROOT)}")

    daily_sum.to_csv(OUT_DAILY_SUMMARY, index=False)
    print(
        f"[paper_execution_v0_trigger_retest] wrote {OUT_DAILY_SUMMARY.relative_to(ROOT)}"
    )

    audit = emit_preview_audit_stub()
    audit.to_csv(OUT_PREVIEW_AUDIT, index=False)
    print(
        f"[paper_execution_v0_trigger_retest] wrote {OUT_PREVIEW_AUDIT.relative_to(ROOT)} (empty stub)"
    )

    md = emit_summary_md(log, raw_metrics, capped_metrics, tr_split, manifest, asof)
    OUT_SUMMARY_MD.write_text(md)
    print(f"[paper_execution_v0_trigger_retest] wrote {OUT_SUMMARY_MD.relative_to(ROOT)}")

    print(f"[paper_execution_v0_trigger_retest] done in {time.time() - t0:.1f}s")
    print("[paper_execution_v0_trigger_retest] PRIMARY VIEW headline (raw_all_candidates):")
    print(
        f"  n_closed={raw_metrics['n_closed']} / n_open={raw_metrics['n_open']}"
    )
    if raw_metrics["avg_R_paper"] is not None:
        print(
            f"  avg_R_paper={raw_metrics['avg_R_paper']:.4f} / "
            f"median_R_paper={raw_metrics['median_R_paper']:.4f}"
        )
        print(
            f"  win_rate={raw_metrics['win_rate']:.4f} / "
            f"PF={raw_metrics['profit_factor']:.4f}"
        )


if __name__ == "__main__":
    main()
