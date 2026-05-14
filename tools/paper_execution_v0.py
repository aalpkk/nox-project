"""paper_execution_v0 — forward paper/shadow execution for EMA-HB TIER_A_PAPER.

Spec: memory/paper_execution_v0_spec.md (LOCKED 2026-05-03, ONAY 2026-05-04).

This is PAPER / SHADOW only. NOT live trading. NOT a live entry filter.
NOT a position sizing rule. NOT a ranking signal.

Two-lane architecture:
  --mode close_confirmed (Lane 2, canonical paper signal source)
  --mode preview         (Lane 1, operational pre-scan only — no perf accounting)

Eligibility: extended HB events with earliness_score_pct <= -6.
Entry: next trading-day open after close-confirmed signal day.
Exit:  +10 trading-day close. No TP/SL/trailing.

Two views materialized via boolean flags:
  in_view_raw_all_candidates  — primary, no cap, all close-confirmed TIER_A
  in_view_capped_portfolio    — operational, max 5/day, alphabetical ticker

Frozen risk denominator: R_per_share = entry_reference_price - invalidation_level
(reused from HB event parquet to make realized_R_paper denominator-comparable to
Pilot 5/6 historical realized_R_10d denominator). Numerator uses paper-trade
prices (next-open entry, +10d close exit) so paper R != historical R numerically;
only the risk basis is reused.

Anti-tweak: spec is LOCKED end-to-end. Bug affecting eligibility/return → void
+ rerun + manifest note. No threshold/horizon/cost/selection sweep.
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

EARLY_THRESHOLD = -6.0  # spec §4.1, LOCKED
HORIZON_TRADING_DAYS = 10  # spec §4.3, LOCKED
COST_BPS_ENTRY = 10.0  # spec §4.6
COST_BPS_EXIT = 10.0  # spec §4.6
COST_ROUNDTRIP_FRAC = (COST_BPS_ENTRY + COST_BPS_EXIT) / 10000.0  # 0.0020
MAX_NEW_POSITIONS_PER_DAY = 5  # spec §5.2

EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470

# Validity revision (LOCKED 2026-05-04 + ONAY §14 additive clarification).
SOURCE_LINE = "EXTENDED"
VALIDITY_WINDOW_DEFAULT = "next_trading_day"
SIGNATURE_HEX_LEN = 16
WIDTH_ROUND_DECIMALS = 6

# ---------------------------------------------------------------------------
# paths

HB_PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
EARLINESS_PARQUET = ROOT / "output/ema_context_pilot4_earliness_per_event.parquet"
PILOT6_PANEL = ROOT / "output/ema_context_pilot6_panel.parquet"
PILOT5_PANEL = ROOT / "output/ema_context_pilot5_panel.parquet"
OHLCV_MASTER = ROOT / "output/extfeed_intraday_1h_3y_master.parquet"
EMA_CONTEXT_DAILY = ROOT / "output/ema_context_daily.parquet"

# Forward mode (LOCKED 2026-05-06 paper_execution_v0_forward_run_spec.md):
# When --forward is set, swap input paths to forward EMA artifacts and replace
# positional join with stable_event_key 5-tuple key-based merge.
_FORWARD_MODE = False
EARLINESS_PARQUET_FORWARD = ROOT / "output/ema_context_pilot4_earliness_per_event_forward.parquet"
PILOT6_PANEL_FORWARD = ROOT / "output/ema_context_pilot6_panel_forward.parquet"
PILOT5_PANEL_FORWARD = ROOT / "output/ema_context_pilot5_panel_forward.parquet"
STABLE_EVENT_KEY_NORM = ["ticker", "_bar_date_d", "setup_family", "signal_type", "_breakout_d"]
_FORWARD_MERGE_DIAG: dict = {}


def _hb_parquet() -> Path:
    return HB_PARQUET


def _earliness_parquet() -> Path:
    return EARLINESS_PARQUET_FORWARD if _FORWARD_MODE else EARLINESS_PARQUET


def _pilot6_panel() -> Path:
    return PILOT6_PANEL_FORWARD if _FORWARD_MODE else PILOT6_PANEL


def _pilot5_panel() -> Path:
    return PILOT5_PANEL_FORWARD if _FORWARD_MODE else PILOT5_PANEL


OUT_TRADES_PARQUET = ROOT / "output/paper_execution_v0_trades.parquet"
OUT_TRADES_CSV = ROOT / "output/paper_execution_v0_trades.csv"
OUT_DAILY_SUMMARY = ROOT / "output/paper_execution_v0_daily_summary.csv"
OUT_PREVIEW_AUDIT = ROOT / "output/paper_execution_v0_preview_audit.csv"
OUT_SUMMARY_MD = ROOT / "output/paper_execution_v0_summary.md"
OUT_MANIFEST = ROOT / "output/paper_execution_v0_manifest.json"

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
    print(f"[paper_execution_v0] STOP — {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Validity revision helpers (LOCKED 2026-05-04 + §14 ONAY clarification).
# Spec: memory/paper_execution_v0_validity_revision_spec.md §3, §4, §5, §13, §14.


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


# ---------------------------------------------------------------------------
# §6.2 boot integrity


def boot_integrity() -> dict:
    """Validates required artifacts (spec §6.1+§6.2). Aborts via STOP on any
    failure. Returns dict of resolved paths + hashes for manifest emission.

    Forward mode (LOCKED 2026-05-06): paths are resolved via _hb_parquet() /
    _earliness_parquet() / _pilot6_panel() / _pilot5_panel(); locked-research
    `EXPECTED_HB_ROWS=10470` row check is skipped and replaced by HB↔earliness
    parity (both must equal whatever the refreshed-HB universe row count is).
    """
    info = {"checked_at": _utc_now(), "forward_mode": bool(_FORWARD_MODE)}

    hb_path = _hb_parquet()
    earl_path = _earliness_parquet()
    pilot6_path = _pilot6_panel()
    pilot5_path = _pilot5_panel()

    for label, path in [
        ("hb_event_parquet", hb_path),
        ("earliness_parquet", earl_path),
        ("ohlcv_master", OHLCV_MASTER),
        ("ema_context_daily", EMA_CONTEXT_DAILY),
    ]:
        if not path.exists():
            _stop(f"required artifact missing: {label} at {path}")
        info[label + "_path"] = str(path.relative_to(ROOT))
        info[label + "_size_bytes"] = path.stat().st_size

    # Pilot 5/6 panel — at least one must exist
    if pilot6_path.exists():
        info["pilot_panel_path"] = str(pilot6_path.relative_to(ROOT))
        info["pilot_panel_choice"] = "pilot6"
    elif pilot5_path.exists():
        info["pilot_panel_path"] = str(pilot5_path.relative_to(ROOT))
        info["pilot_panel_choice"] = "pilot5"
    else:
        _stop("required artifact missing: neither pilot5 nor pilot6 panel exists")

    # Hashes (size-only for OHLCV due to size; SHA256 for the others is fast)
    info["hb_event_parquet_sha256"] = _file_hash(hb_path)
    info["earliness_parquet_sha256"] = _file_hash(earl_path)
    # OHLCV: hash is required by spec §6.2(4); paid one-time cost.
    print("[paper_execution_v0] hashing OHLCV master (one-time)...")
    info["ohlcv_master_sha256"] = _file_hash(OHLCV_MASTER)

    # Manifest cross-check from HB parquet itself
    hb = pd.read_parquet(hb_path, columns=["ticker", "scanner_version"])
    if (hb["scanner_version"] != EXPECTED_SCANNER_VERSION).any():
        _stop(
            f"hb scanner_version mismatch — expected {EXPECTED_SCANNER_VERSION}, "
            f"observed {hb['scanner_version'].unique().tolist()}"
        )
    if _FORWARD_MODE:
        # Forward mode: row-count parity replaces the locked-research constant.
        earliness_rows = pd.read_parquet(earl_path, columns=["event_id"])
        if len(hb) != len(earliness_rows):
            _stop(
                f"forward HB↔earliness row mismatch — hb={len(hb)} "
                f"earl={len(earliness_rows)}"
            )
        info["hb_event_rows"] = int(len(hb))
        info["earliness_event_rows"] = int(len(earliness_rows))
    else:
        if len(hb) != EXPECTED_HB_ROWS:
            _stop(f"hb row count mismatch — expected {EXPECTED_HB_ROWS}, got {len(hb)}")
        info["hb_event_rows"] = int(EXPECTED_HB_ROWS)
        # Earliness panel row count must match HB (positional join basis)
        earliness = pd.read_parquet(earl_path, columns=["event_id"])
        if len(earliness) != EXPECTED_HB_ROWS:
            _stop(
                f"earliness panel row mismatch — expected {EXPECTED_HB_ROWS}, "
                f"got {len(earliness)}"
            )
    info["hb_scanner_version"] = EXPECTED_SCANNER_VERSION

    return info


# ---------------------------------------------------------------------------
# data loading


def load_signals() -> pd.DataFrame:
    """HB extended events joined to Pilot 4 earliness.

    Default mode: positional join (locked-research, byte-equal).
    Forward mode (LOCKED 2026-05-06): key-based merge on stable_event_key
    5-tuple `(ticker, bar_date, setup_family, signal_type, breakout_bar_date)`
    with `validate="one_to_one"`. Refuses any duplicate or unmatched key.
    """
    hb = pd.read_parquet(_hb_parquet())
    earl = pd.read_parquet(_earliness_parquet())

    if _FORWARD_MODE:
        if len(hb) != len(earl):
            _stop(
                f"forward HB↔earliness row count mismatch hb={len(hb)} earl={len(earl)}"
            )
        hb_keyed = hb.copy()
        hb_keyed["_bar_date_d"] = pd.to_datetime(hb_keyed["bar_date"]).dt.date
        hb_keyed["_breakout_d"] = pd.to_datetime(hb_keyed["breakout_bar_date"]).dt.date
        earl_keyed = earl.copy()
        earl_keyed["_bar_date_d"] = pd.to_datetime(earl_keyed["bar_date"]).dt.date
        earl_keyed["_breakout_d"] = pd.to_datetime(earl_keyed["breakout_bar_date"]).dt.date

        for label, df in (("hb", hb_keyed), ("earliness", earl_keyed)):
            dups = int(df.duplicated(subset=STABLE_EVENT_KEY_NORM).sum())
            if dups > 0:
                _stop(
                    f"forward: {label} has {dups} duplicate stable_event_key tuples; "
                    "spec §13.2 Q9 requires stable key duplicates = 0"
                )

        merged = hb_keyed.merge(
            earl_keyed[STABLE_EVENT_KEY_NORM + ["earliness_score_pct", "earliness_score_atr", "event_id"]],
            on=STABLE_EVENT_KEY_NORM,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched_count = int((merged["_merge"] != "both").sum())
        if unmatched_count > 0:
            _stop(
                f"forward HB↔earliness key-merge unmatched: {unmatched_count} HB rows "
                "had no earliness match on stable_event_key 5-tuple"
            )
        # Null earliness_score_pct is allowed: cohort filters (signal_state ∧ score
        # thresholds) drop those rows naturally. Halt is reserved for true
        # key-mismatch only (Remediation A, ONAY 2026-05-06).
        earliness_null_count = int(merged["earliness_score_pct"].isna().sum())
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
            if "track" in earl_keyed.columns:
                tr = earl_keyed[STABLE_EVENT_KEY_NORM + ["track"]]
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
                "earliness_rows": int(len(earl_keyed)),
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
        out["earliness_score_atr"] = merged["earliness_score_atr"].values
        out["event_id"] = merged["event_id"].values
    else:
        if len(hb) != len(earl):
            _stop("HB ↔ earliness row count mismatch on positional join")
        # Positional alignment audit
        if not (hb["ticker"].values == earl["ticker"].values).all():
            _stop("HB ↔ earliness ticker mismatch on positional join")
        if not (hb["signal_state"].values == earl["signal_state"].values).all():
            _stop("HB ↔ earliness signal_state mismatch on positional join")

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
        out["earliness_score_pct"] = earl["earliness_score_pct"].values
        out["earliness_score_atr"] = earl["earliness_score_atr"].values
        out["event_id"] = earl["event_id"].values

    # bar_date → date object for daily-OHLCV joins
    out["bar_date"] = pd.to_datetime(out["bar_date"]).dt.date

    return out


def load_daily_ohlcv() -> pd.DataFrame:
    """1h master → daily OHLCV via the same canonical adapter Pilot 5/6 used."""
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
    """Per-ticker daily DataFrame (sorted) for fast searchsorted."""
    cal = {}
    for tk, sub in daily.groupby("ticker", sort=False):
        cal[tk] = sub.reset_index(drop=True)
    return cal


def next_open_lookup(
    cal: dict[str, pd.DataFrame], ticker: str, after: pd.Timestamp.date
):
    """Open price on the first trading day strictly > `after` for `ticker`."""
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None, None
    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(after), side="right")
    if idx >= len(sub):
        return None, None
    return sub.iloc[idx]["date"], sub.iloc[idx]["open"]


def nth_trading_day_close(
    cal: dict[str, pd.DataFrame], ticker: str, start: pd.Timestamp.date, n: int
):
    """Close on the nth trading day after `start` (start = entry_date, n=10
    means the close 10 trading days after entry)."""
    sub = cal.get(ticker)
    if sub is None or sub.empty:
        return None, None
    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(start), side="left")
    target = idx + n  # entry is at idx; target close is n trading days later
    if target >= len(sub):
        return None, None
    return sub.iloc[target]["date"], sub.iloc[target]["close"]


# ---------------------------------------------------------------------------
# core trade-log build (Lane 2)


def build_trade_log(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    asof: pd.Timestamp.date,
    ema_daily_index: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One row per HB extended event. Tier label per spec §4.4. Entry =
    next-open after bar_date. Exit = +10 trading-day close from entry. Status:
    SKIPPED (TIER_B / no_next_open / out_of_scope) | OPEN (entry exists, exit
    not yet observable) | CLOSED (entry + exit both observed).

    Validity revision (LOCKED + §14 ONAY):
      - For TIER_A_PAPER candidates that would be emitted as OPEN/CLOSED, the
        five validity columns + ema_signature_id are computed via
        `attach_validity_fields(...)`. Fail-fast on missing inputs.
      - R1 (stale-signature anti-repeat) and R2 (open-duplicate guard) are
        applied via `apply_r1_r2_filter(...)` against the in-progress log.
        Blocked candidates are NOT emitted; counters are returned in
        `validity_stats` for manifest emission.
      - TIER_B / SKIPPED rows carry NULL validity columns and are NOT subject
        to R1/R2 (they are bookkeeping, not paper-trade emissions).
    """
    if ema_daily_index is None:
        _stop(
            "build_trade_log: ema_daily_index is required (validity revision "
            "LOCKED 2026-05-04 + §14 ONAY). Load via load_ema_daily_index()."
        )

    # Filter to extended (eligible universe) + TIER_B (warning, displayed but
    # SKIPPED). OUT_OF_SCOPE rows (trigger / retest_bounce / NaN earliness)
    # do NOT enter the trade log (spec §2.1: "not in paper universe at all").
    sig = signals[signals["signal_state"] == "extended"].copy()

    # Tier assignment
    def _assign_tier(row) -> str:
        e = row["earliness_score_pct"]
        if pd.isna(e):
            return "OUT_OF_SCOPE"
        if e <= EARLY_THRESHOLD:
            return "TIER_A_PAPER"
        return "TIER_B_WARNING"

    sig["ema_tier"] = sig.apply(_assign_tier, axis=1)

    # Signal-vs-outcome semantics fix (LOCKED 2026-05-06):
    # Previously OUT_OF_SCOPE rows (NaN earliness) were dropped here, which
    # blocked current-day signal coverage in the daily report. Per Q-E /
    # Q-D / §13.3 we now retain them and emit as SKIPPED with
    # skip_reason="ema_out_of_scope" inside the loop. Performance metrics
    # already filter status=="CLOSED", so SKIPPED rows do not affect them.

    # Deterministic iteration order — required for R1/R2 to be stable across
    # runs. Earlier (ticker, bar_date) wins; later duplicates are blocked.
    sig = sig.sort_values(["ticker", "bar_date"], kind="stable").reset_index(drop=True)

    cal = build_ticker_calendar(daily)

    # Validity counters per §2.1.7 / §13.3.
    r1_blocks = 0
    r2_blocks = 0
    signal_date_unavailable_halts = 0  # wired through _stop() but kept for symmetry
    ema_out_of_scope_skipped_count = 0  # Q-D counter (LOCKED 2026-05-06)

    _NULL_VALIDITY = dict(
        paper_valid_from=None,
        paper_valid_until=None,
        paper_signal_age=None,
        paper_expired_flag=None,
        ema_signature_id=None,
    )

    rows = []
    for _, ev in sig.iterrows():
        ticker = ev["ticker"]
        bar_date = ev["bar_date"]
        ema_tier = ev["ema_tier"]
        earliness = ev["earliness_score_pct"]

        base = dict(
            asof_date=bar_date,
            signal_asof_date=bar_date,  # Q-C explicit name (alias of asof_date)
            ticker=ticker,
            line=SOURCE_LINE,
            lane_source="close_confirmed",
            preview_seen_17=False,
            close_confirmed=True,
            signal_state=ev["signal_state"],
            ema_tier=ema_tier,
            earliness_score_pct=earliness,
            entry_reference_price=ev["entry_reference_price"],
            invalidation_level=ev["invalidation_level"],
            risk_per_share=ev["entry_reference_price"] - ev["invalidation_level"],
        )

        # OUT_OF_SCOPE (NaN earliness): emit SKIPPED with ema_out_of_scope so
        # current-day signal coverage reaches the daily report, but the row
        # carries no trade. Q-E: NaN earliness → SKIPPED, not OPEN.
        if ema_tier == "OUT_OF_SCOPE":
            ema_out_of_scope_skipped_count += 1
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    outcome_asof_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="ema_out_of_scope",
                ) | _NULL_VALIDITY
            )
            continue

        # TIER_B is SKIPPED — recorded but not traded; NULL validity columns,
        # no R1/R2.
        if ema_tier == "TIER_B_WARNING":
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    outcome_asof_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="tier_b_warning",
                ) | _NULL_VALIDITY
            )
            continue

        # TIER_A: try to build paper trade
        entry_date, entry_price = next_open_lookup(cal, ticker, bar_date)
        if entry_date is None or pd.isna(entry_price):
            rows.append(
                base | dict(
                    entry_date=None,
                    entry_price=np.nan,
                    exit_date=None,
                    outcome_asof_date=None,
                    exit_price=np.nan,
                    status="SKIPPED",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason="no_next_open",
                ) | _NULL_VALIDITY
            )
            continue

        # Compute validity + signature for the candidate (fail-fast inside).
        validity = attach_validity_fields(
            ticker=str(ticker),
            source_line=SOURCE_LINE,
            signal_state=str(ev["signal_state"]),
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

        exit_date, exit_price = nth_trading_day_close(
            cal, ticker, entry_date, HORIZON_TRADING_DAYS
        )

        validity_columns = {
            "paper_valid_from": validity["paper_valid_from"],
            "paper_valid_until": validity["paper_valid_until"],
            "paper_signal_age": validity["paper_signal_age"],
            "paper_expired_flag": validity["paper_expired_flag"],
            "ema_signature_id": validity["ema_signature_id"],
        }

        if exit_date is None or pd.isna(exit_price):
            # entry observed, exit not yet — OPEN status
            rows.append(
                base | dict(
                    entry_date=entry_date,
                    entry_price=float(entry_price),
                    exit_date=None,
                    outcome_asof_date=None,
                    exit_price=np.nan,
                    status="OPEN",
                    gross_return_pct=np.nan,
                    net_return_pct=np.nan,
                    realized_R_paper=np.nan,
                    skip_reason=None,
                ) | validity_columns
            )
            continue

        # CLOSED — both entry and exit observed
        gross = float(exit_price) / float(entry_price) - 1.0
        net = gross - COST_ROUNDTRIP_FRAC
        rps = float(ev["entry_reference_price"]) - float(ev["invalidation_level"])
        realized_R_paper = (
            (float(exit_price) - float(entry_price)) / rps if rps > 0 else np.nan
        )

        rows.append(
            base | dict(
                entry_date=entry_date,
                entry_price=float(entry_price),
                exit_date=exit_date,
                outcome_asof_date=exit_date,  # Q-C: outcome date = exit_date for CLOSED
                exit_price=float(exit_price),
                status="CLOSED",
                gross_return_pct=gross,
                net_return_pct=net,
                realized_R_paper=realized_R_paper,
                skip_reason=None,
            ) | validity_columns
        )

    log = pd.DataFrame(rows)

    # Backfill flag — events whose +10d window is past `asof` are backfilled
    # historical observations; events whose entry is on/after `asof` are
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
        skipped_ema_out_of_scope=int(ema_out_of_scope_skipped_count),
    )

    return log, validity_stats


# ---------------------------------------------------------------------------
# §5 view materialization (boolean flags)


def materialize_views(log: pd.DataFrame) -> pd.DataFrame:
    """Adds in_view_raw_all_candidates + in_view_capped_portfolio flags."""
    # Primary: every TIER_A candidate that is not SKIPPED for OUT-OF-SCOPE
    # reasons. SKIPPED-due-to-no_next_open candidates remain in the log but
    # not in the view (no trade ever existed). TIER_B is in log but not in
    # views (it's tracked SKIPPED only).
    in_raw = (
        (log["ema_tier"] == "TIER_A_PAPER")
        & (log["status"].isin(["OPEN", "CLOSED"]))
    )
    log["in_view_raw_all_candidates"] = in_raw

    # Secondary capped view: per asof_date, ascending alphabetical ticker,
    # max 5/day. This selection rule is operational — NOT alpha ranking
    # (spec §5.2).
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
    """Per-view metrics on closed trades only. Open trades counted but no
    return aggregation."""
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
            "avg_R": None,
            "median_R": None,
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

    # Equal-notional shadow equity curve = 1 + cumsum of gross returns
    # (each trade is an independent $1 notional event ordered by exit_date).
    # Multiplicative cumprod would model continuous reinvestment of a single
    # position, which is wrong for overlapping equal-notional shadow trades.
    eq_series = (
        closed.sort_values(["exit_date", "ticker"])["gross_return_pct"]
        .astype(float)
        .cumsum()
        + 1.0
    )
    rolling_max = eq_series.cummax()
    dd = float((eq_series / rolling_max - 1).min())

    return {
        "n_closed": int(len(closed)),
        "n_open": open_ct,
        "avg_gross_return_pct": float(gross.mean()),
        "median_gross_return_pct": float(gross.median()),
        "avg_net_return_pct": float(net.mean()),
        "median_net_return_pct": float(net.median()),
        "avg_R": float(R.mean()),
        "median_R": float(R.median()),
        "win_rate": float((gross > 0).mean()),
        "profit_factor": pf,
        "max_drawdown_gross": dd if not np.isnan(dd) else None,
    }


def daily_summary(log: pd.DataFrame) -> pd.DataFrame:
    """Per-asof_date roll-up."""
    g = log.groupby("asof_date", group_keys=False)
    out = pd.DataFrame(
        dict(
            n_candidates=g.size(),
            n_tier_a=g.apply(lambda d: int((d["ema_tier"] == "TIER_A_PAPER").sum())),
            n_tier_b_skipped=g.apply(
                lambda d: int((d["ema_tier"] == "TIER_B_WARNING").sum())
            ),
            n_open=g.apply(lambda d: int((d["status"] == "OPEN").sum())),
            n_closed=g.apply(lambda d: int((d["status"] == "CLOSED").sum())),
            n_skipped_no_open=g.apply(
                lambda d: int((d["skip_reason"] == "no_next_open").sum())
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


def emit_preview_audit_stub(asof: pd.Timestamp.date) -> pd.DataFrame:
    """Emit empty preview audit with header schema. Lane 1 preview infra
    requires a 17:00 partial-bar HB scanner pipeline that is NOT part of v0
    implementation scope (spec §10 — `--mode preview` orchestrator entry
    exists, but its body is intentionally not wired for first emission).
    Future runs MAY populate this CSV; until then preview_seen_17=False on
    all canonical trades, audit file is structurally present but empty."""
    return pd.DataFrame(
        columns=[
            "asof_date",
            "ticker",
            "preview_state",
            "close_state",
            "preview_ema_tier",
            "close_ema_tier",
            "preview_to_close_status",
        ]
    )


# ---------------------------------------------------------------------------
# summary md


def emit_summary_md(
    log: pd.DataFrame,
    raw_metrics: dict,
    capped_metrics: dict,
    manifest: dict,
    asof: pd.Timestamp.date,
) -> str:
    n_total = len(log)
    n_tier_a = int((log["ema_tier"] == "TIER_A_PAPER").sum())
    n_tier_b = int((log["ema_tier"] == "TIER_B_WARNING").sum())
    n_open = int((log["status"] == "OPEN").sum())
    n_closed = int((log["status"] == "CLOSED").sum())
    n_skipped = int((log["status"] == "SKIPPED").sum())
    n_backfill = int(log["is_backfill"].sum())
    n_forward = n_total - n_backfill

    lines = []
    lines.append("# paper_execution_v0 — Run Summary\n")
    lines.append(
        f"- Spec: `memory/paper_execution_v0_spec.md` LOCKED 2026-05-03, ONAY 2026-05-04\n"
    )
    lines.append(f"- Run timestamp: {manifest['run_started_utc']}\n")
    lines.append(f"- asof: {asof}\n")
    lines.append(
        f"- Mode: {manifest['mode']}  |  Return-unit choice: {manifest['return_unit']}\n"
    )
    lines.append("\n## Cohort census\n")
    lines.append(f"- Trade-log rows: {n_total}\n")
    lines.append(f"- TIER_A_PAPER: {n_tier_a}\n")
    lines.append(f"- TIER_B_WARNING (SKIPPED, not traded): {n_tier_b}\n")
    lines.append(f"- Status: CLOSED={n_closed} / OPEN={n_open} / SKIPPED={n_skipped}\n")
    lines.append(f"- Backfilled historical: {n_backfill}  |  Forward-clock: {n_forward}\n")

    lines.append("\n## Lane discipline\n")
    lines.append(
        "- Lane 1 (17:00 preview) — operational pre-scan only, NO performance accounting.\n"
    )
    lines.append(
        "- Lane 2 (close-confirmed) — canonical paper signal source; **all** trade-log rows here are `lane_source=close_confirmed`.\n"
    )
    lines.append(
        "- Preview infra not yet wired for first emission; `preview_seen_17 = False` on all rows; preview_audit.csv emitted as empty schema-only file.\n"
    )

    lines.append("\n## §5.1 Primary view — `raw_all_candidates` (HEADLINE)\n")
    lines.append("\n*Every close-confirmed TIER_A_PAPER event, no cap. Equal-notional shadow.*\n")
    lines.append("\n")
    for k, v in raw_metrics.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}\n")
        else:
            lines.append(f"- **{k}**: {v}\n")

    lines.append("\n## §5.2 Operational secondary view — `capped_portfolio`\n")
    lines.append(
        "\n*Operational triage only — NOT alpha ranking. max_new_positions_per_day=5, selection rule = ascending alphabetical ticker.*\n"
    )
    lines.append(
        "*If this view looks better than `raw_all_candidates`, that is selection artifact / sample-size effect — NOT a justification to cap in any future spec (spec §5.2).*\n"
    )
    lines.append("\n")
    for k, v in capped_metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}\n")
        else:
            lines.append(f"- {k}: {v}\n")

    lines.append("\n## Locked constants (spec §4 / §5 / §6)\n")
    lines.append(f"- early threshold: earliness_score_pct ≤ {EARLY_THRESHOLD} (LOCKED)\n")
    lines.append(f"- horizon: {HORIZON_TRADING_DAYS} trading days (LOCKED)\n")
    lines.append(
        f"- cost: {COST_BPS_ENTRY} bps entry + {COST_BPS_EXIT} bps exit = "
        f"{COST_ROUNDTRIP_FRAC*100:.2f}% round-trip (LOCKED, percentage view only)\n"
    )
    lines.append(f"- max_new_positions_per_day: {MAX_NEW_POSITIONS_PER_DAY} (capped view, LOCKED)\n")
    lines.append("- selection rule (capped view): ascending alphabetical ticker (LOCKED, operational, not alpha)\n")
    lines.append("- entry: next trading-day open after close-confirmed signal (LOCKED)\n")
    lines.append(f"- exit: close at +{HORIZON_TRADING_DAYS} trading days from entry (LOCKED)\n")
    lines.append("- TP / SL / trailing: NONE (LOCKED)\n")
    lines.append(
        "- realized_R_paper: numerator = (paper_exit − paper_entry); denominator = "
        "frozen R_per_share = entry_reference_price − invalidation_level (reused from HB parquet). "
        "Paper R is denominator-comparable to Pilot 5/6 historical realized_R_10d "
        "but NOT numerator-identical (paper entry = next-open ≠ historical signal-day entry).\n"
    )
    lines.append("\n## Forbidden interpretations (spec §13)\n")
    lines.append("- ❌ EMA is a live buy signal\n")
    lines.append("- ❌ Extended non-early is a short / avoid signal\n")
    lines.append("- ❌ EMA should rank scanner output\n")
    lines.append("- ❌ This can be traded live now\n")
    lines.append("- ❌ 17:00 preview is an entry trigger\n")
    lines.append("- ❌ Pilot 5/6 historical result guarantees forward performance\n")
    lines.append("- ❌ `capped_portfolio` is the better strategy\n")
    lines.append("- ❌ Cascade to decision_engine / NOX / production HTML without per-target ONAY\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["close_confirmed", "preview"],
        default="close_confirmed",
    )
    ap.add_argument(
        "--asof",
        default=None,
        help="YYYY-MM-DD (default = today's date in TR)",
    )
    ap.add_argument(
        "--forward",
        action="store_true",
        help="Forward mode (LOCKED 2026-05-06): use refreshed HB + EMA forward "
             "artifacts (pilot4_earliness/pilot5/pilot6 _forward.parquet) and "
             "stable_event_key 5-tuple key-based merge. Locked-research path "
             "is unchanged when --forward is omitted.",
    )
    args = ap.parse_args()

    global _FORWARD_MODE
    _FORWARD_MODE = bool(args.forward)

    t0 = time.time()
    print(f"[paper_execution_v0] mode={args.mode} forward={_FORWARD_MODE} starting...")

    info = boot_integrity()
    print("[paper_execution_v0] boot integrity OK")

    asof = (
        pd.Timestamp(args.asof).date()
        if args.asof
        else pd.Timestamp.now(tz="Europe/Istanbul").date()
    )

    if args.mode == "preview":
        # Lane 1 only — operational. NOT yet wired (spec §10 deferred for v0).
        # We emit the audit stub and exit. Lane 1 → counted-perf leak is
        # impossible because no trade is ever generated here.
        print("[paper_execution_v0] preview mode: writing empty audit stub")
        audit = emit_preview_audit_stub(asof)
        audit.to_csv(OUT_PREVIEW_AUDIT, index=False)
        print(f"[paper_execution_v0] wrote {OUT_PREVIEW_AUDIT.relative_to(ROOT)}")
        print(f"[paper_execution_v0] done in {time.time() - t0:.1f}s")
        return

    # Lane 2 — close confirmed
    signals = load_signals()
    print(f"[paper_execution_v0] HB extended-eligible signals: {len(signals)}")

    daily = load_daily_ohlcv()
    print(f"[paper_execution_v0] daily OHLCV rows: {len(daily)} (resampled from 1h)")

    ema_daily_index = load_ema_daily_index()
    print(f"[paper_execution_v0] ema_context_daily index rows: {len(ema_daily_index)}")

    log, validity_stats = build_trade_log(signals, daily, asof, ema_daily_index)
    log = materialize_views(log)
    print(f"[paper_execution_v0] trade log rows: {len(log)}")

    # Verify lane purity invariant
    bad_lane = log[log["lane_source"] != "close_confirmed"]
    if len(bad_lane) > 0:
        _stop(f"lane purity violation — {len(bad_lane)} rows with lane_source != close_confirmed")

    raw_metrics = compute_view_metrics(log, "in_view_raw_all_candidates")
    capped_metrics = compute_view_metrics(log, "in_view_capped_portfolio")
    daily_sum = daily_summary(log)

    # Manifest
    manifest = {
        "run_started_utc": _utc_now(),
        "manifest_notes": [
            "First emission used multiplicative cumprod for max_drawdown which "
            "incorrectly modeled overlapping equal-notional trades as continuous "
            "reinvestment. Fixed to additive cumsum equity curve (1 + cumsum(gross)). "
            "Trade-log eligibility / entry / exit / per-trade returns UNCHANGED; "
            "only the drawdown summary metric was affected. Re-emitted under spec §11."
        ],
        "mode": args.mode,
        "asof": str(asof),
        "spec_path": "memory/paper_execution_v0_spec.md",
        "spec_status": "LOCKED 2026-05-03, ONAY 2026-05-04",
        "view_choice": "boolean_flags (in_view_raw_all_candidates / in_view_capped_portfolio)",
        "primary_view": "raw_all_candidates",
        "secondary_view": "capped_portfolio",
        "selection_rule_capped": "ascending alphabetical ticker (operational, NOT alpha ranking)",
        "return_unit": "realized_R_paper (numerator: paper-trade prices; denominator: frozen R_per_share = entry_reference_price - invalidation_level reused from HB parquet) + percentage gross/net",
        "cost_assumption_pct": f"{COST_BPS_ENTRY}+{COST_BPS_EXIT} bps round-trip = {COST_ROUNDTRIP_FRAC}",
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "early_threshold": EARLY_THRESHOLD,
        "max_new_positions_per_day": MAX_NEW_POSITIONS_PER_DAY,
        "ohlcv_pipeline": "data.intraday_1h.daily_resample(extfeed_intraday_1h_3y_master.parquet) — same canonical adapter Pilot 5/6 historical labels traced through; this is NOT silent resampling, it is reuse of the historical canonical pipeline",
        "tier_b_handling": "SKIPPED, displayed in trade log, NOT contra-signal, NOT paper-traded",
        "lane_purity_check": "passed (all rows lane_source=close_confirmed)",
        "boot_integrity": info,
        "forward_merge_diag": dict(_FORWARD_MERGE_DIAG) if _FORWARD_MODE else None,
        "trade_log_rows": int(len(log)),
        "tier_a_count": int((log["ema_tier"] == "TIER_A_PAPER").sum()),
        "tier_b_count": int((log["ema_tier"] == "TIER_B_WARNING").sum()),
        "status_distribution": log["status"].value_counts().to_dict(),
        "primary_view_metrics": raw_metrics,
        "secondary_view_metrics": capped_metrics,
        # Validity revision (LOCKED 2026-05-04 + §14 ONAY) — five additive fields.
        **validity_stats,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[paper_execution_v0] wrote {OUT_MANIFEST.relative_to(ROOT)}")

    # Trade log outputs
    log.to_parquet(OUT_TRADES_PARQUET, index=False)
    log.to_csv(OUT_TRADES_CSV, index=False)
    print(f"[paper_execution_v0] wrote {OUT_TRADES_PARQUET.relative_to(ROOT)} ({len(log)} rows)")
    print(f"[paper_execution_v0] wrote {OUT_TRADES_CSV.relative_to(ROOT)}")

    daily_sum.to_csv(OUT_DAILY_SUMMARY, index=False)
    print(f"[paper_execution_v0] wrote {OUT_DAILY_SUMMARY.relative_to(ROOT)}")

    # Preview audit emitted as empty stub on close_confirmed runs too, so
    # downstream tooling can rely on its existence.
    audit = emit_preview_audit_stub(asof)
    audit.to_csv(OUT_PREVIEW_AUDIT, index=False)
    print(f"[paper_execution_v0] wrote {OUT_PREVIEW_AUDIT.relative_to(ROOT)} (empty stub)")

    # Summary md
    md = emit_summary_md(log, raw_metrics, capped_metrics, manifest, asof)
    OUT_SUMMARY_MD.write_text(md)
    print(f"[paper_execution_v0] wrote {OUT_SUMMARY_MD.relative_to(ROOT)}")

    print(f"[paper_execution_v0] done in {time.time() - t0:.1f}s")
    print(f"[paper_execution_v0] PRIMARY VIEW headline (raw_all_candidates):")
    print(f"  n_closed={raw_metrics['n_closed']} / n_open={raw_metrics['n_open']}")
    if raw_metrics["avg_R"] is not None:
        print(f"  avg_R={raw_metrics['avg_R']:.4f} / median_R={raw_metrics['median_R']:.4f}")
        print(f"  win_rate={raw_metrics['win_rate']:.4f} / PF={raw_metrics['profit_factor']:.4f}")


if __name__ == "__main__":
    main()
