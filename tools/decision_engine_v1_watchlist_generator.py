#!/usr/bin/env python3
"""Decision Engine v1 — post-Class-B tomorrow-watchlist generator.

Reads:  output/decision_engine_v1_events.parquet
Writes: output/decision_engine_v1_tomorrow_watchlist_post_class_b_<asof>.{md,csv}

asof resolution (PR-DE-2b):
  - `--asof-date YYYY-MM-DD` wins. Must match the events parquet's asof set
    (otherwise hard FAIL — refusing to write a filename that misrepresents
    the underlying source).
  - else auto-detect from max(events.date).

Sections (CSV column 0 and MD H2 ordering):
  EXECUTABLE -> SIZE_REDUCED -> WAIT_BETTER_ENTRY -> WAIT_RETEST -> PAPER_ONLY

EXECUTABLE row order: multi-cell tickers first (overlap-count desc, ticker
                      alpha tie-break), then single-cell rows ticker alpha.
Other sections: ticker alpha, then family / timeframe tie-break.

Manual-review watchlist only. Live gate CLOSED — no live_execution path
touched, no portfolio/rank/score emitted.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
EVENTS_PATH = OUT_DIR / "decision_engine_v1_events.parquet"
STAGE6_MANIFEST_PATH = OUT_DIR / "paper_execution_v0_forward_run_manifest.json"
TRIDENT_PROBE_PATH = OUT_DIR / "trident_probe_mb_3y.parquet"
XU100_DAILY_PATH = OUT_DIR / "xu100_extfeed_daily.parquet"

SECTIONS = ["EXECUTABLE", "SIZE_REDUCED", "WAIT_BETTER_ENTRY", "WAIT_RETEST", "PAPER_ONLY"]

CSV_COLS = [
    "section", "ticker", "source", "family", "state", "timeframe",
    "setup_label", "execution_label", "market_context", "reason_codes",
    "entry_ref", "stop_ref", "atr", "fill_assumption", "risk_atr",
    "paper_stream_ref", "notes",
    "trident_tag", "trident_D_pct", "trident_SIL_pct_below_B",
    "trident_zone_width_pct", "trident_zone_age_bars",
    "trident_weekly_family",
    "trident_tier1_active",
]

ALLOWED_TRIDENT_COLS = frozenset({
    "ticker", "family", "setup_family", "event_bar_date", "event_ts",
    "A_price", "B_price", "C_price", "D_target",
    "D_pct", "D_minus_B_atr", "B_to_D_pct",
    "atr_at_event", "atr_pct_at_event",
    "zone_width_pct", "zone_age_bars",
    "bos_distance_atr_at_event", "vol_ratio_20_at_event",
    "structural_invalidation_low", "structural_invalidation_pct_below_B",
})
FORBIDDEN_TRIDENT_COLS = frozenset({
    "forward_high_5d", "forward_high_10d", "forward_high_20d", "forward_high_30d",
    "forward_mfe_pct_5d", "forward_mfe_pct_10d", "forward_mfe_pct_20d", "forward_mfe_pct_30d",
    "gap_to_D_atr_5d", "gap_to_D_atr_10d", "gap_to_D_atr_20d", "gap_to_D_atr_30d",
    "D_touched_5d", "D_touched_10d", "D_touched_20d", "D_touched_30d",
    "days_to_D_strict", "days_to_D_loose",
    "outcome_strict", "structural_break_day", "forward_bars_available",
})
SURFACED_TRIDENT_COLS = (
    "D_pct",
    "structural_invalidation_pct_below_B",
    "zone_width_pct",
    "zone_age_bars",
    "bos_distance_atr_at_event",
)
TRIDENT_ELIGIBLE_SETUP_KIND = "above_mb_birth"
TRIDENT_ELIGIBLE_SOURCE = "mb_scanner"

# PR-DE-3.17 (+ lead-lag tightening): cross-timeframe weekly confluence.
# LEAD-LAG semantics — the weekly leader is born FIRST, the sub-weekly follows
# THE WEEK AFTER. A sub-weekly (mb_5h / mb_1d / bb_5h / bb_1d) above_mb_birth is
# "weekly-confluent" iff BOTH hold for the same ticker:
#   (LEAD) a mb_1w / bb_1w above_mb_birth whose weekly bar is the LAST FULLY
#          COMPLETED week prior to asof (Friday-anchored, range (prev_Fri, this_Fri]).
#   (LAG)  the sub-weekly above_mb_birth's OWN event_bar_date falls in the
#          CURRENT (containing) week: event_bar_date ∈ (_prior_completed_friday, asof].
# The LAG gate rejects same-week (coincident) births — only genuine next-week
# follow-through counts. Strict: immediately prior completed week, no wider lookback.
WEEKLY_CONFLUENCE_WEEKLY_FAMILIES = frozenset({"mb_1w", "bb_1w"})
WEEKLY_CONFLUENCE_ELIGIBLE_FAMILIES = frozenset({"mb_5h", "mb_1d", "bb_5h", "bb_1d"})
WEEKLY_CONFLUENCE_TAG = "WEEKLY_BIRTH_ACTIVE"

# PR-DE-3.18: Trident Tier-1 strict daily filter.
# A mb_scanner above_mb_birth event on family `mb_1d` whose event_bar_date ==
# asof is tagged TRIDENT_TIER1_ACTIVE iff ALL four gates AND together:
#   G1: D_pct ∈ [TIER1_D_PCT_LOW, TIER1_D_PCT_HIGH)
#   G2: structural_invalidation_pct_below_B ≤ trailing-365d tercile-33%
#       (mb_1d cohort only; strictly before asof)
#   G3: bos_distance_atr_at_event ≥ trailing-365d tercile-67%
#       (mb_1d cohort only; strictly before asof)
#   G4: XU100 daily close ret_20d > 0 at most-recent-available bar
#       (regime up; lagged feed accepted as proxy with stale-days log).
# Tag scope: ONLY the originating mb_1d watchlist row. No cross-TF surfacing —
# sibling mb_5h/bb_5h/bb_1d/mb_1w/etc. rows on the same ticker remain untagged.
# Trailing terciles are computed in-process on the existing trident probe
# (no new ETL; reads are ALLOWED_TRIDENT_COLS only — leak guard preserved).
TIER1_FAMILY = "mb_1d"
TIER1_TAG = "TRIDENT_TIER1_ACTIVE"
TIER1_D_PCT_LOW = 20.0
TIER1_D_PCT_HIGH = 30.0  # exclusive upper bound
TIER1_TRAILING_WINDOW_DAYS = 365
TIER1_TRAILING_MIN_N = 30


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decision Engine v1 — post-Class-B tomorrow-watchlist generator. "
            "Reads decision_engine_v1_events.parquet; writes "
            "decision_engine_v1_tomorrow_watchlist_post_class_b_<asof>.{md,csv}. "
            "asof resolves from --asof-date (must match events parquet) or "
            "max(events.date)."
        )
    )
    p.add_argument(
        "--asof-date",
        type=lambda s: dt.date.fromisoformat(s).isoformat(),
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Operational asof date. If omitted, auto-detect from "
            "max(events.date). Mismatch with events parquet is a hard FAIL."
        ),
    )
    return p.parse_args()


def _resolve_asof(arg_asof: str | None, events_dates: set[str]) -> str:
    if not events_dates:
        print(
            f"[watchlist_generator] FAIL: events parquet has zero `date` values",
            flush=True,
        )
        sys.exit(1)
    if arg_asof is not None:
        if arg_asof not in events_dates:
            print(
                f"[watchlist_generator] FAIL: --asof-date {arg_asof} not "
                f"present in events.parquet (have: {sorted(events_dates)})",
                flush=True,
            )
            sys.exit(1)
        return arg_asof
    return sorted(events_dates)[-1]


def _fmt_num(v, digits=4):
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return ""
    if f != f:
        return ""
    return f"{f:.{digits}f}"


def _fmt_risk(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return ""
    if f != f:
        return ""
    return f"{f:.2f}"


def _read_degraded_state() -> dict:
    """PR-DE-3.15: read Stage 6 manifest if present to detect degraded mode.

    Returns a dict with keys:
      - degraded_mode: bool
      - degraded_sources: list[str]            (e.g. ["line_tr"])
      - excluded_paper_origins: list[str]      (e.g. ["TRIGGER_RETEST"])
      - line_e_status / line_tr_status         (PASS|STALE|FAIL|"unknown")
      - line_e_signal_asof_max / line_tr_signal_asof_max
      - manifest_present: bool

    Missing manifest or missing fields → degraded_mode=False (silent
    backward-compat with pre-PR-DE-3.15 callers).
    """
    state = {
        "degraded_mode": False,
        "degraded_sources": [],
        "excluded_paper_origins": [],
        "line_e_status": "unknown",
        "line_tr_status": "unknown",
        "line_e_signal_asof_max": None,
        "line_tr_signal_asof_max": None,
        "manifest_present": False,
    }
    if not STAGE6_MANIFEST_PATH.exists():
        return state
    try:
        m = json.loads(STAGE6_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return state
    state["manifest_present"] = True
    state["degraded_mode"] = bool(m.get("degraded_mode", False))
    state["degraded_sources"] = list(m.get("degraded_sources") or [])
    state["excluded_paper_origins"] = list(m.get("excluded_paper_origins") or [])
    state["line_e_status"] = m.get("line_e_status", "unknown")
    state["line_tr_status"] = m.get("line_tr_status", "unknown")
    state["line_e_signal_asof_max"] = m.get("line_e_signal_asof_max")
    state["line_tr_signal_asof_max"] = m.get("line_tr_signal_asof_max")
    return state


def _load_trident_attachments(asof: str) -> dict[tuple[str, str], dict]:
    """PR-DE-3.16: read trident_probe_mb_3y.parquet and return per-(ticker,
    family_short) attachment dict for event_bar_date == asof.

    Returns {} silently when the probe parquet is missing (Stage 7.5 may
    have failed or producer unavailable in local smoke). Hard FAIL when
    the parquet contains a column outside the ALLOWED ∪ FORBIDDEN universe
    (unknown column = potentially a new leak channel).

    Leak guard: every column read from the parquet is checked against
    FORBIDDEN_TRIDENT_COLS; matched columns are dropped before the row dict
    leaves this function. ALLOWED ∩ FORBIDDEN is asserted disjoint at
    module level via this function's first sanity check.
    """
    assert ALLOWED_TRIDENT_COLS.isdisjoint(FORBIDDEN_TRIDENT_COLS), (
        "[watchlist_generator] BUG: Trident allowlist overlaps forbidden list"
    )
    if not TRIDENT_PROBE_PATH.exists():
        return {}
    try:
        table = pq.read_table(TRIDENT_PROBE_PATH)
    except Exception as exc:
        print(
            f"[watchlist_generator] WARN: trident probe read failed "
            f"({TRIDENT_PROBE_PATH.relative_to(ROOT)}): {exc} — attachments skipped",
            flush=True,
        )
        return {}

    probe_cols = set(table.column_names)
    unknown = probe_cols - ALLOWED_TRIDENT_COLS - FORBIDDEN_TRIDENT_COLS
    if unknown:
        print(
            f"[watchlist_generator] FAIL: trident probe contains "
            f"unrecognized columns {sorted(unknown)} (not in ALLOWED or "
            f"FORBIDDEN sets) — refusing to attach (potential leak channel)",
            flush=True,
        )
        sys.exit(1)

    keep = [c for c in table.column_names if c in ALLOWED_TRIDENT_COLS]
    sub = table.select(keep).to_pylist()

    attachments: dict[tuple[str, str], dict] = {}
    asof_str = str(asof)
    for r in sub:
        ed = r.get("event_bar_date")
        ed_str = str(ed)[:10] if ed is not None else ""
        if ed_str != asof_str:
            continue
        tk = r.get("ticker") or ""
        fam = r.get("family") or ""
        key = (tk, fam)
        attachments[key] = {
            "D_pct": r.get("D_pct"),
            "structural_invalidation_pct_below_B": r.get("structural_invalidation_pct_below_B"),
            "zone_width_pct": r.get("zone_width_pct"),
            "zone_age_bars": r.get("zone_age_bars"),
            "bos_distance_atr_at_event": r.get("bos_distance_atr_at_event"),
        }
    return attachments


def _trident_lookup(
    attachments: dict[tuple[str, str], dict],
    source: str,
    family: str,
    ticker: str,
) -> dict:
    if source != TRIDENT_ELIGIBLE_SOURCE:
        return {}
    parts = (family or "").split("__", 1)
    if len(parts) != 2:
        return {}
    family_short, setup_kind = parts[0], parts[1]
    if setup_kind != TRIDENT_ELIGIBLE_SETUP_KIND:
        return {}
    return attachments.get((ticker, family_short), {})


def _prior_completed_friday(asof: str) -> str:
    """PR-DE-3.17 / PR-DE-3.21: last completed Friday ON OR BEFORE asof.

    The weekly bar labeled at this Friday closes the range
    (prev_Friday, this_Friday] which spans Mon-Fri sessions. mb_1w /
    bb_1w above_mb_birth events with `event_bar_date == this_Friday`
    represent structural birth that completed during that week.

    Close-mode runs (master-data-pull 15:30 UTC = 18:30 TR weekday cron)
    fire AFTER BIST close (18:00 TR), so when asof is itself a Friday the
    weekly bar dated 2026-05-22 is already closed and IS the most-recent
    completed weekly bar. Return asof itself in that case.
    """
    d = dt.date.fromisoformat(asof)
    delta = (d.weekday() - 4) % 7
    return (d - dt.timedelta(days=delta)).isoformat()


def _load_weekly_birth_tickers(asof: str) -> dict[str, list[str]]:
    """PR-DE-3.17: return {ticker: sorted list of weekly families} for any
    mb_1w / bb_1w above_mb_birth event whose `event_bar_date` matches
    `_prior_completed_friday(asof)`.

    Reads only ALLOWED columns from the trident probe (ticker, family,
    event_bar_date) — leak guard preserved. Returns {} silently when the
    probe is absent.
    """
    if not TRIDENT_PROBE_PATH.exists():
        return {}
    target = _prior_completed_friday(asof)
    try:
        table = pq.read_table(
            TRIDENT_PROBE_PATH,
            columns=["ticker", "family", "event_bar_date"],
        )
    except Exception as exc:
        print(
            f"[watchlist_generator] WARN: trident probe read (weekly) failed "
            f"({TRIDENT_PROBE_PATH.relative_to(ROOT)}): {exc} — confluence skipped",
            flush=True,
        )
        return {}

    fams_by_ticker: dict[str, set[str]] = defaultdict(set)
    for r in table.to_pylist():
        fam = r.get("family") or ""
        if fam not in WEEKLY_CONFLUENCE_WEEKLY_FAMILIES:
            continue
        ed = r.get("event_bar_date")
        ed_str = str(ed)[:10] if ed is not None else ""
        if ed_str != target:
            continue
        tk = r.get("ticker") or ""
        if tk:
            fams_by_ticker[tk].add(fam)
    return {tk: sorted(fams) for tk, fams in fams_by_ticker.items()}


def _load_current_week_subweekly_births(asof: str) -> set:
    """Lead-lag LAG gate: return {(ticker, family_short)} for every sub-weekly
    (mb_5h / mb_1d / bb_5h / bb_1d) above_mb_birth whose OWN event_bar_date falls
    in the CURRENT (containing) week — strictly after the LEAD weekly Friday and
    at-or-before asof: event_bar_date ∈ (_prior_completed_friday(asof), asof].

    This is what separates a genuine lead-lag follow-through (sub-weekly birth the
    week AFTER the weekly leader) from a same-week coincidence. Reads only ALLOWED
    columns (ticker, family, event_bar_date) — leak guard preserved. Returns an
    empty set silently when the probe is absent.
    """
    if not TRIDENT_PROBE_PATH.exists():
        return set()
    lower = _prior_completed_friday(asof)  # exclusive lower bound (LEAD weekly Fri)
    upper = str(asof)                       # inclusive upper bound (asof)
    try:
        table = pq.read_table(
            TRIDENT_PROBE_PATH,
            columns=["ticker", "family", "event_bar_date"],
        )
    except Exception as exc:
        print(
            f"[watchlist_generator] WARN: trident probe read (lag gate) failed "
            f"({TRIDENT_PROBE_PATH.relative_to(ROOT)}): {exc} — confluence skipped",
            flush=True,
        )
        return set()

    out: set = set()
    for r in table.to_pylist():
        fam = r.get("family") or ""
        if fam not in WEEKLY_CONFLUENCE_ELIGIBLE_FAMILIES:
            continue
        ed = r.get("event_bar_date")
        ed_str = str(ed)[:10] if ed is not None else ""
        if not ed_str or not (lower < ed_str <= upper):
            continue
        tk = r.get("ticker") or ""
        if tk:
            out.add((tk, fam))
    return out


def _compute_xu100_regime(asof: str) -> dict:
    """PR-DE-3.18 (G4): read xu100_extfeed_daily.parquet, compute ret_20d at
    the most-recent close bar at-or-before asof. Returns a state dict:

      {
        "available": bool,                # parquet present & non-empty
        "asof_available": "YYYY-MM-DD",   # date of the close used
        "ret_20d_pct": float,             # last close pct_change(20) * 100
        "pass": bool,                     # ret_20d_pct > 0
        "days_stale": int,                # asof - asof_available (calendar)
        "reason": str,                    # short status string for logs
      }

    Graceful degrade: missing/empty/short feed → available=False, pass=False
    (Tier-1 evaluates to zero rows; clear log line emitted). NO hard FAIL —
    the watchlist always renders even when the regime input is unavailable.
    """
    state = {
        "available": False,
        "asof_available": "",
        "ret_20d_pct": float("nan"),
        "pass": False,
        "days_stale": 0,
        "reason": "feed_missing",
    }
    if not XU100_DAILY_PATH.exists():
        return state
    try:
        import pandas as _pd
        df = pq.read_table(XU100_DAILY_PATH).to_pandas()
    except Exception as exc:
        state["reason"] = f"read_failed:{exc.__class__.__name__}"
        return state
    if df.empty or "close" not in df.columns:
        state["reason"] = "empty_or_no_close"
        return state
    if not isinstance(df.index, _pd.DatetimeIndex):
        # Fall back: look for a Date column.
        for cand in ("Date", "date", "ts"):
            if cand in df.columns:
                df = df.set_index(_pd.to_datetime(df[cand]))
                break
        if not isinstance(df.index, _pd.DatetimeIndex):
            state["reason"] = "no_datetime_index"
            return state
    df = df.sort_index()
    cutoff = _pd.Timestamp(asof) + _pd.Timedelta(days=1)
    df = df.loc[df.index < cutoff]
    if len(df) < 21:
        state["reason"] = f"short_history:n={len(df)}"
        return state
    last_close = float(df["close"].iloc[-1])
    ref_close = float(df["close"].iloc[-21])
    last_dt = df.index[-1]
    ret_20d_pct = (last_close / ref_close - 1.0) * 100.0
    asof_avail = str(last_dt.date())
    days_stale = max(0, (_pd.Timestamp(asof) - _pd.Timestamp(asof_avail)).days)
    state.update({
        "available": True,
        "asof_available": asof_avail,
        "ret_20d_pct": ret_20d_pct,
        "pass": ret_20d_pct > 0.0,
        "days_stale": days_stale,
        "reason": "ok" if days_stale == 0 else f"stale_{days_stale}d",
    })
    return state


def _load_tier1_eligible_keys(
    asof: str,
    xu100_state: dict,
) -> tuple[set[tuple[str, str]], dict]:
    """PR-DE-3.18: return ({(ticker, "mb_1d"), ...}, summary_dict).

    Eligible set contains every (ticker, "mb_1d") whose trident probe row
    satisfies G1+G2+G3 on event_bar_date==asof, when G4 (XU100 regime up)
    is also true. If G4 fails, returns empty set with full breakdown.

    Reads only ALLOWED_TRIDENT_COLS — leak guard preserved.
    """
    summary = {
        "universe_size": 0,
        "g1_pass": 0,
        "g2_pass": 0,
        "g3_pass": 0,
        "g4_pass": bool(xu100_state.get("pass")),
        "eligible_count": 0,
        "sil_t33": None,
        "bos_t67": None,
        "trailing_n": 0,
        "reason": "ok",
    }
    if not TRIDENT_PROBE_PATH.exists():
        summary["reason"] = "probe_missing"
        return set(), summary
    try:
        table = pq.read_table(
            TRIDENT_PROBE_PATH,
            columns=[
                "ticker", "family", "event_bar_date", "D_pct",
                "structural_invalidation_pct_below_B",
                "bos_distance_atr_at_event",
            ],
        )
    except Exception as exc:
        summary["reason"] = f"probe_read_failed:{exc.__class__.__name__}"
        return set(), summary
    rows = table.to_pylist()
    asof_str = str(asof)
    # Trailing window for G2/G3 terciles: strictly before asof.
    try:
        asof_dt = dt.date.fromisoformat(asof)
    except Exception:
        summary["reason"] = "asof_parse_failed"
        return set(), summary
    win_start = (asof_dt - dt.timedelta(days=TIER1_TRAILING_WINDOW_DAYS)).isoformat()
    sil_train: list[float] = []
    bos_train: list[float] = []
    today_rows: list[dict] = []
    for r in rows:
        fam = r.get("family") or ""
        if fam != TIER1_FAMILY:
            continue
        ed = r.get("event_bar_date")
        ed_str = str(ed)[:10] if ed is not None else ""
        if not ed_str:
            continue
        if ed_str == asof_str:
            today_rows.append(r)
            continue
        if win_start <= ed_str < asof_str:
            sil = r.get("structural_invalidation_pct_below_B")
            bos = r.get("bos_distance_atr_at_event")
            try:
                if sil is not None:
                    sf = float(sil)
                    if sf == sf:
                        sil_train.append(sf)
            except (TypeError, ValueError):
                pass
            try:
                if bos is not None:
                    bf = float(bos)
                    if bf == bf:
                        bos_train.append(bf)
            except (TypeError, ValueError):
                pass
    summary["universe_size"] = len(today_rows)
    summary["trailing_n"] = min(len(sil_train), len(bos_train))
    if summary["trailing_n"] < TIER1_TRAILING_MIN_N:
        summary["reason"] = f"trailing_too_short:n={summary['trailing_n']}"
        return set(), summary
    sil_train.sort()
    bos_train.sort()
    sil_t33 = sil_train[int(round(len(sil_train) * (1 / 3))) - 1] if sil_train else None
    bos_t67 = bos_train[int(round(len(bos_train) * (2 / 3))) - 1] if bos_train else None
    summary["sil_t33"] = sil_t33
    summary["bos_t67"] = bos_t67
    eligible: set[tuple[str, str]] = set()
    for r in today_rows:
        d_pct = r.get("D_pct")
        sil = r.get("structural_invalidation_pct_below_B")
        bos = r.get("bos_distance_atr_at_event")
        try:
            d_f = float(d_pct) if d_pct is not None else None
        except (TypeError, ValueError):
            d_f = None
        try:
            s_f = float(sil) if sil is not None else None
        except (TypeError, ValueError):
            s_f = None
        try:
            b_f = float(bos) if bos is not None else None
        except (TypeError, ValueError):
            b_f = None
        g1 = (d_f is not None) and (TIER1_D_PCT_LOW <= d_f < TIER1_D_PCT_HIGH)
        g2 = (s_f is not None) and (sil_t33 is not None) and (s_f <= sil_t33)
        g3 = (b_f is not None) and (bos_t67 is not None) and (b_f >= bos_t67)
        if g1: summary["g1_pass"] += 1
        if g2: summary["g2_pass"] += 1
        if g3: summary["g3_pass"] += 1
        if g1 and g2 and g3 and summary["g4_pass"]:
            tk = r.get("ticker") or ""
            if tk:
                eligible.add((tk, TIER1_FAMILY))
    summary["eligible_count"] = len(eligible)
    return eligible, summary


def _tier1_lookup(
    eligible_keys: set[tuple[str, str]],
    source: str,
    family: str,
    ticker: str,
) -> bool:
    """Return True iff this watchlist row IS the originating mb_1d Tier-1
    ELIGIBLE event. Strict same-row semantics — sibling sub-TF rows on the
    same ticker are NOT tagged."""
    if source != TRIDENT_ELIGIBLE_SOURCE:
        return False
    parts = (family or "").split("__", 1)
    if len(parts) != 2:
        return False
    family_short, setup_kind = parts[0], parts[1]
    if setup_kind != TRIDENT_ELIGIBLE_SETUP_KIND:
        return False
    if family_short != TIER1_FAMILY:
        return False
    return (ticker, family_short) in eligible_keys


def _weekly_confluence_lookup(
    weekly_birth_tickers: dict[str, list[str]],
    current_week_subweekly_births: set,
    source: str,
    family: str,
    ticker: str,
) -> str:
    """Return ';'-joined weekly family list iff this row is a genuine LEAD-LAG
    weekly confluence: the ticker has a prior-completed-week weekly birth (LEAD)
    AND this eligible sub-weekly family's own birth falls in the CURRENT week
    (LAG), i.e. (ticker, family_short) ∈ current_week_subweekly_births.
    Empty string otherwise. The LAG gate rejects same-week (coincident) births.
    """
    if source != TRIDENT_ELIGIBLE_SOURCE:
        return ""
    parts = (family or "").split("__", 1)
    if len(parts) != 2:
        return ""
    family_short, setup_kind = parts[0], parts[1]
    if setup_kind != TRIDENT_ELIGIBLE_SETUP_KIND:
        return ""
    if family_short not in WEEKLY_CONFLUENCE_ELIGIBLE_FAMILIES:
        return ""
    # LAG gate: the sub-weekly birth must fall in the current (containing) week.
    if (ticker, family_short) not in current_week_subweekly_births:
        return ""
    fams = weekly_birth_tickers.get(ticker)
    if not fams:
        return ""
    return ";".join(fams)


def _compute_trident_tag(att: dict) -> str:
    if not att:
        return ""
    tags: list[str] = []
    d_pct = att.get("D_pct")
    sil = att.get("structural_invalidation_pct_below_B")
    try:
        if d_pct is not None and float(d_pct) >= 30.0:
            tags.append("D_PCT_30PLUS")
    except (TypeError, ValueError):
        pass
    try:
        if sil is not None and float(sil) <= -10.0:
            tags.append("SIL_DEEP")
    except (TypeError, ValueError):
        pass
    return ";".join(tags)


def _fmt_trident_pct(v) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:
        return ""
    return f"{f:.2f}"


def _fmt_trident_int(v) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:
        return ""
    return f"{int(f)}"


def _join_codes(*lists):
    out = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for c in lst:
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
    return ";".join(out)


def main():
    args = _parse_args()

    if not EVENTS_PATH.exists():
        print(
            f"[watchlist_generator] FAIL: events parquet missing at "
            f"{EVENTS_PATH.relative_to(ROOT)}",
            flush=True,
        )
        sys.exit(1)

    t = pq.read_table(EVENTS_PATH)
    rows = t.to_pylist()
    events_dates = {str(r["date"]) for r in rows}
    asof = _resolve_asof(args.asof_date, events_dates)
    total = len(rows)
    degraded = _read_degraded_state()
    trident_attachments = _load_trident_attachments(asof)
    weekly_birth_tickers = _load_weekly_birth_tickers(asof)
    current_week_subweekly_births = _load_current_week_subweekly_births(asof)
    prior_friday = _prior_completed_friday(asof)
    xu100_state = _compute_xu100_regime(asof)
    tier1_keys, tier1_summary = _load_tier1_eligible_keys(asof, xu100_state)
    trident_attach_count = 0
    weekly_confluence_count = 0
    tier1_tag_count = 0

    out_md = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.md"
    out_csv = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.csv"

    records = []
    for r in rows:
        att = _trident_lookup(
            trident_attachments,
            r["source"],
            r["family"],
            r["ticker"],
        )
        if att:
            trident_attach_count += 1
        weekly_fam = _weekly_confluence_lookup(
            weekly_birth_tickers,
            current_week_subweekly_births,
            r["source"],
            r["family"],
            r["ticker"],
        )
        base_tag = _compute_trident_tag(att)
        tag_parts: list[str] = [base_tag] if base_tag else []
        if weekly_fam:
            weekly_confluence_count += 1
            tag_parts.append(WEEKLY_CONFLUENCE_TAG)
        tier1_active = _tier1_lookup(
            tier1_keys,
            r["source"],
            r["family"],
            r["ticker"],
        )
        if tier1_active:
            tier1_tag_count += 1
            tag_parts.append(TIER1_TAG)
        trident_tag = ";".join(tag_parts)
        rec = {
            "ticker": r["ticker"],
            "source": r["source"],
            "family": r["family"],
            "state": r["state"],
            "timeframe": r["timeframe"],
            "setup_label": r["setup_label"],
            "execution_label": r["execution_label"],
            "market_context": r["regime"] or "",
            "reason_codes": _join_codes(
                r.get("setup_reason_codes"),
                r.get("execution_reason_codes"),
                r.get("context_reason_codes"),
            ),
            "entry_ref": r["entry_ref"],
            "stop_ref": r["stop_ref"],
            "atr": r["atr"],
            "fill_assumption": r["fill_assumption"] or "",
            "risk_atr": r["risk_atr"],
            "paper_stream_ref": r["paper_stream_ref"] or "",
            "notes": "",
            "trident_tag": trident_tag,
            "trident_D_pct": _fmt_trident_pct(att.get("D_pct")),
            "trident_SIL_pct_below_B": _fmt_trident_pct(
                att.get("structural_invalidation_pct_below_B")
            ),
            "trident_zone_width_pct": _fmt_trident_pct(att.get("zone_width_pct")),
            "trident_zone_age_bars": _fmt_trident_int(att.get("zone_age_bars")),
            "trident_weekly_family": weekly_fam,
            "trident_tier1_active": "TRUE" if tier1_active else "",
        }
        records.append(rec)

    by_section: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sec = rec["execution_label"]
        if sec in SECTIONS:
            by_section[sec].append(rec)

    exec_rows = by_section.get("EXECUTABLE", [])
    overlap_counts = Counter(r["ticker"] for r in exec_rows)
    multi_tickers = {tk for tk, c in overlap_counts.items() if c > 1}

    def exec_sort_key(r):
        tk = r["ticker"]
        cnt = overlap_counts[tk]
        primary = 0 if cnt > 1 else 1
        secondary = -cnt
        return (primary, secondary, tk, r["family"], r["timeframe"])

    sorted_exec = sorted(exec_rows, key=exec_sort_key)

    sorted_sections: dict[str, list[dict]] = {"EXECUTABLE": sorted_exec}
    for sec in SECTIONS[1:]:
        sorted_sections[sec] = sorted(
            by_section.get(sec, []),
            key=lambda r: (r["ticker"], r["family"], r["timeframe"]),
        )

    source_dist: dict[str, Counter] = {sec: Counter() for sec in SECTIONS}
    for sec in SECTIONS:
        for r in sorted_sections[sec]:
            source_dist[sec][r["source"]] += 1

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for sec in SECTIONS:
            for r in sorted_sections[sec]:
                row = {
                    "section": sec,
                    "ticker": r["ticker"],
                    "source": r["source"],
                    "family": r["family"],
                    "state": r["state"],
                    "timeframe": r["timeframe"],
                    "setup_label": r["setup_label"],
                    "execution_label": r["execution_label"],
                    "market_context": r["market_context"],
                    "reason_codes": r["reason_codes"],
                    "entry_ref": r["entry_ref"] if r["entry_ref"] is not None else "",
                    "stop_ref": r["stop_ref"] if r["stop_ref"] is not None else "",
                    "atr": r["atr"] if r["atr"] is not None else "",
                    "fill_assumption": r["fill_assumption"],
                    "risk_atr": r["risk_atr"] if r["risk_atr"] is not None else "",
                    "paper_stream_ref": r["paper_stream_ref"],
                    "notes": r["notes"],
                    "trident_tag": r["trident_tag"],
                    "trident_D_pct": r["trident_D_pct"],
                    "trident_SIL_pct_below_B": r["trident_SIL_pct_below_B"],
                    "trident_zone_width_pct": r["trident_zone_width_pct"],
                    "trident_zone_age_bars": r["trident_zone_age_bars"],
                    "trident_weekly_family": r["trident_weekly_family"],
                    "trident_tier1_active": r["trident_tier1_active"],
                }
                w.writerow(row)

    now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")
    lines: list[str] = []
    A = lines.append
    A(f"# Decision Engine v1 — Tomorrow Watchlist (post-Class-B {asof})")
    A("")
    A(f"- Generated (UTC): {now_utc}")
    A(f"- Source events: `output/decision_engine_v1_events.parquet` (Tier 2 PASS post-Class-B {asof})")
    A(f"- asof_date in events: {asof}")
    A(f"- Total events processed: {total}")
    if degraded["degraded_mode"]:
        A(f"- **Run mode: DEGRADED** (paper_origin(s) excluded: "
          f"{', '.join(degraded['excluded_paper_origins']) or 'none'})")
    A("")

    if degraded["degraded_mode"]:
        A("## ⚠ Degraded mode")
        A("")
        for src in degraded["degraded_sources"]:
            if src == "line_e":
                asof_max = degraded["line_e_signal_asof_max"]
                origin = "EXTENDED"
                line_label = "Line E"
            elif src == "line_tr":
                asof_max = degraded["line_tr_signal_asof_max"]
                origin = "TRIGGER_RETEST"
                line_label = "Line TR"
            else:
                asof_max = None
                origin = src.upper()
                line_label = src
            A(
                f"- **{line_label} stale** (signal_asof_max="
                f"{asof_max or '<unknown>'} < target {asof}). "
                f"`{origin}` paper_origin events were excluded from this watchlist."
            )
        A(
            "- Watchlist is **partial**: the other paper stream remained fresh "
            "and produced events; this is NOT a full-coverage day."
        )
        A(
            "- Do **not** infer that excluded-origin events are unfit — they are "
            "absent because their source did not advance past the prior session."
        )
        A("")

    A("## Hard warnings")
    A("")
    A("- **NOT live auto-trade.** Manual-review watchlist only.")
    A("- **Live gate CLOSED.** No `live_execution_allowed=True` row reaches an executor.")
    A("- **Manual review required** before any paper or live action.")
    A("- **EXECUTABLE = LOCKED v1 risk branch passed thresholds (risk_atr ≤ 2.0).** This is NOT portfolio approval; gap, liquidity, regime context still need review.")
    A("- **No score, no rank, no portfolio_weight emitted.** No PF/WR/meanR. No backtest.")
    A("")
    A("## Distribution")
    A("")
    A("| section | count |")
    A("|---|---:|")
    sec_total = 0
    for sec in SECTIONS:
        n = len(sorted_sections[sec])
        sec_total += n
        A(f"| {sec} | {n} |")
    A(f"| **total** | **{sec_total}** |")
    A("")

    A("## Source distribution per section")
    A("")
    A("| section | source | count |")
    A("|---|---|---:|")
    for sec in SECTIONS:
        for src, n in sorted(source_dist[sec].items(), key=lambda x: (-x[1], x[0])):
            A(f"| {sec} | {src} | {n} |")
    A("")

    A("## EXECUTABLE overlap summary (descriptive)")
    A("")
    multi_sorted = sorted(
        [(tk, overlap_counts[tk]) for tk in multi_tickers],
        key=lambda x: (-x[1], x[0]),
    )
    overlap_rows = sum(overlap_counts[tk] for tk in multi_tickers)
    if multi_sorted:
        A(f"_{len(multi_sorted)} tickers carry >1 EXECUTABLE row across (family, timeframe) cells. {overlap_rows} of {len(exec_rows)} EXECUTABLE rows participate. Listed in descending overlap count._")
        A("")
        A("| ticker | exec rows | family × timeframe cells |")
        A("|---|---:|---|")
        ticker_cells: dict[str, list[str]] = defaultdict(list)
        for r in sorted_exec:
            ticker_cells[r["ticker"]].append(f"{r['family']}@{r['timeframe']}")
        for tk, cnt in multi_sorted:
            cells = ", ".join(ticker_cells[tk])
            A(f"| {tk} | {cnt} | {cells} |")
    else:
        A("_No tickers carry >1 EXECUTABLE row._")
    A("")

    def write_section_table(section: str, rows: list[dict], extra_note: str = ""):
        A(f"## {section} ({len(rows)} rows)")
        A("")
        if extra_note:
            A(f"_{extra_note}_")
            A("")
        A("| ticker | source | family | state | tf | setup | exec | regime | risk_atr | entry | stop | atr | fill | paper_ref | reasons | trident |")
        A("|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|")
        for r in rows:
            trident_cell = r["trident_tag"]
            if r["trident_D_pct"] or r["trident_SIL_pct_below_B"]:
                d_pct = r["trident_D_pct"] or "—"
                sil = r["trident_SIL_pct_below_B"] or "—"
                trident_cell = (
                    f"{r['trident_tag']} (D%={d_pct}, SIL={sil})"
                    if r["trident_tag"]
                    else f"(D%={d_pct}, SIL={sil})"
                )
            if r["trident_weekly_family"]:
                wf = r["trident_weekly_family"]
                trident_cell = (
                    f"{trident_cell} [wk:{wf}]" if trident_cell
                    else f"[wk:{wf}]"
                )
            if r["trident_tier1_active"]:
                trident_cell = (
                    f"{trident_cell} [t1]" if trident_cell else "[t1]"
                )
            A(
                "| "
                + " | ".join([
                    r["ticker"],
                    r["source"],
                    r["family"],
                    r["state"],
                    r["timeframe"],
                    r["setup_label"],
                    r["execution_label"],
                    r["market_context"],
                    _fmt_risk(r["risk_atr"]),
                    _fmt_num(r["entry_ref"], 4),
                    _fmt_num(r["stop_ref"], 4),
                    _fmt_num(r["atr"], 4),
                    r["fill_assumption"],
                    r["paper_stream_ref"],
                    r["reason_codes"],
                    trident_cell,
                ])
                + " |"
            )
        A("")

    exec_note = ""
    if multi_sorted:
        exec_note = f"Overlap candidates: {len(multi_sorted)} tickers carry >1 EXECUTABLE row ({overlap_rows} rows). Listed first."
    write_section_table("EXECUTABLE", sorted_sections["EXECUTABLE"], exec_note)
    write_section_table("SIZE_REDUCED", sorted_sections["SIZE_REDUCED"])
    write_section_table("WAIT_BETTER_ENTRY", sorted_sections["WAIT_BETTER_ENTRY"])
    write_section_table("WAIT_RETEST", sorted_sections["WAIT_RETEST"])
    write_section_table("PAPER_ONLY", sorted_sections["PAPER_ONLY"])

    out_md.write_text("\n".join(lines), encoding="utf-8")

    md_bytes = out_md.read_bytes()
    csv_bytes = out_csv.read_bytes()
    md_sha = hashlib.sha256(md_bytes).hexdigest()
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()

    print(f"=== watchlist post-class-b {asof} ===")
    print(f"asof_date: {asof}")
    if degraded["degraded_mode"]:
        print(
            f"run_mode: DEGRADED degraded_sources={degraded['degraded_sources']} "
            f"excluded_paper_origins={degraded['excluded_paper_origins']}"
        )
    else:
        print("run_mode: NORMAL")
    print(f"total events processed: {total}")
    for sec in SECTIONS:
        print(f"  {sec}: {len(sorted_sections[sec])}")
    print(f"  total sectioned: {sec_total}")
    print(f"EXECUTABLE multi-cell tickers: {len(multi_sorted)} ({overlap_rows} rows)")
    if TRIDENT_PROBE_PATH.exists():
        print(
            f"trident attachments: {trident_attach_count}/{total} rows enriched "
            f"(source=mb_scanner + setup_kind=above_mb_birth eligible cohort)"
        )
        print(
            f"weekly confluence (lead-lag): {weekly_confluence_count} rows tagged "
            f"{WEEKLY_CONFLUENCE_TAG} (LEAD weekly bar={prior_friday}; "
            f"LAG current-week window=({prior_friday}, {asof}]; "
            f"eligible_fresh_families={sorted(WEEKLY_CONFLUENCE_ELIGIBLE_FAMILIES)}; "
            f"weekly_birth_tickers={len(weekly_birth_tickers)}; "
            f"current_week_subweekly_births={len(current_week_subweekly_births)})"
        )
        sil_t33_str = (
            f"{tier1_summary['sil_t33']:.2f}"
            if tier1_summary["sil_t33"] is not None else "—"
        )
        bos_t67_str = (
            f"{tier1_summary['bos_t67']:.2f}"
            if tier1_summary["bos_t67"] is not None else "—"
        )
        xu100_ret_str = (
            f"{xu100_state['ret_20d_pct']:+.2f}%"
            if xu100_state.get("available") else "n/a"
        )
        print(
            f"trident tier-1: {tier1_tag_count} rows tagged {TIER1_TAG} "
            f"(family={TIER1_FAMILY}; universe={tier1_summary['universe_size']}; "
            f"G1={tier1_summary['g1_pass']} G2={tier1_summary['g2_pass']} "
            f"G3={tier1_summary['g3_pass']} G4_pass={tier1_summary['g4_pass']}; "
            f"sil_t33={sil_t33_str} bos_t67={bos_t67_str} "
            f"trailing_n={tier1_summary['trailing_n']}; "
            f"xu100_ret_20d={xu100_ret_str} "
            f"xu100_asof_available={xu100_state.get('asof_available') or '—'} "
            f"xu100_days_stale={xu100_state.get('days_stale', 0)} "
            f"xu100_reason={xu100_state.get('reason','?')}; "
            f"tier1_reason={tier1_summary['reason']})"
        )
    else:
        print(
            f"trident attachments: 0/{total} (probe parquet absent at "
            f"{TRIDENT_PROBE_PATH.relative_to(ROOT)} — Stage 7.5 skipped or unavailable)"
        )
        print(
            f"weekly confluence: 0 (probe parquet absent — confluence skipped)"
        )
        print(
            f"trident tier-1: 0 (probe parquet absent — tier-1 skipped)"
        )
    print(f"OUT_MD: {out_md} bytes={len(md_bytes)} sha256={md_sha}")
    print(f"OUT_CSV: {out_csv} bytes={len(csv_bytes)} sha256={csv_sha}")


if __name__ == "__main__":
    main()
