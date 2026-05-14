"""Decision Engine v1 — Panel Refresh Runner.

LOCKED: memory/decision_engine_v1_panel_refresh_execution_spec.md (DRAFT v3
LOCKed 2026-05-05). Single-fire authorized execution.

Refreshes output/decision_v0_classification_panel.parquet from current
extfeed + scanner EVENT-HISTORY sources. Closes the Class A panel-carry
gap by emitting two additional columns on the panel:

  - `atr` carried verbatim from upstream scanner emission (mb_scanner
    `atr_at_event`, horizontal_base `common__atr_14`, nyxexpansion
    `atr_14`).
  - `fill_assumption` per the LOCK §7 Q10 / spec §4.6.2 per-source/state
    contract (replaces v0 hardcoded "unresolved" with deterministic
    enum values).

Discipline:
  - EVENT-HISTORY sources only; SNAPSHOT paths NOT authorized (spec §1.4).
  - No scanner code modification; no v0 backtest code modification (we
    import its build_daily_panel/attach_forward_paths/replay_v0).
  - No Decision Engine v1 mapping/risk/label change.
  - No Tier 2/Tier 3 rerun.
  - No ranking/portfolio/live integration.
  - No forward returns recompute beyond what v0's pipeline already produces.
  - Atomic write via tempfile + os.replace; archive backup mode 0444 BEFORE
    replacing the panel; protected files (paper streams, Tier 2/3 outputs)
    sha256+size byte-equal pre/post.
  - Class A carry-fix verification: per executable-capable cell `atr_pct>0`
    AND `fill_pct>0` post-refresh; else fail-fast.
  - JSON manifest at output/decision_v1_panel_refresh_manifest.json.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from decision_engine.risk import derive_risk  # noqa: E402
from decision_engine.phase import map_phase  # noqa: E402
from decision_engine.schema import EVENT_COLUMNS, canonical_family  # noqa: E402

# Re-use v0 backtest pipeline pieces (read-only import; no modification).
# v0's `attach_forward_paths` applies a 20-trading-day cutoff buffer that is
# correct for backtest (forward-return validity) but wrong for v1 panel
# refresh (we need panel_max_date == LCTD per spec §3). v1 does NOT need
# forward-return columns (§4.5 forbidden). We therefore use a v1-specific
# extfeed-alignment filter (`_align_to_extfeed`) that keeps events up to
# LCTD without the 20-day buffer, and we still call v0's `replay_v0` for
# the regime + handoff + hard-rule pipeline (its forward_cols re-attach is
# a no-op when those columns are absent on the input events).
from tools.decision_engine_v0_classification_backtest import (  # noqa: E402
    build_daily_panel,
    replay_v0,
    _MB_FAMILIES,
)

OUT = ROOT / "output"
ARCHIVE = OUT / "_archive"

PANEL_PATH = OUT / "decision_v0_classification_panel.parquet"
EXTFEED_PATH = OUT / "extfeed_intraday_1h_3y_master.parquet"
MANIFEST_PATH = OUT / "decision_v1_panel_refresh_manifest.json"

PAPER_E_PATH = OUT / "paper_execution_v0_trades.parquet"
PAPER_TR_PATH = OUT / "paper_execution_v0_trigger_retest_trades.parquet"

# Tier 2 / Tier 3 PASS artefacts that MUST stay byte-equal pre/post.
TIER2_EVENTS_PATH = OUT / "decision_engine_v1_events.parquet"
TIER2_DIST_PATH = OUT / "decision_engine_v1_tier2_label_distribution.csv"
TIER2_LINK_PATH = OUT / "decision_engine_v1_paper_link_integrity.csv"
TIER2_SUMMARY_PATH = OUT / "decision_engine_v1_tier2_label_table_summary.md"
TIER3_REPORT_PATH = OUT / "decision_engine_v1_tier3_review_report.md"

PROTECTED_FILES = (
    PAPER_E_PATH,
    PAPER_TR_PATH,
    TIER2_EVENTS_PATH,
    TIER2_DIST_PATH,
    TIER2_LINK_PATH,
    TIER2_SUMMARY_PATH,
    TIER3_REPORT_PATH,
)

# Snapshot paths to detect (and ignore) per spec §5.6.
SNAPSHOT_PATHS = [
    OUT / f"mb_scanner_{fam}_{tf}.parquet"
    for fam, tf in [("mb", "5h"), ("mb", "1d"), ("mb", "1w"), ("mb", "1M"),
                     ("bb", "5h"), ("bb", "1d"), ("bb", "1w"), ("bb", "1M")]
]
SNAPSHOT_PATHS += [
    OUT / "horizontal_base_live_2026-04-29.parquet",
    OUT / "horizontal_base_live_2026-04-30.parquet",
]

NOX_HIST_DIR = OUT / "_nox_historical"
NOX_RT_RE = re.compile(r"^nox_v3_signals_(\d{8})\.csv$")
NOX_WK_RE = re.compile(r"^nox_v3_signals_weekly_(\d{8})\.csv$")
NYXMOM_PICKS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(m0|v5)_picks\.csv$")


# ─── helpers ──────────────────────────────────────────────────────────────


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": _sha256_of(path),
    }


def _empty_event() -> dict:
    return {col: None for col in EVENT_COLUMNS}


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _fail(reason: str, *, manifest_partial: dict | None = None) -> None:
    """Halt with explicit reason; leave panel + protected files untouched."""
    print(f"[panel-refresh] FAIL {reason}", flush=True)
    if manifest_partial is not None:
        # Surface partial diagnostic on stderr only; do NOT write a manifest.
        sys.stderr.write(json.dumps(manifest_partial, default=str, indent=2))
        sys.stderr.write("\n")
    sys.exit(2)


# ─── snapshot vs event-history disambiguation (spec §1.4 / §5.6) ──────────


def detect_snapshot_files() -> list[str]:
    """Return paths of snapshot files that exist on disk (informational only)."""
    return [str(p.relative_to(ROOT)) for p in SNAPSHOT_PATHS if p.exists()]


# ─── per-source event loaders (mirror v0 backtest verbatim, plus carry) ───
#
# The runner re-uses v0 backtest's filter predicates and column names verbatim,
# but additionally:
#   - Emits the additional `_atr_carry` column carrying the upstream ATR value.
#   - Sets `fill_assumption` per the LOCK §7 Q10 contract (replacing v0's
#     hardcoded "unresolved").
# v0 backtest source remains UNTOUCHED.


def load_mb_scanner_events_v1() -> pd.DataFrame:
    rows = []
    for fam, tf in _MB_FAMILIES:
        path = OUT / f"mb_scanner_events_{fam}.parquet"
        if not path.exists():
            _fail(f"event_history_source_missing: {path.relative_to(ROOT)}")
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            ev_type = str(r.get("event_type", ""))
            family_key = f"{fam}__{ev_type}"
            entry_ref = r.get("event_close")
            stop_ref = r.get("structural_invalidation_low")
            atr = r.get("atr_at_event")
            risk_pct, risk_atr = derive_risk(
                entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
            )
            ev = _empty_event()
            ev.update(
                date=pd.to_datetime(r.get("event_bar_date")).date()
                if pd.notna(r.get("event_bar_date")) else None,
                ticker=str(r.get("ticker", "")),
                source="mb_scanner",
                family=canonical_family(family_key),
                state=ev_type,
                phase=map_phase(source="mb_scanner", family=family_key, state=ev_type),
                timeframe=tf,
                direction="long",
                raw_signal_present=True,
                entry_ref=float(entry_ref) if pd.notna(entry_ref) else None,
                stop_ref=float(stop_ref) if pd.notna(stop_ref) else None,
                risk_pct=risk_pct,
                risk_atr=risk_atr,
                extension_atr=r.get("bos_distance_atr_at_event"),
                liquidity_score=None,
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=None,
                fill_assumption="close_based_signal",  # Q10 contract
                bar_timestamp=r.get("event_ts"),
            )
            ev["_atr_carry"] = float(atr) if pd.notna(atr) else None
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    return pd.DataFrame(rows)


def load_horizontal_base_events_v1() -> pd.DataFrame:
    path = OUT / "horizontal_base_event_v1.parquet"
    if not path.exists():
        _fail(f"event_history_source_missing: {path.relative_to(ROOT)}")
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        state = str(r.get("signal_state", ""))
        family_key = f"horizontal_base__{state}"
        entry_ref = (
            r.get("entry_reference_price")
            if pd.notna(r.get("entry_reference_price"))
            else r.get("family__trigger_level")
        )
        stop_ref = r.get("invalidation_level")
        atr = r.get("common__atr_14")
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
        )
        # Q10 contract for HB: "close_or_event_price_from_scanner" if the event
        # has an explicit entry_reference_price; otherwise "close_based_signal".
        if pd.notna(r.get("entry_reference_price")):
            fa = "close_or_event_price_from_scanner"
        else:
            fa = "close_based_signal"
        ev = _empty_event()
        ev.update(
            date=pd.to_datetime(r.get("bar_date")).date()
            if pd.notna(r.get("bar_date")) else None,
            ticker=str(r.get("ticker", "")),
            source="horizontal_base",
            family=family_key,
            state=state,
            phase=map_phase(source="horizontal_base", family=family_key, state=state),
            timeframe="1d",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if pd.notna(entry_ref) else None,
            stop_ref=float(stop_ref) if pd.notna(stop_ref) else None,
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            extension_atr=r.get("common__extension_from_trigger"),
            liquidity_score=r.get("common__liquidity_score"),
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=None,
            fill_assumption=fa,
            bar_timestamp=r.get("as_of_ts"),
        )
        ev["_atr_carry"] = float(atr) if pd.notna(atr) else None
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    return pd.DataFrame(rows)


def load_nyxexpansion_events_v1() -> pd.DataFrame:
    path = OUT / "nyxexp_dataset_v4.parquet"
    if not path.exists():
        _fail(f"event_history_source_missing: {path.relative_to(ROOT)}")
    df = pd.read_parquet(path)
    fired = df[
        (df["close"] > df["prior_high_20"])
        & (df["rvol"] >= 1.5)
        & (df["close_loc"] >= 0.70)
    ].copy()
    rows = []
    for _, r in fired.iterrows():
        entry_ref = r.get("close")
        atr = r.get("atr_14")
        ets = r.get("entry_to_stop_atr")
        stop_ref = None
        if pd.notna(entry_ref) and pd.notna(atr) and pd.notna(ets):
            stop_ref = float(entry_ref) - float(atr) * float(ets)
        risk_pct, risk_atr = derive_risk(
            entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
        )
        ev = _empty_event()
        ev.update(
            date=pd.to_datetime(r.get("date")).date()
            if pd.notna(r.get("date")) else None,
            ticker=str(r.get("ticker", "")),
            source="nyxexpansion",
            family="nyxexpansion__triggerA",
            state="trigger",
            phase="trigger",
            timeframe="1d",
            direction="long",
            raw_signal_present=True,
            entry_ref=float(entry_ref) if pd.notna(entry_ref) else None,
            stop_ref=stop_ref,
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            extension_atr=r.get("dist_above_trigger_atr"),
            liquidity_score=None,
            higher_tf_context=None,
            lower_tf_context=None,
            reason_candidates=[],
            raw_score=None,
            fill_assumption="close_based_signal",  # Q10 contract
        )
        ev["_atr_carry"] = float(atr) if pd.notna(atr) else None
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    return pd.DataFrame(rows)


def load_nyxmomentum_events_v1() -> pd.DataFrame:
    live_dir = OUT / "nyxmomentum" / "live"
    if not live_dir.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    rows = []
    for f in sorted(live_dir.iterdir()):
        m = NYXMOM_PICKS_RE.match(f.name)
        if not m:
            continue
        asof = m.group(1)
        variant = m.group(2)
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        for _, r in df.iterrows():
            ev = _empty_event()
            ev.update(
                date=pd.to_datetime(asof).date(),
                ticker=str(r.get("ticker", "")),
                source="nyxmomentum",
                family=f"nyxmomentum__{variant}",
                state="strength_top_decile",
                phase="strength_context",
                timeframe="1d",
                direction="long",
                raw_signal_present=True,
                entry_ref=None,
                stop_ref=None,
                risk_pct=None,
                risk_atr=None,
                extension_atr=None,
                liquidity_score=None,
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=r.get("score") if pd.notna(r.get("score")) else None,
                fill_assumption="context_only_not_applicable",  # Q10
            )
            ev["_atr_carry"] = None
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    return pd.DataFrame(rows)


def _load_nox_v3_historical_v1(
    pattern: re.Pattern, *, source: str, family: str, timeframe: str,
    fill_assumption: str,
) -> pd.DataFrame:
    if not NOX_HIST_DIR.exists():
        _fail(f"event_history_source_missing: {NOX_HIST_DIR.relative_to(ROOT)}")
    rows = []
    for f in sorted(NOX_HIST_DIR.iterdir()):
        if not pattern.match(f.name):
            continue
        try:
            df = pd.read_csv(f, parse_dates=["pivot_date", "signal_date"])
        except Exception:
            continue
        if df.empty or "signal" not in df.columns:
            continue
        df = df[df["signal"] == "PIVOT_AL"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            entry_ref = r.get("close")
            stop_ref = r.get("pivot_price")
            risk_pct, risk_atr = derive_risk(
                entry_ref=entry_ref, stop_ref=stop_ref, atr=None
            )
            ev = _empty_event()
            ev.update(
                date=pd.to_datetime(r.get("signal_date")).date()
                if pd.notna(r.get("signal_date")) else None,
                ticker=str(r.get("ticker", "")),
                source=source,
                family=family,
                state="pivot_al",
                phase="trigger",
                timeframe=timeframe,
                direction="long",
                raw_signal_present=True,
                entry_ref=float(entry_ref) if pd.notna(entry_ref) else None,
                stop_ref=float(stop_ref) if pd.notna(stop_ref) else None,
                risk_pct=risk_pct,
                risk_atr=risk_atr,
                extension_atr=None,
                liquidity_score=None,
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=r.get("rg_score") if pd.notna(r.get("rg_score")) else None,
                fill_assumption=fill_assumption,
            )
            ev["_atr_carry"] = None
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_atr_carry"])
    return pd.DataFrame(rows)


def load_nox_rt_daily_events_v1() -> pd.DataFrame:
    return _load_nox_v3_historical_v1(
        NOX_RT_RE, source="nox_rt_daily",
        family="nox_rt_daily__pivot_al", timeframe="1d",
        fill_assumption="context_only_not_applicable",
    )


def load_nox_weekly_events_v1() -> pd.DataFrame:
    return _load_nox_v3_historical_v1(
        NOX_WK_RE, source="nox_weekly",
        family="nox_weekly__weekly_pivot_al", timeframe="1w",
        fill_assumption="wait_confirmation_not_executable",
    )


# ─── source freshness check (spec §1.2.4 / §3) ────────────────────────────


def _source_max_date(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    d = pd.to_datetime(df["date"]).dropna().max()
    return d.date().isoformat() if pd.notna(d) else None


# ─── v1-specific extfeed alignment (replaces v0 attach_forward_paths) ─────


def _align_to_extfeed(
    events: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Drop events whose (ticker, date) is not in extfeed daily index.

    Differs from v0 attach_forward_paths in two ways: (1) no 20-trading-day
    cutoff buffer (panel must reach LCTD per spec §3); (2) no forward-return
    columns attached (v1 doesn't need them; they're §4.5 forbidden).

    Returns (filtered_events, skipped_no_ticker, skipped_no_tdx).
    """
    if events.empty:
        return events, 0, 0
    by_ticker: dict[str, dict] = {}
    for tkr, grp in daily.groupby("ticker"):
        g = grp.reset_index(drop=True)
        by_ticker[tkr] = {
            "tdx_by_date": dict(zip(g["date"], g["tdx"])),
        }
    out_rows = []
    skipped_no_ticker = 0
    skipped_no_tdx = 0
    for ev in events.to_dict(orient="records"):
        tkr = ev.get("ticker")
        d = ev.get("date")
        if tkr not in by_ticker:
            skipped_no_ticker += 1
            continue
        if d not in by_ticker[tkr]["tdx_by_date"]:
            skipped_no_tdx += 1
            continue
        out_rows.append(ev)
    if not out_rows:
        return pd.DataFrame(columns=events.columns), skipped_no_ticker, skipped_no_tdx
    return pd.DataFrame(out_rows), skipped_no_ticker, skipped_no_tdx


# ─── orchestrator ─────────────────────────────────────────────────────────


def main() -> int:
    t0 = time.time()
    refresh_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    print(f"[panel-refresh] start {refresh_utc}", flush=True)

    # ─── 1. pre-write protected-file snapshot ─────────────────────────────
    print("[panel-refresh] computing pre-write sha256 of protected files …", flush=True)
    pre_meta = {str(p.relative_to(ROOT)): _file_meta(p) for p in PROTECTED_FILES}
    for p, m in pre_meta.items():
        if not m.get("exists"):
            _fail(f"protected_file_missing_pre: {p}")
        print(f"  {p}: size={m['size']} sha256={m['sha256'][:12]}…", flush=True)

    # Old panel meta for backup naming.
    if not PANEL_PATH.exists():
        _fail(f"old_panel_missing: {PANEL_PATH.relative_to(ROOT)}")
    old_panel_meta = _file_meta(PANEL_PATH)
    old_panel_max_date = pd.read_parquet(PANEL_PATH, columns=["date"])["date"].max()
    old_panel_max_date_iso = (
        old_panel_max_date.isoformat()
        if hasattr(old_panel_max_date, "isoformat")
        else str(old_panel_max_date)
    )

    # ─── 2. snapshot-vs-event-history discipline (spec §1.4 / §5.6) ──────
    snapshot_observed = detect_snapshot_files()
    if snapshot_observed:
        print(
            f"[panel-refresh] snapshot_path_observed_but_ignored: {snapshot_observed}",
            flush=True,
        )

    # ─── 3. extfeed precondition (spec §1.1 + §3) ────────────────────────
    if not EXTFEED_PATH.exists():
        _fail(f"extfeed_missing: {EXTFEED_PATH.relative_to(ROOT)}")
    extfeed_ts_max = pd.read_parquet(
        EXTFEED_PATH, columns=["ts_istanbul"]
    )["ts_istanbul"].max()
    extfeed_ts_max = pd.to_datetime(extfeed_ts_max)
    extfeed_max_date = extfeed_ts_max.date().isoformat()
    extfeed_meta = _file_meta(EXTFEED_PATH)
    print(
        f"[panel-refresh] extfeed: max_date={extfeed_max_date} "
        f"size={extfeed_meta['size']} sha256={extfeed_meta['sha256'][:12]}…",
        flush=True,
    )

    # latest_completed_trading_day per Q5: max completed trading date from
    # extfeed, rolled to TR-daily.
    lctd = extfeed_max_date

    # ─── 4. load events from all 6 event-history sources ─────────────────
    print("[panel-refresh] loading event-history sources …", flush=True)
    parts: dict[str, pd.DataFrame] = {}
    parts["mb_scanner"] = load_mb_scanner_events_v1()
    print(f"  mb_scanner: {len(parts['mb_scanner'])} rows", flush=True)
    parts["horizontal_base"] = load_horizontal_base_events_v1()
    print(f"  horizontal_base: {len(parts['horizontal_base'])} rows", flush=True)
    parts["nyxexpansion"] = load_nyxexpansion_events_v1()
    print(f"  nyxexpansion: {len(parts['nyxexpansion'])} rows", flush=True)
    parts["nyxmomentum"] = load_nyxmomentum_events_v1()
    print(f"  nyxmomentum: {len(parts['nyxmomentum'])} rows", flush=True)
    parts["nox_rt_daily"] = load_nox_rt_daily_events_v1()
    print(f"  nox_rt_daily: {len(parts['nox_rt_daily'])} rows", flush=True)
    parts["nox_weekly"] = load_nox_weekly_events_v1()
    print(f"  nox_weekly: {len(parts['nox_weekly'])} rows", flush=True)

    scanner_max_dates = {s: _source_max_date(df) for s, df in parts.items()}

    # ─── 5. executable-capable scanner freshness gate (spec §3) ──────────
    # Spec semantics: executable-capable scanners must REACH LCTD.
    # AHEAD-of-LCTD is acceptable (extfeed-side `_align_to_extfeed` will
    # bound the panel to extfeed dates). Only behind-LCTD is stale.
    executable_capable = ("mb_scanner", "horizontal_base", "nyxexpansion")
    ahead_of_extfeed: list[str] = []
    for s in executable_capable:
        m = scanner_max_dates.get(s)
        if m is None:
            _fail(f"scanner_stale: {s} (no rows)")
        if m < lctd:
            _fail(
                f"scanner_stale: {s} max={m} lctd={lctd}",
                manifest_partial={"scanner_max_dates": scanner_max_dates, "lctd": lctd},
            )
        if m > lctd:
            ahead_of_extfeed.append(f"{s}={m}")
    if ahead_of_extfeed:
        print(
            f"[panel-refresh] WARN ahead_of_extfeed (lctd={lctd}): "
            f"{', '.join(ahead_of_extfeed)} — rows past lctd will be dropped "
            f"by `_align_to_extfeed`; not a fail.",
            flush=True,
        )
    print(
        f"[panel-refresh] executable-capable freshness ok: "
        f"{ {s: scanner_max_dates[s] for s in executable_capable} }",
        flush=True,
    )

    # ─── 6. concat → daily panel → forward paths → replay v0 ─────────────
    events = pd.concat(
        [df for df in parts.values() if not df.empty], ignore_index=True
    )
    print(f"[panel-refresh] events total: {len(events)}", flush=True)

    print("[panel-refresh] building daily panel + trading-day index …", flush=True)
    daily = build_daily_panel()
    daily_max = str(daily["date"].max())
    print(f"  daily: {len(daily)} rows · max_date={daily_max}", flush=True)

    print("[panel-refresh] aligning events to extfeed daily index "
          "(no 20-day cutoff; v1 needs panel up to LCTD) …", flush=True)
    events_aligned, skipped_no_ticker, skipped_no_tdx = _align_to_extfeed(
        events, daily,
    )
    print(
        f"  aligned: {len(events_aligned)} rows "
        f"(skipped: no_ticker={skipped_no_ticker}, no_tdx={skipped_no_tdx})",
        flush=True,
    )
    events_with_path = events_aligned  # same name to keep downstream uniform

    # Shape 2 carry: rename _atr_carry → atr so apply_actions keeps it via the
    # extended ACTION_COLUMNS (decision_engine/schema.py). atr and
    # fill_assumption now ride through replay_v0/_attach_regime/apply_handoffs/
    # apply_actions on the same row, eliminating the previous date-sort vs
    # concat-order positional misalignment regression.
    if "_atr_carry" not in events_with_path.columns:
        _fail("shape2_precondition: _atr_carry missing on events_with_path")
    if "fill_assumption" not in events_with_path.columns:
        _fail("shape2_precondition: fill_assumption missing on events_with_path")
    events_with_path = events_with_path.rename(columns={"_atr_carry": "atr"})

    print("[panel-refresh] replaying v0 (handoffs + hard-rule) …", flush=True)
    actions, _events_full = replay_v0(events_with_path)
    print(f"  actions: {len(actions)} rows", flush=True)

    # ─── 7. validate Shape 2 carry — atr + fill_assumption rode through ──
    if len(actions) != len(events_with_path):
        _fail(
            f"replay_v0_length_mismatch: actions={len(actions)} "
            f"events={len(events_with_path)}"
        )
    for col in ("atr", "fill_assumption"):
        if col not in actions.columns:
            _fail(
                f"shape2_carry_missing: {col} not in actions after replay_v0; "
                "ACTION_COLUMNS may not include it"
            )
    print(
        f"[panel-refresh] shape2 carry verified: atr null_rate="
        f"{actions['atr'].isna().mean():.4f} "
        f"fill null_rate={actions['fill_assumption'].isna().mean():.4f}",
        flush=True,
    )

    # ─── 8. determinism — stable sort (spec §5.5 / §7 Q7) ───────────────
    sort_cols = ["source", "family", "phase", "timeframe", "ticker", "date"]
    actions = actions.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # ─── 9. data contract enforcement (spec §4) ──────────────────────────
    required_identity = ["date", "ticker", "source", "family", "phase", "timeframe"]
    for c in required_identity:
        if c not in actions.columns:
            _fail(f"panel_contract_violation: {c} missing")
        n_null = int(actions[c].isna().sum())
        if n_null > 0:
            _fail(f"panel_contract_violation: {c} has {n_null} nulls")
    for c in ("atr", "fill_assumption"):
        if c not in actions.columns:
            _fail(f"panel_contract_violation: {c} missing post carry-fix")

    # ─── 10. class A carry-fix verification (spec §4.6.3) ────────────────
    print("[panel-refresh] computing per-cell atr_pct + fill_pct …", flush=True)
    cell_keys = ["source", "family", "phase", "timeframe"]
    grp = actions.groupby(cell_keys, dropna=False)
    cell_stats = grp.agg(
        n=("date", "size"),
        atr_pct=("atr", lambda x: float((~pd.isna(x)).mean())),
        fill_pct=("fill_assumption", lambda x: float((~pd.isna(x)).mean())),
    ).reset_index()

    class_a_violations: list[dict] = []
    for _, r in cell_stats.iterrows():
        if r["source"] in executable_capable:
            if not (r["atr_pct"] > 0 and r["fill_pct"] > 0):
                class_a_violations.append({
                    "source": r["source"], "family": r["family"],
                    "phase": r["phase"], "timeframe": r["timeframe"],
                    "n": int(r["n"]),
                    "atr_pct": float(r["atr_pct"]),
                    "fill_pct": float(r["fill_pct"]),
                })
    if class_a_violations:
        _fail(
            "class_a_carry_fix_incomplete",
            manifest_partial={"violations": class_a_violations},
        )
    print(
        f"  class A cells verified: "
        f"{int((cell_stats['source'].isin(executable_capable)).sum())} "
        "(all atr_pct>0 AND fill_pct>0)",
        flush=True,
    )

    # ─── 11. dropped-row aggregation per Q9 ──────────────────────────────
    # attach_forward_paths drops rows; we report aggregate by source.
    # Compute pre-drop source breakdown vs. post-drop.
    pre_breakdown = (
        events.groupby("source").size().to_dict() if not events.empty else {}
    )
    post_breakdown = (
        actions.groupby("source").size().to_dict() if not actions.empty else {}
    )
    dropped_breakdown = {
        s: int(pre_breakdown.get(s, 0)) - int(post_breakdown.get(s, 0))
        for s in set(pre_breakdown) | set(post_breakdown)
    }
    print(f"[panel-refresh] dropped_by_source: {dropped_breakdown}", flush=True)

    # ATR carry miss aggregation (executable-capable rows whose atr is null).
    atr_carry_misses_by_cell: dict[str, int] = {}
    for _, r in cell_stats.iterrows():
        if r["source"] in executable_capable and r["atr_pct"] < 1.0:
            key = f"{r['source']}__{r['family']}__{r['phase']}__{r['timeframe']}"
            atr_carry_misses_by_cell[key] = int(round(r["n"] * (1 - r["atr_pct"])))
    if atr_carry_misses_by_cell:
        print(
            f"[panel-refresh] atr_carry_misses_by_cell: {atr_carry_misses_by_cell}",
            flush=True,
        )

    # ─── 12. archive backup (spec §2.2) — BEFORE panel write ────────────
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backup_filename = (
        f"decision_v0_classification_panel__asof_{old_panel_max_date_iso}__"
        f"sha256_{old_panel_meta['sha256'][:8]}.parquet"
    )
    backup_path = ARCHIVE / backup_filename
    backup_sidecar = ARCHIVE / f"manifest_asof_{old_panel_max_date_iso}.json"

    if backup_path.exists():
        _fail(f"archive_backup_already_exists: {backup_path.relative_to(ROOT)}")

    print(f"[panel-refresh] writing archive backup → {backup_path.relative_to(ROOT)}", flush=True)
    shutil.copy2(PANEL_PATH, backup_path)
    # mode 0444 (read-only)
    os.chmod(backup_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    backup_meta = _file_meta(backup_path)

    # Verify backup byte-equals old panel.
    if (backup_meta["size"] != old_panel_meta["size"]
            or backup_meta["sha256"] != old_panel_meta["sha256"]):
        _fail(
            "archive_backup_mismatch",
            manifest_partial={
                "old_panel": old_panel_meta, "backup": backup_meta,
            },
        )
    sidecar_doc = {
        "panel_path": str(PANEL_PATH.relative_to(ROOT)),
        "panel_size": old_panel_meta["size"],
        "panel_sha256": old_panel_meta["sha256"],
        "panel_max_date": old_panel_max_date_iso,
        "backup_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    backup_sidecar.write_text(json.dumps(sidecar_doc, indent=2), encoding="utf-8")

    # ─── 13. atomic panel write (spec §2.1 / §5.1) ───────────────────────
    print("[panel-refresh] writing new panel atomically …", flush=True)
    panel_out = actions.copy()
    if "reason_codes" in panel_out.columns:
        panel_out["reason_codes"] = panel_out["reason_codes"].apply(
            lambda v: [str(x) for x in v]
            if isinstance(v, (list, tuple)) else []
        )
    if "horizon_review_due" in panel_out.columns:
        panel_out["horizon_review_due"] = panel_out["horizon_review_due"].astype("string")

    fd, tmp_path_str = tempfile.mkstemp(
        prefix=".decision_v0_classification_panel_tmp_",
        suffix=".parquet",
        dir=str(OUT),
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        panel_out.to_parquet(tmp_path, index=False)
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        _fail(f"panel_write_failed: {e}")

    # Verify protected files BYTE-EQUAL before final replace.
    print("[panel-refresh] verifying protected files byte-equal pre-replace …", flush=True)
    for p, pre in pre_meta.items():
        cur = _file_meta(ROOT / p)
        if cur != pre:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            _fail(
                f"protected_file_mutated_pre_replace: {p}",
                manifest_partial={"pre": pre, "post": cur},
            )

    os.replace(tmp_path, PANEL_PATH)

    # Verify protected files BYTE-EQUAL after replace.
    print("[panel-refresh] verifying protected files byte-equal post-replace …", flush=True)
    post_meta = {p: _file_meta(ROOT / p) for p in pre_meta}
    for p, pre in pre_meta.items():
        if post_meta[p] != pre:
            _fail(
                f"protected_file_mutated_post_replace: {p}",
                manifest_partial={"pre": pre, "post": post_meta[p]},
            )

    # New panel meta.
    new_panel_meta = _file_meta(PANEL_PATH)
    new_panel_df = pd.read_parquet(PANEL_PATH, columns=["date", "ticker", "source", "family", "phase", "timeframe"])
    new_panel_max = str(pd.to_datetime(new_panel_df["date"]).max().date())
    new_panel_min = str(pd.to_datetime(new_panel_df["date"]).min().date())
    n_unique_cells = int(new_panel_df.groupby(
        ["source", "family", "phase", "timeframe"], dropna=False
    ).ngroups)
    n_unique_tickers = int(new_panel_df["ticker"].nunique())

    # ─── 14. freshness target validation ─────────────────────────────────
    if new_panel_max != lctd:
        # Backup chain stays in place; the panel write succeeded but failed
        # the freshness target. Per spec §3 this is fail-fast.
        # Restore backup over written panel to keep operational state untouched.
        shutil.copy2(backup_path, PANEL_PATH)
        _fail(
            f"panel_max_date_below_lctd: panel_max={new_panel_max} lctd={lctd}",
        )

    # paper_max - panel_max relation
    paper_e_max = pd.read_parquet(
        PAPER_E_PATH, columns=["asof_date"]
    )["asof_date"].max()
    paper_tr_max = pd.read_parquet(
        PAPER_TR_PATH, columns=["asof_date"]
    )["asof_date"].max()
    paper_e_max_iso = pd.to_datetime(paper_e_max).date().isoformat()
    paper_tr_max_iso = pd.to_datetime(paper_tr_max).date().isoformat()
    paper_max = max(paper_e_max_iso, paper_tr_max_iso)
    if new_panel_max < paper_max:
        shutil.copy2(backup_path, PANEL_PATH)
        _fail(
            f"panel_max_below_paper_max: panel={new_panel_max} paper={paper_max}",
        )

    # ─── 15. manifest emit (spec §2.3) ───────────────────────────────────
    print(f"[panel-refresh] writing manifest → {MANIFEST_PATH.relative_to(ROOT)}", flush=True)
    null_counts = {c: int(panel_out[c].isna().sum()) for c in panel_out.columns}

    cell_stats_records = []
    for _, r in cell_stats.iterrows():
        cell_stats_records.append({
            "source": r["source"], "family": r["family"],
            "phase": r["phase"], "timeframe": r["timeframe"],
            "n": int(r["n"]),
            "atr_pct": round(float(r["atr_pct"]), 6),
            "fill_pct": round(float(r["fill_pct"]), 6),
        })

    manifest = {
        "spec": "memory/decision_engine_v1_panel_refresh_execution_spec.md",
        "spec_version": "DRAFT v3 LOCKED 2026-05-05",
        "refresh_utc": refresh_utc,
        "refresh_runner_git_sha": _git_sha(),
        "extfeed_source_path": str(EXTFEED_PATH.relative_to(ROOT)),
        "extfeed_max_date": extfeed_max_date,
        "extfeed_size": extfeed_meta["size"],
        "extfeed_sha256": extfeed_meta["sha256"],
        "scanner_source_max_dates": scanner_max_dates,
        "executable_capable_scanner_max_dates": {
            s: scanner_max_dates[s] for s in executable_capable
        },
        "nox_rt_daily_max_date": scanner_max_dates.get("nox_rt_daily"),
        "nox_weekly_max_date": scanner_max_dates.get("nox_weekly"),
        "paper_stream_max_dates": {
            "line_e": paper_e_max_iso,
            "line_tr": paper_tr_max_iso,
        },
        "paper_stream_sha256_pre": {
            "line_e": pre_meta[str(PAPER_E_PATH.relative_to(ROOT))]["sha256"],
            "line_tr": pre_meta[str(PAPER_TR_PATH.relative_to(ROOT))]["sha256"],
        },
        "paper_stream_sha256_post": {
            "line_e": post_meta[str(PAPER_E_PATH.relative_to(ROOT))]["sha256"],
            "line_tr": post_meta[str(PAPER_TR_PATH.relative_to(ROOT))]["sha256"],
        },
        "protected_files_pre": pre_meta,
        "protected_files_post": post_meta,
        "panel_path": str(PANEL_PATH.relative_to(ROOT)),
        "panel_size": new_panel_meta["size"],
        "panel_sha256": new_panel_meta["sha256"],
        "panel_row_count": int(len(new_panel_df)),
        "panel_max_date": new_panel_max,
        "panel_min_date": new_panel_min,
        "panel_unique_cells": n_unique_cells,
        "panel_unique_tickers": n_unique_tickers,
        "panel_columns": list(panel_out.columns),
        "panel_schema_version": "v0_event_history_assembly__plus_atr_fill_assumption_v1",
        "panel_per_column_null_counts": null_counts,
        "latest_completed_trading_day": lctd,
        "panel_max_date_equals_lctd": (new_panel_max == lctd),
        "backup_path": str(backup_path.relative_to(ROOT)),
        "backup_size": backup_meta["size"],
        "backup_sha256": backup_meta["sha256"],
        "backup_sidecar_path": str(backup_sidecar.relative_to(ROOT)),
        "old_panel_size": old_panel_meta["size"],
        "old_panel_sha256": old_panel_meta["sha256"],
        "old_panel_max_date": old_panel_max_date_iso,
        "row_count_delta": int(len(new_panel_df))
            - int(pd.read_parquet(backup_path, columns=["date"]).shape[0]),
        "per_cell_carry_fix_stats": cell_stats_records,
        "dropped_by_source": dropped_breakdown,
        "atr_carry_misses_by_cell": atr_carry_misses_by_cell,
        "snapshot_path_observed_but_ignored": snapshot_observed,
        "source_path_disambiguation_check": True,
        "lock_decisions": {
            "Q1": "extfeed_intraday_1h_3y_master.parquet rolled to TR-daily internally",
            "Q2": "event-history sources only (spec §1.2)",
            "Q3": "output/_archive/decision_v0_classification_panel__asof_<d>__sha256_<8>.parquet mode 0444",
            "Q4": "JSON at output/decision_v1_panel_refresh_manifest.json",
            "Q5": "max completed trading date from extfeed",
            "Q6": "spec §4 column contract",
            "Q7": "deterministic with stable sort by (source,family,phase,timeframe,ticker,date)",
            "Q8": "tools/decision_engine_v1_panel_refresh.py (this runner)",
            "Q9": "aggregate by reason + source/family/state/timeframe",
            "Q10": "spec §4.6.2 per-source/state contract",
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )

    elapsed = time.time() - t0
    print(f"[panel-refresh] PASS — completed in {elapsed:.1f}s", flush=True)
    print(f"  panel: {new_panel_meta['size']} B / sha256 {new_panel_meta['sha256'][:12]}…", flush=True)
    print(f"  rows: {len(new_panel_df)} (was {pd.read_parquet(backup_path, columns=['date']).shape[0]})", flush=True)
    print(f"  panel_max_date: {new_panel_max} (lctd={lctd})", flush=True)
    print(f"  unique cells: {n_unique_cells} · unique tickers: {n_unique_tickers}", flush=True)
    print(f"  manifest: {MANIFEST_PATH.relative_to(ROOT)}", flush=True)
    print(f"  backup: {backup_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
