"""Decision Engine v1 — EMA forward-alignment orchestrator (single-fire).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Operation: forward-emission alignment after HB event-history grew from
10,470 → 10,582 rows. Locked Pilot 3-7 research outputs remain byte-equal;
new `_forward.*` siblings are emitted under their own namespace via key-based
remap (stable_event_key 5-tuple). One-shot research-frozen breakpoints sidecar.

Cascade (halt-on-FAIL):
    Step 1  tools/ema_context_pilot3_forward.py  → 2 files
    Step 2  tools/ema_context_pilot4_forward.py  → 5 files
    Step 3  tools/ema_context_pilot5_forward.py  → 6 files
    Step 4  tools/ema_context_pilot6_forward.py  → 5 files
    Step 5  tools/ema_context_pilot7_forward.py  → 6 files

Phases:
    A. Pre-state capture (forward outputs + 31 locked-research + 24 protected
                          non-target + sidecar; Q8 + sub-ceiling baseline)
    B. One-shot research_frozen sidecar emission (mode 0444; refuses to
       clobber on subsequent runs)
       + archive any pre-existing forward outputs (skip locked-research)
    C. Run 5 forward pilot subprocesses in sequence (halt-on-FAIL); capture
       each pilot's PILOTn_FORWARD_SUMMARY_JSON_BEGIN/END block
    D. Validate: §3.5 key_remap counts match Pilot 3 summary + sub-ceiling
       satisfied + Q8 ceilings satisfied; locked-research 31 files byte-equal;
       24 protected byte-equal; per-pilot row reconciliation; forward panel
       max_date >= operational_target_date - max_offset_window
    E. Manifest at output/decision_v1_ema_forward_alignment_manifest.json

Forbidden: paper_execution, Tier 2/3, ranking, portfolio, live trade gate,
market-data download, git mutation, modification of locked pilot scripts,
modification of locked Pilot 3-7 parquet outputs, positional event_id
fallback, synthetic event injection.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
ARCHIVE = OUT / "_archive"
MANIFEST_PATH = OUT / "decision_v1_ema_forward_alignment_manifest.json"

LCTD_REQUIRED = dt.date(2026, 5, 12)
MAX_OFFSET_WINDOW = 10  # Pilot 3's offset upper bound (§6.1)

HB_PATH = OUT / "horizontal_base_event_v1.parquet"
RESEARCH_BASELINE_HB_ROWS = 10470
LOCKED_HB_ARCHIVE_SHA = "2eb8a9a5d68e7e4831158f6a3e97c8b74521591af55e29a9682aa3ae7107b818"
LOCKED_HB_ARCHIVE = (
    ARCHIVE
    / "horizontal_base_event_v1__pre_refresh__asof_2026-04-29__sha256_2eb8a9a5.parquet"
)

# Source for one-shot research_frozen sidecar — archived ema_context_daily_metadata.json
# captured at sha 5aa1936b before the EMA cascade refresh wrote the current breakpoints.
RESEARCH_FROZEN_METADATA_ARCHIVE = (
    ARCHIVE
    / "ema_context_daily_metadata__pre_ema_cascade__asof_2026-05-05__sha256_5aa1936b.json"
)
RESEARCH_FROZEN_SIDECAR = OUT / "ema_context_breakpoints_research_frozen.json"

# Q8 ceilings (§3.2 + §6.4)
MAX_FORWARD_DELTA_PCT = 0.05
MAX_FORWARD_DELTA_ROWS = 1000
UNMAPPED_LOCKED_COUNT_SUBCEILING = max(10, math.ceil(0.001 * RESEARCH_BASELINE_HB_ROWS))
# = max(10, ceil(0.001*10470)) = 11

# ─── Step targets (24 forward outputs) ───────────────────────────────────────

STEP_TARGETS: dict[str, list[str]] = {
    "step1_pilot3_forward": [
        "ema_context_pilot3_panel_forward.parquet",
        "ema_context_pilot3_event_id_to_stable_key.parquet",
    ],
    "step2_pilot4_forward": [
        "ema_context_pilot4_earliness_per_event_forward.parquet",
        "ema_context_pilot4_distribution_pct_forward.csv",
        "ema_context_pilot4_distribution_atr_forward.csv",
        "ema_context_pilot4_gates_forward.csv",
        "ema_context_pilot4_summary_forward.md",
    ],
    "step3_pilot5_forward": [
        "ema_context_pilot5_panel_forward.parquet",
        "ema_context_pilot5_gates_primary_forward.csv",
        "ema_context_pilot5_gates_secondary_forward.csv",
        "ema_context_pilot5_stability_forward.csv",
        "ema_context_pilot5_supplementary_forward.csv",
        "ema_context_pilot5_summary_forward.md",
    ],
    "step4_pilot6_forward": [
        "ema_context_pilot6_panel_forward.parquet",
        "ema_context_pilot6_gates_main_forward.csv",
        "ema_context_pilot6_stability_forward.csv",
        "ema_context_pilot6_supplementary_forward.csv",
        "ema_context_pilot6_summary_forward.md",
    ],
    "step5_pilot7_forward": [
        "ema_context_pilot7_panel_forward.parquet",
        "ema_context_pilot7_gates_main_forward.csv",
        "ema_context_pilot7_state_stability_forward.csv",
        "ema_context_pilot7_supplementary_forward.csv",
        "ema_context_pilot7_drop_diagnostic_census_forward.csv",
        "ema_context_pilot7_summary_forward.md",
    ],
}

STEP_COMMANDS: dict[str, list[str]] = {
    "step1_pilot3_forward": [sys.executable, "tools/ema_context_pilot3_forward.py"],
    "step2_pilot4_forward": [sys.executable, "tools/ema_context_pilot4_forward.py"],
    "step3_pilot5_forward": [sys.executable, "tools/ema_context_pilot5_forward.py"],
    "step4_pilot6_forward": [sys.executable, "tools/ema_context_pilot6_forward.py"],
    "step5_pilot7_forward": [sys.executable, "tools/ema_context_pilot7_forward.py"],
}

STEP_JSON_TAG: dict[str, str] = {
    "step1_pilot3_forward": "PILOT3_FORWARD_SUMMARY_JSON",
    "step2_pilot4_forward": "PILOT4_FORWARD_SUMMARY_JSON",
    "step3_pilot5_forward": "PILOT5_FORWARD_SUMMARY_JSON",
    "step4_pilot6_forward": "PILOT6_FORWARD_SUMMARY_JSON",
    "step5_pilot7_forward": "PILOT7_FORWARD_SUMMARY_JSON",
}

# ─── Locked-research protected (byte-equal) ────────────────────────────────
# Spec §3.1 + §7 item 4: Pilot 3-7 research outputs MUST remain byte-equal pre/post.
LOCKED_RESEARCH_PROTECTED = [
    # Pilot 3
    "ema_context_pilot3_panel.parquet",
    "ema_context_pilot3_q1_combined_atr.csv",
    "ema_context_pilot3_q1_combined_pct.csv",
    "ema_context_pilot3_q2_per_state_atr.csv",
    "ema_context_pilot3_q2_per_state_pct.csv",
    "ema_context_pilot3_q3_slope_tier_atr.csv",
    "ema_context_pilot3_q3_slope_tier_pct.csv",
    "ema_context_pilot3_q3_width_tier_atr.csv",
    "ema_context_pilot3_q3_width_tier_pct.csv",
    "ema_context_pilot3_summary.md",
    # Pilot 4
    "ema_context_pilot4_earliness_per_event.parquet",
    "ema_context_pilot4_distribution_pct.csv",
    "ema_context_pilot4_distribution_atr.csv",
    "ema_context_pilot4_gates.csv",
    "ema_context_pilot4_summary.md",
    # Pilot 5
    "ema_context_pilot5_panel.parquet",
    "ema_context_pilot5_gates_primary.csv",
    "ema_context_pilot5_gates_secondary.csv",
    "ema_context_pilot5_stability.csv",
    "ema_context_pilot5_supplementary.csv",
    "ema_context_pilot5_summary.md",
    # Pilot 6
    "ema_context_pilot6_panel.parquet",
    "ema_context_pilot6_gates_main.csv",
    "ema_context_pilot6_stability.csv",
    "ema_context_pilot6_supplementary.csv",
    "ema_context_pilot6_summary.md",
    # Pilot 7
    "ema_context_pilot7_panel.parquet",
    "ema_context_pilot7_gates_main.csv",
    "ema_context_pilot7_state_stability.csv",
    "ema_context_pilot7_supplementary.csv",
    "ema_context_pilot7_drop_diagnostic_census.csv",
    "ema_context_pilot7_summary.md",
]

# ─── 24 protected non-target files (§7 item 5) ─────────────────────────────
PROTECTED_NON_TARGET = [
    # Decision engine v1 artefacts
    "decision_engine_v1_panel.parquet",
    "decision_engine_v1_events.parquet",
    "decision_engine_v1_tier2_label_distribution.csv",
    "decision_engine_v1_paper_link_integrity.csv",
    "decision_engine_v1_tier2_label_table_summary.md",
    "decision_engine_v1_tier3_review_report.md",
    "paper_execution_v0_trades.parquet",
    "paper_execution_v0_trigger_retest_trades.parquet",
    # Upstream backbone / scanner / regime / events
    "horizontal_base_event_v1.parquet",
    "nyxexp_dataset_v4.parquet",
    "mb_scanner_events_mb_5h.parquet",
    "mb_scanner_events_mb_1d.parquet",
    "mb_scanner_events_mb_1w.parquet",
    "mb_scanner_events_mb_1M.parquet",
    "mb_scanner_events_bb_5h.parquet",
    "mb_scanner_events_bb_1d.parquet",
    "mb_scanner_events_bb_1w.parquet",
    "mb_scanner_events_bb_1M.parquet",
    "scanner_v1_event_quality_events.csv",
    "regime_labels_daily_rdp_v1.csv",
    "xu100_extfeed_daily.parquet",
    "ohlcv_10y_fintables_master.parquet",
    "extfeed_intraday_1h_3y_master.parquet",
    "decision_v0_classification_panel.parquet",
]


# ─── helpers ───────────────────────────────────────────────────────────────


def _fail(reason: str) -> None:
    print(f"\n[ema_forward_alignment] FAIL: {reason}", flush=True)
    raise SystemExit(reason)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    m: dict = {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }
    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            df = pd.read_parquet(path)
            m["row_count"] = int(len(df))
            for c in ("date", "bar_date", "event_date", "asof_date", "ts_utc"):
                if c in df.columns:
                    s = pd.to_datetime(df[c], errors="coerce")
                    if s.notna().any():
                        m["max_date"] = str(s.max().date())
                        m["max_date_col"] = c
                    break
        except Exception as e:
            m["read_error"] = repr(e)
    elif suf == ".csv":
        try:
            df = pd.read_csv(path)
            m["row_count"] = int(len(df))
        except Exception as e:
            m["read_error"] = repr(e)
    return m


def _meta_set(paths: list[Path]) -> dict[str, dict]:
    return {str(p.relative_to(ROOT)): _file_meta(p) for p in paths}


def _archive(path: Path, asof: str) -> dict:
    """Archive a file to output/_archive/ at mode 0444 + JSON sidecar.
    Skip if file does not exist or archive already present.
    """
    if not path.exists():
        return {"archived": False, "reason": "source_missing"}
    sha8 = _sha256(path)[:8]
    arch = ARCHIVE / f"{path.stem}__pre_ema_forward__asof_{asof}__sha256_{sha8}{path.suffix}"
    sidecar = arch.with_suffix(arch.suffix + ".json")
    if arch.exists():
        return {
            "archived": False,
            "reason": "archive_exists",
            "archive_path": str(arch.relative_to(ROOT)),
        }
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, arch)
    os.chmod(arch, 0o444)
    sidecar.write_text(json.dumps({
        "source_path": str(path.relative_to(ROOT)),
        "archive_path": str(arch.relative_to(ROOT)),
        "archived_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "sha256": _sha256(arch),
        "size": arch.stat().st_size,
        "operation": "decision_v1_ema_forward_alignment",
    }, indent=2))
    os.chmod(sidecar, 0o444)
    return {
        "archived": True,
        "archive_path": str(arch.relative_to(ROOT)),
        "sidecar_path": str(sidecar.relative_to(ROOT)),
    }


def _emit_research_frozen_sidecar() -> dict:
    """One-shot emission of ema_context_breakpoints_research_frozen.json (mode 0444).
    Source: archived metadata at sha 5aa1936b (pre-cascade snapshot).
    Refuses to clobber if sidecar already exists; verifies byte-equality on rerun.
    """
    if not RESEARCH_FROZEN_METADATA_ARCHIVE.exists():
        return {
            "emitted": False,
            "reason": "metadata_archive_missing",
            "archive_path": str(RESEARCH_FROZEN_METADATA_ARCHIVE.relative_to(ROOT)),
        }
    archived_meta = json.loads(RESEARCH_FROZEN_METADATA_ARCHIVE.read_text())
    bp = archived_meta.get("tag_breakpoints", {})
    if not bp:
        return {
            "emitted": False,
            "reason": "archived_metadata_missing_tag_breakpoints",
            "archive_path": str(RESEARCH_FROZEN_METADATA_ARCHIVE.relative_to(ROOT)),
        }
    sidecar_payload = {
        "track": "research_frozen",
        "source_archive": str(RESEARCH_FROZEN_METADATA_ARCHIVE.relative_to(ROOT)),
        "source_archive_sha256_tag": "5aa1936b",
        "captured_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "operation": "decision_v1_ema_forward_alignment_one_shot_sidecar",
        "tag_breakpoints": bp,
        "phase0_run_date": archived_meta.get("phase0_run_date"),
        "acceptance": archived_meta.get("acceptance"),
        "notes": (
            "One-shot research_frozen breakpoints sidecar per spec §4 Option C. "
            "Mode 0444 — refuses overwrite. Used by forward consumers to assign "
            "tag_*_research_frozen columns when the dual-track ema_context_daily "
            "tagging is later implemented."
        ),
    }
    new_text = json.dumps(sidecar_payload, indent=2, default=str)
    if RESEARCH_FROZEN_SIDECAR.exists():
        existing_text = RESEARCH_FROZEN_SIDECAR.read_text()
        try:
            existing = json.loads(existing_text)
        except Exception:
            return {
                "emitted": False,
                "reason": "existing_sidecar_unparseable",
                "sidecar_path": str(RESEARCH_FROZEN_SIDECAR.relative_to(ROOT)),
            }
        # Compare on the immutable parts (tag_breakpoints + source provenance).
        if (
            existing.get("tag_breakpoints") == bp
            and existing.get("source_archive_sha256_tag") == "5aa1936b"
        ):
            return {
                "emitted": False,
                "reason": "sidecar_exists_content_matches_locked_baseline",
                "sidecar_path": str(RESEARCH_FROZEN_SIDECAR.relative_to(ROOT)),
                "sha256": _sha256(RESEARCH_FROZEN_SIDECAR),
            }
        return {
            "emitted": False,
            "reason": "sidecar_exists_content_mismatch_HALT",
            "sidecar_path": str(RESEARCH_FROZEN_SIDECAR.relative_to(ROOT)),
        }
    tmp = RESEARCH_FROZEN_SIDECAR.with_suffix(RESEARCH_FROZEN_SIDECAR.suffix + ".tmp")
    tmp.write_text(new_text)
    tmp.replace(RESEARCH_FROZEN_SIDECAR)
    os.chmod(RESEARCH_FROZEN_SIDECAR, 0o444)
    return {
        "emitted": True,
        "sidecar_path": str(RESEARCH_FROZEN_SIDECAR.relative_to(ROOT)),
        "sha256": _sha256(RESEARCH_FROZEN_SIDECAR),
        "size": RESEARCH_FROZEN_SIDECAR.stat().st_size,
        "research_frozen_breakpoints": bp,
    }


def _extract_pilot_summary_json(stdout: str, tag: str) -> dict | None:
    begin = f"{tag}_BEGIN"
    end = f"{tag}_END"
    if begin not in stdout or end not in stdout:
        return None
    try:
        body = stdout.split(begin, 1)[1].split(end, 1)[0].strip()
        return json.loads(body)
    except Exception:
        return None


def _emit_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


# ─── orchestration ─────────────────────────────────────────────────────────


def main() -> None:
    run_started = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    asof_tag = LCTD_REQUIRED.isoformat()

    manifest: dict = {
        "spec_path": "memory/ema_context_forward_alignment_spec.md",
        "spec_version": "LOCK candidate v2 → LOCKED 2026-05-05",
        "operation": "EMA forward alignment refresh (key-based remap, dual-track breakpoints)",
        "run_started_utc": run_started,
        "operational_target_date": asof_tag,
        "max_offset_window": MAX_OFFSET_WINDOW,
        "verdict": None,
        "fail_reasons": [],
        "research_baseline": {
            "hb_rows": RESEARCH_BASELINE_HB_ROWS,
            "archive_sha256_locked_hb": LOCKED_HB_ARCHIVE_SHA,
            "locked_hb_archive_path": str(LOCKED_HB_ARCHIVE.relative_to(ROOT)),
        },
        "subceilings": {
            "max_forward_delta_pct": MAX_FORWARD_DELTA_PCT,
            "max_forward_delta_rows": MAX_FORWARD_DELTA_ROWS,
            "unmapped_locked_count_subceiling": UNMAPPED_LOCKED_COUNT_SUBCEILING,
        },
        "phases": {},
    }

    # ─── Pre-flight: verify locked HB archive sha + current HB existence ────
    print("[ema_forward_alignment] Pre-flight integrity checks", flush=True)
    if not LOCKED_HB_ARCHIVE.exists():
        _fail(f"locked_hb_archive_missing: {LOCKED_HB_ARCHIVE.relative_to(ROOT)}")
    arch_sha = _sha256(LOCKED_HB_ARCHIVE)
    if arch_sha != LOCKED_HB_ARCHIVE_SHA:
        _fail(
            f"locked_hb_archive_sha_mismatch: got {arch_sha[:16]}… "
            f"expected {LOCKED_HB_ARCHIVE_SHA[:16]}…"
        )
    if not HB_PATH.exists():
        _fail(f"current_hb_missing: {HB_PATH.relative_to(ROOT)}")
    hb_pre = pd.read_parquet(HB_PATH)
    current_hb_rows = int(len(hb_pre))
    delta_rows = current_hb_rows - RESEARCH_BASELINE_HB_ROWS
    delta_pct = delta_rows / RESEARCH_BASELINE_HB_ROWS
    if delta_pct > MAX_FORWARD_DELTA_PCT or delta_rows > MAX_FORWARD_DELTA_ROWS:
        _fail(
            f"forward_delta_exceeds_q8_ceiling: delta_rows={delta_rows} "
            f"delta_pct={delta_pct:.4f} (max_pct={MAX_FORWARD_DELTA_PCT}, "
            f"max_rows={MAX_FORWARD_DELTA_ROWS})"
        )
    print(
        f"  current_hb_rows={current_hb_rows} delta_rows={delta_rows} "
        f"delta_pct={delta_pct:.4f} OK",
        flush=True,
    )

    # ─── Phase A: Pre-state ────────────────────────────────────────────────
    print("[ema_forward_alignment] Phase A: pre-state capture", flush=True)
    forward_target_paths = [
        OUT / fn for files in STEP_TARGETS.values() for fn in files
    ]
    locked_research_paths = [OUT / fn for fn in LOCKED_RESEARCH_PROTECTED]
    protected_non_target_paths = [OUT / fn for fn in PROTECTED_NON_TARGET]

    pre_forward = _meta_set(forward_target_paths)
    pre_locked_research = _meta_set(locked_research_paths)
    pre_protected = _meta_set(protected_non_target_paths)
    pre_sidecar = _file_meta(RESEARCH_FROZEN_SIDECAR)

    manifest["phases"]["A_pre_state"] = {
        "forward_targets_pre": pre_forward,
        "locked_research_pre": pre_locked_research,
        "protected_non_target_pre": pre_protected,
        "research_frozen_sidecar_pre": pre_sidecar,
        "current_hb_rows": current_hb_rows,
        "current_hb_delta_rows": delta_rows,
        "current_hb_delta_pct": delta_pct,
    }

    # ─── Phase B: Sidecar one-shot + archive pre-existing forward outputs ──
    print("[ema_forward_alignment] Phase B: research_frozen sidecar + archive", flush=True)
    sidecar_result = _emit_research_frozen_sidecar()
    print(f"  sidecar: {sidecar_result.get('reason') or 'emitted'}", flush=True)
    if sidecar_result.get("reason") in (
        "metadata_archive_missing",
        "archived_metadata_missing_tag_breakpoints",
        "existing_sidecar_unparseable",
        "sidecar_exists_content_mismatch_HALT",
    ):
        manifest["phases"]["B_archive"] = {"sidecar": sidecar_result}
        manifest["verdict"] = "FAIL"
        manifest["fail_reasons"].append(f"sidecar_emission_blocked: {sidecar_result.get('reason')}")
        _emit_manifest(manifest)
        _fail(f"sidecar_emission_blocked: {sidecar_result.get('reason')}")

    forward_archives: dict[str, dict] = {}
    for step, files in STEP_TARGETS.items():
        for fn in files:
            forward_archives[fn] = _archive(OUT / fn, asof_tag)
    # Locked-research files are NOT archived (NOT being overwritten by spec §7 item 2).
    manifest["phases"]["B_archive"] = {
        "sidecar": sidecar_result,
        "forward_outputs_archived": forward_archives,
    }

    # ─── Phase C: Run cascade ──────────────────────────────────────────────
    print("[ema_forward_alignment] Phase C: 5-step forward cascade", flush=True)
    step_results: list[dict] = []
    pilot_summaries: dict[str, dict | None] = {}
    for step in (
        "step1_pilot3_forward",
        "step2_pilot4_forward",
        "step3_pilot5_forward",
        "step4_pilot6_forward",
        "step5_pilot7_forward",
    ):
        cmd = STEP_COMMANDS[step]
        print(f"  → running {step}: {' '.join(cmd)}", flush=True)
        t0 = dt.datetime.now(tz=dt.timezone.utc)
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        t1 = dt.datetime.now(tz=dt.timezone.utc)
        rec = {
            "step": step,
            "cmd": cmd,
            "returncode": proc.returncode,
            "started_utc": t0.isoformat(),
            "finished_utc": t1.isoformat(),
            "elapsed_s": round((t1 - t0).total_seconds(), 2),
            "stdout_tail": proc.stdout[-3000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-3000:] if proc.stderr else "",
        }
        step_results.append(rec)
        pilot_summary = _extract_pilot_summary_json(proc.stdout or "", STEP_JSON_TAG[step])
        pilot_summaries[step] = pilot_summary
        if proc.returncode != 0:
            manifest["phases"]["C_cascade"] = {
                "steps": step_results,
                "pilot_summaries": pilot_summaries,
            }
            manifest["verdict"] = "FAIL"
            manifest["fail_reasons"].append(
                f"{step}_subprocess_returncode_{proc.returncode}"
            )
            _emit_manifest(manifest)
            _fail(f"{step}_returncode={proc.returncode}; halting cascade")
    manifest["phases"]["C_cascade"] = {
        "steps": step_results,
        "pilot_summaries": pilot_summaries,
    }

    # ─── Phase D: Post-state + validation ──────────────────────────────────
    print("[ema_forward_alignment] Phase D: post-state + validation", flush=True)
    post_forward = _meta_set(forward_target_paths)
    post_locked_research = _meta_set(locked_research_paths)
    post_protected = _meta_set(protected_non_target_paths)
    post_sidecar = _file_meta(RESEARCH_FROZEN_SIDECAR)

    fail_reasons: list[str] = []

    # D.1 — Pilot 3 key_remap counts + sub-ceiling + Q8 ceiling (re-verify)
    p3 = pilot_summaries.get("step1_pilot3_forward") or {}
    matched_old = p3.get("matched_old_event_count")
    unmatched_old = p3.get("unmatched_old_event_count")
    new_current = p3.get("new_current_event_count")
    dup_locked = p3.get("duplicate_stable_event_key_count_locked")
    dup_current = p3.get("duplicate_stable_event_key_count_current")
    p3_unmapped_records = p3.get("unmapped_locked_events", [])

    if dup_locked not in (0, None) and dup_locked != 0:
        fail_reasons.append(
            f"stable_key_duplicate_detected_in_locked_hb: {dup_locked}"
        )
    if dup_current not in (0, None) and dup_current != 0:
        fail_reasons.append(
            f"stable_key_duplicate_detected_in_current_hb: {dup_current}"
        )
    if unmatched_old is not None and unmatched_old > UNMAPPED_LOCKED_COUNT_SUBCEILING:
        fail_reasons.append(
            f"emission_diff_exceeds_subceiling: unmatched_old={unmatched_old} "
            f"> sub-ceiling={UNMAPPED_LOCKED_COUNT_SUBCEILING}"
        )
    p3_delta_rows = p3.get("delta_rows")
    p3_delta_pct = p3.get("delta_pct")
    if (
        p3_delta_rows is not None
        and (p3_delta_rows > MAX_FORWARD_DELTA_ROWS or (p3_delta_pct or 0) > MAX_FORWARD_DELTA_PCT)
    ):
        fail_reasons.append(
            f"forward_delta_exceeds_q8_ceiling: delta_rows={p3_delta_rows} "
            f"delta_pct={p3_delta_pct}"
        )

    # D.2 — Per-pilot row reconciliation
    p3_unique_eid = p3.get("forward_panel_unique_event_id")
    expected_unique_eid = (matched_old or 0) + (new_current or 0)
    pilot_row_check = {
        "pilot3_forward_panel_unique_event_id": p3_unique_eid,
        "pilot3_forward_panel_expected_unique_event_id": expected_unique_eid,
        "pilot3_forward_panel_matches_expected": (
            p3_unique_eid == expected_unique_eid if (p3_unique_eid is not None) else None
        ),
    }
    if pilot_row_check["pilot3_forward_panel_matches_expected"] is False:
        fail_reasons.append(
            f"pilot3_forward_panel_unique_event_id_mismatch: "
            f"got={p3_unique_eid} expected={expected_unique_eid}"
        )

    # Pilot 4-7 row counts (panel_rows from JSON)
    p4 = pilot_summaries.get("step2_pilot4_forward") or {}
    p5 = pilot_summaries.get("step3_pilot5_forward") or {}
    p6 = pilot_summaries.get("step4_pilot6_forward") or {}
    p7 = pilot_summaries.get("step5_pilot7_forward") or {}

    pilot_row_check.update({
        "pilot4_forward_earliness_rows": p4.get("earliness_rows"),
        "pilot5_forward_panel_rows": p5.get("panel_rows"),
        "pilot6_forward_panel_rows": p6.get("panel_rows"),
        "pilot7_forward_panel_rows": p7.get("panel_rows"),
    })

    # D.3 — Locked-research byte-equal (31 files)
    locked_research_drift = []
    for k, pre in pre_locked_research.items():
        post = post_locked_research.get(k, {})
        if pre.get("exists") != post.get("exists"):
            locked_research_drift.append({
                "file": k, "reason": "existence_changed",
                "pre": pre, "post": post,
            })
            continue
        if not pre.get("exists"):
            continue
        if pre.get("sha256") != post.get("sha256"):
            locked_research_drift.append({
                "file": k, "reason": "sha256_drift",
                "pre_sha256": pre.get("sha256"),
                "post_sha256": post.get("sha256"),
            })
    if locked_research_drift:
        fail_reasons.append(
            f"locked_research_drift_detected: count={len(locked_research_drift)}"
        )

    # D.4 — Protected non-target byte-equal (24 files)
    protected_drift = []
    for k, pre in pre_protected.items():
        post = post_protected.get(k, {})
        if pre.get("exists") != post.get("exists"):
            protected_drift.append({
                "file": k, "reason": "existence_changed",
                "pre": pre, "post": post,
            })
            continue
        if not pre.get("exists"):
            continue
        if pre.get("sha256") != post.get("sha256"):
            protected_drift.append({
                "file": k, "reason": "sha256_drift",
                "pre_sha256": pre.get("sha256"),
                "post_sha256": post.get("sha256"),
            })
    if protected_drift:
        fail_reasons.append(f"protected_non_target_drift: count={len(protected_drift)}")

    # D.5 — Forward panel max_date >= LCTD - max_offset_window
    floor_date = (LCTD_REQUIRED - dt.timedelta(days=MAX_OFFSET_WINDOW)).isoformat()
    pilot3_panel_post = post_forward.get("output/ema_context_pilot3_panel_forward.parquet", {})
    p3_panel_max = pilot3_panel_post.get("max_date")
    p3_panel_max_ok = (p3_panel_max is not None) and (p3_panel_max >= floor_date)
    if not p3_panel_max_ok:
        fail_reasons.append(
            f"pilot3_forward_panel_max_date_below_floor: max={p3_panel_max} "
            f"required>={floor_date}"
        )

    # D.6 — Sidecar verification (mode + presence)
    sidecar_post_present = post_sidecar.get("exists", False)
    if not sidecar_post_present:
        fail_reasons.append("research_frozen_sidecar_missing_post_run")
    sidecar_mode = None
    if sidecar_post_present:
        try:
            sidecar_mode = oct(RESEARCH_FROZEN_SIDECAR.stat().st_mode & 0o777)
        except Exception:
            pass

    manifest["phases"]["D_validate"] = {
        "key_remap": {
            "stable_event_key": [
                "ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date",
            ],
            "matched_old_event_count": matched_old,
            "unmatched_old_event_count": unmatched_old,
            "new_current_event_count": new_current,
            "duplicate_stable_event_key_count_locked": dup_locked,
            "duplicate_stable_event_key_count_current": dup_current,
            "unmapped_locked_events": p3_unmapped_records,
        },
        "subceiling_check": {
            "unmapped_locked_count_subceiling": UNMAPPED_LOCKED_COUNT_SUBCEILING,
            "unmatched_old_event_count": unmatched_old,
            "subceiling_ok": (
                unmatched_old is not None and unmatched_old <= UNMAPPED_LOCKED_COUNT_SUBCEILING
            ),
        },
        "q8_ceiling_check": {
            "max_forward_delta_pct": MAX_FORWARD_DELTA_PCT,
            "max_forward_delta_rows": MAX_FORWARD_DELTA_ROWS,
            "current_hb_delta_rows": delta_rows,
            "current_hb_delta_pct": delta_pct,
            "q8_ok": delta_rows <= MAX_FORWARD_DELTA_ROWS and delta_pct <= MAX_FORWARD_DELTA_PCT,
        },
        "pilot_row_reconciliation": pilot_row_check,
        "pilot3_forward_panel_max_date": p3_panel_max,
        "pilot3_forward_panel_max_date_floor": floor_date,
        "pilot3_forward_panel_max_date_ok": p3_panel_max_ok,
        "research_frozen_sidecar_post": post_sidecar,
        "research_frozen_sidecar_mode": sidecar_mode,
        "forward_targets_post": post_forward,
        "locked_research_post": post_locked_research,
        "locked_research_drift": locked_research_drift,
        "locked_research_byte_equal": len(locked_research_drift) == 0,
        "protected_non_target_post": post_protected,
        "protected_non_target_drift": protected_drift,
        "protected_non_target_byte_equal": len(protected_drift) == 0,
    }

    # ─── Phase E: Manifest + verdict ──────────────────────────────────────
    if fail_reasons:
        manifest["verdict"] = "FAIL"
        manifest["fail_reasons"] = fail_reasons
    else:
        manifest["verdict"] = "PASS"
        manifest["fail_reasons"] = []

    manifest["run_finished_utc"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    manifest["phases"]["E_manifest"] = {
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
    }
    _emit_manifest(manifest)

    print(f"\n[ema_forward_alignment] verdict: {manifest['verdict']}", flush=True)
    if fail_reasons:
        for r in fail_reasons:
            print(f"  reason: {r}", flush=True)
        sys.exit(1)
    print(
        f"[ema_forward_alignment] manifest: {MANIFEST_PATH.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
