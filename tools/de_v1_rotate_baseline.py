#!/usr/bin/env python3
"""Rotate the DE v1 frozen baseline to the CURRENT HB (rotate-on-HALT).

Regenerates the pilot3/4/5/6/7 base panels against whatever
``output/horizontal_base_event_v1.parquet`` + ``ema_context_daily.parquet`` are
currently present (running each base pilot with ``DE_V1_ROTATE=1`` so its
locked-baseline ``EXPECTED_*`` row/sha asserts are bypassed — see
``tools/_de_rotate_mode.py``), freezes the new panels + HB as ``__locked__`` /
``__pre_refresh__`` archives, and rewrites the relevant ``tools/pin_baselines.toml``
anchors (HB + pilot3/5/6/7 panels + pilot7 gates/census) in lockstep with their
new sha256 and row counts.

Because the forward scripts (ema_context_pilot3_forward.py,
decision_engine_v1_ema_forward_alignment.py) read every baseline count/sha/path
from the pin, they pick up the rotated baseline with NO source edit — which is
what lets CI rotate-on-HALT without committing Python changes.

NOT rotated: research_frozen_metadata_archive (breakpoints permanently frozen at
2026-05-05 for cross-chain label consistency) and the de_v1/de_v0 downstream
snapshot anchors (those are produced by Stage 7/8 of the chain).

Usage:
    python tools/de_v1_rotate_baseline.py            # asof = HB max bar_date
    python tools/de_v1_rotate_baseline.py --asof 2026-06-09
    python tools/de_v1_rotate_baseline.py --dry-run  # report only, no writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
ARCHIVE = OUT / "_archive"
PIN_PATH = ROOT / "tools" / "pin_baselines.toml"

HB_PARQUET = OUT / "horizontal_base_event_v1.parquet"
HB_MANIFEST = OUT / "horizontal_base_event_v1_manifest.json"

# Base pilots, in dependency order (3 → 4 → 5 → 6 → 7).
PILOTS = [
    "tools/ema_context_pilot3.py",
    "tools/ema_context_pilot4.py",
    "tools/ema_context_pilot5.py",
    "tools/ema_context_pilot6.py",
    "tools/ema_context_pilot7.py",
]

# (operational output, archive stem, pin section) for each frozen artefact.
PANEL_ARTEFACTS = [
    (OUT / "ema_context_pilot3_panel.parquet", "ema_context_pilot3_panel", "locked_pilot3_panel"),
    (OUT / "ema_context_pilot5_panel.parquet", "ema_context_pilot5_panel", "locked_pilot5_panel"),
    (OUT / "ema_context_pilot6_panel.parquet", "ema_context_pilot6_panel", "locked_pilot6_panel"),
    (OUT / "ema_context_pilot7_panel.parquet", "ema_context_pilot7_panel", "locked_pilot7_panel"),
    (OUT / "ema_context_pilot7_gates_main.csv", "ema_context_pilot7_gates_main", "locked_pilot7_gates_main"),
    (
        OUT / "ema_context_pilot7_drop_diagnostic_census.csv",
        "ema_context_pilot7_drop_diagnostic_census",
        "locked_pilot7_drop_diagnostic_census",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_pilots() -> None:
    env = dict(os.environ)
    env["DE_V1_ROTATE"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for rel in PILOTS:
        print(f"[rotate] regen {rel} (DE_V1_ROTATE=1) …", flush=True)
        r = subprocess.run([sys.executable, rel], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
            raise SystemExit(f"[rotate] FAIL: {rel} returncode={r.returncode}")
        tail = r.stdout.strip().splitlines()[-1:] or [""]
        print(f"  ok — {tail[0]}", flush=True)


def _update_pin_section(text: str, section: str, updates: dict) -> str:
    """Replace key = value lines within [section]; append missing keys."""
    pat = re.compile(rf'(\[{re.escape(section)}\]\n)(.*?)(?=\n\[|\Z)', re.DOTALL)
    m = pat.search(text)
    if not m:
        raise SystemExit(f"[rotate] pin section [{section}] not found")
    body = m.group(2)
    for key, val in updates.items():
        rep = f'{key} = "{val}"' if isinstance(val, str) else f'{key} = {val}'
        kpat = re.compile(rf'^{re.escape(key)} = .*$', re.MULTILINE)
        if kpat.search(body):
            body = kpat.sub(rep, body, count=1)
        else:
            body = body.rstrip("\n") + f"\n{rep}\n"
    return text[: m.start(2)] + body + text[m.end(2):]


def main() -> int:
    ap = argparse.ArgumentParser(description="Rotate DE v1 frozen baseline to current HB.")
    ap.add_argument("--asof", default=None, help="ISO date for HB archive name; default = HB max bar_date.")
    ap.add_argument("--dry-run", action="store_true", help="Report new values; do not freeze/write.")
    args = ap.parse_args()

    if not HB_PARQUET.exists() or not HB_MANIFEST.exists():
        raise SystemExit(f"[rotate] current HB missing: {HB_PARQUET}")

    # 1) Regenerate base panels against current HB (rotate-mode bypass). Runs in
    #    dry-run too — it overwrites the operational panels (git-restorable) so the
    #    bypass path is exercised; only freeze + pin write are skipped on dry-run.
    _run_pilots()

    # 2) Read new baseline facts.
    manifest = json.loads(HB_MANIFEST.read_text())
    hb_rows = int(manifest["rows"])
    hb_sha = _sha256(HB_PARQUET)
    bar_max = pd.to_datetime(pd.read_parquet(HB_PARQUET, columns=["bar_date"])["bar_date"]).max()
    asof = args.asof or bar_max.date().isoformat()

    p3 = pd.read_parquet(OUT / "ema_context_pilot3_panel.parquet", columns=["event_id"])
    p3_rows, p3_eid = int(len(p3)), int(p3["event_id"].nunique())

    print(f"[rotate] new baseline: HB rows={hb_rows} sha={hb_sha[:8]} asof={asof} "
          f"pilot3 panel_rows={p3_rows} unique_eid={p3_eid}", flush=True)

    if args.dry_run:
        print("[rotate] dry-run: no archives frozen, pin untouched.")
        return 0

    # 3) Freeze HB archive + read pin text. Track exactly which files we write so
    #    CI can `git add` precisely (NOT the forward cascade's __pre_ema_forward__ snapshots).
    changed: list[str] = ["tools/pin_baselines.toml"]
    ARCHIVE.mkdir(exist_ok=True)
    hb_arch_rel = f"output/_archive/horizontal_base_event_v1__pre_refresh__asof_{asof}__sha256_{hb_sha[:8]}.parquet"
    shutil.copy2(HB_PARQUET, ROOT / hb_arch_rel)
    changed.append(hb_arch_rel)
    text = PIN_PATH.read_text()

    text = _update_pin_section(text, "locked_hb_archive", {
        "path": hb_arch_rel, "sha256": hb_sha, "asof_date": asof, "rows": hb_rows,
    })

    # 4) Freeze each panel/sidecar + update its pin section.
    for src, stem, section in PANEL_ARTEFACTS:
        sha = _sha256(src)
        ext = src.suffix.lstrip(".")
        arch_rel = f"output/_archive/{stem}__locked__sha256_{sha[:8]}.{ext}"
        shutil.copy2(src, ROOT / arch_rel)
        changed.append(arch_rel)
        updates = {"path": arch_rel, "sha256": sha}
        if section == "locked_pilot3_panel":
            updates["panel_rows"] = p3_rows
            updates["unique_eid"] = p3_eid
        text = _update_pin_section(text, section, updates)
        print(f"  froze {section}: sha={sha[:8]}", flush=True)

    PIN_PATH.write_text(text)
    # Emit the exact changed-file list for CI to stage (one path per line).
    (OUT / "_de_rotate_changed.txt").write_text("\n".join(changed) + "\n")
    print(f"[rotate] pin_baselines.toml updated (HB + 4 panels + 2 pilot7 sidecars).")
    print(f"[rotate] changed files → output/_de_rotate_changed.txt ({len(changed)} paths)")
    print(f"[rotate] DONE — baseline rotated to asof={asof} (HB {hb_rows} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
