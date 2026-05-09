"""Master data pull orchestrator (PR-1).

Single entrypoint that wraps the canonical refresh tools so downstream
consumers (mb_scanner, HB, nyxexpansion, Decision Engine) can later restore
one shared artifact instead of each running its own pull. PR-1 only produces
the artifact + manifest; consumer migration is PR-2.

Modes:
    intraday_1700  refresh extfeed_intraday_1h_3y_master.parquet
    close          refresh extfeed master + ohlcv_10y_fintables_master.parquet
                   + xu100_extfeed_daily.parquet

Outputs:
    output/extfeed_intraday_1h_3y_master.parquet (always rewritten)
    output/ohlcv_10y_fintables_master.parquet     (mode=close only)
    output/xu100_extfeed_daily.parquet            (mode=close only)
    output/master_data_manifest.json              (always written, even on FAIL)

Exit code: 0 PASS, 1 FAIL (any required step or freshness check failed).

Out of scope (PR-2): consumer-workflow migration, Decision Engine cron,
removal of existing per-workflow delta_pull steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTFEED_MASTER = REPO_ROOT / "output" / "extfeed_intraday_1h_3y_master.parquet"
FINTABLES_MASTER = REPO_ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
XU100_DAILY = REPO_ROOT / "output" / "xu100_extfeed_daily.parquet"
MANIFEST_PATH = REPO_ROOT / "output" / "master_data_manifest.json"

EXTFEED_REQUIRED_SECRETS = ("INTRADAY_SID", "INTRADAY_SIGN", "INTRADAY_HOST", "INTRADAY_WS_URL")

ROW_COLLAPSE_FLOOR = 0.99
SUBPROCESS_TIMEOUT_S = 1800

ISTANBUL = timezone(timedelta(hours=3))


@dataclass
class ParquetMeta:
    path: str
    exists: bool
    sha256: str | None = None
    row_count: int | None = None
    max_ts_utc: str | None = None
    max_date: str | None = None
    bytes: int | None = None


@dataclass
class StepResult:
    name: str
    ok: bool
    duration_s: float
    detail: str = ""


@dataclass
class OrchestratorResult:
    mode: str
    run_timestamp_utc: str
    operational_target_date: str
    extfeed_pre: ParquetMeta
    extfeed_post: ParquetMeta
    fintables_pre: ParquetMeta
    fintables_post: ParquetMeta
    xu100_pre: ParquetMeta
    xu100_post: ParquetMeta
    freshness_status: str
    verdict: str
    failures: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)


def compute_operational_target_date(now_utc: datetime | None = None) -> date:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    tr = now_utc.astimezone(ISTANBUL).date()
    while tr.weekday() >= 5:
        tr -= timedelta(days=1)
    return tr


def check_required_secrets(mode: str) -> list[str]:
    missing = [name for name in EXTFEED_REQUIRED_SECRETS if not os.environ.get(name)]
    return missing


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_extfeed(path: Path) -> ParquetMeta:
    if not path.exists():
        return ParquetMeta(path=str(path), exists=False)
    df = pd.read_parquet(path, columns=["ts_utc"])
    max_ts = pd.to_datetime(df["ts_utc"]).max()
    if pd.isna(max_ts):
        max_ts_str = None
    else:
        if max_ts.tzinfo is None:
            max_ts = max_ts.tz_localize("UTC")
        max_ts_str = max_ts.tz_convert("UTC").isoformat()
    return ParquetMeta(
        path=str(path),
        exists=True,
        sha256=_sha256(path),
        row_count=int(len(df)),
        max_ts_utc=max_ts_str,
        bytes=path.stat().st_size,
    )


def inspect_fintables(path: Path) -> ParquetMeta:
    if not path.exists():
        return ParquetMeta(path=str(path), exists=False)
    df = pd.read_parquet(path)
    # Canonical fintables master stores Date as parquet index, not column.
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        max_d = df.index.max()
    elif "Date" in df.columns:
        max_d = pd.to_datetime(df["Date"]).max()
    elif "date" in df.columns:
        max_d = pd.to_datetime(df["date"]).max()
    else:
        max_d = None
    max_d_str = None if max_d is None or pd.isna(max_d) else pd.Timestamp(max_d).date().isoformat()
    return ParquetMeta(
        path=str(path),
        exists=True,
        sha256=_sha256(path),
        row_count=int(len(df)),
        max_date=max_d_str,
        bytes=path.stat().st_size,
    )


def inspect_xu100(path: Path) -> ParquetMeta:
    if not path.exists():
        return ParquetMeta(path=str(path), exists=False)
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        max_d = df.index.max()
    elif "Date" in df.columns:
        max_d = pd.to_datetime(df["Date"]).max()
    elif "date" in df.columns:
        max_d = pd.to_datetime(df["date"]).max()
    else:
        max_d = None
    max_d_str = None if max_d is None or pd.isna(max_d) else pd.Timestamp(max_d).date().isoformat()
    return ParquetMeta(
        path=str(path),
        exists=True,
        sha256=_sha256(path),
        row_count=int(len(df)),
        max_date=max_d_str,
        bytes=path.stat().st_size,
    )


def run_subprocess(cmd: list[str], step_name: str) -> StepResult:
    t0 = time.time()
    print(f"[{step_name}] $ {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            timeout=SUBPROCESS_TIMEOUT_S,
            check=False,
        )
        dur = time.time() - t0
        ok = proc.returncode == 0
        return StepResult(
            name=step_name,
            ok=ok,
            duration_s=dur,
            detail=f"exit={proc.returncode}",
        )
    except subprocess.TimeoutExpired:
        dur = time.time() - t0
        return StepResult(name=step_name, ok=False, duration_s=dur, detail="timeout")
    except Exception as exc:
        dur = time.time() - t0
        return StepResult(name=step_name, ok=False, duration_s=dur, detail=f"exception:{exc}")


def freshness_check(
    mode: str,
    target: date,
    extfeed: ParquetMeta,
    fintables: ParquetMeta,
    xu100: ParquetMeta,
) -> tuple[str, list[str]]:
    failures: list[str] = []

    if not extfeed.exists or extfeed.max_ts_utc is None:
        failures.append("extfeed master missing or unreadable")
    else:
        max_ts = pd.Timestamp(extfeed.max_ts_utc)
        if max_ts.tzinfo is None:
            max_ts = max_ts.tz_localize("UTC")
        max_ts_tr_date = max_ts.tz_convert("Europe/Istanbul").date()
        if max_ts_tr_date < target:
            failures.append(
                f"extfeed stale: max_ts_tr_date={max_ts_tr_date} < target={target}"
            )
        if mode == "close":
            close_threshold = pd.Timestamp(f"{target.isoformat()}T14:00:00Z")
            if max_ts < close_threshold:
                failures.append(
                    f"extfeed pre-close: max_ts_utc={max_ts.isoformat()} "
                    f"< {close_threshold.isoformat()} (BIST 17:00 TR)"
                )

    if mode == "close":
        if not fintables.exists or fintables.max_date is None:
            failures.append("fintables master missing or unreadable")
        else:
            fmax = date.fromisoformat(fintables.max_date)
            if fmax < target:
                failures.append(
                    f"fintables stale: max_date={fmax} < target={target}"
                )

        if not xu100.exists or xu100.max_date is None:
            failures.append("xu100 master missing or unreadable")
        else:
            xmax = date.fromisoformat(xu100.max_date)
            if xmax != target:
                failures.append(
                    f"xu100 stale: max_date={xmax} != target={target}"
                )

    return ("PASS" if not failures else "FAIL"), failures


def row_collapse_check(pre: ParquetMeta, post: ParquetMeta, label: str) -> str | None:
    if not pre.exists or pre.row_count is None:
        return None
    if not post.exists or post.row_count is None:
        return f"{label}: post parquet missing after refresh"
    if post.row_count < pre.row_count * ROW_COLLAPSE_FLOOR:
        return (
            f"{label}: row collapse pre={pre.row_count:,} → "
            f"post={post.row_count:,} (floor={ROW_COLLAPSE_FLOOR:.0%})"
        )
    return None


def _meta_to_dict(meta: ParquetMeta) -> dict[str, Any]:
    return {
        "path": meta.path,
        "exists": meta.exists,
        "sha256": meta.sha256,
        "row_count": meta.row_count,
        "max_ts_utc": meta.max_ts_utc,
        "max_date": meta.max_date,
        "bytes": meta.bytes,
    }


def write_manifest(result: OrchestratorResult) -> None:
    payload = {
        "mode": result.mode,
        "run_timestamp_utc": result.run_timestamp_utc,
        "operational_target_date": result.operational_target_date,
        "extfeed": {
            "pre": _meta_to_dict(result.extfeed_pre),
            "post": _meta_to_dict(result.extfeed_post),
            "max_ts_utc": result.extfeed_post.max_ts_utc,
            "sha256": result.extfeed_post.sha256,
            "row_count": result.extfeed_post.row_count,
        },
        "fintables": {
            "pre": _meta_to_dict(result.fintables_pre),
            "post": _meta_to_dict(result.fintables_post),
            "max_date": result.fintables_post.max_date,
            "sha256": result.fintables_post.sha256,
            "row_count": result.fintables_post.row_count,
        },
        "xu100": {
            "pre": _meta_to_dict(result.xu100_pre),
            "post": _meta_to_dict(result.xu100_post),
            "max_date": result.xu100_post.max_date,
            "sha256": result.xu100_post.sha256,
            "row_count": result.xu100_post.row_count,
            "bytes": result.xu100_post.bytes,
        },
        "freshness_status": result.freshness_status,
        "verdict": result.verdict,
        "failures": result.failures,
        "steps": [
            {"name": s.name, "ok": s.ok, "duration_s": round(s.duration_s, 2), "detail": s.detail}
            for s in result.steps
        ],
        "schema_version": 1,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[manifest] wrote {MANIFEST_PATH}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Master data pull orchestrator (PR-1).")
    ap.add_argument("--mode", required=True, choices=["intraday_1700", "close"])
    ap.add_argument("--target-date", default=None, help="YYYY-MM-DD; defaults to today (Europe/Istanbul, weekend→Friday)")
    args = ap.parse_args()

    run_ts = datetime.now(timezone.utc).isoformat()
    if args.target_date:
        target = date.fromisoformat(args.target_date)
    else:
        target = compute_operational_target_date()
    print(f"[run] mode={args.mode} target_date={target} run_ts={run_ts}", flush=True)

    failures: list[str] = []
    steps: list[StepResult] = []

    missing_secrets = check_required_secrets(args.mode)
    if missing_secrets:
        failures.append(f"missing secrets: {missing_secrets}")
        steps.append(StepResult(name="check_secrets", ok=False, duration_s=0.0,
                                detail=f"missing={missing_secrets}"))
    else:
        steps.append(StepResult(name="check_secrets", ok=True, duration_s=0.0))

    extfeed_pre = inspect_extfeed(EXTFEED_MASTER)
    fintables_pre = inspect_fintables(FINTABLES_MASTER) if args.mode == "close" else ParquetMeta(
        path=str(FINTABLES_MASTER), exists=False
    )
    xu100_pre = inspect_xu100(XU100_DAILY) if args.mode == "close" else ParquetMeta(
        path=str(XU100_DAILY), exists=False
    )
    print(
        f"[pre] extfeed exists={extfeed_pre.exists} "
        f"rows={extfeed_pre.row_count} max_ts={extfeed_pre.max_ts_utc}",
        flush=True,
    )
    if args.mode == "close":
        print(
            f"[pre] fintables exists={fintables_pre.exists} "
            f"rows={fintables_pre.row_count} max_date={fintables_pre.max_date}",
            flush=True,
        )
        print(
            f"[pre] xu100 exists={xu100_pre.exists} "
            f"rows={xu100_pre.row_count} max_date={xu100_pre.max_date}",
            flush=True,
        )

    if not extfeed_pre.exists:
        failures.append(f"extfeed master missing pre-run: {EXTFEED_MASTER}")
    if args.mode == "close" and not fintables_pre.exists:
        failures.append(f"fintables master missing pre-run: {FINTABLES_MASTER}")

    if not failures:
        ext_step = run_subprocess(
            [sys.executable, "tools/extfeed_delta_pull.py"],
            step_name="extfeed_delta_pull",
        )
        steps.append(ext_step)
        if not ext_step.ok:
            failures.append(f"extfeed_delta_pull failed: {ext_step.detail}")

        if args.mode == "close" and not failures:
            fin_step = run_subprocess(
                [sys.executable, "-m", "nyxexpansion.tools.rebuild_dataset_delta",
                 "--date", target.isoformat()],
                step_name="rebuild_dataset_delta",
            )
            steps.append(fin_step)
            if not fin_step.ok:
                failures.append(f"rebuild_dataset_delta failed: {fin_step.detail}")

        if args.mode == "close" and not failures:
            xu_step = run_subprocess(
                [sys.executable, "tools/extfeed_pull_xu100.py"],
                step_name="extfeed_pull_xu100",
            )
            steps.append(xu_step)
            if not xu_step.ok:
                failures.append(f"extfeed_pull_xu100 failed: {xu_step.detail}")

    extfeed_post = inspect_extfeed(EXTFEED_MASTER)
    fintables_post = inspect_fintables(FINTABLES_MASTER) if args.mode == "close" else ParquetMeta(
        path=str(FINTABLES_MASTER), exists=False
    )
    xu100_post = inspect_xu100(XU100_DAILY) if args.mode == "close" else ParquetMeta(
        path=str(XU100_DAILY), exists=False
    )
    print(
        f"[post] extfeed exists={extfeed_post.exists} "
        f"rows={extfeed_post.row_count} max_ts={extfeed_post.max_ts_utc}",
        flush=True,
    )
    if args.mode == "close":
        print(
            f"[post] fintables exists={fintables_post.exists} "
            f"rows={fintables_post.row_count} max_date={fintables_post.max_date}",
            flush=True,
        )
        print(
            f"[post] xu100 exists={xu100_post.exists} "
            f"rows={xu100_post.row_count} max_date={xu100_post.max_date}",
            flush=True,
        )

    collapse = row_collapse_check(extfeed_pre, extfeed_post, "extfeed")
    if collapse:
        failures.append(collapse)
    if args.mode == "close":
        collapse_f = row_collapse_check(fintables_pre, fintables_post, "fintables")
        if collapse_f:
            failures.append(collapse_f)
        collapse_x = row_collapse_check(xu100_pre, xu100_post, "xu100")
        if collapse_x:
            failures.append(collapse_x)

    freshness_status, freshness_failures = freshness_check(
        args.mode, target, extfeed_post, fintables_post, xu100_post
    )
    failures.extend(freshness_failures)

    verdict = "PASS" if not failures else "FAIL"
    result = OrchestratorResult(
        mode=args.mode,
        run_timestamp_utc=run_ts,
        operational_target_date=target.isoformat(),
        extfeed_pre=extfeed_pre,
        extfeed_post=extfeed_post,
        fintables_pre=fintables_pre,
        fintables_post=fintables_post,
        xu100_pre=xu100_pre,
        xu100_post=xu100_post,
        freshness_status=freshness_status,
        verdict=verdict,
        failures=failures,
        steps=steps,
    )
    write_manifest(result)

    print(f"[verdict] {verdict}", flush=True)
    if failures:
        for f in failures:
            print(f"  - {f}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
