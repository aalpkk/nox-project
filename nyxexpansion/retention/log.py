"""Daily retention log writer.

Appends one row per scan run to ``output/nyxexp_retention_log.csv`` so we
can audit how the timing-clean retention filter is behaving in production
over time. Captures pass/drop/unscored counts, surrogate schema version, and
the per-source bar provider breakdown (which tier of fetch_layered served
each candidate).

Idempotency: if a row with the same ``date`` already exists, it is replaced
rather than duplicated. This makes intra-day reruns safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOG_PATH = Path("output/nyxexp_retention_log.csv")
LOG_COLUMNS = (
    "date",
    "n_total",
    "n_pass",
    "n_drop",
    "n_unscored",
    "rank_threshold",
    "surrogate_schema",
    "surrogate_artifact",
    "source_breakdown",
    "notes",
)


def _serialize(d: dict) -> str:
    return json.dumps({str(k): int(v) for k, v in (d or {}).items()},
                      sort_keys=True, ensure_ascii=False)


def append_row(
    target_date: pd.Timestamp,
    *,
    retention_meta: dict,
    surrogate_schema: str,
    out_path: Path = LOG_PATH,
) -> Path:
    """Append (or replace) the row for ``target_date`` in the retention log.

    ``retention_meta`` is the dict ``run.py`` already builds for the HTML
    report — we just persist its key counts plus per-source breakdown.
    """
    out_path = Path(out_path)
    target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    row = {
        "date": target_str,
        "n_total": int(
            (retention_meta.get("n_pass", 0) or 0)
            + (retention_meta.get("n_drop", 0) or 0)
        ),
        "n_pass": int(retention_meta.get("n_pass", 0) or 0),
        "n_drop": int(retention_meta.get("n_drop", 0) or 0),
        "n_unscored": int(retention_meta.get("n_unscored", 0) or 0),
        "rank_threshold": int(retention_meta.get("rank_threshold", 10) or 10),
        "surrogate_schema": str(surrogate_schema),
        "surrogate_artifact": str(retention_meta.get("artifact", "")),
        "source_breakdown": _serialize(retention_meta.get("source_breakdown", {})),
        "notes": _serialize(retention_meta.get("notes", {})),
    }

    if out_path.exists():
        existing = pd.read_csv(out_path, dtype={"date": str})
        existing = existing[existing["date"] != target_str]
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.sort_values("date").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, columns=list(LOG_COLUMNS))
    return out_path
