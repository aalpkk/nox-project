"""Decision Engine v1 — ingestion layer.

Reads scanner outputs and normalizes them into a common record shape
(`ScannerEvent`) that the rest of v1 (label.py / risk.py / paper_stream
_link.py / write.py) consumes.

Critical invariants (from spec §0.1, §0.2, §3.4):

  - This module is allowed to read scanner output parquets / CSVs.
  - This module is **not** allowed to compute `earliness_score_pct`,
    `ema_context_tag`, or any EMA-side classifier from raw OHLCV. Those
    fields are passively carried only when the upstream scanner record
    already exposes them.
  - This module is **not** allowed to read `paper_execution_v0_*` or
    `portfolio_merge_paper_*` parquets — those reads are exclusively
    `paper_stream_link.py`'s responsibility.
  - This module is **not** allowed to invoke any EMA pipeline runner.

Tier 0: scaffold only. Concrete reader bodies are stubbed; no scanner
parquet is opened from this module under Tier 0. The `ScannerEvent`
shape and `normalize_record()` adapter are usable for unit-style
verification by other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScannerEvent:
    """Common record shape consumed by the rest of the v1 pipeline.

    Identity columns mirror v0 (spec §3.1). Optional execution-side
    inputs are nullable; risk.py treats None as missing-input → routes
    to NOT_EXECUTABLE on risk-conditional cells (spec §3 Step 3).

    EMA context fields (`ema_context_tag`, `earliness_score_pct`) are
    descriptive carry only (spec §3.4); they are populated when the
    upstream scanner record already exposes the value, otherwise None.
    """

    # Identity (spec §3.1)
    date: str          # ISO yyyy-mm-dd; v1's asof_date for the event row
    ticker: str
    source: str        # one of V1_SOURCE_WHITELIST
    family: str
    state: str         # panel `phase` discriminator
    timeframe: str
    direction: str     # "long" | "short" (downstream v0 carry; never "AVOID")

    # Execution-side raw inputs (nullable). risk.py owns the four-way branch.
    entry_ref: float | None = None
    stop_ref: float | None = None
    atr: float | None = None
    next_open_gap_if_available: float | None = None
    # Panel-carried, bridged to v1 FILL_ASSUMPTIONS enum at ingest. None if
    # upstream did not supply one (downstream defaults via execution_risk_status).
    fill_assumption: str | None = None

    # Liquidity / capacity (deferred per spec §3.3; carried as nullable)
    liquidity_score: float | None = None
    capacity_score: float | None = None

    # Descriptive EMA context (spec §3.4 — passive carry only)
    ema_context_tag: str | None = None
    earliness_score_pct: float | None = None

    # Regime carry (spec §3.4)
    regime: str | None = None
    regime_stale_flag: str | None = None
    higher_tf_context: str | None = None
    lower_tf_context: str | None = None

    # Auxiliary descriptive lists (carried verbatim from upstream when present)
    supporting_signals: tuple[str, ...] = field(default_factory=tuple)
    conflict_flags: tuple[str, ...] = field(default_factory=tuple)
    hw_context_tags: tuple[str, ...] = field(default_factory=tuple)
    ema_context_tags: tuple[str, ...] = field(default_factory=tuple)


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if isinstance(value, str):
        return (value,) if value else ()
    raise TypeError(
        f"expected list/tuple/str/None for str-tuple field, got {type(value)!r}"
    )


def _coerce_optional(value: Any, cast) -> Any:
    if value is None:
        return None
    try:
        # NaN floats from pandas become real None for our downstream
        if isinstance(value, float) and value != value:
            return None
    except Exception:
        pass
    return cast(value)


def normalize_record(raw: dict[str, Any]) -> ScannerEvent:
    """Adapt one raw scanner-row dict into a `ScannerEvent`.

    The contract:
      - The seven identity fields (`date`, `ticker`, `source`, `family`,
        `state`, `timeframe`, `direction`) are mandatory; missing → KeyError.
      - All other fields are optional; missing keys default to None / ().
      - `earliness_score_pct` and `ema_context_tag` are passively copied;
        the adapter does NOT compute them.

    No paper-stream lookup happens here.
    """
    return ScannerEvent(
        date=str(raw["date"]),
        ticker=str(raw["ticker"]),
        source=str(raw["source"]),
        family=str(raw["family"]),
        state=str(raw["state"]),
        timeframe=str(raw["timeframe"]),
        direction=str(raw["direction"]),
        entry_ref=_coerce_optional(raw.get("entry_ref"), float),
        stop_ref=_coerce_optional(raw.get("stop_ref"), float),
        atr=_coerce_optional(raw.get("atr"), float),
        next_open_gap_if_available=_coerce_optional(
            raw.get("next_open_gap_if_available"), float
        ),
        fill_assumption=_bridge_fill_assumption(raw.get("fill_assumption")),
        liquidity_score=_coerce_optional(raw.get("liquidity_score"), float),
        capacity_score=_coerce_optional(raw.get("capacity_score"), float),
        ema_context_tag=_coerce_optional(raw.get("ema_context_tag"), str),
        earliness_score_pct=_coerce_optional(raw.get("earliness_score_pct"), float),
        regime=_coerce_optional(raw.get("regime"), str),
        regime_stale_flag=_coerce_optional(raw.get("regime_stale_flag"), str),
        higher_tf_context=_coerce_optional(raw.get("higher_tf_context"), str),
        lower_tf_context=_coerce_optional(raw.get("lower_tf_context"), str),
        supporting_signals=_coerce_str_tuple(raw.get("supporting_signals")),
        conflict_flags=_coerce_str_tuple(raw.get("conflict_flags")),
        hw_context_tags=_coerce_str_tuple(raw.get("hw_context_tags")),
        ema_context_tags=_coerce_str_tuple(raw.get("ema_context_tags")),
    )


# LOCKED panel path (decision_v0 classification panel — frozen v0 output;
# v1 reads the cell-mapping rows verbatim, drops v0-side ranker / forward-
# return / reason-code columns; v1 NEVER mutates this file).
DEFAULT_PANEL_PATH: str = "output/decision_v0_classification_panel.parquet"

# Columns we drop on ingest because they are v0-side artefacts that v1
# either recomputes (risk_pct / risk_atr) or is FORBIDDEN to consume
# (final_action, forward returns ret_* / mfe_* / mae_*, v0 reason_codes,
# extension_atr — out-of-scope at Tier 1 dry-run).
_PANEL_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "final_action",
        "reason_codes",
        "ret_5d", "ret_10d", "ret_20d",
        "mfe_5d", "mfe_10d", "mfe_20d",
        "mae_5d", "mae_10d", "mae_20d",
        "risk_pct",
        "risk_atr",
        "extension_atr",
    }
)


# Panel `fill_assumption` (v0 enum) → v1 `FILL_ASSUMPTIONS` enum bridge.
# Panel produces the per-source/state v0 enum during panel-refresh; v1's
# downstream (schema.FILL_ASSUMPTIONS) accepts only {next_open, close_confirmed,
# unresolved}. The mapping is mechanical, source-agnostic, and reversible:
_PANEL_TO_V1_FILL_ASSUMPTION: dict[str, str] = {
    # close-based signal — emitted at bar close; execution semantics = next_open.
    "close_based_signal": "next_open",
    # horizontal_base scanner emits price at close OR at the event bar; v1
    # treats both as next_open execution semantics for risk-conditional rows.
    "close_or_event_price_from_scanner": "next_open",
    # context-only rows have no executable fill expectation.
    "context_only_not_applicable": "unresolved",
    # wait-confirmation rows are not directly executable; carry as unresolved.
    "wait_confirmation_not_executable": "unresolved",
    # already in v1 enum — passthrough.
    "next_open": "next_open",
    "close_confirmed": "close_confirmed",
    "unresolved": "unresolved",
}


def _bridge_fill_assumption(value: Any) -> str | None:
    """Map upstream `fill_assumption` (panel v0 enum or already-v1 enum) to
    a v1 `FILL_ASSUMPTIONS` token. Returns None when the upstream value is
    missing (None / NaN / empty); callers may then fall back to the runner's
    derived default. Raises ValueError if the upstream value is non-empty
    but unknown — surfacing schema drift loudly rather than silently
    forcing `unresolved`.
    """
    if value is None:
        return None
    # NaN-float guard (pandas/pyarrow can hand back NaN floats).
    try:
        if isinstance(value, float) and value != value:
            return None
    except Exception:
        pass
    s = str(value)
    if s == "" or s.lower() == "nan":
        return None
    if s not in _PANEL_TO_V1_FILL_ASSUMPTION:
        raise ValueError(
            f"unknown panel fill_assumption {s!r}; expected one of "
            f"{sorted(_PANEL_TO_V1_FILL_ASSUMPTION.keys())}"
        )
    return _PANEL_TO_V1_FILL_ASSUMPTION[s]


def _regime_stale_flag_from_days(days: float | None) -> str | None:
    """Map panel `regime_stale_days` (float) → v1 `regime_stale_flag` enum.

    v1 carries this as a passive descriptive string per spec §3.4. Mapping
    is mechanical — no policy decision:
        None / NaN          → None    (carry-as-missing; not "missing" enum)
        days <  1.0         → "fresh"
        1.0 ≤ days < 2.0    → "stale_1d"
        days ≥ 2.0          → "stale_2d_plus"
    """
    if days is None:
        return None
    try:
        if days != days:  # NaN
            return None
    except Exception:
        return None
    if days < 1.0:
        return "fresh"
    if days < 2.0:
        return "stale_1d"
    return "stale_2d_plus"


def read_scanner_outputs(
    asof_date: str,
    panel_path: str = DEFAULT_PANEL_PATH,
) -> list[ScannerEvent]:
    """Read v0 classification panel rows for `asof_date` and produce
    `ScannerEvent` records.

    Tier 1 spec §2 + LOCK §16.2 authorize reading the v0 classification
    panel parquet read-only as the consolidated cell-mapping source for
    the six LOCKED v1 sources. The file is NOT mutated. v0-specific
    columns are dropped per `_PANEL_FORBIDDEN_COLUMNS`. Post-Class-A
    panel-refresh patch (Shape 2 carry, 2026-05-06) the panel exposes
    `atr` and `fill_assumption` directly; ingest reads both, bridges
    `fill_assumption` to the v1 enum, and lets `risk.py` route the
    four-way branch on real values rather than always falling through
    to `fill_realism_unresolved`.

    Args:
        asof_date: ISO yyyy-mm-dd. The panel is filtered to rows whose
            `date` column equals this value. If the panel has zero rows
            for `asof_date`, returns an empty list (caller decides whether
            empty is a halt condition).
        panel_path: panel parquet path (defaults to LOCKED).

    Returns:
        list[ScannerEvent] — one event per panel row. Identity columns
        always populated; execution-side raw inputs (entry_ref, stop_ref)
        passed through (None → None).
    """
    import datetime as _date
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    table = pq.read_table(panel_path)
    # Drop forbidden v0 columns up-front.
    keep_cols = [c for c in table.column_names if c not in _PANEL_FORBIDDEN_COLUMNS]
    table = table.select(keep_cols)
    # Filter to asof_date. Panel `date` column is pyarrow date32; convert
    # ISO string to a real date object before scalar construction (pa.scalar
    # with type=date32 does not auto-parse strings).
    asof_date_obj = _date.date.fromisoformat(asof_date)
    date_col = table.column("date")
    mask = pc.equal(date_col, pa.scalar(asof_date_obj, type=date_col.type))
    table = table.filter(mask)

    records: list[ScannerEvent] = []
    for row in table.to_pylist():
        date_val = row.get("date")
        date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)
        regime_stale_flag = _regime_stale_flag_from_days(row.get("regime_stale_days"))
        records.append(
            ScannerEvent(
                date=date_str,
                ticker=str(row["ticker"]),
                source=str(row["source"]),
                family=str(row["family"]),
                # panel uses `phase`; v1 schema name is `state`.
                state=str(row["phase"]),
                timeframe=str(row["timeframe"]),
                # panel does not expose `direction`; v0 long-bias carry.
                direction="long",
                entry_ref=_coerce_optional(row.get("entry_ref"), float),
                stop_ref=_coerce_optional(row.get("stop_ref"), float),
                # Class A patch (2026-05-06): panel now carries `atr` and
                # `fill_assumption` through replay_v0 via extended
                # ACTION_COLUMNS. Read both directly from the panel row.
                atr=_coerce_optional(row.get("atr"), float),
                next_open_gap_if_available=None,
                fill_assumption=_bridge_fill_assumption(row.get("fill_assumption")),
                liquidity_score=None,
                capacity_score=None,
                ema_context_tag=None,
                earliness_score_pct=None,
                regime=_coerce_optional(row.get("regime"), str),
                regime_stale_flag=regime_stale_flag,
                higher_tf_context=None,
                lower_tf_context=None,
                supporting_signals=(),
                conflict_flags=(),
                hw_context_tags=(),
                ema_context_tags=(),
            )
        )
    return records


def panel_max_date(panel_path: str = DEFAULT_PANEL_PATH) -> str | None:
    """Return ISO yyyy-mm-dd of the latest `date` in the panel (for
    runtime asof_date resolution per LOCK §16.1). Returns None if the
    panel is empty.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    table = pq.read_table(panel_path, columns=["date"])
    if table.num_rows == 0:
        return None
    max_date = pc.max(table.column("date")).as_py()
    return max_date.isoformat() if hasattr(max_date, "isoformat") else str(max_date)


__all__ = [
    "ScannerEvent",
    "normalize_record",
    "read_scanner_outputs",
    "panel_max_date",
    "DEFAULT_PANEL_PATH",
]
