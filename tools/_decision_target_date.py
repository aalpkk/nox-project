"""Decision Engine — Operational Target Date helper.

Replaces the hardcoded `LCTD = "2026-04-30"` constants in upstream-refresh runners
with a calendar-driven derivation backed by `exchange_calendars` (XIST = Borsa
Istanbul cash session). LOCKED 2026-05-05 by
`memory/decision_engine_v1_backbone_refresh_lctd_spec.md` §5 with corrected design
(09:00 TR cutoff rule + 4 asof_modes; legacy `LCTD` term retired in helper API).

Why "operational target date" instead of "LCTD"? "Latest completed trading day"
implied EOD-complete data, blocking intraday scans before BIST cash close
(18:00 TR). The corrected rule lets a 17:00 TR scan run against today's
partial-day data without forcing a regression to yesterday's session.

Public API:
    derive_operational_target(now_utc=None) -> OperationalContext
    assert_freshness_contract(ctx, *, extfeed_master_path, fintables_master_path,
                              output_path=None, require_output_meets_target=False,
                              require_eod=False) -> None

Forbidden by LOCK §5.3:
    - No naive weekday-arithmetic fallback if exchange_calendars unavailable.
    - No fallback to stale-backbone-derived target.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

try:
    import exchange_calendars as _xcals
except ImportError as _e:  # noqa: BLE001
    raise ImportError(
        "lctd_dependency_missing: exchange_calendars (install via "
        "`pip install exchange_calendars`)"
    ) from _e

_TR_TZ = ZoneInfo("Europe/Istanbul")
_UTC = _dt.timezone.utc
_CUTOFF_TR = _dt.time(9, 0)        # premarket vs intraday boundary
_BIST_CLOSE_TR = _dt.time(18, 0)   # cash session close (intraday vs postclose)

AsofMode = Literal[
    "premarket_previous_day",
    "intraday_today",
    "postclose_today",
    "nontrading_previous_session",
]


@dataclass(frozen=True)
class OperationalContext:
    operational_target_date: _dt.date
    asof_mode: AsofMode
    now_utc: _dt.datetime
    now_tr: _dt.datetime
    is_today_trading: bool

    def as_iso(self) -> str:
        return self.operational_target_date.isoformat()


class FreshnessContractError(AssertionError):
    """Raised by assert_freshness_contract on any FAIL."""


def _xist():
    return _xcals.get_calendar("XIST")


def _previous_session_strict_before(xist, d: _dt.date) -> _dt.date:
    """Last XIST session strictly before `d`, regardless of whether `d` itself
    is a session. `xist.previous_session(d)` requires `d` to be a session, so it
    can't be used directly on Saturdays / holidays.
    """
    end = pd.Timestamp(d) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=21)
    sessions = xist.sessions_in_range(start, end)
    if len(sessions) == 0:
        raise RuntimeError(
            f"no_prior_session_in_21d_window: anchor={d} window=[{start.date()},{end.date()}]"
        )
    return sessions[-1].date()


def derive_operational_target(
    now_utc: Optional[_dt.datetime] = None,
) -> OperationalContext:
    """Derive operational_target_date + asof_mode from XIST trading calendar.

    Rule (09:00 TR cutoff):
        - today is XIST trading session AND TR time >= 09:00:
              target = today
              mode = "intraday_today" if TR time < 18:00 else "postclose_today"
        - today is XIST trading session AND TR time <  09:00:
              target = previous XIST session
              mode = "premarket_previous_day"
        - today is NOT a trading session:
              target = previous XIST session
              mode = "nontrading_previous_session"
    """
    if now_utc is None:
        now_utc = _dt.datetime.now(tz=_UTC)
    elif now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware (UTC)")
    else:
        now_utc = now_utc.astimezone(_UTC)

    now_tr = now_utc.astimezone(_TR_TZ)
    today_tr = now_tr.date()

    xist = _xist()
    is_today_trading = bool(xist.is_session(today_tr))

    if is_today_trading:
        if now_tr.time() >= _CUTOFF_TR:
            target = today_tr
            mode: AsofMode = (
                "postclose_today" if now_tr.time() >= _BIST_CLOSE_TR else "intraday_today"
            )
        else:
            target = _previous_session_strict_before(xist, today_tr)
            mode = "premarket_previous_day"
    else:
        target = _previous_session_strict_before(xist, today_tr)
        mode = "nontrading_previous_session"

    return OperationalContext(
        operational_target_date=target,
        asof_mode=mode,
        now_utc=now_utc,
        now_tr=now_tr,
        is_today_trading=is_today_trading,
    )


# ────────────────────────────────────────────────────────────────────────────
# Freshness contract

def _read_max_date_parquet(path: Path, candidates: tuple[str, ...]) -> _dt.date:
    """Return max date across the first matching column in `candidates`.

    For files whose canonical date lives in the parquet index, pyarrow stores
    it as `__index_level_0__`. Try named columns first, then fall back to
    pandas read_parquet which materializes the index.
    """
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)

    for col in candidates:
        if col in schema_names:
            tbl = pq.read_table(path, columns=[col])
            ser = tbl.column(col).to_pandas()
            return pd.to_datetime(ser, utc=True, errors="coerce").max().date()

    # Fallback: load index via pandas (slower but robust for index-stored dates)
    df = pd.read_parquet(path)
    if df.index.name in candidates or df.index.name is None:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        return idx.max().date()
    raise FreshnessContractError(
        f"date_column_not_found: path={path} candidates={candidates} "
        f"available={sorted(schema_names)} index_name={df.index.name!r}"
    )


def _max_date_extfeed(path: Path) -> _dt.date:
    return _read_max_date_parquet(path, ("ts_utc", "ts", "datetime"))


def _max_date_fintables(path: Path) -> _dt.date:
    return _read_max_date_parquet(path, ("Date", "date", "bar_date"))


def assert_freshness_contract(
    ctx: OperationalContext,
    *,
    extfeed_master_path: Path,
    fintables_master_path: Path,
    output_path: Optional[Path] = None,
    require_output_meets_target: bool = False,
    require_eod: bool = False,
) -> None:
    """Mode-aware freshness contract.

    Strict checks (raise on FAIL):
        - extfeed master max_date >= operational_target_date
            (relaxed: in intraday_today, accept master max_date == target - 1
             trading day if today's first bar hasn't landed yet AND
             require_eod is False; FAIL otherwise)
        - fintables master max_date >= operational_target_date
            (relaxed: in intraday_today / postclose_today, accept lag of one
             trading session if require_eod is False; FAIL otherwise)
        - output max_date >= operational_target_date (only when
          require_output_meets_target is True)

    Structural checks:
        - ctx must be an OperationalContext (rejects str literals — this
          replaces the §5.4 (1) "lctd_source != hardcoded" assertion per Q14.C).
    """
    if not isinstance(ctx, OperationalContext):
        raise FreshnessContractError(
            f"target_source_not_dynamic: ctx_type={type(ctx).__name__}; "
            "callers must pass derive_operational_target() output, not literals"
        )

    target = ctx.operational_target_date
    mode = ctx.asof_mode
    extfeed_master_path = Path(extfeed_master_path)
    fintables_master_path = Path(fintables_master_path)

    if not extfeed_master_path.exists():
        raise FreshnessContractError(
            f"extfeed_master_missing: path={extfeed_master_path}"
        )
    if not fintables_master_path.exists():
        raise FreshnessContractError(
            f"fintables_master_missing: path={fintables_master_path}"
        )

    extfeed_max = _max_date_extfeed(extfeed_master_path)
    fintables_max = _max_date_fintables(fintables_master_path)

    xist = _xist()

    def _prev_session(d: _dt.date) -> _dt.date:
        # `target` is always a session by construction in derive_operational_target,
        # so xist.previous_session(target) is safe here.
        return xist.previous_session(d).date()

    intraday_modes = {"intraday_today", "postclose_today"}

    # extfeed check
    if extfeed_max < target:
        if mode in intraday_modes and not require_eod and extfeed_max >= _prev_session(target):
            pass  # acceptable lag during intraday window
        else:
            raise FreshnessContractError(
                f"extfeed_stale: max={extfeed_max} target={target} mode={mode} "
                f"require_eod={require_eod}"
            )

    # fintables check
    if fintables_max < target:
        if mode in intraday_modes and not require_eod and fintables_max >= _prev_session(target):
            pass  # acceptable lag during intraday window (EOD producer hasn't run yet)
        else:
            raise FreshnessContractError(
                f"fintables_stale: max={fintables_max} target={target} mode={mode} "
                f"require_eod={require_eod}"
            )

    # output check (post-refresh callers only)
    if require_output_meets_target:
        if output_path is None:
            raise FreshnessContractError(
                "output_path_missing_when_required: pass output_path when "
                "require_output_meets_target=True"
            )
        output_path = Path(output_path)
        if not output_path.exists():
            raise FreshnessContractError(f"output_missing: path={output_path}")
        # output may be parquet or csv; try both
        suffix = output_path.suffix.lower()
        if suffix == ".parquet":
            out_max = _read_max_date_parquet(
                output_path, ("date", "Date", "bar_date", "ts_utc", "asof_date")
            )
        elif suffix == ".csv":
            df = pd.read_csv(output_path)
            for c in ("date", "Date", "bar_date", "ts_utc"):
                if c in df.columns:
                    out_max = pd.to_datetime(df[c], utc=True, errors="coerce").max().date()
                    break
            else:
                raise FreshnessContractError(
                    f"output_date_column_not_found: path={output_path} "
                    f"columns={list(df.columns)}"
                )
        else:
            raise FreshnessContractError(
                f"output_unsupported_suffix: path={output_path} suffix={suffix}"
            )
        if out_max < target:
            raise FreshnessContractError(
                f"output_stale: max={out_max} target={target} path={output_path}"
            )


__all__ = [
    "OperationalContext",
    "AsofMode",
    "FreshnessContractError",
    "derive_operational_target",
    "assert_freshness_contract",
]
