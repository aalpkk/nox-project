"""Decision Engine v0 — H1/H2/HW handoff metadata application.

Locked spec: memory/decision_engine_v0_spec.md §Locked handoffs / §Stale-prior policy.

Responsibilities:
1. Promote H1 cohorts (mb_5h__above_mb_birth, mb_1d__above_mb_birth) from
   `continuation` → `accepted_continuation` and stamp horizon metadata.
2. Stamp H2 cohort (mb_1M__above_mb_birth) with default_unchanged horizon
   metadata (NO phase upgrade).
3. Apply stale-prior re-eval flags (6 months OR 1,000 OOS events).
4. HW signals: descriptive only; v0 carries no HW-derived exit field.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from .schema import ACCEPTED_PRIORS, AcceptedPrior

# ─── stale-prior policy ──────────────────────────────────────────────────
STALE_MONTHS = 6  # ~180 days


def _parse_iso(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _is_review_due(prior: AcceptedPrior, today: date, oos_events: int) -> bool:
    """Re-eval triggers on FIRST of: 6 months elapsed OR 1,000 OOS events."""
    accepted = _parse_iso(prior.horizon_accepted_date)
    elapsed_days = (today - accepted).days
    if elapsed_days >= 30 * STALE_MONTHS:
        return True
    if oos_events >= prior.review_event_count:
        return True
    return False


def apply_handoffs(events: pd.DataFrame, *, today: date | None = None) -> pd.DataFrame:
    """Mutate events with H1/H2 metadata; return new DataFrame.

    `today` defaults to max event date (deterministic in single-run mode).
    """
    if events.empty:
        return events

    df = events.copy()
    if today is None:
        try:
            today = pd.to_datetime(df["date"].max()).date()
        except Exception:
            today = date.today()

    # default fill — nothing accepted, default 10d horizon, source unresolved
    df["expected_horizon"] = 10
    df["horizon_source"] = "default_10d"
    df["horizon_status"] = "default"
    df["horizon_accepted_date"] = None
    df["horizon_review_due"] = None

    prior_map = {p.family: p for p in ACCEPTED_PRIORS}

    # OOS event counts per H1 cohort — deterministic from current event table.
    # Counter starts at 0 on 2026-05-03 accepted_date and accumulates strictly
    # *after* that date.
    oos_counts: dict[str, int] = {}
    for fam in prior_map:
        accepted = _parse_iso(prior_map[fam].horizon_accepted_date)
        mask = (df["family"] == fam) & (
            pd.to_datetime(df["date"]).dt.date > accepted
        )
        oos_counts[fam] = int(mask.sum())

    # H1: mb_5h__above_mb_birth, mb_1d__above_mb_birth
    for fam, prior in prior_map.items():
        sel = df["family"] == fam
        if not sel.any():
            continue
        df.loc[sel, "phase"] = "accepted_continuation"
        df.loc[sel, "expected_horizon"] = prior.expected_horizon
        df.loc[sel, "horizon_source"] = prior.horizon_source
        df.loc[sel, "horizon_accepted_date"] = prior.horizon_accepted_date
        df.loc[sel, "horizon_review_due"] = prior.horizon_review_due

        if _is_review_due(prior, today, oos_counts[fam]):
            df.loc[sel, "horizon_status"] = "review_due"
            df.loc[sel, "reason_candidates"] = df.loc[sel, "reason_candidates"].apply(
                lambda lst: list(lst or []) + ["horizon_review_due"]
            )
        else:
            df.loc[sel, "horizon_status"] = prior.horizon_status

    # H2: mb_1M__above_mb_birth → default unchanged + h2_failed_no_15d_rule tag
    h2_sel = df["family"] == "mb_1M__above_mb_birth"
    if h2_sel.any():
        df.loc[h2_sel, "expected_horizon"] = 10
        df.loc[h2_sel, "horizon_source"] = "exit_framework_v1_h2_fail"
        df.loc[h2_sel, "horizon_status"] = "default_unchanged"
        df.loc[h2_sel, "reason_candidates"] = df.loc[h2_sel, "reason_candidates"].apply(
            lambda lst: list(lst or []) + ["h2_failed_no_15d_rule"]
        )

    # Regime-stale tag (regime_labels file is offline-only; live detector missing)
    if "regime_stale_days" in df.columns:
        stale_sel = (df["regime_stale_days"].fillna(0) > 0)
        if stale_sel.any():
            df.loc[stale_sel, "reason_candidates"] = df.loc[
                stale_sel, "reason_candidates"
            ].apply(lambda lst: list(lst or []) + ["horizon_status_stale"])

    return df


__all__ = ["apply_handoffs", "STALE_MONTHS"]
