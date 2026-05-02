"""mb_scanner Phase 1 Event-Quality Diagnostic — outcome + baseline runner.

Implements the locked spec at
`memory/mb_scanner_phase1_event_quality.md` (`mb_scanner_phase1_event_quality_v1`):

- 24 cohorts = 8 families × 3 active states (above_mb_birth, mit_touch_first,
  retest_bounce_first). `extended` is reference-only, not a Phase 1 cohort.
- Horizons: [1, 3, 5, 10, 20]. For 5h family these are 5h-bars; for
  1d / 1w / 1M they are trading-days resolved on the daily panel.
- Entry bar = event-bar close (event_close from event log).
- Cost: 0 bps.
- 3 baselines: B0 calendar-matched mean (active universe at event date,
  same panel-frequency), B1 = 1.57 (q60 rebound PF from
  baseline_decomp_v1), B2 = same-ticker buy-and-hold mean (no selection).
- Acceptance gate per cohort: PASS ≥3/5 horizons clear; horizon clears
  iff `mean(events.r_h) > mean(B0.r_h) AND PF(events.r_h) > 1.57`.
  WEAK if 1-2 clear, FAIL if 0, thin if N<30.

Pre-registration: this is a single-authorized-run diagnostic. Do not
post-hoc tweak params. Successor threads not rescues.

Public entry: `run_phase1(...) -> dict`. Driver: `tools/mb_scanner_phase1_run.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data import intraday_1h

from .engine import _PARAMS as FAM_PARAMS
from .events import EVENT_TYPES
from .resample import per_ticker_panel, to_5h, to_daily

PHASE1_SPEC_ID = "mb_scanner_phase1_event_quality_v1"
HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)

Q60_REBOUND_PF = 1.57          # B1 (locked from baseline_decomp_v1)
COHORT_THIN_N = 30             # below this we skip verdict
COHORT_PASS_HORIZONS = 3       # ≥3/5 → PASS
COHORT_WEAK_HORIZONS = 1       # 1-2 → WEAK

# nox_intraday_v1 frozen split bounds (memory/intraday_dataset_phase3.md).
SPLIT_BOUNDS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "TRAIN": (pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-31")),
    "VAL":   (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31")),
    "TEST":  (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-04-24")),
}


# ----------------------------- forward features -----------------------------

def _forward_features(panel: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Per-bar forward-window features used both for events and baselines.

    For each horizon h, position i gets:
        r_h    = (close[i+h] - close[i]) / close[i]
        mfe_h  = max(high[i+1 : i+h+1]) - close[i]
        mae_h  = close[i] - min(low [i+1 : i+h+1])
    NaN where the forward window runs off the panel end.
    """
    n = len(panel)
    closes = panel["close"].to_numpy(dtype=float)
    highs = panel["high"].to_numpy(dtype=float)
    lows = panel["low"].to_numpy(dtype=float)

    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        r = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        if n > h:
            entry = closes[:n - h]
            target = closes[h:]
            with np.errstate(invalid="ignore", divide="ignore"):
                r[:n - h] = np.where(entry > 0, (target - entry) / entry, np.nan)
            # forward window high/low max/min for window of size h ending at i+h
            from numpy.lib.stride_tricks import sliding_window_view
            hw = sliding_window_view(highs, h)  # rows = positions i.. ; row j = highs[j:j+h]
            lw = sliding_window_view(lows, h)
            # we want max(highs[i+1:i+h+1]) at position i, i.e. window starting at i+1
            # → row index = i+1
            for_max = hw[1:n - h + 1].max(axis=1)
            for_min = lw[1:n - h + 1].min(axis=1)
            mfe[:n - h] = for_max - entry
            mae[:n - h] = entry - for_min
        out[f"r_{h}"] = r
        out[f"mfe_{h}"] = mfe
        out[f"mae_{h}"] = mae
    return out


def _build_forward_panel_long(
    panel_map: dict[str, pd.DataFrame],
    horizons: Iterable[int],
    *,
    date_col: str,
) -> pd.DataFrame:
    """Stack per-ticker panels with forward features into a single long frame.

    Columns: ticker, idx (positional), `date_col`, close, plus r_h/mfe_h/mae_h.
    `date_col` is the bar-date (calendar date — Asia/Istanbul-day for 5h panel).
    """
    rows = []
    for ticker, df in panel_map.items():
        if df.empty or "close" not in df.columns:
            continue
        feats = _forward_features(df, horizons)
        bar_dates = pd.to_datetime(df.index).tz_localize(None).normalize() if df.index.tz is not None else pd.to_datetime(df.index).normalize()
        sub = pd.DataFrame({
            "ticker": ticker,
            "idx": np.arange(len(df), dtype=np.int64),
            date_col: bar_dates,
            "close": df["close"].to_numpy(dtype=float),
        })
        for col in feats.columns:
            sub[col] = feats[col].to_numpy()
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ----------------------------- outcome attachment -----------------------------

def _attach_outcomes_one_family(
    events_df: pd.DataFrame,
    family: str,
    fwd_5h: pd.DataFrame,
    fwd_daily: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Join forward features (close-based r_h, mfe_h, mae_h) onto events.

    For 5h family: join on (ticker, idx) where idx = event_idx in 5h panel.
    For 1d/1w/1M: join on (ticker, bar_date) in daily panel — daily horizons.
    """
    if events_df.empty:
        return events_df
    fam = FAM_PARAMS[family]

    h_cols = [f"r_{h}" for h in horizons] + [f"mfe_{h}" for h in horizons] + [f"mae_{h}" for h in horizons]

    if fam.frequency == "5h":
        right = fwd_5h.rename(columns={"close": "_panel_close"})
        merged = events_df.merge(
            right[["ticker", "idx", "_panel_close"] + h_cols],
            left_on=["ticker", "event_idx"],
            right_on=["ticker", "idx"],
            how="left",
        )
        merged = merged.drop(columns=["idx"])
    else:
        right = fwd_daily.rename(columns={"close": "_panel_close"})
        ev = events_df.copy()
        ev["_event_bar_date_norm"] = pd.to_datetime(ev["event_bar_date"]).dt.normalize()
        merged = ev.merge(
            right[["ticker", "date", "_panel_close"] + h_cols],
            left_on=["ticker", "_event_bar_date_norm"],
            right_on=["ticker", "date"],
            how="left",
        )
        merged = merged.drop(columns=["_event_bar_date_norm", "date"])

    # Convert MFE/MAE to R units using event's structural invalidation.
    inval = merged["structural_invalidation_low"].to_numpy(dtype=float)
    entry = merged["event_close"].to_numpy(dtype=float)
    risk = entry - inval
    risk_safe = np.where(risk > 0, risk, np.nan)
    for h in horizons:
        merged[f"mfe_r_{h}"] = merged[f"mfe_{h}"].to_numpy(dtype=float) / risk_safe
        merged[f"mae_r_{h}"] = merged[f"mae_{h}"].to_numpy(dtype=float) / risk_safe

    return merged


def attach_outcomes(
    events_by_family: dict[str, pd.DataFrame],
    panel_5h_map: dict[str, pd.DataFrame],
    panel_daily_map: dict[str, pd.DataFrame],
    horizons: Iterable[int] = HORIZONS,
) -> dict[str, pd.DataFrame]:
    """Compute forward outcomes for every event in every family.

    For 5h family, forward window is 5h-bars on the 5h panel.
    For 1d / 1w / 1M, forward window is trading-days on the daily panel.
    """
    fwd_5h = _build_forward_panel_long(panel_5h_map, horizons, date_col="date_5h")
    fwd_daily = _build_forward_panel_long(panel_daily_map, horizons, date_col="date")
    out: dict[str, pd.DataFrame] = {}
    for fam_key, ev in events_by_family.items():
        out[fam_key] = _attach_outcomes_one_family(
            ev, fam_key, fwd_5h=fwd_5h, fwd_daily=fwd_daily, horizons=horizons,
        )
    return out


# ----------------------------- baselines -----------------------------

def compute_b0_calendar_mean(
    fwd_panel_long: pd.DataFrame,
    horizons: Iterable[int],
    date_col: str,
) -> pd.DataFrame:
    """B0 calendar-matched baseline: mean forward return per bar-date, pooled
    across all tickers active that date in the same-frequency panel.
    Returns a frame indexed by `date_col` with one column per horizon.
    """
    if fwd_panel_long.empty:
        return pd.DataFrame()
    cols = [f"r_{h}" for h in horizons]
    grp = fwd_panel_long.groupby(date_col)[cols].mean()
    return grp


def compute_b2_ticker_mean(
    fwd_panel_long: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """B2 same-ticker buy-and-hold baseline: mean forward return per ticker
    (no date selection). Returns a frame indexed by `ticker`.
    """
    if fwd_panel_long.empty:
        return pd.DataFrame()
    cols = [f"r_{h}" for h in horizons]
    return fwd_panel_long.groupby("ticker")[cols].mean()


def attach_baseline_lookups(
    events_with_outcomes: dict[str, pd.DataFrame],
    fwd_5h: pd.DataFrame,
    fwd_daily: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
) -> dict[str, pd.DataFrame]:
    """Add per-event B0 (calendar) and B2 (ticker) mean forward returns.

    For each event row e and horizon h:
        b0_r_h = mean r_h over (panel_freq, e.event_bar_date)  [active universe]
        b2_r_h = mean r_h over panel_freq for same ticker      [all dates]
    Where panel_freq is 5h for 5h family, daily otherwise.
    """
    b0_5h = compute_b0_calendar_mean(fwd_5h, horizons, "date_5h")
    b0_d = compute_b0_calendar_mean(fwd_daily, horizons, "date")
    b2_5h = compute_b2_ticker_mean(fwd_5h, horizons)
    b2_d = compute_b2_ticker_mean(fwd_daily, horizons)

    out: dict[str, pd.DataFrame] = {}
    for fam_key, ev in events_with_outcomes.items():
        if ev.empty:
            out[fam_key] = ev
            continue
        fam = FAM_PARAMS[fam_key]
        e = ev.copy()
        ev_dates = pd.to_datetime(e["event_bar_date"]).dt.normalize()

        if fam.frequency == "5h":
            b0_idx = b0_5h.reindex(ev_dates)
            b2_idx = b2_5h.reindex(e["ticker"].to_numpy())
        else:
            b0_idx = b0_d.reindex(ev_dates)
            b2_idx = b2_d.reindex(e["ticker"].to_numpy())

        for h in horizons:
            e[f"b0_r_{h}"] = b0_idx[f"r_{h}"].to_numpy() if not b0_idx.empty else np.nan
            e[f"b2_r_{h}"] = b2_idx[f"r_{h}"].to_numpy() if not b2_idx.empty else np.nan
        out[fam_key] = e
    return out


# ----------------------------- splits -----------------------------

def assign_split(date) -> str:
    if pd.isna(date):
        return "OUT_OF_SCOPE"
    d = pd.Timestamp(date).normalize()
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        if lo <= d <= hi:
            return name
    return "OUT_OF_SCOPE"


# ----------------------------- cohort aggregation + gate -----------------------------

def _profit_factor(rets: np.ndarray) -> float:
    rets = rets[~np.isnan(rets)]
    if rets.size == 0:
        return float("nan")
    pos = rets[rets > 0].sum()
    neg = -rets[rets < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def _hit_rate(rets: np.ndarray) -> float:
    rets = rets[~np.isnan(rets)]
    if rets.size == 0:
        return float("nan")
    return float((rets > 0).mean())


def cohort_metrics(
    events_with_outcomes: dict[str, pd.DataFrame],
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Aggregate per (family, event_type, horizon) cohort stats incl. baselines."""
    rows = []
    for fam_key, ev in events_with_outcomes.items():
        if ev.empty:
            continue
        for et in EVENT_TYPES:
            sub = ev[ev["event_type"] == et]
            n_total = len(sub)
            for h in horizons:
                col = f"r_{h}"
                if col not in sub.columns:
                    continue
                r = sub[col].to_numpy(dtype=float)
                mask = ~np.isnan(r)
                rr = r[mask]
                n = int(rr.size)
                if n == 0:
                    rows.append({
                        "family": fam_key, "event_type": et, "horizon": h,
                        "N_total": n_total, "N_eval": 0,
                    })
                    continue

                mfe_r = sub[f"mfe_r_{h}"].to_numpy(dtype=float)
                mae_r = sub[f"mae_r_{h}"].to_numpy(dtype=float)
                b0 = sub[f"b0_r_{h}"].to_numpy(dtype=float)
                b2 = sub[f"b2_r_{h}"].to_numpy(dtype=float)

                rows.append({
                    "family": fam_key,
                    "event_type": et,
                    "horizon": h,
                    "N_total": n_total,
                    "N_eval": n,
                    "raw_ret_mean": float(np.nanmean(rr)),
                    "raw_ret_median": float(np.nanmedian(rr)),
                    "raw_ret_std": float(np.nanstd(rr, ddof=1)) if n > 1 else float("nan"),
                    "hit_rate": _hit_rate(rr),
                    "PF": _profit_factor(rr),
                    "MFE_R_mean": float(np.nanmean(mfe_r)),
                    "MAE_R_mean": float(np.nanmean(mae_r)),
                    "B0_mean": float(np.nanmean(b0)),
                    "B2_mean": float(np.nanmean(b2)),
                })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CohortVerdict:
    family: str
    event_type: str
    horizons_clear: int
    horizons_eval: int
    N_total: int
    verdict: str  # PASS / WEAK / FAIL / thin
    cleared_horizons: tuple[int, ...]


def apply_acceptance_gate(metrics: pd.DataFrame, b1_pf: float = Q60_REBOUND_PF) -> pd.DataFrame:
    """For each (family, event_type) cohort, count horizons that clear gate.

    Horizon clears iff:  raw_ret_mean > B0_mean  AND  PF > b1_pf.
    Verdict tagging: PASS ≥3/5, WEAK 1-2, FAIL 0, thin if N_total < 30.
    """
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (fam, et), sub in metrics.groupby(["family", "event_type"]):
        n_total = int(sub["N_total"].max()) if "N_total" in sub.columns else 0
        cleared = []
        for _, r in sub.iterrows():
            if r.get("N_eval", 0) <= 0:
                continue
            mean_ok = pd.notna(r["raw_ret_mean"]) and pd.notna(r["B0_mean"]) and r["raw_ret_mean"] > r["B0_mean"]
            pf_val = r["PF"]
            pf_ok = pd.notna(pf_val) and (pf_val == float("inf") or pf_val > b1_pf)
            if mean_ok and pf_ok:
                cleared.append(int(r["horizon"]))
        n_eval_horizons = int((sub["N_eval"] > 0).sum())
        if n_total < COHORT_THIN_N:
            verdict = "thin"
        elif len(cleared) >= COHORT_PASS_HORIZONS:
            verdict = "PASS"
        elif len(cleared) >= COHORT_WEAK_HORIZONS:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        rows.append({
            "family": fam, "event_type": et,
            "N_total": n_total,
            "horizons_eval": n_eval_horizons,
            "horizons_clear": len(cleared),
            "cleared_horizons": ",".join(map(str, cleared)) if cleared else "",
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# ----------------------------- driver -----------------------------

def _load_events(events_dir: Path, families: Iterable[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for fam in families:
        p = events_dir / f"mb_scanner_events_{fam}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"event log missing for {fam}: {p} — run tools/mb_scanner_events_run.py")
        out[fam] = pd.read_parquet(p)
    return out


def run_phase1(
    *,
    families: Iterable[str] | None = None,
    tickers: Iterable[str] | None = None,
    min_coverage: float = 0.0,
    events_dir: str | Path = Path("output"),
    horizons: Iterable[int] = HORIZONS,
    write_outputs: bool = True,
    out_dir: str | Path = Path("output"),
) -> dict:
    """Single authorized Phase 1 run — diagnostic only, no model.

    Returns a dict with:
        events_with_outcomes:  {family: DataFrame} (event-level per-row outcomes)
        cohort_metrics:        DataFrame (per family×event_type×horizon)
        verdicts:              DataFrame (per family×event_type cohort)
    """
    if families is None:
        families = list(FAM_PARAMS.keys())
    fam_keys = list(families)

    events = _load_events(Path(events_dir), fam_keys)

    bars = intraday_1h.load_intraday(
        tickers=list(tickers) if tickers is not None else None,
        start=None, end=None, min_coverage=min_coverage,
    )
    panel_5h = per_ticker_panel(to_5h(bars), "ts_istanbul")
    panel_daily = per_ticker_panel(to_daily(bars), "date")

    fwd_5h = _build_forward_panel_long(panel_5h, horizons, date_col="date_5h")
    fwd_daily = _build_forward_panel_long(panel_daily, horizons, date_col="date")

    ev_outcomes = attach_outcomes(events, panel_5h, panel_daily, horizons)
    ev_with_b = attach_baseline_lookups(ev_outcomes, fwd_5h, fwd_daily, horizons)

    # split labels for bookkeeping (does not affect gate)
    for fam_key, e in ev_with_b.items():
        if e.empty:
            continue
        e["split"] = pd.to_datetime(e["event_bar_date"]).map(assign_split)

    metrics = cohort_metrics(ev_with_b, horizons)
    verdicts = apply_acceptance_gate(metrics)

    if write_outputs:
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        # per-family event+outcome parquet
        for fam_key, e in ev_with_b.items():
            if e.empty:
                continue
            target = out_dir_p / f"mb_scanner_phase1_events_{fam_key}.parquet"
            e.to_parquet(target, index=False)
        metrics.to_csv(out_dir_p / "mb_scanner_phase1_cohort_metrics.csv", index=False)
        verdicts.to_csv(out_dir_p / "mb_scanner_phase1_verdicts.csv", index=False)

    return {
        "spec_id": PHASE1_SPEC_ID,
        "horizons": tuple(horizons),
        "events_with_outcomes": ev_with_b,
        "cohort_metrics": metrics,
        "verdicts": verdicts,
        "b1_pf": Q60_REBOUND_PF,
    }
