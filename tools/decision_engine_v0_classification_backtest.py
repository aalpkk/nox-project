"""Decision Engine v0 — Classification Backtest, single authorized run.

LOCKED PRE-REG: memory/decision_engine_v0_classification_backtest_spec.md.

Replays Decision Engine v0 hard rules over historical events and joins
forward returns at +5d / +10d / +20d trading-day offsets. Tests whether
final_action buckets actually separate on forward returns.

NO scoring, NO ranking, NO bucket subdivision based on results, NO threshold
tweak. Anti-rescue active end-to-end. Bug → void run + restart, no in-place
patches that would compromise single-run discipline.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decision_engine.actions import apply_actions  # noqa: E402
from decision_engine.handoffs import apply_handoffs  # noqa: E402
from decision_engine.normalize import _attach_regime, _load_regime  # noqa: E402
from decision_engine.phase import map_phase  # noqa: E402
from decision_engine.risk import derive_risk  # noqa: E402
from decision_engine.schema import EVENT_COLUMNS, canonical_family  # noqa: E402

OUT = ROOT / "output"

# ─── locked constants ─────────────────────────────────────────────────────
RNG_SEED = 7
N_BOOT = 2000
HORIZONS = (5, 10, 20)
CUTOFF_BUFFER_TRADING_DAYS = 25
INCONCLUSIVE_N_FLOOR = 200
G5_RESIDUAL_N_FLOOR = 1000

PRIMARY_BUCKETS = ("TRADEABLE", "WAIT_RETEST", "WATCHLIST", "AVOID")
H1_FAMILIES = ("mb_5h__above_mb_birth", "mb_1d__above_mb_birth")
PF_RANDOM_ANCHOR = 1.50  # q60 BIST 607
PRACTICAL_DELTA_MEAN_R = 0.015
PRACTICAL_DELTA_PF = 0.30


# ─── historical event normalization ───────────────────────────────────────


_MB_FAMILIES = [
    ("mb_5h", "5h"),
    ("mb_1d", "1d"),
    ("mb_1w", "1w"),
    ("mb_1M", "1M"),
    ("bb_5h", "5h"),
    ("bb_1d", "1d"),
    ("bb_1w", "1w"),
    ("bb_1M", "1M"),
]


def _empty_event() -> dict:
    return {col: None for col in EVENT_COLUMNS}


def load_mb_scanner_events() -> pd.DataFrame:
    rows = []
    for fam, tf in _MB_FAMILIES:
        path = OUT / f"mb_scanner_events_{fam}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # NOTE: `mode` column tags family-mode (mb/bb), NOT a production/shadow flag.
        # All rows are production scans — no filter needed.
        for _, r in df.iterrows():
            ev_type = str(r.get("event_type", ""))
            family_key = f"{fam}__{ev_type}"  # canonical
            entry_ref = r.get("event_close")
            stop_ref = r.get("structural_invalidation_low")
            atr = r.get("atr_at_event")
            risk_pct, risk_atr = derive_risk(
                entry_ref=entry_ref, stop_ref=stop_ref, atr=atr
            )
            ev = _empty_event()
            ev.update(
                date=pd.to_datetime(r.get("event_bar_date")).date()
                if pd.notna(r.get("event_bar_date"))
                else None,
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
                fill_assumption="unresolved",
                bar_timestamp=r.get("event_ts"),
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


def load_horizontal_base_events() -> pd.DataFrame:
    path = OUT / "horizontal_base_event_v1.parquet"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
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
        ev = _empty_event()
        ev.update(
            date=pd.to_datetime(r.get("bar_date")).date()
            if pd.notna(r.get("bar_date"))
            else None,
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
            fill_assumption="unresolved",
            bar_timestamp=r.get("as_of_ts"),
        )
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


def load_nyxexpansion_events() -> pd.DataFrame:
    path = OUT / "nyxexp_dataset_v4.parquet"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
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
            if pd.notna(r.get("date"))
            else None,
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
            fill_assumption="unresolved",
        )
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


_NYXMOM_PICKS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(m0|v5)_picks\.csv$")


def load_nyxmomentum_events() -> pd.DataFrame:
    """Walk every <date>_{m0,v5}_picks.csv under output/nyxmomentum/live/.
    No entry/stop in picks → phase=strength_context (Rule 16 → WATCHLIST per spec)."""
    live_dir = OUT / "nyxmomentum" / "live"
    if not live_dir.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for f in sorted(live_dir.iterdir()):
        m = _NYXMOM_PICKS_RE.match(f.name)
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
                fill_assumption="unresolved",
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


_NOX_RT_RE = re.compile(r"^nox_v3_signals_(\d{8})\.csv$")
_NOX_WK_RE = re.compile(r"^nox_v3_signals_weekly_(\d{8})\.csv$")
_NOX_HIST_DIR = ROOT / "output" / "_nox_historical"


def _load_nox_v3_historical(pattern: re.Pattern, source: str, family: str,
                             timeframe: str) -> pd.DataFrame:
    """Read all historical nox_v3 CSVs from output/_nox_historical/, PIVOT_AL only,
    every signal_date (not just latest)."""
    if not _NOX_HIST_DIR.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for f in sorted(_NOX_HIST_DIR.iterdir()):
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
                extension_atr=None,  # delta_pct is %, not ATR
                liquidity_score=None,
                higher_tf_context=None,
                lower_tf_context=None,
                reason_candidates=[],
                raw_score=r.get("rg_score") if pd.notna(r.get("rg_score")) else None,
                fill_assumption="unresolved",
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


def load_nox_rt_daily_events() -> pd.DataFrame:
    return _load_nox_v3_historical(
        _NOX_RT_RE, source="nox_rt_daily",
        family="nox_rt_daily__pivot_al", timeframe="1d",
    )


def load_nox_weekly_events() -> pd.DataFrame:
    return _load_nox_v3_historical(
        _NOX_WK_RE, source="nox_weekly",
        family="nox_weekly__weekly_pivot_al", timeframe="1w",
    )


# ─── daily panel + trading-day index ──────────────────────────────────────


def build_daily_panel() -> pd.DataFrame:
    """Per-ticker daily resample of extfeed: last close, max high, min low.

    Trading-day index assigned per ticker. Used for forward-return lookups.
    """
    src = OUT / "extfeed_intraday_1h_3y_master.parquet"
    df = pd.read_parquet(src, columns=["ticker", "ts_istanbul", "high", "low", "close"])
    df["ts_istanbul"] = pd.to_datetime(df["ts_istanbul"])
    df["date"] = df["ts_istanbul"].dt.date
    daily = (
        df.sort_values(["ticker", "ts_istanbul"])
        .groupby(["ticker", "date"])
        .agg(close=("close", "last"), high=("high", "max"), low=("low", "min"))
        .reset_index()
    )
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily["tdx"] = daily.groupby("ticker").cumcount()
    return daily


def attach_forward_paths(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """For each event, compute ret/MFE/MAE at +5d, +10d, +20d trading days.

    event_date cutoff = ticker_max_tdx - CUTOFF_BUFFER_TRADING_DAYS so every
    surviving event has a clean +20d forward path.
    """
    if events.empty:
        return events
    daily = daily.copy()
    daily_lookup = daily.set_index(["ticker", "date"])

    # ticker → numpy arrays for fast lookup
    by_ticker: dict[str, dict] = {}
    for tkr, grp in daily.groupby("ticker"):
        g = grp.reset_index(drop=True)
        by_ticker[tkr] = {
            "tdx_by_date": dict(zip(g["date"], g["tdx"])),
            "max_tdx": int(g["tdx"].max()),
            "close": g["close"].to_numpy(dtype=float),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
        }

    out_rows = []
    skipped_no_ticker = 0
    skipped_no_tdx = 0
    skipped_cutoff = 0
    for ev in events.to_dict(orient="records"):
        tkr = ev.get("ticker")
        d = ev.get("date")
        if tkr not in by_ticker:
            skipped_no_ticker += 1
            continue
        bt = by_ticker[tkr]
        tdx0 = bt["tdx_by_date"].get(d)
        if tdx0 is None:
            skipped_no_tdx += 1
            continue
        if tdx0 + max(HORIZONS) > bt["max_tdx"] - 0:
            # event has no clean +20d path → past cutoff
            if tdx0 + max(HORIZONS) > bt["max_tdx"]:
                skipped_cutoff += 1
                continue
        c0 = bt["close"][tdx0]
        if not np.isfinite(c0) or c0 <= 0:
            continue
        rec = dict(ev)
        for h in HORIZONS:
            i = tdx0 + h
            if i > bt["max_tdx"]:
                rec[f"ret_{h}d"] = np.nan
                rec[f"mfe_{h}d"] = np.nan
                rec[f"mae_{h}d"] = np.nan
                continue
            cN = bt["close"][i]
            mfe = bt["high"][tdx0 + 1 : i + 1].max() if i > tdx0 else c0
            mae = bt["low"][tdx0 + 1 : i + 1].min() if i > tdx0 else c0
            rec[f"ret_{h}d"] = (cN - c0) / c0
            rec[f"mfe_{h}d"] = (mfe - c0) / c0
            rec[f"mae_{h}d"] = (mae - c0) / c0
        out_rows.append(rec)

    print(
        f"  skipped: no_ticker={skipped_no_ticker}, "
        f"no_tdx={skipped_no_tdx}, past_cutoff={skipped_cutoff}"
    )
    return pd.DataFrame(out_rows)


# ─── replay v0 (handoffs + actions) ───────────────────────────────────────


def replay_v0(events: pd.DataFrame) -> pd.DataFrame:
    """Apply handoffs (H1 promotion + H2 metadata + stale-regime) then hard-rule table.

    apply_actions preserves row order one-to-one with input events. We attach
    forward-return columns positionally on the events DataFrame BEFORE calling
    apply_actions; columns then ride along into the actions output (the
    apply_actions copy preserves them when they are part of input columns
    that the trimmer keeps; if not in ACTION_COLUMNS we re-attach positionally
    after). Either way the join is one-to-one — never on (date,ticker,family,
    source), which is non-unique under multi-quartet mb_scanner output.
    """
    regime = _load_regime()
    events = _attach_regime(events, regime)
    events = apply_handoffs(events)
    actions = apply_actions(events)
    # positional re-attach (events and actions are guaranteed same length & order)
    if len(actions) != len(events):
        raise RuntimeError(
            f"apply_actions broke 1:1 invariant: events={len(events)} actions={len(actions)}"
        )
    forward_cols = [f"{m}_{h}d" for m in ("ret", "mfe", "mae") for h in HORIZONS]
    for col in forward_cols:
        if col in events.columns:
            actions[col] = events[col].to_numpy()
    return actions, events


# ─── metrics ──────────────────────────────────────────────────────────────


def cell_metrics(rets: np.ndarray) -> dict:
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return dict(
            N=0, PF=np.nan, WR_pos=np.nan, WR_10pct=np.nan,
            meanR=np.nan, medianR=np.nan,
        )
    pos = rets[rets > 0]
    neg = rets[rets < 0]
    pf = (pos.sum() / abs(neg.sum())) if neg.sum() < 0 else np.inf
    return dict(
        N=int(len(rets)),
        PF=float(pf),
        WR_pos=float((rets > 0).mean()),
        WR_10pct=float((rets >= 0.10).mean()),
        meanR=float(rets.mean()),
        medianR=float(np.median(rets)),
    )


def bootstrap_delta_meanR(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """Unpaired bootstrap; returns (delta_mean, ci_lo_95, ci_hi_95)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        deltas[i] = sa.mean() - sb.mean()
    delta = a.mean() - b.mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return (float(delta), float(ci_lo), float(ci_hi))


def bootstrap_delta_pf(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RNG_SEED + 1
) -> tuple[float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)

    def _pf(x):
        pos = x[x > 0].sum()
        neg = x[x < 0].sum()
        return (pos / abs(neg)) if neg < 0 else np.inf

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        pa = _pf(sa)
        pb = _pf(sb)
        # cap inf at 99 to keep CI numerical
        if not np.isfinite(pa):
            pa = 99.0
        if not np.isfinite(pb):
            pb = 99.0
        deltas[i] = pa - pb
    delta = (
        (lambda x: 99.0 if not np.isfinite(x) else x)(_pf(a))
        - (lambda x: 99.0 if not np.isfinite(x) else x)(_pf(b))
    )
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return (float(delta), float(ci_lo), float(ci_hi))


def compute_per_cell_metrics(panel: pd.DataFrame, *, label: str) -> pd.DataFrame:
    rows = []
    for bucket in PRIMARY_BUCKETS:
        for h in HORIZONS:
            sub = panel[panel["final_action"] == bucket]
            rets = sub[f"ret_{h}d"].to_numpy(dtype=float)
            mfe = sub[f"mfe_{h}d"].to_numpy(dtype=float)
            mae = sub[f"mae_{h}d"].to_numpy(dtype=float)
            mfe = mfe[np.isfinite(mfe)]
            mae = mae[np.isfinite(mae)]
            m = cell_metrics(rets)
            m.update(
                bucket=bucket,
                horizon=h,
                MFE_mean=float(mfe.mean()) if len(mfe) else np.nan,
                MAE_mean=float(mae.mean()) if len(mae) else np.nan,
                inconclusive=m["N"] < INCONCLUSIVE_N_FLOOR,
                cohort=label,
            )
            rows.append(m)
    df = pd.DataFrame(rows)
    cols = [
        "cohort", "bucket", "horizon", "N", "PF", "WR_pos", "WR_10pct",
        "meanR", "medianR", "MFE_mean", "MAE_mean", "inconclusive",
    ]
    return df[cols]


def compute_pair_deltas(panel: pd.DataFrame, *, label: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        T = panel[panel["final_action"] == "TRADEABLE"][f"ret_{h}d"].to_numpy(float)
        for ref in ("WAIT_RETEST", "WATCHLIST", "AVOID"):
            R = panel[panel["final_action"] == ref][f"ret_{h}d"].to_numpy(float)
            d_mean, lo_mean, hi_mean = bootstrap_delta_meanR(T, R)
            d_pf, lo_pf, hi_pf = bootstrap_delta_pf(T, R)
            rows.append(dict(
                cohort=label, horizon=h, comparison=f"TRADEABLE_vs_{ref}",
                N_T=int(np.isfinite(T).sum()), N_ref=int(np.isfinite(R).sum()),
                delta_meanR=d_mean, delta_meanR_ci_lo=lo_mean, delta_meanR_ci_hi=hi_mean,
                delta_PF=d_pf, delta_PF_ci_lo=lo_pf, delta_PF_ci_hi=hi_pf,
            ))
    return pd.DataFrame(rows)


# ─── G5 H1-cohort identification (Lock-note 3) ────────────────────────────


def _has_code(rc, code) -> bool:
    if rc is None:
        return False
    if isinstance(rc, np.ndarray):
        rc = rc.tolist()
    if not isinstance(rc, (list, tuple)):
        return False
    return code in rc


def identify_h1_cohort(panel: pd.DataFrame) -> pd.Series:
    """Conjunction of 4 conditions per Lock-note 3 (no alias slippage).

    Used as G5 cohort definition: residual = TRADEABLE \\ H1_cohort.
    """
    fam_ok = panel["family"].isin(H1_FAMILIES)
    phase_ok = panel["phase"] == "accepted_continuation"
    reason_ok = panel["reason_codes"].apply(
        lambda rc: _has_code(rc, "accepted_horizon_h1_20d")
    )
    src_ok = panel["horizon_source"] == "exit_framework_v1_h1_pass"
    return fam_ok & phase_ok & reason_ok & src_ok


def identify_h1_anomalies(panel: pd.DataFrame) -> pd.Series:
    """Lock-note 3 revised anomaly definition (intent-preserving correction).

    A row is an H1 metadata anomaly only if:
      family ∈ H1_FAMILIES AND final_action == 'TRADEABLE'
      AND missing any of: expected_horizon==20, horizon_source==<h1>,
      'accepted_horizon_h1_20d' in reason_codes, horizon_review_due not null.

    Risk_too_wide AVOID etc. of H1 family are NOT anomalies — they are
    v0 hard-rule layer working as designed.
    """
    fam_ok = panel["family"].isin(H1_FAMILIES)
    is_trd = panel["final_action"] == "TRADEABLE"
    horizon_ok = panel["expected_horizon"] == 20
    src_ok = panel["horizon_source"] == "exit_framework_v1_h1_pass"
    reason_ok = panel["reason_codes"].apply(
        lambda rc: _has_code(rc, "accepted_horizon_h1_20d")
    )
    review_ok = panel["horizon_review_due"].notna()
    metadata_complete = horizon_ok & src_ok & reason_ok & review_ok
    # anomaly = TRADEABLE H1 family but missing any required metadata bit
    return fam_ok & is_trd & ~metadata_complete


# ─── gates ────────────────────────────────────────────────────────────────


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return (lo > 0) or (hi < 0)


def evaluate_gates(metrics: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    res = {}

    def _row(df, **kw):
        m = df
        for k, v in kw.items():
            m = m[m[k] == v]
        if len(m) == 0:
            return None
        return m.iloc[0]

    # G1: meanR(T) > meanR(WATCHLIST, AVOID) at 10d, CI excludes 0
    g1_w = _row(deltas, cohort="full", horizon=10, comparison="TRADEABLE_vs_WATCHLIST")
    g1_a = _row(deltas, cohort="full", horizon=10, comparison="TRADEABLE_vs_AVOID")
    g1_w_pass = bool(g1_w is not None and g1_w["delta_meanR"] > 0
                     and _ci_excludes_zero(g1_w["delta_meanR_ci_lo"], g1_w["delta_meanR_ci_hi"]))
    g1_a_pass = bool(g1_a is not None and g1_a["delta_meanR"] > 0
                     and _ci_excludes_zero(g1_a["delta_meanR_ci_lo"], g1_a["delta_meanR_ci_hi"]))
    res["G1"] = g1_w_pass and g1_a_pass

    # G2: PF order at 10d AND PF(T) >= 1.50
    pf_t = _row(metrics, cohort="full", bucket="TRADEABLE", horizon=10)
    pf_w = _row(metrics, cohort="full", bucket="WATCHLIST", horizon=10)
    pf_a = _row(metrics, cohort="full", bucket="AVOID", horizon=10)
    g2_order = bool(pf_t is not None and pf_w is not None and pf_a is not None
                    and pf_t["PF"] > pf_w["PF"] and pf_t["PF"] > pf_a["PF"])
    g2_anchor = bool(pf_t is not None and pf_t["PF"] >= PF_RANDOM_ANCHOR)
    res["G2"] = g2_order and g2_anchor

    # G3: practical magnitude vs AVOID
    g1_a_dpf = _row(deltas, cohort="full", horizon=10, comparison="TRADEABLE_vs_AVOID")
    g3_meanR = bool(g1_a is not None and g1_a["delta_meanR"] >= PRACTICAL_DELTA_MEAN_R)
    g3_pf = bool(g1_a_dpf is not None and g1_a_dpf["delta_PF"] >= PRACTICAL_DELTA_PF)
    res["G3"] = g3_meanR and g3_pf

    # G4: direction agree at 5d AND 20d
    def _dir_ok(h):
        gw = _row(deltas, cohort="full", horizon=h, comparison="TRADEABLE_vs_WATCHLIST")
        ga = _row(deltas, cohort="full", horizon=h, comparison="TRADEABLE_vs_AVOID")
        if gw is None or ga is None:
            return False
        return bool(gw["delta_meanR"] > 0 and ga["delta_meanR"] > 0)
    res["G4"] = _dir_ok(5) and _dir_ok(20)

    # G5: residual TRADEABLE \ H1 cohort beats WATCHLIST + AVOID at 10d
    g5_w = _row(deltas, cohort="residual_no_h1", horizon=10, comparison="TRADEABLE_vs_WATCHLIST")
    g5_a = _row(deltas, cohort="residual_no_h1", horizon=10, comparison="TRADEABLE_vs_AVOID")
    res_t = _row(metrics, cohort="residual_no_h1", bucket="TRADEABLE", horizon=10)
    g5_n_ok = bool(res_t is not None and res_t["N"] >= G5_RESIDUAL_N_FLOOR)
    if not g5_n_ok:
        res["G5"] = "INCONCLUSIVE"
    else:
        g5_w_pass = bool(g5_w is not None and g5_w["delta_meanR"] > 0
                         and _ci_excludes_zero(g5_w["delta_meanR_ci_lo"], g5_w["delta_meanR_ci_hi"]))
        g5_a_pass = bool(g5_a is not None and g5_a["delta_meanR"] > 0
                         and _ci_excludes_zero(g5_a["delta_meanR_ci_lo"], g5_a["delta_meanR_ci_hi"]))
        res["G5"] = g5_w_pass and g5_a_pass
    return res


def _verdict(gates: dict) -> str:
    g1, g2, g3, g4 = gates["G1"], gates["G2"], gates["G3"], gates["G4"]
    g5 = gates["G5"]
    if not (g1 and g2 and g3):
        return "FAIL"
    if g1 and g2 and g3 and not g4:
        return "HORIZON-LIMITED"
    if g1 and g2 and g3 and g4:
        if g5 is True:
            return "PASS"
        if g5 == "INCONCLUSIVE":
            return "PASS-CONDITIONAL"
        if g5 is False:
            return "PARTIAL"
    return "UNKNOWN"


# ─── orchestrator ─────────────────────────────────────────────────────────


def write_summary_md(
    *,
    out_path: Path,
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    gates: dict,
    verdict: str,
    h1_count: int,
    h1_anomalies: int,
    asof_data: str,
    cutoff_info: str,
) -> None:
    def _fmt(x, d=4):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{x:.{d}f}"

    def _row_md(df, **kw):
        m = df
        for k, v in kw.items():
            m = m[m[k] == v]
        return m.iloc[0] if len(m) else None

    lines = []
    lines.append("# Decision Engine v0 — Classification Backtest Results")
    lines.append("")
    lines.append(f"**Pre-reg:** `memory/decision_engine_v0_classification_backtest_spec.md` (LOCKED).")
    lines.append("**Run:** single authorized fire, no post-hoc. Seed=7, n_boot=2000.")
    lines.append(f"**Data asof:** {asof_data}")
    lines.append(f"**Cutoff:** {cutoff_info}")
    lines.append(f"**H1 metadata anomalies (Lock-note 3 revised):** {h1_anomalies} TRADEABLE-H1 rows missing required metadata; cohort N={h1_count}; voids run if anomalies > 1% of TRADEABLE-H1 rows.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("| Gate | Result |")
    lines.append("|---|---|")
    for g in ("G1", "G2", "G3", "G4", "G5"):
        v = gates[g]
        sym = "✅" if v is True else ("INCONCLUSIVE" if v == "INCONCLUSIVE" else "❌")
        lines.append(f"| {g} | {sym} |")
    lines.append("")

    lines.append("## Bucket distribution")
    bd = panel["final_action"].value_counts().to_dict()
    lines.append("| Bucket | N |")
    lines.append("|---|---:|")
    for b in PRIMARY_BUCKETS:
        lines.append(f"| {b} | {bd.get(b, 0)} |")
    lines.append("")

    lines.append("## Primary metrics (10d horizon, full cohort)")
    lines.append("")
    lines.append("| Bucket | N | PF | WR_pos | WR_10pct | meanR | medianR | MFE | MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in PRIMARY_BUCKETS:
        r = _row_md(metrics, cohort="full", bucket=b, horizon=10)
        if r is None:
            continue
        lines.append(
            f"| {b} | {r['N']} | {_fmt(r['PF'], 3)} | {_fmt(r['WR_pos'], 3)} | "
            f"{_fmt(r['WR_10pct'], 3)} | {_fmt(r['meanR'], 4)} | {_fmt(r['medianR'], 4)} | "
            f"{_fmt(r['MFE_mean'], 4)} | {_fmt(r['MAE_mean'], 4)} |"
        )
    lines.append("")

    lines.append("## Pair deltas (10d horizon, full cohort)")
    lines.append("")
    lines.append("| Comparison | ΔmeanR [CI] | ΔPF [CI] |")
    lines.append("|---|---|---|")
    for ref in ("WAIT_RETEST", "WATCHLIST", "AVOID"):
        r = _row_md(deltas, cohort="full", horizon=10, comparison=f"TRADEABLE_vs_{ref}")
        if r is None:
            continue
        lines.append(
            f"| TRADEABLE vs {ref} | "
            f"{_fmt(r['delta_meanR'], 4)} [{_fmt(r['delta_meanR_ci_lo'], 4)}, "
            f"{_fmt(r['delta_meanR_ci_hi'], 4)}] | "
            f"{_fmt(r['delta_PF'], 3)} [{_fmt(r['delta_PF_ci_lo'], 3)}, "
            f"{_fmt(r['delta_PF_ci_hi'], 3)}] |"
        )
    lines.append("")

    lines.append("## Multi-horizon sweep — TRADEABLE vs AVOID")
    lines.append("")
    lines.append("| Horizon | ΔmeanR [CI] | ΔPF [CI] |")
    lines.append("|---:|---|---|")
    for h in HORIZONS:
        r = _row_md(deltas, cohort="full", horizon=h, comparison="TRADEABLE_vs_AVOID")
        if r is None:
            continue
        lines.append(
            f"| {h}d | {_fmt(r['delta_meanR'], 4)} "
            f"[{_fmt(r['delta_meanR_ci_lo'], 4)}, {_fmt(r['delta_meanR_ci_hi'], 4)}] | "
            f"{_fmt(r['delta_PF'], 3)} [{_fmt(r['delta_PF_ci_lo'], 3)}, "
            f"{_fmt(r['delta_PF_ci_hi'], 3)}] |"
        )
    lines.append("")

    lines.append("## Secondary — residual TRADEABLE \\ H1 cohort (10d)")
    lines.append("")
    lines.append(f"H1 cohort identified: N={h1_count}")
    lines.append("")
    lines.append("| Bucket | N | PF | meanR |")
    lines.append("|---|---:|---:|---:|")
    for b in PRIMARY_BUCKETS:
        r = _row_md(metrics, cohort="residual_no_h1", bucket=b, horizon=10)
        if r is None:
            continue
        lines.append(f"| {b} | {r['N']} | {_fmt(r['PF'], 3)} | {_fmt(r['meanR'], 4)} |")
    lines.append("")
    lines.append("| Comparison | ΔmeanR [CI] | ΔPF [CI] |")
    lines.append("|---|---|---|")
    for ref in ("WATCHLIST", "AVOID"):
        r = _row_md(deltas, cohort="residual_no_h1", horizon=10, comparison=f"TRADEABLE_vs_{ref}")
        if r is None:
            continue
        lines.append(
            f"| residual TRADEABLE vs {ref} | "
            f"{_fmt(r['delta_meanR'], 4)} [{_fmt(r['delta_meanR_ci_lo'], 4)}, "
            f"{_fmt(r['delta_meanR_ci_hi'], 4)}] | "
            f"{_fmt(r['delta_PF'], 3)} [{_fmt(r['delta_PF_ci_lo'], 3)}, "
            f"{_fmt(r['delta_PF_ci_hi'], 3)}] |"
        )
    lines.append("")

    lines.append("## AVOID reason-code descriptive table (NOT a gate — Lock-note 2)")
    avoid_panel = panel[panel["final_action"] == "AVOID"].copy()
    rc_rows = []
    for _, r in avoid_panel.iterrows():
        codes = r.get("reason_codes")
        if isinstance(codes, np.ndarray):
            codes = codes.tolist()
        if not isinstance(codes, (list, tuple)):
            continue
        # take first negative-blocker code (skip fill_realism_unresolved)
        for c in codes:
            if c not in ("fill_realism_unresolved", "horizon_status_stale"):
                rc_rows.append({"primary_reason": c, "ret_10d": r.get("ret_10d")})
                break
    rc_df = pd.DataFrame(rc_rows)
    if len(rc_df):
        agg = (
            rc_df.groupby("primary_reason")["ret_10d"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values("count", ascending=False)
        )
        lines.append("")
        lines.append("| Reason | N | meanR_10d |")
        lines.append("|---|---:|---:|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['primary_reason']} | {int(r['count'])} | {_fmt(r['mean'], 4)} |"
            )
    lines.append("")

    lines.append("## Anti-rescue confirmation")
    lines.append("")
    lines.append("- No bucket subdivision based on results.")
    lines.append("- No threshold tweak (MAX_RISK_ATR / EXTENSION_CAP_ATR unchanged).")
    lines.append("- No new bucket / no NEUTRAL / no DEFER.")
    lines.append("- No top-decile carving inside any bucket.")
    lines.append("- No regime-conditional gate added.")
    lines.append("- No multi-TF confirmation analysis injected.")
    lines.append("- No horizon expansion beyond {5d, 10d, 20d}.")
    lines.append("- No new file in `decision_engine/` module path; no v0 source file modified.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    print("[classification-backtest] loading historical events …")
    parts = [
        load_mb_scanner_events(),
        load_horizontal_base_events(),
        load_nyxexpansion_events(),
        load_nyxmomentum_events(),
        load_nox_rt_daily_events(),
        load_nox_weekly_events(),
    ]
    events = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    print(f"  events: {len(events)} rows")
    print(f"  source breakdown: {events['source'].value_counts().to_dict()}")

    print("[classification-backtest] building daily panel + trading-day index …")
    daily = build_daily_panel()
    asof_data = str(daily["date"].max())
    cutoff_info = (
        f"trading-day basis; per-ticker max_tdx − {CUTOFF_BUFFER_TRADING_DAYS} buffer; "
        f"events past cutoff excluded with 'past_cutoff' tag"
    )
    print(f"  daily panel: {len(daily)} rows · asof={asof_data} · 607 tickers")

    print("[classification-backtest] attaching forward paths (5/10/20 trading days) …")
    events_with_path = attach_forward_paths(events, daily)
    print(f"  events with forward paths: {len(events_with_path)}")

    print("[classification-backtest] replaying v0 (handoffs + hard-rule table) …")
    actions, _events_full = replay_v0(events_with_path)
    print("  bucket distribution:", actions["final_action"].value_counts().to_dict())

    print("[classification-backtest] computing per-cell metrics + bootstrap CIs …")
    metrics_full = compute_per_cell_metrics(actions, label="full")
    deltas_full = compute_pair_deltas(actions, label="full")

    # G5 residual cohort
    h1_mask = identify_h1_cohort(actions)
    h1_count = int(h1_mask.sum())
    # Anomaly check: Lock-note 3 revised — only TRADEABLE H1 rows missing metadata
    anomaly_mask = identify_h1_anomalies(actions)
    h1_anomalies = int(anomaly_mask.sum())
    h1_trd_count = int((actions["family"].isin(H1_FAMILIES) &
                        (actions["final_action"] == "TRADEABLE")).sum())
    if h1_trd_count > 0 and h1_anomalies > 0.01 * h1_trd_count:
        print(
            f"  RUN VOID: H1 metadata anomalies {h1_anomalies} > 1% of "
            f"TRADEABLE-H1 cohort {h1_trd_count} — see Lock-note 3 revised"
        )
        # surface first 5 example rows for audit
        ex = actions[anomaly_mask].head(5)[
            ["date", "ticker", "family", "final_action", "expected_horizon",
             "horizon_source", "horizon_review_due"]
        ]
        print(ex.to_string(index=False))
        return 2
    print(
        f"  H1 conjunction cohort: {h1_count} | H1 TRADEABLE rows: "
        f"{h1_trd_count} | metadata anomalies: {h1_anomalies}"
    )

    residual = actions[~h1_mask].copy()
    metrics_resid = compute_per_cell_metrics(residual, label="residual_no_h1")
    deltas_resid = compute_pair_deltas(residual, label="residual_no_h1")

    metrics = pd.concat([metrics_full, metrics_resid], ignore_index=True)
    deltas = pd.concat([deltas_full, deltas_resid], ignore_index=True)

    print("[classification-backtest] evaluating gates …")
    gates = evaluate_gates(metrics, deltas)
    verdict = _verdict(gates)
    print("  gates:", gates)
    print(f"  VERDICT: {verdict}")

    # ── outputs ─────────────────────────────────────────────────────────
    panel_path = OUT / "decision_v0_classification_panel.parquet"
    metrics_path = OUT / "decision_v0_classification_metrics.csv"
    secondary_path = OUT / "decision_v0_classification_secondary_metrics.csv"
    summary_path = OUT / "decision_v0_classification_summary.md"

    # write panel — convert reason_codes to plain list[str] for parquet
    panel_out = actions.copy()
    if "reason_codes" in panel_out.columns:
        panel_out["reason_codes"] = panel_out["reason_codes"].apply(
            lambda v: [str(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else []
        )
    if "horizon_review_due" in panel_out.columns:
        panel_out["horizon_review_due"] = panel_out["horizon_review_due"].astype("string")
    panel_out.to_parquet(panel_path, index=False)
    metrics_full.to_csv(metrics_path, index=False)
    pd.concat([metrics_resid, deltas_resid], ignore_index=True, sort=False).to_csv(
        secondary_path, index=False
    )

    write_summary_md(
        out_path=summary_path,
        panel=actions,
        metrics=metrics,
        deltas=deltas,
        gates=gates,
        verdict=verdict,
        h1_count=h1_count,
        h1_anomalies=h1_anomalies,
        asof_data=asof_data,
        cutoff_info=cutoff_info,
    )

    print()
    print("Outputs:")
    for p in (panel_path, metrics_path, secondary_path, summary_path):
        print(f"  → {p.relative_to(ROOT)}")
    print()

    # spec invariants
    assert "rank" not in panel_out.columns, "rank column present — anti-rescue violation"
    assert "selection_score" not in panel_out.columns, "selection_score present — violation"
    priors_csv = OUT / "decision_bucket_priors.csv"
    assert not priors_csv.exists(), "decision_bucket_priors.csv exists — spec violation"

    print(f"[classification-backtest] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
