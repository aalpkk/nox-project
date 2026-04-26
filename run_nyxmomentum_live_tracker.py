"""
run_nyxmomentum_live_tracker.py — parallel live tracker for V5 (default)
and M0 (benchmark), reading from the frozen v1.1 pipeline artifacts.

Records per-rebalance picks + realized returns for both profiles and
compares against the backtest envelope. Enforces append-only discipline
so historical rows are never silently rewritten.

## Usage
  python3 run_nyxmomentum_live_tracker.py              # append new rebalances
  python3 run_nyxmomentum_live_tracker.py --reseed     # rebuild from scratch
                                                         (use only on a freeze bump)
  python3 run_nyxmomentum_live_tracker.py --bps 60     # cost override

## Inputs (read-only, from frozen pipeline)
  output/nyxmomentum/reports/step5_selection_V5_M0M1_ensemble_damp.parquet  — V5 picks
  output/nyxmomentum/reports/step5_selection_V0_M0_baseline.parquet         — M0 picks
  output/nyxmomentum/reports/step1_labels.parquet                           — realized next-month returns
  output/nyxmomentum/artifacts/frozen_v1_1.json                             — envelope, hashes, version

## Outputs (output/nyxmomentum/live/)
  <YYYY-MM-DD>_v5_picks.csv       V5 basket snapshot per rebalance (ticker, rank, score, weight)
  <YYYY-MM-DD>_m0_picks.csv       M0 basket snapshot per rebalance
  <YYYY-MM-DD>_v5_picks.html      V5 basket rendered in NOX aesthetic
  <YYYY-MM-DD>_m0_picks.html      M0 basket rendered in NOX aesthetic
  tracker.csv                     append-only per-rebalance log (both profiles)
  tracker.html                    dashboard: envelope diff + per-rebalance table
  tracker_envelope_diff.csv       realized vs frozen-envelope aggregate
  run_meta.json                   frozen-version pointer, timestamp, row counts

## Append-only semantics
  - Rebalance dates already in tracker.csv keep their stored values.
  - The ONE allowed update: status='pending' → 'realized' when next-month
    return becomes observable.
  - A recomputed value that conflicts with a stored realized row → WARN;
    the stored row wins. Do not paper over — investigate.
  - Pick CSVs are write-once per (date, profile). Mismatch on re-run → WARN,
    no overwrite.
  - Selection is the frozen pipeline's output; this script does not
    re-rank, re-score, or reinvent selection logic.

## First real post-freeze rebalance
  1. Run full pipeline (step0 → step5) with the new rebalance date in-range.
  2. Run this script — the new date is appended (status='pending' until
     label is observable, flipped to 'realized' at the next cycle).
  No code changes needed.

## Review metrics (monthly)
  - tracker.csv: net_return_60bps, rolling_maxdd_net60, cagr_so_far_net60,
    sharpe_net60_so_far — month-over-month.
  - tracker_envelope_diff.csv: realized vs frozen envelope deltas
    (nCAGR60, nShp60, MaxDD60, turnover).
  - Kill-switch thresholds / review cadence: DEPLOYMENT_SPEC.md §7/§8.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
LIVE_DIR    = os.path.join(CONFIG.paths.root, "live")
FROZEN_META = os.path.join(CONFIG.paths.root, "artifacts", "frozen_v1_1.json")

M0_SEL = f"{REPORTS_DIR}/step5_selection_V0_M0_baseline.parquet"
V5_SEL = f"{REPORTS_DIR}/step5_selection_V5_M0M1_ensemble_damp.parquet"
LABELS = f"{REPORTS_DIR}/step1_labels.parquet"

TRACKER_COLS = [
    "rebalance_date", "profile", "status", "n_names",
    "portfolio_return", "turnover_fraction", "names_changed",
    "net_return_60bps",
    "cum_equity_net60", "rolling_maxdd_net60",
    "cagr_so_far_net60", "mean_monthly_net60_so_far",
    "sharpe_net60_so_far",
]


# ── Frozen meta load ─────────────────────────────────────────────────────────

def _load_frozen(path: str) -> tuple[dict, str]:
    if not os.path.exists(path):
        print(f"  [WARN] frozen meta not found at {path}", file=sys.stderr)
        return {}, "unknown"
    with open(path, "r") as fh:
        meta = json.load(fh)
    # v1.1 key: frozen_metrics_at_60bps_v1_1; v1.0 key: frozen_metrics_at_60bps
    env = meta.get("frozen_metrics_at_60bps_v1_1") or meta.get("frozen_metrics_at_60bps") or {}
    version = meta.get("version", "unknown")
    return env, version


# ── Pick emission per profile ────────────────────────────────────────────────

def _profile_picks_by_date(selection: pd.DataFrame, score_col: str) -> dict:
    sel = selection.loc[selection["selected"].astype(bool)].copy()
    if "rank" not in sel.columns:
        sel["rank"] = (sel.groupby("rebalance_date")[score_col]
                         .rank(method="first", ascending=False).astype(int))
    out: dict = {}
    for d, g in sel.groupby("rebalance_date", sort=True):
        df = (g.sort_values("rank")[["ticker", "rank", score_col]]
                .rename(columns={score_col: "score"})
                .assign(weight=lambda x: 1.0 / len(x))
                .reset_index(drop=True))
        out[pd.Timestamp(d)] = df
    return out


def _write_pick_files_append_only(picks_by_date: dict, profile_lo: str,
                                   live_dir: str) -> tuple[int, int, int]:
    """Write per-date pick CSVs write-once. Returns (new, skipped_match, skipped_mismatch)."""
    new = same = mism = 0
    for d, df in picks_by_date.items():
        p = os.path.join(live_dir, f"{d.date().isoformat()}_{profile_lo}_picks.csv")
        if os.path.exists(p):
            try:
                prev = pd.read_csv(p)
                if (list(prev["ticker"]) == list(df["ticker"])
                        and np.allclose(prev["rank"].values, df["rank"].values)):
                    same += 1
                    continue
                mism += 1
                print(f"  [WARN] pick file differs from recomputed, preserving stored: {p}",
                      file=sys.stderr)
                continue
            except Exception:
                mism += 1
                print(f"  [WARN] could not read existing pick file, preserving: {p}",
                      file=sys.stderr)
                continue
        df.to_csv(p, index=False)
        new += 1
    return new, same, mism


def _profile_eligible_by_date(selection: pd.DataFrame, score_col: str,
                                top_cut: int = 30, n_enter: int = 20) -> dict:
    """All top-<top_cut> rows per date with dampener-status labels.

    Dampener (DampenerConfig(n_enter=20, n_exit=30)): keepers = prev ∩ rank≤30;
    new fill = rank≤20 that aren't keepers, until 20 slots full. So "top-20
    candidates locked out" are ranks ≤20 that are neither prev nor selected.

    status values:
      - core_keeper  : selected AND in prev basket, rank ≤ n_enter
      - sticky_keep  : selected AND in prev basket, rank in (n_enter, top_cut]
                       (these are the "why 21/22/26" sticky names)
      - new_entrant  : selected AND NOT in prev basket (rank ≤ n_enter)
      - crowded_out  : NOT selected, rank ≤ n_enter, NOT in prev basket
                       (would have been a top-20 new entrant but sticky
                        keepers filled the slot first)
      - dropped      : prev basket, NOT selected (rank > n_exit)
      - bystander    : other (rank > n_enter, NOT prev, NOT selected)
    """
    out = {}
    for d, g in selection.groupby("rebalance_date", sort=True):
        g = g.sort_values("rank")
        top = g.loc[g["rank"] <= top_cut,
                     ["ticker", "rank", score_col, "selected", "prev_selected"]].copy()
        top = top.rename(columns={score_col: "score"}).reset_index(drop=True)

        def _label(r):
            sel, prev = bool(r["selected"]), bool(r["prev_selected"])
            rk = int(r["rank"])
            if sel and prev:
                return "core_keeper" if rk <= n_enter else "sticky_keep"
            if sel and not prev:
                return "new_entrant"
            if (not sel) and rk <= n_enter and not prev:
                return "crowded_out"
            if prev and not sel:
                return "dropped"
            return "bystander"

        top["status"] = top.apply(_label, axis=1)
        out[pd.Timestamp(d)] = top
    return out


def _write_eligible_files_append_only(eligible_by_date: dict,
                                        picks_by_date: dict,
                                        profile_lo: str,
                                        live_dir: str) -> tuple[int, int, int]:
    """Write per-date eligible CSVs (top-30 with dampener labels), write-once.

    Guard: only write when parquet's selected-set matches the stored pick
    CSV's ticker list. On drift, skip silently — otherwise the eligible list
    would describe a different selection event than the preserved picks.
    """
    new = skipped_drift = skipped_exist = 0
    for d, el_df in eligible_by_date.items():
        iso = d.date().isoformat()
        elig_p = os.path.join(live_dir, f"{iso}_{profile_lo}_eligible.csv")
        pick_p = os.path.join(live_dir, f"{iso}_{profile_lo}_picks.csv")
        if os.path.exists(elig_p):
            skipped_exist += 1
            continue
        if not os.path.exists(pick_p):
            continue
        try:
            stored_picks = set(pd.read_csv(pick_p)["ticker"])
        except Exception:
            skipped_drift += 1
            continue
        parquet_picks = set(el_df.loc[el_df["selected"].astype(bool), "ticker"])
        if stored_picks != parquet_picks:
            skipped_drift += 1
            continue
        el_df.to_csv(elig_p, index=False)
        new += 1
    return new, skipped_exist, skipped_drift


# ── Tracker row construction ─────────────────────────────────────────────────

def _basket_row(d: pd.Timestamp, profile: str,
                 basket: set, prev_basket: set | None,
                 labels_by_date: dict, bps: int) -> dict:
    n = len(basket)
    if n == 0:
        return {"rebalance_date": d, "profile": profile, "status": "empty",
                "n_names": 0, "portfolio_return": np.nan,
                "turnover_fraction": np.nan, "names_changed": np.nan,
                "net_return_60bps": np.nan}

    has_label = d in labels_by_date
    if not has_label:
        # First rebalance after freeze — picks recorded, return not yet observable.
        if prev_basket is None:
            turn, changed = np.nan, np.nan
        else:
            changed = len(basket - prev_basket)
            turn = changed / max(n, 1)
        return {"rebalance_date": d, "profile": profile, "status": "pending",
                "n_names": n, "portfolio_return": np.nan,
                "turnover_fraction": turn,
                "names_changed": int(changed) if isinstance(changed, int) else np.nan,
                "net_return_60bps": np.nan}

    rets = labels_by_date[d].reindex(list(basket))
    r_gross = float(rets.mean())
    if prev_basket is None:
        turn, changed, cost = np.nan, np.nan, 0.0
    else:
        changed = len(basket - prev_basket)
        turn = changed / max(n, 1)
        cost = turn * (bps / 10000.0)
    r_net = r_gross - cost
    return {"rebalance_date": d, "profile": profile, "status": "realized",
            "n_names": n, "portfolio_return": r_gross,
            "turnover_fraction": turn if not (isinstance(turn, float) and np.isnan(turn)) else np.nan,
            "names_changed": int(changed) if isinstance(changed, int) else np.nan,
            "net_return_60bps": r_net}


def _build_profile_tracker(picks_by_date: dict, profile: str,
                            labels_by_date: dict, bps: int) -> pd.DataFrame:
    rows = []
    prev: set | None = None
    for d in sorted(picks_by_date.keys()):
        basket = set(picks_by_date[d]["ticker"])
        rows.append(_basket_row(d, profile, basket, prev, labels_by_date, bps))
        prev = basket
    return pd.DataFrame(rows)


def _rolling_envelope(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row rolling metrics (equity, DD, CAGR-so-far, Sharpe-so-far).

    Pending rows contribute 0 to return (neutral) so rolling columns stay
    defined; they will be recomputed once realized.
    """
    r_net = df["net_return_60bps"].fillna(0.0).astype(float)
    eq = (1.0 + r_net).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    n = np.arange(1, len(df) + 1)
    cagr_so_far = eq ** (12.0 / np.maximum(n, 1)) - 1.0
    df = df.copy()
    df["cum_equity_net60"] = eq.values
    df["rolling_maxdd_net60"] = dd.values
    df["cagr_so_far_net60"] = cagr_so_far.values
    df["mean_monthly_net60_so_far"] = r_net.expanding().mean().values
    std_so_far = r_net.expanding().std(ddof=0).replace(0, np.nan).values
    df["sharpe_net60_so_far"] = (df["mean_monthly_net60_so_far"].values
                                   / std_so_far) * np.sqrt(12.0)
    return df


# ── Append-only merge ────────────────────────────────────────────────────────

def _merge_append_only(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Keep existing rows. Allow 'pending' → 'realized' transitions. Warn on
    realized-to-different conflict; preserve stored row.
    """
    if existing is None or existing.empty:
        return incoming.copy()

    existing = existing.copy()
    existing["rebalance_date"] = pd.to_datetime(existing["rebalance_date"])
    incoming = incoming.copy()
    incoming["rebalance_date"] = pd.to_datetime(incoming["rebalance_date"])

    key = ["rebalance_date", "profile"]
    existing_idx = existing.set_index(key)
    incoming_idx = incoming.set_index(key)

    merged_rows = []
    promoted = conflicts = 0
    for k, inc_row in incoming_idx.iterrows():
        if k in existing_idx.index:
            ex_row = existing_idx.loc[k]
            ex_status = ex_row.get("status", "realized")
            # Allow promotion from pending → realized
            if ex_status == "pending" and inc_row.get("status") == "realized":
                merged_rows.append(inc_row.to_dict() | {"rebalance_date": k[0], "profile": k[1]})
                promoted += 1
                continue
            # Conflict detection on realized rows
            if ex_status == "realized" and inc_row.get("status") == "realized":
                def _neq(a, b, tol=1e-9):
                    if pd.isna(a) and pd.isna(b):
                        return False
                    try:
                        return abs(float(a) - float(b)) > tol
                    except (TypeError, ValueError):
                        return a != b
                for col in ("portfolio_return", "net_return_60bps", "turnover_fraction", "n_names"):
                    if _neq(ex_row.get(col), inc_row.get(col)):
                        conflicts += 1
                        print(f"  [WARN] conflict at {k[0].date()} / {k[1]} col={col}: "
                              f"stored={ex_row.get(col)} recomputed={inc_row.get(col)} "
                              f"(keeping stored)", file=sys.stderr)
                        break
            # Keep stored row verbatim
            merged_rows.append(ex_row.to_dict() | {"rebalance_date": k[0], "profile": k[1]})
        else:
            merged_rows.append(inc_row.to_dict() | {"rebalance_date": k[0], "profile": k[1]})

    # Keep any existing keys not in incoming (defensive — shouldn't normally happen)
    for k, ex_row in existing_idx.iterrows():
        if k not in incoming_idx.index:
            merged_rows.append(ex_row.to_dict() | {"rebalance_date": k[0], "profile": k[1]})

    df = pd.DataFrame(merged_rows)
    # Ensure column order
    df = df.reindex(columns=[c for c in TRACKER_COLS if c in df.columns]
                             + [c for c in df.columns if c not in TRACKER_COLS])
    df = df.sort_values(["rebalance_date", "profile"]).reset_index(drop=True)
    if promoted:
        print(f"  {promoted} row(s) promoted pending → realized")
    if conflicts:
        print(f"  {conflicts} conflict(s) — stored rows preserved (see warnings)")
    return df


def _recompute_rolling_for_tracker(tracker: pd.DataFrame) -> pd.DataFrame:
    """Re-derive rolling columns per profile from current (merged) returns."""
    out = []
    for profile, g in tracker.groupby("profile", sort=True):
        g = g.sort_values("rebalance_date").reset_index(drop=True)
        g = _rolling_envelope(g)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["rebalance_date", "profile"]).reset_index(drop=True)


# ── Envelope diff (aggregate, realized-only) ─────────────────────────────────

def _envelope_diff(tracker: pd.DataFrame, envelope: dict) -> pd.DataFrame:
    rows = []
    for profile, key in [("V5", "always_V5"), ("M0", "always_M0")]:
        g = tracker.loc[(tracker["profile"] == profile)
                        & (tracker["status"] == "realized")]
        if g.empty:
            continue
        r_net = g["net_return_60bps"].dropna().astype(float)
        n = len(r_net)
        if n == 0:
            continue
        eq = (1.0 + r_net).cumprod()
        peak = eq.cummax()
        dd = float((eq / peak - 1.0).min())
        cagr = float(eq.iloc[-1] ** (12.0 / n) - 1.0)
        mu, sd = float(r_net.mean()), float(r_net.std(ddof=0))
        shp = float(mu / sd * np.sqrt(12.0)) if sd > 0 else np.nan
        tm = (float(g["turnover_fraction"].dropna().mean())
              if g["turnover_fraction"].notna().any() else np.nan)
        exp = envelope.get(key, {}) or {}
        def _delta(real, exp_key):
            ev = exp.get(exp_key)
            return (real - float(ev)) if ev is not None else np.nan
        rows.append({
            "profile": profile,
            "n_realized_rebalances": int(n),
            "n_pending_rebalances": int((tracker["profile"] == profile).sum() - n),
            "realized_nCAGR_60":  cagr,
            "expected_nCAGR_60":  exp.get("nCAGR", np.nan),
            "nCAGR_delta":        _delta(cagr, "nCAGR"),
            "realized_nShp_60":   shp,
            "expected_nShp_60":   exp.get("nShp", np.nan),
            "nShp_delta":         _delta(shp, "nShp"),
            "realized_MaxDD_60":  dd,
            "expected_MaxDD_60":  exp.get("DD", np.nan),
            "MaxDD_delta":        _delta(dd, "DD"),
            "realized_turnover":  tm,
            "expected_turnover":  exp.get("turnover", np.nan),
            "turnover_delta":     _delta(tm, "turnover"),
        })
    return pd.DataFrame(rows)


# ── Print summary ────────────────────────────────────────────────────────────

def _fmt_pct(v, w: int = 8) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "   —  ".rjust(w)
    return f"{v:+.1%}".rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_summary(tracker: pd.DataFrame, diff: pd.DataFrame,
                    live_dir: str, version: str, t0: float) -> None:
    n_v5 = int((tracker["profile"] == "V5").sum())
    n_m0 = int((tracker["profile"] == "M0").sum())
    n_pending = int((tracker["status"] == "pending").sum())
    last_date = tracker["rebalance_date"].max()
    last_date_s = pd.Timestamp(last_date).date().isoformat() if pd.notna(last_date) else "—"
    print()
    print("══ nyxmomentum Parallel Live Tracker ══")
    print(f"  frozen version: {version}")
    print(f"  last rebalance: {last_date_s}")
    print(f"  rows: V5={n_v5}  M0={n_m0}  (pending={n_pending})")
    print(f"  output dir:     {live_dir}")
    print()
    print("  REALIZED-vs-FROZEN-ENVELOPE:")
    print(f"  {'profile':<8}{'N':>4}{'nCAGR60':>10}{'exp':>10}{'Δ':>8}"
          f"{'nShp60':>8}{'exp':>7}{'Δ':>7}"
          f"{'DD60':>8}{'exp':>8}{'Δ':>7}"
          f"{'turn':>7}{'exp':>7}{'Δ':>7}")
    for _, r in diff.iterrows():
        print(f"  {r['profile']:<8}{int(r['n_realized_rebalances']):>4}"
              f"{_fmt_pct(r['realized_nCAGR_60'], 10)}{_fmt_pct(r['expected_nCAGR_60'], 10)}{_fmt_pct(r['nCAGR_delta'], 8)}"
              f"{_fmt_num(r['realized_nShp_60'], 8)}{_fmt_num(r['expected_nShp_60'], 7)}{_fmt_num(r['nShp_delta'], 7)}"
              f"{_fmt_pct(r['realized_MaxDD_60'], 8)}{_fmt_pct(r['expected_MaxDD_60'], 8)}{_fmt_pct(r['MaxDD_delta'], 7)}"
              f"{_fmt_pct(r['realized_turnover'], 7)}{_fmt_pct(r['expected_turnover'], 7)}{_fmt_pct(r['turnover_delta'], 7)}")
    print()
    print(f"  tracker:    {os.path.join(live_dir, 'tracker.csv')}")
    print(f"  dashboard:  {os.path.join(live_dir, 'tracker.html')}")
    print(f"  diff:       {os.path.join(live_dir, 'tracker_envelope_diff.csv')}")
    print(f"  picks:      {live_dir}/<YYYY-MM-DD>_{{v5,m0}}_picks.{{csv,html}}")
    print(f"  (elapsed {time.time() - t0:.1f}s)")


# ── HTML rendering (NOX aesthetic) ───────────────────────────────────────────

def _nox_css() -> str:
    try:
        from core.reports import _NOX_CSS  # type: ignore
        return _NOX_CSS
    except Exception:
        # Minimal fallback so the HTML is still usable off-repo
        return (
            "body{background:#060709;color:#e8e4dc;font-family:"
            "'Inter',system-ui,sans-serif;margin:0}"
            "table{width:100%;border-collapse:collapse;font-family:"
            "'JetBrains Mono',monospace;font-size:.78rem}"
            "th{color:#8a8580;text-transform:uppercase;font-size:.68rem;"
            "text-align:left;padding:10px 8px;border-bottom:1px solid #1e1e23}"
            "td{padding:8px;border-bottom:1px solid rgba(39,39,42,.5)}"
        )


_PROFILE_COLOR = {"V5": "var(--nox-gold)", "M0": "var(--nox-blue)"}


def _fmt_pct_html(v, digits: int = 2) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return '<span class="mono muted">—</span>'
    sign = "pos" if v >= 0 else "neg"
    return f'<span class="mono rs-{sign}">{v*100:+.{digits}f}%</span>'


def _fmt_num_html(v, digits: int = 2) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return '<span class="mono muted">—</span>'
    sign = "pos" if v >= 0 else "neg"
    return f'<span class="mono rs-{sign}">{v:+.{digits}f}</span>'


def _fmt_plain(v, digits: int = 2) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return '<span class="mono muted">—</span>'
    return f'<span class="mono">{v:.{digits}f}</span>'


def _status_badge(s: str) -> str:
    if s == "realized":
        return '<span class="st-badge st-realized">realized</span>'
    if s == "pending":
        return '<span class="st-badge st-pending">pending</span>'
    return f'<span class="st-badge st-empty">{s}</span>'


def _render_tracker_html(tracker: pd.DataFrame, diff: pd.DataFrame,
                           version: str, live_dir: str,
                           oos_dates: list,
                           pick_date_index: dict) -> str:
    """Render parallel-tracker dashboard in briefing aesthetic."""
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    n_v5 = int((tracker["profile"] == "V5").sum())
    n_m0 = int((tracker["profile"] == "M0").sum())
    n_pending = int((tracker["status"] == "pending").sum())
    n_realized = int((tracker["status"] == "realized").sum())
    last_date = tracker["rebalance_date"].max()
    last_date_s = pd.Timestamp(last_date).date().isoformat() if pd.notna(last_date) else "—"

    # ── Envelope diff cards (action-card style) ──
    def _delta_cls(v) -> str:
        if v is None or pd.isna(v) or not np.isfinite(v):
            return ""
        return "rs-pos" if v >= 0 else "rs-neg"

    diff_cards = []
    for _, r in diff.iterrows():
        p = r["profile"]
        tier_cls = "tier1" if p == "V5" else "tier2b"
        tag_cls  = "t1"    if p == "V5" else "t2b"
        role = "default live" if p == "V5" else "benchmark"
        diff_cards.append(
            f'<div class="action-card env-card {tier_cls}">'
              '<div class="card-head">'
                f'<span class="ticker">{p}</span>'
                f'<span class="tier-tag {tag_cls}">{role}</span>'
                f'<span class="env-n">N={int(r["n_realized_rebalances"])} realized · '
                f'{int(r["n_pending_rebalances"])} pending</span>'
              '</div>'
              '<div class="env-grid">'
                '<div class="env-cell"><div class="env-lbl">nCAGR<sub>60</sub></div>'
                  f'<div class="env-real">{_fmt_pct_html(r["realized_nCAGR_60"], 1)}</div>'
                  f'<div class="env-exp">exp {_fmt_pct_html(r["expected_nCAGR_60"], 1)} · '
                  f'Δ {_fmt_pct_html(r["nCAGR_delta"], 1)}</div></div>'
                '<div class="env-cell"><div class="env-lbl">Sharpe<sub>60</sub></div>'
                  f'<div class="env-real">{_fmt_num_html(r["realized_nShp_60"], 2)}</div>'
                  f'<div class="env-exp">exp {_fmt_num_html(r["expected_nShp_60"], 2)} · '
                  f'Δ {_fmt_num_html(r["nShp_delta"], 2)}</div></div>'
                '<div class="env-cell"><div class="env-lbl">MaxDD<sub>60</sub></div>'
                  f'<div class="env-real">{_fmt_pct_html(r["realized_MaxDD_60"], 1)}</div>'
                  f'<div class="env-exp">exp {_fmt_pct_html(r["expected_MaxDD_60"], 1)} · '
                  f'Δ {_fmt_pct_html(r["MaxDD_delta"], 1)}</div></div>'
                '<div class="env-cell"><div class="env-lbl">Turnover</div>'
                  f'<div class="env-real">{_fmt_pct_html(r["realized_turnover"], 1)}</div>'
                  f'<div class="env-exp">exp {_fmt_pct_html(r["expected_turnover"], 1)} · '
                  f'Δ {_fmt_pct_html(r["turnover_delta"], 1)}</div></div>'
              '</div>'
            '</div>'
        )

    # ── Per-rebalance log rows ──
    body_rows = []
    t = tracker.copy().sort_values(["rebalance_date", "profile"],
                                     ascending=[False, True]).reset_index(drop=True)
    for _, row in t.iterrows():
        d = pd.Timestamp(row["rebalance_date"]).date().isoformat()
        p = row["profile"]
        color = _PROFILE_COLOR.get(p, "var(--text-primary)")
        pick_href = f"{d}_{p.lower()}_picks.html"
        pick_link = (f'<a class="tv-link" href="{pick_href}">{p}</a>'
                     if pick_date_index.get((d, p.lower()), False)
                     else f'<span style="color:{color};font-weight:600">{p}</span>')
        st = row.get("status", "—")
        st_cls = ("st-new" if st == "realized"
                  else "st-keep" if st == "pending"
                  else "bdg-dim")
        body_rows.append(f"""
        <tr>
          <td class="mono">{d}</td>
          <td style="color:{color}">{pick_link}</td>
          <td><span class="badge {st_cls}">{st}</span></td>
          <td class="mono" style="text-align:right">{int(row['n_names']) if not pd.isna(row.get('n_names')) else '—'}</td>
          <td style="text-align:right">{_fmt_pct_html(row.get('portfolio_return'), 2)}</td>
          <td style="text-align:right">{_fmt_pct_html(row.get('net_return_60bps'), 2)}</td>
          <td style="text-align:right">{_fmt_pct_html(row.get('turnover_fraction'), 1)}</td>
          <td style="text-align:right">{_fmt_pct_html(row.get('rolling_maxdd_net60'), 1)}</td>
          <td style="text-align:right">{_fmt_pct_html(row.get('cagr_so_far_net60'), 1)}</td>
          <td style="text-align:right">{_fmt_num_html(row.get('sharpe_net60_so_far'), 2)}</td>
          <td style="text-align:right">{_fmt_num_html(row.get('cum_equity_net60'), 3)}</td>
        </tr>""")

    oos_range = ""
    if oos_dates:
        oos_range = (f"{oos_dates[0].date().isoformat()} → "
                      f"{oos_dates[-1].date().isoformat()}")

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nyxmomentum — Live Tracker · {last_date_s}</title>
<style>
{_nox_css()}

.mono {{ font-family: var(--font-mono); }}
.muted {{ color: var(--text-muted); }}
.rs-pos {{ color: var(--nox-green); }}
.rs-neg {{ color: var(--nox-red); }}

/* ── briefing-style container ── */
.briefing-container {{
  position: relative; z-index: 1;
  max-width: 1100px; margin: 0 auto;
  padding: 0 1.5rem 2rem;
}}

/* ── sticky status bar ── */
.nox-status-bar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(6,6,8,0.55);
  backdrop-filter: blur(24px) saturate(1.3);
  -webkit-backdrop-filter: blur(24px) saturate(1.3);
  border-bottom: 1px solid rgba(201,169,110,0.08);
  padding: 0.6rem 1.5rem;
  margin: 0 -1.5rem 1.5rem;
  display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
}}
.nox-status-bar .logo {{
  display: inline-flex; align-items: baseline;
  gap: 0.15rem; white-space: nowrap;
}}
.nox-status-bar .logo .nox-text {{
  font-family: var(--font-brand); font-size: 4.2rem;
  color: #fff; letter-spacing: 0.06em; line-height: 0.85;
}}
.nox-status-bar .logo .brief-text {{
  font-family: var(--font-handwrite); font-size: 1.1rem;
  color: var(--nox-gold); margin-left: 0.2rem;
  position: relative; top: -0.15rem;
}}
.nox-status-bar .mpill {{
  font-family: var(--font-mono); font-size: 0.7rem;
  padding: 0.15rem 0.55rem; border-radius: 0.75rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); white-space: nowrap;
}}
.nox-status-bar .mpill b {{ font-weight: 700; color: var(--nox-gold); }}
.nox-status-bar .meta-right {{
  font-size: 0.7rem; color: var(--text-muted);
  font-family: var(--font-mono); white-space: nowrap; margin-left: auto;
}}

/* ── action cards (briefing-style) ── */
.action-card {{
  background: rgba(199,189,190,0.08);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: none; border-radius: 18px;
  padding: 0.95rem 1.15rem;
  border-left: 3px solid transparent;
  transition: all 0.25s ease;
  animation: cardFadeIn 0.3s ease-out both;
  position: relative; overflow: hidden;
}}
.action-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(184,149,110,0.15), transparent);
  opacity: 0; transition: opacity 0.3s;
}}
.action-card:hover {{
  background: rgba(199,189,190,0.12);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(184,149,110,0.03);
  transform: translateY(-1px);
}}
.action-card:hover::before {{ opacity: 1; }}
.action-card.tier1  {{ border-left: 3px solid rgba(201,169,110,0.45); }}
.action-card.tier2a {{ border-left: 3px solid rgba(184,149,110,0.25); }}
.action-card.tier2b {{ border-left: 3px solid rgba(138,122,158,0.30); }}
@keyframes cardFadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.action-card .card-head {{
  display: flex; align-items: center; gap: 0.55rem;
  margin-bottom: 0.65rem;
}}
.action-card .card-head .ticker {{
  font-family: 'Inter', sans-serif; font-weight: 900;
  font-size: 1.15rem; color: #fff;
}}
.action-card .card-head .tier-tag {{
  font-size: 0.6rem; font-family: var(--font-mono);
  padding: 0.1rem 0.4rem; border-radius: 0.25rem;
  font-weight: 600; text-transform: uppercase;
}}
.action-card .card-head .tier-tag.t1  {{ background: rgba(201,169,110,0.12); color: #c9a96e; }}
.action-card .card-head .tier-tag.t2a {{ background: rgba(168,135,106,0.12); color: #a8876a; }}
.action-card .card-head .tier-tag.t2b {{ background: rgba(138,122,158,0.12); color: #8a7a9e; }}
.action-card .card-head .env-n {{
  margin-left: auto; font-family: var(--font-mono);
  font-size: 0.7rem; color: var(--text-muted);
}}

/* envelope-card metric grid (4 cells per profile card) */
.env-wrap {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
  gap: 0.85rem; margin-bottom: 1.5rem;
}}
.env-card .env-grid {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.55rem;
}}
.env-card .env-cell {{
  padding: 0.5rem 0.6rem;
  background: rgba(6,7,9,0.45);
  border-radius: 12px;
}}
.env-card .env-lbl {{
  font-family: var(--font-mono); font-size: 0.62rem;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 0.25rem;
}}
.env-card .env-real {{
  font-size: 1rem; font-weight: 600; font-family: var(--font-mono);
}}
.env-card .env-exp {{
  font-size: 0.62rem; color: var(--text-muted);
  margin-top: 0.15rem; font-family: var(--font-mono);
}}

/* badges (status pills) */
.badge {{
  font-family: var(--font-mono); font-size: 0.62rem;
  padding: 0.12rem 0.42rem; border-radius: 0.4rem;
  font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; white-space: nowrap;
  border: 1px solid transparent; display: inline-block;
}}
.badge.bdg-dim   {{ background: rgba(138,133,128,0.10); color: var(--text-muted); }}
.badge.st-keep   {{ background: rgba(201,169,110,0.14); color: #c9a96e;
                    border-color: rgba(201,169,110,0.25); }}
.badge.st-new    {{ background: rgba(122,158,122,0.16); color: var(--nox-green);
                    border-color: rgba(122,158,122,0.30); }}

/* tier-group-label: section heading */
.tier-group-label {{
  font-family: var(--font-display); font-weight: 600;
  font-size: 0.92rem; color: var(--text-secondary);
  margin: 1.4rem 0 0.6rem;
  display: flex; align-items: center; gap: 0.4rem;
}}
.tier-group-label:first-of-type {{ margin-top: 0.4rem; }}
.tier-group-label .cnt {{
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--text-muted);
}}

/* per-rebalance log table inside backdrop-blur container */
.log-shell {{
  background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: none; border-radius: 18px;
  padding: 0.4rem 0.4rem 0.6rem;
  margin-bottom: 1.5rem;
  overflow-x: auto;
}}
.log-shell table {{
  width: 100%; border-collapse: collapse;
  font-family: var(--font-mono); font-size: 0.74rem;
}}
.log-shell thead th {{
  position: sticky; top: 0;
  background: rgba(6,7,9,0.85);
  font-family: var(--font-display); font-weight: 600;
  font-size: 0.7rem; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.06em;
  text-align: left; padding: 0.6rem 0.7rem;
  border-bottom: 1px solid var(--border-subtle);
}}
.log-shell tbody td {{
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid rgba(39,39,42,0.4);
  white-space: nowrap;
}}
.log-shell tbody tr:hover {{
  background: rgba(199,189,190,0.04);
}}
.log-shell tbody tr:last-child td {{ border-bottom: none; }}

.footer-meta {{
  font-family: var(--font-mono); font-size: 0.68rem;
  color: var(--text-muted); margin-top: 1.5rem;
  padding-top: 0.8rem; border-top: 1px solid var(--border-subtle);
  text-align: center;
}}
.footer-meta b {{ color: var(--nox-gold); font-weight: 700; }}
</style>
</head><body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="briefing-container">
  <div class="nox-status-bar">
    <span class="logo">
      <span class="nox-text">NYX</span><span class="brief-text">momentum</span>
    </span>
    <span class="mpill">frozen <b>{version}</b></span>
    <span class="mpill">last rebalance <b>{last_date_s}</b></span>
    <span class="mpill">V5 <b>{n_v5}</b> · M0 <b>{n_m0}</b></span>
    <span class="mpill">realized <b>{n_realized}</b> · pending <b>{n_pending}</b></span>
    <span class="mpill">OOS {oos_range or '—'}</span>
    <span class="meta-right">rendered {now}</span>
  </div>

  <div class="tier-group-label">Realized vs frozen envelope
    <span class="cnt">· {len(diff_cards)} profile{'s' if len(diff_cards) != 1 else ''}</span>
  </div>
  <div class="env-wrap">{''.join(diff_cards) if diff_cards else
      '<div class="action-card env-card"><div class="card-head"><span class="env-n">no realized rebalances yet</span></div></div>'}</div>

  <div class="tier-group-label">Per-rebalance log
    <span class="cnt">· {len(t)} rows · newest first</span>
  </div>
  <div class="log-shell">
    <table>
      <thead>
        <tr>
          <th>Date</th><th>Profile</th><th>Status</th>
          <th style="text-align:right">N</th>
          <th style="text-align:right">Gross</th>
          <th style="text-align:right">Net 60bps</th>
          <th style="text-align:right">Turnover</th>
          <th style="text-align:right">Rolling DD</th>
          <th style="text-align:right">CAGR sofar</th>
          <th style="text-align:right">Sharpe sofar</th>
          <th style="text-align:right">Equity</th>
        </tr>
      </thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>

  <div class="footer-meta">
    nyxmomentum v1.1 · frozen <b>{version}</b> · dual profile (V5 default · M0 bench) · append-only tracker
  </div>
</div>
</body></html>"""


def _render_picks_html(date_iso: str, profile: str,
                        picks_df: pd.DataFrame, version: str,
                        eligible_df: pd.DataFrame | None = None,
                        ice_map: dict | None = None,
                        sms_map: dict | None = None) -> str:
    """Render picks as briefing-style action cards (backdrop-blur, tier-coded)."""
    profile_color = _PROFILE_COLOR.get(profile, "var(--text-primary)")
    # Card tier mapping: V5 → gold (tier1), M0 → purple/blue (tier2b)
    pick_tier_class = "tier1" if profile == "V5" else "tier2b"
    pick_tag_class  = "t1"    if profile == "V5" else "t2b"
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    show_flow = bool(ice_map) or bool(sms_map)

    # Status → (label, badge css class)
    _STATUS_LABEL = {
        "core_keeper": ("KEEP",   "st-keep"),
        "sticky_keep": ("STICKY", "st-sticky"),
        "new_entrant": ("NEW",    "st-new"),
        "crowded_out": ("CROWD",  "st-crowd"),
        "dropped":     ("DROP",   "st-drop"),
        "bystander":   ("",       ""),
    }

    # Lookup status by ticker (only V5 main-grid uses it; preview wires it via eligible)
    status_by_ticker: dict[str, str] = {}
    prev_by_ticker: dict[str, bool] = {}
    if eligible_df is not None and not eligible_df.empty:
        for _, r in eligible_df.iterrows():
            status_by_ticker[r["ticker"]] = str(r["status"])
            if "prev_selected" in eligible_df.columns:
                prev_by_ticker[r["ticker"]] = bool(r["prev_selected"])

    def _ice_pill(ticker: str) -> str:
        if not ice_map:
            return ""
        r = ice_map.get(ticker)
        if not r:
            return '<span class="badge bdg-dim" title="ICE: no data">ICE —</span>'
        mult = getattr(r, "multiplier", None)
        icon = getattr(r, "icon", "") or ""
        mult_s = f"{mult:.2f}" if mult is not None else "—"
        labels = getattr(r, "labels", {}) or {}
        def _v(k):
            lbl = labels.get(k)
            return getattr(lbl, "value", "") if lbl else ""
        title = (f"kt={_v('kurumsal_teyit')} · tb={_v('tasinan_birikim')} · "
                 f"kv={_v('kisa_vade')} · ma={_v('maliyet_avantaji')}")
        cls = "bdg-pos" if (mult is not None and mult >= 1.05) else (
              "bdg-neg" if (mult is not None and mult < 0.95) else "bdg-neu")
        return f'<span class="badge {cls}" title="{title}">{icon} ICE {mult_s}</span>'

    def _sms_pill(ticker: str) -> str:
        if not sms_map:
            return ""
        r = sms_map.get(ticker)
        if not r:
            return '<span class="badge bdg-dim" title="SMS: no data">SMS —</span>'
        score = getattr(r, "score", None)
        icon = getattr(r, "icon", "") or ""
        title = (f"S1={getattr(r,'s1','?')} S2={getattr(r,'s2','?')} "
                 f"S3={getattr(r,'s3','?')} S4={getattr(r,'s4','?')} "
                 f"S5={getattr(r,'s5','?')}")
        try:
            sv = int(score) if score is not None else 0
        except Exception:
            sv = 0
        cls = "bdg-pos" if sv >= 50 else ("bdg-neg" if sv < 30 else "bdg-neu")
        return f'<span class="badge {cls}" title="{title}">{icon} SMS {score}</span>'

    def _status_pill(ticker: str, fallback: str = "") -> str:
        st = status_by_ticker.get(ticker, fallback)
        if not st:
            return ""
        lbl, cls = _STATUS_LABEL.get(st, ("", ""))
        if not lbl:
            return ""
        return f'<span class="badge {cls}">{lbl}</span>'

    def _render_pick_card(row, tier_class: str, tag_class: str, tag_label: str,
                            include_weight: bool = True,
                            status_fallback: str = "") -> str:
        ticker = row["ticker"]
        rank = int(row["rank"])
        score = float(row["score"])
        weight_html = ""
        if include_weight and "weight" in row.index and pd.notna(row.get("weight")):
            weight_html = (
                '<span class="card-entry"><span class="e-label">w</span>'
                f'<span class="e-limit">{float(row["weight"])*100:.2f}%</span></span>')
        st_pill  = _status_pill(ticker, status_fallback)
        ice_pill = _ice_pill(ticker)
        sms_pill = _sms_pill(ticker)
        badges = "".join(p for p in (st_pill, ice_pill, sms_pill) if p)
        badges_html = f'<div class="card-badges">{badges}</div>' if badges else ""
        return (
            f'<div class="action-card {tier_class}">'
              '<div class="card-head">'
                f'<span class="rank-tag">#{rank}</span>'
                f'<span class="ticker">{ticker}</span>'
                f'<span class="tier-tag {tag_class}">{tag_label}</span>'
                f'<span class="score-pill">{score:.4f}</span>'
              '</div>'
              f'{badges_html}'
              f'{weight_html}'
            '</div>'
        )

    # ── Main picks grid ──
    pick_cards = []
    for _, r in picks_df.iterrows():
        # Default status fallback: derive from prev/rank if eligible map missing
        fb = ""
        if not status_by_ticker:
            fb = ""  # unknown
        pick_cards.append(
            _render_pick_card(r, pick_tier_class, pick_tag_class, profile,
                                include_weight=True, status_fallback=fb))
    main_grid = (
        f'<div class="tier-group-label">{profile} picks '
        f'<span class="cnt">· {len(picks_df)} names</span></div>'
        f'<div class="action-grid">{"".join(pick_cards)}</div>'
    )

    # ── Crowded-out + sticky sections ──
    extra_sections = ""
    if eligible_df is not None and not eligible_df.empty:
        crowded = eligible_df.loc[eligible_df["status"] == "crowded_out"].sort_values("rank")
        sticky  = eligible_df.loc[eligible_df["status"] == "sticky_keep"].sort_values("rank")

        if not crowded.empty:
            cards = "".join(
                _render_pick_card(r, "tier2a", "t2a", "OUT",
                                    include_weight=False,
                                    status_fallback="crowded_out")
                for _, r in crowded.iterrows())
            extra_sections += (
                '<div class="tier-group-label">Crowded-out new candidates '
                f'<span class="cnt">· {len(crowded)} names</span></div>'
                '<div class="aux-note">Would have been top-20 new entrants, but '
                'sticky keepers (rank ≤ 30 from prev basket) filled the slots '
                'first. n_enter=20, n_exit=30.</div>'
                f'<div class="action-grid">{cards}</div>'
            )

        if not sticky.empty:
            cards = "".join(
                _render_pick_card(r, "tier2b", "t2b", "STICKY",
                                    include_weight=False,
                                    status_fallback="sticky_keep")
                for _, r in sticky.iterrows())
            extra_sections += (
                '<div class="tier-group-label">Sticky keepers below top-20 '
                f'<span class="cnt">· {len(sticky)} names</span></div>'
                '<div class="aux-note">Carried over from previous basket at '
                'rank &gt; 20 but ≤ 30 — this is why non-contiguous ranks like '
                '21, 22, 26 appear in the main list.</div>'
                f'<div class="action-grid">{cards}</div>'
            )

    flow_note = (
        '<div class="aux-note" style="margin: 0 0 1rem">'
        'ICE / SMS = display-only institutional-flow context — selection is '
        'NOT filtered. Signals ~1–5 day horizon; refresh on actual rebalance '
        'day.</div>' if show_flow else "")

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nyxmomentum — {profile} picks · {date_iso}</title>
<style>
{_nox_css()}

/* ── briefing-style container ── */
.briefing-container {{
  position: relative; z-index: 1;
  max-width: 960px; margin: 0 auto;
  padding: 0 1.5rem 2rem;
}}

/* ── sticky status bar ── */
.nox-status-bar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(6,6,8,0.55);
  backdrop-filter: blur(24px) saturate(1.3);
  -webkit-backdrop-filter: blur(24px) saturate(1.3);
  border-bottom: 1px solid rgba(201,169,110,0.08);
  padding: 0.6rem 1.5rem;
  margin: 0 -1.5rem 1.5rem;
  display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
}}
.nox-status-bar .logo {{
  display: inline-flex; align-items: baseline;
  gap: 0.15rem; white-space: nowrap;
}}
.nox-status-bar .logo .nox-text {{
  font-family: var(--font-brand); font-size: 4.2rem;
  color: #fff; letter-spacing: 0.06em; line-height: 0.85;
}}
.nox-status-bar .logo .brief-text {{
  font-family: var(--font-handwrite); font-size: 1.1rem;
  color: {profile_color}; margin-left: 0.2rem;
  position: relative; top: -0.15rem;
}}
.nox-status-bar .profile-pill {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.2rem 0.7rem; border-radius: 1rem;
  font-weight: 600; font-size: 0.72rem;
  border: 1px solid {profile_color}; color: {profile_color};
  font-family: var(--font-mono);
  text-transform: uppercase; letter-spacing: 0.05em;
}}
.nox-status-bar .mpill {{
  font-family: var(--font-mono); font-size: 0.7rem;
  padding: 0.15rem 0.5rem; border-radius: 0.75rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); white-space: nowrap;
}}
.nox-status-bar .mpill b {{ font-weight: 600; color: #fff; }}
.nox-status-bar .meta-right {{
  font-size: 0.7rem; color: var(--text-muted);
  font-family: var(--font-mono); white-space: nowrap; margin-left: auto;
}}
.nox-status-bar .back-link {{
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--text-secondary); text-decoration: none;
  padding: 0.15rem 0.5rem; border-radius: 0.75rem;
  border: 1px solid var(--border-subtle);
}}
.nox-status-bar .back-link:hover {{
  color: {profile_color}; border-color: {profile_color};
}}

/* ── action cards (briefing-style) ── */
.action-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 0.75rem; margin-bottom: 1.5rem;
}}
.action-card {{
  background: rgba(199,189,190,0.08);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: none; border-radius: 18px;
  padding: 0.85rem 1.05rem;
  border-left: 3px solid transparent;
  transition: all 0.25s ease;
  animation: cardFadeIn 0.3s ease-out both;
  position: relative; overflow: hidden;
}}
.action-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(184,149,110,0.15), transparent);
  opacity: 0; transition: opacity 0.3s;
}}
.action-card:hover {{
  background: rgba(199,189,190,0.12);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(184,149,110,0.03);
  transform: translateY(-1px);
}}
.action-card:hover::before {{ opacity: 1; }}
.action-card.tier1  {{ border-left: 3px solid rgba(201,169,110,0.45); }}
.action-card.tier2a {{ border-left: 3px solid rgba(184,149,110,0.25); }}
.action-card.tier2b {{ border-left: 3px solid rgba(138,122,158,0.30); }}
@keyframes cardFadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.action-card .card-head {{
  display: flex; align-items: center; gap: 0.45rem;
  margin-bottom: 0.4rem;
}}
.action-card .card-head .rank-tag {{
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--text-muted);
}}
.action-card .card-head .ticker {{
  font-family: 'Inter', sans-serif; font-weight: 900;
  font-size: 1.05rem; color: #fff;
}}
.action-card .card-head .tier-tag {{
  font-size: 0.58rem; font-family: var(--font-mono);
  padding: 0.1rem 0.35rem; border-radius: 0.25rem;
  font-weight: 600; text-transform: uppercase;
}}
.action-card .card-head .tier-tag.t1  {{ background: rgba(201,169,110,0.12); color: #c9a96e; }}
.action-card .card-head .tier-tag.t2a {{ background: rgba(168,135,106,0.12); color: #a8876a; }}
.action-card .card-head .tier-tag.t2b {{ background: rgba(138,122,158,0.12); color: #8a7a9e; }}
.action-card .card-head .score-pill {{
  font-family: var(--font-mono); font-size: 0.68rem;
  padding: 0.1rem 0.4rem; border-radius: 4px;
  background: var(--nox-gold-dim); color: var(--nox-gold);
  margin-left: auto;
}}
.action-card .card-badges {{
  display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.4rem;
}}
.action-card .card-entry {{
  display: inline-flex; gap: 0.4rem; flex-wrap: wrap;
  padding-top: 0.4rem; border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono); font-size: 0.7rem;
  width: 100%;
}}
.action-card .card-entry .e-label {{ color: var(--text-muted); }}
.action-card .card-entry .e-limit {{ color: var(--nox-gold); font-weight: 600; }}

/* badge styles for status / ICE / SMS */
.badge {{
  font-family: var(--font-mono); font-size: 0.62rem;
  padding: 0.12rem 0.42rem; border-radius: 0.4rem;
  font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; white-space: nowrap;
  border: 1px solid transparent;
}}
.badge.bdg-pos    {{ background: rgba(122,158,122,0.14); color: var(--nox-green); border-color: rgba(122,158,122,0.25); }}
.badge.bdg-neg    {{ background: rgba(158,90,90,0.14);   color: var(--nox-red);   border-color: rgba(158,90,90,0.25); }}
.badge.bdg-neu    {{ background: rgba(122,143,165,0.12); color: var(--nox-blue);  border-color: rgba(122,143,165,0.22); }}
.badge.bdg-dim    {{ background: rgba(138,133,128,0.10); color: var(--text-muted); }}
.badge.st-keep    {{ background: rgba(201,169,110,0.12); color: #c9a96e; border-color: rgba(201,169,110,0.25); }}
.badge.st-sticky  {{ background: rgba(138,122,158,0.14); color: #b8a4d4; border-color: rgba(138,122,158,0.25); }}
.badge.st-new     {{ background: rgba(122,158,122,0.16); color: var(--nox-green); border-color: rgba(122,158,122,0.30); }}
.badge.st-crowd   {{ background: rgba(168,135,106,0.14); color: #c8a484; border-color: rgba(168,135,106,0.25); }}
.badge.st-drop    {{ background: rgba(158,90,90,0.14);   color: var(--nox-red);   border-color: rgba(158,90,90,0.25); }}

/* tier-group-label: section heading */
.tier-group-label {{
  font-family: var(--font-display); font-weight: 600;
  font-size: 0.92rem; color: var(--text-secondary);
  margin: 1.3rem 0 0.5rem;
  display: flex; align-items: center; gap: 0.4rem;
}}
.tier-group-label:first-of-type {{ margin-top: 0.2rem; }}
.tier-group-label .cnt {{
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--text-muted);
}}
.aux-note {{
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--text-muted); margin: -0.2rem 0 0.6rem;
  max-width: 720px; line-height: 1.45;
}}
.footer-meta {{
  font-family: var(--font-mono); font-size: 0.68rem;
  color: var(--text-muted); margin-top: 1.5rem;
  padding-top: 0.8rem; border-top: 1px solid var(--border-subtle);
  text-align: center;
}}
</style></head><body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>
<div class="briefing-container">
  <div class="nox-status-bar">
    <span class="logo">
      <span class="nox-text">NYX</span><span class="brief-text">momentum</span>
    </span>
    <span class="profile-pill">{profile} · {date_iso}</span>
    <span class="mpill">frozen <b>{version}</b></span>
    <span class="meta-right">rendered {now}</span>
    <a class="back-link" href="tracker.html">← tracker</a>
  </div>
  {flow_note}
  {main_grid}
  {extra_sections}
  <div class="footer-meta">
    nyxmomentum v1.1 · {profile} · {len(picks_df)} names · frozen {version}
  </div>
</div>
</body></html>"""


def _write_html_outputs(tracker: pd.DataFrame, diff: pd.DataFrame,
                         version: str, live_dir: str,
                         v5_by_date: dict, m0_by_date: dict,
                         oos_dates: list) -> tuple[int, int]:
    """Write tracker.html + per-date picks HTML. Returns (tracker_written, picks_written).

    HTML picks are rendered from the stored CSV on disk (single source of
    truth) — never from in-memory fresh parquet data. This keeps HTML and
    CSV consistent when the append-only guardrail preserves stored rows.
    """
    def _load_eligible(iso: str, profile_lo: str) -> pd.DataFrame | None:
        p = os.path.join(live_dir, f"{iso}_{profile_lo}_eligible.csv")
        if not os.path.exists(p):
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    picks_written = 0
    pick_index: dict = {}
    for d in v5_by_date.keys():
        iso = d.date().isoformat()
        csv_p  = os.path.join(live_dir, f"{iso}_v5_picks.csv")
        html_p = os.path.join(live_dir, f"{iso}_v5_picks.html")
        if not os.path.exists(csv_p):
            continue
        picks = pd.read_csv(csv_p)
        el = _load_eligible(iso, "v5")
        with open(html_p, "w", encoding="utf-8") as fh:
            fh.write(_render_picks_html(iso, "V5", picks, version, el))
        pick_index[(iso, "v5")] = True
        picks_written += 1
    for d in m0_by_date.keys():
        iso = d.date().isoformat()
        csv_p  = os.path.join(live_dir, f"{iso}_m0_picks.csv")
        html_p = os.path.join(live_dir, f"{iso}_m0_picks.html")
        if not os.path.exists(csv_p):
            continue
        picks = pd.read_csv(csv_p)
        el = _load_eligible(iso, "m0")
        with open(html_p, "w", encoding="utf-8") as fh:
            fh.write(_render_picks_html(iso, "M0", picks, version, el))
        pick_index[(iso, "m0")] = True
        picks_written += 1

    # Tracker dashboard
    html = _render_tracker_html(tracker, diff, version, live_dir,
                                  oos_dates, pick_index)
    with open(os.path.join(live_dir, "tracker.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    return 1, picks_written


# ── Run ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    live_dir = ensure_dir(LIVE_DIR)
    tracker_path = os.path.join(live_dir, "tracker.csv")
    diff_path = os.path.join(live_dir, "tracker_envelope_diff.csv")

    print("[1/6] Loading selections + labels + frozen envelope …")
    m0_sel = pd.read_parquet(args.m0_selection)
    v5_sel = pd.read_parquet(args.v5_selection)
    labels = pd.read_parquet(args.labels)
    envelope, version = _load_frozen(args.frozen_meta)
    print(f"  frozen meta: {args.frozen_meta}  (version={version})")

    score_col_m0 = "prediction" if "prediction" in m0_sel.columns else "prediction_smoothed"
    score_col_v5 = "prediction_smoothed" if "prediction_smoothed" in v5_sel.columns else "prediction"

    m0_by_date = _profile_picks_by_date(m0_sel, score_col_m0)
    v5_by_date = _profile_picks_by_date(v5_sel, score_col_v5)
    oos_dates = sorted(set(m0_by_date.keys()) | set(v5_by_date.keys()))
    if oos_dates:
        print(f"  OOS dates in selection: {len(oos_dates)}  "
              f"({oos_dates[0].date()} → {oos_dates[-1].date()})")

    print("[2/6] Writing per-rebalance pick files (write-once) …")
    v5_stats = _write_pick_files_append_only(v5_by_date, "v5", live_dir)
    m0_stats = _write_pick_files_append_only(m0_by_date, "m0", live_dir)
    print(f"  V5 picks: {v5_stats[0]} new, {v5_stats[1]} unchanged, {v5_stats[2]} mismatch-preserved")
    print(f"  M0 picks: {m0_stats[0]} new, {m0_stats[1]} unchanged, {m0_stats[2]} mismatch-preserved")

    # Eligible (top-30 with dampener-status labels) — drift-guarded write-once.
    v5_el = _profile_eligible_by_date(v5_sel, score_col_v5, top_cut=30, n_enter=20)
    m0_el = _profile_eligible_by_date(m0_sel, score_col_m0, top_cut=30, n_enter=20)
    v5_el_stats = _write_eligible_files_append_only(v5_el, v5_by_date, "v5", live_dir)
    m0_el_stats = _write_eligible_files_append_only(m0_el, m0_by_date, "m0", live_dir)
    print(f"  V5 eligible: {v5_el_stats[0]} new, {v5_el_stats[1]} exists, {v5_el_stats[2]} drift-skipped")
    print(f"  M0 eligible: {m0_el_stats[0]} new, {m0_el_stats[1]} exists, {m0_el_stats[2]} drift-skipped")

    print("[3/6] Building tracker rows (V5 + M0) …")
    labels_by_date = {
        pd.Timestamp(d): g.set_index("ticker")["l1_forward_return"]
        for d, g in labels[["ticker", "rebalance_date", "l1_forward_return"]]
        .groupby("rebalance_date", sort=True)
    }
    v5_inc = _build_profile_tracker(v5_by_date, "V5", labels_by_date, args.bps)
    m0_inc = _build_profile_tracker(m0_by_date, "M0", labels_by_date, args.bps)
    incoming = pd.concat([v5_inc, m0_inc], ignore_index=True)

    existing = None
    if os.path.exists(tracker_path) and not args.reseed:
        try:
            existing = pd.read_csv(tracker_path)
            print(f"  existing tracker rows: {len(existing)}  (append-only mode)")
        except Exception as e:
            print(f"  [WARN] could not read existing tracker, starting fresh: {e}",
                  file=sys.stderr)
            existing = None
    elif args.reseed:
        print("  --reseed: rebuilding tracker from scratch")
        # Archive previous tracker with timestamp (never delete history)
        if os.path.exists(tracker_path):
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            archive = os.path.join(live_dir, f"tracker_before_reseed_{stamp}.csv")
            os.rename(tracker_path, archive)
            print(f"  previous tracker archived: {archive}")

    merged = _merge_append_only(existing, incoming)
    # Re-derive rolling envelope from the merged return series.
    tracker = _recompute_rolling_for_tracker(merged)
    tracker.to_csv(tracker_path, index=False)

    print("[4/6] Computing realized-vs-envelope diff …")
    diff = _envelope_diff(tracker, envelope)
    diff.to_csv(diff_path, index=False)

    print("[5/6] Rendering HTML (tracker dashboard + per-date picks) …")
    _, n_picks_html = _write_html_outputs(
        tracker, diff, version, live_dir,
        v5_by_date, m0_by_date, oos_dates,
    )
    print(f"  tracker.html + {n_picks_html} pick HTML files")

    print("[6/6] Writing run meta …")
    meta = {
        "produced_at":          pd.Timestamp.utcnow().isoformat(),
        "frozen_version":       version,
        "frozen_meta_path":     args.frozen_meta,
        "n_rebalances_v5":      int((tracker["profile"] == "V5").sum()),
        "n_rebalances_m0":      int((tracker["profile"] == "M0").sum()),
        "n_pending":            int((tracker["status"] == "pending").sum()),
        "cost_bps":             args.bps,
        "append_only":          not args.reseed,
        "elapsed_sec":          time.time() - t0,
    }
    save_json(os.path.join(live_dir, "run_meta.json"), meta)

    _print_summary(tracker, diff, live_dir, version, t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m0-selection", default=M0_SEL,
                   help="M0 selection parquet from step5")
    p.add_argument("--v5-selection", default=V5_SEL,
                   help="V5 selection parquet from step5")
    p.add_argument("--labels",       default=LABELS,
                   help="step1_labels.parquet (realized returns)")
    p.add_argument("--frozen-meta",  default=FROZEN_META,
                   help="frozen_v1_1.json (envelope + version)")
    p.add_argument("--bps",          type=int, default=60,
                   help="round-trip cost in bps for net return (default 60)")
    p.add_argument("--reseed",       action="store_true",
                   help="rebuild tracker from scratch (archives previous). "
                        "Use only on freeze bump, not routine runs.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
