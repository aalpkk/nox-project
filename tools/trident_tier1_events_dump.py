"""PR-DE-3.23: event-level Trident Tier-1 CSV producer.

Emits one row per trident-probe event (all 8 BIST families) at the per-family
completed-bar ASOF, with the 4 Tier-1 gates evaluated independently and
``tier1_eligible`` = AND of all four.

This is the **event-level** companion to the watchlist generator's same-row
TRIDENT_TIER1_ACTIVE tag. The watchlist generator only tags ``mb_1d`` events
on the literal asof date; this dump covers every family on its own completed
bar (5h/1d use asof; 1w uses the last Friday ≤ asof; 1M uses the last
month-end ≤ asof), so weekly/monthly Tier-1 candidates are visible even when
the watchlist target date is mid-week.

Gate semantics — verbatim with PR-DE-3.18:
  G1: D_pct ∈ [TIER1_D_PCT_LOW, TIER1_D_PCT_HIGH)  — D-zone band
  G2: structural_invalidation_pct_below_B ≤ sil_t33 — deep stop (t33 over
      365d strictly before family_asof, **per family**)
  G3: bos_distance_atr_at_event ≥ bos_t67           — strong BoS (t67 over
      same trailing window, per family)
  G4: XU100 ret_20d > 0 at the most-recent close ≤ asof — global regime

``pivot_post_confirmed`` (PR-DE-3.22) is carried through as defensive context
metadata — it is **not** a gate (currently 100% True given find_all_quartets
enforces the HL after-side window).

Usage:
    python3 tools/trident_tier1_events_dump.py --asof-date 2026-05-22

Output:
    output/decision_engine_v1_trident_tier1_events_<ASOF>.csv

Pre-req:
  - output/trident_probe_mb_3y.parquet (current regen)
  - output/xu100_extfeed_daily.parquet (for G4)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"

TRIDENT_PROBE_PATH = OUT_DIR / "trident_probe_mb_3y.parquet"
XU100_DAILY_PATH = OUT_DIR / "xu100_extfeed_daily.parquet"

TIER1_D_PCT_LOW = 20.0
TIER1_D_PCT_HIGH = 30.0
TIER1_TRAILING_WINDOW_DAYS = 365
TIER1_TRAILING_MIN_N = 30

OUTPUT_COLUMNS = [
    "ticker", "family", "event_bar_date",
    "D_pct", "SIL_pct_below_B", "bos_distance_atr",
    "sil_t33_for_family", "bos_t67_for_family",
    "pivot_post_confirmed",
    "G1_D_pct_20_30", "G2_sil_low_deep", "G3_bos_high", "G4_regime_up",
    "gates_passed_count", "gates_passed", "tier1_eligible",
]


def _compute_xu100_regime(asof: str) -> dict:
    """Mirror tools/decision_engine_v1_watchlist_generator._compute_xu100_regime.

    Returns dict with keys: available, asof_available, ret_20d_pct, pass,
    days_stale, reason.
    """
    state = {
        "available": False, "asof_available": "", "ret_20d_pct": float("nan"),
        "pass": False, "days_stale": 0, "reason": "feed_missing",
    }
    if not XU100_DAILY_PATH.exists():
        return state
    import pandas as _pd
    try:
        df = pq.read_table(XU100_DAILY_PATH).to_pandas()
    except Exception as exc:
        state["reason"] = f"read_failed:{exc.__class__.__name__}"
        return state
    if df.empty or "close" not in df.columns:
        state["reason"] = "empty_or_no_close"
        return state
    if not isinstance(df.index, _pd.DatetimeIndex):
        for cand in ("Date", "date", "ts"):
            if cand in df.columns:
                df = df.set_index(_pd.to_datetime(df[cand]))
                break
        if not isinstance(df.index, _pd.DatetimeIndex):
            state["reason"] = "no_datetime_index"
            return state
    df = df.sort_index()
    cutoff = _pd.Timestamp(asof) + _pd.Timedelta(days=1)
    df = df.loc[df.index < cutoff]
    if len(df) < 21:
        state["reason"] = f"short_history:n={len(df)}"
        return state
    last_close = float(df["close"].iloc[-1])
    ref_close = float(df["close"].iloc[-21])
    last_dt = df.index[-1]
    ret_20d_pct = (last_close / ref_close - 1.0) * 100.0
    asof_avail = str(last_dt.date())
    days_stale = max(0, (_pd.Timestamp(asof) - _pd.Timestamp(asof_avail)).days)
    state.update({
        "available": True, "asof_available": asof_avail,
        "ret_20d_pct": ret_20d_pct, "pass": ret_20d_pct > 0.0,
        "days_stale": days_stale,
        "reason": "ok" if days_stale == 0 else f"stale_{days_stale}d",
    })
    return state


def _tercile(values: list[float], frac: float) -> float | None:
    """Reproduce watchlist generator's quantile (1-indexed rank-of-frac)."""
    if not values:
        return None
    s = sorted(values)
    idx = int(round(len(s) * frac)) - 1
    if idx < 0:
        idx = 0
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


def _safe_float(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _fmt_float(v: float | None, ndigits: int = 2) -> str:
    if v is None:
        return ""
    return f"{v:.{ndigits}f}"


def build_events(asof: str) -> tuple[list[dict], dict]:
    """Build event-level Tier-1 rows for the given asof.

    Returns (rows, summary). Rows have keys matching OUTPUT_COLUMNS.
    """
    if not TRIDENT_PROBE_PATH.exists():
        return [], {"reason": "probe_missing"}
    asof_dt = dt.date.fromisoformat(asof)
    asof_str = asof_dt.isoformat()
    table = pq.read_table(
        TRIDENT_PROBE_PATH,
        columns=[
            "ticker", "family", "event_bar_date", "D_pct",
            "structural_invalidation_pct_below_B",
            "bos_distance_atr_at_event",
            "pivot_post_confirmed",
        ],
    )
    raw = table.to_pylist()
    by_fam: dict[str, list[dict]] = {}
    for r in raw:
        fam = r.get("family") or ""
        if not fam:
            continue
        ed = r.get("event_bar_date")
        ed_str = str(ed)[:10] if ed is not None else ""
        if not ed_str:
            continue
        r["_event_bar_date_str"] = ed_str
        by_fam.setdefault(fam, []).append(r)

    xu100_state = _compute_xu100_regime(asof_str)
    g4 = bool(xu100_state.get("pass"))

    rows: list[dict] = []
    summary = {
        "asof": asof_str,
        "g4_pass": g4,
        "xu100_state": xu100_state,
        "per_family": {},
    }

    for fam in sorted(by_fam.keys()):
        fam_rows = by_fam[fam]
        candidates = [r["_event_bar_date_str"] for r in fam_rows
                      if r["_event_bar_date_str"] <= asof_str]
        if not candidates:
            summary["per_family"][fam] = {"reason": "no_events_at_or_before_asof"}
            continue
        family_asof = max(candidates)
        win_start = (dt.date.fromisoformat(family_asof)
                     - dt.timedelta(days=TIER1_TRAILING_WINDOW_DAYS)).isoformat()
        sil_train: list[float] = []
        bos_train: list[float] = []
        today_rows: list[dict] = []
        for r in fam_rows:
            ed_str = r["_event_bar_date_str"]
            if ed_str == family_asof:
                today_rows.append(r)
                continue
            if win_start <= ed_str < family_asof:
                sf = _safe_float(r.get("structural_invalidation_pct_below_B"))
                if sf is not None:
                    sil_train.append(sf)
                bf = _safe_float(r.get("bos_distance_atr_at_event"))
                if bf is not None:
                    bos_train.append(bf)
        trailing_n = min(len(sil_train), len(bos_train))
        if trailing_n < TIER1_TRAILING_MIN_N:
            summary["per_family"][fam] = {
                "family_asof": family_asof, "trailing_n": trailing_n,
                "reason": f"trailing_too_short:n={trailing_n}",
            }
            continue
        sil_t33 = _tercile(sil_train, 1 / 3)
        bos_t67 = _tercile(bos_train, 2 / 3)
        fam_summary = {
            "family_asof": family_asof, "trailing_n": trailing_n,
            "sil_t33": sil_t33, "bos_t67": bos_t67,
            "today_n": len(today_rows), "eligible_n": 0,
        }
        for r in today_rows:
            d_f = _safe_float(r.get("D_pct"))
            s_f = _safe_float(r.get("structural_invalidation_pct_below_B"))
            b_f = _safe_float(r.get("bos_distance_atr_at_event"))
            g1 = (d_f is not None) and (TIER1_D_PCT_LOW <= d_f < TIER1_D_PCT_HIGH)
            g2 = (s_f is not None) and (sil_t33 is not None) and (s_f <= sil_t33)
            g3 = (b_f is not None) and (bos_t67 is not None) and (b_f >= bos_t67)
            ppc = r.get("pivot_post_confirmed")
            gates_pass = []
            if g1: gates_pass.append("G1_D_pct_20_30")
            if g2: gates_pass.append("G2_sil_low_deep")
            if g3: gates_pass.append("G3_bos_high")
            if g4: gates_pass.append("G4_regime_up")
            eligible = bool(g1 and g2 and g3 and g4)
            if eligible:
                fam_summary["eligible_n"] += 1
            rows.append({
                "ticker": r.get("ticker") or "",
                "family": fam,
                "event_bar_date": r["_event_bar_date_str"],
                "D_pct": _fmt_float(d_f),
                "SIL_pct_below_B": _fmt_float(s_f),
                "bos_distance_atr": _fmt_float(b_f),
                "sil_t33_for_family": _fmt_float(sil_t33),
                "bos_t67_for_family": _fmt_float(bos_t67),
                "pivot_post_confirmed": "" if ppc is None else str(bool(ppc)),
                "G1_D_pct_20_30": str(g1),
                "G2_sil_low_deep": str(g2),
                "G3_bos_high": str(g3),
                "G4_regime_up": str(g4),
                "gates_passed_count": str(len(gates_pass)),
                "gates_passed": ";".join(gates_pass),
                "tier1_eligible": str(eligible),
            })
        summary["per_family"][fam] = fam_summary

    rows.sort(key=lambda x: (
        x["tier1_eligible"] != "True",
        -int(x["gates_passed_count"]),
        x["family"],
        x["ticker"],
    ))
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    asof = args.asof_date
    try:
        dt.date.fromisoformat(asof)
    except ValueError:
        print(f"!! invalid --asof-date: {asof}", file=sys.stderr)
        return 2

    rows, summary = build_events(asof)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"decision_engine_v1_trident_tier1_events_{asof}.csv"

    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    eligible = sum(1 for r in rows if r["tier1_eligible"] == "True")
    print(f"[trident_tier1_events_dump] asof={asof}  "
          f"rows={len(rows)}  ELIGIBLE={eligible}  → {target}")
    print(f"  G4 regime: pass={summary['g4_pass']} "
          f"(ret_20d_pct={summary['xu100_state'].get('ret_20d_pct'):+.2f}, "
          f"asof_available={summary['xu100_state'].get('asof_available')})")
    print()
    print("  per-family:")
    for fam in sorted(summary["per_family"].keys()):
        fs = summary["per_family"][fam]
        if "reason" in fs and "family_asof" not in fs:
            print(f"    {fam:8s} → {fs['reason']}")
            continue
        if "reason" in fs:
            print(f"    {fam:8s} family_asof={fs['family_asof']} "
                  f"({fs['reason']})")
            continue
        print(f"    {fam:8s} family_asof={fs['family_asof']}  "
              f"trailing_n={fs['trailing_n']}  "
              f"sil_t33={fs['sil_t33']:+.2f}  bos_t67={fs['bos_t67']:+.2f}  "
              f"today_n={fs['today_n']}  eligible={fs['eligible_n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
