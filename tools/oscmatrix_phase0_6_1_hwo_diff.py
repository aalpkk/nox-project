"""Phase 0.6.1 — HWO Up/Down event rule diff diagnostic.

Inputs: 24 TV CSVs (DISCOVERY 4 + CORE 11 + STRESS 9).

Goal: explain why OURS fires +30-42% more HWO Up events than TV CSV, and
prescribe a single-knob fix. Phase 4 stays blocked until HWO Up F1 ±1
≥ 0.85 and over-fire ≤ +15%.

This script does:
  1) Recomputes HW + signal under {close, hl2, hlc3, ohlc4} sources to
     pin down whether RSI source alone closes the gap.
  2) For each OURS HWO Up event, tags it as TP_eq (TV fire within ±1) /
     near_miss_2 / near_miss_3 / pure_extra.
  3) For each pure_extra and near-miss event collects: hw at fire,
     hw-sig distance, prev_hw - prev_sig distance, bars_since_prev_OURS,
     bars_to_nearest_TV (signed).
  4) Tests three candidate suppressors at OURS fires:
       Hysteresis  : require (hw - sig)        ≥ EPS
       PrevGap     : require (prev_hw - prev_sig) ≤ -EPS
       Cooldown    : drop OURS same-direction fire if previous OURS fire
                     was within K bars
     Sweep EPS ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 3.0} and K ∈ {2, 3, 4, 5}.
  5) Pick best knob per source on (F1 ±1, over-fire %, recall floor 0.80).

Outputs:
  output/oscmatrix_phase0_6_1_hwo_diff_per_event.csv
  output/oscmatrix_phase0_6_1_hwo_diff_per_ticker.csv
  output/oscmatrix_phase0_6_1_hwo_diff_knob_sweep.csv
  output/oscmatrix_phase0_6_1_hwo_diff_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = Path("output")
OUT_PER_EVENT = OUT_DIR / "oscmatrix_phase0_6_1_hwo_diff_per_event.csv"
OUT_PER_TICKER = OUT_DIR / "oscmatrix_phase0_6_1_hwo_diff_per_ticker.csv"
OUT_KNOB_SWEEP = OUT_DIR / "oscmatrix_phase0_6_1_hwo_diff_knob_sweep.csv"
OUT_REPORT = OUT_DIR / "oscmatrix_phase0_6_1_hwo_diff_report.md"

CSV_DIR_A = "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1"
CSV_DIR_B = f"{CSV_DIR_A}/New Folder With Items 3"

DISCOVERY = {
    "THYAO": f"{CSV_DIR_A}/BIST_THYAO, 1D.csv",
    "GARAN": f"{CSV_DIR_B}/BIST_GARAN, 1D.csv",
    "EREGL": f"{CSV_DIR_B}/BIST_EREGL, 1D.csv",
    "ASELS": f"{CSV_DIR_B}/BIST_ASELS, 1D.csv",
}
CORE = {t: f"{CSV_DIR_B}/BIST_{t}, 1D.csv" for t in [
    "AKBNK", "SISE", "TUPRS", "KCHOL", "BIMAS", "FROTO",
    "PETKM", "KRDMD", "YKBNK", "TOASO", "SAHOL",
]}
STRESS = {t: f"{CSV_DIR_B}/BIST_{t}, 1D.csv" for t in [
    "KAREL", "PARSN", "BRISA", "GOODY", "INDES",
    "NETAS", "KONTR", "KLSER", "MAVI",
]}
COHORTS = {"DISCOVERY": DISCOVERY, "CORE": CORE, "STRESS": STRESS}

SOURCES = ["close", "hl2", "hlc3", "ohlc4"]
EPS_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
COOLDOWN_GRID = [0, 2, 3, 4, 5]
TOL_PRIMARY = 1


# =========================================================================
# Primitives
# =========================================================================

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(src: pd.Series, n: int) -> pd.Series:
    d = src.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = _rma(up, n) / _rma(dn, n).replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _src(tv: pd.DataFrame, src_name: str) -> pd.Series:
    if src_name == "close":
        return tv["close"]
    if src_name == "hl2":
        return (tv["high"] + tv["low"]) / 2.0
    if src_name == "hlc3":
        return (tv["high"] + tv["low"] + tv["close"]) / 3.0
    if src_name == "ohlc4":
        return (tv["open"] + tv["high"] + tv["low"] + tv["close"]) / 4.0
    raise ValueError(src_name)


def hw_signal(tv: pd.DataFrame, src_name: str, length: int = 7,
              sig_len: int = 3) -> tuple[pd.Series, pd.Series]:
    src = _src(tv, src_name)
    hw = _rsi(src, length).rolling(sig_len, min_periods=sig_len).mean()
    sig = hw.rolling(sig_len, min_periods=sig_len).mean()
    return hw, sig


def cross_events(hw: pd.Series, sig: pd.Series) -> tuple[pd.Series, pd.Series]:
    hp = hw.shift(1)
    sp = sig.shift(1)
    cu = (hp <= sp) & (hw > sig)
    cd = (hp >= sp) & (hw < sig)
    return cu.fillna(False), cd.fillna(False)


def event_metrics(tv_evt: pd.Series, our_evt: pd.Series, tol: int) -> dict:
    tv_ix = np.where(tv_evt.values)[0]
    ou_ix = np.where(our_evt.values)[0]
    if tv_ix.size == 0 and ou_ix.size == 0:
        return dict(p=np.nan, r=np.nan, f1=np.nan, n_tv=0, n_ours=0)
    if tv_ix.size == 0:
        return dict(p=0.0, r=np.nan, f1=0.0, n_tv=0, n_ours=int(ou_ix.size))
    if ou_ix.size == 0:
        return dict(p=np.nan, r=0.0, f1=0.0, n_tv=int(tv_ix.size), n_ours=0)
    matched_ours = sum(np.any(np.abs(tv_ix - i) <= tol) for i in ou_ix)
    matched_tv = sum(np.any(np.abs(ou_ix - i) <= tol) for i in tv_ix)
    p = matched_ours / ou_ix.size
    r = matched_tv / tv_ix.size
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(p=p, r=r, f1=f1, n_tv=int(tv_ix.size), n_ours=int(ou_ix.size))


def signed_dist_to_nearest(events_a: np.ndarray, events_b: np.ndarray) -> np.ndarray:
    """For each index in `events_a`, return signed (a - nearest_b)."""
    out = np.full(events_a.size, np.nan)
    if events_b.size == 0:
        return out
    for k, i in enumerate(events_a):
        diffs = i - events_b
        out[k] = diffs[np.argmin(np.abs(diffs))]
    return out


def apply_hysteresis(cu: pd.Series, hw: pd.Series, sig: pd.Series,
                     eps_now: float = 0.0, eps_prev: float = 0.0) -> pd.Series:
    if eps_now == 0.0 and eps_prev == 0.0:
        return cu
    gap_now = (hw - sig)
    gap_prev = (hw.shift(1) - sig.shift(1))
    keep = (gap_now >= eps_now) & (gap_prev <= -eps_prev)
    return cu & keep.fillna(False)


def apply_cooldown(cu: pd.Series, k: int) -> pd.Series:
    if k <= 0:
        return cu
    arr = cu.values.copy()
    last = -10_000
    for i in range(arr.size):
        if not arr[i]:
            continue
        if i - last < k:
            arr[i] = False
        else:
            last = i
    return pd.Series(arr, index=cu.index)


# =========================================================================
# Load
# =========================================================================

def load(path: str) -> pd.DataFrame:
    tv = pd.read_csv(path)
    tv["ts"] = pd.to_datetime(tv["time"], unit="s", utc=True).dt.tz_convert("Europe/Istanbul")
    tv = tv.set_index("ts").sort_index()
    return tv


# =========================================================================
# Per-event diagnostic
# =========================================================================

def per_event_records(cohort: str, tkr: str, tv: pd.DataFrame,
                      cu_ours: pd.Series, cu_tv_evt: pd.Series,
                      hw: pd.Series, sig: pd.Series) -> list[dict]:
    tv_ix = np.where(cu_tv_evt.values)[0]
    ou_ix = np.where(cu_ours.values)[0]
    rows: list[dict] = []
    # OURS-side: each ours fire — class
    for i in ou_ix:
        d_signed = signed_dist_to_nearest(np.array([i]), tv_ix)[0]
        d_abs = np.abs(d_signed) if not np.isnan(d_signed) else np.nan
        if np.isnan(d_abs):
            kind = "pure_extra"
        elif d_abs <= 1:
            kind = "tp"
        elif d_abs <= 2:
            kind = "near_miss_2"
        elif d_abs <= 3:
            kind = "near_miss_3"
        else:
            kind = "pure_extra"
        # bars since prev OURS fire (same direction)
        prev_ours = ou_ix[ou_ix < i]
        bars_prev = (i - prev_ours[-1]) if prev_ours.size else np.nan
        rows.append({
            "cohort": cohort, "ticker": tkr, "side": "OURS",
            "bar_idx": int(i),
            "ts": tv.index[i].date().isoformat() if i < len(tv.index) else "",
            "kind": kind,
            "signed_dist_to_tv": float(d_signed) if not np.isnan(d_signed) else np.nan,
            "hw_at_fire": float(hw.iloc[i]) if i < len(hw) else np.nan,
            "sig_at_fire": float(sig.iloc[i]) if i < len(sig) else np.nan,
            "gap_at_fire": float(hw.iloc[i] - sig.iloc[i]) if i < len(hw) else np.nan,
            "gap_prev_at_fire": (
                float(hw.iloc[i - 1] - sig.iloc[i - 1])
                if i >= 1 else np.nan
            ),
            "bars_since_prev_ours": float(bars_prev),
        })
    # TV-side: each TV fire — class as recall TP / FN
    for i in tv_ix:
        d_signed = signed_dist_to_nearest(np.array([i]), ou_ix)[0]
        d_abs = np.abs(d_signed) if not np.isnan(d_signed) else np.nan
        if np.isnan(d_abs):
            kind = "fn_pure_miss"
        elif d_abs <= 1:
            kind = "tp"
        elif d_abs <= 2:
            kind = "fn_near_miss_2"
        elif d_abs <= 3:
            kind = "fn_near_miss_3"
        else:
            kind = "fn_pure_miss"
        rows.append({
            "cohort": cohort, "ticker": tkr, "side": "TV",
            "bar_idx": int(i),
            "ts": tv.index[i].date().isoformat() if i < len(tv.index) else "",
            "kind": kind,
            "signed_dist_to_tv": float(d_signed) if not np.isnan(d_signed) else np.nan,
            "hw_at_fire": float(hw.iloc[i]) if i < len(hw) else np.nan,
            "sig_at_fire": float(sig.iloc[i]) if i < len(sig) else np.nan,
            "gap_at_fire": float(hw.iloc[i] - sig.iloc[i]) if i < len(hw) else np.nan,
            "gap_prev_at_fire": (
                float(hw.iloc[i - 1] - sig.iloc[i - 1])
                if i >= 1 else np.nan
            ),
            "bars_since_prev_ours": np.nan,
        })
    return rows


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("=== Phase 0.6.1 — HWO event rule diff diagnostic ===\n")

    # All tickers
    all_t: list[tuple[str, str, str]] = []
    for cohort, mp in COHORTS.items():
        for tkr, path in mp.items():
            all_t.append((cohort, tkr, path))

    # 1) Per-ticker, per-source baseline metrics + per-event tags
    per_ticker_rows: list[dict] = []
    per_event_rows_all: list[dict] = []

    for cohort, tkr, path in all_t:
        tv = load(path)
        tv_hwo_up = tv["HWO Up"].notna()
        tv_hwo_dn = tv["HWO Down"].notna()
        for src in SOURCES:
            hw, sig = hw_signal(tv, src)
            cu, cd = cross_events(hw, sig)
            up_m = event_metrics(tv_hwo_up, cu, TOL_PRIMARY)
            dn_m = event_metrics(tv_hwo_dn, cd, TOL_PRIMARY)
            per_ticker_rows.append({
                "cohort": cohort, "ticker": tkr, "src": src,
                "tv_up": up_m["n_tv"], "our_up": up_m["n_ours"],
                "up_p": up_m["p"], "up_r": up_m["r"], "up_f1": up_m["f1"],
                "tv_dn": dn_m["n_tv"], "our_dn": dn_m["n_ours"],
                "dn_p": dn_m["p"], "dn_r": dn_m["r"], "dn_f1": dn_m["f1"],
            })
            if src == "close":
                # Per-event records only for the current production source
                per_event_rows_all.extend(per_event_records(
                    cohort, tkr, tv, cu, tv_hwo_up, hw, sig
                ))

    pt = pd.DataFrame(per_ticker_rows)
    pe = pd.DataFrame(per_event_rows_all)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt.to_csv(OUT_PER_TICKER, index=False)
    pe.to_csv(OUT_PER_EVENT, index=False)
    print(f"  wrote {OUT_PER_TICKER}")
    print(f"  wrote {OUT_PER_EVENT}")
    print()

    # 2) Source-level aggregate
    print("=== Source aggregate (HWO Up F1 ±1) ===")
    agg_src = pt.groupby("src").agg(
        tv_up=("tv_up", "sum"),
        our_up=("our_up", "sum"),
        up_p=("up_p", "mean"),
        up_r=("up_r", "mean"),
        up_f1=("up_f1", "mean"),
        dn_f1=("dn_f1", "mean"),
    )
    agg_src["over_fire_pct"] = (agg_src["our_up"] / agg_src["tv_up"] - 1.0) * 100
    print(agg_src.reindex(SOURCES).to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    # 3) Per-event class breakdown (close source)
    print("=== Per-event class breakdown (current src=close) ===")
    ours_evt = pe[pe["side"] == "OURS"]
    tv_evt = pe[pe["side"] == "TV"]
    print("OURS-side fire classes:")
    print(ours_evt["kind"].value_counts().to_string())
    print()
    print("TV-side fire classes:")
    print(tv_evt["kind"].value_counts().to_string())
    print()

    # 4) Pure-extra OURS distribution
    extras = ours_evt[ours_evt["kind"] == "pure_extra"].copy()
    if len(extras):
        print("=== Pure-extra OURS HWO Up — feature distribution ===")
        print(extras[[
            "hw_at_fire", "gap_at_fire", "gap_prev_at_fire", "bars_since_prev_ours"
        ]].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).to_string(float_format=lambda x: f"{x:.3f}"))
        print()
    near = ours_evt[ours_evt["kind"].isin(["near_miss_2", "near_miss_3"])].copy()
    if len(near):
        print("=== Near-miss (±2/±3) OURS HWO Up — feature distribution ===")
        print(near[[
            "hw_at_fire", "gap_at_fire", "gap_prev_at_fire",
            "bars_since_prev_ours", "signed_dist_to_tv"
        ]].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).to_string(float_format=lambda x: f"{x:.3f}"))
        print()

    # 5) Knob sweep — Hysteresis (eps_now), PrevGap (eps_prev), Cooldown
    print("=== Knob sweep (HWO Up F1 ±1, src=close unless noted) ===")
    knob_rows: list[dict] = []
    for src in SOURCES:
        for cohort, tkr, path in all_t:
            tv = load(path)
            tv_hwo_up = tv["HWO Up"].notna()
            hw, sig = hw_signal(tv, src)
            cu, _ = cross_events(hw, sig)
            for eps in EPS_GRID:
                for k in COOLDOWN_GRID:
                    cu2 = apply_hysteresis(cu, hw, sig, eps_now=eps, eps_prev=0.0)
                    cu2 = apply_cooldown(cu2, k)
                    m = event_metrics(tv_hwo_up, cu2, TOL_PRIMARY)
                    knob_rows.append({
                        "src": src, "ticker": tkr, "cohort": cohort,
                        "eps_now": eps, "cooldown": k,
                        "tv_up": m["n_tv"], "our_up": m["n_ours"],
                        "p": m["p"], "r": m["r"], "f1": m["f1"],
                    })
    ks = pd.DataFrame(knob_rows)
    ks.to_csv(OUT_KNOB_SWEEP, index=False)
    print(f"  wrote {OUT_KNOB_SWEEP}\n")

    # 6) Aggregate knob sweep
    sweep_agg = ks.groupby(["src", "eps_now", "cooldown"]).agg(
        tv_up=("tv_up", "sum"),
        our_up=("our_up", "sum"),
        f1=("f1", "mean"),
        p=("p", "mean"),
        r=("r", "mean"),
    )
    sweep_agg["over_fire_pct"] = (sweep_agg["our_up"] / sweep_agg["tv_up"] - 1.0) * 100
    sweep_agg = sweep_agg.reset_index()

    print("=== Knob sweep — top 12 by mean F1 (recall ≥ 0.80) ===")
    elig = sweep_agg[sweep_agg["r"] >= 0.80].sort_values("f1", ascending=False)
    print(elig.head(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Knob sweep — top 12 by smallest |over_fire_pct| (F1 ≥ 0.80) ===")
    elig2 = sweep_agg[sweep_agg["f1"] >= 0.80].copy()
    elig2["abs_over"] = elig2["over_fire_pct"].abs()
    print(elig2.sort_values("abs_over").head(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # 7) Final pick: best F1 with over-fire ≤ +15%
    candidate_pool = sweep_agg[
        (sweep_agg["r"] >= 0.80)
        & (sweep_agg["over_fire_pct"].abs() <= 15.0)
    ].sort_values("f1", ascending=False)
    if len(candidate_pool):
        winner = candidate_pool.iloc[0]
        print("=== Best knob meeting acceptance (F1 max | r≥0.80 ∧ |over|≤15%) ===")
        print(winner.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        print("=== No knob meets full acceptance (F1 ≥ 0.85, |over| ≤ 15%, r ≥ 0.80) ===")
        # Fallback: best F1 with looser constraint
        best_f1 = sweep_agg[sweep_agg["r"] >= 0.80].sort_values("f1", ascending=False).iloc[0]
        print("Best F1 with r≥0.80:")
        print(best_f1.to_string(float_format=lambda x: f"{x:.3f}"))
        best_over = sweep_agg[
            (sweep_agg["f1"] >= 0.80) & (sweep_agg["r"] >= 0.80)
        ].sort_values(by="over_fire_pct", key=lambda x: x.abs()).head(1)
        if len(best_over):
            print("Smallest |over| with f1≥0.80, r≥0.80:")
            print(best_over.iloc[0].to_string(float_format=lambda x: f"{x:.3f}"))

    # 8) Markdown report
    lines = []
    lines.append("# Phase 0.6.1 — HWO event rule diff diagnostic\n")
    lines.append("## Source aggregate (no knob)\n")
    lines.append(agg_src.reindex(SOURCES).to_markdown(floatfmt=".3f"))
    lines.append("\n")
    lines.append("## Per-event class breakdown (src=close)\n")
    lines.append("OURS HWO Up classes:\n\n```\n" + ours_evt["kind"].value_counts().to_string() + "\n```\n")
    lines.append("TV HWO Up classes:\n\n```\n" + tv_evt["kind"].value_counts().to_string() + "\n```\n")
    lines.append("## Pure-extra OURS distribution\n")
    if len(extras):
        lines.append("```\n" + extras[[
            "hw_at_fire", "gap_at_fire", "gap_prev_at_fire", "bars_since_prev_ours"
        ]].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).to_string() + "\n```\n")
    lines.append("## Knob sweep — top 12 by F1 (recall ≥ 0.80)\n")
    lines.append(elig.head(12).to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n")
    lines.append("## Knob sweep — top 12 by smallest |over| (F1 ≥ 0.80)\n")
    lines.append(elig2.sort_values("abs_over").head(12).to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n")
    lines.append("## Acceptance pick\n")
    if len(candidate_pool):
        winner = candidate_pool.iloc[0]
        lines.append("```\n" + winner.to_string() + "\n```\n")
    else:
        lines.append("**No knob hits acceptance (F1 ≥ 0.85 ∧ |over| ≤ 15% ∧ r ≥ 0.80).** Showing best fallbacks below.\n")
        best_f1 = sweep_agg[sweep_agg["r"] >= 0.80].sort_values("f1", ascending=False).iloc[0]
        lines.append("Best F1 with r≥0.80:\n```\n" + best_f1.to_string() + "\n```\n")
        best_over = sweep_agg[
            (sweep_agg["f1"] >= 0.80) & (sweep_agg["r"] >= 0.80)
        ].sort_values(by="over_fire_pct", key=lambda x: x.abs()).head(1)
        if len(best_over):
            lines.append("Smallest |over| with f1≥0.80, r≥0.80:\n```\n" + best_over.iloc[0].to_string() + "\n```\n")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"\n  wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
