"""triangle_break runner — descriptive converging-line scan over 5h/1d/1w/1M.

Writes per-family triangle_break_<fam>.parquet (asc/sym/desc only;
expanding & ambiguous filtered, parallelism ≤ 0.25 channels filtered).

Usage:
    python tools/triangle_break_scan_live.py [--asof "2026-04-29"]
                                             [--families tr_5h tr_1d ...]
                                             [--tickers ASELS KRDMD]
                                             [--min-coverage 0.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from triangle_break.engine import scan
from triangle_break.schema import FAMILIES


_STATE_ORD = {"trigger": 0, "extended": 1, "pre_breakout": 2}
_SUBTYPE_ORD = {"ascending": 0, "symmetric": 1, "descending": 2}


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_tier_ord"] = (~df["tier_a"].fillna(False).astype(bool)).astype(int)
    df["_state_ord"] = df["signal_state"].map(_STATE_ORD).fillna(9).astype(int)
    df["_sub_ord"] = df["triangle_subtype"].map(_SUBTYPE_ORD).fillna(9).astype(int)
    df["_contract"] = df["width_contraction_ratio"].fillna(1.0).astype(float)
    df = df.sort_values(
        ["_tier_ord", "_state_ord", "_sub_ord", "_contract"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return df.drop(columns=["_tier_ord", "_state_ord", "_sub_ord", "_contract"])


def _summarize(family: str, df: pd.DataFrame) -> None:
    print()
    print(f"=== {family} | triangles={len(df)} ===")
    if df.empty:
        print("  (no triangles)")
        return
    states = df["signal_state"].value_counts().to_dict()
    subtypes = df["triangle_subtype"].value_counts().to_dict()
    n_tier_a = int(df["tier_a"].fillna(False).sum())
    print(f"  states: {states}")
    print(f"  subtypes: {subtypes}")
    print(f"  tier_a: {n_tier_a}/{len(df)}")

    # State × subtype matrix
    st = list(_STATE_ORD)
    sub = list(_SUBTYPE_ORD)
    mat = (
        df.assign(s=df["signal_state"], k=df["triangle_subtype"])
          .pivot_table(index="s", columns="k", aggfunc="size", fill_value=0)
    )
    mat = mat.reindex(index=[r for r in st if r in mat.index],
                      columns=[c for c in sub if c in mat.columns])
    if not mat.empty:
        print()
        print("  state × subtype:")
        print(mat.to_string().replace("\n", "\n    "))

    cols = [
        "ticker", "signal_state", "triangle_subtype", "tier_a",
        "n_pivots_upper", "n_pivots_lower",
        "channel_width_pct", "width_contraction_ratio", "bars_to_apex",
        "fit_quality", "asof_close", "breakout_age_bars",
    ]
    avail = [c for c in cols if c in df.columns]
    df_sorted = _rank(df)
    print()
    print(df_sorted[avail].to_string(index=False))


def _grand_summary(out: dict[str, pd.DataFrame]) -> None:
    parts = []
    by_state = {"trigger": 0, "extended": 0, "pre_breakout": 0}
    by_sub = {"ascending": 0, "symmetric": 0, "descending": 0}
    n_total = 0
    n_tier_a = 0
    tier_a_rows: list[pd.Series] = []
    for fam, df in out.items():
        if df.empty:
            continue
        n_total += len(df)
        for k, v in df["signal_state"].value_counts().items():
            by_state[k] = by_state.get(k, 0) + int(v)
        for k, v in df["triangle_subtype"].value_counts().items():
            by_sub[k] = by_sub.get(k, 0) + int(v)
        ta = df[df["tier_a"].fillna(False) == True]  # noqa: E712
        if not ta.empty:
            n_tier_a += len(ta)
            for _, r in ta.iterrows():
                tier_a_rows.append(r)

    parts.append(f"TOTAL = {n_total}")
    parts.append(f"tier_A = {n_tier_a}")
    parts.append("states[" + " ".join(f"{k}={by_state[k]}" for k in ("trigger","extended","pre_breakout")) + "]")
    parts.append("subs[" + " ".join(f"{k}={by_sub[k]}" for k in ("ascending","symmetric","descending")) + "]")
    print()
    print("=" * 78)
    print("[grand] " + " · ".join(parts))
    print("=" * 78)

    # Tier A spotlight
    print()
    if not tier_a_rows:
        print("  TIER-A spotlight: (none — strict containment + ≥3/3 touch threshold met by zero setup)")
    else:
        print(f"  TIER-A spotlight ({len(tier_a_rows)}):")
        cols = [
            "ticker", "setup_family", "signal_state", "triangle_subtype",
            "n_pivots_upper", "n_pivots_lower",
            "channel_width_pct", "width_contraction_ratio", "bars_to_apex",
            "asof_close",
        ]
        ta_df = pd.DataFrame(tier_a_rows)
        avail = [c for c in cols if c in ta_df.columns]
        print(ta_df[avail].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp (Europe/Istanbul). Default = last bar in each panel.")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES),
                    help=f"Families to run; default = all {list(FAMILIES)}")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Subset tickers (default: full universe).")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    args = ap.parse_args()

    print(f"[run] asof={args.asof or 'latest'}  families={args.families}  "
          f"tickers={'all' if args.tickers is None else args.tickers}")

    t0 = time.time()
    out = scan(
        families=args.families,
        tickers=args.tickers,
        asof=args.asof,
        min_coverage=args.min_coverage,
        write_parquet=True,
    )
    elapsed = time.time() - t0
    total = sum(len(v) for v in out.values())
    print(f"\n[run] done in {elapsed:.1f}s  triangles={total}")

    for f in args.families:
        v = out.get(f, pd.DataFrame())
        _summarize(f, v)

    _grand_summary(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
