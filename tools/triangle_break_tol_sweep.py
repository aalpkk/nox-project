"""triangle_break tolerance sweep (v0.3) — tier-A vs ATR-fraction.

Sweeps `containment_tol_atr_k` (fraction of ATR_14). Holds floor/cap at
schema defaults and max_line_violations=0. Writes per-k summary + tier-A
spotlights to stdout. No parquet output (write_parquet=False).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from triangle_break import schema as tb_schema
from triangle_break.engine import scan


KS = [0.15, 0.20, 0.25, 0.30, 0.40]


def _run_one(k: float, families: list[str]) -> dict[str, pd.DataFrame]:
    saved = {f: tb_schema.FAMILIES[f] for f in families}
    try:
        for f in families:
            tb_schema.FAMILIES[f] = replace(saved[f], containment_tol_atr_k=k)
        return scan(families=families, write_parquet=False)
    finally:
        for f, p in saved.items():
            tb_schema.FAMILIES[f] = p


def main() -> int:
    families = list(tb_schema.FAMILIES)

    summary_rows = []
    spotlights: dict[float, list[pd.Series]] = {}

    for k in KS:
        out = _run_one(k, families)
        for f in families:
            df = out.get(f, pd.DataFrame())
            n = len(df)
            tier_a = int(df["tier_a"].fillna(False).sum()) if n else 0
            states = df["signal_state"].value_counts().to_dict() if n else {}
            summary_rows.append({
                "k_atr": k,
                "family": f,
                "total": n,
                "tier_a": tier_a,
                "trigger": states.get("trigger", 0),
                "extended": states.get("extended", 0),
                "pre_breakout": states.get("pre_breakout", 0),
            })
        ta_rows: list[pd.Series] = []
        for f in families:
            df = out.get(f, pd.DataFrame())
            if df.empty:
                continue
            ta = df[df["tier_a"].fillna(False) == True]  # noqa: E712
            for _, r in ta.iterrows():
                ta_rows.append(r)
        spotlights[k] = ta_rows

    sm = pd.DataFrame(summary_rows)
    print()
    print("=== sweep summary (tier-A by k × family) ===")
    pivot = sm.pivot(index="k_atr", columns="family", values="tier_a").fillna(0).astype(int)
    pivot["GRAND"] = pivot.sum(axis=1)
    print(pivot.to_string())

    print()
    print("=== full per-k breakdown ===")
    print(sm.to_string(index=False))

    print()
    print("=== tier-A spotlights ===")
    cols = [
        "ticker", "setup_family", "signal_state", "triangle_subtype",
        "n_pivots_upper", "n_pivots_lower",
        "channel_width_pct", "width_contraction_ratio", "bars_to_apex",
        "fit_quality", "asof_close",
    ]
    for k, rows in spotlights.items():
        print()
        print(f"--- k={k:.2f} | tier_A={len(rows)} ---")
        if not rows:
            print("  (none)")
            continue
        df = pd.DataFrame(rows)
        avail = [c for c in cols if c in df.columns]
        print(df[avail].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
