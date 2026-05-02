"""Heuristic composite ranker for mb_scanner output (Phase-1).

Each family parquet (`output/mb_scanner_<family>.parquet`) carries 0..N
rows per (ticker) at as-of — one per simultaneously-active quartet. The
ranker collapses to one row per (ticker, family) by max composite score:

    state_w  = {retest_bounce: 3.0, above_mb: 2.0, mitigation_touch: 1.5}
    fresh    = max(0, 1 - zone_age_bars / family.max_zone_age_bars)
    bos_pen  = 1.0 if bos_distance_atr < 5  else max(0.4, 5.0/bos)
    rk_b     = {deep_touch: 1.20, shallow_touch: 1.05, no_touch: 0.95, '': 1.00}
    nest_b   = min(1.3, 1.0 + 0.1 * (n_active_quartets - 1))
    score    = state_w * fresh * bos_pen * rk_b * nest_b

`extended` rows are excluded — a quartet whose retest already happened is
no longer a fresh-entry candidate.

`also_fires_in` is an info column (not used in scoring) listing other
families where the same ticker has any active fresh state.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import _PARAMS as FAM_PARAMS
from .schema import FAMILIES

ACTIVE_STATES = ("retest_bounce", "above_mb", "mitigation_touch")
STATE_W = {"retest_bounce": 3.0, "above_mb": 2.0, "mitigation_touch": 1.5}
RK_B = {"deep_touch": 1.20, "shallow_touch": 1.05, "no_touch": 0.95, "": 1.00}

OUT_DIR = Path("output")


def score_row(row: pd.Series, max_zone_age: int) -> float:
    state = row["signal_state"]
    if state not in STATE_W:
        return float("nan")
    state_w = STATE_W[state]

    age = float(row["zone_age_bars"]) if pd.notna(row["zone_age_bars"]) else max_zone_age
    fresh = max(0.0, 1.0 - age / max_zone_age) if max_zone_age > 0 else 0.0

    bos = row["bos_distance_atr"]
    if pd.isna(bos):
        bos_pen = 1.0
    else:
        bos = float(bos)
        bos_pen = 1.0 if bos < 5.0 else max(0.4, 5.0 / bos) if bos > 0 else 1.0

    rk = row.get("retest_kind") or ""
    rk_b = RK_B.get(rk, 1.0)

    n_q = int(row["n_active_quartets"]) if pd.notna(row["n_active_quartets"]) else 1
    nest_b = min(1.3, 1.0 + 0.1 * (n_q - 1))

    return state_w * fresh * bos_pen * rk_b * nest_b


def load_family(fam: str, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    path = out_dir / f"mb_scanner_{fam}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_cross_tf_index(out_dir: Path = OUT_DIR) -> dict[str, set[str]]:
    """Map ticker → set of families where it has any active fresh state."""
    out: dict[str, set[str]] = {}
    for fam in FAMILIES:
        df = load_family(fam, out_dir=out_dir)
        if df.empty:
            continue
        active = df[df["signal_state"].isin(ACTIVE_STATES)]["ticker"].unique()
        for t in active:
            out.setdefault(t, set()).add(fam)
    return out


def rank_family(
    fam: str,
    cross_tf: dict[str, set[str]] | None = None,
    out_dir: Path = OUT_DIR,
) -> pd.DataFrame:
    df = load_family(fam, out_dir=out_dir)
    if df.empty:
        return df
    params = FAM_PARAMS[fam]
    df = df[df["signal_state"].isin(ACTIVE_STATES)].copy()
    if df.empty:
        return df
    df["score"] = df.apply(lambda r: score_row(r, params.max_zone_age_bars), axis=1)
    df = df.dropna(subset=["score"])
    df = df.sort_values("score", ascending=False).drop_duplicates("ticker", keep="first")
    if cross_tf is not None:
        df["also_fires_in"] = df["ticker"].map(
            lambda t: ",".join(sorted(cross_tf.get(t, set()) - {fam})) or ""
        )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def rank_all(out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    cross_tf = build_cross_tf_index(out_dir=out_dir)
    return {fam: rank_family(fam, cross_tf, out_dir=out_dir) for fam in FAMILIES}
