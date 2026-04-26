"""SBT-1700 — exit discovery on TRAIN: variant×cohort metrics + selection.

This module is split-agnostic: callers pass an already-filtered TRAIN
DataFrame and the daily master OHLCV. Splits are routed via
`sbt1700.splits.load_split` upstream so that discovery can never read
validation or test rows.

Outputs:
- A `pd.DataFrame` of variant-level metrics (`metrics_table`)
- A `pd.DataFrame` of carried candidates (max 3, with diversity gates)

Selection methodology (revised)
-------------------------------
Carry max 3 candidates, each tied to a distinct *hypothesis*:

* slot 1 — **raw trend control**: best no-partial trend variant
  (F0 or F4 without partial), ranked by penalty-adjusted score. Carries
  the strongest "let the trend run" hypothesis even if it has fragile
  flags — its job is to be the control.
* slot 2 — **best balanced partial**: best partial variant, restricted
  to F1 / F2 / F3 (the families whose *purpose* is partial profit-take).
  F4-with-partial is excluded from this slot to avoid duplicate-behavior
  with slot 1 (slot 1 is also F4-eligible). Ranked by penalty-adjusted
  score.
* slot 3 — **MFE-capture / low-giveback**: required even if avg_R is
  lower than slot 1's. Eligible iff PF > 1, avg_R > 0, N ≥ 30 and the
  variant's `captured_MFE_ratio_cohort` is **strictly greater than**
  slot 1's AND its `avg_giveback_R` is strictly lower. Ranked by an
  MFE-capture-weighted score, preferring trend kinds not already
  covered by slots 1+2. (The earlier `+0.05` margin was relaxed when
  the F0–F4 grammar gave no qualifying variant; the F5/F6/F7
  extension lifts that constraint.)

Penalties (shared across slots): top-5 positive-R share > 0.50,
giveback / MFE > 0.60, mean-vs-median sign mismatch, BE-cuts-winners
(BE-on with hit-rate > 0.20 AND captured_MFE below the no-BE sibling).
N < 30 zeros the score; 30 ≤ N < 60 takes 0.20.

Slot score = `avg_R_net × max(0, 1 − Σ penalties)`. Slot-3 score also
adds `0.5 × captured_MFE_ratio_cohort` and subtracts giveback/MFE
beyond 0.50 to prioritise the MFE-capture hypothesis.

Duplicate-behavior rule: a variant is rejected from later slots if it
shares (`trend_kind`, `initial_sl_atr`, `partial_size`, `partial_R`,
`max_hold_bars`) with an already-carried slot. Rejected top-N
performers are surfaced in the recommendation MD so the operator sees
*why* something with high avg_R was not carried.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sbt1700.exits_v2 import ExitConfigV2, simulate_exit_v2


SMALL_EPS = 1e-9


# ---------- per-row simulation ------------------------------------------------

def _by_ticker_master(daily_master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "ticker" not in daily_master.columns:
        raise KeyError("daily master must have a 'ticker' column")
    return {
        tk: g[["Open", "High", "Low", "Close"]].sort_index()
        for tk, g in daily_master.groupby("ticker")
    }


def simulate_variant_on_panel(
    cfg: ExitConfigV2,
    panel: pd.DataFrame,
    by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply a variant to every row in `panel`. Returns a per-trade frame.

    Required panel columns:
        ticker, date, close_1700, atr14_prior
    Optional columns (used only by some variants):
        box_top, box_bottom
    """
    needed = {"ticker", "date", "close_1700", "atr14_prior"}
    missing = needed - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    pcl = panel.copy()
    pcl["date"] = pd.to_datetime(pcl["date"])
    use_box = ("box_top" in pcl.columns) and ("box_bottom" in pcl.columns)

    for r in pcl.itertuples(index=False):
        sub = by_ticker.get(getattr(r, "ticker"))
        if sub is None or sub.empty:
            continue
        entry_date = pd.Timestamp(getattr(r, "date"))
        prior_closes = sub.loc[sub.index < entry_date, "Close"].values.astype(float)
        bt = float(getattr(r, "box_top", float("nan"))) if use_box else float("nan")
        bb = float(getattr(r, "box_bottom", float("nan"))) if use_box else float("nan")
        sim = simulate_exit_v2(
            cfg,
            entry_date=entry_date,
            entry_px=float(getattr(r, "close_1700")),
            atr_1700=float(getattr(r, "atr14_prior")),
            forward_ohlc=sub,
            prior_closes=prior_closes,
            box_top=bt if np.isfinite(bt) else None,
            box_bottom=bb if np.isfinite(bb) else None,
        )
        merged = {"ticker": getattr(r, "ticker"), "date": entry_date}
        merged.update(sim)
        rows.append(merged)
    return pd.DataFrame(rows)


# ---------- variant-level metrics --------------------------------------------

def variant_metrics(trade_df: pd.DataFrame, cfg: ExitConfigV2) -> dict:
    sub = trade_df.dropna(subset=["realized_R_net"]).copy()
    n = len(sub)
    if n == 0:
        return _empty_metrics(cfg)

    R = sub["realized_R_net"].astype(float)
    Rg = sub["realized_R_gross"].astype(float)
    wins = R[R > 0].sum()
    losses = -R[R < 0].sum()
    pf_net = float("inf") if (losses == 0 and wins > 0) else (
        float(wins / losses) if losses > 0 else float("nan"))
    total = float(R.sum())
    R_sorted_desc = R.sort_values(ascending=False)

    top5_share = float(R_sorted_desc.head(5).sum() / total) if total != 0 else float("nan")
    top5_pos_share = (
        float(R_sorted_desc.head(5).clip(lower=0).sum() / R.clip(lower=0).sum())
        if R.clip(lower=0).sum() > 0 else float("nan")
    )

    mfe = sub["MFE_R"].astype(float)
    avg_mfe = float(mfe.mean())
    sum_mfe = float(mfe.sum())
    avg_realized = float(R.mean())
    avg_giveback = float(sub["giveback_R"].astype(float).mean())
    # Per-trade-average ratio is unbounded near MFE→0; report it filtered
    # to trades with meaningful MFE (≥ 0.10 R) AND a cohort-aggregate ratio.
    meaningful_mfe = mfe >= 0.10
    cap_ratios = (R[meaningful_mfe] / mfe[meaningful_mfe]).replace([np.inf, -np.inf], np.nan).dropna()
    avg_cap_ratio = float(cap_ratios.mean()) if len(cap_ratios) else float("nan")
    cohort_cap_ratio = float(R.sum() / sum_mfe) if sum_mfe > SMALL_EPS else float("nan")

    p25 = float(R.quantile(0.25))
    p50 = float(R.quantile(0.50))
    p75 = float(R.quantile(0.75))

    # MFE-bucket diagnostics: how often each bucket is reached and the
    # realised R conditional on having reached it. Conditional means
    # require finite Mfe and finite realized_R (already dropna'd above).
    mfe_buckets: dict[float, tuple[float, float]] = {}
    for thr in (1.0, 2.0, 3.0):
        mask = mfe >= thr
        share = float(mask.mean())
        cond_mean = float(R[mask].mean()) if mask.any() else float("nan")
        mfe_buckets[thr] = (share, cond_mean)

    # Worst rolling 20-trade R (chronological order by exit_date when present).
    if "exit_date" in sub.columns:
        ordered = sub.sort_values("exit_date")
    else:
        ordered = sub
    if len(ordered) >= 20:
        rolling20_min = float(ordered["realized_R_net"].rolling(20).sum().min())
    else:
        rolling20_min = float(ordered["realized_R_net"].sum())

    # Yearly metrics.
    yearly = []
    if "date" in sub.columns:
        tmp = sub.copy()
        tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
        for y, g in tmp.groupby("year"):
            yR = g["realized_R_net"].astype(float)
            yw = yR[yR > 0].sum()
            yl = -yR[yR < 0].sum()
            ypf = float("inf") if (yl == 0 and yw > 0) else (
                float(yw / yl) if yl > 0 else float("nan"))
            yearly.append({
                "year": int(y), "n": int(len(g)),
                "avg_R": float(yR.mean()), "total_R": float(yR.sum()),
                "PF": ypf, "WR": float((yR > 0).mean()),
            })
    yearly_min_avg_R = float(min((y["avg_R"] for y in yearly), default=float("nan")))
    yearly_max_avg_R = float(max((y["avg_R"] for y in yearly), default=float("nan")))

    # Concentration.
    if "ticker" in sub.columns:
        tk = sub.groupby("ticker")["realized_R_net"].agg(["count", "sum"])
        ticker_top1_share = float(tk["count"].max() / len(sub)) if len(sub) else float("nan")
        ticker_top3_share = float(tk["count"].sort_values(ascending=False).head(3).sum() / len(sub))
    else:
        ticker_top1_share = ticker_top3_share = float("nan")
    if "date" in sub.columns:
        dt = sub.groupby(pd.to_datetime(sub["date"]).dt.date).size()
        date_top1_share = float(dt.max() / len(sub))
        date_top3_share = float(dt.sort_values(ascending=False).head(3).sum() / len(sub))
    else:
        date_top1_share = date_top3_share = float("nan")

    return {
        "exit_variant": cfg.name,
        "exit_family": cfg.family,
        "trend_kind": cfg.trend_kind,
        "initial_sl_atr": cfg.initial_sl_atr,
        "partial_size": cfg.partial_size,
        "partial_R": cfg.partial_R,
        "partial2_size": cfg.partial2_size,
        "partial2_R": cfg.partial2_R,
        "breakeven_R": cfg.breakeven_R if cfg.breakeven_R is not None else float("nan"),
        "activation_R": cfg.activation_R,
        "max_hold_bars": cfg.max_hold_bars,
        "n": int(n),
        "wr": float((R > 0).mean()),
        "pf_net": pf_net,
        "avg_R_net": avg_realized,
        "median_R_net": p50,
        "total_R_net": total,
        "p25_R_net": p25,
        "p75_R_net": p75,
        "avg_R_gross": float(Rg.mean()),
        "avg_bars_held": float(sub["bars_held"].astype(float).mean()),
        "initial_stop_hit_rate": float(sub["initial_stop_hit"].astype(bool).mean()),
        "breakeven_stop_hit_rate": float(sub["breakeven_stop_hit"].astype(bool).mean()),
        "profit_lock_stop_hit_rate": (
            float(sub["profit_lock_stop_hit"].astype(bool).mean())
            if "profit_lock_stop_hit" in sub.columns else 0.0),
        "partial_hit_rate": float(sub["partial_hit"].astype(bool).mean()),
        "partial2_hit_rate": float(sub["partial2_hit"].astype(bool).mean()),
        "trend_exit_hit_rate": float(sub["trend_exit_hit"].astype(bool).mean()),
        "max_hold_hit_rate": float(sub["max_hold_hit"].astype(bool).mean()),
        "avg_MFE_R": avg_mfe,
        "sum_MFE_R": sum_mfe,
        "avg_realized_R": avg_realized,
        "avg_giveback_R": avg_giveback,
        "captured_MFE_ratio": avg_cap_ratio,           # per-trade avg (MFE≥0.10R only)
        "captured_MFE_ratio_cohort": cohort_cap_ratio, # sum_realized / sum_MFE
        "top5_R_share": top5_share,
        "top5_pos_share": top5_pos_share,
        "ticker_top1_share": ticker_top1_share,
        "ticker_top3_share": ticker_top3_share,
        "date_top1_share": date_top1_share,
        "date_top3_share": date_top3_share,
        "worst_20trade_rolling_R": rolling20_min,
        "yearly_min_avg_R": yearly_min_avg_R,
        "yearly_max_avg_R": yearly_max_avg_R,
        "pct_MFE_ge_1R": mfe_buckets[1.0][0],
        "pct_MFE_ge_2R": mfe_buckets[2.0][0],
        "pct_MFE_ge_3R": mfe_buckets[3.0][0],
        "avg_R_given_MFE_ge_1R": mfe_buckets[1.0][1],
        "avg_R_given_MFE_ge_2R": mfe_buckets[2.0][1],
        "avg_R_given_MFE_ge_3R": mfe_buckets[3.0][1],
        "yearly_records": yearly,  # kept as nested list; serialized separately
    }


def _empty_metrics(cfg: ExitConfigV2) -> dict:
    nan = float("nan")
    return {
        "exit_variant": cfg.name, "exit_family": cfg.family,
        "trend_kind": cfg.trend_kind, "initial_sl_atr": cfg.initial_sl_atr,
        "partial_size": cfg.partial_size, "partial_R": cfg.partial_R,
        "partial2_size": cfg.partial2_size, "partial2_R": cfg.partial2_R,
        "breakeven_R": cfg.breakeven_R if cfg.breakeven_R is not None else nan,
        "activation_R": cfg.activation_R,
        "max_hold_bars": cfg.max_hold_bars,
        "n": 0, "wr": nan, "pf_net": nan, "avg_R_net": nan,
        "median_R_net": nan, "total_R_net": 0.0,
        "p25_R_net": nan, "p75_R_net": nan, "avg_R_gross": nan,
        "avg_bars_held": nan,
        "initial_stop_hit_rate": nan, "breakeven_stop_hit_rate": nan,
        "profit_lock_stop_hit_rate": nan,
        "partial_hit_rate": nan, "partial2_hit_rate": nan,
        "trend_exit_hit_rate": nan, "max_hold_hit_rate": nan,
        "avg_MFE_R": nan, "sum_MFE_R": 0.0, "avg_realized_R": nan,
        "avg_giveback_R": nan, "captured_MFE_ratio": nan,
        "captured_MFE_ratio_cohort": nan,
        "top5_R_share": nan, "top5_pos_share": nan,
        "ticker_top1_share": nan, "ticker_top3_share": nan,
        "date_top1_share": nan, "date_top3_share": nan,
        "worst_20trade_rolling_R": nan,
        "yearly_min_avg_R": nan,
        "yearly_max_avg_R": nan,
        "pct_MFE_ge_1R": nan, "pct_MFE_ge_2R": nan, "pct_MFE_ge_3R": nan,
        "avg_R_given_MFE_ge_1R": nan,
        "avg_R_given_MFE_ge_2R": nan,
        "avg_R_given_MFE_ge_3R": nan,
        "yearly_records": [],
    }


# ---------- selection rule ----------------------------------------------------

# Slot-2 (best balanced partial) is restricted to families whose *purpose*
# is partial profit-take. F4-with-partial is intentionally excluded from
# slot 2 — it duplicates slot-1 behavior (slot 1 is also F4-eligible).
PARTIAL_FAMILIES_FOR_SLOT2: frozenset[str] = frozenset({
    "F1_partial_no_breakeven",
    "F2_partial_with_breakeven",
    "F3_stepout",
})

# Slot 4 — capture-focused, F8 family only, conditional. The earlier
# F0..F7 grammar produced slot-3 carries with `captured_MFE_ratio_cohort`
# around 0.23 even when the explicit MFE-capture extension was
# available; before stepping back to validation we reserve a slot for
# any F8 profit-lock-ladder variant that passes a *higher* bar than
# slot 3. If no F8 variant clears these gates, the report records the
# negative finding rather than carrying a marginal candidate.
SLOT4_FAMILY_PREFIX: str = "F8"
SLOT4_CAP_FLOOR: float = 0.30           # absolute capture floor (≥ 0.30)
SLOT4_MIN_CAP_OVER_SLOT1: float = 0.05  # also require ≥ slot1 + 5pp
SLOT4_MIN_GB_DROP_R: float = 0.50       # giveback must drop ≥ 0.50 R vs slot1
SLOT4_MIN_PF: float = 1.20              # PF gate (slot 3 only requires > 1.0)
SLOT4_MEDIAN_FLOOR_R: float = -0.30     # median_R may not be deeper than this


def _has_partial(row: pd.Series) -> bool:
    return bool((row["partial_size"] > 0) or (row["partial2_size"] > 0))


def _is_no_partial_trend(row: pd.Series) -> bool:
    fam = row["exit_family"]
    if fam == "F0_no_partial_trend":
        return True
    if fam == "F4_structure" and not _has_partial(row):
        return True
    return False


def _behavior_signature(row) -> tuple:
    """Coarse fingerprint used for the duplicate-behavior gate.

    Two carries are 'effectively the same behavior' when they share the
    same trend_kind, stop sizing, partial schedule, profit-lock ladder
    and max-hold. BE is excluded — adding/removing BE on the same
    skeleton is a meaningful variation, not a duplicate. The lock
    ladder is part of the signature because F8a/b/c/d share TREND_NONE
    (or TREND_EMA for F8d) and would otherwise collide across families.
    """
    get = (lambda k: row[k]) if hasattr(row, "__getitem__") else (lambda k: getattr(row, k))
    # The ladder is propagated through scoring as a tuple; if absent we
    # use the family name as a tiebreaker so F8a vs F8b stay distinct
    # even if a downstream metrics frame drops the ladder column.
    ladder = None
    try:
        ladder = get("profit_lock_ladder")
    except (KeyError, AttributeError, IndexError):
        ladder = None
    fam = ""
    try:
        fam = str(get("exit_family"))
    except (KeyError, AttributeError, IndexError):
        fam = ""
    if ladder is None or (isinstance(ladder, float) and not np.isfinite(ladder)):
        ladder_key: object = fam if "F8" in fam else None
    else:
        try:
            ladder_key = tuple((round(float(a), 3), round(float(b), 3))
                               for a, b in ladder)
        except (TypeError, ValueError):
            ladder_key = fam if "F8" in fam else None
    return (
        get("trend_kind"),
        round(float(get("initial_sl_atr")), 3),
        round(float(get("partial_size")), 3),
        round(float(get("partial_R") or 0.0), 3),
        round(float(get("partial2_size")), 3),
        round(float(get("partial2_R") or 0.0), 3),
        int(get("max_hold_bars")),
        ladder_key,
    )


def _giveback_ratio(row: pd.Series) -> float:
    mfe = row.get("avg_MFE_R", float("nan"))
    gb = row.get("avg_giveback_R", float("nan"))
    if not (np.isfinite(mfe) and np.isfinite(gb)) or mfe <= SMALL_EPS:
        return float("nan")
    return float(gb / mfe)


def _penalty_score(row: pd.Series, sibling_lookup: dict) -> tuple[float, list[str]]:
    """Return (penalty_total, list_of_flag_strings)."""
    flags: list[str] = []
    pen = 0.0

    n = row["n"]
    if n < 30:
        pen += 1.0  # effectively zeros out the score
        flags.append(f"low_n_{int(n)}")
    elif n < 60:
        pen += 0.20
        flags.append(f"borderline_n_{int(n)}")

    top5_pos = row.get("top5_pos_share", float("nan"))
    if np.isfinite(top5_pos) and top5_pos > 0.50:
        pen += 0.15
        flags.append(f"top5_pos_share_{top5_pos:.2f}")

    gb_ratio = _giveback_ratio(row)
    if np.isfinite(gb_ratio) and gb_ratio > 0.60:
        pen += 0.15
        flags.append(f"giveback_{gb_ratio:.2f}")

    avg = row.get("avg_R_net", float("nan"))
    med = row.get("median_R_net", float("nan"))
    if np.isfinite(avg) and np.isfinite(med):
        if (avg > 0 > med) or (avg < 0 < med):
            pen += 0.10
            flags.append("mean_median_sign_mismatch")

    # Breakeven-likely-cuts-winners flag: only if BE is on AND there is a
    # no-BE sibling in the same family with same partial config.
    if (np.isfinite(row.get("breakeven_R", float("nan"))) and
            row.get("breakeven_stop_hit_rate", 0.0) > 0.20):
        sibling_key = (
            row["exit_family"], row["initial_sl_atr"],
            row["partial_size"], row["partial_R"],
            row["partial2_size"], row["partial2_R"],
            row["trend_kind"], row.get("max_hold_bars", 40),
        )
        sib = sibling_lookup.get(("noBE",) + sibling_key)
        if sib is not None and np.isfinite(sib.get("captured_MFE_ratio_cohort", float("nan"))):
            if (np.isfinite(row.get("captured_MFE_ratio_cohort", float("nan"))) and
                    row["captured_MFE_ratio_cohort"] < sib["captured_MFE_ratio_cohort"] - 0.05):
                pen += 0.20
                flags.append("breakeven_lowers_capture_vs_sibling")

    return pen, flags


def _build_sibling_lookup(metrics: pd.DataFrame) -> dict:
    """Map (BE-tag, family, sl, p1, p2, trend, hold) → row dict for noBE rows."""
    out: dict = {}
    for _, r in metrics.iterrows():
        if not np.isfinite(r.get("breakeven_R", float("nan"))):
            key = (
                "noBE", r["exit_family"], r["initial_sl_atr"],
                r["partial_size"], r["partial_R"],
                r["partial2_size"], r["partial2_R"],
                r["trend_kind"], r.get("max_hold_bars", 40),
            )
            out[key] = r.to_dict()
    return out


_SCORED_COLS: list[str] = [
    "exit_variant", "exit_family", "trend_kind", "is_no_partial_trend",
    "has_partial",
    # signature columns — needed by the duplicate-behavior gate
    "initial_sl_atr", "partial_size", "partial_R",
    "partial2_size", "partial2_R", "max_hold_bars",
    "n", "wr", "pf_net",
    "avg_R_net", "median_R_net", "total_R_net",
    "avg_MFE_R", "avg_giveback_R", "giveback_over_mfe",
    "captured_MFE_ratio_cohort",
    "top5_R_share", "top5_pos_share",
    "worst_20trade_rolling_R", "yearly_min_avg_R", "yearly_max_avg_R",
    "breakeven_stop_hit_rate", "profit_lock_stop_hit_rate",
    "max_hold_hit_rate",
    "pct_MFE_ge_1R", "pct_MFE_ge_2R", "pct_MFE_ge_3R",
    "avg_R_given_MFE_ge_1R", "avg_R_given_MFE_ge_2R", "avg_R_given_MFE_ge_3R",
    "penalty", "penalty_flags",
    "score_balanced", "score_mfe_capture",
]


def _score_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Annotate each variant with penalties and the two slot-specific scores."""
    if metrics.empty:
        return pd.DataFrame(columns=_SCORED_COLS)
    sibling_lookup = _build_sibling_lookup(metrics)
    rows: list[dict] = []
    for _, r in metrics.iterrows():
        pen, flags = _penalty_score(r, sibling_lookup)
        avg = r.get("avg_R_net", float("nan"))
        cap = r.get("captured_MFE_ratio_cohort", float("nan"))
        gb_ratio = _giveback_ratio(r)
        avg_f = float(avg) if np.isfinite(avg) else 0.0
        score_balanced = max(0.0, avg_f * max(0.0, 1.0 - pen))
        # MFE-capture score: rewards capture, lightly penalises giveback past
        # 0.50, gates on avg_R>0 and PF>1 outside the score itself.
        cap_term = float(cap) if np.isfinite(cap) else 0.0
        gb_term = max(0.0, (gb_ratio if np.isfinite(gb_ratio) else 0.0) - 0.50)
        score_mfe = (avg_f + 0.50 * cap_term - 0.30 * gb_term) * max(0.0, 1.0 - pen)
        rows.append({
            "exit_variant": r["exit_variant"],
            "exit_family": r["exit_family"],
            "trend_kind": r["trend_kind"],
            "is_no_partial_trend": _is_no_partial_trend(r),
            "has_partial": _has_partial(r),
            "initial_sl_atr": float(r["initial_sl_atr"]),
            "partial_size": float(r["partial_size"]),
            "partial_R": (float(r["partial_R"])
                          if r["partial_R"] is not None
                          and np.isfinite(r["partial_R"]) else 0.0),
            "partial2_size": float(r["partial2_size"]),
            "partial2_R": (float(r["partial2_R"])
                           if r["partial2_R"] is not None
                           and np.isfinite(r["partial2_R"]) else 0.0),
            "max_hold_bars": int(r["max_hold_bars"]),
            "n": int(r["n"]),
            "wr": float(r.get("wr", float("nan"))),
            "pf_net": float(r["pf_net"]) if np.isfinite(r["pf_net"]) else float("nan"),
            "avg_R_net": float(avg) if np.isfinite(avg) else float("nan"),
            "median_R_net": float(r["median_R_net"]),
            "total_R_net": float(r["total_R_net"]),
            "avg_MFE_R": float(r.get("avg_MFE_R", float("nan"))),
            "avg_giveback_R": float(r.get("avg_giveback_R", float("nan"))),
            "giveback_over_mfe": gb_ratio,
            "captured_MFE_ratio_cohort": float(cap) if np.isfinite(cap) else float("nan"),
            "top5_R_share": float(r.get("top5_R_share", float("nan"))),
            "top5_pos_share": float(r.get("top5_pos_share", float("nan"))),
            "worst_20trade_rolling_R": float(r.get("worst_20trade_rolling_R", float("nan"))),
            "yearly_min_avg_R": float(r.get("yearly_min_avg_R", float("nan"))),
            "yearly_max_avg_R": float(r.get("yearly_max_avg_R", float("nan"))),
            "breakeven_stop_hit_rate": float(r.get("breakeven_stop_hit_rate", 0.0)),
            "profit_lock_stop_hit_rate": float(
                r.get("profit_lock_stop_hit_rate", 0.0)
                if np.isfinite(r.get("profit_lock_stop_hit_rate", 0.0)) else 0.0),
            "max_hold_hit_rate": float(r.get("max_hold_hit_rate", 0.0)),
            "pct_MFE_ge_1R": float(r.get("pct_MFE_ge_1R", float("nan"))),
            "pct_MFE_ge_2R": float(r.get("pct_MFE_ge_2R", float("nan"))),
            "pct_MFE_ge_3R": float(r.get("pct_MFE_ge_3R", float("nan"))),
            "avg_R_given_MFE_ge_1R": float(r.get("avg_R_given_MFE_ge_1R", float("nan"))),
            "avg_R_given_MFE_ge_2R": float(r.get("avg_R_given_MFE_ge_2R", float("nan"))),
            "avg_R_given_MFE_ge_3R": float(r.get("avg_R_given_MFE_ge_3R", float("nan"))),
            "penalty": float(pen),
            "penalty_flags": ";".join(flags) if flags else "",
            "score_balanced": float(score_balanced),
            "score_mfe_capture": float(score_mfe),
        })
    return pd.DataFrame(rows, columns=_SCORED_COLS)


def _eligible_for_carry(scored: pd.DataFrame) -> pd.DataFrame:
    """N≥30, finite avg_R/PF/score_balanced. Slot-3 also requires PF>1, avg_R>0."""
    if scored.empty:
        return scored
    return scored[
        (scored["n"] >= 30)
        & scored["avg_R_net"].notna()
        & scored["pf_net"].notna()
    ].copy()


@dataclass
class _SlotPick:
    slot: str
    rationale: str
    family_diversity_penalty: float
    duplicate_behavior_flag: bool
    row: dict


def select_carried(
    metrics: pd.DataFrame, max_carry: int = 4
) -> pd.DataFrame:
    """Four-hypothesis carry selector. Returns carried-only DataFrame.

    For the rejected-top-performer table, call `select_carried_with_audit`.
    """
    carried, _rejected = select_carried_with_audit(metrics, max_carry=max_carry)
    return carried


def select_carried_with_audit(
    metrics: pd.DataFrame, max_carry: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selector that also returns the rejected-top-performer audit table.

    Returns:
        (carried_df, rejected_df).

    `rejected_df` contains, for each non-carried row that ranks in the
    top-10 by raw `avg_R_net` among eligible variants, a per-row reason
    string explaining why it was not picked (duplicate-behavior, did
    not meet slot-3 MFE-capture gate, slot already filled, etc.).
    """
    if metrics.empty:
        return (
            pd.DataFrame(columns=["slot"] + _SCORED_COLS
                         + ["family_diversity_penalty",
                            "duplicate_behavior_flag", "rationale"]),
            pd.DataFrame(columns=["exit_variant", "rank_by_avg_R",
                                  "avg_R_net", "reason"]),
        )

    scored = _score_table(metrics)
    eligible = _eligible_for_carry(scored)

    picks: list[_SlotPick] = []
    used_variants: set[str] = set()
    used_signatures: list[tuple] = []

    def _diversity_penalty(row: pd.Series) -> float:
        if not picks:
            return 0.0
        fam_overlap = sum(1 for p in picks if p.row["exit_family"] == row["exit_family"])
        kind_overlap = sum(1 for p in picks if p.row["trend_kind"] == row["trend_kind"])
        return float(0.10 * fam_overlap + 0.10 * kind_overlap)

    def _is_duplicate(row: pd.Series) -> bool:
        sig = _behavior_signature(row)
        return sig in used_signatures

    rejection_reasons: dict[str, str] = {}

    # ---------- slot 1 — raw trend control ----------
    pool = eligible[eligible["is_no_partial_trend"]].copy()
    pool = pool.sort_values("score_balanced", ascending=False)
    if not pool.empty and pool.iloc[0]["score_balanced"] > 0:
        r = pool.iloc[0]
        picks.append(_SlotPick(
            slot="1_raw_trend_control",
            rationale=(
                "best no-partial trend variant (raw 'let trend run' control). "
                "Carried even with fragile flags so validation can test the "
                "control hypothesis directly."
            ),
            family_diversity_penalty=0.0,
            duplicate_behavior_flag=False,
            row=r.to_dict(),
        ))
        used_variants.add(r["exit_variant"])
        used_signatures.append(_behavior_signature(r))

    # ---------- slot 2 — best balanced partial in F1/F2/F3 ----------
    pool2 = eligible[
        eligible["exit_family"].isin(PARTIAL_FAMILIES_FOR_SLOT2)
        & eligible["has_partial"]
        & ~eligible["exit_variant"].isin(used_variants)
    ].copy()
    pool2 = pool2[~pool2.apply(_is_duplicate, axis=1)] if not pool2.empty else pool2
    pool2 = pool2.sort_values("score_balanced", ascending=False)
    slot2_rejected_dups: list[str] = []
    if not pool2.empty and pool2.iloc[0]["score_balanced"] > 0:
        r = pool2.iloc[0]
        div_pen = _diversity_penalty(r)
        picks.append(_SlotPick(
            slot="2_best_balanced_partial",
            rationale=(
                f"best partial variant in F1/F2/F3 by penalty-adjusted "
                f"balanced score ({r['score_balanced']:+.3f}). F4-with-partial "
                "is excluded from this slot to avoid duplicating slot-1 behavior."
            ),
            family_diversity_penalty=div_pen,
            duplicate_behavior_flag=False,
            row=r.to_dict(),
        ))
        used_variants.add(r["exit_variant"])
        used_signatures.append(_behavior_signature(r))

    # ---------- slot 3 — MFE-capture / low-giveback ----------
    slot1 = picks[0].row if picks else None
    if slot1 is not None:
        slot1_cap = slot1.get("captured_MFE_ratio_cohort", float("nan"))
        slot1_gb = slot1.get("avg_giveback_R", float("nan"))
    else:
        slot1_cap = slot1_gb = float("nan")
    cap_floor = slot1_cap if np.isfinite(slot1_cap) else float("nan")

    pool3 = eligible[
        (eligible["pf_net"] > 1.0)
        & (eligible["avg_R_net"] > 0)
        & ~eligible["exit_variant"].isin(used_variants)
    ].copy()
    pool3 = pool3[~pool3.apply(_is_duplicate, axis=1)] if not pool3.empty else pool3
    if np.isfinite(cap_floor):
        pool3 = pool3[pool3["captured_MFE_ratio_cohort"] > cap_floor]
    if np.isfinite(slot1_gb):
        pool3 = pool3[pool3["avg_giveback_R"] < slot1_gb]
    pool3 = pool3.sort_values("score_mfe_capture", ascending=False)

    # Snapshot pre-pick state so the rejected-top builder can attribute
    # diversity-rule rejections truthfully.
    slot3_covered_kinds: set = {p.row.get("trend_kind") for p in picks}
    slot3_diverse_pool_size: int = 0
    slot3_total_pool_size: int = int(len(pool3))

    if len(picks) < max_carry and not pool3.empty:
        # Prefer trend kinds not yet covered.
        diverse = pool3[~pool3["trend_kind"].isin(slot3_covered_kinds)]
        slot3_diverse_pool_size = int(len(diverse))
        candidate = diverse if not diverse.empty else pool3
        r = candidate.iloc[0]
        if r["score_mfe_capture"] > 0:
            div_pen = _diversity_penalty(r)
            picks.append(_SlotPick(
                slot="3_mfe_capture_low_giveback",
                rationale=(
                    f"MFE-capture / low-giveback hypothesis. "
                    f"captured_MFE_ratio_cohort={r['captured_MFE_ratio_cohort']:+.3f} "
                    f"vs slot-1 {slot1_cap:+.3f} (gate strictly > slot1); "
                    f"avg_giveback_R={r['avg_giveback_R']:+.3f} "
                    f"vs slot-1 {slot1_gb:+.3f}. Selected for capture quality, "
                    "not raw avg_R."
                ),
                family_diversity_penalty=div_pen,
                duplicate_behavior_flag=False,
                row=r.to_dict(),
            ))
            used_variants.add(r["exit_variant"])
            used_signatures.append(_behavior_signature(r))

    # ---------- slot 4 — F8 profit-lock ladder, conditional ----------
    # Strictly higher bar than slot 3: explicit absolute capture floor,
    # explicit giveback drop, PF > 1.2 (vs slot-3's > 1.0), and a
    # median_R floor that rejects deeply-negative-tail variants.
    slot4_pool_size = 0
    slot4_candidates_with_reason: list[tuple[str, str]] = []
    if len(picks) < max_carry:
        pool4 = eligible[
            eligible["exit_family"].astype(str).str.startswith(SLOT4_FAMILY_PREFIX)
            & ~eligible["exit_variant"].isin(used_variants)
        ].copy()
        slot4_pool_size = int(len(pool4))
        if not pool4.empty:
            pool4 = pool4[~pool4.apply(_is_duplicate, axis=1)]

        cap_floor4 = max(
            SLOT4_CAP_FLOOR,
            (slot1_cap + SLOT4_MIN_CAP_OVER_SLOT1) if np.isfinite(slot1_cap) else SLOT4_CAP_FLOOR,
        )
        gb_ceiling4 = (slot1_gb - SLOT4_MIN_GB_DROP_R) if np.isfinite(slot1_gb) else float("inf")

        if not pool4.empty:
            qual = pool4[
                (pool4["captured_MFE_ratio_cohort"] >= cap_floor4)
                & (pool4["avg_giveback_R"] <= gb_ceiling4)
                & (pool4["pf_net"] > SLOT4_MIN_PF)
                & (pool4["avg_R_net"] > 0.0)
                & (pool4["median_R_net"] >= SLOT4_MEDIAN_FLOOR_R)
            ].copy()
            qual = qual.sort_values("score_mfe_capture", ascending=False)
            if not qual.empty and qual.iloc[0]["score_mfe_capture"] > 0:
                r = qual.iloc[0]
                div_pen = _diversity_penalty(r)
                picks.append(_SlotPick(
                    slot="4_profit_lock_capture",
                    rationale=(
                        f"F8 profit-lock-ladder carry. "
                        f"captured_MFE_ratio_cohort={r['captured_MFE_ratio_cohort']:.3f} "
                        f"≥ slot-4 floor {cap_floor4:.3f} (= max(0.30, slot1+0.05)); "
                        f"avg_giveback_R={r['avg_giveback_R']:+.3f} "
                        f"≤ slot1 {slot1_gb:+.3f} − {SLOT4_MIN_GB_DROP_R:.2f} R; "
                        f"PF={r['pf_net']:.2f} > {SLOT4_MIN_PF:.2f}; "
                        f"avg_R_net={r['avg_R_net']:+.3f}; "
                        f"median_R_net={r['median_R_net']:+.3f} ≥ {SLOT4_MEDIAN_FLOOR_R:+.2f}. "
                        f"profit_lock_stop_hit_rate="
                        f"{r.get('profit_lock_stop_hit_rate', float('nan')):.3f}."
                    ),
                    family_diversity_penalty=div_pen,
                    duplicate_behavior_flag=False,
                    row=r.to_dict(),
                ))
                used_variants.add(r["exit_variant"])
                used_signatures.append(_behavior_signature(r))
            else:
                # Record the closest miss so the report can describe it.
                # Score the failure mode of the top pool4 row by raw cap.
                if not pool4.empty:
                    top4 = pool4.sort_values(
                        "captured_MFE_ratio_cohort", ascending=False).iloc[0]
                    why: list[str] = []
                    if not (top4["captured_MFE_ratio_cohort"] >= cap_floor4):
                        why.append(
                            f"cap {top4['captured_MFE_ratio_cohort']:.3f} < "
                            f"{cap_floor4:.3f}")
                    if not (top4["avg_giveback_R"] <= gb_ceiling4):
                        why.append(
                            f"giveback {top4['avg_giveback_R']:+.3f} > "
                            f"{gb_ceiling4:+.3f}")
                    if not (top4["pf_net"] > SLOT4_MIN_PF):
                        why.append(f"PF {top4['pf_net']:.2f} ≤ {SLOT4_MIN_PF:.2f}")
                    if not (top4["avg_R_net"] > 0.0):
                        why.append(f"avg_R {top4['avg_R_net']:+.3f} ≤ 0")
                    if not (top4["median_R_net"] >= SLOT4_MEDIAN_FLOOR_R):
                        why.append(
                            f"median_R {top4['median_R_net']:+.3f} < "
                            f"{SLOT4_MEDIAN_FLOOR_R:+.2f}")
                    slot4_candidates_with_reason.append(
                        (top4["exit_variant"], "; ".join(why) or "no failed gate matched"))

    # ---------- carried frame ----------
    carried_rows: list[dict] = []
    for p in picks:
        out = {"slot": p.slot}
        out.update(p.row)
        out["family_diversity_penalty"] = p.family_diversity_penalty
        out["duplicate_behavior_flag"] = p.duplicate_behavior_flag
        out["rationale"] = p.rationale
        carried_rows.append(out)
    carried_df = pd.DataFrame(carried_rows)

    # ---------- rejected top performers ----------
    elig_sorted = eligible.sort_values("avg_R_net", ascending=False).reset_index(drop=True)
    top_n = min(10, len(elig_sorted))
    rejected_records: list[dict] = []
    for rank, (_, r) in enumerate(elig_sorted.head(top_n).iterrows(), 1):
        if r["exit_variant"] in used_variants:
            continue
        # Determine reason
        sig = _behavior_signature(r)
        reasons: list[str] = []
        if sig in used_signatures:
            reasons.append("duplicate-behavior with an already-carried slot")
        if (
            r["exit_family"] in PARTIAL_FAMILIES_FOR_SLOT2
            and r["has_partial"]
            and len(picks) >= 2
            and not any(p.slot == "2_best_balanced_partial" for p in picks if p.row["exit_variant"] == r["exit_variant"])
        ):
            reasons.append("partial slot already filled by higher balanced score")
        if r["is_no_partial_trend"] and len(picks) >= 1:
            reasons.append("trend-control slot already filled")
        passed_cap_gate = (
            not np.isfinite(cap_floor)
            or r["captured_MFE_ratio_cohort"] > cap_floor
        )
        passed_gb_gate = (
            not np.isfinite(slot1_gb)
            or r["avg_giveback_R"] < slot1_gb
        )
        if not passed_cap_gate:
            reasons.append(
                f"failed slot-3 MFE-capture gate "
                f"(cap {r['captured_MFE_ratio_cohort']:+.3f} ≤ slot1 {cap_floor:+.3f})"
            )
        if not passed_gb_gate:
            reasons.append(
                f"failed slot-3 giveback gate (giveback {r['avg_giveback_R']:+.3f} "
                f"≥ slot1 {slot1_gb:+.3f})"
            )
        # Diversity rule: variant cleared both slot-3 gates AND shares its
        # trend_kind with an already-carried slot AND a non-overlapping-kind
        # candidate was available in pool3. The selector preferred the diverse
        # pool, so this row was deprioritised even though it qualifies.
        if (
            passed_cap_gate
            and passed_gb_gate
            and r["trend_kind"] in slot3_covered_kinds
            and slot3_diverse_pool_size > 0
            and sig not in used_signatures
        ):
            reasons.append(
                f"lost to slot-3 diversity preference "
                f"(trend_kind={r['trend_kind']} already covered by an earlier "
                f"slot; a non-overlapping-kind candidate was available)"
            )
        if not reasons:
            reasons.append("did not score above carried picks in any slot")
        rejected_records.append({
            "rank_by_avg_R": rank,
            "exit_variant": r["exit_variant"],
            "exit_family": r["exit_family"],
            "trend_kind": r["trend_kind"],
            "n": int(r["n"]),
            "avg_R_net": float(r["avg_R_net"]),
            "pf_net": float(r["pf_net"]),
            "median_R_net": float(r["median_R_net"]),
            "captured_MFE_ratio_cohort": float(r["captured_MFE_ratio_cohort"]),
            "avg_giveback_R": float(r["avg_giveback_R"]),
            "top5_pos_share": float(r["top5_pos_share"]),
            "yearly_min_avg_R": float(r["yearly_min_avg_R"]),
            "penalty_flags": r["penalty_flags"],
            "reason": " | ".join(reasons),
        })
    rejected_df = pd.DataFrame(rejected_records)

    return carried_df, rejected_df


# ---------- top-level discovery driver ---------------------------------------

def run_discovery_v2(
    train_panel: pd.DataFrame,
    daily_master: pd.DataFrame,
    grid: Iterable[ExitConfigV2],
    out_dir: Path,
    suffix: str = "v2",
) -> dict:
    """Run the full grid on TRAIN, write metrics + carried CSVs.

    `train_panel` MUST already be filtered to the train slice; this
    function does not consult `splits`. Returns a small summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_ticker = _by_ticker_master(daily_master)
    metrics_rows: list[dict] = []
    yearly_rows: list[dict] = []

    grid = list(grid)
    n_variants = len(grid)
    for i, cfg in enumerate(grid, 1):
        trade_df = simulate_variant_on_panel(cfg, train_panel, by_ticker)
        m = variant_metrics(trade_df, cfg)
        for yrec in m.pop("yearly_records", []):
            yearly_rows.append({"exit_variant": cfg.name, "exit_family": cfg.family, **yrec})
        metrics_rows.append(m)
        if i == 1 or i % 10 == 0 or i == n_variants:
            print(f"  [{i:>3}/{n_variants}] {cfg.name:<70} N={m['n']} "
                  f"avg_R={m['avg_R_net'] if isinstance(m['avg_R_net'], float) else float('nan'):+.3f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / f"sbt_1700_exit_discovery_{suffix}_train.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[discovery_v2] wrote {metrics_path}  ({len(metrics_df)} variants)")

    yearly_df = pd.DataFrame(yearly_rows)
    yearly_path = out_dir / f"sbt_1700_exit_discovery_{suffix}_yearly.csv"
    yearly_df.to_csv(yearly_path, index=False)
    print(f"[discovery_v2] wrote {yearly_path}  ({len(yearly_df)} variant×year rows)")

    carried = select_carried(metrics_df, max_carry=3)
    carried_path = out_dir / f"sbt_1700_exit_discovery_{suffix}_carried.csv"
    carried.to_csv(carried_path, index=False)
    print(f"[discovery_v2] wrote {carried_path}  ({len(carried)} carried)")

    return {
        "n_variants": int(len(metrics_df)),
        "metrics_path": str(metrics_path),
        "yearly_path": str(yearly_path),
        "carried_path": str(carried_path),
        "n_carried": int(len(carried)),
    }


# ---------- family summary + markdown rendering ------------------------------

def family_summary_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-family aggregate (count + medians + best-of-family by avg_R_net).

    The "best" within each family is the row that maximises `avg_R_net`
    among rows with `n >= 30`. Variants below the n-floor are kept in
    counts but never become "best" picks.
    """
    if metrics_df.empty:
        return metrics_df.copy()
    out: list[dict] = []
    for fam, g in metrics_df.groupby("exit_family"):
        eligible = g[g["n"] >= 30]
        best = (eligible.sort_values("avg_R_net", ascending=False).iloc[0]
                if not eligible.empty else None)
        out.append({
            "exit_family": fam,
            "n_variants": int(len(g)),
            "n_eligible": int(len(eligible)),
            "median_n": float(g["n"].median()),
            "median_avg_R": float(g["avg_R_net"].median()),
            "median_pf_net": float(g["pf_net"].median()),
            "median_total_R": float(g["total_R_net"].median()),
            "best_variant": best["exit_variant"] if best is not None else None,
            "best_trend_kind": best["trend_kind"] if best is not None else None,
            "best_n": int(best["n"]) if best is not None else 0,
            "best_avg_R": float(best["avg_R_net"]) if best is not None else float("nan"),
            "best_pf_net": float(best["pf_net"]) if best is not None else float("nan"),
            "best_total_R": float(best["total_R_net"]) if best is not None else float("nan"),
            "best_captured_MFE_ratio": (
                float(best["captured_MFE_ratio"]) if best is not None else float("nan")),
            "best_top5_R_share": (
                float(best["top5_R_share"]) if best is not None else float("nan")),
        })
    return (pd.DataFrame(out)
            .sort_values("best_avg_R", ascending=False, na_position="last")
            .reset_index(drop=True))


def _fmt_num(x, fmt: str = "{:+.3f}") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return fmt.format(v)


def render_recommendation_md(
    metrics_df: pd.DataFrame,
    family_df: pd.DataFrame,
    carried_df: pd.DataFrame,
    rejected_df: pd.DataFrame | None = None,
) -> str:
    """Markdown report for the discovery_grid phase."""
    lines: list[str] = []
    lines.append("# SBT-1700 RESET — Exit Discovery (TRAIN only)")
    lines.append("")
    lines.append(
        "> Discovery output. **Discovery pass ≠ validated.** The validation "
        "phase runs only the carries listed below on the 2025-07-01 → "
        "2025-12-31 slice and freezes one for the locked test readout."
    )
    lines.append("")

    lines.append("## Selection rule (4 hypotheses, ≤4 carries)")
    lines.append("")
    lines.append(
        "- **Slot 1 — raw trend control**: best no-partial trend variant "
        "(F0 or F4 without partial), ranked by penalty-adjusted balanced "
        "score. Carried as-is so the validation slice can test the 'let "
        "the trend run' control hypothesis directly.\n"
        "- **Slot 2 — best balanced partial**: best partial variant "
        "restricted to F1 / F2 / F3 (F4-with-partial is excluded to avoid "
        "duplicating slot-1 behavior). Ranked by the same balanced score.\n"
        "- **Slot 3 — MFE-capture / low-giveback**: required even if "
        "avg_R is lower than slot 1's. Eligible iff PF > 1, avg_R > 0, "
        "N ≥ 30, `captured_MFE_ratio_cohort` strictly greater than "
        "slot-1's and `avg_giveback_R` strictly lower than slot 1's. "
        "Ranked by an MFE-capture-weighted score. The grid was extended "
        "with F5 (MFE-giveback w/ activation), F6 (tight ATR trail) and "
        "F7 (structure reclaim + EMA10 sibling) so that this slot has "
        "candidates the F0–F4 grammar could not produce.\n"
        "- **Slot 4 — F8 profit-lock ladder, conditional**: F8 family only "
        f"(profit_lock_ladder ratchets stop at MFE milestones). Strictly "
        f"higher bar than slot 3 — captured_MFE_ratio_cohort ≥ "
        f"max({SLOT4_CAP_FLOOR:.2f}, slot1 + {SLOT4_MIN_CAP_OVER_SLOT1:+.2f}); "
        f"avg_giveback_R ≤ slot1 − {SLOT4_MIN_GB_DROP_R:.2f} R; "
        f"PF > {SLOT4_MIN_PF:.2f}; avg_R > 0; "
        f"median_R ≥ {SLOT4_MEDIAN_FLOOR_R:+.2f}. If no F8 variant clears "
        "all five gates, slot 4 is left empty and the report records a "
        "negative finding rather than carrying a marginal candidate.\n"
        "- Penalties (shared across slots): top-5 positive-R share > "
        "0.50, giveback / MFE > 0.60, mean-vs-median sign mismatch, "
        "BE-cuts-winners (BE-on with hit-rate > 0.20 AND "
        "captured_MFE_ratio_cohort < no-BE sibling).\n"
        "- N < 30 zeros the score; 30 ≤ N < 60 takes 0.20.\n"
        "- **Duplicate-behavior gate**: a variant sharing "
        "(trend_kind, sl_atr, partial_size, partial_R, max_hold, "
        "profit_lock_ladder) with an already-carried slot is rejected from "
        "later slots.\n"
        "- Slot score (1 & 2) = `avg_R_net × max(0, 1 − Σ penalties)`. "
        "Slot 3 & 4 score = same skeleton plus "
        "`+0.50 × captured_MFE_ratio_cohort` and "
        "`−0.30 × max(0, giveback_over_mfe − 0.50)`."
    )
    lines.append("")

    lines.append("## Recommended carries (≤4)")
    lines.append("")
    slot3_present = (carried_df is not None and not carried_df.empty
                     and (carried_df.get("slot") == "3_mfe_capture_low_giveback").any())
    slot4_present = (carried_df is not None and not carried_df.empty
                     and (carried_df.get("slot") == "4_profit_lock_capture").any())
    slot1_row = (carried_df.iloc[0].to_dict()
                 if carried_df is not None and not carried_df.empty
                 else None)
    if not slot3_present and slot1_row is not None:
        lines.append(
            "> ⚠ **Slot 3 (MFE-capture / low-giveback) was NOT filled.** "
            f"Slot 1 captured_MFE_ratio_cohort = "
            f"{_fmt_num(slot1_row.get('captured_MFE_ratio_cohort'), '{:.3f}')}, "
            "and no eligible variant in the grid — including the F5/F6/F7 "
            "extension (MFE-giveback w/ activation, tight ATR trail, "
            "structure reclaim) — strictly exceeds that while also lowering "
            "avg_giveback_R. **Read this as a negative finding about the "
            "TRAIN slice, not the selector**: even with the explicit "
            "MFE-capture grammar, the slice does not contain a configuration "
            "that simultaneously captures more MFE and gives back less."
        )
        lines.append("")
    if not slot4_present and not metrics_df.empty:
        f8_rows = metrics_df[metrics_df["exit_family"].astype(str).str.startswith("F8")]
        if not f8_rows.empty:
            f8_eligible = f8_rows[f8_rows["n"] >= 30]
            top_cap = float(f8_eligible["captured_MFE_ratio_cohort"].max()) if not f8_eligible.empty else float("nan")
            top_pf = float(f8_eligible["pf_net"].max()) if not f8_eligible.empty else float("nan")
            slot1_cap_v = (slot1_row.get("captured_MFE_ratio_cohort")
                           if slot1_row else float("nan"))
            lines.append(
                "> ⚠ **Slot 4 (F8 profit-lock ladder) was NOT filled.** "
                f"The F8 family ran {len(f8_rows)} variants (4 sub-families × "
                "8 cells: sl × partial × hold) with explicit profit-lock "
                "ladders that ratchet the runner stop at +1R / +2R / +3R "
                "MFE milestones. Among N ≥ 30 F8 variants the best "
                "captured_MFE_ratio_cohort observed was "
                f"{_fmt_num(top_cap, '{:.3f}')} (slot 4 floor: "
                f"{max(SLOT4_CAP_FLOOR, (slot1_cap_v + SLOT4_MIN_CAP_OVER_SLOT1) if isinstance(slot1_cap_v, float) and np.isfinite(slot1_cap_v) else SLOT4_CAP_FLOOR):.3f}); "
                f"the best PF among F8 was {_fmt_num(top_pf, '{:.2f}')} "
                f"(gate > {SLOT4_MIN_PF:.2f}). **Read this as: in this "
                "TRAIN slice, MFE capture conflicts with trend "
                "continuation — locking accumulated profit early enough "
                "to materially raise capture also clips the trades that "
                "carry the avg_R, so the slice does not admit a profit-"
                "lock ladder that simultaneously raises capture and "
                "preserves expectancy.** Validation will run with the "
                "non-F8 carries only."
            )
            lines.append("")
    if carried_df is None or carried_df.empty:
        lines.append("_No variant cleared the eligibility gate._")
    else:
        cols = ["slot", "exit_variant", "exit_family", "trend_kind", "n",
                "avg_R_net", "median_R_net", "pf_net",
                "captured_MFE_ratio_cohort", "avg_giveback_R",
                "giveback_over_mfe", "top5_pos_share",
                "worst_20trade_rolling_R", "yearly_min_avg_R",
                "score_balanced", "score_mfe_capture",
                "family_diversity_penalty", "duplicate_behavior_flag",
                "penalty_flags", "rationale"]
        present = [c for c in cols if c in carried_df.columns]
        disp = carried_df[present].copy()
        for c in ("avg_R_net", "median_R_net", "avg_giveback_R",
                  "worst_20trade_rolling_R", "yearly_min_avg_R",
                  "score_balanced", "score_mfe_capture"):
            if c in disp.columns:
                disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:+.3f}"))
        for c in ("captured_MFE_ratio_cohort", "giveback_over_mfe",
                  "top5_pos_share", "family_diversity_penalty"):
            if c in disp.columns:
                disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:.3f}"))
        if "pf_net" in disp.columns:
            disp["pf_net"] = disp["pf_net"].map(lambda v: _fmt_num(v, "{:.2f}"))
        lines.append(disp.to_markdown(index=False))
    lines.append("")

    if rejected_df is not None and not rejected_df.empty:
        lines.append("## Rejected top performers")
        lines.append("")
        lines.append(
            "Top-10 variants by raw `avg_R_net` that were *not* carried. "
            "Each row's `reason` column says why — duplicate-behavior, "
            "slot already filled, or failure of the slot-3 MFE-capture / "
            "giveback gate. Use this table to argue with the selection."
        )
        lines.append("")
        rcols = ["rank_by_avg_R", "exit_variant", "exit_family", "trend_kind",
                 "n", "avg_R_net", "pf_net", "median_R_net",
                 "captured_MFE_ratio_cohort", "avg_giveback_R",
                 "top5_pos_share", "yearly_min_avg_R", "penalty_flags",
                 "reason"]
        rpresent = [c for c in rcols if c in rejected_df.columns]
        rdisp = rejected_df[rpresent].copy()
        for c in ("avg_R_net", "median_R_net", "avg_giveback_R",
                  "yearly_min_avg_R"):
            if c in rdisp.columns:
                rdisp[c] = rdisp[c].map(lambda v: _fmt_num(v, "{:+.3f}"))
        for c in ("captured_MFE_ratio_cohort", "top5_pos_share"):
            if c in rdisp.columns:
                rdisp[c] = rdisp[c].map(lambda v: _fmt_num(v, "{:.3f}"))
        if "pf_net" in rdisp.columns:
            rdisp["pf_net"] = rdisp["pf_net"].map(lambda v: _fmt_num(v, "{:.2f}"))
        lines.append(rdisp.to_markdown(index=False))
        lines.append("")

    lines.append("## Family summary")
    lines.append("")
    if family_df is None or family_df.empty:
        lines.append("_(no families)_")
    else:
        cols = ["exit_family", "n_variants", "n_eligible",
                "median_avg_R", "median_pf_net", "median_total_R",
                "best_variant", "best_trend_kind", "best_n",
                "best_avg_R", "best_pf_net", "best_total_R",
                "best_captured_MFE_ratio", "best_top5_R_share"]
        present = [c for c in cols if c in family_df.columns]
        disp = family_df[present].copy()
        for c in ("median_avg_R", "best_avg_R"):
            if c in disp.columns:
                disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:+.3f}"))
        for c in ("median_pf_net", "best_pf_net",
                  "best_captured_MFE_ratio", "best_top5_R_share"):
            if c in disp.columns:
                disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:.2f}"))
        for c in ("median_total_R", "best_total_R"):
            if c in disp.columns:
                disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:+.1f}"))
        lines.append(disp.to_markdown(index=False))
    lines.append("")

    lines.append("## Top 10 variants by avg_R_net (eligible only)")
    lines.append("")
    if metrics_df.empty:
        lines.append("_(empty)_")
    else:
        eligible = metrics_df[metrics_df["n"] >= 30].copy()
        if eligible.empty:
            lines.append("_No variant met the n ≥ 30 floor._")
        else:
            top = (eligible.sort_values("avg_R_net", ascending=False)
                   .head(10).copy())
            cols = ["exit_variant", "exit_family", "trend_kind", "n", "wr",
                    "pf_net", "avg_R_net", "median_R_net", "total_R_net",
                    "top5_R_share", "captured_MFE_ratio",
                    "breakeven_stop_hit_rate", "max_hold_hit_rate"]
            present = [c for c in cols if c in top.columns]
            disp = top[present].copy()
            for c in ("wr", "top5_R_share", "captured_MFE_ratio",
                      "breakeven_stop_hit_rate", "max_hold_hit_rate"):
                if c in disp.columns:
                    disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:.3f}"))
            if "pf_net" in disp.columns:
                disp["pf_net"] = disp["pf_net"].map(lambda v: _fmt_num(v, "{:.2f}"))
            for c in ("avg_R_net", "median_R_net"):
                if c in disp.columns:
                    disp[c] = disp[c].map(lambda v: _fmt_num(v, "{:+.3f}"))
            if "total_R_net" in disp.columns:
                disp["total_R_net"] = disp["total_R_net"].map(
                    lambda v: _fmt_num(v, "{:+.1f}"))
            lines.append(disp.to_markdown(index=False))
    lines.append("")

    lines.append(
        "## Decision language\n\n"
        "- This file is a TRAIN-only diagnostic. It surfaces candidates; it "
        "does not validate them.\n"
        "- The validation phase runs only the carried exits on the "
        "2025-07-01 → 2025-12-31 slice and freezes one for the locked test "
        "readout.\n"
        "- Re-running discovery does not re-pick anything by itself; the "
        "validation phase reads the carry list passed in by the operator."
    )
    return "\n".join(lines) + "\n"


def run_discovery_grid(
    train_panel: pd.DataFrame,
    daily_master: pd.DataFrame,
    grid: Iterable[ExitConfigV2],
    out_dir: Path,
) -> dict:
    """Spec'd entry point for the `discovery_grid` reset phase.

    Output filenames (locked by reset spec):
        sbt_1700_exit_discovery_grid.csv
        sbt_1700_exit_discovery_family_summary.csv
        sbt_1700_exit_discovery_recommended_validation_exits.md
        sbt_1700_exit_discovery_yearly.csv          (supplementary)
        sbt_1700_exit_discovery_carried.csv         (supplementary, machine-readable)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_ticker = _by_ticker_master(daily_master)
    metrics_rows: list[dict] = []
    yearly_rows: list[dict] = []

    grid = list(grid)
    n_variants = len(grid)
    print(f"[discovery_grid] simulating {n_variants} variants on "
          f"{len(train_panel)} train rows")
    for i, cfg in enumerate(grid, 1):
        trade_df = simulate_variant_on_panel(cfg, train_panel, by_ticker)
        m = variant_metrics(trade_df, cfg)
        for yrec in m.pop("yearly_records", []):
            yearly_rows.append({
                "exit_variant": cfg.name, "exit_family": cfg.family, **yrec})
        metrics_rows.append(m)
        if i == 1 or i % 10 == 0 or i == n_variants:
            avg_R = m["avg_R_net"]
            avg_str = (f"{avg_R:+.3f}"
                       if isinstance(avg_R, float) and np.isfinite(avg_R)
                       else "  nan")
            print(f"  [{i:>3}/{n_variants}] {cfg.name:<70} "
                  f"N={m['n']:>4} avg_R={avg_str}")

    metrics_df = pd.DataFrame(metrics_rows)
    fam_df = family_summary_df(metrics_df)
    carried_df, rejected_df = select_carried_with_audit(metrics_df, max_carry=3)
    md = render_recommendation_md(metrics_df, fam_df, carried_df, rejected_df)

    grid_path = out_dir / "sbt_1700_exit_discovery_grid.csv"
    fam_path = out_dir / "sbt_1700_exit_discovery_family_summary.csv"
    md_path = out_dir / "sbt_1700_exit_discovery_recommended_validation_exits.md"
    yearly_path = out_dir / "sbt_1700_exit_discovery_yearly.csv"
    carried_csv_path = out_dir / "sbt_1700_exit_discovery_carried.csv"
    rejected_csv_path = out_dir / "sbt_1700_exit_discovery_rejected_top.csv"

    metrics_df.to_csv(grid_path, index=False)
    fam_df.to_csv(fam_path, index=False)
    md_path.write_text(md)
    pd.DataFrame(yearly_rows).to_csv(yearly_path, index=False)
    carried_df.to_csv(carried_csv_path, index=False)
    rejected_df.to_csv(rejected_csv_path, index=False)

    print(f"[discovery_grid] wrote {grid_path}")
    print(f"[discovery_grid] wrote {fam_path}")
    print(f"[discovery_grid] wrote {md_path}")
    print(f"[discovery_grid] wrote {yearly_path}  ({len(yearly_rows)} rows)")
    print(f"[discovery_grid] wrote {carried_csv_path}  ({len(carried_df)} carried)")
    print(f"[discovery_grid] wrote {rejected_csv_path}  ({len(rejected_df)} rejected-top)")
    return {
        "n_variants": int(len(metrics_df)),
        "n_carried": int(len(carried_df)),
        "n_rejected_top": int(len(rejected_df)),
        "grid_csv": str(grid_path),
        "family_csv": str(fam_path),
        "recommendation_md": str(md_path),
        "yearly_csv": str(yearly_path),
        "carried_csv": str(carried_csv_path),
        "rejected_csv": str(rejected_csv_path),
    }
