"""Lindsay Trident projection probe over mb_scanner above_mb_birth events.

Read-only. Does NOT import or modify mb_scanner / decision_engine* /
decision_engine_lab / nyxexpansion. Reads scanner event parquets and the
extfeed master parquet, computes D = B*C/A per event (A=ll_price,
B=hh_close, C=zone_high), then evaluates whether the forward bars
reached D within fixed horizons.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

FAMILIES = ["mb_5h", "mb_1d", "mb_1w", "mb_1M",
            "bb_5h", "bb_1d", "bb_1w", "bb_1M"]
HORIZONS_DAYS = [5, 10, 20, 30]

EXTFEED_PATH = OUTPUT / "extfeed_intraday_1h_3y_master.parquet"
DETAIL_PATH = OUTPUT / "trident_probe_mb_3y.parquet"
SUMMARY_PATH = OUTPUT / "trident_probe_mb_3y.md"


def load_birth_events() -> pd.DataFrame:
    frames = []
    for fam in FAMILIES:
        p = OUTPUT / f"mb_scanner_events_{fam}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df["event_type"] == "above_mb_birth"].copy()
        df["family"] = fam
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def normalize_event_dates(series: pd.Series) -> np.ndarray:
    s = pd.to_datetime(series)
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    s = s.dt.normalize()
    return s.to_numpy().astype("datetime64[ns]")


def build_daily_hl_lookup(extfeed_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.read_parquet(extfeed_path, columns=["ticker", "ts_istanbul", "high", "low"])
    df["bar_date"] = df["ts_istanbul"].dt.tz_localize(None).dt.normalize()
    daily = (df.groupby(["ticker", "bar_date"], as_index=False, sort=True)
               .agg(high=("high", "max"), low=("low", "min")))
    daily = daily.sort_values(["ticker", "bar_date"], kind="mergesort")
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ticker, grp in daily.groupby("ticker", sort=False):
        out[ticker] = (
            grp["bar_date"].to_numpy().astype("datetime64[ns]"),
            grp["high"].to_numpy(dtype=float),
            grp["low"].to_numpy(dtype=float),
        )
    return out


def compute_forward_highs(
    events: pd.DataFrame,
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    horizons: list[int],
) -> dict[int, np.ndarray]:
    n = len(events)
    out = {h: np.full(n, np.nan, dtype=float) for h in horizons}
    ed = events["_event_date64"].to_numpy()
    pos = np.arange(n)
    work = pd.DataFrame({"ticker": events["ticker"].to_numpy(), "ed": ed, "pos": pos})
    for ticker, grp in work.groupby("ticker", sort=False):
        info = lookup.get(ticker)
        if info is None:
            continue
        dates, highs, _lows = info
        ed_grp = grp["ed"].to_numpy()
        idxs = np.searchsorted(dates, ed_grp, side="right")
        positions = grp["pos"].to_numpy()
        for h in horizons:
            ends = np.minimum(idxs + h, len(dates))
            for k in range(len(idxs)):
                s, e = idxs[k], ends[k]
                if e > s:
                    out[h][positions[k]] = highs[s:e].max()
    return out


def compute_days_to_D(
    events: pd.DataFrame,
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_horizon: int,
) -> dict[str, np.ndarray]:
    """For each event, walk forward day by day up to max_horizon.

    loose_day = first trading day where forward high reaches D.
    strict_day = same as loose_day BUT only valid if the structural invalidation
                 level was not violated on or before that day. Conservative tie:
                 if low<inv and high>=D on the same bar, break wins (not a hit).
    break_day  = first trading day where low < structural_invalidation_low.
    bars_seen  = number of forward trading bars actually available (<= max_horizon).
    """
    n = len(events)
    loose_day = np.full(n, np.nan, dtype=float)
    strict_day = np.full(n, np.nan, dtype=float)
    break_day = np.full(n, np.nan, dtype=float)
    bars_seen = np.zeros(n, dtype=np.int32)

    ed = events["_event_date64"].to_numpy()
    tickers = events["ticker"].to_numpy()
    D_arr = events["D_target"].to_numpy(dtype=float)
    inv_arr = events["structural_invalidation_low"].to_numpy(dtype=float)
    pos = np.arange(n)

    work = pd.DataFrame({"ticker": tickers, "ed": ed, "pos": pos})
    for ticker, grp in work.groupby("ticker", sort=False):
        info = lookup.get(ticker)
        if info is None:
            continue
        dates, highs, lows = info
        ed_grp = grp["ed"].to_numpy()
        positions = grp["pos"].to_numpy()
        idxs = np.searchsorted(dates, ed_grp, side="right")
        for k in range(len(idxs)):
            i = positions[k]
            start = idxs[k]
            end = min(start + max_horizon, len(dates))
            bars_seen[i] = end - start
            if end <= start:
                continue
            D_i = D_arr[i]
            inv_i = inv_arr[i]
            inv_active = np.isfinite(inv_i)

            ld = -1
            sd = -1
            bd = -1
            for k2 in range(start, end):
                day = k2 - start + 1
                broken_today = inv_active and (lows[k2] < inv_i)
                hit_today = highs[k2] >= D_i
                if ld < 0 and hit_today:
                    ld = day
                if bd < 0 and broken_today:
                    bd = day
                if sd < 0 and bd < 0 and hit_today:
                    sd = day
                if ld >= 0 and (sd >= 0 or bd >= 0):
                    break
            if ld >= 0:
                loose_day[i] = ld
            if sd >= 0:
                strict_day[i] = sd
            if bd >= 0:
                break_day[i] = bd

    return {
        "days_to_D_loose": loose_day,
        "days_to_D_strict": strict_day,
        "structural_break_day": break_day,
        "forward_bars_available": bars_seen,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x*100:.1f}%"


def fmt_atr(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:+.2f}"


def main() -> None:
    print("[trident-probe] loading mb_scanner above_mb_birth events ...", flush=True)
    events = load_birth_events()
    print(f"  total events: {len(events):,}", flush=True)

    print("[trident-probe] building forward daily high/low lookup from extfeed ...", flush=True)
    lookup = build_daily_hl_lookup(EXTFEED_PATH)
    print(f"  tickers in lookup: {len(lookup):,}", flush=True)

    A = events["ll_price"].astype(float).to_numpy()
    B = events["hh_close"].astype(float).to_numpy()
    C = events["zone_high"].astype(float).to_numpy()
    atr = events["atr_at_event"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        D = (B * C) / A
        D_minus_B_atr = (D - B) / atr
        B_to_D_pct = (D / B) - 1.0

    events["A_price"] = A
    events["B_price"] = B
    events["C_price"] = C
    events["D_target"] = D
    events["D_minus_B_atr"] = D_minus_B_atr
    events["B_to_D_pct"] = B_to_D_pct

    events["_event_date64"] = normalize_event_dates(events["event_bar_date"])

    print("[trident-probe] computing forward MFE per horizon ...", flush=True)
    forward = compute_forward_highs(events, lookup, HORIZONS_DAYS)

    for h in HORIZONS_DAYS:
        fh = forward[h]
        events[f"forward_high_{h}d"] = fh
        with np.errstate(divide="ignore", invalid="ignore"):
            touched = np.where(np.isnan(fh), np.nan, (fh >= D).astype(float))
            gap_atr = (D - fh) / atr
            mfe_pct = (fh / B) - 1.0
        events[f"D_touched_{h}d"] = touched
        events[f"gap_to_D_atr_{h}d"] = gap_atr
        events[f"forward_mfe_pct_{h}d"] = mfe_pct

    print("[trident-probe] computing days-to-D with structural-invalidation guard ...", flush=True)
    max_h = max(HORIZONS_DAYS)
    dtd = compute_days_to_D(events, lookup, max_h)
    events["days_to_D_loose"] = dtd["days_to_D_loose"]
    events["days_to_D_strict"] = dtd["days_to_D_strict"]
    events["structural_break_day"] = dtd["structural_break_day"]
    events["forward_bars_available"] = dtd["forward_bars_available"]
    events["D_pct"] = events["B_to_D_pct"] * 100.0
    events["structural_invalidation_pct_below_B"] = (
        (events["structural_invalidation_low"].astype(float) / B) - 1.0
    ) * 100.0
    valid_window = events["forward_bars_available"] >= max_h
    outcome = np.where(
        ~valid_window, "edge_truncated",
        np.where(events["days_to_D_strict"].notna(), "hit_strict",
                 np.where(events["structural_break_day"].notna(),
                          "broken_before_hit", "neither_in_window"))
    )
    events["outcome_strict"] = outcome

    keep = [
        "ticker", "family", "setup_family",
        "event_bar_date", "event_ts",
        "A_price", "B_price", "C_price", "D_target",
        "D_pct", "D_minus_B_atr", "B_to_D_pct",
        "atr_at_event", "atr_pct_at_event",
        "zone_width_pct", "zone_age_bars",
        "bos_distance_atr_at_event", "vol_ratio_20_at_event",
        "structural_invalidation_low", "structural_invalidation_pct_below_B",
        "days_to_D_loose", "days_to_D_strict",
        "structural_break_day", "forward_bars_available", "outcome_strict",
    ]
    for h in HORIZONS_DAYS:
        keep += [f"forward_high_{h}d", f"D_touched_{h}d",
                 f"gap_to_D_atr_{h}d", f"forward_mfe_pct_{h}d"]

    detail = events[keep].copy()
    detail.to_parquet(DETAIL_PATH, index=False)
    print(f"[trident-probe] wrote {DETAIL_PATH} ({len(detail):,} rows)", flush=True)

    lines: list[str] = []
    lines.append("# Trident Probe — mb_scanner above_mb_birth events")
    lines.append("")
    lines.append(f"_Generated: {pd.Timestamp.utcnow().isoformat(timespec='seconds')}Z_")
    lines.append("")
    lines.append("Formula: **D = B · C / A** where")
    lines.append("- **A** = `ll_price` (quartet base low, mb_scanner output)")
    lines.append("- **B** = `hh_close` (BoS confirmation close)")
    lines.append("- **C** = `zone_high` (zone ceiling / retest reference)")
    lines.append("")
    lines.append(f"Total events: **{len(detail):,}** (`above_mb_birth` only)")
    lines.append("")
    lines.append("Forward MFE = max of daily-aggregated 1h `high` from extfeed, "
                 "look-forward N trading days from `event_bar_date` (event bar excluded).")
    lines.append("ATR sourced verbatim from scanner `atr_at_event` (no recomputation).")
    lines.append("")

    lines.append("## Setup geometry (D's projected distance from B)")
    lines.append("")
    geo_headers = ["family", "n", "median(D-B)/atr", "median (D/B-1)%", "p90 (D-B)/atr"]
    geo_rows = []
    for fam in FAMILIES:
        sub = detail[detail["family"] == fam]
        if len(sub) == 0:
            continue
        med_atr = sub["D_minus_B_atr"].median()
        med_pct = sub["B_to_D_pct"].median() * 100.0
        p90_atr = sub["D_minus_B_atr"].quantile(0.90)
        geo_rows.append([fam, f"{len(sub):,}",
                         fmt_atr(med_atr), f"{med_pct:+.2f}%",
                         fmt_atr(p90_atr)])
    lines.extend(md_table(geo_headers, geo_rows))
    lines.append("")

    lines.append("## D-touch rate by family × horizon")
    lines.append("")
    h_headers = ["family", "n"] + [f"D_touch_{h}d" for h in HORIZONS_DAYS] \
        + [f"med_gap_atr_{h}d" for h in HORIZONS_DAYS]
    h_rows = []
    for fam in FAMILIES:
        sub = detail[detail["family"] == fam]
        if len(sub) == 0:
            continue
        row = [fam, f"{len(sub):,}"]
        for h in HORIZONS_DAYS:
            t = sub[f"D_touched_{h}d"]
            mask = t.notna()
            if mask.sum() == 0:
                row.append("n/a")
            else:
                row.append(fmt_pct(float(t[mask].mean())))
        for h in HORIZONS_DAYS:
            row.append(fmt_atr(float(sub[f"gap_to_D_atr_{h}d"].median())))
        h_rows.append(row)
    lines.extend(md_table(h_headers, h_rows))
    lines.append("")

    lines.append("## D-touch rate by year (all families)")
    lines.append("")
    detail["_year"] = pd.to_datetime(detail["event_bar_date"]).dt.year
    y_headers = ["year", "n"] + [f"D_touch_{h}d" for h in HORIZONS_DAYS]
    y_rows = []
    for yr in sorted(detail["_year"].unique()):
        sub = detail[detail["_year"] == yr]
        row = [str(int(yr)), f"{len(sub):,}"]
        for h in HORIZONS_DAYS:
            t = sub[f"D_touched_{h}d"]
            mask = t.notna()
            if mask.sum() == 0:
                row.append("n/a")
            else:
                row.append(fmt_pct(float(t[mask].mean())))
        y_rows.append(row)
    lines.extend(md_table(y_headers, y_rows))
    lines.append("")

    lines.append("## D-touch rate by D-distance bucket (D−B in ATR units, all families)")
    lines.append("")
    bins = [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf]
    labels = ["<0.5", "0.5–1.0", "1.0–2.0", "2.0–4.0", ">4.0"]
    detail["_db_bucket"] = pd.cut(detail["D_minus_B_atr"], bins=bins, labels=labels)
    b_headers = ["D−B (atr)", "n"] + [f"D_touch_{h}d" for h in HORIZONS_DAYS]
    b_rows = []
    for lab in labels:
        sub = detail[detail["_db_bucket"] == lab]
        if len(sub) == 0:
            continue
        row = [lab, f"{len(sub):,}"]
        for h in HORIZONS_DAYS:
            t = sub[f"D_touched_{h}d"]
            mask = t.notna()
            if mask.sum() == 0:
                row.append("n/a")
            else:
                row.append(fmt_pct(float(t[mask].mean())))
        b_rows.append(row)
    lines.extend(md_table(b_headers, b_rows))
    lines.append("")

    lines.append("## D-touch rate by D-distance bucket (D/B-1 in percent, all families)")
    lines.append("")
    pct_bins = [-np.inf, 0.02, 0.05, 0.10, 0.20, 0.50, np.inf]
    pct_labels = ["<2%", "2–5%", "5–10%", "10–20%", "20–50%", ">50%"]
    detail["_db_pct_bucket"] = pd.cut(detail["B_to_D_pct"], bins=pct_bins, labels=pct_labels)
    p_headers = ["D/B−1", "n"] + [f"D_touch_{h}d" for h in HORIZONS_DAYS]
    p_rows = []
    for lab in pct_labels:
        sub = detail[detail["_db_pct_bucket"] == lab]
        if len(sub) == 0:
            continue
        row = [lab, f"{len(sub):,}"]
        for h in HORIZONS_DAYS:
            t = sub[f"D_touched_{h}d"]
            mask = t.notna()
            if mask.sum() == 0:
                row.append("n/a")
            else:
                row.append(fmt_pct(float(t[mask].mean())))
        p_rows.append(row)
    lines.extend(md_table(p_headers, p_rows))
    lines.append("")

    lines.append("### % bucket per family (5d / 10d / 20d / 30d touch rate)")
    lines.append("")
    for fam in FAMILIES:
        fam_sub = detail[detail["family"] == fam]
        if len(fam_sub) == 0:
            continue
        lines.append(f"**{fam}** (n={len(fam_sub):,})")
        lines.append("")
        f_headers = ["D/B−1", "n"] + [f"D_touch_{h}d" for h in HORIZONS_DAYS]
        f_rows = []
        for lab in pct_labels:
            sub = fam_sub[fam_sub["_db_pct_bucket"] == lab]
            if len(sub) == 0:
                continue
            row = [lab, f"{len(sub):,}"]
            for h in HORIZONS_DAYS:
                t = sub[f"D_touched_{h}d"]
                mask = t.notna()
                if mask.sum() == 0:
                    row.append("n/a")
                else:
                    row.append(fmt_pct(float(t[mask].mean())))
            f_rows.append(row)
        lines.extend(md_table(f_headers, f_rows))
        lines.append("")

    lines.append("## Forward MFE (median %) by family × horizon — for context")
    lines.append("")
    m_headers = ["family", "n"] + [f"med_mfe_{h}d" for h in HORIZONS_DAYS]
    m_rows = []
    for fam in FAMILIES:
        sub = detail[detail["family"] == fam]
        if len(sub) == 0:
            continue
        row = [fam, f"{len(sub):,}"]
        for h in HORIZONS_DAYS:
            v = sub[f"forward_mfe_pct_{h}d"].median()
            row.append("n/a" if not np.isfinite(v) else f"{v*100:+.2f}%")
        m_rows.append(row)
    lines.extend(md_table(m_headers, m_rows))
    lines.append("")

    lines.append("## D/B−1 percent distribution (percentiles)")
    lines.append("")
    pct_pcts = [10, 25, 50, 75, 90, 95]
    pd_headers = ["family", "n"] + [f"p{p}" for p in pct_pcts]
    pd_rows = []
    for fam in ["__ALL__"] + FAMILIES:
        sub = detail if fam == "__ALL__" else detail[detail["family"] == fam]
        if len(sub) == 0:
            continue
        row = ["all" if fam == "__ALL__" else fam, f"{len(sub):,}"]
        vals = sub["D_pct"].dropna()
        for p in pct_pcts:
            row.append(f"{np.percentile(vals, p):+.2f}%")
        pd_rows.append(row)
    lines.extend(md_table(pd_headers, pd_rows))
    lines.append("")

    lines.append("## structural_invalidation_low — typical distance below B (percentiles)")
    lines.append("")
    si_headers = ["family", "n"] + [f"p{p}" for p in pct_pcts]
    si_rows = []
    for fam in ["__ALL__"] + FAMILIES:
        sub = detail if fam == "__ALL__" else detail[detail["family"] == fam]
        sub = sub[sub["structural_invalidation_pct_below_B"].notna()]
        if len(sub) == 0:
            continue
        row = ["all" if fam == "__ALL__" else fam, f"{len(sub):,}"]
        vals = sub["structural_invalidation_pct_below_B"].dropna()
        for p in pct_pcts:
            row.append(f"{np.percentile(vals, p):+.2f}%")
        si_rows.append(row)
    lines.extend(md_table(si_headers, si_rows))
    lines.append("")
    lines.append("_Negative ⇒ invalidation level sits below B (the closer to 0, the tighter the structural stop)._")
    lines.append("")

    lines.append("## Outcome breakdown by horizon (strict — break invalidates hit)")
    lines.append("")
    lines.append("`hit_strict`: forward_high reached D **before** any bar with low < structural_invalidation_low.  ")
    lines.append("`broken`: SIL violated before D was hit (or D never hit; same-day tie ⇒ break wins).  ")
    lines.append("`neither`: window elapsed with no hit and no break.  ")
    lines.append("`edge`: not enough forward bars in extfeed to fill the window (event too recent).")
    lines.append("")

    bars_arr = detail["forward_bars_available"].to_numpy()
    sd_arr = detail["days_to_D_strict"].to_numpy(dtype=float)
    bd_arr = detail["structural_break_day"].to_numpy(dtype=float)
    fam_arr = detail["family"].to_numpy()

    for h in [3, 5, 10, 20, 30]:
        lines.append(f"### Horizon = {h} day(s)")
        lines.append("")
        lines.append("| family | n | hit_strict | broken | neither | edge |")
        lines.append("|---|---|---|---|---|---|")
        edge_h = bars_arr < h
        hit_h = (~edge_h) & np.isfinite(sd_arr) & (sd_arr <= h)
        brk_h = (~edge_h) & ~hit_h & np.isfinite(bd_arr) & (bd_arr <= h)
        nei_h = (~edge_h) & ~hit_h & ~brk_h
        for fam in ["__ALL__"] + FAMILIES:
            mask = np.ones(len(detail), dtype=bool) if fam == "__ALL__" else (fam_arr == fam)
            n = int(mask.sum())
            if n == 0:
                continue
            n_hit = int((mask & hit_h).sum())
            n_brk = int((mask & brk_h).sum())
            n_nei = int((mask & nei_h).sum())
            n_edg = int((mask & edge_h).sum())
            label = "all" if fam == "__ALL__" else fam
            lines.append(
                f"| {label} | {n:,} | {n_hit/n:.1%} | {n_brk/n:.1%} | "
                f"{n_nei/n:.1%} | {n_edg/n:.1%} |"
            )
        lines.append("")

    lines.append("## Outcome by horizon — FILTERED to D/B−1 ≥ 20% (large-projection slice)")
    lines.append("")
    lines.append("Below excludes events where D is within 20% of B (those are easy hits and dilute the picture).")
    lines.append("")

    big_mask = detail["D_pct"].to_numpy() >= 20.0
    big = detail[big_mask].copy()
    bars_big = big["forward_bars_available"].to_numpy()
    sd_big = big["days_to_D_strict"].to_numpy(dtype=float)
    bd_big = big["structural_break_day"].to_numpy(dtype=float)
    fam_big = big["family"].to_numpy()
    pct_big = big["D_pct"].to_numpy()

    for h in [3, 5, 10, 20, 30]:
        lines.append(f"### Horizon = {h} day(s) — D/B−1 ≥ 20%")
        lines.append("")
        lines.append("| family | n | hit_strict | broken | neither | edge |")
        lines.append("|---|---|---|---|---|---|")
        edge_h = bars_big < h
        hit_h = (~edge_h) & np.isfinite(sd_big) & (sd_big <= h)
        brk_h = (~edge_h) & ~hit_h & np.isfinite(bd_big) & (bd_big <= h)
        nei_h = (~edge_h) & ~hit_h & ~brk_h
        for fam in ["__ALL__"] + FAMILIES:
            mask = np.ones(len(big), dtype=bool) if fam == "__ALL__" else (fam_big == fam)
            n = int(mask.sum())
            if n == 0:
                continue
            n_hit = int((mask & hit_h).sum())
            n_brk = int((mask & brk_h).sum())
            n_nei = int((mask & nei_h).sum())
            n_edg = int((mask & edge_h).sum())
            label = "all" if fam == "__ALL__" else fam
            lines.append(
                f"| {label} | {n:,} | {n_hit/n:.1%} | {n_brk/n:.1%} | "
                f"{n_nei/n:.1%} | {n_edg/n:.1%} |"
            )
        lines.append("")

    lines.append("### Sub-bucket within D/B−1 ≥ 20% (all families combined)")
    lines.append("")
    sub_bins = [20, 30, 50, 100, np.inf]
    sub_labels = ["20–30%", "30–50%", "50–100%", ">100%"]
    sub_idx = np.digitize(pct_big, sub_bins) - 1
    for h in [3, 5, 10, 20, 30]:
        lines.append(f"#### Horizon = {h} day(s)")
        lines.append("")
        lines.append("| D/B−1 | n | hit_strict | broken | neither | edge |")
        lines.append("|---|---|---|---|---|---|")
        edge_h = bars_big < h
        hit_h = (~edge_h) & np.isfinite(sd_big) & (sd_big <= h)
        brk_h = (~edge_h) & ~hit_h & np.isfinite(bd_big) & (bd_big <= h)
        nei_h = (~edge_h) & ~hit_h & ~brk_h
        for i, lab in enumerate(sub_labels):
            mask = sub_idx == i
            n = int(mask.sum())
            if n == 0:
                continue
            n_hit = int((mask & hit_h).sum())
            n_brk = int((mask & brk_h).sum())
            n_nei = int((mask & nei_h).sum())
            n_edg = int((mask & edge_h).sum())
            lines.append(
                f"| {lab} | {n:,} | {n_hit/n:.1%} | {n_brk/n:.1%} | "
                f"{n_nei/n:.1%} | {n_edg/n:.1%} |"
            )
        lines.append("")

    lines.append("## days_to_D_strict distribution (CONDITIONAL on hit_strict; full-window events only)")
    lines.append("")
    sub_full = detail[(detail["forward_bars_available"] >= max_h)]
    dtd_headers = ["family", "n_hits", "hit_rate", "p25_days", "p50_days", "p75_days", "p90_days", "max_days"]
    dtd_rows = []
    for fam in ["__ALL__"] + FAMILIES:
        ss = sub_full if fam == "__ALL__" else sub_full[sub_full["family"] == fam]
        if len(ss) == 0:
            continue
        n_total = len(ss)
        hit_mask = ss["days_to_D_strict"].notna()
        n_hits = int(hit_mask.sum())
        if n_hits == 0:
            dtd_rows.append([("all" if fam == "__ALL__" else fam),
                             "0", "0.0%", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        days = ss.loc[hit_mask, "days_to_D_strict"].to_numpy()
        dtd_rows.append([
            "all" if fam == "__ALL__" else fam,
            f"{n_hits:,}",
            f"{n_hits / n_total:.1%}",
            f"{np.percentile(days, 25):.0f}",
            f"{np.percentile(days, 50):.0f}",
            f"{np.percentile(days, 75):.0f}",
            f"{np.percentile(days, 90):.0f}",
            f"{int(days.max())}",
        ])
    lines.extend(md_table(dtd_headers, dtd_rows))
    lines.append("")

    lines.append("## Cumulative hit_strict % by day (full-window events, all families)")
    lines.append("")
    n_full = len(sub_full)
    cum_headers = ["day"] + [f"≤{d}d" for d in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]]
    cum_row = ["all"] + []
    for d in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        within = (sub_full["days_to_D_strict"] <= d).sum()
        cum_row.append(f"{within / n_full:.1%}" if n_full > 0 else "n/a")
    lines.append("| " + " | ".join(cum_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(cum_headers)) + "|")
    lines.append("| " + " | ".join(cum_row) + " |")
    lines.append("")

    lines.append("## days_to_D_strict by D/B-1 bucket (CONDITIONAL on hit_strict; full-window events)")
    lines.append("")
    pct_bins2 = [-np.inf, 2, 5, 10, 20, 50, np.inf]
    pct_labels2 = ["<2%", "2–5%", "5–10%", "10–20%", "20–50%", ">50%"]
    sub_full = sub_full.copy()
    sub_full["_pct_bucket"] = pd.cut(sub_full["D_pct"], bins=pct_bins2, labels=pct_labels2)
    db_headers = ["D/B−1", "n_total", "hit_strict %", "broken %", "p25_days", "p50_days", "p75_days"]
    db_rows = []
    for lab in pct_labels2:
        ss = sub_full[sub_full["_pct_bucket"] == lab]
        if len(ss) == 0:
            continue
        n_total = len(ss)
        n_hit = int(ss["days_to_D_strict"].notna().sum())
        n_brk = int((ss["outcome_strict"] == "broken_before_hit").sum())
        days = ss["days_to_D_strict"].dropna().to_numpy()
        if len(days) == 0:
            p25 = p50 = p75 = "n/a"
        else:
            p25 = f"{np.percentile(days, 25):.0f}"
            p50 = f"{np.percentile(days, 50):.0f}"
            p75 = f"{np.percentile(days, 75):.0f}"
        db_rows.append([lab, f"{n_total:,}",
                        f"{n_hit/n_total:.1%}", f"{n_brk/n_total:.1%}",
                        p25, p50, p75])
    lines.extend(md_table(db_headers, db_rows))
    lines.append("")

    lines.append("## Strict vs loose hit-rate gap (how much does the structural-break filter cost?)")
    lines.append("")
    sl_headers = ["family", "n_full", "loose_hit_30d", "strict_hit_30d", "Δ (loose−strict)"]
    sl_rows = []
    for fam in ["__ALL__"] + FAMILIES:
        ss = sub_full if fam == "__ALL__" else sub_full[sub_full["family"] == fam]
        if len(ss) == 0:
            continue
        loose = ss["days_to_D_loose"].notna().mean()
        strict = ss["days_to_D_strict"].notna().mean()
        sl_rows.append([
            "all" if fam == "__ALL__" else fam,
            f"{len(ss):,}",
            fmt_pct(loose), fmt_pct(strict),
            f"{(loose - strict)*100:+.1f}pp",
        ])
    lines.extend(md_table(sl_headers, sl_rows))
    lines.append("")

    lines.append("## Notes")
    lines.append("- `D_touched_Nd = forward_high_Nd ≥ D`. NaN when ticker missing from extfeed or no forward bars within window (e.g. last events near data edge).")
    lines.append("- `gap_to_D_atr_Nd = (D − forward_high_Nd) / atr_at_event`. Negative ⇒ D was exceeded.")
    lines.append("- Probe is independent: no edits to mb_scanner / decision_engine / decision_engine_v1 / decision_engine_lab / nyxexpansion.")
    lines.append(f"- Detail parquet: `{DETAIL_PATH.name}` ({len(detail):,} rows).")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"[trident-probe] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
