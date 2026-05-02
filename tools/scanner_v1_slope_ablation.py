"""V1.3.1 slope ablation — body-ref × slope-dim × cap matrix.

Two body references:
  A0_ref     : body floor = 0.65 (V1.2.9 semantics — slope only)
  B0_current : body floor = 0.35 (V1.3.1 production)

Three slope dimensions (the OTHER slope held at strict 0.15%/d when applicable):
  base       : |base_slope| cap varies, |res_slope| ≤ 0.15%/d
  resistance : |res_slope|  cap varies, |base_slope| ≤ 0.15%/d
  both       : |base_slope| AND |res_slope| ≤ cap (joint)

Five caps (per-day, fraction of price):
  0.0015 (0.15%) baseline strict
  0.0020 (0.20%)
  0.0030 (0.30%)
  0.0050 (0.50%)
  inf    (no_cap, diagnostic)

Method: scan once per body_ref with slope filter DISABLED. Each cycle's
(|base_slope|, |res_slope|) is captured on its trigger event. The 15 (dim×cap)
slices are then post-filters over the same superset — no rescans, no
selection-bias reuse.

Per (body_ref, dim, cap) the report covers:
  total cohort metrics — N, R_mean, MFE_R, MAE_R, win%, PF, EF_5d%, FB_5d%
  mild-slope incremental cohort — events admitted ONLY by the relaxed cap
                                   (|relevant slope| > 0.0015 strict)
  year × cohort PF (5d) for the mild incremental
  body_class distribution within the mild incremental
  acceptance verdict against user-stated criteria

Trigger semantics: per-asof first-break == asof_idx (one row per cycle, the
T1 bar). Matches body-ablation A0/A1 numerator. retest_bounce/extended are
out of scope for this study — slope is a base-geometry filter; cohort focus
on T1 keeps the comparison clean.
"""
from __future__ import annotations

import sys
import time
from math import inf
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import intraday_1h
from scanner.triggers.horizontal_base import (
    BB_LENGTH,
    BODY_LARGE_THRESH,
    BODY_STRICT_THRESH,
    BREAKOUT_BUFFER,
    MAX_SQUEEZE_AGE_BARS,
    TRIG_RANGE_POS_MIN,
    TRIG_VOL_RATIO_MIN,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)
from scanner.schema import FEATURE_VERSION, SCANNER_VERSION

ATR_SL_MULT = 0.30                  # match _build_row R denominator
STRICT_CAP = 0.0015
CAPS = [0.0015, 0.0020, 0.0030, 0.0050, inf]
CAP_LABELS = {0.0015: "0.15%", 0.0020: "0.20%", 0.0030: "0.30%",
              0.0050: "0.50%", inf: "no_cap"}
HORIZONS = [5, 20]
DIMS = ("base", "resistance", "both")

BODY_REFS = [
    ("A0_ref",     0.65),
    ("B0_current", 0.35),
]


# ------------------------------------------------------------------ scan
def _is_breakout(df, idx, robust, hard, body_floor):
    h = float(df["high"].iat[idx])
    l = float(df["low"].iat[idx])
    c = float(df["close"].iat[idx])
    o = float(df["open"].iat[idx])
    atr = float(df["atr_sq"].iat[idx]) if pd.notna(df["atr_sq"].iat[idx]) else 0.0
    vol = float(df["volume"].iat[idx])
    vol_sma = float(df["vol_sma"].iat[idx]) if pd.notna(df["vol_sma"].iat[idx]) else 0.0
    sma20 = float(df["sma20"].iat[idx]) if pd.notna(df["sma20"].iat[idx]) else 0.0
    vwap20 = float(df["vwap20"].iat[idx]) if pd.notna(df["vwap20"].iat[idx]) else 0.0
    vwap60 = float(df["vwap60"].iat[idx]) if pd.notna(df["vwap60"].iat[idx]) else 0.0
    if atr <= 0 or vol_sma <= 0 or sma20 <= 0:
        return False
    body = abs(c - o)
    rng = h - l
    range_pos = (c - l) / rng if rng > 0 else 0.5
    vwap_ok = (vwap20 > 0 and c > vwap20) or (vwap60 > 0 and c > vwap60)
    return (
        c > robust * (1.0 + BREAKOUT_BUFFER)
        and h >= hard
        and c > o
        and range_pos >= TRIG_RANGE_POS_MIN
        and body > atr * body_floor
        and vol > vol_sma * TRIG_VOL_RATIO_MIN
        and c > sma20
        and vwap_ok
    )


def _classify_body(body_atr: float) -> str:
    if not (body_atr == body_atr):
        return ""
    if body_atr >= BODY_LARGE_THRESH:
        return "large_body"
    if body_atr >= BODY_STRICT_THRESH:
        return "strict_body"
    return "mid_body"


def scan_ticker(daily_df: pd.DataFrame, ticker: str, body_floor: float) -> list[dict]:
    """Per-asof first-trigger scan with slope filter DISABLED.

    Returns one event per cycle that produces a T1 (one row per ticker/cycle).
    Each event carries its cycle's |base_slope| / |res_slope| for post-filter.
    """
    if len(daily_df) < BB_LENGTH * 3:
        return []
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return []
    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])
    n = len(df)
    cycle_struct: dict = {}
    seen_cycles: set[int] = set()
    out: list[dict] = []
    n_max = n - 1

    for asof_idx in range(BB_LENGTH * 2, n):
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue
        if pos in seen_cycles:
            continue
        if pos not in cycle_struct:
            base = df.iloc[sq_s: sq_e + 1]
            b_slope, _ = _line_fit(base["close"].values)
            r_slope, _ = _line_fit(base["high"].values)
            box_top, hard, box_bot = _box_from_squeeze(df, sq_s, sq_e)
            if not (box_top > box_bot > 0):
                cycle_struct[pos] = None
            else:
                cycle_struct[pos] = (box_top, hard, box_bot,
                                     abs(float(b_slope)), abs(float(r_slope)))
        cs = cycle_struct[pos]
        if cs is None:
            continue
        box_top, hard, box_bot, abs_b, abs_r = cs

        asof_close = float(df["close"].iat[asof_idx])
        if asof_close <= 0:
            continue
        if (box_top - box_bot) / asof_close > WIDTH_PCT_REJECT:
            continue

        if not _is_breakout(df, asof_idx, box_top, hard, body_floor):
            continue
        # T1 confirmed for this cycle
        seen_cycles.add(pos)

        atr_sq = float(df["atr_sq"].iat[asof_idx]) if pd.notna(df["atr_sq"].iat[asof_idx]) else 0.0
        if atr_sq <= 0:
            continue
        entry = float(df["close"].iat[asof_idx])
        invalidation = box_bot - ATR_SL_MULT * atr_sq
        R = entry - invalidation
        if R <= 0:
            continue
        body = abs(entry - float(df["open"].iat[asof_idx]))
        body_atr = body / atr_sq
        rec: dict = {
            "ticker": ticker,
            "date": df.index[asof_idx],
            "year": int(pd.Timestamp(df.index[asof_idx]).year),
            "abs_base_slope": abs_b,
            "abs_res_slope": abs_r,
            "body_atr": float(body_atr),
            "body_class": _classify_body(body_atr),
            "entry": entry,
            "R": R,
            "box_top": box_top,
            "atr_sq": atr_sq,
        }
        for h in HORIZONS:
            end_idx = min(asof_idx + h, n_max)
            if end_idx <= asof_idx:
                rec[f"R_{h}"] = np.nan
                rec[f"MFE_R_{h}"] = np.nan
                rec[f"MAE_R_{h}"] = np.nan
                if h == 5:
                    rec["EF_5d"] = False
                    rec["FB_5d"] = False
                continue
            window = df.iloc[asof_idx + 1: end_idx + 1]
            close_h = float(df["close"].iat[end_idx])
            rec[f"R_{h}"] = (close_h - entry) / R
            rec[f"MFE_R_{h}"] = (float(window["high"].max()) - entry) / R
            rec[f"MAE_R_{h}"] = (float(window["low"].min()) - entry) / R
            if h == 5:
                rec["EF_5d"] = bool(rec["MAE_R_5"] <= -1.0)
                rec["FB_5d"] = bool((window["close"] < (box_top - 0.20 * atr_sq)).any())
        out.append(rec)
    return out


# ------------------------------------------------------------------ slicing
def _filter_total(df: pd.DataFrame, dim: str, cap: float) -> pd.DataFrame:
    if dim == "base":
        return df[(df["abs_base_slope"] <= cap) & (df["abs_res_slope"] <= STRICT_CAP)]
    if dim == "resistance":
        return df[(df["abs_res_slope"] <= cap) & (df["abs_base_slope"] <= STRICT_CAP)]
    if dim == "both":
        return df[(df["abs_base_slope"] <= cap) & (df["abs_res_slope"] <= cap)]
    raise ValueError(dim)


def _filter_mild(df: pd.DataFrame, dim: str, cap: float) -> pd.DataFrame:
    """Events admitted by relaxed cap but NOT by strict 0.15% on the relaxed dim."""
    rel = _filter_total(df, dim, cap)
    if cap <= STRICT_CAP:
        return rel.iloc[0:0]
    if dim == "base":
        return rel[rel["abs_base_slope"] > STRICT_CAP]
    if dim == "resistance":
        return rel[rel["abs_res_slope"] > STRICT_CAP]
    if dim == "both":
        return rel[(rel["abs_base_slope"] > STRICT_CAP) | (rel["abs_res_slope"] > STRICT_CAP)]
    raise ValueError(dim)


def _metrics(df: pd.DataFrame) -> dict:
    out: dict = {"N": int(len(df))}
    if df.empty:
        for h in HORIZONS:
            out[f"R_mean_{h}"] = np.nan
            out[f"MFE_{h}"]    = np.nan
            out[f"MAE_{h}"]    = np.nan
            out[f"win_{h}"]    = np.nan
            out[f"PF_{h}"]     = np.nan
        out["EF_5d"] = np.nan
        out["FB_5d"] = np.nan
        return out
    for h in HORIZONS:
        s = df[f"R_{h}"].dropna()
        if s.empty:
            out[f"R_mean_{h}"] = np.nan
            out[f"win_{h}"]    = np.nan
            out[f"PF_{h}"]     = np.nan
        else:
            out[f"R_mean_{h}"] = float(s.mean())
            wins = float(s[s > 0].sum())
            losses = float(-s[s < 0].sum())
            out[f"PF_{h}"] = (wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
            out[f"win_{h}"] = float((s > 0).mean() * 100.0)
        out[f"MFE_{h}"] = float(df[f"MFE_R_{h}"].dropna().mean())
        out[f"MAE_{h}"] = float(df[f"MAE_R_{h}"].dropna().mean())
    out["EF_5d"] = float(df["EF_5d"].mean() * 100.0)
    out["FB_5d"] = float(df["FB_5d"].mean() * 100.0)
    return out


def _year_pf(df: pd.DataFrame, year: int, h: int = 5) -> tuple[int, float, float]:
    sub = df[df["year"] == year]
    n = int(len(sub))
    if n == 0:
        return 0, np.nan, np.nan
    s = sub[f"R_{h}"].dropna()
    rmean = float(s.mean()) if not s.empty else np.nan
    wins = float(s[s > 0].sum()); losses = float(-s[s < 0].sum())
    pf = (wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
    return n, rmean, pf


def _body_dist(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"mid": 0, "strict": 0, "large": 0}
    vc = df["body_class"].value_counts()
    return {"mid":    int(vc.get("mid_body", 0)),
            "strict": int(vc.get("strict_body", 0)),
            "large":  int(vc.get("large_body", 0))}


def _verdict(strict: dict, mild: dict, mild_2026_pf: float, mild_body: dict) -> str:
    """Apply user acceptance criteria; return short tag."""
    if mild["N"] < 30:
        return "REJECT (N too small)"
    fail: list[str] = []
    if not (mild["PF_5"] >= strict["PF_5"]):
        fail.append(f"PF5 {mild['PF_5']:.2f}<{strict['PF_5']:.2f}")
    if not (mild["PF_20"] >= strict["PF_20"]):
        fail.append(f"PF20 {mild['PF_20']:.2f}<{strict['PF_20']:.2f}")
    if mild["EF_5d"] - strict["EF_5d"] > 1.5:
        fail.append(f"EF +{mild['EF_5d']-strict['EF_5d']:.1f}pp")
    if mild["FB_5d"] - strict["FB_5d"] > 5.0:
        fail.append(f"FB +{mild['FB_5d']-strict['FB_5d']:.1f}pp")
    if not (np.isfinite(mild_2026_pf) and mild_2026_pf > 1.0):
        fail.append(f"2026 PF {mild_2026_pf:.2f}")
    total = sum(mild_body.values())
    if total > 0:
        share = max(mild_body.values()) / total
        if share > 0.85:
            fail.append(f"body skew {share*100:.0f}%")
    if fail:
        return "REJECT (" + "; ".join(fail) + ")"
    return "ACCEPT"


# ------------------------------------------------------------------ render
def _fmt(x, fmt=".2f", nan="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return nan
    return format(x, fmt)


def report_body_ref(label: str, body_floor: float, events: pd.DataFrame, lines: list[str]) -> None:
    H = "=" * 96
    lines.append(""); lines.append(H)
    lines.append(f"[{label}]   body_floor = {body_floor}   N_no_cap = {len(events):,}")
    lines.append(H)

    for dim in DIMS:
        strict_df = _filter_total(events, dim, STRICT_CAP)
        strict = _metrics(strict_df)
        # also year cell + body dist for strict (context)
        s2026_n, s2026_rm, s2026_pf = _year_pf(strict_df, 2026, 5)

        lines.append("")
        lines.append(f"  --- dim = {dim} ---")
        lines.append(f"     STRICT 0.15%: N={strict['N']:>4d}  "
                     f"PF_5d={_fmt(strict['PF_5'])}  R5={_fmt(strict['R_mean_5'],'.3f')}  win5={_fmt(strict['win_5'],'.1f')}  "
                     f"PF_20d={_fmt(strict['PF_20'])}  R20={_fmt(strict['R_mean_20'],'.3f')}  "
                     f"EF={_fmt(strict['EF_5d'],'.1f')}%  FB={_fmt(strict['FB_5d'],'.1f')}%  "
                     f"2026 N={s2026_n} PF={_fmt(s2026_pf)}")

        lines.append("     " + "-" * 90)
        lines.append("     cap     | total                                      | mild incremental                                                  | verdict")
        lines.append("     " + "-" * 90)

        for cap in CAPS:
            if cap <= STRICT_CAP:
                continue
            tot_df = _filter_total(events, dim, cap)
            tot = _metrics(tot_df)
            mild_df = _filter_mild(events, dim, cap)
            mild = _metrics(mild_df)
            mild_pack = {"N": mild["N"], "PF_5": mild["PF_5"], "PF_20": mild["PF_20"],
                         "EF_5d": mild["EF_5d"], "FB_5d": mild["FB_5d"]}
            strict_pack = {"PF_5": strict["PF_5"], "PF_20": strict["PF_20"],
                           "EF_5d": strict["EF_5d"], "FB_5d": strict["FB_5d"]}
            _, _, m2026 = _year_pf(mild_df, 2026, 5)
            verd = _verdict(strict_pack, mild_pack, m2026,
                            _body_dist(mild_df))

            lines.append(
                f"     {CAP_LABELS[cap]:<7s} | "
                f"N={tot['N']:>4d}  PF5={_fmt(tot['PF_5']):>5s}  PF20={_fmt(tot['PF_20']):>5s}  "
                f"EF={_fmt(tot['EF_5d'],'.1f'):>4s}  FB={_fmt(tot['FB_5d'],'.1f'):>5s}  | "
                f"N={mild['N']:>4d}  PF5={_fmt(mild['PF_5']):>5s}  PF20={_fmt(mild['PF_20']):>5s}  "
                f"R5={_fmt(mild['R_mean_5'],'.3f'):>6s}  R20={_fmt(mild['R_mean_20'],'.3f'):>6s}  "
                f"EF={_fmt(mild['EF_5d'],'.1f'):>4s}  FB={_fmt(mild['FB_5d'],'.1f'):>5s}  | "
                f"{verd}"
            )

        # year stability for the most-relaxed mild cohort (no_cap)
        mild_inf = _filter_mild(events, dim, inf)
        if not mild_inf.empty:
            lines.append("")
            lines.append(f"     mild@no_cap year × PF_5d:")
            for y in (2023, 2024, 2025, 2026):
                n, rm, pf = _year_pf(mild_inf, y, 5)
                lines.append(f"        {y}  N={n:>3d}  R5={_fmt(rm,'.3f'):>6s}  PF={_fmt(pf):>5s}")

        # body-class distribution for the no_cap mild cohort
        if not mild_inf.empty:
            bd = _body_dist(mild_inf)
            tot = sum(bd.values())
            if tot > 0:
                lines.append(f"     mild@no_cap body_class: "
                             f"mid={bd['mid']} ({100*bd['mid']/tot:.0f}%)  "
                             f"strict={bd['strict']} ({100*bd['strict']/tot:.0f}%)  "
                             f"large={bd['large']} ({100*bd['large']/tot:.0f}%)")


def main() -> int:
    t0 = time.time()
    print(f"V1.3.1 slope ablation (SCANNER={SCANNER_VERSION} FEATURE={FEATURE_VERSION})", flush=True)
    print(f"  caps: {[CAP_LABELS[c] for c in CAPS]}   dims: {DIMS}   horizons: {HORIZONS}", flush=True)

    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(min_coverage=0.0)
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in "
          f"{time.time()-t0:.1f}s", flush=True)

    print("[2/3] daily resample …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    by_ref: dict[str, pd.DataFrame] = {}
    for label, body_floor in BODY_REFS:
        print(f"[3/3] scan body_ref={label} body_floor={body_floor} …", flush=True)
        t2 = time.time()
        all_rows: list[dict] = []
        tickers = daily["ticker"].unique()
        for i, t in enumerate(tickers, 1):
            g = daily[daily["ticker"] == t].sort_values("date")
            idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
            sub_df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
            all_rows.extend(scan_ticker(sub_df, str(t), body_floor))
            if i % 100 == 0:
                print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s "
                      f"events_so_far={len(all_rows):,}", flush=True)
        print(f"  scan {label}: {time.time()-t2:.1f}s  events={len(all_rows):,}", flush=True)
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        by_ref[label] = df

    # write supersets (for re-use / re-analysis)
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    for label, df in by_ref.items():
        path = out_dir / f"scanner_v1_slope_ablation_{label}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {len(df):,} events → {path.name}", flush=True)

    # render report
    lines: list[str] = []
    lines.append("V1.3.1 SLOPE ABLATION REPORT")
    lines.append(f"SCANNER={SCANNER_VERSION}  FEATURE={FEATURE_VERSION}")
    lines.append(f"caps={[CAP_LABELS[c] for c in CAPS]}   dims={DIMS}")
    lines.append("strict_cap = 0.15%/d (baseline). mild = relaxed cap admits, strict 0.15% rejects.")
    lines.append("acceptance: N>=30, PF_5d>=strict, PF_20d>=strict, EF_5d<=strict+1.5pp, "
                 "FB_5d<=strict+5pp, 2026 PF_5d>1.0, body_class no single bucket >85%.")
    for label, body_floor in BODY_REFS:
        report_body_ref(label, body_floor, by_ref[label], lines)

    log_path = out_dir / "scanner_v1_slope_ablation.log"
    log_path.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nreport → {log_path}")
    print(f"total elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
