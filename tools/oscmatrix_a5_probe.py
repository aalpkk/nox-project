"""Layer A.5 probe — does divergence/overflow context explain Reversal arrows?

Budget: ≤1h. Goal: NOT bit-exact. Only "do these features lift the rate at
which Reversal Up/Down (+/-) fires occur?"

Features:
  TV ground truth (from CSV):
    bull_overflow_active     : TV Bullish Overflow != 50
    bear_overflow_active     : TV Bearish Overflow != 50
    bull_overflow_recent_K   : any active in last K bars
    bear_overflow_recent_K   : any active in last K bars
    upper_zone (TV)          : Upper Confluence Value (0/1/2)
    lower_zone (TV)          : Lower Confluence Value (0/1/2)
  Ours (computed):
    hwo_up_recent_K          : any HWO Up cross in last K bars
    hwo_down_recent_K        : any HWO Down cross in last K bars
    hw_lt_50 / hw_gt_50      : HW state
    bull_div_recent_K        : pivot-based proxy (price LL + HW HL within K bars)
    bear_div_recent_K        : symmetric

Lift = P(feature | TV fire) / P(feature | random bar). Lift > 1.5 with N≥10 fires
across 4 tickers is a meaningful signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_PATHS = {
    "THYAO": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/BIST_THYAO, 1D.csv",
    "GARAN": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

PIVOT_LB = 3  # bars on each side for pivot detection (small for short-term divergences)


def divergence_proxy(close: pd.Series, hw: pd.Series, lb: int = PIVOT_LB) -> pd.DataFrame:
    """Return bull_div / bear_div boolean series.

    Bull div (regular): price makes a lower pivot low while HW makes a higher pivot low.
    Bear div (regular): price makes a higher pivot high while HW makes a lower pivot high.
    Pivot = strict local extremum over ±lb window.
    Confirmed at the pivot bar itself (lookahead-free needs +lb shift; we shift below).
    """
    n = len(close)
    is_pivlow = pd.Series(False, index=close.index)
    is_pivhigh = pd.Series(False, index=close.index)
    for i in range(lb, n - lb):
        win = slice(i - lb, i + lb + 1)
        if close.iloc[i] == close.iloc[win].min() and (close.iloc[win] == close.iloc[i]).sum() == 1:
            is_pivlow.iloc[i] = True
        if close.iloc[i] == close.iloc[win].max() and (close.iloc[win] == close.iloc[i]).sum() == 1:
            is_pivhigh.iloc[i] = True

    # Compare consecutive pivots
    bull_div = pd.Series(False, index=close.index)
    bear_div = pd.Series(False, index=close.index)
    pl_idx = list(np.where(is_pivlow.values)[0])
    ph_idx = list(np.where(is_pivhigh.values)[0])
    for a, b in zip(pl_idx[:-1], pl_idx[1:]):
        if close.iloc[b] < close.iloc[a] and hw.iloc[b] > hw.iloc[a]:
            bull_div.iloc[b + lb] = True if b + lb < n else False  # confirm lb bars later
    for a, b in zip(ph_idx[:-1], ph_idx[1:]):
        if close.iloc[b] > close.iloc[a] and hw.iloc[b] < hw.iloc[a]:
            bear_div.iloc[b + lb] = True if b + lb < n else False
    return pd.DataFrame({"bull_div": bull_div, "bear_div": bear_div}, index=close.index)


def recent_any(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).max().astype(bool)


def build_features(tv: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=tv.index)
    bull_ovf = tv["Bullish Overflow"].fillna(50) != 50
    bear_ovf = tv["Bearish Overflow"].fillna(50) != 50
    f["bull_ovf_active"] = bull_ovf
    f["bear_ovf_active"] = bear_ovf
    for k in (3, 5, 10):
        f[f"bull_ovf_recent_{k}"] = recent_any(bull_ovf, k)
        f[f"bear_ovf_recent_{k}"] = recent_any(bear_ovf, k)

    f["upper_zone_tv"] = tv["Upper Confluence Value"].astype("Int64")
    f["lower_zone_tv"] = tv["Lower Confluence Value"].astype("Int64")
    f["upper_zone_ge1"] = (f["upper_zone_tv"] >= 1).astype(bool)
    f["upper_zone_eq2"] = (f["upper_zone_tv"] == 2).astype(bool)
    f["lower_zone_ge1"] = (f["lower_zone_tv"] >= 1).astype(bool)
    f["lower_zone_eq2"] = (f["lower_zone_tv"] == 2).astype(bool)

    hw = ours["hyperwave"]
    f["hw_lt_50"] = hw < 50
    f["hw_gt_50"] = hw > 50
    f["hw_lt_30"] = hw < 30
    f["hw_gt_70"] = hw > 70

    hwo_up = ours["hwo_up"].fillna(False).astype(bool)
    hwo_dn = ours["hwo_down"].fillna(False).astype(bool)
    for k in (3, 5):
        f[f"hwo_up_recent_{k}"] = recent_any(hwo_up, k)
        f[f"hwo_down_recent_{k}"] = recent_any(hwo_dn, k)

    div = divergence_proxy(tv["close"], hw)
    for k in (3, 5, 10):
        f[f"bull_div_recent_{k}"] = recent_any(div["bull_div"], k)
        f[f"bear_div_recent_{k}"] = recent_any(div["bear_div"], k)

    return f


def lift_table(features: pd.DataFrame, fire_idx: pd.Index) -> pd.DataFrame:
    rows = []
    n_total = len(features)
    n_fire = len(fire_idx)
    for col in features.columns:
        if features[col].dtype.name in ("Int64", "object"):
            continue
        s = features[col].astype(bool)
        base = s.mean()
        if n_fire == 0 or base == 0:
            rows.append({"feature": col, "base": base, "fire_rate": np.nan, "lift": np.nan, "n_fire_active": 0})
            continue
        fire_rate = s.loc[fire_idx].mean()
        n_active = int(s.loc[fire_idx].sum())
        rows.append({
            "feature": col,
            "base": float(base),
            "fire_rate": float(fire_rate),
            "lift": float(fire_rate / base) if base else np.nan,
            "n_fire_active": n_active,
        })
    return pd.DataFrame(rows)


def aggregate_lift(per_ticker: list[pd.DataFrame], n_fires_total: int) -> pd.DataFrame:
    if not per_ticker:
        return pd.DataFrame()
    merged = pd.concat(per_ticker, ignore_index=False)
    agg = merged.groupby("feature", sort=False).agg(
        base_mean=("base", "mean"),
        fire_rate_mean=("fire_rate", "mean"),
        lift_mean=("lift", "mean"),
        n_fire_active_total=("n_fire_active", "sum"),
    )
    agg["n_fires_total"] = n_fires_total
    agg["coverage"] = agg["n_fire_active_total"] / max(n_fires_total, 1)
    return agg.sort_values("lift_mean", ascending=False)


def main() -> None:
    fire_kinds = ["Reversal Up +", "Reversal Up -", "Reversal Down +", "Reversal Down -"]
    per_kind: dict[str, list[pd.DataFrame]] = {k: [] for k in fire_kinds}
    fires_count: dict[str, int] = {k: 0 for k in fire_kinds}

    for ticker, path in CSV_PATHS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
        features = build_features(tv, ours)
        for kind in fire_kinds:
            if kind not in tv.columns:
                continue
            fire_idx = tv.index[tv[kind].notna()]
            fires_count[kind] += len(fire_idx)
            tab = lift_table(features, fire_idx)
            tab["ticker"] = ticker
            per_kind[kind].append(tab)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print("=== Fire counts (4-ticker total) ===")
    for k, v in fires_count.items():
        print(f"  {k:<20} {v}")
    print()

    for kind in fire_kinds:
        agg = aggregate_lift(per_kind[kind], fires_count[kind])
        if agg.empty:
            continue
        print(f"\n=== Lift table — {kind}  (n={fires_count[kind]} across 4 tickers) ===")
        print(agg.head(15).to_string())


if __name__ == "__main__":
    main()
