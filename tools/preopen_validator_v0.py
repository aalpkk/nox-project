"""Pre-open validator for US Trident/BOS T-close candidates.

Takes a T-close candidate file + a T+1 pre-market snapshot (~09:20 ET) and
labels each candidate as one of:

    PASS_PREOPEN     setup still valid, watch open
    WATCH_OPEN       pm mixed / weak data / needs open confirmation
    REJECT_PREOPEN   setup broken or chase-risk excessive

Does NOT generate new signals. Does NOT re-select candidates. Only annotates
whether yesterday's setup is still tradeable into today's open.

Usage:
    python tools/preopen_validator_v0.py \\
        --candidates path/to/candidates.{csv,parquet} \\
        --premarket  path/to/premarket_0920et.{csv,parquet} \\
        --output     path/to/output.{csv,parquet} \\
        [--config    path/to/config.yml] \\
        [--snapshot-time-check]

    python tools/preopen_validator_v0.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONFIG: dict[str, float] = {
    "meaningful_pm_volume_ratio": 0.02,
    "strong_pm_volume_ratio": 0.05,
    "very_low_pm_volume_ratio": 0.005,
    "reject_gap_atr": 2.0,
    "watch_gap_atr": 1.0,
    "reject_gap_pct": 0.20,
    "watch_gap_pct": 0.08,
    "reject_spread_pct": 0.02,
    "watch_spread_pct": 0.01,
}

REQUIRED_CANDIDATE_COLS = [
    "ticker", "signal_date", "close_t", "bos_level", "stop_ref",
    "atr_14", "volume_t", "avg_volume_20", "dollar_volume_t",
]

OPTIONAL_CANDIDATE_COLS = ["setup_score", "trident_score", "signal_type"]

REQUIRED_PREMARKET_COLS = [
    "ticker", "snapshot_datetime_et", "pm_last", "pm_bid", "pm_ask",
    "pm_volume",
]

OPTIONAL_PREMARKET_COLS = [
    "pm_vwap", "pm_high", "pm_low", "prev_close", "close_t",
    "halted", "news_flag",
]


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported extension: {p.suffix} (path={p})")


def _write_any(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"unsupported extension: {p.suffix} (path={p})")


def load_inputs(
    candidates_path: str | Path, premarket_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_any(candidates_path), _read_any(premarket_path)


def validate_schema(cand: pd.DataFrame, pm: pd.DataFrame) -> None:
    missing_c = [c for c in REQUIRED_CANDIDATE_COLS if c not in cand.columns]
    missing_p = [c for c in REQUIRED_PREMARKET_COLS if c not in pm.columns]
    errs: list[str] = []
    if missing_c:
        errs.append(f"candidates missing columns: {missing_c}")
    if missing_p:
        errs.append(f"premarket missing columns: {missing_p}")
    if errs:
        raise ValueError("schema validation failed — " + " | ".join(errs))


def _load_config(config_path: str | Path | None) -> dict[str, float]:
    cfg = dict(DEFAULT_CONFIG)
    if config_path is None:
        return cfg
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pyyaml required for --config; install via `pip install pyyaml`"
        ) from e
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}
    if "defaults" in user_cfg and isinstance(user_cfg["defaults"], dict):
        user_cfg = user_cfg["defaults"]
    unknown = [k for k in user_cfg if k not in cfg]
    if unknown:
        print(f"[warn] unknown config keys ignored: {unknown}", file=sys.stderr)
    cfg.update({k: float(v) for k, v in user_cfg.items() if k in cfg})
    return cfg


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return (pd.Timestamp(d).normalize() + bd).normalize()


def compute_preopen_features(
    cand: pd.DataFrame, pm: pd.DataFrame,
    *, snapshot_time_check: bool = False,
) -> pd.DataFrame:
    """Merge candidates × pre-market by ticker, compute derived metrics."""
    pm = pm.copy()
    for col in OPTIONAL_PREMARKET_COLS:
        if col not in pm.columns:
            pm[col] = np.nan
    if "halted" in pm.columns:
        pm["halted"] = pm["halted"].fillna(False).astype(bool)
    if "news_flag" in pm.columns:
        pm["news_flag"] = pm["news_flag"].fillna(False).astype(bool)

    pm_cols = ["ticker", "snapshot_datetime_et", "pm_last", "pm_bid", "pm_ask",
               "pm_volume", "pm_vwap", "pm_high", "pm_low", "halted", "news_flag"]
    df = cand.merge(pm[pm_cols], on="ticker", how="left")

    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    df["snapshot_datetime_et"] = pd.to_datetime(df["snapshot_datetime_et"], errors="coerce")

    if snapshot_time_check:
        expected = df["signal_date"].apply(_next_trading_day)
        snap_date = df["snapshot_datetime_et"].dt.normalize()
        mismatched = (snap_date != expected) & snap_date.notna()
        if mismatched.any():
            bad = df.loc[mismatched, ["ticker", "signal_date", "snapshot_datetime_et"]].head(5)
            raise ValueError(
                f"snapshot-time-check failed for {int(mismatched.sum())} rows; "
                f"first offenders:\n{bad.to_string(index=False)}"
            )

    df["gap_pct"] = df["pm_last"] / df["close_t"] - 1.0
    df["gap_atr"] = (df["pm_last"] - df["close_t"]) / df["atr_14"]
    mid = (df["pm_bid"] + df["pm_ask"]) / 2.0
    df["spread_pct"] = (df["pm_ask"] - df["pm_bid"]) / mid
    df["pm_volume_ratio"] = df["pm_volume"] / df["avg_volume_20"]
    df["pm_last_vs_bos_pct"] = df["pm_last"] / df["bos_level"] - 1.0
    df["pm_last_vs_stop_pct"] = df["pm_last"] / df["stop_ref"] - 1.0
    df["pm_vwap_vs_bos_pct"] = df["pm_vwap"] / df["bos_level"] - 1.0
    return df


def classify_preopen_status(
    df: pd.DataFrame, cfg: dict[str, float] | None = None,
) -> pd.DataFrame:
    cfg = dict(DEFAULT_CONFIG) if cfg is None else cfg
    df = df.copy()
    df["pm_volume_meaningful"] = df["pm_volume_ratio"] >= cfg["meaningful_pm_volume_ratio"]
    df["pm_volume_strong"] = df["pm_volume_ratio"] >= cfg["strong_pm_volume_ratio"]
    df["pm_volume_very_low"] = df["pm_volume_ratio"] < cfg["very_low_pm_volume_ratio"]

    statuses: list[str] = []
    reasons_list: list[str] = []

    for _, r in df.iterrows():
        rejects: list[str] = []
        watches: list[str] = []

        # ---- Minimum data checks → WATCH ----
        pm_last_missing = pd.isna(r["pm_last"])
        if pm_last_missing:
            watches.append("missing_pm_last")
        if pd.isna(r["atr_14"]) or r["atr_14"] <= 0:
            watches.append("invalid_atr")
        if pd.isna(r["bos_level"]):
            watches.append("missing_bos_level")
        if pd.isna(r["stop_ref"]):
            watches.append("missing_stop_ref")
        missing_spread = pd.isna(r["pm_bid"]) or pd.isna(r["pm_ask"])
        if missing_spread:
            watches.append("missing_spread_data")

        # Reject rules need pm_last + the relevant level
        can_check = not pm_last_missing

        # ---- Reject rules ----
        if can_check and not pd.isna(r["bos_level"]):
            pm_meaningful = bool(r["pm_volume_meaningful"])
            if r["pm_last"] < r["bos_level"] and pm_meaningful:
                rejects.append("pm_last_below_bos_with_volume")
            if (not pd.isna(r["pm_vwap"]) and r["pm_vwap"] < r["bos_level"]
                    and pm_meaningful):
                rejects.append("pm_vwap_below_bos_with_volume")
        if can_check and not pd.isna(r["stop_ref"]):
            if r["pm_last"] <= r["stop_ref"]:
                rejects.append("pm_last_below_stop_ref")
        if (can_check and not pd.isna(r["pm_low"])
                and not pd.isna(r["stop_ref"]) and not pd.isna(r["bos_level"])):
            if r["pm_low"] < r["stop_ref"] and r["pm_last"] < r["bos_level"]:
                rejects.append("pm_low_below_stop_and_not_recovered")
        if not pd.isna(r["gap_atr"]) and r["gap_atr"] > cfg["reject_gap_atr"]:
            rejects.append("gap_atr_too_large")
        if not pd.isna(r["gap_pct"]) and r["gap_pct"] > cfg["reject_gap_pct"]:
            rejects.append("gap_pct_too_large")
        if (not missing_spread and not pd.isna(r["spread_pct"])
                and r["spread_pct"] > cfg["reject_spread_pct"]):
            rejects.append("spread_too_wide")
        if bool(r.get("halted", False)):
            rejects.append("halted_preopen")

        # ---- Watch rules ----
        if can_check and not pd.isna(r["bos_level"]):
            if r["pm_last"] < r["bos_level"] and not bool(r["pm_volume_meaningful"]):
                watches.append("pm_below_bos_low_volume")
            if (not pd.isna(r["pm_low"]) and r["pm_low"] < r["bos_level"]
                    and r["pm_last"] >= r["bos_level"]):
                watches.append("intrapremarket_bos_breach_recovered")
        if not pd.isna(r["gap_atr"]):
            if cfg["watch_gap_atr"] < r["gap_atr"] <= cfg["reject_gap_atr"]:
                watches.append("moderate_gap_atr_chase_risk")
        if not pd.isna(r["gap_pct"]):
            if cfg["watch_gap_pct"] < r["gap_pct"] <= cfg["reject_gap_pct"]:
                watches.append("moderate_gap_pct_chase_risk")
        if (not missing_spread and not pd.isna(r["spread_pct"])
                and cfg["watch_spread_pct"] < r["spread_pct"] <= cfg["reject_spread_pct"]):
            watches.append("moderate_spread")
        if bool(r["pm_volume_very_low"]):
            watches.append("very_low_pm_volume")
        if bool(r.get("news_flag", False)):
            watches.append("news_flag_open_confirmation_needed")

        # ---- Precedence: REJECT > WATCH > PASS ----
        if rejects:
            statuses.append("REJECT_PREOPEN")
            reasons_list.append("|".join(rejects))
        elif watches:
            statuses.append("WATCH_OPEN")
            reasons_list.append("|".join(watches))
        else:
            statuses.append("PASS_PREOPEN")
            reasons_list.append("")

    df["preopen_status"] = statuses
    df["preopen_reasons"] = reasons_list
    return df


def write_output(df: pd.DataFrame, path: str | Path) -> None:
    extras = [
        "preopen_status", "preopen_reasons",
        "gap_pct", "gap_atr", "spread_pct", "pm_volume_ratio",
        "pm_last_vs_bos_pct", "pm_last_vs_stop_pct", "pm_vwap_vs_bos_pct",
        "pm_volume_meaningful", "pm_volume_strong", "pm_volume_very_low",
        "snapshot_datetime_et",
    ]
    original = [c for c in df.columns if c not in extras]
    out = df[original + [c for c in extras if c in df.columns]]
    _write_any(out, path)


def _self_test() -> None:
    """Three rows: one each for PASS, WATCH, REJECT — assert correctness."""
    cand = pd.DataFrame([
        {"ticker": "PASS1", "signal_date": "2026-05-22", "close_t": 100.0,
         "bos_level": 100.0, "stop_ref": 95.0, "atr_14": 2.0,
         "volume_t": 1_000_000, "avg_volume_20": 1_000_000,
         "dollar_volume_t": 100_000_000, "setup_score": 0.81},
        {"ticker": "WATCH1", "signal_date": "2026-05-22", "close_t": 100.0,
         "bos_level": 100.0, "stop_ref": 95.0, "atr_14": 2.0,
         "volume_t": 1_000_000, "avg_volume_20": 1_000_000,
         "dollar_volume_t": 100_000_000, "setup_score": 0.74},
        {"ticker": "REJECT1", "signal_date": "2026-05-22", "close_t": 100.0,
         "bos_level": 100.0, "stop_ref": 95.0, "atr_14": 2.0,
         "volume_t": 1_000_000, "avg_volume_20": 1_000_000,
         "dollar_volume_t": 100_000_000, "setup_score": 0.79},
    ])
    pm = pd.DataFrame([
        # PASS: pm 101 > bos, modest gap (+1% / +0.5 ATR), strong volume
        {"ticker": "PASS1", "snapshot_datetime_et": "2026-05-26 09:20:00",
         "pm_last": 101.0, "pm_bid": 100.95, "pm_ask": 101.05,
         "pm_volume": 100_000, "pm_vwap": 100.8, "pm_high": 101.2, "pm_low": 100.6,
         "halted": False, "news_flag": False},
        # WATCH: pm 99 < bos but pm_volume 2k / 1M = 0.002 (very_low, not meaningful)
        {"ticker": "WATCH1", "snapshot_datetime_et": "2026-05-26 09:20:00",
         "pm_last": 99.0, "pm_bid": 98.95, "pm_ask": 99.05,
         "pm_volume": 2_000, "pm_vwap": 99.0, "pm_high": 99.5, "pm_low": 98.5,
         "halted": False, "news_flag": False},
        # REJECT: pm 98 < bos and pm_volume 100k / 1M = 0.10 (meaningful)
        {"ticker": "REJECT1", "snapshot_datetime_et": "2026-05-26 09:20:00",
         "pm_last": 98.0, "pm_bid": 97.95, "pm_ask": 98.05,
         "pm_volume": 100_000, "pm_vwap": 98.5, "pm_high": 98.8, "pm_low": 97.5,
         "halted": False, "news_flag": False},
    ])

    validate_schema(cand, pm)
    feat = compute_preopen_features(cand, pm)
    out = classify_preopen_status(feat)
    statuses = dict(zip(out["ticker"], out["preopen_status"]))
    reasons = dict(zip(out["ticker"], out["preopen_reasons"]))

    assert statuses["PASS1"] == "PASS_PREOPEN", (
        f"PASS1 expected PASS_PREOPEN, got {statuses['PASS1']} "
        f"(reasons={reasons['PASS1']!r})"
    )
    assert reasons["PASS1"] == "", f"PASS1 reasons should be empty, got {reasons['PASS1']!r}"

    assert statuses["WATCH1"] == "WATCH_OPEN", (
        f"WATCH1 expected WATCH_OPEN, got {statuses['WATCH1']} "
        f"(reasons={reasons['WATCH1']!r})"
    )
    assert "pm_below_bos_low_volume" in reasons["WATCH1"], (
        f"WATCH1 should flag pm_below_bos_low_volume; got {reasons['WATCH1']!r}"
    )
    assert "very_low_pm_volume" in reasons["WATCH1"], (
        f"WATCH1 should flag very_low_pm_volume; got {reasons['WATCH1']!r}"
    )

    assert statuses["REJECT1"] == "REJECT_PREOPEN", (
        f"REJECT1 expected REJECT_PREOPEN, got {statuses['REJECT1']} "
        f"(reasons={reasons['REJECT1']!r})"
    )
    assert "pm_last_below_bos_with_volume" in reasons["REJECT1"], (
        f"REJECT1 should flag pm_last_below_bos_with_volume; got {reasons['REJECT1']!r}"
    )

    print("self-test PASSED")
    cols = ["ticker", "preopen_status", "preopen_reasons", "gap_pct", "gap_atr",
            "pm_volume_ratio", "pm_last_vs_bos_pct", "pm_last_vs_stop_pct"]
    print(out[cols].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    ap.add_argument("--candidates")
    ap.add_argument("--premarket")
    ap.add_argument("--output")
    ap.add_argument("--config", default=None)
    ap.add_argument("--snapshot-time-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not (args.candidates and args.premarket and args.output):
        ap.error("--candidates, --premarket, --output required (or use --self-test)")

    cfg = _load_config(args.config)
    cand, pm = load_inputs(args.candidates, args.premarket)
    validate_schema(cand, pm)
    feat = compute_preopen_features(cand, pm, snapshot_time_check=args.snapshot_time_check)
    out = classify_preopen_status(feat, cfg)
    write_output(out, args.output)

    dist = out["preopen_status"].value_counts().to_dict()
    print(f"wrote {len(out)} rows → {args.output}")
    print(f"status: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
