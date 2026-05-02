"""HW Overlay v1 trade builder.

Spec: memory/hw_overlay_v1_spec.md.

Logic per event:
  1. Find first bullish HW 1d cross (kind ∈ {AL, AL_OS}) within
     [event_date, event_date + N_trading_days), inclusive of event_date.
     N = 10 trading days for nyxmomentum, 3 for the other 5 scanners.
  2. If no entry → entry_date NaT, no metrics row, kept in roster (entry_fill=False).
  3. From entry, look for first bearish HW 1d cross (kind ∈ {SAT, SAT_OB}) OR
     entry_bar + 10 trading days (whichever first).
  4. Compute realized_R = (exit_close - entry_close) / entry_close.
  5. MFE_R during holding = max(daily_high) / entry_close - 1, daily series
     between entry_date and exit_date inclusive.
  6. is_win = (realized_R >= +0.10) AND (holding_days <= 10).

Output: output/hw_overlay_v1_events.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.hw_overlay.adapters import build_all

DEFAULT_HW_CSV = "output/hwo_full_1d_scan_2026-04-30.csv"
DEFAULT_MASTER = "output/extfeed_intraday_1h_3y_master.parquet"
DEFAULT_OUT = "output/hw_overlay_v1_events.parquet"

WAIT_N_BY_SCANNER = {
    "nyxmomentum": 10,
}
DEFAULT_WAIT_N = 3
TIME_STOP_BARS = 10

# Arms E/F/G: SAT-as-profit-protection variants on the Arm B entry.
#  E:   SAT_OB-only exit, normal SAT ignored, force-close at +10d.
#  F5:  any SAT/SAT_OB, but only takes effect once running MFE ≥ +5%.
#  F10: any SAT/SAT_OB, gated by running MFE ≥ +10%.
#  G5:  SAT_OB-only, gated by MFE ≥ +5%.
#  G10: SAT_OB-only, gated by MFE ≥ +10%.
EXIT_POLICIES: dict[str, dict] = {
    "E":   {"allowed_kinds": frozenset({"SAT_OB"}),         "mfe_threshold": None},
    "F5":  {"allowed_kinds": frozenset({"SAT", "SAT_OB"}),  "mfe_threshold": 0.05},
    "F10": {"allowed_kinds": frozenset({"SAT", "SAT_OB"}),  "mfe_threshold": 0.10},
    "G5":  {"allowed_kinds": frozenset({"SAT_OB"}),         "mfe_threshold": 0.05},
    "G10": {"allowed_kinds": frozenset({"SAT_OB"}),         "mfe_threshold": 0.10},
}


def _apply_exit_policy(
    e_idx: int,
    entry_close: float,
    dates: np.ndarray,
    daily: pd.DataFrame,
    bear_in: pd.DataFrame,
    *,
    allowed_kinds: frozenset,
    mfe_threshold: float | None,
) -> dict:
    """Apply an exit policy on top of the Arm B entry.

    Walks bear HW events in (entry, entry+TIME_STOP_BARS]; takes the first one
    whose kind ∈ allowed_kinds AND (no gate, or running MFE ≥ mfe_threshold).
    Falls back to force_close at entry + TIME_STOP_BARS if no exit triggers.
    """
    stop_idx = min(e_idx + TIME_STOP_BARS, len(dates) - 1)
    if stop_idx <= e_idx:
        return {
            "exit_close": np.nan, "holding": np.nan, "exit_kind": None,
            "realized_R": np.nan, "mfe_R": np.nan, "giveback": np.nan,
            "is_win": False,
        }
    highs = daily["high"].values[e_idx: stop_idx + 1]
    cummax_highs = np.maximum.accumulate(highs)
    triggered_idx = None
    triggered_kind = None
    if not bear_in.empty:
        for ts_val, kind_val in zip(bear_in["ts"].values, bear_in["kind"].values):
            if kind_val not in allowed_kinds:
                continue
            ts = pd.Timestamp(ts_val)
            h_idx = int(np.searchsorted(dates, ts.to_datetime64(), side="left"))
            if h_idx >= len(dates) or pd.Timestamp(dates[h_idx]) != ts:
                continue
            if h_idx <= e_idx or h_idx > stop_idx:
                continue
            if mfe_threshold is not None:
                local_offset = h_idx - e_idx
                mfe_so_far = (cummax_highs[local_offset] - entry_close) / entry_close
                if mfe_so_far < mfe_threshold:
                    continue
            triggered_idx = h_idx
            triggered_kind = str(kind_val)
            break
    if triggered_idx is None:
        exit_idx = stop_idx
        exit_kind = "force_close"
    else:
        exit_idx = triggered_idx
        exit_kind = triggered_kind
    exit_close = float(daily.iloc[exit_idx]["close"])
    holding = exit_idx - e_idx
    realized_R = (exit_close - entry_close) / entry_close
    mfe_over_hold = float(cummax_highs[exit_idx - e_idx])
    mfe_R_arm = (mfe_over_hold - entry_close) / entry_close
    giveback = mfe_R_arm - realized_R
    is_win = bool(realized_R >= 0.10 and holding <= TIME_STOP_BARS)
    return {
        "exit_close": exit_close, "holding": holding, "exit_kind": exit_kind,
        "realized_R": realized_R, "mfe_R": mfe_R_arm, "giveback": giveback,
        "is_win": is_win,
    }


def _empty_exit_arm_block() -> dict:
    """Null block for the 5 EFG arms when entry didn't fill or window is empty."""
    out = {}
    for k in EXIT_POLICIES:
        out[f"arm{k}_exit_close"] = np.nan
        out[f"arm{k}_holding_days"] = np.nan
        out[f"arm{k}_exit_kind"] = None
        out[f"arm{k}_realized_R"] = np.nan
        out[f"arm{k}_mfe_R"] = np.nan
        out[f"arm{k}_giveback"] = np.nan
        out[f"arm{k}_is_win"] = False
    return out


def daily_panel_from_master(master_path: str) -> dict[str, pd.DataFrame]:
    """Build per-ticker daily OHLCV panel from 1h master."""
    print(f"[panel] loading master {master_path}")
    mp = pd.read_parquet(master_path)
    mp["ts_istanbul"] = pd.to_datetime(mp["ts_istanbul"])
    mp["date"] = mp["ts_istanbul"].dt.tz_localize(None).dt.normalize()
    daily = (
        mp.groupby(["ticker", "date"])
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"),
             volume=("volume", "sum"))
        .reset_index()
        .sort_values(["ticker", "date"])
    )
    panel = {tk: g.reset_index(drop=True) for tk, g in daily.groupby("ticker")}
    print(f"[panel] {len(panel)} tickers, total daily rows {len(daily):,}")
    return panel


def hw_events_by_ticker(hw_csv: str) -> dict[str, pd.DataFrame]:
    print(f"[hw] loading {hw_csv}")
    h = pd.read_csv(hw_csv)
    h["ts"] = pd.to_datetime(h["ts"]).dt.normalize()
    h = h.sort_values(["ticker", "ts"])
    bull = {"AL", "AL_OS"}
    bear = {"SAT", "SAT_OB"}
    h["dir"] = h["kind"].apply(lambda k: "bull" if k in bull else ("bear" if k in bear else None))
    return {tk: g.reset_index(drop=True) for tk, g in h.groupby("ticker")}


def build_trade(
    event_row: pd.Series,
    daily: pd.DataFrame,
    hw: pd.DataFrame,
    wait_n: int,
) -> dict | None:
    ticker = event_row["ticker"]
    event_date = pd.Timestamp(event_row["event_date"])
    # ticker calendar
    dates = daily["date"].values
    if event_date < dates[0] or event_date > dates[-1]:
        return None
    # find first daily bar idx with date >= event_date
    idx = int(np.searchsorted(dates, event_date.to_datetime64(), side="left"))
    if idx >= len(dates):
        return None
    # === Arm A: scanner-alone baseline (enter event_date close, hold 10 trading days) ===
    armA_exit_idx = min(idx + TIME_STOP_BARS, len(dates) - 1)
    if armA_exit_idx > idx:
        armA_entry_close = float(daily.iloc[idx]["close"])
        armA_exit_close = float(daily.iloc[armA_exit_idx]["close"])
        armA_holding = armA_exit_idx - idx
        armA_R = (armA_exit_close - armA_entry_close) / armA_entry_close
        armA_is_win = bool(armA_R >= 0.10 and armA_holding <= TIME_STOP_BARS)
    else:
        armA_entry_close = np.nan
        armA_exit_close = np.nan
        armA_holding = np.nan
        armA_R = np.nan
        armA_is_win = False

    entry_window_end_idx = min(idx + wait_n, len(dates))
    window_dates = pd.to_datetime(dates[idx:entry_window_end_idx])
    if len(window_dates) == 0:
        return None
    # find first bullish HW cross in this window
    if hw is None or hw.empty:
        bull_in = pd.DataFrame()
    else:
        mask = (hw["ts"] >= window_dates[0]) & (hw["ts"] <= window_dates[-1]) & (hw["dir"] == "bull")
        bull_in = hw[mask]
    if bull_in.empty:
        out = {
            "ticker": ticker,
            "event_date": event_date,
            "entry_filled": False,
            "entry_date": pd.NaT,
            "entry_signal_kind": None,
            "exit_date": pd.NaT,
            "exit_signal_kind": None,
            "entry_close": np.nan,
            "exit_close": np.nan,
            "holding_days": np.nan,
            "realized_R": np.nan,
            "mfe_R": np.nan,
            "is_win": False,
            "armA_entry_close": armA_entry_close,
            "armA_exit_close": armA_exit_close,
            "armA_holding_days": armA_holding,
            "armA_realized_R": armA_R,
            "armA_is_win": armA_is_win,
            "armC_exit_close": np.nan,
            "armC_holding_days": np.nan,
            "armC_realized_R": np.nan,
            "armC_is_win": False,
            "armD_realized_R": np.nan,
            "armD_n_cycles": 0,
            "armD_is_win": False,
        }
        out.update(_empty_exit_arm_block())
        return out
    entry_row = bull_in.iloc[0]
    entry_date = pd.Timestamp(entry_row["ts"])
    entry_kind = entry_row["kind"]
    # locate entry bar in daily
    e_idx = int(np.searchsorted(dates, entry_date.to_datetime64(), side="left"))
    if e_idx >= len(dates) or pd.Timestamp(dates[e_idx]) != entry_date:
        # HW cross is on a date not in daily (rare); fall back to next available
        if e_idx >= len(dates):
            return None
    entry_close = float(daily.iloc[e_idx]["close"])
    # exit search window: (entry_date, entry_date + TIME_STOP_BARS]
    stop_idx = min(e_idx + TIME_STOP_BARS, len(dates) - 1)
    exit_window_dates = pd.to_datetime(dates[e_idx + 1: stop_idx + 1])
    if len(exit_window_dates) == 0:
        return None  # no future bars to hold
    bear_mask = (hw["ts"] >= exit_window_dates[0]) & (hw["ts"] <= exit_window_dates[-1]) & (hw["dir"] == "bear")
    bear_in = hw[bear_mask]
    if not bear_in.empty:
        exit_row = bear_in.iloc[0]
        exit_date = pd.Timestamp(exit_row["ts"])
        exit_kind = exit_row["kind"]
    else:
        exit_date = pd.Timestamp(dates[stop_idx])
        exit_kind = "time_stop"
    x_idx = int(np.searchsorted(dates, exit_date.to_datetime64(), side="left"))
    if x_idx >= len(dates) or pd.Timestamp(dates[x_idx]) != exit_date:
        return None
    exit_close = float(daily.iloc[x_idx]["close"])
    holding_days = x_idx - e_idx
    realized_R = (exit_close - entry_close) / entry_close
    # MFE during holding (entry_date through exit_date inclusive, daily highs)
    holding_slice = daily.iloc[e_idx: x_idx + 1]
    if holding_slice.empty:
        mfe_R = np.nan
    else:
        peak_high = float(holding_slice["high"].max())
        mfe_R = (peak_high - entry_close) / entry_close
    is_win = bool((realized_R >= 0.10) and (holding_days <= 10))
    # === Arm C: HW filter (entry on bullish HW), but NO SAT exit — fixed +10d hold ===
    armC_exit_idx = min(e_idx + TIME_STOP_BARS, len(dates) - 1)
    if armC_exit_idx > e_idx:
        armC_exit_close = float(daily.iloc[armC_exit_idx]["close"])
        armC_holding = armC_exit_idx - e_idx
        armC_R = (armC_exit_close - entry_close) / entry_close
        armC_is_win = bool(armC_R >= 0.10 and armC_holding <= TIME_STOP_BARS)
    else:
        armC_exit_close = np.nan
        armC_holding = np.nan
        armC_R = np.nan
        armC_is_win = False
    # === Arm D: multi-cycle in [entry, entry+10d] — AL→enter, SAT→exit, force close at window end ===
    armD_window_end_idx = min(e_idx + TIME_STOP_BARS, len(dates) - 1)
    if armD_window_end_idx <= e_idx:
        armD_R = np.nan
        armD_n_cycles = 0
        armD_is_win = False
    else:
        window_start_date = pd.Timestamp(dates[e_idx])
        window_end_date = pd.Timestamp(dates[armD_window_end_idx])
        # HW events strictly after entry, up to and including window end
        win_hw = hw[(hw["ts"] > window_start_date) & (hw["ts"] <= window_end_date)]
        cycles_R = []
        # Already entered at first AL (e_idx, entry_close)
        cur_entry_idx = e_idx
        cur_entry_close = entry_close
        in_position = True
        for hw_ts_val, hw_dir_val in zip(win_hw["ts"].values, win_hw["dir"].values):
            hw_ts = pd.Timestamp(hw_ts_val)
            hw_idx = int(np.searchsorted(dates, hw_ts.to_datetime64(), side="left"))
            if hw_idx >= len(dates) or pd.Timestamp(dates[hw_idx]) != hw_ts:
                continue
            if in_position:
                if hw_dir_val == "bear" and hw_idx > cur_entry_idx:
                    cyc_exit_close = float(daily.iloc[hw_idx]["close"])
                    cycles_R.append((cyc_exit_close - cur_entry_close) / cur_entry_close)
                    in_position = False
            else:
                if hw_dir_val == "bull" and hw_idx > cur_entry_idx:
                    cur_entry_idx = hw_idx
                    cur_entry_close = float(daily.iloc[hw_idx]["close"])
                    in_position = True
        # Force-close at window end if still in position
        if in_position and armD_window_end_idx > cur_entry_idx:
            forced_close = float(daily.iloc[armD_window_end_idx]["close"])
            cycles_R.append((forced_close - cur_entry_close) / cur_entry_close)
        armD_n_cycles = len(cycles_R)
        if armD_n_cycles > 0:
            cum = 1.0
            for r in cycles_R:
                cum *= (1.0 + r)
            armD_R = cum - 1.0
            armD_is_win = bool(armD_R >= 0.10)
        else:
            armD_R = np.nan
            armD_is_win = False
    # === Arms E/F5/F10/G5/G10 (SAT-as-profit-protection variants on Arm B entry) ===
    arm_efg_block: dict = {}
    for arm_key, cfg in EXIT_POLICIES.items():
        res = _apply_exit_policy(
            e_idx, entry_close, dates, daily, bear_in,
            allowed_kinds=cfg["allowed_kinds"],
            mfe_threshold=cfg["mfe_threshold"],
        )
        arm_efg_block[f"arm{arm_key}_exit_close"] = res["exit_close"]
        arm_efg_block[f"arm{arm_key}_holding_days"] = res["holding"]
        arm_efg_block[f"arm{arm_key}_exit_kind"] = res["exit_kind"]
        arm_efg_block[f"arm{arm_key}_realized_R"] = res["realized_R"]
        arm_efg_block[f"arm{arm_key}_mfe_R"] = res["mfe_R"]
        arm_efg_block[f"arm{arm_key}_giveback"] = res["giveback"]
        arm_efg_block[f"arm{arm_key}_is_win"] = res["is_win"]
    out = {
        "ticker": ticker,
        "event_date": event_date,
        "entry_filled": True,
        "entry_date": entry_date,
        "entry_signal_kind": entry_kind,
        "exit_date": exit_date,
        "exit_signal_kind": exit_kind,
        "entry_close": entry_close,
        "exit_close": exit_close,
        "holding_days": holding_days,
        "realized_R": realized_R,
        "mfe_R": mfe_R,
        "is_win": is_win,
        "armA_entry_close": armA_entry_close,
        "armA_exit_close": armA_exit_close,
        "armA_holding_days": armA_holding,
        "armA_realized_R": armA_R,
        "armA_is_win": armA_is_win,
        "armC_exit_close": armC_exit_close,
        "armC_holding_days": armC_holding,
        "armC_realized_R": armC_R,
        "armC_is_win": armC_is_win,
        "armD_realized_R": armD_R,
        "armD_n_cycles": armD_n_cycles,
        "armD_is_win": armD_is_win,
    }
    out.update(arm_efg_block)
    return out


def run(
    hw_csv: str = DEFAULT_HW_CSV,
    master_path: str = DEFAULT_MASTER,
    out_path: str = DEFAULT_OUT,
) -> pd.DataFrame:
    roster = build_all()
    print(f"[roster] {len(roster):,} events across {roster['scanner'].nunique()} scanners")
    panel = daily_panel_from_master(master_path)
    hw_by_ticker = hw_events_by_ticker(hw_csv)

    out_rows = []
    n = len(roster)
    for i, ev in enumerate(roster.itertuples(index=False), 1):
        ev_dict = ev._asdict()
        ticker = ev_dict["ticker"]
        scanner = ev_dict["scanner"]
        wait_n = WAIT_N_BY_SCANNER.get(scanner, DEFAULT_WAIT_N)
        daily = panel.get(ticker)
        if daily is None or daily.empty:
            continue
        hw = hw_by_ticker.get(ticker, pd.DataFrame())
        trade = build_trade(pd.Series(ev_dict), daily, hw, wait_n)
        if trade is None:
            continue
        trade.update({
            "scanner": scanner,
            "family": ev_dict["family"],
            "slice_tags": ev_dict["slice_tags"],
        })
        out_rows.append(trade)
        if i % 20000 == 0:
            print(f"[build] {i}/{n}")

    out = pd.DataFrame(out_rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[done] wrote {len(out):,} rows → {out_path}")
    print("\n--- by scanner × entry_filled")
    print(out.groupby(["scanner", "entry_filled"]).size().to_string())
    print("\n--- entry_signal_kind for filled trades")
    print(out[out["entry_filled"]]["entry_signal_kind"].value_counts().to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-csv", default=DEFAULT_HW_CSV)
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args.hw_csv, args.master, args.out)


if __name__ == "__main__":
    main()
