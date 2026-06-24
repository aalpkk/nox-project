"""Ranking Lab v0 — Cluster-3 Archetype Forward Scanner / Paper-Tracker (research-only).

The high-momentum winner archetype ("Cluster 3") cleared the pre-registered gates on
the F4 OOS holdout (lift 3.37x) and held across all 4 leave-one-fold-out rotations
(3.14-3.56x, strongest-of-4 in 4/4 — see ranking_lab_winner_cluster_lofo_v0.md). The
only missing leg is FORWARD evidence: snapshot archetype matches in real time, let the
20d window close, and measure realized y20_50 going forward. This scanner does exactly
that, with the model FROZEN so forward results are not contaminated by re-fitting.

Modes:
  freeze  Fit canonical KMeans(k=4) on F1+F2+F3 winners, pick archetype = max
          fast_rank_early centroid, persist the frozen model artifact (JSON).
  scan    Project a panel as-of a date onto the frozen centroids, flag archetype
          members, append LOCKED paper-tracking snapshot rows (forward outcomes NaN).
  fill    Read OHLCV master, back-fill realized mfe_pct_20d / y20_50 for snapshots
          whose 20-trading-day window has closed. Report rolling archetype lift.
  status  Print snapshot ledger summary (open vs realized, realized lift).

Target def (matches training): mfe_pct_20d = max(High over next 20 trading days,
  EXCLUDING signal day) / Close[signal_date] - 1 ; y20_50 = mfe_pct_20d >= 0.50.

Frozen archetype is identical across runs (seed=42, n_init=20, fixed train folds).
No tuning. FAST_RANK_v0 untouched. Live gate CLOSED — paper-tracking only, no trades.
No commit without authorization.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEATURES = OUT / "ranking_lab_features_v0.parquet"
SCORES = OUT / "ranking_lab_fast_rank_scores_v0.parquet"
OHLCV = OUT / "ohlcv_10y_fintables_master.parquet"
EXTFEED = OUT / "extfeed_intraday_1h_3y_master.parquet"

ARTIFACT = OUT / "ranking_lab_cluster3_model_v0.json"
SNAPSHOTS = OUT / "ranking_lab_cluster3_paper_snapshots_v0.parquet"
STATUS_MD = OUT / "ranking_lab_cluster3_paper_status_v0.md"

TARGET_NAME = "y20_50"
TARGET_COL = "mfe_pct_20d"
TARGET_THR = 0.50
FWD_BARS = 20

CONT_FEATURES = [
    "fast_rank_early", "p3_8", "p5_10", "p10_15",
    "ret_1d", "ret_5d",
    "price_vs_20d_high", "price_vs_60d_high",
    "atr_pct", "vol_z_20", "gap_pct", "xu100_atr_pct",
    "liquidity_proxy_log",
    "n_concurrent_sources", "n_concurrent_timeframes",
]
K_FIXED = 4
RANDOM_STATE = 42
TRAIN_FOLDS = ["F1", "F2", "F3"]
ARCHETYPE_KEY = "fast_rank_early"
SNAPSHOT_VERSION = "CLUSTER3_v0"

# reference (frozen at validation): archetype validated rate on F4 OOS = 0.1475
REF_VALIDATED_RATE = 0.1475
KILL_ROLLING_LIFT = 1.5  # STOP if rolling realized lift drops below this (>=30 closed)


# ----------------------------------------------------------------------------- io
def load_panel() -> pd.DataFrame:
    feat = pd.read_parquet(FEATURES)
    feat["signal_date"] = pd.to_datetime(feat["signal_date"]).dt.normalize()
    sc = pd.read_parquet(SCORES)
    sc["signal_date"] = pd.to_datetime(sc["signal_date"]).dt.normalize()
    sc = sc[["ticker", "signal_date", "family", "fold",
             "p3_8", "p5_10", "p10_15", "fast_rank_early"]].drop_duplicates(
        ["ticker", "signal_date", "family"], keep="first")
    df = feat.merge(sc, on=["ticker", "signal_date", "family"], how="left")
    df["liquidity_proxy_log"] = np.log1p(df["liquidity_proxy"].clip(lower=0))
    df[TARGET_NAME] = (df[TARGET_COL] >= TARGET_THR).astype("Int64")
    df.loc[df[TARGET_COL].isna(), TARGET_NAME] = pd.NA
    return df


def impute(df: pd.DataFrame, medians: dict) -> np.ndarray:
    X = df[CONT_FEATURES].copy()
    for c in CONT_FEATURES:
        X[c] = X[c].fillna(medians[c])
    return X.to_numpy()


# -------------------------------------------------------------------------- freeze
def cmd_freeze(force: bool = False) -> int:
    if ARTIFACT.exists() and not force:
        print(f"[freeze] artifact exists: {ARTIFACT} (use --force to refit)")
        return 0
    df = load_panel()
    tw = df[df["fold"].isin(TRAIN_FOLDS) & (df[TARGET_NAME] == 1)].copy()
    medians = {c: float(tw[CONT_FEATURES][c].median(skipna=True)) for c in CONT_FEATURES}
    X_raw = impute(tw, medians)
    scaler = StandardScaler().fit(X_raw)
    km = KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE).fit(
        scaler.transform(X_raw))
    cent_raw = scaler.inverse_transform(km.cluster_centers_)
    cent_df = pd.DataFrame(cent_raw, columns=CONT_FEATURES)
    arche = int(cent_df[ARCHETYPE_KEY].idxmax())

    artifact = {
        "version": SNAPSHOT_VERSION,
        "frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "train_folds": TRAIN_FOLDS,
        "n_train_winners": int(len(tw)),
        "k": K_FIXED, "random_state": RANDOM_STATE,
        "features": CONT_FEATURES,
        "medians": medians,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "centroids_std": km.cluster_centers_.tolist(),
        "archetype_cluster": arche,
        "archetype_centroid_raw": cent_df.loc[arche].to_dict(),
        "target": {"name": TARGET_NAME, "col": TARGET_COL, "thr": TARGET_THR,
                   "fwd_bars": FWD_BARS},
        "ref_validated_rate_F4": REF_VALIDATED_RATE,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2))
    print(f"[freeze] archetype cluster={arche}  fast_rank_early="
          f"{cent_df.loc[arche, ARCHETYPE_KEY]:.3f}  p3_8={cent_df.loc[arche,'p3_8']:.3f}  "
          f"ret_5d={cent_df.loc[arche,'ret_5d']:.3f}  "
          f"price_vs_60d_high={cent_df.loc[arche,'price_vs_60d_high']:.3f}")
    print(f"[freeze] wrote {ARTIFACT}  (n_train_winners={len(tw)})")
    return 0


def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise SystemExit("[err] model artifact missing — run `freeze` first")
    return json.loads(ARTIFACT.read_text())


def assign_clusters(df: pd.DataFrame, art: dict) -> np.ndarray:
    medians = art["medians"]
    mean = np.array(art["scaler_mean"]); scale = np.array(art["scaler_scale"])
    cent = np.array(art["centroids_std"])
    X = impute(df, medians)
    Xs = (X - mean) / scale
    # nearest centroid (euclidean in standardized space)
    d = ((Xs[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


# ---------------------------------------------------------------------------- scan
def cmd_scan(asof: str | None, seed_backlog: bool) -> int:
    art = load_artifact()
    arche = art["archetype_cluster"]
    df = load_panel()
    asof_ts = pd.Timestamp(asof).normalize() if asof else df["signal_date"].max()

    pool = df[df["signal_date"] <= asof_ts].copy()
    if seed_backlog:
        # all currently-open (unrealized) candidates up to asof
        cand = pool[pool[TARGET_COL].isna()].copy()
        scope = "seed-backlog (all unrealized)"
    else:
        cand = pool[pool["signal_date"] == asof_ts].copy()
        scope = f"asof-only ({asof_ts.date()})"
    if cand.empty:
        print(f"[scan] no candidates for scope={scope}")
        return 0

    cand["_cluster"] = assign_clusters(cand, art)
    arch = cand[cand["_cluster"] == arche].copy()
    print(f"[scan] scope={scope}  pool_rows={len(cand)}  archetype_matches={len(arch)} "
          f"({100*len(arch)/len(cand):.1f}%)")
    if arch.empty:
        return 0

    snap = pd.DataFrame({
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_asof_date": asof_ts.normalize(),
        "snapshot_run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "snapshot_locked": True,
        "ticker": arch["ticker"].values,
        "signal_date": arch["signal_date"].values,
        "family": arch["family"].values,
        "source": arch["source"].values,
        "cluster_assigned": arch["_cluster"].values,
        "is_archetype": True,
    })
    for c in CONT_FEATURES:
        snap[c] = arch[c].values
    # forward outcome cols (filled later)
    snap["mfe_pct_20d_realized"] = np.nan
    snap["y20_50_realized"] = pd.NA
    snap["realized_date_20d"] = pd.NaT
    snap["entry_close"] = np.nan
    # if already realized in the historical panel (backfill-known), keep for reference
    snap["mfe_pct_20d_known_panel"] = arch[TARGET_COL].values

    key = ["snapshot_version", "ticker", "signal_date", "family"]
    if SNAPSHOTS.exists():
        prev = pd.read_parquet(SNAPSHOTS)
        prev["signal_date"] = pd.to_datetime(prev["signal_date"])
        merged = pd.concat([prev, snap], ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=key, keep="first")
        n_new = len(merged) - len(prev)
        print(f"[scan] appended {n_new} new locked rows (dedup removed "
              f"{before - len(merged)}); ledger now {len(merged)}")
    else:
        merged = snap
        print(f"[scan] created ledger with {len(merged)} locked rows")
    merged.to_parquet(SNAPSHOTS, index=False)
    print(f"[scan] wrote {SNAPSHOTS}")
    return 0


# ---------------------------------------------------------------------------- fill
def _load_ohlcv() -> pd.DataFrame:
    """Daily High/Close per ticker, resampled from the 1h extfeed master.

    The features pipeline derives mfe_pct_20d from the SAME extfeed (1h→daily,
    Europe/Istanbul), so this is the definition-matched forward source. The
    fintables OHLCV master is sparse for many tickers and is NOT used here.
    Bar COUNT defines the 20-trading-day window, so market holidays (e.g. the
    2026-05-27..29 Kurban Bayramı closure) are handled correctly — closed days
    simply have no bar and do not consume a forward slot.
    """
    ext = pd.read_parquet(EXTFEED, columns=["ticker", "ts_utc", "high", "close"])
    d = pd.to_datetime(ext["ts_utc"]).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    ext["Date"] = d.dt.normalize()
    daily = (ext.groupby(["ticker", "Date"])
             .agg(High=("high", "max"), Close=("close", "last"))
             .reset_index())
    return daily.sort_values(["ticker", "Date"])


def cmd_fill() -> int:
    if not SNAPSHOTS.exists():
        raise SystemExit("[err] no snapshots to fill — run `scan` first")
    led = pd.read_parquet(SNAPSHOTS)
    led["signal_date"] = pd.to_datetime(led["signal_date"])
    open_mask = led["mfe_pct_20d_realized"].isna()
    n_open = int(open_mask.sum())
    if n_open == 0:
        print("[fill] nothing open to fill")
        return _write_status(led)
    o = _load_ohlcv()
    by_tk = {tk: g.reset_index(drop=True) for tk, g in o.groupby("ticker")}

    filled = 0
    for i in led[open_mask].index:
        tk = led.at[i, "ticker"]; d = led.at[i, "signal_date"]
        g = by_tk.get(tk)
        if g is None:
            continue
        after = g[g["Date"] >= d]
        if len(after) < FWD_BARS + 1:   # need signal bar + 20 forward bars closed
            continue
        entry = float(after.iloc[0]["Close"])
        fwd = after.iloc[1:FWD_BARS + 1]
        if entry <= 0 or fwd["High"].isna().all():
            continue
        mfe = float(fwd["High"].max() / entry - 1.0)
        led.at[i, "entry_close"] = entry
        led.at[i, "mfe_pct_20d_realized"] = mfe
        led.at[i, "y20_50_realized"] = bool(mfe >= TARGET_THR)
        led.at[i, "realized_date_20d"] = fwd.iloc[-1]["Date"]
        filled += 1
    led.to_parquet(SNAPSHOTS, index=False)
    print(f"[fill] filled {filled} newly-closed windows; "
          f"still open: {int(led['mfe_pct_20d_realized'].isna().sum())}")
    return _write_status(led)


def _write_status(led: pd.DataFrame) -> int:
    art = load_artifact() if ARTIFACT.exists() else {}
    realized = led[led["mfe_pct_20d_realized"].notna()].copy()
    n_real = len(realized)
    n_open = int(led["mfe_pct_20d_realized"].isna().sum())
    lines = ["# Cluster-3 Archetype Paper-Tracker — status", ""]
    lines.append(f"- snapshot_version: **{SNAPSHOT_VERSION}**  •  live gate: **CLOSED** (paper only)")
    lines.append(f"- ledger rows: **{len(led)}**  (realized: {n_real}, open: {n_open})")
    if n_real:
        rate = float((realized["y20_50_realized"] == True).mean())  # noqa: E712
        lines.append(f"- realized y20_50 hit-rate: **{rate:.4f}**  "
                     f"(validation F4 ref: {REF_VALIDATED_RATE})")
        lines.append(f"- realized MFE_20d: median **{realized['mfe_pct_20d_realized'].median():.4f}**, "
                     f"p90 **{realized['mfe_pct_20d_realized'].quantile(0.90):.4f}**")
        # rolling lift kill-criterion (needs a panel base; use ref rate as proxy base)
        if n_real >= 30:
            lift_vs_ref = rate / REF_VALIDATED_RATE if REF_VALIDATED_RATE > 0 else float("nan")
            verdict = ("STOP_OPERATIONALIZATION" if rate < (KILL_ROLLING_LIFT * 0.0437)
                       else "HEALTHY")
            lines.append(f"- realized rate vs validated ref ratio: **{lift_vs_ref:.2f}** "
                         f"(≥30 closed) → **{verdict}**")
        else:
            lines.append(f"- kill-criteria pending: need ≥30 closed (have {n_real})")
        # top realized names
        top = realized.sort_values("mfe_pct_20d_realized", ascending=False).head(10)
        lines.append("")
        lines.append("## Top realized (by MFE_20d)")
        lines.append(top[["ticker", "signal_date", "mfe_pct_20d_realized",
                          "y20_50_realized"]].to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("- no closed windows yet (all candidates still open)")
    STATUS_MD.write_text("\n".join(lines))
    print(f"[status] wrote {STATUS_MD}  (realized={n_real}, open={n_open})")
    return 0


def cmd_status() -> int:
    if not SNAPSHOTS.exists():
        print("[status] no ledger yet")
        return 0
    led = pd.read_parquet(SNAPSHOTS)
    led["signal_date"] = pd.to_datetime(led["signal_date"])
    return _write_status(led)


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster-3 archetype forward paper-tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_fr = sub.add_parser("freeze"); p_fr.add_argument("--force", action="store_true")
    p_sc = sub.add_parser("scan")
    p_sc.add_argument("--asof", default=None, help="YYYY-MM-DD (default: latest panel date)")
    p_sc.add_argument("--seed-backlog", action="store_true",
                      help="snapshot ALL currently-unrealized archetype candidates")
    sub.add_parser("fill")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "freeze":
        return cmd_freeze(force=a.force)
    if a.cmd == "scan":
        return cmd_scan(asof=a.asof, seed_backlog=a.seed_backlog)
    if a.cmd == "fill":
        return cmd_fill()
    if a.cmd == "status":
        return cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
