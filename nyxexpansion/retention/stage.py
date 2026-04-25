"""Live timing-clean retention stage for the daily nyxexpansion v4C scan.

Pipeline (locked rule, see memory/nyxexp_timing_clean_retention_filter_locked.md):

    candidate_panel  (full-close winner_R_pred from scan_latest)
        ↓
    [truncated bars]  10:15..17:00 TR 15m bars from cache → daily OHLCV
        ↓
    [truncated features]  rebuild via compute_per_ticker_features with T patched
        ↓
    [surrogate score]  persisted UP/NONUP LightGBM heads → winner_R_pred_tr
        ↓
    [pessimistic rank]  rank within day's panel (competitors stay at full-close)
        ↓
    retention_pass = (rank ≤ 10)

Output columns added to the scan DataFrame:

- ``retention_pass``        bool
- ``rank_1700_surrogate``   nullable int (NaN if not scored)
- ``score_1700_surrogate``  float (NaN if not scored)
- ``timing_clean_note``     str: ok | bars_missing | feature_rebuild_failed
                                  | regime_unknown | stage_disabled

Fail-fast contract:
- Surrogate artifact missing → ``FileNotFoundError`` (no silent fallback).
- Surrogate schema version mismatch → ``SchemaVersionMismatch`` from the loader.
- Per-candidate failures (missing 15m bars, regime unknown, rebuild error) are
  recorded as ``retention_pass=False`` with a specific ``timing_clean_note``.
- If the caller bypasses this stage entirely, ``mark_stage_disabled`` writes a
  uniform note so the HTML report can surface that the gate did not run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nyxexpansion.retention import truncate as ret_truncate
from nyxexpansion.retention import surrogate as ret_surrogate

CS_FEATURES = ("rs_rank_cs_today", "breadth_ad_20d", "chase_score_soft")

NOTE_OK = "ok"
NOTE_BARS_MISSING = "bars_missing"
NOTE_REBUILD_FAILED = "feature_rebuild_failed"
NOTE_REGIME_UNKNOWN = "regime_unknown"
NOTE_STAGE_DISABLED = "stage_disabled"

NEW_COLS = (
    "retention_pass",
    "rank_1700_surrogate",
    "score_1700_surrogate",
    "timing_clean_note",
    "bars_source",
)


@dataclass
class StageOutcome:
    enriched: pd.DataFrame
    n_total: int
    n_pass: int
    n_drop: int
    n_unscored: int
    notes: dict[str, int]
    source_breakdown: dict[str, int]


def mark_stage_disabled(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with stage_disabled and ``retention_pass=False``."""
    out = scan_df.copy()
    out["retention_pass"] = False
    out["rank_1700_surrogate"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["score_1700_surrogate"] = np.nan
    out["timing_clean_note"] = NOTE_STAGE_DISABLED
    out["bars_source"] = pd.NA
    return out


def _load_intraday_for_target(
    intraday_15m_path: Path, target_date: pd.Timestamp,
    tickers: list[str],
) -> pd.DataFrame:
    """Pull only the rows of the cache that belong to (target_date, tickers).

    Returned frame has the schema expected by ``aggregate_truncated_bars``.
    """
    df = pd.read_parquet(intraday_15m_path)
    if "signal_date" in df.columns:
        sd = pd.to_datetime(df["signal_date"]).dt.normalize()
    else:
        sd = pd.to_datetime(df["bar_ts"], unit="ms", utc=True).dt.tz_convert(
            "Europe/Istanbul",
        ).dt.normalize().dt.tz_localize(None)
        df = df.assign(signal_date=sd)
    target_norm = pd.Timestamp(target_date).normalize()
    df = df[(sd == target_norm) & (df["ticker"].isin(tickers))].copy()
    df["signal_date"] = target_norm
    if "bars_source" not in df.columns:
        df["bars_source"] = "matriks_15m"
    return df


def _build_source_lookup(intraday: pd.DataFrame) -> dict[str, str]:
    """Per-ticker bars_source from the loaded intraday slice.

    If a ticker's bars come from a single source, return that source string.
    If multiple sources contribute (rare; would only happen across a mixed
    cache merge), return ``mixed``.
    """
    if intraday.empty or "bars_source" not in intraday.columns:
        return {}
    out: dict[str, str] = {}
    for tk, g in intraday.groupby("ticker"):
        srcs = g["bars_source"].dropna().astype(str).unique().tolist()
        if not srcs:
            continue
        out[str(tk)] = srcs[0] if len(srcs) == 1 else "mixed"
    return out


def _attach_panel_features(
    feats_tr: pd.DataFrame,
    scan_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Merge cross-sectional + regime metadata from the live scan panel.

    The cross-sectional features (rs_rank_cs, breadth, chase) are read from
    the candidate's row in ``scan_df`` because we don't have intraday data
    for ALL universe tickers. This mirrors the research surrogate's CS handling.
    """
    target_norm = pd.Timestamp(target_date).normalize()
    if feats_tr.empty:
        return feats_tr
    panel = scan_df[scan_df["date"].dt.normalize() == target_norm].copy()
    keep = ["ticker"] + [c for c in CS_FEATURES if c in panel.columns]
    keep += [c for c in ("model_kind", "xu_regime") for _ in [0] if c in panel.columns]
    keep = list(dict.fromkeys(keep))
    panel_keep = panel[keep].drop_duplicates("ticker")
    out = feats_tr.merge(panel_keep, on="ticker", how="left", suffixes=("", "_scan"))
    if "model_kind" not in out.columns and "xu_regime" in out.columns:
        out["model_kind"] = out["xu_regime"].apply(
            lambda r: "up" if r == "uptrend" else "nonup"
        )
    return out


def run_retention_stage(
    scan_df: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    intraday_15m_path: Path,
    master_ohlcv_path: Path,
    surrogate_artifact_path: Path,
    rank_threshold: int = 10,
    xu100_close: pd.Series | None = None,
) -> StageOutcome:
    """Apply the timing-clean retention filter to a live scan output.

    The scan_df is expected to already contain ``ticker``, ``date``,
    ``winner_R_pred``, ``model_kind`` (or ``xu_regime``), and the cross-sectional
    features listed in ``CS_FEATURES``.
    """
    if not Path(surrogate_artifact_path).exists():
        raise FileNotFoundError(
            f"retention surrogate artifact not found: {surrogate_artifact_path}"
        )
    surrogate = ret_surrogate.load(surrogate_artifact_path)

    target_norm = pd.Timestamp(target_date).normalize()
    panel = scan_df[scan_df["date"].dt.normalize() == target_norm].copy()
    if panel.empty:
        return StageOutcome(
            enriched=mark_stage_disabled(scan_df),
            n_total=0, n_pass=0, n_drop=0, n_unscored=0,
            notes={NOTE_STAGE_DISABLED: 0},
            source_breakdown={},
        )
    if "model_kind" not in panel.columns:
        if "xu_regime" not in panel.columns:
            raise ValueError("scan_df needs 'model_kind' or 'xu_regime'")
        panel["model_kind"] = panel["xu_regime"].apply(
            lambda r: "up" if r == "uptrend" else "nonup"
        )

    tickers = panel["ticker"].astype(str).tolist()
    intraday = _load_intraday_for_target(
        Path(intraday_15m_path), target_norm, tickers,
    )
    source_lookup = _build_source_lookup(intraday)

    note_per_ticker: dict[str, str] = {}
    for tk in tickers:
        if not (intraday["ticker"] == tk).any():
            note_per_ticker[tk] = NOTE_BARS_MISSING

    bars = ret_truncate.aggregate_truncated_bars(intraday) if not intraday.empty \
        else pd.DataFrame(columns=["ticker", "signal_date"])

    if bars.empty:
        feats_tr = pd.DataFrame()
    else:
        try:
            feats_tr = ret_truncate.rebuild_truncated_features(
                bars, master_ohlcv_path=master_ohlcv_path,
                xu100_close=xu100_close, n_limit=None, progress_every=0,
            )
        except Exception as exc:
            for tk in bars["ticker"].astype(str).unique():
                note_per_ticker.setdefault(tk, NOTE_REBUILD_FAILED)
            print(f"  [retention] feature rebuild failed for {len(bars)} pairs: {exc}")
            feats_tr = pd.DataFrame()

    if not feats_tr.empty:
        feats_tr["date"] = pd.to_datetime(feats_tr["date"]).dt.normalize()
        feats_tr = feats_tr[feats_tr["date"] == target_norm].copy()
        feats_tr = _attach_panel_features(feats_tr, scan_df, target_norm)
        unknown_kind = feats_tr["model_kind"].isna()
        for tk in feats_tr.loc[unknown_kind, "ticker"].astype(str).unique():
            note_per_ticker.setdefault(tk, NOTE_REGIME_UNKNOWN)
        feats_tr = feats_tr[~unknown_kind]

    score_lookup: dict[str, float] = {}
    if not feats_tr.empty:
        try:
            yhat = surrogate.predict(feats_tr)
        except ret_surrogate.MissingFeatureError as exc:
            for tk in feats_tr["ticker"].astype(str).unique():
                note_per_ticker.setdefault(tk, NOTE_REBUILD_FAILED)
            print(f"  [retention] surrogate rejected schema: {exc}")
        else:
            feats_tr["score_1700_surrogate"] = yhat.values
            for tk, sc in zip(feats_tr["ticker"].astype(str), yhat.values):
                score_lookup[tk] = float(sc)

    panel_full_scores = panel["winner_R_pred"].astype(float).values
    panel_tickers = panel["ticker"].astype(str).tolist()

    rows: list[dict] = []
    for tk, full_score in zip(panel_tickers, panel_full_scores):
        src = source_lookup.get(tk, pd.NA)
        if tk in score_lookup:
            tr_score = score_lookup[tk]
            n_above = int(np.sum(panel_full_scores > tr_score))
            rank = n_above + 1
            rows.append({
                "ticker": tk,
                "score_1700_surrogate": tr_score,
                "rank_1700_surrogate": rank,
                "retention_pass": rank <= rank_threshold,
                "timing_clean_note": NOTE_OK,
                "bars_source": src,
            })
        else:
            rows.append({
                "ticker": tk,
                "score_1700_surrogate": np.nan,
                "rank_1700_surrogate": pd.NA,
                "retention_pass": False,
                "timing_clean_note": note_per_ticker.get(tk, NOTE_BARS_MISSING),
                "bars_source": src,
            })
    ret_df = pd.DataFrame(rows)
    ret_df["rank_1700_surrogate"] = ret_df["rank_1700_surrogate"].astype("Int64")

    enriched = scan_df.copy()
    other_dates = enriched["date"].dt.normalize() != target_norm
    enriched.loc[other_dates, "retention_pass"] = False
    enriched.loc[other_dates, "score_1700_surrogate"] = np.nan
    enriched.loc[other_dates, "timing_clean_note"] = NOTE_STAGE_DISABLED
    if "rank_1700_surrogate" not in enriched.columns:
        enriched["rank_1700_surrogate"] = pd.array(
            [pd.NA] * len(enriched), dtype="Int64",
        )
    if "bars_source" not in enriched.columns:
        enriched["bars_source"] = pd.NA

    enriched = enriched.merge(
        ret_df, on="ticker", how="left", suffixes=("", "_ret"),
    )
    for col in NEW_COLS:
        if f"{col}_ret" in enriched.columns:
            mask = enriched["date"].dt.normalize() == target_norm
            enriched.loc[mask, col] = enriched.loc[mask, f"{col}_ret"]
            enriched = enriched.drop(columns=[f"{col}_ret"])
    enriched["retention_pass"] = enriched["retention_pass"].astype("boolean").fillna(False).astype(bool)
    enriched["timing_clean_note"] = enriched["timing_clean_note"].fillna(
        NOTE_STAGE_DISABLED
    )
    enriched["rank_1700_surrogate"] = enriched["rank_1700_surrogate"].astype("Int64")

    panel_view = enriched[enriched["date"].dt.normalize() == target_norm]
    n_total = len(panel_view)
    n_pass = int(panel_view["retention_pass"].sum())
    n_unscored = int(panel_view["score_1700_surrogate"].isna().sum())
    n_drop = n_total - n_pass
    note_counts = panel_view["timing_clean_note"].value_counts().to_dict()
    src_series = panel_view["bars_source"].astype("object")
    src_counts: dict[str, int] = {}
    for v, n in src_series.value_counts(dropna=False).items():
        key = "missing" if (v is pd.NA or pd.isna(v)) else str(v)
        src_counts[key] = int(n)

    return StageOutcome(
        enriched=enriched,
        n_total=n_total, n_pass=n_pass, n_drop=n_drop, n_unscored=n_unscored,
        notes=note_counts,
        source_breakdown=src_counts,
    )
