"""
nyxmomentum_preview_today.py — month-end preview scan

Simulates "today is the monthly rebalance" without reseeding the tracker
and without rerunning the full step0→step5 pipeline. Reuses the already-
computed step2 features (which include the current partial-month anchor)
and the frozen v1.1 prev-basket as the dampener carryover.

What it does (for the latest rebalance_date in step2 features that is NOT
yet in step5_selection, i.e., the in-progress month):

  1. Loads step2 features + step1 labels + frozen V5/M0 selections.
  2. Trains a single ridge (M1) on ALL labeled history; predicts on today.
     M0 = rule-based handcrafted composite (no fit).
  3. Rank-blends M0+M1 percentiles for today to get ensemble scores.
  4. Smooths today's scores against the frozen 2026-03-31 smoothed column
     (α=0.5), preserving the exact v1.1 continuity.
  5. Applies hysteresis with the frozen 2026-03-31 V5 basket as prev
     portfolio (n_enter=20, n_exit=30).
  6. Emits preview CSVs + annotated HTML in output/nyxmomentum/live/,
     clearly tagged `_preview` so they NEVER collide with freeze-locked
     tracker artifacts.

This is a diagnostic scan, not a tracker update. tracker.csv is NOT
touched. No monthly_rebalance workflow lock is involved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nyxmomentum.baselines import handcrafted_composite_score
from nyxmomentum.features import FEATURE_COLUMNS
from nyxmomentum.models import cs_zscore_columns
from run_nyxmomentum_live_tracker import _render_picks_html

REPORTS = "output/nyxmomentum/reports"
LIVE    = "output/nyxmomentum/live"
VERSION = "v1.1.0-frozen+preview"

FEAT   = f"{REPORTS}/step2_features.parquet"
LABELS = f"{REPORTS}/step1_labels.parquet"
V5_SEL = f"{REPORTS}/step5_selection_V5_M0M1_ensemble_damp.parquet"
M0_SEL = f"{REPORTS}/step5_selection_V0_M0_baseline.parquet"

N_ENTER, N_EXIT = 20, 30
ALPHA = 0.5


def _compute_m1_predictions(features: pd.DataFrame, labels: pd.DataFrame,
                              predict_date: pd.Timestamp) -> pd.DataFrame:
    """Ridge on all labeled history, predict on predict_date."""
    feat_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    panel = features.merge(
        labels[["ticker", "rebalance_date", "l2_excess_vs_universe_median"]],
        on=["ticker", "rebalance_date"], how="left",
    )
    z = cs_zscore_columns(panel, feat_cols)
    z[feat_cols] = z[feat_cols].fillna(0.0)

    train = z.loc[
        z["eligible"].astype(bool)
        & z["l2_excess_vs_universe_median"].notna()
        & (z["rebalance_date"] < predict_date)
    ]
    X_tr = train[feat_cols].values
    y_tr = train["l2_excess_vs_universe_median"].values.astype(float)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_tr, y_tr)

    te = z.loc[z["rebalance_date"] == predict_date].copy()
    te["prediction"] = model.predict(te[feat_cols].values)
    return te[["ticker", "rebalance_date", "eligible", "prediction"]]


def _rank_blend_single_date(m0: pd.Series, m1: pd.Series) -> pd.Series:
    ra = m0.rank(pct=True, method="average")
    rb = m1.rank(pct=True, method="average")
    return 0.5 * ra + 0.5 * rb


def _smooth_with_frozen_prev(today: pd.DataFrame, prev_smoothed: pd.Series) -> pd.Series:
    """smoothed_t = α·raw_t + (1-α)·smoothed_{t-1}. If ticker absent from
    prev, smoothed = raw (first observation semantics in baselines.py)."""
    raw = today.set_index("ticker")["prediction"]
    prev = prev_smoothed.reindex(raw.index)
    smoothed = np.where(prev.isna(), raw.values,
                         ALPHA * raw.values + (1 - ALPHA) * prev.values)
    return pd.Series(smoothed, index=raw.index, name="prediction_smoothed")


def _label_status(rank: int, selected: bool, prev: bool,
                   n_enter: int = 20) -> str:
    if selected and prev:
        return "core_keeper" if rank <= n_enter else "sticky_keep"
    if selected and not prev:
        return "new_entrant"
    if (not selected) and rank <= n_enter and not prev:
        return "crowded_out"
    if prev and not selected:
        return "dropped"
    return "bystander"


def _apply_hysteresis_single(scored: pd.DataFrame, prev_basket: set,
                              n_enter: int, n_exit: int) -> pd.DataFrame:
    """scored: columns [ticker, score, eligible]. Returns ranked table +
    selected flag + status, top-30 rows."""
    df = scored.loc[scored["eligible"].astype(bool)].copy()
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)

    keepers = set(df.loc[(df["ticker"].isin(prev_basket))
                          & (df["rank"] <= n_exit), "ticker"])
    if len(keepers) > n_enter:
        keepers = set(df.loc[df["ticker"].isin(keepers)]
                        .sort_values("rank").head(n_enter)["ticker"])
    slots_left = n_enter - len(keepers)
    if slots_left > 0:
        fresh = df.loc[(df["rank"] <= n_enter)
                        & (~df["ticker"].isin(keepers))].head(slots_left)
        new_basket = keepers | set(fresh["ticker"])
    else:
        new_basket = keepers

    top = df.loc[df["rank"] <= 30, ["ticker", "rank", "score"]].copy()
    top["selected"] = top["ticker"].isin(new_basket)
    top["prev_selected"] = top["ticker"].isin(prev_basket)
    top["status"] = [_label_status(int(r["rank"]), bool(r["selected"]),
                                      bool(r["prev_selected"]), n_enter)
                       for _, r in top.iterrows()]
    return top.reset_index(drop=True)


def _build_picks(el_top: pd.DataFrame) -> pd.DataFrame:
    picks = el_top.loc[el_top["selected"]].copy()
    picks = picks.sort_values("rank")[["ticker", "rank", "score"]]
    picks["weight"] = 1.0 / len(picks)
    return picks.reset_index(drop=True)


def _fetch_ice_sms(tickers: list[str], cache_path: str = "output/matriks_cache.json"
                   ) -> tuple[dict, dict]:
    """Matriks'ten kurumsal akış + takas çek, ICE + SMS hesapla.

    Reuses the existing matriks_cache.json (delta-fetches only missing
    tickers / dates). Returns ({ticker: ICEResult}, {ticker: SMSResult}).
    On any error returns ({}, {}) — overlay missing cells fall back to '—'.
    """
    if not tickers:
        return {}, {}

    api_key = os.environ.get("MATRIKS_API_KEY")
    if not api_key:
        print("  ⚠️ MATRIKS_API_KEY yok — ICE/SMS atlanıyor")
        return {}, {}

    try:
        from agent.matriks_client import MatriksClient
        from agent.matriks_adapter import process_matriks_batch
        from agent.institutional import calc_batch_ice
        from agent.smart_money import calc_batch_sms
    except Exception as e:
        print(f"  ⚠️ Matriks modülleri yüklenemedi: {e}")
        return {}, {}

    # Cache load (reuses briefing's format).
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    daily_store = cache.get("_daily", {}) if cache.get("_today") == today_str else {}
    history_store = cache.get("_history", {})
    trend_store = cache.get("_trend")

    missing = [t for t in tickers if t not in daily_store]
    print(f"\n⬡ Matriks: {len(tickers)} ticker — "
          f"cache hit {len(tickers) - len(missing)} / fetch {len(missing)}")

    if missing:
        try:
            client = MatriksClient(api_key=api_key)
            new_data = client.fetch_batch(missing, include_history=False,
                                            history_days=0,
                                            history_cache=history_store)
            if new_data:
                nt = new_data.pop("_trend", None)
                if nt:
                    trend_store = nt
                for t, d in new_data.items():
                    if t.startswith("_"):
                        continue
                    daily_store[t] = {k: d[k] for k in
                                        ("flows", "settlement", "price", "investor")
                                        if k in d}
                    if "daily_flows" in d:
                        history_store.setdefault(t, {}).update(d["daily_flows"])
            with open(cache_path, "w") as f:
                json.dump({"_today": today_str, "_daily": daily_store,
                            "_history": history_store, "_trend": trend_store}, f,
                            ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️ Matriks fetch hatası: {e}")
            return {}, {}

    matriks_raw = {}
    for t in tickers:
        d = dict(daily_store.get(t, {}))
        if t in history_store:
            d["daily_flows"] = dict(history_store[t])
        if d:
            matriks_raw[t] = d
    if trend_store:
        matriks_raw["_trend"] = trend_store

    try:
        takas_map, cost_map, takas_hist, mkk_map = process_matriks_batch(matriks_raw)
        sms_map = calc_batch_sms(tickers, takas_map, mkk_map) if takas_map else {}
        ice_map = calc_batch_ice(tickers, takas_hist, takas_map, mkk_map,
                                   cost_data_map=cost_map) or {}
    except Exception as e:
        print(f"  ⚠️ ICE/SMS işleme hatası: {e}")
        return {}, {}

    n_ice = sum(1 for r in ice_map.values() if r)
    n_sms = sum(1 for r in sms_map.values() if r)
    print(f"  ICE: {n_ice}/{len(tickers)} · SMS: {n_sms}/{len(tickers)}")
    return ice_map, sms_map


def _overlay_scope(v5_picks: pd.DataFrame, v5_eligible: pd.DataFrame,
                    m0_picks: pd.DataFrame | None, top: int | None,
                    include_m0: bool) -> list[str]:
    """Ticker set for Matriks overlay: V5 picks (optionally capped by --top)
    + crowded_out + sticky_keep  [+ M0 net-new if --include-m0].
    Deduped, order-preserving."""
    v5 = v5_picks.sort_values("rank")["ticker"].tolist()
    if top is not None and top > 0:
        v5 = v5[:top]
    crowded = v5_eligible.loc[v5_eligible["status"] == "crowded_out",
                                "ticker"].tolist()
    sticky = v5_eligible.loc[v5_eligible["status"] == "sticky_keep",
                              "ticker"].tolist()
    scope = list(dict.fromkeys(v5 + crowded + sticky))
    if include_m0 and m0_picks is not None:
        for t in m0_picks.sort_values("rank")["ticker"].tolist():
            if t not in scope:
                scope.append(t)
    return scope


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--top", type=int, default=None,
                   help="Limit V5 scope to top N picks for ICE/SMS overlay "
                        "(default: all 20). Crowded-out + sticky still included.")
    p.add_argument("--include-m0", action="store_true",
                   help="Also fetch ICE/SMS for M0 picks not already in V5 scope.")
    p.add_argument("--no-ice-sms", action="store_true",
                   help="Skip Matriks fetch entirely (pure selection preview).")
    args = p.parse_args()

    os.makedirs(LIVE, exist_ok=True)

    feat = pd.read_parquet(FEAT)
    labels = pd.read_parquet(LABELS)
    v5_sel = pd.read_parquet(V5_SEL)
    m0_sel = pd.read_parquet(M0_SEL)

    freeze_end = v5_sel["rebalance_date"].max()
    today_anchor = feat["rebalance_date"].max()
    if today_anchor <= freeze_end:
        print(f"Latest step2 anchor {today_anchor.date()} is not past frozen "
              f"end {freeze_end.date()}. Nothing to preview — exiting.")
        sys.exit(0)

    print(f"Frozen end:      {freeze_end.date()}")
    print(f"Preview anchor:  {today_anchor.date()}  (as-if month-close)")
    print()

    today_feat = feat.loc[feat["rebalance_date"] == today_anchor].copy()
    n_eligible = int(today_feat["eligible"].astype(bool).sum())
    print(f"  universe on preview date: {len(today_feat)} rows "
          f"({n_eligible} eligible)")

    # ── M0 + M1 predictions for today ────────────────────────────────────────
    print("  computing M0 (handcrafted composite)  …")
    m0_today = today_feat[["ticker", "rebalance_date", "eligible"]].copy()
    m0_today["prediction"] = handcrafted_composite_score(today_feat).values

    print("  training M1 ridge on all labeled history  …")
    m1_today = _compute_m1_predictions(feat, labels, today_anchor)

    # ── M0 strict top-20 (benchmark) ─────────────────────────────────────────
    frozen_m0_prev = set(
        m0_sel.loc[(m0_sel["rebalance_date"] == freeze_end)
                   & m0_sel["selected"], "ticker"]
    )
    m0_scored = m0_today.rename(columns={"prediction": "score"})[
        ["ticker", "score", "eligible"]
    ]
    m0_eligible = _apply_hysteresis_single(
        m0_scored, prev_basket=frozen_m0_prev, n_enter=20, n_exit=20,
    )
    m0_picks = _build_picks(m0_eligible)

    # ── V5 ensemble + α-smooth (vs frozen prev) + hysteresis 20/30 ──────────
    print("  rank-blending M0+M1, smoothing against frozen 2026-03-31  …")
    merged = m0_today[["ticker", "eligible"]].merge(
        m0_today[["ticker", "prediction"]].rename(columns={"prediction": "m0"}),
        on="ticker"
    ).merge(
        m1_today[["ticker", "prediction"]].rename(columns={"prediction": "m1"}),
        on="ticker",
    )
    merged["raw_ensemble"] = _rank_blend_single_date(merged["m0"], merged["m1"])

    # Prev smoothed series from frozen V5 selection at freeze_end.
    prev_smoothed = (v5_sel.loc[v5_sel["rebalance_date"] == freeze_end]
                             .set_index("ticker")["prediction_smoothed"])
    merged["score"] = _smooth_with_frozen_prev(
        merged.rename(columns={"raw_ensemble": "prediction"}), prev_smoothed,
    ).values

    frozen_v5_prev = set(
        v5_sel.loc[(v5_sel["rebalance_date"] == freeze_end)
                   & v5_sel["selected"], "ticker"]
    )
    v5_eligible = _apply_hysteresis_single(
        merged[["ticker", "score", "eligible"]],
        prev_basket=frozen_v5_prev, n_enter=N_ENTER, n_exit=N_EXIT,
    )
    v5_picks = _build_picks(v5_eligible)

    # ── Write preview outputs ────────────────────────────────────────────────
    iso = today_anchor.date().isoformat()
    tag = "preview"

    v5_csv  = os.path.join(LIVE, f"{iso}_v5_{tag}_picks.csv")
    v5_el   = os.path.join(LIVE, f"{iso}_v5_{tag}_eligible.csv")
    v5_html = os.path.join(LIVE, f"{iso}_v5_{tag}_picks.html")
    m0_csv  = os.path.join(LIVE, f"{iso}_m0_{tag}_picks.csv")
    m0_el   = os.path.join(LIVE, f"{iso}_m0_{tag}_eligible.csv")
    m0_html = os.path.join(LIVE, f"{iso}_m0_{tag}_picks.html")

    v5_picks.to_csv(v5_csv, index=False)
    v5_eligible.to_csv(v5_el, index=False)
    m0_picks.to_csv(m0_csv, index=False)
    m0_eligible.to_csv(m0_el, index=False)

    # ── ICE + SMS overlay (display-only context, selection untouched) ────────
    ice_map, sms_map = {}, {}
    if not args.no_ice_sms:
        scope = _overlay_scope(v5_picks, v5_eligible, m0_picks,
                                 top=args.top, include_m0=args.include_m0)
        print(f"  overlay scope: {len(scope)} ticker "
              f"(V5 picks{'+crowded+sticky' if not args.top else f' top-{args.top}+crowded+sticky'}"
              f"{' + M0 net-new' if args.include_m0 else ''})")
        ice_map, sms_map = _fetch_ice_sms(scope)

    with open(v5_html, "w") as fh:
        fh.write(_render_picks_html(iso, "V5", v5_picks, VERSION,
                                      eligible_df=v5_eligible,
                                      ice_map=ice_map or None,
                                      sms_map=sms_map or None))
    with open(m0_html, "w") as fh:
        # M0 HTML only gets overlay if --include-m0 pulled those tickers in.
        m0_ice = {t: ice_map[t] for t in m0_picks["ticker"] if t in ice_map}
        m0_sms = {t: sms_map[t] for t in m0_picks["ticker"] if t in sms_map}
        fh.write(_render_picks_html(iso, "M0", m0_picks, VERSION,
                                      eligible_df=m0_eligible,
                                      ice_map=m0_ice or None,
                                      sms_map=m0_sms or None))

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(f"══ nyxmomentum preview · {iso} (as-if month-close) ══")
    print(f"  prev V5 basket: {len(frozen_v5_prev)} names @ {freeze_end.date()}")
    crowded = v5_eligible.loc[v5_eligible["status"] == "crowded_out"]
    sticky  = v5_eligible.loc[v5_eligible["status"] == "sticky_keep"]
    new     = v5_eligible.loc[v5_eligible["status"] == "new_entrant"]
    keepers = v5_eligible.loc[v5_eligible["status"].isin(
        ["core_keeper", "sticky_keep"]
    )]
    print()
    print(f"  V5 basket:  20 names "
          f"({len(keepers)} keepers, {len(new)} new entrants)")
    print(f"  Crowded-out:  {len(crowded)}  ·  sticky-below-top-20: {len(sticky)}")
    print()
    print("  V5 top-20 preview:")
    for _, r in v5_picks.iterrows():
        status = v5_eligible.loc[v5_eligible["ticker"] == r["ticker"], "status"].iloc[0]
        flag = " " if status == "core_keeper" else \
               "*" if status == "new_entrant" else \
               "s" if status == "sticky_keep" else "?"
        print(f"    {int(r['rank']):>2}. {flag} {r['ticker']:<6}  "
              f"score={float(r['score']):+.4f}")
    if len(crowded):
        print()
        print("  Crowded-out new candidates (would have been top-20 new entrants):")
        for _, r in crowded.sort_values("rank").iterrows():
            print(f"    {int(r['rank']):>2}.   {r['ticker']:<6}  "
                  f"score={float(r['score']):+.4f}")
    if len(sticky):
        print()
        print("  Sticky-below-top-20 (carried over at rank>20, ≤30):")
        for _, r in sticky.sort_values("rank").iterrows():
            print(f"    {int(r['rank']):>2}. s {r['ticker']:<6}  "
                  f"score={float(r['score']):+.4f}")
    print()
    print(f"  M0 benchmark basket: 20 names (strict top-20 from M0 composite)")
    print()
    print("  outputs (preview — tracker.csv NOT touched):")
    for p in (v5_csv, v5_el, v5_html, m0_csv, m0_el, m0_html):
        print(f"    {p}")


if __name__ == "__main__":
    main()
