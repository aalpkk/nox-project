"""SBT-1700 — live-style daily scan (E04_C01 paper/watchlist).

Frozen 5D ranker + frozen E04_C01 execution geometry. Same protocol as
``sbt1700.final_test_5d`` (TRAIN+VAL fit, seed=17, no tuning), applied to
candidates that triggered on a given scan date.

Output:
    output/sbt_1700_E04_scan_<YYYY-MM-DD>.html
    output/sbt_1700_E04_scan_latest.html

CLI:
    python -m sbt1700.scan_live --date 2026-04-24
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.reports import _NOX_CSS
from sbt1700 import splits as splits_mod
from sbt1700.eval_ranker_5d import (
    PRIMARY_LABEL,
    _fit_head,
    _select_features_5d,
)


DATASET = Path("output/sbt_1700_dataset_5d_intraday_v1.parquet")
EXISTING_TEST_PREDS = Path("output/sbt_1700_5d_ranker_preds_test.parquet")
PORTFOLIO_DECISION = Path("output/sbt_1700_E04_portfolio_decision.json")

OUT_DIR = Path("output")

SL_ATR = 2.0
COST_BPS_RT = 15
PORTFOLIO_RISK_PCT = 0.005
PORTFOLIO_NOTIONAL_CAP = 0.20
START_EQUITY = 1_000_000.0


# ── Ranker fit + score ──────────────────────────────────────────────

def _fit_ranker():
    train = splits_mod.load_split(DATASET, "train")
    val = splits_mod.load_split(DATASET, "validation")
    train_val = pd.concat([train, val], ignore_index=True)
    train_val = train_val.dropna(subset=[PRIMARY_LABEL]).reset_index(drop=True)
    feature_cols = _select_features_5d(train_val)
    head = _fit_head(train_val, PRIMARY_LABEL, feature_cols, is_classifier=False)
    return head, feature_cols, len(train_val)


def _verify_reproduces_test_preds(head, feature_cols) -> dict:
    test = splits_mod.load_split(DATASET, "test", allow_test=True)
    labelled = test.dropna(subset=[PRIMARY_LABEL]).reset_index(drop=True)
    fresh = pd.Series(head.booster.predict(labelled[feature_cols]),
                      index=labelled.index)
    if not EXISTING_TEST_PREDS.exists():
        return {"available": False}
    existing = pd.read_parquet(EXISTING_TEST_PREDS)
    merged = labelled[["ticker", "date"]].copy()
    merged["score_fresh"] = fresh.values
    merged = merged.merge(existing[["ticker", "date", "score_primary"]],
                          on=["ticker", "date"], how="inner")
    diff = (merged["score_fresh"] - merged["score_primary"]).abs()
    return {
        "available": True,
        "n_compared": int(len(merged)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bit_identical": bool(diff.max() < 1e-9),
    }


def _candidates_on(head, feature_cols, scan_date: pd.Timestamp) -> pd.DataFrame:
    """All SBT-1700 candidates triggered on scan_date (no NaN-label drop)."""
    df = pd.read_parquet(DATASET)
    cands = df[df["date"] == scan_date].copy()
    if cands.empty:
        return cands
    cands["score_primary"] = head.booster.predict(cands[feature_cols])
    return cands.reset_index(drop=True)


def _test_score_distribution() -> dict:
    df = pd.read_parquet(EXISTING_TEST_PREDS)
    s = df["score_primary"]
    return {
        "n_test": int(len(df)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q80": float(s.quantile(0.80)),
        "q90": float(s.quantile(0.90)),
        "rank25_threshold": float(s.nlargest(25).min()),
    }


def _e04_params(entry: float, atr: float) -> dict:
    return {
        "entry": entry,
        "atr14_prior": atr,
        "initial_stop": entry - SL_ATR * atr,
        "stop_distance_pct": (SL_ATR * atr) / entry,
        "plus_1R_level": entry + atr,
        "plus_2R_level": entry + 2.0 * atr,
        "round_trip_cost_pct": COST_BPS_RT / 10_000.0,
    }


def _portfolio_sizing(entry: float, atr: float, equity: float) -> dict:
    stop_dist = SL_ATR * atr
    risk_budget = equity * PORTFOLIO_RISK_PCT
    notional_cap = equity * PORTFOLIO_NOTIONAL_CAP
    raw_qty = risk_budget / stop_dist
    notional_raw = raw_qty * entry
    if notional_raw > notional_cap:
        qty = int(notional_cap / entry)
        binding = "notional_cap"
    else:
        qty = int(raw_qty)
        binding = "risk_budget"
    return {
        "equity_TRY": equity,
        "risk_pct": PORTFOLIO_RISK_PCT,
        "notional_cap_pct": PORTFOLIO_NOTIONAL_CAP,
        "stop_distance_per_share": stop_dist,
        "risk_budget_TRY": risk_budget,
        "qty_shares": qty,
        "notional_TRY": qty * entry,
        "actual_risk_TRY": qty * stop_dist,
        "binding_constraint": binding,
    }


def _load_portfolio_meta() -> dict | None:
    if not PORTFOLIO_DECISION.exists():
        return None
    pj = json.loads(PORTFOLIO_DECISION.read_text())
    out = {"verdict": pj.get("verdict"),
           "fails": pj.get("fails", [])}
    for pol in pj.get("policy_results", []):
        if pol.get("policy") == "P2_equal_risk_top3":
            for split, prefix in (("TEST", "test"), ("FULL", "full")):
                sp = pol["splits"].get(split, {})
                pm = sp.get("portfolio_metrics", {})
                tm = sp.get("trade_metrics", {})
                out[f"{prefix}_total_return"] = pm.get("total_return")
                out[f"{prefix}_max_dd"] = pm.get("max_drawdown")
                out[f"{prefix}_sharpe"] = pm.get("Sharpe")
                out[f"{prefix}_sortino"] = pm.get("Sortino")
                out[f"{prefix}_n_trades"] = tm.get("n_trades")
            break
    return out


# ── HTML rendering (briefing aesthetic) ─────────────────────────────

_LOCAL_CSS = """
.sbt-grid { display: grid; grid-template-columns: 1fr; gap: 1.1rem; margin-bottom: 1.5rem; }
@media (min-width: 880px) { .sbt-grid { grid-template-columns: 1.4fr 1fr; } }

.sbt-card {
  background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: none;
  border-radius: 18px;
  padding: 1rem 1.2rem;
  position: relative;
}
.sbt-card .card-title {
  font-family: var(--font-display);
  font-size: 0.7rem; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin-bottom: 0.6rem;
}
.sbt-card .card-title b { color: var(--nox-gold); font-weight: 700; }

.sbt-banner {
  background: rgba(158,90,90,0.10);
  border: 1px solid rgba(158,90,90,0.35);
  border-left: 3px solid var(--nox-red);
  border-radius: 10px;
  padding: 0.7rem 1rem;
  margin-bottom: 1.2rem;
  font-size: 0.82rem;
  color: var(--text-primary);
  line-height: 1.5;
}
.sbt-banner b { color: var(--nox-red); }
.sbt-banner.info {
  background: rgba(122,143,165,0.08);
  border-color: rgba(122,143,165,0.30);
  border-left-color: var(--nox-blue);
}
.sbt-banner.info b { color: var(--nox-blue); }
.sbt-banner.gold {
  background: rgba(201,169,110,0.07);
  border-color: rgba(201,169,110,0.30);
  border-left-color: var(--nox-gold);
}
.sbt-banner.gold b { color: var(--nox-gold); }

.sbt-cand {
  display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 0.65rem;
}
.sbt-cand .ticker {
  font-family: 'Inter', sans-serif;
  font-size: 1.6rem; font-weight: 900;
  color: #fff; letter-spacing: -0.01em;
}
.sbt-cand .pill {
  display: inline-flex; align-items: center;
  padding: 0.18rem 0.6rem;
  border-radius: 1rem;
  font-family: var(--font-mono);
  font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
}
.sbt-cand .pill.in-decile { background: rgba(201,169,110,0.18); color: var(--nox-gold); border: 1px solid rgba(201,169,110,0.45); }
.sbt-cand .pill.in-quintile { background: rgba(122,158,122,0.15); color: var(--nox-green); border: 1px solid rgba(122,158,122,0.4); }
.sbt-cand .pill.below { background: rgba(168,135,106,0.12); color: var(--nox-orange); border: 1px solid rgba(168,135,106,0.35); }

.sbt1700-badge {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.18rem 0.55rem;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.05em; text-transform: uppercase;
  border: 1px solid transparent;
}
.sbt1700-badge.d10 {
  background: rgba(201,169,110,0.20);
  color: var(--nox-gold);
  border-color: rgba(201,169,110,0.50);
  box-shadow: 0 0 0 1px rgba(201,169,110,0.10) inset;
}
.sbt1700-badge.q5 {
  background: rgba(168,135,106,0.14);
  color: var(--nox-copper, #a8876a);
  border-color: rgba(168,135,106,0.40);
}
.sbt-cand .score-chip {
  font-family: var(--font-mono); font-size: 0.78rem;
  padding: 0.2rem 0.55rem; border-radius: 6px;
  background: var(--nox-gold-dim); color: var(--nox-gold);
  font-weight: 700;
}
.sbt-cand .rank-chip {
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--text-secondary);
}

.sbt-kv { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem 1rem; }
.sbt-kv .row {
  display: flex; align-items: baseline; justify-content: space-between;
  border-bottom: 1px dashed var(--border-subtle);
  padding: 0.25rem 0;
}
.sbt-kv .row .k {
  font-family: var(--font-display);
  font-size: 0.74rem; color: var(--text-muted);
  letter-spacing: 0.02em;
}
.sbt-kv .row .v {
  font-family: var(--font-mono);
  font-size: 0.82rem; color: var(--text-primary);
  font-weight: 600;
}
.sbt-kv .row .v.gold { color: var(--nox-gold); }
.sbt-kv .row .v.red { color: var(--nox-red); }
.sbt-kv .row .v.green { color: var(--nox-green); }

table.sbt {
  width: 100%; border-collapse: collapse;
  font-size: 0.78rem;
}
table.sbt th, table.sbt td {
  padding: 7px 10px;
  border-bottom: 1px solid var(--border-subtle);
  text-align: left;
}
table.sbt th {
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-family: var(--font-display);
  font-size: 0.66rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
}
table.sbt td {
  font-family: var(--font-mono);
  color: var(--text-primary);
}
table.sbt td.ticker { font-family: 'Inter', sans-serif; font-weight: 800; color: #fff; }

.sbt-empty {
  text-align: center;
  font-family: var(--font-mono);
  color: var(--text-muted);
  padding: 1.4rem 0;
  font-size: 0.85rem;
}

.sbt-footer {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.6;
}
.sbt-footer code {
  background: var(--bg-elevated);
  padding: 1px 6px; border-radius: 3px;
  color: var(--text-secondary);
}

details.sbt-details {
  margin-bottom: 0.75rem;
}
details.sbt-details > summary {
  cursor: pointer;
  list-style: none;
  font-family: var(--font-display);
  color: var(--nox-gold);
  font-size: 0.85rem;
  font-weight: 600;
  padding: 0.55rem 0.85rem;
  background: rgba(199,189,190,0.06);
  backdrop-filter: blur(16px);
  border-radius: 14px;
  display: flex; align-items: center; gap: 0.4rem;
}
details.sbt-details > summary::-webkit-details-marker { display: none; }
details.sbt-details > summary::before {
  content: '▸';
  font-size: 0.78rem;
  color: var(--text-muted);
  transition: transform 0.2s;
}
details.sbt-details[open] > summary::before { transform: rotate(90deg); }
details.sbt-details > .body {
  padding: 0.8rem;
  background: rgba(6,7,9,0.55);
  border-radius: 0 0 14px 14px;
}

.sanity-ok { color: var(--nox-green); font-weight: 700; }
.sanity-fail { color: var(--nox-red); font-weight: 700; }
"""


def _fmt_pct(v):
    return f"{v*100:+.2f}%" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"


def _fmt_num(v, prec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{prec}f}"


def _kv_row(k, v, klass=""):
    klass_attr = f' class="v {klass}"' if klass else ' class="v"'
    return f'<div class="row"><span class="k">{html.escape(k)}</span><span{klass_attr}>{html.escape(str(v))}</span></div>'


def _candidate_card(cand: dict, score_dist: dict) -> str:
    score = cand["score_primary"]
    in_quintile = score >= score_dist["q80"]
    in_decile = score >= score_dist["q90"]
    if in_decile:
        pill_class, pill_text = "in-decile", "TOP-DECILE"
    elif in_quintile:
        pill_class, pill_text = "in-quintile", "TOP-QUINTILE"
    else:
        pill_class, pill_text = "below", "BELOW TOP-Q"

    if in_decile:
        sbt_badge_html = '<span class="sbt1700-badge d10" title="SBT-1700 / E04_C01 paper-pick · top-decile">🎯 SBT-1700 · D10</span>'
    elif in_quintile:
        sbt_badge_html = '<span class="sbt1700-badge q5" title="SBT-1700 / E04_C01 paper-pick · top-quintile">🎯 SBT-1700 · Q5</span>'
    else:
        sbt_badge_html = ""

    rank_in_test = (
        f"#{int(cand['rank_in_test'])} / {score_dist['n_test']}"
        if cand.get("rank_in_test") is not None else "—"
    )
    e04 = _e04_params(cand["entry"], cand["atr14_prior"])
    sizing = _portfolio_sizing(cand["entry"], cand["atr14_prior"], START_EQUITY)
    tkr = html.escape(cand["ticker"])
    return f"""
    <div class="sbt-card">
      <div class="card-title">Aday · <b>{tkr}</b></div>
      <div class="sbt-cand">
        <span class="ticker">{tkr}</span>
        <span class="pill {pill_class}">{pill_text}</span>
        {sbt_badge_html}
        <span class="score-chip">score {score:.4f}</span>
        <span class="rank-chip">cohort rank {rank_in_test}</span>
      </div>
      <div class="sbt-kv">
        {_kv_row('Entry (17:00 open)', _fmt_num(e04['entry'], 4), 'gold')}
        {_kv_row('ATR14 prior', _fmt_num(e04['atr14_prior'], 4))}
        {_kv_row('Initial stop (−2.0 ATR)', _fmt_num(e04['initial_stop'], 4), 'red')}
        {_kv_row('Stop distance', _fmt_pct(e04['stop_distance_pct']), 'red')}
        {_kv_row('+1R (trail arm)', _fmt_num(e04['plus_1R_level'], 4), 'green')}
        {_kv_row('+2R level', _fmt_num(e04['plus_2R_level'], 4), 'green')}
        {_kv_row('box_top', _fmt_num(cand['box_top'], 4))}
        {_kv_row('low_1700', _fmt_num(cand['low_1700'], 4))}
        {_kv_row('intraday cov.', _fmt_num(cand['intraday_coverage'], 2))}
        {_kv_row('Round-trip cost', _fmt_pct(e04['round_trip_cost_pct']))}
      </div>
      <div style="margin-top:0.8rem;padding-top:0.55rem;border-top:1px solid var(--border-subtle);">
        <div class="card-title" style="margin-bottom:0.4rem;">P2 sizing · 1M TRY ref · binding: <b>{html.escape(sizing['binding_constraint'])}</b></div>
        <div class="sbt-kv">
          {_kv_row('Risk budget', f"{sizing['risk_budget_TRY']:.0f} TRY")}
          {_kv_row('Quantity', f"{sizing['qty_shares']} pay")}
          {_kv_row('Notional', f"{sizing['notional_TRY']:.0f} TRY")}
          {_kv_row('Actual risk', f"{sizing['actual_risk_TRY']:.0f} TRY")}
        </div>
      </div>
    </div>
    """


def _build_html(scan_date: pd.Timestamp, cands: pd.DataFrame, score_dist: dict,
                sanity: dict, n_train: int, portfolio_meta: dict | None) -> str:
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n_cands = len(cands)
    cands_q = int((cands["score_primary"] >= score_dist["q80"]).sum()) if n_cands else 0
    cands_d = int((cands["score_primary"] >= score_dist["q90"]).sum()) if n_cands else 0

    # Verdict banner
    if n_cands == 0:
        verdict_html = (
            '<div class="sbt-banner info">'
            '<b>NO TRIGGER</b> · Scan günü için tetiklenen aday yok. '
            'Portföy paper takibi açısından NO-ENTRY.</div>'
        )
    elif cands_q > 0:
        names = ", ".join(
            f"{r['ticker']} ({r['score_primary']:.3f})"
            for _, r in cands[cands["score_primary"] >= score_dist["q80"]].iterrows()
        )
        verdict_html = (
            f'<div class="sbt-banner gold">'
            f'<b>TOP-QUINTILE PASS</b> · {cands_q} aday locked P2 cohort eşiğini '
            f'aştı: {html.escape(names)}. Paper portföye uygun.</div>'
        )
    else:
        verdict_html = (
            f'<div class="sbt-banner info">'
            f'<b>BELOW CUTOFF</b> · {n_cands} aday tetikledi ama hiçbiri TEST '
            f'top-quintile eşiğini (q80 = {score_dist["q80"]:.3f}) aşmadı. '
            f'Locked policy altında NO-ENTRY günü.</div>'
        )

    # Candidate cards
    rank_lookup = {}
    if EXISTING_TEST_PREDS.exists():
        all_test_scores = pd.read_parquet(EXISTING_TEST_PREDS)["score_primary"].values
        for _, r in cands.iterrows():
            rank_lookup[(r["ticker"], r["date"])] = int((all_test_scores >= r["score_primary"]).sum()) + 1

    if n_cands == 0:
        candidates_html = '<div class="sbt-card"><div class="sbt-empty">Aday yok</div></div>'
    else:
        cand_blocks = []
        for _, r in cands.sort_values("score_primary", ascending=False).iterrows():
            cand_blocks.append(_candidate_card({
                "ticker": r["ticker"],
                "score_primary": float(r["score_primary"]),
                "entry": float(r["close_1700"]),
                "atr14_prior": float(r["atr14_prior"]),
                "box_top": float(r["box_top"]),
                "low_1700": float(r["low_1700"]),
                "intraday_coverage": float(r["intraday_coverage"]),
                "rank_in_test": rank_lookup.get((r["ticker"], r["date"])),
            }, score_dist))
        candidates_html = "\n".join(cand_blocks)

    # Embedded JSON marker for downstream consumers (briefing parser).
    # tier: "D10" (top-decile) > "Q5" (top-quintile) > null (below cutoff).
    picks_json = {
        "schema_version": 1,
        "system": "SBT-1700/E04_C01",
        "scan_date": str(scan_date.date()),
        "cutoffs": {"q80": float(score_dist["q80"]), "q90": float(score_dist["q90"])},
        "picks": [],
    }
    if n_cands:
        for _, r in cands.sort_values("score_primary", ascending=False).iterrows():
            score = float(r["score_primary"])
            if score >= score_dist["q90"]:
                tier = "D10"
            elif score >= score_dist["q80"]:
                tier = "Q5"
            else:
                continue  # only emit picks (top-quintile and above)
            picks_json["picks"].append({
                "ticker": str(r["ticker"]),
                "score": round(score, 4),
                "tier": tier,
            })
    picks_marker_html = (
        '<script id="sbt1700-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    # Distribution card
    dist_rows = "\n".join([
        _kv_row("TEST n", score_dist["n_test"]),
        _kv_row("score mean", _fmt_num(score_dist["mean"], 4)),
        _kv_row("score std", _fmt_num(score_dist["std"], 4)),
        _kv_row("min", _fmt_num(score_dist["min"], 4)),
        _kv_row("max", _fmt_num(score_dist["max"], 4)),
        _kv_row("q80 (top-quintile cutoff)", _fmt_num(score_dist["q80"], 4), "gold"),
        _kv_row("q90 (top-decile cutoff)", _fmt_num(score_dist["q90"], 4), "gold"),
        _kv_row("rank-25 score", _fmt_num(score_dist["rank25_threshold"], 4)),
    ])

    # Portfolio reference
    pm_block = ""
    if portfolio_meta is not None:
        pm_rows = "\n".join([
            _kv_row("Verdict", portfolio_meta.get("verdict", "—")),
            _kv_row("P2 / TEST total return", _fmt_pct(portfolio_meta.get("test_total_return"))),
            _kv_row("P2 / TEST max DD", _fmt_pct(portfolio_meta.get("test_max_dd")), "red"),
            _kv_row("P2 / TEST Sharpe (small-N)", _fmt_num(portfolio_meta.get("test_sharpe"), 3)),
            _kv_row("P2 / TEST Sortino (small-N)", _fmt_num(portfolio_meta.get("test_sortino"), 3)),
            _kv_row("P2 / FULL total return (2.2y)", _fmt_pct(portfolio_meta.get("full_total_return"))),
            _kv_row("P2 / FULL max DD", _fmt_pct(portfolio_meta.get("full_max_dd")), "red"),
            _kv_row("P2 / FULL Sharpe", _fmt_num(portfolio_meta.get("full_sharpe"), 3), "gold"),
            _kv_row("P2 / FULL Sortino", _fmt_num(portfolio_meta.get("full_sortino"), 3)),
        ])
        pm_block = f"""
        <details class="sbt-details">
          <summary>P2 portföy backtest referansı (locked)</summary>
          <div class="body">
            <div class="sbt-kv">{pm_rows}</div>
            <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.6rem;line-height:1.5;">
              TEST n=25 → küçük örneklem; Sharpe/Sortino ~66 günden anneksiyona dayalı, kalitatif okuyun.
              FULL view ~2.2y non-OOS (TRAIN+VAL OOS skorları + TEST primary skorları), risk framing için tercih edilen referans.
            </div>
          </div>
        </details>
        """

    # Sanity card
    if not sanity.get("available"):
        sanity_html = '<span class="sanity-ok">— (no preds_test artifact)</span>'
    elif sanity.get("bit_identical"):
        sanity_html = (
            f'<span class="sanity-ok">PASS</span> — fresh fit reproduces existing TEST '
            f'preds bit-identically (n={sanity["n_compared"]}, max |Δ|={sanity["max_abs_diff"]:.2e}).'
        )
    else:
        sanity_html = (
            f'<span class="sanity-fail">DRIFT</span> — max |Δ| = {sanity["max_abs_diff"]:.2e} '
            f'over n={sanity["n_compared"]} TEST rows.'
        )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SBT-1700 Scan — {scan_date.date()} (E04_C01 paper)</title>
<style>
{_NOX_CSS}

.briefing-container {{
  position: relative; z-index: 1;
  max-width: 1080px; margin: 0 auto;
  padding: 0 1.5rem 2rem;
}}

{_LOCAL_CSS}
</style>
</head>
<body>
{picks_marker_html}

<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="briefing-container">

  <div class="nox-header" style="margin-bottom:1.2rem;">
    <div class="nox-logo">
      SBT-1700<span class="proj">  · E04_C01</span>
      <span class="mode">paper · live-style scan</span>
    </div>
    <div class="nox-meta">
      As-of close: <b>{scan_date.date()}</b><br>
      n_train (TRAIN+VAL) = <b>{n_train}</b><br>
      generated {now_str}
    </div>
  </div>

  <div class="nox-stats">
    <div class="nox-stat"><span class="dot" style="background:var(--nox-gold);"></span>
      <span>Tetikli</span><span class="cnt">{n_cands}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-green);"></span>
      <span>Top-quintile</span><span class="cnt">{cands_q}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-gold);"></span>
      <span>Top-decile</span><span class="cnt">{cands_d}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-red);"></span>
      <span>q80 cutoff</span><span class="cnt">{score_dist['q80']:.3f}</span></div>
  </div>

  <div class="sbt-banner">
    <b>RESEARCH WATCHLIST · NOT LIVE.</b>
    Frozen 5D ranker + frozen E04_C01. Paper/shadow portföy izleme amaçlıdır,
    canlı para yok. Forward path bu güne ait sinyallerde 5 işlem günü dolana
    kadar etiketlenmez. TEST cohort small-N (n=25 top-quintile, 6 rolling-20
    pencere, top-5 winners = 62% pozitif R) caveats geçerli.
  </div>

  {verdict_html}

  <div class="sbt-grid">
    <div>
      {candidates_html}
    </div>

    <div class="sbt-card">
      <div class="card-title">TEST cohort skor dağılımı <b>· locked</b></div>
      <div class="sbt-kv">
        {dist_rows}
      </div>
      <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.6rem;line-height:1.5;">
        Locked TEST cohort: 124 sinyal, 2026-01-05 → 2026-04-16 (etkili).
        Top-quintile = q80 üstü; top-decile = q90 üstü.
      </div>
    </div>
  </div>

  {pm_block}

  <details class="sbt-details">
    <summary>E04_C01 spec recap</summary>
    <div class="body">
      <ul style="font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary);
                 line-height:1.7; padding-left:1.2rem;">
        <li><b style="color:var(--nox-gold);">Entry</b>: T'nin 17:00–18:00 bar açılışı.</li>
        <li><b style="color:var(--nox-red);">Initial stop</b>: 2.0 × ATR14_prior.</li>
        <li><b style="color:var(--nox-green);">C01 trail</b>: MFE +1R'i geçtiği sonraki barda kurulur,
          stop = MFE − 0.5 ATR.</li>
        <li><b style="color:var(--nox-gold);">E04 overlay</b>: bar 8 (T+1 mid-session 17:00 close)
          MFE &lt; +1R ise close exit.</li>
        <li><b>Cost</b>: 15 bps round-trip, same-bar pessimism.</li>
        <li><b>Priority</b>: initial_stop → prior-armed giveback → no_follow_through_close → timeout.</li>
      </ul>
    </div>
  </details>

  <details class="sbt-details">
    <summary>Reproducibility sanity</summary>
    <div class="body" style="font-family:var(--font-mono);font-size:0.78rem;line-height:1.7;">
      {sanity_html}
      <div style="color:var(--text-muted);margin-top:0.4rem;">
        Drift varsa locked test readout ile bu scan arasında model değişmiş demektir;
        bu durumda raporun hiçbir sayısı geçerli değildir, kontrol edilmeden
        kullanılmaz.
      </div>
    </div>
  </details>

  <div class="sbt-footer">
    Driver: <code>sbt1700/scan_live.py</code><br>
    Locked TEST preds: <code>{EXISTING_TEST_PREDS}</code><br>
    Portföy decision: <code>{PORTFOLIO_DECISION}</code><br>
    Paper-only · NOT live · canlı para yok · forward path henüz etiketsiz.
  </div>

</div>

</body>
</html>
"""


# ── Driver ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="SBT-1700 / E04_C01 live-style scan.")
    ap.add_argument("--date", required=True, help="Scan date YYYY-MM-DD.")
    args = ap.parse_args()

    scan_date = pd.Timestamp(args.date)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    head, feature_cols, n_train = _fit_ranker()
    sanity = _verify_reproduces_test_preds(head, feature_cols)
    cands = _candidates_on(head, feature_cols, scan_date)
    score_dist = _test_score_distribution()
    portfolio_meta = _load_portfolio_meta()

    html_str = _build_html(
        scan_date=scan_date,
        cands=cands,
        score_dist=score_dist,
        sanity=sanity,
        n_train=n_train,
        portfolio_meta=portfolio_meta,
    )

    dated_path = OUT_DIR / f"sbt_1700_E04_scan_{scan_date.date()}.html"
    latest_path = OUT_DIR / "sbt_1700_E04_scan_latest.html"
    dated_path.write_text(html_str, encoding="utf-8")
    latest_path.write_text(html_str, encoding="utf-8")
    print(f"[scan_live] wrote {dated_path}")
    print(f"[scan_live] wrote {latest_path}")
    print(f"[scan_live] candidates on {scan_date.date()}: {len(cands)}")
    if len(cands):
        for _, r in cands.sort_values("score_primary", ascending=False).iterrows():
            tag = ("TOP-DECILE" if r["score_primary"] >= score_dist["q90"]
                   else "TOP-QUINTILE" if r["score_primary"] >= score_dist["q80"]
                   else "below")
            print(f"[scan_live]   {r['ticker']:6s} score={r['score_primary']:.4f} ({tag})")
    print(f"[scan_live] cutoffs: q80={score_dist['q80']:.4f}  q90={score_dist['q90']:.4f}")
    if sanity.get("available"):
        print(f"[scan_live] sanity bit_identical={sanity['bit_identical']} "
              f"max|Δ|={sanity['max_abs_diff']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
