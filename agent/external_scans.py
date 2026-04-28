"""
External scan fetcher + meta-Markowitz for the briefing HTML.

Sections produced (each independently optional, fail-soft):
  1. Alpha Scan          — alpha_scan.html   (table parse)
  2. Nyxpansion Daily    — nyxexp_scan.html  (table parse)
  3. Screener Combo      — screener_combo_latest.html (table parse, may 404)
  4. SBT-1700 E04_C01    — sbt_1700_E04_scan.html (JSON marker)
  5. 4-list Markowitz    — combinatorial 4-stock max-Sharpe + risk-parity (ERC)
                           over the union of (1)-(4); fallback ticker pool when
                           the union is too thin.

All HTTP failures → empty dicts, so a missing scan never breaks the briefing.
"""
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

# ────────────────────────────── constants ──────────────────────────────

_TZ_TR = timezone(timedelta(hours=3))
_DEFAULT_BASE = "https://aalpkk.github.io/nox-signals"

OHLCV_PATH = Path("output/ohlcv_10y_fintables_master.parquet")

# Markowitz / risk-parity params
LOOKBACK_DAYS = 60
WEIGHT_MIN = 0.10
WEIGHT_MAX = 0.50
PORTFOLIO_SIZE = 4
RF_RATE = 0.0  # raw Sharpe, rf=0 (matches nyxexp scan convention)

# Cap pre-Markowitz universe so combinatorial enumeration stays cheap.
META_UNIVERSE_CAP = 20

# Fallback pool when the 4-scan union is too thin to enumerate (<4 tickers
# with full price history). Liquid BIST names that almost always have data.
_FALLBACK_POOL = [
    "AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN",
    "ASELS", "TUPRS", "SAHOL", "KCHOL", "EREGL",
    "BIMAS", "MGROS", "TCELL", "SISE", "FROTO",
]

# ────────────────────────────── small helpers ──────────────────────────────


def _base_url() -> str:
    return os.environ.get("GH_PAGES_BASE_URL", _DEFAULT_BASE).rstrip("/")


def _get(url: str, timeout: int = 15) -> str | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _parse_tables(html: str) -> list[dict]:
    """Return list of {headers: [...], rows: [[cell_text...]]} for each <table>."""
    out: list[dict] = []
    for tbl in re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL):
        headers = [_strip_tags(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", tbl, re.DOTALL)]
        rows: list[list[str]] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.DOTALL):
            cells = [_strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)]
            if cells:
                rows.append(cells)
        out.append({"headers": headers, "rows": rows})
    return out


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    s = s.replace("%", "").replace(",", ".").replace("+", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ────────────────────────────── 1. ALPHA SCAN ──────────────────────────────


def fetch_alpha_picks() -> dict:
    """alpha_scan.html → all candidates from the second (full) table.

    Schema observed: # | Hisse | Skor | ML1g | ML3g | ADX | CMF | RSI | Fiyat | Stop | Stop% | Trail
    Returns: {date, picks: [{ticker, score, ml1g, ml3g, price}, ...], url, source}
    """
    url = f"{_base_url()}/alpha_scan.html"
    html = _get(url)
    if not html:
        return {"picks": [], "url": url, "error": "fetch_failed"}

    # cutoff date — best-effort
    m_date = re.search(r"cutoff[=:\s]+(\d{4}-\d{2}-\d{2})", html, re.I)
    scan_date = m_date.group(1) if m_date else ""

    tables = _parse_tables(html)
    picks: list[dict] = []
    for tbl in tables:
        hdr = [h.lower() for h in tbl["headers"]]
        if "hisse" in hdr and "skor" in hdr and "#" in hdr:
            t_idx = hdr.index("hisse")
            s_idx = hdr.index("skor")
            ml1_idx = hdr.index("ml1g") if "ml1g" in hdr else None
            ml3_idx = hdr.index("ml3g") if "ml3g" in hdr else None
            pr_idx = hdr.index("fiyat") if "fiyat" in hdr else None
            for row in tbl["rows"]:
                if len(row) <= s_idx:
                    continue
                ticker = row[t_idx].strip()
                if not re.match(r"^[A-Z][A-Z0-9]{2,9}$", ticker):
                    continue
                picks.append({
                    "ticker": ticker,
                    "score": _to_float(row[s_idx]),
                    "ml1g": _to_float(row[ml1_idx]) if ml1_idx is not None and len(row) > ml1_idx else None,
                    "ml3g": _to_float(row[ml3_idx]) if ml3_idx is not None and len(row) > ml3_idx else None,
                    "price": _to_float(row[pr_idx]) if pr_idx is not None and len(row) > pr_idx else None,
                })
            break

    return {
        "picks": picks,
        "scan_date": scan_date,
        "url": url,
        "source": "alpha_scan.html",
        "error": None,
    }


# ───────────────────────────── 2. NYXPANSION DAILY ─────────────────────────


def fetch_nyxexp_picks() -> dict:
    """nyxexp_scan.html (18:05 EOD canonical) → PASS candidates only.

    Two ranked tables observed: PASS (table 1) + DROP (table 2). Schema:
      # | Ticker | winR | pct | winR_1700 | rank_1700 | exec_tag | risk_bucket
      | rscr | retention | Trade Plan | Not
    Returns: {picks: [{ticker, winR, retention, exec_tag, risk_bucket}, ...]}
    """
    url = f"{_base_url()}/nyxexp_scan.html"
    html = _get(url)
    if not html:
        return {"picks": [], "url": url, "error": "fetch_failed"}

    tables = _parse_tables(html)
    picks: list[dict] = []
    for tbl in tables:
        hdr = [h.lower() for h in tbl["headers"]]
        if "ticker" in hdr and "winr" in hdr and "retention" in hdr:
            t_idx = hdr.index("ticker")
            w_idx = hdr.index("winr")
            ret_idx = hdr.index("retention")
            exec_idx = hdr.index("exec_tag") if "exec_tag" in hdr else None
            rb_idx = hdr.index("risk_bucket") if "risk_bucket" in hdr else None
            for row in tbl["rows"]:
                if len(row) <= ret_idx:
                    continue
                ticker = row[t_idx].strip()
                retention = row[ret_idx].strip().upper()
                if retention != "PASS":
                    continue  # DROP rows excluded — only retention-clean candidates
                if not re.match(r"^[A-Z][A-Z0-9]{2,9}$", ticker):
                    continue
                picks.append({
                    "ticker": ticker,
                    "winR": _to_float(row[w_idx]),
                    "retention": retention,
                    "exec_tag": row[exec_idx] if exec_idx is not None and len(row) > exec_idx else "",
                    "risk_bucket": row[rb_idx] if rb_idx is not None and len(row) > rb_idx else "",
                })

    return {
        "picks": picks,
        "url": url,
        "source": "nyxexp_scan.html",
        "error": None,
    }


# ───────────────────────────── 3. SCREENER COMBO ──────────────────────────


def fetch_screener_combo_picks() -> dict:
    """screener_combo_latest.html → all 9-list picks (gate-tagged)."""
    url = f"{_base_url()}/screener_combo_latest.html"
    html = _get(url)
    if not html:
        return {"picks": [], "url": url, "error": "fetch_failed"}

    tables = _parse_tables(html)
    picks: list[dict] = []
    seen: set[str] = set()
    for tbl in tables:
        hdr = [h.lower() for h in tbl["headers"]]
        # Look for any table whose header includes a ticker-like column
        ticker_keys = ("ticker", "hisse", "symbol")
        score_keys = ("rank_score", "score", "rank")
        t_idx = next((hdr.index(k) for k in ticker_keys if k in hdr), None)
        s_idx = next((hdr.index(k) for k in score_keys if k in hdr), None)
        if t_idx is None:
            continue
        gate_idx = hdr.index("gate") if "gate" in hdr else None
        for row in tbl["rows"]:
            if len(row) <= t_idx:
                continue
            ticker = row[t_idx].strip()
            if not re.match(r"^[A-Z][A-Z0-9]{2,9}$", ticker):
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            picks.append({
                "ticker": ticker,
                "score": _to_float(row[s_idx]) if s_idx is not None and len(row) > s_idx else None,
                "gate": row[gate_idx] if gate_idx is not None and len(row) > gate_idx else "",
            })

    return {
        "picks": picks,
        "url": url,
        "source": "screener_combo_latest.html",
        "error": None,
    }


# ───────────────────────────── 4. SBT-1700 ────────────────────────────────


def fetch_sbt1700_picks() -> dict:
    """sbt_1700_E04_scan.html → JSON-embedded picks list.

    Returns: {scan_date, picks: [{ticker, tier, score}, ...]}
    """
    url = f"{_base_url()}/sbt_1700_E04_scan.html"
    html = _get(url)
    if not html:
        return {"picks": [], "scan_date": "", "url": url, "error": "fetch_failed"}

    m = re.search(
        r'<script\s+id="sbt1700-data"\s+type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    if not m:
        return {"picks": [], "scan_date": "", "url": url, "error": "no_marker"}
    try:
        payload = json.loads(m.group(1))
    except Exception:
        return {"picks": [], "scan_date": "", "url": url, "error": "json_parse"}

    picks = []
    for p in payload.get("picks", []):
        tkr = p.get("ticker")
        tier = p.get("tier")
        if tkr and tier:
            picks.append({
                "ticker": tkr,
                "tier": tier,
                "score": _to_float(str(p.get("score", 0))),
            })

    return {
        "picks": picks,
        "scan_date": payload.get("scan_date", ""),
        "url": url,
        "source": "sbt_1700_E04_scan.html",
        "error": None,
    }


# ───────────────────────── 5. META-MARKOWITZ ──────────────────────────────


def _ledoit_wolf(sample_cov: np.ndarray, n_obs: int) -> np.ndarray:
    p = sample_cov.shape[0]
    if p <= 1 or n_obs <= 1:
        return sample_cov
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)
    shrink = min(1.0, (p / n_obs) * 0.5)
    return (1.0 - shrink) * sample_cov + shrink * target


def _load_returns(tickers: list[str], as_of: pd.Timestamp,
                  lookback: int) -> tuple[pd.DataFrame, list[str]]:
    if not OHLCV_PATH.exists():
        return pd.DataFrame(), []
    oh = pd.read_parquet(OHLCV_PATH)
    if "Date" not in oh.columns:
        oh = oh.reset_index()
    oh["Date"] = pd.to_datetime(oh["Date"])

    closes: dict[str, np.ndarray] = {}
    for t in tickers:
        g = oh[(oh["ticker"] == t) & (oh["Date"] <= as_of)].sort_values("Date")
        if len(g) < lookback + 2:
            continue
        closes[t] = g["Close"].iloc[-(lookback + 1):].values

    if len(closes) < PORTFOLIO_SIZE:
        return pd.DataFrame(), []

    df = pd.DataFrame(closes)
    lr = np.log(df / df.shift(1)).dropna()
    if len(lr) < lookback - 5:
        return pd.DataFrame(), []
    return lr, list(df.columns)


def _opt_max_sharpe(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float] | None:
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-10:
            return 1e10
        return -(port_ret - RF_RATE) / port_vol

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(WEIGHT_MIN, WEIGHT_MAX)] * n
    w0 = np.ones(n) / n
    try:
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
        if res.success and np.isfinite(res.fun):
            return res.x, -float(res.fun)
    except Exception:
        return None
    return None


def _opt_risk_parity(cov: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Equal Risk Contribution under long-only + weight bounds."""
    n = cov.shape[0]

    def erc_obj(w):
        port_var = float(w @ cov @ w)
        if port_var < 1e-12:
            return 1e10
        rc = w * (cov @ w)              # risk contribution per asset
        target = port_var / n           # equal target
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(WEIGHT_MIN, WEIGHT_MAX)] * n
    w0 = np.ones(n) / n
    try:
        res = minimize(erc_obj, w0, method="SLSQP", bounds=bnds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        if res.success and np.isfinite(res.fun):
            return res.x, float(res.fun)
    except Exception:
        return None
    return None


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> dict:
    port_ret = float(w @ mu)
    port_risk = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - RF_RATE) / port_risk if port_risk > 1e-10 else 0.0
    return {
        "expected_return": round(port_ret * 100, 2),
        "expected_risk": round(port_risk * 100, 2),
        "sharpe": round(sharpe, 3),
    }


def _rank_universe(union_picks: list[dict], cap: int) -> list[str]:
    """Rank tickers by (scan-source count desc, normalised score desc).

    Each pick dict has: ticker, score (may be None), source.
    Multi-source picks (in 2+ scans) bubble up to the top.
    """
    by_ticker: dict[str, dict] = {}
    for p in union_picks:
        t = p["ticker"]
        if t not in by_ticker:
            by_ticker[t] = {"sources": set(), "scores": []}
        by_ticker[t]["sources"].add(p["source"])
        if p.get("score") is not None:
            by_ticker[t]["scores"].append(float(p["score"]))

    ranked = sorted(
        by_ticker.items(),
        key=lambda kv: (
            -len(kv[1]["sources"]),
            -(max(kv[1]["scores"]) if kv[1]["scores"] else 0.0),
        ),
    )
    return [t for t, _ in ranked[:cap]]


def compute_meta_markowitz(union_picks: list[dict],
                           as_of: pd.Timestamp | None = None) -> dict:
    """Combinatorial 4-stock max-Sharpe + risk-parity over the union universe.

    Args:
        union_picks: list of {ticker, score?, source} from all 4 scans
        as_of: cutoff date; defaults to today TR

    Returns dict:
      {
        universe: [...],           # capped pool sent to the optimizer
        n_universe_with_data: int, # how many of those had OHLCV history
        used_fallback: bool,
        max_sharpe: { tickers, weights, expected_return, expected_risk, sharpe },
        risk_parity: { tickers, weights, expected_return, expected_risk, sharpe,
                       risk_contributions },
        combos_evaluated: int,
        error: None | str,
      }
    """
    if as_of is None:
        as_of = pd.Timestamp(datetime.now(_TZ_TR).date())

    empty = {
        "universe": [], "n_universe_with_data": 0, "used_fallback": False,
        "max_sharpe": None, "risk_parity": None,
        "combos_evaluated": 0, "error": None,
    }

    if not union_picks:
        # No candidates at all → use fallback pool
        union_picks = [{"ticker": t, "score": None, "source": "fallback"}
                       for t in _FALLBACK_POOL]
        empty["used_fallback"] = True

    universe = _rank_universe(union_picks, META_UNIVERSE_CAP)
    if len(universe) < PORTFOLIO_SIZE:
        empty["error"] = "universe_too_thin"
        return empty

    lr, kept = _load_returns(universe, as_of, LOOKBACK_DAYS)
    if lr.empty:
        # Try fallback once if not already
        if not empty["used_fallback"]:
            fb_picks = [{"ticker": t, "score": None, "source": "fallback"}
                        for t in _FALLBACK_POOL]
            lr, kept = _load_returns([p["ticker"] for p in fb_picks],
                                     as_of, LOOKBACK_DAYS)
            empty["used_fallback"] = True
            universe = [p["ticker"] for p in fb_picks]
        if lr.empty:
            empty["error"] = "no_price_data"
            return empty

    n_obs = len(lr)
    mu_all = lr.mean().values * 252
    cov_all = _ledoit_wolf(lr.cov().values * 252, n_obs)

    n = len(kept)
    if n < PORTFOLIO_SIZE:
        empty["error"] = "data_too_thin"
        return empty

    best_ms = {"sharpe": -np.inf, "w": None, "idx": None}
    best_rp = {"sharpe": -np.inf, "w": None, "idx": None}
    combos_evaluated = 0

    for combo in combinations(range(n), PORTFOLIO_SIZE):
        combos_evaluated += 1
        idx = list(combo)
        sub_mu = mu_all[idx]
        sub_cov = cov_all[np.ix_(idx, idx)]

        ms = _opt_max_sharpe(sub_mu, sub_cov)
        if ms is not None and ms[1] > best_ms["sharpe"]:
            best_ms = {"sharpe": ms[1], "w": ms[0], "idx": idx}

        rp = _opt_risk_parity(sub_cov)
        if rp is not None:
            stats = _portfolio_stats(rp[0], sub_mu, sub_cov)
            if stats["sharpe"] > best_rp["sharpe"]:
                best_rp = {"sharpe": stats["sharpe"], "w": rp[0], "idx": idx,
                           "stats": stats}

    out = {
        "universe": kept,
        "n_universe_with_data": len(kept),
        "used_fallback": empty["used_fallback"],
        "combos_evaluated": combos_evaluated,
        "lookback_days": LOOKBACK_DAYS,
        "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
        "as_of": as_of.strftime("%Y-%m-%d"),
        "error": None,
    }

    # Max-Sharpe portfolio
    if best_ms["w"] is not None:
        idx = best_ms["idx"]
        sel = [kept[i] for i in idx]
        sub_mu = mu_all[idx]
        sub_cov = cov_all[np.ix_(idx, idx)]
        stats = _portfolio_stats(best_ms["w"], sub_mu, sub_cov)
        out["max_sharpe"] = {
            "tickers": sel,
            "weights": {t: round(float(w), 4) for t, w in zip(sel, best_ms["w"])},
            **stats,
        }
    else:
        out["max_sharpe"] = None

    # Risk-parity portfolio
    if best_rp["w"] is not None:
        idx = best_rp["idx"]
        sel = [kept[i] for i in idx]
        sub_mu = mu_all[idx]
        sub_cov = cov_all[np.ix_(idx, idx)]
        w = best_rp["w"]
        rc = w * (sub_cov @ w)
        port_var = float(w @ sub_cov @ w)
        rc_pct = (rc / port_var) if port_var > 1e-10 else np.zeros_like(rc)
        out["risk_parity"] = {
            "tickers": sel,
            "weights": {t: round(float(wi), 4) for t, wi in zip(sel, w)},
            "risk_contributions": {t: round(float(r) * 100, 1) for t, r in zip(sel, rc_pct)},
            **best_rp["stats"],
        }
    else:
        out["risk_parity"] = None

    return out


# ───────────────────────────── orchestrator ────────────────────────────


def gather_all() -> dict:
    """Fetch all 4 scans + compute meta-Markowitz. Safe to call from briefing.

    Returns: {alpha, nyxexp, screener_combo, sbt1700, meta, fetched_at}
    """
    alpha = fetch_alpha_picks()
    nyxexp = fetch_nyxexp_picks()
    screener = fetch_screener_combo_picks()
    sbt = fetch_sbt1700_picks()

    # Build union for meta-Markowitz (each pick tagged by source)
    union: list[dict] = []
    for p in alpha.get("picks", []):
        union.append({"ticker": p["ticker"], "score": p.get("score"), "source": "alpha"})
    for p in nyxexp.get("picks", []):
        union.append({"ticker": p["ticker"], "score": p.get("winR"), "source": "nyxexp"})
    for p in screener.get("picks", []):
        union.append({"ticker": p["ticker"], "score": p.get("score"), "source": "screener_combo"})
    for p in sbt.get("picks", []):
        union.append({"ticker": p["ticker"], "score": p.get("score"), "source": "sbt1700"})

    meta = compute_meta_markowitz(union)

    return {
        "alpha": alpha,
        "nyxexp": nyxexp,
        "screener_combo": screener,
        "sbt1700": sbt,
        "meta": meta,
        "union_size": len({p["ticker"] for p in union}),
        "fetched_at": datetime.now(_TZ_TR).strftime("%Y-%m-%d %H:%M"),
    }


if __name__ == "__main__":
    import pprint
    out = gather_all()
    summary = {
        "alpha": len(out["alpha"]["picks"]),
        "nyxexp": len(out["nyxexp"]["picks"]),
        "screener_combo": len(out["screener_combo"]["picks"]),
        "sbt1700": len(out["sbt1700"]["picks"]),
        "union_size": out["union_size"],
        "meta_universe": len(out["meta"]["universe"]),
        "meta_used_fallback": out["meta"]["used_fallback"],
        "meta_error": out["meta"]["error"],
    }
    pprint.pprint(summary)
    if out["meta"]["max_sharpe"]:
        print("\nmax_sharpe:")
        pprint.pprint(out["meta"]["max_sharpe"])
    if out["meta"]["risk_parity"]:
        print("\nrisk_parity:")
        pprint.pprint(out["meta"]["risk_parity"])
