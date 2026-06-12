"""
NOX Advisor — sinyal yükleyiciler.

Her yükleyici status-etiketli blok döner:
  {"status": "OK|NO_DE_DAY|UNAVAILABLE|STALE", "validation": "...", ...}

Doğrulama etiketleri DÜRÜST: tüm birincil hatlar paper-track (tek-rejim 2025-26
boğa) — advisor bu etiketi her gerekçede tekrarlamak ZORUNDA.
"""
import os
import json
import datetime
from pathlib import Path

import pandas as pd

OUT_DIR = Path(os.environ.get("NOX_OUTPUT_DIR", "output"))

VALIDATION_DE = "paper-track (tek-rejim)"
VALIDATION_TAVAN = "paper-track (tek-rejim 2025-26 boğa)"
VALIDATION_CLUSTER3 = "paper-track (forward ledger)"
VALIDATION_CONTEXT = "context-only / keşifsel"

# DE watchlist'te AL-adayı olabilecek section'lar (öncelik sırasıyla)
DE_BUY_SECTIONS = ["EXECUTABLE", "SIZE_REDUCED"]


# ── Decision Engine v1 watchlist ──

def load_de_watchlist(asof, lookback_days=5):
    """Günün watchlist CSV'si; yoksa NO_DE_DAY advisory; o da yoksa geriye yürü (STALE)."""
    path = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.csv"
    if path.exists():
        return _parse_de_csv(path, asof, status="OK")

    no_de = OUT_DIR / f"decision_engine_v1_no_de_day_advisory_{asof}.json"
    if no_de.exists():
        try:
            advisory = json.loads(no_de.read_text(encoding="utf-8"))
        except Exception:
            advisory = {}
        return {"status": "NO_DE_DAY", "validation": VALIDATION_DE, "asof": asof,
                "buy_rows": [], "watch_rows": [], "held_rows": [],
                "note": advisory.get("reason", "NO_DE_DAY")}

    # lokal/dispatch yolunda: en yakın eski güne düş
    d = datetime.date.fromisoformat(asof)
    for i in range(1, lookback_days + 1):
        prev = (d - datetime.timedelta(days=i)).isoformat()
        p = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{prev}.csv"
        if p.exists():
            block = _parse_de_csv(p, prev, status="STALE")
            block["note"] = f"{asof} watchlist yok; {prev} kullanıldı (BAYAT)"
            return block

    return {"status": "UNAVAILABLE", "validation": VALIDATION_DE, "asof": asof,
            "buy_rows": [], "watch_rows": [], "held_rows": []}


def _parse_de_csv(path, asof, status="OK"):
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()

    def _row(r):
        return {
            "ticker": r["ticker"], "section": r["section"],
            "family": r.get("family", ""), "timeframe": str(r.get("timeframe", "")),
            "state": r.get("state", ""), "market_context": r.get("market_context", ""),
            "entry_ref": _f(r.get("entry_ref")), "stop_ref": _f(r.get("stop_ref")),
            "atr": _f(r.get("atr")), "risk_atr": _f(r.get("risk_atr")),
            "reason_codes": r.get("reason_codes", ""),
            "trident_tier1": str(r.get("trident_tier1_active", "")).lower() in ("true", "1"),
        }

    buy_rows, watch_rows = [], []
    # ticker başına TEK al-adayı satırı: section önceliği, sonra 1d > diğer timeframe
    for section in DE_BUY_SECTIONS:
        sub = df[df["section"] == section]
        for tkr, grp in sub.groupby("ticker"):
            if any(b["ticker"] == tkr for b in buy_rows):
                continue  # EXECUTABLE varken SIZE_REDUCED ekleme
            grp = grp.copy()
            grp["_tf_rank"] = (grp["timeframe"].astype(str) != "1d").astype(int)
            best = grp.sort_values(["_tf_rank", "risk_atr"]).iloc[0]
            row = _row(best)
            row["n_cells"] = int(len(grp))  # konfluens: kaç hücre (family) tetikledi
            row["families"] = ";".join(sorted(grp["family"].astype(str).unique()))
            buy_rows.append(row)

    for _, r in df[df["section"].astype(str).str.startswith("WAIT")].iterrows():
        watch_rows.append(_row(r))

    return {"status": status, "validation": VALIDATION_DE, "asof": asof,
            "buy_rows": buy_rows, "watch_rows": watch_rows,
            "sections": df["section"].value_counts().to_dict()}


# ── Tavan lock sistemi (sabah 11:00 pick'leri — akşam için POZİSYON BAĞLAMI) ──

TAVAN_V1_EXIT = "V1 exit: SL −10% / TP1 +4% yarı sat / %2 trail / H25"


def load_tavan_lock(asof):
    path = OUT_DIR / f"tavan_lock_scan_{asof}.json"
    if not path.exists():
        return {"status": "UNAVAILABLE", "validation": VALIDATION_TAVAN,
                "asof": asof, "picks": [], "exit_rules": TAVAN_V1_EXIT}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        picks = obj.get("picks", obj if isinstance(obj, list) else [])
        return {"status": "OK", "validation": VALIDATION_TAVAN, "asof": asof,
                "picks": picks, "exit_rules": TAVAN_V1_EXIT,
                "note": "Sabah 11:00 lock pick'leri — akşam AL adayı DEĞİL, pozisyon bağlamı"}
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_TAVAN,
                "asof": asof, "picks": [], "exit_rules": TAVAN_V1_EXIT, "note": str(e)}


# ── Cluster-3 arketip (forward paper ledger) ──

def load_cluster3(asof, open_window_days=30, stale_days=7):
    path = OUT_DIR / "ranking_lab_cluster3_paper_snapshots_v0.parquet"
    if not path.exists():
        return {"status": "UNAVAILABLE", "validation": VALIDATION_CLUSTER3,
                "asof": asof, "open_candidates": [], "realized_hit_rate": None}
    try:
        df = pd.read_parquet(path)
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
        d = datetime.date.fromisoformat(asof)

        filled = df[df["y20_50_realized"].notna()]
        hit = float(filled["y20_50_realized"].mean()) if len(filled) else None

        cutoff = d - datetime.timedelta(days=open_window_days)
        open_df = df[(df["y20_50_realized"].isna()) & (df["signal_date"] >= cutoff)]
        cands = [{"ticker": str(r["ticker"]).upper(),
                  "signal_date": r["signal_date"].isoformat(),
                  "entry_close": _f(r.get("entry_close"))}
                 for _, r in open_df.iterrows()]

        status = "OK"
        if len(df) and (d - df["signal_date"].max()).days > stale_days:
            status = "STALE"
        return {"status": status, "validation": VALIDATION_CLUSTER3, "asof": asof,
                "open_candidates": cands, "realized_hit_rate": hit,
                "n_realized": int(len(filled)),
                "last_signal_date": df["signal_date"].max().isoformat() if len(df) else None}
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_CLUSTER3,
                "asof": asof, "open_candidates": [], "realized_hit_rate": None,
                "note": str(e)}


def load_validated_signals(asof):
    return {
        "decision_engine": load_de_watchlist(asof),
        "tavan_lock": load_tavan_lock(asof),
        "cluster3": load_cluster3(asof),
    }


# ── Bağlam sinyalleri (context-only) ──

def load_context_signals(asof, tickers_of_interest=None):
    """Mevcut scanner_reader havuzu — yalnızca bağlam, AL gerekçesi olamaz."""
    try:
        from agent.scanner_reader import get_latest_signals, summarize_signals
        signals = get_latest_signals()
        summary = summarize_signals(signals)
        per_ticker = {}
        for tkr in (tickers_of_interest or []):
            hits = _ticker_memberships(signals, tkr)
            if hits:
                per_ticker[tkr] = hits
        return {"status": "OK", "validation": VALIDATION_CONTEXT,
                "summary": summary, "per_ticker": per_ticker}
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_CONTEXT,
                "summary": None, "per_ticker": {}, "note": str(e)}


def _ticker_memberships(signals, ticker):
    """Bir ticker hangi tarayıcı listelerinde geçiyor (kompakt)."""
    hits = []
    try:
        from agent.scanner_reader import get_signals_for_ticker
        for s in get_signals_for_ticker(signals, ticker) or []:
            scr = s.get("screener") or s.get("source") or "?"
            hits.append(scr)
    except Exception:
        pass
    return sorted(set(hits))


def load_macro():
    try:
        from agent.macro import fetch_macro_snapshot, assess_macro_regime
        snapshot = fetch_macro_snapshot()
        regime = assess_macro_regime(snapshot)
        return {"status": "OK", "validation": VALIDATION_CONTEXT, "regime": regime}
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_CONTEXT,
                "regime": None, "note": str(e)}


# ── Canlı fiyatlar ──

def fetch_prices(tickers):
    """yfinance toplu son kapanış. {ticker: float}; bulunamayan atlanır."""
    out = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
        symbols = {f"{t}.IS": t for t in tickers}
        data = yf.download(list(symbols), period="5d", interval="1d",
                           progress=False, group_by="ticker", threads=True)
        for sym, tkr in symbols.items():
            try:
                if len(symbols) == 1:
                    closes = data["Close"].dropna()
                else:
                    closes = data[sym]["Close"].dropna()
                if len(closes):
                    out[tkr] = float(closes.iloc[-1])
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ fiyat çekme hatası: {e}")
    return out


def _f(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
