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


# ── Sektör rotasyon monitörü (PAPER-FORWARD — canlı kapı KAPALI) ──

VALIDATION_ROTATION = ("paper-forward monitör (trade-sistemi era-fragile FAIL, canlı kapı "
                       "KAPALI; rotasyon overlay'i 8y stabil; BANK_IGNITION keşifsel)")


def load_sector_rotation(asof, stale_days=5):
    """tools/sector_rotation_monitor_v0.py log'undan durum + son olaylar.

    REJİM RENGİ: iki hücre durumu (PRIMARY confirm-ON / keşifsel confirm-OFF),
    fren bayrakları, son BANK_IGNITION. AL kapısı DEĞİL."""
    path = OUT_DIR / "sector_rotation_monitor_log.csv"
    if not path.exists():
        return {"status": "UNAVAILABLE", "validation": VALIDATION_ROTATION, "asof": asof}
    try:
        df = pd.read_csv(path)
        last = df.iloc[-1]
        bar = pd.to_datetime(last["bar_date"]).date()
        gap = (datetime.date.fromisoformat(asof) - bar).days
        status = "OK" if gap <= stale_days else "STALE"

        events = [str(e) for e in df["new_events"].dropna().tail(15) if str(e).strip()]
        ignition = next((e for e in reversed(events) if "BANK_IGNITION" in e), None)
        return {
            "status": status, "validation": VALIDATION_ROTATION, "asof": asof,
            "bar_date": bar.isoformat(), "xu100": _f(last.get("xu100")),
            "state_primary_confirm_on": str(last.get("state_on")),
            "brake_primary": bool(last.get("brake_on")),
            "state_explore_confirm_off": str(last.get("state_off")),
            "brake_explore": bool(last.get("brake_off")),
            "recent_events": events[-6:],
            "last_bank_ignition": ignition,
            "note": ("BANK_IGNITION sonrası tarihsel desen (keşifsel, pre-spec'siz): "
                     "banka bacağı söner, ~4-8 hafta endeks + küçük taraf (XHARZ/XTUMY) "
                     "lehte; sıçrayan bankayı kovalama"),
        }
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_ROTATION,
                "asof": asof, "note": str(e)}


# ── HW OB/OS çoklu-TF dönüş taraması (BETİMSEL — AL tarafı backtest'te REDDEDİLDİ) ──

VALIDATION_HW = ("betimsel çoklu-TF dönüş (hwo_mtf_v1 AL_OS backtest REDDEDİLDİ — "
                 "rastgeleden kötü, rank 0.43; SAT_OB çıkış-uyarısı izinli ama "
                 "doğrulanmamış) — edge YOK, yalnızca rejim-rengi/breadth")
HW_TFS = ["5h", "1d", "1w", "1mo"]


def load_hw_obos(asof, recency_days=15, sector_map_path=None):
    """HW OB/OS çoklu-TF dönüş olaylarını oku → ticker bazında TF haritası +
    sektör genişlik (breadth) toplaması. EDGE YOK: AL_OS selection-negative,
    SAT_OB doğrulanmamış çıkış-uyarısı. Yalnızca betimsel bağlam."""
    import glob

    # asof'a en yakın tarama tarihini bul (TF başına aynı tarih kullanılır)
    dates = set()
    for tf in HW_TFS:
        for p in glob.glob(str(OUT_DIR / f"hw_obos_{tf}_scan_*.csv")):
            d = p.rsplit("_", 1)[-1].replace(".csv", "")
            if d <= asof:
                dates.add(d)
    if not dates:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_HW, "asof": asof,
                "per_ticker": {}, "sector_breadth": {}}
    scan_date = max(dates)

    asof_d = datetime.date.fromisoformat(asof)
    cutoff = asof_d - datetime.timedelta(days=recency_days)
    per_ticker = {}
    for tf in HW_TFS:
        path = OUT_DIR / f"hw_obos_{tf}_scan_{scan_date}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["ts"] = pd.to_datetime(df["ts"]).dt.date
        # (ticker) başına bu TF'deki EN SON olay; recency penceresi içindeyse al
        for tkr, grp in df.groupby("ticker"):
            r = grp.sort_values("ts").iloc[-1]
            if r["ts"] < cutoff:
                continue
            slot = per_ticker.setdefault(str(tkr).upper(),
                                         {"AL_OS": [], "SAT_OB": [], "last_ts": None})
            kind = "AL_OS" if r["kind"] == "AL_OS" else "SAT_OB"
            slot[kind].append(tf)
            slot["last_ts"] = max(slot["last_ts"], r["ts"]) if slot["last_ts"] else r["ts"]

    for slot in per_ticker.values():
        if slot["last_ts"]:
            slot["last_ts"] = slot["last_ts"].isoformat()

    # sektör genişlik toplaması
    breadth = {}
    try:
        smp = Path(sector_map_path) if sector_map_path else (OUT_DIR / "bist_sector_map.csv")
        if smp.exists():
            sm = pd.read_csv(smp)
            sec = dict(zip(sm["ticker"].astype(str).str.upper(), sm["sektor_id"]))
            for tkr, slot in per_ticker.items():
                sid = sec.get(tkr)
                if sid is None:
                    continue
                b = breadth.setdefault(int(sid), {"AL_OS": [], "SAT_OB": []})
                if slot["AL_OS"]:
                    b["AL_OS"].append(tkr)
                if slot["SAT_OB"]:
                    b["SAT_OB"].append(tkr)
    except Exception:
        pass

    gap = (asof_d - datetime.date.fromisoformat(scan_date)).days
    return {
        "status": "OK" if gap <= 5 else "STALE",
        "validation": VALIDATION_HW, "asof": asof, "scan_date": scan_date,
        "per_ticker": per_ticker, "sector_breadth": breadth,
        "n_al_os": sum(1 for s in per_ticker.values() if s["AL_OS"]),
        "n_sat_ob": sum(1 for s in per_ticker.values() if s["SAT_OB"]),
    }


# ── Backtest-doğrulanmış seçim feature'ları (panel join, ranking tilt) ──

SELECTION_FEATURES = ["n_concurrent_sources", "n_concurrent_families",
                      "event_multiplicity", "price_vs_20d_high"]


def load_panel_features(asof, tickers):
    """ranking_lab_features_v0 panelinden asof günü seçim-feature'ları (ticker bazında).
    advisor_selection_backtest_v0 ağırlıklarıyla _quality tilt'i için. Yoksa boş."""
    path = OUT_DIR / "ranking_lab_features_v0.parquet"
    if not path.exists() or not tickers:
        return {}
    try:
        cols = ["ticker", "signal_date"] + SELECTION_FEATURES
        df = pd.read_parquet(path, columns=cols)
        df = df[df["signal_date"] == pd.Timestamp(asof)]
        if df.empty:  # asof paneldeyse boşsa en yakın önceki güne düş
            df2 = pd.read_parquet(path, columns=cols)
            df2 = df2[df2["signal_date"] <= pd.Timestamp(asof)]
            if df2.empty:
                return {}
            last = df2["signal_date"].max()
            df = df2[df2["signal_date"] == last]
        tset = {t.upper() for t in tickers}
        out = {}
        for _, r in df.iterrows():
            t = str(r["ticker"]).upper()
            if t in tset:
                # ticker başına en güçlü konfluens satırı
                cand = {f: _f(r.get(f)) for f in SELECTION_FEATURES}
                prev = out.get(t)
                if prev is None or (cand.get("event_multiplicity") or 0) > \
                        (prev.get("event_multiplicity") or 0):
                    out[t] = cand
        return out
    except Exception as e:
        print(f"⚠️ panel feature yükleme hatası: {e}")
        return {}


def load_selection_weights():
    """Backtest verdict PROCEED ise ağırlıklar; NO_EDGE/yoksa None (sezgisel fallback)."""
    path = OUT_DIR / "advisor_selection_backtest_v0_weights.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
        return obj.get("weights") if obj.get("verdict") == "PROCEED_WEIGHTS" else None
    except Exception:
        return None


def load_validated_signals(asof):
    return {
        "decision_engine": load_de_watchlist(asof),
        "tavan_lock": load_tavan_lock(asof),
        "cluster3": load_cluster3(asof),
        "sector_rotation": load_sector_rotation(asof),
        "hw_obos": load_hw_obos(asof),
    }


# ── Bağlam sinyalleri (context-only) ──

def load_context_signals(asof, tickers_of_interest=None):
    """Mevcut scanner_reader havuzu — yalnızca bağlam, AL gerekçesi olamaz.

    Lokal CSV yoksa (CI/bot ortamı) GH Pages latest_signals.json fallback'i
    (agent/tools._get_signals ile aynı desen; GH_PAGES_BASE_URL gerekir)."""
    try:
        from agent.scanner_reader import (get_latest_signals, summarize_signals,
                                          fetch_signals_from_url)
        signals, _ = get_latest_signals()
        if not signals:
            signals, _ = fetch_signals_from_url()
        if not signals:
            return {"status": "UNAVAILABLE", "validation": VALIDATION_CONTEXT,
                    "summary": None, "per_ticker": {},
                    "note": "lokal CSV yok + GH Pages fallback boş"}
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


# ── GitHub Actions artifact'ından DE watchlist çekme (bot /danis tam) ──

def fetch_latest_de_artifact(repo=None, dest_dir=None, max_artifacts=100):
    """Public repodaki en yeni de-v1-watchlist-*/no-de-day artifact'ını indir.

    Bot host'ta lokal output/ yok — artifact zip'i indirip dest_dir'e açar.
    (asof, kind) döner; kind ∈ {WATCHLIST, NO_DE_DAY}; bulunamazsa (None, None).
    """
    import io
    import re
    import zipfile
    import requests as rq

    repo = repo or os.environ.get("NOX_SIGNALS_REPO", "aalpkk/nox-project")
    dest = Path(dest_dir) if dest_dir else OUT_DIR
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return None, None
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    try:
        resp = rq.get(f"https://api.github.com/repos/{repo}/actions/artifacts",
                      params={"per_page": max_artifacts}, headers=headers, timeout=20)
        resp.raise_for_status()
        arts = resp.json().get("artifacts", [])
        best = None
        for a in arts:  # API en-yeni-önce döner
            if a.get("expired"):
                continue
            m = re.match(r"^de-v1-(watchlist|no-de-day)-(\d{4}-\d{2}-\d{2})$", a["name"])
            if not m:
                continue
            kind = "WATCHLIST" if m.group(1) == "watchlist" else "NO_DE_DAY"
            cand = (m.group(2), kind, a)
            # watchlist > no-de-day aynı gün; daha yeni tarih her zaman kazanır
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and kind == "WATCHLIST"):
                best = cand
        if best is None:
            return None, None
        asof, kind, art = best
        zresp = rq.get(art["archive_download_url"], headers=headers, timeout=60)
        zresp.raise_for_status()
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zresp.content)) as zf:
            zf.extractall(dest)
        return asof, kind
    except Exception as e:
        print(f"⚠️ DE artifact indirme hatası: {e}")
        return None, None
