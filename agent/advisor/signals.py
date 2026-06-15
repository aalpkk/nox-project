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


def _trident_sil_p33(default=-10.49):
    """Trident SIL 33-yüzdebirlik (derin eşiği) — probe parquet'inden, cache'li.
    backtest ile aynı kohort eşiği. probe yoksa son bilinen sabit."""
    if not hasattr(_trident_sil_p33, "_v"):
        try:
            p = pd.read_parquet(OUT_DIR / "trident_probe_mb_3y.parquet",
                                columns=["structural_invalidation_pct_below_B"])
            _trident_sil_p33._v = float(p["structural_invalidation_pct_below_B"].quantile(0.33))
        except Exception:
            _trident_sil_p33._v = default
    return _trident_sil_p33._v


def _trident_bos(asof):
    """G3 için: (ticker→en son ≤asof mb/bb_1d event BoS) + events BoS p67 eşiği.
    BoS canlı DE CSV'sinde YOK → mb_scanner_events parquet'inden join (güncel kaynak,
    eşik de aynı dağılımdan = iç-tutarlı). cache'li."""
    if not hasattr(_trident_bos, "_v"):
        try:
            frames = []
            for f in ("mb_scanner_events_mb_1d", "mb_scanner_events_bb_1d"):
                p = OUT_DIR / f"{f}.parquet"
                if p.exists():
                    frames.append(pd.read_parquet(
                        p, columns=["ticker", "event_bar_date", "bos_distance_atr_at_event"]))
            ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if ev.empty:
                _trident_bos._v = ({}, 0.573)
            else:
                ev["event_bar_date"] = pd.to_datetime(ev["event_bar_date"])
                ev["tkr"] = ev["ticker"].astype(str).str.upper()
                p67 = float(ev["bos_distance_atr_at_event"].quantile(0.67))
                ev = ev[ev["event_bar_date"] <= pd.Timestamp(asof)].copy()
                latest = ev.sort_values("event_bar_date").groupby("tkr").tail(1)
                bmap = dict(zip(latest["tkr"], latest["bos_distance_atr_at_event"]))
                _trident_bos._v = (bmap, p67)
        except Exception:
            _trident_bos._v = ({}, 0.573)
    return _trident_bos._v


def mb_birth_xtf(asof):
    """Çapraz-TF above_mb TAZE çakışması — DE CSV'den DEĞİL, mb_scanner_events
    parquet'lerinden (point-in-time, event_bar_date ≤ asof).

    HAFTALIK-LİDER çakışma (kullanıcı isteği): her TF'de mb above_mb_birth'ün
    **SON BARDA + TAZE** olması — son KAPANMIŞ haftalık barda mb_1w above + SON
    günlük/5h barda mb_1d/mb_5h above. "Taze" = ticker'ın o TF'deki EN SON olayı
    above_mb_birth (sonradan mit_touch ile bölgeye geri DÜŞMEMİŞ) VE o TF'nin SON
    barında. Eski 4-günlük pencere KULLANILMAZ — doğup geri düşen sinyali saymaz.
    DE watchlist haftalık STATE'i aday-satırı olarak emit etmez → DE CSV'de GÖRÜNMEZ;
    parquet'ten kurtarılır (cache'li).

    NOT: parquet son bar = master-data tarihi (örn. Cuma 06-12). asof Pazartesi
    olsa da hafta-sonu BIST kapalı; canlı gün-içi (yeni iş günü) barlar bu veride
    YOK → "son bar" = son master barı.

    Döner: {"weekly_bar": "YYYY-MM-DD",
            "per_ticker": {T: {"weekly_bar","tf":[...],"weekly_lead":bool}}}
    """
    if not hasattr(mb_birth_xtf, "_cache"):
        mb_birth_xtf._cache = {}
    if asof in mb_birth_xtf._cache:
        return mb_birth_xtf._cache[asof]

    asof_ts = pd.Timestamp(asof)

    def _fresh_above(tf, bar_cap):
        """(set(taze-above ticker), son_bar) — ticker'ın EN SON olayı bar_cap'a
        kadar above_mb_birth VE o TF'nin SON barında ise TAZE sayılır."""
        p = OUT_DIR / f"mb_scanner_events_mb_{tf}.parquet"
        if not p.exists():
            return set(), None
        df = pd.read_parquet(p, columns=["ticker", "event_type", "event_bar_date"])
        df["d"] = pd.to_datetime(df["event_bar_date"])
        df = df[df["d"] <= bar_cap].copy()
        if df.empty:
            return set(), None
        df["tkr"] = df["ticker"].astype(str).str.upper()
        last_bar = df["d"].max()
        # ticker başına EN SON olay; above_mb_birth VE son barda mı?
        latest = df.sort_values("d").groupby("tkr").tail(1)
        fresh = latest[(latest["d"] == last_bar) &
                       (latest["event_type"] == "above_mb_birth")]
        return set(fresh["tkr"]), last_bar

    # KAPANMIŞ haftalık bar (W-FRI): asof'un HAFTASI oluşuyor olabilir → provizyonel
    # Cuma'yı düş (Pzt-Per → önceki Cuma; Cuma/hafta-sonu → o hafta).
    wd = asof_ts.weekday()
    days_since_fri = (wd - 4) % 7
    last_closed_fri = (asof_ts - pd.Timedelta(days=days_since_fri)).normalize()

    try:
        w_set, w_bar = _fresh_above("1w", last_closed_fri)
        d_set, _ = _fresh_above("1d", asof_ts)
        h_set, _ = _fresh_above("5h", asof_ts)
        if not w_set or w_bar is None:
            out = {"weekly_bar": str(w_bar.date()) if w_bar is not None else None,
                   "per_ticker": {}}
        else:
            per = {}
            for t in w_set:                               # haftalık TAZE above şart
                tf = (["1d"] if t in d_set else []) + (["5h"] if t in h_set else [])
                per[t] = {"weekly_bar": str(pd.Timestamp(w_bar).date()),
                          "tf": tf, "weekly_lead": bool(tf)}  # + 1d|5h TAZE
            out = {"weekly_bar": str(pd.Timestamp(w_bar).date()), "per_ticker": per}
    except Exception as e:
        out = {"weekly_bar": None, "per_ticker": {}, "error": str(e)}

    mb_birth_xtf._cache[asof] = out
    return out


def _parse_de_csv(path, asof, status="OK"):
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    sil_p33 = _trident_sil_p33()
    bos_map, bos_p67 = _trident_bos(asof)

    def _row(r):
        # trident_geo = G1(D∈[20,30)) ∧ G2(SIL≤p33) ∧ G3(BoS≥p67) — G4 (XU100 rejim) YOK.
        # G1/G2 DE CSV'den; G3 BoS mb_scanner_events'ten join. Backtest (G1∧G2∧G3,
        # G4'süz) trident'i birincil ayrıştıran gösterdi; rejim-yukarı şartı (G4) yok.
        d = _f(r.get("trident_D_pct")); s = _f(r.get("trident_SIL_pct_below_B"))
        b = bos_map.get(r["ticker"])
        g1 = d is not None and 20.0 <= d < 30.0
        g2 = s is not None and s <= sil_p33
        g3 = b is not None and b >= bos_p67
        return {
            "ticker": r["ticker"], "section": r["section"],
            "family": r.get("family", ""), "timeframe": str(r.get("timeframe", "")),
            "state": r.get("state", ""), "market_context": r.get("market_context", ""),
            "entry_ref": _f(r.get("entry_ref")), "stop_ref": _f(r.get("stop_ref")),
            "atr": _f(r.get("atr")), "risk_atr": _f(r.get("risk_atr")),
            "reason_codes": r.get("reason_codes", ""),
            "trident_tier1": str(r.get("trident_tier1_active", "")).lower() in ("true", "1"),
            "trident_geo": bool(g1 and g2 and g3),  # tam G1∧G2∧G3 (G4'süz)
            "trident_gates": f"{'G1' if g1 else '-'}{'G2' if g2 else '-'}{'G3' if g3 else '-'}",
            "trident_D_pct": d, "trident_sil": s, "trident_bos": b,
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
            # ÇOKLU-TF birth çakışması: above_mb_birth hangi FARKLI timeframe'lerde
            # (haftalık birth + 5h/1d birth = iç-içe yapı hizalanması, kullanıcı isteği;
            # backtest n_concurrent_timeframes OOS-pozitif)
            fam = grp["family"].astype(str)
            birth = grp[fam.str.contains("above_mb_birth", na=False)]
            mtf = sorted(birth["timeframe"].astype(str).unique(),
                         key=lambda t: {"5h": 0, "1d": 1, "1w": 2, "1mo": 3}.get(t, 9))
            row["mtf_birth"] = mtf            # örn. ['5h','1d']
            row["n_tf_birth"] = len(mtf)      # çoklu-TF birth derinliği
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

        # ÖNE ÇIKAN SEKTÖRLER: her sektörün kendi-dibi taraması (sector_scan, fiyat-only,
        # backtest'siz BİLGİ). durum TETİK = dipten taze dönüş; dipten% = dipten kazanç.
        # _fetch_isy_single ~8 sektör endeksi çeker (hafif). Hata → sektörsüz devam.
        sectors, favorable = [], []
        try:
            from tools.sector_rotation_monitor_v0 import sector_scan
            sc = sector_scan()
            for kod, r in sc.iterrows():
                dipten = float(r.get("dipten")) if pd.notna(r.get("dipten")) else None
                durum = str(r.get("durum"))
                row = {"kod": kod, "durum": durum,
                       "dipten_pct": round(dipten * 100, 1) if dipten is not None else None,
                       "dd_pct": round(float(r["dd"]) * 100, 1) if pd.notna(r.get("dd")) else None}
                sectors.append(row)
                # favori = taze tetik (TETİK) VEYA dipten anlamlı dönüş (>+3%)
                if durum == "TETİK" or (dipten is not None and dipten > 0.03):
                    favorable.append(kod)
        except Exception:
            pass

        return {
            "status": status, "validation": VALIDATION_ROTATION, "asof": asof,
            "bar_date": bar.isoformat(), "xu100": _f(last.get("xu100")),
            "state_primary_confirm_on": str(last.get("state_on")),
            "brake_primary": bool(last.get("brake_on")),
            "state_explore_confirm_off": str(last.get("state_off")),
            "brake_explore": bool(last.get("brake_off")),
            "recent_events": events[-6:],
            "last_bank_ignition": ignition,
            "sectors": sectors,            # her sektör durumu (TETİK/ARMED/TREND + dipten%)
            "favorable_sectors": favorable,  # öne çıkan sektör endeksleri (TETİK/dipten>3%)
            "note": ("BANK_IGNITION sonrası tarihsel desen (keşifsel, pre-spec'siz): "
                     "banka bacağı söner, ~4-8 hafta endeks + küçük taraf (XHARZ/XTUMY) "
                     "lehte; sıçrayan bankayı kovalama"),
        }
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_ROTATION,
                "asof": asof, "note": str(e)}


def cross_signal_membership(asof, tickers, recent_days=15):
    """triangle_break + horizontal_base CROSS-tag — DE aday-family'sinde tr/hb YOK
    (adaylar yalnız mb/bb taşıyor), bu yüzden event parquet'lerinden okunur. DE üretir,
    CI'da de-v1-mb-scanner artifact'ı ile taşınır. Döner {ticker: [tag...]}.

    Yalnız BİLGİ/çakışma — bu sinyaller advisor AL-kapısı değil. recent_days içinde
    (son ~3 hafta) event'i olan ticker tag'lenir (triangle/hb yavaş kurulum sinyalleri).
    """
    asof_ts = pd.Timestamp(asof)
    lo = asof_ts - pd.Timedelta(days=recent_days)
    want = {str(t).upper() for t in (tickers or [])}
    out = {}
    if not want:
        return out
    # triangle_break (yön=long/up yanlısı)
    try:
        p = OUT_DIR / "decision_engine_lab_triangle_break_events_3y.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["ticker", "signal_date", "side"])
            df["d"] = pd.to_datetime(df["signal_date"])
            df = df[(df["d"] >= lo) & (df["d"] <= asof_ts)].copy()
            if "side" in df.columns:
                sd = df["side"].astype(str).str.lower()
                df = df[sd.isin(("long", "up", "bull", "buy", "")) | sd.str.contains("long|up|bull", na=False)]
            for t in {str(x).upper() for x in df["ticker"]} & want:
                out.setdefault(t, []).append("triangle_break")
    except Exception:
        pass
    # horizontal_base (breakout/above/bounce durumları)
    try:
        p = OUT_DIR / "horizontal_base_event_v1.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["ticker", "bar_date", "signal_state"])
            df["d"] = pd.to_datetime(df["bar_date"])
            df = df[(df["d"] >= lo) & (df["d"] <= asof_ts)].copy()
            if "signal_state" in df.columns:
                # 'trigger' = yatay-taban kırılımı (ana sinyal), 'retest_bounce' = retest;
                # 'extended' (kırılım üstü uzamış/eski) HARİÇ → taze kırılım/retest tag'le.
                ss = df["signal_state"].astype(str).str.lower()
                df = df[ss.str.contains("trigger|break|above|bounce|retest|birth", na=False)]
            for t in {str(x).upper() for x in df["ticker"]} & want:
                out.setdefault(t, []).append("horizontal_break")
    except Exception:
        pass
    return out


def load_sector_map():
    """ticker → sector_index eşlemesi (tools/sector_map.json, KAP kaynaklı, cache'li)."""
    if not hasattr(load_sector_map, "_v"):
        try:
            import json as _json
            p = Path("tools/sector_map.json")
            m = _json.loads(p.read_text(encoding="utf-8"))
            load_sector_map._v = {
                t.upper(): (info.get("sector_index"), info.get("sector"))
                for t, info in (m.get("tickers") or {}).items()}
        except Exception:
            load_sector_map._v = {}
    return load_sector_map._v


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


def load_xu100_rdp(asof, stale_days=4):
    """XU100 RDP rejim durumu (regime_labels_daily_rdp_v1.csv'den son etiket).

    RDP = pivot-bazlı low-TIM rejim dedektörü ([[regime_beta_overlay_v1_outcome]] C6).
    regime ∈ {long, flat, short}. long DIŞI (flat/short) = 'risk-kapalı/SAT' bilgisi.
    Canlı kapı DEĞİL — advisor'da BİLGİ/bağlam olarak gösterilir. DE Stage 3 taze
    üretir; CI'da artifact ile taşınır (commit'li dosya bayat olabilir → tarih kontrol).
    """
    path = OUT_DIR / "regime_labels_daily_rdp_v1.csv"
    if not path.exists():
        return {"status": "UNAVAILABLE", "regime": None, "date": None}
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(asof)].sort_values("date")
        if df.empty:
            return {"status": "UNAVAILABLE", "regime": None, "date": None}
        last = df.iloc[-1]
        d = datetime.date.fromisoformat(asof)
        bar_d = last["date"].date()
        stale = (d - bar_d).days > stale_days
        regime = str(last["regime"]).lower()
        return {"status": "STALE" if stale else "OK",
                "regime": regime,
                "is_sell": regime in ("flat", "short", "risk_off", "out"),
                "date": bar_d.isoformat(),
                "close": _f(last.get("close")),
                "window_id": str(last.get("window_id", "")),
                "validation": VALIDATION_CONTEXT}
    except Exception as e:
        return {"status": "UNAVAILABLE", "regime": None, "date": None, "note": str(e)}


def load_macro():
    try:
        from agent.macro import fetch_macro_snapshot, assess_macro_regime
        snapshot = fetch_macro_snapshot()
        regime = assess_macro_regime(snapshot)
        return {"status": "OK", "validation": VALIDATION_CONTEXT,
                "regime": regime, "snapshot": snapshot}
    except Exception as e:
        return {"status": "UNAVAILABLE", "validation": VALIDATION_CONTEXT,
                "regime": None, "snapshot": [], "note": str(e)}


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
