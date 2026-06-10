"""
NOX — TEFAS Fon Akis Izleyici (Fund Flow Monitor)
=================================================

ChatGPT'deki "Izle TLY fon akisini" otomasyonunun repo-ici / GitHub Actions
versiyonu. TLY/PBR/PHE/DFI (ve eklenen) TEFAS fonlari icin asagidaki kosullari
kontrol eder, tetiklenirse Telegram'a uyari yollar:

  * Akis yavaslama      — net giris pozitif ama onceki pencereye gore yavasliyor
  * Akis negatife donme — net giris pozitiften negatife dondu
  * Yatirimci dususu    — yatirimci sayisi N gun once gore azaldi
  * Fon buyuklugu zayifl.— buyukluk pencere icinde geriledi

VERI KAYNAKLARI (oncelik sirasi):
  1. Fintables MCP (`veri_sorgula` → gunluk_fon_degerleri) — GERCEK gunluk net
     nakit giris/cikis + yatirimci + buyukluk GECMIS serisi. CI'da Bearer token
     (FINTABLES_MCP_TOKEN) ile, mevcut FintablesMCPClient uzerinden. Isinma YOK.
  2. borsapy (TEFAS) — token yoksa fallback. Sadece ANLIK snapshot verir; net
     akisi pay-sayisi (=buyukluk/fiyat) degisiminden PROXY'ler ve gecmisi state
     CSV'sinde biriktirir (ilk birkac gun isinma).

Kullanim:
    python tools/fund_flow_monitor.py                 # tara + tetiklenirse uyar
    python tools/fund_flow_monitor.py --digest        # her halukarda durum tablosu
    python tools/fund_flow_monitor.py --dry-run       # Telegram'a gonderme
    python tools/fund_flow_monitor.py --source borsapy # kaynagi zorla
    python tools/fund_flow_monitor.py --list / --add KOD / --remove KOD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Istanbul")
except Exception:  # pragma: no cover
    _TZ = None

CONFIG_PATH = ROOT / "data" / "fund_flow_monitor.json"
STATE_PATH = ROOT / "output" / "fund_flow_monitor_state.csv"

STATE_COLS = ["date", "code", "name", "fund_size", "investor_count", "net_flow", "shares"]

DEFAULT_THRESHOLDS = {
    "flow_window": 5,
    "investor_window": 3,
    "size_window": 5,
    "slowdown_ratio": 0.5,
    "investor_drop_pct": 0.0,
    "size_drop_pct": 0.0,
    "min_history": 4,
}


# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    funds = [str(c).strip().upper() for c in cfg.get("funds", []) if str(c).strip()]
    if not funds:
        funds = ["TLY", "PBR", "PHE", "DFI"]
    th = dict(DEFAULT_THRESHOLDS)
    th.update(cfg.get("thresholds", {}) or {})
    return {"funds": funds, "thresholds": th, "_raw": cfg}


def save_config(cfg_raw: dict, funds: list, thresholds: dict) -> None:
    out = dict(cfg_raw)
    out["funds"] = funds
    out["thresholds"] = thresholds
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _today():
    from datetime import datetime
    return datetime.now(_TZ) if _TZ else datetime.now()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=STATE_COLS)


# ── Kaynak 1: Fintables MCP (veri_sorgula) ───────────────────────────────────

# veri_sorgula explicit LIMIT'i max 300 satirla sinirlar; LIMIT verilmezse 50'de
# keser. Fon basina son-N satiri ROW_NUMBER ile garanti alir, fonlari 300-satir
# butcesine sigacak batch'lere boleriz (yoksa cok fonda sessiz fon dususu olur).
_MCP_ROW_CAP = 300


def fetch_fintables(codes: list, th: dict) -> pd.DataFrame:
    """gunluk_fon_degerleri'nden fon basina son-N gunluk gercek net-akis serisi."""
    from nyxexpansion.intraday.fetchers.fintables import (
        FintablesMCPClient, _parse_markdown_table, _validate_tickers,
    )

    safe = _validate_tickers(codes)
    wmax = max(int(th.get("flow_window", 5)), int(th.get("investor_window", 3)), int(th.get("size_window", 5)))
    per_fund = 2 * wmax + 5  # pencere hesabi icin yeterli son gozlem
    batch_size = max(1, (_MCP_ROW_CAP - 5) // per_fund)

    cli = FintablesMCPClient()
    out = []
    for i in range(0, len(safe), batch_size):
        batch = safe[i:i + batch_size]
        in_clause = ", ".join(f"'{c}'" for c in batch)
        limit = min(len(batch) * per_fund + 5, _MCP_ROW_CAP)
        sql = (
            "WITH ranked AS ("
            " SELECT g.fon_kodu, g.tarih_europe_istanbul, g.fon_buyuklugu, g.yatirimci_sayisi,"
            " g.gunluk_nakit_giris_cikisi, f.unvan,"
            " ROW_NUMBER() OVER (PARTITION BY g.fon_kodu ORDER BY g.tarih_europe_istanbul DESC) AS rn"
            " FROM gunluk_fon_degerleri g LEFT JOIN fonlar f ON f.fon_kodu = g.fon_kodu"
            f" WHERE g.fon_kodu IN ({in_clause}))"
            " SELECT fon_kodu, tarih_europe_istanbul, fon_buyuklugu, yatirimci_sayisi,"
            " gunluk_nakit_giris_cikisi, unvan"
            f" FROM ranked WHERE rn <= {per_fund}"
            f" ORDER BY fon_kodu, tarih_europe_istanbul LIMIT {limit}"
        )
        payload = cli.call_tool("veri_sorgula", {"sql": sql, "purpose": "NOX fon akis izleme"})
        if not isinstance(payload, dict):
            raise RuntimeError(f"veri_sorgula beklenmeyen tip: {type(payload)}")
        for r in _parse_markdown_table(payload.get("table") or ""):
            out.append({
                "date": str(r.get("tarih_europe_istanbul", ""))[:10],
                "code": r.get("fon_kodu"),
                "name": (r.get("unvan") or "")[:48],
                "fund_size": _f(r.get("fon_buyuklugu")),
                "investor_count": _i(r.get("yatirimci_sayisi")),
                "net_flow": _f(r.get("gunluk_nakit_giris_cikisi")),
                "shares": float("nan"),
            })

    df = pd.DataFrame(out, columns=STATE_COLS)
    got = set(df["code"]) if not df.empty else set()
    missing = [c for c in safe if c not in got]
    if not df.empty:
        print(f"  ✓ Fintables: {len(df)} satir / {df['code'].nunique()}/{len(safe)} fon (fon basina ≤{per_fund})")
    if missing:
        print(f"  ⚠️ Fintables'ta veri bulunamayan fon(lar): {', '.join(missing)}")
    return df


def _f(s):
    try:
        if s is None or str(s).strip().lower() in ("", "null", "none", "nan"):
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _i(s):
    v = _f(s)
    return int(v) if v == v else None  # NaN -> None


# ── Kaynak 2: borsapy (fallback, snapshot) ───────────────────────────────────

def fetch_borsapy(codes: list) -> pd.DataFrame:
    import borsapy as bp
    rows = []
    for code in codes:
        try:
            info = bp.Fund(code).info or {}
            price, size, inv = info.get("price"), info.get("fund_size"), info.get("investor_count")
            if not price or not size:
                print(f"  ⚠️ {code}: fiyat/buyukluk bos, atlandi")
                continue
            tdate = info.get("date")
            tdate = pd.to_datetime(tdate).strftime("%Y-%m-%d") if tdate else _today().strftime("%Y-%m-%d")
            rows.append({
                "date": tdate, "code": code, "name": (info.get("name") or "")[:48],
                "fund_size": float(size), "investor_count": _i(inv),
                "net_flow": float("nan"), "shares": float(size) / float(price),
            })
            print(f"  ✓ borsapy {code}: {tdate} buyukluk={float(size):,.0f}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ {code}: borsapy cekilemedi — {type(e).__name__}: {e}")
    return pd.DataFrame(rows, columns=STATE_COLS)


def fetch_history(funds: list, th: dict, source: str) -> tuple:
    """(df, kullanilan_kaynak). source: 'auto'|'fintables'|'borsapy'."""
    token = os.environ.get("FINTABLES_MCP_TOKEN")
    want = source
    if want == "auto":
        want = "fintables" if token else "borsapy"

    if want == "fintables":
        try:
            df = fetch_fintables(funds, th)
            if not df.empty:
                return df, "fintables"
            print("  ⚠️ Fintables 0 satir — borsapy'ye dusuluyor")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ Fintables kaynak hatasi ({type(e).__name__}: {e}) — borsapy'ye dusuluyor")
    return fetch_borsapy(funds), "borsapy"


# ── State (sadece borsapy fallback gecmisi icin birikim) ─────────────────────

def load_state() -> pd.DataFrame:
    if STATE_PATH.exists():
        df = pd.read_csv(STATE_PATH)
        for c in STATE_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[STATE_COLS]
    return _empty_df()


def merge_state(state: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if fresh.empty:
        return state
    combined = fresh.copy() if state.empty else pd.concat([state, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "code"], keep="last")
    return combined.sort_values(["code", "date"]).reset_index(drop=True)


def save_state(state: pd.DataFrame) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(STATE_PATH, index=False)


# ── Sinyaller ────────────────────────────────────────────────────────────────

def _pct(a, b):
    if b in (None, 0) or pd.isna(b):
        return None
    return (a - b) / b


def evaluate_fund(hist: pd.DataFrame, th: dict) -> dict:
    hist = hist.sort_values("date").reset_index(drop=True)
    n = len(hist)
    latest = hist.iloc[-1]
    out = {
        "code": latest["code"], "name": latest["name"], "date": latest["date"],
        "fund_size": _f(latest["fund_size"]), "investor_count": latest["investor_count"],
        "history": n, "alerts": [], "warming": False,
    }
    if n < int(th.get("min_history", 4)):
        out["warming"] = True
        return out

    fw, iw, sw = int(th["flow_window"]), int(th["investor_window"]), int(th["size_window"])
    size = [_f(x) for x in hist["fund_size"].tolist()]
    inv = hist["investor_count"].tolist()
    net = [_f(x) for x in hist["net_flow"].tolist()]
    shares = [_f(x) for x in hist["shares"].tolist()]

    # --- Akis ---
    has_true_flow = any(v == v for v in net)  # en az bir non-NaN
    if has_true_flow:
        # gercek gunluk net akisin pencere toplami (TL)
        def wsum(lo, hi):
            if lo < 0:
                return None
            vals = [v for v in net[lo:hi] if v == v]
            return sum(vals) if vals else 0.0
        f_now = wsum(n - fw, n)
        f_prev = wsum(n - 2 * fw, n - fw)
        unit = "₺"
    else:
        # proxy: pay sayisi degisimi (NAV-arindirilmis)
        def wdiff(lo, hi):
            if lo < 0 or shares[hi - 1] != shares[hi - 1] or shares[lo] != shares[lo]:
                return None
            return shares[hi - 1] - shares[lo]
        f_now = wdiff(n - 1 - fw, n)
        f_prev = wdiff(n - 1 - 2 * fw, n - fw)
        unit = "pay"

    if f_now is not None:
        out["flow_now"], out["flow_prev"], out["flow_unit"] = f_now, f_prev, unit
        if f_now < 0 and (f_prev is None or f_prev >= 0):
            out["alerts"].append(("AKIS_NEGATIF", f"net akis negatife dondu (son {fw}g {f_now:,.0f} {unit})"))
        elif f_now > 0 and f_prev and f_prev > 0 and f_now < f_prev * float(th["slowdown_ratio"]):
            out["alerts"].append(("AKIS_YAVAS", f"net giris yavasliyor (son {fw}g {f_now:,.0f} < onceki {f_prev:,.0f} {unit})"))

    # --- Yatirimci dususu ---
    iv_now, iv_ref = _f(inv[-1]), _f(inv[max(0, n - 1 - iw)])
    if iv_now == iv_now and iv_ref == iv_ref:
        ch = _pct(iv_now, iv_ref)
        out["investor_change"] = ch
        if ch is not None and ch < -abs(float(th["investor_drop_pct"])):
            out["alerts"].append(("YATIRIMCI_DUSUS", f"yatirimci {iw}g once {iv_ref:,.0f} → {iv_now:,.0f} ({ch:+.2%})"))

    # --- Buyukluk zayiflama ---
    if n - 1 - sw >= 0:
        size_ch = _pct(size[-1], size[n - 1 - sw])
        out["size_change"] = size_ch
        if size_ch is not None and size_ch < -abs(float(th["size_drop_pct"])):
            out["alerts"].append(("BUYUKLUK_ZAYIF", f"buyukluk {sw}g icinde {size_ch:+.2%}"))

    return out


# ── Mesaj ────────────────────────────────────────────────────────────────────

_ALERT_EMOJI = {"AKIS_NEGATIF": "🔴", "AKIS_YAVAS": "🟡", "YATIRIMCI_DUSUS": "👥", "BUYUKLUK_ZAYIF": "📉"}


def build_message(results: list, source: str, digest: bool) -> tuple:
    triggered = [r for r in results if r["alerts"]]
    warming = [r for r in results if r["warming"]]
    lines = ["⚠️ <b>NOX Fon Akis Uyarisi</b>" if triggered else "🟢 <b>NOX Fon Akis — durum normal</b>"]

    for r in triggered:
        lines.append("")
        lines.append(f"<b>{r['code']}</b> · {r['name']} ({r['date']})")
        lines.append(f"  buyukluk {r['fund_size']:,.0f} ₺ · yatirimci {_fmt_int(r['investor_count'])}")
        for tag, desc in r["alerts"]:
            lines.append(f"  {_ALERT_EMOJI.get(tag, '•')} {desc}")

    if digest:
        lines.append("")
        lines.append("— <b>Tum fonlar</b> —")
        for r in sorted(results, key=lambda x: x["code"]):
            flag = "⚠️" if r["alerts"] else ("⏳" if r["warming"] else "🟢")
            extra = ""
            if r.get("investor_change") is not None:
                extra += f" yat {r['investor_change']:+.1%}"
            if r.get("size_change") is not None:
                extra += f" buy {r['size_change']:+.1%}"
            warm = " (isinma)" if r["warming"] else ""
            lines.append(f"{flag} <b>{r['code']}</b> {r['fund_size']:,.0f}₺{extra}{warm}")

    if warming and not digest:
        lines.append("")
        lines.append(f"⏳ isinma (yetersiz gecmis): {', '.join(r['code'] for r in warming)}")

    lines.append("")
    lines.append(f"<i>kaynak: {source}</i>")
    return "\n".join(lines), bool(triggered)


def _fmt_int(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{int(v):,}"


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_list(cfg):
    print("Izlenen fonlar:", ", ".join(cfg["funds"]))
    print("Esikler:", json.dumps(cfg["thresholds"], ensure_ascii=False))


def cmd_add(cfg, code, source):
    code = code.strip().upper()
    if code in cfg["funds"]:
        print(f"{code} zaten listede.")
        return
    df, _ = fetch_history([code], cfg["thresholds"], source)
    if df.empty:
        print(f"❌ {code} cekilemedi — eklenmedi.")
        return
    cfg["funds"].append(code)
    save_config(cfg["_raw"], cfg["funds"], cfg["thresholds"])
    print(f"✓ {code} eklendi → {', '.join(cfg['funds'])}")


def cmd_remove(cfg, code):
    code = code.strip().upper()
    if code not in cfg["funds"]:
        print(f"{code} listede yok.")
        return
    cfg["funds"].remove(code)
    save_config(cfg["_raw"], cfg["funds"], cfg["thresholds"])
    print(f"✓ {code} cikarildi → {', '.join(cfg['funds'])}")


def run_monitor(cfg, source: str, digest: bool, dry_run: bool) -> int:
    funds, th = cfg["funds"], cfg["thresholds"]
    print(f"📡 Fon verisi cekiliyor ({source}): {', '.join(funds)}")
    fresh, used = fetch_history(funds, th, source)
    if fresh.empty:
        print("❌ Hicbir fon cekilemedi — cikiliyor.")
        return 1

    # borsapy snapshot ise state'le birikim; fintables zaten gecmis getirir.
    state = merge_state(load_state(), fresh)
    save_state(state)
    print(f"💾 State: {STATE_PATH} ({len(state)} satir)")

    results = [evaluate_fund(state[state["code"] == c], th) for c in funds if not state[state["code"] == c].empty]

    msg, triggered = build_message(results, used, digest=digest)
    print("\n" + "=" * 60)
    print(msg.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    print("=" * 60 + "\n")

    if (digest or triggered) and not dry_run:
        try:
            from core.reports import send_telegram
            send_telegram(msg)
            print("📤 Telegram gonderildi.")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Telegram gonderilemedi: {e}")
    elif dry_run:
        print("ℹ️ --dry-run: gonderilmedi.")
    elif not triggered:
        print("ℹ️ Tetikleyen kosul yok (--digest ile her zaman yollanir).")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="NOX TEFAS Fon Akis Izleyici")
    p.add_argument("--digest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", choices=["auto", "fintables", "borsapy"], default="auto")
    p.add_argument("--list", action="store_true")
    p.add_argument("--add", metavar="KOD")
    p.add_argument("--remove", metavar="KOD")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.list:
        cmd_list(cfg); return 0
    if args.add:
        cmd_add(cfg, args.add, args.source); return 0
    if args.remove:
        cmd_remove(cfg, args.remove); return 0
    return run_monitor(cfg, args.source, digest=args.digest, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
