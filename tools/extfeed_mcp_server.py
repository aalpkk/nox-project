#!/usr/bin/env python3
"""NOX extfeed MCP sunucusu — claude.ai custom connector için canlı veri erişimi.

Render'da web servisi olarak koşar (render.yaml: nox-extfeed-mcp). Claude app
Settings > Connectors > "Add custom connector" ile şu URL bağlanır:

    https://<render-host>/<MCP_TOKEN>/mcp

Tool'lar:
  us_bars / bist_bars   — canlı extfeed (TV WS) OHLCV; ABD sembolleri NASDAQ
                          Trader haritasıyla borsaya yönlendirilir.
  us_snapshot           — <=12 sembol için canlı günlük metrik + kurulum etiketi
                          (us_watch_scan ile aynı kural seti).
  us_premarket_gaps     — izleme evreni premarket gap listesi (yfinance prepost).
  published_scan        — nox-signals'taki son yayınlanmış tarama özetleri.

Korkuluklar: TV WS erişimi tek kanaldan (Lock), 120 sn cache, bar sayısı ve
sembol sayısı sınırlı — BIST kapanış zincirinin paylaştığı TV session'ı korunur.

Env: INTRADAY_SID, INTRADAY_SIGN (TV session), MCP_TOKEN (URL gizliliği), PORT.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:  # lokal koşularda repo/.env ve bir üst klasördeki .env otomatik yüklenir
    from dotenv import load_dotenv
    for _p in (REPO / ".env", REPO.parent / ".env"):
        load_dotenv(_p)
except ImportError:
    pass

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

TF_OK = {"1", "5", "15", "60", "240", "D", "W"}
N_CAP = 500
SNAP_CAP = 12
CACHE_TTL = 120.0

_lock = threading.Lock()
_cache: dict[tuple, tuple[float, object]] = {}
_auth = None

server = MCPServer(
    name="nox-extfeed",
    instructions=(
        "NOX canlı piyasa verisi. ABD/BIST OHLCV için us_bars/bist_bars, "
        "izleme evreni anlık durumu için us_snapshot, premarket için "
        "us_premarket_gaps, son yayınlanmış taramalar için published_scan. "
        "Etiketler betimseldir, valide edge değildir."
    ),
)


def _get_auth():
    global _auth
    if _auth is None:
        from markets.extfeed import auth_from_env
        _auth = auth_from_env()
    return _auth


def _cached(key, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _fetch_live(symbol: str, timeframe: str, n: int) -> pd.DataFrame:
    from markets.extfeed import fetch_bars
    if timeframe not in TF_OK:
        raise ValueError(f"timeframe {timeframe!r} desteklenmiyor ({sorted(TF_OK)})")
    n = max(2, min(int(n), N_CAP))

    def pull():
        with _lock:  # TV WS'e ayni anda tek istek
            return fetch_bars(symbol, timeframe, n_bars=n, auth=_get_auth())
    return _cached(("bars", symbol, timeframe, n), pull)


def _us_symbol(code: str) -> str:
    from markets.us.data import _fetch_us_exchange_map
    code = code.upper().strip()
    exch = _cached(("exmap",), _fetch_us_exchange_map).get(code, "NASDAQ")
    return f"{exch}:{code}"


def _bars_out(df: pd.DataFrame, tz: str) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        t = r["time"].tz_convert(tz)
        out.append({"t": str(t)[:16], "o": round(float(r["open"]), 4),
                    "h": round(float(r["high"]), 4), "l": round(float(r["low"]), 4),
                    "c": round(float(r["close"]), 4), "v": int(r["volume"])})
    return out


@server.tool()
def us_bars(symbol: str, timeframe: str = "D", n: int = 100) -> list[dict]:
    """ABD hissesi/ETF için canlı OHLCV barları (extfeed TV kaynağı).

    Args:
        symbol: Ticker, ör. NVDA, SPY. Borsa öneki otomatik bulunur.
        timeframe: 1/5/15/60/240 (dakika), D (günlük), W (haftalık).
        n: Bar sayısı (en çok 500).
    Returns: [{t,o,h,l,c,v}] — t New York saati. Seans içinde son bar KISMİDİR.
    """
    return _bars_out(_fetch_live(_us_symbol(symbol), timeframe, n), "America/New_York")


@server.tool()
def bist_bars(symbol: str, timeframe: str = "D", n: int = 100) -> list[dict]:
    """BIST hissesi için canlı OHLCV barları (extfeed TV kaynağı).

    Args:
        symbol: Ticker, ör. THYAO, ASELS ("BIST:" öneki otomatik eklenir).
        timeframe: 1/5/15/60/240 (dakika), D (günlük), W (haftalık).
        n: Bar sayısı (en çok 500).
    Returns: [{t,o,h,l,c,v}] — t İstanbul saati. Seans içinde son bar KISMİDİR.
    """
    sym = symbol.upper().strip()
    if not sym.startswith("BIST:"):
        sym = f"BIST:{sym}"
    return _bars_out(_fetch_live(sym, timeframe, n), "Europe/Istanbul")


@server.tool()
def us_snapshot(symbols: list[str]) -> dict:
    """Verilen ABD sembolleri (en çok 12) için CANLI günlük metrik + kurulum etiketi.

    us_watch_scan kural seti: SINDIRME/BREAKOUT_1G/BOLGEDE/TABAN/UZAMIS/ZAYIF.
    Seans içindeyken son bar kısmi — 1g% ve hacim oranı gün sonu değildir.
    """
    from tools.us_watch_scan import summarize, classify, session_state_ny
    syms = [s.upper().strip() for s in symbols][:SNAP_CAP]
    out, errors = {}, {}
    for s in syms:
        try:
            df = _fetch_live(_us_symbol(s), "D", 300).copy()
            df["date"] = df["time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            rec = summarize("-", df)
            rec["tags"], rec["ext20_pct"] = classify(rec)
            del rec["group"]
            out[s] = rec
        except Exception as e:
            errors[s] = str(e)[:120]
    return {"session_state_ny": session_state_ny(), "symbols": out, "errors": errors}


@server.tool()
def us_premarket_gaps() -> dict:
    """İzleme evreni (config/us_watch_universe.yaml) premarket gap listesi.

    Kaynak: yfinance prepost 5m. Premarket penceresi 04:00-09:30 ET; pencere
    dışında en son premarket verisi döner. Gap referansı önceki seans kapanışı.
    """
    def pull():
        import yfinance as yf
        from tools.us_watch_scan import load_universe
        universe, _ = load_universe()
        syms = list(universe)
        now_ny = pd.Timestamp.now(tz="America/New_York")
        today = now_ny.normalize()
        pm_s, pm_e = today + pd.Timedelta(hours=4), today + pd.Timedelta(hours=9, minutes=30)
        m5 = yf.download(syms, period="5d", interval="5m", prepost=True,
                         group_by="ticker", threads=True, progress=False,
                         auto_adjust=False)
        rows = {}
        for s in syms:
            try:
                b_all = m5[s].dropna(subset=["Close"])
                prev = b_all[b_all.index < today]
                prev = prev[(prev.index.hour * 60 + prev.index.minute >= 570)
                            & (prev.index.hour < 16)]
                pc = float(prev["Close"].iloc[-1])
                b = b_all[(b_all.index >= pm_s) & (b_all.index < pm_e)]
                if b.empty:
                    rows[s] = {"group": universe[s], "prev_close": round(pc, 2),
                               "pm_last": None, "pm_gap_pct": None}
                    continue
                last = float(b["Close"].iloc[-1])
                rows[s] = {"group": universe[s], "prev_close": round(pc, 2),
                           "pm_last": round(last, 2),
                           "pm_gap_pct": round(100 * (last / pc - 1), 2),
                           "pm_lo": round(float(b["Low"].min()), 2),
                           "pm_hi": round(float(b["High"].max()), 2),
                           "pm_last_bar_ny": str(b.index[-1])[:16]}
            except Exception as e:
                rows[s] = {"error": str(e)[:80]}
        return {"ny_time": str(now_ny)[:16], "symbols": rows}
    return _cached(("premarket",), pull)


@server.tool()
def published_scan(which: str = "summary") -> str:
    """nox-signals'taki son YAYINLANMIŞ tarama çıktısını getirir (canlı değil).

    Args:
        which: summary (izleme evreni json) | us_scan | premarket | full_scan.
    """
    import requests
    files = {"summary": "summary.json", "us_scan": "us_scan.md",
             "premarket": "premarket_scan.md", "full_scan": "full_scan.md"}
    if which not in files:
        raise ValueError(f"which {which!r}: {sorted(files)} bekleniyor")
    url = f"https://raw.githubusercontent.com/aalpkk/nox-signals/main/usdata/{files[which]}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text[:60000]


def main() -> None:
    import uvicorn
    token = os.environ.get("MCP_TOKEN", "").strip()
    path = f"/{token}/mcp" if token else "/mcp"
    app = server.streamable_http_app(
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False),
    )
    port = int(os.environ.get("PORT", "8756"))
    print(f"[nox-extfeed-mcp] 0.0.0.0:{port}{path}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
