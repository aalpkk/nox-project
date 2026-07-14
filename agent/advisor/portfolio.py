"""
NOX Advisor — Portföy durumu (gerçek para).

Kaynak çözümleme sırası:
  1. Açık path (--portfolio-path / NOX_PORTFOLIO_PATH)
  2. GitHub Contents API (GH_TOKEN + NOX_PORTFOLIO_REPO, özel yan repo)
  3. Lokal agent/portfolio.json (gitignored, dev fallback)

agent/state.py Watchlist'ten AYRI: o aday takipçisi, bu gerçek pozisyonlar.
Yazma çakışması: Contents API PUT-with-sha = optimistic lock; sha uyuşmazlığında
mutasyon yeniden uygulanıp tekrar denenir (max 3).
"""
import os
import json
import base64
import datetime
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPO_PATH = os.environ.get("NOX_PORTFOLIO_REPO_PATH", "portfolio/portfolio.json")
LOCAL_FALLBACK = Path(__file__).resolve().parent.parent / "portfolio.json"
ADVISORY_LATEST_PATH = "advisor/advisory_latest.json"
# Public GH Pages advisor HTML yolu (kullanıcı onayıyla — rapor portföyü içerir).
PAGES_HTML_DIR = "advisor"

DEFAULT_SETTINGS = {
    "max_position_pct": 15.0,
    "per_trade_risk_pct": 1.0,
    "max_total_risk_pct": 6.0,
    "max_positions": 12,
    "min_cash_buffer_pct": 5.0,
}


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def empty_portfolio():
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "updated_by": "init",
        "cash_tl": 0.0,
        "settings": dict(DEFAULT_SETTINGS),
        "positions": [],
        "trades": [],
    }


# ── GitHub Contents API (push_html_to_github deseninin JSON varyantı) ──

def _gh_headers():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def gh_get_json(repo, path):
    """Private repodan JSON dosyası oku. (obj, sha) döner; yoksa (None, None)."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    resp = requests.get(api_url, headers=_gh_headers(), timeout=15)
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    body = resp.json()
    raw = base64.b64decode(body["content"]).decode("utf-8")
    return json.loads(raw), body.get("sha")


def gh_put_json(repo, path, obj, message, sha=None):
    """JSON dosyasını private repoya yaz. Yeni blob sha döner.
    sha uyuşmazlığında requests.HTTPError(409/422) fırlatır — çağıran retry eder."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    content_b64 = base64.b64encode(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    ).decode("ascii")
    payload = {"message": message, "content": content_b64, "branch": "main"}
    if sha:
        payload["sha"] = sha
    resp = requests.put(api_url, headers=_gh_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["content"]["sha"]


class Portfolio:
    """Gerçek portföy durumu + audit log. positions türetilmiş, trades append-only."""

    def __init__(self, data, rev=None, source="local", local_path=None):
        self.data = data
        self.rev = rev          # github blob sha (lokalde None)
        self.source = source    # "github" | "local"
        self.local_path = local_path

    # ── yükleme ──

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("NOX_PORTFOLIO_PATH", "")
        if path:
            p = Path(path)
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else empty_portfolio()
            return cls(cls._migrate(data), source="local", local_path=p)

        repo = os.environ.get("NOX_PORTFOLIO_REPO", "")
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
        if repo and token:
            obj, sha = gh_get_json(repo, DEFAULT_REPO_PATH)
            if obj is not None:
                return cls(cls._migrate(obj), rev=sha, source="github")
            # repo var ama dosya yok — boş başlat (ilk save oluşturur)
            return cls(empty_portfolio(), rev=None, source="github")

        p = LOCAL_FALLBACK
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else empty_portfolio()
        return cls(cls._migrate(data), source="local", local_path=p)

    @staticmethod
    def _migrate(data):
        """Eski şema (holdings/watchlist) veya eksik alanlar → v1 şema."""
        if "positions" not in data:
            # eski agent/portfolio.json formatı: holdings sadece ticker+note
            old = data.get("holdings", [])
            data = empty_portfolio() | {
                "positions": [
                    {"ticker": h["ticker"], "qty": 0, "avg_cost": 0.0,
                     "entry_date": None, "stop": None, "target": None,
                     "notes": h.get("note", "")} for h in old
                ],
            }
        data.setdefault("version", 1)
        data.setdefault("cash_tl", 0.0)
        data.setdefault("trades", [])
        data["settings"] = dict(DEFAULT_SETTINGS) | data.get("settings", {})
        for pos in data["positions"]:
            pos.setdefault("stop", None)
            pos.setdefault("target", None)
            pos.setdefault("notes", "")
        return data

    # ── kaydetme (optimistic lock) ──

    def save(self, updated_by="advisor", message=None):
        self.data["updated_at"] = _now_iso()
        self.data["updated_by"] = updated_by
        if self.source == "github":
            repo = os.environ["NOX_PORTFOLIO_REPO"]
            msg = message or f"portfolio update ({updated_by})"
            self.rev = gh_put_json(repo, DEFAULT_REPO_PATH, self.data, msg, sha=self.rev)
        else:
            p = self.local_path or LOCAL_FALLBACK
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
        return self.rev

    @classmethod
    def mutate(cls, mutator, updated_by="telegram_bot", message=None, max_retries=3):
        """Yükle → mutator(data) → kaydet; sha çakışmasında yeniden yükle + uygula.
        mutator: data dict'i yerinde değiştiren ve özet str döndüren fonksiyon."""
        last_err = None
        for _ in range(max_retries):
            pf = cls.load()
            summary = mutator(pf.data)
            try:
                pf.save(updated_by=updated_by, message=message or summary)
                return pf, summary
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (409, 422):
                    last_err = e
                    continue  # bayat sha — yeniden yükle, yeniden uygula
                raise
        raise RuntimeError(f"Portföy kaydı {max_retries} denemede çakıştı: {last_err}")

    # ── mutasyonlar (saf data-dict işlemleri) ──

    @staticmethod
    def apply_buy(data, ticker, qty, price, date=None, source="telegram"):
        """Ağırlıklı ortalama maliyet; nakit düşer; trade log'a eklenir."""
        ticker = ticker.upper().strip()
        qty = int(qty)
        price = float(price)
        if qty <= 0 or price <= 0:
            raise ValueError("qty ve price pozitif olmalı")
        cost = qty * price
        if cost > data["cash_tl"] + 1e-6:
            raise ValueError(f"Yetersiz nakit: {cost:,.0f} TL gerekli, {data['cash_tl']:,.0f} TL var")
        pos = next((p for p in data["positions"] if p["ticker"] == ticker), None)
        if pos:
            total_qty = pos["qty"] + qty
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + cost) / total_qty
            pos["qty"] = total_qty
        else:
            data["positions"].append({
                "ticker": ticker, "qty": qty, "avg_cost": price,
                "entry_date": date or datetime.date.today().isoformat(),
                "stop": None, "target": None, "notes": "",
            })
        data["cash_tl"] -= cost
        data["trades"].append({"ts": _now_iso(), "action": "BUY", "ticker": ticker,
                               "qty": qty, "price": price, "source": source})
        return f"BUY {ticker} {qty}@{price}"

    @staticmethod
    def apply_sell(data, ticker, qty=None, price=None, source="telegram"):
        """qty None = tamamı. Realized PnL döner (price verilmişse)."""
        ticker = ticker.upper().strip()
        pos = next((p for p in data["positions"] if p["ticker"] == ticker), None)
        if not pos:
            raise ValueError(f"{ticker} portföyde yok")
        sell_qty = int(qty) if qty else pos["qty"]
        if sell_qty <= 0 or sell_qty > pos["qty"]:
            raise ValueError(f"Geçersiz adet: {sell_qty} (elde {pos['qty']})")
        price = float(price) if price else None
        realized = (price - pos["avg_cost"]) * sell_qty if price else None
        pos["qty"] -= sell_qty
        if price:
            data["cash_tl"] += sell_qty * price
        if pos["qty"] == 0:
            data["positions"] = [p for p in data["positions"] if p["ticker"] != ticker]
        data["trades"].append({"ts": _now_iso(), "action": "SELL", "ticker": ticker,
                               "qty": sell_qty, "price": price, "source": source,
                               "realized_pnl_tl": realized})
        return f"SELL {ticker} {sell_qty}" + (f"@{price}" if price else "")

    @staticmethod
    def apply_set_cash(data, amount):
        amount = float(amount)
        if amount < 0:
            raise ValueError("Nakit negatif olamaz")
        data["cash_tl"] = amount
        return f"CASH={amount:,.0f} TL"

    # ── okuma yardımcıları ──

    def tickers(self):
        return [p["ticker"] for p in self.data["positions"] if p["qty"] > 0]

    def equity(self, prices):
        """prices: {ticker: last}; fiyatı olmayan pozisyon avg_cost ile değerlenir."""
        val = self.data["cash_tl"]
        for p in self.data["positions"]:
            last = prices.get(p["ticker"]) or p["avg_cost"]
            val += p["qty"] * last
        return val


def publish_advisory_latest(advisory):
    """advisory_latest.json'ı private repoya yaz (best-effort, bot /danis için)."""
    repo = os.environ.get("NOX_PORTFOLIO_REPO", "")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    try:
        _, sha = gh_get_json(repo, ADVISORY_LATEST_PATH)
        return gh_put_json(repo, ADVISORY_LATEST_PATH, advisory,
                           f"advisory {advisory.get('asof', '?')}", sha=sha)
    except Exception as e:
        print(f"⚠️ advisory_latest publish hatası: {e}")
        return None


def _gh_put_raw(repo, path, content_bytes, message):
    """Ham (JSON olmayan) içeriği repoya yaz — mevcut sha varsa üstüne."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    sha = None
    try:
        r = requests.get(api_url, headers=_gh_headers(), timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(api_url, headers=_gh_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["commit"]["sha"]


def publish_html_to_pages(html_path, asof):
    """Advisor HTML raporunu public GH Pages reposuna (GH_PAGES_REPO) yaz.

    ⚠️ Rapor portföyü (pozisyon/adet/maliyet/PnL/nakit) içerir; bu yayın
    KULLANICI ONAYIYLA açıldı (2026-07-14). GH_PAGES_REPO tanımlı değilse
    sessizce atlar (dev/gizli mod). {asof}'lu dosya + index.html (en son) yazar."""
    repo = os.environ.get("GH_PAGES_REPO", "")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    try:
        html_bytes = Path(html_path).read_bytes()
    except Exception as e:
        print(f"⚠️ advisor HTML okunamadı ({html_path}): {e}")
        return None
    urls = []
    for fn in (f"{PAGES_HTML_DIR}/advisor_report_{asof}.html",
               f"{PAGES_HTML_DIR}/index.html"):
        try:
            _gh_put_raw(repo, fn, html_bytes, f"advisor rapor {asof}")
            owner, name = repo.split("/")
            urls.append(f"https://{owner}.github.io/{name}/{fn}")
        except Exception as e:
            print(f"⚠️ advisor HTML GH Pages publish hatası ({fn}): {e}")
    if urls:
        print(f"✅ advisor HTML yayınlandı: {urls[0]}")
        return urls[0]
    return None


def fetch_advisory_latest():
    """Private repodan son advisory'yi çek (bot /danis)."""
    repo = os.environ.get("NOX_PORTFOLIO_REPO", "")
    if not repo:
        return None
    obj, _ = gh_get_json(repo, ADVISORY_LATEST_PATH)
    return obj
