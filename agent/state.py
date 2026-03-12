"""
NOX Agent — State Management
Watchlist yönetimi: JSON dosya + SQLite.
- GH Actions: output/watchlist.json
- Railway bot: SQLite nox_agent.db
"""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta

_TZ_TR = timezone(timedelta(hours=3))

# ── JSON dosya bazlı watchlist (GH Actions) ──

_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'watchlist.json'
)


def _load_json(path=None):
    path = path or _DEFAULT_JSON_PATH
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"positions": [], "updated_at": None}


def _save_json(data, path=None):
    path = path or _DEFAULT_JSON_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data["updated_at"] = datetime.now(_TZ_TR).isoformat()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── SQLite bazlı watchlist (Railway bot) ──

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'nox_agent.db'
)


def _get_db(db_path=None):
    db_path = db_path or _DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            current_price REAL,
            note TEXT,
            added_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL,
            pnl_pct REAL,
            created_at TEXT NOT NULL,
            UNIQUE(date, ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ── Unified Watchlist API ──

class Watchlist:
    """Watchlist yönetimi — JSON + SQLite senkronize."""

    def __init__(self, json_path=None, db_path=None):
        self.json_path = json_path or _DEFAULT_JSON_PATH
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._ensure_sync()

    def _ensure_sync(self):
        """JSON'dan SQLite'a başlangıç senkronizasyonu."""
        json_data = _load_json(self.json_path)
        if not json_data.get("positions"):
            return
        conn = _get_db(self.db_path)
        for pos in json_data["positions"]:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO watchlist
                    (ticker, entry_price, stop_price, target_price, note, added_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos['ticker'],
                    pos.get('entry_price'),
                    pos.get('stop_price'),
                    pos.get('target_price'),
                    pos.get('note', ''),
                    pos.get('added_at', datetime.now(_TZ_TR).isoformat()),
                    datetime.now(_TZ_TR).isoformat(),
                ))
            except Exception:
                pass
        conn.commit()
        conn.close()

    def list_positions(self):
        """Tüm pozisyonları listele."""
        conn = _get_db(self.db_path)
        rows = conn.execute("SELECT * FROM watchlist ORDER BY ticker").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_position(self, ticker, entry_price=None, stop_price=None,
                     target_price=None, note=""):
        """Pozisyon ekle."""
        ticker = ticker.upper().strip().replace('.IS', '')
        now = datetime.now(_TZ_TR).isoformat()
        conn = _get_db(self.db_path)
        try:
            conn.execute("""
                INSERT INTO watchlist
                (ticker, entry_price, stop_price, target_price, note, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, entry_price, stop_price, target_price, note, now, now))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.execute("""
                UPDATE watchlist SET entry_price=?, stop_price=?, target_price=?,
                note=?, updated_at=? WHERE ticker=?
            """, (entry_price, stop_price, target_price, note, now, ticker))
            conn.commit()
        conn.close()
        self._sync_to_json()
        return True

    def remove_position(self, ticker):
        """Pozisyon çıkar."""
        ticker = ticker.upper().strip().replace('.IS', '')
        conn = _get_db(self.db_path)
        conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))
        conn.commit()
        conn.close()
        self._sync_to_json()
        return True

    def update_position(self, ticker, **kwargs):
        """Pozisyon güncelle."""
        ticker = ticker.upper().strip().replace('.IS', '')
        now = datetime.now(_TZ_TR).isoformat()
        conn = _get_db(self.db_path)
        sets = ["updated_at=?"]
        vals = [now]
        for key in ('entry_price', 'stop_price', 'target_price',
                     'current_price', 'note'):
            if key in kwargs and kwargs[key] is not None:
                sets.append(f"{key}=?")
                vals.append(kwargs[key])
        vals.append(ticker)
        conn.execute(f"UPDATE watchlist SET {','.join(sets)} WHERE ticker=?", vals)
        conn.commit()
        conn.close()
        self._sync_to_json()
        return True

    def update_current_prices(self, price_map):
        """Güncel fiyatları toplu güncelle. price_map: {ticker: price}"""
        conn = _get_db(self.db_path)
        now = datetime.now(_TZ_TR).isoformat()
        for ticker, price in price_map.items():
            ticker = ticker.upper().strip().replace('.IS', '')
            conn.execute(
                "UPDATE watchlist SET current_price=?, updated_at=? WHERE ticker=?",
                (price, now, ticker))
        conn.commit()
        conn.close()

    def save_daily_snapshot(self, date_str=None):
        """Günlük snapshot kaydet."""
        if date_str is None:
            date_str = datetime.now(_TZ_TR).strftime('%Y-%m-%d')
        positions = self.list_positions()
        if not positions:
            return
        conn = _get_db(self.db_path)
        now = datetime.now(_TZ_TR).isoformat()
        for pos in positions:
            entry = pos.get('entry_price')
            current = pos.get('current_price')
            pnl = None
            if entry and current and entry > 0:
                pnl = round(((current / entry) - 1) * 100, 2)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_snapshots
                    (date, ticker, close_price, pnl_pct, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (date_str, pos['ticker'], current, pnl, now))
            except Exception:
                pass
        conn.commit()
        conn.close()

    def _sync_to_json(self):
        """SQLite → JSON senkronizasyonu."""
        positions = self.list_positions()
        json_positions = []
        for pos in positions:
            json_positions.append({
                "ticker": pos['ticker'],
                "entry_price": pos.get('entry_price'),
                "stop_price": pos.get('stop_price'),
                "target_price": pos.get('target_price'),
                "note": pos.get('note', ''),
                "added_at": pos.get('added_at', ''),
            })
        _save_json({"positions": json_positions}, self.json_path)

    def format_watchlist(self):
        """Watchlist'i okunabilir formatta döndür."""
        positions = self.list_positions()
        if not positions:
            return "📋 Watchlist boş."

        lines = [f"<b>📋 Watchlist — {len(positions)} pozisyon</b>", ""]
        for pos in positions:
            ticker = pos['ticker']
            entry = pos.get('entry_price')
            current = pos.get('current_price')
            stop = pos.get('stop_price')
            target = pos.get('target_price')
            note = pos.get('note', '')

            pnl_str = ""
            if entry and current and entry > 0:
                pnl = ((current / entry) - 1) * 100
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                pnl_str = f" {pnl_emoji} {pnl:+.1f}%"

            line = f"<b>{ticker}</b>"
            if entry:
                line += f" giriş:{entry:.2f}"
            if current:
                line += f" şimdi:{current:.2f}"
            line += pnl_str
            if stop:
                line += f" S:{stop:.2f}"
            if target:
                line += f" TP:{target:.2f}"
            lines.append(line)
            if note:
                lines.append(f"  💬 {note}")

        return "\n".join(lines)

    # ── Conversation tracking ──

    def save_message(self, chat_id, role, content):
        """Mesajı kaydet."""
        conn = _get_db(self.db_path)
        now = datetime.now(_TZ_TR).isoformat()
        conn.execute("""
            INSERT INTO conversations (chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (str(chat_id), role, content, now))
        conn.commit()
        conn.close()

    def get_recent_messages(self, chat_id, limit=10):
        """Son mesajları getir."""
        conn = _get_db(self.db_path)
        rows = conn.execute("""
            SELECT role, content FROM conversations
            WHERE chat_id=? ORDER BY id DESC LIMIT ?
        """, (str(chat_id), limit)).fetchall()
        conn.close()
        return [{"role": r['role'], "content": r['content']} for r in reversed(rows)]
