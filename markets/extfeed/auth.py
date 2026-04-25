"""Cookie-based JWT helper for the external intraday feed.

Session cookies (long-lived) are exchanged for a short-lived JWT by fetching
the home page and parsing `auth_token` from the rendered HTML. The JWT is
cached in-process and refreshed when within 5 minutes of expiry.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from threading import Lock

import requests

_DEFAULT_HOST = "www.tradingview.com"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
)
_TOKEN_RE = re.compile(r'"auth_token":"([^"]+)"')


def _decode_jwt_exp(jwt: str) -> int:
    payload_b64 = jwt.split(".")[1]
    payload_b64 += "=" * (4 - len(payload_b64) % 4)
    return int(json.loads(base64.urlsafe_b64decode(payload_b64))["exp"])


class AuthCache:
    """Thread-safe JWT cache backed by session cookies."""

    def __init__(self, sid: str, sign: str, host: str | None = None):
        if not sid or not sign:
            raise ValueError("sid and sign are required")
        self._sid = sid
        self._sign = sign
        self._host = host or os.getenv("INTRADAY_HOST", _DEFAULT_HOST)
        self._jwt: str | None = None
        self._exp: int = 0
        self._lock = Lock()

    @property
    def expires_at(self) -> int:
        return self._exp

    def token(self) -> str:
        with self._lock:
            now = int(time.time())
            if self._jwt and now < self._exp - 300:
                return self._jwt
            self._jwt = self._fetch()
            self._exp = _decode_jwt_exp(self._jwt)
            return self._jwt

    def _fetch(self) -> str:
        url = f"https://{self._host}/"
        cookies = {"sessionid": self._sid, "sessionid_sign": self._sign}
        r = requests.get(
            url,
            cookies=cookies,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            timeout=20,
        )
        r.raise_for_status()
        m = _TOKEN_RE.search(r.text)
        if not m:
            raise RuntimeError(
                "auth_token not found in response — session cookies may be expired"
            )
        return m.group(1)


def auth_from_env() -> AuthCache:
    sid = os.getenv("INTRADAY_SID")
    sign = os.getenv("INTRADAY_SIGN")
    if not sid or not sign:
        raise RuntimeError(
            "INTRADAY_SID / INTRADAY_SIGN env vars not set"
        )
    return AuthCache(sid, sign)
