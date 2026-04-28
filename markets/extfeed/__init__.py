"""External intraday bar feed via WebSocket — cookie-based session auth.

Public surface:
    fetch_bars(symbol, timeframe, n_bars) -> pd.DataFrame
    auth_from_env() -> AuthCache
"""
from .auth import AuthCache, auth_from_env
from .ws_client import fetch_bars, fetch_bars_until

__all__ = ["AuthCache", "auth_from_env", "fetch_bars", "fetch_bars_until"]
