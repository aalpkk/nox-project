"""Daily-bar layered fetcher (3-tier: fintables -> matriks -> yfinance).

Mirrors ``nyxexpansion.intraday`` for daily OHLCV. Used by GitHub Actions
scan workflows so every BIST scanner shares one canonical fetch path.
"""
