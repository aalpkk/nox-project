"""
NOX Project — Core Config
Ortak parametreler + market-specific override sistemi
"""

# ── SHARED INDICATOR PARAMS ──
EMA_FAST = 21
EMA_SLOW = 55
ST_LEN = 10
ST_MULT = 3.0
ADX_LEN = 14
ADX_SLOPE_LEN = 5
ADX_SLOPE_THRESH = 0.5
BB_LEN = 20
BB_MULT = 2.0
SQ_LEN = 20
SQ_MULT_BB = 2.0
SQ_MULT_KC = 1.5
DONCH_LEN = 20
MR_RSI_LEN = 14
MR_RSI_THRESH = 35
ATR_LEN = 14
RS_LEN1 = 10
RS_LEN2 = 50
BOS_LOOKBACK = 10
BOS_TIGHT = 3
CHOCH_TIGHT = 2

# ── SHARED SIGNAL NAMES & EMOJI ──
SIGNAL_NAMES = {
    "COMBO+": "COMBO+", "COMBO": "COMBO", "STRONG": "STRONG",
    "WEAK": "WEAK", "REVERSAL": "REVERSAL", "EARLY": "EARLY",
    "PULLBACK": "PULLBACK", "SQUEEZE": "SQUEEZE", "MEANREV": "MEANREV",
    "PARTIAL": "PARTIAL",
    "DIP+": "DIP+", "DIP": "DIP", "DIP_E": "DIP_E", "DIP_W": "DIP_W",
    "RECOVER": "RECOVER", "INSTITUTIONAL": "INSTITUTIONAL",
    "SIDEWAYS_MR": "SIDEWAYS_MR", "SIDEWAYS_SQ": "SIDEWAYS_SQ",
}

SIGNAL_EMOJI = {
    "COMBO+": "🔴", "COMBO": "🟠", "STRONG": "🟢", "WEAK": "🟡",
    "REVERSAL": "🟣", "EARLY": "🟠", "PULLBACK": "🔵",
    "SQUEEZE": "🟩", "MEANREV": "⚪", "PARTIAL": "⬜",
    "DIP+": "🔶", "DIP": "🔸", "DIP_E": "📙", "DIP_W": "👀",
    "RECOVER": "💎", "INSTITUTIONAL": "🏦", "WHALE": "🐋",
    "SUPPLY": "📦",
    "SIDEWAYS_MR": "🔷", "SIDEWAYS_SQ": "🔶",
}

SIGNAL_COLORS = {
    "COMBO+": "#ef4444", "COMBO": "#f97316", "STRONG": "#22c55e",
    "WEAK": "#eab308", "REVERSAL": "#a855f7", "EARLY": "#f97316",
    "PULLBACK": "#3b82f6", "SQUEEZE": "#4ade80", "MEANREV": "#d1d5db",
    "PARTIAL": "#9ca3af",
    "DIP+": "#f59e0b", "DIP": "#fb923c", "DIP_E": "#fbbf24",
    "DIP_W": "#a3a3a3", "RECOVER": "#06b6d4", "INSTITUTIONAL": "#8b5cf6",
    "WHALE": "#0ea5e9",
    "SUPPLY": "#f59e0b",
    "SIDEWAYS_MR": "#60a5fa", "SIDEWAYS_SQ": "#f59e0b",
}

SIGNAL_PRIORITY_TREND = {
    "COMBO+": 1, "COMBO": 2, "STRONG": 3, "WEAK": 4,
    "REVERSAL": 5, "EARLY": 6, "PULLBACK": 7,
    "SQUEEZE": 8, "MEANREV": 9, "PARTIAL": 10,
}

SIGNAL_PRIORITY_DIP = {
    "DIP+": 1, "DIP": 2, "DIP_E": 3, "RECOVER": 4,
    "INSTITUTIONAL": 4, "WHALE": 4, "SUPPLY": 4, "DIP_W": 5,
}

SIGNAL_PRIORITY_SIDEWAYS = {"SIDEWAYS_SQ": 1, "SIDEWAYS_MR": 2}

# ── REGIME CLASSIFICATION ──
ADX_TREND = 20
ADX_CHOPPY = 15
REGIME_COLORS = {
    "FULL_TREND": "#22c55e", "TREND": "#3b82f6",
    "GRI_BOLGE": "#d1d5db", "CHOPPY": "#ef4444",
}
REGIME_SHORT = {
    "FULL_TREND": "FT", "TREND": "TR",
    "GRI_BOLGE": "GR", "CHOPPY": "CH",
}

# ── MARKET REGISTRY ──
# Her market kendi config override'larını sağlar.
# Market-specific config'ler markets/<name>/config.py'de.

MARKETS = {
    'bist': {
        'name': 'BIST',
        'benchmark': 'XU100.IS',
        'suffix': '.IS',
        'currency': 'TRY',
        'tv_prefix': 'BIST',
        'min_avg_volume': 5_000_000,  # 5M TL
        'needs_usd': True,  # RECOVER modülü USD dönüşümü gerektirir
    },
    'us': {
        'name': 'US',
        'benchmark': 'SPY',
        'suffix': '',
        'currency': 'USD',
        'tv_prefix': '',  # yfinance ticker'ı direkt kullanılır
        'min_avg_volume': 1_000_000,  # 1M USD
        'needs_usd': False,
    },
    'crypto': {
        'name': 'CRYPTO',
        'benchmark': 'BTC-USD',
        'suffix': '',
        'currency': 'USD',
        'tv_prefix': '',
        'min_avg_volume': 5_000_000,
        'needs_usd': False,
    },
    'commodity': {
        'name': 'COMMODITY',
        'benchmark': 'DBC',
        'suffix': '',
        'currency': 'USD',
        'tv_prefix': '',
        'min_avg_volume': 500_000,
        'needs_usd': False,
    },
}


def get_market_config(market_name):
    """Market-specific config'i döndür."""
    base = MARKETS.get(market_name, MARKETS['bist'])
    return base


# ── INDICATOR-SPECIFIC PARAMS (shared) ──
WT_CH_LEN = 10
WT_AVG_LEN = 21
WT_MA_LEN = 4
WT_LOOKBACK = 3
PMAX_ATR_LEN = 10
PMAX_ATR_MULT = 3.0
PMAX_MA_LEN = 10
PMAX_MA_TYPE = "EMA"
SMC_INTERNAL_LEN = 5
MR_TIME_BASE = 5
MIN_DATA_DAYS = 80
OVEREXT_WT1_THRESH = 40
OVEREXT_RSI_THRESH = 70
OVEREXT_MOMENTUM_PCT = 8.0   # son 5 günde %8+ yükseliş
OVEREXT_MOMENTUM_DAYS = 5
