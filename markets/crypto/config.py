"""
NOX Project — Crypto Market Config
Kripto piyasasına özel parametreler.
24/7 market, yüksek volatilite, farklı dinamikler.
"""

# ── CRYPTO-SPECIFIC ──
MIN_AVG_VOLUME_USD = 5_000_000  # min 5M USD günlük hacim
RVOL_THRESH = 0.5

# Stop / TP multipliers (kripto daha volatil → daha geniş)
TREND_STOP = 2.0
GRI_STOP = 1.5
MR_STOP = 1.2
DONUS_STOP = 2.0
COMBO_STOP = 1.5
TREND_TP = 2.0
GRI_TP = 2.0
DONUS_TP = 3.0
COMBO_TP = 2.0

# Quality thresholds
QUAL_MIN_GRI = 20
QUAL_MIN_TREND = 15
RS_THRESHOLD = 5.0

# DIP specific
PINK_EMA89 = 89
PINK_EMA144 = 144
PINK_RSI_LEN = 14
PINK_RSI_DIV_LOOKBACK = 10
PINK_TOUCH_WINDOW = 80   # kripto daha hızlı döngüler
PINK_TOUCH_COUNT = 12
PINK_STOP_MULT = 2.0     # daha geniş stop

# ── ON-CHAIN / WHALE SIGNAL ──
# Funding rate + open interest + whale accumulation proxy
WHALE_MIN_SCORE = 35
WHALE_OI_WINDOW = 14      # open interest değişim penceresi
WHALE_VOL_SPIKE = 2.0     # volume spike threshold (kripto daha volatil)
WHALE_ACCUM_WINDOW = 14   # accumulation tespiti penceresi
