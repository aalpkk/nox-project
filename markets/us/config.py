"""
NOX Project — US Market Config
ABD borsalarına özel parametre override'ları.
US hisseleri genelde daha likit, spread dar, volatilite farklı.
"""

# ── US-SPECIFIC ──
MIN_AVG_VOLUME_USD = 1_000_000  # min 1M USD günlük hacim
RVOL_THRESH = 0.5

# Stop / TP multipliers (US daha dar ATR'lı, daha sıkı)
TREND_STOP = 1.5
GRI_STOP = 1.2
MR_STOP = 1.0
DONUS_STOP = 1.5
COMBO_STOP = 1.0
TREND_TP = 1.5
GRI_TP = 1.5
DONUS_TP = 2.0
COMBO_TP = 1.5

# Quality thresholds
QUAL_MIN_GRI = 20
QUAL_MIN_TREND = 15
RS_THRESHOLD = 5.0

# Pink V2 (DIP) specific — US için aynı EMA yapısı
PINK_EMA89 = 89
PINK_EMA144 = 144
PINK_RSI_LEN = 14
PINK_RSI_DIV_LOOKBACK = 10
PINK_TOUCH_WINDOW = 100
PINK_TOUCH_COUNT = 15
PINK_STOP_MULT = 1.5

# ── INSTITUTIONAL SIGNAL (RECOVER yerine) ──
# Earnings momentum + dark pool + institutional accumulation
INST_MIN_SCORE = 40       # minimum score for INSTITUTIONAL signal
INST_ACCUM_WINDOW = 20    # gün — accumulation tespiti penceresi
INST_VOL_SPIKE = 1.5      # volume spike threshold
INST_PRICE_RANGE = 0.05   # %5 dar range = accumulation
