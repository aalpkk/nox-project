"""
NOX Project — BIST Market Config
BIST'e özel parametreler ve override'lar.
"""

# ── BIST-SPECIFIC ──
MIN_AVG_VOLUME_TL = 5_000_000
RVOL_THRESH = 0.5

# Stop / TP multipliers
TREND_STOP = 2.0       # was 1.5 — daha geniş stop, erken çıkışı azalt
GRI_STOP = 1.2
MR_STOP = 1.0          # legacy — MEANREV devre dışı ama regime.py import eder
DONUS_STOP = 1.3       # was 1.8 — REVERSAL'da daha sıkı stop
COMBO_STOP = 1.2       # was 1.0
TREND_TP = 5.0         # was 1.5 — trendi koştur, asimetrik R:R
GRI_TP = 2.5           # was 1.5 — pullback'te de daha geniş TP
DONUS_TP = 2.5         # was 2.0
COMBO_TP = 2.5         # was 2.0 — rr20 uyumu: 2.5/1.2 = 2.08 R:R

# Trailing stop multiplier (highest close - trail_mult * ATR)
TRAIL_MULT = 2.0

# Quality thresholds
QUAL_MIN_GRI = 60
QUAL_MIN_TREND = 75
RS_THRESHOLD = 5.0

# ── PRODUCTION RULESET V1 ──

# Panic block: ATR percentile threshold
ATR_PANIC_PCTILE = 0.85
ATR_PCTILE_WINDOW = 100

# Quality gate thresholds (per signal type)
CORE_Q_MIN_PARTIAL_STRONG = 40
CORE_Q_MIN_EARLY = 50
CORE_Q_MIN_COMBO = 50

# COMBO special filters
COMBO_RS_MAX = 10
COMBO_OE_MAX = 2
COMBO_DIST_EMA_MAX = 1.5

# Position sizing (by trade mode and signal)
POS_SIZE_CORE = {"STRONG": 1.0, "PARTIAL": 1.0, "EARLY": 0.7, "COMBO": 0.6}
POS_SIZE_MOMENTUM = {"STRONG": 0.5, "PARTIAL": 1.0}

# Risk-on detection
RISK_ON_EMA_LEN = 50
MOMENTUM_RS_THRESH = 20

# Allowed signal sets per mode
CORE_SIGNALS = {"STRONG", "PARTIAL", "EARLY", "COMBO"}
MOMENTUM_SIGNALS = {"STRONG", "PARTIAL"}

# Pink V2 (DIP) specific
PINK_EMA89 = 89
PINK_EMA144 = 144
PINK_RSI_LEN = 14
PINK_RSI_DIV_LOOKBACK = 10
PINK_TOUCH_WINDOW = 100
PINK_TOUCH_COUNT = 15
PINK_STOP_MULT = 1.5
