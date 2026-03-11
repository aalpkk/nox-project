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

# ── CATALYST SCREENER ──
# Unusual Volume
RVOL_UNUSUAL = 2.0          # Unusual volume eşiği (volume / SMA20)
RVOL_HIGH = 3.0             # Çok yüksek hacim
MIN_PRICE_USD = 5.0         # Penny stock filtresi
MIN_CHANGE_PCT = 3.0        # Min fiyat değişimi %
VOL_SMA_PERIOD = 20         # Hacim ortalaması periyodu

# Short Squeeze
SHORT_FLOAT_MIN = 15.0      # Min short % of float
FLOAT_SHARES_MAX = 100e6    # Max float shares
DAYS_TO_COVER_MIN = 3.0     # Min days to cover

# Insider Buying
INSIDER_LOOKBACK = 60       # Gün — geriye bakış penceresi
INSIDER_MIN_BUYERS = 2      # Min alıcı sayısı
INSIDER_MIN_VALUE = 100_000  # Min toplam alım ($)

# Biotech Catalyst
BIOTECH_MCAP_MAX = 10e9     # Max market cap ($10B)

# Earnings Momentum
EARNINGS_WINDOW = 14        # Bilanço yakınlık penceresi (gün)
EARNINGS_BB_WIDTH_MAX = 10  # Max BB width % (sıkışma tespiti)
EARNINGS_GAP_MIN = 5.0      # Post-earnings min gap %

# Technical Breakout
ATR_COMPRESS_RATIO = 0.6    # ATR < 60% of 60-day avg = sıkışma
CONSOL_MIN_DAYS = 15        # Min konsolidasyon süresi
BREAKOUT_VOL_MULT = 2.0     # Breakout'ta min hacim çarpanı

# Accumulation (sessiz birikim)
ACCUM_RVOL_MIN = 1.3        # min ortalama RVOL (sessiz ama yükselen)
ACCUM_RVOL_MAX = 2.5        # max — bunun üstü zaten VOLUME modülüne düşer
ACCUM_WINDOW = 10            # birikim penceresi (gün)
ACCUM_MAX_DAILY_MOVE = 4.0   # max tek gün hareket % (büyükse reaktif)
ACCUM_RANGE_ATR_MULT = 2.0   # range < N*ATR = sıkışma

# ── SPY REGIME ──
REGIME_BULL_THRESH = 5       # >= 5/6 = BULL
REGIME_NEUTRAL_THRESH = 3    # >= 3/6 = NEUTRAL, < 3 = RISK_OFF

# ── REGIME TRANSITION (per-stock) ──
# BIST RT_CFG ile aynı başlangıç değerleri — sonra US'e özel tune edilir
RT_CFG_US = {
    'ema_fast': 21, 'ema_slow': 55,
    'st_period': 10, 'st_mult': 3.0,
    'weekly_ema_len': 21,
    'cmf_period': 20, 'rvol_period': 20,
    'obv_ema_len': 10, 'obv_slope_len': 5,
    'adx_len': 14, 'adx_slope_len': 5,
    'atr_len': 14, 'atr_sma_len': 20,
    'atr_expand_mult': 1.05, 'di_spread_thresh': 5,
    'exit_ema_len': 21, 'exit_close_below_bars': 2, 'exit_adx_slope_bars': 3,
    'regime_lookback': 5,
    'stop_swing_lb': 10, 'stop_atr_initial': 0.5, 'stop_atr_trail': 2.0,
    'oe_rsi_period': 14, 'oe_rsi_thresh': 80,
    'oe_bb_period': 20, 'oe_bb_mult': 2.0,
    'oe_momentum_bars': 5, 'oe_momentum_thresh': 8, 'oe_ema_dist_thresh': 5,
    'entry_atr_pct_thresh': 3.0,
}
