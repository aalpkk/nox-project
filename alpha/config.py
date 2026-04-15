"""
Alpha Pipeline — Konfigürasyon
===============================
Tüm parametreler tek dosyada.
"""

# ── ML ENTEGRASYONU (Stage 1-2 yerine) ─────────────────────
ML_STAGE1_ENABLED = True         # True = ML ile momentum, False = klasik WT/SBT/ST
ML_SCORE_THRESHOLD = 0.48        # Minimum ML skor (universe_1g) — aday olmak için
ML_SWING_THRESHOLD = 0.45        # Minimum swing skor (universe_3g)
ML_SLOPE_LOOKBACK = 5            # ML skor eğim penceresi (bar)
ML_SLOPE_MIN = 0.0               # Minimum ML skor eğimi (>= 0 = iyileşiyor)
ML_COMPOSITE_WEIGHT = 0.6        # ML skor'un composite'deki ağırlığı (0-1)

# ── AŞAMA 1: Momentum Yakalama (klasik — ML_STAGE1_ENABLED=False ise) ──
WT_CROSS_LOOKBACK = 8           # WaveTrend cross-up son N bar (was 5)
WT1_ZONE_LO = -40               # WT1 giriş zonu alt sınır (was -30)
WT1_ZONE_HI = 50                # WT1 giriş zonu üst sınır (was 30)
SBT_SQUEEZE_COMPLEMENT = True   # Squeeze breakout da sinyal olarak kullan
ST_FLIP_LOOKBACK = 3            # SuperTrend flip son N bar

# Squeeze parametreleri (SBT'den alınmış, değiştirilmemiş)
BB_LENGTH = 20
BB_MULT = 2.0
BB_WIDTH_THRESH = 0.80
ATR_LENGTH = 10
ATR_SMA_LENGTH = 20
ATR_SQUEEZE_RATIO = 1.00
MIN_SQUEEZE_BARS = 5
MAX_SQUEEZE_BARS = 40
IMPULSE_ATR_MULT = 0.35
VOL_SMA_LENGTH = 20
VOL_MULT = 1.5

# ── AŞAMA 2: Eğim Doğrulama ────────────────────────────────
SLOPE_EMA_FAST = 20             # Fiyat eğimi EMA periyodu
SLOPE_EMA_SLOW = 50             # Trend hizası EMA periyodu
SLOPE_WINDOW = 5                # Eğim ölçüm penceresi (bar)
SLOPE_MIN_THRESHOLD = 0.0       # Minimum eğim (> 0 yeterli)
SLOPE_MIN_CHECKS = 2            # 3 eğim kontrolünden minimum kaçı geçmeli (2/3 = gevşek)
SIGNAL_SLOPE_WINDOW = 3         # WT1 eğim penceresi (bar)

# ── AŞAMA 3: Teknik Onay ───────────────────────────────────
ADX_MIN = 20                    # Minimum ADX (trend gücü)
CMF_MIN = 0.0                   # CMF > 0 = birikim
RSI_LO = 40                     # RSI alt sınır (momentum sweet spot)
RSI_HI = 70                     # RSI üst sınır
CANDLE_LOOKBACK = 3             # Mum formasyonu arama penceresi
CONFIRMATION_MIN_SCORE = 2      # Minimum onay skoru (0-4) — en az 2/4 teknik onay gerekli

# ── AŞAMA 4: Portföy Optimizasyonu ─────────────────────────
COV_LOOKBACK_DAYS = 120         # Kovaryans penceresi (~6 ay)
MIN_STOCKS = 3                  # Minimum portföy büyüklüğü
MAX_STOCKS = 12                 # Maksimum portföy büyüklüğü
WEIGHT_MIN = 0.02               # %2 minimum ağırlık
WEIGHT_MAX = 0.20               # %20 maksimum ağırlık
RISK_FREE_RATE = 0.50           # %50 yıllık (Türk T-bill)
PORTFOLIO_METHOD = "markowitz"  # "equal" | "markowitz"

# ── AŞAMA 5: Seçim ────────────────────────────────────────
TARGET_PORTFOLIO_SIZE = 8       # Hedef portföy büyüklüğü
SCORE_TILT_FACTOR = 0.05        # Sinyal skoru → μ tilt katsayısı

# ── GÖRECELİ GÜÇ ─────────────────────────────────────────
RS_LOOKBACK = 60                # Göreceli güç penceresi (gün)
RS_MIN_OUTPERFORM = 0.0         # Minimum outperformance (%) — 0 = XU100'ü geçmesi yeterli
MOM_LOOKBACK = 63               # 3 aylık momentum penceresi (composite score'a eklenir)
REBALANCE_FREQ = "biweekly"     # "weekly" | "biweekly" | "monthly"

# ── WALK-FORWARD BACKTEST ──────────────────────────────────
WF_TRAIN_DAYS = 252             # Eğitim penceresi (1 yıl)
WF_STEP_DAYS = 10               # Rebalance adımı (iş günü)
INITIAL_CAPITAL = 1_000_000     # Başlangıç sermayesi (TL)
COMMISSION_PCT = 0.002          # %0.2 komisyon (tek taraf)
SLIPPAGE_PCT = 0.001            # %0.1 slippage

# ── STOP LOSS ──────────────────────────────────────────────
POSITION_STOP_PCT = 0           # 0 = devre dışı
POSITION_STOP_ATR_MULT = 2.0   # Emergency stop: entry - 2×ATR
PORTFOLIO_DD_LIMIT = -20.0      # Portföy %20 DD'de tüm pozisyonlar kapatılır

# ── BATCH MODU ────────────────────────────────────────────
BATCH_MODE = False              # False = klasik rebalance modu
BATCH_REOPEN_THRESHOLD = 0      # 0 = tüm pozisyonlar kapanınca yeni batch aç
MAX_HOLD_DAYS = 40              # Sinyal yoksa max tutma süresi (backtest'te uygulanır)

# ── TRAILING STOP ─────────────────────────────────────────
TRAILING_TRIGGER_PCT = 0        # 0 = devre dışı
TRAILING_STOP_PCT = 0           # 0 = devre dışı
TRAILING_TRIGGER_ATR = 1.5      # Trailing başlama: +1.5 ATR kâr (dinamik)
TRAILING_ATR_MULT = 1.5         # Trail: zirve - 1.5 ATR

# ── REJİM FİLTRESİ ───────────────────────────────────────
REGIME_EMA_LENGTH = 21          # XU100 trend EMA periyodu
REGIME_BULL_WEIGHT = 1.0        # Bull rejimde tam ağırlık
REGIME_BEAR_WEIGHT = 0.5        # Bear rejimde yarı ağırlık

# ── UZAMA FİLTRESİ ────────────────────────────────────────
MAX_RALLY_PCT = 0               # 0 = devre dışı, >0 = son N günde %X+ çıkanı ele
MAX_RALLY_DAYS = 40             # Uzama ölçüm penceresi (iş günü)

# ── VERİ ───────────────────────────────────────────────────
DATA_PERIOD = "2y"              # Backtest veri penceresi
MIN_VOLUME_TL = 5_000_000      # Minimum ortalama günlük hacim (TL)
MIN_DATA_DAYS = 80              # Minimum veri uzunluğu (bar)
