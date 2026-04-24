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
RSI_HI = 60                     # RSI üst sınır
CANDLE_LOOKBACK = 3             # Mum formasyonu arama penceresi
CONFIRMATION_MIN_SCORE = 2      # Minimum onay skoru (0-4) — en az 2/4 teknik onay gerekli

# ── AŞAMA 4: Portföy Optimizasyonu ─────────────────────────
COV_LOOKBACK_DAYS = 120         # Kovaryans penceresi (~6 ay)
MIN_STOCKS = 3                  # Minimum portföy büyüklüğü
MAX_STOCKS = 12                 # Maksimum portföy büyüklüğü
WEIGHT_MIN = 0.02               # %2 minimum ağırlık
WEIGHT_MAX = 0.2               # %20 maksimum ağırlık
RISK_FREE_RATE = 0.50           # %50 yıllık (Türk T-bill)
PORTFOLIO_METHOD = "score"  # "equal" | "markowitz" | "score"
SCORE_POWER = 2.0               # Conviction tilt gücü (score ** power)
VOL_ADJUSTED_SIZING = False     # True = ağırlık ∝ score^power / ATR% (volatil hisse küçük)
VOL_ADJUST_FLOOR_PCT = 1.0      # Min ATR% (div-by-zero koruması, çok likit hisseler için)

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
COMMISSION_PCT = 0.0            # 0 komisyon (kullanıcı broker: 0)
SLIPPAGE_PCT = 0.001            # %0.1 slippage

# ── EXECUTION (gerçekçi giriş kuralı) ──────────────────────
EXECUTION_MODE = "close"        # "close" (eski/unrealistic) | "next_open" (T+1 açılış, gap-skip) | "limit" (T+1: close×(1+limit_pct) limit order)
GAP_SKIP_PCT = 0.07             # next_open modu: T+1 open/T close - 1 >= bu ise skip (chasing filter)
LIMIT_ENTRY_PCT = 0.07          # limit modu: limit price = T close × (1 + bu). T+1 Low <= limit ve Open < limit ise fill.

# ── STOP LOSS ──────────────────────────────────────────────
POSITION_STOP_PCT = 0           # 0 = devre dışı
POSITION_STOP_ATR_MULT = 2.0   # Emergency stop: entry - 2×ATR (normal-vol)
DYNAMIC_STOP_ENABLED = False    # True = yüksek ATR% hissede daha sıkı stop
DYNAMIC_STOP_ATR_PCT_HI = 4.0   # Bu %'nin üstü "yüksek vol" sayılır
DYNAMIC_STOP_MULT_HI = 1.5      # Yüksek vol hisse: entry - 1.5×ATR (daha sıkı)

# ── BREAKEVEN SHIFT ───────────────────────────────────────
# +N·R kâra ulaşınca emergency stop entry'ye çekilir (zarar → 0/küçük).
# R = stop_mult × entry_atr (default 2×ATR). BE_SHIFT_R=1.0 → +1R (≈+2×ATR) trigger.
BE_SHIFT_ENABLED = True         # True = breakeven shift aktif
BE_SHIFT_R = 2.0                # +N·R noktasında stop'u entry'ye çek

# ── KORELASYON FİLTRESİ ─────────────────────────────────
CORR_FILTER_ENABLED = True      # True = portföyde yüksek korele çiftleri ele
CORR_MAX_PAIR = 0.4             # İki hisse arası max mutlak korelasyon
CORR_LOOKBACK_DAYS = 60         # Korelasyon ölçüm penceresi
PORTFOLIO_DD_LIMIT = -99.0      # Devre dışı (peak-to-trough DD stop)
DAILY_LOSS_LIMIT = -5.0         # Tek gün kaybı bu %'ye ulaşırsa tüm pozisyonları kapat (-99 = devre dışı)

# ── STRESS-CORR (XU100 red-days'de conditional corr) ─────
STRESS_CORR_ENABLED = True      # True = aynı lookback'te red-day conditional corr filtresi
STRESS_CORR_MAX_PAIR = 0.4      # Red-day conditional pair corr tavanı (noise'lu, daha gevşek)
STRESS_CORR_XU_THRESHOLD = 0.0  # XU100 günlük getirisi bu altındaysa "red day"
STRESS_CORR_MIN_DAYS = 12       # Minimum red-day örnek sayısı (altındaysa filtre pas geçilir)

# ── SEKTÖR CAP ────────────────────────────────────────────
SECTOR_CAP_ENABLED = True       # Sektör yoğunluğunu sınırla
SECTOR_MAX_WEIGHT = 0.25        # Tek sektör toplam ağırlık tavanı
SECTOR_MAX_NAMES = 2            # Aynı sektörden max isim sayısı

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
REGIME_BEAR_WEIGHT = 1.0        # Bear rejimde yarı ağırlık (0 = tüm pozisyon kapat)

# ── KAMA REJİM FİLTRESİ (weekly, Kaufman Adaptive MA) ────
REGIME_KAMA_ENABLED = False     # True = EMA yerine weekly KAMA kullan
KAMA_ER_PERIOD = 5              # Efficiency Ratio periyodu (hafta)
KAMA_FAST = 2                   # Fast EMA periyodu
KAMA_SLOW = 80                  # Slow EMA periyodu

# ── ADAPTIVE SUPERTREND (K-means vol clustering) ────────
REGIME_AST_ENABLED = False      # True = AlgoAlpha K-means Adaptive SuperTrend
AST_ATR_LEN = 10                # ATR periyodu
AST_FACTOR = 2.0                # SuperTrend band çarpanı
AST_TRAINING = 100              # K-means eğitim penceresi (bar)
AST_HI_INIT = 0.75              # İlk high-vol percentile
AST_MID_INIT = 0.50             # İlk mid-vol percentile
AST_LO_INIT = 0.25              # İlk low-vol percentile

# ── UZAMA FİLTRESİ ────────────────────────────────────────
MAX_RALLY_PCT = 0               # 0 = devre dışı, >0 = son N günde %X+ çıkanı ele
MAX_RALLY_DAYS = 40             # Uzama ölçüm penceresi (iş günü)
EMA50_DIST_MAX = 0              # 0 = devre dışı, >1 = close/EMA50 tavanı (1.15 = %15 üstü max)

# ── EXTENSION FILTER (stop-risk veto) ─────────────────────
# K=2/4 composite trigger: STOP-prone (overbought/stretched) aday vetolanır.
# Kaynak: 207-position EDA (2026-04-23), train/test split +7pp WR lift.
EXT_FILTER_ENABLED = False      # True = filtre aktif
EXT_FILTER_MIN_TRIGGERS = 2     # >= K trigger → veto (K∈{1,2,3,4})
EXT_20D_HIGH_PCT = -3.0         # close_vs_20d_high_pct > bu → trigger (zirveye yakın)
EXT_EMA21_DIST_PCT = 10.0       # ema21_dist_pct > bu → trigger (stretched)
EXT_MFI_14 = 75                 # mfi_14 > bu → trigger (overbought)
EXT_BB_PCTB = 0.85              # bb_pctb > bu → trigger (BB üst bandına yakın)

# ── VERİ ───────────────────────────────────────────────────
DATA_PERIOD = "3y"              # Backtest veri penceresi
MIN_VOLUME_TL = 5_000_000      # Minimum ortalama günlük hacim (TL)
MIN_DATA_DAYS = 80              # Minimum veri uzunluğu (bar)
