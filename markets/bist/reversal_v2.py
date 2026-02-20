"""
NOX Reversal Screener v2 - Geliştirilmiş Rejim Dönüşü Tespit Sistemi
======================================================================
v1 → v2 Değişiklikler:
  1. Sequential Gating: Volume Climax → Breadth → Regime → Entry sıralaması
  2. Market Gate: Breadth/Regime gate olmadan DIP skoru %70 kırpılır
  3. Structure Break Modülü: Swing high/low + higher high/higher low tespiti
  4. Makro vs Swing ayrımı: İki ayrı çıktı katmanı
  5. ATR/Volatilite normalizasyonu: DI farkı, decline ölçümleri normalize
  6. Percentile-bazlı eşikler: Sabit threshold yerine hisse bazlı adaptif
  7. Coverage cezası: Breadth'te düşük coverage günleri cezalandırılır
  8. Signal Study Framework: Forward return + MAE/MFE analizi

NOT: Bu modül lowercase kolon isimleri kullanır (close, high, low, open, volume).
     Runner script (run_reversal.py) uppercase→lowercase dönüşümünü yapar.

Kullanım:
    screener = ReversalScreenerV2()

    # Makro rejim sorgusu
    macro = screener.macro_regime(bist100_df, stock_dfs)

    # Swing giriş taraması (makro gate ile)
    entries = screener.swing_scan(stock_dfs, bist100_df, macro)

    # Signal study (backtest)
    study = screener.signal_study(stock_dfs, bist100_df, forward_days=[1, 5, 10])
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class MacroPhase(Enum):
    """Makro piyasa fazları - sıralı geçiş"""
    DOWNTREND = "DOWNTREND"
    CAPITULATION = "CAPITULATION"          # Volume climax tespit
    BOTTOMING = "BOTTOMING"                # Breadth toparlanma başlıyor
    REGIME_SHIFT = "REGIME_SHIFT"          # ADX + DI konfirme
    EARLY_UPTREND = "EARLY_UPTREND"        # Structure break
    UPTREND = "UPTREND"                    # Yerleşik boğa
    DISTRIBUTION = "DISTRIBUTION"          # Tepe bölgesi


class SwingState(Enum):
    """Bireysel hisse swing durumu"""
    NO_SETUP = "NO_SETUP"
    DIP_CANDIDATE = "DIP_CANDIDATE"        # Oversold ama gate kapalı
    GATED_READY = "GATED_READY"            # Makro gate açık, dip uygun
    BREAKOUT_PENDING = "BREAKOUT_PENDING"  # Structure break bekleniyor
    ENTRY_SIGNAL = "ENTRY_SIGNAL"          # Tüm koşullar tamam


class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4


@dataclass
class MacroContext:
    """Piyasa geneli makro durum - gate olarak kullanılır"""
    date: pd.Timestamp
    phase: MacroPhase
    phase_score: float              # 0-100
    breadth_score: float            # 0-100
    regime_score: float             # 0-100
    volume_climax_score: float      # 0-100
    structure_score: float          # 0-100
    gate_open: bool                 # Swing tarama için gate açık mı?
    gate_strength: float            # 0-1, DIP skoruna çarpan
    details: Dict = field(default_factory=dict)

    def __repr__(self):
        gate_str = f"AÇIK ({self.gate_strength:.0%})" if self.gate_open else "KAPALI"
        return (f"{'='*60}\n"
                f"  MAKRO REJİM | {self.date.strftime('%Y-%m-%d')}\n"
                f"  Faz: {self.phase.value} | Score: {self.phase_score:.0f}/100\n"
                f"  Gate: {gate_str}\n"
                f"  Breadth: {self.breadth_score:.0f} | Regime: {self.regime_score:.0f} | "
                f"VolClimax: {self.volume_climax_score:.0f} | Structure: {self.structure_score:.0f}\n"
                f"{'='*60}")


@dataclass
class SwingEntry:
    """Bireysel hisse swing giriş sinyali"""
    ticker: str
    date: pd.Timestamp
    state: SwingState
    raw_score: float               # Gate öncesi ham skor
    gated_score: float             # Gate sonrası efektif skor
    rs_score: float                # Relative strength
    structure_break: bool
    details: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"  {self.ticker:<8} | {self.state.value:<20} | "
                f"Raw: {self.raw_score:>5.1f} → Gated: {self.gated_score:>5.1f} | "
                f"RS: {self.rs_score:>+6.2f} | StructBreak: {'✓' if self.structure_break else '✗'}")


@dataclass
class SignalStudyRow:
    """Signal study tek kayıt"""
    ticker: str
    signal_date: pd.Timestamp
    signal_type: str
    signal_score: float
    entry_price: float
    fwd_return_1d: Optional[float] = None
    fwd_return_5d: Optional[float] = None
    fwd_return_10d: Optional[float] = None
    mae_5d: Optional[float] = None        # Max Adverse Excursion (max drawdown)
    mfe_5d: Optional[float] = None        # Max Favorable Excursion (max gain)
    mae_10d: Optional[float] = None
    mfe_10d: Optional[float] = None


# =============================================================================
# VECTORIZED INDICATORS (ATR-normalized versions)
# =============================================================================

class Indicators:
    """Vectorized teknik gösterge hesaplamaları - ATR normalizasyon dahil"""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX, +DI, -DI"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.ewm(alpha=1/period, min_periods=period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index
        ).ewm(alpha=1/period, min_periods=period).mean()
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index
        ).ewm(alpha=1/period, min_periods=period).mean()

        plus_di = 100 * plus_dm / atr_val.replace(0, np.nan)
        minus_di = 100 * minus_dm / atr_val.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx_val, plus_di, minus_di

    @staticmethod
    def di_diff_normalized(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """DI farkı ATR-normalize edilmiş (v2 iyileştirme)"""
        adx_val, plus_di, minus_di = Indicators.adx(high, low, close, period)
        atr_val = Indicators.atr(high, low, close, period)
        atr_pct = atr_val / close * 100
        di_diff = plus_di - minus_di
        # Yüksek volatilite → DI farkı daha az anlamlı → normalize
        return di_diff / (1 + atr_pct * 0.1)

    @staticmethod
    def decline_atr_normalized(close: pd.Series, high: pd.Series, low: pd.Series,
                                lookback: int = 5, atr_period: int = 14) -> pd.Series:
        """Düşüş miktarını ATR cinsinden ölç (v2 iyileştirme)"""
        atr_val = Indicators.atr(high, low, close, atr_period)
        raw_decline = close - close.rolling(lookback).max()
        return raw_decline / atr_val.replace(0, np.nan)

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        bandwidth = (upper - lower) / middle * 100
        return upper, middle, lower, bandwidth

    @staticmethod
    def percent_b(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        upper, middle, lower, _ = Indicators.bollinger_bands(close, period, std_dev)
        return (close - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def wavetrend(hlc3: pd.Series, channel_len: int = 10,
                  avg_len: int = 21) -> Tuple[pd.Series, pd.Series]:
        esa = hlc3.ewm(span=channel_len, adjust=False).mean()
        d = (hlc3 - esa).abs().ewm(span=channel_len, adjust=False).mean()
        ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
        wt1 = ci.ewm(span=avg_len, adjust=False).mean()
        wt2 = wt1.rolling(window=4).mean()
        return wt1, wt2

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        tp = (high + low + close) / 3
        rmf = tp * volume
        delta = tp.diff()
        pos = rmf.where(delta > 0, 0.0).rolling(period).sum()
        neg = rmf.where(delta <= 0, 0.0).rolling(period).sum()
        return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

    @staticmethod
    def volume_percentile(volume: pd.Series, lookback: int = 60) -> pd.Series:
        """Hacmi son N gün içindeki persentil olarak döndür (v2 iyileştirme)"""
        def _pct(window):
            if len(window) < 2:
                return 50.0
            current = window.iloc[-1]
            return (window.iloc[:-1] < current).mean() * 100
        return volume.rolling(window=lookback, min_periods=20).apply(_pct, raw=False)

    @staticmethod
    def swing_highs_lows(high: pd.Series, low: pd.Series,
                          lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Swing high ve swing low serileri (v2 - Structure Break için)"""
        swing_high = high.rolling(window=2*lookback+1, center=True).max()
        swing_low = low.rolling(window=2*lookback+1, center=True).min()
        # Sadece gerçek pivot noktaları
        is_swing_high = (high == swing_high)
        is_swing_low = (low == swing_low)
        return is_swing_high, is_swing_low

    @staticmethod
    def relative_strength(close: pd.Series, benchmark: pd.Series,
                          period: int = 20) -> pd.Series:
        """RS = hisse performansı / endeks performansı"""
        stock_ret = close.pct_change(period)
        bench_ret = benchmark.pct_change(period)
        return stock_ret - bench_ret

    @staticmethod
    def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 20) -> pd.Series:
        """Rolling VWAP — günlük VWAP proxy (~1 ay maliyet ortalaması)"""
        typical_price = (high + low + close) / 3
        tp_vol = (typical_price * volume).rolling(period).sum()
        vol_sum = volume.rolling(period).sum()
        return tp_vol / vol_sum.replace(0, np.nan)

    @staticmethod
    def anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, anchor_mask: pd.Series) -> pd.Series:
        """
        Anchored VWAP — anchor_mask True olan son bardan itibaren cumulative VWAP.
        Her yeni anchor event'te sıfırlanıp yeniden başlar.
        """
        typical_price = (high + low + close) / 3
        tp_vol = typical_price * volume

        result = pd.Series(np.nan, index=close.index)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        active = False

        for i in range(len(close)):
            if anchor_mask.iloc[i]:
                cum_tp_vol = tp_vol.iloc[i]
                cum_vol = volume.iloc[i]
                active = True
            elif active:
                cum_tp_vol += tp_vol.iloc[i]
                cum_vol += volume.iloc[i]

            if active and cum_vol > 0:
                result.iloc[i] = cum_tp_vol / cum_vol

        return result

    @staticmethod
    def vwap_band(vwap: pd.Series, close: pd.Series, volume: pd.Series,
                  period: int = 20, mult: float = 1.0) -> Tuple[pd.Series, pd.Series]:
        """VWAP etrafında volume-weighted standart sapma bandı"""
        deviation = (close - vwap)
        vwap_std = deviation.rolling(period).std()
        upper = vwap + mult * vwap_std
        lower = vwap - mult * vwap_std
        return upper, lower


# =============================================================================
# MODÜL 0: REVERSAL CANDLE (Dönüş Mumu Tespiti)
# =============================================================================

class ReversalCandleModule:
    """
    Dönüş mumu tespiti + hacim teyidi.
    dip.py'deki _pink_reversal_candle ile uyumlu isimlendirme.
    """

    def __init__(self, vol_confirm_ratio: float = 1.2):
        self.vol_confirm_ratio = vol_confirm_ratio

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Returns: (score_series, candle_name_series)
        Sadece düşüş sonrası anlamlı — RSI < 45 veya close < EMA21 koşulunda aranır.
        """
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']

        body = (close - open_).abs()
        candle_range = (high - low).replace(0, np.nan)
        body_pct = body / candle_range

        upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

        is_green = close > open_
        is_red = close < open_

        vol_sma20 = Indicators.sma(volume, 20)
        vol_high = volume > (vol_sma20 * self.vol_confirm_ratio)

        rsi = Indicators.rsi(close, 14)
        ema21 = Indicators.ema(close, 21)
        dip_context = (rsi < 45) | (close < ema21)

        scores = pd.Series(0.0, index=df.index)
        names = pd.Series('', index=df.index)

        # --- CEKIC (Hammer) ---
        # Alt gölge >= 2x gövde, üst gölge < 0.5x gövde, body < %35 range
        cekic = (
            (lower_shadow >= 2 * body) &
            (upper_shadow < 0.5 * body) &
            (body_pct < 0.35) &
            dip_context
        )
        scores = scores.where(~cekic, 25.0)
        names = names.where(~cekic, 'CEKIC')

        # --- ENGULF (Bullish Engulfing) ---
        # Yeşil mum, önceki kırmızı mumu tamamen kapsar
        engulf = (
            is_green &
            is_red.shift(1) &
            (close > open_.shift(1)) &
            (open_ < close.shift(1)) &
            dip_context
        )
        # Engulf daha güçlü — override
        scores = scores.where(~engulf, 25.0)
        names = names.where(~engulf, 'ENGULF')

        # --- SABAH_YILDIZI (Morning Star) ---
        # 3-mum: büyük kırmızı → küçük gövde → büyük yeşil
        big_red_2ago = is_red.shift(2) & (body.shift(2) > candle_range.shift(2) * 0.5)
        small_body_1ago = body_pct.shift(1) < 0.30
        big_green_now = is_green & (body > candle_range * 0.5)
        sabah = (
            big_red_2ago &
            small_body_1ago &
            big_green_now &
            dip_context
        )
        # Sabah yıldızı — sadece CEKIC/ENGULF yoksa
        mask_sabah = sabah & (scores == 0)
        scores = scores.where(~mask_sabah, 20.0)
        names = names.where(~mask_sabah, 'SABAH_YILDIZI')

        # --- DOJI_YILDIZ (Doji Star) ---
        # Doji + önceki kırmızı + sonraki yeşil açılış (son bar doji ise bak)
        doji = body_pct < 0.10
        doji_star = (
            doji &
            is_red.shift(1) &
            dip_context
        )
        mask_doji = doji_star & (scores == 0)
        scores = scores.where(~mask_doji, 15.0)
        names = names.where(~mask_doji, 'DOJI_YILDIZ')

        # --- TOPAC (Spinning Top) ---
        # Küçük gövde, iki tarafta uzun gölge, dipte
        topac = (
            (body_pct < 0.30) &
            (lower_shadow > body) &
            (upper_shadow > body) &
            dip_context
        )
        mask_topac = topac & (scores == 0)
        scores = scores.where(~mask_topac, 10.0)
        names = names.where(~mask_topac, 'TOPAC')

        # --- Hacim Teyidi: hacim > SMA(20) × vol_confirm_ratio → puan 1.5x ---
        scores = scores * np.where(vol_high & (scores > 0), 1.5, 1.0)

        return scores, names


# =============================================================================
# MODÜL 1: VOLUME CLIMAX (Sequential Step 1)
# =============================================================================

class VolumeClimaxModule:
    """
    İlk tetikleyici: Kapitülasyon tespiti.
    Percentile-bazlı threshold kullanır (v2).
    """

    def __init__(self,
                 vol_percentile_threshold: float = 92.0,
                 body_atr_ratio: float = 1.0,
                 lookback_decline_atr: float = -2.0):
        self.vol_percentile_threshold = vol_percentile_threshold
        self.body_atr_ratio = body_atr_ratio
        self.lookback_decline_atr = lookback_decline_atr

    def score_and_detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Returns: (score_series 0-100, climax_mask boolean Series)
        climax_mask: True = bu bar bir volume climax event'i (anchored VWAP anchor noktası)
        """
        atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
        vol_pct = Indicators.volume_percentile(df['volume'], 60)
        decline_atr = Indicators.decline_atr_normalized(
            df['close'], df['high'], df['low'], lookback=5
        )

        body = (df['close'] - df['open']).abs()
        body_atr = body / atr.replace(0, np.nan)

        lower_shadow = pd.Series(
            np.where(df['close'] >= df['open'],
                     df['open'] - df['low'],
                     df['close'] - df['low']),
            index=df.index
        )
        candle_range = df['high'] - df['low']
        shadow_ratio = lower_shadow / candle_range.replace(0, np.nan)

        scores = pd.Series(0.0, index=df.index)

        # --- Selling Climax ---
        selling_climax = (
            (vol_pct > self.vol_percentile_threshold) &
            (body_atr > self.body_atr_ratio) &
            (df['close'] < df['open']) &
            (decline_atr < self.lookback_decline_atr)
        )
        scores = scores + selling_climax.astype(float) * 40

        # Bonus: alt gölge (alıcı absorpsiyonu)
        absorption = selling_climax & (shadow_ratio > 0.35)
        scores = scores + absorption.astype(float) * 15

        # --- Reversal Volume ---
        prev_red = (df['close'].shift(1) < df['open'].shift(1))
        prev_red2 = (df['close'].shift(2) < df['open'].shift(2))
        green_reversal = (
            (df['close'] > df['open']) &
            (vol_pct > 85) &
            prev_red & prev_red2 &
            (decline_atr.shift(1) < -1.5)
        )
        scores = scores + green_reversal.astype(float) * 30

        # --- Volume Dry-up ---
        vol_sma = Indicators.sma(df['volume'], 20)
        vol_ratio_now = df['volume'] / vol_sma.replace(0, np.nan)
        vol_ratio_prev = df['volume'].shift(5).rolling(5).mean() / vol_sma.shift(5).replace(0, np.nan)
        dryup = (
            (vol_ratio_now < 0.5) &
            (vol_ratio_prev > 1.2) &
            (decline_atr < -1.0)
        )
        scores = scores + dryup.astype(float) * 20

        # Climax mask: selling climax veya green reversal event'leri
        climax_mask = selling_climax | green_reversal

        return scores.clip(0, 100), climax_mask

    def score(self, df: pd.DataFrame) -> pd.Series:
        """0-100 volume climax skoru (backward compat)"""
        scores, _ = self.score_and_detect(df)
        return scores


# =============================================================================
# MODÜL 2: BREADTH THRUST (Sequential Step 2)
# =============================================================================

class BreadthModule:
    """
    Piyasa genişliği modülü.
    Coverage cezası: düşük veri coverage günlerini cezalandırır (v2).
    """

    def __init__(self,
                 ema_above_period: int = 21,
                 thrust_low: float = 0.25,
                 thrust_high: float = 0.55,
                 thrust_window: int = 10,
                 min_coverage: float = 0.70):
        self.ema_above_period = ema_above_period
        self.thrust_low = thrust_low
        self.thrust_high = thrust_high
        self.thrust_window = thrust_window
        self.min_coverage = min_coverage

    def compute(self, stock_dfs: Dict[str, pd.DataFrame],
                date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Breadth metrikleri hesapla + coverage tracking"""
        n_stocks = len(stock_dfs)

        above_ema21 = pd.DataFrame(index=date_index)
        above_ema50 = pd.DataFrame(index=date_index)
        advancing = pd.DataFrame(index=date_index)
        has_data = pd.DataFrame(index=date_index)

        for ticker, df in stock_dfs.items():
            if len(df) < 50:
                continue
            close = df['close'].reindex(date_index)
            ema21 = Indicators.ema(close, 21)
            ema50 = Indicators.ema(close, 50)

            above_ema21[ticker] = (close > ema21).astype(float)
            above_ema50[ticker] = (close > ema50).astype(float)
            advancing[ticker] = (close.pct_change() > 0).astype(float)
            has_data[ticker] = close.notna().astype(float)

        breadth = pd.DataFrame(index=date_index)

        # Coverage: kaç hissenin verisi var
        breadth['coverage'] = has_data.mean(axis=1)
        coverage_penalty = (breadth['coverage'] / self.min_coverage).clip(0, 1)

        # Temel metrikler × coverage cezası
        breadth['pct_above_ema21'] = above_ema21.mean(axis=1) * coverage_penalty
        breadth['pct_above_ema50'] = above_ema50.mean(axis=1) * coverage_penalty
        breadth['advance_ratio'] = advancing.mean(axis=1) * coverage_penalty

        # Advance/Decline oranı (ham)
        adv_count = advancing.sum(axis=1)
        dec_count = has_data.sum(axis=1) - adv_count
        breadth['ad_ratio'] = adv_count / dec_count.replace(0, np.nan)

        return breadth

    def score(self, breadth: pd.DataFrame) -> pd.Series:
        """0-100 breadth thrust skoru"""
        scores = pd.Series(0.0, index=breadth.index)

        pct21 = breadth['pct_above_ema21']

        # --- Thrust: düşükten yükseğe sıçrama ---
        rolling_min = pct21.rolling(self.thrust_window).min()
        thrust = (
            (rolling_min < self.thrust_low) &
            (pct21 > self.thrust_high)
        )
        thrust_magnitude = (pct21 - rolling_min).clip(0, 1)
        scores = scores + thrust.astype(float) * 50

        # --- Gradual recovery ---
        pct21_5d_change = pct21 - pct21.shift(5)
        gradual = (
            (pct21_5d_change > 0.15) &
            (pct21 > 0.40) &
            (rolling_min < 0.35)
        )
        scores = scores + gradual.astype(float) * 25

        # --- A/D ratio surge ---
        ad = breadth['ad_ratio']
        ad_surge = (ad > 2.0) & (ad.shift(1) > 1.5)
        scores = scores + ad_surge.astype(float) * 25

        # Coverage cezası doğrudan
        scores = scores * breadth['coverage'].clip(0.5, 1.0)

        return scores.clip(0, 100)


# =============================================================================
# MODÜL 3: REGIME SHIFT (Sequential Step 3)
# =============================================================================

class RegimeModule:
    """
    ADX bazlı rejim değişikliği.
    DI farkı ATR-normalize (v2).
    """

    def __init__(self,
                 adx_trend_threshold: float = 25.0,
                 adx_dead_threshold: float = 18.0,
                 adx_rebirth_threshold: float = 18.0,
                 slope_lookback: int = 5):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_dead_threshold = adx_dead_threshold
        self.adx_rebirth_threshold = adx_rebirth_threshold
        self.slope_lookback = slope_lookback

    def score(self, df: pd.DataFrame) -> pd.Series:
        """0-100 regime shift skoru"""
        adx, plus_di, minus_di = Indicators.adx(df['high'], df['low'], df['close'])
        di_norm = Indicators.di_diff_normalized(df['high'], df['low'], df['close'])
        adx_slope = adx.diff(self.slope_lookback) / self.slope_lookback
        ema21 = Indicators.ema(df['close'], 21)

        scores = pd.Series(0.0, index=df.index)

        # --- Faz 1 tespiti: önceden trend vardı, şimdi ölüyor ---
        was_trending_down = (
            adx.rolling(20).max() > self.adx_trend_threshold
        )

        # --- Faz 2: yeni trend doğumu ---
        adx_rebirth = (
            (adx > self.adx_rebirth_threshold) &
            (adx_slope > 0) &
            (di_norm > 0)
        )

        # Close > EMA21 konfirmasyonu
        above_ema = df['close'] > ema21

        # Skor bileşenleri
        slope_score = (adx_slope.clip(0, 2) * 15).fillna(0)
        di_score = (di_norm.clip(0, 10) * 3).fillna(0)
        ema_score = above_ema.astype(float) * 15
        adx_min_20 = adx.rolling(20).min()
        adx_rebound = ((adx - adx_min_20) > 5).astype(float) * 15

        scores = (slope_score + di_score + ema_score + adx_rebound).clip(0, 100)

        # Sadece önceden düşüş trendi varsa anlamlı
        scores = scores * was_trending_down.astype(float)

        return scores

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Her bar için MacroPhase döndür — RSI ve momentum dahil"""
        adx, plus_di, minus_di = Indicators.adx(df['high'], df['low'], df['close'])
        di_diff = plus_di - minus_di
        ema21 = Indicators.ema(df['close'], 21)
        ema50 = Indicators.ema(df['close'], 50)
        adx_slope = adx.diff(5) / 5
        rsi = Indicators.rsi(df['close'], 14)
        # Fiyat momentumu: son 5 günlük değişim
        momentum_5d = df['close'].pct_change(5)

        phases = []
        for i in range(len(df)):
            if i < 30:
                phases.append(MacroPhase.DOWNTREND)
                continue

            a = adx.iloc[i]
            slope = adx_slope.iloc[i]
            did = di_diff.iloc[i]
            r = rsi.iloc[i]
            mom = momentum_5d.iloc[i] if pd.notna(momentum_5d.iloc[i]) else 0
            close_i = df['close'].iloc[i]
            e21 = ema21.iloc[i]
            e50 = ema50.iloc[i]

            # Sert düşüş tespiti: RSI hızlı düşüş veya güçlü negatif momentum
            sharp_decline = (r < 40 and mom < -0.03) or (r < 35)

            if a > 25 and did < 0:
                phases.append(MacroPhase.DOWNTREND)
            elif sharp_decline and close_i < e21:
                # RSI düşük + EMA21 altı → aktif düşüş
                phases.append(MacroPhase.DOWNTREND)
            elif sharp_decline:
                # RSI düşük ama henüz EMA21 üstü → dağıtım/düşüş başlangıcı
                phases.append(MacroPhase.DISTRIBUTION)
            elif a < 18:
                phases.append(MacroPhase.BOTTOMING)
            elif a > 18 and slope > 0 and did > 0 and close_i > e21:
                phases.append(MacroPhase.EARLY_UPTREND)
            elif a > 25 and did > 0 and r > 50 and close_i > e21:
                phases.append(MacroPhase.UPTREND)
            elif a > 25 and did > 0:
                # DI+ hala pozitif ama momentum zayıflıyor
                if r < 50 or mom < -0.02:
                    phases.append(MacroPhase.DISTRIBUTION)
                else:
                    phases.append(MacroPhase.UPTREND)
            else:
                phases.append(MacroPhase.BOTTOMING)

        return pd.Series(phases, index=df.index)


# =============================================================================
# MODÜL 4: STRUCTURE BREAK (Sequential Step 4)
# =============================================================================

class StructureBreakModule:
    """
    Higher high / higher low tespiti.
    Swing high kırılma = trend dönüşü konfirmasyonu.
    """

    def __init__(self, pivot_lookback: int = 5, break_buffer_pct: float = 0.002):
        self.pivot_lookback = pivot_lookback
        self.break_buffer_pct = break_buffer_pct

    def find_pivots(self, high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Pivot high ve pivot low bul"""
        lb = self.pivot_lookback
        pivot_highs = pd.Series(np.nan, index=high.index)
        pivot_lows = pd.Series(np.nan, index=low.index)

        for i in range(lb, len(high) - lb):
            window_h = high.iloc[i-lb:i+lb+1]
            if high.iloc[i] == window_h.max():
                pivot_highs.iloc[i] = high.iloc[i]

            window_l = low.iloc[i-lb:i+lb+1]
            if low.iloc[i] == window_l.min():
                pivot_lows.iloc[i] = low.iloc[i]

        return pivot_highs, pivot_lows

    def score(self, df: pd.DataFrame) -> pd.Series:
        """0-100 structure break skoru"""
        pivot_highs, pivot_lows = self.find_pivots(df['high'], df['low'])

        scores = pd.Series(0.0, index=df.index)

        # Forward fill pivots to get "last known" pivot
        last_pivot_high = pivot_highs.ffill()
        last_pivot_low = pivot_lows.ffill()

        # İkinci önceki pivotlar (higher high/low tespiti için)
        prev_pivot_high = pd.Series(np.nan, index=df.index)
        prev_pivot_low = pd.Series(np.nan, index=df.index)

        ph_values = pivot_highs.dropna()
        for j in range(1, len(ph_values)):
            idx = ph_values.index[j]
            prev_pivot_high.loc[idx] = ph_values.iloc[j-1]
        prev_pivot_high = prev_pivot_high.ffill()

        pl_values = pivot_lows.dropna()
        for j in range(1, len(pl_values)):
            idx = pl_values.index[j]
            prev_pivot_low.loc[idx] = pl_values.iloc[j-1]
        prev_pivot_low = prev_pivot_low.ffill()

        # --- Structure Break: Close > last pivot high ---
        buffer = last_pivot_high * self.break_buffer_pct
        breakout = (df['close'] > last_pivot_high + buffer)
        scores = scores + breakout.astype(float) * 40

        # --- Higher High: son pivot high > önceki pivot high ---
        higher_high = (last_pivot_high > prev_pivot_high)
        scores = scores + (breakout & higher_high).astype(float) * 25

        # --- Higher Low: son pivot low > önceki pivot low ---
        higher_low = (last_pivot_low > prev_pivot_low)
        scores = scores + higher_low.astype(float) * 20

        # --- 40 günlük en yüksek kapanış ---
        highest_40 = df['close'].rolling(40).max()
        near_high = (df['close'] > highest_40 * 0.97)
        scores = scores + near_high.astype(float) * 15

        return scores.clip(0, 100)

    def has_break(self, df: pd.DataFrame) -> pd.Series:
        """Boolean: structure break var mı?"""
        pivot_highs, _ = self.find_pivots(df['high'], df['low'])
        last_ph = pivot_highs.ffill()
        return df['close'] > last_ph * (1 + self.break_buffer_pct)


# =============================================================================
# MODÜL 4b: BOUNCE QUALITY (Tepki vs Kalıcı Dönüş)
# =============================================================================

class BounceQualityModule:
    """
    Tepki rallisi mi, kalıcı dönüş mü?
    DIP sinyali tetiklendikten sonra çalışır.
    """

    def __init__(self, hold_days: int = 3, avwap_buffer: float = 0.005):
        self.hold_days = hold_days
        self.avwap_buffer = avwap_buffer

    def assess(self, df: pd.DataFrame,
               anchored_vwap: Optional[pd.Series],
               structure_has_break: pd.Series,
               candle_score: Optional[pd.Series] = None) -> Tuple[float, str]:
        """
        Son bar için bounce kalitesini değerlendir.
        Returns: (quality_score 0-100, label)
        Label: "KALICI_DONUS" | "TEPKI" | "BELIRSIZ"
        """
        quality = 0.0
        close = df['close']
        rsi = Indicators.rsi(close, 14)

        # --- Kriter 1: Anchored VWAP üstünde kapanış (25 puan) ---
        if anchored_vwap is not None and pd.notna(anchored_vwap.iloc[-1]):
            avwap_val = anchored_vwap.iloc[-1]
            if close.iloc[-1] > avwap_val * (1 + self.avwap_buffer):
                quality += 25

            # --- Kriter 2: A-VWAP üstünde N gün tutunma (25 puan) ---
            if len(close) >= self.hold_days:
                recent_close = close.iloc[-self.hold_days:]
                recent_avwap = anchored_vwap.iloc[-self.hold_days:]
                valid = recent_avwap.notna()
                if valid.any() and (recent_close[valid] > recent_avwap[valid]).all():
                    quality += 25

        # --- Kriter 3: Structure break (higher high) (25 puan) ---
        if pd.notna(structure_has_break.iloc[-1]) and structure_has_break.iloc[-1]:
            quality += 25

        # --- Kriter 4: Dönüş mumu + hacim teyidi (15 puan) ---
        if candle_score is not None:
            # Son 3 barda dönüş mumu var mı?
            recent_candle = candle_score.iloc[-3:]
            if (recent_candle > 0).any():
                quality += 15

        # --- Kriter 5: RSI yükselen trend (10 puan) ---
        if len(rsi) >= 5:
            rsi_recent = rsi.iloc[-5:]
            if rsi_recent.is_monotonic_increasing or (rsi_recent.iloc[-1] > rsi_recent.iloc[0] + 5):
                quality += 10

        quality = min(quality, 100.0)

        # --- Etiketleme ---
        has_break = (pd.notna(structure_has_break.iloc[-1]) and
                     structure_has_break.iloc[-1])
        if quality >= 60 and has_break:
            label = "KALICI_DONUS"
        elif quality >= 40:
            label = "BELIRSIZ"
        else:
            label = "TEPKI"

        # A-VWAP üstünde ama quality < 40 ise en az BELIRSIZ
        if anchored_vwap is not None and pd.notna(anchored_vwap.iloc[-1]):
            if close.iloc[-1] > anchored_vwap.iloc[-1] and label == "TEPKI":
                label = "BELIRSIZ"

        return quality, label


# =============================================================================
# MODÜL 5: DIP REVERSAL (Gate'li)
# =============================================================================

class DipReversalModule:
    """
    Oversold bounce - MARKET GATE uygulanır.
    Gate kapalıysa skor %70 kırpılır (v2).
    RS filtresi dahil.
    VWAP Recovery + Dönüş Mumu confluence sinyalleri (v2.1).
    """

    def __init__(self,
                 rsi_oversold: float = 30.0,
                 rsi_recovery: float = 40.0,
                 wt_oversold: float = -60.0,
                 min_confluence: int = 2,
                 gate_penalty: float = 0.30):
        self.rsi_oversold = rsi_oversold
        self.rsi_recovery = rsi_recovery
        self.wt_oversold = wt_oversold
        self.min_confluence = min_confluence
        self.gate_penalty = gate_penalty
        self.reversal_candle = ReversalCandleModule()

    def score(self, df: pd.DataFrame,
              gate_strength: Optional[pd.Series] = None,
              anchored_vwap: Optional[pd.Series] = None) -> pd.Series:
        """
        0-100 dip reversal skoru.
        gate_strength: 0-1 arası, makro gate çarpanı. None ise gate yok.
        anchored_vwap: pd.Series veya None — AVWAP pozisyon gate'i için.
        """
        close = df['close']
        rsi = Indicators.rsi(close, 14)
        hlc3 = (df['high'] + df['low'] + close) / 3
        wt1, wt2 = Indicators.wavetrend(hlc3)
        pct_b = Indicators.percent_b(close)
        mfi = Indicators.mfi(df['high'], df['low'], close, df['volume'])
        ema21 = Indicators.ema(close, 21)
        ema21_slope = ema21.diff(5) / 5

        scores = pd.Series(0.0, index=df.index)
        confluence = pd.Series(0, index=df.index)

        # --- RSI oversold recovery ---
        rsi_min5 = rsi.rolling(5).min()
        was_oversold = rsi_min5 < self.rsi_oversold
        recovering = (rsi > self.rsi_recovery) & (rsi > rsi.shift(1))
        rsi_signal = was_oversold & recovering
        scores = scores + rsi_signal.astype(float) * 20
        confluence = confluence + rsi_signal.astype(int)

        # --- RSI divergence ---
        close_5d_min = close.rolling(10).min()
        rsi_5d_min = rsi.rolling(10).min()
        price_lower = close < close_5d_min.shift(1) * 1.01
        rsi_higher = rsi > rsi_5d_min.shift(1) + 3
        div_signal = price_lower & rsi_higher & (rsi < 45)
        scores = scores + div_signal.astype(float) * 15
        confluence = confluence + div_signal.astype(int)

        # --- WaveTrend crossover ---
        wt_min3 = wt1.rolling(3).min()
        wt_was_os = wt_min3 < self.wt_oversold
        wt_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
        wt_signal = wt_was_os & wt_cross
        scores = scores + wt_signal.astype(float) * 18
        confluence = confluence + wt_signal.astype(int)

        # --- Bollinger %B recovery ---
        pctb_min5 = pct_b.rolling(5).min()
        pctb_signal = (pctb_min5 < 0.05) & (pct_b > pct_b.shift(1)) & (pct_b > 0.05)
        scores = scores + pctb_signal.astype(float) * 12
        confluence = confluence + pctb_signal.astype(int)

        # --- MFI oversold ---
        mfi_signal = mfi < 25
        scores = scores + mfi_signal.astype(float) * 10
        confluence = confluence + mfi_signal.astype(int)

        # --- EMA21 üstü + slope pozitif (trend çekirdeği) ---
        trend_core = (close > ema21) & (ema21_slope > 0)
        scores = scores + trend_core.astype(float) * 15
        confluence = confluence + trend_core.astype(int)

        # --- Candlestick bonus (legacy) ---
        body = close - df['open']
        body_pct = body / df['open'].replace(0, np.nan)
        engulfing = (
            (body_pct > 0.01) &
            (body_pct.shift(1) < -0.005) &
            (close > df['open'].shift(1))
        )
        scores = scores + engulfing.astype(float) * 10

        # --- VWAP Recovery (15 puan, confluence++) ---
        vwap = Indicators.rolling_vwap(df['high'], df['low'], close,
                                       df['volume'], period=20)
        vwap_cross_up = (close > vwap) & (close.shift(1) <= vwap.shift(1))
        vwap_hold = (close > vwap) & (close.shift(1) > vwap.shift(1))
        vwap_signal = vwap_cross_up | vwap_hold
        scores = scores + vwap_signal.astype(float) * 15
        confluence = confluence + vwap_signal.astype(int)

        # --- Dönüş Mumu (20 puan, confluence++) ---
        candle_score, candle_name = self.reversal_candle.detect(df)
        candle_signal = candle_score > 0
        scores = scores + candle_score  # candle_score zaten 10-37.5 arası
        confluence = confluence + candle_signal.astype(int)

        # --- Confluence multiplier ---
        # 0 sinyal → 0.30x, 1 sinyal → 0.60x, 2+ → 1.0x, 4+ → 1.2x
        conf_mult = pd.Series(1.0, index=df.index)
        conf_mult = conf_mult.where(confluence >= self.min_confluence, 0.60)
        conf_mult = conf_mult.where(confluence >= 1, self.gate_penalty)
        conf_mult = conf_mult * np.where(confluence >= 4, 1.2, 1.0)
        scores = scores * conf_mult

        # --- MARKET GATE UYGULA ---
        if gate_strength is not None:
            effective_gate = gate_strength.clip(self.gate_penalty, 1.0)
            scores = scores * effective_gate

        # --- AVWAP POZISYON GATE ---
        if anchored_vwap is not None:
            avwap_valid = anchored_vwap.notna()
            below_avwap = (close < anchored_vwap) & avwap_valid
            above_avwap = (close >= anchored_vwap) & avwap_valid
            avwap_mult = np.where(below_avwap, 0.70, np.where(above_avwap, 1.15, 1.0))
            scores = scores * avwap_mult

        return scores.clip(0, 100)


# =============================================================================
# ANA SCREENER V2
# =============================================================================

class ReversalScreenerV2:
    """
    NOX Reversal Screener v2 - Katmanlı tetikleme sistemi.

    İki ayrı çıktı katmanı:
    1. macro_regime() → Piyasa geneli durum + gate
    2. swing_scan()   → Bireysel hisse giriş sinyalleri (gate'li)

    Kullanım:
        screener = ReversalScreenerV2()
        macro = screener.macro_regime(bist100_df, stock_dfs)
        entries = screener.swing_scan(stock_dfs, bist100_df, macro)
    """

    def __init__(self):
        self.vol_climax = VolumeClimaxModule()
        self.breadth = BreadthModule()
        self.regime = RegimeModule()
        self.structure = StructureBreakModule()
        self.dip = DipReversalModule()
        self.reversal_candle = ReversalCandleModule()
        self.bounce_quality = BounceQualityModule()

    # =========================================================================
    # MAKRO REJİM
    # =========================================================================

    def macro_regime(self, index_df: pd.DataFrame,
                     stock_dfs: Dict[str, pd.DataFrame]) -> MacroContext:
        """
        Piyasa geneli makro durum değerlendirmesi.
        Sequential: VolClimax → Breadth → Regime → Structure

        Args:
            index_df: BIST100 endeks OHLCV
            stock_dfs: {ticker: DataFrame} tüm hisseler

        Returns:
            MacroContext
        """
        # 1. Volume Climax (endeks) + climax mask
        vc_scores, idx_climax_mask = self.vol_climax.score_and_detect(index_df)
        vc_latest = vc_scores.iloc[-1]
        vc_recent_max = vc_scores.iloc[-10:].max()

        # Endeks Anchored VWAP
        idx_avwap = Indicators.anchored_vwap(
            index_df['high'], index_df['low'], index_df['close'],
            index_df['volume'], idx_climax_mask
        )

        # 2. Breadth
        all_dates = sorted(set().union(*(df.index for df in stock_dfs.values())))
        date_index = pd.DatetimeIndex(all_dates)
        breadth_df = self.breadth.compute(stock_dfs, date_index)
        br_scores = self.breadth.score(breadth_df)
        # Endeks index'ine reindex
        br_scores_reindexed = br_scores.reindex(index_df.index, method='ffill').fillna(0)
        br_latest = br_scores_reindexed.iloc[-1]

        # 3. Regime
        reg_scores = self.regime.score(index_df)
        reg_latest = reg_scores.iloc[-1]
        phase_series = self.regime.classify(index_df)
        current_phase_raw = phase_series.iloc[-1]

        # 4. Structure Break
        str_scores = self.structure.score(index_df)
        str_latest = str_scores.iloc[-1]
        has_break = self.structure.has_break(index_df).iloc[-1]

        # === Sequential Phase Determination ===
        # Breadth metrikleri
        br_pct21 = 0.5  # default
        br_adv_ratio = 0.5
        if len(breadth_df) > 0:
            last_br = breadth_df.iloc[-1]
            br_pct21 = last_br.get('pct_above_ema21', 0.5)
            br_adv_ratio = last_br.get('advance_ratio', 0.5)

        # Endeks RSI
        idx_rsi = Indicators.rsi(index_df['close'], 14).iloc[-1]
        idx_mom_5d = index_df['close'].pct_change(5).iloc[-1]

        # Breadth çökmüş mü? (hisselerin çoğu düşüşte)
        breadth_weak = br_pct21 < 0.40 and br_adv_ratio < 0.30
        breadth_collapse = br_pct21 < 0.30 or br_adv_ratio < 0.15

        if vc_recent_max >= 30:
            if br_latest >= 30:
                if reg_latest >= 30:
                    if has_break:
                        phase = MacroPhase.EARLY_UPTREND
                    else:
                        phase = MacroPhase.REGIME_SHIFT
                else:
                    phase = MacroPhase.BOTTOMING
            else:
                phase = MacroPhase.CAPITULATION
        elif breadth_collapse:
            # Breadth çökmüş — endeks düşmese bile çoğu hisse düşüşte
            if idx_rsi < 40:
                phase = MacroPhase.DOWNTREND
            elif idx_rsi < 50 or idx_mom_5d < -0.02:
                phase = MacroPhase.CAPITULATION
            else:
                phase = MacroPhase.DISTRIBUTION
        elif breadth_weak and (idx_rsi < 55 or idx_mom_5d < -0.01):
            # Breadth zayıf + endeks de zayıflıyor
            phase = MacroPhase.DISTRIBUTION
        else:
            if current_phase_raw == MacroPhase.UPTREND:
                phase = MacroPhase.UPTREND
            elif current_phase_raw == MacroPhase.EARLY_UPTREND:
                phase = MacroPhase.EARLY_UPTREND
            elif current_phase_raw == MacroPhase.DISTRIBUTION:
                phase = MacroPhase.DISTRIBUTION
            elif br_latest >= 40 and reg_latest >= 30:
                phase = MacroPhase.REGIME_SHIFT
            else:
                phase = current_phase_raw

        # === Gate Determination ===
        gate_open = (br_latest >= 25) or (reg_latest >= 25) or (phase in (
            MacroPhase.REGIME_SHIFT, MacroPhase.EARLY_UPTREND, MacroPhase.UPTREND
        ))

        if gate_open:
            gate_strength = min(1.0, (br_latest + reg_latest) / 100)
            gate_strength = max(gate_strength, 0.50)
        else:
            gate_strength = 0.30

        # Composite phase score
        phase_score = (vc_latest * 0.20 + br_latest * 0.30 +
                       reg_latest * 0.25 + str_latest * 0.25)

        # Breadth details
        br_details = {}
        if len(breadth_df) > 0:
            last_row = breadth_df.iloc[-1]
            br_details = {
                'pct_above_ema21': round(last_row.get('pct_above_ema21', 0), 3),
                'pct_above_ema50': round(last_row.get('pct_above_ema50', 0), 3),
                'advance_ratio': round(last_row.get('advance_ratio', 0), 3),
                'ad_ratio': round(last_row.get('ad_ratio', 0) if pd.notna(last_row.get('ad_ratio', 0)) else 0, 2),
                'coverage': round(last_row.get('coverage', 0), 3),
            }

        return MacroContext(
            date=index_df.index[-1],
            phase=phase,
            phase_score=min(phase_score, 100),
            breadth_score=br_latest,
            regime_score=reg_latest,
            volume_climax_score=vc_latest,
            structure_score=str_latest,
            gate_open=gate_open,
            gate_strength=gate_strength,
            details={
                'breadth': br_details,
                'adx': round(Indicators.adx(index_df['high'], index_df['low'], index_df['close'])[0].iloc[-1], 2),
                'has_structure_break': has_break,
                'vc_recent_max_10d': round(vc_recent_max, 1),
                'idx_avwap_latest': round(idx_avwap.iloc[-1], 2) if pd.notna(idx_avwap.iloc[-1]) else None,
                'close_vs_avwap': ('ABOVE' if index_df['close'].iloc[-1] > idx_avwap.iloc[-1]
                                   else 'BELOW') if pd.notna(idx_avwap.iloc[-1]) else None,
            }
        )

    # =========================================================================
    # SWING TARAMA
    # =========================================================================

    def swing_scan(self, stock_dfs: Dict[str, pd.DataFrame],
                   index_df: pd.DataFrame,
                   macro: MacroContext) -> List[SwingEntry]:
        """
        Bireysel hisse swing giriş taraması - makro gate ile.

        Args:
            stock_dfs: {ticker: DataFrame}
            index_df: BIST100 endeks datası (RS hesabı için)
            macro: macro_regime() çıktısı

        Returns:
            List[SwingEntry] sorted by gated_score desc
        """
        entries = []
        bench_close = index_df['close'].reindex(
            sorted(set().union(*(df.index for df in stock_dfs.values()))),
            method='ffill'
        )

        for ticker, df in stock_dfs.items():
            if len(df) < 60:
                continue

            df.attrs['ticker'] = ticker

            try:
                # Gate strength series (tüm barlar için aynı)
                gate_series = pd.Series(macro.gate_strength, index=df.index)

                # 1. Volume Climax + climax mask
                climax_scores, climax_mask = self.vol_climax.score_and_detect(df)

                # 2. Anchored VWAP (climax barlarından itibaren)
                avwap_series = Indicators.anchored_vwap(
                    df['high'], df['low'], df['close'],
                    df['volume'], climax_mask
                )

                # 3. Rolling VWAP
                rvwap_series = Indicators.rolling_vwap(
                    df['high'], df['low'], df['close'],
                    df['volume'], period=20
                )

                # 4. Dip score (gate'li, VWAP + candle dahil)
                raw_dip_scores = self.dip.score(df, anchored_vwap=avwap_series)
                gated_dip_scores = self.dip.score(df, gate_strength=gate_series,
                                                   anchored_vwap=avwap_series)

                # 5. Structure break
                str_scores = self.structure.score(df)
                has_break = self.structure.has_break(df)

                # Dönüş mumu tespiti
                candle_score, candle_name = self.reversal_candle.detect(df)

                # 6. Bounce quality
                bq_quality, bq_label = self.bounce_quality.assess(
                    df, avwap_series, has_break, candle_score
                )

                # Relative Strength
                bench_aligned = bench_close.reindex(df.index, method='ffill')
                if bench_aligned.notna().sum() > 20:
                    rs = Indicators.relative_strength(df['close'], bench_aligned, 20)
                else:
                    rs = pd.Series(0.0, index=df.index)

                # RS trend (son 10 gün)
                rs_trend = rs.diff(10)

                # Latest values
                raw_latest = raw_dip_scores.iloc[-1]
                gated_latest = gated_dip_scores.iloc[-1]
                rs_latest = rs.iloc[-1] if pd.notna(rs.iloc[-1]) else 0
                rs_trend_latest = rs_trend.iloc[-1] if pd.notna(rs_trend.iloc[-1]) else 0
                break_latest = has_break.iloc[-1] if pd.notna(has_break.iloc[-1]) else False

                # Structure score bonus
                str_latest = str_scores.iloc[-1]
                total_gated = gated_latest * 0.60 + str_latest * 0.25 + max(rs_latest * 100, 0) * 0.15
                total_gated = min(total_gated, 100)

                # State classification
                if total_gated >= 60 and macro.gate_open and break_latest and rs_latest > 0:
                    state = SwingState.ENTRY_SIGNAL
                elif total_gated >= 40 and macro.gate_open:
                    if break_latest:
                        state = SwingState.ENTRY_SIGNAL
                    else:
                        state = SwingState.BREAKOUT_PENDING
                elif raw_latest >= 30 and macro.gate_open:
                    state = SwingState.GATED_READY
                elif raw_latest >= 25:
                    state = SwingState.DIP_CANDIDATE
                else:
                    state = SwingState.NO_SETUP

                # EMA21 ve RSI bilgisi
                ema21 = Indicators.ema(df['close'], 21)
                rsi = Indicators.rsi(df['close'], 14)

                # VWAP pozisyon bilgileri
                close_latest = df['close'].iloc[-1]
                rvwap_latest = rvwap_series.iloc[-1]
                avwap_latest = avwap_series.iloc[-1]
                candle_name_latest = candle_name.iloc[-1]

                entries.append(SwingEntry(
                    ticker=ticker,
                    date=df.index[-1],
                    state=state,
                    raw_score=raw_latest,
                    gated_score=total_gated,
                    rs_score=rs_latest,
                    structure_break=bool(break_latest),
                    details={
                        'close': round(close_latest, 2),
                        'rsi': round(rsi.iloc[-1], 1),
                        'above_ema21': bool(close_latest > ema21.iloc[-1]),
                        'rs_trend_10d': round(rs_trend_latest, 4),
                        'structure_score': round(str_latest, 1),
                        'gate_multiplier': round(macro.gate_strength, 2),
                        'rolling_vwap': round(rvwap_latest, 2) if pd.notna(rvwap_latest) else None,
                        'anchored_vwap': round(avwap_latest, 2) if pd.notna(avwap_latest) else None,
                        'vwap_position': ('ABOVE' if close_latest > rvwap_latest else 'BELOW') if pd.notna(rvwap_latest) else None,
                        'avwap_position': ('ABOVE' if close_latest > avwap_latest else 'BELOW') if pd.notna(avwap_latest) else None,
                        'reversal_candle': candle_name_latest if candle_name_latest else None,
                        'bounce_quality': round(bq_quality, 1),
                        'bounce_label': bq_label,
                    }
                ))

            except Exception as e:
                print(f"  ⚠ {ticker}: {e}")
                continue

        # Sort by gated_score desc
        entries.sort(key=lambda e: e.gated_score, reverse=True)
        return entries

    # =========================================================================
    # SIGNAL STUDY (Backtest framework)
    # =========================================================================

    def signal_study(self, stock_dfs: Dict[str, pd.DataFrame],
                     index_df: pd.DataFrame,
                     forward_days: List[int] = [1, 5, 10],
                     score_threshold: float = 30.0) -> pd.DataFrame:
        """
        Her sinyal türü için forward return + MAE/MFE analizi.

        Returns:
            DataFrame with columns: ticker, signal_date, signal_type, score,
                                     fwd_return_Xd, mae_Xd, mfe_Xd
        """
        all_dates = sorted(set().union(*(df.index for df in stock_dfs.values())))
        date_index = pd.DatetimeIndex(all_dates)

        # Breadth precompute
        breadth_df = self.breadth.compute(stock_dfs, date_index)
        br_scores = self.breadth.score(breadth_df)

        rows = []

        for ticker, df in stock_dfs.items():
            if len(df) < 80:
                continue

            df.attrs['ticker'] = ticker

            # Run all modules
            vc_scores = self.vol_climax.score(df)
            reg_scores = self.regime.score(df)
            str_scores = self.structure.score(df)
            dip_scores = self.dip.score(df)
            br_aligned = br_scores.reindex(df.index, method='ffill').fillna(0)

            signal_map = {
                'VOLUME_CLIMAX': vc_scores,
                'BREADTH': br_aligned,
                'REGIME': reg_scores,
                'STRUCTURE': str_scores,
                'DIP_REVERSAL': dip_scores,
            }

            max_fwd = max(forward_days)

            for sig_type, score_series in signal_map.items():
                for i in range(50, len(df) - max_fwd):
                    if score_series.iloc[i] < score_threshold:
                        continue

                    entry_price = df['close'].iloc[i]
                    row = SignalStudyRow(
                        ticker=ticker,
                        signal_date=df.index[i],
                        signal_type=sig_type,
                        signal_score=round(score_series.iloc[i], 1),
                        entry_price=entry_price,
                    )

                    for fd in forward_days:
                        if i + fd < len(df):
                            fwd_close = df['close'].iloc[i + fd]
                            fwd_ret = (fwd_close - entry_price) / entry_price * 100

                            fwd_lows = df['low'].iloc[i+1:i+fd+1]
                            mae = ((fwd_lows.min() - entry_price) / entry_price * 100) if len(fwd_lows) > 0 else 0

                            fwd_highs = df['high'].iloc[i+1:i+fd+1]
                            mfe = ((fwd_highs.max() - entry_price) / entry_price * 100) if len(fwd_highs) > 0 else 0

                            setattr(row, f'fwd_return_{fd}d', round(fwd_ret, 3))
                            setattr(row, f'mae_{fd}d', round(mae, 3))
                            setattr(row, f'mfe_{fd}d', round(mfe, 3))

                    rows.append(row)

        # Convert to DataFrame
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            rec = {
                'ticker': r.ticker,
                'signal_date': r.signal_date,
                'signal_type': r.signal_type,
                'signal_score': r.signal_score,
                'entry_price': r.entry_price,
            }
            for fd in forward_days:
                rec[f'fwd_return_{fd}d'] = getattr(r, f'fwd_return_{fd}d', None)
                rec[f'mae_{fd}d'] = getattr(r, f'mae_{fd}d', None)
                rec[f'mfe_{fd}d'] = getattr(r, f'mfe_{fd}d', None)
            records.append(rec)

        return pd.DataFrame(records)

    # =========================================================================
    # RAPORLAMA
    # =========================================================================

    def print_macro_report(self, macro: MacroContext):
        """Makro rejim raporu"""
        print(macro)
        print(f"\n  Detaylar:")
        for k, v in macro.details.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"    {k}: {v}")

        print(f"\n  Gate Durumu:")
        if macro.gate_open:
            print(f"    GATE ACIK - Swing tarama aktif (strength: {macro.gate_strength:.0%})")
            print(f"    -> DIP sinyalleri {macro.gate_strength:.0%} carpanla degerlendirilecek")
        else:
            print(f"    GATE KAPALI - DIP sinyalleri %70 kirpiliyor")
            print(f"    -> Makro teyit bekleniyor (breadth veya regime sinyali)")

    def print_swing_report(self, entries: List[SwingEntry], macro: MacroContext,
                           top_n: int = 20):
        """Swing tarama raporu"""
        print(f"\n{'='*75}")
        print(f"  NOX REVERSAL SCREENER v2 - SWING TARAMA RAPORU")
        print(f"  {macro.date.strftime('%Y-%m-%d')} | Makro: {macro.phase.value} | "
              f"Gate: {'ACIK' if macro.gate_open else 'KAPALI'} ({macro.gate_strength:.0%})")
        print(f"{'='*75}")

        # Summary
        state_counts = {}
        for e in entries:
            state_counts[e.state.value] = state_counts.get(e.state.value, 0) + 1

        print(f"\n  Taranan: {len(entries)} hisse")
        for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
            marker = {'ENTRY_SIGNAL': '[ENTRY]', 'BREAKOUT_PENDING': '[BRKOUT]',
                      'GATED_READY': '[READY]', 'DIP_CANDIDATE': '[DIP]',
                      'NO_SETUP': '[NONE]'}.get(state, '[?]')
            print(f"    {marker} {state}: {count}")

        # Top entries
        actionable = [e for e in entries if e.state.value != 'NO_SETUP']
        if actionable:
            print(f"\n  {'─'*95}")
            print(f"  {'Hisse':<8} {'Durum':<22} {'Raw':>5} {'Gated':>6} "
                  f"{'RS':>7} {'Break':>5} {'RSI':>5} {'VWAP':>5} "
                  f"{'Mum':<14} {'BQ':>4}")
            print(f"  {'─'*95}")

            for e in actionable[:top_n]:
                rsi_str = f"{e.details.get('rsi', 0):.0f}"
                break_str = "Y" if e.structure_break else "N"
                state_marker = {'ENTRY_SIGNAL': '[E]', 'BREAKOUT_PENDING': '[B]',
                                'GATED_READY': '[G]', 'DIP_CANDIDATE': '[D]'}.get(e.state.value, '')

                # VWAP position
                vwap_pos = e.details.get('vwap_position')
                vwap_str = '^' if vwap_pos == 'ABOVE' else ('v' if vwap_pos == 'BELOW' else '-')

                # Dönüş mumu
                candle = e.details.get('reversal_candle') or '-'

                # Bounce quality
                bq = e.details.get('bounce_quality', 0)
                bq_str = f"{bq:.0f}"

                print(f"  {e.ticker:<8} {state_marker} {e.state.value:<19} "
                      f"{e.raw_score:>5.1f} {e.gated_score:>6.1f} "
                      f"{e.rs_score:>+7.3f} {break_str:>5} {rsi_str:>5} {vwap_str:>5} "
                      f"{candle:<14} {bq_str:>4}")

            print(f"  {'─'*95}")
        else:
            print(f"\n  Aksiyon alinabilir sinyal yok.")

    def print_signal_study(self, study_df: pd.DataFrame):
        """Signal study ozet raporu"""
        if study_df.empty:
            print("  Signal study: Veri yok.")
            return

        print(f"\n{'='*80}")
        print(f"  SIGNAL STUDY RAPORU - Forward Return & MAE/MFE Analizi")
        print(f"{'='*80}")

        for sig_type in study_df['signal_type'].unique():
            subset = study_df[study_df['signal_type'] == sig_type]
            print(f"\n  {sig_type} ({len(subset)} sinyal)")
            print(f"  {'─'*55}")

            for col in subset.columns:
                if col.startswith('fwd_return_') or col.startswith('mae_') or col.startswith('mfe_'):
                    vals = subset[col].dropna()
                    if len(vals) > 0:
                        label = col.replace('fwd_return_', 'Return ').replace('mae_', 'MAE ').replace('mfe_', 'MFE ')
                        print(f"    {label:<15} "
                              f"Mean: {vals.mean():>+7.2f}%  "
                              f"Med: {vals.median():>+7.2f}%  "
                              f"Win%: {(vals > 0).mean()*100:>5.1f}%  "
                              f"Std: {vals.std():>6.2f}%")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_macro(index_df: pd.DataFrame, stock_dfs: Dict[str, pd.DataFrame]) -> MacroContext:
    """Hızlı makro rejim sorgusu"""
    screener = ReversalScreenerV2()
    return screener.macro_regime(index_df, stock_dfs)


def quick_swing(stock_dfs: Dict[str, pd.DataFrame], index_df: pd.DataFrame) -> List[SwingEntry]:
    """Hızlı swing tarama"""
    screener = ReversalScreenerV2()
    macro = screener.macro_regime(index_df, stock_dfs)
    return screener.swing_scan(stock_dfs, index_df, macro)
