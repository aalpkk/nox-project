"""
NOX SMC Pattern Screener — Price Action Kalip Tespit Modulu
===========================================================
Smart Money Concepts (SMC) tabanli pattern detection:
  QM (Quasimodo), Fakeout V1/V2, Flag Breakout, 3-Drive,
  Compression, Can-Can (S/R Flip), 2R/2S Fakeout.

Lowercase kolon konvansiyonu: close, high, low, open, volume.
Runner script (run_smc.py) uppercase→lowercase donusumunu yapar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from markets.bist.nox_v3_signals import find_pivot_lows, find_pivot_highs
from core.indicators import ema, sma


# =============================================================================
# CONSTANTS
# =============================================================================

SMC_CFG = {
    'pivot_lb': 5,
    'atr_len': 14,
    'same_level_atr': 0.5,
    'qm_max_retest_bars': 5,
    'qm_late_min_bars': 5,
    'qm_late_max_bars': 25,
    'fakeout_max_bars': 3,
    'flag_min_impulse_atr': 2.0,
    'flag_max_retrace_pct': 0.55,
    'flag_min_retrace_pct': 0.20,
    'compression_min_bars': 5,
    'drive_min_bars': 3,
    'cancan_tolerance_atr': 0.3,
    'scan_bars': 5,
}


# =============================================================================
# VERI YAPILARI
# =============================================================================

@dataclass
class Pivot:
    idx: int            # Pivot bar index (onay bari degil)
    confirm_idx: int    # Onay bari index
    price: float
    ptype: str          # 'high' veya 'low'
    label: str          # 'HH', 'HL', 'LH', 'LL' veya ilk pivot icin 'H', 'L'


@dataclass
class PatternSignal:
    bar_idx: int        # Sinyal bari (son onay)
    direction: str      # 'BUY' veya 'SELL'
    pattern: str        # 'QM_QUICK', 'QM_LATE', 'FAKEOUT_V1', vb.
    key_level: float    # Kritik fiyat seviyesi
    quality: int        # 0-100 kalite skoru
    stop: float         # Onerilen stop seviyesi
    target: float       # Onerilen hedef seviyesi
    details: dict = field(default_factory=dict)


# =============================================================================
# YARDIMCI — ATR / RSI (lowercase kolonlar)
# =============================================================================

def _true_range(df):
    """True Range — lowercase columns."""
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _rma(series, period):
    """Wilder's RMA (EMA with alpha=1/period)."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def _calc_atr(df, period=14):
    """ATR — lowercase columns."""
    return _rma(_true_range(df), period)


def _calc_rsi(series, period=14):
    """RSI — standard Wilder."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# =============================================================================
# 1. YAPI TESPITI — detect_structure
# =============================================================================

def detect_structure(df, lb=5):
    """
    Pivot noktalarini bul ve HH/HL/LH/LL olarak etiketle.

    find_pivot_lows/highs onay barinda (confirm_idx = i) tetiklenir,
    gercek pivot bari = i - lb.

    Returns: list[Pivot] kronolojik sirada (confirm_idx'e gore).
    """
    pivot_low_series = find_pivot_lows(df['low'], lb)
    pivot_high_series = find_pivot_highs(df['high'], lb)

    pivots = []

    # Tum NaN olmayan pivotlari topla
    for i in range(len(df)):
        if pd.notna(pivot_high_series.iloc[i]):
            pivot_bar = i - lb
            pivots.append(Pivot(
                idx=pivot_bar,
                confirm_idx=i,
                price=float(pivot_high_series.iloc[i]),
                ptype='high',
                label='H',  # gecici
            ))
        if pd.notna(pivot_low_series.iloc[i]):
            pivot_bar = i - lb
            pivots.append(Pivot(
                idx=pivot_bar,
                confirm_idx=i,
                price=float(pivot_low_series.iloc[i]),
                ptype='low',
                label='L',  # gecici
            ))

    # Kronolojik sira (onay barina gore, esitlikte idx'e gore)
    pivots.sort(key=lambda p: (p.confirm_idx, p.idx))

    # HH/HL/LH/LL etiketleme
    last_high_price = None
    last_low_price = None

    for p in pivots:
        if p.ptype == 'high':
            if last_high_price is None:
                p.label = 'H'
            elif p.price > last_high_price:
                p.label = 'HH'
            else:
                p.label = 'LH'
            last_high_price = p.price
        else:  # low
            if last_low_price is None:
                p.label = 'L'
            elif p.price > last_low_price:
                p.label = 'HL'
            else:
                p.label = 'LL'
            last_low_price = p.price

    return pivots


# =============================================================================
# 2. MARKET BIAS
# =============================================================================

def get_market_bias(pivots):
    """
    Son pivotlara bakarak yapi yonu belirle.
    Returns: 'bullish', 'bearish', veya 'neutral'.
    """
    if len(pivots) < 4:
        return 'neutral'

    # Son high ve son low'u bul
    last_high = None
    last_low = None
    for p in reversed(pivots):
        if p.ptype == 'high' and last_high is None:
            last_high = p
        if p.ptype == 'low' and last_low is None:
            last_low = p
        if last_high and last_low:
            break

    if last_high is None or last_low is None:
        return 'neutral'

    if last_high.label == 'HH' and last_low.label == 'HL':
        return 'bullish'
    elif last_high.label == 'LH' and last_low.label == 'LL':
        return 'bearish'
    return 'neutral'


# =============================================================================
# YARDIMCI — seviye karsilastirma
# =============================================================================

def _same_level(price1, price2, atr_val, tolerance=None):
    """Iki fiyat 'ayni seviyede' mi? (ATR toleransi icinde)"""
    tol = tolerance if tolerance is not None else SMC_CFG['same_level_atr']
    return abs(price1 - price2) <= tol * atr_val


def _get_pivots_of_type(pivots, ptype):
    """Belirli tipteki pivotlari filtrele."""
    return [p for p in pivots if p.ptype == ptype]


def _vol_score(vol_val, vol_sma_val, max_pts=20):
    """Gradient volume score: higher ratio = more points."""
    if vol_sma_val <= 0:
        return 0
    ratio = vol_val / vol_sma_val
    if ratio >= 2.0:
        return max_pts
    if ratio >= 1.5:
        return int(max_pts * 0.75)
    if ratio >= 1.2:
        return int(max_pts * 0.5)
    if ratio >= 1.0:
        return int(max_pts * 0.25)
    return 0


def _proximity_score(distance, atr_val, max_pts=20):
    """Gradient proximity score: closer = more points."""
    if atr_val <= 0:
        return 0
    ratio = distance / atr_val
    if ratio <= 0.2:
        return max_pts
    if ratio <= 0.5:
        return int(max_pts * 0.75)
    if ratio <= 0.8:
        return int(max_pts * 0.5)
    if ratio <= 1.5:
        return int(max_pts * 0.25)
    return 0


# =============================================================================
# 3. QM (Quasimodo) — Quick + Late Retest
# =============================================================================

def detect_qm(df, pivots, atr):
    """
    Quasimodo pattern tespiti.

    Bullish QM:
      ... LH — LL — fiyat LH'nin ustune cikar (CHoCH) — geri LH seviyesine test → BUY
    Bearish QM:
      ... HL — HH — fiyat HL'nin altina duser (CHoCH) — geri HL seviyesine test → SELL
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values
    vol_sma20 = sma(df['volume'], 20).values

    # --- BULLISH QM ---
    # LH → LL → fiyat LH'yi yukari kirsin → retest LH seviyesi
    highs = _get_pivots_of_type(pivots, 'high')
    lows = _get_pivots_of_type(pivots, 'low')

    for i, lh_pivot in enumerate(highs):
        if lh_pivot.label != 'LH':
            continue
        neckline = lh_pivot.price

        # LH'dan sonra bir LL olmali
        ll_pivot = None
        for lp in lows:
            if lp.confirm_idx > lh_pivot.confirm_idx and lp.label == 'LL':
                ll_pivot = lp
                break
        if ll_pivot is None:
            continue

        # LL'den sonra fiyat neckline'i yukari kirmali (CHoCH)
        break_bar = None
        search_start = ll_pivot.confirm_idx + 1
        search_end = min(search_start + SMC_CFG['qm_late_max_bars'] + 10, n)
        for b in range(search_start, search_end):
            if close[b] > neckline:
                break_bar = b
                break
        if break_bar is None:
            continue

        # Kirilma body ile mi?
        break_body = close[break_bar] > neckline
        break_by_body = break_body and df['open'].iloc[break_bar] < neckline

        # Retest: fiyat neckline seviyesine geri donmeli
        atr_at_break = atr.iloc[break_bar] if break_bar < n else atr.iloc[-1]

        for b in range(break_bar + 1, min(break_bar + SMC_CFG['qm_late_max_bars'] + 1, n)):
            # Fiyat neckline yakinina dondu mu?
            bar_low = low[b]
            bar_close = close[b]

            retest_distance = abs(bar_low - neckline)
            if retest_distance > 1.5 * atr_at_break:
                continue

            # Close neckline'in ustunde olmali (bullish retest)
            if bar_close <= neckline:
                continue

            # --- Neckline invalidation ---
            # Break sonrasi herhangi bir barda close < neckline ise QM iptal
            neckline_lost = False
            for cb in range(break_bar + 1, b):
                if close[cb] < neckline:
                    neckline_lost = True
                    break
            if neckline_lost:
                continue

            # --- Momentum kontrolu ---
            # Son 4 barda fiyat 1.5+ ATR dustuyse bu retest degil, dusus trendi
            if b >= 4:
                price_drop = close[b - 4] - close[b]
                if atr_at_break > 0 and price_drop > 1.5 * atr_at_break:
                    continue

            # Quick vs Late
            bars_since_break = b - break_bar
            if bars_since_break <= SMC_CFG['qm_max_retest_bars']:
                pattern_type = 'QM_QUICK'
            elif bars_since_break >= SMC_CFG['qm_late_min_bars']:
                pattern_type = 'QM_LATE'
            else:
                continue

            # Kalite skoru (gradyan — max ~80)
            quality = 0
            # CHoCH temizligi (0-20)
            if break_by_body:
                break_body_size = abs(close[break_bar] - df['open'].iloc[break_bar])
                quality += min(int(break_body_size / atr_at_break * 15), 20) if atr_at_break > 0 else 12
            else:
                quality += 5
            # Retest hassasiyeti (0-20)
            quality += _proximity_score(retest_distance, atr_at_break, 20)
            # Hacim (0-20)
            if break_bar < len(vol_sma20):
                quality += _vol_score(vol[break_bar], vol_sma20[break_bar], 20)
            # Yapi netligi (0-20)
            struct_dist = abs(lh_pivot.price - ll_pivot.price)
            if atr_at_break > 0:
                sd_ratio = struct_dist / atr_at_break
                quality += min(int(sd_ratio * 10), 20)

            # --- Excursion filtresi ---
            # Fiyat neckline'dan 3+ ATR uzaklasip geri geldiyse QM iptal
            # (yeni yapi olusmus, neckline artik gecersiz)
            max_high_since = max(high[break_bar:b + 1])
            excursion_atr = (max_high_since - neckline) / atr_at_break if atr_at_break > 0 else 0
            if excursion_atr > 3.0:
                continue

            stop_price = ll_pivot.price - 0.5 * atr_at_break
            risk = bar_close - stop_price
            target_price = bar_close + 2.0 * risk

            signals.append(PatternSignal(
                bar_idx=b,
                direction='BUY',
                pattern=pattern_type,
                key_level=neckline,
                quality=max(min(quality, 100), 0),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'neckline': neckline,
                    'lh_idx': lh_pivot.idx,
                    'll_idx': ll_pivot.idx,
                    'break_bar': break_bar,
                    'retest_bar': b,
                },
            ))
            break  # Ilk retest yeterli

    # --- BEARISH QM ---
    # HL → HH → fiyat HL'yi asagi kirsin → retest HL seviyesi
    for i, hl_pivot in enumerate(lows):
        if hl_pivot.label != 'HL':
            continue
        neckline = hl_pivot.price

        # HL'den sonra bir HH olmali
        hh_pivot = None
        for hp in highs:
            if hp.confirm_idx > hl_pivot.confirm_idx and hp.label == 'HH':
                hh_pivot = hp
                break
        if hh_pivot is None:
            continue

        # HH'den sonra fiyat neckline'i asagi kirmali (CHoCH)
        break_bar = None
        search_start = hh_pivot.confirm_idx + 1
        search_end = min(search_start + SMC_CFG['qm_late_max_bars'] + 10, n)
        for b in range(search_start, search_end):
            if close[b] < neckline:
                break_bar = b
                break
        if break_bar is None:
            continue

        break_by_body = close[break_bar] < neckline and df['open'].iloc[break_bar] > neckline

        atr_at_break = atr.iloc[break_bar] if break_bar < n else atr.iloc[-1]

        for b in range(break_bar + 1, min(break_bar + SMC_CFG['qm_late_max_bars'] + 1, n)):
            bar_high = high[b]
            bar_close = close[b]

            retest_distance = abs(bar_high - neckline)
            if retest_distance > 1.5 * atr_at_break:
                continue

            if bar_close >= neckline:
                continue

            # --- Neckline invalidation ---
            # Break sonrasi herhangi bir barda close > neckline ise QM iptal
            neckline_lost = False
            for cb in range(break_bar + 1, b):
                if close[cb] > neckline:
                    neckline_lost = True
                    break
            if neckline_lost:
                continue

            # --- Momentum kontrolu ---
            # Son 4 barda fiyat 1.5+ ATR yukseldiyse bu retest degil, yukselis trendi
            if b >= 4:
                price_rise = close[b] - close[b - 4]
                if atr_at_break > 0 and price_rise > 1.5 * atr_at_break:
                    continue

            bars_since_break = b - break_bar
            if bars_since_break <= SMC_CFG['qm_max_retest_bars']:
                pattern_type = 'QM_QUICK'
            elif bars_since_break >= SMC_CFG['qm_late_min_bars']:
                pattern_type = 'QM_LATE'
            else:
                continue

            # Kalite skoru (gradyan — max ~80)
            quality = 0
            # CHoCH temizligi (0-20)
            if break_by_body:
                break_body_size = abs(close[break_bar] - df['open'].iloc[break_bar])
                quality += min(int(break_body_size / atr_at_break * 15), 20) if atr_at_break > 0 else 12
            else:
                quality += 5
            # Retest hassasiyeti (0-20)
            quality += _proximity_score(retest_distance, atr_at_break, 20)
            # Hacim (0-20)
            if break_bar < len(vol_sma20):
                quality += _vol_score(vol[break_bar], vol_sma20[break_bar], 20)
            # Yapi netligi (0-20)
            struct_dist = abs(hh_pivot.price - hl_pivot.price)
            if atr_at_break > 0:
                sd_ratio = struct_dist / atr_at_break
                quality += min(int(sd_ratio * 10), 20)

            # --- Excursion filtresi ---
            # Fiyat neckline'dan 3+ ATR uzaklasip geri geldiyse QM iptal
            min_low_since = min(low[break_bar:b + 1])
            excursion_atr = (neckline - min_low_since) / atr_at_break if atr_at_break > 0 else 0
            if excursion_atr > 3.0:
                continue

            stop_price = hh_pivot.price + 0.5 * atr_at_break
            risk = stop_price - bar_close
            target_price = bar_close - 2.0 * risk

            signals.append(PatternSignal(
                bar_idx=b,
                direction='SELL',
                pattern=pattern_type,
                key_level=neckline,
                quality=max(min(quality, 100), 0),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'neckline': neckline,
                    'hl_idx': hl_pivot.idx,
                    'hh_idx': hh_pivot.idx,
                    'break_bar': break_bar,
                    'retest_bar': b,
                },
            ))
            break

    return signals


# =============================================================================
# 4. FAKEOUT V1 (Liquidity Sweep)
# =============================================================================

def detect_fakeout_v1(df, pivots, atr):
    """
    Fakeout V1 — Liquidity Sweep.

    Bullish: Swing low altina kisa sureli kirilma → geri donusse BUY
    Bearish: Swing high ustune kisa sureli kirilma → geri donusse SELL
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    vol = df['volume'].values
    vol_sma20 = sma(df['volume'], 20).values
    max_bars = SMC_CFG['fakeout_max_bars']

    # --- BULLISH FAKEOUT ---
    swing_lows = [p for p in pivots if p.ptype == 'low']
    for sl in swing_lows:
        level = sl.price
        # Sweep: son barlarda fiyat seviyenin altina kirdi mi?
        search_start = sl.confirm_idx + 1
        search_end = min(search_start + 30, n)  # makul arama penceresi

        for b in range(search_start, search_end):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if atr_b <= 0:
                continue

            # Sweep: low seviyenin altina gecti
            if low[b] >= level:
                continue

            sweep_depth = level - low[b]
            # Max 2 ATR derinlik
            if sweep_depth > 2.0 * atr_b:
                break  # gercek kirilim, fakeout degil

            # Geri donus kontrolu: ayni bar veya sonraki max_bars barda close > level
            recovered = False
            recovery_bar = b
            for rb in range(b, min(b + max_bars + 1, n)):
                if close[rb] > level:
                    recovered = True
                    recovery_bar = rb
                    break

            if not recovered:
                continue

            # Wick orani: sweep barinda alt wick > body
            body = abs(close[b] - open_[b])
            lower_wick = min(close[b], open_[b]) - low[b]
            wick_rejection = lower_wick > body if body > 0 else lower_wick > 0

            # Kalite skoru (gradyan — max ~80)
            quality = 0
            # Rejection wick (0-20)
            if atr_b > 0:
                wick_ratio = lower_wick / atr_b
                quality += min(int(wick_ratio * 20), 20)
            # Hacim spike (0-20)
            if b < len(vol_sma20):
                quality += _vol_score(vol[b], vol_sma20[b], 20)
            # Recovery bar quality (0-20)
            if recovery_bar < n and close[recovery_bar] > open_[recovery_bar]:
                rec_body = close[recovery_bar] - open_[recovery_bar]
                quality += min(int(rec_body / atr_b * 20), 20) if atr_b > 0 else 12
            # Sweep shallowness (0-20)
            if atr_b > 0:
                depth_ratio = sweep_depth / atr_b
                if depth_ratio <= 0.3:
                    quality += 20
                elif depth_ratio <= 0.5:
                    quality += 15
                elif depth_ratio <= 1.0:
                    quality += 8

            stop_price = low[b] - 0.3 * atr_b
            risk = close[recovery_bar] - stop_price
            target_price = close[recovery_bar] + 2.0 * risk

            signals.append(PatternSignal(
                bar_idx=recovery_bar,
                direction='BUY',
                pattern='FAKEOUT_V1',
                key_level=level,
                quality=min(quality, 100),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'sweep_low': float(low[b]),
                    'sweep_depth_atr': round(sweep_depth / atr_b, 2),
                    'swing_low_idx': sl.idx,
                    'sweep_bar': b,
                    'recovery_bar': recovery_bar,
                    'wick_rejection': wick_rejection,
                },
            ))
            break  # Bu swing low icin ilk fakeout yeterli

    # --- BEARISH FAKEOUT ---
    swing_highs = [p for p in pivots if p.ptype == 'high']
    for sh in swing_highs:
        level = sh.price
        search_start = sh.confirm_idx + 1
        search_end = min(search_start + 30, n)

        for b in range(search_start, search_end):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if atr_b <= 0:
                continue

            if high[b] <= level:
                continue

            sweep_depth = high[b] - level
            if sweep_depth > 2.0 * atr_b:
                break

            recovered = False
            recovery_bar = b
            for rb in range(b, min(b + max_bars + 1, n)):
                if close[rb] < level:
                    recovered = True
                    recovery_bar = rb
                    break

            if not recovered:
                continue

            body = abs(close[b] - open_[b])
            upper_wick = high[b] - max(close[b], open_[b])

            # Kalite skoru (gradyan — max ~80)
            quality = 0
            # Rejection wick (0-20)
            if atr_b > 0:
                wick_ratio = upper_wick / atr_b
                quality += min(int(wick_ratio * 20), 20)
            # Hacim spike (0-20)
            if b < len(vol_sma20):
                quality += _vol_score(vol[b], vol_sma20[b], 20)
            # Recovery bar quality (0-20)
            if recovery_bar < n and close[recovery_bar] < open_[recovery_bar]:
                rec_body = open_[recovery_bar] - close[recovery_bar]
                quality += min(int(rec_body / atr_b * 20), 20) if atr_b > 0 else 12
            # Sweep shallowness (0-20)
            if atr_b > 0:
                depth_ratio = sweep_depth / atr_b
                if depth_ratio <= 0.3:
                    quality += 20
                elif depth_ratio <= 0.5:
                    quality += 15
                elif depth_ratio <= 1.0:
                    quality += 8

            stop_price = high[b] + 0.3 * atr_b
            risk = stop_price - close[recovery_bar]
            target_price = close[recovery_bar] - 2.0 * risk

            signals.append(PatternSignal(
                bar_idx=recovery_bar,
                direction='SELL',
                pattern='FAKEOUT_V1',
                key_level=level,
                quality=min(quality, 100),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'sweep_high': float(high[b]),
                    'sweep_depth_atr': round(sweep_depth / atr_b, 2),
                    'swing_high_idx': sh.idx,
                    'sweep_bar': b,
                    'recovery_bar': recovery_bar,
                },
            ))
            break

    return signals


# =============================================================================
# 5. FAKEOUT V2 (Fakeout + CHoCH)
# =============================================================================

def detect_fakeout_v2(df, pivots, atr):
    """
    Fakeout V2 = Fakeout V1 + yapi degisikligi (CHoCH).
    V1'den daha guclu sinyal.
    """
    v1_signals = detect_fakeout_v1(df, pivots, atr)
    signals = []
    n = len(df)
    close = df['close'].values

    for sig in v1_signals:
        recovery_bar = sig.bar_idx
        direction = sig.direction

        # CHoCH kontrolu: recovery sonrasi yapi degisimi
        # Bullish fakeout + CHoCH: recovery sonrasi fiyat onceki LH'yi kirsin
        # Bearish fakeout + CHoCH: recovery sonrasi fiyat onceki HL'yi kirsin
        has_choch = False

        if direction == 'BUY':
            # Recovery sonrasi: onceki LH'yi yukari kirdi mi?
            lh_pivots = [p for p in pivots if p.label == 'LH' and p.confirm_idx < recovery_bar]
            if lh_pivots:
                last_lh = lh_pivots[-1]
                # Recovery bar veya sonraki 5 bar icinde LH kirilmis mi?
                check_end = min(recovery_bar + 6, n)
                for b in range(recovery_bar, check_end):
                    if close[b] > last_lh.price:
                        has_choch = True
                        break
        else:  # SELL
            hl_pivots = [p for p in pivots if p.label == 'HL' and p.confirm_idx < recovery_bar]
            if hl_pivots:
                last_hl = hl_pivots[-1]
                check_end = min(recovery_bar + 6, n)
                for b in range(recovery_bar, check_end):
                    if close[b] < last_hl.price:
                        has_choch = True
                        break

        if has_choch:
            # V2 olarak yeniden olustur, kaliteye bonus ekle
            v2_quality = min(sig.quality + 15, 100)
            signals.append(PatternSignal(
                bar_idx=sig.bar_idx,
                direction=sig.direction,
                pattern='FAKEOUT_V2',
                key_level=sig.key_level,
                quality=v2_quality,
                stop=sig.stop,
                target=sig.target,
                details={**sig.details, 'choch': True},
            ))

    return signals


# =============================================================================
# 6. FLAG BREAKOUT
# =============================================================================

def detect_flag_b(df, pivots, atr):
    """
    Flag Breakout pattern tespiti.

    Bullish: Guclu impulse yukari → konsolidasyon (%20-55 geri cekilme) → breakout yukari
    Bearish: Guclu impulse asagi → konsolidasyon → breakout asagi
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    vol = df['volume'].values
    vol_sma20 = sma(df['volume'], 20).values
    cfg = SMC_CFG

    # --- BULLISH FLAG ---
    # Ardisik pivot low → pivot high ciftleri: impulse + retrace
    highs = _get_pivots_of_type(pivots, 'high')
    lows = _get_pivots_of_type(pivots, 'low')

    for ph in highs:
        if ph.label not in ('HH', 'H'):
            continue
        impulse_end = ph.price
        impulse_end_idx = ph.idx

        # Bu HH'dan onceki son pivot low = impulse baslangici
        preceding_lows = [p for p in lows if p.confirm_idx < ph.confirm_idx]
        if not preceding_lows:
            continue
        impulse_start_pivot = preceding_lows[-1]
        impulse_start = impulse_start_pivot.price

        impulse_size = impulse_end - impulse_start
        atr_at_peak = atr.iloc[ph.idx] if ph.idx < n else atr.iloc[-1]
        if atr_at_peak <= 0:
            continue

        # Impulse min 2 ATR olmali
        if impulse_size < cfg['flag_min_impulse_atr'] * atr_at_peak:
            continue

        # Konsolidasyon donemi: HH'dan sonraki barlar
        consol_start = ph.confirm_idx + 1
        if consol_start >= n:
            continue

        # Konsolidasyon araligi bul (max 20 bar)
        consol_end = min(consol_start + 20, n)
        consol_highs = high[consol_start:consol_end]
        consol_lows = low[consol_start:consol_end]

        if len(consol_highs) < 3:
            continue

        flag_high = consol_highs.max()
        flag_low = consol_lows.min()

        # Retrace hesapla
        retrace_pct = (impulse_end - flag_low) / impulse_size if impulse_size > 0 else 999

        if retrace_pct < cfg['flag_min_retrace_pct'] or retrace_pct > cfg['flag_max_retrace_pct']:
            continue

        # Breakout kontrolu
        for b in range(consol_start + 3, consol_end):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            body = close[b] - open_[b]

            # Breakout: close > konsolidasyon high'i AND body > 0.5*ATR
            if close[b] > flag_high and body > 0.5 * atr_b:
                # Kalite (gradyan — max ~80)
                quality = 0
                # Impulse temizligi (0-20)
                imp_atr = impulse_size / atr_at_peak if atr_at_peak > 0 else 0
                quality += min(int((imp_atr - 1.5) * 13), 20) if imp_atr > 1.5 else 0
                # Retrace ideal araligi (0-20) — Fibo ~0.382
                fibo_dist = abs(retrace_pct - 0.382)
                if fibo_dist <= 0.05:
                    quality += 20
                elif fibo_dist <= 0.10:
                    quality += 15
                elif fibo_dist <= 0.17:
                    quality += 8
                # Breakout hacmi (0-20)
                if b < len(vol_sma20):
                    quality += _vol_score(vol[b], vol_sma20[b], 20)
                # Flag suresi (0-20)
                flag_bars = b - consol_start
                if 4 <= flag_bars <= 7:
                    quality += 20
                elif 3 <= flag_bars <= 10:
                    quality += 15
                elif flag_bars <= 15:
                    quality += 8

                stop_price = flag_low - 0.3 * atr_b
                risk = close[b] - stop_price
                target_price = close[b] + 2.0 * risk

                signals.append(PatternSignal(
                    bar_idx=b,
                    direction='BUY',
                    pattern='FLAG_B',
                    key_level=flag_high,
                    quality=min(quality, 100),
                    stop=round(stop_price, 4),
                    target=round(target_price, 4),
                    details={
                        'impulse_start': impulse_start,
                        'impulse_end': impulse_end,
                        'impulse_atr': round(impulse_size / atr_at_peak, 2),
                        'retrace_pct': round(retrace_pct, 3),
                        'flag_bars': flag_bars,
                        'flag_high': float(flag_high),
                        'flag_low': float(flag_low),
                    },
                ))
                break  # Ilk breakout yeterli

    # --- BEARISH FLAG ---
    for pl in lows:
        if pl.label not in ('LL', 'L'):
            continue
        impulse_end = pl.price
        impulse_end_idx = pl.idx

        preceding_highs = [p for p in highs if p.confirm_idx < pl.confirm_idx]
        if not preceding_highs:
            continue
        impulse_start_pivot = preceding_highs[-1]
        impulse_start = impulse_start_pivot.price

        impulse_size = impulse_start - impulse_end  # asagi = pozitif
        atr_at_bottom = atr.iloc[pl.idx] if pl.idx < n else atr.iloc[-1]
        if atr_at_bottom <= 0:
            continue

        if impulse_size < cfg['flag_min_impulse_atr'] * atr_at_bottom:
            continue

        consol_start = pl.confirm_idx + 1
        if consol_start >= n:
            continue

        consol_end = min(consol_start + 20, n)
        consol_highs = high[consol_start:consol_end]
        consol_lows = low[consol_start:consol_end]

        if len(consol_highs) < 3:
            continue

        flag_high = consol_highs.max()
        flag_low = consol_lows.min()

        retrace_pct = (flag_high - impulse_end) / impulse_size if impulse_size > 0 else 999

        if retrace_pct < cfg['flag_min_retrace_pct'] or retrace_pct > cfg['flag_max_retrace_pct']:
            continue

        for b in range(consol_start + 3, consol_end):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            body = open_[b] - close[b]  # bearish body

            if close[b] < flag_low and body > 0.5 * atr_b:
                # Kalite (gradyan — max ~80)
                quality = 0
                # Impulse temizligi (0-20)
                imp_atr = impulse_size / atr_at_bottom if atr_at_bottom > 0 else 0
                quality += min(int((imp_atr - 1.5) * 13), 20) if imp_atr > 1.5 else 0
                # Retrace ideal araligi (0-20)
                fibo_dist = abs(retrace_pct - 0.382)
                if fibo_dist <= 0.05:
                    quality += 20
                elif fibo_dist <= 0.10:
                    quality += 15
                elif fibo_dist <= 0.17:
                    quality += 8
                # Breakout hacmi (0-20)
                if b < len(vol_sma20):
                    quality += _vol_score(vol[b], vol_sma20[b], 20)
                # Flag suresi (0-20)
                flag_bars = b - consol_start
                if 4 <= flag_bars <= 7:
                    quality += 20
                elif 3 <= flag_bars <= 10:
                    quality += 15
                elif flag_bars <= 15:
                    quality += 8

                stop_price = flag_high + 0.3 * atr_b
                risk = stop_price - close[b]
                target_price = close[b] - 2.0 * risk

                signals.append(PatternSignal(
                    bar_idx=b,
                    direction='SELL',
                    pattern='FLAG_B',
                    key_level=flag_low,
                    quality=min(quality, 100),
                    stop=round(stop_price, 4),
                    target=round(target_price, 4),
                    details={
                        'impulse_start': impulse_start,
                        'impulse_end': impulse_end,
                        'impulse_atr': round(impulse_size / atr_at_bottom, 2),
                        'retrace_pct': round(retrace_pct, 3),
                        'flag_bars': flag_bars,
                        'flag_high': float(flag_high),
                        'flag_low': float(flag_low),
                    },
                ))
                break

    return signals


# =============================================================================
# 7. 3-DRIVE (Exhaustion)
# =============================================================================

def detect_3drive(df, pivots, atr):
    """
    3-Drive pattern — momentum tukenmesi.

    Bearish: Uc ardisik HH, her drive'in boyutu kuculuyor → SELL
    Bullish: Uc ardisik LL, her drive'in boyutu kuculuyor → BUY
    """
    signals = []
    n = len(df)
    close = df['close'].values
    vol = df['volume'].values
    rsi = _calc_rsi(df['close']).values

    highs = _get_pivots_of_type(pivots, 'high')
    lows = _get_pivots_of_type(pivots, 'low')

    # --- BEARISH 3-DRIVE (uc ardisik HH) ---
    hh_pivots = [p for p in highs if p.label == 'HH']
    for i in range(len(hh_pivots) - 2):
        hh1 = hh_pivots[i]
        hh2 = hh_pivots[i + 1]
        hh3 = hh_pivots[i + 2]

        # Her HH'dan sonraki ilk pivot low'u bul (drive baslangici)
        def _find_low_after(pivot_h, all_lows):
            for lp in all_lows:
                if lp.confirm_idx > pivot_h.confirm_idx:
                    return lp
            return None

        def _find_low_before(pivot_h, all_lows):
            candidates = [lp for lp in all_lows if lp.idx < pivot_h.idx]
            return candidates[-1] if candidates else None

        low_before_hh1 = _find_low_before(hh1, lows)
        low_after_hh1 = _find_low_after(hh1, lows)
        low_after_hh2 = _find_low_after(hh2, lows)

        if low_before_hh1 is None or low_after_hh1 is None or low_after_hh2 is None:
            continue

        drive1 = hh1.price - low_before_hh1.price
        drive2 = hh2.price - low_after_hh1.price
        drive3 = hh3.price - low_after_hh2.price

        if drive1 <= 0 or drive2 <= 0 or drive3 <= 0:
            continue

        # Azalan momentum: drive2 < drive1 AND drive3 < drive2
        if drive2 >= drive1 or drive3 >= drive2:
            continue

        atr_at_hh3 = atr.iloc[hh3.idx] if hh3.idx < n else atr.iloc[-1]
        if atr_at_hh3 <= 0:
            continue

        # Kalite (gradyan — max ~75)
        quality = 0
        # Momentum azalmasi netligi (0-20)
        ratio = drive3 / drive1 if drive1 > 0 else 1
        if ratio < 0.3:
            quality += 20
        elif ratio < 0.5:
            quality += 15
        elif ratio < 0.75:
            quality += 8

        # RSI divergence: fiyat HH yaparken RSI dusuyor (0-25)
        if hh1.idx < len(rsi) and hh3.idx < len(rsi):
            rsi1 = rsi[hh1.idx]
            rsi3 = rsi[hh3.idx]
            if not (np.isnan(rsi1) or np.isnan(rsi3)) and rsi3 < rsi1:
                rsi_diff = rsi1 - rsi3
                if rsi_diff > 15:
                    quality += 25
                elif rsi_diff > 10:
                    quality += 18
                elif rsi_diff > 5:
                    quality += 12
                else:
                    quality += 5

        # Son drive wick rejection (0-15)
        if hh3.idx < n:
            bar_body = abs(close[hh3.idx] - df['open'].iloc[hh3.idx])
            bar_upper_wick = df['high'].iloc[hh3.idx] - max(close[hh3.idx], df['open'].iloc[hh3.idx])
            if bar_body > 0 and bar_upper_wick > bar_body:
                quality += min(int(bar_upper_wick / max(bar_body, 0.01) * 8), 15)

        # Hacim dususu (0-15)
        if hh1.idx < len(vol) and hh2.idx < len(vol) and hh3.idx < len(vol):
            v1 = vol[hh1.idx]
            v2 = vol[hh2.idx]
            v3 = vol[hh3.idx]
            if v1 > 0 and v2 < v1 and v3 < v2:
                quality += 15
            elif v1 > 0 and v3 < v1:
                quality += 8

        signal_bar = hh3.confirm_idx
        if signal_bar >= n:
            continue

        stop_price = hh3.price + 0.5 * atr_at_hh3
        risk = stop_price - close[signal_bar]
        target_price = close[signal_bar] - 2.0 * risk if risk > 0 else close[signal_bar] - 2 * atr_at_hh3

        signals.append(PatternSignal(
            bar_idx=signal_bar,
            direction='SELL',
            pattern='3DRIVE',
            key_level=hh3.price,
            quality=min(quality, 100),
            stop=round(stop_price, 4),
            target=round(target_price, 4),
            details={
                'hh1': hh1.price, 'hh2': hh2.price, 'hh3': hh3.price,
                'drive1': round(drive1, 4), 'drive2': round(drive2, 4),
                'drive3': round(drive3, 4),
                'ratio': round(ratio, 3),
            },
        ))

    # --- BULLISH 3-DRIVE (uc ardisik LL) ---
    ll_pivots = [p for p in lows if p.label == 'LL']
    for i in range(len(ll_pivots) - 2):
        ll1 = ll_pivots[i]
        ll2 = ll_pivots[i + 1]
        ll3 = ll_pivots[i + 2]

        def _find_high_after(pivot_l, all_highs):
            for hp in all_highs:
                if hp.confirm_idx > pivot_l.confirm_idx:
                    return hp
            return None

        def _find_high_before(pivot_l, all_highs):
            candidates = [hp for hp in all_highs if hp.idx < pivot_l.idx]
            return candidates[-1] if candidates else None

        high_before_ll1 = _find_high_before(ll1, highs)
        high_after_ll1 = _find_high_after(ll1, highs)
        high_after_ll2 = _find_high_after(ll2, highs)

        if high_before_ll1 is None or high_after_ll1 is None or high_after_ll2 is None:
            continue

        drive1 = high_before_ll1.price - ll1.price
        drive2 = high_after_ll1.price - ll2.price
        drive3 = high_after_ll2.price - ll3.price

        if drive1 <= 0 or drive2 <= 0 or drive3 <= 0:
            continue

        if drive2 >= drive1 or drive3 >= drive2:
            continue

        atr_at_ll3 = atr.iloc[ll3.idx] if ll3.idx < n else atr.iloc[-1]
        if atr_at_ll3 <= 0:
            continue

        # Kalite (gradyan — max ~75)
        quality = 0
        ratio = drive3 / drive1 if drive1 > 0 else 1
        if ratio < 0.3:
            quality += 20
        elif ratio < 0.5:
            quality += 15
        elif ratio < 0.75:
            quality += 8

        # RSI divergence (0-25)
        if ll1.idx < len(rsi) and ll3.idx < len(rsi):
            rsi1 = rsi[ll1.idx]
            rsi3 = rsi[ll3.idx]
            if not (np.isnan(rsi1) or np.isnan(rsi3)) and rsi3 > rsi1:
                rsi_diff = rsi3 - rsi1
                if rsi_diff > 15:
                    quality += 25
                elif rsi_diff > 10:
                    quality += 18
                elif rsi_diff > 5:
                    quality += 12
                else:
                    quality += 5

        # Wick rejection (0-15)
        if ll3.idx < n:
            bar_body = abs(close[ll3.idx] - df['open'].iloc[ll3.idx])
            bar_lower_wick = min(close[ll3.idx], df['open'].iloc[ll3.idx]) - df['low'].iloc[ll3.idx]
            if bar_body > 0 and bar_lower_wick > bar_body:
                quality += min(int(bar_lower_wick / max(bar_body, 0.01) * 8), 15)

        # Hacim dususu (0-15)
        if ll1.idx < len(vol) and ll2.idx < len(vol) and ll3.idx < len(vol):
            v1 = vol[ll1.idx]
            v2 = vol[ll2.idx]
            v3 = vol[ll3.idx]
            if v1 > 0 and v2 < v1 and v3 < v2:
                quality += 15
            elif v1 > 0 and v3 < v1:
                quality += 8

        signal_bar = ll3.confirm_idx
        if signal_bar >= n:
            continue

        stop_price = ll3.price - 0.5 * atr_at_ll3
        risk = close[signal_bar] - stop_price
        target_price = close[signal_bar] + 2.0 * risk if risk > 0 else close[signal_bar] + 2 * atr_at_ll3

        signals.append(PatternSignal(
            bar_idx=signal_bar,
            direction='BUY',
            pattern='3DRIVE',
            key_level=ll3.price,
            quality=min(quality, 100),
            stop=round(stop_price, 4),
            target=round(target_price, 4),
            details={
                'll1': ll1.price, 'll2': ll2.price, 'll3': ll3.price,
                'drive1': round(drive1, 4), 'drive2': round(drive2, 4),
                'drive3': round(drive3, 4),
                'ratio': round(ratio, 3),
            },
        ))

    return signals


# =============================================================================
# 8. COMPRESSION (Sikisma + Breakout)
# =============================================================================

def detect_compression(df, atr):
    """
    Compression (sikisma) pattern.

    High'lar azalan, low'lar artan trend → range daraliyor → breakout.
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    vol = df['volume'].values
    vol_sma20 = sma(df['volume'], 20).values
    min_bars = SMC_CFG['compression_min_bars']

    # Son 60 barlik pencerede compression ara
    search_start = max(min_bars + 5, n - 60)

    for end_bar in range(search_start, n):
        atr_b = atr.iloc[end_bar]
        if atr_b <= 0:
            continue

        # Geriye dogru compression penceresi bul
        best_comp_start = None
        best_comp_len = 0

        for comp_len in range(min_bars, min(20, end_bar)):
            start_bar = end_bar - comp_len

            window_highs = high[start_bar:end_bar + 1]
            window_lows = low[start_bar:end_bar + 1]

            if len(window_highs) < min_bars:
                continue

            # High'lar azaliyor mu? (linear regression slope < 0)
            x = np.arange(len(window_highs))
            if np.std(x) == 0:
                continue

            # Basit slope hesaplama
            h_slope = np.polyfit(x, window_highs, 1)[0]
            l_slope = np.polyfit(x, window_lows, 1)[0]

            # High'lar azalan, low'lar artan
            if h_slope < 0 and l_slope > 0:
                # Range daralma kontrolu
                first_range = window_highs[0] - window_lows[0]
                last_range = window_highs[-1] - window_lows[-1]

                if last_range < first_range * 0.7:  # en az %30 daralma
                    if comp_len > best_comp_len:
                        best_comp_start = start_bar
                        best_comp_len = comp_len

        if best_comp_start is None:
            continue

        # Compression bulundu, breakout bekle
        comp_end = end_bar
        comp_highs = high[best_comp_start:comp_end + 1]
        comp_lows = low[best_comp_start:comp_end + 1]
        comp_upper = comp_highs.max()
        comp_lower = comp_lows.min()

        # Breakout: bir sonraki barlarda
        for b in range(comp_end, min(comp_end + 4, n)):
            body = close[b] - open_[b]
            atr_at_b = atr.iloc[b] if b < n else atr.iloc[-1]

            if close[b] > comp_upper and abs(body) > 0.6 * atr_at_b:
                # Bullish breakout — kalite (gradyan — max ~80)
                quality = 0
                # Sikisma suresi (0-20)
                if best_comp_len >= 12:
                    quality += 20
                elif best_comp_len >= 10:
                    quality += 15
                elif best_comp_len >= 7:
                    quality += 10
                elif best_comp_len >= 5:
                    quality += 5
                # Range daralma hizi (0-20)
                first_r = high[best_comp_start] - low[best_comp_start]
                last_r = high[comp_end] - low[comp_end]
                if first_r > 0:
                    narrow_ratio = last_r / first_r
                    if narrow_ratio < 0.3:
                        quality += 20
                    elif narrow_ratio < 0.5:
                        quality += 15
                    elif narrow_ratio < 0.7:
                        quality += 8
                # Breakout body/ATR (0-20)
                if atr_at_b > 0:
                    body_ratio = abs(body) / atr_at_b
                    quality += min(int(body_ratio * 15), 20)
                # Hacim kontrast (0-20)
                comp_avg_vol = vol[best_comp_start:comp_end + 1].mean()
                if comp_avg_vol > 0:
                    quality += _vol_score(vol[b], comp_avg_vol, 20)

                comp_range = comp_upper - comp_lower
                stop_price = comp_lower - 0.3 * atr_at_b
                target_price = close[b] + comp_range

                signals.append(PatternSignal(
                    bar_idx=b,
                    direction='BUY',
                    pattern='COMPRESSION',
                    key_level=comp_upper,
                    quality=min(quality, 100),
                    stop=round(stop_price, 4),
                    target=round(target_price, 4),
                    details={
                        'comp_bars': best_comp_len,
                        'comp_upper': float(comp_upper),
                        'comp_lower': float(comp_lower),
                        'range': round(comp_range, 4),
                    },
                ))
                break

            elif close[b] < comp_lower and abs(body) > 0.6 * atr_at_b:
                # Bearish breakout — kalite (gradyan — max ~80)
                quality = 0
                # Sikisma suresi (0-20)
                if best_comp_len >= 12:
                    quality += 20
                elif best_comp_len >= 10:
                    quality += 15
                elif best_comp_len >= 7:
                    quality += 10
                elif best_comp_len >= 5:
                    quality += 5
                # Range daralma (0-20)
                first_r = high[best_comp_start] - low[best_comp_start]
                last_r = high[comp_end] - low[comp_end]
                if first_r > 0:
                    narrow_ratio = last_r / first_r
                    if narrow_ratio < 0.3:
                        quality += 20
                    elif narrow_ratio < 0.5:
                        quality += 15
                    elif narrow_ratio < 0.7:
                        quality += 8
                # Breakout body/ATR (0-20)
                if atr_at_b > 0:
                    body_ratio = abs(body) / atr_at_b
                    quality += min(int(body_ratio * 15), 20)
                # Hacim kontrast (0-20)
                comp_avg_vol = vol[best_comp_start:comp_end + 1].mean()
                if comp_avg_vol > 0:
                    quality += _vol_score(vol[b], comp_avg_vol, 20)

                comp_range = comp_upper - comp_lower
                stop_price = comp_upper + 0.3 * atr_at_b
                target_price = close[b] - comp_range

                signals.append(PatternSignal(
                    bar_idx=b,
                    direction='SELL',
                    pattern='COMPRESSION',
                    key_level=comp_lower,
                    quality=min(quality, 100),
                    stop=round(stop_price, 4),
                    target=round(target_price, 4),
                    details={
                        'comp_bars': best_comp_len,
                        'comp_upper': float(comp_upper),
                        'comp_lower': float(comp_lower),
                        'range': round(comp_range, 4),
                    },
                ))
                break

    return signals


# =============================================================================
# 9. CAN-CAN (S/R Flip)
# =============================================================================

def detect_cancan(df, pivots, atr):
    """
    Can-Can (Support/Resistance Flip) pattern.

    Bearish: Destek kirilir → eski destek yeni direnc → retest & red → SELL
    Bullish: Direnc kirilir → eski direnc yeni destek → retest & red → BUY
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    cfg = SMC_CFG

    # --- DESTEK seviyeleri bul (2+ pivot low ayni yerde) ---
    swing_lows = [p for p in pivots if p.ptype == 'low']
    support_levels = _find_cluster_levels(swing_lows, atr, cfg['same_level_atr'])

    # --- DIRENC seviyeleri bul (2+ pivot high ayni yerde) ---
    swing_highs = [p for p in pivots if p.ptype == 'high']
    resistance_levels = _find_cluster_levels(swing_highs, atr, cfg['same_level_atr'])

    # --- BEARISH CAN-CAN (destek kirilmasi → retest → red) ---
    for level_info in support_levels:
        level = level_info['price']
        touch_count = level_info['count']
        last_touch_idx = level_info['last_confirm_idx']

        # Kirilma: close < destek - 0.3*ATR
        break_bar = None
        for b in range(last_touch_idx + 1, min(last_touch_idx + 30, n)):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if close[b] < level - cfg['cancan_tolerance_atr'] * atr_b:
                break_bar = b
                break

        if break_bar is None:
            continue

        # Retest: fiyat seviyeye geri donup reddedilsin
        atr_at_break = atr.iloc[break_bar]
        for b in range(break_bar + 1, min(break_bar + 20, n)):
            # Fiyat seviyeye yaklasti mi?
            if abs(high[b] - level) <= 0.5 * atr_at_break or high[b] >= level:
                # Rejection: close < seviye
                if close[b] < level:
                    # Wick rejection kontrolu
                    wick = high[b] - max(close[b], df['open'].iloc[b])

                    # Kalite (gradyan — max ~75)
                    quality = 0
                    # Touch count (0-20)
                    if touch_count >= 4:
                        quality += 20
                    elif touch_count >= 3:
                        quality += 15
                    else:
                        quality += 8
                    # Kirilma netligi (0-20)
                    break_depth = level - close[break_bar]
                    if atr_at_break > 0:
                        bd_ratio = break_depth / atr_at_break
                        quality += min(int(bd_ratio * 12), 20)
                    # Retest hassasiyeti (0-20)
                    retest_dist = abs(high[b] - level)
                    quality += _proximity_score(retest_dist, atr_at_break, 20)
                    # Wick rejection (0-15)
                    bar_body = abs(close[b] - df['open'].iloc[b])
                    if bar_body > 0 and wick > bar_body:
                        quality += min(int(wick / bar_body * 8), 15)

                    stop_price = level + 0.5 * atr_at_break
                    risk = stop_price - close[b]
                    target_price = close[b] - 2.0 * risk if risk > 0 else close[b] - 2 * atr_at_break

                    signals.append(PatternSignal(
                        bar_idx=b,
                        direction='SELL',
                        pattern='CANCAN',
                        key_level=level,
                        quality=min(quality, 100),
                        stop=round(stop_price, 4),
                        target=round(target_price, 4),
                        details={
                            'level': level,
                            'touch_count': touch_count,
                            'break_bar': break_bar,
                            'retest_bar': b,
                            'flip': 'S→R',
                        },
                    ))
                    break

    # --- BULLISH CAN-CAN (direnc kirilmasi → retest → BUY) ---
    for level_info in resistance_levels:
        level = level_info['price']
        touch_count = level_info['count']
        last_touch_idx = level_info['last_confirm_idx']

        break_bar = None
        for b in range(last_touch_idx + 1, min(last_touch_idx + 30, n)):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if close[b] > level + cfg['cancan_tolerance_atr'] * atr_b:
                break_bar = b
                break

        if break_bar is None:
            continue

        atr_at_break = atr.iloc[break_bar]
        for b in range(break_bar + 1, min(break_bar + 20, n)):
            if abs(low[b] - level) <= 0.5 * atr_at_break or low[b] <= level:
                if close[b] > level:
                    wick = min(close[b], df['open'].iloc[b]) - low[b]

                    # Kalite (gradyan — max ~75)
                    quality = 0
                    # Touch count (0-20)
                    if touch_count >= 4:
                        quality += 20
                    elif touch_count >= 3:
                        quality += 15
                    else:
                        quality += 8
                    # Kirilma netligi (0-20)
                    break_depth = close[break_bar] - level
                    if atr_at_break > 0:
                        bd_ratio = break_depth / atr_at_break
                        quality += min(int(bd_ratio * 12), 20)
                    # Retest hassasiyeti (0-20)
                    retest_dist = abs(low[b] - level)
                    quality += _proximity_score(retest_dist, atr_at_break, 20)
                    # Wick rejection (0-15)
                    bar_body = abs(close[b] - df['open'].iloc[b])
                    if bar_body > 0 and wick > bar_body:
                        quality += min(int(wick / bar_body * 8), 15)

                    stop_price = level - 0.5 * atr_at_break
                    risk = close[b] - stop_price
                    target_price = close[b] + 2.0 * risk if risk > 0 else close[b] + 2 * atr_at_break

                    signals.append(PatternSignal(
                        bar_idx=b,
                        direction='BUY',
                        pattern='CANCAN',
                        key_level=level,
                        quality=min(quality, 100),
                        stop=round(stop_price, 4),
                        target=round(target_price, 4),
                        details={
                            'level': level,
                            'touch_count': touch_count,
                            'break_bar': break_bar,
                            'retest_bar': b,
                            'flip': 'R→S',
                        },
                    ))
                    break

    return signals


def _find_cluster_levels(pivot_list, atr, tolerance_atr):
    """
    Pivot listesinden cluster (ayni seviye) gruplarini bul.
    Returns: list of {'price': float, 'count': int, 'last_confirm_idx': int}
    """
    if not pivot_list:
        return []

    levels = []
    used = set()

    for i, p1 in enumerate(pivot_list):
        if i in used:
            continue
        cluster = [p1]
        used.add(i)

        atr_val = atr.iloc[p1.idx] if p1.idx < len(atr) else atr.iloc[-1]
        if atr_val <= 0:
            continue

        for j, p2 in enumerate(pivot_list):
            if j in used or j <= i:
                continue
            if _same_level(p1.price, p2.price, atr_val, tolerance_atr):
                cluster.append(p2)
                used.add(j)

        if len(cluster) >= 2:
            avg_price = np.mean([p.price for p in cluster])
            last_idx = max(p.confirm_idx for p in cluster)
            levels.append({
                'price': avg_price,
                'count': len(cluster),
                'last_confirm_idx': last_idx,
            })

    return levels


# =============================================================================
# 10. 2R/2S FAKEOUT (Double Resistance/Support Fakeout)
# =============================================================================

def detect_2r2s_fakeout(df, pivots, atr):
    """
    2R Fakeout (Double Resistance) — Bearish:
      Ayni seviyeye 2 kez dokunmus pivot high → ucuncu yaklasimda sweep → geri donus → SELL
    2S Fakeout (Double Support) — Bullish:
      Ayni seviyeye 2 kez dokunmus pivot low → ucuncu yaklasimda sweep → geri donus → BUY
    """
    signals = []
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    cfg = SMC_CFG
    max_bars = cfg['fakeout_max_bars']

    # --- 2R FAKEOUT (Bearish) ---
    swing_highs = [p for p in pivots if p.ptype == 'high']
    resistance_levels = _find_cluster_levels(swing_highs, atr, cfg['same_level_atr'])

    for level_info in resistance_levels:
        level = level_info['price']
        touch_count = level_info['count']
        last_touch_idx = level_info['last_confirm_idx']

        # Ucuncu yaklasim: fiyat seviyeyi kisa sureli assin, geri donusse
        for b in range(last_touch_idx + 1, min(last_touch_idx + 25, n)):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if atr_b <= 0:
                continue

            # Sweep: high seviyenin ustune gecti
            if high[b] <= level:
                continue

            sweep_depth = high[b] - level
            if sweep_depth > 2.0 * atr_b:
                break  # gercek kirilim

            # Geri donus
            recovered = False
            recovery_bar = b
            for rb in range(b, min(b + max_bars + 1, n)):
                if close[rb] < level:
                    recovered = True
                    recovery_bar = rb
                    break

            if not recovered:
                continue

            # Kalite (gradyan — max ~75)
            quality = 0
            # Touch count (0-20)
            if touch_count >= 4:
                quality += 20
            elif touch_count >= 3:
                quality += 15
            else:
                quality += 8
            # Sweep shallowness (0-20)
            if atr_b > 0:
                depth_ratio = sweep_depth / atr_b
                if depth_ratio <= 0.3:
                    quality += 20
                elif depth_ratio <= 0.5:
                    quality += 15
                elif depth_ratio <= 1.0:
                    quality += 10
            # Rejection wick (0-20)
            upper_wick = high[b] - max(close[b], open_[b])
            bar_body = abs(close[b] - open_[b])
            if bar_body > 0 and upper_wick > bar_body:
                quality += min(int(upper_wick / bar_body * 10), 20)
            # Hacim (0-15)
            vol_sma20 = sma(df['volume'], 20).values
            if b < len(vol_sma20):
                quality += _vol_score(df['volume'].iloc[b], vol_sma20[b], 15)

            stop_price = high[b] + 0.3 * atr_b
            risk = stop_price - close[recovery_bar]
            target_price = close[recovery_bar] - 2.0 * risk if risk > 0 else close[recovery_bar] - 2 * atr_b

            signals.append(PatternSignal(
                bar_idx=recovery_bar,
                direction='SELL',
                pattern='2R_FAKEOUT',
                key_level=level,
                quality=min(quality, 100),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'level': level,
                    'touch_count': touch_count,
                    'sweep_high': float(high[b]),
                    'sweep_depth_atr': round(sweep_depth / atr_b, 2),
                },
            ))
            break

    # --- 2S FAKEOUT (Bullish) ---
    swing_lows = [p for p in pivots if p.ptype == 'low']
    support_levels = _find_cluster_levels(swing_lows, atr, cfg['same_level_atr'])

    for level_info in support_levels:
        level = level_info['price']
        touch_count = level_info['count']
        last_touch_idx = level_info['last_confirm_idx']

        for b in range(last_touch_idx + 1, min(last_touch_idx + 25, n)):
            atr_b = atr.iloc[b] if b < n else atr.iloc[-1]
            if atr_b <= 0:
                continue

            if low[b] >= level:
                continue

            sweep_depth = level - low[b]
            if sweep_depth > 2.0 * atr_b:
                break

            recovered = False
            recovery_bar = b
            for rb in range(b, min(b + max_bars + 1, n)):
                if close[rb] > level:
                    recovered = True
                    recovery_bar = rb
                    break

            if not recovered:
                continue

            # Kalite (gradyan — max ~75)
            quality = 0
            # Touch count (0-20)
            if touch_count >= 4:
                quality += 20
            elif touch_count >= 3:
                quality += 15
            else:
                quality += 8
            # Sweep shallowness (0-20)
            if atr_b > 0:
                depth_ratio = sweep_depth / atr_b
                if depth_ratio <= 0.3:
                    quality += 20
                elif depth_ratio <= 0.5:
                    quality += 15
                elif depth_ratio <= 1.0:
                    quality += 10
            # Rejection wick (0-20)
            lower_wick = min(close[b], open_[b]) - low[b]
            bar_body = abs(close[b] - open_[b])
            if bar_body > 0 and lower_wick > bar_body:
                quality += min(int(lower_wick / bar_body * 10), 20)
            # Hacim (0-15)
            vol_sma20 = sma(df['volume'], 20).values
            if b < len(vol_sma20):
                quality += _vol_score(df['volume'].iloc[b], vol_sma20[b], 15)

            stop_price = low[b] - 0.3 * atr_b
            risk = close[recovery_bar] - stop_price
            target_price = close[recovery_bar] + 2.0 * risk if risk > 0 else close[recovery_bar] + 2 * atr_b

            signals.append(PatternSignal(
                bar_idx=recovery_bar,
                direction='BUY',
                pattern='2S_FAKEOUT',
                key_level=level,
                quality=min(quality, 100),
                stop=round(stop_price, 4),
                target=round(target_price, 4),
                details={
                    'level': level,
                    'touch_count': touch_count,
                    'sweep_low': float(low[b]),
                    'sweep_depth_atr': round(sweep_depth / atr_b, 2),
                },
            ))
            break

    return signals


# =============================================================================
# 11. ANA TARAMA FONKSIYONU
# =============================================================================

def scan_patterns(df, scan_bars=None):
    """
    Tum SMC pattern'leri tara.

    Args:
        df: DataFrame (lowercase kolonlar: close, high, low, open, volume)
        scan_bars: Son kac bar taranacak (default: SMC_CFG['scan_bars'])

    Returns: list[PatternSignal] kaliteye gore sirali.
    """
    if scan_bars is None:
        scan_bars = SMC_CFG['scan_bars']

    atr = _calc_atr(df, SMC_CFG['atr_len'])
    pivots = detect_structure(df, SMC_CFG['pivot_lb'])

    signals = []
    signals += detect_qm(df, pivots, atr)
    signals += detect_fakeout_v1(df, pivots, atr)
    signals += detect_fakeout_v2(df, pivots, atr)
    signals += detect_flag_b(df, pivots, atr)
    signals += detect_3drive(df, pivots, atr)
    signals += detect_compression(df, atr)
    signals += detect_cancan(df, pivots, atr)
    signals += detect_2r2s_fakeout(df, pivots, atr)

    # Son scan_bars icindeki sinyalleri filtrele
    last = len(df) - 1
    cutoff = last - scan_bars
    signals = [s for s in signals if s.bar_idx > cutoff]

    # Duplikatlari kaldir: ayni bar + ayni pattern + ayni yon
    seen = set()
    unique = []
    for s in signals:
        key = (s.bar_idx, s.pattern, s.direction)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    signals = unique

    # Bias alignment post-processing: +15 uyumlu, -15 counter-trend
    bias = get_market_bias(pivots)
    for s in signals:
        if bias == 'bullish':
            if s.direction == 'BUY':
                s.quality = min(s.quality + 15, 100)
            else:
                s.quality = max(s.quality - 15, 0)
        elif bias == 'bearish':
            if s.direction == 'SELL':
                s.quality = min(s.quality + 15, 100)
            else:
                s.quality = max(s.quality - 15, 0)
        # neutral: degisiklik yok
        s.details['bias'] = bias

    # Kaliteye gore sirala
    signals.sort(key=lambda s: s.quality, reverse=True)
    return signals
