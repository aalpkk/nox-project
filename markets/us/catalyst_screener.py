"""
NOX Project — US Catalyst Screener
ABD borsasında %30-60 potansiyelli hisseleri önceden tespit eden 7 modüllü tarama.

Modüller:
1. Unusual Volume — hacim anomalisi (reaktif, bugün patlayan)
2. Accumulation — sessiz birikim tespiti (prediktif, kurumsal alım)
3. Short Squeeze — kısa pozisyon sıkışma adayları
4. Insider Buying — yönetici alım kümeleri
5. Biotech Catalyst — biyotek katalizör adayları + FDA/PDUFA takvimi
6. Earnings Momentum — bilanço momentum
7. Technical Breakout — teknik kırılım setupları (SETUP ağırlıklı)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from markets.us.config import (
    MIN_AVG_VOLUME_USD, MIN_PRICE_USD, MIN_CHANGE_PCT,
    RVOL_UNUSUAL, RVOL_HIGH, VOL_SMA_PERIOD,
    SHORT_FLOAT_MIN, FLOAT_SHARES_MAX, DAYS_TO_COVER_MIN,
    INSIDER_LOOKBACK, INSIDER_MIN_BUYERS, INSIDER_MIN_VALUE,
    BIOTECH_MCAP_MAX,
    EARNINGS_WINDOW, EARNINGS_BB_WIDTH_MAX, EARNINGS_GAP_MIN,
    ATR_COMPRESS_RATIO, CONSOL_MIN_DAYS, BREAKOUT_VOL_MULT,
    ACCUM_RVOL_MIN, ACCUM_RVOL_MAX, ACCUM_WINDOW,
    ACCUM_MAX_DAILY_MOVE, ACCUM_RANGE_ATR_MULT,
    REGIME_BULL_THRESH, REGIME_NEUTRAL_THRESH,
)


# ══════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════

def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _bb_width_pct(close, n=20, mult=2.0):
    """Bollinger Band genişliği yüzde olarak."""
    mid = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return ((upper - lower) / mid * 100).iloc[-1] if len(close) >= n else None


def _rs_vs_spy(stock_close, spy_close, period=20):
    """Relative Strength vs SPY."""
    if len(stock_close) < period + 1 or len(spy_close) < period + 1:
        return None
    stock_ret = stock_close.iloc[-1] / stock_close.iloc[-period - 1] - 1
    spy_ret = spy_close.iloc[-1] / spy_close.iloc[-period - 1] - 1
    if spy_ret == 0:
        return None
    return round(stock_ret / spy_ret, 2)


def _dollar_volume(close, volume, n=20):
    """Ortalama günlük dolar hacmi."""
    dv = close * volume
    return _sma(dv, n).iloc[-1] if len(close) >= n else 0


# ══════════════════════════════════════════
# SPY REGIME
# ══════════════════════════════════════════

def compute_spy_regime(spy_df):
    """SPY rejim tespiti: BULL / NEUTRAL / RISK_OFF.

    3 bileşen (BIST regime_transition.py pattern):
    - Trend (0-2): close > EMA21, EMA21 > EMA55
    - Momentum (0-2): RSI > 50, MACD histogram > 0
    - Breadth (0-2): close > SMA200, son 10g daha yüksek kapanış sayısı >= 5

    Toplam (0-6):
    - >= 5: BULL
    - >= 3: NEUTRAL
    - < 3: RISK_OFF
    """
    if spy_df is None or len(spy_df) < 200:
        return {'regime': 'NEUTRAL', 'score': 3, 'max_score': 6}

    close = spy_df['Close']
    cur = close.iloc[-1]

    # Trend (0-2)
    ema21 = _ema(close, 21).iloc[-1]
    ema55 = _ema(close, 55).iloc[-1]
    trend = (1 if cur > ema21 else 0) + (1 if ema21 > ema55 else 0)

    # Momentum (0-2): RSI > 50, MACD histogram > 0
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = (macd_line - macd_signal).iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1] if len(rs) >= 14 else 50

    momentum = (1 if rsi > 50 else 0) + (1 if macd_hist > 0 else 0)

    # Breadth (0-2): close > SMA200, son 10g'de >= 5 yüksek kapanış
    sma200 = _sma(close, 200).iloc[-1]
    above_sma200 = 1 if cur > sma200 else 0
    higher_closes = sum(1 for i in range(-10, 0) if close.iloc[i] > close.iloc[i - 1])
    breadth = above_sma200 + (1 if higher_closes >= 5 else 0)

    total = trend + momentum + breadth
    if total >= REGIME_BULL_THRESH:
        regime = 'BULL'
    elif total >= REGIME_NEUTRAL_THRESH:
        regime = 'NEUTRAL'
    else:
        regime = 'RISK_OFF'

    return {'regime': regime, 'score': total, 'max_score': 6}


# ══════════════════════════════════════════
# EXECUTION QUALITY HELPERS
# ══════════════════════════════════════════

def _gap_risk(df, lookback=20):
    """Son N günde max overnight gap yüzdesi."""
    if len(df) < lookback + 1:
        return 0.0
    opens = df['Open'].iloc[-lookback:]
    prev_closes = df['Close'].iloc[-lookback - 1:-1]
    gaps = (opens.values - prev_closes.values) / prev_closes.values * 100
    return round(max(abs(gaps.min()), abs(gaps.max())), 1) if len(gaps) > 0 else 0.0


def _spread_proxy(atr_val, close_val):
    """ATR/close % — efektif spread tahmini."""
    if close_val <= 0 or pd.isna(atr_val):
        return 0.0
    return round(atr_val / close_val * 100, 2)


# ══════════════════════════════════════════
# MODÜL 1: UNUSUAL VOLUME
# ══════════════════════════════════════════

def scan_unusual_volume(stock_dfs, spy_df=None):
    """Anormal hacim tespiti.

    Kriter:
    - RVOL >= 2.0 (volume / SMA20)
    - |change| > 3%
    - Price > $5
    - Dollar volume > $1M
    """
    results = []
    for ticker, df in stock_dfs.items():
        if len(df) < VOL_SMA_PERIOD + 5:
            continue

        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']

        cur_close = close.iloc[-1]
        cur_vol = volume.iloc[-1]

        if cur_close < MIN_PRICE_USD:
            continue

        avg_vol = _sma(volume, VOL_SMA_PERIOD).iloc[-1]
        if pd.isna(avg_vol) or avg_vol <= 0:
            continue

        rvol = cur_vol / avg_vol
        if rvol < RVOL_UNUSUAL:
            continue

        prev_close = close.iloc[-2]
        change_pct = (cur_close - prev_close) / prev_close * 100

        if abs(change_pct) < MIN_CHANGE_PCT:
            continue

        dv = _dollar_volume(close, volume, VOL_SMA_PERIOD)
        if dv < MIN_AVG_VOLUME_USD:
            continue

        # Ardışık yüksek hacim günleri (birikim tespiti)
        consec = 0
        for j in range(len(volume) - 1, max(len(volume) - 11, -1), -1):
            avg_j = _sma(volume, VOL_SMA_PERIOD).iloc[j] if j >= VOL_SMA_PERIOD else avg_vol
            if pd.notna(avg_j) and avg_j > 0 and volume.iloc[j] / avg_j >= 1.5:
                consec += 1
            else:
                break

        # Skor
        score = min(100, int(
            min(rvol / 5.0, 1.0) * 40 +
            min(abs(change_pct) / 10.0, 1.0) * 30 +
            min(consec / 5.0, 1.0) * 15 +
            min(dv / 50_000_000, 1.0) * 15
        ))

        rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None

        # Execution quality
        atr_14 = _atr(high, low, close, 14)
        atr_val = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else 0
        trigger = round(cur_close, 2)  # hemen
        stop = round(cur_close - 1.5 * atr_val, 2)
        inval = round(low.iloc[-5:].min(), 2)  # son 5g low
        risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0

        results.append({
            'module': 'VOLUME',
            'ticker': ticker,
            'close': round(cur_close, 2),
            'change_pct': round(change_pct, 2),
            'rvol': round(rvol, 1),
            'avg_volume': int(avg_vol),
            'dollar_vol': int(dv),
            'direction': 'UP' if change_pct > 0 else 'DOWN',
            'consecutive_days': consec,
            'score': score,
            'rs': rs,
            'trigger': trigger,
            'stop': stop,
            'invalidation': inval,
            'risk_pct': risk_pct,
            'spread_proxy': _spread_proxy(atr_val, cur_close),
            'gap_risk': _gap_risk(df),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 2: ACCUMULATION (SESSİZ BİRİKİM)
# ══════════════════════════════════════════

def scan_accumulation(stock_dfs, spy_df=None):
    """Sessiz kurumsal birikim tespiti (prediktif).

    Büyük bir hareket ÖNCE sessizce hacim artar ama fiyat hareket etmez.
    Bu pattern genellikle kurumsal alımı (institutional accumulation) gösterir.

    Kriter:
    - Son 10g ortalama RVOL 1.3-2.5 arası (yükselen ama patlamayan)
    - Son 10g fiyat range < 2x ATR (dar, sıkışmış)
    - Hiçbir tek gün >4% hareket yok (reaktif değil)
    - Hacim eğilimi yukarı (son 5g > ilk 5g)
    - Fiyat EMA21 civarında veya üstünde
    """
    results = []
    for ticker, df in stock_dfs.items():
        if len(df) < 60:
            continue

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        cur_close = close.iloc[-1]

        if cur_close < MIN_PRICE_USD:
            continue

        dv = _dollar_volume(close, volume, 20)
        if dv < MIN_AVG_VOLUME_USD:
            continue

        # Son 10 günün RVOL ortalaması
        avg_vol_20 = _sma(volume, 20)
        if pd.isna(avg_vol_20.iloc[-ACCUM_WINDOW]) or avg_vol_20.iloc[-ACCUM_WINDOW] <= 0:
            continue

        rvol_window = []
        for j in range(-ACCUM_WINDOW, 0):
            av = avg_vol_20.iloc[j]
            if pd.notna(av) and av > 0:
                rvol_window.append(volume.iloc[j] / av)
        if not rvol_window:
            continue
        avg_rvol = np.mean(rvol_window)

        if avg_rvol < ACCUM_RVOL_MIN or avg_rvol > ACCUM_RVOL_MAX:
            continue

        # Son 10g fiyat range kontrolü — dar olmalı
        recent_high = high.iloc[-ACCUM_WINDOW:].max()
        recent_low = low.iloc[-ACCUM_WINDOW:].min()
        atr_14 = _atr(high, low, close, 14)
        if pd.isna(atr_14.iloc[-1]) or atr_14.iloc[-1] <= 0:
            continue
        range_vs_atr = (recent_high - recent_low) / atr_14.iloc[-1]

        if range_vs_atr > ACCUM_RANGE_ATR_MULT:
            continue

        # Tek gün büyük hareket kontrolü — olmamalı
        daily_changes = close.iloc[-ACCUM_WINDOW:].pct_change().abs() * 100
        if daily_changes.max() > ACCUM_MAX_DAILY_MOVE:
            continue

        # Hacim eğilimi: son 5g > ilk 5g (yükselen trend)
        vol_first5 = volume.iloc[-ACCUM_WINDOW:-5].mean()
        vol_last5 = volume.iloc[-5:].mean()
        vol_slope = vol_last5 / vol_first5 if vol_first5 > 0 else 1.0

        # EMA21 civarında mı?
        ema21 = _ema(close, 21).iloc[-1]
        above_ema = cur_close >= ema21 * 0.97  # %3 altına kadar tolerans

        if not above_ema:
            continue

        # Toplam birikim gün sayısı (RVOL > 1.2 olan ardışık günler)
        accum_days = 0
        for j in range(len(volume) - 1, max(len(volume) - 31, -1), -1):
            av = avg_vol_20.iloc[j] if j < len(avg_vol_20) else None
            if pd.notna(av) and av > 0 and volume.iloc[j] / av >= 1.2:
                accum_days += 1
            else:
                break

        # Skor
        s_rvol = min((avg_rvol - 1.0) / 1.0, 1.0) * 25  # 1.3-2.3 arası
        s_range = max(0, 1.0 - range_vs_atr / ACCUM_RANGE_ATR_MULT) * 25  # dar = iyi
        s_slope = min(vol_slope / 1.5, 1.0) * 25  # yükselen hacim
        s_days = min(accum_days / 15.0, 1.0) * 25  # uzun birikim = iyi
        score = int(s_rvol + s_range + s_slope + s_days)

        rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None

        # Execution quality
        atr_val = atr_14.iloc[-1]
        trigger = round(high.iloc[-ACCUM_WINDOW:].max(), 2)  # son 10g high kırılması
        stop = round(low.iloc[-ACCUM_WINDOW:].min(), 2)  # son 10g low
        inval = round(ema21 - atr_val, 2) if pd.notna(atr_val) else round(ema21 * 0.97, 2)  # EMA21 altı
        risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0

        results.append({
            'module': 'ACCUM',
            'ticker': ticker,
            'close': round(cur_close, 2),
            'avg_rvol': round(avg_rvol, 2),
            'range_vs_atr': round(range_vs_atr, 2),
            'vol_slope': round(vol_slope, 2),
            'accum_days': accum_days,
            'range_pct': round((recent_high - recent_low) / recent_low * 100, 1),
            'dollar_vol': int(dv),
            'score': score,
            'rs': rs,
            'trigger': trigger,
            'stop': stop,
            'invalidation': inval,
            'risk_pct': risk_pct,
            'spread_proxy': _spread_proxy(atr_val, cur_close),
            'gap_risk': _gap_risk(df),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 3: SHORT SQUEEZE SETUP
# ══════════════════════════════════════════

def scan_short_squeeze(stock_dfs, ticker_info):
    """Short squeeze potansiyeli olan hisseler.

    Kriter:
    - Short % of float > 15%
    - Float < 100M
    - Days to cover > 3
    - Price > EMA21 (yukarı trend)
    - Volume artış trendi
    """
    results = []
    for ticker, info in ticker_info.items():
        short_pct = info.get('short_pct')
        if short_pct is None or short_pct * 100 < SHORT_FLOAT_MIN:
            continue

        short_pct_val = short_pct * 100  # 0.xx -> %xx

        float_shares = info.get('float_shares')
        short_ratio = info.get('short_ratio')  # days to cover

        df = stock_dfs.get(ticker)
        if df is None or len(df) < 30:
            continue

        close = df['Close']
        volume = df['Volume']
        cur_close = close.iloc[-1]

        if cur_close < MIN_PRICE_USD:
            continue

        # Trend check: close > EMA21
        ema21 = _ema(close, 21).iloc[-1]
        above_ema = cur_close > ema21

        # Volume trend: son 5g ort > 20g ort
        vol_5d = volume.iloc[-5:].mean()
        vol_20d = _sma(volume, 20).iloc[-1]
        vol_trend = vol_5d / vol_20d if pd.notna(vol_20d) and vol_20d > 0 else 1.0

        # Skor hesaplama
        s_short = min(short_pct_val / 40.0, 1.0) * 40
        s_dtc = min((short_ratio or 0) / 8.0, 1.0) * 20
        s_float = min(1.0 - (float_shares or FLOAT_SHARES_MAX) / FLOAT_SHARES_MAX, 0) * 20 \
            if float_shares and float_shares < FLOAT_SHARES_MAX else 0
        # Float küçükse puan yüksek
        if float_shares and float_shares > 0:
            s_float = max(0, min(1.0 - float_shares / FLOAT_SHARES_MAX, 1.0)) * 20
        s_vol = min(vol_trend / 2.0, 1.0) * 20

        score = int(s_short + s_dtc + s_float + s_vol)

        # Bonus: EMA üstü ise +10
        if above_ema:
            score = min(100, score + 10)

        mcap = info.get('market_cap')

        # Execution quality
        high = df['High']
        low = df['Low']
        atr_14 = _atr(high, low, close, 14)
        atr_val = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else 0
        trigger = round(cur_close, 2)  # hemen
        stop = round(ema21, 2)  # EMA21
        swing_low = round(low.iloc[-20:].min(), 2) if len(low) >= 20 else round(low.iloc[-5:].min(), 2)
        inval = swing_low  # son swing low
        risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 and trigger > stop else 0
        dv = _dollar_volume(close, volume, 20)

        results.append({
            'module': 'SQUEEZE',
            'ticker': ticker,
            'name': info.get('name', ticker),
            'close': round(cur_close, 2),
            'short_pct': round(short_pct_val, 1),
            'short_ratio': round(short_ratio, 1) if short_ratio else None,
            'float_shares': float_shares,
            'float_m': round(float_shares / 1e6, 1) if float_shares else None,
            'market_cap': mcap,
            'mcap_b': round(mcap / 1e9, 2) if mcap else None,
            'above_ema21': above_ema,
            'vol_trend': round(vol_trend, 2),
            'score': score,
            'sector': info.get('sector', ''),
            'dollar_vol': int(dv),
            'trigger': trigger,
            'stop': stop,
            'invalidation': inval,
            'risk_pct': risk_pct,
            'spread_proxy': _spread_proxy(atr_val, cur_close),
            'gap_risk': _gap_risk(df),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 3: INSIDER BUYING
# ══════════════════════════════════════════

def scan_insider_buying(stock_dfs, insider_data, ticker_info):
    """Insider alım kümeleri tespiti.

    Kriter:
    - Son 60 günde 2+ insider alımı
    - Toplam alım > $100K
    - Net alım (satış çıkarılmış)
    """
    results = []
    cutoff = datetime.now() - timedelta(days=INSIDER_LOOKBACK)

    for ticker, txns in insider_data.items():
        if txns.empty or 'Text' not in txns.columns:
            continue

        # Alım/satım ayır
        text_col = txns['Text'].fillna('')
        buys = txns[text_col.str.contains('Purchase|Buy|Acquisition', case=False, na=False)]
        sells = txns[text_col.str.contains('Sale|Sell|Disposition', case=False, na=False)]

        if len(buys) < INSIDER_MIN_BUYERS:
            continue

        # Toplam alım değeri
        total_buy_value = 0
        if 'Value' in buys.columns:
            total_buy_value = buys['Value'].sum()
        elif 'Shares' in buys.columns:
            df = stock_dfs.get(ticker)
            if df is not None and len(df) > 0:
                total_buy_value = buys['Shares'].abs().sum() * df['Close'].iloc[-1]

        if total_buy_value < INSIDER_MIN_VALUE:
            continue

        # Net alım check (satış varsa dikkat)
        total_sell_value = 0
        if not sells.empty and 'Value' in sells.columns:
            total_sell_value = sells['Value'].abs().sum()
        net_positive = total_buy_value > total_sell_value

        if not net_positive:
            continue

        # Unique buyers
        n_buyers = buys['Insider'].nunique() if 'Insider' in buys.columns else len(buys)

        # Recency: en son alım kaç gün önce
        if 'Start Date' in buys.columns:
            latest_buy = buys['Start Date'].max()
            days_ago = (datetime.now() - latest_buy).days if pd.notna(latest_buy) else 999
        else:
            days_ago = 30  # fallback

        # Role seniority
        senior_roles = ['CEO', 'CFO', 'COO', 'President', 'Chairman', 'Director']
        has_senior = False
        if 'Insider' in buys.columns:
            for role in senior_roles:
                if buys['Insider'].str.contains(role, case=False, na=False).any():
                    has_senior = True
                    break

        # Skor
        s_buyers = min(n_buyers / 5.0, 1.0) * 30
        s_value = min(total_buy_value / 1_000_000, 1.0) * 30
        s_recency = max(0, 1.0 - days_ago / INSIDER_LOOKBACK) * 20
        s_role = 20 if has_senior else 5
        score = int(s_buyers + s_value + s_recency + s_role)

        df = stock_dfs.get(ticker)
        cur_close = df['Close'].iloc[-1] if df is not None and len(df) > 0 else 0

        info = ticker_info.get(ticker, {})

        # Execution quality
        trigger = 0
        stop = 0
        inval = 0
        risk_pct = 0
        sp = 0.0
        gr = 0.0
        dv = 0
        if df is not None and len(df) >= 20:
            high = df['High']
            low = df['Low']
            close_s = df['Close']
            volume_s = df['Volume']
            atr_14 = _atr(high, low, close_s, 14)
            atr_val = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else 0
            trigger = round(high.iloc[-10:].max(), 2)  # son 10g high
            stop = round(low.iloc[-20:].min(), 2)  # son 20g low
            inval = round(low.iloc[-60:].min(), 2) if len(low) >= 60 else stop  # son 60g low
            risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0
            sp = _spread_proxy(atr_val, cur_close)
            gr = _gap_risk(df)
            dv = int(_dollar_volume(close_s, volume_s, 20))

        results.append({
            'module': 'INSIDER',
            'ticker': ticker,
            'name': info.get('name', ticker),
            'close': round(cur_close, 2),
            'n_buyers': n_buyers,
            'total_buy_value': int(total_buy_value),
            'buy_value_k': round(total_buy_value / 1000, 0),
            'n_sells': len(sells),
            'days_ago': days_ago,
            'has_senior': has_senior,
            'sector': info.get('sector', ''),
            'score': score,
            'dollar_vol': dv,
            'trigger': trigger,
            'stop': stop,
            'invalidation': inval,
            'risk_pct': risk_pct,
            'spread_proxy': sp,
            'gap_risk': gr,
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 4: BIOTECH CATALYST
# ══════════════════════════════════════════

def _biotech_analyze(ticker, df, info, spy_df, fda_cal):
    """Tek bir biotech ticker'ı analiz et. Ortak mantık."""
    close = df['Close']
    volume = df['Volume']
    high = df['High']
    low = df['Low']
    cur_close = close.iloc[-1]

    if cur_close < MIN_PRICE_USD:
        return None

    # Hacim build-up
    avg_vol = _sma(volume, 20).iloc[-1]
    if pd.isna(avg_vol) or avg_vol <= 0:
        return None

    vol_5d = volume.iloc[-5:].mean()
    vol_10d = volume.iloc[-10:].mean()
    vol_trend = vol_5d / avg_vol if avg_vol > 0 else 1.0
    vol_accel = vol_5d / vol_10d if vol_10d > 0 else 1.0

    # ATR genişleme
    atr_14 = _atr(high, low, close, 14)
    if len(atr_14) < 60:
        return None
    atr_cur = atr_14.iloc[-1]
    atr_avg = atr_14.iloc[-60:].mean()
    atr_expansion = atr_cur / atr_avg if atr_avg > 0 else 1.0

    # Earnings
    earn_date = info.get('earnings_date') if info else None
    days_to_earnings = None
    if earn_date:
        days_to_earnings = (earn_date - datetime.now()).days
        if days_to_earnings < 0:
            days_to_earnings = None

    # FDA/PDUFA takvimi
    fda_info = fda_cal.get(ticker)
    fda_date_str = None
    days_to_fda = None
    fda_drug = None
    fda_phase = None
    if fda_info:
        fda_date_str = fda_info.get('date')
        fda_drug = fda_info.get('drug')
        fda_phase = fda_info.get('phase')
        if fda_date_str:
            try:
                fda_dt = datetime.strptime(fda_date_str, '%Y-%m-%d')
                days_to_fda = (fda_dt - datetime.now()).days
                if days_to_fda < 0:
                    days_to_fda = None
            except ValueError:
                pass

    # Skor
    mcap = info.get('market_cap') if info else None
    s_vol = min(vol_trend / 3.0, 1.0) * 20
    s_atr = min(atr_expansion / 2.0, 1.0) * 20
    s_mcap = 20 if (mcap and mcap < 2e9) else (12 if mcap and mcap < 5e9 else 5)

    s_cat = 0
    if days_to_fda is not None and days_to_fda <= 60:
        s_cat = 40 if days_to_fda <= 14 else (30 if days_to_fda <= 30 else 20)
    elif days_to_earnings and days_to_earnings <= 14:
        s_cat = 15
    else:
        s_cat = 5

    score = int(s_vol + s_atr + s_mcap + s_cat)

    rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None
    industry = info.get('industry', '') if info else ''

    # Execution quality
    trigger = round(cur_close, 2)  # hemen
    stop = round(cur_close - 2 * atr_cur, 2)
    inval = round(low.iloc[-20:].min(), 2)  # son 20g low
    risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0
    dv = _dollar_volume(close, volume, 20)

    return {
        'module': 'BIOTECH',
        'ticker': ticker,
        'name': info.get('name', ticker) if info else ticker,
        'close': round(cur_close, 2),
        'mcap_b': round(mcap / 1e9, 2) if mcap else None,
        'vol_trend': round(vol_trend, 2),
        'vol_accel': round(vol_accel, 2),
        'atr_expansion': round(atr_expansion, 2),
        'days_to_earnings': days_to_earnings,
        'fda_date': fda_date_str,
        'days_to_fda': days_to_fda,
        'fda_drug': fda_drug,
        'fda_phase': fda_phase,
        'industry': industry,
        'score': score,
        'rs': rs,
        'dollar_vol': int(dv),
        'trigger': trigger,
        'stop': stop,
        'invalidation': inval,
        'risk_pct': risk_pct,
        'spread_proxy': _spread_proxy(atr_cur, cur_close),
        'gap_risk': _gap_risk(df),
    }


def scan_biotech_catalyst(stock_dfs, ticker_info, spy_df=None, fda_calendar=None):
    """Biyotek katalizör adayları + FDA/PDUFA takvimi.

    İki kaynak:
    1. ticker_info'daki biotech/pharma şirketleri (sektör filtresi)
    2. FDA takvimindeki tüm ticker'lar (sektör filtresi atlanır — FDA varsa biotech'tir)
    """
    results = []
    seen = set()
    fda_cal = fda_calendar or {}

    # ── Kaynak 1: ticker_info'daki biotech'ler ──
    for ticker, info in ticker_info.items():
        industry = info.get('industry', '')

        is_biotech = (
            'Biotechnology' in industry or
            'Pharmaceutical' in industry or
            'Drug' in industry or
            'Genomics' in industry
        )
        if not is_biotech:
            continue

        mcap = info.get('market_cap')
        if mcap and mcap > BIOTECH_MCAP_MAX:
            continue

        df = stock_dfs.get(ticker)
        if df is None or len(df) < 30:
            continue

        entry = _biotech_analyze(ticker, df, info, spy_df, fda_cal)
        if entry:
            results.append(entry)
            seen.add(ticker)

    # ── Kaynak 2: FDA takvimindeki ama henüz taranmayan ticker'lar ──
    for ticker in fda_cal:
        if ticker in seen:
            continue
        df = stock_dfs.get(ticker)
        if df is None or len(df) < 30:
            continue
        info = ticker_info.get(ticker)  # None olabilir — sorun değil
        entry = _biotech_analyze(ticker, df, info, spy_df, fda_cal)
        if entry:
            results.append(entry)
            seen.add(ticker)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 5: EARNINGS MOMENTUM
# ══════════════════════════════════════════

def scan_earnings_momentum(stock_dfs, ticker_info, spy_df=None):
    """Bilanço momentum tespiti.

    Pre-earnings: Bilanço yakın + sıkışma + hacim build-up.
    Post-earnings: Gap up + tutunma.
    """
    results = []
    for ticker, info in ticker_info.items():
        df = stock_dfs.get(ticker)
        if df is None or len(df) < 30:
            continue

        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        cur_close = close.iloc[-1]

        if cur_close < MIN_PRICE_USD:
            continue

        earn_date = info.get('earnings_date')
        if earn_date is None:
            continue

        days_to = (earn_date - datetime.now()).days
        subtype = None

        if 0 < days_to <= EARNINGS_WINDOW:
            # Pre-earnings setup
            subtype = 'PRE'

            # BB width check (sıkışma)
            bb_width = _bb_width_pct(close, 20, 2.0)
            if bb_width is None:
                continue

            # Volume trend
            avg_vol = _sma(volume, 20).iloc[-1]
            vol_5d = volume.iloc[-5:].mean()
            vol_trend = vol_5d / avg_vol if pd.notna(avg_vol) and avg_vol > 0 else 1.0

            # Skor
            s_bb = max(0, min(1.0 - bb_width / 15.0, 1.0)) * 30  # dar BB = yüksek skor
            s_days = max(0, 1.0 - days_to / EARNINGS_WINDOW) * 25  # yakın = yüksek
            s_vol = min(vol_trend / 2.0, 1.0) * 25
            s_trend = 20 if cur_close > _ema(close, 21).iloc[-1] else 5
            score = int(s_bb + s_days + s_vol + s_trend)

            rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None

            # Execution quality (PRE)
            atr_14 = _atr(high, low, close, 14)
            atr_val = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else 0
            trigger = round(high.iloc[-5:].max(), 2)  # son 5g high
            stop = round(low.iloc[-5:].min(), 2)  # son 5g low
            bb_mid = _sma(close, 20).iloc[-1]
            bb_std = close.rolling(20, min_periods=20).std().iloc[-1]
            bb_lower = bb_mid - 2 * bb_std if pd.notna(bb_std) else stop
            inval = round(bb_lower, 2)  # BB alt bandı
            risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0
            dv = int(_dollar_volume(close, volume, 20))

            results.append({
                'module': 'EARNINGS',
                'subtype': subtype,
                'ticker': ticker,
                'name': info.get('name', ticker),
                'close': round(cur_close, 2),
                'days_to_earnings': days_to,
                'bb_width': round(bb_width, 1),
                'vol_trend': round(vol_trend, 2),
                'earnings_date': earn_date.strftime('%Y-%m-%d'),
                'sector': info.get('sector', ''),
                'score': score,
                'rs': rs,
                'dollar_vol': dv,
                'trigger': trigger,
                'stop': stop,
                'invalidation': inval,
                'risk_pct': risk_pct,
                'spread_proxy': _spread_proxy(atr_val, cur_close),
                'gap_risk': _gap_risk(df),
            })

        elif -3 <= days_to <= 0:
            # Post-earnings: gap up kontrolü
            subtype = 'POST'
            gap_idx = max(0, len(close) + days_to - 1)
            if gap_idx <= 0 or gap_idx >= len(close):
                continue

            pre_close = close.iloc[gap_idx - 1]
            post_open = df['Open'].iloc[gap_idx]
            gap_pct = (post_open - pre_close) / pre_close * 100

            if gap_pct < EARNINGS_GAP_MIN:
                continue

            # Gap tutunma: mevcut fiyat vs gap seviyesi
            gap_hold = (cur_close - pre_close) / pre_close * 100
            pullback = gap_pct - gap_hold  # ne kadar geri gelmiş

            # Hacim devamı
            avg_vol = _sma(volume, 20).iloc[-1]
            vol_ratio = volume.iloc[-1] / avg_vol if pd.notna(avg_vol) and avg_vol > 0 else 1.0

            score = int(
                min(gap_pct / 15.0, 1.0) * 30 +
                max(0, 1.0 - pullback / gap_pct) * 30 +
                min(vol_ratio / 3.0, 1.0) * 20 +
                20  # post-earnings bonus
            )

            rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None

            # Execution quality (POST)
            atr_14 = _atr(high, low, close, 14)
            atr_val = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else 0
            trigger = round(cur_close, 2)  # hemen
            stop = round(pre_close, 2)  # gap seviyesi
            inval = round(pre_close - atr_val, 2)  # gap seviyesi - ATR
            risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 and trigger > stop else 0
            dv = int(_dollar_volume(close, volume, 20))

            results.append({
                'module': 'EARNINGS',
                'subtype': subtype,
                'ticker': ticker,
                'name': info.get('name', ticker),
                'close': round(cur_close, 2),
                'days_to_earnings': days_to,
                'gap_pct': round(gap_pct, 1),
                'gap_hold_pct': round(gap_hold, 1),
                'vol_trend': round(vol_ratio, 2),
                'earnings_date': earn_date.strftime('%Y-%m-%d'),
                'sector': info.get('sector', ''),
                'score': score,
                'rs': rs,
                'dollar_vol': dv,
                'trigger': trigger,
                'stop': stop,
                'invalidation': inval,
                'risk_pct': risk_pct,
                'spread_proxy': _spread_proxy(atr_val, cur_close),
                'gap_risk': _gap_risk(df),
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# MODÜL 6: TECHNICAL BREAKOUT
# ══════════════════════════════════════════

def scan_technical_breakout(stock_dfs, spy_df=None):
    """Teknik kırılım setupları.

    Kriter:
    - ATR sıkışma: mevcut ATR < 60% of 60-day ATR avg
    - Konsolidasyon: 15+ gün dar range
    - Breakout tetik: close > son 20g high + hacim patlaması
    - RS vs SPY > 1.0
    """
    results = []
    for ticker, df in stock_dfs.items():
        if len(df) < 80:
            continue

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        cur_close = close.iloc[-1]

        if cur_close < MIN_PRICE_USD:
            continue

        dv = _dollar_volume(close, volume, 20)
        if dv < MIN_AVG_VOLUME_USD:
            continue

        # ATR sıkışma
        atr_14 = _atr(high, low, close, 14)
        if len(atr_14) < 60 or pd.isna(atr_14.iloc[-1]):
            continue
        atr_cur = atr_14.iloc[-1]
        atr_60_avg = atr_14.iloc[-60:].mean()
        if atr_60_avg <= 0:
            continue
        atr_ratio = atr_cur / atr_60_avg

        # Konsolidasyon tespiti: son N gün range / close
        lookback = 20
        recent_high = high.iloc[-lookback:].max()
        recent_low = low.iloc[-lookback:].min()
        range_pct = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 999

        # Konsolidasyon süresi (range dar kaldığı gün sayısı)
        consol_days = 0
        daily_range_avg = (high - low).iloc[-60:].mean()
        for j in range(len(close) - 1, max(len(close) - 61, -1), -1):
            daily_r = high.iloc[j] - low.iloc[j]
            if daily_r <= daily_range_avg * 0.8:
                consol_days += 1
            else:
                break

        # Breakout check: close > son 20g high (dünün verisine kadar)
        prev_20_high = high.iloc[-lookback - 1:-1].max()
        is_breakout = cur_close > prev_20_high

        # Volume patlaması
        avg_vol = _sma(volume, VOL_SMA_PERIOD).iloc[-1]
        vol_ratio = volume.iloc[-1] / avg_vol if pd.notna(avg_vol) and avg_vol > 0 else 1.0

        # RS
        rs = _rs_vs_spy(close, spy_df['Close'], 20) if spy_df is not None else None

        # İki tip setup:
        # A) SETUP (sıkışma devam, henüz kırmadı) — PRE-BREAKOUT, prediktif
        # B) BREAKOUT (bugün kırdı) — reaktif, ama momentum devamı olabilir
        if atr_ratio <= ATR_COMPRESS_RATIO and consol_days >= CONSOL_MIN_DAYS:
            subtype = 'SETUP'
        elif is_breakout and vol_ratio >= BREAKOUT_VOL_MULT:
            subtype = 'BREAKOUT'
        else:
            continue

        # Skor — SETUP'a daha yüksek ağırlık (prediktif)
        if subtype == 'SETUP':
            s_consol = min(consol_days / 25.0, 1.0) * 30  # uzun sıkışma = iyi
            s_atr = max(0, 1.0 - atr_ratio) * 30  # dar ATR = patlama potansiyeli
            s_range = max(0, 1.0 - range_pct / 15.0) * 20  # dar range = iyi
            s_rs = min(max(rs or 0, 0) / 2.0, 1.0) * 20
            score = int(s_consol + s_atr + s_range + s_rs)
        else:  # BREAKOUT
            s_vol = min(vol_ratio / 4.0, 1.0) * 30
            s_atr = max(0, 1.0 - atr_ratio) * 20
            s_consol = min(consol_days / 30.0, 1.0) * 25
            s_rs = min(max(rs or 0, 0) / 2.0, 1.0) * 25
            score = int(s_vol + s_atr + s_consol + s_rs)

        # Execution quality
        if subtype == 'SETUP':
            trigger = round(prev_20_high + 0.01, 2)  # konsolidasyon high kırılması
            stop = round(recent_low, 2)  # konsolidasyon low
            inval = round(recent_low - atr_cur, 2)  # konsolidasyon low - ATR
        else:  # BREAKOUT (aktif)
            trigger = round(cur_close, 2)  # hemen
            stop = round(cur_close - 2 * atr_cur, 2)
            inval = round(prev_20_high, 2)  # breakout seviyesi
        risk_pct = round((trigger - stop) / trigger * 100, 1) if trigger > 0 else 0

        results.append({
            'module': 'BREAKOUT',
            'subtype': subtype,
            'ticker': ticker,
            'close': round(cur_close, 2),
            'atr_ratio': round(atr_ratio, 2),
            'consol_days': consol_days,
            'range_pct': round(range_pct, 1),
            'vol_ratio': round(vol_ratio, 1),
            'prev_20_high': round(prev_20_high, 2),
            'is_breakout': is_breakout,
            'score': score,
            'rs': rs,
            'dollar_vol': int(dv),
            'trigger': trigger,
            'stop': stop,
            'invalidation': inval,
            'risk_pct': risk_pct,
            'spread_proxy': _spread_proxy(atr_cur, cur_close),
            'gap_risk': _gap_risk(df),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ══════════════════════════════════════════
# ANA TARAMA FONKSİYONU
# ══════════════════════════════════════════

def run_all_modules(stock_dfs, spy_df=None, ticker_info=None, insider_data=None,
                     fda_calendar=None):
    """Tüm modülleri çalıştır ve sonuçları birleştir.

    Faz 1 modülleri (sadece OHLCV): Unusual Volume, Accumulation, Technical Breakout
    Faz 2 modülleri (enrichment gerekli): Short Squeeze, Insider, Biotech, Earnings
    """
    all_results = {}

    # ── Faz 1: Batch tarama ──
    print("\n═══ FAZ 1: Teknik Tarama ═══")

    vol_results = scan_unusual_volume(stock_dfs, spy_df)
    all_results['VOLUME'] = vol_results
    print(f"  📈 Unusual Volume: {len(vol_results)} sinyal (reaktif)")

    acc_results = scan_accumulation(stock_dfs, spy_df)
    all_results['ACCUM'] = acc_results
    print(f"  🔍 Accumulation: {len(acc_results)} sinyal (prediktif)")

    brk_results = scan_technical_breakout(stock_dfs, spy_df)
    all_results['BREAKOUT'] = brk_results
    n_setup = sum(1 for r in brk_results if r.get('subtype') == 'SETUP')
    n_brk = len(brk_results) - n_setup
    print(f"  🔺 Technical Breakout: {len(brk_results)} sinyal ({n_setup} setup + {n_brk} aktif)")

    # ── Faz 2: Zenginleştirilmiş tarama ──
    if ticker_info:
        print("\n═══ FAZ 2: Zenginleştirilmiş Tarama ═══")

        sq_results = scan_short_squeeze(stock_dfs, ticker_info)
        all_results['SQUEEZE'] = sq_results
        print(f"  🔴 Short Squeeze: {len(sq_results)} sinyal")

        bt_results = scan_biotech_catalyst(stock_dfs, ticker_info, spy_df, fda_calendar)
        all_results['BIOTECH'] = bt_results
        print(f"  🧬 Biotech Catalyst: {len(bt_results)} sinyal")

        er_results = scan_earnings_momentum(stock_dfs, ticker_info, spy_df)
        all_results['EARNINGS'] = er_results
        print(f"  📊 Earnings Momentum: {len(er_results)} sinyal")

        if insider_data:
            ins_results = scan_insider_buying(stock_dfs, insider_data, ticker_info)
            all_results['INSIDER'] = ins_results
            print(f"  💼 Insider Buying: {len(ins_results)} sinyal")
        else:
            all_results['INSIDER'] = []
    else:
        all_results['SQUEEZE'] = []
        all_results['BIOTECH'] = []
        all_results['EARNINGS'] = []
        all_results['INSIDER'] = []

    total = sum(len(v) for v in all_results.values())
    print(f"\n📋 Toplam: {total} sinyal")
    return all_results
