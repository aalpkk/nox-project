"""
BIST REJİM v3 SCREENER — Tüm 9 Sinyal + Rejim Filtresi + Telegram
====================================================================
Pine Script v3 indikatörünün birebir Python karşılığı.

Sinyaller (öncelik sırası):
  CMB+   = WT Cross + PMAX Long + CHoCH (en güçlü)
  CMB    = WT Cross + PMAX Long + BOS
  GUCLU  = Trend + Squeeze + Hacim + RS + Quality
  ZAYIF  = Trend + Squeeze + RS + Quality (hacimsiz)
  DONUS  = EMA55 cross + WT cross + RS
  ERKEN  = Yapı kırılımı/momentum + ADX dönüş
  PB     = Pullback alım (EMA dip + reclaim)
  SQ     = Squeeze expansion + ATR genişleme
  MR     = Mean reversion (BB/Donch oversold)

Rejim:
  3 = FULL TREND  (haftalık ADX güçlü + günlük teyit)
  2 = TREND       (haftalık ADX trend üstü)
  1 = GRI BOLGE   (trend teyitli ama ADX zayıf)
  0 = CHOPPY      (trend yok veya ADX düşük)

Veri: yfinance (.IS suffix)
Çıktı: Telegram + CSV + konsol
"""

# NOX: yfinance yerine extfeed (taze nox-project verisi) — veri-çekme katmanı swap edildi.
# Tarayıcı mantığı (indikatör/sinyal/yayın) DOKUNULMADI; sadece get_all_bist_tickers /
# fetch_data / fetch_benchmark gövdeleri shim'e delege edildi.
from _extfeed_daily import (
    fetch_data_batch as _xf_fetch_data_batch,
    fetch_benchmark as _xf_fetch_benchmark,
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# PARAMETRELER (Pine Script v3 ile birebir)
# ============================================================

# ADX
ADX_LEN = 14
ADX_TREND = 20.0
ADX_CHOPPY = 15.0
ADX_SLOPE_LEN = 3
ADX_SLOPE_THRESH = 0.5

# EMA / SuperTrend
EMA_FAST = 21
EMA_SLOW = 55
ST_LEN = 10
ST_MULT = 3.0

# WaveTrend
WT_CH_LEN = 10
WT_AVG_LEN = 21
WT_MA_LEN = 4
WT_LOOKBACK = 3  # cross lookback (v3'te 3)

# PMAX
PMAX_ATR_LEN = 10
PMAX_ATR_MULT = 3.0
PMAX_MA_LEN = 10
PMAX_MA_TYPE = "EMA"

# SMC
SMC_INTERNAL_LEN = 5
BOS_LOOKBACK = 10
BOS_TIGHT = 5  # combo için tight lookback

# Squeeze
SQ_LEN = 20
SQ_MULT_BB = 2.0
SQ_MULT_KC = 1.5

# BB / Mean Reversion
BB_LEN = 20
BB_MULT = 2.0
DONCH_LEN = 20
MR_RSI_LEN = 2
MR_RSI_THRESH = 20.0  # RSI(2) < 20 (eskisi 10 idi, çok sıkıydı)
MR_TIME_BASE = 5

# Quality
QUAL_MIN_GRI = 60.0
QUAL_MIN_TREND = 40.0
RVOL_THRESH = 1.5

# RS
RS_LEN1 = 21
RS_LEN2 = 63
RS_THRESHOLD = 0.0

# Stop / TP çarpanları
ATR_LEN = 14
TREND_STOP = 2.0
GRI_STOP = 2.0
MR_STOP = 2.5
DONUS_STOP = 1.5
COMBO_STOP = 2.0
TREND_TP = 3.0
GRI_TP = 2.5    # eskisi 1.5 idi, R:R düşüktü (PB/SQ için)
DONUS_TP = 2.0   # eskisi 1.0 idi, R:R çok düşüktü
COMBO_TP = 3.0

# Filtreler
MIN_DATA_DAYS = 80
MIN_AVG_VOLUME_TL = 500_000

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ============================================================
# HİSSE LİSTESİ
# ============================================================

def get_all_bist_tickers():
    """Tüm BIST hisselerini çek."""
    for path in ['tickers.txt', 'hisseler.txt', 'xtum.txt']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                tickers = [l.strip().upper() for l in f if l.strip()]
            if len(tickers) > 50:
                print(f"✅ {path} → {len(tickers)} hisse")
                return tickers

    tickers = _fetch_from_tradingview()
    if tickers and len(tickers) > 100:
        return tickers

    tickers = _fetch_from_isyatirim()
    if tickers and len(tickers) > 100:
        return tickers

    print("⚠️ Statik liste kullanılıyor")
    return _static_tickers()


def _fetch_from_tradingview():
    try:
        url = "https://scanner.tradingview.com/turkey/scan"
        payload = {
            "columns": ["name"],
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BIST"},
                {"left": "type", "operation": "equal", "right": "stock"},
                {"left": "is_primary", "operation": "equal", "right": True},
            ],
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [0, 1000],
        }
        r = requests.post(url, json=payload, headers={
            'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'
        }, timeout=15)
        if r.status_code == 200:
            tickers = []
            for item in r.json().get('data', []):
                sym = item.get('s', '').split(':')[-1]
                if sym and len(sym) >= 2:
                    tickers.append(sym)
            if tickers:
                print(f"✅ TradingView → {len(tickers)} hisse")
            return tickers
    except Exception as e:
        print(f"  [!] TradingView: {e}")
    return None


def _fetch_from_isyatirim():
    try:
        d = datetime.now()
        for _ in range(10):
            d -= timedelta(days=1)
            if d.weekday() < 5:
                break
        tarih = d.strftime("%d-%m-%Y")
        url = ("https://www.isyatirim.com.tr/_layouts/15/"
               "IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetYabanciOranlarXHR")
        payload = {"baslangicTarih": tarih, "bitisTarihi": tarih,
                   "sektor": None, "endeks": "09", "hisse": None}
        headers = {"Content-Type": "application/json; charset=UTF-8",
                   "X-Requested-With": "XMLHttpRequest", "User-Agent": "Mozilla/5.0"}
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            tickers = [i["HISSE_KODU"] for i in r.json().get("d", []) if i.get("HISSE_KODU")]
            if len(tickers) > 100:
                print(f"✅ İsyatırım → {len(tickers)} hisse")
            return tickers
    except:
        pass
    return None


def _static_tickers():
    return [
        "ACSEL","ADEL","AEFES","AFYON","AGESA","AGHOL","AKBNK","AKCNS","AKENR",
        "AKFGY","AKGRT","AKSA","AKSEN","AKSGY","ALARK","ALFAS","ALGYO","ALKIM",
        "ANHYT","ANSGR","ARCLK","ARDYZ","ARENA","ASELS","ASGYO","ASTOR","ASUZU",
        "ATAGY","ATAKP","ATLAS","AYEN","AYGAZ","BAGFS","BERA","BFREN","BIENY",
        "BIGCH","BIMAS","BIOEN","BJKAS","BOBET","BRISA","BRSAN","BRYAT","BTCIM",
        "BUCIM","CCOLA","CIMSA","CWENE","DOAS","DOHOL","ECILC","ECZYT","EGEEN",
        "EKGYO","ENJSA","ENKAI","ERBOS","EREGL","EUPWR","EUREN","FROTO","GARAN",
        "GENIL","GESAN","GLYHO","GOLTS","GSRAY","GUBRF","HEKTS","HLGYO","HUBVC",
        "IMASM","INDES","IPEKE","ISDMR","ISGYO","ISMEN","ISSEN","KARSN","KAYSE",
        "KCHOL","KLSER","KMPUR","KONTR","KONYA","KOZAA","KOZAL","KRDMD","KZBGY",
        "LMKDC","MAVI","MGROS","MIATK","MPARK","ODAS","OTKAR","OYAKC","PAPIL",
        "PEKGY","PETKM","PGSUS","REEDR","SAHOL","SASA","SISE","SKBNK","SOKM",
        "SRVGY","TABGD","TATGD","TAVHL","TCELL","THYAO","TKFEN","TKNSA","TOASO",
        "TRGYO","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VERUS","VESTL",
        "YEOTK","YKBNK","YUNSA","ZOREN",
    ]


# ============================================================
# VERİ ÇEKME
# ============================================================

def fetch_data(tickers, period="1y", batch_size=50):
    """NOX: extfeed günlük OHLCV (yfinance yerine). İmza + dönüş şekli değişmedi:
    {ticker: kapitalize-OHLCV DataFrame (DatetimeIndex)}. Haftalık rejim için 1y yeterli."""
    return _xf_fetch_data_batch(tickers, period=period, batch_size=batch_size,
                                min_days=MIN_DATA_DAYS)


def fetch_benchmark(period="1y"):
    """NOX: XU100 günlük OHLCV — extfeed (xu100_extfeed_daily) (yf XU100.IS yerine)."""
    return _xf_fetch_benchmark(period=period)


# ============================================================
# TEKNİK HESAPLAMALAR
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def wma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def rma(series, period):
    """Pine Script'in ta.rma karşılığı (Wilder's smoothing)."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def true_range(df):
    h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def calc_atr(df, period=14):
    return rma(true_range(df), period)

def calc_adx(df, length=14):
    """ADX hesaplama — Pine Script calcADX ile birebir."""
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr = rma(tr, length)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), length) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = rma(dx, length)
    return adx


def calc_supertrend(df, period=10, mult=3.0):
    """SuperTrend."""
    atr = calc_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    st_dir = pd.Series(1, index=df.index)
    final_up = up.copy()
    final_dn = dn.copy()
    for i in range(1, len(df)):
        if up.iloc[i] > final_up.iloc[i-1]:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = final_up.iloc[i-1] if df['Close'].iloc[i-1] > final_up.iloc[i-1] else up.iloc[i]
        if dn.iloc[i] < final_dn.iloc[i-1]:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = final_dn.iloc[i-1] if df['Close'].iloc[i-1] < final_dn.iloc[i-1] else dn.iloc[i]
        prev_dir = st_dir.iloc[i-1]
        if prev_dir == -1 and df['Close'].iloc[i] > final_dn.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif prev_dir == 1 and df['Close'].iloc[i] < final_up.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = prev_dir
    return st_dir  # 1 = up, -1 = down


def calc_wavetrend(df):
    """WaveTrend + bearish invalidation (v3 mantığı)."""
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    esa_val = ema(hlc3, WT_CH_LEN)
    d = ema((hlc3 - esa_val).abs(), WT_CH_LEN)
    ci = (hlc3 - esa_val) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, WT_AVG_LEN)
    wt2 = sma(wt1, WT_MA_LEN)
    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_dn = (wt2 > wt1) & (wt2.shift(1) <= wt1.shift(1))

    # Son N gün cross + bearish invalidation (v3 mantığı)
    n = len(df)
    wt_recent = pd.Series(False, index=df.index)
    wt_killed = pd.Series(False, index=df.index)
    for i in range(WT_LOOKBACK, n):
        killed = False
        for j in range(WT_LOOKBACK + 1):
            if i - j >= 0 and cross_dn.iloc[i - j]:
                killed = True
                break
        wt_killed.iloc[i] = killed
        if not killed:
            for j in range(WT_LOOKBACK + 1):
                if i - j >= 0 and cross_up.iloc[i - j]:
                    wt_recent.iloc[i] = True
                    break

    return {
        'wt1': wt1, 'wt2': wt2,
        'cross_up': cross_up, 'cross_dn': cross_dn,
        'wt_recent': wt_recent, 'wt_killed': wt_killed,
        'wt_bullish': wt1 > wt2,
    }


def calc_pmax(df):
    """PMAX — Kıvanç Özbilgiç."""
    src = (df['High'] + df['Low']) / 2
    if PMAX_MA_TYPE == "EMA":
        ma_val = ema(src, PMAX_MA_LEN)
    elif PMAX_MA_TYPE == "WMA":
        ma_val = wma(src, PMAX_MA_LEN)
    else:
        ma_val = sma(src, PMAX_MA_LEN)
    atr = calc_atr(df, PMAX_ATR_LEN)
    n = len(df)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    pmax_dir = np.ones(n)
    for i in range(1, n):
        ls = ma_val.iloc[i] - PMAX_ATR_MULT * atr.iloc[i]
        ss = ma_val.iloc[i] + PMAX_ATR_MULT * atr.iloc[i]
        long_stop[i] = max(ls, long_stop[i-1]) if not np.isnan(long_stop[i-1]) and ma_val.iloc[i] > long_stop[i-1] else ls
        short_stop[i] = min(ss, short_stop[i-1]) if not np.isnan(short_stop[i-1]) and ma_val.iloc[i] < short_stop[i-1] else ss
        prev = pmax_dir[i-1]
        if prev == -1 and ma_val.iloc[i] > short_stop[i-1]:
            pmax_dir[i] = 1
        elif prev == 1 and ma_val.iloc[i] < long_stop[i-1]:
            pmax_dir[i] = -1
        else:
            pmax_dir[i] = prev
    pmax_long = pmax_dir == 1
    return {'pmax_dir': pmax_dir, 'pmax_long': pmax_long, 'pmax_val': long_stop}


def calc_smc(df):
    """SMC BOS / CHoCH — bullish yapı kırılımı, bearish invalidation dahil."""
    n = len(df)
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    k = SMC_INTERNAL_LEN
    last_sh = np.nan
    last_sl = np.nan
    swing_bias = 0
    bull_bos = np.zeros(n, dtype=bool)
    bull_choch = np.zeros(n, dtype=bool)
    bos_bar = np.full(n, -999)
    choch_bar = np.full(n, -999)
    # pivot detection
    for i in range(k, n):
        # check i-k as pivot
        pi = i - k
        if pi >= k:
            is_ph = h[pi] == np.max(h[pi-k:pi+k+1])
            is_pl = l[pi] == np.min(l[pi-k:pi+k+1])
            if is_ph:
                last_sh = h[pi]
            if is_pl:
                last_sl = l[pi]
        # BOS/CHoCH check
        if not np.isnan(last_sh) and i > 0:
            if c[i] > last_sh and c[i-1] <= last_sh:
                if swing_bias <= 0:
                    bull_choch[i] = True
                    choch_bar[i] = i
                bull_bos[i] = True
                bos_bar[i] = i
                swing_bias = 1
        # Bearish invalidation
        if not np.isnan(last_sl) and i > 0:
            if c[i] < last_sl and c[i-1] >= last_sl:
                swing_bias = -1

    # En son BOS/CHoCH bar indeksini propaget et
    last_bos_bar = -999
    last_choch_bar = -999
    bos_bar_prop = np.full(n, -999)
    choch_bar_prop = np.full(n, -999)
    for i in range(n):
        if bull_bos[i]:
            last_bos_bar = i
        if bull_choch[i]:
            last_choch_bar = i
        bos_bar_prop[i] = last_bos_bar
        choch_bar_prop[i] = last_choch_bar

    return {
        'bull_bos': bull_bos, 'bull_choch': bull_choch,
        'bos_bar': bos_bar_prop, 'choch_bar': choch_bar_prop,
        'swing_bias': swing_bias,
    }


def calc_order_blocks(df):
    """Bearish OB (direnç) tespiti."""
    n = len(df)
    h, l, o, c = df['High'].values, df['Low'].values, df['Open'].values, df['Close'].values
    k = SMC_INTERNAL_LEN
    ob_top, ob_bot = np.nan, np.nan
    for i in range(k, n - k):
        is_ph = h[i] == np.max(h[max(0,i-k):i+k+1])
        if is_ph:
            for j in range(i-1, max(i-k-10, 0), -1):
                if c[j] < o[j]:
                    ob_top, ob_bot = h[j], l[j]
                    break
    # mitigation
    if not np.isnan(ob_top) and c[-1] > ob_top:
        ob_top, ob_bot = np.nan, np.nan
    return {'ob_resist_top': ob_top, 'ob_resist_bot': ob_bot}


def resample_weekly(df):
    """Günlük veriyi haftalığa çevir."""
    w = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return w


# ============================================================
# ANA ANALİZ
# ============================================================

def analyze(ticker, df, xu_df, dbg=None):
    """Tek hisseyi v3 indikatör mantığıyla analiz et."""
    try:
        if dbg: dbg['total'] += 1
        n = len(df)
        c = df['Close']
        h = df['High']
        l = df['Low']
        o = df['Open']
        v = df['Volume']

        # ATR
        atr = calc_atr(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if np.isnan(atr_val) or atr_val == 0:
            if dbg: dbg['no_atr'] += 1
            return None

        # Hacim filtresi
        vol_sma20 = sma(v, 20)
        avg_turnover = vol_sma20.iloc[-1] * c.iloc[-1]
        if avg_turnover < MIN_AVG_VOLUME_TL:
            if dbg: dbg['low_vol'] += 1
            return None

        # === TREND ===
        ema_f = ema(c, EMA_FAST)
        ema_s = ema(c, EMA_SLOW)
        ema_trend_up = ema_f.iloc[-1] > ema_s.iloc[-1]
        st_dir = calc_supertrend(df, ST_LEN, ST_MULT)
        super_trend_up = st_dir.iloc[-1] == 1

        # === HTF (haftalık) ===
        wdf = resample_weekly(df)
        if len(wdf) < 20:
            htf_adx, htf_slope, htf_rising, htf_trend_up = 0, 0, False, False
        else:
            htf_adx_s = calc_adx(wdf, ADX_LEN)
            htf_adx = htf_adx_s.iloc[-1] if not np.isnan(htf_adx_s.iloc[-1]) else 0
            htf_slope = (htf_adx - htf_adx_s.iloc[-1-ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if len(htf_adx_s) > ADX_SLOPE_LEN else 0
            htf_rising = htf_slope > ADX_SLOPE_THRESH
            htf_ema_f = ema(wdf['Close'], EMA_FAST)
            htf_ema_s = ema(wdf['Close'], EMA_SLOW)
            htf_trend_up = htf_ema_f.iloc[-1] > htf_ema_s.iloc[-1]

        # === GÜNLÜK ADX ===
        adx_s = calc_adx(df, ADX_LEN)
        adx_val = adx_s.iloc[-1] if not np.isnan(adx_s.iloc[-1]) else 0
        adx_slope = (adx_val - adx_s.iloc[-1-ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if n > ADX_SLOPE_LEN else 0
        adx_rising = adx_slope > ADX_SLOPE_THRESH

        # === REJİM ===
        trend_up_count = int(ema_trend_up) + int(super_trend_up) + int(htf_trend_up)
        confirmed_trend_up = trend_up_count >= 2 and c.iloc[-1] > ema_s.iloc[-1]

        if htf_adx > ADX_TREND and htf_rising:
            htf_r = 2
        elif htf_adx > ADX_TREND:
            htf_r = 1
        elif htf_adx > ADX_CHOPPY:
            htf_r = 0
        else:
            htf_r = -1

        daily_confirm = adx_val > ADX_CHOPPY and adx_rising

        if not confirmed_trend_up:
            regime = 0
        elif htf_r == 2 and daily_confirm:
            regime = 3
        elif htf_r >= 1:
            regime = 2
        elif htf_r == 0:
            regime = 1
        else:
            regime = 0

        regime_name = {3: "FULL_TREND", 2: "TREND", 1: "GRI_BOLGE", 0: "CHOPPY"}[regime]
        if dbg: dbg['regime'][regime] = dbg['regime'].get(regime, 0) + 1

        # === WAVETREND ===
        wt = calc_wavetrend(df)
        wt_cross_up = bool(wt['cross_up'].iloc[-1])
        wt_recent = bool(wt['wt_recent'].iloc[-1])
        wt_bullish = bool(wt['wt_bullish'].iloc[-1])
        wt_killed = bool(wt['wt_killed'].iloc[-1])

        # === PMAX ===
        pm = calc_pmax(df)
        pmax_long = bool(pm['pmax_long'][-1])

        # === SMC ===
        smc = calc_smc(df)
        last_idx = n - 1
        bos_bar = smc['bos_bar'][last_idx]
        choch_bar = smc['choch_bar'][last_idx]
        swing_bias = smc['swing_bias']

        recent_bos = (last_idx - bos_bar) <= BOS_LOOKBACK and smc['bull_bos'].any() and swing_bias == 1
        recent_choch = (last_idx - choch_bar) <= BOS_LOOKBACK and smc['bull_choch'].any() and swing_bias == 1
        bos_tight = (last_idx - bos_bar) <= BOS_TIGHT and swing_bias == 1
        choch_tight = (last_idx - choch_bar) <= BOS_TIGHT and swing_bias == 1
        bos_age = last_idx - bos_bar if bos_bar >= 0 and (last_idx - bos_bar) <= BOS_LOOKBACK else None
        choch_age = last_idx - choch_bar if choch_bar >= 0 and (last_idx - choch_bar) <= BOS_LOOKBACK else None

        # === OB ===
        ob = calc_order_blocks(df)
        close_price = c.iloc[-1]
        nearest_ob = np.nan
        if not np.isnan(ob['ob_resist_bot']) and ob['ob_resist_bot'] > close_price:
            nearest_ob = (ob['ob_resist_bot'] + ob['ob_resist_top']) / 2

        # === SQUEEZE ===
        sq_basis = sma(c, SQ_LEN)
        sq_dev = c.rolling(SQ_LEN).std() * SQ_MULT_BB
        sq_rng = sma(true_range(df), SQ_LEN)
        sqz_on = ((sq_basis - sq_dev) > (sq_basis - SQ_MULT_KC * sq_rng)) & \
                 ((sq_basis + sq_dev) < (sq_basis + SQ_MULT_KC * sq_rng))
        # Squeeze momentum (linreg proxy)
        hh = h.rolling(SQ_LEN).max()
        ll = l.rolling(SQ_LEN).min()
        sq_mid = (hh + ll) / 2
        sq_mom_src = c - (sq_mid + sq_basis) / 2
        # Simple linreg approximation
        sq_mom = sq_mom_src.rolling(SQ_LEN).mean()
        sq_release = (~sqz_on) & sqz_on.shift(1) & (sq_mom > 0) & (sq_mom > sq_mom.shift(1))
        sq_release_last = any(bool(sq_release.iloc[-1-i]) for i in range(3) if len(sq_release) > i)  # son 3 gün

        atr_ma = sma(atr, 20)
        atr_expanding = any(atr.iloc[-1-i] > atr_ma.iloc[-1-i] * 1.05 for i in range(3) if len(atr) > i)  # son 3 gün, %5 yeterli

        trend_sq = sq_release_last or (sq_mom.iloc[-1] > 0 and sq_mom.iloc[-1] > sq_mom.iloc[-2] and not sqz_on.iloc[-1])

        # === BB / MR ===
        bb_basis = sma(c, BB_LEN)
        bb_dev_val = c.rolling(BB_LEN).std() * BB_MULT
        bb_upper = bb_basis + bb_dev_val
        bb_lower = bb_basis - bb_dev_val
        bb_pctb = (c - bb_lower) / (bb_upper - bb_lower)
        donch_lower = l.rolling(DONCH_LEN).min()
        rsi_short = _rsi(c, MR_RSI_LEN)

        # === HACİM META (DÖNÜŞ tier için) ===
        # CMF (Chaikin Money Flow — 20 gün)
        _hl_range = (h - l).replace(0, np.nan)
        _mfv = ((c - l) - (h - c)) / _hl_range * v
        _cmf_val = float(_mfv.rolling(20).sum().iloc[-1] / v.rolling(20).sum().iloc[-1]) \
            if v.rolling(20).sum().iloc[-1] > 0 else 0

        # OBV EMA slope (13 EMA, 5 gün slope)
        _obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        _obv_ema = _obv.ewm(span=13, adjust=False).mean()
        _obv_slp = float(_obv_ema.diff(5).iloc[-1]) if len(_obv_ema) > 5 else 0

        # === QUALITY ===
        rvol = v.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 0
        part_score = int(_cmf_val > 0) + int(rvol > 1) + int(_obv_slp > 0)
        _atr_pct = round(atr_val / close_price * 100, 2) if close_price > 0 else 99
        candle_range = h.iloc[-1] - l.iloc[-1]
        clv = (c.iloc[-1] - l.iloc[-1]) / candle_range if candle_range > 0 else 0.5
        upper_wick = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        range_atr = candle_range / atr_val if atr_val > 0 else 0

        rvol_s = 25 if rvol >= 2 else 20 if rvol >= RVOL_THRESH else 10 if rvol >= 1 else 0
        clv_s = 25 if clv >= 0.75 else 15 if clv >= 0.5 else 5 if clv >= 0.25 else 0
        wick_s = 25 if wick_ratio <= 0.15 else 15 if wick_ratio <= 0.3 else 5 if wick_ratio <= 0.5 else 0
        range_s = 25 if range_atr >= 1.2 else 15 if range_atr >= 0.8 else 5 if range_atr >= 0.5 else 0
        quality = rvol_s + clv_s + wick_s + range_s
        q_pass_gri = quality >= QUAL_MIN_GRI
        q_pass_trend = quality >= QUAL_MIN_TREND

        # === RS ===
        rs_pass = True
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2 + 5:
            # Align dates
            aligned = pd.DataFrame({'stock': c, 'bench': xu_df['Close']}).dropna()
            if len(aligned) >= RS_LEN2 + 5:
                sc = aligned['stock']
                bc = aligned['bench']
                sp1 = (sc.iloc[-1] - sc.iloc[-1-RS_LEN1]) / sc.iloc[-1-RS_LEN1] * 100
                sp2 = (sc.iloc[-1] - sc.iloc[-1-RS_LEN2]) / sc.iloc[-1-RS_LEN2] * 100
                bp1 = (bc.iloc[-1] - bc.iloc[-1-RS_LEN1]) / bc.iloc[-1-RS_LEN1] * 100
                bp2 = (bc.iloc[-1] - bc.iloc[-1-RS_LEN2]) / bc.iloc[-1-RS_LEN2] * 100
                rs_score = (sp1 - bp1) * 0.6 + (sp2 - bp2) * 0.4
                rs_pass = rs_score > RS_THRESHOLD

        # ============================================================
        # SİNYAL TESPİTİ (v3 mantığı birebir)
        # ============================================================

        vol_high = v.iloc[-1] > vol_sma20.iloc[-1] * RVOL_THRESH

        # A: TREND (regime >= 2)
        strong = regime >= 2 and ema_trend_up and super_trend_up and trend_sq and vol_high and rs_pass and q_pass_trend
        weak = regime >= 2 and ema_trend_up and super_trend_up and trend_sq and (not vol_high) and rs_pass and q_pass_trend

        # B: PULLBACK
        pb_rsi = _rsi(c, 5)
        pb_dipped = any(pb_rsi.iloc[-i] < 40 and pb_rsi.iloc[-i] > 20 for i in range(1, 6) if -i >= -n)
        pb_vol_dry = any(v.iloc[-i] < vol_sma20.iloc[-i] * 0.8 for i in range(1, 6) if -i >= -n)
        pb_reclaim = any(c.iloc[-1-i] > ema_f.iloc[-1-i] and c.iloc[-2-i] <= ema_f.iloc[-2-i] for i in range(3) if n > 2+i)
        pullback = confirmed_trend_up and super_trend_up and rs_pass and pb_dipped and pb_vol_dry and pb_reclaim and (q_pass_gri if regime == 1 else q_pass_trend)

        # C: SQUEEZE EXPANSION
        sq_exp = regime >= 1 and confirmed_trend_up and rs_pass and sq_release_last and atr_expanding and c.iloc[-1] > sq_basis.iloc[-1] and clv >= 0.4 and q_pass_gri

        # D: MEAN REVERSION
        mr_bb = l.iloc[-1] <= bb_lower.iloc[-1] and c.iloc[-1] > bb_lower.iloc[-1]
        mr_donch = l.iloc[-1] <= donch_lower.iloc[-2] and c.iloc[-1] > donch_lower.iloc[-2] if len(donch_lower) > 1 else False
        mean_rev = regime <= 1 and (mr_bb or mr_donch) and rsi_short.iloc[-1] < MR_RSI_THRESH and rsi_short.iloc[-1] > rsi_short.iloc[-2] and c.iloc[-1] > ema_s.iloc[-1] * 0.90  # EMA55'ten max %10 uzak

        # E: DONUS
        ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
        recent_e55 = any(ema55_cross.iloc[-i] for i in range(1, 11) if -i >= -n)
        recent_wt_cross = any(wt['cross_up'].iloc[-i] for i in range(1, 11) if -i >= -n)
        ema55_dist = (c.iloc[-1] - ema_s.iloc[-1]) / ema_s.iloc[-1] * 100
        approaching = -3 < ema55_dist < 3 and c.iloc[-1] > c.iloc[-4] and c.iloc[-4] > c.iloc[-7] if n > 7 else False
        donus = (recent_e55 and recent_wt_cross and rs_pass and c.iloc[-1] > ema_s.iloc[-1]) or \
                (recent_wt_cross and rs_pass and approaching and v.iloc[-1] > vol_sma20.iloc[-1])

        # F: ERKEN
        sw_hl = h.rolling(20).max().shift(1)
        struct_break = c.iloc[-1] > sw_hl.iloc[-1] if not np.isnan(sw_hl.iloc[-1]) else False
        mom_up5 = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100 if n > 6 else 0
        green_cnt = sum(1 for i in range(5) if c.iloc[-1-i] > o.iloc[-1-i])
        adx_turn = adx_val > adx_s.iloc[-2] and adx_s.iloc[-2] > adx_s.iloc[-3] if n > 3 else False
        early_rsi = _rsi(c, 14).iloc[-1]
        highest5 = c.rolling(5).max().iloc[-1]
        early_struct = struct_break and v.iloc[-1] > vol_sma20.iloc[-1] * 1.2 and adx_turn
        early_mom = mom_up5 > 5.0 and c.iloc[-1] >= highest5 and green_cnt >= 3 and v.iloc[-1] > vol_sma20.iloc[-1] * 1.2
        early = regime <= 1 and (early_struct or early_mom) and early_rsi < 75  # RS zorunlu değil, düşük rejimde normal

        # G: COMBO
        combo_base = (wt_cross_up or wt_recent) and wt_bullish and pmax_long
        combo_plus = combo_base and choch_tight
        combo_bos = combo_base and bos_tight and not choch_tight

        # Debug bileşenler
        if dbg:
            if wt_cross_up or wt_recent: dbg['wt_recent'] += 1
            if pmax_long: dbg['pmax_long'] += 1
            if recent_bos: dbg['bos'] += 1
            if recent_choch: dbg['choch'] += 1
            if combo_base: dbg['combo_base'] += 1
            if combo_plus: dbg['combo_plus'] += 1
            if combo_bos: dbg['combo_bos'] += 1
            if not rs_pass: dbg['rs_fail'] += 1
            if strong: dbg['strong_check'] += 1
            if weak: dbg['weak_check'] += 1
            if donus: dbg['donus_check'] += 1
            if early: dbg['early_check'] += 1
            if pullback: dbg['pb_check'] += 1
            if sq_exp: dbg['sq_check'] += 1
            if mean_rev: dbg['mr_check'] += 1

            # PB sub-conditions debug
            if confirmed_trend_up and super_trend_up:
                dbg['pb_trend_ok'] = dbg.get('pb_trend_ok', 0) + 1
                if pb_dipped: dbg['pb_dipped'] = dbg.get('pb_dipped', 0) + 1
                if pb_vol_dry: dbg['pb_vol_dry'] = dbg.get('pb_vol_dry', 0) + 1
                if pb_reclaim: dbg['pb_reclaim'] = dbg.get('pb_reclaim', 0) + 1
            # SQ sub-conditions
            if regime >= 1 and confirmed_trend_up:
                dbg['sq_regime_ok'] = dbg.get('sq_regime_ok', 0) + 1
                if sq_release_last: dbg['sq_release'] = dbg.get('sq_release', 0) + 1
                if atr_expanding: dbg['sq_atr_exp'] = dbg.get('sq_atr_exp', 0) + 1
            # MR sub-conditions
            if regime <= 1:
                dbg['mr_regime_ok'] = dbg.get('mr_regime_ok', 0) + 1
                if mr_bb: dbg['mr_bb_hit'] = dbg.get('mr_bb_hit', 0) + 1
                if mr_donch: dbg['mr_donch_hit'] = dbg.get('mr_donch_hit', 0) + 1
            # ERKEN sub
            if regime <= 1:
                if struct_break: dbg['early_struct_brk'] = dbg.get('early_struct_brk', 0) + 1
                if early_mom: dbg['early_mom_ok'] = dbg.get('early_mom_ok', 0) + 1

        # === SİNYAL PRİORİTE ===
        signal = None
        if combo_plus:
            signal = "CMB+"
        elif combo_bos:
            signal = "CMB"
        elif strong:
            signal = "GUCLU"
        elif weak:
            signal = "ZAYIF"
        elif donus:
            signal = "DONUS"
        elif early:
            signal = "ERKEN"
        elif pullback:
            signal = "PB"
        elif sq_exp:
            signal = "SQ"
        elif mean_rev:
            signal = "MR"

        if signal is None:
            # Combo bileşenleri — en az 2 aktif ve RS pozitif ise raporla
            has_wt = wt_recent or wt_cross_up
            has_pmax = pmax_long
            has_smc = recent_bos or recent_choch
            active = sum([has_wt, has_pmax, has_smc])
            if active >= 2 and rs_pass:
                signal = "BILESEN"
            else:
                if dbg: dbg['low_active'] += 1
                return None

        if dbg: dbg['signal'] += 1

        # === STOP / TP ===
        stop_mult = {
            "CMB+": COMBO_STOP, "CMB": COMBO_STOP,
            "GUCLU": TREND_STOP, "ZAYIF": TREND_STOP,
            "PB": GRI_STOP, "SQ": GRI_STOP, "ERKEN": GRI_STOP,
            "DONUS": DONUS_STOP, "MR": MR_STOP,
            "BILESEN": TREND_STOP,
        }[signal]

        if signal in ("CMB+", "CMB"):
            atr_tp = close_price + atr_val * COMBO_TP
            if not np.isnan(nearest_ob) and nearest_ob > close_price and nearest_ob < atr_tp:
                tp_price = nearest_ob
                tp_src = "OB"
            else:
                tp_price = atr_tp
                tp_src = "ATR"
        elif signal == "MR":
            tp_price = bb_basis.iloc[-1]
            tp_src = "BB"
        elif signal == "PB":
            tp_price = close_price + atr_val * GRI_TP
            tp_src = "ATR"
        elif signal == "DONUS":
            tp_price = close_price + atr_val * DONUS_TP
            tp_src = "ATR"
        elif signal == "ZAYIF":
            tp_price = close_price + atr_val * TREND_TP * 0.75  # 2.25x ATR, R:R ~1.13
            tp_src = "ATR"
        elif signal == "ERKEN":
            tp_price = close_price + atr_val * TREND_TP * 0.8  # 2.4x ATR
            tp_src = "ATR"
        else:
            tp_price = close_price + atr_val * TREND_TP
            tp_src = "ATR"

        stop_price = close_price - atr_val * stop_mult
        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        return {
            'ticker': ticker,
            'signal': signal,
            'regime': regime_name,
            'regime_score': regime,
            'close': round(close_price, 2),
            'stop': round(stop_price, 2),
            'tp': round(tp_price, 2),
            'tp_src': tp_src,
            'rr': round(rr, 2),
            'atr': round(atr_val, 2),
            'quality': int(quality),
            'rs_score': round(rs_score, 1),
            'rs_pass': rs_pass,
            'rvol': round(rvol, 1),
            'bb_pctb': round(float(bb_pctb.iloc[-1]), 2) if not np.isnan(bb_pctb.iloc[-1]) else None,
            # Bileşenler
            'wt_cross': wt_cross_up,
            'wt_recent': wt_recent,
            'wt_bullish': wt_bullish,
            'wt_killed': wt_killed,
            'wt1': round(float(wt['wt1'].iloc[-1]), 1),
            'wt2': round(float(wt['wt2'].iloc[-1]), 1),
            'pmax_long': pmax_long,
            'smc_bos': recent_bos,
            'smc_choch': recent_choch,
            'bos_age': bos_age,
            'choch_age': choch_age,
            'swing_bias': swing_bias,
            'ob_resist': f"{ob['ob_resist_bot']:.2f}-{ob['ob_resist_top']:.2f}" if not np.isnan(ob.get('ob_resist_top', np.nan)) else None,
            # Trend
            'ema_trend': ema_trend_up,
            'supertrend': super_trend_up,
            'adx': round(adx_val, 1),
            'adx_slope': round(adx_slope, 2),
            'htf_adx': round(htf_adx, 1),
            'squeeze': bool(sqz_on.iloc[-1]),
            'sq_release': sq_release_last,
            # Meta
            'turnover_m': round(avg_turnover / 1e6, 1),
            'vol_ratio': round(rvol, 1),
            # Hacim meta (DONUS tier)
            'cmf': round(_cmf_val, 3),
            'part_score': part_score,
            'atr_pct': _atr_pct,
        }

    except Exception as e:
        if dbg:
            dbg['exception'] += 1
            if dbg['exception'] <= 3:  # ilk 3 hatayı detaylı göster
                print(f"  [HATA] {ticker}: {type(e).__name__}: {e}")
                print(f"         Kolonlar: {list(df.columns)[:10]}")
                print(f"         Kolon tipi: {type(df.columns)}")
                if isinstance(df.columns, pd.MultiIndex):
                    print(f"         Levels: {df.columns.names}")
                    print(f"         İlk 5: {list(df.columns[:5])}")
        return None


def _rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram yok, konsola yazdırılıyor:")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram max 4096 char
    for i in range(0, len(msg), 4000):
        chunk = msg[i:i+4000]
        try:
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                "parse_mode": "HTML", "disable_web_page_preview": True
            }, timeout=10)
        except:
            pass


def send_telegram_document(filepath):
    """HTML dosyasını Telegram'a document olarak gönder."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if not os.path.exists(filepath):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        with open(filepath, 'rb') as f:
            requests.post(url, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": "📊 Detaylı rapor — tarayıcıda aç"
            }, files={"document": f}, timeout=30)
        print("📤 HTML rapor Telegram'a gönderildi")
    except Exception as e:
        print(f"⚠️ Telegram document hata: {e}")


def format_telegram_summary(results, total, html_url=None):
    """Telegram'a kısa özet — sadece en iyi sinyaller + link."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    signal_groups = {}
    for r in results:
        signal_groups.setdefault(r['signal'], []).append(r)

    priority = ["CMB+", "CMB", "GUCLU", "ZAYIF", "DONUS", "ERKEN", "PB", "SQ", "MR"]
    emoji_map = {"CMB+": "🔴", "CMB": "🟠", "GUCLU": "🟢", "ZAYIF": "🟡",
                 "DONUS": "🟣", "ERKEN": "🟠", "PB": "🔵", "SQ": "🟩", "MR": "⚪"}

    lines = [f"<b>📊 BIST REJİM v3 — {now}</b>", ""]

    if not results:
        lines.append("Bugün sinyal yok. 🔇")
        lines.append(f"📋 Taranan: {total}")
        return "\n".join(lines)

    # Dağılım özeti
    dist = []
    for sig in priority:
        cnt = len(signal_groups.get(sig, []))
        if cnt > 0:
            dist.append(f"{emoji_map[sig]}{sig}:{cnt}")
    bilesen_cnt = len(signal_groups.get("BILESEN", []))
    lines.append(f"📋 {total} taranan | {len(results)} sinyal" + (f" (+{bilesen_cnt} bileşen)" if bilesen_cnt else ""))
    lines.append(" ".join(dist))
    lines.append("")

    # En iyi sinyaller — CMB+, GUCLU, DONUS, ERKEN (hepsi), CMB ve ZAYIF (RS+ top 5)
    top_signals = []

    for sig in ["CMB+", "GUCLU", "DONUS", "ERKEN", "PB", "SQ", "MR"]:
        for r in signal_groups.get(sig, []):
            top_signals.append(r)

    # CMB'den RS+ ve Q>=50 olanlar, en iyi 5
    cmb_best = sorted(
        [r for r in signal_groups.get("CMB", []) if r['rs_score'] > 0 and r['quality'] >= 50],
        key=lambda x: (-x['quality'], -x['rs_score'])
    )[:5]
    top_signals.extend(cmb_best)

    # ZAYIF'tan en iyi 3
    zayif_best = sorted(
        [r for r in signal_groups.get("ZAYIF", [])],
        key=lambda x: (-x['quality'], -x['rs_score'])
    )[:3]
    top_signals.extend(zayif_best)

    # Sırala
    prio = {"CMB+": 0, "CMB": 1, "GUCLU": 2, "ZAYIF": 3, "DONUS": 4,
            "ERKEN": 5, "PB": 6, "SQ": 7, "MR": 8}
    top_signals.sort(key=lambda x: (prio.get(x['signal'], 99), -x['rr']))

    lines.append(f"<b>⭐ Öne Çıkanlar ({len(top_signals)})</b>")
    lines.append("─────────────────")

    for r in top_signals:
        emoji = emoji_map.get(r['signal'], "")
        rs_warn = "⚠️" if r['rs_score'] < 0 else ""
        regime_short = {"FULL_TREND": "FT", "TREND": "TR", "GRI_BOLGE": "GR", "CHOPPY": "CH"}
        rs = regime_short.get(r['regime'], "?")
        ob_str = f" OB" if r['ob_resist'] else ""
        lines.append(
            f"{emoji}<b>{r['ticker']}</b> {r['close']} [{r['signal']}] {rs}{rs_warn}\n"
            f"  S:{r['stop']} TP:{r['tp']} R:R={r['rr']} RS:{r['rs_score']} Q:{r['quality']}{ob_str}"
        )

    lines.append("")

    if html_url:
        lines.append(f"🔗 <a href=\"{html_url}\">Detaylı Rapor</a>")
    lines.append(f"\n📋 Taranan: {total} | Toplam Sinyal: {len(results)}")

    return "\n".join(lines)


def generate_html_report(results, total):
    """Filtrelenebilir, sıralanabilir HTML rapor oluştur."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    date_str = datetime.now().strftime('%Y%m%d')

    signal_groups = {}
    for r in results:
        signal_groups.setdefault(r['signal'], []).append(r)

    priority = ["CMB+", "CMB", "GUCLU", "ZAYIF", "DONUS", "ERKEN", "PB", "SQ", "MR", "BILESEN"]

    # Her sinyal için renk
    sig_colors = {
        "CMB+": "#dc2626", "CMB": "#ea580c", "GUCLU": "#16a34a", "ZAYIF": "#ca8a04",
        "DONUS": "#9333ea", "ERKEN": "#ea580c", "PB": "#2563eb", "SQ": "#059669",
        "MR": "#6b7280", "BILESEN": "#9ca3af"
    }
    regime_colors = {
        "FULL_TREND": "#fbbf24", "TREND": "#3b82f6", "GRI_BOLGE": "#d1d5db", "CHOPPY": "#374151"
    }

    # JSON veri — numpy tiplerini Python native'e çevir
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    rows_json = json.dumps(sanitize(results), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BIST Rejim v3 — {now}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 12px; }}
  .header {{ text-align: center; padding: 16px 0; border-bottom: 1px solid #334155; margin-bottom: 16px; }}
  .header h1 {{ font-size: 1.4rem; color: #f8fafc; }}
  .header .sub {{ color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }}

  .summary {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 16px; }}
  .summary .chip {{ padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
                    cursor: pointer; transition: all 0.2s; border: 2px solid transparent; }}
  .summary .chip:hover {{ transform: scale(1.05); }}
  .summary .chip.active {{ border-color: #f8fafc; box-shadow: 0 0 8px rgba(255,255,255,0.3); }}

  .filters {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 12px; }}
  .filters select, .filters input {{ background: #1e293b; color: #e2e8f0; border: 1px solid #475569;
                                      padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ background: #1e293b; color: #94a3b8; padding: 8px 6px; text-align: left;
        cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 10;
        border-bottom: 2px solid #334155; white-space: nowrap; }}
  th:hover {{ color: #f8fafc; }}
  th .arrow {{ font-size: 0.6rem; margin-left: 3px; }}
  td {{ padding: 6px; border-bottom: 1px solid #1e293b; white-space: nowrap; }}
  tr:hover {{ background: #1e293b; }}
  tr.highlight {{ background: #1e3a5f; }}

  .sig-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-weight: 700;
                font-size: 0.7rem; color: #fff; }}
  .regime-badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .rs-neg {{ color: #ef4444; }}
  .rs-pos {{ color: #22c55e; }}
  .q-high {{ color: #fbbf24; font-weight: 700; }}
  .q-low {{ color: #6b7280; }}
  .rvol-high {{ color: #f97316; font-weight: 700; }}

  .tv-link {{ color: #60a5fa; text-decoration: none; font-weight: 600; }}
  .tv-link:hover {{ text-decoration: underline; }}

  .footer {{ text-align: center; padding: 16px 0; color: #475569; font-size: 0.75rem; margin-top: 16px;
             border-top: 1px solid #334155; }}

  @media (max-width: 768px) {{
    table {{ font-size: 0.7rem; }}
    td, th {{ padding: 4px 3px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 BIST Rejim v3 Tarama</h1>
  <div class="sub">{now} · {total} taranan · {len(results)} sinyal</div>
</div>

<div class="summary" id="chips"></div>

<div class="filters">
  <select id="fRegime" onchange="applyFilters()">
    <option value="">Tüm Rejimler</option>
    <option value="FULL_TREND">Full Trend</option>
    <option value="TREND">Trend</option>
    <option value="GRI_BOLGE">Gri Bölge</option>
    <option value="CHOPPY">Choppy</option>
  </select>
  <select id="fRS" onchange="applyFilters()">
    <option value="">Tüm RS</option>
    <option value="pos">RS Pozitif</option>
    <option value="neg">RS Negatif</option>
    <option value="high">RS &gt; 20</option>
  </select>
  <select id="fQ" onchange="applyFilters()">
    <option value="">Tüm Kalite</option>
    <option value="60">Q ≥ 60</option>
    <option value="40">Q ≥ 40</option>
  </select>
  <input type="text" id="fSearch" placeholder="Hisse ara..." oninput="applyFilters()" style="width:100px;">
  <button style="background:#334155;color:#94a3b8;border:1px solid #475569;padding:6px 10px;border-radius:6px;font-size:0.8rem;cursor:pointer" onclick="document.getElementById('fwdNote').style.display=document.getElementById('fwdNote').style.display==='none'?'block':'none'">📊 Forward Test Notları</button>
</div>

<div id="fwdNote" style="display:none;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;margin-bottom:16px;font-size:0.78rem;line-height:1.7;color:#cbd5e1">
  <div style="font-weight:700;color:#38bdf8;margin-bottom:8px">📊 Forward Test Bulguları (10 günlük veri, Şubat 2026)</div>
  <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:10px">⚠ XU100 düşüş döneminde ölçüldü — yükseliş döneminde farklılık gösterebilir</div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #f59e0b">
      <div style="font-weight:700;color:#f59e0b;margin-bottom:4px">🏆 Genel 3 Kural (Tüm Sinyaller)</div>
      <div><span style="color:#22c55e;font-weight:600">1) RS 30-60</span> — düşük=zayıf, yüksek=aşırı uzamış</div>
      <div><span style="color:#22c55e;font-weight:600">2) MACD > 0</span> — momentum pozitif olmalı</div>
      <div><span style="color:#22c55e;font-weight:600">3) Q ≥ 70</span> — kalite skoru yüksek</div>
      <div style="color:#fbbf24;margin-top:4px">3 kural birlikte → <b>5G +9.52%, WR %73</b></div>
      <div style="color:#94a3b8;font-size:0.7rem;margin-top:2px">XU100 düşüşte RS 30-60 çok daha güçlü (+4.36% vs -0.47% uptrend)</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #16a34a">
      <div style="font-weight:700;color:#16a34a;margin-bottom:4px">🟢 GÜÇLÜ (N=109)</div>
      <div>Ham: 1G <span style="color:#ef4444">-0.98%</span> · 3G <span style="color:#ef4444">-1.51%</span> · 5G <span style="color:#ef4444">-1.87%</span></div>
      <div><span style="color:#22c55e">RS 30-60 + MACD>0</span>:</div>
      <div style="padding-left:8px">1G <b>+0.38%</b> WR%50 · 3G <b style="color:#22c55e">+1.74%</b> WR%63 · 5G <b style="color:#22c55e">+4.55%</b> WR<b>%75</b> (N=16)</div>
      <div style="color:#94a3b8;font-size:0.7rem">Q≥80 tek başına yetersiz · RS 40-60 bandı en iyi</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #9333ea">
      <div style="font-weight:700;color:#9333ea;margin-bottom:4px">🟣 DÖNÜŞ — Sürpriz Şampiyon (N=42)</div>
      <div>Ham: 1G <span style="color:#22c55e">+0.43%</span> · 3G <span style="color:#22c55e">+2.57%</span> · 5G <span style="color:#22c55e">+2.84%</span></div>
      <div style="color:#22c55e;font-weight:600">Tüm pencereler pozitif, WR >%50!</div>
      <div style="color:#ef4444;font-size:0.7rem">⚠ Q≥70+MACD>0 filtresi bozuyor (-7.45%) — filtresiz kullan</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #dc2626">
      <div style="font-weight:700;color:#dc2626;margin-bottom:4px">🔴 CMB+ (N=28) — En Kötü</div>
      <div>Ham: 1G +0.14% · 3G <span style="color:#ef4444">-1.60%</span> · 5G <span style="color:#ef4444">-3.68%</span> WR<span style="color:#ef4444">%22</span></div>
      <div style="color:#ef4444">⚠ RSI 70+ ise 5G -11% · Hiçbir filtre kurtaramıyor</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #ea580c">
      <div style="font-weight:700;color:#ea580c;margin-bottom:4px">🟠 CMB (N=242)</div>
      <div>Ham: 1G -0.02% · 3G <span style="color:#ef4444">-1.27%</span> · 5G <span style="color:#ef4444">-2.72%</span></div>
      <div><span style="color:#22c55e">Q≥70+MACD>0</span>: 1G <b>+0.97%</b> · 3G <b>+1.02%</b> · 5G <b style="color:#22c55e">+1.04%</b> (N=38)</div>
      <div><span style="color:#22c55e">Q≥80</span>: 5G <b style="color:#22c55e">+2.67%</b>, WR <b>%44</b></div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #ca8a04">
      <div style="font-weight:700;color:#ca8a04;margin-bottom:4px">🟡 ZAYIF (N=259) — En İyi Filtreli</div>
      <div>Ham: 1G <span style="color:#22c55e">+0.42%</span> · 3G -0.03% · 5G -0.80% (fazla getiri <span style="color:#22c55e">+1.22%</span>)</div>
      <div><span style="color:#22c55e">3 Kural</span>:</div>
      <div style="padding-left:8px">1G <b style="color:#22c55e">+2.72%</b> WR<b>%71</b> · 3G <b style="color:#22c55e">+2.48%</b> WR%50 · 5G <b style="color:#22c55e">+11.96%</b> WR<b>%71</b> (N=7-17)</div>
      <div style="color:#94a3b8;font-size:0.7rem">ADX&lt;20 sürpriz güçlü: 5G +7.76%, WR %86</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #9ca3af">
      <div style="font-weight:700;color:#9ca3af;margin-bottom:4px">⬜ BİLEŞEN (N=362)</div>
      <div>Ham: 1G -0.24% · 3G -0.62% · 5G -1.08%</div>
      <div>MACD>0 kritik: <span style="color:#22c55e">+0.92%</span> vs <span style="color:#ef4444">-3.91%</span></div>
      <div><span style="color:#22c55e">Q≥70+MACD>0</span>: 3G <b style="color:#22c55e">+4.84%</b> · 5G <b style="color:#22c55e">+9.58%</b> WR<b>%78</b> (N=9)</div>
    </div>

    <div style="background:#0f172a;border-radius:6px;padding:10px;border-left:3px solid #ef4444">
      <div style="font-weight:700;color:#ef4444;margin-bottom:4px">🚫 ERKEN — Kaçın (N=85)</div>
      <div>1G -0.67% · 3G <span style="color:#ef4444;font-weight:700">-6.17%</span> WR<span style="color:#ef4444">%12</span> · 5G <span style="color:#ef4444;font-weight:700">-7.53%</span> WR<span style="color:#ef4444">%11</span></div>
      <div style="color:#ef4444">En tehlikeli sinyal · Filtre de kurtaramıyor</div>
    </div>

  </div>
  <div style="color:#64748b;font-size:0.68rem;margin-top:8px;text-align:right">Forward test: 12-25 Şubat 2026 · XU100 düşüş dönemi · Kaynak: nox_project</div>
</div>

<table>
  <thead>
    <tr>
      <th onclick="sortBy('ticker')">Hisse <span class="arrow"></span></th>
      <th onclick="sortBy('signal')">Sinyal <span class="arrow"></span></th>
      <th onclick="sortBy('regime')">Rejim <span class="arrow"></span></th>
      <th onclick="sortBy('close')">Fiyat <span class="arrow"></span></th>
      <th onclick="sortBy('stop')">Stop <span class="arrow"></span></th>
      <th onclick="sortBy('tp')">TP <span class="arrow"></span></th>
      <th onclick="sortBy('rr')">R:R <span class="arrow"></span></th>
      <th onclick="sortBy('rs_score')">RS <span class="arrow"></span></th>
      <th onclick="sortBy('quality')">Q <span class="arrow"></span></th>
      <th onclick="sortBy('rvol')">RVOL <span class="arrow"></span></th>
      <th>OB</th>
      <th>BOS</th>
      <th>CHoCH</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<div class="footer">
  BIST Rejim v3 Screener · Veriler bilgi amaçlıdır, yatırım tavsiyesi değildir.
</div>

<script>
const DATA = {rows_json};

const SIG_COLORS = {json.dumps(sig_colors)};
const REG_COLORS = {json.dumps(regime_colors)};
const SIG_PRIO = {{"CMB+":0,"CMB":1,"GUCLU":2,"ZAYIF":3,"DONUS":4,"ERKEN":5,"PB":6,"SQ":7,"MR":8,"BILESEN":9}};

let sortCol = 'signal', sortAsc = true;
let activeChip = null;

function init() {{
  // Chip'leri oluştur
  const counts = {{}};
  DATA.forEach(r => {{ counts[r.signal] = (counts[r.signal]||0)+1; }});
  const chips = document.getElementById('chips');
  const prios = ["CMB+","CMB","GUCLU","ZAYIF","DONUS","ERKEN","PB","SQ","MR","BILESEN"];
  prios.forEach(sig => {{
    if (!counts[sig]) return;
    const c = document.createElement('span');
    c.className = 'chip';
    c.style.background = SIG_COLORS[sig] || '#6b7280';
    c.textContent = sig + ' (' + counts[sig] + ')';
    c.onclick = () => {{
      if (activeChip === sig) {{ activeChip = null; c.classList.remove('active'); }}
      else {{
        document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
        activeChip = sig; c.classList.add('active');
      }}
      applyFilters();
    }};
    chips.appendChild(c);
  }});
  applyFilters();
}}

function applyFilters() {{
  const regime = document.getElementById('fRegime').value;
  const rs = document.getElementById('fRS').value;
  const q = document.getElementById('fQ').value;
  const search = document.getElementById('fSearch').value.toUpperCase();

  let filtered = DATA.filter(r => {{
    if (activeChip && r.signal !== activeChip) return false;
    if (regime && r.regime !== regime) return false;
    if (rs === 'pos' && r.rs_score <= 0) return false;
    if (rs === 'neg' && r.rs_score >= 0) return false;
    if (rs === 'high' && r.rs_score <= 20) return false;
    if (q && r.quality < parseInt(q)) return false;
    if (search && !r.ticker.includes(search)) return false;
    return true;
  }});

  // Sırala
  filtered.sort((a,b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (sortCol === 'signal') {{ va = SIG_PRIO[va]||99; vb = SIG_PRIO[vb]||99; }}
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  }});

  renderTable(filtered);
}}

function sortBy(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = col === 'ticker' || col === 'signal' || col === 'regime'; }}
  applyFilters();
}}

function renderTable(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  data.forEach(r => {{
    const tr = document.createElement('tr');
    const isBest = ["CMB+","GUCLU"].includes(r.signal) || (r.signal==="CMB" && r.rs_score > 0 && r.quality >= 60);
    if (isBest) tr.classList.add('highlight');

    const rsClass = r.rs_score > 0 ? 'rs-pos' : 'rs-neg';
    const qClass = r.quality >= 60 ? 'q-high' : r.quality < 30 ? 'q-low' : '';
    const rvolStr = r.rvol >= 1.5 ? `<span class="rvol-high">${{r.rvol}}x</span>` : r.rvol + 'x';
    const regColor = REG_COLORS[r.regime] || '#6b7280';
    const regShort = {{"FULL_TREND":"FULL","TREND":"TREND","GRI_BOLGE":"GRİ","CHOPPY":"CHOP"}}[r.regime]||r.regime;

    tr.innerHTML = `
      <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
      <td><span class="sig-badge" style="background:${{SIG_COLORS[r.signal]||'#6b7280'}}">${{r.signal}}</span></td>
      <td><span class="regime-badge" style="background:${{regColor}};color:${{regColor==='#d1d5db'?'#1e293b':'#fff'}}">${{regShort}}</span></td>
      <td>${{r.close}}</td>
      <td>${{r.stop}}</td>
      <td>${{r.tp}} <small style="color:#64748b">${{r.tp_src}}</small></td>
      <td style="color:${{r.rr>=1.3?'#22c55e':r.rr>=1?'#fbbf24':'#ef4444'}};font-weight:700">${{r.rr}}</td>
      <td class="${{rsClass}}">${{r.rs_score}}</td>
      <td class="${{qClass}}">${{r.quality}}</td>
      <td>${{rvolStr}}</td>
      <td style="font-size:0.7rem;color:#94a3b8">${{r.ob_resist||'—'}}</td>
      <td style="color:#94a3b8">${{r.bos_age !== null && r.bos_age !== undefined ? r.bos_age+'g' : '—'}}</td>
      <td style="color:#94a3b8">${{r.choch_age !== null && r.choch_age !== undefined ? r.choch_age+'g' : '—'}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

init();
</script>
</body>
</html>"""
    return html


def push_html_to_github(html_content, date_str):
    """HTML raporu GitHub Pages repo'suna push et."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")  # örn: "alpkarakasli/bist-signals"

    if not token or not repo:
        print("⚠️ GH_TOKEN veya GH_PAGES_REPO tanımlı değil, HTML lokal kaydedildi.")
        return None

    # GitHub API ile dosya oluştur/güncelle
    api_url = f"https://api.github.com/repos/{repo}/contents/index.html"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    import base64
    content_b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')

    # Mevcut dosyanın SHA'sını al (güncelleme için gerekli)
    sha = None
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except:
        pass

    payload = {
        "message": f"📊 Günlük tarama — {date_str}",
        "content": content_b64,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            # GitHub Pages URL
            owner, name = repo.split("/")
            page_url = f"https://{owner}.github.io/{name}/"
            print(f"✅ HTML rapor yayınlandı: {page_url}")
            return page_url
        else:
            print(f"⚠️ GitHub push hata: {resp.status_code} — {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"⚠️ GitHub push hata: {e}")
        return None


# ============================================================
# ANA
# ============================================================

def main():
    print("=" * 60)
    print("📊 BIST REJİM v3 SCREENER")
    print("=" * 60)

    import sys
    debug_mode = '--debug' in sys.argv

    tickers = get_all_bist_tickers()
    all_data = fetch_data(tickers, period="1y")
    xu_df = fetch_benchmark(period="1y")

    print("🔄 Analiz ediliyor...\n")
    results = []
    # Debug sayaçları
    dbg = {'total': 0, 'no_atr': 0, 'low_vol': 0, 'no_signal': 0, 'low_active': 0,
           'exception': 0, 'signal': 0,
           'regime': {0: 0, 1: 0, 2: 0, 3: 0},
           'wt_recent': 0, 'pmax_long': 0, 'bos': 0, 'choch': 0,
           'rs_fail': 0, 'q_fail': 0,
           'strong_check': 0, 'weak_check': 0, 'donus_check': 0,
           'early_check': 0, 'pb_check': 0, 'sq_check': 0, 'mr_check': 0,
           'combo_base': 0, 'combo_plus': 0, 'combo_bos': 0}

    for t, df in all_data.items():
        r = analyze(t, df, xu_df, dbg if debug_mode else None)
        if r:
            results.append(r)

    # Debug raporu
    if debug_mode:
        print("\n" + "=" * 60)
        print("🔍 DEBUG RAPORU")
        print("=" * 60)
        print(f"  Toplam analiz: {dbg['total']}")
        print(f"  ATR yok/sıfır: {dbg['no_atr']}")
        print(f"  Düşük hacim:   {dbg['low_vol']}")
        print(f"  Exception:     {dbg['exception']}")
        print(f"  RS başarısız:  {dbg['rs_fail']}")
        print(f"  Quality düşük: {dbg['q_fail']}")
        print(f"\n  Rejim dağılımı:")
        for k2, v2 in dbg['regime'].items():
            name = {0: 'CHOPPY', 1: 'GRI', 2: 'TREND', 3: 'FULL_TREND'}[k2]
            print(f"    {name}: {v2}")
        print(f"\n  Bileşen durumu (son bar):")
        print(f"    WT recent/cross: {dbg['wt_recent']}")
        print(f"    PMAX long:       {dbg['pmax_long']}")
        print(f"    Recent BOS:      {dbg['bos']}")
        print(f"    Recent CHoCH:    {dbg['choch']}")
        print(f"    Combo base:      {dbg['combo_base']}")
        print(f"\n  Sinyal kontrol (True sayısı):")
        print(f"    GUCLU: {dbg['strong_check']}  ZAYIF: {dbg['weak_check']}")
        print(f"    DONUS: {dbg['donus_check']}  ERKEN: {dbg['early_check']}")
        print(f"    PB: {dbg['pb_check']}  SQ: {dbg['sq_check']}  MR: {dbg['mr_check']}")
        print(f"    CMB+: {dbg['combo_plus']}  CMB: {dbg['combo_bos']}")
        print(f"\n  Sinyal yok:     {dbg['no_signal']}")
        print(f"  Düşük aktif:    {dbg['low_active']}")
        print(f"  ✅ Sinyal var:  {dbg['signal']}")
        print(f"\n  📊 Alt koşul detayları:")
        print(f"    PB: trend_ok={dbg.get('pb_trend_ok',0)} dipped={dbg.get('pb_dipped',0)} vol_dry={dbg.get('pb_vol_dry',0)} reclaim={dbg.get('pb_reclaim',0)}")
        print(f"    SQ: regime_ok={dbg.get('sq_regime_ok',0)} release={dbg.get('sq_release',0)} atr_exp={dbg.get('sq_atr_exp',0)}")
        print(f"    MR: regime_ok={dbg.get('mr_regime_ok',0)} bb_hit={dbg.get('mr_bb_hit',0)} donch_hit={dbg.get('mr_donch_hit',0)}")
        print(f"    ERKEN: struct_brk={dbg.get('early_struct_brk',0)} mom_ok={dbg.get('early_mom_ok',0)}")
        print("=" * 60)

    # Sırala
    prio = {"CMB+": 0, "CMB": 1, "GUCLU": 2, "ZAYIF": 3, "DONUS": 4,
            "ERKEN": 5, "PB": 6, "SQ": 7, "MR": 8, "BILESEN": 9}
    results.sort(key=lambda x: (prio.get(x['signal'], 99), -x['rr']))

    # Konsol — kısa özet
    for r in results:
        emoji = {"CMB+": "🔴", "CMB": "🟠", "GUCLU": "🟢", "ZAYIF": "🟡",
                 "DONUS": "🟣", "ERKEN": "🟠", "PB": "🔵", "SQ": "🟩",
                 "MR": "⚪", "BILESEN": "⬜"}.get(r['signal'], "")
        print(f"{emoji} {r['ticker']:8s} [{r['signal']:6s}] {r['regime']:12s} "
              f"Fiyat:{r['close']:8.2f} S:{r['stop']:8.2f} TP:{r['tp']:8.2f}({r['tp_src']}) "
              f"R:R={r['rr']:.1f} RS:{r['rs_score']:5.1f} Q:{r['quality']} RVOL:{r['rvol']}x")

    print(f"\n📋 Taranan: {len(all_data)} | Sinyal: {len(results)}")

    date_str = datetime.now().strftime('%Y%m%d')

    # HTML rapor oluştur
    html_content = generate_html_report(results, len(all_data))
    html_fname = f"rejim_v3_{date_str}.html"
    with open(html_fname, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"📄 {html_fname}")

    # GitHub Pages'e push
    html_url = push_html_to_github(html_content, date_str)

    # Telegram — kısa özet + link
    msg = format_telegram_summary(results, len(all_data), html_url)
    send_telegram(msg)

    # Telegram'a HTML dosyasını da gönder (document olarak)
    send_telegram_document(html_fname)

    # CSV
    if results:
        df_out = pd.DataFrame(results)
        fname = f"rejim_v3_signals_{date_str}.csv"
        df_out.to_csv(fname, index=False)
        print(f"💾 {fname}")

    return results


if __name__ == "__main__":
    main()
