"""
NOX Agent — Makro Veri Toplama
DXY, USDTRY, EURTRY, XU100, SPY, QQQ, VIX, altın, petrol, bakır, BTC, ETH, TNX
yfinance batch download + trend hesaplama + rejim sınıflandırma
"""
import pandas as pd
import numpy as np
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)

# ── Makro Enstrüman Registry ──
MACRO_TICKERS = {
    # FX
    "DX-Y.NYB":   {"name": "DXY (Dolar Endeksi)", "category": "FX"},
    "USDTRY=X":   {"name": "USD/TRY", "category": "FX"},
    "EURTRY=X":   {"name": "EUR/TRY", "category": "FX"},
    # BIST
    "XU100.IS":   {"name": "BIST 100", "category": "BIST"},
    "XU030.IS":   {"name": "BIST 30", "category": "BIST"},
    "XBANK.IS":   {"name": "BIST Banka", "category": "BIST"},
    "XUTEK.IS":   {"name": "BIST Teknoloji", "category": "BIST"},
    "XHOLD.IS":   {"name": "BIST Holding", "category": "BIST"},
    "XELKT.IS":   {"name": "BIST Elektrik", "category": "BIST"},
    "XTRZM.IS":   {"name": "BIST Turizm", "category": "BIST"},
    # US
    "SPY":        {"name": "S&P 500 (SPY)", "category": "US"},
    "QQQ":        {"name": "NASDAQ 100 (QQQ)", "category": "US"},
    "^VIX":       {"name": "VIX", "category": "US"},
    # Emtia
    "GC=F":       {"name": "Altın (Gold)", "category": "Emtia"},
    "CL=F":       {"name": "WTI Petrol", "category": "Emtia"},
    "HG=F":       {"name": "Bakır (Copper)", "category": "Emtia"},
    # Kripto
    "BTC-USD":    {"name": "Bitcoin", "category": "Kripto"},
    "ETH-USD":    {"name": "Ethereum", "category": "Kripto"},
    # Faiz
    "^TNX":       {"name": "US 10Y Faiz", "category": "Faiz"},
}


def _normalize_df(df, ticker=None):
    """yfinance MultiIndex → düz DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker:
            try:
                cols = df.columns.get_level_values(0).unique().tolist()
                if 'Price' in cols:
                    sub = df.xs(ticker, level='Ticker', axis=1)
                else:
                    sub = df.xs(ticker, level=1, axis=1)
                sub = sub[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                return sub
            except Exception:
                pass
        try:
            df.columns = df.columns.droplevel(0)
        except Exception:
            try:
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass
    needed = ['Close']
    for col in needed:
        if col not in df.columns:
            return None
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    return df[cols].dropna()


def _calc_ema(series, span):
    """EMA hesapla."""
    return series.ewm(span=span, adjust=False).mean()


def _calc_change_pct(series, days):
    """N gün yüzde değişim."""
    if len(series) < days + 1:
        return None
    cur = float(series.iloc[-1])
    prev = float(series.iloc[-1 - days])
    if prev == 0:
        return None
    return round(((cur / prev) - 1) * 100, 2)


def _calc_rsi(close, period=14):
    """RSI hesapla (lightweight, pandas ewm)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _calc_trend(series, ema21):
    """Trend yönü: UP, DOWN, FLAT."""
    if len(series) < 5:
        return "N/A"
    last = float(series.iloc[-1])
    ema_val = float(ema21.iloc[-1])
    if last > ema_val * 1.005:
        return "UP"
    elif last < ema_val * 0.995:
        return "DOWN"
    return "FLAT"


def fetch_macro_data(period="6mo"):
    """Tüm makro enstrümanları tek tek download (MultiIndex sorunu önlemek için)."""
    tickers = list(MACRO_TICKERS.keys())
    print(f"📡 {len(tickers)} makro enstrüman çekiliyor...")
    result = {}

    for t in tickers:
        try:
            raw = yf.download(t, period=period, auto_adjust=True,
                              progress=False)
            if raw.empty:
                continue
            # MultiIndex temizle — yfinance yeni sürüm (Price, Ticker) döndürüyor
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel('Ticker')
            # Close kolonu zorunlu
            if 'Close' not in raw.columns:
                continue
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume']
                    if c in raw.columns]
            sub = raw[cols].dropna()
            if len(sub) >= 20:
                result[t] = sub
        except Exception as e:
            print(f"  ⚠️ {t} hata: {e}")

    print(f"✅ {len(result)}/{len(tickers)} makro enstrüman yüklendi")
    return result


def fetch_macro_snapshot(period="6mo"):
    """Makro snapshot: her enstrüman için son fiyat, değişimler, trend."""
    data = fetch_macro_data(period)
    snapshot = []

    for ticker, info in MACRO_TICKERS.items():
        df = data.get(ticker)
        if df is None or len(df) < 5:
            snapshot.append({
                "ticker": ticker,
                "name": info["name"],
                "category": info["category"],
                "price": None,
                "chg_1d": None,
                "chg_5d": None,
                "chg_1m": None,
                "above_ema21": None,
                "trend": "N/A",
            })
            continue

        try:
            close = df['Close'].squeeze()  # DataFrame → Series garanti
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ema21 = _calc_ema(close, 21)
            last_val = float(close.iloc[-1])
            ema_val = float(ema21.iloc[-1])
            last_price = round(last_val, 4)
            above_ema = last_val > ema_val

            snapshot.append({
                "ticker": ticker,
                "name": info["name"],
                "category": info["category"],
                "price": last_price,
                "chg_1d": _calc_change_pct(close, 1),
                "chg_5d": _calc_change_pct(close, 5),
                "chg_1m": _calc_change_pct(close, 21),
                "above_ema21": above_ema,
                "trend": _calc_trend(close, ema21),
            })
        except Exception as e:
            print(f"  ⚠️ {ticker} işleme hatası: {e}")
            snapshot.append({
                "ticker": ticker,
                "name": info["name"],
                "category": info["category"],
                "price": None, "chg_1d": None, "chg_5d": None,
                "chg_1m": None, "above_ema21": None, "trend": "N/A",
            })

    return snapshot


def calc_category_regimes(data):
    """
    Her kategori için rejim hesapla.
    EMA21 trend (1/0), RSI>50 (1/0), 20-gün range pozisyon → kategori skoru.
    Return: {category: {regime, score, instruments: [{name, trend, rsi, range_pct}]}}
    """
    # Enstrüman bazında metrik hesapla
    instruments = {}
    for ticker, info in MACRO_TICKERS.items():
        df = data.get(ticker)
        if df is None or len(df) < 21:
            continue
        try:
            close = df['Close'].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ema21 = _calc_ema(close, 21)
            rsi = _calc_rsi(close, 14)
            last_close = float(close.iloc[-1])
            ema_val = float(ema21.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            # 20 gün range pozisyonu (0-100)
            high_20 = float(close.iloc[-20:].max())
            low_20 = float(close.iloc[-20:].min())
            if high_20 != low_20:
                range_pct = round((last_close - low_20) / (high_20 - low_20) * 100, 1)
            else:
                range_pct = 50.0

            above_ema = 1 if last_close > ema_val else 0
            rsi_above = 1 if rsi_val > 50 else 0
            # Enstrüman skoru: 0-3
            inst_score = above_ema + rsi_above + (1 if range_pct > 60 else 0)

            instruments[ticker] = {
                "name": info["name"],
                "category": info["category"],
                "above_ema": above_ema,
                "rsi": round(rsi_val, 1),
                "range_pct": range_pct,
                "score": inst_score,
            }
        except Exception:
            continue

    # Kategori bazında aggregate
    cat_data = {}
    for ticker, inst in instruments.items():
        cat = inst["category"]
        cat_data.setdefault(cat, []).append(inst)

    result = {}
    for cat, items in cat_data.items():
        avg_score = sum(i["score"] for i in items) / len(items) if items else 0
        # Rejim belirleme
        if avg_score >= 2.3:
            regime = "GÜÇLÜ_YUKARI"
        elif avg_score >= 1.7:
            regime = "YUKARI"
        elif avg_score >= 1.0:
            regime = "NÖTR"
        elif avg_score >= 0.5:
            regime = "AŞAĞI"
        else:
            regime = "GÜÇLÜ_AŞAĞI"

        result[cat] = {
            "regime": regime,
            "score": round(avg_score, 2),
            "instruments": items,
        }

    return result


def assess_macro_regime(snapshot):
    """
    Makro rejim sınıflandırması.
    Kurallar:
      DXY↑ + USDTRY↑ = BIST olumsuz
      VIX > 20 = risk_off
      VIX < 15 = risk_on
      XU100 uptrend + DXY↓ = risk_on
      TNX↑ hızlı = büyüme endişesi
    """
    lookup = {s["ticker"]: s for s in snapshot}

    dxy = lookup.get("DX-Y.NYB", {})
    usdtry = lookup.get("USDTRY=X", {})
    vix = lookup.get("^VIX", {})
    xu100 = lookup.get("XU100.IS", {})
    spy = lookup.get("SPY", {})
    tnx = lookup.get("^TNX", {})
    gold = lookup.get("GC=F", {})
    btc = lookup.get("BTC-USD", {})

    signals = []
    regime = "NÖTR"
    risk_score = 0  # negatif = risk_off, pozitif = risk_on

    # VIX bazlı risk
    vix_price = vix.get("price")
    if vix_price is not None:
        if vix_price > 25:
            signals.append("🔴 VIX yüksek (>25) — güçlü risk_off")
            risk_score -= 3
        elif vix_price > 20:
            signals.append("🟡 VIX yüksek (>20) — risk_off")
            risk_score -= 1
        elif vix_price < 15:
            signals.append("🟢 VIX düşük (<15) — risk_on")
            risk_score += 1

    # DXY + USDTRY = BIST etkisi
    dxy_trend = dxy.get("trend")
    usdtry_trend = usdtry.get("trend")
    if dxy_trend == "UP" and usdtry_trend == "UP":
        signals.append("🔴 DXY↑ + USDTRY↑ — BIST olumsuz")
        risk_score -= 2
    elif dxy_trend == "DOWN":
        signals.append("🟢 DXY↓ — EM pozitif")
        risk_score += 1

    # XU100 trend
    xu100_trend = xu100.get("trend")
    if xu100_trend == "UP":
        signals.append("🟢 XU100 uptrend")
        risk_score += 1
    elif xu100_trend == "DOWN":
        signals.append("🔴 XU100 downtrend")
        risk_score -= 1

    # US piyasaları
    spy_trend = spy.get("trend")
    if spy_trend == "DOWN":
        signals.append("🔴 SPY downtrend — küresel risk_off")
        risk_score -= 1
    elif spy_trend == "UP":
        signals.append("🟢 SPY uptrend")
        risk_score += 1

    # TNX hızlı yükseliş
    tnx_5d = tnx.get("chg_5d")
    if tnx_5d is not None and tnx_5d > 5:
        signals.append("🟡 US 10Y faiz hızlı yükseliş — büyüme endişesi")
        risk_score -= 1

    # Altın güvenli liman
    gold_trend = gold.get("trend")
    if gold_trend == "UP" and risk_score < 0:
        signals.append("🟡 Altın uptrend + risk_off — güvenli liman talebi")

    # BTC risk iştahı
    btc_trend = btc.get("trend")
    if btc_trend == "UP":
        signals.append("🟢 BTC uptrend — risk iştahı")
        risk_score += 1
    elif btc_trend == "DOWN":
        signals.append("🔴 BTC downtrend")
        risk_score -= 1

    # Rejim belirleme
    if risk_score >= 3:
        regime = "GÜÇLÜ_RISK_ON"
    elif risk_score >= 1:
        regime = "RISK_ON"
    elif risk_score <= -3:
        regime = "GÜÇLÜ_RISK_OFF"
    elif risk_score <= -1:
        regime = "RISK_OFF"
    else:
        regime = "NÖTR"

    return {
        "regime": regime,
        "risk_score": risk_score,
        "signals": signals,
        "snapshot": snapshot,
    }


def format_macro_summary(macro_result):
    """Makro özeti Telegram mesajı formatında döndür."""
    regime = macro_result["regime"]
    risk_score = macro_result["risk_score"]
    signals = macro_result["signals"]
    snapshot = macro_result["snapshot"]

    regime_emoji = {
        "GÜÇLÜ_RISK_ON": "🟢🟢",
        "RISK_ON": "🟢",
        "NÖTR": "⚪",
        "RISK_OFF": "🔴",
        "GÜÇLÜ_RISK_OFF": "🔴🔴",
    }

    lines = [
        f"<b>🌍 Makro Rejim: {regime_emoji.get(regime, '')} {regime}</b> (skor: {risk_score})",
        "",
    ]

    # Kategori bazlı özet
    by_cat = {}
    for s in snapshot:
        cat = s["category"]
        by_cat.setdefault(cat, []).append(s)

    for cat in ["BIST", "US", "FX", "Emtia", "Kripto", "Faiz"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        lines.append(f"<b>{cat}</b>")
        for item in items:
            price = item.get("price")
            if price is None:
                continue
            chg_1d = item.get("chg_1d")
            chg_5d = item.get("chg_5d")
            trend = item.get("trend", "")
            trend_icon = "↑" if trend == "UP" else "↓" if trend == "DOWN" else "→"

            chg_str = ""
            if chg_1d is not None:
                chg_str += f"1G:{chg_1d:+.1f}%"
            if chg_5d is not None:
                chg_str += f" 5G:{chg_5d:+.1f}%"

            lines.append(f"  {trend_icon} {item['name']}: {price:,.2f} ({chg_str})")
        lines.append("")

    # Rejim sinyalleri
    if signals:
        lines.append("<b>Sinyaller:</b>")
        for s in signals:
            lines.append(f"  {s}")

    return "\n".join(lines)


if __name__ == "__main__":
    snapshot = fetch_macro_snapshot()
    result = assess_macro_regime(snapshot)
    print(format_macro_summary(result))
