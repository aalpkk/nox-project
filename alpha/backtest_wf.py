"""
Alpha Pipeline — Walk-Forward Portföy Backtester
==================================================
Her rebalance tarihinde:
  1. scan_universe() → Aşama 1-3
  2. build_portfolio() → Aşama 4-5
  3. Portföy yeniden dengele (komisyon + slippage)
  4. Günlük stop-loss kontrolü
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import OrderedDict

from core.indicators import calc_atr
from alpha.config import (
    WF_TRAIN_DAYS, WF_STEP_DAYS, INITIAL_CAPITAL,
    COMMISSION_PCT, SLIPPAGE_PCT,
    POSITION_STOP_PCT, POSITION_STOP_ATR_MULT, PORTFOLIO_DD_LIMIT,
    MIN_DATA_DAYS, MIN_VOLUME_TL, MAX_HOLD_DAYS,
    TRAILING_TRIGGER_PCT, TRAILING_STOP_PCT,
    TRAILING_TRIGGER_ATR, TRAILING_ATR_MULT,
    REGIME_EMA_LENGTH, REGIME_BULL_WEIGHT, REGIME_BEAR_WEIGHT,
    REGIME_KAMA_ENABLED, KAMA_ER_PERIOD, KAMA_FAST, KAMA_SLOW,
    REGIME_AST_ENABLED, AST_ATR_LEN, AST_FACTOR, AST_TRAINING,
    AST_HI_INIT, AST_MID_INIT, AST_LO_INIT,
    BATCH_MODE, BATCH_REOPEN_THRESHOLD,
    RS_LOOKBACK, RS_MIN_OUTPERFORM, MOM_LOOKBACK,
    PORTFOLIO_METHOD, MAX_STOCKS, TARGET_PORTFOLIO_SIZE,
    ML_STAGE1_ENABLED, ML_SCORE_THRESHOLD, ML_SWING_THRESHOLD,
    ML_SLOPE_LOOKBACK, ML_SLOPE_MIN, ML_COMPOSITE_WEIGHT,
    CONFIRMATION_MIN_SCORE,
)
from alpha.stages import scan_universe
from alpha.portfolio import build_portfolio


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    weight: float
    shares: float
    stop_price: float
    atr_at_entry: float
    stop_mult: float = 2.0
    highest_since_entry: float = 0.0
    trailing_active: bool = False      # iz sürme modu aktif mi
    first_tp_done: bool = False        # ilk %50 kâr alım yapıldı mı
    prev_day_low: float = 0.0         # önceki günün Low'u
    be_shifted: bool = False           # breakeven shift: +N·R sonra stop entry'ye çekildi


@dataclass
class TradeRecord:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    weight: float
    pnl_pct: float
    exit_reason: str
    hold_days: int


@dataclass
class RebalanceEvent:
    date: pd.Timestamp
    old_tickers: list
    new_tickers: list
    candidates_total: int
    candidates_passed: int
    final_stocks: int
    turnover: float
    sharpe: float


def _adaptive_supertrend_bear_flag(high: pd.Series, low: pd.Series, close: pd.Series,
                                    atr_len: int, factor: float, training: int,
                                    hi_init: float, mid_init: float, lo_init: float,
                                    max_iter: int = 30) -> pd.Series:
    """AlgoAlpha Adaptive SuperTrend — K-means vol clustering + SuperTrend.

    Returns bear_flag (1=bear, 0=bull) series aligned to `close.index`.
    Pine direction convention: direction==1 means close broke below lower band = bear.
    No look-ahead: K-means her bar için past training-bar ATR'leri ile çalışır.
    """
    # ── Wilder ATR (pine ta.atr) ──
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_len, adjust=False).mean()

    n = len(close)
    atr_vals = atr.values
    adaptive_atr = np.full(n, np.nan)

    for i in range(training, n):
        window = atr_vals[i - training:i]
        window = window[~np.isnan(window)]
        if len(window) < 10:
            continue
        lo_v = window.min(); hi_v = window.max()
        rng = hi_v - lo_v
        if rng <= 0:
            adaptive_atr[i] = atr_vals[i]
            continue
        h = lo_v + rng * hi_init
        m = lo_v + rng * mid_init
        l = lo_v + rng * lo_init
        for _ in range(max_iter):
            d_h = np.abs(window - h)
            d_m = np.abs(window - m)
            d_l = np.abs(window - l)
            stacked = np.stack([d_h, d_m, d_l], axis=0)
            labels = np.argmin(stacked, axis=0)
            mh = window[labels == 0]; mm = window[labels == 1]; ml = window[labels == 2]
            new_h = mh.mean() if len(mh) else h
            new_m = mm.mean() if len(mm) else m
            new_l = ml.mean() if len(ml) else l
            if np.isclose(new_h, h) and np.isclose(new_m, m) and np.isclose(new_l, l):
                h, m, l = new_h, new_m, new_l
                break
            h, m, l = new_h, new_m, new_l
        # Current bar ATR → nearest centroid
        curr = atr_vals[i]
        if np.isnan(curr):
            continue
        dists = [abs(curr - h), abs(curr - m), abs(curr - l)]
        centroids = [h, m, l]
        adaptive_atr[i] = centroids[int(np.argmin(dists))]

    # ── SuperTrend with adaptive ATR ──
    hl2 = (high.values + low.values) / 2.0
    cl = close.values
    ub = hl2 + factor * adaptive_atr
    lb = hl2 - factor * adaptive_atr

    direction = np.ones(n, dtype=np.int8)  # 1=bear, -1=bull (pine convention)
    st = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        if i > 0 and not np.isnan(lb[i-1]):
            if not (lb[i] > lb[i-1] or cl[i-1] < lb[i-1]):
                lb[i] = lb[i-1]
        if i > 0 and not np.isnan(ub[i-1]):
            if not (ub[i] < ub[i-1] or cl[i-1] > ub[i-1]):
                ub[i] = ub[i-1]
        if i == 0 or np.isnan(st[i-1]):
            direction[i] = 1  # bear default start
        elif st[i-1] == ub[i-1]:
            direction[i] = -1 if cl[i] > ub[i] else 1
        else:
            direction[i] = 1 if cl[i] < lb[i] else -1
        st[i] = lb[i] if direction[i] == -1 else ub[i]

    bear = (direction == 1).astype(int)
    return pd.Series(bear, index=close.index)


def _find_support(df: pd.DataFrame, lookback: int = 20) -> float:
    """Son N barda swing low destek seviyesini bul.

    Swing low: bir önceki ve sonraki bardan düşük olan Low noktası.
    Birden fazla varsa fiyata en yakın olanı seç.
    """
    if len(df) < lookback + 2:
        return float(df['Low'].iloc[-lookback:].min())

    lows = df['Low'].iloc[-(lookback + 2):]
    swing_lows = []

    for i in range(1, len(lows) - 1):
        if lows.iloc[i] <= lows.iloc[i - 1] and lows.iloc[i] <= lows.iloc[i + 1]:
            swing_lows.append(float(lows.iloc[i]))

    if not swing_lows:
        # Swing low bulunamazsa periyodun en düşüğünü al
        return float(lows.min())

    # Mevcut fiyata en yakın (ve altında olan) swing low
    current_price = float(df['Close'].iloc[-1])
    below = [s for s in swing_lows if s < current_price]
    if below:
        return max(below)  # fiyata en yakın destek
    return min(swing_lows)


class WalkForwardBacktester:
    """Walk-forward portföy backtester.

    Rebalance'lar arası günlük stop-loss + portföy DD limiti.
    """

    def __init__(self, all_data: dict, xu_df: pd.DataFrame):
        self.all_data = all_data
        self.xu_df = xu_df

        # Ortak tarih indeksi (tüm hisseler için)
        all_dates = set()
        for df in all_data.values():
            if df is not None and len(df) > 0:
                all_dates.update(df.index.tolist())
        self.trading_days = sorted(all_dates)

        # XU100 close serisi (benchmark)
        xu_close = xu_df['Close'] if 'Close' in xu_df.columns else xu_df['close']
        self.xu_close = xu_close

        # Sektör haritası (cache'li). Başarısızsa boş dict → sektör cap pas geçilir.
        self.sector_map = {}
        try:
            from alpha.config import SECTOR_CAP_ENABLED
            if SECTOR_CAP_ENABLED:
                from markets.bist.data import get_bist_sector_mapping
                self.sector_map = get_bist_sector_mapping() or {}
        except Exception as e:
            print(f"  [!] Sektör haritası yüklenemedi: {e}")

    def _get_rebalance_dates(self) -> list:
        """Rebalance tarihlerini üret."""
        days = self.trading_days
        if len(days) <= WF_TRAIN_DAYS:
            return []
        start = WF_TRAIN_DAYS
        dates = []
        i = start
        while i < len(days):
            dates.append((i, days[i]))
            i += WF_STEP_DAYS
        return dates

    def _get_price(self, ticker: str, date: pd.Timestamp, col: str = 'Close') -> float:
        """Hisse fiyatını al (belirli tarih)."""
        df = self.all_data.get(ticker)
        if df is None:
            return np.nan
        if date in df.index:
            return float(df.loc[date, col])
        # En yakın önceki tarihi bul
        mask = df.index <= date
        if mask.any():
            return float(df.loc[mask].iloc[-1][col])
        return np.nan

    def _get_open_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Next-open entry fiyatı."""
        return self._get_price(ticker, date, 'Open')

    def _get_atr(self, ticker: str, date_idx: int) -> float:
        """ATR değerini al."""
        df = self.all_data.get(ticker)
        if df is None or len(df) < 20:
            return np.nan
        atr = calc_atr(df.iloc[:date_idx + 1])
        if len(atr) == 0 or pd.isna(atr.iloc[-1]):
            return np.nan
        return float(atr.iloc[-1])

    def run(self) -> dict:
        """Walk-forward backtest — batch modu destekli.

        BATCH_MODE=True: portföy kapanmadan yenisi açılmaz.
        BATCH_MODE=False: klasik biweekly rebalance.
        """
        rebalance_schedule = self._get_rebalance_dates()
        if not rebalance_schedule:
            print("  [ALPHA] Yetersiz veri — rebalance yapılamaz")
            return self._empty_result()

        # ML pre-compute
        ml_features = {}
        ml_scorer = None
        if ML_STAGE1_ENABLED:
            try:
                from ml.scorer import MLScorer
                from ml.features import compute_all_features
                ml_scorer = MLScorer()
                ml_scorer._load_models()
                if ml_scorer.loaded:
                    print("  [ML] Feature'lar hesaplanıyor...")
                    import time as _time
                    _t0 = _time.time()
                    for ticker, df in self.all_data.items():
                        if df is None or len(df) < 80:
                            continue
                        try:
                            feats = compute_all_features(df, xu_df=self.xu_df)
                            if not feats.empty:
                                ml_features[ticker] = feats
                        except Exception:
                            continue
                    print(f"  [ML] {len(ml_features)} hisse feature hesaplandı ({_time.time()-_t0:.0f}s)")
                else:
                    ml_scorer = None
            except ImportError:
                ml_scorer = None

        cash = INITIAL_CAPITAL
        positions: dict[str, Position] = {}
        peak_equity = INITIAL_CAPITAL

        equity_curve = []
        trades = []
        rebalance_events = []
        stage_funnel = []

        from core.indicators import ema as _ema
        xu_ema = _ema(self.xu_close, REGIME_EMA_LENGTH)

        # ── KAMA weekly regime flag (close < kama = bear) ──
        xu_bear_flag = None
        if REGIME_KAMA_ENABLED:
            _w = self.xu_close.resample('W-FRI').last().dropna()
            _chg = (_w - _w.shift(KAMA_ER_PERIOD)).abs()
            _vol = _w.diff().abs().rolling(KAMA_ER_PERIOD).sum()
            _er = (_chg / _vol).replace([np.inf, -np.inf], 0).fillna(0)
            _fast_sc = 2.0 / (KAMA_FAST + 1)
            _slow_sc = 2.0 / (KAMA_SLOW + 1)
            _sc = (_er * (_fast_sc - _slow_sc) + _slow_sc) ** 2
            _k = _w.copy().astype(float)
            _k.iloc[:KAMA_ER_PERIOD] = np.nan
            _k.iloc[KAMA_ER_PERIOD] = _w.iloc[KAMA_ER_PERIOD]
            for _i in range(KAMA_ER_PERIOD + 1, len(_w)):
                _prev = _k.iloc[_i - 1]
                if pd.isna(_prev):
                    _k.iloc[_i] = _w.iloc[_i]
                else:
                    _k.iloc[_i] = _prev + _sc.iloc[_i] * (_w.iloc[_i] - _prev)
            _bear_w = (_w < _k).astype(int)
            xu_bear_flag = _bear_w.reindex(self.xu_close.index, method='ffill').fillna(0).astype(int)

        # ── Adaptive SuperTrend (K-means vol clustering) regime flag ──
        if REGIME_AST_ENABLED and xu_bear_flag is None:
            _h = self.xu_df['High'] if 'High' in self.xu_df.columns else self.xu_df['high']
            _l = self.xu_df['Low'] if 'Low' in self.xu_df.columns else self.xu_df['low']
            xu_bear_flag = _adaptive_supertrend_bear_flag(
                _h, _l, self.xu_close,
                atr_len=AST_ATR_LEN, factor=AST_FACTOR, training=AST_TRAINING,
                hi_init=AST_HI_INIT, mid_init=AST_MID_INIT, lo_init=AST_LO_INIT,
            )
            print(f"  [AST] bear days: {int(xu_bear_flag.sum())}/{len(xu_bear_flag)} ({xu_bear_flag.mean()*100:.0f}%)")

        xu_start = None
        for _, date in rebalance_schedule:
            if date in self.xu_close.index:
                xu_start = float(self.xu_close.loc[date])
                break
        if xu_start is None and len(self.xu_close) > WF_TRAIN_DAYS:
            xu_start = float(self.xu_close.iloc[WF_TRAIN_DAYS])

        first_rb_day_idx = rebalance_schedule[0][0]
        last_batch_date = None  # son batch açılış tarihi
        bull_streak = 0
        REGIME_REENTRY_CONFIRM_DAYS = 5  # bear→bull flip'i teyit gün sayısı

        for day_i in range(first_rb_day_idx, len(self.trading_days)):
            date = self.trading_days[day_i]

            # ── Günlük rejim tespiti ──
            curr_regime = 'bull'
            if xu_bear_flag is not None and date in xu_bear_flag.index:
                if int(xu_bear_flag.loc[date]) == 1:
                    curr_regime = 'bear'
            elif date in xu_ema.index and date in self.xu_close.index:
                _xu_now = float(self.xu_close.loc[date])
                _xu_ema_now = float(xu_ema.loc[date])
                if not np.isnan(_xu_ema_now) and _xu_now < _xu_ema_now:
                    curr_regime = 'bear'
            bull_streak = bull_streak + 1 if curr_regime == 'bull' else 0

            # ── Yeni batch açılacak mı? ──
            open_new_batch = False
            if BATCH_MODE:
                # Batch: pozisyon kalmadıysa ve minimum bekleme geçtiyse
                if len(positions) <= BATCH_REOPEN_THRESHOLD:
                    if last_batch_date is None:
                        open_new_batch = True
                    else:
                        days_since = sum(1 for d in self.trading_days if last_batch_date < d <= date)
                        if days_since >= WF_STEP_DAYS:
                            open_new_batch = True
            else:
                # Klasik: rebalance takvimindeyse
                open_new_batch = any(d == date for _, d in rebalance_schedule)

            # ── Rejim flip re-entry: N gün sürekli bull + cash'teysek hemen aç ──
            if (not open_new_batch and len(positions) == 0
                    and bull_streak == REGIME_REENTRY_CONFIRM_DAYS):
                open_new_batch = True

            if open_new_batch and day_i >= first_rb_day_idx:
                # Rejim filtresi
                regime_weight = REGIME_BULL_WEIGHT
                _is_bear = False
                if xu_bear_flag is not None and date in xu_bear_flag.index:
                    _is_bear = int(xu_bear_flag.loc[date]) == 1
                elif date in xu_ema.index and date in self.xu_close.index:
                    xu_now = float(self.xu_close.loc[date])
                    xu_ema_now = float(xu_ema.loc[date])
                    if not np.isnan(xu_ema_now) and xu_now < xu_ema_now:
                        _is_bear = True
                if _is_bear:
                    regime_weight = REGIME_BEAR_WEIGHT

                # Bear rejimde pozisyon açma + mevcutları kapat
                if regime_weight <= 0:
                    for t, pos in list(positions.items()):
                        exit_px = self._get_price(t, date) * (1 - SLIPPAGE_PCT)
                        if np.isnan(exit_px):
                            exit_px = pos.entry_price
                        pnl = (exit_px / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                        hold_days = (date - pos.entry_date).days
                        trades.append(TradeRecord(
                            ticker=t, entry_date=pos.entry_date, exit_date=date,
                            entry_price=pos.entry_price, exit_price=exit_px,
                            weight=pos.weight, pnl_pct=round(pnl, 2),
                            exit_reason='REGIME', hold_days=hold_days,
                        ))
                        cash += pos.shares * exit_px * (1 - COMMISSION_PCT)
                    positions.clear()
                    # Equity güncelle
                    equity = cash
                    if equity > peak_equity:
                        peak_equity = equity
                    xu_val = float(self.xu_close.loc[date]) if date in self.xu_close.index else np.nan
                    xu_norm = (xu_val / xu_start * INITIAL_CAPITAL) if (xu_start and not np.isnan(xu_val)) else np.nan
                    equity_curve.append({
                        'date': date, 'equity': round(equity, 2),
                        'benchmark': round(xu_norm, 2) if not np.isnan(xu_norm) else None,
                        'n_positions': 0,
                        'cash_pct': 100.0,
                        'dd_pct': round((equity / peak_equity - 1) * 100, 2),
                    })
                    continue

                # Veri kes (look-ahead bias yok)
                truncated = {}
                for t, df in self.all_data.items():
                    if df is not None:
                        sub = df.loc[df.index <= date]
                        if len(sub) >= MIN_DATA_DAYS:
                            truncated[t] = sub

                # ML veya Klasik aday seçimi
                if ml_scorer is not None and ml_scorer.loaded:
                    ml_candidates = []
                    for ticker, feats_df in ml_features.items():
                        if ticker not in truncated:
                            continue
                        feats_sub = feats_df.loc[feats_df.index <= date]
                        if len(feats_sub) < 5:
                            continue
                        row = feats_sub.iloc[-1]
                        vec = ml_scorer._make_feature_vector(row)
                        if vec is None:
                            continue
                        preds = ml_scorer._predict_all(vec)
                        ml_1g, ml_3g = preds['ml_a_1g'], preds['ml_a_3g']
                        if ml_1g is None or ml_1g < ML_SCORE_THRESHOLD:
                            continue
                        if ml_3g is not None and ml_3g < ML_SWING_THRESHOLD:
                            continue
                        # ML eğim
                        if len(feats_sub) > ML_SLOPE_LOOKBACK:
                            row_ago = feats_sub.iloc[-1 - ML_SLOPE_LOOKBACK]
                            vec_ago = ml_scorer._make_feature_vector(row_ago)
                            if vec_ago is not None:
                                p_ago = ml_scorer._predict_all(vec_ago)
                                if p_ago['ml_a_1g'] is not None and (ml_1g - p_ago['ml_a_1g']) < ML_SLOPE_MIN:
                                    continue
                        # Teknik onay
                        from alpha.stages import stage3_confirmation
                        confirmation = stage3_confirmation(truncated[ticker])
                        if confirmation['score'] < CONFIRMATION_MIN_SCORE:
                            continue
                        # Uzama filtresi
                        from alpha.config import MAX_RALLY_PCT, MAX_RALLY_DAYS, EMA50_DIST_MAX
                        if MAX_RALLY_PCT > 0:
                            _df = truncated[ticker]
                            if len(_df) >= MAX_RALLY_DAYS:
                                _rally = (_df['Close'].iloc[-1] / _df['Close'].iloc[-MAX_RALLY_DAYS] - 1) * 100
                                if _rally >= MAX_RALLY_PCT:
                                    continue
                        if EMA50_DIST_MAX > 0:
                            _df = truncated[ticker]
                            if len(_df) >= 50:
                                _ema50 = _df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                                if _ema50 > 0 and (_df['Close'].iloc[-1] / _ema50) > EMA50_DIST_MAX:
                                    continue
                        # Extension filter (stop-risk veto) — K=2/4 composite trigger
                        from alpha.config import (EXT_FILTER_ENABLED, EXT_FILTER_MIN_TRIGGERS,
                            EXT_20D_HIGH_PCT, EXT_EMA21_DIST_PCT, EXT_MFI_14, EXT_BB_PCTB)
                        if EXT_FILTER_ENABLED:
                            ext = 0
                            if 'close_vs_20d_high_pct' in row and pd.notna(row['close_vs_20d_high_pct']) and row['close_vs_20d_high_pct'] > EXT_20D_HIGH_PCT: ext += 1
                            if 'ema21_dist_pct' in row and pd.notna(row['ema21_dist_pct']) and row['ema21_dist_pct'] > EXT_EMA21_DIST_PCT: ext += 1
                            if 'mfi_14' in row and pd.notna(row['mfi_14']) and row['mfi_14'] > EXT_MFI_14: ext += 1
                            if 'bb_pctb' in row and pd.notna(row['bb_pctb']) and row['bb_pctb'] > EXT_BB_PCTB: ext += 1
                            if ext >= EXT_FILTER_MIN_TRIGGERS:
                                continue
                        ml_avg = ml_1g if ml_3g is None else (ml_1g + ml_3g) / 2
                        composite = ml_avg * 100 * ML_COMPOSITE_WEIGHT + confirmation['score'] * 10 * (1 - ML_COMPOSITE_WEIGHT)
                        ml_candidates.append({
                            'ticker': ticker, 'composite_score': round(min(100, composite), 1), 'passed': True,
                        })
                    ml_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
                    passed = ml_candidates
                    n_total = n_passed = len(passed)
                else:
                    candidates = scan_universe(truncated)
                    passed = [c for c in candidates if c.get('passed')]
                    n_total = len(candidates)
                    n_passed = len(passed)

                from alpha.config import MIN_STOCKS
                if n_passed >= MIN_STOCKS:
                    # Portföy oluştur
                    if PORTFOLIO_METHOD == "equal" and len(passed) >= 3:
                        top_n = passed[:min(TARGET_PORTFOLIO_SIZE, MAX_STOCKS)]
                        w = 1.0 / len(top_n)
                        target_weights = {c['ticker']: w for c in top_n}
                        sharpe = 0.0
                    elif PORTFOLIO_METHOD == "score" and len(passed) >= 3:
                        from alpha.config import (
                            WEIGHT_MIN, WEIGHT_MAX, SCORE_POWER,
                            VOL_ADJUSTED_SIZING, VOL_ADJUST_FLOOR_PCT,
                            CORR_FILTER_ENABLED, CORR_MAX_PAIR, CORR_LOOKBACK_DAYS,
                            SECTOR_CAP_ENABLED, SECTOR_MAX_WEIGHT, SECTOR_MAX_NAMES,
                            STRESS_CORR_ENABLED, STRESS_CORR_MAX_PAIR,
                            STRESS_CORR_XU_THRESHOLD, STRESS_CORR_MIN_DAYS,
                        )
                        _limit = min(TARGET_PORTFOLIO_SIZE, MAX_STOCKS)
                        _sec_enabled = SECTOR_CAP_ENABLED and bool(self.sector_map)
                        _sec_counts = {}
                        # Stress-corr: XU100 red-day mask (bu rebalance'a özel, lookback penceresi)
                        _xu_red_dates = None
                        if STRESS_CORR_ENABLED:
                            try:
                                _xu_slice = self.xu_close.loc[:date].tail(CORR_LOOKBACK_DAYS + 1)
                                _xu_ret = _xu_slice.pct_change().dropna()
                                _red = _xu_ret[_xu_ret < STRESS_CORR_XU_THRESHOLD].index
                                if len(_red) >= STRESS_CORR_MIN_DAYS:
                                    _xu_red_dates = _red
                            except Exception:
                                _xu_red_dates = None
                        _stress_active = _xu_red_dates is not None
                        _red_ret_cache = {}
                        if CORR_FILTER_ENABLED or _sec_enabled or _stress_active:
                            top_n = []
                            _ret_cache = {}
                            for _c in passed:
                                if len(top_n) >= _limit:
                                    break
                                _tck = _c['ticker']
                                if _sec_enabled:
                                    _sec = self.sector_map.get(_tck, 'Unknown')
                                    if _sec_counts.get(_sec, 0) >= SECTOR_MAX_NAMES:
                                        continue
                                _df = truncated.get(_tck)
                                _ok = True
                                if (CORR_FILTER_ENABLED or _stress_active):
                                    if _df is None or len(_df) < CORR_LOOKBACK_DAYS:
                                        top_n.append(_c)
                                        if _sec_enabled:
                                            _sec_counts[_sec] = _sec_counts.get(_sec, 0) + 1
                                        continue
                                    if _tck not in _ret_cache:
                                        _ret_cache[_tck] = _df['Close'].pct_change().tail(CORR_LOOKBACK_DAYS)
                                    _r_t = _ret_cache[_tck]
                                    if _stress_active and _tck not in _red_ret_cache:
                                        _red_ret_cache[_tck] = _r_t.reindex(_xu_red_dates)
                                    for _p in top_n:
                                        _pt = _p['ticker']
                                        if _pt not in _ret_cache:
                                            _dfp = truncated.get(_pt)
                                            if _dfp is None:
                                                continue
                                            _ret_cache[_pt] = _dfp['Close'].pct_change().tail(CORR_LOOKBACK_DAYS)
                                        if CORR_FILTER_ENABLED:
                                            _corr = _r_t.corr(_ret_cache[_pt])
                                            if not pd.isna(_corr) and abs(_corr) > CORR_MAX_PAIR:
                                                _ok = False
                                                break
                                        if _stress_active:
                                            if _pt not in _red_ret_cache:
                                                _red_ret_cache[_pt] = _ret_cache[_pt].reindex(_xu_red_dates)
                                            _sc = _red_ret_cache[_tck].corr(_red_ret_cache[_pt])
                                            if not pd.isna(_sc) and abs(_sc) > STRESS_CORR_MAX_PAIR:
                                                _ok = False
                                                break
                                    if not _ok:
                                        continue
                                top_n.append(_c)
                                if _sec_enabled:
                                    _sec_counts[_sec] = _sec_counts.get(_sec, 0) + 1
                        else:
                            top_n = passed[:_limit]
                        scores = np.array([c['composite_score'] for c in top_n], dtype=float)
                        scores = np.clip(scores, 1.0, None)
                        raw = scores ** SCORE_POWER
                        if VOL_ADJUSTED_SIZING:
                            atr_pcts = []
                            for c in top_n:
                                _df = truncated[c['ticker']]
                                _atr_s = calc_atr(_df)
                                _close = float(_df['Close'].iloc[-1]) if len(_df) else 0.0
                                if len(_atr_s) and not pd.isna(_atr_s.iloc[-1]) and _close > 0:
                                    atr_pcts.append(float(_atr_s.iloc[-1]) / _close * 100)
                                else:
                                    atr_pcts.append(3.0)
                            atr_pcts = np.clip(np.array(atr_pcts, dtype=float), VOL_ADJUST_FLOOR_PCT, None)
                            raw = raw / atr_pcts
                        raw = raw / raw.sum()
                        # Cap'le, sonra re-normalize (iteratif clamp)
                        for _ in range(5):
                            raw = np.clip(raw, WEIGHT_MIN, WEIGHT_MAX)
                            s = raw.sum()
                            if abs(s - 1.0) < 1e-6:
                                break
                            raw = raw / s
                        # Sektör ağırlık cap: cap'i aşan sektörleri oransal küçült.
                        if _sec_enabled and SECTOR_MAX_WEIGHT < 1.0:
                            _sec_ids = [self.sector_map.get(c['ticker'], 'Unknown') for c in top_n]
                            for _ in range(8):
                                _tot = {}
                                for s_, w_ in zip(_sec_ids, raw):
                                    _tot[s_] = _tot.get(s_, 0.0) + float(w_)
                                _over = {s_: t_ for s_, t_ in _tot.items() if t_ > SECTOR_MAX_WEIGHT + 1e-9}
                                if not _over:
                                    break
                                for i, s_ in enumerate(_sec_ids):
                                    if s_ in _over:
                                        raw[i] *= SECTOR_MAX_WEIGHT / _over[s_]
                                # WEIGHT_MIN floor'ı koru, sonra re-normalize
                                raw = np.clip(raw, WEIGHT_MIN, WEIGHT_MAX)
                                raw = raw / raw.sum()
                        target_weights = {c['ticker']: float(w) for c, w in zip(top_n, raw)}
                        sharpe = 0.0
                    else:
                        portfolio = build_portfolio(truncated, passed, as_of_idx=-1)
                        target_weights = portfolio.get('weights', {})
                        sharpe = portfolio.get('sharpe_ratio', 0.0)

                    if regime_weight < 1.0:
                        target_weights = {t: w * regime_weight for t, w in target_weights.items()}

                    # Equity hesapla
                    equity = cash
                    for t, pos in positions.items():
                        px = self._get_price(t, date)
                        if not np.isnan(px):
                            equity += pos.shares * px

                    # Yeni pozisyonlar aç (mevcut pozisyonlara dokunma)
                    old_tickers = list(positions.keys())
                    from alpha.config import EXECUTION_MODE, GAP_SKIP_PCT, LIMIT_ENTRY_PCT
                    for t, w in target_weights.items():
                        if t in positions:
                            continue
                        signal_close = self._get_price(t, date)
                        if np.isnan(signal_close) or signal_close <= 0:
                            continue
                        actual_entry_date = date
                        if EXECUTION_MODE == "close":
                            entry_price = signal_close * (1 + SLIPPAGE_PCT)
                        else:
                            df_t_exec = self.all_data.get(t)
                            if df_t_exec is None:
                                continue
                            nxt_idx = df_t_exec.index.searchsorted(date, side='right')
                            if nxt_idx >= len(df_t_exec):
                                continue
                            nxt_date = df_t_exec.index[nxt_idx]
                            nxt_open = float(df_t_exec.loc[nxt_date, 'Open'])
                            nxt_low = float(df_t_exec.loc[nxt_date, 'Low'])
                            if np.isnan(nxt_open) or nxt_open <= 0:
                                continue
                            if EXECUTION_MODE == "next_open":
                                gap = (nxt_open / signal_close) - 1
                                if gap >= GAP_SKIP_PCT:
                                    continue
                                entry_price = nxt_open * (1 + SLIPPAGE_PCT)
                            elif EXECUTION_MODE == "limit":
                                limit_px = signal_close * (1 + LIMIT_ENTRY_PCT)
                                if nxt_open >= limit_px:
                                    continue
                                if nxt_low > limit_px:
                                    continue
                                entry_price = limit_px * (1 + SLIPPAGE_PCT)
                            else:
                                entry_price = signal_close * (1 + SLIPPAGE_PCT)
                            actual_entry_date = nxt_date
                        if np.isnan(entry_price) or entry_price <= 0:
                            continue
                        # ATR stop
                        atr_val = entry_price * 0.05
                        df_t = self.all_data.get(t)
                        if POSITION_STOP_ATR_MULT > 0 and df_t is not None:
                            sub_t = df_t.loc[df_t.index <= date]
                            if len(sub_t) >= 20:
                                _atr = calc_atr(sub_t)
                                atr_val = float(_atr.iloc[-1]) if not pd.isna(_atr.iloc[-1]) else entry_price * 0.05
                            # Dinamik stop: yüksek ATR% → daha sıkı çarpan
                            from alpha.config import (
                                DYNAMIC_STOP_ENABLED, DYNAMIC_STOP_ATR_PCT_HI, DYNAMIC_STOP_MULT_HI,
                            )
                            mult = POSITION_STOP_ATR_MULT
                            if DYNAMIC_STOP_ENABLED and entry_price > 0:
                                atr_pct = atr_val / entry_price * 100
                                if atr_pct > DYNAMIC_STOP_ATR_PCT_HI:
                                    mult = DYNAMIC_STOP_MULT_HI
                            stop = entry_price - mult * atr_val
                        elif POSITION_STOP_PCT > 0:
                            stop = entry_price * (1 - POSITION_STOP_PCT)
                            atr_val = entry_price * POSITION_STOP_PCT
                        else:
                            stop = entry_price * 0.94

                        target_value = equity * w
                        total_cost = target_value * (1 + COMMISSION_PCT)
                        if total_cost > cash:
                            continue
                        positions[t] = Position(
                            ticker=t, entry_date=actual_entry_date, entry_price=entry_price,
                            weight=w, shares=target_value / entry_price,
                            stop_price=stop, atr_at_entry=atr_val,
                            stop_mult=mult if POSITION_STOP_ATR_MULT > 0 else POSITION_STOP_ATR_MULT,
                            highest_since_entry=entry_price,
                        )
                        cash -= total_cost

                    n_final = len(target_weights)
                    stage_funnel.append((date, n_total, n_passed, n_final))
                    rebalance_events.append(RebalanceEvent(
                        date=date, old_tickers=old_tickers,
                        new_tickers=list(target_weights.keys()),
                        candidates_total=n_total, candidates_passed=n_passed,
                        final_stocks=n_final, turnover=1.0, sharpe=round(sharpe, 3),
                    ))
                    last_batch_date = date

            # ── Günlük mark-to-market + çok aşamalı çıkış ──
            # NOT: equity sonradan hesaplanıyor (cash exit'lerle güncelleniyor).
            positions_value = 0.0
            stopped_out = []

            for t, pos in positions.items():
                low = self._get_price(t, date, 'Low')
                high = self._get_price(t, date, 'High')
                close_px = self._get_price(t, date)

                if np.isnan(close_px):
                    positions_value += pos.shares * pos.entry_price
                    continue

                # Güncel ATR hesapla (dinamik trailing için)
                cur_atr = pos.atr_at_entry
                df_t = self.all_data.get(t)
                if df_t is not None:
                    sub_t = df_t.loc[df_t.index <= date]
                    if len(sub_t) >= 20:
                        _atr_s = calc_atr(sub_t)
                        if not pd.isna(_atr_s.iloc[-1]):
                            cur_atr = float(_atr_s.iloc[-1])

                # Highest güncelle
                if not np.isnan(high) and high > pos.highest_since_entry:
                    pos.highest_since_entry = high

                hold_days = (date - pos.entry_date).days

                # ── 1) Emergency stop: Entry - mult×ATR → tüm pozisyon sat ──
                # Breakeven shift: +N·R kâra ulaşıldıysa stop entry'ye çekilir
                from alpha.config import BE_SHIFT_ENABLED, BE_SHIFT_R
                if BE_SHIFT_ENABLED and not pos.be_shifted and pos.atr_at_entry > 0:
                    R_dist = pos.stop_mult * pos.atr_at_entry
                    be_trigger = pos.entry_price + BE_SHIFT_R * R_dist
                    if not np.isnan(high) and high >= be_trigger:
                        pos.be_shifted = True
                emergency_stop = pos.entry_price - pos.stop_mult * pos.atr_at_entry
                if pos.be_shifted:
                    emergency_stop = max(emergency_stop, pos.entry_price)
                if not np.isnan(low) and low <= emergency_stop:
                    exit_price = emergency_stop * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                    trades.append(TradeRecord(
                        ticker=t, entry_date=pos.entry_date, exit_date=date,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        weight=pos.weight, pnl_pct=round(pnl, 2),
                        exit_reason='STOP', hold_days=hold_days,
                    ))
                    cash += pos.shares * exit_price * (1 - COMMISSION_PCT)
                    stopped_out.append(t)
                    continue

                # ── 2) Max hold kontrolü ──
                if hold_days >= MAX_HOLD_DAYS:
                    exit_price = close_px * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                    trades.append(TradeRecord(
                        ticker=t, entry_date=pos.entry_date, exit_date=date,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        weight=pos.weight, pnl_pct=round(pnl, 2),
                        exit_reason='MAX_HOLD', hold_days=hold_days,
                    ))
                    cash += pos.shares * exit_price * (1 - COMMISSION_PCT)
                    stopped_out.append(t)
                    continue

                # ── 3) Trailing mod aktivasyonu: Fiyat >= Entry + 1.5×güncel ATR ──
                trailing_target = pos.entry_price + TRAILING_TRIGGER_ATR * cur_atr
                if not pos.trailing_active and close_px >= trailing_target:
                    pos.trailing_active = True

                # ── 4) İlk kâr al (%50): Trailing modda, kapanış < önceki günün Low'u ──
                if pos.trailing_active and not pos.first_tp_done and pos.prev_day_low > 0:
                    if close_px < pos.prev_day_low:
                        # %50 sat
                        sell_shares = pos.shares * 0.5
                        exit_price = close_px * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                        trades.append(TradeRecord(
                            ticker=t, entry_date=pos.entry_date, exit_date=date,
                            entry_price=pos.entry_price, exit_price=exit_price,
                            weight=pos.weight * 0.5, pnl_pct=round(pnl, 2),
                            exit_reason='TP_50', hold_days=hold_days,
                        ))
                        cash += sell_shares * exit_price * (1 - COMMISSION_PCT)
                        pos.shares -= sell_shares
                        pos.first_tp_done = True
                        # Kalan %50 için trailing stop'u güncelle
                        pos.stop_price = pos.highest_since_entry - TRAILING_ATR_MULT * cur_atr

                # ── 5) Kalan %50 trailing stop: zirve - 1.5×ATR ──
                if pos.first_tp_done:
                    trail = pos.highest_since_entry - TRAILING_ATR_MULT * cur_atr
                    if trail > pos.stop_price:
                        pos.stop_price = trail
                    if not np.isnan(low) and low <= pos.stop_price:
                        exit_price = pos.stop_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                        trades.append(TradeRecord(
                            ticker=t, entry_date=pos.entry_date, exit_date=date,
                            entry_price=pos.entry_price, exit_price=exit_price,
                            weight=pos.weight * 0.5, pnl_pct=round(pnl, 2),
                            exit_reason='TRAIL', hold_days=hold_days,
                        ))
                        cash += pos.shares * exit_price * (1 - COMMISSION_PCT)
                        stopped_out.append(t)
                        continue

                # Prev day low güncelle (bugünün low'u yarın için)
                if not np.isnan(low):
                    pos.prev_day_low = low

                positions_value += pos.shares * close_px

            for t in stopped_out:
                del positions[t]

            # Equity = güncellenmiş cash + kalan pozisyonların değeri
            equity = cash + positions_value

            # Equity tracking
            if equity > peak_equity:
                peak_equity = equity
            dd_pct = (equity / peak_equity - 1) * 100

            # Portfolio DD limit — tüm pozisyonları kapat
            if PORTFOLIO_DD_LIMIT > -99.0 and dd_pct <= PORTFOLIO_DD_LIMIT and positions:
                for t, pos in list(positions.items()):
                    exit_px = self._get_price(t, date) * (1 - SLIPPAGE_PCT)
                    if np.isnan(exit_px):
                        exit_px = pos.entry_price
                    pnl = (exit_px / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                    hold_days = (date - pos.entry_date).days
                    trades.append(TradeRecord(
                        ticker=t, entry_date=pos.entry_date, exit_date=date,
                        entry_price=pos.entry_price, exit_price=exit_px,
                        weight=pos.weight, pnl_pct=round(pnl, 2),
                        exit_reason='DD_LIMIT', hold_days=hold_days,
                    ))
                    cash += pos.shares * exit_px * (1 - COMMISSION_PCT)
                positions.clear()
                equity = cash

            # Daily loss limit — tek gün kaybı bu eşikten büyükse tüm pozisyonları kapat
            from alpha.config import DAILY_LOSS_LIMIT
            if DAILY_LOSS_LIMIT > -99.0 and positions and len(equity_curve) > 0:
                _prev_eq = equity_curve[-1]['equity']
                if _prev_eq > 0:
                    _daily_chg = (equity / _prev_eq - 1) * 100
                    if _daily_chg <= DAILY_LOSS_LIMIT:
                        for t, pos in list(positions.items()):
                            exit_px = self._get_price(t, date) * (1 - SLIPPAGE_PCT)
                            if np.isnan(exit_px):
                                exit_px = pos.entry_price
                            pnl = (exit_px / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                            hold_days = (date - pos.entry_date).days
                            trades.append(TradeRecord(
                                ticker=t, entry_date=pos.entry_date, exit_date=date,
                                entry_price=pos.entry_price, exit_price=exit_px,
                                weight=pos.weight, pnl_pct=round(pnl, 2),
                                exit_reason='DAILY_LOSS', hold_days=hold_days,
                            ))
                            cash += pos.shares * exit_px * (1 - COMMISSION_PCT)
                        positions.clear()
                        equity = cash

            xu_val = float(self.xu_close.loc[date]) if date in self.xu_close.index else np.nan
            xu_norm = (xu_val / xu_start * INITIAL_CAPITAL) if (xu_start and not np.isnan(xu_val)) else np.nan

            equity_curve.append({
                'date': date, 'equity': round(equity, 2),
                'benchmark': round(xu_norm, 2) if not np.isnan(xu_norm) else None,
                'n_positions': len(positions),
                'cash_pct': round(cash / max(equity, 1) * 100, 1),
                'dd_pct': round(dd_pct, 2),
            })

        # Kalan pozisyonları kapat
        if positions:
            last_date = self.trading_days[-1]
            for t, pos in positions.items():
                exit_price = self._get_price(t, last_date) * (1 - SLIPPAGE_PCT)
                if np.isnan(exit_price):
                    exit_price = pos.entry_price
                pnl = (exit_price / pos.entry_price - 1) * 100 - COMMISSION_PCT * 200
                trades.append(TradeRecord(
                    ticker=t, entry_date=pos.entry_date, exit_date=last_date,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    weight=pos.weight, pnl_pct=round(pnl, 2),
                    exit_reason='END', hold_days=(last_date - pos.entry_date).days,
                ))

        return {
            'equity_curve': equity_curve, 'trades': trades,
            'rebalance_events': rebalance_events, 'stage_funnel': stage_funnel,
            'initial_capital': INITIAL_CAPITAL,
            'final_equity': equity_curve[-1]['equity'] if equity_curve else INITIAL_CAPITAL,
            'benchmark_final': equity_curve[-1]['benchmark'] if equity_curve else INITIAL_CAPITAL,
        }

    def _empty_result(self):
        return {
            'equity_curve': [],
            'trades': [],
            'rebalance_events': [],
            'stage_funnel': [],
            'initial_capital': INITIAL_CAPITAL,
            'final_equity': INITIAL_CAPITAL,
            'benchmark_final': INITIAL_CAPITAL,
        }
