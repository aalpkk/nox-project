"""
ML Scorer — Production ML Skorlama Modülü (v2 — Dual Horizon)
===============================================================
Shortlist sinyallerini ML modelleriyle skorlar.

4-Model Mimari:
  - universe_up_1g.txt  → Universe kısa vade (1 gün)
  - universe_up_3g.txt  → Universe swing (3 gün)
  - reranker_up_1g.txt  → Reranker kısa vade
  - reranker_up_3g.txt  → Reranker swing

Model D mantığı (TVN conditional blend):
  - TVN sinyali → 0.65 * reranker + 0.35 * universe (her iki horizon)
  - Diğer sinyaller → pure universe (Model A)

Feature flag: ML_SCORING_ENABLED env var (default: False)

Kullanım:
    from ml.scorer import MLScorer
    scorer = MLScorer()
    if scorer.loaded:
        scores = scorer.score_tickers_dual(tickers, price_data, xu_df)
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

# Model dizini: git'te track edilen ml/models/ veya output/ml_meta/
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
_FALLBACK_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'output', 'ml_meta')

# screener_derived feature grubu — modelin eğitildiği kesin sıralama
# run_ml_train.py FEATURE_GROUPS['screener_derived'] + ml/features.py çıktısından
_SCREENER_DERIVED_COLS = [
    'consecutive_green', 'q_rvol_s', 'q_clv_s', 'q_wick_s', 'q_range_s', 'q_total',
    'br_rsi_thrust', 'br_rsi_gradual', 'br_ad_proxy', 'br_ema_reclaim', 'br_score',
    'rg_slope_score', 'rg_di_score', 'rg_ema_above', 'rg_adx_rebound', 'rg_was_trending',
    'rg_score', 'gate_open', 'sell_severity', 'pivot_delta_pct',
    'rt_ema_bull', 'rt_st_bull', 'rt_wk_trend_up',
    'rt_cmf_pos', 'rt_rvol_high', 'rt_obv_slope_pos',
    'rt_adx_slope_pos', 'rt_atr_expanding', 'rt_di_bull',
    'trend_score', 'regime_score', 'entry_score', 'oe_score',
    'pb_ema_dist', 'pb_rsi_low', 'exit_stage', 'days_in_trade',
    'is_tavan', 'tavan_streak', 'close_to_high', 'tavan_locked',
    'hit_tavan_intraday', 'recent_tavan_10d',
    'rs_10', 'rs_60', 'rs_composite',
]

# 4-model dosya adları (shortlist)
_MODEL_SLOTS = {
    'universe_1g': 'universe_up_1g.txt',
    'universe_3g': 'universe_up_3g.txt',
    'reranker_1g': 'reranker_up_1g.txt',
    'reranker_3g': 'reranker_up_3g.txt',
}

# Breakout model dosya adları
_BREAKOUT_MODEL_SLOTS = {
    'tavan_3d': 'breakout_tavan_3d.txt',
    'tavan_1d': 'breakout_tavan_1d.txt',
    'rally_3d': 'breakout_rally_3d.txt',
    'rally_5d': 'breakout_rally_5d.txt',
}

# Breakout model dizini alternatifleri (sırayla denenir)
_BREAKOUT_ALT_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'ml_breakout_v3'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'ml_breakout_v2'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'ml_breakout'),
]

# Breakout modelin kullandığı feature'lardan çıkartılacaklar (leakage, v3: 17 feature)
_BREAKOUT_LEAKAGE_FEATURES = {
    # Orijinal 4 — tavan durumu
    'is_tavan', 'tavan_streak', 'tavan_locked', 'hit_tavan_intraday',
    # +13 yeni — bugünkü fiyat/hareket sızdıran feature'lar
    'returns_1d', 'close_position', 'gap_pct',
    'close_to_high', 'recent_tavan_10d',
    'consecutive_green', 'consecutive_higher_close',
    'daily_move_atr',
    'near_tavan_miss', 'recent_near_tavan_5d',
    'max_daily_ret_5d',
    'vol_surge_today', 'vol_pattern_score',
}


def _get_screener_derived_columns():
    """Model'in eğitildiği screener_derived feature kolon sırası (46 feature)."""
    return list(_SCREENER_DERIVED_COLS)


def is_ml_scoring_enabled():
    """ML_SCORING_ENABLED env var kontrolü."""
    return os.getenv('ML_SCORING_ENABLED', '').lower() in ('1', 'true', 'yes')


def is_breakout_ml_enabled():
    """BREAKOUT_ML_ENABLED env var kontrolü."""
    return os.getenv('BREAKOUT_ML_ENABLED', '').lower() in ('1', 'true', 'yes')


# ═══════════════════════════════════════════
# Composite Scoring Helpers
# ═══════════════════════════════════════════

def calc_ml_rerank_bonus(ml_score):
    """ML skoru bazlı rerank bonusu.

    Returns: int — composite score'a eklenecek bonus
    """
    if ml_score is None:
        return 0
    if ml_score >= 0.60:
        return 3
    elif ml_score >= 0.55:
        return 2
    elif ml_score >= 0.50:
        return 1
    return 0


def calc_ml_rerank_bonus_v2(ml_swing):
    """W-score (swing/3g) bazlı 3-zone rerank bonusu.

    Returns: int — composite score'a eklenecek bonus
    """
    if ml_swing is None:
        return 0
    if ml_swing >= 0.60:
        return 4
    elif ml_swing >= 0.55:
        return 3
    elif ml_swing >= 0.45:
        return 0   # NEUTRAL zone
    return -1       # WEAK zone


ML_EFFECT_ICONS = {'up': 'ML↑', 'down': 'ML↓', 'neutral': 'ML='}


def calc_ml_effect_label(ml_swing, rerank_bonus):
    """ML'nin sinyale etkisini belirle.

    Returns: str — 'up', 'down', or 'neutral'
    """
    if ml_swing is None:
        return 'neutral'
    if rerank_bonus >= 2:
        return 'up'
    elif rerank_bonus <= -1:
        return 'down'
    return 'neutral'


def calc_source_quality_bonus(sbt_bucket):
    """SBT bucket bazlı kaynak kalite bonusu.

    Args:
        sbt_bucket: str — 'A+', 'A', 'B', 'C', 'X', or None

    Returns: int — composite score'a eklenecek bonus
    """
    if not sbt_bucket:
        return 0
    mapping = {'A+': 2, 'A': 1, 'B': 0, 'C': 0, 'X': -2}
    return mapping.get(sbt_bucket, 0)


def calc_overlap_bonus(source_count):
    """Kaynak sayısı bazlı çakışma bonusu.

    Args:
        source_count: int — kaç farklı listede mevcut

    Returns: int — composite score'a eklenecek bonus
    """
    if source_count >= 4:
        return 3
    elif source_count == 3:
        return 2
    elif source_count == 2:
        return 1
    return 0


def ml_badge_dual(ml_short, ml_swing):
    """Dual ML badge formatla.

    Args:
        ml_short: float or None — 1g ML skoru
        ml_swing: float or None — 3g ML skoru

    Returns: str — örn. '🤖S58🔵·W61🟡'
    """
    if ml_short is None and ml_swing is None:
        return ''

    parts = []
    if ml_short is not None:
        s100 = int(ml_short * 100)
        icon = '🟡' if ml_short >= 0.60 else ('🔵' if ml_short >= 0.40 else '🔴')
        parts.append(f'S{s100}{icon}')
    if ml_swing is not None:
        w100 = int(ml_swing * 100)
        icon = '🟡' if ml_swing >= 0.60 else ('🔵' if ml_swing >= 0.40 else '🔴')
        parts.append(f'W{w100}{icon}')

    return '🤖' + '·'.join(parts) if parts else ''


class MLScorer:
    """ML model yükleyici + skorlayıcı (v2 — 4-model dual horizon).

    Lazy-load: modeller ilk score çağrısında yüklenir.
    Backward compat: score_tickers() ve ml_badge() korunur.
    """

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or _DEFAULT_MODEL_DIR
        if not os.path.isdir(self.model_dir):
            self.model_dir = _FALLBACK_MODEL_DIR

        # 4-model slots
        self._universe_1g = None
        self._universe_3g = None
        self._reranker_1g = None
        self._reranker_3g = None
        self._feat_cols = None
        self.loaded = False
        self._load_attempted = False

        # Backward compat aliases
        self._booster_a = None  # = _universe_1g
        self._booster_b = None  # = _reranker_1g

    def _load_models(self):
        """4 modeli yükle (lazy, tek seferlik)."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            import lightgbm as lgb
        except ImportError:
            print("  [ML] lightgbm yüklü değil — ML scoring devre dışı")
            return

        loaded_count = 0
        for slot, filename in _MODEL_SLOTS.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                # 1g modelleri zorunlu, 3g opsiyonel
                if slot in ('universe_1g',):
                    print(f"  [ML] Zorunlu model bulunamadı: {path}")
                    return
                continue
            try:
                booster = lgb.Booster(model_file=path)
                setattr(self, f'_{slot}', booster)
                loaded_count += 1
            except Exception as e:
                print(f"  [ML] Model yükleme hatası ({filename}): {e}")
                if slot == 'universe_1g':
                    return

        if self._universe_1g is None:
            print(f"  [ML] Universe 1g model yüklenemedi")
            return

        self._feat_cols = _get_screener_derived_columns()
        # Feature sayısı doğrulama
        if self._universe_1g.num_feature() != len(self._feat_cols):
            print(f"  [ML] Feature sayısı uyuşmazlığı: model={self._universe_1g.num_feature()}, "
                  f"beklenen={len(self._feat_cols)}")
            return

        self.loaded = True
        # Backward compat aliases
        self._booster_a = self._universe_1g
        self._booster_b = self._reranker_1g
        print(f"  [ML] {loaded_count} model yüklendi")

    def _predict_all(self, vec):
        """Feature vektörü için 4 modelin tahminlerini hesapla.

        Returns: dict with keys ml_a_1g, ml_a_3g, ml_b_1g, ml_b_3g (None if unavailable)
        """
        x = vec.reshape(1, -1)
        result = {
            'ml_a_1g': float(self._universe_1g.predict(x)[0]) if self._universe_1g else None,
            'ml_a_3g': float(self._universe_3g.predict(x)[0]) if self._universe_3g else None,
            'ml_b_1g': float(self._reranker_1g.predict(x)[0]) if self._reranker_1g else None,
            'ml_b_3g': float(self._reranker_3g.predict(x)[0]) if self._reranker_3g else None,
        }
        return result

    def score_tickers_dual(self, tickers, price_data, xu_df):
        """Her ticker için dual-horizon ML skorları hesapla.

        Args:
            tickers: list of ticker strings
            price_data: {ticker: DataFrame (OHLCV, Uppercase)}
            xu_df: XU100 DataFrame

        Returns:
            {ticker: {
                'ml_score_short': float,  # up_1g (kısa vade) — universe
                'ml_score_swing': float,  # up_3g (swing) — universe
                'ml_a_1g': float,         # universe 1g raw
                'ml_a_3g': float,         # universe 3g raw
                'ml_b_1g': float,         # reranker 1g raw
                'ml_b_3g': float,         # reranker 3g raw
                'ml_score': float,        # backward compat (= ml_score_short)
            }}
        """
        self._load_models()
        if not self.loaded:
            return {}

        from ml.features import compute_all_features

        results = {}
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or len(df) < 80:
                continue

            try:
                feats = compute_all_features(df, xu_df=xu_df)
                if feats.empty:
                    continue

                row = feats.iloc[-1]
                vec = self._make_feature_vector(row)
                if vec is None:
                    continue

                preds = self._predict_all(vec)
                # ml_score_short/swing = universe raw (Model D uygulanmadan)
                results[ticker] = {
                    'ml_score_short': preds['ml_a_1g'],
                    'ml_score_swing': preds['ml_a_3g'],
                    'ml_a_1g': preds['ml_a_1g'],
                    'ml_a_3g': preds['ml_a_3g'],
                    'ml_b_1g': preds['ml_b_1g'],
                    'ml_b_3g': preds['ml_b_3g'],
                    # Backward compat
                    'ml_score': preds['ml_a_1g'],
                    'ml_a': preds['ml_a_1g'],
                    'ml_b': preds['ml_b_1g'],
                }
            except Exception:
                continue

        return results

    def score_tickers(self, tickers, price_data, xu_df):
        """Her ticker için ML skoru hesapla (backward compat — uses 1g universe).

        Returns:
            {ticker: {'ml_score': float, 'ml_a': float, 'ml_b': float}}
        """
        self._load_models()
        if not self.loaded:
            return {}

        from ml.features import compute_all_features

        results = {}
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or len(df) < 80:
                continue

            try:
                feats = compute_all_features(df, xu_df=xu_df)
                if feats.empty:
                    continue

                row = feats.iloc[-1]
                vec = self._make_feature_vector(row)
                if vec is None:
                    continue

                pred_a = float(self._universe_1g.predict(vec.reshape(1, -1))[0])
                pred_b = None
                if self._reranker_1g is not None:
                    pred_b = float(self._reranker_1g.predict(vec.reshape(1, -1))[0])

                results[ticker] = {
                    'ml_score': pred_a,
                    'ml_a': pred_a,
                    'ml_b': pred_b,
                }
            except Exception:
                continue

        return results

    def score_signal(self, ticker, screener, features_row):
        """Tek sinyal skorla (feature row hazırsa).

        Model D mantığı:
          TVN → 0.65B + 0.35A
          Else → pure A

        Args:
            ticker: str
            screener: str — 'tavan', 'alsat', 'nw', 'rt', etc.
            features_row: pd.Series veya dict (feature values)

        Returns:
            float — ML skoru (0-1 arası probability)
        """
        self._load_models()
        if not self.loaded:
            return None

        if isinstance(features_row, dict):
            features_row = pd.Series(features_row)

        vec = self._make_feature_vector(features_row)
        if vec is None:
            return None

        pred_a = float(self._universe_1g.predict(vec.reshape(1, -1))[0])

        # Model D: TVN → conditional ensemble
        is_tvn = screener in ('tavan', 'tavan_kandidat')
        if is_tvn and self._reranker_1g is not None:
            pred_b = float(self._reranker_1g.predict(vec.reshape(1, -1))[0])
            return 0.65 * pred_b + 0.35 * pred_a

        return pred_a

    def score_signal_batch(self, signals_with_features):
        """Çoklu sinyal skorla.

        Args:
            signals_with_features: list of (ticker, screener, features_row)

        Returns:
            list of float (ML skorları, None for failures)
        """
        self._load_models()
        if not self.loaded:
            return [None] * len(signals_with_features)

        results = []
        for ticker, screener, features_row in signals_with_features:
            score = self.score_signal(ticker, screener, features_row)
            results.append(score)
        return results

    def _make_feature_vector(self, row):
        """Feature row'dan model'in beklediği sırada vektör oluştur.

        Args:
            row: pd.Series (feature values)

        Returns:
            np.array veya None
        """
        if self._feat_cols is None:
            return None

        vec = np.full(len(self._feat_cols), np.nan)
        for i, col in enumerate(self._feat_cols):
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    vec[i] = float(val)

        # Çok fazla NaN varsa atla
        valid_pct = np.sum(~np.isnan(vec)) / len(vec)
        if valid_pct < 0.5:
            return None

        return vec

    def ml_badge(self, ml_score):
        """ML skoru için badge string oluştur (backward compat — tek skor).

        Returns: str — örn. '🤖62🟢'
        """
        if ml_score is None:
            return ''

        score_100 = int(ml_score * 100)
        if ml_score >= 0.60:
            return f'🤖{score_100}🟡'
        elif ml_score >= 0.40:
            return f'🤖{score_100}🔵'
        else:
            return f'🤖{score_100}🔴'

    # ═══════════════════════════════════════
    # BREAKOUT SCORING
    # ═══════════════════════════════════════

    def _load_breakout_models(self):
        """Breakout modellerini yükle (lazy, tek seferlik)."""
        if hasattr(self, '_breakout_load_attempted') and self._breakout_load_attempted:
            return
        self._breakout_load_attempted = True
        self._breakout_models = {}
        self._breakout_feat_cols = None
        self.breakout_loaded = False

        try:
            import lightgbm as lgb
        except ImportError:
            return

        # Arama sırası: ml/models/ → output/ml_breakout_v2/ → output/ml_breakout/
        search_dirs = [self.model_dir] + _BREAKOUT_ALT_DIRS

        loaded = 0
        model_dir_used = None
        for slot, filename in _BREAKOUT_MODEL_SLOTS.items():
            path = None
            for d in search_dirs:
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path is None:
                continue
            try:
                booster = lgb.Booster(model_file=path)
                self._breakout_models[slot] = booster
                loaded += 1
                if model_dir_used is None:
                    model_dir_used = os.path.dirname(path)
            except Exception as e:
                print(f"  [BRK] Model yükleme hatası ({filename}): {e}")

        if loaded == 0:
            return

        # Feature kolon listesi yükle (model yanında veya alt dizinlerde)
        import json
        for d in ([model_dir_used] if model_dir_used else []) + search_dirs:
            feat_path = os.path.join(d, 'feature_columns.json')
            if os.path.exists(feat_path):
                try:
                    with open(feat_path) as f:
                        self._breakout_feat_cols = json.load(f)
                    print(f"  [BRK] Feature listesi: {len(self._breakout_feat_cols)} kolon ({feat_path})")
                    break
                except Exception:
                    pass

        self.breakout_loaded = True
        print(f"  [BRK] {loaded} breakout model yüklendi ({model_dir_used})")

    def score_breakout(self, tickers, price_data, xu_df, threshold=0.10):
        """Tüm BIST evreni için breakout olasılığı hesapla.

        Fusion skor: 0.40 × master_pctile + 0.60 × ml_s_pctile
        ML S = universe_1g (kısa dönem getiri modeli) — zamanlama ekler.
        Backtest: Top 5 → %44 ≥%5 getiri, %31 tavan, medyan +%2.8

        Args:
            tickers: list of ticker strings
            price_data: {ticker: DataFrame (OHLCV, Uppercase)}
            xu_df: XU100 DataFrame
            threshold: alert eşiği (fusion_score pctile >= threshold)

        Returns:
            {ticker: {
                'tavan_prob': float,    # tavan composite
                'rally_prob': float,    # rally composite
                'breakout_master': float,  # 0.5*tavan + 0.5*rally (raw)
                'ml_s_score': float,    # kısa dönem ML skoru
                'fusion_score': float,  # 0.40*master_pctile + 0.60*ml_s_pctile
                'tier': str,            # 'top5' | 'top10' | None
                'alert': bool,
            }}
        """
        self._load_breakout_models()
        if not self.breakout_loaded:
            return {}

        # ML S modeli de yükle (shortlist universe_1g)
        self._load_models()

        from ml.features import compute_all_features

        # Faz 1: tüm ticker'ları skorla (raw skorlar)
        raw_results = {}
        already_moved = 0
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or len(df) < 80:
                continue

            try:
                # Zaten hareket etmiş hisseleri ele — son 3 günde herhangi gün ≥%7
                close_col = 'Close' if 'Close' in df.columns else 'close'
                if close_col in df.columns and len(df) >= 4:
                    recent_rets = df[close_col].pct_change().iloc[-3:] * 100
                    if recent_rets.max() >= 7.0:
                        already_moved += 1
                        continue

                feats = compute_all_features(df, xu_df=xu_df)
                if feats.empty:
                    continue

                row = feats.iloc[-1]
                # Breakout feature vector — sabit eğitim sırasıyla
                brk_cols = self._breakout_feat_cols
                if brk_cols is None:
                    # Fallback: dinamik kolon listesi (leakage+close hariç)
                    brk_cols = [c for c in feats.columns
                               if c not in _BREAKOUT_LEAKAGE_FEATURES
                               and c != 'close']
                vec = np.full(len(brk_cols), np.nan)
                for i, col in enumerate(brk_cols):
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val):
                            vec[i] = float(val)

                valid_pct = np.sum(~np.isnan(vec)) / len(vec)
                if valid_pct < 0.5:
                    continue

                x = vec.reshape(1, -1)

                # Breakout modelleri
                preds = {}
                for slot, booster in self._breakout_models.items():
                    if booster.num_feature() == len(brk_cols):
                        preds[slot] = float(booster.predict(x)[0])

                # Composite skorlar
                tavan_3d = preds.get('tavan_3d')
                tavan_1d = preds.get('tavan_1d')
                rally_3d = preds.get('rally_3d')
                rally_5d = preds.get('rally_5d')

                tavan_comp = None
                if tavan_3d is not None and tavan_1d is not None:
                    tavan_comp = 0.6 * tavan_3d + 0.4 * tavan_1d
                elif tavan_3d is not None:
                    tavan_comp = tavan_3d

                rally_comp = None
                if rally_3d is not None and rally_5d is not None:
                    rally_comp = 0.6 * rally_3d + 0.4 * rally_5d
                elif rally_3d is not None:
                    rally_comp = rally_3d

                master = None
                if tavan_comp is not None and rally_comp is not None:
                    master = 0.5 * tavan_comp + 0.5 * rally_comp
                elif tavan_comp is not None:
                    master = tavan_comp
                elif rally_comp is not None:
                    master = rally_comp

                if master is None:
                    continue

                # ML S skoru (universe_1g — kısa dönem getiri)
                ml_s = None
                if self.loaded and self._universe_1g is not None:
                    s_cols = _get_screener_derived_columns()
                    s_vec = np.full(len(s_cols), np.nan)
                    for i, col in enumerate(s_cols):
                        if col in row.index:
                            val = row[col]
                            if pd.notna(val):
                                s_vec[i] = float(val)
                    s_valid = np.sum(~np.isnan(s_vec)) / len(s_vec)
                    if s_valid >= 0.5:
                        ml_s = float(self._universe_1g.predict(
                            s_vec.reshape(1, -1))[0])

                raw_results[ticker] = {
                    'tavan_prob': tavan_comp,
                    'rally_prob': rally_comp,
                    'breakout_master': master,
                    'ml_s_score': ml_s,
                    'tavan_3d': tavan_3d,
                    'tavan_1d': tavan_1d,
                    'rally_3d': rally_3d,
                    'rally_5d': rally_5d,
                }
            except Exception:
                continue

        if not raw_results:
            return {}

        if already_moved > 0:
            print(f"  [BRK] {already_moved} hisse elendi (son 3G ≥%7 hareket)")

        # Faz 2: percentile hesapla + fusion skor
        masters = {t: d['breakout_master'] for t, d in raw_results.items()}
        ml_s_scores = {t: d['ml_s_score'] for t, d in raw_results.items()
                       if d['ml_s_score'] is not None}

        # Percentile rank (0-1)
        sorted_masters = sorted(masters.values())
        sorted_ml_s = sorted(ml_s_scores.values()) if ml_s_scores else []
        n_m = len(sorted_masters)
        n_s = len(sorted_ml_s)

        def _pctile(val, sorted_vals, n):
            if n == 0:
                return 0.5
            rank = sum(1 for v in sorted_vals if v <= val)
            return rank / n

        results = {}
        for ticker, data in raw_results.items():
            m_pct = _pctile(data['breakout_master'], sorted_masters, n_m)
            s_pct = _pctile(data['ml_s_score'], sorted_ml_s, n_s) \
                if data['ml_s_score'] is not None else 0.5

            # Fusion: 0.40 × master + 0.60 × ML S
            fusion = 0.40 * m_pct + 0.60 * s_pct

            data['fusion_score'] = fusion
            data['master_pctile'] = m_pct
            data['ml_s_pctile'] = s_pct
            data['tier'] = None
            data['alert'] = False
            results[ticker] = data

        # Faz 3: Top 5 / Top 10 tier ataması
        sorted_by_fusion = sorted(results.items(),
                                   key=lambda x: x[1]['fusion_score'],
                                   reverse=True)
        for i, (ticker, data) in enumerate(sorted_by_fusion):
            if i < 5:
                data['tier'] = 'top5'
                data['alert'] = True
            elif i < 10:
                data['tier'] = 'top10'
                data['alert'] = True

        return results


def breakout_badge(breakout_data):
    """Breakout tier için badge string.

    Args:
        breakout_data: dict with 'tier', 'fusion_score' keys
                       OR float (backward compat — breakout_master raw value)

    Returns: str — örn. 'BRK🎯T5' veya 'BRK⚡T10'
    """
    if breakout_data is None:
        return ''

    # Backward compat: eski kod float gönderebilir
    if isinstance(breakout_data, (int, float)):
        pct = int(breakout_data * 100)
        if breakout_data >= 0.20:
            return f'BRK🎯{pct}%'
        elif breakout_data >= 0.10:
            return f'BRK⚡{pct}%'
        return ''

    tier = breakout_data.get('tier')
    fusion = breakout_data.get('fusion_score', 0)
    f_pct = int(fusion * 100)
    if tier == 'top5':
        return f'BRK🎯T5·F{f_pct}'
    elif tier == 'top10':
        return f'BRK⚡T10·F{f_pct}'
    return ''
