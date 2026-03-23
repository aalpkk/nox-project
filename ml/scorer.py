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

# 4-model dosya adları
_MODEL_SLOTS = {
    'universe_1g': 'universe_up_1g.txt',
    'universe_3g': 'universe_up_3g.txt',
    'reranker_1g': 'reranker_up_1g.txt',
    'reranker_3g': 'reranker_up_3g.txt',
}


def _get_screener_derived_columns():
    """Model'in eğitildiği screener_derived feature kolon sırası (46 feature)."""
    return list(_SCREENER_DERIVED_COLS)


def is_ml_scoring_enabled():
    """ML_SCORING_ENABLED env var kontrolü."""
    return os.getenv('ML_SCORING_ENABLED', '').lower() in ('1', 'true', 'yes')


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
