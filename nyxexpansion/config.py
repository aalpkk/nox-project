"""
nyxexpansion v1 config.

Trigger A (v1): clean breakout — close > prior 20-day high AND rvol≥1.5 AND close_loc≥0.70
Trigger B (rafta): near-tavan / pseudo-ignition — ayrı experiment, v1'de karışmaz.

LEAKAGE KURALI:
  - US-close kaynakları (VIX/DXY/SPY/USDTRY/crypto) ZORUNLU .shift(1).
    Same-day reindex = 4-saatlik look-ahead (BIST 18:00, US 22:00 TRT).
  - XU100 ve sektör endeksleri BIST aynı seans → shift YOK.
  - Detaylar: memory/macro_timing_leakage.md
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TriggerAParams:
    """Clean breakout trigger."""
    lookback_high: int = 20       # prior N-gün high
    rvol_min: float = 1.5         # today vol / sma(vol, rvol_win)
    rvol_win: int = 20
    close_loc_min: float = 0.70   # (close - low) / (high - low) bar içi


@dataclass(frozen=True)
class LabelParams:
    """MFE/MAE + continuation label parametreleri."""
    horizons: tuple[int, ...] = (3, 5, 7, 10)
    primary_h: int = 10                          # L3 cont train target
    cont_mfe_atr_mult: float = 2.0               # MFE ≥ 2R
    cont_mae_atr_mult: float = 1.0               # MAE ≤ 1R
    mfe_mae_ratio_clip: float = 5.0              # winsorized version için
    mfe_mae_mae_floor_atr_frac: float = 0.3      # ratio paydası floor'u: max(MAE, 0.3 × ATR/close)
    atr_window: int = 14
    # L2 follow_through_3 (eval-only)
    ft3_min_close_gain: float = 0.02             # +%2
    ft3_max_drawdown: float = 0.015              # low[1..3] ≥ close[0] * (1 - 0.015)
    # L4 expansion_score
    expansion_lambda: float = 1.5                # MFE/ATR - λ×MAE/ATR
    # P2 structural risk
    struct_swing_lookback: int = 10              # son N bar low → swing low
    struct_trigger_buffer_atr: float = 0.3       # trigger_level − 0.3×ATR


@dataclass(frozen=True)
class DataParams:
    yf_cache_path: str = "output/ohlcv_10y_fintables_master.parquet"   # 10y × 589 Fintables-verified equity
    yf_cache_path_fallback: str = "output/ohlcv_6y.parquet"             # legacy 6y (yfinance seed, 399 ticker)
    xu100_symbol: str = "XU100.IS"
    min_bars_per_ticker: int = 80
    min_data_end_date: str | None = None    # e.g. "2025-10-01"; None → no filter


@dataclass(frozen=True)
class FoldSpec:
    """Önden belirlenmiş (donmuş) train/val/test tarihleri.

    Signal_date'e uygulanır. Label horizon 10 bar → train_end ↔ val_start
    arası 10+ işgünü embargo zorunlu. Aşağıdaki tarihler 15 gün boşluk bırakır.
    """
    name: str
    train_end: str      # inclusive
    val_start: str      # inclusive, train_end + ≥15 takvim günü (10 işgünü + güvenlik)
    val_end: str        # inclusive
    test_start: str     # inclusive, val_end + ≥15 takvim günü
    test_end: str       # inclusive


@dataclass(frozen=True)
class SplitParams:
    """3-fold expanding walk-forward — DATES FROZEN, hyperparam tuning
    kesinlikle split'i değiştirmez.

    Universe: ml_dataset.parquet'teki ticker listesi (aynı evren). Veri
    ohlcv_6y.parquet içinde: 2020-04-20 → 2026-04-20.

    Embargo: label horizon 10 bar → train'deki son sinyalin label pencerleri
    (t+1..t+10) val başlangıcından önce kapanır. train_end → val_start arası
    ≥15 takvim günü (10 işgünü + weekend buffer) bıraktığımız için güvenli.

    train_start sabit: 2020-04-20 (expanding window).
    """
    train_start: str = "2020-04-20"
    folds: tuple[FoldSpec, ...] = (
        FoldSpec(
            name="fold1",
            train_end="2023-04-30",
            val_start="2023-05-15", val_end="2023-10-31",
            test_start="2023-11-15", test_end="2024-04-30",
        ),
        FoldSpec(
            name="fold2",
            train_end="2024-04-30",
            val_start="2024-05-15", val_end="2024-10-31",
            test_start="2024-11-15", test_end="2025-04-30",
        ),
        FoldSpec(
            name="fold3",
            train_end="2025-04-30",
            val_start="2025-05-15", val_end="2025-10-31",
            test_start="2025-11-15", test_end="2026-04-30",
        ),
    )
    # Embargo signal_date'e uygulanmaz doğrudan; yukarıdaki val_start/test_start
    # boşluğu zaten embargo'yu sağlar. Ama label hesabı sırasında signal_date+10
    # satırları farklı split'e düşerse split boundary'i net tutmak için
    # filter_label_horizon_days>0 ile güvenlik eklenebilir.
    label_horizon_bars: int = 10


@dataclass(frozen=True)
class Config:
    trigger: TriggerAParams = field(default_factory=TriggerAParams)
    label: LabelParams = field(default_factory=LabelParams)
    data: DataParams = field(default_factory=DataParams)
    split: SplitParams = field(default_factory=SplitParams)


CONFIG = Config()


# ── Leakage-safe macro helper ─────────────────────────────────────────────────
# US-close kaynaklarının listesi. Bu symbol ailesinden makro okuyorsan,
# features.py içinde _align_us_macro_to_bist() kullan (shift(1) yapar).

US_CLOSE_MACRO_SYMBOLS = frozenset({
    'VIX', '^VIX',
    'DXY', 'DX-Y.NYB',
    'SPY', 'QQQ', '^DJI', '^IXIC', '^GSPC',
    'USDTRY', 'USDTRY=X',
    'BTC-USD', 'ETH-USD',
})

BIST_SESSION_SYMBOLS = frozenset({
    'XU100', 'XU100.IS', 'XU030.IS', 'XU050.IS', 'XU100D.IS',
    # sektör endeksleri — hepsi BIST seansı
    'XBANK.IS', 'XUSIN.IS', 'XTCRT.IS', 'XELKT.IS', 'XINSA.IS',
    'XGIDA.IS', 'XKMYA.IS', 'XULAS.IS', 'XHOLD.IS', 'XTRZM.IS',
    'XMADN.IS', 'XILTM.IS', 'XGMYO.IS', 'XTEKS.IS',
})
