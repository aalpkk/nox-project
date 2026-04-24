"""
CTE (Compression-to-Expansion) config.

Hedef: compression yapısından (horizontal base veya falling channel) gelen
ilk anlamlı yukarı breakout'u erken yakala; breakout'un runner'a dönüp
dönmediğini label ile ölç.

Sade: single-head LGBM, target=runner_15 default. quality head YOK (ignito'dan
farklı). İlk run noise-free için trigger parametreleri sıkı.

Leakage:
  - Structure hesapları [t-W, t-1] penceresinden; breakout barı asla içeri alınmaz.
  - US-close türevleri (VIX/DXY/SPY) `.shift(1)` zorunlu — context feature'lar
    features.py içinde `_align_us_macro_to_bist` pattern'i kullanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Structure / compression detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CompressionParams:
    """Structure geometry parametreleri — hem horizontal base hem falling channel
    için ortak window ve ATR referansları."""
    structure_window: int = 20          # primary lookback: [t-W, t-1]
    min_bars_active: int = 15           # en az bu kadar bar geçerli veri şart
    atr_window: int = 14                # width/distance normalizasyonu için ATR
    bb_window: int = 20                 # bollinger width percentile
    bb_pctile_lookback: int = 60        # percentile ranking horizon
    touch_tolerance_atr: float = 0.25   # upper/lower tesmaslara ATR cinsinden tolerans
    close_density_window: int = 20      # close std / ATR → tightness


@dataclass(frozen=True)
class HorizontalBaseParams:
    """Horizontal base geometry filter — yapısal validity."""
    max_width_atr: float = 3.5          # kutu genişliği ATR cinsinden üst sınır
    max_abs_slope_atr_per_bar: float = 0.08  # trend slope aşırı pozitif/negatif olmasın
    min_touches_upper: int = 2
    min_touches_lower: int = 2
    max_close_std_atr: float = 1.0      # close dispersiyonu darsın


@dataclass(frozen=True)
class FallingChannelParams:
    """Falling channel — slope + band convergence + residual stability.

    False-positive önleme:
      - min_bars_in_structure: çok kısa düşüşler fc sayılmasın
      - min_lower_high_count: sadece slope değil, lower-high yapı gereksin
      - max_width_cv: width çok değişkense yapı bozuk demektir
      - max_abs_upper_slope / min_neg_upper_slope: upper line AŞAĞI eğimli olmalı
    """
    min_bars_in_structure: int = 15
    min_lower_high_count: int = 2
    max_upper_slope_atr_per_bar: float = -0.005  # upper aşağı eğimli (küçük negatif bile kabul)
    max_width_atr: float = 4.0
    max_width_cv: float = 0.55          # coefficient of variation of (high-low)
    min_convergence_ratio: float = 0.0  # (lower_slope - upper_slope): >0 = convergence


@dataclass(frozen=True)
class DryupParams:
    """Volume dry-up ve breakout expansion metrikleri."""
    dryup_windows_short: tuple[int, ...] = (3, 5)
    dryup_windows_long: tuple[int, ...] = (20, 30)
    quiet_bar_window: int = 5
    quiet_bar_rel_vol_thresh: float = 0.70   # vol < thresh * avg_vol_structure → quiet
    structure_vol_ref_window: int = 20       # breakout_vol karşılaştırma tabanı


@dataclass(frozen=True)
class FirstBreakParams:
    """İlk çıkış diskriminatoru — kaç kere zaten kırıldı, ne kadar süredir içeride."""
    lookback_window: int = 12               # son K barda üst sınır üstü close sayımı
    failed_break_window: int = 3            # breakout sonrası bu kadar bar içinde geri dönüş
    failed_break_atr_frac: float = 0.5      # boundary - failback_atr_frac * ATR altına kapanış = failed
    max_prior_attempts: int = 1             # trigger'da HARD filter (soft sinyal model'de)


@dataclass(frozen=True)
class BreakoutBarParams:
    """Breakout bar kalitesi — trigger'ın giriş barı.

    v2 relax_m test (2026-04-24): ret 5→4%, rvol 1.8→1.5, loc 0.70→0.65,
    body 0.40→0.35. Density +49% (N 1951→2907), runner_15 base rate 23.6→25.1%
    (↑), AMA model lift@10 HB 1.14x→0.75x (−34%), FC 1.79x→1.20x (−33%).
    Rolled back: eklenen satırlar compression heuristic'e okunabilir ama
    LGBM için noise, ranker edge'ini yiyor. Sweep ve eval çıktıları:
    output/cte_trigger_sweep.csv, memory/cte_trigger_relax_finding.md.
    """
    min_return_1d: float = 0.05             # günlük getiri ≥ %5 (ilk faz sıkı)
    min_rvol: float = 1.8                   # volume/sma(vol,20) ≥ 1.8
    min_close_loc_bar: float = 0.70         # (close-low)/(high-low) ≥ 0.70
    min_body_pct_range: float = 0.40        # |close-open|/(high-low) ≥ 0.40


# ═══════════════════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LabelParams:
    """Iki ailesi: early-validation (hold/failed_break) + runner/expansion.

    breakout_level ikili üretilir:
      - _struct: max(hb_upper, fc_upper_at_t) — yapı ihlali semantiği
      - _close:  close_0                       — trade PnL semantiği

    Runner label spike-reject'li: t+h'de close, t+1..t+h peak'in altında
    aşırı çökmüşse (close/peak < spike_reject_ratio) runner sayılmaz.
    """
    early_horizons: tuple[int, ...] = (3, 5)
    runner_horizons: tuple[int, ...] = (10, 15, 20)
    primary_target: str = "runner_15"

    failback_atr_frac: float = 0.5           # failed_break eşiği
    runner_mfe_atr: float = 3.0              # MFE ≥ 3R şart
    runner_max_mae_atr: float = 1.5          # MAE ≤ 1.5R şart
    runner_min_hold_h: int = 5               # hold_5_close zorunlu
    spike_reject_close_peak_ratio: float = 0.70  # close_h / peak_h ≥ 0.70 aksi halde spike reddi

    atr_window: int = 14                     # MFE/MAE normalize için ATR


# ═══════════════════════════════════════════════════════════════════════════════
# Data + split
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataParams:
    yf_cache_path: str = "output/ohlcv_10y_fintables_master.parquet"
    xu100_symbol: str = "XU100.IS"
    min_bars_per_ticker: int = 60        # structure_window=20 → ≥60 bar şart


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class SplitParams:
    """ignito ile aynı 3-fold walk-forward (aynı veri pencere'si, karşılaştırma kolaylığı)."""
    train_start: str = "2020-04-20"
    folds: tuple[FoldSpec, ...] = (
        FoldSpec(name="fold1",
                 train_end="2023-04-30",
                 val_start="2023-05-15", val_end="2023-10-31",
                 test_start="2023-11-15", test_end="2024-04-30"),
        FoldSpec(name="fold2",
                 train_end="2024-04-30",
                 val_start="2024-05-15", val_end="2024-10-31",
                 test_start="2024-11-15", test_end="2025-04-30"),
        FoldSpec(name="fold3",
                 train_end="2025-04-30",
                 val_start="2025-05-15", val_end="2025-10-31",
                 test_start="2025-11-15", test_end="2026-04-30"),
    )
    label_horizon_bars: int = 20             # longest runner horizon


# ═══════════════════════════════════════════════════════════════════════════════
# Production line defaults (HB pipeline / FC pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LineParams:
    """Per-line training mode.

    mode options:
      - "pure":  train ONLY on rows where setup_type == line_name (hb or fc).
                 Scoring pool is still all trigger_{line} == True rows, so
                 'both' rows receive a score at inference even though they
                 weren't in training.
      - "mixed": train on all trigger_{line} == True rows (includes 'both').

    v2b ablation showed FC pure > FC mixed (1.89x vs 1.35x lift@10) while
    HB mixed > HB pure (1.14x vs 0.91x). Defaults reflect that.

    FC-local LGBM overrides: cte_fc_v2 sweep (2026-04-24) found min_child_samples
    20→30 gives lift@10 1.79x→~2.0x seed-averaged (rho +0.07→+0.16 avg),
    PF_top30 0.72→0.91, robust across 4 seeds, no per-fold regression.
    Applied only in FC line; HB keeps train.py LGBMParams default.
    """
    hb_mode_default: str = "mixed"     # HB geometry is wider; "both" rows regularize
    fc_mode_default: str = "pure"      # FC geometry is narrower; "both" rows dilute
    fc_lgbm_min_child_samples: int = 30  # FC-specific; HB stays at LGBMParams default (20)


@dataclass(frozen=True)
class ShortlistParams:
    """Per-line daily shortlist selection."""
    top_k_hb: int = 3
    top_k_fc: int = 3
    min_score_model: float = 0.0        # 0.0 = no floor; tune in v1b
    drop_failed_break_5_close: bool = False  # optional hygiene filter


@dataclass(frozen=True)
class PortfolioMergeParams:
    """Optional merge step — runs AFTER per-line shortlisting."""
    merge_mode: str = "fixed_quota"      # "fixed_quota" | "alternating" | "normalized_rank"
    max_total_positions: int = 4         # cap across lines per date
    per_line_cap: int = 2                # max from a single line per date
    dedup_on_both: bool = True           # setup_type == 'both' → keep best-ranked appearance


@dataclass(frozen=True)
class Config:
    compression: CompressionParams = field(default_factory=CompressionParams)
    hb: HorizontalBaseParams = field(default_factory=HorizontalBaseParams)
    fc: FallingChannelParams = field(default_factory=FallingChannelParams)
    dryup: DryupParams = field(default_factory=DryupParams)
    firstness: FirstBreakParams = field(default_factory=FirstBreakParams)
    bar: BreakoutBarParams = field(default_factory=BreakoutBarParams)
    label: LabelParams = field(default_factory=LabelParams)
    data: DataParams = field(default_factory=DataParams)
    split: SplitParams = field(default_factory=SplitParams)
    line: LineParams = field(default_factory=LineParams)
    shortlist: ShortlistParams = field(default_factory=ShortlistParams)
    portfolio: PortfolioMergeParams = field(default_factory=PortfolioMergeParams)


CONFIG = Config()
