"""
nyxmomentum config — frozen dataclasses, modular by stage.

Problem type: cross-sectional leadership / portfolio construction for BIST.
NOT event-driven daily breakout prediction. Rebalance-granularity only.

LEAKAGE CONTRACT
  - Rebalance date cutoff is INCLUSIVE. All features use data up to and
    including rebalance_date close. Execution is assumed NEXT-OPEN of the
    following session (see BacktestConfig.execution).
  - US-close macro sources (VIX/DXY/SPY/USDTRY/crypto) MUST be .shift(1)
    before reindexing to BIST calendar. Same-day reindex = 4h look-ahead
    (BIST close 18:00 TRT vs US close 22:00+ TRT).
  - XU100 and BIST sector indices trade on the same session → no shift.
  - See memory/macro_timing_leakage.md for full detail.

SURVIVORSHIP NOTE
  - yfinance-backed BIST history lacks reliable delisted coverage. Results
    must be reported as potentially up-biased. Record the universe roster
    at each rebalance to make the bias visible.

TRADABILITY
  - Liquidity filter is mandatory. Pure 12-1 momentum without liquidity gating
    is not a valid result (spec §3.3, §15).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Rebalance calendar ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RebalanceConfig:
    """How rebalance dates are chosen from the trading calendar."""
    frequency: str = "M"               # "M" monthly, "W" weekly
    anchor: str = "last_trading_day"   # "last_trading_day" | "first_trading_day"
    start_date: str = "2021-01-01"     # first rebalance candidate
    end_date: str | None = None        # None → use latest trading day available
    min_rebalances: int = 12           # abort if fewer than this


# ── Universe filter ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UniverseConfig:
    """
    Per-rebalance tradability gate. Filters must not be so aggressive that the
    universe collapses, but must eliminate stocks you cannot realistically buy.
    """
    min_history_days: int = 252              # ≥1y daily bars
    min_price_tl: float = 2.0                # nominal price floor
    min_tl_volume_20d: float = 5_000_000.0   # 5M TL average daily turnover
    min_turnover_days_available: int = 200   # non-NaN volume days in lookback
    exclude_recent_ipo_days: int = 60        # first_trade_date within N days → drop
    max_missing_ratio: float = 0.05          # NaN close ratio in trailing 252d

    # Limit-move / tavan-taban frequency proxy. Uses abs(daily_ret) > threshold
    # as a surrogate for hitting limit (historical BIST: ±10%; post-2020 wider).
    # Reject if frequency over trailing window exceeds cap.
    limit_move_threshold: float = 0.095      # |daily_ret| above this → limit day
    limit_move_lookback: int = 60
    max_limit_move_freq: float = 0.15        # max 15% of last 60d

    # Diagnostic only; never blocks membership on its own
    zero_return_day_threshold: float = 0.30  # fraction of zero-return days in lookback


# ── Labels ────────────────────────────────────────────────────────────────────
#
# DECISION (V1, locked):
#   train_target = "l2_excess_vs_universe_median" — SINGLE target
#   primary_benchmark = "universe_median" — cross-sectional, no size tilt
#   diagnostic_benchmark = "xu100" — external reporting only, never training
#
# Everything else (L1 raw, L4 quality-adjusted, L5 drawdown-aware binary,
# forward_max_dd, xu100_excess) is DIAGNOSTIC only. These columns are written
# to the label panel for analysis but never fed to a learner.
#
# Rationale (spec alignment):
#   - L4 (quality-adjusted) target engineering risk: λ dominates outcome, 1mo DD
#     is noisy, model learns anti-vol/anti-gap confound instead of momentum edge.
#     Makes ablation muddy (is it alpha or just low-vol tilt?).
#   - L5 (drawdown-aware binary) is a future-DD object — putting it in selection
#     leaks the future. OK as evaluation diagnostic.
#   - Risk path is handled SEPARATELY via ex-ante execution proxies + risk
#     overlay at portfolio construction time, not inside the training target.

@dataclass(frozen=True)
class LabelConfig:
    """Cross-sectional label family. V1 locks L2 as the sole train target."""

    # ── Train target (single) ─────────────────────────────────────────────
    train_target: str = "l2_excess_vs_universe_median"

    # ── Horizon ───────────────────────────────────────────────────────────
    # Measured in rebalance periods (monthly → 1 month). Entry is next-open of
    # the trading day after rebalance_date; exit is close of the next
    # rebalance_date. Last rebalance has NaN labels by construction.
    primary_horizon: int = 1
    research_horizons: tuple[int, ...] = (2, 3)  # diagnostic only

    entry_mode: str = "next_open"       # "next_open" | "rebalance_close"
    exit_mode: str = "rebalance_close"  # "rebalance_close" only in v1

    # ── Benchmark policy ──────────────────────────────────────────────────
    primary_benchmark: str = "universe_median"      # for L2 (train target)
    diagnostic_benchmark: str = "xu100"             # for xu100_excess column

    # Universe scope for universe_median: computed over tickers flagged
    # eligible at the rebalance_date. Non-finite forward returns dropped
    # before median.
    universe_median_scope: str = "eligible_only"    # "eligible_only" | "all"

    # ── L3 outperform binary (diagnostic) ─────────────────────────────────
    outperform_mode: str = "vs_universe_median"     # "vs_universe_median" | "vs_xu100" | "top_quintile"
    top_quintile_threshold: float = 0.20

    # ── L4 quality-adjusted (diagnostic) ──────────────────────────────────
    quality_lambda: float = 0.5                     # ret − λ · max_dd
    quality_mode: str = "dd_penalty"                # "dd_penalty" | "vol_normalized"

    # ── L5 drawdown-aware binary (diagnostic) ─────────────────────────────
    l5_excess_threshold: float = 0.02               # excess > 2%
    l5_max_dd_threshold: float = 0.10               # intra-period DD < 10%


# Columns produced by the label module, tagged by role. The training pipeline
# consumes only entries where role == "train_target". Everything else is
# written for evaluation and audit but never fed to a learner.
LABEL_COLUMN_ROLES: dict[str, str] = {
    "l1_forward_return":              "diagnostic",
    "l2_excess_vs_universe_median":   "train_target",
    "l3_outperform_binary":           "diagnostic",
    "l4_quality_adjusted_return":     "diagnostic",
    # L5 IS A DIAGNOSTIC, NOT A SELECTION RULE. It reads future drawdown.
    # Using it to filter selected names would leak the future. Reporting only.
    "l5_drawdown_aware_binary":       "diagnostic",
    "forward_max_dd":                 "diagnostic",  # close-only
    "forward_max_dd_intraperiod":     "diagnostic",  # High/Low-based, tighter
    "xu100_excess_return":            "diagnostic",
    "universe_median_return":         "context",     # per-date reference value
    "entry_date":                     "context",
    "exit_date":                      "context",
    "holding_days":                   "context",
}


# ── Execution proxy (Step 0.5) ────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionProxyConfig:
    """
    DAILY-OHLCV-DERIVED EXECUTION PROXIES — NOT intraday truth.

    These are ex-ante approximations of opening-auction friction, computed
    from trailing daily bars ONLY. They do NOT measure true open liquidity,
    true fill rate, or true spread. If Matriks MCP intraday is enabled in
    V2, these can be replaced with real measurements.

    Honest naming: columns carry a 'proxy_' prefix or '_est' suffix, and the
    run report states the approximation explicitly.

    Realized fillability (t+1 gap / t+1 limit-open / t+1 traded flag) is
    computed as a SEPARATE diagnostic and must never be fed back into the
    selection/overlay feature set — doing so would leak the future.
    """
    short_lookback: int = 20
    long_lookback: int = 60
    atr_window: int = 14

    # limit-open proxy threshold; aligned with UniverseConfig.limit_move_threshold
    limit_threshold: float = 0.095

    # compute t+1 realized diagnostics (separate column, excluded from overlay)
    emit_realized_diagnostics: bool = True

    # realized limit-open threshold (looser than ex-ante; diagnostic only)
    realized_gap_flag_threshold: float = 0.05   # |t+1 open - t close| / t close


# Column role manifest for the execution module — keeps selection/diagnostic
# separation explicit and machine-readable.
EXECUTION_COLUMN_ROLES: dict[str, str] = {
    # Ex-ante proxies (KNOWN at rebalance_date close → may enter overlay)
    "proxy_open_dislocation_20d":    "ex_ante",
    "proxy_stale_open_freq_60d":     "ex_ante",
    "proxy_limit_open_freq_60d":     "ex_ante",
    "proxy_daily_range_ratio_20d":   "ex_ante",
    "proxy_gap_dispersion_20d":      "ex_ante",
    # Realized diagnostics (OBSERVED at t+1 → diagnostic only, never overlay input)
    # Split into three existence/outcome flags so NaN = "unknown" stays clean:
    "realized_t1_has_open":          "diagnostic_only",
    "realized_t1_has_volume":        "diagnostic_only",
    "realized_t1_traded":            "diagnostic_only",
    "realized_next_open_gap":        "diagnostic_only",
    "realized_t1_limit_open":        "diagnostic_only",
}


# ── Features ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature block toggles + shaping. v1 defaults to CORE set (~25-35 features).
    Extended set (~45-70) gated behind include_extended flag for ablation runs.
    """
    # Block toggles (core)
    include_momentum: bool = True
    include_relative_strength: bool = True
    include_liquidity: bool = True
    include_trend_quality: bool = True
    include_volatility: bool = True
    include_exhaustion: bool = True
    include_regime: bool = True
    include_sector: bool = True
    include_suitability: bool = False           # extended only

    include_extended: bool = False              # unlock extended features

    # Cross-sectional transforms
    emit_cs_rank: bool = True                   # percentile rank by rebalance_date
    emit_cs_zscore: bool = False                # z-score by rebalance_date
    emit_raw: bool = True                       # keep raw values alongside ranks

    # Momentum horizons (trading days)
    momentum_lookbacks: tuple[int, ...] = (21, 63, 126, 252)   # ~1m/3m/6m/12m
    skip_recent_days: int = 21                                  # 12-1, 6-1, 3-1 pattern

    # Regime & macro
    macro_us_close_shift: int = 1               # leakage shift for US-close series
    macro_bist_session_shift: int = 0


# ── Dataset / walk-forward ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitConfig:
    """
    Expanding walk-forward, date-based. Random splits are forbidden.
    For monthly rebalances, use months as the fold unit.

    Strategy (v1): fixed anchor splits that expand train window forward.
    Embargo between train_end and val_start ≥ primary_horizon + safety margin.
    """
    train_start: str = "2015-01-01"
    embargo_months: int = 2          # ≥ primary_horizon + 1

    # Fold anchors: (train_end, val_start, val_end, test_start, test_end)
    # Frozen. Hyperparameter tuning must not reshape folds.
    folds: tuple[tuple[str, str, str, str, str, str], ...] = (
        ("fold1", "2021-12-31", "2022-03-01", "2022-08-31", "2022-11-01", "2023-04-30"),
        ("fold2", "2022-12-31", "2023-03-01", "2023-08-31", "2023-11-01", "2024-04-30"),
        ("fold3", "2023-12-31", "2024-03-01", "2024-08-31", "2024-11-01", "2025-04-30"),
        ("fold4", "2024-12-31", "2025-03-01", "2025-08-31", "2025-11-01", "2026-04-30"),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """
    Two modelling tracks:
      - rank_composite: weighted linear of standardized features (explainable)
      - lightgbm: tree ensemble (regressor on L2 excess, classifier on L3 outperform)
    """
    track: str = "rank_composite"    # "rank_composite" | "lightgbm_reg" | "lightgbm_cls"

    # Rank composite weights (raw name → weight). Sum does not need to be 1;
    # features are z-scored cross-sectionally before weighting.
    rank_weights: dict[str, float] = field(default_factory=lambda: {
        "ret_12m_skip_1m": 1.0,
        "ret_6m_skip_1m": 0.6,
        "rs_xu100_6m": 0.5,
        "liquidity_rank_cs": 0.3,
        "trend_quality_score": 0.4,
        "exhaustion_penalty": -0.5,
    })

    # LightGBM hyperparams (used when track starts with lightgbm_)
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_n_estimators: int = 500
    lgbm_min_data_in_leaf: int = 200
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.8
    lgbm_early_stopping_rounds: int = 50

    # Calibration for classifier track
    classifier_calibration: str = "none"   # "none" | "platt" | "isotonic"


# ── Portfolio construction ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioConfig:
    """
    Score → portfolio mapping. v1 defaults: top 10 equal-weight, hold to next
    rebalance. Caps and weighting variants available for ablation.
    """
    top_n: int | None = 10                      # set None to use top_quantile
    top_quantile: float | None = None           # e.g. 0.10 → top decile

    weighting: str = "equal"                    # "equal" | "score"
    max_weight_per_stock: float = 0.15
    max_weight_per_sector: float = 0.40

    hold_until_next_rebalance: bool = True
    skip_if_liquidity_below: float | None = None  # optional recheck at execution

    # ── Risk overlay (ex-ante only) ──────────────────────────────────────
    # Soft downweight (not hard filter) applied to the primary score using
    # execution-proxy columns. Each weight multiplies the corresponding
    # ex-ante proxy before subtracting from score. ALL features here MUST
    # be rebalance_date-known — realized t+1 diagnostics are NEVER accepted.
    # Applied identically to baseline and ML scores so the comparison is
    # not contaminated by overlay-only differences.
    risk_overlay_enabled: bool = True
    risk_overlay_weights: dict[str, float] = field(default_factory=lambda: {
        "proxy_limit_open_freq_60d":   0.40,
        "proxy_stale_open_freq_60d":   0.30,
        "proxy_open_dislocation_20d":  0.20,
        "proxy_gap_dispersion_20d":    0.10,
    })
    risk_overlay_mode: str = "soft_penalty"   # "soft_penalty" only in v1
    risk_overlay_strength: float = 0.25        # scales total penalty vs score std


# ── Backtest cost model ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    """
    Cost model is mandatory. Gross-only results are not acceptable as a final
    claim (spec §15).
    """
    commission_bps: float = 25.0       # round-trip split across buy+sell legs
    slippage_bps: float = 15.0         # per-leg market impact + spread proxy
    execution: str = "next_open"       # "next_open" | "rebalance_close"
    initial_capital: float = 1_000_000.0
    benchmark_symbol: str = "XU100.IS"


# ── Paths ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathConfig:
    root: str = "output/nyxmomentum"
    data: str = "output/nyxmomentum/data"
    artifacts: str = "output/nyxmomentum/artifacts"
    reports: str = "output/nyxmomentum/reports"
    cache: str = "output/nyxmomentum/cache"


# ── Root ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    execution_proxy: ExecutionProxyConfig = field(default_factory=ExecutionProxyConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathConfig = field(default_factory=PathConfig)


CONFIG = Config()


# Leakage helper sets — identical to nyxexpansion for consistency.
US_CLOSE_MACRO_SYMBOLS = frozenset({
    'VIX', '^VIX',
    'DXY', 'DX-Y.NYB',
    'SPY', 'QQQ', '^DJI', '^IXIC', '^GSPC',
    'USDTRY', 'USDTRY=X',
    'BTC-USD', 'ETH-USD',
})

BIST_SESSION_SYMBOLS = frozenset({
    'XU100', 'XU100.IS', 'XU030.IS', 'XU050.IS', 'XU100D.IS',
    'XBANK.IS', 'XUSIN.IS', 'XTCRT.IS', 'XELKT.IS', 'XINSA.IS',
    'XGIDA.IS', 'XKMYA.IS', 'XULAS.IS', 'XHOLD.IS', 'XTRZM.IS',
    'XMADN.IS', 'XILTM.IS', 'XGMYO.IS', 'XTEKS.IS',
})
