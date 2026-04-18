"""
Backtest Engine — 7 katmanı baştan sona zincirleyen walking-skeleton pipeline.

Akış (point-in-time değil; walking skeleton tüm panelleri tek seferde hesaplar,
sonra trigger tarihlerini kronolojik işler — ilk iterasyon için yeterli):

    1. Layer 0 — universe filter (per-bar tradability)
    2. Layer 1 — factor panel (4 block × 3 feature)
    3. Layer 2 — vetos + setup scorer → watchlist
    4. Layer 3 — trigger detection (bucket router + dedup)
    5. Layer 4 — edge filter (exit-sim heuristic; approved=True kalanlar)
    6. Layer 5 — sizing (quality × Kelly)
    7. Layer 6 — portfolio constraints + portfolio state (simulate fills/exits)
    8. Layer 7 — attribution summary + factor decay

Point-in-time leakage: walking skeleton seviyesinde faktör paneli lookahead
kullanmaz (rolling/shift), trigger 'date' bar'ında tetikleniyor ve giriş
bir sonraki bar'ın open'ı olarak modellenmiyor — entry = close[date]. Bu
basitleştirmeyi sonraki iterasyonda sıkılaştırırız.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alpha_v2.config import Config
from alpha_v2.layer0_universe.filters import apply_universe_filter, summarize_universe
from alpha_v2.layer1_factory.factory import compute_factors
from alpha_v2.layer2_setup.vetos import compute_vetos
from alpha_v2.layer2_setup.scorer import compute_setup_scores
from alpha_v2.layer3_trigger.router import detect_triggers
from alpha_v2.layer4_edge.filter import filter_edge
from alpha_v2.layer5_execution.sizing import size_position
from alpha_v2.layer6_portfolio.state import PortfolioState, Position
from alpha_v2.layer6_portfolio.constraints import apply_portfolio_constraints
from alpha_v2.layer7_monitoring.attribution import (
    performance_summary,
    per_bucket_attribution,
    per_regime_attribution,
)
from exits import BUCKET_CONFIGS, simulate_exit
from exits.simulator import _compute_atr, _extract_ohlc


@dataclass
class BacktestResult:
    trades: list[dict]
    summary: dict
    bucket_attribution: pd.DataFrame
    regime_attribution: pd.DataFrame | None
    universe_summary: dict
    watchlist_size: int
    n_triggers: int
    n_edge_approved: int
    n_admitted: int
    factor_panel: pd.DataFrame = field(default_factory=pd.DataFrame)

    def brief(self) -> str:
        s = self.summary
        return (
            f"n_trades={s.get('n_trades', 0)} "
            f"WR={s.get('win_rate', 0.0)*100:.1f}% "
            f"total_ret={s.get('total_return_pct', 0.0):.2f}% "
            f"CAGR={s.get('cagr_pct', 0.0):.2f}% "
            f"Sharpe={s.get('sharpe', 0.0):.2f} "
            f"Sortino={s.get('sortino', 0.0):.2f} "
            f"maxDD={s.get('max_dd_pct', 0.0):.2f}% "
            f"PF={s.get('profit_factor', 0.0):.2f} "
            f"alpha={s.get('alpha_cagr_pct', 0.0):.2f}pp/yr"
        )


def _simulate_one_trade(
    df: pd.DataFrame,
    bar_idx: int,
    bucket: str,
    entry_price: float,
    stop_price: float,
) -> dict:
    """Layer 4 ile aynı simulator, şimdi gerçek fill olarak. Exit döner."""
    cfg_bucket = BUCKET_CONFIGS[bucket]
    _, h, l, c = _extract_ohlc(df)
    atr = _compute_atr(h, l, c, period=14)
    result = simulate_exit(df, bar_idx, cfg_bucket, atr=atr, apply_costs=True)
    return {
        'return_pct_net': result.return_pct,
        'exit_price': result.exit_price,
        'exit_bar_idx': result.exit_idx,
        'bars_held': result.bars_held,
        'reason': result.reason,
    }


def run_backtest(
    all_data: dict[str, pd.DataFrame],
    xu100_df: pd.DataFrame | None,
    cfg: Config,
    *,
    sector_map: dict[str, str] | None = None,
    sector_data_map: dict[str, pd.DataFrame] | None = None,
    enrichment_map: dict[str, pd.DataFrame] | None = None,
    initial_capital: float = 100_000.0,
    ml_models: dict[str, dict] | None = None,
    verbose: bool = True,
) -> BacktestResult:
    """End-to-end 7 katman backtest."""

    # ──────────────── LAYER 0 ────────────────
    uni = apply_universe_filter(all_data, cfg.universe)
    uni_summary = summarize_universe(uni)
    if verbose:
        print(f"[L0] universe: {uni_summary.get('tradable_bars', 0)}/"
              f"{uni_summary.get('total_bars', 0)} bar tradable")

    # Tradable bar filtresi: sadece is_tradable=True olan hisseleri Layer 1+'a geçir
    tradable_idx = uni[uni['is_tradable']].index

    # ──────────────── LAYER 1 ────────────────
    factor_panel = compute_factors(
        all_data, xu100_df, cfg.factory, sector_map=sector_data_map,
        enrichment_map=enrichment_map,
    )
    if verbose:
        n_sec = len(sector_data_map) if sector_data_map else 0
        print(f"[L1] factor panel: {len(factor_panel)} rows, "
              f"{factor_panel.shape[1]} cols  "
              f"(sector map: {n_sec}/{len(all_data)} ticker)")

    # Sadece tradable bar'larda faktör skorlarını tut
    if not factor_panel.empty:
        factor_panel = factor_panel.loc[factor_panel.index.isin(tradable_idx)]

    # ──────────────── LAYER 2 ────────────────
    veto_panel = compute_vetos(all_data, xu100_df, cfg.setup.vetos)
    setup_result = compute_setup_scores(factor_panel, veto_panel, cfg.setup)
    watchlist = setup_result.watchlist
    if verbose:
        print(f"[L2] watchlist: {len(watchlist)} (ticker,date) kombinasyonu")

    # ──────────────── LAYER 3 ────────────────
    triggers = detect_triggers(all_data, watchlist, cfg.trigger,
                                buckets=cfg.buckets)
    if verbose:
        print(f"[L3] triggers: {len(triggers)} event")

    # ──────────────── LAYER 4 ────────────────
    edge_cfg = cfg.edge
    if ml_models:
        # ML model varsa use_ml_model flag'ini override et
        from dataclasses import replace as _dc_replace
        edge_cfg = _dc_replace(edge_cfg, use_ml_model=True)

    edge_decisions = filter_edge(
        triggers, all_data, edge_cfg,
        ml_models=ml_models, factor_panel=factor_panel,
    )
    approved = [d for d in edge_decisions if d.approved]
    if verbose:
        tag = ' [ML]' if ml_models else ' [heuristik]'
        print(f"[L4] edge approved{tag}: {len(approved)}/{len(edge_decisions)}")

    # ──────────────── LAYER 5 ────────────────
    sizings = [size_position(d, cfg.execution) for d in approved]
    # filter size=0
    sizings = [s for s in sizings if s.position_size_pct > 0.0]
    if verbose:
        print(f"[L5] sized positions: {len(sizings)}")

    # ──────────────── LAYER 6 ────────────────
    # Tarihe göre gruplayıp her günün sinyallerini state üzerinde constraint check'ten
    # geçir + kabul edilenleri exit simülasyonuyla kapat
    state = PortfolioState(
        capital=initial_capital, equity=initial_capital, peak_equity=initial_capital,
    )

    # Tarih bazlı iterasyon (naive — tüm siparişi simulate_exit ile ayrı kapatıyor).
    # Not: Açık pozisyonlar birbirinden bağımsız bu walking-skeleton'da. İlerleyen
    # iterasyonlarda gerçek event-driven state (bar-bar fiyat güncelleme) gelir.
    sizings_by_date: dict[pd.Timestamp, list] = {}
    for s in sizings:
        sizings_by_date.setdefault(pd.Timestamp(s.date), []).append(s)

    # Event-driven state: açık pozisyonlar gerçek exit_date'lerine kadar yaşar
    # pending_exits: {exit_date: [(ticker, exit_price, reason), ...]}
    pending_exits: dict[pd.Timestamp, list[tuple[str, float, str]]] = {}

    def _flush_exits(up_to_date: pd.Timestamp) -> None:
        for exit_d in sorted(pending_exits.keys()):
            if exit_d > up_to_date:
                break
            for tkr, exit_px, rsn in pending_exits.pop(exit_d):
                state.close_position(
                    ticker=tkr, exit_date=exit_d,
                    exit_price=exit_px, reason=rsn,
                )

    n_admitted = 0
    for date in sorted(sizings_by_date.keys()):
        # Önce bugüne kadar biten pozisyonları kapat (constraint doğru olsun)
        _flush_exits(date)

        day_candidates = sizings_by_date[date]
        admissions = apply_portfolio_constraints(
            day_candidates, state, cfg.portfolio, sector_map=sector_map,
        )
        for adm in admissions:
            if not adm.admitted:
                continue
            n_admitted += 1
            size_pct = adm.adjusted_size_pct or adm.sizing.position_size_pct
            sector = sector_map.get(adm.sizing.ticker) if sector_map else None

            # Aç
            pos = Position(
                ticker=adm.sizing.ticker,
                bucket=adm.sizing.bucket,
                entry_date=date,
                entry_price=adm.sizing.entry_price,
                stop_price=adm.sizing.stop_price,
                size_pct=size_pct,
                sector=sector,
            )
            state.add_position(pos)

            # Exit tarihi ve fiyatını hesapla, pending'e koy (constraint'in
            # ilerideki günlerde aktif kalması için)
            df = all_data.get(adm.sizing.ticker)
            if df is None:
                continue
            try:
                iloc = df.index.get_loc(date)
            except KeyError:
                continue
            if iloc >= len(df) - 1:
                continue

            trade_res = _simulate_one_trade(
                df, iloc, adm.sizing.bucket,
                adm.sizing.entry_price, adm.sizing.stop_price,
            )
            exit_date = pd.Timestamp(
                df.index[min(iloc + trade_res['bars_held'], len(df) - 1)]
            )
            pending_exits.setdefault(exit_date, []).append(
                (adm.sizing.ticker, trade_res['exit_price'], trade_res['reason'])
            )

    # Son kalan pozisyonları kapat
    if pending_exits:
        last = max(pending_exits.keys())
        _flush_exits(last)

    if verbose:
        print(f"[L6] admitted: {n_admitted}, closed: {len(state.closed_trades)}")

    # ──────────────── LAYER 7 ────────────────
    xu_ret = None
    if xu100_df is not None and not xu100_df.empty:
        xu_c = xu100_df['Close'] if 'Close' in xu100_df.columns else xu100_df['close']
        xu_ret = xu_c.pct_change().dropna()

    summary = performance_summary(
        state.closed_trades, initial_capital=initial_capital,
        xu100_ret_series=xu_ret, all_data=all_data,
    )
    bucket_attr = per_bucket_attribution(state.closed_trades)

    # Regime series (basit proxy): XU100 50g trend up/down
    regime_attr = None
    if xu100_df is not None and not xu100_df.empty:
        xu_c = xu100_df['Close'] if 'Close' in xu100_df.columns else xu100_df['close']
        xu_ema = xu_c.ewm(span=50, adjust=False).mean()
        regime_ser = (xu_c > xu_ema).map({True: 'up', False: 'down'})
        regime_attr = per_regime_attribution(state.closed_trades, regime_ser)

    return BacktestResult(
        trades=state.closed_trades,
        summary=summary,
        bucket_attribution=bucket_attr,
        regime_attribution=regime_attr,
        universe_summary=uni_summary,
        watchlist_size=len(watchlist),
        n_triggers=len(triggers),
        n_edge_approved=len(approved),
        n_admitted=n_admitted,
        factor_panel=factor_panel,
    )
