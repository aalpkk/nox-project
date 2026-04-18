"""
Exit Simulator — bucket-bazlı hybrid çıkış simülatörü.

Giriş: OHLC + entry_idx + bucket
Çıkış: ExitResult (exit_idx, exit_price, return_pct, reason, partial_info)

Intra-bar tetiklenme önceliği (aynı bar'da birden fazla koşul yaşanabilir):
    1. Initial stop (failed breakout dahil) → konservatif, zararı öne al
    2. Partial exit trigger → middle, kazancı locklamaya öncelik
    3. Trailing stop hit
    4. Time stop → bar kapanışında uygulanır

Bu sıra pessimistic (worst-case-first) — backtest'te aşırı iyimserlik olmaması için.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exits.config import (
    BUCKET_CONFIGS,
    BucketConfig,
    CostModel,
    DEFAULT_COST_MODEL,
    round_trip_cost,
)
from exits.structure import last_swing_low


@dataclass
class ExitResult:
    entry_idx: int
    entry_price: float
    exit_idx: int
    exit_price: float
    bars_held: int
    reason: str                       # 'stop', 'structure', 'trail', 'time', 'fixed_tp', 'failed_breakout', 'open'
    return_pct: float                 # net return (cost included if requested)
    gross_return_pct: float           # cost-öncesi return
    partial_taken: bool = False
    partial_idx: int | None = None
    partial_price: float | None = None
    partial_return_pct: float = 0.0
    bucket: str = ''

    def as_dict(self) -> dict:
        return {
            'entry_idx': self.entry_idx,
            'entry_price': self.entry_price,
            'exit_idx': self.exit_idx,
            'exit_price': self.exit_price,
            'bars_held': self.bars_held,
            'reason': self.reason,
            'return_pct': self.return_pct,
            'gross_return_pct': self.gross_return_pct,
            'partial_taken': self.partial_taken,
            'partial_return_pct': self.partial_return_pct,
            'bucket': self.bucket,
        }


def _extract_ohlc(df: pd.DataFrame):
    """OHLC kolonlarını numpy array olarak döndür (uppercase veya lowercase destekle)."""
    cols = df.columns
    def _get(name):
        for c in (name, name.lower(), name.upper(), name.capitalize()):
            if c in cols:
                return df[c].to_numpy(dtype=float)
        raise KeyError(f"OHLC kolonu bulunamadı: {name}")
    return _get('Open'), _get('High'), _get('Low'), _get('Close')


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder ATR — numpy native, NaN boşlukları 0'a çekilir."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    # ilk bölüm için atr[period-1] ile doldur
    if n >= period:
        atr[:period - 1] = atr[period - 1]
    return atr


def simulate_exit(
    df: pd.DataFrame,
    entry_idx: int,
    bucket: str | BucketConfig = 'swing',
    *,
    entry_price: float | None = None,
    atr: np.ndarray | None = None,
    cost_model: CostModel | None = None,
    apply_costs: bool = True,
) -> ExitResult:
    """
    Tek bir trade için hybrid exit simülasyonu.

    Args:
        df: OHLC DataFrame (monoton artan index)
        entry_idx: iloc-bazlı giriş bar index'i (entry bar'ın KAPANIŞINDA giriliyor varsayılır)
        bucket: 'intraday' | 'swing' | 'position' veya BucketConfig
        entry_price: None ise df.Close[entry_idx] kullanılır
        atr: önceden hesaplanmış ATR serisi (None ise 14-period hesaplanır)
        cost_model: BIST cost modeli
        apply_costs: True ise return_pct cost düşülmüş döner

    Returns:
        ExitResult
    """
    cfg = bucket if isinstance(bucket, BucketConfig) else BUCKET_CONFIGS[bucket]
    cost = cost_model or DEFAULT_COST_MODEL

    o, h, l, c = _extract_ohlc(df)
    n = len(c)

    if entry_idx >= n - 1:
        # Entry bar'ın ertesi günü yok → giremeyiz
        return ExitResult(
            entry_idx=entry_idx, entry_price=float(c[entry_idx]),
            exit_idx=entry_idx, exit_price=float(c[entry_idx]),
            bars_held=0, reason='no_future', return_pct=0.0,
            gross_return_pct=0.0, bucket=cfg.name,
        )

    if atr is None:
        atr = _compute_atr(h, l, c, period=14)

    entry_price = float(entry_price if entry_price is not None else c[entry_idx])
    entry_atr = float(atr[entry_idx]) if atr[entry_idx] > 0 else max(1e-6, entry_price * 0.02)

    # ══════════════════════════════════════════
    # STOP & TRAIL DURUM DEĞIŞKENLERI
    # ══════════════════════════════════════════

    # Initial stop: entry - N*ATR (veya fixed SL intraday için)
    if cfg.use_fixed_tp:
        initial_stop = entry_price * (1 - cfg.fixed_sl_pct / 100.0)
        fixed_tp = entry_price * (1 + cfg.fixed_tp_pct / 100.0)
    else:
        initial_stop = entry_price - cfg.initial_stop_atr * entry_atr
        fixed_tp = None

    current_stop = initial_stop
    highest_since_entry = entry_price
    trail_activated = False
    partial_taken = False
    partial_idx: int | None = None
    partial_price: float | None = None

    # Structure stop: giriş anında hesap
    if cfg.use_structure:
        sw = last_swing_low(
            l, entry_idx,
            pivot_strength=cfg.pivot_strength,
            lookback=cfg.structure_lookback,
        )
        structure_stop = sw.price if sw else None
        if structure_stop is not None:
            # Initial stop ile structure'ın SIKI olanını al (daha yüksek = risk az)
            current_stop = max(current_stop, structure_stop)
    else:
        structure_stop = None

    # ══════════════════════════════════════════
    # BAR BY BAR SIMULATION
    # ══════════════════════════════════════════

    exit_idx = entry_idx
    exit_price = entry_price
    reason = 'open'

    for i in range(entry_idx + 1, min(entry_idx + 1 + cfg.time_stop_bars, n)):
        bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
        bars_held = i - entry_idx

        # ─── 0. GAP checks (open jumps past a level → fill at open)
        # Gap down past stop
        if bar_o <= current_stop:
            exit_idx = i
            exit_price = bar_o
            reason = 'stop'
            break
        # Gap up past TP (intraday)
        if cfg.use_fixed_tp and fixed_tp is not None and bar_o >= fixed_tp:
            exit_idx = i
            exit_price = bar_o
            reason = 'fixed_tp'
            break

        tp_hit = cfg.use_fixed_tp and fixed_tp is not None and bar_h >= fixed_tp
        sl_hit = bar_l <= current_stop

        # ─── 1+2. TP vs SL race — open-direction ile karar ver
        #         (pessimistic: open'dan çıkış yönü)
        if tp_hit and sl_hit:
            # Open direction belirleyici: aşağı açıldıysa SL önce, yukarı açıldıysa TP önce
            if bar_o < entry_price:
                exit_idx = i
                exit_price = current_stop
                reason = 'stop'
                break
            else:
                exit_idx = i
                exit_price = fixed_tp  # type: ignore
                reason = 'fixed_tp'
                break
        elif tp_hit:
            exit_idx = i
            exit_price = fixed_tp  # type: ignore
            reason = 'fixed_tp'
            break
        elif sl_hit:
            exit_idx = i
            exit_price = current_stop
            reason = 'stop'
            break

        # ─── 3. FAILED BREAKOUT (ilk N bar içinde entry altı kapanış)
        if (cfg.use_failed_breakout
                and bars_held <= cfg.failed_breakout_bars
                and bar_c < entry_price * 0.995):  # %0.5 buffer
            exit_idx = i
            exit_price = bar_c
            reason = 'failed_breakout'
            break

        # ─── 4. PARTIAL EXIT TRIGGER (henüz alınmadıysa)
        partial_target = entry_price + cfg.partial_trigger_atr * entry_atr
        if (cfg.use_partial and not partial_taken
                and bar_h >= partial_target):
            partial_taken = True
            partial_idx = i
            partial_price = partial_target
            if cfg.partial_move_stop_to_be:
                # stop'u breakeven'a çek (ama yukarı doğru)
                current_stop = max(current_stop, entry_price)

        # ─── 5. ATR TRAIL ACTIVATION & UPDATE
        if cfg.use_atr_trail:
            highest_since_entry = max(highest_since_entry, bar_h)
            activation_price = entry_price + cfg.trail_activate_atr * entry_atr
            if not trail_activated and highest_since_entry >= activation_price:
                trail_activated = True
            if trail_activated:
                trail_stop = highest_since_entry - cfg.trail_atr_mult * entry_atr
                current_stop = max(current_stop, trail_stop)

        # ─── 6. STRUCTURE TRAIL (yeni swing low oluştuysa stop'u yukarı çek)
        if cfg.use_structure:
            new_sw = last_swing_low(
                l, i, pivot_strength=cfg.pivot_strength,
                lookback=cfg.structure_lookback,
            )
            if new_sw is not None and new_sw.price > current_stop:
                current_stop = new_sw.price

    else:
        # for-else: break olmadıysa time stop
        final_idx = min(entry_idx + cfg.time_stop_bars, n - 1)
        exit_idx = final_idx
        exit_price = float(c[final_idx])
        reason = 'time'

    # ══════════════════════════════════════════
    # RETURN HESABI (partial + final blend)
    # ══════════════════════════════════════════

    bars_held = exit_idx - entry_idx
    final_gross_ret = (exit_price / entry_price - 1.0) * 100.0

    if partial_taken and partial_price is not None:
        partial_gross_ret = (partial_price / entry_price - 1.0) * 100.0
        blended_gross = (
            cfg.partial_fraction * partial_gross_ret
            + (1.0 - cfg.partial_fraction) * final_gross_ret
        )
    else:
        partial_gross_ret = 0.0
        blended_gross = final_gross_ret

    if apply_costs:
        net_ret = blended_gross - round_trip_cost(cost)
    else:
        net_ret = blended_gross

    return ExitResult(
        entry_idx=entry_idx,
        entry_price=entry_price,
        exit_idx=exit_idx,
        exit_price=float(exit_price),
        bars_held=int(bars_held),
        reason=reason,
        return_pct=float(net_ret),
        gross_return_pct=float(blended_gross),
        partial_taken=partial_taken,
        partial_idx=partial_idx,
        partial_price=float(partial_price) if partial_price else None,
        partial_return_pct=float(partial_gross_ret),
        bucket=cfg.name,
    )


def simulate_exits_batch(
    df: pd.DataFrame,
    entry_indices: list[int],
    bucket: str | BucketConfig = 'swing',
    *,
    apply_costs: bool = True,
) -> list[ExitResult]:
    """Aynı hisse üzerinde birden çok trade simülasyonu — ATR'yi tek seferde hesaplar."""
    cfg = bucket if isinstance(bucket, BucketConfig) else BUCKET_CONFIGS[bucket]
    o, h, l, c = _extract_ohlc(df)
    atr = _compute_atr(h, l, c, period=14)
    return [
        simulate_exit(df, ei, cfg, atr=atr, apply_costs=apply_costs)
        for ei in entry_indices
    ]
