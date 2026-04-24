"""
CTE Trigger — compression-to-expansion breakout detector.

Tek-ticker OHLCV'den per-bar trigger flag'leri + long-panel üretir.

İki bağımsız trigger ailesi:
  trigger_hb : horizontal base breakout
    hb_valid ∧ hb_is_first_break ∧ bar_quality_pass
  trigger_fc : falling channel breakout
    fc_valid ∧ fc_is_first_break ∧ bar_quality_pass

Birlik:
  trigger_cte = trigger_hb | trigger_fc

setup_type ∈ {"hb", "fc", "both"} — overlap kaybolmaz. Primary seçim eval
tarafında ya da bir quality tie-breaker'la yapılır; burada kör kural yok.

Leakage
-------
Tüm girdi blokları kendi içinde `[t-W, t-1]` penceresinde hesaplandı. Bar-t
verisi yalnızca `is_first_break` ve `bar_quality_pass`'te kullanılır — bu
iki bileşen tetikleme SEMANTICS'in kendisidir (bar t'nin kapanışı).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import (
    BreakoutBarParams,
    CompressionParams,
    DryupParams,
    FallingChannelParams,
    FirstBreakParams,
    HorizontalBaseParams,
)
from cte.breakout_bar import compute_breakout_bar
from cte.compression import compute_compression
from cte.firstness import compute_firstness
from cte.structure import compute_structure
from cte.volume import compute_volume


def compute_trigger(
    df: pd.DataFrame,
    *,
    comp: CompressionParams | None = None,
    hb: HorizontalBaseParams | None = None,
    fc: FallingChannelParams | None = None,
    dry: DryupParams | None = None,
    fb: FirstBreakParams | None = None,
    bar: BreakoutBarParams | None = None,
) -> pd.DataFrame:
    """Single-ticker OHLCV → per-bar trigger frame.

    Returns DataFrame indexed like `df` with columns:
        trigger_hb, trigger_fc, trigger_cte, setup_type
        + all structure/compression/volume/bar/firstness feature columns
          (merged, prefix'ler çakışmaz).
    """
    if comp is None:
        comp = CompressionParams()
    if hb is None:
        hb = HorizontalBaseParams()
    if fc is None:
        fc = FallingChannelParams()
    if dry is None:
        dry = DryupParams()
    if fb is None:
        fb = FirstBreakParams()
    if bar is None:
        bar = BreakoutBarParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    st = compute_structure(df, comp=comp, hb=hb, fc=fc)
    cp = compute_compression(df, comp=comp)
    vl = compute_volume(df, dry=dry, comp=comp)
    bq = compute_breakout_bar(
        df, structure_vol_ref=vl.get("structure_vol_ref"), bar=bar,
    )
    fn = compute_firstness(df, st, fb=fb, comp=comp)

    trigger_hb = (
        st.get("hb_valid", pd.Series(False, index=df.index))
        & fn.get("hb_is_first_break", pd.Series(False, index=df.index))
        & bq.get("bar_quality_pass", pd.Series(False, index=df.index))
    )
    trigger_fc = (
        st.get("fc_valid", pd.Series(False, index=df.index))
        & fn.get("fc_is_first_break", pd.Series(False, index=df.index))
        & bq.get("bar_quality_pass", pd.Series(False, index=df.index))
    )
    trigger_cte = (trigger_hb | trigger_fc).astype(bool)

    setup_type = pd.Series("", index=df.index, dtype=object)
    setup_type = setup_type.where(~(trigger_hb & ~trigger_fc), "hb")
    setup_type = setup_type.where(~(trigger_fc & ~trigger_hb), "fc")
    setup_type = setup_type.where(~(trigger_hb & trigger_fc), "both")

    meta = pd.DataFrame(
        {
            "trigger_hb": trigger_hb.astype(bool),
            "trigger_fc": trigger_fc.astype(bool),
            "trigger_cte": trigger_cte,
            "setup_type": setup_type,
        },
        index=df.index,
    )

    return pd.concat([meta, st, cp, vl, bq, fn], axis=1)


def compute_trigger_panel(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    comp: CompressionParams | None = None,
    hb: HorizontalBaseParams | None = None,
    fc: FallingChannelParams | None = None,
    dry: DryupParams | None = None,
    fb: FirstBreakParams | None = None,
    bar: BreakoutBarParams | None = None,
    min_bars: int = 60,
) -> pd.DataFrame:
    """Multi-ticker panel → long DataFrame of trigger_cte=True rows.

    Returns columns:
        ticker, date, close, trigger_hb, trigger_fc, trigger_cte, setup_type,
        + all feature columns from compute_trigger.
    """
    rows: list[pd.DataFrame] = []
    for ticker, sub in data_by_ticker.items():
        if sub is None or sub.empty or len(sub) < min_bars:
            continue
        feat = compute_trigger(
            sub, comp=comp, hb=hb, fc=fc, dry=dry, fb=fb, bar=bar,
        )
        if feat.empty:
            continue
        mask = feat["trigger_cte"]
        if not mask.any():
            continue
        keep = feat.loc[mask].copy()
        keep["ticker"] = ticker
        keep["close"] = sub.loc[mask, "Close"].astype(float)
        keep.index.name = "date"
        keep = keep.reset_index()
        rows.append(keep)

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker", "date", "close",
                "trigger_hb", "trigger_fc", "trigger_cte", "setup_type",
            ],
        )

    panel = pd.concat(rows, ignore_index=True)
    front = ["ticker", "date", "close",
             "trigger_hb", "trigger_fc", "trigger_cte", "setup_type"]
    rest = [c for c in panel.columns if c not in front]
    panel = panel[front + rest]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
