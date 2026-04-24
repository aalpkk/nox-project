"""
CTE trigger relaxation sweep.

Altyapı oturumu diagnostic: kaç sinyal geliyor, per-line/day density nasıl,
runner_15 base rate bozulmadan ne kadar gevşetebiliriz?

Usage:
  python -m cte.tools.trigger_sweep
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.config import BreakoutBarParams, CONFIG
from cte.dataset import _build_single_ticker
from cte.features import FeatureParams


SWEEP_CONFIGS = {
    "baseline":  BreakoutBarParams(),  # current
    "relax_m":   BreakoutBarParams(min_return_1d=0.04, min_rvol=1.5,
                                   min_close_loc_bar=0.65, min_body_pct_range=0.35),
    "relax_L":   BreakoutBarParams(min_return_1d=0.03, min_rvol=1.3,
                                   min_close_loc_bar=0.60, min_body_pct_range=0.30),
    "relax_ret": BreakoutBarParams(min_return_1d=0.03),  # only return lowered
    "relax_vol": BreakoutBarParams(min_rvol=1.3),        # only vol lowered
}


def _panel_summary(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {"N": 0}
    trig = panel[panel["trigger_cte"]]
    hb = trig[trig["trigger_hb"]]
    fc = trig[trig["trigger_fc"]]

    hb_per_day = hb.groupby("date").size()
    fc_per_day = fc.groupby("date").size()

    out = {
        "N": len(trig),
        "N_hb": int(trig["trigger_hb"].sum()),
        "N_fc": int(trig["trigger_fc"].sum()),
        "N_both": int(((trig["setup_type"] == "both")).sum()),
        "hb_median_per_day": float(hb_per_day.median()) if len(hb_per_day) else 0,
        "hb_max_per_day":    int(hb_per_day.max()) if len(hb_per_day) else 0,
        "hb_p90_per_day":    float(hb_per_day.quantile(0.90)) if len(hb_per_day) else 0,
        "fc_median_per_day": float(fc_per_day.median()) if len(fc_per_day) else 0,
        "fc_max_per_day":    int(fc_per_day.max()) if len(fc_per_day) else 0,
        "fc_p90_per_day":    float(fc_per_day.quantile(0.90)) if len(fc_per_day) else 0,
    }
    for tgt in ("runner_15", "runner_10", "runner_20"):
        if tgt in trig.columns:
            out[f"{tgt}_rate"] = float(trig[tgt].dropna().mean())
            if "trigger_hb" in trig.columns:
                out[f"{tgt}_rate_hb"] = float(
                    trig.loc[trig["trigger_hb"], tgt].dropna().mean()
                )
            if "trigger_fc" in trig.columns:
                out[f"{tgt}_rate_fc"] = float(
                    trig.loc[trig["trigger_fc"], tgt].dropna().mean()
                )
    return out


def main() -> int:
    os.chdir(str(_ROOT))
    master_path = Path(CONFIG.data.yf_cache_path)
    xu100_path = Path("output/xu100_cache.parquet")
    if not master_path.exists():
        print(f"❌ master not found: {master_path}")
        return 2

    print(f"[load] {master_path}")
    master = pd.read_parquet(master_path)
    ohlcv_by_ticker = {
        tk: sub.sort_index() for tk, sub in master.groupby("ticker")
    }
    print(f"[load] {len(ohlcv_by_ticker)} tickers")

    xu100_close = None
    if xu100_path.exists():
        xu100_close = pd.read_parquet(xu100_path)["Close"].astype(float)

    feat_params = FeatureParams()

    results = {}
    for name, bar in SWEEP_CONFIGS.items():
        print(f"\n═══ config={name} ret≥{bar.min_return_1d:.2f} rvol≥{bar.min_rvol:.1f} "
              f"close_loc≥{bar.min_close_loc_bar:.2f} body≥{bar.min_body_pct_range:.2f} ═══")
        cfg_v = replace(CONFIG, bar=bar)
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i, (tk, sub) in enumerate(ohlcv_by_ticker.items(), start=1):
                out = _build_single_ticker(tk, sub, xu100_close, cfg_v, feat_params)
                if out is not None:
                    rows.append(out)
                if i % 100 == 0:
                    print(f"  {i}/{len(ohlcv_by_ticker)}  acc={sum(len(r) for r in rows)}")
        panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        summary = _panel_summary(panel)
        results[name] = summary
        print(f"  → N={summary.get('N', 0)}  "
              f"hb={summary.get('N_hb', 0)} fc={summary.get('N_fc', 0)} "
              f"both={summary.get('N_both', 0)}  "
              f"runner_15_rate={summary.get('runner_15_rate', 0):.3f}")
        print(f"  per-day hb: med={summary.get('hb_median_per_day', 0):.1f} "
              f"p90={summary.get('hb_p90_per_day', 0):.1f} "
              f"max={summary.get('hb_max_per_day', 0)}")
        print(f"  per-day fc: med={summary.get('fc_median_per_day', 0):.1f} "
              f"p90={summary.get('fc_p90_per_day', 0):.1f} "
              f"max={summary.get('fc_max_per_day', 0)}")

    print("\n\n════ SWEEP TABLE ════")
    cols = ["N", "N_hb", "N_fc", "N_both",
            "hb_median_per_day", "hb_p90_per_day", "hb_max_per_day",
            "fc_median_per_day", "fc_p90_per_day", "fc_max_per_day",
            "runner_15_rate", "runner_15_rate_hb", "runner_15_rate_fc",
            "runner_10_rate", "runner_20_rate"]
    df = pd.DataFrame(results).T.reindex(columns=cols)
    print(df.round(3).to_string())
    out_path = Path("output/cte_trigger_sweep.csv")
    df.to_csv(out_path)
    print(f"\n[WRITE] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
