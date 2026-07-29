#!/usr/bin/env python3
"""index_bt koşumlarını tek özet CSV'ye toplar (rapor için)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "output" / "index_bt"

SCAN_OF = {
    "alpha": "alpha_scan", "combo": "screener_combo", "nyxexp": "nyxexpansion",
    "tavan": "tavan_scanner", "hwobos": "hw_obos", "sbt": "sbt_1700",
    "trident": "trident_de_v1",
}


def _uni(name: str) -> tuple[str, str]:
    if name.endswith("_cur"):
        return name.split("_")[1].upper(), "current"
    return name.split("_", 1)[1].upper(), "pit"


def main():
    rows = []
    for p in sorted(BT.glob("*/results.json")):
        run = p.parent.name
        scan = SCAN_OF.get(run.split("_")[0], run.split("_")[0])
        uni, mode = _uni(run)
        d = json.loads(p.read_text(encoding="utf-8"))

        if "cohorts" in d:                       # ortak motor çıktısı
            for c, b in d["cohorts"].items():
                if not b.get("n_signals"):
                    continue
                s = b["signal_h5"]; u = b["universe_h5"]
                r = (b.get("random") or {}).get("h5", {})
                t = b.get("trades", {})
                rows.append(dict(
                    scan=scan, run=run, universe=uni, mode=mode, cohort=c, segment="full",
                    n_sig=s["n"], hit5=s["hit"], mean5=s["mean"], med5=s["median"],
                    uni_mean5=u["mean"], rnd_mean5=r.get("mean_mean"), rnd_p95_5=r.get("mean_p95"),
                    edge5=s["mean"] - (r.get("mean_mean") or 0),
                    beats_rnd_p95=bool(s["mean"] > (r.get("mean_p95") or 1e9)),
                    trade_n=t.get("n"), trade_hit=t.get("hit"), trade_mean=t.get("mean"),
                    trade_med=t.get("median"), trade_pf=t.get("pf"),
                ))
        elif "layerA_signal" in d:                # alpha sim çıktısı
            for seg, b in d["layerA_signal"].items():
                for coh, key, rkey in (("all_candidates", "signal_h5", "random_all"),
                                       ("top4", "top4_h5", "random_top4")):
                    s = b[key]; u = b["universe_h5"]; r = b[rkey]["h5"]
                    tb = d["layerB_trade"]["conservative"].get(seg, {})
                    t = tb.get("all" if coh == "all_candidates" else "top4", {})
                    rows.append(dict(
                        scan=scan, run=run, universe=uni, mode=mode, cohort=coh, segment=seg,
                        n_sig=s["n"], hit5=s["hit"], mean5=s["mean"], med5=s["median"],
                        uni_mean5=u["mean"], rnd_mean5=r["mean_mean"], rnd_p95_5=r["mean_p95"],
                        edge5=s["mean"] - r["mean_mean"],
                        beats_rnd_p95=bool(s["mean"] > r["mean_p95"]),
                        trade_n=t.get("n"), trade_hit=t.get("hit"), trade_mean=t.get("mean"),
                        trade_med=t.get("median"), trade_pf=t.get("pf"),
                    ))

    df = pd.DataFrame(rows)
    out = BT / "summary.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"{len(df)} satır → {out}")

    # portföy katmanı (yalnızca alpha)
    prows = []
    for p in sorted(BT.glob("alpha_*/results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        uni, mode = _uni(p.parent.name)
        for k, m in d.get("layerC_portfolio", {}).items():
            if not k.endswith("_full") or "hiliq" in k or "RAW" in k:
                continue
            prows.append(dict(run=p.parent.name, universe=uni, mode=mode, config=k,
                              **{x: m.get(x) for x in
                                 ("cagr_pct", "max_dd_pct", "sharpe", "total_return_pct",
                                  "benchmark_total_pct", "alpha_vs_xu100_pct", "beta",
                                  "n_trades")}))
    pd.DataFrame(prows).to_csv(BT / "summary_portfolio.csv", index=False, float_format="%.4f")
    print(f"{len(prows)} satır → {BT/'summary_portfolio.csv'}")


if __name__ == "__main__":
    main()
