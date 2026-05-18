"""SMS + ICE — on-demand runner for arbitrary ticker list.

Pulls Matriks institutional flow periods + N-day daily history per ticker,
loads MKK from GH Pages, runs calc_smart_money_score + calc_ice, writes
JSON + Markdown summary under output/.

Run locally:
    PYTHONPATH=. MATRIKS_API_KEY=... MATRIKS_CLIENT_ID=... \
        python3 tools/sms_ice_ondemand.py --tickers VBTYZ GOKNR TUCLK DAPGM \
                                         --hist-days 20 --asof 2026-05-18

CI usage: TICKERS env (space- or comma-separated), HIST_DAYS env, ASOF env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.matriks_client import MatriksClient
from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day
from agent.smart_money import calc_smart_money_score, format_sms_detail
from agent.institutional import calc_ice, format_sms_detail as format_ice_detail
from agent.scanner_reader import fetch_mkk_data


def _parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in re.split(r"[,\s]+", s) if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Ticker list (overrides TICKERS env).")
    ap.add_argument("--hist-days", type=int,
                    default=int(os.environ.get("HIST_DAYS", "20")),
                    help="Daily flow history window (default 20).")
    ap.add_argument("--asof", default=os.environ.get("ASOF") or
                    datetime.utcnow().strftime("%Y-%m-%d"),
                    help="Asof tag for output filenames.")
    args = ap.parse_args()

    tickers = args.tickers
    if not tickers:
        env_t = os.environ.get("TICKERS", "")
        tickers = _parse_tickers(env_t)
    if not tickers:
        print("ERROR: no tickers (pass --tickers or TICKERS env)", file=sys.stderr)
        return 2

    if not os.environ.get("MATRIKS_API_KEY") or not os.environ.get("MATRIKS_CLIENT_ID"):
        print("ERROR: MATRIKS_API_KEY / MATRIKS_CLIENT_ID not set", file=sys.stderr)
        return 2

    asof = args.asof
    hist_days = args.hist_days

    os.makedirs("output", exist_ok=True)
    out_md = f"output/sms_ice_{asof}.md"
    out_json = f"output/sms_ice_{asof}.json"

    client = MatriksClient()

    print(f"[sms-ice] asof={asof}  tickers={tickers}  hist_days={hist_days}")
    print("[sms-ice] fetching MKK from GH pages ...")
    mkk_resp = fetch_mkk_data()
    mkk_map = (mkk_resp or {}).get("data") if mkk_resp else None
    if mkk_map:
        print(f"  MKK: {len(mkk_map)} tickers")
    else:
        print("  MKK: unavailable (S5 will fall back)")

    takas_data: dict = {}
    takas_history: dict = {}
    failures: list[tuple[str, str]] = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            flows = client.get_institutional_flow_periods(ticker, top=10)
            if flows:
                td = flows_to_takas_data(flows, ticker)
                takas_data.update(td)
                n_brokers = len(td.get(ticker, {}).get("kurumlar", []))
                print(f"  flow periods OK ({n_brokers} brokers)")
            else:
                print("  no flow periods")
                failures.append((ticker, "no_flow_periods"))
                continue

            hist = client.get_daily_flow_history(ticker, days=hist_days, top=10)
            for date_str, flow_resp in hist.items():
                day = flow_to_takas_history_day(flow_resp, ticker, date_str)
                for d, d_data in day.items():
                    if d not in takas_history:
                        takas_history[d] = {}
                    takas_history[d].update(d_data)
            print(f"  history days: {len(hist)}")
        except Exception as e:
            failures.append((ticker, repr(e)))
            print(f"  {ticker} failed: {e}")

    print(f"\n[sms-ice] takas_history: {len(takas_history)} unique dates")
    print(f"[sms-ice] takas_data: {len(takas_data)} tickers")

    results = []
    for t in tickers:
        rec: dict = {"ticker": t}
        try:
            sms = calc_smart_money_score(t, takas_data, mkk_map)
            if sms:
                rec["sms"] = {
                    "score": sms.score, "label": sms.label, "icon": sms.icon,
                    "s1": sms.s1, "s1_detail": sms.s1_detail,
                    "s2": sms.s2, "s2_detail": sms.s2_detail,
                    "s3": sms.s3, "s3_detail": sms.s3_detail,
                    "s4": sms.s4, "s4_detail": sms.s4_detail,
                    "s5": sms.s5, "s5_detail": sms.s5_detail,
                    "detail": format_sms_detail(sms),
                }
            else:
                rec["sms"] = {"error": "no takas snapshot"}
        except Exception as e:
            rec["sms"] = {"error": repr(e)}

        try:
            ice = calc_ice(t, takas_history, takas_data, mkk_map)
            if ice:
                labels_out = {k: {"value": lbl.value, "detail": lbl.detail}
                              for k, lbl in ice.labels.items()}
                rec["ice"] = {
                    "multiplier": ice.multiplier,
                    "score": ice.score, "label": ice.label, "icon": ice.icon,
                    "labels": labels_out,
                    "metrics": ice.metrics,
                    "detail": format_ice_detail(ice),
                }
            else:
                rec["ice"] = {"error": "no history"}
        except Exception as e:
            rec["ice"] = {"error": repr(e)}

        results.append(rec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "asof": asof,
            "tickers": tickers,
            "hist_days_requested": hist_days,
            "history_actual_days": len(takas_history),
            "failures": failures,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[sms-ice] wrote {out_json}")

    md = [
        f"# SMS + ICE — {asof}",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        "",
        f"**Tickers:** {', '.join(tickers)}",
        f"**History window requested:** {hist_days}gün  ·  **fetched:** {len(takas_history)}gün",
        f"**MKK:** {'OK ('+str(len(mkk_map))+' tickers)' if mkk_map else 'unavailable'}",
        f"**Failures:** {len(failures)}",
        "",
        "## Özet tablo",
        "",
        "| ticker | SMS skor | SMS label | ICE × | ICE label | S1 | S2 | S3 | S4 | S5 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        sms = r.get("sms", {})
        ice = r.get("ice", {})
        sms_score = sms.get("score", "—")
        sms_lab = sms.get("label", sms.get("error", "—"))
        ice_mul = f"{ice.get('multiplier'):.2f}" if isinstance(ice.get("multiplier"), (int, float)) else "—"
        ice_lab = ice.get("label", ice.get("error", "—"))
        md.append(
            f"| **{r['ticker']}** | {sms_score} | {sms_lab} | {ice_mul} | {ice_lab} | "
            f"{sms.get('s1','—')} | {sms.get('s2','—')} | {sms.get('s3','—')} | "
            f"{sms.get('s4','—')} | {sms.get('s5','—')} |"
        )

    md.append("\n## Ticker detay\n")
    for r in results:
        md.append(f"### {r['ticker']}")
        sms = r.get("sms", {})
        ice = r.get("ice", {})
        if "error" not in sms:
            md.append(f"**SMS:** {sms.get('icon','')} {sms.get('score')} ({sms.get('label')})")
            md.append(f"  - S1 birikim: **{sms.get('s1')}** — {sms.get('s1_detail','')}")
            md.append(f"  - S2 yoğunlaşma: **{sms.get('s2')}** — {sms.get('s2_detail','')}")
            md.append(f"  - S3 karşı taraf: **{sms.get('s3')}** — {sms.get('s3_detail','')}")
            md.append(f"  - S4 süreklilik: **{sms.get('s4')}** — {sms.get('s4_detail','')}")
            md.append(f"  - S5 MKK: **{sms.get('s5')}** — {sms.get('s5_detail','')}")
        else:
            md.append(f"**SMS:** error — {sms.get('error')}")

        if "error" not in ice:
            md.append(f"\n**ICE:** {ice.get('icon','')} ×{ice.get('multiplier'):.2f} ({ice.get('label')})  s100={ice.get('score')}")
            for k, lbl in ice.get("labels", {}).items():
                md.append(f"  - {k}: **{lbl.get('value')}** — {lbl.get('detail','')}")
        else:
            md.append(f"\n**ICE:** error — {ice.get('error')}")
        md.append("")

    if failures:
        md.append("## Failures")
        for t, err in failures:
            md.append(f"- **{t}**: {err}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[sms-ice] wrote {out_md}")

    print("\n=== ÖZET ===")
    for r in results:
        sms = r.get("sms", {})
        ice = r.get("ice", {})
        sms_str = (f"SMS={sms.get('score','?')} ({sms.get('label','?')})"
                   if "error" not in sms else f"SMS=err({sms['error']})")
        ice_mul = ice.get("multiplier")
        ice_str = (f"ICE×{ice_mul:.2f} ({ice.get('label','?')})"
                   if isinstance(ice_mul, (int, float)) else "ICE=err")
        print(f"  {r['ticker']:>6s}  {sms_str:<35s}  {ice_str}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
