"""
Takas Batch — 17 tickers ICE-like broker-distribution analysis for 2026-05-14.

Outputs:
  output/takas_batch_2026-05-14.csv      — broker-level G/H/A/3A net quantities per ticker
  output/takas_batch_2026-05-14_ice.csv  — per-ticker per-day SM/Yerli net + top3 concentration
  output/takas_batch_2026-05-14.md       — human-readable summary

Run:
  PYTHONPATH=. python tools/takas_batch_2026_05_14.py

Requires:
  MATRIKS_API_KEY, MATRIKS_CLIENT_ID  (GitHub Actions secrets)
"""
import csv
import os
import sys
from datetime import datetime

from agent.matriks_client import MatriksClient
from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day


ASOF = "2026-05-14"
TICKERS = [
    "BORSK", "TARKM", "RGYAS", "TATGD", "RTALB", "KARTN", "ETYAT",
    "ALKLC", "MOPAS", "KLYPV", "ECOGR", "BESTE", "AKHAN", "BINBN",
    "KOCMT", "HRKET", "FRMPL",
]
HIST_DAYS = 5  # last 5 trading days for ICE history

OUT_BROKER_CSV = f"output/takas_batch_{ASOF}.csv"
OUT_ICE_CSV = f"output/takas_batch_{ASOF}_ice.csv"
OUT_MD = f"output/takas_batch_{ASOF}.md"


def main():
    if not os.environ.get("MATRIKS_API_KEY"):
        print("ERROR: MATRIKS_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    os.makedirs("output", exist_ok=True)
    client = MatriksClient()

    broker_rows = []  # ticker, broker, daily, weekly, monthly, quarterly, pct, position
    ice_rows = []     # ticker, date, net_yab_banka, net_fon, net_yerli, top3_pct, sm_buy_qty, sm_buy_vol, sm_sell_qty, sm_sell_vol
    md_blocks = []
    failures = []

    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}] {ticker} ...")
        try:
            flows = client.get_institutional_flow_periods(ticker, top=10)
            if not flows:
                failures.append((ticker, "no_periods"))
                continue
            takas = flows_to_takas_data(flows, ticker)
            kurumlar = takas.get(ticker, {}).get("kurumlar", [])
            for k in kurumlar:
                broker_rows.append({
                    "ticker": ticker,
                    "broker": k.get("Aracı Kurum", ""),
                    "daily_qty": k.get("Günlük Fark", 0),
                    "weekly_qty": k.get("Haftalık Fark", 0),
                    "monthly_qty": k.get("Aylık Fark", 0),
                    "quarterly_qty": k.get("3 Aylık Fark", 0),
                    "net_pct": k.get("%", 0),
                    "position": k.get("Pozisyon", 0),
                })

            # ICE history (last N days)
            hist = client.get_daily_flow_history(ticker, days=HIST_DAYS, top=10)
            for date_str, flow_resp in sorted(hist.items()):
                day = flow_to_takas_history_day(flow_resp, ticker, date_str)
                tdata = day.get(date_str, {}).get(ticker, {})
                net_tip = tdata.get("net_tip", {})
                ice_rows.append({
                    "ticker": ticker,
                    "date": date_str,
                    "net_yab_banka": net_tip.get("yab_banka", 0),
                    "net_fon": net_tip.get("fon", 0),
                    "net_yerli": net_tip.get("yerli", 0),
                    "net_diger": net_tip.get("diger", 0),
                    "top3_alici_pct": tdata.get("top3_alici_pct", 0),
                    "sm_buy_qty": tdata.get("sm_buy_qty", 0),
                    "sm_buy_vol": tdata.get("sm_buy_vol", 0),
                    "sm_sell_qty": tdata.get("sm_sell_qty", 0),
                    "sm_sell_vol": tdata.get("sm_sell_vol", 0),
                })

            # Markdown block per ticker
            top_buyers = [k for k in kurumlar if k.get("Günlük Fark", 0) > 0][:5]
            top_sellers = [k for k in sorted(kurumlar, key=lambda x: x.get("Günlük Fark", 0))
                           if k.get("Günlük Fark", 0) < 0][:5]
            block = [f"### {ticker}"]
            if top_buyers:
                block.append("**Top alıcılar (G):**")
                for k in top_buyers:
                    block.append(
                        f"  - {k.get('Aracı Kurum','?'):20s} "
                        f"G={k.get('Günlük Fark',0):>+10,}  "
                        f"H={k.get('Haftalık Fark',0):>+12,}  "
                        f"A={k.get('Aylık Fark',0):>+14,}  "
                        f"3A={k.get('3 Aylık Fark',0):>+14,}"
                    )
            if top_sellers:
                block.append("**Top satıcılar (G):**")
                for k in top_sellers:
                    block.append(
                        f"  - {k.get('Aracı Kurum','?'):20s} "
                        f"G={k.get('Günlük Fark',0):>+10,}  "
                        f"H={k.get('Haftalık Fark',0):>+12,}  "
                        f"A={k.get('Aylık Fark',0):>+14,}  "
                        f"3A={k.get('3 Aylık Fark',0):>+14,}"
                    )
            md_blocks.append("\n".join(block))
        except Exception as e:
            failures.append((ticker, repr(e)))
            print(f"  ⚠️ {ticker} failed: {e}")

    # Write CSVs
    if broker_rows:
        with open(OUT_BROKER_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(broker_rows[0].keys()))
            w.writeheader()
            w.writerows(broker_rows)
        print(f"  ✓ {OUT_BROKER_CSV}  ({len(broker_rows)} broker rows)")

    if ice_rows:
        with open(OUT_ICE_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ice_rows[0].keys()))
            w.writeheader()
            w.writerows(ice_rows)
        print(f"  ✓ {OUT_ICE_CSV}  ({len(ice_rows)} ICE day-rows)")

    # Write markdown
    md = [
        f"# Takas Batch — {ASOF}",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        f"",
        f"**Tickers ({len(TICKERS)}):** {', '.join(TICKERS)}",
        f"**History days:** {HIST_DAYS}",
        f"**Failures:** {len(failures)}",
        f"",
        "## Per-ticker broker flow (G=daily / H=weekly / A=monthly / 3A=quarterly)",
        "",
    ]
    md.extend(md_blocks)
    if failures:
        md.append("\n## Failures")
        for t, err in failures:
            md.append(f"- **{t}**: {err}")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  ✓ {OUT_MD}")

    print(f"\nDONE — {len(TICKERS) - len(failures)}/{len(TICKERS)} tickers OK")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
