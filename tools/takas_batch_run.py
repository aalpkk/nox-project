"""
Takas Batch GENERIC — ticker listesi parametreli broker-distribution (AKD) + ICE.

takas_batch_2026_05_1*_set*.py / takas_batch_ipo_rbr_2026_06_11.py kalıbının
tek-dosya jenerik hali: ticker'lar CLI/env'den gelir, set dosyası çoğaltılmaz.

Run (lokal):
  set -a; source .env; set +a
  PYTHONPATH=. python tools/takas_batch_run.py --tickers "GENKM BESTE NETCD" --slug rbr_full
Run (CI):
  TICKERS="..." SLUG=... PYTHONPATH=. python tools/takas_batch_run.py

Outputs:
  output/takas_batch_{ASOF}_{SLUG}.csv      — broker-level G/H/A/3A per ticker
  output/takas_batch_{ASOF}_{SLUG}_ice.csv  — per-ticker per-day ICE
  output/takas_batch_{ASOF}_{SLUG}.md       — readable summary

Requires: MATRIKS_API_KEY, MATRIKS_CLIENT_ID
"""
import argparse
import csv
import os
import sys
from datetime import datetime, date

from agent.matriks_client import MatriksClient
from agent.matriks_adapter import flows_to_takas_data, flow_to_takas_history_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=os.environ.get("TICKERS", ""),
                    help="Boşluk/virgül ayrımlı ticker listesi")
    ap.add_argument("--slug", default=os.environ.get("SLUG", "adhoc"))
    ap.add_argument("--hist-days", type=int,
                    default=int(os.environ.get("HIST_DAYS", "10")))
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.replace(",", " ").split() if t.strip()]
    if not tickers:
        print("ERROR: --tickers boş", file=sys.stderr)
        sys.exit(2)
    if not os.environ.get("MATRIKS_API_KEY"):
        print("ERROR: MATRIKS_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    asof = date.today().isoformat()
    out_broker = f"output/takas_batch_{asof}_{args.slug}.csv"
    out_ice = f"output/takas_batch_{asof}_{args.slug}_ice.csv"
    out_md = f"output/takas_batch_{asof}_{args.slug}.md"

    os.makedirs("output", exist_ok=True)
    client = MatriksClient()

    broker_rows, ice_rows, md_blocks, failures = [], [], [], []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker} ...")
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

            hist = client.get_daily_flow_history(ticker, days=args.hist_days, top=10)
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

            top_buyers = [k for k in kurumlar if k.get("Günlük Fark", 0) > 0][:5]
            top_sellers = [k for k in sorted(kurumlar, key=lambda x: x.get("Günlük Fark", 0))
                           if k.get("Günlük Fark", 0) < 0][:5]
            block = [f"### {ticker}"]
            for title, rows in (("**Top alıcılar (G):**", top_buyers),
                                ("**Top satıcılar (G):**", top_sellers)):
                if rows:
                    block.append(title)
                    for k in rows:
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

    if broker_rows:
        with open(out_broker, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(broker_rows[0].keys()))
            w.writeheader()
            w.writerows(broker_rows)
        print(f"  ✓ {out_broker}  ({len(broker_rows)} broker rows)")
    if ice_rows:
        with open(out_ice, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ice_rows[0].keys()))
            w.writeheader()
            w.writerows(ice_rows)
        print(f"  ✓ {out_ice}  ({len(ice_rows)} ICE day-rows)")

    md = [
        f"# Takas Batch {args.slug} — {asof}",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        "",
        f"**Tickers ({len(tickers)}):** {', '.join(tickers)}",
        f"**History days:** {args.hist_days}",
        f"**Failures:** {len(failures)}",
        "",
        "## Per-ticker broker flow (G=daily / H=weekly / A=monthly / 3A=quarterly)",
        "",
    ]
    md.extend(md_blocks)
    if failures:
        md.append("\n## Failures")
        for t, err in failures:
            md.append(f"- **{t}**: {err}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  ✓ {out_md}")

    print(f"\nDONE — {len(tickers) - len(failures)}/{len(tickers)} tickers OK")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
