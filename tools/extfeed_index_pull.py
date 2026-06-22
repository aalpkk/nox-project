"""
Endeks extfeed pull — BIST endekslerini (XU100 + 27 sektör) TradingView WS'den çek.

Hisse extfeed master'ı (output/extfeed_intraday_1h_3y_master.parquet) SADECE
hisse içerir; endeksler orada YOK. Bu script onların İZOLE eşdeğerini üretir:
  output/extfeed_index_1h_master.parquet  (ticker, ts_utc, ts_istanbul, OHLCV; 1h)
Aynı TradingView kaynağı (BIST:XU100 ...), aynı şema → nyx rotasyon tek-kaynak.
Ana hisse pipeline'ına / DE v1 / scanner'lara DOKUNMAZ (B planı, izole).

Auth: INTRADAY_SID/SIGN/HOST/WS_URL (markets.extfeed). Lokalde .env, CI'da secret.

Kullanım:
    set -a; source .env; set +a
    python tools/extfeed_index_pull.py --backfill        # 3 yıl full
    python tools/extfeed_index_pull.py                   # delta (son ~7 gün, mevcut master'a yapıştır)
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from agent.sector_regime import _ALL_INDICES
from markets.extfeed.ws_client import fetch_bars_until

OUT = os.path.join(ROOT, 'output', 'extfeed_index_1h_master.parquet')
TIMEFRAME = '60'
# XU100 + tüm sektör endeksleri (anatomi/rotasyon için gerekenler)
INDICES = ['XU100'] + list(_ALL_INDICES['sektor'])


def _fetch(code, until_utc, max_chunks):
    df, meta = fetch_bars_until(f"BIST:{code}", TIMEFRAME, until_utc,
                                chunk_n=5000, max_chunks=max_chunks, chunk_timeout_s=30)
    if df is None or df.empty:
        return None
    df = df.rename(columns={'time': 'ts_istanbul'})
    df['ts_istanbul'] = pd.to_datetime(df['ts_istanbul'])
    df['ts_utc'] = df['ts_istanbul'].dt.tz_convert('UTC')
    df['ticker'] = code
    return df[['ticker', 'ts_utc', 'ts_istanbul', 'open', 'high', 'low', 'close', 'volume']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill', action='store_true', help='3 yıl full (yoksa son ~7 gün delta)')
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--delta-days', type=int, default=7)
    args = ap.parse_args()

    if not os.environ.get('INTRADAY_SID'):
        raise SystemExit("🔴 INTRADAY_SID yok (set -a; source .env; set +a — ya da CI secret)")

    if args.backfill:
        until = pd.Timestamp(datetime.utcnow() - timedelta(days=365 * args.years)).tz_localize('UTC')
        max_chunks = 60
        mode = f'BACKFILL {args.years}y'
    else:
        until = pd.Timestamp(datetime.utcnow() - timedelta(days=args.delta_days)).tz_localize('UTC')
        max_chunks = 3
        mode = f'DELTA son {args.delta_days}g'

    print(f"Endeks pull [{mode}] — {len(INDICES)} endeks, until {until.date()}")
    frames, fail = [], []
    for i, code in enumerate(INDICES, 1):
        try:
            df = _fetch(code, until, max_chunks)
            if df is None:
                fail.append(code); print(f"  ⚠️ {i:2d}/{len(INDICES)} {code}: veri yok"); continue
            frames.append(df)
            print(f"  ✓ {i:2d}/{len(INDICES)} {code}: {len(df)} bar  {df['ts_istanbul'].min().date()}→{df['ts_istanbul'].max().date()}")
        except Exception as e:  # noqa: BLE001
            fail.append(code); print(f"  ⚠️ {i:2d}/{len(INDICES)} {code}: {type(e).__name__}: {e}")
    if fail:
        print(f"  ⚠️ çekilemeyen: {', '.join(fail)}")
    if not frames:
        raise SystemExit("🔴 hiç endeks çekilemedi")
    new = pd.concat(frames, ignore_index=True)

    if not args.backfill and os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        new = pd.concat([old, new], ignore_index=True)
    master = (new.drop_duplicates(['ticker', 'ts_utc'], keep='last')
                 .sort_values(['ticker', 'ts_utc']).reset_index(drop=True))
    master.to_parquet(OUT, index=False)
    print(f"\n✓ {OUT}: {len(master)} satır, {master['ticker'].nunique()} endeks, "
          f"{master['ts_istanbul'].min().date()} → {master['ts_istanbul'].max().date()}")


if __name__ == '__main__':
    main()
