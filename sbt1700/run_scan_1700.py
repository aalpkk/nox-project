"""Live SBT-1700 scanner — STUB (PR-3).

Cron target: 14:01 UTC = 17:01 TR, Mon-Fri.

Pipeline (planned):
  1. Pull the day's 15m bars (Matriks) for all BIST tickers.
  2. Aggregate to 17:00-truncated bars (cutoff=16:45 TR).
  3. Detect SBT-1700 signal candidates against today's daily-master
     bars (T-1 anchored prior).
  4. Compute features (sbt1700.features.build_features).
  5. Score with the frozen ranker artifact (sbt1700_ranker.txt).
  6. Emit:
       output/sbt_1700_live.html        (TradingView links + ranker score)
       Telegram top-K message
       GH Pages publish (gated by the PR-2 acceptance gate)

The cron is gated: it MUST NOT publish to Telegram or GH Pages until the
acceptance gate (eval_ranker) is green. This file refuses to run until
sbt1700_ranker.txt and sbt1700_ranker_metrics.json exist with
gate_passed=True.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "sbt1700.run_scan_1700 is a PR-3 stub. "
        "Live scan only enables after the ranker passes the locked "
        "acceptance gate in PR-2."
    )


if __name__ == "__main__":
    raise SystemExit(main())
