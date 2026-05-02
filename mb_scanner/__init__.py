"""mb_scanner — strict MSS-validated Mitigation/Breaker Block scanner.

Pipeline B (separate from `scanner/` V1). Detects bullish MB and BB families
on 5h, 1d, 1w, 1M bars under a strict 4-pivot quartet structure:

  LL → LH → HL → HH   (chronological)

  MB variant: HL > LL  (failed-bounce reversal, "double-bottom-with-MSS")
  BB variant: L2 < L1  (sweep below first low, then BoS above first high)

Zone = body of the last bullish (close>open) bar walking back from LH
(inclusive). After HH (= close > LH), the scanner classifies each ticker
into one of four states:

  above_mb         — HH confirmed, zone untouched after HH
  mitigation_touch — bar(s) overlapped zone, no bullish reclaim yet
  retest_bounce    — asof bar is the first bullish reclaim of the zone
  extended         — retest happened earlier; asof is post-bounce

Eight output families: mb_5h, mb_1d, mb_1w, mb_1M, bb_5h, bb_1d, bb_1w,
bb_1M. The 5h frequency mirrors TV's "5 saatlik" timeframe for BIST
equities (09:00 morning + 14:00 afternoon, both clean session-aligned).
"""
from __future__ import annotations

VERSION = "0.1.0"
