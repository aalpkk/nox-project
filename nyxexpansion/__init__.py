"""
nyxexpansion — WinMag / Momentum-Continuation Ranker (PIPELINE A).

Rol (2026-04-21 itibariyle cementli):
  v4C = daily candidate ranker, NOT auto-entry strategy.

Değil:
  - "yarın açılışta otomatik alım modeli"  ❌
  - fresh breakout modeli  (ayrı pipeline: `ignito/`)

Evet:
  - "yarın yakından izlenecek güçlü aday listesi"  ✓
  - candidate scorer / watchlist
  - momentum-continuation adayları içinde büyük winner potansiyelini sıralama

Kritik bulgu (execution haircut):
  Close-entry PF 3.51 → next-open PF 1.23 (-65%). Portföy sim'de mc=10 PF
  2.64 → 1.19, haircut her configte uniform ~55%. Günlük OHLCV ile
  "executable entry" illüzyonuna düşme.

Sezgiye aykırı nuance:
  Gap/stretch hard filter çalışmıyor. "gap > 5% ise alma" gibi mekanik
  veto edge'i öldürüyor. AMA bu "gap > 8%'i auto-buy yap" demek DEĞİL —
  uç buckets tail-driven, küçük N. Büyük gap adaylar = *special handling
  needed* (red flag değil, green light da değil).

Ana eksik parça (scope dışı bu oturumda):
  Entry/execution engine (intraday 5dk/15dk bar): retest, opening range
  break, gap sonrası hold/fail, limit-order fill. Günlük barla simüle
  edilemez.

Portfolio dual view (artık mecburi):
  - SELECTION QUALITY = close-entry metrikleri (ranker ne kadar iyi)
  - EXECUTION-ADJUSTED REALIZED = next-open metrikleri (günlük OHLCV
    ile realistik tavan)
  Live expectation için ikincisi kullanılmalı.

Trigger + target:
  - Trigger A (v1): classic 20-day breakout with volume + close-location
  - Target primary: winner_R (MFE_h/ATR, range veto + path-dependent)
  - Secondary: L3 cont_10 (diagnostic)
  - Two-head: clf(cont_10) × reg(winner_R) → v4C variant
  - Features: 48, 8 blok (A-H), sade — oscillator kalabalığı yok

Kardeş pipeline:
  `ignito/` — fresh-breakout, freshness/execution odaklı (ayrı oturumda).
"""
