# Advisor Aday-Seçim Backtest v0 (çekirdek edge, replay'siz)

- havuz: 236,707 event, 2024-01-01+ | ufuk 20g
- GOOD=49064 FAILED=90868 NEUTRAL=96775
- 3 lens medyan: MFE20 +0.111 (fırsat-tavan) | V1-capture +0.038 (GERÇEKÇİ çıkış) | d20-hold +0.003 (strawman) | MAE20 -0.077

## Feature separability (tüm dönem, GOOD-vs-FAILED within-pool)
| feature | n | GOOD-lift(hi−lo) | null_p95 | p | geçer |
|---|---|---|---|---|---|
| n_concurrent_sources | 139932 | +0.0103 | 0.0051 | 0.000 | ✅ |
| n_concurrent_families | 139932 | +0.0179 | 0.0043 | 0.000 | ✅ |
| n_concurrent_timeframes | 139932 | +0.0222 | 0.0102 | 0.000 | ✅ |
| event_multiplicity | 139932 | +0.0225 | 0.0043 | 0.000 | ✅ |
| c3_member | 139932 | -0.1362 | 0.0578 | 1.000 | ❌ |
| atr_pct | 139903 | -0.0345 | 0.0045 | 1.000 | ❌ |
| vol_z_20 | 139855 | -0.0018 | 0.0042 | 0.742 | ❌ |
| price_vs_20d_high | 139855 | +0.0340 | 0.0042 | 0.000 | ✅ |
| price_vs_60d_high | 138905 | -0.0196 | 0.0044 | 1.000 | ❌ |
| ret_5d | 139932 | -0.0099 | 0.0041 | 1.000 | ❌ |

## Era-OOS işaret tutarlılığı (geçen feature'lar)
| feature | 2024 | 2025 | 2026 | tutarlı |
|---|---|---|---|---|
| n_concurrent_sources | +0.005 | +0.016 | +0.004 | ✅ |
| n_concurrent_families | +0.008 | +0.019 | +0.035 | ✅ |
| n_concurrent_timeframes | -0.012 | +0.040 | +0.045 | ❌ |
| event_multiplicity | +0.026 | +0.014 | +0.031 | ✅ |
| price_vs_20d_high | +0.049 | +0.023 | +0.025 | ✅ |

## VERDICT: PROCEED_WEIGHTS
OOS-doğrulanmış seçim feature'ları: ['n_concurrent_sources', 'n_concurrent_families', 'event_multiplicity', 'price_vs_20d_high']
Normalize ağırlık: {'n_concurrent_sources': 0.1216, 'n_concurrent_families': 0.2113, 'event_multiplicity': 0.2656, 'price_vs_20d_high': 0.4014}

## Capture reality-check (V1-exit gerçekçi çıkış)
- GOOD V1-capture medyan +0.060 | FAILED -0.100 | gap +0.160

### TÜM feature'lar — çeyrek Q4−Q1 (MFE-tavan vs V1-capture: fantezi tespiti)
| feature | MFE Q4−Q1 | V1cap Q4−Q1 | d20 Q4−Q1 | V1-pozitif |
|---|---|---|---|---|
| n_concurrent_sources | -0.003 | +0.001 | -0.010 | ✅ |
| n_concurrent_families | +0.006 | +0.002 | -0.005 | ✅ |
| n_concurrent_timeframes | +0.005 | +0.001 | -0.004 | ✅ |
| event_multiplicity | +0.005 | +0.002 | -0.003 | ✅ |
| c3_member | -0.004 | +0.000 | -0.007 | ✅ |
| atr_pct | +0.050 | +0.003 | -0.015 | ✅ |
| vol_z_20 | +0.009 | +0.002 | -0.006 | ✅ |
| price_vs_20d_high | +0.011 | +0.003 | +0.016 | ✅ |
| price_vs_60d_high | -0.007 | +0.000 | -0.001 | ✅ |
| ret_5d | +0.025 | +0.004 | -0.006 | ✅ |