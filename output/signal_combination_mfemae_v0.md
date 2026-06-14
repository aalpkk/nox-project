# Sinyal-Kombinasyon × MFE+MAE Fırsat Profili v0

- havuz: 108,155 benzersiz olay, 2024-01-01+ | ufuk 20g
- HAVUZ BASELINE: MFE_med +0.109 | MAE_med -0.076 | temiz-koşucu %21.6
- 'temiz koşucu' = MFE≥20% ∧ MAE≥-8% (capture YOK, fırsat-tavan)
- ⚠️ replay'e giren: nox_v3/smart_breakout (audit-geçti); regime/alpha forward-log'da (henüz YOK)

## En yüksek temiz-koşucu oranlı kombinasyonlar (min-n=40)
| k | kombinasyon | n | MFE_med | MAE_med | temiz% | baseline-lift |
|---|---|---|---|---|---|---|
| 2 | trident_tier1 + smart_breakout | 78 | +0.157 | -0.110 | 33.3 | +11.7 |
| 3 | mb_scanner + trident_tier1 + smart_breakout | 78 | +0.157 | -0.110 | 33.3 | +11.7 |
| 2 | triangle_break + trident_tier1 | 98 | +0.153 | -0.102 | 30.6 | +9.0 |
| 3 | mb_scanner + triangle_break + trident_tier1 | 98 | +0.153 | -0.102 | 30.6 | +9.0 |
| 2 | mb_scanner + trident_tier1 | 381 | +0.142 | -0.106 | 28.3 | +6.7 |
| 1 | trident_tier1 | 381 | +0.142 | -0.106 | 28.3 | +6.7 |
| 2 | mb_scanner + cluster3 | 72 | +0.184 | -0.178 | 26.4 | +4.8 |
| 1 | cluster3 | 81 | +0.182 | -0.167 | 25.9 | +4.3 |
| 3 | mb_scanner + triangle_break + smart_breakout | 814 | +0.125 | -0.087 | 24.6 | +3.0 |
| 3 | mb_scanner + horizontal_base + nox_v3_weekly | 136 | +0.105 | -0.077 | 24.3 | +2.7 |
| 2 | horizontal_base + nox_v3_weekly | 274 | +0.104 | -0.075 | 23.7 | +2.1 |
| 3 | horizontal_base + nox_v3_weekly + smart_breakout | 161 | +0.085 | -0.078 | 23.6 | +2.0 |
| 2 | mb_scanner + triangle_break | 5673 | +0.118 | -0.086 | 23.6 | +2.0 |
| 1 | triangle_break | 7292 | +0.118 | -0.087 | 23.2 | +1.6 |
| 2 | triangle_break + hw_al_os | 69 | +0.105 | -0.072 | 23.2 | +1.6 |
| 2 | triangle_break + smart_breakout | 1045 | +0.121 | -0.092 | 22.8 | +1.2 |
| 3 | mb_scanner + triangle_break + nox_v3_daily | 248 | +0.116 | -0.093 | 22.6 | +1.0 |
| 3 | mb_scanner + nox_v3_daily + smart_breakout | 500 | +0.100 | -0.094 | 22.6 | +1.0 |
| 2 | mb_scanner + smart_breakout | 9582 | +0.113 | -0.080 | 22.5 | +0.9 |
| 2 | nox_v3_weekly + smart_breakout | 587 | +0.104 | -0.097 | 22.3 | +0.7 |
| 3 | mb_scanner + nox_v3_weekly + smart_breakout | 486 | +0.106 | -0.099 | 22.2 | +0.6 |
| 1 | smart_breakout | 11875 | +0.111 | -0.081 | 22.1 | +0.5 |
| 3 | mb_scanner + horizontal_base + triangle_break | 586 | +0.126 | -0.089 | 21.8 | +0.2 |
| 3 | horizontal_base + triangle_break + nox_v3_daily | 60 | +0.103 | -0.090 | 21.7 | +0.1 |
| 1 | mb_scanner | 102290 | +0.110 | -0.076 | 21.7 | +0.1 |
| 2 | horizontal_base + triangle_break | 690 | +0.123 | -0.089 | 21.4 | -0.2 |
| 2 | mb_scanner + nox_v3_daily | 3910 | +0.108 | -0.086 | 21.1 | -0.5 |
| 2 | mb_scanner + horizontal_base | 4371 | +0.107 | -0.077 | 20.7 | -0.9 |
| 1 | nox_v3_daily | 4200 | +0.107 | -0.088 | 20.5 | -1.1 |
| 1 | horizontal_base | 8721 | +0.106 | -0.077 | 20.4 | -1.2 |

## Tekil sinyaller (referans)
| sinyal | n | MFE_med | MAE_med | temiz% | lift |
|---|---|---|---|---|---|
| trident_tier1 | 381 | +0.142 | -0.106 | 28.3 | +6.7 |
| cluster3 | 81 | +0.182 | -0.167 | 25.9 | +4.3 |
| triangle_break | 7292 | +0.118 | -0.087 | 23.2 | +1.6 |
| smart_breakout | 11875 | +0.111 | -0.081 | 22.1 | +0.5 |
| mb_scanner | 102290 | +0.110 | -0.076 | 21.7 | +0.1 |
| nox_v3_daily | 4200 | +0.107 | -0.088 | 20.5 | -1.1 |
| horizontal_base | 8721 | +0.106 | -0.077 | 20.4 | -1.2 |
| nox_v3_weekly | 3021 | +0.105 | -0.090 | 20.0 | -1.6 |
| hw_al_os | 1488 | +0.102 | -0.075 | 18.5 | -3.1 |