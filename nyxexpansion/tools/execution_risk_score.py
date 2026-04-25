"""
Execution risk score — composite tag'in yanında sayısal sıralanabilir metrik.

Amaç: candidate_tags.py 3-bucket tag'i UX için kolay; ama execution-space'de
ayrım zayıf (PF 1.22 / 1.24 / 1.30). Per-feature validation daha keskin:

  upside_room above_52w → PF 0.98  (edge dead)
  stretch moderate      → PF 0.77  (deadzone)
  rs_10 parabolic       → PF 1.19
  upside_room ample     → PF 1.46  (en temiz)

Skor bu per-feature sinyalleri ağırlıklı olarak bir scalar'a sıkıştırır.
Yüksek skor = kötü execution profili (edge eriyor). Ranking/filter için.

Yalnız v4C top-D picks üstünde kayda değer dağılımı olan feature'lar
kullanılıyor; upper_wick_pct, close_loc_bar, entry_to_stop_atr v4C'nin
kendi seçiminde saturate (wick~0, close~1, stop hep geniş) — ayrım yok.
"""
from __future__ import annotations

import pandas as pd


# ── Ağırlıklar — per-feature PF findings'e kalibrelendi ──────────────────────
# above_52w edge'i tamamen siliyor (PF 0.98) → en büyük penalty
W_ABOVE_52W = 3.0

# tight (0-2.22 ATR room) orta penalty
W_TIGHT_ROOM = 1.0

# ample (>5 ATR room) küçük bonus (temiz profil PF 1.46)
W_AMPLE_ROOM = -0.5
AMPLE_THRESHOLD = 5.0

# stretch deadzone (0.86-1.43, PF 0.77) → ağır penalty
W_STRETCH_DEADZONE = 2.0

# stretch very_high (>=2.25) mild penalty — PF 1.35, çok kötü değil ama ekstrem
W_STRETCH_VERY_HIGH = 0.5

# parabolic rs_10 (>=0.50) orta penalty
W_PARABOLIC = 1.0

# gap exhaustion (20 gün içinde ≥8 gap) crowding proxy
W_GAP_EXHAUSTION = 1.0


def compute_risk_score(row: pd.Series) -> float:
    """Signal-row'a risk skoru. Yüksek = kötü execution profili."""
    score = 0.0

    room = row.get('upside_room_52w_atr')
    if room is not None and not pd.isna(room):
        if room < 0:
            score += W_ABOVE_52W
        elif room < 2.22:
            score += W_TIGHT_ROOM
        elif room >= AMPLE_THRESHOLD:
            score += W_AMPLE_ROOM

    stretch = row.get('dist_above_trigger_atr')
    if stretch is not None and not pd.isna(stretch):
        if 0.86 <= stretch < 1.43:
            score += W_STRETCH_DEADZONE
        elif stretch >= 2.25:
            score += W_STRETCH_VERY_HIGH

    rs = row.get('rs_10')
    if rs is not None and not pd.isna(rs) and rs >= 0.50:
        score += W_PARABOLIC

    gap_cnt = row.get('gap_up_count_20d')
    if gap_cnt is not None and not pd.isna(gap_cnt) and gap_cnt >= 8:
        score += W_GAP_EXHAUSTION

    return float(score)


def risk_bucket(score: float) -> str:
    """Skora göre 4-seviye bucket."""
    if score <= 0.0:
        return 'clean'
    if score < 2.0:
        return 'mild'
    if score < 4.0:
        return 'elevated'
    return 'severe'


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """df'e execution_risk_score + risk_bucket kolonları ekler. In-place değil."""
    out = df.copy()
    out['execution_risk_score'] = out.apply(compute_risk_score, axis=1)
    out['risk_bucket'] = out['execution_risk_score'].apply(risk_bucket)
    return out
