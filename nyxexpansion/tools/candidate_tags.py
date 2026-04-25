"""
Daily candidate execution metadata tagger (WinMag v4C).

Her v4C adayına execution-quality tag'leri bas — kullanıcı için "ne ile
nasıl davranılmalı" bilgisi explicit olsun. Mantık:

  clean_watch       — stretch/extension/momentum hepsi moderate.
                      Standard watchlist. Ertesi gün retest/ORB ile
                      hedef fiyat mantıklı.

  extended_watch    — stretch/extension high ama aşırı değil.
                      Tradeable ama wider stop / pullback bekle.

  special_handling  — stretch very_high OR parabolic momentum OR
                      above_52w + dense new highs.
                      Günlük OHLCV ile auto-entry anlamsız; intraday
                      execution katmanı gerekli. (red flag DEĞİL; auto-
                      buy'a da hazır DEĞİL — "needs intraday handling".)

Eşikler dataset quantile'larından çıktı (v3 dataset, 18132 rows):
  dist_above_trigger_atr: p50=0.86, p75=1.43, p95=2.25
  trend_extension_ema50_atr: p75=4.82, p95=7.02
  rs_10: p75=0.23, p95=0.50
  new_high_count_20d: p75=6, p95=10
  rvol_today: p75=2.98, p95=4.74
  upside_room_52w_atr: p50=2.22, p95=27
"""
from __future__ import annotations

import pandas as pd


# ── Eşikler (v3 dataset quantile'larından, 2026-04-21) ──────────────────────
STRETCH_MODERATE = 0.86   # p50
STRETCH_HIGH = 1.43       # p75
STRETCH_VERY_HIGH = 2.25  # p95

EXT_EXTENDED = 4.82       # ema50 ext p75
EXT_VERY_EXT = 7.02       # ema50 ext p95

RS10_STRONG = 0.23        # p75
RS10_PARABOLIC = 0.50     # p95 — "hot" eşiği

NEWHIGH_DENSE = 6         # p75
NEWHIGH_VERY_DENSE = 10   # p95

RVOL_SURGE = 4.74         # p95

ROOM_ABOVE_52W = 0.0      # negatif = 52w üstünde
ROOM_TIGHT = 2.22         # p50


def _stretch_rating(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < STRETCH_MODERATE: return 'low'
    if v < STRETCH_HIGH: return 'moderate'
    if v < STRETCH_VERY_HIGH: return 'high'
    return 'very_high'


def _extension_rating(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < EXT_EXTENDED: return 'normal'
    if v < EXT_VERY_EXT: return 'extended'
    return 'very_extended'


def _momentum_intensity(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < RS10_STRONG: return 'mild'
    if v < RS10_PARABOLIC: return 'strong'
    return 'parabolic'


def _new_high_density(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < NEWHIGH_DENSE: return 'fresh'
    if v < NEWHIGH_VERY_DENSE: return 'dense'
    return 'very_dense'


def _volume_rating(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < RVOL_SURGE: return 'normal'
    return 'surge'


def _upside_room(v: float) -> str:
    if pd.isna(v): return 'unknown'
    if v < ROOM_ABOVE_52W: return 'above_52w'
    if v < ROOM_TIGHT: return 'tight'
    return 'ample'


def _overall_tag(
    stretch: str, ext: str, mom: str, dens: str, room: str,
) -> str:
    """Tag hiyerarşisi: special_handling > extended_watch > clean_watch."""
    # special_handling — auto-entry için anlamsız; intraday katmanı gerekli
    if stretch == 'very_high':
        return 'special_handling'
    if ext == 'very_extended':
        return 'special_handling'
    if mom == 'parabolic':
        return 'special_handling'
    if dens == 'very_dense' and room == 'above_52w':
        return 'special_handling'
    # extended_watch — tradeable ama pullback gerekli
    if stretch == 'high':
        return 'extended_watch'
    if ext == 'extended':
        return 'extended_watch'
    if mom == 'strong' and room == 'tight':
        return 'extended_watch'
    # clean_watch — moderate her şey, standard watchlist
    return 'clean_watch'


def tag_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """df'e execution metadata kolonları ekler. In-place değil."""
    out = df.copy()
    out['stretch_rating'] = out.get('dist_above_trigger_atr', pd.Series(index=out.index)).apply(_stretch_rating)
    out['extension_rating'] = out.get('trend_extension_ema50_atr', pd.Series(index=out.index)).apply(_extension_rating)
    out['momentum_intensity'] = out.get('rs_10', pd.Series(index=out.index)).apply(_momentum_intensity)
    out['new_high_density'] = out.get('new_high_count_20d', pd.Series(index=out.index)).apply(_new_high_density)
    out['volume_rating'] = out.get('rvol_today', pd.Series(index=out.index)).apply(_volume_rating)
    out['upside_room'] = out.get('upside_room_52w_atr', pd.Series(index=out.index)).apply(_upside_room)

    tags = []
    for i in out.index:
        tags.append(_overall_tag(
            out.at[i, 'stretch_rating'],
            out.at[i, 'extension_rating'],
            out.at[i, 'momentum_intensity'],
            out.at[i, 'new_high_density'],
            out.at[i, 'upside_room'],
        ))
    out['exec_tag'] = tags
    return out


def print_tag_summary(df: pd.DataFrame) -> None:
    """Tagged df için tag dağılımı ve per-tag detay."""
    if 'exec_tag' not in df.columns:
        print("[!] exec_tag missing — run tag_candidates first")
        return
    print("\n─── EXEC TAG DAĞILIMI ──────────────────────────────")
    vc = df['exec_tag'].value_counts()
    for t, n in vc.items():
        print(f"  {t:<20}  N={n:>3}  ({n/len(df)*100:5.1f}%)")

    print("\n─── TAG x KRITER DAĞILIM ───────────────────────────")
    for col in ['stretch_rating', 'extension_rating', 'momentum_intensity',
                'new_high_density', 'volume_rating', 'upside_room']:
        if col in df.columns:
            vc = df[col].value_counts()
            print(f"  {col:<22} {dict(vc)}")
