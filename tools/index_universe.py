#!/usr/bin/env python3
"""BIST-100 / BIST-30 evren matrisi — point-in-time (PIT) yeniden kurulum.

NEDEN PROXY: Repoda ve erişilebilir veri kaynaklarında endeks üyeliğinin
TARİHSEL serisi yok; Matriks `stockExplorer(symbol_group=XU100/XU030)` yalnızca
BUGÜNKÜ listeyi verir. Bugünkü listeyi 5 yıllık bir backtest'e uygulamak
survivorship + look-ahead sokar (bugün XU100'de olan hisse 2022'de değildi).

Bu yüzden iki evren üretilir ve her ikisi de raporlanır:

  1. `pit`     — Borsa İstanbul'un endeks gözden geçirme takvimini (çeyreklik:
                 Ocak/Nisan/Temmuz/Ekim) taklit eden, her gözden geçirmede
                 ÖNCEKİ 6 ayın medyan günlük TL işlem hacmine göre sıralanmış
                 top-100 / top-30. Yalnızca geçmiş veri kullanır → PIT temiz.
  2. `current` — Matriks'ten alınan bugünkü gerçek XU100/XU030 listesi, tüm
                 pencereye sabit uygulanır. Survivorship-yanlı; sadece
                 duyarlılık kontrolü olarak raporlanır.

BIST gerçek metodolojisi fiili dolaşımdaki piyasa değeri + işlem hacmi
birleşimidir; elimizde free-float verisi PIT olarak olmadığından hacim tek
kriter. `validate()` proxy'nin bugünkü gerçek listeyle örtüşmesini ölçer.

Kullanım:
    python tools/index_universe.py --build
    python tools/index_universe.py --validate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "output" / "ohlcv_10y_fintables_master.parquet"
OUT_DIR = ROOT / "output" / "index_universe"

# Gözden geçirme başına bakılan geçmiş pencere (işgünü) ve minimum geçmiş.
LOOKBACK_BARS = 126          # ~6 ay
MIN_HISTORY_BARS = 180       # yeni halka arzları eleyen eşik
REVIEW_MONTHS = (1, 4, 7, 10)

SIZES = {"XU100": 100, "XU030": 30}
# Tampon (buffer) kuralı: gerçek endeks metodolojilerindeki gibi, mevcut üye
# sıralaması N*BUFFER'a düşene kadar endekste kalır; yeni giren N içine
# girmelidir. Saf top-N kuralı hacim gürültüsüyle yılda ~%40 devir üretiyordu.
BUFFER = 1.3

# Matriks stockExplorer(symbol_group=...) — 2026-07-29 çekimi. Bu liste
# ZAMANA BAĞLIDIR; yenilemek için tools/index_universe.py --refresh-current
# yerine MCP'den tekrar çekip aşağıyı güncelle.
CURRENT_ASOF = "2026-07-29"

CURRENT_XU100 = [
    "AEFES", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALTNY", "ANSGR", "ARCLK",
    "ASELS", "ASTOR", "BALSU", "BERA", "BIMAS", "BRSAN", "BRYAT", "BSOKE",
    "BTCIM", "CANTE", "CCOLA", "CIMSA", "CVKMD", "CWENE", "DAPGM", "DOAS",
    "DOHOL", "DSTKF", "ECILC", "EFOR", "EKGYO", "ENERY", "ENJSA", "ENKAI",
    "EREGL", "ESEN", "EUPWR", "EUREN", "FENER", "FROTO", "GARAN", "GENIL",
    "GESAN", "GLRMK", "GRSEL", "GRTHO", "GSRAY", "GUBRF", "HALKB", "HEKTS",
    "IEYHO", "ISCTR", "ISMEN", "IZENR", "KCHOL", "KLRHO", "KRDMD", "KTLEV",
    "KUYAS", "MAGEN", "MAVI", "MGROS", "MIATK", "MPARK", "OBAMS", "ODAS",
    "ODINE", "OTKAR", "OYAKC", "PAHOL", "PASEU", "PATEK", "PETKM", "PGSUS",
    "PSGYO", "QUAGR", "RALYH", "REEDR", "SAHOL", "SARKY", "SASA", "SISE",
    "SKBNK", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TRALT",
    "TRENJ", "TRMET", "TSKB", "TTKOM", "TUKAS", "TUPRS", "TURSG", "ULKER",
    "VAKBN", "VESTL", "YKBNK", "ZOREN",
]

CURRENT_XU030 = [
    "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "DSTKF", "EKGYO", "ENKAI",
    "EREGL", "FROTO", "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS",
    "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO",
    "TOASO", "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK",
]

CURRENT = {"XU100": CURRENT_XU100, "XU030": CURRENT_XU030}


# ────────────────────────────── PIT kurulum ──────────────────────────────


def _load_tl_volume() -> pd.DataFrame:
    """(date x ticker) günlük TL işlem hacmi matrisi."""
    raw = pd.read_parquet(OHLCV_PATH, columns=["Close", "Volume", "ticker"])
    raw = raw.reset_index().rename(columns={"index": "Date"})
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw["tl"] = raw["Close"] * raw["Volume"]
    piv = raw.pivot_table(index=date_col, columns="ticker", values="tl", aggfunc="last")
    return piv.sort_index()


def _review_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Her Ocak/Nisan/Temmuz/Ekim'in İLK işlem günü = yeni dönemin ilk günü."""
    df = pd.DataFrame({"d": index})
    df["y"] = df["d"].dt.year
    df["m"] = df["d"].dt.month
    df = df[df["m"].isin(REVIEW_MONTHS)]
    firsts = df.groupby(["y", "m"])["d"].min().sort_values()
    return list(firsts)


def build_membership() -> pd.DataFrame:
    """Uzun formatta üyelik: (date, ticker, in_xu100, in_xu030, adv_rank).

    Bir gözden geçirme gününde sıralama YALNIZCA o günden ÖNCEKİ barlarla
    yapılır (`.iloc[:pos]`), yani gözden geçirme gününün kendisi dahil değil →
    look-ahead yok.
    """
    tl = _load_tl_volume()
    dates = tl.index
    reviews = _review_dates(dates)

    # Her ticker için geçerli (NaN olmayan) bar sayısının kümülatifi → geçmiş kontrolü
    valid_cum = tl.notna().cumsum()

    def _select(ranked: pd.Index, prev: list[str], n: int) -> list[str]:
        """Tamponlu seçim: önceki üyeler rank<=n*BUFFER iken korunur, boşluklar
        sıralamanın tepesinden doldurulur."""
        pos_of = {t: i + 1 for i, t in enumerate(ranked)}
        keep = [t for t in prev if pos_of.get(t, 10**9) <= n * BUFFER]
        keep = sorted(keep, key=lambda t: pos_of[t])[:n]
        out = list(keep)
        seen = set(out)
        for t in ranked:
            if len(out) >= n:
                break
            if t not in seen:
                out.append(t)
                seen.add(t)
        return sorted(out, key=lambda t: pos_of[t])

    periods: list[tuple[pd.Timestamp, pd.Timestamp, list[str], list[str], pd.Series]] = []
    prev100: list[str] = []
    prev30: list[str] = []
    for i, rv in enumerate(reviews):
        pos = dates.get_loc(rv)
        if pos < MIN_HISTORY_BARS:
            continue
        window = tl.iloc[max(0, pos - LOOKBACK_BARS):pos]
        med = window.median(skipna=True)
        hist = valid_cum.iloc[pos - 1]
        med = med[(hist >= MIN_HISTORY_BARS) & med.notna() & (med > 0)]
        ranked = med.sort_values(ascending=False).index
        top100 = _select(ranked, prev100, SIZES["XU100"])
        # BIST30 ⊂ BIST100 — seçim yalnızca o dönemin 100'lüğü içinden yapılır.
        r30 = ranked[ranked.isin(top100)]
        top30 = _select(r30, [t for t in prev30 if t in set(top100)], SIZES["XU030"])
        prev100, prev30 = top100, top30
        end = reviews[i + 1] if i + 1 < len(reviews) else dates[-1] + pd.Timedelta(days=1)
        rank_s = pd.Series(np.arange(1, len(ranked) + 1), index=ranked)
        periods.append((rv, end, top100, top30, rank_s))

    rows = []
    for start, end, top100, top30, rank_s in periods:
        sl = dates[(dates >= start) & (dates < end)]
        s100, s30 = set(top100), set(top30)
        for t in top100:
            r = int(rank_s[t])
            for d in sl:
                rows.append((d, t, True, t in s30, r))
    memb = pd.DataFrame(rows, columns=["date", "ticker", "in_xu100", "in_xu030", "adv_rank"])
    return memb.sort_values(["date", "adv_rank"]).reset_index(drop=True)


# ────────────────────────────── API ──────────────────────────────


def load_membership() -> pd.DataFrame:
    p = OUT_DIR / "membership.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} yok — önce `python tools/index_universe.py --build`")
    return pd.read_parquet(p)


def member_sets(index: str = "XU100", mode: str = "pit") -> dict[pd.Timestamp, set[str]]:
    """{tarih -> o gün endekste olan tickerlar}. mode='current' ise sabit liste."""
    if mode == "current":
        raise ValueError("current modda tarih-bazlı set gereksiz; CURRENT[index] kullan")
    memb = load_membership()
    col = "in_xu100" if index.upper() in ("XU100", "BIST100") else "in_xu030"
    sub = memb[memb[col]]
    return {d: set(g["ticker"]) for d, g in sub.groupby("date")}


def mask_for(df_dates: pd.Series, df_tickers: pd.Series, index: str = "XU100",
             mode: str = "pit") -> np.ndarray:
    """Verilen (date, ticker) çiftleri için boolean üyelik maskesi."""
    if mode == "current":
        keys = set(CURRENT[_norm(index)])
        return df_tickers.isin(keys).to_numpy()
    memb = load_membership()
    col = "in_xu100" if _norm(index) == "XU100" else "in_xu030"
    keys = set(zip(memb.loc[memb[col], "date"], memb.loc[memb[col], "ticker"]))
    pairs = zip(pd.to_datetime(df_dates), df_tickers)
    return np.fromiter((p in keys for p in pairs), dtype=bool, count=len(df_tickers))


def _norm(index: str) -> str:
    u = index.upper().replace("BIST", "").strip()
    if u in ("100", "XU100"):
        return "XU100"
    if u in ("30", "030", "XU030", "XU30"):
        return "XU030"
    # XU100 \ XU030 — endeksin "orta kapitalizasyon" dilimi (BIST-30 hariç 70 isim).
    if u in ("100EX30", "XU100EX30", "100-30", "MID70"):
        return "XU100EX30"
    raise ValueError(f"bilinmeyen endeks: {index}")


# ────────────────────────────── doğrulama ──────────────────────────────


def validate() -> dict:
    """PIT proxy'nin son dönemi ile bugünkü GERÇEK listenin örtüşmesi."""
    memb = load_membership()
    last_date = memb["date"].max()
    last = memb[memb["date"] == last_date]
    out = {"pit_last_period_date": str(last_date.date()),
           "current_asof": CURRENT_ASOF, "indices": {}}

    ohl = pd.read_parquet(OHLCV_PATH, columns=["ticker"])
    covered = set(ohl["ticker"].unique())

    for idx, col in (("XU100", "in_xu100"), ("XU030", "in_xu030")):
        pit = set(last.loc[last[col], "ticker"])
        cur = set(CURRENT[idx])
        cur_cov = cur & covered          # OHLCV master'da fiyatı olanlar
        inter = pit & cur_cov
        out["indices"][idx] = {
            "pit_n": len(pit),
            "current_n": len(cur),
            "current_in_ohlcv": len(cur_cov),
            "overlap": len(inter),
            "overlap_pct_of_current_covered": round(100 * len(inter) / max(1, len(cur_cov)), 1),
            "in_current_not_pit": sorted(cur_cov - pit),
            "in_pit_not_current": sorted(pit - cur_cov),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.build:
        memb = build_membership()
        memb.to_parquet(OUT_DIR / "membership.parquet", index=False)
        n_days = memb["date"].nunique()
        print(f"membership: {len(memb):,} satır · {n_days} seans "
              f"· {memb['date'].min().date()} → {memb['date'].max().date()}")
        print(f"  XU100 ticker havuzu (tüm dönemler): {memb.loc[memb.in_xu100,'ticker'].nunique()}")
        print(f"  XU030 ticker havuzu (tüm dönemler): {memb.loc[memb.in_xu030,'ticker'].nunique()}")

    if args.validate:
        v = validate()
        (OUT_DIR / "validation.json").write_text(json.dumps(v, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
