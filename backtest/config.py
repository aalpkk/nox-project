"""
NOX Backtest — Config
Rejim dönemleri, filtre parametreleri, test matrisi.
"""

# ── REJIM DÖNEMLERI (BIST, XU100 bazlı) ──
REGIMES = {
    # 10 yıllık BIST dönemleri
    "bear_2018":     {"start": "2018-06-01", "end": "2019-01-01", "label": "🐻 2018 Krizi",       "type": "bear"},
    "recovery_2019": {"start": "2019-01-01", "end": "2020-01-01", "label": "📈 2019 Toparlanma",  "type": "bull"},
    "covid_crash":   {"start": "2020-02-01", "end": "2020-04-01", "label": "🐻 Covid Çöküş",      "type": "bear"},
    "covid_recov":   {"start": "2020-04-01", "end": "2020-11-01", "label": "📈 Covid V-Recovery",  "type": "bull"},
    "sideways_2021": {"start": "2021-01-01", "end": "2021-09-01", "label": "➡️ 2021 Yatay",       "type": "sideways"},
    "infl_rally":    {"start": "2021-09-01", "end": "2022-06-01", "label": "🐂 Enflasyon Rally",   "type": "bull"},
    "bear_2022":     {"start": "2022-06-01", "end": "2022-12-01", "label": "🐻 2022 Düzeltme",    "type": "bear"},
    "super_bull":    {"start": "2023-01-01", "end": "2023-07-01", "label": "🐂 2023 Süper Boğa",   "type": "bull"},
    "correct_2023":  {"start": "2023-07-01", "end": "2023-10-01", "label": "📉 2023 Düzeltme",    "type": "bear"},
    "sideways_2024": {"start": "2023-10-01", "end": "2024-02-01", "label": "➡️ 2023-24 Yatay",    "type": "sideways"},
    "bull_2024":     {"start": "2024-02-01", "end": "2024-05-01", "label": "🐂 2024 Rally",        "type": "bull"},
    "bear_2024":     {"start": "2024-05-01", "end": "2024-08-01", "label": "🐻 2024 Satış",       "type": "bear"},
    "sideways_2024b":{"start": "2024-08-01", "end": "2024-12-01", "label": "➡️ 2024 Yatay-2",     "type": "sideways"},
    "bull_2025":     {"start": "2025-01-01", "end": "2025-06-01", "label": "🐂 2025 Rally",        "type": "bull"},
}

# Rejim tipleri gruplanmış
REGIME_GROUPS = {
    "bull":     [k for k, v in REGIMES.items() if v['type'] == 'bull'],
    "bear":     [k for k, v in REGIMES.items() if v['type'] == 'bear'],
    "sideways": [k for k, v in REGIMES.items() if v['type'] == 'sideways'],
    "all":      list(REGIMES.keys()),
}

# ── TRADE SİMÜLASYON PARAMETRELERİ ──
ENTRY_METHOD = "next_open"     # "close" veya "next_open"
MAX_HOLD_DAYS = 30             # was 20 — kazanan trade'lere daha çok zaman
COMMISSION_PCT = 0.002         # %0.2 komisyon (alış + satış)
SLIPPAGE_PCT = 0.001           # %0.1 slippage

# ── FİLTRE TEST MATRİSİ ──
FILTER_TESTS = {
    "baseline":       {"quality": 0,  "rs": -999, "rr": 0,   "oe_max": 999, "desc": "Filtre yok"},
    "q50":            {"quality": 50, "rs": -999, "rr": 0,   "oe_max": 999, "desc": "Quality ≥50"},
    "q70":            {"quality": 70, "rs": -999, "rr": 0,   "oe_max": 999, "desc": "Quality ≥70"},
    "rs10":           {"quality": 0,  "rs": 10,   "rr": 0,   "oe_max": 999, "desc": "RS ≥10"},
    "rs20":           {"quality": 0,  "rs": 20,   "rr": 0,   "oe_max": 999, "desc": "RS ≥20"},
    "oe_filter":      {"quality": 0,  "rs": -999, "rr": 0,   "oe_max": 3,   "desc": "OE <3"},
    "rr15":           {"quality": 0,  "rs": -999, "rr": 1.5, "oe_max": 999, "desc": "R:R ≥1.5"},
    "rr20":           {"quality": 0,  "rs": -999, "rr": 2.0, "oe_max": 999, "desc": "R:R ≥2.0"},
    "conservative":   {"quality": 70, "rs": 20,   "rr": 2.0, "oe_max": 3,   "desc": "Conservative"},
    "balanced":       {"quality": 50, "rs": 10,   "rr": 1.5, "oe_max": 4,   "desc": "Balanced"},
    "aggressive":     {"quality": 30, "rs": 0,    "rr": 1.0, "oe_max": 5,   "desc": "Aggressive"},
}

# ── SİNYAL GRUPLARI (ayrı analiz) ──
SIGNAL_GROUPS = {
    "trend_all":  ["COMBO", "STRONG", "WEAK", "EARLY", "PARTIAL"],
    "trend_top":  ["STRONG", "EARLY", "PARTIAL"],
    "dip_all":    ["DIP+", "DIP", "DIP_E", "DIP_W", "RECOVER"],
    "dip_top":    ["DIP+", "DIP", "RECOVER"],
}
