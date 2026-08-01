"""Sinyal tazeliği / yeni-eski ayrımı / çelişki yüzeyi — regresyon testleri.

Senaryo: 2026-07-29 brifinginde IZFAS Tier1'deydi.
  - SBT sinyali dünün tavan mumundan (entry 71.65, bars_ago=1)
  - alpha ⏳3G (tarih alanı yok)
  - aynı gün regime_transition SAT (csv_date 20260729, fiyat 64.50 = taban)
  - CONF: has_conflict=True, recommendation=BEKLE

Beklenen yeni davranış:
  (a) "taşınan sinyaller" blokunda (carried=True)
  (b) 🕐D-1 rozeti
  (c) "⚠️ÇELİŞKİ: regime_transition SAT" + "BEKLE"
Ve: Tier1 ÜYELİĞİ değişmez — overlay yalnızca annotate eder.
"""
from datetime import datetime, timedelta, timezone

import pytest

from agent.briefing import (
    _build_freshness_ctx, _signal_age, _freshness_overlay, _split_carried,
)
from agent.html_report import _freshness_fields

_TZ = timezone(timedelta(hours=3))
REPORT_DAY = datetime(2026, 7, 29, 18, 0, tzinfo=_TZ)
SD = "2026-07-29"
CD = "20260729"


def _izfas_signals():
    return [
        # SBT: bugün taranmış ama sinyal dünün mumundan → bars_ago=1
        {'screener': 'sbt', 'ticker': 'IZFAS', 'signal_date': SD, 'csv_date': CD,
         'direction': 'AL', 'bars_ago': 1, 'entry': 71.65},
        # regime_transition: BUGÜN tarihli SAT
        {'screener': 'regime_transition', 'ticker': 'IZFAS', 'signal_date': SD,
         'csv_date': CD, 'direction': 'SAT', 'close': 64.50},
        # başka bir hisse, bugün taze AL — kontrol grubu
        {'screener': 'tavan', 'ticker': 'FRESHCO', 'signal_date': SD, 'csv_date': CD,
         'direction': 'AL', 'skor': 60},
    ]


def _lists_dict():
    izfas_sbt = {'screener': 'sbt', 'ticker': 'IZFAS', 'signal_date': SD,
                 'csv_date': CD, 'direction': 'AL', 'bars_ago': 1, 'entry': 71.65}
    izfas_alpha = {'ticker': 'IZFAS', 'screener': 'alpha', 'ml_1g': 0.6,
                   'close': 64.50}          # tarih alanı YOK
    fresh_tvn = {'screener': 'tavan', 'ticker': 'FRESHCO', 'signal_date': SD,
                 'csv_date': CD, 'direction': 'AL', 'skor': 60}
    fresh_sbt = {'screener': 'sbt', 'ticker': 'FRESHCO', 'signal_date': SD,
                 'csv_date': CD, 'direction': 'AL', 'bars_ago': 0}
    return {
        'sbt': [('IZFAS', 100, ['SBT A'], izfas_sbt),
                ('FRESHCO', 90, ['SBT A'], fresh_sbt)],
        'alpha': [('IZFAS', 80, ['⏳3G', 'ML:0.60'], izfas_alpha)],
        'tavan': [('FRESHCO', 95, ['tvn=60'], fresh_tvn)],
        'tier1': [
            ('IZFAS', 55, ['⏳3G ⚡SBT+ALP [55p]'],
             {'overlap_count': 2, 'in_lists': ['sbt', 'alpha']}),
            ('FRESHCO', 60, ['⚡1G ⚡TVN+SBT [60p]'],
             {'overlap_count': 2, 'in_lists': ['tavan', 'sbt']}),
        ],
        'tier2a': [],
        'tier2b': [],
    }


def _conf():
    return [
        {'ticker': 'IZFAS', 'has_conflict': True, 'recommendation': 'BEKLE',
         'conflict_sources': ['regime_transition'], 'score': 5, 'source_count': 2},
        {'ticker': 'FRESHCO', 'has_conflict': False, 'recommendation': 'TRADEABLE',
         'score': 8, 'source_count': 2},
    ]


@pytest.fixture
def applied():
    ld = _lists_dict()
    before = [t for t, *_ in ld['tier1']]
    _freshness_overlay(ld, _izfas_signals(), _conf())
    return ld, before


def _meta(ld, ticker, key='tier1'):
    for t, _s, _r, m in ld[key]:
        if t == ticker:
            return m
    raise AssertionError(f"{ticker} {key} icinde yok")


def _reasons(ld, ticker, key='tier1'):
    for t, _s, r, _m in ld[key]:
        if t == ticker:
            return r
    raise AssertionError(f"{ticker} {key} icinde yok")


# ── yaş kovası birim testleri ─────────────────────────────────

def test_bars_ago_signal_date_oncelikli():
    """signal_date=bugün ama bars_ago=1 → D-1 (signal_date kazanmamalı)."""
    ctx = _build_freshness_ctx(_izfas_signals(), now=REPORT_DAY)
    sig = {'screener': 'sbt', 'signal_date': SD, 'csv_date': CD, 'bars_ago': 1}
    assert _signal_age(sig, ctx) == 'd1'


def test_bugun_sinyali():
    ctx = _build_freshness_ctx(_izfas_signals(), now=REPORT_DAY)
    assert _signal_age({'screener': 'tavan', 'signal_date': SD, 'csv_date': CD}, ctx) == 'today'
    assert _signal_age({'screener': 'sbt', 'signal_date': SD, 'bars_ago': 0}, ctx) == 'today'


def test_tarihsiz_sinyal_unknown():
    """alpha'da signal_date/csv_date/bars_ago yok → taze sayılmaz."""
    ctx = _build_freshness_ctx(_izfas_signals(), now=REPORT_DAY)
    assert _signal_age({'ticker': 'IZFAS', 'screener': 'alpha'}, ctx) == 'unknown'


def test_eski_sinyal_d2_d3():
    ctx = _build_freshness_ctx(_izfas_signals(), now=REPORT_DAY)
    assert _signal_age({'screener': 'sbt', 'bars_ago': 2}, ctx) == 'd2'
    assert _signal_age({'screener': 'sbt', 'bars_ago': 3}, ctx) == 'd3'
    assert _signal_age({'screener': 'sbt', 'bars_ago': 9}, ctx) == 'old'


# ── IZFAS senaryosu ───────────────────────────────────────────

def test_a_izfas_tasinan_bloka_gider(applied):
    ld, _ = applied
    assert _meta(ld, 'IZFAS')['carried'] is True
    fresh, carried = _split_carried(ld['tier1'])
    assert [t for t, *_ in carried] == ['IZFAS']
    assert 'IZFAS' not in [t for t, *_ in fresh]


def test_b_izfas_d1_rozeti(applied):
    ld, _ = applied
    m = _meta(ld, 'IZFAS')
    assert m['freshness_badge'] == '🕐D-1'
    assert m['freshness'] == {'today': [], 'stale': ['alpha', 'sbt']}
    assert m['freshness_ages'] == {'sbt': 'd1', 'alpha': 'unknown'}
    # rozet reasons dizisinde de olmalı ve başta gelmeli
    assert _reasons(ld, 'IZFAS')[0] == '🕐D-1'


def test_c_izfas_celiski_ve_oneri(applied):
    ld, _ = applied
    m = _meta(ld, 'IZFAS')
    assert m['has_conflict'] is True
    assert m['conflict_badge'] == '⚠️ÇELİŞKİ: regime_transition SAT'
    assert m['conf_recommendation'] == 'BEKLE'
    rs = _reasons(ld, 'IZFAS')
    assert '⚠️ÇELİŞKİ: regime_transition SAT' in rs
    assert 'CONF:BEKLE' in rs


def test_taze_hisse_normal_tier1de_kalir(applied):
    ld, _ = applied
    m = _meta(ld, 'FRESHCO')
    assert m['carried'] is False
    assert m['freshness_badge'] == '🟢BUGÜN'
    assert m['has_conflict'] is False
    assert m['conf_recommendation'] == 'TRADEABLE'
    fresh, _c = _split_carried(ld['tier1'])
    assert 'FRESHCO' in [t for t, *_ in fresh]


# ── regresyon: üyelik ve skor değişmez ────────────────────────

def test_tier1_uyeligi_degismez(applied):
    ld, before = applied
    assert [t for t, *_ in ld['tier1']] == before


def test_skorlar_degismez():
    ld = _lists_dict()
    scores_before = {t: s for t, s, *_ in ld['tier1']}
    _freshness_overlay(ld, _izfas_signals(), _conf())
    assert {t: s for t, s, *_ in ld['tier1']} == scores_before


def test_orijinal_reasons_korunur():
    ld = _lists_dict()
    orig = {t: list(r) for t, _s, r, _m in ld['tier1']}
    _freshness_overlay(ld, _izfas_signals(), _conf())
    for t, _s, r, _m in ld['tier1']:
        for old in orig[t]:
            assert old in r, f"{t}: '{old}' reasons'tan düştü"


def test_kaynak_listeleri_mutasyona_ugramaz():
    """tier2a/2b meta'sı ham sinyal dict'i — kopyalanmalı, kirletilmemeli."""
    ld = _lists_dict()
    ld['tier2b'] = [('IZFAS', 80, ['⏳3G'], ld['alpha'][0][3])]
    alpha_sig = ld['alpha'][0][3]
    _freshness_overlay(ld, _izfas_signals(), _conf())
    assert 'carried' not in alpha_sig, "ham sinyal dict'i kirletildi"
    assert 'freshness' not in alpha_sig


# ── JSON sözleşmesi (downstream advisor) ──────────────────────

def test_json_alanlari(applied):
    ld, _ = applied
    f = _freshness_fields(_meta(ld, 'IZFAS'))
    assert f['freshness'] == {'today': [], 'stale': ['alpha', 'sbt']}
    assert f['has_conflict'] is True
    assert f['conf_recommendation'] == 'BEKLE'
    assert f['carried'] is True
    assert f['freshness_badge'] == '🕐D-1'


def test_json_alanlari_meta_yoksa():
    assert _freshness_fields(None) == {}
    f = _freshness_fields({})
    assert f['freshness'] == {'today': [], 'stale': []}
    assert f['has_conflict'] is False
    assert f['conf_recommendation'] == ''


# ── boş / eksik veri dayanıklılığı ────────────────────────────

def test_bos_lists_dict_patlamaz():
    ld = {'tier1': [], 'tier2a': [], 'tier2b': []}
    _freshness_overlay(ld, [], None)
    assert ld['tier1'] == []


def test_conf_yoksa_celiski_bugunku_sattan_gelir():
    """CONF verilmese bile bugün tarihli SAT çelişki üretmeli."""
    ld = _lists_dict()
    _freshness_overlay(ld, _izfas_signals(), None)
    m = _meta(ld, 'IZFAS')
    assert m['has_conflict'] is True
    assert m['conflict_sources'] == ['regime_transition']


# ── değişmezlik: rastgele üretilmiş brifing kesitleri ─────────
#
# NOT: Geçmiş brifinglerin birebir replay'i bu ortamda yapılamıyor —
# pipeline canlı tarayıcı HTML'i + Matriks çekiyor, üretilen brifingler
# yalnızca CI artifact'i olarak saklanıyor ve yerelde arşivlenmiş GİRDİ yok
# (gh CLI de yok). Replay'in kanıtlamayı amaçladığı şey doğrudan test
# ediliyor: overlay üyeliği/sırayı/skoru değiştiremez.

import random  # noqa: E402

_SRC = ['alsat', 'tavan', 'nw', 'rt', 'sbt', 'alpha']
_SCR = {'alsat': 'alsat', 'tavan': 'tavan', 'nw': 'nox_v3_daily',
        'rt': 'regime_transition', 'sbt': 'sbt', 'alpha': 'alpha'}


def _random_case(rng):
    tickers = [f"T{i:03d}" for i in range(rng.randint(3, 25))]
    signals, lists = [], {k: [] for k in _SRC}
    for t in tickers:
        for lk in rng.sample(_SRC, rng.randint(1, 4)):
            sig = {'screener': _SCR[lk], 'ticker': t, 'direction': 'AL'}
            mode = rng.choice(['today', 'bars', 'old', 'none'])
            if mode == 'today':
                sig.update({'signal_date': SD, 'csv_date': CD})
            elif mode == 'bars':
                sig.update({'signal_date': SD, 'csv_date': CD,
                            'bars_ago': rng.randint(0, 5)})
            elif mode == 'old':
                sig.update({'signal_date': '2026-07-20', 'csv_date': '20260720'})
            lists[lk].append((t, rng.randint(1, 200), [f'{lk}-r'], sig))
            signals.append(sig)
        if rng.random() < 0.3:
            signals.append({'screener': 'regime_transition', 'ticker': t,
                            'direction': 'SAT', 'signal_date': SD, 'csv_date': CD})
    tier1 = []
    for t in tickers:
        inl = [lk for lk in _SRC if any(x[0] == t for x in lists[lk])]
        if len(inl) >= 2:
            tier1.append((t, rng.randint(40, 120), ['base'],
                          {'overlap_count': len(inl), 'in_lists': inl}))
    ld = dict(lists)
    ld['tier1'] = tier1
    ld['tier2a'] = [(t, s, list(r), dict(g)) for t, s, r, g in lists['tavan'][:3]]
    ld['tier2b'] = [(t, s, list(r), dict(g)) for t, s, r, g in lists['alpha'][:3]]
    conf = [{'ticker': t, 'has_conflict': rng.random() < 0.25,
             'recommendation': rng.choice(['BEKLE', 'TRADEABLE', 'İZLE'])}
            for t in tickers]
    return ld, signals, conf


@pytest.mark.parametrize('seed', range(40))
def test_degismezlik_uyelik_sira_skor(seed):
    rng = random.Random(seed)
    ld, signals, conf = _random_case(rng)
    before = {k: [(t, s) for t, s, *_ in ld.get(k, [])]
              for k in ('tier1', 'tier2a', 'tier2b')}
    _freshness_overlay(ld, signals, conf)
    after = {k: [(t, s) for t, s, *_ in ld.get(k, [])]
             for k in ('tier1', 'tier2a', 'tier2b')}
    assert after == before, "overlay üyeliği/sırayı/skoru değiştirdi"


@pytest.mark.parametrize('seed', range(40))
def test_degismezlik_gruplama_kayipsiz(seed):
    """fresh + carried, tier1'in tamamını ve sırasını korur."""
    rng = random.Random(seed)
    ld, signals, conf = _random_case(rng)
    _freshness_overlay(ld, signals, conf)
    fresh, carried = _split_carried(ld['tier1'])
    assert len(fresh) + len(carried) == len(ld['tier1'])
    assert {t for t, *_ in fresh} | {t for t, *_ in carried} == \
           {t for t, *_ in ld['tier1']}
    for block in (fresh, carried):
        idx = [[t for t, *_ in ld['tier1']].index(t) for t, *_ in block]
        assert idx == sorted(idx), "blok içi sıra bozuldu"


@pytest.mark.parametrize('seed', range(20))
def test_her_item_sozlesmeli_alanlari_tasir(seed):
    rng = random.Random(seed)
    ld, signals, conf = _random_case(rng)
    _freshness_overlay(ld, signals, conf)
    for key in ('tier1', 'tier2a', 'tier2b'):
        for t, _s, _r, m in ld.get(key, []):
            f = _freshness_fields(m)
            assert set(['freshness', 'has_conflict', 'conf_recommendation']) <= set(f)
            assert set(f['freshness']) == {'today', 'stale'}
            assert isinstance(f['has_conflict'], bool)
            # carried, today listesinin boşluğuyla tutarlı olmalı
            assert m['carried'] == (len(m['freshness']['today']) == 0)


def test_shortlist_json_alanlari_uctan_uca():
    """_prepare_shortlist_json çıktısı downstream sözleşmesini taşıyor mu."""
    from agent.html_report import _prepare_shortlist_json
    ld = _lists_dict()
    _freshness_overlay(ld, _izfas_signals(), _conf())
    js = _prepare_shortlist_json(ld)
    izfas = [e for e in js['tier1']['items'] if e['ticker'] == 'IZFAS'][0]
    assert izfas['carried'] is True
    assert izfas['freshness'] == {'today': [], 'stale': ['alpha', 'sbt']}
    assert izfas['has_conflict'] is True
    assert izfas['conf_recommendation'] == 'BEKLE'
    assert izfas['freshness_badge'] == '🕐D-1'
    assert izfas['conflict_badge'] == '⚠️ÇELİŞKİ: regime_transition SAT'
    fresh = [e for e in js['tier1']['items'] if e['ticker'] == 'FRESHCO'][0]
    assert fresh['carried'] is False
    assert fresh['freshness']['today'] == ['sbt', 'tavan']
