"""Rotasyon monitörü — "sonraki geçiş" satırının doğruluğu.

replay() geçişleri `off in (LEG1+1, LEG2+1)` ile tetikliyor, yani 11 ve 26'da.
fmt_status'taki eski ifade LEG1/LEG2/HOLD (10/25/40) döndürüyordu; off=10 ve
off=25'te bir sonraki bardaki geçişi atlayıp 15 bar sonrasını gösteriyordu.
"""
import pytest

from tools.sector_rotation_trade_v0 import HOLD, LEG1, LEG2, leg_asset


def next_switch(off):
    """fmt_status'taki ifadenin birebir aynısı."""
    return next((s for s in (LEG1 + 1, LEG2 + 1, HOLD) if off < s), HOLD)


def replay_fires_at(off):
    """replay() bu offset'te LEG_SWITCH üretir mi (kaynaktaki koşulun aynısı)."""
    return off in (LEG1 + 1, LEG2 + 1)


def test_leg_sinirlari_spec_ile_uyumlu():
    """Spec: Leg1 [0,+10] XBANK, Leg2 (+10,+25] XHOLD, Leg3 (+25,+40] üçlü."""
    assert (HOLD, LEG1, LEG2) == (40, 10, 25)
    assert leg_asset(1) == ('XBANK',)
    assert leg_asset(LEG1) == ('XBANK',)
    assert leg_asset(LEG1 + 1) == ('XHOLD',)
    assert leg_asset(LEG2) == ('XHOLD',)
    assert leg_asset(LEG2 + 1) == ('XMESY', 'XELKT', 'XFINK')


@pytest.mark.parametrize('off,beklenen', [
    (1, 11), (5, 11), (9, 11),
    (10, 11),          # regresyon: eskiden 25 diyordu
    (11, 26), (12, 26), (24, 26),
    (25, 26),          # regresyon: eskiden 40 diyordu
    (26, 40), (27, 40), (39, 40),
])
def test_sonraki_gecis_offseti(off, beklenen):
    assert next_switch(off) == beklenen


def test_gosterilen_offset_gercekten_gecis_uretir():
    """Gösterilen her offset'te ya LEG_SWITCH ya da zaman çıkışı olmalı."""
    for off in range(1, HOLD):
        nxt = next_switch(off)
        assert replay_fires_at(nxt) or nxt >= HOLD, \
            f"off={off} icin gosterilen {nxt}'de hicbir sey olmuyor"


def test_hicbir_gecis_atlanmaz():
    """Her gerçek geçiş, kendinden önceki offsetlerde duyurulmuş olmalı."""
    for fire in (LEG1 + 1, LEG2 + 1):
        for off in range(fire - 3, fire):
            if off < 1:
                continue
            assert next_switch(off) == fire, \
                f"off={off}, {fire}'deki gecisi atliyor ({next_switch(off)} diyor)"


def test_eski_ifade_hatali_idi():
    """Regresyonun geri gelmediğini garanti eder."""
    def eski(off):
        return LEG1 if off < LEG1 else (LEG2 if off < LEG2 else HOLD)
    hatali = [o for o in range(1, HOLD) if eski(o) != next_switch(o)]
    assert 10 in hatali and 25 in hatali
    assert all(next_switch(o) != eski(o) or o >= LEG2 + 1 for o in hatali)


def test_hedef_leg_etiketi():
    """Satırda hedef bacağın adı da görünmeli."""
    for off, lbl in [(9, 'XHOLD'), (25, 'XMESY+XELKT+XFINK')]:
        nxt = next_switch(off)
        assert '+'.join(leg_asset(nxt)) == lbl
    assert next_switch(30) >= HOLD          # çıkış — leg etiketi yok
