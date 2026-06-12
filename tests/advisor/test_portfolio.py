"""Portföy mutasyonları — ağırlıklı ortalama maliyet, nakit, audit log."""
import pytest

from agent.advisor.portfolio import Portfolio, empty_portfolio


def _data(cash=100_000.0):
    d = empty_portfolio()
    d["cash_tl"] = cash
    return d


class TestBuy:
    def test_new_position(self):
        d = _data()
        Portfolio.apply_buy(d, "bimas", 100, 480.0, source="test")
        p = d["positions"][0]
        assert p["ticker"] == "BIMAS" and p["qty"] == 100 and p["avg_cost"] == 480.0
        assert d["cash_tl"] == 100_000.0 - 48_000.0
        assert d["trades"][-1]["action"] == "BUY"

    def test_weighted_avg_cost(self):
        d = _data(cash=1_000_000.0)
        Portfolio.apply_buy(d, "AKBNK", 100, 10.0)
        Portfolio.apply_buy(d, "AKBNK", 100, 20.0)
        p = d["positions"][0]
        assert p["qty"] == 200 and p["avg_cost"] == 15.0

    def test_insufficient_cash_raises(self):
        d = _data(cash=1_000.0)
        with pytest.raises(ValueError, match="Yetersiz nakit"):
            Portfolio.apply_buy(d, "BIMAS", 100, 480.0)
        assert d["positions"] == []  # değişiklik yok

    def test_invalid_qty_raises(self):
        with pytest.raises(ValueError):
            Portfolio.apply_buy(_data(), "BIMAS", 0, 480.0)


class TestSell:
    def _with_pos(self):
        d = _data(cash=0.0)
        d["positions"].append({"ticker": "BIMAS", "qty": 100, "avg_cost": 480.0,
                               "entry_date": "2026-05-02", "stop": None,
                               "target": None, "notes": ""})
        return d

    def test_partial_sell_realized_pnl(self):
        d = self._with_pos()
        Portfolio.apply_sell(d, "BIMAS", qty=40, price=500.0)
        assert d["positions"][0]["qty"] == 60
        assert d["cash_tl"] == 40 * 500.0
        assert d["trades"][-1]["realized_pnl_tl"] == pytest.approx(40 * 20.0)

    def test_full_sell_removes_position(self):
        d = self._with_pos()
        Portfolio.apply_sell(d, "BIMAS", price=500.0)  # qty None = tamamı
        assert d["positions"] == []
        assert d["cash_tl"] == 50_000.0

    def test_oversell_raises(self):
        with pytest.raises(ValueError, match="Geçersiz adet"):
            Portfolio.apply_sell(self._with_pos(), "BIMAS", qty=150, price=500.0)

    def test_unknown_ticker_raises(self):
        with pytest.raises(ValueError, match="portföyde yok"):
            Portfolio.apply_sell(_data(), "THYAO", qty=1, price=1.0)


class TestMigrate:
    def test_old_holdings_schema(self):
        old = {"updated": "2026-03-11",
               "holdings": [{"ticker": "GENKM", "note": "x"}],
               "watchlist": ["PRDGS"]}
        data = Portfolio._migrate(old)
        assert data["positions"][0]["ticker"] == "GENKM"
        assert data["positions"][0]["qty"] == 0  # adet bilinmiyor — kullanıcı dolduracak
        assert "settings" in data and data["settings"]["max_position_pct"] == 15.0

    def test_settings_defaults_merged(self):
        d = {"version": 1, "cash_tl": 5.0, "positions": [],
             "settings": {"max_position_pct": 20.0}}
        out = Portfolio._migrate(d)
        assert out["settings"]["max_position_pct"] == 20.0   # kullanıcı değeri korunur
        assert out["settings"]["max_total_risk_pct"] == 6.0  # eksikler default

    def test_local_roundtrip(self, tmp_path):
        path = tmp_path / "pf.json"
        pf = Portfolio.load(path=str(path))  # yoksa boş başlar
        Portfolio.apply_set_cash(pf.data, 50_000.0)
        Portfolio.apply_buy(pf.data, "BIMAS", 10, 100.0)
        pf.save(updated_by="test")
        pf2 = Portfolio.load(path=str(path))
        assert pf2.data["cash_tl"] == 49_000.0
        assert pf2.data["positions"][0]["ticker"] == "BIMAS"
        assert pf2.data["updated_by"] == "test"
