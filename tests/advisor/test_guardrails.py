"""Korkuluk matematiği — sentetik portföyler, LLM'siz saf fonksiyon testleri."""
import pytest

from agent.advisor import guardrails
from agent.advisor.portfolio import empty_portfolio


def _pf(cash=500_000.0, positions=None, **settings):
    data = empty_portfolio()
    data["cash_tl"] = cash
    data["positions"] = positions or []
    data["settings"].update(settings)
    return data


def _cand(ticker="AKBNK", section="EXECUTABLE", entry=100.0, stop=90.0, **kw):
    return {"ticker": ticker, "section": section, "entry_ref": entry,
            "stop_ref": stop, "atr": 2.0, "risk_atr": 1.5,
            "n_cells": 1, "families": "mb_1d", **kw}


class TestSizing:
    def test_basic_risk_sizing(self):
        # equity 500k, %1 risk = 5000 TL; entry 100 stop 90 → 10 TL/hisse → 500 adet
        pre = guardrails.pre_check(_pf(), {}, [_cand()])
        c = pre["buy_table"][0]
        assert c["cand_status"] == guardrails.CAND_OK
        assert c["suggested_qty"] == 500
        assert c["risk_tl"] == 5000.0

    def test_max_position_clamp(self):
        # %15 tavan = 75k notional → 100 TL'den max 750 adet; dar stopla sınır bağlar
        pre = guardrails.pre_check(_pf(), {}, [_cand(entry=100.0, stop=99.0)])
        c = pre["buy_table"][0]
        assert c["suggested_qty"] == 750  # risk-bazlı 5000 olurdu, notional clamp'ı bağladı

    def test_cash_clamp(self):
        # nakit 10k, tampon %5 → kullanılabilir azalır; 100 TL'lik hisse
        pf = _pf(cash=10_000.0)
        pre = guardrails.pre_check(pf, {}, [_cand()])
        c = pre["buy_table"][0]
        assert c["suggested_qty"] <= 10_000 // 100
        assert c["suggested_qty"] > 0

    def test_size_reduced_half_budget(self):
        pre = guardrails.pre_check(_pf(), {}, [_cand(section="SIZE_REDUCED")])
        assert pre["buy_table"][0]["suggested_qty"] == 250  # tam bütçenin yarısı

    def test_total_risk_cap(self):
        # %6 cap = 30k; her aday 5k risk → 6 kabul, 7. RISK_CAP
        cands = [_cand(ticker=f"T{i:02d}", entry=100.0 + i, stop=90.0 + i) for i in range(8)]
        pre = guardrails.pre_check(_pf(cash=5_000_000.0, max_positions=20,
                                       max_position_pct=100.0), {}, cands)
        ok = [c for c in pre["buy_table"] if c["cand_status"] == guardrails.CAND_OK]
        capped = [c for c in pre["buy_table"] if c["cand_status"] == guardrails.CAND_RISK_CAP]
        assert len(ok) == 6
        assert len(capped) == 2
        assert pre["risk"]["within_cap"]

    def test_missing_refs_blocked(self):
        pre = guardrails.pre_check(_pf(), {}, [_cand(entry=None, stop=None)])
        assert pre["buy_table"][0]["cand_status"] == guardrails.CAND_BLOCKED_NO_REFS

    def test_max_positions_blocked(self):
        positions = [{"ticker": f"P{i}", "qty": 10, "avg_cost": 10.0,
                      "stop": None, "target": None} for i in range(12)]
        pre = guardrails.pre_check(_pf(positions=positions), {}, [_cand()])
        assert pre["buy_table"][0]["cand_status"] == guardrails.CAND_BLOCKED_MAXPOS


class TestPositionFlags:
    def test_stop_violated(self):
        pos = [{"ticker": "BIMAS", "qty": 100, "avg_cost": 480.0, "stop": 455.0, "target": None}]
        pre = guardrails.pre_check(_pf(positions=pos), {"BIMAS": 450.0}, [])
        assert "STOP_VIOLATED" in pre["positions"][0]["flags"]

    def test_overweight_flag_and_buy_block(self):
        pos = [{"ticker": "BIMAS", "qty": 1000, "avg_cost": 480.0, "stop": None, "target": None}]
        pf = _pf(cash=100_000.0, positions=pos)  # 480k pozisyon / 580k equity = %83
        pre = guardrails.pre_check(pf, {"BIMAS": 480.0}, [_cand(ticker="BIMAS")])
        assert "OVERWEIGHT" in pre["positions"][0]["flags"]
        assert pre["buy_table"][0]["cand_status"] == guardrails.CAND_BLOCKED_HELD

    def test_held_underweight_becomes_add(self):
        pos = [{"ticker": "AKBNK", "qty": 10, "avg_cost": 95.0, "stop": None, "target": None}]
        pre = guardrails.pre_check(_pf(positions=pos), {"AKBNK": 100.0}, [_cand(ticker="AKBNK")])
        assert pre["buy_table"][0]["cand_status"] == guardrails.CAND_ADD

    def test_price_unavailable_flag(self):
        pos = [{"ticker": "XYZAB", "qty": 100, "avg_cost": 10.0, "stop": None, "target": None}]
        pre = guardrails.pre_check(_pf(positions=pos), {}, [])
        assert "PRICE_UNAVAILABLE" in pre["positions"][0]["flags"]


class TestPostValidate:
    def _pack(self, pre):
        return {"asof": "2026-06-13", "portfolio_rev": "abc123", "pre_check": pre,
                "inputs_status": {"decision_engine": "OK", "tavan_lock": "UNAVAILABLE",
                                  "cluster3": "UNAVAILABLE", "context_scanners": "OK",
                                  "macro": "OK", "prices": "OK"},
                "validated_signals": {"decision_engine": {"validation": "paper-track (tek-rejim)"}}}

    def test_malformed_llm_output(self):
        pos = [{"ticker": "BIMAS", "qty": 100, "avg_cost": 480.0, "stop": 455.0, "target": None},
               {"ticker": "AKBNK", "qty": 50, "avg_cost": 65.0, "stop": None, "target": None}]
        pre = guardrails.pre_check(_pf(positions=pos), {"BIMAS": 500.0, "AKBNK": 67.0},
                                   [_cand(ticker="THYAO")])
        raw = {
            "mode": "llm",
            "position_recommendations": [
                {"ticker": "BIMAS", "action": "YOLO", "confidence": "extreme",
                 "rationale_tr": "x"},                       # geçersiz enum → HOLD/low
                {"ticker": "FAKE1", "action": "SELL", "confidence": "high",
                 "rationale_tr": "uydurma"},                  # evren-dışı → drop
                # AKBNK atlandı → otomatik HOLD
            ],
            "buy_candidates": [
                {"ticker": "THYAO", "suggested_qty": 99999, "rationale_tr": "al"},  # sayı restamp
                {"ticker": "FAKE2", "rationale_tr": "uydurma"},                      # drop
            ],
            "narrative_tr": "test",
        }
        adv = guardrails.post_validate(raw, self._pack(pre))
        tickers = {r["ticker"] for r in adv["position_recommendations"]}
        assert tickers == {"BIMAS", "AKBNK"}                       # FAKE1 düştü, AKBNK auto-HOLD
        bimas = next(r for r in adv["position_recommendations"] if r["ticker"] == "BIMAS")
        assert bimas["action"] == "HOLD" and bimas["confidence"] == "low"
        akbnk = next(r for r in adv["position_recommendations"] if r["ticker"] == "AKBNK")
        assert akbnk["action"] == "HOLD" and "atladı" in " ".join(adv["guardrail_log"])
        assert len(adv["buy_candidates"]) == 1
        thyao = adv["buy_candidates"][0]
        assert thyao["suggested_qty"] != 99999                     # tablodan restamp
        assert thyao["guardrail_adjusted"]
        assert any("FAKE2" in g for g in adv["guardrail_log"])

    def test_stop_violated_hold_forced_to_sell(self):
        pos = [{"ticker": "BIMAS", "qty": 100, "avg_cost": 480.0, "stop": 455.0, "target": None}]
        pre = guardrails.pre_check(_pf(positions=pos), {"BIMAS": 450.0}, [])
        raw = {"mode": "llm",
               "position_recommendations": [{"ticker": "BIMAS", "action": "HOLD",
                                             "confidence": "high", "rationale_tr": "tutalım"}],
               "buy_candidates": [], "narrative_tr": ""}
        adv = guardrails.post_validate(raw, self._pack(pre))
        assert adv["position_recommendations"][0]["action"] == "SELL"

    def test_blocked_candidate_selection_dropped(self):
        pre = guardrails.pre_check(_pf(cash=0.0), {}, [_cand(ticker="THYAO")])
        assert pre["buy_table"][0]["cand_status"] == guardrails.CAND_BLOCKED_CASH
        raw = {"mode": "llm", "position_recommendations": [],
               "buy_candidates": [{"ticker": "THYAO", "rationale_tr": "al"}],
               "narrative_tr": ""}
        adv = guardrails.post_validate(raw, self._pack(pre))
        assert adv["buy_candidates"] == []
        assert adv["disclaimer_tr"]  # yapısal disclaimer her zaman var


class TestExtractJson:
    """synthesize_via_cli'nin serbest-metin JSON çıkarımı."""

    def test_plain_json(self):
        from agent.advisor.synthesis import _extract_json
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        from agent.advisor.synthesis import _extract_json
        assert _extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}

    def test_surrounding_prose(self):
        from agent.advisor.synthesis import _extract_json
        out = _extract_json('İşte sonuç:\n{"a": {"b": "iç {süslü} ve \\" tırnak"}}\nbitti.')
        assert out["a"]["b"] == 'iç {süslü} ve " tırnak'

    def test_no_json_raises(self):
        from agent.advisor.synthesis import _extract_json
        with pytest.raises(ValueError):
            _extract_json("hiç json yok burada")
