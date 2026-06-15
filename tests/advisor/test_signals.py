"""Sinyal yükleyiciler — eksik dosyada dürüst status, DE CSV parse."""
import json

import pandas as pd
import pytest

from agent.advisor import signals


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(signals, "OUT_DIR", tmp_path)
    return tmp_path


DE_COLS = ["section", "ticker", "source", "family", "state", "timeframe",
           "setup_label", "execution_label", "market_context", "reason_codes",
           "entry_ref", "stop_ref", "atr", "fill_assumption", "risk_atr",
           "trident_tier1_active"]


def _de_csv(out_dir, asof, rows):
    df = pd.DataFrame(rows, columns=DE_COLS)
    path = out_dir / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.csv"
    df.to_csv(path, index=False)
    return path


def _row(section="EXECUTABLE", ticker="AKBNK", tf="1d", entry=67.15, stop=62.86,
         risk_atr=1.53):
    return [section, ticker, "mb_scanner", f"mb_{tf}__retest_bounce_first", "retest",
            tf, "RETEST_SETUP", "EXECUTABLE", "short", "retest_confirmed",
            entry, stop, 2.81, "next_open", risk_atr, ""]


class TestDeLoader:
    def test_missing_everything_unavailable(self, out_dir):
        block = signals.load_de_watchlist("2026-06-13")
        assert block["status"] == "UNAVAILABLE"
        assert block["buy_rows"] == []

    def test_no_de_day(self, out_dir):
        (out_dir / "decision_engine_v1_no_de_day_advisory_2026-06-13.json").write_text(
            json.dumps({"reason": "both paper lines stale"}))
        block = signals.load_de_watchlist("2026-06-13")
        assert block["status"] == "NO_DE_DAY"

    def test_parse_and_dedupe(self, out_dir):
        _de_csv(out_dir, "2026-06-13", [
            _row(tf="1d"), _row(tf="5h", risk_atr=1.85),       # aynı ticker 2 hücre
            _row(section="SIZE_REDUCED", ticker="THYAO"),
            _row(section="WAIT_RETEST", ticker="GARAN"),
        ])
        block = signals.load_de_watchlist("2026-06-13")
        assert block["status"] == "OK"
        assert len(block["buy_rows"]) == 2                      # AKBNK + THYAO
        akbnk = next(r for r in block["buy_rows"] if r["ticker"] == "AKBNK")
        assert akbnk["timeframe"] == "1d"                       # 1d tercih edildi
        assert akbnk["n_cells"] == 2
        assert block["buy_rows"][0]["section"] == "EXECUTABLE"  # sıra korunur
        assert len(block["watch_rows"]) == 1

    def test_executable_wins_over_size_reduced_same_ticker(self, out_dir):
        _de_csv(out_dir, "2026-06-13", [
            _row(section="SIZE_REDUCED", ticker="AKBNK"),
            _row(section="EXECUTABLE", ticker="AKBNK"),
        ])
        block = signals.load_de_watchlist("2026-06-13")
        assert len(block["buy_rows"]) == 1
        assert block["buy_rows"][0]["section"] == "EXECUTABLE"

    def test_stale_walkback(self, out_dir):
        _de_csv(out_dir, "2026-06-11", [_row()])
        block = signals.load_de_watchlist("2026-06-13")
        assert block["status"] == "STALE"
        assert "2026-06-11" in block["note"]


class TestOtherLoaders:
    def test_tavan_missing(self, out_dir):
        block = signals.load_tavan_lock("2026-06-13")
        assert block["status"] == "UNAVAILABLE"
        assert "V1 exit" in block["exit_rules"]

    def test_tavan_present(self, out_dir):
        (out_dir / "tavan_lock_scan_2026-06-13.json").write_text(
            json.dumps({"asof": "2026-06-13",
                        "picks": [{"ticker": "EMPAE", "prob": 0.91}]}))
        block = signals.load_tavan_lock("2026-06-13")
        assert block["status"] == "OK"
        assert block["picks"][0]["ticker"] == "EMPAE"

    def test_cluster3_missing(self, out_dir):
        block = signals.load_cluster3("2026-06-13")
        assert block["status"] == "UNAVAILABLE"

    def test_cluster3_stale_and_open(self, out_dir):
        df = pd.DataFrame({
            "ticker": ["KARTN", "SANFM"],
            "signal_date": pd.to_datetime(["2026-05-20", "2026-05-25"]),
            "entry_close": [10.0, 20.0],
            "mfe_pct_20d_realized": [0.15, None],
            "y20_50_realized": [1.0, None],
            "realized_date_20d": [pd.Timestamp("2026-06-10"), pd.NaT],
        })
        df.to_parquet(out_dir / "ranking_lab_cluster3_paper_snapshots_v0.parquet")
        block = signals.load_cluster3("2026-06-13")
        assert block["status"] == "STALE"                # son sinyal > 7 gün eski
        assert block["realized_hit_rate"] == 1.0
        assert len(block["open_candidates"]) == 1        # sadece SANFM açık
        assert block["open_candidates"][0]["ticker"] == "SANFM"


class TestSectorRotation:
    def test_missing_unavailable(self, out_dir):
        block = signals.load_sector_rotation("2026-06-13")
        assert block["status"] == "UNAVAILABLE"

    def test_fresh_ok_with_ignition(self, out_dir):
        (out_dir / "sector_rotation_monitor_log.csv").write_text(
            "run_ts,bar_date,xu100,state_on,brake_on,state_off,brake_off,new_events\n"
            "2026-06-10T20:30:00,2026-06-10,13800.0,ARMED,False,IN_TRADE,False,\n"
            "2026-06-12T20:30:00,2026-06-11,13938.48,ARMED,False,IN_TRADE,False,"
            "\"[bilgi] 2026-06-11 BANK_IGNITION — XBANK 1g rel +6.4pp\"\n")
        block = signals.load_sector_rotation("2026-06-13")
        assert block["status"] == "OK"
        assert block["state_primary_confirm_on"] == "ARMED"
        assert block["brake_primary"] is False
        assert "BANK_IGNITION" in (block["last_bank_ignition"] or "")

    def test_stale(self, out_dir):
        (out_dir / "sector_rotation_monitor_log.csv").write_text(
            "run_ts,bar_date,xu100,state_on,brake_on,state_off,brake_off,new_events\n"
            "2026-06-01T20:30:00,2026-06-01,13500.0,IDLE,False,IDLE,False,\n")
        block = signals.load_sector_rotation("2026-06-13")
        assert block["status"] == "STALE"


class TestHwObos:
    def _write(self, out_dir, tf, date, rows):
        df = pd.DataFrame(rows, columns=["ticker","tf","ts","kind","hyperwave","signal","close","volume"])
        df.to_csv(out_dir / f"hw_obos_{tf}_scan_{date}.csv", index=False)

    def test_missing_unavailable(self, out_dir):
        assert signals.load_hw_obos("2026-06-13")["status"] == "UNAVAILABLE"

    def test_recency_filter_and_tf_map(self, out_dir):
        # taze SAT_OB (3 gün önce) + bayat AL_OS (60 gün önce, elenmeli)
        self._write(out_dir, "1w", "2026-06-12", [
            ["BIMAS","1w","2026-06-10","SAT_OB",85,86,500,1e6],
            ["ESKI","1w","2026-04-10","AL_OS",15,14,10,1e6]])
        self._write(out_dir, "1d", "2026-06-12", [
            ["BIMAS","1d","2026-06-11","SAT_OB",82,83,500,1e6]])
        blk = signals.load_hw_obos("2026-06-13")
        assert blk["status"] == "OK"
        assert set(blk["per_ticker"]["BIMAS"]["SAT_OB"]) == {"1w","1d"}  # iki TF
        assert "ESKI" not in blk["per_ticker"]                          # recency eledi
        assert blk["n_sat_ob"] == 1 and blk["n_al_os"] == 0

    def test_sector_breadth(self, out_dir):
        (out_dir / "bist_sector_map.csv").write_text("ticker,sektor_id\nBIMAS,10\nMGROS,10\n")
        self._write(out_dir, "1d", "2026-06-12", [
            ["BIMAS","1d","2026-06-11","SAT_OB",82,83,500,1e6],
            ["MGROS","1d","2026-06-11","SAT_OB",80,81,400,1e6]])
        blk = signals.load_hw_obos("2026-06-13")
        assert sorted(blk["sector_breadth"][10]["SAT_OB"]) == ["BIMAS","MGROS"]


class TestHwSoftFlag:
    def test_sat_ob_flags_position_without_forcing_sell(self):
        from agent.advisor import guardrails
        from agent.advisor.portfolio import empty_portfolio
        d = empty_portfolio(); d["cash_tl"] = 100_000.0
        d["positions"] = [{"ticker":"BIMAS","qty":100,"avg_cost":480.0,"stop":None,"target":None}]
        pre = guardrails.pre_check(d, {"BIMAS": 500.0}, [],
                                   hw_per_ticker={"BIMAS": {"AL_OS": [], "SAT_OB": ["1w","1mo"]}})
        flags = pre["positions"][0]["flags"]
        assert any(f.startswith("HW_ROLLING_OVER") for f in flags)
        assert "STOP_VIOLATED" not in flags                              # forced-sell tetiklemez


class TestTakas:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.delenv("MATRIKS_API_KEY", raising=False)
        from agent.advisor import takas
        r = takas.load_takas(["BIMAS"])
        assert r["status"] == "DISABLED" and r["per_ticker"] == {}

    def test_render_matriks_error_banner(self):
        from agent.advisor.render import render_html
        adv = {"asof":"2026-06-12","mode":"llm","model":"x","portfolio_rev":"r",
               "inputs_status":{"takas":"MATRIKS_ERROR"},
               "portfolio_summary":{"equity_tl":1,"cash_tl":1,"invested_pct":0,"open_risk_pct":0,"n_positions":0},
               "position_recommendations":[],"buy_candidates":[],"risk_summary":{"cap_pct":6},
               "guardrail_log":[],"narrative_tr":"","disclaimer_tr":"d",
               "takas":{"status":"MATRIKS_ERROR","error":"401 unauthorized","per_ticker":{}}}
        path = render_html(adv)
        html = open(path, encoding="utf-8").read()
        assert "MATRİKS API HATASI" in html and "401 unauthorized" in html
