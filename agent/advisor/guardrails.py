"""
NOX Advisor — sabit korkuluklar (LLM'den BAĞIMSIZ, saf deterministik).

pre_check  : pozisyon flag'leri + AL-adayı boyutlandırma tablosu + risk özeti
post_validate: LLM çıktısını şemaya zorlar, SAYILARI pack'ten yeniden damgalar,
               eksik pozisyona otomatik HOLD, evren-dışı ticker drop.

LLM'in doldurmasına izin verilenler: action / confidence / rationale_tr /
narrative_tr + AL adayları arasından SEÇİM. Başka her şey burada hesaplanır.
"""
import math

POSITION_ACTIONS = ("HOLD", "TRIM", "SELL")
CONFIDENCES = ("high", "medium", "low")

# Aday kabul durumları
CAND_OK = "OK"
CAND_RISK_CAP = "RISK_CAP_EXCEEDED"
CAND_BLOCKED_HELD = "BLOCKED_ALREADY_HELD"
CAND_BLOCKED_MAXPOS = "BLOCKED_MAX_POSITIONS"
CAND_BLOCKED_CASH = "BLOCKED_INSUFFICIENT_CASH"
CAND_BLOCKED_NO_REFS = "BLOCKED_MISSING_REFS"
CAND_ADD = "ADD"  # eldeki underweight pozisyona EXECUTABLE ekleme


def pre_check(pf_data, prices, de_buy_rows):
    """Portföy flag'leri + boyutlandırılmış AL tablosu.

    pf_data: portföy şema v1 dict
    prices : {ticker: last_close}
    de_buy_rows: signals.load_de_watchlist()["buy_rows"]
    """
    settings = pf_data["settings"]
    positions = [p for p in pf_data["positions"] if p["qty"] > 0]
    cash = float(pf_data["cash_tl"])

    equity = cash + sum(p["qty"] * (prices.get(p["ticker"]) or p["avg_cost"])
                        for p in positions)
    equity = max(equity, 1e-9)

    # ── pozisyon flag'leri ──
    pos_view, open_risk_tl = [], 0.0
    for p in positions:
        last = prices.get(p["ticker"])
        flags = []
        if last is None:
            flags.append("PRICE_UNAVAILABLE")
            last_eff = p["avg_cost"]
        else:
            last_eff = last
        notional = p["qty"] * last_eff
        weight = 100.0 * notional / equity
        pnl = 100.0 * (last_eff / p["avg_cost"] - 1.0) if p["avg_cost"] else 0.0

        if p.get("stop") and last is not None and last < p["stop"]:
            flags.append("STOP_VIOLATED")
        if weight > settings["max_position_pct"]:
            flags.append("OVERWEIGHT")
        if p.get("stop") and last is not None and last > p["stop"]:
            open_risk_tl += p["qty"] * (last - p["stop"])

        pos_view.append({
            "ticker": p["ticker"], "qty": p["qty"], "avg_cost": p["avg_cost"],
            "last": last, "stop": p.get("stop"), "target": p.get("target"),
            "weight_pct": round(weight, 2), "pnl_pct": round(pnl, 2),
            "notional_tl": round(notional, 2), "flags": flags,
            "entry_date": p.get("entry_date"), "notes": p.get("notes", ""),
        })

    # ── AL adayı boyutlandırma ──
    held = {p["ticker"] for p in positions}
    overweight = {v["ticker"] for v in pos_view if "OVERWEIGHT" in v["flags"]}
    min_cash = equity * settings["min_cash_buffer_pct"] / 100.0
    avail_cash = max(0.0, cash - min_cash)
    risk_cap_tl = equity * settings["max_total_risk_pct"] / 100.0
    used_risk = open_risk_tl
    n_pos = len(positions)
    base_risk_budget = equity * settings["per_trade_risk_pct"] / 100.0
    max_notional = equity * settings["max_position_pct"] / 100.0

    buy_table = []
    # sıra: EXECUTABLE tam bütçe → SIZE_REDUCED yarım bütçe (signals.py sıralı verir)
    for row in de_buy_rows:
        cand = dict(row)
        cand["suggested_qty"] = 0
        cand["risk_tl"] = 0.0
        cand["notional_tl"] = 0.0

        entry, stop = row.get("entry_ref"), row.get("stop_ref")
        if not entry or not stop or entry <= stop:
            cand["cand_status"] = CAND_BLOCKED_NO_REFS
            buy_table.append(cand)
            continue
        if row["ticker"] in overweight:
            cand["cand_status"] = CAND_BLOCKED_HELD
            cand["note"] = "max ağırlıkta — ekleme bloklu"
            buy_table.append(cand)
            continue
        is_add = row["ticker"] in held
        if not is_add and n_pos + sum(1 for c in buy_table if c["cand_status"] == CAND_OK) \
                >= settings["max_positions"]:
            cand["cand_status"] = CAND_BLOCKED_MAXPOS
            buy_table.append(cand)
            continue

        budget = base_risk_budget * (0.5 if row["section"] == "SIZE_REDUCED" else 1.0)
        per_share_risk = entry - stop
        qty = math.floor(budget / per_share_risk)
        # clamp: pozisyon tavanı + nakit
        qty = min(qty, math.floor(max_notional / entry))
        qty = min(qty, math.floor(avail_cash / entry))
        if qty <= 0:
            cand["cand_status"] = CAND_BLOCKED_CASH
            buy_table.append(cand)
            continue
        risk_tl = qty * per_share_risk
        if used_risk + risk_tl > risk_cap_tl:
            cand["cand_status"] = CAND_RISK_CAP
            cand["note"] = f"toplam risk tavanı (%{settings['max_total_risk_pct']}) doldu"
            buy_table.append(cand)
            continue

        used_risk += risk_tl
        avail_cash -= qty * entry
        cand["cand_status"] = CAND_ADD if is_add else CAND_OK
        cand["suggested_qty"] = qty
        cand["risk_tl"] = round(risk_tl, 2)
        cand["notional_tl"] = round(qty * entry, 2)
        cand["risk_pct_equity"] = round(100.0 * risk_tl / equity, 3)
        buy_table.append(cand)

    invested = equity - cash
    return {
        "equity_tl": round(equity, 2),
        "cash_tl": round(cash, 2),
        "invested_pct": round(100.0 * invested / equity, 2),
        "n_positions": n_pos,
        "positions": pos_view,
        "buy_table": buy_table,
        "risk": {
            "existing_open_risk_tl": round(open_risk_tl, 2),
            "existing_open_risk_pct": round(100.0 * open_risk_tl / equity, 2),
            "new_risk_tl": round(used_risk - open_risk_tl, 2),
            "total_risk_tl": round(used_risk, 2),
            "total_risk_pct": round(100.0 * used_risk / equity, 2),
            "cap_pct": settings["max_total_risk_pct"],
            "within_cap": used_risk <= risk_cap_tl + 1e-9,
        },
    }


def post_validate(advisory, pack):
    """LLM çıktısını zorla hizala. Sayılar pack'ten yeniden damgalanır."""
    log = list(advisory.get("guardrail_log", []))
    pre = pack["pre_check"]
    pos_by_tkr = {p["ticker"]: p for p in pre["positions"]}
    cand_by_tkr = {c["ticker"]: c for c in pre["buy_table"]}

    # ── pozisyon önerileri ──
    seen = set()
    clean_recs = []
    for rec in advisory.get("position_recommendations", []) or []:
        tkr = str(rec.get("ticker", "")).upper().strip()
        if tkr not in pos_by_tkr:
            log.append(f"{tkr}: portföyde yok — öneri düşürüldü")
            continue
        if tkr in seen:
            continue
        seen.add(tkr)
        p = pos_by_tkr[tkr]
        action = rec.get("action") if rec.get("action") in POSITION_ACTIONS else "HOLD"
        if action != rec.get("action"):
            log.append(f"{tkr}: geçersiz action '{rec.get('action')}' → HOLD")
        clean_recs.append({
            "ticker": tkr, "action": action,
            "confidence": rec.get("confidence") if rec.get("confidence") in CONFIDENCES else "low",
            "weight_pct": p["weight_pct"], "pnl_pct": p["pnl_pct"],
            "last": p["last"], "qty": p["qty"], "avg_cost": p["avg_cost"],
            "stop": p["stop"], "flags": p["flags"],
            "signal_refs": rec.get("signal_refs", []),
            "rationale_tr": str(rec.get("rationale_tr", ""))[:600],
        })
    # eksik pozisyonlara otomatik HOLD
    for tkr, p in pos_by_tkr.items():
        if tkr not in seen:
            log.append(f"{tkr}: model atladı → otomatik HOLD (low)")
            clean_recs.append({
                "ticker": tkr, "action": "HOLD", "confidence": "low",
                "weight_pct": p["weight_pct"], "pnl_pct": p["pnl_pct"],
                "last": p["last"], "qty": p["qty"], "avg_cost": p["avg_cost"],
                "stop": p["stop"], "flags": p["flags"], "signal_refs": [],
                "rationale_tr": "Model değerlendirmedi — varsayılan TUT.",
            })
    # STOP_VIOLATED asla sessiz HOLD kalamaz
    for rec in clean_recs:
        if "STOP_VIOLATED" in rec["flags"] and rec["action"] == "HOLD" \
                and "stop" not in rec["rationale_tr"].lower():
            rec["action"] = "SELL"
            rec["confidence"] = "high"
            rec["rationale_tr"] = ("Stop ihlal edildi (son fiyat < stop) — korkuluk kuralı: "
                                   "SAT/GÖZDEN GEÇİR. ") + rec["rationale_tr"]
            log.append(f"{rec['ticker']}: STOP_VIOLATED + HOLD → SELL'e zorlandı")

    # ── AL adayları: LLM yalnızca SEÇER, sayılar tablodan ──
    clean_buys = []
    for sel in advisory.get("buy_candidates", []) or []:
        tkr = str(sel.get("ticker", "")).upper().strip()
        cand = cand_by_tkr.get(tkr)
        if cand is None:
            log.append(f"{tkr}: aday tablosunda yok — AL önerisi düşürüldü")
            continue
        if cand["cand_status"] not in (CAND_OK, CAND_ADD):
            log.append(f"{tkr}: {cand['cand_status']} — AL önerisi düşürüldü")
            continue
        if any(b["ticker"] == tkr for b in clean_buys):
            continue
        adjusted = False
        for field in ("suggested_qty", "entry_ref", "stop_ref", "risk_tl"):
            if sel.get(field) is not None and sel.get(field) != cand.get(field):
                adjusted = True
        if adjusted:
            log.append(f"{tkr}: LLM sayısal alanları tabloya geri damgalandı")
        clean_buys.append({
            "ticker": tkr, "source": "decision_engine_v1",
            "section": cand["section"], "action": cand["cand_status"],
            "validation": pack["validated_signals"]["decision_engine"]["validation"],
            "entry_ref": cand["entry_ref"], "stop_ref": cand["stop_ref"],
            "atr": cand.get("atr"), "risk_atr": cand.get("risk_atr"),
            "n_cells": cand.get("n_cells"), "families": cand.get("families"),
            "suggested_qty": cand["suggested_qty"],
            "risk_tl": cand["risk_tl"], "notional_tl": cand["notional_tl"],
            "risk_pct_equity": cand.get("risk_pct_equity"),
            "guardrail_adjusted": adjusted,
            "rationale_tr": str(sel.get("rationale_tr", ""))[:600],
        })

    advisory_out = {
        "schema_version": 1,
        "asof": pack["asof"],
        "generated_at": advisory.get("generated_at"),
        "model": advisory.get("model"),
        "mode": advisory.get("mode", "llm"),
        "portfolio_rev": pack.get("portfolio_rev"),
        "inputs_status": pack["inputs_status"],
        "portfolio_summary": {
            "equity_tl": pre["equity_tl"], "cash_tl": pre["cash_tl"],
            "invested_pct": pre["invested_pct"], "n_positions": pre["n_positions"],
            "open_risk_pct": pre["risk"]["existing_open_risk_pct"],
        },
        "position_recommendations": clean_recs,
        "buy_candidates": clean_buys,
        "skipped_candidates": [
            {"ticker": c["ticker"], "status": c["cand_status"], "note": c.get("note", "")}
            for c in pre["buy_table"] if c["cand_status"] not in (CAND_OK, CAND_ADD)
        ],
        "risk_summary": pre["risk"],
        "guardrail_log": log,
        "narrative_tr": str(advisory.get("narrative_tr", ""))[:1500],
        "disclaimer_tr": ("Tüm birincil sinyal hatları kağıt-doğrulamalı ve tek-rejim "
                          "(2025-26 boğa) — canlı doğrulama yok. Yatırım tavsiyesi değildir."),
    }
    return advisory_out
