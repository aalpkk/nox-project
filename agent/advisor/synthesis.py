"""
NOX Advisor — LLM sentezi + deterministik fallback.

Tek yapılandırılmış Claude çağrısı (tool_choice ile zorlanmış emit_advisory).
LLM yalnızca action/confidence/rationale_tr/narrative_tr doldurur ve önceden
boyutlandırılmış AL adayları arasından SEÇER. Çağrı başarısızsa fallback —
rapor her halükârda çıkar.
"""
import json
import datetime

from agent.advisor import guardrails

ADVISORY_TOOL = {
    "name": "emit_advisory",
    "description": "Günlük portföy değerlendirmesini yapılandırılmış olarak yayınla.",
    "input_schema": {
        "type": "object",
        "properties": {
            "position_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "action": {"type": "string", "enum": ["HOLD", "TRIM", "SELL"]},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "signal_refs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "validation": {"type": "string"},
                                },
                                "required": ["source", "validation"],
                            },
                        },
                        "rationale_tr": {"type": "string",
                                         "description": "1-3 cümle Türkçe gerekçe; sinyalin doğrulama etiketini tekrarla"},
                    },
                    "required": ["ticker", "action", "confidence", "rationale_tr"],
                },
            },
            "buy_candidates": {
                "type": "array",
                "description": "Aday tablosundan SEÇİLEN AL'lar. Yalnızca cand_status OK/ADD satırları; sayıları DEĞİŞTİRME.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "rationale_tr": {"type": "string"},
                    },
                    "required": ["ticker", "rationale_tr"],
                },
            },
            "narrative_tr": {"type": "string",
                             "description": "3-5 cümle genel değerlendirme (makro rejim + portföy duruşu)"},
        },
        "required": ["position_recommendations", "buy_candidates", "narrative_tr"],
    },
}

SYSTEM_PROMPT_TR = """Sen NOX kişisel portföy danışmanısın (BIST). Sana deterministik bir
"context pack" verilecek: gerçek portföy, korkuluk ön-hesapları, doğrulanmış sinyal hatları
ve bağlam taramaları.

KURALLAR (ihlal post-validasyonda düzeltilir, uğraşma):
1. SAYI ÜRETME. Tüm giriş/stop/adet/risk sayıları pack'te hesaplanmış durumda; sen yalnızca
   action/confidence/gerekçe yazar ve AL adayları arasından seçim yaparsın.
2. AL adayı olarak YALNIZCA buy_table'da cand_status OK veya ADD olan satırları seçebilirsin.
   Seçmediğin OK satırı varsa nedenini narrative'de belirt.
3. Her gerekçede dayandığın sinyalin doğrulama etiketini aynen tekrarla
   (örn. "paper-track (tek-rejim)"). context-only/keşifsel etiketli bir sinyal hiçbir zaman
   birincil AL/SAT gerekçesi olamaz — yalnızca destekleyici renk.
4. STOP_VIOLATED flag'li pozisyonda HOLD önerme; SELL veya gerekçeli TRIM.
5. OVERWEIGHT flag'li pozisyonda TRIM'i ciddi değerlendir.
6. Tavan lock pick'leri pozisyon BAĞLAMIDIR (sabah 11:00 anlık görüntüsü) — akşam AL adayı değil.
   Elde tavan-lock kaynaklı pozisyon varsa V1 exit kurallarını gerekçeye yansıt.
7. Tüm birincil hatlar kağıt-doğrulamalı ve tek-rejim — kesinlik iddia etme, ölçülü konuş.
8. Türkçe yaz, kısa ve operasyonel ol."""


def _pack_for_llm(pack):
    """Pack'in LLM'e giden kompakt hâli (ham fiyat dict'i vb. atılır)."""
    ctx = pack["context_signals"]
    return {
        "asof": pack["asof"],
        "inputs_status": pack["inputs_status"],
        "settings": pack["settings"],
        "portfolio": {
            "summary": {k: pack["pre_check"][k] for k in
                        ("equity_tl", "cash_tl", "invested_pct", "n_positions")},
            "risk": pack["pre_check"]["risk"],
            "positions": pack["pre_check"]["positions"],
        },
        "buy_table": pack["pre_check"]["buy_table"],
        "validated_signals": {
            "decision_engine": {k: v for k, v in pack["validated_signals"]["decision_engine"].items()
                                if k != "buy_rows"},  # buy_table zaten boyutlandırılmış hali
            "tavan_lock": pack["validated_signals"]["tavan_lock"],
            "cluster3": {**pack["validated_signals"]["cluster3"],
                         "open_candidates": pack["validated_signals"]["cluster3"]["open_candidates"][:20]},
        },
        "context_signals": {"status": ctx["status"], "validation": ctx["validation"],
                            "summary": ctx.get("summary"), "per_ticker": ctx.get("per_ticker", {})},
        "macro": pack["macro"],
    }


def synthesize(pack, model=None):
    """Tek yapılandırılmış Claude çağrısı → ham advisory dict."""
    from agent.claude_client import structured_prompt, MODEL_ANALYSIS
    import os

    use_model = model or os.environ.get("NOX_MODEL_ADVISOR") or MODEL_ANALYSIS
    prompt = (
        "Context pack aşağıda. Portföydeki HER pozisyon için tam bir öneri ver, "
        "AL adaylarını seç ve genel değerlendirme yaz.\n\n"
        f"```json\n{json.dumps(_pack_for_llm(pack), ensure_ascii=False, default=str)}\n```"
    )
    result = structured_prompt(prompt, ADVISORY_TOOL,
                               system_prompt=SYSTEM_PROMPT_TR, model=use_model)
    result["mode"] = "llm"
    result["model"] = use_model
    result["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return result


def deterministic_fallback(pack):
    """LLM'siz kural-tabanlı advisory — korkuluk çıktıları aynen rapora döner."""
    pre = pack["pre_check"]
    recs = []
    for p in pre["positions"]:
        if "STOP_VIOLATED" in p["flags"]:
            action, conf = "SELL", "high"
            why = "Stop ihlal edildi (son fiyat < stop) — korkuluk kuralı: SAT/GÖZDEN GEÇİR."
        elif "OVERWEIGHT" in p["flags"]:
            action, conf = "TRIM", "medium"
            why = f"Pozisyon ağırlığı %{p['weight_pct']} > tavan %{pack['settings']['max_position_pct']} — azalt."
        else:
            action, conf = "HOLD", "low"
            why = "Kural-tabanlı mod: flag yok, varsayılan TUT."
        recs.append({"ticker": p["ticker"], "action": action, "confidence": conf,
                     "signal_refs": [], "rationale_tr": why})

    buys = [{"ticker": c["ticker"],
             "rationale_tr": f"DE v1 {c['section']} (kural-tabanlı seçim, paper-track tek-rejim)."}
            for c in pre["buy_table"]
            if c["cand_status"] in (guardrails.CAND_OK, guardrails.CAND_ADD)]

    de_status = pack["inputs_status"]["decision_engine"]
    narrative = (f"LLM kullanılamadı — kural-tabanlı rapor. DE v1 durumu: {de_status}. "
                 f"Toplam risk %{pre['risk']['total_risk_pct']} "
                 f"(tavan %{pre['risk']['cap_pct']}).")
    return {
        "mode": "deterministic_fallback", "model": None,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "position_recommendations": recs,
        "buy_candidates": buys,
        "narrative_tr": narrative,
    }


def build_advisory(pack, use_llm=True):
    """Sentez + post-validasyon. LLM hatasında fallback — rapor her zaman çıkar."""
    raw = None
    if use_llm:
        try:
            raw = synthesize(pack)
        except Exception as e:
            print(f"⚠️ LLM sentez hatası — deterministik fallback: {e}")
    if raw is None:
        raw = deterministic_fallback(pack)
    return guardrails.post_validate(raw, pack)
