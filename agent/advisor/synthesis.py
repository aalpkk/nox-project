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
8. Türkçe yaz, kısa ve operasyonel ol.
9. ÇIKTI DİSİPLİNİ: önce position_recommendations'ı yaz (her pozisyon için MUTLAKA bir
   kayıt), sonra buy_candidates, en son narrative. Her rationale_tr en fazla 1-2 cümle
   (~200 karakter) — uzun gerekçe çıktının kesilmesine ve alan kaybına yol açar."""


MAX_BLOCKED_FOR_LLM = 10   # bloklu adaylardan LLM'e giden örnek sayısı
MAX_WATCH_FOR_LLM = 10
_DROP_CAND_FIELDS = ("reason_codes",)  # uzun, karar LLM'in değil — token tasarrufu


def _slim_cand(c):
    return {k: v for k, v in c.items() if k not in _DROP_CAND_FIELDS}


def _pack_for_llm(pack):
    """Pack'in LLM'e giden kompakt hâli — korkuluk hesapları TAM pack'i görür,
    burası yalnızca prompt kopyasını inceltir (token maliyeti).

    buy_table: korkuluğu geçenler (OK/ADD) tamamı + bloklulardan ilk N örnek
    (cand_status görünür kalır ki model 'neden seçilmedi'yi bilsin)."""
    ctx = pack["context_signals"]
    table = pack["pre_check"]["buy_table"]
    passed = [_slim_cand(c) for c in table if c["cand_status"] in ("OK", "ADD")]
    blocked = [_slim_cand(c) for c in table
               if c["cand_status"] not in ("OK", "ADD")][:MAX_BLOCKED_FOR_LLM]
    n_blocked_total = sum(1 for c in table if c["cand_status"] not in ("OK", "ADD"))

    de = pack["validated_signals"]["decision_engine"]
    de_slim = {k: v for k, v in de.items() if k not in ("buy_rows", "watch_rows")}
    de_slim["watch_rows"] = [_slim_cand(r) for r in de.get("watch_rows", [])[:MAX_WATCH_FOR_LLM]]
    de_slim["n_watch_total"] = len(de.get("watch_rows", []))

    return {
        "asof": pack["asof"],
        "inputs_status": pack["inputs_status"],
        "settings": pack["settings"],
        "portfolio": {
            "summary": {k: pack["pre_check"][k] for k in
                        ("equity_tl", "cash_tl", "invested_pct", "n_positions")},
            # BIST-kolu dışı varlıklar: SADECE anlatı bağlamı, boyutlandırmaya girmez
            "other_assets": pack["pre_check"].get("other_assets", {}),
            "risk": pack["pre_check"]["risk"],
            "positions": pack["pre_check"]["positions"],
        },
        "buy_table": passed + blocked,
        "n_blocked_not_shown": max(0, n_blocked_total - len(blocked)),
        "validated_signals": {
            "decision_engine": de_slim,
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


def synthesize_via_cli(pack, model=None):
    """Headless Claude Code (`claude -p`) ile sentez — API key'siz, abonelikle.

    tool_choice zorlaması CLI'da yok → model SALT-JSON çıktıya yönlendirilir;
    bozuk çıktı zaten post_validate'te yakalanır (auto-HOLD/restamp/drop).
    """
    import os
    import shutil
    import subprocess

    binary = shutil.which("claude")
    if not binary:
        raise RuntimeError("claude CLI bulunamadı (PATH)")
    use_model = model or os.environ.get("NOX_ADVISOR_CLI_MODEL", "opus")

    prompt = (
        SYSTEM_PROMPT_TR
        + "\n\nContext pack aşağıda. Portföydeki HER pozisyon için tam bir öneri ver, "
          "AL adaylarını seç ve genel değerlendirme yaz.\n\n"
        + f"```json\n{json.dumps(_pack_for_llm(pack), ensure_ascii=False, default=str)}\n```\n\n"
        + "ÇIKTI KURALI: YALNIZCA tek bir JSON nesnesi yaz (öncesinde/sonrasında metin, "
          "açıklama veya kod bloğu işareti OLMASIN). Şema:\n"
        + json.dumps(ADVISORY_TOOL["input_schema"], ensure_ascii=False)
    )
    # not: tool kısıtlamaya gerek yok — headless'ta izinsiz tool çağrıları
    # zaten reddedilir, model düz yanıta döner; istenen salt-JSON üretimi
    proc = subprocess.run(
        [binary, "-p", "--model", use_model, "--output-format", "json"],
        input=prompt, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p hata ({proc.returncode}): {proc.stderr[:300]}")
    envelope = json.loads(proc.stdout)
    text = envelope.get("result", "") if isinstance(envelope, dict) else str(envelope)
    result = _extract_json(text)
    result["mode"] = "llm"
    result["model"] = f"claude-code-cli/{use_model}"
    result["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return result


def _extract_json(text):
    """Serbest metinden ilk dengeli JSON nesnesini çıkar (kod-bloğu toleranslı)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text[4:] if text.startswith("json") else text
    start = text.find("{")
    if start < 0:
        raise ValueError("çıktıda JSON nesnesi yok")
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("dengesiz JSON")


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


def resolve_llm_mode(llm_mode="auto"):
    """auto: API key varsa api, yoksa claude CLI varsa cli, o da yoksa none."""
    import os
    import shutil
    if llm_mode != "auto":
        return llm_mode
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if shutil.which("claude"):
        return "cli"
    return "none"


def _persist_raw(raw, asof):
    """Ham LLM çıktısını debug için sakla (post_validate ÖNCESİ hali)."""
    try:
        from agent.advisor.context_pack import ADVISOR_DIR
        ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
        path = ADVISOR_DIR / f"raw_llm_{asof}.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    except Exception as e:
        print(f"⚠️ raw persist hatası: {e}")


def build_advisory(pack, use_llm=True, llm_mode="auto"):
    """Sentez + post-validasyon. LLM hatasında fallback — rapor her zaman çıkar.

    llm_mode: auto | api (Anthropic API) | cli (headless claude -p, abonelik) | none
    """
    raw = None
    mode = resolve_llm_mode(llm_mode) if use_llm else "none"
    if mode == "api":
        try:
            raw = synthesize(pack)
        except Exception as e:
            print(f"⚠️ API sentez hatası — deterministik fallback: {e}")
    elif mode == "cli":
        try:
            raw = synthesize_via_cli(pack)
        except Exception as e:
            print(f"⚠️ CLI sentez hatası — deterministik fallback: {e}")
    if raw is not None:
        _persist_raw(raw, pack["asof"])
    if raw is None:
        raw = deterministic_fallback(pack)
    return guardrails.post_validate(raw, pack)
