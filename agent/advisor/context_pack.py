"""
NOX Advisor — context pack: LLM çağrısından ÖNCE diske yazılan deterministik girdi.

Aynı pack → karşılaştırılabilir çıktı; tüm sayılar burada, LLM yalnızca
gerekçe/seçim yazar. output/advisor/context_pack_{asof}.json olarak persist edilir.
"""
import json
import os
from pathlib import Path

ADVISOR_DIR = Path(os.environ.get("NOX_OUTPUT_DIR", "output")) / "advisor"


def build_context_pack(asof, portfolio, prices, validated, context, macro, pre):
    """Tüm girdileri tek deterministik dict'te topla."""
    inputs_status = {
        "decision_engine": validated["decision_engine"]["status"],
        "tavan_lock": validated["tavan_lock"]["status"],
        "cluster3": validated["cluster3"]["status"],
        "sector_rotation": validated["sector_rotation"]["status"],
        "context_scanners": context["status"],
        "macro": macro["status"],
        "prices": "OK" if prices else "UNAVAILABLE",
    }
    return {
        "asof": asof,
        "portfolio_rev": portfolio.rev,
        "portfolio_source": portfolio.source,
        "settings": portfolio.data["settings"],
        "inputs_status": inputs_status,
        "pre_check": pre,
        "validated_signals": validated,
        "context_signals": context,
        "macro": macro,
        "prices": prices,
    }


def persist_pack(pack):
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"context_pack_{pack['asof']}.json"
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


def persist_advisory(advisory):
    ADVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ADVISOR_DIR / f"advisory_{advisory['asof']}.json"
    path.write_text(json.dumps(advisory, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path
