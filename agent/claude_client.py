"""
NOX Agent — Claude API Client
Anthropic SDK ile tool use loop.
"""
import os
import json
from anthropic import Anthropic

from agent.prompts import SYSTEM_PROMPT
from agent.tool_definitions import TOOLS

# Maliyet optimizasyonu
MAX_HISTORY = 10
MODEL_CHAT = "claude-haiku-4-5-20251001"      # Bot chat — ucuz, hızlı
MODEL_BRIEFING = "claude-sonnet-4-5-20250929"  # Brifing — dengeli
MODEL_ANALYSIS = "claude-sonnet-4-5-20250929"  # Detaylı analiz


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil")
    return Anthropic(api_key=api_key)


def chat(messages, tool_handler, system_prompt=None, max_turns=10,
         model=None):
    """
    Claude ile tool use loop.

    Args:
        messages: Mesaj listesi [{role, content}, ...]
        tool_handler: Tool çağrılarını işleyen fonksiyon (name, input) → result
        system_prompt: Özel sistem promptu (None ise varsayılan)
        max_turns: Maksimum tool use döngüsü
        model: Model seçimi (None ise MODEL_CHAT)

    Returns:
        str: Claude'un son yanıtı
    """
    client = _get_client()
    sys_prompt = system_prompt or SYSTEM_PROMPT
    use_model = model or MODEL_CHAT

    # Mesaj geçmişini sınırla
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]

    for turn in range(max_turns):
        response = client.messages.create(
            model=use_model,
            max_tokens=4096,
            system=sys_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Tool use var mı kontrol et
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            # Sadece metin yanıt — döngüyü bitir
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts)

        # Tool çağrılarını işle
        # Önce assistant mesajını ekle (Pydantic → dict, SDK serializasyon hatası önleme)
        messages.append({"role": "assistant", "content": [
            b.model_dump() if hasattr(b, 'model_dump') else b
            for b in response.content
        ]})

        tool_results = []
        for tool_use in tool_uses:
            try:
                result = tool_handler(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    # Max turn'e ulaşıldı
    return "⚠️ Maksimum işlem adımına ulaşıldı. Lütfen sorunuzu daraltın."


def single_prompt(prompt, tool_handler=None, system_prompt=None,
                  max_tokens=4096, model=None):
    """Tek seferlik prompt gönder, yanıt al.

    model: None ise MODEL_BRIEFING kullanılır.
    """
    use_model = model or MODEL_BRIEFING
    messages = [{"role": "user", "content": prompt}]
    if tool_handler:
        return chat(messages, tool_handler, system_prompt, model=use_model)

    client = _get_client()
    sys_prompt = system_prompt or SYSTEM_PROMPT

    response = client.messages.create(
        model=use_model,
        max_tokens=max_tokens,
        system=sys_prompt,
        messages=messages,
    )

    text_parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_parts)


def analyze_image(image_path, prompt="Bu görseli analiz et."):
    """Claude vision ile görsel analiz.
    Kademe/takas ekran görüntüsü, grafik, tablo fotoğrafları için.
    """
    import base64
    import mimetypes

    client = _get_client()

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    response = client.messages.create(
        model=MODEL_ANALYSIS,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )

    text_parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_parts)
