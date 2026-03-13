"""
NOX Agent — Telegram Bot
python-telegram-bot v21+ async, webhook veya polling mode.

Komutlar:
    /brifing  — Son piyasa brifing'i
    /analiz THYAO — Detaylı hisse analizi
    /watchlist — Watchlist görüntüle
    /ekle THYAO 45.50 — Watchlist'e pozisyon ekle
    /cikar THYAO — Watchlist'ten çıkar
    /tavsiye — Al/Sat tavsiyeleri
    /makro — Makro özet
    /yardim — Komut listesi
    Serbest metin — Claude API ile yanıt

Kullanım:
    python -m agent.bot                    # polling mode (lokal geliştirme)
    python -m agent.bot --webhook URL      # webhook mode (Railway)
"""
import argparse
import logging
import os
import sys

# Proje kökünü path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes,
)

from agent.tools import handle_tool, invalidate_cache
from agent.state import Watchlist
from agent.macro import fetch_macro_snapshot, assess_macro_regime, format_macro_summary
from agent.confluence import calc_all_confluence, format_confluence_summary
from agent.scanner_reader import get_latest_signals, summarize_signals

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Yetkilendirilmiş chat ID'leri
ALLOWED_CHAT_IDS = set()
_chat_id_env = os.environ.get("TG_CHAT_ID", "")
if _chat_id_env:
    ALLOWED_CHAT_IDS.update(_chat_id_env.split(","))


def _authorized(update: Update) -> bool:
    """Chat ID kontrolü."""
    if not ALLOWED_CHAT_IDS:
        return True  # Kısıtlama yok
    return str(update.effective_chat.id) in ALLOWED_CHAT_IDS


async def _send_long(update: Update, text: str):
    """4000 char chunk'larla mesaj gönder."""
    for i in range(0, len(text), 4000):
        chunk = text[i:i + 4000]
        try:
            await update.message.reply_text(
                chunk, parse_mode='HTML', disable_web_page_preview=True)
        except Exception:
            # HTML parse hatası — düz metin gönder
            await update.message.reply_text(chunk)


# ── Komut Handler'ları ──

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text(
        "⬡ <b>NOX Piyasa Analiz Asistanı</b>\n\n"
        "Komutlar için /yardim yazın.\n"
        "Veya doğrudan soru sorun.",
        parse_mode='HTML')


async def cmd_yardim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text(
        "<b>⬡ NOX Agent Komutları</b>\n\n"
        "/brifing — Son piyasa brifing'i\n"
        "/analiz THYAO — Detaylı hisse analizi\n"
        "/watchlist — Watchlist görüntüle\n"
        "/ekle THYAO 45.50 — Pozisyon ekle\n"
        "/cikar THYAO — Pozisyon çıkar\n"
        "/tavsiye — Al/Sat tavsiyeleri\n"
        "/makro — Makro piyasa özeti\n"
        "/model — Model seç (haiku/sonnet/opus)\n"
        "/yardim — Bu mesaj\n\n"
        "📎 Excel/CSV yükle → kademe/takas analizi\n"
        "📷 Fotoğraf yükle → görsel analiz\n"
        "💬 Serbest metin yazarak da soru sorabilirsiniz.",
        parse_mode='HTML')


async def cmd_brifing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text("⏳ Brifing hazırlanıyor...")

    try:
        from agent.briefing import run_briefing
        result = run_briefing(notify=False, use_ai=True)
        briefing = result.get("briefing", "Brifing üretilemedi.")
        await _send_long(update, briefing)
    except Exception as e:
        logger.error(f"Brifing hatası: {e}")
        # Fallback: AI'sız brifing
        try:
            from agent.briefing import run_briefing
            result = run_briefing(notify=False, use_ai=False)
            await _send_long(update, result.get("briefing", str(e)))
        except Exception as e2:
            await update.message.reply_text(f"⚠️ Brifing hatası: {e2}")


async def cmd_analiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Kullanım: /analiz THYAO")
        return

    ticker = args[0].upper().strip()
    await update.message.reply_text(f"⏳ {ticker} analiz ediliyor...")

    try:
        result = handle_tool("get_stock_analysis", {"ticker": ticker})
        if "error" in result:
            await update.message.reply_text(f"⚠️ {result['error']}")
            return

        # Claude ile detaylı analiz
        from agent.claude_client import single_prompt
        from agent.prompts import ANALYSIS_PROMPT
        import json

        prompt = ANALYSIS_PROMPT + f"\n\n## Veri\n```json\n{json.dumps(result, ensure_ascii=False, default=str)}\n```"
        analysis = single_prompt(prompt)
        await _send_long(update, analysis)
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        # Fallback: tool verisi
        try:
            result = handle_tool("get_stock_analysis", {"ticker": ticker})
            conf = result.get("confluence", {})
            lines = [
                f"<b>⬡ {ticker} Analiz</b>",
                f"Skor: {conf.get('score', '?')} — {conf.get('recommendation', '?')}",
            ]
            for d in conf.get("details", []):
                lines.append(f"  {d}")
            price = result.get("price", {})
            if price.get("price"):
                lines.append(f"\nFiyat: {price['price']} ({price.get('change_pct', 0):+.1f}%)")
            await _send_long(update, "\n".join(lines))
        except Exception as e2:
            await update.message.reply_text(f"⚠️ Hata: {e2}")


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    wl = Watchlist()
    await _send_long(update, wl.format_watchlist())


async def cmd_ekle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Kullanım: /ekle THYAO [fiyat] [stop] [hedef]")
        return

    ticker = args[0].upper().strip()
    entry_price = float(args[1]) if len(args) > 1 else None
    stop_price = float(args[2]) if len(args) > 2 else None
    target_price = float(args[3]) if len(args) > 3 else None

    wl = Watchlist()
    wl.add_position(ticker, entry_price, stop_price, target_price)
    await update.message.reply_text(f"✅ {ticker} watchlist'e eklendi")


async def cmd_cikar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Kullanım: /cikar THYAO")
        return
    ticker = args[0].upper().strip()
    wl = Watchlist()
    wl.remove_position(ticker)
    await update.message.reply_text(f"✅ {ticker} watchlist'ten çıkarıldı")


async def cmd_tavsiye(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text("⏳ Tavsiyeler hesaplanıyor...")

    try:
        signals, _ = get_latest_signals()
        macro = None
        try:
            snapshot = fetch_macro_snapshot()
            macro = assess_macro_regime(snapshot)
        except Exception:
            pass

        results = calc_all_confluence(signals, macro, min_score=3)
        if not results:
            await update.message.reply_text("Güçlü çakışma sinyali bulunamadı.")
            return
        await _send_long(update, format_confluence_summary(results, top_n=10))
    except Exception as e:
        await update.message.reply_text(f"⚠️ Hata: {e}")


async def cmd_makro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text("⏳ Makro veri çekiliyor...")

    try:
        snapshot = fetch_macro_snapshot()
        result = assess_macro_regime(snapshot)
        await _send_long(update, format_macro_summary(result))
    except Exception as e:
        await update.message.reply_text(f"⚠️ Makro hatası: {e}")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Model seçimi: /model [haiku|sonnet|opus] [chat|analiz|all]"""
    if not _authorized(update):
        return

    from agent.claude_client import (
        set_model, get_model_status, MODEL_ALIASES, MODEL_INFO,
    )

    args = context.args
    if not args:
        # Mevcut durumu göster
        await update.message.reply_text(
            "<b>⬡ Aktif Modeller</b>\n\n"
            f"{get_model_status()}\n\n"
            "<b>Kullanım:</b>\n"
            "/model haiku — Tümü Haiku (en ucuz)\n"
            "/model sonnet — Tümü Sonnet (dengeli)\n"
            "/model opus — Tümü Opus (en iyi, pahalı)\n"
            "/model opus analiz — Sadece analiz Opus\n"
            "/model sonnet chat — Sadece chat Sonnet",
            parse_mode='HTML')
        return

    model_name = args[0].lower().strip()
    if model_name not in MODEL_ALIASES:
        await update.message.reply_text(
            f"⚠️ Bilinmeyen model: {model_name}\n"
            "Seçenekler: haiku, sonnet, opus")
        return

    model_id = MODEL_ALIASES[model_name]
    display_name, cost = MODEL_INFO[model_id]

    # Rol belirleme
    role = "all"
    if len(args) > 1:
        role_arg = args[1].lower().strip()
        role_map = {"chat": "chat", "analiz": "analysis", "analysis": "analysis",
                    "brifing": "briefing", "briefing": "briefing", "all": "all",
                    "hepsi": "all", "tum": "all"}
        role = role_map.get(role_arg, "all")

    set_model(role, model_id)

    role_display = {"all": "Tümü", "chat": "Chat", "analysis": "Analiz", "briefing": "Brifing"}
    await update.message.reply_text(
        f"✅ <b>{role_display.get(role, role)}</b> → {display_name} ({cost})\n\n"
        f"{get_model_status()}",
        parse_mode='HTML')


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Serbest metin → Claude API + tool use."""
    if not _authorized(update):
        return
    user_text = update.message.text
    if not user_text or user_text.startswith('/'):
        return

    await update.message.reply_text("⏳ Düşünüyorum...")

    try:
        from agent.claude_client import chat
        from agent.state import Watchlist

        wl = Watchlist()
        chat_id = str(update.effective_chat.id)

        # Önceki mesajları al
        history = wl.get_recent_messages(chat_id, limit=6)
        messages = history + [{"role": "user", "content": user_text}]

        response = chat(messages, handle_tool)

        # Mesajları kaydet
        wl.save_message(chat_id, "user", user_text)
        wl.save_message(chat_id, "assistant", response)

        await _send_long(update, response)
    except Exception as e:
        logger.error(f"Chat hatası: {e}")
        await update.message.reply_text(
            f"⚠️ Yanıt üretilemedi: {e}\n\n"
            "Komutları kullanmayı deneyin: /yardim")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Excel/CSV dosya yüklemesi → kademe veya takas analizi."""
    if not _authorized(update):
        return

    doc = update.message.document
    if not doc:
        return

    fname = doc.file_name or ""
    caption = update.message.caption or ""

    # Desteklenen formatlar
    if not any(fname.lower().endswith(ext) for ext in ('.xlsx', '.xls', '.csv')):
        await update.message.reply_text(
            "⚠️ Desteklenen formatlar: .xlsx, .xls, .csv\n"
            "Kademe veya takas Excel'i yükleyin.")
        return

    await update.message.reply_text(f"⏳ {fname} analiz ediliyor...")

    # Dosyayı indir
    import tempfile
    try:
        file = await doc.get_file()
        tmp_dir = os.path.join(ROOT, 'output', 'uploads')
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, fname)
        await file.download_to_drive(local_path)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Dosya indirme hatası: {e}")
        return

    # Ticker tespiti: dosya adı veya caption'dan
    ticker = ""
    for part in (fname.split('.')[0] + " " + caption).upper().split():
        part = part.replace('.IS', '').strip()
        if len(part) >= 3 and part.isalpha():
            ticker = part
            break

    # Analiz tipi: caption veya dosya adına göre
    text_lower = (fname + " " + caption).lower()
    if 'takas' in text_lower or 'kurum' in text_lower or 'custody' in text_lower:
        analysis_type = 'takas'
    elif 'kademe' in text_lower or 'depth' in text_lower or 'sa' in text_lower:
        analysis_type = 'kademe'
    else:
        # İçeriğe bak
        try:
            import pandas as pd
            df_peek = pd.read_excel(local_path) if fname.endswith(('.xlsx', '.xls')) else pd.read_csv(local_path)
            cols_lower = [str(c).lower() for c in df_peek.columns]
            if any('kurum' in c or 'broker' in c or 'aracı' in c for c in cols_lower):
                analysis_type = 'takas'
            else:
                analysis_type = 'kademe'
        except Exception:
            analysis_type = 'kademe'

    try:
        if analysis_type == 'takas':
            result = handle_tool("analyze_takas", {"file_path": local_path, "ticker": ticker})
        else:
            result = handle_tool("analyze_kademe", {"file_path": local_path, "ticker": ticker})

        if "error" in result:
            await update.message.reply_text(f"⚠️ {result['error']}")
            return

        # Claude ile detaylı yorum — takas/kademe için özel prompt
        try:
            from agent.claude_client import single_prompt, MODEL_ANALYSIS
            from agent.prompts import TAKAS_ANALYSIS_PROMPT, KADEME_ANALYSIS_PROMPT
            import json

            if analysis_type == 'takas':
                sys_prompt = TAKAS_ANALYSIS_PROMPT
            else:
                sys_prompt = KADEME_ANALYSIS_PROMPT

            # Sinyal çapraz bilgisi ekle
            cross_info = ""
            if ticker:
                try:
                    ticker_signals = handle_tool("get_stock_analysis", {"ticker": ticker})
                    if "error" not in ticker_signals:
                        conf = ticker_signals.get("confluence", {})
                        cross_info = (
                            f"\n\n## Scanner Sinyal Çakışması — {ticker}\n"
                            f"Skor: {conf.get('score', '?')}, "
                            f"Tavsiye: {conf.get('recommendation', '?')}\n"
                            f"Sinyaller: {json.dumps(ticker_signals.get('signals', []), ensure_ascii=False, default=str)}\n"
                        )
                except Exception:
                    pass

            prompt = (f"## Veri\n```json\n{json.dumps(result, ensure_ascii=False, default=str)}\n```"
                      f"{cross_info}")
            analysis = single_prompt(prompt, system_prompt=sys_prompt,
                                     model=MODEL_ANALYSIS)
            await _send_long(update, analysis)
        except Exception:
            # Fallback: ham sonuç
            await _send_long(update, _format_analysis_result(result, analysis_type))
    except Exception as e:
        logger.error(f"Dosya analiz hatası: {e}")
        await update.message.reply_text(f"⚠️ Analiz hatası: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fotoğraf yüklemesi → Claude vision ile analiz."""
    if not _authorized(update):
        return

    photos = update.message.photo
    if not photos:
        return

    raw_caption = update.message.caption or ""
    await update.message.reply_text("⏳ Görsel analiz ediliyor...")

    # En yüksek çözünürlüklü fotoğrafı al
    photo = photos[-1]
    try:
        file = await photo.get_file()
        tmp_dir = os.path.join(ROOT, 'output', 'uploads')
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, f"photo_{photo.file_id[:8]}.jpg")
        await file.download_to_drive(local_path)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Fotoğraf indirme hatası: {e}")
        return

    try:
        from agent.claude_client import analyze_image
        from agent.prompts import TAKAS_IMAGE_PROMPT

        # Takas ekran görüntüsü tespiti
        caption_lower = raw_caption.lower()
        if any(k in caption_lower for k in ('takas', 'kurum', 'custody', 'akd')):
            prompt = TAKAS_IMAGE_PROMPT
            if raw_caption:
                prompt += f"\n\nKullanıcı notu: {raw_caption}"
        else:
            prompt = raw_caption or "Bu görseli analiz et."

        response = analyze_image(local_path, prompt)
        await _send_long(update, response)
    except Exception as e:
        logger.error(f"Görsel analiz hatası: {e}")
        await update.message.reply_text(f"⚠️ Görsel analiz hatası: {e}")


def _format_analysis_result(result, analysis_type):
    """Ham analiz sonucunu metin formatına dönüştür (fallback)."""
    lines = []
    if analysis_type == 'takas':
        ticker = result.get('ticker', '?')
        lines.append(f"<b>⬡ Takas Analizi{' — ' + ticker if ticker else ''}</b>")
        lines.append(f"Toplam kurum: {result.get('toplam_kurum', 0)}")
        lines.append(f"Yabancı: {result.get('yabanci_count', 0)} "
                      f"(alıcı: {result.get('yabanci_alici', 0)}, "
                      f"satıcı: {result.get('yabanci_satici', 0)})")
        net = result.get('net_yabanci_lot', 0)
        lines.append(f"Net yabancı: {net:+,} lot")
        for u in result.get('uyarilar', []):
            lines.append(u)
        lines.append("\n<b>Top Alıcılar:</b>")
        for k in result.get('top_alici', [])[:5]:
            fark = k.get('lot_fark', 0)
            lines.append(f"  {k['kurum']} ({k['tip']}): {fark:+,}")
        lines.append("\n<b>Top Satıcılar:</b>")
        for k in result.get('top_satici', [])[:5]:
            fark = k.get('lot_fark', 0)
            lines.append(f"  {k['kurum']} ({k['tip']}): {fark:+,}")
    else:
        lines.append(f"<b>⬡ Kademe Analizi</b>")
        fmt = result.get('format', 'unknown')
        if fmt == 'scanner_csv':
            lines.append(f"Toplam: {result.get('total', 0)} hisse")
            lines.append(f"Güçlü AL: {result.get('guclu_al_count', 0)}")
            lines.append(f"Dikkat: {result.get('dikkat_count', 0)}")
            lines.append(f"Kilitli: {result.get('kilitli_count', 0)}")
            for r in result.get('results', [])[:10]:
                lines.append(f"  {r['ticker']}: S/A={r['sa_ort']} ({r['karar']})")
        elif fmt == 'excel':
            lines.append(f"S/A: {result.get('sa_ratio', '?')} — {result.get('karar_aciklama', '')}")
            for dd in result.get('destek_direnc', []):
                lines.append(f"  {dd['tip']}: {dd['fiyat']} ({dd['lot']:,} lot)")
            tavan = result.get('tavan')
            if tavan:
                lines.append(f"\nTavan: {tavan['fiyat']} — {tavan['yorum']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='NOX Agent Telegram Bot')
    parser.add_argument('--webhook', type=str, default=None,
                        help='Webhook URL (Railway deploy)')
    parser.add_argument('--port', type=int, default=8443,
                        help='Webhook port')
    args = parser.parse_args()

    token = os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("⚠️ TG_BOT_TOKEN tanımlı değil!")
        sys.exit(1)

    app = Application.builder().token(token).build()

    # Komut handler'ları
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("yardim", cmd_yardim))
    app.add_handler(CommandHandler("help", cmd_yardim))
    app.add_handler(CommandHandler("brifing", cmd_brifing))
    app.add_handler(CommandHandler("analiz", cmd_analiz))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("ekle", cmd_ekle))
    app.add_handler(CommandHandler("cikar", cmd_cikar))
    app.add_handler(CommandHandler("tavsiye", cmd_tavsiye))
    app.add_handler(CommandHandler("makro", cmd_makro))
    app.add_handler(CommandHandler("model", cmd_model))

    # Serbest metin handler
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_text))

    # Dosya ve fotoğraf handler'ları
    app.add_handler(MessageHandler(
        filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(
        filters.PHOTO, handle_photo))

    if args.webhook:
        print(f"🌐 Webhook mode: {args.webhook}")
        app.run_webhook(
            listen="0.0.0.0",
            port=args.port,
            url_path=token,
            webhook_url=f"{args.webhook}/{token}",
        )
    else:
        print("📡 Polling mode (lokal geliştirme)")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
