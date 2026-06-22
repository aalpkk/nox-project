"""
BIST TAVAN TARAYICI — OTOMATİK TRADE SİSTEMİ
===============================================
Risk yönetimi, Telegram bildirimi, günlük otomatik çalıştırma.

Modüller:
  1) Risk Yönetimi — Pozisyon boyutlama, stop-loss, portföy takibi
  2) Telegram Bot — Sinyal bildirimlerini gönderir
  3) Scheduler — Günlük 18:30'da otomatik çalıştırma

Kurulum:
  1) pip install yfinance pandas numpy requests beautifulsoup4 python-telegram-bot schedule
  2) Telegram @BotFather'dan bot oluştur, token al
  3) Bu dosyadaki CONFIG bölümünü doldur
  4) python bist_auto_trader.py --setup    (Telegram test + chat_id bul)
  5) python bist_auto_trader.py            (manuel çalıştır)
  6) python bist_auto_trader.py --daemon   (arka planda sürekli çalış)

Gereksinimler:
  pip install yfinance pandas numpy requests beautifulsoup4 schedule
  + bist_tavan_scanner_v2.py aynı klasörde olmalı
"""

import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import base64
import requests
import pandas as pd

# Scanner import
from bist_tavan_scanner_v2 import run_scanner


# ============================================================
# KONFİGÜRASYON
# ============================================================

CONFIG_FILE = "bist_config.json"

DEFAULT_CONFIG = {
    # ---- PORTFÖY ----
    "portfoy_toplam": 2_000_000,     # Toplam sermaye (TL)
    "maks_pozisyon_pct": 15,         # Tek pozisyon max %portföy
    "maks_pozisyon_adet": 4,         # Aynı anda max açık pozisyon
    "likit_olmayan_pct": 7.5,        # Likidite uyarısı varsa max %
    
    # ---- SKOR EŞİKLERİ ----
    "devam_al_esik": 60,             # Bu skor ve üstü → AL sinyali
    "devam_guclu_esik": 80,          # Bu skor ve üstü → GÜÇLÜ AL
    "pretavan_izle_esik": 50,        # Pre-tavan watchlist eşiği
    
    # ---- RİSK YÖNETİMİ ----
    "stop_loss_pct": -5.0,           # Stop-loss (%)
    "trailing_stop_pct": -3.0,       # Trailing stop (tavandan düşüş %)
    "gap_down_kes_pct": -3.0,        # Bu gap ve altı → hemen kes
    "maks_gunluk_kayip_pct": -3.0,   # Günlük max portföy kaybı
    
    # ---- TELEGRAM ----
    "telegram_token": "",            # @BotFather'dan alınan token
    "telegram_chat_id": "",          # Mesaj gönderilecek chat ID
    
    # ---- ZAMANLAMA ----
    "scan_saati": "18:35",           # Günlük tarama saati (HH:MM)
    "pazar_aktif": False,            # Pazar günü de çalışsın mı
    
    # ---- DOSYA YOLLARI ----
    "portfoy_dosya": "portfoy.json",       # Açık pozisyonlar
    "trade_log_dosya": "trade_log.csv",    # Trade geçmişi
    "sinyal_log_dosya": "sinyal_log.csv",  # Sinyal geçmişi
}


def load_config() -> dict:
    """Config dosyasını yükler, yoksa oluşturur."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
            # Eksik key'leri default'tan doldur
            config = {**DEFAULT_CONFIG, **saved}
            return config
    else:
        save_config(DEFAULT_CONFIG)
        print(f"⚙️  Config dosyası oluşturuldu: {CONFIG_FILE}")
        print(f"   → Telegram token ve chat_id'yi doldur!")
        return DEFAULT_CONFIG


def save_config(config: dict):
    """Config dosyasını kaydeder."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# 1) RİSK YÖNETİMİ
# ============================================================

class RiskManager:
    """Pozisyon boyutlama ve risk kontrolü."""
    
    def __init__(self, config: dict):
        self.config = config
        self.portfoy = self._load_portfoy()
    
    def pozisyon_boyutu(self, sinyal: dict) -> dict:
        """
        Sinyal için uygun pozisyon boyutunu hesaplar.
        
        Returns:
            {
                'al': True/False,
                'miktar_tl': 200000,
                'sebep': 'Skor 75, güçlü sinyal',
                'risk_notu': '',
            }
        """
        skor = sinyal.get('score', 0)
        ticker = sinyal.get('ticker', '')
        fiyat = sinyal.get('close', 0)
        is_liquid = sinyal.get('is_liquid', True)
        
        toplam = self.config['portfoy_toplam']
        acik_pozisyon = len(self.portfoy.get('pozisyonlar', []))
        maks_adet = self.config['maks_pozisyon_adet']
        
        # Zaten bu hissede pozisyon var mı?
        for p in self.portfoy.get('pozisyonlar', []):
            if p['ticker'] == ticker:
                return {
                    'al': False,
                    'miktar_tl': 0,
                    'sebep': f'Zaten açık pozisyon var ({ticker})',
                    'risk_notu': '',
                }
        
        # Max pozisyon sayısı
        if acik_pozisyon >= maks_adet:
            return {
                'al': False,
                'miktar_tl': 0,
                'sebep': f'Max pozisyon sayısına ulaşıldı ({acik_pozisyon}/{maks_adet})',
                'risk_notu': '',
            }
        
        # Skor eşiği
        if skor < self.config['devam_al_esik']:
            return {
                'al': False,
                'miktar_tl': 0,
                'sebep': f'Skor yetersiz ({skor} < {self.config["devam_al_esik"]})',
                'risk_notu': '',
            }
        
        # Günlük kayıp limiti
        gunluk_pnl = self._gunluk_pnl()
        maks_kayip = toplam * (self.config['maks_gunluk_kayip_pct'] / 100)
        if gunluk_pnl <= maks_kayip:
            return {
                'al': False,
                'miktar_tl': 0,
                'sebep': f'Günlük kayıp limitine ulaşıldı ({gunluk_pnl:,.0f} TL)',
                'risk_notu': '⚠️ Bugün trade yapma',
            }
        
        # Pozisyon boyutu hesapla
        if skor >= self.config['devam_guclu_esik']:
            pct = self.config['maks_pozisyon_pct']
            sinyal_tipi = "GÜÇLÜ AL"
        else:
            pct = self.config['maks_pozisyon_pct'] * 0.67  # %10
            sinyal_tipi = "AL"
        
        # Likidite düşükse yarıla
        risk_notu = ""
        if not is_liquid:
            pct *= 0.5
            risk_notu = "⚠️ Düşük likidite — pozisyon yarılandı"
        
        miktar_tl = toplam * (pct / 100)
        
        # Kullanılabilir sermaye kontrolü
        kullanilan = sum(p.get('maliyet', 0) for p in self.portfoy.get('pozisyonlar', []))
        kalan = toplam - kullanilan
        miktar_tl = min(miktar_tl, kalan)
        
        if miktar_tl < 10_000:
            return {
                'al': False,
                'miktar_tl': 0,
                'sebep': 'Yetersiz sermaye',
                'risk_notu': '',
            }
        
        lot = int(miktar_tl / fiyat) if fiyat > 0 else 0
        
        # Dinamik stop-loss (tavan trade'e uygun — dar stop'lar)
        # Tavan hissesi %10 yukarı gitmiş, %5 geri veriyorsa zaten bozulmuş
        streak = sinyal.get('streak', 1)
        if streak >= 4 or skor >= 80:
            sl_pct = -2.0   # Güçlü seri — tavandan açılmalı
        elif streak >= 2 or skor >= 60:
            sl_pct = -3.0   # Orta seri — küçük gap tolere
        else:
            sl_pct = -4.0   # İlk tavan — biraz daha geniş
        
        stop_loss = fiyat * (1 + sl_pct / 100)
        
        return {
            'al': True,
            'miktar_tl': round(miktar_tl, 0),
            'lot': lot,
            'sinyal_tipi': sinyal_tipi,
            'stop_loss': stop_loss,
            'stop_loss_pct': sl_pct,
            'sebep': f'{sinyal_tipi} — Skor {skor}, {sinyal.get("streak", 0)}. seri',
            'risk_notu': risk_notu,
        }
    
    def cikis_kontrol(self, pozisyon: dict, guncel_fiyat: float) -> dict:
        """
        Açık pozisyon için çıkış kararı.
        
        Returns:
            {'sat': True/False, 'sebep': '...'}
        """
        giris = pozisyon.get('giris_fiyat', 0)
        en_yuksek = pozisyon.get('en_yuksek', giris)
        
        if guncel_fiyat <= 0 or giris <= 0:
            return {'sat': False, 'sebep': ''}
        
        getiri_pct = (guncel_fiyat - giris) / giris * 100
        dusus_tepeden = (guncel_fiyat - en_yuksek) / en_yuksek * 100 if en_yuksek > 0 else 0
        
        # Stop-loss
        if getiri_pct <= self.config['stop_loss_pct']:
            return {
                'sat': True,
                'sebep': f'🔴 STOP-LOSS: {getiri_pct:.1f}% kayıp (limit: {self.config["stop_loss_pct"]}%)',
            }
        
        # Trailing stop (sadece kârdayken)
        if getiri_pct > 0 and dusus_tepeden <= self.config['trailing_stop_pct']:
            return {
                'sat': True,
                'sebep': f'🟡 TRAILING STOP: Tepeden {dusus_tepeden:.1f}% düşüş (kâr: {getiri_pct:.1f}%)',
            }
        
        return {'sat': False, 'sebep': ''}
    
    def pozisyon_ekle(self, ticker: str, fiyat: float, lot: int, skor: int):
        """Yeni pozisyon ekler."""
        maliyet = fiyat * lot
        pozisyon = {
            'ticker': ticker,
            'giris_fiyat': fiyat,
            'lot': lot,
            'maliyet': maliyet,
            'skor': skor,
            'en_yuksek': fiyat,
            'tarih': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        if 'pozisyonlar' not in self.portfoy:
            self.portfoy['pozisyonlar'] = []
        self.portfoy['pozisyonlar'].append(pozisyon)
        self._save_portfoy()
        self._log_trade(ticker, 'AL', fiyat, lot, maliyet, skor)
    
    def pozisyon_kapat(self, ticker: str, fiyat: float, sebep: str):
        """Pozisyonu kapatır."""
        pozisyonlar = self.portfoy.get('pozisyonlar', [])
        yeni = []
        for p in pozisyonlar:
            if p['ticker'] == ticker:
                lot = p['lot']
                giris = p['giris_fiyat']
                satis_tutari = fiyat * lot
                kar = satis_tutari - p['maliyet']
                self._log_trade(ticker, 'SAT', fiyat, lot, satis_tutari, 
                               p.get('skor', 0), kar=kar, sebep=sebep)
            else:
                yeni.append(p)
        self.portfoy['pozisyonlar'] = yeni
        self._save_portfoy()
    
    def portfoy_ozet(self) -> str:
        """Portföy özetini string olarak döner."""
        pozisyonlar = self.portfoy.get('pozisyonlar', [])
        toplam = self.config['portfoy_toplam']
        
        if not pozisyonlar:
            return f"💼 Portföy: {toplam:,.0f} TL (açık pozisyon yok)"
        
        lines = [f"💼 Portföy ({len(pozisyonlar)} pozisyon):"]
        toplam_maliyet = 0
        for p in pozisyonlar:
            toplam_maliyet += p['maliyet']
            lines.append(
                f"  {p['ticker']:8s} | Giriş: {p['giris_fiyat']:.2f} | "
                f"{p['lot']} lot | {p['maliyet']:,.0f} TL | Skor: {p['skor']}"
            )
        
        kalan = toplam - toplam_maliyet
        lines.append(f"  Kullanılan: {toplam_maliyet:,.0f} TL ({toplam_maliyet/toplam*100:.1f}%)")
        lines.append(f"  Kalan nakit: {kalan:,.0f} TL")
        
        return "\n".join(lines)
    
    def _gunluk_pnl(self) -> float:
        """Bugünkü toplam PnL (basitleştirilmiş)."""
        # Gerçek implementasyonda güncel fiyatlarla hesaplanır
        return 0
    
    def _load_portfoy(self) -> dict:
        dosya = self.config.get('portfoy_dosya', 'portfoy.json')
        try:
            if os.path.exists(dosya):
                with open(dosya, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'pozisyonlar': []}
    
    def _save_portfoy(self):
        dosya = self.config.get('portfoy_dosya', 'portfoy.json')
        with open(dosya, 'w', encoding='utf-8') as f:
            json.dump(self.portfoy, f, ensure_ascii=False, indent=2)
    
    def _log_trade(self, ticker, islem, fiyat, lot, tutar, skor, kar=None, sebep=""):
        dosya = self.config.get('trade_log_dosya', 'trade_log.csv')
        header = not os.path.exists(dosya)
        with open(dosya, 'a', encoding='utf-8') as f:
            if header:
                f.write("tarih,ticker,islem,fiyat,lot,tutar,skor,kar,sebep\n")
            kar_str = f"{kar:.2f}" if kar is not None else ""
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')},{ticker},{islem},"
                    f"{fiyat:.2f},{lot},{tutar:.0f},{skor},{kar_str},{sebep}\n")


# ============================================================
# 2) TELEGRAM BOT
# ============================================================

class TelegramBot:
    """Telegram üzerinden sinyal bildirimi."""
    
    def __init__(self, config: dict):
        self.token = config.get('telegram_token', '')
        self.chat_id = config.get('telegram_chat_id', '')
        self.aktif = bool(self.token and self.chat_id)
    
    def mesaj_gonder(self, mesaj: str, parse_mode: str = "HTML") -> bool:
        """Telegram mesajı gönderir. 4096 karakterden uzunsa böler."""
        if not self.aktif:
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        
        # Telegram limiti 4096 — uzunsa böl
        parcalar = self._mesaj_bol(mesaj, 4000)
        
        success = True
        for parca in parcalar:
            payload = {
                "chat_id": self.chat_id,
                "text": parca,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code != 200:
                    # HTML parse hatası — tag'leri temizleyip düz metin gönder
                    import re
                    clean = re.sub(r'<[^>]+>', '', parca)
                    payload2 = {
                        "chat_id": self.chat_id,
                        "text": clean,
                        "disable_web_page_preview": True,
                    }
                    resp = requests.post(url, json=payload2, timeout=10)
                    if resp.status_code != 200:
                        print(f"⚠️ Telegram hatası: {resp.status_code} — {resp.text[:200]}")
                        success = False
            except Exception as e:
                print(f"⚠️ Telegram bağlantı hatası: {e}")
                success = False
            
            # Rate limit — mesajlar arası 0.5s bekle
            if len(parcalar) > 1:
                time.sleep(0.5)
        
        return success
    
    def _mesaj_bol(self, mesaj: str, limit: int = 4000) -> list:
        """Uzun mesajı satır bazında böler."""
        if len(mesaj) <= limit:
            return [mesaj]
        
        parcalar = []
        mevcut = ""
        
        for satir in mesaj.split("\n"):
            if len(mevcut) + len(satir) + 1 > limit:
                if mevcut:
                    parcalar.append(mevcut)
                mevcut = satir
            else:
                mevcut += ("\n" if mevcut else "") + satir
        
        if mevcut:
            parcalar.append(mevcut)
        
        return parcalar
    
    def dokuman_gonder(self, filepath: str) -> bool:
        """HTML dosyasını Telegram'a document olarak gönder."""
        if not self.aktif or not os.path.exists(filepath):
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendDocument"
        try:
            with open(filepath, 'rb') as f:
                requests.post(url, data={
                    "chat_id": self.chat_id,
                    "caption": "📊 Tavan tarayıcı detaylı rapor"
                }, files={"document": f}, timeout=30)
            print("📤 HTML rapor Telegram'a gönderildi")
            return True
        except Exception as e:
            print(f"⚠️ Telegram document hata: {e}")
            return False

    def sinyal_bildir(self, tavan_devam: list, tavan_kandidat: list,
                       risk_manager: RiskManager, html_url: str = None) -> bool:
        """Scanner sonuçlarını Telegram'a gönderir — terminal çıktısının tamamı."""
        if not self.aktif:
            print("⚠️ Telegram yapılandırılmamış — bildirim atlandı")
            return False

        tarih = datetime.now().strftime('%d/%m/%Y %H:%M')

        # ==================== MESAJ 1: TAVAN DEVAM (DETAYLI) ====================
        mesaj = f"🔔 <b>BIST Tavan Tarayıcı</b>\n"
        if html_url:
            mesaj += f'🔗 <a href="{html_url}">Detaylı Rapor</a>\n'
        mesaj += f"📅 {tarih}\n"
        mesaj += f"{'='*40}\n"
        mesaj += f"🔴 <b>TAVAN SERİSİ — YARIN DEVAM EDER Mİ?</b>\n"
        mesaj += f"{'='*40}\n\n"
        
        if tavan_devam:
            for s in sorted(tavan_devam, key=lambda x: x['score'], reverse=True):
                emoji = "🟢" if s['score'] >= 60 else "🟡" if s['score'] >= 40 else "🔴"
                
                # Header
                yab_str = ""
                if s.get('yab_oran') is not None:
                    yab_str = f"  Yab:%{s['yab_oran']:.1f}"
                    if s.get('yab_degisim') is not None:
                        yab_str += f"({s['yab_degisim']:+.2f})"
                
                mesaj += f"{emoji} <b>{s['ticker']}</b>  SKOR: {s['score']}/100\n"
                mesaj += f"   Seri: {s['streak']} gün  RS: {s['rs']:+.1f}%"
                mesaj += f"  Hacim: {s['vol_ratio']:.1f}x"
                mesaj += f"  Fiyat: {s['close']:.2f}{yab_str}\n"
                
                # Reasons
                for reason in s.get('reasons', []):
                    mesaj += f"   {reason}\n"
                
                # Risk değerlendirmesi
                risk = risk_manager.pozisyon_boyutu(s)
                if risk.get('al'):
                    mesaj += f"   💰 → <b>{risk['sinyal_tipi']}: {risk['miktar_tl']:,.0f} TL ({risk['lot']} lot) SL:{risk['stop_loss']:.2f}</b>\n"
                    if risk.get('risk_notu'):
                        mesaj += f"   {risk['risk_notu']}\n"
                
                mesaj += "\n"
        else:
            mesaj += "ℹ️ Bugün tavan yapan hisse yok.\n\n"
        
        # Gönder (otomatik böler)
        ok1 = self.mesaj_gonder(mesaj)
        
        # ==================== MESAJ 2: RİSK DEĞERLENDİRMESİ + PORTFÖY ====================
        mesaj2 = f"{'='*40}\n"
        mesaj2 += f"💰 <b>RİSK DEĞERLENDİRMESİ</b>\n"
        mesaj2 += f"{'='*40}\n\n"
        
        for sinyal in sorted(tavan_devam, key=lambda x: x['score'], reverse=True):
            karar = risk_manager.pozisyon_boyutu(sinyal)
            if karar.get('al'):
                mesaj2 += f"✅ <b>{sinyal['ticker']}</b> → {karar['sinyal_tipi']}\n"
                mesaj2 += f"   Skor: {sinyal['score']}  Seri: {sinyal['streak']}  Fiyat: {sinyal['close']:.2f}\n"
                mesaj2 += f"   Pozisyon: {karar['miktar_tl']:,.0f} TL ({karar['lot']} lot)\n"
                mesaj2 += f"   Stop-Loss: {karar['stop_loss']:.2f} ({karar['stop_loss_pct']}%)\n"
                if karar.get('risk_notu'):
                    mesaj2 += f"   {karar['risk_notu']}\n"
                mesaj2 += "\n"
            elif sinyal['score'] >= 40:
                mesaj2 += f"❌ {sinyal['ticker']} → {karar['sebep']}\n\n"
        
        mesaj2 += f"{'='*40}\n"
        mesaj2 += f"📊 Taranan: 433+ hisse\n"
        mesaj2 += f"🔴 Tavan serisinde: {len(tavan_devam)}\n"
        mesaj2 += f"🟡 Kandidat: {len(tavan_kandidat)}\n\n"
        mesaj2 += risk_manager.portfoy_ozet()
        
        ok2 = self.mesaj_gonder(mesaj2)
        
        # ==================== MESAJ 3: KANDIDATLAR ====================
        mesaj3 = f"{'='*40}\n"
        mesaj3 += f"🟡 <b>TAVAN KANDİDATLARI (Top 20)</b>\n"
        mesaj3 += f"{'='*40}\n\n"
        
        top_pre = sorted(tavan_kandidat, key=lambda x: x['score'], reverse=True)[:20]
        if top_pre:
            for s in top_pre:
                emoji = "🟡" if s['score'] >= 40 else "⚪"
                ret = s.get('returns', 0) * 100
                ret_str = f"+{ret:.1f}%" if ret >= 0 else f"{ret:.1f}%"
                
                mesaj3 += f"{emoji} <b>{s['ticker']}</b>  SKOR: {s['score']}/100  Getiri: {ret_str}\n"
                mesaj3 += f"   RS: {s['rs']:+.1f}%  Hacim: {s['vol_ratio']:.1f}x  Fiyat: {s['close']:.2f}\n"
                
                for reason in s.get('reasons', []):
                    mesaj3 += f"   {reason}\n"
                mesaj3 += "\n"
        else:
            mesaj3 += "ℹ️ Eşik üstü kandidat yok.\n\n"
        
        ok3 = self.mesaj_gonder(mesaj3)
        
        return ok1 and ok2 and ok3
    
    def _sinyal_satir(self, sinyal: dict, risk: dict) -> str:
        """Tek sinyal için Telegram formatı."""
        line = f"  <b>{sinyal['ticker']}</b>"
        line += f" — Skor:{sinyal['score']} Seri:{sinyal['streak']}"
        line += f" Fiyat:{sinyal['close']:.2f}"
        
        if sinyal.get('yab_oran'):
            line += f" Yab:%{sinyal['yab_oran']:.1f}"
        
        if risk.get('al'):
            line += f"\n    → {risk['sinyal_tipi']}: {risk['miktar_tl']:,.0f} TL"
            line += f" ({risk['lot']} lot)"
            line += f" SL:{risk['stop_loss']:.2f}"
            if risk.get('risk_notu'):
                line += f"\n    {risk['risk_notu']}"
        
        line += "\n"
        return line
    
    def test(self) -> bool:
        """Telegram bağlantısını test eder."""
        return self.mesaj_gonder("✅ BIST Tavan Tarayıcı bağlantı testi başarılı!")
    
    def chat_id_bul(self):
        """Bot'a gelen son mesajdan chat_id'yi bulur."""
        if not self.token:
            print("❌ Telegram token boş! Önce @BotFather'dan bot oluştur.")
            return
        
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            
            if data.get('ok') and data.get('result'):
                for update in data['result']:
                    msg = update.get('message', {})
                    chat = msg.get('chat', {})
                    chat_id = chat.get('id', '')
                    username = chat.get('username', '?')
                    print(f"✅ Chat ID bulundu: {chat_id} (kullanıcı: @{username})")
                    print(f"   → Config'e kaydetmek için:")
                    print(f'   "telegram_chat_id": "{chat_id}"')
                    return str(chat_id)
            else:
                print("❌ Henüz bot'a mesaj gelmemiş.")
                print("   → Telegram'da bot'una bir mesaj gönder, sonra tekrar dene.")
        except Exception as e:
            print(f"❌ Hata: {e}")
        return None


# ============================================================
# 3) GÜNLÜK ZAMANLAYICI
# ============================================================

def gunluk_tarama(config: dict):
    """Günlük tarama + bildirim + risk kontrolü."""
    
    print(f"\n{'='*70}")
    print(f"⏰ Otomatik tarama başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")
    
    # Scanner çalıştır
    results = run_scanner(mode="ex_bist100")
    
    tavan_devam = results.get('tavan_devam', [])
    tavan_kandidat = results.get('tavan_kandidat', [])
    
    # Risk yöneticisi
    risk = RiskManager(config)
    
    # AL sinyallerini değerlendir
    print("\n" + "=" * 70)
    print("💰 RİSK DEĞERLENDİRMESİ")
    print("=" * 70)
    
    al_sinyalleri = []
    for sinyal in tavan_devam:
        karar = risk.pozisyon_boyutu(sinyal)
        if karar['al']:
            al_sinyalleri.append((sinyal, karar))
            print(f"\n✅ {sinyal['ticker']:8s} → {karar['sinyal_tipi']}")
            print(f"   Skor: {sinyal['score']}  Seri: {sinyal['streak']}  Fiyat: {sinyal['close']:.2f}")
            print(f"   Pozisyon: {karar['miktar_tl']:,.0f} TL ({karar['lot']} lot)")
            print(f"   Stop-Loss: {karar['stop_loss']:.2f}")
            if karar['risk_notu']:
                print(f"   {karar['risk_notu']}")
        else:
            if sinyal['score'] >= 40:  # Sadece anlamlı olanları göster
                print(f"\n❌ {sinyal['ticker']:8s} → {karar['sebep']}")
    
    if not al_sinyalleri:
        print("\n  ℹ️ Bugün AL koşulunu karşılayan sinyal yok.")
    
    # Portföy özeti
    print(f"\n{risk.portfoy_ozet()}")
    
    # HTML rapor oluştur
    total_scanned = len(results.get('data', {}))
    html_content = generate_tavan_html(tavan_devam, tavan_kandidat, total_scanned, risk)
    date_str = datetime.now().strftime('%Y%m%d')
    html_fname = f"tavan_{date_str}.html"
    with open(html_fname, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n📄 HTML: {html_fname}")

    # GitHub Pages push
    html_url = push_tavan_html_to_github(html_content, date_str)
    if html_url:
        print(f"🔗 {html_url}")

    # Telegram bildirim
    bot = TelegramBot(config)
    if bot.aktif:
        ok = bot.sinyal_bildir(tavan_devam, tavan_kandidat, risk, html_url=html_url)
        if ok:
            print("\n📱 Telegram bildirimi gönderildi ✅")
        else:
            print("\n📱 Telegram bildirimi gönderilemedi ❌")

        # HTML dosyasını da document olarak gönder
        bot.dokuman_gonder(html_fname)

    # Sinyal log
    _log_sinyaller(config, tavan_devam, tavan_kandidat)

    return results


# ============================================================
# 4) HTML RAPOR
# ============================================================

def generate_tavan_html(tavan_devam, tavan_kandidat, total_scanned, risk_manager=None):
    """Tavan tarayıcı sonuçları için HTML dashboard oluştur."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    al_list = [s for s in tavan_devam if s['score'] >= 60]
    izle_list = [s for s in tavan_devam if 40 <= s['score'] < 60]
    dusuk_list = [s for s in tavan_devam if s['score'] < 40]

    def skor_badge(score):
        if score >= 80:
            return f'<span style="background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{score}</span>'
        elif score >= 60:
            return f'<span style="background:#2ecc71;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{score}</span>'
        elif score >= 40:
            return f'<span style="background:#f39c12;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{score}</span>'
        else:
            return f'<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{score}</span>'

    def karar_badge(score, risk_info=None):
        if score >= 60:
            label = "GÜÇLÜ AL" if score >= 80 else "AL"
            extra = ""
            if risk_info and risk_info.get('al'):
                extra = f' — {risk_info["miktar_tl"]:,.0f} TL ({risk_info["lot"]} lot)'
            return f'<span style="background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{label}{extra}</span>'
        elif score >= 40:
            return '<span style="background:#f39c12;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">İZLE</span>'
        else:
            return '<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">ZAYIF</span>'

    def rs_badge(rs_val):
        if rs_val >= 10:
            return f'<span style="color:#27ae60;font-weight:bold">{rs_val:+.1f}%</span>'
        elif rs_val >= 0:
            return f'<span style="color:#2ecc71">{rs_val:+.1f}%</span>'
        elif rs_val >= -10:
            return f'<span style="color:#e67e22">{rs_val:+.1f}%</span>'
        else:
            return f'<span style="color:#e74c3c">{rs_val:+.1f}%</span>'

    def reason_chips(reasons):
        chips = []
        for r in reasons:
            if '✅' in r or '🟢' in r or '💰' in r or '🔒' in r:
                color = '#27ae60'
            elif '⚠️' in r or '🟡' in r:
                color = '#f39c12'
            elif '❌' in r or '🔴' in r or '📉' in r:
                color = '#e74c3c'
            elif '🚀' in r or '💪' in r or '🔥' in r or '💥' in r:
                color = '#3498db'
            else:
                color = '#555'
            chips.append(f'<span style="background:{color}22;color:{color};padding:1px 6px;border-radius:3px;font-size:0.8em;margin:1px;display:inline-block">{r}</span>')
        return ' '.join(chips)

    def devam_row(s, idx):
        tv_url = f"https://www.tradingview.com/chart/?symbol=BIST:{s['ticker']}"
        risk_info = None
        if risk_manager:
            risk_info = risk_manager.pozisyon_boyutu(s)

        yab_str = ""
        if s.get('yab_oran') is not None:
            yab_str = f"%{s['yab_oran']:.1f}"
            if s.get('yab_degisim') is not None:
                deg = s['yab_degisim']
                color = '#27ae60' if deg > 0 else '#e74c3c' if deg < 0 else '#888'
                yab_str += f' <span style="color:{color}">({deg:+.2f})</span>'

        liq_badge = ""
        if not s.get('is_liquid', True):
            liq_badge = ' <span style="background:#e74c3c22;color:#e74c3c;padding:1px 4px;border-radius:3px;font-size:0.75em">DÜŞ.LİK</span>'

        score_val = s['score']
        if score_val >= 60:
            row_bg = "rgba(39,174,96,0.08)"
        elif score_val >= 40:
            row_bg = "rgba(241,196,15,0.06)"
        else:
            row_bg = "rgba(231,76,60,0.06)"

        ema_dist = s.get('ema21_dist', 0)
        ema_color = '#27ae60' if ema_dist <= 20 else '#f39c12' if ema_dist <= 35 else '#e74c3c'
        rsi_val = s.get('rsi', 0)
        rsi_color = '#27ae60' if 80 <= rsi_val < 85 else '#58a6ff' if rsi_val >= 70 else '#888'

        return f'''<tr style="background:{row_bg}" data-score="{score_val}" data-ticker="{s['ticker']}">
  <td>{idx}</td>
  <td><a href="{tv_url}" target="_blank" style="color:#58a6ff;text-decoration:none;font-weight:bold">{s['ticker']}</a>{liq_badge}</td>
  <td style="text-align:center">{skor_badge(score_val)}</td>
  <td style="text-align:center">{karar_badge(score_val, risk_info)}</td>
  <td style="text-align:center;font-weight:bold">{s.get('streak', 0)}</td>
  <td style="text-align:right">{s['close']:.2f}</td>
  <td style="text-align:right">{s.get('vol_ratio', 0):.1f}x</td>
  <td style="text-align:center;color:{ema_color}">{ema_dist:+.1f}%</td>
  <td style="text-align:center;color:{rsi_color}">{rsi_val:.0f}</td>
  <td style="text-align:center">{rs_badge(s.get('rs', 0))}</td>
  <td style="text-align:center">{yab_str or '-'}</td>
  <td style="font-size:0.8em">{reason_chips(s.get('reasons', []))}</td>
</tr>'''

    def kandidat_row(s, idx):
        tv_url = f"https://www.tradingview.com/chart/?symbol=BIST:{s['ticker']}"
        ret = s.get('returns', 0)
        if isinstance(ret, (int, float)):
            ret_pct = ret * 100
            ret_color = '#27ae60' if ret_pct >= 0 else '#e74c3c'
            ret_str = f'<span style="color:{ret_color}">{ret_pct:+.1f}%</span>'
        else:
            ret_str = '-'

        yab_str = ""
        if s.get('yab_oran') is not None:
            yab_str = f"%{s['yab_oran']:.1f}"
            if s.get('yab_degisim') is not None:
                deg = s['yab_degisim']
                color = '#27ae60' if deg > 0 else '#e74c3c' if deg < 0 else '#888'
                yab_str += f' <span style="color:{color}">({deg:+.2f})</span>'

        score_val = s['score']
        if score_val >= 60:
            row_bg = "rgba(39,174,96,0.06)"
        elif score_val >= 40:
            row_bg = "rgba(241,196,15,0.04)"
        else:
            row_bg = ""

        ema_dist = s.get('ema21_dist', 0)
        ema_color = '#27ae60' if ema_dist <= 20 else '#f39c12' if ema_dist <= 35 else '#e74c3c'
        rsi_val = s.get('rsi', 0)
        rsi_color = '#27ae60' if 80 <= rsi_val < 85 else '#58a6ff' if rsi_val >= 70 else '#888'

        return f'''<tr style="background:{row_bg}" data-score="{score_val}" data-ticker="{s['ticker']}">
  <td>{idx}</td>
  <td><a href="{tv_url}" target="_blank" style="color:#58a6ff;text-decoration:none;font-weight:bold">{s['ticker']}</a></td>
  <td style="text-align:center">{skor_badge(score_val)}</td>
  <td style="text-align:right">{s['close']:.2f}</td>
  <td style="text-align:center">{ret_str}</td>
  <td style="text-align:right">{s.get('vol_ratio', 0):.1f}x</td>
  <td style="text-align:center;color:{ema_color}">{ema_dist:+.1f}%</td>
  <td style="text-align:center;color:{rsi_color}">{rsi_val:.0f}</td>
  <td style="text-align:center">{rs_badge(s.get('rs', 0))}</td>
  <td style="text-align:center">{yab_str or '-'}</td>
  <td style="font-size:0.8em">{reason_chips(s.get('reasons', []))}</td>
</tr>'''

    rows_devam = '\n'.join(devam_row(s, i+1) for i, s in enumerate(
        sorted(tavan_devam, key=lambda x: x['score'], reverse=True)))
    rows_kandidat = '\n'.join(kandidat_row(s, i+1) for i, s in enumerate(
        sorted(tavan_kandidat, key=lambda x: x['score'], reverse=True)[:30]))

    html = f'''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BIST Tavan Tarayıcı — {now}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#0d1117; color:#e6edf3; padding:16px; }}
  .header {{ text-align:center; margin-bottom:20px; }}
  .header h1 {{ font-size:1.6em; margin-bottom:4px; }}
  .header .sub {{ color:#8b949e; font-size:0.9em; }}
  .cards {{ display:flex; gap:12px; justify-content:center; margin:16px 0; flex-wrap:wrap; }}
  .card {{ padding:16px 28px; border-radius:10px; text-align:center; min-width:120px; }}
  .card-al {{ background:rgba(39,174,96,0.15); border:1px solid #27ae60; }}
  .card-izle {{ background:rgba(241,196,15,0.15); border:1px solid #f1c40f; }}
  .card-dusuk {{ background:rgba(231,76,60,0.15); border:1px solid #e74c3c; }}
  .card-kand {{ background:rgba(52,152,219,0.15); border:1px solid #3498db; }}
  .card .num {{ font-size:2em; font-weight:bold; }}
  .card .lbl {{ font-size:0.85em; color:#8b949e; margin-top:4px; }}
  .card-al .num {{ color:#27ae60; }}
  .card-izle .num {{ color:#f1c40f; }}
  .card-dusuk .num {{ color:#e74c3c; }}
  .card-kand .num {{ color:#3498db; }}
  .filters {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; align-items:center; justify-content:center; }}
  .filters input {{
    background:#161b22; color:#e6edf3; border:1px solid #30363d;
    padding:6px 10px; border-radius:6px; font-size:0.9em;
  }}
  .section {{ margin:24px 0; }}
  .section h2 {{ font-size:1.2em; margin-bottom:8px; padding-left:4px;
                  border-left:3px solid; padding:4px 8px; }}
  .section-devam h2 {{ border-color:#e74c3c; color:#e74c3c; }}
  .section-kand h2 {{ border-color:#3498db; color:#3498db; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  th {{ background:#161b22; padding:8px 6px; text-align:left; cursor:pointer;
       border-bottom:2px solid #30363d; position:sticky; top:0; z-index:1; }}
  th:hover {{ background:#1c2333; }}
  td {{ padding:6px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:rgba(56,139,253,0.1) !important; }}
  .guide {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
            padding:16px; margin:16px 0; font-size:0.9em; line-height:1.6; }}
  .guide summary {{ cursor:pointer; font-weight:bold; color:#58a6ff; }}
  a {{ color:#58a6ff; }}
  @media(max-width:768px) {{
    table {{ font-size:0.75em; }}
    .card {{ min-width:90px; padding:12px 14px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>BIST Tavan Tarayici</h1>
  <div class="sub">{now} | Taranan: {total_scanned} hisse</div>
</div>

<div class="cards">
  <div class="card card-al"><div class="num">{len(al_list)}</div><div class="lbl">AL (Skor &ge;60)</div></div>
  <div class="card card-izle"><div class="num">{len(izle_list)}</div><div class="lbl">IZLE (40-59)</div></div>
  <div class="card card-dusuk"><div class="num">{len(dusuk_list)}</div><div class="lbl">ZAYIF (&lt;40)</div></div>
  <div class="card card-kand"><div class="num">{len(tavan_kandidat)}</div><div class="lbl">Kandidat</div></div>
</div>

<div class="filters">
  <input id="fTicker" type="text" placeholder="Hisse ara..." oninput="applyFilter()">
</div>

<details class="guide">
  <summary>Skor Rehberi &amp; Forward Test Bulgulari</summary>
  <br>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Skor</th><th style="padding:4px 8px">Karar</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">Not</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.08)">
        <td style="padding:3px 8px;font-weight:bold;color:#27ae60">&ge;60</td>
        <td style="padding:3px 8px">AL</td><td style="color:#27ae60;font-weight:bold">%75+</td>
        <td style="color:#27ae60">+3-5%</td><td style="padding:3px 8px">Guclu seri, hacim ve momentum destekli</td></tr>
      <tr>
        <td style="padding:3px 8px;font-weight:bold;color:#f39c12">40-59</td>
        <td style="padding:3px 8px">IZLE</td><td>%56</td>
        <td>+2%</td><td style="padding:3px 8px">Orta gucte, ek teyit bekle</td></tr>
      <tr>
        <td style="padding:3px 8px;font-weight:bold;color:#e74c3c">&lt;40</td>
        <td style="padding:3px 8px">ZAYIF</td><td style="color:#e74c3c">%35</td>
        <td style="color:#e74c3c">-0.5%</td><td style="padding:3px 8px">Zayif sinyal, uzak dur</td></tr>
    </tbody>
  </table>
  <br>
  <b>Onemli Sinyaller:</b><br>
  <span style="color:#27ae60">hacim_dusus</span>: Dusuk hacimde tavan = guclu talep (WR %84.6)<br>
  <span style="color:#27ae60">kilitli</span>: Kilitli kapanis = ertesi gun tavandan acilma (WR %60.7)<br>
  <span style="color:#e74c3c">RS &lt; 0</span>: Endekse gore zayif performans = risk (WR %34.6)<br>
  <span style="color:#27ae60">yabanci artis</span>: Yabanci oraninda artis = kurumsal talep<br>
  <br>
  <i style="color:#8b949e">Forward test: 10 Feb — 3 Mar 2026, N=229 tavan sinyali</i>
</details>

<details class="guide">
  <summary>Forward Test Detaylari (16 gun, 545 sinyal)</summary>
  <br>
  <b>Genel Performans</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Kategori</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">3G WR</th><th style="padding:4px 8px">3G Ort</th>
      <th style="padding:4px 8px">5G WR</th><th style="padding:4px 8px">5G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;font-weight:bold">Tavan Serisi</td><td>229</td>
        <td style="color:#27ae60;font-weight:bold">%56.8</td><td style="color:#27ae60">+2.02%</td>
        <td>%45.6</td><td>+1.19%</td><td>%43.1</td><td>+0.97%</td></tr>
      <tr><td style="padding:3px 8px;font-weight:bold">Kandidat</td><td>316</td>
        <td style="color:#e74c3c">%41.0</td><td style="color:#e74c3c">-0.63%</td>
        <td style="color:#e74c3c">%36.4</td><td style="color:#e74c3c">-1.70%</td>
        <td style="color:#e74c3c">%37.4</td><td style="color:#e74c3c">-1.72%</td></tr>
    </tbody>
  </table>

  <b>Skor Bantlari — Tavan Serisi</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Skor</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">5G WR</th><th style="padding:4px 8px">5G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.12)"><td style="padding:3px 8px;color:#27ae60;font-weight:bold">&ge;60 (AL)</td><td>14</td>
        <td style="color:#27ae60;font-weight:bold">%75.0</td><td style="color:#27ae60">+3.40%</td>
        <td>%57.1</td><td style="color:#27ae60">+11.68%</td></tr>
      <tr style="background:rgba(39,174,96,0.06)"><td style="padding:3px 8px;color:#2ecc71">60-69</td><td>10</td>
        <td style="color:#27ae60;font-weight:bold">%83.3</td><td style="color:#27ae60">+5.25%</td>
        <td>%66.7</td><td style="color:#27ae60">+15.44%</td></tr>
      <tr><td style="padding:3px 8px;color:#f39c12">50-59</td><td>38</td>
        <td>%58.3</td><td>+1.52%</td><td>%47.1</td><td>+1.27%</td></tr>
      <tr><td style="padding:3px 8px;color:#f39c12">40-49</td><td>111</td>
        <td>%56.2</td><td>+2.00%</td><td>%40.8</td><td>+0.81%</td></tr>
      <tr><td style="padding:3px 8px;color:#e74c3c">30-39</td><td>55</td>
        <td>%54.9</td><td>+2.07%</td><td>%45.0</td><td style="color:#e74c3c">-0.97%</td></tr>
      <tr><td style="padding:3px 8px;color:#e74c3c">&lt;30</td><td>10</td>
        <td>%50.0</td><td>+1.74%</td><td>%37.5</td><td>+3.77%</td></tr>
    </tbody>
  </table>

  <b>Seri Uzunlugu</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Seri</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">3G Ort</th><th style="padding:4px 8px">Devam %</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:3px 8px">1 gun</td><td>164</td><td>%54.3</td><td>+1.81%</td><td>+0.63%</td><td>~%22</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;font-weight:bold">2 gun</td><td>45</td>
        <td style="color:#27ae60;font-weight:bold">%64.5</td><td style="color:#27ae60">+3.22%</td>
        <td style="color:#27ae60">+4.47%</td><td>~%33</td></tr>
      <tr><td style="padding:3px 8px">3 gun</td><td>17</td><td>%63.6</td><td>+1.68%</td><td>-1.05%</td><td>~%41</td></tr>
      <tr><td style="padding:3px 8px">4+ gun</td><td>3</td><td>%66.7</td><td>+3.30%</td><td>+3.28%</td><td>~%59</td></tr>
    </tbody>
  </table>

  <b>Bireysel Sinyaller (Tavan) — En Guclular</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Sinyal</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">Etki</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.1)"><td style="padding:3px 8px;color:#27ae60;font-weight:bold">hacim_dusus</td><td>19</td>
        <td style="color:#27ae60;font-weight:bold">%84.6</td><td style="color:#27ae60">+5.73%</td>
        <td style="color:#27ae60">En guclu sinyal</td></tr>
      <tr style="background:rgba(39,174,96,0.06)"><td style="padding:3px 8px;color:#27ae60">dusuk_hacim</td><td>53</td>
        <td style="color:#27ae60;font-weight:bold">%69.7</td><td style="color:#27ae60">+3.44%</td>
        <td style="color:#27ae60">Kilitli, satici yok</td></tr>
      <tr><td style="padding:3px 8px;color:#27ae60">pozitif_gap</td><td>37</td>
        <td>%61.9</td><td>+2.11%</td><td>Momentum teyidi</td></tr>
      <tr><td style="padding:3px 8px;color:#27ae60">yab_artis</td><td>62</td>
        <td>%60.4</td><td>+2.41%</td><td>Kurumsal talep</td></tr>
      <tr><td style="padding:3px 8px;color:#27ae60">para_girisi</td><td>87</td>
        <td>%61.2</td><td>+2.56%</td><td>CMF pozitif</td></tr>
      <tr style="background:rgba(231,76,60,0.06)"><td style="padding:3px 8px;color:#e74c3c">para_cikisi</td><td>18</td>
        <td style="color:#e74c3c">%42.9</td><td style="color:#e74c3c">+0.43%</td>
        <td style="color:#e74c3c">Fake tavan riski</td></tr>
    </tbody>
  </table>

  <b>RS Bantlari (Tavan)</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">RS</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">3G WR</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(231,76,60,0.08)"><td style="padding:3px 8px;color:#e74c3c;font-weight:bold">&lt; 0</td><td>34</td>
        <td style="color:#e74c3c;font-weight:bold">%34.6</td><td style="color:#e74c3c">-0.46%</td>
        <td style="color:#e74c3c">%29.2</td></tr>
      <tr><td style="padding:3px 8px">0-20%</td><td>98</td><td>%60.5</td><td>+2.50%</td><td>%44.7</td></tr>
      <tr><td style="padding:3px 8px">20-50%</td><td>61</td><td>%56.9</td><td>+2.08%</td><td>%45.7</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;color:#27ae60;font-weight:bold">50-100%</td><td>28</td>
        <td style="color:#27ae60;font-weight:bold">%68.8</td><td style="color:#27ae60">+2.79%</td>
        <td style="color:#27ae60;font-weight:bold">%83.3</td></tr>
      <tr><td style="padding:3px 8px">100%+</td><td>8</td><td>%75.0</td><td>+3.96%</td><td>%50.0</td></tr>
    </tbody>
  </table>

  <b>En Iyi Kombinasyonlar (Top 10, WR bazli)</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Kombinasyon</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;font-weight:bold">skor&ge;60 + pozitif_gap</td><td>10</td>
        <td style="color:#27ae60;font-weight:bold">%100</td><td style="color:#27ae60">+7.06%</td></tr>
      <tr style="background:rgba(39,174,96,0.06)"><td style="padding:3px 8px">kilitli + hacim_dusus</td><td>16</td>
        <td style="color:#27ae60;font-weight:bold">%90.0</td><td style="color:#27ae60">+7.31%</td></tr>
      <tr><td style="padding:3px 8px">yab_artis + dusuk_hacim</td><td>17</td>
        <td style="color:#27ae60">%88.9</td><td style="color:#27ae60">+5.99%</td></tr>
      <tr><td style="padding:3px 8px">hacim&lt;1x + yab_artis</td><td>14</td>
        <td style="color:#27ae60">%87.5</td><td style="color:#27ae60">+5.49%</td></tr>
      <tr><td style="padding:3px 8px">hacim&lt;1x + para_girisi</td><td>15</td>
        <td style="color:#27ae60">%85.7</td><td>+3.35%</td></tr>
      <tr><td style="padding:3px 8px">para_girisi + dusuk_hacim</td><td>17</td>
        <td style="color:#27ae60">%85.7</td><td>+3.35%</td></tr>
      <tr><td style="padding:3px 8px">hacim_dusus (tek basina)</td><td>19</td>
        <td style="color:#27ae60">%84.6</td><td style="color:#27ae60">+5.73%</td></tr>
      <tr><td style="padding:3px 8px">skor&ge;50 + hacim&lt;1x</td><td>18</td>
        <td style="color:#27ae60">%83.3</td><td>+3.75%</td></tr>
      <tr><td style="padding:3px 8px">seri&ge;2 + hacim&lt;1x</td><td>13</td>
        <td>%81.8</td><td style="color:#27ae60">+4.96%</td></tr>
      <tr><td style="padding:3px 8px">hacim&lt;1x + hacim_dusus</td><td>12</td>
        <td>%80.0</td><td style="color:#27ae60">+4.47%</td></tr>
    </tbody>
  </table>

  <b>Tavan Devam Orani</b><br>
  <span style="color:#8b949e">Tum tavanlar: %30.6 | AL (skor&ge;60): %37.5 | Skor 50+: %30.3</span>
  <br><br>
  <i style="color:#8b949e">Donem: 10 Sub — 3 Mar 2026 (16 tarama gunu, N=545)</i>
</details>

<details class="guide">
  <summary>2 Yillik Backtest Sonuclari (2024-2026)</summary>
  <br>
  <b>Screener Karsilastirmasi (499 gun)</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Screener</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">3G WR</th><th style="padding:4px 8px">5G WR</th><th style="padding:4px 8px">5G Ort</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:3px 8px;font-weight:bold">v3 AL/SAT Screener</td><td>71979</td>
        <td>%47.7</td><td>+0.13%</td><td>%47.9</td><td>%48.3</td><td>+0.48%</td></tr>
      <tr style="background:rgba(39,174,96,0.06)"><td style="padding:3px 8px;font-weight:bold">v4 Rejim Filtresi</td><td>12070</td>
        <td style="color:#27ae60">%49.8</td><td style="color:#27ae60">+0.37%</td>
        <td>%48.2</td><td>%48.6</td><td style="color:#27ae60">+0.74%</td></tr>
    </tbody>
  </table>

  <b>v3 AL/SAT — Sinyal Tipleri</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Tip</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">5G WR</th><th style="padding:4px 8px">5G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;color:#2ecc71;font-weight:bold">SQ</td><td>20</td>
        <td style="color:#27ae60;font-weight:bold">%65.0</td><td style="color:#27ae60">+2.41%</td>
        <td style="color:#27ae60">%60.0</td><td style="color:#27ae60">+6.02%</td></tr>
      <tr><td style="padding:3px 8px;color:#f1c40f">ZAYIF</td><td>10114</td>
        <td style="color:#27ae60">%51.1</td><td>+0.35%</td><td>%52.2</td><td>+1.03%</td></tr>
      <tr><td style="padding:3px 8px;color:#95a5a6">MR</td><td>624</td>
        <td>%51.3</td><td>+0.20%</td><td>%53.3</td><td>+0.95%</td></tr>
      <tr><td style="padding:3px 8px;color:#e67e22">CMB</td><td>5800</td>
        <td>%50.6</td><td>+0.36%</td><td>%50.7</td><td>+1.06%</td></tr>
      <tr><td style="padding:3px 8px;color:#27ae60">GUCLU</td><td>4408</td>
        <td>%47.1</td><td>-0.04%</td><td>%46.6</td><td>+0.07%</td></tr>
      <tr><td style="padding:3px 8px;color:#e74c3c">CMB+</td><td>1321</td>
        <td>%46.2</td><td>+0.03%</td><td>%46.1</td><td>+0.60%</td></tr>
      <tr><td style="padding:3px 8px;color:#9b59b6">DONUS</td><td>17886</td>
        <td>%45.6</td><td>+0.04%</td><td>%44.6</td><td>+0.06%</td></tr>
    </tbody>
  </table>

  <b>v4 Rejim — Sinyal Tipleri</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Tip</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">5G WR</th><th style="padding:4px 8px">5G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;color:#f1c40f;font-weight:bold">ZAYIF</td><td>3967</td>
        <td style="color:#27ae60;font-weight:bold">%53.5</td><td style="color:#27ae60">+0.61%</td>
        <td style="color:#27ae60">%53.0</td><td style="color:#27ae60">+1.30%</td></tr>
      <tr><td style="padding:3px 8px;color:#2ecc71">SQ</td><td>97</td>
        <td>%52.6</td><td>+0.80%</td><td>%49.0</td><td>+1.74%</td></tr>
      <tr><td style="padding:3px 8px;color:#27ae60">GUCLU</td><td>2971</td>
        <td>%49.4</td><td>+0.27%</td><td>%48.9</td><td>+0.58%</td></tr>
      <tr><td style="padding:3px 8px;color:#9b59b6">DONUS</td><td>4859</td>
        <td>%47.0</td><td>+0.25%</td><td>%44.8</td><td>+0.36%</td></tr>
    </tbody>
  </table>

  <b>OE (Overextended) Etkisi</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">OE</th><th style="padding:4px 8px" colspan="2">v3 AL/SAT</th>
      <th style="padding:4px 8px" colspan="2">v4 Rejim</th>
    </tr>
    <tr style="border-bottom:1px solid #21262d;text-align:left">
      <th style="padding:4px 8px"></th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">1G Ort</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:3px 8px">OE &le; 2</td>
        <td>%46.4</td><td>+0.01%</td><td>%47.1</td><td>+0.12%</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td style="padding:3px 8px;font-weight:bold">OE &ge; 3</td>
        <td style="color:#27ae60;font-weight:bold">%50.8</td><td style="color:#27ae60">+0.38%</td>
        <td style="color:#27ae60;font-weight:bold">%51.5</td><td style="color:#27ae60">+0.52%</td></tr>
      <tr style="background:rgba(39,174,96,0.06)"><td style="padding:3px 8px">OE &ge; 4</td>
        <td style="color:#27ae60">%50.7</td><td style="color:#27ae60">+0.41%</td>
        <td style="color:#27ae60">%51.9</td><td style="color:#27ae60">+0.59%</td></tr>
    </tbody>
  </table>

  <b>En Iyi Kombinasyonlar (v3)</b>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Kombinasyon</th><th style="padding:4px 8px">N</th>
      <th style="padding:4px 8px">1G WR</th><th style="padding:4px 8px">3G WR</th><th style="padding:4px 8px">3G Ort</th>
    </tr></thead>
    <tbody>
      <tr style="background:rgba(39,174,96,0.1)"><td style="padding:3px 8px;font-weight:bold">OE=0 + FULL_TREND + GUCLU/CMB</td><td>32</td>
        <td style="color:#27ae60;font-weight:bold">%71.9</td><td style="color:#27ae60;font-weight:bold">%68.8</td>
        <td style="color:#27ae60">+4.30%</td></tr>
      <tr><td style="padding:3px 8px">Q&ge;80 + OE=3 + ZAYIF</td><td>199</td>
        <td style="color:#27ae60">%58.8</td><td>%55.3</td><td style="color:#27ae60">+2.10%</td></tr>
      <tr><td style="padding:3px 8px">OE=0 + FULL_TREND + ZAYIF</td><td>63</td>
        <td>%58.7</td><td style="color:#27ae60">%61.9</td><td>+0.78%</td></tr>
      <tr><td style="padding:3px 8px">OE&ge;3 + FULL_TREND + DONUS</td><td>58</td>
        <td>%58.6</td><td>%55.2</td><td>+0.95%</td></tr>
      <tr><td style="padding:3px 8px">Q&ge;80 + OE&ge;3 + ZAYIF</td><td>678</td>
        <td>%57.8</td><td>%52.3</td><td>+0.75%</td></tr>
    </tbody>
  </table>

  <br>
  <i style="color:#8b949e">Backtest: 4 Mar 2024 — 2 Mar 2026 (499 gun, 539 hisse)</i>
</details>

<div class="section section-devam">
  <h2>Tavan Serisi — Yarin Devam Eder Mi? ({len(tavan_devam)})</h2>
  <table id="table-devam">
    <thead><tr>
      <th>#</th><th>Hisse</th><th>Skor</th><th>Karar</th><th>Seri</th>
      <th>Fiyat</th><th>Hacim</th><th>EMA21</th><th>RSI</th><th>RS</th><th>Yabanci</th><th>Nedenler</th>
    </tr></thead>
    <tbody>{rows_devam}</tbody>
  </table>
</div>

<div class="section section-kand">
  <h2>Tavan Kandidatlari — Yarin Tavan Olabilir (Top {min(30, len(tavan_kandidat))})</h2>
  <table id="table-kand">
    <thead><tr>
      <th>#</th><th>Hisse</th><th>Skor</th>
      <th>Fiyat</th><th>Getiri</th><th>Hacim</th><th>EMA21</th><th>RSI</th><th>RS</th><th>Yabanci</th><th>Nedenler</th>
    </tr></thead>
    <tbody>{rows_kandidat}</tbody>
  </table>
</div>

<script>
function applyFilter() {{
  const q = document.getElementById('fTicker').value.toUpperCase();
  document.querySelectorAll('tbody tr').forEach(tr => {{
    const ticker = tr.dataset.ticker || '';
    tr.style.display = (!q || ticker.includes(q)) ? '' : 'none';
  }});
}}

// Tablo siralama
document.querySelectorAll('th').forEach((th, colIdx) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.sort !== 'asc';
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {{
      const aVal = a.children[colIdx]?.textContent.trim() || '';
      const bVal = b.children[colIdx]?.textContent.trim() || '';
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
      return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>

</body>
</html>'''
    return html


def push_tavan_html_to_github(html_content, date_str):
    """HTML raporu GitHub Pages repo'suna push et (tavan/ klasörüne)."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")

    if not token or not repo:
        print("⚠️ GH_TOKEN veya GH_PAGES_REPO tanımlı değil, HTML lokal kaydedildi.")
        return None

    api_url = f"https://api.github.com/repos/{repo}/contents/tavan.html"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    content_b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')

    # Mevcut dosyanın SHA'sını al (güncelleme için gerekli)
    sha = None
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass

    payload = {
        "message": f"tavan tarama — {date_str}",
        "content": content_b64,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            owner, name = repo.split("/")
            page_url = f"https://{owner}.github.io/{name}/tavan.html"
            print(f"✅ Tavan HTML rapor yayınlandı: {page_url}")
            return page_url
        else:
            print(f"⚠️ GitHub push hata: {resp.status_code} — {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"⚠️ GitHub push hata: {e}")
        return None


def _log_sinyaller(config, tavan_devam, tavan_kandidat):
    """Sinyalleri CSV'ye loglar."""
    dosya = config.get('sinyal_log_dosya', 'sinyal_log.csv')
    header = not os.path.exists(dosya)
    tarih = datetime.now().strftime('%Y-%m-%d')
    
    with open(dosya, 'a', encoding='utf-8') as f:
        if header:
            f.write("tarih,tip,ticker,skor,streak,fiyat,vol_ratio,yab_oran\n")
        
        for s in tavan_devam:
            yab = s.get('yab_oran', '')
            f.write(f"{tarih},DEVAM,{s['ticker']},{s['score']},{s['streak']},"
                    f"{s['close']:.2f},{s['vol_ratio']:.1f},{yab}\n")
        
        for s in tavan_kandidat[:20]:
            yab = s.get('yab_oran', '')
            f.write(f"{tarih},KANDIDAT,{s['ticker']},{s['score']},,"
                    f"{s['close']:.2f},{s['vol_ratio']:.1f},{yab}\n")


def daemon_mod(config: dict):
    """
    Arka planda sürekli çalışır, her gün belirlenen saatte tarama yapar.
    """
    try:
        import schedule
    except ImportError:
        print("❌ 'schedule' paketi gerekli: pip install schedule")
        return
    
    scan_saati = config.get('scan_saati', '18:35')
    
    print(f"🤖 Daemon modu başlatıldı")
    print(f"⏰ Günlük tarama saati: {scan_saati}")
    print(f"   Durdurmak için Ctrl+C")
    print()
    
    def scheduled_scan():
        today = datetime.now()
        # Hafta sonu kontrolü
        if today.weekday() >= 5:  # Cumartesi/Pazar
            if not config.get('pazar_aktif', False):
                print(f"📅 Hafta sonu — tarama atlandı ({today.strftime('%A')})")
                return
        
        try:
            gunluk_tarama(config)
        except Exception as e:
            print(f"❌ Tarama hatası: {e}")
            # Hata bildirimi
            bot = TelegramBot(config)
            if bot.aktif:
                bot.mesaj_gonder(f"❌ Tarama hatası: {str(e)[:200]}")
    
    schedule.every().day.at(scan_saati).do(scheduled_scan)
    
    # Ayrıca her gün 09:45'te portföy kontrolü (piyasa açılışından sonra)
    def sabah_kontrol():
        today = datetime.now()
        if today.weekday() >= 5:
            return
        
        print(f"\n⏰ Sabah portföy kontrolü — {today.strftime('%H:%M')}")
        risk = RiskManager(config)
        print(risk.portfoy_ozet())
    
    schedule.every().day.at("09:45").do(sabah_kontrol)
    
    while True:
        schedule.run_pending()
        time.sleep(30)


# ============================================================
# 4) SETUP (İlk Kurulum)
# ============================================================

def setup(config: dict):
    """İlk kurulum — Telegram test ve yapılandırma."""
    
    print("=" * 70)
    print("⚙️  BIST TAVAN TARAYICI — KURULUM")
    print("=" * 70)
    
    # Config durumu
    print(f"\n📁 Config dosyası: {CONFIG_FILE}")
    print(f"💰 Portföy: {config['portfoy_toplam']:,.0f} TL")
    print(f"📊 AL eşiği: Skor ≥ {config['devam_al_esik']}")
    print(f"🛑 Stop-loss: {config['stop_loss_pct']}%")
    print(f"⏰ Tarama saati: {config['scan_saati']}")
    
    # Telegram
    print(f"\n{'─'*40}")
    print("📱 TELEGRAM KURULUMU")
    print(f"{'─'*40}")
    
    if not config['telegram_token']:
        print("""
Telegram bot kurmak için:
  1) Telegram'da @BotFather'ı bul
  2) /newbot komutunu gönder
  3) Bot adı ve kullanıcı adı belirle
  4) Verilen token'ı kopyala
  5) {config_file} dosyasındaki "telegram_token" alanına yapıştır
  6) Bot'una bir mesaj gönder (herhangi bir şey yaz)
  7) python bist_auto_trader.py --setup komutunu tekrar çalıştır
""".format(config_file=CONFIG_FILE))
    else:
        print(f"  Token: {config['telegram_token'][:10]}...")
        
        bot = TelegramBot(config)
        
        if not config['telegram_chat_id']:
            print("\n  Chat ID bulunamadı. Bot'a mesaj gönderdin mi?")
            chat_id = bot.chat_id_bul()
            if chat_id:
                config['telegram_chat_id'] = chat_id
                save_config(config)
                print(f"  ✅ Chat ID kaydedildi: {chat_id}")
                
                # Test mesajı gönder
                bot = TelegramBot(config)  # Yeniden oluştur
                if bot.test():
                    print("  ✅ Test mesajı gönderildi!")
                else:
                    print("  ❌ Test mesajı gönderilemedi")
        else:
            print(f"  Chat ID: {config['telegram_chat_id']}")
            if bot.test():
                print("  ✅ Telegram bağlantısı aktif!")
            else:
                print("  ❌ Telegram bağlantısı başarısız")
    
    # macOS LaunchAgent
    print(f"\n{'─'*40}")
    print("🖥  OTOMATİK ÇALIŞTIRMA (macOS)")
    print(f"{'─'*40}")
    
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    working_dir = os.path.dirname(script_path)
    
    plist_name = "com.bist.tavanscanner"
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
    
    # Saat ve dakika parse
    saat, dakika = config['scan_saati'].split(':')
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{int(saat)}</integer>
        <key>Minute</key>
        <integer>{int(dakika)}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{working_dir}/scanner_output.log</string>
    <key>StandardErrorPath</key>
    <string>{working_dir}/scanner_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{os.path.dirname(python_path)}</string>
    </dict>
</dict>
</plist>"""
    
    print(f"""
  Otomatik günlük çalıştırma için (macOS LaunchAgent):
  
  1) Aşağıdaki komutu çalıştır:
  
     cat > {plist_path} << 'EOF'
{plist_content}
EOF

  2) Aktifleştir:
     launchctl load {plist_path}

  3) Durdurmak için:
     launchctl unload {plist_path}

  4) Kontrol:
     launchctl list | grep bist
""")
    
    print(f"{'─'*40}")
    print("✅ Kurulum bilgileri tamamlandı!")
    print(f"{'─'*40}")
    print(f"""
  Sonraki adımlar:
  1) {CONFIG_FILE} dosyasını düzenle (Telegram token, portföy büyüklüğü)
  2) python bist_auto_trader.py --setup    (Telegram test)
  3) python bist_auto_trader.py            (manuel tarama)
  4) python bist_auto_trader.py --daemon   (sürekli çalıştır)
""")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    config = load_config()
    
    if "--setup" in sys.argv:
        setup(config)
    
    elif "--daemon" in sys.argv:
        daemon_mod(config)
    
    elif "--portfoy" in sys.argv:
        risk = RiskManager(config)
        print(risk.portfoy_ozet())
    
    elif "--test-telegram" in sys.argv:
        bot = TelegramBot(config)
        if bot.test():
            print("✅ Telegram çalışıyor!")
        else:
            print("❌ Telegram bağlantısı başarısız")
    
    else:
        # Normal çalıştırma — tarama + bildirim
        gunluk_tarama(config)
