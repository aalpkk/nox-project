# NOX Portfolio Advisor — Runbook

AI portföy danışmanı: günlük tarama verisini okur, **gerçek portföyü** değerlendirir,
TUT/SAT/AZALT + AL önerisi üretir. Mimari: deterministik context pack → tek
yapılandırılmış Claude çağrısı (`emit_advisory`, tool_choice zorlamalı) →
deterministik post-validasyon. **LLM sayı üretemez** — boyut/risk/stop hepsi
`guardrails.py`'da hesaplanır, LLM çıktısı pack'e geri damgalanır.

## Veri yerleşimi (gizlilik)

| Veri | Yer | Neden |
|---|---|---|
| Portföy (adet/maliyet/nakit) | `aalpkk/nox-private` → `portfolio/portfolio.json` | nox-project PUBLIC — gerçek para verisi giremez |
| `advisory_latest.json` + `scorecard.json` | `aalpkk/nox-private` → `advisor/` | bot `/danis` buradan okur |
| Günlük advisory/pack/HTML | Actions artifact (30g) | public repoya commit YOK |
| Cluster-3 ledger | public repo `output/…parquet` | kişisel veri yok; lokalde `fill` çalıştırıp commit'le (advisor >7g'de STALE etiketler) |

Erişim: mevcut `GH_TOKEN` PAT (classic, `repo` scope — doğrulandı). Yazma çakışması:
Contents API PUT-with-sha optimistic lock, `Portfolio.mutate()` 3 retry.

## Çalıştırma

```bash
# lokal
python -m agent.advisor --asof 2026-06-13 --dry-run            # LLM+Telegram yok
python -m agent.advisor --notify                               # tam akış
python -m agent.advisor --no-llm --notify                      # deterministik fallback

# env
NOX_PORTFOLIO_REPO=aalpkk/nox-private    # yoksa lokal agent/portfolio.json
NOX_PORTFOLIO_PATH=/path/pf.json         # açık lokal override
NOX_OUTPUT_DIR=output                    # sinyal dosyaları kökü
NOX_MODEL_ADVISOR=claude-sonnet-4-5-...  # model override (vars: MODEL_ANALYSIS)
```

CI: `.github/workflows/portfolio-advisor.yml` — "decision engine v1" `workflow_run`
zinciri (NO_DE_DAY dahil), `de-v1-watchlist-{asof}` artifact'ından asof türetir,
tavan lock JSON'ını stage'ler. LLM hatası → fallback raporu yine Telegram'a gider;
job fail = gerçek bug. Dispatch ile manuel: asof boş bırakılırsa en son DE run'ı.

## Sentez yolları (HİBRİT tasarım)

`--llm-mode {auto,api,cli,none}`:
- **CI = `none`**: her akşam deterministik garanti raporu, $0 (korkuluk flag'leri +
  boyutlandırılmış adaylar). API'ye dönmek istersen workflow'a ANTHROPIC_API_KEY
  ekleyip `--llm-mode auto` yap (Sonnet ~$2/ay).
- **Lokal = `cli`**: headless `claude -p --model opus` — **API key'siz, abonelikten,
  marjinal maliyet $0**. `synthesize_via_cli()`: pack stdin'den girer, salt-JSON
  çıkar (`_extract_json` kod-bloğu/prose toleranslı); post_validate korkulukları
  AYNEN geçerli (sayılar yine pack'ten damgalanır). Model: `NOX_ADVISOR_CLI_MODEL`
  (vars: opus).
- `auto`: ANTHROPIC_API_KEY varsa api, yoksa claude CLI varsa cli, o da yoksa none.

Lokal akşam cron'u (launchd):
```bash
cp scripts/com.nox.advisor.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nox.advisor.plist
# kaldırmak: launchctl unload ~/Library/LaunchAgents/com.nox.advisor.plist
```
`scripts/advisor_local.sh`: en güncel DE artifact'ını indirir (gh token ile) →
`run_advisor(llm_mode="cli", notify=True)` → log `logs/advisor_local.log`.
Mac uykudaysa koşmaz — o akşam CI'nın deterministik raporu yine gelmiştir; iki
rapor da geldiğinde zengin olan (lokal) `advisory_latest`'i ezer, `/danis` onu görür.

## Telegram bot komutları

- `/portfoy` — pozisyonlar + canlı PnL + risk
- `/poz_al THYAO 100 45.50 [tarih]` · `/poz_sat THYAO [adet] [fiyat]` · `/nakit 150000`
  (yazma SADECE komutla; Claude tool'ları salt-okunur)
- `/danis` — son advisory (private repodan); portföy rev eskiyse uyarır
- `/danis tam` — DE artifact'ını indirip tam yeniden-sentez (~1-2 dk)
- Serbest metin: `get_portfolio` + `get_validated_signals` tool'ları Claude'a açık

Bot host (Render) env ekleri: `GH_TOKEN`, `NOX_PORTFOLIO_REPO=aalpkk/nox-private`.

## Korkuluklar (LLM'den bağımsız)

stop ihlali → SELL'e zorlanır · overweight → BUY blok · boyut = equity·%1risk/(entry−stop),
nakit-tamponu+%15-pozisyon+lot clamp · toplam yeni+mevcut risk ≤ %6 (EXECUTABLE→
SIZE_REDUCED-yarım sırayla) · eldekine yeniden-AL yok (underweight+EXECUTABLE=ADD) ·
max 12 pozisyon · şema dışı ticker drop · eksik pozisyona auto-HOLD · disclaimer
şema alanı (prompt değil).

Ayarlar `portfolio.json → settings`'ten değiştirilebilir.

## Skorkart

Her koşu, yayından önce `advisory_latest`'i (önceki rapor) bugünkü fiyatlarla skorlar:
AL adayları entry_ref→şimdi getirisi, SAT/AZALT sonrası hareket. `nox-private/advisor/
scorecard.json`'a append (aynı gün re-run'da atlanır), rapora "Önceki rapor" satırı düşer.
Naif d+1 yaklaşımı — fillability/slippage yok, kaba gösterge.

## Sinyal doğrulama etiketleri (yapısal dürüstlük)

DE v1 `paper-track (tek-rejim)` · tavan lock `paper-track (tek-rejim 2025-26 boğa)`
(sabah 11:00 pick'i = pozisyon bağlamı, akşam AL adayı DEĞİL; V1 exit: SL−10/TP1+4/
trail2/H25) · cluster-3 `paper-track (forward ledger)` · diğer tüm taramalar
`context-only/keşifsel` — prompt + post-check context-only sinyali birincil gerekçe
yapamaz. Hiçbir hat canlı-doğrulanmış değildir; rapor yatırım tavsiyesi değildir.
