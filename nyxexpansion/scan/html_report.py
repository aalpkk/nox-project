"""HTML report for nyxexpansion daily scan + Markowitz 4-stock portfolio.

Visual aesthetic borrowed from the briefing report (NOX dark theme, aurora bg,
glassmorphism cards). Tickers link to TradingView (BIST: prefix).
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from core.reports import _NOX_CSS

_TV_BASE = "https://www.tradingview.com/chart/?symbol=BIST:"


def _fnum(v, fmt: str = ".2f", default: str = "—") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return default


def _ticker_link(ticker: str) -> str:
    t = html.escape(str(ticker))
    return (f'<a class="tv-link" href="{_TV_BASE}{t}" target="_blank" '
            f'rel="noopener" title="{t} — TradingView">{t}</a>')


def _bucket_chip(bucket: str) -> str:
    b = str(bucket)
    color_map = {
        "clean":    ("rgba(122,158,122,0.18)", "var(--nox-green)"),
        "mild":     ("rgba(122,143,165,0.18)", "var(--nox-blue)"),
        "elevated": ("rgba(184,149,110,0.18)", "var(--nox-copper)"),
        "severe":   ("rgba(158,90,90,0.20)",   "var(--nox-red)"),
    }
    bg, fg = color_map.get(b, ("rgba(85,82,80,0.18)", "var(--text-muted)"))
    return (f'<span class="reg-badge" style="background:{bg};color:{fg}">'
            f'{html.escape(b)}</span>')


def _tag_chip(tag: str) -> str:
    t = str(tag)
    color_map = {
        "clean_watch":      ("rgba(122,158,122,0.14)", "var(--nox-green)"),
        "extended_watch":   ("rgba(184,149,110,0.16)", "var(--nox-copper)"),
        "special_handling": ("rgba(138,122,158,0.16)", "var(--nox-purple)"),
    }
    bg, fg = color_map.get(t, ("rgba(85,82,80,0.14)", "var(--text-muted)"))
    return (f'<span class="kc-badge" style="background:{bg};color:{fg}">'
            f'{html.escape(t)}</span>')


def _retention_chip(pass_flag: bool, note: str) -> str:
    cls = "kc-hi" if pass_flag else "kc-lo"
    label = "PASS" if pass_flag else "DROP"
    return (f'<span class="kc-badge {cls}" title="{html.escape(note)}">'
            f'{label}</span>')


def _note_for_row(row) -> str:
    parts = []
    stretch = str(row.get("stretch_rating", ""))
    ext = str(row.get("extension_rating", ""))
    mom = str(row.get("momentum_intensity", ""))
    room = str(row.get("upside_room", ""))
    if ext == "very_extended":
        parts.append("very_extended")
    if stretch in ("high", "very_high"):
        parts.append(f"stretch={stretch}")
    if room == "tight":
        parts.append("tight_room")
    elif room == "above_52w":
        parts.append("above_52w")
    if mom == "strong":
        parts.append("strong_momo")
    if row.get("risk_bucket") == "severe":
        parts.append("SEVERE→exclude")
    return ", ".join(parts) if parts else "—"


def _retention_cell(d: dict) -> tuple[str, str, str]:
    rank = d.get("rank_1700_surrogate")
    score = d.get("score_1700_surrogate")
    note = d.get("timing_clean_note") or "—"
    rank_str = "—" if rank is None or pd.isna(rank) else str(int(rank))
    score_str = "—" if score is None or pd.isna(score) else f"{float(score):.2f}"
    return rank_str, score_str, str(note)


def _trade_plan_cell(d: dict) -> str:
    entry = d.get("close_0")
    atr = d.get("atr_14")
    if (entry is None or pd.isna(entry) or
            atr is None or pd.isna(atr) or float(atr) <= 0):
        return "<td class='trade-plan'>—</td>"
    e = float(entry); a = float(atr)
    stop = e - 1.5 * a
    sl_pct = (stop / e - 1.0) * 100
    bucket = str(d.get("risk_bucket", "")).lower()

    if bucket == "clean":
        partials = [
            ("TP1-25", e + 0.7 * a),
            ("TP2-15", e + 1.5 * a),
            ("TP3-15", e + 3.0 * a),
        ]
        trail_pct = 45
    else:
        partials = [
            ("TP1-15", e + 1.5 * a),
            ("TP2-15", e + 3.0 * a),
        ]
        trail_pct = 70

    rows = [
        f"<div class='tp-row tp-head'>E {e:.2f} · ATR {a:.2f}</div>",
        f"<div class='tp-row tp-sl'>SL {stop:.2f} <span>({sl_pct:+.1f}%)</span></div>",
    ]
    for label, price in partials:
        pct = (price / e - 1.0) * 100
        rows.append(
            f"<div class='tp-row'>{label} → {price:.2f} "
            f"<span>({pct:+.1f}%)</span></div>"
        )
    rows.append(
        f"<div class='tp-row tp-trail'>TR-{trail_pct} <span>trail</span></div>"
    )
    return f"<td class='trade-plan'>{''.join(rows)}</td>"


def _row_html(i: int, d: dict) -> str:
    note = _note_for_row(d)
    bucket = d.get("risk_bucket", "—")
    tag = d.get("exec_tag", "—")
    rank_str, score_str, ret_note = _retention_cell(d)
    ret_pass = bool(d.get("retention_pass", False))
    return (
        f"<tr>"
        f"<td class='num'>{i}</td>"
        f"<td>{_ticker_link(d.get('ticker', ''))}</td>"
        f"<td class='num'>{_fnum(d.get('winner_R_pred'))}</td>"
        f"<td class='num'>{_fnum(d.get('score_pct'))}</td>"
        f"<td class='num'>{score_str}</td>"
        f"<td class='num'>{rank_str}</td>"
        f"<td>{_tag_chip(tag)}</td>"
        f"<td>{_bucket_chip(bucket)}</td>"
        f"<td class='num'>{_fnum(d.get('execution_risk_score'), '.1f')}</td>"
        f"<td>{_retention_chip(ret_pass, ret_note)}</td>"
        f"{_trade_plan_cell(d)}"
        f"<td class='detail-cell'>{html.escape(note)}</td>"
        f"</tr>"
    )


def _candidates_table(rows_html: list[str], table_id: str) -> str:
    body = "\n".join(rows_html) if rows_html else (
        "<tr><td colspan='12' style='text-align:center;color:var(--text-muted);"
        "padding:24px'>—</td></tr>"
    )
    return f"""
  <div class="nox-table-wrap">
    <table id="{table_id}">
      <thead><tr>
        <th>#</th>
        <th>Ticker</th>
        <th class="th-tip" title="Model'in tahmini kazanan büyüklüğü (regresyon R skoru). Yüksek = ranker daha güçlü kazanan adayı görüyor.">winR</th>
        <th class="th-tip" title="v4C skorunun günün tarama evrenindeki yüzdelik dilimi (0–1). 1.0 = günün en iyisi.">pct</th>
        <th class="th-tip" title="Surrogate model'in 17:00 truncated panelde verdiği skor (timing-clean view, look-ahead'sız).">winR_1700</th>
        <th class="th-tip" title="17:00 panelinde sıralama. Live retention filtresi rank ≤ 10 olanı geçirir.">rank_1700</th>
        <th class="th-tip" title="Bileşik execution etiketi (clean / extended_watch / special_handling / ample_only). Girişin niteliksel özeti.">exec_tag</th>
        <th class="th-tip" title="Risk skoru 4 kova: clean / mild / elevated / severe. severe = pratik olarak işleme alınmaz.">risk_bucket</th>
        <th class="th-tip" title="execution_risk_score — sayısal risk skoru (0–7+). Yüksek = giriş kötü (gap, parabolic uzanma vs.).">rscr</th>
        <th class="th-tip" title="Timing-clean retention filtresi sonucu (PASS / SKIP / DROP) — 17:00 surrogate rank ≤ 10 gate'inden geçti mi?">retention</th>
        <th class="th-tip" title="P8 omurga işlem planı: SL = entry − 1.5 ATR; TP'ler ATR çarpanlarında, etiketin yanında satılan % yüzdesi; TR = trail kalan miktar.">Trade Plan</th>
        <th>Not</th>
      </tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
  </div>
"""


def _portfolio_section(portfolio: dict, scan_df: pd.DataFrame) -> str:
    if not portfolio or not portfolio.get("weights") or portfolio.get("error"):
        err = portfolio.get("error") if portfolio else "Markowitz çalıştırılamadı"
        return f"""
        <section class="nox-card warn-card">
          <h2>💼 Markowitz 4'lü Portföy — ÜRETİLMEDİ</h2>
          <div class="card-body">Sebep: {html.escape(str(err))}</div>
        </section>
        """

    per = portfolio.get("per_stock_stats", {})
    sel = sorted(portfolio["weights"].items(), key=lambda x: -x[1])
    pf_rows = []
    for t, w in sel:
        stats = per.get(t, {})
        match = scan_df[scan_df["ticker"] == t]
        winR = match["winner_R_pred"].iloc[0] if not match.empty else None
        bucket = match["risk_bucket"].iloc[0] if not match.empty else "—"
        last_pct = stats.get("last_return_pct", 0.0)
        last_cls = "rs-pos" if last_pct >= 0 else "rs-neg"
        pf_rows.append(
            f"<tr>"
            f"<td>{_ticker_link(t)}</td>"
            f"<td class='num'><b>{w*100:.1f}%</b></td>"
            f"<td class='num'>{_fnum(winR)}</td>"
            f"<td>{_bucket_chip(bucket)}</td>"
            f"<td class='num'>{_fnum(stats.get('mean_ann_pct'), '+.1f')}%</td>"
            f"<td class='num'>{_fnum(stats.get('vol_ann_pct'), '.1f')}%</td>"
            f"<td class='num {last_cls}'>{_fnum(last_pct, '+.2f')}%</td>"
            f"</tr>"
        )

    wmin, wmax = portfolio.get("weight_bounds", [0.10, 0.50])
    kpis = (
        ("Sharpe",     f"{portfolio['sharpe']:.3f}"),
        ("μ (ann.)",   f"{portfolio['expected_return']:+.2f}%"),
        ("σ (ann.)",   f"{portfolio['expected_risk']:.2f}%"),
        ("Hisse",      f"{len(portfolio['weights'])}"),
        ("Lookback",   f"{portfolio.get('lookback_days', 60)}g"),
        ("w-aralığı",  f"[{wmin:.2f}, {wmax:.2f}]"),
        ("Combos",     f"{portfolio.get('combos_evaluated', 0):,}"),
    )
    kpi_html = "".join(
        f'<div class="kpi"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(v)}</span></div>'
        for k, v in kpis
    )

    n_universe = len(portfolio.get('universe_used', []))
    return f"""
    <section class="nox-card pf-card">
      <h2>💼 Markowitz 4'lü Portföy <span class="sub">— Max Sharpe (combinatorial)</span></h2>
      <div class="kpi-strip">{kpi_html}</div>
      <div class="nox-table-wrap">
        <table>
          <thead><tr>
            <th>Hisse</th>
            <th class="th-tip" title="Markowitz optimizasyonundan çıkan portföy ağırlığı (long-only, sum=1, w∈[0.10, 0.50]).">Ağırlık</th>
            <th class="th-tip" title="Model'in tahmini kazanan büyüklüğü (regresyon R skoru).">winR</th>
            <th class="th-tip" title="Risk skoru 4 kova: clean / mild / elevated / severe. severe = portföye alınmaz.">Bucket</th>
            <th class="th-tip" title="Beklenen yıllık getiri (son 60 günlük log-return ortalamasının yıllıklandırılmış hâli, %).">μ (60g ann.)</th>
            <th class="th-tip" title="Yıllık volatilite (60 günlük penceredeki standart sapmanın yıllıklandırılmış hâli, %).">σ (60g ann.)</th>
            <th class="th-tip" title="Hissenin en son günkü realize getirisi (%).">Son Gün</th>
          </tr></thead>
          <tbody>{''.join(pf_rows)}</tbody>
        </table>
      </div>
      <div class="card-footer">
        <b>Aday havuzu:</b> winR sıralı top-{n_universe} (risk_bucket ≠ severe) ·
        60g daily log-return, yıllıklandırılmış μ/Σ + Ledoit-Wolf shrinkage ·
        SLSQP, long-only, sum(w)=1.
      </div>
    </section>
    """


def _stat_chip(label: str, value, color_var: str = "var(--text-secondary)") -> str:
    return (
        f'<div class="nox-stat">'
        f'<span class="dot" style="background:{color_var}"></span>'
        f'<span>{html.escape(label)}</span>'
        f'<span class="cnt">{html.escape(str(value))}</span>'
        f'</div>'
    )


def render_html(
    scan_df: pd.DataFrame,
    portfolio: dict,
    target_date: pd.Timestamp,
    meta: dict,
) -> str:
    n_total = len(scan_df)
    n_severe = int((scan_df.get("risk_bucket") == "severe").sum())

    retention = meta.get("retention") or {}
    retention_enabled = bool(retention.get("enabled", False))

    if retention_enabled:
        tradeable_df = scan_df[scan_df.get("retention_pass") == True].copy()
        watchlist_df = scan_df[scan_df.get("retention_pass") != True].copy()
    else:
        tradeable_df = scan_df.iloc[0:0].copy()
        watchlist_df = scan_df.copy()

    tradeable_rows = [
        _row_html(i, dict(zip(scan_df.columns, r)))
        for i, r in enumerate(tradeable_df.itertuples(index=False, name=None), 1)
    ]
    watchlist_rows = [
        _row_html(i, dict(zip(scan_df.columns, r)))
        for i, r in enumerate(watchlist_df.itertuples(index=False, name=None), 1)
    ]

    universe_size = meta.get("universe_size", "?")
    dataset_path = meta.get("dataset_path", "?")
    regime_dist = meta.get("regime_dist", {}) or {}
    regime_str = ", ".join(f"{k}={v}" for k, v in regime_dist.items()) if regime_dist else "—"
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Stats strip ──
    stats_html = (
        _stat_chip("Evren", universe_size, "var(--nox-blue)")
        + _stat_chip("Trigger", n_total, "var(--nox-cyan)")
        + _stat_chip("Severe", n_severe, "var(--nox-red)")
        + _stat_chip("Rejim", regime_str, "var(--nox-copper)")
    )
    if retention_enabled:
        stats_html += (
            _stat_chip("PASS", retention.get("n_pass", 0), "var(--nox-green)")
            + _stat_chip("DROP", retention.get("n_drop", 0), "var(--nox-orange)")
        )

    # ── Warning banner (always-on) ──
    warn_banner = """
    <section class="nox-card banner warn-banner">
      <span class="banner-icon">⚠️</span>
      <div>
        <b>Daily candidate ranker — auto-entry DEĞİL.</b>
        Bu liste trigger'dan geçen adayları winR sırasıyla gösterir; pozisyon kararı için
        execution risk (bucket / very_extended / tight_room) ve canlı 17:30 dip/volume
        gözlemi ayrıca değerlendirilmelidir. Severe bucket hard-exclude edilmiştir.
      </div>
    </section>
    """

    # ── Retention banner ──
    if retention_enabled:
        rank_t = retention.get("rank_threshold", 10)
        n_pass = retention.get("n_pass", 0)
        n_drop = retention.get("n_drop", 0)
        n_unscored = retention.get("n_unscored", 0)
        notes = retention.get("notes", {}) or {}
        notes_str = ", ".join(f"{k}={v}" for k, v in notes.items()) or "—"
        sources = retention.get("source_breakdown", {}) or {}
        if sources:
            src_str = " · ".join(f"{k}={v}" for k, v in sorted(sources.items()))
            src_line = (
                f'<div class="banner-sub">Bugün veri kaynakları: '
                f'<code>{html.escape(src_str)}</code></div>'
            )
        else:
            src_line = ""
        retention_banner = f"""
        <section class="nox-card banner ret-banner">
          <span class="banner-icon">🛡</span>
          <div>
            <b>Timing-clean retention filter (17:00 TR top-{rank_t})</b> —
            PASS=<b>{n_pass}</b> · DROP=<b>{n_drop}</b> · unscored=<b>{n_unscored}</b>.
            Tradeable list = retention_pass=True; watchlist'te kalanlar gözlem amaçlı,
            17:30 proxy'de tradeable kabul edilmez.
            <div class="banner-sub">Notes: <code>{html.escape(notes_str)}</code></div>
            {src_line}
          </div>
        </section>
        """
    else:
        retention_banner = """
        <section class="nox-card banner ret-banner-off">
          <span class="banner-icon">⚠️</span>
          <div>
            <b>Timing-clean retention stage SKIPPED.</b> Hiçbir aday için 17:00
            truncated re-rank yapılmadı; aşağıdaki tüm satırlar Watchlist olarak
            işaretlendi. Tradeable list bu raporda BOŞ.
          </div>
        </section>
        """

    pf_section = _portfolio_section(portfolio, scan_df)

    howto_dropdown = """
    <details class="howto-card">
      <summary>Bu tarama nasıl kullanılır?</summary>
      <div class="howto-body">

        <h3>Önce neye bakacağım?</h3>
        <p>Bu raporda hisseler iki gruba ayrılır:</p>
        <ul>
          <li><b>Tradeable Candidates</b>: işlem için daha ciddi bakılacak grup</li>
          <li><b>Watchlist Only</b>: izlemeye değer ama işlem için daha temkinli yaklaşılacak grup</li>
        </ul>
        <p>İşlem düşünüyorsan önce <b>Tradeable Candidates</b> listesine bakılır.</p>

        <h3>İşleme giriş mantığı nedir?</h3>
        <p>Basit kullanım:</p>
        <ol>
          <li>Tradeable Candidates listesini aç</li>
          <li>En yüksek skorlu hisselere bak</li>
          <li>Aynı hissede zaten pozisyonun var mı kontrol et</li>
          <li>Sektörde aşırı yığılma var mı bak</li>
          <li>İşlem yapacaksan 17:30 giriş mantığına göre hareket et</li>
        </ol>
        <p>Bu rapor doğrudan "hemen al" butonu değildir.<br>
        Önce adayları sıralar, sonra işlem için daha uygun olanları ayırır.</p>

        <h3>Girişten sonra pozisyon nasıl yönetilir?</h3>
        <p>Bu sistemde asıl önemli kısım girişten sonrası, yani çıkış planıdır.</p>
        <p>Ana mantık:</p>
        <ul>
          <li>işlem açıldıktan sonra pozisyon tek parça tutulmaz</li>
          <li>belli seviyelerde küçük kâr kilitleri alınır</li>
          <li>kalan kısım trend sürerse taşınır</li>
          <li>trend bozulursa çıkılır</li>
        </ul>

        <h3>Ana çıkış sistemi çok basit nasıl okunur?</h3>
        <p>Temel omurga şudur:</p>
        <ul>
          <li><b>başlangıç stopu</b> vardır</li>
          <li><b>trend çıkışı</b> vardır</li>
          <li>işlem çok uzarsa maksimum bekleme süresi vardır</li>
        </ul>
        <p>Açık hali:</p>
        <ul>
          <li>başlangıçta 1.5R stop vardır</li>
          <li>fiyat trendi EMA-10 altına bozulursa trend çıkışı çalışır</li>
          <li>hiçbir şey olmazsa işlem en fazla 40 bar tutulur</li>
        </ul>
        <p>Basit anlamı:</p>
        <ul>
          <li>zarar baştan sınırlanır</li>
          <li>trend devam ederse pozisyon taşınır</li>
          <li>trend bozulursa çıkılır</li>
          <li>çok uzun süre sürünürse sonsuza kadar elde tutulmaz</li>
        </ul>

        <h4>Clean hisselerde çıkış nasıl işler?</h4>
        <p>Clean grupta erken küçük kâr kilidi de vardır.</p>
        <p>Clean işlemde:</p>
        <ul>
          <li>+0.7 ATR görülürse pozisyonun %25'i satılır</li>
          <li>+1.5 ATR görülürse %15 daha satılır</li>
          <li>+3.0 ATR görülürse %15 daha satılır</li>
          <li>kalan %45, ana trend çıkış sistemiyle taşınır</li>
        </ul>
        <p>Yani clean hissede:</p>
        <ul>
          <li>önce biraz kâr cebe alınır</li>
          <li>sonra biraz daha alınır</li>
          <li>kalan parça trend devam ederse koşmaya bırakılır</li>
        </ul>

        <h4>Mild ve Elevated hisselerde çıkış nasıl işler?</h4>
        <p>Bu gruplarda clean'deki erken ek satış yoktur.</p>
        <p>Mild / Elevated işlemde:</p>
        <ul>
          <li>+1.5 ATR görülürse %15 satılır</li>
          <li>+3.0 ATR görülürse %15 daha satılır</li>
          <li>kalan %70, ana trend çıkış sistemiyle taşınır</li>
        </ul>
        <p>Yani:</p>
        <ul>
          <li>biraz kâr kilitlenir</li>
          <li>ama pozisyonun büyük kısmı trend için açık bırakılır</li>
        </ul>

        <h4>Parabolic koruma ne demek?</h4>
        <p>Bazen hisse 1 saatlik tek bir barda aşırı sert yukarı patlar.<br>
        Bu durumda sistem ek bir koruma kullanır.</p>
        <p>Eğer 1 saatlik <b>parabolik</b> bir bar oluşursa:</p>
        <ul>
          <li>bir sonraki 1 saat kapanışında pozisyon tamamen kapatılır</li>
        </ul>
        <p>Basit anlamı:</p>
        <ul>
          <li>hisse çok sert patladıysa</li>
          <li>sonra geri vermesin diye</li>
          <li>sistem daha hızlı çıkabilir</li>
        </ul>
        <p>Bu, normal çıkış sisteminin üstüne eklenen özel <b>parabolik koruma</b>dır.</p>

        <h3>Ben bunu pratikte nasıl düşüneceğim?</h3>
        <p>En kısa haliyle:</p>
        <ul>
          <li>liste bana hangi hisselere önce bakacağımı söyler</li>
          <li><b>Tradeable Candidates</b> işlem için daha ciddi gruptur</li>
          <li>girişten sonra sistem kârı parça parça kilitler</li>
          <li>kalan pozisyonu trend sürerse taşır</li>
          <li>çok sert ani patlamalarda ise daha hızlı çıkabilir</li>
        </ul>

        <h3>Önemli uyarı</h3>
        <p>Bu sistem şunu garanti etmez:</p>
        <ul>
          <li>listedeki ilk hisse kesin en iyi gidecek</li>
          <li>her PASS hisse mutlaka kazandıracak</li>
          <li>her işlem tam tepesinden satılacak</li>
        </ul>
        <p>Doğru okuma:</p>
        <ul>
          <li>bu bir aday seçme ve pozisyon yönetme çerçevesidir</li>
          <li>kesinlik değil, daha düzenli karar verme sağlar</li>
        </ul>

        <h3>En kısa özet</h3>
        <ul>
          <li><b>Tradeable Candidates</b> = işlem için daha ciddi bakılacak grup</li>
          <li><b>Watchlist Only</b> = izlenecek ama daha temkinli olunacak grup</li>
          <li>Clean hissede daha erken küçük kâr alınır</li>
          <li>Mild / Elevated hissede kâr daha geç kilitlenir</li>
          <li>Ani parabolik hareketlerde sistem daha hızlı çıkabilir</li>
          <li>Kalan pozisyon trend bozulana kadar taşınır</li>
        </ul>

      </div>
    </details>
    """

    return f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nyx-WinMag: Momentum Continuation Ranker — {target_date.date()}</title>
<style>{_NOX_CSS}

/* ── scan-specific overrides ── */
.nox-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 18px;
  backdrop-filter: blur(8px);
}}
.nox-card h2 {{
  font-family: var(--font-display);
  font-size: 0.95rem; font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--text-primary);
  margin-bottom: 12px;
}}
.nox-card h2 .sub {{
  font-weight: 400; color: var(--text-muted); font-size: 0.85em;
  margin-left: 4px;
}}
.nox-card .card-body {{ font-size: 0.85rem; color: var(--text-secondary); }}
.nox-card .card-footer {{
  margin-top: 10px;
  font-size: 0.72rem; color: var(--text-muted);
  font-family: var(--font-mono);
  border-top: 1px solid var(--border-subtle);
  padding-top: 10px;
}}

/* Banners */
.banner {{
  display: flex; gap: 12px; align-items: flex-start;
  font-size: 0.82rem; line-height: 1.5;
  border-left: 3px solid var(--nox-gold);
  background: rgba(201,169,110,0.04);
}}
.banner b {{ color: var(--text-primary); }}
.banner .banner-icon {{ font-size: 1.2rem; flex-shrink: 0; line-height: 1.2; }}
.banner-sub {{ margin-top: 6px; font-size: 0.78rem; color: var(--text-muted); }}
.banner-sub code {{
  font-family: var(--font-mono); font-size: 0.78rem;
  background: var(--bg-elevated); padding: 1px 6px; border-radius: 4px;
  color: var(--text-secondary);
}}
.warn-banner {{ border-left-color: var(--nox-orange); background: rgba(168,135,106,0.05); }}
.ret-banner {{ border-left-color: var(--nox-green); background: rgba(122,158,122,0.05); }}
.ret-banner-off {{ border-left-color: var(--nox-red); background: rgba(158,90,90,0.05); }}
.warn-card {{ border-left: 3px solid var(--nox-red); }}

/* KPI strip in portfolio */
.kpi-strip {{
  display: flex; flex-wrap: wrap; gap: 18px;
  margin-bottom: 14px; padding: 10px 12px;
  background: var(--bg-elevated); border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
}}
.kpi {{ display: flex; flex-direction: column; gap: 2px; min-width: 75px; }}
.kpi .k {{ font-size: 0.62rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi .v {{ font-family: var(--font-mono); font-weight: 600;
  font-size: 0.88rem; color: var(--text-primary); }}

/* Section count in h2 */
.sec-count {{
  display: inline-block; margin-left: 6px;
  font-family: var(--font-mono); font-size: 0.78em;
  color: var(--nox-cyan); font-weight: 500;
}}

/* Center every header and cell across scan tables */
.nox-table-wrap th,
.nox-table-wrap td {{ text-align: center; vertical-align: middle; }}

/* Header tooltip cue: dotted underline + help cursor */
.nox-table-wrap th.th-tip {{
  text-decoration: underline dotted var(--text-muted);
  text-underline-offset: 3px;
  cursor: help;
}}

/* Make number cells tabular but centered */
td.num {{ font-variant-numeric: tabular-nums;
  font-family: var(--font-mono); }}
.detail-cell {{ white-space: normal; max-width: 320px; }}

td.trade-plan {{
  font-family: var(--font-mono);
  font-size: 0.74rem;
  line-height: 1.35;
  white-space: nowrap;
  color: var(--text-secondary);
  padding: 6px 10px;
}}
td.trade-plan .tp-row {{ display: block; }}
td.trade-plan .tp-head {{
  color: var(--text-primary);
  font-weight: 600;
  margin-bottom: 2px;
  padding-bottom: 2px;
  border-bottom: 1px dashed var(--border-subtle);
}}
td.trade-plan .tp-sl {{ color: var(--nox-red); }}
td.trade-plan .tp-trail {{
  color: var(--text-muted);
  font-style: italic;
  margin-top: 2px;
  padding-top: 2px;
  border-top: 1px dashed var(--border-subtle);
}}
td.trade-plan span {{
  color: var(--text-muted);
  font-size: 0.70rem;
  margin-left: 2px;
}}

/* How-to dropdown */
details.howto-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 18px;
}}
details.howto-card > summary {{
  font-family: var(--font-display);
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--text-primary);
  cursor: pointer;
  list-style: none;
  padding: 4px 0;
  letter-spacing: 0.01em;
}}
details.howto-card > summary::-webkit-details-marker {{ display: none; }}
details.howto-card > summary::before {{
  content: "▸ ";
  color: var(--nox-gold);
  margin-right: 4px;
}}
details.howto-card[open] > summary::before {{ content: "▾ "; }}
details.howto-card .howto-body {{
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-subtle);
  font-size: 0.85rem;
  line-height: 1.55;
  color: var(--text-secondary);
}}
details.howto-card h3 {{
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 16px 0 6px 0;
  letter-spacing: 0.01em;
}}
details.howto-card h3:first-child {{ margin-top: 0; }}
details.howto-card h4 {{
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--nox-gold);
  margin: 12px 0 4px 0;
}}
details.howto-card p {{ margin: 4px 0 8px 0; }}
details.howto-card ul,
details.howto-card ol {{ margin: 4px 0 8px 18px; padding: 0; }}
details.howto-card li {{ margin: 2px 0; }}
details.howto-card b {{ color: var(--text-primary); }}
</style></head>
<body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="nox-container">

  <header class="nox-header">
    <div class="nox-logo">
      Nyx-WinMag<span class="proj"></span>
      <span class="mode">Momentum Continuation Ranker · {target_date.date()}</span>
    </div>
    <div class="nox-meta">
      Generated <b>{now_str}</b><br>
      Dataset: <code>{html.escape(str(dataset_path))}</code>
    </div>
  </header>

  <div class="nox-stats">{stats_html}</div>

  {warn_banner}
  {retention_banner}
  {pf_section}

  {howto_dropdown}

  <section class="nox-card">
    <h2>✅ Tradeable Candidates<span class="sec-count">{len(tradeable_df)}</span></h2>
    <div class="card-body">retention_pass=True · 17:30 proxy için kabul edilen liste</div>
    {_candidates_table(tradeable_rows, "tradeable")}
  </section>

  <section class="nox-card">
    <h2>👁 Watchlist Only<span class="sec-count">{len(watchlist_df)}</span></h2>
    <div class="card-body">ranker'da çıktı ama timing-clean retention'da elendi · gözlem amaçlı, tradeable DEĞİL</div>
    {_candidates_table(watchlist_rows, "watchlist")}
  </section>

  <div class="nox-status">
    nyxexpansion daily scan · <b>{now_str}</b> · v4C ·
    dataset=<code>{html.escape(str(dataset_path))}</code>
  </div>

</div>
</body></html>"""
