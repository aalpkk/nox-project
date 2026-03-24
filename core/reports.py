"""
NOX Project — Raporlama
HTML rapor + Telegram + GitHub Pages push
Iki mod: regime (gunluk) ve dip (haftalik)
"""
import os
import json
import base64
import datetime
import requests
from dotenv import load_dotenv

load_dotenv()
import numpy as np
import pandas as pd
from collections import Counter
from core.config import (
    SIGNAL_EMOJI, SIGNAL_COLORS, SIGNAL_PRIORITY_TREND as SIGNAL_PRIORITY_REGIME,
    SIGNAL_PRIORITY_DIP, SIGNAL_PRIORITY_SIDEWAYS,
    REGIME_COLORS, REGIME_SHORT,
)


# ── TELEGRAM ──

def send_telegram(msg):
    token = os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("⚠️ Telegram yok, konsola yazdırılıyor:")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(msg), 4000):
        chunk = msg[i:i + 4000]
        try:
            requests.post(url, json={
                "chat_id": chat, "text": chunk,
                "parse_mode": "HTML", "disable_web_page_preview": True
            }, timeout=10)
        except:
            pass


def send_telegram_document(filepath):
    token = os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    if not os.path.exists(filepath):
        return
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(filepath, 'rb') as f:
            requests.post(url, data={
                "chat_id": chat, "caption": "📊 NOX — Detaylı rapor"
            }, files={"document": f}, timeout=30)
        print("📤 HTML rapor Telegram'a gönderildi")
    except Exception as e:
        print(f"⚠️ Telegram document hata: {e}")


# ── GITHUB PAGES ──

def push_html_to_github(html_content, filename, date_str):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")
    if not token or not repo:
        print("⚠️ GH_TOKEN/GH_PAGES_REPO tanımlı değil")
        return None
    api_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    content_b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')
    sha = None
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except:
        pass
    payload = {"message": f"NOX {filename} — {date_str}", "content": content_b64, "branch": "main"}
    if sha:
        payload["sha"] = sha

    import time
    for attempt in range(3):
        try:
            resp = requests.put(api_url, headers=headers, json=payload, timeout=30)
            if resp.status_code in (200, 201):
                owner, name = repo.split("/")
                page_url = f"https://{owner}.github.io/{name}/{filename}"
                print(f"✅ HTML rapor yayınlandı: {page_url}")
                return page_url
            elif resp.status_code >= 500 and attempt < 2:
                print(f"  ⚠️ GitHub 5xx ({resp.status_code}), retry {attempt+1}/2...")
                time.sleep(3)
                continue
            else:
                print(f"⚠️ GitHub push hata: {resp.status_code} — {resp.text[:200]}")
                break
        except Exception as e:
            if attempt < 2:
                print(f"  ⚠️ Push timeout/hata, retry {attempt+1}/2...")
                time.sleep(3)
                continue
            print(f"⚠️ GitHub push hata: {e}")
            break
    return None


# ── numpy sanitizer ──

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ═══════════════════════════════════════════
# ORTAK NOX HTML TEMA
# ═══════════════════════════════════════════

_NOX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Sora:wght@300;400;500;600;700;800&display=swap');
:root {
  --bg-primary: #09090b;
  --bg-card: #111113;
  --bg-elevated: #18181b;
  --bg-hover: #1f1f23;
  --border-subtle: #27272a;
  --border-dim: #3f3f46;
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --nox-cyan: #22d3ee;
  --nox-cyan-dim: rgba(34,211,238,0.12);
  --nox-orange: #fb923c;
  --nox-orange-dim: rgba(251,146,60,0.12);
  --nox-green: #4ade80;
  --nox-red: #f87171;
  --nox-purple: #c084fc;
  --nox-yellow: #facc15;
  --nox-blue: #60a5fa;
  --font-display: 'Sora', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --radius: 10px;
  --radius-sm: 6px;
}
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: var(--font-display);
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background:
    radial-gradient(ellipse 80% 60% at 50% -20%, rgba(34,211,238,0.06), transparent),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(251,146,60,0.04), transparent);
  pointer-events: none;
  z-index: 0;
}
.nox-container { position: relative; z-index: 1; max-width: 1440px; margin: 0 auto; padding: 20px 16px; }

/* HEADER */
.nox-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 0; margin-bottom: 24px;
  border-bottom: 1px solid var(--border-subtle);
}
.nox-logo {
  font-family: var(--font-display);
  font-size: 1.6rem; font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--nox-cyan), var(--nox-orange));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.nox-logo .proj { font-weight: 400; -webkit-text-fill-color: var(--text-secondary); font-size: 0.75em; margin-left: 2px; letter-spacing: 0.02em; }
.nox-logo .mode { display: block; font-size: 0.45em; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; -webkit-text-fill-color: var(--text-muted); margin-top: 2px; }
.nox-meta { text-align: right; font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono); }
.nox-meta b { color: var(--nox-cyan); }

/* STAT STRIP */
.nox-stats {
  display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px;
}
.nox-stat {
  display: flex; align-items: center; gap: 6px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 20px; padding: 6px 14px;
  font-size: 0.78rem; font-weight: 500;
  cursor: pointer; transition: all 0.2s;
  user-select: none;
}
.nox-stat:hover { border-color: var(--border-dim); background: var(--bg-elevated); }
.nox-stat.active { border-color: var(--nox-cyan); background: var(--nox-cyan-dim); color: var(--nox-cyan); }
.nox-stat .cnt { font-family: var(--font-mono); font-weight: 700; font-size: 0.85rem; }
.nox-stat .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* FILTER BAR */
.nox-filters {
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  margin-bottom: 16px;
  padding: 10px 14px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
}
.nox-filters label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-right: 2px; }
.nox-filters select, .nox-filters input {
  background: var(--bg-primary); color: var(--text-primary);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  padding: 5px 10px; font-size: 0.78rem; font-family: var(--font-mono);
  outline: none; transition: border-color 0.2s;
}
.nox-filters select:focus, .nox-filters input:focus { border-color: var(--nox-cyan); }
.nox-filters input[type=number] { width: 60px; }
.nox-filters input[type=text] { width: 100px; }
.nox-btn { background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm); padding: 5px 12px; cursor: pointer; font-size: 0.72rem;
  font-family: var(--font-display); font-weight: 500; transition: all 0.15s; }
.nox-btn:hover { color: var(--text-primary); border-color: var(--border-dim); }

/* TABLE */
.nox-table-wrap { overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border-subtle); background: var(--bg-card); }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
thead { position: sticky; top: 0; z-index: 10; }
th {
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-weight: 600; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 10px 8px; text-align: left;
  border-bottom: 1px solid var(--border-subtle);
  cursor: pointer; user-select: none;
  white-space: nowrap; transition: color 0.15s;
}
th:hover { color: var(--nox-cyan); }
th.sorted { color: var(--nox-cyan); }
td { padding: 8px; border-bottom: 1px solid rgba(39,39,42,0.5); white-space: nowrap; font-family: var(--font-mono); font-size: 0.74rem; }
tr { transition: background 0.1s; }
tr:hover { background: var(--bg-hover); }
tr.hl { background: rgba(34,211,238,0.04); }
tr.hidden { display: none; }

/* BADGES */
.sig-badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-weight: 700; font-size: 0.68rem; font-family: var(--font-display);
  letter-spacing: 0.02em; color: #fff;
}
.reg-badge {
  display: inline-block; padding: 2px 8px; border-radius: var(--radius-sm);
  font-size: 0.65rem; font-weight: 600; font-family: var(--font-display);
}
.oe-badge {
  display: inline-block; padding: 2px 7px; border-radius: 8px;
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
  background: rgba(248,113,113,0.15); color: var(--nox-red);
}
.kc-badge {
  display: inline-block; padding: 2px 6px; border-radius: var(--radius-sm);
  font-size: 0.68rem; font-weight: 600; font-family: var(--font-mono);
}
.kc-hi { background: rgba(74,222,128,0.12); color: var(--nox-green); }
.kc-mid { background: rgba(96,165,250,0.12); color: var(--nox-blue); }
.kc-lo { background: rgba(113,113,122,0.12); color: var(--text-muted); }

.tv-link {
  color: var(--text-primary); text-decoration: none; font-weight: 600;
  font-family: var(--font-display); font-size: 0.8rem;
  transition: color 0.15s;
}
.tv-link:hover { color: var(--nox-cyan); }
.rr-val { font-weight: 700; }
.rs-pos { color: var(--nox-green); }
.rs-neg { color: var(--nox-red); }
.detail-cell { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); max-width: 280px; overflow: hidden; text-overflow: ellipsis; }
.detail-cell .kctag { color: var(--nox-blue); }

/* STATUS BAR */
.nox-status {
  text-align: center; padding: 16px 0; margin-top: 16px;
  font-size: 0.72rem; color: var(--text-muted);
  font-family: var(--font-mono);
  border-top: 1px solid var(--border-subtle);
}
.nox-status b { color: var(--nox-cyan); }

/* RESPONSIVE */
@media (max-width: 768px) {
  .nox-header { flex-direction: column; gap: 8px; text-align: center; }
  .nox-meta { text-align: center; }
  table { font-size: 0.7rem; }
  td, th { padding: 5px 4px; }
  .nox-logo { font-size: 1.2rem; }
}
"""


# ═══════════════════════════════════════════
# REGIME HTML
# ═══════════════════════════════════════════

def generate_regime_html(results, total, market_label="BIST"):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    rows_json = json.dumps(_sanitize(results), ensure_ascii=False)
    sig_counts = Counter(r['signal'] for r in results)
    priority = ["COMBO+","COMBO","STRONG","WEAK","REVERSAL","EARLY","BUILDUP","PULLBACK","SQUEEZE","MEANREV","PARTIAL"]

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Trend · {market_label} · {now}</title>
<style>{_NOX_CSS}
.oe-detail {{ display: none; position: absolute; right: 0; top: 100%; z-index: 20;
  background: var(--bg-elevated); border: 1px solid var(--border-dim); border-radius: var(--radius-sm);
  padding: 6px 10px; font-size: 0.65rem; white-space: nowrap; color: var(--text-secondary);
  font-family: var(--font-mono); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.oe-wrap {{ position: relative; display: inline-block; cursor: help; }}
.oe-wrap:hover .oe-detail {{ display: block; }}
.oe-bar {{ display: flex; gap: 2px; align-items: center; height: 10px; }}
.oe-pip {{ width: 6px; height: 6px; border-radius: 50%; }}
.oe-pip.on {{ background: var(--nox-red); box-shadow: 0 0 4px var(--nox-red); }}
.oe-pip.off {{ background: var(--border-subtle); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">trend · {market_label}</span></div>
  <div class="nox-meta"><b>{len(results)}</b> sinyal / {total} taranan<br>{now}</div>
</div>
<div class="nox-stats" id="chips"></div>
<div class="nox-filters">
  <div><label>Rejim</label>
  <select id="fReg" onchange="af()"><option value="">Tümü</option>
  <option value="FULL_TREND">Full Trend</option><option value="TREND">Trend</option>
  <option value="GRI_BOLGE">Gri</option><option value="CHOPPY">Choppy</option></select></div>
  <div><label>RS</label>
  <select id="fRS" onchange="af()"><option value="">Tümü</option>
  <option value="pos">Pozitif</option><option value="hi">RS&gt;20</option></select></div>
  <div><label>Q≥</label><input type="number" id="fQ" value="0" step="10" min="0" oninput="af()"></div>
  <div><label>KC≥</label><input type="number" id="fKC" value="0" step="5" min="0" oninput="af()"></div>
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="af()"></div>
  <div><button class="nox-btn" onclick="reset()">Sıfırla</button></div>
</div>
<div class="nox-table-wrap">
<table><thead><tr>
<th onclick="sb('ticker')">Hisse</th><th onclick="sb('signal')">Sinyal</th>
<th onclick="sb('trade_mode')">Mode</th><th onclick="sb('pos_size')">Pos</th>
<th onclick="sb('regime')">Rejim</th><th onclick="sb('close')">Fiyat</th>
<th onclick="sb('stop')">Stop</th><th onclick="sb('tp')">TP</th>
<th onclick="sb('rr')">R:R</th><th onclick="sb('rs_score')">RS</th>
<th onclick="sb('quality')">Q</th><th onclick="sb('rvol')">RVOL</th>
<th onclick="sb('atr_pctile')">ATR%</th>
<th>OB</th><th>BOS</th><th>CHoCH</th>
<th onclick="sb('overext_score')">OE</th>
<th onclick="sb('kc_score')">RECOVER</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>
<div class="nox-status" id="st"><b>{len(results)}</b> / {len(results)}</div>
</div>
<script>
const TV_PFX='{market_label}:'==':'?'':'{market_label}:';
const D={rows_json};
const SC={json.dumps(SIGNAL_COLORS)};
const RC={json.dumps(REGIME_COLORS)};
const SP={json.dumps(SIGNAL_PRIORITY_REGIME)};
const RS={{"FULL_TREND":"FULL","TREND":"TREND","GRI_BOLGE":"GRİ","CHOPPY":"CHOP"}};
let col='signal',asc=true,chip=null;
function init(){{const cn={{}};D.forEach(r=>cn[r.signal]=(cn[r.signal]||0)+1);
const el=document.getElementById('chips');
{json.dumps(priority)}.forEach(s=>{{if(!cn[s])return;
const d=document.createElement('div');d.className='nox-stat';d.dataset.s=s;
d.innerHTML='<span class="dot" style="background:'+(SC[s]||'#71717a')+'"></span>'+s+' <span class="cnt">'+cn[s]+'</span>';
d.onclick=()=>{{if(chip===s){{chip=null;d.classList.remove('active')}}else{{
document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));chip=s;d.classList.add('active')}};af()}};
el.appendChild(d)}});af()}}
function af(){{const reg=document.getElementById('fReg').value;
const rs=document.getElementById('fRS').value;const q=parseInt(document.getElementById('fQ').value)||0;
const kc=parseInt(document.getElementById('fKC').value)||0;
const sr=document.getElementById('fS').value.toUpperCase();
let f=D.filter(r=>{{if(chip&&r.signal!==chip)return false;if(reg&&r.regime!==reg)return false;
if(rs==='pos'&&r.rs_score<=0)return false;if(rs==='hi'&&r.rs_score<=20)return false;
if(r.quality<q)return false;if(kc>0&&(r.kc_score||0)<kc)return false;
if(sr&&!r.ticker.includes(sr))return false;return true}});
f.sort((a,b)=>{{let va=a[col],vb=b[col];if(col==='signal'){{va=SP[va]||99;vb=SP[vb]||99}};
if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
return asc?(va||0)-(vb||0):(vb||0)-(va||0)}});render(f)}}
function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'}};af()}}
function reset(){{chip=null;document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));
document.getElementById('fReg').value='';document.getElementById('fRS').value='';
document.getElementById('fQ').value='0';document.getElementById('fKC').value='0';
document.getElementById('fS').value='';af()}}
function mkOE(score,tags){{
if(!tags||!tags.length)return '<span style="color:var(--text-muted)">—</span>';
let pips='';for(let i=0;i<5;i++)pips+='<span class="oe-pip '+(i<score?'on':'off')+'"></span>';
const cls=score>=3?'oe-badge':'';
const lbl=score>=3?'⚠'+score:score;
return '<span class="oe-wrap"><span class="'+(score>=3?'oe-badge':'')+'" style="'+(score<3?'color:var(--nox-yellow);font-size:.68rem':'')+'">'+lbl+'</span><div class="oe-detail"><div class="oe-bar">'+pips+'</div><div style="margin-top:4px">'+tags.join('<br>')+'</div></div></span>'}}
function mkKC(score,tags){{
if(!score)return '<span style="color:var(--text-muted)">—</span>';
const cls=score>=50?'kc-hi':score>=30?'kc-mid':'kc-lo';
const t=tags&&tags.length?tags.join(','):'';
return '<span class="oe-wrap"><span class="kc-badge '+cls+'">'+score+'</span>'+(t?'<div class="oe-detail" style="left:0;right:auto">💎 '+t+'</div>':'')+'</span>'}}
function render(data){{const tb=document.getElementById('tb');tb.innerHTML='';
data.forEach(r=>{{const tr=document.createElement('tr');
if(["COMBO+","STRONG"].includes(r.signal)||(r.signal==="COMBO"&&r.rs_score>0&&r.quality>=60))tr.classList.add('hl');
const rsC=r.rs_score>0?'rs-pos':'rs-neg';
const rrC=r.rr>=1.3?'var(--nox-green)':r.rr>=1?'var(--nox-yellow)':'var(--nox-red)';
const modeC=r.trade_mode==='MOMENTUM'?'var(--nox-cyan)':'var(--nox-orange)';
const posC=r.pos_size>=1?'var(--nox-green)':r.pos_size>=0.7?'var(--nox-yellow)':'var(--text-muted)';
const atrP=r.atr_pctile!=null?(r.atr_pctile*100).toFixed(0)+'%':'—';
const atrPC=r.atr_pctile>=0.7?'var(--nox-red)':'var(--text-muted)';
tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=${{TV_PFX}}${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
<td><span class="sig-badge" style="background:${{SC[r.signal]||'#71717a'}}">${{r.signal}}</span></td>
<td style="color:${{modeC}};font-size:.68rem;font-weight:600">${{r.trade_mode||'—'}}</td>
<td style="color:${{posC}}">${{r.pos_size!=null?r.pos_size:'—'}}</td>
<td><span class="reg-badge" style="background:${{RC[r.regime]||'#71717a'}};color:${{RC[r.regime]==='#d1d5db'?'#18181b':'#fff'}}">${{RS[r.regime]||r.regime}}</span></td>
<td>${{r.close}}</td><td>${{r.stop}}</td><td>${{r.tp}} <span style="color:var(--text-muted);font-size:.65rem">${{r.tp_src}}</span></td>
<td class="rr-val" style="color:${{rrC}}">${{r.rr}}</td>
<td class="${{rsC}}">${{r.rs_score}}</td><td>${{r.quality}}</td>
<td style="color:${{r.rvol>=1.5?'var(--nox-orange)':'var(--text-muted)'}}">${{r.rvol}}x</td>
<td style="color:${{atrPC}}">${{atrP}}</td>
<td style="font-size:.65rem;color:var(--text-muted)">${{r.ob_resist||'—'}}</td>
<td style="color:var(--text-muted)">${{r.bos_age!=null?r.bos_age+'g':'—'}}</td>
<td style="color:var(--text-muted)">${{r.choch_age!=null?r.choch_age+'g':'—'}}</td>
<td>${{mkOE(r.overext_score||0,r.overext_tags||[])}}</td>
<td>${{mkKC(r.kc_score||0,r.kc_tags||[])}}</td>`;tb.appendChild(tr)}});
document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length}}
init();
</script></body></html>"""
    return html


# ═══════════════════════════════════════════
# DIP HTML
# ═══════════════════════════════════════════

def generate_dip_html(results, total, market_label="BIST"):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    sig_counts = Counter(r['signal'] for r in results)
    sig_order = ["DIP+","DIP","DIP_E","RECOVER","DIP_W"]
    rows_json = json.dumps(_sanitize(results), ensure_ascii=False)

    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — DIP · {market_label} · {now}</title>
<style>{_NOX_CSS}
.oe-detail {{ display: none; position: absolute; z-index: 20;
  background: var(--bg-elevated); border: 1px solid var(--border-dim); border-radius: var(--radius-sm);
  padding: 6px 10px; font-size: 0.65rem; white-space: nowrap; color: var(--text-secondary);
  font-family: var(--font-mono); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.oe-wrap {{ position: relative; display: inline-block; cursor: help; }}
.oe-wrap:hover .oe-detail {{ display: block; left: 0; top: 100%; }}
.oe-bar {{ display: flex; gap: 2px; align-items: center; height: 10px; }}
.oe-pip {{ width: 6px; height: 6px; border-radius: 50%; }}
.oe-pip.on {{ background: var(--nox-red); box-shadow: 0 0 4px var(--nox-red); }}
.oe-pip.off {{ background: var(--border-subtle); }}
.detail-tags {{ font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); }}
.detail-tags .t {{ display: inline-block; padding: 1px 5px; margin: 1px; border-radius: 4px;
  background: rgba(96,165,250,0.1); color: var(--nox-blue); font-size: 0.6rem; }}
.detail-tags .pk {{ color: var(--nox-purple); background: rgba(192,132,252,0.1); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">dip · {market_label}</span></div>
  <div class="nox-meta"><b>{len(results)}</b> sinyal / {total} taranan<br>{now}</div>
</div>
<div class="nox-stats" id="chips"></div>
<div class="nox-filters">
  <div><label>Rejim</label>
  <select id="fReg" onchange="af()"><option value="">Tümü</option>
  <option value="FULL_TREND">Full Trend</option><option value="TREND">Trend</option>
  <option value="GRI_BOLGE">Gri</option><option value="CHOPPY">Choppy</option></select></div>
  <div><label>RS</label>
  <select id="fRS" onchange="af()"><option value="">Tümü</option>
  <option value="pos">Pozitif</option><option value="neg">Negatif</option></select></div>
  <div><label>R:R≥</label><input type="number" id="fRR" value="0" step="0.5" min="0" oninput="af()"></div>
  <div><label>KC≥</label><input type="number" id="fKC" value="0" step="5" min="0" oninput="af()"></div>
  <div><label>Q≥</label><input type="number" id="fQ" value="0" step="10" min="0" oninput="af()"></div>
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="af()"></div>
  <div><button class="nox-btn" onclick="reset()">Sıfırla</button></div>
</div>
<div class="nox-table-wrap">
<table><thead><tr>
<th onclick="sb('ticker')">Hisse</th><th onclick="sb('signal')">Sinyal</th>
<th onclick="sb('regime')">Rejim</th><th onclick="sb('close')">Fiyat</th>
<th onclick="sb('stop')">Stop</th><th onclick="sb('tp')">TP</th>
<th onclick="sb('rr')">R:R</th><th onclick="sb('rs_score')">RS</th>
<th onclick="sb('quality')">Q</th><th onclick="sb('rvol')">RVOL</th>
<th onclick="sb('overext_score')">OE</th>
<th onclick="sb('kc_score')">RECOVER</th><th>Detay</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>
<div class="nox-status" id="st"><b>{len(results)}</b> / {len(results)}</div>
</div>
<script>
const TV_PFX='{market_label}:'==':'?'':'{market_label}:';
const D={rows_json};
const SC={json.dumps(SIGNAL_COLORS)};
const RC={json.dumps(REGIME_COLORS)};
const SP={json.dumps(SIGNAL_PRIORITY_DIP)};
const RS={{"FULL_TREND":"FULL","TREND":"TREND","GRI_BOLGE":"GRİ","CHOPPY":"CHOP"}};
let col='signal',asc=true,chip=null;
function init(){{const cn={{}};D.forEach(r=>cn[r.signal]=(cn[r.signal]||0)+1);
const el=document.getElementById('chips');
const allChip=document.createElement('div');allChip.className='nox-stat active';allChip.dataset.s='';
allChip.innerHTML='<span class="dot" style="background:var(--nox-cyan)"></span>Tümü <span class="cnt">'+D.length+'</span>';
allChip.onclick=()=>{{chip=null;document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));allChip.classList.add('active');af()}};
el.appendChild(allChip);
{json.dumps(sig_order)}.forEach(s=>{{if(!cn[s])return;
const d=document.createElement('div');d.className='nox-stat';d.dataset.s=s;
d.innerHTML='<span class="dot" style="background:'+(SC[s]||'#71717a')+'"></span>'+s+' <span class="cnt">'+cn[s]+'</span>';
d.onclick=()=>{{document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));
if(chip===s){{chip=null;allChip.classList.add('active')}}else{{chip=s;d.classList.add('active')}};af()}};
el.appendChild(d)}});af()}}
function af(){{const reg=document.getElementById('fReg').value;
const rs=document.getElementById('fRS').value;
const minRR=parseFloat(document.getElementById('fRR').value)||0;
const minKC=parseInt(document.getElementById('fKC').value)||0;
const minQ=parseInt(document.getElementById('fQ').value)||0;
const sr=document.getElementById('fS').value.toUpperCase();
let f=D.filter(r=>{{if(chip&&r.signal!==chip)return false;if(reg&&r.regime!==reg)return false;
if(rs==='pos'&&r.rs_score<=0)return false;if(rs==='neg'&&r.rs_score>=0)return false;
if(r.rr<minRR)return false;if(minKC>0&&(r.kc_score||0)<minKC)return false;
if(r.quality<minQ)return false;if(sr&&!r.ticker.includes(sr))return false;return true}});
f.sort((a,b)=>{{let va=a[col],vb=b[col];if(col==='signal'){{va=SP[va]||99;vb=SP[vb]||99}};
if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
return asc?(va||0)-(vb||0):(vb||0)-(va||0)}});render(f)}}
function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'}};af()}}
function reset(){{chip=null;document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));
document.querySelector('.nox-stat').classList.add('active');
document.getElementById('fReg').value='';document.getElementById('fRS').value='';
document.getElementById('fRR').value='0';document.getElementById('fKC').value='0';
document.getElementById('fQ').value='0';document.getElementById('fS').value='';af()}}
function mkOE(score,tags){{
if(!tags||!tags.length)return '<span style="color:var(--text-muted)">—</span>';
let pips='';for(let i=0;i<5;i++)pips+='<span class="oe-pip '+(i<score?'on':'off')+'"></span>';
const lbl=score>=3?'⚠'+score:score;
return '<span class="oe-wrap"><span class="'+(score>=3?'oe-badge':'')+'" style="'+(score<3?'color:var(--nox-yellow);font-size:.68rem':'')+'">'+lbl+'</span><div class="oe-detail"><div class="oe-bar">'+pips+'</div><div style="margin-top:4px">'+tags.join('<br>')+'</div></div></span>'}}
function mkKC(score,tags){{
if(!score)return '<span style="color:var(--text-muted)">—</span>';
const cls=score>=50?'kc-hi':score>=30?'kc-mid':'kc-lo';
const t=tags&&tags.length?tags.join(', '):'';
return '<span class="oe-wrap"><span class="kc-badge '+cls+'">💎 '+score+'</span>'+(t?'<div class="oe-detail">'+t+'</div>':'')+'</span>'}}
function mkDetail(r){{
let parts=[];
const pd=r.pink_detail||'';if(pd)parts.push('<span class="pk">'+pd.replace(/</g,'&lt;')+'</span>');
const kt=r.kc_tags||[];if(kt.length)kt.forEach(t=>parts.push('<span class="t">'+t+'</span>'));
return parts.length?'<span class="detail-tags">'+parts.join(' ')+'</span>':'<span style="color:var(--text-muted)">—</span>'}}
function render(data){{const tb=document.getElementById('tb');tb.innerHTML='';
data.forEach(r=>{{const tr=document.createElement('tr');
if(["DIP+","DIP"].includes(r.signal))tr.classList.add('hl');
const rsC=r.rs_score>0?'rs-pos':'rs-neg';
const rrC=r.rr>=2?'var(--nox-green)':r.rr>=1?'var(--nox-yellow)':'var(--nox-red)';
tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=${{TV_PFX}}${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
<td><span class="sig-badge" style="background:${{SC[r.signal]||'#71717a'}}">${{r.signal}}</span></td>
<td><span class="reg-badge" style="background:${{RC[r.regime]||'#71717a'}};color:#fff">${{RS[r.regime]||r.regime}}</span></td>
<td>${{r.close}}</td><td>${{r.stop}}</td><td>${{r.tp}} <span style="color:var(--text-muted);font-size:.65rem">${{r.tp_src}}</span></td>
<td class="rr-val" style="color:${{rrC}}">${{r.rr}}</td>
<td class="${{rsC}}">${{r.rs_score}}</td><td>${{r.quality}}</td>
<td style="color:${{r.rvol>=1.5?'var(--nox-orange)':'var(--text-muted)'}}">${{r.rvol}}x</td>
<td>${{mkOE(r.overext_score||0,r.overext_tags||[])}}</td>
<td>${{mkKC(r.kc_score||0,r.kc_tags||[])}}</td>
<td>${{mkDetail(r)}}</td>`;tb.appendChild(tr)}});
document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length}}
init();
</script></body></html>"""
    return html


# ═══════════════════════════════════════════
# TELEGRAM MESAJLARI
# ═══════════════════════════════════════════

def format_regime_telegram(results, total, html_url=None):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    signal_groups = {}
    for r in results:
        signal_groups.setdefault(r['signal'], []).append(r)
    priority = ["COMBO+","COMBO","STRONG","WEAK","REVERSAL","EARLY","BUILDUP","PULLBACK","SQUEEZE","MEANREV"]

    lines = [f"<b>⬡ NOX Regime — {now}</b>", ""]
    if not results:
        lines.append("Bugün sinyal yok. 🔇")
        lines.append(f"📋 Taranan: {total}")
        return "\n".join(lines)

    dist = []
    for sig in priority:
        cnt = len(signal_groups.get(sig, []))
        if cnt > 0:
            dist.append(f"{SIGNAL_EMOJI.get(sig,'')}{sig}:{cnt}")
    partial_cnt = len(signal_groups.get("PARTIAL", []))
    lines.append(f"📋 {total} taranan | {len(results)} sinyal" + (f" (+{partial_cnt} partial)" if partial_cnt else ""))
    lines.append(" ".join(dist))
    lines.append("")

    top_signals = []
    for sig in ["COMBO+","STRONG","REVERSAL","EARLY","BUILDUP","PULLBACK","SQUEEZE","MEANREV"]:
        for r in signal_groups.get(sig, []):
            top_signals.append(r)
    cmb_best = sorted([r for r in signal_groups.get("COMBO", []) if r['rs_score'] > 0 and r['quality'] >= 50],
                       key=lambda x: (-x['quality'], -x['rs_score']))[:5]
    top_signals.extend(cmb_best)
    weak_best = sorted(signal_groups.get("WEAK", []), key=lambda x: (-x['quality'], -x['rs_score']))[:3]
    top_signals.extend(weak_best)
    top_signals.sort(key=lambda x: (SIGNAL_PRIORITY_REGIME.get(x['signal'], 99), -x['rr']))

    lines.append(f"<b>⭐ Öne Çıkanlar ({len(top_signals)})</b>")
    lines.append("─────────────────")
    for r in top_signals:
        emoji = SIGNAL_EMOJI.get(r['signal'], "")
        rs = REGIME_SHORT.get(r['regime'], "?")
        oe = f" ⚠OE{r['overext_score']}" if r.get('overext_warning') else ""
        mode_tag = f" [{r['trade_mode']}]" if r.get('trade_mode') else ""
        pos_tag = f" pos:{r['pos_size']}" if r.get('pos_size') is not None else ""
        lines.append(f"{emoji}<b>{r['ticker']}</b> {r['close']} [{r['signal']}]{mode_tag} {rs}{oe}\n"
                     f"  S:{r['stop']} TP:{r['tp']} R:R={r['rr']} RS:{r['rs_score']} Q:{r['quality']}{pos_tag}")
    lines.append("")
    if html_url:
        lines.append(f"🔗 <a href=\"{html_url}\">NOX Rapor</a>")
    lines.append(f"\n📋 Taranan: {total} | Toplam: {len(results)}")
    return "\n".join(lines)


def format_dip_telegram(results, total, html_url=None):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    sig_counts = Counter(r['signal'] for r in results)
    sig_order = ["DIP+","DIP","DIP_E","RECOVER","DIP_W"]

    lines = [f"<b>⬡ NOX DIP — {now}</b>\n"]
    lines.append(f"📋 {total} taranan | {len(results)} sinyal")
    sig_line = " ".join(f"{SIGNAL_EMOJI.get(s,'')}{s}:{c}" for s, c in sig_counts.items())
    lines.append(sig_line)

    top = [r for r in results if r['signal'] in ("DIP+","DIP")]
    if top:
        lines.append(f"\n<b>⭐ DIP Safe ({len(top)})</b>")
        for r in top[:20]:
            rs = REGIME_SHORT.get(r['regime'], "?")
            kc_tags = ",".join(r.get('kc_tags', []))
            kc_str = f" 💎{kc_tags}" if kc_tags else ""
            lines.append(f"🔸<b>{r['ticker']}</b> {r['close']} [{r['signal']}] {rs}")
            lines.append(f"  S:{r['stop']} TP:{r['tp']} R:R={r['rr']} RS:{r['rs_score']} Q:{r['quality']}{kc_str}")

    early = [r for r in results if r['signal'] == "DIP_E"]
    if early:
        lines.append(f"\n<b>📙 DIP_E ({len(early)})</b>")
        for r in early[:15]:
            rs = REGIME_SHORT.get(r['regime'], "?")
            lines.append(f"📙<b>{r['ticker']}</b> {r['close']} R:R={r['rr']} RS:{r['rs_score']} {rs}")

    recover = [r for r in results if r['signal'] == "RECOVER"]
    if recover:
        lines.append(f"\n<b>💎 RECOVER ({len(recover)})</b>")
        for r in recover[:15]:
            kc_tags = ",".join(r.get('kc_tags', []))
            lines.append(f"💎<b>{r['ticker']}</b> {r['close']} | {kc_tags}")

    watch = [r for r in results if r['signal'] == "DIP_W"]
    if watch:
        tickers_str = " ".join(r['ticker'] for r in watch[:20])
        lines.append(f"\n👀 DIP_W ({len(watch)}): {tickers_str}")

    if html_url:
        lines.append(f"\n🔗 <a href=\"{html_url}\">NOX Rapor</a>")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# SIDEWAYS HTML
# ═══════════════════════════════════════════

def generate_sideways_html(results, total, market_label="BIST"):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    rows_json = json.dumps(_sanitize(results), ensure_ascii=False)
    sig_counts = Counter(r['signal'] for r in results)
    priority = ["SIDEWAYS_SQ", "SIDEWAYS_MR"]

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Sideways · {market_label} · {now}</title>
<style>{_NOX_CSS}
.oe-detail {{ display: none; position: absolute; right: 0; top: 100%; z-index: 20;
  background: var(--bg-elevated); border: 1px solid var(--border-dim); border-radius: var(--radius-sm);
  padding: 6px 10px; font-size: 0.65rem; white-space: nowrap; color: var(--text-secondary);
  font-family: var(--font-mono); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.oe-wrap {{ position: relative; display: inline-block; cursor: help; }}
.oe-wrap:hover .oe-detail {{ display: block; }}
.oe-bar {{ display: flex; gap: 2px; align-items: center; height: 10px; }}
.oe-pip {{ width: 6px; height: 6px; border-radius: 50%; }}
.oe-pip.on {{ background: var(--nox-red); box-shadow: 0 0 4px var(--nox-red); }}
.oe-pip.off {{ background: var(--border-subtle); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">sideways · {market_label}</span></div>
  <div class="nox-meta"><b>{len(results)}</b> sinyal / {total} taranan<br>{now}</div>
</div>
<div class="nox-stats" id="chips"></div>
<div class="nox-filters">
  <div><label>Sinyal</label>
  <select id="fSig" onchange="af()"><option value="">Tümü</option>
  <option value="SIDEWAYS_SQ">Squeeze</option><option value="SIDEWAYS_MR">Mean Rev</option></select></div>
  <div><label>RS</label>
  <select id="fRS" onchange="af()"><option value="">Tümü</option>
  <option value="pos">Pozitif</option></select></div>
  <div><label>Q≥</label><input type="number" id="fQ" value="0" step="10" min="0" oninput="af()"></div>
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="af()"></div>
  <div><button class="nox-btn" onclick="reset()">Sıfırla</button></div>
</div>
<div class="nox-table-wrap">
<table><thead><tr>
<th onclick="sb('ticker')">Hisse</th><th onclick="sb('signal')">Sinyal</th>
<th onclick="sb('module')">Modul</th><th onclick="sb('pos_size')">Pos</th>
<th onclick="sb('close')">Fiyat</th><th onclick="sb('stop')">Stop</th>
<th onclick="sb('tp')">TP</th><th onclick="sb('rr')">R:R</th>
<th onclick="sb('rs_score')">RS</th><th onclick="sb('quality')">Q</th>
<th onclick="sb('rvol')">RVOL</th><th onclick="sb('bb_w_pctile')">BB_W%</th>
<th onclick="sb('atr_pctile')">ATR%</th>
<th onclick="sb('overext_score')">OE</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>
<div class="nox-status" id="st"><b>{len(results)}</b> / {len(results)}</div>
</div>
<script>
const TV_PFX='{market_label}:'==':'?'':'{market_label}:';
const D={rows_json};
const SC={json.dumps(SIGNAL_COLORS)};
const SP={json.dumps(SIGNAL_PRIORITY_SIDEWAYS)};
let col='signal',asc=true,chip=null;
function init(){{const cn={{}};D.forEach(r=>cn[r.signal]=(cn[r.signal]||0)+1);
const el=document.getElementById('chips');
{json.dumps(priority)}.forEach(s=>{{if(!cn[s])return;
const d=document.createElement('div');d.className='nox-stat';d.dataset.s=s;
d.innerHTML='<span class="dot" style="background:'+(SC[s]||'#71717a')+'"></span>'+s+' <span class="cnt">'+cn[s]+'</span>';
d.onclick=()=>{{if(chip===s){{chip=null;d.classList.remove('active')}}else{{
document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));chip=s;d.classList.add('active')}};af()}};
el.appendChild(d)}});af()}}
function af(){{const sig=document.getElementById('fSig').value;
const rs=document.getElementById('fRS').value;const q=parseInt(document.getElementById('fQ').value)||0;
const sr=document.getElementById('fS').value.toUpperCase();
let f=D.filter(r=>{{if(chip&&r.signal!==chip)return false;if(sig&&r.signal!==sig)return false;
if(rs==='pos'&&r.rs_score<=0)return false;if(r.quality<q)return false;
if(sr&&!r.ticker.includes(sr))return false;return true}});
f.sort((a,b)=>{{let va=a[col],vb=b[col];if(col==='signal'){{va=SP[va]||99;vb=SP[vb]||99}};
if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
return asc?(va||0)-(vb||0):(vb||0)-(va||0)}});render(f)}}
function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'}};af()}}
function reset(){{chip=null;document.querySelectorAll('.nox-stat').forEach(x=>x.classList.remove('active'));
document.getElementById('fSig').value='';document.getElementById('fRS').value='';
document.getElementById('fQ').value='0';document.getElementById('fS').value='';af()}}
function mkOE(score,tags){{
if(!tags||!tags.length)return '<span style="color:var(--text-muted)">—</span>';
let pips='';for(let i=0;i<5;i++)pips+='<span class="oe-pip '+(i<score?'on':'off')+'"></span>';
const lbl=score>=3?'⚠'+score:score;
return '<span class="oe-wrap"><span class="'+(score>=3?'oe-badge':'')+'" style="'+(score<3?'color:var(--nox-yellow);font-size:.68rem':'')+'">'+lbl+'</span><div class="oe-detail"><div class="oe-bar">'+pips+'</div><div style="margin-top:4px">'+tags.join('<br>')+'</div></div></span>'}}
function render(data){{const tb=document.getElementById('tb');tb.innerHTML='';
data.forEach(r=>{{const tr=document.createElement('tr');
const rsC=r.rs_score>0?'rs-pos':'rs-neg';
const rrC=r.rr>=1.5?'var(--nox-green)':r.rr>=1?'var(--nox-yellow)':'var(--nox-red)';
const posC=r.pos_size>=0.8?'var(--nox-green)':'var(--text-muted)';
const atrP=r.atr_pctile!=null?(r.atr_pctile*100).toFixed(0)+'%':'—';
const bbW=r.bb_w_pctile!=null?r.bb_w_pctile.toFixed(0)+'%':'—';
tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=${{TV_PFX}}${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
<td><span class="sig-badge" style="background:${{SC[r.signal]||'#71717a'}}">${{r.signal}}</span></td>
<td style="color:var(--text-muted);font-size:.68rem">${{r.module||'—'}}</td>
<td style="color:${{posC}}">${{r.pos_size!=null?r.pos_size:'—'}}</td>
<td>${{r.close}}</td><td>${{r.stop}}</td><td>${{r.tp}} <span style="color:var(--text-muted);font-size:.65rem">${{r.tp_src}}</span></td>
<td class="rr-val" style="color:${{rrC}}">${{r.rr}}</td>
<td class="${{rsC}}">${{r.rs_score}}</td><td>${{r.quality}}</td>
<td style="color:${{r.rvol>=1.5?'var(--nox-orange)':'var(--text-muted)'}}">${{r.rvol}}x</td>
<td>${{bbW}}</td><td>${{atrP}}</td>
<td>${{mkOE(r.overext_score||0,r.overext_tags||[])}}</td>`;tb.appendChild(tr)}});
document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length}}
init();
</script></body></html>"""
    return html


# ═══════════════════════════════════════════
# SIDEWAYS TELEGRAM
# ═══════════════════════════════════════════

def format_sideways_telegram(results, total, html_url=None):
    now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
    sig_counts = Counter(r['signal'] for r in results)

    lines = [f"<b>⬡ NOX Sideways — {now}</b>", ""]
    if not results:
        lines.append("Sideways modda sinyal yok. 🔇")
        lines.append(f"📋 Taranan: {total}")
        return "\n".join(lines)

    lines.append(f"📋 {total} taranan | {len(results)} sinyal")
    dist = []
    for sig in ["SIDEWAYS_SQ", "SIDEWAYS_MR"]:
        cnt = sig_counts.get(sig, 0)
        if cnt > 0:
            dist.append(f"{SIGNAL_EMOJI.get(sig, '')}{sig}:{cnt}")
    lines.append(" ".join(dist))
    lines.append("")

    # Squeeze breakouts
    sq = [r for r in results if r['signal'] == "SIDEWAYS_SQ"]
    if sq:
        lines.append(f"<b>🔶 Squeeze Breakout ({len(sq)})</b>")
        lines.append("─────────────────")
        for r in sq[:20]:
            oe = f" ⚠OE{r['overext_score']}" if r.get('overext_warning') else ""
            lines.append(f"🔶<b>{r['ticker']}</b> {r['close']} [{r['signal']}] pos:{r['pos_size']}{oe}\n"
                         f"  S:{r['stop']} TP:{r['tp']} R:R={r['rr']} RS:{r['rs_score']} Q:{r['quality']}")
        lines.append("")

    # Mean reversion
    mr = [r for r in results if r['signal'] == "SIDEWAYS_MR"]
    if mr:
        lines.append(f"<b>🔷 Mean Reversion ({len(mr)})</b>")
        lines.append("─────────────────")
        for r in mr[:20]:
            lines.append(f"🔷<b>{r['ticker']}</b> {r['close']} [{r['signal']}] pos:{r['pos_size']}\n"
                         f"  S:{r['stop']} TP:{r['tp']} R:R={r['rr']} RS:{r['rs_score']} Q:{r['quality']}")
        lines.append("")

    if html_url:
        lines.append(f"🔗 <a href=\"{html_url}\">NOX Rapor</a>")
    lines.append(f"\n📋 Taranan: {total} | Toplam: {len(results)}")
    return "\n".join(lines)
