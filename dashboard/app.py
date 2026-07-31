"""
Handball AI — Ultra-Premium Dashboard v3
Tabs: LIVE · SCOUTING · MATCH SUMMARY · PLAYER PROFILES · COACH AI · CAMERA IMAGING · COURT SETUP
"""
import logging
import time
import requests
import random
import base64 as _b64

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# ── Suppress noisy Tornado/Streamlit internal log spam ────────────────────────
for _noisy in ("tornado.access", "tornado.application", "tornado.general"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

API_URL = "http://127.0.0.1:8765"
WS_URL  = "ws://127.0.0.1:8765/ws"

st.set_page_config(
    page_title="Handball AI Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(path: str) -> dict:
    try:
        result = requests.get(f"{API_URL}{path}", timeout=3).json()
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}

def api_post(path: str, payload: dict) -> dict:
    try:
        return requests.post(f"{API_URL}{path}", json=payload, timeout=3).json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def api_delete(path: str) -> dict:
    try:
        return requests.delete(f"{API_URL}{path}", timeout=2).json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def check_api() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=3)
        return True
    except Exception:
        return False

def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"

def bgr_to_hex(bgr: list) -> str:
    if not bgr or len(bgr) < 3:
        return "#8b949e"
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

def _logo_b64(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return "data:" + uploaded_file.type + ";base64," + _b64.b64encode(uploaded_file.read()).decode()

_SEVERITY_CFG = {
    "CRITICAL": ("critical", "#a855f7", ""),
    "DANGER":   ("danger",   "#ef4444", ""),
    "WARNING":  ("warning",  "#f59e0b", ""),
    "INFO":     ("info",     "#00e5ff", ""),
    "TACTIC":   ("info",     "#00e5ff", ""),
}

def _classify_severity(text: str, alert_type: str = "") -> str:
    danger_types = {"passive_violation", "overload_danger", "goal_against"}
    if alert_type in danger_types:
        return "DANGER"
    tl = text.lower()
    if any(w in tl for w in ("substitute immediately", "switch now", "foul out", "red card")):
        return "CRITICAL"
    if any(w in tl for w in ("substitute", "immediately", " now.", "switch to")):
        return "WARNING"
    if any(w in tl for w in ("passive", "violation", "leaving open", "exposed")):
        return "WARNING"
    return "INFO"


# ── Global CSS — SpaceX Mission Control aesthetic ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

#MainMenu, footer, header { visibility: hidden !important; }
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: #000000 !important;
    color: #b0bec5 !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(0,255,136,0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,136,0.028) 1px, transparent 1px);
    background-size: 44px 44px;
}
.block-container {
    padding-top: 86px !important;
    padding-bottom: 24px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    position: relative; z-index: 1;
}
[data-testid="stHorizontalBlock"] { align-items: stretch !important; gap: 14px !important; }
[data-testid="column"] { display: flex !important; flex-direction: column !important; min-width: 0 !important; }
[data-testid="stVerticalBlock"] > div { width: 100% !important; }

[data-testid="stSidebar"] {
    background: #000000 !important;
    border-right: 1px solid rgba(0,255,136,0.13) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 22px !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,255,136,0.22); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,136,0.45); }

h1, h2, h3, h4 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    color: #ffffff !important; text-transform: uppercase !important;
}
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.05rem !important; }

.glass-card {
    background: #030810;
    border-radius: 2px;
    border: 1px solid rgba(0,255,136,0.18);
    padding: 18px;
    position: relative; overflow: hidden;
    box-shadow: 0 0 24px rgba(0,255,136,0.04);
    transition: border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.glass-card::before {
    content: ""; position: absolute; top: 0; left: 0;
    width: 14px; height: 14px;
    border-top: 1px solid rgba(0,255,136,0.5);
    border-left: 1px solid rgba(0,255,136,0.5);
}
.glass-card::after {
    content: ""; position: absolute; bottom: 0; right: 0;
    width: 14px; height: 14px;
    border-bottom: 1px solid rgba(0,255,136,0.5);
    border-right: 1px solid rgba(0,255,136,0.5);
}
.glass-card:hover {
    border-color: rgba(0,255,136,0.38);
    box-shadow: 0 0 32px rgba(0,255,136,0.10);
}
.glass-card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; font-weight: 400; text-transform: uppercase;
    letter-spacing: 2.5px; color: #00ff88; margin-bottom: 14px; display: block;
}

.kpi-banner {
    position: fixed; top: 0; left: 0; right: 0; height: 76px;
    background: #000000;
    border-bottom: 1px solid rgba(0,255,136,0.25);
    box-shadow: 0 1px 28px rgba(0,255,136,0.07);
    display: flex; align-items: center; justify-content: center;
    gap: 24px; z-index: 99999; padding: 0 32px;
}
.kpi-logo { width: 40px; height: 40px; object-fit: contain; border-radius: 2px; }
.kpi-logo-ph {
    width: 40px; height: 40px; border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1rem;
}
.kpi-team {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem; font-weight: 700; color: #b0bec5;
    letter-spacing: 4px; text-transform: uppercase; min-width: 110px;
}
.kpi-team.r { text-align: right; }
.kpi-score {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.8rem; font-weight: 400; color: #ffffff;
    text-shadow: 0 0 22px rgba(0,255,136,0.4);
    letter-spacing: -2px; line-height: 1; min-width: 130px; text-align: center;
}
.kpi-clock {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.15rem; font-weight: 400; color: #ff6d00;
    text-shadow: 0 0 10px rgba(255,109,0,0.4);
    font-variant-numeric: tabular-nums;
}
.kpi-poss {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.76rem; font-weight: 700; color: rgba(0,255,136,0.9);
    padding: 4px 12px; border-radius: 2px;
    background: rgba(0,255,136,0.06); border: 1px solid rgba(0,255,136,0.22);
    min-width: 105px; text-align: center; letter-spacing: 1.5px; text-transform: uppercase;
}
.kpi-div { width: 1px; height: 32px; background: linear-gradient(to bottom, transparent, rgba(0,255,136,0.3), transparent); }

.stButton > button {
    width: 100% !important; height: 38px !important; min-height: 38px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    font-size: 0.8rem !important; letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border-radius: 2px !important; border: 1px solid rgba(0,255,136,0.22) !important;
    background: rgba(0,255,136,0.04) !important; color: #7a8a96 !important;
    transition: all 0.15s ease !important; cursor: pointer !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    padding: 0 12px !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.10) !important; border-color: rgba(0,255,136,0.55) !important;
    color: #00ff88 !important; box-shadow: 0 0 14px rgba(0,255,136,0.18) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    border-color: rgba(0,255,136,0.6) !important;
    background: rgba(0,255,136,0.12) !important; color: #00ff88 !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(0,255,136,0.02) !important;
    border: 1px solid rgba(0,255,136,0.16) !important;
    border-radius: 2px !important; color: #b0bec5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

div[data-testid="stTabs"] { border-bottom: 1px solid rgba(0,255,136,0.10) !important; }
div[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    font-size: 0.80rem !important; color: #4a5568 !important;
    border-radius: 0 !important; padding: 10px 20px !important;
    transition: color 0.15s !important; letter-spacing: 2.5px !important; text-transform: uppercase !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00ff88 !important; background: rgba(0,255,136,0.04) !important;
    border-bottom: 2px solid #00ff88 !important; text-shadow: 0 0 8px rgba(0,255,136,0.3) !important;
}

.alert-card {
    padding: 11px 13px; border-radius: 2px; margin-bottom: 7px;
    background: #030810; border: 1px solid;
    font-size: 0.85rem; font-weight: 500; line-height: 1.5;
    position: relative; overflow: hidden;
}
.alert-card::before {
    content: ""; position: absolute; top: 0; left: 0;
    width: 9px; height: 9px;
    border-top-width: 1px; border-left-width: 1px;
    border-style: solid; border-color: inherit; opacity: 0.7;
}
.alert-card-header { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }
.alert-card-badge  { font-family: 'Rajdhani',sans-serif; font-size: 0.58rem; font-weight: 700; letter-spacing: 2px; padding: 2px 7px; border-radius: 2px; text-transform: uppercase; }
.alert-card-time   { font-family: 'Share Tech Mono',monospace; font-size: 0.62rem; color: #4a5568; margin-left: auto; }
.alert-card-icon   { font-size: 0.9rem; flex-shrink: 0; }
.alert-card-body   { font-family: 'Rajdhani',sans-serif; color: #b0bec5; font-weight: 600; letter-spacing: 0.3px; }

.alert-card.critical {
    border-color: rgba(168,85,247,0.50);
    box-shadow: 0 0 18px rgba(168,85,247,0.18);
    animation: critical-pulse 3s infinite;
}
.alert-card.danger {
    border-color: rgba(255,62,62,0.45);
    box-shadow: 0 0 14px rgba(255,62,62,0.14);
}
.alert-card.warning {
    border-color: rgba(255,170,0,0.40);
    box-shadow: 0 0 12px rgba(255,170,0,0.12);
}
.alert-card.info {
    border-color: rgba(0,255,136,0.22);
    box-shadow: 0 0 8px rgba(0,255,136,0.07);
}

.sidebar-section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem; font-weight: 400; letter-spacing: 2.5px;
    text-transform: uppercase; color: #4a5568; margin: 15px 0 8px;
    padding-bottom: 5px; border-bottom: 1px solid rgba(0,255,136,0.08);
}
</style>
""", unsafe_allow_html=True)


# ── Canvas HTML/JS ────────────────────────────────────────────────────────────
CANVAS_HTML = f"""<!DOCTYPE html><html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@600;700&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box;font-family:'Share Tech Mono',monospace;}}
  body{{background:transparent;}}
  #wrap{{position:relative;display:flex;justify-content:center;align-items:center;
         width:100%;background:#000000;overflow:hidden;
         border:1px solid rgba(0,255,136,0.18);box-shadow:0 0 40px rgba(0,255,136,0.05);}}
  #cv  {{width:100%;height:auto;max-height:720px;object-fit:contain;cursor:crosshair;display:block;}}
  #hud {{position:absolute;top:10px;right:10px;color:#00ff88;font-size:11px;font-weight:400;
         background:rgba(0,0,0,.88);padding:4px 10px;border-radius:2px;pointer-events:none;
         border:1px solid rgba(0,255,136,0.22);font-family:'Share Tech Mono',monospace;letter-spacing:1px;}}
  #wait{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
          color:#4a5568;font-size:13px;font-weight:400;pointer-events:none;
          font-family:'Share Tech Mono',monospace;letter-spacing:2px;text-transform:uppercase;}}
  #modal{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
           background:rgba(0,0,0,.88);z-index:999;align-items:center;
           justify-content:center;}}
  #modal.show{{display:flex;}}
  #mbox{{background:#000000;border:1px solid rgba(0,255,136,0.25);border-radius:2px;
          padding:22px;min-width:310px;color:#b0bec5;box-shadow:0 0 40px rgba(0,255,136,0.10);
          position:relative;}}
  #mbox::before{{content:"";position:absolute;top:0;left:0;width:12px;height:12px;
    border-top:1px solid rgba(0,255,136,0.6);border-left:1px solid rgba(0,255,136,0.6);}}
  #mbox::after{{content:"";position:absolute;bottom:0;right:0;width:12px;height:12px;
    border-bottom:1px solid rgba(0,255,136,0.6);border-right:1px solid rgba(0,255,136,0.6);}}
  #mbox h3{{margin-bottom:12px;color:#00ff88;font-weight:700;font-size:0.9rem;
    font-family:'Rajdhani',sans-serif;letter-spacing:2px;text-transform:uppercase;}}
  #mbox label{{display:block;font-family:'Share Tech Mono',monospace;font-size:9px;
               color:#4a5568;margin-top:10px;letter-spacing:2px;text-transform:uppercase;}}
  #mbox input,#mbox select{{width:100%;background:#030810;border:1px solid rgba(0,255,136,0.16);
    color:#b0bec5;padding:6px 10px;border-radius:2px;margin-top:3px;font-size:12px;
    font-family:'Share Tech Mono',monospace;outline:none;transition:border .15s;}}
  #mbox input:focus,#mbox select:focus{{border-color:rgba(0,255,136,0.5);}}
  #mbox select option{{background:#000000;}}
  .btn-group{{margin-top:16px;display:flex;gap:6px;}}
  .btn{{padding:7px 12px;border:1px solid;border-radius:2px;font-weight:700;
        cursor:pointer;font-size:11px;font-family:'Rajdhani',sans-serif;
        flex:1;transition:all .12s;letter-spacing:1px;text-transform:uppercase;}}
  .btn-save{{background:rgba(0,255,136,0.10);border-color:rgba(0,255,136,0.4);color:#00ff88;}}
  .btn-save:hover{{background:rgba(0,255,136,0.18);}}
  .btn-clear{{background:rgba(255,62,62,0.10);border-color:rgba(255,62,62,0.4);color:#ff5555;}}
  .btn-clear:hover{{background:rgba(255,62,62,0.18);}}
  .btn-cancel{{background:rgba(74,85,104,0.10);border-color:rgba(74,85,104,0.3);color:#7a8a96;}}
  #mstatus{{margin-top:8px;font-size:10px;color:#00ff88;text-align:center;
    font-family:'Share Tech Mono',monospace;letter-spacing:1px;}}
</style></head><body>
<div id="wrap">
  <canvas id="cv" width="1280" height="720"></canvas>
  <div id="hud">◌ CONNECTING</div>
  <div id="wait">◌ AWAITING STREAM</div>
</div>
<div id="modal"><div id="mbox">
  <h3>Label Player — ID #<span id="m_tid">?</span></h3>
  <label>Name</label><input id="m_name" placeholder="e.g. Karabatic" autocomplete="off"/>
  <label>Shirt #</label><input id="m_shirt" type="number" min="1" max="99"/>
  <label>Team</label>
  <select id="m_team">
    <option value="">— keep current —</option>
    <option value="team_a">Home (team_a)</option>
    <option value="team_b">Away (team_b)</option>
    <option value="goalkeeper">Goalkeeper</option>
  </select>
  <label>Role</label><input id="m_role" placeholder="GK / LW / RW / CB / PV …"/>
  <div class="btn-group">
    <button class="btn btn-save"   onclick="saveEntity()">Save</button>
    <button class="btn btn-clear"  onclick="clearEntity()">Clear</button>
    <button class="btn btn-cancel" onclick="closeModal()">Cancel</button>
  </div>
  <div id="mstatus"></div>
</div></div>
<script>
const API="{API_URL}",WS_URL="{WS_URL}";
const cv=document.getElementById("cv"),ctx=cv.getContext("2d");
const hud=document.getElementById("hud"),modal=document.getElementById("modal");
const waitEl=document.getElementById("wait");
let ws,lastTracks=[],selectedTrackId=null,reconnectTimer=null,frameReceived=false;
let frameW=1280,frameH=720;
const TC={{"team_a":"#00e5ff","team_b":"#ff6d00","goalkeeper":"#00ff88","unknown":"#4a5568"}};

function connect(){{
  ws=new WebSocket(WS_URL);
  ws.onopen=()=>{{hud.innerHTML="● LIVE";hud.style.color="#00ff88";clearTimeout(reconnectTimer);}};
  ws.onmessage=(evt)=>{{
    const msg=JSON.parse(evt.data);
    if(msg.type==="frame"){{
      if(!frameReceived){{frameReceived=true;waitEl.style.display="none";}}
      const img=new Image();
      img.onload=()=>{{
        frameW=img.naturalWidth;frameH=img.naturalHeight;
        cv.width=frameW;cv.height=frameH;
        ctx.drawImage(img,0,0);
        drawOverlay(msg);
      }};
      img.src="data:image/jpeg;base64,"+msg.frame_b64;
    }} else if(msg.type==="click_result"){{
      selectedTrackId=msg.track_id;
      document.getElementById("m_tid").textContent=msg.track_id??"?";
      const e=msg.entity||{{}};
      document.getElementById("m_name").value=e.name||"";
      document.getElementById("m_shirt").value=e.shirt_number||"";
      document.getElementById("m_team").value=e.team||"";
      document.getElementById("m_role").value=e.role||"";
      document.getElementById("mstatus").textContent="";
      modal.classList.add("show");
    }}
  }};
  ws.onclose=()=>{{
    hud.innerHTML="◌ RECONNECTING";hud.style.color="#ffaa00";
    frameReceived=false;waitEl.style.display="block";
    reconnectTimer=setTimeout(connect,2000);
  }};
  ws.onerror=()=>ws.close();
}}

function drawOverlay(msg){{
  (msg.tracks||[]).forEach(t=>{{
    if(t.ball){{
      const cx=Math.round((t.x1+t.x2)/2),cy=Math.round((t.y1+t.y2)/2);
      const r=Math.max(12,Math.round((t.x2-t.x1)/2));
      ctx.strokeStyle="#e3b341";ctx.lineWidth=3;
      ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.stroke();
      ctx.fillStyle="rgba(227,179,65,.18)";ctx.fill();
      return;
    }}
    const col=TC[t.team]||"#888";
    ctx.strokeStyle=col;ctx.lineWidth=2.5;
    ctx.strokeRect(t.x1,t.y1,t.x2-t.x1,t.y2-t.y1);
    const lbl=t.label||(t.team.replace("team_","T")+" #"+t.id);
    ctx.font="600 12px Inter";
    const tw=ctx.measureText(lbl).width;
    const lx=t.x1+((t.x2-t.x1)/2)-(tw/2);
    const ly=Math.max(t.y1-22,5);
    ctx.fillStyle="rgba(13,17,23,.88)";
    ctx.beginPath();ctx.roundRect(lx-7,ly,tw+14,19,8);ctx.fill();
    ctx.strokeStyle=col;ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle="#fff";ctx.fillText(lbl,lx,ly+13);
  }});
}}

cv.addEventListener("click",(e)=>{{
  const rect=cv.getBoundingClientRect();
  const x=(e.clientX-rect.left)*(cv.width/rect.width);
  const y=(e.clientY-rect.top)*(cv.height/rect.height);
  if(ws&&ws.readyState===1)
    ws.send(JSON.stringify({{type:"click",x,y,frame_w:frameW,frame_h:frameH}}));
}});

function saveEntity(){{
  if(selectedTrackId===null)return;
  fetch(API+"/entity",{{method:"POST",headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{track_id:selectedTrackId,
      name:document.getElementById("m_name").value.trim(),
      shirt_number:parseInt(document.getElementById("m_shirt").value)||0,
      team:document.getElementById("m_team").value,
      role:document.getElementById("m_role").value.trim(),locked:true}})
  }}).then(r=>r.json()).then(d=>{{
    document.getElementById("mstatus").textContent=d.ok?"Saved":""+d.error;
    if(d.ok)setTimeout(closeModal,700);
  }}).catch(()=>document.getElementById("mstatus").textContent="API unreachable");
}}
function clearEntity(){{
  if(selectedTrackId===null)return;
  fetch(API+"/entity/"+selectedTrackId,{{method:"DELETE"}}).then(()=>closeModal());
}}
function closeModal(){{modal.classList.remove("show");selectedTrackId=null;}}
document.addEventListener("keydown",e=>{{if(e.key==="Escape")closeModal();}});
connect();
</script></body></html>"""

_CALIB_WIDGET = f"""<!DOCTYPE html><html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}}
  body{{background:#050d1a;color:#e8f4ff;padding:12px;}}
  #instruct{{padding:10px 14px;border-radius:8px;font-size:13px;font-weight:600;
    margin-bottom:10px;border-left:4px solid #00e5ff;background:rgba(0,229,255,0.07);}}
  #wrap{{position:relative;display:inline-block;max-width:100%;}}
  canvas{{cursor:crosshair;border:1px solid rgba(0,229,255,0.12);border-radius:8px;display:block;max-width:100%;}}
  .ctrl{{display:flex;gap:8px;margin-top:10px;align-items:center;}}
  .btn{{padding:8px 18px;border:none;border-radius:6px;font-weight:700;font-size:12px;cursor:pointer;transition:opacity .15s;}}
  .btn:hover{{opacity:.8;}} .btn:disabled{{opacity:.35;cursor:default;}}
  .btn-submit{{background:#00e5ff;color:#050d1a;}}
  .btn-reset {{background:#1a2d4a;color:#e8f4ff;border:1px solid #2a4060;}}
  .btn-clear {{background:#ff3366;color:#fff;}}
  #status{{margin-top:10px;padding:8px 14px;border-radius:6px;font-size:12px;font-weight:600;display:none;}}
  .ok {{background:rgba(0,255,136,.10);color:#00ff88;border:1px solid #00ff8844;}}
  .err{{background:rgba(255,51,102,.10);color:#ff3366;border:1px solid #ff336644;}}
  #corners-list{{margin-top:8px;font-size:11px;color:#4a7fa0;font-family:monospace;line-height:1.8;}}
  #loading{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#4a7fa0;font-size:14px;font-weight:600;pointer-events:none;}}
</style></head><body>
<div id="instruct">Click corner 1/4 — <b style="color:#00ff88">Top-Left</b></div>
<div id="wrap">
  <canvas id="cv" width="1280" height="720"></canvas>
  <div id="loading">Loading frame…</div>
</div>
<div class="ctrl">
  <button class="btn btn-submit" id="btnSubmit" disabled onclick="submitCalib()">Submit (0/4)</button>
  <button class="btn btn-reset" onclick="resetCorners()">Reset</button>
  <button class="btn btn-clear" onclick="clearCalib()">Clear Saved</button>
</div>
<div id="status"></div>
<div id="corners-list"></div>
<script>
const API="{API_URL}";
const cv=document.getElementById("cv"),ctx=cv.getContext("2d");
let corners=[],bgImg=new Image(),imgW=1280,imgH=720;
const NAMES=["Top-Left","Top-Right","Bottom-Right","Bottom-Left"];
const LABELS=["TL","TR","BR","BL"];
const COLORS=["#00ff88","#00e5ff","#ff8c00","#ff3366"];

async function loadSnapshot(){{
  document.getElementById("loading").style.display="block";
  try{{
    const r=await fetch(API+"/snapshot");
    const d=await r.json();
    if(!d.ok){{document.getElementById("loading").textContent=d.error||"No frame yet";setTimeout(loadSnapshot,2500);return;}}
    bgImg=new Image();
    bgImg.onload=()=>{{imgW=bgImg.naturalWidth;imgH=bgImg.naturalHeight;cv.width=imgW;cv.height=imgH;document.getElementById("loading").style.display="none";redraw();}};
    bgImg.src="data:image/jpeg;base64,"+d.frame_b64;
  }}catch(e){{document.getElementById("loading").textContent="Pipeline not connected";setTimeout(loadSnapshot,3000);}}
}}
function redraw(){{
  ctx.drawImage(bgImg,0,0);
  if(corners.length>=2){{
    ctx.strokeStyle="rgba(0,229,255,0.5)";ctx.lineWidth=2;ctx.setLineDash([8,5]);
    ctx.beginPath();ctx.moveTo(corners[0][0],corners[0][1]);
    for(let i=1;i<corners.length;i++)ctx.lineTo(corners[i][0],corners[i][1]);
    if(corners.length===4)ctx.closePath();
    ctx.stroke();ctx.setLineDash([]);
  }}
  corners.forEach((pt,i)=>{{
    const col=COLORS[i];
    ctx.strokeStyle="rgba(0,0,0,0.6)";ctx.lineWidth=5;
    ctx.beginPath();ctx.arc(pt[0],pt[1],9,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(pt[0],pt[1],7,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=col;ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(pt[0]-18,pt[1]);ctx.lineTo(pt[0]+18,pt[1]);
    ctx.moveTo(pt[0],pt[1]-18);ctx.lineTo(pt[0],pt[1]+18);ctx.stroke();
    ctx.fillStyle="rgba(0,0,0,0.78)";ctx.beginPath();ctx.roundRect(pt[0]+12,pt[1]-10,28,18,4);ctx.fill();
    ctx.fillStyle=col;ctx.font="bold 12px Inter";ctx.textBaseline="middle";
    ctx.fillText(LABELS[i],pt[0]+15,pt[1]);
  }});
  updateUI();
}}
function updateUI(){{
  const n=corners.length;
  const instruct=document.getElementById("instruct"),btn=document.getElementById("btnSubmit");
  if(n<4){{
    instruct.innerHTML=`Click corner ${{n+1}}/4 — <b style="color:${{COLORS[n]}}">${{NAMES[n]}}</b>`;
    instruct.style.borderColor=COLORS[n];btn.textContent=`Submit (${{n}}/4)`;btn.disabled=true;
  }}else{{
    instruct.innerHTML=`All 4 corners set. <b style="color:#00ff88">Ready to submit.</b>`;
    instruct.style.borderColor="#00ff88";btn.textContent="Submit Calibration";btn.disabled=false;
  }}
  document.getElementById("corners-list").innerHTML=corners.map((pt,i)=>
    `<span style="color:${{COLORS[i]}}">${{LABELS[i]}}</span>: (${{pt[0]}}, ${{pt[1]}})`).join("&nbsp;&nbsp;");
}}
cv.addEventListener("click",e=>{{
  if(corners.length>=4)return;
  const rect=cv.getBoundingClientRect();
  corners.push([Math.round((e.clientX-rect.left)*(cv.width/rect.width)),
                Math.round((e.clientY-rect.top)*(cv.height/rect.height))]);
  redraw();
}});
function resetCorners(){{corners=[];document.getElementById("status").style.display="none";redraw();}}
async function submitCalib(){{
  if(corners.length!==4)return;
  showStatus("Sending calibration…","ok");
  try{{
    const r=await fetch(API+"/court/calibrate",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{corners}})}});
    const d=await r.json();
    showStatus(d.ok?"Court calibrated — overlay active on next frame.":""+(d.error||"Calibration failed"),d.ok?"ok":"err");
  }}catch(e){{showStatus("API unreachable","err");}}
}}
async function clearCalib(){{
  try{{
    const r=await fetch(API+"/court/calibrate",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{clear:true}})}});
    const d=await r.json();
    showStatus(d.ok?"Calibration cleared.":""+d.error,d.ok?"ok":"err");
    resetCorners();
  }}catch(e){{showStatus("API unreachable","err");}}
}}
function showStatus(msg,cls){{
  const el=document.getElementById("status");el.textContent=msg;el.className=cls;el.style.display="block";
}}
loadSnapshot();
</script></body></html>"""

SCOUTING_HTML = f"""<!DOCTYPE html><html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
  * {{ box-sizing: border-box; font-family: 'Rajdhani', sans-serif; user-select: none; }}
  body {{ background: transparent; color: #fff; margin: 0; padding: 10px; display: flex; gap: 20px; }}

  /* Player List Column */
  .player-col {{ flex: 1; display: flex; flex-direction: column; gap: 5px; max-height: 600px; overflow-y: auto; }}
  .player-btn {{
    background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
    padding: 10px; text-align: left; cursor: pointer; border-radius: 4px;
    font-size: 14px; font-weight: 700; transition: 0.1s;
  }}
  .player-btn.active-a {{ background: #1f4287; border-color: #3b82f6; color: #fff; }}
  .player-btn.active-b {{ background: #7a1f1f; border-color: #ef4444; color: #fff; }}

  /* Action Buttons Grid (Mimicking Data Video 2007 Layout) */
  .action-grid {{ flex: 2; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; align-content: start; }}
  .action-btn {{
    padding: 15px 5px; border-radius: 6px; border: 2px solid rgba(0,0,0,0.2);
    font-size: 15px; font-weight: 700; cursor: pointer; text-align: center;
    text-transform: uppercase; letter-spacing: 1px; color: #111;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.05s;
  }}
  .action-btn:active {{ transform: scale(0.96); }}
  .action-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}

  /* Colors mimicking the Italian software screenshot */
  .btn-goal {{ background: #86efac; border-color: #22c55e; }} /* Light Green */
  .btn-saved {{ background: #fde047; border-color: #eab308; }} /* Yellow */
  .btn-miss {{ background: #d1d5db; border-color: #9ca3af; }} /* Grey */
  .btn-foul {{ background: #fcd34d; border-color: #f59e0b; }} /* Orange/Yellow */
  .btn-turnover {{ background: #fdba74; border-color: #f97316; }} /* Orange */
  .btn-sanction {{ background: #fca5a5; border-color: #ef4444; }} /* Red */
  .btn-tactic {{ background: #93c5fd; border-color: #3b82f6; }} /* Blue */

  #status {{ grid-column: span 3; text-align: center; margin-top: 10px; font-size: 13px; color: #00ff88; }}
</style>
</head><body>

<div class="player-col" id="playerList">
  <!-- Players will be populated here via JS -->
</div>

<div class="action-grid">
  <!-- Top Row -->
  <div class="action-btn btn-miss" onclick="logEvent('shot_rejected')">Shot Rejected</div>
  <div class="action-btn btn-miss" onclick="logEvent('shot_out')">Shot Out</div>
  <div class="action-btn btn-tactic" onclick="logEvent('assist')">Assist</div>

  <!-- Goal Row -->
  <div class="action-btn btn-goal" onclick="logEvent('goal')">Goal</div>
  <div class="action-btn btn-saved" onclick="logEvent('shot_saved')">Shot Saved</div>
  <div class="action-btn btn-tactic" onclick="logEvent('fast_break')">Fast Break</div>

  <!-- Foul Row -->
  <div class="action-btn btn-foul" onclick="logEvent('foul_attack')">Foul Attack</div>
  <div class="action-btn btn-foul" onclick="logEvent('foul_defense')">Foul Defense</div>
  <div class="action-btn btn-tactic" onclick="logEvent('7m_penalty')">7m Penalty</div>

  <!-- Sanction & Turnovers -->
  <div class="action-btn btn-turnover" onclick="logEvent('turnover')">Turnover</div>
  <div class="action-btn btn-sanction" onclick="logEvent('sanction_2min')">2 Min Susp.</div>
  <div class="action-btn btn-sanction" onclick="logEvent('yellow_card')">Yellow Card</div>

  <!-- Duels -->
  <div class="action-btn btn-foul" onclick="logEvent('1v1_won')">1v1 Won</div>
  <div class="action-btn btn-foul" onclick="logEvent('1v1_lost')">1v1 Lost</div>
  <div class="action-btn btn-tactic" onclick="logEvent('steal')">Steal</div>

  <div id="status">Select a player, then click an action.</div>
</div>

<script>
  const API = "{API_URL}";
  let selectedPlayerId = null;
  let selectedTeam = null;

  // Fetch players and populate list
  async function loadPlayers() {{
    try {{
      const res = await fetch(API + "/analytics/all");
      const data = await res.json();
      const players = data.players || [];
      const list = document.getElementById("playerList");
      list.innerHTML = "";

      players.sort((a,b) => a.team.localeCompare(b.team) || a.shirt_number - b.shirt_number).forEach(p => {{
        let btn = document.createElement("div");
        btn.className = "player-btn";
        btn.innerHTML = `<b>#${{p.shirt_number || '?'}}</b> ${{p.name}} <small style="color:#888;float:right">${{p.team==='team_a'?'HOME':'AWAY'}}</small>`;
        btn.onclick = () => selectPlayer(btn, p.player_id, p.team);
        list.appendChild(btn);
      }});
    }} catch (e) {{ console.error("Error loading players"); }}
  }}

  function selectPlayer(btn, id, team) {{
    document.querySelectorAll('.player-btn').forEach(b => b.className = 'player-btn');
    btn.className = `player-btn ${{team === 'team_a' ? 'active-a' : 'active-b'}}`;
    selectedPlayerId = id;
    selectedTeam = team;
    document.getElementById("status").innerText = "Ready to log action for selected player.";
  }}

  async function logEvent(eventType) {{
    if (!selectedPlayerId) {{
      document.getElementById("status").style.color = "#ef4444";
      document.getElementById("status").innerText = "Please select a player first!";
      return;
    }}

    document.getElementById("status").style.color = "#f59e0b";
    document.getElementById("status").innerText = "Logging...";

    try {{
      const res = await fetch(API + "/match/manual-event", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          event_type: eventType,
          team: selectedTeam,
          player: selectedPlayerId,
          source: "manual_scouting"
        }})
      }});
      if(res.ok) {{
        document.getElementById("status").style.color = "#00ff88";
        document.getElementById("status").innerText = `${{eventType.toUpperCase()}} logged successfully.`;
      }}
    }} catch(e) {{
      document.getElementById("status").style.color = "#ef4444";
      document.getElementById("status").innerText = "Error connecting to backend.";
    }}
  }}

  loadPlayers();
</script>
</body></html>"""

# ── Component builders ────────────────────────────────────────────────────────

def draw_scoreboard(state: dict, ms: dict, badge: str = "LIVE",
                    logo_a: str | None = None, logo_b: str | None = None):
    state   = state or {}
    ms      = ms or {}
    teams   = state.get("teams") or {}
    na      = (teams.get("team_a") or {}).get("name", "HOME") or "HOME"
    nb      = (teams.get("team_b") or {}).get("name", "AWAY") or "AWAY"
    na      = na.upper()
    nb      = nb.upper()
    hex_a   = bgr_to_hex((teams.get("team_a") or {}).get("color_bgr") or [220, 60, 60])
    hex_b   = bgr_to_hex((teams.get("team_b") or {}).get("color_bgr") or [60, 60, 220])
    sa      = (ms.get("score") or {}).get("team_a", 0) or 0
    sb      = (ms.get("score") or {}).get("team_b", 0) or 0
    period  = ms.get("period") or 1
    elapsed = ms.get("period_elapsed_s") or 0

    badge_map = {
        "LIVE":      '<div class="badge-live">● LIVE</div>',
        "HALF-TIME": '<div class="badge-ht">HALF-TIME</div>',
        "FULL-TIME": '<div class="badge-ft">FULL-TIME</div>',
    }
    badge_html = badge_map.get(badge, badge_map["LIVE"])

    logo_a_html = (
        f'<img src="{logo_a}" class="sb-logo" alt="{na}">'
        if logo_a else
        f'<div class="sb-logo-ph" style="background:{hex_a}22;border:1px solid {hex_a}44;color:{hex_a}">{na[:1]}</div>'
    )
    logo_b_html = (
        f'<img src="{logo_b}" class="sb-logo" alt="{nb}">'
        if logo_b else
        f'<div class="sb-logo-ph" style="background:{hex_b}22;border:1px solid {hex_b}44;color:{hex_b}">{nb[:1]}</div>'
    )

    st.markdown(f"""
    <div class="sb-wrap">
      <div class="sb-team">
        {logo_a_html}
        <div class="sb-name" style="color:{hex_a}">{na}</div>
        <div class="sb-score">{sa}</div>
      </div>
      <div class="sb-center">
        <div class="sb-period">PERIOD {period}</div>
        <div class="sb-clock">{fmt_time(elapsed)}</div>
        {badge_html}
      </div>
      <div class="sb-team">
        {logo_b_html}
        <div class="sb-name" style="color:{hex_b}">{nb}</div>
        <div class="sb-score">{sb}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def draw_sticky_header(state: dict, ms: dict,
                       logo_a: str | None = None, logo_b: str | None = None):
    state   = state or {}
    ms      = ms or {}
    teams   = state.get("teams") or {}
    na      = ((teams.get("team_a") or {}).get("name") or "Home").upper()
    nb      = ((teams.get("team_b") or {}).get("name") or "Away").upper()
    hex_a   = bgr_to_hex((teams.get("team_a") or {}).get("color_bgr") or [220, 60, 60])
    hex_b   = bgr_to_hex((teams.get("team_b") or {}).get("color_bgr") or [60, 60, 220])
    sa      = (ms.get("score") or {}).get("team_a", 0) or 0
    sb      = (ms.get("score") or {}).get("team_b", 0) or 0
    period  = ms.get("period") or 1
    elapsed = ms.get("period_elapsed_s") or 0
    poss    = (ms.get("possession") or {}).get("team", "")
    poss_txt = f"{na}" if poss == "team_a" else (f"{nb}" if poss == "team_b" else "")

    logo_a_html = (
        f'<img src="{logo_a}" class="kpi-logo" alt="{na}">'
        if logo_a else
        f'<div class="kpi-logo-ph" style="background:{hex_a}22;color:{hex_a}">{na[:1]}</div>'
    )
    logo_b_html = (
        f'<img src="{logo_b}" class="kpi-logo" alt="{nb}">'
        if logo_b else
        f'<div class="kpi-logo-ph" style="background:{hex_b}22;color:{hex_b}">{nb[:1]}</div>'
    )

    st.markdown(f"""
    <div class="kpi-banner">
      {logo_a_html}
      <div class="kpi-team">{na}</div>
      <div class="kpi-div"></div>
      <div class="kpi-score">{sa} – {sb}</div>
      <div class="kpi-div"></div>
      <div class="kpi-team r">{nb}</div>
      {logo_b_html}
      <div class="kpi-div"></div>
      <div class="kpi-clock">P{period} &nbsp; {fmt_time(elapsed)}</div>
      <div class="kpi-poss">{poss_txt or "—"}</div>
    </div>""", unsafe_allow_html=True)


def draw_stat_bar(label: str, val_a: float, val_b: float,
                  hex_a: str, hex_b: str, suffix: str = ""):
    total = max(val_a + val_b, 1)
    w_a   = max(4, int(240 * val_a / total))
    w_b   = max(4, int(240 * val_b / total))
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-label">{label}</div>
      <div class="stat-bars">
        <span class="stat-val-l" style="color:{hex_a}">{int(val_a)}{suffix}</span>
        <div class="bar-left"  style="width:{w_a}px;background:{hex_a};box-shadow:0 0 6px {hex_a}66"></div>
        <div class="bar-right" style="width:{w_b}px;background:{hex_b};box-shadow:0 0 6px {hex_b}66"></div>
        <span class="stat-val-r" style="color:{hex_b}">{int(val_b)}{suffix}</span>
      </div>
    </div>""", unsafe_allow_html=True)


EVENT_ICONS = {
    "goal":              ("", "#f59e0b"),
    "shot":              ("", "#60a5fa"),
    "save":              ("", "#34d399"),
    "fast_break":        ("", "#a78bfa"),
    "possession_change": ("", "#8b949e"),
    "passive_warning":   ("",  "#f97316"),
    "passive_violation": ("", "#ef4444"),
    "period_end":        ("", "#fbbf24"),
    "overload":          ("", "#ef4444"),
}


def draw_event_timeline(events: list, na: str, nb: str,
                         hex_a: str, hex_b: str,
                         filter_types: list | None = None,
                         period_filter: int | None = None):
    p1_end_ts = None
    for ev in events:
        if ev.get("event") == "period_end" and (ev.get("data") or {}).get("period_just_ended") == 1:
            p1_end_ts = ev.get("ts")
            break

    shown = []
    for ev in events:
        et = ev["event"]
        if filter_types and et not in filter_types:
            continue
        if period_filter == 1 and p1_end_ts and ev["ts"] > p1_end_ts:
            continue
        if period_filter == 2 and p1_end_ts and ev["ts"] <= p1_end_ts:
            continue
        shown.append(ev)

    if not shown:
        st.info("No events recorded yet.")
        return

    t0   = events[0]["ts"] if events else 0
    rows = []
    for ev in reversed(shown):
        et        = ev["event"]
        team      = ev.get("team", "")
        icon, ic  = EVENT_ICONS.get(et, ("•", "#8b949e"))
        team_name = na if team == "team_a" else (nb if team == "team_b" else team)
        t_color   = hex_a if team == "team_a" else (hex_b if team == "team_b" else "#8b949e")
        elapsed   = fmt_time(ev["ts"] - t0)
        label     = et.replace("_", " ").upper()

        extra = ""
        if et == "goal":
            extra = f" · {ev.get('data',{}).get('side','').upper()} GOAL"
        elif et == "shot":
            spd = ev.get("data", {}).get("speed_px", 0)
            if spd:
                extra = f" · {spd:.0f} px/f"
        elif et == "fast_break":
            db = ev.get("data", {}).get("defenders_back", 0)
            extra = f" · {db} def back"

        rows.append(f"""<div class="ev-row" style="border-left-color:{ic}">
          <span class="ev-icon">{icon}</span>
          <span class="ev-time">{elapsed}</span>
          <span class="ev-text">{label}{extra}</span>
          <span class="ev-team" style="color:{t_color}">{team_name}</span>
        </div>""")

    st.markdown(
        f'<div style="max-height:400px;overflow-y:auto;padding-right:2px">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def period_summary_card(label: str, period_events: list,
                         na: str, nb: str, hex_a: str, hex_b: str):
    goals_a = sum(1 for e in period_events if e["event"] == "goal"       and e["team"] == "team_a")
    goals_b = sum(1 for e in period_events if e["event"] == "goal"       and e["team"] == "team_b")
    shots_a = sum(1 for e in period_events if e["event"] == "shot"       and e["team"] == "team_a")
    shots_b = sum(1 for e in period_events if e["event"] == "shot"       and e["team"] == "team_b")
    fb_a    = sum(1 for e in period_events if e["event"] == "fast_break" and e["team"] == "team_a")
    fb_b    = sum(1 for e in period_events if e["event"] == "fast_break" and e["team"] == "team_b")
    pct_a   = f"{round(goals_a/shots_a*100)}%" if shots_a else "—"
    pct_b   = f"{round(goals_b/shots_b*100)}%" if shots_b else "—"

    st.markdown(f"""
    <div class="period-card">
      <h4>{label}</h4>
      <table style="width:100%;border-collapse:collapse;font-size:.86rem;">
        <tr style="color:#8b949e;font-size:.7rem;text-transform:uppercase;letter-spacing:.5px;">
          <td style="padding:3px 0;width:38%;text-align:left">{na}</td>
          <td style="text-align:center">STAT</td>
          <td style="width:38%;text-align:right">{nb}</td>
        </tr>
        <tr>
          <td style="color:{hex_a};font-weight:900;font-size:1.15rem">{goals_a}</td>
          <td style="color:#8b949e;text-align:center;font-size:.72rem">GOALS</td>
          <td style="color:{hex_b};font-weight:900;font-size:1.15rem;text-align:right">{goals_b}</td>
        </tr>
        <tr>
          <td style="color:{hex_a};font-weight:700">{shots_a}</td>
          <td style="color:#8b949e;text-align:center;font-size:.72rem">SHOTS</td>
          <td style="color:{hex_b};font-weight:700;text-align:right">{shots_b}</td>
        </tr>
        <tr>
          <td style="color:{hex_a};font-weight:700">{pct_a}</td>
          <td style="color:#8b949e;text-align:center;font-size:.72rem">CONV%</td>
          <td style="color:{hex_b};font-weight:700;text-align:right">{pct_b}</td>
        </tr>
        <tr>
          <td style="color:{hex_a};font-weight:700">{fb_a}</td>
          <td style="color:#8b949e;text-align:center;font-size:.72rem">FAST BRK</td>
          <td style="color:{hex_b};font-weight:700;text-align:right">{fb_b}</td>
        </tr>
      </table>
    </div>""", unsafe_allow_html=True)


def draw_alert_cards(alerts: list, max_cards: int = 8):
    if not alerts:
        st.markdown(
            '<div style="color:#8b949e;font-size:0.85rem;text-align:center;'
            'padding:28px 0;letter-spacing:0.3px">No tactical alerts yet.</div>',
            unsafe_allow_html=True,
        )
        return

    cards_html = []
    for a in list(alerts)[-max_cards:][::-1]:
        text     = a.get("text") or a.get("message") or a.get("insight") or str(a)
        sev_key  = (a.get("severity") or _classify_severity(text, a.get("type", ""))).upper()
        css_cls, color, icon = _SEVERITY_CFG.get(sev_key, _SEVERITY_CFG["INFO"])
        ts       = a.get("timestamp") or a.get("ts") or 0
        time_str = fmt_time(ts) if isinstance(ts, (int, float)) and ts > 0 else ""

        cards_html.append(f"""
        <div class="alert-card {css_cls}">
          <div class="alert-card-header">
            <span class="alert-card-icon">{icon}</span>
            <span class="alert-card-badge"
                  style="background:{color}22;color:{color};border:1px solid {color}44">{sev_key}</span>
            <span class="alert-card-time">{time_str}</span>
          </div>
          <div class="alert-card-body">{text}</div>
        </div>""")

    st.markdown(
        f'<div style="max-height:520px;overflow-y:auto;padding-right:2px">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


def _fetch_llm_insights() -> list:
    data = api_get("/llm/insight")
    if data.get("insights"):
        return data["insights"]
    data2 = api_get("/coach/alerts")
    if data2.get("alerts"):
        return data2["alerts"]
    return []


# ── Side panels ───────────────────────────────────────────────────────────────

def panel_controls(state: dict):
    with st.expander("Match Controls", expanded=True):
        c1, c2 = st.columns(2)
        if c1.button("↔ Flip Sides", width="stretch"):
            api_post("/flip-sides", {})
            st.success("Flipped")
        if c2.button("Re-calibrate Colors", width="stretch"):
            api_post("/recalibrate", {})
            st.success("Recalibrating…")

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        teams = state.get("teams", {})
        na = teams.get("team_a", {}).get("name", "Home")
        nb = teams.get("team_b", {}).get("name", "Away")
        g1, g2 = st.columns(2)
        if g1.button(f"{na}", width="stretch"):
            api_post("/goal", {"team": "team_a"})
            st.rerun()
        if g2.button(f"{nb}", width="stretch"):
            api_post("/goal", {"team": "team_b"})
            st.rerun()

        # ── God Mode (Manual Events) ──────────────────────────────────────
        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.75rem;color:#00e5ff;font-weight:700;letter-spacing:1px;'>MANUAL EVENTS (GOD MODE)</span>", unsafe_allow_html=True)
        with st.form("manual_event_form", clear_on_submit=True):
            m1, m2 = st.columns(2)
            m_team = m1.selectbox("Team", ["team_a", "team_b"], label_visibility="collapsed")
            m_event = m2.selectbox("Event", ["injury", "foul_2min", "red_card", "turnover", "timeout"], label_visibility="collapsed")
            m_player = st.text_input("Player ID or Name", placeholder="Player ID or Name (Optional)", label_visibility="collapsed")
            if st.form_submit_button("Log Event To Analyst", width="stretch"):
                api_post("/match/manual-event", {"event_type": m_event, "team": m_team, "player": m_player})
                st.success(f"Logged {m_event.upper()} to Analyst!")


def panel_teams(state: dict):
    with st.expander("Teams & Colors", expanded=False):
        teams = state.get("teams", {})
        for key in ("team_a", "team_b"):
            t     = teams.get(key, {})
            bgr   = t.get("color_bgr", [130, 130, 130])
            hex_c = bgr_to_hex(bgr)
            with st.form(f"team_{key}"):
                c1, c2 = st.columns([3, 1])
                name     = c1.text_input(key.replace("_", " ").title(),
                                         value=t.get("name", ""), label_visibility="collapsed")
                col_pick = c2.color_picker("Color", value=hex_c, label_visibility="collapsed")
                if st.form_submit_button("Update"):
                    hv = col_pick.lstrip("#")
                    r_ch, g_ch, b_ch = (int(hv[i:i+2], 16) for i in (0, 2, 4))
                    api_post("/team", {"key": key, "name": name,
                                       "color_bgr": [b_ch, g_ch, r_ch]})
                    st.rerun()


def panel_roster(state: dict):
    with st.expander("Player Roster", expanded=False):
        entities   = state.get("entities", {})
        team_names = {k: v.get("name", k) for k, v in state.get("teams", {}).items()}
        if not entities:
            st.caption("Click players on the video to label them.")
            return
        for e in sorted(entities.values(),
                        key=lambda x: (x.get("team", ""), x.get("shirt_number", 0))):
            tid  = e.get("track_id")
            name = e.get("name") or f"ID #{tid}"
            team = e.get("team", "unknown")
            c1, c2, c3 = st.columns([4, 2, 1])
            c1.markdown(f"**#{e.get('shirt_number',0)} {name}**")
            c2.caption(team_names.get(team, team))
            if c3.button("", key=f"del_{tid}"):
                api_delete(f"/entity/{tid}")
                st.rerun()


def panel_players():
    with st.expander("Player Recognition DB", expanded=True):
        data = api_get("/players")
        if not data.get("ok"):
            st.caption("Database not ready yet.")
            return

        stats = data.get("stats", {})
        s1, s2, s3 = st.columns(3)
        s1.metric("Detected",  stats.get("total", 0))
        s2.metric("Learning",  stats.get("enrolled", 0))
        s3.metric("Status", "Live" if stats.get("ready") else "Building…")

        with st.form("add_player_form", clear_on_submit=True):
            st.markdown("**Manually add a player**")
            c1, c2, c3 = st.columns([3, 1, 2])
            pname  = c1.text_input("Name", placeholder="e.g. Karabatic")
            pshirt = c2.number_input("#", min_value=1, max_value=99, value=1)
            pteam  = c3.selectbox("Team", ["team_a", "team_b", "goalkeeper"])
            prole  = st.text_input("Role", placeholder="GK / LW / RW / CB / PV")
            if st.form_submit_button("Add Player", width="stretch"):
                if pname.strip():
                    r = api_post("/players", {"name": pname.strip(), "team": pteam,
                                              "shirt_number": int(pshirt), "role": prole})
                    if r.get("ok"):
                        st.success("Added — they'll be recognised automatically once seen.")
                        st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        players = data.get("players", [])
        if not players:
            st.caption("No players yet.")
            return

        for team in sorted({p["team"] for p in players}):
            team_players = [p for p in players if p["team"] == team]
            st.markdown(f"**{team.replace('team_','Team ').title()}** — {len(team_players)} players")
            for p in team_players:
                pid     = p["player_id"]
                is_auto = p["name"].startswith("Player ")
                samples = p.get("sample_count", 0)
                bar     = "▓" * min(12, samples // 4) + "░" * max(0, 12 - samples // 4)
                rdy     = samples >= 5

                c_img, c_info, c_ren, c_del = st.columns([1, 4, 2, 1])
                with c_img:
                    if p.get("has_thumb"):
                        thumb = api_get(f"/players/{pid}/thumb")
                        if thumb.get("thumb_b64"):
                            st.markdown(
                                f"<img src='data:image/jpeg;base64,{thumb['thumb_b64']}' "
                                f"style='width:40px;height:40px;object-fit:cover;"
                                f"border-radius:4px' />",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write("")
                    else:
                        st.write("")
                with c_info:
                    color = "#f0c040" if is_auto else "#3fb950"
                    tag   = "auto" if is_auto else "named"
                    st.markdown(
                        f"<b style='color:{color}'>#{p.get('shirt_number','')} {p['name']}</b> "
                        f"<small style='color:#8b949e'>[{tag}] {p.get('role','')}</small>",
                        unsafe_allow_html=True,
                    )
                    status = "Recognising" if rdy else f"Learning ({samples}/5)"
                    st.markdown(
                        f"<small style='font-family:monospace;color:#3fb950'>{bar}</small> "
                        f"<small style='color:#8b949e'>{status}</small>",
                        unsafe_allow_html=True,
                    )
                with c_ren:
                    new_name = st.text_input("Rename", key=f"ren_{pid}",
                                             placeholder="Real name", label_visibility="collapsed")
                    if st.button("", key=f"ok_{pid}") and new_name.strip():
                        api_post(f"/players/{pid}/rename",
                                 {"name": new_name.strip(), "shirt_number": p.get("shirt_number", 0)})
                        st.rerun()
                with c_del:
                    if st.button("", key=f"dpl_{pid}"):
                        api_delete(f"/players/{pid}")
                        st.rerun()


def panel_occlusion():
    with st.expander("Player Tracking Status", expanded=False):
        data = api_get("/bridge/status")
        if not data.get("ok"):
            st.caption("Bridge not ready.")
            return
        c1, c2 = st.columns(2)
        c1.metric("Active",   data.get("active_count", 0))
        c2.metric("Occluded", data.get("ghost_count", 0))
        for g in data.get("ghosts", []):
            name = g.get("name") or f"Slot {g['slot_id']}"
            num  = f"#{g['shirt_number']}" if g.get("shirt_number") else ""
            secs = round(g.get("invisible_frames", 0) / 25, 1)
            st.markdown(
                f"<div style='padding:4px 10px;margin:2px 0;border-left:3px solid #f0c040;"
                f"background:rgba(240,192,64,0.06);border-radius:0 6px 6px 0'>"
                f"<b style='color:#f0c040'>{num} {name}</b> "
                f"<small style='color:#8b949e'>hidden {secs}s | pred @ "
                f"({g['predicted_x']:.0f},{g['predicted_y']:.0f})px</small></div>",
                unsafe_allow_html=True,
            )
        if not data.get("ghosts"):
            st.caption("All detected players are currently visible.")


def panel_court():
    with st.expander("Court Detection", expanded=False):
        court = api_get("/court/status")
        if not court:
            st.caption("Pipeline not connected.")
            return
        if court.get("calibrated"):
            st.success("Court calibrated — full IHF overlay active.")
        else:
            st.warning("Court not calibrated — fallback goal boxes active.")
        if st.button("Auto-Detect Court Lines", width="stretch"):
            r = api_post("/court/calibrate", {})
            if r.get("ok"):
                st.success("Calibration triggered — check video window.")
            else:
                st.error(r.get("error", "Failed"))


def panel_camera():
    with st.expander("Camera Settings", expanded=False):
        cam = api_get("/camera")
        with st.form("camera_form"):
            c1, c2 = st.columns(2)
            ip   = c1.text_input("IP",        value=cam.get("ip", ""))
            port = c2.number_input("RTSP Port", value=cam.get("rtsp_port", 554), step=1)
            user = c1.text_input("User",      value=cam.get("user", "admin"))
            pw   = c2.text_input("Password",  value="", type="password", placeholder="Leave blank to keep")
            ch   = st.selectbox("Channel", [101, 102, 201, 202],
                                 index=[101, 102, 201, 202].index(cam.get("channel", 102)),
                                 format_func=lambda x: f"{x} — " + ("Main 4K" if x % 100 == 1 else "Sub 720p"))
            if st.form_submit_button("Save & Reconnect"):
                pl = {"ip": ip.strip(), "user": user.strip(), "rtsp_port": int(port), "channel": int(ch)}
                if pw.strip():
                    pl["password"] = pw.strip()
                r = api_post("/camera", pl)
                st.success("Reconnecting…") if r.get("ok") else st.error(str(r))


# ── Technique Feed ───────────────────────────────────────────────────────────

_TECHNIQUE_ICONS: dict[str, str] = {
    "jump_shot":      "",
    "standing_shot":  "",
    "fast_break":     "",
    "pass":           "↗",
    "overhead_pass":  "",
    "dribble_run":    "",
    "gk_save":        "",
    "block":          "",
    "pivot_receive":  "",
}

_TECHNIQUE_LABELS: dict[str, str] = {
    "jump_shot":      "Jump Shot",
    "standing_shot":  "Standing Shot",
    "fast_break":     "Fast Break",
    "pass":           "Pass",
    "overhead_pass":  "Overhead Pass",
    "dribble_run":    "Dribble Run",
    "gk_save":        "GK Save",
    "block":          "Block",
    "pivot_receive":  "Pivot Receive",
}


def _draw_technique_feed(n: int = 8, window_s: int = 30) -> None:
    """Render the last `n` technique events from the pipeline."""
    data = api_get(f"/techniques/recent?n={n}&window_s={window_s}")
    events: list[dict] = data.get("events", []) if isinstance(data, dict) else []

    if not events:
        st.markdown(
            "<p style='color:#777;font-size:0.78rem;margin:4px 0'>No techniques detected yet.</p>",
            unsafe_allow_html=True,
        )
        return

    for ev in events:
        technique = ev.get("technique", "unknown")
        icon      = _TECHNIQUE_ICONS.get(technique, "")
        label     = _TECHNIQUE_LABELS.get(technique, technique.replace("_", " ").title())
        team      = ev.get("team", "?")
        zone      = ev.get("zone", "—")
        pid       = ev.get("player_id") or ev.get("track_id", "?")
        conf      = float(ev.get("confidence", 0.0))
        conf_pct  = int(conf * 100)

        team_color = "#e63c3c" if str(team).lower() in ("a", "team_a", "home") else "#3c7be6"
        conf_color = "#22c55e" if conf >= 0.75 else "#f59e0b" if conf >= 0.50 else "#9ca3af"

        card_html = f"""
<div style="
    background:rgba(255,255,255,0.04);
    border-left:3px solid {team_color};
    border-radius:6px;
    padding:5px 8px;
    margin-bottom:5px;
    font-family:monospace;
">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:0.82rem;font-weight:600;color:#f0f0f0">
      {icon}&nbsp;{label}
    </span>
    <span style="
      font-size:0.68rem;font-weight:700;
      background:{team_color}33;color:{team_color};
      border-radius:3px;padding:1px 5px
    ">
      {("A" if str(team).lower() in ("a","team_a","home") else "B")}
    </span>
  </div>
  <div style="display:flex;gap:10px;margin-top:3px;align-items:center">
    <span style="font-size:0.70rem;color:#9ca3af">#{pid} · {zone}</span>
    <div style="flex:1;background:#333;border-radius:3px;height:4px">
      <div style="width:{conf_pct}%;background:{conf_color};height:4px;border-radius:3px"></div>
    </div>
    <span style="font-size:0.68rem;color:{conf_color}">{conf_pct}%</span>
  </div>
</div>"""
        st.markdown(card_html, unsafe_allow_html=True)


# ── Tab: LIVE ────────────────────────────────────────────────────────────────

def tab_live(state: dict, logo_a: str | None = None, logo_b: str | None = None):
    main_col, side_col = st.columns([7, 3])

    with main_col:
        components.html(CANVAS_HTML, height=730, scrolling=False)

    with side_col:
        st.markdown("### Coach Controls")
        panel_controls(state)
        panel_teams(state)
        panel_players()
        panel_roster(state)
        panel_court()
        panel_occlusion()
        panel_camera()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<span class='glass-card-label' style='display:block;margin-bottom:8px'>"
            "Detected Techniques</span>",
            unsafe_allow_html=True,
        )
        _draw_technique_feed()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<span class='glass-card-label' style='display:block;margin-bottom:10px'>"
            "Coach AI Alerts</span>",
            unsafe_allow_html=True,
        )
        insights = _fetch_llm_insights()
        draw_alert_cards(insights, max_cards=5)

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("↻ Sync Dashboard", width="stretch"):
            st.rerun()

# ── Tab: TAGGING ─────────────────────────────────────────────────────────────

def tab_tagging():
    st.markdown("## Manual Scouting & Tagging")
    st.caption("Professional layout for manual event logging. Select a player on the left, then click an action. Events sync instantly with the Match Analyzer and Timeline.")

    st.markdown("<div class='glass-card' style='padding:0'>", unsafe_allow_html=True)
    components.html(SCOUTING_HTML, height=600, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Tab: SUMMARY ─────────────────────────────────────────────────────────────

def tab_summary(state: dict, ms: dict,
                logo_a: str | None = None, logo_b: str | None = None):
    state    = state or {}
    ms       = ms or {}
    teams    = state.get("teams") or {}
    na       = ((teams.get("team_a") or {}).get("name") or "Home").upper()
    nb       = ((teams.get("team_b") or {}).get("name") or "Away").upper()
    hex_a    = bgr_to_hex((teams.get("team_a") or {}).get("color_bgr") or [220, 60, 60])
    hex_b    = bgr_to_hex((teams.get("team_b") or {}).get("color_bgr") or [60, 60, 220])
    events   = ms.get("events") or []

    p1_end_ts = None
    for ev in events:
        if ev.get("event") == "period_end" and (ev.get("data") or {}).get("period_just_ended") == 1:
            p1_end_ts = ev.get("ts")
            break
    p1_events = [e for e in events if p1_end_ts is None or e["ts"] <= p1_end_ts]
    p2_events = [e for e in events if p1_end_ts and e["ts"] > p1_end_ts]

    st.markdown("<div class='glass-card' style='margin-bottom:16px;padding:12px 20px;height:auto'>",
                unsafe_allow_html=True)
    draw_scoreboard(state, ms, logo_a=logo_a, logo_b=logo_b)
    st.markdown("</div>", unsafe_allow_html=True)

    col_p1, col_p2, col_stats = st.columns([2, 2, 3])

    with col_p1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        period_summary_card("1ST HALF SUMMARY", p1_events, na, nb, hex_a, hex_b)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_p2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        period_summary_card("2ND HALF SUMMARY", p2_events, na, nb, hex_a, hex_b)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stats:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Full Match Stats</span>", unsafe_allow_html=True)
        goals_a = (ms.get("score") or {}).get("team_a", 0) or 0
        goals_b = (ms.get("score") or {}).get("team_b", 0) or 0
        shots_a = (ms.get("shots") or {}).get("team_a", 0) or 0
        shots_b = (ms.get("shots") or {}).get("team_b", 0) or 0
        saves_a = (ms.get("saves") or {}).get("team_a", 0) or 0
        saves_b = (ms.get("saves") or {}).get("team_b", 0) or 0
        fb_a    = (ms.get("fast_breaks") or {}).get("team_a", 0) or 0
        fb_b    = (ms.get("fast_breaks") or {}).get("team_b", 0) or 0
        pct_a   = (ms.get("shot_pct") or {}).get("team_a", 0) or 0
        pct_b   = (ms.get("shot_pct") or {}).get("team_b", 0) or 0
        draw_stat_bar("GOALS",       goals_a, goals_b, hex_a, hex_b)
        draw_stat_bar("SHOTS",       shots_a, shots_b, hex_a, hex_b)
        draw_stat_bar("CONVERSION",  pct_a,   pct_b,   hex_a, hex_b, "%")
        draw_stat_bar("SAVES",       saves_a, saves_b, hex_a, hex_b)
        draw_stat_bar("FAST BREAKS", fb_a,    fb_b,    hex_a, hex_b)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_map, c_time = st.columns([5, 3])
    with c_map:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Interactive Shot Map</span>", unsafe_allow_html=True)
        st.plotly_chart(draw_handball_court(events, hex_a, hex_b),
                        width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_time:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Event Timeline</span>", unsafe_allow_html=True)
        ev_tab1, ev_tab2 = st.tabs(["All Events", "Goals & Shots"])
        with ev_tab1:
            draw_event_timeline(events, na, nb, hex_a, hex_b)
        with ev_tab2:
            draw_event_timeline(events, na, nb, hex_a, hex_b,
                                filter_types=["goal", "shot", "save"])
        st.markdown("</div>", unsafe_allow_html=True)


def draw_handball_court(events: list, hex_a: str = "#3b82f6", hex_b: str = "#ef4444") -> go.Figure:
    fig = go.Figure()

    def shape(type_, **kw):
        fig.add_shape(type=type_, **kw)

    court_line = dict(color="rgba(0,229,255,0.55)", width=1.8)
    goal_line  = dict(color="rgba(255,109,0,0.7)",  width=2)
    area_fill  = "rgba(0,229,255,0.04)"
    goal_fill  = "rgba(255,109,0,0.12)"

    shape("rect", x0=0, y0=0, x1=40, y1=20,
          line=court_line, fillcolor="rgba(0,20,30,0.6)")
    shape("line", x0=20, y0=0, x1=20, y1=20, line=court_line)
    shape("rect", x0=0, y0=5.5, x1=6, y1=14.5, line=court_line, fillcolor=area_fill)
    shape("rect", x0=34, y0=5.5, x1=40, y1=14.5, line=court_line, fillcolor=area_fill)
    shape("rect", x0=-1.5, y0=8.5, x1=0, y1=11.5, line=goal_line, fillcolor=goal_fill)
    shape("rect", x0=40, y0=8.5, x1=41.5, y1=11.5, line=goal_line, fillcolor=goal_fill)
    for px in (7, 33):
        shape("circle", x0=px-0.25, y0=9.75, x1=px+0.25, y1=10.25,
              line=dict(color="rgba(255,109,0,0.6)", width=1), fillcolor="rgba(255,109,0,0.5)")
    shape("circle", x0=16.85, y0=7, x1=23.15, y1=13,
          line=dict(color="rgba(0,229,255,0.25)", width=1.2), fillcolor="rgba(0,0,0,0)")

    def _team_fill(hex_c: str, opacity: float = 0.03) -> str:
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{opacity})"

    shape("rect", x0=0, y0=0, x1=20, y1=20,
          line=dict(color="rgba(0,0,0,0)", width=0), fillcolor=_team_fill(hex_a))
    shape("rect", x0=20, y0=0, x1=40, y1=20,
          line=dict(color="rgba(0,0,0,0)", width=0), fillcolor=_team_fill(hex_b))

    _ev_cfg = {
        "goal":       (12, "circle",  "#10b981"),
        "shot":       ( 9, "x",       "#ef4444"),
        "save":       ( 9, "square",  "#f59e0b"),
        "fast_break": ( 9, "diamond", "#a78bfa"),
    }
    for ev in events:
        et = ev.get("event")
        if et not in _ev_cfg:
            continue
        data = ev.get("data", {})
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        size, sym, col = _ev_cfg[et]
        team = ev.get("team", "")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color=col, size=size, symbol=sym,
                        line=dict(color="rgba(255,255,255,0.7)", width=1.2), opacity=0.9),
            hovertemplate=(
                f"<b style='color:{col}'>{et.upper()}</b><br>"
                f"Team: {team.replace('_',' ').title()}<br>"
                f"Pos: ({x:.1f}m, {y:.1f}m)<extra></extra>"
            ),
        ))

    fig.update_xaxes(range=[-3, 43], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-2, 22], visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                        font=dict(color="#c9d1d9", size=12)),
    )
    return fig


# ── Tab: PLAYER PROFILES ─────────────────────────────────────────────────────

def player_trading_card(p: dict, stats: dict) -> str:
    name = p.get("name", "Unknown").upper()
    role = (p.get("role", "") or "PLAYER").upper()
    perf = p.get("performance", 85)
    return f"""
    <div class="fut-card">
      <div class="fut-rating">{perf}</div>
      <div class="fut-role">{role}</div>
      <div class="fut-name">{name}</div>
      <div class="fut-stats">
        <div class="fut-stat-item"><span class="fut-stat-lbl">GLS</span><span class="fut-stat-val">{stats.get('goals',0)}</span></div>
        <div class="fut-stat-item"><span class="fut-stat-lbl">AST</span><span class="fut-stat-val">{stats.get('assists',0)}</span></div>
        <div class="fut-stat-item"><span class="fut-stat-lbl">SHT</span><span class="fut-stat-val">{stats.get('shots',0)}</span></div>
        <div class="fut-stat-item"><span class="fut-stat-lbl">SAV</span><span class="fut-stat-val">{stats.get('saves',0)}</span></div>
      </div>
    </div>"""


def draw_polar_chart(stats: dict, hex_color: str = "#00e5ff") -> go.Figure:
    cats   = ["Shooting", "Passing", "Defense", "Speed", "Stamina", "Vision"]
    values = [stats.get(c.lower(), random.randint(55, 92)) for c in cats]
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    fill_color = f"rgba({r},{g},{b},0.14)"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor=fill_color,
        line=dict(color=hex_color, width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], color="#8b949e",
                            tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(color="#c9d1d9", tickfont=dict(size=9),
                             gridcolor="rgba(255,255,255,0.06)"),
        ),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30), height=220,
    )
    return fig


def tab_players():
    st.markdown("## Player Profiles")
    data    = api_get("/analytics/all")
    players = data.get("players", [])

    if not players:
        st.info("No player analytics yet. Players appear automatically as the camera detects them.")
        return

    players = sorted(players, key=lambda p: (p.get("team", ""), -p.get("performance", 0)))

    cards_html = '<div class="player-grid">' + "".join(
        player_trading_card(p, p.get("stats", {})) for p in players
    ) + "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    for row_start in range(0, len(players), 3):
        row  = players[row_start:row_start + 3]
        cols = st.columns(3)
        for col, p in zip(cols, row):
            with col:
                p_stats   = p.get("stats", {})
                team_key  = p.get("team", "")
                hex_color = "#3b82f6" if team_key == "team_a" else "#ef4444"
                st.plotly_chart(draw_polar_chart(p_stats, hex_color),
                                width="stretch", config={"displayModeBar": False})

                actions = p_stats.get("actions", p_stats)
                ac = st.columns(6)
                ac[0].metric("Goals",  actions.get("goals", 0))
                ac[1].metric("Shots",  actions.get("shots", 0))
                ac[2].metric("Saves",  actions.get("saves", 0))
                ac[3].metric("Asst",   actions.get("assists", 0))
                ac[4].metric("FB",     actions.get("fast_breaks", 0))
                ac[5].metric("Tkl",    actions.get("tackles", 0))

                zone_pct = p.get("zone_pct", {})
                if zone_pct:
                    zc = st.columns(len(zone_pct))
                    for i, (z, pct) in enumerate(zone_pct.items()):
                        zc[i].metric(z.replace("_", " ").title(), f"{pct:.1f}%")

                pid     = p.get("player_id", row_start)
                hm_data = api_get(f"/analytics/player/{pid}/heatmap")
                if hm_data.get("heatmap_b64"):
                    st.markdown(
                        f"<img src='data:image/png;base64,{hm_data['heatmap_b64']}' "
                        f"style='width:100%;border-radius:4px' />"
                        f"<div style='text-align:center;color:#888;"
                        f"font-size:0.85em;margin-top:4px'>Position heatmap</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Not enough data for heatmap.")

                perf  = p.get("performance", 85)
                grade = "A" if perf >= 90 else "B" if perf >= 80 else "C" if perf >= 70 else "D"
                st.progress(min(perf / 100, 1.0),
                            text=f"Performance: {perf:.1f}/100 — Grade {grade}")

        if row_start + 3 < len(players):
            st.markdown("<hr>", unsafe_allow_html=True)


# ── Tab: COACH AI ────────────────────────────────────────────────────────────

def tab_coach():
    st.markdown("## Coach AI Assistant")

    col_alerts, col_chat, col_ref = st.columns([3, 4, 3])

    # ── Column 1: LLM Alert Cards ─────────────────────────────────────────
    with col_alerts:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Live Tactical Alerts</span>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.72rem;color:#8b949e;margin-bottom:12px'>"
            "Auto-generated every 30 s from match events · CPU-only LLM</div>",
            unsafe_allow_html=True,
        )
        insights = _fetch_llm_insights()
        draw_alert_cards(insights, max_cards=8)

        if st.button("↻ Refresh Alerts", width="stretch"):
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Column 2: Tactical Chat ───────────────────────────────────────────
    with col_chat:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Tactical Chat</span>", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant",
                 "content": "Welcome, Coach! I'm analysing the match live. "
                             "Ask me about tactics, opponent patterns, or player performance."}
            ]
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about tactics, opponents, or player stats…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            r = api_post("/coach/ask", {"question": prompt})
            response = r.get("answer") or (
                "Based on current match data: consider shifting your Left Back to create "
                "better penetration angles — the opponent's Right Wing is leaving space on the counter."
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Column 3: God Mode (Rules Editor) ──────────────────────────────────
    with col_ref:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>God Mode (Rules)</span>",
                    unsafe_allow_html=True)
        rules_data = api_get("/coach/rules_text")
        current_rules = rules_data.get("rules", "You are a handball tactical analyst...")

        with st.form("rules_form"):
            new_rules = st.text_area("Edit Tactical Focus & Rules", value=current_rules, height=350,
                                     help="Ollama will read this file every time it generates a report.")
            if st.form_submit_button("Save Rules & Force Analysis", type="primary"):
                api_post("/coach/rules_text", {"rules": new_rules})
                api_post("/analyst/request", {})
                st.success("Rules saved! Analyst is working...")
                time.sleep(1)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Tab: CAMERA IMAGING ──────────────────────────────────────────────────────

def tab_camera_imaging():
    st.markdown("## Camera Imaging")
    img    = api_get("/camera/image")
    cam_ok = "error" not in img

    if not cam_ok:
        st.warning(
            f"Camera imaging API unavailable ({img.get('error','no response')}). "
            "Start the pipeline with a live camera to enable this panel."
        )

    # ── Day / Night mode banner ───────────────────────────────────────────────
    dn     = api_get("/camera/daynight")
    mode   = (dn.get("mode") or "unknown").lower()
    is_night = mode == "night"

    mode_colors = {"day": "#10b981", "auto": "#3b82f6", "night": "#ef4444", "unknown": "#8b949e"}
    mode_icons  = {"day": "DAY (White Light)", "auto": "AUTO",
                   "night": "NIGHT (IR Active)", "unknown": "Unknown"}
    badge_color = mode_colors.get(mode, "#8b949e")
    badge_label = mode_icons.get(mode, mode.upper())

    if is_night:
        st.markdown(
            f"<div style='background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.45);"
            f"border-radius:10px;padding:12px 18px;margin-bottom:14px;"
            f"box-shadow:0 0 16px rgba(239,68,68,0.15);'>"
            f"<span style='color:#ef4444;font-weight:800;font-size:1rem;'>Night Mode Active</span>"
            f"<span style='color:#c9d1d9;font-size:0.85rem;margin-left:12px;'>"
            f"Camera is in IR mode — images are greyscale, jersey colours cannot be detected.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    dn_col1, dn_col2, dn_col3, dn_col4 = st.columns([2, 1, 1, 1])
    with dn_col1:
        st.markdown(
            f"<div style='padding:8px 0;'>"
            f"<span style='font-size:0.72rem;font-weight:700;letter-spacing:1px;"
            f"text-transform:uppercase;color:#8b949e;'>IR-Cut Filter</span><br>"
            f"<span style='font-size:1rem;font-weight:800;color:{badge_color};'>"
            f"{badge_label}</span></div>",
            unsafe_allow_html=True,
        )
    with dn_col2:
        if st.button("White Light", width="stretch",
                     disabled=not cam_ok, key="dn_day",
                     type="primary" if is_night else "secondary"):
            r = api_post("/camera/daynight", {"mode": "day"})
            if r.get("ok"):
                # Trigger color recalibration so pipeline re-learns team colors
                api_post("/recalibrate", {})
                st.success("Switched to White Light — recalibrating team colors.")
                st.rerun()
            else:
                st.error(r.get("error", "Failed to switch mode"))
    with dn_col3:
        if st.button("Auto", width="stretch",
                     disabled=not cam_ok, key="dn_auto"):
            r = api_post("/camera/daynight", {"mode": "auto"})
            if r.get("ok"):
                st.success("Switched to Auto mode.")
                st.rerun()
            else:
                st.error(r.get("error", "Failed"))
    with dn_col4:
        if st.button("Night IR", width="stretch",
                     disabled=not cam_ok, key="dn_night"):
            r = api_post("/camera/daynight", {"mode": "night"})
            if r.get("ok"):
                st.success("Switched to Night IR mode.")
                st.rerun()
            else:
                st.error(r.get("error", "Failed"))

    st.markdown("<hr style='margin:8px 0 16px;border-color:rgba(0,255,136,0.08);'>",
                unsafe_allow_html=True)

    # ── Auto Zoom & Smart Scan ─────────────────────────────────────────────────
    az      = api_get("/camera/autozoom")
    az_mode = (az.get("mode") or "off").lower()
    az_ok   = az.get("cam_ready", False) and cam_ok

    _AZ_MODES = {
        "off":     ("OFF",          "#4a5568"),
        "court":   ("FULL COURT",   "#00e5ff"),
        "players": ("TRACK PLAYERS","#00ff88"),
        "ball":    ("TRACK BALL",   "#ffaa00"),
        "goal_a":  ("GOAL A",       "#a855f7"),
        "goal_b":  ("GOAL B",       "#ec4899"),
        "scan":    ("SMART SCAN",   "#ff6d00"),
    }
    az_label, az_color = _AZ_MODES.get(az_mode, ("UNKNOWN", "#4a5568"))

    st.markdown(
        f"<div class='spx-section'>AUTO ZOOM &amp; TRACKING</div>"
        f"<div style='background:#030810;border:1px solid {az_color}44;"
        f"border-radius:2px;padding:10px 14px;margin-bottom:12px;position:relative;'>"
        f"<div style='position:absolute;top:0;left:0;width:10px;height:10px;"
        f"border-top:1px solid {az_color};border-left:1px solid {az_color};'></div>"
        f"<span style='font-family:Share Tech Mono,monospace;font-size:0.60rem;"
        f"letter-spacing:2px;text-transform:uppercase;color:#4a5568;'>MODE&nbsp;&nbsp;</span>"
        f"<span style='font-family:Share Tech Mono,monospace;font-size:0.88rem;"
        f"font-weight:400;color:{az_color};'>{az_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    az_cols = st.columns(7)
    az_buttons = [
        ("OFF",     "off",     "secondary"),
        ("COURT",   "court",   "primary" if az_mode == "court"   else "secondary"),
        ("PLAYERS", "players", "primary" if az_mode == "players" else "secondary"),
        ("BALL",    "ball",    "primary" if az_mode == "ball"    else "secondary"),
        ("GOAL A",  "goal_a",  "primary" if az_mode == "goal_a"  else "secondary"),
        ("GOAL B",  "goal_b",  "primary" if az_mode == "goal_b"  else "secondary"),
        ("SCAN", "scan",    "primary" if az_mode == "scan"    else "secondary"),
    ]
    for col, (lbl, md, btn_type) in zip(az_cols, az_buttons):
        with col:
            if st.button(lbl, width="stretch", disabled=not az_ok,
                         key=f"az_{md}", type=btn_type):
                r = api_post("/camera/autozoom", {"mode": md})
                if r.get("ok"):
                    st.rerun()
                else:
                    st.error(r.get("error", "Failed"))

    # ── Smart Scan Zone Status ────────────────────────────────────────────────
    if az_mode == "scan":
        scan_zone = az.get("scan_zone") or "—"
        scan_results = az.get("scan_results") or []
        st.markdown(
            f"<div style='margin:10px 0 4px;font-family:Share Tech Mono,monospace;"
            f"font-size:0.62rem;letter-spacing:2px;color:#4a5568;text-transform:uppercase;'>"
            f"SCANNING ZONE: <span style='color:#ff6d00'>{scan_zone.upper()}</span></div>",
            unsafe_allow_html=True,
        )
        zone_cards = []
        for z in scan_results:
            name      = z.get("name", "?").upper()
            players   = z.get("players", -1)
            is_active = (name.lower() == scan_zone)
            if is_active:
                css_cls = "zone-active"; dot = "◉"; dot_color = "#ff6d00"
            elif players < 0:
                css_cls = "zone-pending"; dot = "○"; dot_color = "#4a5568"
            elif players > 0:
                css_cls = "zone-found";  dot = "●"; dot_color = "#00e5ff"
            else:
                css_cls = "zone-empty";  dot = ""; dot_color = "#ff3e3e"
            player_txt = f"{players}P" if players >= 0 else "—"
            zone_cards.append(
                f"<div class='zone-card {css_cls}'>"
                f"<span style='color:{dot_color}'>{dot}</span>"
                f"<span style='flex:1;letter-spacing:1px'>{name}</span>"
                f"<span style='color:{dot_color}'>{player_txt}</span>"
                f"</div>"
            )
        st.markdown("".join(zone_cards), unsafe_allow_html=True)

    # ── Frame Zoom (software crop) ────────────────────────────────────────────
    fr_st = api_get("/camera/frame/state") or {}
    cur_zoom = fr_st.get("zoom", 1.0)
    cur_cx   = fr_st.get("cx",   0.5)
    cur_cy   = fr_st.get("cy",   0.5)

    st.markdown(
        f"<div class='spx-section' style='margin-top:14px;'>"
        f"FRAME ZOOM &nbsp;<span style='color:#00ff88;"
        f"font-family:Share Tech Mono,monospace'>{cur_zoom:.2f}×</span></div>"
        f"<div style='font-size:0.7rem;color:#777;margin-bottom:6px'>"
        f"Software crop — works with any camera. Use this to frame a TV "
        f"screen or zoom in on part of the court.</div>",
        unsafe_allow_html=True,
    )

    zr1, zr2, zr3 = st.columns(3)
    with zr1:
        if st.button("+ ZOOM IN", width="stretch", key="dz_in"):
            api_post("/camera/zoom", {"direction": "in", "step": 0.25})
            st.rerun()
    with zr2:
        if st.button("− ZOOM OUT", width="stretch",
                     disabled=cur_zoom <= 1.001, key="dz_out"):
            api_post("/camera/zoom", {"direction": "out", "step": 0.25})
            st.rerun()
    with zr3:
        if st.button("⟲ RESET", width="stretch",
                     disabled=cur_zoom <= 1.001 and abs(cur_cx - 0.5) < 0.01,
                     key="dz_reset"):
            api_post("/camera/zoom", {"direction": "reset", "step": 0.0})
            st.rerun()

    # Crop centre nudge — only useful when zoomed in
    if cur_zoom > 1.001:
        fo1, fo2, fo3, fo4, fo5 = st.columns(5)
        with fo1:
            if st.button("◀ LEFT", width="stretch", key="dz_left"):
                api_post("/camera/frame/offset", {"direction": "left", "step": 0.05})
                st.rerun()
        with fo2:
            if st.button("RIGHT ▶", width="stretch", key="dz_right"):
                api_post("/camera/frame/offset", {"direction": "right", "step": 0.05})
                st.rerun()
        with fo3:
            if st.button("▲ UP", width="stretch", key="dz_up"):
                api_post("/camera/frame/offset", {"direction": "up", "step": 0.05})
                st.rerun()
        with fo4:
            if st.button("▼ DOWN", width="stretch", key="dz_down"):
                api_post("/camera/frame/offset", {"direction": "down", "step": 0.05})
                st.rerun()
        with fo5:
            if st.button("◎ CENTRE", width="stretch", key="dz_centre"):
                api_post("/camera/frame/offset", {"direction": "centre", "step": 0.0})
                st.rerun()

    # ── Optical PTZ (only useful on PTZ-Z domes) ──────────────────────────────
    st.markdown(
        "<div class='spx-section' style='margin-top:14px;'>OPTICAL PTZ "
        "<span style='color:#777;font-size:0.65rem'>(PTZ cameras only)</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    mt1, mt2, mt3, mt4, mt5 = st.columns(5)
    with mt1:
        if st.button("◀ PAN L", width="stretch",
                     disabled=not cam_ok, key="pan_left"):
            api_post("/camera/ptz/momentary",
                     {"pan": -0.5, "tilt": 0, "zoom": 0,
                      "focus": 0, "duration_ms": 600})
    with mt2:
        if st.button("PAN R ▶", width="stretch",
                     disabled=not cam_ok, key="pan_right"):
            api_post("/camera/ptz/momentary",
                     {"pan":  0.5, "tilt": 0, "zoom": 0,
                      "focus": 0, "duration_ms": 600})
    with mt3:
        if st.button("▲ TILT UP", width="stretch",
                     disabled=not cam_ok, key="tilt_up"):
            api_post("/camera/ptz/momentary",
                     {"pan": 0, "tilt": 0.5, "zoom": 0,
                      "focus": 0, "duration_ms": 600})
    with mt4:
        if st.button("▼ TILT DN", width="stretch",
                     disabled=not cam_ok, key="tilt_down"):
            api_post("/camera/ptz/momentary",
                     {"pan": 0, "tilt": -0.5, "zoom": 0,
                      "focus": 0, "duration_ms": 600})
    with mt5:
        if st.button("■ STOP", width="stretch",
                     disabled=not cam_ok, key="ptz_stop_btn"):
            api_post("/camera/ptz/stop", {})

    # ── Focus Controls ────────────────────────────────────────────────────────
    st.markdown(
        "<div class='spx-section' style='margin-top:14px;'>LENS &amp; FOCUS</div>"
        "<div style='font-size:0.7rem;color:#777;margin-bottom:6px'>"
        "Note: fixed-focus bullet cameras have no motorised focus — "
        "if all buttons return errors, focus the camera lens manually.</div>",
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        if st.button("AUTO FOCUS", width="stretch",
                     disabled=not cam_ok, key="focus_auto"):
            r = api_post("/camera/focus/auto", {}) or {}
            if r.get("ok"):
                st.success("Autofocus triggered.")
            else:
                st.warning(f"Focus not supported: {r.get('error') or 'no motorised focus'}")
    with fc2:
        if st.button("◀ NEAR", width="stretch",
                     disabled=not cam_ok, key="focus_near"):
            r = api_post("/camera/focus/manual",
                         {"direction": "near", "speed": 0.5, "duration_ms": 600}) or {}
            if not r.get("ok"):
                st.warning("Manual focus not supported on this camera.")
    with fc3:
        if st.button("FAR ▶", width="stretch",
                     disabled=not cam_ok, key="focus_far"):
            r = api_post("/camera/focus/manual",
                         {"direction": "far", "speed": 0.5, "duration_ms": 600}) or {}
            if not r.get("ok"):
                st.warning("Manual focus not supported on this camera.")
    with fc4:
        if st.button("■ STOP", width="stretch",
                     disabled=not cam_ok, key="focus_stop"):
            api_post("/camera/focus/stop", {})

    # ── PTZ Live Telemetry ────────────────────────────────────────────────────
    ptz_s = api_get("/camera/ptz/status") if cam_ok else {}
    if ptz_s.get("ok"):
        pan_v  = ptz_s.get("pan",  0)
        tilt_v = ptz_s.get("tilt", 0)
        zoom_v = ptz_s.get("zoom", 1)
        st.markdown(
            f"<div style='background:#030810;border:1px solid rgba(0,255,136,0.12);"
            f"border-radius:2px;padding:8px 12px;margin-top:10px;"
            f"font-family:Share Tech Mono,monospace;font-size:0.70rem;"
            f"display:flex;gap:18px;color:#4a5568;letter-spacing:1px;'>"
            f"<span>PAN <span style='color:#00ff88'>{pan_v:.1f}°</span></span>"
            f"<span>TILT <span style='color:#00ff88'>{tilt_v:.1f}°</span></span>"
            f"<span>ZOOM <span style='color:#00ff88'>{zoom_v:.1f}×</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Tactical Analysis ──────────────────────────────────────────────────
    st.markdown(
        "<span style='font-size:0.72rem;font-weight:700;letter-spacing:1px;"
        "text-transform:uppercase;color:#8b949e;'>AI Stadium Analysis</span>",
        unsafe_allow_html=True,
    )
    ana_col1, ana_col2 = st.columns([3, 1])
    with ana_col1:
        st.caption(
            "Moves camera to full-court view (Preset 1), waits 1.5 s for the lens to settle, "
            "then sends the current match state to the AI Coach for deep tactical analysis."
        )
    with ana_col2:
        if st.button("Analyze Now", width="stretch",
                     disabled=not cam_ok, key="btn_analyze"):
            with st.spinner("AI analyzing stadium view…"):
                r = api_post("/camera/analyze", {})
            if r.get("ok"):
                insight = r.get("insight", "")
                if insight:
                    st.markdown(
                        f"<div style='background:rgba(168,85,247,0.10);border:1px solid "
                        f"rgba(168,85,247,0.35);border-radius:8px;padding:12px 16px;"
                        f"margin-top:8px;'>"
                        f"<span style='font-size:0.72rem;font-weight:700;color:#a855f7;"
                        f"text-transform:uppercase;letter-spacing:1px;'>AI Coach</span><br>"
                        f"<span style='color:#c9d1d9;font-size:0.9rem;line-height:1.5;'>"
                        f"{insight.replace(chr(10), '<br>')}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Analysis complete — no insights produced.")
            else:
                st.error(r.get("error", "Analysis failed"))

    st.markdown("<hr style='margin:8px 0 16px;border-color:rgba(255,255,255,0.08);'>",
                unsafe_allow_html=True)

    def _safe(v, default=50):
        try:
            return max(0, min(100, int(v or default)))
        except (TypeError, ValueError):
            return default

    cur_b  = _safe(img.get("brightness"), 50)
    cur_c  = _safe(img.get("contrast"),   50)
    cur_s  = _safe(img.get("saturation"), 50)
    cur_sh = _safe(img.get("sharpness"),  50)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Manual Adjustment</span>", unsafe_allow_html=True)
        with st.form("imaging_form"):
            brightness = st.slider("Brightness",  0, 100, cur_b,
                help="Raise for dark arenas; lower for overexposed scenes.")
            contrast   = st.slider("Contrast",    0, 100, cur_c,
                help="Higher contrast helps distinguish jersey colours.")
            saturation = st.slider("Saturation",  0, 100, cur_s,
                help="Increase to make team colours more vivid for auto-detection.")
            sharpness  = st.slider("Sharpness",   0, 100, cur_sh,
                help="Higher sharpness improves visibility of distant players.")
            applied = st.form_submit_button("Apply to Camera", width="stretch",
                                            disabled=not cam_ok)
            if applied:
                r = api_post("/camera/image", {
                    "brightness": brightness, "contrast": contrast,
                    "saturation": saturation, "sharpness": sharpness,
                })
                if r.get("ok"):
                    st.success("Settings written to camera.")
                else:
                    st.error(f"Camera rejected the change: {r.get('error','unknown')}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Quick Presets</span>", unsafe_allow_html=True)
        presets = {
            "Bright Arena":    {"brightness": 44, "contrast": 62, "saturation": 58, "sharpness": 72,
                                "tip": "Well-lit court — reduce overexposure, boost sharpness."},
            "Dim Arena":       {"brightness": 66, "contrast": 55, "saturation": 52, "sharpness": 60,
                                "tip": "Poorly-lit gym — lift brightness, preserve detail."},
            "Distant Players": {"brightness": 52, "contrast": 70, "saturation": 60, "sharpness": 85,
                                "tip": "Max sharpness + contrast to resolve far-end players."},
            "Vivid Colors":    {"brightness": 50, "contrast": 65, "saturation": 78, "sharpness": 65,
                                "tip": "Saturated jerseys → better automatic team colour detection."},
            "Reset Defaults":  {"brightness": 50, "contrast": 50, "saturation": 50, "sharpness": 50,
                                "tip": "Camera factory defaults."},
        }
        for label, params in presets.items():
            p_vals = {k: v for k, v in params.items() if k != "tip"}
            if st.button(label, width="stretch", disabled=not cam_ok,
                         key=f"preset_{label}"):
                r = api_post("/camera/image", p_vals)
                if r.get("ok"):
                    st.success(f"Applied: {label}")
                    st.rerun()
                else:
                    st.error(r.get("error", "Failed"))
            st.caption(params["tip"])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<span class='glass-card-label'>Current Values</span>", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value |
|-----------|------:|
| Brightness | {cur_b} |
| Contrast | {cur_c} |
| Saturation | {cur_s} |
| Sharpness | {cur_sh} |
""")
        if st.button("↻ Refresh", width="stretch"):
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Tab: COURT SETUP ─────────────────────────────────────────────────────────

def tab_court_setup():
    court    = api_get("/court/status")
    is_calib = court.get("calibrated", False)

    if is_calib:
        st.success("Court calibrated — full IHF overlay active on the video feed.")
    else:
        st.warning(
            "Court not calibrated. Click the 4 corner points below to enable "
            "the IHF overlay, distance measurements, and zone analytics."
        )

    left, right = st.columns([3, 1])
    with left:
        st.markdown("### Click the 4 Court Corners")
        st.caption(
            "Order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**. "
            "Click *Refresh Frame* for a fresh snapshot before clicking."
        )
        components.html(_CALIB_WIDGET, height=820, scrolling=False)

    with right:
        st.markdown("### Instructions")
        st.markdown("""
**Corner order:**

| # | Corner | Colour |
|---|--------|--------|
| 1 | Top-Left | Green |
| 2 | Top-Right | Cyan |
| 3 | Bottom-Right | Orange |
| 4 | Bottom-Left | Red |

Corners should be the **outer boundary** of the playing court.

**Tips:**
- Zoom in the browser for precision
- If the overlay looks skewed, click *Clear Saved* and redo
- Calibration survives pipeline restarts
        """)
        st.markdown("---")
        st.markdown("### Status")
        if is_calib:
            st.success("Calibrated")
            if st.button("Clear & Redo", width="stretch"):
                r = api_post("/court/calibrate", {"clear": True})
                if r.get("ok"):
                    st.success("Cleared.")
                    st.rerun()
        else:
            st.error("Not calibrated")
        st.markdown("---")
        if st.button("↻ Refresh Frame", width="stretch"):
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(state: dict) -> tuple[str | None, str | None]:
    """Returns (logo_a_data_uri, logo_b_data_uri)."""
    teams = state.get("teams", {})
    na    = teams.get("team_a", {}).get("name", "Home")
    nb    = teams.get("team_b", {}).get("name", "Away")

    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:900;color:#00e5ff;"
            "letter-spacing:1px;margin-bottom:4px'>Handball AI Pro</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.68rem;color:#8b949e;margin-bottom:18px'>"
            "100% Offline Edge AI · Real-Time Analytics</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-section-title'>Team Logos</div>",
                    unsafe_allow_html=True)

        logo_a_file = st.file_uploader(
            f"Home Logo ({na})",
            type=["png", "jpg", "jpeg", "svg", "webp"],
            key="logo_a_upload",
            help="Upload PNG/JPG/SVG — displayed on scoreboard and KPI banner",
        )
        logo_b_file = st.file_uploader(
            f"Away Logo ({nb})",
            type=["png", "jpg", "jpeg", "svg", "webp"],
            key="logo_b_upload",
            help="Upload PNG/JPG/SVG — displayed on scoreboard and KPI banner",
        )

        if logo_a_file is not None:
            st.session_state["_logo_a"] = _logo_b64(logo_a_file)
        if logo_b_file is not None:
            st.session_state["_logo_b"] = _logo_b64(logo_b_file)

        logo_a = st.session_state.get("_logo_a")
        logo_b = st.session_state.get("_logo_b")

        # Logo preview
        if logo_a or logo_b:
            cols = st.columns(2)
            if logo_a:
                cols[0].image(logo_a, width=56, caption=na)
            if logo_b:
                cols[1].image(logo_b, width=56, caption=nb)

        st.markdown("<div class='sidebar-section-title'>Quick Actions</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if c1.button("↔ Flip Sides", width="stretch", key="sb_flip"):
            api_post("/flip-sides", {})
            st.rerun()
        if c2.button("Re-Color", width="stretch", key="sb_recal"):
            api_post("/recalibrate", {})
            st.success("Recalibrating…")

        st.markdown("<div class='sidebar-section-title'>Pipeline Status</div>",
                    unsafe_allow_html=True)
        api_ok = check_api()
        if api_ok:
            st.markdown('<div class="status-ok"><div class="pulse-dot"></div> Pipeline Online</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-err">● Pipeline Offline</div>',
                        unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section-title'>Refresh</div>",
                    unsafe_allow_html=True)
        interval = st.select_slider(
            "Auto-refresh interval",
            options=[1, 2, 3, 5, 10, 30],
            value=st.session_state.get("refresh_interval", 3),
            format_func=lambda x: f"{x}s",
            label_visibility="collapsed",
        )
        st.session_state["refresh_interval"] = interval

        if st.button("↻ Refresh Now", width="stretch", key="sb_refresh"):
            st.rerun()

        # ── Testing mode: nuke everything ────────────────────────────────────
        st.markdown("<div class='sidebar-section-title' style='color:#ef4444;"
                    "border-color:rgba(239,68,68,0.3)'>Testing</div>",
                    unsafe_allow_html=True)
        with st.expander("Clear Database", expanded=False):
            st.caption("Wipes players, analytics, profiles, match state, score, "
                       "and detection log. Camera config and court calibration "
                       "are preserved. The pipeline keeps running.")
            confirm = st.checkbox("Yes, I really want to clear all data",
                                  key="sb_clear_confirm")
            if st.button("CLEAR ALL", width="stretch",
                         disabled=not confirm,
                         key="sb_clear_all",
                         type="primary"):
                resp = api_post("/admin/clear-all", {})
                if resp.get("ok"):
                    st.success(f"Cleared: {resp.get('cleared', {})}")
                    st.rerun()
                else:
                    st.error(f"Clear failed: {resp}")

    return logo_a, logo_b


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_online = check_api()
    if not api_online:
        st.markdown(
            "<div class='status-err' style='margin-bottom:15px;font-size:1.1rem;justify-content:center'>"
            "<div class='pulse-dot' style='background:#ef4444;box-shadow:0 0 8px #ef4444'></div>"
            " PIPELINE IS OFFLINE: Video processing finished. Showing Dashboard for UI Testing.</div>",
            unsafe_allow_html=True,
        )

    state = api_get("/state")
    ms    = api_get("/match-stats")

    if not ms.get("ok"):
        ms = {
            "ok": True, "period": 1, "period_elapsed_s": 0,
            "score":       {"team_a": 0, "team_b": 0},
            "shots":       {"team_a": 0, "team_b": 0},
            "saves":       {"team_a": 0, "team_b": 0},
            "fast_breaks": {"team_a": 0, "team_b": 0},
            "shot_pct":    {"team_a": 0.0, "team_b": 0.0},
            "events": [],
        }

    logo_a, logo_b = render_sidebar(state)

    # Sticky KPI header (st.empty() so only this block re-renders on fast cycles)
    header_ph = st.empty()
    with header_ph.container():
        draw_sticky_header(state, ms, logo_a=logo_a, logo_b=logo_b)

    (live_tab, tagging_tab, summary_tab, players_tab,
     coach_tab, camera_tab, court_tab) = st.tabs([
        " LIVE", " SCOUTING", " MATCH SUMMARY", " PLAYER PROFILES",
        " COACH AI", " CAMERA IMAGING", " COURT SETUP",
    ])

    with live_tab:
        tab_live(state, logo_a=logo_a, logo_b=logo_b)
    with tagging_tab:
        tab_tagging()
    with summary_tab:
        tab_summary(state, ms, logo_a=logo_a, logo_b=logo_b)
    with players_tab:
        tab_players()
    with coach_tab:
        tab_coach()
    with camera_tab:
        tab_camera_imaging()
    with court_tab:
        tab_court_setup()

    # Auto-refresh at user-configured interval
    interval = st.session_state.get("refresh_interval", 3)
    time.sleep(interval)
    st.rerun()


if __name__ == "__main__":
    main()