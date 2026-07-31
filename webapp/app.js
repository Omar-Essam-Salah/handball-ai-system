/* Handball Studio — mobile web app.
   All requests are RELATIVE by default, so the app talks to whatever server
   served it (dynamic IP, nothing hardcoded). When bundled in the WebView APK
   it can also point at a discovered/typed server via localStorage 'srv'. */
const App = (() => {
  const $ = id => document.getElementById(id);
  const TEAM = {team_a:'Home', team_b:'Away', goalkeeper:'GK'};
  const TC   = {team_a:'#ff5658', team_b:'#36b6ff', goalkeeper:'#3fe07a', unknown:'#9aa0aa'};

  let base = localStorage.getItem('srv') || '';     // '' = same origin
  let connected = false, deferredPrompt = null, statusTimer, goalTimer;

  const api = p => base + p;

  async function ping() {
    try {
      const r = await fetch(api('/ping'), {cache:'no-store'});
      if (!r.ok) return false;
      const j = await r.json();
      return j && j.app === 'handball';
    } catch (e) { return false; }
  }

  async function tryConnect(showSpin = true) {
    $('connSpin').style.display = showSpin ? 'block' : 'none';
    if (await ping()) { onConnected(); return true; }
    $('connMsg').textContent =
      base ? ('No server at ' + base) : 'Open the URL shown on the PC, or type it below.';
    return false;
  }

  function onConnected() {
    if (connected) return;
    connected = true;
    $('connect').classList.add('hidden');
    $('app').classList.remove('hidden');
    showLoading();
    const vid = $('vid');
    vid.addEventListener('load', hideLoading, {once:false});
    vid.src = api('/video');                          // start MJPEG
    $('srvLbl').textContent = 'server: ' + (base || location.host);
    bindControls();
    poll(); loadGoals();
    statusTimer = setInterval(poll, 700);
    goalTimer   = setInterval(loadGoals, 2500);
  }

  /* ---------- immersive loading screen ---------- */
  let loadRAF = null, loadTimeout = null;
  function showLoading() {
    const el = $('loading'); if (!el) return;
    el.classList.remove('hidden');
    startParticles();
    clearTimeout(loadTimeout);
    loadTimeout = setTimeout(hideLoading, 30000);     // safety fallback
  }
  function hideLoading() {
    const el = $('loading'); if (!el || el.classList.contains('hidden')) return;
    el.classList.add('hidden');
    if (loadRAF) cancelAnimationFrame(loadRAF), loadRAF = null;
    clearTimeout(loadTimeout);
  }
  function startParticles() {
    const cv = $('loadCanvas'); if (!cv) return;
    const ctx = cv.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    function resize(){ cv.width = innerWidth*dpr; cv.height = innerHeight*dpr; }
    resize();
    const N = Math.min(70, Math.floor(innerWidth/8));
    const ps = Array.from({length:N}, () => ({
      x:Math.random()*cv.width, y:Math.random()*cv.height,
      vx:(Math.random()-.5)*0.6*dpr, vy:(Math.random()-.5)*0.6*dpr
    }));
    function frame(){
      ctx.clearRect(0,0,cv.width,cv.height);
      for (const p of ps){ p.x+=p.vx; p.y+=p.vy;
        if(p.x<0||p.x>cv.width)p.vx*=-1; if(p.y<0||p.y>cv.height)p.vy*=-1; }
      for (let i=0;i<N;i++){ for (let j=i+1;j<N;j++){
        const dx=ps[i].x-ps[j].x, dy=ps[i].y-ps[j].y, d=Math.hypot(dx,dy);
        if (d<120*dpr){ ctx.strokeStyle='rgba(54,182,255,'+(1-d/(120*dpr))*0.25+')';
          ctx.lineWidth=dpr; ctx.beginPath(); ctx.moveTo(ps[i].x,ps[i].y);
          ctx.lineTo(ps[j].x,ps[j].y); ctx.stroke(); } } }
      for (const p of ps){ ctx.fillStyle='rgba(127,176,255,.9)';
        ctx.beginPath(); ctx.arc(p.x,p.y,1.6*dpr,0,7); ctx.fill(); }
      loadRAF = requestAnimationFrame(frame);
    }
    frame();
  }

  function onLost() {
    if (!connected) return;
    $('livePill').className = 'live-pill off';
    $('liveTxt').textContent = 'offline';
  }

  function manualConnect() {
    let v = $('serverInput').value.trim();
    if (!v) return;
    if (!/^https?:\/\//.test(v)) v = 'http://' + v;
    base = v.replace(/\/$/, '');
    localStorage.setItem('srv', base);
    connected = false;
    tryConnect();
  }

  async function poll() {
    let s;
    try { s = await (await fetch(api('/status'), {cache:'no-store'})).json(); }
    catch (e) { onLost(); return; }
    const h = s.hud || {}, sc = h.score || {};
    $('sa').textContent = sc.team_a ?? 0;
    $('sb').textContent = sc.team_b ?? 0;
    const e = Math.floor(h.period_elapsed_s || 0);
    $('clock').textContent = 'P' + (h.period || 1) + ' ' +
      String(Math.floor(e/60)).padStart(2,'0') + ':' + String(e%60).padStart(2,'0');
    $('poss').textContent = (TEAM[h.possession] || '—') +
      (h.possession_duration_s ? '  ' + Math.round(h.possession_duration_s) + 's' : '');
    const sh = h.shots||{}, fb = h.fast_breaks||{}, ps = h.passes||{};
    $('shots').textContent  = 'H ' + (sh.team_a||0) + '  ·  A ' + (sh.team_b||0);
    $('fb').textContent     = 'H ' + (fb.team_a||0) + '  ·  A ' + (fb.team_b||0);
    $('passes').textContent = 'H ' + (ps.team_a||0) + '  ·  A ' + (ps.team_b||0);
    $('ball').textContent = s.ball_state || '—';
    $('fps').textContent  = (s.fps || 0).toFixed(1);

    const live = !!s.running;
    $('livePill').className = 'live-pill ' + (live ? 'on' : 'off');
    $('liveTxt').textContent = live ? 'LIVE' : 'offline';
    if (s.hud && s.hud.score !== undefined) hideLoading();
    syncControls(s.controls);
    // playback sync (don't fight the user mid-interaction)
    const c = s.controls || {};
    if (document.activeElement !== $('seekBar')) {
      paused = !!c.paused; $('pbPlay').textContent = paused ? '▶' : '⏸';
      document.querySelectorAll('#speeds button').forEach(b =>
        b.classList.toggle('on', parseFloat(b.dataset.spd) === parseFloat(c.speed || 1)));
    }

    // events
    $('events').innerHTML = (s.events || []).map(x => `<div class="ev">${esc(x)}</div>`).join('')
      || '<div class="muted small">no events yet</div>';

    // players
    const pl = s.players || [];
    $('reidBanner').textContent =
      `Persistent identities: ${s.reid_known||0}   ·   on court: ${pl.length}`;
    $('players').innerHTML = pl.map(p =>
      `<div class="row player"><span class="num">${esc(p.label)}</span>
       <span class="team" style="color:${TC[p.team]||'#9aa0aa'}">${TEAM[p.team]||'—'}</span></div>`
    ).join('') || '<div class="muted small">no players detected yet</div>';

    // biomechanics technique cards
    const tech = s.technique || [];
    $('technique').innerHTML = tech.length
      ? tech.map(techCard).join('')
      : '<div class="muted small">no clear poses yet — players need to be well-framed for biomechanics</div>';

    renderTactics(s.formation, s.shotmap);

    // AI vision-LLM verdict on the last goal
    const av = s.ai_verdict, ave = $('aiVerdict');
    if (ave) {
      if (av && av.outcome && av.outcome !== '?') {
        const col = {GOAL:'#3fe07a', SAVE:'#36b6ff', MISS:'#ffb648', PASS:'#9aa0aa'}[av.outcome] || '#9aa0aa';
        ave.style.display = 'block';
        ave.style.borderColor = col;
        ave.innerHTML = `🧠 <b style="color:${col}">AI: ${av.outcome}</b> ` +
          `<span class="muted">${Math.round((av.confidence||0)*100)}%</span> ` +
          `<span class="muted">· ${esc(av.reason||'')}</span>`;
      } else { ave.style.display = 'none'; }
    }
  }

  function renderTactics(form, shots) {
    form = form || {}; shots = shots || [];
    const card = (name, f, color) => f
      ? `<div class="form-card"><div class="fc-h" style="color:${color}">${name}</div>
           <div class="fc-row"><span>Players</span><b>${f.n}</b></div>
           <div class="fc-row"><span>Width</span><b>${f.width_pct}%</b></div>
           <div class="fc-row"><span>Depth</span><b>${f.depth_pct}%</b></div>
           <div class="fc-row"><span>Compactness</span><b>${f.compactness}%</b></div></div>`
      : `<div class="form-card"><div class="fc-h" style="color:${color}">${name}</div>
           <div class="muted small">not enough players</div></div>`;
    $('formation').innerHTML = card('Home', form.team_a, '#ff5658') + card('Away', form.team_b, '#36b6ff');
    drawCourt(shots);
  }

  function drawCourt(shots) {
    const cv = $('court'); if (!cv) return;
    const ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0c1b14'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#2a5b44'; ctx.lineWidth = 2;
    ctx.strokeRect(8, 8, W - 16, H - 16);
    ctx.beginPath(); ctx.moveTo(W/2, 8); ctx.lineTo(W/2, H - 8); ctx.stroke();
    ctx.beginPath(); ctx.arc(8, H/2, H*0.34, -Math.PI/2.3, Math.PI/2.3); ctx.stroke();
    ctx.beginPath(); ctx.arc(W-8, H/2, H*0.34, Math.PI-Math.PI/2.3, Math.PI+Math.PI/2.3); ctx.stroke();
    $('shotCount').textContent = shots.length ? `· ${shots.length} attempts` : '';
    for (const s of shots) {
      const x = 8 + s.x*(W-16), y = 8 + s.y*(H-16);
      const col = s.team === 'team_a' ? '#ff5658' : '#36b6ff';
      ctx.beginPath(); ctx.arc(x, y, s.goal ? 6 : 5, 0, 7);
      if (s.goal) { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke(); }
    }
  }

  function techCard(t) {
    const chip = (k, v, u) => (v === null || v === undefined) ? '' :
      `<div class="chip"><span>${k}</span><b>${v}${u || ''}</b></div>`;
    return `<div class="tech-card${t.airborne ? ' airborne' : ''}">
      <div class="tech-head">
        <span class="num" style="color:${TC[t.team] || '#9aa0aa'}">${esc(t.label)}</span>
        ${t.airborne ? '<span class="air-pill">▲ AIRBORNE</span>' : ''}
      </div>
      <div class="chips">
        ${chip('Jump', t.jump_cm, ' cm')}${chip('Peak', t.peak_jump_cm, ' cm')}
        ${chip('Arm', t.arm_angle, '°')}${chip('Torque', t.torque, '°')}${chip('Release', t.release_cm, ' cm')}
      </div></div>`;
  }

  async function loadGoals() {
    try {
      const g = (await (await fetch(api('/goals'),{cache:'no-store'})).json()).goals || [];
      $('goals').innerHTML = g.map(n =>
        `<figure onclick="App.view('${api('/goal/'+n)}')">
           <img src="${api('/goal/'+n)}" loading="lazy"><figcaption>${n.replace('.png','')}</figcaption>
         </figure>`).join('') || '<div class="muted small">no goals saved yet</div>';
    } catch (e) {}
  }

  /* ---------- pro Shot/Save reports (matches the IDEA PDFs) ---------- */
  let R_ZONES = [], R_CELLS = [], R_OFF = [], R_OUT = [];
  let repMode_ = "shot", tagCell = "CH", tagOut = "GOAL";
  const fj = async p => (await fetch(api(p), {cache:"no-store"})).json();

  async function loadReports() {
    let m; try { m = await fj("/report/meta"); } catch (e) { return; }
    R_ZONES = m.zones; R_CELLS = m.cells; R_OFF = m.off_target; R_OUT = m.outcomes;
    fillPlayerSel(m);
    $("tgZone").innerHTML = R_ZONES.map(z => `<option>${z}</option>`).join("");
    $("tgCells").innerHTML = R_CELLS.map(c =>
      `<button class="gp${c===tagCell?' on':''}" onclick="App.pickCell('${c}',this)">${c}</button>`).join("");
    $("tgOff").innerHTML = R_OFF.map(c =>
      `<button class="gp off" onclick="App.pickCell('${c}',this)">${c}</button>`).join("");
    $("tgOutcome").innerHTML = R_OUT.map(o =>
      `<button onclick="App.pickOut('${o}',this)" class="${o===tagOut?'on':''}">${o}</button>`).join("");
    $("tgGk").innerHTML = '<option value="">GK (optional)</option>' +
      (m.goalkeepers||[]).map(g => `<option>${g.gk}</option>`).join("");
    renderReport();
  }
  function fillPlayerSel(m) {
    const sel = $("repPlayer");
    if (repMode_ === "save") {
      sel.innerHTML = (m.goalkeepers||[]).map(g => `<option value="${g.gk}|gk">${g.gk} (GK ${g.saves}/${g.faced})</option>`).join("")
        || '<option value="">— no keepers tagged yet —</option>';
    } else {
      sel.innerHTML = (m.players||[]).map(p => `<option value="${p.player}|${p.team}">${esc(p.player)}${p.number?(' #'+p.number):''}  (${p.goals}/${p.shots})</option>`).join("")
        || '<option value="">— no shots tagged yet —</option>';
    }
  }
  async function repMode(m) {
    repMode_ = m;
    $("repModeShot").classList.toggle("on", m==="shot");
    $("repModeSave").classList.toggle("on", m==="save");
    try { fillPlayerSel(await fj("/report/meta")); } catch(e){}
    renderReport();
  }
  function goalGrid(cells, opt) {
    opt = opt || {};
    const order = ["LU","CU","RU","LH","CH","RH","LD","CD","RD"];
    let h = `<div class="goalmouth${opt.big?' big':''}${opt.save?' save':''}">`;
    for (const c of order) {
      const [hit, tot] = cells[c] || [0,0];
      let cls = "", txt = "";
      if (tot > 0) {
        const r = hit/tot;
        txt = opt.pct ? `${hit}/${tot}<small>${Math.round(r*100)}%</small>` : `${hit}\\${tot}`;
        cls = r >= 0.5 ? "good" : (hit>0 ? "mid" : "bad");
      }
      h += `<div class="gm ${cls}"><span class="gm-t">${c}</span>${txt?`<b>${txt}</b>`:''}</div>`;
    }
    return h + "</div>";
  }
  async function renderReport() {
    const v = $("repPlayer").value; if (!v) { $("repHeader").innerHTML=""; $("repSummary").innerHTML=""; $("repZones").innerHTML=""; $("repOutcome").innerHTML=""; return; }
    const [name, team] = v.split("|");
    if (repMode_ === "save") {
      const r = await fj("/report/save?gk=" + encodeURIComponent(name));
      $("repHeader").innerHTML = `<b>${esc(name)}</b> · Save Report · faced ${r.faced} · saves ${r.saves} · <span class="gA" style="color:#3fe07a">${r.save_pct}%</span>`;
      $("repSummary").innerHTML = goalGrid(r.grid.summary, {save:true, big:true, pct:true});
      $("repZones").innerHTML = R_ZONES.map(z => `<div class="zcard"><div class="zname">${z}</div>${goalGrid(r.grid.zones[z], {save:true})}</div>`).join("");
      $("repOutcome").innerHTML = "";
    } else {
      const r = await fj(`/report/shot?player=${encodeURIComponent(name)}&team=${team}`);
      $("repHeader").innerHTML = `<b>${esc(name)}${r.number?(' #'+r.number):''}</b> · Shot Report · <span style="color:#3fe07a">${r.goals}/${r.total_shots}</span> goals · ${r.efficiency}% eff`;
      $("repSummary").innerHTML = goalGrid(r.grid.summary, {big:true});
      $("repZones").innerHTML = R_ZONES.map(z => `<div class="zcard"><div class="zname">${z}</div>${goalGrid(r.grid.zones[z], {})}</div>`).join("");
      const oc = r.by_outcome, tot = r.total_shots || 1;
      $("repOutcome").innerHTML = ["GOAL","SAVE","MISS","POST","BLOCK"].map(o =>
        `<div class="ob"><span>${o}</span><div class="obar"><i style="width:${100*(oc[o]||0)/tot}%"></i></div><b>${oc[o]||0}</b></div>`).join("");
    }
  }
  function pickCell(c, el) { tagCell = c; document.querySelectorAll("#tgCells .gp,#tgOff .gp").forEach(b=>b.classList.remove("on")); el.classList.add("on"); }
  function pickOut(o, el) { tagOut = o; document.querySelectorAll("#tgOutcome button").forEach(b=>b.classList.remove("on")); el.classList.add("on"); }
  async function tagShot() {
    const body = { player: $("tgName").value || "?", number: parseInt($("tgNum").value)||null,
      team: $("tgTeam").value, zone: $("tgZone").value, cell: tagCell, outcome: tagOut,
      gk: $("tgGk").value || null };
    await fetch(api("/report/shot/add"), {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    loadReports();
  }

  function go(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.toggle('on', p.id === 'page-'+page));
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.page === page));
  }

  function cmd(c) {
    fetch(api('/cmd/' + encodeURIComponent(c)), {method:'POST'}).catch(()=>{});
    if (c === 'd') $('dbgBtn').classList.toggle('on');
  }

  /* ---------- live AI control center ---------- */
  let ctlBound = false, dragging = null, sendTimer = null;
  function sendControl(obj) {
    fetch(api('/control'), {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(obj)}).catch(()=>{});
  }
  function bindControls() {
    if (ctlBound) return; ctlBound = true;
    document.querySelectorAll('#speeds button').forEach(b =>
      b.addEventListener('click', () => setSpeed(b.dataset.spd)));
    document.querySelectorAll('input[data-ctl]').forEach(el => {
      const key = el.dataset.ctl;
      if (el.type === 'checkbox') {
        el.addEventListener('change', () => sendControl({[key]: el.checked}));
      } else { // range
        const out = $('out-' + key);
        el.addEventListener('input', () => {
          dragging = key; if (out) out.textContent = parseFloat(el.value).toFixed(2);
          clearTimeout(sendTimer);
          sendTimer = setTimeout(() => sendControl({[key]: parseFloat(el.value)}), 90);
        });
        el.addEventListener('change', () => { dragging = null; });
      }
    });
  }
  function syncControls(c) {
    if (!c) return;
    document.querySelectorAll('input[data-ctl]').forEach(el => {
      const key = el.dataset.ctl;
      if (c[key] === undefined) return;
      if (el.type === 'checkbox') { if (document.activeElement !== el) el.checked = !!c[key]; }
      else if (dragging !== key) {
        el.value = c[key];
        const out = $('out-' + key); if (out) out.textContent = parseFloat(c[key]).toFixed(2);
      }
    });
  }

  function openSource() {
    const s = $('src').value.trim(); if (!s) return;
    showLoading();
    fetch(api('/open'), {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({source: s})}).catch(()=>{});
  }
  function stopPipeline() { fetch(api('/stop'), {method:'POST'}).catch(()=>{}); }

  async function scanCameras() {
    $('scanResults').innerHTML = '<div class="cam">Scanning network… (a few seconds)</div>';
    try {
      const r = await (await fetch(api('/scan'))).json();
      $('scanResults').innerHTML = (r.cameras || []).map(c =>
        `<div class="cam" onclick="App.pick('${c.url}')">${c.ip}
          ${c.w ? '· ' + c.w + '×' + c.h + ' ✓' : '· tap to try'}</div>`
      ).join('') || `<div class="cam">No RTSP cameras on ${r.subnet}.0/24</div>`;
    } catch (e) { $('scanResults').innerHTML = '<div class="cam">scan failed</div>'; }
  }
  function pick(url) { $('src').value = url; openSource(); }

  function view(u) { $('viewerImg').src = u; $('viewer').classList.remove('hidden'); }
  function closeViewer() { $('viewer').classList.add('hidden'); }

  /* ---------- playback ---------- */
  let paused = false, seekTimer = null;
  function togglePause() {
    paused = !paused;
    $('pbPlay').textContent = paused ? '▶' : '⏸';
    sendControl({paused});
  }
  function step() {
    paused = true; $('pbPlay').textContent = '▶';
    sendControl({paused: true, step: 1});
  }
  function setSpeed(v) {
    document.querySelectorAll('#speeds button').forEach(b =>
      b.classList.toggle('on', parseFloat(b.dataset.spd) === parseFloat(v)));
    sendControl({speed: parseFloat(v)});
  }
  function seek(v) {
    clearTimeout(seekTimer);
    seekTimer = setTimeout(() => sendControl({seek: parseFloat(v) / 1000}), 120);
  }

  function install() { if (deferredPrompt) { deferredPrompt.prompt(); deferredPrompt = null; } }

  function tool(name) {
    fetch(api("/tools/" + name), {method:"POST"})
      .then(r => r.json())
      .then(j => alert(j.ok ? `Launched "${name}" on the PC (check the PC screen).`
                            : `Could not launch: ${j.error||name}`))
      .catch(() => alert("Tools only work on the PC running the server."));
  }

  // Court calibration now lives fully in the browser (live view + zoom).
  function openCalib() { location.href = (base || '') + '/calib.html'; }

  const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

  // boot
  window.addEventListener('beforeinstallprompt', e => {
    e.preventDefault(); deferredPrompt = e;
    const b = $('installBtn'); if (b) b.hidden = false;
  });
  window.addEventListener('load', () => {
    if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(()=>{});
    tryConnect();
    // keep retrying while on the connect screen
    setInterval(() => { if (!connected) tryConnect(false); }, 2500);
  });

  return {go, cmd, openSource, stopPipeline, scanCameras, pick, view,
          closeViewer, install, manualConnect,
          togglePause, step, setSpeed, seek,
          loadReports, renderReport, repMode, pickCell, pickOut, tagShot, tool,
          openCalib};
})();
