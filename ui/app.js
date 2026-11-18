'use strict';
const el = id => document.getElementById(id);
const canvas = el('map'), ctx = canvas.getContext('2d');
let api = '', generation = 0, active = null, result = null;
let zoom = 1, offset = [0, 0], drag = null, geometry = null;

function status(message, error = false) {
  el('notice').textContent = message;
  el('notice').classList.toggle('error', error);
}
function clearResult() {
  result = geometry = null;
  el('details').hidden = true;
  el('raw').textContent = '';
  el('empty').hidden = false;
  el('model').textContent = 'Model pending';
  for (const id of ['accuracy','continuity','points','latency']) el(id).textContent = '—';
  fit();
}
function cancel() {
  generation++;
  if (active) active.abort();
}
function invalidate() {
  cancel();
  api = '';
  el('connection').classList.remove('live');
  el('connection').textContent = 'Disconnected';
  el('trip').replaceChildren(new Option('Connect to load trips',''));
  el('mapid').textContent = 'No trip selected';
  el('trip').disabled = el('predict').disabled = true;
  el('link').disabled = false;
  clearResult();
}
async function request(base, path, controller, trip) {
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const options = {signal:controller.signal, cache:'no-store', credentials:'omit'};
    if (trip !== undefined) Object.assign(options, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({trip_id:trip})});
    const response = await fetch(base + path, options);
    if (!response.ok) throw new Error(path + ' returned HTTP ' + response.status + '.');
    return await response.json();
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Request timed out. Retry when Laptop 1 is ready.');
    if (e instanceof TypeError) throw new Error('Cannot reach Laptop 1. Check the address, network and API origin settings.');
    if (e instanceof SyntaxError) throw new Error('The API returned invalid JSON.');
    throw e;
  } finally { clearTimeout(timer); }
}
el('url').addEventListener('input', () => {invalidate(); status('Connect to load the trip catalog.');});
el('connect').onsubmit = async event => {
  event.preventDefault();
  invalidate();
  const id = generation, controller = active = new AbortController();
  try {
    const url = new URL(el('url').value.trim());
    if (!['http:','https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('Enter a valid HTTP or HTTPS API address.');
    const base = url.href.replace(/\/+$/, '');
    el('link').disabled = true;
    status('Checking Laptop 1 and loading trips…');
    const health = await request(base, '/health', controller);
    if (health?.status !== 'ok') throw new Error('Laptop 1 did not report healthy status.');
    const trips = await request(base, '/trips', controller);
    if (id !== generation) return;
    if (!Array.isArray(trips) || !trips.every(t => typeof t === 'string' && t.length)) throw new Error('Unexpected trip catalog format.');
    api = base;
    el('connection').textContent = 'Connected';
    el('connection').classList.add('live');
    el('trip').replaceChildren(...trips.map(t => new Option(t,t)));
    el('trip').disabled = el('predict').disabled = !trips.length;
    el('mapid').textContent = trips[0] || 'No trips available';
    status(trips.length ? trips.length + ' trips available. Select a trip to begin.' : 'Laptop 1 returned an empty catalog.');
  } catch (e) { if (id === generation) status(e.message, true); }
  finally { if (id === generation) el('link').disabled = false; }
};
el('trip').onchange = () => {
  cancel(); clearResult();
  el('mapid').textContent = el('trip').value;
  el('predict').disabled = !api;
  status('Ready to reconstruct the selected trip.');
};
function coordinate(p) {
  return Array.isArray(p) && p.length === 2 && p.every(Number.isFinite) && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90;
}
function validate(r, trip) {
  if (r?.trip_id !== trip || !Array.isArray(r.points) || r.n_points !== r.points.length ||
      !r.points.every(p => p && coordinate([p.lon,p.lat])) || !Array.isArray(r.route_coords) ||
      !r.route_coords.every(coordinate) || typeof r.model_version !== 'string' ||
      !Number.isFinite(r.latency_ms) || r.latency_ms < 0 ||
      ![r.accuracy,r.route_continuity].every(v => v === null || Number.isFinite(v) && v >= 0 && v <= 1))
    throw new Error('The prediction does not match the selected trip or API format.');
}
el('predict').onclick = async () => {
  cancel(); clearResult();
  const trip = el('trip').value, id = generation, controller = active = new AbortController();
  status('Reconstructing trip ' + trip + '…'); el('predict').disabled = true;
  try {
    const r = await request(api,'/predict',controller,trip);
    if (id !== generation) return;
    validate(r,trip); result = r;
    geometry = project(r.points.map(p => [p.lon,p.lat]), r.route_coords);
    const pct = v => v === null ? 'Unavailable' : (100*v).toFixed(1)+'%';
    el('accuracy').textContent = pct(r.accuracy); el('continuity').textContent = pct(r.route_continuity);
    el('points').textContent = r.n_points.toLocaleString(); el('latency').textContent = Math.round(r.latency_ms)+' ms';
    el('model').textContent = r.model_version;
    el('details').hidden = false; el('raw').textContent = JSON.stringify(r,null,2);
    el('empty').hidden = true; fit();
    status(geometry ? 'Reconstruction complete.' : 'Completed. The API returned no map coordinates.');
  } catch(e) { if (id === generation) status(e.message,true); }
  finally { if (id === generation) el('predict').disabled = false; }
};
function project(gps, route) {
  const origin = gps[0] || route[0];
  if (!origin) return null;
  const factor = Math.max(.0001, Math.cos(origin[1]*Math.PI/180));
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  function xy([lon,lat]) {
    const x = (((lon-origin[0]+540)%360)-180)*factor, y = lat-origin[1];
    x0=Math.min(x0,x); x1=Math.max(x1,x); y0=Math.min(y0,y); y1=Math.max(y1,y);
    return [x,y];
  }
  const observations = gps.map(xy), predicted = route.map(xy);
  return {observations,predicted,x0,x1,y0,y1,origin,factor};
}
function fit() {zoom=1; offset=[0,0]; draw();}
function adjust(mult) {if (geometry) {zoom=Math.min(64,Math.max(.25,zoom*mult)); draw();}}
function draw() {
  const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio || 1;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.scale(dpr,dpr);
  ctx.fillStyle='#100f1d';ctx.fillRect(0,0,w,h);
  ctx.lineWidth=1;ctx.strokeStyle='#211e33';ctx.beginPath();
  for(let x=0;x<w;x+=40){ctx.moveTo(x,0);ctx.lineTo(x,h);}
  for(let y=0;y<h;y+=40){ctx.moveTo(0,y);ctx.lineTo(w,y);}ctx.stroke();
  for(const id of ['plus','minus','fit','gps','route']) el(id).disabled=!geometry;
  if(!geometry) {
    if (result) return;
    // An illustrative relief surface occupies the idle map. Real trips use the projection below.
    ctx.save();
    ctx.scale(w / 1000, h / 460);
    const glow = ctx.createRadialGradient(630, 195, 12, 630, 195, 490);
    glow.addColorStop(0, '#322843'); glow.addColorStop(1, '#11101c');
    ctx.fillStyle = glow; ctx.fillRect(0, 0, 1000, 460);
    function elevation(x, y) {
      const ridge = x - (570 + 100 * Math.sin(y / 89) - .28 * y);
      const peak = Math.exp(-ridge * ridge / 36000) * (132 + 43 * Math.cos(y / 65));
      return peak + 32 * Math.sin(x / 99 + y / 59) + 16 * Math.cos(x / 53 - y / 101);
    }
    for (let y = -70; y < 620; y += 6) {
      ctx.beginPath();
      for (let x = -15; x <= 1020; x += 4) {
        const py = y - elevation(x, y);
        if (x === -15) ctx.moveTo(x, py); else ctx.lineTo(x, py);
      }
      ctx.lineTo(1020, 650); ctx.lineTo(-15, 650); ctx.closePath();
      ctx.fillStyle = '#171424'; ctx.fill();
      const ink = ctx.createLinearGradient(0, 0, 1000, 0);
      ink.addColorStop(0, '#211d33'); ink.addColorStop(.28, '#584576');
      ink.addColorStop(.53, '#a68bcc'); ink.addColorStop(.67, '#dbab89');
      ink.addColorStop(.84, '#76618d'); ink.addColorStop(1, '#393146');
      ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.restore(); return;
  }
  const g=geometry, scale=Math.min((w-66)/Math.max(.00001,g.x1-g.x0),(h-62)/Math.max(.00001,g.y1-g.y0))*zoom;
  const screen=([x,y])=>[w/2+offset[0]+(x-(g.x0+g.x1)/2)*scale,h/2+offset[1]-(y-(g.y0+g.y1)/2)*scale];
  if(el('route').checked && g.predicted.length) {
    ctx.beginPath();g.predicted.forEach((p,i)=>{const [x,y]=screen(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});
    ctx.strokeStyle='#ffbc69';ctx.lineWidth=4;ctx.lineJoin='round';ctx.stroke();
    const [x,y]=screen(g.predicted[0]);ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle='#ffbc69';ctx.fill();
  }
  if(el('gps').checked) {
    ctx.fillStyle='#98acff';ctx.beginPath();
    for(const p of g.observations){const [x,y]=screen(p);ctx.moveTo(x+2,y);ctx.arc(x,y,2,0,2*Math.PI);}ctx.fill();
  }
  const centerLat=g.origin[1]+(g.y0+g.y1)/2+offset[1]/scale;
  const centerLon=g.origin[0]+((g.x0+g.x1)/2-offset[0]/scale)/g.factor;
  ctx.fillStyle='#d6e7dc';ctx.font='11px monospace';
  ctx.fillText('CENTER  '+centerLat.toFixed(4)+'°, '+centerLon.toFixed(4)+'°',12,h-12);
}
el('gps').onchange=el('route').onchange=draw;
el('plus').onclick=()=>adjust(1.5);el('minus').onclick=()=>adjust(1/1.5);el('fit').onclick=fit;
canvas.onpointerdown=e=>{if(geometry){drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);}};
canvas.onpointermove=e=>{if(drag){offset[0]+=e.clientX-drag[0];offset[1]+=e.clientY-drag[1];drag=[e.clientX,e.clientY];draw();}};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
new ResizeObserver(draw).observe(canvas);
draw();
