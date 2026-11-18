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
function drawInterchange(w, h) {
  // Orthographic illustration in local scene units. It never supplies trip coordinates.
  ctx.save();
  const scale = Math.min(w / 740, h / 360);
  ctx.translate(w * .5, h * .49);
  ctx.scale(scale, scale);
  const project = (x, y, z = 0) => [(x - y) * .84, (x + y) * .39 - z];
  function polygon(vertices, color) {
    ctx.beginPath();
    vertices.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
  }
  function line(points, color, width) {
    ctx.beginPath();
    points.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = ctx.lineCap = 'round'; ctx.stroke();
  }
  function block(x,y,a,b,z,colors) {
    const base = [[x,y],[x+a,y],[x+a,y+b],[x,y+b]].map(([i,j])=>project(i,j));
    const top = base.map(([i,j])=>[i,j-z]);
    polygon([base[1],base[2],top[2],top[1]],colors[0]);
    polygon([base[2],base[3],top[3],top[2]],colors[1]);
    polygon(top,colors[2]);
    return top;
  }
  const floor = '#2b2936';
  block(-202,-174,404,348,7,['#191722','#201d2a',floor]);
  for (let x=-192;x<=192;x+=32) line([project(x,-172,7),project(x,172,7)],'#494252',2);
  for (let y=-160;y<=160;y+=32) line([project(-200,y,7),project(200,y,7)],'#494252',2);
  const buildings=[];
  for(let x=-188;x<187;x+=32) for(let y=-155;y<153;y+=32) {
    if(Math.abs(x+9)<35 || Math.abs(y+10)<34 || (Math.abs(x)<105 && Math.abs(y)<94)) continue;
    buildings.push([x,y]);
  }
  buildings.sort((a,b)=>a[0]+a[1]-b[0]-b[1]);
  for(const [x,y] of buildings) {
    const n=Math.abs(Math.sin(x*7.1+y*3.37)), z=12+Math.round(n*25);
    const top=block(x,y,15+n*7,17,z,['#383344','#454052','#666074']);
    line([top[3],top[2],top[1]],'#83798b',.65);
    if(n>.56) block(x+4,y+4,6,7,z+2,['#494352','#4a4352','#807888']);
    for(let level=10;level<z-3;level+=7)
      line([project(x+3,y+17,level),project(x+12,y+17,level)],'#8b79926b',1);
  }
  function ramp(fn,height,width) {
    const points=Array.from({length:101},(_,i)=>{const [x,y]=fn(i/100);return project(x,y,height);});
    line(points.map(([x,y])=>[x+3,y+height*.65]),'#15131c99',width+3);
    line(points,'#897c93',width+2); line(points,'#423d4c',width);
    ctx.setLineDash([4,5]);line(points,'#ac9bad',.85);ctx.setLineDash([]);
    return points;
  }
  ramp(t=>[-200+400*t,0],9,13);
  for(const sign of [-1,1]) {
    ramp(t=>[sign*(92-92*Math.sin(t*Math.PI/2)),sign*(73-73*Math.cos(t*Math.PI/2))],13,6);
    ramp(t=>[sign*(51-49*Math.cos(t*1.5*Math.PI)),sign*(-39+37*Math.sin(t*1.5*Math.PI))],15,6);
  }
  for(let y=-154;y<170;y+=36) block(-3,y,6,7,20,['#36313e','#53495c','#796a83']);
  const flyover=ramp(t=>[9*Math.sin(t*Math.PI*2),-170+340*t],24,15);
  ctx.shadowColor='#ff963e';ctx.shadowBlur=11;
  line(flyover,'#ff9e44',3.5);ctx.shadowBlur=0;line(flyover,'#ffe0a3',1.1);
  for(let i=3;i<100;i+=4) {
    const [x,y]=flyover[i];ctx.beginPath();ctx.arc(x+4+Math.sin(i)*2,y-1,1.9,0,Math.PI*2);
    ctx.shadowColor='#53acff';ctx.shadowBlur=5;ctx.fillStyle='#81caff';ctx.fill();ctx.shadowBlur=0;
  }
  ctx.restore();
}

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
    drawInterchange(w, h); return;
  }
  const g=geometry, scale=Math.min((w-66)/Math.max(.00001,g.x1-g.x0),(h-62)/Math.max(.00001,g.y1-g.y0))*zoom;
  const screen=([x,y])=>[w/2+offset[0]+(x-(g.x0+g.x1)/2)*scale,h/2+offset[1]-(y-(g.y0+g.y1)/2)*scale];
  if(el('route').checked && g.predicted.length) {
    ctx.beginPath();g.predicted.forEach((p,i)=>{const [x,y]=screen(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});
    ctx.strokeStyle='#ffbc69';ctx.lineWidth=4;ctx.lineJoin='round';ctx.stroke();
    const [x,y]=screen(g.predicted[0]);ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle='#ffbc69';ctx.fill();
  }
  if(el('gps').checked) {
    ctx.fillStyle='#6bbbff';ctx.beginPath();
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
