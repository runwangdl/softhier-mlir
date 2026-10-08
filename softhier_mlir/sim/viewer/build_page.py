"""Build a self-contained HTML activity viewer from one or more clipped-event JSON files
(produced by `python -m softhier_mlir.sim.trace <log> --json events.json`).

    python -m softhier_mlir.sim.viewer.build_page out.html "label A=a.json" "label B=b.json"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>SigLIP 层活动时间线</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* layout: one scrolling column; the timeline canvas is the hero, facts above it, per-cluster table below. */
:root{
  --bg:#f6f5f0; --panel:#ffffff; --fg:#1d2126; --muted:#5c6670; --line:#d9d6cc;
  --redmule:#d0641c; --idma:#1f6fc2; --sync:#c9c6bd; --core:#5a8f5a; --grid:#e6e3da;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#141a21; --panel:#1b232c; --fg:#e8ebee; --muted:#9aa5b1; --line:#2d3842;
  --redmule:#f08a3c; --idma:#5aa3ea; --sync:#3a4650; --core:#7cb77c; --grid:#243039; color-scheme:dark }}
:root[data-theme="dark"]{
  --bg:#141a21; --panel:#1b232c; --fg:#e8ebee; --muted:#9aa5b1; --line:#2d3842;
  --redmule:#f08a3c; --idma:#5aa3ea; --sync:#3a4650; --core:#7cb77c; --grid:#243039; color-scheme:dark }
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.45;padding-inline:16px;padding-block:20px 40px;max-width:1240px;margin:0 auto}
h1{font-size:1.45rem;font-weight:600;margin:0 0 4px;text-wrap:balance}
.sub{color:var(--muted);margin:0 0 18px;max-width:70ch}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
select,button{font:inherit;font-size:.92rem;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
button{cursor:pointer}button:focus-visible,select:focus-visible{outline:2px solid var(--idma);outline-offset:2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
.stat .k{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.stat .v{font-family:var(--mono);font-size:1.25rem;font-variant-numeric:tabular-nums;margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.85rem;color:var(--muted);margin:6px 0 10px}
.legend i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.wrap{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;position:relative}
canvas{display:block;width:100%;height:auto;touch-action:none}
.tip{position:absolute;pointer-events:none;background:var(--fg);color:var(--bg);font-family:var(--mono);font-size:.78rem;padding:6px 8px;border-radius:4px;white-space:pre;max-width:60ch;display:none;z-index:2}
.hint{font-size:.82rem;color:var(--muted);margin:8px 0 18px}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);font-family:var(--mono);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-weight:500;color:var(--muted);font-size:.78rem;letter-spacing:.04em}
tr:last-child td{border-bottom:0}
.bar{display:inline-block;height:8px;background:var(--redmule);vertical-align:middle;border-radius:2px}
h2{font-size:1.05rem;font-weight:600;margin:22px 0 8px}
p.note{color:var(--muted);font-size:.9rem;max-width:75ch}
</style>

<h1>SigLIP 层活动时间线</h1>
<p class="sub">SoftHier gvsoc 的 RedMulE / iDMA / barrier trace，按 cluster 展开。一层 SigLIP encoder，S=256、d=768、12 头、FFN 3072，16 个 cluster。时间轴只含计时的 kernel 窗口。</p>

<div class="row">
  <label for="trace">变体</label><select id="trace"></select>
  <button id="reset" type="button">重置缩放</button>
  <span class="hint" style="margin:0">滚轮缩放，拖动平移，悬停看明细</span>
</div>
<div class="stats" id="stats"></div>
<div class="legend">
  <span><i style="background:var(--redmule)"></i>RedMulE 在算（bar 内标注利用率）</span>
  <span><i style="background:var(--idma)"></i>iDMA 传输</span>
  <span><i style="background:var(--core)"></i>核在执行（不在 barrier 里：行算子、标量 softmax、DMA 发起、等待）</span>
  <span><i style="background:var(--sync)"></i>barrier 等待</span>
</div>
<div class="wrap"><canvas id="cv" width="1200" height="620"></canvas><div class="tip" id="tip"></div></div>
<p class="hint">RedMulE 和 iDMA 来自硬件 trace；"核在执行"是推断值：kernel 窗口里不在 barrier 的时间。行算子在核上跑，模拟器没有单独的 trace。</p>

<h2>每个 cluster 的占用</h2>
<div class="tw"><table id="tbl"></table></div>
<p class="note" id="reading"></p>

<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sel = document.getElementById('trace'), cv = document.getElementById('cv'), tip = document.getElementById('tip');
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.label;sel.appendChild(o);});
let cur = 0, view = {t0:0,t1:1}, lanes = [], hover=null;
const LANE_H = 34, TOP = 28, LEFT = 86;
function prep(d){
  const clusters=[...new Set(d.events.map(e=>e.cluster))].sort((a,b)=>a-b);
  const per = clusters.map(c=>{
    const ev=d.events.filter(e=>e.cluster===c);
    const sync=ev.filter(e=>e.unit==='sync').sort((a,b)=>a.t0-b.t0);
    // inferred core activity = complement of barrier waits inside the window
    const core=[]; let t=0;
    for(const s of sync){ if(s.t0>t) core.push({t0:t,t1:s.t0}); t=Math.max(t,s.t1); }
    if(t<d.span_ns) core.push({t0:t,t1:d.span_ns});
    const busy=u=>ev.filter(e=>e.unit===u).reduce((a,e)=>a+e.t1-e.t0,0);
    return {c, ev, core, red:busy('redmule'), dma:busy('idma'), sync:busy('sync'), nred:ev.filter(e=>e.unit==='redmule').length};
  });
  return {clusters, per};
}
function fmt(ns){return ns>=1e6?(ns/1e6).toFixed(2)+' ms':ns>=1e3?(ns/1e3).toFixed(1)+' us':ns+' ns';}
function draw(){
  const d=DATA[cur]; const dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth, H=TOP+lanes.per.length*LANE_H+26;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+'px';
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const x=t=>LEFT+(t-view.t0)/(view.t1-view.t0)*(W-LEFT-8);
  g.font='11px '+css('--mono'); g.fillStyle=css('--muted'); g.strokeStyle=css('--grid');
  const span=view.t1-view.t0, step=Math.pow(10,Math.floor(Math.log10(span/6))), nice=[1,2,5].map(k=>k*step).find(s=>span/s<=12)||step*10;
  for(let t=Math.ceil(view.t0/nice)*nice;t<=view.t1;t+=nice){const px=x(t);g.beginPath();g.moveTo(px,TOP-4);g.lineTo(px,H-22);g.stroke();g.fillText(fmt(t),px-14,H-8);}
  lanes.per.forEach((L,i)=>{
    const y=TOP+i*LANE_H; g.fillStyle=css('--fg'); g.font='12px '+css('--mono'); g.fillText('cluster '+L.c, 6, y+LANE_H/2+4);
    const bar=(t0,t1,yy,hh,col)=>{const a=Math.max(x(t0),LEFT),b=Math.min(x(t1),W-8); if(b<=a-0.3)return; g.fillStyle=col; g.fillRect(a,yy,Math.max(b-a,0.8),hh);};
    L.ev.filter(e=>e.unit==='sync').forEach(e=>bar(e.t0,e.t1,y+3,LANE_H-6,css('--sync')));
    L.core.forEach(e=>bar(e.t0,e.t1,y+LANE_H-9,5,css('--core')));
    L.ev.filter(e=>e.unit==='idma').forEach(e=>bar(e.t0,e.t1,y+15,9,css('--idma')));
    L.ev.filter(e=>e.unit==='redmule').forEach(e=>bar(e.t0,e.t1,y+4,10,css('--redmule')));
  });
  if(hover){ g.strokeStyle=css('--fg'); g.beginPath(); g.moveTo(hover.x,TOP-4); g.lineTo(hover.x,H-22); g.stroke(); }
}
function load(i){
  cur=i; const d=DATA[i]; lanes=prep(d); view={t0:0,t1:d.span_ns};
  const n=lanes.per.length, red=lanes.per.reduce((a,L)=>a+L.red,0)/(n*d.span_ns), idle=lanes.per.filter(L=>L.sync/d.span_ns>0.5).length;
  const busiest=lanes.per.reduce((a,L)=>L.red>a.red?L:a,lanes.per[0]);
  document.getElementById('stats').innerHTML=[['kernel 窗口',fmt(d.span_ns)],['RedMulE 平均占用',(100*red).toFixed(1)+' %'],['最忙的 cluster','#'+busiest.c+' · '+(100*busiest.red/d.span_ns).toFixed(1)+' %'],['一半以上时间在 barrier 的 cluster',idle+' / '+n],['RedMulE 调用',lanes.per.reduce((a,L)=>a+L.nred,0)]].map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  const tb=document.getElementById('tbl'); const mx=Math.max(...lanes.per.map(L=>L.red))||1;
  tb.innerHTML='<tr><th>cluster</th><th>RedMulE</th><th></th><th>iDMA</th><th>核在执行</th><th>barrier</th><th>RedMulE 调用</th></tr>'+lanes.per.map(L=>{const core=L.core.reduce((a,e)=>a+e.t1-e.t0,0);return `<tr><td>${L.c}</td><td>${(100*L.red/d.span_ns).toFixed(1)} %</td><td style="text-align:left;min-width:90px"><span class="bar" style="width:${Math.round(80*L.red/mx)}px"></span></td><td>${(100*L.dma/d.span_ns).toFixed(1)} %</td><td>${(100*core/d.span_ns).toFixed(1)} %</td><td>${(100*L.sync/d.span_ns).toFixed(1)} %</td><td>${L.nred}</td></tr>`;}).join('');
  document.getElementById('reading').textContent=d.reading||'';
  draw();
}
function pick(mx,my){
  const d=DATA[cur]; const W=cv.clientWidth; const t=view.t0+(mx-LEFT)/(W-LEFT-8)*(view.t1-view.t0);
  const i=Math.floor((my-TOP)/LANE_H); if(i<0||i>=lanes.per.length||mx<LEFT) return null;
  const L=lanes.per[i]; const tol=(view.t1-view.t0)/(W-LEFT)*2;
  const hit=u=>L.ev.find(e=>e.unit===u&&e.t0-tol<=t&&t<=e.t1+tol);
  const e=hit('redmule')||hit('idma')||hit('sync'); if(e) return {L,e,t};
  const c=L.core.find(e=>e.t0<=t&&t<=e.t1); return c?{L,e:{unit:'core',t0:c.t0,t1:c.t1,info:'inferred: not in a barrier'},t}:{L,e:null,t};
}
cv.addEventListener('mousemove',ev=>{const r=cv.getBoundingClientRect(); const p=pick(ev.clientX-r.left,ev.clientY-r.top); hover={x:ev.clientX-r.left};
  if(p&&p.e){tip.style.display='block';tip.style.left=Math.min(ev.clientX-r.left+12,r.width-260)+'px';tip.style.top=(ev.clientY-r.top+12)+'px';
    tip.textContent=`cluster ${p.L.c} · ${p.e.unit}\n${fmt(p.e.t0)} → ${fmt(p.e.t1)}  (${fmt(p.e.t1-p.e.t0)})\n${p.e.info||''}`;} else tip.style.display='none'; draw();});
cv.addEventListener('mouseleave',()=>{tip.style.display='none';hover=null;draw();});
cv.addEventListener('wheel',ev=>{ev.preventDefault(); const r=cv.getBoundingClientRect(); const W=cv.clientWidth; const f=ev.deltaY>0?1.25:0.8;
  const t=view.t0+(ev.clientX-r.left-LEFT)/(W-LEFT-8)*(view.t1-view.t0); const span=Math.max(2000,Math.min(DATA[cur].span_ns,(view.t1-view.t0)*f));
  let t0=t-(t-view.t0)*span/(view.t1-view.t0); t0=Math.max(0,Math.min(DATA[cur].span_ns-span,t0)); view={t0,t1:t0+span}; draw();},{passive:false});
let drag=null; cv.addEventListener('pointerdown',ev=>{drag={x:ev.clientX,v:{...view}};cv.setPointerCapture(ev.pointerId);});
cv.addEventListener('pointermove',ev=>{if(!drag)return; const W=cv.clientWidth; const dt=-(ev.clientX-drag.x)/(W-LEFT-8)*(drag.v.t1-drag.v.t0); const span=drag.v.t1-drag.v.t0;
  let t0=Math.max(0,Math.min(DATA[cur].span_ns-span,drag.v.t0+dt)); view={t0,t1:t0+span}; draw();});
cv.addEventListener('pointerup',()=>drag=null); cv.addEventListener('pointercancel',()=>drag=null);
document.getElementById('reset').onclick=()=>{view={t0:0,t1:DATA[cur].span_ns};draw();};
sel.onchange=()=>load(+sel.value);
window.addEventListener('resize',draw);
if(window.matchMedia){window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);}
load(0);
</script>
"""


def main() -> None:
    out = Path(sys.argv[1])
    items = []
    for spec in sys.argv[2:]:
        label, _, path = spec.partition("=")
        d = json.loads(Path(path).read_text())
        reading = ""
        rp = Path(path).with_suffix(".reading.txt")
        if rp.exists():
            reading = rp.read_text().strip()
        items.append({"label": label, "span_ns": d["span_ns"], "events": d["events"], "reading": reading})
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(items, separators=(",", ":"))))
    print(out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
