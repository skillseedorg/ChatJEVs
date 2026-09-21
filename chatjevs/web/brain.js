/* The jev network drawn as neurons: one node per unit, lit by live stage events.
   Shared by the chat view and the writeup. */
/* ---------- the brain: jev units as neurons, lit by the live stage events ---------- */
const BRAIN_LAYERS = [
  ['plan','plan','#f5b74a'], ['routing','route','#b07cff'],
  ['semantic','sem','#4ade80'], ['syntax','syn','#60a5fa'],
  ['memory','mem','#f472b6'], ['concepts','concept','#22d3ee'],
  ['decoder','decode','#cbd5e1'], ['critics','critic','#fb923c'],
];
const STAGE_LAYER = {
  plan:['plan'], routing:['routing'],
  banks:['semantic','syntax','memory','concepts'],
  form:['decoder'], bucket:['decoder'], word:['decoder'],
  rerank:['decoder'], answer:['decoder'],
  critics:['critics'], judge:['critics'], revise:['critics'],
};

class Brain {
  constructor(canvas, arch){
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.live = true; this.pulses = []; this.layers = [];
    for(const [key,label,color] of BRAIN_LAYERS){
      const n = arch[key] || 0;
      if(!n) continue;
      this.layers.push({key, label, color, count:n, x:0,
        nodes: Array.from({length: Math.min(n, 44)}, () => ({a:0,x:0,y:0}))});
    }
    this.out = {a:0,x:0,y:0};
    this.resize(); this.wire();
    this.tick = this.tick.bind(this);
    this._ro = new ResizeObserver(() => { this.resize(); this.wire(); });
    this._ro.observe(canvas);
    this._running = true;
    requestAnimationFrame(this.tick);
  }
  /* demo mode: replay a version's real firing order on a loop */
  play(seq, step = 430){
    this._seq = seq; this._i = 0;
    const run = () => {
      if(!this.live) return;
      if(this._paused){ this._timer = setTimeout(run, 320); return; }
      const last = this._i % this._seq.length === this._seq.length - 1;
      this.fire(this._seq[this._i % this._seq.length]);
      if(last) this.emit();
      this._i++;
      this._timer = setTimeout(run, last ? step * 2.4 : step);
    };
    run();
  }
  setPaused(p){
    this._paused = p;
    if(!p && !this._running){ this._running = true; requestAnimationFrame(this.tick); }
  }
  resize(){
    const dpr = devicePixelRatio || 1;
    const w = this.c.clientWidth || 700, h = this.c.clientHeight || 184;
    this.c.width = w*dpr; this.c.height = h*dpr;
    this.ctx.setTransform(dpr,0,0,dpr,0,0);
    this.w = w; this.h = h;
    const cols = this.layers.length + 1, pad = 30;
    const span = (w - pad*2) / Math.max(cols-1, 1);
    this.layers.forEach((L,i) => {
      L.x = pad + i*span;
      const rows = Math.min(L.nodes.length, 11);
      const ncols = Math.ceil(L.nodes.length / rows);
      L.nodes.forEach((nd,j) => {
        const col = Math.floor(j/rows), row = j % rows;
        const inCol = Math.min(rows, L.nodes.length - col*rows);
        nd.x = L.x + (col - (ncols-1)/2) * 8.5;
        nd.y = (h-24)/2 + (row - (inCol-1)/2) * (96/Math.max(rows-1,1));
      });
    });
    this.out.x = pad + (cols-1)*span; this.out.y = (h-24)/2;
  }
  wire(){
    this.edges = [];
    for(let i=0;i<this.layers.length-1;i++){
      const A = this.layers[i].nodes, B = this.layers[i+1].nodes;
      A.forEach((a,ai) => { for(let k=0;k<2;k++)
        this.edges.push({a, b:B[(ai*3 + k*7) % B.length]}); });
    }
    const last = this.layers[this.layers.length-1];
    if(last) last.nodes.forEach(a => this.edges.push({a, b:this.out}));
  }
  fire(stage){
    const keys = STAGE_LAYER[stage] || (stage.startsWith('draft') ? ['decoder'] : []);
    for(const key of keys){
      const L = this.layers.find(l => l.key === key);
      if(!L) continue;
      L.nodes.forEach((nd,i) => setTimeout(() => { nd.a = 1; }, i*8));
      let n = 0;
      for(const e of this.edges){
        if(L.nodes.includes(e.b) && n++ < 16) this.pulses.push({e, t:0});
      }
    }
  }
  emit(){
    this.out.a = 1;
    let n = 0;
    for(const e of this.edges) if(e.b === this.out && n++ < 10) this.pulses.push({e, t:0});
  }
  tick(){
    const ctx = this.ctx;
    ctx.clearRect(0,0,this.w,this.h);
    ctx.lineWidth = 0.6;
    for(const e of this.edges){
      const act = Math.max(e.a.a, e.b.a);
      if(act < 0.02 && Math.random() > 0.35) continue;
      ctx.strokeStyle = `rgba(125,155,200,${0.045 + act*0.33})`;
      ctx.beginPath(); ctx.moveTo(e.a.x,e.a.y); ctx.lineTo(e.b.x,e.b.y); ctx.stroke();
    }
    this.pulses = this.pulses.filter(p => p.t < 1);
    for(const p of this.pulses){
      p.t += 0.05;
      const x = p.e.a.x + (p.e.b.x-p.e.a.x)*p.t, y = p.e.a.y + (p.e.b.y-p.e.a.y)*p.t;
      ctx.fillStyle = `rgba(190,232,255,${1-p.t})`;
      ctx.beginPath(); ctx.arc(x,y,1.8,0,7); ctx.fill();
    }
    ctx.textAlign = 'center'; ctx.font = '9px ui-monospace,monospace';
    for(const L of this.layers){
      for(const nd of L.nodes){
        nd.a *= 0.955;
        ctx.beginPath(); ctx.arc(nd.x, nd.y, 1.7 + nd.a*2.7, 0, 7);
        if(nd.a > 0.05){
          ctx.globalAlpha = 0.35 + nd.a*0.65; ctx.fillStyle = L.color;
          ctx.shadowBlur = nd.a*10; ctx.shadowColor = L.color;
        } else { ctx.globalAlpha = 1; ctx.fillStyle = '#2b3140'; }
        ctx.fill(); ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      }
      ctx.fillStyle = 'rgba(150,162,180,.55)';
      ctx.fillText(`${L.label} ${L.count}`, L.x, this.h - 8);
    }
    this.out.a *= 0.94;
    ctx.beginPath(); ctx.arc(this.out.x, this.out.y, 3 + this.out.a*4.5, 0, 7);
    ctx.fillStyle = this.out.a > 0.05 ? '#fff' : '#39415a';
    ctx.shadowBlur = this.out.a*18; ctx.shadowColor = '#9fe8ff';
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(150,162,180,.55)';
    ctx.fillText('word', this.out.x, this.h - 8);
    if(this._paused){ this._running = false; return; }
    if(this.live) requestAnimationFrame(this.tick);
    else this._running = false;
  }
  stop(){ setTimeout(() => { this.live = false; this._ro.disconnect(); }, 1600); }
}
