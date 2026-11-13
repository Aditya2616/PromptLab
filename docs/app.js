// PromptLab viewer. Replays recorded runs (docs/data) or a live run streamed by the local server.
// Every number on the page is read from the run files and summary.json; none is typed in here.
const $ = (s) => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const S = { run: null, nodes: [], byId: {}, G: 0, maxStep: 0, step: 0, sel: null, playing: false, timer: null,
            speed: 1, inView: true, lineage: new Set(), anc: new Set(), els: {}, live: false, layout: {},
            index: [], summary: null, task: null, seed: null };

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (x, d = 1) => (x * 100).toFixed(d);
const pts = (x) => `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(1)}`;
const fmtP = (p) => (p < 0.001 ? '< 0.001' : `= ${p.toFixed(3)}`);
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const isPhone = () => matchMedia('(max-width: 960px)').matches;
const LABEL = { seed0: 'Seed prompt', best_seed: 'Best seed', random: 'Random-search winner', evolved: 'Evolved winner', evolved_alt: 'Evolved, 1.5B mutator' };

/* ---------- data ---------- */
async function init() {
  bindControls();
  try {
    const idx = await (await fetch('data/index.json')).json();
    S.index = idx.runs;
    if (idx.summary) S.summary = await (await fetch(`data/${idx.summary}`)).json();
    const want = new URLSearchParams(location.search).get('run');
    const first = S.index.find((r) => r.file === `${want}.json`) || S.index[0];
    if (first) await selectRun(first.name, first.seed);
  } catch (e) {
    $('#caption').textContent = 'Could not load recorded runs (data/index.json).';
  }
  detectLive();
}

function buildPicker() {
  const tabs = $('#runTabs'), seeds = $('#seedTabs');
  tabs.replaceChildren(); seeds.replaceChildren();
  const names = [...new Set(S.index.map((r) => r.name))];
  names.forEach((n) => {
    const r = S.index.find((x) => x.name === n);
    const b = document.createElement('button');
    b.className = 'tab'; b.setAttribute('role', 'tab'); b.textContent = r.task; b.dataset.name = n;
    b.setAttribute('aria-selected', n === S.task);
    b.onclick = () => selectRun(n, (S.index.find((x) => x.name === n && x.seed === S.seed) || S.index.find((x) => x.name === n)).seed);
    tabs.appendChild(b);
  });
  S.index.filter((r) => r.name === S.task).forEach((r) => {
    const b = document.createElement('button');
    b.className = 'tab seed'; b.setAttribute('role', 'tab'); b.textContent = `seed ${r.seed}`; b.dataset.seed = r.seed;
    b.setAttribute('aria-selected', r.seed === S.seed);
    b.onclick = () => selectRun(S.task, r.seed);
    seeds.appendChild(b);
  });
}

async function selectRun(name, seed) {
  stopLive();
  S.task = name; S.seed = seed;
  buildPicker();
  const meta = S.index.find((r) => r.name === name && r.seed === seed);
  const run = await (await fetch(`data/${meta.file}`)).json();
  loadRun(run, { fresh: true });
  renderSeeds(); renderOps();
}

function derive(run) {
  S.run = run;
  S.nodes = run.nodes;
  S.byId = Object.fromEntries(run.nodes.map((n) => [n.id, n]));
  S.G = Math.max(0, ...run.generations.map((g) => g.gen));
  const done = !!run.report;
  S.maxStep = 2 * S.G + 1 + (done ? 1 : 0);
  S.winner = run.winner || (() => {
    const last = run.generations[run.generations.length - 1];
    if (!last) return null;
    return last.survivors.slice().sort((a, b) => S.byId[b].dev_acc - S.byId[a].dev_acc)[0];
  })();
  S.lineage = ancestry(S.winner);
}
function ancestry(id) {
  const out = new Set();
  const walk = (x) => { if (!x || out.has(x)) return; out.add(x); (S.byId[x]?.parents || []).forEach(walk); };
  walk(id);
  return out;
}

function loadRun(run, { fresh = false } = {}) {
  derive(run);
  const t = run.task, g = run.git || {};
  $('#runMeta').textContent = (run.recorded === false || S.live)
    ? `LIVE RUN · ${run.model} · backend ${run.backend} · seed ${run.seed}`
    : `RECORDED RUN · ${run.model}${run.device ? ` (${run.device} ${run.dtype})` : ''} · seed ${run.seed} · commit ${g.commit || 'n/a'}${g.dirty ? '+dirty' : ''} · ` +
      `${run.config.generations} generations, population ${run.config.pop} · ${t.n_dev} dev / ${t.n_test} test items · ` +
      `${(run.calls?.model_calls ?? 0).toLocaleString('en-GB')} model calls · ${Math.round((run.seconds || 0) / 60)} min · data: ${t.source}`;
  S.sel = fresh ? null : S.sel; S.anc = ancestry(S.sel);
  buildGraph();
  renderCurve();
  renderHeldout();
  renderResults();
  $('#scrub').max = S.maxStep;
  if (fresh) {
    renderDetail(null);
    stopPlay();
    if (reduce || S.live) setStep(S.maxStep); else { setStep(0); startPlay(); }
  } else {
    setStep(S.maxStep);
  }
}

/* ---------- graph ---------- */
const PAD = { l: 46, r: 40, t: 34, b: 22 };
function buildGraph() {
  const svg = $('#graph'); svg.replaceChildren();
  const box = $('#graphScroll');
  const H = isPhone() ? 340 : 390;
  const colW = Math.max(isPhone() ? 112 : 108, Math.floor((box.clientWidth - PAD.l - PAD.r) / Math.max(1, S.G)));
  const W = PAD.l + S.G * colW + PAD.r;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const accs = S.nodes.map((n) => n.dev_acc), sorted = accs.slice().sort((a, b) => a - b);
  let lo = Math.max(0, sorted[0] - 0.03), hi = Math.min(1, sorted[sorted.length - 1] + 0.04);
  if (hi - lo < 0.2) { const m = (hi + lo) / 2; lo = Math.max(0, m - 0.1); hi = Math.min(1, m + 0.1); }
  // a few very weak variants would squash everything else: give them a compressed strip below an axis break
  const q = sorted[Math.floor(sorted.length * 0.12)];
  const split = Math.floor((q - 0.02) * 20) / 20;
  const tail = split - lo > 0.12 && (hi - split) < 0.7 * (hi - lo);
  const ph = H - PAD.t - PAD.b, topH = tail ? ph * 0.8 : ph, gap = 14;
  const Y = (a) => (!tail || a >= split)
    ? PAD.t + (1 - (a - (tail ? split : lo)) / (hi - (tail ? split : lo))) * topH
    : PAD.t + topH + gap + ((split - a) / (split - lo)) * (ph - topH - gap);
  const X = (g) => PAD.l + g * colW;
  const base = tail ? split : lo, step = hi - base > 0.5 ? 0.2 : hi - base > 0.2 ? 0.1 : 0.05;
  for (let v = Math.ceil(base / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    el('line', { class: 'grid-line', x1: PAD.l - 6, x2: W - PAD.r + 14, y1: Y(v), y2: Y(v) }, svg);
    el('text', { class: 'grid-label', x: 2, y: Y(v) + 3 }, svg).textContent = pct(v, 0) + '%';
  }
  if (tail) {
    const yb = PAD.t + topH + gap / 2;
    el('line', { class: 'break-line', x1: PAD.l - 6, x2: W - PAD.r + 14, y1: yb, y2: yb }, svg);
    el('text', { class: 'grid-label', x: 2, y: yb + 11 }, svg).textContent = `<${pct(split, 0)}%`;
    el('text', { class: 'grid-label', x: 2, y: H - PAD.b + 12 }, svg).textContent = 'scale compressed';
  }
  const pos = {}, labelFor = new Set();
  const offs = [0, 22, -22, 44, -44, 66, -66];
  for (let g = 0; g <= S.G; g++) {
    el('line', { class: 'col-rule', x1: X(g), x2: X(g), y1: 20, y2: H - PAD.b + 6 }, svg);
    el('text', { class: 'col-label', x: X(g), y: 12, 'text-anchor': 'middle' }, svg).textContent = g === 0 ? 'GEN 0 · SEEDS' : `GEN ${g}`;
    const col = S.nodes.filter((n) => n.gen === g).sort((a, b) => b.dev_acc - a.dev_acc);
    if (col.length) labelFor.add(col[0].id);
    const lastY = offs.map(() => -99);
    for (const n of col) {
      const y = Y(n.dev_acc);
      let lane = offs.findIndex((_, i) => y - lastY[i] >= 20);
      if (lane < 0) lane = 0;
      lastY[lane] = y;
      pos[n.id] = [X(g) + offs[lane], y];
    }
  }
  S.layout = { pos, W, H, colW };
  const gE = el('g', {}, svg), gN = el('g', {}, svg), els = { nodes: {}, edges: [], labels: {} };
  for (const n of S.nodes) {
    for (const p of n.parents) {
      if (!pos[p]) continue;
      const [x1, y1] = pos[p], [x2, y2] = pos[n.id], xm = (x1 + x2) / 2;
      const path = el('path', { class: 'edge', d: `M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}`, pathLength: 1 }, gE);
      els.edges.push({ path, child: n, parent: S.byId[p] });
    }
  }
  for (const n of S.nodes) {
    const [x, y] = pos[n.id];
    const g = el('g', { class: 'node', transform: `translate(${x},${y})`, tabindex: 0, role: 'button',
      'aria-label': `Variant ${n.id}, generation ${n.gen}, dev accuracy ${pct(n.dev_acc)} percent` }, gN);
    el('circle', { class: 'halo', r: 15 }, g);
    el('circle', { class: 'dot', r: 6.5 }, g);
    el('circle', { class: 'ring', r: 12 }, g);
    el('circle', { class: 'hit', r: 22, fill: 'transparent' }, g);
    g.addEventListener('click', () => selectNode(n.id, true));
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectNode(n.id, true); } });
    els.nodes[n.id] = g;
    if (labelFor.has(n.id)) {
      const t = el('text', { class: 'node-label', x, y: y - 15, 'text-anchor': 'middle' }, svg);
      t.textContent = pct(n.dev_acc);
      els.labels[n.id] = t;
    }
  }
  S.els = els;
  applyStep(S.step);
  edgeFades();
}

function edgeFades() {
  const box = $('#graphScroll'), wrap = $('#graphWrap');
  const max = box.scrollWidth - box.clientWidth;
  wrap.classList.toggle('fade-l', box.scrollLeft > 4);
  wrap.classList.toggle('fade-r', max > 4 && box.scrollLeft < max - 4);
}

function applyStep(s) {
  S.step = s;
  const final = !!S.run.report && s >= S.maxStep;
  const state = {};
  for (const n of S.nodes) {
    const vis = 2 * n.gen <= s;
    const culled = vis && n.culled_at != null && s >= 2 * n.culled_at + 1;
    const keep = vis && !culled && s >= 2 * n.gen + 1;
    state[n.id] = { vis, culled, keep };
    const g = S.els.nodes[n.id];
    g.classList.toggle('on', vis);
    g.classList.toggle('keep', keep);
    g.classList.toggle('cull', culled);
    g.classList.toggle('best', (final || (!S.run.report && s >= S.maxStep)) && n.id === S.winner);
    g.classList.toggle('sel', n.id === S.sel);
    S.els.labels[n.id]?.classList.toggle('on', vis && !culled);
  }
  for (const e of S.els.edges) {
    const c = state[e.child.id];
    e.path.classList.toggle('on', c.vis);
    e.path.classList.toggle('keep', c.keep);
    e.path.classList.toggle('cull', c.culled);
    e.path.classList.toggle('win', final && S.lineage.has(e.child.id) && S.lineage.has(e.parent.id));
    e.path.classList.toggle('anc', !!S.sel && S.anc.has(e.child.id) && S.anc.has(e.parent.id));
  }
  const g = Math.min(S.G, Math.floor(s / 2));
  const gen = S.run.generations.find((x) => x.gen === g);
  let cap;
  if (s > 2 * S.G + 1) cap = 'HELD-OUT TEST · winners scored once on data the search never saw';
  else if (s % 2 === 0) cap = `GEN ${g} · ${g === 0 ? 'seed prompts' : 'children'} scored on dev${gen ? ` (${gen.born.length} variants)` : ''}`;
  else cap = `GEN ${g} · select: ${gen ? gen.survivors.length : '?'} survive · best dev ${gen ? pct(gen.best_dev) : '?'}%`;
  $('#caption').textContent = cap;
  $('#genInd').textContent = `generation ${g} of ${S.G}`;
  $('#scrub').value = s;
  $('#scrub').setAttribute('aria-valuetext', cap);
  updateCurve(s);
  if (!reduce) followColumn(g);
}

function followColumn(g) {
  const box = $('#graphScroll');
  if (box.scrollWidth <= box.clientWidth + 2) return;
  const x = PAD.l + g * S.layout.colW;
  box.scrollTo({ left: Math.max(0, x - box.clientWidth / 2), behavior: 'smooth' });
}

/* ---------- playback ---------- */
function setStep(s) {
  applyStep(Math.max(0, Math.min(S.maxStep, s)));
  if (S.step >= S.maxStep && !S.sel && S.winner && !isPhone()) selectNode(S.winner, false);
}
function startPlay() {
  if (S.step >= S.maxStep) S.step = 0;
  S.playing = true; $('#playIcon').setAttribute('d', 'M7 5h4v14H7zM13 5h4v14h-4z'); $('#btnPlay').setAttribute('aria-label', 'Pause');
  const tick = () => {
    if (!S.playing) return;
    if (!S.inView) { S.timer = setTimeout(tick, 400); return; }
    if (S.step >= S.maxStep) { stopPlay(); return; }
    setStep(S.step + 1);
    S.timer = setTimeout(tick, 1000 / S.speed);
  };
  clearTimeout(S.timer);
  S.timer = setTimeout(tick, S.step === 0 ? 700 : 100);
}
function stopPlay() {
  S.playing = false; clearTimeout(S.timer);
  $('#playIcon').setAttribute('d', 'M8 5v14l11-7z'); $('#btnPlay').setAttribute('aria-label', 'Play');
}
function bindControls() {
  $('#btnPlay').onclick = () => (S.playing ? stopPlay() : startPlay());
  $('#btnPrev').onclick = () => { stopPlay(); setStep(S.step - 1); };
  $('#btnNext').onclick = () => { stopPlay(); setStep(S.step + 1); };
  $('#scrub').oninput = (e) => { stopPlay(); setStep(+e.target.value); };
  $('#btnSpeed').onclick = (e) => { S.speed = S.speed === 1 ? 2 : S.speed === 2 ? 4 : S.speed === 4 ? 0.5 : 1; e.target.textContent = `${S.speed}x`; };
  $('#btnClose').onclick = closeSheet; $('#scrim').onclick = closeSheet;
  $('#graphScroll').addEventListener('scroll', edgeFades, { passive: true });
  new IntersectionObserver(([e]) => { S.inView = e.isIntersecting; }, { threshold: 0.15 }).observe($('#graphScroll'));
  let rt, lastW = innerWidth;
  addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      if (!S.run || innerWidth === lastW) return;  // ignore mobile URL-bar height changes
      lastW = innerWidth; buildGraph(); renderCurve(); renderHeldout(); updateCurve(S.step);
    }, 150);
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });
}

/* ---------- detail ---------- */
function openSheet() { if (isPhone()) { $('#detail').classList.add('open'); $('#scrim').classList.add('open'); } }
function closeSheet() { $('#detail').classList.remove('open'); $('#scrim').classList.remove('open'); }
function selectNode(id, open) {
  S.sel = id; S.anc = ancestry(id); applyStep(S.step); renderDetail(id);
  if (open) openSheet();
}

function lcsDiff(a, b) {
  const A = a.split(/(\s+)/).filter(Boolean), B = b.split(/(\s+)/).filter(Boolean);
  const n = A.length, m = B.length, dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  let i = 0, j = 0, out = '';
  while (i < n && j < m) {
    if (A[i] === B[j]) { out += esc(B[j]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out += /^\s+$/.test(A[i]) ? '' : `<del>${esc(A[i])}</del>`; i++; }
    else { out += /^\s+$/.test(B[j]) ? esc(B[j]) : `<ins>${esc(B[j])}</ins>`; j++; }
  }
  while (i < n) { out += /^\s+$/.test(A[i]) ? '' : `<del>${esc(A[i])}</del>`; i++; }
  while (j < m) { out += /^\s+$/.test(B[j]) ? esc(B[j]) : `<ins>${esc(B[j])}</ins>`; j++; }
  return out;
}

function renderDetail(id) {
  const body = $('#detailBody');
  if (!id) {
    body.innerHTML = `<h3>Pick a variant</h3><p class="empty">Tap any node in the graph to see its prompt, diffed against its parent, with its dev score and what the mutation changed.</p>
      ${S.winner ? '<button class="linkbtn" id="goWin">Show the winner</button>' : ''}`;
    $('#goWin')?.addEventListener('click', () => selectNode(S.winner, true));
    return;
  }
  const n = S.byId[id], par = n.parents.map((p) => S.byId[p]).filter(Boolean);
  const ci = n.dev_ci.map((v) => pct(v, 0)).join('–');
  const chips = [`<span class="chip">gen ${n.gen}</span>`, `<span class="chip">${esc(n.operator)}</span>`];
  if (id === S.winner) chips.push('<span class="chip lime">winner</span>');
  if (n.culled_at != null) chips.push(`<span class="chip ember">culled gen ${n.culled_at}</span>`);
  else chips.push('<span class="chip lime">survivor</span>');
  let delta = '';
  if (par[0]) {
    let fixed = 0, broke = 0;
    for (let i = 0; i < n.dev_bits.length; i++) {
      if (n.dev_bits[i] === '1' && par[0].dev_bits[i] === '0') fixed++;
      if (n.dev_bits[i] === '0' && par[0].dev_bits[i] === '1') broke++;
    }
    delta = `<dt>vs parent</dt><dd>${pts(n.dev_acc - par[0].dev_acc)} pts (fixed ${fixed}, broke ${broke})</dd>`;
  }
  let held = '';
  const arms = S.run.report?.arms || {};
  for (const a of Object.values(arms)) if (a.id === id) held += `<dt>held-out as ${esc(a.label)}</dt><dd>${pct(a.test_acc)}% [${pct(a.test_ci[0], 0)}–${pct(a.test_ci[1], 0)}]</dd>`;
  const text = par[0] ? lcsDiff(par[0].text, n.text) : esc(n.text);
  const parBtns = par.map((p) => `<button class="linkbtn" data-id="${p.id}">${esc(p.id)} (${pct(p.dev_acc)}%)</button>`).join(' ');
  body.innerHTML = `<h3>variant ${esc(n.id)}</h3><div class="chips">${chips.join('')}</div>
    <dl class="kv"><dt>dev accuracy</dt><dd>${pct(n.dev_acc)}% [${ci}]</dd>${delta}${held}
    <dt>few-shot examples</dt><dd>${n.n_shots}</dd></dl>
    <p class="sub">${par[0] ? `Prompt, diffed against parent${par.length > 1 ? 's (first shown)' : ''}` : 'Seed prompt'}</p>
    <pre class="prompt">${text}</pre>
    ${par.length ? `<p class="sub">Parents</p>${parBtns}` : ''}`;
  body.querySelectorAll('.linkbtn[data-id]').forEach((b) => (b.onclick = () => selectNode(b.dataset.id, true)));
}

/* ---------- score curve ---------- */
const CV = { W: 760, H: 250, l: 46, r: 16, t: 16, b: 30 };
function renderCurve() {
  const svg = $('#curve'); svg.replaceChildren();
  const w = svg.parentElement.clientWidth - 32;
  CV.W = Math.max(300, w || 760); CV.H = CV.W < 560 ? 220 : 250;
  svg.setAttribute('viewBox', `0 0 ${CV.W} ${CV.H}`);
  const gens = S.run.generations;
  const vals = gens.flatMap((g) => [g.best_dev, g.mean_dev]);
  let lo = Math.max(0, Math.min(...vals) - 0.04), hi = Math.min(1, Math.max(...vals) + 0.04);
  if (hi - lo < 0.2) hi = lo + 0.2;
  const pw = CV.W - CV.l - CV.r, ph = CV.H - CV.t - CV.b;
  const X = (g) => CV.l + (S.G ? (g / S.G) * pw : 0), Y = (a) => CV.t + (1 - (a - lo) / (hi - lo)) * ph;
  const step = hi - lo > 0.5 ? 0.2 : 0.1;
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    el('line', { class: 'grid-line', x1: CV.l, x2: CV.W - CV.r, y1: Y(v), y2: Y(v) }, svg);
    el('text', { class: 'grid-label', x: 4, y: Y(v) + 3 }, svg).textContent = pct(v, 0) + '%';
  }
  const every = CV.W < 560 ? 2 : 1;
  gens.forEach((g) => { if (g.gen % every === 0) el('text', { class: 'grid-label', x: X(g.gen), y: CV.H - 10, 'text-anchor': 'middle' }, svg).textContent = `gen ${g.gen}`; });
  const line = (key, cls) => el('path', { class: cls, pathLength: 1, d: gens.map((g, i) => `${i ? 'L' : 'M'}${X(g.gen)},${Y(g[key])}`).join('') }, svg);
  S.cv = { mean: line('mean_dev', 'curve-mean'), best: line('best_dev', 'curve-line'), pts: [] };
  gens.forEach((g) => S.cv.pts.push(el('circle', { class: 'curve-pt', cx: X(g.gen), cy: Y(g.best_dev), r: 4.5 }, svg)));
}
function updateCurve(s) {
  if (!S.cv) return;
  const gens = S.run.generations, v = gens.filter((g) => 2 * g.gen + 1 <= s).length;
  const off = v <= 1 ? 1 : 1 - (v - 1) / Math.max(1, S.G);
  S.cv.best.style.strokeDashoffset = off; S.cv.mean.style.strokeDashoffset = off;
  S.cv.pts.forEach((p, i) => (p.style.opacity = i < v ? 1 : 0));
  const final = !!S.run.report && s >= S.maxStep;
  $('#heldout').classList.toggle('shown', final);
}

/* ---------- held-out markers ---------- */
function renderHeldout() {
  const svg = $('#heldout'); svg.replaceChildren();
  const rep = S.run.report;
  if (!rep) { $('#heldSub').textContent = 'appears when the run finishes'; svg.setAttribute('viewBox', '0 0 10 1'); return; }
  const arms = rep.arms;
  const keys = ['seed0', 'random', 'evolved', 'evolved_alt'].filter((k) => arms[k]);
  const W = Math.max(300, $('#curvePanel').clientWidth - 32), narrow = W < 560;
  const rowH = narrow ? 58 : 46, H = 24 + keys.length * rowH + 26;
  const px0 = narrow ? 8 : 200, px1 = W - 14;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const vals = keys.flatMap((k) => [arms[k].test_ci[0], arms[k].test_ci[1], arms[k].dev_acc]);
  if (rep.majority_rate != null) vals.push(rep.majority_rate);
  const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.03) * 20) / 20);
  let hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.03) * 20) / 20);
  if (hi - lo < 0.2) hi = lo + 0.2;
  const X = (a) => px0 + ((a - lo) / (hi - lo)) * (px1 - px0);
  const step = hi - lo > 0.5 ? 0.2 : 0.1;
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    el('line', { class: 'grid-line', x1: X(v), x2: X(v), y1: 14, y2: H - 22 }, svg);
    el('text', { class: 'grid-label', x: X(v), y: H - 6, 'text-anchor': 'middle' }, svg).textContent = pct(v, 0) + '%';
  }
  if (rep.majority_rate != null) {
    el('line', { class: 'maj-line', x1: X(rep.majority_rate), x2: X(rep.majority_rate), y1: 14, y2: H - 22 }, svg);
    el('text', { class: 'grid-label maj', x: X(rep.majority_rate) + 4, y: 11 }, svg).textContent = 'majority answer';
  }
  keys.forEach((k, i) => {
    const a = arms[k], y = 24 + i * rowH + (narrow ? 38 : rowH / 2 - 2), hl = k === 'evolved' || k === 'evolved_alt';
    const g = el('g', { class: `held-row ${hl ? 'evo' : ''}` }, svg);
    const ly = narrow ? 24 + i * rowH + 10 : y - 2;
    el('text', { class: 'held-label', x: 0, y: ly }, g).textContent = LABEL[k];
    el('text', { class: 'held-val', x: 0, y: ly + 15 }, g).textContent = `${pct(a.test_acc)}%  [${pct(a.test_ci[0])}, ${pct(a.test_ci[1])}]`;
    el('line', { class: 'whisk', x1: X(a.test_ci[0]), x2: X(a.test_ci[1]), y1: y, y2: y }, g);
    el('line', { class: 'whisk', x1: X(a.test_ci[0]), x2: X(a.test_ci[0]), y1: y - 5, y2: y + 5 }, g);
    el('line', { class: 'whisk', x1: X(a.test_ci[1]), x2: X(a.test_ci[1]), y1: y - 5, y2: y + 5 }, g);
    el('circle', { class: 'held-dot', cx: X(a.test_acc), cy: y, r: 6 }, g);
    const d = X(a.dev_acc);
    el('path', { class: 'dev-mark', d: `M${d},${y - 7}l6,7l-6,7l-6,-7z` }, g);
  });
  $('#heldSub').textContent = `n = ${rep.n_test} test items${rep.majority_rate != null ? `, majority answer ${pct(rep.majority_rate)}%` : ''}. The gap between a diamond and its dot is the winner's curse.`;
}

/* ---------- tables ---------- */
const ARM_LABEL = { seed0: 'First-draft seed prompt', best_seed: 'Best seed (picked on dev)', random: 'Random search, same budget', evolved: 'Evolved winner', evolved_alt: 'Evolved winner, 1.5B mutator' };
const ARM_KEYS = ['seed0', 'best_seed', 'random', 'evolved', 'evolved_alt'];
function renderResults() {
  const rep = S.run.report, tbl = $('#armsTable'), ver = $('#verdicts');
  if (!rep) { tbl.innerHTML = ''; ver.innerHTML = '<p class="verdict mono">Held-out numbers appear when the run finishes.</p>'; $('#resultsSub').textContent = ''; return; }
  $('#resultsSub').textContent = `seed ${S.run.seed} · n = ${rep.n_test} test items, never seen by the search`;
  const rows = ARM_KEYS.filter((k) => rep.arms[k]).map((k) => {
    const a = rep.arms[k];
    return `<tr class="${k === 'evolved' ? 'hl' : ''}"><td>${ARM_LABEL[k]}</td><td>${a.id}</td><td>${pct(a.dev_acc)}%</td><td>${pct(a.test_acc)}%</td>
      <td class="ci">${pct(a.test_ci[0])}–${pct(a.test_ci[1])}</td><td>${pts(-a.gap)}</td></tr>`;
  }).join('');
  tbl.innerHTML = `<thead><tr><th>Prompt</th><th>id</th><th>dev</th><th>test</th><th>95% CI</th><th>test − dev (pts)</th></tr></thead><tbody>${rows}</tbody>`;
  const names = { evolved_vs_seed0: 'first-draft seed', evolved_vs_best_seed: 'best seed', evolved_vs_random: 'random search' };
  ver.innerHTML = Object.entries(rep.comparisons).filter(([k]) => names[k]).map(([k, c]) => {
    const good = c.p_mcnemar < 0.05 && c.diff > 0;
    const text = `Evolved vs ${names[k]}: ${pts(c.diff)} pts (95% CI ${pts(c.lo)} to ${pts(c.hi)}), exact McNemar p ${fmtP(c.p_mcnemar)} ` +
      `(${c.a_only} better, ${c.b_only} worse). ${good ? 'Significant at 0.05.' : c.diff <= 0 ? 'No gain.' : 'Not significant at 0.05.'}`;
    return `<p class="verdict ${good ? 'ok' : 'warn'}">${text}</p>`;
  }).join('');
}

function renderSeeds() {
  const T = S.summary?.tasks?.[S.task], tbl = $('#seedsTable'), ver = $('#seedVerdicts');
  if (!T) { tbl.innerHTML = ''; ver.innerHTML = ''; $('#seedsSub').textContent = ''; return; }
  $('#seedsSub').textContent = `${T.title} · ${T.seeds.length} seeds (${T.seeds.join(', ')}) · ${T.config.generations} generations, population ${T.config.pop} · ${T.n_dev} dev / ${T.n_test} test items · ${T.device} ${T.dtype}`;
  const rows = ARM_KEYS.filter((k) => T.arms[k]).map((k) => {
    const a = T.arms[k];
    return `<tr class="${k === 'evolved' ? 'hl' : ''}"><td>${ARM_LABEL[k]}</td><td>${a.test.map((x) => pct(x)).join(' / ')}</td>
      <td>${pct(a.test_mean)} ± ${pct(a.test_sd)}</td><td class="ci">${pct(a.pooled_ci[0])}–${pct(a.pooled_ci[1])}</td><td>${pts(-a.gap_mean)}</td></tr>`;
  }).join('');
  tbl.innerHTML = `<thead><tr><th>Prompt</th><th>test % per seed</th><th>mean ± SD</th><th>95% CI, items</th><th>test − dev (pts)</th></tr></thead><tbody>${rows}</tbody>`;
  const names = { evolved_vs_random: 'random search', evolved_vs_best_seed: 'best seed', evolved_vs_seed0: 'first-draft seed' };
  ver.innerHTML = Object.entries(names).filter(([k]) => T.comparisons[k]).map(([k, nm]) => {
    const c = T.comparisons[k], good = c.lo > 0;
    return `<p class="verdict ${good ? 'ok' : 'warn'}">Evolved minus ${nm}: ${pts(c.diff)} pts (95% CI ${pts(c.lo)} to ${pts(c.hi)}), ahead in ${c.seeds_ahead} of ${T.seeds.length} seeds; ` +
      `per seed ${c.seed_diffs.map(pts).join(' / ')}; exact McNemar p ${c.per_seed.map((p) => (p.p_mcnemar < 0.001 ? '<0.001' : p.p_mcnemar.toFixed(3))).join(' / ')}.</p>`;
  }).join('');
}

function renderOps() {
  const T = S.summary?.tasks?.[S.task], tbl = $('#opsTable');
  if (!T || !T.operators.length) { tbl.innerHTML = '<tbody><tr><td class="muted">Available when the run finishes.</td></tr></tbody>'; $('#opsSub').textContent = ''; return; }
  $('#opsSub').textContent = `${T.title}, ${T.seeds.length} seeds pooled. The share of children that beat their best parent on dev is the bar. Held-out columns use ${T.operator_n_test} test items and only the children that beat their parent on dev.`;
  const gain = (x, k) => (k ? pts(x) : '–');
  const row = (r, tag = '') => `<tr><td>${esc(r.operator)}${tag}</td><td>${r.children}</td>
    <td class="barcell"><span class="bar"><i style="width:${(100 * r.beat_dev_rate).toFixed(0)}%"></i></span><span>${r.beat_dev} (${pct(r.beat_dev_rate, 0)}%)</span></td>
    <td>${r.beat_dev ? `${r.beat_test} of ${r.beat_dev}` : '–'}</td><td>${gain(r.mean_dev_gain, r.beat_dev)}</td><td>${gain(r.mean_test_gain, r.beat_dev)}</td></tr>`;
  let body = T.operators.map((r) => row(r)).join('');
  const alt = (T.operators_alt || []).filter((r) => r.operator === 'llm_rewrite' || r.operator === 'llm_reflect');
  if (alt.length) body += `<tr class="sep"><td colspan="6">Same LLM operators with the 1.5B model as mutator (separate arm)</td></tr>` + alt.map((r) => row(r, ' (1.5B)')).join('');
  tbl.innerHTML = `<thead><tr><th>Operator</th><th>children</th><th>beat parent on dev</th><th>held-out: still better</th><th>dev gain (pts)</th><th>test gain (pts)</th></tr></thead><tbody>${body}</tbody>`;
}

/* ---------- live ---------- */
async function detectLive() {
  // the local server tags every response; a static host does not, so no request ever 404s
  try {
    const r = await fetch(location.pathname, { method: 'HEAD' });
    if (r.headers.get('x-promptlab') !== 'live') return;
  } catch { return; }
  $('#live').hidden = false;
  $('#liveForm').addEventListener('submit', startLive);
}
function stopLive() { S.es?.close(); S.es = null; S.live = false; $('#modePill').textContent = 'Recorded run'; $('#modePill').classList.remove('live'); }
async function startLive(e) {
  e.preventDefault();
  const st = $('#liveStatus'); st.textContent = 'Starting...';
  const body = { task: $('#liveTask').value, backend: $('#liveBackend').value, seed: +$('#liveSeed').value };
  const r = await fetch('/api/live', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) { st.textContent = r.status === 409 ? 'A live run is already in progress.' : `Could not start (${r.status}).`; return; }
  const { id } = await r.json();
  stopLive(); S.live = true; stopPlay();
  $('#modePill').textContent = 'Live run'; $('#modePill').classList.add('live');
  document.querySelectorAll('#runTabs .tab, #seedTabs .tab').forEach((t) => t.setAttribute('aria-selected', 'false'));
  S.es = new EventSource(`/api/live/${id}/stream`);
  let first = true;
  S.es.onmessage = (m) => {
    const state = JSON.parse(m.data);
    st.textContent = `Generation ${state.generations?.length ? state.generations[state.generations.length - 1].gen : 0} done...`;
    loadRun(state, { fresh: first }); first = false;
    $('#lab').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  };
  S.es.addEventListener('end', (m) => {
    const j = JSON.parse(m.data); st.textContent = j.status === 'done' ? 'Done. This run was produced just now by the local server.' : `Error: ${j.error}`;
    S.es.close();
  });
}

init();
