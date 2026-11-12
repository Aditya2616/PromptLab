// PromptLab viewer. Renders a run.json (recorded or streamed live) as a lineage graph.
const $ = (s) => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const S = { run: null, nodes: [], byId: {}, G: 0, maxStep: 0, step: 0, sel: null, playing: false, timer: null,
            speed: 1, inView: true, lineage: new Set(), els: {}, live: false, layout: {} };

const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const pct = (x, d = 1) => (x * 100).toFixed(d);
const pts = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)} pts`;
const fmtP = (p) => (p < 0.001 ? '<0.001' : p.toFixed(3));
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const isPhone = () => matchMedia('(max-width: 960px)').matches;

/* ---------- data ---------- */
async function init() {
  try {
    const idx = await (await fetch('data/index.json')).json();
    S.index = idx.runs;
    const tabs = $('#runTabs');
    idx.runs.forEach((r, i) => {
      const b = document.createElement('button');
      b.className = 'tab'; b.role = 'tab'; b.textContent = r.task; b.dataset.file = r.file;
      b.setAttribute('aria-selected', i === 0);
      b.onclick = () => selectTab(r.file);
      tabs.appendChild(b);
    });
    if (idx.runs.length) await selectTab(idx.runs[0].file);
  } catch (e) {
    $('#caption').textContent = 'Could not load recorded runs (data/index.json).';
  }
  bindControls();
  detectLive();
}

async function selectTab(file) {
  stopLive();
  document.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', t.dataset.file === file));
  const run = await (await fetch(`data/${file}`)).json();
  loadRun(run, { fresh: true });
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
  S.lineage = new Set();
  const walk = (id) => { if (!id || S.lineage.has(id)) return; S.lineage.add(id); (S.byId[id]?.parents || []).forEach(walk); };
  walk(S.winner);
}

function loadRun(run, { fresh = false } = {}) {
  derive(run);
  const t = run.task, g = run.git || {};
  $('#runMeta').textContent = (run.recorded === false || S.live)
    ? `LIVE RUN · ${run.model} · backend ${run.backend} · seed ${run.seed}`
    : `RECORDED RUN · ${run.model} · backend ${run.backend} · seed ${run.seed} · commit ${g.commit || 'n/a'}${g.dirty ? '+dirty' : ''} · ` +
      `${t.n_dev} dev / ${t.n_test} test items · ${run.calls?.model_calls ?? '?'} model calls · ${Math.round((run.seconds || 0) / 60)} min · data: ${t.source}`;
  buildGraph();
  renderCurve();
  renderResults();
  renderOps();
  if (fresh) {
    S.sel = null;
    renderDetail(null);
    stopPlay();
    if (reduce || S.live) setStep(S.maxStep); else { setStep(0); S.sel = null; startPlay(); }
  } else {
    setStep(S.maxStep);
  }
  const sc = $('#scrub');
  sc.max = S.maxStep;
}

/* ---------- graph ---------- */
function buildGraph() {
  const svg = $('#graph'); svg.replaceChildren();
  const box = $('#graphScroll');
  const padL = 46, padR = 34, padT = 34, padB = 22, H = isPhone() ? 300 : 360;
  const colW = Math.max(120, Math.floor((box.clientWidth - padL - padR) / Math.max(1, S.G)));
  const W = padL + S.G * colW + padR;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const accs = S.nodes.map((n) => n.dev_acc);
  let lo = Math.max(0, Math.min(...accs) - 0.05), hi = Math.min(1, Math.max(...accs) + 0.05);
  if (hi - lo < 0.2) { const m = (hi + lo) / 2; lo = Math.max(0, m - 0.1); hi = Math.min(1, m + 0.1); }
  const Y = (a) => padT + (1 - (a - lo) / (hi - lo)) * (H - padT - padB);
  const X = (g) => padL + g * colW;
  const step = hi - lo > 0.5 ? 0.2 : 0.1;
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    el('line', { class: 'grid-line', x1: padL - 6, x2: W - padR + 10, y1: Y(v), y2: Y(v) }, svg);
    el('text', { class: 'grid-label', x: 0, y: Y(v) + 3 }, svg).textContent = pct(v, 0);
  }
  const pos = {}, labelFor = new Set();
  for (let g = 0; g <= S.G; g++) {
    el('line', { class: 'col-rule', x1: X(g), x2: X(g), y1: 20, y2: H - padB + 6 }, svg);
    el('text', { class: 'col-label', x: X(g), y: 12, 'text-anchor': 'middle' }, svg).textContent = g === 0 ? 'GEN 0 · SEEDS' : `GEN ${g}`;
    const col = S.nodes.filter((n) => n.gen === g).sort((a, b) => b.dev_acc - a.dev_acc);
    if (col.length) labelFor.add(col[0].id);
    const offs = [0, 26, -26, 52, -52], lastY = offs.map(() => -99);
    for (const n of col) {
      const y = Y(n.dev_acc);
      let lane = offs.findIndex((_, i) => y - lastY[i] >= 24);
      if (lane < 0) lane = 0;
      lastY[lane] = y;
      pos[n.id] = [X(g) + offs[lane], y];
    }
  }
  S.layout = { pos, W, H };
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
    el('circle', { class: 'dot', r: 7 }, g);
    el('circle', { class: 'ring', r: 13 }, g);
    el('circle', { r: 22, fill: 'transparent' }, g);
    g.addEventListener('click', () => selectNode(n.id, true));
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectNode(n.id, true); } });
    els.nodes[n.id] = g;
    if (labelFor.has(n.id)) {
      const t = el('text', { class: 'node-label', x, y: y - 16, 'text-anchor': 'middle' }, svg);
      t.textContent = pct(n.dev_acc);
      els.labels[n.id] = t;
    }
  }
  S.els = els;
  applyStep(S.step);
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
  }
  const g = Math.floor(s / 2);
  let cap;
  if (s > 2 * S.G + 1) cap = 'HELD-OUT TEST · winner scored once on data the search never saw';
  else if (s % 2 === 0) {
    const gen = S.run.generations.find((x) => x.gen === g);
    cap = `GEN ${g} · ${g === 0 ? 'seed prompts' : 'children'} scored on dev${gen ? ` (${gen.born.length} variants)` : ''}`;
  } else {
    const gen = S.run.generations.find((x) => x.gen === g);
    cap = `GEN ${g} · select: ${gen ? gen.survivors.length : '?'} survive · best dev ${gen ? pct(gen.best_dev) : '?'}%`;
  }
  $('#caption').textContent = cap;
  $('#scrub').value = s;
  updateCurve(s);
  if (S.playing && !reduce) followColumn(Math.min(S.G, g));
}

function followColumn(g) {
  const box = $('#graphScroll');
  if (box.scrollWidth <= box.clientWidth + 2) return;
  const x = S.layout.pos ? (46 + g * (S.layout.W - 80) / Math.max(1, S.G)) : 0;
  box.scrollTo({ left: Math.max(0, x - box.clientWidth / 2), behavior: 'smooth' });
}

/* ---------- playback ---------- */
function setStep(s) { applyStep(Math.max(0, Math.min(S.maxStep, s))); if (S.step >= S.maxStep && !S.sel && S.winner && !isPhone()) selectNode(S.winner, false); }
function startPlay() {
  if (S.step >= S.maxStep) S.step = 0;
  S.playing = true; $('#playIcon').setAttribute('d', 'M7 5h4v14H7zM13 5h4v14h-4z'); $('#btnPlay').setAttribute('aria-label', 'Pause');
  const tick = () => {
    if (!S.playing) return;
    if (!S.inView) { S.timer = setTimeout(tick, 400); return; }
    if (S.step >= S.maxStep) { stopPlay(); return; }
    setStep(S.step + 1);
    S.timer = setTimeout(tick, 1150 / S.speed);
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
  $('#btnSpeed').onclick = (e) => { S.speed = S.speed === 1 ? 2 : S.speed === 2 ? 0.5 : 1; e.target.textContent = `${S.speed}x`; };
  $('#btnClose').onclick = closeSheet; $('#scrim').onclick = closeSheet;
  new IntersectionObserver(([e]) => { S.inView = e.isIntersecting; }, { threshold: 0.15 }).observe($('#graphScroll'));
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (S.run) { buildGraph(); renderCurve(); updateCurve(S.step); } }, 150); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });
}

/* ---------- detail ---------- */
function openSheet() { if (isPhone()) { $('#detail').classList.add('open'); $('#scrim').classList.add('open'); } }
function closeSheet() { $('#detail').classList.remove('open'); $('#scrim').classList.remove('open'); }
function selectNode(id, open) {
  S.sel = id; applyStep(S.step); renderDetail(id);
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
    delta = `<dt>vs parent</dt><dd>${pts(n.dev_acc - par[0].dev_acc)} (fixed ${fixed}, broke ${broke})</dd>`;
  }
  let held = '';
  const arms = S.run.report?.arms || {};
  for (const a of Object.values(arms)) if (a.id === id) held += `<dt>held-out as ${esc(a.label)}</dt><dd>${pct(a.test_acc)}% [${pct(a.test_ci[0], 0)}–${pct(a.test_ci[1], 0)}]</dd>`;
  const text = par[0] ? lcsDiff(par[0].text, n.text) : esc(n.text);
  const parBtns = par.map((p) => `<button class="linkbtn" data-id="${p.id}">${esc(p.id)} (${pct(p.dev_acc)}%)</button>`).join(' ');
  $('#detailBody').innerHTML = `<h3>variant ${esc(n.id)}</h3><div class="chips">${chips.join('')}</div>
    <dl class="kv"><dt>dev accuracy</dt><dd>${pct(n.dev_acc)}% [${ci}]</dd>${delta}${held}
    <dt>few-shot examples</dt><dd>${n.n_shots}</dd></dl>
    <p class="sub">${par[0] ? `Prompt, diffed against parent${par.length > 1 ? 's (first shown)' : ''}` : 'Seed prompt'}</p>
    <pre class="prompt">${text}</pre>
    ${par.length ? `<p class="sub">Parents</p>${parBtns}` : ''}`;
  body.querySelectorAll('.linkbtn[data-id]').forEach((b) => (b.onclick = () => selectNode(b.dataset.id, true)));
}

/* ---------- curve ---------- */
const CV = { W: 760, H: 270, l: 46, r: 190, t: 18, b: 34 };
function renderCurve() {
  const svg = $('#curve'); svg.replaceChildren();
  const narrow = (svg.parentElement.clientWidth || 760) < 560;
  CV.W = narrow ? Math.max(320, svg.parentElement.clientWidth - 32) : 760; CV.r = narrow ? 120 : 190; CV.H = narrow ? 240 : 270;
  svg.setAttribute('viewBox', `0 0 ${CV.W} ${CV.H}`);
  const gens = S.run.generations, arms = S.run.report?.arms;
  const vals = gens.flatMap((g) => [g.best_dev, g.mean_dev]);
  if (arms) for (const a of Object.values(arms)) vals.push(a.test_ci[0], a.test_ci[1], a.dev_acc);
  let lo = Math.max(0, Math.min(...vals) - 0.05), hi = Math.min(1, Math.max(...vals) + 0.05);
  if (hi - lo < 0.2) hi = lo + 0.2;
  const pw = CV.W - CV.l - CV.r, ph = CV.H - CV.t - CV.b;
  const X = (g) => CV.l + (S.G ? (g / S.G) * pw : 0), Y = (a) => CV.t + (1 - (a - lo) / (hi - lo)) * ph;
  const step = hi - lo > 0.5 ? 0.2 : 0.1;
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    el('line', { class: 'grid-line', x1: CV.l, x2: CV.W - CV.r + 10, y1: Y(v), y2: Y(v) }, svg);
    el('text', { class: 'grid-label', x: 4, y: Y(v) + 3 }, svg).textContent = pct(v, 0) + '%';
  }
  gens.forEach((g) => el('text', { class: 'grid-label', x: X(g.gen), y: CV.H - 12, 'text-anchor': 'middle' }, svg).textContent = `gen ${g.gen}`);
  const line = (key, cls) => el('path', { class: cls, pathLength: 1, d: gens.map((g, i) => `${i ? 'L' : 'M'}${X(g.gen)},${Y(g[key])}`).join('') }, svg);
  S.cv = { mean: line('mean_dev', 'curve-mean'), best: line('best_dev', 'curve-line'), pts: [] };
  gens.forEach((g) => S.cv.pts.push(el('circle', { class: 'curve-pt', cx: X(g.gen), cy: Y(g.best_dev), r: 4.5 }, svg)));
  S.cv.test = [];
  if (arms) {
    const x0 = CV.W - CV.r + (narrow ? 22 : 50), sp = narrow ? 34 : 40;
    el('text', { class: 'col-label', x: x0 + sp, y: 12, 'text-anchor': 'middle' }, svg).textContent = narrow ? 'TEST' : 'HELD-OUT TEST';
    const order = narrow ? [['seed0', 'seed'], ['random', 'rand'], ['evolved', 'evo']] : [['seed0', 'seed 0'], ['random', 'random'], ['evolved', 'evolved']];
    order.forEach(([k, label], i) => {
      const a = arms[k], x = x0 + i * sp;
      const g = el('g', { class: 'curve-test' }, svg);
      el('line', { x1: x, x2: x, y1: Y(a.test_ci[0]), y2: Y(a.test_ci[1]), stroke: k === 'evolved' ? '#d7ff3f' : 'rgba(237,236,230,.45)', 'stroke-width': 2 }, g);
      el('circle', { cx: x, cy: Y(a.test_acc), r: 5.5, fill: k === 'evolved' ? '#07070a' : 'rgba(237,236,230,.6)', stroke: k === 'evolved' ? '#d7ff3f' : 'none', 'stroke-width': 2 }, g);
      el('text', { class: 'grid-label', x, y: CV.H - 12, 'text-anchor': 'middle' }, g).textContent = label;
      S.cv.test.push(g);
    });
  }
}
function updateCurve(s) {
  if (!S.cv) return;
  const gens = S.run.generations, v = gens.filter((g) => 2 * g.gen + 1 <= s).length;
  const off = v <= 1 ? 1 : 1 - (v - 1) / Math.max(1, S.G);
  S.cv.best.style.strokeDashoffset = off; S.cv.mean.style.strokeDashoffset = off;
  S.cv.pts.forEach((p, i) => (p.style.opacity = i < v ? 1 : 0));
  const final = !!S.run.report && s >= S.maxStep;
  S.cv.test.forEach((g) => (g.style.opacity = final ? 1 : 0));
}

/* ---------- results ---------- */
const ARM_LABEL = { seed0: 'First-draft seed prompt', best_seed: 'Best seed (picked on dev)', random: 'Random search, same budget', evolved: 'Evolved winner' };
function renderResults() {
  const rep = S.run.report, tbl = $('#armsTable'), ver = $('#verdicts');
  if (!rep) { tbl.innerHTML = ''; ver.innerHTML = '<p class="verdict mono">Held-out numbers appear when the run finishes.</p>'; $('#resultsSub').textContent = ''; return; }
  $('#resultsSub').textContent = `n = ${rep.n_test} test items, never seen by the search${rep.majority_rate != null ? ` · majority-class answer scores ${pct(rep.majority_rate)}%` : ''}`;
  const rows = ['seed0', 'best_seed', 'random', 'evolved'].map((k) => {
    const a = rep.arms[k];
    return `<tr class="${k === 'evolved' ? 'hl' : ''}"><td>${ARM_LABEL[k]}</td><td>${a.id}</td><td>${pct(a.dev_acc)}%</td><td>${pct(a.test_acc)}%</td>
      <td class="ci">${pct(a.test_ci[0])}–${pct(a.test_ci[1])}</td><td>${pts(-a.gap).replace('pts', '')}</td></tr>`;
  }).join('');
  tbl.innerHTML = `<thead><tr><th>Prompt</th><th>id</th><th>dev</th><th>test</th><th>95% CI</th><th>test − dev</th></tr></thead><tbody>${rows}</tbody>`;
  const names = { evolved_vs_seed0: 'first-draft seed', evolved_vs_best_seed: 'best seed', evolved_vs_random: 'random search' };
  ver.innerHTML = Object.entries(rep.comparisons).map(([k, c]) => {
    const good = c.p_mcnemar < 0.05 && c.diff > 0;
    const text = `Evolved vs ${names[k]}: ${pts(c.diff)} (95% CI ${pts(c.lo)} to ${pts(c.hi)}), exact McNemar p = ${fmtP(c.p_mcnemar)} ` +
      `(${c.a_only} better, ${c.b_only} worse). ${good ? 'Significant at 0.05.' : c.diff <= 0 ? 'No gain.' : 'Not significant at 0.05.'}`;
    return `<p class="verdict ${good ? 'ok' : 'warn'}">${text}</p>`;
  }).join('');
}

function renderOps() {
  const list = $('#opsList'), rows = S.run.operator_stats || [];
  if (!rows.length) { list.innerHTML = '<p class="muted mono">Available when the run finishes.</p>'; return; }
  list.innerHTML = rows.map((r) => `<div class="op"><span>${esc(r.operator)}</span>
    <span class="bar"><i style="width:${(100 * r.improved / r.n).toFixed(0)}%"></i></span>
    <span class="num">${r.improved}/${r.n} improved · ${r.survived} survived · mean ${pts(r.mean_delta)}</span></div>`).join('');
}

/* ---------- live ---------- */
async function detectLive() {
  if (!/^https?:$/.test(location.protocol) || !/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)) return;
  try {
    const r = await fetch('/health'); if (!r.ok) return;
    const j = await r.json(); if (j.status !== 'ok') return;
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
  document.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', 'false'));
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
