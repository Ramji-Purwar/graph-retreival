let selectedMethod = 'lsh';
let queryGraphNetwork = null;
let currentTotal = parseInt(document.getElementById('header-total').textContent, 10) || 0;

// ── Vis.js options ────────────────────────────────────────────────────────────

const VIS_OPTIONS_QUERY = {
  autoResize: true,
  physics: {
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -30,
      centralGravity: 0.015,
      springLength: 60,
      springConstant: 0.08,
      damping: 0.4,
    },
    stabilization: { iterations: 120 },
  },
  nodes: {
    shape: 'dot',
    size: 10,
    color: {
      background: '#4f8ef7',
      border: '#6ba1ff',
      highlight: { background: '#a78bfa', border: '#c4b5fd' },
      hover:     { background: '#6ba1ff', border: '#93bbff' },
    },
    borderWidth: 2,
    shadow: { enabled: true, color: 'rgba(79,142,247,0.25)', size: 8 },
    font:  { color: '#94a3b8', size: 10, face: 'JetBrains Mono' },
  },
  edges: {
    color:  { color: '#334155', highlight: '#a78bfa', hover: '#475569' },
    width:  1.5,
    smooth: { type: 'continuous' },
  },
  interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
};

const VIS_OPTIONS_MINI = {
  autoResize: true,
  physics: {
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -20,
      centralGravity: 0.02,
      springLength: 40,
      springConstant: 0.1,
      damping: 0.5,
    },
    stabilization: { iterations: 80 },
  },
  nodes: {
    shape: 'dot',
    size:  7,
    color: {
      background: '#a78bfa',
      border:     '#c4b5fd',
      highlight:  { background: '#4f8ef7', border: '#6ba1ff' },
      hover:      { background: '#c4b5fd', border: '#ddd6fe' },
    },
    borderWidth: 1.5,
    shadow: { enabled: true, color: 'rgba(167,139,250,0.2)', size: 5 },
    font:   { size: 0 },
  },
  edges: {
    color:  { color: '#2e3446', highlight: '#4f8ef7', hover: '#475569' },
    width:  1,
    smooth: { type: 'continuous' },
  },
  interaction: { hover: true, zoomView: true, dragView: true, dragNodes: true },
};

// ── Label helpers ─────────────────────────────────────────────────────────────

// Dataset-specific label names; falls back to generic "Class N" for unknown datasets
const LABEL_NAMES = {
  'mutag':         { 0: 'non-mut.',  1: 'mutagenic' },
  'proteins':      { 0: 'non-enzyme',1: 'enzyme'    },
  'imdb-binary':   { 0: 'Romance',   1: 'Action'    },
  'aids':          { 0: 'inactive',  1: 'active'    },
  'reddit-binary': { 0: 'Thread',    1: 'QA'        },
};

// Derive active dataset from the active ds-pill
function getActiveDataset() {
  const active = document.querySelector('.ds-pill.active');
  return active ? active.dataset.ds : null;
}

function labelName(label) {
  const ds = getActiveDataset();
  if (ds && LABEL_NAMES[ds] && LABEL_NAMES[ds][label] !== undefined) {
    return LABEL_NAMES[ds][label];
  }
  return `Class ${label}`;
}

function labelPillHtml(label) {
  return `<span class="label-pill label-${label}">${labelName(label)}</span>`;
}

// ── Graph rendering ───────────────────────────────────────────────────────────

function renderGraph(container, topology, options) {
  const nodes   = new vis.DataSet(topology.nodes.map(id => ({ id, label: String(id) })));
  const edges   = new vis.DataSet(topology.edges.map((e, i) => ({ id: i, from: e.from, to: e.to })));
  const network = new vis.Network(container, { nodes, edges }, options);
  network.once('stabilized', () => network.setOptions({ physics: { enabled: false } }));
  return network;
}

// ── Dataset switching ─────────────────────────────────────────────────────────

async function switchDataset(ds) {
  if (document.querySelector(`.ds-pill[data-ds="${ds}"]`)?.classList.contains('active')) return;

  // Optimistically mark as loading
  document.querySelectorAll('.ds-pill').forEach(b => b.classList.remove('active'));
  const pill = document.querySelector(`.ds-pill[data-ds="${ds}"]`);
  if (pill) pill.classList.add('active', 'loading');

  try {
    const res  = await fetch('/switch', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ dataset: ds }),
    });
    const data = await res.json();

    if (data.error) {
      showToast(`⚠ ${data.error}`, 'error');
      // Revert pills
      document.querySelectorAll('.ds-pill').forEach(b => b.classList.remove('active', 'loading'));
      const prev = getActiveDataset();
      if (prev) document.querySelector(`.ds-pill[data-ds="${prev}"]`)?.classList.add('active');
      return;
    }

    if (pill) { pill.classList.remove('loading'); }
    currentTotal = data.total;

    // Update header
    document.getElementById('header-total').textContent = data.total;
    document.getElementById('header-tag').innerHTML =
      `${ds.toUpperCase()} · <span id="header-total">${data.total}</span> graphs`;

    // Update index input bounds
    const idxInput = document.getElementById('graph-idx');
    idxInput.max = data.total - 1;
    document.getElementById('max-idx-label').textContent = data.total - 1;
    idxInput.value = 0;

    // Update page title
    document.title = `Graph Retrieval — ${ds.toUpperCase()}`;

    // Clear results
    document.getElementById('main').innerHTML = `
      <div class="empty">
        <div class="big">◈</div>
        <p>Switched to <strong>${ds}</strong>.<br/>Enter a graph index and click Retrieve.</p>
      </div>`;

    // Reset query card
    document.getElementById('query-card').classList.remove('visible');
    if (queryGraphNetwork) { queryGraphNetwork.destroy(); queryGraphNetwork = null; }

    // Load first graph preview
    loadGraph(0);
    showToast(`Switched to ${ds}`, 'ok');
  } catch (e) {
    showToast('Server error during switch', 'error');
    if (pill) pill.classList.remove('loading');
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg, type = 'ok') {
  const t = document.getElementById('switch-toast');
  t.textContent = msg;
  t.className   = `switch-toast visible ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('visible'), 2800);
}

// ── Method toggle ─────────────────────────────────────────────────────────────

function setMethod(m) {
  selectedMethod = m;
  document.querySelectorAll('.toggle-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.method === m);
  });
}

// ── Load graph preview ────────────────────────────────────────────────────────

const idxInput = document.getElementById('graph-idx');
idxInput.addEventListener('input', () => loadGraph(idxInput.value));
loadGraph(0);

async function loadGraph(idx) {
  if (idx === '' || idx < 0) return;
  try {
    const res  = await fetch(`/graph/${idx}`);
    const data = await res.json();
    if (data.error) return;

    document.getElementById('query-card').classList.add('visible');
    document.getElementById('q-index').textContent = `#${data.index}`;
    document.getElementById('q-nodes').textContent = data.nodes;
    document.getElementById('q-edges').textContent = data.edges;
    document.getElementById('q-label-pill').innerHTML = labelPillHtml(data.label);

    const vizContainer = document.getElementById('query-graph-viz');
    if (queryGraphNetwork) { queryGraphNetwork.destroy(); queryGraphNetwork = null; }
    if (data.topology) {
      queryGraphNetwork = renderGraph(vizContainer, data.topology, VIS_OPTIONS_QUERY);
    }
  } catch(e) {}
}

// ── Retrieve ──────────────────────────────────────────────────────────────────

async function retrieve() {
  const idx = parseInt(document.getElementById('graph-idx').value);
  const k   = parseInt(document.getElementById('k-select').value);
  const btn = document.getElementById('retrieve-btn');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Retrieving…';

  const main = document.getElementById('main');
  main.innerHTML = '';

  try {
    const res  = await fetch('/retrieve', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ index: idx, k, method: selectedMethod }),
    });
    const data = await res.json();

    if (data.error) {
      main.innerHTML = `<div class="error-msg">${data.error}</div>`;
      return;
    }

    // Header
    const hdr = document.createElement('div');
    hdr.className = 'results-header';
    hdr.innerHTML = `
      <div class="results-title">Top-${k} results · <span style="color:var(--accent2)">${data.method === 'lsh' ? 'LSH-ANN' : 'Brute-force'}</span></div>
      <div class="time-badge">Query time: <span>${data.time_ms} ms</span></div>
    `;
    main.appendChild(hdr);

    // Grid
    const grid = document.createElement('div');
    grid.className = 'results-grid';

    data.results.forEach((g, i) => {
      const card  = document.createElement('div');
      card.className = 'result-card';
      const vizId = `result-graph-${i}`;
      card.innerHTML = `
        <div class="rank"># ${i + 1}</div>
        <div class="ridx">${g.index}</div>
        <div class="graph-viz-mini" id="${vizId}"></div>
        <div class="stat-row">
          <div class="stat"><div class="val">${g.nodes}</div><div class="key">nodes</div></div>
          <div class="stat"><div class="val">${g.edges}</div><div class="key">edges</div></div>
        </div>
        <div>${labelPillHtml(g.label)}</div>
      `;
      grid.appendChild(card);
    });

    main.appendChild(grid);

    // Render mini graphs after DOM update
    requestAnimationFrame(() => {
      data.results.forEach((g, i) => {
        if (g.topology) {
          const container = document.getElementById(`result-graph-${i}`);
          if (container) renderGraph(container, g.topology, VIS_OPTIONS_MINI);
        }
      });
    });

  } catch(e) {
    main.innerHTML = `<div class="error-msg">Server error. Is app.py running?</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Retrieve';
  }
}
