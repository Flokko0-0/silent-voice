// Нейросеть чтения по губам прямо в браузере (ONNX Runtime Web).
import * as ort from './vendor/ort/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const MODEL_URL = new URL('./vendor/lipfront.onnx', import.meta.url).href;
const CROP = 96;
const SIZE = 88;
const FPS = 25;
const DIM = 768;

let session = null;
let loading = null;
let queue = Promise.resolve();
let items = []; // { id, lang, label, T, feats: Float32Array }
let cache = null; // { lang, mu, entries: [{ label, T, seq }] }

export const browserNet = { ready: false, progress: 0 };

export function loadModel(onProgress = () => {}) {
  loading ??= (async () => {
    const res = await fetch(MODEL_URL);
    const total = Number(res.headers.get('content-length')) || 46e6;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      browserNet.progress = Math.min(99, Math.round((got / total) * 100));
      onProgress(browserNet.progress);
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    items = await dbAll();
    cache = null;
    browserNet.ready = true;
    browserNet.progress = 100;
    onProgress(100);
  })();
  return loading;
}

function resample(rois, times) {
  if (!times || times.length !== rois.length || rois.length < 2) return rois;
  const out = [];
  let j = 0;
  for (let t = times[0]; t <= times[times.length - 1] + 1e-6; t += 1000 / FPS) {
    while (j < times.length - 1 && times[j] < t) j++;
    out.push(rois[j]);
  }
  return out;
}

function embed(rois, times) {
  const run = async () => {
    const clip = resample(rois, times);
    const T = clip.length;
    const off = (CROP - SIZE) / 2;
    const x = new Float32Array(T * SIZE * SIZE);
    for (let t = 0; t < T; t++) {
      const f = clip[t];
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          x[t * SIZE * SIZE + r * SIZE + c] = (f[(r + off) * CROP + c + off] / 255 - 0.421) / 0.165;
        }
      }
    }
    const out = await session.run({ frames: new ort.Tensor('float32', x, [1, T, 1, SIZE, SIZE]) });
    return { T, feats: new Float32Array(out.features.data) };
  };
  const p = queue.then(run, run);
  queue = p.catch(() => {});
  return p;
}

function prepared(lang) {
  if (cache?.lang === lang) return cache;
  const mine = items.filter((it) => it.lang === lang);
  const mu = new Float32Array(DIM);
  let n = 0;
  for (const it of mine) {
    for (let t = 0; t < it.T; t++) for (let k = 0; k < DIM; k++) mu[k] += it.feats[t * DIM + k];
    n += it.T;
  }
  if (n) for (let k = 0; k < DIM; k++) mu[k] /= n;
  cache = { lang, mu, entries: mine.map((it) => ({ label: it.label, T: it.T, seq: centered(it.feats, it.T, mu) })) };
  return cache;
}

function centered(feats, T, mu) {
  const out = new Float32Array(T * DIM);
  for (let t = 0; t < T; t++) {
    let norm = 0;
    for (let k = 0; k < DIM; k++) {
      const v = feats[t * DIM + k] - mu[k];
      out[t * DIM + k] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm) + 1e-8;
    for (let k = 0; k < DIM; k++) out[t * DIM + k] /= norm;
  }
  return out;
}

function dtwCos(a, n, b, m) {
  let prev = new Float64Array(m + 1).fill(Infinity);
  let cur = new Float64Array(m + 1);
  prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(Infinity);
    const ai = (i - 1) * DIM;
    for (let j = 1; j <= m; j++) {
      const bj = (j - 1) * DIM;
      let dot = 0;
      for (let k = 0; k < DIM; k++) dot += a[ai + k] * b[bj + k];
      cur[j] = 1 - dot + Math.min(prev[j], cur[j - 1], prev[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m] / (n + m);
}

function rank(entries, seq, T, exclude = -1) {
  const per = new Map();
  entries.forEach((e, i) => {
    if (i === exclude) return;
    if (!per.has(e.label)) per.set(e.label, []);
    per.get(e.label).push(dtwCos(seq, T, e.seq, e.T));
  });
  const ranked = [];
  for (const [label, ds] of per) {
    ds.sort((x, y) => x - y);
    const best = ds.slice(0, 2);
    ranked.push({ label, dist: best.reduce((s, v) => s + v, 0) / best.length });
  }
  return ranked.sort((x, y) => x.dist - y.dist);
}

export async function enroll(lang, label, rois, times) {
  const { T, feats } = await embed(rois, times);
  const item = { lang, label, T, feats };
  item.id = await dbPut(item);
  items.push(item);
  cache = null;
  return items.filter((it) => it.lang === lang && it.label === label).length;
}

export async function classify(lang, rois, times) {
  const t0 = performance.now();
  const { T, feats } = await embed(rois, times);
  const { mu, entries } = prepared(lang);
  if (!entries.length) return { ranked: [], ms: 0 };
  const ranked = rank(entries, centered(feats, T, mu), T);
  return { ranked, ms: Math.round(performance.now() - t0) };
}

export function evaluate(lang) {
  const { entries } = prepared(lang);
  let correct = 0;
  let total = 0;
  entries.forEach((e, i) => {
    if (entries.filter((o) => o.label === e.label).length < 2) return;
    const r = rank(entries, e.seq, e.T, i);
    total++;
    if (r[0]?.label === e.label) correct++;
  });
  return { correct, total };
}

export async function clear(lang, label = null) {
  const drop = items.filter((it) => (lang === null || it.lang === lang) && (label === null || it.label === label));
  await Promise.all(drop.map((it) => dbDelete(it.id)));
  items = items.filter((it) => !drop.includes(it));
  cache = null;
}

export function count() {
  return items.length;
}

// ---------- IndexedDB ----------
let dbp = null;
function db() {
  dbp ??= new Promise((resolve, reject) => {
    const req = indexedDB.open('silent-voice', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('clips', { keyPath: 'id', autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction('clips', mode);
    const req = fn(t.objectStore('clips'));
    t.oncomplete = () => resolve(req?.result);
    t.onerror = () => reject(t.error);
  });
}
const dbPut = (item) => tx('readwrite', (s) => s.add({ lang: item.lang, label: item.label, T: item.T, feats: item.feats }));
const dbDelete = (id) => tx('readwrite', (s) => s.delete(id));
async function dbAll() {
  try {
    return (await tx('readonly', (s) => s.getAll())) || [];
  } catch {
    return [];
  }
}
