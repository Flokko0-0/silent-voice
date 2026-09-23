import fs from 'node:fs';
import { dtw } from '../recognizer.js';
import { FEATURE_GROUPS } from '../features.js';

const file = process.argv[2];
const s = JSON.parse(fs.readFileSync(file, 'utf8'));
const data = [];
for (const [k, seqs] of Object.entries(s.templates)) {
  const [id, lang] = k.split('|');
  if (lang !== s.lang) continue;
  for (const seq of seqs) data.push({ label: id, seq });
}

function run(name, { groups = [0, 1, 2], weights = [1, 1, 0.8], deltas = 0, minLen = 0, resample = 0, k = 2, cosine = false, seqMean = false }) {
  let items = data.filter((d) => d.seq.length >= minLen);
  const dims = FEATURE_GROUPS.map((g, i) => (groups.includes(g) ? i : -1)).filter((i) => i >= 0);
  const pre = (seq) => {
    let x = seq.map((f) => dims.map((i) => f[i]));
    if (resample) x = Array.from({ length: resample }, (_, t) => x[Math.min(x.length - 1, Math.round((t * (x.length - 1)) / (resample - 1)))]);
    if (seqMean) {
      const m = x[0].map((_, j) => x.reduce((a, f) => a + f[j], 0) / x.length);
      x = x.map((f) => f.map((v, j) => v - m[j]));
    }
    if (deltas) x = x.map((f, t) => { const p = x[Math.max(0, t - 1)], n = x[Math.min(x.length - 1, t + 1)]; return [...f, ...f.map((_, j) => ((n[j] - p[j]) / 2) * deltas)]; });
    return x;
  };
  let seqs = items.map((d) => pre(d.seq));
  const D = seqs[0][0].length;
  const mu = new Array(D).fill(0), sq = new Array(D).fill(0); let n = 0;
  for (const q of seqs) for (const f of q) { f.forEach((v, j) => { mu[j] += v; sq[j] += v * v; }); n++; }
  const g2 = dims.map((i) => FEATURE_GROUPS[i]);
  const gAll = deltas ? [...g2, ...g2] : g2;
  const size = {}; gAll.forEach((g) => (size[g] = (size[g] || 0) + 1));
  const sc = mu.map((m, j) => { const mm = m / n; const sd = Math.max(Math.sqrt(Math.max(sq[j] / n - mm * mm, 0)), 1e-3); return weights[gAll[j]] / Math.sqrt(size[gAll[j]]) / sd; });
  const mus = mu.map((m) => m / n);
  seqs = seqs.map((q) => q.map((f) => f.map((v, j) => (v - mus[j]) * sc[j])));
  let correct = 0; const conf = {};
  for (let i = 0; i < seqs.length; i++) {
    const per = {};
    for (let j = 0; j < seqs.length; j++) if (j !== i) (per[items[j].label] ||= []).push(dtw(seqs[i], seqs[j]));
    const ranked = Object.entries(per).map(([l, ds]) => { ds.sort((a, b) => a - b); const b = ds.slice(0, k); return [l, b.reduce((a, v) => a + v, 0) / b.length]; }).sort((a, b) => a[1] - b[1]);
    if (ranked[0][0] === items[i].label) correct++; else { const key = `${items[i].label}->${ranked[0][0]}`; conf[key] = (conf[key] || 0) + 1; }
  }
  console.log(`${(100 * correct / seqs.length).toFixed(0).padStart(3)}%  ${correct}/${seqs.length}  ${name}`);
  return conf;
}

const conf = run('baseline (all groups)', {});
run('geom only', { groups: [0] });
run('blendshapes only', { groups: [1] });
run('contour only', { groups: [2] });
run('geom+bs', { groups: [0, 1] });
run('geom+bs +deltas', { groups: [0, 1], deltas: 1 });
run('all +deltas', { deltas: 1 });
run('geom+bs, k=1', { groups: [0, 1], k: 1 });
run('geom+bs, seqMean', { groups: [0, 1], seqMean: true });
run('all, minLen 25', { minLen: 25 });
run('geom+bs, minLen 25', { groups: [0, 1], minLen: 25 });
run('geom+bs +deltas, minLen 25', { groups: [0, 1], deltas: 1, minLen: 25 });
console.log('baseline confusions', conf);

console.log('--- more');
run('all +deltas2, minLen 25', { deltas: 2, minLen: 25 });
run('geom+bs +deltas2, minLen 25', { groups: [0, 1], deltas: 2, minLen: 25 });
run('all w[1,.5,.5] +deltas, minLen 25', { weights: [1, 0.5, 0.5], deltas: 1, minLen: 25 });
run('geom+contour +deltas, minLen 25', { groups: [0, 2], deltas: 1, minLen: 25 });
const keep = (ids) => { const saved = data.splice(0); data.push(...saved.filter((d) => ids.includes(d.label))); return saved; };
const sets = { five: ['drink', 'cold', 'family', 'turn', 'thanks'], six: ['drink', 'pain', 'cold', 'family', 'turn', 'thanks'], lipOnly: ['drink', 'cold', 'breath', 'family', 'turn', 'thanks'] };
for (const [nm, ids] of Object.entries(sets)) {
  const saved = keep(ids);
  run(`${nm}: geom+bs +deltas, minLen 25`, { groups: [0, 1], deltas: 1, minLen: 25 });
  run(`${nm}: all +deltas, minLen 25`, { deltas: 1, minLen: 25 });
  data.splice(0); data.push(...saved);
}
