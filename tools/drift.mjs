import fs from 'node:fs';
import { Recognizer } from '../recognizer.js';
import { FEATURE_GROUPS } from '../features.js';
const s = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const data = [];
for (const [k, v] of Object.entries(s.templates)) for (const q of v) if (q.length >= 20) data.push({ label: k, seq: q });
const D = data[0].seq[0].length;
// per-dim spread across all frames, to size the offsets realistically
const sd = new Array(D).fill(0), mu = new Array(D).fill(0); let n = 0;
for (const d of data) for (const f of d.seq) { f.forEach((v, j) => (mu[j] += v)); n++; }
mu.forEach((v, j) => (mu[j] = v / n));
for (const d of data) for (const f of d.seq) f.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / n));
sd.forEach((v, j) => (sd[j] = Math.sqrt(v)));
let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;

function lo(transform, drift) {
  seed = 7;
  const items = data.map((d) => { const off = sd.map((s) => s * drift * rnd()); return { label: d.label, seq: transform(d.seq.map((f) => f.map((v, j) => v + off[j]))) }; });
  const r = new Recognizer(FEATURE_GROUPS); r.fit(items); const e = r.evaluate();
  return Math.round((100 * e.correct) / e.total) + '%';
}
const none = (q) => q;
const firstFrames = (q) => { const b = q.slice(0, 3); const m = b[0].map((_, j) => b.reduce((a, f) => a + f[j], 0) / b.length); return q.map((f) => f.map((v, j) => v - m[j])); };
const edges = (q) => { const b = [...q.slice(0, 3), ...q.slice(-3)]; const m = b[0].map((_, j) => b.reduce((a, f) => a + f[j], 0) / b.length); return q.map((f) => f.map((v, j) => v - m[j])); };
for (const drift of [0, 0.3, 0.6, 1.0]) console.log(`drift ${drift}:  rest-only ${lo(none, drift)}   first-3 ${lo(firstFrames, drift)}   edges ${lo(edges, drift)}`);
