const GROUP_WEIGHTS = [1.0, 0.5, 0.5];

function withDeltas(seq) {
  return seq.map((f, t) => {
    const prev = seq[Math.max(0, t - 1)];
    const next = seq[Math.min(seq.length - 1, t + 1)];
    const out = new Array(f.length * 2);
    for (let k = 0; k < f.length; k++) {
      out[k] = f[k];
      out[f.length + k] = (next[k] - prev[k]) / 2;
    }
    return out;
  });
}

export class Recognizer {
  constructor(groups) {
    this.groups = [...groups, ...groups];
    this.items = [];
  }

  // templates: [{ label, seq: number[][] }]
  fit(templates) {
    templates = templates.map((t) => ({ label: t.label, seq: withDeltas(t.seq) }));
    const dim = this.groups.length;
    const mu = new Float64Array(dim);
    const sq = new Float64Array(dim);
    let n = 0;
    for (const t of templates) {
      for (const f of t.seq) {
        for (let k = 0; k < dim; k++) {
          mu[k] += f[k];
          sq[k] += f[k] * f[k];
        }
        n++;
      }
    }
    const groupSize = [0, 0, 0];
    for (const g of this.groups) groupSize[g]++;
    this.mu = mu.map((v) => (n ? v / n : 0));
    this.scale = new Float64Array(dim);
    for (let k = 0; k < dim; k++) {
      const variance = n ? sq[k] / n - this.mu[k] * this.mu[k] : 1;
      const sd = Math.max(Math.sqrt(Math.max(variance, 0)), 1e-3);
      const g = this.groups[k];
      this.scale[k] = GROUP_WEIGHTS[g] / Math.sqrt(groupSize[g]) / sd;
    }
    this.items = templates.map((t, index) => ({ label: t.label, index, seq: t.seq.map((f) => this.#norm(f)) }));
    this.labels = [...new Set(templates.map((t) => t.label))];
  }

  #norm(f) {
    const out = new Float64Array(f.length);
    for (let k = 0; k < f.length; k++) out[k] = (f[k] - this.mu[k]) * this.scale[k];
    return out;
  }

  classify(seq, excludeIndex = -1) {
    if (!this.items.length) return [];
    return this.#classifyNormalized(withDeltas(seq).map((f) => this.#norm(f)), excludeIndex);
  }

  evaluate() {
    let correct = 0;
    let total = 0;
    const correctDists = [];
    const mistakes = [];
    for (const item of this.items) {
      const siblings = this.items.filter((o) => o.label === item.label).length;
      if (siblings < 2) continue;
      const ranked = this.#classifyNormalized(item.seq, item.index);
      if (!ranked.length) continue;
      total++;
      if (ranked[0].label === item.label) {
        correct++;
        correctDists.push(ranked[0].dist);
      } else {
        mistakes.push({ expected: item.label, got: ranked[0].label });
      }
    }
    return { correct, total, correctDists, mistakes };
  }

  #classifyNormalized(q, excludeIndex) {
    const perLabel = new Map();
    for (const item of this.items) {
      if (item.index === excludeIndex) continue;
      const dist = dtw(q, item.seq);
      if (!perLabel.has(item.label)) perLabel.set(item.label, []);
      perLabel.get(item.label).push(dist);
    }
    const ranked = [];
    for (const [label, dists] of perLabel) {
      dists.sort((x, y) => x - y);
      const best = dists.slice(0, 2);
      ranked.push({ label, dist: best.reduce((s, v) => s + v, 0) / best.length });
    }
    return ranked.sort((x, y) => x.dist - y.dist);
  }
}

function frameDist(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) {
    const v = a[k] - b[k];
    s += v * v;
  }
  return Math.sqrt(s);
}

export function dtw(a, b) {
  const n = a.length;
  const m = b.length;
  if (!n || !m) return Infinity;
  const band = Math.max(Math.ceil(Math.max(n, m) * 0.3), Math.abs(n - m) + 1);
  let prev = new Float64Array(m + 1).fill(Infinity);
  let cur = new Float64Array(m + 1);
  prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(Infinity);
    const from = Math.max(1, i - band);
    const to = Math.min(m, i + band);
    for (let j = from; j <= to; j++) {
      const best = Math.min(prev[j], cur[j - 1], prev[j - 1]);
      cur[j] = frameDist(a[i - 1], b[j - 1]) + best;
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m] / (n + m);
}
