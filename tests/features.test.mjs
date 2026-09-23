import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, FEATURE_DIM, FEATURE_GROUPS, OUTER_LIPS, INNER_LIPS } from '../features.js';

// Синтетическое лицо: 478 точек, глаза и рот на своих местах.
function face({ shift = [0, 0], scale = 1, angle = 0, open = 0.02 } = {}) {
  const pts = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const put = (i, x, y) => {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const rx = x * c - y * s;
    const ry = x * s + y * c;
    pts[i] = { x: 0.5 + shift[0] + rx * scale, y: 0.5 + shift[1] + ry * scale, z: 0 };
  };
  put(33, -0.1, -0.1);
  put(263, 0.1, -0.1);
  put(1, 0, 0);
  put(152, 0, 0.18);
  OUTER_LIPS.forEach((i, k) => {
    const t = (k / OUTER_LIPS.length) * Math.PI * 2;
    put(i, Math.cos(t) * 0.05, 0.1 + Math.sin(t) * (0.02 + open));
  });
  INNER_LIPS.forEach((i, k) => {
    const t = (k / INNER_LIPS.length) * Math.PI * 2;
    put(i, Math.cos(t) * 0.04, 0.1 + Math.sin(t) * open);
  });
  put(61, -0.05, 0.1);
  put(291, 0.05, 0.1);
  put(13, 0, 0.1 - open);
  put(14, 0, 0.1 + open);
  return pts;
}

const blend = [{ categoryName: 'jawOpen', score: 0.3 }];

test('features: длина вектора совпадает с FEATURE_DIM', () => {
  const f = extractFeatures(face(), blend, 640, 480);
  assert.equal(f.vec.length, FEATURE_DIM);
  assert.equal(FEATURE_GROUPS.length, FEATURE_DIM);
});

test('features: не зависят от сдвига лица в кадре', () => {
  const a = extractFeatures(face(), blend, 640, 480).vec;
  const b = extractFeatures(face({ shift: [0.1, -0.05] }), blend, 640, 480).vec;
  a.forEach((v, k) => assert.ok(Math.abs(v - b[k]) < 1e-6));
});

test('features: не зависят от расстояния до камеры', () => {
  const a = extractFeatures(face(), blend, 640, 480).vec;
  const b = extractFeatures(face({ scale: 0.6 }), blend, 640, 480).vec;
  a.forEach((v, k) => assert.ok(Math.abs(v - b[k]) < 1e-6));
});

test('features: не зависят от наклона головы', () => {
  const a = extractFeatures(face(), blend, 480, 480).vec;
  const b = extractFeatures(face({ angle: 0.2 }), blend, 480, 480).vec;
  a.forEach((v, k) => assert.ok(Math.abs(v - b[k]) < 1e-5));
});

test('features: раскрытый рот даёт большее внутреннее раскрытие', () => {
  const closed = extractFeatures(face({ open: 0.005 }), blend, 640, 480).vec[1];
  const open = extractFeatures(face({ open: 0.04 }), blend, 640, 480).vec[1];
  assert.ok(open > closed);
});

test('features: blendshapes попадают в вектор', () => {
  const f = extractFeatures(face(), [{ categoryName: 'jawOpen', score: 0.77 }], 640, 480).vec;
  assert.ok(f.includes(0.77));
});
