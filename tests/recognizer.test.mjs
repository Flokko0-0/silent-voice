import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Recognizer, dtw } from '../recognizer.js';
import { FEATURE_GROUPS, FEATURE_DIM } from '../features.js';

let seed = 42;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
const proto = (len) => Array.from({ length: len }, () => Array.from({ length: FEATURE_DIM }, rnd));
const variant = (p, len, noise = 0.3) =>
  Array.from({ length: len }, (_, i) => p[Math.floor((i * p.length) / len)].map((v) => v + rnd() * noise));

test('dtw: одинаковые последовательности дают 0', () => {
  const a = proto(20);
  assert.equal(dtw(a, a), 0);
});

test('dtw: симметрична', () => {
  const a = proto(15);
  const b = proto(22);
  assert.ok(Math.abs(dtw(a, b) - dtw(b, a)) < 1e-9);
});

test('dtw: терпит разную скорость произнесения', () => {
  const p = proto(30);
  const slow = variant(p, 45, 0);
  const other = proto(30);
  assert.ok(dtw(p, slow) < dtw(p, other));
});

test('dtw: пустая последовательность даёт Infinity', () => {
  assert.equal(dtw([], proto(5)), Infinity);
});

test('recognizer: узнаёт фразу по вариантам', () => {
  const a = proto(30);
  const b = proto(30);
  const c = proto(30);
  const r = new Recognizer(FEATURE_GROUPS);
  r.fit([
    ...[0, 1, 2].map((i) => ({ label: 'a', seq: variant(a, 28 + i * 4) })),
    ...[0, 1, 2].map((i) => ({ label: 'b', seq: variant(b, 28 + i * 4) })),
    ...[0, 1, 2].map((i) => ({ label: 'c', seq: variant(c, 28 + i * 4) })),
  ]);
  assert.equal(r.classify(variant(b, 35))[0].label, 'b');
});

test('recognizer: результаты отсортированы по расстоянию', () => {
  const r = new Recognizer(FEATURE_GROUPS);
  r.fit(['a', 'b', 'c'].flatMap((l) => [0, 1].map(() => ({ label: l, seq: proto(20) }))));
  const ranked = r.classify(proto(20));
  for (let i = 1; i < ranked.length; i++) assert.ok(ranked[i - 1].dist <= ranked[i].dist);
});

test('recognizer: без образцов возвращает пустой список', () => {
  const r = new Recognizer(FEATURE_GROUPS);
  r.fit([]);
  assert.deepEqual(r.classify(proto(10)), []);
});

test('recognizer: leave-one-out не использует саму запись', () => {
  const a = proto(25);
  const b = proto(25);
  const r = new Recognizer(FEATURE_GROUPS);
  r.fit([
    ...[0, 1, 2].map(() => ({ label: 'a', seq: variant(a, 25) })),
    ...[0, 1, 2].map(() => ({ label: 'b', seq: variant(b, 25) })),
  ]);
  const e = r.evaluate();
  assert.equal(e.total, 6);
  assert.equal(e.correct, 6);
});

test('recognizer: фразы с одной записью не участвуют в проверке точности', () => {
  const r = new Recognizer(FEATURE_GROUPS);
  r.fit([{ label: 'a', seq: proto(20) }, { label: 'b', seq: proto(20) }]);
  assert.equal(r.evaluate().total, 0);
});
