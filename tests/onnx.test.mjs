import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as ort from '../vendor/ort/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

const session = await ort.InferenceSession.create(fs.readFileSync(new URL('../vendor/lipfront.onnx', import.meta.url)), {
  executionProviders: ['wasm'],
});
const run = async (T, fill) => {
  const x = new Float32Array(T * 88 * 88).map((_, i) => fill(i));
  const out = await session.run({ frames: new ort.Tensor('float32', x, [1, T, 1, 88, 88]) });
  return out.features;
};

test('onnx: отпечаток — 768 чисел на кадр при любой длине фразы', async () => {
  for (const T of [12, 37]) assert.deepEqual((await run(T, () => 0)).dims, [T, 768]);
});

test('onnx: результат детерминирован', async () => {
  const a = await run(20, (i) => Math.sin(i * 0.01));
  const b = await run(20, (i) => Math.sin(i * 0.01));
  assert.deepEqual(Array.from(a.data.slice(0, 50)), Array.from(b.data.slice(0, 50)));
});

test('onnx: совпадает с эталоном PyTorch', async () => {
  // эталон посчитан в ml/export_onnx.py тем же входом sin(i*0.001), T=50
  const f = await run(50, (i) => Math.sin(i * 0.001));
  assert.deepEqual(Array.from(f.data.slice(0, 3)).map((v) => v.toFixed(4)), ['-0.0212', '0.0040', '-0.0120']);
});
