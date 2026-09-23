// Нейросеть чтения по губам в браузере. Сама работа идёт в фоновом потоке (neural-worker.js),
// чтобы видео и интерфейс не замирали, пока модель загружается и думает.
export const browserNet = { ready: false, progress: 0 };

const CROP = 96;
let worker = null;
let nextId = 1;
const pending = new Map();
let onProgress = () => {};
let loading = null;

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./neural-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'progress') {
      browserNet.progress = msg.value;
      onProgress(msg.value);
      return;
    }
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error));
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || 'ошибка фонового потока'));
    pending.clear();
  };
  return worker;
}

function call(cmd, args = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, cmd, args }, transfer);
  });
}

function pack(rois) {
  const bytes = new Uint8Array(rois.length * CROP * CROP);
  rois.forEach((f, i) => bytes.set(f, i * CROP * CROP));
  return bytes;
}

export function loadModel(progress = () => {}) {
  onProgress = progress;
  loading ??= call('load').then(() => {
    browserNet.ready = true;
    browserNet.progress = 100;
    onProgress(100);
  });
  return loading;
}

export function enroll(lang, label, rois, times) {
  const bytes = pack(rois);
  return call('enroll', { lang, label, bytes, times }, [bytes.buffer]);
}

export function classify(lang, rois, times) {
  const bytes = pack(rois);
  return call('classify', { lang, bytes, times }, [bytes.buffer]);
}

export const evaluate = (lang) => call('evaluate', { lang });
export const clear = (lang, label = null) => call('clear', { lang, label });
export const loadPreset = (base) => call('loadPreset', { base: String(base) });
export const dropPreset = () => call('dropPreset').catch(() => {});
