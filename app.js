import { FaceLandmarker, FilesetResolver, GestureRecognizer } from './vendor/mediapipe/vision_bundle.mjs';
import { extractFeatures, FEATURE_GROUPS, FEATURE_DIM, ACTIVITY_FLOOR } from './features.js';
import { Recognizer } from './recognizer.js';
import { cropMouth, packFrames } from './mouth-crop.js';
import * as nb from './neural-browser.js';
import { openRoom, newRoomCode, formatRoom } from './relay.js';

const { browserNet } = nb;

const NEURAL_URL = 'http://localhost:5174';
const neural = { available: false };

async function neuralCall(path, body) {
  const res = await fetch(`${NEURAL_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const IS_LOCAL = ['localhost', '127.0.0.1'].includes(location.hostname);
const PRESET_URL = new URL('./preset/', import.meta.url);

async function loadLipsPreset() {
  if (!state.settings.preset || presetLips) return;
  try {
    presetLips = await (await fetch(new URL('lips-ru.json', PRESET_URL))).json();
    refitRecognizer();
    renderPhrases();
    renderPresetNote();
  } catch {
    presetLips = null;
  }
}

function renderPresetNote() {
  const el = document.getElementById('preset-note');
  if (el) el.classList.toggle('hidden', !(state.settings.preset && presetLips && state.lang === 'ru'));
}

// server: python-сервер на ноутбуке (быстрее), browser: та же сеть в ONNX прямо на странице
function neuralMode() {
  if (!state.settings.neural) return null;
  if (neural.available) return 'server';
  if (browserNet.ready) return 'browser';
  return null;
}

async function checkNeural() {
  if (!IS_LOCAL) {
    neural.available = false;
  } else {
    try {
      const res = await fetch(`${NEURAL_URL}/health`);
      neural.available = res.ok;
    } catch {
      neural.available = false;
    }
  }
  if (!neural.available && state.settings.neural && !browserNet.ready) {
    nb.loadModel(() => renderNeural())
      .then(() => (state.settings.preset ? nb.loadPreset(PRESET_URL) : null))
      .then(renderNeural)
      .catch((err) => {
      toast(`Нейросеть не загрузилась: ${err.message}`);
    });
  }
  renderNeural();
}

function renderNeural() {
  const mode = neuralMode();
  const loadingNow = state.settings.neural && !mode && browserNet.progress > 0 && browserNet.progress < 100;
  const el = document.getElementById('neural-status');
  if (el) {
    el.textContent = mode === 'server' ? 'Нейросеть: сервер на ноутбуке'
      : mode === 'browser' ? 'Нейросеть: работает в браузере'
      : loadingNow ? `Нейросеть: загрузка ${browserNet.progress}%`
      : 'Нейросеть: выключена';
  }
  const badge = document.getElementById('neural-badge');
  if (badge) {
    badge.classList.toggle('off', !mode);
    badge.textContent = mode ? 'нейросеть' : loadingNow ? `нейросеть ${browserNet.progress}%` : 'точки губ';
  }
}

const clipPayload = (clip) => ({ frames: packFrames(clip.rois), n: clip.rois.length, times: clip.times, lang: state.lang });

function neuralEnroll(phraseId, clip) {
  if (!clip) return;
  const fail = (err) => toast(`Нейросеть: ${err.message}`);
  if (neural.available) neuralCall('/enroll', { ...clipPayload(clip), label: phraseId }).catch(fail);
  else if (browserNet.ready) nb.enroll(state.lang, phraseId, clip.rois, clip.times).catch(fail);
}

const DEFAULT_PHRASES = [
  { id: 'drink', ru: 'Хочу пить', kk: 'Су ішкім келеді', urgent: false },
  { id: 'pain', ru: 'Мне больно', kk: 'Ауырып тұр', urgent: true },
  { id: 'doctor', ru: 'Позовите врача', kk: 'Дәрігерді шақырыңыз', urgent: true },
  { id: 'breath', ru: 'Трудно дышать', kk: 'Тыныс алу қиын', urgent: true },
  { id: 'cold', ru: 'Мне холодно', kk: 'Тоңып тұрмын', urgent: false },
  { id: 'yes', ru: 'Да', kk: 'Иә', urgent: false },
  { id: 'no', ru: 'Нет', kk: 'Жоқ', urgent: false },
  { id: 'family', ru: 'Позвоните родным', kk: 'Туыстарыма хабарласыңыз', urgent: false },
  { id: 'turn', ru: 'Поверните меня', kk: 'Мені аударыңыз', urgent: false },
  { id: 'thanks', ru: 'Спасибо', kk: 'Рақмет', urgent: false },
];

const GESTURE_PHRASES = {
  Thumb_Up: 'yes',
  Thumb_Down: 'no',
  Open_Palm: 'doctor',
  Closed_Fist: 'pain',
};
const GESTURE_SOURCE = 'жест';
const LIPS_SOURCE = 'по губам';
const GESTURE_MIN_SCORE = 0.65;
const GESTURE_HOLD_MS = 700;
const GESTURE_COOLDOWN_MS = 2500;

const STORAGE_KEY = 'silent-voice-v1';
const TAKES_PER_PHRASE = 3;
const REST_FRAMES = 45;
const PREROLL_FRAMES = 6;
const START_FRAMES = 3;
const MIN_FRAMES = 8;
const MIN_TEMPLATE_FRAMES = 20;
const MAX_FRAMES = 240;
const SMOOTHING = 0.6;

const defaultState = () => ({
  lang: 'ru',
  phrases: structuredClone(DEFAULT_PHRASES),
  templates: {}, // `${phraseId}|${lang}` -> number[][][]
  settings: {
    startThr: 4, endThr: 2.5, endHoldMs: 900, rejectThr: 0, speak: true, debug: false, gestures: true, confirm: true, neural: true, preset: true,
    patient: 'Ерлан Н., 58 лет', ward: 'Реанимация, палата 3',
  },
});

const state = loadState();

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const statusEl = $('status');
const promptEl = $('prompt');
const meterFill = $('meter-fill');
const outputText = $('output-text');
const alternativesEl = $('alternatives');

// ---------- runtime ----------
let landmarker = null;
let gestureRecognizer = null;
const gesture = { name: 'None', since: 0, lastFired: 0, firedName: null };
let lastVideoTime = -1;
let smooth = null;
let rest = null; // { mean: number[], std: number[] }
let restFrames = null; // collecting while non-null
let armed = null; // { phraseId, wizard: boolean }
let wizardQueue = [];
let activity = 0;
const recognizer = new Recognizer(FEATURE_GROUPS);
const seg = { active: false, manual: false, seq: [], preroll: [], loud: 0, quietSince: 0 };

// ---------- persistence ----------
function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      const base = defaultState();
      const loaded = { ...base, ...saved, settings: { ...base.settings, ...saved.settings } };
      // Older builds ended phrases after 550 ms of stillness, which split multi-word phrases.
      if (loaded.settings.endHoldMs < 900) loaded.settings.endHoldMs = 900;
      // Recognizer settings changed, so the old rejection threshold no longer applies.
      loaded.settings.rejectThr = 0;
      for (const k of Object.keys(loaded.templates)) {
        loaded.templates[k] = loaded.templates[k].filter((seq) => seq.length >= MIN_TEMPLATE_FRAMES);
      }
      return loaded;
    }
  } catch {
    // storage blocked or corrupted — start fresh
  }
  return defaultState();
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (err) {
    toast(`Не удалось сохранить калибровку: ${err.message}`);
  }
}

const key = (phraseId, lang = state.lang) => `${phraseId}|${lang}`;
const ownTakes = (phraseId) => state.templates[key(phraseId)] || [];
// Стартовый набор автора подмешивается, пока человек не откалибровал фразу сам.
let presetLips = null;
const presetTakes = (phraseId) =>
  state.settings.preset && presetLips && ownTakes(phraseId).length < 3 ? presetLips[key(phraseId)] || [] : [];
const takes = (phraseId) => [...ownTakes(phraseId), ...presetTakes(phraseId)];
const phraseText = (p) => p[state.lang] || p.ru || p.kk || '';
const phraseById = (id) => state.phrases.find((p) => p.id === id);

function refitRecognizer() {
  const templates = [];
  for (const p of state.phrases) {
    for (const seq of takes(p.id)) templates.push({ label: p.id, seq });
  }
  recognizer.fit(templates);
}

// ---------- camera + model ----------
async function start() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    setStatus('Нет доступа к камере', 'warn');
    showPrompt(`Разрешите доступ к камере и обновите страницу (${err.name})`);
    return;
  }

  setStatus('Загрузка модели…');
  const fileset = await FilesetResolver.forVisionTasks(new URL('./vendor/mediapipe/wasm', import.meta.url).href);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: new URL('./vendor/face_landmarker.task', import.meta.url).href, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  try {
    landmarker = await FaceLandmarker.createFromOptions(fileset, options('GPU'));
  } catch {
    landmarker = await FaceLandmarker.createFromOptions(fileset, options('CPU'));
  }

  const gestureOptions = (delegate) => ({
    baseOptions: { modelAssetPath: new URL('./vendor/gesture_recognizer.task', import.meta.url).href, delegate },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  try {
    gestureRecognizer = await GestureRecognizer.createFromOptions(fileset, gestureOptions('GPU'));
  } catch {
    try {
      gestureRecognizer = await GestureRecognizer.createFromOptions(fileset, gestureOptions('CPU'));
    } catch (err) {
      console.warn('Gesture recognizer unavailable', err);
    }
  }

  refitRecognizer();
  beginRestCalibration();
  requestAnimationFrame(loop);
}

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.clearRect(0, 0, w, h);

  const now = performance.now();
  const result = landmarker.detectForVideo(video, now);
  if (gestureRecognizer && state.settings.gestures) stepGestures(gestureRecognizer.recognizeForVideo(video, now), now, w, h);
  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) {
    setStatus('Лицо не найдено', 'warn');
    if (seg.active && !seg.manual) finishSegment();
    renderMeter(0);
    return;
  }

  const feat = extractFeatures(landmarks, result.faceBlendshapes?.[0]?.categories, w, h);
  if (!feat) return;
  smooth = smooth ? smooth.map((v, k) => v * (1 - SMOOTHING) + feat.vec[k] * SMOOTHING) : feat.vec;

  if (restFrames) {
    restFrames.push(smooth);
    drawLips(feat, '#9fc9b4');
    if (restFrames.length >= REST_FRAMES) finishRestCalibration();
    return;
  }
  if (!rest) return;

  activity = computeActivity(smooth);
  const rel = smooth.map((v, k) => v - rest.mean[k]);
  const roi = neural.available || browserNet.ready ? cropMouth(video, landmarks) : null;
  stepSegmenter(rel, now, roi);
  drawLips(feat, seg.active ? '#f0a020' : '#ffffff');
  drawBrandLips(feat);
  renderMeter(activity);
  if (!armed) setStatus(seg.active ? 'Слушаю губы…' : 'Готов', seg.active ? 'warn' : 'ok');
}

// ---------- hand gestures ----------
function stepGestures(result, now, w, h) {
  const top = result.gestures?.[0]?.[0];
  const hand = result.landmarks?.[0];
  const name = top && top.score >= GESTURE_MIN_SCORE && GESTURE_PHRASES[top.categoryName] ? top.categoryName : 'None';

  if (hand) {
    ctx.fillStyle = name === 'None' ? '#ffffff99' : '#f0a020';
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (name !== gesture.name) {
    gesture.name = name;
    gesture.since = now;
    if (name !== gesture.firedName) gesture.firedName = null;
    return;
  }
  if (name === 'None' || gesture.firedName === name) return;
  const cooldown = pending ? 800 : GESTURE_COOLDOWN_MS;
  if (now - gesture.since < GESTURE_HOLD_MS || now - gesture.lastFired < cooldown) return;

  gesture.lastFired = now;
  gesture.firedName = name;
  // A hand in front of the face would otherwise be read as lip movement.
  seg.active = false;
  seg.seq = [];

  // While a lip-read guess is waiting, thumbs answer the question instead of meaning "Да"/"Нет".
  if (pending && name === 'Thumb_Up') return confirmPending();
  if (pending && name === 'Thumb_Down') return rejectPending();

  const phrase = phraseById(GESTURE_PHRASES[name]);
  if (!phrase) return;
  say(phrase, GESTURE_SOURCE);
}

// ---------- resting mouth ----------
function beginRestCalibration() {
  restFrames = [];
  showPrompt('Сомкните губы и расслабьтесь — 2 секунды');
  setStatus('Калибровка покоя…', 'warn');
}

function finishRestCalibration() {
  const frames = restFrames;
  restFrames = null;
  const mean = new Array(FEATURE_DIM).fill(0);
  const std = new Array(FEATURE_DIM).fill(0);
  for (const f of frames) for (let k = 0; k < FEATURE_DIM; k++) mean[k] += f[k] / frames.length;
  for (const f of frames) for (let k = 0; k < FEATURE_DIM; k++) std[k] += (f[k] - mean[k]) ** 2 / frames.length;
  rest = { mean, std: std.map(Math.sqrt) };
  hidePrompt();
  if (armed) promptForArmed();
  toast('Покой откалиброван');
}

function computeActivity(f) {
  const devs = [];
  for (let k = 0; k < FEATURE_DIM; k++) {
    const floor = ACTIVITY_FLOOR[k];
    if (floor === Infinity) continue;
    devs.push(Math.abs(f[k] - rest.mean[k]) / Math.max(rest.std[k] * 3, floor));
  }
  devs.sort((a, b) => b - a);
  return (devs[0] + devs[1] + devs[2] + devs[3] + devs[4]) / 5;
}

// ---------- utterance segmentation ----------
function stepSegmenter(rel, now, roi = null) {
  const { startThr, endThr, endHoldMs } = state.settings;
  if (!seg.active) {
    seg.preroll.push({ rel, act: activity, roi, t: now });
    if (seg.preroll.length > PREROLL_FRAMES) seg.preroll.shift();
    seg.loud = activity > startThr ? seg.loud + 1 : 0;
    if (seg.manual || seg.loud >= START_FRAMES) {
      seg.active = true;
      seg.seq = [...seg.preroll];
      seg.quietSince = 0;
    }
    return;
  }
  seg.seq.push({ rel, act: activity, roi, t: now });
  if (seg.manual) return;
  if (activity < endThr) {
    if (!seg.quietSince) seg.quietSince = now;
    if (now - seg.quietSince >= endHoldMs) finishSegment();
  } else {
    seg.quietSince = 0;
  }
  if (seg.seq.length >= MAX_FRAMES) finishSegment();
}

function trimToMotion(frames) {
  const endThr = state.settings.endThr;
  let first = frames.findIndex((f) => f.act >= endThr);
  let last = frames.length - 1;
  while (last > 0 && frames[last].act < endThr) last--;
  if (first < 0) return frames;
  first = Math.max(0, first - 2);
  last = Math.min(frames.length - 1, last + 3);
  return frames.slice(first, last + 1);
}

function finishSegment() {
  const frames = trimToMotion(seg.seq);
  seg.active = false;
  seg.seq = [];
  seg.preroll = [];
  seg.loud = 0;
  seg.quietSince = 0;
  if (frames.length < MIN_FRAMES) return;
  const clip = frames.every((f) => f.roi) ? { rois: frames.map((f) => f.roi), times: frames.map((f) => f.t) } : null;
  onUtterance(frames.map((f) => f.rel.map((v) => Math.round(v * 1e4) / 1e4)), clip);
}

// ---------- what to do with an utterance ----------
function onUtterance(seq, clip) {
  if (armed) {
    recordTake(armed.phraseId, seq, clip);
    return;
  }
  recognize(seq, clip);
}

function recordTake(phraseId, seq, clip) {
  if (seq.length < MIN_TEMPLATE_FRAMES) {
    toast('Запись оборвалась — скажите фразу целиком ещё раз');
    return;
  }
  neuralEnroll(phraseId, clip);
  const k = key(phraseId);
  state.templates[k] = [...(state.templates[k] || []), seq];
  saveState();
  refitRecognizer();
  renderPhrases();
  const n = ownTakes(phraseId).length;
  toast(`Записано: «${phraseText(phraseById(phraseId))}» (${n})`);

  if (armed.wizard) {
    if (n % TAKES_PER_PHRASE !== 0 || n < TAKES_PER_PHRASE) {
      promptForArmed();
      return;
    }
    const next = wizardQueue.shift();
    if (next) {
      armed = { phraseId: next, wizard: true };
      promptForArmed();
    } else {
      armed = null;
      hidePrompt();
      toast('Калибровка завершена — переходите в «Разговор»');
      evaluate();
    }
  } else {
    armed = null;
    hidePrompt();
  }
  renderPhrases();
}

function promptForArmed() {
  const p = phraseById(armed.phraseId);
  const n = ownTakes(p.id).length;
  const progress = armed.wizard ? ` (${(n % TAKES_PER_PHRASE) + 1}/${TAKES_PER_PHRASE})` : '';
  showPrompt(`Беззвучно скажите: «${phraseText(p)}»${progress}`);
  setStatus('Запись образца', 'warn');
}

async function recognize(seq, clip) {
  const available = new Set(state.phrases.filter((p) => takes(p.id).length).map((p) => p.id));
  if (available.size < 2) {
    showOutput('Сначала откалибруйте хотя бы 2 фразы', { muted: true });
    return;
  }
  let ranked = null;
  let via = '';
  const mode = neuralMode();
  if (mode && clip) {
    try {
      if (mode === 'browser') setStatus('Нейросеть думает…', 'warn');
      const res = mode === 'server'
        ? await neuralCall('/classify', clipPayload(clip))
        : await nb.classify(state.lang, clip.rois, clip.times);
      ranked = res.ranked.filter((r) => phraseById(r.label));
      via = `нейросеть, ${res.ms} мс`;
    } catch (err) {
      toast(`Нейросеть: ${err.message} — использую точки губ`);
    }
  }
  if (!ranked?.length) {
    ranked = recognizer.classify(seq);
    via = 'точки губ';
  }
  if (!ranked.length) return;
  const [best, second] = ranked;
  const margin = second ? second.dist / best.dist : Infinity;
  const { rejectThr, debug, confirm } = state.settings;
  const confident = (!rejectThr || best.dist <= rejectThr) && margin >= 1.04;
  const debugLine = debug
    ? `[${via}] ` + ranked.slice(0, 3).map((r) => `${phraseText(phraseById(r.label))}: ${r.dist.toFixed(3)}`).join(' · ')
    : '';

  if (confirm) {
    askToConfirm(ranked.slice(0, 3).map((r) => phraseById(r.label)), seq, debugLine, clip);
    return;
  }
  if (confident) {
    const phrase = phraseById(best.label);
    say(phrase);
    renderAlternatives(ranked.slice(1, 3), seq, 'Не то? Выберите:', debugLine);
  } else {
    showOutput('Не расслышал… Вы сказали:', { muted: true });
    renderAlternatives(ranked.slice(0, 3), seq, '', debugLine);
  }
}

// ---------- confirmation (SRAVI-style) ----------
let pending = null; // { candidates, idx, seq, timer }
const CONFIRM_TIMEOUT_MS = 15000;

function askToConfirm(candidates, seq, debugLine = '', clip = null) {
  cancelPending();
  pending = { candidates, idx: 0, seq, clip, debugLine, timer: 0 };
  renderPending();
}

function renderPending() {
  clearTimeout(pending.timer);
  pending.timer = setTimeout(() => {
    cancelPending();
    showOutput(state.lang === 'kk' ? 'Қайталаңыз' : 'Повторите, пожалуйста', { muted: true });
  }, CONFIRM_TIMEOUT_MS);

  const phrase = pending.candidates[pending.idx];
  outputText.textContent = `${phraseText(phrase)}?`;
  $('output-source').textContent = 'нужно подтверждение';
  outputText.classList.remove('urgent', 'muted');
  outputText.classList.add('question');
  alternativesEl.innerHTML = '';
  const help = document.createElement('span');
  help.className = 'alt-label';
  help.textContent = 'Палец вверх — да, палец вниз — другой вариант, или выберите:';
  alternativesEl.append(help);
  pending.candidates.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = 'chip' + (i === pending.idx ? ' current' : '');
    btn.textContent = phraseText(p);
    btn.onclick = () => {
      pending.idx = i;
      confirmPending();
    };
    alternativesEl.append(btn);
  });
  if (pending.debugLine) {
    const d = document.createElement('div');
    d.className = 'alt-label';
    d.style.width = '100%';
    d.textContent = pending.debugLine;
    alternativesEl.append(d);
  }
}

function confirmPending() {
  const { candidates, idx, seq, clip } = pending;
  const phrase = candidates[idx];
  cancelPending();
  neuralEnroll(phrase.id, clip);
  state.templates[key(phrase.id)] = [...ownTakes(phrase.id), seq];
  saveState();
  refitRecognizer();
  renderPhrases();
  say(phrase);
}

function rejectPending() {
  pending.idx++;
  if (pending.idx >= pending.candidates.length) {
    cancelPending();
    showOutput(state.lang === 'kk' ? 'Түсінбедім. Қайталаңыз' : 'Не понял. Повторите, пожалуйста', { muted: true });
    return;
  }
  renderPending();
}

function cancelPending() {
  if (pending) clearTimeout(pending.timer);
  pending = null;
  outputText.classList.remove('question');
}

function say(phrase, via = LIPS_SOURCE) {
  cancelPending();
  tourEvent(via === GESTURE_SOURCE ? 'gesture' : 'lips');
  const text = phraseText(phrase);
  showOutput(text, { urgent: phrase.urgent, source: via });
  addHistory(text, phrase.urgent, via);
  if (phrase.urgent) raiseAlert(text);
  speak(text);
  notifyNurse(text, phrase.urgent, via);
  if (phrase.urgent) watchUrgent(text, via);
}

// ---------- nurse station link ----------
// server: локальный сервер на ноутбуке (Wi-Fi), cloud: интернет по коду палаты, со сквозным шифрованием
let relayMode = 'server';
let room = null;
const ESCALATE_MS = 60000;
const ESCALATE_MAX = 3;
let urgentWatch = null;

function ensureRoomCode() {
  if (!/^\d{8}$/.test(state.settings.room || '')) {
    state.settings.room = newRoomCode();
    saveState();
  }
  return state.settings.room;
}

function notifyNurse(text, urgent, via, extra = {}) {
  const { patient, ward } = state.settings;
  const msg = { text, urgent, via, patient, ward, ...extra };
  if (relayMode === 'cloud') {
    const ev = { type: 'message', id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, at: Date.now(), ...msg };
    (room ? room.send(ev) : Promise.reject()).catch(() => toast('Нет связи с постом медсестры'));
    return;
  }
  fetch('/api/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(msg),
  }).catch(() => toast('Пост медсестры недоступен'));
}

// Срочный вызов без ответа повторяется каждую минуту (до трёх раз).
function watchUrgent(text, via) {
  clearInterval(urgentWatch);
  let attempt = 0;
  urgentWatch = setInterval(() => {
    attempt++;
    if (attempt > ESCALATE_MAX) {
      clearInterval(urgentWatch);
      return;
    }
    notifyNurse(text, true, via, { repeat: attempt });
    beep();
    toast(`Медсестра не ответила. Повторный вызов ${attempt} из ${ESCALATE_MAX}`);
  }, ESCALATE_MS);
}

function onNurseAck(ev) {
  clearInterval(urgentWatch);
  $('alert').classList.add('hidden');
  const ackEl = $('nurse-ack');
  ackEl.textContent = `${ev.by || 'Медсестра'} идёт к вам`;
  ackEl.classList.remove('hidden');
  speak(state.lang === 'kk' ? 'Мейірбике келе жатыр' : 'Медсестра идёт к вам');
  setTimeout(() => ackEl.classList.add('hidden'), 8000);
}

async function connectRoom() {
  room?.close();
  room = await openRoom(ensureRoomCode(), (ev) => ev.type === 'ack' && onNurseAck(ev));
  showNurseUrls();
}

function listenForNurse() {
  if (relayMode === 'cloud') {
    connectRoom().catch((err) => toast(`Связь: ${err.message}`));
    return;
  }
  const es = new EventSource('/events');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'ack') onNurseAck(ev);
  };
}

async function showNurseUrls() {
  const el = $('nurse-urls');
  if (relayMode === 'server') {
    try {
      const res = await fetch('/api/info');
      if (!res.ok) throw new Error(res.statusText);
      const { nurseUrls } = await res.json();
      el.textContent = nurseUrls.length ? nurseUrls.join('\n') : 'Нет сети — подключитесь к Wi-Fi или точке доступа';
      return;
    } catch {
      relayMode = 'cloud';
    }
  }
  const code = ensureRoomCode();
  const url = new URL(`nurse.html#room=${code}`, location.href).href;
  el.innerHTML = '';
  const b = document.createElement('b');
  b.textContent = `Код палаты: ${formatRoom(code)}`;
  const a = document.createElement('a');
  a.href = url;
  a.target = '_blank';
  a.textContent = url;
  el.append(b, document.createTextNode('\nОткройте на телефоне:\n'), a, document.createTextNode('\nСообщения шифруются кодом палаты.'));
}

function renderAlternatives(options, seq, label, debugLine) {
  alternativesEl.innerHTML = '';
  if (label && options.length) {
    const span = document.createElement('span');
    span.className = 'alt-label';
    span.textContent = label;
    alternativesEl.append(span);
  }
  for (const opt of options) {
    const phrase = phraseById(opt.label);
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = phraseText(phrase);
    btn.onclick = () => {
      state.templates[key(phrase.id)] = [...ownTakes(phrase.id), seq];
      saveState();
      refitRecognizer();
      renderPhrases();
      alternativesEl.innerHTML = '';
      say(phrase);
    };
    alternativesEl.append(btn);
  }
  if (debugLine) {
    const d = document.createElement('div');
    d.className = 'alt-label';
    d.style.width = '100%';
    d.textContent = debugLine;
    alternativesEl.append(d);
  }
}

// ---------- voice + alert ----------
function speak(text) {
  if (!state.settings.speak || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const voices = speechSynthesis.getVoices();
  const want = state.lang === 'kk' ? 'kk' : 'ru';
  const voice =
    voices.find((v) => v.lang.toLowerCase().startsWith(want)) ||
    voices.find((v) => v.lang.toLowerCase().startsWith('ru')) ||
    null;
  if (voice) u.voice = voice;
  u.lang = voice?.lang || (want === 'kk' ? 'kk-KZ' : 'ru-RU');
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

let audioCtx = null;
function beep() {
  try {
    audioCtx ??= new AudioContext();
    for (let i = 0; i < 3; i++) {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.frequency.value = 880;
      osc.connect(gain).connect(audioCtx.destination);
      const t = audioCtx.currentTime + i * 0.35;
      gain.gain.setValueAtTime(0.25, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      osc.start(t);
      osc.stop(t + 0.26);
    }
  } catch {
    // audio blocked — the visual alert is still shown
  }
}

function raiseAlert(text) {
  $('alert-text').textContent = text;
  $('alert').classList.remove('hidden');
  beep();
}
$('alert-close').onclick = () => $('alert').classList.add('hidden');

// ---------- evaluation ----------
async function evaluate() {
  const mode = neuralMode();
  if (mode) {
    try {
      const r = mode === 'server' ? await neuralCall('/evaluate', { lang: state.lang }) : nb.evaluate(state.lang);
      const el = $('neural-eval');
      el.textContent = r.total
        ? `Нейросеть: ${Math.round((100 * r.correct) / r.total)}% (${r.correct} из ${r.total})`
        : 'Нейросеть: пока мало записей (нужно по 2 на фразу)';
    } catch (err) {
      toast(`Нейросеть: ${err.message}`);
    }
  }
  const { correct, total, correctDists, mistakes } = recognizer.evaluate();
  const el = $('eval-result');
  if (!total) {
    el.textContent = 'Нужно минимум по 2 записи у двух фраз.';
    return;
  }
  const pct = Math.round((correct / total) * 100);
  const confusions = mistakes
    .map((m) => `«${phraseText(phraseById(m.expected))}» → «${phraseText(phraseById(m.got))}»`)
    .join('<br>');
  el.innerHTML = `<div class="big">${pct}%</div>
    <div>Точность (каждая запись проверена по остальным): ${correct} из ${total}</div>
    ${confusions ? `<div class="muted" style="margin-top:8px">Путает:<br>${confusions}</div>` : ''}`;
  if (correctDists.length) {
    // Suggest a rejection threshold just above the worst correct match.
    const suggested = Math.max(...correctDists) * 1.3;
    state.settings.rejectThr = Math.round(suggested * 100) / 100;
    saveState();
    renderSettings();
  }
}

// ---------- rendering ----------
function drawLips(feat, color) {
  const poly = (pts) => {
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
  };
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  poly(feat.outer);
  ctx.stroke();
  ctx.lineWidth = 1.5;
  poly(feat.inner);
  ctx.stroke();
}

// Логотип в шапке повторяет контур губ человека перед камерой.
const brandCanvas = $('brand-lips');
const brandCtx = brandCanvas.getContext('2d');
function drawBrandLips(feat) {
  const { width: W, height: H } = brandCanvas;
  brandCtx.clearRect(0, 0, W, H);
  const xs = feat.outer.map((p) => p[0]);
  const ys = feat.outer.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const k = (W - 8) / (maxX - minX || 1);
  const map = ([x, y]) => [W - 4 - (x - minX) * k, H / 2 + (y - midY) * k];
  const poly = (pts) => {
    brandCtx.beginPath();
    pts.map(map).forEach(([x, y], i) => (i ? brandCtx.lineTo(x, y) : brandCtx.moveTo(x, y)));
    brandCtx.closePath();
  };
  brandCtx.strokeStyle = '#f3f1ec';
  brandCtx.lineJoin = 'round';
  brandCtx.lineWidth = 3;
  poly(feat.outer);
  brandCtx.stroke();
  brandCtx.lineWidth = 2;
  poly(feat.inner);
  brandCtx.stroke();
}

// До включения камеры — спокойные сомкнутые губы.
(function drawBrandIdle() {
  const t = [-1, -0.7, -0.4, -0.15, 0, 0.15, 0.4, 0.7, 1];
  const top = t.map((x) => [x, -0.3 * Math.sqrt(1 - x * x) + 0.08 * Math.exp(-((x / 0.15) ** 2))]);
  const bottom = t.slice(1, -1).reverse().map((x) => [x, 0.35 * Math.sqrt(1 - x * x)]);
  const line = t.map((x) => [x, 0.02 * (1 - x * x)]);
  drawBrandLips({ outer: [...top, ...bottom], inner: [...line, ...line.slice().reverse()] });
})();

function renderMeter(value) {
  const max = state.settings.startThr * 2;
  meterFill.style.width = `${Math.min(value / max, 1) * 100}%`;
  meterFill.classList.toggle('active', seg.active);
  $('meter-start').style.left = '50%';
  $('meter-end').style.left = `${(state.settings.endThr / max) * 100}%`;
}

function showOutput(text, { urgent = false, muted = false, source = '' } = {}) {
  outputText.textContent = text;
  $('output-source').textContent = source;
  outputText.classList.toggle('urgent', urgent);
  outputText.classList.toggle('muted', muted);
  alternativesEl.innerHTML = '';
}

function addHistory(text, urgent, source = '') {
  const list = $('history');
  if (list.querySelector('.muted')) list.innerHTML = '';
  const li = document.createElement('li');
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const span = document.createElement('span');
  span.textContent = text;
  if (urgent) span.className = 'urgent';
  const src = document.createElement('small');
  src.textContent = source;
  li.append(time, span, src);
  list.prepend(li);
}

function renderPhrases() {
  const list = $('phrases');
  list.innerHTML = '';
  for (const p of state.phrases) {
    const n = ownTakes(p.id).length;
    const extra = presetTakes(p.id).length;
    const li = document.createElement('li');
    li.className = 'phrase' + (armed?.phraseId === p.id ? ' armed' : '');

    const urgent = document.createElement('button');
    urgent.className = 'urgent-toggle' + (p.urgent ? ' on' : '');
    urgent.title = 'Срочная фраза (тревога)';
    urgent.textContent = 'срочно';
    urgent.onclick = () => {
      p.urgent = !p.urgent;
      saveState();
      renderPhrases();
    };

    const input = document.createElement('input');
    input.type = 'text';
    input.value = phraseText(p);
    input.onchange = () => {
      p[state.lang] = input.value.trim();
      saveState();
    };

    const count = document.createElement('span');
    count.className = 'count' + (n >= TAKES_PER_PHRASE ? ' ready' : '');
    count.textContent = extra ? `${n} + ${extra}` : `${n}`;
    if (extra) count.title = `${extra} образцов из стартового набора автора`;

    const rec = document.createElement('button');
    rec.className = 'secondary' + (armed?.phraseId === p.id ? ' active' : '');
    rec.textContent = armed?.phraseId === p.id ? 'Жду…' : 'Записать';
    rec.onclick = () => {
      armed = armed?.phraseId === p.id ? null : { phraseId: p.id, wizard: false };
      if (armed) promptForArmed();
      else hidePrompt();
      renderPhrases();
    };

    const del = document.createElement('button');
    del.className = 'icon-btn';
    del.title = 'Удалить записи этой фразы';
    del.textContent = 'удалить';
    del.onclick = () => {
      if (n && !confirm(`Удалить ${n} записей «${phraseText(p)}»?`)) return;
      if (!n && !confirm(`Удалить фразу «${phraseText(p)}»?`)) return;
      if (n) {
        delete state.templates[key(p.id)];
        if (neural.available) neuralCall('/clear', { lang: state.lang, label: p.id }).catch(() => {});
        nb.clear(state.lang, p.id).catch(() => {});
      }
      else state.phrases = state.phrases.filter((x) => x !== p);
      saveState();
      refitRecognizer();
      renderPhrases();
    };

    li.append(urgent, input, count, rec, del);
    list.append(li);
  }
}

// Браслет: код палаты и штрихкод-рисунок, собранный из его цифр.
function renderBand() {
  const code = ensureRoomCode();
  $('band-room').textContent = formatRoom(code);
  let seed = Number(code) + 7;
  const rnd = () => ((seed = (seed * 48271) % 2147483647) / 2147483647);
  let x = 2;
  let bars = '';
  while (x < 118) {
    const w = 1 + Math.floor(rnd() * 3);
    bars += `<rect x="${x}" y="0" width="${w}" height="30" fill="#1a1a18"/>`;
    x += w + 1 + Math.floor(rnd() * 2);
  }
  $('band-code').innerHTML = bars;
}

function renderSettings() {
  renderBand();
  const s = state.settings;
  $('start-thr').value = s.startThr;
  $('end-thr').value = s.endThr;
  $('end-hold').value = s.endHoldMs;
  $('reject-thr').value = s.rejectThr;
  $('start-val').textContent = s.startThr.toFixed(1);
  $('end-val').textContent = s.endThr.toFixed(1);
  $('hold-val').textContent = s.endHoldMs;
  $('reject-val').textContent = s.rejectThr ? s.rejectThr.toFixed(2) : 'выкл';
  $('speak-toggle').checked = s.speak;
  $('patient-name').value = s.patient;
  $('pin-input').value = s.pin || '';
  $('patient-ward').value = s.ward;
  $('patient-chip-name').textContent = s.patient || 'Пациент';
  $('patient-chip-ward').textContent = s.ward;
  $('gesture-toggle').checked = s.gestures;
  $('confirm-toggle').checked = s.confirm;
  $('neural-toggle').checked = s.neural;
  $('debug-toggle').checked = s.debug;
}

function setStatus(text, kind = '') {
  if (statusEl.textContent === text) return;
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}
function showPrompt(text) {
  promptEl.textContent = text;
  promptEl.classList.remove('hidden');
}
function hidePrompt() {
  promptEl.classList.add('hidden');
}
let toastTimer = 0;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 2200);
}

// ---------- controls ----------
// PIN персонала: калибровку и настройки нельзя изменить случайно (пациентом или посетителем).
let unlockedUntil = 0;
function staffUnlocked() {
  const pin = state.settings.pin;
  if (!pin || Date.now() < unlockedUntil) return true;
  const entered = prompt('PIN персонала');
  if (entered === pin) {
    unlockedUntil = Date.now() + 5 * 60 * 1000;
    return true;
  }
  if (entered !== null) toast('Неверный PIN');
  return false;
}

document.querySelectorAll('.tabs button').forEach((btn) => {
  btn.onclick = () => {
    if (btn.dataset.tab !== 'talk' && !staffUnlocked()) return;
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('[data-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== btn.dataset.tab));
  };
});

document.querySelectorAll('.lang-switch button').forEach((btn) => {
  btn.classList.toggle('active', btn.dataset.lang === state.lang);
  btn.onclick = () => {
    state.lang = btn.dataset.lang;
    renderPresetNote();
    document.querySelectorAll('.lang-switch button').forEach((b) => b.classList.toggle('active', b === btn));
    saveState();
    refitRecognizer();
    renderPhrases();
    showOutput(state.lang === 'kk' ? 'Сөзді дыбыссыз айтыңыз — тек ернімен' : 'Скажите фразу беззвучно — только губами', { muted: true });
  };
});

$('wizard-btn').onclick = () => {
  wizardQueue = state.phrases.filter((p) => ownTakes(p.id).length < TAKES_PER_PHRASE).map((p) => p.id);
  const first = wizardQueue.shift();
  if (!first) {
    toast('Все фразы уже откалиброваны');
    return;
  }
  armed = { phraseId: first, wizard: true };
  promptForArmed();
  renderPhrases();
};
$('rest-btn').onclick = beginRestCalibration;
$('eval-btn').onclick = evaluate;

$('add-phrase').onclick = () => {
  const text = $('new-phrase').value.trim();
  if (!text) return;
  state.phrases.push({ id: `p${Date.now()}`, ru: state.lang === 'ru' ? text : '', kk: state.lang === 'kk' ? text : '', urgent: false });
  $('new-phrase').value = '';
  saveState();
  renderPhrases();
};

const bindRange = (id, field, parse = Number) => {
  $(id).oninput = (e) => {
    state.settings[field] = parse(e.target.value);
    saveState();
    renderSettings();
  };
};
bindRange('start-thr', 'startThr');
bindRange('end-thr', 'endThr');
bindRange('end-hold', 'endHoldMs');
bindRange('reject-thr', 'rejectThr');
$('speak-toggle').onchange = (e) => {
  state.settings.speak = e.target.checked;
  saveState();
};
for (const [id, field] of [['patient-name', 'patient'], ['patient-ward', 'ward']]) {
  $(id).onchange = (e) => {
    state.settings[field] = e.target.value.trim();
    saveState();
    renderSettings();
  };
}
$('neural-toggle').onchange = (e) => {
  state.settings.neural = e.target.checked;
  saveState();
  checkNeural();
};
$('confirm-toggle').onchange = (e) => {
  state.settings.confirm = e.target.checked;
  saveState();
};
$('gesture-toggle').onchange = (e) => {
  state.settings.gestures = e.target.checked;
  saveState();
};
$('debug-toggle').onchange = (e) => {
  state.settings.debug = e.target.checked;
  saveState();
};

$('export-btn').onclick = () => {
  const blob = new Blob([JSON.stringify(state)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `silent-voice-calibration-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
};
$('import-input').onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    Object.assign(state, defaultState(), data, { settings: { ...defaultState().settings, ...data.settings } });
    saveState();
    refitRecognizer();
    renderPhrases();
    renderSettings();
    toast('Калибровка загружена');
  } catch (err) {
    toast(`Не удалось прочитать файл: ${err.message}`);
  }
};
$('pin-input').onchange = (e) => {
  state.settings.pin = e.target.value.replace(/\D/g, '').slice(0, 8);
  e.target.value = state.settings.pin;
  saveState();
  toast(state.settings.pin ? 'PIN установлен' : 'Защита PIN выключена');
};
$('discharge-btn').onclick = async () => {
  if (!confirm('Выписка: удалить с этого устройства все записи, журнал и данные пациента?')) return;
  state.templates = {};
  state.settings.patient = 'Пациент';
  state.settings.room = '';
  if (neural.available) for (const lang of ['ru', 'kk']) neuralCall('/clear', { lang }).catch(() => {});
  await nb.clear(null).catch(() => {});
  saveState();
  refitRecognizer();
  renderPhrases();
  renderSettings();
  $('history').innerHTML = '<li class="muted">Пока пусто</li>';
  $('eval-result').textContent = '';
  $('neural-eval').textContent = '';
  if (relayMode === 'cloud') connectRoom().catch(() => {});
  toast('Данные пациента удалены, выдан новый код палаты');
};
$('reset-btn').onclick = () => {
  if (!confirm('Удалить все записанные образцы?')) return;
  state.templates = {};
  if (neural.available) neuralCall('/clear', { lang: state.lang }).catch(() => {});
  nb.clear(state.lang).catch(() => {});
  saveState();
  refitRecognizer();
  renderPhrases();
  $('eval-result').textContent = '';
};

const holdStart = () => {
  if (!rest || seg.manual) return;
  seg.manual = true;
  $('hold-btn').classList.add('active');
};
const holdEnd = () => {
  if (!seg.manual) return;
  seg.manual = false;
  $('hold-btn').classList.remove('active');
  if (seg.active) finishSegment();
};
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    holdStart();
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space') holdEnd();
});
$('hold-btn').addEventListener('pointerdown', holdStart);
$('hold-btn').addEventListener('pointerup', holdEnd);
$('hold-btn').addEventListener('pointerleave', holdEnd);

// Chrome loads voices asynchronously; touching the list early makes them available on first speak.
if ('speechSynthesis' in window) speechSynthesis.getVoices();

$('preset-off').onclick = () => {
  if (!confirm('Убрать стартовый набор автора? Останутся только ваши записи.')) return;
  state.settings.preset = false;
  saveState();
  nb.dropPreset();
  refitRecognizer();
  renderPhrases();
  renderPresetNote();
};

// ---------- обучение ----------
// Запускается только по кнопке. Шаги с действием засчитываются сами, когда человек его выполнил.
const TOUR_KEY = 'silent-voice-tour-seen';
const TOUR = [
  {
    target: '.stage',
    title: 'Камера',
    text: 'Сядьте лицом к свету, чтобы лицо было в кадре. Две секунды держите губы сомкнутыми — система запомнит их в покое. Контур губ появится на видео.',
  },
  {
    target: '.gestures',
    title: 'Жест рукой',
    text: 'Покажите открытую ладонь и подержите секунду. Сработает «Позовите врача» и тревога. Жесты работают у любого человека без настройки.',
    waitFor: 'gesture',
  },
  {
    target: '.output',
    title: 'Фраза губами',
    text: 'Беззвучно и чётко скажите губами «Хочу пить» и замолчите. Система переспросит — покажите большой палец вверх, и фраза прозвучит вслух.',
    waitFor: 'lips',
  },
  {
    target: '#nurse-urls',
    title: 'Пост медсестры',
    text: 'Откройте эту ссылку на телефоне. Срочные фразы придут туда со звуком, а ответ «Иду» вернётся на этот экран.',
    tab: 'talk',
  },
  {
    target: '.tabs button[data-tab="calib"]',
    title: 'Точнее под себя',
    text: 'Сейчас работает стартовый набор автора. На вкладке «Калибровка» запишите свои фразы по 3 раза — около двух минут, — и точность вырастет.',
  },
];
let tourStep = -1;
let tourTimer = 0;

function markSeen() {
  try {
    localStorage.setItem(TOUR_KEY, '1');
  } catch {
    // без хранилища приглашение просто покажется снова
  }
  $('tour-invite').classList.add('hidden');
}

function showTourStep(i) {
  document.querySelectorAll('.tour-target').forEach((el) => el.classList.remove('tour-target'));
  clearTimeout(tourTimer);
  tourStep = i;
  if (i < 0 || i >= TOUR.length) {
    $('tour').classList.add('hidden');
    tourStep = -1;
    return;
  }
  const step = TOUR[i];
  if (step.tab) document.querySelector(`.tabs button[data-tab="${step.tab}"]`)?.click();
  $('tour-step').textContent = `шаг ${i + 1} из ${TOUR.length}`;
  $('tour-title').textContent = step.title;
  $('tour-text').textContent = step.text;
  $('tour-done').classList.add('hidden');
  $('tour-back').classList.toggle('hidden', i === 0);
  $('tour-next').textContent = i === TOUR.length - 1 ? 'Готово' : step.waitFor ? 'Пропустить' : 'Дальше';
  const target = document.querySelector(step.target);
  if (target) {
    target.classList.add('tour-target');
    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  $('tour').classList.remove('hidden');
}

function tourEvent(kind) {
  if (tourStep < 0 || TOUR[tourStep].waitFor !== kind) return;
  $('tour-done').classList.remove('hidden');
  $('tour-next').textContent = 'Дальше';
  tourTimer = setTimeout(() => showTourStep(tourStep + 1), 1800);
}

function startTour() {
  markSeen();
  showTourStep(0);
}

$('guide-open').onclick = startTour;
$('tour-invite-start').onclick = startTour;
$('tour-invite-close').onclick = markSeen;
$('tour-next').onclick = () => showTourStep(tourStep + 1);
$('tour-back').onclick = () => showTourStep(tourStep - 1);
$('tour-skip').onclick = () => showTourStep(-1);
try {
  if (!localStorage.getItem(TOUR_KEY)) $('tour-invite').classList.remove('hidden');
} catch {
  $('tour-invite').classList.remove('hidden');
}

renderPhrases();
renderSettings();
renderPresetNote();
loadLipsPreset();
showNurseUrls().then(listenForNurse);
checkNeural();
if (IS_LOCAL) setInterval(checkNeural, 5000);
start().catch((err) => {
  console.error(err);
  setStatus('Ошибка запуска', 'warn');
  showPrompt(`Ошибка: ${err.message}`);
});
