// Связь пациент ↔ пост медсестры через интернет по коду палаты.
// Сообщения шифруются AES-GCM ключом из кода палаты, посредник (ntfy.sh) видит только шифр.
const RELAY = 'https://ntfy.sh';
const enc = new TextEncoder();
const dec = new TextDecoder();

export function newRoomCode() {
  const a = new Uint32Array(1);
  crypto.getRandomValues(a);
  return String(a[0] % 1e8).padStart(8, '0');
}

export function formatRoom(code) {
  return `${code.slice(0, 4)}-${code.slice(4)}`;
}

async function sha256hex(text) {
  const h = await crypto.subtle.digest('SHA-256', enc.encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function topicFor(code) {
  return 'sv-' + (await sha256hex('silent-voice/topic/' + code)).slice(0, 32);
}

export async function deriveKey(code) {
  const base = await crypto.subtle.importKey('raw', enc.encode(code), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: enc.encode('silent-voice/room'), iterations: 200000, hash: 'SHA-256' },
    base,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

const b64 = (bytes) => btoa(String.fromCharCode(...bytes));
const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export async function seal(key, event) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, enc.encode(JSON.stringify(event))));
  const body = new Uint8Array(12 + ct.length);
  body.set(iv);
  body.set(ct, 12);
  return b64(body);
}

export async function unseal(key, text) {
  const raw = unb64(text);
  const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: raw.slice(0, 12) }, key, raw.slice(12));
  return JSON.parse(dec.decode(plain));
}

export async function openRoom(code, onEvent, onStatus = () => {}) {
  const clean = String(code).replace(/\D/g, '');
  const topic = await topicFor(clean);
  const key = await deriveKey(clean);
  const since = Math.floor(Date.now() / 1000) - 60;
  const es = new EventSource(`${RELAY}/${topic}/sse?since=${since}`);
  es.onopen = () => onStatus(true);
  es.onerror = () => onStatus(false);
  es.onmessage = async (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.event !== 'message') return;
      onEvent(await unseal(key, msg.message));
    } catch {
      // чужое или повреждённое сообщение — пропускаем
    }
  };
  return {
    async send(event) {
      await fetch(`${RELAY}/${topic}`, { method: 'POST', body: await seal(key, event) });
    },
    close() {
      es.close();
    },
  };
}
