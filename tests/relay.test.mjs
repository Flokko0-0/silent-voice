import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deriveKey, seal, unseal, topicFor, newRoomCode, formatRoom } from '../relay.js';

test('relay: код палаты — 8 цифр', () => {
  for (let i = 0; i < 50; i++) assert.match(newRoomCode(), /^\d{8}$/);
});

test('relay: код форматируется как 1234-5678', () => {
  assert.equal(formatRoom('12345678'), '1234-5678');
});

test('relay: сообщение расшифровывается тем же кодом', async () => {
  const key = await deriveKey('12345678');
  const ev = { type: 'message', text: 'Трудно дышать', urgent: true };
  assert.deepEqual(await unseal(key, await seal(key, ev)), ev);
});

test('relay: чужой код палаты не может прочитать сообщение', async () => {
  const sealed = await seal(await deriveKey('12345678'), { text: 'Мне больно' });
  await assert.rejects(unseal(await deriveKey('87654321'), sealed));
});

test('relay: шифр не содержит текст сообщения', async () => {
  const sealed = await seal(await deriveKey('12345678'), { text: 'Позовите врача' });
  assert.ok(!Buffer.from(sealed, 'base64').toString('utf8').includes('врача'));
});

test('relay: одно и то же сообщение шифруется каждый раз по-разному', async () => {
  const key = await deriveKey('12345678');
  assert.notEqual(await seal(key, { a: 1 }), await seal(key, { a: 1 }));
});

test('relay: канал не раскрывает код палаты', async () => {
  const topic = await topicFor('12345678');
  assert.ok(!topic.includes('12345678'));
  assert.equal(topic, await topicFor('12345678'));
  assert.notEqual(topic, await topicFor('12345679'));
});
