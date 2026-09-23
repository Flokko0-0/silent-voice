import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT) || 5173;
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.task': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

// Server-Sent Events subscribers (patient screen and any number of nurse phones).
const clients = new Set();
const recent = []; // last messages, replayed to a nurse page that connects late

function broadcast(event) {
  const payload = `data: ${JSON.stringify(event)}\n\n`;
  for (const res of clients) res.write(payload);
  if (event.type === 'message') {
    recent.push(event);
    if (recent.length > 30) recent.shift();
  }
}

function lanUrls() {
  const urls = [];
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const a of addrs || []) {
      if (a.family === 'IPv4' && !a.internal) urls.push(`http://${a.address}:${port}/nurse.html`);
    }
  }
  return urls;
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1e5) req.destroy();
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(body || '{}'));
      } catch (err) {
        reject(err);
      }
    });
  });
}

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');

    if (url.pathname === '/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });
      res.write(`data: ${JSON.stringify({ type: 'hello', recent })}\n\n`);
      clients.add(res);
      const ping = setInterval(() => res.write(': ping\n\n'), 20000);
      req.on('close', () => {
        clearInterval(ping);
        clients.delete(res);
      });
      return;
    }

    if (url.pathname === '/api/message' && req.method === 'POST') {
      try {
        const msg = await readJson(req);
        broadcast({
          type: 'message',
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          at: Date.now(),
          text: String(msg.text || '').slice(0, 200),
          urgent: Boolean(msg.urgent),
          via: String(msg.via || '').slice(0, 24),
          patient: String(msg.patient || '').slice(0, 80),
          ward: String(msg.ward || '').slice(0, 40),
        });
        res.writeHead(204);
      } catch {
        res.writeHead(400);
      }
      return res.end();
    }

    if (url.pathname === '/api/ack' && req.method === 'POST') {
      try {
        const msg = await readJson(req);
        broadcast({ type: 'ack', id: String(msg.id || ''), by: String(msg.by || 'Медсестра').slice(0, 60), at: Date.now() });
        res.writeHead(204);
      } catch {
        res.writeHead(400);
      }
      return res.end();
    }

    if (url.pathname === '/api/info') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ nurseUrls: lanUrls() }));
    }

    let pathname = decodeURIComponent(url.pathname);
    if (pathname.endsWith('/')) pathname += 'index.html';
    const file = path.join(root, pathname);
    if (!file.startsWith(root)) {
      res.writeHead(403);
      return res.end();
    }
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        return res.end('Not found');
      }
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
      res.end(data);
    });
  })
  .listen(port, () => {
    console.log(`Экран пациента: http://localhost:${port}`);
    for (const u of lanUrls()) console.log(`Пульт медсестры (с телефона): ${u}`);
  });
