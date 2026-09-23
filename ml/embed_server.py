"""Локальный сервер нейросети чтения по губам (порт 5174)."""

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "auto_avsr-main"))
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder  # noqa: E402
from espnet.nets.pytorch_backend.frontend.resnet import video_resnet  # noqa: E402

PORT = 5174
CROP = 96
MODEL_SIZE = 88
TARGET_FPS = 25
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
torch.set_num_threads(max(1, torch.get_num_threads()))


class VisualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = video_resnet()
        self.proj_encoder = torch.nn.Linear(512, 768)
        self.encoder = ConformerEncoder(
            attention_dim=768, attention_heads=12, linear_units=3072, num_blocks=12, cnn_module_kernel=31
        )

    @torch.no_grad()
    def forward(self, x):  # x: (1, T, 1, 88, 88)
        feats = self.frontend(x)
        feats = self.proj_encoder(feats)
        enc, _ = self.encoder(feats, None)
        return feats[0], enc[0]  # front-end (T,768) and encoder (T,768)


def load_model():
    state = torch.load(ROOT / "pytorch_model.pt", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model = VisualEncoder()
    wanted = {k: v for k, v in state.items() if k.split(".")[0] in ("frontend", "proj_encoder", "encoder")}
    missing, unexpected = model.load_state_dict(wanted, strict=False)
    if missing:
        raise RuntimeError(f"Missing weights: {missing[:5]}… ({len(missing)})")
    model.eval()
    return model


def decode_frames(payload):
    raw = np.frombuffer(base64.b64decode(payload["frames"]), dtype=np.uint8)
    n = int(payload["n"])
    clip = raw.reshape(n, CROP, CROP)
    times = payload.get("times")
    if times and len(times) == n and n > 1:
        # Resample to the 25 fps the model was trained on (webcams usually deliver ~30).
        t = np.asarray(times, dtype=np.float64)
        grid = np.arange(t[0], t[-1] + 1e-6, 1000.0 / TARGET_FPS)
        idx = np.clip(np.searchsorted(t, grid), 0, n - 1)
        clip = clip[idx]
    return clip


def to_tensor(clip):
    off = (CROP - MODEL_SIZE) // 2
    x = clip[:, off : off + MODEL_SIZE, off : off + MODEL_SIZE].astype(np.float32) / 255.0
    x = (x - 0.421) / 0.165
    return torch.from_numpy(x)[None, :, None]  # (1, T, 1, 88, 88)


def normalize_rows(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def dtw_cosine(a, b):
    cost = 1.0 - a @ b.T
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = row[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return acc[n, m] / (n + m)


class Store:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.items = []  # {lang, label, path, seq (T,768) raw front-end features}

    def load(self):
        for f in sorted(self.data_dir.glob("*.npy")):
            lang, rest = f.stem.split("_", 1)
            label = rest.rsplit("_", 1)[0]
            self.items.append({"lang": lang, "label": label, "path": f, "seq": features(np.load(f))})

    def add(self, lang, label, clip):
        path = self.data_dir / f"{lang}_{label}_{int(time.time() * 1000)}.npy"
        np.save(path, clip)
        self.items.append({"lang": lang, "label": label, "path": path, "seq": features(clip)})

    def remove(self, lang, label=None):
        keep = []
        for it in self.items:
            if it["lang"] == lang and label in (None, it["label"]):
                it["path"].unlink(missing_ok=True)
            else:
                keep.append(it)
        self.items = keep


def features(clip):
    front, _ = MODEL(to_tensor(clip))
    return front.numpy().astype(np.float32)


def prepared(store, lang):
    seqs = [it for it in store.items if it["lang"] == lang]
    if not seqs:
        return [], None
    mu = np.concatenate([it["seq"] for it in seqs]).mean(axis=0)
    return [(it["label"], normalize_rows(it["seq"] - mu)) for it in seqs], mu


def rank(entries, query, exclude=None):
    per = {}
    for i, (label, seq) in enumerate(entries):
        if i == exclude:
            continue
        per.setdefault(label, []).append(dtw_cosine(query, seq))
    ranked = []
    for label, ds in per.items():
        ds.sort()
        best = ds[:2]
        ranked.append({"label": label, "dist": float(sum(best) / len(best))})
    ranked.sort(key=lambda r: r["dist"])
    return ranked


MODEL = None
STORE = Store(DATA_DIR)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj=None):
        body = json.dumps(obj or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "templates": len(STORE.items)})
        self._send(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            lang = payload.get("lang", "ru")
            t0 = time.time()
            if self.path == "/enroll":
                clip = decode_frames(payload)
                if len(clip) < 5:
                    raise ValueError("too few frames")
                with STORE.lock:
                    STORE.add(lang, payload["label"], clip)
                    count = sum(1 for it in STORE.items if it["lang"] == lang and it["label"] == payload["label"])
                return self._send(200, {"count": count, "ms": int((time.time() - t0) * 1000)})
            if self.path == "/classify":
                clip = decode_frames(payload)
                if len(clip) < 5:
                    raise ValueError("too few frames")
                query = features(clip)
                with STORE.lock:
                    entries, mu = prepared(STORE, lang)
                if not entries:
                    return self._send(200, {"ranked": []})
                ranked = rank(entries, normalize_rows(query - mu))
                return self._send(200, {"ranked": ranked, "ms": int((time.time() - t0) * 1000)})
            if self.path == "/evaluate":
                with STORE.lock:
                    entries, _ = prepared(STORE, lang)
                correct, total, mistakes = 0, 0, []
                for i, (label, seq) in enumerate(entries):
                    if sum(1 for l, _ in entries if l == label) < 2:
                        continue
                    r = rank(entries, seq, exclude=i)
                    total += 1
                    if r and r[0]["label"] == label:
                        correct += 1
                    elif r:
                        mistakes.append({"expected": label, "got": r[0]["label"]})
                return self._send(200, {"correct": correct, "total": total, "mistakes": mistakes})
            if self.path == "/clear":
                with STORE.lock:
                    STORE.remove(lang, payload.get("label"))
                return self._send(200, {"ok": True})
            self._send(404)
        except Exception as err:  # report to the page instead of dropping the connection
            self._send(500, {"error": str(err)})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print("Loading lip-reading model…", flush=True)
    MODEL = load_model()
    STORE.load()
    print(f"Lip-reading server ready on http://localhost:{PORT} ({len(STORE.items)} templates)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
