"""Offline leave-one-out comparison of embedding layers / matching methods on saved crops."""
import glob, re, sys
from pathlib import Path
import numpy as np, torch
import embed_server as s

lang = sys.argv[1] if len(sys.argv) > 1 else "ru"
files = sorted(glob.glob(f"data/{lang}_*.npy"))
labels = [re.match(rf"{lang}_(.+)_\d+\.npy", Path(f).name).group(1) for f in files]
model = s.load_model()
layers = {}
hooks = [blk.register_forward_hook(lambda m, i, o, k=k: layers.__setitem__(k, (o[0] if isinstance(o, tuple) else o))) for k, blk in enumerate(model.encoder.encoders)]

feats = {}  # name -> list of (T,D) arrays
for f in files:
    clip = np.load(f)
    x = s.to_tensor(clip)
    with torch.no_grad():
        front, enc = model(x)
    add = lambda name, a: feats.setdefault(name, []).append(np.asarray(a, dtype=np.float32))
    add("frontend", front.numpy())
    add("final", enc.numpy())
    for k in (2, 4, 6, 8, 10):
        t = layers[k]
        t = t[0] if isinstance(t, tuple) else t
        add(f"layer{k}", t[0].numpy())

def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

def loo(seqs, method, k=2, center=False):
    if center:
        mu = np.concatenate(seqs).mean(0)
        seqs = [q - mu for q in seqs]
    seqs = [norm(q) for q in seqs]
    pooled = [norm(q.mean(0)) for q in seqs]
    ok = 0; top3 = 0
    for i in range(len(seqs)):
        per = {}
        for j in range(len(seqs)):
            if i == j: continue
            d = s.dtw_cosine(seqs[i], seqs[j]) if method == "dtw" else 1 - float(pooled[i] @ pooled[j])
            per.setdefault(labels[j], []).append(d)
        r = sorted(((sum(sorted(v)[:k]) / len(sorted(v)[:k]), l) for l, v in per.items()))
        ok += r[0][1] == labels[i]
        top3 += labels[i] in [l for _, l in r[:3]]
    n = len(seqs)
    return f"{100*ok/n:4.0f}%  top3 {100*top3/n:4.0f}%"

print("n =", len(files), "classes =", len(set(labels)))
for name, seqs in feats.items():
    print(f"{name:9s} dtw {loo(seqs,'dtw')} | dtw+center {loo(seqs,'dtw',center=True)} | pool {loo(seqs,'pool')} | pool+center {loo(seqs,'pool',center=True)}")
