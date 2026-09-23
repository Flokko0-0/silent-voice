"""Стартовый набор для онлайн-версии: отпечатки нейросети и точки губ из калибровки автора.

В набор попадают только числа (признаки), а не изображения лица.
Запуск: python export_preset.py <экспорт-калибровки.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

import embed_server as s

OUT = Path(__file__).parent.parent / "preset"
OUT.mkdir(exist_ok=True)
MIN_FRAMES = 20

s.MODEL = s.load_model()
index, chunks, offset = [], [], 0
for f in sorted(s.DATA_DIR.glob("ru_*.npy")):
    label = f.stem.split("_", 1)[1].rsplit("_", 1)[0]
    feats = s.features(np.load(f)).astype(np.float16)
    index.append({"lang": "ru", "label": label, "T": int(feats.shape[0]), "offset": offset})
    chunks.append(feats.tobytes())
    offset += feats.nbytes
(OUT / "neural-ru.bin").write_bytes(b"".join(chunks))
(OUT / "neural-ru.json").write_text(json.dumps(index), encoding="utf-8")
print("neural clips:", len(index), "bytes:", offset)

labels = {it["label"] for it in index}
export = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lips = {}
for key, seqs in export["templates"].items():
    label, lang = key.split("|")
    if lang != "ru" or label not in labels:
        continue
    keep = [[[round(v, 3) for v in frame] for frame in seq] for seq in seqs if len(seq) >= MIN_FRAMES]
    if keep:
        lips[key] = keep
(OUT / "lips-ru.json").write_text(json.dumps(lips, separators=(",", ":")), encoding="utf-8")
print("lip templates:", {k: len(v) for k, v in lips.items()})
