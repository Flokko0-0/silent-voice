"""Тесты сервера нейросети без загрузки модели."""
import base64
import unittest

import numpy as np

import embed_server as s


def payload(n, times=None):
    clip = (np.arange(n * 96 * 96) % 251).astype(np.uint8)
    p = {"frames": base64.b64encode(clip.tobytes()).decode(), "n": n}
    if times is not None:
        p["times"] = times
    return p


class ServerTests(unittest.TestCase):
    def test_decode_shape(self):
        self.assertEqual(s.decode_frames(payload(10)).shape, (10, 96, 96))

    def test_resample_to_25fps(self):
        times = [i * 1000 / 30 for i in range(31)]  # 1 секунда при 30 к/с
        self.assertEqual(len(s.decode_frames(payload(31, times))), 26)

    def test_tensor_crop_and_normalize(self):
        clip = np.full((4, 96, 96), 255 * 0.421, dtype=np.float32).astype(np.uint8)
        x = s.to_tensor(clip)
        self.assertEqual(tuple(x.shape), (1, 4, 1, 88, 88))
        self.assertLess(float(x.abs().max()), 0.02)

    def test_dtw_identical_is_zero(self):
        a = s.normalize_rows(np.random.RandomState(0).randn(12, 768))
        self.assertAlmostEqual(s.dtw_cosine(a, a), 0.0, places=6)

    def test_dtw_symmetric(self):
        r = np.random.RandomState(1)
        a = s.normalize_rows(r.randn(10, 768))
        b = s.normalize_rows(r.randn(14, 768))
        self.assertAlmostEqual(s.dtw_cosine(a, b), s.dtw_cosine(b, a), places=9)

    def test_rank_prefers_same_phrase(self):
        r = np.random.RandomState(2)
        base = {k: r.randn(20, 768) for k in ("drink", "pain")}
        entries = [(k, s.normalize_rows(base[k] + 0.2 * r.randn(20, 768))) for k in ("drink", "pain") for _ in range(3)]
        query = s.normalize_rows(base["pain"] + 0.2 * r.randn(20, 768))
        self.assertEqual(s.rank(entries, query)[0]["label"], "pain")

    def test_rank_excludes_given_index(self):
        a = s.normalize_rows(np.ones((5, 768)))
        entries = [("x", a), ("y", a)]
        self.assertEqual([r["label"] for r in s.rank(entries, a, exclude=0)], ["y"])


if __name__ == "__main__":
    unittest.main()
