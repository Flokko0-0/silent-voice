"""Экспорт визуального фронтенда Auto-AVSR в ONNX для работы в браузере."""
import numpy as np
import onnxruntime as ort
import torch
import embed_server as s


class Front(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.frontend = m.frontend
        self.proj_encoder = m.proj_encoder

    def forward(self, x):
        return self.proj_encoder(self.frontend(x))[0]


m = s.load_model()
front = Front(m).eval()
x = torch.randn(1, 40, 1, 88, 88)
torch.onnx.export(front, (x,), "../vendor/lipfront.onnx", input_names=["frames"], output_names=["features"],
                  dynamic_axes={"frames": {1: "T"}, "features": {0: "T"}}, opset_version=17, dynamo=False)

sess = ort.InferenceSession("../vendor/lipfront.onnx")
for T in (25, 63):
    clip = torch.randn(1, T, 1, 88, 88)
    ref = front(clip).detach().numpy()
    got = sess.run(None, {"frames": clip.numpy()})[0]
    print(T, got.shape, "max diff", float(np.abs(ref - got).max()))
