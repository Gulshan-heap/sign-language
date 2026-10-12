"""Forward pass of the trained WordTransformer in plain numpy (no torch: ~175 MB less memory, faster to start).

Weights come from `python tools/export_words_np.py` (<lang>_words.pt -> <lang>_words_np.npz).
Mirrors slr.word_model.WordTransformer in eval mode: input Linear+LayerNorm, + positional embedding,
pre-norm encoder layers (multi-head self-attention, ReLU feed-forward), mean over time, LayerNorm, Linear.
"""
import numpy as np

from .features import preprocess


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class NumpyWordModel:
    def __init__(self, path, heads=4):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "labels"}
        self.labels = [str(l) for l in z["labels"]]
        self.heads = heads
        self.n_layers = 1 + max(int(k.split(".")[2]) for k in self.w if k.startswith("enc.layers."))

    def _layer(self, x, i):
        p = lambda n: self.w[f"enc.layers.{i}.{n}"]                         # noqa: E731
        T, d = x.shape
        h = _ln(x, p("norm1.weight"), p("norm1.bias"))
        qkv = h @ p("self_attn.in_proj_weight").T + p("self_attn.in_proj_bias")
        q, k, v = (a.reshape(T, self.heads, d // self.heads).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=-1))
        att = q @ k.transpose(0, 2, 1) / np.sqrt(d // self.heads)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + o @ p("self_attn.out_proj.weight").T + p("self_attn.out_proj.bias")
        h = _ln(x, p("norm2.weight"), p("norm2.bias"))
        h = np.maximum(h @ p("linear1.weight").T + p("linear1.bias"), 0)
        return x + h @ p("linear2.weight").T + p("linear2.bias")

    def logits(self, feats):
        """feats: (T, FEAT_DIM) from slr.features.preprocess"""
        w = self.w
        x = _ln(feats @ w["inp.0.weight"].T + w["inp.0.bias"], w["inp.1.weight"], w["inp.1.bias"])
        x = x + w["pos"][0, :len(x)]
        for i in range(self.n_layers):
            x = self._layer(x, i)
        x = _ln(x.mean(0), w["head.0.weight"], w["head.0.bias"])
        return x @ w["head.1.weight"].T + w["head.1.bias"]

    def predict(self, seq, topk=3):
        """raw (T, 67, 2) keypoints -> [(label_index, prob), ...] best first (same as word_model.predict)."""
        z = self.logits(preprocess(seq))
        p = np.exp(z - z.max())
        p /= p.sum()
        return [(int(i), float(p[i])) for i in p.argsort()[::-1][:topk]]
