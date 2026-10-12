"""Loading trained models and predicting with them (used by the demo and the evaluation script)."""
from pathlib import Path

import numpy as np

from .landmarks import hands_to_features


class NumpyMLP:
    """Forward pass of the exported scikit-learn MLP using only numpy."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale = z["mean"], z["scale"]
        self.classes, self.n_hands = [str(c) for c in z["classes"]], int(z["n_hands"])
        n = int(z["n_layers"])
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]

    def predict_proba(self, f):
        x = (f - self.mean) / self.scale
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max())
        return e / e.sum()


class SklearnWrap:
    def __init__(self, d):
        self.model, self.classes, self.n_hands = d["model"], d["classes"], d["n_hands"]

    def predict_proba(self, f):
        return self.model.predict_proba(f[None])[0]


class Models:
    def __init__(self, d):
        d = Path(d)
        self.letters, self.words = {}, {}
        for lang in ("asl", "isl"):
            if (d / f"{lang}_letters_np.npz").exists():          # numpy-only model (works on a Raspberry Pi)
                self.letters[lang] = NumpyMLP(d / f"{lang}_letters_np.npz")
            elif (d / f"{lang}_letters.joblib").exists():
                import joblib
                self.letters[lang] = joblib.load(d / f"{lang}_letters.joblib")
        for lang in ("asl", "isl"):
            if (d / f"{lang}_words_np.npz").exists():             # numpy-only model: no torch needed (see tools/export_words_np.py)
                from .numpy_word_model import NumpyWordModel
                m = NumpyWordModel(d / f"{lang}_words_np.npz")
                self.words[lang] = (m, m.labels)
        todo = [l for l in ("asl", "isl") if l not in self.words and (d / f"{l}_words.pt").exists()]
        if todo:                                                   # torch is only imported when a .pt has no numpy version
            try:
                from .word_model import load_model
                for lang in todo:
                    self.words[lang] = load_model(d / f"{lang}_words.pt")
            except ImportError:
                pass

    def available(self, mode):
        lang, kind = mode.split("_")
        if kind == "letters":
            return lang in self.letters
        return lang in self.words

    def letter(self, lang, hand_lists):
        m = self.letters[lang]
        if isinstance(m, dict):
            m = SklearnWrap(m)
        f = hands_to_features(hand_lists, m.n_hands)
        if f is None:
            return None
        p = m.predict_proba(f)
        i = int(p.argmax())
        return m.classes[i], float(p[i])

    def word(self, lang, kp67):
        model, labels = self.words[lang]
        if hasattr(model, "predict"):                              # NumpyWordModel
            return [(labels[i], p) for i, p in model.predict(kp67)]
        from .word_model import predict
        return [(labels[i], p) for i, p in predict(model, kp67)]
