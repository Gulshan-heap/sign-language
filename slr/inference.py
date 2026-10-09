"""Loading trained models and predicting with them (used by the demo and the evaluation script)."""
import json
from pathlib import Path

import numpy as np

from .landmarks import hands_to_features


def full543(res):
    """Holistic result -> (543, 3) in Google-ISLR order (face, left hand, pose, right hand), NaN if missing."""
    def arr(l, n):
        return np.array([[p.x, p.y, p.z] for p in l.landmark], np.float32) if l else np.full((n, 3), np.nan, np.float32)
    return np.concatenate([arr(res.face_landmarks, 468), arr(res.left_hand_landmarks, 21),
                           arr(res.pose_landmarks, 33), arr(res.right_hand_landmarks, 21)])


class Models:
    def __init__(self, d):
        d = Path(d)
        self.letters, self.words, self.tflite = {}, {}, None
        try:
            import joblib
            for lang in ("asl", "isl"):
                if (d / f"{lang}_letters.joblib").exists():
                    self.letters[lang] = joblib.load(d / f"{lang}_letters.joblib")
        except ImportError:
            pass
        try:
            from .word_model import load_model
            for lang in ("asl", "isl"):
                if (d / f"{lang}_words.pt").exists():
                    self.words[lang] = load_model(d / f"{lang}_words.pt")
        except ImportError:
            pass
        tfl, mp_ = d / "asl_words.tflite", d / "sign_to_prediction_index_map.json"
        if tfl.exists() and mp_.exists():
            try:
                import tensorflow as tf
                s2i = json.loads(mp_.read_text())
                self.tflite = (tf.lite.Interpreter(model_path=str(tfl)).get_signature_runner("serving_default"),
                               {v: k for k, v in s2i.items()})
            except Exception as e:
                print("pretrained ASL tflite unavailable:", e)

    def available(self, mode):
        lang, kind = mode.split("_")
        if kind == "letters":
            return lang in self.letters
        return lang in self.words or (lang == "asl" and self.tflite is not None)

    def letter(self, lang, hand_lists):
        m = self.letters[lang]
        f = hands_to_features(hand_lists, m["n_hands"])
        if f is None:
            return None
        p = m["model"].predict_proba(f[None])[0]
        i = int(p.argmax())
        return m["classes"][i], float(p[i])

    def word(self, lang, kp67, kp543):
        if lang in self.words:
            from .word_model import predict
            model, labels = self.words[lang]
            return [(labels[i], p) for i, p in predict(model, kp67)]
        runner, i2s = self.tflite
        out = runner(inputs=kp543)["outputs"]
        e = np.exp(out - out.max()); e /= e.sum()
        return [(i2s[int(i)], float(e[i])) for i in np.argsort(e)[::-1][:3]]
