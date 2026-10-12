"""Live sign-language dashboard: webcam in the browser -> landmarks -> words (or letters) decoded in real time.

  python web/live.py --models models          # then open http://localhost:5001

The browser streams ~12 frames/s to /api/frame. The server tracks hands + pose with MediaPipe (same pipeline as
training), cuts the stream into signs (a sign = hands appear ... hands leave) and decodes each one with the word model.
Letters mode types fingerspelled letters live. Nothing leaves your machine. (Camera access needs http://localhost.)
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slr.inference import Models  # noqa: E402
from slr.landmarks import FrameTracker, HandDetector, LH, RH, N_POSE, model_path  # noqa: E402
from slr.live import LetterStabiliser, WordSegmenter  # noqa: E402

app = Flask(__name__)
LOCK = threading.Lock()            # MediaPipe objects are not thread-safe; one camera at a time
STATE = {"trackers": {}, "key": None, "seg": None, "stab": None}
HTML = Path(__file__).with_name("live.html")


def pts(a):
    """(n, 2+) array -> JSON-safe [[x, y] | None, ...] (NaN -> None)."""
    return [None if not np.isfinite(p[:2]).all() else [round(float(p[0]), 3), round(float(p[1]), 3)] for p in a]


def tracker_for(kind, lang):
    """Trackers are kept after first use: loading MediaPipe is slow, and video-mode timestamps must keep increasing."""
    k = ("words",) if kind == "words" else ("letters", 1 if lang == "asl" else 2)
    if k not in STATE["trackers"]:
        STATE["trackers"][k] = FrameTracker() if kind == "words" else HandDetector(k[1], video=True)
    return STATE["trackers"][k]


def session(kind, lang):
    """The segmenter / stabiliser for the current (kind, lang); a fresh one whenever the user switches."""
    if STATE["key"] != (kind, lang):
        M = STATE["models"]
        STATE["key"] = (kind, lang)
        STATE["seg"] = WordSegmenter(lambda seq: M.word(lang, seq)) if kind == "words" else None
        STATE["stab"] = LetterStabiliser() if kind == "letters" else None
    return STATE["seg"], STATE["stab"]


@app.get("/")
def index():
    return HTML.read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    M = STATE["models"]
    return jsonify(words={l: M.words[l][1] for l in M.words},
                   letters={l: (M.letters[l].classes if not isinstance(M.letters[l], dict) else M.letters[l]["classes"])
                            for l in M.letters})


@app.post("/api/frame")
def frame():
    t0 = time.time()
    M, f = STATE["models"], request.form
    kind, lang = f.get("kind", "words"), f.get("lang", "isl")
    if not (lang in M.words if kind == "words" else lang in M.letters):
        return jsonify(error=f"No {lang.upper()} {kind} model loaded in {STATE['models_dir']}"), 400
    img = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="bad image"), 400
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # un-mirrored, exactly what the models were trained on
    cmd = f.get("cmd", "")
    out = {}
    with LOCK:
        seg, stab = session(kind, lang)
        trk = tracker_for(kind, lang)
        if kind == "words":
            seg.threshold = float(f.get("thr", 0.5))
            if cmd == "reset":
                seg.reset()
            if cmd == "start":
                seg.start()
            fr = trk(rgb)
            ev = seg.stop() if cmd == "stop" else seg.push(fr)
            if ev:
                out["event"] = ev
            out.update(state=seg.state, frames=len(seg.buf), seconds=round(time.time() - seg.t0, 1) if seg.buf else 0,
                       hands=[h for h in (pts(fr[LH]), pts(fr[RH])) if h[0]], pose=pts(fr[:N_POSE]))
        else:
            hands = trk(rgb)
            pred = M.letter(lang, hands) if hands else None
            if cmd == "reset":
                stab.reset()
            cand, typed = stab.push(pred)
            out.update(letter=cand, prob=pred[1] if pred else 0.0, typed=typed, progress=min(1.0, stab.run / stab.hold),
                       hands=[pts(h) for h in hands], pose=None)
    out["ms"] = round((time.time() - t0) * 1000)
    return jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another device (camera needs https there)")
    a = ap.parse_args()
    for name in ("hand_landmarker.task", "pose_landmarker_lite.task"):
        model_path(name)                                  # fetch once now instead of during the first frame
    STATE.update(models=Models(a.models), models_dir=a.models)
    M = STATE["models"]
    print("word models:", list(M.words), "| letter models:", list(M.letters))
    print(f"open http://localhost:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
