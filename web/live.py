"""Live sign-language dashboard: webcam in the browser -> landmarks -> words (or letters) decoded in real time.

  python web/live.py --models models          # then open http://localhost:5001

The browser streams ~12 frames/s to /api/frame. The server tracks hands + pose with MediaPipe (same pipeline as
training), cuts the stream into signs (a sign = hands appear ... hands leave) and decodes each one with the word model.
Letters mode types fingerspelled letters live. Nothing leaves your machine. (Camera access needs http://localhost.)
"""
import argparse
import os
import sys
import tempfile
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
from slr.live import MIN_HAND_FRAMES, LetterStabiliser, WordSegmenter, hands_visible  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024        # uploaded videos
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


def video_frames(path, fps_target=15, max_s=40, max_w=640):
    """Yield (seconds, rgb) from a video file at ~fps_target (every n-th frame), downscaled for speed/memory."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 < (fps or 0) < 241:                         # browser-recorded webm often reports 0 / 1000
        fps = 30.0
    stride, i = max(1, round(fps / fps_target)), 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            if i % stride == 0:
                t = i / fps
                if t > max_s:
                    return
                if img.shape[1] > max_w:
                    img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1], interpolation=cv2.INTER_AREA)
                yield t, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


@app.post("/api/video")
def video():
    """Decode an uploaded recording. words: mode=word (whole clip = one sign) or mode=sentence (split into signs);
    letters: types the fingerspelled letters."""
    t0 = time.time()
    M, f = STATE["models"], request.form
    kind, lang, mode, thr = f.get("kind", "words"), f.get("lang", "isl"), f.get("mode", "word"), float(f.get("thr", 0.5))
    if not (lang in M.words if kind == "words" else lang in M.letters):
        return jsonify(error=f"No {lang.upper()} {kind} model loaded in {STATE['models_dir']}"), 400
    up = request.files.get("video")
    if up is None:
        return jsonify(error="no file"), 400
    budget, truncated, n, with_hands = 100, False, 0, 0
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("clip" + (Path(up.filename or "").suffix or ".mp4"))
        up.save(src)
        events, kps = [], []
        with LOCK:
            trk = tracker_for(kind, lang)
            seg = WordSegmenter(lambda seq: M.word(lang, seq), threshold=thr) if kind == "words" and mode == "sentence" else None
            stab = LetterStabiliser(hold=6, repeat=15) if kind == "letters" else None
            for t, rgb in video_frames(src):
                if time.time() - t0 > budget:
                    truncated = True
                    break
                n += 1
                if kind == "words":
                    fr = trk(rgb)
                    with_hands += hands_visible(fr)
                    if seg:
                        ev = seg.push(fr, now=t)
                        if ev:
                            events.append(ev)
                    else:
                        kps.append(fr)
                else:
                    hands = trk(rgb)
                    with_hands += bool(hands)
                    pred = M.letter(lang, hands) if hands else None
                    typed = stab.push(pred)[1]
                    if typed:
                        events.append({"label": typed, "prob": pred[1]})
            if seg:
                ev = seg.stop()
                if ev:
                    events.append(ev)
            elif kind == "words" and with_hands >= MIN_HAND_FRAMES:
                top = [{"label": l, "prob": float(p)} for l, p in M.word(lang, np.stack(kps))]
                events.append({"accepted": top[0]["prob"] >= thr, "top": top, "frames": len(kps)})
    if n == 0:
        return jsonify(error="Could not read any frames from that file. Try an .mp4 or .webm."), 400
    if with_hands < MIN_HAND_FRAMES:
        return jsonify(error="No hands were found in the video. Make sure your hands are visible and well lit."), 400
    return jsonify(kind=kind, mode=mode, events=events, frames=n, hands_pct=round(with_hands / n, 2),
                   seconds=round(time.time() - t0, 1), truncated=truncated)


def init(models_dir):
    for name in ("hand_landmarker.task", "pose_landmarker_lite.task"):
        model_path(name)                                  # fetch once now instead of during the first frame
    STATE.update(models=Models(models_dir), models_dir=models_dir)
    M = STATE["models"]
    print("word models:", list(M.words), "| letter models:", list(M.letters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another device (camera needs https there)")
    a = ap.parse_args()
    init(a.models)
    print(f"open http://localhost:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
else:                                                      # imported by a WSGI server (Render): gunicorn --chdir web live:app
    init(os.environ.get("SLR_MODELS", str(ROOT / "models")))
