"""Local webcam demo (OpenCV). Run on your laptop, not on Colab/Kaggle.

  python demo/app.py --models models

Keys:  1 ASL letters   2 ISL letters   3 ASL words   4 ISL words
       letters: A add current letter to the text, BACKSPACE delete, C clear
       words:   SPACE start / stop recording a sign, then the prediction is shown
       Q quit
Expected files in --models (missing ones just disable that mode):
  asl_letters.joblib  isl_letters.joblib  asl_words.pt  isl_words.pt
  optional pretrained ASL: asl_words.tflite + sign_to_prediction_index_map.json (needs tensorflow)
"""
import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slr.inference import Models, full543  # noqa: E402
from slr.landmarks import frame_from_holistic, hands_from_result, make_hands, make_holistic  # noqa: E402

MODES = {ord("1"): "asl_letters", ord("2"): "isl_letters", ord("3"): "asl_words", ord("4"): "isl_words"}
GREEN, RED, WHITE = (80, 220, 100), (60, 60, 255), (255, 255, 255)


def put(img, text, y, color=WHITE, scale=0.7):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    M = Models(a.models)
    mode = next((m for m in MODES.values() if M.available(m)), None)
    if mode is None:
        sys.exit(f"No models found in {a.models}. Download them from your Kaggle output first.")
    print("available:", [m for m in MODES.values() if M.available(m)])

    cap = cv2.VideoCapture(a.camera)
    hands = {1: make_hands(False, 1, 1), 2: make_hands(False, 2, 1)}
    holistic = make_holistic(0)
    recent = collections.deque(maxlen=8)
    text, last_letter = "", None
    recording, rec67, rec543, result = False, [], [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # un-mirrored frame goes to the models
        lang, kind = mode.split("_")
        view = cv2.flip(frame, 1)                              # mirrored view is only for display
        put(view, f"[{mode}]  1 ASL-letters 2 ISL-letters 3 ASL-words 4 ISL-words  Q quit", 24, scale=0.5)

        if kind == "letters":
            found = hands_from_result(hands[1 if lang == "asl" else 2].process(rgb))
            for h in found:
                for x, y, _ in h:
                    cv2.circle(view, (int((1 - x) * view.shape[1]), int(y * view.shape[0])), 3, GREEN, -1)
            pred = M.letter(lang, found) if found else None
            recent.append(pred[0] if pred and pred[1] > 0.5 else None)
            top = collections.Counter(recent).most_common(1)[0][0] if recent else None
            last_letter = top
            put(view, f"letter: {top or '-'}" + (f"  ({pred[1]:.0%})" if pred else ""), 60, GREEN, 1.0)
            put(view, f"text: {text}", 100)
            put(view, "A add  BACKSPACE del  C clear", view.shape[0] - 12, scale=0.5)
        else:
            res = holistic.process(rgb)
            if recording:
                rec67.append(frame_from_holistic(res)); rec543.append(full543(res))
            for lms in (res.left_hand_landmarks, res.right_hand_landmarks):
                if lms:
                    for p in lms.landmark:
                        cv2.circle(view, (int((1 - p.x) * view.shape[1]), int(p.y * view.shape[0])), 3, GREEN, -1)
            put(view, f"RECORDING {len(rec67)} frames - SPACE to stop" if recording else "SPACE to record a sign",
                60, RED if recording else WHITE, 0.8)
            for i, (lbl, p) in enumerate(result or []):
                put(view, f"{i + 1}. {lbl}  {p:.0%}", 100 + 32 * i, GREEN if i == 0 else WHITE, 0.9 if i == 0 else 0.7)

        cv2.imshow("sign-language demo", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in MODES and M.available(MODES[k]):
            mode, recording, result = MODES[k], False, None
        elif k in MODES:
            print(f"{MODES[k]}: model file missing")
        elif kind == "letters":
            if k == ord("a") and last_letter:
                text += last_letter
            elif k == 8:
                text = text[:-1]
            elif k == ord("c"):
                text = ""
        elif k == 32:
            if not recording:
                recording, rec67, rec543, result = True, [], [], None
            else:
                recording = False
                if len(rec67) >= 8:
                    result = M.word(lang, np.stack(rec67), np.stack(rec543))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
