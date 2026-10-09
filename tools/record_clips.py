"""Record your own test clips with the laptop webcam: N clips per word.

  python tools/record_clips.py --lang isl --words hello thank_you water --per-word 5

Saves clips/<lang>/<word>/<k>.mp4 (3 s each, un-mirrored, like the training videos).
Press SPACE when you are ready for each clip (3-2-1 countdown), S to skip a word, Q to quit.
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["asl", "isl"])
    ap.add_argument("--words", nargs="+", required=True)
    ap.add_argument("--per-word", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="clips")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.camera)
    w, h = int(cap.get(3)), int(cap.get(4))

    def banner(img, text):
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    for word in a.words:
        d = Path(a.out) / a.lang / word
        d.mkdir(parents=True, exist_ok=True)
        k = len(list(d.glob("*.mp4")))
        while k < a.per_word:
            while True:                                         # wait for SPACE
                ok, f = cap.read()
                if not ok:
                    return
                v = cv2.flip(f, 1)
                banner(v, f"{a.lang}:{word}  clip {k + 1}/{a.per_word}  SPACE=go S=skip Q=quit")
                cv2.imshow("record", v)
                key = cv2.waitKey(1) & 0xFF
                if key == 32 or key in (ord("s"), ord("q")):
                    break
            if key == ord("q"):
                return
            if key == ord("s"):
                break
            t0 = time.time()
            while time.time() - t0 < 3:                         # countdown
                ok, f = cap.read(); v = cv2.flip(f, 1)
                banner(v, f"{word}: starting in {3 - int(time.time() - t0)}")
                cv2.imshow("record", v); cv2.waitKey(1)
            out = cv2.VideoWriter(str(d / f"{k + 1}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            t0 = time.time()
            while time.time() - t0 < a.seconds:
                ok, f = cap.read()
                out.write(f)                                     # save un-mirrored
                v = cv2.flip(f, 1); banner(v, "RECORDING"); cv2.imshow("record", v); cv2.waitKey(1)
            out.release()
            k += 1
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
