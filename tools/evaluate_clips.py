"""Evaluate trained models on your own recordings.

  # words: clips/<lang>/<word>/*.mp4   (made by tools/record_clips.py)
  python tools/evaluate_clips.py words --lang isl --models models --clips clips --out results
  # letters: images/<lang>/<letter>/*.jpg
  python tools/evaluate_clips.py letters --lang asl --models models --images images --out results

Class names in the folders must match the model labels (ISL labels are lower-case letters only,
e.g. 'good morning' -> goodmorning, see models/isl_words.json).
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slr.inference import Models  # noqa: E402
from slr.landmarks import FrameTracker, HandDetector  # noqa: E402


def report(rows, out, name):
    """rows: (truth, [top1, top2, top3]) -> printed summary + json + confusion csv."""
    n = len(rows)
    top1 = np.mean([t == p[0] for t, p in rows]); top3 = np.mean([t in p for t, p in rows])
    per = defaultdict(list)
    for t, p in rows:
        per[t].append(p[0] == t)
    print(f"\n{name}: {n} samples  top1 {top1:.1%}  top3 {top3:.1%}")
    for t, v in sorted(per.items()):
        print(f"  {t:>14}: {np.mean(v):.0%} ({sum(v)}/{len(v)})")
    Path(out).mkdir(parents=True, exist_ok=True)
    labels = sorted({t for t, _ in rows} | {p[0] for _, p in rows})
    conf = Counter((t, p[0]) for t, p in rows)
    with open(Path(out) / f"{name}_confusion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + labels)
        for t in labels:
            w.writerow([t] + [conf[(t, p)] for p in labels])
    (Path(out) / f"{name}.json").write_text(json.dumps(
        {"n": n, "top1": top1, "top3": top3, "per_class": {k: float(np.mean(v)) for k, v in per.items()}}, indent=1))


def eval_words(a):
    M = Models(a.models)
    rows = []
    for d in sorted((Path(a.clips) / a.lang).iterdir()):
        for v in sorted(d.glob("*.mp4")):
            tracker, cap, kp = FrameTracker(), cv2.VideoCapture(str(v)), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                kp.append(tracker(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            tracker.close()
            if len(kp) < 8:
                print("too short:", v); continue
            rows.append((d.name, [l for l, _ in M.word(a.lang, np.stack(kp))]))
    report(rows, a.out, f"{a.lang}_words")


def eval_letters(a):
    M = Models(a.models)
    hands = HandDetector(1 if a.lang == "asl" else 2, video=False)
    rows = []
    for d in sorted((Path(a.images) / a.lang).iterdir()):
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            found = hands(cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB))
            pred = M.letter(a.lang, found) if found else None
            rows.append((d.name.upper(), [pred[0] if pred else "(no hand)"]))
    report(rows, a.out, f"{a.lang}_letters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("words", "letters"):
        p = sub.add_parser(n)
        p.add_argument("--lang", required=True, choices=["asl", "isl"])
        p.add_argument("--models", default="models"); p.add_argument("--out", default="results")
        p.add_argument("--clips" if n == "words" else "--images", default="clips" if n == "words" else "images")
    args = ap.parse_args()
    (eval_words if args.cmd == "words" else eval_letters)(args)
