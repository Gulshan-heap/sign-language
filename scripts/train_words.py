"""Train the word classifier on keypoint .npz files (layout: <data>/{train,val,test}/<label>__<id>.npz).

  python scripts/train_words.py --data work/include50 --out models/isl_words.pt --epochs 150
Use a GPU session on Kaggle for this (T4); CPU also works, just slower. Keypoints are tiny.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slr import word_model as wm  # noqa: E402


def load_split(root, split, label_to_i):
    seqs, ys = [], []
    for p in sorted((Path(root) / split).glob("*.npz")):
        label = p.name.split("__")[0]
        if label in label_to_i:
            seqs.append(np.load(p)["kp"]); ys.append(label_to_i[label])
    return seqs, ys


def train_words(data, out, epochs=150, seeds=1):
    """Train the word transformer on <data>/{train,val,test}/*.npz; save <out> (.pt) and its .json metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    labels = sorted({p.name.split("__")[0] for p in Path(data).glob("*/*.npz")})
    li = {l: i for i, l in enumerate(labels)}
    tr, va, te = (load_split(data, s, li) for s in ("train", "val", "test"))
    print(f"{len(labels)} classes | train {len(tr[0])} val {len(va[0])} test {len(te[0])}")
    if not tr[0]:
        sys.exit(f"No training files found under {data}/train - check the path (expected <label>__<id>.npz files)")

    best_model, best_val = None, -1
    for seed in range(seeds):
        model, val = wm.train(*tr, *va, len(labels), epochs=epochs, device=device, seed=seed)
        if val > best_val:
            best_model, best_val = model, val
    metrics = {}
    for name, (s, y) in (("val", va), ("test", te)):
        if len(s):
            metrics[name], _ = wm.evaluate(best_model, s, y, device)
            print(name, metrics[name])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wm.save_model(out, best_model.cpu(), labels, {"metrics": metrics})
    wm.dump_json(Path(out).with_suffix(".json"), {"labels": labels, "metrics": metrics})
    print("saved", out, f"({Path(out).stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1, help="train N seeds, keep the best on val")
    a = ap.parse_args()
    train_words(a.data, a.out, a.epochs, a.seeds)


if __name__ == "__main__":
    main()
