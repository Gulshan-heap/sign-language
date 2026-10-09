"""Letter classifiers from hand landmarks (ASL Alphabet, ISL fingerspelling). CPU, minutes.

  python scripts/letters.py extract --images <root with one folder per letter> --lang asl --out work/letters
  python scripts/letters.py train   --lang asl --data work/letters --out models

`--images` can point at the dataset root; the class-folder level is auto-detected.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slr.landmarks import HandDetector, MODELS, hands_to_features, mirror_hands, model_path  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
N_HANDS = {"asl": 1, "isl": 2}


def find_class_root(root):
    """Directory (at any depth) with the most sub-folders that directly contain images."""
    best, best_n = None, 0
    for dp, dirs, _ in os.walk(root):
        n = sum(1 for d in dirs if any(Path(dp, d, f).suffix.lower() in IMG_EXT for f in os.listdir(Path(dp, d))[:5]))
        if n > best_n:
            best, best_n = dp, n
    if best is None:
        sys.exit(f"No class folders with images found under {root}")
    return Path(best)


def _extract_chunk(args):
    paths, n_hands = args
    hands = HandDetector(n_hands, video=False)
    import cv2
    out = []
    for p, y in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        found = hands(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        f = hands_to_features(found, n_hands)
        if f is None:
            continue
        out.append((f, y, 0))
        out.append((hands_to_features(mirror_hands(found), n_hands), y, 1))   # mirrored copy
    return out


def cmd_extract(a):
    from joblib import Parallel, delayed
    n_hands = N_HANDS[a.lang]
    model_path("hand_landmarker.task")          # download once here, not in every worker
    root = find_class_root(a.images)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in set(a.skip))
    print(f"class root: {root}\n{len(classes)} classes: {classes}")
    rng = random.Random(0)
    items = []
    for ci, c in enumerate(classes):
        files = sorted(f for f in (root / c).iterdir() if f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        items += [(f, ci) for f in files[: a.per_class]]
    rng.shuffle(items)
    chunks = [(items[i:i + 100], n_hands) for i in range(0, len(items), 100)]
    res = Parallel(n_jobs=a.workers, verbose=5)(delayed(_extract_chunk)(c) for c in chunks)
    rows = [r for chunk in res for r in chunk]
    if not rows:
        sys.exit("No hands detected in any image - check --images points at the right dataset")
    X = np.stack([r[0] for r in rows]); y = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
    # image-level split ids so the mirrored copy of an image never lands in a different split
    img_id = np.repeat(np.arange(len(rows) // 2), 2)
    print(f"{len(rows) // 2}/{len(items)} images had a detectable hand")
    for ci, c in enumerate(classes):
        tot = sum(1 for _, yy in items if yy == ci)
        print(f"  {c:>8}: detected {int((y == ci).sum()) // 2}/{tot}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"{a.lang}_letters.npz", X=X, y=y, mirrored=m, img_id=img_id, classes=np.array(classes))
    print("saved", Path(a.out) / f"{a.lang}_letters.npz")


def cmd_train(a):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import classification_report
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    d = np.load(Path(a.data) / f"{a.lang}_letters.npz")
    X, y, img, classes = d["X"], d["y"], d["img_id"], list(d["classes"])
    ids = np.unique(img)
    rng = np.random.default_rng(0); rng.shuffle(ids)
    n = len(ids)
    split = {"train": set(ids[: int(.7 * n)]), "val": set(ids[int(.7 * n): int(.85 * n)]), "test": set(ids[int(.85 * n):])}
    m = {k: np.array([i in v for i in img]) for k, v in split.items()}
    models = {
        "svc": make_pipeline(StandardScaler(), SVC(C=10, probability=True)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((256, 128), max_iter=400, early_stopping=True, random_state=0)),
        "extratrees": ExtraTreesClassifier(400, n_jobs=-1, random_state=0),
    }
    best_name, best_acc = None, -1
    for name, mdl in models.items():
        mdl.fit(X[m["train"]], y[m["train"]])
        acc = mdl.score(X[m["val"]], y[m["val"]])
        print(f"{name:>10}  val acc {acc:.4f}")
        if acc > best_acc:
            best_name, best_acc = name, acc
    best = models[best_name]
    best.fit(X[m["train"] | m["val"]], y[m["train"] | m["val"]])
    print(f"\nbest = {best_name}; test acc {best.score(X[m['test']], y[m['test']]):.4f}")
    print(classification_report(y[m["test"]], best.predict(X[m["test"]]), target_names=classes, zero_division=0))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "classes": classes, "n_hands": N_HANDS[a.lang], "kind": best_name},
                Path(a.out) / f"{a.lang}_letters.joblib")
    print("saved", Path(a.out) / f"{a.lang}_letters.joblib")

    # Raspberry-Pi-friendly copy: plain numpy weights, so the Pi needs neither scikit-learn nor a matching version
    mlp = models["mlp"]
    mlp.fit(X[m["train"] | m["val"]], y[m["train"] | m["val"]])
    print(f"numpy-export MLP test acc {mlp.score(X[m['test']], y[m['test']]):.4f}")
    sc, net = mlp.steps[0][1], mlp.steps[1][1]
    arrays = {"mean": sc.mean_, "scale": sc.scale_, "classes": np.array(classes), "n_hands": np.array(N_HANDS[a.lang]),
              "n_layers": np.array(len(net.coefs_))}
    for i, (w, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    np.savez(Path(a.out) / f"{a.lang}_letters_np.npz", **arrays)
    print("saved", Path(a.out) / f"{a.lang}_letters_np.npz", "(use this one on the Raspberry Pi)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--images", required=True); e.add_argument("--lang", choices=N_HANDS, required=True)
    e.add_argument("--out", required=True); e.add_argument("--per-class", type=int, default=300)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--skip", nargs="*", default=["nothing"], help="class folders to ignore")
    t = sub.add_parser("train")
    t.add_argument("--lang", choices=N_HANDS, required=True); t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train}[args.cmd](args)
