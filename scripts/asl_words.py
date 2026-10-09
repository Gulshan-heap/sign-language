"""ASL words from the Google ISLR competition data (asl-signs). Reads ONLY the chosen signs' parquet files.

Attach the competition to the Kaggle notebook (Add Input -> Competition -> "asl-signs"); never download it.

  # A) pretrained TFLite weights (209sontung/sign-language or abhinand5/isolated-sign-language-recognition)
  python scripts/asl_words.py inspect --model-dir /kaggle/input/sign-language
  python scripts/asl_words.py infer   --model-dir /kaggle/input/sign-language --data /kaggle/input/asl-signs \
         --signs hello thankyou please --per-sign 20 --out work/asl_infer.json

  # B) fallback / fine-tune: convert only the chosen signs to our keypoint format, then use train_words.py
  python scripts/asl_words.py extract --data /kaggle/input/asl-signs --signs hello thankyou ... --out work/asl_words
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SIGNS = ["hello", "thankyou", "please", "yes", "no", "mom", "dad", "water", "eat", "drink", "go",
                 "sleep", "happy", "sad", "like", "want", "home", "cat", "dog", "bird", "red", "blue",
                 "green", "hungry", "sick"]
OFFSET = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}   # Google ISLR 543-point layout


def read_sign_parquet(path):
    """Google ISLR parquet -> (T, 543, 3) float32 with NaN for missing points."""
    import pandas as pd
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df["frame"].unique())
    arr = np.full((len(frames), 543, 3), np.nan, np.float32)
    fi = np.searchsorted(frames, df["frame"].values)
    li = df["type"].map(OFFSET).values + df["landmark_index"].values
    arr[fi, li] = df[["x", "y", "z"]].values
    return arr


def to_compact(arr543):
    """(T, 543, 3) -> (T, 67, 2): pose 0..24, left hand, right hand."""
    pose = arr543[:, 489:489 + 25, :2]
    return np.concatenate([pose, arr543[:, 468:489, :2], arr543[:, 522:543, :2]], 1).astype(np.float16)


def pick_rows(data, signs, per_sign, seed=0):
    import pandas as pd
    df = pd.read_csv(Path(data) / "train.csv")
    have = set(df["sign"].unique())
    missing = [s for s in signs if s not in have]
    if missing:
        print("WARNING: not in the 250-sign vocabulary, skipped:", missing)
    df = df[df["sign"].isin(signs)]
    if per_sign:
        df = df.groupby("sign", group_keys=False).apply(lambda g: g.sample(min(len(g), per_sign), random_state=seed))
    return df.reset_index(drop=True)


def find_model_files(model_dir):
    d = Path(model_dir)
    return (sorted(d.rglob("*.tflite")), sorted(d.rglob("*.json")),
            [p for p in d.rglob("*") if p.suffix in {".h5", ".keras", ".pb", ".pt", ".pth", ".onnx"}])


def cmd_inspect(a):
    tfl, js, other = find_model_files(a.model_dir)
    print("tflite:", [str(p) for p in tfl]); print("json:", [str(p) for p in js]); print("other:", [str(p) for p in other])
    if tfl:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=str(tfl[0]))
        print("signatures:", it.get_signature_list())
        print("inputs:", it.get_input_details()); print("outputs:", it.get_output_details())
    else:
        print("\nNo .tflite found. If TensorFlow versions clash or the weights are another format, "
              "switch to abhinand5/isolated-sign-language-recognition (the plan's fallback), or use `extract` + train_words.py.")


def cmd_infer(a):
    import tensorflow as tf
    tfl, js, _ = find_model_files(a.model_dir)
    if not tfl:
        sys.exit("no .tflite model in --model-dir (run `inspect`)")
    maps = [p for p in js if "sign_to_prediction" in p.name] or js
    s2i = json.loads(maps[0].read_text()); i2s = {v: k for k, v in s2i.items()}
    it = tf.lite.Interpreter(model_path=str(tfl[0]))
    runner = it.get_signature_runner("serving_default")
    rows = pick_rows(a.data, a.signs, a.per_sign)
    results = []
    for _, r in rows.iterrows():
        arr = read_sign_parquet(Path(a.data) / r["path"])
        out = runner(inputs=arr)["outputs"]
        top = np.argsort(out)[::-1][:3]
        results.append({"truth": r["sign"], "top3": [i2s[int(i)] for i in top]})
    top1 = np.mean([x["top3"][0] == x["truth"] for x in results]); top3 = np.mean([x["truth"] in x["top3"] for x in results])
    print(f"{len(results)} clips | top1 {top1:.3f} | top3 {top3:.3f}")
    for s in a.signs:
        rs = [x for x in results if x["truth"] == s]
        if rs:
            print(f"  {s:>10}: {np.mean([x['top3'][0] == s for x in rs]):.2f} ({len(rs)})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"top1": top1, "top3": top3, "results": results}, indent=1))


def cmd_extract(a):
    rows = pick_rows(a.data, a.signs, a.per_sign or 0)
    rng = np.random.default_rng(0)
    # split by signer so test accuracy is honest
    people = rows["participant_id"].unique(); rng.shuffle(people)
    sp = {p: ("test" if i < max(1, len(people) // 8) else "val" if i < max(2, len(people) // 4) else "train")
          for i, p in enumerate(people)}
    rows["split"] = rows["participant_id"].map(sp)
    for _, r in rows.iterrows():
        dst = Path(a.out) / r["split"] / f"{r['sign']}__{r['sequence_id']}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, kp=to_compact(read_sign_parquet(Path(a.data) / r["path"])))
    print(rows["split"].value_counts().to_dict(), "->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("--model-dir", required=True)
    for name in ("infer", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--signs", nargs="+", default=DEFAULT_SIGNS)
        p.add_argument("--per-sign", type=int, default=20 if name == "infer" else 0)
        if name == "infer":
            p.add_argument("--model-dir", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"inspect": cmd_inspect, "infer": cmd_infer, "extract": cmd_extract}[args.cmd](args)
