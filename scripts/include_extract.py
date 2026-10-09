"""INCLUDE-50 -> keypoints, without ever holding more than a few videos on disk.

For every Zenodo zip part we read only the zip's table of contents over HTTP (remotezip), pull
JUST the videos that belong to the INCLUDE-50 split, extract keypoints, then delete the video.
So you download ~1000 videos instead of the 56 GB archive. Resumable: finished videos are skipped.

  python scripts/include_extract.py --out work/include50 --workers 8
  python scripts/include_extract.py --out work/include50 --parts Greetings_1of2.zip   # one part only
  python scripts/include_extract.py --out work/include50 --list                        # show parts, no work

Output: <out>/<split>/<label>__<video>.npz  with key `kp` = (T, 67, 2) float16 (see slr/landmarks.py)
"""
import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from slr.landmarks import video_to_keypoints  # noqa: E402

ZENODO = "https://zenodo.org/api/records/4010759"
SPLIT_URL = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include50_{}.txt"


def get(url, **kw):
    for i in range(6):
        try:
            r = requests.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception as e:                       # Zenodo throws sporadic 5xx
            print(f"  retry {i + 1} ({e})")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"giving up on {url}")


def slug(label):
    """'23. Court' -> 'court' (same convention as the official INCLUDE code)."""
    return "".join(c for c in label if c.isalpha()).lower()


def load_splits():
    """-> {(class_folder, filename): split}"""
    out = {}
    for split in ("train", "val", "test"):
        for line in get(SPLIT_URL.format(split)).text.splitlines():
            parts = line.strip().split("/")
            if len(parts) >= 3:
                out[(parts[-2], parts[-1])] = split
    return out


def list_parts():
    files = get(ZENODO).json()["files"]
    return sorted((f["key"], f["size"]) for f in files if f["key"].endswith(".zip"))


def out_path(out, split, member):
    cls, fn = Path(member).parts[-2], Path(member).parts[-1]
    return Path(out) / split / f"{slug(cls)}__{Path(fn).stem}.npz"


def process_part(part, out, splits, stride, tmp_root):
    from remotezip import RemoteZip
    url = f"{ZENODO}/files/{part}/content"
    done = skipped = 0
    for attempt in range(4):
        try:
            zf = RemoteZip(url)
            break
        except Exception as e:
            print(f"[{part}] open failed ({e}); retry")
            time.sleep(5 * (attempt + 1))
    else:
        return part, 0, -1
    with zf:
        todo = []
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) >= 2 and (parts[-2], parts[-1]) in splits:
                todo.append((name, splits[(parts[-2], parts[-1])]))
        for name, split in todo:
            dst = out_path(out, split, name)
            if dst.exists():
                skipped += 1
                continue
            with tempfile.TemporaryDirectory(dir=tmp_root) as td:
                try:
                    vid = zf.extract(name, td)
                    kp = video_to_keypoints(vid, stride=stride)
                except Exception as e:
                    print(f"[{part}] {name}: {e}")
                    continue
                # tempdir (and the video in it) is deleted right here
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, kp=kp)
            done += 1
    return part, done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="*", help="zip names (default: every part)")
    ap.add_argument("--workers", type=int, default=8, help="downloads dominate (Zenodo is slow per connection), so use more workers than cores")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame (2 halves CPU time)")
    ap.add_argument("--tmp", default=None, help="scratch dir for the one-video-at-a-time download")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    parts = list_parts()
    if a.list:
        for k, s in parts:
            print(f"{k:40s} {s / 1e9:5.2f} GB")
        print(f"{len(parts)} parts, {sum(s for _, s in parts) / 1e9:.1f} GB total (we won't download all of it)")
        return
    splits = load_splits()
    print(f"INCLUDE-50 split: {len(splits)} videos, {len({slug(k[0]) for k in splits})} classes")
    names = a.parts or [k for k, _ in parts]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp_root = a.tmp or tempfile.gettempdir()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_part, p, a.out, splits, a.stride, tmp_root) for p in names]
        for f in as_completed(futs):
            part, done, skipped = f.result()
            print(f"[{time.time() - t0:6.0f}s] {part}: {done} new, {skipped} already done" if skipped >= 0
                  else f"[{part}] FAILED - rerun to retry")
    n = {s: len(list((Path(a.out) / s).glob('*.npz'))) for s in ("train", "val", "test")}
    print("keypoint files:", n, "of expected", {s: sum(v == s for v in splits.values()) for s in n})
    labels = sorted({p.name.split("__")[0] for p in Path(a.out).glob("*/*.npz")})
    Path(a.out, "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"{len(labels)} classes found -> {Path(a.out, 'labels.json')}")


if __name__ == "__main__":
    main()
