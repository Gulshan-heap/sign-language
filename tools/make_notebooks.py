"""Generate the Kaggle/Colab notebooks and the code bundle to upload.

  python tools/make_notebooks.py
-> notebooks/day1_*.ipynb, day2a_asl_words.ipynb, day2b_isl_words.ipynb, dist/slr_code.zip
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = ("code", '''
# Finds the uploaded "slr-code" dataset wherever Kaggle mounts it (the path differs between Kaggle versions).
import glob, os, shutil, subprocess, sys, zipfile
ON_KAGGLE = os.path.exists("/kaggle/input")
WORK = "/kaggle/working" if ON_KAGGLE else "/content"
CODE = f"{WORK}/code"
hits = [h for d in ("*", "*/*", "*/*/*") for h in glob.glob(f"/kaggle/input/{d}/slr/__init__.py")] + glob.glob("/content/slr_code/**/slr/__init__.py", recursive=True)  # shallow search: ASL Alphabet has ~87k files
if hits:
    shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
else:
    zips = [h for d in ("*", "*/*", "*/*/*") for h in glob.glob(f"/kaggle/input/{d}/slr_code.zip")] + glob.glob("/content/**/slr_code.zip", recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall(CODE)
    else:
        print("slr-code NOT FOUND. Add it: right sidebar -> + Add Input -> Your Datasets -> slr-code. Currently attached:")
        for p in sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")):
            print("  ", p)
        raise SystemExit("Attach the slr-code dataset, then run this cell again.")
os.chdir(CODE)
print("code ready in", CODE, os.listdir(CODE))
# Install each package on its own so one failure cannot skip the others. Kaggle now runs Python 3.13, so we use the
# current mediapipe (Tasks API) - not the old pinned version.
for pkg in ("mediapipe", "remotezip", "opencv-python-headless"):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    print(pkg, "OK" if r.returncode == 0 else "FAILED\\n" + r.stderr[-600:])
import mediapipe; print("mediapipe", mediapipe.__version__)
''')


DAY1 = nb([
    ("markdown", "# Day 1 - letters + INCLUDE keypoints (CPU only, Internet ON)\n"
                 "Inputs needed: `slr-code`, ASL Alphabet (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and, optionally, an ISL "
                 "alphabet image dataset. The pretrained-ASL smoke test and the `asl-signs` competition are NOT needed here: they belong to Day 2 (ASL)."),
    SETUP,
    ("markdown", "## 1. Letters: ~300 images per letter -> hand landmarks -> classifier (minutes on CPU)\n"
                 "Open the *Input* panel (right sidebar), hover the dataset, click the copy-path icon, paste below. "
                 "Set `ISL_IMAGES = None` if you have no ISL letter images yet - ASL still runs, ISL letters can be added later."),
    ("code", '''
import glob
_asl = [p for d in ("*", "*/*", "*/*/*") for p in glob.glob(f"/kaggle/input/{d}") if "asl-alphabet" in p.lower() and os.path.isdir(p)]
print("ASL alphabet candidates:", _asl)
ASL_IMAGES = min(_asl, key=len)                      # shortest = the dataset root; class folders are auto-detected
ISL_IMAGES = None                                    # e.g. "/kaggle/input/<your-isl-alphabet-dataset>"
for lang, root in [("asl", ASL_IMAGES), ("isl", ISL_IMAGES)]:
    if root is None:
        print("skipping", lang); continue
    !python scripts/letters.py extract --images $root --lang $lang --out $WORK/letters --per-class 300 --workers 4
    !python scripts/letters.py train --lang $lang --data $WORK/letters --out $WORK/models
'''),
    ("markdown", "## 2. INCLUDE-50 keypoints (reads only the 50 signs' videos from each Zenodo part, deletes each video right after)\n"
                 "Run `--list` first to check Internet/Zenodo. The second cell is the long one - use Save & Run All for it."),
    ("code", '''
!python scripts/include_extract.py --out $WORK/include50 --list
'''),
    ("code", '''
!python scripts/include_extract.py --out $WORK/include50 --workers 8 --stride 2
!du -sh $WORK/include50
'''),
    ("markdown", "## 3. Save everything\n"
                 "Save Version -> Save & Run All, then Output -> **New Dataset** to keep `models/`, `letters/`, `include50/`."),
])

DAY2_ASL = nb([
    ("markdown", "# Day 2 (teammate A) - ASL words. CPU is enough; no Day-1 data needed.\n"
                 "Inputs: `slr-code`, `209sontung/sign-language`, competition `asl-signs`. Internet ON."),
    SETUP,
    ("markdown", "## 1. Pretrained inference on the chosen signs (no training)\n"
                 "Set `MODEL_DIR` to the pretrained-model folder shown in the sidebar (hover -> copy path)."),
    ("code", '''
MODEL_DIR = "/kaggle/input/sign-language"
SIGNS = "hello thankyou please yes no mom dad water eat drink go sleep happy sad like want home cat dog bird red blue green hungry sick".split()
!python scripts/asl_words.py inspect --model-dir $MODEL_DIR
!python scripts/asl_words.py infer --model-dir $MODEL_DIR --data /kaggle/input/asl-signs --signs {" ".join(SIGNS)} --per-sign 20 --out $WORK/models/asl_pretrained_eval.json
'''),
    ("markdown", "## 2. Train only on those signs (run this if step 1 failed, or if the team lead asks)\n"
                 "Also fine to run even if step 1 worked: we then compare both."),
    ("code", '''
!python scripts/asl_words.py extract --data /kaggle/input/asl-signs --signs {" ".join(SIGNS)} --out $WORK/asl_words
!python scripts/train_words.py --data $WORK/asl_words --out $WORK/models/asl_words.pt --epochs 120
'''),
    ("markdown", "## 3. Package for download\nAlso copy the pretrained model files so the demo can use them."),
    ("code", '''
!cp $MODEL_DIR/*.tflite $MODEL_DIR/*.json $WORK/models/ 2>/dev/null
!cd $WORK && zip -r asl_models.zip models && ls -lh asl_models.zip
'''),
    ("markdown", "Then **Save Version -> Save & Run All**, open the Output tab and download `asl_models.zip`."),
])

DAY2_ISL = nb([
    ("markdown", "# Day 2 (teammate B) - ISL words (needs the Day-1 output dataset)\n"
                 "Inputs: `slr-code` and the Day-1 output dataset (`include50/` inside). Internet ON. "
                 "Set the accelerator to **GPU T4** before the training cell."),
    SETUP,
    ("markdown", "## 1. Point at the Day-1 keypoints\n"
                 "Edit `INCLUDE_KP`: the sidebar path of the Day-1 dataset + `/include50`. It should contain `train`, `val`, `test` folders."),
    ("code", '''
INCLUDE_KP = "/kaggle/input/REPLACE-WITH-DAY1-OUTPUT-DATASET/include50"
!ls $INCLUDE_KP && ls $INCLUDE_KP/train | wc -l
'''),
    ("markdown", "## 2. Train (GPU T4, about 5-15 minutes)"),
    ("code", '''
!python scripts/train_words.py --data $INCLUDE_KP --out $WORK/models/isl_words.pt --epochs 150 --seeds 3
'''),
    ("markdown", "## 3. Package for download (a few MB)"),
    ("code", '''
!cd $WORK && zip -r isl_models.zip models && ls -lh isl_models.zip
'''),
    ("markdown", "Then **Save Version -> Save & Run All**, open the Output tab and download `isl_models.zip`."),
])


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for name, book in (("day1_setup_letters_include", DAY1), ("day2a_asl_words", DAY2_ASL), ("day2b_isl_words", DAY2_ISL)):
        (ROOT / "notebooks" / f"{name}.ipynb").write_text(json.dumps(book, indent=1))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    with zipfile.ZipFile(dist / "slr_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("slr", "scripts"):
            for p in (ROOT / folder).rglob("*.py"):
                z.write(p, p.relative_to(ROOT))
    print("wrote notebooks/ and dist/slr_code.zip")


if __name__ == "__main__":
    main()
