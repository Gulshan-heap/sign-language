"""Generate the Kaggle/Colab notebooks and the code bundle to upload.

  python tools/make_notebooks.py
-> notebooks/day1_setup_letters_include.ipynb, notebooks/day2_words.ipynb, dist/slr_code.zip
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
# Code bundle: upload dist/slr_code.zip as a Kaggle Dataset named "slr-code", then add it as input.
import glob, os, shutil, subprocess, sys
src = (glob.glob("/kaggle/input/slr-code*") or glob.glob("/content/slr_code"))[0]
shutil.copytree(src, "/kaggle/working/code" if os.path.exists("/kaggle") else "/content/code", dirs_exist_ok=True)
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "/content"
CODE = f"{WORK}/code"
os.chdir(CODE)
# mediapipe 0.10.14 still has the `mp.solutions` API we use. If pip complains about numpy, run
# `!pip install "numpy<2"` and restart the session once.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.14", "remotezip", "opencv-python-headless"])
''')

DAY1 = nb([
    ("markdown", "# Day 1 - setup + letters + INCLUDE keypoints (CPU only)\n"
                 "Add inputs first (right sidebar): `slr-code`, ASL Alphabet (`grassknoted/asl-alphabet`), an ISL fingerspelling image "
                 "dataset, the pretrained-ASL dataset (`209sontung/sign-language`), and the `asl-signs` competition. Internet ON."),
    SETUP,
    ("markdown", "## 1. Smoke-test the pretrained ASL weights on ONE clip (<= 1 hour; else switch to `abhinand5/isolated-sign-language-recognition`)"),
    ("code", '''
MODEL_DIR = "/kaggle/input/sign-language"   # <- folder of the pretrained model dataset (check the sidebar)
!python scripts/asl_words.py inspect --model-dir $MODEL_DIR
!python scripts/asl_words.py infer --model-dir $MODEL_DIR --data /kaggle/input/asl-signs --signs hello --per-sign 1 --out $WORK/smoke.json
'''),
    ("markdown", "## 2. Letters: ~300 images per letter -> hand landmarks -> classifier (minutes on CPU)"),
    ("code", '''
ASL_IMAGES = "/kaggle/input/asl-alphabet"          # root of the ASL Alphabet dataset (class folder auto-detected)
ISL_IMAGES = "/kaggle/input/REPLACE-WITH-ISL-FINGERSPELLING-DATASET"
for lang, root in [("asl", ASL_IMAGES), ("isl", ISL_IMAGES)]:
    !python scripts/letters.py extract --images $root --lang $lang --out $WORK/letters --per-class 300 --workers 4
    !python scripts/letters.py train --lang $lang --data $WORK/letters --out $WORK/models
'''),
    ("markdown", "## 3. INCLUDE-50 keypoints (reads only the 50 signs' videos from each Zenodo part, deletes each video right after)\n"
                 "Run `--list` first to see the parts. Re-running resumes where it stopped, so a session timeout costs nothing."),
    ("code", '''
!python scripts/include_extract.py --out $WORK/include50 --list
'''),
    ("code", '''
!python scripts/include_extract.py --out $WORK/include50 --workers 8 --stride 2
!du -sh $WORK/include50
'''),
    ("markdown", "## 4. Save everything\n"
                 "Commit the notebook (Save Version -> Save & Run All), then Output -> **New Dataset** to keep `models/`, `letters/`, `include50/`."),
])

DAY2 = nb([
    ("markdown", "# Day 2 - words (GPU session only for the ISL training cell)\nInputs: `slr-code`, your Day-1 output dataset, `209sontung/sign-language`, `asl-signs`."),
    SETUP,
    ("markdown", "## ASL: pretrained inference on your chosen 20-30 signs (no training)"),
    ("code", '''
MODEL_DIR = "/kaggle/input/sign-language"
SIGNS = "hello thankyou please yes no mom dad water eat drink go sleep happy sad like want home cat dog bird red blue green hungry sick".split()
!python scripts/asl_words.py infer --model-dir $MODEL_DIR --data /kaggle/input/asl-signs --signs {" ".join(SIGNS)} --per-sign 20 --out $WORK/asl_pretrained_eval.json
'''),
    ("markdown", "### Optional: train/fine-tune only on those signs (also the fallback if the pretrained weights won't load)"),
    ("code", '''
!python scripts/asl_words.py extract --data /kaggle/input/asl-signs --signs {" ".join(SIGNS)} --out $WORK/asl_words
!python scripts/train_words.py --data $WORK/asl_words --out $WORK/models/asl_words.pt --epochs 120
'''),
    ("markdown", "## ISL: train on the INCLUDE-50 keypoints from Day 1 (enable GPU T4)"),
    ("code", '''
INCLUDE_KP = "/kaggle/input/REPLACE-WITH-DAY1-OUTPUT-DATASET/include50"   # or $WORK/include50 if same session
!python scripts/train_words.py --data $INCLUDE_KP --out $WORK/models/isl_words.pt --epochs 150 --seeds 3
'''),
    ("markdown", "## Package the small models for download (a few MB each)"),
    ("code", '''
!cp /kaggle/input/REPLACE-WITH-DAY1-OUTPUT-DATASET/models/*.joblib $WORK/models/ 2>/dev/null
!cd $WORK && zip -r models.zip models && ls -lh models.zip
'''),
])


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for name, book in (("day1_setup_letters_include", DAY1), ("day2_words", DAY2)):
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
