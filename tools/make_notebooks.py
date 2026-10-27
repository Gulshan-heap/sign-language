"""Generate the notebooks and the code bundle.

  python tools/make_notebooks.py
-> notebooks/day1_*.ipynb, day2a_asl_words.ipynb, day2b_isl_words.ipynb, dist/slr_code.zip

The notebooks are plain Python: they import the functions from scripts/ and slr/ instead of running shell commands.
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
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "slr").exists() else Path.cwd().parent   # the project folder
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)      # so worker processes can import our code too

MODELS = ROOT / "models"                  # the models we already trained (only read here)
WORK = ROOT / "work"                      # everything this notebook creates goes here, so MODELS is never overwritten
print("project folder:", ROOT)
''')


def architecture_cells(name):
    return [
        ("markdown", "## 0b. The transformer architecture\n"
                     "The class is `WordTransformer` in `slr/word_model.py`. Each of the 32 frames of a sign is one token: "
                     "`Linear` -> + positional embedding -> 3 x `TransformerEncoderLayer` (4 attention heads) -> average over frames -> `Linear` -> word."),
        ("code", f'''
from slr.word_model import load_model

model, labels = load_model(MODELS / "{name}_words.pt")
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
'''),
        ("code", '''
import torch

x = torch.zeros(1, 32, 270)               # one sign: 32 frames x 270 features per frame
h = model.inp(x) + model.pos              # every frame embedded to 128 numbers + its position
print("after embedding      :", tuple(h.shape))
h = model.enc(h)                          # 3 self-attention layers: frames look at each other
print("after transformer    :", tuple(h.shape))
print("final scores (words) :", tuple(model(x).shape))
'''),
    ]


DAY1 = nb([
    ("markdown", "# Day 1 - letters + INCLUDE keypoints (CPU only)\n"
                 "Needs the ASL Alphabet images (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and, optionally, an ISL "
                 "alphabet image dataset. Download them once and point the paths below at them. The training cells are skipped "
                 "when a folder is missing."),
    SETUP,
    ("markdown", "## 0. The letter model we already trained\n"
                 "It is a small neural network (MLP) on the 21 hand landmarks of one image."),
    ("code", '''
from slr.inference import NumpyMLP

asl_letters = NumpyMLP(MODELS / "asl_letters_np.npz")
print(len(asl_letters.classes), "letters:", asl_letters.classes)
'''),
    ("markdown", "## 1. Letters: ~300 images per letter -> hand landmarks -> classifier (minutes on CPU)\n"
                 "Set the two paths to the folders that contain one sub-folder per letter. Leave `ISL_IMAGES = None` if you have no "
                 "ISL letter images."),
    ("code", '''
from scripts.letters import extract_letters, train_letters

ASL_IMAGES = ROOT / "data" / "asl-alphabet"          # <- edit: folder with one sub-folder per letter
ISL_IMAGES = None                                    # e.g. ROOT / "data" / "isl-alphabet"

for lang, images in [("asl", ASL_IMAGES), ("isl", ISL_IMAGES)]:
    if images is None or not Path(images).exists():
        print("skipping", lang, "- image folder not found")
        continue
    extract_letters(images, lang, WORK / "letters", per_class=300, workers=4)
    train_letters(lang, WORK / "letters", WORK / "models")
'''),
    ("markdown", "## 2. INCLUDE-50 keypoints (reads only the 50 signs' videos from each Zenodo part, deletes each video right after)\n"
                 "Needs Internet and `pip install remotezip`. Run the first cell to check that Zenodo answers; the second one is the long one."),
    ("code", '''
from scripts.include_extract import show_parts, extract_include

show_parts()
'''),
    ("code", '''
extract_include(WORK / "include50", workers=8, stride=2)
print({split: len(list((WORK / "include50" / split).glob("*.npz"))) for split in ("train", "val", "test")})
'''),
])

DAY2_ASL = nb([
    ("markdown", "# Day 2 (teammate A) - ASL words. CPU is enough; no Day-1 data needed.\n"
                 "Needs the competition data **Google - Isolated Sign Language Recognition** "
                 "(https://www.kaggle.com/competitions/asl-signs). Download it once and point `ASL_DATA` at the folder that contains "
                 "`train.csv` and `train_landmark_files`. The training cells are skipped when the folder is missing."),
    SETUP,
    ("markdown", "## 0. The word model we already trained\n"
                 "It is a transformer encoder over a 32-frame keypoint sequence (see `slr/word_model.py`). Its words and accuracy:"),
    ("code", '''
info = json.loads((MODELS / "asl_words.json").read_text())
print(len(info["labels"]), "words:", info["labels"])
print("accuracy:", info["metrics"])
'''),
    *architecture_cells("asl"),
    ("markdown", "## 1. Pick the data and the signs"),
    ("code", '''
ASL_DATA = ROOT / "data" / "asl-signs"               # <- edit: folder with train.csv and train_landmark_files/
SIGNS = "hello thankyou please yes no mom dad water eat drink go sleep happy sad like want home cat dog bird red blue green hungry sick".split()
print("data found:", ASL_DATA.exists())
'''),
    ("markdown", "## 2. Convert only these signs to keypoints and train (about 20-40 minutes on CPU)"),
    ("code", '''
from scripts.asl_words import extract_words
from scripts.train_words import train_words

if ASL_DATA.exists():
    extract_words(ASL_DATA, SIGNS, WORK / "asl_words", per_sign=300)
    train_words(WORK / "asl_words", WORK / "models" / "asl_words.pt", epochs=60)
else:
    print("skipping - ASL_DATA folder not found")
'''),
    ("markdown", "## 3. (Optional) pretrained model benchmark - skip unless the team lead asks\n"
                 "Needs a pretrained TFLite ASL model in a folder and `pip install tensorflow`; set `RUN_PRETRAINED = True` and `MODEL_DIR`."),
    ("code", '''
from scripts.asl_words import benchmark_pretrained

RUN_PRETRAINED = False
MODEL_DIR = ROOT / "data" / "sign-language"
if RUN_PRETRAINED:
    benchmark_pretrained(MODEL_DIR, ASL_DATA, SIGNS, WORK / "models" / "asl_pretrained_eval.json", per_sign=20)
'''),
    ("markdown", "## 4. Package the trained model into one zip"),
    ("code", '''
if (WORK / "models").exists():
    print(shutil.make_archive(str(WORK / "asl_models"), "zip", WORK, "models"))
'''),
])

DAY2_ISL = nb([
    ("markdown", "# Day 2 (teammate B) - ISL words (needs the Day-1 INCLUDE-50 keypoints)\n"
                 "Needs the `include50/` folder made by Day 1 (default: `work/include50`). CPU is fine (5-15 minutes); a GPU is "
                 "used automatically if there is one. The training cell is skipped when the folder is missing."),
    SETUP,
    ("markdown", "## 0. The word model we already trained\n"
                 "It is a transformer encoder over a 32-frame keypoint sequence (see `slr/word_model.py`). Its words and accuracy:"),
    ("code", '''
info = json.loads((MODELS / "isl_words.json").read_text())
print(len(info["labels"]), "words:", info["labels"])
print("accuracy:", info["metrics"])
'''),
    *architecture_cells("isl"),
    ("markdown", "## 1. Find the Day-1 keypoints"),
    ("code", '''
INCLUDE_KP = WORK / "include50"                      # <- edit if the Day-1 output is somewhere else
for split in ("train", "val", "test"):
    print(split, len(list((INCLUDE_KP / split).glob("*.npz"))), "files")
'''),
    ("markdown", "## 2. Train (about 5-15 minutes)"),
    ("code", '''
from scripts.train_words import train_words

if (INCLUDE_KP / "train").exists():
    train_words(INCLUDE_KP, WORK / "models" / "isl_words.pt", epochs=150, seeds=3)
else:
    print("skipping - run Day 1 first (no keypoints in", INCLUDE_KP, ")")
'''),
    ("markdown", "## 3. Package the trained model into one zip"),
    ("code", '''
if (WORK / "models").exists():
    print(shutil.make_archive(str(WORK / "isl_models"), "zip", WORK, "models"))
'''),
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
