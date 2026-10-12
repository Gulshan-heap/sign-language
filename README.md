# Sign Language Recognition (ASL + ISL)

Real-time recognition of **American (ASL)** and **Indian (ISL)** sign language in the browser. Sign in front of a webcam
or upload a recorded video and the app decodes the signs into words (or, for ASL fingerspelling, letters) and builds a
sentence you can copy or have read aloud.

Everything runs locally on CPU. Hand and body landmarks come from [MediaPipe](https://ai.google.dev/edge/mediapipe);
the classifiers are small (a ~1.7 MB transformer for words, a tiny MLP for letters) and need no GPU at inference time.

| | Words | Letters |
|---|---|---|
| **ASL** | 23 signs | A-Z (+ `space`, `del`) |
| **ISL** | 51 signs | not included |

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Using the dashboard](#using-the-dashboard)
- [Models and accuracy](#models-and-accuracy)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Improving the models](#improving-the-models)
- [Training from scratch](#training-from-scratch)
- [Deployment (Render)](#deployment-render)
- [HTTP API](#http-api)
- [Other tools](#other-tools)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Features

- **Live decoding from the webcam**: signs are detected automatically (a sign starts when your hands appear and ends
  when they leave the picture) or marked manually with *Hold to sign*.
- **Video upload**: decode a recorded clip, either as one sign or split into several words. Works without a camera.
- **Fingerspelling** (ASL): hold a letter steady to type it.
- **Sentence builder**: confidence threshold, top-3 suggestions for uncertain signs, undo / clear / copy, text-to-speech.
- **Landmark overlay and framing check**: see what the tracker sees, and get a warning when your shoulders are out of view.
- **Collect your own training data**: one click saves a decoded clip, labelled with the word you really signed.
- **Lightweight**: inference needs only numpy and MediaPipe (no PyTorch), so it fits on a small hosted instance.
- **Privacy**: camera frames are processed in memory and not stored; uploaded videos are deleted after processing.

## Quick start

Requirements: Python 3.10-3.12 (3.12 is what the project is tested with) and a webcam for live mode.

```bash
git clone <this repository> && cd sign_language
python -m venv .venv
.venv\Scripts\activate            # Windows      (macOS / Linux: source .venv/bin/activate)
pip install -r requirements-local.txt
python web/live.py --models models
```

Open <http://localhost:5001> and press **Start camera**. Browsers only allow camera access on `localhost` or HTTPS.

The trained models ship in `models/`. On first start MediaPipe's hand and pose models (~14 MB) are downloaded once to
`~/.cache/slr`; copy them to another machine to run fully offline.

Options: `--models <dir>`, `--port <n>` (default 5001), `--host 0.0.0.0` (serve other devices; the camera then needs HTTPS).

> `requirements-local.txt` includes PyTorch, which is only needed to **train** or to **export** models. The dashboard itself
> does not import it when the `*_np.npz` model files are present.

## Using the dashboard

0. Press **Start camera** to begin and **Stop camera** to end the feed; stopping releases the camera completely.
1. Choose **Words** or **Letters** and **ISL** or **ASL** at the top. The panel at the bottom lists every sign the
   selected model knows; anything else is mapped to the closest known sign.
2. **Words, Auto mode**: sign one word, then lower your hands out of view. It is decoded about 0.7 s after your hands leave.
   Results above the *Min. confidence* slider are added to the sentence. For uncertain results, tap the correct word among
   the top-3.
3. **Words, Hold to sign**: hold the button (or the `Space` key) for the duration of the sign.
4. **Upload a video**: use the button or drag a file onto the camera window. Pick *One sign in the video* or *Several signs
   (split into words)*. Videos are sampled at ~15 fps; the first 40 s and up to 100 MB are used.
5. **Letters (ASL)**: hold each letter steady for about a second. The model's `space` and `del` signs act as keys.

**For best results**: good, even lighting; plain background; sit back so your head, both shoulders and both hands are in
view (the word model measures hand positions relative to your shoulders); sign at a natural pace and hold the end of the sign.

## Models and accuracy

| Model | Classes | Training data | Validation top-1 | Test top-1 | Test top-3 |
|---|---|---|---|---|---|
| ASL words | 23 | Google ISLR (`asl-signs`) | 92.2 % (n=990) | 86.3 % (n=629) | 95.4 % |
| ISL words | 51 | INCLUDE-50 (AI4Bharat) | 98.5 % (n=65) | 94.6 % (n=167) | 99.4 % |
| ASL letters | 28 | ASL Alphabet images | - | - | - |

These are offline numbers on held-out clips from the training datasets, which were filmed under controlled conditions.
**Expect lower accuracy on your own webcam**, especially for ISL, which was trained on only ~14 clips per sign. See
[Improving the models](#improving-the-models), and measure your own accuracy with the [web evaluator](#other-tools).

ASL words: bird, blue, cat, dad, dog, drink, go, green, happy, hello, home, hungry, like, mom, no, please, red, sad, sick,
sleep, thankyou, water, yes.

ISL words: see [`models/isl_words.json`](models/isl_words.json) (51 signs, e.g. hello, thankyou, good, goodmorning, house,
car, teacher, white).

## How it works

```
 webcam frame / video ─► MediaPipe hands + pose ─► 67-point keypoint frame (x, y)
                                                          │
            words: segmenter cuts the stream into signs ◄─┘
                   resample to 32 frames ─► normalise to shoulders ─► Transformer ─► top-3 words
            letters: one hand ─► wrist-centred features ─► MLP ─► letter, typed once held steady
```

- **Keypoints** (`slr/landmarks.py`): pose points 0-24, left hand (21) and right hand (21), 67 points per frame.
  Missing parts are `NaN`.
- **Segmentation** (`slr/live.py`): `WordSegmenter` turns a frame stream into signs; `LetterStabiliser` types a letter
  once it has been held for several frames.
- **Word model** (`slr/word_model.py`, `slr/numpy_word_model.py`): a 3-layer transformer trained with PyTorch and run at
  inference in plain numpy (identical output to ~1e-7, ~175 MB less memory).
- **Letter model** (`slr/inference.py`): an MLP exported to numpy, so letters also run without scikit-learn or PyTorch.
- **Server** (`web/live.py`): Flask. The browser sends ~13 frames/s; the server tracks, segments and decodes.

## Project structure

```
app.py                  entry point for `gunicorn app:app` (serves web/live.py)
gunicorn.conf.py        production server settings (1 worker, 4 threads, $PORT)
requirements.txt        hosted deployment (no PyTorch)
requirements-local.txt  laptop: dashboard + training / export tools
slr/                    shared library
  landmarks.py            MediaPipe hand / pose tracking, 67-point layout
  features.py             keypoint preprocessing and augmentation (numpy only)
  word_model.py           PyTorch word transformer + training loop
  numpy_word_model.py     numpy inference for the word transformer
  live.py                 sign segmentation and letter stabilising
  inference.py            loads models and predicts
web/
  live.py, live.html      live dashboard (webcam, upload, sentence builder)
  app.py                  evaluator: record clips, log accuracy per word
demo/app.py             minimal OpenCV desktop demo
scripts/                data extraction and training (run on Kaggle or locally)
tools/                  record_clips, evaluate_clips, export_words_np, make_notebooks
notebooks/              generated Kaggle notebooks
models/                 trained models (*.pt = PyTorch, *_np.npz = numpy)
TEAM_TRAINING_GUIDE.md  step-by-step Kaggle instructions for non-programmers
```

## Improving the models

The most effective improvement is **your own data**, recorded the way you will use the app.

1. Run the dashboard locally and sign a word.
2. In the *Wrong? Save this clip as* row, choose the word you really signed and press **Save for training**.
   Aim for 5-10 clips per word you care about.
3. Clips are written to `own_clips/<lang>/train/<word>__own_<time>.npz`, the same format as the INCLUDE keypoints.
4. Copy that `train/` folder into the training data (`include50/train/` on Kaggle) and retrain
   ([Training from scratch](#training-from-scratch)).
5. Convert the new weights for the server and replace the files in `models/`:
   ```bash
   python tools/export_words_np.py models      # writes <lang>_words_np.npz and checks them against PyTorch
   ```

Saving clips needs the local server; a hosted instance loses files when it restarts.

## Training from scratch

Training is done on Kaggle or Colab; only the small keypoint files are involved.

```bash
python tools/make_notebooks.py        # writes dist/slr_code.zip and notebooks/*.ipynb
```

Upload `dist/slr_code.zip` as a Kaggle dataset named **`slr-code`**, import the notebooks, add it as an input and turn
Internet on. Detailed, click-by-click instructions are in [TEAM_TRAINING_GUIDE.md](TEAM_TRAINING_GUIDE.md).

| Step | Notebook / script | Output |
|---|---|---|
| Letters + ISL keypoints (CPU) | `day1_setup_letters_include.ipynb` → `scripts/letters.py`, `scripts/include_extract.py` | `letters/`, `include50/` keypoints |
| ASL words (CPU) | `day2a_asl_words.ipynb` → `scripts/asl_words.py` | `asl_words.pt` |
| ISL words (GPU, minutes) | `day2b_isl_words.ipynb` → `scripts/train_words.py` | `isl_words.pt` |

- `include_extract.py` downloads only the ~955 INCLUDE-50 videos (~12 GB, never the full 56 GB archive), extracts keypoints
  and deletes each video. It is resumable.
- `train_words.py --data <dir> --out <model.pt>` expects `<dir>/{train,val,test}/<label>__<id>.npz`.
- Download the notebook outputs into `models/`, then run `tools/export_words_np.py models`.

## Deployment (Render)

The repository is ready for a Render *Web Service* (Python 3.12 is pinned in `.python-version`).

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt && pip uninstall -y opencv-contrib-python opencv-python && pip install --force-reinstall --no-deps opencv-python-headless` |
| Start command | `gunicorn app:app` (settings come from `gunicorn.conf.py`) |

Notes:

- **Memory**: about 300-360 MB with the numpy models (measured locally). The free 512 MB tier is tight; use a larger instance if you see
  out-of-memory restarts.
- **One user at a time**: the server keeps a single shared detection session, so concurrent visitors would interfere.
- **HTTPS** is provided by Render, so the camera works. The free tier sleeps when idle (slow first load).
- Every frame travels to the server, so live decoding is slower than on a laptop. Running locally is more reliable for demos.
- Environment variables: `PORT` (listen port), `SLR_MODELS` (models directory), `SLR_MODEL_DIR` (MediaPipe `.task` files).
- `models/*_np.npz` must be committed (the repository's `.gitignore` allows them).

## HTTP API

Served by `web/live.py`. All POST bodies are `multipart/form-data`.

| Endpoint | Purpose |
|---|---|
| `GET /` | the dashboard |
| `GET /api/info` | available words and letters per language |
| `POST /api/frame` | one camera frame (`image`, `kind`, `lang`, `thr`, optional `cmd` = `start` / `stop` / `reset`) → landmarks, state and any decoded sign |
| `POST /api/video` | decode an uploaded recording (`video`, `kind`, `lang`, `mode` = `word` / `sentence`, `thr`) |
| `POST /api/save_clip` | save a decoded clip as training data (`id`, `lang`, `word`) |

## Other tools

| Command | Purpose |
|---|---|
| `python web/app.py --models models` | Evaluator at <http://localhost:5000>: choose the true word, record, and get per-word accuracy and confusions in `results/web_log.csv` |
| `python demo/app.py --models models` | Minimal OpenCV window demo (letters and words, keyboard controlled) |
| `python tools/record_clips.py --lang isl --words hello water --per-word 5` | Record test clips with the webcam into `clips/` |
| `python tools/evaluate_clips.py words --lang isl --clips clips` | Batch evaluation: top-1 / top-3, per-word accuracy, confusion CSV in `results/` |
| `python tools/evaluate_clips.py letters --lang asl --images images` | Same for letter images |
| `python tools/export_words_np.py models` | Convert `.pt` word models to numpy (needed after every retraining) |

## Troubleshooting

| Problem | Fix |
|---|---|
| "Camera blocked" | Allow camera access for the site; use `http://localhost` or HTTPS, not a LAN IP over plain HTTP |
| Nothing is decoded | Make sure the hands are visible and well lit; check the status badge and the landmark overlay |
| Warning about shoulders | Sit further back so both shoulders are in the picture |
| "No ISL letters model" | No ISL letter model is shipped; use ASL letters or train one with `scripts/letters.py` |
| Slow / low frame rate | Close other apps that use the camera; run locally instead of hosted |
| `No module named ...` | Activate the virtual environment and run `pip install -r requirements-local.txt` |
| First start is slow | MediaPipe model files are being downloaded (needs internet once) |
| Out-of-memory restarts when hosted | Use an instance with more RAM; make sure `models/*_words_np.npz` are present so PyTorch is not loaded |

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe) hand and pose landmarkers (Google).
- [INCLUDE](https://github.com/AI4Bharat/INCLUDE) Indian Sign Language dataset (AI4Bharat); the INCLUDE-50 subset is used here.
- [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) (`asl-signs`) data for ASL words.
- ASL Alphabet and ISL fingerspelling image datasets for the letter models.


