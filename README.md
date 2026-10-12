# ASL + ISL sign recognition (letters and words)

Letters: MediaPipe hand landmarks -> small scikit-learn classifier (ASL one hand, ISL two hands).
Words: MediaPipe hand + pose keypoint sequence (Tasks API) (67 points x 32 frames) -> small Transformer (~1 MB).
Training runs on Kaggle/Colab, the demo runs on your laptop.

```
slr/            shared library (landmarks, word model, inference)
scripts/        letters.py  include_extract.py  asl_words.py  train_words.py   (run on Kaggle or locally)
notebooks/      day1_setup_letters_include.ipynb  day2a_asl_words.ipynb  day2b_isl_words.ipynb  (generated)
demo/app.py     OpenCV webcam demo
web/app.py      browser interface: record with the webcam, predict, log, accuracy panel
tools/          record_clips.py  evaluate_clips.py  make_notebooks.py
models/         put the downloaded Kaggle output here
```

## One-time: ship the code to Kaggle
```
python tools/make_notebooks.py          # writes dist/slr_code.zip + notebooks/
```
Kaggle -> Datasets -> New Dataset -> upload `dist/slr_code.zip`, name it **slr-code**. Import both notebooks
(File -> Import notebook), and add `slr-code` as an input to each. Internet must be ON (Zenodo, GitHub, pip).

## Day 1 (CPU) - `day1_setup_letters_include.ipynb`
1. **Smoke test, max 1 h.** `asl_words.py inspect` shows what is inside the pretrained dataset; `infer` runs one clip.
   If TensorFlow clashes, attach `abhinand5/isolated-sign-language-recognition` instead and point `MODEL_DIR` at it.
   Needs a `.tflite` + `sign_to_prediction_index_map.json` (Google ISLR format). If the weights are another format,
   use the fallback in Day 2 (train on the same data, no pretrained weights needed).
2. **Letters.** `letters.py extract` samples 300 images/letter, keeps only detected hands, adds a mirrored copy of each
   (so left-handed signers and a mirrored webcam still work), `train` compares SVC / MLP / ExtraTrees and saves the best.
3. **INCLUDE-50.** `include_extract.py` reads each Zenodo zip's table of contents over HTTP and downloads **only the
   ~955 INCLUDE-50 videos** (~12 GB in total, never the 56 GB), extracts keypoints, deletes the video. Resumable.
4. Save Version, then Output -> New Dataset (`models/`, `letters/`, `include50/`).

## Day 2 - `day2a_asl_words.ipynb` (CPU) and `day2b_isl_words.ipynb` (GPU) - one per teammate, see TEAM_TRAINING_GUIDE.md
- **ASL**: `asl_words.py infer` on your 20-30 signs from the `asl-signs` competition (only those parquet files are read).
  Optional: `extract` + `train_words.py` to train just on those signs.
- **ISL**: `train_words.py` on the INCLUDE-50 keypoints (T4, a few minutes).
- Record your own clips locally: `python tools/record_clips.py --lang isl --words hello water ... --per-word 5`

## Day 3 (laptop)
```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements-local.txt
# download models.zip from the Kaggle output into ./models  (asl_letters.joblib isl_letters.joblib asl_words.pt isl_words.pt)
python demo/app.py --models models
python tools/evaluate_clips.py words --lang isl --clips clips
python tools/evaluate_clips.py letters --lang asl --images images
```
`evaluate_clips.py` writes top-1/top-3, per-word accuracy and a confusion CSV to `results/` for the report.

## Things I could not verify from here - check these first
- The contents/format of `209sontung/sign-language` (I could not find its page). `inspect` tells you in seconds.
- The slug/layout of your ISL fingerspelling image dataset (set `ISL_IMAGES` in the Day-1 notebook; the class-folder level is auto-detected).
- Word-level ISL uses my own trainer rather than AI4Bharat's `runner.py`: their `generate_keypoints.py` targets an old
  MediaPipe API and a full local copy of the dataset. The keypoints here are a different format, so `runner.py` cannot read them.
- INCLUDE-50 gets ~94% in the paper with their pipeline; expect lower with this small model - report what you measure.

## Live dashboard (browser) - sign in front of the camera, get the word
```
python web/live.py --models models       # open http://localhost:5001 and press "Start camera"
```
- **Words** (ISL / ASL): in *Auto* mode a sign starts when your hands appear and is decoded when they leave the picture
  (~0.7 s). *Hold to sign* (button or Space) lets you mark the start/end yourself. Words above the confidence slider are
  added to the sentence; unsure ones show their top-3 so you can tap the right one. Speak / copy / undo are built in.
- **Letters** (ASL): fingerspell; hold each letter ~1 s to type it. The `space` and `del` signs act as keys.
- Uses `slr/live.py` for the segmentation; needs `models/{asl,isl}_words.pt` and `asl_letters_np.npz` (unzip the Kaggle
  output, e.g. `models/asl_models.zip`, into the project root). A mode whose model is missing says so instead of decoding.
- Low-memory / hosted use: `python tools/export_words_np.py models` writes `<lang>_words_np.npz` (run once, needs torch).
  The server prefers these and then does not import torch (~175 MB less RAM); `requirements.txt` is for Render.
  Re-run the export after retraining a word model.
- The page shows what each model can recognise - signs outside that list will be mapped to the nearest known word.

## Web evaluator (browser)
```
pip install -r requirements-local.txt
python web/app.py --models models        # open http://localhost:5000
```
Choose ISL/ASL, pick the *true* word, press Record (3-2-1, then it films for N seconds). The server extracts keypoints,
shows top-3 predictions, and logs the attempt to `results/web_log.csv`; the Evaluation panel shows top-1/top-3, per-word
accuracy and common confusions. Clips are saved to `clips/<lang>/<word>/`. Letters mode predicts live from the camera.
Browsers only allow camera access on `http://localhost` (or https), so open the page on the machine that runs the server.
On a Raspberry Pi with a screen and camera, run it there and open `http://localhost:5000` in the Pi's browser.
Opening `http://<pi-address>:5000` from another computer will NOT get camera access without https.
