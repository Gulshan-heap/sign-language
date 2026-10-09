# Training guide for teammates (no coding needed)

You will run **one ready-made Kaggle notebook** and hand back **one small zip file**. You do not need to understand or edit
the code; everything runs in the browser.

**What we are building:** a program that recognises sign-language letters and words (ASL = American, ISL = Indian) from a webcam.
Training is the heavy part, so it is done on Kaggle's free computers.

| Who | Notebook | Hand back |
|---|---|---|
| Team lead | `day1_setup_letters_include.ipynb` (letters + ISL keypoints) | Day-1 output dataset |
| **Teammate A - ASL words** | `day2a_asl_words.ipynb` | `asl_models.zip` + printed results |
| **Teammate B - ISL words** | `day2b_isl_words.ipynb` (needs the Day-1 dataset first) | `isl_models.zip` + printed results |

A and B work independently. B must wait until the team lead shares the Day-1 dataset.

**You will be given:** `slr_code.zip` and your notebook file.

---

## Step 0 - one-time setup (both teammates, ~10 minutes)

1. Make a free account at kaggle.com and **verify your phone number** (Settings -> Phone verification).
   Without this, Internet and GPU stay disabled and nothing works.
2. **Datasets -> New Dataset** -> drag in `slr_code.zip` -> name it exactly **`slr-code`** -> Create.
3. **Code -> New Notebook -> File -> Import Notebook** -> pick your notebook file.
4. Right sidebar -> **Session options**: **Internet = On**.
5. Add inputs with **+ Add Input** (search by the names in your section below). To get a path, hover the input in the sidebar and click the copy-path icon.

Run a cell: click it, press **Shift+Enter**. Run them top to bottom, wait for each to finish.

---

## Teammate A - ASL words (CPU, about 1 hour)

Accelerator: **None**.

**Inputs to add:** `slr-code`, `209sontung/sign-language` (pretrained model), and the competition
**Google - Isolated Sign Language Recognition** (`asl-signs`): open it, click *Join / accept rules*, then *Add Input -> Competition*.
Never download that competition's data (40 GB); only attach it.

**Edit one line:** `MODEL_DIR = "/kaggle/input/sign-language"` -> the real folder of the pretrained model from the sidebar.

**Run:**
1. Setup cell. If it errors about `numpy`: new cell `!pip install "numpy<2"`, then Run -> Restart session, run the setup cell again.
2. **Step 1 (pretrained inference).** Success looks like `N clips | top1 0.xx | top3 0.xx` plus one line per sign.
   If it fails with TensorFlow errors: do not spend more than **1 hour** on it. Copy the error text, then continue with step 2.
   (If asked, you can instead attach `abhinand5/isolated-sign-language-recognition` and point `MODEL_DIR` at it.)
3. **Step 2 (train on our signs).** Prints `epoch ... val_acc ...`, ends with a `test {...'top1'...}` line. Run it unless the lead says to skip.
4. **Step 3 (package).** Creates `asl_models.zip`.

**Save:** top right **Save Version -> Save & Run All (Commit)**, wait for it to finish, open the **Output** tab, download `asl_models.zip`.

**Hand back:** `asl_models.zip`, the printed text of steps 1 and 2 (or the error message), the notebook link.

---

## Teammate B - ISL words (GPU, about 30 minutes)

**Wait for the team lead's Day-1 dataset** (it contains a folder `include50/` with `train`, `val`, `test`).

**Inputs to add:** `slr-code` and the Day-1 dataset.

**Edit one line:** `INCLUDE_KP = ".../REPLACE-WITH-DAY1-OUTPUT-DATASET/include50"` -> the Day-1 dataset path from the sidebar, ending in `/include50`.
The cell right after it must list `test train val` and a number around 50 (class folders count is not shown; file counts are in the next step's output).

**Run:**
1. Setup cell (same numpy note as above).
2. Path-check cell. If it says "No such file", the path is wrong.
3. Session options -> Accelerator -> **GPU T4** (turn it back off when finished; the weekly quota is limited). Then run the **training** cell.
   It first prints `50 classes | train ~688 val ~76 test ~191`, then `epoch ... val_acc ...` lines (5-15 minutes), and ends with
   `val {...}` / `test {...}`. Good: test top1 above ~0.60; far below that (e.g. under 0.30), tell the lead before continuing.
4. **Package** cell. Creates `isl_models.zip` (a few MB).

**Save:** **Save Version -> Save & Run All (Commit)** (GPU selected), wait, open **Output**, download `isl_models.zip`.

**Hand back:** `isl_models.zip`, the printed `val` / `test` lines, the notebook link.

---

## What the team lead should receive

- [ ] `asl_models.zip` (contains `asl_words.pt`, `asl_words.json`, `asl_pretrained_eval.json`, and the pretrained `.tflite` + `sign_to_prediction_index_map.json` if they were found)
- [ ] `isl_models.zip` (contains `isl_words.pt`, `isl_words.json`)
- [ ] Printed results from each notebook (or the error text)

Do not rename files inside the zips.

## If something goes wrong

| Symptom | Fix |
|---|---|
| "Internet is disabled" / pip fails | Session options -> Internet On (needs phone verification) |
| `no .tflite model in --model-dir` | `MODEL_DIR` is wrong, or the weights are another format: skip to step 2 (train our own) |
| `not in the 250-sign vocabulary` warning | normal for a few words, ignore |
| `No such file or directory` on the data path | path copied wrongly - re-copy it from the sidebar |
| Session stopped mid-run | files from an interactive session are lost; re-run, or use **Save & Run All** (runs in the background, 12 h limit) |
| Anything else | copy the red error text and send it to the team lead |
